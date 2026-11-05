"""Confirmed-fill lifecycle for entry bracket orders.

A broker acknowledgement is an order state, not a position state.  This module
records submitted entry brackets and initializes trailing protection only after
a broker execution report confirms a non-zero cumulative fill.

Broker adapter contract:
    get_order_snapshot(order_id) -> mapping/object with an order quantity and
    a non-terminal status.  This module fails closed when protection cannot be
    verified for a confirmed fill.
"""
from __future__ import annotations

from datetime import datetime
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from strategies.base import TradeSignal
from strategies.webhook import send_discord_alert

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"canceled", "cancelled", "rejected", "expired", "done_for_day", "filled"}


class LineageOutcome(Enum):
    ACTIVE_REPLACEMENT_FOUND = "active_replacement_found"
    TERMINAL_REPLACEMENT_FOUND = "terminal_replacement_found"
    LINEAGE_CYCLE_DETECTED = "lineage_cycle_detected"
    LINEAGE_DEPTH_EXCEEDED = "lineage_depth_exceeded"
    ORDER_NOT_FOUND = "order_not_found"
    BROKER_ERROR = "broker_error"


class ProtectionState(Enum):
    PROTECTED_COMPLETE = "protected_complete"
    PROTECTED_STOP_ONLY = "protected_stop_only"
    PROTECTED_PENDING_PARENT_FILL = "protected_pending_parent_fill"
    CLOSED_BY_STOP = "closed_by_stop"
    CLOSED_BY_TARGET = "closed_by_target"
    CLOSED_BY_MIXED_EXITS = "closed_by_mixed_exits"
    CLOSED_BY_OTHER_CONFIRMED_EXIT = "closed_by_other_confirmed_exit"
    RECONCILIATION_REQUIRED = "reconciliation_required"
    PROTECTION_FAILURE = "protection_failure"


@dataclass(frozen=True)
class ProtectionEvaluation:
    state: ProtectionState
    reason: str
    broker_position_qty: float = 0.0
    entry_filled_qty: float = 0.0
    stop_filled_qty: float = 0.0
    target_filled_qty: float = 0.0
    other_exit_filled_qty: float = 0.0
    active_stop_remaining_qty: float = 0.0
    active_target_remaining_qty: float = 0.0
    resolved_stop_id: Optional[str] = None
    resolved_target_id: Optional[str] = None
    stop_lineage_outcome: Optional[LineageOutcome] = None
    target_lineage_outcome: Optional[LineageOutcome] = None


class EntryLifecycleState(Enum):
    WORKING = "working"
    TTL_EXPIRED = "ttl_expired"
    CANCEL_REQUESTED = "cancel_requested"
    RECONCILING = "reconciling"
    PARTIAL_FILL_PROTECTING = "partial_fill_protecting"
    FINALIZED = "finalized"
    REJECTED = "rejected"


@dataclass
class PendingEntry:
    parent_order_id: str
    symbol: str
    signal: TradeSignal
    requested_qty: float
    stop_order_id: Optional[str]
    target_order_id: Optional[str]
    submitted_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    decision_bar_timestamp: str = ""
    state: EntryLifecycleState = EntryLifecycleState.WORKING
    owner_id: Optional[str] = None
    cumulative_filled_qty: float = 0.0
    average_fill_price: float = 0.0
    execution_ids: Set[str] = field(default_factory=set)
    submitted_entry_price: Optional[float] = None
    submitted_stop_price: Optional[float] = None
    submitted_target_price: Optional[float] = None
    # Forensic Telemetry Timestamps (UTC epoch / ISO)
    signal_ts: Optional[float] = None
    submit_ts: Optional[float] = None
    broker_accepted_ts: Optional[float] = None
    first_fill_ts: Optional[float] = None
    last_fill_ts: Optional[float] = None
    ttl_expiry_ts: Optional[float] = None
    cancel_request_ts: Optional[float] = None
    cancel_ack_ts: Optional[float] = None
    final_order_status_ts: Optional[float] = None
    final_position_reconcile_ts: Optional[float] = None
    protection_verified_ts: Optional[float] = None


@dataclass(frozen=True)
class FillUpdate:
    entry: PendingEntry
    fill_delta: float
    is_complete: bool
    incremental_price: float = 0.0


class EntryOrderLifecycle:
    """Thread-safe pending-entry registry plus confirmed-fill handler."""

    def __init__(self, trailing_manager: object):
        self._trailing_manager = trailing_manager
        self._lock = threading.RLock()
        self._pending: Dict[str, PendingEntry] = {}

    @staticmethod
    def _order_id(value: object) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def compare_and_swap_state(
        self,
        parent_order_id: object = None,
        expected_states: Any = None,
        new_state: Optional[EntryLifecycleState] = None,
        worker_id: str = "worker",
        *,
        order_id: object = None,
        expected: Any = None,
    ) -> bool:
        """
        Atomic Compare-And-Swap transition enforcing Amendment 1 (Single-Owner Lifecycle).
        Ensures only one worker (background reaper vs sync sweep) claims and transitions an order.
        """
        pid = parent_order_id if parent_order_id is not None else order_id
        parent_id = self._order_id(pid)
        if not parent_id or new_state is None:
            return False
        exp_input = expected_states if expected_states is not None else expected
        if isinstance(exp_input, EntryLifecycleState):
            expected_set = {exp_input}
        elif exp_input is not None:
            expected_set = set(exp_input)
        else:
            expected_set = {EntryLifecycleState.WORKING}

        with self._lock:
            entry = self._pending.get(parent_id)
            if entry is None:
                return False
            if entry.state not in expected_set:
                logger.debug(
                    "[CAS REJECTED] %s: current state %s not in expected %s (worker: %s, owner: %s)",
                    parent_id, entry.state, expected_set, worker_id, entry.owner_id
                )
                return False
            old_state = entry.state
            entry.state = new_state
            entry.owner_id = worker_id
            logger.info(
                "[CAS TRANSITION] %s: %s -> %s claimed by %s",
                parent_id, old_state.value, new_state.value, worker_id
            )
            return True

    def update_state(
        self,
        parent_order_id: object = None,
        new_state: Optional[EntryLifecycleState] = None,
        *,
        order_id: object = None,
    ) -> bool:
        """Update lifecycle state of an active entry."""
        pid = parent_order_id if parent_order_id is not None else order_id
        parent_id = self._order_id(pid)
        if not parent_id or new_state is None:
            return False
        with self._lock:
            entry = self._pending.get(parent_id)
            if entry:
                entry.state = new_state
                return True
            return False

    def get_pending_entry(self, parent_order_id: object) -> Optional[PendingEntry]:
        """Fetch pending entry safely by parent order ID."""
        parent_id = self._order_id(parent_order_id)
        with self._lock:
            return self._pending.get(parent_id)

    def get_expired_entries(
        self, now: float, current_closed_bar_start: Optional[str] = None
    ) -> List[PendingEntry]:
        """
        Identify working entries that have exceeded TTL or whose decision bar has been superseded.
        Adheres strictly to Amendment 2 (current_closed_bar_start > decision_bar_timestamp).
        """
        expired = []
        with self._lock:
            for entry in list(self._pending.values()):
                if entry.state != EntryLifecycleState.WORKING:
                    continue

                is_ttl_expired = bool(entry.expires_at > 0 and now >= entry.expires_at)
                is_bar_superseded = False
                if current_closed_bar_start and entry.decision_bar_timestamp:
                    cur_bar = str(current_closed_bar_start).strip()
                    dec_bar = str(entry.decision_bar_timestamp).strip()
                    if cur_bar and dec_bar and cur_bar > dec_bar:
                        is_bar_superseded = True

                if is_ttl_expired or is_bar_superseded:
                    expired.append(entry)
        return expired

    def finalize_entry(
        self,
        parent_order_id: object = None,
        reason: str = "",
        *,
        order_id: object = None,
    ) -> Optional[PendingEntry]:
        """Mark entry finalized and remove from pending tracking."""
        pid = parent_order_id if parent_order_id is not None else order_id
        parent_id = self._order_id(pid)
        with self._lock:
            entry = self._pending.pop(parent_id, None)
            if entry:
                entry.state = EntryLifecycleState.FINALIZED
            return entry

    def register_submission(
        self,
        *,
        parent_order_id: object,
        signal: TradeSignal,
        requested_qty: float,
        stop_order_id: object = None,
        target_order_id: object = None,
        submitted_entry_price: Optional[float] = None,
        submitted_stop_price: Optional[float] = None,
        submitted_target_price: Optional[float] = None,
        expires_at: Optional[float] = None,
        decision_bar_timestamp: Optional[str] = None,
        signal_ts: Optional[float] = None,
        submit_ts: Optional[float] = None,
        **kwargs: Any,
    ) -> PendingEntry:
        """Record a submitted bracket with explicit TTL and bar boundary tracking."""
        parent_id = self._order_id(parent_order_id)
        if parent_id is None:
            raise ValueError("Cannot register entry lifecycle without parent order ID")
        if requested_qty <= 0:
            raise ValueError("requested_qty must be positive")

        stop_id = self._order_id(stop_order_id)
        target_id = self._order_id(target_order_id)
        if stop_id is None or target_id is None or stop_id == target_id:
            raise ValueError("Bracket must provide distinct stop and target child order IDs")

        # P1.5.2: Propagate submitted geometry to effective signal
        from dataclasses import replace
        entry_px = float(submitted_entry_price) if submitted_entry_price is not None else float(signal.entry_price)
        stop_px = float(submitted_stop_price) if submitted_stop_price is not None else float(signal.stop_price)
        target_px = float(submitted_target_price) if submitted_target_price is not None else getattr(signal, "target_price", None)

        effective_signal = replace(
            signal,
            entry_price=entry_px,
            stop_price=stop_px,
            target_price=target_px,
            risk_per_share=abs(entry_px - stop_px),
        )

        now_t = time.time()
        sub_t = submit_ts if submit_ts is not None else now_t
        exp_t = expires_at if (expires_at is not None and expires_at > 0) else (sub_t + 60.0)
        bar_ts = str(decision_bar_timestamp or getattr(signal, "bar_timestamp", "") or "").strip()

        sig_t = signal_ts or kwargs.get("signal_generation_timestamp") or sub_t
        if isinstance(sig_t, datetime):
            sig_t = sig_t.timestamp()
        elif isinstance(sig_t, str):
            try:
                sig_t = float(sig_t)
            except ValueError:
                try:
                    sig_t = datetime.fromisoformat(sig_t.replace("Z", "+00:00")).timestamp()
                except Exception:
                    sig_t = sub_t
        elif not isinstance(sig_t, (int, float)):
            sig_t = sub_t

        entry = PendingEntry(
            parent_order_id=parent_id,
            symbol=effective_signal.symbol.upper().strip(),
            signal=effective_signal,
            requested_qty=float(requested_qty),
            stop_order_id=stop_id,
            target_order_id=target_id,
            submitted_at=sub_t,
            expires_at=exp_t,
            decision_bar_timestamp=bar_ts,
            state=EntryLifecycleState.WORKING,
            submitted_entry_price=entry_px,
            submitted_stop_price=stop_px,
            submitted_target_price=target_px,
            signal_ts=float(sig_t),
            submit_ts=sub_t,
            broker_accepted_ts=now_t,
            ttl_expiry_ts=exp_t,
        )
        with self._lock:
            if parent_id in self._pending:
                raise RuntimeError(f"Duplicate pending parent order ID: {parent_id}")
            self._pending[parent_id] = entry
        logger.info(
            "[ENTRY LIFECYCLE] Registered pending %s parent=%s qty=%.4f stop=%s target=%s ttl=%.1fs bar=%s",
            entry.symbol,
            entry.parent_order_id,
            entry.requested_qty,
            entry.stop_order_id,
            entry.target_order_id,
            entry.expires_at - entry.submitted_at,
            entry.decision_bar_timestamp,
        )
        return entry


    def record_parent_fill(
        self,
        *,
        parent_order_id: object,
        cumulative_filled_qty: float,
        average_fill_price: float,
        execution_id: object = None,
    ) -> Optional[FillUpdate]:
        """Store a cumulative fill report and return only a new fill transition.

        The broker adapter must pass cumulative filled quantity, not a per-event
        quantity.  Execution IDs are deduplicated when the venue supplies them.
        """
        parent_id = self._order_id(parent_order_id)
        if parent_id is None or cumulative_filled_qty <= 0 or average_fill_price <= 0:
            logger.warning("[ENTRY LIFECYCLE] Ignoring invalid fill parent=%r qty=%r price=%r", parent_order_id, cumulative_filled_qty, average_fill_price)
            return None

        with self._lock:
            entry = self._pending.get(parent_id)
            if entry is None:
                logger.warning("[ENTRY LIFECYCLE] Fill for unknown parent order %s", parent_id)
                return None

            exec_id = self._order_id(execution_id)
            if exec_id and exec_id in entry.execution_ids:
                return None
            if exec_id:
                entry.execution_ids.add(exec_id)

            reported_qty = min(float(cumulative_filled_qty), entry.requested_qty)
            fill_delta = reported_qty - entry.cumulative_filled_qty
            if fill_delta <= 0:
                return None

            prev_qty = entry.cumulative_filled_qty
            prev_avg_px = entry.average_fill_price
            curr_avg_px = float(average_fill_price)

            if prev_qty > 0 and curr_avg_px > 0:
                prev_notional = prev_qty * prev_avg_px
                curr_notional = reported_qty * curr_avg_px
                delta_notional = curr_notional - prev_notional
                incremental_price = delta_notional / fill_delta if fill_delta > 0 else curr_avg_px
            else:
                incremental_price = curr_avg_px

            entry.cumulative_filled_qty = reported_qty
            entry.average_fill_price = float(average_fill_price)
            return FillUpdate(
                entry=entry,
                fill_delta=fill_delta,
                is_complete=reported_qty >= entry.requested_qty,
                incremental_price=incremental_price,
            )

    @staticmethod
    def _field(snapshot: object, *names: str, default: Any = None) -> Any:
        if isinstance(snapshot, dict):
            for name in names:
                if name in snapshot:
                    return snapshot[name]
            return default
        for name in names:
            if hasattr(snapshot, name):
                return getattr(snapshot, name)
        return default

    def _resolve_order_lineage(
        self, bridge: object, order_id: Optional[str], max_depth: int = 5
    ) -> Tuple[LineageOutcome, Optional[str], Optional[dict]]:
        """Traverse broker replacement chains to find the latest active or terminal child order snapshot.

        Returns: (outcome, resolved_order_id, snapshot)
        """
        if not order_id:
            return LineageOutcome.ORDER_NOT_FOUND, None, None
        getter = getattr(bridge, "get_order_snapshot", None)
        if not callable(getter):
            return LineageOutcome.BROKER_ERROR, str(order_id).strip(), None

        curr_id = str(order_id).strip()
        visited: Set[str] = set()
        last_snap = None

        for hop in range(max_depth):
            if curr_id in visited:
                logger.warning(
                    "[ENTRY LIFECYCLE] Lineage cycle detected at order_id=%s (visited=%s, hop=%d)",
                    curr_id, visited, hop
                )
                return LineageOutcome.LINEAGE_CYCLE_DETECTED, curr_id, last_snap
            visited.add(curr_id)

            try:
                snap = getter(curr_id)
                if snap is None:
                    return LineageOutcome.ORDER_NOT_FOUND, curr_id, last_snap
                last_snap = snap
                status = str(self._field(snap, "status", default="")).lower()
                if status == "replaced":
                    replaced_by = self._field(snap, "replaced_by")
                    if replaced_by:
                        curr_id = str(replaced_by).strip()
                        continue
                    return LineageOutcome.TERMINAL_REPLACEMENT_FOUND, curr_id, last_snap

                if status in _TERMINAL_STATUSES:
                    return LineageOutcome.TERMINAL_REPLACEMENT_FOUND, curr_id, last_snap
                else:
                    return LineageOutcome.ACTIVE_REPLACEMENT_FOUND, curr_id, last_snap
            except Exception as exc:
                logger.debug("[ENTRY LIFECYCLE] Error traversing lineage for %s: %s", curr_id, exc)
                return LineageOutcome.BROKER_ERROR, curr_id, last_snap

        logger.warning(
            "[ENTRY LIFECYCLE] Lineage depth exceeded (max_depth=%d) at order_id=%s",
            max_depth, curr_id
        )
        return LineageOutcome.LINEAGE_DEPTH_EXCEEDED, curr_id, last_snap

    def _protective_order_covers_fill(self, bridge: object, order_id: Optional[str], filled_qty: float, order_label: str = "child") -> bool:
        if not order_id:
            logger.warning("[ENTRY LIFECYCLE] %s order_id is missing/empty", order_label)
            return False
        getter = getattr(bridge, "get_order_snapshot", None)
        if not callable(getter):
            logger.error("[ENTRY LIFECYCLE] Bridge has no get_order_snapshot; cannot verify protection for %s (%s)", order_label, order_id)
            return False
        try:
            outcome, resolved_id, snapshot = self._resolve_order_lineage(bridge, order_id)
            if snapshot is None or outcome not in (LineageOutcome.ACTIVE_REPLACEMENT_FOUND, LineageOutcome.TERMINAL_REPLACEMENT_FOUND):
                logger.warning("[ENTRY LIFECYCLE] Order snapshot unavailable or lineage failed (%s) for %s (%s)", outcome.value if outcome else "None", order_label, order_id)
                return False
            status = str(self._field(snapshot, "status", default="")).lower()
            quantity = float(self._field(snapshot, "qty", "quantity", "total_quantity", default=0.0) or 0.0)
            if status in _TERMINAL_STATUSES:
                logger.warning("[ENTRY LIFECYCLE] %s order (%s) has terminal status: %s", order_label, resolved_id, status)
                return False
            if quantity < float(filled_qty):
                logger.warning("[ENTRY LIFECYCLE] %s order (%s) qty %.4f < filled_qty %.4f", order_label, resolved_id, quantity, float(filled_qty))
                return False
            return True
        except Exception:
            logger.exception("[ENTRY LIFECYCLE] Failed protection lookup for %s order %s", order_label, order_id)
            return False

    def evaluate_protection_state(self, bridge: object, entry: PendingEntry) -> ProtectionEvaluation:
        """Authoritative canonical evaluation of entry protection across broker position and child orders."""
        entry_filled = float(entry.cumulative_filled_qty)
        symbol = entry.symbol.upper().strip()

        # 1. Fetch Authoritative Broker Position
        broker_position_qty = None
        get_pos = getattr(bridge, "get_position", None)
        if callable(get_pos):
            try:
                pos = get_pos(symbol)
                if isinstance(pos, (int, float)):
                    broker_position_qty = abs(float(pos))
                elif pos is not None:
                    broker_position_qty = abs(float(self._field(pos, "qty", default=0.0) or 0.0))
            except Exception as exc:
                logger.debug("[ENTRY LIFECYCLE] Querying get_position(%s) failed: %s", symbol, exc)

        if broker_position_qty is None:
            get_open_pos = getattr(bridge, "get_open_positions", None)
            if callable(get_open_pos):
                try:
                    open_positions = get_open_pos()
                    broker_position_qty = 0.0
                    for p in (open_positions or []):
                        sym = str(self._field(p, "symbol", default="")).upper().strip()
                        if sym == symbol:
                            broker_position_qty = abs(float(self._field(p, "qty", default=0.0) or 0.0))
                            break
                except Exception as exc:
                    logger.debug("[ENTRY LIFECYCLE] Querying get_open_positions failed: %s", exc)

        # 2. Resolve Child Orders & Lineage
        stop_outcome, resolved_stop_id, stop_snap = self._resolve_order_lineage(bridge, entry.stop_order_id)
        target_outcome, resolved_target_id, target_snap = self._resolve_order_lineage(bridge, entry.target_order_id)

        resolved_stop_id = str(self._field(stop_snap, "id", default=resolved_stop_id or entry.stop_order_id)) if stop_snap else (resolved_stop_id or entry.stop_order_id)
        resolved_target_id = str(self._field(target_snap, "id", default=resolved_target_id or entry.target_order_id)) if target_snap else (resolved_target_id or entry.target_order_id)

        # 3. Extract Canonical Stop Quantities
        stop_status = str(self._field(stop_snap, "status", default="")).lower() if stop_snap else ""
        stop_qty = float(self._field(stop_snap, "qty", "quantity", "total_quantity", default=0.0) or 0.0)
        stop_filled = float(self._field(stop_snap, "filled_qty", default=0.0) or 0.0)
        if stop_status == "filled" and stop_filled == 0.0 and stop_qty > 0:
            stop_filled = stop_qty
        active_stop_qty = (stop_qty - stop_filled) if (stop_status and stop_status not in _TERMINAL_STATUSES) else 0.0

        # 4. Extract Canonical Target Quantities
        target_status = str(self._field(target_snap, "status", default="")).lower() if target_snap else ""
        target_qty = float(self._field(target_snap, "qty", "quantity", "total_quantity", default=0.0) or 0.0)
        target_filled = float(self._field(target_snap, "filled_qty", default=0.0) or 0.0)
        if target_status == "filled" and target_filled == 0.0 and target_qty > 0:
            target_filled = target_qty
        active_target_qty = (target_qty - target_filled) if (target_status and target_status not in _TERMINAL_STATUSES) else 0.0

        confirmed_exit_qty = stop_filled + target_filled
        ledger_remaining_qty = max(entry_filled - confirmed_exit_qty, 0.0)

        eff_pos_qty = broker_position_qty if broker_position_qty is not None else ledger_remaining_qty

        # 5. Authoritative Evaluation Invariants
        if eff_pos_qty > 0:
            # Check for lineage failures while position is open
            if stop_outcome in (LineageOutcome.LINEAGE_CYCLE_DETECTED, LineageOutcome.LINEAGE_DEPTH_EXCEEDED):
                return ProtectionEvaluation(
                    state=ProtectionState.RECONCILIATION_REQUIRED,
                    reason=f"Stop lineage resolution failed ({stop_outcome.value}) while position open ({eff_pos_qty:.4f} shares)",
                    broker_position_qty=eff_pos_qty,
                    entry_filled_qty=entry_filled,
                    stop_filled_qty=stop_filled,
                    target_filled_qty=target_filled,
                    active_stop_remaining_qty=active_stop_qty,
                    active_target_remaining_qty=active_target_qty,
                    resolved_stop_id=resolved_stop_id,
                    resolved_target_id=resolved_target_id,
                    stop_lineage_outcome=stop_outcome,
                    target_lineage_outcome=target_outcome,
                )

            # Position is open -> must have active stop covering remaining open quantity
            if active_stop_qty >= eff_pos_qty:
                if stop_status == "held":
                    return ProtectionEvaluation(
                        state=ProtectionState.PROTECTED_PENDING_PARENT_FILL,
                        reason="Bracket child stop is held awaiting parent fill completion",
                        broker_position_qty=eff_pos_qty,
                        entry_filled_qty=entry_filled,
                        stop_filled_qty=stop_filled,
                        target_filled_qty=target_filled,
                        active_stop_remaining_qty=active_stop_qty,
                        active_target_remaining_qty=active_target_qty,
                        resolved_stop_id=resolved_stop_id,
                        resolved_target_id=resolved_target_id,
                        stop_lineage_outcome=stop_outcome,
                        target_lineage_outcome=target_outcome,
                    )
                elif active_target_qty >= eff_pos_qty:
                    return ProtectionEvaluation(
                        state=ProtectionState.PROTECTED_COMPLETE,
                        reason="Active stop and target orders cover open position",
                        broker_position_qty=eff_pos_qty,
                        entry_filled_qty=entry_filled,
                        stop_filled_qty=stop_filled,
                        target_filled_qty=target_filled,
                        active_stop_remaining_qty=active_stop_qty,
                        active_target_remaining_qty=active_target_qty,
                        resolved_stop_id=resolved_stop_id,
                        resolved_target_id=resolved_target_id,
                        stop_lineage_outcome=stop_outcome,
                        target_lineage_outcome=target_outcome,
                    )
                else:
                    logger.warning(
                        "[ENTRY LIFECYCLE] TARGET_UNAVAILABLE_NON_FATAL: %s target leg unavailable (status=%s, qty=%.4f); downside stop is active (qty=%.4f)",
                        symbol, target_status, active_target_qty, active_stop_qty
                    )
                    return ProtectionEvaluation(
                        state=ProtectionState.PROTECTED_STOP_ONLY,
                        reason="Active stop covers open position; target unavailable (non-fatal)",
                        broker_position_qty=eff_pos_qty,
                        entry_filled_qty=entry_filled,
                        stop_filled_qty=stop_filled,
                        target_filled_qty=target_filled,
                        active_stop_remaining_qty=active_stop_qty,
                        active_target_remaining_qty=active_target_qty,
                        resolved_stop_id=resolved_stop_id,
                        resolved_target_id=resolved_target_id,
                        stop_lineage_outcome=stop_outcome,
                        target_lineage_outcome=target_outcome,
                    )
            else:
                return ProtectionEvaluation(
                    state=ProtectionState.PROTECTION_FAILURE,
                    reason=f"Active stop quantity {active_stop_qty:.4f} < open position {eff_pos_qty:.4f} (stop_status={stop_status})",
                    broker_position_qty=eff_pos_qty,
                    entry_filled_qty=entry_filled,
                    stop_filled_qty=stop_filled,
                    target_filled_qty=target_filled,
                    active_stop_remaining_qty=active_stop_qty,
                    active_target_remaining_qty=active_target_qty,
                    resolved_stop_id=resolved_stop_id,
                    resolved_target_id=resolved_target_id,
                    stop_lineage_outcome=stop_outcome,
                    target_lineage_outcome=target_outcome,
                )

        # eff_pos_qty == 0: Position is flat
        eff_entry_qty = entry_filled if entry_filled > 0 else float(getattr(entry, "requested_qty", 0.0) or 0.0)
        if eff_entry_qty > 0:
            if math.isclose(stop_filled, eff_entry_qty, abs_tol=1e-5) and target_filled == 0.0:
                return ProtectionEvaluation(
                    state=ProtectionState.CLOSED_BY_STOP,
                    reason=f"Position flat; confirmed stop exit ({stop_filled:.4f} shares)",
                    broker_position_qty=0.0,
                    entry_filled_qty=entry_filled,
                    stop_filled_qty=stop_filled,
                    target_filled_qty=target_filled,
                    resolved_stop_id=resolved_stop_id,
                    resolved_target_id=resolved_target_id,
                    stop_lineage_outcome=stop_outcome,
                    target_lineage_outcome=target_outcome,
                )
            elif math.isclose(target_filled, eff_entry_qty, abs_tol=1e-5) and stop_filled == 0.0:
                return ProtectionEvaluation(
                    state=ProtectionState.CLOSED_BY_TARGET,
                    reason=f"Position flat; confirmed target exit ({target_filled:.4f} shares)",
                    broker_position_qty=0.0,
                    entry_filled_qty=entry_filled,
                    stop_filled_qty=stop_filled,
                    target_filled_qty=target_filled,
                    resolved_stop_id=resolved_stop_id,
                    resolved_target_id=resolved_target_id,
                    stop_lineage_outcome=stop_outcome,
                    target_lineage_outcome=target_outcome,
                )
            elif stop_filled > 0 and target_filled > 0 and math.isclose(stop_filled + target_filled, eff_entry_qty, abs_tol=1e-5):
                return ProtectionEvaluation(
                    state=ProtectionState.CLOSED_BY_MIXED_EXITS,
                    reason=f"Position flat; confirmed mixed exits (stop={stop_filled:.4f}, target={target_filled:.4f}, total={confirmed_exit_qty:.4f})",
                    broker_position_qty=0.0,
                    entry_filled_qty=entry_filled,
                    stop_filled_qty=stop_filled,
                    target_filled_qty=target_filled,
                    resolved_stop_id=resolved_stop_id,
                    resolved_target_id=resolved_target_id,
                    stop_lineage_outcome=stop_outcome,
                    target_lineage_outcome=target_outcome,
                )
            elif math.isclose(confirmed_exit_qty, eff_entry_qty, abs_tol=1e-5):
                return ProtectionEvaluation(
                    state=ProtectionState.CLOSED_BY_OTHER_CONFIRMED_EXIT,
                    reason=f"Position flat; confirmed other exit quantity ({confirmed_exit_qty:.4f} shares)",
                    broker_position_qty=0.0,
                    entry_filled_qty=entry_filled,
                    stop_filled_qty=stop_filled,
                    target_filled_qty=target_filled,
                    resolved_stop_id=resolved_stop_id,
                    resolved_target_id=resolved_target_id,
                    stop_lineage_outcome=stop_outcome,
                    target_lineage_outcome=target_outcome,
                )
            elif confirmed_exit_qty > eff_entry_qty + 1e-5:
                return ProtectionEvaluation(
                    state=ProtectionState.RECONCILIATION_REQUIRED,
                    reason=f"Position flat on broker but OVERFILL detected: exit quantity ({confirmed_exit_qty:.4f}) > entry filled ({eff_entry_qty:.4f})",
                    broker_position_qty=0.0,
                    entry_filled_qty=entry_filled,
                    stop_filled_qty=stop_filled,
                    target_filled_qty=target_filled,
                    resolved_stop_id=resolved_stop_id,
                    resolved_target_id=resolved_target_id,
                    stop_lineage_outcome=stop_outcome,
                    target_lineage_outcome=target_outcome,
                )
            else:
                return ProtectionEvaluation(
                    state=ProtectionState.RECONCILIATION_REQUIRED,
                    reason=f"Position flat on broker but confirmed exit quantity ({confirmed_exit_qty:.4f}) < entry filled ({eff_entry_qty:.4f})",
                    broker_position_qty=0.0,
                    entry_filled_qty=entry_filled,
                    stop_filled_qty=stop_filled,
                    target_filled_qty=target_filled,
                    resolved_stop_id=resolved_stop_id,
                    resolved_target_id=resolved_target_id,
                    stop_lineage_outcome=stop_outcome,
                    target_lineage_outcome=target_outcome,
                )
        else:
            return ProtectionEvaluation(
                state=ProtectionState.RECONCILIATION_REQUIRED,
                reason="Position flat and entry filled qty is 0.0",
                broker_position_qty=0.0,
                entry_filled_qty=0.0,
                resolved_stop_id=resolved_stop_id,
                resolved_target_id=resolved_target_id,
                stop_lineage_outcome=stop_outcome,
                target_lineage_outcome=target_outcome,
            )

    def _protection_covers_fill(self, bridge: object, entry: PendingEntry) -> bool:
        eval_res = self.evaluate_protection_state(bridge, entry)
        return eval_res.state in (
            ProtectionState.PROTECTED_COMPLETE,
            ProtectionState.PROTECTED_STOP_ONLY,
            ProtectionState.CLOSED_BY_STOP,
            ProtectionState.CLOSED_BY_TARGET,
            ProtectionState.CLOSED_BY_MIXED_EXITS,
            ProtectionState.CLOSED_BY_OTHER_CONFIRMED_EXIT,
        )

    @staticmethod
    def _halt_entries(bridge: object, reason: str) -> None:
        try:
            from strategies.halt_manager import halt_manager, LATCHING_HALT_PROTECTION_FAILURE
            halt_manager.add(LATCHING_HALT_PROTECTION_FAILURE)
        except Exception:
            logger.exception("[ENTRY LIFECYCLE] Failed to register latched halt with halt_manager")

        gate = getattr(bridge, "entry_gate", None)
        if gate is not None:
            try:
                if callable(getattr(gate, "halt", None)):
                    gate.halt(reason)
                else:
                    gate.halted = True
            except Exception:
                logger.exception("[ENTRY LIFECYCLE] Failed to halt entry gate")
        try:
            setattr(bridge, "_daytrade_suspended", True)
        except Exception:
            logger.exception("[ENTRY LIFECYCLE] Failed to mark bridge entry suspension")

    def _fail_protection(self, bridge: object, entry: PendingEntry, reason: str) -> None:
        """Block entries and attempt a broker close when confirmed shares lack protection."""
        filled_qty = getattr(entry, "cumulative_filled_qty", 0.0)
        avg_px = getattr(entry, "average_fill_price", 0.0)
        stop_id = getattr(entry, "stop_order_id", "NONE")
        tp_id = getattr(entry, "target_order_id", "NONE")
        parent_id = getattr(entry, "parent_order_id", "NONE")

        msg = (
            f"[PROTECTION FAILURE] symbol={entry.symbol} parent={parent_id} "
            f"filled_qty={filled_qty} avg_px={avg_px:.2f} stop_id={stop_id} tp_id={tp_id} "
            f"reason='{reason}'; new entries halted"
        )
        logger.critical(msg)
        self._halt_entries(bridge, msg)

        # Trigger LatchStateMachine with structured context
        try:
            from observability import get_latch_state_machine, LatchTrigger
            sm = get_latch_state_machine()
            sm.latch(
                LatchTrigger.UNPROTECTED_POSITION,
                bridge=bridge,
                symbol=getattr(entry, "symbol", None),
                parent_order_id=str(parent_id) if parent_id else None,
                stop_order_id=str(stop_id) if stop_id else None,
                target_order_id=str(tp_id) if tp_id else None,
                filled_qty=filled_qty,
                reasons=[f"protection_failure: {reason}"],
            )
        except Exception as sm_exc:
            logger.debug("[ENTRY LIFECYCLE] Observability latch hook non-fatal error: %s", sm_exc)

        try:
            send_discord_alert(msg)
        except Exception:
            logger.exception("[ENTRY LIFECYCLE] Failed to send protection alert")

        close_position = getattr(bridge, "close_position", None)
        if callable(close_position):
            try:
                close_position(entry.symbol)
                logger.warning("[ENTRY LIFECYCLE] Requested emergency close for %s", entry.symbol)
            except Exception:
                logger.exception("[ENTRY LIFECYCLE] Emergency close failed for %s", entry.symbol)
        else:
            logger.critical("[ENTRY LIFECYCLE] Bridge cannot close unprotected %s", entry.symbol)

    def fail_submission_tracking(
        self,
        *,
        bridge: object,
        signal: TradeSignal,
        parent_order_id: object,
        child_order_ids: Iterable[object],
        reason: str,
    ) -> None:
        """Fail closed if a submitted bracket cannot be lifecycle-tracked.

        This path is intentionally conservative.  It halts future entries and
        asks the adapter to cancel every known order leg.  The adapter should
        subsequently reconcile/close any unexpected filled shares.
        """
        msg = f"[ENTRY LIFECYCLE FAILURE] {signal.symbol}: {reason}; new entries halted"
        logger.critical(msg)
        self._halt_entries(bridge, msg)

        # Trigger LatchStateMachine with structured context
        try:
            from observability import get_latch_state_machine, LatchTrigger
            sm = get_latch_state_machine()
            sm.latch(
                LatchTrigger.ORDER_REJECT,
                bridge=bridge,
                symbol=getattr(signal, "symbol", None),
                parent_order_id=str(parent_order_id) if parent_order_id else None,
                reasons=[f"submission_tracking_failure: {reason}"],
            )
        except Exception as sm_exc:
            logger.debug("[ENTRY LIFECYCLE] Observability latch hook non-fatal error: %s", sm_exc)

        try:
            send_discord_alert(msg)
        except Exception:
            logger.exception("[ENTRY LIFECYCLE] Failed to send tracking-failure alert")

        cancel = getattr(bridge, "cancel_order", None)
        if callable(cancel):
            for order_id in (parent_order_id, *child_order_ids):
                normalized = self._order_id(order_id)
                if normalized is None:
                    continue
                try:
                    cancel(normalized)
                except Exception:
                    logger.exception("[ENTRY LIFECYCLE] Failed to cancel untracked order %s", normalized)

    def on_parent_fill(
        self,
        *,
        bridge: object,
        parent_order_id: object,
        cumulative_filled_qty: float,
        average_fill_price: float,
        execution_id: object = None,
    ) -> bool:
        """Apply a confirmed entry fill. Returns True only after state is protected or cleanly closed."""
        update = self.record_parent_fill(
            parent_order_id=parent_order_id,
            cumulative_filled_qty=cumulative_filled_qty,
            average_fill_price=average_fill_price,
            execution_id=execution_id,
        )
        if update is None:
            return False

        entry = update.entry
        eval_res = self.evaluate_protection_state(bridge, entry)

        # Log confirmed entry fill to trade journal
        try:
            from strategies.trade_journal import log_fill, get_journal_lock
            delta_qty = int(update.fill_delta) if (update.fill_delta % 1 == 0) else float(update.fill_delta)
            fill_px = update.incremental_price if update.incremental_price > 0 else average_fill_price
            exp_px = entry.submitted_entry_price or entry.signal.entry_price
            with get_journal_lock():
                log_fill(
                    symbol=entry.symbol,
                    side=entry.signal.side,
                    qty=delta_qty,
                    fill_price=fill_px,
                    expected_price=exp_px,
                    slippage=(fill_px - exp_px) / exp_px if exp_px else 0.0,
                    fill_latency_ms=max(0.0, (time.time() - entry.submitted_at) * 1000.0) if entry.submitted_at else None,
                    order_id=str(entry.parent_order_id),
                    broker_order_id=str(entry.parent_order_id),
                    execution_id=str(execution_id) if execution_id else str(entry.parent_order_id),
                    signal_id=getattr(entry.signal, "signal_id", None),
                    strategy=getattr(entry.signal, "strategy", "UNKNOWN"),
                    account_id=getattr(bridge, "account_id", None) or "ALPACA",
                )
        except Exception as j_err:
            logger.debug("[ENTRY LIFECYCLE] Non-fatal journal fill log error: %s", j_err)

        # Handle closed states immediately (e.g. stop filled immediately at broker)
        if eval_res.state in (
            ProtectionState.CLOSED_BY_STOP,
            ProtectionState.CLOSED_BY_TARGET,
            ProtectionState.CLOSED_BY_MIXED_EXITS,
            ProtectionState.CLOSED_BY_OTHER_CONFIRMED_EXIT,
        ):
            logger.info(
                "[ENTRY LIFECYCLE] %s position already closed on broker (%s: %s). Bypassing trailing initialization; no latch.",
                entry.symbol, eval_res.state.value, eval_res.reason
            )
            with self._lock:
                self._pending.pop(str(entry.parent_order_id), None)
            try:
                from strategies.order_idempotency import idempotency_manager
                idempotency_manager.release_on_terminal(str(entry.parent_order_id), "filled")
            except Exception:
                pass
            return True

        # Handle reconciliation required (bounded retry for in-flight exit fill report)
        if eval_res.state == ProtectionState.RECONCILIATION_REQUIRED:
            logger.info(
                "[ENTRY LIFECYCLE] %s flat but awaiting exit fill confirmation (%s). Retrying bounded reconciliation...",
                entry.symbol, eval_res.reason
            )
            SETTLEMENT_MAX_ATTEMPTS = 3
            backoff_s = 0.5
            for attempt in range(SETTLEMENT_MAX_ATTEMPTS):
                time.sleep(backoff_s)
                backoff_s *= 1.5
                eval_res = self.evaluate_protection_state(bridge, entry)
                if eval_res.state in (
                    ProtectionState.CLOSED_BY_STOP,
                    ProtectionState.CLOSED_BY_TARGET,
                    ProtectionState.CLOSED_BY_MIXED_EXITS,
                    ProtectionState.CLOSED_BY_OTHER_CONFIRMED_EXIT,
                ):
                    logger.info(
                        "[ENTRY LIFECYCLE] %s resolved to %s on retry %d",
                        entry.symbol, eval_res.state.value, attempt + 1
                    )
                    with self._lock:
                        self._pending.pop(str(entry.parent_order_id), None)
                    return True
                if eval_res.state != ProtectionState.RECONCILIATION_REQUIRED:
                    break

            if eval_res.state == ProtectionState.RECONCILIATION_REQUIRED:
                logger.warning(
                    "[ENTRY LIFECYCLE] RECONCILIATION_REQUIRED for %s: flat on broker without matching exit execution. Non-fatal for new entries.",
                    entry.symbol
                )
                with self._lock:
                    self._pending.pop(str(entry.parent_order_id), None)
                return True

        # Handle protection failure (open position with insufficient stop protection)
        if eval_res.state == ProtectionState.PROTECTION_FAILURE:
            self._fail_protection(bridge, entry, eval_res.reason)
            return False

        # Handle protected open states (PROTECTED_COMPLETE or PROTECTED_STOP_ONLY)
        try:
            self._trailing_manager.apply_confirmed_fill(
                entry.signal,
                average_fill_price=entry.average_fill_price,
                cumulative_qty=entry.cumulative_filled_qty,
                stop_order_id=eval_res.resolved_stop_id,
                bridge=bridge,
                tp_order_id=eval_res.resolved_target_id,
            )
        except Exception as exc:
            self._fail_protection(bridge, entry, f"trailing-state initialization failed: {exc}")
            return False

        status = "FULL" if update.is_complete else "PARTIAL"
        logger.info(
            "[ENTRY LIFECYCLE] %s fill protected (%s): symbol=%s parent=%s qty=%.4f avg=%.6f stop_id=%s tp_id=%s",
            status,
            eval_res.state.value,
            entry.symbol,
            entry.parent_order_id,
            entry.cumulative_filled_qty,
            entry.average_fill_price,
            eval_res.resolved_stop_id,
            eval_res.resolved_target_id,
        )
        return True

    def on_parent_terminal(self, parent_order_id: object, status: str) -> None:
        """Retain terminal parent tracking until broker fill/position/leg reconciliation.

        Stream terminal events may precede a late fill callback or the final
        broker position update. Never release a reservation or discard the
        parent solely because a canceled/expired/rejected status arrived.
        The verified reaper owns zero-fill finalization and protected fills
        follow on_parent_fill's broker-authoritative path.
        """
        parent_id = self._order_id(parent_order_id)
        if parent_id is None:
            return
        with self._lock:
            entry = self._pending.get(parent_id)
            if entry is not None:
                entry.state = EntryLifecycleState.RECONCILING
        if entry is not None:
            logger.warning(
                "[ENTRY LIFECYCLE] Terminal event deferred: parent=%s symbol=%s "
                "status=%s cumulative_fill=%.4f; reservation retained until "
                "authoritative broker position/stop reconciliation",
                parent_id, entry.symbol, status, entry.cumulative_filled_qty,
            )


    def pending_snapshot(self) -> Dict[str, Dict[str, object]]:
        """Read-only diagnostics; do not use this snapshot as broker truth."""
        with self._lock:
            return {
                parent_id: {
                    "symbol": entry.symbol,
                    "requested_qty": entry.requested_qty,
                    "cumulative_filled_qty": entry.cumulative_filled_qty,
                    "average_fill_price": entry.average_fill_price,
                    "stop_order_id": entry.stop_order_id,
                    "target_order_id": entry.target_order_id,
                    "submitted_at": entry.submitted_at,
                    "submitted_entry_price": entry.submitted_entry_price,
                    "submitted_stop_price": entry.submitted_stop_price,
                    "submitted_target_price": entry.submitted_target_price,
                    "state": entry.state.value if hasattr(entry.state, "value") else str(entry.state),
                    "expires_at": entry.expires_at,
                    "decision_bar_timestamp": entry.decision_bar_timestamp,
                }
                for parent_id, entry in self._pending.items()
            }

    def reconcile_and_protect_partial_entry(
        self,
        bridge: object,
        entry: PendingEntry,
        intended_filled_qty: float,
    ) -> Tuple[bool, str]:
        """
        Executes explicit partial-fill protection sequence (Amendment 3):
        1. Check if an active stop covering intended_filled_qty exists.
        2. If not, place an explicit protective stop for exact intended_filled_qty.
        3. Register position with TrailingStopManager.
        4. Verify exact protection via verify_exact_protection().
        """
        sym = entry.symbol.upper().strip()
        side = entry.signal.side.upper().strip()
        stop_px = entry.submitted_stop_price or entry.signal.stop_price
        exit_side = "SELL" if side == "BUY" else "BUY"

        # Check existing protection first to prevent duplicate stops
        ok, reason, active_stop_id = verify_exact_protection(
            bridge=bridge,
            symbol=sym,
            filled_qty=intended_filled_qty,
            entry_side=side,
            stop_price=stop_px,
        )
        if ok and active_stop_id:
            logger.info("[PARTIAL FILL] %s already exactly protected by stop_id=%s", sym, active_stop_id)
            try:
                self._trailing_manager.apply_confirmed_fill(
                    entry.signal,
                    average_fill_price=entry.average_fill_price,
                    cumulative_qty=intended_filled_qty,
                    stop_order_id=active_stop_id,
                    bridge=bridge,
                    tp_order_id=None,
                )
            except Exception as t_exc:
                logger.warning("[PARTIAL FILL] Trailing manager registration warning: %s", t_exc)
            return True, active_stop_id

        # If duplicate stops detected, fail closed
        if "DUPLICATE_STOPS_FORBIDDEN" in reason:
            return False, reason

        # Place explicit protective stop for exact intended filled quantity
        stop_id = None
        if hasattr(bridge, "place_protective_stop"):
            try:
                stop_id = bridge.place_protective_stop(
                    symbol=sym,
                    side=exit_side,
                    qty=intended_filled_qty,
                    stop_price=stop_px,
                )
            except Exception as exc:
                return False, f"FAILED_TO_PLACE_PROTECTIVE_STOP: {exc}"
        elif hasattr(bridge, "place_stop_order"):
            try:
                res = bridge.place_stop_order(
                    symbol=sym,
                    side=exit_side,
                    qty=int(intended_filled_qty),
                    stop_price=stop_px,
                )
                stop_id = res.get("order_id") if res else None
            except Exception as exc:
                return False, f"FAILED_TO_PLACE_STOP_ORDER: {exc}"

        if not stop_id:
            return False, "PROTECTIVE_STOP_SUBMISSION_RETURNED_EMPTY_ID"

        # Register with trailing stop manager
        try:
            self._trailing_manager.apply_confirmed_fill(
                entry.signal,
                average_fill_price=entry.average_fill_price,
                cumulative_qty=intended_filled_qty,
                stop_order_id=stop_id,
                bridge=bridge,
                tp_order_id=None,
            )
        except Exception as t_exc:
            logger.warning("[PARTIAL FILL] Trailing manager registration warning: %s", t_exc)

        # Re-verify exact protection
        ver_ok, ver_reason, final_stop_id = verify_exact_protection(
            bridge=bridge,
            symbol=sym,
            filled_qty=intended_filled_qty,
            entry_side=side,
            stop_price=stop_px,
        )
        if not ver_ok:
            return False, f"PROTECTION_VERIFICATION_FAILED_AFTER_PLACEMENT: {ver_reason}"

        return True, final_stop_id or stop_id


def verify_exact_protection(
    bridge: object,
    symbol: str,
    filled_qty: float,
    entry_side: str,
    stop_price: float,
) -> Tuple[bool, str, Optional[str]]:
    """
    Authoritative Protection Verification strictly enforcing Amendment 3:
      broker_position_qty == protected_stop_qty == intended_filled_qty
    Requires:
      - Valid finite positive intended filled_qty and stop_price
      - Authoritative broker position matches intended filled qty
      - Exactly one working protective order covering the position verified via verify_broker_stop_order_evidence
        (matching identity, symbol, opposite exit side, type, working status whitelist, remaining capacity, quantized stop price)
      - Duplicate protective stops strictly forbidden
    Returns: (is_verified: bool, reason: str, active_stop_id: Optional[str])
    """
    sym = symbol.upper().strip() if symbol else ""
    if not sym:
        return False, "EXACT_PROTECTION_INVALID_SYMBOL: Symbol must be non-empty", None

    # Validate intended filled_qty at boundary
    try:
        intended_qty = float(filled_qty)
        if not math.isfinite(intended_qty) or intended_qty <= 0.0:
            return False, f"EXACT_PROTECTION_INVALID_QTY: intended filled_qty {filled_qty} is not finite or <= 0", None
    except (ValueError, TypeError):
        return False, f"EXACT_PROTECTION_INVALID_QTY: could not parse intended filled_qty {filled_qty}", None

    # Validate expected stop_price at boundary
    try:
        exp_stop_px = float(stop_price)
        if not math.isfinite(exp_stop_px) or exp_stop_px <= 0.0:
            return False, f"EXACT_PROTECTION_INVALID_STOP_PRICE: stop_price {stop_price} is not finite or <= 0", None
    except (ValueError, TypeError):
        return False, f"EXACT_PROTECTION_INVALID_STOP_PRICE: could not parse stop_price {stop_price}", None

    # 1. Fetch broker position truth
    if hasattr(bridge, "get_position_truth"):
        pos_truth = bridge.get_position_truth(sym)
        raw_pos_qty = getattr(pos_truth, "qty", 0.0)
    else:
        get_pos = getattr(bridge, "get_position", None)
        raw_pos_qty = get_pos(sym) if callable(get_pos) else 0.0

    try:
        broker_position_qty = abs(float(raw_pos_qty or 0.0))
        if not math.isfinite(broker_position_qty) or broker_position_qty <= 0.0:
            return False, f"EXACT_PROTECTION_INVALID_BROKER_QTY: broker position qty {raw_pos_qty} is not finite or <= 0", None
    except (ValueError, TypeError):
        return False, f"EXACT_PROTECTION_INVALID_BROKER_QTY: could not parse broker position qty {raw_pos_qty}", None

    if abs(broker_position_qty - intended_qty) > 1e-4:
        return (
            False,
            f"QTY_MISMATCH: broker_position_qty={broker_position_qty:.4f} != intended_filled_qty={intended_qty:.4f}",
            None,
        )

    # 2. Query active protective stops for symbol
    if hasattr(bridge, "get_active_stops_for_symbol"):
        stops = bridge.get_active_stops_for_symbol(sym)
    else:
        # Fallback inspection of active orders
        get_active = getattr(bridge, "_get_active_orders", None)
        al_orders = get_active() if callable(get_active) else []
        stops = []
        for o in al_orders:
            if str(getattr(o, "symbol", "")).upper().strip() != sym:
                continue
            otype = str(getattr(o, "order_type", getattr(o, "type", ""))).lower().replace("ordertype.", "")
            if otype in ("stop", "stop_limit", "trailing_stop"):
                raw_s = getattr(o, "side", "")
                s_norm = (raw_s.value if hasattr(raw_s, "value") else str(raw_s)).upper().replace("ORDERSIDE.", "").strip()
                raw_st = getattr(o, "status", "")
                st_norm = (raw_st.value if hasattr(raw_st, "value") else str(raw_st)).lower().replace("orderstatus.", "").strip()
                stops.append({
                    "id": str(getattr(o, "id", "")),
                    "symbol": sym,
                    "side": s_norm,
                    "qty": float(getattr(o, "qty", 0.0) or 0.0),
                    "filled_qty": float(getattr(o, "filled_qty", 0.0) or 0.0),
                    "stop_price": float(getattr(o, "stop_price", 0.0) or 0.0),
                    "status": st_norm,
                    "type": otype,
                })

    if not stops:
        return (
            False,
            f"NO_ACTIVE_STOP: Zero active protective stops found for {sym}",
            None,
        )

    # 3. Verify candidate stops using canonical verify_broker_stop_order_evidence
    from strategies.trailing_stop import verify_broker_stop_order_evidence, _extract_order_field

    verified_stops = []
    rejection_reasons = []
    for s in stops:
        s_id = str(_extract_order_field(s, "id", "order_id") or "")
        if not s_id:
            rejection_reasons.append("Stop lacks id")
            continue
        is_valid, reason = verify_broker_stop_order_evidence(
            order_data=s,
            expected_order_id=s_id,
            expected_symbol=sym,
            position_side=entry_side,
            expected_stop_price=exp_stop_px,
            expected_qty=intended_qty,
            require_qty=True,
        )
        if is_valid:
            verified_stops.append((s, s_id))
        else:
            rejection_reasons.append(f"{s_id}: {reason}")

    # Invariant: Duplicate protective stops are strictly forbidden
    if len(verified_stops) > 1:
        return (
            False,
            f"DUPLICATE_STOPS_FORBIDDEN: Found {len(verified_stops)} active stops for {sym}: {[vid for _, vid in verified_stops]}",
            None,
        )

    if not verified_stops:
        reasons_summary = "; ".join(rejection_reasons[:3])
        return (
            False,
            f"NO_ACTIVE_STOP: Zero active protective stops verified for {sym} ({reasons_summary})",
            None,
        )

    active_stop, active_stop_id = verified_stops[0]
    return (
        True,
        f"EXACT_PROTECTION_VERIFIED: broker_position={broker_position_qty:.4f} == stop_qty={intended_qty:.4f} (stop_id={active_stop_id})",
        active_stop_id,
    )


from strategies.trailing_stop import manager as _trailing_manager
entry_lifecycle = EntryOrderLifecycle(_trailing_manager)

