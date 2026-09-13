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


@dataclass
class PendingEntry:
    parent_order_id: str
    symbol: str
    signal: TradeSignal
    requested_qty: float
    stop_order_id: Optional[str]
    target_order_id: Optional[str]
    submitted_at: float = field(default_factory=time.time)
    cumulative_filled_qty: float = 0.0
    average_fill_price: float = 0.0
    execution_ids: Set[str] = field(default_factory=set)
    submitted_entry_price: Optional[float] = None
    submitted_stop_price: Optional[float] = None
    submitted_target_price: Optional[float] = None


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
    ) -> PendingEntry:
        """Record a submitted bracket; this does *not* create position state."""
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

        entry = PendingEntry(
            parent_order_id=parent_id,
            symbol=effective_signal.symbol.upper().strip(),
            signal=effective_signal,
            requested_qty=float(requested_qty),
            stop_order_id=stop_id,
            target_order_id=target_id,
            submitted_entry_price=entry_px,
            submitted_stop_price=stop_px,
            submitted_target_price=target_px,
        )
        with self._lock:
            if parent_id in self._pending:
                raise RuntimeError(f"Duplicate pending parent order ID: {parent_id}")
            self._pending[parent_id] = entry
        logger.info(
            "[ENTRY LIFECYCLE] Registered pending %s parent=%s qty=%.4f stop=%s target=%s",
            entry.symbol,
            entry.parent_order_id,
            entry.requested_qty,
            entry.stop_order_id,
            entry.target_order_id,
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
                if active_target_qty >= eff_pos_qty:
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
        """Remove a pending record after an unfilled terminal parent state."""
        parent_id = self._order_id(parent_order_id)
        if parent_id is None:
            return
        with self._lock:
            entry = self._pending.pop(parent_id, None)
        if entry:
            logger.info("[ENTRY LIFECYCLE] Terminal parent=%s symbol=%s status=%s", parent_id, entry.symbol, status)

        try:
            from strategies.order_idempotency import idempotency_manager
            idempotency_manager.release_on_terminal(str(parent_id), status)
        except Exception as e:
            logger.debug("[ENTRY LIFECYCLE] Idempotency release error: %s", e)


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
                }
                for parent_id, entry in self._pending.items()
            }


from strategies.trailing_stop import manager as _trailing_manager
entry_lifecycle = EntryOrderLifecycle(_trailing_manager)
