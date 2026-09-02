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
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from strategies.base import TradeSignal
from strategies.webhook import send_discord_alert

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"canceled", "cancelled", "rejected", "expired", "done_for_day", "filled"}


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

            entry.cumulative_filled_qty = reported_qty
            entry.average_fill_price = float(average_fill_price)
            return FillUpdate(
                entry=entry,
                fill_delta=fill_delta,
                is_complete=reported_qty >= entry.requested_qty,
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

    def _protective_order_covers_fill(self, bridge: object, order_id: Optional[str], filled_qty: float, order_label: str = "child") -> bool:
        if not order_id:
            logger.warning("[ENTRY LIFECYCLE] %s order_id is missing/empty", order_label)
            return False
        getter = getattr(bridge, "get_order_snapshot", None)
        if not callable(getter):
            logger.error("[ENTRY LIFECYCLE] Bridge has no get_order_snapshot; cannot verify protection for %s (%s)", order_label, order_id)
            return False
        try:
            snapshot = getter(order_id)
            if snapshot is None:
                logger.warning("[ENTRY LIFECYCLE] Order snapshot returned None for %s (%s)", order_label, order_id)
                return False
            status = str(self._field(snapshot, "status", default="")).lower()
            quantity = float(self._field(snapshot, "qty", "quantity", "total_quantity", default=0.0) or 0.0)
            if status in _TERMINAL_STATUSES:
                logger.warning("[ENTRY LIFECYCLE] %s order (%s) has terminal status: %s", order_label, order_id, status)
                return False
            if quantity < float(filled_qty):
                logger.warning("[ENTRY LIFECYCLE] %s order (%s) qty %.4f < filled_qty %.4f", order_label, order_id, quantity, float(filled_qty))
                return False
            return True
        except Exception:
            logger.exception("[ENTRY LIFECYCLE] Failed protection lookup for %s order %s", order_label, order_id)
            return False

    def _protection_covers_fill(self, bridge: object, entry: PendingEntry) -> bool:
        filled = entry.cumulative_filled_qty
        stop_ok = self._protective_order_covers_fill(bridge, entry.stop_order_id, filled, order_label=f"STOP:{entry.symbol}")
        tp_ok = self._protective_order_covers_fill(bridge, entry.target_order_id, filled, order_label=f"TARGET:{entry.symbol}")
        return stop_ok and tp_ok

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
        """Apply a confirmed entry fill.  Returns True only after state is protected."""
        update = self.record_parent_fill(
            parent_order_id=parent_order_id,
            cumulative_filled_qty=cumulative_filled_qty,
            average_fill_price=average_fill_price,
            execution_id=execution_id,
        )
        if update is None:
            return False

        entry = update.entry
        if not self._protection_covers_fill(bridge, entry):
            self._fail_protection(
                bridge,
                entry,
                "stop and/or target child order does not cover confirmed filled quantity",
            )
            return False

        try:
            self._trailing_manager.apply_confirmed_fill(
                entry.signal,
                average_fill_price=entry.average_fill_price,
                cumulative_qty=entry.cumulative_filled_qty,
                stop_order_id=entry.stop_order_id,
                bridge=bridge,
                tp_order_id=entry.target_order_id,
            )
        except Exception as exc:
            self._fail_protection(bridge, entry, f"trailing-state initialization failed: {exc}")
            return False

        status = "FULL" if update.is_complete else "PARTIAL"
        logger.info(
            "[ENTRY LIFECYCLE] %s fill protected: symbol=%s parent=%s qty=%.4f avg=%.6f",
            status,
            entry.symbol,
            entry.parent_order_id,
            entry.cumulative_filled_qty,
            entry.average_fill_price,
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
