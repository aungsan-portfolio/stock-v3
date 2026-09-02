"""
strategies/halt_manager.py -- Thread-safe entry halt state manager.

Maintains latching vs transient halt separation with explicit membership frozensets.
Guarantees that latched halts (protection failure, daily loss breach) cannot be
accidentally cleared by transient recovery paths (e.g. regime data recovery).
"""
from __future__ import annotations

import logging
import threading
from typing import FrozenSet, Set

logger = logging.getLogger(__name__)

LATCHING_HALT_PROTECTION_FAILURE = "LATCHED_PROTECTION_FAILURE"
LATCHING_HALT_DAILY_LOSS_BREACH = "LATCHED_DAILY_LOSS_BREACH"
LATCHING_HALT_ORDER_TRACKING_FAILURE = "LATCHED_ORDER_TRACKING_FAILURE"

TRANSIENT_HALT_DATA_DEGRADED = "DATA_DEGRADED"
TRANSIENT_HALT_GLOBAL_OPENING_BUFFER = "GLOBAL_OPENING_BUFFER"
TRANSIENT_HALT_FILE_LOCK = "FILE_LOCK_HALT"
TRANSIENT_HALT_RECONCILIATION = "RECONCILIATION_IN_PROGRESS"
TRANSIENT_HALT_EVENT_BLACKOUT = "EVENT_BLACKOUT"

_KNOWN_LATCHING_HALTS: FrozenSet[str] = frozenset({
    LATCHING_HALT_PROTECTION_FAILURE,
    LATCHING_HALT_DAILY_LOSS_BREACH,
    LATCHING_HALT_ORDER_TRACKING_FAILURE,
})

_KNOWN_TRANSIENT_HALTS: FrozenSet[str] = frozenset({
    TRANSIENT_HALT_DATA_DEGRADED,
    TRANSIENT_HALT_GLOBAL_OPENING_BUFFER,
    TRANSIENT_HALT_FILE_LOCK,
    TRANSIENT_HALT_RECONCILIATION,
    TRANSIENT_HALT_EVENT_BLACKOUT,
})


class EntryHaltManager:
    """
    Thread-safe registry for active entry halts.
    Guarantees that latched halts cannot be cleared by transient un-halt paths.
    Unknown halt reasons default to latching (fail-closed).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active_halts: Set[str] = set()

    def add(self, reason: str, bridge: Optional[object] = None) -> bool:
        """Add an active halt reason. Logs warning only on initial state transition (prevents log spam)."""
        if not reason:
            return False
        with self._lock:
            newly_added = reason not in self._active_halts
            self._active_halts.add(reason)
            if newly_added:
                logger.warning("[HALT MANAGER] Entry halt activated: %s (Active: %s)", reason, sorted(self._active_halts))
                # Trigger LatchStateMachine for latching halt events
                try:
                    from observability import get_latch_state_machine, LatchTrigger
                    sm = get_latch_state_machine()
                    trigger_map = {
                        LATCHING_HALT_PROTECTION_FAILURE: LatchTrigger.UNPROTECTED_POSITION,
                        LATCHING_HALT_DAILY_LOSS_BREACH: LatchTrigger.DAILY_LOSS_LIMIT,
                        LATCHING_HALT_ORDER_TRACKING_FAILURE: LatchTrigger.ORDER_REJECT,
                        "MANUAL_HALT": LatchTrigger.MANUAL_HALT,
                    }
                    trig = trigger_map.get(reason)
                    if trig is not None:
                        sm.latch(trig, bridge=bridge, reasons=[reason])
                except Exception as sm_exc:
                    logger.debug("[HALT MANAGER] Observability latch hook non-fatal error: %s", sm_exc)
            return newly_added

    def clear_transient(self, reason: str) -> bool:
        """
        Clear a transient halt reason.
        Refuses to clear latching halts or unknown reasons (fail-closed).
        Logs info only on initial state transition (prevents log spam).
        """
        if not reason:
            return False
        with self._lock:
            # Explicit guard: only known transient halts can be cleared here
            if reason not in _KNOWN_TRANSIENT_HALTS or reason in _KNOWN_LATCHING_HALTS:
                logger.warning(
                    "[HALT MANAGER] Refusing to clear non-transient / latched halt: %s (Active: %s)",
                    reason,
                    sorted(self._active_halts),
                )
                return False

            if reason in self._active_halts:
                self._active_halts.discard(reason)
                logger.info(
                    "[HALT MANAGER] Transient halt cleared: %s (Remaining: %s)",
                    reason,
                    sorted(self._active_halts),
                )
                return True
            return False

    def clear_latched_for_reconciliation(self, reason: str) -> bool:
        """
        Explicit operator/reconciliation clearance for latched halts.
        Requires explicit call during successful state reconciliation.
        """
        if not reason:
            return False
        with self._lock:
            if reason in self._active_halts:
                self._active_halts.discard(reason)
                logger.warning(
                    "[HALT MANAGER] Latched halt explicitly cleared via reconciliation: %s (Remaining: %s)",
                    reason,
                    sorted(self._active_halts),
                )
                try:
                    from observability import get_latch_state_machine, ClearReason
                    sm = get_latch_state_machine()
                    if not any(h in self._active_halts for h in _KNOWN_LATCHING_HALTS):
                        sm.clear(ClearReason.STARTUP_RECONCILIATION_CLEAN)
                except Exception:
                    pass
                return True
            return False

    def reset_all(self) -> None:
        """Clear all active halts (e.g. at daily session start)."""
        with self._lock:
            self._active_halts.clear()
            logger.info("[HALT MANAGER] All active entry halts reset.")
            try:
                from observability import get_latch_state_machine, ClearReason
                sm = get_latch_state_machine()
                sm.clear(ClearReason.MANUAL_RESET)
            except Exception:
                pass

    def entries_allowed(self) -> bool:
        """True if and only if no active halts are present."""
        with self._lock:
            return len(self._active_halts) == 0

    def active_halts(self) -> Set[str]:
        """Return a copy of the active halt reasons."""
        with self._lock:
            return set(self._active_halts)

    def summary(self) -> str:
        with self._lock:
            if not self._active_halts:
                return "Entries Allowed (No active halts)"
            return f"Entries Halted: {', '.join(sorted(self._active_halts))}"

    def check_halt_lock(self) -> bool:
        """
        Checks for HALT.lock file presence in config.DATA_DIR or config.BASE_DIR.
        Fail-closed: Any filesystem/permission exception triggers FILE_LOCK_HALT.
        Returns True if halted, False if normal.
        """
        try:
            import config
            from pathlib import Path
            data_lock = getattr(config, "HALT_LOCK_FILE", getattr(config, "DATA_DIR", Path("data")) / "HALT.lock")
            base_lock = getattr(config, "BASE_DIR", Path(".")) / "HALT.lock"

            lock_found = False
            if Path(data_lock).exists():
                lock_found = True
            elif Path(base_lock).exists():
                lock_found = True

            if lock_found:
                self.add(TRANSIENT_HALT_FILE_LOCK)
                return True
            else:
                self.clear_transient(TRANSIENT_HALT_FILE_LOCK)
                return False
        except Exception as exc:
            logger.exception("[HALT MANAGER] Exception checking HALT.lock file; failing closed: %s", exc)
            self.add(TRANSIENT_HALT_FILE_LOCK)
            return True

    def apply_event_blackout(self, active: bool, tag: Optional[str] = None) -> bool:
        """Toggle event blackout halt. Mirrors opening-buffer halt handling."""
        if active:
            return self.add(TRANSIENT_HALT_EVENT_BLACKOUT)
        else:
            return self.clear_transient(TRANSIENT_HALT_EVENT_BLACKOUT)


# Global singleton instance
halt_manager = EntryHaltManager()

