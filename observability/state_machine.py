"""
observability/state_machine.py -- Latch state machine (ARMED -> LATCHED -> CLEARED).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from observability.events import (
    LatchTrigger,
    ClearReason,
    build_latch_set_event,
    build_latch_clear_event,
    build_latch_active_heartbeat,
)
from observability.incidents import IncidentBuilder

logger = logging.getLogger(__name__)


class LatchStateMachine:
    """
    Edge-triggered Latch State Machine.
    Guarantees:
      - Exactly ONE PROTECTION_LATCH_SET emitted per latch episode (no log spam).
      - Rejection of UNKNOWN trigger at runtime (Change Request #1).
      - Never-raise incident packaging (Change Request #2).
      - Periodic heartbeat while in LATCHED state.
    """

    def __init__(self, run_id: Optional[str] = None, incident_builder: Optional[IncidentBuilder] = None) -> None:
        self.run_id = run_id or self._generate_run_id()
        self.incident_builder = incident_builder or IncidentBuilder()
        self._lock = threading.RLock()

        self._state: str = "ARMED"  # "ARMED" or "LATCHED"
        self._latched_at: Optional[float] = None
        self._trigger_source: Optional[LatchTrigger] = None
        self._last_heartbeat_ts: float = 0.0
        self._latest_incident_dir: Optional[str] = None

    @staticmethod
    def _generate_run_id() -> str:
        import uuid
        return uuid.uuid4().hex[:12]

    @property
    def is_latched(self) -> bool:
        with self._lock:
            return self._state == "LATCHED"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def trigger_source(self) -> Optional[LatchTrigger]:
        with self._lock:
            return self._trigger_source

    def latch(
        self,
        trigger: LatchTrigger,
        *,
        bridge: Optional[object] = None,
        symbol: Optional[str] = None,
        parent_order_id: Optional[str] = None,
        stop_order_id: Optional[str] = None,
        target_order_id: Optional[str] = None,
        filled_qty: Optional[float] = None,
        reasons: Optional[list] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Transition ARMED -> LATCHED (Edge-triggered).
        Returns True if this call triggered the state transition; False if already latched.
        """
        # Change Request #1: Runtime guard against UNKNOWN or invalid triggers
        if not isinstance(trigger, LatchTrigger) or trigger == LatchTrigger.UNKNOWN:
            err_msg = f"[LATCH STATE MACHINE] Invalid or UNKNOWN latch trigger: {trigger}. UNKNOWN is strictly forbidden at runtime."
            logger.critical(err_msg)
            raise ValueError(err_msg)

        with self._lock:
            if self._state == "LATCHED":
                # Edge-triggered: repeat detections are no-ops
                logger.debug("[LATCH STATE MACHINE] Latch already active for %s; ignoring duplicate trigger %s", self._trigger_source, trigger)
                return False

            self._state = "LATCHED"
            self._latched_at = time.time()
            self._trigger_source = trigger
            self._last_heartbeat_ts = self._latched_at

            # Gather account context safely
            equity = None
            sod_equity = None
            drawdown_pct = None
            working_orders = 0
            positions = 0
            if bridge is not None:
                try:
                    if hasattr(bridge, "get_account"):
                        acc = bridge.get_account()
                        equity = float(getattr(acc, "equity", 0.0) or 0.0)
                    elif hasattr(bridge, "get_account_balance"):
                        equity = bridge.get_account_balance()
                    
                    sod_equity = getattr(bridge, "_start_of_day_equity", equity)
                    if equity and sod_equity and sod_equity > 0:
                        drawdown_pct = round((equity - sod_equity) / sod_equity * 100.0, 3)

                    if hasattr(bridge, "get_working_orders"):
                        working_orders = len(bridge.get_working_orders() or [])
                    if hasattr(bridge, "get_positions"):
                        positions = len(bridge.get_positions() or [])
                except Exception as ctx_exc:
                    logger.warning("[LATCH STATE MACHINE] Context gather partial error: %s", ctx_exc)

            latch_event = build_latch_set_event(
                run_id=self.run_id,
                trigger_source=trigger,
                symbol=symbol,
                parent_order_id=parent_order_id,
                stop_order_id=stop_order_id,
                target_order_id=target_order_id,
                filled_qty=filled_qty,
                equity=equity,
                sod_equity=sod_equity,
                drawdown_pct=drawdown_pct,
                working_orders=working_orders,
                positions=positions,
                reasons=reasons or ([str(trigger.value)]),
            )

            # Dump 5-file Incident Bundle (Never-Raise contract guaranteed by IncidentBuilder)
            incident_path = self.incident_builder.dump_bundle(latch_event=latch_event, bridge=bridge)
            if incident_path:
                self._latest_incident_dir = str(incident_path)
                latch_event["incident_dir"] = str(incident_path)

            logger.critical(
                "[PROTECTION_LATCH_SET] Latch activated: trigger=%s symbol=%s parent=%s incident_dir=%s",
                trigger.value,
                symbol,
                parent_order_id,
                self._latest_incident_dir,
            )
            return True

    def clear(
        self,
        reason: ClearReason,
        *,
        reconcile: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Transition LATCHED -> ARMED (Clear Latch).
        """
        with self._lock:
            if self._state != "LATCHED":
                return False

            now = time.time()
            latch_age = (now - self._latched_at) if self._latched_at else 0.0
            orig_trigger = self._trigger_source.value if self._trigger_source else "UNKNOWN"

            clear_event = build_latch_clear_event(
                run_id=self.run_id,
                reason=reason,
                original_trigger_source=orig_trigger,
                latch_age_seconds=latch_age,
                reconcile=reconcile or {},
            )

            self._state = "ARMED"
            self._latched_at = None
            self._trigger_source = None
            self._last_heartbeat_ts = 0.0

            logger.warning(
                "[PROTECTION_LATCH_CLEAR] Latch cleared: reason=%s orig_trigger=%s age=%.1fs",
                reason.value,
                orig_trigger,
                latch_age,
            )
            return True

    def heartbeat_if_due(self, interval_seconds: float = 300.0, active_halts: Optional[list] = None) -> bool:
        """Emit PROTECTION_LATCH_ACTIVE heartbeat if interval has elapsed."""
        with self._lock:
            if self._state != "LATCHED":
                return False
            now = time.time()
            if (now - self._last_heartbeat_ts) >= interval_seconds:
                self._last_heartbeat_ts = now
                elapsed = (now - self._latched_at) if self._latched_at else 0.0
                hb_event = build_latch_active_heartbeat(
                    run_id=self.run_id,
                    elapsed_seconds=elapsed,
                    original_trigger_source=self._trigger_source.value if self._trigger_source else "UNKNOWN",
                    active_halts=active_halts or [],
                )
                logger.error(
                    "[PROTECTION_LATCH_ACTIVE] Latch active for %.0fs: orig_trigger=%s halts=%s",
                    elapsed,
                    hb_event["original_trigger_source"],
                    hb_event["active_halts"],
                )
                return True
            return False

    def export_state_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "latched_at": self._latched_at,
                "trigger_source": self._trigger_source.value if self._trigger_source else None,
                "incident_dir": self._latest_incident_dir,
                "run_id": self.run_id,
            }

    def import_state_dict(self, data: Optional[Dict[str, Any]]) -> None:
        if not data or not isinstance(data, dict):
            return
        with self._lock:
            state_val = data.get("state")
            if state_val == "LATCHED":
                self._state = "LATCHED"
                self._latched_at = data.get("latched_at") or time.time()
                trig_str = data.get("trigger_source")
                try:
                    self._trigger_source = LatchTrigger(trig_str) if trig_str else LatchTrigger.ORDER_REJECT
                except Exception:
                    self._trigger_source = LatchTrigger.ORDER_REJECT
                self._latest_incident_dir = data.get("incident_dir")
                self._last_heartbeat_ts = time.time()
                logger.warning(
                    "[LATCH STATE MACHINE] Restored active latch state: trigger=%s incident_dir=%s",
                    self._trigger_source.value,
                    self._latest_incident_dir,
                )


_STATE_MACHINE: Optional[LatchStateMachine] = None
_SM_LOCK = threading.Lock()


def get_latch_state_machine(run_id: Optional[str] = None) -> LatchStateMachine:
    global _STATE_MACHINE
    if _STATE_MACHINE is None:
        with _SM_LOCK:
            if _STATE_MACHINE is None:
                _STATE_MACHINE = LatchStateMachine(run_id=run_id)
    return _STATE_MACHINE
