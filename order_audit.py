"""
order_audit.py - Structured (JSONL) audit log of the order lifecycle.

Phase 0 of the live-trading readiness plan (see
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md). This module is intentionally
tiny, dependency-light, and SIDE-EFFECT-ONLY: it appends one JSON object per
order-lifecycle event to logs/order_audit.jsonl. It NEVER raises into the
caller, so adding `order_audit.log_event(...)` calls inside the order path
cannot change trading behavior - every call is wrapped so a logging failure is
swallowed (and best-effort echoed to the standard logger).

Later phases (fill verification, server-side stops) assert against this log to
confirm the order path actually reached `filled` / `stop_confirmed`, instead of
stopping at `accepted`.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# One line per event. Lives next to the rotating text log but stays
# machine-readable (one JSON object per line) for later assertions.
AUDIT_FILE = Path(config.LOG_DIR) / "order_audit.jsonl"

# Canonical lifecycle stages, ordered. A healthy live entry should progress all
# the way to `stop_confirmed`; today (Phase 0) the path stops at `accepted`.
STAGE_SUBMIT = "submit"            # order handed to placeOrder()
STAGE_ACK = "ack"                  # broker acknowledged (PreSubmitted/Submitted)
STAGE_FILLED = "filled"            # actually filled (Phase 2 will emit this)
STAGE_PARTIAL = "partial_fill"     # partially filled (Phase 2)
STAGE_REJECTED = "rejected"        # broker rejected/cancelled
STAGE_STOP_PLACED = "stop_placed"  # protective child placed (Phase 3)
STAGE_STOP_CONFIRMED = "stop_confirmed"  # protective child verified live (Phase 2/3)
STAGE_TRADE_RECORDED = "trade_recorded"  # risk_state.record_trade() called
STAGE_FLATTEN = "flatten"          # emergency/normal flatten action
STAGE_PROTECT = "protect"          # GTC/OCA protective exit set placed + verified (Phase 3)
STAGE_HALT = "halt"                # new-entry halt: daily-loss / drawdown / unprotected startup (Phase 3)
STAGE_SNAPSHOT = "equity_snapshot" # start-of-day NetLiquidation snapshot (Phase 3)
STAGE_QUOTE = "quote"              # order-price quote validated for data integrity (Phase 4)
STAGE_RECONCILE = "reconcile"      # startup broker-truth reconciliation snapshot (Phase 5A, H18)


def log_event(stage: str, **fields) -> None:
    """Append one audit event. Never raises into the caller.

    Args:
        stage: one of the STAGE_* constants (free-form strings are tolerated).
        **fields: JSON-serializable context (symbol, action, qty, status, ...).
    """
    try:
        record = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "stage": str(stage)}
        for key, value in fields.items():
            record[key] = _safe(value)
        AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except Exception:
        # Audit logging must never break the order path. Best-effort only.
        try:
            logger.debug("order_audit.log_event failed for stage=%s", stage, exc_info=True)
        except Exception:
            pass


def _safe(value):
    """Coerce a value into something json.dumps can handle without raising."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


def read_events(limit: int | None = None) -> list:
    """Read back audit events (most-recent-last). Returns [] if no file yet.

    Read-only convenience for the `live-readiness` self-check and tests.
    """
    try:
        if not AUDIT_FILE.exists():
            return []
        lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
        if limit is not None and limit >= 0:
            lines = lines[-limit:]
        events = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                events.append(json.loads(ln))
            except Exception:
                continue
        return events
    except Exception:
        return []
