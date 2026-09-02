"""
observability/events.py -- Event definitions, taxonomy enums, and schemas.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime, timezone
import hashlib


class LatchTrigger(str, Enum):
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    ORDER_REJECT = "ORDER_REJECT"
    PARTIAL_FILL_MISMATCH = "PARTIAL_FILL_MISMATCH"
    UNPROTECTED_POSITION = "UNPROTECTED_POSITION"
    RECONCILE_MISMATCH = "RECONCILE_MISMATCH"
    DUPLICATE_CLIENT_ORDER_ID = "DUPLICATE_CLIENT_ORDER_ID"
    ORPHAN_EXIT_DETECTED = "ORPHAN_EXIT_DETECTED"
    STREAM_STALE = "STREAM_STALE"
    MANUAL_HALT = "MANUAL_HALT"
    UNKNOWN = "UNKNOWN"


class ClearReason(str, Enum):
    STARTUP_RECONCILIATION_CLEAN = "STARTUP_RECONCILIATION_CLEAN"
    MANUAL_RESET = "MANUAL_RESET"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def build_latch_set_event(
    *,
    run_id: str,
    trigger_source: LatchTrigger,
    symbol: Optional[str] = None,
    parent_order_id: Optional[str] = None,
    stop_order_id: Optional[str] = None,
    target_order_id: Optional[str] = None,
    filled_qty: Optional[float] = None,
    equity: Optional[float] = None,
    sod_equity: Optional[float] = None,
    drawdown_pct: Optional[float] = None,
    working_orders: Optional[int] = None,
    positions: Optional[int] = None,
    regime: Optional[Dict[str, Any]] = None,
    incident_dir: Optional[str] = None,
    git_commit: Optional[str] = None,
    config_hash: Optional[str] = None,
    model_hash: Optional[str] = None,
    reasons: Optional[list] = None,
) -> Dict[str, Any]:
    """Build structured PROTECTION_LATCH_SET event payload."""
    return {
        "event": "PROTECTION_LATCH_SET",
        "ts": now_iso(),
        "run_id": run_id,
        "git_commit": git_commit or "unknown",
        "config_hash": config_hash or "unknown",
        "model_hash": model_hash or "unknown",
        "trigger_source": trigger_source.value if isinstance(trigger_source, LatchTrigger) else str(trigger_source),
        "symbol": symbol,
        "parent_order_id": parent_order_id,
        "stop_order_id": stop_order_id,
        "target_order_id": target_order_id,
        "filled_qty": filled_qty,
        "equity": equity,
        "sod_equity": sod_equity,
        "drawdown_pct": drawdown_pct,
        "working_orders": working_orders,
        "positions": positions,
        "regime": regime or {},
        "reasons": reasons or [],
        "allow_new_entries": {"before": True, "after": False},
        "incident_dir": incident_dir,
    }


def build_latch_clear_event(
    *,
    run_id: str,
    reason: ClearReason,
    original_trigger_source: Optional[str] = None,
    latch_age_seconds: Optional[float] = None,
    reconcile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build structured PROTECTION_LATCH_CLEAR event payload."""
    return {
        "event": "PROTECTION_LATCH_CLEAR",
        "ts": now_iso(),
        "run_id": run_id,
        "reason": reason.value if isinstance(reason, ClearReason) else str(reason),
        "original_trigger_source": original_trigger_source,
        "latch_age_seconds": round(latch_age_seconds, 3) if latch_age_seconds is not None else None,
        "reconcile": reconcile or {},
        "allow_new_entries": {"before": False, "after": True},
    }


def build_latch_active_heartbeat(
    *,
    run_id: str,
    elapsed_seconds: float,
    original_trigger_source: Optional[str] = None,
    active_halts: Optional[list] = None,
) -> Dict[str, Any]:
    """Build structured PROTECTION_LATCH_ACTIVE heartbeat payload."""
    return {
        "event": "PROTECTION_LATCH_ACTIVE",
        "ts": now_iso(),
        "run_id": run_id,
        "elapsed_seconds": round(elapsed_seconds, 1),
        "original_trigger_source": original_trigger_source,
        "active_halts": active_halts or [],
    }
