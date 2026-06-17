"""
alerts.py - Phase 5B-4: SAFE, offline-testable alerting layer for the ONE-SHOT bot.

See reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md (task 5.5). This module turns the
safety EVENTS the bot already detects -- a mid-run disconnect / reconnect failure,
an unprotected long at startup or shutdown, a duplicate orderRef, an orphan exit
order, a daily-loss kill-switch trip, an order rejection, a partial fill, a
protective-child failure / emergency flatten, a blocked scheduled run -- into
operator ALERTS.

It is deliberately INERT by default and CANNOT affect trading:

  * DISABLED by default. ``ALERTS_ENABLED`` ships False, so ``emit()`` is a no-op
    (no logging, no external action) until an operator turns it on.
  * LOG-ONLY even when enabled. ``ALERTS_LOG_ONLY`` ships True, so an alert goes
    ONLY to the standard logger + the order_audit trail; NOTHING leaves the box.
  * It NEVER places, cancels, or modifies an order (it imports no ib_insync and
    references no order verb).
  * It NEVER enables live trading and never relaxes a gate (it never writes config).
  * It NEVER blocks. ``emit()`` is synchronous, loop-free, and wrapped so it can
    never raise into the trading path -- an alert failure can never stop an
    emergency flatten or a protective repair.
  * NO daemon / NO loop.

Real email / SMS / Telegram / webhook delivery is intentionally NOT implemented in
this phase (req 10). The external-channel registry is EMPTY and the ``send_*``
functions are inert disabled stubs: they send nothing off-box and exist only to
document the shape a later, deliberately-configured phase would fill in.

Like data_integrity.py / reconnect_watchdog.py the module is PURE and offline:
config is read at call time (so tests can patch it), severity filtering and payload
construction are deterministic, and DELIVERED alerts are captured in an in-memory
ring buffer (``recent_alerts()``) so the whole thing is unit-testable without a
real mail server, a real socket, or real time.
"""
from __future__ import annotations

import datetime as _dt
import logging
from collections import deque
from typing import List, Optional

import config
import order_audit

logger = logging.getLogger(__name__)

# ── Severity levels (ordered low -> high) ─────────────────────────────────────
SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"

_SEVERITY_RANK = {SEVERITY_INFO: 10, SEVERITY_WARNING: 20, SEVERITY_CRITICAL: 30}

_LOG_LEVEL = {
    SEVERITY_INFO: logging.INFO,
    SEVERITY_WARNING: logging.WARNING,
    SEVERITY_CRITICAL: logging.CRITICAL,
}

# ── Event types (req 7) ───────────────────────────────────────────────────────
EVENT_DISCONNECT = "disconnect"                            # mid-run IBKR drop
EVENT_RECONNECT_FAILURE = "reconnect_failure"              # bounded retry gave up
EVENT_RECONCILE_UNPROTECTED_LONG = "reconcile_unprotected_long"  # startup recon
EVENT_ORPHAN_EXIT_ORDER = "orphan_exit_order"             # resting SELL, no long
EVENT_DUPLICATE_ORDER_REF = "duplicate_order_ref"         # idempotency violation
EVENT_DAILY_LOSS_KILLSWITCH = "daily_loss_killswitch"     # loss limit tripped
EVENT_ORDER_REJECTED = "order_rejected"                   # broker rejected order
EVENT_PARTIAL_FILL = "partial_fill"                       # order only partly filled
EVENT_PROTECTIVE_CHILD_FAILURE = "protective_child_failure"  # stop not confirmed
EVENT_EMERGENCY_FLATTEN = "emergency_flatten"             # forced flatten executed
EVENT_SCHEDULER_BLOCKED = "scheduler_blocked"             # scheduled run blocked
EVENT_SHUTDOWN_WARNING = "shutdown_warning"               # graceful-shutdown warn
EVENT_SHUTDOWN_UNPROTECTED_LONG = "shutdown_unprotected_long"  # naked long at close

# Default severity per event (callers may override per-call). Routine, expected
# events (a market-closed scheduled-run block) default to INFO so the default
# ALERT_MIN_SEVERITY=warning floor filters them out instead of spamming; events
# that mean "a long may be unprotected" or "the link is down" are CRITICAL.
_DEFAULT_SEVERITY = {
    EVENT_DISCONNECT: SEVERITY_CRITICAL,
    EVENT_RECONNECT_FAILURE: SEVERITY_CRITICAL,
    EVENT_RECONCILE_UNPROTECTED_LONG: SEVERITY_CRITICAL,
    EVENT_ORPHAN_EXIT_ORDER: SEVERITY_CRITICAL,
    EVENT_DUPLICATE_ORDER_REF: SEVERITY_WARNING,
    EVENT_DAILY_LOSS_KILLSWITCH: SEVERITY_CRITICAL,
    EVENT_ORDER_REJECTED: SEVERITY_WARNING,
    EVENT_PARTIAL_FILL: SEVERITY_WARNING,
    EVENT_PROTECTIVE_CHILD_FAILURE: SEVERITY_CRITICAL,
    EVENT_EMERGENCY_FLATTEN: SEVERITY_CRITICAL,
    EVENT_SCHEDULER_BLOCKED: SEVERITY_INFO,
    EVENT_SHUTDOWN_WARNING: SEVERITY_WARNING,
    EVENT_SHUTDOWN_UNPROTECTED_LONG: SEVERITY_CRITICAL,
}

# ── In-memory capture of DELIVERED alerts (offline test hook) ─────────────────
# Bounded so a long-lived process cannot grow it without limit. Captures ONLY
# alerts that actually passed the enabled + severity gates and were delivered, so
# a test can assert what WOULD reach an operator (and that disabled => nothing).
_RECENT: deque = deque(maxlen=500)


def recent_alerts() -> list:
    """DELIVERED alerts captured this process (most-recent-last). Test/ops hook."""
    return list(_RECENT)


def reset() -> None:
    """Clear the in-memory capture. Tests call this in setUp for isolation."""
    _RECENT.clear()


# ── Config resolution (read at call time so tests can patch config) ──────────
def alerts_enabled() -> bool:
    """Master switch. Ships False => emit() is a no-op (no external action)."""
    return bool(getattr(config, "ALERTS_ENABLED", False))


def alerts_log_only() -> bool:
    """Log-only mode. Ships True => alerts go ONLY to the logger + audit trail;
    no external channel is ever consulted."""
    return bool(getattr(config, "ALERTS_LOG_ONLY", True))


def min_severity() -> str:
    """Configured minimum severity to deliver. Unrecognised => 'warning'."""
    raw = str(getattr(config, "ALERT_MIN_SEVERITY", SEVERITY_WARNING) or "").strip().lower()
    return raw if raw in _SEVERITY_RANK else SEVERITY_WARNING


def _normalize_severity(severity: Optional[str], event: str) -> str:
    if severity:
        s = str(severity).strip().lower()
        if s in _SEVERITY_RANK:
            return s
    return _DEFAULT_SEVERITY.get(event, SEVERITY_WARNING)


def _passes_severity(severity: str, threshold: Optional[str] = None) -> bool:
    if threshold is None:
        threshold = min_severity()
    sev = _SEVERITY_RANK.get(severity, _SEVERITY_RANK[SEVERITY_INFO])
    thr = _SEVERITY_RANK.get(threshold, _SEVERITY_RANK[SEVERITY_WARNING])
    return sev >= thr


# ── JSON-safe coercion (alerts may be persisted / shipped later) ─────────────
def _json_safe(value):
    """Coerce a value into something json.dumps can handle without raising.
    Mirrors order_audit._safe so an audit-trail write can never fail on a payload."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


# ── External delivery is NOT implemented in Phase 5B-4 (req 10) ───────────────
# The registry is EMPTY and the channel functions are inert disabled stubs: they
# send NOTHING off-box and always return False ("not configured"). They exist only
# to document the shape a later, deliberately-configured phase would implement.
# ``_deliver`` only ever runs the local LOG channel this phase.
EXTERNAL_CHANNELS: List = []  # intentionally empty in Phase 5B-4


def _stub_channel(name: str) -> bool:
    """Disabled external-channel stub. Sends NOTHING. Always returns False."""
    logger.debug("alerts: external channel %r is a disabled Phase-5B-4 stub "
                 "-> nothing sent", name)
    return False


def send_email(payload: dict) -> bool:     # pragma: no cover - inert disabled stub
    return _stub_channel("email")


def send_sms(payload: dict) -> bool:       # pragma: no cover - inert disabled stub
    return _stub_channel("sms")


def send_telegram(payload: dict) -> bool:  # pragma: no cover - inert disabled stub
    return _stub_channel("telegram")


def send_webhook(payload: dict) -> bool:   # pragma: no cover - inert disabled stub
    return _stub_channel("webhook")


# ── Delivery (LOG-ONLY this phase) ────────────────────────────────────────────
def _deliver_log(payload: dict) -> None:
    """The ONLY live delivery channel in Phase 5B-4: the local logger plus the
    order_audit trail. Sends nothing off-box. Best-effort; never raises."""
    level = _LOG_LEVEL.get(payload.get("severity"), logging.WARNING)
    logger.log(level, "ALERT [%s] %s | %s | %s",
               payload.get("severity"), payload.get("event"),
               payload.get("message") or "(no message)", payload.get("context"))
    # Persist to the same JSONL trail the order path / live-readiness already use
    # (order_audit.log_event swallows its own failures, so this never raises).
    order_audit.log_event(order_audit.STAGE_ALERT, **payload)


# ── Public API ────────────────────────────────────────────────────────────────
def emit(event: str, severity: Optional[str] = None, message: str = "",
         **context) -> Optional[dict]:
    """Emit one operator alert. The single entry point for every alert.

    SAFETY CONTRACT (all enforced + unit-tested in test_alerts.py):
      * NEVER raises into the caller (the whole body is wrapped); an alert failure
        can never stop an emergency flatten or a protective repair.
      * NEVER places, cancels, or modifies an order (no ib_insync here).
      * NEVER enables live trading / relaxes a gate (never writes config).
      * NEVER blocks (synchronous, loop-free, no sleeps).

    Behaviour:
      * ``ALERTS_ENABLED`` False  -> no-op, returns None (no logging, no delivery).
      * severity below ``ALERT_MIN_SEVERITY`` -> suppressed, returns None.
      * otherwise -> delivered LOG-ONLY (logger + order_audit trail) and captured
        in ``recent_alerts()``; returns the JSON-safe payload that was delivered.

    Args:
        event: one of the EVENT_* constants (free-form strings are tolerated).
        severity: optional override; defaults to the event's default severity.
        message: optional human-readable summary.
        **context: JSON-serializable context (symbols, reason, qty, ...).
    """
    try:
        sev = _normalize_severity(severity, event)

        # 1) Master switch: disabled => no action of any kind.
        if not alerts_enabled():
            return None

        # 2) Severity floor.
        if not _passes_severity(sev):
            return None

        payload = {
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "event": str(event),
            "severity": sev,
            "message": str(message or ""),
            "context": _json_safe(context),
            "channels": ["log"],          # the only live channel this phase
            "log_only": alerts_log_only(),
        }

        # 3) Deliver. LOG-ONLY by default; the external registry is empty and the
        # stubs are inert, so nothing ever leaves the box this phase. Even with
        # log_only off, the empty EXTERNAL_CHANNELS loop is a structural no-op.
        _deliver_log(payload)
        if not alerts_log_only():
            for channel in EXTERNAL_CHANNELS:   # empty in Phase 5B-4
                try:
                    channel(payload)
                except Exception:
                    logger.debug("alerts: external channel raised (ignored)", exc_info=True)

        _RECENT.append(payload)
        return payload
    except Exception:
        # An alert must NEVER break the trading path. Swallow everything.
        try:
            logger.debug("alerts.emit failed for event=%s", event, exc_info=True)
        except Exception:
            pass
        return None
