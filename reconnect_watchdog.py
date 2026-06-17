"""
reconnect_watchdog.py - Phase 5B-2: bounded reconnect / connection-resilience
helper for the ONE-SHOT Path A bot.

See reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md (task 5.3). The Path A bot is
ONE-SHOT: every command does connect -> execute -> disconnect and exits; there is
no long-running loop. This module hardens ONLY that one-shot connection. It does
TWO small things and nothing else:

  1. BOUNDED retry of the INITIAL connect. ``attempt_connect`` calls an injected
     ``connect_once`` callable up to ``IBKR_RECONNECT_MAX_ATTEMPTS`` times with
     exponential backoff, then GIVES UP. It NEVER retries forever, NEVER loops as
     a daemon, and NEVER places, cancels, or modifies an order.
  2. A disconnect handler. ``make_disconnect_handler`` builds an
     ``ib.disconnectedEvent`` callback that, on an UNEXPECTED mid-run drop, marks
     a ``ConnectionHealth`` object UNHEALTHY and audits it. The order gate reads
     that health and FAILS CLOSED (refuses NEW entries). The handler likewise
     never touches an order.

Design rules (all enforced and unit-tested in test_reconnect_watchdog.py):
  * NO infinite loop / NO daemon. Retry is bounded by max-attempts.
  * It NEVER bypasses a safety gate. In particular the PAPER-PORT LOCK is enforced
    by IBKRBridge.connect() BEFORE this helper is ever reached, so a non-paper
    port is refused outright and never retried. Startup reconciliation, the
    data-integrity gate, the market-hours gate, the daily-loss kill-switch, and
    the model gate all run elsewhere and are untouched.
  * It adds NO live-readiness capability flag.

Like data_integrity.py / scheduler_runner.py the compute half is PURE and offline:
``backoff_delay`` / ``backoff_schedule`` are deterministic (no jitter, no clock),
and ``attempt_connect`` takes injected ``connect_once`` / ``sleep`` callables, so
the whole module is testable without a real IBKR, a real socket, or real time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import alerts
import config
import order_audit

logger = logging.getLogger(__name__)

# ── Event / reason codes (recorded on ConnectionHealth + the audit log) ──────
REASON_INIT = "init"
REASON_CONNECTED = "connected"
REASON_DISCONNECTED = "disconnected"
REASON_GAVE_UP = "gave_up"
REASON_PAPER_PORT = "paper_port_violation"


# ── Config resolution (read at call time so tests can patch config) ──────────
def reconnect_enabled() -> bool:
    return bool(getattr(config, "IBKR_RECONNECT_ENABLED", False))


def max_attempts() -> int:
    try:
        n = int(getattr(config, "IBKR_RECONNECT_MAX_ATTEMPTS", 1))
    except (TypeError, ValueError):
        n = 1
    return max(1, n)


def base_delay() -> float:
    return _non_negative_float(getattr(config, "IBKR_RECONNECT_BASE_DELAY_SECONDS", 0.0))


def max_delay() -> float:
    return _non_negative_float(getattr(config, "IBKR_RECONNECT_MAX_DELAY_SECONDS", 0.0))


def request_timeout() -> float:
    return _non_negative_float(getattr(config, "IBKR_REQUEST_TIMEOUT_SECONDS", 0.0))


def _non_negative_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return f if f >= 0.0 else float(default)


# ── Pure bounded backoff ─────────────────────────────────────────────────────
def backoff_delay(attempt_index: int,
                  base: Optional[float] = None,
                  cap: Optional[float] = None) -> float:
    """Bounded exponential backoff for the n-th retry (0-indexed). PURE.

    ``delay = min(cap, base * 2**attempt_index)``, clamped to ``[0, cap]``.
    Deterministic (no jitter, no clock) so tests can assert the exact schedule
    without sleeping real time.
    """
    if base is None:
        base = base_delay()
    if cap is None:
        cap = max_delay()
    base = max(0.0, float(base))
    cap = max(0.0, float(cap))
    n = max(0, int(attempt_index))
    # Guard the exponent so an absurd attempt count cannot overflow; the cap
    # dominates long before this anyway.
    n = min(n, 30)
    raw = base * (2 ** n)
    if cap > 0.0:
        return min(cap, raw)
    return raw


def backoff_schedule(attempts: Optional[int] = None,
                     base: Optional[float] = None,
                     cap: Optional[float] = None) -> List[float]:
    """The delays the watchdog WOULD sleep across ``attempts`` total tries. PURE.

    A run of N attempts sleeps only BETWEEN attempts -> N-1 delays. Used by tests
    to prove the backoff is bounded (every entry <= cap) without real time.
    """
    if attempts is None:
        attempts = max_attempts()
    attempts = max(1, int(attempts))
    return [backoff_delay(i, base, cap) for i in range(attempts - 1)]


# ── Connection health (pure state; NEVER places orders) ──────────────────────
@dataclass
class ConnectionHealth:
    """Tracks whether the IBKR link is currently usable for a one-shot run.

    Pure state holder. It NEVER touches IB and NEVER places/cancels/modifies an
    order; it only records healthy/unhealthy + a reason so the order gate can
    FAIL CLOSED (refuse NEW entries) after a mid-run disconnect.

    Defaults to ``healthy=True`` ("presumed usable until we learn otherwise"):
    a freshly-built bridge that has not connected yet behaves exactly as before
    (the order path's other gates still apply), and only an observed give-up or
    an unexpected disconnect flips it closed. ``connect()`` sets it explicitly
    on success.
    """
    healthy: bool = True
    reason: str = REASON_INIT
    disconnect_count: int = 0

    @property
    def is_healthy(self) -> bool:
        return bool(self.healthy)

    def mark_healthy(self, reason: str = REASON_CONNECTED) -> None:
        self.healthy = True
        self.reason = str(reason)

    def mark_unhealthy(self, reason: str = REASON_DISCONNECTED) -> None:
        self.healthy = False
        self.reason = str(reason)

    def record_disconnect(self, reason: str = REASON_DISCONNECTED) -> None:
        """An UNEXPECTED disconnect was observed: count it and fail closed."""
        self.disconnect_count += 1
        self.mark_unhealthy(reason)


# ── Bounded connect orchestrator ─────────────────────────────────────────────
@dataclass
class ReconnectResult:
    """Outcome of a bounded connect run. Pure data; the bridge maps it onto
    ConnectionHealth + the audit log."""
    connected: bool
    attempts: int
    delays: List[float] = field(default_factory=list)
    gave_up: bool = False
    enabled: bool = True


def attempt_connect(connect_once: Callable[[], bool], *,
                    enabled: Optional[bool] = None,
                    attempts: Optional[int] = None,
                    base: Optional[float] = None,
                    cap: Optional[float] = None,
                    sleep: Optional[Callable[[float], None]] = None) -> ReconnectResult:
    """Call ``connect_once`` with BOUNDED exponential-backoff retry.

    NEVER loops forever (bounded by ``attempts``) and NEVER places orders: it only
    invokes the injected ``connect_once`` and ``sleep``. The caller
    (IBKRBridge.connect) MUST have already enforced the paper-port lock before
    reaching here — this helper does not, and so can never bypass it.

      * ``enabled`` False -> exactly ONE attempt, no retry, no sleep.
      * ``enabled`` True  -> up to ``attempts`` total tries; sleep
        ``backoff_delay()`` between failures; give up (return) after the last try.

    ``connect_once`` is expected to return truthy on success; an exception from it
    is treated as a failed attempt (logged), never propagated.
    """
    if enabled is None:
        enabled = reconnect_enabled()
    if attempts is None:
        attempts = max_attempts()
    if sleep is None:
        import time
        sleep = time.sleep
    attempts = max(1, int(attempts))

    def _try() -> bool:
        try:
            return bool(connect_once())
        except Exception:
            logger.warning("reconnect_watchdog: connect attempt raised", exc_info=True)
            return False

    if not enabled:
        ok = _try()
        return ReconnectResult(connected=ok, attempts=1, delays=[],
                               gave_up=not ok, enabled=False)

    delays: List[float] = []
    for n in range(attempts):
        if _try():
            return ReconnectResult(connected=True, attempts=n + 1, delays=delays,
                                   gave_up=False, enabled=True)
        # Sleep only BETWEEN attempts, never after the final failed try.
        if n < attempts - 1:
            d = backoff_delay(n, base, cap)
            delays.append(d)
            try:
                sleep(d)
            except Exception:
                logger.debug("reconnect_watchdog: sleep failed", exc_info=True)
    return ReconnectResult(connected=False, attempts=attempts, delays=delays,
                           gave_up=True, enabled=True)


# ── Disconnect handler ───────────────────────────────────────────────────────
def make_disconnect_handler(health: ConnectionHealth, *,
                            is_intentional: Optional[Callable[[], bool]] = None,
                            audit: bool = True) -> Callable[..., None]:
    """Build an ``ib.disconnectedEvent`` handler.

    The handler ONLY marks the connection unhealthy + audits; it NEVER places,
    cancels, or modifies any order. An intentional shutdown (caller-provided
    ``is_intentional()`` returns True) is treated as a clean close: no unhealthy
    mark and no disconnect audit. The handler never raises into ib_insync.
    """
    def _on_disconnected(*_args, **_kwargs) -> None:
        try:
            if is_intentional is not None and is_intentional():
                logger.debug("reconnect_watchdog: intentional disconnect (clean shutdown)")
                return
            health.record_disconnect(REASON_DISCONNECTED)
            logger.error(
                "reconnect_watchdog: IBKR disconnected mid-run -> connection "
                "UNHEALTHY, failing closed (no new entries)"
            )
            if audit:
                audit_connection(REASON_DISCONNECTED, healthy=False,
                                 disconnect_count=health.disconnect_count)
            # Phase 5B-4: surface the mid-run drop to the operator. emit() is
            # disabled/log-only by default, never raises, and never touches an
            # order -- it only records/notifies, so it cannot affect the fail-
            # closed posture above.
            alerts.emit(alerts.EVENT_DISCONNECT,
                        message="IBKR disconnected mid-run -> failing closed",
                        reason=REASON_DISCONNECTED,
                        disconnect_count=health.disconnect_count,
                        intentional=False)
        except Exception:
            logger.debug("reconnect_watchdog: disconnect handler error", exc_info=True)
    return _on_disconnected


# ── Audit ────────────────────────────────────────────────────────────────────
def audit_connection(event: str, **fields) -> None:
    """Best-effort audit of a watchdog/connection event (never raises — order_audit
    swallows its own failures). Recorded under STAGE_WATCHDOG."""
    order_audit.log_event(order_audit.STAGE_WATCHDOG, event=str(event), **fields)
