"""
test_reconnect_watchdog.py - Phase 5B-2 reconnect watchdog / connection resilience.

Fully offline and deterministic: NO live IBKR, NO network, NO real sleep, NO wall
clock. A tiny FakeIB simulates connect()/disconnect()/disconnectedEvent and
RECORDS every backoff delay instead of sleeping, so the real IBKRBridge.connect()
retry path and the disconnect handler run end to end in microseconds.

These back the Phase-5B-2 contract from reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md
(task 5.3) — a BOUNDED reconnect for the ONE-SHOT bot (no daemon, no forever
loop) that NEVER bypasses a safety gate and NEVER places an order:

  * connect succeeds first try            -> 1 attempt, no backoff, healthy
  * connect retries then succeeds         -> bounded backoff, then healthy
  * connect gives up after max attempts   -> bounded tries, then fail-closed
  * paper-port violation blocks BEFORE any retry (lock is never bypassed)
  * reconnect disabled                    -> exactly one attempt, no retry
  * backoff is bounded + testable without sleeping real time
  * an unexpected disconnect marks the link unhealthy -> new entries blocked
  * the watchdog NEVER places an order
  * live-readiness is STILL NOT READY (Phase 6 gates remain False) and Phase 5B-2
    adds NO new live-readiness capability flag

Run with:
    python -m unittest test_reconnect_watchdog -v
    python test_reconnect_watchdog.py
"""
import asyncio
import contextlib
import io
import unittest
from unittest import mock


# ib_insync touches the event loop at import time; prepare it first (as main.py
# and the other ibkr tests do).
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import config            # noqa: E402
import order_audit       # noqa: E402
import ibkr_bridge       # noqa: E402
import reconnect_watchdog as rw  # noqa: E402


# ── Minimal fake ib_insync surface ───────────────────────────────────────────
class _FakeEvent:
    """Stand-in for an ib_insync Event: supports ``+=`` and manual ``emit()``."""

    def __init__(self):
        self._handlers = []

    def __iadd__(self, handler):
        self._handlers.append(handler)
        return self

    def emit(self, *args, **kwargs):
        for h in list(self._handlers):
            h(*args, **kwargs)


class FakeIB:
    """Minimal ib_insync.IB replacement for the connection path.

    ``fail_times`` initial connect() calls raise (simulate a refused/transient
    TWS); after that connect() succeeds. ``sleep`` RECORDS the requested delay
    instead of sleeping, so backoff is asserted without real time. ``placeOrder``
    raises if ever called — the watchdog must never reach it.
    """

    def __init__(self, *, fail_times=0):
        self.fail_times = fail_times
        self.connect_calls = 0
        self.slept = []            # recorded backoff delays (NO real sleep)
        self.place_calls = []      # any order placement (MUST stay empty)
        self.RequestTimeout = None
        self.market_data_type = None
        self.last_timeout = None
        self._connected = False
        self.disconnect_calls = 0
        self.disconnectedEvent = _FakeEvent()

    def connect(self, host=None, port=None, clientId=None, timeout=None):
        self.connect_calls += 1
        self.last_timeout = timeout
        if self.connect_calls <= self.fail_times:
            raise ConnectionError(f"refused (attempt {self.connect_calls})")
        self._connected = True

    def reqMarketDataType(self, t):
        self.market_data_type = t

    def managedAccounts(self):
        return ["DU1234567"]

    def isConnected(self):
        return self._connected

    def disconnect(self):
        self.disconnect_calls += 1
        self._connected = False
        # Real IB fires disconnectedEvent on teardown; mimic it so the
        # intentional-vs-unexpected distinction is exercised.
        self.disconnectedEvent.emit()

    def sleep(self, secs=0):
        self.slept.append(float(secs))

    def placeOrder(self, *args, **kwargs):  # pragma: no cover - must never run
        self.place_calls.append((args, kwargs))
        raise AssertionError("reconnect watchdog must never place orders")


def _make_bridge(fail_times=0):
    br = ibkr_bridge.IBKRBridge()
    br.ib = FakeIB(fail_times=fail_times)
    return br


# ── Base: paper-safe config + silenced audit log ─────────────────────────────
class _WatchdogBase(unittest.TestCase):
    # Small, fast, fully deterministic knobs unless a test overrides them.
    CONFIG = {
        "REQUIRE_PAPER_PORT": True,
        "IBKR_PORT": 7497,
        "PAPER_IBKR_PORT": 7497,
        "IBKR_RECONNECT_ENABLED": True,
        "IBKR_RECONNECT_MAX_ATTEMPTS": 3,
        "IBKR_RECONNECT_BASE_DELAY_SECONDS": 0.5,
        "IBKR_RECONNECT_MAX_DELAY_SECONDS": 2.0,
        "IBKR_REQUEST_TIMEOUT_SECONDS": 5.0,
    }

    def setUp(self):
        for name, val in self.CONFIG.items():
            mock.patch.object(config, name, val).start()
            self.addCleanup(mock.patch.stopall)
        # Never touch the real order_audit.jsonl during tests.
        mock.patch.object(order_audit, "log_event").start()


# ── 1. connect succeeds first try ────────────────────────────────────────────
class TestConnectFirstTry(_WatchdogBase):
    def test_first_try_no_backoff_and_healthy(self):
        br = _make_bridge(fail_times=0)
        self.assertTrue(br.connect())
        self.assertEqual(br.ib.connect_calls, 1)
        self.assertEqual(br.ib.slept, [], "no backoff sleep on a first-try success")
        self.assertTrue(br._conn_health.is_healthy)
        self.assertTrue(br._connection_healthy())
        # The request timeout was applied to the socket + connect() call.
        self.assertEqual(br.ib.RequestTimeout, 5.0)
        self.assertEqual(br.ib.last_timeout, 5.0)
        self.assertEqual(br.ib.market_data_type, config.IBKR_MARKET_DATA_TYPE)


# ── 2. connect retries then succeeds ─────────────────────────────────────────
class TestConnectRetriesThenSucceeds(_WatchdogBase):
    def test_two_failures_then_success_with_bounded_backoff(self):
        br = _make_bridge(fail_times=2)
        self.assertTrue(br.connect())
        self.assertEqual(br.ib.connect_calls, 3)
        # Backoff sleeps occur only BETWEEN tries: 2 failures -> 2 sleeps.
        self.assertEqual(br.ib.slept, [0.5, 1.0])
        self.assertTrue(br._conn_health.is_healthy)


# ── 3. connect gives up after max attempts ───────────────────────────────────
class TestConnectGivesUp(_WatchdogBase):
    def test_gives_up_after_max_attempts_fail_closed(self):
        br = _make_bridge(fail_times=99)  # always fails
        self.assertFalse(br.connect())
        self.assertEqual(br.ib.connect_calls, 3, "bounded at IBKR_RECONNECT_MAX_ATTEMPTS")
        self.assertEqual(br.ib.slept, [0.5, 1.0], "no sleep after the final failed try")
        self.assertFalse(br._conn_health.is_healthy)
        self.assertEqual(br._conn_health.reason, rw.REASON_GAVE_UP)


# ── 4. paper-port violation blocks BEFORE any retry ──────────────────────────
class TestPaperPortBlocksBeforeRetry(_WatchdogBase):
    def test_non_paper_port_refused_without_connecting_or_retrying(self):
        with mock.patch.object(config, "IBKR_PORT", 7496):  # live-ish port
            br = _make_bridge(fail_times=0)  # would succeed if ever reached
            self.assertFalse(br.connect())
            self.assertEqual(br.ib.connect_calls, 0, "lock must block before any connect")
            self.assertEqual(br.ib.slept, [], "no backoff for a paper-port violation")
            self.assertFalse(br._conn_health.is_healthy)
            self.assertEqual(br._conn_health.reason, rw.REASON_PAPER_PORT)


# ── 5. reconnect disabled -> no retry ────────────────────────────────────────
class TestReconnectDisabled(_WatchdogBase):
    def test_disabled_means_single_attempt_no_backoff(self):
        with mock.patch.object(config, "IBKR_RECONNECT_ENABLED", False):
            br = _make_bridge(fail_times=99)
            self.assertFalse(br.connect())
            self.assertEqual(br.ib.connect_calls, 1, "exactly one try when reconnect disabled")
            self.assertEqual(br.ib.slept, [])

    def test_disabled_still_connects_on_a_good_link(self):
        with mock.patch.object(config, "IBKR_RECONNECT_ENABLED", False):
            br = _make_bridge(fail_times=0)
            self.assertTrue(br.connect())
            self.assertEqual(br.ib.connect_calls, 1)
            self.assertTrue(br._conn_health.is_healthy)


# ── 6. backoff is bounded + testable without real sleep ──────────────────────
class TestBackoffBounded(_WatchdogBase):
    def test_backoff_delay_is_capped(self):
        # base=1, cap=4 -> 1, 2, 4, 4, 4 ... every value <= cap, monotonic to cap.
        delays = [rw.backoff_delay(i, base=1.0, cap=4.0) for i in range(6)]
        self.assertEqual(delays, [1.0, 2.0, 4.0, 4.0, 4.0, 4.0])
        self.assertTrue(all(d <= 4.0 for d in delays))

    def test_backoff_schedule_length_and_bound(self):
        # N attempts -> N-1 inter-attempt delays, each bounded by the cap.
        sched = rw.backoff_schedule(attempts=5, base=1.0, cap=4.0)
        self.assertEqual(len(sched), 4)
        self.assertEqual(sched, [1.0, 2.0, 4.0, 4.0])

    def test_zero_base_means_zero_delays(self):
        self.assertEqual(rw.backoff_schedule(attempts=4, base=0.0, cap=10.0), [0.0, 0.0, 0.0])

    def test_attempt_connect_uses_injected_sleep_only(self):
        # A recorder sleep proves no real time.sleep is used; delays match schedule.
        recorded = []
        calls = {"n": 0}

        def connect_once():
            calls["n"] += 1
            return False  # always fail -> exhaust the bounded retries

        result = rw.attempt_connect(
            connect_once, enabled=True, attempts=4, base=0.5, cap=2.0,
            sleep=recorded.append,
        )
        self.assertFalse(result.connected)
        self.assertTrue(result.gave_up)
        self.assertEqual(result.attempts, 4)
        self.assertEqual(calls["n"], 4)
        self.assertEqual(recorded, [0.5, 1.0, 2.0])  # bounded by cap=2.0
        self.assertEqual(result.delays, recorded)

    def test_attempt_connect_stops_early_on_success(self):
        calls = {"n": 0}

        def connect_once():
            calls["n"] += 1
            return calls["n"] >= 2  # fail once, then succeed

        recorded = []
        result = rw.attempt_connect(
            connect_once, enabled=True, attempts=5, base=0.5, cap=2.0,
            sleep=recorded.append,
        )
        self.assertTrue(result.connected)
        self.assertFalse(result.gave_up)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(recorded, [0.5])


# ── 7. disconnect handler marks unhealthy -> new entries blocked ─────────────
class TestDisconnectHandler(_WatchdogBase):
    def test_unexpected_disconnect_marks_unhealthy_and_blocks_entries(self):
        br = _make_bridge(fail_times=0)
        self.assertTrue(br.connect())
        self.assertTrue(br._connection_healthy())

        # Simulate an UNEXPECTED mid-run drop (not via bridge.disconnect()).
        br.ib.disconnectedEvent.emit()

        self.assertFalse(br._conn_health.is_healthy)
        self.assertEqual(br._conn_health.disconnect_count, 1)
        blocked, reason = br._new_entries_blocked("AAPL")
        self.assertTrue(blocked)
        self.assertEqual(reason, "connection_unhealthy")

    def test_intentional_disconnect_is_a_clean_shutdown(self):
        br = _make_bridge(fail_times=0)
        self.assertTrue(br.connect())
        # Our own teardown must NOT be treated as an unhealthy mid-run drop.
        br.disconnect()
        self.assertTrue(br._conn_health.is_healthy)
        self.assertEqual(br._conn_health.disconnect_count, 0)

    def test_health_default_does_not_block_a_connected_run(self):
        # A bridge that never disconnected stays healthy -> the gate does not add
        # a spurious block (default-healthy invariant the existing tests rely on).
        br = _make_bridge(fail_times=0)
        self.assertTrue(br._connection_healthy())
        self.assertTrue(br.connect())
        self.assertTrue(br._connection_healthy())


# ── 8. the watchdog NEVER places an order ────────────────────────────────────
class TestWatchdogNeverPlacesOrders(_WatchdogBase):
    def test_no_orders_across_retry_and_disconnect(self):
        br = _make_bridge(fail_times=2)   # exercise the retry path
        self.assertTrue(br.connect())
        br.ib.disconnectedEvent.emit()    # exercise the disconnect handler
        br.disconnect()                   # exercise clean teardown
        self.assertEqual(br.ib.place_calls, [], "watchdog placed an order — forbidden")

    def test_module_has_no_order_placement_symbols(self):
        import inspect
        src = inspect.getsource(rw)
        for forbidden in ("placeOrder", "MarketOrder", "LimitOrder", "qualifyContracts"):
            self.assertNotIn(forbidden, src,
                             f"reconnect_watchdog must not reference {forbidden}")


# ── 9. live-readiness still NOT READY + no new capability flag ────────────────
class TestStillNotReady(_WatchdogBase):
    # The complete, intentional set of live-readiness capability flags. Phase
    # 5B-2 must NOT add one (req 14), so this set is exhaustive.
    EXPECTED_CAPABILITY_FLAGS = {
        "SUPPORTS_FILL_VERIFICATION",
        "SUPPORTS_PARTIAL_FILL_HANDLING",
        "SUPPORTS_PROTECTIVE_CHILD_VERIFY",
        "SUPPORTS_SERVER_SIDE_GTC_STOP",
        "SUPPORTS_DAILY_LOSS_KILLSWITCH",
        "SUPPORTS_REALTIME_DATA_GUARD",
        "SUPPORTS_MARKET_HOURS_GATE",
        "SUPPORTS_STARTUP_RECONCILIATION",
        "SUPPORTS_ACCOUNT_TYPE_ASSERTION",
    }

    def test_no_new_capability_flag_added(self):
        present = {a for a in dir(ibkr_bridge) if a.startswith("SUPPORTS_")}
        self.assertEqual(present, self.EXPECTED_CAPABILITY_FLAGS,
                         "Phase 5B-2 must not add a live-readiness capability flag")
        # The Phase-6 gate stays False (fail-closed).
        self.assertFalse(ibkr_bridge.SUPPORTS_ACCOUNT_TYPE_ASSERTION)

    def test_live_readiness_reports_not_ready(self):
        import main
        args = mock.Mock(connect=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main.cmd_live_readiness(args)
        self.assertEqual(rc, 1, "live-readiness must still report NOT READY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
