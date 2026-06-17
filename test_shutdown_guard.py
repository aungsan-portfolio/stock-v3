"""
test_shutdown_guard.py - Phase 5B-3 graceful shutdown for the ONE-SHOT bot.

Fully offline and deterministic: NO live IBKR, NO network, NO real sleep, NO wall
clock. A tiny FakeIB simulates positions()/openTrades()/disconnect()/
disconnectedEvent; its placeOrder()/cancelOrder() RAISE so any attempt to place or
cancel an order during shutdown fails the test outright.

These back the Phase-5B-3 contract from
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md (task 5.3) -- a GRACEFUL shutdown of
the one-shot run that leaves the broker SAFE:

  * the shutdown planner / prepare step places NO order and cancels NO order
  * an intentional shutdown is a CLEAN disconnect (the 5B-2 watchdog must NOT mark
    it unhealthy / audit a mid-run drop)
  * every resting protective GTC stop is PRESERVED (orders_to_cancel is always [])
  * an unprotected long is DETECTED
  * shutdown does NOT cancel protective children
  * shutdown does NOT open a new entry
  * repair (if any) goes ONLY through the existing ensure_protective_stops() path,
    is fail-closed on a dropped link, and never guesses avgCost
  * live-readiness is STILL NOT READY (Phase 6 gates remain False) and Phase 5B-3
    adds NO new live-readiness capability flag

Run with:
    python -X utf8 -m unittest test_shutdown_guard -v
    python test_shutdown_guard.py
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
import live_invariants   # noqa: E402
import shutdown_guard as sg  # noqa: E402
import ibkr_bridge       # noqa: E402


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


class FakeContract:
    def __init__(self, symbol):
        self.symbol = symbol


class FakePosition:
    def __init__(self, symbol, qty, avgCost=0.0):
        self.contract = FakeContract(symbol)
        self.position = qty
        self.avgCost = avgCost


class FakeOrder:
    def __init__(self, action, orderType, qty, tif="", orderRef="", parentId=0):
        self.action = action
        self.orderType = orderType
        self.totalQuantity = qty
        self.tif = tif
        self.orderRef = orderRef
        self.parentId = parentId


class FakeOrderStatus:
    def __init__(self, status):
        self.status = status


class FakeTrade:
    def __init__(self, symbol, order, status="Submitted"):
        self.contract = FakeContract(symbol)
        self.order = order
        self.orderStatus = FakeOrderStatus(status)


class FakeIB:
    """Minimal ib_insync.IB replacement for the shutdown path.

    Returns the injected positions/openTrades. ``placeOrder``/``cancelOrder``
    RAISE -- the graceful-shutdown planner and the prepare step must never reach
    them, and the only legitimate placement (protective repair) is exercised via a
    monkeypatched ensure_protective_stops, so the real placeOrder is never hit.
    """

    def __init__(self, positions=None, trades=None):
        self._positions = list(positions or [])
        self._trades = list(trades or [])
        self._connected = True
        self.place_calls = []
        self.cancel_calls = []
        self.disconnect_calls = 0
        self.connect_calls = 0
        self.RequestTimeout = None
        self.market_data_type = None
        self.disconnectedEvent = _FakeEvent()

    # connection
    def connect(self, host=None, port=None, clientId=None, timeout=None):
        self.connect_calls += 1
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
        self.disconnectedEvent.emit()  # real IB fires this on teardown

    def sleep(self, secs=0):
        pass

    # state reads
    def positions(self):
        return list(self._positions)

    def openTrades(self):
        return list(self._trades)

    def reqAllOpenOrders(self):
        pass

    # contracts (only used by the protective-repair path)
    def qualifyContracts(self, contract):
        return [contract]

    # forbidden during shutdown
    def placeOrder(self, *a, **k):  # pragma: no cover - must never run
        self.place_calls.append((a, k))
        raise AssertionError("graceful shutdown must never place an order")

    def cancelOrder(self, *a, **k):  # pragma: no cover - must never run
        self.cancel_calls.append((a, k))
        raise AssertionError("graceful shutdown must never cancel an order")


def _gtc_sell_stop(symbol, qty, order_type="STP"):
    return FakeTrade(symbol, FakeOrder("SELL", order_type, qty, tif="GTC",
                                       orderRef=f"{symbol}:SELL_STOP"))


def _make_bridge(positions=None, trades=None):
    br = ibkr_bridge.IBKRBridge()
    br.ib = FakeIB(positions=positions, trades=trades)
    return br


# ── Base: paper-safe config + silenced audit log ─────────────────────────────
class _ShutdownBase(unittest.TestCase):
    CONFIG = {
        "REQUIRE_PAPER_PORT": True,
        "IBKR_PORT": 7497,
        "PAPER_IBKR_PORT": 7497,
        "ALLOW_HISTORICAL_PRICE_FOR_ORDERS": False,
    }

    def setUp(self):
        for name, val in self.CONFIG.items():
            mock.patch.object(config, name, val).start()
        self.addCleanup(mock.patch.stopall)
        # Never touch the real order_audit.jsonl during tests.
        mock.patch.object(order_audit, "log_event").start()


# ── 1. Pure planner: detection, preservation, safety ─────────────────────────
class TestPurePlanner(_ShutdownBase):
    def test_empty_state_is_clean_and_safe(self):
        plan = sg.build_shutdown_plan([], [])
        self.assertTrue(plan["safe"])
        self.assertEqual(plan["reason"], sg.REASON_CLEAN)
        self.assertEqual(plan["orders_to_cancel"], [])
        self.assertFalse(plan["opens_new_entries"])
        self.assertEqual(plan["open_longs"], [])
        self.assertEqual(plan["unprotected_longs"], [])

    def test_protected_long_is_safe_and_stop_preserved(self):
        positions = [{"symbol": "AAPL", "qty": 10}]
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "STP", "tif": "GTC", "qty": 10}]
        plan = sg.build_shutdown_plan(positions, working)
        self.assertTrue(plan["safe"])
        self.assertEqual(plan["open_longs"], ["AAPL"])
        self.assertEqual(plan["protected_longs"], ["AAPL"])
        self.assertEqual(plan["unprotected_longs"], [])
        # The resting stop is PRESERVED, not cancelled.
        self.assertEqual(len(plan["protective_stops_preserved"]), 1)
        self.assertEqual(len(plan["gtc_protective_stops"]), 1)
        self.assertEqual(plan["orders_to_cancel"], [])

    def test_unprotected_long_is_detected(self):
        positions = [{"symbol": "MSFT", "qty": 5}]
        plan = sg.build_shutdown_plan(positions, [])
        self.assertFalse(plan["safe"])
        self.assertEqual(plan["unprotected_longs"], ["MSFT"])
        self.assertEqual(plan["protected_longs"], [])
        self.assertEqual(plan["reason"], sg.REASON_UNPROTECTED_LONG)
        self.assertTrue(sg.needs_protection_repair(plan))

    def test_take_profit_limit_is_not_downside_protection(self):
        # A plain SELL LIMIT (take-profit) does NOT cap downside -> still unprotected.
        positions = [{"symbol": "NVDA", "qty": 3}]
        working = [{"symbol": "NVDA", "action": "SELL", "order_type": "LMT", "tif": "GTC", "qty": 3}]
        plan = sg.build_shutdown_plan(positions, working)
        self.assertEqual(plan["unprotected_longs"], ["NVDA"])
        self.assertEqual(plan["protective_stops_preserved"], [])

    def test_day_stop_preserved_but_not_counted_as_gtc(self):
        positions = [{"symbol": "AAPL", "qty": 10}]
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "STP", "tif": "DAY", "qty": 10}]
        plan = sg.build_shutdown_plan(positions, working)
        # live_invariants counts any protective SELL as protection -> "safe"...
        self.assertTrue(plan["safe"])
        # ...but only the GTC subset survives between one-shot runs.
        self.assertEqual(len(plan["protective_stops_preserved"]), 1)
        self.assertEqual(len(plan["gtc_protective_stops"]), 0)
        self.assertEqual(plan["orders_to_cancel"], [])

    def test_flat_and_short_positions_ignored(self):
        positions = [{"symbol": "AAPL", "qty": 0}, {"symbol": "TSLA", "qty": -5}]
        plan = sg.build_shutdown_plan(positions, [])
        self.assertEqual(plan["open_longs"], [])
        self.assertTrue(plan["safe"])

    def test_module_has_no_order_placement_symbols(self):
        import inspect
        src = inspect.getsource(sg)
        for forbidden in ("placeOrder", "cancelOrder", "MarketOrder", "LimitOrder", "qualifyContracts"):
            self.assertNotIn(forbidden, src,
                             f"shutdown_guard must not reference {forbidden}")


# ── 2. prepare_graceful_shutdown is READ-ONLY (no placement / no cancel) ──────
class TestPrepareReadOnly(_ShutdownBase):
    def test_prepare_places_and_cancels_nothing(self):
        positions = [FakePosition("AAPL", 10, avgCost=190.0)]
        trades = [_gtc_sell_stop("AAPL", 10)]
        br = _make_bridge(positions=positions, trades=trades)
        plan = br.prepare_graceful_shutdown()
        self.assertEqual(br.ib.place_calls, [], "prepare must not place orders")
        self.assertEqual(br.ib.cancel_calls, [], "prepare must not cancel orders")
        self.assertTrue(plan["safe"])
        self.assertEqual(plan["open_longs"], ["AAPL"])
        self.assertEqual(plan["orders_to_cancel"], [])

    def test_prepare_detects_unprotected_long_without_touching_orders(self):
        positions = [FakePosition("MSFT", 5, avgCost=300.0)]
        br = _make_bridge(positions=positions, trades=[])
        plan = br.prepare_graceful_shutdown()
        self.assertEqual(plan["unprotected_longs"], ["MSFT"])
        self.assertEqual(br.ib.place_calls, [])
        self.assertEqual(br.ib.cancel_calls, [])


# ── 3. graceful_shutdown: clean disconnect, preserve stops, no entries ────────
class TestGracefulShutdown(_ShutdownBase):
    def test_clean_disconnect_keeps_connection_healthy(self):
        # Register the real 5B-2 disconnect handler via connect(), then shut down.
        br = _make_bridge(positions=[FakePosition("AAPL", 10, avgCost=190.0)],
                          trades=[_gtc_sell_stop("AAPL", 10)])
        self.assertTrue(br.connect())
        self.assertTrue(br._connection_healthy())

        report = br.graceful_shutdown()

        self.assertTrue(report["disconnected"])
        self.assertEqual(br.ib.disconnect_calls, 1)
        # Our own teardown must NOT be read as an unexpected mid-run drop.
        self.assertTrue(br._conn_health.is_healthy)
        self.assertEqual(br._conn_health.disconnect_count, 0)
        self.assertTrue(br._intentional_disconnect)

    def test_protective_gtc_stops_are_left_alone(self):
        trades = [_gtc_sell_stop("AAPL", 10)]
        br = _make_bridge(positions=[FakePosition("AAPL", 10, avgCost=190.0)], trades=trades)
        report = br.graceful_shutdown()
        # No cancel of the resting protective child; plan preserves it.
        self.assertEqual(br.ib.cancel_calls, [])
        self.assertEqual(report["plan"]["orders_to_cancel"], [])
        self.assertEqual(len(report["plan"]["gtc_protective_stops"]), 1)

    def test_shutdown_opens_no_new_entry(self):
        # An unprotected long with repair OFF: still no order is opened/placed.
        br = _make_bridge(positions=[FakePosition("MSFT", 5, avgCost=300.0)], trades=[])
        report = br.graceful_shutdown(repair=False)
        self.assertEqual(br.ib.place_calls, [], "shutdown must not open a new entry")
        self.assertEqual(br.ib.cancel_calls, [])
        self.assertFalse(report["repair_attempted"])
        self.assertFalse(report["plan"]["opens_new_entries"])
        self.assertTrue(report["disconnected"])

    def test_repair_delegates_to_ensure_protective_stops_only(self):
        # Healthy + unprotected long + repair ON -> repair runs, but ONLY through
        # the existing ensure_protective_stops() path (monkeypatched here so the
        # real placeOrder is never reached).
        br = _make_bridge(positions=[FakePosition("MSFT", 5, avgCost=300.0)], trades=[])
        calls = {"n": 0}

        def fake_ensure():
            calls["n"] += 1
            return {"checked": 1, "unprotected": ["MSFT"], "repaired": ["MSFT"], "failed": []}

        br.ensure_protective_stops = fake_ensure
        report = br.graceful_shutdown(repair=True)
        self.assertEqual(calls["n"], 1, "repair must go through ensure_protective_stops")
        self.assertTrue(report["repair_attempted"])
        self.assertEqual(report["repaired"], ["MSFT"])
        self.assertEqual(br.ib.place_calls, [], "no direct placement outside ensure_protective_stops")
        self.assertTrue(report["disconnected"])

    def test_repair_is_failclosed_on_unhealthy_link(self):
        # A dropped/unhealthy link must NOT trigger any repair placement attempt.
        br = _make_bridge(positions=[FakePosition("MSFT", 5, avgCost=300.0)], trades=[])
        br._conn_health.mark_unhealthy(ibkr_bridge.reconnect_watchdog.REASON_DISCONNECTED)
        called = {"n": 0}

        def fake_ensure():
            called["n"] += 1
            return {"checked": 0, "unprotected": [], "repaired": [], "failed": []}

        br.ensure_protective_stops = fake_ensure
        report = br.graceful_shutdown(repair=True)
        self.assertEqual(called["n"], 0, "no repair on an unhealthy link (fail-closed)")
        self.assertFalse(report["repair_attempted"])
        self.assertEqual(br.ib.place_calls, [])
        self.assertTrue(report["disconnected"])

    def test_shutdown_never_raises_even_if_state_read_fails(self):
        br = _make_bridge(positions=[FakePosition("AAPL", 10, avgCost=190.0)], trades=[])

        def boom():
            raise RuntimeError("broker read failed")

        br.ib.positions = boom  # force the plan-build to hit its guard
        report = br.graceful_shutdown()  # must not raise
        self.assertTrue(report["disconnected"])


# ── 4. live-readiness still NOT READY + no new capability flag ────────────────
class TestStillNotReady(_ShutdownBase):
    # The complete, intentional set of live-readiness capability flags. Phase
    # 5B-3 must NOT add one, so this set is exhaustive.
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
                         "Phase 5B-3 must not add a live-readiness capability flag")
        # The Phase-6 gate stays False (fail-closed).
        self.assertFalse(ibkr_bridge.SUPPORTS_ACCOUNT_TYPE_ASSERTION)

    def test_live_readiness_reports_not_ready(self):
        import main
        args = mock.Mock(connect=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main.cmd_live_readiness(args)
        self.assertEqual(rc, 1, "live-readiness must still report NOT READY")


# ── 5. bridge surface smoke ──────────────────────────────────────────────────
class TestBridgeSurface(_ShutdownBase):
    def test_bridge_exposes_graceful_shutdown_methods(self):
        self.assertTrue(hasattr(ibkr_bridge.IBKRBridge, "prepare_graceful_shutdown"))
        self.assertTrue(hasattr(ibkr_bridge.IBKRBridge, "graceful_shutdown"))

    def test_order_audit_has_shutdown_stage(self):
        self.assertEqual(order_audit.STAGE_SHUTDOWN, "shutdown")


if __name__ == "__main__":
    unittest.main(verbosity=2)
