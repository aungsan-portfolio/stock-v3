"""
test_alerts.py - Phase 5B-4 alerting layer.

Fully offline and deterministic: NO live IBKR, NO network, NO real mail server, NO
real time. The alerting layer is INERT by default and CANNOT affect trading; these
tests prove the Phase-5B-4 contract from
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md (task 5.5):

  * alerts DISABLED -> emit() is a no-op (no logging, no external action)
  * a LOG-ONLY alert records the expected JSON-safe payload (logger + audit trail)
  * severity filtering works (below ALERT_MIN_SEVERITY is suppressed)
  * alert payloads are JSON-safe (a non-serializable context never breaks emit)
  * emit() NEVER raises into the trading path (even if delivery blows up)
  * emit() NEVER places an order and NEVER enables live trading
  * external email/SMS/Telegram/webhook are inert disabled stubs (send nothing)
  * a watchdog disconnect can emit an alert WITHOUT placing an order
  * a blocked scheduled run can emit an alert WITHOUT dispatching
  * a graceful-shutdown warning can emit an alert WITHOUT cancelling a stop
  * live-readiness is STILL NOT READY (Phase 6 gates remain False) and Phase 5B-4
    adds NO new live-readiness capability flag

Run with:
    python -X utf8 -m unittest test_alerts -v
    python test_alerts.py
"""
import asyncio
import contextlib
import io
import json
import unittest
from datetime import datetime
from unittest import mock


# ib_insync touches the event loop at import time; prepare it first (as main.py
# and the other ibkr tests do).
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import config              # noqa: E402
import order_audit         # noqa: E402
import alerts              # noqa: E402
import reconnect_watchdog as rw  # noqa: E402
import scheduler_runner as sr    # noqa: E402
import ibkr_bridge         # noqa: E402


# 2026-06-17 is a Wednesday trading day; 2026-06-20 is a Saturday (weekend).
RTH_OPEN = datetime(2026, 6, 17, 10, 0)
SATURDAY = datetime(2026, 6, 20, 10, 0)


# ── Minimal fake ib_insync surface (union of the watchdog + shutdown fakes) ──
class _FakeEvent:
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


class FakeIB:
    """Minimal ib_insync.IB replacement. placeOrder/cancelOrder RAISE so any
    attempt to place or cancel an order while emitting an alert fails the test."""

    def __init__(self, positions=None, trades=None, fail_times=0):
        self._positions = list(positions or [])
        self._trades = list(trades or [])
        self.fail_times = fail_times
        self.connect_calls = 0
        self._connected = False
        self.place_calls = []
        self.cancel_calls = []
        self.disconnect_calls = 0
        self.slept = []
        self.RequestTimeout = None
        self.market_data_type = None
        self.disconnectedEvent = _FakeEvent()

    def connect(self, host=None, port=None, clientId=None, timeout=None):
        self.connect_calls += 1
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
        self.disconnectedEvent.emit()

    def sleep(self, secs=0):
        self.slept.append(float(secs))

    def positions(self):
        return list(self._positions)

    def openTrades(self):
        return list(self._trades)

    def reqAllOpenOrders(self):
        pass

    def qualifyContracts(self, contract):
        return [contract]

    def placeOrder(self, *a, **k):  # pragma: no cover - must never run
        self.place_calls.append((a, k))
        raise AssertionError("alerting must never place an order")

    def cancelOrder(self, *a, **k):  # pragma: no cover - must never run
        self.cancel_calls.append((a, k))
        raise AssertionError("alerting must never cancel an order")


def _make_bridge(positions=None, trades=None, fail_times=0):
    br = ibkr_bridge.IBKRBridge()
    br.ib = FakeIB(positions=positions, trades=trades, fail_times=fail_times)
    return br


class _Dispatch:
    """Records dispatch calls; never places an order."""

    def __init__(self, rc=0):
        self.calls = []
        self.rc = rc

    def __call__(self, dry_run):
        self.calls.append(bool(dry_run))
        return self.rc


# ── Base: alerts ENABLED + log-only, audit silenced, recorder reset ──────────
class _AlertsBase(unittest.TestCase):
    # Default to a low floor so a test observes everything unless it overrides it.
    CONFIG = {
        "ALERTS_ENABLED": True,
        "ALERTS_LOG_ONLY": True,
        "ALERT_MIN_SEVERITY": "info",
    }

    def setUp(self):
        for name, val in self.CONFIG.items():
            mock.patch.object(config, name, val).start()
        self.addCleanup(mock.patch.stopall)
        # Never touch the real order_audit.jsonl; capture the calls instead.
        self.audit = mock.patch.object(order_audit, "log_event").start()
        alerts.reset()
        self.addCleanup(alerts.reset)


# ── 1. Disabled -> no external action ─────────────────────────────────────────
class TestDisabled(_AlertsBase):
    CONFIG = {"ALERTS_ENABLED": False, "ALERTS_LOG_ONLY": True, "ALERT_MIN_SEVERITY": "info"}

    def test_disabled_emit_is_a_noop(self):
        result = alerts.emit(alerts.EVENT_DISCONNECT, message="x", reason="drop")
        self.assertIsNone(result, "disabled emit must return None")
        self.assertEqual(alerts.recent_alerts(), [], "nothing recorded when disabled")
        self.audit.assert_not_called()  # no audit-trail write -> no external action

    def test_external_registry_is_empty(self):
        self.assertEqual(alerts.EXTERNAL_CHANNELS, [],
                         "Phase 5B-4 ships with no external channels")


# ── 2. Log-only records expected payload ──────────────────────────────────────
class TestLogOnlyPayload(_AlertsBase):
    def test_log_only_records_payload(self):
        payload = alerts.emit(alerts.EVENT_ORDER_REJECTED, symbol="AAPL", status="rejected")
        self.assertIsNotNone(payload)
        self.assertEqual(payload["event"], alerts.EVENT_ORDER_REJECTED)
        self.assertEqual(payload["severity"], alerts.SEVERITY_WARNING)  # default for reject
        self.assertEqual(payload["channels"], ["log"], "log-only delivers via the log channel only")
        self.assertTrue(payload["log_only"])
        self.assertEqual(payload["context"]["symbol"], "AAPL")
        self.assertIn("ts", payload)

        recent = alerts.recent_alerts()
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["event"], alerts.EVENT_ORDER_REJECTED)

        # Persisted to the order_audit trail under STAGE_ALERT.
        self.audit.assert_called_once()
        self.assertEqual(self.audit.call_args.args[0], order_audit.STAGE_ALERT)

    def test_default_severity_per_event(self):
        self.assertEqual(alerts._normalize_severity(None, alerts.EVENT_DISCONNECT),
                         alerts.SEVERITY_CRITICAL)
        self.assertEqual(alerts._normalize_severity(None, alerts.EVENT_PARTIAL_FILL),
                         alerts.SEVERITY_WARNING)
        # An explicit override wins.
        self.assertEqual(alerts._normalize_severity("info", alerts.EVENT_DISCONNECT),
                         alerts.SEVERITY_INFO)


# ── 3. Severity filtering ─────────────────────────────────────────────────────
class TestSeverityFilter(_AlertsBase):
    def test_below_threshold_is_suppressed(self):
        with mock.patch.object(config, "ALERT_MIN_SEVERITY", "critical"):
            warned = alerts.emit(alerts.EVENT_PARTIAL_FILL, severity="warning", symbol="A")
            self.assertIsNone(warned, "warning is below a critical floor -> suppressed")
            self.assertEqual(alerts.recent_alerts(), [])

            crit = alerts.emit(alerts.EVENT_DISCONNECT, severity="critical")
            self.assertIsNotNone(crit)
            self.assertEqual(len(alerts.recent_alerts()), 1)

    def test_warning_floor_suppresses_info(self):
        with mock.patch.object(config, "ALERT_MIN_SEVERITY", "warning"):
            info = alerts.emit(alerts.EVENT_SCHEDULER_BLOCKED, severity="info", reason="weekend")
            self.assertIsNone(info)
            self.assertEqual(alerts.recent_alerts(), [])

    def test_unknown_threshold_falls_back_to_warning(self):
        with mock.patch.object(config, "ALERT_MIN_SEVERITY", "garbage"):
            self.assertEqual(alerts.min_severity(), "warning")
            self.assertIsNone(alerts.emit(alerts.EVENT_SCHEDULER_BLOCKED, severity="info"))
            self.assertIsNotNone(alerts.emit(alerts.EVENT_DISCONNECT, severity="critical"))


# ── 4. Payloads are JSON-safe ─────────────────────────────────────────────────
class TestJsonSafe(_AlertsBase):
    def test_non_serializable_context_is_coerced(self):
        class Weird:
            def __repr__(self):
                return "WEIRD"

        payload = alerts.emit(alerts.EVENT_DISCONNECT, severity="critical",
                              obj=Weird(), nested={"k": Weird()}, items=(1, 2, 3))
        self.assertIsNotNone(payload)
        # The whole payload must round-trip through json.dumps without raising.
        encoded = json.dumps(payload)
        self.assertIn("WEIRD", encoded)
        self.assertIsInstance(payload["context"]["obj"], str)
        self.assertEqual(payload["context"]["items"], [1, 2, 3])
        self.assertIsInstance(payload["context"]["nested"]["k"], str)


# ── 5. emit() never raises into the trading path ─────────────────────────────
class TestNeverRaises(_AlertsBase):
    def test_emit_swallows_delivery_failure(self):
        with mock.patch.object(alerts, "_deliver_log", side_effect=RuntimeError("boom")):
            try:
                result = alerts.emit(alerts.EVENT_DISCONNECT, severity="critical")
            except Exception as exc:  # pragma: no cover - the whole point is no raise
                self.fail(f"emit() raised into the caller: {exc}")
        self.assertIsNone(result)

    def test_emit_tolerates_bad_arguments(self):
        # A None event / weird severity must not raise.
        try:
            alerts.emit(None, severity=object())  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            self.fail(f"emit() raised on bad args: {exc}")


# ── 6. emit() places no order and enables no live trading ─────────────────────
class TestNoOrderNoLive(_AlertsBase):
    def test_module_references_no_order_verbs(self):
        import inspect
        src = inspect.getsource(alerts)
        for forbidden in ("placeOrder", "cancelOrder", "MarketOrder", "LimitOrder",
                          "qualifyContracts"):
            self.assertNotIn(forbidden, src,
                             f"alerts.py must not reference {forbidden}")
        # It must not IMPORT ib_insync (the docstring may mention it by name only
        # to state that it does not).
        self.assertNotIn("import ib_insync", src)
        self.assertNotIn("from ib_insync", src)

    def test_emit_does_not_enable_live_trading(self):
        before = getattr(config, "COACH_LIVE_TRADING_ENABLED", False)
        alerts.emit(alerts.EVENT_DAILY_LOSS_KILLSWITCH, severity="critical")
        after = getattr(config, "COACH_LIVE_TRADING_ENABLED", False)
        self.assertEqual(before, after)
        self.assertFalse(after, "live trading must stay disabled")

    def test_external_channels_are_inert_stubs(self):
        self.assertEqual(alerts.EXTERNAL_CHANNELS, [])
        self.assertFalse(alerts.send_email({"event": "x"}))
        self.assertFalse(alerts.send_sms({"event": "x"}))
        self.assertFalse(alerts.send_telegram({"event": "x"}))
        self.assertFalse(alerts.send_webhook({"event": "x"}))

    def test_all_required_event_types_defined(self):
        required = [
            alerts.EVENT_DISCONNECT, alerts.EVENT_RECONNECT_FAILURE,
            alerts.EVENT_RECONCILE_UNPROTECTED_LONG, alerts.EVENT_ORPHAN_EXIT_ORDER,
            alerts.EVENT_DUPLICATE_ORDER_REF, alerts.EVENT_DAILY_LOSS_KILLSWITCH,
            alerts.EVENT_ORDER_REJECTED, alerts.EVENT_PARTIAL_FILL,
            alerts.EVENT_PROTECTIVE_CHILD_FAILURE, alerts.EVENT_EMERGENCY_FLATTEN,
            alerts.EVENT_SCHEDULER_BLOCKED, alerts.EVENT_SHUTDOWN_WARNING,
            alerts.EVENT_SHUTDOWN_UNPROTECTED_LONG,
        ]
        for ev in required:
            self.assertIn(ev, alerts._DEFAULT_SEVERITY, f"{ev} needs a default severity")


# ── 7. Watchdog disconnect emits without placing orders ──────────────────────
class TestWatchdogDisconnectAlert(_AlertsBase):
    WATCHDOG_CONFIG = {
        "REQUIRE_PAPER_PORT": True,
        "IBKR_PORT": 7497,
        "PAPER_IBKR_PORT": 7497,
        "IBKR_RECONNECT_ENABLED": True,
        "IBKR_RECONNECT_MAX_ATTEMPTS": 1,
        "IBKR_RECONNECT_BASE_DELAY_SECONDS": 0.0,
        "IBKR_RECONNECT_MAX_DELAY_SECONDS": 0.0,
        "IBKR_REQUEST_TIMEOUT_SECONDS": 0.0,
    }

    def setUp(self):
        super().setUp()
        for name, val in self.WATCHDOG_CONFIG.items():
            mock.patch.object(config, name, val).start()

    def test_unexpected_disconnect_emits_alert_no_order(self):
        br = _make_bridge(fail_times=0)
        self.assertTrue(br.connect())
        self.assertTrue(br._connection_healthy())

        # Simulate an UNEXPECTED mid-run drop (not via bridge.disconnect()).
        br.ib.disconnectedEvent.emit()

        self.assertFalse(br._conn_health.is_healthy)
        disconnects = [a for a in alerts.recent_alerts() if a["event"] == alerts.EVENT_DISCONNECT]
        self.assertEqual(len(disconnects), 1, "an unexpected disconnect must alert")
        self.assertEqual(disconnects[0]["severity"], alerts.SEVERITY_CRITICAL)
        self.assertEqual(br.ib.place_calls, [], "alerting placed an order -- forbidden")

    def test_intentional_disconnect_emits_no_alert(self):
        br = _make_bridge(fail_times=0)
        self.assertTrue(br.connect())
        br.disconnect()  # clean teardown -> not an unexpected drop
        disconnects = [a for a in alerts.recent_alerts() if a["event"] == alerts.EVENT_DISCONNECT]
        self.assertEqual(disconnects, [], "a clean shutdown must NOT alert as a drop")


# ── 8. Scheduler block emits without dispatching ─────────────────────────────
class TestSchedulerBlockedAlert(_AlertsBase):
    def _paper_only(self):
        for name, val in {
            "COACH_LIVE_TRADING_ENABLED": False, "REQUIRE_PAPER_PORT": True,
            "IBKR_PORT": 7497, "PAPER_IBKR_PORT": 7497, "ALLOW_SHORT": False,
            "ALLOW_HISTORICAL_PRICE_FOR_ORDERS": False, "SCHEDULER_ENABLED": True,
            "SCHEDULER_REQUIRE_RTH": True, "SCHEDULER_DRY_RUN_DEFAULT": True,
        }.items():
            mock.patch.object(config, name, val).start()

    def test_paper_safety_block_emits_critical_without_dispatch(self):
        self._paper_only()
        disp = _Dispatch()
        with mock.patch.object(config, "COACH_LIVE_TRADING_ENABLED", True):
            result = sr.run_scheduled(now_et=RTH_OPEN, dispatch=disp)
        self.assertFalse(result["ran"])
        self.assertEqual(disp.calls, [], "a blocked run must NOT dispatch")
        blocked = [a for a in alerts.recent_alerts() if a["event"] == alerts.EVENT_SCHEDULER_BLOCKED]
        self.assertTrue(blocked)
        self.assertEqual(blocked[-1]["severity"], alerts.SEVERITY_CRITICAL)

    def test_routine_weekend_block_is_info_and_filtered_by_default_floor(self):
        self._paper_only()
        disp = _Dispatch()
        # With the SHIPPED default floor (warning), a routine weekend block (INFO)
        # is suppressed -> no alert spam, still no dispatch.
        with mock.patch.object(config, "ALERT_MIN_SEVERITY", "warning"):
            result = sr.run_scheduled(now_et=SATURDAY, dispatch=disp)
        self.assertFalse(result["ran"])
        self.assertEqual(disp.calls, [])
        self.assertEqual(
            [a for a in alerts.recent_alerts() if a["event"] == alerts.EVENT_SCHEDULER_BLOCKED],
            [], "a routine market-closed block must not spam at the default floor")

    def test_routine_weekend_block_visible_at_info_floor(self):
        self._paper_only()
        disp = _Dispatch()
        with mock.patch.object(config, "ALERT_MIN_SEVERITY", "info"):
            sr.run_scheduled(now_et=SATURDAY, dispatch=disp)
        blocked = [a for a in alerts.recent_alerts() if a["event"] == alerts.EVENT_SCHEDULER_BLOCKED]
        self.assertTrue(blocked)
        self.assertEqual(blocked[-1]["severity"], alerts.SEVERITY_INFO)
        self.assertEqual(disp.calls, [])


# ── 9. Shutdown warning emits without cancelling a stop ──────────────────────
class TestShutdownAlert(_AlertsBase):
    SHUTDOWN_CONFIG = {
        "REQUIRE_PAPER_PORT": True, "IBKR_PORT": 7497, "PAPER_IBKR_PORT": 7497,
        "ALLOW_HISTORICAL_PRICE_FOR_ORDERS": False,
    }

    def setUp(self):
        super().setUp()
        for name, val in self.SHUTDOWN_CONFIG.items():
            mock.patch.object(config, name, val).start()

    def test_unprotected_long_at_shutdown_emits_without_cancel(self):
        br = _make_bridge(positions=[FakePosition("MSFT", 5, avgCost=300.0)], trades=[])
        report = br.graceful_shutdown(repair=False)
        self.assertTrue(report["disconnected"])
        self.assertEqual(br.ib.cancel_calls, [], "shutdown alert must not cancel a stop")
        self.assertEqual(br.ib.place_calls, [], "shutdown alert must not place an order")
        warnings = [a for a in alerts.recent_alerts()
                    if a["event"] == alerts.EVENT_SHUTDOWN_UNPROTECTED_LONG]
        self.assertTrue(warnings, "an unprotected long at shutdown must alert")
        self.assertEqual(warnings[-1]["context"]["symbols"], ["MSFT"])

    def test_safe_shutdown_emits_no_unprotected_alert(self):
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "STP",
                    "tif": "GTC", "qty": 10}]
        from test_shutdown_guard import FakeTrade, FakeOrder  # reuse the trade shape
        trades = [FakeTrade("AAPL", FakeOrder("SELL", "STP", 10, tif="GTC",
                                              orderRef="AAPL:SELL_STOP"))]
        br = _make_bridge(positions=[FakePosition("AAPL", 10, avgCost=190.0)], trades=trades)
        report = br.graceful_shutdown(repair=False)
        self.assertTrue(report["disconnected"])
        self.assertEqual(
            [a for a in alerts.recent_alerts()
             if a["event"] == alerts.EVENT_SHUTDOWN_UNPROTECTED_LONG],
            [], "a protected long must not raise a shutdown alert")


# ── 10. Live-readiness still NOT READY + no new capability flag ──────────────
class TestStillNotReady(_AlertsBase):
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

    def test_alerts_module_exposes_no_capability_flag(self):
        flags = [n for n in dir(alerts) if n.startswith("SUPPORTS_")]
        self.assertEqual(flags, [], "Phase 5B-4 must not add a capability flag")

    def test_no_new_capability_flag_on_bridge(self):
        present = {a for a in dir(ibkr_bridge) if a.startswith("SUPPORTS_")}
        self.assertEqual(present, self.EXPECTED_CAPABILITY_FLAGS)
        # Phase 6.1 ships the account-type assertion (built + fail-closed tested),
        # so its capability flag is honestly True; Phase 5B-4 adds no flag.
        self.assertTrue(ibkr_bridge.SUPPORTS_ACCOUNT_TYPE_ASSERTION,
                        "Phase 6.1 account-type assertion is built + tested")

    def test_live_readiness_reports_not_ready(self):
        import main
        args = mock.Mock(connect=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main.cmd_live_readiness(args)
        self.assertEqual(rc, 1, "live-readiness must still report NOT READY")


if __name__ == "__main__":
    unittest.main(verbosity=2)
