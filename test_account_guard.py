"""
test_account_guard.py - Phase 6.1 account-type assertion (paper DU / live U).

Fully offline and deterministic: NO live IBKR, NO network, NO real sleep, NO wall
clock. The pure decision layer (account_guard.assert_account) is exercised
directly, and a tiny FakeIB drives the REAL IBKRBridge.connect() path so the
fail-closed refusal of a wrong account is proven end to end in microseconds.

These back the Phase-6.1 contract from reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md
(task 6.1) -- the guard the paper-port lock does NOT provide: the port lock guards
the PORT (7497); this guards the ACCOUNT logged into that port. PAPER ONLY: live
mode is implemented but INERT (the bridge never selects it for the shipped paper
config) and nothing here enables live trading.

Run with:
    python -m unittest test_account_guard -v
    python test_account_guard.py
"""
import asyncio
import unittest
from unittest import mock

# ib_insync touches the event loop at import time; prepare it first (as main.py
# and the other ibkr tests do).
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import account_guard  # noqa: E402
import config         # noqa: E402
import order_audit    # noqa: E402
import ibkr_bridge    # noqa: E402


# ── 1. Pure assert_account: PAPER mode ───────────────────────────────────────
class TestPaperAssertions(unittest.TestCase):
    def test_paper_du_account_passes(self):
        r = account_guard.assert_account(["DU1234567"])
        self.assertTrue(r.ok)
        self.assertEqual(r.mode, account_guard.MODE_PAPER)
        self.assertEqual(r.account, "DU1234567")
        self.assertEqual(r.reason, account_guard.REASON_OK)

    def test_paper_du_account_lowercase_and_whitespace_normalized(self):
        r = account_guard.assert_account(["  du7654321 "])
        self.assertTrue(r.ok)
        self.assertEqual(r.account, "DU7654321")

    def test_paper_non_du_live_account_fails_closed(self):
        # The critical gap: a LIVE "U..." account on the paper port must fail.
        r = account_guard.assert_account(["U1234567"])
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_WRONG_PREFIX)
        # The offending account id is recorded (for the audit) but the verdict is
        # a hard reject -- it is never treated as usable.
        self.assertEqual(r.account, "U1234567")

    def test_paper_other_prefix_fails_closed(self):
        for acct in (["F1234567"], ["C7654321"], ["X1112223"]):
            with self.subTest(acct=acct):
                r = account_guard.assert_account(acct)
                self.assertFalse(r.ok)
                self.assertEqual(r.reason, account_guard.REASON_WRONG_PREFIX)

    def test_empty_managed_accounts_fails_closed(self):
        for empty in ([], None, "", "   ", [""], [None], ()):
            with self.subTest(empty=empty):
                r = account_guard.assert_account(empty)
                self.assertFalse(r.ok)
                self.assertEqual(r.reason, account_guard.REASON_EMPTY)

    def test_non_iterable_managed_accounts_fails_closed(self):
        # A bogus non-iterable (e.g. an int) must be treated as empty, not crash.
        r = account_guard.assert_account(12345)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_EMPTY)

    def test_malformed_account_id_fails_closed(self):
        for bad in (["BADACCT"], ["12345"], ["DU"], ["DU-99"], ["DU 123"],
                    ["DU12"], ["!!!"], [object()]):
            with self.subTest(bad=bad):
                r = account_guard.assert_account(bad)
                self.assertFalse(r.ok)
                self.assertEqual(r.reason, account_guard.REASON_MALFORMED)

    def test_multiple_accounts_without_expected_fails_closed(self):
        r = account_guard.assert_account(["DU1111111", "DU2222222"])
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_AMBIGUOUS)
        self.assertIsNone(r.account)

    def test_multiple_accounts_never_silently_picks_a_live_account(self):
        # A paper + a live account both present, no explicit expected -> ambiguous;
        # the guard must NOT auto-select either (and certainly not the live one).
        r = account_guard.assert_account(["DU1111111", "U2222222"])
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_AMBIGUOUS)
        self.assertIsNone(r.account)

    def test_explicit_expected_du_account_passes_when_present(self):
        r = account_guard.assert_account(
            ["DU1111111", "DU2222222"], expected_account="DU2222222")
        self.assertTrue(r.ok)
        self.assertEqual(r.account, "DU2222222")

    def test_explicit_expected_account_missing_fails_closed(self):
        r = account_guard.assert_account(
            ["DU1111111", "DU2222222"], expected_account="DU9999999")
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_EXPECTED_MISSING)

    def test_explicit_expected_single_account_mismatch_fails_closed(self):
        r = account_guard.assert_account(["DU1111111"], expected_account="DU2222222")
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_EXPECTED_MISSING)

    def test_expected_live_account_in_paper_mode_fails_prefix(self):
        # Even if someone names a U... account as the paper expected, paper mode
        # still demands the DU prefix -> fail closed (never trades the live one).
        r = account_guard.assert_account(
            ["DU1111111", "U2222222"], expected_account="U2222222")
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_WRONG_PREFIX)

    def test_comma_joined_string_form_is_supported(self):
        # Some ib_insync versions return a comma-joined string (with a trailing
        # comma). The trailing-blank artifact is dropped; the DU account passes.
        r = account_guard.assert_account("DU1234567,")
        self.assertTrue(r.ok)
        self.assertEqual(r.account, "DU1234567")


# ── 2. Pure assert_account: LIVE mode (implemented but INERT this phase) ──────
class TestLiveAssertionsInert(unittest.TestCase):
    def test_live_u_account_with_explicit_expected_passes(self):
        r = account_guard.assert_account(
            ["U1234567"], live_mode=True, expected_account="U1234567")
        self.assertTrue(r.ok)
        self.assertEqual(r.mode, account_guard.MODE_LIVE)
        self.assertEqual(r.account, "U1234567")

    def test_live_requires_explicit_expected_account(self):
        # Live NEVER auto-selects, even a single account -> needs LIVE_ACCOUNT_ID.
        r = account_guard.assert_account(["U1234567"], live_mode=True)
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_EXPECTED_MISSING)

    def test_live_rejects_paper_account(self):
        r = account_guard.assert_account(
            ["DU1234567"], live_mode=True, expected_account="DU1234567")
        self.assertFalse(r.ok)
        self.assertEqual(r.reason, account_guard.REASON_WRONG_PREFIX)

    def test_live_mode_is_inert_for_shipped_paper_config(self):
        # The bridge derives the mode from config; the shipped paper config must
        # NEVER resolve to live mode (COACH_LIVE_TRADING_ENABLED False,
        # REQUIRE_PAPER_PORT True, LIVE_ACCOUNT_ID None).
        self.assertFalse(account_guard.live_mode_enabled(config))

    def test_live_mode_stays_false_unless_all_live_switches_set(self):
        # Even with the live flag on, the paper-port lock + missing LIVE_ACCOUNT_ID
        # keep us in paper mode (fail-safe default). Only ALL three together flip.
        with mock.patch.object(config, "COACH_LIVE_TRADING_ENABLED", True):
            self.assertFalse(account_guard.live_mode_enabled(config))
            with mock.patch.object(config, "REQUIRE_PAPER_PORT", False):
                self.assertFalse(account_guard.live_mode_enabled(config))
                with mock.patch.object(config, "LIVE_ACCOUNT_ID", "U1234567"):
                    self.assertTrue(account_guard.live_mode_enabled(config))


# ── 3. Validation / helper primitives ────────────────────────────────────────
class TestHelpers(unittest.TestCase):
    def test_is_well_formed(self):
        for good in ("DU1234567", "U1234567", "du1234567", " U7654321 "):
            self.assertTrue(account_guard.is_well_formed(good), good)
        for bad in ("DU", "12345", "BADACCT", "DU-99", "", None, "DU12"):
            self.assertFalse(account_guard.is_well_formed(bad), bad)

    def test_has_environment_prefix(self):
        self.assertTrue(account_guard.has_environment_prefix("DU1234567", live=False))
        self.assertFalse(account_guard.has_environment_prefix("U1234567", live=False))
        self.assertTrue(account_guard.has_environment_prefix("U1234567", live=True))
        # A paper "DU..." can never satisfy the live "U" prefix.
        self.assertFalse(account_guard.has_environment_prefix("DU1234567", live=True))
        # A stray "UABC123" must not masquerade as a live account.
        self.assertFalse(account_guard.has_environment_prefix("UABC123", live=True))

    def test_audit_fields_are_json_safe(self):
        r = account_guard.assert_account(["DU1234567"])
        fields = account_guard.audit_fields(r)
        self.assertTrue(fields["ok"])
        self.assertEqual(fields["mode"], "paper")
        self.assertEqual(fields["account"], "DU1234567")
        self.assertIsInstance(fields["accounts"], list)


# ── 4. Connect-path: the bridge FAILS CLOSED on a wrong account ───────────────
class _FakeEvent:
    def __init__(self):
        self._handlers = []

    def __iadd__(self, handler):
        self._handlers.append(handler)
        return self

    def emit(self, *args, **kwargs):
        for h in list(self._handlers):
            h(*args, **kwargs)


class FakeIB:
    """Minimal ib_insync.IB replacement whose managedAccounts() is configurable.

    connect() always succeeds (the socket is fine); the ACCOUNT is what is under
    test. placeOrder raises if ever called -- the guard must never place an order.
    """

    def __init__(self, accounts):
        self._accounts = accounts
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.place_calls = []
        self.RequestTimeout = None
        self.market_data_type = None
        self._connected = False
        self.disconnectedEvent = _FakeEvent()

    def connect(self, host=None, port=None, clientId=None, timeout=None):
        self.connect_calls += 1
        self._connected = True

    def reqMarketDataType(self, t):
        self.market_data_type = t

    def managedAccounts(self):
        return self._accounts

    def isConnected(self):
        return self._connected

    def disconnect(self):
        self.disconnect_calls += 1
        self._connected = False
        self.disconnectedEvent.emit()

    def sleep(self, secs=0):
        pass

    def placeOrder(self, *args, **kwargs):  # pragma: no cover - must never run
        self.place_calls.append((args, kwargs))
        raise AssertionError("account guard must never place an order")


class _ConnectBase(unittest.TestCase):
    CONFIG = {
        "REQUIRE_PAPER_PORT": True,
        "IBKR_PORT": 7497,
        "PAPER_IBKR_PORT": 7497,
        "IBKR_RECONNECT_ENABLED": True,
        "IBKR_RECONNECT_MAX_ATTEMPTS": 1,
        "IBKR_RECONNECT_BASE_DELAY_SECONDS": 0.0,
        "IBKR_RECONNECT_MAX_DELAY_SECONDS": 0.0,
        "IBKR_REQUEST_TIMEOUT_SECONDS": 5.0,
        "ASSERT_ACCOUNT_TYPE": True,
        "EXPECTED_PAPER_ACCOUNT_ID": None,
        "LIVE_ACCOUNT_ID": None,
        "COACH_LIVE_TRADING_ENABLED": False,
    }

    def setUp(self):
        for name, val in self.CONFIG.items():
            mock.patch.object(config, name, val).start()
            self.addCleanup(mock.patch.stopall)
        # Never write the real order_audit.jsonl during tests.
        mock.patch.object(order_audit, "log_event").start()

    def _bridge(self, accounts):
        br = ibkr_bridge.IBKRBridge()
        br.ib = FakeIB(accounts)
        return br


@unittest.skipIf(getattr(ibkr_bridge.IBKRBridge, "__name__", "") == "AlpacaBridge", "Skip connection checks when using AlpacaBridge compatibility wrapper")
class TestConnectRefusesWrongAccount(_ConnectBase):
    def test_paper_du_account_connects(self):
        br = self._bridge(["DU1234567"])
        self.assertTrue(br.connect())
        self.assertTrue(br._conn_health.is_healthy)
        self.assertEqual(br.ib.place_calls, [])

    def test_live_account_on_paper_port_refused(self):
        br = self._bridge(["U1234567"])              # a LIVE account on 7497
        self.assertFalse(br.connect(), "must refuse a live account on the paper port")
        self.assertFalse(br._conn_health.is_healthy, "fail closed -> unhealthy")
        self.assertEqual(br._conn_health.reason, account_guard.HEALTH_REASON)
        self.assertGreaterEqual(br.ib.disconnect_calls, 1, "must not stay connected")
        self.assertEqual(br.ib.place_calls, [])

    def test_empty_accounts_refused(self):
        br = self._bridge([])
        self.assertFalse(br.connect())
        self.assertFalse(br._conn_health.is_healthy)

    def test_malformed_account_refused(self):
        br = self._bridge(["NOTANACCOUNT"])
        self.assertFalse(br.connect())
        self.assertFalse(br._conn_health.is_healthy)

    def test_multiple_accounts_without_expected_refused(self):
        br = self._bridge(["DU1111111", "DU2222222"])
        self.assertFalse(br.connect(), "ambiguous -> fail closed")
        self.assertFalse(br._conn_health.is_healthy)

    def test_explicit_expected_paper_account_connects(self):
        with mock.patch.object(config, "EXPECTED_PAPER_ACCOUNT_ID", "DU2222222"):
            br = self._bridge(["DU1111111", "DU2222222"])
            self.assertTrue(br.connect())
            self.assertTrue(br._conn_health.is_healthy)

    def test_disabled_switch_skips_assertion(self):
        # The documented escape hatch: with the switch off, connect() no longer
        # asserts the account (it still connects). Ships True, so this is opt-out.
        with mock.patch.object(config, "ASSERT_ACCOUNT_TYPE", False):
            br = self._bridge(["U1234567"])  # would normally be refused
            self.assertTrue(br.connect())

    def test_managed_accounts_error_fails_closed(self):
        br = self._bridge(["DU1234567"])

        def _raise():
            raise RuntimeError("managedAccounts unavailable")

        br.ib.managedAccounts = _raise
        self.assertFalse(br.connect(), "a managedAccounts() error must fail closed")
        self.assertFalse(br._conn_health.is_healthy)


# ── 5. The assertion does NOT enable live trading ─────────────────────────────
class TestDoesNotEnableLiveTrading(_ConnectBase):
    def test_connect_keeps_paper_only_invariants(self):
        br = self._bridge(["DU1234567"])
        self.assertTrue(br.connect())
        # Nothing about the guard flips any live-trading switch.
        self.assertFalse(bool(getattr(config, "COACH_LIVE_TRADING_ENABLED", False)))
        self.assertTrue(bool(getattr(config, "REQUIRE_PAPER_PORT", True)))
        self.assertEqual(int(config.IBKR_PORT), int(config.PAPER_IBKR_PORT))
        self.assertIsNone(getattr(config, "LIVE_ACCOUNT_ID", None))
        self.assertFalse(account_guard.live_mode_enabled(config))

    def test_capability_flag_true_but_live_trading_stays_disabled(self):
        # Phase 6.1 builds AND fail-closed tests the guard, so the live-readiness
        # capability flag is honestly True. That does NOT enable live trading: the
        # paper-port lock holds, live mode stays inert, and the config-only
        # readiness gates (market-data type, LIVE_ACCOUNT_ID) still fail.
        self.assertTrue(ibkr_bridge.SUPPORTS_ACCOUNT_TYPE_ASSERTION)
        self.assertFalse(account_guard.live_mode_enabled(config))
        self.assertTrue(bool(getattr(config, "REQUIRE_PAPER_PORT", True)))
        self.assertIsNone(getattr(config, "LIVE_ACCOUNT_ID", None))

    def test_account_guard_module_places_no_orders(self):
        import inspect
        src = inspect.getsource(account_guard)
        for forbidden in ("placeOrder", "MarketOrder", "LimitOrder",
                          "qualifyContracts", "import ib_insync"):
            self.assertNotIn(forbidden, src,
                             f"account_guard must not reference {forbidden}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
