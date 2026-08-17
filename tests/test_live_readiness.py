"""
test_live_readiness.py - Phase 0 tests for the live-trading readiness scaffolding.

Pure stdlib `unittest`, fully offline and deterministic (no IBKR, no network).
These exercise the new Phase-0 building blocks from
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md:

  * live_invariants  - pure safety-invariant checkers (unprotected longs, dup refs)
  * order_audit      - side-effect-only JSONL order-lifecycle log
  * ibkr_bridge      - live-readiness capability flags (Phase-2 flags now True;
                       Phases 3-6 still False until built)
  * main             - the read-only `live-readiness` scorecard (NOT READY today)

Run with:
    python -m unittest test_live_readiness -v
    python test_live_readiness.py
"""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config  # noqa: F401  (import smoke + ensures package importable)
import live_invariants
import order_audit


# -- 1. live_invariants.unprotected_longs ------------------------------------
class TestUnprotectedLongs(unittest.TestCase):
    def test_long_without_stop_is_unprotected(self):
        positions = [{"symbol": "AAPL", "qty": 10}]
        self.assertEqual(live_invariants.unprotected_longs(positions, []), ["AAPL"])

    def test_long_with_trailing_stop_is_protected(self):
        positions = [{"symbol": "AAPL", "qty": 10}]
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "TRAIL"}]
        self.assertEqual(live_invariants.unprotected_longs(positions, working), [])

    def test_long_with_stop_is_protected(self):
        positions = [{"symbol": "MSFT", "qty": 5}]
        working = [{"symbol": "MSFT", "action": "SELL", "order_type": "STP"}]
        self.assertEqual(live_invariants.unprotected_longs(positions, working), [])

    def test_sell_limit_take_profit_is_not_downside_protection(self):
        # A plain SELL LIMIT (take-profit) does NOT cap downside -> still unprotected.
        positions = [{"symbol": "NVDA", "qty": 3}]
        working = [{"symbol": "NVDA", "action": "SELL", "order_type": "LMT"}]
        self.assertEqual(live_invariants.unprotected_longs(positions, working), ["NVDA"])

    def test_flat_and_short_positions_ignored(self):
        positions = [{"symbol": "AAPL", "qty": 0}, {"symbol": "TSLA", "qty": -5}]
        self.assertEqual(live_invariants.unprotected_longs(positions, []), [])

    def test_stop_for_other_symbol_does_not_protect(self):
        positions = [{"symbol": "AAPL", "qty": 10}]
        working = [{"symbol": "MSFT", "action": "SELL", "order_type": "STP"}]
        self.assertEqual(live_invariants.unprotected_longs(positions, working), ["AAPL"])

    def test_buy_stop_does_not_count_as_long_protection(self):
        # A BUY stop is not a protective SELL for a long.
        positions = [{"symbol": "AAPL", "qty": 10}]
        working = [{"symbol": "AAPL", "action": "BUY", "order_type": "STP"}]
        self.assertEqual(live_invariants.unprotected_longs(positions, working), ["AAPL"])


# -- 2. live_invariants.duplicate_order_refs ---------------------------------
class TestDuplicateOrderRefs(unittest.TestCase):
    def test_detects_duplicate(self):
        refs = ["2026-06-16:AAPL:BUY", "2026-06-16:AAPL:BUY", "2026-06-16:MSFT:BUY"]
        self.assertEqual(live_invariants.duplicate_order_refs(refs), ["2026-06-16:AAPL:BUY"])

    def test_no_duplicates(self):
        self.assertEqual(live_invariants.duplicate_order_refs(["a", "b", "c"]), [])

    def test_blanks_and_none_ignored(self):
        self.assertEqual(live_invariants.duplicate_order_refs(["", None, "  "]), [])


# -- 3. order_audit (side-effect-only JSONL) ---------------------------------
class TestOrderAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "order_audit.jsonl"
        self._patch = mock.patch.object(order_audit, "AUDIT_FILE", self.path)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self.tmp.cleanup()

    def test_log_event_writes_one_jsonl_record(self):
        order_audit.log_event(order_audit.STAGE_SUBMIT, symbol="AAPL", action="BUY", qty=10)
        events = order_audit.read_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["stage"], "submit")
        self.assertEqual(events[0]["symbol"], "AAPL")
        self.assertEqual(events[0]["qty"], 10)
        self.assertIn("ts", events[0])

    def test_log_event_appends(self):
        order_audit.log_event("submit", symbol="A")
        order_audit.log_event("ack", symbol="A")
        self.assertEqual(len(order_audit.read_events()), 2)

    def test_log_event_never_raises_on_unserializable(self):
        order_audit.log_event("custom", obj=object())  # must not raise
        events = order_audit.read_events()
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0]["obj"], str)

    def test_read_events_empty_when_no_file(self):
        self.assertEqual(order_audit.read_events(), [])


# -- 4. ibkr_bridge live-readiness capability flags (Phase-0 baseline) -------
def _import_bridge():
    """Import ibkr_bridge with the asyncio loop prepared, or None if unavailable."""
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    try:
        import ibkr_bridge
        return ibkr_bridge
    except Exception:
        return None


class TestCapabilityFlagsBaseline(unittest.TestCase):
    # Every capability flag must exist on the bridge so the scorecard can read it.
    CAPABILITIES = [
        "SUPPORTS_FILL_VERIFICATION",
        "SUPPORTS_PARTIAL_FILL_HANDLING",
        "SUPPORTS_PROTECTIVE_CHILD_VERIFY",
        "SUPPORTS_SERVER_SIDE_GTC_STOP",
        "SUPPORTS_DAILY_LOSS_KILLSWITCH",
        "SUPPORTS_REALTIME_DATA_GUARD",
        "SUPPORTS_MARKET_HOURS_GATE",
        "SUPPORTS_STARTUP_RECONCILIATION",
        "SUPPORTS_ACCOUNT_TYPE_ASSERTION",
    ]
    # Phases 2, 3, 4, 5A and 6.1 are implemented AND covered (test_order_exec.py,
    # test_ibkr_fill_flow.py, test_risk_engine.py, test_phase3_protection.py,
    # test_data_integrity.py, test_reconciliation.py, test_account_guard.py), so
    # these flip True -- the only honest way to flip a flag.
    # See reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md.
    IMPLEMENTED_TRUE = [
        "SUPPORTS_FILL_VERIFICATION",        # Phase 2 (H4/H5)
        "SUPPORTS_PARTIAL_FILL_HANDLING",    # Phase 2 (H6)
        "SUPPORTS_PROTECTIVE_CHILD_VERIFY",  # Phase 2 (H7)
        "SUPPORTS_SERVER_SIDE_GTC_STOP",     # Phase 3 (C2/H19)
        "SUPPORTS_DAILY_LOSS_KILLSWITCH",    # Phase 3 (H1)
        "SUPPORTS_REALTIME_DATA_GUARD",      # Phase 4 (H11-H13/H16)
        "SUPPORTS_MARKET_HOURS_GATE",        # Phase 4 (H15)
        "SUPPORTS_STARTUP_RECONCILIATION",   # Phase 5A (H18)
        "SUPPORTS_ACCOUNT_TYPE_ASSERTION",   # Phase 6.1 (account_guard.assert_account)
    ]
    # Phase 6.1 ships the account-type assertion (built + fail-closed tested), so no
    # capability flag remains False. Live trading still stays disabled via the
    # config-only gates (IBKR_MARKET_DATA_TYPE, LIVE_ACCOUNT_ID), not via a flag.
    LATER_PHASE_FALSE = []

    def test_all_capability_flags_present(self):
        br = _import_bridge()
        if br is None:
            self.skipTest("ib_insync/ibkr_bridge unavailable")
        for name in self.CAPABILITIES:
            self.assertTrue(hasattr(br, name), f"missing capability flag {name}")

    def test_implemented_flags_true_because_covered(self):
        br = _import_bridge()
        if br is None:
            self.skipTest("ib_insync/ibkr_bridge unavailable")
        for name in self.IMPLEMENTED_TRUE:
            self.assertTrue(getattr(br, name), f"{name} must be True once its phase ships")

    def test_later_phase_flags_still_false(self):
        br = _import_bridge()
        if br is None:
            self.skipTest("ib_insync/ibkr_bridge unavailable")
        for name in self.LATER_PHASE_FALSE:
            self.assertFalse(getattr(br, name), f"{name} must be False until its phase ships")

    def test_flatten_all_method_exists(self):
        br = _import_bridge()
        if br is None:
            self.skipTest("ib_insync/ibkr_bridge unavailable")
        self.assertTrue(hasattr(br.IBKRBridge, "flatten_all"))


# -- 5. live-readiness scorecard (read-only; NOT READY today) ----------------
class TestReadinessScorecard(unittest.TestCase):
    def test_scorecard_reports_failures_with_no_bridge(self):
        import main
        gates = main._collect_readiness_gates(None)
        self.assertGreater(len(gates), 0)
        fails = [g for g in gates if g["status"] == "FAIL"]
        self.assertGreater(len(fails), 0, "Phase 0 baseline must have failing gates")
        # The historical-pricing-disabled gate is already satisfied by config.
        hist = [g for g in gates if "Historical-close" in g["title"]]
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["status"], "PASS")
        # There must be manual gates that need operator sign-off.
        self.assertTrue(any(g["status"] == "MANUAL" for g in gates))

    def test_cmd_live_readiness_exit_nonzero_today(self):
        import main
        args = mock.Mock(connect=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main.cmd_live_readiness(args)
        self.assertEqual(rc, 1, "live-readiness must report NOT READY in Phase 0")


# -- 6. import smoke ---------------------------------------------------------
class TestImportSmoke(unittest.TestCase):
    def test_new_modules_import(self):
        import order_audit  # noqa: F401
        import live_invariants  # noqa: F401
        self.assertTrue(hasattr(order_audit, "log_event"))
        self.assertTrue(hasattr(live_invariants, "unprotected_longs"))

    def test_main_exposes_new_commands(self):
        import main
        self.assertTrue(hasattr(main, "cmd_live_readiness"))
        self.assertTrue(hasattr(main, "cmd_panic_flatten"))
        self.assertTrue(hasattr(main, "_collect_readiness_gates"))


# -- 7. connection-gated integration (skipped unless paper TWS is reachable) -
@unittest.skipUnless(
    os.environ.get("RUN_IBKR_INTEGRATION") == "1",
    "set RUN_IBKR_INTEGRATION=1 with a paper TWS running to exercise this",
)
class TestPaperIntegration(unittest.TestCase):
    def test_live_readiness_connect_reports_unprotected(self):
        import main
        args = mock.Mock(connect=True)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main.cmd_live_readiness(args)
        # Still NOT READY (capabilities unimplemented), but the broker gate ran.
        self.assertIn(rc, (0, 1))

    def test_panic_flatten_dry_run_places_nothing(self):
        import main
        args = mock.Mock(confirm=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main.cmd_panic_flatten(args)
        self.assertEqual(rc, 0)


# -- 8. Phase 6.1 account-type assertion: built, but live-readiness STILL NOT
#       READY (LIVE_ACCOUNT_ID is None and IBKR_MARKET_DATA_TYPE is still 3) -----
class TestPhase61AccountTypeReadiness(unittest.TestCase):
    def test_account_guard_module_imports(self):
        import account_guard
        self.assertTrue(hasattr(account_guard, "assert_account"))
        self.assertTrue(hasattr(account_guard, "live_mode_enabled"))

    def test_live_mode_is_inert_for_shipped_config(self):
        import account_guard
        # The shipped paper config must never resolve to live mode.
        self.assertFalse(account_guard.live_mode_enabled(config))
        self.assertIsNone(getattr(config, "LIVE_ACCOUNT_ID", None))

    def _gate(self, title_fragment):
        import main
        # config-only gates (market-data type, LIVE_ACCOUNT_ID) do not depend on
        # the bridge module, so None keeps this deterministic + offline.
        gates = main._collect_readiness_gates(None)
        matches = [g for g in gates if title_fragment in g["title"]]
        self.assertEqual(len(matches), 1, f"expected exactly one gate for {title_fragment!r}")
        return matches[0]

    def test_live_account_id_gate_fails_because_none(self):
        g = self._gate("Explicit LIVE_ACCOUNT_ID configured")
        self.assertEqual(g["status"], "FAIL")

    def test_market_data_type_gate_fails_because_delayed(self):
        # IBKR_MARKET_DATA_TYPE is still 3 (delayed) -> the live real-time gate fails.
        self.assertEqual(int(getattr(config, "IBKR_MARKET_DATA_TYPE", 3)), 3)
        g = self._gate("Real-time market-data type")
        self.assertEqual(g["status"], "FAIL")

    def test_live_readiness_still_not_ready(self):
        import main
        args = mock.Mock(connect=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main.cmd_live_readiness(args)
        self.assertEqual(rc, 1, "live-readiness must STILL report NOT READY after Phase 6.1")

    def test_account_type_capability_flag_true_but_still_not_ready(self):
        # Phase 6.1 ships the account-type assertion (wired into connect() as
        # SAFETY GATE #2, fail-closed tested), so the capability flag is honestly
        # True. live-readiness is STILL NOT READY -- see test_live_readiness_still_
        # not_ready -- because the config-only gates (market-data type,
        # LIVE_ACCOUNT_ID) still FAIL, NOT because of this flag.
        br = _import_bridge()
        if br is None:
            self.skipTest("ib_insync/ibkr_bridge unavailable")
        self.assertTrue(br.SUPPORTS_ACCOUNT_TYPE_ASSERTION)


if __name__ == "__main__":
    unittest.main(verbosity=2)
