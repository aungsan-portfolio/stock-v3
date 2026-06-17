"""
test_scheduler_runner.py - Phase 5B-1 supervised scheduler / market-hours runner.

Fully offline and deterministic (no IBKR, no network, no wall clock): every test
feeds a fixed US/Eastern ``now`` into the PURE decision layer and injects a fake
dispatch into the runner, so no order path is ever touched. These prove the
Phase-5B-1 contract from reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md (task 5.1):

  * the scheduler BLOCKS outside US regular trading hours (weekend / after close);
  * it ALLOWS inside regular hours on a trading day;
  * it respects the SCHEDULER_ENABLED master switch;
  * it does NOT bypass paper-only safety (a live-trading config blocks it, and
    --execute cannot place orders while SCHEDULER_DRY_RUN_DEFAULT ships True);
  * it runs in dry-run / plan mode WITHOUT placing orders, and never loops;
  * it adds NO new live-readiness capability flag, and live-readiness is still
    NOT READY (Phase 6 gates remain False).

Run with:
    python -m unittest test_scheduler_runner -v
    python test_scheduler_runner.py
"""
import contextlib
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import config
import order_audit
import scheduler_runner as sr


# 2026-06-17 is a Wednesday and a normal trading day (Juneteenth is observed
# Fri 2026-06-19, not the 17th), so these are deterministic calendar anchors.
RTH_OPEN = datetime(2026, 6, 17, 10, 0)      # Wed 10:00 ET -> regular hours
BEFORE_OPEN = datetime(2026, 6, 17, 8, 0)    # Wed 08:00 ET -> before_open
AFTER_CLOSE = datetime(2026, 6, 17, 17, 0)   # Wed 17:00 ET -> after_close
SATURDAY = datetime(2026, 6, 20, 10, 0)      # Sat 10:00 ET -> weekend


class _Dispatch:
    """Records every dispatch call so a test can assert IF and HOW the underlying
    command was invoked. Returns a fixed exit code; never places an order."""

    def __init__(self, rc: int = 0):
        self.calls = []
        self.rc = rc

    def __call__(self, dry_run):
        self.calls.append(bool(dry_run))
        return self.rc


def _paper_only_config():
    """Patch config to an unambiguously paper-only, scheduler-enabled state so a
    single dimension can be varied per test. Returns a list of started patchers
    the caller must stop (done in tearDown)."""
    patchers = [
        mock.patch.object(config, "COACH_LIVE_TRADING_ENABLED", False),
        mock.patch.object(config, "REQUIRE_PAPER_PORT", True),
        mock.patch.object(config, "IBKR_PORT", 7497),
        mock.patch.object(config, "PAPER_IBKR_PORT", 7497),
        mock.patch.object(config, "ALLOW_SHORT", False),
        mock.patch.object(config, "ALLOW_HISTORICAL_PRICE_FOR_ORDERS", False),
        mock.patch.object(config, "SCHEDULER_ENABLED", True),
        mock.patch.object(config, "SCHEDULER_REQUIRE_RTH", True),
        mock.patch.object(config, "SCHEDULER_DRY_RUN_DEFAULT", True),
    ]
    for p in patchers:
        p.start()
    return patchers


class _SchedulerTestBase(unittest.TestCase):
    def setUp(self):
        self._patchers = _paper_only_config()
        # Redirect the audit log to a temp file so tests never touch the real one
        # (and so we can assert a decision was actually audited).
        self._tmp = tempfile.TemporaryDirectory()
        self.audit_path = Path(self._tmp.name) / "order_audit.jsonl"
        self._audit_patch = mock.patch.object(order_audit, "AUDIT_FILE", self.audit_path)
        self._audit_patch.start()

    def tearDown(self):
        self._audit_patch.stop()
        self._tmp.cleanup()
        for p in reversed(self._patchers):
            p.stop()


# -- 1. Pure decision: regular-hours gate ------------------------------------
class TestEvaluateMarketHours(_SchedulerTestBase):
    def test_blocks_on_weekend(self):
        d = sr.evaluate(SATURDAY)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "weekend")

    def test_blocks_after_close(self):
        d = sr.evaluate(AFTER_CLOSE)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "after_close")

    def test_blocks_before_open(self):
        d = sr.evaluate(BEFORE_OPEN)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, "before_open")

    def test_allows_inside_regular_hours(self):
        d = sr.evaluate(RTH_OPEN)
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, sr.REASON_ALLOWED)
        self.assertEqual(d.rth_reason, "open")

    def test_require_rth_false_allows_outside_hours(self):
        # When RTH is not required, an after-hours run is allowed (paper practice).
        d = sr.evaluate(AFTER_CLOSE, require_rth=False)
        self.assertTrue(d.allowed)

    def test_clock_unresolved_fails_closed(self):
        d = sr.evaluate(None)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, sr.REASON_CLOCK_ERROR)


# -- 2. Pure decision: disabled master switch --------------------------------
class TestEvaluateDisabled(_SchedulerTestBase):
    def test_disabled_blocks_even_in_hours(self):
        with mock.patch.object(config, "SCHEDULER_ENABLED", False):
            d = sr.evaluate(RTH_OPEN)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, sr.REASON_DISABLED)

    def test_disabled_via_explicit_arg(self):
        d = sr.evaluate(RTH_OPEN, enabled=False)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, sr.REASON_DISABLED)


# -- 3. Pure decision: paper-only safety is never bypassed -------------------
class TestEvaluatePaperSafety(_SchedulerTestBase):
    def test_live_trading_enabled_blocks(self):
        with mock.patch.object(config, "COACH_LIVE_TRADING_ENABLED", True):
            d = sr.evaluate(RTH_OPEN)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, sr.REASON_PAPER_SAFETY)
        self.assertFalse(d.paper_safe)

    def test_non_paper_port_blocks(self):
        with mock.patch.object(config, "IBKR_PORT", 7496):
            d = sr.evaluate(RTH_OPEN)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, sr.REASON_PAPER_SAFETY)

    def test_require_paper_port_off_blocks(self):
        with mock.patch.object(config, "REQUIRE_PAPER_PORT", False):
            d = sr.evaluate(RTH_OPEN)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, sr.REASON_PAPER_SAFETY)

    def test_historical_pricing_blocks(self):
        with mock.patch.object(config, "ALLOW_HISTORICAL_PRICE_FOR_ORDERS", True):
            d = sr.evaluate(RTH_OPEN)
        self.assertFalse(d.allowed)
        self.assertEqual(d.reason, sr.REASON_PAPER_SAFETY)

    def test_paper_safety_outranks_everything(self):
        # Even disabled + outside hours, paper-safety is reported first (highest
        # priority, fail-closed) so the operator sees the real blocker.
        with mock.patch.object(config, "COACH_LIVE_TRADING_ENABLED", True), \
             mock.patch.object(config, "SCHEDULER_ENABLED", False):
            d = sr.evaluate(SATURDAY)
        self.assertEqual(d.reason, sr.REASON_PAPER_SAFETY)


# -- 4. run_scheduled dispatch: blocked never dispatches ---------------------
class TestRunScheduledBlocked(_SchedulerTestBase):
    def test_weekend_does_not_dispatch(self):
        disp = _Dispatch()
        result = sr.run_scheduled(now_et=SATURDAY, dispatch=disp)
        self.assertFalse(result["ran"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(disp.calls, [])  # NOTHING ran

    def test_disabled_does_not_dispatch(self):
        disp = _Dispatch()
        with mock.patch.object(config, "SCHEDULER_ENABLED", False):
            result = sr.run_scheduled(now_et=RTH_OPEN, dispatch=disp)
        self.assertFalse(result["ran"])
        self.assertEqual(disp.calls, [])

    def test_paper_safety_violation_does_not_dispatch(self):
        disp = _Dispatch()
        with mock.patch.object(config, "COACH_LIVE_TRADING_ENABLED", True):
            result = sr.run_scheduled(now_et=RTH_OPEN, dispatch=disp)
        self.assertFalse(result["ran"])
        self.assertEqual(result["decision"].reason, sr.REASON_PAPER_SAFETY)
        self.assertEqual(disp.calls, [])

    def test_blocked_decision_is_audited(self):
        sr.run_scheduled(now_et=SATURDAY, dispatch=_Dispatch())
        events = order_audit.read_events()
        sched = [e for e in events if e.get("stage") == "schedule"]
        self.assertTrue(sched, "scheduler decision must be audited")
        self.assertEqual(sched[-1]["decision"], "blocked")
        self.assertEqual(sched[-1]["reason"], "weekend")


# -- 5. run_scheduled dispatch: allowed -> plan/dry-run, no orders -----------
class TestRunScheduledAllowed(_SchedulerTestBase):
    def test_allowed_dispatches_in_dry_run_by_default(self):
        disp = _Dispatch()
        result = sr.run_scheduled(now_et=RTH_OPEN, dispatch=disp)
        self.assertTrue(result["ran"])
        self.assertTrue(result["dry_run"])          # plan mode, no orders
        self.assertEqual(disp.calls, [True])        # dispatched once, dry-run=True

    def test_execute_stays_dry_run_when_config_default_true(self):
        # --execute must NOT place orders while SCHEDULER_DRY_RUN_DEFAULT ships True.
        disp = _Dispatch()
        result = sr.run_scheduled(now_et=RTH_OPEN, execute=True, dispatch=disp)
        self.assertTrue(result["ran"])
        self.assertTrue(result["dry_run"])
        self.assertEqual(disp.calls, [True])

    def test_execute_places_only_when_config_opts_in(self):
        # The ONLY way to reach the (paper) order path: --execute AND an operator
        # deliberately flipping SCHEDULER_DRY_RUN_DEFAULT False. Still paper-locked.
        disp = _Dispatch()
        with mock.patch.object(config, "SCHEDULER_DRY_RUN_DEFAULT", False):
            result = sr.run_scheduled(now_et=RTH_OPEN, execute=True, dispatch=disp)
        self.assertTrue(result["ran"])
        self.assertFalse(result["dry_run"])
        self.assertEqual(disp.calls, [False])

    def test_dispatch_called_at_most_once_no_loop(self):
        disp = _Dispatch()
        sr.run_scheduled(now_et=RTH_OPEN, dispatch=disp)
        self.assertEqual(len(disp.calls), 1)  # single-shot; never loops

    def test_allowed_dispatch_is_audited(self):
        sr.run_scheduled(now_et=RTH_OPEN, dispatch=_Dispatch())
        events = order_audit.read_events()
        stages = [e.get("decision") for e in events if e.get("stage") == "schedule"]
        self.assertIn("allowed", stages)
        self.assertIn("dispatched", stages)

    def test_underlying_exit_code_is_propagated(self):
        disp = _Dispatch(rc=7)
        result = sr.run_scheduled(now_et=RTH_OPEN, dispatch=disp)
        self.assertEqual(result["exit_code"], 7)


# -- 6. No new capability flag (Phase 5B adds none) --------------------------
class TestNoNewCapabilityFlag(unittest.TestCase):
    def test_scheduler_module_exposes_no_supports_flag(self):
        flags = [n for n in dir(sr) if n.startswith("SUPPORTS_")]
        self.assertEqual(flags, [], "Phase 5B must not add a capability flag")


# -- 7. Live-readiness is still NOT READY ------------------------------------
class TestLiveReadinessStillNotReady(unittest.TestCase):
    def test_live_readiness_reports_not_ready(self):
        import main
        args = mock.Mock(connect=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rc = main.cmd_live_readiness(args)
        self.assertEqual(rc, 1, "live-readiness must still report NOT READY")

    def test_account_type_assertion_flag_still_false(self):
        try:
            import ibkr_bridge
        except Exception:
            self.skipTest("ib_insync/ibkr_bridge unavailable")
        # Phase 6 gate stays False; Phase 5A reconciliation flag stays True.
        self.assertFalse(ibkr_bridge.SUPPORTS_ACCOUNT_TYPE_ASSERTION)
        self.assertTrue(ibkr_bridge.SUPPORTS_STARTUP_RECONCILIATION)


# -- 8. command wiring smoke -------------------------------------------------
class TestCommandWiring(unittest.TestCase):
    def test_main_exposes_run_scheduled_command(self):
        import main
        self.assertTrue(hasattr(main, "cmd_run_scheduled"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
