"""test_forward_test_safety.py - guardrails proving the forward paper-test
feature is PAPER-ONLY, READ-ONLY, and never touches the live/order path.

Pure unittest, no network, no IBKR. Asserts:
  * forward_test / paper_ledger import WITHOUT pulling ib_insync or ibkr_bridge
    (verified both in-process and in a fresh interpreter);
  * the journal writers are never-raise even when the file write fails;
  * the R risk_source label is the unambiguous broker one (not Minervini);
  * the module sources contain no order-submission / live-switch tokens;
  * the live trading gates remain set to their shipped paper-only values.

    python -m unittest test_forward_test_safety
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import forward_test
import main
import paper_ledger

# In-process: importing the report modules must not have pulled the IBKR path.
_IN_PROCESS_MODULES = set(sys.modules)


class TestOfflineImports(unittest.TestCase):
    def test_no_ibkr_modules_in_process(self):
        for banned in ("ib_insync", "ibkr_bridge"):
            self.assertNotIn(banned, _IN_PROCESS_MODULES,
                             f"{banned} must not be imported by the report path")

    def test_no_ibkr_modules_fresh_interpreter(self):
        code = (
            "import forward_test, paper_ledger, sys\n"
            "bad = [m for m in ('ib_insync', 'ibkr_bridge') if m in sys.modules]\n"
            "assert not bad, bad\n"
            "print('OK')\n"
        )
        proc = subprocess.run([sys.executable, "-c", code],
                              cwd=str(Path(__file__).resolve().parent),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK", proc.stdout)

    def test_generate_report_runs_without_ibkr(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = forward_test.generate_report(Path(tmp) / "none.jsonl")
        self.assertEqual(report["metrics"]["n_trades_included"], 0)
        self.assertNotIn("ib_insync", sys.modules)
        self.assertNotIn("ibkr_bridge", sys.modules)


class TestNeverRaise(unittest.TestCase):
    def test_writers_never_raise_when_path_breaks(self):
        with mock.patch.object(paper_ledger, "_journal_path", side_effect=OSError("x")):
            paper_ledger.record_entry(
                type("S", (), {"symbol": "AAPL", "action": "BUY"})(),
                type("R", (), {"outcome": "FILLED", "filled": 1.0,
                               "avg_fill_price": 1.0, "aborted": False})())
            paper_ledger.record_stop("AAPL", 1, 1.0)
            paper_ledger.record_exit("AAPL", exit_kind="signal_close", qty=1,
                                     exit_avg_price=1.0, exec_id="Z")
        # Reaching here without an exception is the assertion.


class TestRiskLabelIsUnambiguous(unittest.TestCase):
    def test_label_is_broker_protective_stop(self):
        self.assertEqual(paper_ledger.RISK_SOURCE_PROTECTIVE_STOP,
                         "broker_protective_stop")

    def test_reconstructed_trade_uses_broker_label_not_minervini(self):
        events = [
            {"event": "entry", "ts": "2026-06-01T10:00:00", "symbol": "AAPL",
             "qty": 10, "entry_avg_price": 100.0},
            {"event": "stop", "ts": "2026-06-01T10:00:01", "symbol": "AAPL",
             "qty": 10, "stop_price": 95.0},
            {"event": "exit", "ts": "2026-06-05T15:00:00", "symbol": "AAPL",
             "qty": 10, "exit_avg_price": 110.0, "exec_id": "E1"},
        ]
        t = paper_ledger.reconstruct_paper_trades(events)[0]
        self.assertEqual(t.risk_source, "broker_protective_stop")
        self.assertNotIn("minervini", t.risk_source.lower())


class TestNoOrderPathTokens(unittest.TestCase):
    def test_sources_have_no_order_or_live_switch_tokens(self):
        for mod in (forward_test, paper_ledger):
            src = Path(mod.__file__).read_text(encoding="utf-8")
            for token in ("placeOrder", "COACH_LIVE_TRADING_ENABLED",
                          "REQUIRE_PAPER_PORT", "LIVE_ACCOUNT_ID"):
                self.assertNotIn(token, src,
                                 f"{mod.__name__} must not reference {token}")


class TestLiveGatesUntouched(unittest.TestCase):
    def test_paper_only_config_intact(self):
        self.assertIs(config.COACH_LIVE_TRADING_ENABLED, False)
        self.assertIs(config.REQUIRE_PAPER_PORT, True)
        self.assertIsNone(getattr(config, "LIVE_ACCOUNT_ID", None))
        self.assertEqual(int(config.IBKR_PORT), int(config.PAPER_IBKR_PORT))

    def test_forward_test_paths_live_under_reports(self):
        for attr in ("FORWARD_TEST_JOURNAL", "FORWARD_TEST_REPORT_JSON",
                     "FORWARD_TEST_REPORT_MD"):
            p = Path(getattr(config, attr))
            self.assertEqual(p.parent, Path(config.REPORTS_DIR))


class TestConnectedPaperCommandCoverage(unittest.TestCase):
    def _fake_bridge(self):
        bridge = mock.Mock()
        bridge.get_cash.return_value = 100_000.0
        bridge.ib.positions.return_value = []
        bridge.working_order_symbols.return_value = set()
        bridge.has_working_order.return_value = False
        bridge.execute_signal.return_value = False
        bridge.disconnect.return_value = None
        return bridge

    def test_paper_coach_runs_shared_startup_checks(self):
        signal = type("S", (), {
            "symbol": "AAPL", "action": "BUY", "confidence": 0.9,
            "rf_score": 0.8, "lstm_score": 0.7, "tech_score": 0.6,
            "price": 100.0, "reason": "test",
        })()
        bridge = self._fake_bridge()
        args = type("A", (), {"confirm": True, "chart_checked": True})()
        with mock.patch.object(main.Predictor, "predict_all", return_value=[signal]), \
             mock.patch.object(main, "_connect_bridge", return_value=(bridge, 0)), \
             mock.patch.object(main, "_run_paper_startup_checks", return_value={
                 "snapshot": {"open_positions": [], "n_working_orders": 0,
                               "clean": True, "duplicate_refs": [], "orphan_exits": []},
                  "protect": {"repaired": [], "failed": []},
              }) as startup, \
             mock.patch.object(main, "build_trade_preview", return_value={
                 "tradeable": True,
                 "rr_2r": 2.0,
                 "risk_per_share": 1.0,
                 "suggested_shares_by_risk": 1,
             }), \
             mock.patch.object(main, "print_trade_lesson"), \
             mock.patch.object(main, "print_trade_preview"), \
             mock.patch.object(main, "write_trade_note"):
            self.assertEqual(main.cmd_paper_coach(args), 0)
        startup.assert_called_once_with(bridge)

    def test_daily_coach_execute_runs_shared_startup_checks(self):
        candidate = type("S", (), {
            "symbol": "AAPL", "action": "BUY", "confidence": 0.9,
            "rf_score": 0.8, "lstm_score": 0.7, "tech_score": 0.6,
            "price": 100.0, "reason": "test",
        })()
        bridge = self._fake_bridge()
        with mock.patch.object(main, "assert_paper_trading_only"), \
             mock.patch.object(main, "_connect_bridge", return_value=(bridge, 0)), \
             mock.patch.object(main, "_run_paper_startup_checks", return_value={
                 "snapshot": {"open_positions": [], "n_working_orders": 0,
                              "clean": True, "duplicate_refs": [], "orphan_exits": []},
                 "protect": {"repaired": [], "failed": []},
             }) as startup, \
             mock.patch.object(main, "build_trade_preview", return_value={"tradeable": True}), \
             mock.patch.object(main, "assess_chart_status", return_value=("ok", "")), \
             mock.patch.object(main, "evaluate_daily_candidate", return_value={
                 "symbol": "AAPL", "accepted": False, "skip_reason": "test skip",
             }), \
             mock.patch.object(main, "print_daily_candidate"), \
             mock.patch.object(main, "write_trade_note"):
            self.assertEqual(main._daily_coach_execute([candidate], 1), 0)
        startup.assert_called_once_with(bridge)


if __name__ == "__main__":
    unittest.main(verbosity=2)
