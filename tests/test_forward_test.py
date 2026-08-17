"""test_forward_test.py - offline deterministic tests for the read-only forward
paper-test report (forward_test.py).

Pure unittest, no network, no IBKR, no order path. File IO uses temp dirs. The
headline metrics are produced by the SAME expectancy.compute_metrics as the
backtest report, so a reconcile check guarantees the two stay consistent. Run:

    python -m unittest test_forward_test
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import expectancy
import forward_test
import paper_ledger


def _trade_events(symbol, entry_ts, entry_px, stop_px, exit_ts, exit_px,
                  qty=10, exec_id="X", session=None):
    return [
        {"event": "entry", "ts": entry_ts, "symbol": symbol, "qty": qty,
         "entry_avg_price": entry_px, "confidence": 0.72, "reason": "RF..",
         "session": session},
        {"event": "stop", "ts": entry_ts, "symbol": symbol, "qty": qty,
         "stop_price": stop_px, "session": session},
        {"event": "exit", "ts": exit_ts, "symbol": symbol, "qty": qty,
         "exit_avg_price": exit_px, "exit_kind": "signal_close",
         "exec_id": exec_id, "session": session},
    ]


def _write(path, events):
    with open(path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


class TestPureAggregates(unittest.TestCase):
    def test_drawdown(self):
        dd_usd, dd_pct = forward_test._drawdown([10.0, -30.0, 5.0], base=100.0)
        self.assertAlmostEqual(dd_usd, -30.0, places=6)
        self.assertAlmostEqual(dd_pct, -30.0 / 110.0, places=6)

    def test_drawdown_no_loss(self):
        dd_usd, dd_pct = forward_test._drawdown([10.0, 5.0], base=100.0)
        self.assertEqual(dd_usd, 0.0)
        self.assertEqual(dd_pct, 0.0)

    def test_sharpe(self):
        self.assertGreater(forward_test._sharpe([0.01, 0.02, 0.015]), 0.0)
        self.assertEqual(forward_test._sharpe([0.01]), 0.0)
        self.assertEqual(forward_test._sharpe([]), 0.0)

    def test_confidence_band(self):
        self.assertEqual(forward_test._confidence_band(0.72), "0.70-0.80")
        self.assertEqual(forward_test._confidence_band(None), "unknown")


class TestGenerateReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal = Path(self.tmp.name) / "paper_trades.jsonl"
        # A win (+100, R=+2) and a loss (-100, R=-2).
        self.events = (
            _trade_events("AAPL", "2026-06-01T10:00:00", 100.0, 95.0,
                          "2026-06-05T15:00:00", 110.0, exec_id="A", session="S1")
            + _trade_events("MSFT", "2026-06-02T10:00:00", 50.0, 45.0,
                            "2026-06-10T15:00:00", 40.0, exec_id="B", session="S2")
        )
        _write(self.journal, self.events)

    def test_headline_reconciles_with_shared_metric_engine(self):
        report = forward_test.generate_report(self.journal, start_equity=100_000.0)
        m = report["metrics"]
        # Independent recompute through the shared engine.
        trades = paper_ledger.reconstruct_paper_trades(
            paper_ledger.load_journal(self.journal))
        expect = expectancy.compute_metrics([t.to_closed_trade() for t in trades])
        self.assertEqual(m, expect)
        # Hand-calculated truths.
        self.assertEqual(m["n_trades_included"], 2)
        self.assertEqual((m["wins"], m["losses"]), (1, 1))
        self.assertAlmostEqual(m["win_rate"], 0.5)
        self.assertAlmostEqual(m["total_realized_pnl"], 0.0)
        self.assertEqual(m["profit_factor_display"], "1.00")
        self.assertAlmostEqual(m["r"]["expectancy_R"], 0.0)
        self.assertEqual(m["r"]["n_r_included"], 2)
        # R risk source label is the unambiguous broker one.
        self.assertIn("broker_protective_stop", m["r"]["risk_source_counts"])

    def test_drawdown_and_sharpe(self):
        report = forward_test.generate_report(self.journal, start_equity=100_000.0)
        self.assertAlmostEqual(report["drawdown"]["max_drawdown_usd"], -100.0)
        self.assertLess(report["drawdown"]["max_drawdown_pct"], 0.0)
        # Symmetric +0.001 / -0.001 daily returns -> mean 0 -> Sharpe 0.
        self.assertEqual(report["sharpe"], 0.0)

    def test_daily_breakdown(self):
        report = forward_test.generate_report(self.journal, start_equity=100_000.0)
        days = {d["date"]: d for d in report["daily"]}
        self.assertEqual(days["2026-06-05"]["realized_pnl"], 100.0)
        self.assertEqual(days["2026-06-10"]["realized_pnl"], -100.0)
        # Default cap (150) not breached by a -100 day.
        self.assertFalse(days["2026-06-10"]["daily_loss_breach"])

    def test_daily_loss_breach_flag(self):
        # A bigger loss day that breaches a patched cap.
        _write(self.journal,
               _trade_events("AAPL", "2026-06-01T10:00:00", 50.0, 45.0,
                             "2026-06-10T15:00:00", 30.0, exec_id="C"))  # -200
        with mock.patch.object(config, "MAX_DAILY_LOSS_USD", 150.0):
            report = forward_test.generate_report(self.journal, start_equity=100_000.0)
        day = {d["date"]: d for d in report["daily"]}["2026-06-10"]
        self.assertEqual(day["realized_pnl"], -200.0)
        self.assertTrue(day["daily_loss_breach"])

    def test_since_window_is_consistent(self):
        report = forward_test.generate_report(
            self.journal, since="2026-06-08", start_equity=100_000.0)
        m = report["metrics"]
        # Only the MSFT loss (exit 06-10) is in-window; AAPL (06-05) drops out.
        self.assertEqual(m["n_trades_included"], 1)
        self.assertEqual(m["n_trades_excluded"], 0)
        self.assertAlmostEqual(m["total_realized_pnl"], -100.0)

    def test_session_filter(self):
        report = forward_test.generate_report(self.journal, session="S2",
                                              start_equity=100_000.0)
        self.assertEqual(report["metrics"]["n_trades_included"], 1)
        self.assertAlmostEqual(report["metrics"]["total_realized_pnl"], -100.0)

    def test_open_position_listed_and_excluded(self):
        _write(self.journal, [
            {"event": "entry", "ts": "2026-06-01T10:00:00", "symbol": "NVDA",
             "qty": 3, "entry_avg_price": 120.0},
        ])
        report = forward_test.generate_report(self.journal, start_equity=100_000.0)
        self.assertEqual(report["metrics"]["n_trades_included"], 0)
        self.assertEqual(len(report["open_positions"]), 1)
        self.assertEqual(report["open_positions"][0]["symbol"], "NVDA")

    def test_empty_journal(self):
        missing = Path(self.tmp.name) / "does_not_exist.jsonl"
        report = forward_test.generate_report(missing)
        self.assertEqual(report["n_events_read"], 0)
        self.assertEqual(report["metrics"]["n_trades_included"], 0)
        self.assertEqual(report["daily"], [])
        self.assertEqual(report["drawdown"]["max_drawdown_usd"], 0.0)

    def test_write_report_artifacts(self):
        report = forward_test.generate_report(self.journal, start_equity=100_000.0)
        jp = Path(self.tmp.name) / "forward_test_metrics.json"
        mp = Path(self.tmp.name) / "forward_test_report.md"
        forward_test.write_report(report, json_path=jp, md_path=mp)
        self.assertTrue(jp.exists() and mp.exists())
        loaded = json.loads(jp.read_text(encoding="utf-8"))
        self.assertEqual(loaded["source"], "paper_journal")
        self.assertIn("Forward Paper-Test Report", mp.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
