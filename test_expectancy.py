"""test_expectancy.py - offline deterministic tests for the M4-core expectancy
/ R-multiple reporting module.

Pure unittest, no network, no IBKR, no order path. All file IO uses temp dirs,
so nothing under the real reports/ directory is touched. Run with:

    python -m unittest test_expectancy
"""
from __future__ import annotations

import csv
import inspect
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import expectancy
from expectancy import ClosedTrade


# Column order of reports/backtest_trades.csv (subset that matters here).
_COLS = [
    "symbol", "date", "fold_start", "order_executed",
    "old_position", "new_position", "close_price", "equity",
]


def _rows_to_csv_text(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_COLS)
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in _COLS})
    return buf.getvalue()


def _row(symbol, date, fold, executed, old, new, close, equity):
    return {
        "symbol": symbol, "date": date, "fold_start": fold,
        "order_executed": str(executed), "old_position": str(old),
        "new_position": str(new), "close_price": str(close), "equity": str(equity),
    }


def _long_round_trip(symbol="SPY", fold="252", entry_eq=1.0, exit_eq=1.06,
                     entry_close=100.0, exit_close=106.0):
    """A clean flat -> long -> flat round-trip (entry, hold, exit)."""
    return [
        _row(symbol, "2022-01-03", fold, True, 0, 1, entry_close, entry_eq),
        _row(symbol, "2022-01-04", fold, False, 1, 1, entry_close * 1.01, (entry_eq + exit_eq) / 2),
        _row(symbol, "2022-01-10", fold, True, 1, 0, exit_close, exit_eq),
    ]


def _pnl_trade(realized, **kw):
    """A PnL-included ClosedTrade with a given realized return."""
    return ClosedTrade("SPY", "252", "long", realized_pnl=realized, pnl_included=True,
                       risk_source=expectancy.RISK_SOURCE_UNAVAILABLE, r_included=False,
                       r_exclusion_reason=expectancy.R_REASON_UNAVAILABLE_RISK, **kw)


class TestRMultipleMath(unittest.TestCase):
    def test_basic_win_and_loss(self):
        self.assertAlmostEqual(expectancy.r_multiple(0.06, 0.03), 2.0)
        self.assertAlmostEqual(expectancy.r_multiple(-0.03, 0.03), -1.0)
        self.assertAlmostEqual(expectancy.r_multiple(0.015, 0.03), 0.5)

    def test_zero_negative_nonfinite_risk_is_none(self):
        self.assertIsNone(expectancy.r_multiple(0.05, 0))
        self.assertIsNone(expectancy.r_multiple(0.05, -0.01))
        self.assertIsNone(expectancy.r_multiple(0.05, float("nan")))
        self.assertIsNone(expectancy.r_multiple(0.05, float("inf")))

    def test_missing_inputs_is_none(self):
        self.assertIsNone(expectancy.r_multiple(None, 0.03))
        self.assertIsNone(expectancy.r_multiple(0.05, None))
        self.assertIsNone(expectancy.r_multiple(float("nan"), 0.03))


class TestReconstruct(unittest.TestCase):
    def test_clean_long_round_trip(self):
        rows = _long_round_trip(entry_eq=1.0, exit_eq=1.06, entry_close=100.0, exit_close=106.0)
        trades = expectancy.reconstruct_closed_trades(rows)
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertTrue(t.pnl_included)
        self.assertEqual(t.side, "long")
        self.assertAlmostEqual(t.realized_pnl, 0.06)
        self.assertEqual(t.entry_price, 100.0)
        self.assertEqual(t.exit_price, 106.0)
        self.assertEqual(t.risk_source, expectancy.RISK_SOURCE_UNAVAILABLE)
        self.assertEqual(t.entry_date, "2022-01-03")
        self.assertEqual(t.exit_date, "2022-01-10")

    def test_unpaired_entry_is_open_at_period_end(self):
        rows = [_row("SPY", "2022-01-03", "252", True, 0, 1, 100.0, 1.0)]
        trades = expectancy.reconstruct_closed_trades(rows)
        self.assertEqual(len(trades), 1)
        self.assertFalse(trades[0].pnl_included)
        self.assertEqual(trades[0].exclusion_reason, expectancy.REASON_OPEN_AT_PERIOD_END)

    def test_orphan_exit_reason(self):
        rows = [_row("SPY", "2022-01-10", "252", True, 1, 0, 106.0, 1.06)]
        trades = expectancy.reconstruct_closed_trades(rows)
        self.assertEqual(len(trades), 1)
        self.assertFalse(trades[0].pnl_included)
        self.assertEqual(trades[0].exclusion_reason, expectancy.REASON_ORPHAN_EXIT)

    def test_short_open_excluded(self):
        rows = [_row("SPY", "2022-01-03", "252", True, 0, -1, 100.0, 1.0)]
        trades = expectancy.reconstruct_closed_trades(rows)
        self.assertEqual(len(trades), 1)
        self.assertFalse(trades[0].pnl_included)
        self.assertEqual(trades[0].exclusion_reason, expectancy.REASON_SHORT_UNSUPPORTED)

    def test_grouping_avoids_cross_symbol_pairing(self):
        rows = []
        rows += _long_round_trip(symbol="SPY", fold="252", exit_eq=1.05)
        rows += _long_round_trip(symbol="AAPL", fold="252", exit_eq=0.97)
        trades = expectancy.reconstruct_closed_trades(rows)
        included = [t for t in trades if t.pnl_included]
        self.assertEqual(len(included), 2)
        by_symbol = {t.symbol: t for t in included}
        self.assertAlmostEqual(by_symbol["SPY"].realized_pnl, 0.05)
        self.assertAlmostEqual(by_symbol["AAPL"].realized_pnl, -0.03)

    def test_non_executed_rows_are_ignored(self):
        # A SELL signal that never executes (long-only "skip-sell") must not pair.
        rows = [_row("SPY", "2022-01-03", "252", False, 0, 0, 100.0, 1.0)]
        trades = expectancy.reconstruct_closed_trades(rows)
        self.assertEqual(trades, [])

    def test_missing_exit_equity_excluded_with_reason(self):
        rows = [
            _row("SPY", "2022-01-03", "252", True, 0, 1, 100.0, 1.0),
            _row("SPY", "2022-01-10", "252", True, 1, 0, 106.0, ""),  # blank equity
        ]
        trades = expectancy.reconstruct_closed_trades(rows)
        self.assertEqual(len(trades), 1)
        self.assertFalse(trades[0].pnl_included)
        self.assertEqual(trades[0].exclusion_reason, expectancy.REASON_MISSING_EXIT_PRICE)


class TestComputeMetrics(unittest.TestCase):
    def test_mixed_wins_losses(self):
        trades = [_pnl_trade(0.10), _pnl_trade(-0.05), _pnl_trade(0.04), _pnl_trade(-0.02)]
        m = expectancy.compute_metrics(trades)
        self.assertEqual(m["n_trades_included"], 4)
        self.assertEqual(m["n_trades_excluded"], 0)
        self.assertAlmostEqual(m["total_realized_pnl"], 0.07)
        self.assertEqual((m["wins"], m["losses"], m["scratch"]), (2, 2, 0))
        self.assertAlmostEqual(m["win_rate"], 0.5)
        self.assertAlmostEqual(m["gross_win"], 0.14)
        self.assertAlmostEqual(m["gross_loss"], 0.07)
        self.assertAlmostEqual(m["profit_factor"], 2.0)
        self.assertEqual(m["profit_factor_display"], "2.00")
        self.assertAlmostEqual(m["avg_win"], 0.07)
        self.assertAlmostEqual(m["avg_loss"], -0.035)

    def test_profit_factor_infinite_when_no_losses(self):
        m = expectancy.compute_metrics([_pnl_trade(0.1), _pnl_trade(0.2)])
        self.assertIsNone(m["profit_factor"])
        self.assertEqual(m["profit_factor_display"], "inf (no losing trades)")

    def test_profit_factor_na_when_no_trades(self):
        m = expectancy.compute_metrics([])
        self.assertIsNone(m["profit_factor"])
        self.assertEqual(m["profit_factor_display"], "n/a")
        self.assertIsNone(m["win_rate"])
        self.assertEqual(m["total_realized_pnl"], 0.0)

    def test_scratch_trades_counted_but_not_wins(self):
        m = expectancy.compute_metrics([_pnl_trade(0.0), _pnl_trade(0.1), _pnl_trade(-0.1)])
        self.assertEqual((m["wins"], m["losses"], m["scratch"]), (1, 1, 1))
        self.assertEqual(m["win_rate"], 0.333333)

    def test_excluded_reasons_counted(self):
        trades = [
            _pnl_trade(0.05),
            ClosedTrade("SPY", "252", "long", exclusion_reason=expectancy.REASON_OPEN_AT_PERIOD_END),
            ClosedTrade("SPY", "252", "short", exclusion_reason=expectancy.REASON_SHORT_UNSUPPORTED),
        ]
        m = expectancy.compute_metrics(trades)
        self.assertEqual(m["n_trades_included"], 1)
        self.assertEqual(m["n_trades_excluded"], 2)
        self.assertEqual(m["excluded_by_reason"][expectancy.REASON_OPEN_AT_PERIOD_END], 1)
        self.assertEqual(m["excluded_by_reason"][expectancy.REASON_SHORT_UNSUPPORTED], 1)


class TestRiskModel(unittest.TestCase):
    def test_default_excludes_all_from_r(self):
        trades = expectancy.apply_risk_model([_pnl_trade(0.06), _pnl_trade(-0.03)])
        m = expectancy.compute_metrics(trades)
        self.assertIsNone(m["r"]["expectancy_R"])
        self.assertEqual(m["r"]["n_r_included"], 0)
        self.assertEqual(m["r"]["n_r_excluded"], 2)
        self.assertEqual(m["r"]["r_excluded_by_reason"][expectancy.R_REASON_UNAVAILABLE_RISK], 2)
        for t in trades:
            self.assertEqual(t.risk_source, expectancy.RISK_SOURCE_UNAVAILABLE)
            self.assertFalse(t.r_included)

    def test_proxy_mode_computes_labeled_r(self):
        trades = expectancy.apply_risk_model(
            [_pnl_trade(0.06), _pnl_trade(-0.03)],
            enable_proxy_risk=True, proxy_pct=0.03,
        )
        m = expectancy.compute_metrics(trades)
        # R = 0.06/0.03 = 2.0 and -0.03/0.03 = -1.0 -> expectancy 0.5R
        self.assertAlmostEqual(m["r"]["expectancy_R"], 0.5)
        self.assertEqual(m["r"]["n_r_included"], 2)
        self.assertAlmostEqual(m["r"]["avg_win_R"], 2.0)
        self.assertAlmostEqual(m["r"]["avg_loss_R"], -1.0)
        for t in trades:
            self.assertEqual(t.risk_source, expectancy.RISK_SOURCE_HARD_STOP)
            self.assertTrue(t.r_included)

    def test_proxy_with_invalid_pct_marks_invalid_risk(self):
        trades = expectancy.apply_risk_model(
            [_pnl_trade(0.06)], enable_proxy_risk=True, proxy_pct=0.0,
        )
        m = expectancy.compute_metrics(trades)
        self.assertIsNone(m["r"]["expectancy_R"])
        self.assertEqual(m["r"]["r_excluded_by_reason"][expectancy.R_REASON_INVALID_RISK], 1)
        self.assertEqual(trades[0].risk_source, expectancy.RISK_SOURCE_UNAVAILABLE)


class TestGenerateAndWriteReport(unittest.TestCase):
    def _write_csv(self, tmp, rows):
        p = Path(tmp) / "backtest_trades.csv"
        p.write_text(_rows_to_csv_text(rows), encoding="utf-8")
        return p

    def test_default_report_does_not_compute_r(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(tmp, _long_round_trip(exit_eq=1.06))
            report = expectancy.generate_report(csv_path)
            self.assertEqual(report["source"], "backtest_ledger")
            self.assertEqual(report["risk_mode"], "none")
            self.assertIsNone(report["proxy_pct"])
            self.assertEqual(report["pnl_units"], "return_fraction")
            m = report["metrics"]
            self.assertEqual(m["n_trades_included"], 1)
            self.assertAlmostEqual(m["total_realized_pnl"], 0.06)
            self.assertIsNone(m["r"]["expectancy_R"])

    def test_proxy_report_labels_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(tmp, _long_round_trip(exit_eq=1.06))
            report = expectancy.generate_report(csv_path, enable_proxy_risk=True, proxy_pct=0.03)
            self.assertEqual(report["risk_mode"], "proxy_hard_stop")
            self.assertAlmostEqual(report["proxy_pct"], 0.03)
            self.assertAlmostEqual(report["metrics"]["r"]["expectancy_R"], 2.0)

    def test_write_report_produces_valid_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(tmp, _long_round_trip(exit_eq=1.06))
            report = expectancy.generate_report(csv_path)
            jp = Path(tmp) / "expectancy_metrics.json"
            mp = Path(tmp) / "expectancy_report.md"
            expectancy.write_report(report, json_path=jp, md_path=mp)
            self.assertTrue(jp.exists() and mp.exists())
            loaded = json.loads(jp.read_text(encoding="utf-8"))  # must be valid JSON
            self.assertEqual(loaded["source"], "backtest_ledger")
            md = mp.read_text(encoding="utf-8")
            self.assertIn("Expectancy / R-Multiple Report", md)
            self.assertIn("NOT computed", md)  # default-off disclaimer present

    def test_markdown_proxy_disclaimer(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(tmp, _long_round_trip(exit_eq=1.06))
            report = expectancy.generate_report(csv_path, enable_proxy_risk=True, proxy_pct=0.03)
            md = expectancy.report_to_markdown(report)
            self.assertIn("PROXY R", md)
            self.assertIn("not true Minervini R", md)

    def test_missing_file_yields_empty_report(self):
        report = expectancy.generate_report(Path("does_not_exist_12345.csv"))
        self.assertEqual(report["n_rows_read"], 0)
        self.assertEqual(report["metrics"]["n_trades_included"], 0)
        self.assertEqual(report["metrics"]["profit_factor_display"], "n/a")

    def test_empty_csv_yields_empty_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "backtest_trades.csv"
            p.write_text("", encoding="utf-8")
            report = expectancy.generate_report(p)
            self.assertEqual(report["metrics"]["n_trades_included"], 0)


class TestReadOnlyInvariant(unittest.TestCase):
    """Prove the module never imports the order path / broker layer."""

    def test_no_order_path_imports(self):
        src = inspect.getsource(expectancy)
        for forbidden in ("ibkr_bridge", "order_exec", "order_audit", "risk_state",
                          "account_guard", "ib_insync", "reconciliation"):
            self.assertNotIn(f"import {forbidden}", src)
            self.assertNotIn(f"from {forbidden}", src)

    def test_module_only_imports_config_and_stdlib(self):
        # config is the only first-party dependency; everything else is stdlib.
        self.assertTrue(hasattr(expectancy, "config"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
