"""test_backtest_summary_reconcile.py - prove `backtest-summary` and
`expectancy-report` reconstruct closed trades identically.

Regression guard for the bug where ``cmd_backtest_summary`` paired entries and
exits with a GLOBAL ``zip`` over the whole ledger (ignoring symbol + fold), which
mis-paired one symbol's entry with an unrelated symbol's / fold's exit and
inflated both the closed-trade count and the profit factor. Both reports now route
through the single canonical reconstruction (``expectancy.reconstruct_closed_trades``
+ ``compute_metrics``), so they agree on the same ``backtest_trades.csv``.

Offline + deterministic: no network, no IBKR, no order path. File IO uses temp
dirs (plus an optional read-only check against the real reports/ ledger).

    python -m unittest test_backtest_summary_reconcile
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import config
import expectancy
import main


# Column order of reports/backtest_trades.csv (subset that matters here).
_COLS = [
    "symbol", "date", "fold_start", "order_executed",
    "old_position", "new_position", "close_price", "equity",
]


def _row(symbol, date, fold, executed, old, new, close, equity):
    return {
        "symbol": symbol, "date": date, "fold_start": fold,
        "order_executed": str(executed), "old_position": str(old),
        "new_position": str(new), "close_price": str(close), "equity": str(equity),
    }


def _rows_to_csv_text(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_COLS)
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in _COLS})
    return buf.getvalue()


def _write_csv(tmp, rows):
    p = Path(tmp) / "backtest_trades.csv"
    p.write_text(_rows_to_csv_text(rows), encoding="utf-8")
    return p


def _old_global_zip_closed_count(rows):
    """Reproduce the PRE-FIX backtest-summary pairing (global zip, no grouping).

    Kept here only so the tests can assert the new canonical reconstruction
    actually diverges from the old buggy behavior on a crafted ledger.
    """
    ex = [r for r in rows if str(r.get("order_executed")).strip().lower() == "true"]
    entries = [r for r in ex if r.get("old_position") == "0" and r.get("new_position") == "1"]
    exits = [r for r in ex if r.get("old_position") == "1" and r.get("new_position") == "0"]
    n = 0
    for e, x in zip(entries, exits):
        try:
            eeq = float(e.get("equity"))
            xeq = float(x.get("equity"))
            ret = (xeq / eeq) - 1.0 if eeq else None
        except (TypeError, ValueError, ZeroDivisionError):
            ret = None
        if ret is None:
            continue
        n += 1
    return n


def _ledger_with_cross_fold_carry():
    """A ledger where a GLOBAL zip mis-pairs across symbol + fold boundaries.

    Row order matters: a global zip would pair SPY's fold-100 entry with SPY's
    fold-200 exit (a cross-fold bogus trade) and AAPL's entry with AAPL's exit,
    etc. The canonical reconstruction pairs WITHIN each (symbol, fold_start):
      * SPY fold 100: lone entry  -> open_at_period_end (excluded)
      * SPY fold 200: lone exit   -> orphan_exit        (excluded)
      * AAPL fold 100: clean round-trip -> -5% (loss)
      * SPY fold 300:  clean round-trip -> +8% (win)
    => 2 included round-trips, 2 excluded; PF = 0.08 / 0.05 = 1.60.
    """
    return [
        _row("SPY", "2022-01-03", "100", True, 0, 1, 100.0, 1.00),   # open, never closes in fold 100
        _row("SPY", "2022-02-01", "200", True, 1, 0, 110.0, 1.10),   # exit with no entry in fold 200
        _row("AAPL", "2022-01-03", "100", True, 0, 1, 50.0, 1.00),   # clean -5% round-trip
        _row("AAPL", "2022-01-20", "100", True, 1, 0, 47.5, 0.95),
        _row("SPY", "2022-03-01", "300", True, 0, 1, 120.0, 1.00),   # clean +8% round-trip
        _row("SPY", "2022-03-15", "300", True, 1, 0, 129.6, 1.08),
    ]


def _assert_metrics_reconcile(tc, m_bt, m_exp):
    """Assert two compute_metrics dicts agree on every reconciled field."""
    for key in ("n_trades_included", "n_trades_excluded", "wins", "losses",
                "scratch", "win_rate", "total_realized_pnl", "gross_win",
                "gross_loss", "profit_factor", "profit_factor_display",
                "avg_win", "avg_loss"):
        tc.assertEqual(m_bt[key], m_exp[key], f"metric mismatch on {key!r}")
    tc.assertEqual(m_bt["excluded_by_reason"], m_exp["excluded_by_reason"])


class TestCanonicalReconstructionShared(unittest.TestCase):
    def test_helper_matches_expectancy_on_synthetic_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_csv(tmp, _ledger_with_cross_fold_carry())
            rows = expectancy.load_backtest_rows(csv_path)

            included, m_bt = main._reconstruct_backtest_trades(rows)
            m_exp = expectancy.generate_report(csv_path)["metrics"]

            _assert_metrics_reconcile(self, m_bt, m_exp)

            # Deterministic expectations for this crafted ledger.
            self.assertEqual(len(included), 2)
            self.assertEqual(m_bt["n_trades_included"], 2)
            self.assertEqual((m_bt["wins"], m_bt["losses"], m_bt["scratch"]), (1, 1, 0))
            self.assertEqual(m_bt["profit_factor_display"], "1.60")
            self.assertEqual(
                m_bt["excluded_by_reason"],
                {expectancy.REASON_OPEN_AT_PERIOD_END: 1, expectancy.REASON_ORPHAN_EXIT: 1},
            )

    def test_canonical_avoids_global_zip_mispairing(self):
        # The whole point: the canonical reconstruction must NOT reproduce the
        # old global-zip count, and must equal the expectancy report's count.
        with tempfile.TemporaryDirectory() as tmp:
            rows = _ledger_with_cross_fold_carry()
            csv_path = _write_csv(tmp, rows)

            old_count = _old_global_zip_closed_count(rows)
            _, m_bt = main._reconstruct_backtest_trades(
                expectancy.load_backtest_rows(csv_path))
            m_exp = expectancy.generate_report(csv_path)["metrics"]

            self.assertEqual(old_count, 3)                      # buggy global zip
            self.assertEqual(m_bt["n_trades_included"], 2)      # canonical
            self.assertNotEqual(old_count, m_bt["n_trades_included"])
            self.assertEqual(m_bt["n_trades_included"], m_exp["n_trades_included"])

    def test_clean_round_trips_only_reconcile_exactly(self):
        # When every position closes inside its own (symbol, fold), the old and
        # new logic agree on the COUNT, but must still reconcile on all metrics.
        with tempfile.TemporaryDirectory() as tmp:
            rows = [
                _row("SPY", "2022-01-03", "100", True, 0, 1, 100.0, 1.00),
                _row("SPY", "2022-01-20", "100", True, 1, 0, 106.0, 1.06),
                _row("AAPL", "2022-01-03", "100", True, 0, 1, 50.0, 1.00),
                _row("AAPL", "2022-01-20", "100", True, 1, 0, 48.0, 0.96),
            ]
            csv_path = _write_csv(tmp, rows)
            _, m_bt = main._reconstruct_backtest_trades(
                expectancy.load_backtest_rows(csv_path))
            m_exp = expectancy.generate_report(csv_path)["metrics"]
            _assert_metrics_reconcile(self, m_bt, m_exp)
            self.assertEqual(m_bt["n_trades_included"], 2)
            self.assertEqual(m_bt["n_trades_excluded"], 0)

    def test_helper_matches_expectancy_on_real_ledger(self):
        real = config.REPORTS_DIR / "backtest_trades.csv"
        if not real.exists():
            self.skipTest(f"no real ledger at {real}")
        rows = expectancy.load_backtest_rows(real)
        _, m_bt = main._reconstruct_backtest_trades(rows)
        m_exp = expectancy.generate_report(real)["metrics"]
        _assert_metrics_reconcile(self, m_bt, m_exp)
        self.assertGreater(m_bt["n_trades_included"], 0)


class TestCommandsReconcile(unittest.TestCase):
    """End-to-end: run both CLI commands on the SAME ledger and prove the
    backtest-summary markdown emits the same canonical numbers expectancy wrote.
    """

    def test_backtest_summary_and_expectancy_report_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            trades = tmp_dir / "backtest_trades.csv"
            trades.write_text(_rows_to_csv_text(_ledger_with_cross_fold_carry()),
                              encoding="utf-8")
            (tmp_dir / "backtest_metrics.json").write_text(
                json.dumps({"symbols_tested": 2,
                            "backtest_model": "TEST",
                            "include_lstm": False}),
                encoding="utf-8")

            exp_json = tmp_dir / "expectancy_metrics.json"
            with mock.patch.object(config, "REPORTS_DIR", tmp_dir), \
                 mock.patch.object(config, "EXPECTANCY_BACKTEST_TRADES", trades), \
                 mock.patch.object(config, "EXPECTANCY_REPORT_JSON", exp_json), \
                 mock.patch.object(config, "EXPECTANCY_REPORT_MD",
                                   tmp_dir / "expectancy_report.md"), \
                 mock.patch.object(config, "EXPECTANCY_ENABLE_PROXY_RISK", False):
                args = types.SimpleNamespace(language=None, proxy_risk=False)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main.cmd_backtest_summary(args), 0)
                    self.assertEqual(main.cmd_expectancy_report(args), 0)

                summary_md = (tmp_dir / "backtest_summary.md").read_text(encoding="utf-8")
                m = json.loads(exp_json.read_text(encoding="utf-8"))["metrics"]

            # The summary markdown must carry the SAME canonical numbers the
            # expectancy report computed from the identical ledger.
            self.assertIn(f"- **Closed trades:** {m['n_trades_included']}", summary_md)
            self.assertIn(
                f"- **Wins / Losses / Scratch:** {m['wins']} / {m['losses']} / {m['scratch']}",
                summary_md)
            self.assertIn(
                f"- **Profit factor (approx):** {m['profit_factor_display']}", summary_md)
            self.assertIn(
                f"- **Total realized PnL (return units):** {m['total_realized_pnl']}",
                summary_md)

            # And those numbers are the canonical (not global-zip) ones.
            self.assertEqual(m["n_trades_included"], 2)
            self.assertEqual(m["profit_factor_display"], "1.60")


if __name__ == "__main__":
    unittest.main(verbosity=2)
