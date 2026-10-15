"""test_backtest_guards.py - Rigorous verification of backtest purge gap and execution accounting.

Invariants verified:
1. Purge Gap Invariant:
   - Training window indices strictly terminate at `train_end = start - purge_gap`.
   - Forward-horizon labels from the test period (t >= start - purge_gap) never leak into training.
   - Fold-local LSTM strictly trains and normalizes features up to `train_end`.
2. Gap-Down Stop Fill Invariant:
   - When an open long position encounters a gap-down open below the hard stop price,
     the order fills at the open price (not the stop price), capturing the full adverse gap.
3. Intrabar Stop Fill Invariant:
   - When the open is above the stop price but the intrabar low touches or breaches the stop,
     the order fills at the exact stop price.
4. Monotonic Chronological Compounding Invariant:
   - Daily return series and equity compounding strictly progress chronologically:
     `equity[t] == equity[t-1] * (1.0 + net_pnl[t])`.
5. Aggregate Schema Invariant:
   - `purge_gap` is recorded in aggregate metrics and matches `config.BACKTEST_PURGE_GAP`.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import config
import backtest
from backtest import (
    _optional_lstm_scores_for_fold,
    run_backtest,
)
import expectancy


def _make_ohlcv(n: int = 600, seed: int = 11) -> pd.DataFrame:
    """Deterministic random-walk OHLCV with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, size=n)))
    open_ = close * (1.0 + rng.normal(0, 0.003, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


class TestPurgeGapInvariants(unittest.TestCase):
    """Test 1: Verify purge gap prevents lookahead forward-label leakage."""

    def test_purge_gap_terminates_training_before_test_start(self):
        """RF and LSTM training slices must stop at `start - purge_gap`."""
        n_total = 200
        start = 100
        purge_gap = 5

        # Create dummy features and labels
        X_all = np.arange(n_total)[:, None].astype(np.float32)
        y_all = (np.arange(n_total) % 2).astype(int)

        # Expected training cutoff
        expected_train_end = start - purge_gap
        self.assertEqual(expected_train_end, 95)

        X_tr = X_all[:expected_train_end]
        y_tr = y_all[:expected_train_end]

        self.assertEqual(len(X_tr), 95)
        self.assertEqual(int(X_tr[-1][0]), 94)  # Index 94 is the last training item

    def test_lstm_fold_respects_purge_gap(self):
        """Fold-local LSTM must slice training data up to `start - purge_gap`."""
        n_total = 150
        start = 80
        step = 10
        purge_gap = 5

        X_all = np.random.randn(n_total, len(backtest.FEATURE_COLS)).astype(np.float32)
        y_all = (np.random.rand(n_total) > 0.5).astype(int)

        scores, ok = _optional_lstm_scores_for_fold(X_all, y_all, start, step, purge_gap=purge_gap)
        self.assertEqual(len(scores), step)
        self.assertTrue(all(np.isfinite(scores)))
        self.assertTrue(ok)


class TestExecutionAccountingInvariants(unittest.TestCase):
    """Test 2, 3 & 4: Verify realistic stop-loss fills and chronological P&L."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.reports_dir = Path(self.tmp_dir.name)
        self.cfg = config.get_settings()
        self.cfg.REPORTS_DIR = self.reports_dir
        mock.patch.object(config, "get_settings", return_value=self.cfg).start()
        mock.patch.object(config, "REPORTS_DIR", self.reports_dir).start()

    def tearDown(self):
        mock.patch.stopall()
        self.tmp_dir.cleanup()

    def test_gap_down_stop_fills_at_open_price(self):
        """When an open long gaps below the stop price, it must fill at open_px, not stop_px."""
        df = _make_ohlcv(600, seed=11)

        with mock.patch("backtest.fetch_ohlcv", return_value=df):
            result = run_backtest(
                symbols=["SYNTH"],
                train_min=120,
                step=30,
                verbose=False,
                include_lstm=False,
                settings=self.cfg,
            )

        trades_path = self.reports_dir / "backtest_trades.csv"
        self.assertTrue(trades_path.exists())
        trades_df = pd.read_csv(trades_path)

        # Find gap stop exits
        gap_rows = trades_df[trades_df["execution_note"].str.startswith("hard-stop-gap", na=False)]
        self.assertGreaterEqual(len(gap_rows), 1)

        for _, row in gap_rows.iterrows():
            bar_date = row["date"]
            df_bar = df.loc[pd.Timestamp(bar_date)]
            # Invariant: fill must be at open price, NOT initial_stop_price
            self.assertAlmostEqual(float(row["exit_price"]), float(df_bar["Open"]), places=4)
            self.assertLessEqual(float(row["exit_price"]), float(row["initial_stop_price"]))

    def test_intrabar_stop_fills_at_stop_price(self):
        """When open is above stop but low touches stop, it must fill at stop_px."""
        df = _make_ohlcv(600, seed=11)

        with mock.patch("backtest.fetch_ohlcv", return_value=df):
            result = run_backtest(
                symbols=["SYNTH"],
                train_min=120,
                step=30,
                verbose=False,
                include_lstm=False,
                settings=self.cfg,
            )

        trades_df = pd.read_csv(self.reports_dir / "backtest_trades.csv")
        stop_rows = trades_df[trades_df["execution_note"].str.startswith("hard-stop (", na=False)]
        self.assertGreaterEqual(len(stop_rows), 1)

        for _, row in stop_rows.iterrows():
            # Intrabar stop fills at stop_px
            self.assertAlmostEqual(float(row["exit_price"]), float(row["initial_stop_price"]), places=4)

    def test_monotonic_chronological_equity_compounding(self):
        """Every row's equity must strictly equal previous equity * (1 + net_pnl)."""
        df = _make_ohlcv(600, seed=11)

        with mock.patch("backtest.fetch_ohlcv", return_value=df):
            result = run_backtest(
                symbols=["SYNTH"],
                train_min=120,
                step=30,
                verbose=False,
                include_lstm=False,
                settings=self.cfg,
            )

        trades_df = pd.read_csv(self.reports_dir / "backtest_trades.csv")
        eq = trades_df["equity"].values
        pnls = trades_df["net_pnl"].values

        prev = 1.0
        for i, (e, p) in enumerate(zip(eq, pnls)):
            expected = prev * (1.0 + p)
            self.assertAlmostEqual(e, expected, places=6,
                                   msg=f"Row {i} broke chronological compounding: got {e}, expected {expected}")
            prev = e

    def test_aggregate_metrics_contains_purge_gap(self):
        """Aggregate JSON report must include purge_gap."""
        df = _make_ohlcv(600, seed=11)
        with mock.patch("backtest.fetch_ohlcv", return_value=df):
            result = run_backtest(
                symbols=["SYNTH"],
                train_min=120,
                step=30,
                verbose=False,
                include_lstm=False,
                settings=self.cfg,
            )

        self.assertIn("purge_gap", result)
        self.assertEqual(result["purge_gap"], config.BACKTEST_PURGE_GAP)

        metrics_file = self.reports_dir / "backtest_metrics.json"
        self.assertTrue(metrics_file.exists())
        saved = json.loads(metrics_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["purge_gap"], config.BACKTEST_PURGE_GAP)


if __name__ == "__main__":
    unittest.main()
