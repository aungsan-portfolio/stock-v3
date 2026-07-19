"""
test_features_regime.py — Offline unit tests for the Item-3 regime /
relative-strength feature set (data_manager).

Pure stdlib ``unittest``. No network, no yfinance, no trained models — every
test feeds synthetic OHLCV and toggles the config flags directly, so the suite
is deterministic and fast:

    python -m unittest test_features_regime -v

What is covered:
- Default (flags off) behaviour is unchanged: get_feature_columns() returns the
  exact frozen 14-feature list and build_features() adds no regime/market cols.
- USE_REGIME_FEATURES on: 6 single-symbol features appended and finite, values
  match an independent recomputation, and the base-14 features are untouched.
- USE_MARKET_RELATIVE_FEATURES on: 4 features from an aligned benchmark, plus
  the neutral-0.0 fallback when no market_df is supplied (no crash).
- Backward compatibility (one-arg call) and fetch_market_benchmark() wrapper.
"""
import unittest
from contextlib import ExitStack
from unittest import mock

import numpy as np
import pandas as pd

import config
import data_manager
from data_manager import build_features, get_feature_columns


# ── Frozen expectations (guard against accidental renames/reordering) ──
BASE_14 = [
    "sma_cross", "dist_ema", "rsi", "macd", "macd_sig", "macd_hist",
    "bb_pct", "atr_pct", "vol_ratio", "ret_1d", "ret_5d", "ret_20d",
    "dist_sma20", "dist_sma50",
]
REGIME_COLS = ["realized_vol", "vol_regime", "mom_short", "mom_long",
               "ts_rank", "dist_high"]
MARKET_COLS = ["rel_ret_short", "rel_ret_long", "rs_slope", "mkt_trend"]


# ── Synthetic data helpers (mirrors test_logic.make_ohlcv) ───────────
def make_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Deterministic random-walk OHLCV with a business-day DatetimeIndex."""
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


def make_market(index: pd.DatetimeIndex, seed: int = 11) -> pd.DataFrame:
    """A benchmark OHLCV frame on the same index (only Close is used)."""
    rng = np.random.default_rng(seed)
    n = len(index)
    steps = rng.normal(0.0003, 0.008, size=n)
    close = 300.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(
        {
            "Open": close, "High": close * 1.001, "Low": close * 0.999,
            "Close": close, "Volume": np.full(n, 1e8),
        },
        index=index,
    )


def _flags(**kwargs):
    """Context manager patching config feature flags for the duration of a test."""
    stack = ExitStack()
    for name, value in kwargs.items():
        stack.enter_context(mock.patch.object(config, name, value))
    return stack


# ── 1. Default behaviour is unchanged (both flags off) ───────────────
class TestDefaultUnchanged(unittest.TestCase):
    def setUp(self):
        # Be explicit so the test is correct even if a future default flips.
        self._stack = _flags(USE_REGIME_FEATURES=False,
                             USE_MARKET_RELATIVE_FEATURES=False,
                             USE_CANDLESTICK_FEATURES=False)
        self._stack.__enter__()
        self.df = make_ohlcv(n=400, seed=3)
        self.feat = build_features(self.df)

    def tearDown(self):
        self._stack.__exit__(None, None, None)

    def test_feature_columns_are_exactly_the_frozen_14(self):
        self.assertEqual(get_feature_columns(), BASE_14)

    def test_no_regime_or_market_columns_added(self):
        for col in REGIME_COLS + MARKET_COLS:
            self.assertNotIn(col, self.feat.columns)

    def test_base_features_present_and_finite(self):
        for col in BASE_14:
            self.assertIn(col, self.feat.columns)
        self.assertTrue(np.isfinite(self.feat[BASE_14].values).all())


# ── 2. Regime features (single-symbol, USE_REGIME_FEATURES on) ───────
class TestRegimeFeatures(unittest.TestCase):
    def setUp(self):
        self._stack = _flags(USE_REGIME_FEATURES=True,
                             USE_MARKET_RELATIVE_FEATURES=False,
                             USE_CANDLESTICK_FEATURES=False)
        self._stack.__enter__()
        self.df = make_ohlcv(n=400, seed=5)
        self.feat = build_features(self.df)
        self.close = self.df["Close"]

    def tearDown(self):
        self._stack.__exit__(None, None, None)

    def test_columns_appended_after_base_14(self):
        self.assertEqual(get_feature_columns(), BASE_14 + REGIME_COLS)

    def test_all_regime_columns_present_and_finite(self):
        self.assertGreater(len(self.feat), 50)  # warmup left usable rows
        for col in REGIME_COLS:
            self.assertIn(col, self.feat.columns)
        self.assertTrue(np.isfinite(self.feat[REGIME_COLS].values).all())

    def test_realized_vol_matches_recompute(self):
        expected = self.close.pct_change(1).rolling(config.REGIME_VOL_SHORT).std()
        np.testing.assert_allclose(
            self.feat["realized_vol"].values,
            expected.reindex(self.feat.index).values, rtol=1e-9, atol=1e-12,
        )

    def test_momentum_matches_recompute(self):
        exp_s = self.close.pct_change(config.REGIME_MOM_SHORT)
        exp_l = self.close.pct_change(config.REGIME_MOM_LONG)
        np.testing.assert_allclose(
            self.feat["mom_short"].values,
            exp_s.reindex(self.feat.index).values, rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            self.feat["mom_long"].values,
            exp_l.reindex(self.feat.index).values, rtol=1e-9, atol=1e-12,
        )

    def test_ts_rank_bounded_unit_interval(self):
        v = self.feat["ts_rank"].values
        self.assertTrue((v > 0.0).all())
        self.assertTrue((v <= 1.0 + 1e-12).all())

    def test_dist_high_is_non_positive_and_matches_recompute(self):
        # rolling max includes the current bar, so distance below high <= 0.
        self.assertLessEqual(float(self.feat["dist_high"].max()), 1e-12)
        roll_high = self.close.rolling(config.REGIME_HIGH_WINDOW).max()
        expected = self.close / roll_high - 1.0
        np.testing.assert_allclose(
            self.feat["dist_high"].values,
            expected.reindex(self.feat.index).values, rtol=1e-9, atol=1e-12,
        )

    def test_base_14_unchanged_on_shared_rows(self):
        # Regime features must not alter how the base features are computed; on the
        # rows that survive both warmups the values must be identical.
        with _flags(USE_REGIME_FEATURES=False,
                   USE_MARKET_RELATIVE_FEATURES=False,
                   USE_CANDLESTICK_FEATURES=False):
            feat_off = build_features(self.df)
        shared = self.feat.index
        self.assertTrue(set(shared).issubset(set(feat_off.index)))
        np.testing.assert_allclose(
            self.feat[BASE_14].values,
            feat_off.loc[shared, BASE_14].values, rtol=1e-9, atol=1e-12,
        )


# ── 3. Market-relative features (USE_MARKET_RELATIVE_FEATURES on) ─────
class TestMarketRelativeFeatures(unittest.TestCase):
    def setUp(self):
        self._stack = _flags(USE_REGIME_FEATURES=False,
                             USE_MARKET_RELATIVE_FEATURES=True,
                             USE_CANDLESTICK_FEATURES=False)
        self._stack.__enter__()
        self.df = make_ohlcv(n=400, seed=9)
        self.market = make_market(self.df.index, seed=21)

    def tearDown(self):
        self._stack.__exit__(None, None, None)

    def test_columns_appended_after_base_14(self):
        self.assertEqual(get_feature_columns(), BASE_14 + MARKET_COLS)

    def test_with_market_df_present_and_finite(self):
        feat = build_features(self.df, self.market)
        for col in MARKET_COLS:
            self.assertIn(col, feat.columns)
        self.assertTrue(np.isfinite(feat[MARKET_COLS].values).all())

    def test_rel_ret_matches_recompute(self):
        feat = build_features(self.df, self.market)
        mkt_close = self.market["Close"].reindex(self.df.index).ffill()
        s = config.REL_RET_SHORT
        expected = self.df["Close"].pct_change(s) - mkt_close.pct_change(s)
        np.testing.assert_allclose(
            feat["rel_ret_short"].values,
            expected.reindex(feat.index).values, rtol=1e-9, atol=1e-12,
        )

    def test_missing_market_df_emits_neutral_zero(self):
        # Reset the one-shot warning latch so the branch is exercised cleanly.
        with mock.patch.object(data_manager, "_MARKET_DF_WARNED", False):
            feat = build_features(self.df)  # no market_df
        for col in MARKET_COLS:
            self.assertIn(col, feat.columns)
            self.assertTrue((feat[col].values == 0.0).all())
        # Base features still present — neutral fill must not shrink the frame
        # beyond the normal warmup.
        self.assertTrue(np.isfinite(feat[BASE_14].values).all())


# ── 4. Backward compatibility + benchmark fetch wrapper ──────────────
class TestCompatAndBenchmark(unittest.TestCase):
    def test_one_arg_call_still_works_default(self):
        with _flags(USE_REGIME_FEATURES=False, USE_MARKET_RELATIVE_FEATURES=False, USE_CANDLESTICK_FEATURES=False):
            feat = build_features(make_ohlcv(n=300, seed=4))
        self.assertEqual(list(feat[BASE_14].columns), BASE_14)

    def test_both_flags_on_yields_full_column_set(self):
        df = make_ohlcv(n=400, seed=6)
        market = make_market(df.index, seed=31)
        with _flags(USE_REGIME_FEATURES=True, USE_MARKET_RELATIVE_FEATURES=True, USE_CANDLESTICK_FEATURES=False):
            self.assertEqual(get_feature_columns(),
                             BASE_14 + REGIME_COLS + MARKET_COLS)
            feat = build_features(df, market)
        for col in BASE_14 + REGIME_COLS + MARKET_COLS:
            self.assertIn(col, feat.columns)
        self.assertTrue(
            np.isfinite(feat[BASE_14 + REGIME_COLS + MARKET_COLS].values).all()
        )

    def test_fetch_market_benchmark_wraps_fetch_ohlcv(self):
        sentinel = make_ohlcv(n=10)
        with mock.patch.object(data_manager, "fetch_ohlcv",
                              return_value=sentinel) as m:
            out = data_manager.fetch_market_benchmark()
        self.assertIs(out, sentinel)
        args, kwargs = m.call_args
        # First positional arg is the configured benchmark symbol.
        self.assertEqual(args[0], config.MARKET_BENCHMARK_SYMBOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
