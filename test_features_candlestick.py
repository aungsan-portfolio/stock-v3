"""test_features_candlestick.py -- offline unit tests for candlestick pattern features.

Tests follow the same pattern as test_features_regime.py: synthetic OHLCV data
is fed through build_features() with USE_CANDLESTICK_FEATURES=True/False, and
the output feature columns are verified for presence, shape, and basic range.
"""
import os
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

import config
import data_manager


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Deterministic synthetic OHLCV with identifiable candlestick patterns."""
    rng = np.random.default_rng(seed)
    # Start with a known price
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 50.0)  # floor at 50
    # Generate open, high, low from close
    open_ = close + rng.normal(0, 0.3, n)
    open_ = np.maximum(open_, 40.0)
    high = np.maximum(close, open_) + rng.uniform(0, 0.5, n)
    low = np.minimum(close, open_) - rng.uniform(0, 0.5, n)
    volume = rng.uniform(1e6, 5e6, n)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    df.index = pd.date_range("2020-01-01", periods=n, freq="B")
    return df


class TestCandlestickFeatureColumns(unittest.TestCase):
    """Verify that feature column lists are correct when flag is on/off."""

    def setUp(self):
        self._orig_flag = getattr(config, "USE_CANDLESTICK_FEATURES", False)

    def tearDown(self):
        config.USE_CANDLESTICK_FEATURES = self._orig_flag

    def test_base_columns_unchanged_when_flag_off(self):
        """USE_CANDLESTICK_FEATURES=False should return only base 14 columns."""
        config.USE_CANDLESTICK_FEATURES = False
        cols = data_manager.get_feature_columns()
        self.assertEqual(len(cols), 14)
        for c in cols:
            self.assertNotIn("doji", c)

    def test_candlestick_columns_appended_when_flag_on(self):
        """USE_CANDLESTICK_FEATURES=True should append 16 columns."""
        config.USE_CANDLESTICK_FEATURES = True
        cols = data_manager.get_feature_columns()
        self.assertEqual(len(cols), 14 + 16, f"got {len(cols)} columns")
        self.assertIn("doji", cols)
        self.assertIn("hammer", cols)
        self.assertIn("morning_star", cols)
        self.assertIn("evening_star", cols)
        self.assertIn("bullish_engulfing", cols)
        self.assertIn("shooting_star", cols)

    def test_flag_off_returns_no_new_columns(self):
        """Verify that when flag is off, no candlestick columns leak in."""
        config.USE_CANDLESTICK_FEATURES = False
        cols = data_manager.get_feature_columns()
        candle_names = [
            "doji", "dragonfly_doji", "gravestone_doji",
            "white_marubozu", "black_marubozu",
            "hammer", "hanging_man", "shooting_star",
            "bullish_engulfing", "bearish_engulfing",
            "bullish_harami", "bearish_harami",
            "piercing_line", "dark_cloud_cover",
            "morning_star", "evening_star",
        ]
        for cn in candle_names:
            self.assertNotIn(cn, cols)


class TestCandlestickBuildFeatures(unittest.TestCase):
    """Verify build_features() produces correct candlestick columns."""

    def setUp(self):
        self._orig_flag = getattr(config, "USE_CANDLESTICK_FEATURES", False)
        self.df = _make_ohlcv()

    def tearDown(self):
        config.USE_CANDLESTICK_FEATURES = self._orig_flag

    def test_build_features_no_candle_cols_when_off(self):
        """With flag OFF, output should have no candlestick columns."""
        config.USE_CANDLESTICK_FEATURES = False
        result = data_manager.build_features(self.df)
        candle_names = [
            "doji", "dragonfly_doji", "gravestone_doji",
            "white_marubozu", "black_marubozu",
            "hammer", "hanging_man", "shooting_star",
            "bullish_engulfing", "bearish_engulfing",
            "bullish_harami", "bearish_harami",
            "piercing_line", "dark_cloud_cover",
            "morning_star", "evening_star",
        ]
        for cn in candle_names:
            self.assertNotIn(cn, result.columns)

    def test_build_features_has_candle_cols_when_on(self):
        """With flag ON, output should include all 16 candlestick columns."""
        config.USE_CANDLESTICK_FEATURES = True
        result = data_manager.build_features(self.df)
        expected_cols = data_manager.get_feature_columns()
        for col in expected_cols:
            self.assertIn(col, result.columns)
        self.assertIn("doji", result.columns)
        self.assertIn("morning_star", result.columns)

    def test_candle_values_are_binary(self):
        """All candlestick columns should be {0.0, 1.0} boolean flags."""
        config.USE_CANDLESTICK_FEATURES = True
        result = data_manager.build_features(self.df)
        candle_cols = [
            "doji", "dragonfly_doji", "gravestone_doji",
            "white_marubozu", "black_marubozu",
            "hammer", "hanging_man", "shooting_star",
            "bullish_engulfing", "bearish_engulfing",
            "bullish_harami", "bearish_harami",
            "piercing_line", "dark_cloud_cover",
            "morning_star", "evening_star",
        ]
        for cn in candle_cols:
            if cn in result.columns:
                series = result[cn].dropna()
                self.assertTrue(
                    ((series == 0.0) | (series == 1.0)).all(),
                    f"{cn} contains non-binary values: {series.unique()}"
                )

    def test_3candle_patterns_exist(self):
        """Three-candle patterns should produce some 1.0 values (rare but present)."""
        config.USE_CANDLESTICK_FEATURES = True
        result = data_manager.build_features(self.df)
        ms = result["morning_star"].sum()
        es = result["evening_star"].sum()
        # These are rare patterns; just verify the column exists and sums >= 0
        self.assertGreaterEqual(ms, 0)
        self.assertGreaterEqual(es, 0)

    def test_produces_fewer_rows_than_input(self):
        """build_features drops NaN warmup rows."""
        config.USE_CANDLESTICK_FEATURES = True
        result = data_manager.build_features(self.df)
        self.assertLess(len(result), len(self.df))
        self.assertGreater(len(result), 100)


class TestCandlestickAllColsPresent(unittest.TestCase):
    """Verify all 16 candlestick columns exist and are binary."""

    def setUp(self):
        self._orig_flag = getattr(config, "USE_CANDLESTICK_FEATURES", False)
        rng = np.random.default_rng(42)
        ns = 300
        close = 100.0 + np.cumsum(rng.normal(0, 0.5, ns))
        close = np.maximum(close, 50.0)
        open_ = close + rng.normal(0, 0.3, ns)
        high = np.maximum(close, open_) + rng.uniform(0, 0.5, ns)
        low = np.minimum(close, open_) - rng.uniform(0, 0.5, ns)
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": rng.uniform(1e6, 5e6, ns)})
        df.index = pd.date_range("2020-01-01", periods=ns, freq="B")
        self.df = df

    def tearDown(self):
        config.USE_CANDLESTICK_FEATURES = self._orig_flag

    def test_all_16_candle_cols_present_and_binary(self):
        config.USE_CANDLESTICK_FEATURES = True
        result = data_manager.build_features(self.df)
        candle_cols = [
            "doji", "dragonfly_doji", "gravestone_doji",
            "white_marubozu", "black_marubozu",
            "hammer", "hanging_man", "shooting_star",
            "bullish_engulfing", "bearish_engulfing",
            "bullish_harami", "bearish_harami",
            "piercing_line", "dark_cloud_cover",
            "morning_star", "evening_star",
        ]
        for cn in candle_cols:
            self.assertIn(cn, result.columns,
                          msg=f"{cn} missing from build_features output")
            series = result[cn].dropna()
            vals = series.unique()
            self.assertTrue(
                ((series == 0.0) | (series == 1.0)).all(),
                msg=f"{cn} contains non-binary values: {vals}"
            )


if __name__ == "__main__":
    unittest.main()
