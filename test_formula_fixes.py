import unittest
import numpy as np
import pandas as pd

import config
import data_manager
from predictor import apply_position_rule_with_hold


class TestFormulaFixes(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 200
        close = 100 + np.cumsum(np.random.randn(n))
        high = close + np.abs(np.random.randn(n))
        low = close - np.abs(np.random.randn(n))
        open_ = close + np.random.randn(n) * 0.5
        volume = np.random.randint(1000, 5000, size=n)

        self.df = pd.DataFrame({
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        })

    def test_volume_shock_no_future_leakage(self):
        cfg = config.Settings()
        cfg.USE_MICRO_FEATURES = True
        feat_1 = data_manager.build_features(self.df.iloc[:150].copy(), cfg=cfg)
        feat_2 = data_manager.build_features(self.df.copy(), cfg=cfg)

        target_idx = feat_1.index[-10]
        val_bar50_short = feat_1.loc[target_idx, "volume_shock"]
        val_bar50_full = feat_2.loc[target_idx, "volume_shock"]

        self.assertFalse(np.isnan(val_bar50_short))
        self.assertAlmostEqual(val_bar50_short, val_bar50_full, places=5)

    def test_rolling_efficiency_range(self):
        cfg = config.Settings()
        cfg.USE_MICRO_FEATURES = True
        feat = data_manager.build_features(self.df.copy(), cfg=cfg)
        eff = feat["rolling_efficiency"].dropna()
        self.assertTrue((eff >= 0.0).all())
        self.assertTrue((eff <= 1.0).all())

    def test_rsi_smoothing_config(self):
        cfg_wilder = config.Settings()
        cfg_wilder.RSI_SMOOTHING = "wilder"

        cfg_simple = config.Settings()
        cfg_simple.RSI_SMOOTHING = "simple"

        feat_wilder = data_manager.build_features(self.df.copy(), cfg=cfg_wilder)
        feat_simple = data_manager.build_features(self.df.copy(), cfg=cfg_simple)

        self.assertTrue((feat_wilder["rsi"].dropna() >= 0).all())
        self.assertTrue((feat_wilder["rsi"].dropna() <= 100).all())
        self.assertTrue((feat_simple["rsi"].dropna() >= 0).all())
        self.assertTrue((feat_simple["rsi"].dropna() <= 100).all())

        self.assertNotEqual(
            round(float(feat_wilder["rsi"].iloc[50]), 4),
            round(float(feat_simple["rsi"].iloc[50]), 4)
        )

    def test_low_price_hard_stop_trigger(self):
        pos, exec_, note, bars = apply_position_rule_with_hold(
            position=1,
            signal="HOLD",
            allow_short=False,
            bars_held=2,
            min_hold=1,
            entry_price=100.0,
            current_price=98.0,
            hard_stop_pct=0.05,
            low_price=94.0,
        )

        self.assertEqual(pos, 0)
        self.assertTrue(exec_)
        self.assertTrue(note.startswith("hard-stop"))


if __name__ == "__main__":
    unittest.main()
