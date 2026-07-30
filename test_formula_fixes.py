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

    def test_entry_gate_symbol_exposure_with_existing_position(self):
        from entry_gate_service import EntryGateService

        # Equity $10,000, Max Symbol Exposure 20% = $2,000
        # Existing AAPL position = $1,800. New intended AAPL = $500. Total = $2,300 -> Should block!
        gate = EntryGateService(
            net_liq_fn=lambda: 10000.0,
            symbol_exposure_fn=lambda sym: 1800.0 if sym == "AAPL" else 0.0
        )
        gate.market_hours_ok = lambda: True

        blocked, reason = gate.entry_blocked("AAPL", intended_value=500.0)
        self.assertTrue(blocked)
        self.assertEqual(reason, "symbol_exposure")

    def test_risk_sized_qty_fail_closed_setting(self):
        from entry_gate_service import EntryGateService

        gate = EntryGateService()
        gate._c = lambda name, default=None: True if name in (
            "MINERVINI_OVERLAY_ENABLED", "MINERVINI_SIZING_ENABLED", "MINERVINI_SIZING_FAIL_CLOSED"
        ) else default

        # Invalid entry/stop -> should return 0 (fail-closed) when MINERVINI_SIZING_FAIL_CLOSED is True
        qty = gate.risk_sized_qty("AAPL", signal=type("Signal", (), {"action": "BUY"})(), entry_price=0.0, notional_qty=100)
        self.assertEqual(qty, 0)


if __name__ == "__main__":
    unittest.main()
