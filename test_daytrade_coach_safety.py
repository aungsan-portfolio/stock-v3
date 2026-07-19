import types
import unittest

import pandas as pd

import config
import trade_coach


def fake_signal(action="BUY", confidence=0.9, price=100.0):
    return types.SimpleNamespace(
        symbol="AAPL",
        action=action,
        confidence=confidence,
        rf_score=0.8,
        lstm_score=0.7,
        tech_score=0.6,
        price=price,
        reason="test",
    )


def ohlcv():
    rows = []
    for i in range(20):
        close = 90 + i
        rows.append(
            {
                "Open": close - 0.25,
                "High": close + 1,
                "Low": close - 1,
                "Close": close,
                "Volume": 1_000_000 + i,
            }
        )
    return pd.DataFrame(rows)


class DaytradeCoachSafetyTests(unittest.TestCase):
    def test_formula_preview_is_read_only_and_keeps_ibkr_quantity(self):
        preview = trade_coach.build_trade_preview(
            fake_signal(price=100),
            cash=100_000,
            current_positions={},
            open_orders=[],
            ohlcv=ohlcv(),
        )
        self.assertTrue(preview["tradeable"])
        self.assertEqual(preview["quantity"], 5)  # existing IBKRBridge cap mirror
        self.assertGreater(preview["suggested_shares_by_risk"], 0)
        self.assertAlmostEqual(preview["rr_2r"], 2.0)
        self.assertIn("existing IBKRBridge sizing still applies", preview["daytrade_formula_note"])
        self.assertIsNotNone(preview["previous_day_pivot_levels"])
        self.assertIsNotNone(preview["gap_pct"])
        self.assertIsNotNone(preview["atr_dollars"])

    def test_daytrade_refusal_reasons_block_invalid_risk(self):
        preview = {"tradeable": True, "rr_2r": 1.0, "risk_per_share": 0, "suggested_shares_by_risk": 0}
        reasons = trade_coach.daytrade_refusal_reasons(fake_signal(), preview)
        self.assertTrue(any("DAYTRADE_MIN_RR" in r for r in reasons))
        self.assertIn("risk/share is invalid", reasons)
        self.assertIn("suggested shares by risk is 0", reasons)

    def test_evaluate_daily_candidate_includes_formula_gates(self):
        preview = trade_coach.build_trade_preview(fake_signal(price=100), cash=50, ohlcv=ohlcv())
        ev = trade_coach.evaluate_daily_candidate(
            fake_signal(price=100), preview, trade_coach.CHART_OK, "ok"
        )
        self.assertFalse(ev["accepted"])
        self.assertIn("suggested shares by risk is 0", ev["skip_reason"])

    def test_default_daytrade_cap_is_one(self):
        self.assertEqual(config.DAYTRADE_MAX_PAPER_TRADES_PER_RUN, 1)
        self.assertEqual(config.COACH_MAX_PAPER_TRADES_PER_RUN, 1)


if __name__ == "__main__":
    unittest.main()
