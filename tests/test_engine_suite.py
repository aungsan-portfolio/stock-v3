# ─────────────────────────────────────────────────────────────────────────────
#  test_engine_suite.py
#  Full Integration & Regression Test Suite for Enhancements 1 – 4
# ─────────────────────────────────────────────────────────────────────────────

import sys
import io
import logging
import unittest
import pandas as pd
import numpy as np

if __name__ == "__main__" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Imports from codebase
from data_manager import with_exponential_backoff
from predictor import apply_confidence_filter, MIN_SWING_CONFIDENCE, MIN_BUY_CONFIDENCE, MIN_SELL_CONFIDENCE
from position_rotator import PositionRotator, OpenPosition, IncomingSignal
from rs_calculator import RelativeStrengthEvaluator, calculate_wilder_rsi, calculate_alpha_vs_benchmark
from strategies.intraday_risk import check_soxx_tech_long_gate


class TestEngineSuite(unittest.TestCase):

    def test_enhancement_1_retry_decorator(self):
        """Enhancement 1: Retry decorator retries transient failures and eventually succeeds or re-raises."""
        attempts = 0

        @with_exponential_backoff(max_retries=3, base_delay=0.01, backoff_mult=1.0)
        def mock_fetch(symbol):
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Transient network glitch")
            return pd.DataFrame({"Close": [100, 101, 102]})

        df = mock_fetch("AAPL")
        self.assertEqual(len(df), 3)
        self.assertEqual(attempts, 3)

    def test_enhancement_2_confidence_filter(self):
        """Enhancement 2: Noise-filter demotes low-confidence signals to HOLD."""
        # SPY SELL 0.36 -> HOLD (below 0.45 floor)
        sig, conf = apply_confidence_filter("SELL", 0.36, "SPY")
        self.assertEqual(sig, "HOLD")

        # MSFT BUY 0.52 -> HOLD (below 0.55 BUY floor)
        sig, conf = apply_confidence_filter("BUY", 0.52, "MSFT")
        self.assertEqual(sig, "HOLD")

        # TSLA BUY 0.65 -> BUY (passes 0.55 BUY floor)
        sig, conf = apply_confidence_filter("BUY", 0.65, "TSLA")
        self.assertEqual(sig, "BUY")

    def test_enhancement_3_position_rotator(self):
        """Enhancement 3: Rotator protects winners (>=8%) and rotates out weak positions for strong incoming signals."""
        rotator = PositionRotator()
        positions = [
            OpenPosition(symbol="MSFT", side="long", qty=10, avg_entry_price=450, current_price=512, unrealized_pnl=620, current_conf=0.55), # PnL +13.8% (Winner protected)
            OpenPosition(symbol="NFLX", side="short", qty=10, avg_entry_price=78, current_price=80.5, unrealized_pnl=-25, current_conf=0.40), # Weak position (Conf 0.40)
        ]
        signals = [
            IncomingSignal(symbol="GOOGL", action="BUY", confidence=0.62, price=355), # High conviction incoming (Conf 0.62)
        ]

        decision = rotator.evaluate(positions, signals)
        self.assertTrue(decision.should_rotate)
        self.assertEqual(decision.exit_position.symbol, "NFLX")
        self.assertEqual(decision.enter_signal.symbol, "GOOGL")

    def test_enhancement_4_soxx_rs_bypass(self):
        """Enhancement 4: Relative Strength Bypass allows strong NVDA entry despite weak SOXX (-1.5%)."""
        evaluator = RelativeStrengthEvaluator()

        # Create dummy strong NVDA data (rising prices -> high RSI)
        dates = pd.date_range("2026-01-01", periods=30)
        nvda_prices = np.linspace(200, 240, 30) # Uptrend
        nvda_df = pd.DataFrame({"Open": nvda_prices - 1.0, "Close": nvda_prices}, index=dates)

        # Create dummy weak SOXX data
        soxx_prices = np.linspace(500, 480, 30) # Downtrend
        soxx_df = pd.DataFrame({"Open": soxx_prices + 1.0, "Close": soxx_prices}, index=dates)

        # Evaluate RS
        res = evaluator.evaluate(
            symbol="NVDA",
            symbol_df=nvda_df,
            soxx_df=soxx_df,
            soxx_chg_pct=-0.015, # SOXX is -1.5%
            pred_conf=0.60,
        )

        self.assertTrue(res.rsi_ok or res.alpha_ok)
        self.assertTrue(res.bypass_granted)


if __name__ == "__main__":
    unittest.main()
