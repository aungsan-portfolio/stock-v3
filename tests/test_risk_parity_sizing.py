import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from strategies.base import TradeSignal
from strategies.position_sizer import calculate_shares, estimated_risk


class TestRiskParitySizing(unittest.TestCase):
    """
    Validates that calculate_shares caps trade risk at MAX_RISK_PER_TRADE_DOLLARS ($25.00),
    preventing volatile low-priced stocks like RIOT from generating oversized dollar losses.
    """

    def test_riot_risk_parity_cap(self):
        # RIOT @ $19.65, stop $19.54 -> risk_per_share = $0.11
        sig = TradeSignal(
            symbol="RIOT",
            side="BUY",
            strategy="MOMENTUM_SCALP",
            entry_price=19.65,
            stop_price=19.54,
            target_price=19.80,
            confidence=0.89,
            reason="Test",
            risk_per_share=0.11,
            atr=0.15
        )
        equity = 100000.0  # $100k equity

        from unittest.mock import patch
        with patch("config.MAX_RISK_PER_TRADE_DOLLARS", 25.0), patch("config.MAX_POSITION_SIZE_DOLLARS", 5000.0):
            shares = calculate_shares(sig, equity)
            risk_dollars = estimated_risk(sig, shares)

            # Expected: max_risk = $25.00 -> shares = floor(25 / 0.11) = 227
            self.assertLessEqual(risk_dollars, 25.0)
            self.assertEqual(shares, 227)

    def test_tsla_risk_parity_cap(self):
        # TSLA @ $332.07, stop $331.87 -> risk_per_share = $0.20
        sig = TradeSignal(
            symbol="TSLA",
            side="BUY",
            strategy="MOMENTUM_SCALP",
            entry_price=332.07,
            stop_price=331.87,
            target_price=332.39,
            confidence=0.90,
            reason="Test",
            risk_per_share=0.20,
            atr=0.30
        )
        equity = 100000.0

        from unittest.mock import patch
        with patch("config.MAX_RISK_PER_TRADE_DOLLARS", 25.0), patch("config.MAX_POSITION_SIZE_DOLLARS", 5000.0):
            shares = calculate_shares(sig, equity)
            risk_dollars = estimated_risk(sig, shares)

            # Expected: max_risk = $25.00 -> shares = floor(25 / 0.20) = 125
            # Capital cap = floor(5000 / 332.07) = 15 shares
            # Final qty = min(125, 15) = 15 shares
            self.assertEqual(shares, 15)
            self.assertLessEqual(risk_dollars, 25.0)

    def test_noise_floor_hard_veto_ratio(self):
        # Configured noise floor ratio calibrated between 0.70 and 0.80
        self.assertIn(config.NOISE_FLOOR_HARD_VETO_RATIO, [0.75, 0.80])

    def test_symbol_entry_gate_limit(self):
        from strategies.order_manager import execute_signal
        from unittest.mock import patch

        sig = TradeSignal(
            symbol="PLTR",
            side="BUY",
            strategy="MOMENTUM_SCALP",
            entry_price=175.0,
            stop_price=174.0,
            target_price=176.7,
            confidence=0.85,
            reason="Test",
            risk_per_share=1.00,
            atr=0.50
        )

        mock_trades = [
            {"symbol": "PLTR", "event_type": "ORDER_PLACED"},
            {"symbol": "PLTR", "event_type": "ORDER_PLACED"}
        ]

        bridge = MagicMock()
        bridge.is_connected = True
        bridge.open_position_count.return_value = 0
        bridge.has_position.return_value = False
        bridge.has_working_order.return_value = False
        bridge.get_positions.return_value = []

        with patch("strategies.trade_journal.today_trades", return_value=mock_trades), patch.object(config, "SPY_TREND_GUARD_ENABLED", False):
            res = execute_signal(sig, equity=100000.0, current_pnl=0.0, bridge=bridge, dry_run=True)
            self.assertEqual(res["status"], "REJECTED")
            self.assertIn("Max Symbol Entries Gate", res["reason"])

    def test_breakeven_stop_lock(self):
        from strategies.trailing_stop import TrailingStopState, DynamicTrailingStopManager
        import time
        tsm = DynamicTrailingStopManager()
        sig = TradeSignal(
            symbol="IREN",
            side="BUY",
            strategy="MOMENTUM_SCALP",
            entry_price=42.00,
            stop_price=41.50,
            target_price=43.50,
            confidence=0.85,
            reason="Test",
            risk_per_share=0.50,
            atr=0.50
        )
        state = tsm.initialize_position(sig, fill_price=42.00, stop_order_id="mock_id")
        # Bypass 30s update cooldown for test
        state.last_updated = time.time() - 35
        # Price moves up by +1.0x ATR ($0.50) to $42.55
        updated_state = tsm.update_stop("IREN", current_price=42.55, dry_run=True)
        self.assertIsNotNone(updated_state)
        self.assertGreaterEqual(updated_state.stop_price, 42.00)

    def test_spy_trend_guard(self):
        from strategies.order_manager import execute_signal
        from unittest.mock import patch, MagicMock
        import pandas as pd
        sig = TradeSignal(
            symbol="NVDA",
            side="BUY",
            strategy="VWAP_BOUNCE",
            entry_price=217.50,
            stop_price=216.50,
            target_price=220.00,
            confidence=0.80,
            reason="Test",
            risk_per_share=1.00,
            atr=1.20
        )
        mock_spy_df = pd.DataFrame([
            {"open": 500.0, "close": 501.0},
            {"open": 501.0, "close": 499.0} # Bearish candle (close < open)
        ])
        bridge = MagicMock()
        bridge.is_connected = True
        bridge.open_position_count.return_value = 0

        with patch("strategies.intraday_data.fetch_intraday", return_value=mock_spy_df), patch.object(config, "SPY_TREND_GUARD_ENABLED", True):
            res = execute_signal(sig, equity=100000.0, current_pnl=0.0, bridge=bridge, dry_run=False)
            self.assertEqual(res["status"], "REJECTED")
            self.assertIn("SPY Trend Guard", res["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
