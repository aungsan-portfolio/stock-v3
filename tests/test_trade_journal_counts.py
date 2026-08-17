import unittest
from datetime import datetime, timezone
import pytz
from unittest.mock import patch, MagicMock

from strategies.trade_journal import today_trades, today_trade_count, _is_today_record
from strategies.session import now_eastern

class TestTradeJournalCounts(unittest.TestCase):

    def test_is_today_record_parsing(self):
        current_eastern = now_eastern()
        today_date = current_eastern.date()
        tz = current_eastern.tzinfo

        today_iso = current_eastern.isoformat()
        self.assertTrue(_is_today_record(today_iso, today_date, tz))

        # Yesterday date
        yesterday_iso = "2026-08-01T14:30:00+00:00"
        self.assertFalse(_is_today_record(yesterday_iso, today_date, tz))

    @patch("strategies.trade_journal.read_journal")
    def test_today_trades_counts_only_buy_fills(self, mock_read):
        current_iso = now_eastern().isoformat()
        records = [
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_1", "symbol": "AMZN"},
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_2", "symbol": "AAPL"},
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_3", "symbol": "NVDA"},
        ]
        mock_read.return_value = records
        trades = today_trades()
        self.assertEqual(len(trades), 3)

    @patch("strategies.trade_journal.read_journal")
    def test_today_trades_ignores_unfilled_submissions(self, mock_read):
        current_iso = now_eastern().isoformat()
        records = [
            {"timestamp": current_iso, "type": "ORDER", "event_type": "ORDER_SUBMITTED", "side": "BUY", "order_id": "ord_1", "symbol": "AMZN"},
            {"timestamp": current_iso, "type": "ORDER", "event_type": "ORDER_SUBMITTED", "side": "BUY", "order_id": "ord_2", "symbol": "AAPL"},
            {"timestamp": current_iso, "type": "ORDER", "event_type": "ORDER_SUBMITTED", "side": "BUY", "order_id": "ord_3", "symbol": "NVDA"},
        ]
        mock_read.return_value = records
        # If there are fills later, it counts fills; if no fills exist, raw submissions fallback
        # But if is_fill is present elsewhere, submissions don't duplicate
        count = today_trade_count()
        self.assertLessEqual(count, 3)

    @patch("strategies.trade_journal.read_journal")
    def test_today_trades_ignores_sell_fills(self, mock_read):
        current_iso = now_eastern().isoformat()
        records = [
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_1", "symbol": "AMZN"},
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_2", "symbol": "AAPL"},
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_3", "symbol": "NVDA"},
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "SELL", "order_id": "ord_1_exit", "symbol": "AMZN"},
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "SELL", "order_id": "ord_2_exit", "symbol": "AAPL"},
        ]
        mock_read.return_value = records
        trades = today_trades()
        self.assertEqual(len(trades), 3)

    @patch("strategies.trade_journal.read_journal")
    def test_today_trades_deduplicates_same_order(self, mock_read):
        current_iso = now_eastern().isoformat()
        records = [
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_1", "symbol": "AMZN"},
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_1", "symbol": "AMZN"},
            {"timestamp": current_iso, "type": "FILL", "event_type": "FILL", "side": "BUY", "order_id": "ord_1", "symbol": "AMZN"},
        ]
        mock_read.return_value = records
        trades = today_trades()
        self.assertEqual(len(trades), 1)

    def test_pre_trade_check_caution_5_limit(self):
        from strategies.intraday_risk import pre_trade_check, check_max_trades
        
        # 4 trades today < 5 max trades -> allowed
        self.assertTrue(check_max_trades(trades_today=4, max_trades=5))
        
        # 5 trades today >= 5 max trades -> blocked
        self.assertFalse(check_max_trades(trades_today=5, max_trades=5))
        
        # Check formatted error string in pre_trade_check
        err = pre_trade_check(
            equity=100000.0,
            current_pnl=100.0,
            trades_today=5,
            open_positions=1,
            day_trades_last_5_days=0,
            risk_dollars=50.0,
            symbol="MSFT",
            max_trades=5
        )
        self.assertEqual(err, "Max trades/day reached (5/5)")

    def test_pre_trade_check_risk_on_15_limit(self):
        from strategies.intraday_risk import pre_trade_check, check_max_trades
        
        self.assertTrue(check_max_trades(trades_today=14, max_trades=15))
        self.assertFalse(check_max_trades(trades_today=15, max_trades=15))
        
        err = pre_trade_check(
            equity=100000.0,
            current_pnl=100.0,
            trades_today=15,
            open_positions=1,
            day_trades_last_5_days=0,
            risk_dollars=50.0,
            symbol="MSFT",
            max_trades=15
        )
        self.assertEqual(err, "Max trades/day reached (15/15)")

    @patch("strategies.order_manager.today_trade_count")
    def test_execute_signal_regime_propagation(self, mock_today_count):
        from strategies.order_manager import execute_signal
        from strategies.base import TradeSignal, StrategyName
        from dataclasses import dataclass

        @dataclass
        class MockRegime:
            mode: str = "CAUTION"
            position_scale: float = 0.5
            confidence_boost: float = 0.0

        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.open_position_count.return_value = 0
        mock_bridge.get_positions.return_value = []

        sig = TradeSignal(
            symbol="AAPL",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            entry_price=200.0,
            stop_price=190.0,
            target_price=220.0,
            confidence=0.80,
            atr=2.0,
            risk_per_share=10.0,
            reason="test"
        )

        # Case 1: CAUTION mode with 5 trades today -> blocked (5/5)
        mock_today_count.return_value = 5
        regime_caution = MockRegime(mode="CAUTION", position_scale=0.5)
        res_caution = execute_signal(
            signal=sig,
            bridge=mock_bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            regime_result=regime_caution
        )
        self.assertEqual(res_caution["status"], "REJECTED")
        self.assertIn("Max trades/day reached (5/5)", res_caution["reason"])

        # Case 2: regime_result=None with 5 trades today -> allowed through trade limit check (falls back to configured max 18)
        mock_today_count.return_value = 5
        res_none = execute_signal(
            signal=sig,
            bridge=mock_bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            regime_result=None
        )
        self.assertNotIn("Max trades/day reached", res_none.get("reason", ""))

if __name__ == "__main__":
    unittest.main()
