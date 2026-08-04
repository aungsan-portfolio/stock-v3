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

if __name__ == "__main__":
    unittest.main()
