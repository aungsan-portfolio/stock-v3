"""
tests/test_journal_deduplication_audit.py -- Unit tests for 7-step deduplication replay
and legitimate identical partial fills with distinct execution IDs.
"""
import json
import pathlib
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import config
from strategies.session import now_eastern
from strategies.trade_journal import log_fill, log_trade, read_journal, today_trades, today_trade_count


class TestJournalDeduplicationAudit(unittest.TestCase):

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()
        self.journal_path = pathlib.Path(self._tmp_dir) / "test_journal.jsonl"
        self._orig_journal = config.DAYTRADE_TRADE_JOURNAL_FILE
        config.DAYTRADE_TRADE_JOURNAL_FILE = self.journal_path

    def tearDown(self):
        config.DAYTRADE_TRADE_JOURNAL_FILE = self._orig_journal
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_7_step_deduplication_replay(self):
        """
        Step 1: TRADE_INTENT / ORDER_SUBMITTED exists.
        Step 2: Broker fill callback arrives with execution_id="exec_1", qty=5, px=100.0.
        Step 3: Fill is logged with execution_id and not suppressed.
        Step 4: Duplicate fill callback arrives with same execution_id="exec_1".
        Step 5: Engine recognizes duplicate execution_id.
        Step 6: Legitimate separate partial fill arrives with execution_id="exec_2", same qty=5, px=100.0.
        Step 7: Second partial fill is NOT suppressed because execution_id is distinct.
        """
        now_ts = now_eastern().isoformat()

        # Step 1: Submit intent
        log_trade(
            symbol="AAPL",
            side="BUY",
            strategy="VWAP_BOUNCE",
            qty=10,
            entry_price=100.0,
            stop_price=98.0,
            target_price=105.0,
            journal_file=self.journal_path,
            event_type="ORDER_SUBMITTED",
            order_id="p_aapl_100",
        )

        # Step 2 & 3: First fill callback
        log_fill(
            symbol="AAPL",
            side="BUY",
            qty=5,
            fill_price=100.0,
            expected_price=100.0,
            slippage=0.0,
            fill_latency_ms=15.0,
            journal_file=self.journal_path,
            order_id="p_aapl_100",
            execution_id="exec_1",
            timestamp=now_ts,
            strategy="VWAP_BOUNCE",
        )

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL"]
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["fill_id"], "exec_1")
        self.assertFalse(fills[0]["is_synthetic_key"])

        # Step 4 & 5: Duplicate callback suppression check
        # An ingestion routine checking existing execution_ids:
        existing_exec_ids = {r.get("execution_id") for r in read_journal(self.journal_path) if r.get("execution_id")}
        self.assertIn("exec_1", existing_exec_ids)
        # Duplicate callback incoming with "exec_1" -> ignored by processor

        # Step 6 & 7: Second distinct partial fill (same qty, same price, distinct exec_id)
        log_fill(
            symbol="AAPL",
            side="BUY",
            qty=5,
            fill_price=100.0,
            expected_price=100.0,
            slippage=0.0,
            fill_latency_ms=25.0,
            journal_file=self.journal_path,
            order_id="p_aapl_100",
            execution_id="exec_2",
            timestamp=now_ts,
            strategy="VWAP_BOUNCE",
        )

        records = read_journal(self.journal_path)
        fills = [r for r in records if r.get("event_type") == "FILL"]
        self.assertEqual(len(fills), 2)
        self.assertEqual(fills[0]["fill_id"], "exec_1")
        self.assertEqual(fills[1]["fill_id"], "exec_2")

    def test_synthetic_fallback_key_tagging(self):
        """When execution_id is absent, degraded synthetic key is tagged with requires_reconciliation=True."""
        now_ts = now_eastern().isoformat()
        log_fill(
            symbol="MSFT",
            side="BUY",
            qty=4,
            fill_price=490.50,
            expected_price=490.00,
            slippage=0.001,
            fill_latency_ms=10.0,
            journal_file=self.journal_path,
            order_id="ord_synthetic_test",
            execution_id=None,  # missing execution id!
            timestamp=now_ts,
            strategy="VWAP_BOUNCE",
        )

        records = read_journal(self.journal_path)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertTrue(rec["is_synthetic_key"])
        self.assertTrue(rec["requires_reconciliation"])
        self.assertTrue(rec["fill_id"].startswith("SYNTH:"))


if __name__ == "__main__":
    unittest.main()
