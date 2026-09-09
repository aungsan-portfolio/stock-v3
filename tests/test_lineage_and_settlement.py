"""
tests/test_lineage_and_settlement.py -- Unit tests for replacement order lineage cycle detection,
depth-exceeded escalation, mixed exits attribution, and settlement window retries.
"""
import math
import pathlib
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock

import config
from strategies.base import TradeSignal, StrategyName
from strategies.entry_lifecycle import EntryOrderLifecycle, ProtectionState, LineageOutcome
from strategies.halt_manager import halt_manager


class TestLineageAndSettlement(unittest.TestCase):

    def setUp(self):
        halt_manager.reset_all()
        self._tmp_dir = tempfile.mkdtemp()
        self._orig_journal = config.DAYTRADE_TRADE_JOURNAL_FILE
        config.DAYTRADE_TRADE_JOURNAL_FILE = pathlib.Path(self._tmp_dir) / "test_journal.jsonl"
        self.trailing_mgr = MagicMock()
        self.lifecycle = EntryOrderLifecycle(self.trailing_mgr)
        self.sig = TradeSignal(
            symbol="AAPL",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.80,
            entry_price=150.0,
            stop_price=148.0,
            target_price=155.0,
            atr=1.5,
            risk_per_share=2.0,
            reason="Test signal",
        )

    def tearDown(self):
        halt_manager.reset_all()
        config.DAYTRADE_TRADE_JOURNAL_FILE = self._orig_journal
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_lineage_cycle_detection(self):
        """Replacement chain A -> B -> A must detect cycle and escalate."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_cycle",
            signal=self.sig,
            requested_qty=10.0,
            stop_order_id="stop_A",
            target_order_id="target_A",
        )
        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=10.0)

        snapshots = {
            "stop_A": {"id": "stop_A", "status": "replaced", "qty": 10.0, "replaced_by": "stop_B"},
            "stop_B": {"id": "stop_B", "status": "replaced", "qty": 10.0, "replaced_by": "stop_A"},
            "target_A": {"id": "target_A", "status": "accepted", "qty": 10.0, "filled_qty": 0.0},
        }
        mock_bridge.get_order_snapshot = MagicMock(side_effect=lambda oid: snapshots.get(oid))

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.RECONCILIATION_REQUIRED)
        self.assertEqual(eval_res.stop_lineage_outcome, LineageOutcome.LINEAGE_CYCLE_DETECTED)

    def test_lineage_depth_exceeded(self):
        """Replacement chain longer than max_depth must report LINEAGE_DEPTH_EXCEEDED."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_deep",
            signal=self.sig,
            requested_qty=5.0,
            stop_order_id="stop_0",
            target_order_id="target_0",
        )
        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=5.0)

        snapshots = {
            f"stop_{i}": {"id": f"stop_{i}", "status": "replaced", "qty": 5.0, "replaced_by": f"stop_{i+1}"}
            for i in range(10)
        }
        snapshots["target_0"] = {"id": "target_0", "status": "accepted", "qty": 5.0, "filled_qty": 0.0}
        mock_bridge.get_order_snapshot = MagicMock(side_effect=lambda oid: snapshots.get(oid))

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.RECONCILIATION_REQUIRED)
        self.assertEqual(eval_res.stop_lineage_outcome, LineageOutcome.LINEAGE_DEPTH_EXCEEDED)

    def test_mixed_exits_resolution(self):
        """Partial stop fill (6 shares) + partial target fill (4 shares) = 10 shares flat -> CLOSED_BY_MIXED_EXITS."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_mixed",
            signal=self.sig,
            requested_qty=10.0,
            stop_order_id="stop_mix",
            target_order_id="target_mix",
        )
        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=0.0)

        snapshots = {
            "stop_mix": {"id": "stop_mix", "status": "filled", "qty": 10.0, "filled_qty": 6.0},
            "target_mix": {"id": "target_mix", "status": "filled", "qty": 10.0, "filled_qty": 4.0},
        }
        mock_bridge.get_order_snapshot = MagicMock(side_effect=lambda oid: snapshots.get(oid))

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.CLOSED_BY_MIXED_EXITS)
        self.assertAlmostEqual(eval_res.stop_filled_qty, 6.0)
        self.assertAlmostEqual(eval_res.target_filled_qty, 4.0)

    def test_overfill_reconciliation_required(self):
        """Exits report 12 shares when only 10 were requested/filled -> overfill safety hazard."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_overfill",
            signal=self.sig,
            requested_qty=10.0,
            stop_order_id="stop_over",
            target_order_id="target_over",
        )
        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=0.0)

        snapshots = {
            "stop_over": {"id": "stop_over", "status": "filled", "qty": 12.0, "filled_qty": 12.0},
            "target_over": {"id": "target_over", "status": "canceled", "qty": 10.0, "filled_qty": 0.0},
        }
        mock_bridge.get_order_snapshot = MagicMock(side_effect=lambda oid: snapshots.get(oid))

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.RECONCILIATION_REQUIRED)
        self.assertIn("OVERFILL", eval_res.reason)

    def test_underfill_reconciliation_required(self):
        """Broker position is 0 but only 8 shares of exits confirmed for 10 share entry -> missing exit fill."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_underfill",
            signal=self.sig,
            requested_qty=10.0,
            stop_order_id="stop_under",
            target_order_id="target_under",
        )
        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=0.0)

        snapshots = {
            "stop_under": {"id": "stop_under", "status": "filled", "qty": 10.0, "filled_qty": 8.0},
            "target_under": {"id": "target_under", "status": "canceled", "qty": 10.0, "filled_qty": 0.0},
        }
        mock_bridge.get_order_snapshot = MagicMock(side_effect=lambda oid: snapshots.get(oid))

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.RECONCILIATION_REQUIRED)
        self.assertIn("confirmed exit quantity (8.0000) < entry filled (10.0000)", eval_res.reason)

    def test_settlement_retry_resolves_delayed_stop_fill(self):
        """On parent fill, stop exit report is initially delayed, but arrives within retry window."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_delayed_fill",
            signal=self.sig,
            requested_qty=5.0,
            stop_order_id="stop_delayed",
            target_order_id="target_delayed",
        )
        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=0.0)

        call_count = {"count": 0}

        def mock_snapshot(order_id):
            if order_id == "stop_delayed":
                call_count["count"] += 1
                if call_count["count"] <= 1:
                    # Initially still open/unfilled
                    return {"id": "stop_delayed", "status": "accepted", "qty": 5.0, "filled_qty": 0.0}
                else:
                    # Fills on retry
                    return {"id": "stop_delayed", "status": "filled", "qty": 5.0, "filled_qty": 5.0}
            if order_id == "target_delayed":
                return {"id": "target_delayed", "status": "canceled", "qty": 5.0, "filled_qty": 0.0}
            return None

        mock_bridge.get_order_snapshot = MagicMock(side_effect=mock_snapshot)

        ok = self.lifecycle.on_parent_fill(
            bridge=mock_bridge,
            parent_order_id="p_delayed_fill",
            cumulative_filled_qty=5.0,
            average_fill_price=150.0,
        )
        self.assertTrue(ok)
        self.assertNotIn("p_delayed_fill", self.lifecycle.pending_snapshot())


if __name__ == "__main__":
    unittest.main()
