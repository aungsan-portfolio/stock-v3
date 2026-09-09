"""
tests/test_protection_state_machine.py -- Comprehensive scenario replays and unit tests
for the hardened ProtectionState state machine in strategies/entry_lifecycle.py.
"""
import unittest
from unittest.mock import MagicMock

from strategies.base import TradeSignal, StrategyName
from strategies.entry_lifecycle import EntryOrderLifecycle, ProtectionState
import pathlib
import tempfile
import shutil
import config
from strategies.halt_manager import halt_manager, LATCHING_HALT_PROTECTION_FAILURE


class TestProtectionStateMachine(unittest.TestCase):

    def setUp(self):
        halt_manager.reset_all()
        self._tmp_dir = tempfile.mkdtemp()
        self._orig_journal = config.DAYTRADE_TRADE_JOURNAL_FILE
        config.DAYTRADE_TRADE_JOURNAL_FILE = pathlib.Path(self._tmp_dir) / "test_journal.jsonl"
        self.trailing_mgr = MagicMock()
        self.lifecycle = EntryOrderLifecycle(self.trailing_mgr)
        self.sig = TradeSignal(
            symbol="AVGO",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.75,
            entry_price=368.11,
            stop_price=367.91,
            target_price=368.84,
            atr=0.22,
            risk_per_share=0.20,
            reason="Test AVGO signal",
        )

    def tearDown(self):
        halt_manager.reset_all()
        config.DAYTRADE_TRADE_JOURNAL_FILE = self._orig_journal
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_avgo_replay_instant_stop_fill_and_target_cancel(self):
        """Replay the exact 2026-09-09 02:40 AVGO incident:
        Parent fills 5 shares @ 367.94.
        Stop fills 5 shares @ 367.86.
        Target leg cancels due to broker OCO.
        Broker position is FLAT (0 shares).
        INVARIANT: Must resolve to CLOSED_BY_STOP without protection latch or emergency close!
        """
        entry = self.lifecycle.register_submission(
            parent_order_id="p_avgo_0240",
            signal=self.sig,
            requested_qty=5.0,
            stop_order_id="stop_avgo_0240",
            target_order_id="target_avgo_0240",
        )

        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=0.0)  # Position is flat!
        mock_bridge.close_position = MagicMock()

        # Stop is filled, target is canceled (both terminal)
        def mock_snapshot(order_id):
            if order_id == "stop_avgo_0240":
                return {"id": "stop_avgo_0240", "status": "filled", "qty": 5.0, "filled_qty": 5.0}
            if order_id == "target_avgo_0240":
                return {"id": "target_avgo_0240", "status": "canceled", "qty": 5.0, "filled_qty": 0.0}
            return None

        mock_bridge.get_order_snapshot = MagicMock(side_effect=mock_snapshot)

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.CLOSED_BY_STOP)

        ok = self.lifecycle.on_parent_fill(
            bridge=mock_bridge,
            parent_order_id="p_avgo_0240",
            cumulative_filled_qty=5.0,
            average_fill_price=367.94,
        )

        self.assertTrue(ok)
        self.assertTrue(halt_manager.entries_allowed())
        self.assertNotIn(LATCHING_HALT_PROTECTION_FAILURE, halt_manager.active_halts())
        mock_bridge.close_position.assert_not_called()
        self.assertNotIn("p_avgo_0240", self.lifecycle.pending_snapshot())

    def test_target_filled_scenario(self):
        """Target reaches profit, stop is canceled, position flat."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_target_win",
            signal=self.sig,
            requested_qty=10.0,
            stop_order_id="s_target_win",
            target_order_id="t_target_win",
        )

        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=0.0)

        def mock_snapshot(order_id):
            if order_id == "t_target_win":
                return {"id": "t_target_win", "status": "filled", "qty": 10.0, "filled_qty": 10.0}
            if order_id == "s_target_win":
                return {"id": "s_target_win", "status": "canceled", "qty": 10.0, "filled_qty": 0.0}
            return None

        mock_bridge.get_order_snapshot = MagicMock(side_effect=mock_snapshot)

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.CLOSED_BY_TARGET)

        ok = self.lifecycle.on_parent_fill(
            bridge=mock_bridge,
            parent_order_id="p_target_win",
            cumulative_filled_qty=10.0,
            average_fill_price=368.00,
        )
        self.assertTrue(ok)
        self.assertTrue(halt_manager.entries_allowed())

    def test_open_position_fully_protected(self):
        """Normal active position: 10 shares open, stop active, target active."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_norm_open",
            signal=self.sig,
            requested_qty=10.0,
            stop_order_id="s_norm_open",
            target_order_id="t_norm_open",
        )

        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=10.0)
        mock_bridge.get_order_snapshot = MagicMock(side_effect=lambda oid: {"id": oid, "status": "accepted", "qty": 10.0, "filled_qty": 0.0})

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.PROTECTED_COMPLETE)

        ok = self.lifecycle.on_parent_fill(
            bridge=mock_bridge,
            parent_order_id="p_norm_open",
            cumulative_filled_qty=10.0,
            average_fill_price=368.10,
        )
        self.assertTrue(ok)
        self.trailing_mgr.apply_confirmed_fill.assert_called_once()
        self.assertTrue(halt_manager.entries_allowed())

    def test_open_position_target_unavailable_non_fatal(self):
        """Target unavailable / canceled, but stop is active and covers open shares."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_target_missing",
            signal=self.sig,
            requested_qty=8.0,
            stop_order_id="s_target_missing",
            target_order_id="t_target_missing",
        )

        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=8.0)

        def mock_snapshot(order_id):
            if order_id == "s_target_missing":
                return {"id": "s_target_missing", "status": "accepted", "qty": 8.0, "filled_qty": 0.0}
            if order_id == "t_target_missing":
                return {"id": "t_target_missing", "status": "canceled", "qty": 8.0, "filled_qty": 0.0}
            return None

        mock_bridge.get_order_snapshot = MagicMock(side_effect=mock_snapshot)

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.PROTECTED_STOP_ONLY)

        ok = self.lifecycle.on_parent_fill(
            bridge=mock_bridge,
            parent_order_id="p_target_missing",
            cumulative_filled_qty=8.0,
            average_fill_price=368.10,
        )
        self.assertTrue(ok)
        self.assertTrue(halt_manager.entries_allowed())
        self.assertNotIn(LATCHING_HALT_PROTECTION_FAILURE, halt_manager.active_halts())

    def test_open_position_insufficient_stop_protection_failure(self):
        """Genuine safety failure: Open position 10 shares, but stop is missing/canceled."""
        entry = self.lifecycle.register_submission(
            parent_order_id="p_unprotected",
            signal=self.sig,
            requested_qty=10.0,
            stop_order_id="s_unprotected",
            target_order_id="t_unprotected",
        )

        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=10.0)
        mock_bridge.close_position = MagicMock()

        # Stop is canceled, leaving open position naked!
        def mock_snapshot(order_id):
            if order_id == "s_unprotected":
                return {"id": "s_unprotected", "status": "canceled", "qty": 10.0, "filled_qty": 0.0}
            if order_id == "t_unprotected":
                return {"id": "t_unprotected", "status": "accepted", "qty": 10.0, "filled_qty": 0.0}
            return None

        mock_bridge.get_order_snapshot = MagicMock(side_effect=mock_snapshot)

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.PROTECTION_FAILURE)

        ok = self.lifecycle.on_parent_fill(
            bridge=mock_bridge,
            parent_order_id="p_unprotected",
            cumulative_filled_qty=10.0,
            average_fill_price=368.10,
        )
        self.assertFalse(ok)
        self.assertFalse(halt_manager.entries_allowed())
        self.assertIn(LATCHING_HALT_PROTECTION_FAILURE, halt_manager.active_halts())
        mock_bridge.close_position.assert_called_once_with("AVGO")

    def test_aapl_replacement_lineage_resolution(self):
        """AAPL stop replacement chain: stop_1 was replaced by stop_2.
        Engine must follow lineage and adopt active stop_2 without latching."""
        sig_aapl = TradeSignal(
            symbol="AAPL",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.80,
            entry_price=315.40,
            stop_price=315.26,
            target_price=315.92,
            atr=0.15,
            risk_per_share=0.14,
            reason="Test AAPL signal",
        )

        entry = self.lifecycle.register_submission(
            parent_order_id="p_aapl_rep",
            signal=sig_aapl,
            requested_qty=12.0,
            stop_order_id="stop_aapl_orig",
            target_order_id="target_aapl_orig",
        )

        mock_bridge = MagicMock()
        mock_bridge.get_position = MagicMock(return_value=12.0)

        def mock_snapshot(order_id):
            if order_id == "stop_aapl_orig":
                return {"id": "stop_aapl_orig", "status": "replaced", "qty": 12.0, "replaced_by": "stop_aapl_new"}
            if order_id == "stop_aapl_new":
                return {"id": "stop_aapl_new", "status": "accepted", "qty": 12.0, "filled_qty": 0.0}
            if order_id == "target_aapl_orig":
                return {"id": "target_aapl_orig", "status": "accepted", "qty": 12.0, "filled_qty": 0.0}
            return None

        mock_bridge.get_order_snapshot = MagicMock(side_effect=mock_snapshot)

        eval_res = self.lifecycle.evaluate_protection_state(mock_bridge, entry)
        self.assertEqual(eval_res.state, ProtectionState.PROTECTED_COMPLETE)
        self.assertEqual(eval_res.resolved_stop_id, "stop_aapl_new")

        ok = self.lifecycle.on_parent_fill(
            bridge=mock_bridge,
            parent_order_id="p_aapl_rep",
            cumulative_filled_qty=12.0,
            average_fill_price=315.40,
        )
        self.assertTrue(ok)
        self.assertTrue(halt_manager.entries_allowed())


if __name__ == "__main__":
    unittest.main()
