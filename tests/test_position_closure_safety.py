"""
tests/test_position_closure_safety.py -- Comprehensive position closure and retry safety tests.

Covers:
1. Normalization coverage across diverse broker types (None, int, str, Decimal, dict, object)
2. Pre-exit validation: Close detects already-flat positions and skips broker calls
3. First close fills but outcome polling raises an exception -> reconciles via position query
4. Working order guard: No second close submitted while original close order is still active
5. Partial fill reconciliation and retry with remaining quantity
6. First close rejected -> one controlled retry after reconciliation
7. Anti-reversal guard: Position reversal detected -> refuse retry to prevent reversal escalation
8. Idempotent double close: Second call detects flat and submits no orders
"""

import unittest
from unittest.mock import MagicMock, patch
from decimal import Decimal
from dataclasses import dataclass
import order_exec
from alpaca_bridge import AlpacaBridge, normalize_position_qty


@dataclass
class MockBrokerPosition:
    qty: object


class TestPositionClosureSafety(unittest.TestCase):

    def setUp(self):
        self.bridge = AlpacaBridge(mode="daytrade", dry_run=False)
        self.bridge._connected = True
        self.bridge._client = MagicMock()
        self.bridge._conn_health = MagicMock()
        self.bridge._conn_health.mark_healthy = MagicMock()
        self.bridge._conn_health.mark_unhealthy = MagicMock()

    def test_normalize_position_qty_exhaustive(self):
        """Test normalization across all broker response formats."""
        self.assertEqual(normalize_position_qty(None), 0.0)
        self.assertEqual(normalize_position_qty(0), 0.0)
        self.assertEqual(normalize_position_qty(0.0), 0.0)
        self.assertEqual(normalize_position_qty("0"), 0.0)
        self.assertEqual(normalize_position_qty("0.0"), 0.0)
        self.assertEqual(normalize_position_qty(Decimal("0")), 0.0)
        self.assertEqual(normalize_position_qty({"qty": "0"}), 0.0)
        self.assertEqual(normalize_position_qty({"position": 0}), 0.0)
        self.assertEqual(normalize_position_qty(MockBrokerPosition(qty="0")), 0.0)
        self.assertEqual(normalize_position_qty(MockBrokerPosition(qty=Decimal("0.00"))), 0.0)

        # Non-zero quantities
        self.assertEqual(normalize_position_qty(MockBrokerPosition(qty="-2")), -2.0)
        self.assertEqual(normalize_position_qty(MockBrokerPosition(qty=5)), 5.0)
        self.assertEqual(normalize_position_qty({"qty": "12.5"}), 12.5)
        self.assertEqual(normalize_position_qty(Decimal("-10.25")), -10.25)

    def test_pre_exit_validation_detects_already_flat(self):
        """Pre-exit check: If broker position is already flat, close_position returns True immediately."""
        self.bridge.get_position = MagicMock(return_value=0.0)
        res = self.bridge.close_position("AAPL")
        self.assertTrue(res)
        self.bridge._client.close_position.assert_not_called()

    def test_first_close_fills_but_polling_raises_exception(self):
        """When order polling raises an error, bridge falls back to authoritative broker position query."""
        self.bridge.get_position = MagicMock(side_effect=[10.0, 0.0])
        mock_order = MagicMock(id="close_101")
        self.bridge._client.close_position.return_value = mock_order
        self.bridge._await_order_outcome = MagicMock(side_effect=RuntimeError("Network drop during polling"))

        res = self.bridge.close_position("AAPL")
        self.assertTrue(res, "Must return True because broker position query confirmed flat")
        # Crucial: only 1 close order submitted (no duplicate retry)
        self.assertEqual(self.bridge._client.close_position.call_count, 1)

    def test_working_order_guard_prevents_duplicate_retry(self):
        """If an order is still working, retry is blocked until working orders are cleared/reconciled."""
        self.bridge.get_position = MagicMock(side_effect=[10.0, 10.0, 0.0])
        mock_order = MagicMock(id="close_102")
        self.bridge._client.close_position.return_value = mock_order

        # Order polling times out (still working)
        self.bridge._await_order_outcome = MagicMock(return_value=order_exec.OrderResult(
            outcome=order_exec.REJECTED, status="Working", filled=0.0, remaining=10.0
        ))

        mock_working_order = MagicMock(symbol="AAPL", status=MagicMock(value="new"))
        self.bridge._get_active_orders = MagicMock(return_value=[mock_working_order])

        # Cancel orders clearing makes it flat on re-verify
        with patch.object(self.bridge, "_cancel_all_orders_for_symbol") as mock_cancel:
            res = self.bridge.close_position("AAPL")
            self.assertTrue(res)
            # Cancel was invoked to clear the stalled order
            self.assertTrue(mock_cancel.called)

    def test_first_close_rejected_triggers_controlled_retry(self):
        """When first close is rejected, a controlled retry is submitted once after reconciliation."""
        # Initial pos 10, still 10 after first failure, flat (0) after retry
        self.bridge.get_position = MagicMock(side_effect=[10.0, 10.0, 10.0, 0.0])
        self.bridge._get_active_orders = MagicMock(return_value=[])

        first_order = MagicMock(id="close_rej_1")
        retry_order = MagicMock(id="close_ret_2")
        self.bridge._client.close_position.side_effect = [first_order, retry_order]

        # First attempt rejected, second attempt filled
        self.bridge._await_order_outcome = MagicMock(side_effect=[
            order_exec.OrderResult(outcome=order_exec.REJECTED, status="Rejected", filled=0.0, remaining=10.0),
            order_exec.OrderResult(outcome=order_exec.FILLED, status="Filled", filled=10.0, remaining=0.0),
        ])

        res = self.bridge.close_position("AAPL")
        self.assertTrue(res)
        self.assertEqual(self.bridge._client.close_position.call_count, 2)

    def test_partial_fill_reconciliation_and_retry(self):
        """When first close partially fills, bridge reconciles remaining quantity and submits controlled retry."""
        # Initial pos 10, partial fill leaves 4 shares, after retry position is 0 (flat)
        self.bridge.get_position = MagicMock(side_effect=[10.0, 4.0, 4.0, 0.0])
        self.bridge._get_active_orders = MagicMock(return_value=[])

        first_order = MagicMock(id="close_part_1")
        retry_order = MagicMock(id="close_part_2")
        self.bridge._client.close_position.side_effect = [first_order, retry_order]

        # First attempt partially filled (6 of 10 filled, 4 remaining), second attempt filled 4
        self.bridge._await_order_outcome = MagicMock(side_effect=[
            order_exec.OrderResult(outcome=order_exec.PARTIALLY_FILLED, status="Partially Filled", filled=6.0, remaining=4.0),
            order_exec.OrderResult(outcome=order_exec.FILLED, status="Filled", filled=4.0, remaining=0.0),
        ])

        res = self.bridge.close_position("AAPL")
        self.assertTrue(res)
        self.assertEqual(self.bridge._client.close_position.call_count, 2)

    def test_anti_reversal_guard_blocks_retry_if_position_flips(self):
        """If an initial LONG position flipped to SHORT, retry is strictly refused to prevent escalation."""
        # Initial +10 LONG, but broker re-query reports -5 SHORT!
        self.bridge.get_position = MagicMock(side_effect=[10.0, -5.0, -5.0])
        self.bridge._get_active_orders = MagicMock(return_value=[])

        close_order = MagicMock(id="close_flip_1")
        self.bridge._client.close_position.return_value = close_order
        self.bridge._await_order_outcome = MagicMock(return_value=order_exec.OrderResult(
            outcome=order_exec.REJECTED, status="Overfilled", filled=15.0, remaining=0.0
        ))

        res = self.bridge.close_position("AAPL")
        self.assertFalse(res, "Must return False and refuse duplicate close to avoid long reversal")
        # Crucial: exactly 1 close order, no second retry
        self.assertEqual(self.bridge._client.close_position.call_count, 1)

    def test_idempotent_double_close_submits_nothing_second_time(self):
        """Calling close_position twice in a row: second call detects flat and submits 0 orders."""
        self.bridge.get_position = MagicMock(side_effect=[5.0, 0.0, 0.0])
        mock_order = MagicMock(id="close_idem_1")
        self.bridge._client.close_position.return_value = mock_order
        self.bridge._await_order_outcome = MagicMock(return_value=order_exec.OrderResult(
            outcome=order_exec.FILLED, status="Filled", filled=5.0, remaining=0.0
        ))

        # First call closes position
        res1 = self.bridge.close_position("QQQ")
        self.assertTrue(res1)
        self.assertEqual(self.bridge._client.close_position.call_count, 1)

        # Second call detects flat
        res2 = self.bridge.close_position("QQQ")
        self.assertTrue(res2)
        # Call count remains 1
        self.assertEqual(self.bridge._client.close_position.call_count, 1)


if __name__ == "__main__":
    unittest.main()
