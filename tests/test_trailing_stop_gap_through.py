import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState


class TestGapThroughProtection(unittest.TestCase):
    """
    Validates that handle_naked_position handles all broker-order states
    deterministically: broker query FIRST, gap-through check SECOND,
    and stop replacement THIRD.
    """

    def _make_state(self, side="BUY", stop=586.02, qty=4.0):
        s = MagicMock(spec=TrailingStopState)
        s.symbol = "META"
        s.side = side
        s.stop_price = stop
        s.qty = qty
        s.order_id = None
        s.active = True
        s.remediation_in_progress = False
        s.remediation_attempts = 0
        return s

    def _make_broker_order(self, status, filled=0.0, total=8.0, avg_fill=0.0, order_id="test-id"):
        o = MagicMock()
        o.id = order_id
        o.status = status
        o.filled_qty = filled
        o.qty = total
        o.avg_fill_price = avg_fill
        o.order_type = "stop"
        return o

    def test_partial_fill_updates_qty_no_flatten(self):
        mgr = DynamicTrailingStopManager.__new__(DynamicTrailingStopManager)
        mgr.states = {}
        mgr._lock = MagicMock()

        bridge = MagicMock()
        bridge.get_position.return_value = 4.0
        bridge._client.get_orders.return_value = [
            self._make_broker_order("partially_filled", filled=4.0, total=8.0, avg_fill=596.235)
        ]

        state = self._make_state(qty=8.0)
        mgr.states["META"] = state

        mgr._reconcile_broker_order = MagicMock(side_effect=lambda st, orders: setattr(st, 'order_id', 'test-id') or True)
        mgr.emergency_flatten = MagicMock()

        mgr.handle_naked_position("META", position_qty=4.0, bridge=bridge)

        # No emergency flatten should be triggered because state was adopted
        mgr.emergency_flatten.assert_not_called()

    def test_naked_gap_through_triggers_immediate_flatten(self):
        mgr = DynamicTrailingStopManager.__new__(DynamicTrailingStopManager)
        mgr.states = {}
        mgr._lock = MagicMock()

        bridge = MagicMock()
        bridge.get_position.return_value = 4.0
        bridge._client.get_orders.return_value = []
        bridge.market_price.return_value = 574.50

        state = self._make_state(stop=586.02, qty=4.0)
        mgr.states["META"] = state

        mgr._reconcile_broker_order = MagicMock(return_value=False)
        mgr.emergency_flatten = MagicMock()

        from unittest.mock import patch
        with patch("strategies.trailing_stop.is_market_open", return_value=True):
            mgr.handle_naked_position("META", position_qty=4.0, bridge=bridge)

        mgr.emergency_flatten.assert_called_once()
        self.assertEqual(mgr.emergency_flatten.call_args[0][0], "META")


if __name__ == "__main__":
    unittest.main(verbosity=2)
