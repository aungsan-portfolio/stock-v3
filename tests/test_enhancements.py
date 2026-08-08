import datetime
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from strategies.intraday_data import _in_market_open_transition
from strategies.session import is_past_opening_buffer
from strategies.trailing_stop import MAX_REMEDIATION_STRIKES, _classify_held_order


class TestScannerGapThreshold(unittest.TestCase):
    def test_gap_threshold_updated(self):
        self.assertLessEqual(
            config.SCAN_MIN_GAP_PCT, 1.4,
            "Gap threshold should be 1.4% or lower"
        )
        self.assertGreater(
            config.SCAN_MIN_GAP_PCT, 0,
            "Gap threshold must be positive"
        )


class TestOpeningBuffer(unittest.TestCase):
    def test_buffer_config_enabled(self):
        self.assertTrue(getattr(config, "ENFORCE_GLOBAL_OPENING_BUFFER", False))
        self.assertEqual(getattr(config, "GLOBAL_OPENING_BUFFER_END_ET", ""), "09:45")

    def test_is_past_opening_buffer_at_0945(self):
        dt_0945 = datetime.datetime(2026, 8, 7, 9, 45, 0)
        self.assertTrue(is_past_opening_buffer(15, now=dt_0945))

    def test_is_past_opening_buffer_at_0931(self):
        dt_0931 = datetime.datetime(2026, 8, 7, 9, 31, 0)
        self.assertFalse(is_past_opening_buffer(15, now=dt_0931))


class TestOrderClassification(unittest.TestCase):
    def _make_order(self, order_type: str, client_order_id: str = "") -> MagicMock:
        o = MagicMock()
        o.type = order_type
        o.order_type = order_type
        o.client_order_id = client_order_id
        return o

    def test_stop_order_recognised(self):
        for t in ["stop", "stop_limit", "trailing_stop"]:
            with self.subTest(order_type=t):
                order = self._make_order(t)
                self.assertEqual(_classify_held_order(order), "stop")

    def test_exit_market_order_recognised(self):
        for prefix in ["dt_", "exit_", "flatten_", "close_", "sell_", "buy_"]:
            with self.subTest(prefix=prefix):
                order = self._make_order("market", client_order_id=f"{prefix}FIG_20260807")
                self.assertEqual(_classify_held_order(order), "exit")

    def test_unrelated_market_order_flagged(self):
        order = self._make_order("market", client_order_id="unknown_order_123")
        self.assertEqual(_classify_held_order(order), "unrelated")

    def test_limit_order_without_prefix_flagged_as_unrelated(self):
        order = self._make_order("limit", client_order_id="random_order_999")
        self.assertEqual(_classify_held_order(order), "unrelated")


class TestMaxRemediationStrikes(unittest.TestCase):
    def test_max_strikes_is_two(self):
        self.assertEqual(
            MAX_REMEDIATION_STRIKES, 2,
            "MAX_REMEDIATION_STRIKES should be 2 (was 3)"
        )


class TestOrphanAutoClean(unittest.TestCase):
    def test_reconcile_startup_auto_cleans_orphans(self):
        from alpaca_bridge import AlpacaBridge
        bridge = AlpacaBridge()
        bridge.positions_plain = MagicMock(return_value=[])
        bridge.working_orders_plain = MagicMock(side_effect=[
            [{"id": "ord-1", "symbol": "AMD", "action": "SELL", "order_type": "limit", "qty": 10}],
            []
        ])
        bridge.entry_gate = MagicMock(halted=False)
        bridge._client = MagicMock()
        
        res = bridge.reconcile_startup_state()
        bridge._client.cancel_order_by_id.assert_called_with("ord-1")
        self.assertEqual(res["auto_cleaned_orphans"], ["ord-1"])
        self.assertTrue(res["clean"])


class TestShutdownGuardPreserving(unittest.TestCase):
    def test_shutdown_guard_preserves_dict_stop_orders(self):
        from alpaca_bridge import AlpacaBridge
        bridge = AlpacaBridge()
        bridge._connected = True
        pos = MagicMock()
        pos.symbol = "META"
        pos.qty = 8.0
        bridge._client = MagicMock()
        bridge._client.get_all_positions.return_value = [pos]
        bridge.working_orders_plain = MagicMock(return_value=[
            {"id": "stop-1", "symbol": "META", "order_type": "stop", "type": "stop", "qty": 4.0}
        ])

        res = bridge.flatten_on_shutdown(only_unprotected=True)
        bridge._client.close_position.assert_not_called()
        self.assertEqual(res.get("preserved_protected"), 1)


class TestSymbolDenylist(unittest.TestCase):
    def test_denylist_excludes_ttd(self):
        import config
        from strategies.scanner.premarket_scanner import _load_universe
        self.assertIn("TTD", config.SYMBOL_DENYLIST)
        universe = _load_universe()
        self.assertNotIn("TTD", universe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
