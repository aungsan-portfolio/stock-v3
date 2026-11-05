"""Regression: get_active_stops_for_symbol must expose order type so the
broker-evidence verifier does not reject a valid working stop with
EVIDENCE_TYPE_MISSING (which latched the entry gate after a partial fill)."""
import types
import unittest

from alpaca_bridge import AlpacaBridge
from strategies.trailing_stop import verify_broker_stop_order_evidence


def _order(**kw):
    base = dict(
        id="6b9dda94-0000-0000-0000-000000000001",
        symbol="NVDA",
        side="buy",
        order_type="stop",
        qty=2,
        filled_qty=0,
        stop_price=205.0,
        status="new",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _bridge(orders):
    b = object.__new__(AlpacaBridge)
    b._require_connection = lambda: None
    b._get_active_orders = lambda: orders
    return b


class ActiveStopsTypeEvidence(unittest.TestCase):
    def test_stop_dict_includes_type_and_passes_evidence(self):
        stops = _bridge([_order()]).get_active_stops_for_symbol("NVDA")
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]["type"], "stop")
        ok, reason = verify_broker_stop_order_evidence(
            stops[0],
            expected_order_id=stops[0]["id"],
            expected_symbol="NVDA",
            position_side="SELL",
            expected_stop_price=205.0,
            expected_qty=2,
            require_qty=True,
        )
        self.assertTrue(ok, reason)

    def test_wrong_qty_still_rejected(self):
        stops = _bridge([_order(qty=1)]).get_active_stops_for_symbol("NVDA")
        ok, _ = verify_broker_stop_order_evidence(
            stops[0], stops[0]["id"], "NVDA", "SELL", 205.0, expected_qty=2, require_qty=True
        )
        self.assertFalse(ok)

    def test_enum_style_type_is_normalised(self):
        enum_like = types.SimpleNamespace(value="OrderType.STOP")
        stops = _bridge([_order(order_type=enum_like)]).get_active_stops_for_symbol("NVDA")
        self.assertEqual(stops[0]["type"], "stop")


if __name__ == "__main__":
    unittest.main()
