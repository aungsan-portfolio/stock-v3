"""
tests/test_bracket_reconciliation_hardening.py — Tests for Group-Level Position Coverage & Held Leg Visibility.

Covers Amendment 4:
  • Visibility of Alpaca 'held' conditional bracket orders via QueryOrderStatus.ALL merge.
  • Preservation of Take Profit (TP) orders when Stop Loss (SL) coverage is complete.
  • Preserves TP and attaches missing SL without cancelling TP when TP is a verified bracket leg.
  • 4-tier hierarchy recognition (Broker bracket, Order type, Disk metadata, In-memory registry).
  • Rogue non-bracket order cancellation without touching verified bracket orders.
"""

from unittest.mock import MagicMock, patch
from alpaca.trading.enums import QueryOrderStatus
import pytest

from alpaca_bridge import AlpacaBridge


def _make_bridge_for_reconciliation():
    bridge = AlpacaBridge(mode="swing", dry_run=False)
    bridge._client = MagicMock()
    bridge._connected = True
    bridge.dry_run = False
    bridge.entry_gate = MagicMock()
    return bridge


def test_held_bracket_orders_visible_in_active_orders():
    """Amendment 4: _get_active_orders() queries both OPEN and ALL to capture 'held' conditional stop legs."""
    bridge = _make_bridge_for_reconciliation()

    # Open query returns parent limit order (e.g. TP)
    tp_order = MagicMock()
    tp_order.id = "tp_1"
    tp_order.status.value = "accepted"
    tp_order.symbol = "JPM"
    tp_order.side.value = "sell"
    tp_order.legs = []

    # All query returns the conditional stop loss leg in 'held' status
    sl_leg = MagicMock()
    sl_leg.id = "sl_held_1"
    sl_leg.status.value = "held"
    sl_leg.symbol = "JPM"
    sl_leg.side.value = "sell"
    sl_leg.legs = []

    def _get_orders_mock(filter=None):
        if filter and filter.status == QueryOrderStatus.OPEN:
            return [tp_order]
        elif filter and filter.status == QueryOrderStatus.ALL:
            return [sl_leg]
        return []

    bridge._client.get_orders = MagicMock(side_effect=_get_orders_mock)

    active_orders = bridge._get_active_orders()
    active_ids = {str(o.id) for o in active_orders}

    assert "tp_1" in active_ids
    assert "sl_held_1" in active_ids
    assert len(active_orders) == 2


def test_bracket_tp_and_sl_both_preserved():
    """Amendment 4: When sl_coverage >= abs_qty, both SL and TP orders are preserved; zero cancellations."""
    bridge = _make_bridge_for_reconciliation()

    mock_position = MagicMock()
    mock_position.symbol = "JPM"
    mock_position.qty = "10.0"
    mock_position.avg_entry_price = "200.0"
    mock_position.current_price = "205.0"
    bridge._client.get_all_positions = MagicMock(return_value=[mock_position])

    # Working orders contains both SL (held or open) and TP limit
    bridge.working_orders_plain = MagicMock(return_value=[
        {
            "id": "sl_ord_1",
            "symbol": "JPM",
            "action": "SELL",
            "order_type": "STOP",
            "qty": 10.0,
            "stop_price": 190.0,
            "tif": "GTC",
            "status": "HELD",
        },
        {
            "id": "tp_ord_1",
            "symbol": "JPM",
            "action": "SELL",
            "order_type": "LIMIT",
            "qty": 10.0,
            "limit_price": 220.0,
            "tif": "GTC",
            "status": "OPEN",
            "order_class": "bracket",
        },
    ])

    report = bridge.ensure_protective_stops()

    # Zero cancellations on client
    bridge._client.cancel_order_by_id.assert_not_called()
    bridge._client.submit_order.assert_not_called()
    assert "JPM" in report["protected"]
    assert "JPM" not in report["unprotected"]
    assert report["clean"] is True


def test_repair_missing_sl_preserves_verified_tp():
    """Amendment 4: When SL is missing but TP is verified bracket leg, never cancel TP."""
    bridge = _make_bridge_for_reconciliation()

    mock_position = MagicMock()
    mock_position.symbol = "NVDA"
    mock_position.qty = "5.0"
    mock_position.avg_entry_price = "120.0"
    mock_position.current_price = "125.0"
    bridge._client.get_all_positions = MagicMock(return_value=[mock_position])

    # TP order exists (registered as bracket leg in role registry)
    bridge.working_orders_plain = MagicMock(return_value=[
        {
            "id": "tp_nvda_1",
            "symbol": "NVDA",
            "action": "SELL",
            "order_type": "LIMIT",
            "qty": 5.0,
            "limit_price": 140.0,
            "tif": "GTC",
            "status": "OPEN",
            "order_class": "bracket",
        }
    ])
    # Role registry knows this is a TAKE_PROFIT leg linked to bracket parent
    bridge._order_role_registry = {
        "tp_nvda_1": {
            "role": "TAKE_PROFIT",
            "parent_order_id": "parent_entry_123",
            "stop_order_id": "sl_nvda_1",
        }
    }

    # Broker accepts newly submitted stop order
    mock_new_stop = MagicMock()
    mock_new_stop.id = "repaired_stop_nvda"
    bridge._client.submit_order = MagicMock(return_value=mock_new_stop)

    report = bridge.ensure_protective_stops()

    # Critical invariant: TP order must NEVER be cancelled
    bridge._client.cancel_order_by_id.assert_not_called()
    # GTC stop order was submitted to repair coverage
    bridge._client.submit_order.assert_called_once()
    assert "NVDA" in report["repaired"]
    assert "NVDA" in report["protected"]


def test_rogue_non_bracket_order_cleaned_without_touching_bracket_orders():
    """Amendment 4: Truly rogue non-bracket sell order is cleaned, while bracket orders are untouched."""
    bridge = _make_bridge_for_reconciliation()

    mock_position = MagicMock()
    mock_position.symbol = "AAPL"
    mock_position.qty = "10.0"
    mock_position.avg_entry_price = "220.0"
    mock_position.current_price = "225.0"
    bridge._client.get_all_positions = MagicMock(return_value=[mock_position])

    bridge.working_orders_plain = MagicMock(return_value=[
        # Verified TP order
        {
            "id": "tp_aapl_1",
            "symbol": "AAPL",
            "action": "SELL",
            "order_type": "LIMIT",
            "qty": 10.0,
            "limit_price": 240.0,
            "tif": "GTC",
            "status": "OPEN",
            "order_class": "bracket",
        },
        # Rogue unprotective order holding shares (e.g. DAY order with unknown origin)
        {
            "id": "rogue_ord_999",
            "symbol": "AAPL",
            "action": "SELL",
            "order_type": "MARKET",
            "qty": 5.0,
            "tif": "DAY",
            "status": "OPEN",
            "order_class": "",
        },
    ])
    bridge._order_role_registry = {
        "tp_aapl_1": {"role": "TAKE_PROFIT", "parent_order_id": "p1"},
    }

    bridge._client.cancel_order_by_id = MagicMock()
    bridge._client.submit_order = MagicMock()

    bridge.ensure_protective_stops()

    # Rogue order cancelled
    bridge._client.cancel_order_by_id.assert_called_once_with("rogue_ord_999")
    # Verified TP order tp_aapl_1 was NEVER cancelled
    cancelled_ids = [call[0][0] for call in bridge._client.cancel_order_by_id.call_args_list]
    assert "tp_aapl_1" not in cancelled_ids
