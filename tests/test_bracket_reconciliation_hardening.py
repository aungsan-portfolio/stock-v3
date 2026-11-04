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
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
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


@pytest.mark.parametrize("position_qty,wrong_side", [(-1, "SELL"), (1, "BUY")])
def test_wrong_side_stop_never_counts_as_protection_or_triggers_blind_repair(position_qty, wrong_side):
    bridge = _make_bridge_for_reconciliation()
    position = MagicMock(symbol="SPY", qty=str(position_qty), avg_entry_price="100", current_price="100")
    bridge._client.get_all_positions.return_value = [position]
    bridge.working_orders_plain = MagicMock(return_value=[{
        "id": "unexpected-stop", "symbol": "SPY", "action": wrong_side,
        "order_type": "STOP", "qty": 1.0, "filled_qty": 0.0,
        "status": "Submitted", "broker_status": "new", "stop_price": 101.0, "tif": "GTC",
    }])
    report = bridge.ensure_protective_stops()
    assert "SPY" not in report["protected"]
    assert "SPY" in report["failed"]
    assert report["clean"] is False
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


def test_partially_filled_stop_uses_remaining_quantity_and_fails_closed_on_shortfall():
    bridge = _make_bridge_for_reconciliation()
    position = MagicMock(symbol="SPY", qty="-5", avg_entry_price="100", current_price="100")
    bridge._client.get_all_positions.return_value = [position]
    bridge.working_orders_plain = MagicMock(return_value=[{
        "id": "partial-stop", "symbol": "SPY", "action": "BUY",
        "order_type": "STOP", "qty": 5.0, "filled_qty": 2.0,
        "status": "Submitted", "broker_status": "partially_filled", "stop_price": 101.0, "tif": "GTC",
    }])
    report = bridge.ensure_protective_stops()
    assert "SPY" not in report["protected"]
    assert "SPY" in report["failed"]
    assert report["clean"] is False
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


def test_working_order_snapshot_preserves_filled_quantity_and_raw_status():
    bridge = _make_bridge_for_reconciliation()
    bridge._get_active_orders = MagicMock(return_value=[SimpleNamespace(
        id="partial-stop", symbol="SPY", side="buy", status="partially_filled",
        type="stop", qty="5", filled_qty="2", stop_price="101", limit_price=None,
        time_in_force="gtc", client_order_id="partial-stop-client",
    )])
    orders = bridge.working_orders_plain()
    assert len(orders) == 1
    assert float(orders[0]["qty"]) == 5.0
    assert float(orders[0]["filled_qty"]) == 2.0
    assert orders[0]["broker_status"] == "partially_filled"
    assert orders[0]["broker_side"] == "buy"


@pytest.mark.parametrize("changes", [
    {"broker_side": "unknown", "action": "SELL"},
    {"filled_qty": None},
    {"broker_status": "held"},
    {"broker_status": "pending_new"},
    {"broker_status": "pending_cancel"},
    {"qty": 2.0},
    {"stop_price": None},
    {"stop_price": 0.0},
    {"order_type": "STOP_LIMIT", "limit_price": None},
    {"order_type": "STOP_LIMIT", "limit_price": 100.0},
])
def test_unverifiable_or_excess_stop_is_never_marked_protected(changes):
    bridge = _make_bridge_for_reconciliation()
    bridge._client.get_all_positions.return_value = [
        MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    ]
    stop = {"id": "s", "symbol": "SPY", "action": "SELL", "broker_side": "sell",
            "order_type": "STOP", "qty": 1.0, "filled_qty": 0.0,
            "status": "Submitted", "broker_status": "new", "stop_price": 99.0, "tif": "GTC"}
    stop.update(changes)
    bridge.working_orders_plain = MagicMock(return_value=[stop])
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "SPY" not in report["protected"]
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


def test_order_query_failure_never_triggers_blind_repair():
    bridge = _make_bridge_for_reconciliation()
    bridge._client.get_all_positions.return_value = [
        MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    ]
    bridge._get_active_orders = MagicMock(side_effect=TimeoutError("broker timeout"))
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "ORDER_FETCH_ERROR" in report["failed"]
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


def test_verified_empty_positions_need_no_order_repair():
    bridge = _make_bridge_for_reconciliation()
    bridge._client.get_all_positions.return_value = []
    bridge.working_orders_plain = MagicMock(side_effect=AssertionError("not needed for no positions"))
    report = bridge.ensure_protective_stops()
    assert report["clean"] is True and report["checked"] == 0
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


def test_missing_position_snapshot_is_unknown():
    bridge = _make_bridge_for_reconciliation()
    bridge._client.get_all_positions.return_value = None
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "PORTFOLIO_FETCH_ERROR" in report["failed"]
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()


def test_position_change_during_order_lookup_prevents_repair():
    bridge = _make_bridge_for_reconciliation()
    first = MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    changed = MagicMock(symbol="SPY", qty="2", avg_entry_price="100", current_price="100")
    bridge._client.get_all_positions.side_effect = [[first], [changed]]
    bridge.working_orders_plain = MagicMock(return_value=[])
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "POSITION_RECHECK_ERROR" in report["failed"]
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


@pytest.mark.parametrize("submission", [None, SimpleNamespace(id="new-stop")])
def test_unconfirmed_stop_submission_never_reports_protected(submission):
    bridge = _make_bridge_for_reconciliation()
    position = MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    bridge._client.get_all_positions.return_value = [position]
    bridge.working_orders_plain = MagicMock(return_value=[])
    bridge._initial_stop_price = MagicMock(return_value=99.0)
    bridge.market_price = MagicMock(return_value=100.0)
    bridge._client.submit_order.return_value = submission
    bridge.get_order_snapshot = MagicMock(return_value=None)
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "SPY" not in report["protected"]
    assert "SPY" in report["failed"]
    bridge.entry_gate.halt.assert_called()
    assert bridge._client.submit_order.call_count == 1
    bridge._client.cancel_order_by_id.assert_not_called()


def test_tp_metadata_after_failed_stop_submission_never_proves_protection():
    bridge = _make_bridge_for_reconciliation()
    bridge._client.get_all_positions.return_value = [
        MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    ]
    bridge.working_orders_plain = MagicMock(return_value=[{
        "id": "tp", "symbol": "SPY", "action": "SELL", "order_type": "LIMIT",
        "qty": 1, "status": "OPEN", "limit_price": 110,
    }])
    bridge.get_order_role_info = MagicMock(return_value={"role": "TAKE_PROFIT", "parent_order_id": "parent"})
    bridge._initial_stop_price = MagicMock(return_value=99.0)
    bridge.market_price = MagicMock(return_value=100.0)
    bridge._client.submit_order.side_effect = RuntimeError("insufficient qty")
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "SPY" not in report["protected"]
    assert "SPY" in report["failed"]
    bridge.entry_gate.halt.assert_called()
    bridge._client.cancel_order_by_id.assert_not_called()


@pytest.mark.parametrize("order_qty,filled_qty", [(2, 0), (1, 0.5)])
def test_repair_snapshot_with_extra_or_filled_quantity_is_not_verified(order_qty, filled_qty):
    bridge = _make_bridge_for_reconciliation()
    position = MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    bridge._client.get_all_positions.return_value = [position]
    bridge.working_orders_plain = MagicMock(return_value=[])
    bridge._initial_stop_price = MagicMock(return_value=99.0)
    bridge.market_price = MagicMock(return_value=100.0)
    bridge._client.submit_order.return_value = SimpleNamespace(id="new-stop")
    bridge.get_order_snapshot = MagicMock(return_value={
        "id": "new-stop", "symbol": "SPY", "side": "sell", "type": "stop",
        "status": "new", "qty": order_qty, "filled_qty": filled_qty, "stop_price": 99.0,
    })
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "SPY" in report["failed"]
    assert "SPY" not in report["protected"]
    bridge.entry_gate.halt.assert_called()
    assert bridge._client.submit_order.call_count == 1
    bridge._client.cancel_order_by_id.assert_not_called()


def test_position_change_after_stop_submission_is_reported_unknown_without_retry():
    bridge = _make_bridge_for_reconciliation()
    position = MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    bridge._client.get_all_positions.side_effect = [[position], [position], [position], []]
    bridge.working_orders_plain = MagicMock(return_value=[])
    bridge._initial_stop_price = MagicMock(return_value=99.0)
    bridge.market_price = MagicMock(return_value=100.0)
    bridge._client.submit_order.return_value = SimpleNamespace(id="new-stop")
    bridge.get_order_snapshot = MagicMock(return_value={
        "id": "new-stop", "symbol": "SPY", "side": "sell", "type": "stop",
        "status": "new", "qty": 1, "filled_qty": 0, "stop_price": 99.0,
    })
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "SPY" in report["failed"]
    assert "SPY" not in report["protected"]
    bridge.entry_gate.halt.assert_called()
    assert bridge._client.submit_order.call_count == 1
    bridge._client.cancel_order_by_id.assert_not_called()


def test_fractional_unprotected_position_is_not_rounded_up_to_one_share():
    bridge = _make_bridge_for_reconciliation()
    bridge._client.get_all_positions.return_value = [
        MagicMock(symbol="SPY", qty="0.25", avg_entry_price="100", current_price="100")
    ]
    bridge.working_orders_plain = MagicMock(return_value=[])
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "SPY" in report["failed"]
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


def test_real_normalization_does_not_turn_unknown_broker_side_into_protection():
    bridge = _make_bridge_for_reconciliation()
    bridge._client.get_all_positions.return_value = [
        MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    ]
    bridge._get_active_orders = MagicMock(return_value=[SimpleNamespace(
        id="s", symbol="SPY", side=None, status="new", type="stop", qty="1",
        filled_qty="0", stop_price="99", limit_price=None, time_in_force="gtc", client_order_id="s",
    )])
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "SPY" not in report["protected"]
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


@pytest.mark.parametrize("second_query", [TimeoutError("all orders timed out"), [MagicMock()] * 200])
def test_incomplete_conditional_order_snapshot_fails_closed(second_query):
    bridge = _make_bridge_for_reconciliation()
    bridge._client.get_all_positions.return_value = [
        MagicMock(symbol="SPY", qty="1", avg_entry_price="100", current_price="100")
    ]
    bridge._call_with_retry = lambda fn: fn()
    bridge._client.get_orders.side_effect = [[], second_query]
    report = bridge.ensure_protective_stops()
    assert report["clean"] is False and "ORDER_FETCH_ERROR" in report["failed"]
    bridge.entry_gate.halt.assert_called()
    bridge._client.submit_order.assert_not_called()
    bridge._client.cancel_order_by_id.assert_not_called()


def test_strict_order_history_pages_with_overlap_without_losing_older_active_order():
    bridge = _make_bridge_for_reconciliation()
    now = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    page = [SimpleNamespace(id=f"old-{i}", status="filled", symbol="SPY", legs=[],
                            submitted_at=now - timedelta(seconds=i)) for i in range(200)]
    older_active = SimpleNamespace(id="active-older", status="new", symbol="SPY", legs=[],
                                   submitted_at=now - timedelta(seconds=300))
    bridge._call_with_retry = lambda fn: fn()
    bridge._client.get_orders.side_effect = [[], page, [page[-1], older_active]]
    orders = bridge._get_active_orders(force_refresh=True, require_complete=True)
    assert {str(o.id) for o in orders} == {"active-older"}
    assert bridge._client.get_orders.call_count == 3
    assert bridge._client.get_orders.call_args.kwargs["filter"].until is not None


def test_strict_order_history_repeated_boundary_fails_closed():
    bridge = _make_bridge_for_reconciliation()
    same_time = datetime(2026, 11, 3, 15, 0, tzinfo=timezone.utc)
    page = [SimpleNamespace(id=f"o-{i}", status="filled", symbol="SPY", legs=[],
                            submitted_at=same_time) for i in range(200)]
    bridge._call_with_retry = lambda fn: fn()
    bridge._client.get_orders.side_effect = [[], page, page]
    with pytest.raises(RuntimeError):
        bridge._get_active_orders(force_refresh=True, require_complete=True)


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
            "status": "Submitted",
            "broker_status": "new",
            "filled_qty": 0.0,
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
    bridge._initial_stop_price = MagicMock(return_value=115.0)
    bridge.market_price = MagicMock(return_value=125.0)
    bridge.get_order_snapshot = MagicMock(return_value={
        "id": "repaired_stop_nvda", "symbol": "NVDA", "side": "sell", "type": "stop",
        "status": "new", "qty": 5.0, "filled_qty": 0.0, "stop_price": 115.0,
    })

    report = bridge.ensure_protective_stops()

    # Critical invariant: TP order must NEVER be cancelled
    bridge._client.cancel_order_by_id.assert_not_called()
    # GTC stop order was submitted to repair coverage
    bridge._client.submit_order.assert_called_once()
    assert "NVDA" in report["repaired"]
    assert "NVDA" in report["protected"]


def test_unknown_order_is_preserved_and_requires_reconciliation():
    """An unrecognized order is not blindly canceled or used to justify a new stop."""
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

    report = bridge.ensure_protective_stops()
    bridge._client.cancel_order_by_id.assert_not_called()
    bridge._client.submit_order.assert_not_called()
    bridge.entry_gate.halt.assert_called()
    assert "AAPL" in report["failed"] and report["clean"] is False
