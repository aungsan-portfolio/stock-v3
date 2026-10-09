"""
tests/test_replacement_lineage.py
Unit and integration tests for stop replacement lineage preservation,
canonical mutation accounting, and validation ledger isolation (Breach 5 resolution).
"""
import pytest
from unittest.mock import MagicMock, patch
from alpaca_bridge import AlpacaBridge, OrderTruth
from strategies.order_registry import get_registered_order, _ORDER_REGISTRY
from strategies.session import now_eastern


@pytest.fixture
def mock_bridge(tmp_path):
    bridge = AlpacaBridge(mode="daytrade")
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._client._paper = True
    bridge._client._base_url = "https://paper-api.alpaca.markets"
    bridge.assert_account_mode_safety = MagicMock(return_value=True)
    # Direct order role registry to isolated temp file
    temp_role_path = tmp_path / "test_order_roles.json"
    bridge._get_order_role_registry_path = MagicMock(return_value=temp_role_path)
    bridge._order_role_registry = {}
    return bridge


def test_modify_stop_order_dry_run(mock_bridge):
    """Dry-run mode must record mutation counter and block live broker transmission."""
    mock_bridge.dry_run = True
    res = mock_bridge.modify_stop_order("ord-123", 150.0, symbol="AAPL")
    assert res == "dry_ord-123"
    assert mock_bridge.mutation_counters.get("modify_stop_order") == 0
    assert mock_bridge.mutation_counters.mutation_blocked == 1
    mock_bridge._client.replace_order_by_id.assert_not_called()


def test_modify_stop_order_role_lineage_propagation(mock_bridge):
    """Predecessor role, parent linkage, and validation flags must propagate to successor."""
    pred_id = "pred-order-001"
    succ_id = "succ-order-002"
    parent_id = "parent-order-000"

    # Setup predecessor role registry
    mock_bridge._order_role_registry[parent_id] = {
        "role": "ENTRY",
        "parent_order_id": parent_id,
        "client_order_id": "val_jpm_56560",
        "stop_order_id": pred_id,
        "strategy": "MOMENTUM_SCALP",
    }
    mock_bridge._order_role_registry[pred_id] = {
        "role": "STOP_LOSS",
        "parent_order_id": parent_id,
        "client_order_id": "val_jpm_56560",
        "stop_order_id": pred_id,
        "strategy": "MOMENTUM_SCALP",
    }

    # Mock Alpaca replace response
    mock_succ_order = MagicMock()
    mock_succ_order.id = succ_id
    mock_succ_order.symbol = "JPM"
    mock_bridge._client.replace_order_by_id.return_value = mock_succ_order

    # Execute modification
    result_id = mock_bridge.modify_stop_order(pred_id, 148.50, symbol="JPM")
    assert result_id == succ_id

    # Verify mutation counter recorded
    assert mock_bridge.mutation_counters.get("modify_stop_order") == 1
    assert mock_bridge.mutation_counters.replace_order_passed_through == 1

    # Verify successor inherited role and predecessor pointer
    succ_role = mock_bridge.get_order_role_info(succ_id)
    assert succ_role["role"] == "STOP_LOSS"
    assert succ_role["parent_order_id"] == parent_id
    assert succ_role["client_order_id"] == "val_jpm_56560"
    assert succ_role["predecessor_order_id"] == pred_id
    assert succ_role["strategy"] == "MOMENTUM_SCALP"
    assert succ_role["replacement_status"] == "REPLACED"

    # Verify parent updated to point to new stop leg
    parent_role = mock_bridge.get_order_role_info(parent_id)
    assert parent_role["stop_order_id"] == succ_id

    # Verify predecessor marked as superseded
    pred_role = mock_bridge.get_order_role_info(pred_id)
    assert pred_role["replaced_by_order_id"] == succ_id
    assert pred_role["replacement_status"] == "SUPERSEDED"

    # Verify order_registry registration
    reg = get_registered_order(succ_id)
    assert reg is not None
    assert reg["is_validation"] is True
    assert reg["expected_price"] == 148.50
    assert reg["client_order_id"] == "val_jpm_56560"


def test_modify_stop_order_price_alias(mock_bridge):
    """modify_stop_order_price must exist as a canonical alias."""
    assert hasattr(mock_bridge, "modify_stop_order_price")
    assert mock_bridge.modify_stop_order_price == mock_bridge.modify_stop_order


def test_unknown_replacement_outcome_reconciliation(mock_bridge):
    """Network failure during replace is reconciled against broker truth."""
    pred_id = "pred-network-err"
    succ_id = "succ-remote-reconciled"

    mock_bridge._client.replace_order_by_id.side_effect = TimeoutError("Alpaca read timeout")

    # Mock broker truth revealing that Alpaca actually replaced it
    raw_pred = MagicMock()
    raw_pred.replaced_by = succ_id
    mock_truth = OrderTruth(
        id=pred_id,
        symbol="TSLA",
        side="SELL",
        status="replaced",
        qty=10.0,
        filled_qty=0.0,
        filled_avg_price=0.0,
        is_terminal=True,
        raw=raw_pred,
    )
    mock_bridge.get_order_truth = MagicMock(return_value=mock_truth)

    mock_succ_order = MagicMock()
    mock_succ_order.id = succ_id
    mock_succ_order.symbol = "TSLA"
    mock_bridge._client.get_order_by_id.return_value = mock_succ_order

    res = mock_bridge.modify_stop_order(pred_id, 245.0, symbol="TSLA")
    assert res == succ_id
    assert mock_bridge.mutation_counters.get("modify_stop_order") == 1


def test_terminal_order_race_handling(mock_bridge):
    """HTTP 422 or already terminal orders fail gracefully with None."""
    pred_id = "pred-already-filled"
    mock_bridge._client.replace_order_by_id.side_effect = Exception("HTTP 422: order is not cancelable/already filled")

    res = mock_bridge.modify_stop_order(pred_id, 245.0, symbol="TSLA")
    assert res is None
    # Must not register successor or increment mutation counter
    assert mock_bridge.mutation_counters.get("modify_stop_order") == 0


def test_validation_fill_isolation_with_replacement_chain(mock_bridge, tmp_path):
    """Validation fills on successor stop orders must route to validation ledger and never pollute trade_journal."""
    succ_id = "succ-val-fill-001"
    pred_id = "pred-val-fill-000"

    mock_bridge._order_role_registry[succ_id] = {
        "role": "STOP_LOSS",
        "parent_order_id": "parent-001",
        "client_order_id": "val_uber_57772",
        "predecessor_order_id": pred_id,
        "strategy": "CANDLESTICK",
    }

    mock_order = MagicMock()
    mock_order.id = succ_id
    mock_order.symbol = "UBER"
    mock_order.side = "sell"
    mock_order.status = "filled"
    mock_order.qty = "5"
    mock_order.filled_qty = "5"
    mock_order.filled_avg_price = "75.25"
    mock_order.client_order_id = ""  # Broker erased client_order_id on PATCH
    mock_order.replaces = pred_id
    mock_order.legs = []
    mock_order.filled_at = now_eastern()

    mock_bridge._client.get_orders.return_value = [mock_order]

    with patch("strategies.validation_lane.record_validation_exit", return_value=True) as mock_val_exit, \
         patch("strategies.trade_journal.log_fill") as mock_prod_log_fill:

        synced = mock_bridge.sync_today_trades_to_journal(orders=[mock_order])

        assert synced == 1
        # Validation exit must be recorded
        assert mock_val_exit.called
        assert mock_val_exit.call_args.kwargs["order_id"] == succ_id
        assert mock_val_exit.call_args.kwargs["client_order_id"] == "val_uber_57772"

        # Production trade_journal MUST NEVER be called
        mock_prod_log_fill.assert_not_called()
