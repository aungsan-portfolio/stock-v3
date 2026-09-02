from unittest.mock import MagicMock, patch
import pytest
from alpaca_bridge import AlpacaBridge
from predictor import Signal
import order_exec


def test_execute_signal_soxx_gate_veto_blocks_tech_long():
    bridge = AlpacaBridge(mode="swing")
    bridge._connected = True
    bridge._client = MagicMock()
    
    # Mock quote decision
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=150.0))
    bridge.get_position = MagicMock(return_value=0)
    bridge.has_working_order = MagicMock(return_value=False)
    bridge.get_net_liquidation = MagicMock(return_value=100000.0)

    # Patch check_soxx_tech_long_gate to return FAIL
    with patch("strategies.intraday_risk.check_soxx_tech_long_gate", return_value=(False, "SOXX is -1.90% below threshold")):
        sig = Signal(
            symbol="AMD", action="BUY", confidence=0.56,
            rf_score=0.55, lstm_score=0.57, xgb_score=0.61, trans_score=0.61, tech_score=0.55,
            price=150.0, reason="ML Buy"
        )
        result = bridge.execute_signal(sig)
        
        assert result is False
        # Ensure bracket order submission was NOT attempted
        bridge._client.submit_order.assert_not_called()


def test_await_order_outcome_reconciles_and_cancels_unfilled_order():
    bridge = AlpacaBridge(mode="swing")
    bridge._client = MagicMock()

    mock_working_order = MagicMock()
    mock_working_order.status = "new"
    mock_working_order.filled_qty = 0
    mock_working_order.qty = 4
    mock_working_order.limit_price = 150.0

    mock_cancelled_order = MagicMock()
    mock_cancelled_order.status = "canceled"
    mock_cancelled_order.filled_qty = 0
    mock_cancelled_order.qty = 4

    # First returns working order (during timeout polling), then returns cancelled after cancel call
    bridge._client.get_order_by_id.side_effect = [
        mock_working_order, # initial poll
        mock_working_order, # post-timeout query
        mock_cancelled_order # cancel confirmation
    ]

    res = bridge._await_order_outcome(order_id="test-order-123", timeout=0.1)

    assert res.outcome == order_exec.CANCELLED
    assert res.filled == 0.0
    bridge._client.cancel_order_by_id.assert_called_once_with("test-order-123")


def test_await_order_outcome_handles_immediate_fill():
    bridge = AlpacaBridge(mode="swing")
    bridge._client = MagicMock()

    mock_filled_order = MagicMock()
    mock_filled_order.status = "filled"
    mock_filled_order.filled_qty = 4
    mock_filled_order.qty = 4
    mock_filled_order.filled_avg_price = 149.95

    bridge._client.get_order_by_id.return_value = mock_filled_order

    res = bridge._await_order_outcome(order_id="test-fill-123", timeout=1.0)

    assert res.outcome == order_exec.FILLED
    assert res.filled == 4.0
    assert res.avg_fill_price == 149.95
    bridge._client.cancel_order_by_id.assert_not_called()
