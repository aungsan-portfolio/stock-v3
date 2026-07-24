import sys
import os
sys.path.insert(0, os.path.abspath("."))
from unittest.mock import MagicMock
from strategies.base import TradeSignal
from strategies.trailing_stop import DynamicTrailingStopManager

def test_rebuild_flow_and_broker_replace():
    mgr = DynamicTrailingStopManager()

    mock_bridge = MagicMock()
    mock_bridge.replace_order_by_id.return_value = MagicMock(id="new_broker_stop_999")

    # Inverted signal scenario: BUY GOOGL fill @ 98.00, but initial stop was 99.00 (above fill!)
    signal = TradeSignal(
        symbol="GOOGL",
        side="BUY",
        strategy="VWAP_BOUNCE",
        entry_price=100.00,
        stop_price=99.00,
        target_price=105.00,
        risk_per_share=1.00,
        atr=1.5,
        confidence=0.75,
        reason="Inverted test"
    )

    state = mgr.initialize_position(
        signal=signal,
        fill_price=98.00, # Inverted! Fill 98.00 < Stop 99.00
        qty=10,
        bridge=mock_bridge,
        stop_order_id="orig_stop_111",
        tp_order_id="orig_tp_222"
    )

    # Assert new stop price is valid (stop < fill)
    assert state.stop_price < 98.00
    assert (98.00 - state.stop_price) >= max(0.05, 98.00 * 0.002)

    # Assert broker replace_order_by_id was called for Stop Loss Order
    assert mock_bridge.replace_order_by_id.called
    print(f"[PASS F3 Part 1] REBUILD Broker Replace & Re-Validation Verified: new_stop=${state.stop_price:.2f}")

    # Assert Reconcile conflict warning path
    mock_order = MagicMock()
    mock_order.symbol = "GOOGL"
    mock_order.side.value = "sell"
    mock_order.stop_price = "97.50"
    mock_order.id = "new_broker_stop_999"

    recon_state = mgr.ensure_initialized(
        symbol="GOOGL",
        side="BUY",
        avg_cost=98.00,
        open_orders=[mock_order],
        current_price=98.50
    )

    assert recon_state.order_id == "new_broker_stop_999"
    print(f"[PASS F3 Part 2] Reconcile Order ID Adoption Verified: order_id={recon_state.order_id}")

if __name__ == "__main__":
    test_rebuild_flow_and_broker_replace()
