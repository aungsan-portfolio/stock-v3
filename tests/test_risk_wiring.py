import sys
import os
sys.path.insert(0, os.path.abspath("."))
import logging
from unittest.mock import MagicMock
from strategies.base import TradeSignal
from strategies.order_manager import execute_signal

def test_cluster_cap_and_fail_loud():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Signal for 3rd tech stock (AAPL)
    signal = TradeSignal(
        symbol="AAPL",
        side="BUY",
        strategy="VWAP_BOUNCE",
        entry_price=324.26,
        stop_price=321.79,
        target_price=329.05,
        risk_per_share=2.47,
        atr=1.5,
        confidence=0.80,
        reason="VWAP bounce test"
    )

    # 1. Test Cluster Cap Block: 2 active tech positions (GOOGL, MSFT)
    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    
    mock_pos_1 = MagicMock()
    mock_pos_1.symbol = "GOOGL"
    mock_pos_2 = MagicMock()
    mock_pos_2.symbol = "MSFT"
    
    mock_bridge.get_positions.return_value = [mock_pos_1, mock_pos_2]
    mock_bridge.open_position_count.return_value = 2

    res_1 = execute_signal(
        signal=signal,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        dry_run=False
    )

    assert res_1["status"] == "REJECTED"
    assert "Correlated Tech Cluster Cap reached (2/2): Blocked AAPL" in res_1["reason"]
    print(f"[PASS F2 Part 1] Cluster Cap Block Verified: {res_1['reason']}")

    # 2. Test Fail-Loud Error Guard: Bridge position method raises exception
    broken_bridge = MagicMock()
    broken_bridge.is_connected = True
    broken_bridge.get_positions.side_effect = Exception("Bridge connection dropped during position query")
    broken_bridge.get_open_positions.side_effect = Exception("Bridge connection dropped during position query")
    broken_bridge.open_position_count.return_value = 0

    res_2 = execute_signal(
        signal=signal,
        bridge=broken_bridge,
        equity=100000.0,
        current_pnl=0.0,
        dry_run=False
    )

    assert res_2["status"] == "REJECTED"
    assert "Risk guard failure" in res_2["reason"]
    print(f"[PASS F2 Part 2] Fail-Loud Guard Verified: {res_2['reason']}")

if __name__ == "__main__":
    test_cluster_cap_and_fail_loud()
