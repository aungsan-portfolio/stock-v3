import sys
import os
sys.path.insert(0, os.path.abspath("."))
from strategies.base import TradeSignal
from strategies.order_manager import _validate_signal_prices

def test_pre_submission_veto():
    # 1. Too-tight signal (Entry 100.00, Stop 99.90 -> dist 0.10 < min required 0.20)
    tight_signal = TradeSignal(
        symbol="SPY",
        side="BUY",
        strategy="VWAP_BOUNCE",
        entry_price=100.00,
        stop_price=99.90,
        target_price=100.50,
        risk_per_share=0.10,
        atr=0.5,
        confidence=0.70,
        reason="Test tight signal"
    )

    reason_1 = _validate_signal_prices(tight_signal)
    assert "Pre-submission Veto" in reason_1, f"Expected Pre-submission Veto, got {reason_1}"
    print(f"[PASS Phase 3 Part 1] Too-Tight Signal Vetoed: {reason_1}")

    # 2. Valid signal (Entry 100.00, Stop 99.70 -> dist 0.30 >= min required 0.20)
    valid_signal = TradeSignal(
        symbol="SPY",
        side="BUY",
        strategy="VWAP_BOUNCE",
        entry_price=100.00,
        stop_price=99.70,
        target_price=100.60,
        risk_per_share=0.30,
        atr=0.5,
        confidence=0.70,
        reason="Test valid signal"
    )

    reason_2 = _validate_signal_prices(valid_signal)
    assert reason_2 == "", f"Expected empty reason for valid signal, got {reason_2}"
    print(f"[PASS Phase 3 Part 2] Valid Signal Passed Geometry Check")

if __name__ == "__main__":
    test_pre_submission_veto()
