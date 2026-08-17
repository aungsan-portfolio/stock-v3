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

    # 2. Valid signal (Entry 100.00, Stop 99.40 -> dist 0.60 >= min required hard veto line 0.375)
    valid_signal = TradeSignal(
        symbol="SPY",
        side="BUY",
        strategy="VWAP_BOUNCE",
        entry_price=100.00,
        stop_price=99.40,
        target_price=101.20,
        risk_per_share=0.60,
        atr=0.5,
        confidence=0.70,
        reason="Test valid signal"
    )

    reason_2 = _validate_signal_prices(valid_signal)
    assert reason_2 == "", f"Expected empty reason for valid signal, got {reason_2}"
    print(f"[PASS Phase 3 Part 2] Valid Signal Passed Geometry Check")


def test_execute_signal_no_unbound_limit_price():
    """Verify execute_signal does not crash with UnboundLocalError for limit_price."""
    from strategies.order_manager import execute_signal

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self.ib = type("MockIB", (), {"positions": lambda: []})()
        def open_position_count(self): return 0
        def get_positions(self): return []
        def has_position(self, sym): return False
        def has_working_order(self, sym): return False
        def place_bracket_order(self, symbol, side, qty, entry_price, stop_price, target_price):
            return {"order_id": "test_123", "status": "Submitted"}

    signal = TradeSignal(
        symbol="TSLA",
        side="BUY",
        strategy="MEAN_REVERSION",
        entry_price=347.93,
        stop_price=345.50,
        target_price=352.00,
        risk_per_share=2.43,
        atr=1.5,
        confidence=0.85,
        reason="Test mean reversion signal"
    )

    b = MockBridge()
    res = execute_signal(signal=signal, bridge=b, equity=10000.0, current_pnl=0.0, dry_run=True)
    assert isinstance(res, dict)
    assert res.get("reason") != "cannot access local variable 'limit_price' where it is not associated with a value"


if __name__ == "__main__":
    test_pre_submission_veto()
    test_execute_signal_no_unbound_limit_price()

