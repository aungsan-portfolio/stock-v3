"""
tests/test_alpaca_bridge_short_execution.py -- Forensic regression tests for AlpacaBridge short execution path.

Validates that:
1. Flat position cleanly evaluates normal short SELL path without NameError.
2. Bare 'config' reference is resolved to 'config_module'.
3. Inverse ETFs cannot be sold short (hedging invariant).
4. Mode == 'swing' or ALLOW_SHORT=False cleanly rejects short entry.
5. Missing INVERSE_ETF_SYMBOLS attribute safely falls back to empty set.
6. Rejected short entries never invoke broker order placement.
"""
import pytest
from unittest import mock

from alpaca_bridge import AlpacaBridge
import config as config_module
from data_integrity import QuoteDecision
import order_exec
from predictor import Signal


def _make_dummy_signal(symbol: str, action: str = "SELL", price: float = 100.0,
                       stop_price: float = 105.0, target_price: float = 90.0) -> Signal:
    sig = Signal(
        symbol=symbol,
        action=action,
        confidence=0.85,
        rf_score=0.85,
        lstm_score=0.85,
        xgb_score=0.85,
        trans_score=0.85,
        tech_score=0.85,
        price=price,
        reason="Test momentum short",
    )
    sig.stop_price = stop_price
    sig.target_price = target_price
    sig.strategy = "TEST_MOMENTUM_SHORT"
    return sig


@pytest.fixture
def daytrade_bridge():
    bridge = AlpacaBridge(mode="daytrade")
    bridge._connected = True
    bridge.entry_gate.halted = False
    bridge.entry_gate.reasons = set()
    bridge.get_net_liquidation = mock.MagicMock(return_value=50000.0)
    bridge.get_cash = mock.MagicMock(return_value=50000.0)
    bridge.get_position = mock.MagicMock(return_value=0.0)
    bridge.has_working_order = mock.MagicMock(return_value=False)
    bridge.get_order_quote = mock.MagicMock(return_value=QuoteDecision(ok=True, price=100.0, source="mid", reason="OK"))
    return bridge


class TestAlpacaBridgeShortExecution:
    def test_short_sell_normal_reaches_execution_without_name_error(self, daytrade_bridge):
        """Verify normal short SELL executes through config_module without NameError."""
        signal = _make_dummy_signal("AAPL", action="SELL", price=100.0, stop_price=105.0, target_price=90.0)

        fake_exec_result = order_exec.OrderResult(
            outcome=order_exec.FILLED,
            filled=10.0,
            avg_fill_price=100.0,
        )

        with mock.patch.object(daytrade_bridge, "_place_open_bracket", return_value=fake_exec_result) as mock_bracket:
            with mock.patch.object(config_module, "ALLOW_SHORT", True):
                result = daytrade_bridge.execute_signal(signal)

        assert result is True
        mock_bracket.assert_called_once()
        call_args = mock_bracket.call_args
        assert call_args[0][0] == "AAPL"
        assert call_args[0][1] == "SELL"
        assert call_args[0][2] > 0  # qty
        assert call_args[0][3] == 100.0  # price

    def test_inverse_etf_short_is_rejected(self, daytrade_bridge):
        """Verify inverse ETFs cannot be shorted and no order is placed."""
        signal = _make_dummy_signal("SQQQ", action="SELL", price=50.0, stop_price=53.0, target_price=45.0)
        daytrade_bridge.get_order_quote = mock.MagicMock(return_value=QuoteDecision(ok=True, price=50.0, source="mid", reason="OK"))

        with mock.patch.object(daytrade_bridge, "_place_open_bracket") as mock_bracket:
            with mock.patch.object(config_module, "ALLOW_SHORT", True):
                with mock.patch.object(config_module, "INVERSE_ETF_SYMBOLS", {"SQQQ", "SH", "SOXS"}):
                    result = daytrade_bridge.execute_signal(signal)

        assert result is False
        mock_bracket.assert_not_called()

    def test_allow_short_false_rejects_entry(self, daytrade_bridge):
        """Verify ALLOW_SHORT=False skips SELL when flat."""
        signal = _make_dummy_signal("MSFT", action="SELL", price=200.0, stop_price=210.0, target_price=180.0)
        daytrade_bridge.get_order_quote = mock.MagicMock(return_value=QuoteDecision(ok=True, price=200.0, source="mid", reason="OK"))

        with mock.patch.object(daytrade_bridge, "_place_open_bracket") as mock_bracket:
            with mock.patch.object(daytrade_bridge._cfg, "ALLOW_SHORT", False):
                with mock.patch.object(config_module, "ALLOW_SHORT", False):
                    result = daytrade_bridge.execute_signal(signal)

        assert result is False
        mock_bracket.assert_not_called()

    def test_swing_mode_rejects_short_entry(self):
        """Verify swing mode is strictly long-only and rejects flat SELL."""
        swing_bridge = AlpacaBridge(mode="swing")
        swing_bridge._connected = True
        swing_bridge.get_position = mock.MagicMock(return_value=0.0)
        swing_bridge.get_order_quote = mock.MagicMock(return_value=QuoteDecision(ok=True, price=120.0, source="mid", reason="OK"))
        signal = _make_dummy_signal("NVDA", action="SELL", price=120.0, stop_price=126.0, target_price=110.0)

        with mock.patch.object(swing_bridge, "_place_open_bracket") as mock_bracket:
            with mock.patch.object(config_module, "ALLOW_SHORT", True):
                result = swing_bridge.execute_signal(signal)

        assert result is False
        mock_bracket.assert_not_called()

    def test_missing_inverse_etf_symbols_falls_back_to_empty_set(self, daytrade_bridge):
        """Verify that if config_module does not define INVERSE_ETF_SYMBOLS, it safely falls back to empty set."""
        signal = _make_dummy_signal("AMD", action="SELL", price=150.0, stop_price=157.0, target_price=135.0)
        daytrade_bridge.get_order_quote = mock.MagicMock(return_value=QuoteDecision(ok=True, price=150.0, source="mid", reason="OK"))

        fake_exec_result = order_exec.OrderResult(
            outcome=order_exec.FILLED,
            filled=5.0,
            avg_fill_price=150.0,
        )

        orig_val = getattr(config_module, "INVERSE_ETF_SYMBOLS", None)
        try:
            if hasattr(config_module, "INVERSE_ETF_SYMBOLS"):
                delattr(config_module, "INVERSE_ETF_SYMBOLS")

            with mock.patch.object(daytrade_bridge, "_place_open_bracket", return_value=fake_exec_result) as mock_bracket:
                with mock.patch.object(config_module, "ALLOW_SHORT", True):
                    result = daytrade_bridge.execute_signal(signal)

            assert result is True
            mock_bracket.assert_called_once()
        finally:
            if orig_val is not None:
                setattr(config_module, "INVERSE_ETF_SYMBOLS", orig_val)
