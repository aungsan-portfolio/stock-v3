import pytest
from unittest.mock import patch, MagicMock
import numpy as np

import config
from strategies.constants import StrategyName
from strategies.base import TradeSignal
from strategies.position_sizer import calculate_shares

def create_dummy_signal(symbol="AAPL", atr=1.0, risk=1.0, entry=100.0):
    return TradeSignal(
        symbol=symbol,
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.8,
        entry_price=entry,
        stop_price=entry - risk,
        target_price=entry + 2.0,
        atr=atr,
        risk_per_share=risk,
        reason="Test",
    )

def test_fixed_shares():
    signal = create_dummy_signal()
    with patch("config.SIZING_METHOD", "fixed_shares"), \
         patch("config.FIXED_SHARE_COUNT", 75), \
         patch("config.MAX_POSITION_SIZE_DOLLARS", 10000.0), \
         patch("config.MAX_POSITION_PCT_OF_EQUITY", 0.5):
        shares = calculate_shares(signal, equity=50000.0)
        assert shares == 75

def test_fixed_dollars():
    signal = create_dummy_signal(entry=50.0)
    with patch("config.SIZING_METHOD", "fixed_dollars"), \
         patch("config.FIXED_DOLLAR_AMOUNT", 2000.0), \
         patch("config.MAX_POSITION_SIZE_DOLLARS", 10000.0), \
         patch("config.MAX_POSITION_PCT_OF_EQUITY", 0.5):
        shares = calculate_shares(signal, equity=50000.0)
        # 2000 / 50 = 40 shares
        assert shares == 40

def test_risk_based():
    signal = create_dummy_signal(risk=2.0)
    with patch("config.SIZING_METHOD", "risk_based"), \
         patch("config.STRATEGY_SETTINGS", {StrategyName.MOMENTUM_SCALP: MagicMock(max_risk_pct=1.0)}, create=True), \
         patch("config.MAX_POSITION_SIZE_DOLLARS", 10000.0), \
         patch("config.MAX_POSITION_PCT_OF_EQUITY", 0.5):
        shares = calculate_shares(signal, equity=10000.0)
        # Risk Budget = 10000 * 1% = 100. Risk per share = 2.0. Shares = 100 / 2.0 = 50.
        assert shares == 50

def test_atr_adjusted_risk_valid():
    signal = create_dummy_signal(atr=2.0)
    with patch("config.SIZING_METHOD", "atr_adjusted_risk"), \
         patch("config.ATR_SIZING_MULTIPLIER", 1.5), \
         patch("config.MIN_ATR_THRESHOLD", 0.01), \
         patch("config.STRATEGY_SETTINGS", {StrategyName.MOMENTUM_SCALP: MagicMock(max_risk_pct=1.0)}, create=True), \
         patch("config.MAX_POSITION_SIZE_DOLLARS", 10000.0), \
         patch("config.MAX_POSITION_PCT_OF_EQUITY", 0.5):
        shares = calculate_shares(signal, equity=15000.0)
        # Risk Budget = 15000 * 1% = 150
        # Divisor = 2.0 (atr) * 1.5 (mult) = 3.0
        # Shares = 150 / 3.0 = 50
        assert shares == 50

def test_atr_adjusted_risk_min_threshold_breach():
    # ATR = 0.005, which is < MIN_ATR_THRESHOLD (0.01)
    signal = create_dummy_signal(atr=0.005)
    with patch("config.SIZING_METHOD", "atr_adjusted_risk"), \
         patch("config.ATR_SIZING_MULTIPLIER", 1.5), \
         patch("config.MIN_ATR_THRESHOLD", 0.01), \
         patch("config.STRATEGY_SETTINGS", {StrategyName.MOMENTUM_SCALP: MagicMock(max_risk_pct=1.0)}, create=True):
        shares = calculate_shares(signal, equity=10000.0)
        # Should breach threshold and return 0
        assert shares == 0

def test_atr_adjusted_risk_invalid_atr():
    # NaN ATR
    signal_nan = create_dummy_signal(atr=np.nan)
    # Zero ATR
    signal_zero = create_dummy_signal(atr=0.0)
    
    with patch("config.SIZING_METHOD", "atr_adjusted_risk"), \
         patch("config.MIN_ATR_THRESHOLD", 0.01), \
         patch("config.STRATEGY_SETTINGS", {StrategyName.MOMENTUM_SCALP: MagicMock(max_risk_pct=1.0)}, create=True):
        
        assert calculate_shares(signal_nan, equity=10000.0) == 0
        assert calculate_shares(signal_zero, equity=10000.0) == 0

def test_atr_adjusted_risk_extreme_high_atr():
    # Extreme high ATR representing glitch or flash crash
    signal = create_dummy_signal(atr=1000.0)
    with patch("config.SIZING_METHOD", "atr_adjusted_risk"), \
         patch("config.ATR_SIZING_MULTIPLIER", 2.0), \
         patch("config.MIN_ATR_THRESHOLD", 0.01), \
         patch("config.STRATEGY_SETTINGS", {StrategyName.MOMENTUM_SCALP: MagicMock(max_risk_pct=1.0)}, create=True), \
         patch("config.MAX_POSITION_SIZE_DOLLARS", 10000.0), \
         patch("config.MAX_POSITION_PCT_OF_EQUITY", 0.5):
        shares = calculate_shares(signal, equity=10000.0)
        # Risk budget = 100. Divisor = 1000 * 2 = 2000. Shares = 100 / 2000 = 0.05. Floor to int -> 0.
        assert shares == 0

def test_atr_adjusted_risk_capped_by_limits():
    signal = create_dummy_signal(atr=0.1, entry=100.0)
    with patch("config.SIZING_METHOD", "atr_adjusted_risk"), \
         patch("config.ATR_SIZING_MULTIPLIER", 1.0), \
         patch("config.MIN_ATR_THRESHOLD", 0.01), \
         patch("config.STRATEGY_SETTINGS", {StrategyName.MOMENTUM_SCALP: MagicMock(max_risk_pct=5.0)}, create=True), \
         patch("config.MAX_POSITION_SIZE_DOLLARS", 2000.0), \
         patch("config.MAX_POSITION_PCT_OF_EQUITY", 0.10):
        
        shares = calculate_shares(signal, equity=50000.0)
        # Risk budget = 50000 * 5% = 2500
        # Divisor = 0.1 * 1.0 = 0.1
        # Raw shares = 2500 / 0.1 = 25000 shares
        # Dollar cap = 2000 / 100 = 20 shares
        # Equity cap = (50000 * 10%) / 100 = 50 shares
        # Capped shares = min(25000, 20, 50) = 20
        assert shares == 20

if __name__ == "__main__":
    pytest.main(["-v", __file__])
