"""tests/test_mean_reversion.py

Unit tests for Multi-Layer Mean Reversion Strategy and Sideways Market Texture integration.
"""

import pytest
import pandas as pd
import numpy as np

from strategies.mean_reversion import MeanReversionStrategy, STRATEGY_NAME
from strategies.regime_engine import (
    detect_market_texture,
    TEXTURE_TRENDING,
    TEXTURE_SIDEWAYS,
    TEXTURE_VOLATILE,
    TEXTURE_TRANSITIONING,
)


@pytest.fixture
def sample_sideways_df():
    """Generates synthetic DataFrame with VWAP deviation, oversold RSI, low volume ratio, and price <= lower BB."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2026-08-13 09:30", periods=n, freq="5min")
    
    # Base price flat around 100.0, last bar dips to 95.0
    prices = np.full(n, 100.0)
    prices[-1] = 95.0
    
    highs = prices + 0.5
    lows = prices - 0.5
    highs[-1] = 95.5
    lows[-1] = 94.5
    
    volumes = np.full(n, 10000.0)
    volumes[-1] = 12000.0  # Vol ratio 1.2 <= 2.0
    
    df = pd.DataFrame({
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "vwap": np.full(n, 100.0)  # VWAP = 100.0, close = 95.0 (dev = 5.0 >= 1.2x ATR)
    }, index=dates)
    
    return df


def test_detect_market_texture_breakout_guard():
    """Test 1: 5m price change >= 1% forces TRENDING texture."""
    texture = detect_market_texture(adx=15.0, bb_width=0.01, atr_ratio=1.0, price_change_5m=0.015)
    assert texture == TEXTURE_TRENDING


def test_detect_market_texture_volatile():
    """Test 2: ATR ratio >= 1.5 forces VOLATILE texture."""
    texture = detect_market_texture(adx=15.0, bb_width=0.01, atr_ratio=1.6, price_change_5m=0.002)
    assert texture == TEXTURE_VOLATILE


def test_detect_market_texture_sideways():
    """Test 3: ADX < 20 AND BB Width <= 0.025 returns SIDEWAYS texture."""
    texture = detect_market_texture(adx=15.0, bb_width=0.02, atr_ratio=1.0, price_change_5m=0.002)
    assert texture == TEXTURE_SIDEWAYS


def test_detect_market_texture_transitioning():
    """Test 4: Ambiguous state returns TRANSITIONING texture."""
    texture = detect_market_texture(adx=22.0, bb_width=0.03, atr_ratio=1.1, price_change_5m=0.002)
    assert texture == TEXTURE_TRANSITIONING


def test_mean_reversion_skips_non_sideways(sample_sideways_df):
    """Test 5: MeanReversionStrategy returns None if market texture is not SIDEWAYS."""
    strategy = MeanReversionStrategy()
    sig = strategy.generate_signal("PLTR", sample_sideways_df, market_texture=TEXTURE_TRENDING)
    assert sig is None


def test_mean_reversion_rsi_confirmation_rejection(sample_sideways_df):
    """Test 6: Fails Layer 2 if RSI > 35."""
    strategy = MeanReversionStrategy()
    # Modify last close to be closer to VWAP so RSI > 35
    df = sample_sideways_df.copy()
    df.loc[df.index[-1], "close"] = 99.5
    sig = strategy.generate_signal("PLTR", df, market_texture=TEXTURE_SIDEWAYS)
    assert sig is None


def test_mean_reversion_volume_spike_rejection(sample_sideways_df):
    """Test 7: Fails Layer 3 if Volume Ratio > 2.0 (heavy breakdown)."""
    strategy = MeanReversionStrategy()
    df = sample_sideways_df.copy()
    df.loc[df.index[-1], "volume"] = 50000.0  # 5x avg volume
    sig = strategy.generate_signal("PLTR", df, market_texture=TEXTURE_SIDEWAYS)
    assert sig is None
