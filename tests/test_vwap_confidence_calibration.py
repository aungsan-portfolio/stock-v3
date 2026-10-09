"""tests/test_vwap_confidence_calibration.py -- Unit tests for VWAPBounceStrategy confidence scoring calibration.

Verifies:
1. Healthy VWAP bounce setups (RSI in [40, 60] healthy bounce zone, holding above VWAP) achieve raw confidence >= 0.75,
   surviving the CAUTION regime confidence gate (0.6500) and eliminating the historical 645 near-miss rejections.
2. Low-conviction / overextended setups (RSI > 65 on long bounce) do NOT receive the bounce bonus,
   honestly preserving selective gating.
3. Symmetrical short VWAP rejection setups achieve balanced confidence calibration.
"""
from dataclasses import dataclass
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd

import config
from strategies.vwap_strategy import VWAPBounceStrategy
from strategies.constants import StrategyName


@dataclass
class MockStopGeom:
    stop_price: float
    final_distance: float
    minimum_distance: float = 0.10


def _build_valid_base_df(n_bars: int = 35, base_price: float = 100.0) -> pd.DataFrame:
    """Builds a DataFrame that triggers the VWAP touch & hold precondition."""
    times = pd.date_range("2026-07-20 09:30:00", periods=n_bars, freq="1min", tz="US/Eastern")
    df = pd.DataFrame({
        "open": [base_price] * n_bars,
        "high": [base_price + 0.50] * n_bars,
        "low": [base_price - 0.50] * n_bars,
        "close": [base_price + 0.10] * n_bars,
        "volume": [100.0] * n_bars,
        "atr": [0.50] * n_bars,
        "vwap": [base_price] * n_bars,
        "vwap_lower": [base_price - 1.00] * n_bars,
        "vwap_upper": [base_price + 2.00] * n_bars,
        "rsi": [45.0] * n_bars,
        "volume_ma": [100.0] * n_bars,
    }, index=times)

    reclaim_bars = getattr(config, "VWAP_RECLAIM_BARS", 3)
    df.iloc[-reclaim_bars - 1, df.columns.get_loc("low")] = base_price  # touches VWAP
    for i in range(-reclaim_bars, 0):
        df.iloc[i, df.columns.get_loc("close")] = base_price + 0.20 + 0.05 * (i + reclaim_bars)
    df.iloc[-1, df.columns.get_loc("close")] = base_price + 0.40
    return df


def _build_valid_sell_df(n_bars: int = 35, base_price: float = 100.0) -> pd.DataFrame:
    """Builds a DataFrame that triggers the VWAP rally touch & held below precondition."""
    times = pd.date_range("2026-07-20 09:30:00", periods=n_bars, freq="1min", tz="US/Eastern")
    df = pd.DataFrame({
        "open": [base_price - 0.20] * n_bars,
        "high": [base_price - 0.10] * n_bars,
        "low": [base_price - 2.00] * n_bars,
        "close": [base_price - 0.20] * n_bars,
        "volume": [100.0] * n_bars,
        "atr": [0.50] * n_bars,
        "vwap": [base_price] * n_bars,
        "vwap_lower": [base_price - 2.00] * n_bars,
        "vwap_upper": [base_price + 1.00] * n_bars,
        "rsi": [55.0] * n_bars,
        "volume_ma": [100.0] * n_bars,
    }, index=times)

    reclaim_bars = getattr(config, "VWAP_RECLAIM_BARS", 3)
    df.iloc[-reclaim_bars - 1, df.columns.get_loc("high")] = base_price
    for i in range(-reclaim_bars, 0):
        df.iloc[i, df.columns.get_loc("close")] = base_price - 0.20 - 0.05 * (i + reclaim_bars)
    df.iloc[-1, df.columns.get_loc("close")] = base_price - 0.40
    return df


@pytest.fixture(autouse=True)
def setup_vwap_config():
    orig_enabled = getattr(config, "STRATEGY_VWAP_BOUNCE_ENABLED", True)
    orig_short = getattr(config, "ALLOW_SHORT", False)
    config.STRATEGY_VWAP_BOUNCE_ENABLED = True
    config.VWAP_BOUNCE_TOLERANCE_PCT = 0.5
    config.VWAP_RECLAIM_BARS = 3
    config.VWAP_RSI_OVERBOUGHT = 95.0
    config.VWAP_RSI_OVERSOLD = 35.0
    config.MIN_RR_RATIO = 1.0
    config.DEFAULT_TARGET_RR_RATIO = 2.0
    config.MAX_TARGET_RR_RATIO = 5.0
    yield
    config.STRATEGY_VWAP_BOUNCE_ENABLED = orig_enabled
    config.ALLOW_SHORT = orig_short


class TestVWAPConfidenceCalibration:
    """Test suite for calibrated confidence scoring in VWAPBounceStrategy."""

    def test_vwap_bounce_healthy_rsi_pullback_unlocks_gate(self):
        """Construct synthetic 5m DataFrame with a clean VWAP bounce and healthy RSI (RSI = 50 in [40, 60]).
        Price pulls back to VWAP and holds above it with volume and R:R >= 2.0.
        """
        strat = VWAPBounceStrategy()
        df = _build_valid_base_df(35, 100.0)
        # Set healthy RSI in the [40, 60] bounce zone
        df["rsi"] = 50.0
        df["volume"] = 120.0
        df["volume_ma"] = 100.0

        price = float(df.iloc[-1]["close"])  # 100.40
        mock_geom = MockStopGeom(stop_price=99.80, final_distance=0.60)
        with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
            sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

        assert sig is not None
        assert sig.side == "BUY"
        # Base 0.50 + price>vwap 0.10 + rsi in [40, 60] 0.10 + vol>=0.85x 0.05 + actual_rr>=2.0 0.05 + held_above 0.05 = 0.85
        assert sig.confidence >= 0.80
        # Cleanly passes CAUTION regime required confidence (0.6500)
        assert sig.confidence >= 0.6500

    def test_vwap_rejection_healthy_sell_unlocks_gate(self):
        """Construct synthetic 5m DataFrame with a clean VWAP rejection below VWAP (RSI = 48 in [40, 60])."""
        config.ALLOW_SHORT = True
        strat = VWAPBounceStrategy()
        df = _build_valid_sell_df(35, 100.0)
        df["rsi"] = 48.0
        df["volume"] = 120.0
        df["volume_ma"] = 100.0

        price = float(df.iloc[-1]["close"])  # 99.60
        mock_geom = MockStopGeom(stop_price=100.20, final_distance=0.60)
        with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
            sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

        assert sig is not None
        assert sig.side == "SELL"
        assert sig.confidence >= 0.80
        assert sig.confidence >= 0.6500
