"""
tests/test_vwap_geometry_rejection.py -- Comprehensive regression tests for VWAP Bounce geometry validation.

Covers:
1. Valid BUY geometry returns accepted TradeSignal.
2. Invalid BUY ordering (stop >= price or target <= price) returns None.
3. BUY insufficient R:R (actual_rr < min_rr - 1e-4) returns None.
4. Missing BUY stop geometry (stop_geom is None) returns None without UnboundLocalError.
5. Valid SELL geometry returns accepted TradeSignal.
6. Invalid SELL ordering (price >= stop or price <= target) returns None.
7. SELL insufficient R:R (actual_rr < min_rr - 1e-4) returns None.
8. Missing SELL stop geometry (stop_geom is None) returns None without UnboundLocalError.
9. Non-finite geometry values (NaN, Inf) return None.
10. Zero actual risk or zero actual reward returns None.
11. Exact threshold & floating point tolerance boundary cases.
"""
from dataclasses import dataclass
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest

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

    # Condition: pullback touches VWAP tolerance and holds above
    reclaim_bars = getattr(config, "VWAP_RECLAIM_BARS", 3)
    df.iloc[-reclaim_bars - 1, df.columns.get_loc("low")] = base_price  # touches VWAP
    for i in range(-reclaim_bars, 0):
        df.iloc[i, df.columns.get_loc("close")] = base_price + 0.20 + 0.05 * (i + reclaim_bars)
    df.iloc[-1, df.columns.get_loc("close")] = base_price + 0.40
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


# ----------------------------------------------------------------------
# 1. Valid BUY geometry returns accepted TradeSignal
# ----------------------------------------------------------------------
def test_valid_buy_geometry_returns_signal():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)
    price = float(df.iloc[-1]["close"])  # 100.40

    # Stop at 99.80 (risk = 0.60), upper vwap target 102.00 (reward = 1.60), rr = 2.67 >= 1.0
    mock_geom = MockStopGeom(stop_price=99.80, final_distance=0.60)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is not None
    assert sig.side == "BUY"
    assert sig.entry_price == price
    assert sig.stop_price == 99.80
    assert sig.target_price > price
    assert 0 < sig.stop_price < sig.entry_price < sig.target_price
    assert sig.strategy == StrategyName.VWAP_BOUNCE


# ----------------------------------------------------------------------
# 2. Invalid BUY ordering returns None
# ----------------------------------------------------------------------
def test_invalid_buy_ordering_stop_above_price_returns_none():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)
    price = float(df.iloc[-1]["close"])

    # Inverted: stop price (101.00) is ABOVE entry price (100.40)
    mock_geom = MockStopGeom(stop_price=price + 0.60, final_distance=0.60)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


def test_invalid_buy_ordering_target_below_price_returns_none():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)
    # vwap_upper lower than price
    df["vwap_upper"] = 99.0

    mock_geom = MockStopGeom(stop_price=99.80, final_distance=0.60)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        with patch("config.DEFAULT_TARGET_RR_RATIO", -1.0):
            sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


# ----------------------------------------------------------------------
# 3. BUY insufficient R:R returns None
# ----------------------------------------------------------------------
def test_buy_insufficient_rr_returns_none():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)
    price = float(df.iloc[-1]["close"])  # 100.40

    # Risk = 2.00 (stop = 98.40), target at vwap_upper = 101.00 (reward = 0.60) -> rr = 0.30 < min_rr (1.0)
    df["vwap_upper"] = 101.00
    mock_geom = MockStopGeom(stop_price=98.40, final_distance=2.00)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


# ----------------------------------------------------------------------
# 4. Missing BUY stop geometry returns None without UnboundLocalError
# ----------------------------------------------------------------------
def test_missing_buy_stop_geom_returns_none_without_exception():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)

    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=None):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


def test_validator_exception_returns_none_without_unbound_error():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)

    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", side_effect=ValueError("Invalid geometry")):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


# ----------------------------------------------------------------------
# 5. Valid SELL geometry returns accepted TradeSignal
# ----------------------------------------------------------------------
def _build_valid_sell_df(n_bars: int = 35, base_price: float = 100.0) -> pd.DataFrame:
    """Builds a DataFrame that triggers the VWAP rally touch & held below precondition."""
    times = pd.date_range("2026-07-20 09:30:00", periods=n_bars, freq="1min", tz="US/Eastern")
    # Set low well below VWAP (base_price - 2.0) so pullback_touched is False
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
    # Rally touch: bar_high touches VWAP within tolerance (base_price = 100.0)
    df.iloc[-reclaim_bars - 1, df.columns.get_loc("high")] = base_price
    for i in range(-reclaim_bars, 0):
        df.iloc[i, df.columns.get_loc("close")] = base_price - 0.20 - 0.05 * (i + reclaim_bars)
    df.iloc[-1, df.columns.get_loc("close")] = base_price - 0.40
    return df


def test_valid_sell_geometry_returns_signal():
    config.ALLOW_SHORT = True
    strat = VWAPBounceStrategy()
    df = _build_valid_sell_df(35, 100.0)
    price = float(df.iloc[-1]["close"])  # 99.60

    # SELL: stop at 100.20 (risk = 0.60), target at vwap_lower = 98.00 (reward = 1.60), rr = 2.67 >= 1.0
    mock_geom = MockStopGeom(stop_price=100.20, final_distance=0.60)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is not None
    assert sig.side == "SELL"
    assert sig.entry_price == price
    assert sig.stop_price == 100.20
    assert sig.target_price < price
    assert 0 < sig.target_price < sig.entry_price < sig.stop_price


# ----------------------------------------------------------------------
# 6. Invalid SELL ordering returns None
# ----------------------------------------------------------------------
def test_invalid_sell_ordering_stop_below_price_returns_none():
    config.ALLOW_SHORT = True
    strat = VWAPBounceStrategy()
    df = _build_valid_sell_df(35, 100.0)
    price = float(df.iloc[-1]["close"])

    # Stop is BELOW entry price (inverted for short)
    mock_geom = MockStopGeom(stop_price=price - 0.50, final_distance=0.50)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


# ----------------------------------------------------------------------
# 7. SELL insufficient R:R returns None
# ----------------------------------------------------------------------
def test_sell_insufficient_rr_returns_none():
    config.ALLOW_SHORT = True
    strat = VWAPBounceStrategy()
    df = _build_valid_sell_df(35, 100.0)

    # Risk = 2.00 (stop = 101.60), target at vwap_lower = 99.00 (reward = 0.60) -> rr = 0.30 < 1.0
    df["vwap_lower"] = 99.00
    mock_geom = MockStopGeom(stop_price=101.60, final_distance=2.00)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


# ----------------------------------------------------------------------
# 8. Missing SELL stop geometry returns None without UnboundLocalError
# ----------------------------------------------------------------------
def test_missing_sell_stop_geom_returns_none_without_exception():
    config.ALLOW_SHORT = True
    strat = VWAPBounceStrategy()
    df = _build_valid_sell_df(35, 100.0)

    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=None):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


# ----------------------------------------------------------------------
# 9. Non-finite values return None
# ----------------------------------------------------------------------
def test_nan_stop_geom_returns_none():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)

    mock_geom = MockStopGeom(stop_price=float("nan"), final_distance=0.60)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


def test_inf_risk_returns_none():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)

    mock_geom = MockStopGeom(stop_price=99.50, final_distance=float("inf"))
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


# ----------------------------------------------------------------------
# 10. Zero actual risk or reward returns None
# ----------------------------------------------------------------------
def test_zero_risk_returns_none():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)

    mock_geom = MockStopGeom(stop_price=100.40, final_distance=0.0)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None


# ----------------------------------------------------------------------
# 11. Exact threshold & floating point tolerance boundary cases
# ----------------------------------------------------------------------
def test_rr_exactly_at_min_threshold_passes():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)
    price = float(df.iloc[-1]["close"])  # 100.40
    # Risk = 1.00 (stop = 99.40), Target = 101.40 (reward = 1.00) -> actual_rr = 1.00 == min_rr (1.0)
    df["vwap_upper"] = price + 1.00
    mock_geom = MockStopGeom(stop_price=price - 1.00, final_distance=1.00)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is not None
    assert sig.side == "BUY"


def test_rr_within_tolerance_passes():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)
    price = float(df.iloc[-1]["close"])  # 100.40
    # min_rr is 1.0; 1e-4 tolerance means actual_rr >= 0.9999 passes
    # reward = 1.00, risk = 1.00005 -> actual_rr = 1.00 / 1.00005 = 0.99995 >= (1.0 - 1e-4 = 0.9999) -> passes
    df["vwap_upper"] = price + 1.00
    mock_geom = MockStopGeom(stop_price=price - 1.00, final_distance=1.00005)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is not None
    assert sig.side == "BUY"


def test_rr_below_tolerance_fails():
    strat = VWAPBounceStrategy()
    df = _build_valid_base_df(35, 100.0)
    price = float(df.iloc[-1]["close"])  # 100.40
    # reward = 1.00, risk = 1.0002 -> actual_rr = 1.00 / 1.0002 = 0.9998 < (1.0 - 1e-4 = 0.9999) -> fails
    df["vwap_upper"] = price + 1.00
    mock_geom = MockStopGeom(stop_price=price - 1.00, final_distance=1.0002)
    with patch("strategies.vwap_strategy.calculate_directional_stop_and_validate", return_value=mock_geom):
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)

    assert sig is None
