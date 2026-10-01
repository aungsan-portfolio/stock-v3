"""tests/test_vwap_breadth_and_hysteresis.py -- Targeted tests for Lane 4 (D1-Breadth & Texture Dwell Hysteresis)."""

import time
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from strategies.regime_engine import (
    compute_vwap_breadth,
    evaluate_market_regime,
    RegimeStateTracker,
    MarketRegimeResult,
    MODE_RISK_ON,
    MODE_DATA_DEGRADED,
    TEXTURE_TRENDING,
    TEXTURE_TRANSITIONING,
    TEXTURE_VOLATILE,
    TEXTURE_SIDEWAYS,
)


def _make_dummy_df(above_vwap=True):
    """Create a 5-row intraday DataFrame where latest close is above or below VWAP."""
    closes = [100.0, 100.5, 101.0, 101.5, 102.0 if above_vwap else 98.0]
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    volumes = [1000.0] * 5
    df = pd.DataFrame({
        "open": closes,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })
    return df


def test_watchlist_vwap_breadth_with_sufficient_symbols():
    """When >= 5 symbols have valid data, compute breadth as % above intraday VWAP."""
    symbol_data = {
        "AAPL": _make_dummy_df(above_vwap=True),
        "MSFT": _make_dummy_df(above_vwap=True),
        "NVDA": _make_dummy_df(above_vwap=True),
        "AMZN": _make_dummy_df(above_vwap=True),
        "GOOGL": _make_dummy_df(above_vwap=False),
        "META": _make_dummy_df(above_vwap=False),
    }
    symbols = list(symbol_data.keys())

    breadth_pct, mode, details = compute_vwap_breadth(
        symbols=symbols,
        symbol_data=symbol_data,
    )

    assert mode == "WATCHLIST_VWAP"
    assert details["evaluated"] == 6
    assert details["above_vwap"] == 4
    assert pytest.approx(breadth_pct, 0.01) == 66.67


def test_watchlist_vwap_breadth_fallback_to_indices():
    """When < 5 symbols have valid data, fall back to index breadth."""
    symbol_data = {
        "AAPL": _make_dummy_df(above_vwap=True),
        "MSFT": _make_dummy_df(above_vwap=True),
    }
    symbols = ["AAPL", "MSFT"]
    data = {
        "SPY": {"pct": 0.50},
        "QQQ": {"pct": 0.30},
        "SOXX": {"pct": -0.80},
        "IWM": {"pct": 0.10},
    }

    breadth_pct, mode, details = compute_vwap_breadth(
        symbols=symbols,
        symbol_data=symbol_data,
        data=data,
    )

    assert mode == "INDEX_FALLBACK"
    assert details["evaluated"] == 4
    assert details["positive_count"] == 3  # SPY, QQQ, IWM positive
    assert pytest.approx(breadth_pct, 0.01) == 75.00


def test_macro_score_zero_risk_on_invariant():
    """Score = +0 -> RISK_ON architecture remains completely untouched."""
    bridge = MagicMock()
    now_ts = time.time()
    # SPY 0.0%, QQQ 0.0%, SOXX 0.0%
    bridge.get_price = lambda s, allow_historical=True: 100.0
    bridge.fetch_historical_data = lambda sym, *args, **kwargs: [
        MagicMock(close=100.0, timestamp=now_ts),
        MagicMock(close=100.0, timestamp=now_ts),
    ]

    res = evaluate_market_regime(
        bridge=bridge,
        scanner_candidate_count=5,
        symbols=["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"],
    )

    # Score must be 0 and mode must be RISK_ON
    assert res.score == 0
    assert res.mode == MODE_RISK_ON
    assert res.allow_new_longs is True
    assert res.halt_new_entries is False
    assert "market_breadth" in res.data


def test_texture_dwell_hysteresis_prevents_single_bar_whipping():
    """RegimeStateTracker requires N=3 consecutive bars before transitioning texture."""
    tracker = RegimeStateTracker(texture_dwell_bars=3)

    def _make_regime_res(texture):
        return MarketRegimeResult(
            mode=MODE_RISK_ON,
            score=0,
            allow_new_longs=True,
            requires_opening_buffer=False,
            position_scale=1.0,
            confidence_boost=0.0,
            market_texture=texture,
        )

    # Bar 1: Initial state is TRENDING
    r1 = tracker.update(_make_regime_res(TEXTURE_TRENDING))
    assert r1.market_texture == TEXTURE_TRENDING
    assert tracker.active_texture == TEXTURE_TRENDING

    # Bar 2: Transient 1-bar flicker to TRANSITIONING -> should be rejected by hysteresis
    r2 = tracker.update(_make_regime_res(TEXTURE_TRANSITIONING))
    assert r2.market_texture == TEXTURE_TRENDING
    assert tracker.active_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture == TEXTURE_TRANSITIONING
    assert tracker.candidate_texture_count == 1

    # Bar 3: Returns to TRENDING -> candidate resets
    r3 = tracker.update(_make_regime_res(TEXTURE_TRENDING))
    assert r3.market_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 0

    # Bar 4: TRANSITIONING bar 1
    r4 = tracker.update(_make_regime_res(TEXTURE_TRANSITIONING))
    assert r4.market_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 1

    # Bar 5: TRANSITIONING bar 2
    r5 = tracker.update(_make_regime_res(TEXTURE_TRANSITIONING))
    assert r5.market_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 2

    # Bar 6: TRANSITIONING bar 3 -> transition confirmed!
    r6 = tracker.update(_make_regime_res(TEXTURE_TRANSITIONING))
    assert r6.market_texture == TEXTURE_TRANSITIONING
    assert tracker.active_texture == TEXTURE_TRANSITIONING
    assert tracker.candidate_texture_count == 0
