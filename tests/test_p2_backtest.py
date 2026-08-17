"""
test_p2_backtest.py -- Phase 2 Safety Tests

Tests H-06 (Causal t+1 Open Fill) and H-07 (Unified Risk Validator & Execution Timestamps).
"""
import pandas as pd
import pytest

from strategies.backtester import run_backtest
from strategies.base import TradeSignal
from strategies.constants import StrategyName


def test_backtest_next_bar_open_fill(monkeypatch):
    """H-06: Backtest signal generated at bar t fills at bar t+1's Open price (+ slippage)."""
    # Create 35 bars: Bar 30 triggers signal, Bar 31 fills at Open=105.0
    dates = pd.date_range("2026-08-14 09:30", periods=35, freq="5min")
    opens = [100.0] * 30 + [105.0] + [106.0] * 4
    closes = [100.0] * 30 + [105.5] + [106.5] * 4

    df = pd.DataFrame(
        {
            "open": opens,
            "high": [x + 1.0 for x in opens],
            "low": [x - 1.0 for x in opens],
            "close": closes,
            "volume": [10000] * 35,
            "vwap": [100.0] * 35,
        },
        index=dates,
    )

    class MockStrategy:
        name = "VWAP_BOUNCE"

        def evaluate(self, symbol, current_df):
            if len(current_df) == 30:
                return TradeSignal(
                    symbol=symbol,
                    strategy=StrategyName.VWAP_BOUNCE,
                    side="BUY",
                    confidence=0.85,
                    entry_price=100.0,  # Signal price at bar 30
                    stop_price=98.0,
                    target_price=110.0,
                    atr=1.0,
                    risk_per_share=2.0,
                    reason="Test signal bar 30",
                )
            return None

    monkeypatch.setattr("strategies.backtester.fetch_intraday_yfinance", lambda sym, lookback_days=5: df)
    monkeypatch.setattr("strategies.backtester.CandlestickPatternStrategy", MockStrategy)

    result = run_backtest("NVDA", strategy_name="CANDLESTICK", lookback_days=5, override_min_confidence=0.60)
    trades = result.get("trades", [])

    assert len(trades) >= 1
    trade = trades[0]
    # Causal fill must happen at bar 31's Open (105.0 + slippage) rather than bar 30's entry_price (100.0)
    assert trade["entry"] > 104.0
    assert trade["decision_time"] == dates[29]  # Bar 30
    assert trade["fill_time"] == dates[30]      # Bar 31


def test_backtest_trade_timestamps(monkeypatch):
    """H-07: Trade records contain decision_time, submit_time, and fill_time timestamps."""
    dates = pd.date_range("2026-08-14 09:30", periods=35, freq="5min")
    opens = [100.0] * 35
    df = pd.DataFrame(
        {
            "open": opens,
            "high": [101.0] * 35,
            "low": [99.0] * 35,
            "close": opens,
            "volume": [10000] * 35,
            "vwap": [100.0] * 35,
        },
        index=dates,
    )

    class MockStrategy:
        name = "ORB"

        def evaluate(self, symbol, current_df):
            if len(current_df) == 30:
                return TradeSignal(
                    symbol=symbol,
                    strategy=StrategyName.ORB,
                    side="BUY",
                    confidence=0.80,
                    entry_price=100.0,
                    stop_price=99.0,
                    target_price=102.0,
                    atr=0.5,
                    risk_per_share=1.0,
                    reason="Test timestamp signal",
                )
            return None

    monkeypatch.setattr("strategies.backtester.fetch_intraday_yfinance", lambda sym, lookback_days=5: df)
    monkeypatch.setattr("strategies.backtester.ORBStrategy", MockStrategy)

    result = run_backtest("AAPL", strategy_name="ORB", lookback_days=5, override_min_confidence=0.60)
    trades = result.get("trades", [])

    assert len(trades) >= 1
    t = trades[0]
    assert "decision_time" in t
    assert "submit_time" in t
    assert "fill_time" in t
    assert t["decision_time"] < t["fill_time"]


def test_backtest_unified_risk_gate(monkeypatch):
    """H-07: Unified pre_trade_check vetoes candidate entries in backtest."""
    dates = pd.date_range("2026-08-14 09:30", periods=35, freq="5min")
    opens = [100.0] * 35
    df = pd.DataFrame(
        {
            "open": opens,
            "high": [101.0] * 35,
            "low": [99.0] * 35,
            "close": opens,
            "volume": [10000] * 35,
            "vwap": [100.0] * 35,
        },
        index=dates,
    )

    class MockStrategy:
        name = "MOMENTUM_SCALP"

        def evaluate(self, symbol, current_df):
            if len(current_df) == 30:
                return TradeSignal(
                    symbol=symbol,
                    strategy=StrategyName.MOMENTUM_SCALP,
                    side="BUY",
                    confidence=0.85,
                    entry_price=100.0,
                    stop_price=99.0,
                    target_price=102.0,
                    atr=0.5,
                    risk_per_share=1.0,
                    reason="Signal vetoed by risk gate",
                )
            return None

    monkeypatch.setattr("strategies.backtester.fetch_intraday_yfinance", lambda sym, lookback_days=5: df)
    monkeypatch.setattr("strategies.backtester.MomentumScalpStrategy", MockStrategy)

    # Force pre_trade_check to veto entry
    import strategies.intraday_risk as risk
    monkeypatch.setattr(risk, "pre_trade_check", lambda **kwargs: (False, "Simulated Risk Veto"))

    result = run_backtest("TSLA", strategy_name="MOMENTUM_SCALP", lookback_days=5, override_min_confidence=0.60)
    trades = result.get("trades", [])

    # Trade must be vetoed and zero trades returned
    assert len(trades) == 0
