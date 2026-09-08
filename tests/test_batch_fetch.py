import sys
import os
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

import config
from strategies.intraday_data import fetch_intraday_batch, fetch_batch_intraday_alpaca
import strategies.orchestrator as orch


def _create_mock_df(symbol: str, count: int = 10, interval_seconds: int = 300) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    last_close = now - timedelta(seconds=15)
    last_start = last_close - timedelta(seconds=interval_seconds)
    first_start = last_start - timedelta(seconds=interval_seconds * (count - 1))
    idx = pd.date_range(start=first_start, periods=count, freq=f"{interval_seconds}s", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(100.0, 105.0, count),
        "high": np.linspace(101.0, 106.0, count),
        "low": np.linspace(99.0, 104.0, count),
        "close": np.linspace(100.5, 105.5, count),
        "volume": [1000.0] * count,
    }, index=idx)
    df.index.name = "datetime"
    return df


def test_batch_fetch_telemetry_and_coverage():
    """Verify that fetch_intraday_batch computes exact telemetry and coverage."""
    symbols = ["AAPL", "MSFT", "GOOGL"]
    df_aapl = _create_mock_df("AAPL")
    df_msft = _create_mock_df("MSFT")

    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    # Bridge returns 2 out of 3 symbols
    mock_bridge.fetch_batch_historical_data.return_value = {
        "AAPL": [MagicMock(date=t, open=100.0 + i, high=101.0 + i, low=99.0 + i, close=100.5 + i, volume=1000) for i, t in enumerate(df_aapl.index)],
        "MSFT": [MagicMock(date=t, open=200.0 + i, high=201.0 + i, low=199.0 + i, close=200.5 + i, volume=2000) for i, t in enumerate(df_msft.index)],
    }

    with patch("strategies.intraday_data.fetch_batch_intraday_alpaca", return_value={}):
        data, telemetry = fetch_intraday_batch(symbols, interval="5m", bridge=mock_bridge)
        assert "AAPL" in data
        assert "MSFT" in data
        assert telemetry["requested"] == 3
        assert telemetry["received"] == 2
        assert telemetry["coverage"] == round(2 / 3, 4)
        assert telemetry["duration_ms"] >= 0.0


def test_batch_fetch_schema_parity():
    """Verify DataFrame structure matches legacy single-symbol fetch requirements."""
    df_mock = _create_mock_df("NVDA", count=5)
    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.fetch_batch_historical_data.return_value = {
        "NVDA": [MagicMock(date=t, open=150.0 + i, high=155.0 + i, low=149.0 + i, close=152.0 + i, volume=5000) for i, t in enumerate(df_mock.index)]
    }

    data, telemetry = fetch_intraday_batch(["NVDA"], interval="5m", bridge=mock_bridge)
    assert "NVDA" in data
    df = data["NVDA"]

    # 1. Monotonic increasing DatetimeIndex in UTC
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert df.index.tz is not None
    assert str(df.index.tz).upper() == "UTC"
    assert df.index.name == "datetime"

    # 2. Required columns present
    required_cols = {"open", "high", "low", "close", "volume"}
    assert required_cols.issubset(set(df.columns))
    assert len(df) == 5


def test_circuit_breaker_full_fallback():
    """Verify that when batch fetch raises an exception, orchestrator degrades to sequential ThreadPool without crashing."""
    symbols = ["AAPL", "MSFT", "NVDA", "SPY"]
    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.ib.positions.return_value = []
    mock_bridge.get_net_liquidation.return_value = 100000.0

    # Mock batch fetch to simulate an unhandled network failure
    def mock_broken_batch(*args, **kwargs):
        raise RuntimeError("Simulated Alpaca Gateway Timeout 504")

    sequential_called = []
    def mock_sequential(sym, *args, **kwargs):
        sequential_called.append(sym)
        return _create_mock_df(sym, count=5)

    with patch("config.DAYTRADE_BATCH_FETCH_ENABLED", True), \
         patch("strategies.intraday_data.fetch_intraday_batch", side_effect=mock_broken_batch), \
         patch("strategies.intraday_data.fetch_intraday", side_effect=mock_sequential), \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.is_past_opening_buffer", return_value=True):

        res = orch.evaluate_symbols_parallel(symbols, bridge=mock_bridge)
        # All 4 symbols must have been recovered via sequential fallback
        assert set(sequential_called) == set(symbols)


def test_partial_symbol_recovery():
    """Verify that if batch fetch misses 1 symbol (e.g. JPM), orchestrator recovers only the missing symbol sequentially."""
    symbols = ["AAPL", "MSFT", "JPM"]
    df_aapl = _create_mock_df("AAPL", count=5)
    df_msft = _create_mock_df("MSFT", count=5)
    df_jpm = _create_mock_df("JPM", count=5)

    # Batch returns AAPL and MSFT only (missing JPM)
    batch_data = {"AAPL": df_aapl, "MSFT": df_msft}
    telemetry = {"requested": 3, "received": 2, "coverage": 0.67, "duration_ms": 12.5}

    targeted_seq_called = []
    def mock_targeted_fetch(sym, *args, **kwargs):
        targeted_seq_called.append(sym)
        if sym == "JPM":
            return df_jpm
        return pd.DataFrame()

    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.ib.positions.return_value = []
    mock_bridge.get_net_liquidation.return_value = 100000.0

    with patch("config.DAYTRADE_BATCH_FETCH_ENABLED", True), \
         patch("strategies.intraday_data.fetch_intraday_batch", return_value=(batch_data, telemetry)), \
         patch("strategies.intraday_data.fetch_intraday", side_effect=mock_targeted_fetch), \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.is_past_opening_buffer", return_value=True):

        orch.evaluate_symbols_parallel(symbols, bridge=mock_bridge)
        # ONLY JPM should have been targeted for sequential recovery, not AAPL or MSFT!
        assert targeted_seq_called == ["JPM"]


if __name__ == "__main__":
    test_batch_fetch_telemetry_and_coverage()
    test_batch_fetch_schema_parity()
    test_circuit_breaker_full_fallback()
    test_partial_symbol_recovery()
    print("All batch fetch tests passed!")
