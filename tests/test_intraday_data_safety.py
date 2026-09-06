"""
test_intraday_data_safety.py — Targeted safety and regression tests for intraday data integrity.

Covers:
1. Non-finite OHLC and volume values rejected/quarantined.
2. Duplicate timestamp index rejected/quarantined.
3. Non-monotonic timestamp index rejected/quarantined.
4. Future-only bars produce no features (fail-closed empty DataFrame).
5. Incomplete latest bar excluded from features (closed-bar guarantee).
6. Completed-bar selection failure returns empty DataFrame (no raw bypass).
7. Interval aliases parse to identical durations in seconds.
8. Cached malformed intraday frame is revalidated and invalidated.
9. Future bar age treated as infinite (not zero or fresh).
10. Valid completed bars produce full technical indicators.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from strategies.intraday_data import (
    _validate_intraday_data,
    _parse_interval_seconds,
    _bar_age,
    get_intraday_features,
    fetch_intraday,
    fetch_intraday_alpaca,
    fetch_intraday_yfinance,
    is_quarantined,
)
from strategies.scanner.premarket_scanner import clear_quarantine


@pytest.fixture(autouse=True)
def reset_quarantine():
    clear_quarantine()
    yield
    clear_quarantine()


def _make_sample_df(n_bars=30, base_time=None, freq="5min", seed=42):
    """Generate clean sample intraday OHLCV DataFrame."""
    np.random.seed(seed)
    if base_time is None:
        base_time = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4)
    timestamps = pd.date_range(start=base_time, periods=n_bars, freq=freq, tz="UTC")
    close = 150.0 + np.cumsum(np.random.randn(n_bars) * 0.2)
    df = pd.DataFrame(
        {
            "open": close * (1 + np.random.randn(n_bars) * 0.001),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.random.randint(1000, 50000, n_bars),
        },
        index=timestamps,
    )
    return df


# 1. Non-finite OHLC and volume values rejected/quarantined
def test_non_finite_ohlc_and_volume_rejected_and_quarantined():
    # NaN in close
    df_nan = _make_sample_df(10)
    df_nan.iloc[5, df_nan.columns.get_loc("close")] = np.nan
    res = _validate_intraday_data(df_nan, "NAN_TICK", "5m")
    assert res.empty
    assert is_quarantined("NAN_TICK")

    # Inf in high
    df_inf = _make_sample_df(10)
    df_inf.iloc[3, df_inf.columns.get_loc("high")] = np.inf
    res_inf = _validate_intraday_data(df_inf, "INF_TICK", "5m")
    assert res_inf.empty
    assert is_quarantined("INF_TICK")

    # NaN in volume
    df_vol = _make_sample_df(10)
    df_vol.iloc[2, df_vol.columns.get_loc("volume")] = np.nan
    res_vol = _validate_intraday_data(df_vol, "VOL_TICK", "5m")
    assert res_vol.empty
    assert is_quarantined("VOL_TICK")


# 2. Duplicate timestamp index rejected/quarantined
def test_duplicate_timestamp_index_rejected_and_quarantined():
    df = _make_sample_df(10)
    # Duplicate the 3rd index timestamp onto the 4th
    idx = list(df.index)
    idx[3] = idx[2]
    df.index = pd.DatetimeIndex(idx)
    res = _validate_intraday_data(df, "DUP_TICK", "5m")
    assert res.empty
    assert is_quarantined("DUP_TICK")


# 3. Non-monotonic timestamp index rejected/quarantined
def test_non_monotonic_timestamp_index_rejected_and_quarantined():
    df = _make_sample_df(10)
    # Swap two rows so timestamps are not monotonically increasing
    idx = list(df.index)
    idx[4], idx[5] = idx[5], idx[4]
    df.index = pd.DatetimeIndex(idx)
    res = _validate_intraday_data(df, "MONO_TICK", "5m")
    assert res.empty
    assert is_quarantined("MONO_TICK")


# 4. Future-only bars produce no features (fail-closed empty DataFrame)
def test_future_only_bars_produce_no_features():
    now_utc = pd.Timestamp.now(tz="UTC")
    future_start = now_utc + pd.Timedelta(hours=2)
    df_future = _make_sample_df(10, base_time=future_start)

    with patch("strategies.intraday_data.fetch_intraday", return_value=df_future):
        features = get_intraday_features("FUTURE_TICK", interval="5m")
        assert features.empty


# 5. Incomplete latest bar excluded from features (closed-bar guarantee)
def test_incomplete_latest_bar_excluded_from_features():
    # Construct bars where Bar 0, 1 are in past, and Bar 2 is currently active
    now_utc = pd.Timestamp.now(tz="UTC")
    # Bar 0: now - 12m to now - 7m (closed)
    # Bar 1: now - 7m to now - 2m (closed)
    # Bar 2: now - 2m to now + 3m (in progress)
    t0 = now_utc - pd.Timedelta(minutes=12)
    t1 = now_utc - pd.Timedelta(minutes=7)
    t2 = now_utc - pd.Timedelta(minutes=2)

    times = pd.DatetimeIndex([t0, t1, t2])
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [1000, 2000, 3000],
        },
        index=times,
    )

    with patch("strategies.intraday_data.fetch_intraday", return_value=df):
        features = get_intraday_features("INC_TICK", interval="5m")
        assert not features.empty
        # Bar 2 must NOT be included in features
        assert t2 not in features.index
        # Last row must be Bar 1 (t1)
        assert features.index[-1] == t1


# 6. Completed-bar selection failure returns empty DataFrame (no raw bypass)
def test_completed_bar_selection_failure_returns_empty_frame():
    df = _make_sample_df(10)
    with patch("strategies.intraday_data.fetch_intraday", return_value=df):
        with patch(
            "strategies.intraday_data.select_last_completed_bar",
            side_effect=ValueError("Simulated completion selection failure"),
        ):
            features = get_intraday_features("ERR_TICK", interval="5m")
            # Must strictly return empty DataFrame, never unvalidated raw data
            assert features.empty


# 7. Interval aliases parse to identical durations in seconds
def test_interval_aliases_parse_to_equal_duration():
    assert _parse_interval_seconds("5m") == 300
    assert _parse_interval_seconds("5Min") == 300
    assert _parse_interval_seconds("1m") == 60
    assert _parse_interval_seconds("1min") == 60
    assert _parse_interval_seconds("15m") == 900
    assert _parse_interval_seconds("15min") == 900
    assert _parse_interval_seconds("30m") == 1800
    assert _parse_interval_seconds("30min") == 1800
    assert _parse_interval_seconds("1h") == 3600
    assert _parse_interval_seconds("1d") == 86400

    with pytest.raises(ValueError, match="Invalid interval"):
        _parse_interval_seconds("invalid_interval")

    with pytest.raises(ValueError, match="Interval must be positive"):
        _parse_interval_seconds("0m")


# 8. Cached malformed intraday frame is revalidated and invalidated
def test_cached_malformed_intraday_frame_is_revalidated_and_invalidated():
    bad_cached_df = _make_sample_df(10)
    bad_cached_df.iloc[2, bad_cached_df.columns.get_loc("close")] = np.nan

    # Test fetch_intraday cache revalidation
    with patch("strategies.intraday_data._read_cache", return_value=bad_cached_df):
        with patch("strategies.intraday_data.fetch_intraday_alpaca", return_value=pd.DataFrame()):
            with patch("strategies.intraday_data.fetch_intraday_yfinance", return_value=pd.DataFrame()):
                df_out = fetch_intraday("AAPL", interval="5m", bridge=None)
                # The corrupted cache must have been rejected
                assert df_out.empty

    # Test fetch_intraday_alpaca cache revalidation
    with patch("strategies.intraday_data._read_cache", return_value=bad_cached_df):
        with patch("os.environ.get", side_effect=lambda k, default="": "mock_val" if "KEY" in k else default):
            with patch("strategies.intraday_data._get_alpaca_client", return_value=None):
                df_alpaca = fetch_intraday_alpaca("AAPL", interval="5m")
                # Should reject cache and return empty when client unavailable
                assert df_alpaca.empty

    # Test fetch_intraday_yfinance cache revalidation
    with patch("strategies.intraday_data._read_cache", return_value=bad_cached_df):
        with patch("strategies.intraday_data.fetch_intraday_alpaca", return_value=pd.DataFrame()):
            with patch("strategies.intraday_data.yf.download", return_value=pd.DataFrame()):
                df_yf = fetch_intraday_yfinance("AAPL", interval="5m")
                assert df_yf.empty


# 9. Future bar age treated as infinite (not zero or fresh)
def test_future_bar_age_treated_as_infinite():
    t0 = 1000000.0
    future_time = pd.Timestamp.fromtimestamp(t0 + 60.0, tz="UTC")
    df_future = pd.DataFrame(
        {"close": [100.0]},
        index=pd.DatetimeIndex([future_time]),
    )

    age = _bar_age(df_future, now=t0)
    assert age == float("inf")

    # A normal past bar should return positive age
    past_time = pd.Timestamp.fromtimestamp(t0 - 15.0, tz="UTC")
    df_past = pd.DataFrame(
        {"close": [100.0]},
        index=pd.DatetimeIndex([past_time]),
    )
    age_past = _bar_age(df_past, now=t0)
    assert abs(age_past - 15.0) < 1e-4


# 10. Valid completed bars produce full technical indicators
def test_valid_completed_bars_produce_indicators():
    # 60 bars in the past
    now_utc = pd.Timestamp.now(tz="UTC")
    past_start = now_utc - pd.Timedelta(hours=6)
    df_valid = _make_sample_df(n_bars=60, base_time=past_start, freq="5min")

    with patch("strategies.intraday_data.fetch_intraday", return_value=df_valid):
        features = get_intraday_features("VALID_TICK", interval="5m")
        assert not features.empty
        assert len(features) == 60
        # Verify essential indicators are calculated
        assert "vwap" in features.columns
        assert "ema_9" in features.columns
        assert "rsi" in features.columns
        assert "atr" in features.columns
        assert "macd" in features.columns
        # Guarantee closed-bar truth on the last row
        assert features.index[-1] == df_valid.index[-1]
