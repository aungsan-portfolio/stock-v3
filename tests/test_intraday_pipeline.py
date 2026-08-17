"""
test_intraday_pipeline.py — Unit tests for the intraday data pipeline.

Tests:
1) Daily-resetting VWAP.
2) ORB levels projection.
3) 1m to 5m aggregation.
4) M5 trend indicators.
"""
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from intraday_pipeline import (
    calculate_vwap,
    calculate_orb,
    aggregate_to_5m,
    calculate_m5_indicators,
    IntradayPipeline,
)


def _fake_1m_data(n_days=2) -> pd.DataFrame:
    """Generate fake 1m intraday data across n_days (9:30 AM to 4:00 PM ET)."""
    np.random.seed(42)
    dates = []
    close_val = 100.0

    for day in range(n_days):
        dt_range = pd.date_range(
            start=f"2023-01-{day+1:02d} 09:30:00",
            end=f"2023-01-{day+1:02d} 16:00:00",
            freq="1min",
            tz="America/New_York",
        )
        dates.extend(dt_range)

    n = len(dates)
    close = close_val + np.cumsum(np.random.randn(n) * 0.1)

    df = pd.DataFrame({
        "Open": close * (1 + np.random.randn(n) * 0.0005),
        "High": close * (1 + np.abs(np.random.randn(n)) * 0.001),
        "Low": close * (1 - np.abs(np.random.randn(n)) * 0.001),
        "Close": close,
        "Volume": np.random.randint(100, 1000, n),
    }, index=dates)

    # Convert index back to UTC (standard pipeline representation)
    df.index = df.index.tz_convert("UTC")
    return df


class TestIntradayPipeline:
    def test_vwap_resets_daily(self):
        """VWAP should reset its cumulative summation at each market open."""
        df = _fake_1m_data(n_days=2)
        vwap = calculate_vwap(df)

        assert len(vwap) == len(df)
        assert not vwap.isna().all()

        # Check that VWAP on day 2 open is close to Typical Price of that day 2 first bar,
        # rather than carrying the cumulative sum of day 1.
        tp = (df["High"] + df["Low"] + df["Close"]) / 3.0
        day2_idx = df.index.date == pd.Timestamp("2023-01-02").date()
        day2_first_pos = np.where(day2_idx)[0][0]

        # VWAP on first bar of day 2 must be exactly equal to Typical Price of that bar
        assert abs(vwap.iloc[day2_first_pos] - tp.iloc[day2_first_pos]) < 1e-6

    def test_orb_high_low_levels(self):
        """ORB high and low must match the highest High/lowest Low of the first 15 mins."""
        df = _fake_1m_data(n_days=1)
        # First 15 mins (9:30 to 9:45 ET) is index 0 to 14 (15 bars)
        expected_high = df["High"].iloc[:15].max()
        expected_low = df["Low"].iloc[:15].min()

        orb_high, orb_low = calculate_orb(df, minutes=15)

        # Check that levels are projected correctly
        assert orb_high.iloc[-1] == expected_high
        assert orb_low.iloc[-1] == expected_low
        assert (orb_high == expected_high).all()

    def test_aggregation_to_5m(self):
        """1m bars should aggregate to 5m correctly."""
        df_1m = _fake_1m_data(n_days=1)
        df_5m = aggregate_to_5m(df_1m)

        # 391 minutes in RTH (9:30 to 16:00 inclusive is 391 bars)
        # 391 / 5 = 78.2 -> Resampled to 5Min should give 79 bars
        assert len(df_5m) == 79
        assert df_5m["Volume"].sum() == df_1m["Volume"].sum()

    def test_m5_indicators(self):
        """Check EMAs, crossover flags, and MACD calculation on 5m data."""
        df_1m = _fake_1m_data(n_days=1)
        df_5m = aggregate_to_5m(df_1m)
        df_ind = calculate_m5_indicators(df_5m)

        assert "ema_8" in df_ind.columns
        assert "ema_21" in df_ind.columns
        assert "ema_crossover" in df_ind.columns
        assert "macd" in df_ind.columns
        assert "typical_price" in df_ind.columns
        assert not df_ind["ema_crossover"].isna().all()
