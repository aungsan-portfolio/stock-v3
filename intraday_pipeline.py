"""
intraday_pipeline.py — 1m/5m Intraday Data Pipeline.

Provides:
1. Multi-source intraday data fetching (yfinance / Polygon.io / IBKR).
2. Daily-resetting Volume Weighted Average Price (VWAP) calculation.
3. Opening Range Breakout (ORB) level tracker (5m, 15m, 30m).
4. M5/5-minute aggregation + trend indicators (8/21 EMA crossover, MACD, typical price).
"""
import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

import config as config_module

logger = logging.getLogger(__name__)


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate daily-resetting Volume Weighted Average Price (VWAP).
    Resets cumulative sums at the start of each trading session (market open).
    Expects DatetimeIndex.
    """
    if df.empty:
        return pd.Series(dtype=np.float64)

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_v = typical_price * df["Volume"]

    # Group by calendar date to reset VWAP daily
    dates = df.index.date
    cum_tp_v = tp_v.groupby(dates).cumsum()
    cum_vol = df["Volume"].groupby(dates).cumsum()

    vwap = cum_tp_v / cum_vol.replace(0, np.nan)
    return vwap.ffill()


def calculate_orb(df: pd.DataFrame, minutes: int = 15) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Opening Range Breakout (ORB) High and Low levels.
    For each calendar day:
    - Locates the first 'minutes' of trading (e.g. 9:30 AM to 9:45 AM ET).
    - Computes the highest High and lowest Low in that range.
    - Projects these static levels forward for the remainder of the session.
    """
    if df.empty:
        return pd.Series(dtype=np.float64), pd.Series(dtype=np.float64)

    # Ensure index is timezone-aware and localized to US/Eastern
    df_et = df.copy()
    if df_et.index.tz is None:
        df_et.index = df_et.index.tz_localize("UTC").tz_convert("America/New_York")
    else:
        df_et.index = df_et.index.tz_convert("America/New_York")

    dates = df_et.index.date
    orb_high = pd.Series(index=df.index, dtype=np.float64)
    orb_low = pd.Series(index=df.index, dtype=np.float64)

    unique_dates = np.unique(dates)
    for d in unique_dates:
        day_mask = dates == d
        day_df = df_et[day_mask]
        if day_df.empty:
            continue

        # Market open is 09:30 AM Eastern
        session_start = pd.Timestamp.combine(d, pd.Timestamp("09:30:00").time())
        session_start = session_start.tz_localize("America/New_York")
        cutoff = session_start + pd.Timedelta(minutes=minutes)

        # Opening range bars
        orb_bars = day_df[(day_df.index >= session_start) & (day_df.index < cutoff)]
        if orb_bars.empty:
            # Fallback to the first available bar of the day
            high_val = day_df["High"].iloc[0]
            low_val = day_df["Low"].iloc[0]
        else:
            high_val = orb_bars["High"].max()
            low_val = orb_bars["Low"].min()

        # Set these values for the entire day's sequence
        # Convert index back to original timezone or match by position
        orb_high.iloc[day_mask] = high_val
        orb_low.iloc[day_mask] = low_val

    return orb_high, orb_low


def aggregate_to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 1-minute OHLCV data into 5-minute bars.
    Aligns open, high, low, close, and volume correctly.
    """
    if df_1m.empty:
        return pd.DataFrame()

    resampler = df_1m.resample("5Min")
    df_5m = pd.DataFrame()
    df_5m["Open"] = resampler["Open"].first()
    df_5m["High"] = resampler["High"].max()
    df_5m["Low"] = resampler["Low"].min()
    df_5m["Close"] = resampler["Close"].last()
    df_5m["Volume"] = resampler["Volume"].sum()
    df_5m.dropna(subset=["Close"], inplace=True)
    return df_5m


def calculate_m5_indicators(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute M5 trend following indicators:
    - 8 EMA & 21 EMA.
    - Crossover signal (1.0 when 8 EMA > 21 EMA, else 0.0).
    - MACD (12, 26, 9) and MACD Hist.
    - Typical Price (HLC3).
    """
    if df_5m.empty:
        return df_5m

    f = df_5m.copy()
    close = f["Close"]

    # 8 & 21 EMAs
    f["ema_8"] = close.ewm(span=8, adjust=False).mean()
    f["ema_21"] = close.ewm(span=21, adjust=False).mean()
    f["ema_crossover"] = (f["ema_8"] > f["ema_21"]).astype(float)

    # MACD
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    f["macd"] = macd_fast - macd_slow
    f["macd_signal"] = f["macd"].ewm(span=9, adjust=False).mean()
    f["macd_hist"] = f["macd"] - f["macd_signal"]

    # Typical Price
    f["typical_price"] = (f["High"] + f["Low"] + f["Close"]) / 3.0

    return f


class IntradayPipeline:
    """
    Intraday Pipeline orchestrator loading 1m/5m bars, calculating daily VWAP,
    ORB levels, and aggregation trends.
    """

    def __init__(self, data_provider=None):
        from data_providers import MultiSourceDataProvider
        self.provider = data_provider or MultiSourceDataProvider()

    def get_processed_intraday(self, symbol: str, interval: str = "5m",
                               days_back: int = 5) -> pd.DataFrame:
        """
        Fetch intraday data, process VWAP, compute ORB, and append indicator streams.
        Returns a rich DataFrame.
        """
        symbol = symbol.upper().strip()
        # Resolve interval/period parameters
        period = f"{days_back}d"

        # yfinance/Polygon use '1m' or '5m' directly
        df = self.provider.fetch(symbol, period=period, interval=interval, force_refresh=True)

        if df.empty:
            raise ValueError(f"No intraday data fetched for {symbol}")

        # Compute VWAP
        df["VWAP"] = calculate_vwap(df)

        # Compute ORB (default to 15-minute range)
        df["ORB_High"], df["ORB_Low"] = calculate_orb(df, minutes=15)

        # Add indicators if it is 5m data
        if interval == "5m":
            df = calculate_m5_indicators(df)
        elif interval == "1m":
            # For 1m data, we also provide a resampled 5m frame as attributes or metadata if needed
            pass

        return df
