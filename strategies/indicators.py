"""
indicators.py -- Intraday technical indicators.

All functions take a DataFrame with columns: open, high, low, close, volume
and return the DataFrame with new indicator columns appended (never mutate in place).
"""
import numpy as np
import pandas as pd

import config


def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Weighted Average Price -- resets each trading day."""
    df = df.copy()
    if "volume" not in df.columns or "close" not in df.columns:
        df["vwap"] = np.nan
        return df

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)

    if hasattr(df.index, "date"):
        dates = df.index.date
    else:
        dates = pd.Series([0] * len(df))

    # Vectorized VWAP calculation
    tp_v = typical * vol
    cum_tp_v = tp_v.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    
    vwap = cum_tp_v / cum_vol
    df["vwap"] = vwap.fillna(typical)

    # Vectorized VWAP standard deviation bands
    tp2_v = (typical ** 2) * vol
    cum_tp2_v = tp2_v.groupby(dates).cumsum()
    
    # variance = E[X^2] - (E[X])^2
    variance = (cum_tp2_v / cum_vol) - (df["vwap"] ** 2)
    variance = variance.clip(lower=0.0)
    std = variance ** 0.5
    
    df["vwap_upper"] = df["vwap"] + config.VWAP_STD_BANDS * std
    df["vwap_lower"] = df["vwap"] - config.VWAP_STD_BANDS * std
    
    return df


def add_ema(df: pd.DataFrame, period: int = None, col_name: str = None) -> pd.DataFrame:
    df = df.copy()
    period = period or config.EMA_9
    col_name = col_name or f"ema_{period}"
    df[col_name] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    df = add_ema(df, config.EMA_9, "ema_9")
    df = add_ema(df, config.EMA_20, "ema_20")
    df = add_ema(df, config.EMA_50, "ema_50")
    return df


def add_rsi(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    df = df.copy()
    period = period or config.RSI_PERIOD
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.mask((avg_gain > 0) & (avg_loss == 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    df["rsi"] = rsi.fillna(50.0)
    return df


def add_atr(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    df = df.copy()
    period = period or config.ATR_PERIOD
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=config.MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=config.MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def add_volume_ma(df: pd.DataFrame, period: int = None) -> pd.DataFrame:
    df = df.copy()
    period = period or config.VOLUME_MA_PERIOD
    df["volume_ma"] = df["volume"].rolling(period).mean()
    df["relative_volume"] = df["volume"] / df["volume_ma"].replace(0, np.nan)
    return df


def add_opening_range(df: pd.DataFrame, orb_minutes: int = None) -> pd.DataFrame:
    """Mark the opening range high/low for each day."""
    df = df.copy()
    orb_minutes = orb_minutes or config.ORB_WINDOW_MINUTES

    df["orb_high"] = np.nan
    df["orb_low"] = np.nan

    if not hasattr(df.index, "date"):
        return df

    # Localize or convert index to America/New_York timezone to handle timezone naive/aware inputs robustly
    idx = df.index
    if idx.tz is None:
        idx_eastern = idx.tz_localize("UTC").tz_convert("America/New_York")
    else:
        idx_eastern = idx.tz_convert("America/New_York")

    # Get unique dates in America/New_York
    unique_dates = pd.Series(idx_eastern.date).unique()

    for date_val in unique_dates:
        # Filter day data using New York timezone index dates
        eastern_day_mask = idx_eastern.date == date_val
        day_df = df.loc[eastern_day_mask]
        day_idx_eastern = idx_eastern[eastern_day_mask]

        if day_df.empty:
            continue

        # Get regular session bars starting at 9:30 AM America/New_York
        market_open_time = config.MARKET_OPEN  # Typically datetime.time(9, 30)
        regular_mask = day_idx_eastern.time >= market_open_time
        regular_bars = day_df.loc[regular_mask]

        if regular_bars.empty:
            continue

        day_start = regular_bars.index[0]
        orb_end = day_start + pd.Timedelta(minutes=orb_minutes)
        orb_bars = day_df.loc[(day_df.index >= day_start) & (day_df.index <= orb_end)]

        if not orb_bars.empty:
            orb_h = orb_bars["high"].max()
            orb_l = orb_bars["low"].min()
            # Causal assignment: Only populate orb_high/orb_low AFTER the ORB window completion
            post_orb_mask = eastern_day_mask & (df.index >= orb_end)
            df.loc[post_orb_mask, "orb_high"] = orb_h
            df.loc[post_orb_mask, "orb_low"] = orb_l

    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all standard intraday indicators in one call."""
    if "ema_9" in df.columns and "vwap" in df.columns:
        return df
    
    df = add_vwap(df)
    df = add_emas(df)
    df = add_rsi(df)
    df = add_atr(df)
    df = add_macd(df)
    df = add_volume_ma(df)
    df = add_opening_range(df)
    return df
