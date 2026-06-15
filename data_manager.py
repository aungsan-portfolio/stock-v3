"""
data_manager.py — OHLCV data fetch + caching + feature engineering.

Priority-1 production fixes:
- Cache key includes (symbol, period, interval), not symbol only.
- Cached DataFrames are copied on read/write to avoid accidental mutation.
- make_labels() preserves unknown future rows as NaN instead of converting them to 0.
"""
import logging
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# key = (symbol, period, interval), value = (timestamp, dataframe)
_cache: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}


def fetch_ohlcv(
    symbol: str,
    period: str = None,
    interval: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance with a small in-memory TTL cache."""
    symbol = symbol.upper().strip()
    period = period or config.PRICE_PERIOD
    interval = interval or config.PRICE_INTERVAL
    cache_key = (symbol, period, interval)

    cached = _cache.get(cache_key)
    if not force_refresh and cached:
        ts, cached_df = cached
        if time.time() - ts < config.CACHE_TTL_SECONDS:
            return cached_df.copy()

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} data missing required columns: {missing}")

    df = df[required_cols].copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    if len(df) < config.MIN_HISTORY_DAYS:
        raise ValueError(
            f"{symbol} only {len(df)} days — need {config.MIN_HISTORY_DAYS}"
        )

    _cache[cache_key] = (time.time(), df.copy())
    logger.info("Fetched %d rows for %s", len(df), symbol)
    return df.copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    close, high, low, volume = f["Close"], f["High"], f["Low"], f["Volume"]

    f["sma_short"] = close.rolling(config.SMA_SHORT).mean()
    f["sma_long"] = close.rolling(config.SMA_LONG).mean()
    f["ema"] = close.ewm(span=config.EMA_PERIOD, adjust=False).mean()
    # Close-normalized so the feature is comparable across price levels/symbols
    # instead of carrying raw dollar magnitude.
    f["sma_cross"] = (f["sma_short"] - f["sma_long"]) / close
    f["dist_ema"] = (close - f["ema"]) / close

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = close.ewm(span=config.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=config.MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd - macd_sig
    # Close-normalized MACD family so magnitudes are price-level independent.
    f["macd"] = macd / close
    f["macd_sig"] = macd_sig / close
    f["macd_hist"] = macd_hist / close

    bb_mid = close.rolling(config.BOLLINGER_PERIOD).mean()
    bb_std = close.rolling(config.BOLLINGER_PERIOD).std()
    f["bb_upper"] = bb_mid + config.BOLLINGER_STD * bb_std
    f["bb_lower"] = bb_mid - config.BOLLINGER_STD * bb_std
    f["bb_pct"] = (close - f["bb_lower"]) / (
        f["bb_upper"] - f["bb_lower"]
    ).replace(0, np.nan)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    f["atr"] = tr.rolling(config.ATR_PERIOD).mean()
    f["atr_pct"] = f["atr"] / close

    f["vol_ma"] = volume.rolling(config.VOLUME_MA_PERIOD).mean()
    f["vol_ratio"] = volume / f["vol_ma"].replace(0, np.nan)

    f["ret_1d"] = close.pct_change(1)
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)

    f["dist_sma20"] = (close - f["sma_short"]) / close
    f["dist_sma50"] = (close - f["sma_long"]) / close

    f.dropna(inplace=True)
    return f


def make_labels(
    df: pd.DataFrame, horizon: int = None, min_profit_margin: float = None
) -> pd.Series:
    """
    Create binary labels for future direction.

    A row is labeled 1.0 only when the forward return clears
    ``min_profit_margin`` (default ``config.MIN_PROFIT_MARGIN``), so the models
    learn to predict moves big enough to cover round-trip costs, not microscopic
    noise that nets a loss after fees + slippage.

    Returns:
        1.0 if Close[t+horizon] / Close[t] - 1 > min_profit_margin
        0.0 otherwise (move too small, flat, or negative)
        NaN for rows where the future close is unknown

    Important: do NOT cast `(future_return > margin)` directly to int over the
    whole series, because NaN > margin becomes False and silently creates fake
    0 labels at the end.
    """
    horizon = horizon or config.ML_HORIZON
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if "Close" not in df.columns:
        raise ValueError("make_labels() requires a 'Close' column")
    if min_profit_margin is None:
        min_profit_margin = config.MIN_PROFIT_MARGIN

    future_return = df["Close"].shift(-horizon) / df["Close"] - 1.0
    labels = pd.Series(index=df.index, dtype="float64")

    valid = future_return.notna()
    labels.loc[valid] = (future_return.loc[valid] > min_profit_margin).astype(float)
    return labels


def get_feature_columns() -> list:
    return [
        "sma_cross",
        "dist_ema",
        "rsi",
        "macd",
        "macd_sig",
        "macd_hist",
        "bb_pct",
        "atr_pct",
        "vol_ratio",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "dist_sma20",
        "dist_sma50",
    ]
