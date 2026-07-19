"""
intraday_data.py -- Fetch intraday OHLCV data via yfinance / Alpaca.

Provides 1m/5m bars, previous day close, and premarket data.
Caching is intentionally short-lived (30s default) for live use.
"""
import logging
import asyncio
import time as _time
import pickle
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
import os

import config
from strategies.performance import profile_latency

logger = logging.getLogger(__name__)

_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
os.makedirs(_cache_dir, exist_ok=True)
os.environ['YFINANCE_DATA'] = _cache_dir


def _ensure_asyncio_event_loop() -> None:
    try:
        asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _cache_key(
    symbol: str,
    interval: str,
    lookback_days: int = None,
    prepost: bool = False,
    source: str = "yf",
) -> str:
    session = "prepost" if prepost else "regular"
    lookback = lookback_days if lookback_days is not None else "default"
    return f"{source}_{symbol.upper()}_{interval}_{lookback}_{session}"


def _get_cache_file(key: str) -> str:
    return os.path.join(_cache_dir, f"{key}.pkl")


def _read_cache(key: str) -> Optional[pd.DataFrame]:
    cache_file = _get_cache_file(key)
    if not os.path.exists(cache_file):
        return None
    try:
        mtime = os.path.getmtime(cache_file)
        if (_time.time() - mtime) > config.CACHE_TTL_SECONDS:
            return None
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _write_cache(key: str, df: pd.DataFrame):
    cache_file = _get_cache_file(key)
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(df, f)
    except Exception as e:
        logger.warning(f"Failed to write cache for {key}: {e}")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance OHLCV columns to lower snake_case."""
    _PRICE_COLS = {"open", "high", "low", "close", "adj close", "volume"}

    def _normalize_yfinance_col(col):
        if isinstance(col, tuple):
            parts = [
                str(x).strip()
                for x in col
                if x is not None and str(x).strip()
            ]
            for part in parts:
                if part.lower() in _PRICE_COLS:
                    return part.lower().replace(" ", "_")
            return "_".join(parts).lower().replace(" ", "_")

        return str(col).strip().lower().replace(" ", "_")

    df = df.copy()
    df.columns = [_normalize_yfinance_col(c) for c in df.columns]
    return df


def _validate_intraday_data(df: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """Validates intraday OHLCV data for staleness, NaN values, and frozen feeds."""
    if df.empty:
        return df

    try:
        # 1. NaN / None Guard
        if df[["open", "high", "low", "close"]].isna().any().any():
            logger.warning("Data validation failed for %s: NaN values detected.", symbol)
            return pd.DataFrame()

        # 2. Frozen Price Guard (Price Unchanged AND Volume is Zero)
        if len(df) >= 8:
            last_8 = df.iloc[-8:]
            price_frozen = last_8["close"].nunique() == 1
            volume_frozen = last_8["volume"].sum() == 0
            if price_frozen and volume_frozen:
                logger.warning("Data validation failed for %s: True frozen price feed detected.", symbol)
                return pd.DataFrame()

        # 3. Stale Quote Guard
        last_timestamp = df.index[-1]
        
        if last_timestamp.tzinfo is None:
            import pytz
            tz = pytz.timezone(getattr(config, "TIMEZONE", "US/Eastern"))
            last_timestamp = tz.localize(last_timestamp)
            
        current_time = pd.Timestamp.now(tz=last_timestamp.tz)
        
        # We only apply staleness check if market is currently open
        from strategies.session import is_market_open
        if is_market_open():
            staleness_seconds = (current_time - last_timestamp).total_seconds()
            
            # Interval-based formula: interval_seconds * 2 + buffer
            interval_seconds = 60 # Default to 1m
            if "m" in interval:
                interval_seconds = int(interval.replace("m", "")) * 60
            elif "h" in interval:
                interval_seconds = int(interval.replace("h", "")) * 3600
                
            buffer_seconds = getattr(config, "STALE_QUOTE_BUFFER_SECONDS", 30)
            max_stale = (interval_seconds * 2) + buffer_seconds 

            if staleness_seconds > max_stale:
                logger.warning(
                    "Data validation failed for %s: Stale quote. "
                    "Last bar was %ss ago (Limit: %ss).",
                    symbol, staleness_seconds, max_stale
                )
                return pd.DataFrame()
    except Exception as e:
        logger.error("Error checking validation for %s: %s", symbol, e)
        return pd.DataFrame()

    return df


def _interval_to_alpaca(interval: str) -> str:
    if interval == "1m": return "1 min"
    if interval == "5m": return "5 mins"
    if interval == "15m": return "15 mins"
    if interval == "1d": return "1 day"
    return "1 min"


def _bars_to_frame(bars) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "date": getattr(bar, "date", getattr(bar, "timestamp", None)),
            "open": getattr(bar, "open", None),
            "high": getattr(bar, "high", None),
            "low": getattr(bar, "low", None),
            "close": getattr(bar, "close", None),
            "volume": getattr(bar, "volume", None),
        }
        for bar in bars
    ])


@profile_latency(sample_rate=1.0)
def fetch_intraday(
    symbol: str,
    interval: str = None,
    lookback_days: int = None,
    prepost: bool = False,
    bridge = None,
) -> pd.DataFrame:
    interval = interval or config.INTRADAY_INTERVAL
    lookback_days = lookback_days or config.INTRADAY_LOOKBACK_DAYS

    source = "alpaca" if bridge is not None and bridge.is_connected else "yf"
    key = _cache_key(symbol, interval, lookback_days, prepost, source=source)
    cached_df = _read_cache(key)
    if cached_df is not None:
        return cached_df.copy()

    if bridge is None or not bridge.is_connected:
        return fetch_intraday_yfinance(symbol, interval, lookback_days, prepost)

    fallback_key = _cache_key(
        symbol,
        interval,
        lookback_days,
        prepost,
        source=f"{source}_fallback",
    )
    fallback_df = _read_cache(fallback_key)
    if fallback_df is not None:
        return fallback_df.copy()
        
    try:
        bars = []
        for attempt in range(1, 4):
            bars = bridge.fetch_historical_data(
                symbol=symbol,
                durationStr=f"{lookback_days} D",
                barSizeSetting=_interval_to_alpaca(interval),
                useRTH=not prepost
            )
            if bars:
                break
            logger.debug(f"broker intraday fetch failed for {symbol} (attempt {attempt}/3). Retrying in {attempt}s...")
            bridge.ib.sleep(attempt)

        if not bars:
            logger.warning("No broker intraday data returned for %s after 3 attempts. Falling back to yfinance.", symbol)
            df = fetch_intraday_yfinance(symbol, interval, lookback_days, prepost)
            if not df.empty:
                _write_cache(fallback_key, df.copy())
            return df
            
        df = _bars_to_frame(bars)
        df = df.rename(columns={"date": "datetime"})
        df.set_index("datetime", inplace=True)
        df = _validate_intraday_data(df, symbol, interval)
        if not df.empty:
            _write_cache(key, df.copy())
        logger.debug("Fetched %d broker bars for %s", len(df), symbol)
        return df
    except Exception as e:
        logger.exception("Failed to fetch broker data for %s: %s. Falling back to yfinance.", symbol, e)
        df = fetch_intraday_yfinance(symbol, interval, lookback_days, prepost)
        if not df.empty:
            _write_cache(fallback_key, df.copy())
        return df


def fetch_intraday_yfinance(
    symbol: str,
    interval: str = None,
    lookback_days: int = None,
    prepost: bool = False,
) -> pd.DataFrame:
    interval = interval or config.INTRADAY_INTERVAL
    lookback_days = lookback_days or config.INTRADAY_LOOKBACK_DAYS

    if interval == "1m" and lookback_days > 30:
        logger.warning("yfinance only supports up to 30 days of 1m data. Capping lookback to 30 days.")
        lookback_days = 30

    key = _cache_key(symbol, interval, lookback_days, prepost, source="yf")
    cached_df = _read_cache(key)
    if cached_df is not None:
        return cached_df.copy()

    period = f"{lookback_days}d"
    df = pd.DataFrame()
    try:
        if interval == "1m" and lookback_days > 7:
            import datetime
            dfs = []
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=lookback_days)
            
            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + datetime.timedelta(days=7), end_date)
                start_str = current_start.strftime("%Y-%m-%d")
                end_str = current_end.strftime("%Y-%m-%d")
                
                chunk_df = yf.download(
                    symbol,
                    start=start_str,
                    end=end_str,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                    prepost=prepost,
                )
                if not chunk_df.empty:
                    dfs.append(chunk_df)
                current_start = current_end
                
            if dfs:
                df = pd.concat(dfs)
                df = df[~df.index.duplicated(keep='first')]
                df = df.sort_index()
        else:
            df = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                prepost=prepost,
            )
    except Exception as e:
        logger.error(
            "yfinance intraday error for %s (period=%s, interval=%s): %s",
            symbol, period, interval, str(e),
        )
        return pd.DataFrame()

    if df.empty:
        logger.warning(
            "yfinance EMPTY intraday %s (period=%s, interval=%s).",
            symbol, period, interval,
        )
        return df

    df = _normalize_columns(df)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            logger.warning("Missing column %s for %s", col, symbol)
            return pd.DataFrame()

    df = df.dropna(subset=["close"])
    df = _validate_intraday_data(df, symbol, interval)
    if not df.empty:
        _write_cache(key, df.copy())
    logger.debug(
        "Fetched %d bars for %s (%s, prepost=%s)",
        len(df), symbol, interval, prepost,
    )
    return df


@profile_latency(sample_rate=1.0)
def fetch_daily(symbol: str, lookback_days: int = None, bridge = None) -> pd.DataFrame:
    lookback_days = lookback_days or config.DAILY_LOOKBACK_DAYS
    source = "alpaca" if bridge is not None and bridge.is_connected else "yf"
    key = _cache_key(symbol, "1d", lookback_days, False, source=source)
    cached_df = _read_cache(key)
    if cached_df is not None:
        return cached_df.copy()

    if bridge is None or not bridge.is_connected:
        return fetch_daily_yfinance(symbol, lookback_days)

    fallback_key = _cache_key(
        symbol,
        "1d",
        lookback_days,
        False,
        source=f"{source}_fallback",
    )
    fallback_df = _read_cache(fallback_key)
    if fallback_df is not None:
        return fallback_df.copy()
        
    try:
        bars = []
        for attempt in range(1, 4):
            bars = bridge.fetch_historical_data(
                symbol=symbol,
                durationStr=f"{lookback_days} D",
                barSizeSetting="1 day",
                useRTH=True
            )
            if bars:
                break
            logger.debug(f"broker daily fetch failed for {symbol} (attempt {attempt}/3). Retrying in {attempt}s...")
            bridge.ib.sleep(attempt)
            
        if not bars:
            logger.warning("No broker daily data returned for %s after 3 attempts. Falling back to yfinance.", symbol)
            df = fetch_daily_yfinance(symbol, lookback_days)
            if not df.empty:
                _write_cache(fallback_key, df.copy())
            return df
            
        df = _bars_to_frame(bars)
        df = df.rename(columns={"date": "datetime"})
        df.set_index("datetime", inplace=True)
        _write_cache(key, df.copy())
        return df
    except Exception as e:
        logger.exception("Failed to fetch broker daily data for %s: %s. Falling back to yfinance.", symbol, e)
        df = fetch_daily_yfinance(symbol, lookback_days)
        if not df.empty:
            _write_cache(fallback_key, df.copy())
        return df


def fetch_daily_yfinance(symbol: str, lookback_days: int = None) -> pd.DataFrame:
    lookback_days = lookback_days or config.DAILY_LOOKBACK_DAYS
    key = _cache_key(symbol, "1d", lookback_days, False, source="yf")
    cached_df = _read_cache(key)
    if cached_df is not None:
        return cached_df.copy()

    period = f"{lookback_days}d"
    try:
        df = yf.download(
            symbol, period=period, interval="1d",
            progress=False, auto_adjust=True,
        )
    except Exception as e:
        logger.error(
            "yfinance daily error for %s (period=%s): %s",
            symbol, period, str(e),
        )
        return pd.DataFrame()

    if df.empty:
        logger.warning(
            "yfinance EMPTY daily %s (period=%s).",
            symbol, period,
        )
        return df

    df = _normalize_columns(df)
    df = df.dropna(subset=["close"])
    _write_cache(key, df.copy())
    return df


def previous_close(symbol: str, bridge=None) -> Optional[float]:
    df = fetch_daily(symbol, bridge=bridge)
    if df.empty:
        return None
    return _previous_session_close(df)


def previous_close_yfinance(symbol: str) -> Optional[float]:
    df = fetch_daily_yfinance(symbol)
    if df.empty:
        return None
    return _previous_session_close(df)


def today_open(symbol: str, bridge=None) -> Optional[float]:
    df = fetch_intraday(symbol, bridge=bridge)
    if df.empty:
        return None
    return _latest_session_open(df)


def today_open_yfinance(symbol: str) -> Optional[float]:
    df = fetch_intraday_yfinance(symbol)
    if df.empty:
        return None
    return _latest_session_open(df)


def _previous_session_close(df: pd.DataFrame) -> Optional[float]:
    if "close" not in df.columns or df.empty:
        return None
    index_dates = _index_dates(df.index)
    if index_dates is None:
        return float(df["close"].iloc[-1])

    today = pd.Timestamp.now(tz=config.TIMEZONE).date()
    prior_rows = df.loc[index_dates < today]
    if prior_rows.empty:
        return None
    return float(prior_rows["close"].iloc[-1])


def _latest_session_open(df: pd.DataFrame) -> Optional[float]:
    if "open" not in df.columns or df.empty:
        return None
    index_dates = _index_dates(df.index)
    if index_dates is None:
        return float(df["open"].iloc[0])
    latest_date = index_dates[-1]
    return float(df.loc[index_dates == latest_date, "open"].iloc[0])


def _index_dates(index: pd.Index):
    if isinstance(index, pd.DatetimeIndex):
        return index.date
    if len(index) and isinstance(index[0], (date, datetime, pd.Timestamp)):
        converted = pd.to_datetime(index, errors="coerce")
        if not converted.isna().any():
            return converted.date
    return None


def clear_cache():
    import glob
    files = glob.glob(os.path.join(_cache_dir, "*.pkl"))
    for f in files:
        try:
            os.remove(f)
        except OSError:
            pass
