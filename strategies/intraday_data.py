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
from dataclasses import dataclass

import pandas as pd
import numpy as np
import yfinance as yf
import os
import threading

import config
from strategies.performance import profile_latency

_cache_lock = threading.RLock()

logger = logging.getLogger(__name__)

_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
os.makedirs(_cache_dir, exist_ok=True)
os.environ['YFINANCE_DATA'] = _cache_dir


def _ensure_asyncio_event_loop() -> None:
    try:
        asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


_alpaca_client = None
_client_lock = threading.Lock()

def _get_alpaca_client():
    global _alpaca_client
    if _alpaca_client is None:
        with _client_lock:
            if _alpaca_client is None:
                api_key = (
                    os.environ.get("APCA_API_KEY_ID", "")
                    or os.environ.get("DAYTRADE_APCA_API_KEY_ID", "")
                    or os.environ.get("SWING_APCA_API_KEY_ID", "")
                )
                secret_key = (
                    os.environ.get("APCA_API_SECRET_KEY", "")
                    or os.environ.get("DAYTRADE_APCA_API_SECRET_KEY", "")
                    or os.environ.get("SWING_APCA_API_SECRET_KEY", "")
                )
                if api_key and secret_key:
                    from alpaca.data.historical import StockHistoricalDataClient
                    _alpaca_client = StockHistoricalDataClient(api_key, secret_key)
    return _alpaca_client


def _cache_key(
    symbol: str,
    interval: str,
    lookback_days: int = None,
    prepost: bool = False,
    source: str = "yf",
) -> str:
    session = "prepost" if prepost else "regular"
    lookback = lookback_days if lookback_days is not None else "default"
    from strategies.session import now_eastern
    now = now_eastern()
    today_str = now.strftime("%Y-%m-%d")
    if interval in ("5m", "5Min"):
        bar_minute = (now.minute // 5) * 5
        bar_time = now.replace(minute=bar_minute, second=0, microsecond=0)
        today_str = f"{today_str}_{bar_time.strftime('%H%M')}"
    return f"{source}_{symbol.upper()}_{interval}_{lookback}_{session}_{today_str}"


def _get_cache_file(key: str) -> str:
    return os.path.join(_cache_dir, f"{key}.json")


def _read_cache(key: str, ttl: int = None) -> Optional[pd.DataFrame]:
    with _cache_lock:
        cache_file = _get_cache_file(key)
        # Delete legacy .pkl if present for security
        legacy_pkl = os.path.join(_cache_dir, f"{key}.pkl")
        if os.path.exists(legacy_pkl):
            try:
                os.remove(legacy_pkl)
            except Exception:
                pass

        if not os.path.exists(cache_file):
            return None
        try:
            mtime = os.path.getmtime(cache_file)
            limit = ttl if ttl is not None else getattr(config, "INTRADAY_CACHE_TTL_SECONDS", 30)
            if (_time.time() - mtime) > limit:
                return None
            df = pd.read_json(cache_file, orient="split")
            return df
        except Exception as e:
            logger.debug(f"Failed to read JSON cache for {key}: {e}")
            return None


def _write_cache(key: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    with _cache_lock:
        cache_file = _get_cache_file(key)
        temp_file = cache_file + ".tmp"
        try:
            df.to_json(temp_file, orient="split", date_format="iso")
            os.replace(temp_file, cache_file)
        except Exception as e:
            logger.warning(f"Failed to write JSON cache for {key}: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass



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


QUARANTINE_TTL: float = 1800.0          # 30 minutes
STALE_THRESHOLD: float = 300.0          # >300s = stale


def is_quarantined(symbol: str, now: Optional[float] = None) -> bool:
    """True if symbol is quarantined. Auto-releases on TTL expiry."""
    try:
        from strategies.scanner.premarket_scanner import is_quarantined as _scan_is_quarantined
        return _scan_is_quarantined(symbol, now=now)
    except Exception as exc:
        logger.exception(
            "[QUARANTINE LOOKUP FAILURE] symbol=%s error=%s",
            symbol,
            exc,
        )
        return False


def quarantine(symbol: str, reason: str = "", now: Optional[float] = None) -> None:
    """Quarantine a symbol for QUARANTINE_TTL."""
    try:
        from strategies.scanner.premarket_scanner import quarantine_symbol
        quarantine_symbol(symbol, reason=reason, now=now)
    except Exception:
        pass


def clear_quarantine() -> None:
    """Session-reset hook."""
    try:
        from strategies.scanner.premarket_scanner import clear_quarantine as _scan_clear
        _scan_clear()
    except Exception:
        pass


def _is_frozen_feed(bars) -> bool:
    """Helper to detect constant close prices in bars or DataFrame."""
    if bars is None:
        return False
    if hasattr(bars, "frozen"):
        return bool(getattr(bars, "frozen", False))
    if isinstance(bars, pd.DataFrame):
        if 'close' in bars.columns and len(bars) >= 10:
            return bool(bars['close'].tail(10).nunique() == 1)
    elif isinstance(bars, list) and len(bars) >= 10:
        closes = [getattr(b, "close", None) for b in bars[-10:]]
        return bool(len(set(closes)) == 1 and closes[0] is not None)
    return False


def _bar_age(bars, now: Optional[float] = None) -> float:
    """Calculate the age in seconds of the latest bar timestamp."""
    if bars is None:
        return 999999.0
    if hasattr(bars, "age_seconds"):
        return float(getattr(bars, "age_seconds", 0.0))
    if isinstance(bars, pd.DataFrame) and not bars.empty:
        last_ts = bars.index[-1]
        if hasattr(last_ts, "timestamp"):
            bar_epoch = last_ts.timestamp()
        else:
            return 0.0
        cur_epoch = now if now is not None else _time.time()
        age = cur_epoch - bar_epoch
        future_tolerance = float(
            getattr(
                config,
                "BAR_FUTURE_TOLERANCE_SECONDS",
                5.0,
            )
        )
        if age < -future_tolerance:
            return float("inf")
        return max(0.0, age)
    return 0.0


def validate_feed(symbol: str, bars, now: Optional[float] = None) -> tuple[bool, str]:
    """Per-cycle feed gate. Quarantines on frozen/stale, skips if already caged."""
    if is_quarantined(symbol, now=now):
        return (False, "QUARANTINED")

    if _is_frozen_feed(bars):
        quarantine(symbol, "FROZEN_FEED", now=now)
        return (False, "FROZEN_FEED")

    if _bar_age(bars, now=now) > STALE_THRESHOLD:
        quarantine(symbol, f"STALE>{int(STALE_THRESHOLD)}s", now=now)
        return (False, "STALE")

    return (True, "OK")


def _in_market_open_transition() -> bool:
    try:
        from strategies.session import current_session_time
        t = current_session_time()
        return t.hour == 9 and 30 <= t.minute < 35
    except Exception:
        return False


@dataclass(frozen=True)
class CompletedBar:
    row: pd.Series
    start_time: pd.Timestamp
    close_time: pd.Timestamp
    post_close_delay_seconds: float


def _normalize_utc_timestamp(
    value,
    *,
    field_name: str = "timestamp",
) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        source_tz = getattr(config, "INTRADAY_DATA_SOURCE_TIMEZONE", "US/Eastern")
        if source_tz:
            ts = ts.tz_localize(source_tz)
        else:
            raise ValueError(f"{field_name} must be timezone-aware (or INTRADAY_DATA_SOURCE_TIMEZONE configured)")
    return ts.tz_convert("UTC")


def _normalize_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)
    if index.tz is None:
        source_tz = getattr(config, "INTRADAY_DATA_SOURCE_TIMEZONE", "US/Eastern")
        if source_tz:
            index = index.tz_localize(source_tz)
        else:
            raise ValueError("DataFrame index must be timezone-aware (or INTRADAY_DATA_SOURCE_TIMEZONE configured)")
    return index.tz_convert("UTC")


def _parse_interval_seconds(interval: str) -> int:
    """Parse interval string into duration in seconds, supporting standard aliases."""
    normalized = str(interval).strip().lower()

    aliases = {
        "1min": "1m",
        "5min": "5m",
        "15min": "15m",
        "30min": "30m",
        "60min": "1h",
    }

    normalized = aliases.get(normalized, normalized)

    try:
        if normalized.endswith("m"):
            value = int(normalized[:-1])
            seconds = value * 60
        elif normalized.endswith("h"):
            value = int(normalized[:-1])
            seconds = value * 3600
        elif normalized.endswith("d"):
            value = int(normalized[:-1])
            seconds = value * 86400
        elif normalized.endswith("s"):
            value = int(normalized[:-1])
            seconds = value
        else:
            raise ValueError(
                f"Unsupported interval format: {interval!r}"
            )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid interval: {interval!r}"
        ) from exc

    if seconds <= 0:
        raise ValueError(
            f"Interval must be positive: {interval!r}"
        )

    return seconds


def select_last_completed_bar(
    df: pd.DataFrame,
    interval_seconds: int,
    now_utc: pd.Timestamp,
    close_tolerance_seconds: float = 0.0,
) -> CompletedBar:
    """
    Return the latest bar that has completed by now_utc.

    close_tolerance_seconds permits a small positive allowance for
    clock-alignment differences. It is not the stale-data grace period.
    """
    if df is None or df.empty:
        raise ValueError("No bars available")

    if interval_seconds <= 0:
        raise ValueError(
            f"interval_seconds must be positive, got {interval_seconds}"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError("Intraday bar index must be monotonic increasing")

    if df.index.has_duplicates:
        raise ValueError("Intraday bar index contains duplicate timestamps")

    normalized_now = _normalize_utc_timestamp(
        now_utc,
        field_name="now_utc",
    )

    normalized_index = _normalize_utc_index(df.index)

    close_times = normalized_index + pd.Timedelta(
        seconds=interval_seconds
    )

    selection_cutoff = normalized_now + pd.Timedelta(
        seconds=close_tolerance_seconds
    )

    completed_mask = close_times <= selection_cutoff

    if not completed_mask.any():
        earliest_close = close_times.min()
        raise ValueError(
            "No completed bars available. "
            f"Earliest close={earliest_close.isoformat()}, "
            f"now={normalized_now.isoformat()}"
        )

    completed_positions = completed_mask.nonzero()[0]
    selected_position = int(completed_positions[-1])

    bar_start = normalized_index[selected_position]
    bar_close = close_times[selected_position]

    return CompletedBar(
        row=df.iloc[selected_position],
        start_time=bar_start,
        close_time=bar_close,
        post_close_delay_seconds=float(
            (normalized_now - bar_close).total_seconds()
        ),
    )


def _validate_intraday_data(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    validate_staleness: bool = True,
    purpose: str = "ENTRY_SIGNAL",
) -> pd.DataFrame:
    """
    Validates intraday DataFrame for zero/negative prices, frozen feeds, and stale quotes.
    Returns empty DataFrame if validation fails.
    Quarantines symbol on persistent data failures to suppress log spam.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    def _quarantine(reason: str) -> pd.DataFrame:
        """Helper: quarantine and return empty df."""
        try:
            quarantine(symbol, reason=reason)
        except Exception:
            pass
        return pd.DataFrame()

    try:
        # 0. Timestamp & Index Integrity Check (Duplicate / Non-monotonic index or column)
        if df.index.has_duplicates:
            logger.warning("Data validation failed for %s: Duplicate timestamp index detected.", symbol)
            return _quarantine("duplicate_timestamp_index")

        if not df.index.is_monotonic_increasing:
            logger.warning("Data validation failed for %s: Non-monotonic timestamp index detected.", symbol)
            return _quarantine("non_monotonic_timestamp_index")

        if "timestamp" in df.columns:
            ts_col = df["timestamp"]
            if ts_col.duplicated().any():
                logger.warning("Data validation failed for %s: Duplicate timestamp column detected.", symbol)
                return _quarantine("duplicate_timestamp_column")
            if not ts_col.is_monotonic_increasing:
                logger.warning("Data validation failed for %s: Non-monotonic timestamp column detected.", symbol)
                return _quarantine("non_monotonic_timestamp_column")

        # 1. Price Sanitization & Non-Finite Check (NaN, Inf, Non-positive prices)
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        if price_cols:
            try:
                price_frame = df[price_cols].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
            except Exception as exc:
                logger.warning(
                    "Data validation failed for %s: price conversion error: %s",
                    symbol,
                    exc,
                )
                return _quarantine("invalid_price_type")

            finite_mask = np.isfinite(
                price_frame.to_numpy(dtype=float)
            )

            if not finite_mask.all():
                logger.warning(
                    "Data validation failed for %s: NaN/Inf OHLC price detected.",
                    symbol,
                )
                return _quarantine("non_finite_price")

            if (price_frame <= 0).any().any():
                logger.warning(
                    "Data validation failed for %s: non-positive price detected.",
                    symbol,
                )
                return _quarantine("non_positive_price")

        # 1.1. Volume Sanitization & Non-Finite Check
        if "volume" in df.columns:
            try:
                volume = pd.to_numeric(
                    df["volume"],
                    errors="coerce",
                )
                if not np.isfinite(volume.to_numpy(dtype=float)).all() or (volume < 0).any():
                    logger.warning(
                        "Data validation failed for %s: NaN/Inf or negative volume detected.",
                        symbol,
                    )
                    return _quarantine("invalid_volume")
            except Exception as exc:
                logger.warning(
                    "Data validation failed for %s: volume conversion error: %s",
                    symbol,
                    exc,
                )
                return _quarantine("invalid_volume")

        # 1.5. Price Integrity Check (High < Low)
        if 'high' in df.columns and 'low' in df.columns:
            if (df['high'] < df['low']).any():
                logger.warning("Data validation failed for %s: High < Low inverted price detected.", symbol)
                return _quarantine("high_lt_low_inverted")

        # 1.6. Price Sanity & Discontinuity Guard (Single-bar jump/crash > 50%)
        if 'close' in df.columns and len(df) >= 2:
            pct_change = df['close'].pct_change().dropna()
            if (pct_change.abs() > 0.50).any():
                # [SHADOW OBSERVABILITY]: Evaluate RVOL and Multi-bar surge for research audit
                if 'volume' in df.columns and len(df) >= 5:
                    recent_vol = float(df['volume'].tail(3).mean())
                    avg_vol = float(df['volume'].tail(50).mean()) if len(df) >= 50 else float(df['volume'].mean())
                    rvol = (recent_vol / avg_vol) if avg_vol > 0 else 0.0
                    multi_bar = int((df['volume'].tail(3) > avg_vol * 2.0).sum()) if avg_vol > 0 else 0
                    logger.info("[SHADOW CATALYST AUDIT] %s extreme move >50%% detected | RVOL=%.1fx | multi_bar_surge=%d/3 bars", symbol, rvol, multi_bar)
                logger.warning("Data validation failed for %s: Extreme single-bar price discontinuity (>50%%) detected.", symbol)
                return _quarantine("price_discontinuity_gt50pct")

        # 1.7. Series-Wide Median Drift & Outlier Sanity Guard
        if 'close' in df.columns and len(df) >= 10:
            median_p = float(df['close'].median())
            last_p = float(df['close'].iloc[-1])
            if median_p > 0 and (last_p / median_p > 2.5 or last_p / median_p < 0.40):
                logger.warning(
                    "Data validation failed for %s: Extreme series-wide drift detected (Last: $%.2f vs Median: $%.2f).",
                    symbol, last_p, median_p
                )
                return _quarantine("series_wide_drift")

        # 2. Frozen Feed Check (Constant close prices)
        if 'close' in df.columns and len(df) >= 10:
            recent_closes = df['close'].tail(10)
            if recent_closes.nunique() == 1:
                logger.warning("Data validation failed for %s: True frozen price feed detected.", symbol)
                return _quarantine("frozen_price_feed")

        # 2.5. ATR & Feature Integrity Check
        if "atr" in df.columns and len(df) > 0:
            last_atr = float(df["atr"].iloc[-1])
            if np.isnan(last_atr) or last_atr <= 0:
                logger.warning("Data validation failed for %s: Invalid ATR %.4f detected.", symbol, last_atr)
                return _quarantine("invalid_atr")

        # 3. Stale Quote Guard (Only for live real-time feeds when validate_staleness is True)
        if validate_staleness:
            from strategies.session import is_market_open
            if is_market_open():
                interval_seconds = _parse_interval_seconds(interval)

                # Determine current UTC time
                current_time = pd.Timestamp.now(tz="UTC")
                future_tolerance = float(getattr(config, "BAR_FUTURE_TOLERANCE_SECONDS", 5.0))

                if purpose == "POSITION_MONITORING":
                    grace_seconds = float(getattr(config, "POSITION_ATR_MAX_AGE_SECONDS", 360.0))
                else:
                    grace_seconds = float(getattr(config, "ENTRY_BAR_CLOSE_GRACE_SECONDS", getattr(config, "BAR_CLOSE_GRACE_SECONDS", 45.0)))

                try:
                    completed_bar = select_last_completed_bar(
                        df=df,
                        interval_seconds=interval_seconds,
                        now_utc=current_time,
                        close_tolerance_seconds=future_tolerance,
                    )
                except ValueError as ve:
                    err_msg = str(ve)
                    if "No completed bars available" in err_msg:
                        msg = f"Data validation failed for {symbol}: No completed bars available ({err_msg})."
                        logger.warning("[FUTURE-BAR-GUARD] %s", msg)
                        return _quarantine("future_dated_bar")
                    else:
                        logger.warning("Data validation failed for %s: %s", symbol, err_msg)
                        return _quarantine("invalid_bar_data")

                delay_seconds = completed_bar.post_close_delay_seconds

                if delay_seconds > grace_seconds:
                    msg = f"Data validation: {symbol} completed bar is stale: post-close delay={delay_seconds:.1f}s (grace={grace_seconds:.1f}s, purpose={purpose}). Skipping candle for this cycle."
                    blacklist = set(getattr(config, "SYMBOL_STALE_QUOTE_BLACKLIST", []))
                    if symbol in blacklist or _in_market_open_transition():
                        logger.debug("[STALE-QUOTE-SKIP] %s", msg)
                    else:
                        logger.warning("[STALE-BAR-SKIP] %s", msg)
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


def fetch_intraday_alpaca(
    symbol: str,
    interval: str = None,
    lookback_days: int = None,
    prepost: bool = False,
    purpose: str = "ENTRY_SIGNAL",
) -> pd.DataFrame:
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
    if not api_key or not secret_key:
        return pd.DataFrame()

    interval = interval or config.INTRADAY_INTERVAL
    lookback_days = lookback_days or config.INTRADAY_LOOKBACK_DAYS

    key = _cache_key(symbol, interval, lookback_days, prepost, source="alpaca_api")
    cached_df = _read_cache(key)
    if cached_df is not None:
        validated_cache = _validate_intraday_data(
            cached_df.copy(),
            symbol=symbol,
            interval=interval,
            validate_staleness=(
                lookback_days is not None
                and lookback_days <= 1
            ),
            purpose=purpose,
        )
        if not validated_cache.empty:
            return validated_cache
        logger.warning(
            "[CACHE INVALIDATED] %s cached alpaca intraday data failed validation",
            symbol,
        )

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import Adjustment
        from datetime import timezone

        client = _get_alpaca_client()
        if client is None:
            return pd.DataFrame()
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days or 1)
        tf = TimeFrame.Minute
        if interval == "5m":
            tf = TimeFrame(5, TimeFrameUnit.Minute)
        elif interval == "15m":
            tf = TimeFrame(15, TimeFrameUnit.Minute)
        elif interval == "1h":
            tf = TimeFrame.Hour

        bars_dict = None
        primary_feed = str(getattr(config, "ALPACA_DATA_FEED", "iex")).lower()
        backup_feed = "sip" if primary_feed == "iex" else "iex"

        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol.upper(),
                timeframe=tf,
                start=start,
                feed=primary_feed,
                adjustment=Adjustment.ALL,
            )
            bars_dict = client.get_stock_bars(req)
        except Exception as primary_err:
            logger.debug("Alpaca %s feed error for %s (%s). Trying %s feed.", primary_feed.upper(), symbol, primary_err, backup_feed.upper())
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol.upper(),
                    timeframe=tf,
                    start=start,
                    feed=backup_feed,
                    adjustment=Adjustment.ALL,
                )
                bars_dict = client.get_stock_bars(req)
            except Exception as backup_err:
                logger.warning("Alpaca data feed error for %s (%s).", symbol, backup_err)
                return pd.DataFrame()

        # If primary feed returned empty, try fallback feed before giving up
        if not hasattr(bars_dict, "df") or bars_dict.df.empty:
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol.upper(),
                    timeframe=tf,
                    start=start,
                    feed=backup_feed,
                    adjustment=Adjustment.ALL,
                )
                bars_dict = client.get_stock_bars(req)
            except Exception:
                pass

        if not hasattr(bars_dict, "df") or bars_dict.df.empty:
            return pd.DataFrame()

        df = bars_dict.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol.upper(), level="symbol")

        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        df.index.name = "datetime"
        validate_staleness = (lookback_days is not None and lookback_days <= 1)
        df = _validate_intraday_data(df, symbol, interval, validate_staleness=validate_staleness, purpose=purpose)
        if not df.empty:
            _write_cache(key, df.copy())
        return df
    except Exception as e:
        logger.warning("Alpaca data fetch failed for %s: %s", symbol, e)
        return pd.DataFrame()


@profile_latency(sample_rate=1.0)
def fetch_intraday(
    symbol: str,
    interval: str = None,
    lookback_days: int = None,
    prepost: bool = False,
    bridge = None,
    purpose: str = "ENTRY_SIGNAL",
) -> pd.DataFrame:
    if is_quarantined(symbol):
        return pd.DataFrame()

    interval = interval or config.INTRADAY_INTERVAL
    lookback_days = lookback_days or config.INTRADAY_LOOKBACK_DAYS

    has_alpaca_keys = bool(os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"))
    if bridge is not None and bridge.is_connected:
        source = "alpaca"
    elif has_alpaca_keys:
        source = "alpaca_api"
    else:
        source = "yf"

    key = _cache_key(symbol, interval, lookback_days, prepost, source=source)
    cached_df = _read_cache(key)
    if cached_df is not None:
        validated_cache = _validate_intraday_data(
            cached_df.copy(),
            symbol=symbol,
            interval=interval,
            validate_staleness=(
                lookback_days is not None
                and lookback_days <= 1
            ),
            purpose=purpose,
        )
        if not validated_cache.empty:
            return validated_cache
        logger.warning(
            "[CACHE INVALIDATED] %s cached intraday data failed validation",
            symbol,
        )

    if bridge is None or not bridge.is_connected:
        df_alpaca = fetch_intraday_alpaca(symbol, interval, lookback_days, prepost, purpose=purpose)
        if not df_alpaca.empty:
            return df_alpaca
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
        validated_fallback = _validate_intraday_data(
            fallback_df.copy(),
            symbol=symbol,
            interval=interval,
            validate_staleness=(
                lookback_days is not None
                and lookback_days <= 1
            ),
            purpose=purpose,
        )
        if not validated_fallback.empty:
            return validated_fallback
        logger.warning(
            "[CACHE INVALIDATED] %s cached fallback data failed validation",
            symbol,
        )
        
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
        df = _validate_intraday_data(df, symbol, interval, purpose=purpose)
        if not df.empty:
            _write_cache(key, df.copy())
        logger.debug("Fetched %d broker bars for %s", len(df), symbol)
        return df
    except Exception as e:
        logger.exception("Failed to fetch broker data for %s: %s. Falling back to yfinance.", symbol, e)
        df = fetch_intraday_yfinance(symbol, interval, lookback_days, prepost, purpose=purpose)
        if not df.empty:
            _write_cache(fallback_key, df.copy())
        return df


def fetch_intraday_yfinance(
    symbol: str,
    interval: str = None,
    lookback_days: int = None,
    prepost: bool = False,
    purpose: str = None,
) -> pd.DataFrame:
    interval = interval or config.INTRADAY_INTERVAL
    lookback_days = lookback_days or config.INTRADAY_LOOKBACK_DAYS

    if interval == "1m" and lookback_days > 7:
        logger.info("yfinance 1m data requested for >7 days. Automatically using 5m interval for 30-day lookback.")
        interval = "5m"

    key = _cache_key(symbol, interval, lookback_days, prepost, source="yf")
    cached_df = _read_cache(key)
    if cached_df is not None:
        validated_cache = _validate_intraday_data(
            cached_df.copy(),
            symbol=symbol,
            interval=interval,
            validate_staleness=(
                lookback_days is not None
                and lookback_days <= 1
            ),
            purpose=purpose,
        )
        if not validated_cache.empty:
            return validated_cache
        logger.warning(
            "[CACHE INVALIDATED] %s cached yfinance intraday data failed validation",
            symbol,
        )

    yf_symbol = symbol.strip().upper().replace(".", "-")
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
                    yf_symbol,
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
                yf_symbol,
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
        # Attempt Alpaca fallback on yfinance exception
        return fetch_intraday_alpaca(symbol, interval, lookback_days, prepost, purpose=purpose)

    if df.empty:
        # Fallback to Alpaca data feed if yfinance is empty or rate-limited
        df_alpaca = fetch_intraday_alpaca(symbol, interval, lookback_days, prepost, purpose=purpose)
        if not df_alpaca.empty:
            return df_alpaca
        logger.debug(
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
    validate_staleness = (lookback_days is not None and lookback_days <= 1)
    df = _validate_intraday_data(df, symbol, interval, validate_staleness=validate_staleness, purpose=purpose)
    if not df.empty:
        _write_cache(key, df.copy())
    logger.debug(
        "Fetched %d bars for %s (%s, prepost=%s)",
        len(df), symbol, interval, prepost,
    )
    return df


def fetch_daily_alpaca(symbol: str, lookback_days: int = None) -> pd.DataFrame:
    api_key = os.environ.get("APCA_API_KEY_ID", "")
    secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
    if not api_key or not secret_key:
        return pd.DataFrame()

    lookback_days = lookback_days or config.DAILY_LOOKBACK_DAYS
    key = _cache_key(symbol, "1d", lookback_days, False, source="alpaca_daily")
    # Daily bars don't change during market day — cache for 12 hours (43200s)
    cached_df = _read_cache(key, ttl=43200)
    if cached_df is not None:
        return cached_df.copy()

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment
        from datetime import timezone

        client = StockHistoricalDataClient(api_key, secret_key)
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        bars_dict = None
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol.upper().strip(),
                timeframe=TimeFrame.Day,
                start=start,
                feed="iex",
                adjustment=Adjustment.ALL,
            )
            bars_dict = client.get_stock_bars(req)
        except Exception:
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol.upper().strip(),
                    timeframe=TimeFrame.Day,
                    start=start,
                    feed="sip",
                    adjustment=Adjustment.ALL,
                )
                bars_dict = client.get_stock_bars(req)
            except Exception:
                return pd.DataFrame()
        if not hasattr(bars_dict, "df") or bars_dict.df.empty:
            return pd.DataFrame()

        df = bars_dict.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol.upper().strip(), level="symbol")

        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        df.index.name = "datetime"
        if not df.empty:
            _write_cache(key, df.copy())
        return df
    except Exception as e:
        logger.warning("fetch_daily_alpaca failed for %s: %s", symbol, e)
        return pd.DataFrame()


@profile_latency(sample_rate=1.0)
def fetch_daily(symbol: str, lookback_days: int = None, bridge = None) -> pd.DataFrame:
    lookback_days = lookback_days or config.DAILY_LOOKBACK_DAYS
    has_alpaca_keys = bool(os.environ.get("APCA_API_KEY_ID") and os.environ.get("APCA_API_SECRET_KEY"))

    if bridge is not None and bridge.is_connected:
        source = "alpaca"
    elif has_alpaca_keys:
        source = "alpaca_daily"
    else:
        source = "yf"

    key = _cache_key(symbol, "1d", lookback_days, False, source=source)
    ttl = 43200 if "alpaca" in source else config.CACHE_TTL_SECONDS
    cached_df = _read_cache(key, ttl=ttl)
    if cached_df is not None:
        return cached_df.copy()

    if bridge is None or not bridge.is_connected:
        if has_alpaca_keys:
            df_alpaca = fetch_daily_alpaca(symbol, lookback_days)
            if not df_alpaca.empty:
                return df_alpaca
        return fetch_daily_yfinance(symbol, lookback_days)

    fallback_key = _cache_key(
        symbol,
        "1d",
        lookback_days,
        False,
        source=f"{source}_fallback",
    )
    fallback_df = _read_cache(fallback_key, ttl=config.CACHE_TTL_SECONDS)
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
            logger.warning("No broker daily data returned for %s after 3 attempts. Falling back to Alpaca/yfinance.", symbol)
            if has_alpaca_keys:
                df_alpaca = fetch_daily_alpaca(symbol, lookback_days)
                if not df_alpaca.empty:
                    _write_cache(fallback_key, df_alpaca.copy())
                    return df_alpaca
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
        logger.exception("Failed to fetch broker daily data for %s: %s. Falling back to Alpaca/yfinance.", symbol, e)
        if has_alpaca_keys:
            df_alpaca = fetch_daily_alpaca(symbol, lookback_days)
            if not df_alpaca.empty:
                _write_cache(fallback_key, df_alpaca.copy())
                return df_alpaca
        df = fetch_daily_yfinance(symbol, lookback_days)
        if not df.empty:
            _write_cache(fallback_key, df.copy())
        return df


def fetch_daily_yfinance(symbol: str, lookback_days: int = None) -> pd.DataFrame:
    lookback_days = lookback_days or config.DAILY_LOOKBACK_DAYS
    key = _cache_key(symbol, "1d", lookback_days, False, source="yf")
    cached_df = _read_cache(key, ttl=config.CACHE_TTL_SECONDS)
    if cached_df is not None:
        return cached_df.copy()

    yf_symbol = symbol.strip().upper().replace(".", "-")
    period = f"{lookback_days}d"
    try:
        df = yf.download(
            yf_symbol, period=period, interval="1d",
            progress=False, auto_adjust=True,
        )
    except Exception as e:
        logger.error(
            "yfinance daily error for %s (period=%s): %s",
            symbol, period, str(e),
        )
        return fetch_daily_alpaca(symbol, lookback_days)

    if df.empty:
        df_alpaca = fetch_daily_alpaca(symbol, lookback_days)
        if not df_alpaca.empty:
            return df_alpaca
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


def verify_cross_source_price(symbol: str, price: float, max_deviation_pct: float = 0.20) -> tuple[bool, str]:
    """Cross-verify primary price against secondary market data source (e.g. yfinance vs Alpaca).
    Returns (True, 'OK') if within acceptable deviation (default 20%), or (False, reason) if anomaly detected.
    """
    if price <= 0:
        return False, "Non-positive price"
    try:
        import yfinance as yf
        t = yf.Ticker(symbol.upper().strip())
        sec_price = getattr(t.fast_info, "last_price", None)
        if sec_price is None or sec_price <= 0:
            sec_price = getattr(t.fast_info, "previous_close", None)
        if sec_price and sec_price > 0:
            dev = abs(price - sec_price) / sec_price
            if dev > max_deviation_pct:
                return False, f"Cross-source deviation {dev*100:.1f}% exceeds max {max_deviation_pct*100:.1f}% (Primary: ${price:.2f} vs Secondary: ${sec_price:.2f})"
        return True, "Verified"
    except Exception as exc:
        logger.debug("Cross-source price check error for %s: %s", symbol, exc)
        return True, f"Check bypassed: {exc}"


def get_intraday_features(
    symbol: str,
    interval: str = "5m",
    lookback_days: int = None,
    bridge = None,
    purpose: str = "POSITION_MONITORING",
) -> pd.DataFrame:
    """
    Fetch intraday data with purpose-specific validation, truncate at latest completed bar,
    and compute indicators so downstream features.iloc[-1] is guaranteed closed-bar truth.
    """
    df = fetch_intraday(symbol, interval=interval, lookback_days=lookback_days, bridge=bridge, purpose=purpose)
    if df is None or df.empty:
        return pd.DataFrame()

    try:
        interval_seconds = _parse_interval_seconds(interval)
        now_utc = pd.Timestamp.now(tz="UTC")
        future_tolerance = float(getattr(config, "BAR_FUTURE_TOLERANCE_SECONDS", 5.0))
        completed = select_last_completed_bar(
            df,
            interval_seconds=interval_seconds,
            now_utc=now_utc,
            close_tolerance_seconds=future_tolerance,
        )
        normalized_index = _normalize_utc_index(df.index)
        completed_positions = (normalized_index <= completed.start_time).nonzero()[0]
        if len(completed_positions) == 0:
            raise ValueError("Completed bar position could not be resolved")
        selected_position = int(completed_positions[-1])
        df = df.iloc[: selected_position + 1].copy()
    except Exception as exc:
        logger.warning(
            "[COMPLETED BAR FEATURE GATE] %s feature generation blocked: %s",
            symbol,
            exc,
        )
        return pd.DataFrame()

    from strategies.indicators import add_all_indicators
    try:
        features = add_all_indicators(df)
    except Exception as exc:
        logger.exception("[FEATURE COMPUTATION FAILURE] %s: %s", symbol, exc)
        return pd.DataFrame()

    if features is None or features.empty:
        return pd.DataFrame()
    return features

