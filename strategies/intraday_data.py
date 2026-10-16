"""
intraday_data.py -- Fetch intraday OHLCV data via yfinance / Alpaca.

Provides 1m/5m bars, previous day close, and premarket data.
Caching is intentionally short-lived (30s default) for live use.
"""
import logging
import asyncio
import time
import pickle
import itertools
from datetime import date, datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import pandas as pd
import numpy as np
import yfinance as yf
import os
import json
import threading
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

import config
from strategies.performance import profile_latency

_cache_lock = threading.RLock()

logger = logging.getLogger(__name__)

def _get_cache_dir() -> str:
    try:
        import config
        cdir = os.path.join(config.DATA_DIR, "cache")
    except Exception:
        cdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'cache')
    os.makedirs(cdir, exist_ok=True)
    return cdir

def __getattr__(name: str):
    if name == "_cache_dir":
        return _get_cache_dir()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

os.environ['YFINANCE_DATA'] = _get_cache_dir()


class DataProvider(str, Enum):
    ALPACA = "alpaca"
    YFINANCE = "yfinance"


@dataclass(frozen=True)
class HttpTimeoutBudget:
    connect_seconds: float = 0.75
    read_seconds: float = 1.75
    pool_seconds: float = 0.50

    @property
    def total_seconds(self) -> float:
        return self.connect_seconds + self.read_seconds + self.pool_seconds


ALPACA_TIMEOUT = HttpTimeoutBudget()


class MarketDataTransportTimeout(TimeoutError):
    pass


class MarketDataCapacityError(RuntimeError):
    pass


class RecoverableMarketDataError(Exception):
    pass


class MarketDataUnavailableError(Exception):
    pass


# ── Dedicated Position ATR Cache (Independent from bar cache) ──
@dataclass(frozen=True)
class CachedAtr:
    symbol: str
    interval: str
    atr: float
    source_bar_timestamp: datetime
    computed_at_monotonic: float


_position_atr_cache: Dict[Tuple[str, str], CachedAtr] = {}
_position_atr_lock = threading.Lock()


def update_position_atr_cache(symbol: str, interval: str, atr: float, source_bar_timestamp: Any) -> None:
    with _position_atr_lock:
        _position_atr_cache[(symbol.upper().strip(), interval)] = CachedAtr(
            symbol=symbol.upper().strip(),
            interval=interval,
            atr=float(atr),
            source_bar_timestamp=source_bar_timestamp,
            computed_at_monotonic=time.monotonic(),
        )


def _bar_timestamp_is_acceptable(source_bar_timestamp: Any, interval: str = "5m", max_bar_delay_seconds: float = 600.0) -> bool:
    try:
        now_utc = pd.Timestamp.now(tz="UTC")
        ts = pd.Timestamp(source_bar_timestamp)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        delay = (now_utc - ts).total_seconds()
        return 0 <= delay <= max_bar_delay_seconds
    except Exception:
        return False


def get_position_atr_if_fresh(
    symbol: str,
    *,
    interval: str = "5m",
    max_age_seconds: float = 360.0,
) -> Optional[float]:
    with _position_atr_lock:
        cached = _position_atr_cache.get((symbol.upper().strip(), interval))
    if cached is None:
        return None
    age = time.monotonic() - cached.computed_at_monotonic
    if age > max_age_seconds:
        return None
    if not _bar_timestamp_is_acceptable(cached.source_bar_timestamp, interval=interval):
        return None
    return cached.atr


def clear_position_atr_cache() -> None:
    with _position_atr_lock:
        _position_atr_cache.clear()


# ── Bounded Alpaca Client Execution Bridge ──
ALPACA_CALL_WORKERS = 4
ALPACA_CALL_QUEUE_SIZE = 4
ALPACA_FUTURE_TIMEOUT_SECONDS = 2.5

_alpaca_call_pool = ThreadPoolExecutor(
    max_workers=ALPACA_CALL_WORKERS,
    thread_name_prefix="alpaca-bars",
)
_alpaca_call_slots = threading.BoundedSemaphore(
    ALPACA_CALL_WORKERS + ALPACA_CALL_QUEUE_SIZE
)
_call_counter = itertools.count(1)

MARKET_DATA_METRICS = {
    "executor_active": 0,
    "executor_queue_depth": 0,
    "saturation_total": 0,
    "soft_timeout_total": 0,
    "transport_timeout_total": 0,
    "late_completion_total": 0,
    "in_flight_calls": 0,
    "timed_out_in_flight": 0,
}
_metrics_lock = threading.Lock()


def get_market_data_metrics() -> dict:
    with _metrics_lock:
        return dict(MARKET_DATA_METRICS)


_feed_telemetry = {
    "stale_bar_consecutive": 0,
    "stale_observation_total": 0,
    "stale_bar_total": 0,  # Backward compatibility mirror for stale_observation_total (cumulative checks)
    "stale_bar_skip_count": 0,
    "poll_count": 0,
    "last_batch_duration_seconds": 0.0,
    "batch_timeout_count": 0,
    "consecutive_fetch_timeouts": 0,
    "last_batch_timestamp": 0.0,
    "total_bars_fetched": 0,
    "last_batch_timeout_count": 0,
    "last_batch_forming_count": 0,
    "last_batch_reasons": {},
}
_feed_telemetry_lock = threading.Lock()
_last_stale_bar_ts: Dict[str, Any] = {}
_last_stale_cycle_bar_id: Optional[Any] = None

_breadth_telemetry = {
    "stale_bar_consecutive": 0,
    "stale_observation_total": 0,
    "stale_bar_total": 0,
    "stale_bar_skip_count": 0,
    "poll_count": 0,
    "last_batch_duration_seconds": 0.0,
    "batch_timeout_count": 0,
    "consecutive_fetch_timeouts": 0,
    "last_batch_timestamp": 0.0,
    "total_bars_fetched": 0,
    "last_batch_timeout_count": 0,
    "last_batch_forming_count": 0,
    "last_batch_reasons": {},
}
_breadth_telemetry_lock = threading.Lock()
_last_breadth_stale_bar_ts: Dict[str, Any] = {}


def get_intraday_feed_telemetry(purpose: Optional[str] = None) -> dict:
    if str(purpose or "").upper() == "REGIME_BREADTH":
        with _breadth_telemetry_lock:
            data = dict(_breadth_telemetry)
            data["last_stale_bar_ts"] = dict(_last_breadth_stale_bar_ts)
            return data
    with _feed_telemetry_lock:
        data = dict(_feed_telemetry)
        data["last_stale_bar_ts"] = dict(_last_stale_bar_ts)
        return data


get_feed_telemetry = get_intraday_feed_telemetry


def reset_intraday_feed_telemetry(purpose: Optional[str] = None) -> None:
    global _last_stale_cycle_bar_id
    for lane, lock, enabled in (
        (_breadth_telemetry, _breadth_telemetry_lock, purpose is None or str(purpose).upper() == "REGIME_BREADTH"),
        (_feed_telemetry, _feed_telemetry_lock, purpose is None or str(purpose).upper() != "REGIME_BREADTH"),
    ):
        if enabled:
            with lock:
                lane.update(last_batch_timeout_count=0, last_batch_forming_count=0,
                            last_batch_reasons={}, last_batch_requested=0,
                            last_batch_received=0, last_batch_coverage=0.0,
                            consecutive_fetch_timeouts=0)
    if purpose is None or str(purpose).upper() == "REGIME_BREADTH":
        with _breadth_telemetry_lock:
            for k in _breadth_telemetry:
                if isinstance(_breadth_telemetry[k], float):
                    _breadth_telemetry[k] = 0.0
                elif isinstance(_breadth_telemetry[k], int):
                    _breadth_telemetry[k] = 0
            _last_breadth_stale_bar_ts.clear()
    if purpose is None or str(purpose).upper() != "REGIME_BREADTH":
        with _feed_telemetry_lock:
            _feed_telemetry["stale_bar_consecutive"] = 0
            _feed_telemetry["stale_observation_total"] = 0
            _feed_telemetry["stale_bar_total"] = 0
            _feed_telemetry["stale_bar_skip_count"] = 0
            _feed_telemetry["poll_count"] = 0
            _feed_telemetry["last_batch_duration_seconds"] = 0.0
            _feed_telemetry["batch_timeout_count"] = 0
            _feed_telemetry["consecutive_fetch_timeouts"] = 0
            _feed_telemetry["last_batch_timestamp"] = 0.0
            _feed_telemetry["total_bars_fetched"] = 0
        _last_stale_bar_ts.clear()
        _last_stale_cycle_bar_id = None


def get_feed_provenance() -> dict:
    tier = str(getattr(config, "ALPACA_SUBSCRIPTION_TIER", "free")).lower().strip()
    cfg_feed = str(getattr(config, "ALPACA_DATA_FEED", "iex")).lower().strip()
    if tier == "free":
        actual_feed = "iex"
        capability = "IEX_ONLY (~2.5% US Volume / Single Exchange)"
    else:
        actual_feed = cfg_feed
        capability = "SIP_CONSOLIDATED (100% US National Volume)" if cfg_feed == "sip" else "IEX_ONLY (~2.5% US Volume / Single Exchange)"
    return {
        "configured_feed": cfg_feed,
        "actual_feed": actual_feed,
        "subscription_tier": tier,
        "feed_capability": capability,
        "bar_source": "alpaca_api",
    }


def _sdk_get_stock_bars_soft_bounded(client, request, timeout: Optional[float] = None):
    call_id = f"call_{next(_call_counter)}"
    start_mono = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()

    acquired = _alpaca_call_slots.acquire(blocking=False)
    if not acquired:
        with _metrics_lock:
            MARKET_DATA_METRICS["saturation_total"] += 1
        raise MarketDataCapacityError("Alpaca market-data executor is saturated")

    with _metrics_lock:
        MARKET_DATA_METRICS["executor_active"] += 1
        MARKET_DATA_METRICS["in_flight_calls"] += 1

    slot_released = [False]
    def _release_slot():
        if not slot_released[0]:
            slot_released[0] = True
            _alpaca_call_slots.release()
            with _metrics_lock:
                MARKET_DATA_METRICS["executor_active"] = max(0, MARKET_DATA_METRICS["executor_active"] - 1)
                MARKET_DATA_METRICS["in_flight_calls"] = max(0, MARKET_DATA_METRICS["in_flight_calls"] - 1)

    def invoke():
        try:
            return client.get_stock_bars(request)
        finally:
            _release_slot()

    try:
        future = _alpaca_call_pool.submit(invoke)
    except Exception:
        _release_slot()
        raise

    eff_timeout = timeout if timeout is not None else ALPACA_FUTURE_TIMEOUT_SECONDS
    req_symbols = getattr(request, "symbol_or_symbols", None)
    req_tf = getattr(request, "timeframe", None)
    req_start = getattr(request, "start", None)
    req_end = getattr(request, "end", None)
    req_feed = getattr(request, "feed", None)

    try:
        return future.result(timeout=eff_timeout)
    except (TimeoutError, FutureTimeoutError) as ex:
        cancelled = future.cancel()
        with _metrics_lock:
            MARKET_DATA_METRICS["soft_timeout_total"] += 1
        if cancelled:
            _release_slot()
        else:
            with _metrics_lock:
                MARKET_DATA_METRICS["timed_out_in_flight"] += 1

            def _on_late_completion(f):
                with _metrics_lock:
                    MARKET_DATA_METRICS["timed_out_in_flight"] = max(0, MARKET_DATA_METRICS["timed_out_in_flight"] - 1)
                    try:
                        f.result()
                        MARKET_DATA_METRICS["late_completion_total"] += 1
                    except Exception:
                        pass
            future.add_done_callback(_on_late_completion)

        elapsed = time.monotonic() - start_mono
        end_utc = datetime.now(timezone.utc).isoformat()
        with _metrics_lock:
            in_flight = MARKET_DATA_METRICS["in_flight_calls"]
            timed_out_flight = MARKET_DATA_METRICS["timed_out_in_flight"]

        logger.error(
            "[ALPACA SDK TIMEOUT] call_id=%s timeout=%.2fs elapsed=%.2fs cancelled=%s "
            "in_flight=%d timed_out_in_flight=%d start_utc=%s end_utc=%s "
            "symbols=%s timeframe=%s window=[%s -> %s] feed=%s",
            call_id, eff_timeout, elapsed, cancelled,
            in_flight, timed_out_flight, start_utc, end_utc,
            req_symbols, req_tf, req_start, req_end, req_feed,
        )
        raise MarketDataTransportTimeout(
            f"Alpaca SDK call exceeded caller deadline ({eff_timeout:.2f}s)"
        ) from ex
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.Timeout) as rex:
        with _metrics_lock:
            MARKET_DATA_METRICS["transport_timeout_total"] += 1
            in_flight = MARKET_DATA_METRICS["in_flight_calls"]
        elapsed = time.monotonic() - start_mono
        end_utc = datetime.now(timezone.utc).isoformat()
        logger.error(
            "[ALPACA TRANSPORT TIMEOUT] call_id=%s elapsed=%.2fs in_flight=%d "
            "start_utc=%s end_utc=%s symbols=%s feed=%s exc=%s",
            call_id, elapsed, in_flight, start_utc, end_utc, req_symbols, req_feed, rex,
        )
        raise MarketDataTransportTimeout(f"Alpaca transport timeout: {rex}") from rex


class TimeoutHTTPAdapter(HTTPAdapter):
    """HTTP adapter enforcing hard connect and read timeouts at socket transport level."""
    def __init__(self, connect_timeout: float = 0.75, read_timeout: float = 1.75, max_retries: Optional[Retry] = None, **kwargs):
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        if max_retries is None:
            max_retries = Retry(total=1, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
        super().__init__(max_retries=max_retries, **kwargs)

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (self.connect_timeout, self.read_timeout)
        return super().send(request, **kwargs)


def _ensure_asyncio_event_loop() -> None:
    try:
        asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


@dataclass(frozen=True)
class ValidatedBarBatch:
    symbol: str
    dataframe: pd.DataFrame
    latest_completed_bar_end: Optional[datetime]
    fetched_at: datetime
    bar_age_seconds: float
    fetch_generation_id: str
    source: str
    validation_status: str


def should_validate_staleness(purpose: Optional[str], lookback_days: Optional[int]) -> bool:
    """
    Determines whether staleness validation must be enforced.
    Strict Invariant: Real-time decision contexts (ENTRY_SIGNAL, REALTIME, EXECUTION, PREFLIGHT, REGIME_BREADTH)
    MUST validate freshness unconditionally, even if historical lookback_days > 1.
    """
    if str(purpose or "").upper() in ("ENTRY_SIGNAL", "REALTIME", "EXECUTION", "PREFLIGHT", "REGIME_BREADTH"):
        return True
    return bool(lookback_days is not None and lookback_days <= 1)


_alpaca_client = None
_client_lock = threading.Lock()


def _load_env_if_needed() -> None:
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip('"').strip("'")
                            os.environ.setdefault(k, v)
            except Exception as e:
                logger.warning("Failed to auto-parse .env in intraday_data: %s", e)


def _get_alpaca_credentials() -> Tuple[str, str]:
    _load_env_if_needed()
    mode = os.environ.get("TRADING_MODE", "daytrade").lower()
    if mode == "swing":
        key = (
            os.environ.get("SWING_APCA_API_KEY_ID", "")
            or os.environ.get("APCA_API_KEY_ID", "")
            or os.environ.get("DAYTRADE_APCA_API_KEY_ID", "")
        )
        secret = (
            os.environ.get("SWING_APCA_API_SECRET_KEY", "")
            or os.environ.get("APCA_API_SECRET_KEY", "")
            or os.environ.get("DAYTRADE_APCA_API_SECRET_KEY", "")
        )
    else:
        key = (
            os.environ.get("DAYTRADE_APCA_API_KEY_ID", "")
            or os.environ.get("SWING_APCA_API_KEY_ID", "")
            or os.environ.get("APCA_API_KEY_ID", "")
        )
        secret = (
            os.environ.get("DAYTRADE_APCA_API_SECRET_KEY", "")
            or os.environ.get("SWING_APCA_API_SECRET_KEY", "")
            or os.environ.get("APCA_API_SECRET_KEY", "")
        )
    return key, secret


def _has_alpaca_credentials() -> bool:
    key, secret = _get_alpaca_credentials()
    return bool(key and secret)


def configure_alpaca_data_client_timeouts(client) -> None:
    """Configures socket connect and read timeouts on an Alpaca data client session."""
    if client is not None and hasattr(client, "_session"):
        connect_t = float(getattr(config, "ALPACA_DATA_CONNECT_TIMEOUT", 0.75))
        read_t = float(getattr(config, "ALPACA_DATA_READ_TIMEOUT", 1.75))
        retries = Retry(total=1, backoff_factor=0.2, status_forcelist=[429, 500, 502, 503, 504], raise_on_status=False)
        adapter = TimeoutHTTPAdapter(connect_timeout=connect_t, read_timeout=read_t, max_retries=retries)
        client._session.mount("https://", adapter)
        client._session.mount("http://", adapter)


def _get_alpaca_client():
    global _alpaca_client
    if _alpaca_client is None:
        with _client_lock:
            if _alpaca_client is None:
                api_key, secret_key = _get_alpaca_credentials()
                if api_key and secret_key:
                    from alpaca.data.historical import StockHistoricalDataClient
                    client = StockHistoricalDataClient(api_key, secret_key)
                    configure_alpaca_data_client_timeouts(client)
                    _alpaca_client = client
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
    return os.path.join(_get_cache_dir(), f"{key}.json")


def _read_cache(key: str, ttl: int = None) -> Optional[pd.DataFrame]:
    with _cache_lock:
        cache_file = _get_cache_file(key)
        # Delete legacy .pkl if present for security
        legacy_pkl = os.path.join(_get_cache_dir(), f"{key}.pkl")
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
            if (time.time() - mtime) > limit:
                return None
            df = pd.read_json(cache_file, orient="split")
            meta_file = cache_file + ".meta"
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        loaded_attrs = json.load(f)
                        if isinstance(loaded_attrs, dict):
                            df.attrs.update(loaded_attrs)
                except Exception as meta_e:
                    logger.debug(f"Failed to read cache metadata for {key}: {meta_e}")
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
            if getattr(df, "attrs", None):
                meta_file = cache_file + ".meta"
                meta_temp = meta_file + ".tmp"
                try:
                    with open(meta_temp, "w", encoding="utf-8") as f:
                        json.dump(dict(df.attrs), f)
                    os.replace(meta_temp, meta_file)
                except Exception as meta_e:
                    logger.debug(f"Failed to write cache metadata for {key}: {meta_e}")
                    if os.path.exists(meta_temp):
                        try:
                            os.remove(meta_temp)
                        except Exception:
                            pass
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
    bar_start_ts: Optional[pd.Timestamp] = None
    bar_close_ts: Optional[pd.Timestamp] = None
    current_ts: Optional[pd.Timestamp] = None
    completed_bar_age: float = 0.0
    publication_delay: float = 0.0


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

    # An intraday candle is only completed when its close time has been reached by now_utc.
    # Positive tolerance must not treat forming/unclosed candles as completed.
    completed_mask = close_times <= normalized_now

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
    delay_sec = float((normalized_now - bar_close).total_seconds())

    return CompletedBar(
        row=df.iloc[selected_position],
        start_time=bar_start,
        close_time=bar_close,
        post_close_delay_seconds=delay_sec,
        bar_start_ts=bar_start,
        bar_close_ts=bar_close,
        current_ts=normalized_now,
        completed_bar_age=delay_sec,
        publication_delay=delay_sec,
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

        # 3. Stale Quote Guard (Enforced for live feeds or ENTRY_SIGNAL)
        effective_staleness = bool(validate_staleness or should_validate_staleness(purpose, None))
        if effective_staleness:
            from strategies.session import is_market_open
            if is_market_open():
                interval_seconds = _parse_interval_seconds(interval)

                # Determine current UTC time
                current_time = pd.Timestamp.now(tz="UTC")
                future_tolerance = float(getattr(config, "BAR_FUTURE_TOLERANCE_SECONDS", 5.0))

                if purpose == "PREFLIGHT":
                    grace_seconds = float(getattr(config, "PREFLIGHT_MAX_RECOVERABLE_DELAY_SECONDS", 360.0))
                elif purpose == "POSITION_MONITORING":
                    grace_seconds = float(getattr(config, "POSITION_ATR_MAX_AGE_SECONDS", 360.0))
                else:
                    grace_seconds = float(getattr(config, "ENTRY_BAR_CLOSE_GRACE_SECONDS", getattr(config, "BAR_CLOSE_GRACE_SECONDS", 60.0)))

                try:
                    completed_bar = select_last_completed_bar(
                        df=df,
                        interval_seconds=interval_seconds,
                        now_utc=current_time,
                        close_tolerance_seconds=0.0,
                    )
                except ValueError as ve:
                    err_msg = str(ve)
                    if "No completed bars available" in err_msg:
                        msg = f"Data validation: {symbol} has no completed {interval} bars yet by {current_time.isoformat()} ({err_msg}). Candle still forming."
                        if _in_market_open_transition():
                            logger.debug("[FUTURE-BAR-GUARD] [MARKET-OPEN-FORMING-BAR] %s", msg)
                        else:
                            logger.info("[FUTURE-BAR-GUARD] [FORMING-BAR-WAIT] %s", msg)
                        return pd.DataFrame()
                    else:
                        logger.warning("Data validation failed for %s: %s", symbol, err_msg)
                        return _quarantine("invalid_bar_data")

                delay_seconds = completed_bar.post_close_delay_seconds
                bar_id = getattr(completed_bar, "close_time", None) or getattr(completed_bar, "start_time", None)

                if str(purpose or "").upper() == "REGIME_BREADTH":
                    with _breadth_telemetry_lock:
                        _breadth_telemetry["poll_count"] += 1
                        if delay_seconds <= grace_seconds:
                            _breadth_telemetry["stale_bar_consecutive"] = 0
                            _breadth_telemetry["stale_bar_skip_count"] = 0
                        else:
                            _breadth_telemetry["stale_observation_total"] += 1
                            _breadth_telemetry["stale_bar_total"] = _breadth_telemetry["stale_observation_total"]
                            last_ts = _last_breadth_stale_bar_ts.get(symbol)
                            if last_ts != bar_id:
                                _breadth_telemetry["stale_bar_consecutive"] += 1
                                _breadth_telemetry["stale_bar_skip_count"] = _breadth_telemetry["stale_bar_consecutive"]
                                _last_breadth_stale_bar_ts[symbol] = bar_id
                else:
                    with _feed_telemetry_lock:
                        _feed_telemetry["poll_count"] += 1
                        if delay_seconds <= grace_seconds:
                            if purpose == "ENTRY_SIGNAL":
                                # Fresh completed bar resets consecutive stale counter for entry gate
                                _feed_telemetry["stale_bar_consecutive"] = 0
                                _feed_telemetry["stale_bar_skip_count"] = 0
                        else:
                            # Cumulative stale observations recorded across all polling ticks
                            _feed_telemetry["stale_observation_total"] += 1
                            _feed_telemetry["stale_bar_total"] = _feed_telemetry["stale_observation_total"]
                            last_ts = _last_stale_bar_ts.get(symbol)
                            if last_ts != bar_id:
                                # Only ENTRY_SIGNAL stale bars increment the entry circuit-breaker counter
                                if purpose == "ENTRY_SIGNAL":
                                    global _last_stale_cycle_bar_id
                                    if _last_stale_cycle_bar_id != bar_id:
                                        _last_stale_cycle_bar_id = bar_id
                                        _feed_telemetry["stale_bar_consecutive"] += 1
                                        _feed_telemetry["stale_bar_skip_count"] = _feed_telemetry["stale_bar_consecutive"]
                                else:
                                    _feed_telemetry["stale_monitoring_count"] = _feed_telemetry.get("stale_monitoring_count", 0) + 1
                                _last_stale_bar_ts[symbol] = bar_id
                            else:
                                # Same bar timestamp re-polled within intra-bar cadence: do NOT increment consecutive
                                pass

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


def _normalize_data_feed(feed: Any = None) -> Any:
    """Normalize string or DataFeed enum to alpaca.data.enums.DataFeed."""
    try:
        from alpaca.data.enums import DataFeed
        if isinstance(feed, DataFeed):
            return feed
        feed_str = str(feed or getattr(config, "ALPACA_DATA_FEED", "iex")).lower().strip()
        if feed_str == "sip":
            return DataFeed.SIP
        elif feed_str == "otc":
            return DataFeed.OTC
        return DataFeed.IEX
    except Exception:
        return str(feed or "iex").lower().strip()


def fetch_intraday_alpaca(
    symbol: str,
    interval: str = None,
    lookback_days: int = None,
    prepost: bool = False,
    purpose: str = "ENTRY_SIGNAL",
    feed: Optional[str] = None,
    timeout: Optional[float] = None,
) -> pd.DataFrame:
    """Alpaca only.
    This function must not invoke yfinance and must not try additional
    feeds unless the caller explicitly requests them.
    """
    api_key, secret_key = _get_alpaca_credentials()
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
            validate_staleness=should_validate_staleness(purpose, lookback_days),
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
        from alpaca.data.enums import Adjustment, DataFeed
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

        tier = str(getattr(config, "ALPACA_SUBSCRIPTION_TIER", "free")).lower().strip()
        if tier == "free":
            chosen_feed = DataFeed.IEX
        else:
            chosen_feed = _normalize_data_feed(feed or getattr(config, "ALPACA_DATA_FEED", "iex"))

        req = StockBarsRequest(
            symbol_or_symbols=symbol.upper().strip(),
            timeframe=tf,
            start=start,
            feed=chosen_feed,
            adjustment=Adjustment.ALL,
        )
        bars_dict = _sdk_get_stock_bars_soft_bounded(client, req, timeout=timeout)

        if not hasattr(bars_dict, "df") or bars_dict.df.empty:
            return pd.DataFrame()

        df = bars_dict.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol.upper().strip(), level="symbol")

        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        df.index.name = "datetime"
        validate_staleness = should_validate_staleness(purpose, lookback_days)
        df = _validate_intraday_data(df, symbol, interval, validate_staleness=validate_staleness, purpose=purpose)
        if not df.empty:
            feed_name = getattr(chosen_feed, "value", str(chosen_feed)).lower().strip()
            df.attrs["feed_identity"] = feed_name
            df.attrs["requested_feed"] = feed_name
            df.attrs["subscription_tier"] = tier
            if hasattr(df, "index") and len(df.index) > 0:
                last_idx = df.index[-1]
                int_sec = _parse_interval_seconds(interval)
                bar_end = last_idx + pd.Timedelta(seconds=int_sec)
                df.attrs["bar_start_utc"] = last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx)
                df.attrs["bar_end_utc"] = bar_end.isoformat() if hasattr(bar_end, "isoformat") else str(bar_end)
            _write_cache(key, df.copy())
        return df
    except Exception as e:
        logger.warning("Alpaca data fetch failed for %s (feed=%s): %s", symbol, feed, e)
        return pd.DataFrame()



def fetch_batch_intraday_alpaca(
    symbols: List[str],
    interval: str = None,
    lookback_days: int = None,
    prepost: bool = False,
    purpose: str = "ENTRY_SIGNAL",
    total_timeout_seconds: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch 5m bars for multiple symbols in a single vectorized Alpaca REST request."""
    if not symbols:
        return {}

    api_key, secret_key = _get_alpaca_credentials()
    if not api_key or not secret_key:
        return {}

    interval = interval or config.INTRADAY_INTERVAL
    lookback_days = lookback_days or config.INTRADAY_LOOKBACK_DAYS

    clean_symbols = list(dict.fromkeys(str(s).upper().strip() for s in symbols if str(s).strip()))
    if not clean_symbols:
        return {}

    start_mono = time.monotonic()
    def _rem():
        if total_timeout_seconds is None:
            return None
        return max(0.0, total_timeout_seconds - (time.monotonic() - start_mono))

    # Check cache first for any freshly cached symbols
    results: Dict[str, pd.DataFrame] = {}
    pending_symbols: List[str] = []
    validate_staleness = should_validate_staleness(purpose, lookback_days)

    for sym in clean_symbols:
        key = _cache_key(sym, interval, lookback_days, prepost, source="alpaca_api")
        cached_df = _read_cache(key)
        if cached_df is not None:
            validated_cache = _validate_intraday_data(
                cached_df.copy(),
                symbol=sym,
                interval=interval,
                validate_staleness=validate_staleness,
                purpose=purpose,
            )
            if not validated_cache.empty:
                results[sym] = validated_cache
                continue
        pending_symbols.append(sym)

    if not pending_symbols:
        is_breadth = str(purpose or "").upper() == "REGIME_BREADTH"
        telemetry = _breadth_telemetry if is_breadth else _feed_telemetry
        lock = _breadth_telemetry_lock if is_breadth else _feed_telemetry_lock
        with lock:
            telemetry.update(last_batch_duration_seconds=time.monotonic() - start_mono,
                             last_batch_timestamp=time.time(), last_batch_timeout_count=0,
                             last_batch_forming_count=0,
                             last_batch_reasons={s: "cached_valid" for s in results},
                             last_batch_requested=len(clean_symbols), last_batch_received=len(results),
                             last_batch_coverage=len(results) / len(clean_symbols),
                             consecutive_fetch_timeouts=0)
        return results


    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import Adjustment
        from datetime import timezone

        client = _get_alpaca_client()
        if client is None:
            return results

        start = datetime.now(timezone.utc) - timedelta(days=lookback_days or 1)
        tf = TimeFrame.Minute
        if interval == "5m":
            tf = TimeFrame(5, TimeFrameUnit.Minute)
        elif interval == "15m":
            tf = TimeFrame(15, TimeFrameUnit.Minute)
        elif interval == "1h":
            tf = TimeFrame.Hour

        chunk_size = max(1, int(getattr(config, "ALPACA_CHUNK_SIZE", 5)))
        chunk_delay_ms = max(0, int(getattr(config, "ALPACA_CHUNK_DELAY_MS", 200)))
        chunk_timeout_s = float(getattr(config, "ALPACA_CHUNK_TIMEOUT_SECONDS", 2.0))
        tier = str(getattr(config, "ALPACA_SUBSCRIPTION_TIER", "free")).lower().strip()
        is_free_tier = (tier == "free")

        from alpaca.data.enums import DataFeed
        if is_free_tier:
            primary_feed = DataFeed.IEX
            backup_feed = None
        else:
            primary_feed = _normalize_data_feed(getattr(config, "ALPACA_DATA_FEED", "iex"))
            backup_feed = DataFeed.SIP if primary_feed == DataFeed.IEX else DataFeed.IEX

        def _extract_symbol_df(raw_df: pd.DataFrame, sym: str) -> pd.DataFrame:
            if raw_df is None or raw_df.empty:
                return pd.DataFrame()
            is_multi = isinstance(raw_df.index, pd.MultiIndex)
            if is_multi:
                names = raw_df.index.names
                if "symbol" in names:
                    if sym not in raw_df.index.get_level_values("symbol").unique():
                        return pd.DataFrame()
                    sym_df = raw_df.xs(sym, level="symbol").copy()
                else:
                    if sym not in raw_df.index.levels[0].unique():
                        return pd.DataFrame()
                    sym_df = raw_df.xs(sym, level=0).copy()
            else:
                sym_df = raw_df.copy()

            if sym_df.empty:
                return pd.DataFrame()

            sym_df = sym_df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })
            sym_df.index.name = "datetime"
            return sym_df

        chunks = [pending_symbols[i : i + chunk_size] for i in range(0, len(pending_symbols), chunk_size)]
        num_chunks = len(chunks)

        is_breadth = str(purpose or "").upper() == "REGIME_BREADTH"
        target_telemetry = _breadth_telemetry if is_breadth else _feed_telemetry
        target_lock = _breadth_telemetry_lock if is_breadth else _feed_telemetry_lock

        symbol_reasons: Dict[str, str] = {}
        for sym in results:
            symbol_reasons[sym] = "cached_valid"

        for chunk_idx, chunk_symbols in enumerate(chunks):
            rem_t = _rem()
            if total_timeout_seconds is not None and (rem_t is None or rem_t <= 0.05):
                with target_lock:
                    target_telemetry["batch_timeout_count"] += 1
                for s in chunk_symbols:
                    symbol_reasons[s] = "budget_exhausted"
                for rem_chunk in chunks[chunk_idx + 1:]:
                    for s in rem_chunk:
                        symbol_reasons[s] = "budget_exhausted"
                logger.warning(
                    "[BATCH FETCH TIMEOUT] Total budget exhausted (%.2fs remaining) before chunk %d/%d.",
                    rem_t or 0.0, chunk_idx + 1, num_chunks,
                )
                break

            this_timeout = min(chunk_timeout_s, rem_t) if rem_t is not None else chunk_timeout_s

            try:
                # 1. Fetch Primary Feed for this chunk
                primary_bars = None
                primary_status = "SUCCESS"
                primary_err = None
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=chunk_symbols,
                        timeframe=tf,
                        start=start,
                        feed=primary_feed,
                        adjustment=Adjustment.ALL,
                    )
                    primary_bars = _sdk_get_stock_bars_soft_bounded(client, req, timeout=this_timeout)
                except MarketDataTransportTimeout as tex:
                    primary_status = "TIMEOUT"
                    primary_err = tex
                    feed_name = getattr(primary_feed, "value", str(primary_feed)).upper()
                    logger.warning("[BATCH FETCH TIMEOUT] Alpaca %s feed timeout for chunk %d/%d (%s): %s",
                                   feed_name, chunk_idx + 1, num_chunks, chunk_symbols, tex)
                except Exception as primary_err_other:
                    primary_status = "ERROR"
                    primary_err = primary_err_other
                    feed_name = getattr(primary_feed, "value", str(primary_feed)).upper()
                    logger.debug("[BATCH FETCH] Alpaca %s feed error for chunk %s (%s).", feed_name, chunk_symbols, primary_err_other)

                primary_raw_df = getattr(primary_bars, "df", None) if primary_bars is not None else None
                stale_or_missing_symbols: List[str] = []

                for sym in chunk_symbols:
                    key = _cache_key(sym, interval, lookback_days, prepost, source="alpaca_api")
                    sym_df = _extract_symbol_df(primary_raw_df, sym) if primary_raw_df is not None else pd.DataFrame()
                    if sym_df.empty:
                        if primary_status == "TIMEOUT":
                            symbol_reasons[sym] = "sdk_timeout"
                        elif primary_status == "ERROR":
                            symbol_reasons[sym] = "transport_error"
                        else:
                            symbol_reasons[sym] = "no_bars_returned"
                        stale_or_missing_symbols.append(sym)
                        continue

                    # Check whether candle is forming or completed
                    int_sec = _parse_interval_seconds(interval)
                    now_u = pd.Timestamp.now(tz="UTC")
                    has_completed = True
                    try:
                        norm_idx = _normalize_utc_index(sym_df.index)
                        closes = norm_idx + pd.Timedelta(seconds=int_sec)
                        has_completed = bool((closes <= now_u).any())
                    except Exception:
                        has_completed = True

                    validated_df = _validate_intraday_data(
                        sym_df, sym, interval, validate_staleness=validate_staleness, purpose=purpose
                    )
                    if not validated_df.empty:
                        feed_name = getattr(primary_feed, "value", str(primary_feed)).lower().strip()
                        validated_df.attrs["feed_identity"] = feed_name
                        validated_df.attrs["requested_feed"] = feed_name
                        validated_df.attrs["subscription_tier"] = tier
                        if hasattr(validated_df, "index") and len(validated_df.index) > 0:
                            last_idx = validated_df.index[-1]
                            bar_end = last_idx + pd.Timedelta(seconds=int_sec)
                            validated_df.attrs["bar_start_utc"] = last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx)
                            validated_df.attrs["bar_end_utc"] = bar_end.isoformat() if hasattr(bar_end, "isoformat") else str(bar_end)
                        _write_cache(key, validated_df.copy())
                        results[sym] = validated_df
                        symbol_reasons[sym] = "valid"
                    else:
                        if not has_completed:
                            symbol_reasons[sym] = "no_completed_bars_yet"
                        else:
                            symbol_reasons[sym] = "stale_bar"
                        stale_or_missing_symbols.append(sym)

                # 2. Fallback handling for stale or missing symbols
                if stale_or_missing_symbols:
                    if is_free_tier:
                        logger.debug("[BATCH FETCH] Free tier detected: SIP fallback bypassed.")
                    elif backup_feed is None or backup_feed == primary_feed:
                        logger.warning(
                            "[BATCH FETCH FALLBACK SKIPPED] status=FALLBACK_SOURCE_SAME_AS_PRIMARY. "
                            "Refusing duplicate query. Fail-closed: %d stale symbols excluded.",
                            len(stale_or_missing_symbols),
                        )
                    else:
                        rem_fb = _rem()
                        fb_timeout = min(chunk_timeout_s, rem_fb) if rem_fb is not None else chunk_timeout_s
                        backup_name = getattr(backup_feed, "value", str(backup_feed)).upper()
                        logger.info(
                            "[BATCH FETCH FALLBACK] Attempting %s feed for %d symbols: %s",
                            backup_name, len(stale_or_missing_symbols), stale_or_missing_symbols,
                        )
                        try:
                            backup_req = StockBarsRequest(
                                symbol_or_symbols=stale_or_missing_symbols,
                                timeframe=tf,
                                start=start,
                                feed=backup_feed,
                                adjustment=Adjustment.ALL,
                            )
                            backup_bars = _sdk_get_stock_bars_soft_bounded(client, backup_req, timeout=fb_timeout)
                            backup_raw_df = getattr(backup_bars, "df", None) if backup_bars is not None else None

                            for sym in stale_or_missing_symbols:
                                key = _cache_key(sym, interval, lookback_days, prepost, source="alpaca_api")
                                fb_df = _extract_symbol_df(backup_raw_df, sym) if backup_raw_df is not None else pd.DataFrame()
                                if fb_df.empty:
                                    logger.warning("[BATCH FETCH REJECTED] status=FALLBACK_UNAVAILABLE symbol=%s: No bars from %s feed.", sym, backup_name)
                                    continue

                                int_sec = _parse_interval_seconds(interval)
                                now_u = pd.Timestamp.now(tz="UTC")
                                has_completed = True
                                try:
                                    norm_idx = _normalize_utc_index(fb_df.index)
                                    closes = norm_idx + pd.Timedelta(seconds=int_sec)
                                    has_completed = bool((closes <= now_u).any())
                                except Exception:
                                    has_completed = True

                                fb_validated_df = _validate_intraday_data(
                                    fb_df, sym, interval, validate_staleness=validate_staleness, purpose=purpose
                                )
                                if not fb_validated_df.empty:
                                    logger.info("[BATCH FETCH FALLBACK ACCEPTED] status=FALLBACK_FRESH symbol=%s: Accepted from %s feed.", sym, backup_name)
                                    fb_feed_name = getattr(backup_feed, "value", str(backup_feed)).lower().strip()
                                    fb_validated_df.attrs["feed_identity"] = fb_feed_name
                                    fb_validated_df.attrs["requested_feed"] = fb_feed_name
                                    fb_validated_df.attrs["subscription_tier"] = tier
                                    if hasattr(fb_validated_df, "index") and len(fb_validated_df.index) > 0:
                                        last_idx = fb_validated_df.index[-1]
                                        bar_end = last_idx + pd.Timedelta(seconds=int_sec)
                                        fb_validated_df.attrs["bar_start_utc"] = last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx)
                                        fb_validated_df.attrs["bar_end_utc"] = bar_end.isoformat() if hasattr(bar_end, "isoformat") else str(bar_end)
                                    _write_cache(key, fb_validated_df.copy())
                                    results[sym] = fb_validated_df
                                    symbol_reasons[sym] = "valid_fallback"
                                else:
                                    if not has_completed:
                                        symbol_reasons[sym] = "no_completed_bars_yet"
                                    else:
                                        symbol_reasons[sym] = "stale_bar"
                                    logger.warning(
                                        "[BATCH FETCH REJECTED] status=FALLBACK_STALE symbol=%s: Fallback %s bars also stale/invalid. Strictly excluded.",
                                        sym, backup_name,
                                    )
                        except Exception as backup_err:
                            err_str = str(backup_err).lower()
                            if any(w in err_str for w in ("unauthorized", "forbidden", "403", "subscription", "not permit")):
                                fb_status = "FALLBACK_UNAUTHORIZED"
                            elif any(w in err_str for w in ("rate limit", "429", "too many requests")):
                                fb_status = "FALLBACK_RATE_LIMITED"
                            else:
                                fb_status = "FALLBACK_UNAVAILABLE"
                            logger.warning(
                                "[BATCH FETCH FALLBACK FAILED] status=%s feed=%s error=%s. Fail-closed: %d stale symbols excluded.",
                                fb_status, backup_name, backup_err, len(stale_or_missing_symbols),
                            )
            except Exception as chunk_exc:
                logger.warning(
                    "[BATCH FETCH CHUNK FAILED] Chunk %d/%d failed for symbols %s: %s",
                    chunk_idx + 1, num_chunks, chunk_symbols, chunk_exc,
                )

            # Delay consumes the same total budget; never sleep beyond it.
            if chunk_idx < num_chunks - 1 and chunk_delay_ms > 0:
                remaining = _rem()
                delay = chunk_delay_ms / 1000.0
                time.sleep(min(delay, remaining) if remaining is not None else delay)

        elapsed = time.monotonic() - start_mono

        sdk_timeout_count = sum(1 for r in symbol_reasons.values() if r in ("sdk_timeout", "budget_exhausted"))
        forming_count = sum(1 for r in symbol_reasons.values() if r == "no_completed_bars_yet")
        no_bars_count = sum(1 for r in symbol_reasons.values() if r == "no_bars_returned")
        stale_count = sum(1 for r in symbol_reasons.values() if r == "stale_bar")
        valid_count = len(results)

        with target_lock:
            target_telemetry["last_batch_duration_seconds"] = elapsed
            target_telemetry["last_batch_timestamp"] = time.time()
            target_telemetry["total_bars_fetched"] += len(results)
            target_telemetry["last_batch_timeout_count"] = sdk_timeout_count
            target_telemetry["last_batch_forming_count"] = forming_count
            target_telemetry["last_batch_reasons"] = dict(symbol_reasons)
            target_telemetry["last_batch_requested"] = len(clean_symbols)
            target_telemetry["last_batch_received"] = len(results)
            target_telemetry["last_batch_coverage"] = len(results) / len(clean_symbols)
            if len(results) > 0:
                target_telemetry["consecutive_fetch_timeouts"] = 0
            elif sdk_timeout_count > 0:
                target_telemetry["consecutive_fetch_timeouts"] = target_telemetry.get("consecutive_fetch_timeouts", 0) + 1

        coverage_pct = (len(results) / len(clean_symbols)) * 100.0 if clean_symbols else 100.0
        logger.info(
            f"[CHUNK FETCH DONE] Retrieved {len(results)}/{len(clean_symbols)} symbols "
            f"across {num_chunks} chunks in {elapsed:.2f}s (purpose={purpose} scope=rest_subset coverage={coverage_pct:.1f}% | "
            f"valid={valid_count} sdk_timeout={sdk_timeout_count} forming={forming_count} "
            f"no_bars={no_bars_count} stale={stale_count})"
        )
        min_coverage_pct = float(getattr(config, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95)) * 100.0
        if coverage_pct < min_coverage_pct:
            logger.warning(
                f"[CHUNK FETCH WARNING] Batch coverage {coverage_pct:.1f}% below minimum {min_coverage_pct:.1f}% "
                f"(valid={valid_count}, sdk_timeout={sdk_timeout_count}, forming={forming_count}, "
                f"no_bars={no_bars_count}, stale={stale_count})"
            )

        return results
    except Exception as e:
        logger.warning("[BATCH FETCH] Alpaca batch fetch failed: %s", e)
        return results


# Deprecated compatibility alias.
# Remove after all internal callers use fetch_batch_intraday_alpaca.
_fetch_bars_batch_alpaca = fetch_batch_intraday_alpaca



def _is_usable_intraday_data(df: Optional[pd.DataFrame], interval: str = "5m") -> bool:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False
    required_cols = {"open", "high", "low", "close", "volume"}
    if not required_cols.issubset(set(df.columns)):
        return False
    return len(df) > 0


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

    has_alpaca_keys = _has_alpaca_credentials()
    if bridge is not None and getattr(bridge, "is_connected", False):
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
            validate_staleness=should_validate_staleness(purpose, lookback_days),
            purpose=purpose,
        )
        if not validated_cache.empty:
            return validated_cache
        logger.warning(
            "[CACHE INVALIDATED] %s cached intraday data failed validation",
            symbol,
        )

    if bridge is None or not getattr(bridge, "is_connected", False):
        tier = str(getattr(config, "ALPACA_SUBSCRIPTION_TIER", "free")).lower().strip()
        if _has_alpaca_credentials():
            if tier == "free":
                providers = [
                    (DataProvider.ALPACA, "iex"),
                    (DataProvider.YFINANCE, None),
                ]
            else:
                cfg_feed = str(getattr(config, "ALPACA_DATA_FEED", "iex")).lower().strip()
                providers = [
                    (DataProvider.ALPACA, cfg_feed),
                    (DataProvider.YFINANCE, None),
                ]
        else:
            providers = [
                (DataProvider.YFINANCE, None),
            ]

        for provider, feed in providers:
            try:
                if provider == DataProvider.ALPACA:
                    df = fetch_intraday_alpaca(
                        symbol,
                        interval=interval,
                        lookback_days=lookback_days,
                        prepost=prepost,
                        purpose=purpose,
                        feed=feed,
                    )
                else:
                    df = fetch_intraday_yfinance(
                        symbol,
                        interval=interval,
                        lookback_days=lookback_days,
                        prepost=prepost,
                        purpose=purpose,
                    )
                if _is_usable_intraday_data(df, interval=interval):
                    return df
            except Exception as exc:
                logger.warning(
                    "Market-data attempt failed: symbol=%s provider=%s feed=%s purpose=%s error=%r",
                    symbol, getattr(provider, "value", provider), feed, purpose, exc,
                )
        return pd.DataFrame()

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
            validate_staleness=should_validate_staleness(purpose, lookback_days),
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
            logger.warning("No broker intraday data returned for %s after 3 attempts. Falling back to data providers.", symbol)
            if _has_alpaca_credentials():
                df = fetch_intraday_alpaca(symbol, interval=interval, lookback_days=lookback_days, prepost=prepost, purpose=purpose)
            else:
                df = fetch_intraday_yfinance(symbol, interval=interval, lookback_days=lookback_days, prepost=prepost, purpose=purpose)
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
        logger.exception("Failed to fetch broker data for %s: %s. Falling back to data providers.", symbol, e)
        df = pd.DataFrame()
        if _has_alpaca_credentials():
            df = fetch_intraday_alpaca(symbol, interval=interval, lookback_days=lookback_days, prepost=prepost, purpose=purpose)
        if df.empty:
            df = fetch_intraday_yfinance(symbol, interval=interval, lookback_days=lookback_days, prepost=prepost, purpose=purpose)
        if not df.empty:
            _write_cache(fallback_key, df.copy())
        return df



def fetch_intraday_batch(
    symbols: List[str],
    interval: str = None,
    lookback_days: int = None,
    prepost: bool = False,
    bridge = None,
    purpose: str = "ENTRY_SIGNAL",
    total_timeout_seconds: Optional[float] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Fetch 5m bars for multiple symbols concurrently in a single batch.

    Returns:
        (symbol_data_dict, telemetry_metadata)
    """
    t_start = time.perf_counter()
    interval = interval or config.INTRADAY_INTERVAL
    lookback_days = lookback_days or config.INTRADAY_LOOKBACK_DAYS

    # Denominator is the full, unique requested universe, not the retry subset.
    requested_symbols = list(dict.fromkeys(str(s).upper().strip() for s in symbols if str(s).strip()))
    valid_symbols = [s for s in requested_symbols if not is_quarantined(s)]
    total_requested = len(requested_symbols)
    is_breadth = str(purpose or "").upper() == "REGIME_BREADTH"
    def remaining_budget():
        if total_timeout_seconds is None:
            return None
        return max(0.0, total_timeout_seconds - (time.perf_counter() - t_start))
    results: Dict[str, pd.DataFrame] = {}

    # 1. If bridge is connected and supports batch historical data, use broker-native batch fetch
    if (not is_breadth and bridge is not None and getattr(bridge, "is_connected", False)
            and hasattr(bridge, "fetch_batch_historical_data")):
        try:
            bar_size = _interval_to_alpaca(interval)
            raw_batches = bridge.fetch_batch_historical_data(
                valid_symbols,
                durationStr=f"{lookback_days} D",
                barSizeSetting=bar_size,
                useRTH=not prepost
            )
            for sym, bars in raw_batches.items():
                if bars:
                    df = _bars_to_frame(bars)
                    df = df.rename(columns={"date": "datetime"})
                    df.set_index("datetime", inplace=True)
                    df = _validate_intraday_data(df, sym, interval, purpose=purpose)
                    if not df.empty:
                        key = _cache_key(sym, interval, lookback_days, prepost, source="alpaca")
                        _write_cache(key, df.copy())
                        results[sym] = df
        except Exception as b_exc:
            logger.warning("[BATCH FETCH] Bridge batch historical fetch error: %s. Falling back to REST client.", b_exc)

    # 2. Check for missing symbols and fetch via direct Alpaca REST client batch
    missing_from_bridge = [s for s in valid_symbols if s not in results or results[s].empty]
    rest_requested = len(missing_from_bridge)
    budget_left = remaining_budget()
    if missing_from_bridge and (budget_left is None or budget_left > 0):
        try:
            alpaca_batch = fetch_batch_intraday_alpaca(
                missing_from_bridge,
                interval=interval,
                lookback_days=lookback_days,
                prepost=prepost,
                purpose=purpose,
                total_timeout_seconds=budget_left,
            )
            results.update(alpaca_batch)
        except Exception as a_exc:
            logger.warning("[BATCH FETCH] Direct Alpaca batch fetch error: %s", a_exc)

    # 3. Check for any remaining missing symbols (e.g. non-Alpaca or offline fallback)
    still_missing = [s for s in valid_symbols if s not in results or results[s].empty]
    if still_missing and (bridge is None or not getattr(bridge, "is_connected", False)) and str(purpose or "").upper() != "REGIME_BREADTH":
        for msym in still_missing:
            try:
                ydf = fetch_intraday_yfinance(msym, interval, lookback_days, prepost, purpose=purpose)
                if not ydf.empty:
                    results[msym] = ydf
            except Exception as y_exc:
                logger.debug("[BATCH FETCH] yfinance fallback failed for %s: %s", msym, y_exc)

    t_end = time.perf_counter()
    duration_ms = (t_end - t_start) * 1000.0
    results = {sym: results[sym] for sym in requested_symbols
               if sym in results and results[sym] is not None and not results[sym].empty}
    received_count = len(results)
    coverage = (received_count / total_requested) if total_requested > 0 else 0.0

    telemetry = {
        "requested": total_requested,
        "received": received_count,
        "coverage": round(coverage, 4),
        "duration_ms": round(duration_ms, 2),
        "purpose": str(purpose or "ENTRY_SIGNAL").upper(),
        "rest_subset_requested": rest_requested,
        "quarantined": total_requested - len(valid_symbols),
        "missing_symbols": [sym for sym in requested_symbols if sym not in results],
        "budget_exhausted": remaining_budget() == 0.0,
    }
    logger.info(
        "[BATCH COVERAGE] purpose=%s universe_received=%d universe_requested=%d "
        "coverage=%.1f%% rest_subset_requested=%d quarantined=%d budget_exhausted=%s",
        telemetry["purpose"], received_count, total_requested, coverage * 100.0,
        rest_requested, telemetry["quarantined"], telemetry["budget_exhausted"],
    )

    return results, telemetry


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
            validate_staleness=should_validate_staleness(purpose, lookback_days),
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
        return pd.DataFrame()

    if df.empty:
        logger.debug(
            "yfinance EMPTY intraday %s (period=%s, interval=%s).",
            symbol, period, interval,
        )
        return pd.DataFrame()


    df = _normalize_columns(df)
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            logger.warning("Missing column %s for %s", col, symbol)
            return pd.DataFrame()

    df = df.dropna(subset=["close"])
    validate_staleness = should_validate_staleness(purpose, lookback_days)
    df = _validate_intraday_data(df, symbol, interval, validate_staleness=validate_staleness, purpose=purpose)
    if not df.empty:
        _write_cache(key, df.copy())
    logger.debug(
        "Fetched %d bars for %s (%s, prepost=%s)",
        len(df), symbol, interval, prepost,
    )
    return df


def fetch_daily_alpaca(symbol: str, lookback_days: int = None) -> pd.DataFrame:
    api_key, secret_key = _get_alpaca_credentials()
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
        from alpaca.data.enums import Adjustment, DataFeed
        from datetime import timezone

        client = _get_alpaca_client()
        if client is None:
            client = StockHistoricalDataClient(api_key, secret_key)
            configure_alpaca_data_client_timeouts(client)
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        bars_dict = None
        tier = str(getattr(config, "ALPACA_SUBSCRIPTION_TIER", "free")).lower().strip()
        feed_enum = DataFeed.IEX if tier == "free" else _normalize_data_feed(getattr(config, "ALPACA_DATA_FEED", "iex"))
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol.upper().strip(),
                timeframe=TimeFrame.Day,
                start=start,
                feed=feed_enum,
                adjustment=Adjustment.ALL,
            )
            bars_dict = client.get_stock_bars(req)
        except Exception as iex_exc:
            if tier != "free":
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=symbol.upper().strip(),
                        timeframe=TimeFrame.Day,
                        start=start,
                        feed=DataFeed.SIP,
                        adjustment=Adjustment.ALL,
                    )
                    bars_dict = client.get_stock_bars(req)
                except Exception:
                    return pd.DataFrame()
            else:
                logger.debug("Alpaca IEX daily bars fetch failed for %s on free tier: %s", symbol, iex_exc)
                return pd.DataFrame()
        if not hasattr(bars_dict, "df") or bars_dict.df.empty:
            return pd.DataFrame()

        df = bars_dict.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol.upper().strip(), level="symbol")

        df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        df.index.name = "datetime"
        df.attrs["volume_source"] = "alpaca_iex"
        df.attrs["feed_identity"] = "iex" if tier == "free" else getattr(feed_enum, "value", str(feed_enum)).lower()
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
    files = glob.glob(os.path.join(_get_cache_dir(), "*.pkl"))
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
            close_tolerance_seconds=0.0,
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

    if "atr" in features.columns and len(features) > 0:
        last_val = features["atr"].iloc[-1]
        last_ts = features.index[-1]
        if not pd.isna(last_val) and last_val > 0:
            update_position_atr_cache(symbol, interval, float(last_val), last_ts)

    return features

