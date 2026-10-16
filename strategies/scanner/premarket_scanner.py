"""
premarket_scanner.py -- Gap, volume, and open-session momentum scanner.

Discovers candidates by scanning for:
  1. Pre-market gap/volume before the open
  2. Regular-session movement/volume after the open
  3. Price within configured band
  4. Relative volume or session activity

This is DISCOVERY ONLY -- it never places orders.
"""
import csv
import logging
import os
import threading
import time as _time
from dataclasses import dataclass, asdict
from typing import List, Optional

import pandas as pd

import config
from strategies.intraday_data import fetch_intraday, fetch_daily, previous_close
from strategies.session import is_market_open, session_status

logger = logging.getLogger(__name__)


@dataclass
class ScanCandidate:
    symbol: str
    prev_close: float
    current_price: float
    gap_pct: float
    premarket_volume: int
    relative_volume: float
    score: float
    reason: str
    sentiment_score: float = 0.0
    catalyst_type: Optional[str] = None
    top_headline: str = ""


from datetime import date


def is_earnings_near(symbol: str, within_days: int = 1) -> bool:
    """
    Returns True if symbol has an earnings release within `within_days` (default 1 day).
    Blocks pre-earnings volatility traps.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        cal = getattr(ticker, "calendar", None)
        if cal is not None:
            edates = None
            if isinstance(cal, dict) and "Earnings Date" in cal:
                edates = cal["Earnings Date"]
            elif hasattr(cal, "get") and cal.get("Earnings Date") is not None:
                edates = cal.get("Earnings Date")
            
            if edates is not None:
                if hasattr(edates, "iloc") and len(edates) > 0:
                    edate = edates.iloc[0]
                    if hasattr(edate, "date"):
                        edate = edate.date()
                    days_to = (edate - date.today()).days
                    return 0 <= days_to <= within_days
                elif isinstance(edates, (list, tuple)) and len(edates) > 0:
                    edate = edates[0]
                    if hasattr(edate, "date"):
                        edate = edate.date()
                    days_to = (edate - date.today()).days
                    return 0 <= days_to <= within_days
    except Exception as e:
        logger.debug("is_earnings_near(%s): %s", symbol, e)
    return False


from collections import defaultdict
import json
from pathlib import Path

def get_news_fail_persist_path() -> Path:
    """Dynamically resolve news fail counts path against config.DATA_DIR."""
    try:
        import config
        return Path(config.DATA_DIR) / "news_fail_counts.json"
    except Exception:
        return Path("data/news_fail_counts.json")

NEWS_FAIL_PERSIST_PATH = get_news_fail_persist_path()


def _load_news_fail_counts() -> defaultdict:
    """Loads news fail counts from disk if date matches today."""
    path = get_news_fail_persist_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                saved_date = data.get("date", "")
                today_str = date.today().strftime("%Y-%m-%d")
                if saved_date == today_str:
                    counts = defaultdict(int, data.get("counts", {}))
                    logger.info("[NEWS PERSIST] Restored %d news fail count(s) for today (%s)", len(counts), today_str)
                    return counts
        except Exception as e:
            logger.debug("[NEWS PERSIST] Load failed: %s", e)
    return defaultdict(int)


def _save_news_fail_counts():
    """Saves news fail counts to disk."""
    try:
        path = get_news_fail_persist_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "date": date.today().strftime("%Y-%m-%d"),
            "counts": dict(_news_fail_counts)
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.debug("[NEWS PERSIST] Save failed: %s", e)


_news_fail_counts = _load_news_fail_counts()
NEWS_FETCH_FAIL_PENALTY = -0.10
NEWS_FETCH_DROP_THRESHOLD = 2


def apply_news_fetch_hygiene(symbol: str, candidate: Optional[ScanCandidate]) -> Optional[ScanCandidate]:
    """
    Applies score penalty (-0.10) if news fetch fails, and drops candidate if news fails >= 2 times.
    """
    if candidate is None or not getattr(config, "NEWS_HYGIENE_ENABLED", True):
        return candidate
    sym = symbol.upper().strip()
    _news_fail_counts[sym] += 1
    fail_count = _news_fail_counts[sym]
    _save_news_fail_counts()

    logger.warning("[NEWS HYGIENE] %s: News fetch failed (fail #%d/%d)", sym, fail_count, NEWS_FETCH_DROP_THRESHOLD)
    if fail_count >= NEWS_FETCH_DROP_THRESHOLD:
        logger.warning("[NEWS HYGIENE DROP] %s: Removed from candidates after %d news fetch failures.", sym, fail_count)
        return None

    candidate.score = round(candidate.score + NEWS_FETCH_FAIL_PENALTY, 4)
    logger.info("[NEWS HYGIENE PENALTY] %s: Score adjusted to %.2f", sym, candidate.score)
    return candidate


def reset_news_fail_counts():
    """Resets news fail counts per session."""
    global _news_fail_counts
    _news_fail_counts = defaultdict(int)
    _save_news_fail_counts()
    logger.info("[SCANNER] News fail counts reset for new session.")


# ── Data Validation Quarantine Cache ──────────────────────────────────────
# Symbols that fail data validation are quarantined for QUARANTINE_DURATION_S
# to prevent redundant API calls and repeated WARNING log spam.
_quarantine_cache: dict = {}           # {symbol: expiry_epoch_float}
_quarantine_lock = threading.Lock()
QUARANTINE_DURATION_S: float = 1800.0  # 30 minutes


def quarantine_symbol(symbol: str, reason: str = "", now: Optional[float] = None) -> None:
    """Mark a symbol as quarantined for QUARANTINE_DURATION_S seconds."""
    current_time = now if now is not None else _time.time()
    expiry = current_time + QUARANTINE_DURATION_S
    with _quarantine_lock:
        _quarantine_cache[symbol] = expiry
    logger.debug(
        "[QUARANTINE] %s quarantined for %.0f min. Reason: %s",
        symbol, QUARANTINE_DURATION_S / 60, reason or "data_validation_failure",
    )


def is_quarantined(symbol: str, now: Optional[float] = None) -> bool:
    """Returns True if symbol is currently quarantined (not yet expired)."""
    current_time = now if now is not None else _time.time()
    with _quarantine_lock:
        expiry = _quarantine_cache.get(symbol)
        if expiry is None:
            return False
        if current_time < expiry:
            return True
        # Expired — auto-evict
        del _quarantine_cache[symbol]
        logger.info("[QUARANTINE] %s released (TTL expired)", symbol)
        return False


def clear_quarantine() -> None:
    """Clears all quarantine entries (call at session start)."""
    with _quarantine_lock:
        _quarantine_cache.clear()
    logger.info("[QUARANTINE] Cache cleared.")


# ── Universe Liquidity & Dollar-Volume Gate ────────────────────────────────
MIN_DOLLAR_VOLUME: float = 1_000_000.0     # $1M daily turnover floor
MIN_SHARE_VOLUME: float  = 500_000.0
MIN_PRICE: float         = 3.00            # kill sub-$3 penny names

# Watchlist symbols exempt from the liquidity gate.
# IBIT/ETHA/BITO are ETFs, MARA is a large-cap miner — all liquid, but we
# never want the filter to accidentally drop a core inverse/crypto name.
LIQUIDITY_EXEMPT = {"SH", "SQQQ", "SOXS", "IBIT", "ETHA", "BITO", "MARA", "PSQ", "SPDN"}


def passes_liquidity_gate(symbol: str, bar: Optional[Any] = None) -> tuple[bool, str]:
    """Reject illiquid symbols using prior bar data before quote/API pull."""
    sym = symbol.upper().strip()
    if sym in LIQUIDITY_EXEMPT or sym in getattr(config, "INVERSE_ETF_SYMBOLS", set()):
        return (True, "EXEMPT")
    if bar is None:
        return (False, "NO_BAR")

    price = getattr(bar, "close", None)
    volume = getattr(bar, "volume", None)
    if price is None or volume is None:
        return (False, "NO_BAR")

    try:
        price_f = float(price)
        volume_f = float(volume)
    except (ValueError, TypeError):
        return (False, "INVALID_BAR")

    if price_f < MIN_PRICE:
        return (False, f"PRICE<{MIN_PRICE}")
    if volume_f < MIN_SHARE_VOLUME:
        return (False, f"SHVOL<{MIN_SHARE_VOLUME}")
    if price_f * volume_f < MIN_DOLLAR_VOLUME:
        return (False, f"$VOL<{MIN_DOLLAR_VOLUME}")

    return (True, "OK")


def build_universe(candidates: List[str], prev_day_bars: Optional[dict] = None) -> List[str]:
    """Filter candidate universe by liquidity gate while preserving exempt symbols."""
    clean = []
    dropped = []
    prev_bars = prev_day_bars or {}
    for sym in candidates:
        bar = prev_bars.get(sym)
        ok, reason = passes_liquidity_gate(sym, bar)
        if ok:
            clean.append(sym)
        else:
            dropped.append((sym, reason))

    logger.info("[UNIVERSE] kept=%d dropped=%d", len(clean), len(dropped))
    if dropped:
        logger.debug("[UNIVERSE] dropped: %s", dropped[:20])
    return clean
# ─────────────────────────────────────────────────────────────────────────


_screener_cache = {"data": None, "fetched_at": 0}
_cache_lock = threading.Lock()

def _load_universe() -> List[str]:
    if getattr(config, "DYNAMIC_SCREENER_ENABLED", False):
        with _cache_lock:
            now = _time.time()
            if _screener_cache["data"] is not None and (now - _screener_cache["fetched_at"]) < getattr(config, "DYNAMIC_SCREENER_CACHE_TTL", 120):
                return _screener_cache["data"]

            api_key = os.environ.get("APCA_API_KEY_ID", "")
            secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
            
            try:
                from alpaca.data.historical import ScreenerClient
                from alpaca.data.requests import MostActivesRequest, MarketMoversRequest
                from alpaca.data.enums import MostActivesBy, MarketType

                screener = ScreenerClient(api_key, secret_key)
                symbols_pool = set()
                
                # Fetch Most Actives
                actives = screener.get_most_actives(
                    MostActivesRequest(by=MostActivesBy.VOLUME, top=50, market_type=MarketType.STOCKS)
                )
                def _is_valid_equity(sym: str) -> bool:
                    if not sym or not isinstance(sym, str):
                        return False
                    sym = sym.upper().strip()
                    if "." in sym or "-" in sym or "/" in sym or "+" in sym:
                        return False
                    # Exclude 4 or 5-letter warrant/preferred share class suffixes (e.g. HUBCZ, CDIOW, TSIB)
                    known_valid = set(getattr(config, "INVERSE_ETF_SYMBOLS", set())) | set(getattr(config, "LARGE_CAP_PIVOT_SYMBOLS", [])) | {"BAC", "UBER", "META", "AMZN", "AAPL", "MSFT", "GOOGL", "NVDA", "PLTR", "AVGO", "JPM", "NFLX", "COST", "XOM", "UNH", "QQQ", "SPY", "SDOW", "SPXU"}
                    if len(sym) >= 4 and sym.endswith(("Z", "W", "R", "U", "C", "B")) and sym not in known_valid:
                        return False
                    if len(sym) > 4 and sym.endswith(("WS", "RT", "WW", "WT")):
                        return False
                    return True

                for r in actives.most_actives:
                    if _is_valid_equity(r.symbol):
                        symbols_pool.add(r.symbol)
                
                # Fetch Market Movers (Gainers/Losers)
                gainers = screener.get_market_movers(
                    MarketMoversRequest(top=50, market_type=MarketType.STOCKS)
                )
                for r in gainers.gainers:
                    if _is_valid_equity(r.symbol):
                        symbols_pool.add(r.symbol)
                for r in gainers.losers:
                    if _is_valid_equity(r.symbol):
                        symbols_pool.add(r.symbol)

                # Seed symbols_pool with Large-Cap / Mid-Cap Pivot Leaders
                pivot_leaders = getattr(config, "LARGE_CAP_PIVOT_SYMBOLS", ["PLTR", "NVDA", "AMD", "TSLA", "MSFT", "AMZN", "META", "MUU", "FIGS", "MARA"])
                for leader in pivot_leaders:
                    if _is_valid_equity(leader):
                        symbols_pool.add(leader)
                    
                denylist = set(getattr(config, "SYMBOL_DENYLIST", []))
                final_symbols = [s for s in symbols_pool if s not in denylist]

                # Guarantee inverse ETFs are always in the final universe
                if getattr(config, "INVERSE_ETF_SCANNER_EXEMPT", False):
                    for etf in getattr(config, "INVERSE_ETF_SYMBOLS", set()):
                        if etf not in denylist and etf not in final_symbols:
                            final_symbols.append(etf)

                if final_symbols:
                    logger.info("Fetched dynamic symbol universe from Alpaca (%d unique symbols)", len(final_symbols))
                    _screener_cache["data"] = final_symbols
                    _screener_cache["fetched_at"] = now
                    return _inject_inverse_seeds(final_symbols)
                else:
                    logger.warning("Dynamic screener returned empty lists (e.g. premarket stale). Falling back.")
            except Exception as e:
                logger.warning(f"Dynamic screener failed: {e}. Falling back to SYMBOL_UNIVERSE_FILE.")

    # 2nd -> static watchlist & core symbols fallback (prevents yfinance spam on delisted symbols)
    denylist = set(getattr(config, "SYMBOL_DENYLIST", []))
    core_syms = getattr(config, "FULL_MARKET_CORE_SYMBOLS", []) or getattr(config, "WATCHLIST", [])
    if core_syms:
        logger.info("Falling back to core symbols/watchlist universe (%d symbols)", len(core_syms))
        return _inject_inverse_seeds([s for s in list(dict.fromkeys(core_syms)) if s not in denylist])

    # 3rd -> SYMBOL_UNIVERSE_FILE (CSV)
    path = config.SYMBOL_UNIVERSE_FILE
    if path.exists():
        df = pd.read_csv(path)
        if "symbol" in df.columns:
            if "avg_volume" in df.columns:
                df = df[df["avg_volume"] >= 100_000]
            if "close" in df.columns:
                df = df[df["close"] >= 5.0]
            syms = df["symbol"].dropna().astype(str).str.upper().tolist()
            if syms:
                return _inject_inverse_seeds(syms[:100])

    return _inject_inverse_seeds(["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMZN", "AMD", "TSLA"])


def _inject_inverse_seeds(symbols: List[str]) -> List[str]:
    """Ensures Inverse Bear ETFs are always included in the universe regardless of screener source."""
    denylist = set(getattr(config, "SYMBOL_DENYLIST", []))
    res = [s for s in list(dict.fromkeys(symbols)) if s not in denylist]
    if getattr(config, "INVERSE_ETF_SCANNER_EXEMPT", False):
        for etf in getattr(config, "INVERSE_ETF_SYMBOLS", set()):
            if etf not in denylist and etf not in res:
                res.append(etf)
    return res


def _as_et_index(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.warning("Time filter skipped for %s: index is not DatetimeIndex", symbol)
        return pd.DataFrame()

    df_et = df.copy()
    try:
        if df_et.index.tz is None:
            df_et.index = df_et.index.tz_localize(config.TIMEZONE)
        else:
            df_et.index = df_et.index.tz_convert(config.TIMEZONE)
    except Exception:
        logger.warning("Time conversion failed for %s", symbol, exc_info=True)
        return pd.DataFrame()
    return df_et


def _premarket_bars(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """Return 04:00-09:30 ET bars for the latest session date from an intraday DataFrame."""
    df_et = _as_et_index(df, symbol)
    if df_et.empty:
        return pd.DataFrame()
    latest_date = df_et.index.date[-1]
    df_latest = df_et.loc[df_et.index.date == latest_date]
    pm_open = config.PREMARKET_START
    market_open = config.MARKET_OPEN
    mask = [pm_open <= ts.time() < market_open for ts in df_latest.index]
    return df_latest.loc[mask]


def _regular_session_bars(df: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """Return 09:30-16:00 ET bars for the latest session date from an intraday DataFrame."""
    df_et = _as_et_index(df, symbol)
    if df_et.empty:
        return pd.DataFrame()
    latest_date = df_et.index.date[-1]
    df_latest = df_et.loc[df_et.index.date == latest_date]
    mask = [config.MARKET_OPEN <= ts.time() <= config.MARKET_CLOSE for ts in df_latest.index]
    return df_latest.loc[mask]


def _relative_volume(symbol: str, volume: int) -> float:
    daily = fetch_daily(symbol, lookback_days=30)
    if daily.empty or len(daily) < config.VOLUME_MA_PERIOD:
        logger.debug(
            "Insufficient daily volume history for %s; using neutral relative volume",
            symbol,
        )
        return 1.0

    avg_vol = float(daily["volume"].tail(config.VOLUME_MA_PERIOD).mean())
    if avg_vol <= 0:
        return 1.0
    return volume / avg_vol


def _score(gap_pct: float, relative_volume: float, volume: int, sentiment_score: float = 0.0) -> float:
    gap_weight = getattr(config, "SCAN_SCORE_GAP_WEIGHT", 0.4)
    rvol_weight = getattr(config, "SCAN_SCORE_RVOL_WEIGHT", 0.3)
    vol_weight = getattr(config, "SCAN_SCORE_VOL_WEIGHT", 0.3)
    sentiment_weight = getattr(config, "SCAN_SCORE_SENTIMENT_WEIGHT", 0.2)
    rvol_cap = getattr(config, "SCAN_SCORE_RVOL_CAP", 10.0)
    vol_normalizer = getattr(config, "SCAN_SCORE_VOL_NORMALIZER", 100_000)

    base_score = (
        abs(gap_pct) * gap_weight
        + min(relative_volume, rvol_cap) * rvol_weight
        + min(volume / vol_normalizer, rvol_cap) * vol_weight
    )
    
    sentiment_bonus = sentiment_score * sentiment_weight * 10
    return base_score + max(sentiment_bonus, 0)


def _average_daily_volume(symbol: str) -> float:
    daily = fetch_daily(symbol, lookback_days=30)
    if daily.empty or len(daily) < config.VOLUME_MA_PERIOD:
        return 0.0
    return float(daily["volume"].tail(config.VOLUME_MA_PERIOD).mean())


def _record_rejection(reason: str, symbol: str, reject_dict: dict = None, reject_lock: threading.Lock = None):
    logger.debug("SCAN REJECT [%s]: %s", symbol, reason)
    if reject_dict is not None and reject_lock is not None:
        with reject_lock:
            reject_dict[reason] = reject_dict.get(reason, 0) + 1


def _passes_spread_filter(
    symbol: str,
    is_premarket: bool = True,
    quote_dict: dict = None,
    reject_dict: dict = None,
    reject_lock: threading.Lock = None,
) -> bool:
    max_spread = getattr(config, "SCAN_PREMARKET_MAX_SPREAD_PCT", 0.015) if is_premarket else getattr(config, "SCAN_REGULAR_MAX_SPREAD_PCT", 0.005)
    if not quote_dict or symbol not in quote_dict:
        return True
    q = quote_dict[symbol]
    bid = float(getattr(q, "bid_price", 0) or 0)
    ask = float(getattr(q, "ask_price", 0) or 0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else (ask or bid or 0.0)
    if mid <= 0:
        return True
    spread_pct = (ask - bid) / mid
    if spread_pct > max_spread:
        _record_rejection(f"spread_exceeds_max ({spread_pct:.4f} > {max_spread:.4f})", symbol, reject_dict, reject_lock)
        return False
    return True


def _passes_premarket_filters(
    symbol: str,
    gap_pct: float,
    current_price: float,
    volume: int,
    relative_volume: float,
    quote_dict: dict = None,
    reject_dict: dict = None,
    reject_lock: threading.Lock = None,
) -> bool:
    allow_short = getattr(config, "ALLOW_SHORT", False)
    if gap_pct < 0 and not allow_short:
        _record_rejection("gap_down (ALLOW_SHORT=False)", symbol, reject_dict, reject_lock)
        return False

    if current_price < config.SCAN_MIN_PRICE or current_price > config.SCAN_MAX_PRICE:
        _record_rejection(f"price_outside_scan_range (${current_price:.2f})", symbol, reject_dict, reject_lock)
        return False
    if abs(gap_pct) < config.SCAN_MIN_GAP_PCT:
        _record_rejection(f"gap_below_min ({gap_pct:.2f}% < {config.SCAN_MIN_GAP_PCT}%)", symbol, reject_dict, reject_lock)
        return False
    if abs(gap_pct) > config.SCAN_MAX_GAP_PCT:
        _record_rejection(f"gap_above_max ({gap_pct:.2f}% > {config.SCAN_MAX_GAP_PCT}%)", symbol, reject_dict, reject_lock)
        return False
    if volume < config.SCAN_MIN_PREMARKET_VOLUME:
        _record_rejection(f"volume_below_min ({volume} < {config.SCAN_MIN_PREMARKET_VOLUME})", symbol, reject_dict, reject_lock)
        return False
    if relative_volume < config.SCAN_MIN_RELATIVE_VOLUME:
        _record_rejection(f"rvol_below_min ({relative_volume:.2f} < {config.SCAN_MIN_RELATIVE_VOLUME})", symbol, reject_dict, reject_lock)
        return False

    min_avg_vol = getattr(config, "SCAN_MIN_AVG_DAILY_VOLUME", 750_000)
    avg_vol = _average_daily_volume(symbol)
    if avg_vol > 0 and avg_vol < min_avg_vol:
        _record_rejection(f"avg_daily_vol_below_min ({avg_vol:.0f} < {min_avg_vol})", symbol, reject_dict, reject_lock)
        return False

    if not _passes_spread_filter(symbol, is_premarket=True, quote_dict=quote_dict, reject_dict=reject_dict, reject_lock=reject_lock):
        return False

    return True


def _passes_open_filters(
    symbol: str,
    move_pct: float,
    current_price: float,
    volume: int,
    relative_volume: float,
    quote_dict: dict = None,
    reject_dict: dict = None,
    reject_lock: threading.Lock = None,
) -> bool:
    # ── Inverse ETF exemption check ───────────────────────────────────
    inverse_etfs = set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
    is_inverse = symbol.upper().strip() in inverse_etfs
    scanner_exempt = getattr(config, "INVERSE_ETF_SCANNER_EXEMPT", False)
    # ─────────────────────────────────────────────────────────────────

    allow_short = getattr(config, "ALLOW_SHORT", False)
    if move_pct < 0 and not allow_short and not is_inverse:
        _record_rejection("down_move (ALLOW_SHORT=False)", symbol, reject_dict, reject_lock)
        return False

    min_price = getattr(config, "SCAN_MIN_PRICE", 10.0)
    if current_price < min_price or current_price > config.SCAN_MAX_PRICE:
        _record_rejection(f"price_outside_scan_range (${current_price:.2f} < ${min_price:.2f})", symbol, reject_dict, reject_lock)
        return False
    min_move = getattr(config, "SCAN_OPEN_MIN_MOVE_PCT", 0.25)
    if abs(move_pct) < min_move:
        _record_rejection(f"open_move_too_small ({move_pct:.2f}% < {min_move}%)", symbol, reject_dict, reject_lock)
        return False
    min_vol = getattr(config, "SCAN_OPEN_MIN_VOLUME", 100_000)
    if volume < min_vol:
        _record_rejection(f"volume_too_small ({volume} < {min_vol})", symbol, reject_dict, reject_lock)
        return False

    # ── Volume thresholds: relaxed for inverse ETFs ───────────────────
    if is_inverse and scanner_exempt:
        min_rvol = getattr(config, "INVERSE_ETF_MIN_RVOL", 0.30)
        min_avg_vol = getattr(config, "INVERSE_ETF_MIN_AVG_DAILY_VOLUME", 200_000)
    else:
        min_rvol = getattr(config, "SCAN_OPEN_MIN_RELATIVE_VOLUME", 0.05)
        min_avg_vol = getattr(config, "SCAN_MIN_AVG_DAILY_VOLUME", 750_000)
    # ─────────────────────────────────────────────────────────────────

    if relative_volume < min_rvol:
        _record_rejection(f"rvol_too_small ({relative_volume:.2f} < {min_rvol})", symbol, reject_dict, reject_lock)
        return False

    avg_vol = _average_daily_volume(symbol)
    if avg_vol > 0 and avg_vol < min_avg_vol:
        _record_rejection(f"avg_daily_vol_too_small ({avg_vol:.0f} < {min_avg_vol})", symbol, reject_dict, reject_lock)
        return False

    if not _passes_spread_filter(symbol, is_premarket=False, quote_dict=quote_dict, reject_dict=reject_dict, reject_lock=reject_lock):
        return False

    return True


def _compute_premarket_candidate(
    symbol: str,
    quote_dict: dict = None,
    reject_dict: dict = None,
    reject_lock: threading.Lock = None,
) -> Optional[ScanCandidate]:
    prev = previous_close(symbol)
    if prev is None or prev <= 0:
        _record_rejection("invalid_prev_close", symbol, reject_dict, reject_lock)
        return None

    # ── Early price gate: skip API fetch entirely for sub-threshold stocks ──
    early_price_floor = getattr(config, "SCAN_MIN_PRICE", 5.0) * 0.70
    if prev < early_price_floor:
        _record_rejection(
            f"price_gate_early_exit (prev_close ${prev:.2f} < ${early_price_floor:.2f})",
            symbol, reject_dict, reject_lock,
        )
        return None
    # ────────────────────────────────────────────────────────────────────────

    df = fetch_intraday(
        symbol,
        interval=config.PREMARKET_INTERVAL,
        lookback_days=1,
        prepost=True,
    )
    if df.empty:
        _record_rejection("empty_intraday_data", symbol, reject_dict, reject_lock)
        return None

    pm_df = _premarket_bars(df, symbol)
    if pm_df.empty:
        _record_rejection("no_premarket_bars", symbol, reject_dict, reject_lock)
        return None

    current = float(pm_df["close"].iloc[-1])
    volume = int(pm_df["volume"].sum())
    gap_pct = ((current - prev) / prev) * 100.0
    rel_vol = _relative_volume(symbol, volume)

    if not _passes_premarket_filters(
        symbol, gap_pct, current, volume, rel_vol, quote_dict=quote_dict, reject_dict=reject_dict, reject_lock=reject_lock
    ):
        return None

    score = _score(gap_pct, rel_vol, volume)
    direction = "GAP_UP" if gap_pct > 0 else "GAP_DOWN"
    reason = f"{direction} {gap_pct:+.1f}% | PM vol {volume:,} | rvol {rel_vol:.2f}x"

    return ScanCandidate(symbol, prev, current, gap_pct, volume, rel_vol, score, reason)


def _compute_open_candidate(
    symbol: str,
    quote_dict: dict = None,
    reject_dict: dict = None,
    reject_lock: threading.Lock = None,
) -> Optional[ScanCandidate]:
    prev = previous_close(symbol)
    if prev is None or prev <= 0:
        _record_rejection("invalid_prev_close", symbol, reject_dict, reject_lock)
        return None

    # ── Early price gate: skip API fetch entirely for sub-threshold stocks ──
    early_price_floor = getattr(config, "SCAN_MIN_PRICE", 5.0) * 0.70
    if prev < early_price_floor:
        _record_rejection(
            f"price_gate_early_exit (prev_close ${prev:.2f} < ${early_price_floor:.2f})",
            symbol, reject_dict, reject_lock,
        )
        return None
    # ────────────────────────────────────────────────────────────────────────

    df = fetch_intraday(
        symbol,
        interval=config.INTRADAY_INTERVAL,
        lookback_days=1,
    )
    if df.empty:
        _record_rejection("empty_intraday_data", symbol, reject_dict, reject_lock)
        return None

    reg_df = _regular_session_bars(df, symbol)
    if reg_df.empty:
        _record_rejection("no_regular_session_bars", symbol, reject_dict, reject_lock)
        return None

    current = float(reg_df["close"].iloc[-1])
    open_price = float(reg_df["open"].iloc[0])
    volume = int(reg_df["volume"].sum())
    move_pct = ((current - open_price) / open_price) * 100.0 if open_price > 0 else 0.0
    gap_pct = ((current - prev) / prev) * 100.0
    rel_vol = _relative_volume(symbol, volume)

    if not _passes_open_filters(
        symbol, move_pct, current, volume, rel_vol, quote_dict=quote_dict, reject_dict=reject_dict, reject_lock=reject_lock
    ):
        return None

    score = _score(move_pct, rel_vol, volume)
    direction = "OPEN_UP" if move_pct > 0 else "OPEN_DOWN"
    reason = (
        f"{direction} {move_pct:+.2f}% | vs prev {gap_pct:+.2f}% | "
        f"session vol {volume:,} | rvol {rel_vol:.2f}x"
    )

    return ScanCandidate(symbol, prev, current, gap_pct, volume, rel_vol, score, reason)


def scan(symbols: List[str] = None, max_candidates: int = None) -> List[ScanCandidate]:
    from strategies.scanner.news_sentiment import analyze_news_sentiment
    symbols = symbols or _load_universe()
    blacklist = set(getattr(config, "SYMBOL_STALE_QUOTE_BLACKLIST", []))
    denylist = set(getattr(config, "SYMBOL_DENYLIST", []))
    if blacklist or denylist:
        symbols = [s for s in symbols if s not in blacklist and s not in denylist]
    max_candidates = max_candidates or config.SCAN_MAX_CANDIDATES
    mode = "OPEN" if is_market_open() else "PREMARKET"

    quote_dict = {}
    try:
        api_key = os.environ.get("APCA_API_KEY_ID", "")
        secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
        if api_key and secret_key:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            from strategies.intraday_data import configure_alpaca_data_client_timeouts
            data_client = StockHistoricalDataClient(api_key, secret_key)
            configure_alpaca_data_client_timeouts(data_client)
            # Batch fetch quotes for chunk
            quote_dict = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbols[:100], feed="iex"))
    except Exception as exc:
        logger.debug("Failed batch fetching quotes for scan: %s", exc)

    candidates = []
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    reject_dict = {}
    reject_lock = threading.Lock()

    def _eval_sym(sym: str) -> Optional[ScanCandidate]:
        symbol = str(sym).upper()

        # ── Quarantine check ──
        if is_quarantined(symbol):
            with reject_lock:
                reject_dict["quarantined_data_failure"] = reject_dict.get("quarantined_data_failure", 0) + 1
            return None
        # ─────────────────────

        if is_earnings_near(symbol, within_days=getattr(config, "EARNINGS_BLOCK_DAYS", 1)):
            with reject_lock:
                reject_dict["earnings_within_1_day"] = reject_dict.get("earnings_within_1_day", 0) + 1
            logger.info("[EARNINGS BLOCK] Skipped %s: Earnings within %d day(s)", symbol, getattr(config, "EARNINGS_BLOCK_DAYS", 1))
            return None
        try:
            if is_market_open():
                return _compute_open_candidate(symbol, quote_dict=quote_dict, reject_dict=reject_dict, reject_lock=reject_lock)
            else:
                return _compute_premarket_candidate(symbol, quote_dict=quote_dict, reject_dict=reject_dict, reject_lock=reject_lock)
        except Exception:
            logger.warning("Scan error for %s", symbol, exc_info=True)
            return None

    max_workers = getattr(config, "SCANNER_MAX_WORKERS", 5)
    symbol_timeout = getattr(config, "SCANNER_SYMBOL_TIMEOUT", 15.0)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_eval_sym, sym): sym for sym in symbols}
            for future in as_completed(futures):
                try:
                    res = future.result(timeout=symbol_timeout)
                    if res is not None:
                        candidates.append(res)
                except TimeoutError:
                    sym_name = futures.get(future, "UNKNOWN")
                    logger.warning("[SCANNER TIMEOUT] Fetch timed out (%.1fs) for %s — skipping", symbol_timeout, sym_name)
                    continue
                except Exception as exc:
                    sym_name = futures.get(future, "UNKNOWN")
                    logger.warning("[SCANNER ERROR] Error evaluating %s: %s", sym_name, exc)
                    continue
    except KeyboardInterrupt:
        logger.info("[SCANNER] Keyboard interrupt received — shutting down scanner cleanly.")
        raise

    candidates.sort(key=lambda c: c.score, reverse=True)
    top = candidates[:max_candidates * 2]
    
    if getattr(config, "SCAN_ENABLE_SENTIMENT", False):
        logger.info(f"Fetching news sentiment for top {len(top)} candidates...")
        for cand in top:
            news_res = analyze_news_sentiment(cand.symbol)
            cand.sentiment_score = news_res.get("sentiment_score", 0.0)
            cand.catalyst_type = news_res.get("catalyst_type")
            cand.top_headline = news_res.get("top_headline", "")
            
            sentiment_weight = getattr(config, "SCAN_SCORE_SENTIMENT_WEIGHT", 0.2)
            sentiment_bonus = cand.sentiment_score * sentiment_weight * 10
            cand.score += max(sentiment_bonus, 0)
            
        top.sort(key=lambda c: c.score, reverse=True)
        
    top = top[:max_candidates]

    _save_results(top)
    top5_reasons = dict(sorted(reject_dict.items(), key=lambda x: x[1], reverse=True)[:5])
    top5_str = " | ".join(f"{r}x{c}" for r, c in top5_reasons.items()) or "none"
    
    if not top:
        logger.warning(
            "⚠️ Scanner Health: 0 candidates from %d symbols. Top rejection reasons: %s | Fallback watchlist active",
            len(symbols),
            top5_str,
        )
    else:
        logger.info(
            "Scanner mode=%s session=%s found %d candidates (%d/%d symbols passed) | Rejections: [%s]",
            mode,
            session_status(),
            len(top),
            len(top),
            len(symbols),
            top5_str,
        )
    return top


def _save_results(candidates: List[ScanCandidate]):
    if not candidates:
        logger.debug("No scan candidates; skipping CSV write")
        return

    path = config.SCAN_RESULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(candidates[0]).keys()))
        writer.writeheader()
        for c in candidates:
            writer.writerow(asdict(c))


class BackgroundScanner:
    def __init__(self, interval_minutes: int = 10, max_candidates: int = 10, initial_watchlist: List[str] = None):
        self.interval_seconds = interval_minutes * 60
        self.max_candidates = max_candidates
        self.watchlist = initial_watchlist or []
        self.last_scan_completed = _time.time() if initial_watchlist else 0.0
        self.restart_count = 0
        self.last_restart_time = 0.0
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self):
        if self._running and self._thread and self._thread.is_alive():
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="BackgroundScannerThread")
        self._thread.start()
        logger.info("[BackgroundScanner] Periodic background scanner thread started.")

    def stop(self):
        self._running = False
        self._stop_event.set()
        if self._thread:
            try:
                self._thread.join(timeout=2)
            except Exception:
                pass
            self._thread = None
            logger.info("[BackgroundScanner] Periodic background scanner thread stopped.")

    def ensure_running(self):
        """Self-healing watchdog: restarts thread if dead or stale with capped crash-loop backoff."""
        if not self._running or self._thread is None or not self._thread.is_alive():
            now = _time.time()
            if self.last_restart_time > 0 and (now - self.last_restart_time < 10.0):
                # True exponential backoff (2s, 4s, 8s, 16s, 32s, max 60s) to avoid CPU spin
                backoff_delay = min(float(2 ** min(self.restart_count, 6)), 60.0)
                _time.sleep(backoff_delay)
            self.restart_count += 1
            self.last_restart_time = _time.time()
            logger.warning(
                "[BackgroundScanner] Watchdog detected dead/missing scanner thread (restart #%d). Restarting cleanly...",
                self.restart_count
            )
            self.stop()
            self.start()

    def get_health_status(self) -> dict:
        """Returns health diagnostics for observability and test verification."""
        with self._lock:
            age = _time.time() - self.last_scan_completed if self.last_scan_completed > 0 else -1.0
            is_alive = self._thread is not None and self._thread.is_alive()
            return {
                "running": self._running,
                "thread_alive": is_alive,
                "last_scan_completed": self.last_scan_completed,
                "scanner_age_seconds": round(age, 1) if age >= 0 else None,
                "restart_count": self.restart_count,
                "watchlist_size": len(self.watchlist),
            }

    def _run_loop(self):
        from strategies.session import is_market_open, is_premarket
        was_market_open = False
        while self._running:
            try:
                sleep_step = 5.0
                elapsed = 0.0
                currently_open = is_market_open()

                # Force immediate scan on market open transition or if stale
                force_scan = (currently_open and not was_market_open) or (_time.time() - self.last_scan_completed > self.interval_seconds)
                was_market_open = currently_open

                if not force_scan:
                    while elapsed < self.interval_seconds:
                        if not self._running:
                            return
                        if self._stop_event.wait(timeout=sleep_step):
                            return
                        elapsed += sleep_step

                if is_market_open() or is_premarket():
                    logger.info("[BackgroundScanner] Starting periodic background scan...")
                    try:
                        candidates = scan(max_candidates=self.max_candidates)
                        if candidates:
                            new_watchlist = [c.symbol for c in candidates]
                            with self._lock:
                                self.watchlist = new_watchlist
                                self.last_scan_completed = _time.time()
                            logger.info(f"[BackgroundScanner] Background scan completed. New watchlist: {self.watchlist}")
                        else:
                            with self._lock:
                                self.last_scan_completed = _time.time()
                            logger.info("[BackgroundScanner] Background scan completed with no candidates.")
                    except Exception as e:
                        logger.error(f"[BackgroundScanner] Error during periodic scan: {e}", exc_info=True)
                        with self._lock:
                            # Update timestamp on error to prevent runaway tight loops
                            self.last_scan_completed = _time.time()
                else:
                    # Market closed / afterhours: update timestamp so stale guard doesn't fire erroneously
                    with self._lock:
                        self.last_scan_completed = _time.time()
                    self._stop_event.wait(timeout=30.0)
            except Exception as loop_err:
                logger.error(f"[BackgroundScanner] Unhandled error in run loop: {loop_err}", exc_info=True)
                self._stop_event.wait(timeout=10.0)

    def get_watchlist(self) -> List[str]:
        with self._lock:
            return list(self.watchlist)

    def get_last_scan_completed_time(self) -> float:
        with self._lock:
            return self.last_scan_completed
