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
                    
                final_symbols = list(symbols_pool)
                if final_symbols:
                    logger.info("Fetched dynamic symbol universe from Alpaca (%d unique symbols)", len(final_symbols))
                    _screener_cache["data"] = final_symbols
                    _screener_cache["fetched_at"] = now
                    return final_symbols
                else:
                    logger.warning("Dynamic screener returned empty lists (e.g. premarket stale). Falling back.")
            except Exception as e:
                logger.warning(f"Dynamic screener failed: {e}. Falling back to SYMBOL_UNIVERSE_FILE.")

    # 2nd -> static watchlist & core symbols fallback (prevents yfinance spam on delisted symbols)
    core_syms = getattr(config, "FULL_MARKET_CORE_SYMBOLS", []) or getattr(config, "WATCHLIST", [])
    if core_syms:
        logger.info("Falling back to core symbols/watchlist universe (%d symbols)", len(core_syms))
        return list(dict.fromkeys(core_syms))

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
                return syms[:100]

    return ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMZN", "AMD", "TSLA"]


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


def _passes_spread_filter(symbol: str, is_premarket: bool = True, quote_dict: dict = None) -> bool:
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
        logger.debug("%s skipped: spread %.4f exceeds max %.4f (premarket=%s)", symbol, spread_pct, max_spread, is_premarket)
        return False
    return True


def _passes_premarket_filters(
    symbol: str,
    gap_pct: float,
    current_price: float,
    volume: int,
    relative_volume: float,
    quote_dict: dict = None,
) -> bool:
    allow_short = getattr(config, "ALLOW_SHORT", False)
    if gap_pct < 0 and not allow_short:
        logger.debug("%s skipped: gap-down with ALLOW_SHORT=False", symbol)
        return False

    if current_price < config.SCAN_MIN_PRICE or current_price > config.SCAN_MAX_PRICE:
        logger.debug("%s skipped: price %.2f outside scan range", symbol, current_price)
        return False
    if abs(gap_pct) < config.SCAN_MIN_GAP_PCT:
        logger.debug("%s skipped: gap %.2f%% below %.2f%%", symbol, gap_pct, config.SCAN_MIN_GAP_PCT)
        return False
    if abs(gap_pct) > config.SCAN_MAX_GAP_PCT:
        logger.debug("%s skipped: gap %.2f%% above %.2f%%", symbol, gap_pct, config.SCAN_MAX_GAP_PCT)
        return False
    if volume < config.SCAN_MIN_PREMARKET_VOLUME:
        logger.debug("%s skipped: PM volume %d below %d", symbol, volume, config.SCAN_MIN_PREMARKET_VOLUME)
        return False
    if relative_volume < config.SCAN_MIN_RELATIVE_VOLUME:
        logger.debug("%s skipped: rvol %.2f below %.2f", symbol, relative_volume, config.SCAN_MIN_RELATIVE_VOLUME)
        return False

    min_avg_vol = getattr(config, "SCAN_MIN_AVG_DAILY_VOLUME", 750_000)
    avg_vol = _average_daily_volume(symbol)
    if avg_vol > 0 and avg_vol < min_avg_vol:
        logger.debug("%s skipped: avg daily vol %.0f below %d", symbol, avg_vol, min_avg_vol)
        return False

    if not _passes_spread_filter(symbol, is_premarket=True, quote_dict=quote_dict):
        return False

    return True


def _passes_open_filters(
    symbol: str,
    move_pct: float,
    current_price: float,
    volume: int,
    relative_volume: float,
    quote_dict: dict = None,
) -> bool:
    allow_short = getattr(config, "ALLOW_SHORT", False)
    if move_pct < 0 and not allow_short:
        logger.debug("%s skipped: down move with ALLOW_SHORT=False", symbol)
        return False
    if current_price < config.SCAN_MIN_PRICE or current_price > config.SCAN_MAX_PRICE:
        logger.debug("%s skipped: price %.2f outside scan range", symbol, current_price)
        return False
    if abs(move_pct) < getattr(config, "SCAN_OPEN_MIN_MOVE_PCT", 0.25):
        logger.debug("%s skipped: open move %.2f%% too small", symbol, move_pct)
        return False
    if volume < getattr(config, "SCAN_OPEN_MIN_VOLUME", 100_000):
        logger.debug("%s skipped: session volume %d too small", symbol, volume)
        return False
    if relative_volume < getattr(config, "SCAN_OPEN_MIN_RELATIVE_VOLUME", 0.05):
        logger.debug("%s skipped: session rvol %.2f too small", symbol, relative_volume)
        return False

    min_avg_vol = getattr(config, "SCAN_MIN_AVG_DAILY_VOLUME", 750_000)
    avg_vol = _average_daily_volume(symbol)
    if avg_vol > 0 and avg_vol < min_avg_vol:
        logger.debug("%s skipped: avg daily vol %.0f below %d", symbol, avg_vol, min_avg_vol)
        return False

    if not _passes_spread_filter(symbol, is_premarket=False, quote_dict=quote_dict):
        return False

    return True


def _compute_premarket_candidate(symbol: str, quote_dict: dict = None) -> Optional[ScanCandidate]:
    prev = previous_close(symbol)
    if prev is None or prev <= 0:
        return None

    df = fetch_intraday(
        symbol,
        interval=config.PREMARKET_INTERVAL,
        lookback_days=1,
        prepost=True,
    )
    if df.empty:
        return None

    pm_df = _premarket_bars(df, symbol)
    if pm_df.empty:
        logger.debug("No premarket bars for %s", symbol)
        return None

    current = float(pm_df["close"].iloc[-1])
    volume = int(pm_df["volume"].sum())
    gap_pct = ((current - prev) / prev) * 100.0
    rel_vol = _relative_volume(symbol, volume)

    if not _passes_premarket_filters(symbol, gap_pct, current, volume, rel_vol, quote_dict=quote_dict):
        return None

    score = _score(gap_pct, rel_vol, volume)
    direction = "GAP_UP" if gap_pct > 0 else "GAP_DOWN"
    reason = f"{direction} {gap_pct:+.1f}% | PM vol {volume:,} | rvol {rel_vol:.2f}x"

    return ScanCandidate(symbol, prev, current, gap_pct, volume, rel_vol, score, reason)


def _compute_open_candidate(symbol: str, quote_dict: dict = None) -> Optional[ScanCandidate]:
    prev = previous_close(symbol)
    if prev is None or prev <= 0:
        return None

    df = fetch_intraday(
        symbol,
        interval=config.INTRADAY_INTERVAL,
        lookback_days=1,
    )
    if df.empty:
        return None

    reg_df = _regular_session_bars(df, symbol)
    if reg_df.empty:
        logger.debug("No regular-session bars for %s", symbol)
        return None

    current = float(reg_df["close"].iloc[-1])
    open_price = float(reg_df["open"].iloc[0])
    volume = int(reg_df["volume"].sum())
    move_pct = ((current - open_price) / open_price) * 100.0 if open_price > 0 else 0.0
    gap_pct = ((current - prev) / prev) * 100.0
    rel_vol = _relative_volume(symbol, volume)

    if not _passes_open_filters(symbol, move_pct, current, volume, rel_vol, quote_dict=quote_dict):
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
    max_candidates = max_candidates or config.SCAN_MAX_CANDIDATES
    mode = "OPEN" if is_market_open() else "PREMARKET"

    quote_dict = {}
    try:
        api_key = os.environ.get("APCA_API_KEY_ID", "")
        secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
        if api_key and secret_key:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest
            data_client = StockHistoricalDataClient(api_key, secret_key)
            # Batch fetch quotes for chunk
            quote_dict = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbols[:100], feed="iex"))
    except Exception as exc:
        logger.debug("Failed batch fetching quotes for scan: %s", exc)

    candidates = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _eval_sym(sym: str) -> Optional[ScanCandidate]:
        symbol = str(sym).upper()
        try:
            if is_market_open():
                return _compute_open_candidate(symbol, quote_dict=quote_dict)
            else:
                return _compute_premarket_candidate(symbol, quote_dict=quote_dict)
        except Exception:
            logger.warning("Scan error for %s", symbol, exc_info=True)
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_eval_sym, sym): sym for sym in symbols}
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                candidates.append(res)

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
    if not top:
        logger.warning(
            "⚠️ Scanner Health Guard: 0 candidates found from dynamic universe (%d symbols tested). Fallback watchlist active.",
            len(symbols),
        )
    else:
        logger.info(
            "Scanner mode=%s session=%s found %d candidates (top %d from %d symbols)",
            mode, session_status(), len(top), max_candidates, len(symbols),
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
        self._lock = threading.Lock()
        self._thread = None
        self._stop_event = threading.Event()
        self._running = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
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

    def _run_loop(self):
        from strategies.session import is_market_open, is_premarket
        while self._running:
            # Sleep in small steps to be responsive to stop requests
            sleep_step = 5.0
            elapsed = 0.0
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

    def get_watchlist(self) -> List[str]:
        with self._lock:
            return list(self.watchlist)

    def get_last_scan_completed_time(self) -> float:
        with self._lock:
            return self.last_scan_completed
