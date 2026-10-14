"""strategies/regime_engine.py

Pre-Market Market Health Gate & Market Regime Engine for Stock Engine Pro V3.

Evaluates macro market conditions, index gaps, sector shock indicators,
and scanner candidate breadth to dynamically classify market regime into:
  - RISK_ON (Score >= 0): Normal Sizing, Normal Confidence, Normal Entries.
  - CAUTION (Score -1 to -3): 15-Min Opening Entry Buffer (No BUY before 09:45 ET), 50% Sizing, +0.15 Confidence Gate.
  - RISK_OFF (Score <= -4): NO NEW LONG DAY TRADES. Exits and protective actions remain fully enabled.
  - DATA_DEGRADED: Critical benchmark (SPY/QQQ) missing or stale. HALT all new entries (Fail-Closed).
"""

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

import config
from strategies.session import now_eastern

logger = logging.getLogger(__name__)

MODE_RISK_ON = "RISK_ON"
MODE_CAUTION = "CAUTION"
MODE_RISK_OFF = "RISK_OFF"
MODE_DATA_DEGRADED = "DATA_DEGRADED"

TEXTURE_TRENDING = "TRENDING"
TEXTURE_SIDEWAYS = "SIDEWAYS"
TEXTURE_VOLATILE = "VOLATILE"
TEXTURE_TRANSITIONING = "TRANSITIONING"

_last_soxx_lock = threading.Lock()
_last_soxx_pct: Optional[float] = None


def get_soxx_pct_change() -> Optional[float]:
    """Return the latest cached percentage change for SOXX semiconductor ETF."""
    with _last_soxx_lock:
        return _last_soxx_pct


def clear_soxx_cache() -> None:
    """Clear the cached SOXX percentage change to prevent stale reuse."""
    global _last_soxx_pct
    with _last_soxx_lock:
        _last_soxx_pct = None


# ── Benchmark Score Hysteresis Deadbands ─────────────────────────────────────
_SPY_ESCALATE_THRESHOLD = -0.75   # step to -2 when SPY < this
_SPY_DEESC_FROM_2_TO_1  = -0.70   # recover above this to drop back to -1
_SPY_DEESC_FROM_1_TO_0  = -0.35   # recover above this to drop back to  0

_QQQ_ESCALATE_THRESHOLD = -1.00   # step to -2 when QQQ < this
_QQQ_DEESC_FROM_2_TO_1  = -0.90   # recover above this to drop back to -1
_QQQ_DEESC_FROM_1_TO_0  = -0.45   # recover above this to drop back to  0

_factor_hysteresis_lock = threading.Lock()
_last_spy_factor: Optional[int] = None
_last_qqq_factor: Optional[int] = None


def reset_factor_hysteresis() -> None:
    """Reset hysteresis state. Call at session start and from tests."""
    global _last_spy_factor, _last_qqq_factor
    with _factor_hysteresis_lock:
        _last_spy_factor = None
        _last_qqq_factor = None


def _apply_spy_hysteresis(spy_pct: Optional[float]) -> int:
    """
    Return hysteresis-smoothed SPY factor score.
    Escalates immediately; de-escalates only after recovery clears deadband.
    """
    if spy_pct is None:
        return 0

    global _last_spy_factor
    with _factor_hysteresis_lock:
        prev = _last_spy_factor

        if spy_pct < _SPY_ESCALATE_THRESHOLD:        # < -0.75 -> -2
            new_factor = -2
        elif spy_pct < -0.40:                        # < -0.40 -> at most -1
            if prev is not None and prev <= -2:
                new_factor = -2 if spy_pct < _SPY_DEESC_FROM_2_TO_1 else -1
            else:
                new_factor = -1
        else:                                        # >= -0.40 -> 0 or -1
            if prev is not None and prev <= -2:
                new_factor = -2 if spy_pct < _SPY_DEESC_FROM_2_TO_1 else (
                             -1 if spy_pct < _SPY_DEESC_FROM_1_TO_0 else 0)
            elif prev is not None and prev == -1:
                new_factor = -1 if spy_pct < _SPY_DEESC_FROM_1_TO_0 else 0
            else:
                new_factor = 0

        _last_spy_factor = new_factor
        return new_factor


def _apply_qqq_hysteresis(qqq_pct: Optional[float]) -> int:
    """
    Return hysteresis-smoothed QQQ factor score.
    Same escalate-fast / de-escalate-slow principle as SPY.
    """
    if qqq_pct is None:
        return 0

    global _last_qqq_factor
    with _factor_hysteresis_lock:
        prev = _last_qqq_factor

        if qqq_pct < _QQQ_ESCALATE_THRESHOLD:        # < -1.00 -> -2
            new_factor = -2
        elif qqq_pct < -0.50:                        # < -0.50 -> at most -1
            if prev is not None and prev <= -2:
                new_factor = -2 if qqq_pct < _QQQ_DEESC_FROM_2_TO_1 else -1
            else:
                new_factor = -1
        else:                                        # >= -0.50 -> 0 or -1
            if prev is not None and prev <= -2:
                new_factor = -2 if qqq_pct < _QQQ_DEESC_FROM_2_TO_1 else (
                             -1 if qqq_pct < _QQQ_DEESC_FROM_1_TO_0 else 0)
            elif prev is not None and prev == -1:
                new_factor = -1 if qqq_pct < _QQQ_DEESC_FROM_1_TO_0 else 0
            else:
                new_factor = 0

        _last_qqq_factor = new_factor
        return new_factor



def detect_market_texture(adx: float = 20.0, bb_width: float = 0.02, atr_ratio: float = 1.0, price_change_5m: float = 0.0) -> str:
    """
    Classifies market texture into TRENDING, SIDEWAYS, VOLATILE, or TRANSITIONING.
    - Volatile Guard: atr_ratio >= 1.5 OR price_change_5m <= -2% (-0.02) -> VOLATILE
    - Breakout Guard: price_change_5m >= 1% (0.01) -> TRENDING
    - Trending: adx >= 25 -> TRENDING
    - Sideways: adx < 20 AND bb_width <= 0.025 -> SIDEWAYS
    - Otherwise: TRANSITIONING
    """
    if atr_ratio >= 1.5 or price_change_5m <= -0.02:
        return TEXTURE_VOLATILE
    if abs(price_change_5m) >= 0.01:
        return TEXTURE_TRENDING
    if adx >= 25.0:
        return TEXTURE_TRENDING
    if adx < 20.0 and bb_width <= 0.025:
        return TEXTURE_SIDEWAYS
    return TEXTURE_TRANSITIONING


DIAGNOSTIC_VALID = "VALID"
DIAGNOSTIC_UNUSUAL_BUT_ALLOWED = "UNUSUAL_BUT_ALLOWED"
DIAGNOSTIC_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
DIAGNOSTIC_INTERNALLY_INCONSISTENT = "INTERNALLY_INCONSISTENT"


def extract_benchmark_pct(data: Optional[Dict[str, Any]], symbol: str) -> Optional[float]:
    """
    Extracts benchmark percentage return safely from flat or nested dictionary structures.
    Supports:
      - data['spy_pct']
      - data['SPY']['pct']
      - data['SPY'] (scalar float or int)
    """
    if not data or not isinstance(data, dict):
        return None
    flat_key = f"{symbol.lower()}_pct"
    if flat_key in data and data[flat_key] is not None:
        try:
            return float(data[flat_key])
        except (TypeError, ValueError):
            pass
    node = data.get(symbol.upper()) or data.get(symbol.lower())
    if isinstance(node, dict):
        val = node.get("pct")
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    elif isinstance(node, (int, float)):
        return float(node)
    return None


def compute_vwap_breadth(
    symbols: Optional[List[str]] = None,
    bridge=None,
    data: Optional[Dict[str, Any]] = None,
    symbol_data: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str, Dict[str, Any]]:
    """Compute market breadth as % of symbols trading above daily VWAP (Dual-Mode).

    Dual-mode fallback:
      - Primary: Watchlist VWAP breadth (evaluated symbols with valid intraday bars >= 5).
      - Fallback: If valid count < 5, fall back to index breadth (SPY / QQQ / SOXX / IWM).
    """
    above_vwap = 0
    valid_count = 0
    evaluated_symbols = []

    candidates = symbols or []
    use_batch = os.environ.get("REGIME_BREADTH_BATCH_ENABLED", "").lower() in ("true", "1", "yes")
    batch_fetched = {}
    if use_batch and candidates and not symbol_data:
        try:
            from strategies.intraday_data import fetch_intraday_batch
            batch_fetched, _ = fetch_intraday_batch(
                candidates,
                interval="1m",
                bridge=bridge,
                purpose="REGIME_BREADTH",
                total_timeout_seconds=10.0,
            )
        except Exception as b_err:
            logger.debug("[BREADTH] Batch fetch failed: %s", b_err)
            batch_fetched = {}

    for sym in candidates:
        df = None
        if symbol_data and sym in symbol_data:
            df = symbol_data[sym]
        elif batch_fetched and sym in batch_fetched:
            df = batch_fetched[sym]
        elif bridge is not None:
            try:
                from strategies.intraday_data import fetch_intraday
                df = fetch_intraday(sym, interval="1m", bridge=bridge, purpose="REGIME_BREADTH")
            except Exception:
                df = None

        if df is not None and not df.empty:
            try:
                now_utc = pd.Timestamp.now(tz="UTC")
                idx = df.index
                if idx.tz is None:
                    idx_utc = idx.tz_localize("UTC")
                else:
                    idx_utc = idx.tz_convert("UTC")
                closed_mask = (idx_utc + pd.Timedelta(seconds=60)) <= now_utc
                df = df.loc[closed_mask]
            except Exception:
                pass

        if df is not None and not df.empty and len(df) >= 2:
            try:
                from strategies.indicators import add_vwap
                if "vwap" not in df.columns:
                    df = add_vwap(df)
                last_close = float(df["close"].iloc[-1])
                last_vwap = float(df["vwap"].iloc[-1])
                if math.isfinite(last_close) and math.isfinite(last_vwap) and last_vwap > 0:
                    valid_count += 1
                    is_above = last_close > last_vwap
                    if is_above:
                        above_vwap += 1
                    evaluated_symbols.append((sym, is_above, last_close, last_vwap))
            except Exception:
                pass

    if valid_count >= 5:
        breadth_pct = round((above_vwap / valid_count) * 100.0, 2)
        mode = "WATCHLIST_VWAP"
        details = {
            "mode": mode,
            "evaluated": valid_count,
            "above_vwap": above_vwap,
            "breadth_pct": breadth_pct,
            "symbols": [s[0] for s in evaluated_symbols],
        }
        return breadth_pct, mode, details

    # Fallback to Index Breadth
    dt = data or {}
    benchmarks = ["SPY", "QQQ", "SOXX", "IWM"]
    bench_valid = 0
    bench_pos = 0
    for b_sym in benchmarks:
        pct = extract_benchmark_pct(dt, b_sym)
        if pct is not None and math.isfinite(pct):
            bench_valid += 1
            if pct > 0:
                bench_pos += 1

    if bench_valid > 0:
        breadth_pct = round((bench_pos / bench_valid) * 100.0, 2)
    else:
        breadth_pct = 50.0  # Neutral fallback

    mode = "INDEX_FALLBACK"
    details = {
        "mode": mode,
        "evaluated": bench_valid,
        "positive_count": bench_pos,
        "breadth_pct": breadth_pct,
        "fallback_reason": f"Watchlist valid count {valid_count} < 5 minimum threshold",
    }
    return breadth_pct, mode, details


def validate_regime_texture_combination(
    mode: str,
    texture: str,
    component_scores: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Diagnostic-only consistency validator evaluating alignment between macro regime mode and micro market texture.
    Returns (status, message, attribution_dict).

    Adheres strictly to Phase A invariant:
    Emits telemetry/diagnostic logging ONLY.
    DO NOT alter mode, sizing, trade halts, or strategy eligibility!
    """
    scores = component_scores or {}
    dt = data or {}

    spy_pct = extract_benchmark_pct(dt, "SPY")
    qqq_pct = extract_benchmark_pct(dt, "QQQ")
    soxx_pct = extract_benchmark_pct(dt, "SOXX")

    attribution = {
        "mode": mode,
        "texture": texture,
        "spy_pct": spy_pct,
        "qqq_pct": qqq_pct,
        "soxx_pct": soxx_pct,
        "breadth": dt.get("market_breadth", dt.get("scanner_candidate_count")),
        "factor_scores": scores,
    }

    if mode == MODE_DATA_DEGRADED or not mode:
        return (
            DIAGNOSTIC_INSUFFICIENT_DATA,
            "Mode is DATA_DEGRADED or unspecified; insufficient benchmark data for texture validation",
            attribution,
        )

    # Fail-closed benchmark absence check
    if spy_pct is None and qqq_pct is None:
        return (
            DIAGNOSTIC_INSUFFICIENT_DATA,
            "Required benchmark inputs unavailable",
            attribution,
        )

    if not scores and not dt:
        return (
            DIAGNOSTIC_INSUFFICIENT_DATA,
            "Component scores and market data dictionaries are empty",
            attribution,
        )

    # 1. Inconsistent: RISK_ON + VOLATILE
    if mode == MODE_RISK_ON and texture == TEXTURE_VOLATILE:
        msg = (
            "Macro scoring evaluated to RISK_ON while micro texture detected VOLATILE "
            "(ATR ratio or severe 5m drop). Conflicting macro vs micro signals."
        )
        return (DIAGNOSTIC_INTERNALLY_INCONSISTENT, msg, attribution)

    # 2. Inconsistent: RISK_OFF with uniformly bullish components
    if (
        mode == MODE_RISK_OFF
        and spy_pct is not None
        and spy_pct >= 0.50
        and qqq_pct is not None
        and qqq_pct >= 0.50
    ):
        msg = (
            f"Mode evaluated to RISK_OFF despite strong positive benchmark indices "
            f"(SPY={spy_pct:+.2f}%, QQQ={qqq_pct:+.2f}%). Potential factor score conflict."
        )
        return (DIAGNOSTIC_INTERNALLY_INCONSISTENT, msg, attribution)

    # 3. Unusual but allowed: RISK_ON + TRANSITIONING / SIDEWAYS
    if mode == MODE_RISK_ON and texture in (TEXTURE_TRANSITIONING, TEXTURE_SIDEWAYS):
        msg = (
            f"Macro backdrop is RISK_ON but micro texture is {texture}. "
            f"Market exhibiting range consolidation or intraday texture shift."
        )
        return (DIAGNOSTIC_UNUSUAL_BUT_ALLOWED, msg, attribution)

    # 4. Unusual but allowed: CAUTION + VOLATILE
    if mode == MODE_CAUTION and texture == TEXTURE_VOLATILE:
        msg = (
            "Macro mode is CAUTION and micro texture is VOLATILE. "
            "High volatility during cautious macro backdrop."
        )
        return (DIAGNOSTIC_UNUSUAL_BUT_ALLOWED, msg, attribution)

    # 5. Unusual but allowed: RISK_OFF + SIDEWAYS / TRANSITIONING
    if mode == MODE_RISK_OFF and texture in (TEXTURE_SIDEWAYS, TEXTURE_TRANSITIONING):
        msg = (
            f"Macro mode is RISK_OFF with muted micro volatility ({texture}). "
            f"Slow drift or calm market weakness."
        )
        return (DIAGNOSTIC_UNUSUAL_BUT_ALLOWED, msg, attribution)

    # 6. Harmonious / Valid combinations:
    # - RISK_ON + TRENDING
    # - CAUTION + TRENDING / SIDEWAYS / TRANSITIONING
    # - RISK_OFF + VOLATILE / TRENDING
    return (
        DIAGNOSTIC_VALID,
        f"Regime mode {mode} and texture {texture} are harmonious and consistent",
        attribution,
    )


@dataclass
class MarketRegimeResult:
    mode: str  # "RISK_ON", "CAUTION", "RISK_OFF", "DATA_DEGRADED"
    score: int
    allow_new_longs: bool
    requires_opening_buffer: bool  # True if must wait until 09:45 ET
    position_scale: float  # 1.0 for RISK_ON, 0.5 for CAUTION, 0.0 for RISK_OFF/DATA_DEGRADED
    confidence_boost: float  # +0.0 for RISK_ON, +0.15 for CAUTION
    allow_new_shorts: bool = True  # True if short selling allowed for current regime
    allow_inverse_etf_longs: bool = False  # True if inverse/bear ETFs (SPDN/SQQQ) allowed in RISK_OFF
    reasons: List[str] = field(default_factory=list)
    factor_scores: Dict[str, int] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: Optional[datetime] = None
    scanner_candidate_count: Optional[int] = None
    max_trades_day: Optional[int] = None
    halt_new_entries: bool = False
    market_texture: str = TEXTURE_TRENDING
    diagnostic_status: str = DIAGNOSTIC_VALID
    diagnostic_message: Optional[str] = None

    def _buffer_display(self) -> str:
        """Human-readable opening buffer status for log/summary output."""
        if not self.requires_opening_buffer:
            return "False"
        try:
            from strategies.session import is_past_opening_buffer
            return "ELAPSED(past_0945)" if is_past_opening_buffer(15) else "ACTIVE(wait_0945)"
        except Exception:
            return str(self.requires_opening_buffer)

    def summary(self) -> str:
        if self.reasons:
            reasons_str = " | ".join(self.reasons)
        elif self.diagnostic_status == DIAGNOSTIC_INSUFFICIENT_DATA:
            reasons_str = "Insufficient Benchmark Data (Fallback Mode)"
        else:
            reasons_str = "Normal Healthy Market"
        return (
            f"[REGIME] mode={self.mode} texture={self.market_texture} diag={self.diagnostic_status} score={self.score:+d} "
            f"allow_longs={self.allow_new_longs} halt_entries={self.halt_new_entries} "
            f"max_trades={0 if (self.halt_new_entries or not self.allow_new_longs) else self.max_trades_day} "
            f"open_buffer_0945={self._buffer_display()} "
            f"sizing={0 if (self.halt_new_entries or not self.allow_new_longs) else int(self.position_scale*100)}% "
            f"boost=+{self.confidence_boost:.2f} reasons=[{reasons_str}]"
        )


def _extract_prev_session_close(
    bars_list: list,
    today_date=None,
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """
    Extract the previous completed trading session's close price from daily bars.
    Calendar-aware: selects the most recent bar strictly prior to today's trading session (bar_date < today_date).
    Returns: (prev_close, quote_ts, bar_date_str)
    """
    if not bars_list:
        return None, None, None

    if today_date is None:
        try:
            from strategies.session import now_eastern
            today_date = now_eastern().date()
        except Exception:
            today_date = datetime.now(timezone.utc).date()

    import pytz
    et_tz = pytz.timezone("US/Eastern")

    completed_bars = []
    latest_bar_ts = None

    for b in bars_list:
        raw_ts = getattr(b, "timestamp", None) or getattr(b, "date", None)
        if raw_ts is None and isinstance(b, dict):
            raw_ts = b.get("timestamp") or b.get("date")

        b_dt = None
        if isinstance(raw_ts, datetime):
            b_dt = raw_ts if raw_ts.tzinfo else raw_ts.replace(tzinfo=timezone.utc)
        elif isinstance(raw_ts, str):
            try:
                b_dt = datetime.fromisoformat(raw_ts)
                if b_dt.tzinfo is None:
                    b_dt = b_dt.replace(tzinfo=timezone.utc)
            except Exception:
                b_dt = None
        elif isinstance(raw_ts, (int, float)):
            b_dt = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)

        if b_dt is not None:
            latest_bar_ts = b_dt.timestamp()
            b_eastern_date = b_dt.astimezone(et_tz).date()
            if b_eastern_date < today_date:
                completed_bars.append((b_eastern_date, b_dt, b))

    if completed_bars:
        # Sort by date ascending to get the latest completed session
        completed_bars.sort(key=lambda x: x[1])
        latest_date, latest_dt, latest_bar = completed_bars[-1]

        # Full NYSE Exchange Calendar Validation: ensure completed bar is the EXACT expected prior trading session
        try:
            from strategies.session import get_previous_market_day
            expected_prev_session = get_previous_market_day(today_date)
            if latest_date != expected_prev_session:
                logger.warning(
                    "[REGIME CALENDAR MISMATCH] Latest completed bar date %s does not match expected exchange session date %s (today=%s). Failing closed.",
                    latest_date, expected_prev_session, today_date
                )
                return None, None, None
        except Exception as cal_err:
            logger.error("[REGIME CALENDAR ERROR] Failed to determine previous market day: %s. Failing closed.", cal_err)
            return None, None, None

        close_val = getattr(latest_bar, "close", None)
        if close_val is None and isinstance(latest_bar, dict):
            close_val = latest_bar.get("close")
        if close_val is not None:
            try:
                c_float = float(close_val)
                if c_float > 0 and math.isfinite(c_float):
                    return c_float, latest_dt.timestamp(), latest_date.isoformat()
            except (ValueError, TypeError):
                pass

    # No guessing or blind index fallback: missing or unparseable session data fails closed
    logger.warning("[REGIME BASELINE DATA MISSING] No valid completed previous session bar found for calendar date %s", today_date)
    return None, None, None


def _get_premarket_pct_change(symbol: str, bridge=None) -> Tuple[Optional[float], str, Optional[float]]:
    """Helper to fetch pre-market or current daily % change and quote epoch for a benchmark symbol.
    Returns (pct_change, data_source_str, quote_epoch_ts).
    """
    now_epoch = time.time()

    # 1. Try Alpaca bridge / data client
    if bridge is not None:
        try:
            # Check for quote object first to obtain exact broker timestamp and price from same observation
            quote_ts = None
            price = None
            is_live_quote = False
            if hasattr(bridge, "get_quote"):
                try:
                    q = bridge.get_quote(symbol)
                    if q:
                        raw_ts = getattr(q, "timestamp", None) if not isinstance(q, dict) else q.get("timestamp")
                        if isinstance(raw_ts, (int, float)):
                            quote_ts = float(raw_ts)
                        elif isinstance(raw_ts, datetime):
                            quote_ts = raw_ts.timestamp()

                        raw_px = getattr(q, "last", None) or getattr(q, "ask", None) if not isinstance(q, dict) else (q.get("last") or q.get("ask"))
                        if raw_px is None and isinstance(q, dict) and "bid" in q and "ask" in q:
                            b, a = float(q.get("bid", 0.0) or 0.0), float(q.get("ask", 0.0) or 0.0)
                            if b > 0 and a > 0:
                                raw_px = (b + a) / 2.0
                        if raw_px is not None:
                            try:
                                px_f = float(raw_px)
                                if math.isfinite(px_f) and px_f > 0:
                                    price = px_f
                                    is_live_quote = True
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    pass

            if price is None and hasattr(bridge, "get_price"):
                price = bridge.get_price(symbol, allow_historical=True)

            if price is not None and price > 0:
                    prev_close = None
                    baseline_source = "alpaca_quote_vs_prev_close" if is_live_quote else "alpaca_historical_fallback"
                    session_date_str = None
                    bar_epoch = None
                    if hasattr(bridge, "fetch_historical_data"):
                        try:
                            bars = bridge.fetch_historical_data(symbol)
                            if bars:
                                prev_close, bar_epoch, session_date_str = _extract_prev_session_close(bars)
                        except Exception:
                            pass

                    if prev_close is None and hasattr(bridge, "_data_client") and bridge._data_client is not None:
                        try:
                            from alpaca.data.requests import StockBarsRequest
                            from alpaca.data.timeframe import TimeFrame
                            from alpaca.data.enums import Adjustment
                            from datetime import timedelta
                            now_utc = datetime.now(timezone.utc)
                            req = StockBarsRequest(
                                symbol_or_symbols=symbol,
                                timeframe=TimeFrame.Day,
                                start=now_utc - timedelta(days=7),
                                end=now_utc,
                                adjustment=Adjustment.ALL,
                            )
                            bars = bridge._data_client.get_stock_bars(req)
                            symbol_bars = bars.get(symbol, []) if bars else []
                            if symbol_bars:
                                prev_close, bar_epoch, session_date_str = _extract_prev_session_close(symbol_bars)
                        except Exception:
                            pass

                    if prev_close and prev_close > 0:
                        pct = ((price - prev_close) / prev_close) * 100.0
                        logger.info(
                            "[REGIME BASELINE] %s: prev_close=%.2f (session=%s), price=%.2f, change=%.2f%% (src=%s)",
                            symbol, prev_close, session_date_str or "historical", price, pct, baseline_source,
                        )
                        final_ts = quote_ts
                        return round(pct, 2), baseline_source, final_ts
        except Exception as exc:
            logger.debug("Failed fetching premarket pct via bridge for %s: %s", symbol, exc)

    # 2. Robust yfinance fallback
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            last = float(fast_info.last_price or 0.0)
            prev = float(fast_info.previous_close or 0.0)
            if last > 0 and prev > 0:
                pct = ((last - prev) / prev) * 100.0
                return round(pct, 2), "yfinance_fallback", None
    except Exception as yf_err:
        logger.debug("yfinance fallback failed for %s: %s", symbol, yf_err)

    return None, "data_unavailable", None


def _check_data_health(bridge=None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Pre-flight evaluation of freshness and presence for critical macro benchmark data (SPY/QQQ/SOXX).
    Fail-closed: Returns (is_healthy=False, reason, data) upon missing data, stale timestamps, or exception.
    Freshness math strictly uses epoch float (time.time()) to remain timezone safe.
    """
    global _last_soxx_pct
    clear_soxx_cache()
    health_data: Dict[str, Any] = {}
    try:
        health_cfg = getattr(config, "DATA_HEALTH_SETTINGS", {})
        critical_symbols = health_cfg.get("critical_symbols", ["SPY", "QQQ", "SOXX"])
        stale_thresholds = health_cfg.get("stale_threshold_s", {})
        default_stale_thresh = float(stale_thresholds.get("_default", 180.0))
        now_epoch = time.time()

        for sym in critical_symbols:
            pct, source, ts = _get_premarket_pct_change(sym, bridge)
            thresh_s = float(stale_thresholds.get(sym, default_stale_thresh))

            if pct is None:
                health_data[sym] = {"pct": None, "source": source, "timestamp": ts, "status": "MISSING"}
                return False, f"{sym} benchmark data unavailable", health_data

            if ts is None:
                health_data[sym] = {"pct": pct, "source": source, "timestamp": None, "status": "STALE"}
                return False, f"{sym} benchmark data un-timestamped", health_data

            age_s = now_epoch - ts
            if age_s > thresh_s:
                health_data[sym] = {"pct": pct, "source": source, "timestamp": ts, "status": "STALE"}
                return False, f"{sym} benchmark data stale ({age_s:.1f}s > {thresh_s:.1f}s threshold)", health_data

            health_data[sym] = {"pct": pct, "source": source, "timestamp": ts, "status": "VALID"}

        # SOXX cache populated ONLY from a valid current snapshot
        if "SOXX" in health_data and health_data["SOXX"].get("status") == "VALID":
            with _last_soxx_lock:
                _last_soxx_pct = health_data["SOXX"]["pct"]

        return True, "Critical benchmarks healthy", health_data
    except Exception as exc:
        logger.exception("[DATA HEALTH] check failed; failing closed: %s", exc)
        return False, f"Data health check exception: {exc}", health_data


def evaluate_market_regime(
    bridge=None,
    scanner_candidate_count: Optional[int] = None,
    symbols: Optional[List[str]] = None,
) -> MarketRegimeResult:
    """Evaluates market health metrics and determines current trading regime.
    Pure, stateless function.
    
    Factors & Scoring:
      - SPY premarket < -0.75%: -2 pts, < -0.40%: -1 pt
      - QQQ premarket < -1.00%: -2 pts, < -0.50%: -1 pt
      - Chip ETF (SOXX) < -2.00%: -2 pts, < -1.00%: -1 pt, > +2.00%: +1 pt (Symmetric, fail-closed when missing)
      - Dynamic Scanner Candidates == 0: -2 pts
    """
    # 0. Pre-Flight Data Health Gate (Fail-Closed)
    health_ok, health_reason, health_data = _check_data_health(bridge)
    if not health_ok:
        logger.warning("[DATA HEALTH GATE] Critical benchmark health check failed: %s", health_reason)
        return MarketRegimeResult(
            mode=MODE_DATA_DEGRADED,
            score=0,
            allow_new_longs=False,
            requires_opening_buffer=True,
            position_scale=0.0,
            confidence_boost=0.0,
            allow_new_shorts=False,
            allow_inverse_etf_longs=False,
            reasons=[f"DATA_DEGRADED: {health_reason}"],
            factor_scores={},
            data=health_data,
            evaluated_at=now_eastern(),
            scanner_candidate_count=scanner_candidate_count,
            max_trades_day=0,
            halt_new_entries=True,
            market_texture=TEXTURE_VOLATILE,  # Fail-closed texture: no bullish trend assumptions
        )

    score = 0
    reasons: List[str] = []
    factor_scores: Dict[str, int] = {}
    data: Dict[str, Any] = dict(health_data)

    # 1. Benchmark Index Changes (SPY / QQQ) — with hysteresis deadbands to prevent flapping
    spy_pct = health_data["SPY"]["pct"]
    spy_source = health_data["SPY"]["source"]
    spy_factor = _apply_spy_hysteresis(spy_pct)
    factor_scores["SPY"] = spy_factor
    if spy_factor == -2:
        reasons.append(f"SPY benchmark down {spy_pct:.2f}% (< -0.75% / hyst) [{spy_source}]")
    elif spy_factor == -1:
        reasons.append(f"SPY benchmark weak {spy_pct:.2f}% [{spy_source}]")

    qqq_pct = health_data["QQQ"]["pct"]
    qqq_source = health_data["QQQ"]["source"]
    qqq_factor = _apply_qqq_hysteresis(qqq_pct)
    factor_scores["QQQ"] = qqq_factor
    if qqq_factor == -2:
        reasons.append(f"QQQ tech index down {qqq_pct:.2f}% (< -1.00% / hyst) [{qqq_source}]")
    elif qqq_factor == -1:
        reasons.append(f"QQQ tech index weak {qqq_pct:.2f}% [{qqq_source}]")

    # 2. Chip Sector Shock Check (SOXX) — Consumes validated SOXX snapshot (no second unvalidated fetch)
    soxx_info = health_data.get("SOXX", {})
    soxx_pct = soxx_info.get("pct")
    soxx_source = soxx_info.get("source", "health_snapshot")
    data["SOXX"] = dict(soxx_info)
    if soxx_pct is not None:
        if soxx_pct < -2.00:
            factor_scores["SOXX"] = -2
            reasons.append(f"SOXX semiconductor ETF down {soxx_pct:.2f}% (< -2.00% severe) [{soxx_source}]")
        elif soxx_pct < -1.00:
            factor_scores["SOXX"] = -1
            reasons.append(f"SOXX semiconductor ETF weak {soxx_pct:.2f}% (< -1.00%) [{soxx_source}]")
        elif soxx_pct > 2.00:
            factor_scores["SOXX"] = 1
            reasons.append(f"SOXX semiconductor ETF strong +{soxx_pct:.2f}% [{soxx_source}]")
        else:
            factor_scores["SOXX"] = 0
    else:
        factor_scores["SOXX"] = 0

    # 3. Dynamic Universe Quality Signal
    data["scanner_candidate_count"] = scanner_candidate_count
    if scanner_candidate_count is not None:
        if scanner_candidate_count == 0:
            factor_scores["scanner"] = -2
            reasons.append("Scanner Health Guard active (0 dynamic candidates passed safety filters)")
        else:
            factor_scores["scanner"] = 0
    else:
        reasons.append("scanner_candidate_count_unavailable")

    # 4. Market Breadth (Dual-Mode Watchlist VWAP Breadth / Index Fallback)
    breadth_pct, breadth_mode, breadth_details = compute_vwap_breadth(
        symbols=symbols,
        bridge=bridge,
        data=health_data,
    )
    data["market_breadth"] = breadth_pct
    data["market_breadth_mode"] = breadth_mode
    data["market_breadth_details"] = breadth_details

    score = sum(factor_scores.values())

    # Mode Determination
    halt_new_entries = False
    if score <= -4:
        mode = MODE_RISK_OFF
        allow_new_longs = False
        requires_opening_buffer = True
        position_scale = 0.0
        confidence_boost = 0.25
        max_trades_day = 0
        halt_new_entries = True
    elif score <= -1:
        mode = MODE_CAUTION
        requires_opening_buffer = True  # Must wait until 09:45 ET
        position_scale = 0.5
        confidence_boost = float(getattr(config, "CAUTION_CONFIDENCE_BOOST", 0.15))

        # Dynamic max trades per day based on scanner health & Paper/Live mode
        is_paper = getattr(config, "IS_PAPER_TRADING", True) or getattr(config, "PAPER_TRADING", True)
        if is_paper:
            max_trades_day = 50
        elif scanner_candidate_count is not None and scanner_candidate_count == 0:
            max_trades_day = 20  # Live CAUTION + scanner empty → 20
        else:
            max_trades_day = 30  # Live CAUTION + scanner healthy → 30

        # Score <= -3 → Halt new entries
        if score <= -3:
            allow_new_longs = False
            halt_new_entries = True
            reasons.append(f"Regime score {score} <= -3 → No new long entries allowed")
        else:
            allow_new_longs = True
    else:
        mode = MODE_RISK_ON
        allow_new_longs = True
        requires_opening_buffer = False
        position_scale = 1.0
        confidence_boost = 0.0
        max_trades_day = 50

    # Compute Market Texture (VOLATILE and TRANSITIONING evaluated before TRENDING)
    spy_abs = abs(spy_pct) if spy_pct is not None else 0.0
    if score <= -4 or (spy_pct is not None and spy_pct <= -1.5) or (qqq_pct is not None and qqq_pct <= -1.5):
        market_texture = TEXTURE_VOLATILE
    elif (spy_pct is not None and spy_pct < -0.30) or (qqq_pct is not None and qqq_pct < -0.40):
        market_texture = TEXTURE_TRANSITIONING
    elif spy_abs >= 1.0:
        market_texture = TEXTURE_TRENDING
    elif spy_pct is not None and -0.30 <= spy_pct <= 0.30:
        market_texture = TEXTURE_SIDEWAYS
    else:
        market_texture = TEXTURE_TRENDING

    # Enforce texture-specific halt settings from config
    reg_cfg = getattr(config, "MARKET_REGIME_SETTINGS", {}).get(market_texture, {})
    if reg_cfg.get("halt_new_entries", False):
        halt_new_entries = True
        allow_new_longs = False
        reasons.append(f"Market texture '{market_texture}' forces entry halt via config rules")

    # Short selling & Inverse ETF regime eligibility:
    allow_short_cfg = getattr(config, "ALLOW_SHORT", False)
    inverse_etf_exempt = getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
    macro_index_weak = (spy_pct is not None and spy_pct < -0.30) or (qqq_pct is not None and qqq_pct < -0.40) or (score <= -3)
    allow_new_shorts = allow_short_cfg and macro_index_weak

    # Inverse ETF long-buy eligibility: valid when RISK_OFF / severe CAUTION has halted normal longs
    allow_inverse_etf_longs = bool(inverse_etf_exempt and macro_index_weak and halt_new_entries)

    # Diagnostic-only Regime/Texture Consistency Check (Phase A: Telemetry Only, No Halts/Policy Changes)
    diag_status, diag_msg, diag_attribution = validate_regime_texture_combination(
        mode=mode,
        texture=market_texture,
        component_scores=factor_scores,
        data=data,
    )
    data["diagnostic_consistency"] = {
        "status": diag_status,
        "message": diag_msg,
        "attribution": diag_attribution,
    }
    if diag_status in (DIAGNOSTIC_UNUSUAL_BUT_ALLOWED, DIAGNOSTIC_INTERNALLY_INCONSISTENT):
        logger.warning(
            "[REGIME_TEXTURE_DIAGNOSTIC] %s: mode=%s, texture=%s | %s | Attribution: %s",
            diag_status, mode, market_texture, diag_msg, diag_attribution
        )
    else:
        logger.debug(
            "[REGIME_TEXTURE_DIAGNOSTIC] %s: mode=%s, texture=%s | Attribution: %s",
            diag_status, mode, market_texture, diag_attribution
        )

    res = MarketRegimeResult(
        mode=mode,
        score=score,
        allow_new_longs=allow_new_longs,
        requires_opening_buffer=requires_opening_buffer,
        position_scale=position_scale,
        confidence_boost=confidence_boost,
        allow_new_shorts=allow_new_shorts,
        allow_inverse_etf_longs=allow_inverse_etf_longs,
        reasons=reasons,
        factor_scores=factor_scores,
        data=data,
        evaluated_at=now_eastern(),
        scanner_candidate_count=scanner_candidate_count,
        max_trades_day=max_trades_day,
        halt_new_entries=halt_new_entries,
        market_texture=market_texture,
        diagnostic_status=diag_status,
        diagnostic_message=diag_msg,
    )

    logger.info("Market Regime Evaluation: %s", res.summary())
    return res


class RegimeStateTracker:
    """
    Thread-safe hysteresis state tracker wrapping evaluate_market_regime().
    Maintains recovery cool-down to prevent flapping between DATA_DEGRADED and normal modes,
    and texture dwell hysteresis (N=3 consecutive distinct completed bars) to prevent texture whipping.
    """
    def __init__(
        self,
        recovery_bars: Optional[int] = None,
        texture_dwell_bars: Optional[int] = None,
        bar_interval_seconds: float = 300.0,
        max_gap_tolerance_seconds: float = 360.0,
        allow_unanchored_advance: bool = False,
    ):
        cfg_settings = getattr(config, "DATA_HEALTH_SETTINGS", {})
        # Recovery is explicitly measured in healthy poll cycles
        self._recovery_polls = recovery_bars if recovery_bars is not None else int(cfg_settings.get("recovery_bars", 3))
        self._texture_dwell_bars = texture_dwell_bars if texture_dwell_bars is not None else 3
        self._bar_interval_seconds = bar_interval_seconds
        self._max_gap_tolerance_seconds = max_gap_tolerance_seconds
        self._allow_unanchored_advance = allow_unanchored_advance
        self._consecutive_healthy = 0
        self._latched_degraded = False
        self._active_texture: Optional[str] = None
        self._candidate_texture: Optional[str] = None
        self._candidate_texture_count: int = 0
        self._last_processed_bar_dt: Optional[datetime] = None
        self._lock = threading.RLock()

    @property
    def is_degraded(self) -> bool:
        with self._lock:
            return self._latched_degraded

    @property
    def consecutive_healthy(self) -> int:
        with self._lock:
            return self._consecutive_healthy

    @property
    def active_texture(self) -> Optional[str]:
        with self._lock:
            return self._active_texture

    @property
    def candidate_texture(self) -> Optional[str]:
        with self._lock:
            return self._candidate_texture

    @property
    def candidate_texture_count(self) -> int:
        with self._lock:
            return self._candidate_texture_count

    @property
    def last_processed_bar_dt(self) -> Optional[datetime]:
        with self._lock:
            return self._last_processed_bar_dt

    def reset(self) -> None:
        with self._lock:
            self._consecutive_healthy = 0
            self._latched_degraded = False
            self._active_texture = None
            self._candidate_texture = None
            self._candidate_texture_count = 0
            self._last_processed_bar_dt = None

    def _validate_bar_advance(
        self,
        bar_dt: Optional[datetime],
        is_completed_bar_boundary: bool,
    ) -> Tuple[bool, bool]:
        """
        Validates whether incoming bar observation satisfies canonical grid alignment,
        monotonic sequence, and sub-interval/gap tolerances.
        Returns: (is_valid_advance, is_gap)
        """
        if not is_completed_bar_boundary or bar_dt is None:
            return False, False

        # 1. Canonical grid alignment: completed bars must sit on clean interval boundaries
        if self._bar_interval_seconds >= 60.0:
            step_min = int(self._bar_interval_seconds // 60)
            if bar_dt.second != 0 or (bar_dt.minute % step_min != 0):
                logger.warning(
                    "[TEXTURE HYSTERESIS] Off-grid bar timestamp rejected (%s not aligned to %dm grid). Watermark retained.",
                    bar_dt.isoformat(),
                    step_min,
                )
                return False, False
        elif self._bar_interval_seconds > 0.0:
            step_sec = int(self._bar_interval_seconds)
            if (bar_dt.second % step_sec) != 0:
                logger.warning(
                    "[TEXTURE HYSTERESIS] Off-grid bar timestamp rejected (%s not aligned to %ds grid). Watermark retained.",
                    bar_dt.isoformat(),
                    step_sec,
                )
                return False, False

        # 2. Bootstrap on first valid completed bar
        if self._last_processed_bar_dt is None:
            return True, False

        # 3. Monotonicity check: reject duplicate or out-of-order older bars
        if bar_dt <= self._last_processed_bar_dt:
            return False, False

        # 4. Strictly newer bar: enforce minimum interval floor (>= 80% or interval - 60s)
        delta_s = (bar_dt - self._last_processed_bar_dt).total_seconds()
        min_interval_s = max(self._bar_interval_seconds - 60.0, self._bar_interval_seconds * 0.8)
        if delta_s < min_interval_s:
            logger.warning(
                "[TEXTURE HYSTERESIS] Sub-interval bar rejected (delta %.1fs < min %.1fs). Watermark retained.",
                delta_s,
                min_interval_s,
            )
            return False, False

        is_gap = delta_s > self._max_gap_tolerance_seconds
        return True, is_gap

    def update(
        self,
        raw_result: MarketRegimeResult,
        completed_bar_ts: Optional[Union[str, datetime]] = None,
        is_completed_bar_boundary: bool = False,
    ) -> MarketRegimeResult:
        with self._lock:
            # ─────────────────────────────────────────────────────────────
            # 1. EMERGENCY RISK & DATA HEALTH GATES (EVERY POLL INVARIANT)
            # ─────────────────────────────────────────────────────────────
            if raw_result.mode == MODE_DATA_DEGRADED:
                self._consecutive_healthy = 0
                self._latched_degraded = True
                self._active_texture = None
                self._candidate_texture = None
                self._candidate_texture_count = 0
                # Preserve _last_processed_bar_dt across degraded events so old bars cannot be re-accepted upon recovery
                return raw_result

            if self._latched_degraded:
                self._consecutive_healthy += 1
                if self._consecutive_healthy < self._recovery_polls:
                    latched_reasons = [
                        f"DATA_DEGRADED (Cool-down recovery: {self._consecutive_healthy}/{self._recovery_polls} healthy polls)"
                    ] + raw_result.reasons
                    return replace(
                        raw_result,
                        mode=MODE_DATA_DEGRADED,
                        halt_new_entries=True,
                        allow_new_longs=False,
                        reasons=latched_reasons,
                        market_texture=TEXTURE_VOLATILE,
                    )
                self._latched_degraded = False
                logger.info(
                    "[REGIME TRACKER] Data health restored after %d consecutive healthy polls; unlatching %s",
                    self._consecutive_healthy,
                    raw_result.mode,
                )

            # ─────────────────────────────────────────────────────────────
            # 2. PARSE & VALIDATE COMPLETED BAR TIMESTAMP
            # ─────────────────────────────────────────────────────────────
            bar_dt: Optional[datetime] = None
            if completed_bar_ts is not None:
                if isinstance(completed_bar_ts, datetime):
                    bar_dt = completed_bar_ts if completed_bar_ts.tzinfo else completed_bar_ts.replace(tzinfo=timezone.utc)
                elif isinstance(completed_bar_ts, str):
                    try:
                        dt = datetime.fromisoformat(completed_bar_ts.replace("Z", "+00:00"))
                        bar_dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                    except Exception:
                        bar_dt = None

            # ─────────────────────────────────────────────────────────────
            # 3. TEXTURE DWELL HYSTERESIS (DISTINCT COMPLETED BARS ONLY)
            # ─────────────────────────────────────────────────────────────
            raw_texture = raw_result.market_texture

            # Strict mode invariant: unanchored calls (no timestamp) do NOT advance dwell
            if completed_bar_ts is None:
                if self._allow_unanchored_advance:
                    is_valid_advance = True
                    is_gap = False
                else:
                    is_valid_advance = False
                    is_gap = False
            else:
                is_valid_advance, is_gap = self._validate_bar_advance(bar_dt, is_completed_bar_boundary)

            if is_valid_advance:
                if self._active_texture is None:
                    # Bootstrap on first valid completed bar
                    self._active_texture = raw_texture
                    self._candidate_texture = raw_texture
                    self._candidate_texture_count = 1
                elif is_gap:
                    # Missing bar gap resets consecutive streak
                    logger.warning(
                        "[TEXTURE HYSTERESIS] Bar interval gap detected (delta > %.1fs). Resetting consecutive counter streak.",
                        self._max_gap_tolerance_seconds,
                    )
                    self._candidate_texture = raw_texture
                    self._candidate_texture_count = 1
                elif raw_texture == self._active_texture:
                    self._candidate_texture = raw_texture
                    self._candidate_texture_count = 0
                else:
                    if raw_texture == self._candidate_texture:
                        self._candidate_texture_count += 1
                    else:
                        self._candidate_texture = raw_texture
                        self._candidate_texture_count = 1

                    if self._candidate_texture_count >= self._texture_dwell_bars:
                        logger.info(
                            "[TEXTURE HYSTERESIS] Texture transition confirmed: %s -> %s after %d consecutive distinct bars",
                            self._active_texture,
                            self._candidate_texture,
                            self._candidate_texture_count,
                        )
                        self._active_texture = self._candidate_texture
                        self._candidate_texture_count = 0
                    else:
                        logger.debug(
                            "[TEXTURE HYSTERESIS] Dwell window active (%d/%d bars for %s). Retaining confirmed texture: %s",
                            self._candidate_texture_count,
                            self._texture_dwell_bars,
                            self._candidate_texture,
                            self._active_texture,
                        )

                # CRITICAL WATERMARK INVARIANT:
                # Always update watermark immediately upon accepting valid completed bar observation
                if bar_dt is not None:
                    self._last_processed_bar_dt = bar_dt

            confirmed_texture = self._active_texture if self._active_texture is not None else raw_texture

            if confirmed_texture != raw_result.market_texture:
                halt = raw_result.halt_new_entries
                allow_long = raw_result.allow_new_longs
                tex_cfg = getattr(config, "MARKET_REGIME_SETTINGS", {}).get(confirmed_texture, {})
                if tex_cfg.get("halt_new_entries", False):
                    halt = True
                    allow_long = False
                return replace(
                    raw_result,
                    market_texture=confirmed_texture,
                    halt_new_entries=halt,
                    allow_new_longs=allow_long,
                )

            return raw_result


