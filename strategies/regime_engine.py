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
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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

    def summary(self) -> str:
        reasons_str = " | ".join(self.reasons) if self.reasons else "Normal Healthy Market"
        return (
            f"[REGIME] mode={self.mode} texture={self.market_texture} score={self.score:+d} "
            f"allow_longs={self.allow_new_longs} halt_entries={self.halt_new_entries} "
            f"max_trades={self.max_trades_day} open_buffer_0945={self.requires_opening_buffer} "
            f"sizing={self.position_scale*100:.0f}% boost=+{self.confidence_boost:.2f} reasons=[{reasons_str}]"
        )


def _get_premarket_pct_change(symbol: str, bridge=None) -> Tuple[Optional[float], str, Optional[float]]:
    """Helper to fetch pre-market or current daily % change and quote epoch for a benchmark symbol.
    Returns (pct_change, data_source_str, quote_epoch_ts).
    """
    now_epoch = time.time()

    # 1. Try Alpaca bridge / data client
    if bridge is not None:
        try:
            # Check for quote object first to obtain exact broker timestamp if present
            quote_ts = None
            if hasattr(bridge, "get_quote"):
                try:
                    q = bridge.get_quote(symbol)
                    if q:
                        raw_ts = getattr(q, "timestamp", None) if not isinstance(q, dict) else q.get("timestamp")
                        if isinstance(raw_ts, (int, float)):
                            quote_ts = float(raw_ts)
                        elif isinstance(raw_ts, datetime):
                            quote_ts = raw_ts.timestamp()
                except Exception:
                    pass

            if hasattr(bridge, "get_price"):
                price = bridge.get_price(symbol, allow_historical=True)
                if price > 0:
                    prev_close = None
                    if hasattr(bridge, "fetch_historical_data"):
                        try:
                            bars = bridge.fetch_historical_data(symbol)
                            if bars and len(bars) >= 2:
                                prev_close = float(getattr(bars[-2], "close", 0.0) or (bars[-2]["close"] if isinstance(bars[-2], dict) else 0.0))
                                bar_ts = getattr(bars[-1], "timestamp", None) or getattr(bars[-1], "date", None)
                                if bar_ts and quote_ts is None:
                                    if isinstance(bar_ts, (int, float)):
                                        quote_ts = float(bar_ts)
                                    elif isinstance(bar_ts, datetime):
                                        quote_ts = bar_ts.timestamp()
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
                            if len(symbol_bars) >= 2:
                                prev_close = float(symbol_bars[-2].close)
                                quote_ts = quote_ts or symbol_bars[-1].timestamp.timestamp()
                        except Exception:
                            pass

                    if prev_close and prev_close > 0:
                        pct = ((price - prev_close) / prev_close) * 100.0
                        return round(pct, 2), "alpaca_quote_vs_prev_close", quote_ts or now_epoch
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
                return round(pct, 2), "yfinance_realtime", now_epoch
    except Exception as yf_err:
        logger.debug("yfinance fallback failed for %s: %s", symbol, yf_err)

    return None, "data_unavailable", None


def _check_data_health(bridge=None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Pre-flight evaluation of freshness and presence for critical macro benchmark data (SPY/QQQ).
    Fail-closed: Returns (is_healthy=False, reason, data) upon missing data, stale timestamps, or exception.
    Freshness math strictly uses epoch float (time.time()) to remain timezone safe.
    """
    health_data: Dict[str, Any] = {}
    try:
        health_cfg = getattr(config, "DATA_HEALTH_SETTINGS", {})
        critical_symbols = health_cfg.get("critical_symbols", ["SPY", "QQQ"])
        stale_thresholds = health_cfg.get("stale_threshold_s", {})
        default_stale_thresh = float(stale_thresholds.get("_default", 180.0))
        now_epoch = time.time()

        for sym in critical_symbols:
            pct, source, ts = _get_premarket_pct_change(sym, bridge)
            health_data[sym] = {"pct": pct, "source": source, "timestamp": ts}
            if pct is None:
                return False, f"{sym} benchmark data unavailable", health_data

            thresh_s = float(stale_thresholds.get(sym, default_stale_thresh))
            if ts is not None:
                age_s = now_epoch - ts
                if age_s > thresh_s:
                    return False, f"{sym} benchmark data stale ({age_s:.1f}s > {thresh_s:.1f}s threshold)", health_data

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
      - Chip ETF (SOXX) < -2.00%: -2 pts, < -1.00%: -1 pt, > +2.00%: +1 pt (Symmetric, 0 pts if unavailable)
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

    # 1. Benchmark Index Changes (SPY / QQQ) — guaranteed available and fresh by DataHealth gate
    spy_pct = health_data["SPY"]["pct"]
    spy_source = health_data["SPY"]["source"]
    if spy_pct < -0.75:
        factor_scores["SPY"] = -2
        reasons.append(f"SPY benchmark down {spy_pct:.2f}% (< -0.75%) [{spy_source}]")
    elif spy_pct < -0.40:
        factor_scores["SPY"] = -1
        reasons.append(f"SPY benchmark weak {spy_pct:.2f}% [{spy_source}]")
    else:
        factor_scores["SPY"] = 0

    qqq_pct = health_data["QQQ"]["pct"]
    qqq_source = health_data["QQQ"]["source"]
    if qqq_pct < -1.00:
        factor_scores["QQQ"] = -2
        reasons.append(f"QQQ tech index down {qqq_pct:.2f}% (< -1.00%) [{qqq_source}]")
    elif qqq_pct < -0.50:
        factor_scores["QQQ"] = -1
        reasons.append(f"QQQ tech index weak {qqq_pct:.2f}% [{qqq_source}]")
    else:
        factor_scores["QQQ"] = 0

    # 2. Chip Sector Shock Check (SOXX) — Optional sector indicator (no penalty when missing, symmetric scoring)
    soxx_pct, soxx_source, _ = _get_premarket_pct_change("SOXX", bridge)
    data["SOXX"] = {"pct": soxx_pct, "source": soxx_source}
    if soxx_pct is not None:
        if soxx_pct < -3.00:
            factor_scores["SOXX"] = -2
            reasons.append(f"SOXX semiconductor ETF down {soxx_pct:.2f}% (< -3.00% severe) [{soxx_source}]")
        elif soxx_pct < -1.50:
            factor_scores["SOXX"] = -1
            reasons.append(f"SOXX semiconductor ETF weak {soxx_pct:.2f}% (< -1.50%) [{soxx_source}]")
        elif soxx_pct > 2.00:
            factor_scores["SOXX"] = 1
            reasons.append(f"SOXX semiconductor ETF strong +{soxx_pct:.2f}% [{soxx_source}]")
        else:
            factor_scores["SOXX"] = 0
    else:
        reasons.append("SOXX_data_unavailable (optional sector signal, zero penalty)")

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
        confidence_boost = float(getattr(config, "CAUTION_CONFIDENCE_BOOST", 0.0))

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
        market_texture=market_texture
    )

    logger.info("Market Regime Evaluation: %s", res.summary())
    return res


class RegimeStateTracker:
    """
    Thread-safe hysteresis state tracker wrapping evaluate_market_regime().
    Maintains recovery cool-down to prevent flapping between DATA_DEGRADED and normal modes.
    """
    def __init__(self, recovery_bars: Optional[int] = None):
        cfg_settings = getattr(config, "DATA_HEALTH_SETTINGS", {})
        self._recovery_bars = recovery_bars if recovery_bars is not None else int(cfg_settings.get("recovery_bars", 3))
        self._consecutive_healthy = 0
        self._latched_degraded = False
        self._lock = threading.RLock()

    @property
    def is_degraded(self) -> bool:
        with self._lock:
            return self._latched_degraded

    @property
    def consecutive_healthy(self) -> int:
        with self._lock:
            return self._consecutive_healthy

    def reset(self) -> None:
        with self._lock:
            self._consecutive_healthy = 0
            self._latched_degraded = False

    def update(self, raw_result: MarketRegimeResult) -> MarketRegimeResult:
        with self._lock:
            if raw_result.mode == MODE_DATA_DEGRADED:
                self._consecutive_healthy = 0
                self._latched_degraded = True
                return raw_result

            if self._latched_degraded:
                self._consecutive_healthy += 1
                if self._consecutive_healthy < self._recovery_bars:
                    # Still in cool-down recovery window -> keep degraded state latched
                    latched_reasons = [
                        f"DATA_DEGRADED (Cool-down recovery: {self._consecutive_healthy}/{self._recovery_bars} fresh bars)"
                    ] + raw_result.reasons
                    return MarketRegimeResult(
                        mode=MODE_DATA_DEGRADED,
                        score=0,
                        allow_new_longs=False,
                        requires_opening_buffer=True,
                        position_scale=0.0,
                        confidence_boost=0.0,
                        allow_new_shorts=False,
                        allow_inverse_etf_longs=False,
                        reasons=latched_reasons,
                        factor_scores=raw_result.factor_scores,
                        data=raw_result.data,
                        evaluated_at=raw_result.evaluated_at,
                        scanner_candidate_count=raw_result.scanner_candidate_count,
                        max_trades_day=0,
                        halt_new_entries=True,
                        market_texture=TEXTURE_VOLATILE,
                    )
                # Reached recovery threshold -> unlatch
                self._latched_degraded = False
                logger.info(
                    "[REGIME TRACKER] Data health restored after %d consecutive healthy bars; unlatching %s",
                    self._consecutive_healthy,
                    raw_result.mode,
                )

            return raw_result


