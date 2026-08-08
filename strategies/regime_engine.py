"""strategies/regime_engine.py

Pre-Market Market Health Gate & Market Regime Engine for Stock Engine Pro V3.

Evaluates macro market conditions, index gaps, sector shock indicators,
and scanner candidate breadth to dynamically classify market regime into:
  - RISK_ON (Score >= 0): Normal Sizing, Normal Confidence, Normal Entries.
  - CAUTION (Score -1 to -3): 15-Min Opening Entry Buffer (No BUY before 09:45 ET), 50% Sizing, +0.10 Confidence Gate.
  - RISK_OFF (Score <= -4): NO NEW LONG DAY TRADES. Exits and protective actions remain fully enabled.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import config
from strategies.session import now_eastern

logger = logging.getLogger(__name__)

MODE_RISK_ON = "RISK_ON"
MODE_CAUTION = "CAUTION"
MODE_RISK_OFF = "RISK_OFF"


@dataclass
class MarketRegimeResult:
    mode: str  # "RISK_ON", "CAUTION", "RISK_OFF"
    score: int
    allow_new_longs: bool
    requires_opening_buffer: bool  # True if must wait until 09:45 ET
    position_scale: float  # 1.0 for RISK_ON, 0.5 for CAUTION, 0.0 for RISK_OFF
    confidence_boost: float  # +0.0 for RISK_ON, +0.15 for CAUTION
    allow_new_shorts: bool = True  # True if short selling allowed for current regime
    reasons: List[str] = field(default_factory=list)
    factor_scores: Dict[str, int] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: Optional[datetime] = None
    scanner_candidate_count: Optional[int] = None
    max_trades_day: Optional[int] = None
    halt_new_entries: bool = False

    def summary(self) -> str:
        reasons_str = " | ".join(self.reasons) if self.reasons else "Normal Healthy Market"
        return (
            f"[REGIME] mode={self.mode} score={self.score:+d} allow_longs={self.allow_new_longs} "
            f"halt_entries={self.halt_new_entries} max_trades={self.max_trades_day} "
            f"open_buffer_0945={self.requires_opening_buffer} sizing={self.position_scale*100:.0f}% "
            f"boost=+{self.confidence_boost:.2f} reasons=[{reasons_str}]"
        )


def _get_premarket_pct_change(symbol: str, bridge=None) -> tuple[Optional[float], str]:
    """Helper to fetch pre-market or current daily % change for a benchmark symbol.
    Returns (pct_change, data_source_str).
    """
    # 1. Try Alpaca bridge / data client
    if bridge is not None:
        try:
            if hasattr(bridge, "get_price"):
                price = bridge.get_price(symbol, allow_historical=True)
                if price > 0:
                    prev_close = None
                    if hasattr(bridge, "_data_client") and bridge._data_client is not None:
                        try:
                            from alpaca.data.requests import StockBarsRequest
                            from alpaca.data.timeframe import TimeFrame
                            from datetime import datetime, timedelta, timezone
                            now_utc = datetime.now(timezone.utc)
                            req = StockBarsRequest(
                                symbol_or_symbols=symbol,
                                timeframe=TimeFrame.Day,
                                start=now_utc - timedelta(days=7),
                                end=now_utc
                            )
                            bars = bridge._data_client.get_stock_bars(req)
                            symbol_bars = bars.get(symbol, []) if bars else []
                            if len(symbol_bars) >= 2:
                                prev_close = float(symbol_bars[-2].close)
                            elif len(symbol_bars) == 1:
                                prev_close = float(symbol_bars[0].open)
                        except Exception:
                            pass
                    if prev_close and prev_close > 0:
                        pct = ((price - prev_close) / prev_close) * 100.0
                        return round(pct, 2), "alpaca_quote_vs_prev_close"
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
                return round(pct, 2), "yfinance_realtime"
    except Exception as yf_err:
        logger.debug("yfinance fallback failed for %s: %s", symbol, yf_err)

    return None, "data_unavailable"


def evaluate_market_regime(bridge=None, scanner_candidate_count: Optional[int] = None, symbols: Optional[List[str]] = None) -> MarketRegimeResult:
    """Evaluates market health metrics and determines current trading regime.
    
    Factors & Scoring:
      - SPY premarket < -0.75%: -2 pts
      - QQQ premarket < -1.00%: -2 pts
      - Chip ETF (SOXX) < -2.00%: -2 pts
      - Dynamic Scanner Candidates == 0: -2 pts
    """
    score = 0
    reasons = []
    factor_scores = {}
    data = {}

    # 1. Benchmark Index Changes (SPY / QQQ)
    spy_pct, spy_source = _get_premarket_pct_change("SPY", bridge)
    data["SPY"] = {"pct": spy_pct, "source": spy_source}
    if spy_pct is not None:
        if spy_pct < -0.75:
            factor_scores["SPY"] = -2
            reasons.append(f"SPY benchmark down {spy_pct:.2f}% (< -0.75%) [{spy_source}]")
        elif spy_pct < -0.40:
            factor_scores["SPY"] = -1
            reasons.append(f"SPY benchmark weak {spy_pct:.2f}% [{spy_source}]")
        else:
            factor_scores["SPY"] = 0
    else:
        reasons.append("SPY_data_unavailable")

    qqq_pct, qqq_source = _get_premarket_pct_change("QQQ", bridge)
    data["QQQ"] = {"pct": qqq_pct, "source": qqq_source}
    if qqq_pct is not None:
        if qqq_pct < -1.00:
            factor_scores["QQQ"] = -2
            reasons.append(f"QQQ tech index down {qqq_pct:.2f}% (< -1.00%) [{qqq_source}]")
        elif qqq_pct < -0.50:
            factor_scores["QQQ"] = -1
            reasons.append(f"QQQ tech index weak {qqq_pct:.2f}% [{qqq_source}]")
        else:
            factor_scores["QQQ"] = 0
    else:
        reasons.append("QQQ_data_unavailable")

    # 2. Chip Sector Shock Check (SOXX)
    soxx_pct, soxx_source = _get_premarket_pct_change("SOXX", bridge)
    data["SOXX"] = {"pct": soxx_pct, "source": soxx_source}
    if soxx_pct is not None:
        if soxx_pct < -2.00:
            factor_scores["SOXX"] = -2
            reasons.append(f"SOXX semiconductor ETF down {soxx_pct:.2f}% (< -2.00%) [{soxx_source}]")
        elif soxx_pct < -1.00:
            factor_scores["SOXX"] = -1
            reasons.append(f"SOXX semiconductor ETF weak {soxx_pct:.2f}% [{soxx_source}]")
        else:
            factor_scores["SOXX"] = 0
    else:
        reasons.append("SOXX_data_unavailable")

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
        confidence_boost = 0.15  # ★ Fix ①: WAS 0.10, NOW 0.15

        # ★ Fix ②: Dynamic max trades per day based on scanner health & Paper/Live mode
        # In Paper Trading, allow up to 50 trades per day for maximum trade opportunity evaluation
        is_paper = getattr(config, "IS_PAPER_TRADING", True) or getattr(config, "PAPER_TRADING", True)
        if is_paper:
            max_trades_day = 50
        elif scanner_candidate_count is not None and scanner_candidate_count == 0:
            max_trades_day = 20  # Live CAUTION + scanner empty → 20
        else:
            max_trades_day = 30  # Live CAUTION + scanner healthy → 30

        # ★ Fix ③: Score <= -3 → Halt new entries
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

    # Short selling regime eligibility:
    # Only allow short selling if ALLOW_SHORT is True AND macro benchmark index (SPY/QQQ) is actually weak/down.
    # Prevents counter-trend short entries in a strong Bull Market environment.
    allow_short_cfg = getattr(config, "ALLOW_SHORT", False)
    macro_index_weak = (spy_pct is not None and spy_pct < -0.30) or (qqq_pct is not None and qqq_pct < -0.40) or (score <= -3)
    allow_new_shorts = allow_short_cfg and macro_index_weak

    res = MarketRegimeResult(
        mode=mode,
        score=score,
        allow_new_longs=allow_new_longs,
        requires_opening_buffer=requires_opening_buffer,
        position_scale=position_scale,
        confidence_boost=confidence_boost,
        allow_new_shorts=allow_new_shorts,
        reasons=reasons,
        factor_scores=factor_scores,
        data=data,
        evaluated_at=now_eastern(),
        scanner_candidate_count=scanner_candidate_count,
        max_trades_day=max_trades_day,
        halt_new_entries=halt_new_entries
    )
    logger.info("Market Regime Evaluation: %s", res.summary())
    return res
