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
    confidence_boost: float  # +0.0 for RISK_ON, +0.10 for CAUTION
    reasons: List[str] = field(default_factory=list)
    factor_scores: Dict[str, int] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: Optional[datetime] = None
    scanner_candidate_count: Optional[int] = None

    def summary(self) -> str:
        reasons_str = " | ".join(self.reasons) if self.reasons else "Normal Healthy Market"
        return (
            f"[REGIME] mode={self.mode} score={self.score:+d} allow_longs={self.allow_new_longs} "
            f"open_buffer_0945={self.requires_opening_buffer} sizing={self.position_scale*100:.0f}% "
            f"reasons=[{reasons_str}]"
        )


def _get_premarket_pct_change(symbol: str, bridge=None) -> tuple[Optional[float], str]:
    """Helper to fetch pre-market or current daily % change for a benchmark symbol.
    Returns (pct_change, data_source_str).
    """
    if bridge is None:
        return None, "no_bridge"
    try:
        if hasattr(bridge, "get_price"):
            price = bridge.get_price(symbol, allow_historical=True)
            daily_bars = getattr(bridge, "fetch_historical_data", lambda s, l: [])(symbol, 2)
            if price > 0 and len(daily_bars) >= 2:
                prev_close = float(getattr(daily_bars[-2], "close", 0.0) or 0.0)
                if prev_close > 0:
                    pct = ((price - prev_close) / prev_close) * 100.0
                    return round(pct, 2), "live_quote_vs_prev_close"
    except Exception as exc:
        logger.debug("Failed fetching premarket pct for %s: %s", symbol, exc)
    return None, "data_unavailable"


def evaluate_market_regime(bridge=None, scanner_candidate_count: Optional[int] = None) -> MarketRegimeResult:
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
    if score <= -4:
        mode = MODE_RISK_OFF
        allow_new_longs = False
        requires_opening_buffer = True
        position_scale = 0.0
        confidence_boost = 0.20
    elif score <= -1:
        mode = MODE_CAUTION
        allow_new_longs = True
        requires_opening_buffer = True  # Must wait until 09:45 ET
        position_scale = 0.5
        confidence_boost = 0.10
    else:
        mode = MODE_RISK_ON
        allow_new_longs = True
        requires_opening_buffer = False
        position_scale = 1.0
        confidence_boost = 0.0

    res = MarketRegimeResult(
        mode=mode,
        score=score,
        allow_new_longs=allow_new_longs,
        requires_opening_buffer=requires_opening_buffer,
        position_scale=position_scale,
        confidence_boost=confidence_boost,
        reasons=reasons,
        factor_scores=factor_scores,
        data=data,
        evaluated_at=now_eastern(),
        scanner_candidate_count=scanner_candidate_count
    )
    logger.info("Market Regime Evaluation: %s", res.summary())
    return res
