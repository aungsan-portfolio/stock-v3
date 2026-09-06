"""
strategies/regime_classifier.py — Mutually Exclusive & Collectively Exhaustive Market Regime Engine

Partitions market state into deterministic, point-in-time reproducible regimes
covering the complete parameter space with strict precedence ordering and DATA_INVALID handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class MarketRegime(str, Enum):
    DATA_INVALID = "DATA_INVALID"
    TRANSITIONING = "TRANSITIONING"
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    SIDEWAYS_LOW_VOL = "SIDEWAYS_LOW_VOL"
    SIDEWAYS_NORMAL_VOL = "SIDEWAYS_NORMAL_VOL"
    SIDEWAYS_HIGH_VOL = "SIDEWAYS_HIGH_VOL"


@dataclass(frozen=True)
class RegimeClassification:
    regime: MarketRegime
    adx: float
    vix: float
    spy_price: float
    spy_sma50: float
    sma50_slope: float
    is_sustained_bear: bool = False
    consecutive_bear_sessions: int = 0
    allows_new_entries: bool = True
    reason: str = ""


def classify_market_regime(
    adx: Optional[float],
    vix: Optional[float],
    spy_price: Optional[float],
    spy_sma50: Optional[float],
    sma50_slope: Optional[float],
    recent_regimes: Optional[List[MarketRegime]] = None,
    consecutive_bear_sessions: int = 0,
    quote_age_seconds: float = 0.0,
    max_quote_age_seconds: float = 60.0,
) -> RegimeClassification:
    """
    Classify macro market regime with deterministic precedence ordering:
      1. DATA_INVALID: Missing, non-finite, or stale inputs -> entries prohibited
      2. TRANSITIONING (Boundary Cross Override): Rapid shifts across last 5 bars
      3. TRANSITIONING (Conflicting / Intermediate): 20 < ADX < 25 OR conflicting trend indicators
      4. TRENDING_BULL: ADX >= 25, SPY > SMA50, SMA50 slope > 0
      5. TRENDING_BEAR: ADX >= 25, SPY < SMA50, SMA50 slope < 0
      6. SIDEWAYS_LOW_VOL: ADX <= 20, VIX < 18
      7. SIDEWAYS_NORMAL_VOL: ADX <= 20, 18 <= VIX < 25
      8. SIDEWAYS_HIGH_VOL: ADX <= 20, VIX >= 25
    """
    inputs = [
        (adx, "ADX"),
        (vix, "VIX"),
        (spy_price, "SPY price"),
        (spy_sma50, "SPY SMA50"),
        (sma50_slope, "SMA50 slope"),
    ]

    # Precedence 1: DATA_INVALID check
    for val, name in inputs:
        if val is None or not math.isfinite(val):
            return RegimeClassification(
                regime=MarketRegime.DATA_INVALID,
                adx=float(adx or 0.0),
                vix=float(vix or 0.0),
                spy_price=float(spy_price or 0.0),
                spy_sma50=float(spy_sma50 or 0.0),
                sma50_slope=float(sma50_slope or 0.0),
                is_sustained_bear=False,
                consecutive_bear_sessions=0,
                allows_new_entries=False,
                reason=f"DATA_INVALID: {name} is missing or non-finite ({val})",
            )

    if quote_age_seconds > max_quote_age_seconds:
        return RegimeClassification(
            regime=MarketRegime.DATA_INVALID,
            adx=float(adx),
            vix=float(vix),
            spy_price=float(spy_price),
            spy_sma50=float(spy_sma50),
            sma50_slope=float(sma50_slope),
            is_sustained_bear=False,
            consecutive_bear_sessions=0,
            allows_new_entries=False,
            reason=f"DATA_INVALID: Stale quote age {quote_age_seconds:.1f}s > {max_quote_age_seconds:.1f}s",
        )

    # Cast finite floats
    f_adx = float(adx)
    f_vix = float(vix)
    f_spy = float(spy_price)
    f_sma = float(spy_sma50)
    f_slope = float(sma50_slope)

    # Precedence 2: Boundary Cross Override (Rapid shifts across last 5 bars)
    if recent_regimes and len(recent_regimes) >= 2:
        distinct_recent = {r for r in recent_regimes[-5:] if r != MarketRegime.DATA_INVALID}
        if len(distinct_recent) >= 3:
            return RegimeClassification(
                regime=MarketRegime.TRANSITIONING,
                adx=f_adx,
                vix=f_vix,
                spy_price=f_spy,
                spy_sma50=f_sma,
                sma50_slope=f_slope,
                is_sustained_bear=False,
                consecutive_bear_sessions=0,
                allows_new_entries=False, # Restricted per workspace rule
                reason="TRANSITIONING: Boundary cross override (>=3 distinct regimes in last 5 bars)",
            )

    # Precedence 3: Intermediate ADX boundary zone (20 < ADX < 25)
    if 20.0 < f_adx < 25.0:
        return RegimeClassification(
            regime=MarketRegime.TRANSITIONING,
            adx=f_adx,
            vix=f_vix,
            spy_price=f_spy,
            spy_sma50=f_sma,
            sma50_slope=f_slope,
            is_sustained_bear=False,
            consecutive_bear_sessions=0,
            allows_new_entries=False,
            reason=f"TRANSITIONING: Intermediate ADX boundary zone (20 < {f_adx:.1f} < 25)",
        )

    # Precedence 4 & 5: Trending Regimes (ADX >= 25)
    if f_adx >= 25.0:
        if f_spy > f_sma and f_slope > 0.0:
            return RegimeClassification(
                regime=MarketRegime.TRENDING_BULL,
                adx=f_adx,
                vix=f_vix,
                spy_price=f_spy,
                spy_sma50=f_sma,
                sma50_slope=f_slope,
                is_sustained_bear=False,
                consecutive_bear_sessions=0,
                allows_new_entries=True,
                reason="TRENDING_BULL: ADX>=25, SPY>SMA50, slope>0",
            )
        elif f_spy < f_sma and f_slope < 0.0:
            new_bear_sessions = consecutive_bear_sessions + 1
            return RegimeClassification(
                regime=MarketRegime.TRENDING_BEAR,
                adx=f_adx,
                vix=f_vix,
                spy_price=f_spy,
                spy_sma50=f_sma,
                sma50_slope=f_slope,
                is_sustained_bear=(new_bear_sessions >= 10),
                consecutive_bear_sessions=new_bear_sessions,
                allows_new_entries=False, # Long-only swing suppresses entries in bear
                reason=f"TRENDING_BEAR: ADX>=25, SPY<SMA50, slope<0 (session {new_bear_sessions})",
            )
        else:
            # Conflicting trend indicators (e.g. price > SMA50 but slope <= 0, or price == SMA50)
            return RegimeClassification(
                regime=MarketRegime.TRANSITIONING,
                adx=f_adx,
                vix=f_vix,
                spy_price=f_spy,
                spy_sma50=f_sma,
                sma50_slope=f_slope,
                is_sustained_bear=False,
                consecutive_bear_sessions=0,
                allows_new_entries=False,
                reason="TRANSITIONING: Conflicting trend indicators (price/slope mismatch)",
            )

    # Precedence 6, 7, 8: Sideways Regimes (ADX <= 20)
    if f_adx <= 20.0:
        if f_vix < 18.0:
            reg = MarketRegime.SIDEWAYS_LOW_VOL
            r_str = "SIDEWAYS_LOW_VOL: ADX<=20, VIX<18"
        elif 18.0 <= f_vix < 25.0:
            reg = MarketRegime.SIDEWAYS_NORMAL_VOL
            r_str = "SIDEWAYS_NORMAL_VOL: ADX<=20, 18<=VIX<25"
        else:
            reg = MarketRegime.SIDEWAYS_HIGH_VOL
            r_str = "SIDEWAYS_HIGH_VOL: ADX<=20, VIX>=25"

        return RegimeClassification(
            regime=reg,
            adx=f_adx,
            vix=f_vix,
            spy_price=f_spy,
            spy_sma50=f_sma,
            sma50_slope=f_slope,
            is_sustained_bear=False,
            consecutive_bear_sessions=0,
            allows_new_entries=True,
            reason=r_str,
        )

    # Fallback (mathematically unreachable due to prior branch coverage)
    return RegimeClassification(
        regime=MarketRegime.TRANSITIONING,
        adx=f_adx,
        vix=f_vix,
        spy_price=f_spy,
        spy_sma50=f_sma,
        sma50_slope=f_slope,
        is_sustained_bear=False,
        consecutive_bear_sessions=0,
        allows_new_entries=False,
        reason="TRANSITIONING: Fallback state",
    )
