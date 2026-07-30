"""
minervini.py — Minervini / SEPA paper-overlay detectors (PURE, OFFLINE).

This is an ADDITIVE, beginner-safe overlay on top of the existing
RF+LSTM+Technical ensemble. It computes Mark-Minervini-style setup signals from a
plain OHLCV DataFrame and returns a verdict object. It is intentionally:

* PURE / OFFLINE — it imports only numpy/pandas/config. It never touches the
  order path, the IBKR bridge, the risk engine, the scheduler, yfinance, or any
  network/broker resource, so it is trivially unit-testable with synthetic data.
* NON-BLOCKING by itself — this module only DESCRIBES a setup. Wiring a verdict
  into the BUY entry gate / sizing / coach happens in later milestones (M1+).
  Nothing here can place, cancel, or modify an order, or weaken any safety guard.
* NO-LOOKAHEAD — every detector evaluates the last *closed* bar. When
  ``closed_bar=True`` (the default) the most recent (possibly still-forming) row
  is dropped, so an intraday run can never peek at an unfinished bar.

The VCP detector is a deliberately COARSE heuristic and is labelled everywhere as
a "VCP-like approximation" — it is NOT a faithful reproduction of the manual
O'Neil/Minervini Volatility Contraction Pattern.

The frozen 14-feature ML list (data_manager.get_feature_columns) is NOT touched:
these are parallel overlay computations, never model inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

import config


# ── Verdict ──────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MinerviniVerdict:
    """The result of evaluating one symbol's OHLCV against the overlay.

    stage2_ok      : True when the Stage-2 trend template (plus a None-tolerant
                     relative-strength check) passes. This is the value a later
                     milestone consults to (optionally) block a NEW BUY.
    reasons        : machine tokens explaining a failure (empty when stage2_ok),
                     e.g. ["ma_order_fail", "below_52w_high", "rs_below_min"].
    pivot_low      : the VCP-like pivot low (a chart price level) used as the
                     anchor for 1R sizing / a dedicated setup stop. None when it
                     cannot be computed.
    rs_rank        : approximate SPY-relative relative-strength percentile (0-100)
                     or None when no benchmark was supplied / not computable.
    vcp_like       : True when the coarse "VCP-like approximation" qualifies.
    pocket_pivot   : True when the last closed bar is a pocket-pivot up day.
    detail         : numeric scratch (MA values, % from high, contractions, ...)
                     for the coach / preview. Never used for control flow.
    """
    stage2_ok: bool
    reasons: List[str]
    pivot_low: Optional[float]
    rs_rank: Optional[float]
    vcp_like: bool
    pocket_pivot: bool
    detail: dict = field(default_factory=dict)


# ── Internal helpers ─────────────────────────────────────────────────────────
def _cfg(name: str, default):
    """Read a config constant with a safe fallback (matches the codebase's
    getattr(config, ...) defensive pattern so tests can patch/omit a knob)."""
    return getattr(config, name, default)


def _closed(df: Optional[pd.DataFrame], closed_bar: bool) -> Optional[pd.DataFrame]:
    """Return the frame restricted to CLOSED bars.

    With ``closed_bar=True`` the final row is treated as possibly still forming
    and dropped (compute on iloc[:-1], i.e. the last closed bar is iloc[-2] of the
    original). This is the no-lookahead guarantee. A frame with <2 rows is
    returned unchanged (dropping would leave nothing to evaluate).
    """
    if df is None or len(df) == 0:
        return df
    if closed_bar and len(df) >= 2:
        return df.iloc[:-1]
    return df


def _has_cols(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and all(c in df.columns for c in ("Open", "High", "Low", "Close", "Volume"))


# ── Stage-2 trend template ───────────────────────────────────────────────────
def trend_template(df: pd.DataFrame, closed_bar: bool = True):
    """Minervini Stage-2 "Trend Template" (criteria 1-7; relative strength,
    criterion 8, is evaluated separately and None-tolerantly in evaluate_entry).

    Returns (ok: bool, reasons: list[str], detail: dict). ``ok`` is True only when
    ``reasons`` is empty. Conservative on missing history: a frame too short to
    confirm the 200-day SMA (and its slope) yields ok=False / "insufficient_history"
    — a NEW BUY should not be confirmed on a setup that cannot be verified.
    """
    detail: dict = {}
    cf = _closed(df, closed_bar)
    if not _has_cols(cf) or len(cf) < 2:
        return False, ["insufficient_history"], detail

    close = cf["Close"].astype(float)
    high = cf["High"].astype(float)
    low = cf["Low"].astype(float)
    n = len(close)

    rising_lb = int(_cfg("MINERVINI_MA_200_RISING_LOOKBACK", 20))
    need = 200 + rising_lb
    if n < need:
        return False, ["insufficient_history"], detail

    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()
    s50, s150, s200 = sma50.iloc[-1], sma150.iloc[-1], sma200.iloc[-1]
    s200_past = sma200.iloc[-1 - rising_lb]
    if any(pd.isna(v) for v in (s50, s150, s200, s200_past)):
        return False, ["insufficient_history"], detail

    price = float(close.iloc[-1])
    win = min(252, n)
    hi_52 = float(high.iloc[-win:].max())
    lo_52 = float(low.iloc[-win:].min())

    near_high_pct = float(_cfg("MINERVINI_NEAR_52W_HIGH_PCT", 0.25))
    off_low_pct = float(_cfg("MINERVINI_OFF_52W_LOW_PCT", 0.30))

    reasons: List[str] = []
    if not price > s200:
        reasons.append("price_below_200ma")
    if not price > s150:
        reasons.append("price_below_150ma")
    if not price > s50:
        reasons.append("price_below_50ma")
    if not (s50 > s150 > s200):
        reasons.append("ma_order_fail")
    if not s200 > s200_past:
        reasons.append("ma200_not_rising")
    if hi_52 > 0 and price < hi_52 * (1.0 - near_high_pct):
        reasons.append("below_52w_high")
    if lo_52 > 0 and price < lo_52 * (1.0 + off_low_pct):
        reasons.append("near_52w_low")

    detail = {
        "price": price,
        "sma50": float(s50),
        "sma150": float(s150),
        "sma200": float(s200),
        "high_52w": hi_52,
        "low_52w": lo_52,
        "pct_from_52w_high": (hi_52 - price) / hi_52 if hi_52 > 0 else None,
        "pct_above_52w_low": (price - lo_52) / lo_52 if lo_52 > 0 else None,
        "ma200_rising": bool(s200 > s200_past),
    }
    return (len(reasons) == 0), reasons, detail


# ── VCP-like contraction (APPROXIMATION) ─────────────────────────────────────
def detect_vcp_like(df: pd.DataFrame, closed_bar: bool = True):
    """Coarse "VCP-like approximation".

    Splits the recent base into ``MINERVINI_VCP_MIN_CONTRACTIONS + 1`` consecutive
    segments of ``MINERVINI_VCP_PIVOT_LOOKBACK`` bars and checks that each
    segment's relative high-low range is tighter than the previous one (successive
    volatility contractions). Qualifies when there are at least
    MINERVINI_VCP_MIN_CONTRACTIONS such steps AND the overall base depth does not
    exceed MINERVINI_VCP_MAX_BASE_DEPTH_PCT.

    Returns (is_vcp_like: bool, pivot_low: float|None, detail: dict). ``pivot_low``
    is the lowest low of the final (tightest) contraction — the chart level a
    later milestone anchors a dedicated 1R stop to. This is an approximation, NOT
    a true manually-identified VCP pivot.
    """
    detail: dict = {}
    cf = _closed(df, closed_bar)
    if not _has_cols(cf):
        return False, None, {"reason": "insufficient_history"}

    seg_len = int(_cfg("MINERVINI_VCP_PIVOT_LOOKBACK", 15))
    min_contractions = int(_cfg("MINERVINI_VCP_MIN_CONTRACTIONS", 2))
    max_depth = float(_cfg("MINERVINI_VCP_MAX_BASE_DEPTH_PCT", 0.35))
    n_seg = min_contractions + 1
    need = seg_len * n_seg
    if seg_len < 1 or len(cf) < need:
        return False, None, {"reason": "insufficient_history"}

    window = cf.iloc[-need:]
    high = window["High"].astype(float)
    low = window["Low"].astype(float)

    ranges: List[float] = []
    vol_means: List[float] = []
    vol = window["Volume"].astype(float) if "Volume" in window.columns else pd.Series(1.0, index=window.index)
    for i in range(n_seg):
        seg_hi = high.iloc[i * seg_len:(i + 1) * seg_len]
        seg_lo = low.iloc[i * seg_len:(i + 1) * seg_len]
        seg_vo = vol.iloc[i * seg_len:(i + 1) * seg_len]
        top = float(seg_hi.max())
        bot = float(seg_lo.min())
        ranges.append(((top - bot) / top) if top > 0 else 0.0)
        vol_means.append(float(seg_vo.mean()))

    contractions = sum(1 for i in range(1, n_seg) if ranges[i] < ranges[i - 1])
    volume_dry_up = bool(vol_means[-1] < vol_means[-2]) if len(vol_means) >= 2 else True

    base_top = float(high.max())
    base_bot = float(low.min())
    base_depth = ((base_top - base_bot) / base_top) if base_top > 0 else 0.0

    last_seg_low = float(low.iloc[-seg_len:].min())
    is_vcp = bool(contractions >= min_contractions and base_depth <= max_depth)

    detail = {
        "segment_ranges": ranges,
        "segment_vol_means": vol_means,
        "volume_dry_up": volume_dry_up,
        "contractions": contractions,
        "base_depth": base_depth,
        "pivot_low": last_seg_low,
    }
    return is_vcp, last_seg_low, detail


# ── Pocket pivot ─────────────────────────────────────────────────────────────
def detect_pocket_pivot(df: pd.DataFrame, closed_bar: bool = True):
    """Pocket-pivot up day: the last closed bar closes UP versus the prior bar AND
    its volume exceeds the largest DOWN-day volume of the prior
    ``MINERVINI_POCKET_VOL_LOOKBACK`` bars (recent supply absorbed on strength).

    Returns (is_pocket_pivot: bool, detail: dict). If there were no down days in
    the lookback the supply bar is treated as 0, so any up-volume qualifies.
    """
    lookback = int(_cfg("MINERVINI_POCKET_VOL_LOOKBACK", 10))
    cf = _closed(df, closed_bar)
    if not _has_cols(cf) or len(cf) < lookback + 2:
        return False, {"reason": "insufficient_history"}

    close = cf["Close"].astype(float)
    vol = cf["Volume"].astype(float)
    is_up = bool(close.iloc[-1] > close.iloc[-2])

    down_mask = close < close.shift(1)
    prior_vol = vol.iloc[-(lookback + 1):-1]
    prior_down = prior_vol[down_mask.iloc[-(lookback + 1):-1]]
    max_down_vol = float(prior_down.max()) if len(prior_down) else 0.0
    last_vol = float(vol.iloc[-1])

    is_pp = bool(is_up and last_vol > max_down_vol)
    detail = {"is_up": is_up, "last_vol": last_vol, "max_down_vol": max_down_vol}
    return is_pp, detail


# ── Relative strength (SPY-relative proxy & Cross-sectional Universe helper) ─
def relative_strength_rank(df: Optional[pd.DataFrame],
                           benchmark_df: Optional[pd.DataFrame]) -> Optional[float]:
    """Approximate SPY-relative relative-strength percentile in [0, 100].

    Builds the RS line (symbol_close / benchmark_close) over the overlapping dates
    and returns where the current RS sits within its own trailing (<=252-bar)
    min..max range. This is a self-referential proxy, NOT a cross-sectional
    universe rank. NONE-TOLERANT by design: returns None (never raises, never
    blocks) when no benchmark is supplied or there is not enough overlapping data.
    Callers must treat None as "skip the RS check", per the approved decision that
    missing benchmark data must never block a trade.
    """
    if df is None or benchmark_df is None:
        return None
    if "Close" not in getattr(df, "columns", []) or "Close" not in getattr(benchmark_df, "columns", []):
        return None
    try:
        a = df["Close"].astype(float)
        b = benchmark_df["Close"].astype(float)
        common = a.index.intersection(b.index)
        if len(common) < 60:
            return None
        rs = (a.loc[common] / b.loc[common]).dropna()
        if len(rs) < 60:
            return None
        win = min(252, len(rs))
        rs_w = rs.iloc[-win:]
        lo = float(rs_w.min())
        hi = float(rs_w.max())
        if hi <= lo:
            return None
        rank = 100.0 * (float(rs.iloc[-1]) - lo) / (hi - lo)
        return max(0.0, min(100.0, rank))
    except Exception:
        return None


def cross_sectional_rs_rank(performance_dict: Dict[str, float]) -> Dict[str, float]:
    """Calculate true cross-sectional relative strength percentile ranks [0, 100]
    across a universe of symbols for a given lookback performance dictionary.
    """
    if not performance_dict:
        return {}
    s = pd.Series(performance_dict)
    if len(s) == 1:
        return {k: 50.0 for k in s.index}
    ranks = s.rank(pct=True) * 100.0
    return {k: float(v) for k, v in ranks.items()}


# ── Dedicated Minervini stop (1R anchor) ─────────────────────────────────────
def minervini_stop_price(pivot_low: Optional[float], atr: Optional[float] = None) -> Optional[float]:
    """The dedicated setup stop derived from the VCP-like pivot low: placed
    ``MINERVINI_STOP_BUFFER_PCT`` below the pivot. Returns None when no pivot.

    This is the 1R anchor for sizing/preview ONLY (it is NOT the existing
    STOP_LOSS_PCT / HARD_STOP_LOSS_PCT protective stops, and in M0 it is never sent
    to the broker). ``atr`` is reserved for a future ATR-based buffer and is
    currently unused.
    """
    if pivot_low is None:
        return None
    try:
        buf = abs(float(_cfg("MINERVINI_STOP_BUFFER_PCT", 0.005)))
        return round(float(pivot_low) * (1.0 - buf), 2)
    except (TypeError, ValueError):
        return None


# ── Top-level verdict ────────────────────────────────────────────────────────
def evaluate_entry(df: pd.DataFrame, *, benchmark_df: Optional[pd.DataFrame] = None,
                   closed_bar: bool = True) -> MinerviniVerdict:
    """Evaluate one symbol's OHLCV into a MinerviniVerdict.

    stage2_ok = (Stage-2 trend template passes) AND (relative strength passes OR
    is unavailable). VCP-like and pocket-pivot are surfaced as setup-quality
    context but do NOT gate stage2_ok — Stage-2 is the entry filter; VCP/pocket
    refine timing/sizing. Pure and defensive: any internal failure degrades to a
    conservative not-ok verdict rather than raising.
    """
    tt_ok, tt_reasons, tt_detail = trend_template(df, closed_bar)
    vcp_ok, pivot_low, vcp_detail = detect_vcp_like(df, closed_bar)
    pp_ok, pp_detail = detect_pocket_pivot(df, closed_bar)

    bcf = _closed(benchmark_df, closed_bar) if benchmark_df is not None else None
    rs_rank = relative_strength_rank(_closed(df, closed_bar), bcf)

    reasons = list(tt_reasons)
    rs_min = float(_cfg("MINERVINI_RS_RANK_MIN", 70.0))
    if rs_rank is not None and rs_rank < rs_min:
        reasons.append("rs_below_min")

    stage2_ok = bool(tt_ok and ("rs_below_min" not in reasons))

    detail = {
        "trend_template": tt_detail,
        "vcp": vcp_detail,
        "pocket": pp_detail,
        "rs_rank": rs_rank,
    }
    return MinerviniVerdict(
        stage2_ok=stage2_ok,
        reasons=reasons,
        pivot_low=pivot_low,
        rs_rank=rs_rank,
        vcp_like=vcp_ok,
        pocket_pivot=pp_ok,
        detail=detail,
    )
