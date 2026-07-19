"""
candle_confirmation.py -- Paper-only candlestick confirmation for intraday signals.

Provides a lightweight candle_score (0-5) based on the most recent
complete candle and the one before it, plus a decision action.
"""
import numpy as np
import pandas as pd


def _calc_vwap(df: pd.DataFrame) -> pd.Series:
    """Simple cumulative VWAP reset per trading day (inline, no dependency)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)

    if hasattr(df.index, "date"):
        dates = df.index.date
    else:
        dates = pd.Series([0] * len(df), index=df.index)

    tp_v = typical * vol
    cum_tp_v = tp_v.groupby(dates).cumsum()
    cum_vol = vol.groupby(dates).cumsum()
    
    vwap = cum_tp_v / cum_vol
    return vwap.fillna(typical)


def confirm_candle(df: pd.DataFrame, confidence: float) -> dict:
    """
    Evaluate the most recent candle(s) and return a confirmation verdict.
    """
    if df.empty or len(df) < 2:
        return {
            "candle_score": 0,
            "candle_reasons": ["insufficient data"],
            "candle_action": "SKIP",
        }

    # Ensure required columns exist
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            return {
                "candle_score": 0,
                "candle_reasons": [f"missing column: {col}"],
                "candle_action": "SKIP",
            }

    # Compute VWAP if not present
    if "vwap" not in df.columns:
        df = df.copy()
        df["vwap"] = _calc_vwap(df)

    # Use the most recent complete candle (index -1) and the one before (index -2)
    cur = df.iloc[-1]
    prev = df.iloc[-2]
    reasons = []
    score = 0

    # +1 current candle is green (close > open)
    if cur["close"] > cur["open"]:
        score += 1
        reasons.append("green candle")
    else:
        reasons.append("red/down candle")

    # +1 close above VWAP
    vwap_val = cur.get("vwap", np.nan)
    if pd.notna(vwap_val) and cur["close"] > vwap_val:
        score += 1
        reasons.append("close > VWAP")
    else:
        reasons.append("close <= VWAP")

    # +1 short upper wick (wick <= 2 * body)
    body = abs(cur["close"] - cur["open"])
    upper_wick = cur["high"] - max(cur["close"], cur["open"])
    candle_range = cur["high"] - cur["low"]
    if candle_range > 0 and upper_wick <= 2 * body:
        score += 1
        reasons.append("short upper wick")
    else:
        reasons.append("long upper wick")

    # +1 volume >= previous candle volume
    if cur["volume"] >= prev["volume"]:
        score += 1
        reasons.append("volume >= prev")
    else:
        reasons.append("volume < prev")

    # +1 close reclaims previous candle high
    if cur["close"] > prev["high"]:
        score += 1
        reasons.append("close > prev high")
    else:
        reasons.append("close <= prev high")

    # Decision rules
    if 0.60 <= confidence < 0.65:
        if score >= 3:
            action = "ALLOW"
        elif score == 2:
            action = "REDUCE"
        else:
            action = "SKIP"
    elif confidence >= 0.65:
        if score >= 2:
            action = "ALLOW"
        else:
            action = "SKIP"
    else:
        # Below minimum confidence — candle confirmation doesn't override hard gate
        action = "SKIP"

    return {
        "candle_score": score,
        "candle_reasons": reasons,
        "candle_action": action,
    }
