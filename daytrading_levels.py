"""Pure daily OHLCV-derived helper levels for paper-trading practice.

These helpers are read-only context. They do not fetch data, write files, call a
broker, or implement intraday VWAP/ORB signals.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional

import pandas as pd


def _positive(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def gap_pct(today_open, previous_close) -> float:
    """Return today_open / previous_close - 1."""
    today_open = _positive("today_open", today_open)
    previous_close = _positive("previous_close", previous_close)
    return float(today_open / previous_close - 1)


def daily_relative_volume(today_volume, avg_volume) -> float:
    """Return today_volume / avg_volume."""
    try:
        today_volume = float(today_volume)
    except (TypeError, ValueError):
        raise ValueError("today_volume must be numeric") from None
    if not math.isfinite(today_volume) or today_volume < 0:
        raise ValueError(f"today_volume must be non-negative, got {today_volume!r}")
    avg_volume = _positive("avg_volume", avg_volume)
    return float(today_volume / avg_volume)


def pivot_points(prev_high, prev_low, prev_close) -> dict[str, float]:
    """Return classic floor-trader pivot levels."""
    high = _positive("prev_high", prev_high)
    low = _positive("prev_low", prev_low)
    close = _positive("prev_close", prev_close)
    if high < low:
        raise ValueError(f"prev_high must be >= prev_low, got {high} < {low}")

    pp = (high + low + close) / 3.0
    return {
        "pp": float(pp),
        "r1": float(2 * pp - low),
        "r2": float(pp + (high - low)),
        "r3": float(high + 2 * (pp - low)),
        "s1": float(2 * pp - high),
        "s2": float(pp - (high - low)),
        "s3": float(low - 2 * (high - pp)),
    }


def nearest_level(price, levels, direction="UP") -> Optional[float]:
    """Return the nearest level above or below price."""
    price = _positive("price", price)
    direction = str(direction).upper().strip()
    if direction not in {"UP", "DOWN"}:
        raise ValueError(f"direction must be UP or DOWN, got {direction!r}")

    raw_values: Iterable[float]
    if isinstance(levels, Mapping):
        raw_values = levels.values()
    else:
        raw_values = levels

    vals = []
    for value in raw_values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            vals.append(value)

    if direction == "UP":
        candidates = [v for v in vals if v > price]
        return float(min(candidates)) if candidates else None
    candidates = [v for v in vals if v < price]
    return float(max(candidates)) if candidates else None


def level_zone(level, price, buffer_pct=0.0025, min_buffer=0.05) -> tuple[float, float]:
    """Return (level - buffer, level + buffer)."""
    level = _positive("level", level)
    price = _positive("price", price)
    buffer_pct = _positive("buffer_pct", buffer_pct)
    min_buffer = _positive("min_buffer", min_buffer)
    buffer = max(min_buffer, price * buffer_pct)
    return (float(level - buffer), float(level + buffer))


def atr_dollars_from_ohlc(df: pd.DataFrame, period=14) -> Optional[float]:
    """Return the latest ATR in dollars from High/Low/Close columns.

    Returns None when the input is missing columns, has too few rows, or cannot
    produce a finite latest ATR.
    """
    try:
        period = int(period)
    except (TypeError, ValueError):
        raise ValueError("period must be an integer") from None
    if period <= 0:
        raise ValueError("period must be > 0")
    if df is None or len(df) < period + 1:
        return None

    cols = {str(c).lower(): c for c in getattr(df, "columns", [])}
    high_col = cols.get("high")
    low_col = cols.get("low")
    close_col = cols.get("close")
    if high_col is None or low_col is None or close_col is None:
        return None

    high = pd.to_numeric(df[high_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    latest = atr.iloc[-1]
    if pd.isna(latest) or not math.isfinite(float(latest)):
        return None
    return float(latest)
