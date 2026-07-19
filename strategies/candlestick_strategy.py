"""
candlestick_strategy.py -- Candlestick pattern detection + signal generation.
"""
from __future__ import annotations
import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from strategies.constants import StrategyName
import config
from strategies.base import BaseStrategy, TradeSignal

logger = logging.getLogger(__name__)


BULLISH_PATTERNS: List[str] = [
    "HAMMER",
    "BULLISH_ENGULFING",
    "MORNING_STAR",
    "PIERCING_LINE",
    "THREE_WHITE_SOLDIERS",
    "DRAGONFLY_DOJI",
]

BEARISH_PATTERNS: List[str] = [
    "SHOOTING_STAR",
    "BEARISH_ENGULFING",
    "EVENING_STAR",
    "DARK_CLOUD_COVER",
    "THREE_BLACK_CROWS",
    "GRAVESTONE_DOJI",
]


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either lowercase engine columns or uppercase backtest columns."""
    if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
        return df
    rename = {}
    for col in df.columns:
        lo = str(col).lower()
        if lo == "open":
            rename[col] = "Open"
        elif lo == "high":
            rename[col] = "High"
        elif lo == "low":
            rename[col] = "Low"
        elif lo == "close":
            rename[col] = "Close"
        elif lo == "volume":
            rename[col] = "Volume"
    if rename:
        return df.rename(columns=rename)
    return df


def _atr14(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _rsi14(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _is_hammer(o, h, l, c) -> bool:
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0:
        return False
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    return lower_shadow >= 2.0 * body and upper_shadow <= 0.1 * candle_range and body > 0


def _is_shooting_star(o, h, l, c) -> bool:
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0:
        return False
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    return upper_shadow >= 2.0 * body and lower_shadow <= 0.1 * candle_range and body > 0


def _is_bullish_engulfing(o1, c1, o2, c2) -> bool:
    return c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1


def _is_bearish_engulfing(o1, c1, o2, c2) -> bool:
    return c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1


def _is_morning_star(o1, c1, o2, c2, o3, c3) -> bool:
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    body3 = abs(c3 - o3)
    return (
        c1 < o1
        and body2 < 0.3 * body1
        and c3 > o3
        and body3 >= 0.5 * body1
        and c3 > (o1 + c1) / 2
    )


def _is_evening_star(o1, c1, o2, c2, o3, c3) -> bool:
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    body3 = abs(c3 - o3)
    return (
        c1 > o1
        and body2 < 0.3 * body1
        and c3 < o3
        and body3 >= 0.5 * body1
        and c3 < (o1 + c1) / 2
    )


def _is_piercing_line(o1, c1, o2, c2) -> bool:
    midpoint = (o1 + c1) / 2
    return c1 < o1 and c2 > o2 and o2 < c1 and c2 > midpoint and c2 < o1


def _is_dark_cloud_cover(o1, c1, o2, c2) -> bool:
    midpoint = (o1 + c1) / 2
    return c1 > o1 and c2 < o2 and o2 > c1 and c2 < midpoint and c2 > o1


def _is_three_white_soldiers(opens, closes) -> bool:
    return all([
        closes[i] > opens[i]
        and closes[i] > closes[i - 1]
        and opens[i] > opens[i - 1]
        for i in range(1, 3)
    ]) and closes[0] > opens[0]


def _is_three_black_crows(opens, closes) -> bool:
    return all([
        closes[i] < opens[i]
        and closes[i] < closes[i - 1]
        and opens[i] < opens[i - 1]
        for i in range(1, 3)
    ]) and closes[0] < opens[0]


def _is_dragonfly_doji(o, h, l, c) -> bool:
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0:
        return False
    lower_shadow = min(o, c) - l
    upper_shadow = h - max(o, c)
    return body <= 0.05 * candle_range and lower_shadow >= 0.6 * candle_range and upper_shadow <= 0.05 * candle_range


def _is_gravestone_doji(o, h, l, c) -> bool:
    body = abs(c - o)
    candle_range = h - l
    if candle_range == 0:
        return False
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    return body <= 0.05 * candle_range and upper_shadow >= 0.6 * candle_range and lower_shadow <= 0.05 * candle_range


def _patterns_at_index(opens, highs, lows, closes, i: int) -> list[str]:
    """Return candlestick patterns that fire at index i only."""
    patterns: list[str] = []
    o, h, l, c = opens[i], highs[i], lows[i], closes[i]

    if _is_hammer(o, h, l, c):
        patterns.append("HAMMER")
    if _is_shooting_star(o, h, l, c):
        patterns.append("SHOOTING_STAR")
    if _is_dragonfly_doji(o, h, l, c):
        patterns.append("DRAGONFLY_DOJI")
    if _is_gravestone_doji(o, h, l, c):
        patterns.append("GRAVESTONE_DOJI")

    if i >= 1:
        o1, c1 = opens[i - 1], closes[i - 1]
        if _is_bullish_engulfing(o1, c1, o, c):
            patterns.append("BULLISH_ENGULFING")
        if _is_bearish_engulfing(o1, c1, o, c):
            patterns.append("BEARISH_ENGULFING")
        if _is_piercing_line(o1, c1, o, c):
            patterns.append("PIERCING_LINE")
        if _is_dark_cloud_cover(o1, c1, o, c):
            patterns.append("DARK_CLOUD_COVER")

    if i >= 2:
        o1, c1 = opens[i - 2], closes[i - 2]
        o2, c2 = opens[i - 1], closes[i - 1]
        if _is_morning_star(o1, c1, o2, c2, o, c):
            patterns.append("MORNING_STAR")
        if _is_evening_star(o1, c1, o2, c2, o, c):
            patterns.append("EVENING_STAR")
        if _is_three_white_soldiers([opens[i - 2], opens[i - 1], o], [closes[i - 2], closes[i - 1], c]):
            patterns.append("THREE_WHITE_SOLDIERS")
        if _is_three_black_crows([opens[i - 2], opens[i - 1], o], [closes[i - 2], closes[i - 1], c]):
            patterns.append("THREE_BLACK_CROWS")

    return patterns


def _compute_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Add fired_patterns, bullish_count, and bearish_count columns."""
    df = _normalize_ohlcv_columns(df)
    n = len(df)
    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()
    closes = df["Close"].to_numpy()

    fired = [_patterns_at_index(opens, highs, lows, closes, i) for i in range(n)]

    df = df.copy()
    df["fired_patterns"] = fired
    df["bullish_count"] = df["fired_patterns"].apply(lambda ps: sum(1 for p in ps if p in BULLISH_PATTERNS))
    df["bearish_count"] = df["fired_patterns"].apply(lambda ps: sum(1 for p in ps if p in BEARISH_PATTERNS))
    return df


class CandlestickPatternStrategy(BaseStrategy):
    """Generate BUY / SELL signals from candlestick pattern confluence."""

    name = StrategyName.CANDLESTICK

    def __init__(
        self,
        rr_ratio: float = 2.0,
        atr_stop_mult: float = 1.5,
        min_confidence: float = 0.40,
        min_patterns: int = 1,
    ):
        self.rr_ratio = rr_ratio
        self.atr_stop_mult = atr_stop_mult
        self.min_confidence = min_confidence
        self.min_patterns = min_patterns

    @staticmethod
    def _confidence(pattern_count: int, rsi: float, close: float, sma20: float, side: str) -> float:
        """Score = pattern_base + rsi_bonus + trend_bonus, capped at 1.0."""
        base = min(0.60, pattern_count * 0.20)

        rsi_bonus = 0.0
        if side == "BUY" and rsi < 40:
            rsi_bonus = 0.20
        elif side == "SELL" and rsi > 60:
            rsi_bonus = 0.20

        trend_bonus = 0.0
        if pd.notna(sma20):
            if side == "BUY" and close > sma20:
                trend_bonus = 0.20
            elif side == "SELL" and close < sma20:
                trend_bonus = 0.20

        return min(1.0, base + rsi_bonus + trend_bonus)

    def _build_signal(
        self,
        symbol: str,
        row: pd.Series,
        atr: float,
        rsi: float,
        sma20: float,
        side: str,
        patterns: list,
    ) -> Optional[TradeSignal]:
        """Build and validate a TradeSignal from a detected pattern row."""
        entry = float(row["Close"])
        stop_dist = atr * self.atr_stop_mult

        if side == "BUY":
            stop = entry - stop_dist
            target = entry + stop_dist * self.rr_ratio
        else:
            stop = entry + stop_dist
            target = entry - stop_dist * self.rr_ratio

        confidence = self._confidence(
            pattern_count=len(patterns),
            rsi=rsi,
            close=entry,
            sma20=sma20,
            side=side,
        )
        if confidence < self.min_confidence:
            return None

        pattern_str = ", ".join(patterns)
        sig = TradeSignal(
            symbol=symbol,
            strategy=self.name,
            side=side,
            confidence=confidence,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            atr=atr,
            risk_per_share=stop_dist,
            reason=f"{side} signal - patterns: {pattern_str}",
            pattern_name=pattern_str,
            metadata={"patterns": patterns, "rsi": rsi, "sma20": sma20},
        )
        if not sig.is_valid:
            logger.debug("Invalid signal skipped: %s", sig)
            return None
        return sig

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        """BaseStrategy.evaluate() wrapper; honors config strategy flag."""
        if not getattr(config, "STRATEGY_CANDLESTICK_ENABLED", True):
            return None
        return self.generate_signal(df, symbol=symbol)

    def generate_signal(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> Optional[TradeSignal]:
        """Analyse the last row of df and return a TradeSignal or None."""
        df = _normalize_ohlcv_columns(df)
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if len(df) < 20:
            return None

        if "atr" in df.columns:
            atr_ser = df["atr"]
        else:
            atr_ser = _atr14(df)

        if "rsi" in df.columns:
            rsi_ser = df["rsi"]
        else:
            rsi_ser = _rsi14(df["Close"])

        if "ema_20" in df.columns:
            sma_ser = df["ema_20"]
        else:
            sma_ser = _sma(df["Close"], 20)

        last_idx = df.index[-1]
        row = df.loc[last_idx]
        atr = float(atr_ser.loc[last_idx])
        rsi = float(rsi_ser.loc[last_idx])
        sma20 = float(sma_ser.loc[last_idx]) if pd.notna(sma_ser.loc[last_idx]) else float("nan")

        if atr == 0 or pd.isna(atr):
            return None

        opens = df["Open"].to_numpy()
        highs = df["High"].to_numpy()
        lows = df["Low"].to_numpy()
        closes = df["Close"].to_numpy()
        last_pos = len(df) - 1
        fired_patterns = _patterns_at_index(opens, highs, lows, closes, last_pos)

        bullish_patterns = [p for p in fired_patterns if p in BULLISH_PATTERNS]
        bearish_patterns = [p for p in fired_patterns if p in BEARISH_PATTERNS]

        if len(bullish_patterns) >= self.min_patterns and len(bullish_patterns) >= len(bearish_patterns):
            return self._build_signal(symbol, row, atr, rsi, sma20, "BUY", bullish_patterns)
        
        allow_short = getattr(config, "ALLOW_SHORT", False)
        if allow_short and len(bearish_patterns) >= self.min_patterns:
            return self._build_signal(symbol, row, atr, rsi, sma20, "SELL", bearish_patterns)

        return None

    def scan_all(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> List[TradeSignal]:
        """Walk-forward scan: use only rows up to the current bar."""
        df = _normalize_ohlcv_columns(df)
        signals = []
        for i in range(20, len(df)):
            window = df.iloc[: i + 1]
            sig = self.generate_signal(window, symbol=symbol)
            if sig:
                signals.append(sig)
        return signals
