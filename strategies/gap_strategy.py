"""
gap_strategy.py -- Gap-and-Go strategy.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from strategies.constants import StrategyName
from strategies.base import BaseStrategy, TradeSignal
from strategies.indicators import add_all_indicators
from strategies.intraday_data import previous_close

logger = logging.getLogger(__name__)


def _latest_session(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows for the latest trading date in a multi-day intraday frame."""
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    latest_date = df.index.date[-1]
    return df.loc[df.index.date == latest_date]


class GapAndGoStrategy(BaseStrategy):
    name = StrategyName.GAP_AND_GO

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        if not config.STRATEGY_GAP_AND_GO_ENABLED:
            return None

        df = _latest_session(df)
        if len(df) < 10:
            return None

        prev = previous_close(symbol)
        if prev is None or prev <= 0:
            return None

        if "atr" not in df.columns:
            df = add_all_indicators(df)

        open_price = float(df["open"].iloc[0])
        gap_pct = ((open_price - prev) / prev) * 100.0

        if abs(gap_pct) < config.SCAN_MIN_GAP_PCT:
            return None
        if abs(gap_pct) > config.SCAN_MAX_GAP_PCT:
            return None

        last = df.iloc[-1]
        price = float(last["close"])
        vwap = float(last.get("vwap", price))
        atr = float(last.get("atr", 0))
        rsi = float(last.get("rsi", 50))
        volume = float(last.get("volume", 0))
        vol_ma = float(last.get("volume_ma", 1))

        if atr <= 0 or np.isnan(atr):
            return None

        rel_vol = volume / vol_ma if vol_ma > 0 else 0

        if gap_pct > 0 and price > vwap and price >= open_price:
            hold_bars = config.GAP_CONTINUATION_MIN_HOLD_BARS
            recent = df.tail(hold_bars)
            if len(recent) < hold_bars:
                return None

            held_above_vwap = all(
                float(recent["close"].iloc[i]) >= float(recent["vwap"].iloc[i])
                for i in range(len(recent))
                if not np.isnan(recent["vwap"].iloc[i])
            )

            if not held_above_vwap:
                return None

            first_bar_low = float(df["low"].iloc[0])
            stop = max(first_bar_low, vwap) - 0.02
            risk = price - stop
            if risk <= 0:
                return None

            target = price + risk * config.DEFAULT_TARGET_RR_RATIO

            confidence = 0.5
            if gap_pct > 5:
                confidence += 0.10
            if rel_vol > 2.0:
                confidence += 0.10
            if price > open_price:
                confidence += 0.10
            confidence = min(confidence, 1.0)

            return TradeSignal(
                symbol=symbol,
                strategy=self.name,
                side="BUY",
                confidence=confidence,
                entry_price=price,
                stop_price=stop,
                target_price=target,
                atr=atr,
                risk_per_share=risk,
                reason=(
                    f"Gap-and-Go {gap_pct:+.1f}% | above VWAP "
                    f"| rvol {rel_vol:.1f}x | RSI {rsi:.0f}"
                ),
                metadata={
                    "gap_pct": gap_pct,
                    "prev_close": prev,
                    "open_price": open_price,
                },
            )

        return None
