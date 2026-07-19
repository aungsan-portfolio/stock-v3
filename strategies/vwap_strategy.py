"""
vwap_strategy.py -- VWAP Bounce / Reclaim strategy.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

import config
from strategies.constants import StrategyName
from strategies.base import BaseStrategy, TradeSignal
from strategies.indicators import add_all_indicators

logger = logging.getLogger(__name__)


class VWAPBounceStrategy(BaseStrategy):
    name = StrategyName.VWAP_BOUNCE

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        if not config.STRATEGY_VWAP_BOUNCE_ENABLED:
            return None

        if len(df) < 30:
            return None

        df = add_all_indicators(df)

        if "vwap" not in df.columns or df["vwap"].isna().all():
            return None

        reclaim_bars = config.VWAP_RECLAIM_BARS
        tail = df.tail(reclaim_bars + 5)
        if len(tail) < reclaim_bars + 1:
            return None

        last = df.iloc[-1]
        price = float(last["close"])
        vwap = float(last["vwap"])
        rsi = float(last.get("rsi", 50))
        atr = float(last.get("atr", 0))
        vwap_lower = float(last.get("vwap_lower", vwap))
        vwap_upper = float(last.get("vwap_upper", vwap))

        if atr <= 0 or np.isnan(atr):
            return None

        tolerance = vwap * config.VWAP_BOUNCE_TOLERANCE_PCT / 100.0

        recent_closes = tail["close"].values[-reclaim_bars:]
        recent_vwaps = tail["vwap"].values[-reclaim_bars:]

        pullback_touched = False
        rally_touched = False
        for i in range(len(tail) - reclaim_bars - 1, len(tail) - reclaim_bars + 2):
            if i < 0 or i >= len(tail):
                continue
            bar_low = float(tail["low"].iloc[i])
            bar_high = float(tail["high"].iloc[i])
            bar_vwap = float(tail["vwap"].iloc[i])
            if abs(bar_low - bar_vwap) <= tolerance:
                pullback_touched = True
            if abs(bar_high - bar_vwap) <= tolerance:
                rally_touched = True

        if not pullback_touched and not rally_touched:
            return None

        held_above = all(c >= v - tolerance for c, v in zip(recent_closes, recent_vwaps))
        held_below = all(c <= v + tolerance for c, v in zip(recent_closes, recent_vwaps))

        volume = float(last.get("volume", 0))
        vol_ma = float(last.get("volume_ma", 1))

        # BUY Logic
        if pullback_touched and held_above and rsi < config.VWAP_RSI_OVERBOUGHT:
            pullback_low = float(tail["low"].iloc[-reclaim_bars - 2 : -1].min())
            stop = max(pullback_low, vwap_lower) - 0.01
            risk = price - stop
            if risk > 0:
                rr_target = price + risk * config.DEFAULT_TARGET_RR_RATIO
                target = max(vwap_upper, rr_target)
                if target > price:
                    confidence = 0.5
                    if rsi < config.VWAP_RSI_OVERSOLD:
                        confidence += 0.15
                    if price > vwap:
                        confidence += 0.10
                    if vol_ma > 0 and volume > vol_ma:
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
                        reason=f"VWAP bounce reclaim | RSI {rsi:.0f} | held {reclaim_bars} bars",
                        metadata={"vwap": vwap, "pullback_low": pullback_low},
                    )

        allow_short = getattr(config, "ALLOW_SHORT", False)
        # SELL Logic
        if allow_short and rally_touched and held_below and rsi > config.VWAP_RSI_OVERSOLD:
            rally_high = float(tail["high"].iloc[-reclaim_bars - 2 : -1].max())
            stop = min(rally_high, vwap_upper) + 0.01
            risk = stop - price
            if risk > 0:
                rr_target = price - risk * config.DEFAULT_TARGET_RR_RATIO
                target = min(vwap_lower, rr_target)
                if target < price:
                    confidence = 0.5
                    if rsi > config.VWAP_RSI_OVERBOUGHT:
                        confidence += 0.15
                    if price < vwap:
                        confidence += 0.10
                    if vol_ma > 0 and volume > vol_ma:
                        confidence += 0.10
                    confidence = min(confidence, 1.0)

                    return TradeSignal(
                        symbol=symbol,
                        strategy=self.name,
                        side="SELL",
                        confidence=confidence,
                        entry_price=price,
                        stop_price=stop,
                        target_price=target,
                        atr=atr,
                        risk_per_share=risk,
                        reason=f"VWAP rejection | RSI {rsi:.0f} | held below {reclaim_bars} bars",
                        metadata={"vwap": vwap, "rally_high": rally_high},
                    )

        return None
