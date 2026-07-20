"""
orb_strategy.py -- Opening Range Breakout strategy.
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


class ORBStrategy(BaseStrategy):
    name = StrategyName.ORB

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        if not config.STRATEGY_ORB_ENABLED:
            return None

        if len(df) < config.ORB_WINDOW_MINUTES + 5:
            return None

        df = add_all_indicators(df)

        orb_high = df["orb_high"].iloc[-1]
        orb_low = df["orb_low"].iloc[-1]
        if np.isnan(orb_high) or np.isnan(orb_low):
            return None

        orb_range = orb_high - orb_low
        if orb_range <= 0:
            return None

        atr = df["atr"].iloc[-1]
        if np.isnan(atr) or atr <= 0:
            return None

        if orb_range < config.ORB_MIN_RANGE_ATR_RATIO * atr:
            return None

        last = df.iloc[-1]
        price = float(last["close"])
        volume = float(last["volume"])
        vol_ma = float(last.get("volume_ma", 0))
        rsi = float(last.get("rsi", 50))

        vol_ratio = volume / vol_ma if vol_ma > 0 else 0
        vol_ok = vol_ratio >= config.ORB_MIN_VOLUME_RATIO

        if price > orb_high and vol_ok and rsi < 75:
            stop = orb_low
            risk = price - stop
            target = price + risk * config.DEFAULT_TARGET_RR_RATIO
            confidence = min(0.5 + vol_ratio * 0.1 + (orb_range / atr) * 0.1, 1.0)

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
                reason=f"ORB breakout above {orb_high:.2f} | vol {vol_ratio:.1f}x | RSI {rsi:.0f}",
                metadata={"orb_high": orb_high, "orb_low": orb_low, "orb_range": orb_range},
            )

        allow_short = getattr(config, "ALLOW_SHORT", False)
        if price < orb_low and vol_ok and rsi > 25 and allow_short:
            stop = orb_high
            risk = stop - price
            target = price - risk * config.DEFAULT_TARGET_RR_RATIO
            confidence = min(0.5 + vol_ratio * 0.1 + (orb_range / atr) * 0.1, 1.0)

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
                reason=f"ORB breakdown below {orb_low:.2f} | vol {vol_ratio:.1f}x | RSI {rsi:.0f}",
                metadata={"orb_high": orb_high, "orb_low": orb_low, "orb_range": orb_range},
            )

        return None
