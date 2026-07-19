"""
momentum_strategy.py -- Momentum Scalp strategy.
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


class MomentumScalpStrategy(BaseStrategy):
    name = StrategyName.MOMENTUM_SCALP

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        if not config.STRATEGY_MOMENTUM_SCALP_ENABLED:
            return None

        if len(df) < 30:
            return None

        df = add_all_indicators(df)

        last = df.iloc[-1]
        prev = df.iloc[-2]

        price = float(last["close"])
        ema9 = float(last.get("ema_9", np.nan))
        ema20 = float(last.get("ema_20", np.nan))
        rsi = float(last.get("rsi", 50))
        atr = float(last.get("atr", 0))
        macd_hist = float(last.get("macd_hist", 0))
        prev_macd_hist = float(prev.get("macd_hist", 0))
        volume = float(last.get("volume", 0))
        vol_ma = float(last.get("volume_ma", 1))

        if any(np.isnan(v) for v in [ema9, ema20, atr]):
            return None
        if atr <= 0:
            return None

        ema_aligned = ema9 > ema20
        price_above_emas = price > ema9 and price > ema20
        rsi_momentum = 45 <= rsi <= 75
        macd_rising = macd_hist > 0 and macd_hist > prev_macd_hist
        vol_above_avg = volume > vol_ma if vol_ma > 0 else False

        buy_checks = sum([
            ema_aligned,
            price_above_emas,
            rsi_momentum,
            macd_rising,
            vol_above_avg,
        ])

        ema_aligned_down = ema9 < ema20
        price_below_emas = price < ema9 and price < ema20
        rsi_momentum_down = 25 <= rsi <= 55
        macd_falling = macd_hist < 0 and macd_hist < prev_macd_hist

        sell_checks = sum([
            ema_aligned_down,
            price_below_emas,
            rsi_momentum_down,
            macd_falling,
            vol_above_avg,
        ])

        rel_vol = volume / vol_ma if vol_ma > 0 else 0

        if buy_checks >= 4:
            swing_low = float(df["low"].tail(10).min())
            stop = max(swing_low, ema20) - 0.01
            risk = price - stop
            if risk > 0:
                atr_target = price + atr * config.DEFAULT_TARGET_RR_RATIO
                rr_target = price + risk * config.DEFAULT_TARGET_RR_RATIO
                target = max(atr_target, rr_target)

                confidence = 0.4 + buy_checks * 0.10
                if rel_vol > 2.0:
                    confidence += 0.05
                confidence = min(confidence, 1.0)

                reasons = []
                if ema_aligned: reasons.append("EMA9>20")
                if price_above_emas: reasons.append("price>EMAs")
                if rsi_momentum: reasons.append(f"RSI {rsi:.0f}")
                if macd_rising: reasons.append("MACD rising")
                if vol_above_avg: reasons.append(f"vol {rel_vol:.1f}x")

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
                    reason=f"Momentum scalp LONG: {' | '.join(reasons)}",
                    metadata={"checks_passed": buy_checks, "ema9": ema9, "ema20": ema20},
                )

        allow_short = getattr(config, "ALLOW_SHORT", False)
        if allow_short and sell_checks >= 4:
            swing_high = float(df["high"].tail(10).max())
            stop = min(swing_high, ema20) + 0.01
            risk = stop - price
            if risk > 0:
                atr_target = price - atr * config.DEFAULT_TARGET_RR_RATIO
                rr_target = price - risk * config.DEFAULT_TARGET_RR_RATIO
                target = min(atr_target, rr_target)

                confidence = 0.4 + sell_checks * 0.10
                if rel_vol > 2.0:
                    confidence += 0.05
                confidence = min(confidence, 1.0)

                reasons = []
                if ema_aligned_down: reasons.append("EMA9<20")
                if price_below_emas: reasons.append("price<EMAs")
                if rsi_momentum_down: reasons.append(f"RSI {rsi:.0f}")
                if macd_falling: reasons.append("MACD falling")
                if vol_above_avg: reasons.append(f"vol {rel_vol:.1f}x")

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
                    reason=f"Momentum scalp SHORT: {' | '.join(reasons)}",
                    metadata={"checks_passed": sell_checks, "ema9": ema9, "ema20": ema20},
                )

        return None
