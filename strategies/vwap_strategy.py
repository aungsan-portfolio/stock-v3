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

    def evaluate(self, symbol: str, df: pd.DataFrame, spread_dollars: Optional[float] = None) -> Optional[TradeSignal]:
        if not config.STRATEGY_VWAP_BOUNCE_ENABLED:
            return None

        if len(df) < 30:
            return None

        if "atr" not in df.columns:
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

        if spread_dollars is None:
            if "spread" in df.columns:
                try:
                    spread_dollars = float(last.get("spread", 0.0) or 0.0)
                except (TypeError, ValueError):
                    spread_dollars = 0.0
            elif "spread" in getattr(df, "attrs", {}):
                spread_dollars = float(df.attrs.get("spread", 0.0) or 0.0)
            else:
                spread_dollars = 0.0

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

        bar_ts = str(df.index[-1]) if hasattr(df, "index") and len(df.index) > 0 else str(last.get("timestamp", ""))

        # BUY Logic
        if pullback_touched and held_above and rsi < config.VWAP_RSI_OVERBOUGHT:
            pullback_low = float(tail["low"].iloc[-reclaim_bars - 2 : -1].min())
            structural_stop = max(pullback_low, vwap_lower) - 0.01
            raw_risk = price - structural_stop

            # P1 Upstream Geometry Normalization: max(0.75 * ATR, 1.5 * spread, tick_size)
            tick_size = 0.01 if price >= 1.0 else 0.0001
            noise_floor_ratio = float(getattr(config, "NOISE_FLOOR_HARD_VETO_RATIO", 0.75))
            atr_floor = atr * noise_floor_ratio
            spread_mult = float(getattr(config, "GEOMETRY_SPREAD_MULTIPLIER", 1.5))
            spread_floor = (float(spread_dollars) * spread_mult) if (spread_dollars is not None and float(spread_dollars) > 0) else 0.0
            min_stop_distance = max(atr_floor, spread_floor, tick_size)

            final_risk = max(raw_risk, min_stop_distance)
            stop = round(price - final_risk, 2 if price >= 1.0 else 4)

            if final_risk > 0 and stop < price:
                max_rr = float(getattr(config, "MAX_TARGET_RR_RATIO", 5.0))
                min_rr = float(getattr(config, "MIN_RR_RATIO", 1.0))
                default_rr = float(getattr(config, "DEFAULT_TARGET_RR_RATIO", 2.0))

                structural_target_dist = vwap_upper - price if vwap_upper > price else (final_risk * default_rr)
                max_target_dist = final_risk * (max_rr - 0.5)
                target_dist = min(structural_target_dist, max_target_dist)
                target_dist = max(target_dist, final_risk * min_rr)
                target = round(price + target_dist, 2 if price >= 1.0 else 4)

                raw_target_dist = vwap_upper - price if vwap_upper > price else (final_risk * default_rr)
                raw_rr = raw_target_dist / raw_risk if raw_risk > 0 else 0.0
                final_rr = (target - price) / final_risk if final_risk > 0 else 0.0

                if target > price and final_rr >= (min_rr - 1e-4):
                    logger.info(
                        "[SIGNAL GEOMETRY] symbol=%s side=BUY entry=%.2f stop=%.2f target=%.2f | "
                        "structural_stop_distance=%.4f atr_floor=%.4f spread_floor=%.4f tick_floor=%.4f final_stop_distance=%.4f | "
                        "projected_rr_raw=%.2f projected_rr_final=%.2f result=ACCEPTED",
                        symbol, price, stop, target,
                        raw_risk, atr_floor, spread_floor, tick_size, final_risk,
                        raw_rr, final_rr
                    )
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
                        risk_per_share=final_risk,
                        bar_timestamp=bar_ts,
                        reason=f"VWAP bounce reclaim | RSI {rsi:.0f} | held {reclaim_bars} bars",
                        metadata={"vwap": vwap, "pullback_low": pullback_low, "normalized_risk": final_risk},
                    )
                else:
                    logger.debug(
                        "[SIGNAL GEOMETRY] symbol=%s side=BUY rejected | raw_risk=%.4f final_risk=%.4f final_rr=%.2f < min_rr=%.2f",
                        symbol, raw_risk, final_risk, final_rr, min_rr
                    )

        allow_short = getattr(config, "ALLOW_SHORT", False)
        inverse_etfs = getattr(config, "INVERSE_ETF_SYMBOLS", set())
        if symbol.upper().strip() in inverse_etfs:
            allow_short = False

        # SELL Logic
        if allow_short and rally_touched and held_below and rsi > config.VWAP_RSI_OVERSOLD:
            rally_high = float(tail["high"].iloc[-reclaim_bars - 2 : -1].max())
            structural_stop = min(rally_high, vwap_upper) + 0.01
            raw_risk = structural_stop - price

            # P1 Upstream Geometry Normalization: max(0.75 * ATR, 1.5 * spread, tick_size)
            tick_size = 0.01 if price >= 1.0 else 0.0001
            noise_floor_ratio = float(getattr(config, "NOISE_FLOOR_HARD_VETO_RATIO", 0.75))
            atr_floor = atr * noise_floor_ratio
            spread_mult = float(getattr(config, "GEOMETRY_SPREAD_MULTIPLIER", 1.5))
            spread_floor = (float(spread_dollars) * spread_mult) if (spread_dollars is not None and float(spread_dollars) > 0) else 0.0
            min_stop_distance = max(atr_floor, spread_floor, tick_size)

            final_risk = max(raw_risk, min_stop_distance)
            stop = round(price + final_risk, 2 if price >= 1.0 else 4)

            if final_risk > 0 and stop > price:
                max_rr = float(getattr(config, "MAX_TARGET_RR_RATIO", 5.0))
                min_rr = float(getattr(config, "MIN_RR_RATIO", 1.0))
                default_rr = float(getattr(config, "DEFAULT_TARGET_RR_RATIO", 2.0))

                structural_target_dist = price - vwap_lower if vwap_lower < price else (final_risk * default_rr)
                max_target_dist = final_risk * (max_rr - 0.5)
                target_dist = min(structural_target_dist, max_target_dist)
                target_dist = max(target_dist, final_risk * min_rr)
                target = round(price - target_dist, 2 if price >= 1.0 else 4)

                raw_target_dist = price - vwap_lower if vwap_lower < price else (final_risk * default_rr)
                raw_rr = raw_target_dist / raw_risk if raw_risk > 0 else 0.0
                final_rr = (price - target) / final_risk if final_risk > 0 else 0.0

                if target < price and final_rr >= (min_rr - 1e-4):
                    logger.info(
                        "[SIGNAL GEOMETRY] symbol=%s side=SELL entry=%.2f stop=%.2f target=%.2f | "
                        "structural_stop_distance=%.4f atr_floor=%.4f spread_floor=%.4f tick_floor=%.4f final_stop_distance=%.4f | "
                        "projected_rr_raw=%.2f projected_rr_final=%.2f result=ACCEPTED",
                        symbol, price, stop, target,
                        raw_risk, atr_floor, spread_floor, tick_size, final_risk,
                        raw_rr, final_rr
                    )
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
                        risk_per_share=final_risk,
                        bar_timestamp=bar_ts,
                        reason=f"VWAP rejection | RSI {rsi:.0f} | held below {reclaim_bars} bars",
                        metadata={"vwap": vwap, "rally_high": rally_high, "normalized_risk": final_risk},
                    )
                else:
                    logger.debug(
                        "[SIGNAL GEOMETRY] symbol=%s side=SELL rejected | raw_risk=%.4f final_risk=%.4f final_rr=%.2f < min_rr=%.2f",
                        symbol, raw_risk, final_risk, final_rr, min_rr
                    )

        return None
