"""
vwap_strategy.py -- VWAP Bounce / Reclaim strategy.
"""
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

import config
from strategies.constants import StrategyName
from strategies.base import BaseStrategy, TradeSignal
from strategies.indicators import add_all_indicators

from strategies.execution_geometry import (
    calculate_directional_stop_and_validate,
    floor_to_tick_float,
    ceil_to_tick_float,
)

logger = logging.getLogger(__name__)

_floor_to_tick = floor_to_tick_float
_ceil_to_tick = ceil_to_tick_float


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

        if hasattr(df, "index") and len(df.index) > 0:
            last_idx = df.index[-1]
            bar_ts = last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx)
        else:
            bar_ts = str(last.get("timestamp", ""))

        # BUY Logic
        if pullback_touched and held_above and rsi < config.VWAP_RSI_OVERBOUGHT:
            tick_size = 0.01 if price >= 1.0 else 0.0001
            precision = 2 if price >= 1.0 else 4
            pullback_low = float(tail["low"].iloc[-reclaim_bars - 2 : -1].min())
            # Strict geometry contract: structural stop must be strictly below (price - tick_size)
            valid_supports = [lvl for lvl in (pullback_low, vwap_lower) if np.isfinite(lvl) and lvl < (price - tick_size)]
            if valid_supports:
                structural_stop = max(valid_supports) - tick_size
                raw_risk = price - structural_stop
            else:
                structural_stop = float("nan")
                raw_risk = -1.0

            # Defect 4: Validate raw structural risk (reject if negative, zero, or non-finite)
            if not np.isfinite(raw_risk) or raw_risk <= 0 or not np.isfinite(structural_stop) or structural_stop >= (price - tick_size):
                logger.debug(
                    "[SIGNAL GEOMETRY] symbol=%s side=BUY rejected | invalid raw structural risk: price=%.4f structural_stop=%.4f raw_risk=%.4f",
                    symbol, price, structural_stop, raw_risk
                )
            else:
                # Shared Execution Geometry Validation
                spread_mult = float(getattr(config, "GEOMETRY_SPREAD_MULTIPLIER", 1.5))
                spread_floor = (float(spread_dollars) * spread_mult) if (spread_dollars is not None and float(spread_dollars) > 0) else 0.0

                try:
                    stop_geom = calculate_directional_stop_and_validate(
                        side="long",
                        price=price,
                        raw_risk=raw_risk,
                        atr=atr,
                        spread_floor=spread_floor,
                        tick_size=tick_size,
                        precision=precision,
                    )
                except Exception as geom_err:
                    logger.debug("[SIGNAL GEOMETRY] symbol=%s side=BUY rejected by geometry validator: %s", symbol, geom_err)
                    stop_geom = None

                if stop_geom is None:
                    logger.debug(
                        "[SIGNAL GEOMETRY] symbol=%s side=BUY result=REJECTED reason=NO_STOP_GEOMETRY",
                        symbol,
                    )
                    return None

                stop = stop_geom.stop_price
                actual_risk = stop_geom.final_distance

                if not (np.isfinite(stop) and np.isfinite(actual_risk)):
                    logger.debug(
                        "[SIGNAL GEOMETRY] symbol=%s side=BUY result=REJECTED reason=NON_FINITE_STOP_GEOMETRY stop=%s risk=%s",
                        symbol, stop, actual_risk,
                    )
                    return None

                logger.debug(
                    "[STOP GEOMETRY] symbol=%s price=%.4f atr=%.4f raw_stop_distance=%.4f spread_floor=%.4f "
                    "tick_size=%.4f min_stop_distance=%.4f final_stop_distance=%.4f stop=%.4f",
                    symbol, price, atr, raw_risk, spread_floor, tick_size,
                    stop_geom.minimum_distance, stop_geom.final_distance, stop
                )

                max_rr = float(getattr(config, "MAX_TARGET_RR_RATIO", 5.0))
                min_rr = float(getattr(config, "MIN_RR_RATIO", 1.0))
                default_rr = float(getattr(config, "DEFAULT_TARGET_RR_RATIO", 2.0))

                # Structural target calculation
                structural_target_dist = (vwap_upper - price) if (vwap_upper > price and np.isfinite(vwap_upper)) else (actual_risk * default_rr)
                max_target_dist = actual_risk * (max_rr - 0.5)
                target_dist = min(structural_target_dist, max_target_dist)

                # Defect 1: Structure-First target validation (NO artificial force-extension)
                target = _floor_to_tick(price + target_dist, tick_size)
                actual_reward = round(target - price, precision)
                actual_rr = actual_reward / actual_risk if actual_risk > 0 else 0.0

                # Defect 5: Positivity & Ordering Invariant: 0 < stop < price < target
                ordering_valid = bool(0 < stop < price < target and np.isfinite(target))
                risk_valid = bool(actual_risk > 0 and np.isfinite(actual_risk))
                reward_valid = bool(actual_reward > 0 and np.isfinite(actual_reward))
                rr_valid = bool(risk_valid and actual_rr >= (min_rr - 1e-4))

                if not (ordering_valid and risk_valid and reward_valid and rr_valid):
                    logger.debug(
                        "[SIGNAL GEOMETRY] symbol=%s side=BUY result=REJECTED ordering_valid=%s risk_valid=%s reward_valid=%s rr_valid=%s actual_risk=%.4f actual_reward=%.4f actual_rr=%.4f min_rr=%.2f",
                        symbol, ordering_valid, risk_valid, reward_valid, rr_valid, actual_risk, actual_reward, actual_rr, min_rr,
                    )
                    return None

                logger.info(
                    "[SIGNAL GEOMETRY] symbol=%s side=BUY result=ACCEPTED entry=%.2f stop=%.2f target=%.2f | "
                    "structural_stop_distance=%.4f min_stop_distance=%.4f spread_floor=%.4f tick_floor=%.4f | "
                    "actual_risk=%.4f actual_reward=%.4f actual_rr=%.2f",
                    symbol, price, stop, target,
                    raw_risk, stop_geom.minimum_distance, spread_floor, tick_size,
                    actual_risk, actual_reward, actual_rr,
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
                    risk_per_share=actual_risk,
                    bar_timestamp=bar_ts,
                    reason=f"VWAP bounce reclaim | RSI {rsi:.0f} | held {reclaim_bars} bars",
                    metadata={
                        "vwap": vwap,
                        "pullback_low": pullback_low,
                        "normalized_risk": actual_risk,
                        "actual_rr": actual_rr,
                        "raw_risk": raw_risk,
                    },
                )

        allow_short = getattr(config, "ALLOW_SHORT", False)
        inverse_etfs = getattr(config, "INVERSE_ETF_SYMBOLS", set())
        if symbol.upper().strip() in inverse_etfs:
            allow_short = False

        # SELL Logic
        if allow_short and rally_touched and held_below and rsi > config.VWAP_RSI_OVERSOLD:
            tick_size = 0.01 if price >= 1.0 else 0.0001
            precision = 2 if price >= 1.0 else 4
            rally_high = float(tail["high"].iloc[-reclaim_bars - 2 : -1].max())
            # Strict geometry contract: structural stop must be strictly above (price + tick_size)
            valid_resistances = [lvl for lvl in (rally_high, vwap_upper) if np.isfinite(lvl) and lvl > (price + tick_size)]
            if valid_resistances:
                structural_stop = min(valid_resistances) + tick_size
                raw_risk = structural_stop - price
            else:
                structural_stop = float("nan")
                raw_risk = -1.0

            # Defect 4: Validate raw structural risk (reject if negative, zero, or non-finite)
            if not np.isfinite(raw_risk) or raw_risk <= 0 or not np.isfinite(structural_stop) or structural_stop <= (price + tick_size):
                logger.debug(
                    "[SIGNAL GEOMETRY] symbol=%s side=SELL rejected | invalid raw structural risk: price=%.4f structural_stop=%.4f raw_risk=%.4f",
                    symbol, price, structural_stop, raw_risk
                )
            else:
                # Shared Execution Geometry Validation
                spread_mult = float(getattr(config, "GEOMETRY_SPREAD_MULTIPLIER", 1.5))
                spread_floor = (float(spread_dollars) * spread_mult) if (spread_dollars is not None and float(spread_dollars) > 0) else 0.0

                try:
                    stop_geom = calculate_directional_stop_and_validate(
                        side="short",
                        price=price,
                        raw_risk=raw_risk,
                        atr=atr,
                        spread_floor=spread_floor,
                        tick_size=tick_size,
                        precision=precision,
                    )
                except Exception as geom_err:
                    logger.debug("[SIGNAL GEOMETRY] symbol=%s side=SELL rejected by geometry validator: %s", symbol, geom_err)
                    stop_geom = None

                if stop_geom is None:
                    logger.debug(
                        "[SIGNAL GEOMETRY] symbol=%s side=SELL result=REJECTED reason=NO_STOP_GEOMETRY",
                        symbol,
                    )
                    return None

                stop = stop_geom.stop_price
                actual_risk = stop_geom.final_distance

                if not (np.isfinite(stop) and np.isfinite(actual_risk)):
                    logger.debug(
                        "[SIGNAL GEOMETRY] symbol=%s side=SELL result=REJECTED reason=NON_FINITE_STOP_GEOMETRY stop=%s risk=%s",
                        symbol, stop, actual_risk,
                    )
                    return None

                logger.debug(
                    "[STOP GEOMETRY] symbol=%s price=%.4f atr=%.4f raw_stop_distance=%.4f spread_floor=%.4f "
                    "tick_size=%.4f min_stop_distance=%.4f final_stop_distance=%.4f stop=%.4f",
                    symbol, price, atr, raw_risk, spread_floor, tick_size,
                    stop_geom.minimum_distance, stop_geom.final_distance, stop
                )

                max_rr = float(getattr(config, "MAX_TARGET_RR_RATIO", 5.0))
                min_rr = float(getattr(config, "MIN_RR_RATIO", 1.0))
                default_rr = float(getattr(config, "DEFAULT_TARGET_RR_RATIO", 2.0))

                # Structural target calculation
                structural_target_dist = (price - vwap_lower) if (vwap_lower < price and np.isfinite(vwap_lower)) else (actual_risk * default_rr)
                max_target_dist = actual_risk * (max_rr - 0.5)
                target_dist = min(structural_target_dist, max_target_dist)

                # Defect 1: Structure-First target validation (NO artificial force-extension)
                target = _ceil_to_tick(price - target_dist, tick_size)
                actual_reward = round(price - target, precision)
                actual_rr = actual_reward / actual_risk if actual_risk > 0 else 0.0

                # Defect 5: Positivity & Ordering Invariant: 0 < target < price < stop
                ordering_valid = bool(0 < target < price < stop and np.isfinite(target))
                risk_valid = bool(actual_risk > 0 and np.isfinite(actual_risk))
                reward_valid = bool(actual_reward > 0 and np.isfinite(actual_reward))
                rr_valid = bool(risk_valid and actual_rr >= (min_rr - 1e-4))

                if not (ordering_valid and risk_valid and reward_valid and rr_valid):
                    logger.debug(
                        "[SIGNAL GEOMETRY] symbol=%s side=SELL result=REJECTED ordering_valid=%s risk_valid=%s reward_valid=%s rr_valid=%s actual_risk=%.4f actual_reward=%.4f actual_rr=%.4f min_rr=%.2f",
                        symbol, ordering_valid, risk_valid, reward_valid, rr_valid, actual_risk, actual_reward, actual_rr, min_rr,
                    )
                    return None

                logger.info(
                    "[SIGNAL GEOMETRY] symbol=%s side=SELL result=ACCEPTED entry=%.2f stop=%.2f target=%.2f | "
                    "structural_stop_distance=%.4f min_stop_distance=%.4f spread_floor=%.4f tick_floor=%.4f | "
                    "actual_risk=%.4f actual_reward=%.4f actual_rr=%.2f",
                    symbol, price, stop, target,
                    raw_risk, stop_geom.minimum_distance, spread_floor, tick_size,
                    actual_risk, actual_reward, actual_rr,
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
                    risk_per_share=actual_risk,
                    bar_timestamp=bar_ts,
                    reason=f"VWAP rejection | RSI {rsi:.0f} | held below {reclaim_bars} bars",
                    metadata={
                        "vwap": vwap,
                        "rally_high": rally_high,
                        "normalized_risk": actual_risk,
                        "actual_rr": actual_rr,
                        "raw_risk": raw_risk,
                    },
                )

        return None
