"""strategies/mean_reversion.py

Multi-Layer Mean Reversion Strategy for Sideways (Range-Bound) Markets.

Generates BUY signals when price deviates significantly below VWAP in a SIDEWAYS market regime,
subject to 4 strict confirmation layers to eliminate false breakdown signals:
  1. VWAP Deviation: (VWAP - Close) >= 1.2 * ATR
  2. RSI Oversold: RSI <= 35
  3. Volume Spike Guard: Volume Ratio <= 2.0 (ensures no heavy institutional breakdown)
  4. Bollinger Band Hug: Close <= Lower Bollinger Band
"""

import logging
import math
from typing import Optional
import pandas as pd
import numpy as np

import config
from strategies.base import TradeSignal
from strategies.execution_geometry import (
    calculate_directional_stop_and_validate, floor_to_tick_float,
)

logger = logging.getLogger(__name__)


STRATEGY_NAME = "MEAN_REVERSION"


class MeanReversionStrategy:
    """Mean Reversion Strategy tailored for range-bound (SIDEWAYS) market regimes."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config

    def generate_signal(self, symbol: str, df: pd.DataFrame, market_texture: str = "SIDEWAYS",
                        *, quote: Optional[dict] = None, metadata: Optional[dict] = None) -> Optional[TradeSignal]:
        """Evaluates DataFrame against 4-layer confirmation checks for Mean Reversion BUY signal.
        
        Returns TradeSignal if all confirmation layers pass, or None.
        """
        # Only operate in allowed market textures (SIDEWAYS, TRANSITIONING)
        allowed_textures = ("SIDEWAYS", "TRANSITIONING")
        if market_texture not in allowed_textures:
            logger.debug("[MEAN REVERSION] %s: Skipped -- active market texture '%s' is not in %s", symbol, market_texture, allowed_textures)
            return None

        if df is None or df.empty or len(df) < 20:
            logger.debug("[MEAN REVERSION] %s: Insufficient data bars (%d)", symbol, len(df) if df is not None else 0)
            return None

        try:
            close = float(df["close"].iloc[-1])
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            if not math.isfinite(close) or close <= 0:
                return None

            # Calculate 14-period ATR
            high_low = high - low
            high_cp = (high - df["close"].shift(1)).abs()
            low_cp = (low - df["close"].shift(1)).abs()
            tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            atr = float(tr.tail(14).mean())
            if not math.isfinite(atr) or atr <= 0:
                return None

            # Calculate VWAP
            if "vwap" in df.columns:
                vwap = float(df["vwap"].iloc[-1])
            else:
                typical_price = (high + low + df["close"]) / 3.0
                vwap = float((typical_price * volume).tail(20).sum() / volume.tail(20).sum())

            # Calculate 14-period RSI
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).tail(14).mean()
            loss = (-delta.where(delta < 0, 0)).tail(14).mean()
            rs = gain / loss if loss > 0 else 100.0
            rsi = float(100.0 - (100.0 / (1.0 + rs)))

            # Calculate 20-period Bollinger Bands
            sma20 = float(df["close"].tail(20).mean())
            std20 = float(df["close"].tail(20).std())
            lower_bb = sma20 - (2.0 * std20)

            # Calculate Volume Ratio
            avg_vol = float(volume.tail(20).mean())
            last_vol = float(volume.iloc[-1])
            volume_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

            if not all(math.isfinite(value) for value in (vwap, rsi, lower_bb, volume_ratio)):
                return None

            # ── 4 CONFIRMATION LAYERS ─────────────────────────
            # Layer 1: VWAP Deviation
            deviation = vwap - close
            required_dev = 1.2 * atr
            if deviation < required_dev:
                logger.debug(
                    "[MEAN REVERSION] %s: Rejection Layer 1 (VWAP Deviation %.4f < required %.4f)",
                    symbol, deviation, required_dev
                )
                return None

            # Layer 2: RSI Oversold
            rsi_threshold = getattr(self.cfg, "MEAN_REVERSION_RSI_MAX", 35.0)
            if rsi > rsi_threshold:
                logger.debug(
                    "[MEAN REVERSION] %s: Rejection Layer 2 (RSI %.1f > threshold %.1f)",
                    symbol, rsi, rsi_threshold
                )
                return None

            # Layer 3: Volume Spike Guard (no heavy volume breakdown)
            max_vol_ratio = getattr(self.cfg, "MEAN_REVERSION_MAX_VOL_RATIO", 2.0)
            if volume_ratio > max_vol_ratio:
                logger.debug(
                    "[MEAN REVERSION] %s: Rejection Layer 3 (Volume Ratio %.2f > max %.2f)",
                    symbol, volume_ratio, max_vol_ratio
                )
                return None

            # Layer 4: Bollinger Band Hug
            if close > lower_bb:
                logger.debug(
                    "[MEAN REVERSION] %s: Rejection Layer 4 (Close $%.2f > Lower BB $%.2f)",
                    symbol, close, lower_bb
                )
                return None

            # Preserve the four entry layers; harmonize only execution geometry.
            context = dict(getattr(df, "attrs", {}) or {})
            context.update(metadata or {})
            quote = quote if quote is not None else context.get("quote")
            spread_floor = 0.0
            if quote is not None:
                bid = float(quote.get("bid", 0.0) or quote.get("bid_price", 0.0) or 0.0)
                ask = float(quote.get("ask", 0.0) or quote.get("ask_price", 0.0) or 0.0)
                if not (math.isfinite(bid) and math.isfinite(ask) and ask >= bid > 0):
                    logger.debug("[MEAN REVERSION] %s: Invalid supplied quote", symbol)
                    return None
                spread_floor = ask - bid

            # Use the emitted entry precision consistently; do not round ATR/risk independently.
            entry_price = round(close, 4)
            tick_size = 0.01 if entry_price >= 1.0 else 0.0001
            geom = calculate_directional_stop_and_validate(
                side="long", price=entry_price, raw_risk=0.50 * atr,
                atr=atr, spread_floor=spread_floor, tick_size=tick_size,
                precision=2 if tick_size == 0.01 else 4, cfg=self.cfg,
            )
            stop_price = geom.stop_price
            # Conservative tick alignment: never move a long target above VWAP.
            target_price = floor_to_tick_float(vwap, tick_size)
            risk_per_share = entry_price - stop_price
            reward_per_share = target_price - entry_price
            rr_ratio = reward_per_share / risk_per_share
            min_rr = float(getattr(self.cfg, "MIN_RR_RATIO", 1.50))
            max_rr = float(getattr(self.cfg, "MAX_TARGET_RR_RATIO", 5.00))
            if not (math.isfinite(min_rr) and math.isfinite(max_rr) and 0 < min_rr <= max_rr):
                raise ValueError("Invalid R:R configuration")
            if rr_ratio < (min_rr - 1e-6):
                logger.debug("[MEAN REVERSION] %s: Insufficient R:R %.4f < %.4f", symbol, rr_ratio, min_rr)
                return None
            if rr_ratio > (max_rr + 1e-6):
                logger.debug("[MEAN REVERSION] %s: Unreachable target R:R %.4f > %.4f", symbol, rr_ratio, max_rr)
                return None

            daily_atr = context.get("daily_atr")
            if daily_atr is not None:
                daily_atr = float(daily_atr)
                if not (math.isfinite(daily_atr) and daily_atr > 0):
                    return None
                max_reach = float(getattr(self.cfg, "MAX_TARGET_DAILY_ATR_MULTIPLE", 3.5))
                if not (math.isfinite(max_reach) and max_reach > 0):
                    return None
                if reward_per_share > daily_atr * max_reach:
                    logger.debug("[MEAN REVERSION] %s: Target exceeds daily ATR reachability", symbol)
                    return None

            from strategies.constants import StrategyName
            signal = TradeSignal(
                symbol=symbol,
                strategy=StrategyName.MEAN_REVERSION,
                side="BUY",
                confidence=0.65,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                atr=atr,
                risk_per_share=risk_per_share,
                reason=f"Mean Reversion 4-Layer Pass: Dev=${deviation:.2f} (>=1.2xATR), RSI={rsi:.1f}<=35, VolRatio={volume_ratio:.2f}<=2.0, Close<LowerBB",
                metadata={
                    **context,
                    **({"quote": quote} if quote is not None else {}),
                    "timeframe": "5m",
                    "target_anchor": "VWAP",
                    "geometry_version": "MR_v0.3",
                    "spread_floor": spread_floor,
                    "minimum_stop_distance": geom.minimum_distance,
                    "rsi": rsi,
                    "volume_ratio": volume_ratio,
                    "vwap": vwap,
                }
            )
            logger.info("[MEAN REVERSION SIGNAL] %s BUY Entry generated! %s", symbol, signal)
            return signal


        except Exception as exc:
            logger.error("[MEAN REVERSION] Error evaluating %s: %s", symbol, exc)
            return None
