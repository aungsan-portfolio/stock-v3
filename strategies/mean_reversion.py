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
from typing import Optional
import pandas as pd
import numpy as np

import config
from strategies.base import TradeSignal

logger = logging.getLogger(__name__)


STRATEGY_NAME = "MEAN_REVERSION"


class MeanReversionStrategy:
    """Mean Reversion Strategy tailored for range-bound (SIDEWAYS) market regimes."""

    def __init__(self, cfg=None):
        self.cfg = cfg or config

    def generate_signal(self, symbol: str, df: pd.DataFrame, market_texture: str = "SIDEWAYS") -> Optional[TradeSignal]:
        """Evaluates DataFrame against 4-layer confirmation checks for Mean Reversion BUY signal.
        
        Returns TradeSignal if all confirmation layers pass, or None.
        """
        # Only operate in SIDEWAYS market texture
        if market_texture != "SIDEWAYS":
            logger.debug("[MEAN REVERSION] %s: Skipped -- active market texture '%s' is not SIDEWAYS", symbol, market_texture)
            return None

        if df is None or df.empty or len(df) < 20:
            logger.debug("[MEAN REVERSION] %s: Insufficient data bars (%d)", symbol, len(df) if df is not None else 0)
            return None

        try:
            close = float(df["close"].iloc[-1])
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            if close <= 0:
                return None

            # Calculate 14-period ATR
            high_low = high - low
            high_cp = (high - df["close"].shift(1)).abs()
            low_cp = (low - df["close"].shift(1)).abs()
            tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
            atr = float(tr.tail(14).mean())
            if atr <= 0:
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

            # Calculate Target & Stop Geometry
            stop_price = close - (0.5 * atr)
            target_price = vwap
            risk_per_share = close - stop_price
            reward_per_share = target_price - close
            rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0.0

            # Noise floor consistency check at signal generation layer
            abs_min_noise = getattr(config, "NOISE_FLOOR_ABS_MIN", 0.08)
            hard_veto_ratio = getattr(config, "NOISE_FLOOR_HARD_VETO_RATIO", 0.50)
            min_allowable_stop = abs_min_noise * hard_veto_ratio
            if risk_per_share < min_allowable_stop:
                logger.debug(
                    "[MEAN REVERSION] %s: Rejection Noise Floor (Stop distance $%.4f < Min allowable $%.4f)",
                    symbol, risk_per_share, min_allowable_stop
                )
                return None

            if rr_ratio < 1.2:
                logger.debug("[MEAN REVERSION] %s: Rejection Geometry (RR %.2f < 1.2)", symbol, rr_ratio)
                return None

            from strategies.constants import StrategyName
            signal = TradeSignal(
                symbol=symbol,
                strategy=StrategyName.MEAN_REVERSION,
                side="BUY",
                confidence=0.65,
                entry_price=round(close, 4),
                stop_price=round(stop_price, 4),
                target_price=round(target_price, 4),
                atr=round(atr, 4),
                risk_per_share=round(risk_per_share, 4),
                reason=f"Mean Reversion 4-Layer Pass: Dev=${deviation:.2f} (>=1.2xATR), RSI={rsi:.1f}<=35, VolRatio={volume_ratio:.2f}<=2.0, Close<LowerBB",
                metadata={
                    "timeframe": "5m",
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
