"""
gap_strategy.py -- Gap-and-Go strategy.
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
from strategies.intraday_data import previous_close

logger = logging.getLogger(__name__)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Compute a simple rolling ATR from complete OHLC bars using True Range with previous close."""
    required = {"high", "low", "close"}
    if df is None or df.empty or not required.issubset(df.columns):
        return 0.0

    bars = df.loc[:, ["high", "low", "close"]].copy()
    bars = bars.apply(pd.to_numeric, errors="coerce")
    bars = bars.dropna()

    if len(bars) < period + 1:
        return 0.0

    previous_close = bars["close"].shift(1)

    true_range = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - previous_close).abs(),
            (bars["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        window=period,
        min_periods=period,
    ).mean().iloc[-1]

    if not math.isfinite(atr) or atr <= 0.01:
        return 0.0

    return float(atr)


def _safe_atr(
    intraday_df: pd.DataFrame,
    full_df: pd.DataFrame,
    period: int = 14,
    return_source: bool = False,
):
    """Use current session ATR, then fall back to complete full-history bars.

    Returns:
        float if return_source is False, or (float, str) if return_source is True.
        Source is one of: 'INTRADAY_CURRENT', 'FULL_HISTORY_FALLBACK', 'UNAVAILABLE'.
    """
    if intraday_df is not None and "atr" in intraday_df.columns:
        valid_atr = pd.to_numeric(
            intraday_df["atr"],
            errors="coerce",
        ).dropna()
        if not valid_atr.empty:
            current_atr = float(valid_atr.iloc[-1])
            if math.isfinite(current_atr) and current_atr > 0.01:
                return (current_atr, "INTRADAY_CURRENT") if return_source else current_atr

    fallback_atr = _compute_atr(full_df, period=period)
    if fallback_atr > 0.01:
        return (fallback_atr, "FULL_HISTORY_FALLBACK") if return_source else fallback_atr

    return (0.0, "UNAVAILABLE") if return_source else 0.0


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

        full_df = df.copy() if df is not None else None
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
        atr, atr_source = _safe_atr(
            intraday_df=df,
            full_df=full_df,
            period=getattr(config, "ATR_PERIOD", 14),
            return_source=True,
        )
        logger.info(
            "[ATR RESOLUTION] %s atr=%.4f source=%s session_bars=%d history_bars=%d",
            symbol,
            atr,
            atr_source,
            len(df),
            len(full_df) if full_df is not None else 0,
        )
        rsi = float(last.get("rsi", 50))
        volume = float(last.get("volume", 0))
        vol_ma = float(last.get("volume_ma", 1))

        if atr < 0.01 or not math.isfinite(atr):
            logger.warning("[GAP QUALITY FAIL] %s: ATR=%.4f invalid (< $0.01 threshold). Rejecting.", symbol, atr)
            return None

        rel_vol = volume / vol_ma if vol_ma > 0 else 0

        # ── Gap Quality Filter (Wall-Street Institutional Conviction Gates) ──
        # 1. RVOL Gate: Require relative volume >= config.GAP_MIN_RVOL (default 1.5x)
        min_rvol = getattr(config, "GAP_MIN_RVOL", 1.5)
        if rel_vol < min_rvol:
            logger.debug("[GAP QUALITY VETO] %s rel_vol %.2fx < required min %.2fx", symbol, rel_vol, min_rvol)
            return None

        # 2. ATR Dollar Magnitude Gate: Require gap dollar size >= 0.5x ATR(14)
        min_atr_mult = getattr(config, "GAP_MIN_ATR_MULTIPLE", 0.5)
        gap_dollars = abs(open_price - prev)
        required_gap_dollars = min_atr_mult * atr
        if gap_dollars < required_gap_dollars:
            logger.debug("[GAP QUALITY VETO] %s gap dollar $%.2f < required 0.5x ATR $%.2f", symbol, gap_dollars, required_gap_dollars)
            return None

        logger.info("[GAP QUALITY PASS] %s: gap $%.2f (>=0.5x ATR $%.2f), rvol %.2fx (>=1.5x)", symbol, gap_dollars, required_gap_dollars, rel_vol)

        if gap_pct > 0 and price > vwap and price >= open_price:
            hold_bars = config.GAP_CONTINUATION_MIN_HOLD_BARS
            recent = df.tail(hold_bars)
            if len(recent) < hold_bars:
                logger.info("[GAP CONTINUATION VETO] %s reason=insufficient_hold_bars actual=%d required=%d", symbol, len(recent), hold_bars)
                return None

            held_above_vwap = all(
                float(recent["close"].iloc[i]) >= float(recent["vwap"].iloc[i])
                for i in range(len(recent))
                if not np.isnan(recent["vwap"].iloc[i])
            )

            if not held_above_vwap:
                logger.info("[GAP CONTINUATION VETO] %s reason=unheld_vwap hold_bars=%d", symbol, hold_bars)
                return None

            first_bar_low = float(df["low"].iloc[0])
            stop = max(first_bar_low, vwap) - 0.02
            risk = price - stop
            if risk <= 0:
                logger.info("[GAP CONTINUATION VETO] %s reason=nonpositive_risk price=%.4f stop=%.4f risk=%.4f", symbol, price, stop, risk)
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

            if hasattr(df, "index") and len(df.index) > 0:
                last_idx = df.index[-1]
                bar_ts = last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx)
            else:
                bar_ts = str(last.get("timestamp", ""))

            logger.info("[GAP SIGNAL EMITTED] %s strategy=GAP_AND_GO gap_pct=%.4f price=%.4f vwap=%.4f open=%.4f risk=%.4f", symbol, gap_pct, price, vwap, open_price, risk)
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
                bar_timestamp=bar_ts,
            )

        reason = "nonpositive_gap" if gap_pct <= 0 else ("not_above_vwap" if price <= vwap else "below_open_price")
        logger.info(
            "[GAP CONTINUATION VETO] %s reason=%s gap_pct=%.4f price=%.4f vwap=%.4f open=%.4f",
            symbol, reason, gap_pct, price, vwap, open_price,
        )
        return None
