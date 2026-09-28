"""
strategies.shadow.mean_reversion_shadow -- Isolated Mean Reversion Shadow Experiment Runner.

Evaluates Mean Reversion candidates under shadow lane isolation:
- Production default is strictly 1.2 * ATR (zero production impact).
- Experimental shadow evaluation tests 0.8 * ATR.
- Tracks breakdown characteristics (making new low, ATR expansion, Lower BB penetration).
- Measures R:R geometry and forward outcome classification (REVERSION vs BREAKDOWN).
- Emits structured shadow candidate telemetry with zero live broker execution.
"""
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

import config
from strategies.indicators import add_atr, add_ema, add_vwap

logger = logging.getLogger(__name__)

# Canonical shadow threshold for D2 experiment (deviation from VWAP)
SHADOW_DEVIATION_MULT: float = 0.8


@dataclass
class MeanReversionShadowResult:
    symbol: str
    bar_timestamp: str
    close: float
    vwap: float
    atr: float
    deviation: float
    deviation_ratio: float
    l1_passed: bool
    l2_passed: bool
    l3_passed: bool
    l4_passed: bool
    all_layers_passed: bool
    rr_ratio: float
    making_new_low: bool
    atr_expanding: bool
    lower_bb_break: bool
    rejection_layer: Optional[str] = None
    outcome: Optional[str] = None  # "REVERSION" or "BREAKDOWN"


class MeanReversionShadowRunner:
    """Isolated shadow evaluation runner for Mean Reversion strategy experiments."""

    def __init__(self, deviation_mult: Optional[float] = None):
        if deviation_mult is None:
            self.deviation_mult = float(getattr(config, "EXPERIMENTAL_MEAN_REVERSION_ATR_DEV", 1.2))
        else:
            self.deviation_mult = float(deviation_mult)

    def evaluate_bar(self, symbol: str, df: pd.DataFrame) -> Optional[MeanReversionShadowResult]:
        """Evaluate a single bar DataFrame for Mean Reversion under shadow rules."""
        if df is None or len(df) < 20:
            return None

        # Normalize column casing
        df = df.copy()
        col_map = {c: c.lower() for c in df.columns}
        df.rename(columns=col_map, inplace=True)

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return None

        high = df["high"]
        low = df["low"]
        close = float(df["close"].iloc[-1])
        volume = df["volume"]

        # ATR 14
        tr = pd.concat(
            [high - low, (high - df["close"].shift(1)).abs(), (low - df["close"].shift(1)).abs()],
            axis=1,
        ).max(axis=1)
        atr = float(tr.tail(14).mean())
        if atr <= 0:
            return None

        prior_atr = float(tr.iloc[-28:-14].mean()) if len(tr) >= 28 else atr
        atr_expanding = bool(atr > (1.05 * prior_atr))

        # VWAP
        if "vwap" in df.columns:
            vwap = float(df["vwap"].iloc[-1])
        else:
            typical_price = (high + low + df["close"]) / 3.0
            vwap = float((typical_price * volume).tail(20).sum() / volume.tail(20).sum()) if volume.tail(20).sum() > 0 else close

        # 14-period RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).tail(14).mean()
        loss = (-delta.where(delta < 0, 0)).tail(14).mean()
        rs = gain / loss if loss > 0 else 100.0
        rsi = float(100.0 - (100.0 / (1.0 + rs)))

        # 20-period Bollinger Bands
        sma20 = float(df["close"].tail(20).mean())
        std20 = float(df["close"].tail(20).std())
        lower_bb = sma20 - (2.0 * std20)
        lower_bb_break = bool(close <= lower_bb)

        # Volume Ratio
        avg_vol = float(volume.tail(20).mean())
        last_vol = float(volume.iloc[-1])
        volume_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        # Making new low (relative to prior 20 bars)
        min_prior_low = float(low.iloc[-21:-1].min()) if len(df) >= 21 else float(low.iloc[-1])
        making_new_low = bool(float(low.iloc[-1]) <= min_prior_low)

        # 4 Confirmation Layers
        deviation = vwap - close
        required_dev = self.deviation_mult * atr
        deviation_ratio = deviation / atr if atr > 0 else 0.0

        l1_passed = bool(deviation >= required_dev)
        rsi_max = float(getattr(config, "MEAN_REVERSION_RSI_MAX", 35.0))
        l2_passed = bool(rsi <= rsi_max)
        vol_max = float(getattr(config, "MEAN_REVERSION_MAX_VOL_RATIO", 2.0))
        l3_passed = bool(volume_ratio <= vol_max)
        l4_passed = bool(close <= lower_bb)

        all_layers_passed = l1_passed and l2_passed and l3_passed and l4_passed

        rejection_layer = None
        if not l1_passed:
            rejection_layer = "Layer 1 (VWAP Deviation)"
        elif not l2_passed:
            rejection_layer = "Layer 2 (RSI Oversold)"
        elif not l3_passed:
            rejection_layer = "Layer 3 (Volume Spike Guard)"
        elif not l4_passed:
            rejection_layer = "Layer 4 (Bollinger Band Hug)"

        # Target & Stop Geometry: Stop = Close - 0.5*ATR, Target = VWAP
        stop_dist = 0.5 * atr
        reward = deviation
        rr_ratio = reward / stop_dist if stop_dist > 0 else 0.0

        bar_ts = str(df.index[-1]) if hasattr(df.index[-1], "isoformat") else str(df.index[-1])

        return MeanReversionShadowResult(
            symbol=symbol,
            bar_timestamp=bar_ts,
            close=close,
            vwap=vwap,
            atr=atr,
            deviation=deviation,
            deviation_ratio=deviation_ratio,
            l1_passed=l1_passed,
            l2_passed=l2_passed,
            l3_passed=l3_passed,
            l4_passed=l4_passed,
            all_layers_passed=all_layers_passed,
            rr_ratio=rr_ratio,
            making_new_low=making_new_low,
            atr_expanding=atr_expanding,
            lower_bb_break=lower_bb_break,
            rejection_layer=rejection_layer,
        )


def log_mr_shadow_candidate(
    result: MeanReversionShadowResult,
    market_texture: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> None:
    """Appends an isolated MR shadow candidate record to JSONL (Zero Broker Pollution)."""
    if output_path is None:
        base_dir = Path(getattr(config, "SHADOW_BASE_DIR", "data/shadow"))
        output_path = base_dir / "mean_reversion_shadow.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "symbol": result.symbol,
        "bar_timestamp": result.bar_timestamp,
        "close": result.close,
        "vwap": result.vwap,
        "atr": result.atr,
        "deviation": result.deviation,
        "deviation_ratio": result.deviation_ratio,
        "l1_passed": result.l1_passed,
        "l2_passed": result.l2_passed,
        "l3_passed": result.l3_passed,
        "l4_passed": result.l4_passed,
        "all_layers_passed": result.all_layers_passed,
        "rr_ratio": result.rr_ratio,
        "making_new_low": result.making_new_low,
        "atr_expanding": result.atr_expanding,
        "lower_bb_break": result.lower_bb_break,
        "rejection_layer": result.rejection_layer,
        "market_texture": market_texture or "UNKNOWN",
    }

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
