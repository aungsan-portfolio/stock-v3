# ─────────────────────────────────────────────────────────────────────────────
#  rs_calculator.py
#  Relative Strength Calculator
#
#  တာဝန်:
#    • RSI (Wilder's Smoothed) တွက်ချက်သည်
#    • Symbol vs Benchmark (SOXX) Alpha တွက်ချက်သည်
#    • SOXX Gate Bypass ဆုံးဖြတ်ချက် ပြန်ပေးသည်
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from strategies.soxx_gate_config import (
    RS_BYPASS_RSI_FLOOR,
    RS_BYPASS_ALPHA_FLOOR,
    RS_BYPASS_MIN_CONF,
    SOXX_HARD_VETO_FLOOR,
    RS_RSI_PERIOD,
    RS_ALPHA_LOOKBACK_BARS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data Class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RSResult:
    """Single symbol ၏ Relative Strength Analysis Output"""
    symbol:           str
    rsi:              float              # Wilder RSI (14-period)
    alpha_vs_soxx:    float              # Symbol % change - SOXX % change
    symbol_pct_chg:   float              # Symbol intraday % change
    soxx_pct_chg:     float              # SOXX intraday % change
    bypass_granted:   bool               # Final bypass decision
    bypass_reason:    str                # Human-readable explanation
    conf_ok:          bool               # Predictor confidence check passed
    rsi_ok:           bool               # RSI condition passed
    alpha_ok:         bool               # Alpha condition passed
    hard_veto:        bool               # Hard veto floor triggered


# ─────────────────────────────────────────────────────────────────────────────
#  RSI Calculator (Wilder's Smoothed Method)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_wilder_rsi(prices: pd.Series,
                          period: int = RS_RSI_PERIOD) -> float:
    """
    Wilder's Smoothed RSI တွက်ချက်သည်။
    Standard pandas-ta / ta-lib နှင့် တူညီသော algorithm ဖြစ်သည်။

    Args:
        prices: Close prices (oldest → newest), minimum length = period + 1
        period: Lookback periods (default 14)

    Returns:
        float: RSI value 0–100, သို့မဟုတ် 50.0 (neutral) if insufficient data
    """
    if len(prices) < period + 1:
        logger.warning(
            "[RSI] Insufficient data (%d bars, need %d). "
            "Returning neutral 50.0",
            len(prices), period + 1
        )
        return 50.0

    delta  = prices.diff().dropna()
    gains  = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # Initial averages (simple mean for seed)
    avg_gain = gains.iloc[:period].mean()
    avg_loss = losses.iloc[:period].mean()

    # Wilder smoothing for remaining bars
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains.iloc[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses.iloc[i]) / period

    if avg_loss == 0:
        return 100.0

    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return round(rsi, 2)


# ─────────────────────────────────────────────────────────────────────────────
#  Alpha Calculator
# ─────────────────────────────────────────────────────────────────────────────

def calculate_alpha_vs_benchmark(
    symbol_df:    pd.DataFrame,
    benchmark_df: pd.DataFrame,
    lookback_bars: int = RS_ALPHA_LOOKBACK_BARS,
) -> tuple[float, float, float]:
    """
    Symbol ၏ intraday performance ကို SOXX benchmark နှင့် နှိုင်းယှဉ်သည်။

    Returns:
        (alpha, symbol_pct_chg, soxx_pct_chg)
        alpha = symbol_pct_chg - soxx_pct_chg
    """
    try:
        sym_open   = symbol_df["Open"].iloc[-(lookback_bars + 1)]
        sym_close  = symbol_df["Close"].iloc[-1]
        sym_chg    = (sym_close - sym_open) / sym_open

        soxx_open  = benchmark_df["Open"].iloc[-(lookback_bars + 1)]
        soxx_close = benchmark_df["Close"].iloc[-1]
        soxx_chg   = (soxx_close - soxx_open) / soxx_open

        alpha = sym_chg - soxx_chg
        return round(alpha, 6), round(sym_chg, 6), round(soxx_chg, 6)

    except (IndexError, ZeroDivisionError) as exc:
        logger.warning("[ALPHA CALC] Failed: %s — returning zero alpha.", exc)
        return 0.0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Main RS Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class RelativeStrengthEvaluator:
    """
    Usage (intraday_risk.py ထဲမှ):

        evaluator = RelativeStrengthEvaluator()

        result = evaluator.evaluate(
            symbol      = "NVDA",
            symbol_df   = nvda_ohlcv_df,
            soxx_df     = soxx_ohlcv_df,
            soxx_chg_pct= -0.015,           # SOXX daily % change
            pred_conf   = 0.58,             # Predictor confidence
        )

        if result.bypass_granted:
            # SOXX Gate ကို Bypass — entry ခွင့်ပြု
    """

    def evaluate(
        self,
        symbol:        str,
        symbol_df:     pd.DataFrame,
        soxx_df:       pd.DataFrame,
        soxx_chg_pct:  float,
        pred_conf:     float,
    ) -> RSResult:
        """
        SOXX Gate Bypass ကို ဆုံးဖြတ်သည်။

        Bypass ခွင့်ပြုရန် အောက်ပါ အားလုံး ပြည့်မှီရမည်:
            1. SOXX Hard Veto Floor မကျော်ခြင်း  (Absolute safety)
            2. Predictor Confidence >= RS_BYPASS_MIN_CONF
            3. RSI >= RS_BYPASS_RSI_FLOOR  (Condition A)
               OR Alpha vs SOXX >= RS_BYPASS_ALPHA_FLOOR  (Condition B)
        """

        # ── Step 1: Calculate RSI ────────────────────────────────────────────
        rsi = calculate_wilder_rsi(symbol_df["Close"])

        # ── Step 2: Calculate Alpha ──────────────────────────────────────────
        alpha, sym_chg, soxx_chg = calculate_alpha_vs_benchmark(
            symbol_df, soxx_df
        )

        # ── Step 3: Evaluate conditions ──────────────────────────────────────

        # Hard Veto (Absolute — overrides everything)
        hard_veto = soxx_chg_pct <= SOXX_HARD_VETO_FLOOR
        if hard_veto:
            reason = (
                f"HARD VETO ACTIVE: SOXX={soxx_chg_pct*100:.2f}% "
                f"≤ floor {SOXX_HARD_VETO_FLOOR*100:.1f}% — "
                f"No bypass permitted regardless of RS."
            )
            logger.warning("[SOXX HARD VETO] %s — %s", symbol, reason)
            return RSResult(
                symbol=symbol, rsi=rsi,
                alpha_vs_soxx=alpha,
                symbol_pct_chg=sym_chg, soxx_pct_chg=soxx_chg,
                bypass_granted=False, bypass_reason=reason,
                conf_ok=False, rsi_ok=False,
                alpha_ok=False, hard_veto=True,
            )

        # Confidence check (AND — must always pass)
        conf_ok = pred_conf >= RS_BYPASS_MIN_CONF

        # RSI condition (A)
        rsi_ok  = rsi >= RS_BYPASS_RSI_FLOOR

        # Alpha condition (B)
        alpha_ok = alpha >= RS_BYPASS_ALPHA_FLOOR

        # Final bypass decision
        # Must pass: conf_ok AND (rsi_ok OR alpha_ok)
        bypass_granted = conf_ok and (rsi_ok or alpha_ok)

        # Build human-readable reason
        reason = self._build_reason(
            symbol       = symbol,
            soxx_chg_pct = soxx_chg_pct,
            rsi          = rsi,
            alpha        = alpha,
            pred_conf    = pred_conf,
            conf_ok      = conf_ok,
            rsi_ok       = rsi_ok,
            alpha_ok     = alpha_ok,
            bypass_granted = bypass_granted,
        )

        logger.info(
            "[SOXX RS EVAL] %s | RSI=%.1f(%s) | Alpha=%.2f%%(%s) | "
            "Conf=%.2f(%s) → Bypass=%s",
            symbol, rsi, "✅" if rsi_ok else "❌",
            alpha * 100, "✅" if alpha_ok else "❌",
            pred_conf, "✅" if conf_ok else "❌",
            "GRANTED ✅" if bypass_granted else "DENIED ❌"
        )

        return RSResult(
            symbol=symbol, rsi=rsi,
            alpha_vs_soxx=alpha,
            symbol_pct_chg=sym_chg, soxx_pct_chg=soxx_chg,
            bypass_granted=bypass_granted, bypass_reason=reason,
            conf_ok=conf_ok, rsi_ok=rsi_ok,
            alpha_ok=alpha_ok, hard_veto=False,
        )

    @staticmethod
    def _build_reason(
        symbol, soxx_chg_pct, rsi, alpha,
        pred_conf, conf_ok, rsi_ok, alpha_ok, bypass_granted
    ) -> str:
        parts = [
            f"SOXX={soxx_chg_pct*100:.2f}%",
            f"RSI={rsi:.1f}(need≥{RS_BYPASS_RSI_FLOOR})",
            f"Alpha={alpha*100:.2f}%(need≥{RS_BYPASS_ALPHA_FLOOR*100:.1f}%)",
            f"Conf={pred_conf:.2f}(need≥{RS_BYPASS_MIN_CONF})",
        ]
        verdict = "BYPASS GRANTED" if bypass_granted else "BYPASS DENIED"

        if not conf_ok:
            parts.append("→ Blocked: confidence too low")
        elif not (rsi_ok or alpha_ok):
            parts.append("→ Blocked: neither RSI nor Alpha condition met")
        else:
            met = []
            if rsi_ok:   met.append(f"RSI≥{RS_BYPASS_RSI_FLOOR}")
            if alpha_ok: met.append(f"Alpha≥{RS_BYPASS_ALPHA_FLOOR*100:.1f}%")
            parts.append(f"→ Conditions met: {', '.join(met)}")

        return f"{symbol} {verdict} | " + " | ".join(parts)
