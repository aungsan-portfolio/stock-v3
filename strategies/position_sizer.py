"""
position_sizer.py -- Calculate position size based on risk parameters.

Supports:
  1. risk_based: size = (equity * max_risk_pct) / risk_per_share
  2. atr_adjusted_risk: size = (equity * max_risk_pct) / (atr * multiplier)
  3. fixed_shares: always FIXED_SHARE_COUNT
  4. fixed_dollars: FIXED_DOLLAR_AMOUNT / entry_price
"""
import logging
import math

import config
from strategies.base import TradeSignal

logger = logging.getLogger(__name__)


def validate_executable_geometry(
    signal: TradeSignal,
    atr_hard_veto_line: float = 0.0,
) -> tuple[bool, str]:
    """Validate signal pricing, directionality, and geometry before position sizing."""
    entry_price = getattr(signal, "entry_price", 0.0)
    stop_price = getattr(signal, "stop_price", 0.0)
    target_price = getattr(signal, "target_price", 0.0)
    symbol = getattr(signal, "symbol", "UNKNOWN")
    side = getattr(signal, "side", "").upper()

    tick_size = 0.01 if entry_price >= 1.0 else 0.0001

    values = {
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "tick_size": tick_size,
    }

    for name, value in values.items():
        if not math.isfinite(value):
            return False, f"{symbol}: {name} is not finite"

    if min(entry_price, stop_price, target_price, tick_size) <= 0:
        return False, f"{symbol}: prices and tick size must be positive"

    if side in {"BUY", "LONG"}:
        if not (stop_price < entry_price < target_price):
            return (
                False,
                f"{symbol}: invalid LONG geometry stop={stop_price:.4f} entry={entry_price:.4f} target={target_price:.4f}",
            )
    elif side in {"SELL", "SHORT"}:
        if not (target_price < entry_price < stop_price):
            return (
                False,
                f"{symbol}: invalid SHORT geometry target={target_price:.4f} entry={entry_price:.4f} stop={stop_price:.4f}",
            )
    else:
        return False, f"{symbol}: unsupported side={side}"

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= tick_size:
        return (
            False,
            f"{symbol}: stop distance {stop_distance:.6f} is at or below tick size {tick_size:.4f}",
        )

    if atr_hard_veto_line > 0 and stop_distance < atr_hard_veto_line:
        return (
            False,
            f"{symbol}: stop distance {stop_distance:.6f} is below ATR minimum {atr_hard_veto_line:.4f}",
        )

    reward_distance = abs(target_price - entry_price)
    reward_risk = reward_distance / stop_distance
    if not math.isfinite(reward_risk):
        return False, f"{symbol}: reward-risk ratio is not finite"

    return True, "OK"


def calculate_shares(signal: TradeSignal, equity: float) -> int:
    if not math.isfinite(equity) or equity <= 0:
        return 0
    valid, reason = validate_executable_geometry(signal)
    if not valid:
        logger.warning("[SIZER REJECT] %s", reason)
        return 0

    tick_size = 0.01 if signal.entry_price >= 1.0 else 0.0001
    risk_per_share = abs(signal.entry_price - signal.stop_price)

    # Fail-closed guard against non-finite or sub-tick stop gaps
    if not math.isfinite(risk_per_share) or risk_per_share <= tick_size:
        logger.warning(
            "[SIZER REJECT] %s invalid geometry: side=%s entry=%.4f stop=%.4f gap=%.6f <= tick=%.4f",
            signal.symbol, getattr(signal, "side", ""), signal.entry_price, signal.stop_price,
            risk_per_share, tick_size,
        )
        return 0

    if not (math.isfinite(signal.risk_per_share) and signal.risk_per_share > 0
            and math.isclose(signal.risk_per_share, risk_per_share, rel_tol=1e-6, abs_tol=1e-6)):
        logger.warning("[SIZER REJECT] %s inconsistent risk/share", signal.symbol)
        return 0

    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy)
    risk_pct = (float(strat_cfg.get("max_risk_pct", 1.0)) if isinstance(strat_cfg, dict)
                else float(getattr(strat_cfg, "max_risk_pct", 1.0)))
    if not (math.isfinite(risk_pct) and risk_pct > 0):
        return 0
    risk_budget = equity * (risk_pct / 100.0)

    method = config.SIZING_METHOD

    if method == "atr_adjusted_risk":
        atr = getattr(signal, "atr", None)
        if atr is None or not math.isfinite(atr) or atr <= 0:
            logger.warning("ATR is missing, NaN, or <= 0 for %s. Skipping trade.", signal.symbol)
            return 0

        min_atr = getattr(config, "MIN_ATR_THRESHOLD", 0.01)
        if atr < min_atr:
            logger.warning("ATR %.4f is below minimum threshold %.4f for %s. Skipping trade.", atr, min_atr, signal.symbol)
            return 0

        multiplier = getattr(config, "ATR_SIZING_MULTIPLIER", getattr(config, "TRAILING_STOP_ATR_MULTIPLE", 1.5))
        if not (math.isfinite(multiplier) and multiplier > 0):
            logger.warning("Invalid ATR sizing multiplier %.2f; rejecting size", multiplier)
            return 0

        raw = risk_budget / (atr * multiplier)
    elif method == "risk_based":
        raw = risk_budget / risk_per_share
    elif method == "fixed_shares":
        raw = config.FIXED_SHARE_COUNT
    elif method == "fixed_dollars":
        raw = config.FIXED_DOLLAR_AMOUNT / signal.entry_price
    else:
        logger.warning("Unknown sizing method %s, using risk_based", method)
        raw = risk_budget / risk_per_share

    # Volatility Risk Parity Sizing (Cap per-trade dollar risk at MAX_RISK_PER_TRADE_DOLLARS)
    max_risk_dollars = getattr(config, "MAX_RISK_PER_TRADE_DOLLARS", 25.0)
    risk_parity_shares = max_risk_dollars / risk_per_share

    dollar_cap_shares = config.MAX_POSITION_SIZE_DOLLARS / signal.entry_price
    is_inverse = signal.symbol.upper().strip() in set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
    if is_inverse:
        equity_cap_shares = equity / signal.entry_price
    else:
        equity_cap_shares = (equity * config.MAX_POSITION_PCT_OF_EQUITY) / signal.entry_price

    strategy_risk_shares = risk_budget / risk_per_share
    limits = (raw, risk_parity_shares, dollar_cap_shares, equity_cap_shares, strategy_risk_shares)
    if not all(math.isfinite(value) and value >= 0 for value in limits):
        logger.warning("[SIZER REJECT] %s non-finite or negative sizing limits", signal.symbol)
        return 0
    shares = int(min(limits))
    shares = max(shares, 0)
    min_shares = getattr(config, "MIN_SHARE_COUNT", 1)
    if shares > 0 and shares < min_shares:
        logger.info("[SIZER REJECT] %s size %d below minimum %d; caps are not overridden",
                    signal.symbol, shares, min_shares)
        return 0

    if shares == 0:
        logger.info(
            "Position size = 0 for %s (raw=%.1f, dollar_cap=%.1f, equity_cap=%.1f)",
            signal.symbol, raw, dollar_cap_shares, equity_cap_shares,
        )
    else:
        logger.info(
            "[SIZER] %s | side=%s entry=%.2f stop=%.2f gap=%.4f (tick=%.4f) | risk_shares=%d dollar_shares=%d → final=%d shares (~$%.0f)",
            signal.symbol, getattr(signal, "side", "BUY"), signal.entry_price, signal.stop_price, risk_per_share, tick_size,
            int(risk_parity_shares), int(dollar_cap_shares), shares,
            shares * signal.entry_price,
        )

    return shares



def estimated_cost(signal: TradeSignal, shares: int) -> float:
    return shares * signal.entry_price


def estimated_risk(signal: TradeSignal, shares: int) -> float:
    return shares * signal.risk_per_share


def estimated_reward(signal: TradeSignal, shares: int) -> float:
    return shares * abs(signal.target_price - signal.entry_price)
