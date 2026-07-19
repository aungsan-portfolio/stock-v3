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


def calculate_shares(signal: TradeSignal, equity: float) -> int:
    if equity <= 0:
        return 0
    if signal.risk_per_share <= 0:
        return 0
    if signal.entry_price <= 0:
        return 0

    method = config.SIZING_METHOD

    if method == "atr_adjusted_risk":
        import numpy as np
        atr = getattr(signal, "atr", None)
        if atr is None or np.isnan(atr) or atr <= 0:
            logger.warning("ATR is missing, NaN, or <= 0 for %s. Skipping trade.", signal.symbol)
            return 0

        min_atr = getattr(config, "MIN_ATR_THRESHOLD", 0.01)
        if atr < min_atr:
            logger.warning("ATR %.4f is below minimum threshold %.4f for %s. Skipping trade.", atr, min_atr, signal.symbol)
            return 0

        multiplier = getattr(config, "ATR_SIZING_MULTIPLIER", getattr(config, "TRAILING_STOP_ATR_MULTIPLE", 1.5))
        if multiplier <= 0:
            logger.warning("Invalid ATR sizing multiplier %.2f, defaulting to 1.5", multiplier)
            multiplier = 1.5

        strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy)
        risk_pct = strat_cfg.max_risk_pct if strat_cfg else 1.0
        risk_budget = equity * (risk_pct / 100.0)
        raw = risk_budget / (atr * multiplier)
    elif method == "risk_based":
        strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy)
        risk_pct = strat_cfg.max_risk_pct if strat_cfg else 1.0
        risk_budget = equity * (risk_pct / 100.0)
        raw = risk_budget / signal.risk_per_share
    elif method == "fixed_shares":
        raw = config.FIXED_SHARE_COUNT
    elif method == "fixed_dollars":
        raw = config.FIXED_DOLLAR_AMOUNT / signal.entry_price
    else:
        logger.warning("Unknown sizing method %s, using risk_based", method)
        strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy)
        risk_pct = strat_cfg.max_risk_pct if strat_cfg else 1.0
        risk_budget = equity * (risk_pct / 100.0)
        raw = risk_budget / signal.risk_per_share

    dollar_cap_shares = config.MAX_POSITION_SIZE_DOLLARS / signal.entry_price
    equity_cap_shares = (equity * config.MAX_POSITION_PCT_OF_EQUITY) / signal.entry_price

    shares = int(min(raw, dollar_cap_shares, equity_cap_shares))
    shares = max(shares, 0)

    if shares == 0:
        logger.info(
            "Position size = 0 for %s (raw=%.1f, dollar_cap=%.1f, equity_cap=%.1f)",
            signal.symbol, raw, dollar_cap_shares, equity_cap_shares,
        )

    return shares


def estimated_cost(signal: TradeSignal, shares: int) -> float:
    return shares * signal.entry_price


def estimated_risk(signal: TradeSignal, shares: int) -> float:
    return shares * signal.risk_per_share


def estimated_reward(signal: TradeSignal, shares: int) -> float:
    return shares * abs(signal.target_price - signal.entry_price)
