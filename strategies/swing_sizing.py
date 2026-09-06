"""
strategies/swing_sizing.py — Production Sizing Validator & Geometry Engine

Implements fail-closed quantitative sizing algorithms, whole-share lot policies,
asset-specific tick normalizations, composite idempotency guards, and zero-tolerance
portfolio risk enforcement for swing trading.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
from typing import Any, Dict, Optional, Set, Tuple


class PositionIntent(str, Enum):
    OPEN_LONG = "OPEN_LONG"
    ADD_LONG = "ADD_LONG"
    REDUCE_LONG = "REDUCE_LONG"
    CLOSE_LONG = "CLOSE_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    COVER_SHORT = "COVER_SHORT"


class UnsupportedSwingDirection(RuntimeError):
    """Raised when an unsupported direction (e.g. short selling in Stage 0) is requested."""
    pass


# Stage 0 Frozen Risk Constants
RISK_FRACTION = 0.005  # 0.50% equity risk budget
CONFIGURED_MAX_RISK_DOLLARS = 500.00  # Hard dollar ceiling at $100k equity
TARGET_DEPLOYED_EXPOSURE_PCT = 0.10  # 10% target deployment
STAGE_0_GROSS_EXPOSURE_PCT = 0.15   # 15% hard portfolio exposure ceiling
MAX_SYMBOL_EXPOSURE_PCT = 0.10      # 10% max per symbol
SECTOR_EXPOSURE_CAP_PCT = 0.25      # 25% max per sector
AGGREGATE_OPEN_STOP_HEAT_PCT = 0.025 # 2.50% max total open risk
MAX_CONCURRENT_POSITIONS = 7
MAX_DAILY_LOSS_DOLLARS = 300.00

ATR_MULTIPLIER_BY_REGIME = {
    "TRENDING_BULL": 1.50,
    "SIDEWAYS_NORMAL_VOL": 1.80,
    "SIDEWAYS_HIGH_VOL": 2.00,
    "SIDEWAYS_LOW_VOL": 1.50,
    "DEFAULT": 1.50,
}


class SwingSizingRejected(ValueError):
    """Raised when an order fails quantitative sizing validation."""
    pass


def require(condition: bool, message: str) -> None:
    """Fail-closed assertion helper for sizing and risk validation."""
    if not condition:
        raise SwingSizingRejected(message)


def normalize_long_stop_price(price: float, price_increment: float = 0.01) -> float:
    """
    Round price down conservatively to the valid asset price increment.
    Uses decimal floor arithmetic to avoid IEEE-754 floating point inaccuracies.
    """
    require(math.isfinite(price) and price > 0.0, f"Invalid price for tick normalization: {price}")
    require(math.isfinite(price_increment) and price_increment > 0.0, (
        f"Invalid price increment: {price_increment}"
    ))

    d_price = Decimal(str(round(price, 6)))
    d_inc = Decimal(str(round(price_increment, 6)))
    ticks = (d_price / d_inc).to_integral_value(rounding=ROUND_FLOOR)
    return float(ticks * d_inc)


# Backward-compatible alias
normalize_tick_price = normalize_long_stop_price


def calculate_swing_exit_lots(
    initial_shares: int,
    target_weight: float = 0.50,
) -> Dict[str, Any]:
    """
    Calculate integer share allocation for multi-tier exits.

    Rules:
      - If initial_shares < 2: 50/50 partial target is impossible.
        Disable partial target and execute as single-lot.
      - If initial_shares >= 2: floor integer split for Tier 1, remainder to runner.
    """
    require(isinstance(initial_shares, int) and initial_shares >= 1, (
        f"initial_shares must be integer >= 1, got {initial_shares}"
    ))

    if initial_shares < 2:
        return {
            "allow_partial_target": False,
            "intended_partial_weight": float(target_weight),
            "actual_partial_quantity": 0,
            "actual_partial_weight": 0.0,
            "runner_quantity": initial_shares,
            "runner_weight": 1.0,
        }

    partial_qty = int(math.floor(initial_shares * target_weight))
    # Guarantee at least 1 share in runner
    if partial_qty >= initial_shares:
        partial_qty = initial_shares - 1

    runner_qty = initial_shares - partial_qty
    return {
        "allow_partial_target": True,
        "intended_partial_weight": float(target_weight),
        "actual_partial_quantity": partial_qty,
        "actual_partial_weight": round(partial_qty / initial_shares, 4),
        "runner_quantity": runner_qty,
        "runner_weight": round(runner_qty / initial_shares, 4),
    }


def calculate_swing_shares(
    entry_price: float,
    stop_price: float,
    account_equity: float,
    proposed_risk_pct: float = 0.005,
    symbol_notional_cap: float = 12000.0,
    adv_shares_20: Optional[float] = None,
    max_participation_rate: float = 0.01,
    price_increment: float = 0.01,
    entry_slippage_per_share: float = 0.02,
    stop_slippage_per_share: float = 0.04,
    estimated_total_fees: float = 0.50,
) -> int:
    """
    Derive position share count strictly from account risk budget with pre-sizing headroom.
    Sizing budget uses 99% of hard risk limit to guarantee zero overshoot after fees/slippage.
    """
    require(math.isfinite(entry_price) and entry_price > 0.0, f"Invalid entry price: {entry_price}")
    require(math.isfinite(stop_price) and stop_price > 0.0, f"Invalid stop price: {stop_price}")
    require(math.isfinite(account_equity) and account_equity > 0.0, f"Invalid equity: {account_equity}")

    normalized_stop = normalize_long_stop_price(stop_price, price_increment=price_increment)
    base_risk_per_share = entry_price - normalized_stop
    require(base_risk_per_share > 0.0, f"Entry {entry_price} must exceed normalized stop {normalized_stop}")

    # Effective risk per share including slippage
    effective_risk_per_share = base_risk_per_share + entry_slippage_per_share + stop_slippage_per_share

    # Pre-sizing headroom: 99% of hard risk ceiling to absorb fixed fees
    hard_dollar_risk_limit = account_equity * proposed_risk_pct
    sizing_budget = max(0.0, (hard_dollar_risk_limit * 0.99) - estimated_total_fees)

    raw_shares = int(math.floor(sizing_budget / effective_risk_per_share))

    # Notional cap clipping
    notional_cap_shares = int(math.floor(symbol_notional_cap / entry_price))

    # Liquidity cap clipping
    if adv_shares_20 is not None and adv_shares_20 > 0.0:
        liquidity_cap_shares = int(math.floor(adv_shares_20 * max_participation_rate))
    else:
        liquidity_cap_shares = raw_shares

    final_shares = min(raw_shares, notional_cap_shares, liquidity_cap_shares)
    return max(0, final_shares)


def validate_swing_order_safety(
    client_order_id: str,
    strategy_instance_id: str,
    symbol: str,
    signal_version: str,
    entry_price: float,
    stop_price: float,
    shares: int,
    account_equity: float,
    available_buying_power: float,
    current_open_stop_risk: float,
    pending_entry_risk: float,
    side: str = "BUY",
    entry_intent_id: Optional[str] = None,
    active_entry_intents: Optional[Set[Tuple[str, str, str, str, str]]] = None,
    existing_working_symbols: Optional[Set[str]] = None,
    proposed_risk_per_trade_pct: float = 0.005,
    max_portfolio_heat_pct: float = 0.025,
    current_cluster_risk_pct: float = 0.0,
    max_cluster_risk_pct: float = 0.0125,
    adv_shares_20: float = 1_000_000.0,
    max_participation_rate: float = 0.01,
    symbol_notional_cap: float = 12000.0,
    price_increment: float = 0.01,
    quote_age_seconds: float = 5.0,
    max_quote_age_seconds: float = 60.0,
    entry_slippage_per_share: float = 0.02,
    stop_slippage_per_share: float = 0.04,
    estimated_total_fees: float = 0.50,
) -> None:
    """
    Production fail-closed validation for swing orders.
    Enforces exact risk ceilings (zero tolerance overshoot), composite idempotency keys,
    whole-share integrity, and portfolio heat limits before any broker dispatch.
    """
    # A. Account and buying power validation
    require(math.isfinite(account_equity) and account_equity > 0.0, f"Invalid account equity: {account_equity}")
    require(math.isfinite(available_buying_power) and available_buying_power >= 0.0, (
        f"Invalid buying power: {available_buying_power}"
    ))

    # B. Risk parameter ranges
    require(0.0 < proposed_risk_per_trade_pct <= 0.005, (
        f"Risk-per-trade exceeds research specification: {proposed_risk_per_trade_pct}"
    ))
    require(0.0 < max_portfolio_heat_pct <= 0.025, (
        f"Portfolio heat cap exceeds research specification: {max_portfolio_heat_pct}"
    ))
    require(0.0 < max_participation_rate <= 0.01, (
        f"Invalid participation rate: {max_participation_rate}"
    ))

    # C. Order deduplication & composite idempotency
    require(bool(client_order_id), "client_order_id must be provided")
    require(bool(strategy_instance_id), "strategy_instance_id must be provided")
    require(bool(signal_version), "signal_version must be provided")

    norm_sym = symbol.upper().strip()
    norm_side = side.upper().strip()
    intent_id = entry_intent_id or client_order_id

    # Composite idempotency key: (strategy_instance_id, symbol, side, signal_version, entry_intent_id)
    idempotency_key = (strategy_instance_id, norm_sym, norm_side, signal_version, intent_id)
    if active_entry_intents is not None:
        require(idempotency_key not in active_entry_intents, (
            f"Duplicate entry intent rejected for composite key: {idempotency_key}"
        ))

    if existing_working_symbols is not None and norm_side == "BUY":
        require(norm_sym not in existing_working_symbols, (
            f"Duplicate active working order rejected for symbol {norm_sym}"
        ))

    # D. Price and stop geometry
    require(math.isfinite(entry_price) and entry_price > 0.0, f"Invalid entry price: {entry_price}")
    require(math.isfinite(stop_price) and stop_price > 0.0, f"Invalid stop price: {stop_price}")

    normalized_stop = normalize_long_stop_price(stop_price, price_increment=price_increment)
    require(entry_price > normalized_stop, f"Long entry {entry_price} must exceed stop {normalized_stop}")
    require(isinstance(shares, int) and shares >= 1, f"Shares must be integer >= 1, got {shares}")

    # E. ADV and quote freshness
    require(math.isfinite(adv_shares_20) and adv_shares_20 > 0.0, f"Invalid ADV: {adv_shares_20}")
    require(quote_age_seconds <= max_quote_age_seconds, (
        f"Stale quote: {quote_age_seconds}s > {max_quote_age_seconds}s"
    ))

    # F. Notional limits
    notional_value = shares * entry_price
    require(notional_value >= 500.0, f"Order notional ${notional_value:.2f} below $500 minimum")
    require(notional_value <= symbol_notional_cap, (
        f"Order notional ${notional_value:.2f} exceeds cap ${symbol_notional_cap:.2f}"
    ))
    require(notional_value <= available_buying_power, (
        f"Notional ${notional_value:.2f} exceeds buying power ${available_buying_power:.2f}"
    ))

    # G. Liquidity participation cap
    max_shares_liquidity = int(math.floor(adv_shares_20 * max_participation_rate))
    require(shares <= max_shares_liquidity, (
        f"Shares {shares} exceeds ADV liquidity cap {max_shares_liquidity}"
    ))

    # H. Effective trade risk (per-share slippage model, ZERO tolerance overshoot)
    base_trade_risk = shares * (entry_price - normalized_stop)
    slippage_risk = (shares * entry_slippage_per_share) + (shares * stop_slippage_per_share)
    effective_trade_risk = base_trade_risk + slippage_risk + estimated_total_fees

    hard_trade_dollar_risk = account_equity * proposed_risk_per_trade_pct
    require(effective_trade_risk <= hard_trade_dollar_risk, (
        f"Effective trade risk ${effective_trade_risk:.2f} strictly exceeds hard budget ${hard_trade_dollar_risk:.2f}"
    ))

    # I. Projected portfolio heat (including working entries, excluding unfilled exits)
    projected_risk_heat = current_open_stop_risk + pending_entry_risk + effective_trade_risk
    max_portfolio_risk = account_equity * max_portfolio_heat_pct
    require(projected_risk_heat <= max_portfolio_risk, (
        f"Projected portfolio heat ${projected_risk_heat:.2f} exceeds limit ${max_portfolio_risk:.2f}"
    ))

    # J. Cluster risk check
    projected_cluster_risk = current_cluster_risk_pct + (effective_trade_risk / account_equity)
    require(projected_cluster_risk <= max_cluster_risk_pct, (
        f"Cluster risk {projected_cluster_risk:.3f} exceeds cap {max_cluster_risk_pct:.3f}"
    ))


@dataclass(frozen=True)
class SizingDecision:
    shares: int
    binding_constraint: str  # "RISK", "SYMBOL_NOTIONAL", "LIQUIDITY", "PORTFOLIO_EXPOSURE", "BUYING_POWER", "ZERO_SHARES", "REJECTED"
    risk_budget: float
    estimated_risk: float
    gross_notional: float
    q_risk: int
    q_notional: int
    q_liquidity: int
    q_portfolio: int
    q_broker: int
    normalized_stop: float
    stop_distance: float
    rejected: bool = False
    rejection_reason: str = ""

    @classmethod
    def reject(cls, reason: str, normalized_stop: float = 0.0) -> SizingDecision:
        return cls(
            shares=0,
            binding_constraint="REJECTED",
            risk_budget=0.0,
            estimated_risk=0.0,
            gross_notional=0.0,
            q_risk=0,
            q_notional=0,
            q_liquidity=0,
            q_portfolio=0,
            q_broker=0,
            normalized_stop=normalized_stop,
            stop_distance=0.0,
            rejected=True,
            rejection_reason=reason,
        )


def calculate_swing_stop_price(
    entry_price: float,
    atr_14: float,
    regime: str = "TRENDING_BULL",
    swing_low: Optional[float] = None,
    tick_buffer: float = 0.05,
    tick_size: float = 0.01,
) -> Tuple[float, float]:
    """
    Computes conservative ATR-based long stop price with dynamic clamp bounds.
    Never rounds stop price upward toward entry.
    Returns: (normalized_stop, normalized_stop_distance)
    """
    require(math.isfinite(entry_price) and entry_price > 0.0, f"Invalid entry price: {entry_price}")
    require(math.isfinite(atr_14) and atr_14 > 0.0, f"Invalid ATR: {atr_14}")

    multiplier = ATR_MULTIPLIER_BY_REGIME.get(regime.upper(), ATR_MULTIPLIER_BY_REGIME["DEFAULT"])
    p_atr = entry_price - (multiplier * atr_14)

    if swing_low is not None and math.isfinite(swing_low) and swing_low > 0.0:
        p_structural = swing_low - tick_buffer
        raw_stop = min(p_atr, p_structural)
    else:
        raw_stop = p_atr

    # Dynamic Clamp Bounds: [1.50 * ATR, 3.00 * ATR]
    stop_distance_raw = entry_price - raw_stop
    if stop_distance_raw < 1.50 * atr_14:
        raw_stop = entry_price - (1.50 * atr_14)
    elif stop_distance_raw > 3.00 * atr_14:
        raw_stop = entry_price - (3.00 * atr_14)

    # Conservative Floor Normalization (Never round stop up towards entry)
    normalized_stop = normalize_long_stop_price(raw_stop, price_increment=tick_size)
    require(normalized_stop <= raw_stop, "Normalized stop must not round upward")
    require(normalized_stop < entry_price, "Normalized stop must remain strictly below entry")

    normalized_distance = entry_price - normalized_stop
    return normalized_stop, normalized_distance


def calculate_swing_position_size(
    entry_price: float,
    stop_price: float,
    account_equity: float,
    available_buying_power: float,
    existing_symbol_exposure: float = 0.0,
    reserved_symbol_exposure: float = 0.0,
    existing_portfolio_exposure: float = 0.0,
    reserved_portfolio_exposure: float = 0.0,
    adv_shares_20: Optional[float] = None,
    max_participation_rate: float = 0.01,
    tick_size: float = 0.01,
    entry_slippage_per_share: float = 0.02,
    stop_slippage_per_share: float = 0.04,
    estimated_total_fees: float = 0.50,
) -> SizingDecision:
    """
    Computes 5-way clipped position size with deterministic constraint tracking:
    Q_final = min(Q_risk, Q_notional, Q_liquidity, Q_portfolio, Q_broker)
    """
    if not (math.isfinite(entry_price) and entry_price > 0.0):
        return SizingDecision.reject("INVALID_ENTRY_PRICE")
    if not (math.isfinite(stop_price) and stop_price > 0.0):
        return SizingDecision.reject("INVALID_STOP_PRICE")
    if not (math.isfinite(account_equity) and account_equity > 0.0):
        return SizingDecision.reject("INVALID_EQUITY")

    normalized_stop = normalize_long_stop_price(stop_price, price_increment=tick_size)
    normalized_stop_distance = entry_price - normalized_stop
    if normalized_stop_distance <= 0.0:
        return SizingDecision.reject("STOP_NOT_BELOW_ENTRY", normalized_stop=normalized_stop)

    # 1. Risk Budget Ceiling (0.50% equity risk, capped at $500)
    risk_budget = min(account_equity * RISK_FRACTION, CONFIGURED_MAX_RISK_DOLLARS)
    per_share_risk = normalized_stop_distance + entry_slippage_per_share + stop_slippage_per_share
    effective_risk_budget = max(0.0, (risk_budget * 0.99) - estimated_total_fees)
    q_risk = int(math.floor(effective_risk_budget / per_share_risk))

    # 2. Remaining Symbol Notional Ceiling (10% max per symbol)
    max_symbol_budget = account_equity * MAX_SYMBOL_EXPOSURE_PCT
    remaining_symbol_budget = max(0.0, max_symbol_budget - existing_symbol_exposure - reserved_symbol_exposure)
    q_notional = int(math.floor(remaining_symbol_budget / entry_price))

    # 3. Liquidity Participation Ceiling (1% ADV)
    if adv_shares_20 is not None and adv_shares_20 > 0.0:
        q_liquidity = int(math.floor(adv_shares_20 * max_participation_rate))
    else:
        q_liquidity = q_risk  # Unbounded by ADV if not provided

    # 4. Remaining Portfolio Exposure Ceiling (15% max Stage 0 gross exposure)
    max_portfolio_budget = account_equity * STAGE_0_GROSS_EXPOSURE_PCT
    remaining_portfolio_budget = max(0.0, max_portfolio_budget - existing_portfolio_exposure - reserved_portfolio_exposure)
    q_portfolio = int(math.floor(remaining_portfolio_budget / entry_price))

    # 5. Broker Buying Power Ceiling (95% safety buffer)
    safe_buying_power = max(0.0, available_buying_power * 0.95)
    q_broker = int(math.floor(safe_buying_power / entry_price))

    # 5-Way Minimum
    ceilings = [
        (q_risk, "RISK"),
        (q_notional, "SYMBOL_NOTIONAL"),
        (q_liquidity, "LIQUIDITY"),
        (q_portfolio, "PORTFOLIO_EXPOSURE"),
        (q_broker, "BUYING_POWER"),
    ]

    final_shares = min(q for q, _ in ceilings)
    if final_shares < 1:
        return SizingDecision(
            shares=0,
            binding_constraint="ZERO_SHARES",
            risk_budget=risk_budget,
            estimated_risk=0.0,
            gross_notional=0.0,
            q_risk=q_risk,
            q_notional=q_notional,
            q_liquidity=q_liquidity,
            q_portfolio=q_portfolio,
            q_broker=q_broker,
            normalized_stop=normalized_stop,
            stop_distance=normalized_stop_distance,
            rejected=True,
            rejection_reason="NO_WHOLE_SHARE_WITHIN_ALL_RISK_LIMITS",
        )

    # Identify primary binding constraint
    binding_constraint = next(name for q, name in ceilings if q == final_shares)

    gross_notional = final_shares * entry_price
    estimated_risk = (final_shares * per_share_risk) + estimated_total_fees

    return SizingDecision(
        shares=final_shares,
        binding_constraint=binding_constraint,
        risk_budget=risk_budget,
        estimated_risk=round(estimated_risk, 2),
        gross_notional=round(gross_notional, 2),
        q_risk=q_risk,
        q_notional=q_notional,
        q_liquidity=q_liquidity,
        q_portfolio=q_portfolio,
        q_broker=q_broker,
        normalized_stop=normalized_stop,
        stop_distance=round(normalized_stop_distance, 4),
        rejected=False,
        rejection_reason="",
    )


def build_canonical_swing_order_id(
    symbol: str,
    side: str,
    session_date: str,
    bar_timestamp_utc: str,
    signal_kind: str,
    account_number: str = "PA3R71I7JFDZ",
) -> str:
    """
    Build canonical deterministic client order ID for swing trades.
    Format: 'swg_' + sha256(logical_order_key)[:16]
    """
    logical_order_key = "|".join([
        "swing",
        account_number.strip().upper(),
        str(session_date).strip(),
        symbol.strip().upper(),
        side.strip().upper(),
        str(bar_timestamp_utc).strip(),
        signal_kind.strip().upper(),
    ])
    digest = hashlib.sha256(logical_order_key.encode("utf-8")).hexdigest()[:16]
    return f"swg_{digest}"

