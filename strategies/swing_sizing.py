"""
strategies/swing_sizing.py — Production Sizing Validator & Geometry Engine

Implements fail-closed quantitative sizing algorithms, whole-share lot policies,
asset-specific tick normalizations, composite idempotency guards, and zero-tolerance
portfolio risk enforcement for swing trading.
"""

from __future__ import annotations

import hashlib
import math
import warnings
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Set, Tuple


class PositionIntent(str, Enum):
    """
    Position intent classification.
    Stage 0 production broker dispatch strictly supports OPEN_LONG only (single position per symbol).
    ADD_LONG is preserved at the validator layer exclusively for Stage 1 multi-tier entry research.
    """
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
RISK_FRACTION: float = 0.005  # 0.50% equity risk budget
CONFIGURED_MAX_RISK_DOLLARS: float = 500.00  # Hard dollar ceiling at $100k equity (Hard enforced)
TARGET_DEPLOYED_EXPOSURE_PCT: float = 0.20  # 20% target deployment (Soft guideline)
STAGE_0_GROSS_EXPOSURE_PCT: float = 0.28   # 28% hard portfolio exposure ceiling (Hard enforced)
MAX_SYMBOL_EXPOSURE_PCT: float = 0.10      # 10% max per symbol (Hard enforced)
try:
    import config
    SECTOR_EXPOSURE_CAP_PCT: float = float(getattr(config, "CLUSTER_NOTIONAL_CAP", 0.15))
except Exception:
    SECTOR_EXPOSURE_CAP_PCT: float = 0.15      # 15% max gross notional exposure per sector (C7)
MAX_CLUSTER_RISK_PCT: float = 0.0125       # 1.25% max cumulative open stop risk per correlated cluster (Hard enforced)
AGGREGATE_OPEN_STOP_HEAT_PCT: float = 0.025 # 2.50% max total open risk (Hard enforced)
from strategies.rotation_config import MAX_CONCURRENT_POSITIONS  # Canonical Single Source of Truth
MAX_DAILY_LOSS_DOLLARS: float = 300.00     # Bridge circuit breaker ceiling (Hard enforced)
MAX_ADV_PARTICIPATION_RATE: float = 0.01   # 1.00% ADV liquidity ceiling (Hard enforced)
MIN_ORDER_NOTIONAL: float = 500.00         # $500 minimum order notional (Hard enforced)

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


def require_strict_int(value: Any, name: str, min_value: int = 0) -> None:
    """Validate that value is strictly an int (rejecting bool) and >= min_value."""
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= min_value,
        f"{name} must be an integer >= {min_value}, got {value!r}"
    )


def require_finite_nonnegative(value: float, name: str) -> None:
    """Validate that value is a finite float/int and >= 0.0."""
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0.0,
        f"{name} must be finite and non-negative, got {value}"
    )


def require_finite_positive(value: float, name: str) -> None:
    """Validate that value is a finite float/int and > 0.0."""
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0.0,
        f"{name} must be finite and positive, got {value}"
    )


def get_trade_risk_budget(
    account_equity: float,
    risk_fraction: float = RISK_FRACTION,
) -> float:
    """
    Canonical single source of truth for trade dollar risk budget.
    Caps dollar risk at min(account_equity * risk_fraction, CONFIGURED_MAX_RISK_DOLLARS).
    """
    require_finite_positive(account_equity, "account_equity")
    require(
        isinstance(risk_fraction, (int, float)) and not isinstance(risk_fraction, bool)
        and math.isfinite(risk_fraction) and 0.0 < risk_fraction <= RISK_FRACTION,
        f"Invalid risk fraction: {risk_fraction}"
    )
    return min(account_equity * risk_fraction, CONFIGURED_MAX_RISK_DOLLARS)


def normalize_long_stop_price(price: float, price_increment: float = 0.01) -> float:
    """
    Round price down conservatively to the valid asset price increment.
    Uses decimal floor arithmetic to avoid IEEE-754 floating point inaccuracies.
    """
    require(isinstance(price, (int, float)) and not isinstance(price, bool) and math.isfinite(price) and price > 0.0, f"Invalid price for tick normalization: {price}")
    require(isinstance(price_increment, (int, float)) and not isinstance(price_increment, bool) and math.isfinite(price_increment) and price_increment > 0.0, (
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
    require_strict_int(initial_shares, "initial_shares", min_value=1)
    require(
        isinstance(target_weight, (int, float)) and not isinstance(target_weight, bool)
        and math.isfinite(target_weight) and 0.0 < target_weight < 1.0,
        f"target_weight must be between 0 and 1, got {target_weight}"
    )

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
    proposed_risk_pct: float = RISK_FRACTION,
    symbol_notional_cap: float = 12000.0,
    adv_shares_20: Optional[float] = None,
    max_participation_rate: float = MAX_ADV_PARTICIPATION_RATE,
    price_increment: float = 0.01,
    entry_slippage_per_share: float = 0.02,
    stop_slippage_per_share: float = 0.04,
    estimated_total_fees: float = 0.50,
) -> int:
    """
    [DEPRECATED] Derive position share count strictly from account risk budget.
    Use calculate_swing_position_size() instead.
    """
    warnings.warn(
        "calculate_swing_shares() is deprecated; use calculate_swing_position_size()",
        DeprecationWarning,
        stacklevel=2,
    )
    require_finite_positive(entry_price, "entry_price")
    require_finite_positive(stop_price, "stop_price")
    require_finite_positive(account_equity, "account_equity")

    normalized_stop = normalize_long_stop_price(stop_price, price_increment=price_increment)
    base_risk_per_share = entry_price - normalized_stop
    require(base_risk_per_share > 0.0, f"Entry {entry_price} must exceed normalized stop {normalized_stop}")

    # Effective risk per share including slippage
    effective_risk_per_share = base_risk_per_share + entry_slippage_per_share + stop_slippage_per_share

    # Canonical risk budget capped at CONFIGURED_MAX_RISK_DOLLARS
    hard_dollar_risk_limit = get_trade_risk_budget(account_equity, proposed_risk_pct)
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
    proposed_risk_per_trade_pct: float = RISK_FRACTION,
    max_portfolio_heat_pct: float = AGGREGATE_OPEN_STOP_HEAT_PCT,
    current_cluster_risk_pct: float = 0.0,
    max_cluster_risk_pct: float = 0.0125,
    adv_shares_20: float = 1_000_000.0,
    max_participation_rate: float = MAX_ADV_PARTICIPATION_RATE,
    symbol_notional_cap: float = 12000.0,
    price_increment: float = 0.01,
    quote_age_seconds: float = 5.0,
    max_quote_age_seconds: float = 60.0,
    entry_slippage_per_share: float = 0.02,
    stop_slippage_per_share: float = 0.04,
    estimated_total_fees: float = 0.50,
    *,
    account_number: str = "PA3R71I7JFDZ",
    broker_account_number: str = "PA3R71I7JFDZ",
    current_position_count: int = 0,
    pending_open_position_count: int = 0,
    existing_position_symbols: Optional[Set[str]] = None,
    intent: PositionIntent = PositionIntent.OPEN_LONG,
    max_concurrent_positions: int = MAX_CONCURRENT_POSITIONS,
    symbol_sector: str = "INFORMATION_TECHNOLOGY",
    existing_sector_exposure_by_sector: Optional[Mapping[str, Any]] = None,
    pending_sector_exposure_by_sector: Optional[Mapping[str, Any]] = None,
    max_sector_exposure_pct: float = SECTOR_EXPOSURE_CAP_PCT,
) -> None:
    """
    Production fail-closed validation for swing orders.
    Enforces exact risk ceilings (zero tolerance overshoot), composite idempotency keys,
    whole-share integrity, state attribution, and portfolio heat limits before any broker dispatch.
    """
    # Strict type and non-negativity checks
    require_strict_int(shares, "shares", min_value=1)
    require_strict_int(current_position_count, "current_position_count", min_value=0)
    require_strict_int(pending_open_position_count, "pending_open_position_count", min_value=0)
    require_strict_int(max_concurrent_positions, "max_concurrent_positions", min_value=1)

    require_finite_positive(entry_price, "entry_price")
    require_finite_positive(stop_price, "stop_price")
    require_finite_positive(account_equity, "account_equity")
    require_finite_nonnegative(available_buying_power, "available_buying_power")
    require_finite_nonnegative(current_open_stop_risk, "current_open_stop_risk")
    require_finite_nonnegative(pending_entry_risk, "pending_entry_risk")
    require_finite_nonnegative(current_cluster_risk_pct, "current_cluster_risk_pct")
    require_finite_nonnegative(quote_age_seconds, "quote_age_seconds")
    require_finite_nonnegative(entry_slippage_per_share, "entry_slippage_per_share")
    require_finite_nonnegative(stop_slippage_per_share, "stop_slippage_per_share")
    require_finite_nonnegative(estimated_total_fees, "estimated_total_fees")
    require_finite_positive(adv_shares_20, "adv_shares_20")

    # Account consistency
    require(isinstance(account_number, str) and bool(account_number.strip()), "account_number must be provided")
    require(isinstance(broker_account_number, str) and bool(broker_account_number.strip()), "broker_account_number must be provided")
    require(
        account_number.strip().upper() == broker_account_number.strip().upper(),
        f"ACCOUNT_STATE_MISMATCH: configured={account_number} vs broker={broker_account_number}"
    )

    # Side normalization (Stage 0 swing validator is strictly long-only BUY)
    norm_side = str(side).upper().strip()
    require(norm_side == "BUY", f"Stage 0 swing validator only supports BUY orders, got {norm_side}")

    # Symbol normalization
    require(isinstance(symbol, str) and bool(symbol.strip()), "symbol must be provided")
    norm_sym = symbol.upper().strip()

    # State & Intent consistency
    if existing_position_symbols is None:
        existing_position_symbols = set()
    require(isinstance(existing_position_symbols, set), "existing_position_symbols must be a set")
    norm_pos_symbols = {s.strip().upper() for s in existing_position_symbols if isinstance(s, str) and s.strip()}
    require(
        current_position_count == len(norm_pos_symbols),
        f"POSITION_STATE_INCONSISTENT: count={current_position_count}, symbols={len(norm_pos_symbols)}"
    )

    if isinstance(intent, PositionIntent):
        norm_intent = intent
    else:
        try:
            norm_intent = PositionIntent(str(intent))
        except Exception:
            raise UnsupportedSwingDirection(f"Unsupported swing intent: {intent}")

    symbol_exists = norm_sym in norm_pos_symbols
    if norm_intent == PositionIntent.OPEN_LONG:
        require(not symbol_exists, f"OPEN_LONG rejected because {norm_sym} already exists in positions")
        projected_count = current_position_count + pending_open_position_count + 1
        require(
            projected_count <= max_concurrent_positions,
            f"Projected position count {projected_count} exceeds capacity {max_concurrent_positions}"
        )
    elif norm_intent == PositionIntent.ADD_LONG:
        require(symbol_exists, f"ADD_LONG rejected because {norm_sym} does not exist in positions")
    else:
        raise UnsupportedSwingDirection(f"Unsupported swing intent: {norm_intent}")

    # Risk parameter ranges
    require(0.0 < proposed_risk_per_trade_pct <= RISK_FRACTION, (
        f"Risk-per-trade exceeds research specification: {proposed_risk_per_trade_pct}"
    ))
    require(0.0 < max_portfolio_heat_pct <= AGGREGATE_OPEN_STOP_HEAT_PCT, (
        f"Portfolio heat cap exceeds research specification: {max_portfolio_heat_pct}"
    ))
    require(0.0 < max_participation_rate <= MAX_ADV_PARTICIPATION_RATE, (
        f"Invalid participation rate: {max_participation_rate}"
    ))

    # Order deduplication & composite idempotency
    require(bool(client_order_id), "client_order_id must be provided")
    require(bool(strategy_instance_id), "strategy_instance_id must be provided")
    require(bool(signal_version), "signal_version must be provided")

    intent_id = entry_intent_id or client_order_id
    idempotency_key = (strategy_instance_id, norm_sym, norm_side, signal_version, intent_id)
    if active_entry_intents is not None:
        require(idempotency_key not in active_entry_intents, (
            f"Duplicate entry intent rejected for composite key: {idempotency_key}"
        ))

    if existing_working_symbols is not None and norm_side == "BUY":
        require(norm_sym not in existing_working_symbols, (
            f"Duplicate active working order rejected for symbol {norm_sym}"
        ))

    # Price and stop geometry
    normalized_stop = normalize_long_stop_price(stop_price, price_increment=price_increment)
    require(entry_price > normalized_stop, f"Long entry {entry_price} must exceed stop {normalized_stop}")

    # ADV and quote freshness
    require(quote_age_seconds <= max_quote_age_seconds, (
        f"Stale quote: {quote_age_seconds}s > {max_quote_age_seconds}s"
    ))

    # Notional limits
    notional_value = shares * entry_price
    require(notional_value >= MIN_ORDER_NOTIONAL, f"Order notional ${notional_value:.2f} below ${MIN_ORDER_NOTIONAL:.0f} minimum")
    require(notional_value <= symbol_notional_cap, (
        f"Order notional ${notional_value:.2f} exceeds cap ${symbol_notional_cap:.2f}"
    ))
    require(notional_value <= available_buying_power, (
        f"Notional ${notional_value:.2f} exceeds buying power ${available_buying_power:.2f}"
    ))

    # Liquidity participation cap
    max_shares_liquidity = int(math.floor(adv_shares_20 * max_participation_rate))
    require(shares <= max_shares_liquidity, (
        f"Shares {shares} exceeds ADV liquidity cap {max_shares_liquidity}"
    ))

    # Sector notional exposure validation (Aggregating existing, pending, and proposed)
    require(
        isinstance(symbol_sector, str) and bool(symbol_sector.strip()),
        f"SECTOR_ATTRIBUTION_REQUIRED: {norm_sym}"
    )
    norm_sec = symbol_sector.strip().upper()
    sec_exist_map = existing_sector_exposure_by_sector if existing_sector_exposure_by_sector is not None else {}
    sec_pend_map = pending_sector_exposure_by_sector if pending_sector_exposure_by_sector is not None else {}
    d_exist_sec = Decimal(str(sec_exist_map.get(norm_sec, "0.0") or "0.0"))
    d_pend_sec = Decimal(str(sec_pend_map.get(norm_sec, "0.0") or "0.0"))
    d_notional = Decimal(str(notional_value))
    d_projected_sec = d_exist_sec + d_pend_sec + d_notional
    d_max_sec = Decimal(str(account_equity)) * Decimal(str(max_sector_exposure_pct))
    require(
        d_projected_sec <= d_max_sec,
        f"SECTOR_EXPOSURE_LIMIT_EXCEEDED: sector={norm_sec} projected=${float(d_projected_sec):.2f} exceeds cap=${float(d_max_sec):.2f}"
    )

    # Effective trade risk (per-share slippage model, ZERO tolerance overshoot)
    # Uses exact Decimal arithmetic to prevent float epsilon inaccuracies on boundary conditions ($500.00)
    d_entry = Decimal(str(entry_price))
    d_stop = Decimal(str(normalized_stop))
    d_slip = Decimal(str(entry_slippage_per_share)) + Decimal(str(stop_slippage_per_share))
    d_fees = Decimal(str(estimated_total_fees))
    d_shares = Decimal(str(shares))
    d_effective_risk = (d_shares * (d_entry - d_stop + d_slip)) + d_fees

    hard_budget_float = get_trade_risk_budget(account_equity, proposed_risk_per_trade_pct)
    d_hard_budget = Decimal(str(hard_budget_float))
    require(
        d_effective_risk <= d_hard_budget,
        f"Effective trade risk ${float(d_effective_risk):.2f} strictly exceeds hard budget ${float(d_hard_budget):.2f}"
    )

    # Projected portfolio heat (including working entries, excluding unfilled exits)
    d_open_stop = Decimal(str(current_open_stop_risk))
    d_pending = Decimal(str(pending_entry_risk))
    d_projected_heat = d_open_stop + d_pending + d_effective_risk
    d_max_portfolio_heat = Decimal(str(account_equity)) * Decimal(str(max_portfolio_heat_pct))
    require(
        d_projected_heat <= d_max_portfolio_heat,
        f"Projected portfolio heat ${float(d_projected_heat):.2f} exceeds limit ${float(d_max_portfolio_heat):.2f}"
    )

    # Cluster risk check (cumulative open stop risk across correlated cluster assets)
    projected_cluster_risk = current_cluster_risk_pct + (float(d_effective_risk) / account_equity)
    require(
        projected_cluster_risk <= max_cluster_risk_pct,
        f"Cluster risk {projected_cluster_risk:.4f} exceeds cap {max_cluster_risk_pct:.4f}"
    )


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
    portfolio_exposure_used: float = 0.0
    portfolio_exposure_cap: float = 0.0

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
    max_participation_rate: float = MAX_ADV_PARTICIPATION_RATE,
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

    # 1. Risk Budget Ceiling (0.50% equity risk, capped at $500 via canonical helper)
    risk_budget = get_trade_risk_budget(account_equity, RISK_FRACTION)
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

    candidate_notional = final_shares * entry_price
    if candidate_notional < MIN_ORDER_NOTIONAL:
        return SizingDecision(
            shares=0,
            binding_constraint="MINIMUM_NOTIONAL",
            risk_budget=risk_budget,
            estimated_risk=0.0,
            gross_notional=0.0,
            q_risk=q_risk,
            q_notional=q_notional,
            q_liquidity=q_liquidity,
            q_portfolio=q_portfolio,
            q_broker=q_broker,
            normalized_stop=normalized_stop,
            stop_distance=round(normalized_stop_distance, 4),
            rejected=True,
            rejection_reason=(
                f"MIN_ORDER_NOTIONAL_UNSATISFIABLE: Maximum permitted order notional "
                f"${candidate_notional:.2f} ({final_shares} shares @ ${entry_price:.2f}) "
                f"is below the ${MIN_ORDER_NOTIONAL:.2f} order minimum"
            ),
        )

    # Identify primary binding constraint
    binding_constraint = next(name for q, name in ceilings if q == final_shares)

    gross_notional = final_shares * entry_price
    estimated_risk = (final_shares * per_share_risk) + estimated_total_fees
    port_used = existing_portfolio_exposure + reserved_portfolio_exposure

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
        portfolio_exposure_used=round(port_used, 2),
        portfolio_exposure_cap=round(max_portfolio_budget, 2),
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
    require(isinstance(account_number, str) and bool(account_number.strip()), "account_number must be non-empty")
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

