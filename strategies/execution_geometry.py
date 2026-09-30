"""
strategies/execution_geometry.py -- Shared Stop and Execution Geometry Module.

Canonical single source of truth for stop loss calculation, noise floor evaluation,
and tick-size grid quantization across Strategy signal generation and Order Manager
risk verification.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING

import config


@dataclass(frozen=True)
class StopGeometry:
    """Immutable execution geometry contract for stop loss placement."""
    stop_price: float
    minimum_distance: float
    final_distance: float


def floor_to_tick(value: Decimal, tick_size: Decimal) -> Decimal:
    """
    Quantize Decimal value down to nearest tick grid.
    Directionally protective for long stops / short profit targets.
    """
    if tick_size <= Decimal("0"):
        raise ValueError(f"tick_size must be positive Decimal, got {tick_size}")
    steps = (value / tick_size).to_integral_value(rounding=ROUND_FLOOR)
    return steps * tick_size


def ceil_to_tick(value: Decimal, tick_size: Decimal) -> Decimal:
    """
    Quantize Decimal value up to nearest tick grid.
    Directionally protective for short stops / long profit targets.
    """
    if tick_size <= Decimal("0"):
        raise ValueError(f"tick_size must be positive Decimal, got {tick_size}")
    steps = (value / tick_size).to_integral_value(rounding=ROUND_CEILING)
    return steps * tick_size


def floor_to_tick_float(value: float, tick_size: float) -> float:
    """Float wrapper for floor_to_tick."""
    d_val = Decimal(str(value))
    d_tick = Decimal(str(tick_size))
    return float(floor_to_tick(d_val, d_tick))


def ceil_to_tick_float(value: float, tick_size: float) -> float:
    """Float wrapper for ceil_to_tick."""
    d_val = Decimal(str(value))
    d_tick = Decimal(str(tick_size))
    return float(ceil_to_tick(d_val, d_tick))


def _minimum_stop_distance_decimal(atr, spread_floor, tick_size, use_buffer, cfg=None):
    """One Decimal implementation shared by both public geometry functions."""
    cfg = cfg if cfg is not None else config
    multiplier = float(getattr(cfg, "NOISE_FLOOR_ATR_MULTIPLIER", 0.75))
    absolute_min = float(getattr(cfg, "NOISE_FLOOR_ABS_MIN", 0.08))
    if not (math.isfinite(multiplier) and multiplier > 0):
        raise ValueError("NOISE_FLOOR_ATR_MULTIPLIER must be finite and positive")
    if not (math.isfinite(absolute_min) and absolute_min >= 0):
        raise ValueError("NOISE_FLOOR_ABS_MIN must be finite and nonnegative")
    base = max(
        Decimal(str(atr)) * Decimal(str(multiplier)),
        Decimal(str(spread_floor)),
        Decimal(str(tick_size)) * Decimal("2"),
        Decimal(str(absolute_min)),
    )
    return base * Decimal("1.05") if use_buffer else base


def calculate_minimum_stop_distance(
    *,
    atr: float,
    spread_floor: float = 0.0,
    tick_size: float = 0.01,
    use_buffer: bool = True,
    cfg=None,
) -> float:
    """
    Calculates authoritative minimum stop distance based on ATR, spread, two ticks, and configured absolute floor.

    Args:
        atr: Average True Range (> 0, finite).
        spread_floor: Minimum spread allowance (>= 0, finite).
        tick_size: Minimum price increment (> 0, finite).
        use_buffer: If True, applies 1.05 safety buffer (used by Strategy generation).
                    If False, computes baseline noise floor (used by Order Manager veto check).
        cfg: Optional configuration object; defaults to the canonical config module.

    Returns:
        float: Minimum stop distance as float.
    """
    if not (math.isfinite(atr) and atr > 0):
        raise ValueError(f"atr must be finite and > 0, got {atr}")
    if not (math.isfinite(spread_floor) and spread_floor >= 0):
        raise ValueError(f"spread_floor must be finite and >= 0, got {spread_floor}")
    if not (math.isfinite(tick_size) and tick_size > 0):
        raise ValueError(f"tick_size must be finite and > 0, got {tick_size}")

    return float(_minimum_stop_distance_decimal(
        atr, spread_floor, tick_size, use_buffer, cfg=cfg
    ))


def calculate_directional_stop_and_validate(
    *,
    side: str,
    price: float,
    raw_risk: float,
    atr: float,
    spread_floor: float = 0.0,
    tick_size: float = 0.01,
    precision: int = 2,
    cfg=None,
) -> StopGeometry:
    """
    Computes directional stop price and enforces strict geometry invariants.
    Maintains Decimal truth throughout calculations until output boundary.

    Args:
        side: Position direction: strictly "long" or "short".
        price: Entry price (> 0, finite).
        raw_risk: Raw structural risk distance (> 0, finite).
        atr: Average True Range (> 0, finite).
        spread_floor: Minimum spread floor (>= 0, finite).
        tick_size: Tick size (> 0, finite).
        precision: Decimal display precision.
        cfg: Optional configuration object; defaults to the canonical config module.

    Returns:
        StopGeometry: Immutable tuple of (stop_price, minimum_distance, final_distance).

    Raises:
        ValueError: If inputs are invalid or post-quantization invariants fail.
    """
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"long", "short"}:
        raise ValueError(f"Unsupported position direction: {side!r}. Must be 'long' or 'short'.")

    if not (math.isfinite(price) and price > 0):
        raise ValueError(f"price must be finite and > 0, got {price}")
    if not (math.isfinite(raw_risk) and raw_risk > 0):
        raise ValueError(f"raw_risk must be finite and > 0, got {raw_risk}")
    if not (math.isfinite(atr) and atr > 0):
        raise ValueError(f"atr must be finite and > 0, got {atr}")
    if not (math.isfinite(spread_floor) and spread_floor >= 0):
        raise ValueError(f"spread_floor must be finite and >= 0, got {spread_floor}")
    if not (math.isfinite(tick_size) and tick_size > 0):
        raise ValueError(f"tick_size must be finite and > 0, got {tick_size}")

    tick_dec = Decimal(str(tick_size))
    if tick_dec <= Decimal("0"):
        raise ValueError(f"tick_size must be positive, got {tick_size}")

    expected_precision = max(0, -tick_dec.as_tuple().exponent)
    if precision < expected_precision:
        raise ValueError(
            f"precision {precision} cannot represent the configured tick size exponent {expected_precision}"
        )

    price_dec = Decimal(str(price))
    raw_risk_dec = Decimal(str(raw_risk))
    minimum_distance_dec = _minimum_stop_distance_decimal(
        atr, spread_floor, tick_size, True, cfg=cfg
    )

    desired_distance_dec = max(raw_risk_dec, minimum_distance_dec)

    if normalized_side == "long":
        unquantized_stop_dec = price_dec - desired_distance_dec
        stop_dec = floor_to_tick(unquantized_stop_dec, tick_dec)
        actual_distance_dec = price_dec - stop_dec
    else:
        unquantized_stop_dec = price_dec + desired_distance_dec
        stop_dec = ceil_to_tick(unquantized_stop_dec, tick_dec)
        actual_distance_dec = stop_dec - price_dec

    if stop_dec <= 0:
        raise ValueError("Directional stop must be positive")

    if actual_distance_dec < minimum_distance_dec:
        raise ValueError(
            f"Rounded stop breached minimum noise floor: "
            f"actual={actual_distance_dec} < min={minimum_distance_dec}"
        )

    return StopGeometry(
        stop_price=float(stop_dec),
        minimum_distance=float(minimum_distance_dec),
        final_distance=float(actual_distance_dec),
    )
