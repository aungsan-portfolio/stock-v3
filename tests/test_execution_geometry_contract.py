import pytest
import math
from unittest.mock import Mock

import config
from strategies.base import TradeSignal
from strategies.constants import StrategyName
from strategies.execution_geometry import (
    calculate_minimum_stop_distance,
    calculate_directional_stop_and_validate,
)
from strategies.order_manager import (
    compute_effective_noise_floor,
    _pre_submission_veto,
    validate_quote_health,
    unfavorable_chase,
)


def test_shared_geometry_contract_atr_084_spread_059():
    """
    Verify exact mathematical parity between Strategy Generation (buffered)
    and Order Manager Veto (baseline unbuffered) for COST parameters:
    ATR = 0.84, Spread = 0.59, Multiplier = 0.75, Absolute Min = 0.08, Tick = 0.01.
    """
    atr = 0.84
    spread = 0.59
    multiplier = 0.75

    # 1. Baseline unbuffered geometry (Order Manager check):
    # base = max(0.84 * 0.75, 0.59, 0.01 * 2, 0.08) = max(0.63, 0.59, 0.02, 0.08) = 0.6300
    expected_baseline = round(0.84 * 0.75, 4)
    assert expected_baseline == 0.6300

    baseline_dist = calculate_minimum_stop_distance(
        atr=atr,
        spread_floor=spread,
        tick_size=0.01,
        use_buffer=False,
    )
    assert math.isclose(baseline_dist, 0.6300, abs_tol=1e-4)

    # 2. Buffered geometry (Strategy Generation check, 1.05 safety buffer):
    # 0.6300 * 1.05 = 0.6615
    buffered_dist = calculate_minimum_stop_distance(
        atr=atr,
        spread_floor=spread,
        tick_size=0.01,
        use_buffer=True,
    )
    assert math.isclose(buffered_dist, 0.6615, abs_tol=1e-4)

    # 3. Verify directional stop calculation in strategy generator uses buffered_dist
    entry_price = 991.58
    geom = calculate_directional_stop_and_validate(
        side="long",
        price=entry_price,
        raw_risk=0.50 * atr,
        atr=atr,
        spread_floor=spread,
        tick_size=0.01,
        precision=2,
    )
    assert math.isclose(geom.minimum_distance, 0.6615, abs_tol=1e-4)
    # Stop price must be placed at least 0.6615 below entry
    assert geom.stop_price <= (entry_price - 0.6615 + 1e-4)


def test_cost_normal_spread_passes_pre_submission_veto():
    """
    With normal spread (0.59), stop distance 0.6700 > baseline noise floor 0.6300,
    so Order Manager's pre-submission veto MUST NOT veto the signal.
    """
    quote_normal = {"bid": 991.59, "ask": 992.18, "timestamp": 1792524337.269791}
    sig = TradeSignal(
        symbol="COST",
        strategy=StrategyName.MEAN_REVERSION,
        side="BUY",
        confidence=0.65,
        entry_price=991.58,
        stop_price=990.91,  # stop distance = 991.58 - 990.91 = 0.6700
        target_price=994.57,
        atr=0.84,
        risk_per_share=0.67,
        reason="Mean reversion test",
    )

    floor_val, comp = compute_effective_noise_floor(sig, quote=quote_normal, return_components=True)
    assert math.isclose(floor_val, 0.6300, abs_tol=1e-4)
    assert math.isclose(comp["atr_component"], 0.6300, abs_tol=1e-4)
    assert math.isclose(comp["spread_component"], 0.5900, abs_tol=1e-4)

    should_skip, reason = _pre_submission_veto(sig, quote=quote_normal)
    assert not should_skip, f"Signal should not be vetoed when stop_dist (0.67) > floor (0.63): {reason}"


def test_cost_wide_iex_spread_elevates_floor_and_protects_capital():
    """
    When live quote spread widens to $8.4100 on IEX (bid=987.59, ask=996.00),
    effective noise floor becomes $8.4100, and pre-submission veto MUST veto
    a stop distance of 0.6700 to prevent crossing a spread 12.5x wider than stop.
    """
    quote_wide = {"bid": 987.59, "ask": 996.00, "timestamp": 1792524339.8068}
    sig = TradeSignal(
        symbol="COST",
        strategy=StrategyName.MEAN_REVERSION,
        side="BUY",
        confidence=0.65,
        entry_price=991.58,
        stop_price=990.91,  # stop distance = 0.6700
        target_price=994.57,
        atr=0.84,
        risk_per_share=0.67,
        reason="Mean reversion test",
    )

    floor_val, comp = compute_effective_noise_floor(sig, quote=quote_wide, return_components=True)
    assert math.isclose(floor_val, 8.4100, abs_tol=1e-4)
    assert math.isclose(comp["atr_component"], 0.6300, abs_tol=1e-4)
    assert math.isclose(comp["spread_component"], 8.4100, abs_tol=1e-4)

    should_skip, reason = _pre_submission_veto(sig, quote=quote_wide)
    assert should_skip
    assert "REJECTED_NOISE_FLOOR_BREACH" in reason
    assert "stop distance $0.6700 < effective noise floor $8.4100" in reason
    assert "spread_component=$8.4100" in reason
    assert "atr_component=$0.6300" in reason


def test_cost_price_discontinuity_and_chase_protection():
    """
    Verify that if a quote arrives with a large price gap (e.g. quote mid = $914.205 vs signal entry $991.58),
    validate_quote_health rejects it fail-closed as QUOTE_PRICE_DISCONTINUITY.
    Also verify that if executable entry chases unfavorably > $0.15, unfavorable_chase guards trigger.
    """
    signal_entry = 991.58

    # 1. Price Discontinuity check:
    discontinuous_quote = {
        "bid": 910.00,
        "ask": 918.41,
        "timestamp": 1792524339.8068,
    }
    ok, err, metrics = validate_quote_health(
        discontinuous_quote,
        signal_entry_price=signal_entry,
        now_ts=1792524340.0,
    )
    assert not ok
    assert "QUOTE_PRICE_DISCONTINUITY" in err
    assert "deviates" in err

    # 2. Unfavorable Chase check:
    # If ask jumped by $0.92 above entry (from 991.58 to 992.50)
    chase = unfavorable_chase("BUY", signal_entry=991.58, executable_entry=992.50)
    assert math.isclose(chase, 0.92, abs_tol=1e-4)
    assert chase > 0.15  # Breaches MAX_ALLOWED_CHASE = 0.15
