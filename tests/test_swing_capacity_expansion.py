"""
tests/test_swing_capacity_expansion.py — Verification of Swing Capacity Expansion (6 -> 10 Positions, 28% Gross Exposure)

Verifies:
1. Canonical Single Source of Truth:
   - MAX_CONCURRENT_POSITIONS == 10 in both strategies.rotation_config and strategies.swing_sizing.
   - STAGE_0_GROSS_EXPOSURE_PCT == 0.28 in strategies.swing_sizing.
2. Capacity Boundary Invariants:
   - 9 open positions + 1 new entry = 10 projected -> ALLOWED.
   - 10 open positions + 1 new entry = 11 projected -> REJECTED (exceeds capacity 10).
3. Gross Portfolio Exposure 28% Ceiling:
   - Replay of BAC live scenario: with 6 existing positions ($15,150.77 market value) and $100,290.84 equity,
     calculate_swing_position_size() yields valid whole shares without ZERO_SHARES rejection.
   - When existing exposure reaches $28,000 (100% of 28% budget), sizing rejects with ZERO_SHARES.
4. Risk Invariants Strictly Preserved:
   - Per-trade risk budget remains 0.50% (capped at $500).
   - Aggregate open stop heat cap remains 2.50%.
   - Sector cap remains 25%.
   - Cluster risk cap remains 1.25%.
"""

import math
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from strategies.rotation_config import MAX_CONCURRENT_POSITIONS as ROT_MAX_POS
from strategies.swing_sizing import (
    MAX_CONCURRENT_POSITIONS as SIZING_MAX_POS,
    STAGE_0_GROSS_EXPOSURE_PCT,
    RISK_FRACTION,
    AGGREGATE_OPEN_STOP_HEAT_PCT,
    SECTOR_EXPOSURE_CAP_PCT,
    MAX_CLUSTER_RISK_PCT,
    calculate_swing_position_size,
    validate_swing_order_safety,
    PositionIntent,
    SwingSizingRejected,
)
from alpaca_bridge import AlpacaBridge
from predictor import Signal


def test_single_source_of_truth_and_invariants():
    """Verify single source of truth and frozen risk constants."""
    assert ROT_MAX_POS == 10, "Canonical MAX_CONCURRENT_POSITIONS in rotation_config must be 10"
    assert SIZING_MAX_POS == 10, "Imported MAX_CONCURRENT_POSITIONS in swing_sizing must be 10"
    assert ROT_MAX_POS is SIZING_MAX_POS, "Must reference the exact same object (single source of truth)"
    assert STAGE_0_GROSS_EXPOSURE_PCT == 0.28, "STAGE_0_GROSS_EXPOSURE_PCT must be 0.28 (28%)"

    # Strict fail-closed risk invariants unchanged
    assert RISK_FRACTION == 0.005, "Per-trade risk fraction must remain 0.50%"
    assert AGGREGATE_OPEN_STOP_HEAT_PCT == 0.025, "Portfolio heat cap must remain 2.50%"
    assert SECTOR_EXPOSURE_CAP_PCT == 0.25, "Sector cap must remain 25%"
    assert MAX_CLUSTER_RISK_PCT == 0.0125, "Cluster risk cap must remain 1.25%"


def test_capacity_10_positions_boundary_simulation():
    """
    Capacity Boundary:
    - 9 existing + 0 pending + 1 new = 10 -> ALLOWED
    - 10 existing + 0 pending + 1 new = 11 -> REJECTED
    - 9 existing + 1 pending + 1 new = 11 -> REJECTED
    """
    base_args = dict(
        client_order_id="swg_test_cap_001",
        strategy_instance_id="SWING_BOT_PRO",
        symbol="BAC",
        signal_version="v1.6-alpha",
        entry_price=58.00,
        stop_price=56.00,
        shares=10,
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        current_open_stop_risk=500.0,
        pending_entry_risk=0.0,
        current_cluster_risk_pct=0.005,
        adv_shares_20=1_000_000.0,
        symbol_sector="FINANCIALS",
    )

    syms_9 = {f"SYM_{i}" for i in range(1, 10)}
    syms_10 = {f"SYM_{i}" for i in range(1, 11)}

    # 1. 9 positions open -> 10th allowed
    validate_swing_order_safety(
        **base_args,
        current_position_count=9,
        pending_open_position_count=0,
        existing_position_symbols=syms_9,
        intent=PositionIntent.OPEN_LONG,
    )

    # 2. 10 positions open -> 11th rejected
    with pytest.raises(SwingSizingRejected, match="Projected position count 11 exceeds capacity 10"):
        validate_swing_order_safety(
            **base_args,
            current_position_count=10,
            pending_open_position_count=0,
            existing_position_symbols=syms_10,
            intent=PositionIntent.OPEN_LONG,
        )

    # 3. 9 positions open + 1 pending buy -> 11th rejected
    with pytest.raises(SwingSizingRejected, match="Projected position count 11 exceeds capacity 10"):
        validate_swing_order_safety(
            **base_args,
            current_position_count=9,
            pending_open_position_count=1,
            existing_position_symbols=syms_9,
            intent=PositionIntent.OPEN_LONG,
        )


def test_replay_bac_scenario_under_28_percent_cap():
    """
    Empirical Replay of Cycle 1 BAC Sizing:
    Under 15% cap ($15,043.52 budget), existing exposure $15,150.77 caused ZERO_SHARES rejection.
    Under 28% cap ($28,081.44 budget), remaining budget is ~$12,930.67, allowing BAC entry!
    """
    equity = 100_290.84
    existing_exposure = 15_150.77  # 6 open positions: AMZN, AVGO, COST, JPM, MSFT, V
    bac_entry_px = 58.06
    bac_stop_px = 56.40  # 2% ATR stop

    res = calculate_swing_position_size(
        entry_price=bac_entry_px,
        stop_price=bac_stop_px,
        account_equity=equity,
        available_buying_power=85_000.0,
        existing_portfolio_exposure=existing_exposure,
        adv_shares_20=20_000_000.0,
    )

    assert not res.rejected, f"BAC must NOT be rejected under 28% cap: {res.rejection_reason}"
    assert res.shares > 0, "BAC must receive non-zero shares"
    assert res.gross_notional > 0.0
    assert res.binding_constraint in ("RISK", "SYMBOL_NOTIONAL", "PORTFOLIO_EXPOSURE")
    # Gross notional should be around $2,500 - $10,000
    assert 500.0 <= res.gross_notional <= 10_000.0


def test_exposure_28_percent_ceiling_strict_rejection_at_full_capacity():
    """Verify that when portfolio exposure reaches 28% ($28,000 on $100k), sizing rejects with ZERO_SHARES."""
    equity = 100_000.0
    full_exposure = 28_000.0  # 28% cap fully utilized

    res = calculate_swing_position_size(
        entry_price=100.0,
        stop_price=98.0,
        account_equity=equity,
        available_buying_power=50_000.0,
        existing_portfolio_exposure=full_exposure,
    )

    assert res.rejected
    assert res.shares == 0
    assert res.binding_constraint == "ZERO_SHARES"
    assert "NO_WHOLE_SHARE" in res.rejection_reason
