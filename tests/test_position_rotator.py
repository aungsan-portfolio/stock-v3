"""
tests/test_position_rotator.py — Tests for PositionRotator candidate selection and winner protection invariants.

Locks Amendment A invariant (Reading 2: SUPERSEDED by design, winner protection satisfied upstream).
Ensures that accelerating winners (unrealized_r >= 1.0 or pnl_pct >= 0.05) are never selected as rotation candidates,
even if their re-scored confidence is weak.
"""

import pytest
from position_rotator import OpenPosition, PositionRotator, ROTATION_PROTECT_PROFIT_PCT


def test_profit_accelerating_winner_never_rotation_candidate():
    """
    Amendment A Invariant Lock (Reading 2):
    An open position that is an accelerating winner MUST NEVER be included in rotation candidates,
    even if current_conf falls below ROTATION_WEAK_POSITION_CONF (0.35).
    """
    rotator = PositionRotator()

    # Case 1: High R-multiple winner (unrealized_r >= 1.0) with weak confidence
    winner_r = OpenPosition(
        symbol="AAPL",
        side="long",
        qty=10.0,
        avg_entry_price=150.0,
        current_price=165.0,
        unrealized_pnl=150.0,
        current_conf=0.20,  # Below weak threshold 0.35
        unrealized_r=1.2,   # R >= 1.0 winner
    )

    # Case 2: PnL percentage winner (pnl_pct >= ROTATION_PROTECT_PROFIT_PCT = 0.05) with weak confidence
    winner_pnl = OpenPosition(
        symbol="MSFT",
        side="long",
        qty=10.0,
        avg_entry_price=100.0,
        current_price=108.0,  # +8.0% PnL >= 5%
        unrealized_pnl=80.0,
        current_conf=0.20,   # Below weak threshold 0.35
        unrealized_r=0.6,
    )

    # Case 3: Legitimate weak non-winner that SHOULD be a candidate
    weak_pos = OpenPosition(
        symbol="XYZ",
        side="long",
        qty=10.0,
        avg_entry_price=100.0,
        current_price=98.0,   # -2.0% PnL
        unrealized_pnl=-20.0,
        current_conf=0.20,   # Below weak threshold 0.35
        unrealized_r=-0.4,
    )

    candidates = rotator._get_rotation_candidates([winner_r, winner_pnl, weak_pos])

    # Assertions: Accelerating winners are strictly excluded upstream
    assert winner_r not in candidates, "Position with unrealized_r >= 1.0 must be skipped from rotation candidates"
    assert winner_pnl not in candidates, f"Position with pnl_pct >= {ROTATION_PROTECT_PROFIT_PCT} must be skipped from rotation candidates"
    assert weak_pos in candidates, "Weak non-winner position should be selected as rotation candidate"
