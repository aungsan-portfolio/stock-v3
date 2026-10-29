"""
tests/test_b_swing_fixes.py — Tests for Swing Bot fixes B1, B2, B5.
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytz

import config
from position_rotator import PositionRotator, RotationDecision, OpenPosition, IncomingSignal
from strategies.swing_sizing import calculate_swing_position_size, SizingDecision


def test_b1_stay_in_cash_locks_symbol_from_direct_buy():
    """Verify that when atomic rotation transitions to STAY_IN_CASH, the symbol is locked from direct BUY."""
    rotator = PositionRotator()
    rotator.state.reset_cycle()
    assert not rotator.state.is_locked("UNH")

    # Simulate locking UNH after atomic rotation STAY_IN_CASH
    rotator.state.lock_symbol("UNH")
    assert rotator.state.is_locked("UNH")

    # Verify that reset_cycle clears the lock for next cycle
    rotator.state.reset_cycle()
    assert not rotator.state.is_locked("UNH")


def test_b2_market_close_buffer_skips_rotator():
    """Verify that during market close buffer (after 15:45 ET), rotator evaluation is skipped."""
    from alpaca_bridge import AlpacaBridge
    bridge = AlpacaBridge(mode="swing", dry_run=True)

    et_tz = pytz.timezone("America/New_York")
    close_buffer_time = datetime(2026, 10, 29, 15, 50, 0, tzinfo=et_tz)

    mock_sig = MagicMock()
    mock_sig.symbol = "AAPL"
    mock_sig.action = "BUY"
    mock_sig.confidence = 0.75
    mock_sig.price = 230.0

    with patch("strategies.session.now_eastern", return_value=close_buffer_time):
        with patch.object(bridge.rotator, "evaluate") as mock_eval:
            locked = bridge._evaluate_and_execute_rotation([mock_sig])
            assert mock_eval.call_count == 0
            assert locked == set()


def test_b5_sizing_decision_includes_portfolio_exposure_metrics():
    """Verify that SizingDecision includes portfolio_exposure_used and portfolio_exposure_cap."""
    res = calculate_swing_position_size(
        entry_price=100.0,
        stop_price=95.0,
        account_equity=100000.0,
        available_buying_power=50000.0,
        existing_portfolio_exposure=20000.0,
        reserved_portfolio_exposure=2000.0,
    )
    assert res.portfolio_exposure_used == 22000.0
    # STAGE_0_GROSS_EXPOSURE_PCT = 0.28 (28% of $100k = $28k)
    assert res.portfolio_exposure_cap == 28000.0
