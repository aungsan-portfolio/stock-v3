"""
tests/test_swing_rotation_hardening.py — Focused tests for Swing Rotation Hardening.

Covers the 4 contract amendments:
  • Amendment 1: close_position() strictly returns primitive bool (backward compatibility).
  • Amendment 2: close_position_verified() enforces pre-close cancel convergence polling.
  • Amendment 3: Post-close query failure is strictly UNKNOWN (UNKNOWN != FLAT); aborts replacement.
  • Reprotection of residual position on partial / aborted close.
"""

from unittest.mock import MagicMock, patch
import pytest

from alpaca_bridge import AlpacaBridge
from strategies.atomic_rotation import (
    AtomicRotationEngine,
    CloseOutcome,
    PositionCloseResult,
    RotationState,
)


def _make_bridge_for_hardening():
    bridge = AlpacaBridge(mode="swing", dry_run=False)
    bridge._client = MagicMock()
    bridge._connected = True
    bridge.dry_run = False
    bridge.atomic_rotator = AtomicRotationEngine()
    return bridge


def test_close_position_primitive_bool_contract():
    """Amendment 1: close_position() MUST return primitive bool, never a truthy dataclass."""
    bridge = _make_bridge_for_hardening()

    # Case A: Successful flat close -> returns True (primitive bool)
    with patch.object(bridge, "close_position_verified") as mock_verified:
        mock_verified.return_value = PositionCloseResult(
            outcome=CloseOutcome.FLAT,
            symbol="AMZN",
            initial_qty=10.0,
            remaining_qty=0.0,
            filled_qty=10.0,
        )
        res = bridge.close_position("AMZN")
        assert res is True
        assert type(res) is bool

    # Case B: Failed close (e.g. TIMEOUT) -> returns False (primitive bool)
    with patch.object(bridge, "close_position_verified") as mock_verified:
        mock_verified.return_value = PositionCloseResult(
            outcome=CloseOutcome.TIMEOUT,
            symbol="AMZN",
            initial_qty=10.0,
            remaining_qty=10.0,
            filled_qty=0.0,
            error="Close order timed out",
        )
        res = bridge.close_position("AMZN")
        assert res is False
        assert type(res) is bool
        # Critical invariant: must NOT evaluate truthy as a dataclass object would
        assert not res

    # Case C: Unknown state (Amendment 3) -> returns False
    with patch.object(bridge, "close_position_verified") as mock_verified:
        mock_verified.return_value = PositionCloseResult(
            outcome=CloseOutcome.UNKNOWN,
            symbol="AMZN",
            initial_qty=10.0,
            remaining_qty=10.0,
            filled_qty=0.0,
            error="Broker API outage post-close",
        )
        res = bridge.close_position("AMZN")
        assert res is False
        assert type(res) is bool


def test_cancel_convergence_timeout_fails_closed():
    """Amendment 2: If active orders do not converge to zero within bounded window, close fails closed as UNKNOWN."""
    bridge = _make_bridge_for_hardening()

    # Initial position is 10 shares
    mock_pos = MagicMock()
    mock_pos.qty = 10.0
    bridge.get_position = MagicMock(return_value=mock_pos)

    # Mock working orders that refuse to clear (never converge)
    mock_active = MagicMock()
    mock_active.symbol = "COST"
    bridge._get_active_orders = MagicMock(return_value=[mock_active])
    bridge._cancel_all_orders_for_symbol = MagicMock()

    # Fast convergence check in test
    with patch("time.sleep", return_value=None):
        res = bridge.close_position_verified("COST")

    assert res.outcome == CloseOutcome.UNKNOWN
    assert not res.is_flat
    assert "convergence timeout" in (res.error or "").lower()
    # Close order must NEVER be submitted if working orders did not clear
    bridge._client.close_position.assert_not_called()


def test_post_close_query_failure_marked_strictly_unknown():
    """Amendment 3: If broker position re-query fails post-close, outcome is UNKNOWN, never FLAT."""
    bridge = _make_bridge_for_hardening()

    # Initial query succeeds: holds 10 shares
    pos_calls = [10.0]

    def _get_pos_side_effect(sym):
        if pos_calls:
            return pos_calls.pop(0)
        # Post-close query raises network outage error
        raise RuntimeError("Broker connection dropped during position query")

    bridge.get_position = MagicMock(side_effect=_get_pos_side_effect)
    bridge._get_active_orders = MagicMock(return_value=[])
    bridge._cancel_all_orders_for_symbol = MagicMock()

    mock_close_ord = MagicMock()
    mock_close_ord.id = "close_123"
    bridge._client.close_position = MagicMock(return_value=mock_close_ord)

    with patch("time.sleep", return_value=None):
        res = bridge.close_position_verified("COST")

    assert res.outcome == CloseOutcome.UNKNOWN
    assert res.is_flat is False
    assert "post-close position query failed" in (res.error or "").lower()


def test_anti_reversal_guard_prevents_retry():
    """Controlled retry guard: if position flips sign (long->short), refuse retry to prevent escalation."""
    bridge = _make_bridge_for_hardening()

    # Initial query: long 10. Post-close query: short -2 (oversold by competing event)
    pos_calls = [10.0, -2.0]
    bridge.get_position = MagicMock(side_effect=lambda sym: pos_calls.pop(0) if pos_calls else -2.0)
    bridge._get_active_orders = MagicMock(return_value=[])
    bridge._cancel_all_orders_for_symbol = MagicMock()

    mock_close_ord = MagicMock()
    mock_close_ord.id = "close_rev_1"
    bridge._client.close_position = MagicMock(return_value=mock_close_ord)

    with patch("time.sleep", return_value=None):
        res = bridge.close_position_verified("TSLA")

    assert res.outcome != CloseOutcome.FLAT
    assert "reversal detected" in (res.error or "").lower()
    # Exactly 1 close attempt, no retry
    assert bridge._client.close_position.call_count == 1


def test_ensure_symbol_protection_places_gtc_stop():
    """Reprotection: ensure_symbol_protection places a GTC stop order for residual quantity."""
    bridge = _make_bridge_for_hardening()
    bridge.working_orders_plain = MagicMock(return_value=[])
    bridge.market_price = MagicMock(return_value=200.0)

    mock_stop_order = MagicMock()
    mock_stop_order.id = "reprotect_stop_1"
    bridge._client.submit_order = MagicMock(return_value=mock_stop_order)

    success = bridge.ensure_symbol_protection("COST", qty=5.0)
    assert success is True
    bridge._client.submit_order.assert_called_once()
    req = bridge._client.submit_order.call_args[0][0]
    assert req.symbol == "COST"
    assert req.qty == 5.0
    assert req.stop_price == 196.0  # 2% below 200.0


def test_ensure_symbol_protection_skips_when_already_covered():
    """Reprotection skips submission if existing active stop already covers required qty."""
    bridge = _make_bridge_for_hardening()
    bridge.working_orders_plain = MagicMock(return_value=[{
        "symbol": "COST",
        "order_type": "STOP",
        "action": "SELL",
        "qty": 10.0,
        "stop_price": 195.0,
    }])

    success = bridge.ensure_symbol_protection("COST", qty=5.0)
    assert success is True
    bridge._client.submit_order.assert_not_called()
