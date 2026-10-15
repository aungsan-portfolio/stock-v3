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


# ─── P0/P1 Hardening Tests (Daily Loss, Fail-Closed, Rollover, Quote Eligibility) ────────

def _make_rotation_decision(
    exit_symbol: str = "AMZN",
    enter_symbol: str = "JPM",
    exit_qty: float = 8.0,
    exit_entry_price: float = 251.57,
    exit_current_price: float = 255.07,
    unrealized_r: float = 0.5,
    enter_conf: float = 0.62,
    enter_price: float = 215.50,
):
    from position_rotator import IncomingSignal, OpenPosition, RotationDecision
    exit_pos = OpenPosition(
        symbol=exit_symbol,
        side="long",
        qty=exit_qty,
        avg_entry_price=exit_entry_price,
        current_price=exit_current_price,
        unrealized_pnl=(exit_current_price - exit_entry_price) * exit_qty,
        current_conf=0.43,
        unrealized_r=unrealized_r,
    )
    enter_sig = IncomingSignal(
        symbol=enter_symbol,
        action="BUY",
        confidence=enter_conf,
        price=enter_price,
    )
    return RotationDecision(
        should_rotate=True,
        exit_position=exit_pos,
        enter_signal=enter_sig,
        reason="Conviction upgrade",
    )


def test_fresh_unpersisted_daily_loss_breach_vetoes_precheck_and_preserves_incumbent():
    """P0 Gap 1: Fresh live daily loss breach halts engine and vetoes rotation before close."""
    import tempfile
    from pathlib import Path
    import config as config_mod

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        with patch.object(config_mod, "DATA_DIR", tmp_path):
            engine = AtomicRotationEngine()
            bridge = AlpacaBridge(mode="swing", dry_run=True)
            bridge._connected = True
            bridge._client = MagicMock()
            bridge._account_id = "PA3R71I7JFDZ"
            # Starting equity $100,000, current equity $99,690 (loss = $310 > $300 cap)
            bridge._start_of_day_equity = 100000.0
            bridge.get_net_liquidation = MagicMock(return_value=99690.0)

            decision = _make_rotation_decision(exit_symbol="AMZN", enter_symbol="JPM", unrealized_r=0.5)

            feasible, reason = engine.check_feasibility(decision, bridge, [])
            assert feasible is False
            assert "daily loss" in reason.lower()

            # Full execution aborts cleanly without closing incumbent
            bridge.close_position_verified = MagicMock()
            summary = engine.rotate(decision, bridge, [])
            assert summary.rotation_completed is False
            assert summary.replacement_entered is False
            bridge.close_position_verified.assert_not_called()


def test_daily_loss_verification_exception_fails_closed():
    """P0 Gap 2: Risk check exceptions must fail closed (veto rotation/entry)."""
    engine = AtomicRotationEngine()
    bridge = MagicMock()
    # Simulate exception in _is_daily_loss_halted
    bridge._is_daily_loss_halted.side_effect = RuntimeError("Broker database unreachable")

    decision = _make_rotation_decision(exit_symbol="AMZN", enter_symbol="JPM", unrealized_r=0.5)
    feasible, reason = engine.check_feasibility(decision, bridge, [])
    assert feasible is False
    assert "failing closed" in reason.lower()


def test_session_rollover_preserves_stream_disconnect_halt():
    """P1 Gap 3: Session rollover resets expired daily loss halt but preserves stream disconnect."""
    import json
    import tempfile
    from pathlib import Path
    import config as config_mod

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = "PA3R71I7JFDZ"

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        with patch.object(config_mod, "DATA_DIR", tmp_path):
            # Create a previous session halt file (dated yesterday)
            halt_file = tmp_path / "swing_daily_loss_halt_PA3R71I7JFDZ.json"
            halt_file.write_text(json.dumps({
                "account_id": "PA3R71I7JFDZ",
                "session_date_et": "2026-10-13",
                "daily_pnl": "-350.00",
                "daily_loss_limit": "300.00",
                "halted": True,
                "reason": "DAILY_LOSS_LIMIT",
            }), encoding="utf-8")

            # Set entry gate with both DAILY_LOSS_LIMIT and TRADE_STREAM_DISCONNECTED
            bridge.entry_gate.reasons = {"DAILY_LOSS_LIMIT", "TRADE_STREAM_DISCONNECTED"}
            bridge.entry_gate.halted = True
            bridge.entry_gate.halt_reason = "TRADE_STREAM_DISCONNECTED"

            # Check halt state for new session date "2026-10-14"
            is_halted, reason = bridge.entry_gate.check_daily_loss_halt_state("PA3R71I7JFDZ", "2026-10-14")

            # Daily loss limit file must be unlinked
            assert not halt_file.exists()
            assert "DAILY_LOSS_LIMIT" not in bridge.entry_gate.reasons

            # BUT TRADE_STREAM_DISCONNECTED must still be active (never wiped out)
            assert is_halted is True
            assert bridge.entry_gate.halted is True
            assert "TRADE_STREAM_DISCONNECTED" in bridge.entry_gate.reasons


def test_validate_quote_rejects_fallback_price_sources_for_entry():
    """P1 Gap 4: Entry execution strictly requires two-sided quotes; fallbacks are rejected."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True

    from alpaca_bridge import QuoteDecision

    # 1. Alpaca two-sided top-of-book quote -> Accepted
    with patch.object(bridge, "get_order_quote") as mock_goq:
        mock_goq.return_value = QuoteDecision(True, 150.0, "alpaca", "ok")
        ok, px, reason = bridge.validate_quote("AAPL", require_two_sided=True)
        assert ok is True
        assert px == 150.0

    # 2. Alpaca IEX trade fallback -> Rejected for entry
    with patch.object(bridge, "get_order_quote") as mock_goq:
        mock_goq.return_value = QuoteDecision(True, 150.0, "alpaca_iex_trade_fallback", "ok_fallback_from_wide_iex_spread")
        ok, px, reason = bridge.validate_quote("AAPL", require_two_sided=True)
        assert ok is False
        assert "fallback price source" in reason.lower()

    # 3. Alpaca IEX bar fallback -> Rejected for entry
    with patch.object(bridge, "get_order_quote") as mock_goq:
        mock_goq.return_value = QuoteDecision(True, 150.0, "alpaca_iex_bar_fallback", "ok_fallback_from_wide_iex_spread")
        ok, px, reason = bridge.validate_quote("AAPL", require_two_sided=True)
        assert ok is False
        assert "fallback price source" in reason.lower()

    # 4. Yahoo Finance fallback -> Rejected for entry
    with patch.object(bridge, "get_order_quote") as mock_goq:
        mock_goq.return_value = QuoteDecision(True, 150.0, "yfinance_fallback", "ok_fallback_from_wide_iex_spread")
        ok, px, reason = bridge.validate_quote("AAPL", require_two_sided=True)
        assert ok is False
        assert "fallback price source" in reason.lower()


def test_is_daily_loss_halted_fail_closed_on_invalid_equity_and_sod(tmp_path):
    """Verify that _is_daily_loss_halted() strictly fails closed on invalid/missing equity and SOD baseline."""
    import config
    with patch.object(config, "DATA_DIR", tmp_path):
        bridge = AlpacaBridge(mode="swing", dry_run=True)
        bridge._connected = True

        # 1. Invalid current equity (0.0, negative, NaN, Inf) -> Fails closed (returns True)
        for bad_eq in [0.0, -100.0, float("nan"), float("inf"), float("-inf")]:
            with patch.object(bridge, "get_net_liquidation", return_value=bad_eq):
                bridge._start_of_day_equity = 10000.0
                assert bridge._is_daily_loss_halted() is True, f"Failed closed check for bad current equity {bad_eq}"

        # 2. Invalid or missing start_of_day_equity baseline -> Fails closed (returns True)
        with patch.object(bridge, "get_net_liquidation", return_value=10000.0):
            for bad_sod in [0.0, -500.0, float("nan"), float("inf"), None]:
                bridge._start_of_day_equity = bad_sod
                with patch.object(bridge, "snapshot_start_of_day_equity", return_value=bad_sod):
                    assert bridge._is_daily_loss_halted() is True, f"Failed closed check for bad SOD equity {bad_sod}"

        # 3. Valid normal equity within daily loss limit -> Returns False
        with patch.object(bridge, "get_net_liquidation", return_value=9950.0):
            bridge._start_of_day_equity = 10000.0
            # Loss is $50, well under $300 limit
            assert bridge._is_daily_loss_halted() is False

        # 4. Valid equity breaching daily loss limit -> Returns True (halts)
        with patch.object(bridge, "get_net_liquidation", return_value=9650.0):
            bridge._start_of_day_equity = 10000.0
            # Loss is $350 >= $300 limit
            assert bridge._is_daily_loss_halted() is True


def test_snapshot_start_of_day_equity_fail_closed_without_broker_last_equity(tmp_path):
    """snapshot_start_of_day_equity() strictly requires broker_last_equity and rejects falling back to current equity."""
    import config
    from unittest.mock import MagicMock
    with patch.object(config, "DATA_DIR", tmp_path):
        bridge = AlpacaBridge(mode="swing", dry_run=True)
        bridge._connected = True
        bridge._risk_state_dir = str(tmp_path)
        bridge._start_of_day_equity = 0.0
        bridge._start_of_day_equity_date = None

        # Case A: Client get_account returns no last_equity or last_equity <= 0 -> fails closed (returns 0.0)
        mock_client = MagicMock()
        mock_acct = MagicMock()
        mock_acct.last_equity = 0.0
        mock_acct.equity = 9800.0
        mock_client.get_account.return_value = mock_acct
        bridge._client = mock_client
        with patch.object(bridge, "get_net_liquidation", return_value=9800.0):
            sod = bridge.snapshot_start_of_day_equity()
            assert sod == 0.0

        # Case B: Client returns verified last_equity -> accepted and persisted with provenance
        mock_acct.last_equity = 10000.0
        sod = bridge.snapshot_start_of_day_equity()
        assert sod == 10000.0
        saved_state = bridge._load_daytrade_risk_state(bridge._account_id)
        assert saved_state["start_of_day_equity"] == 10000.0
        assert saved_state["provenance"]["source"] == "broker_last_equity"

        # Case C: Same-day positive in-memory SOD exists but provenance is unverified -> NOT preserved
        import os
        state_file = os.path.join(str(tmp_path), f"daytrade_risk_state_{bridge._account_id}.json")
        if os.path.exists(state_file):
            os.remove(state_file)

        from strategies.session import now_eastern
        today_str = str(now_eastern().date())
        bridge._start_of_day_equity = 10000.0
        bridge._start_of_day_equity_date = today_str
        bridge._start_of_day_equity_source = "unverified_nominal"
        mock_acct.last_equity = 0.0  # Broker unavailable
        sod = bridge.snapshot_start_of_day_equity()
        assert sod == 0.0, "In-memory SOD without verified broker_last_equity source must not be preserved"


def test_validate_quote_fails_closed_when_source_attribute_missing():
    """validate_quote() defaults missing source to 'unknown' and fails closed when require_two_sided=True."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True

    class DummyQuoteDecision:
        ok = True
        price = 150.0
        reason = "ok"
        # Notice: source attribute is intentionally missing!

    with patch.object(bridge, "get_order_quote", return_value=DummyQuoteDecision()):
        ok, px, reason = bridge.validate_quote("AAPL", require_two_sided=True)
        assert ok is False
        assert "fallback price source (unknown) rejected" in reason


def test_atomic_rotation_fails_closed_on_unsupported_require_two_sided():
    """AtomicRotationEngine fails closed if bridge.validate_quote raises TypeError on require_two_sided."""
    from strategies.atomic_rotation import AtomicRotationEngine
    engine = AtomicRotationEngine()
    mock_bridge = MagicMock()
    mock_bridge._is_daily_loss_halted.return_value = False
    mock_bridge.validate_quote.side_effect = TypeError("validate_quote() got an unexpected keyword argument 'require_two_sided'")

    mock_decision = MagicMock()
    mock_decision.enter_signal.symbol = "AAPL"
    mock_decision.exit_position.symbol = "MSFT"

    ok, reason = engine.check_feasibility(mock_decision, mock_bridge, [])
    assert ok is False
    assert "does not support require_two_sided enforcement" in reason
