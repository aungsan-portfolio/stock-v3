"""
tests/test_order_manager_p1_5_1.py -- Invariant test suite for Order Manager P1.5.1 Hardening.

Covers 21 critical tests:
1. First partial fill retains pending context.
2. Multiple incremental fills compute weighted average price.
3. Cumulative fill callback does not double-count quantity.
4. Full fill removes pending context.
5. Terminal zero-fill order removes context without creating trailing state.
6. Broker method accepting client_order_id is called exactly once with client_order_id.
7. Broker method without client_order_id is called exactly once without client_order_id.
8. TypeError inside broker method does not result in a second submission attempt.
9. Definite broker rejection releases idempotency lock.
10. Ambiguous submission exception retains idempotency lock.
11. Missing parent order ID retains idempotency lock.
12. Post-submission lifecycle failure retains idempotency lock.
13. Journal failure does not change PLACED result.
14. Discord failure does not change PLACED result.
15. Missing or ambiguous named stop/TP IDs returns reconciliation-required.
16. Dictionary strategy config correctly resolves max_risk_pct.
17. Stale cumulative event is ignored without corrupting average price.
18. Duplicate incremental execution ID is ignored.
19. Partial fill followed by cancel preserves partial position protection.
20. Idempotency attachment failure returns reconciliation-required without releasing reservation.
21. Telemetry warnings do not hide degradation while preserving PLACED status.
"""
from dataclasses import dataclass
import math
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from errors import BrokerOrderRejected, BrokerSubmissionAmbiguous
import strategies.order_manager as om
from strategies.base import TradeSignal


@pytest.fixture(autouse=True)
def reset_order_manager_globals(monkeypatch):
    """Ensure clean global state before and after each test."""
    import config
    monkeypatch.setattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", False)
    monkeypatch.setattr(config, "SPY_TREND_GUARD_ENABLED", False)
    with om._pending_order_contexts_lock:
        om._pending_order_contexts.clear()
    om.reset_session_skip_state()
    yield
    with om._pending_order_contexts_lock:
        om._pending_order_contexts.clear()


def _setup_mock_bridge() -> MagicMock:
    bridge = MagicMock()
    bridge.is_connected = True
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    return bridge


def _dummy_signal(symbol: str = "AAPL", side: str = "BUY", strategy: str = "MOMENTUM") -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        signal_id="sig-test-001",
        strategy=strategy,
        side=side,
        entry_price=100.0,
        stop_price=98.0,
        target_price=105.0,
        confidence=0.85,
        risk_per_share=2.0,
        reason="test signal",
        atr=1.5,
    )


# ---------------------------------------------------------------------------
# P0-1: Partial Fill Lifecycle & Deduplication Tests
# ---------------------------------------------------------------------------

def test_first_partial_fill_retains_pending_context():
    """Test 1: First partial fill retains pending context in _pending_order_contexts."""
    sig = _dummy_signal()
    om.register_pending_order("ord-100", sig, limit_price=100.0, qty=100.0, stop_id="stop-1", tp_id="tp-1")

    with patch("strategies.trailing_stop.get_trailing_manager") as mock_get_tm:
        mock_tm = MagicMock()
        mock_get_tm.return_value = mock_tm

        om.on_broker_fill_event("ord-100", fill_price=100.0, filled_qty=30.0)

        with om._pending_order_contexts_lock:
            assert "ord-100" in om._pending_order_contexts
            ctx = om._pending_order_contexts["ord-100"]
            assert ctx["cumulative_filled_qty"] == 30.0
            assert ctx["average_fill_price"] == 100.0

        mock_tm.apply_confirmed_fill.assert_called_once()
        call_kwargs = mock_tm.apply_confirmed_fill.call_args.kwargs
        assert call_kwargs["cumulative_qty"] == 30.0
        assert call_kwargs["average_fill_price"] == 100.0


def test_multiple_incremental_fills_weighted_average():
    """Test 2: Multiple incremental fills calculate running weighted average price."""
    sig = _dummy_signal()
    om.register_pending_order("ord-101", sig, limit_price=100.0, qty=100.0, stop_id="stop-1", tp_id="tp-1")

    with patch("strategies.trailing_stop.get_trailing_manager") as mock_get_tm:
        mock_tm = MagicMock()
        mock_get_tm.return_value = mock_tm

        # Fill 1: 30 @ 100.0
        om.on_broker_fill_event("ord-101", fill_price=100.0, filled_qty=30.0)
        # Fill 2: 40 @ 102.0
        om.on_broker_fill_event("ord-101", fill_price=102.0, filled_qty=40.0)

        with om._pending_order_contexts_lock:
            assert "ord-101" in om._pending_order_contexts
            ctx = om._pending_order_contexts["ord-101"]
            assert ctx["cumulative_filled_qty"] == 70.0
            # (30*100 + 40*102) / 70 = (3000 + 4080) / 70 = 7080 / 70 = 101.142857...
            assert math.isclose(ctx["average_fill_price"], 7080.0 / 70.0, rel_tol=1e-6)

        assert mock_tm.apply_confirmed_fill.call_count == 2


def test_cumulative_fill_callback_no_double_count():
    """Test 3: Direct cumulative fill callback updates cleanly without double-counting."""
    sig = _dummy_signal()
    om.register_pending_order("ord-102", sig, limit_price=100.0, qty=100.0, stop_id="stop-1", tp_id="tp-1")

    with patch("strategies.trailing_stop.get_trailing_manager") as mock_get_tm:
        mock_tm = MagicMock()
        mock_get_tm.return_value = mock_tm

        # Event 1: Cumulative 30 @ 100.0
        om.on_broker_fill_event("ord-102", fill_price=100.0, filled_qty=30.0, cumulative_filled_qty=30.0, average_fill_price=100.0)
        # Event 2: Cumulative 70 @ 101.14
        om.on_broker_fill_event("ord-102", fill_price=102.0, filled_qty=40.0, cumulative_filled_qty=70.0, average_fill_price=101.14)

        with om._pending_order_contexts_lock:
            ctx = om._pending_order_contexts["ord-102"]
            assert ctx["cumulative_filled_qty"] == 70.0
            assert ctx["average_fill_price"] == 101.14


def test_full_fill_removes_pending_context():
    """Test 4: Full fill removes context from _pending_order_contexts."""
    sig = _dummy_signal()
    om.register_pending_order("ord-103", sig, limit_price=100.0, qty=100.0, stop_id="stop-1", tp_id="tp-1")

    with patch("strategies.trailing_stop.get_trailing_manager"):
        om.on_broker_fill_event("ord-103", fill_price=100.0, filled_qty=100.0, order_status="filled")

        with om._pending_order_contexts_lock:
            assert "ord-103" not in om._pending_order_contexts


def test_terminal_zero_fill_removes_context_without_trailing_state():
    """Test 5: Canceled/rejected order with zero fill cleans context without creating trailing state."""
    sig = _dummy_signal()
    om.register_pending_order("ord-104", sig, limit_price=100.0, qty=100.0, stop_id="stop-1", tp_id="tp-1")

    with patch("strategies.trailing_stop.get_trailing_manager") as mock_get_tm:
        mock_tm = MagicMock()
        mock_get_tm.return_value = mock_tm

        om.on_broker_fill_event("ord-104", fill_price=0.0, filled_qty=0.0, order_status="canceled")

        with om._pending_order_contexts_lock:
            assert "ord-104" not in om._pending_order_contexts

        mock_tm.apply_confirmed_fill.assert_not_called()


def test_stale_cumulative_event_ignored():
    """Test 17: Late callback with smaller cumulative quantity is ignored without overwriting average price."""
    sig = _dummy_signal()
    om.register_pending_order("ord-105", sig, limit_price=100.0, qty=100.0, stop_id="stop-1", tp_id="tp-1")

    with patch("strategies.trailing_stop.get_trailing_manager") as mock_get_tm:
        mock_tm = MagicMock()
        mock_get_tm.return_value = mock_tm

        # Current state: 70 @ 101.14
        om.on_broker_fill_event("ord-105", fill_price=101.14, filled_qty=70.0, cumulative_filled_qty=70.0, average_fill_price=101.14)
        assert mock_tm.apply_confirmed_fill.call_count == 1

        # Late/out-of-order callback: 30 @ 100.00 arrives
        om.on_broker_fill_event("ord-105", fill_price=100.0, filled_qty=30.0, cumulative_filled_qty=30.0, average_fill_price=100.0)

        with om._pending_order_contexts_lock:
            ctx = om._pending_order_contexts["ord-105"]
            assert ctx["cumulative_filled_qty"] == 70.0
            assert ctx["average_fill_price"] == 101.14

        # Mock should NOT have been called a second time
        assert mock_tm.apply_confirmed_fill.call_count == 1


def test_duplicate_incremental_execution_ignored():
    """Test 18: Same execution_id arriving twice does not double-count incremental quantity."""
    sig = _dummy_signal()
    om.register_pending_order("ord-106", sig, limit_price=100.0, qty=100.0, stop_id="stop-1", tp_id="tp-1")

    with patch("strategies.trailing_stop.get_trailing_manager") as mock_get_tm:
        mock_tm = MagicMock()
        mock_get_tm.return_value = mock_tm

        # First arrival: exec-123 fills 30 shares
        om.on_broker_fill_event("ord-106", fill_price=100.0, filled_qty=30.0, execution_id="exec-123")
        assert mock_tm.apply_confirmed_fill.call_count == 1

        # Duplicate arrival: exec-123 arrives again
        om.on_broker_fill_event("ord-106", fill_price=100.0, filled_qty=30.0, execution_id="exec-123")

        with om._pending_order_contexts_lock:
            ctx = om._pending_order_contexts["ord-106"]
            assert ctx["cumulative_filled_qty"] == 30.0

        assert mock_tm.apply_confirmed_fill.call_count == 1


def test_partial_fill_then_canceled_preserves_partial_position():
    """Test 19: Order partially filled (30/100) then canceled preserves active trailing state for 30 shares."""
    sig = _dummy_signal()
    om.register_pending_order("ord-107", sig, limit_price=100.0, qty=100.0, stop_id="stop-1", tp_id="tp-1")

    with patch("strategies.trailing_stop.get_trailing_manager") as mock_get_tm:
        mock_tm = MagicMock()
        mock_get_tm.return_value = mock_tm

        # Partial fill 30 shares
        om.on_broker_fill_event("ord-107", fill_price=100.0, filled_qty=30.0, cumulative_filled_qty=30.0, order_status="partially_filled")
        # Parent order canceled by broker
        om.on_broker_fill_event("ord-107", fill_price=100.0, filled_qty=0.0, cumulative_filled_qty=30.0, order_status="canceled")

        # Context removed because order reached terminal status
        with om._pending_order_contexts_lock:
            assert "ord-107" not in om._pending_order_contexts

        # Position protection remained updated for 30 shares
        call_kwargs = mock_tm.apply_confirmed_fill.call_args.kwargs
        assert call_kwargs["cumulative_qty"] == 30.0


# ---------------------------------------------------------------------------
# P0-2: Deterministic Single-Call Submission Tests
# ---------------------------------------------------------------------------

def test_single_submit_helper_passes_client_order_id_when_supported():
    """Test 6: Broker method accepting client_order_id receives client_order_id."""
    bridge = MagicMock()
    # Define method signature with client_order_id
    def mock_place(symbol, side, qty, entry_price, stop_price, target_price, client_order_id=None):
        return {"status": "ok", "client_order_id": client_order_id}
    bridge.place_bracket_order = mock_place

    res = om._place_bracket_order_once(
        bridge,
        symbol="AAPL",
        side="BUY",
        qty=10,
        entry_price=100.0,
        stop_price=98.0,
        target_price=105.0,
        client_order_id="cid-123",
    )
    assert res["client_order_id"] == "cid-123"


def test_single_submit_helper_omits_client_order_id_when_unsupported():
    """Test 7: Broker method without client_order_id does NOT receive client_order_id."""
    bridge = MagicMock()
    # Define method signature without client_order_id
    def mock_place(symbol, side, qty, entry_price, stop_price, target_price):
        return {"status": "ok", "received_symbol": symbol}
    bridge.place_bracket_order = mock_place

    res = om._place_bracket_order_once(
        bridge,
        symbol="AAPL",
        side="BUY",
        qty=10,
        entry_price=100.0,
        stop_price=98.0,
        target_price=105.0,
        client_order_id="cid-123",
    )
    assert res["received_symbol"] == "AAPL"


def test_typeerror_inside_broker_does_not_retry():
    """Test 8: TypeError occurring inside broker method does not trigger duplicate submission."""
    call_count = 0
    def mock_failing_place(**kwargs):
        nonlocal call_count
        call_count += 1
        raise TypeError("Simulated internal TypeError during response processing")

    bridge = MagicMock()
    bridge.place_bracket_order = mock_failing_place

    with pytest.raises(TypeError):
        om._place_bracket_order_once(
            bridge,
            symbol="AAPL",
            side="BUY",
            qty=10,
            entry_price=100.0,
            stop_price=98.0,
            target_price=105.0,
            client_order_id="cid-123",
        )
    assert call_count == 1


# ---------------------------------------------------------------------------
# P0-3 & P0-4: Submission Error Boundaries & Leg Validation Tests
# ---------------------------------------------------------------------------

def test_definite_rejection_releases_idempotency_lock():
    """Test 9: Explicit broker rejection releases idempotency reservation."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()

    def mock_place(**kwargs):
        raise BrokerOrderRejected("Account buying power exhausted")

    bridge.place_bracket_order = mock_place

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp:
        mock_idemp.reserve.return_value = (True, "reserved")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "REJECTED"
        assert "Broker definitely rejected order" in res["reason"]
        mock_idemp.release.assert_called_once()
        assert "definite_submit_rejection" in mock_idemp.release.call_args.kwargs["reason"]


def test_ambiguous_submission_retains_idempotency_lock():
    """Test 10: Uncertain submission exception (e.g. timeout) retains idempotency reservation."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()

    def mock_place(**kwargs):
        raise BrokerSubmissionAmbiguous("Gateway timeout waiting for broker response")

    bridge.place_bracket_order = mock_place

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp:
        mock_idemp.reserve.return_value = (True, "reserved")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "AMBIGUOUS_SUBMIT"
        assert "Idempotency lock retained" in res["reason"]
        mock_idemp.release.assert_not_called()


def test_missing_parent_id_retains_idempotency_lock():
    """Test 11: Missing parent order ID returns AMBIGUOUS_SUBMIT and retains lock."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()
    # Broker returned dict without parent ID
    bridge.place_bracket_order = lambda **kwargs: {"status": "ok", "order_ids": []}

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp:
        mock_idemp.reserve.return_value = (True, "reserved")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "AMBIGUOUS_SUBMIT"
        mock_idemp.release.assert_not_called()


def test_missing_named_stop_tp_legs_returns_reconciliation_required():
    """Test 15: Missing or identical named stop/TP legs returns SUBMITTED_RECONCILIATION_REQUIRED."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()
    # Missing stop_order_id and tp_order_id
    bridge.place_bracket_order = lambda **kwargs: {
        "parent_order_id": "parent-101",
        "order_ids": ["parent-101", "leg-1", "leg-2"],
    }

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp:
        mock_idemp.reserve.return_value = (True, "reserved")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "SUBMITTED_RECONCILIATION_REQUIRED"
        assert "named stop/target leg IDs are missing" in res["reason"]
        mock_idemp.release.assert_not_called()


def test_idempotency_attachment_failure():
    """Test 20: Idempotency attachment failure returns SUBMITTED_RECONCILIATION_REQUIRED without releasing lock."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()
    bridge.place_bracket_order = lambda **kwargs: {
        "parent_order_id": "parent-102",
        "stop_order_id": "stop-102",
        "tp_order_id": "tp-102",
    }

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp:
        mock_idemp.reserve.return_value = (True, "reserved")
        mock_idemp.attach_parent_order.side_effect = RuntimeError("Disk full writing idempotency journal")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "SUBMITTED_RECONCILIATION_REQUIRED"
        assert "durable idempotency attachment failed" in res["reason"]
        mock_idemp.release.assert_not_called()


def test_post_submit_lifecycle_failure_retains_lock():
    """Test 12: Lifecycle registration error returns SUBMITTED_RECONCILIATION_REQUIRED and retains lock."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()
    bridge.place_bracket_order = lambda **kwargs: {
        "parent_order_id": "parent-103",
        "stop_order_id": "stop-103",
        "tp_order_id": "tp-103",
    }

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp, \
         patch("strategies.entry_lifecycle.entry_lifecycle.register_submission") as mock_reg:
        mock_idemp.reserve.return_value = (True, "reserved")
        mock_reg.side_effect = RuntimeError("Lifecycle registry corruption")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "SUBMITTED_RECONCILIATION_REQUIRED"
        assert "local lifecycle registration failed" in res["reason"]
        mock_idemp.release.assert_not_called()


# ---------------------------------------------------------------------------
# Telemetry & Non-Transactional Reporting Tests
# ---------------------------------------------------------------------------

def test_journal_failure_preserves_placed_status():
    """Test 13: Journal logging failure does NOT prevent PLACED status."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()
    bridge.place_bracket_order = lambda **kwargs: {
        "parent_order_id": "parent-104",
        "stop_order_id": "stop-104",
        "tp_order_id": "tp-104",
    }

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp, \
         patch("strategies.order_manager.log_trade") as mock_log, \
         patch("strategies.order_manager.send_discord_alert"):
        mock_idemp.reserve.return_value = (True, "reserved")
        mock_log.side_effect = IOError("Database locked during trade log")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "PLACED"
        assert any("journal_write_failed" in w for w in res["warnings"])


def test_discord_failure_preserves_placed_status():
    """Test 14: Discord alert webhook failure does NOT prevent PLACED status."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()
    bridge.place_bracket_order = lambda **kwargs: {
        "parent_order_id": "parent-105",
        "stop_order_id": "stop-105",
        "tp_order_id": "tp-105",
    }

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp, \
         patch("strategies.order_manager.log_trade"), \
         patch("strategies.order_manager.send_discord_alert") as mock_alert:
        mock_idemp.reserve.return_value = (True, "reserved")
        mock_alert.side_effect = ConnectionError("Discord API unreachable")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "PLACED"
        assert any("discord_alert_failed" in w for w in res["warnings"])


def test_telemetry_warnings_do_not_hide_degradation():
    """Test 21: Expected price registration failure captures warning without altering PLACED status."""
    sig = _dummy_signal()
    bridge = _setup_mock_bridge()
    bridge.place_bracket_order = lambda **kwargs: {
        "parent_order_id": "parent-106",
        "stop_order_id": "stop-106",
        "tp_order_id": "tp-106",
    }

    with patch("strategies.order_manager.idempotency_manager") as mock_idemp, \
         patch("strategies.order_registry.register_order_expected_price") as mock_reg, \
         patch("strategies.order_manager.log_trade"), \
         patch("strategies.order_manager.send_discord_alert"):
        mock_idemp.reserve.return_value = (True, "reserved")
        mock_reg.side_effect = RuntimeError("Order registry lock contention")

        res = om.execute_signal(sig, bridge, equity=10000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "PLACED"
        assert any("expected_price_reg_failed" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# Configuration & Cleanup Tests
# ---------------------------------------------------------------------------

def test_dict_strategy_settings_resolves_risk_pct():
    """Test 16: STRATEGY_SETTINGS as dictionary resolves max_risk_pct without AttributeError."""
    import config
    original = getattr(config, "STRATEGY_SETTINGS", {})
    try:
        config.STRATEGY_SETTINGS = {
            "MOMENTUM": {"max_risk_pct": 2.5},
            "ORB": {"max_risk_pct": 1.75},
        }
        assert om._strategy_max_risk_pct("MOMENTUM") == 2.5
        assert om._strategy_max_risk_pct("ORB") == 1.75
        assert om._strategy_max_risk_pct("UNKNOWN") == 1.0
    finally:
        config.STRATEGY_SETTINGS = original
