import time
import pytest
from unittest.mock import Mock

from strategies.base import TradeSignal, EntryReadiness
from strategies.constants import StrategyName
from strategies.execution_health import get_execution_health, HealthStatus
from strategies.order_manager import execute_signal


@pytest.fixture
def mock_bridge():
    bridge = Mock()
    bridge.is_connected = True
    bridge._connected = True
    bridge.stream_state = "HEALTHY"
    bridge.entry_gate = Mock()
    bridge.entry_gate.halted = False
    bridge.entry_gate.reasons = set()
    bridge.working_orders_plain.return_value = []
    bridge._get_active_orders.return_value = []
    bridge.get_position.return_value = 0.0
    bridge.get_positions.return_value = []
    bridge.positions_plain.return_value = []
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.open_position_count.return_value = 0
    bridge.daily_loss_dollars.return_value = 0.0
    bridge.get_daily_loss.return_value = 0.0
    bridge.is_circuit_breaker_active.return_value = False
    return bridge


@pytest.fixture
def sample_signal():
    return TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=150.00,
        stop_price=149.00,
        target_price=152.00,
        atr=1.00,
        risk_per_share=1.00,
        reason="Test breakout",
        bar_timestamp="2026-10-21T02:35:00+00:00",
    )


def test_coverage_54_8_pct_and_timeouts_blocks_order_submission(mock_bridge, sample_signal):
    """
    Test exact Oct 21 02:39:02 ICT event:
    Coverage = 17/31 = 54.8%, SDK timeouts = 11.
    Even if market regime is RISK_ON, execution health MUST fail-closed,
    EntryReadiness token MUST be suppressed, and place_bracket_order MUST NEVER be called.
    """
    feed_tel = {
        "coverage": 0.548,
        "consecutive_fetch_timeouts": 11,
        "last_batch_duration_s": 3.5,
        "stale_bar_skip_count": 3,
    }

    # 1. Execution health gate check
    health = get_execution_health(mock_bridge, bar_age_seconds=20.0, feed_telemetry=feed_tel)
    assert not health.allows_new_entries
    assert health.status == HealthStatus.DEGRADED
    assert "coverage 54.8% below required threshold 95.0%" in health.reason or "fetch timeouts" in health.reason

    # 2. EntryReadiness suppression
    now_ts = time.time()
    readiness = EntryReadiness(
        entry_allowed=health.allows_new_entries,
        suppression_reason=health.reason,
        cycle_id="test_cycle",
        bar_ts="2026-10-21T02:35:00+00:00",
        created_at=now_ts,
        expires_at=now_ts + 300.0,
    )
    assert not readiness.entry_allowed

    # 3. Final Execution Boundary Check
    res = execute_signal(
        sample_signal,
        bridge=mock_bridge,
        equity=10000.0,
        current_pnl=0.0,
        entry_readiness=readiness,
        dry_run=False,
    )

    assert res["status"] in ("REJECTED", "VETOED")
    assert "ENTRY_READINESS_SUPPRESSED" in res["reason"]
    assert mock_bridge.place_bracket_order.called is False


def test_stream_reconnecting_blocks_order_submission(mock_bridge, sample_signal):
    """
    Trade updates WebSocket in RECONNECTING state MUST fail-closed at order boundary.
    """
    mock_bridge.stream_state = "RECONNECTING"

    health = get_execution_health(mock_bridge, bar_age_seconds=15.0)
    assert not health.allows_new_entries
    assert health.status == HealthStatus.DEGRADED

    now_ts = time.time()
    readiness = EntryReadiness(
        entry_allowed=health.allows_new_entries,
        suppression_reason=health.reason,
        cycle_id="test_cycle",
        bar_ts="2026-10-21T02:35:00+00:00",
        created_at=now_ts,
        expires_at=now_ts + 300.0,
    )

    res = execute_signal(
        sample_signal,
        bridge=mock_bridge,
        equity=10000.0,
        current_pnl=0.0,
        entry_readiness=readiness,
        dry_run=False,
    )

    assert res["status"] in ("REJECTED", "VETOED")
    assert mock_bridge.place_bracket_order.called is False


def test_broker_reconciliation_pending_blocks_order_submission(mock_bridge, sample_signal):
    """
    Broker reconciliation pending MUST fail-closed at order boundary.
    """
    mock_bridge.entry_gate.halted = True
    mock_bridge.entry_gate.reasons = {"BROKER_RECONCILIATION_PENDING"}

    health = get_execution_health(mock_bridge, bar_age_seconds=15.0)
    assert not health.allows_new_entries
    assert health.status == HealthStatus.BROKEN

    now_ts = time.time()
    readiness = EntryReadiness(
        entry_allowed=health.allows_new_entries,
        suppression_reason=health.reason,
        cycle_id="test_cycle",
        bar_ts="2026-10-21T02:35:00+00:00",
        created_at=now_ts,
        expires_at=now_ts + 300.0,
    )

    res = execute_signal(
        sample_signal,
        bridge=mock_bridge,
        equity=10000.0,
        current_pnl=0.0,
        entry_readiness=readiness,
        dry_run=False,
    )

    assert res["status"] in ("REJECTED", "VETOED")
    assert mock_bridge.place_bracket_order.called is False


def test_stale_quote_blocks_order_submission(mock_bridge, sample_signal):
    """
    Stale real-time quote (> 10.0s age) MUST fail quote health veto and block execution.
    """
    stale_quote = {
        "bid": 150.00,
        "ask": 150.05,
        "timestamp": 1792000000.0,  # Far in the past
    }

    mock_bridge.get_quote = Mock(return_value=stale_quote)

    now_ts = time.time()
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="test_cycle",
        bar_ts="2026-10-21T02:35:00+00:00",
        created_at=now_ts,
        expires_at=now_ts + 300.0,
    )

    res = execute_signal(
        sample_signal,
        bridge=mock_bridge,
        equity=10000.0,
        current_pnl=0.0,
        entry_readiness=readiness,
        dry_run=False,
    )

    assert res["status"] in ("REJECTED", "VETOED")
    assert "STALE_QUOTE" in res["reason"]
    assert mock_bridge.place_bracket_order.called is False


def test_fault_injection_end_to_end_entry_route_fails_closed_then_recovers(mock_bridge, sample_signal):
    """
    Final Acceptance End-to-End Fault-Injection Lifecycle Test:
    1. FAULT INJECTION (Coverage 54.8%, 11 timeouts, stale data):
       Normal entry route MUST NOT place any order on broker (0 broker calls).
    2. RECOVERY (Coverage 96.8%, 0 timeouts, fresh data):
       Normal entry route unsuppresses EntryReadiness and successfully submits order.
    """
    # ── Phase 1: Fault Injection (Fail-Closed) ───────────────────────────────
    degraded_feed = {
        "coverage": 0.548,
        "consecutive_fetch_timeouts": 11,
        "last_batch_duration_s": 3.8,
        "stale_bar_skip_count": 3,
    }

    health_fail = get_execution_health(mock_bridge, bar_age_seconds=25.0, feed_telemetry=degraded_feed)
    assert not health_fail.allows_new_entries
    assert health_fail.status == HealthStatus.DEGRADED

    now_ts = time.time()
    readiness_fail = EntryReadiness(
        entry_allowed=health_fail.allows_new_entries,
        suppression_reason=health_fail.reason,
        cycle_id="cycle_fault_1",
        bar_ts="2026-10-21T02:39:02+00:00",
        created_at=now_ts,
        expires_at=now_ts + 300.0,
    )

    res_fail = execute_signal(
        sample_signal,
        bridge=mock_bridge,
        equity=10000.0,
        current_pnl=0.0,
        entry_readiness=readiness_fail,
        dry_run=False,
    )

    assert res_fail["status"] in ("REJECTED", "VETOED")
    assert "ENTRY_READINESS_SUPPRESSED" in res_fail["reason"]
    assert mock_bridge.place_bracket_order.called is False
    assert mock_bridge.place_bracket_order.call_count == 0

    # ── Phase 2: Recovery (Restored & Healthy) ────────────────────────────────
    recovered_feed = {
        "coverage": 0.968,  # 30/31 symbols
        "consecutive_fetch_timeouts": 0,
        "last_batch_duration_s": 0.45,
        "stale_bar_skip_count": 0,
    }

    health_rec = get_execution_health(mock_bridge, bar_age_seconds=12.0, feed_telemetry=recovered_feed)
    assert health_rec.allows_new_entries
    assert health_rec.status == HealthStatus.HEALTHY

    from datetime import datetime, timezone
    now_rec = time.time()
    # Bar started 320s ago, so it completed 20s ago (within the [0s, 90s] post-close grace window)
    current_bar_iso = datetime.fromtimestamp(now_rec - 320.0, timezone.utc).strftime("%Y-%m-%dT%H:%M:00+00:00")
    sample_signal.bar_timestamp = current_bar_iso
    readiness_rec = EntryReadiness(
        entry_allowed=health_rec.allows_new_entries,
        suppression_reason=None,
        cycle_id="cycle_rec_2",
        bar_ts=current_bar_iso,
        created_at=now_rec,
        expires_at=now_rec + 300.0,
    )

    fresh_quote = {
        "bid": 150.00,
        "ask": 150.05,
        "mid": 150.025,
        "timestamp": now_rec,
        "feed_identity": "iex",
    }
    mock_bridge.get_quote = Mock(return_value=fresh_quote)
    mock_order = {
        "id": "mock_order_recovery_777",
        "parent_order_id": "mock_order_recovery_777",
        "status": "new",
        "order_ids": ["mock_order_recovery_777", "stop_123", "tp_123"],
        "stop_order_id": "stop_123",
        "tp_order_id": "tp_123",
    }
    mock_bridge.place_bracket_order.return_value = mock_order

    res_rec = execute_signal(
        sample_signal,
        bridge=mock_bridge,
        equity=10000.0,
        current_pnl=0.0,
        entry_readiness=readiness_rec,
        dry_run=False,
    )

    # Order must be submitted!
    assert mock_bridge.place_bracket_order.called is True
    assert mock_bridge.place_bracket_order.call_count == 1
    assert res_rec["status"] in ("SUBMITTED", "PLACED", "ACCEPTED")
    assert mock_bridge.place_bracket_order.call_count == 1
    assert res_rec["status"] in ("SUBMITTED", "PLACED", "ACCEPTED")
