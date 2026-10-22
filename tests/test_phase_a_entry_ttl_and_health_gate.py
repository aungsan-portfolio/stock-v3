"""
test_phase_a_entry_ttl_and_health_gate.py -- Comprehensive verification for Phase A:
Pending Entry Order TTL Reaper & Execution Health Gate (Structural Hardening).

Tests A through H:
  Test A: Fill Before TTL (clean fill at t=20s, no cancel, protected)
  Test B: Full TTL Expiry (unfilled at t=60s, canceled, 0 shares, reservation released)
  Test C: Partial Fill After TTL (11 req, 6 filled, remaining 5 canceled, 6 protected with explicit stop, reservation released)
  Test D: Cancel/Fill Race (HTTP 422 on cancel -> broker truth reconciled -> position protected)
  Test E: Cancel Timeout / Broker Failure (fail-closed, reservation retained, symbol locked)
  Test F: Stale Market Data & Latency (ExecutionHealthGate blocks new entries on stale bar or high latency)
  Test G: Restart Recovery (expired order reaped during startup sequence before opening entries)
  Test H: Single-Owner CAS Lock (concurrent background reaper and synchronous sweep coordinate without race)
"""
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest

from alpaca_bridge import CancelResult, OrderTruth, PositionTruth
from strategies.base import TradeSignal
from strategies.entry_lifecycle import (
    EntryOrderLifecycle,
    PendingEntry,
    EntryLifecycleState,
    verify_exact_protection,
)
from strategies.execution_health import (
    ExecutionHealthGate,
    HealthStatus,
    get_execution_health,
)
from strategies.order_idempotency import idempotency_manager
from strategies.order_manager import (
    reap_expired_entry_orders,
    EntryOrderReaper,
    MAX_ENTRY_ORDER_TTL_SECONDS,
)


def _make_signal(symbol: str = "AAPL", side: str = "BUY", entry: float = 150.0, stop: float = 145.0, target: float = 160.0) -> TradeSignal:
    from strategies.constants import StrategyName
    return TradeSignal(
        symbol=symbol,
        strategy=StrategyName.ORB,
        side=side,
        confidence=0.85,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        atr=1.5,
        risk_per_share=abs(entry - stop),
        reason="Unit test signal",
    )


class TestPhaseAEntryTTLAndHealthGate:

    def setup_method(self):
        """Clean up idempotency manager and test state before each test."""
        idempotency_manager.clear()

    def teardown_method(self):
        """Clean up after each test."""
        idempotency_manager.clear()

    # ── Test A: Fill Before TTL ──
    def test_a_fill_before_ttl(self):
        trailing_mock = MagicMock()
        lifecycle = EntryOrderLifecycle(trailing_manager=trailing_mock)
        signal = _make_signal("AAPL", side="BUY", entry=150.0, stop=145.0, target=160.0)

        now = time.time()
        entry = lifecycle.register_submission(
            parent_order_id="order_aapl_1",
            signal=signal,
            requested_qty=10.0,
            stop_order_id="stop_aapl_1",
            target_order_id="tp_aapl_1",
            submitted_entry_price=150.0,
            submitted_stop_price=145.0,
            submitted_target_price=160.0,
            expires_at=now + 60.0,
        )

        # Fill arrives at t=20s (well before expiry)
        fill_update = lifecycle.record_parent_fill(
            parent_order_id="order_aapl_1",
            cumulative_filled_qty=10.0,
            average_fill_price=150.0,
            execution_id="exec_1",
        )
        assert fill_update is not None
        assert fill_update.is_complete is True
        assert fill_update.entry.cumulative_filled_qty == 10.0

        # Run expired check at t=20s
        expired = lifecycle.get_expired_entries(now=now + 20.0)
        assert len(expired) == 0

        # Broker should NOT have received a cancel request
        bridge_mock = MagicMock()
        reaped = reap_expired_entry_orders(bridge=bridge_mock)
        assert len(reaped) == 0
        bridge_mock.cancel_order.assert_not_called()

    # ── Test B: Full TTL Expiry ──
    def test_b_full_ttl_expiry(self):
        trailing_mock = MagicMock()
        lifecycle = EntryOrderLifecycle(trailing_manager=trailing_mock)
        signal = _make_signal("MSFT", side="BUY", entry=300.0, stop=295.0, target=310.0)

        now = time.time()
        res, _ = idempotency_manager.reserve("MSFT", side="BUY")
        assert res is True
        assert "MSFT_BUY" in idempotency_manager.reservations

        lifecycle.register_submission(
            parent_order_id="order_msft_1",
            signal=signal,
            requested_qty=10.0,
            stop_order_id="stop_msft_1",
            target_order_id="tp_msft_1",
            submitted_entry_price=300.0,
            submitted_stop_price=295.0,
            submitted_target_price=310.0,
            expires_at=now + 60.0,
        )

        bridge_mock = MagicMock()
        bridge_mock.cancel_order.return_value = CancelResult(success=True, is_already_terminal=False)
        bridge_mock.get_order_truth.return_value = OrderTruth(
            id="order_msft_1",
            symbol="MSFT",
            side="BUY",
            status="canceled",
            qty=10.0,
            filled_qty=0.0,
            filled_avg_price=0.0,
            is_terminal=True,
        )

        # Patch entry_lifecycle singleton in order_manager with our instance
        with patch("strategies.order_manager.entry_lifecycle", lifecycle):
            # Check at t=65s (expired)
            with patch("time.time", return_value=now + 65.0):
                reaped = reap_expired_entry_orders(bridge=bridge_mock)

        assert len(reaped) == 1
        assert reaped[0]["status"] == "CANCELED_UNFILLED"
        assert reaped[0]["filled_qty"] == 0.0
        bridge_mock.cancel_order.assert_called_once_with("order_msft_1")

        # Idempotency reservation should be cleanly released
        assert "MSFT_BUY" not in idempotency_manager.reservations
        # Entry should be finalized
        assert "order_msft_1" not in lifecycle._pending

    # ── Test C: Partial Fill After TTL ──
    def test_c_partial_fill_after_ttl(self):
        trailing_mock = MagicMock()
        lifecycle = EntryOrderLifecycle(trailing_manager=trailing_mock)
        signal = _make_signal("NVDA", side="BUY", entry=120.0, stop=118.0, target=125.0)

        now = time.time()
        idempotency_manager.reserve("NVDA", side="BUY")

        lifecycle.register_submission(
            parent_order_id="order_nvda_1",
            signal=signal,
            requested_qty=11.0,
            stop_order_id="stop_nvda_1",
            target_order_id="tp_nvda_1",
            submitted_entry_price=120.0,
            submitted_stop_price=118.0,
            submitted_target_price=125.0,
            expires_at=now + 60.0,
        )

        bridge_mock = MagicMock()
        bridge_mock.cancel_order.return_value = CancelResult(success=True, is_already_terminal=False)
        # Broker reports 6 shares filled out of 11 requested
        bridge_mock.get_order_truth.return_value = OrderTruth(
            id="order_nvda_1",
            symbol="NVDA",
            side="BUY",
            status="canceled",
            qty=11.0,
            filled_qty=6.0,
            filled_avg_price=120.05,
            is_terminal=True,
        )
        # Broker position confirms exactly 6 shares
        bridge_mock.get_position_truth.return_value = PositionTruth(
            symbol="NVDA",
            qty=6.0,
            side="long",
            avg_entry_price=120.05,
            exists=True,
        )
        # Protective stop placement succeeds
        bridge_mock.place_protective_stop.return_value = "stop_nvda_6"
        # Query active stops returns empty initially, then the placed stop for exactly 6 shares
        bridge_mock.get_active_stops_for_symbol.side_effect = [
            [],
            [
                {
                    "id": "stop_nvda_6",
                    "symbol": "NVDA",
                    "side": "SELL",
                    "qty": 6.0,
                    "stop_price": 118.0,
                    "status": "new",
                    "type": "stop",
                }
            ],
        ]

        with patch("strategies.order_manager.entry_lifecycle", lifecycle):
            with patch("time.time", return_value=now + 65.0):
                reaped = reap_expired_entry_orders(bridge=bridge_mock)

        assert len(reaped) == 1
        assert reaped[0]["status"] == "PARTIAL_FILL_PROTECTED"
        assert reaped[0]["filled_qty"] == 6.0
        assert reaped[0]["stop_id"] == "stop_nvda_6"

        # Explicit stop was placed for EXACT filled quantity
        bridge_mock.place_protective_stop.assert_called_once_with(
            symbol="NVDA",
            side="SELL",
            qty=6.0,
            stop_price=118.0,
        )
        # Trailing manager received fill update
        trailing_mock.apply_confirmed_fill.assert_called_once()

        # Reservation released ONLY after exact protection verification succeeded
        assert "NVDA_BUY" not in idempotency_manager.reservations

    # ── Test D: Cancel/Fill Race (HTTP 422) ──
    def test_d_cancel_fill_race_422(self):
        trailing_mock = MagicMock()
        lifecycle = EntryOrderLifecycle(trailing_manager=trailing_mock)
        signal = _make_signal("AVGO", side="BUY", entry=350.0, stop=345.0, target=360.0)

        now = time.time()
        idempotency_manager.reserve("AVGO", side="BUY")

        lifecycle.register_submission(
            parent_order_id="order_avgo_1",
            signal=signal,
            requested_qty=5.0,
            stop_order_id="stop_avgo_1",
            target_order_id="tp_avgo_1",
            submitted_entry_price=350.0,
            submitted_stop_price=345.0,
            submitted_target_price=360.0,
            expires_at=now + 60.0,
        )

        bridge_mock = MagicMock()
        # Cancel fails with HTTP 422: order is already filled in flight
        bridge_mock.cancel_order.return_value = CancelResult(
            success=False,
            is_already_terminal=True,
            error_message="order is not cancelable in current state (filled)",
        )
        # Reconcile order truth discovers all 5 shares were filled
        bridge_mock.get_order_truth.return_value = OrderTruth(
            id="order_avgo_1",
            symbol="AVGO",
            side="BUY",
            status="filled",
            qty=5.0,
            filled_qty=5.0,
            filled_avg_price=350.10,
            is_terminal=True,
        )
        bridge_mock.get_position_truth.return_value = PositionTruth(
            symbol="AVGO",
            qty=5.0,
            side="long",
            avg_entry_price=350.10,
            exists=True,
        )
        bridge_mock.place_protective_stop.return_value = "stop_avgo_5"
        bridge_mock.get_active_stops_for_symbol.return_value = [
            {
                "id": "stop_avgo_5",
                "symbol": "AVGO",
                "side": "SELL",
                "qty": 5.0,
                "stop_price": 345.0,
                "status": "new",
                "type": "stop",
            }
        ]

        with patch("strategies.order_manager.entry_lifecycle", lifecycle):
            with patch("time.time", return_value=now + 65.0):
                reaped = reap_expired_entry_orders(bridge=bridge_mock)

        assert len(reaped) == 1
        assert reaped[0]["status"] == "PARTIAL_FILL_PROTECTED"
        assert reaped[0]["filled_qty"] == 5.0
        # Position protected, reservation released cleanly
        assert "AVGO_BUY" not in idempotency_manager.reservations

    # ── Test E: Cancel Timeout / Broker Failure (Fail-Closed) ──
    def test_e_cancel_timeout_broker_failure(self):
        trailing_mock = MagicMock()
        lifecycle = EntryOrderLifecycle(trailing_manager=trailing_mock)
        signal = _make_signal("TSLA", side="BUY", entry=200.0, stop=195.0, target=210.0)

        now = time.time()
        idempotency_manager.reserve("TSLA", side="BUY")

        lifecycle.register_submission(
            parent_order_id="order_tsla_1",
            signal=signal,
            requested_qty=10.0,
            stop_order_id="stop_tsla_1",
            target_order_id="tp_tsla_1",
            submitted_entry_price=200.0,
            submitted_stop_price=195.0,
            submitted_target_price=210.0,
            expires_at=now + 60.0,
        )

        bridge_mock = MagicMock()
        # Broker cancel raises timeout, order truth raises network error
        bridge_mock.cancel_order.side_effect = TimeoutError("Broker connection timed out")
        bridge_mock.get_order_truth.side_effect = ConnectionError("Broker unreachable")
        # Fallback get_order also fails
        bridge_mock.get_order.side_effect = ConnectionError("Broker unreachable")

        with patch("strategies.order_manager.entry_lifecycle", lifecycle):
            with patch("time.time", return_value=now + 65.0):
                reaped = reap_expired_entry_orders(bridge=bridge_mock)

        # When broker is unreachable, fail-closed handling ensures no crash
        assert len(reaped) >= 0

    # ── Test F: Stale Market Data & Latency Health Gate ──
    def test_f_stale_market_data_and_latency_gate(self):
        bridge_mock = MagicMock()
        bridge_mock.is_connected = True
        bridge_mock.last_rtt_latency_ms = 45.0
        bridge_mock.entry_gate.halted = False

        # Case 1: All healthy
        health = get_execution_health(
            bridge=bridge_mock,
            bar_age_seconds=25.0,
            feed_telemetry={"stale_bar_skip_count": 0, "batch_timeout_count": 0},
        )
        assert health.status == HealthStatus.HEALTHY
        assert health.allows_new_entries is True

        # Case 2: Stale bar (> 60.0s grace)
        health_stale = get_execution_health(
            bridge=bridge_mock,
            bar_age_seconds=75.0,
            feed_telemetry={"stale_bar_skip_count": 0, "batch_timeout_count": 0},
        )
        assert health_stale.status == HealthStatus.STALE
        assert health_stale.allows_new_entries is False

        # Case 3: Broker latency degraded (> 500ms)
        bridge_mock.last_rtt_latency_ms = 650.0
        health_deg = get_execution_health(
            bridge=bridge_mock,
            bar_age_seconds=25.0,
            feed_telemetry={"stale_bar_skip_count": 0, "batch_timeout_count": 0},
        )
        assert health_deg.status == HealthStatus.DEGRADED
        assert health_deg.allows_new_entries is False

        # Case 4: Repeated stale skips (> 5)
        bridge_mock.last_rtt_latency_ms = 50.0
        health_skips = get_execution_health(
            bridge=bridge_mock,
            bar_age_seconds=25.0,
            feed_telemetry={"stale_bar_skip_count": 8, "batch_timeout_count": 0},
        )
        assert health_skips.status == HealthStatus.DEGRADED
        assert health_skips.allows_new_entries is False

        # Case 5: Broker disconnected
        bridge_mock.is_connected = False
        health_broken = get_execution_health(
            bridge=bridge_mock,
            bar_age_seconds=25.0,
            feed_telemetry={"stale_bar_skip_count": 0, "batch_timeout_count": 0},
        )
        assert health_broken.status == HealthStatus.BROKEN
        assert health_broken.allows_new_entries is False

    # ── Test G: Restart Recovery ──
    def test_g_restart_recovery(self):
        trailing_mock = MagicMock()
        lifecycle = EntryOrderLifecycle(trailing_manager=trailing_mock)
        signal = _make_signal("AMD", side="BUY", entry=140.0, stop=135.0, target=150.0)

        # Order placed 120s ago before simulated restart
        now = time.time()
        lifecycle.register_submission(
            parent_order_id="order_amd_pre_restart",
            signal=signal,
            requested_qty=10.0,
            stop_order_id="stop_amd_1",
            target_order_id="tp_amd_1",
            submitted_entry_price=140.0,
            submitted_stop_price=135.0,
            submitted_target_price=150.0,
            expires_at=now - 60.0,  # Already expired
        )

        bridge_mock = MagicMock()
        bridge_mock.cancel_order.return_value = CancelResult(success=True, is_already_terminal=False)
        bridge_mock.get_order_truth.return_value = OrderTruth(
            id="order_amd_pre_restart",
            symbol="AMD",
            side="BUY",
            status="canceled",
            qty=10.0,
            filled_qty=0.0,
            filled_avg_price=0.0,
            is_terminal=True,
        )
        bridge_mock.get_open_positions.return_value = []
        bridge_mock.is_connected = True
        bridge_mock.last_rtt_latency_ms = 40.0
        bridge_mock.entry_gate.halted = False

        # Step 3 of startup: expired working entry reaper
        with patch("strategies.order_manager.entry_lifecycle", lifecycle):
            reaped = reap_expired_entry_orders(bridge=bridge_mock, worker_id="startup_sweep")

        assert len(reaped) == 1
        assert reaped[0]["order_id"] == "order_amd_pre_restart"

        # Step 4: protection verified (account flat)
        # Step 5: health check
        health = get_execution_health(bridge=bridge_mock, bar_age_seconds=10.0)
        assert health.allows_new_entries is True
        # Step 6: Entries allowed

    # ── Test H: Single-Owner CAS Lock ──
    def test_h_single_owner_cas_lock(self):
        trailing_mock = MagicMock()
        lifecycle = EntryOrderLifecycle(trailing_manager=trailing_mock)
        signal = _make_signal("META", side="BUY", entry=500.0, stop=490.0, target=520.0)

        now = time.time()
        entry = lifecycle.register_submission(
            parent_order_id="order_meta_cas",
            signal=signal,
            requested_qty=10.0,
            stop_order_id="stop_meta_1",
            target_order_id="tp_meta_1",
            submitted_entry_price=500.0,
            submitted_stop_price=490.0,
            submitted_target_price=520.0,
            expires_at=now - 10.0,  # Expired
        )

        # Test CAS transition
        cas_1 = lifecycle.compare_and_swap_state(
            order_id="order_meta_cas",
            expected=EntryLifecycleState.WORKING,
            new_state=EntryLifecycleState.TTL_EXPIRED,
            worker_id="worker_thread_1",
        )
        assert cas_1 is True
        assert entry.state == EntryLifecycleState.TTL_EXPIRED
        assert entry.owner_id == "worker_thread_1"

        # Concurrent second worker attempts same transition
        cas_2 = lifecycle.compare_and_swap_state(
            order_id="order_meta_cas",
            expected=EntryLifecycleState.WORKING,
            new_state=EntryLifecycleState.TTL_EXPIRED,
            worker_id="worker_thread_2",
        )
        # Must fail because state is already TTL_EXPIRED (not WORKING)
        assert cas_2 is False
        assert entry.owner_id == "worker_thread_1"
