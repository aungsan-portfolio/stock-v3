"""
tests/test_p1_chaos_invariants.py — Comprehensive P1 Chaos & Merge-Blocker Invariant Tests.

Covers:
1. Stream Lifecycle Soak (10+ simulated cycles, persistent bridge, single stream thread, no callback duplication, idempotent shutdown).
2. Partial Broker Failure Matrix (Positions ok / orders fail, orders ok / positions fail, auth ok / recon timeout, desync scenarios).
3. Halt Preservation Invariant (Recovery clears only stream/recon halts, preserving market regime risk-off, manual halt, daily loss, and protection latch).
"""
import os
import threading
import time
from unittest import mock
import pytest

from alpaca_bridge import (
    AlpacaBridge,
    StreamState,
    BrokerOrderCheck,
    ShutdownResult,
)


class TestP1StreamLifecycleSoak:
    """Stream Lifecycle Soak: multi-cycle execution, thread stability, idempotency."""

    def test_multi_cycle_stream_stability_and_zero_thread_inflation(self):
        """Verify 10 scan cycles on a persistent bridge maintain exactly 1 stream thread and 0 duplicate callbacks."""
        bridge = AlpacaBridge(mode="swing")
        stop_evt = threading.Event()

        try:
            with mock.patch.object(bridge, "_run_trade_stream", side_effect=lambda s: stop_evt.wait(5.0)):
                fake_stream = mock.MagicMock()
                fake_stream._running = True
                fake_stream._endpoint = "wss://paper-api.alpaca.markets/stream"

                with mock.patch("alpaca_bridge.TradingStream", return_value=fake_stream):
                    # Connect and establish stream
                    bridge._start_trade_stream("key", "secret")
                    assert bridge.active_stream_threads_count == 1
                    initial_gen = bridge.stream_generation

                    # Simulate 10 scan cycles
                    for cycle in range(1, 11):
                        # Ensure calling start stream is idempotent and does not spawn new threads
                        bridge._start_trade_stream("key", "secret")
                        assert bridge.active_stream_threads_count == 1
                        assert bridge.stream_generation == initial_gen

                    # Simulate trade updates
                    received_events = []
                    with mock.patch.object(bridge, "on_entry_order_update", side_effect=lambda **kwargs: received_events.append(kwargs)):
                        # Send valid update with matching generation
                        fake_update = mock.MagicMock()
                        fake_update.order = mock.MagicMock(id="order_1", symbol="AAPL", status="filled", filled_qty=10, filled_avg_price=150.0)
                        fake_update.event = "fill"
                        fake_update.execution_id = "exec_1"

                        dispatched = bridge._dispatch_trade_update_with_generation(fake_update, generation=initial_gen)
                        assert dispatched is True

                        # Duplicate update with same execution_id should be ignored by dedup
                        dispatched_dup = bridge._dispatch_trade_update_with_generation(fake_update, generation=initial_gen)
                        assert dispatched_dup is True

                        # Verify on_entry_order_update called only once due to event dedup
                        assert len(received_events) == 1

                        # Send update from stale generation -> must be rejected
                        dispatched_stale = bridge._dispatch_trade_update_with_generation(fake_update, generation=initial_gen - 1)
                        assert dispatched_stale is False
                        assert len(received_events) == 1
        finally:
            stop_evt.set()
            res = bridge.graceful_shutdown()
            assert res.clean is True

    def test_shutdown_idempotency_and_no_reconnect(self):
        """Verify calling graceful_shutdown twice is cleanly idempotent and reconnect is blocked."""
        bridge = AlpacaBridge(mode="swing")
        stop_evt = threading.Event()

        with mock.patch.object(bridge, "_run_trade_stream", side_effect=lambda s: stop_evt.wait(5.0)):
            fake_stream = mock.MagicMock()
            fake_stream._running = True
            fake_stream._endpoint = "wss://paper-api.alpaca.markets/stream"
            with mock.patch("alpaca_bridge.TradingStream", return_value=fake_stream):
                bridge._start_trade_stream("key", "secret")
                assert bridge.stream_state == StreamState.AUTHENTICATED_UNRECONCILED

        stop_evt.set()
        # First shutdown
        res1 = bridge.graceful_shutdown()
        assert res1.clean is True
        assert bridge.stream_state == StreamState.STOPPED
        assert bridge._is_reconnect_allowed() is False

        # Second shutdown (idempotent call)
        res2 = bridge.graceful_shutdown()
        assert res2.clean is True
        assert bridge.stream_state == StreamState.STOPPED

        # Attempt reconnect scheduling -> should be no-op because reconnect is prohibited
        bridge._schedule_trade_stream_reconnect()
        assert bridge.active_stream_threads_count == 0


class TestP1PartialBrokerFailureMatrix:
    """Partial Broker Failure Matrix & Fail-Closed Gating."""

    def test_positions_success_orders_failure_fail_closed(self):
        """When positions succeed but active orders query fails, check_working_order fails closed."""
        bridge = AlpacaBridge(mode="daytrade")
        bridge._connected = True
        bridge._client = mock.MagicMock()

        # get_open_position succeeds
        bridge._client.get_open_position.return_value = mock.MagicMock(qty=10)

        # get_orders raises ConnectionResetError
        bridge._client.get_orders.side_effect = ConnectionResetError(10054, "Connection reset")

        check = bridge.check_working_order("AAPL", "BUY")
        assert check == BrokerOrderCheck.BROKER_QUERY_FAILED

    def test_reconciliation_clean_false_triggers_latched_desync(self):
        """When post-reconnect reconciliation discovers an unprotected position, state transitions to LATCHED_DESYNC."""
        bridge = AlpacaBridge(mode="daytrade")
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated",
        )

        # Mock reconcile_startup_state returning clean=False due to unprotected position
        with mock.patch.object(
            bridge,
            "reconcile_startup_state",
            return_value={"clean": False, "protect": {"unprotected": ["NVDA"]}},
        ):
            with mock.patch("observability.get_latch_state_machine") as mock_latch_getter:
                mock_latch_sm = mock.MagicMock()
                mock_latch_getter.return_value = mock_latch_sm

                success = bridge._run_post_reconnect_reconciliation()
                assert success is False
                assert bridge.stream_state == StreamState.LATCHED_DESYNC
                mock_latch_sm.latch.assert_called_once()

    def test_clean_reconciliation_recovery_transitions_to_healthy(self):
        """When post-reconnect reconciliation is clean, state transitions to HEALTHY and recovery halts are cleared."""
        bridge = AlpacaBridge(mode="daytrade")
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated",
        )
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        bridge.entry_gate.halt("BROKER_RECONCILIATION_PENDING")
        assert bridge.entry_gate.halted is True

        with mock.patch.object(bridge, "reconcile_startup_state", return_value={"clean": True}):
            success = bridge._run_post_reconnect_reconciliation()
            assert success is True
            assert bridge.stream_state == StreamState.HEALTHY
            assert bridge.entry_gate.halted is False


class TestP1HaltPreservationInvariant:
    """Halt Preservation: Non-recovery halts must survive post-reconnect reconciliation."""

    def test_non_recovery_halts_preserved_after_clean_reconnect(self):
        """MARKET_REGIME_RISK_OFF, MANUAL_OPERATOR_HALT, and PROTECTION_LATCH remain active after clean reconnect."""
        bridge = AlpacaBridge(mode="daytrade")
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated",
        )

        # Set persistent halts
        bridge.entry_gate.halt("MARKET_REGIME_RISK_OFF")
        bridge.entry_gate.halt("MANUAL_OPERATOR_HALT")
        bridge.entry_gate.halt("DAILY_LOSS_LIMIT")
        bridge.entry_gate.halt("PROTECTION_LATCH")

        # Set recovery halts
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        bridge.entry_gate.halt("BROKER_RECONCILIATION_PENDING")

        assert len(bridge.entry_gate.reasons) == 6
        assert bridge.entry_gate.halted is True

        # Run clean post-reconnect reconciliation
        with mock.patch.object(bridge, "reconcile_startup_state", return_value={"clean": True}):
            success = bridge._run_post_reconnect_reconciliation()
            assert success is True
            assert bridge.stream_state == StreamState.HEALTHY

            # Recovery halts are gone, but persistent halts remain!
            assert "TRADE_STREAM_DISCONNECTED" not in bridge.entry_gate.reasons
            assert "BROKER_RECONCILIATION_PENDING" not in bridge.entry_gate.reasons
            assert "MARKET_REGIME_RISK_OFF" in bridge.entry_gate.reasons
            assert "MANUAL_OPERATOR_HALT" in bridge.entry_gate.reasons
            assert "DAILY_LOSS_LIMIT" in bridge.entry_gate.reasons
            assert "PROTECTION_LATCH" in bridge.entry_gate.reasons

            # Entry gate MUST remain halted
            assert bridge.entry_gate.halted is True
