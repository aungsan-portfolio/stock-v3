"""
P1 Resilience, Reconnect State Machine & Broker Reconciliation Test Suite.

Verifies:
1. StreamState enum completeness & invariants
2. BrokerOrderCheck tri-state enum & fail-closed dedup handling
3. ShutdownResult contract & intentional shutdown reconnect prohibition
4. Single stream invariant & generation token guard against stale callbacks
5. Post-reconnect mandatory REST broker reconciliation before clearing entry halt
6. Recovery-specific halt clearance (preserves macro regime, manual, and protection latches)
"""

import os
import time
import threading
from unittest import mock
import pytest

from alpaca_bridge import AlpacaBridge, StreamState, BrokerOrderCheck, ShutdownResult
from observability import get_latch_state_machine, LatchTrigger


class TestP1StreamStateAndEnums:
    def test_stream_state_enum_completeness(self):
        expected_states = {
            "DISCONNECTED",
            "RECONNECTING",
            "AUTHENTICATED_UNRECONCILED",
            "RECONCILING",
            "HEALTHY",
            "DEGRADED",
            "LATCHED_DESYNC",
            "SHUTTING_DOWN",
            "STOPPED",
        }
        actual_states = {s.value for s in StreamState}
        assert expected_states.issubset(actual_states), f"Missing states: {expected_states - actual_states}"

    def test_broker_order_check_enum(self):
        expected_checks = {
            "WORKING_ORDER_EXISTS",
            "NO_WORKING_ORDER",
            "BROKER_QUERY_FAILED",
        }
        actual_checks = {c.value for c in BrokerOrderCheck}
        assert expected_checks == actual_checks

    def test_shutdown_result_contract(self):
        res_clean = ShutdownResult(
            stop_requested=True,
            reconnect_disabled=True,
            socket_closed=True,
            thread_joined=True,
            thread_alive=False,
        )
        assert res_clean.clean is True

        res_dirty = ShutdownResult(
            stop_requested=True,
            reconnect_disabled=True,
            socket_closed=True,
            thread_joined=False,
            thread_alive=True,
        )
        assert res_dirty.clean is False


class TestP1StreamLifecycleAndInvariants:
    def setUp(self):
        os.environ["APCA_API_KEY_ID"] = "fake_key"
        os.environ["APCA_API_SECRET_KEY"] = "fake_secret"
        os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"

    def test_stream_state_transitions_under_lock(self):
        bridge = AlpacaBridge(mode="swing")
        assert bridge.stream_state == StreamState.DISCONNECTED

        # Transition to RECONNECTING
        ok = bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.RECONNECTING,
            reason="Test start",
        )
        assert ok is True
        assert bridge.stream_state == StreamState.RECONNECTING

        # Invalid transition should fail
        fail = bridge._transition_stream_state(
            expected={StreamState.HEALTHY},
            new_state=StreamState.STOPPED,
            reason="Invalid jump",
        )
        assert fail is False
        assert bridge.stream_state == StreamState.RECONNECTING

    def test_reconnect_prohibited_during_shutdown_or_stopped(self):
        bridge = AlpacaBridge(mode="swing")
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.SHUTTING_DOWN,
            reason="Initiating shutdown",
        )
        assert bridge._is_reconnect_allowed() is False

        bridge._transition_stream_state(
            expected={StreamState.SHUTTING_DOWN},
            new_state=StreamState.STOPPED,
            reason="Shutdown complete",
        )
        assert bridge._is_reconnect_allowed() is False

    def test_generation_token_guards_against_stale_callbacks(self):
        bridge = AlpacaBridge(mode="swing")
        initial_gen = bridge.stream_generation

        # Fake an update with stale generation
        processed = bridge._dispatch_trade_update_with_generation(
            update=mock.MagicMock(order=mock.MagicMock(id="order_1", symbol="AAPL", status="filled")),
            generation=initial_gen - 1,
        )
        assert processed is False

        # Current generation should be accepted
        processed_valid = bridge._dispatch_trade_update_with_generation(
            update=mock.MagicMock(order=mock.MagicMock(id="order_2", symbol="AAPL", status="filled")),
            generation=initial_gen,
        )
        assert processed_valid is True

    def test_single_stream_invariant(self):
        bridge = AlpacaBridge(mode="swing")
        stop_evt = threading.Event()
        try:
            with mock.patch.object(bridge, "_run_trade_stream", side_effect=lambda s: stop_evt.wait(5.0)):
                fake_stream = mock.MagicMock()
                fake_stream._running = True

                with mock.patch("alpaca_bridge.TradingStream", return_value=fake_stream):
                    # Start stream 1
                    bridge._start_trade_stream("key", "secret")
                    assert bridge.active_stream_threads_count == 1

                    # 1. Calling again when ready should be idempotent and keep thread count <= 1
                    bridge._start_trade_stream("key", "secret")
                    assert bridge.active_stream_threads_count == 1

                    # 2. If startup is in progress (thread alive but not ready), it must raise RuntimeError
                    bridge._trade_stream_ready = False
                    with pytest.raises(RuntimeError) as exc:
                        bridge._start_trade_stream("key", "secret")
                    assert "already in progress" in str(exc.value)
        finally:
            stop_evt.set()
            bridge._stop_trade_stream()


class TestP1BrokerOrderCheckDedup:
    def test_check_working_order_tri_state(self):
        bridge = AlpacaBridge(mode="swing")
        bridge._connected = True

        # 1. No working order
        with mock.patch.object(bridge, "_get_active_orders", return_value=[]):
            res = bridge.check_working_order("AAPL", action="BUY")
            assert res == BrokerOrderCheck.NO_WORKING_ORDER

        # 2. Working order exists
        fake_order = mock.MagicMock(
            symbol="AAPL",
            status=mock.MagicMock(value="new"),
            side=mock.MagicMock(value="buy"),
        )
        with mock.patch.object(bridge, "_get_active_orders", return_value=[fake_order]):
            res = bridge.check_working_order("AAPL", action="BUY")
            assert res == BrokerOrderCheck.WORKING_ORDER_EXISTS

        # 3. Broker query failed
        with mock.patch.object(bridge, "_get_active_orders", side_effect=Exception("Network drop 10054")):
            res = bridge.check_working_order("AAPL", action="BUY")
            assert res == BrokerOrderCheck.BROKER_QUERY_FAILED


class TestP1PostReconnectReconciliation:
    def test_reconnect_requires_clean_reconciliation_before_healthy(self):
        bridge = AlpacaBridge(mode="swing")
        bridge._connected = True
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")

        # Transport reconnected -> state should be AUTHENTICATED_UNRECONCILED
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED, StreamState.RECONNECTING},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="WebSocket authenticated",
        )
        assert bridge.stream_state == StreamState.AUTHENTICATED_UNRECONCILED
        assert bridge.entry_gate.halted is True  # Gate must still be halted!

        # Run reconciliation with clean=True
        clean_snapshot = {
            "snapshot": {
                "clean": True,
                "open_positions": [{"symbol": "AMZN", "qty": 10}],
                "n_working_orders": 1,
                "unprotected_longs": [],
                "orphan_exits": [],
                "duplicate_refs": [],
            },
            "protect": {"failed": [], "unprotected": []},
            "clean": True,
        }
        with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_snapshot):
            rec_ok = bridge._run_post_reconnect_reconciliation()
            assert rec_ok is True
            assert bridge.stream_state == StreamState.HEALTHY
            assert bridge.entry_gate.halted is False  # Now entry halt cleared

    def test_reconnect_with_dirty_snapshot_latches_and_blocks_healthy(self):
        bridge = AlpacaBridge(mode="swing")
        bridge._connected = True
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")

        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED, StreamState.RECONNECTING},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="WebSocket authenticated",
        )

        dirty_snapshot = {
            "snapshot": {
                "clean": False,
                "open_positions": [{"symbol": "AMZN", "qty": 10}],
                "n_working_orders": 0,
                "unprotected_longs": ["AMZN"],
                "orphan_exits": [],
                "duplicate_refs": [],
            },
            "protect": {"failed": ["AMZN"], "unprotected": ["AMZN"]},
            "clean": False,
        }
        with mock.patch.object(bridge, "reconcile_startup_state", return_value=dirty_snapshot):
            rec_ok = bridge._run_post_reconnect_reconciliation()
            assert rec_ok is False
            assert bridge.stream_state == StreamState.LATCHED_DESYNC
            assert bridge.entry_gate.halted is True

    def test_recovery_only_clears_recovery_halts(self):
        bridge = AlpacaBridge(mode="swing")
        bridge._connected = True

        # Set multiple halts: 1 recovery-specific, 1 macro regime halt, 1 manual halt
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        bridge.entry_gate.halt("MARKET_REGIME_RISK_OFF")
        bridge.entry_gate.halt("MANUAL_OPERATOR_HALT")

        # Execute clean recovery
        bridge._clear_recovery_halts()

        # TRADE_STREAM_DISCONNECTED should be gone, but MARKET_REGIME_RISK_OFF and MANUAL_OPERATOR_HALT must remain!
        assert bridge.entry_gate.halted is True
        assert "TRADE_STREAM_DISCONNECTED" not in bridge.entry_gate.reasons
        assert "MARKET_REGIME_RISK_OFF" in bridge.entry_gate.reasons
        assert "MANUAL_OPERATOR_HALT" in bridge.entry_gate.reasons
