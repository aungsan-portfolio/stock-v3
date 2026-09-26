"""
tests/test_websocket_resilience_p0.py -- P0 Regression and Resilience Suite for Alpaca WebSocket Reconnect Lifecycle.

Forensic Context:
Discovered during the 2026-09-24 day-trading audit:
Alpaca SDK's `TradingStream._run_forever()` implements an internal reconnect loop that
swallows socket resets ([WinError 64] / ConnectionResetError). Because `stream.run()`
never exits, outer reconnect supervisors and finally blocks in AlpacaBridge are bypassed.
This module verifies that `_BridgeAwareTradingStream` bridges SDK internal reconnect
cycles into the AlpacaBridge lifecycle to guarantee:
  Transport Recovery != Trading-State Recovery.
Broker-state reconciliation is strictly required before new entries can resume.

Test Matrix:
- Test A: Silent SDK internal reconnect traversal (HEALTHY -> RECONNECTING -> AUTHENTICATED_UNRECONCILED -> RECONCILING -> HEALTHY)
- Test B: Active position and orders during drop (clean recovery vs. dirty fallback to LATCHED_DESYNC)
- Test C: Duplicate recovery ownership (single-owner reconciliation via _reconcile_lock)
- Test D: Strict recovery ordering (source-level guard: HEALTHY only reachable via RECONCILING)
"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from alpaca_bridge import (
    AlpacaBridge,
    StreamState,
    StateTransitionError,
    _BridgeAwareTradingStream,
)


class TestWebSocketResilienceP0:
    """Institutional P0 test suite for WebSocket SDK-internal reconnect lifecycle and broker reconciliation."""

    @pytest.fixture
    def bridge(self):
        b = AlpacaBridge(mode="daytrade")
        b.account_id = "PA3UODIC5JTA"
        b._connected = True
        return b

    def test_p0_test_a_silent_sdk_internal_reconnect_traversal(self, bridge):
        """
        Test A: Verify that an SDK-internal reconnect (re-invocation of _start_ws()
        inside the running stream thread) triggers fail-closed entry halting, moves
        through the mandated state lifecycle, executes broker reconciliation, and
        only restores HEALTHY once broker state is verified clean.
        """
        bridge._stream_state = StreamState.HEALTHY
        stream = _BridgeAwareTradingStream("test_key", "test_sec", paper=True, bridge=bridge)
        stream._bridge_ws_ever_connected = True
        bridge._trade_stream = stream

        state_history = []
        orig_transition = bridge._transition_stream_state

        def record_transition(expected, new_state, reason="", **kwargs):
            state_history.append(new_state)
            return orig_transition(expected, new_state, reason=reason, **kwargs)

        bridge._transition_stream_state = record_transition

        clean_recon = {
            "clean": True,
            "protect": {"failed": [], "unprotected": []},
            "snapshot": {"clean": True, "unprotected_longs": []},
        }

        # Mock the SDK base _start_ws to simulate transport connection and auth
        with patch.object(_BridgeAwareTradingStream.__bases__[0], "_start_ws", new_callable=AsyncMock), \
             patch.object(bridge, "reconcile_startup_state", return_value=clean_recon):

            # Simulate the SDK internal reconnect call
            asyncio.run(stream._start_ws())

        # Assert full ordered lifecycle traversal
        expected_sequence = [
            StreamState.RECONNECTING,
            StreamState.AUTHENTICATED_UNRECONCILED,
            StreamState.RECONCILING,
            StreamState.HEALTHY,
        ]
        assert state_history == expected_sequence, f"Unexpected sequence: {state_history}"
        assert bridge.stream_state == StreamState.HEALTHY
        assert bridge.is_stream_ready is True
        assert bridge._trade_stream_recovery_required is False

        # Entry gate recovery halts must be completely cleared
        assert bridge.entry_gate.halted is False
        assert "STREAM_RECOVERY_REQUIRED" not in bridge.entry_gate.reasons
        assert "TRADE_STREAM_DISCONNECTED" not in bridge.entry_gate.reasons
        assert "BROKER_RECONCILIATION_PENDING" not in bridge.entry_gate.reasons

    def test_p0_test_b_active_position_and_order_during_drop_clean_and_dirty(self, bridge):
        """
        Test B: Simulate active position and working orders during stream drop.
        - Verifies immediate entry gate fail-closed protection.
        - Part 1: Clean reconciliation verifies protective stops and releases to HEALTHY.
        - Part 2: Dirty reconciliation (e.g. missing protective stop or orphan fill)
          latches to LATCHED_DESYNC and keeps entry gate halted.
        """
        bridge._stream_state = StreamState.HEALTHY

        # 1. Simulate socket drop initiation
        bridge._on_sdk_stream_reconnect_start()
        assert bridge.stream_state == StreamState.RECONNECTING
        assert bridge.entry_gate.halted is True
        assert "STREAM_RECOVERY_REQUIRED" in bridge.entry_gate.reasons
        assert "TRADE_STREAM_DISCONNECTED" in bridge.entry_gate.reasons
        assert bridge._trade_stream_recovery_required is True

        # 2. Branch A: Reconnect completes with CLEAN broker truth
        clean_result = {
            "clean": True,
            "protect": {"failed": [], "unprotected": []},
            "snapshot": {"clean": True, "unprotected_longs": []},
        }
        with patch.object(bridge, "reconcile_startup_state", return_value=clean_result):
            success = bridge._on_sdk_stream_reconnect_success()
            assert success is True
            assert bridge.stream_state == StreamState.HEALTHY
            assert bridge.entry_gate.halted is False
            assert bridge._trade_stream_recovery_required is False

        # 3. Simulate second socket drop
        bridge._on_sdk_stream_reconnect_start()
        assert bridge.stream_state == StreamState.RECONNECTING
        assert bridge.entry_gate.halted is True

        # 4. Branch B: Reconnect completes with DIRTY broker truth (unprotected position / desync)
        dirty_result = {
            "clean": False,
            "protect": {"failed": ["NVDA"], "unprotected": ["NVDA"]},
            "snapshot": {"clean": False, "unprotected_longs": ["NVDA"]},
        }
        with patch.object(bridge, "reconcile_startup_state", return_value=dirty_result), \
             patch("observability.get_latch_state_machine") as mock_latch:
            mock_sm = MagicMock()
            mock_latch.return_value = mock_sm

            success = bridge._on_sdk_stream_reconnect_success()
            assert success is False
            assert bridge.stream_state == StreamState.LATCHED_DESYNC
            assert bridge.entry_gate.halted is True
            assert bridge._trade_stream_recovery_required is True
            mock_sm.latch.assert_called_once()

    def test_p0_test_c_duplicate_recovery_ownership_single_execution(self, bridge):
        """
        Test C: Verify single-owner reconciliation via _reconcile_lock.
        When simultaneous recovery triggers fire (e.g. SDK reconnect hook and watchdog
        or operator action concurrently), only one reconciliation is allowed to execute.
        The concurrent caller is rejected with already_in_progress, avoiding race conditions.
        """
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED

        reconcile_invocations = []
        recon_gate_event = threading.Event()
        recon_proceed_event = threading.Event()

        def synchronized_reconcile(*args, **kwargs):
            reconcile_invocations.append(time.monotonic())
            recon_gate_event.set()
            recon_proceed_event.wait(timeout=2.0)
            return {"clean": True, "protect": {"failed": [], "unprotected": []}}

        results = []

        with patch.object(bridge, "reconcile_startup_state", side_effect=synchronized_reconcile):
            t1 = threading.Thread(
                target=lambda: results.append(("t1", bridge._run_post_reconnect_reconciliation()))
            )
            t2 = threading.Thread(
                target=lambda: results.append(("t2", bridge._run_post_reconnect_reconciliation()))
            )

            t1.start()
            assert recon_gate_event.wait(timeout=1.0), "Thread 1 failed to acquire and enter reconciliation"

            # Thread 2 attempts concurrent reconciliation while Thread 1 holds _reconcile_lock
            t2.start()
            t2.join(timeout=1.0)

            # Let Thread 1 complete
            recon_proceed_event.set()
            t1.join(timeout=1.0)

        # Assert exactly one reconciliation executed
        assert len(reconcile_invocations) == 1, f"Expected 1 invocation, got {len(reconcile_invocations)}"
        results_dict = dict(results)
        assert results_dict["t1"] is True, "Thread 1 should have succeeded"
        assert results_dict["t2"] is False, "Thread 2 should have been rejected by _reconcile_lock"
        assert bridge.stream_state == StreamState.HEALTHY

    def test_p0_test_d_strict_recovery_ordering_rejects_illegal_jumps(self, bridge):
        """
        Test D: Source-level invariant enforcement.
        StreamState.HEALTHY can ONLY be entered from StreamState.RECONCILING.
        Direct transitions to HEALTHY from DISCONNECTED, RECONNECTING,
        AUTHENTICATED_UNRECONCILED, or DEGRADED must be rejected (returning False
        in standard mode, and raising StateTransitionError in strict mode).
        """
        illegal_source_states = [
            StreamState.DISCONNECTED,
            StreamState.RECONNECTING,
            StreamState.AUTHENTICATED_UNRECONCILED,
            StreamState.DEGRADED,
            StreamState.LATCHED_DESYNC,
            StreamState.STOPPED,
        ]

        for invalid_source in illegal_source_states:
            bridge._stream_state = invalid_source

            # Non-strict mode: must return False and refuse state mutation
            ok = bridge._transition_stream_state(
                expected={invalid_source},
                new_state=StreamState.HEALTHY,
                reason="Illegal direct jump attempt",
            )
            assert ok is False, f"Illegal jump to HEALTHY from {invalid_source} was erroneously accepted"
            assert bridge.stream_state == invalid_source, f"Stream state mutated to {bridge.stream_state}"

            # Strict mode: must raise StateTransitionError
            with pytest.raises(StateTransitionError, match="must be RECONCILING"):
                bridge._transition_stream_state(
                    expected={invalid_source},
                    new_state=StreamState.HEALTHY,
                    reason="Strict illegal direct jump attempt",
                    strict=True,
                )
            assert bridge.stream_state == invalid_source

        # Valid transition sequence: AUTHENTICATED_UNRECONCILED -> RECONCILING -> HEALTHY
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok_step1 = bridge._transition_stream_state(
            expected={StreamState.AUTHENTICATED_UNRECONCILED},
            new_state=StreamState.RECONCILING,
            reason="Legal step 1",
            strict=True,
        )
        assert ok_step1 is True
        assert bridge.stream_state == StreamState.RECONCILING

        ok_step2 = bridge._transition_stream_state(
            expected={StreamState.RECONCILING},
            new_state=StreamState.HEALTHY,
            reason="Legal step 2",
            strict=True,
        )
        assert ok_step2 is True
        assert bridge.stream_state == StreamState.HEALTHY
