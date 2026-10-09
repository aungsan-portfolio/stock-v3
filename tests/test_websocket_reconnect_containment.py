"""
tests/test_websocket_reconnect_containment.py -- Institutional Test Suite for WebSocket Reconnect Containment.

Forensic / Governance Context:
Phase B Tier 2 Runtime Defect Hardening:
Contains upstream alpaca-py (>=0.20.0 through 0.43.5, Issue #740) TradingStream._run_forever()
10ms unthrottled spin loop during DNS / transport failure.
Enforces single reconnect ownership under Application Supervisor (AlpacaBridge._trade_stream_reconnect_loop)
and locks the canonical Phase B reconnect backoff contract (2s -> 4s -> 8s -> 16s -> 30s +- 20% jitter).

Test Matrix:
- Test A: DNS failure (getaddrinfo failed -> fail-close -> RECONNECTING)
- Test B: Bounded retry schedule (injected fake clock verifying 2/4/8/16/30s +- 20%, 10ms retry count == 0)
- Test C: Single reconnect ownership (single supervisor thread, single active stream/generation, single reconciliation owner)
- Test D: Immutable recovery traversal (RECONNECTING -> AUTHENTICATED_UNRECONCILED -> RECONCILING -> HEALTHY)
- Test E: Persistent failure remains fail-closed (while DNS unresolved, entry_gate.halted == True)
- Test F: Log/CPU storm boundedness (single warning per attempt, zero multi-line traceback flood)
- Test G1: Intentional stop (_should_run=False) exits cleanly with zero reconnect
- Test G2: Unexpected normal return from _consume() while _should_run=True treated as transport failure
"""

import asyncio
import logging
import socket
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


class TestWebSocketReconnectContainment:
    """Suite validating containment of SDK retry spin and enforcement of supervisor ownership."""

    @pytest.fixture
    def bridge(self):
        b = AlpacaBridge(mode="daytrade")
        b.account_id = "PA3UODIC5JTA"
        b._connected = True
        return b

    def test_test_a_dns_failure_fails_closed_immediately(self, bridge):
        """
        Test A: Verify that socket.gaierror (Errno 11001 getaddrinfo failed) during
        reconnection immediately fail-closes entry gate and moves to RECONNECTING.
        """
        bridge._stream_state = StreamState.HEALTHY
        stream = _BridgeAwareTradingStream("test_key", "test_sec", paper=True, bridge=bridge)
        stream._bridge_ws_ever_connected = True
        bridge._trade_stream = stream

        # Simulate socket.gaierror on connection
        dns_error = socket.gaierror(11001, "getaddrinfo failed")
        with patch.object(_BridgeAwareTradingStream.__bases__[0], "_start_ws", side_effect=dns_error):
            # Invoking _start_ws should fail-close entry gate before raising
            with pytest.raises(socket.gaierror):
                asyncio.run(stream._start_ws())

        # Assert fail-closed state
        assert bridge.stream_state == StreamState.RECONNECTING
        assert bridge.entry_gate.halted is True
        assert "STREAM_RECOVERY_REQUIRED" in bridge.entry_gate.reasons
        assert bridge._trade_stream_recovery_required is True
        assert bridge.is_stream_ready is False

    def test_test_b_bounded_retry_schedule_and_zero_10ms_spin(self, bridge):
        """
        Test B: Verify canonical Phase B backoff cadence using injected clock (fake scheduler):
        Attempt 1 -> 2s +- 20%
        Attempt 2 -> 4s +- 20%
        Attempt 3 -> 8s +- 20%
        Attempt 4 -> 16s +- 20%
        Attempt 5 -> 30s +- 20%
        Attempt 6 -> capped at 30s +- 20%
        Asserts 10ms retry count is strictly 0.
        """
        recorded_delays = []
        recorded_attempts = []

        def fake_wait(delay):
            recorded_delays.append(delay)
            if len(recorded_delays) >= 6:
                bridge._trade_stream_reconnect_cancel.set()
            # Return False so wait completes without cancellation
            return False

        bridge._trade_stream_credentials = ("key", "sec")
        bridge._stream_state = StreamState.DISCONNECTED

        expected_bases = [2.0, 4.0, 8.0, 16.0, 30.0, 30.0]

        # Patch _start_trade_stream to fail with DNS error for 6 attempts
        attempt_counter = [0]
        def failing_start(*args, **kwargs):
            attempt_counter[0] += 1
            recorded_attempts.append(attempt_counter[0])
            raise socket.gaierror(11001, "getaddrinfo failed")

        with patch.object(bridge._trade_stream_reconnect_cancel, "wait", side_effect=fake_wait), \
             patch.object(bridge, "_start_trade_stream", side_effect=failing_start), \
             patch.object(bridge, "_c", side_effect=lambda k, d: 6 if k == "TRADE_STREAM_RECONNECT_MAX_ATTEMPTS" else d):

            # Run supervisor reconnect loop directly
            bridge._trade_stream_reconnect_loop()

        # Verify all 6 attempts occurred
        assert len(recorded_delays) == 6
        jitter_ratio = 0.20

        for i, (actual_delay, base) in enumerate(zip(recorded_delays, expected_bases), start=1):
            min_bound = base * (1.0 - jitter_ratio)
            max_bound = base * (1.0 + jitter_ratio)
            assert min_bound <= actual_delay <= max_bound, (
                f"Attempt {i}: expected base {base} in [{min_bound:.2f}, {max_bound:.2f}], got {actual_delay:.2f}"
            )

        # Invariant: Zero 10ms retries occurred
        assert all(d >= 1.6 for d in recorded_delays), "Detected unthrottled millisecond retry!"

    def test_test_c_single_reconnect_ownership_and_stream_identity(self, bridge):
        """
        Test C: Single Reconnect Ownership Invariant:
        1. At most ONE active reconnect supervisor thread.
        2. At most ONE stream instance designated as active with matching generation.
        3. At most ONE post-reconnect reconciliation owner via _reconcile_lock.
        """
        bridge._trade_stream_credentials = ("key", "sec")
        bridge._connected = True

        # 1. Single Reconnect Supervisor Thread verification
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = True
        bridge._trade_stream_reconnect_thread = fake_thread

        with patch("threading.Thread") as mock_thread_cls:
            bridge._schedule_trade_stream_reconnect()
            # Must NOT spawn a duplicate supervisor thread
            mock_thread_cls.assert_not_called()

        bridge._trade_stream_reconnect_thread = None

        # 2. Single Active Stream Instance and Generation Matching
        bridge._stream_generation = 5
        stream_gen5 = _BridgeAwareTradingStream("k", "s", paper=True, bridge=bridge)
        bridge._trade_stream = stream_gen5

        # Callback from stale generation (gen 4) must be rejected
        stale_update = MagicMock()
        stale_update.order = MagicMock(id="order_1")
        accepted = bridge._dispatch_trade_update_with_generation(stale_update, generation=4)
        assert accepted is False, "Stale stream generation callback was erroneously processed!"

        # Callback from current generation (gen 5) accepted
        accepted_current = bridge._dispatch_trade_update_with_generation(stale_update, generation=5)
        assert accepted_current is True

        # 3. Single Reconciliation Owner via _reconcile_lock
        recon_calls = []
        def mocked_recon(*args, **kwargs):
            recon_calls.append(1)
            return {"clean": True, "protect": {"failed": [], "unprotected": []}}

        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        with patch.object(bridge, "reconcile_startup_state", side_effect=mocked_recon):
            res1 = bridge._run_post_reconnect_reconciliation()
            assert res1 is True
            assert len(recon_calls) == 1

    def test_test_d_recovery_ordering_traversal(self, bridge):
        """
        Test D: Recovery Ordering Verification:
        RECONNECTING -> AUTHENTICATED_UNRECONCILED -> RECONCILING -> Broker Truth -> HEALTHY -> Entry Release.
        """
        bridge._stream_state = StreamState.RECONNECTING
        state_history = []
        orig_transition = bridge._transition_stream_state

        def record_transition(expected, new_state, reason="", **kwargs):
            state_history.append(new_state)
            return orig_transition(expected, new_state, reason=reason, **kwargs)

        bridge._transition_stream_state = record_transition

        # Transition 1: WebSocket authenticated
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED, StreamState.RECONNECTING, StreamState.STOPPED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated stream ready",
        )

        clean_recon = {
            "clean": True,
            "protect": {"failed": [], "unprotected": []},
            "snapshot": {"clean": True, "unprotected_longs": []},
        }

        # Transition 2 & 3: Reconciliation check
        with patch.object(bridge, "reconcile_startup_state", return_value=clean_recon):
            recon_ok = bridge._run_post_reconnect_reconciliation()
            assert recon_ok is True

        expected_sequence = [
            StreamState.AUTHENTICATED_UNRECONCILED,
            StreamState.RECONCILING,
            StreamState.HEALTHY,
        ]
        assert state_history == expected_sequence
        assert bridge.stream_state == StreamState.HEALTHY
        assert bridge.entry_gate.halted is False
        assert bridge.is_stream_ready is True

    def test_test_e_failure_remains_fail_closed_until_recovery(self, bridge):
        """
        Test E: Failure Remains Fail-Closed Invariant:
        While DNS is unresolved or transport is failed, entry_gate.halted remains True,
        and place_bracket_order is completely blocked (raising RuntimeError).
        """
        bridge._client = MagicMock()
        bridge._stream_state = StreamState.RECONNECTING
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        bridge._trade_stream_ready = False

        assert bridge.entry_gate.halted is True
        assert bridge.is_stream_ready is False
        assert "TRADE_STREAM_DISCONNECTED" in bridge.entry_gate.reasons

        # Attempting place_bracket_order must fail closed with RuntimeError
        with pytest.raises(RuntimeError, match="not ready/healthy|New entries blocked"):
            bridge.place_bracket_order(
                symbol="AAPL",
                side="buy",
                qty=1,
                entry_price=150.0,
                target_price=155.0,
                stop_price=148.0,
            )

        assert bridge.entry_gate.halted is True

    def test_test_f_log_and_cpu_storm_boundedness(self, bridge, caplog):
        """
        Test F: Log / CPU Storm Boundedness:
        Verifies that when transport exceptions occur, _run_forever logs a single concise warning
        without dumping full multi-line stack trace storms.
        """
        stream = _BridgeAwareTradingStream("k", "s", paper=True, bridge=bridge)
        stream._trade_updates_handler = lambda upd: None

        dns_error = socket.gaierror(11001, "getaddrinfo failed")

        with patch.object(stream, "_start_ws", side_effect=dns_error), \
             caplog.at_level(logging.WARNING):

            # Run the single iteration of _run_forever
            asyncio.run(stream._run_forever())

        # Verify: Exactly one warning logged, loop terminated cleanly
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1
        assert "Stream transport failure" in warning_records[0].message
        assert "11001" in warning_records[0].message
        assert stream._running is False

    def test_test_g1_intentional_shutdown_exits_cleanly_without_reconnect(self, bridge):
        """
        Test G1 (Mandatory Correction #1): Intentional stop (_should_run=False)
        must exit cleanly and NEVER schedule supervisor reconnection.
        """
        stream = _BridgeAwareTradingStream("k", "s", paper=True, bridge=bridge)
        stream._trade_updates_handler = lambda upd: None
        stream._should_run = False  # Pre-signaled intentional shutdown

        with patch.object(bridge, "_schedule_trade_stream_reconnect") as mock_sched:
            asyncio.run(stream._run_forever())
            mock_sched.assert_not_called()

        assert stream._running is False

    def test_test_g2_consume_unexpected_normal_return_triggers_reconnect(self, bridge):
        """
        Test G2 (Mandatory Correction #2): If _consume() returns normally while _should_run=True,
        it must be detected as an unexpected termination and yield to supervisor reconnection.
        """
        stream = _BridgeAwareTradingStream("k", "s", paper=True, bridge=bridge)
        stream._trade_updates_handler = lambda upd: None

        with patch.object(stream, "_start_ws", new_callable=AsyncMock), \
             patch.object(stream, "_consume", new_callable=AsyncMock):

            # In this test, _consume returns normally while _should_run is True
            asyncio.run(stream._run_forever())

        # Stream event loop must be cleanly terminated so supervisor can take over
        assert stream._running is False
        assert stream._should_run is True  # Remained True, indicating unexpected termination
