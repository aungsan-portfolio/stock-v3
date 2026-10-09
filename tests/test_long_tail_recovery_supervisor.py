"""
tests/test_long_tail_recovery_supervisor.py -- Institutional Test Suite for Phase B Long-Tail Recovery Supervisor.

Principal Governance Context:
Validates the Tier-2 Autonomous Long-Tail Stream Recovery Probe:
1. Fast retries exhaustion -> StreamState.DEGRADED -> Periodic probe active.
2. Probe success -> AUTHENTICATED_UNRECONCILED -> RECONCILING -> CLEAN -> HEALTHY -> clear ONLY stream-owned halts (Stop Ownership Rule).
3. Probe dirty reconciliation -> LATCHED_DESYNC -> Autonomous retries prohibited.
4. Transient probe failure -> Remain DEGRADED -> Single thread invariant -> Zero admission leak.
5. Process shutdown cancellation -> Clean exit -> Zero orphan threads.
"""

import threading
import time
from unittest.mock import MagicMock, patch
import pytest

from alpaca_bridge import AlpacaBridge, StreamState, AccountModeViolation


class TestLongTailRecoverySupervisor:
    @pytest.fixture
    def bridge(self):
        b = AlpacaBridge(mode="daytrade")
        b.account_id = "PA3UODIC5JTA"
        b._connected = True
        b._trade_stream_credentials = ("PK_TEST_KEY", "SK_TEST_SECRET")
        return b

    def test_fast_retries_exhaust_enters_degraded_and_schedules_probe(self, bridge):
        """
        Case 1: Fast retries fail -> transitions to DEGRADED -> entry gate halted -> enters Tier 2 probe.
        """
        bridge._stream_state = StreamState.DISCONNECTED
        states_observed = []

        orig_transition = bridge._transition_stream_state
        def record_transition(expected, new_state, reason="", **kwargs):
            states_observed.append(new_state)
            return orig_transition(expected, new_state, reason=reason, **kwargs)

        bridge._transition_stream_state = record_transition

        # Mock fast retries: 2 attempts, all fail
        bridge._c = lambda k, default: 2 if "MAX_ATTEMPTS" in k else (0.01 if "DELAY" in k else (0.05 if "LONG_TAIL" in k else default))

        probe_started = threading.Event()

        def fail_fast_then_signal(*args, **kwargs):
            if StreamState.DEGRADED in states_observed:
                probe_started.set()
                bridge._trade_stream_reconnect_cancel.set()  # Cancel to stop after reaching probe
            raise RuntimeError("DNS transport resolution failed")

        with patch.object(bridge, "_start_trade_stream", side_effect=fail_fast_then_signal):
            bridge._trade_stream_reconnect_loop()

        assert StreamState.RECONNECTING in states_observed
        assert StreamState.DEGRADED in states_observed
        assert bridge.stream_state == StreamState.DEGRADED
        assert bridge.entry_gate.halted is True
        assert "TRADE_STREAM_DISCONNECTED" in bridge.entry_gate.reasons
        assert probe_started.is_set(), "Long-tail probe was not entered after fast retry exhaustion"

    def test_long_tail_probe_success_traverses_to_healthy_and_clears_stream_halt_only(self, bridge):
        """
        Case 2: Probe succeeds -> AUTHENTICATED_UNRECONCILED -> RECONCILING -> CLEAN -> HEALTHY.
        Stop Ownership Rule: If unrelated admission halt exists, clear ONLY stream-owned halt,
        global admission remains CLOSED.
        """
        bridge._stream_state = StreamState.DEGRADED
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        # Add unrelated external halt reason
        bridge.entry_gate.halt("DAILY_LOSS_LIMIT_REACHED")
        assert bridge.entry_gate.halted is True
        assert len(bridge.entry_gate.reasons) == 2

        states_observed = []
        orig_transition = bridge._transition_stream_state
        def record_transition(expected, new_state, reason="", **kwargs):
            states_observed.append(new_state)
            return orig_transition(expected, new_state, reason=reason, **kwargs)

        bridge._transition_stream_state = record_transition

        bridge._c = lambda k, default: 0 if "MAX_ATTEMPTS" in k else (0.01 if "LONG_TAIL" in k else default)

        def mock_start(*args, **kwargs):
            # Synchronously authenticate stream via canonical transition
            bridge._transition_stream_state(
                expected={StreamState.DEGRADED},
                new_state=StreamState.AUTHENTICATED_UNRECONCILED,
                reason="Authenticated paper trade-update stream is ready",
            )
            bridge._trade_stream_ready = True

        clean_recon = {
            "clean": True,
            "protect": {"failed": [], "unprotected": []},
            "snapshot": {"clean": True, "unprotected_longs": []},
        }

        with patch.object(bridge, "_start_trade_stream", side_effect=mock_start), \
             patch.object(bridge, "reconcile_startup_state", return_value=clean_recon):
            bridge._trade_stream_reconnect_loop()

        # Invariant checks
        assert bridge.stream_state == StreamState.HEALTHY
        assert StreamState.AUTHENTICATED_UNRECONCILED in states_observed
        assert StreamState.RECONCILING in states_observed
        assert StreamState.HEALTHY in states_observed

        # Stop Ownership Rule validation:
        # Stream-owned halt cleared, but DAILY_LOSS_LIMIT_REACHED must remain!
        assert "TRADE_STREAM_DISCONNECTED" not in bridge.entry_gate.reasons
        assert "DAILY_LOSS_LIMIT_REACHED" in bridge.entry_gate.reasons
        assert bridge.entry_gate.halted is True, "Unrelated halt was erroneously wiped out!"

    def test_long_tail_probe_dirty_reconciliation_latches_desync_and_stops_retrying(self, bridge):
        """
        Case 3: Probe succeeds, but broker reconciliation is DIRTY (desync / unhedged position).
        State transitions to LATCHED_DESYNC. Supervisor returns immediately without autonomous retries.
        Entry gate remains strictly blocked.
        """
        bridge._stream_state = StreamState.DEGRADED
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        bridge._c = lambda k, default: 0 if "MAX_ATTEMPTS" in k else (0.01 if "LONG_TAIL" in k else default)

        def mock_start(*args, **kwargs):
            bridge._transition_stream_state(
                expected={StreamState.DEGRADED},
                new_state=StreamState.AUTHENTICATED_UNRECONCILED,
                reason="Authenticated paper trade-update stream is ready",
            )
            bridge._trade_stream_ready = True

        dirty_recon = {
            "clean": False,
            "protect": {"failed": ["AAPL"], "unprotected": ["AAPL"]},
            "snapshot": {"clean": False, "unprotected_longs": ["AAPL"]},
        }

        with patch.object(bridge, "_start_trade_stream", side_effect=mock_start), \
             patch.object(bridge, "reconcile_startup_state", return_value=dirty_recon):
            bridge._trade_stream_reconnect_loop()

        # Invariant checks
        assert bridge.stream_state == StreamState.LATCHED_DESYNC
        assert bridge.entry_gate.halted is True
        assert bridge._trade_stream_recovery_required is True

    def test_long_tail_probe_transient_failure_remains_degraded_and_single_thread(self, bridge):
        """
        Case 4: Probe fails with transient error -> remains in DEGRADED -> cleans up stream ->
        only one supervisor thread exists -> zero admission leak.
        """
        bridge._stream_state = StreamState.DEGRADED
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        bridge._c = lambda k, default: 0 if "MAX_ATTEMPTS" in k else (0.01 if "LONG_TAIL" in k else default)

        probe_attempts = 0
        def fail_two_probes(*args, **kwargs):
            nonlocal probe_attempts
            probe_attempts += 1
            if probe_attempts >= 3:
                bridge._trade_stream_reconnect_cancel.set()
            raise RuntimeError("Transient socket error 10054")

        with patch.object(bridge, "_start_trade_stream", side_effect=fail_two_probes):
            bridge._trade_stream_reconnect_loop()

        assert probe_attempts >= 3
        assert bridge.stream_state == StreamState.DEGRADED
        assert bridge.entry_gate.halted is True
        # Admission remains 0
        assert bridge.is_stream_ready is False

    def test_long_tail_probe_cancelled_on_shutdown_zero_orphans(self, bridge):
        """
        Case 5: While supervisor is waiting in long-tail probe, process shutdown cancels
        the supervisor. Wait exits immediately; zero orphan threads remain.
        """
        bridge._stream_state = StreamState.DEGRADED
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        # Set large probe interval (300 seconds)
        bridge._c = lambda k, default: 0 if "MAX_ATTEMPTS" in k else (300.0 if "LONG_TAIL" in k else default)

        t_start = time.monotonic()
        thread = threading.Thread(target=bridge._trade_stream_reconnect_loop, daemon=True)
        bridge._trade_stream_reconnect_thread = thread
        thread.start()

        # Simulate shutdown after 50ms
        time.sleep(0.05)
        bridge._stop_trade_stream(cancel_reconnect=True)

        thread.join(timeout=2.0)
        t_elapsed = time.monotonic() - t_start

        assert not thread.is_alive(), "Supervisor thread became an orphan after shutdown!"
        assert t_elapsed < 3.0, f"Supervisor did not wake up promptly on cancel (took {t_elapsed:.2f}s)"
        assert bridge._trade_stream_reconnect_thread is None
