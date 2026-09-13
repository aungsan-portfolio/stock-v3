"""
tests/test_websocket_resilience.py -- Unit tests for Alpaca WebSocket resilience, endpoint guards, and reconciliation lifecycle.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from urllib.parse import urlparse

from alpaca_bridge import (
    AlpacaBridge,
    AccountModeViolation,
    StreamState,
)
import config


class TestWebSocketResilience:
    """Test suite verifying WebSocket parameter passing, paper-only endpoint guards, and reconnect lifecycle."""

    @pytest.fixture
    def bridge(self):
        b = AlpacaBridge(mode="daytrade")
        b.account_id = "PA3UODIC5JTA"
        b._connected = True
        return b

    def test_websocket_params_passed_to_trading_stream(self, bridge):
        with patch("alpaca_bridge.TradingStream") as mock_stream_cls, \
             patch.object(bridge, "_assert_paper_stream_endpoint"), \
             patch("threading.Thread") as mock_thread, \
             patch("time.sleep"):

            mock_stream = MagicMock()
            mock_stream._running = True
            mock_stream_cls.return_value = mock_stream
            mock_thread_inst = MagicMock()
            mock_thread_inst.is_alive.return_value = True
            mock_thread.return_value = mock_thread_inst

            bridge._start_trade_stream("test_key", "test_sec")

            mock_stream_cls.assert_called_once()
            _, kwargs = mock_stream_cls.call_args
            assert kwargs.get("paper") is True
            ws_params = kwargs.get("websocket_params")
            assert ws_params is not None
            assert ws_params["ping_interval"] == 20
            assert ws_params["ping_timeout"] == 10
            assert ws_params["max_queue"] == 1024

    def test_paper_endpoint_verification_accepts_valid_paper_url(self, bridge):
        stream = MagicMock()
        stream._endpoint = "wss://paper-api.alpaca.markets/stream"
        # Must not raise
        bridge._assert_paper_stream_endpoint(stream)

    def test_live_endpoint_triggers_account_mode_violation(self, bridge):
        stream = MagicMock()
        stream._endpoint = "wss://api.alpaca.markets/stream"
        with pytest.raises(AccountModeViolation, match="TradeStream endpoint safety violation"):
            bridge._assert_paper_stream_endpoint(stream)

    def test_wrong_scheme_fails_closed(self, bridge):
        stream = MagicMock()
        stream._endpoint = "ws://paper-api.alpaca.markets/stream"
        with pytest.raises(AccountModeViolation, match="TradeStream endpoint safety violation"):
            bridge._assert_paper_stream_endpoint(stream)

    def test_lookalike_hostname_fails_closed(self, bridge):
        stream = MagicMock()
        stream._endpoint = "wss://paper-api.alpaca.markets.attacker.com/stream"
        with pytest.raises(AccountModeViolation, match="TradeStream endpoint safety violation"):
            bridge._assert_paper_stream_endpoint(stream)

    def test_missing_endpoint_triggers_account_mode_violation(self, bridge):
        stream = object()  # Has no _endpoint
        with pytest.raises(AccountModeViolation, match="missing '_endpoint' attribute"):
            bridge._assert_paper_stream_endpoint(stream)

    def test_reconnect_loop_fails_closed_on_account_mode_violation(self, bridge):
        bridge._trade_stream_credentials = ("key", "secret")
        with patch.object(bridge, "_start_trade_stream", side_effect=AccountModeViolation("Fatal Live Endpoint")), \
             patch("alpaca_bridge.logger.critical") as mock_crit:

            with pytest.raises(AccountModeViolation, match="Fatal Live Endpoint"):
                bridge._trade_stream_reconnect_loop()

            # Must log critical and never retry
            assert mock_crit.call_count == 1
            assert bridge.stream_state != StreamState.HEALTHY

    def test_reconnect_sequence_post_reconciliation_ordering(self, bridge):
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated",
        )

        with patch.object(bridge, "reconcile_startup_state", return_value={"clean": True}), \
             patch.object(bridge, "_clear_recovery_halts") as mock_clear:

            success = bridge._run_post_reconnect_reconciliation()

            assert success is True
            assert bridge.stream_state == StreamState.HEALTHY
            mock_clear.assert_called_once()

    def test_reconnect_dirty_reconciliation_latches_desync(self, bridge):
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated",
        )

        with patch.object(bridge, "reconcile_startup_state", return_value={"clean": False, "snapshot": {}}), \
             patch("observability.get_latch_state_machine") as mock_latch:

            success = bridge._run_post_reconnect_reconciliation()

            assert success is False
            assert bridge.stream_state == StreamState.LATCHED_DESYNC
            mock_latch.return_value.latch.assert_called_once()

    def test_reconnect_reconciliation_crashed_transitions_to_degraded(self, bridge):
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated",
        )

        with patch.object(bridge, "reconcile_startup_state", side_effect=RuntimeError("Broker position query dropped")):
            with pytest.raises(RuntimeError, match="Broker position query dropped"):
                bridge._run_post_reconnect_reconciliation()

            assert bridge.stream_state == StreamState.DEGRADED

    def test_reconnect_reconciliation_respects_global_lock_order(self, bridge):
        """
        Verifies global lock order invariant:
        1. _reconcile_lock
        2. get_journal_lock()
        Never acquire _reconcile_lock while already holding the journal lock.
        """
        order_of_locks = []

        class LockTracker:
            def __init__(self, name, lock):
                self.name = name
                self.lock = lock
            def acquire(self, *args, **kwargs):
                order_of_locks.append(f"ACQUIRE_{self.name}")
                return self.lock.acquire(*args, **kwargs)
            def release(self, *args, **kwargs):
                order_of_locks.append(f"RELEASE_{self.name}")
                return self.lock.release(*args, **kwargs)
            def __enter__(self):
                order_of_locks.append(f"ACQUIRE_{self.name}")
                return self.lock.__enter__()
            def __exit__(self, *args):
                order_of_locks.append(f"RELEASE_{self.name}")
                return self.lock.__exit__(*args)

        orig_reconcile_lock = bridge._reconcile_lock
        tracked_reconcile = LockTracker("_reconcile_lock", orig_reconcile_lock)

        from strategies.trade_journal import get_journal_lock
        orig_journal_lock = get_journal_lock()
        tracked_journal = LockTracker("journal_lock", orig_journal_lock)

        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Test",
        )

        def mock_reconcile():
            # During reconciliation, journal lock is acquired
            with tracked_journal:
                pass
            return {"clean": True}

        with patch.object(bridge, "_reconcile_lock", tracked_reconcile), \
             patch.object(bridge, "reconcile_startup_state", side_effect=mock_reconcile):

            bridge._run_post_reconnect_reconciliation()

            # Verify order: ACQUIRE__reconcile_lock before ACQUIRE_journal_lock
            assert "ACQUIRE__reconcile_lock" in order_of_locks
            assert "ACQUIRE_journal_lock" in order_of_locks
            assert order_of_locks.index("ACQUIRE__reconcile_lock") < order_of_locks.index("ACQUIRE_journal_lock")

    def test_stale_stream_generation_cannot_override_current_state(self, bridge):
        bridge._stream_generation = 2
        bridge._stream_state = StreamState.HEALTHY

        # Attempt to set state from stale generation 1
        applied = bridge._set_stream_state_for_generation(
            generation=1,
            state=StreamState.DISCONNECTED,
            reason="Old thread exit",
        )

        assert applied is False
        assert bridge.stream_state == StreamState.HEALTHY

        # Attempt to set state from current generation 2
        applied_cur = bridge._set_stream_state_for_generation(
            generation=2,
            state=StreamState.DEGRADED,
            reason="Current thread degrade",
        )
        assert applied_cur is True
        assert bridge.stream_state == StreamState.DEGRADED

    def test_no_credentials_appear_in_failure_logs(self, bridge):
        bridge._trade_stream_credentials = ("SECRET_API_KEY_12345", "VERY_SECRET_SECRET_KEY_67890")
        with patch.object(bridge, "_start_trade_stream", side_effect=RuntimeError("Connection refused")), \
             patch.object(bridge._trade_stream_reconnect_cancel, "wait", return_value=False), \
             patch("alpaca_bridge.logger.warning") as mock_warn, \
             patch("alpaca_bridge.logger.error") as mock_err:

            bridge._trade_stream_reconnect_loop()

            logged_text = " ".join(str(c) for c in mock_warn.call_args_list + mock_err.call_args_list)
            assert "SECRET_API_KEY_12345" not in logged_text
            assert "VERY_SECRET_SECRET_KEY_67890" not in logged_text
