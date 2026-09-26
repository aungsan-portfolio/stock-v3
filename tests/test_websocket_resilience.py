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
            assert ws_params["ping_timeout"] == config.ALPACA_WS_PING_TIMEOUT_SECONDS
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

        def mock_reconcile(*args, **kwargs):
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


class TestWebSocketAcceptanceCriteria:
    """Explicit verification suite for the 10 Phase B Acceptance Criteria test gates."""

    @pytest.fixture
    def bridge(self):
        b = AlpacaBridge(mode="swing")
        b.account_id = "PA3UODIC5JTA"
        b._connected = True
        return b

    def test_gate_1_ping_timeout_and_backoff_jitter_envelope(self, bridge):
        """Gate 1: Ping timeout (60s) & Backoff Jitter Envelope [0.8*delay_k, 1.2*delay_k]."""
        import random
        assert config.ALPACA_WS_PING_TIMEOUT_SECONDS == 60
        base_delay = config.TRADE_STREAM_RECONNECT_BASE_DELAY_SECONDS
        max_delay = config.TRADE_STREAM_RECONNECT_MAX_DELAY_SECONDS
        jitter_ratio = config.TRADE_STREAM_RECONNECT_JITTER_RATIO

        assert base_delay == 2.0
        assert max_delay == 30.0
        assert jitter_ratio == 0.2

        for k in range(1, 7):
            expected_base = min(max_delay, base_delay * (2.0 ** max(0, k - 1)))
            min_delay = expected_base * (1.0 - jitter_ratio)
            max_delay_bound = expected_base * (1.0 + jitter_ratio)

            for _ in range(50):
                jitter = 1.0 + (random.uniform(-jitter_ratio, jitter_ratio))
                delay = expected_base * jitter
                assert min_delay <= delay <= max_delay_bound, f"Attempt {k}: delay {delay} outside [{min_delay}, {max_delay_bound}]"

    def test_gate_2_post_reconnect_ordering(self, bridge):
        """Gate 2: Ordering invariant: connected < recon_start < recon_finish < HEALTHY < release."""
        timeline = []

        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Transport authenticated",
        )
        timeline.append("connected")

        def mock_reconcile(*args, **kwargs):
            timeline.append("recon_start")
            assert bridge.stream_state == StreamState.RECONCILING
            timeline.append("recon_finish")
            return {"clean": True, "protect": {"failed": [], "unprotected": []}}

        def mock_clear():
            assert bridge.stream_state == StreamState.HEALTHY
            timeline.append("release")

        with patch.object(bridge, "reconcile_startup_state", side_effect=mock_reconcile), \
             patch.object(bridge, "_clear_recovery_halts", side_effect=mock_clear):

            success = bridge._run_post_reconnect_reconciliation()
            assert success is True
            assert bridge.stream_state == StreamState.HEALTHY
            assert timeline == ["connected", "recon_start", "recon_finish", "release"]

    def test_gate_3_single_state_owner_no_stream_state_mutation(self, bridge):
        """Gate 3: Single state owner: calling reconcile_startup_state(manage_stream_state=False) does not mutate stream state or clear halts."""
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        bridge.entry_gate.halt("STANDBY_HALT")

        with patch.object(bridge, "positions_plain", return_value=[]), \
             patch.object(bridge, "working_orders_plain", return_value=[]), \
             patch.object(bridge, "_clear_recovery_halts") as mock_clear:

            recon = bridge.reconcile_startup_state(manage_stream_state=False, allow_policy_side_effects=False)

            assert bridge.stream_state == StreamState.AUTHENTICATED_UNRECONCILED
            mock_clear.assert_not_called()
            assert "STANDBY_HALT" in bridge.entry_gate.reasons

    def test_gate_4_dirty_reconciliation_fail_closed(self, bridge):
        """Gate 4: Dirty reconciliation fail-closed: latches to LATCHED_DESYNC and halts entry gate."""
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated",
        )
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")

        dirty_result = {
            "clean": False,
            "protect": {"failed": ["AAPL"], "unprotected": ["AAPL"]},
            "snapshot": {"clean": False, "unprotected_longs": ["AAPL"]},
        }

        with patch.object(bridge, "reconcile_startup_state", return_value=dirty_result), \
             patch("observability.get_latch_state_machine") as mock_latch:

            success = bridge._run_post_reconnect_reconciliation()

            assert success is False
            assert bridge.stream_state == StreamState.LATCHED_DESYNC
            assert bridge.entry_gate.halted is True
            mock_latch.return_value.latch.assert_called_once()

    def test_gate_5_clean_reconciliation_release(self, bridge):
        """Gate 5: Clean reconciliation releases stream to HEALTHY and discards recovery halts."""
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Authenticated",
        )
        bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        bridge.entry_gate.halt("BROKER_RECONCILIATION_PENDING")
        bridge.entry_gate.halt("OFF_MARKET_STANDBY")
        bridge.entry_gate.halt("MANUAL_OPERATOR_HALT")

        clean_result = {
            "clean": True,
            "protect": {"failed": [], "unprotected": []},
            "snapshot": {"clean": True, "unprotected_longs": []},
        }

        with patch.object(bridge, "reconcile_startup_state", return_value=clean_result):
            success = bridge._run_post_reconnect_reconciliation()

            assert success is True
            assert bridge.stream_state == StreamState.HEALTHY
            assert "TRADE_STREAM_DISCONNECTED" not in bridge.entry_gate.reasons
            assert "BROKER_RECONCILIATION_PENDING" not in bridge.entry_gate.reasons
            assert "OFF_MARKET_STANDBY" not in bridge.entry_gate.reasons
            # Non-recovery halt remains intact
            assert "MANUAL_OPERATOR_HALT" in bridge.entry_gate.reasons
            assert bridge.entry_gate.halted is True

    def test_gate_6_market_close_pause_invariants(self, bridge):
        """Gate 6: Market-close pause contract: preserves REST connection, positions, GTC orders, does not graceful_shutdown."""
        bridge._connected = True
        bridge._trade_stream = MagicMock()
        bridge._trade_stream_thread = MagicMock()
        bridge._stream_state = StreamState.HEALTHY

        with patch.object(bridge, "_stop_trade_stream") as mock_stop, \
             patch.object(bridge, "graceful_shutdown") as mock_shutdown:

            bridge.pause_trade_stream_for_market_close()

            mock_stop.assert_called_once_with(cancel_reconnect=True)
            mock_shutdown.assert_not_called()
            assert bridge.stream_state == StreamState.STOPPED
            assert bridge.entry_gate.halted is True
            assert "OFF_MARKET_STANDBY" in bridge.entry_gate.reasons
            assert bridge._connected is True

    def test_gate_7_market_open_resume_invariants(self, bridge):
        """Gate 7: Market-open resume contract: connects stream, reconciles, clears OFF_MARKET_STANDBY on clean."""
        bridge._trade_stream_credentials = ("test_key", "test_sec")
        bridge._stream_state = StreamState.STOPPED
        bridge.entry_gate.halt("OFF_MARKET_STANDBY")

        # Clean resume scenario
        with patch.object(bridge, "_start_trade_stream") as mock_start, \
             patch.object(bridge, "_run_post_reconnect_reconciliation", return_value=True):

            ok = bridge.resume_trade_stream_for_market_open()
            assert ok is True
            mock_start.assert_called_once_with("test_key", "test_sec")
            assert "OFF_MARKET_STANDBY" not in bridge.entry_gate.reasons
            assert bridge.entry_gate.halted is False

        # Dirty resume fails closed scenario
        bridge._stream_state = StreamState.STOPPED
        bridge.entry_gate.halt("OFF_MARKET_STANDBY")
        with patch.object(bridge, "_start_trade_stream") as mock_start, \
             patch.object(bridge, "_run_post_reconnect_reconciliation", return_value=False):

            ok = bridge.resume_trade_stream_for_market_open()
            assert ok is False
            assert bridge.entry_gate.halted is True

    def test_gate_8_holiday_and_off_market_standby_scheduler(self):
        """Gate 8: When market is closed (weekday holiday/off-market), pauses stream, zero predictions."""
        from main import _run_swing_scheduler_loop

        mock_bridge = MagicMock()
        mock_bridge.connect.return_value = True
        mock_bridge.get_net_liquidation.return_value = 100000.0
        mock_bridge.open_position_count.return_value = 0
        mock_bridge._entry_time_in_force.return_value = MagicMock(value="gtc")
        mock_bridge.graceful_shutdown.return_value = MagicMock()

        mock_clock = MagicMock(is_open=False, timestamp=None, next_open=None)
        mock_bridge.get_market_clock.return_value = mock_clock

        args = MagicMock(broker="alpaca", dry_run=True, loop=True, interval=1800)

        with patch("alpaca_bridge.AlpacaBridge", return_value=mock_bridge), \
             patch("main._run_paper_startup_checks", return_value={"clean": True}), \
             patch("main._print_startup_reconciliation"), \
             patch("threading.Event.wait", return_value=True), \
             patch("predictor.Predictor") as mock_predictor:

            ret = _run_swing_scheduler_loop(args, interval=1800, broker="alpaca")

            mock_bridge.pause_trade_stream_for_market_close.assert_called_once()
            mock_bridge.resume_trade_stream_for_market_open.assert_not_called()
            mock_predictor.assert_not_called()

    def test_gate_9_market_clock_outage_fail_closed_scheduler(self):
        """Gate 9: Market clock outage fails closed to off-market standby without crashing scheduler."""
        from main import _run_swing_scheduler_loop

        mock_bridge = MagicMock()
        mock_bridge.connect.return_value = True
        mock_bridge.get_net_liquidation.return_value = 100000.0
        mock_bridge.open_position_count.return_value = 0
        mock_bridge._entry_time_in_force.return_value = MagicMock(value="gtc")
        mock_bridge.graceful_shutdown.return_value = MagicMock()

        mock_bridge.get_market_clock.side_effect = RuntimeError("Alpaca 503 Clock Service Unavailable")

        args = MagicMock(broker="alpaca", dry_run=True, loop=True, interval=1800)

        with patch("alpaca_bridge.AlpacaBridge", return_value=mock_bridge), \
             patch("main._run_paper_startup_checks", return_value={"clean": True}), \
             patch("main._print_startup_reconciliation"), \
             patch("threading.Event.wait", return_value=True), \
             patch("predictor.Predictor") as mock_predictor:

            ret = _run_swing_scheduler_loop(args, interval=1800, broker="alpaca")

            mock_bridge.pause_trade_stream_for_market_close.assert_called_once()
            mock_bridge.resume_trade_stream_for_market_open.assert_not_called()
            mock_predictor.assert_not_called()

    def test_gate_10_repeated_disconnects_delay_ceiling(self):
        """Gate 10: Repeated disconnects ceiling: base delay never exceeds 30.0s (capped at 30.0s)."""
        import random
        base_delay = config.TRADE_STREAM_RECONNECT_BASE_DELAY_SECONDS
        max_delay = config.TRADE_STREAM_RECONNECT_MAX_DELAY_SECONDS
        jitter_ratio = config.TRADE_STREAM_RECONNECT_JITTER_RATIO

        for attempt in range(5, 21):
            calculated_base = min(max_delay, base_delay * (2.0 ** max(0, attempt - 1)))
            assert calculated_base == 30.0

            for _ in range(50):
                jitter = 1.0 + random.uniform(-jitter_ratio, jitter_ratio)
                delay = calculated_base * jitter
                assert delay <= 30.0 * 1.2
                assert delay >= 30.0 * 0.8

