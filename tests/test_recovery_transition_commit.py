"""A clean REST snapshot cannot authorize a failed or stale stream recovery."""
from unittest import mock

import pytest

from alpaca_bridge import AlpacaBridge, StreamState


@pytest.fixture
def bridge():
    bridge = AlpacaBridge(mode="daytrade")
    bridge._connected = True
    bridge._client = mock.MagicMock()
    bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
    bridge._trade_stream = object()
    bridge._trade_stream_ready = True
    bridge._trade_stream_recovery_required = True
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
    bridge.entry_gate.halt("BROKER_RECONCILIATION_PENDING")
    return bridge


def clean_snapshot():
    return {"clean": True, "protect": {"failed": [], "unprotected": []}}


def assert_recovery_still_required(bridge):
    assert bridge.entry_gate.halted
    assert "BROKER_RECONCILIATION_PENDING" in bridge.entry_gate.reasons
    assert bridge._trade_stream_recovery_required
    assert not bridge._trade_stream_ready
    assert bridge._cache_requires_reconciliation


def test_disconnect_during_rest_reconciliation_cannot_release_entries(bridge):
    def disconnected_snapshot(**kwargs):
        assert bridge.stream_state == StreamState.RECONCILING
        bridge._transition_stream_state(
            expected={StreamState.RECONCILING},
            new_state=StreamState.DISCONNECTED,
            reason="socket closed during slow broker query",
        )
        bridge._mark_trade_stream_unavailable("socket closed during slow broker query")
        return clean_snapshot()

    with mock.patch.object(bridge, "reconcile_startup_state", side_effect=disconnected_snapshot), \
         mock.patch.object(bridge._conn_health, "mark_healthy") as mark_healthy:
        assert bridge._run_post_reconnect_reconciliation() is False

    assert bridge.stream_state == StreamState.DISCONNECTED
    assert_recovery_still_required(bridge)
    mark_healthy.assert_not_called()


def test_rejected_reconciliation_start_does_not_query_or_release_entries(bridge):
    bridge._stream_state = StreamState.DISCONNECTED
    bridge._trade_stream_ready = False
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_snapshot()) as reconcile:
        assert bridge._run_post_reconnect_reconciliation() is False
    reconcile.assert_not_called()
    assert_recovery_still_required(bridge)


def test_result_for_replaced_stream_generation_cannot_release_entries(bridge):
    new_stream = object()
    def old_generation_snapshot(**kwargs):
        # A different socket generation has started while this REST request
        # was outstanding. Even if it is reconciling, it owns a new recovery.
        with bridge._trade_stream_lock:
            bridge._stream_generation += 1
            bridge._trade_stream = new_stream
            bridge._trade_stream_ready = True
        return clean_snapshot()

    with mock.patch.object(bridge, "reconcile_startup_state", side_effect=old_generation_snapshot), \
         mock.patch.object(bridge._conn_health, "mark_healthy") as mark_healthy:
        assert bridge._run_post_reconnect_reconciliation() is False
    assert bridge._trade_stream is new_stream
    assert bridge._trade_stream_ready is True
    assert bridge._trade_stream_recovery_required
    assert bridge._cache_requires_reconciliation
    assert bridge.entry_gate.halted
    mark_healthy.assert_not_called()


def test_stale_result_does_not_corrupt_new_generation_already_healthy(bridge):
    new_stream = object()
    def old_generation_snapshot(**kwargs):
        with bridge._trade_stream_lock, bridge._stream_state_lock:
            bridge._stream_generation += 1
            bridge._trade_stream = new_stream
            bridge._trade_stream_ready = True
            bridge._trade_stream_recovery_required = False
            bridge._cache_requires_reconciliation = False
            bridge._stream_state = StreamState.HEALTHY
            bridge._clear_recovery_halts()
            bridge.entry_gate.halt("MANUAL_OPERATOR_HALT")
        return clean_snapshot()

    with mock.patch.object(bridge, "reconcile_startup_state", side_effect=old_generation_snapshot), \
         mock.patch.object(bridge._conn_health, "mark_healthy") as mark_healthy:
        assert bridge._run_post_reconnect_reconciliation() is False
    assert bridge.stream_state == StreamState.HEALTHY
    assert bridge._trade_stream is new_stream
    assert bridge._trade_stream_ready is True
    assert bridge._trade_stream_recovery_required is False
    assert bridge._cache_requires_reconciliation is False
    assert bridge.entry_gate.reasons == {"MANUAL_OPERATOR_HALT"}
    mark_healthy.assert_not_called()


@pytest.mark.parametrize("outcome", ["dirty", "timeout"])
def test_stale_failed_result_does_not_corrupt_new_generation(bridge, outcome):
    new_stream = object()
    def old_generation_snapshot(**kwargs):
        with bridge._trade_stream_lock, bridge._stream_state_lock:
            bridge._stream_generation += 1
            bridge._trade_stream = new_stream
            bridge._trade_stream_ready = True
            bridge._trade_stream_recovery_required = False
            bridge._cache_requires_reconciliation = False
            bridge._stream_state = StreamState.HEALTHY
            bridge._clear_recovery_halts()
            bridge.entry_gate.halt("MANUAL_OPERATOR_HALT")
        if outcome == "timeout":
            raise TimeoutError("old request timed out")
        return {"clean": False, "protect": {"failed": ["SPY"], "unprotected": ["SPY"]}}

    with mock.patch.object(bridge, "reconcile_startup_state", side_effect=old_generation_snapshot), \
         mock.patch.object(bridge, "_c", return_value=1), \
         mock.patch("observability.get_latch_state_machine") as latch:
        assert bridge._run_post_reconnect_reconciliation() is False
    assert bridge.stream_state == StreamState.HEALTHY
    assert bridge._trade_stream is new_stream
    assert bridge._trade_stream_ready is True
    assert bridge._trade_stream_recovery_required is False
    assert bridge._cache_requires_reconciliation is False
    assert bridge.entry_gate.reasons == {"MANUAL_OPERATOR_HALT"}
    latch.assert_not_called()


def test_transport_exit_before_disconnect_transition_cannot_release_entries(bridge):
    def exited_during_snapshot(**kwargs):
        # _run_trade_stream clears these under the trade lock, then transitions
        # DISCONNECTED afterward. Simulate the intervening scheduling window.
        with bridge._trade_stream_lock:
            bridge._trade_stream_ready = False
            bridge._trade_stream = None
            bridge._trade_stream_thread = None
        assert bridge.stream_state == StreamState.RECONCILING
        return clean_snapshot()

    with mock.patch.object(bridge, "reconcile_startup_state", side_effect=exited_during_snapshot), \
         mock.patch.object(bridge._conn_health, "mark_healthy") as mark_healthy:
        assert bridge._run_post_reconnect_reconciliation() is False
    assert_recovery_still_required(bridge)
    mark_healthy.assert_not_called()


def test_transport_replaced_without_generation_change_cannot_release_entries(bridge):
    replacement = object()
    def replaced_during_snapshot(**kwargs):
        with bridge._trade_stream_lock:
            bridge._trade_stream = replacement
            bridge._trade_stream_ready = True
        return clean_snapshot()

    with mock.patch.object(bridge, "reconcile_startup_state", side_effect=replaced_during_snapshot), \
         mock.patch.object(bridge._conn_health, "mark_healthy") as mark_healthy:
        assert bridge._run_post_reconnect_reconciliation() is False
    assert bridge._trade_stream is replacement
    assert bridge._trade_stream_ready is True
    assert bridge._trade_stream_recovery_required
    assert bridge._cache_requires_reconciliation
    assert bridge.entry_gate.halted
    mark_healthy.assert_not_called()


def test_current_clean_recovery_still_preserves_nonrecovery_halts(bridge):
    bridge.entry_gate.halt("MANUAL_OPERATOR_HALT")
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_snapshot()):
        assert bridge._run_post_reconnect_reconciliation() is True
    assert bridge.stream_state == StreamState.HEALTHY
    assert bridge._trade_stream_ready
    assert not bridge._trade_stream_recovery_required
    assert not bridge._cache_requires_reconciliation
    assert bridge.entry_gate.reasons == {"MANUAL_OPERATOR_HALT"}
