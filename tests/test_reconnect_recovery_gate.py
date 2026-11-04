"""
tests/test_reconnect_recovery_gate.py — Regression Test Suite for Post-Reconnect Recovery Gate.

Tests 6 mandatory invariants (Forensic Audit Requirements):
1. Clean reconciliation: All protection verified, only recovery reasons cleared, gate opens.
2. Positions timeout: Exact attempt count (3 total: 1 initial + 2 retries: 1.5s, 3.0s),
   operator alarm emitted, stream in DEGRADED, gate remains halted.
3. Dirty reconciliation: Unprotected positions or reconciliation failure latches LATCHED_DESYNC,
   emits critical operator alarm, gate remains halted.
4. Unprotected position failsafe: Even if clean=True, non-empty unprotected list latches LATCHED_DESYNC.
5. Non-recovery halt reasons preservation: Critical invariants (TRADE_STREAM_DISPATCH_ERROR,
   TRACKED_ENTRY_REPLACED, DAILY_LOSS_LIMIT, etc.) are NEVER cleared by recovery and keep gate halted.
6. Execution health reporting: ExecutionHealthGate reports HealthStatus.BROKEN with alphabetically
   sorted entry_gate_reasons in metrics and reason message.
"""
from __future__ import annotations

import logging
import threading
import time
from unittest import mock
import pytest
import reconciliation

from alpaca_bridge import AlpacaBridge, StreamState, EntryGateMock
from strategies.execution_health import ExecutionHealthGate, HealthStatus


@pytest.fixture
def bridge():
    b = AlpacaBridge(mode="daytrade")
    b.account_id = "PA3UODIC5JTA"
    b._connected = True
    b._client = mock.MagicMock()
    # Clean recovery scenarios assume an authenticated stream remains alive.
    b._trade_stream = object()
    b._trade_stream_ready = True
    return b


def test_clean_recon_releases_gate(bridge):
    """
    Test 1: Clean recon -> gate opens.
    When broker truth reconciliation succeeds with clean=True and protection_clean=True,
    all recovery reasons are cleared and entry_gate.halted becomes False.
    """
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
    bridge.entry_gate.halt("BROKER_RECONCILIATION_PENDING")
    assert bridge.entry_gate.halted is True
    assert "TRADE_STREAM_DISCONNECTED" in bridge.entry_gate.reasons
    assert "BROKER_RECONCILIATION_PENDING" in bridge.entry_gate.reasons

    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()

    assert ok is True
    assert bridge.stream_state == StreamState.HEALTHY
    assert bridge.entry_gate.halted is False
    assert len(bridge.entry_gate.reasons) == 0


def test_timeout_reconciliation_exact_retries_and_alarm(bridge, caplog):
    """
    Test 2: Reconcile timeout -> exact 3 attempts (1 initial + 2 retries: 1.5s, 3.0s),
    emits critical alarm, transitions to DEGRADED, and gate remains halted.
    """
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
    assert bridge.entry_gate.halted is True

    with mock.patch.object(bridge, "reconcile_startup_state", side_effect=TimeoutError("REST read timeout")) as mock_recon, \
         mock.patch("time.sleep") as mock_sleep, \
         caplog.at_level(logging.CRITICAL):
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()

        assert ok is False
        assert bridge.stream_state == StreamState.DEGRADED
        assert bridge.entry_gate.halted is True
        assert "TRADE_STREAM_DISCONNECTED" in bridge.entry_gate.reasons

        # Verify exact attempt count: 3 total attempts
        assert mock_recon.call_count == 3

        # Verify exact retry backoff delays: 2 sleeps (1.5s then 3.0s)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_has_calls([mock.call(1.5), mock.call(3.0)])

        # Verify operator alarm was emitted with full context
        alarm_records = [
            rec for rec in caplog.records
            if "OPERATOR ALARM" in rec.message and "POST_RECONNECT_TIMEOUT" in rec.message
        ]
        assert len(alarm_records) == 1
        alarm_msg = alarm_records[0].message
        assert "PA3UODIC5JTA" in alarm_msg
        assert "3 total attempts" in alarm_msg
        assert "2 retries" in alarm_msg


def test_dirty_recon_latches_desync_with_alarm(bridge, caplog):
    """
    Test 3: Dirty recon -> transitions to LATCHED_DESYNC, emits alarm, gate remains halted.
    """
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
    assert bridge.entry_gate.halted is True

    dirty_recon_result = {
        "clean": False,
        "protect": {"failed": ["AAPL"], "unprotected": ["NVDA"]},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=dirty_recon_result), \
         caplog.at_level(logging.CRITICAL):
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()

        assert ok is False
        assert bridge.stream_state == StreamState.LATCHED_DESYNC
        assert bridge.entry_gate.halted is True

        alarm_records = [
            rec for rec in caplog.records
            if "OPERATOR ALARM" in rec.message and "POST_RECONNECT_HALT_PERSISTED" in rec.message
        ]
        assert len(alarm_records) == 1
        assert "PA3UODIC5JTA" in alarm_records[0].message


def test_unprotected_positions_fail_safe_latches_desync(bridge, caplog):
    """
    Test 4: Even if clean=True, unprotected positions must fail closed and latch LATCHED_DESYNC.
    """
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
    assert bridge.entry_gate.halted is True

    unprotected_result = {
        "clean": True,  # broker state clean, but protection failed
        "protect": {"failed": [], "unprotected": ["TSLA"]},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=unprotected_result), \
         caplog.at_level(logging.CRITICAL):
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()

        assert ok is False
        assert bridge.stream_state == StreamState.LATCHED_DESYNC
        assert bridge.entry_gate.halted is True
        assert any("POST_RECONNECT_HALT_PERSISTED" in rec.message for rec in caplog.records)


def test_non_recovery_reasons_prevent_gate_release(bridge):
    """
    Test 5: Critical non-recovery reasons (TRADE_STREAM_DISPATCH_ERROR, TRACKED_ENTRY_REPLACED,
    DAILY_LOSS_LIMIT, LATCHED_PROTECTION_FAILURE) MUST NOT be cleared by recovery and keep gate halted.
    """
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
    bridge.entry_gate.halt("TRADE_STREAM_DISPATCH_ERROR")
    bridge.entry_gate.halt("TRACKED_ENTRY_REPLACED")
    bridge.entry_gate.halt("DAILY_LOSS_LIMIT")
    bridge.entry_gate.halt("LATCHED_PROTECTION_FAILURE")

    assert bridge.entry_gate.halted is True
    assert len(bridge.entry_gate.reasons) == 5

    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()

    # Reconcile itself succeeded, but gate MUST remain halted due to non-recovery reasons
    assert ok is True
    assert bridge.stream_state == StreamState.HEALTHY
    assert bridge.entry_gate.halted is True

    # Recovery reason should be cleared, but non-recovery reasons MUST strictly persist
    assert "TRADE_STREAM_DISCONNECTED" not in bridge.entry_gate.reasons
    assert "TRADE_STREAM_DISPATCH_ERROR" in bridge.entry_gate.reasons
    assert "TRACKED_ENTRY_REPLACED" in bridge.entry_gate.reasons
    assert "DAILY_LOSS_LIMIT" in bridge.entry_gate.reasons
    assert "LATCHED_PROTECTION_FAILURE" in bridge.entry_gate.reasons
    assert len(bridge.entry_gate.reasons) == 4


def test_execution_health_reporting_with_sorted_gate_reasons(bridge):
    """
    Test 6: ExecutionHealthGate reports HealthStatus.BROKEN with alphabetically
    sorted entry_gate_reasons in metrics and reason message.
    """
    gate = ExecutionHealthGate()
    bridge.entry_gate.halt("TRACKED_ENTRY_REPLACED")
    bridge.entry_gate.halt("DAILY_LOSS_LIMIT")
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")

    report = gate.evaluate(bridge)

    assert report.status == HealthStatus.BROKEN
    assert report.allows_new_entries is False
    assert report.metrics["entry_gate_halted"] is True

    expected_sorted = ["DAILY_LOSS_LIMIT", "TRACKED_ENTRY_REPLACED", "TRADE_STREAM_DISCONNECTED"]
    assert report.metrics["entry_gate_reasons"] == expected_sorted
    assert "Broker entry gate circuit breaker is halted (reasons: ['DAILY_LOSS_LIMIT', 'TRACKED_ENTRY_REPLACED', 'TRADE_STREAM_DISCONNECTED'])" in report.reason


def test_mock_ib_positions_timeout_halts_gate_and_marks_unhealthy(bridge):
    """
    Test 7: MockIB.positions() timeout/error marks _conn_health unhealthy,
    halts entry_gate with BROKER_RECONCILIATION_PENDING, and raises RuntimeError.
    """
    assert bridge._conn_health.is_healthy is True
    assert bridge.entry_gate.halted is False

    with mock.patch.object(bridge, "_call_with_retry", side_effect=TimeoutError("Alpaca read timeout")):
        with pytest.raises(RuntimeError, match="Alpaca API outage while fetching positions"):
            bridge.ib.positions()

    assert bridge._conn_health.is_healthy is False
    assert "MockIB.positions error" in bridge._conn_health.last_error
    assert bridge.entry_gate.halted is True
    assert "BROKER_RECONCILIATION_PENDING" in bridge.entry_gate.reasons


def test_mock_ib_positions_recovery_delegates_to_single_recovery_owner(bridge):
    """
    Test 8: When MockIB.positions() succeeds while unhealthy or BROKER_RECONCILIATION_PENDING is set,
    it delegates to _run_post_reconnect_reconciliation() (single recovery owner), clearing the gate
    only after a complete and clean reconciliation.
    """
    # REST recovery can release entries only for an authenticated stream;
    # a clean positions response alone cannot recover DISCONNECTED transport.
    bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
    bridge._conn_health.mark_unhealthy("Prior timeout")
    bridge.entry_gate.halt("BROKER_RECONCILIATION_PENDING")
    assert bridge.entry_gate.halted is True

    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "_call_with_retry", return_value=[]), \
         mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
        positions = bridge.ib.positions()

    assert positions == []
    assert bridge._conn_health.is_healthy is True
    assert bridge.entry_gate.halted is False
    assert "BROKER_RECONCILIATION_PENDING" not in bridge.entry_gate.reasons


def test_healthy_to_reconciling_transition_without_invariant_violation(bridge):
    """
    Test 9: Calling _run_post_reconnect_reconciliation() when stream state is already HEALTHY
    transitions cleanly through RECONCILING -> HEALTHY without StreamStateTransitionError.
    """
    bridge._stream_state = StreamState.HEALTHY

    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
        ok = bridge._run_post_reconnect_reconciliation()

    assert ok is True
    assert bridge.stream_state == StreamState.HEALTHY


def test_reconcile_single_flight_lock_prevents_concurrent_reconciliation(bridge):
    """
    Test 10: Single-flight lock (_reconcile_lock) prevents overlapping or concurrent
    reconciliation runs. A second caller immediately returns False with already_in_progress.
    """
    assert bridge._reconcile_lock.locked() is False

    # Simulate an active reconciliation holding the lock
    acquired = bridge._reconcile_lock.acquire(blocking=False)
    assert acquired is True

    try:
        # A concurrent call should fail fast and return False without mutating stream state
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()
        assert ok is False
        assert bridge.stream_state == StreamState.AUTHENTICATED_UNRECONCILED
    finally:
        bridge._reconcile_lock.release()

    # Now that the lock is released, reconciliation runs normally
    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
        ok = bridge._run_post_reconnect_reconciliation()
    assert ok is True
    assert bridge.stream_state == StreamState.HEALTHY


class MockOrder:
    def __init__(self, order_id: str, symbol: str, status: str, filled_qty: float = 0.0, price: float = 0.0):
        self.id = order_id
        self.symbol = symbol
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = price
        self.client_order_id = "test_cid"


class MockUpdate:
    def __init__(
        self,
        event: str = "fill",
        order_id: str = "ord_1001",
        execution_id: str = "exec_1001",
        order_status: str = "filled",
        filled_qty: float = 10.0,
        price: float = 150.25,
        symbol: str = "AAPL",
    ):
        self.event = event
        self.order_id = order_id
        self.execution_id = execution_id
        self.order_status = order_status
        self.position_qty = filled_qty
        self.price = price
        self.order = MockOrder(order_id, symbol, order_status, filled_qty, price) if order_id else None


def test_partial_fill_invalidates_stale_position_state(bridge):
    """
    Test 11: partial_fill immediately invalidates stale position cache.
    Subsequent has_position query must bypass cache and query broker truth.
    """
    # 1. Prime stale position cache (empty set: cache thinks no positions held)
    bridge._cached_position_symbols = set()
    bridge._cached_positions_ts = time.time()
    bridge._cache_requires_reconciliation = False

    # Stale cache indicates AAPL is not held without querying broker get_position
    with mock.patch.object(bridge, "get_position") as mock_get_pos:
        assert bridge.has_position("AAPL") is False
        mock_get_pos.assert_not_called()

    # 2. Incoming partial_fill trade update arrives via WebSocket stream
    update = MockUpdate(
        event="partial_fill",
        order_id="ord_pf_1",
        order_status="partially_filled",
        filled_qty=5.0,
        price=150.0,
        symbol="AAPL",
    )
    bridge._dispatch_trade_update(update)

    # 3. Verify position cache is completely invalidated
    assert bridge._cached_position_symbols is None
    assert bridge._cached_positions_ts == 0.0

    # 4. Subsequent has_position query queries broker truth directly and returns True
    with mock.patch.object(bridge, "get_position", return_value=5.0) as mock_get_pos:
        assert bridge.has_position("AAPL") is True
        mock_get_pos.assert_called_once_with("AAPL")


def test_order_lifecycle_events_invalidate_active_orders_cache(bridge):
    """
    Test 12: fill, canceled, replaced, accepted, rejected invalidate active orders cache.
    """
    lifecycle_events = ["fill", "canceled", "replaced", "accepted", "rejected"]
    for ev in lifecycle_events:
        # Prime active orders cache with stale order
        stale_order = MockOrder(f"ord_{ev}", "AAPL", "new")
        bridge._cached_active_orders = [stale_order]
        bridge._cached_active_orders_ts = time.time()

        update = MockUpdate(
            event=ev,
            order_id=f"ord_{ev}",
            order_status=ev,
            symbol="AAPL",
        )
        bridge._dispatch_trade_update(update)

        # Cache must be wiped
        assert bridge._cached_active_orders is None, f"Failed invalidation for event {ev}"
        assert bridge._cached_active_orders_ts == 0.0, f"Failed timestamp reset for event {ev}"

    # Also verify cancel_order API call directly invalidates active orders cache
    stale_order = MockOrder("ord_cancel_api", "AAPL", "new")
    bridge._cached_active_orders = [stale_order]
    bridge._cached_active_orders_ts = time.time()

    bridge._client.cancel_order_by_id = mock.MagicMock()
    with mock.patch.object(bridge, "get_order_snapshot", return_value={"status": "canceled"}):
        res = bridge.cancel_order("ord_cancel_api")
        assert res.success is True

    assert bridge._cached_active_orders is None
    assert bridge._cached_active_orders_ts == 0.0


def test_reconnect_and_reconciliation_cannot_reuse_pre_disconnect_cache(bridge):
    """
    Test 13: Reconnect/reconciliation cannot reuse pre-disconnect cache data.
    While _cache_requires_reconciliation is True, has_position and _get_active_orders
    must bypass cache completely and query broker truth.
    """
    # 1. Prime pre-disconnect cache
    bridge._cached_position_symbols = {"AAPL"}
    bridge._cached_positions_ts = time.time()
    bridge._cached_active_orders = [MockOrder("ord_pre", "AAPL", "new")]
    bridge._cached_active_orders_ts = time.time()
    bridge._cache_requires_reconciliation = False

    # 2. Stream disconnect occurs
    bridge._mark_trade_stream_unavailable("Simulated network drop")

    assert bridge._cached_position_symbols is None
    assert bridge._cached_active_orders is None
    assert bridge._cache_requires_reconciliation is True

    # 3. Even if stale cache data were hypothetically present, it must be bypassed
    bridge._cached_position_symbols = {"AAPL"}
    bridge._cached_positions_ts = time.time()
    bridge._cached_active_orders = [MockOrder("ord_stale", "AAPL", "new")]
    bridge._cached_active_orders_ts = time.time()

    # has_position must bypass cache and query broker truth
    with mock.patch.object(bridge, "get_position", return_value=0.0) as mock_get_pos:
        assert bridge.has_position("AAPL") is False
        mock_get_pos.assert_called_once_with("AAPL")

    # _get_active_orders must bypass cache and query broker truth
    bridge._client.get_orders = mock.MagicMock(return_value=[])
    active = bridge._get_active_orders()
    assert active == []
    bridge._client.get_orders.assert_called()

    # 4. Stream reconnects -> caches remain invalidated until full clean reconciliation
    bridge._on_sdk_stream_reconnect_success()
    assert bridge._cached_position_symbols is None
    assert bridge._cached_active_orders is None

    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {"clean": True},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
        ok = bridge._run_post_reconnect_reconciliation()

    assert ok is True
    assert bridge._cache_requires_reconciliation is False


def test_unknown_or_unparseable_event_fails_closed(bridge):
    """
    Test 14: Unknown or unparseable event fails closed:
    - Sets _cache_requires_reconciliation = True
    - Halts entry gate with TRADE_STREAM_DISPATCH_ERROR
    - Invalidates caches
    - Schedules forced reconciliation
    """
    # Sub-case A: Missing parent order id
    bad_update = MockUpdate(event="fill", order_id="")
    bad_update.order = None

    with mock.patch.object(bridge, "_schedule_forced_reconciliation") as mock_forced:
        bridge._dispatch_trade_update(bad_update)

    assert bridge.entry_gate.halted is True
    assert "TRADE_STREAM_DISPATCH_ERROR" in bridge.entry_gate.reasons
    assert bridge._cache_requires_reconciliation is True
    mock_forced.assert_called_once_with(reason="missing_parent_order_id")

    # Sub-case B: Unknown / uncertain event type
    bridge.entry_gate.clear("TRADE_STREAM_DISPATCH_ERROR")
    unknown_update = MockUpdate(event="exotic_quantum_fill", order_status="unknown_status")

    with mock.patch.object(bridge, "_schedule_forced_reconciliation") as mock_forced:
        bridge._dispatch_trade_update(unknown_update)

    assert bridge.entry_gate.halted is True
    assert "TRADE_STREAM_DISPATCH_ERROR" in bridge.entry_gate.reasons
    assert bridge._cache_requires_reconciliation is True
    mock_forced.assert_called_once_with(reason="uncertain_event:exotic_quantum_fill")

    # Sub-case C: Dispatch exception
    bridge.entry_gate.clear("TRADE_STREAM_DISPATCH_ERROR")
    err_update = MockUpdate(event="fill")

    with mock.patch.object(bridge, "_remember_trade_update", side_effect=RuntimeError("corrupt dispatch")), \
         mock.patch.object(bridge, "_schedule_forced_reconciliation") as mock_forced:
        bridge._dispatch_trade_update(err_update)

    assert bridge.entry_gate.halted is True
    assert "TRADE_STREAM_DISPATCH_ERROR" in bridge.entry_gate.reasons
    assert bridge._cache_requires_reconciliation is True
    mock_forced.assert_called_once()


def test_single_flight_recovery_lock_no_deadlock(bridge):
    """
    Test 15: Single-flight recovery lock (_reconcile_lock) does not deadlock.
    Multiple callers fail fast when locked, and burst invocations complete safely.
    """
    assert bridge._reconcile_lock.locked() is False

    # 1. Lock held: concurrent calls return False immediately without blocking
    bridge._reconcile_lock.acquire(blocking=False)
    try:
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()
        assert ok is False

        # _schedule_forced_reconciliation should skip thread spawn when locked
        with mock.patch("threading.Thread") as mock_thread:
            bridge._schedule_forced_reconciliation("concurrent_test")
            mock_thread.assert_not_called()
    finally:
        bridge._reconcile_lock.release()

    # 2. Concurrent burst from multiple threads does not deadlock
    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {"clean": True},
    }

    errors = []

    def _worker():
        try:
            with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
                bridge._run_post_reconnect_reconciliation()
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
        assert not t.is_alive(), "Worker thread deadlocked!"

    assert len(errors) == 0
    assert bridge._reconcile_lock.locked() is False



def test_sdk_reconnect_success_when_already_reconciling_does_not_regress_state(bridge):
    """
    Test 16: Atomic Single-Flight Handshake:
    If SDK stream reconnect success fires while reconciliation is already in-flight (RECONCILING),
    it must NOT interrupt or regress stream state, and must allow active reconciliation to complete.
    """
    bridge._stream_state = StreamState.RECONCILING
    with mock.patch.object(bridge, "_run_post_reconnect_reconciliation") as mock_recon:
        res = bridge._on_sdk_stream_reconnect_success()
        assert res is True
        assert bridge.stream_state == StreamState.RECONCILING
        mock_recon.assert_not_called()


def test_sdk_reconnect_success_when_already_healthy_does_not_halt_entry_gate(bridge):
    """
    Test 17: Zero Spurious Latching:
    If SDK stream reconnect success fires when stream is already HEALTHY and clean,
    it must NOT regress to AUTHENTICATED_UNRECONCILED or halt the entry gate.
    """
    bridge._stream_state = StreamState.HEALTHY
    bridge._trade_stream_recovery_required = False
    bridge.entry_gate.halted = False
    bridge.entry_gate.reasons.clear()

    with mock.patch.object(bridge, "_run_post_reconnect_reconciliation") as mock_recon:
        res = bridge._on_sdk_stream_reconnect_success()
        assert res is True
        assert bridge.stream_state == StreamState.HEALTHY
        assert bridge.entry_gate.halted is False
        assert len(bridge.entry_gate.reasons) == 0
        mock_recon.assert_not_called()


def test_atomic_single_flight_scheduling_guard_and_already_in_progress(bridge):
    """
    Test 18: Atomic Single-Flight Scheduling Guard:
    1. Pre-start flag assignment under _reconcile_scheduling_lock.
    2. Concurrent callers receive 'ALREADY_IN_PROGRESS' and do not spawn duplicate threads.
    3. Worker finally block resets _reconcile_in_progress cleanly.
    """
    assert bridge._reconcile_in_progress is False

    started_threads = []
    worker_can_finish = threading.Event()

    class InstrumentedThread(threading.Thread):
        def start(self):
            # Assert flag was ALREADY assigned BEFORE start()!
            assert bridge._reconcile_in_progress is True, "Pre-start flag was not assigned before thread.start()!"
            started_threads.append(self)
            super().start()

    def fake_recon():
        worker_can_finish.wait(timeout=2.0)
        return True

    with mock.patch("threading.Thread", side_effect=InstrumentedThread), \
         mock.patch.object(bridge, "_run_post_reconnect_reconciliation", side_effect=fake_recon):

        outcome1 = bridge._schedule_forced_reconciliation("trigger_1")
        assert outcome1 == "SCHEDULED"
        assert len(started_threads) == 1

        # While worker 1 is running, concurrent callers must receive ALREADY_IN_PROGRESS
        outcome2 = bridge._schedule_forced_reconciliation("trigger_2")
        assert outcome2 == "ALREADY_IN_PROGRESS"

        outcome3 = bridge._schedule_forced_reconciliation("trigger_3")
        assert outcome3 == "ALREADY_IN_PROGRESS"

        assert len(started_threads) == 1, "Duplicate threads spawned despite single-flight guard!"

        # Let worker complete
        worker_can_finish.set()
        started_threads[0].join(timeout=2.0)

    # Worker finally block must have cleared _reconcile_in_progress
    assert bridge._reconcile_in_progress is False


def test_schedule_forced_reconciliation_thread_start_failure_resets_flag_and_halts_gate(bridge):
    """
    Test 19: Thread startup failure fail-safe:
    If threading.Thread.start() raises RuntimeError,
    1. _reconcile_in_progress flag is reset to False under scheduling lock.
    2. Entry gate is halted with STREAM_RECOVERY_REQUIRED (fail-closed).
    3. Function returns 'THREAD_START_FAILED'.
    4. Subsequent calls are NOT permanently blocked with ALREADY_IN_PROGRESS.
    """
    assert bridge._reconcile_in_progress is False

    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            raise RuntimeError("can't start new thread (system resource exhaustion)")

    with mock.patch("threading.Thread", side_effect=FailingThread):
        outcome = bridge._schedule_forced_reconciliation("simulated_thread_failure")
        assert outcome == "THREAD_START_FAILED"

    # Flag must be reset to False (not permanently stuck in True)
    assert bridge._reconcile_in_progress is False

    # Entry gate must remain halted (fail-closed)
    assert bridge.entry_gate.halted is True
    assert "STREAM_RECOVERY_REQUIRED" in bridge.entry_gate.reasons

    # Subsequent recovery call must NOT be blocked by ALREADY_IN_PROGRESS
    with mock.patch("threading.Thread") as mock_thread:
        second_outcome = bridge._schedule_forced_reconciliation("retry_after_failure")
        assert second_outcome != "ALREADY_IN_PROGRESS"
        assert second_outcome == "SCHEDULED"

def test_thread_startup_failure_ordering_prevents_stale_halt_on_concurrent_recovery(bridge):
    """
    Test 20: Deterministic ordering invariant on thread startup failure:
    1. Proves that EntryGate.halt('STREAM_RECOVERY_REQUIRED') is called STRICTLY BEFORE
       _reconcile_in_progress is reset to False (while _reconcile_in_progress is still True
       under _reconcile_scheduling_lock).
    2. Proves that when a concurrent/subsequent recovery runs cleanly:
       - Gate recovery halts (including STREAM_RECOVERY_REQUIRED) are cleanly cleared.
       - No stale halt lingers from the earlier thread failure.
       - Entry gate is unhalted and operational.
    """
    assert bridge._reconcile_in_progress is False

    # Track the state of _reconcile_in_progress and lock state at the exact instant halt() is invoked
    halt_records = []
    original_halt = bridge.entry_gate.halt

    def tracking_halt(reason=""):
        sched_lock = getattr(bridge, "_reconcile_scheduling_lock", None)
        halt_records.append({
            "reason": reason,
            "in_progress_during_halt": bridge._reconcile_in_progress,
            "lock_held_during_halt": sched_lock.locked() if sched_lock else False,
        })
        return original_halt(reason)

    bridge.entry_gate.halt = tracking_halt

    class FailingThread:
        def __init__(self, *args, **kwargs):
            pass
        def start(self):
            raise RuntimeError("can't start new thread (OS process thread limit)")

    # Trigger failure in _schedule_forced_reconciliation
    with mock.patch("threading.Thread", side_effect=FailingThread):
        res = bridge._schedule_forced_reconciliation("simulated_thread_failure")
        assert res == "THREAD_START_FAILED"

    # Invariant 1: Halt was recorded strictly while _reconcile_in_progress was still True and lock was held
    assert len(halt_records) == 1
    assert halt_records[0]["reason"] == "STREAM_RECOVERY_REQUIRED"
    assert halt_records[0]["in_progress_during_halt"] is True
    assert halt_records[0]["lock_held_during_halt"] is True

    # Invariant 2: After the failure block exits, flag is cleanly reset to False and gate is halted fail-closed
    assert bridge._reconcile_in_progress is False
    assert bridge.entry_gate.halted is True
    assert "STREAM_RECOVERY_REQUIRED" in bridge.entry_gate.reasons

    # Invariant 3: A subsequent/concurrent recovery runs cleanly and clears STREAM_RECOVERY_REQUIRED
    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()
        assert ok is True

    # Invariant 4: Entry gate is completely unhalted, zero stale halts linger
    assert bridge.entry_gate.halted is False
    assert "STREAM_RECOVERY_REQUIRED" not in bridge.entry_gate.reasons
    assert len(bridge.entry_gate.reasons) == 0
    assert bridge.stream_state == StreamState.HEALTHY



# ============================================================================
# Breach 2 Fix: Canonical Order Flattening & Held Event Lifecycle (Tests 21-26)
# ============================================================================

def test_canonical_order_flattening_preserves_single_instance_per_order(bridge):
    """
    Test 21: Bracket parent + stop + take-profit canonical flattening:
    Proves that working_orders_plain() consumes the canonical order list from
    _get_active_orders() without double-unnesting legs. Each child leg (stop, limit)
    appears exactly ONCE in working_orders and duplicate_refs is empty.
    """
    class MockLeg:
        def __init__(self, leg_id, sym, side, otype, client_id, stop_price=None, limit_price=None):
            self.id = leg_id
            self.symbol = sym
            self.side = mock.MagicMock(value=side)
            self.type = mock.MagicMock(value=otype)
            self.status = mock.MagicMock(value="held")
            self.client_order_id = client_id
            self.qty = 14.0
            self.filled_qty = 0.0
            self.stop_price = stop_price
            self.limit_price = limit_price
            self.time_in_force = mock.MagicMock(value="GTC")
            self.legs = []

    class MockParent:
        def __init__(self, parent_id, sym, client_id, legs):
            self.id = parent_id
            self.symbol = sym
            self.side = mock.MagicMock(value="buy")
            self.type = mock.MagicMock(value="limit")
            self.status = mock.MagicMock(value="partially_filled")
            self.client_order_id = client_id
            self.qty = 14.0
            self.filled_qty = 6.0
            self.stop_price = None
            self.limit_price = 264.14
            self.time_in_force = mock.MagicMock(value="GTC")
            self.legs = legs

    stop_leg = MockLeg("stop-id-001", "AMZN", "sell", "stop", "875A4C9F-604A-4CA2", stop_price=263.02)
    tp_leg = MockLeg("tp-id-002", "AMZN", "sell", "limit", "AEA6030B-97AB-4D99", limit_price=266.38)
    parent = MockParent("parent-id-000", "AMZN", "se3-AMZN-B-cand-20261020", [stop_leg, tp_leg])

    # In Alpaca SDK, get_orders returns open orders with nested legs,
    # and _get_active_orders() un-nests legs into orders_map.
    with mock.patch.object(bridge._client, "get_orders", return_value=[parent]):
        # Call working_orders_plain
        wo = bridge.working_orders_plain()

    # Must contain exactly 3 orders (parent, stop, tp), NOT 5!
    assert len(wo) == 3, f"Expected 3 orders, got {len(wo)}: {wo}"
    order_ids = [o["id"] for o in wo]
    assert order_ids == ["parent-id-000", "stop-id-001", "tp-id-002"]

    order_refs = [o["order_ref"] for o in wo]
    assert order_refs == ["se3-AMZN-B-cand-20261020", "875A4C9F-604A-4CA2", "AEA6030B-97AB-4D99"]

    # Invariant: duplicate_order_refs is completely empty
    dups = reconciliation.duplicate_order_refs(wo)
    assert dups == [], f"Unexpected duplicate order refs detected: {dups}"

    # Invariant: reconciliation snapshot is clean
    positions = [{"symbol": "AMZN", "qty": 14.0}]
    snapshot = reconciliation.build_snapshot(positions, wo)
    assert snapshot["clean"] is True
    assert snapshot["duplicate_refs"] == []
    assert snapshot["protected_longs"] == ["AMZN"]
    assert snapshot["unprotected_longs"] == []


def test_reconciliation_duplicate_detector_still_catches_intentional_duplicate_refs(bridge):
    """
    Test 22: Intentional duplicate references detection:
    Proves that no blanket ID deduplication was added that would mask real broker idempotency violations.
    If the broker actually reports two distinct working orders sharing the identical order_ref,
    reconciliation must detect the duplicate ref and halt clean=False.
    """
    class MockOrder:
        def __init__(self, order_id, client_id):
            self.id = order_id
            self.symbol = "AMZN"
            self.side = mock.MagicMock(value="buy")
            self.type = mock.MagicMock(value="limit")
            self.status = mock.MagicMock(value="new")
            self.client_order_id = client_id
            self.qty = 10.0
            self.filled_qty = 0.0
            self.stop_price = None
            self.limit_price = 264.00
            self.time_in_force = mock.MagicMock(value="GTC")
            self.legs = []

    # Two distinct orders from broker with colliding client_order_id
    ord1 = MockOrder("broker-id-1", "DUPLICATE-CLIENT-REF-123")
    ord2 = MockOrder("broker-id-2", "DUPLICATE-CLIENT-REF-123")

    with mock.patch.object(bridge._client, "get_orders", return_value=[ord1, ord2]):
        wo = bridge.working_orders_plain()

    assert len(wo) == 2
    dups = reconciliation.duplicate_order_refs(wo)
    assert dups == ["DUPLICATE-CLIENT-REF-123"]

    positions = []
    snapshot = reconciliation.build_snapshot(positions, wo)
    assert snapshot["clean"] is False
    assert "DUPLICATE-CLIENT-REF-123" in snapshot["duplicate_refs"]


def test_valid_held_trade_update_does_not_halt_entry_gate_or_force_reconcile(bridge):
    """
    Test 23: Valid held events do not trigger dispatch error latch:
    When Alpaca streams 'held' trade update for a bracket child order,
    _dispatch_trade_update recognizes it as a valid event, records stream ack,
    and does NOT halt the entry gate or trigger uncertain-event forced reconciliation.
    """
    bridge.entry_gate.halted = False
    bridge.entry_gate.reasons.clear()

    class MockTradeUpdate:
        def __init__(self):
            self.event = "held"
            self.order_id = "stop-leg-order-id-456"
            self.timestamp = mock.MagicMock()
            self.timestamp.isoformat.return_value = "2026-10-20T14:46:06.000000+00:00"
            self.execution_id = "exec-held-001"
            self.order = mock.MagicMock()
            self.order.id = "stop-leg-order-id-456"
            self.order.status = mock.MagicMock(value="held")
            self.order.client_order_id = "875A4C9F-HELD-TEST"
            self.order.symbol = "AMZN"
            self.order.filled_qty = 0.0
            self.order.filled_avg_price = 0.0

    update = MockTradeUpdate()

    with mock.patch.object(bridge, "_schedule_forced_reconciliation") as mock_sched:
        bridge._dispatch_trade_update(update)
        mock_sched.assert_not_called()

    assert bridge.entry_gate.halted is False
    assert "TRADE_STREAM_DISPATCH_ERROR" not in bridge.entry_gate.reasons


def test_real_dispatch_handler_exception_maintains_fail_closed_halt(bridge):
    """
    Test 24: Real handler exception preserves fail-closed safety:
    If an unexpected unhandled exception occurs inside _dispatch_trade_update,
    it must halt the entry gate with TRADE_STREAM_DISPATCH_ERROR and schedule forced reconciliation.
    """
    bridge.entry_gate.halted = False
    bridge.entry_gate.reasons.clear()

    class MalformedUpdate:
        @property
        def order(self):
            raise RuntimeError("Corrupted memory payload inside Alpaca SDK object")

    with mock.patch.object(bridge, "_schedule_forced_reconciliation") as mock_sched:
        bridge._dispatch_trade_update(MalformedUpdate())
        mock_sched.assert_called_once()

    assert bridge.entry_gate.halted is True
    assert "TRADE_STREAM_DISPATCH_ERROR" in bridge.entry_gate.reasons


def test_reconnect_recovery_does_not_clear_trade_stream_dispatch_error(bridge):
    """
    Test 25: Recovery Preconditions Isolation:
    Stream reconnect recovery ONLY clears stream-owned halts.
    Critical dispatch error halts (TRADE_STREAM_DISPATCH_ERROR) are NEVER cleared
    by clean broker reconciliation alone, enforcing strict fail-closed safety.
    """
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
    bridge.entry_gate.halt("TRADE_STREAM_DISPATCH_ERROR")

    assert bridge.entry_gate.halted is True
    assert "TRADE_STREAM_DISCONNECTED" in bridge.entry_gate.reasons
    assert "TRADE_STREAM_DISPATCH_ERROR" in bridge.entry_gate.reasons

    clean_recon_result = {
        "clean": True,
        "protect": {"failed": [], "unprotected": []},
        "snapshot": {},
    }
    with mock.patch.object(bridge, "reconcile_startup_state", return_value=clean_recon_result):
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        ok = bridge._run_post_reconnect_reconciliation()
        assert ok is True

    # Stream is restored to HEALTHY, but entry gate MUST remain halted due to dispatch error
    assert bridge.stream_state == StreamState.HEALTHY
    assert "TRADE_STREAM_DISCONNECTED" not in bridge.entry_gate.reasons
    assert "TRADE_STREAM_DISPATCH_ERROR" in bridge.entry_gate.reasons
    assert bridge.entry_gate.halted is True


def test_full_amzn_bracket_lifecycle_reconciliation_state_consistency(bridge):
    """
    Test 26: Full AMZN bracket lifecycle simulation:
    Simulates:
    1. Child orders receive 'held' trade updates -> Zero false dispatch errors.
    2. Parent fills 14 shares @ 264.09 -> Reconcile state is clean=True, protected_longs=['AMZN'].
    3. Position exits flat -> Reconcile state is clean=True, positions=[], working_orders=[].
    """
    bridge.entry_gate.halted = False
    bridge.entry_gate.reasons.clear()

    # Step 1: Child order receives 'held'
    class HeldUpdate:
        def __init__(self, oid, clid):
            self.event = "held"
            self.order_id = oid
            self.timestamp = mock.MagicMock()
            self.timestamp.isoformat.return_value = "2026-10-20T14:46:05.978446+00:00"
            self.execution_id = f"exec-{oid}"
            self.order = mock.MagicMock()
            self.order.id = oid
            self.order.status = mock.MagicMock(value="held")
            self.order.client_order_id = clid
            self.order.symbol = "AMZN"
            self.order.filled_qty = 0.0
            self.order.filled_avg_price = 0.0

    stop_update = HeldUpdate("38cefc16-ef0d", "875a4c9f-604a")
    tp_update = HeldUpdate("52cb5ec3-5c51", "aea6030b-97ab")

    bridge._dispatch_trade_update(stop_update)
    bridge._dispatch_trade_update(tp_update)

    assert bridge.entry_gate.halted is False
    assert "TRADE_STREAM_DISPATCH_ERROR" not in bridge.entry_gate.reasons

    # Step 2: Parent fills, position is 14 AMZN with resting stop
    class ActiveStopOrder:
        def __init__(self):
            self.id = "38cefc16-ef0d"
            self.symbol = "AMZN"
            self.side = mock.MagicMock(value="sell")
            self.type = mock.MagicMock(value="stop")
            self.status = mock.MagicMock(value="new")
            self.client_order_id = "875a4c9f-604a"
            self.qty = 14.0
            self.filled_qty = 0.0
            self.stop_price = 263.02
            self.limit_price = None
            self.time_in_force = mock.MagicMock(value="GTC")
            self.legs = []

    class ActiveTPOrder:
        def __init__(self):
            self.id = "52cb5ec3-5c51"
            self.symbol = "AMZN"
            self.side = mock.MagicMock(value="sell")
            self.type = mock.MagicMock(value="limit")
            self.status = mock.MagicMock(value="new")
            self.client_order_id = "aea6030b-97ab"
            self.qty = 14.0
            self.filled_qty = 0.0
            self.stop_price = None
            self.limit_price = 266.38
            self.time_in_force = mock.MagicMock(value="GTC")
            self.legs = []

    active_orders = [ActiveStopOrder(), ActiveTPOrder()]
    active_positions = [mock.MagicMock(symbol="AMZN", qty=14.0)]

    with mock.patch.object(bridge._client, "get_orders", return_value=active_orders),          mock.patch.object(bridge._client, "get_all_positions", return_value=active_positions):
        recon = bridge.reconcile_startup_state(manage_stream_state=False, allow_policy_side_effects=False)

    assert recon["clean"] is True
    assert recon["snapshot"]["duplicate_refs"] == []
    assert recon["snapshot"]["protected_longs"] == ["AMZN"]
    assert recon["snapshot"]["unprotected_longs"] == []

    # Step 3: Flat exit (Positions = 0, Working Orders = 0)
    with mock.patch.object(bridge._client, "get_orders", return_value=[]),          mock.patch.object(bridge._client, "get_all_positions", return_value=[]):
        final_recon = bridge.reconcile_startup_state(manage_stream_state=False, allow_policy_side_effects=False)

    assert final_recon["clean"] is True
    assert final_recon["snapshot"]["open_positions"] == []
    assert final_recon["snapshot"]["n_working_orders"] == 0
    assert final_recon["snapshot"]["duplicate_refs"] == []
    assert bridge.entry_gate.halted is False
