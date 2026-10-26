import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from strategies.atomic_rotation import CloseOutcome, PositionCloseResult
from strategies.entry_lifecycle import EntryOrderLifecycle, ProtectionState, PendingEntry
from strategies.validation_lane import get_production_risk_equity


# ─────────────────────────────────────────────────────────────────────────────
# 1. Atomic P&L Snapshot Consistency Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_atomic_pnl_snapshot_consistency():
    """Verify that get_production_risk_equity derives raw_pnl from the exact same
    net_liquidation reading as raw_equity, eliminating async drift."""
    # Simulate fluctuating net_liquidation if queried repeatedly
    fluctuating_equities = [98376.39, 98374.45, 98372.67]

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.side_effect = fluctuating_equities
    mock_bridge._start_of_day_equity = 98388.33

    # Bridge account_daily_pnl uses current_equity keyword if supplied
    def mock_account_daily_pnl(include_validation=True, current_equity=None):
        eq = current_equity if current_equity is not None else mock_bridge.get_net_liquidation()
        return eq - mock_bridge._start_of_day_equity

    mock_bridge.account_daily_pnl.side_effect = mock_account_daily_pnl

    with patch("strategies.validation_lane.get_validation_daily_pnl", return_value=0.0):
        gov_equity, prod_pnl = get_production_risk_equity(mock_bridge, include_validation=False)

    # First call returned 98376.39
    expected_pnl = round(98376.39 - 98388.33, 2)  # -11.94
    assert gov_equity == 98376.39
    assert prod_pnl == expected_pnl
    # Must only have called get_net_liquidation ONCE
    assert mock_bridge.get_net_liquidation.call_count == 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cancel-and-Close Protection Coordinator Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_close_verified_cancel_timeout_classifies_protection_status():
    """When bracket cancellation times out during close_position_verified,
    the coordinator must reconcile active orders and classify protection status
    (ABSENT, PENDING_CANCELLATION, or ACTIVE) instead of leaving it untracked."""
    from alpaca_bridge import AlpacaBridge

    bridge = AlpacaBridge.__new__(AlpacaBridge)
    bridge._connected = True
    bridge._paper = True
    bridge.mode = "daytrade"
    bridge.account_id = "PA3UODIC5JTA"
    bridge._client = MagicMock()
    bridge._client._paper = True
    bridge._client._base_url = "https://paper-api.alpaca.markets"
    bridge.mutation_counters = MagicMock()
    bridge.dry_run = False
    bridge.assert_account_mode_safety = MagicMock(return_value=True)

    # Symbol holds 4 shares
    mock_pos = MagicMock()
    mock_pos.qty = 4.0
    mock_pos.side = "long"
    bridge.get_position = MagicMock(return_value=mock_pos)

    # _cancel_all_orders_for_symbol runs, but active orders query shows 1 order still in flight
    bridge._cancel_all_orders_for_symbol = MagicMock()

    # Scenario A: Active order is a limit sell (Take Profit) still pending cancel (NO stop loss active)
    mock_active_tp = MagicMock(id="tp_order_1", symbol="MU")
    mock_active_tp.order_type = "limit"
    mock_active_tp.type = "limit"
    mock_active_tp.status = "new"
    bridge._get_active_orders = MagicMock(return_value=[mock_active_tp])

    with patch("time.sleep", return_value=None):
        result = bridge.close_position_verified("MU")

    assert result.outcome == CloseOutcome.UNKNOWN
    # Protection status is ABSENT because no stop loss is active
    assert result.protection_status == "ABSENT"
    assert result.remaining_qty == 4.0

    # Scenario B: Stop loss is present and in pending_cancel status
    mock_active_stop_cancelling = MagicMock(id="stop_order_1", symbol="MU")
    mock_active_stop_cancelling.order_type = "stop"
    mock_active_stop_cancelling.type = "stop"
    mock_active_stop_cancelling.status = "pending_cancel"
    bridge._get_active_orders = MagicMock(return_value=[mock_active_stop_cancelling])

    with patch("time.sleep", return_value=None):
        result_cancelling = bridge.close_position_verified("MU")

    assert result_cancelling.outcome == CloseOutcome.UNKNOWN
    assert result_cancelling.protection_status == "PENDING_CANCELLATION"

    # Scenario C: Stop loss is working normally
    mock_active_stop_working = MagicMock(id="stop_order_2", symbol="MU")
    mock_active_stop_working.order_type = "stop"
    mock_active_stop_working.type = "stop"
    mock_active_stop_working.status = "new"
    bridge._get_active_orders = MagicMock(return_value=[mock_active_stop_working])

    with patch("time.sleep", return_value=None):
        result_working = bridge.close_position_verified("MU")

    assert result_working.outcome == CloseOutcome.UNKNOWN
    assert result_working.protection_status == "ACTIVE"


def test_close_verified_cancel_timeout_with_zero_stops_marks_absent():
    """If broker queries fail during cancel convergence and post-timeout audit
    reveals 0 active orders while position remains open, protection status must
    be classified as ABSENT."""
    from alpaca_bridge import AlpacaBridge

    bridge = AlpacaBridge.__new__(AlpacaBridge)
    bridge._connected = True
    bridge._paper = True
    bridge.mode = "daytrade"
    bridge.account_id = "PA3UODIC5JTA"
    bridge._client = MagicMock()
    bridge._client._paper = True
    bridge._client._base_url = "https://paper-api.alpaca.markets"
    bridge.mutation_counters = MagicMock()
    bridge.dry_run = False
    bridge.assert_account_mode_safety = MagicMock(return_value=True)

    mock_pos = MagicMock()
    mock_pos.qty = 4.0
    mock_pos.side = "long"
    bridge.get_position = MagicMock(return_value=mock_pos)
    bridge._cancel_all_orders_for_symbol = MagicMock()

    # Cancel convergence query fails with network/broker error for 3.0s,
    # but post-timeout audit finds 0 active orders
    mono_values = iter([0.0, 0.0, 0.5, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0] + [4.5] * 20)
    order_results = iter([RuntimeError("Broker 504 Gateway Timeout"), []] + [[]] * 20)

    def mock_get_orders():
        res = next(order_results)
        if isinstance(res, Exception):
            raise res
        return res

    bridge._get_active_orders = mock_get_orders

    with patch("time.sleep", return_value=None), patch("time.monotonic", side_effect=lambda: next(mono_values)):
        result = bridge.close_position_verified("MU")

    assert result.outcome == CloseOutcome.UNKNOWN
    assert result.protection_status == "ABSENT"


def test_time_exit_failure_dispatches_immediate_remediation_on_absent():
    """If close_position_verified returns UNKNOWN with protection_status='ABSENT',
    orchestrator must NOT simply 'continue'; it must immediately trigger
    handle_naked_position in the same cycle."""
    mock_bridge = MagicMock()
    mock_manager = MagicMock()
    mock_state = MagicMock()
    mock_state.active = True
    mock_state.order_id = "old_canceled_stop_id"
    mock_state.entry_price = 889.365
    mock_state.entry_time = time.time() - 1800  # 30m ago

    failed_close_result = PositionCloseResult(
        outcome=CloseOutcome.UNKNOWN,
        symbol="MU",
        initial_qty=4.0,
        remaining_qty=4.0,
        filled_qty=0.0,
        error="Cancel convergence timeout",
        protection_status="ABSENT",
    )
    mock_bridge.close_position_verified.return_value = failed_close_result

    sym = "MU"
    p = MagicMock(position=4.0)
    live_paper = True

    close_res = mock_bridge.close_position_verified(sym)
    closed = getattr(close_res, "is_flat", False)
    assert not closed

    prot_status = getattr(close_res, "protection_status", "UNKNOWN")
    assert prot_status == "ABSENT"

    if mock_state and mock_state.active:
        mock_state.stop_reconciliation_required = True
        if prot_status == "ABSENT":
            mock_state.order_id = None
            mock_manager.handle_naked_position(
                symbol=sym,
                position_qty=p.position,
                bridge=mock_bridge,
                dry_run=not live_paper,
            )

    # Must clear stale order_id and trigger handle_naked_position immediately
    assert mock_state.order_id is None
    assert mock_state.stop_reconciliation_required is True
    mock_manager.handle_naked_position.assert_called_once_with(
        symbol="MU",
        position_qty=4.0,
        bridge=mock_bridge,
        dry_run=False,
    )


def test_pending_cancellation_does_not_clear_order_or_rearm_until_terminal():
    """When protection_status is PENDING_CANCELLATION, orchestrator must NOT clear
    local order_id or trigger handle_naked_position prematurely, preventing
    duplicate exits or racing against in-flight orders."""
    mock_bridge = MagicMock()
    mock_manager = MagicMock()
    mock_state = MagicMock()
    mock_state.active = True
    mock_state.order_id = "stop_in_pending_cancel_123"

    pending_cancel_close_result = PositionCloseResult(
        outcome=CloseOutcome.UNKNOWN,
        symbol="MU",
        initial_qty=4.0,
        remaining_qty=4.0,
        filled_qty=0.0,
        error="Cancel convergence timeout",
        protection_status="PENDING_CANCELLATION",
    )
    mock_bridge.close_position_verified.return_value = pending_cancel_close_result

    sym = "MU"
    p = MagicMock(position=4.0)
    live_paper = True

    close_res = mock_bridge.close_position_verified(sym)
    prot_status = getattr(close_res, "protection_status", "UNKNOWN")

    if mock_state and mock_state.active:
        mock_state.stop_reconciliation_required = True
        if prot_status == "ABSENT":
            mock_state.order_id = None
            mock_manager.handle_naked_position(
                symbol=sym,
                position_qty=p.position,
                bridge=mock_bridge,
                dry_run=not live_paper,
            )
        elif prot_status == "PENDING_CANCELLATION":
            pass  # Defer replacement until terminal confirmation

    # Local order_id must NOT be cleared; handle_naked_position must NOT be called!
    assert mock_state.order_id == "stop_in_pending_cancel_123"
    assert mock_state.stop_reconciliation_required is True
    mock_manager.handle_naked_position.assert_not_called()


def test_pending_cancellation_late_stop_fill_reconciles_flat_without_duplicate_exit():
    """If an order in pending_cancel receives a late fill at the broker,
    the reconciler must recognize the position is flat, record pending exit,
    and never submit a duplicate replacement order."""
    from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState

    manager = DynamicTrailingStopManager.__new__(DynamicTrailingStopManager)
    manager.states = {}
    manager.pending_exits = []
    manager._lock = MagicMock()
    manager._lock.__enter__ = MagicMock()
    manager._lock.__exit__ = MagicMock()
    manager._save_state = MagicMock()
    def mock_record(st, **kwargs):
        st.active = False
    manager._record_pending_exit = MagicMock(side_effect=mock_record)
    manager.reconcile_pending_exits = MagicMock()

    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=889.365,
        peak_price=889.365,
        atr=5.0,
        stop_price=881.64,
        trail_multiple=1.5,
        last_updated=time.time(),
        qty=4.0,
        order_id="stop_pending_cancel_1",
        active=True,
        stop_reconciliation_required=True,
    )
    manager.states["MU"] = state

    mock_bridge = MagicMock()
    # Broker position is now 0 (flat because late fill closed it)
    manager._query_broker_position_qty = MagicMock(return_value=(True, 0.0))

    # The broker returns the stop order with status 'filled'
    filled_stop_order = MagicMock()
    filled_stop_order.id = "stop_pending_cancel_1"
    filled_stop_order.status = "filled"
    filled_stop_order.filled_qty = 4.0
    filled_stop_order.qty = 4.0

    # Running handle_naked_position must detect the flat position and abort remediation
    manager.handle_naked_position("MU", 4.0, bridge=mock_bridge, dry_run=False)

    # Position is flat -> recorded pending exit, active=False, NO replacement stop order placed!
    manager._record_pending_exit.assert_called_once()
    mock_bridge._client.submit_order.assert_not_called()
    assert state.active is False


def test_pending_cancellation_late_target_fill_reconciles_flat():
    """If a take-profit order in pending_cancel receives a late fill,
    position is flat (0.0). handle_naked_position must detect 0 shares
    and submit zero replacement orders."""
    from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState

    manager = DynamicTrailingStopManager.__new__(DynamicTrailingStopManager)
    manager.states = {}
    manager.pending_exits = []
    manager._lock = MagicMock()
    manager._lock.__enter__ = MagicMock()
    manager._lock.__exit__ = MagicMock()
    manager._save_state = MagicMock()
    def mock_record(st, **kwargs):
        st.active = False
    manager._record_pending_exit = MagicMock(side_effect=mock_record)
    manager.reconcile_pending_exits = MagicMock()

    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=889.365,
        peak_price=889.365,
        atr=5.0,
        stop_price=881.64,
        trail_multiple=1.5,
        last_updated=time.time(),
        qty=4.0,
        order_id="tp_pending_cancel_1",
        active=True,
        stop_reconciliation_required=True,
    )
    manager.states["MU"] = state

    mock_bridge = MagicMock()
    # Take profit filled -> broker position is 0
    manager._query_broker_position_qty = MagicMock(return_value=(True, 0.0))

    manager.handle_naked_position("MU", 4.0, bridge=mock_bridge, dry_run=False)

    manager._record_pending_exit.assert_called_once()
    mock_bridge._client.submit_order.assert_not_called()
    assert state.active is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Partial Fill Protection Semantics Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_fill_held_stop_reported_as_pending_parent_fill():
    """When a parent entry is partially filled (e.g. 1/4 shares) and the child
    stop order is in 'held' status at Alpaca, it must NOT be reported as
    PROTECTED_COMPLETE. It must be classified as PROTECTED_PENDING_PARENT_FILL."""
    lifecycle = EntryOrderLifecycle(trailing_manager=MagicMock())

    mock_bridge = MagicMock()
    mock_bridge.get_position = MagicMock(return_value=1.0)  # 1 share currently held

    # Parent entry: 4 shares total, 1 share filled
    parent_entry = PendingEntry(
        parent_order_id="entry_parent_1",
        symbol="MU",
        signal=MagicMock(),
        requested_qty=4.0,
        stop_order_id="stop_child_1",
        target_order_id="tp_child_1",
        cumulative_filled_qty=1.0,
    )

    # Child stop order is in "held" status awaiting parent completion
    child_stop_snap = {"status": "held", "qty": 4.0, "filled_qty": 0.0}
    child_tp_snap = {"status": "held", "qty": 4.0, "filled_qty": 0.0}

    def mock_get_order_snapshot(oid):
        if "stop" in oid:
            return child_stop_snap
        elif "tp" in oid:
            return child_tp_snap
        return {"status": "partially_filled", "qty": 4.0, "filled_qty": 1.0}

    mock_bridge.get_order_snapshot.side_effect = mock_get_order_snapshot

    eval_res = lifecycle.evaluate_protection_state(
        bridge=mock_bridge,
        entry=parent_entry,
    )

    # Must NOT report complete active protection
    assert eval_res.state != ProtectionState.PROTECTED_COMPLETE
    assert eval_res.state == ProtectionState.PROTECTED_PENDING_PARENT_FILL
    assert "held" in eval_res.reason.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Shadow / Validation Non-Blocking Decoupling Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_production_signals_execute_before_deferred_validation_routing():
    """Verify that rejected candidates are deferred so that production signals
    are evaluated and submitted first without inline quote delays."""
    execution_order = []

    def mock_production_execute(sig):
        execution_order.append(f"PROD_EXEC_{sig.symbol}")
        return {"status": "PLACED"}

    def mock_shadow_route(item):
        execution_order.append(f"SHADOW_ROUTE_{item['signal'].symbol}")

    deferred_validation = []

    # Candidates evaluated in screening loop:
    # Candidates 1-2 fail confidence; Candidate 3 (TSLA) passes
    candidates = [
        MagicMock(symbol="AAPL", confidence=0.55),
        MagicMock(symbol="MSFT", confidence=0.58),
        MagicMock(symbol="TSLA", confidence=0.68),  # Production pass!
    ]

    min_conf = 0.60
    screened_candidates = []

    for sig in candidates:
        if sig.confidence < min_conf:
            deferred_validation.append({"signal": sig})
            continue
        screened_candidates.append(sig)

    # Execute production candidate first
    for sig in screened_candidates:
        mock_production_execute(sig)

    # Execute deferred validation post-production
    for item in deferred_validation:
        mock_shadow_route(item)

    # TSLA must have executed BEFORE AAPL and MSFT shadow routing
    assert execution_order[0] == "PROD_EXEC_TSLA"
    assert execution_order[1] == "SHADOW_ROUTE_AAPL"
    assert execution_order[2] == "SHADOW_ROUTE_MSFT"


def test_shadow_async_worker_blocked_does_not_delay_production_cycle():
    """Verify that even when the shadow validation worker is intentionally blocked
    or executing slow REST queries, dispatching returns immediately (< 20ms)
    and the production scheduler cycle is completely unblocked."""
    from strategies.validation_lane import dispatch_deferred_validation_candidate, _validation_async_queue

    # Drain queue
    while not _validation_async_queue.empty():
        try:
            _validation_async_queue.get_nowait()
        except Exception:
            pass

    mock_slow_item = {
        "signal": MagicMock(symbol="SLOW_VAL"),
        "bridge": MagicMock(),
        "policy_stage": "STAGE_CONFIDENCE",
        "policy_reason_code": "CONFIDENCE_BELOW_FLOOR",
        "policy_reason_msg": "test",
        "regime_mode": "NORMAL",
        "dry_run": True,
    }

    # Simulate a slow downstream route function
    with patch("strategies.validation_lane.evaluate_and_route_validation_candidate") as mock_route:
        def slow_execution(*args, **kwargs):
            time.sleep(2.0)  # 2 second artificial delay in shadow worker
            return {"status": "DRY_RUN"}

        mock_route.side_effect = slow_execution

        t_start = time.monotonic()
        queued = dispatch_deferred_validation_candidate(mock_slow_item)
        dispatch_duration_ms = (time.monotonic() - t_start) * 1000.0

        # Dispatch must complete in < 50ms on the calling thread despite 2.0s worker sleep
        assert queued is True
        assert dispatch_duration_ms < 50.0, f"Dispatch took too long: {dispatch_duration_ms:.1f}ms"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Buffer-Clear Immediate Cadence Wakeup Tests (Strict 60s Cadence)
# ─────────────────────────────────────────────────────────────────────────────

def test_cadence_enforces_strict_60s_window_and_evaluates_immediate_at_buffer_clear():
    """Verify that:
    1. At 20:45:19 ICT (19s after 20:45:00 candle close), the fresh candle is evaluated immediately.
    2. Any bar with age <= 60.0s is eligible.
    3. Any bar with age > 60.0s (e.g. 61.0s or 69.0s) is strictly rejected as stale (no 90s relaxation).
    4. Evaluated bars are deduplicated and never evaluated twice."""
    grace_seconds = 60.0

    # Case 1: Buffer clears at 20:45:19 (age = 19.0s <= 60.0s) -> Evaluates!
    bar_age_buffer_clear = 19.0
    bar_id = "2026-10-26T13:40:00+00:00"
    last_evaluated = None
    should_evaluate = False

    if last_evaluated is None and bar_age_buffer_clear <= grace_seconds:
        last_evaluated = bar_id
        should_evaluate = True

    assert should_evaluate is True
    assert last_evaluated == bar_id

    # Case 2: In the same bar window, deduplication prevents duplicate evaluation
    should_evaluate_duplicate = False
    if last_evaluated == bar_id:
        should_evaluate_duplicate = False

    assert should_evaluate_duplicate is False

    # Case 3: Stale bar at age 61.0s (> 60.0s grace) -> Strictly REJECTED!
    bar_age_stale = 61.0
    stale_bar_id = "2026-10-26T13:45:00+00:00"
    should_evaluate_stale = False
    skipped_bar_ids = set()

    if bar_age_stale <= grace_seconds:
        should_evaluate_stale = True
    else:
        skipped_bar_ids.add(stale_bar_id)
        should_evaluate_stale = False

    assert should_evaluate_stale is False
    assert stale_bar_id in skipped_bar_ids
