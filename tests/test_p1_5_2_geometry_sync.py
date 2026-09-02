"""
tests/test_p1_5_2_geometry_sync.py -- Dedicated regression suite for P1.5.2 Hotfix.

Covers all 23 mandated invariants:
 1. test_shifted_submitted_entry_propagates_to_pending_context
 2. test_shifted_submitted_stop_propagates_to_pending_context
 3. test_shifted_submitted_target_propagates_to_pending_context
 4. test_partial_fill_initializes_trailing_with_submitted_stop
 5. test_cumulative_fill_preserves_submitted_stop
 6. test_trailing_state_stores_desired_and_confirmed_broker_stop
 7. test_divergent_broker_stop_flags_reconciliation_required
 8. test_reconciliation_never_silently_max_merges_divergent_stops
 9. test_reconciliation_log_shows_both_desired_and_confirmed_prices
10. test_broker_confirmed_modify_synchronizes_both_prices
11. test_failed_broker_modify_preserves_broker_confirmed_truth
12. test_snapshot_miss_triggers_direct_order_lookup
13. test_snapshot_miss_retains_id_if_broker_reports_held
14. test_snapshot_miss_processes_exit_if_broker_reports_filled
15. test_partial_stop_fill_does_not_hard_reset_remaining_position
16. test_canceled_stop_with_live_position_triggers_remediation
17. test_unchanged_order_id_logs_at_debug_not_info
18. test_old_trailing_state_json_migrates_successfully
19. test_bar_end_calculation_from_bar_start_timestamp
20. test_29s_post_close_bar_accepted_as_fresh
21. test_post_close_bar_exceeding_grace_rejected_as_stale
22. test_future_dated_completed_bar_fails_closed
23. test_entry_lifecycle_partial_fill_uses_submitted_geometry
"""
import json
import logging
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import config
from strategies.base import TradeSignal
from strategies.entry_lifecycle import EntryOrderLifecycle, PendingEntry
from strategies.intraday_data import _validate_intraday_data
from strategies.order_manager import (
    _pending_order_contexts,
    _pending_order_contexts_lock,
    on_broker_fill_event,
    register_pending_order,
)
from strategies.trailing_stop import (
    DynamicTrailingStopManager,
    TrailingStopState,
)


import tempfile
import shutil

@pytest.fixture(autouse=True)
def clean_test_environment():
    """Isolate state files and clear pending contexts between tests."""
    with _pending_order_contexts_lock:
        _pending_order_contexts.clear()

    temp_dir = tempfile.mkdtemp()
    test_state_file = os.path.join(temp_dir, "trailing_state_test.json")
    with patch("strategies.trailing_stop.STATE_FILE", test_state_file):
        yield test_state_file

    with _pending_order_contexts_lock:
        _pending_order_contexts.clear()
    shutil.rmtree(temp_dir, ignore_errors=True)


def _make_dummy_signal(symbol="MU", entry=948.43, stop=945.31, target=956.16):
    return TradeSignal(
        symbol=symbol,
        strategy="MOMENTUM_SCALP",
        side="BUY",
        confidence=0.85,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        risk_per_share=abs(entry - stop),
        atr=2.50,
        reason="Test momentum signal",
    )


# ---------------------------------------------------------------------------
# Tests 1-3: Submitted Geometry Propagation to Pending Context
# ---------------------------------------------------------------------------

def test_shifted_submitted_entry_propagates_to_pending_context():
    signal = _make_dummy_signal(entry=948.43, stop=945.31, target=956.16)
    register_pending_order(
        parent_id="p-001",
        signal=signal,
        limit_price=947.50,
        qty=4.0,
        stop_id="s-001",
        tp_id="t-001",
        submitted_entry_price=947.50,
        submitted_stop_price=944.38,
        submitted_target_price=956.16,
    )
    with _pending_order_contexts_lock:
        ctx = _pending_order_contexts["p-001"]
        assert ctx["submitted_entry_price"] == 947.50
        assert ctx["signal"].entry_price == 947.50


def test_shifted_submitted_stop_propagates_to_pending_context():
    signal = _make_dummy_signal(entry=948.43, stop=945.31, target=956.16)
    register_pending_order(
        parent_id="p-002",
        signal=signal,
        limit_price=947.50,
        qty=4.0,
        stop_id="s-002",
        tp_id="t-002",
        submitted_entry_price=947.50,
        submitted_stop_price=944.38,
        submitted_target_price=956.16,
    )
    with _pending_order_contexts_lock:
        ctx = _pending_order_contexts["p-002"]
        assert ctx["submitted_stop_price"] == 944.38
        assert ctx["signal"].stop_price == 944.38


def test_shifted_submitted_target_propagates_to_pending_context():
    signal = _make_dummy_signal(entry=948.43, stop=945.31, target=956.16)
    register_pending_order(
        parent_id="p-003",
        signal=signal,
        limit_price=947.50,
        qty=4.0,
        stop_id="s-003",
        tp_id="t-003",
        submitted_entry_price=947.50,
        submitted_stop_price=944.38,
        submitted_target_price=956.16,
    )
    with _pending_order_contexts_lock:
        ctx = _pending_order_contexts["p-003"]
        assert ctx["submitted_target_price"] == 956.16
        assert ctx["signal"].target_price == 956.16


# ---------------------------------------------------------------------------
# Tests 4-5: Partial and Cumulative Fills Consume Submitted Stop
# ---------------------------------------------------------------------------

def test_partial_fill_initializes_trailing_with_submitted_stop():
    mgr = DynamicTrailingStopManager()
    signal = _make_dummy_signal(entry=948.43, stop=945.31, target=956.16)

    register_pending_order(
        parent_id="p-fill-1",
        signal=signal,
        limit_price=947.50,
        qty=4.0,
        stop_id="s-fill-1",
        tp_id="t-fill-1",
        submitted_entry_price=947.50,
        submitted_stop_price=944.38,
        submitted_target_price=956.16,
    )

    with patch("strategies.trailing_stop.get_trailing_manager", return_value=mgr):
        on_broker_fill_event(
            parent_id="p-fill-1",
            fill_price=947.40,
            filled_qty=1.0,
            cumulative_filled_qty=1.0,
            average_fill_price=947.40,
            order_status="partially_filled",
            execution_id="exec-1",
        )

    state = mgr.states.get("MU")
    assert state is not None
    assert state.stop_price == 944.38
    assert state.desired_stop_price == 944.38
    assert state.confirmed_broker_stop_price == 944.38
    assert state.qty == 1.0


def test_cumulative_fill_preserves_submitted_stop():
    mgr = DynamicTrailingStopManager()
    signal = _make_dummy_signal(entry=948.43, stop=945.31, target=956.16)

    register_pending_order(
        parent_id="p-fill-2",
        signal=signal,
        limit_price=947.50,
        qty=4.0,
        stop_id="s-fill-2",
        tp_id="t-fill-2",
        submitted_entry_price=947.50,
        submitted_stop_price=944.38,
        submitted_target_price=956.16,
    )

    with patch("strategies.trailing_stop.get_trailing_manager", return_value=mgr):
        # Fill 1
        on_broker_fill_event(
            parent_id="p-fill-2",
            fill_price=947.40,
            filled_qty=1.0,
            cumulative_filled_qty=1.0,
            average_fill_price=947.40,
            order_status="partially_filled",
            execution_id="exec-1",
        )
        # Fill 2: Cumulative 4.0
        on_broker_fill_event(
            parent_id="p-fill-2",
            fill_price=947.40,
            filled_qty=3.0,
            cumulative_filled_qty=4.0,
            average_fill_price=947.40,
            order_status="filled",
            execution_id="exec-2",
        )

    state = mgr.states.get("MU")
    assert state is not None
    assert state.qty == 4.0
    assert state.stop_price == 944.38
    assert state.confirmed_broker_stop_price == 944.38


# ---------------------------------------------------------------------------
# Tests 6-9: Stop Price Semantics & Broker Reconciliation
# ---------------------------------------------------------------------------

def test_trailing_state_stores_desired_and_confirmed_broker_stop():
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        trail_multiple=1.5,
        last_updated=time.time(),
        desired_stop_price=945.31,
        confirmed_broker_stop_price=944.38,
        stop_reconciliation_required=True,
    )
    assert state.stop_price == 944.38
    assert state.desired_stop_price == 945.31
    assert state.confirmed_broker_stop_price == 944.38
    assert state.stop_reconciliation_required is True


def test_divergent_broker_stop_flags_reconciliation_required():
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=945.31,
        trail_multiple=1.5,
        last_updated=time.time(),
        desired_stop_price=945.31,
        confirmed_broker_stop_price=945.31,
    )
    mgr.states["MU"] = state

    # Broker stop order is at $944.38 (divergent by $0.93)
    mock_order = {
        "id": "stop-order-divergent",
        "symbol": "MU",
        "side": "sell",
        "type": "stop",
        "stop_price": 944.38,
        "status": "open",
    }
    reconciled = mgr._reconcile_broker_order(state, [mock_order])
    assert reconciled is True
    assert state.confirmed_broker_stop_price == 944.38
    assert state.stop_price == 944.38
    assert state.stop_reconciliation_required is True


def test_reconciliation_never_silently_max_merges_divergent_stops():
    """CRITICAL INVARIANT: Reconciliation must NEVER silently max() merge."""
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=945.31,  # Local desired stop was higher
        trail_multiple=1.5,
        last_updated=time.time(),
        desired_stop_price=945.31,
        confirmed_broker_stop_price=945.31,
    )
    mgr.states["MU"] = state

    mock_order = {
        "id": "stop-order-truth",
        "symbol": "MU",
        "side": "sell",
        "type": "stop",
        "stop_price": 944.38,  # Actual broker truth
        "status": "open",
    }
    mgr._reconcile_broker_order(state, [mock_order])

    # Must be 944.38 (broker truth), NOT max(945.31, 944.38) = 945.31!
    assert state.stop_price == 944.38
    assert state.confirmed_broker_stop_price == 944.38
    assert state.desired_stop_price == 945.31


def test_reconciliation_log_shows_both_desired_and_confirmed_prices(caplog):
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        trail_multiple=1.5,
        last_updated=time.time(),
        desired_stop_price=945.31,
        confirmed_broker_stop_price=944.38,
    )
    mgr.states["MU"] = state

    mock_order = {
        "id": "stop-order-log-test",
        "symbol": "MU",
        "side": "sell",
        "type": "stop",
        "stop_price": 944.38,
        "status": "open",
    }

    with caplog.at_level(logging.INFO):
        mgr._reconcile_broker_order(state, [mock_order])

    logs = caplog.text
    assert "broker_stop=$944.38" in logs
    assert "desired_stop=$945.31" in logs


# ---------------------------------------------------------------------------
# Tests 10-11: Split-State update_stop() Semantics
# ---------------------------------------------------------------------------

def test_broker_confirmed_modify_synchronizes_both_prices():
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        desired_stop_price=944.38,
        confirmed_broker_stop_price=944.38,
        trail_multiple=1.5,
        last_updated=0.0,
        order_id="stop-mod-1",
    )
    mgr.states["MU"] = state

    mock_bridge = MagicMock()
    mock_bridge.modify_stop_order.return_value = True

    # Price advances significantly to ratchet stop up
    with patch("strategies.trailing_stop.time.time", return_value=1000.0):
        updated_state = mgr.update_stop(
            symbol="MU",
            current_price=955.00,
            bridge=mock_bridge,
            current_atr=2.50,
        )

    assert updated_state is not None
    assert updated_state.desired_stop_price > 944.38
    assert updated_state.stop_price == updated_state.desired_stop_price
    assert updated_state.confirmed_broker_stop_price == updated_state.desired_stop_price
    assert updated_state.stop_reconciliation_required is False


def test_failed_broker_modify_preserves_broker_confirmed_truth():
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        desired_stop_price=944.38,
        confirmed_broker_stop_price=944.38,
        trail_multiple=1.5,
        last_updated=0.0,
        order_id="stop-mod-fail",
    )
    mgr.states["MU"] = state

    mock_bridge = MagicMock()
    mock_bridge.modify_stop_order.return_value = False  # Modification failed

    with patch("strategies.trailing_stop.time.time", return_value=1000.0):
        updated_state = mgr.update_stop(
            symbol="MU",
            current_price=955.00,
            bridge=mock_bridge,
            current_atr=2.50,
        )

    assert updated_state is not None
    # Desired moved higher, but confirmed truth remains 944.38!
    assert updated_state.desired_stop_price > 944.38
    assert updated_state.stop_price == 944.38
    assert updated_state.confirmed_broker_stop_price == 944.38
    assert updated_state.stop_reconciliation_required is True


# ---------------------------------------------------------------------------
# Tests 12-16: Snapshot Miss Direct Lookup & Remediation
# ---------------------------------------------------------------------------

def test_snapshot_miss_triggers_direct_order_lookup():
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="tracked-stop-snap-miss",
    )
    mgr.states["MU"] = state

    mock_bridge = MagicMock()
    mock_bridge.get_order_snapshot.return_value = {
        "id": "tracked-stop-snap-miss",
        "status": "held",
        "stop_price": 944.38,
        "qty": 4,
    }

    mgr.ensure_initialized(
        symbol="MU",
        side="BUY",
        avg_cost=947.40,
        open_orders=[],  # Snapshot is empty (e.g. parent order snapshot omitted held leg)
        current_price=947.40,
        qty=4.0,
        bridge=mock_bridge,
    )

    mock_bridge.get_order_snapshot.assert_called_once_with("tracked-stop-snap-miss")


def test_snapshot_miss_retains_id_if_broker_reports_held():
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="held-stop-id",
    )
    mgr.states["MU"] = state

    mock_bridge = MagicMock()
    mock_bridge.get_order_snapshot.return_value = {
        "id": "held-stop-id",
        "status": "held",
        "stop_price": 944.38,
        "qty": 4,
    }

    res = mgr.ensure_initialized(
        symbol="MU",
        side="BUY",
        avg_cost=947.40,
        open_orders=[],
        current_price=947.40,
        qty=4.0,
        bridge=mock_bridge,
    )

    assert res.order_id == "held-stop-id"  # Not wiped to None!


def test_snapshot_miss_processes_exit_if_broker_reports_filled():
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="filled-stop-id",
        active=True,
    )
    mgr.states["MU"] = state

    mock_bridge = MagicMock()
    mock_bridge.get_order_snapshot.return_value = {
        "id": "filled-stop-id",
        "status": "filled",
        "stop_price": 944.38,
        "qty": 4,
    }
    mock_bridge.get_positions.return_value = []  # Flat position!

    res = mgr.ensure_initialized(
        symbol="MU",
        side="BUY",
        avg_cost=947.40,
        open_orders=[],
        current_price=944.37,
        qty=0.0,
        bridge=mock_bridge,
    )

    assert res.active is False
    assert res.order_id is None


def test_partial_stop_fill_does_not_hard_reset_remaining_position():
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="partial-stop-id",
        active=True,
        qty=4.0,
    )
    mgr.states["MU"] = state

    mock_bridge = MagicMock()
    mock_bridge.get_order_snapshot.return_value = {
        "id": "partial-stop-id",
        "status": "partially_filled",
        "stop_price": 944.38,
        "qty": 4,
    }
    # Broker reports 2 shares remaining
    mock_bridge.get_positions.return_value = [
        {"symbol": "MU", "qty": 2.0}
    ]

    res = mgr.ensure_initialized(
        symbol="MU",
        side="BUY",
        avg_cost=947.40,
        open_orders=[],
        current_price=944.38,
        qty=2.0,
        bridge=mock_bridge,
    )

    assert res.active is True
    assert res.qty == 2.0


def test_canceled_stop_with_live_position_triggers_remediation():
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="canceled-stop-id",
        active=True,
        qty=4.0,
    )
    mgr.states["MU"] = state

    mock_bridge = MagicMock()
    mock_bridge.get_order_snapshot.return_value = {
        "id": "canceled-stop-id",
        "status": "canceled",
        "stop_price": 944.38,
        "qty": 4,
    }
    mock_bridge.get_positions.return_value = [{"symbol": "MU", "qty": 4.0}]

    res = mgr.ensure_initialized(
        symbol="MU",
        side="BUY",
        avg_cost=947.40,
        open_orders=[],
        current_price=947.40,
        qty=4.0,
        bridge=mock_bridge,
    )

    # Order ID must be cleared so remediation can attach a new protective stop
    assert res.order_id is None
    assert res.active is True


def test_unchanged_order_id_logs_at_debug_not_info(caplog):
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="MU",
        side="BUY",
        entry_price=947.40,
        peak_price=947.40,
        atr=2.50,
        stop_price=944.38,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="same-order-id",
    )
    mgr.states["MU"] = state

    mock_order = {
        "id": "same-order-id",
        "symbol": "MU",
        "side": "sell",
        "type": "stop",
        "stop_price": 944.38,
        "status": "open",
    }

    with caplog.at_level(logging.INFO):
        mgr._reconcile_broker_order(state, [mock_order])

    # Reconciled unchanged order must NOT be logged at INFO
    for record in caplog.records:
        if "Order ID same-order-id active" in record.message:
            assert record.levelno < logging.INFO


# ---------------------------------------------------------------------------
# Test 18: Legacy State JSON Migration
# ---------------------------------------------------------------------------

def test_old_trailing_state_json_migrates_successfully():
    td = tempfile.mkdtemp()
    try:
        legacy_payload = {
            "MU": {
                "symbol": "MU",
                "side": "BUY",
                "entry_price": 947.40,
                "peak_price": 947.40,
                "atr": 2.50,
                "stop_price": 944.38,
                "trail_multiple": 1.5,
                "last_updated": 1725280000.0,
                "order_id": "legacy-stop-id",
                "active": True,
                "qty": 4.0,
            }
        }
        legacy_file = os.path.join(td, "legacy_state.json")
        with open(legacy_file, "w") as f:
            json.dump(legacy_payload, f)

        mgr = DynamicTrailingStopManager()
        with patch("strategies.trailing_stop.STATE_FILE", legacy_file):
            mgr._load_state()

        assert "MU" in mgr.states
        loaded = mgr.states["MU"]
        assert loaded.desired_stop_price == 944.38
        assert loaded.confirmed_broker_stop_price == 944.38
        assert loaded.stop_reconciliation_required is False
    finally:
        shutil.rmtree(td, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests 19-22: 5-Minute Bar Staleness & Freshness Semantics
# ---------------------------------------------------------------------------

def _make_sample_bar_df(timestamp_utc):
    idx = pd.DatetimeIndex([timestamp_utc])
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [10000],
            "atr": [1.50],
        },
        index=idx,
    )
    return df


def test_bar_end_calculation_from_bar_start_timestamp():
    bar_start = pd.Timestamp("2026-09-02 14:40:00", tz="UTC")
    bar_close = bar_start + pd.Timedelta(seconds=300)
    assert str(bar_close) == "2026-09-02 14:45:00+00:00"


def test_29s_post_close_bar_accepted_as_fresh():
    """Live September 2 audit: 14:40:00 bar evaluated at 14:45:29 (29s post-close)."""
    bar_start = pd.Timestamp("2026-09-02 14:40:00", tz="UTC")
    df = _make_sample_bar_df(bar_start)

    eval_time = pd.Timestamp("2026-09-02 14:45:29", tz="UTC")

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("pandas.Timestamp.now", return_value=eval_time):
        res = _validate_intraday_data(df, symbol="MU", interval="5m", validate_staleness=True)
        assert not res.empty, "29s post-close bar must be accepted as fresh!"


def test_post_close_bar_exceeding_grace_rejected_as_stale():
    bar_start = pd.Timestamp("2026-09-02 14:40:00", tz="UTC")
    df = _make_sample_bar_df(bar_start)

    # Evaluated at 14:45:50 (50s post-close, exceeding 45s grace)
    eval_time = pd.Timestamp("2026-09-02 14:45:50", tz="UTC")

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("pandas.Timestamp.now", return_value=eval_time):
        res = _validate_intraday_data(df, symbol="MU", interval="5m", validate_staleness=True)
        assert res.empty, "50s post-close bar exceeding 45s grace must be rejected as stale!"


def test_future_dated_completed_bar_fails_closed():
    bar_start = pd.Timestamp("2026-09-02 14:45:00", tz="UTC")
    df = _make_sample_bar_df(bar_start)

    # Evaluated at 14:40:00 (10 minutes before bar closes!)
    eval_time = pd.Timestamp("2026-09-02 14:40:00", tz="UTC")

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("pandas.Timestamp.now", return_value=eval_time):
        res = _validate_intraday_data(df, symbol="MU", interval="5m", validate_staleness=True)
        assert res.empty, "Future-dated bar must fail closed!"


# ---------------------------------------------------------------------------
# Test 23: Entry Lifecycle Partial Fill Uses Submitted Geometry
# ---------------------------------------------------------------------------

def test_entry_lifecycle_partial_fill_uses_submitted_geometry():
    """CRITICAL INTEGRATION INVARIANT:
    Original stop  = $945.31
    Submitted stop = $944.38
    Partial fill report
    -> Trailing state MUST initialize with submitted stop $944.38!
    """
    mgr = DynamicTrailingStopManager()
    lifecycle = EntryOrderLifecycle(mgr)

    original_signal = _make_dummy_signal(entry=948.43, stop=945.31, target=956.16)

    # Register submission with shifted geometry
    lifecycle.register_submission(
        parent_order_id="p-lifecycle-int",
        signal=original_signal,
        requested_qty=4.0,
        stop_order_id="s-lifecycle-int",
        target_order_id="t-lifecycle-int",
        submitted_entry_price=947.50,
        submitted_stop_price=944.38,
        submitted_target_price=956.16,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_order_snapshot.return_value = {
        "status": "held",
        "qty": 4,
    }

    # First partial fill: 1 share filled
    success = lifecycle.on_parent_fill(
        bridge=mock_bridge,
        parent_order_id="p-lifecycle-int",
        cumulative_filled_qty=1.0,
        average_fill_price=947.40,
        execution_id="exec-lifecycle-1",
    )

    assert success is True
    state = mgr.states.get("MU")
    assert state is not None
    assert state.stop_price == 944.38, f"Expected 944.38 but got {state.stop_price}"
    assert state.desired_stop_price == 944.38
    assert state.confirmed_broker_stop_price == 944.38
