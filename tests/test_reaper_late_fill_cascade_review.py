"""Offline, broker-free regression checks for cancel/fill/child-stop races.

NO Alpaca API credentials or network calls are required. Never submit orders.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from strategies import order_manager
from strategies.entry_lifecycle import EntryLifecycleState, EntryOrderLifecycle


def _case(*, before_fill=0.0, after_fill=0.0, post_status="canceled",
          child_status="canceled", position_qty=0.0, truth_error=False):
    entry = SimpleNamespace(
        parent_order_id="parent-nvda", symbol="NVDA",
        signal=SimpleNamespace(side="BUY"), cumulative_filled_qty=0.0,
        stop_order_id="stop-nvda", target_order_id="target-nvda",
        submitted_at=1.0, expires_at=2.0,
    )
    lifecycle = MagicMock()
    lifecycle.get_expired_entries.return_value = [entry]
    lifecycle.compare_and_swap_state.return_value = True
    bridge = MagicMock()
    readings = [
        SimpleNamespace(id="parent-nvda", symbol="NVDA", side="BUY",
                        status="new", qty=13.0, filled_qty=before_fill,
                        is_terminal=False),
        SimpleNamespace(id="parent-nvda", symbol="NVDA", side="BUY",
                        status=post_status, qty=13.0, filled_qty=after_fill,
                        is_terminal=post_status in {"canceled", "filled"}),
        SimpleNamespace(id="parent-nvda", symbol="NVDA", side="BUY",
                        status=post_status, qty=13.0, filled_qty=after_fill,
                        is_terminal=post_status in {"canceled", "filled"}),
    ]
    if truth_error:
        bridge.get_order_truth.side_effect = RuntimeError("broker unavailable")
    else:
        bridge.get_order_truth.side_effect = readings
    bridge.get_position_truth.return_value = SimpleNamespace(
        symbol="NVDA", qty=position_qty,
    )
    bridge.get_order_snapshot.return_value = {"status": child_status}
    return entry, lifecycle, bridge


@pytest.fixture
def release_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        order_manager.idempotency_manager, "release",
        lambda *a, **kw: calls.append((a, kw)),
    )
    monkeypatch.setattr(order_manager, "send_discord_alert", lambda *a, **kw: None)
    return calls


def _reap(lifecycle, bridge):
    return order_manager.reap_expired_entry_orders(
        bridge, lifecycle=lifecycle, worker_id="test-reaper",
    )


def test_prefilled_bracket_not_cancelled_and_reservation_retained(release_spy):
    _, lifecycle, bridge = _case(before_fill=5.0)
    result = _reap(lifecycle, bridge)
    assert result[0]["status"] == "RECONCILIATION_REQUIRED"
    bridge.cancel_order.assert_not_called()
    lifecycle.finalize_entry.assert_not_called()
    lifecycle._halt_entries.assert_called()
    assert not release_spy


def test_late_fill_during_cancel_does_not_release_or_duplicate_stop(release_spy):
    _, lifecycle, bridge = _case(after_fill=5.0)
    result = _reap(lifecycle, bridge)
    assert result[0]["status"] == "RECONCILIATION_REQUIRED"
    bridge.cancel_order.assert_called_once_with("parent-nvda")
    bridge.place_protective_stop.assert_not_called()
    bridge.place_stop_order.assert_not_called()
    lifecycle.finalize_entry.assert_not_called()
    assert not release_spy


def test_pending_cancel_is_not_terminal(release_spy):
    _, lifecycle, bridge = _case(post_status="pending_cancel")
    result = _reap(lifecycle, bridge)
    assert result[0]["status"] == "RECONCILIATION_REQUIRED"
    assert not release_spy
    lifecycle.finalize_entry.assert_not_called()


def test_child_leg_still_held_blocks_release(release_spy):
    _, lifecycle, bridge = _case(child_status="held")
    result = _reap(lifecycle, bridge)
    assert result[0]["status"] == "RECONCILIATION_REQUIRED"
    assert not release_spy
    lifecycle.finalize_entry.assert_not_called()


def test_unknown_order_truth_never_cancels(release_spy):
    _, lifecycle, bridge = _case(truth_error=True)
    result = _reap(lifecycle, bridge)
    assert result[0]["status"] == "RECONCILIATION_REQUIRED"
    bridge.cancel_order.assert_not_called()
    assert not release_spy


def test_nonflat_broker_position_blocks_release(release_spy):
    _, lifecycle, bridge = _case(position_qty=5.0)
    result = _reap(lifecycle, bridge)
    assert result[0]["status"] == "RECONCILIATION_REQUIRED"
    lifecycle.finalize_entry.assert_not_called()
    assert not release_spy


def test_verified_zero_fill_flat_terminal_children_can_release(release_spy):
    _, lifecycle, bridge = _case()
    result = _reap(lifecycle, bridge)
    assert result[0]["status"] == "CANCELED_UNFILLED_VERIFIED"
    lifecycle.finalize_entry.assert_called_once()
    assert len(release_spy) == 1


def test_terminal_stream_event_preserves_pending_record_and_reservation(monkeypatch):
    from strategies import entry_lifecycle as module
    manager = EntryOrderLifecycle(MagicMock())
    entry = SimpleNamespace(
        parent_order_id="parent-nvda", symbol="NVDA",
        cumulative_filled_qty=5.0, state=EntryLifecycleState.WORKING,
    )
    with manager._lock:
        manager._pending["parent-nvda"] = entry
    release = MagicMock()
    monkeypatch.setattr(
        "strategies.order_idempotency.idempotency_manager.release_on_terminal",
        release,
    )
    manager.on_parent_terminal("parent-nvda", "canceled")
    assert manager.get_pending_entry("parent-nvda") is entry
    assert entry.state is EntryLifecycleState.RECONCILING
    release.assert_not_called()
