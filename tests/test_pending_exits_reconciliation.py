import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from strategies.trailing_stop import DynamicTrailingStopManager


@pytest.fixture
def clean_manager(tmp_path):
    state_file = tmp_path / "trailing_state.json"
    mgr = DynamicTrailingStopManager(state_file=str(state_file))
    mgr.pending_exits.clear()
    mgr.states.clear()
    return mgr


def test_reconcile_blocks_when_broker_position_open(clean_manager):
    clean_manager.pending_exits["AAPL:ep1"] = {
        "episode_id": "AAPL:ep1",
        "symbol": "AAPL",
        "side": "BUY",
        "qty": 10.0,
        "entry_price": 150.0,
        "stop_price": 149.0,
    }

    mock_bridge = Mock()
    mock_bridge.get_position.return_value = 10.0  # Still open

    res = clean_manager.reconcile_pending_exits(mock_bridge)
    assert res.get("AAPL") == "POSITION_STILL_OPEN_ON_BROKER"
    assert clean_manager.has_pending_exit("AAPL")


def test_reconcile_blocks_when_working_orders_exist(clean_manager):
    clean_manager.pending_exits["AAPL:ep1"] = {
        "episode_id": "AAPL:ep1",
        "symbol": "AAPL",
        "side": "BUY",
        "qty": 10.0,
        "entry_price": 150.0,
        "stop_price": 149.0,
    }

    mock_bridge = Mock()
    mock_bridge.get_position.return_value = 0.0  # Flat
    mock_bridge.working_orders_plain.return_value = [{"symbol": "AAPL", "qty": 10, "side": "sell"}]

    res = clean_manager.reconcile_pending_exits(mock_bridge)
    assert res.get("AAPL") == "WORKING_ORDERS_PENDING_RETAINED"
    assert clean_manager.has_pending_exit("AAPL")


def test_reconcile_clears_when_broker_flat_no_orders_and_journal_closed(clean_manager):
    clean_manager.pending_exits["AAPL:ep1"] = {
        "episode_id": "AAPL:ep1",
        "symbol": "AAPL",
        "side": "BUY",
        "qty": 10.0,
        "entry_price": 150.0,
        "stop_price": 149.0,
        "signal_id": "ep1",
    }

    mock_bridge = Mock()
    mock_bridge.get_position.return_value = 0.0  # Flat
    mock_bridge.working_orders_plain.return_value = []
    mock_bridge._get_active_orders.return_value = []

    mock_journal_records = [
        {
            "symbol": "AAPL",
            "event_type": "TRADE_CLOSED",
            "qty": 10.0,
            "exit_price": 151.0,
            "pnl": 10.0,
            "signal_id": "ep1",
            "idempotency_key": "exit_AAPL:ep1",
        }
    ]

    with patch("strategies.trade_journal.read_journal", return_value=mock_journal_records):
        res = clean_manager.reconcile_pending_exits(mock_bridge)

    assert res.get("AAPL") == "RECONCILED_FLAT_VERIFIED"
    assert not clean_manager.has_pending_exit("AAPL")
    assert "AAPL:ep1" not in clean_manager.pending_exits


def test_reconcile_rejects_stale_historical_closure_with_different_episode_id(clean_manager):
    """
    CRITICAL SAFETY REGRESSION TEST:
    A historical TRADE_CLOSED record for AAPL with an older/different episode or signal ID
    MUST NEVER clear a newly active pending exit for AAPL.
    """
    clean_manager.pending_exits["AAPL:new_active_ep"] = {
        "episode_id": "AAPL:new_active_ep",
        "symbol": "AAPL",
        "side": "BUY",
        "qty": 10.0,
        "entry_price": 155.0,
        "stop_price": 154.0,
        "signal_id": "new_active_ep",
        "entry_time": 1792500000.0,
    }

    mock_bridge = Mock()
    mock_bridge.get_position.return_value = 0.0  # Flat
    mock_bridge.working_orders_plain.return_value = []
    mock_bridge._get_active_orders.return_value = []

    # Journal only contains an old historical trade closed from earlier in the day
    stale_historical_records = [
        {
            "symbol": "AAPL",
            "event_type": "TRADE_CLOSED",
            "qty": 10.0,
            "exit_price": 148.0,
            "pnl": 5.0,
            "signal_id": "old_yesterday_signal",
            "idempotency_key": "exit_AAPL:old_yesterday_signal",
            "timestamp": "2026-10-19T14:00:00+00:00",  # Prior to entry_time
        }
    ]

    with patch("strategies.trade_journal.read_journal", return_value=stale_historical_records):
        res = clean_manager.reconcile_pending_exits(mock_bridge)

    # Must retain fail-closed!
    assert res.get("AAPL") == "FLAT_PENDING_EXIT_RECONCILIATION"
    assert clean_manager.has_pending_exit("AAPL")
    assert "AAPL:new_active_ep" in clean_manager.pending_exits


def test_reconcile_retains_fail_closed_when_no_closed_evidence(clean_manager):
    clean_manager.pending_exits["MSFT:ep2"] = {
        "episode_id": "MSFT:ep2",
        "symbol": "MSFT",
        "side": "BUY",
        "qty": 5.0,
        "entry_price": 400.0,
        "stop_price": 395.0,
    }

    mock_bridge = Mock()
    mock_bridge.get_position.return_value = 0.0  # Flat
    mock_bridge.working_orders_plain.return_value = []
    mock_bridge._get_active_orders.return_value = []

    # Journal has no records for MSFT
    with patch("strategies.trade_journal.read_journal", return_value=[]):
        res = clean_manager.reconcile_pending_exits(mock_bridge)

    assert res.get("MSFT") == "FLAT_PENDING_EXIT_RECONCILIATION"
    assert clean_manager.has_pending_exit("MSFT")
