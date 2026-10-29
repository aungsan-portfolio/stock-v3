import time
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from strategies.trailing_stop import DynamicTrailingStopManager

def test_stale_pending_exit_purged_when_broker_flat(tmp_path):
    state_file = tmp_path / "trailing_state_test.json"
    mgr = DynamicTrailingStopManager(state_file=state_file)

    symbol = "NVDA"
    seven_days_ago = time.time() - (7 * 86400.0)
    episode_id = f"{symbol}:test_stale_123"

    mgr.pending_exits[episode_id] = {
        "episode_id": episode_id,
        "symbol": symbol,
        "side": "BUY",
        "qty": 17.0,
        "entry_price": 221.42,
        "target_price": 222.43,
        "stop_price": 221.01,
        "entry_time": seven_days_ago,
        "registered_at": seven_days_ago,
        "attempts": 1,
    }
    mgr._save_state()

    assert mgr.has_pending_exit(symbol) is True

    # Mock bridge confirming position is flat (0) and 0 working orders
    bridge = MagicMock()
    bridge.get_position.return_value = 0.0
    bridge.get_positions.return_value = []
    bridge.positions_plain.return_value = []
    bridge.working_orders_plain.return_value = []
    bridge.get_orders.return_value = []
    if hasattr(bridge, "_client"):
        bridge._client.get_orders.return_value = []

    results = mgr.reconcile_pending_exits(bridge=bridge)

    assert results.get(symbol) == "STALE_PENDING_EXIT_PURGED"
    assert mgr.has_pending_exit(symbol) is False
    assert episode_id not in mgr.pending_exits

    # Ensure persistence was saved
    mgr2 = DynamicTrailingStopManager(state_file=state_file)
    assert mgr2.has_pending_exit(symbol) is False
