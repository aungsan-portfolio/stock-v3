import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState
from strategies.intraday_risk import register_symbol_loss, _RECENT_SYMBOL_LOSS_TIMESTAMPS
import alpaca_bridge

def test_stop_leg_fill_dispatches_directly_to_exit(tmp_path):
    state_file = tmp_path / "trailing_state_a1.json"
    mgr = DynamicTrailingStopManager(state_file=state_file)

    symbol = "XOM"
    stop_leg_id = "stop_order_xom_001"

    # Create active state for XOM
    state = TrailingStopState(
        symbol=symbol,
        side="BUY",
        entry_price=182.48,
        peak_price=182.48,
        atr=0.29,
        trail_multiple=1.5,
        stop_price=182.21,
        target_price=183.14,
        qty=21.0,
        entry_qty=21.0,
        active=True,
        order_id=stop_leg_id,
        last_updated=time.time(),
        entry_time=time.time(),
    )
    mgr.states[symbol] = state
    mgr._save_state()

    # Mock bridge instance
    bridge = MagicMock(spec=alpaca_bridge.AlpacaBridge)
    bridge.on_exit_order_update = alpaca_bridge.AlpacaBridge.on_exit_order_update.__get__(bridge, alpaca_bridge.AlpacaBridge)
    bridge.get_trailing_stop_manager = alpaca_bridge.AlpacaBridge.get_trailing_stop_manager.__get__(bridge, alpaca_bridge.AlpacaBridge)

    # Mock order and role info
    mock_order = MagicMock()
    mock_order.symbol = symbol
    mock_order.side = "sell"
    role_info = {"role": "STOP", "symbol": symbol, "parent_order_id": "parent_order_xom"}

    with patch("strategies.trailing_stop.get_trailing_manager", return_value=mgr):
        # Simulate stop leg fill event from stream
        handled = bridge.on_exit_order_update(
            order_id=stop_leg_id,
            cumulative_filled_qty=21.0,
            average_fill_price=182.14,
            execution_id="exec_stop_xom_001",
            status="filled",
            event_name="fill",
            order=mock_order,
            role_info=role_info,
        )

        assert handled is True
        # State active should immediately be False (transitioned to pending exit)
        assert mgr.states[symbol].active is False
        assert any(symbol in k for k in mgr.pending_exits.keys())

def test_symbol_loss_cooldown_deduplication():
    symbol = "XOM_TEST"
    now = datetime.now(timezone.utc)

    _RECENT_SYMBOL_LOSS_TIMESTAMPS.pop(symbol, None)

    # First registration should register
    register_symbol_loss(symbol, now)
    assert _RECENT_SYMBOL_LOSS_TIMESTAMPS[symbol] == now

    # Second registration 5 seconds later should be skipped (debounced)
    later_5s = now + timedelta(seconds=5)
    register_symbol_loss(symbol, later_5s)
    # Timestamp should remain original (not overwritten)
    assert _RECENT_SYMBOL_LOSS_TIMESTAMPS[symbol] == now

    # Registration 70s later should update
    later_70s = now + timedelta(seconds=70)
    register_symbol_loss(symbol, later_70s)
    assert _RECENT_SYMBOL_LOSS_TIMESTAMPS[symbol] == later_70s
