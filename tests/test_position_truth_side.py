"""Broker position direction must come from actual SDK values."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from alpaca.trading.enums import PositionSide
from alpaca_bridge import AlpacaBridge


@pytest.mark.parametrize("side,qty,expected", [
    (PositionSide.LONG, "2", "BUY"),
    (PositionSide.SHORT, "-2", "SELL"),
    ("long", "2", "BUY"),
    ("short", "-2", "SELL"),
    (None, "2", "UNKNOWN"),
    ("unrecognized", "-2", "UNKNOWN"),
])
def test_position_truth_preserves_broker_direction(side, qty, expected):
    bridge = AlpacaBridge(mode="daytrade")
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._client.get_open_position.return_value = SimpleNamespace(
        side=side, qty=qty, avg_entry_price="100"
    )
    bridge._call_with_retry = lambda callback: callback()
    actual = bridge.get_position_truth("EXAMPLE")
    assert actual.side == expected
    assert actual.qty == float(qty)
