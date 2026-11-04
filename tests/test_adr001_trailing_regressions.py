"""Regression coverage for observed trailing-stop failures using a fake broker."""
from unittest.mock import MagicMock

import pytest

from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState


def make_state(side, *, last_updated=0):
    return TrailingStopState(
        symbol="EXAMPLE", side=side, entry_price=100.0, peak_price=100.0,
        atr=1.0, stop_price=98.0 if side == "BUY" else 102.0,
        trail_multiple=1.5, last_updated=last_updated, order_id="fake-stop", qty=1,
    )


def test_short_breakeven_locks_configured_profit_without_widening(tmp_path, monkeypatch):
    import config
    monkeypatch.setattr(config, "BREAKEVEN_LOCK_ENABLED", True)
    monkeypatch.setattr(config, "MR_BREAKEVEN_PROFIT_ATR", 1.5)
    monkeypatch.setattr(config, "MR_BREAKEVEN_LOCK_ATR", 0.1)
    manager = DynamicTrailingStopManager(tmp_path / "state.json")
    state = manager.states["EXAMPLE"] = make_state("SELL")
    broker = MagicMock()
    broker.modify_stop_order.return_value = True

    manager.update_stop("EXAMPLE", current_price=98.4, bridge=broker, dry_run=False)

    assert state.stop_price == pytest.approx(99.9)
    broker.modify_stop_order.assert_called_once_with("fake-stop", pytest.approx(99.9))
    assert state.confirmed_broker_stop_price == pytest.approx(99.9)
