"""Regression: ORCL RECONCILE OVERFILL loop.

A RECONCILED/adopted state recorded qty=5 but the episode's own stop order filled 13
(5 then 8) and the broker ended flat. The overfill guard used to retain the pending
exit forever. It must now resolve ONLY when: broker verified flat, no working orders,
exactly one filled order, and that order is the episode's own stop/tp id. Anything
else (multi-order excess, working orders, foreign order id) must still fail closed.
"""
import pytest
from unittest.mock import patch

from strategies.trailing_stop import DynamicTrailingStopManager

STOP_ID = "c71a4f91-a98f-4655-a2b2-5b060d5e654b"


class FakeBridge:
    def __init__(self, orders, position=0.0, working=None):
        self._orders = {o["id"]: o for o in orders}
        self._position = position
        self._working = working if working is not None else []

    def get_position(self, symbol):
        return self._position

    def get_order_snapshot(self, order_id):
        return self._orders.get(str(order_id))

    def working_orders_plain(self):
        return self._working


def _order(oid, qty, cid="ORCL:sig1", px=125.0):
    return {
        "id": oid,
        "client_order_id": cid,
        "symbol": "ORCL",
        "side": "buy",
        "status": "filled",
        "type": "stop",
        "qty": qty,
        "filled_qty": qty,
        "filled_avg_price": px,
    }


@pytest.fixture
def mgr(tmp_path):
    m = DynamicTrailingStopManager(state_file=str(tmp_path / "trailing_state.json"))
    m.pending_exits.clear()
    m.states.clear()
    m.pending_exits[f"ORCL:{STOP_ID}"] = {
        "episode_id": f"ORCL:{STOP_ID}",
        "symbol": "ORCL",
        "side": "SELL",
        "qty": 5.0,
        "entry_price": 126.01,
        "stop_price": 125.89,
        "tp_order_id": None,
        "stop_order_id": STOP_ID,
        "last_stop_order_id": None,
        "is_validation": False,
        "entry_time": 1793808994.9,
        "signal_id": "sig1",
        "strategy": "RECONCILED",
        "attempts": 1,
    }
    return m


def test_single_episode_order_overfill_resolves_with_corrected_qty(mgr):
    bridge = FakeBridge([_order(STOP_ID, 13.0)])
    with patch("strategies.trade_journal.log_trade") as log_trade:
        res = mgr.reconcile_pending_exits(bridge)

    assert res.get("ORCL") != "EXCESS_EXIT_QTY_DISCREPANCY"
    assert not mgr.has_pending_exit("ORCL")
    log_trade.assert_called_once()
    kwargs = log_trade.call_args.kwargs
    assert kwargs["qty"] == 13
    assert "corrected" in kwargs["notes"]


def test_overfill_still_fails_closed_with_working_orders(mgr):
    bridge = FakeBridge([_order(STOP_ID, 13.0)], working=[{"symbol": "ORCL"}])
    with patch("strategies.trade_journal.log_trade") as log_trade:
        res = mgr.reconcile_pending_exits(bridge)

    assert res.get("ORCL") == "EXCESS_EXIT_QTY_DISCREPANCY"
    assert mgr.has_pending_exit("ORCL")
    log_trade.assert_not_called()


def test_overfill_still_fails_closed_when_order_is_not_episode_own_id(mgr):
    # Lineage matches via client_order_id, but the broker order id is not this episode's.
    bridge = FakeBridge([_order("some-other-order", 13.0)])
    mgr.pending_exits[f"ORCL:{STOP_ID}"]["last_stop_order_id"] = "some-other-order"
    # make it NOT one of the known ids by clearing the only ids that include it
    mgr.pending_exits[f"ORCL:{STOP_ID}"]["last_stop_order_id"] = None
    bridge._orders = {STOP_ID: _order("some-other-order", 13.0)}  # snapshot returns a different id
    with patch("strategies.trade_journal.log_trade") as log_trade:
        res = mgr.reconcile_pending_exits(bridge)

    assert res.get("ORCL") == "EXCESS_EXIT_QTY_DISCREPANCY"
    assert mgr.has_pending_exit("ORCL")
    log_trade.assert_not_called()


def test_multi_order_excess_still_fails_closed(mgr):
    mgr.pending_exits[f"ORCL:{STOP_ID}"]["qty"] = 12.0
    mgr.pending_exits[f"ORCL:{STOP_ID}"]["tp_order_id"] = "tp-1"
    bridge = FakeBridge([_order(STOP_ID, 8.0), _order("tp-1", 8.0)])
    with patch("strategies.trade_journal.log_trade") as log_trade:
        res = mgr.reconcile_pending_exits(bridge)

    assert res.get("ORCL") == "EXCESS_EXIT_QTY_DISCREPANCY"
    assert mgr.has_pending_exit("ORCL")
    log_trade.assert_not_called()
