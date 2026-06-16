"""
test_ibkr_fill_flow.py - Phase 2 integration tests for the fill-driven order
path in ibkr_bridge.py (H4-H9), exercised with FAKE ib_insync objects.

Fully offline and deterministic: no live IBKR, no network. A small FakeIB +
FakeTrade simulate placeOrder/openTrades/waitOnUpdate so the real bridge logic
(await-fill, protective-child verify/resize, emergency flatten, close
escalation, record-only-on-fill, duplicate-on-restart guard) runs end to end.

These back the Phase-2 go/no-go gates from
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md:
  * partial-fill simulate  -> child/exit qty matches the actual filled qty
  * order reject simulate   -> handled as REJECTED, record_trade NOT called
  * restart mid-flight      -> duplicate order NOT placed

Run with:
    python -m unittest test_ibkr_fill_flow -v
"""
import asyncio
import types
import unittest
from unittest import mock


# ib_insync touches the event loop at import time; prepare it first (as main.py
# and test_live_readiness.py do).
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import config              # noqa: E402
import order_audit         # noqa: E402
import order_exec as ox    # noqa: E402
import risk_state          # noqa: E402
import ibkr_bridge         # noqa: E402
from ib_insync import LimitOrder, MarketOrder, Order  # noqa: E402  (real order objects)

ACTIVE = ox.ACTIVE_STATES


# ── Fake ib_insync surface ───────────────────────────────────────────────────
class FakeContract:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeOrder:
    """Stands in for ib_insync Order on the bracket/duplicate paths."""
    def __init__(self, action="BUY", orderType="LMT", totalQuantity=0.0, orderRef=""):
        self.action = action
        self.orderType = orderType
        self.totalQuantity = totalQuantity
        self.orderRef = orderRef
        self.parentId = 0
        self.orderId = 0
        self.transmit = True


class FakeStatus:
    def __init__(self, status="PreSubmitted", filled=0.0, remaining=0.0, avgFillPrice=0.0):
        self.status = status
        self.filled = filled
        self.remaining = remaining
        self.avgFillPrice = avgFillPrice
        self.parentId = 0


class FakeTrade:
    def __init__(self, contract, order, status):
        self.contract = contract
        self.order = order
        self.orderStatus = status


class _Client:
    def __init__(self):
        self._n = 9000

    def getReqId(self):
        self._n += 1
        return self._n


class FakeIB:
    """Minimal ib_insync.IB replacement. `on_wait(ib)` is invoked on every
    waitOnUpdate, letting a test evolve order state (simulate fills)."""

    def __init__(self, on_wait=None, cancel_disabled=False):
        self._all = []          # every FakeTrade placed
        self.cancelled = []
        self.bracket_calls = 0
        self.on_wait = on_wait
        # When True, cancelOrder records the request but the order STAYS active
        # (simulates a child that cannot be cancelled) so the safety gate must
        # fall back to an emergency flatten.
        self.cancel_disabled = cancel_disabled
        self.client = _Client()

    # -- waiting --
    def waitOnUpdate(self, timeout=0):
        if self.on_wait:
            self.on_wait(self)

    def sleep(self, t=0):
        # The bridge uses sleep only to let refreshes settle; don't advance fills.
        return None

    # -- orders --
    def placeOrder(self, contract, order):
        tot = float(getattr(order, "totalQuantity", 0) or 0)
        st = FakeStatus(status="PreSubmitted", filled=0.0, remaining=tot)
        tr = FakeTrade(contract, order, st)
        self._all.append(tr)
        return tr

    def cancelOrder(self, order):
        self.cancelled.append(order)
        if self.cancel_disabled:
            return  # order stays active -> exercises the flatten fallback
        for t in self._all:
            if t.order is order:
                t.orderStatus.status = "Cancelled"

    def working_exit_qtys(self, symbol, exit_action="SELL"):
        """Helper for assertions: qtys of still-working exit-side orders."""
        return [float(getattr(t.order, "totalQuantity", 0))
                for t in self.openTrades()
                if str(getattr(t.contract, "symbol", "")).upper() == symbol.upper()
                and str(getattr(t.order, "action", "")).upper() == exit_action]

    def reqAllOpenOrders(self):
        return None

    def openTrades(self):
        return [t for t in self._all if t.orderStatus.status in ACTIVE]

    def positions(self):
        return []

    def bracketOrder(self, action, quantity, limitPrice, takeProfitPrice, stopLossPrice):
        self.bracket_calls += 1
        exit_a = "SELL" if action == "BUY" else "BUY"
        return [
            FakeOrder(action, "LMT", quantity),    # parent
            FakeOrder(exit_a, "LMT", quantity),    # take-profit
            FakeOrder(exit_a, "STP", quantity),    # protective stop child
        ]


# ── fill scripts (passed as FakeIB.on_wait) ──────────────────────────────────
def _fill_parent(ib, qty=None, status="Filled", avg=100.0):
    """Fill the open PARENT (the LMT order carrying a deterministic orderRef)."""
    for t in ib.openTrades():
        o = t.order
        if str(getattr(o, "orderType", "")) == "LMT" and getattr(o, "orderRef", ""):
            tot = float(getattr(o, "totalQuantity", 0) or 0)
            f = tot if qty is None else float(qty)
            t.orderStatus.status = status
            t.orderStatus.filled = f
            t.orderStatus.remaining = max(tot - f, 0.0)
            t.orderStatus.avgFillPrice = avg
            return


def fill_parent_full(ib):
    _fill_parent(ib, qty=None, status="Filled")


def partial_parent(qty):
    def hook(ib):
        _fill_parent(ib, qty=qty, status="Submitted")  # stays active -> times out partial
    return hook


def fill_market_only(ib):
    """Fill MKT orders (close escalation); leave LMT working."""
    for t in ib.openTrades():
        o = t.order
        if str(getattr(o, "orderType", "")) == "MKT":
            q = float(getattr(o, "totalQuantity", 0) or 0)
            t.orderStatus.status = "Filled"
            t.orderStatus.filled = q
            t.orderStatus.remaining = 0.0
            t.orderStatus.avgFillPrice = 50.0


def reject_parent(ib):
    for t in ib.openTrades():
        o = t.order
        if str(getattr(o, "orderType", "")) == "LMT" and getattr(o, "orderRef", ""):
            t.orderStatus.status = "Rejected"
            t.orderStatus.filled = 0.0


def make_bridge(on_wait=None):
    br = ibkr_bridge.IBKRBridge()
    br.ib = FakeIB(on_wait=on_wait)
    # Avoid contract qualification (no network); symbols map straight through.
    br._contract = lambda symbol: FakeContract(str(symbol).upper().strip())  # type: ignore
    return br


def placed_orders(ib):
    return [t.order for t in ib._all]


class _BridgeBase(unittest.TestCase):
    def setUp(self):
        # Keep audit logging side-effect-free during tests.
        self._audit = mock.patch.object(order_audit, "log_event")
        self._audit.start()
        self.addCleanup(self._audit.stop)
        # Short, fast, deterministic await budget.
        for name, val in {"ORDER_FILL_TIMEOUT_SECONDS": 1.0, "ORDER_POLL_SECONDS": 0.25,
                          "CLOSE_MAX_ATTEMPTS": 3}.items():
            mock.patch.object(config, name, val).start()
        self.addCleanup(mock.patch.stopall)


# ── 1. _await_order_outcome: fill vs accepted vs reject vs timeout (H4/H5) ────
class TestAwaitOutcome(_BridgeBase):
    def _trade(self, status="PreSubmitted", filled=0.0, remaining=100.0):
        return FakeTrade(FakeContract("AAPL"), FakeOrder("BUY", "LMT", 100, "ref"),
                         FakeStatus(status, filled, remaining))

    def test_fills_when_status_becomes_filled(self):
        tr = self._trade()
        br = make_bridge(on_wait=lambda ib: setattr(tr.orderStatus, "status", "Filled")
                         or setattr(tr.orderStatus, "filled", 100.0)
                         or setattr(tr.orderStatus, "remaining", 0.0))
        res = br._await_order_outcome(tr)
        self.assertEqual(res.outcome, ox.FILLED)
        self.assertEqual(res.filled, 100.0)

    def test_timeout_when_never_fills(self):
        tr = self._trade()
        br = make_bridge(on_wait=None)  # nothing ever changes
        res = br._await_order_outcome(tr)
        self.assertEqual(res.outcome, ox.TIMEOUT)

    def test_partial_then_timeout_is_partial(self):
        tr = self._trade()
        br = make_bridge(on_wait=lambda ib: setattr(tr.orderStatus, "filled", 30.0)
                         or setattr(tr.orderStatus, "remaining", 70.0))
        res = br._await_order_outcome(tr)
        self.assertEqual(res.outcome, ox.PARTIALLY_FILLED)
        self.assertEqual(res.filled, 30.0)

    def test_rejected(self):
        tr = self._trade()
        br = make_bridge(on_wait=lambda ib: setattr(tr.orderStatus, "status", "Rejected"))
        res = br._await_order_outcome(tr)
        self.assertEqual(res.outcome, ox.REJECTED)


# ── 2. Open bracket: confirm fill + verify protective child (H4/H7) ───────────
class TestOpenBracket(_BridgeBase):
    def test_full_fill_with_live_stop(self):
        mock.patch.object(config, "USE_TRAILING_EXIT", False).start()
        br = make_bridge(on_wait=fill_parent_full)
        res = br._place_open_bracket("AAPL", "BUY", 100, 100.0, 0.7)
        self.assertEqual(res.outcome, ox.FILLED)
        self.assertTrue(res.protective_ok)
        self.assertFalse(res.aborted)
        self.assertTrue(ox.should_record_trade(res))

    def test_rejected_open_does_not_record(self):
        mock.patch.object(config, "USE_TRAILING_EXIT", False).start()
        br = make_bridge(on_wait=reject_parent)
        res = br._place_open_bracket("AAPL", "BUY", 100, 100.0, 0.7)
        self.assertEqual(res.outcome, ox.REJECTED)
        self.assertFalse(ox.should_record_trade(res))

    def test_partial_fill_handles_both_stop_and_take_profit(self):
        # GO/NO-GO: on a partial fill, NEITHER the protective stop NOR the
        # take-profit limit child may exceed the actual filled qty (H6). The
        # native bracket leaves an oversized STP *and* an oversized TP LMT.
        mock.patch.object(config, "USE_TRAILING_EXIT", False).start()
        br = make_bridge(on_wait=partial_parent(30))
        res = br._place_open_bracket("AAPL", "BUY", 100, 100.0, 0.7)
        self.assertEqual(res.outcome, ox.PARTIALLY_FILLED)
        self.assertEqual(res.filled, 30)
        self.assertTrue(res.protective_ok)
        self.assertFalse(res.aborted)
        # A protective STP sized to exactly the 30 filled shares was placed.
        stops = [o for o in placed_orders(br.ib)
                 if str(getattr(o, "orderType", "")) == "STP"
                 and float(getattr(o, "totalQuantity", 0)) == 30]
        self.assertTrue(stops, "expected a resized protective STP of 30 shares")
        # NO working SELL exit child may exceed the filled qty (no accidental short).
        self.assertTrue(all(q <= 30 for q in br.ib.working_exit_qtys("AAPL")),
                        f"oversized exit child survived: {br.ib.working_exit_qtys('AAPL')}")
        # The original oversized take-profit LMT (100) was cancelled.
        self.assertTrue(any(str(getattr(o, "orderType", "")) == "LMT"
                            and str(getattr(o, "action", "")) == "SELL"
                            and float(getattr(o, "totalQuantity", 0)) == 100
                            for o in br.ib.cancelled),
                        "expected the oversized take-profit LMT to be cancelled")

    def test_partial_fill_cannot_cancel_children_is_flattened(self):
        # If oversized exit children cannot be removed, the only safe action is to
        # flatten and abort (never hold a position with an oversell trap).
        mock.patch.object(config, "USE_TRAILING_EXIT", False).start()
        br = make_bridge(on_wait=partial_parent(30))
        br.ib.cancel_disabled = True  # children refuse to cancel
        with mock.patch.object(br, "_market_close_symbol") as flatten:
            res = br._place_open_bracket("AAPL", "BUY", 100, 100.0, 0.7)
        flatten.assert_called_once()
        self.assertTrue(res.aborted)
        self.assertFalse(res.protective_ok)
        self.assertFalse(ox.should_record_trade(res))

    def test_trailing_full_fill_protected(self):
        mock.patch.object(config, "USE_TRAILING_EXIT", True).start()
        br = make_bridge(on_wait=fill_parent_full)
        res = br._place_open_bracket("AAPL", "BUY", 100, 100.0, 0.7)
        self.assertEqual(res.outcome, ox.FILLED)
        self.assertTrue(res.protective_ok)


# ── 3. Missing protective stop after a fill -> emergency flatten (H7) ─────────
class TestUnprotectedFlatten(_BridgeBase):
    def test_fill_without_confirmable_stop_is_flattened_and_aborted(self):
        br = make_bridge()  # openTrades() will be empty -> no child found
        result = ox.OrderResult(ox.FILLED, status="Filled", filled=100, remaining=0, avg_fill_price=100.0)
        parent = FakeOrder("BUY", "LMT", 100, "ref")
        with mock.patch.object(br, "_place_protective_stop", return_value=False) as place, \
             mock.patch.object(br, "_market_close_symbol") as flatten:
            out = br._verify_or_protect("AAPL", parent, result, stop_price=96.0, exit_action="SELL")
        self.assertTrue(place.called)
        flatten.assert_called_once()
        self.assertTrue(out.aborted)
        self.assertFalse(out.protective_ok)
        self.assertFalse(ox.should_record_trade(out))  # an aborted entry never records

    def test_market_close_cancels_resting_children_before_going_flat(self):
        # A leftover resting SELL stop, if it fires AFTER we go flat, would open a
        # short. Emergency flatten must cancel all symbol orders first.
        br = make_bridge(on_wait=fill_market_only)
        br.get_position = lambda symbol: 100.0  # type: ignore
        stop = FakeOrder("SELL", "STP", 100, "old")
        br.ib._all.append(FakeTrade(FakeContract("AAPL"), stop, FakeStatus("Submitted", 0.0, 100.0)))
        br._market_close_symbol("AAPL")
        self.assertIn(stop, br.ib.cancelled, "resting stop must be cancelled before flattening")
        closes = [o for o in placed_orders(br.ib) if str(getattr(o, "orderType", "")) == "MKT"]
        self.assertTrue(closes, "expected a market close order")


# ── 4. Robust close: escalate marketable-limit -> market (H8) ────────────────
class TestRobustClose(_BridgeBase):
    def test_close_escalates_to_market_and_confirms(self):
        br = make_bridge(on_wait=fill_market_only)  # only a MKT order fills
        res = br._close_position("AAPL", "SELL", 100, 50.0, "Close long")
        self.assertEqual(res.outcome, ox.FILLED)
        self.assertEqual(res.filled, 100)
        orders = placed_orders(br.ib)
        self.assertGreaterEqual(len(orders), 2, "should retry after the limit did not fill")
        self.assertEqual(str(getattr(orders[-1], "orderType", "")), "MKT",
                         "final escalation must be a market order")


# ── 5. execute_signal records a trade ONLY on a real fill (H4) ───────────────
class TestExecuteSignalRecording(_BridgeBase):
    def _wire(self, br):
        br.get_price = lambda *a, **k: 100.0          # type: ignore
        br.get_cash = lambda: 100000.0                # type: ignore
        br.get_position = lambda symbol: 0.0          # type: ignore
        br.has_working_order = lambda symbol, action=None: False  # type: ignore
        mock.patch.object(risk_state, "can_open_more", return_value=True).start()

    def _signal(self):
        return types.SimpleNamespace(symbol="AAPL", action="BUY", confidence=0.7)

    def test_fill_records_one_trade(self):
        br = make_bridge(on_wait=fill_parent_full)
        self._wire(br)
        with mock.patch.object(risk_state, "record_trade", return_value=1) as rec:
            ok = br.execute_signal(self._signal())
        self.assertTrue(ok)
        self.assertEqual(rec.call_count, 1)

    def test_acknowledged_but_unfilled_records_nothing(self):
        # The core H4 fix: an accepted-but-not-filled order must NOT record.
        br = make_bridge(on_wait=None)  # parent never fills -> TIMEOUT
        self._wire(br)
        with mock.patch.object(risk_state, "record_trade", return_value=1) as rec:
            br.execute_signal(self._signal())
        self.assertEqual(rec.call_count, 0)


# ── 6. No duplicate order on restart / mid-flight (H9) ───────────────────────
class TestDuplicateGuard(_BridgeBase):
    def test_existing_orderref_blocks_a_new_open(self):
        br = make_bridge()
        ref = ox.deterministic_order_ref(br._today(), "AAPL", "BUY")
        # Simulate a working order from a previous run already at the broker.
        br.ib._all.append(FakeTrade(
            FakeContract("AAPL"),
            FakeOrder("BUY", "LMT", 5, ref),
            FakeStatus("Submitted", 0.0, 5.0),
        ))
        self.assertTrue(br._duplicate_ref_working("AAPL", "BUY"))

        # And execute_signal short-circuits without placing a new bracket.
        br.get_price = lambda *a, **k: 100.0          # type: ignore
        br.get_cash = lambda: 100000.0                # type: ignore
        br.get_position = lambda symbol: 0.0          # type: ignore
        br.has_working_order = lambda symbol, action=None: False  # type: ignore
        mock.patch.object(risk_state, "can_open_more", return_value=True).start()
        with mock.patch.object(risk_state, "record_trade") as rec:
            ok = br.execute_signal(types.SimpleNamespace(symbol="AAPL", action="BUY", confidence=0.7))
        self.assertFalse(ok)
        self.assertEqual(br.ib.bracket_calls, 0)
        self.assertEqual(rec.call_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
