"""
test_minervini_sizing.py — offline tests for the M3 Minervini 1R risk sizing.

Covers ONLY the bridge-side, paper-only, default-OFF 1R position sizing that the
M3 milestone adds to ibkr_bridge.py:

    IBKRBridge._minervini_risk_sized_qty(symbol, signal, entry_price, notional_qty) -> int
    execute_signal() BUY-branch wiring (shrink-only; skip on a VALID risk_qty == 0)

Invariants under test:
  * BUY-ONLY and DOUBLE-GATED (MINERVINI_OVERLAY_ENABLED AND MINERVINI_SIZING_ENABLED);
  * SHRINK-ONLY — the result never exceeds the notional qty already capped by
    MAX_POSITION_PCT / MAX_TRADE_VALUE in _calc_quantity;
  * FAIL OPEN — returns the notional qty UNCHANGED on any missing/invalid/too-far/
    NaN/inf stop, non-positive entry, insufficient data, or exception;
  * a VALID stop whose per-share risk is too wide for >= 1 share returns 0, and the
    BUY is SKIPPED (no qty=0 order; execute_signal returns False);
  * the SELL / close path never consults sizing and still executes.

Fully offline + deterministic. The M0 detector layer (minervini.evaluate_entry /
minervini.minervini_stop_price) and the data layer (data_manager.fetch_ohlcv) are
mocked so these tests isolate the SIZING math + wiring, not the (separately tested)
detectors. One test exercises the REAL minervini_stop_price anchor math end to end.

Run with either:
    python -m unittest test_minervini_sizing -v
    python test_minervini_sizing.py
"""
import types
import unittest
from unittest import mock

import config
import data_manager
import minervini
import risk_state

# Reuse the fully offline fake-ib_insync harness (FakeIB + bridge factory).
from test_ibkr_fill_flow import (
    _BridgeBase, make_bridge, fill_parent_full, fill_market_only, placed_orders,
)


# A unique sentinel so a test can assert the sizer passed the fetched frame
# straight into the evaluator without caring about its contents.
_SENTINEL_DF = object()


def _signal(action="BUY", symbol="AAPL", confidence=0.7, price=100.0):
    """Minimal Signal stand-in; the sizer reads only .action (entry comes in as a
    parameter, not from the signal)."""
    return types.SimpleNamespace(
        symbol=symbol, action=action, confidence=confidence, price=price,
    )


def _verdict(pivot_low):
    """Minimal MinerviniVerdict stand-in — the sizer reads only .pivot_low."""
    return types.SimpleNamespace(pivot_low=pivot_low, stage2_ok=True, reasons=[])


def _entry_lmt_orders(ib):
    """The BUY entry limit order(s) placed (orderRef-carrying LMT, action BUY)."""
    return [o for o in placed_orders(ib)
            if str(getattr(o, "action", "")).upper() == "BUY"
            and str(getattr(o, "orderType", "")).upper() == "LMT"
            and getattr(o, "orderRef", "")]


# ── 1. _minervini_risk_sized_qty: the sizing helper in isolation ─────────────
class TestRiskSizedQtyHelper(_BridgeBase):
    """Direct tests on the helper, with the M0 detector + data layers mocked."""

    def _enable(self, overlay=True, sizing=True):
        mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", overlay).start()
        mock.patch.object(config, "MINERVINI_SIZING_ENABLED", sizing).start()

    def _params(self, risk_usd=25.0, max_dist=0.10):
        mock.patch.object(config, "MINERVINI_RISK_PER_TRADE_USD", risk_usd).start()
        mock.patch.object(config, "MINERVINI_MAX_STOP_DISTANCE_PCT", max_dist).start()

    def _mock(self, pivot_low=95.0, stop=95.0):
        """Patch the data + evaluator + stop layer. Returns (fetch, ev, stopfn)."""
        fetch = mock.patch.object(
            data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        ev = mock.patch.object(
            minervini, "evaluate_entry", return_value=_verdict(pivot_low)).start()
        stopfn = mock.patch.object(
            minervini, "minervini_stop_price", return_value=stop).start()
        return fetch, ev, stopfn

    # ── default-off / switch gating ─────────────────────────────────────────
    def test_both_switches_off_returns_notional_no_fetch(self):
        br = make_bridge()
        self._enable(overlay=False, sizing=False)
        fetch, ev, _ = self._mock(stop=50.0)  # would shrink hard IF consulted
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10)
        fetch.assert_not_called()
        ev.assert_not_called()

    def test_overlay_on_sizing_off_returns_notional_no_fetch(self):
        br = make_bridge()
        self._enable(overlay=True, sizing=False)
        fetch, ev, _ = self._mock(stop=50.0)
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10)
        fetch.assert_not_called()
        ev.assert_not_called()

    # ── valid shrink / never-increase / caps still bind ─────────────────────
    def test_valid_tight_stop_shrinks(self):
        br = make_bridge()
        self._enable()
        self._params(risk_usd=25.0, max_dist=0.10)
        self._mock(stop=95.0)  # entry 100 -> risk/share 5 -> risk_qty 5
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 5)

    def test_valid_loose_stop_never_increases(self):
        br = make_bridge()
        self._enable()
        self._params(risk_usd=25.0, max_dist=0.10)
        self._mock(stop=99.0)  # risk/share 1 -> risk_qty 25, but notional 3 wins
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 3), 3)

    def test_caps_bind_when_risk_qty_huge(self):
        br = make_bridge()
        self._enable()
        self._params(risk_usd=1000.0, max_dist=0.50)
        self._mock(stop=99.5)  # risk/share 0.5 -> risk_qty 2000, notional 4 wins
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 4), 4)

    # ── BUY-only guard ──────────────────────────────────────────────────────
    def test_non_buy_actions_never_size(self):
        br = make_bridge()
        self._enable()
        self._params(risk_usd=25.0, max_dist=0.10)
        # stop 97 -> entry 100 -> risk/share 3 -> risk_qty 8 (a real shrink IF a BUY).
        fetch, ev, _ = self._mock(stop=97.0)
        for action in ("SELL", "sell", "HOLD", "hold", "", "FOO", None):
            self.assertEqual(
                br._minervini_risk_sized_qty("AAPL", _signal(action), 100.0, 10), 10,
                f"non-BUY action {action!r} must never resize an order",
            )
        fetch.assert_not_called()  # no non-BUY action ever reached the data layer
        # ... while a BUY (case-insensitively) DOES size and reaches the data layer.
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal(" buy "), 100.0, 10), 8)
        fetch.assert_called()

    # ── fail-open: missing / invalid / far / NaN stop, bad entry, errors ────
    def test_pivot_none_fails_open(self):
        br = make_bridge()
        self._enable()
        self._params()
        mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(minervini, "evaluate_entry", return_value=_verdict(None)).start()
        mock.patch.object(minervini, "minervini_stop_price", return_value=None).start()
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10)

    def test_stop_none_fails_open(self):
        br = make_bridge()
        self._enable()
        self._params()
        self._mock(pivot_low=95.0, stop=None)
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10)

    def test_stop_at_or_above_entry_fails_open(self):
        br = make_bridge()
        self._enable()
        self._params()
        for stop in (100.0, 105.0):
            with mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF), \
                 mock.patch.object(minervini, "evaluate_entry", return_value=_verdict(stop)), \
                 mock.patch.object(minervini, "minervini_stop_price", return_value=stop):
                self.assertEqual(
                    br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10,
                    f"stop {stop} >= entry must fail open (no shrink)",
                )

    def test_far_stop_blanks_sizing(self):
        br = make_bridge()
        self._enable()
        self._params(risk_usd=25.0, max_dist=0.10)
        self._mock(stop=85.0)  # risk/share 15 > entry*0.10=10 -> blank (no shrink)
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10)

    def test_fetch_failure_fails_open(self):
        br = make_bridge()
        self._enable()
        self._params()
        mock.patch.object(
            data_manager, "fetch_ohlcv", side_effect=RuntimeError("no data")).start()
        ev = mock.patch.object(minervini, "evaluate_entry").start()
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10)
        ev.assert_not_called()

    def test_evaluate_failure_fails_open(self):
        br = make_bridge()
        self._enable()
        self._params()
        mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(
            minervini, "evaluate_entry", side_effect=ValueError("bad frame")).start()
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10)

    def test_stop_price_failure_fails_open(self):
        br = make_bridge()
        self._enable()
        self._params()
        mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(minervini, "evaluate_entry", return_value=_verdict(95.0)).start()
        mock.patch.object(
            minervini, "minervini_stop_price", side_effect=TypeError("bad pivot")).start()
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10)

    def test_non_positive_or_nonfinite_entry_fails_open(self):
        br = make_bridge()
        self._enable()
        self._params()
        self._mock(stop=95.0)
        for entry in (0.0, -5.0, float("nan"), float("inf")):
            self.assertEqual(
                br._minervini_risk_sized_qty("AAPL", _signal("BUY"), entry, 10), 10,
                f"invalid entry {entry} must fail open (no shrink)",
            )

    def test_nan_inf_stop_fails_open(self):
        br = make_bridge()
        self._enable()
        self._params()
        for stop in (float("nan"), float("inf")):
            with mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF), \
                 mock.patch.object(minervini, "evaluate_entry", return_value=_verdict(95.0)), \
                 mock.patch.object(minervini, "minervini_stop_price", return_value=stop):
                self.assertEqual(
                    br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 10,
                    f"non-finite stop {stop} must fail open (no shrink)",
                )

    # ── notional guards / valid-zero skip ───────────────────────────────────
    def test_notional_qty_zero_or_negative_returns_input_no_fetch(self):
        br = make_bridge()
        self._enable()
        self._params()
        fetch, _, _ = self._mock(stop=95.0)
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 0), 0)
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, -3), -3)
        fetch.assert_not_called()

    def test_valid_risk_qty_zero_returns_zero(self):
        # entry 300, stop 273 -> risk/share 27 (<= 300*0.10=30, in range),
        # budget 25 -> risk_qty floor(25/27) = 0 -> min(notional, 0) = 0 (skip).
        br = make_bridge()
        self._enable()
        self._params(risk_usd=25.0, max_dist=0.10)
        self._mock(stop=273.0)
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 300.0, 1), 0)

    # ── real anchor math (minervini_stop_price NOT mocked) ──────────────────
    def test_real_stop_price_arithmetic(self):
        # Real minervini_stop_price: pivot_low 95, buffer 0.005 -> stop ~94.53
        # (95*0.995); entry 100 -> risk/share ~5.47; budget 25 -> risk_qty 4;
        # notional 10 -> 4. (risk_qty is 4 whether the stop rounds to 94.52 or 94.53.)
        br = make_bridge()
        self._enable()
        self._params(risk_usd=25.0, max_dist=0.10)
        mock.patch.object(config, "MINERVINI_STOP_BUFFER_PCT", 0.005).start()
        mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(minervini, "evaluate_entry", return_value=_verdict(95.0)).start()
        # NOTE: minervini_stop_price is intentionally NOT mocked here.
        self.assertEqual(br._minervini_risk_sized_qty("AAPL", _signal("BUY"), 100.0, 10), 4)


# ── 2. execute_signal BUY-branch wiring (end to end via FakeIB fills) ────────
class TestSizingThroughExecuteSignal(_BridgeBase):
    """The shrink/skip is wired into the BUY branch right after _calc_quantity."""

    def setUp(self):
        super().setUp()
        # Market-hours gate is irrelevant to sizing; disable so the BUY is not
        # blocked outside RTH (restored by the base's mock.patch.stopall cleanup).
        mock.patch.object(config, "MARKET_HOURS_GATE_ENABLED", False).start()

    def _wire(self, br, price=100.0, cash=100000.0, position=0.0):
        br.get_price = lambda *a, **k: price                       # type: ignore
        br.get_cash = lambda: cash                                 # type: ignore
        br.get_position = lambda symbol: position                  # type: ignore
        br.has_working_order = lambda symbol, action=None: False   # type: ignore
        mock.patch.object(risk_state, "can_open_more", return_value=True).start()

    def _enable_sizing(self, risk_usd=25.0, max_dist=0.10):
        mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", True).start()
        mock.patch.object(config, "MINERVINI_SIZING_ENABLED", True).start()
        # Stage-2 hard block stays OFF (default) so ONLY sizing is exercised.
        mock.patch.object(config, "MINERVINI_RISK_PER_TRADE_USD", risk_usd).start()
        mock.patch.object(config, "MINERVINI_MAX_STOP_DISTANCE_PCT", max_dist).start()

    def test_buy_order_qty_is_shrunk(self):
        br = make_bridge(on_wait=fill_parent_full)
        self._wire(br, price=50.0, cash=100000.0)   # notional = int(500/50) = 10
        self._enable_sizing(risk_usd=25.0, max_dist=0.10)
        mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(minervini, "evaluate_entry", return_value=_verdict(47.0)).start()
        # entry 50, stop 47 -> risk/share 3 -> risk_qty floor(25/3) = 8; min(10, 8) = 8.
        mock.patch.object(minervini, "minervini_stop_price", return_value=47.0).start()

        ok = br.execute_signal(_signal("BUY", price=50.0))

        self.assertTrue(ok)
        entries = _entry_lmt_orders(br.ib)
        self.assertTrue(entries, "expected a BUY entry limit order")
        self.assertEqual(float(entries[0].totalQuantity), 8.0,
                         "the placed BUY qty must be the 1R-shrunk size")

    def test_buy_skipped_when_valid_risk_qty_zero(self):
        br = make_bridge(on_wait=fill_parent_full)
        self._wire(br, price=300.0, cash=100000.0)  # notional = int(500/300) = 1
        self._enable_sizing(risk_usd=25.0, max_dist=0.10)
        mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(minervini, "evaluate_entry", return_value=_verdict(273.0)).start()
        # entry 300, stop 273 -> risk/share 27 -> risk_qty 0 -> skip (no order).
        mock.patch.object(minervini, "minervini_stop_price", return_value=273.0).start()

        with mock.patch.object(risk_state, "record_trade") as rec:
            ok = br.execute_signal(_signal("BUY", price=300.0))

        self.assertFalse(ok, "a valid-but-too-wide 1R risk must skip the BUY")
        self.assertEqual(_entry_lmt_orders(br.ib), [], "no entry order may be placed")
        self.assertEqual(rec.call_count, 0, "a skipped BUY must not record a trade")

    def test_sizing_disabled_qty_matches_calc_quantity(self):
        br = make_bridge(on_wait=fill_parent_full)
        self._wire(br, price=50.0, cash=100000.0)
        # Overlay OFF entirely (default) -> sizing is a complete no-op; the placed
        # qty must equal _calc_quantity and the data layer is never consulted.
        fetch = mock.patch.object(
            data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()

        ok = br.execute_signal(_signal("BUY", price=50.0))

        self.assertTrue(ok)
        expected = br._calc_quantity(50.0, 100000.0)  # 10
        entries = _entry_lmt_orders(br.ib)
        self.assertTrue(entries, "expected a BUY entry limit order")
        self.assertEqual(float(entries[0].totalQuantity), float(expected))
        fetch.assert_not_called()  # sizing never consulted when the overlay is off


# ── 3. SELL / close regression: sizing must never touch a close ──────────────
class TestSellCloseUnaffectedBySizing(_BridgeBase):
    """A SELL that closes a long must still execute with sizing fully enabled, and
    the sizing path (fetch/evaluate/stop) must never be consulted on the close —
    the close path never routes through the entry sizing seam."""

    def test_long_close_executes_and_sizing_not_consulted(self):
        br = make_bridge(on_wait=fill_market_only)
        risk_state.snapshot_start_of_day_equity(100000.0)
        br.ib.account_values = [("NetLiquidation", "USD", 100000.0)]
        br.get_price = lambda *a, **k: 100.0                       # type: ignore
        br.get_cash = lambda: 100000.0                             # type: ignore
        br.has_working_order = lambda symbol, action=None: False   # type: ignore
        br.get_position = lambda symbol: 100.0                     # long -> SELL closes it
        mock.patch.object(risk_state, "can_open_more", return_value=True).start()
        mock.patch.object(config, "MARKET_HOURS_GATE_ENABLED", False).start()

        # Sizing fully ON -> would shrink/skip a BUY, must NOT touch this close.
        mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", True).start()
        mock.patch.object(config, "MINERVINI_SIZING_ENABLED", True).start()
        fetch = mock.patch.object(
            data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        ev = mock.patch.object(
            minervini, "evaluate_entry", return_value=_verdict(95.0)).start()
        stopfn = mock.patch.object(
            minervini, "minervini_stop_price", return_value=95.0).start()

        ok = br.execute_signal(
            types.SimpleNamespace(symbol="AAPL", action="SELL", confidence=0.7))

        self.assertTrue(ok, "a long close must execute regardless of sizing")
        self.assertTrue(
            any(str(getattr(o, "orderType", "")) == "MKT" for o in placed_orders(br.ib)),
            "expected a market close order for the SELL",
        )
        # The close path never consults the entry sizing overlay.
        fetch.assert_not_called()
        ev.assert_not_called()
        stopfn.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
