"""
test_phase3_protection.py - Phase 3 integration tests (C2, H19, H1, H3, startup).

Offline + deterministic, driving the real ibkr_bridge.py with the fake ib_insync
objects from test_ibkr_fill_flow.py. Covers the Phase-3 go/no-go gates:

  * server-side GTC + OCA protection placed AFTER the confirmed fill (C2/H19)
  * an independent hard -3% stop sized to the actual filled qty, priced off the
    ACTUAL avgFillPrice (not the signal price)
  * no exit-side SELL qty exceeds the filled qty (Phase-2 invariant preserved)
  * daily-loss kill-switch blocks NEW opens but still allows closes (H1)
  * startup protection invariant: unprotected long repaired; repair/avgCost
    failure halts new entries and alerts (3.6)

Run with:
    python -m unittest test_phase3_protection -v
"""
import types
import unittest
from unittest import mock

import config
import order_exec as ox
import risk_state
from test_ibkr_fill_flow import (
    _BridgeBase, FakeContract, FakeOrder, FakeStatus, FakeTrade, FakePosition,
    _fill_parent, fill_parent_full, fill_market_only, make_bridge, placed_orders,
)


def _sell_exits(ib):
    return [o for o in placed_orders(ib) if str(getattr(o, "action", "")) == "SELL"]


# ── 1. Server-side GTC / OCA protection (C2, H19) ────────────────────────────
class TestGtcOcaProtection(_BridgeBase):
    def setUp(self):
        super().setUp()
        mock.patch.object(config, "USE_TRAILING_EXIT", False).start()  # bracket: stop + TP
        mock.patch.object(config, "HARD_STOP_LOSS_PCT", 0.03).start()

    def test_protection_is_gtc_oca_with_hard_stop_at_filled_qty(self):
        br = make_bridge(on_wait=fill_parent_full)  # avg fill 100.0, full 100
        res = br._place_open_bracket("AAPL", "BUY", 100, 100.0, 0.7)
        self.assertEqual(res.outcome, ox.FILLED)
        self.assertTrue(res.protective_ok)

        exits = _sell_exits(br.ib)
        self.assertTrue(exits, "expected protective SELL exits after the fill")
        # Every exit is GTC (rests between the one-shot bot's runs).
        self.assertTrue(all(str(getattr(o, "tif", "")).upper() == "GTC" for o in exits))
        # Every exit shares ONE non-empty OCA group.
        groups = {str(getattr(o, "ocaGroup", "")) for o in exits}
        self.assertEqual(len(groups), 1)
        self.assertTrue(all(groups), "OCA group must be non-empty")
        # No exit exceeds the filled qty (no accidental-short trap).
        self.assertTrue(all(float(getattr(o, "totalQuantity", 0)) <= 100 for o in exits))
        # An independent hard -3% STP at the filled qty, priced off basis 100 -> 97.0.
        hard = [o for o in exits if str(getattr(o, "orderType", "")) == "STP"
                and round(float(getattr(o, "auxPrice", 0)), 2) == 97.0]
        self.assertTrue(hard, "expected a hard -3% STP at 97.00")
        self.assertEqual(float(hard[0].totalQuantity), 100)

    def test_hard_stop_uses_actual_avg_fill_not_signal_price(self):
        # Signal price 100 but the ACTUAL avg fill is 50 -> hard stop must be 48.50.
        br = make_bridge(on_wait=lambda ib: _fill_parent(ib, avg=50.0))
        res = br._place_open_bracket("AAPL", "BUY", 100, 100.0, 0.7)
        self.assertEqual(res.outcome, ox.FILLED)
        prices = [round(float(getattr(o, "auxPrice", 0)), 2)
                  for o in _sell_exits(br.ib) if str(getattr(o, "orderType", "")) == "STP"]
        self.assertIn(48.50, prices)  # 50 * (1 - 0.03)


# ── 2. Daily-loss kill-switch (H1) ───────────────────────────────────────────
class TestDailyLossKillSwitch(_BridgeBase):
    def setUp(self):
        super().setUp()
        # Time-independence: these tests assert the daily-loss kill-switch's
        # decision on a BUY open, NOT the Phase-4 market-hours gate. Outside US
        # regular hours that gate would block the open first and mask the
        # kill-switch behavior. Disable it for THIS class only; restored by the
        # base's mock.patch.stopall cleanup. Never disabled in production.
        mock.patch.object(config, "MARKET_HOURS_GATE_ENABLED", False).start()

    def _wire(self, br, equity):
        br.ib.account_values = [("NetLiquidation", "USD", equity)]
        br.get_price = lambda *a, **k: 100.0          # type: ignore
        br.get_cash = lambda: 100000.0                # type: ignore
        br.has_working_order = lambda symbol, action=None: False  # type: ignore
        mock.patch.object(risk_state, "can_open_more", return_value=True).start()

    def test_breach_blocks_new_open(self):
        br = make_bridge(on_wait=fill_parent_full)
        risk_state.snapshot_start_of_day_equity(100000.0)   # baseline
        self._wire(br, 99000.0)                             # down 1000 > 150 cap
        br.get_position = lambda symbol: 0.0                # type: ignore
        with mock.patch.object(risk_state, "record_trade") as rec:
            ok = br.execute_signal(types.SimpleNamespace(symbol="AAPL", action="BUY", confidence=0.7))
        self.assertFalse(ok)
        self.assertEqual(rec.call_count, 0)
        self.assertEqual(placed_orders(br.ib), [], "no entry may be placed under a daily-loss halt")

    def test_breach_still_allows_close(self):
        br = make_bridge(on_wait=fill_market_only)
        risk_state.snapshot_start_of_day_equity(100000.0)
        self._wire(br, 99000.0)                             # breached
        br.get_position = lambda symbol: 100.0              # long -> SELL closes it
        ok = br.execute_signal(types.SimpleNamespace(symbol="AAPL", action="SELL", confidence=0.7))
        self.assertTrue(ok, "a close must still execute under a daily-loss halt")
        self.assertTrue(any(str(getattr(o, "orderType", "")) == "MKT" for o in placed_orders(br.ib)))

    def test_no_breach_allows_open(self):
        br = make_bridge(on_wait=fill_parent_full)
        risk_state.snapshot_start_of_day_equity(100000.0)
        self._wire(br, 100000.0)                            # flat, no loss
        br.get_position = lambda symbol: 0.0                # type: ignore
        with mock.patch.object(risk_state, "record_trade", return_value=1) as rec:
            ok = br.execute_signal(types.SimpleNamespace(symbol="AAPL", action="BUY", confidence=0.7))
        self.assertTrue(ok)
        self.assertEqual(rec.call_count, 1)


# ── 3. Startup protection invariant (plan 3.6) ───────────────────────────────
class TestStartupProtection(_BridgeBase):
    def setUp(self):
        super().setUp()
        mock.patch.object(config, "USE_TRAILING_EXIT", False).start()

    def test_unprotected_long_is_repaired(self):
        br = make_bridge()
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=100.0)]
        report = br.ensure_protective_stops()
        self.assertIn("AAPL", report["unprotected"])
        self.assertIn("AAPL", report["repaired"])
        self.assertFalse(br._halt_new_entries)
        self.assertTrue(ox.has_gtc_protective_stop("AAPL", 100, br._working_orders_plain()))

    def test_already_protected_long_is_left_alone(self):
        br = make_bridge()
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=100.0)]
        stop = FakeOrder("SELL", "STP", 100, "existing")
        stop.tif = "GTC"
        br.ib._all.append(FakeTrade(FakeContract("AAPL"), stop, FakeStatus("Submitted", 0.0, 100.0)))
        report = br.ensure_protective_stops()
        self.assertEqual(report["unprotected"], [])
        self.assertEqual(report["repaired"], [])
        self.assertFalse(br._halt_new_entries)

    def test_repair_failure_halts_new_entries(self):
        br = make_bridge()
        br.ib.reject_stops = True  # broker refuses the protective stop
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=100.0)]
        report = br.ensure_protective_stops()
        self.assertIn("AAPL", report["failed"])
        self.assertTrue(br._halt_new_entries)

    def test_missing_avg_cost_halts_without_guessing(self):
        br = make_bridge()
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=0.0)]  # no valid basis
        report = br.ensure_protective_stops()
        self.assertIn("AAPL", report["failed"])
        self.assertTrue(br._halt_new_entries)
        # We must NOT guess a price -> nothing placed.
        self.assertEqual(placed_orders(br.ib), [])

    def test_halt_flag_blocks_subsequent_open(self):
        br = make_bridge(on_wait=fill_parent_full)
        br._halt_new_entries = True
        br.ib.account_values = [("NetLiquidation", "USD", 100000.0)]
        br.get_price = lambda *a, **k: 100.0          # type: ignore
        br.get_cash = lambda: 100000.0                # type: ignore
        br.get_position = lambda symbol: 0.0          # type: ignore
        br.has_working_order = lambda symbol, action=None: False  # type: ignore
        mock.patch.object(risk_state, "can_open_more", return_value=True).start()
        with mock.patch.object(risk_state, "record_trade") as rec:
            ok = br.execute_signal(types.SimpleNamespace(symbol="AAPL", action="BUY", confidence=0.7))
        self.assertFalse(ok)
        self.assertEqual(rec.call_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
