"""
test_reconciliation.py - Phase 5A startup-reconciliation tests (H18).

Two layers, both fully offline and deterministic (no live IBKR, no network):

  1. The PURE reconciliation layer (reconciliation.py): build a broker-truth
     snapshot from plain position/working-order dicts -- protected vs.
     unprotected longs, duplicate orderRefs, orphan exit orders.
  2. The bridge method IBKRBridge.reconcile_startup_state(), driven through the
     real ibkr_bridge.py with the fake ib_insync objects from
     test_ibkr_fill_flow.py. Proves the method treats the BROKER as the source of
     truth, repairs an unprotected long via the existing Phase-3 path, audits the
     result, and -- the core safety invariant -- NEVER opens a new position.

Run with:
    python -m unittest test_reconciliation -v
"""
import unittest
from unittest import mock

import config
import order_exec as ox
import reconciliation as rec
from test_ibkr_fill_flow import (
    _BridgeBase, FakeContract, FakeOrder, FakeStatus, FakeTrade, FakePosition,
    make_bridge, placed_orders,
)


def _gtc_stop(symbol, qty):
    o = FakeOrder("SELL", "STP", qty, f"{symbol}-stop")
    o.tif = "GTC"
    return FakeTrade(FakeContract(symbol), o, FakeStatus("Submitted", 0.0, qty))


# ── 1. Pure snapshot: no positions -> clean ──────────────────────────────────
class TestPureSnapshotClean(unittest.TestCase):
    def test_no_positions_is_clean(self):
        snap = rec.build_snapshot([], [])
        self.assertTrue(snap["clean"])
        self.assertEqual(snap["open_positions"], [])
        self.assertEqual(snap["unprotected_longs"], [])
        self.assertEqual(snap["duplicate_refs"], [])
        self.assertEqual(snap["orphan_exits"], [])

    def test_protected_long_is_clean(self):
        positions = [{"symbol": "AAPL", "qty": 100}]
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "STP",
                    "qty": 100, "tif": "GTC", "order_ref": "r1"}]
        snap = rec.build_snapshot(positions, working)
        self.assertTrue(snap["clean"])
        self.assertEqual(snap["protected_longs"], ["AAPL"])
        self.assertEqual(snap["unprotected_longs"], [])


# ── 2. Pure snapshot: unprotected long detected ──────────────────────────────
class TestPureSnapshotUnprotected(unittest.TestCase):
    def test_long_without_stop_is_unprotected(self):
        snap = rec.build_snapshot([{"symbol": "AAPL", "qty": 100}], [])
        self.assertFalse(snap["clean"])
        self.assertEqual(snap["unprotected_longs"], ["AAPL"])
        self.assertEqual(snap["protected_longs"], [])

    def test_day_stop_is_not_protection(self):
        # A non-GTC (DAY) stop does not rest between the one-shot bot's runs.
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "STP",
                    "qty": 100, "tif": "DAY", "order_ref": "r1"}]
        snap = rec.build_snapshot([{"symbol": "AAPL", "qty": 100}], working)
        self.assertEqual(snap["unprotected_longs"], ["AAPL"])

    def test_take_profit_limit_is_not_protection(self):
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "LMT",
                    "qty": 100, "tif": "GTC", "order_ref": "r1"}]
        snap = rec.build_snapshot([{"symbol": "AAPL", "qty": 100}], working)
        self.assertEqual(snap["unprotected_longs"], ["AAPL"])

    def test_stop_below_filled_qty_is_unprotected(self):
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "STP",
                    "qty": 40, "tif": "GTC", "order_ref": "r1"}]
        snap = rec.build_snapshot([{"symbol": "AAPL", "qty": 100}], working)
        self.assertEqual(snap["unprotected_longs"], ["AAPL"])

    def test_flat_and_short_ignored(self):
        positions = [{"symbol": "AAPL", "qty": 0}, {"symbol": "TSLA", "qty": -5}]
        snap = rec.build_snapshot(positions, [])
        self.assertTrue(snap["clean"])
        self.assertEqual(snap["open_positions"], [])


# ── 3. Pure snapshot: duplicate orderRef detected ────────────────────────────
class TestPureSnapshotDuplicateRefs(unittest.TestCase):
    def test_duplicate_ref_detected(self):
        working = [
            {"symbol": "AAPL", "action": "BUY", "order_type": "LMT",
             "qty": 5, "order_ref": "2026-06-17:AAPL:BUY"},
            {"symbol": "AAPL", "action": "BUY", "order_type": "LMT",
             "qty": 5, "order_ref": "2026-06-17:AAPL:BUY"},
        ]
        snap = rec.build_snapshot([], working)
        self.assertEqual(snap["duplicate_refs"], ["2026-06-17:AAPL:BUY"])
        self.assertFalse(snap["clean"])

    def test_unique_refs_clean(self):
        working = [
            {"symbol": "AAPL", "action": "BUY", "order_type": "LMT", "qty": 5, "order_ref": "a"},
            {"symbol": "MSFT", "action": "BUY", "order_type": "LMT", "qty": 5, "order_ref": "b"},
        ]
        snap = rec.build_snapshot([], working)
        self.assertEqual(snap["duplicate_refs"], [])


# ── 4. Pure snapshot: orphan exit order detected ─────────────────────────────
class TestPureSnapshotOrphanExits(unittest.TestCase):
    def test_sell_without_covering_long_is_orphan(self):
        # A resting SELL for a symbol we do not hold long is an accidental-short trap.
        working = [{"symbol": "ZZZ", "action": "SELL", "order_type": "STP",
                    "qty": 100, "tif": "GTC", "order_ref": "r1"}]
        snap = rec.build_snapshot([], working)
        self.assertEqual([o["symbol"] for o in snap["orphan_exits"]], ["ZZZ"])
        self.assertFalse(snap["clean"])

    def test_sell_with_covering_long_is_not_orphan(self):
        positions = [{"symbol": "AAPL", "qty": 100}]
        working = [{"symbol": "AAPL", "action": "SELL", "order_type": "STP",
                    "qty": 100, "tif": "GTC", "order_ref": "r1"}]
        snap = rec.build_snapshot(positions, working)
        self.assertEqual(snap["orphan_exits"], [])

    def test_audit_fields_and_summary_are_json_safe(self):
        working = [{"symbol": "ZZZ", "action": "SELL", "order_type": "STP",
                    "qty": 100, "tif": "GTC", "order_ref": "r1"}]
        snap = rec.build_snapshot([{"symbol": "AAPL", "qty": 100}], working)
        fields = rec.audit_fields(snap)
        self.assertEqual(fields["n_long"], 1)
        self.assertEqual(fields["unprotected_longs"], ["AAPL"])
        self.assertEqual(fields["orphan_exit_symbols"], ["ZZZ"])
        self.assertFalse(fields["clean"])
        self.assertIsInstance(rec.summary_line(snap), str)


# ── 5. Bridge reconcile_startup_state: broker = source of truth ──────────────
class TestBridgeReconcile(_BridgeBase):
    def setUp(self):
        super().setUp()
        mock.patch.object(config, "USE_TRAILING_EXIT", False).start()  # bracket: STP + TP

    def test_no_positions_is_clean(self):
        br = make_bridge()
        report = br.reconcile_startup_state()
        self.assertTrue(report["clean"])
        self.assertEqual(report["snapshot"]["open_positions"], [])
        self.assertFalse(br._halt_new_entries)
        self.assertEqual(placed_orders(br.ib), [], "reconciliation must place nothing on a clean account")

    def test_already_protected_long_is_clean_and_untouched(self):
        br = make_bridge()
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=100.0)]
        br.ib._all.append(_gtc_stop("AAPL", 100))
        before = len(placed_orders(br.ib))  # the pre-seeded protective stop
        report = br.reconcile_startup_state()
        self.assertTrue(report["clean"])
        self.assertEqual(report["snapshot"]["protected_longs"], ["AAPL"])
        self.assertEqual(report["snapshot"]["unprotected_longs"], [])
        # Nothing NEW is placed for an already-protected long.
        self.assertEqual(len(placed_orders(br.ib)), before)
        self.assertFalse(br._halt_new_entries)

    def test_unprotected_long_is_detected_and_repaired(self):
        br = make_bridge()
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=100.0)]
        report = br.reconcile_startup_state()
        self.assertIn("AAPL", report["snapshot"]["unprotected_longs"])
        self.assertIn("AAPL", report["protect"]["repaired"])
        self.assertFalse(br._halt_new_entries)
        # A covering GTC protective stop now exists at the broker.
        self.assertTrue(ox.has_gtc_protective_stop("AAPL", 100, br._working_orders_plain()))

    def test_reconcile_never_opens_a_new_position(self):
        # The core Phase-5A safety invariant: reconciliation NEVER places a BUY
        # (entry). It may only place protective SELL repair orders.
        br = make_bridge()
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=100.0)]
        br.reconcile_startup_state()
        buys = [o for o in placed_orders(br.ib) if str(getattr(o, "action", "")).upper() == "BUY"]
        self.assertEqual(buys, [], "startup reconciliation must not open any position")

    def test_repair_failure_halts_new_entries(self):
        br = make_bridge()
        br.ib.reject_stops = True  # broker refuses the protective stop
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=100.0)]
        report = br.reconcile_startup_state()
        self.assertIn("AAPL", report["protect"]["failed"])
        self.assertTrue(report["halt_new_entries"])
        self.assertTrue(br._halt_new_entries)
        # Still no BUY/entry was placed.
        buys = [o for o in placed_orders(br.ib) if str(getattr(o, "action", "")).upper() == "BUY"]
        self.assertEqual(buys, [])

    def test_missing_avg_cost_halts_without_guessing(self):
        br = make_bridge()
        br.ib.positions_list = [FakePosition("AAPL", 100, avgCost=0.0)]  # no valid basis
        report = br.reconcile_startup_state()
        self.assertIn("AAPL", report["protect"]["failed"])
        self.assertTrue(br._halt_new_entries)
        self.assertEqual(placed_orders(br.ib), [], "must not guess a price -> nothing placed")

    def test_duplicate_orderref_surfaced(self):
        br = make_bridge()
        ref = ox.deterministic_order_ref(br._today(), "AAPL", "BUY")
        for _ in range(2):
            br.ib._all.append(FakeTrade(
                FakeContract("AAPL"), FakeOrder("BUY", "LMT", 5, ref),
                FakeStatus("Submitted", 0.0, 5.0)))
        report = br.reconcile_startup_state()
        self.assertIn(ref, report["snapshot"]["duplicate_refs"])
        self.assertFalse(report["clean"])
        # Surfacing a duplicate is read-only in Phase 5A: nothing is opened.
        buys = [o for o in placed_orders(br.ib) if str(getattr(o, "action", "")).upper() == "BUY"]
        self.assertEqual(len(buys), 2, "the two pre-seeded orders only; none added")

    def test_orphan_exit_order_detected(self):
        br = make_bridge()
        br.ib._all.append(_gtc_stop("ZZZ", 100))  # SELL stop, but no ZZZ long
        report = br.reconcile_startup_state()
        orphan_syms = [o["symbol"] for o in report["snapshot"]["orphan_exits"]]
        self.assertIn("ZZZ", orphan_syms)
        self.assertFalse(report["clean"])

    def test_startup_reconcile_alias_exists(self):
        import ibkr_bridge
        self.assertTrue(hasattr(ibkr_bridge.IBKRBridge, "startup_reconcile"))
        self.assertIs(ibkr_bridge.IBKRBridge.startup_reconcile,
                      ibkr_bridge.IBKRBridge.reconcile_startup_state)


# ── 6. Capability flag flips True only because the logic exists + is covered ──
class TestCapabilityFlag(unittest.TestCase):
    def test_flag_true(self):
        import ibkr_bridge
        self.assertTrue(ibkr_bridge.SUPPORTS_STARTUP_RECONCILIATION)
        self.assertTrue(hasattr(ibkr_bridge.IBKRBridge, "reconcile_startup_state"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
