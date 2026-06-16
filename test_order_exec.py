"""
test_order_exec.py - Phase 2 tests for the PURE order-execution logic (H4-H9).

Pure stdlib `unittest`, fully offline and deterministic (no IBKR, no network).
Exercises order_exec.py, the ib-free decision layer that ibkr_bridge.py uses to:

  * classify a trade's REAL outcome (FILLED / PARTIALLY_FILLED / WORKING /
    REJECTED / TIMEOUT) instead of trusting "accepted" (H4/H5);
  * record/size from the ACTUAL filled qty + avgFillPrice (H6);
  * verify the protective child stop covers the fill, or flag it (H7);
  * decide close retry/escalation (H8);
  * build a deterministic orderRef and detect duplicates on restart (H9).

The order-path WIRING (await-fill, flatten-on-missing-stop, close escalation,
duplicate guard) is tested with fake ib_insync Trade objects in
test_ibkr_fill_flow.py.

Run with:
    python -m unittest test_order_exec -v
"""
import unittest

import order_exec as ox


# -- tiny duck-typed fakes (no ib_insync needed for the pure layer) -----------
class _Status:
    def __init__(self, status="PreSubmitted", filled=0.0, remaining=0.0,
                 avgFillPrice=0.0, parentId=0):
        self.status = status
        self.filled = filled
        self.remaining = remaining
        self.avgFillPrice = avgFillPrice
        self.parentId = parentId


class _Trade:
    def __init__(self, status):
        self.orderStatus = status


def trade(status="PreSubmitted", filled=0.0, remaining=0.0, avg=0.0):
    return _Trade(_Status(status=status, filled=filled, remaining=remaining, avgFillPrice=avg))


def wo(symbol, action="SELL", order_type="STP", qty=0.0, parent_id=0, order_ref=""):
    return {"symbol": symbol, "action": action, "order_type": order_type,
            "qty": qty, "parent_id": parent_id, "order_ref": order_ref}


# ── 1. classify_trade: accepted is NOT filled (H4/H5) ────────────────────────
class TestClassify(unittest.TestCase):
    def test_filled_full(self):
        r = ox.classify_trade(trade("Filled", filled=100, remaining=0, avg=10.0))
        self.assertEqual(r.outcome, ox.FILLED)
        self.assertTrue(r.is_filled)
        self.assertEqual(r.filled, 100)
        self.assertEqual(r.avg_fill_price, 10.0)

    def test_filled_when_remaining_zero_even_if_status_lags(self):
        r = ox.classify_trade(trade("Submitted", filled=100, remaining=0))
        self.assertEqual(r.outcome, ox.FILLED)

    def test_presubmitted_is_working_not_filled(self):
        # The core H4 bug: PreSubmitted/Submitted must NOT count as a fill.
        for s in ("PreSubmitted", "Submitted", "PendingSubmit", "ApiPending"):
            r = ox.classify_trade(trade(s, filled=0, remaining=100))
            self.assertEqual(r.outcome, ox.WORKING, s)
            self.assertFalse(r.is_filled, s)

    def test_partial_active(self):
        r = ox.classify_trade(trade("Submitted", filled=30, remaining=70, avg=10.0))
        self.assertEqual(r.outcome, ox.PARTIALLY_FILLED)
        self.assertTrue(r.is_partial)
        self.assertTrue(r.has_fill)

    def test_cancelled_with_no_fill_is_rejected(self):
        for s in ("Cancelled", "ApiCancelled", "Inactive", "Rejected"):
            r = ox.classify_trade(trade(s, filled=0, remaining=100))
            self.assertEqual(r.outcome, ox.REJECTED, s)

    def test_cancelled_after_partial_is_partial(self):
        r = ox.classify_trade(trade("Cancelled", filled=40, remaining=60))
        self.assertEqual(r.outcome, ox.PARTIALLY_FILLED)

    def test_nan_and_junk_fields_are_safe(self):
        r = ox.classify_trade(trade("Submitted", filled=float("nan"), remaining=float("inf")))
        self.assertEqual(r.filled, 0.0)
        self.assertEqual(r.remaining, 0.0)
        self.assertEqual(r.outcome, ox.WORKING)


# ── 2. outcome_on_timeout ────────────────────────────────────────────────────
class TestTimeout(unittest.TestCase):
    def test_no_fill_times_out(self):
        r = ox.outcome_on_timeout(trade("Submitted", filled=0, remaining=100))
        self.assertEqual(r.outcome, ox.TIMEOUT)
        self.assertEqual(r.filled, 0.0)

    def test_partial_fill_on_timeout_is_partial(self):
        r = ox.outcome_on_timeout(trade("Submitted", filled=25, remaining=75, avg=9.0))
        self.assertEqual(r.outcome, ox.PARTIALLY_FILLED)
        self.assertEqual(r.filled, 25)

    def test_rejected_is_terminal_not_timeout(self):
        # ib_insync DoneStates omits "Rejected"/"Inactive"; they must still be
        # terminal so a rejected order resolves as REJECTED, never TIMEOUT.
        for s in ("Rejected", "Inactive", "Cancelled", "ApiCancelled", "Filled"):
            self.assertTrue(ox.is_terminal(trade(s)), s)
        self.assertFalse(ox.is_terminal(trade("Submitted")))
        self.assertEqual(ox.outcome_on_timeout(trade("Rejected", filled=0)).outcome, ox.REJECTED)


# ── 3. should_record_trade: only a real held fill counts (H4/H6) ─────────────
class TestShouldRecord(unittest.TestCase):
    def test_filled_records(self):
        self.assertTrue(ox.should_record_trade(ox.OrderResult(ox.FILLED, filled=100)))

    def test_partial_records(self):
        self.assertTrue(ox.should_record_trade(ox.OrderResult(ox.PARTIALLY_FILLED, filled=30)))

    def test_working_timeout_rejected_do_not_record(self):
        for oc in (ox.WORKING, ox.TIMEOUT, ox.REJECTED):
            self.assertFalse(ox.should_record_trade(ox.OrderResult(oc, filled=0)))

    def test_aborted_fill_does_not_record(self):
        # Entered but emergency-flattened (no live stop) -> must not count.
        self.assertFalse(ox.should_record_trade(ox.OrderResult(ox.FILLED, filled=100, aborted=True)))

    def test_filled_outcome_but_zero_qty_does_not_record(self):
        self.assertFalse(ox.should_record_trade(ox.OrderResult(ox.FILLED, filled=0)))


# ── 4. occupies_slot semantics ───────────────────────────────────────────────
class TestOccupiesSlot(unittest.TestCase):
    def test_resting_or_held_occupies(self):
        for oc in (ox.FILLED, ox.PARTIALLY_FILLED, ox.WORKING, ox.TIMEOUT):
            self.assertTrue(ox.OrderResult(oc, filled=1).occupies_slot, oc)

    def test_rejected_frees_slot(self):
        self.assertFalse(ox.OrderResult(ox.REJECTED).occupies_slot)

    def test_aborted_holds_nothing(self):
        self.assertFalse(ox.OrderResult(ox.FILLED, filled=100, aborted=True).occupies_slot)


# ── 5. protective_exit_qty: exact filled qty, floored (H6) ───────────────────
class TestExitQty(unittest.TestCase):
    def test_exact_floor(self):
        self.assertEqual(ox.protective_exit_qty(100.0), 100)
        self.assertEqual(ox.protective_exit_qty(30.9), 30)

    def test_non_positive_is_zero(self):
        self.assertEqual(ox.protective_exit_qty(0), 0)
        self.assertEqual(ox.protective_exit_qty(-5), 0)
        self.assertEqual(ox.protective_exit_qty(float("nan")), 0)


# ── 6. deterministic orderRef + duplicate detection (H9) ─────────────────────
class TestOrderRef(unittest.TestCase):
    def test_ref_is_stable_and_normalised(self):
        self.assertEqual(ox.deterministic_order_ref("2026-06-16", "aapl", "buy"),
                         "2026-06-16:AAPL:BUY")
        # Same intent -> same ref (so a restart duplicate is detectable).
        self.assertEqual(ox.deterministic_order_ref("2026-06-16", "AAPL", "BUY"),
                         ox.deterministic_order_ref("2026-06-16", " aapl ", " buy "))

    def test_has_order_ref_detects_live_duplicate(self):
        ref = ox.deterministic_order_ref("2026-06-16", "AAPL", "BUY")
        working = [wo("AAPL", action="BUY", order_type="LMT", order_ref=ref)]
        self.assertTrue(ox.has_order_ref(ref, working))

    def test_has_order_ref_false_when_absent_or_blank(self):
        self.assertFalse(ox.has_order_ref("2026-06-16:AAPL:BUY", []))
        self.assertFalse(ox.has_order_ref("", [wo("AAPL", order_ref="x")]))


# ── 7. protective-child verification (H7) ────────────────────────────────────
class TestVerifyChild(unittest.TestCase):
    def test_missing_stop_is_unprotected(self):
        v = ox.verify_protective_child("AAPL", parent_id=5, filled_qty_value=100, working_orders=[])
        self.assertFalse(v["ok"])
        self.assertEqual(v["reason"], "no_protective_child")

    def test_take_profit_limit_is_not_protection(self):
        working = [wo("AAPL", action="SELL", order_type="LMT", qty=100)]  # TP, not a stop
        v = ox.verify_protective_child("AAPL", 5, 100, working)
        self.assertFalse(v["ok"])

    def test_covering_stop_protects(self):
        working = [wo("AAPL", action="SELL", order_type="STP", qty=100, parent_id=5)]
        v = ox.verify_protective_child("AAPL", 5, 100, working)
        self.assertTrue(v["ok"])

    def test_trailing_stop_protects(self):
        working = [wo("AAPL", action="SELL", order_type="TRAIL", qty=100)]
        v = ox.verify_protective_child("AAPL", 0, 100, working)
        self.assertTrue(v["ok"])

    def test_stop_too_small_to_cover_fill_is_flagged(self):
        working = [wo("AAPL", action="SELL", order_type="STP", qty=30)]
        v = ox.verify_protective_child("AAPL", 5, 100, working)
        self.assertFalse(v["ok"])
        self.assertEqual(v["reason"], "child_qty_below_filled")

    def test_other_symbol_stop_does_not_protect(self):
        working = [wo("MSFT", action="SELL", order_type="STP", qty=100)]
        v = ox.verify_protective_child("AAPL", 5, 100, working)
        self.assertFalse(v["ok"])


class TestChildResize(unittest.TestCase):
    def test_oversized_child_needs_resize_down(self):
        # Native bracket child = intended 100, but only 30 filled -> must shrink
        # so a trigger cannot sell more than we hold (accidental short).
        self.assertTrue(ox.child_needs_resize(100, 30))

    def test_exact_or_undersized_does_not_resize(self):
        self.assertFalse(ox.child_needs_resize(30, 30))
        self.assertFalse(ox.child_needs_resize(30, 100))


class TestExitChildren(unittest.TestCase):
    """Every EXIT-side (SELL) child counts, not just stops: a take-profit LMT
    can also over-sell and flip a long short (H6)."""

    def _bracket_children(self, qty):
        # As a native bracket leaves them after a parent fill.
        return [
            wo("AAPL", action="SELL", order_type="STP", qty=qty),   # protective stop
            wo("AAPL", action="SELL", order_type="LMT", qty=qty),   # take-profit
        ]

    def test_exit_children_include_take_profit_limit(self):
        kids = ox.exit_children("AAPL", "SELL", self._bracket_children(100))
        types = sorted(k["order_type"] for k in kids)
        self.assertEqual(types, ["LMT", "STP"])  # BOTH, not just the stop

    def test_buy_parent_is_not_an_exit_child(self):
        working = [wo("AAPL", action="BUY", order_type="LMT", qty=100)]
        self.assertEqual(ox.exit_children("AAPL", "SELL", working), [])

    def test_oversized_includes_both_stop_and_take_profit(self):
        over = ox.oversized_exit_children("AAPL", "SELL", 30, self._bracket_children(100))
        self.assertEqual(len(over), 2)  # the 100-share STP *and* the 100-share LMT

    def test_exact_filled_qty_is_not_oversized(self):
        self.assertEqual(ox.oversized_exit_children("AAPL", "SELL", 100, self._bracket_children(100)), [])

    def test_undersized_exit_is_not_oversized(self):
        working = [wo("AAPL", action="SELL", order_type="STP", qty=30)]
        self.assertEqual(ox.oversized_exit_children("AAPL", "SELL", 100, working), [])


# ── 8. close retry / escalation (H8) ─────────────────────────────────────────
class TestCloseFollowup(unittest.TestCase):
    def test_full_fill_is_done(self):
        r = ox.OrderResult(ox.FILLED, filled=100, remaining=0)
        self.assertEqual(ox.close_followup(r, attempt=1, max_attempts=3), "done")

    def test_unfilled_escalates_while_attempts_remain(self):
        r = ox.OrderResult(ox.WORKING, filled=0, remaining=100)
        self.assertEqual(ox.close_followup(r, attempt=1, max_attempts=3), "escalate")

    def test_partial_escalates(self):
        r = ox.OrderResult(ox.PARTIALLY_FILLED, filled=40, remaining=60)
        self.assertEqual(ox.close_followup(r, attempt=2, max_attempts=3), "escalate")

    def test_exhausted_attempts_give_up(self):
        r = ox.OrderResult(ox.WORKING, filled=0, remaining=100)
        self.assertEqual(ox.close_followup(r, attempt=3, max_attempts=3), "giveup")


if __name__ == "__main__":
    unittest.main(verbosity=2)
