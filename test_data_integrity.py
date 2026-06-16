"""
test_data_integrity.py - Phase 4 tests for local data_integrity.py API.

Run:
    python -X utf8 -m unittest test_data_integrity -v
"""
import datetime as dt
import unittest
from zoneinfo import ZoneInfo

import data_integrity as dg


class TestEvaluateOrderQuote(unittest.TestCase):
    def test_realtime_last_allowed(self):
        q = dg.evaluate_order_quote(last=100.25, close=99.0, require_realtime=True)
        self.assertTrue(q.ok)
        self.assertEqual(q.price, 100.25)

    def test_realtime_rejects_close_only(self):
        q = dg.evaluate_order_quote(close=100.0, require_realtime=True)
        self.assertFalse(q.ok)
        self.assertIn(q.reason, (dg.REASON_STALE_CLOSE, dg.REASON_DELAYED_BLOCKED))

    def test_realtime_rejects_delayed_only(self):
        q = dg.evaluate_order_quote(
            delayed_last=100.0,
            delayed_bid=99.9,
            delayed_ask=100.1,
            require_realtime=True,
        )
        self.assertFalse(q.ok)
        self.assertEqual(q.reason, dg.REASON_DELAYED_BLOCKED)

    def test_paper_can_use_delayed_last_when_realtime_not_required(self):
        q = dg.evaluate_order_quote(delayed_last=100.0, require_realtime=False)
        self.assertTrue(q.ok)
        self.assertEqual(q.price, 100.0)

    def test_bid_ask_mid_allowed_when_spread_sane(self):
        q = dg.evaluate_order_quote(bid=99.95, ask=100.05, require_realtime=True)
        self.assertTrue(q.ok)
        self.assertAlmostEqual(q.price, 100.0)

    def test_crossed_or_locked_market_rejected(self):
        for bid, ask in ((100, 100), (101, 100)):
            q = dg.evaluate_order_quote(bid=bid, ask=ask, require_realtime=True)
            self.assertFalse(q.ok)
            self.assertEqual(q.reason, dg.REASON_CROSSED)

    def test_wide_spread_rejected(self):
        q = dg.evaluate_order_quote(
            bid=98,
            ask=102,
            require_realtime=True,
            max_spread_pct=0.01,
        )
        self.assertFalse(q.ok)
        self.assertEqual(q.reason, dg.REASON_WIDE_SPREAD)

    def test_no_quote_rejected(self):
        q = dg.evaluate_order_quote(require_realtime=True)
        self.assertFalse(q.ok)
        self.assertEqual(q.reason, dg.REASON_NO_DATA)


class TestDecisionPriceOk(unittest.TestCase):
    def test_within_deviation_passes(self):
        self.assertTrue(dg.decision_price_ok(100, 100.49, 0.005))

    def test_above_deviation_fails(self):
        self.assertFalse(dg.decision_price_ok(100, 100.60, 0.005))

    def test_missing_decision_price_does_not_spuriously_block(self):
        # Current bridge comments say missing decision price should not block.
        self.assertTrue(dg.decision_price_ok(None, 100, 0.005))
        self.assertTrue(dg.decision_price_ok(0, 100, 0.005))

    def test_missing_order_price_fails(self):
        self.assertFalse(dg.decision_price_ok(100, 0, 0.005))


class TestRegularHours(unittest.TestCase):
    def ny(self, y, m, d, hh, mm=0):
        return dt.datetime(y, m, d, hh, mm, tzinfo=ZoneInfo("America/New_York"))

    def test_rth_weekday_passes(self):
        self.assertTrue(dg.is_regular_hours(self.ny(2026, 6, 16, 10, 0)))

    def test_before_open_blocks(self):
        self.assertFalse(dg.is_regular_hours(self.ny(2026, 6, 16, 9, 29)))

    def test_after_close_blocks(self):
        self.assertFalse(dg.is_regular_hours(self.ny(2026, 6, 16, 16, 0)))

    def test_weekend_blocks(self):
        self.assertFalse(dg.is_regular_hours(self.ny(2026, 6, 20, 10, 0)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
