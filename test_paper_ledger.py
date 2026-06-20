"""test_paper_ledger.py - offline deterministic tests for the forward paper-test
journal writer + round-trip reconstruction (paper_ledger.py).

Pure unittest, no network, no IBKR, no order path. All file IO uses temp dirs, so
nothing under the real reports/ is touched. Run with:

    python -m unittest test_paper_ledger
"""
from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import config
import order_exec
import paper_ledger as pl


def _signal(**kw):
    base = dict(symbol="AAPL", action="BUY", confidence=0.72, rf_score=0.6,
                lstm_score=0.65, tech_score=0.55, price=100.0,
                reason="RF=0.60 LSTM=0.65 Tech=0.55 -> ensemble=0.62")
    base.update(kw)
    return types.SimpleNamespace(**base)


def _filled(filled=10.0, avg=100.0):
    return order_exec.OrderResult(outcome=order_exec.FILLED, filled=filled,
                                  remaining=0.0, avg_fill_price=avg)


# ── Pure classification ──────────────────────────────────────────────────────
class TestClassifyExitKind(unittest.TestCase):
    def test_protective_and_close_kinds(self):
        self.assertEqual(pl.classify_exit_kind("2026-06-05:AAPL:SELL_HARD_STOP"),
                         pl.EXIT_PROTECTIVE_STOP)
        self.assertEqual(pl.classify_exit_kind("2026-06-05:AAPL:SELL_TRAILING_STOP"),
                         pl.EXIT_PROTECTIVE_STOP)
        self.assertEqual(pl.classify_exit_kind("2026-06-05:AAPL:SELL_STOP"),
                         pl.EXIT_PROTECTIVE_STOP)
        self.assertEqual(pl.classify_exit_kind("2026-06-05:AAPL:SELL_TAKE_PROFIT"),
                         pl.EXIT_TAKE_PROFIT)
        self.assertEqual(pl.classify_exit_kind("2026-06-05:AAPL:SELL"),
                         pl.EXIT_SIGNAL_CLOSE)
        self.assertEqual(pl.classify_exit_kind(None), pl.EXIT_UNKNOWN)

    def test_dedupe_key_is_deterministic(self):
        a = pl.exit_dedupe_key(symbol="AAPL", side="SELL", qty=10, price=110.0,
                               ts="2026-06-05T15:00:00", order_ref="r")
        b = pl.exit_dedupe_key(symbol="aapl", side="sell", qty=10.0, price=110.0,
                               ts="2026-06-05T15:00:00", order_ref="r")
        c = pl.exit_dedupe_key(symbol="AAPL", side="SELL", qty=11, price=110.0,
                               ts="2026-06-05T15:00:00", order_ref="r")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


# ── Writer (append-only, never-raise) ────────────────────────────────────────
class TestWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal = Path(self.tmp.name) / "paper_trades.jsonl"
        mock.patch.object(config, "FORWARD_TEST_JOURNAL", self.journal).start()
        mock.patch.object(config, "FORWARD_TEST_SESSION", None).start()
        self.addCleanup(mock.patch.stopall)

    def _events(self):
        return pl.load_journal(self.journal)

    def test_record_entry_writes_event_with_id_and_reason(self):
        pl.record_entry(_signal(), _filled(10, 100.0), order_ref="2026-06-01:AAPL:BUY")
        evs = self._events()
        self.assertEqual(len(evs), 1)
        e = evs[0]
        self.assertEqual(e["event"], pl.EVENT_ENTRY)
        self.assertEqual(e["symbol"], "AAPL")
        self.assertEqual(e["qty"], 10.0)
        self.assertEqual(e["entry_avg_price"], 100.0)
        self.assertIn("RF=0.60", e["reason"])
        self.assertTrue(e.get("event_id"))
        self.assertTrue(e.get("ts"))

    def test_record_entry_skips_non_fill_and_aborted(self):
        # WORKING / REJECTED / TIMEOUT -> not recorded.
        pl.record_entry(_signal(), order_exec.OrderResult(outcome=order_exec.WORKING))
        pl.record_entry(_signal(), order_exec.OrderResult(outcome=order_exec.REJECTED))
        pl.record_entry(_signal(), order_exec.OrderResult(
            outcome=order_exec.TIMEOUT, filled=0.0))
        # Filled but emergency-aborted -> not held -> not recorded.
        pl.record_entry(_signal(), order_exec.OrderResult(
            outcome=order_exec.FILLED, filled=10.0, aborted=True))
        self.assertEqual(self._events(), [])

    def test_record_entry_records_partial_fill(self):
        pl.record_entry(_signal(), order_exec.OrderResult(
            outcome=order_exec.PARTIALLY_FILLED, filled=4.0, remaining=6.0,
            avg_fill_price=100.0))
        evs = self._events()
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["qty"], 4.0)

    def test_record_exit_exec_id_and_fallback(self):
        pl.record_exit("AAPL", exit_kind=pl.EXIT_SIGNAL_CLOSE, qty=10,
                       exit_avg_price=110.0, exec_id="EX1", ts="2026-06-05T15:00:00")
        pl.record_exit("MSFT", exit_kind=pl.EXIT_PROTECTIVE_STOP, qty=5,
                       exit_avg_price=200.0, order_ref="2026-06-05:MSFT:SELL_STOP",
                       ts="2026-06-05T15:01:00")  # no exec_id -> fallback hash
        evs = self._events()
        self.assertEqual(len(evs), 2)
        self.assertEqual(evs[0]["exec_id"], "EX1")
        self.assertEqual(evs[0]["dedupe_id"], "EX1")
        self.assertIsNone(evs[1]["exec_id"])
        self.assertEqual(
            evs[1]["dedupe_id"],
            pl.exit_dedupe_key(symbol="MSFT", side="SELL", qty=5, price=200.0,
                               ts="2026-06-05T15:01:00",
                               order_ref="2026-06-05:MSFT:SELL_STOP"))

    def test_record_exit_blank_exec_id_uses_stable_fallback(self):
        pl.record_exit("AAPL", exit_kind=pl.EXIT_PROTECTIVE_STOP, qty=2,
                       exit_avg_price=94.0, order_ref="2026-06-05:AAPL:SELL_STOP",
                       exec_id="", ts=None)
        evs = self._events()
        self.assertEqual(len(evs), 1)
        self.assertIsNone(evs[0]["exec_id"])
        self.assertEqual(
            evs[0]["dedupe_id"],
            pl.exit_dedupe_key(symbol="AAPL", side="SELL", qty=2, price=94.0,
                               ts="", order_ref="2026-06-05:AAPL:SELL_STOP"))
        self.assertIn(evs[0]["dedupe_id"], pl.recorded_exit_keys(self.journal))

    def test_writer_never_raises_on_junk(self):
        # Bad objects must never propagate an exception into the order path.
        pl.record_entry(None, None)
        pl.record_stop(None, None, None)
        pl.record_exit(None, exit_kind=None, qty="x", exit_avg_price="y")
        # And a broken journal path must be swallowed too.
        with mock.patch.object(pl, "_journal_path", side_effect=OSError("boom")):
            pl.record_entry(_signal(), _filled())
            pl.record_stop("AAPL", 10, 95.0)
            pl.record_exit("AAPL", exit_kind=pl.EXIT_SIGNAL_CLOSE, qty=10,
                           exit_avg_price=110.0, exec_id="Z")

    def test_recorded_exit_keys_and_open_long_qty(self):
        pl.record_entry(_signal(), _filled(10, 100.0))
        pl.record_stop("AAPL", 10, 95.0)
        pl.record_exit("AAPL", exit_kind=pl.EXIT_SIGNAL_CLOSE, qty=4,
                       exit_avg_price=110.0, exec_id="E1")
        self.assertIn("E1", pl.recorded_exit_keys(self.journal))
        # 10 opened - 4 exited = 6 still open.
        self.assertEqual(pl.open_long_qty(self.journal), {"AAPL": 6.0})


# ── Reconstruction (FIFO, dollar PnL, broker protective-stop R) ──────────────
def _e(event, ts, **kw):
    d = {"event": event, "ts": ts}
    d.update(kw)
    return d


class TestReconstruct(unittest.TestCase):
    def _one(self, trades, **expect):
        self.assertEqual(len(trades), 1)
        t = trades[0]
        for k, v in expect.items():
            got = getattr(t, k)
            if isinstance(v, float):
                self.assertAlmostEqual(got, v, places=6, msg=k)
            else:
                self.assertEqual(got, v, msg=k)
        return trades[0]

    def test_single_round_trip_win_with_true_stop_R(self):
        events = [
            _e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=10,
               entry_avg_price=100.0, confidence=0.72, reason="RF..", session="S1"),
            _e("stop", "2026-06-01T10:00:01", symbol="AAPL", qty=10,
               stop_price=95.0, hard_stop=90.0),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exit_kind="signal_close", exec_id="E1"),
        ]
        t = self._one(pl.reconstruct_paper_trades(events),
                      realized_pnl=100.0, initial_risk=50.0, r_multiple=2.0,
                      pnl_included=True, r_included=True,
                      risk_source=pl.RISK_SOURCE_PROTECTIVE_STOP)
        self.assertAlmostEqual(t.realized_pnl_pct, 0.10, places=6)

    def test_real_hook_order_stop_before_entry_still_gets_R(self):
        events = [
            _e("stop", "2026-06-01T10:00:01", symbol="AAPL", qty=10,
               stop_price=95.0, order_ref="2026-06-01:AAPL:BUY"),
            _e("entry", "2026-06-01T10:00:02", symbol="AAPL", qty=10,
               entry_avg_price=100.0, order_ref="2026-06-01:AAPL:BUY"),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exec_id="E1"),
        ]
        self._one(pl.reconstruct_paper_trades(events),
                  realized_pnl=100.0, initial_risk=50.0, r_multiple=2.0,
                  pnl_included=True, r_included=True,
                  risk_source=pl.RISK_SOURCE_PROTECTIVE_STOP)

    def test_pre_entry_stop_does_not_attach_to_wrong_order_ref(self):
        events = [
            _e("stop", "2026-06-01T10:00:01", symbol="AAPL", qty=10,
               stop_price=95.0, order_ref="2026-06-01:AAPL:BUY"),
            _e("entry", "2026-06-01T10:00:02", symbol="AAPL", qty=10,
               entry_avg_price=100.0, order_ref="2026-06-01:AAPL:BUY-OTHER"),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exec_id="E1"),
        ]
        t = self._one(pl.reconstruct_paper_trades(events),
                      realized_pnl=100.0, pnl_included=True, r_included=False,
                      risk_source=pl.RISK_SOURCE_UNAVAILABLE)
        import expectancy
        self.assertEqual(t.r_exclusion_reason, expectancy.R_REASON_UNAVAILABLE_RISK)

    def test_single_round_trip_loss(self):
        events = [
            _e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=10, entry_avg_price=100.0),
            _e("stop", "2026-06-01T10:00:01", symbol="AAPL", qty=10, stop_price=95.0),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=92.0, exec_id="E1"),
        ]
        self._one(pl.reconstruct_paper_trades(events),
                  realized_pnl=-80.0, r_multiple=-1.6, pnl_included=True)

    def test_stop_falls_back_to_hard_stop(self):
        events = [
            _e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=10, entry_avg_price=100.0),
            _e("stop", "2026-06-01T10:00:01", symbol="AAPL", qty=10,
               stop_price=None, hard_stop=90.0),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exec_id="E1"),
        ]
        self._one(pl.reconstruct_paper_trades(events),
                  initial_risk=100.0, r_multiple=1.0,
                  risk_source=pl.RISK_SOURCE_PROTECTIVE_STOP)

    def test_no_valid_stop_excludes_from_R_only(self):
        events = [
            _e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=10, entry_avg_price=100.0),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exec_id="E1"),
        ]
        t = self._one(pl.reconstruct_paper_trades(events),
                      realized_pnl=100.0, pnl_included=True, r_included=False,
                      risk_source=pl.RISK_SOURCE_UNAVAILABLE)
        import expectancy
        self.assertEqual(t.r_exclusion_reason, expectancy.R_REASON_UNAVAILABLE_RISK)

    def test_multiple_lots_fifo(self):
        events = [
            _e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=5, entry_avg_price=100.0),
            _e("entry", "2026-06-01T10:05:00", symbol="AAPL", qty=5, entry_avg_price=102.0),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exec_id="E1"),
        ]
        trades = pl.reconstruct_paper_trades(events)
        self.assertEqual(len(trades), 2)
        self.assertAlmostEqual(trades[0].realized_pnl, 50.0)   # 5*(110-100)
        self.assertAlmostEqual(trades[1].realized_pnl, 40.0)   # 5*(110-102)

    def test_orphan_exit_excluded(self):
        events = [_e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
                     exit_avg_price=110.0, exec_id="E1")]
        import expectancy
        t = self._one(pl.reconstruct_paper_trades(events), pnl_included=False)
        self.assertEqual(t.exclusion_reason, expectancy.REASON_ORPHAN_EXIT)

    def test_surplus_sell_makes_orphan_for_remainder(self):
        events = [
            _e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=5, entry_avg_price=100.0),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=8,
               exit_avg_price=110.0, exec_id="E1"),
        ]
        import expectancy
        trades = pl.reconstruct_paper_trades(events)
        self.assertEqual(len(trades), 2)
        closed = [t for t in trades if t.pnl_included]
        orphan = [t for t in trades if not t.pnl_included]
        self.assertEqual(len(closed), 1)
        self.assertAlmostEqual(closed[0].qty, 5.0)
        self.assertEqual(orphan[0].exclusion_reason, expectancy.REASON_ORPHAN_EXIT)
        self.assertAlmostEqual(orphan[0].qty, 3.0)

    def test_open_position_excluded(self):
        events = [_e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=10,
                     entry_avg_price=100.0)]
        import expectancy
        t = self._one(pl.reconstruct_paper_trades(events), pnl_included=False)
        self.assertEqual(t.exclusion_reason, expectancy.REASON_OPEN_AT_PERIOD_END)

    def test_duplicate_exit_execid_deduped(self):
        events = [
            _e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=10, entry_avg_price=100.0),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exec_id="DUP"),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exec_id="DUP"),  # same execId -> ignored
        ]
        trades = pl.reconstruct_paper_trades(events)
        self.assertEqual(len(trades), 1)
        self.assertTrue(trades[0].pnl_included)

    def test_missing_entry_price_excluded(self):
        events = [
            _e("entry", "2026-06-01T10:00:00", symbol="AAPL", qty=10, entry_avg_price=0.0),
            _e("exit", "2026-06-05T15:00:00", symbol="AAPL", qty=10,
               exit_avg_price=110.0, exec_id="E1"),
        ]
        import expectancy
        t = self._one(pl.reconstruct_paper_trades(events), pnl_included=False)
        self.assertEqual(t.exclusion_reason, expectancy.REASON_MISSING_ENTRY_PRICE)


if __name__ == "__main__":
    unittest.main(verbosity=2)
