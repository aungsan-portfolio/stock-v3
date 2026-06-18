"""
test_minervini.py — offline unit tests for the Minervini/SEPA overlay detectors.

Pure stdlib `unittest`, fully offline (synthetic OHLCV only — no network, no
trained models, no IBKR). Run with either:

    python -m unittest test_minervini -v
    python test_minervini.py

These exercise minervini.py in isolation. M0 does NOT wire the overlay into the
order path, so nothing here touches trading behavior.
"""
import unittest

import numpy as np
import pandas as pd

import config
import minervini


# ── Synthetic OHLCV builders ─────────────────────────────────────────────────
def frame(closes, highs=None, lows=None, vols=None) -> pd.DataFrame:
    """Build an OHLCV frame with a business-day index from a close series."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    highs = closes * 1.01 if highs is None else np.asarray(highs, dtype=float)
    lows = closes * 0.99 if lows is None else np.asarray(lows, dtype=float)
    vols = np.full(n, 1_000_000.0) if vols is None else np.asarray(vols, dtype=float)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


def uptrend(n: int = 300, g: float = 0.004) -> pd.DataFrame:
    """Smooth exponential uptrend: price > 50 > 150 > 200 SMA, 200 rising,
    current bar near the 52-week high and well above the 52-week low."""
    closes = 100.0 * np.power(1.0 + g, np.arange(n))
    return frame(closes)


def downtrend(n: int = 300, g: float = 0.003) -> pd.DataFrame:
    closes = 150.0 * np.power(1.0 - g, np.arange(n))
    return frame(closes)


def spike_high_frame(n: int = 300, g: float = 0.00135,
                     spike_idx: int = 150, spike_mult: float = 3.0) -> pd.DataFrame:
    """Smooth uptrend (so SMAs stay ordered/rising and price > all SMAs) but with
    a single old High spike inside the 52-week window, putting the current price
    more than NEAR_52W_HIGH_PCT below the 52-week high. Isolates 'below_52w_high'."""
    closes = 100.0 * np.power(1.0 + g, np.arange(n))
    highs = closes * 1.01
    lows = closes * 0.99
    highs[spike_idx] = closes[spike_idx] * spike_mult  # spike in High only; Close untouched
    return frame(closes, highs=highs, lows=lows)


def vcp_frame(seg_ranges=(0.113, 0.058, 0.0296), n_pre: int = 20, seg_len: int = 15) -> pd.DataFrame:
    """Flat base whose last 3 segments (the closed window) have the given relative
    high-low ranges. The final (forming) row is appended and will be dropped."""
    base_len = len(seg_ranges) * seg_len
    closes_base = np.full(base_len, 100.0)
    highs_base = np.empty(base_len)
    lows_base = np.empty(base_len)
    for s, r in enumerate(seg_ranges):
        sl = slice(s * seg_len, (s + 1) * seg_len)
        highs_base[sl] = 100.0 * (1.0 + r / 2.0)
        lows_base[sl] = 100.0 * (1.0 - r / 2.0)
    closes = np.concatenate([np.full(n_pre, 100.0), closes_base, [100.0]])
    highs = np.concatenate([np.full(n_pre, 100.5), highs_base, [100.5]])
    lows = np.concatenate([np.full(n_pre, 99.5), lows_base, [99.5]])
    return frame(closes, highs=highs, lows=lows)


def pocket_frame(last_close: float, last_vol: float, n: int = 20) -> pd.DataFrame:
    """Alternating base (creates down days at 1e6 volume) with the last CLOSED bar
    set to (last_close, last_vol). A trailing forming row is appended/dropped."""
    closes = np.tile([100.0, 99.0], (n + 1) // 2)[:n].astype(float)
    vols = np.full(n, 1_000_000.0)
    closes[-3] = 100.0          # bar before the last closed bar
    closes[-2] = last_close     # the last CLOSED bar (index -1 is forming -> dropped)
    closes[-1] = last_close
    vols[-2] = last_vol
    return frame(closes, vols=vols)


# ── Trend template ───────────────────────────────────────────────────────────
class TestTrendTemplate(unittest.TestCase):
    def test_clean_stage2_passes(self):
        ok, reasons, detail = minervini.trend_template(uptrend())
        self.assertTrue(ok, f"clean uptrend should pass; reasons={reasons}")
        self.assertEqual(reasons, [])
        self.assertTrue(detail["ma200_rising"])

    def test_downtrend_fails_ma_order(self):
        ok, reasons, _ = minervini.trend_template(downtrend())
        self.assertFalse(ok)
        self.assertIn("ma_order_fail", reasons)

    def test_below_52w_high_isolated(self):
        ok, reasons, _ = minervini.trend_template(spike_high_frame())
        self.assertFalse(ok)
        self.assertIn("below_52w_high", reasons)
        self.assertNotIn("ma_order_fail", reasons)   # MA structure intact -> token isolated

    def test_insufficient_history(self):
        ok, reasons, _ = minervini.trend_template(uptrend(n=50))
        self.assertFalse(ok)
        self.assertIn("insufficient_history", reasons)


# ── VCP-like ─────────────────────────────────────────────────────────────────
class TestVcpLike(unittest.TestCase):
    def test_two_contractions_qualify(self):
        is_vcp, pivot_low, detail = minervini.detect_vcp_like(vcp_frame())
        self.assertTrue(is_vcp)
        self.assertEqual(detail["contractions"], 2)
        self.assertAlmostEqual(pivot_low, 100.0 * (1.0 - 0.0296 / 2.0), delta=0.1)

    def test_deep_base_rejected(self):
        is_vcp, _, detail = minervini.detect_vcp_like(vcp_frame(seg_ranges=(0.45, 0.058, 0.0296)))
        self.assertFalse(is_vcp)
        self.assertGreater(detail["base_depth"], 0.35)

    def test_non_contracting_rejected(self):
        is_vcp, _, detail = minervini.detect_vcp_like(vcp_frame(seg_ranges=(0.03, 0.06, 0.11)))
        self.assertFalse(is_vcp)
        self.assertLess(detail["contractions"], 2)

    def test_insufficient_history(self):
        is_vcp, pivot_low, _ = minervini.detect_vcp_like(frame(np.full(10, 100.0)))
        self.assertFalse(is_vcp)
        self.assertIsNone(pivot_low)


# ── Pocket pivot ─────────────────────────────────────────────────────────────
class TestPocketPivot(unittest.TestCase):
    def test_up_day_high_volume_is_pocket_pivot(self):
        is_pp, detail = minervini.detect_pocket_pivot(pocket_frame(last_close=101.0, last_vol=3_000_000.0))
        self.assertTrue(is_pp)
        self.assertGreater(detail["last_vol"], detail["max_down_vol"])

    def test_down_day_is_not_pocket_pivot(self):
        is_pp, _ = minervini.detect_pocket_pivot(pocket_frame(last_close=99.0, last_vol=3_000_000.0))
        self.assertFalse(is_pp)

    def test_up_day_low_volume_is_not_pocket_pivot(self):
        is_pp, _ = minervini.detect_pocket_pivot(pocket_frame(last_close=101.0, last_vol=500_000.0))
        self.assertFalse(is_pp)


# ── Dedicated 1R stop ────────────────────────────────────────────────────────
class TestMinerviniStop(unittest.TestCase):
    def test_stop_below_pivot(self):
        stop = minervini.minervini_stop_price(100.0)
        self.assertEqual(stop, round(100.0 * (1.0 - config.MINERVINI_STOP_BUFFER_PCT), 2))
        self.assertLess(stop, 100.0)

    def test_none_pivot_returns_none(self):
        self.assertIsNone(minervini.minervini_stop_price(None))


# ── Relative strength (None-tolerant) ────────────────────────────────────────
class TestRelativeStrength(unittest.TestCase):
    def test_none_benchmark_returns_none(self):
        self.assertIsNone(minervini.relative_strength_rank(uptrend(), None))

    def test_outperformer_ranks_high(self):
        rank = minervini.relative_strength_rank(uptrend(g=0.004), uptrend(g=0.001))
        self.assertIsNotNone(rank)
        self.assertGreaterEqual(rank, 70.0)

    def test_underperformer_ranks_low(self):
        rank = minervini.relative_strength_rank(uptrend(g=0.003), uptrend(g=0.006))
        self.assertIsNotNone(rank)
        self.assertLess(rank, 70.0)


# ── evaluate_entry (top-level verdict) ───────────────────────────────────────
class TestEvaluateEntry(unittest.TestCase):
    def test_clean_uptrend_no_benchmark_passes_stage2(self):
        v = minervini.evaluate_entry(uptrend())
        self.assertTrue(v.stage2_ok)
        self.assertIsNone(v.rs_rank)            # no benchmark -> RS skipped, not blocked
        self.assertEqual(v.reasons, [])
        self.assertIsNotNone(v.pivot_low)

    def test_rs_below_min_blocks_even_on_clean_trend(self):
        v = minervini.evaluate_entry(uptrend(g=0.003), benchmark_df=uptrend(g=0.006))
        self.assertFalse(v.stage2_ok)
        self.assertIn("rs_below_min", v.reasons)
        self.assertLess(v.rs_rank, config.MINERVINI_RS_RANK_MIN)

    def test_strong_rs_keeps_stage2(self):
        v = minervini.evaluate_entry(uptrend(g=0.004), benchmark_df=uptrend(g=0.001))
        self.assertTrue(v.stage2_ok)
        self.assertNotIn("rs_below_min", v.reasons)

    def test_tiny_frame_does_not_raise(self):
        v = minervini.evaluate_entry(frame([100.0, 101.0, 102.0]))
        self.assertFalse(v.stage2_ok)
        self.assertIsNone(v.pivot_low)


# ── No-lookahead: the forming (last) bar must never change the verdict ────────
class TestClosedBarNoLookahead(unittest.TestCase):
    def _corrupt_last_bar(self, df: pd.DataFrame) -> pd.DataFrame:
        b = df.copy()
        last = b.index[-1]
        b.loc[last, ["Open", "High", "Low", "Close"]] = [1.0, 1.0, 0.5, 1.0]  # absurd crash bar
        return b

    def test_corrupting_forming_bar_does_not_change_closed_verdict(self):
        clean = uptrend()
        corrupted = self._corrupt_last_bar(clean)
        va = minervini.evaluate_entry(clean)              # closed_bar=True (default)
        vb = minervini.evaluate_entry(corrupted)          # last bar dropped -> identical input
        self.assertEqual(va.stage2_ok, vb.stage2_ok)
        self.assertTrue(va.stage2_ok)
        self.assertEqual(va.pivot_low, vb.pivot_low)
        self.assertEqual(va.reasons, vb.reasons)

    def test_disabling_closed_bar_makes_the_forming_bar_matter(self):
        clean = uptrend()
        corrupted = self._corrupt_last_bar(clean)
        # With closed_bar=False the corrupted final bar IS included, proving the
        # default drop is what protects against forming-bar lookahead.
        self.assertTrue(minervini.evaluate_entry(clean, closed_bar=False).stage2_ok)
        self.assertFalse(minervini.evaluate_entry(corrupted, closed_bar=False).stage2_ok)


if __name__ == "__main__":
    unittest.main(verbosity=2)
