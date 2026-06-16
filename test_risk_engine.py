"""
test_risk_engine.py - Phase 3 pure tests for the account-risk layer (H1, H3).

Fully offline and deterministic. Covers:
  * risk_engine  - drawdown halt + per-symbol exposure cap (pure math)
  * risk_state   - start-of-day equity snapshot (file-backed, restart-safe) and
                   the daily-loss-blocked decision built on it

Run with:
    python -m unittest test_risk_engine -v
"""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import risk_engine
import risk_state


# ── 1. drawdown halt (H3 groundwork) ─────────────────────────────────────────
class TestDrawdownHalt(unittest.TestCase):
    def test_no_drawdown_when_flat_or_up(self):
        self.assertFalse(risk_engine.drawdown_halt_breached(100000, 100000, 0.10))
        self.assertFalse(risk_engine.drawdown_halt_breached(100000, 105000, 0.10))

    def test_halts_at_or_above_cap(self):
        self.assertTrue(risk_engine.drawdown_halt_breached(100000, 90000, 0.10))   # 10% == cap
        self.assertTrue(risk_engine.drawdown_halt_breached(100000, 80000, 0.10))   # 20% > cap

    def test_below_cap_does_not_halt(self):
        self.assertFalse(risk_engine.drawdown_halt_breached(100000, 95000, 0.10))  # 5% < cap

    def test_missing_baseline_or_disabled_cap_never_halts(self):
        self.assertFalse(risk_engine.drawdown_halt_breached(0, 50000, 0.10))       # no baseline
        self.assertFalse(risk_engine.drawdown_halt_breached(100000, 1, 0.0))       # cap disabled

    def test_drawdown_fraction(self):
        self.assertAlmostEqual(risk_engine.drawdown_fraction(100000, 90000), 0.10)
        self.assertEqual(risk_engine.drawdown_fraction(100000, 120000), 0.0)


# ── 2. per-symbol exposure cap (H3 groundwork) ───────────────────────────────
class TestExposureCap(unittest.TestCase):
    def test_within_cap_ok(self):
        self.assertFalse(risk_engine.symbol_exposure_exceeded(5000, 0, 100000, 0.10))  # 5% < 10%

    def test_over_cap_blocked(self):
        self.assertTrue(risk_engine.symbol_exposure_exceeded(12000, 0, 100000, 0.10))  # 12% > 10%

    def test_existing_plus_new_counts(self):
        self.assertTrue(risk_engine.symbol_exposure_exceeded(6000, 6000, 100000, 0.10))  # 12% > 10%

    def test_unknown_equity_or_disabled_never_blocks(self):
        self.assertFalse(risk_engine.symbol_exposure_exceeded(5000, 0, 0, 0.10))
        self.assertFalse(risk_engine.symbol_exposure_exceeded(99999, 0, 100000, 0.0))


# ── 3. start-of-day equity snapshot + daily-loss-blocked (H1) ────────────────
class TestStartOfDaySnapshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "daily_risk_state.json"
        mock.patch.object(risk_state, "_STATE_PATH", self.path).start()
        mock.patch.object(config, "MAX_DAILY_LOSS_USD", 150.0).start()
        self.addCleanup(mock.patch.stopall)

    def test_snapshot_persists_and_is_idempotent_per_day(self):
        stored = risk_state.snapshot_start_of_day_equity(100000.0, day="2026-06-16")
        self.assertEqual(stored, 100000.0)
        # A later (mid-day restart) call must NOT overwrite the baseline.
        again = risk_state.snapshot_start_of_day_equity(98000.0, day="2026-06-16")
        self.assertEqual(again, 100000.0)
        self.assertEqual(risk_state.start_of_day_equity(day="2026-06-16"), 100000.0)

    def test_snapshot_ignores_nonpositive_equity(self):
        self.assertEqual(risk_state.snapshot_start_of_day_equity(0.0, day="2026-06-16"), 0.0)
        self.assertEqual(risk_state.snapshot_start_of_day_equity(-5.0, day="2026-06-16"), 0.0)

    def test_daily_loss_blocked_only_after_snapshot(self):
        # No snapshot yet -> cannot claim a breach.
        self.assertFalse(risk_state.daily_loss_blocked(50000.0, day="2026-06-16"))
        risk_state.snapshot_start_of_day_equity(100000.0, day="2026-06-16")
        # Loss of exactly the cap blocks; a smaller loss does not.
        self.assertTrue(risk_state.daily_loss_blocked(100000.0 - 150.0, day="2026-06-16"))
        self.assertFalse(risk_state.daily_loss_blocked(100000.0 - 149.0, day="2026-06-16"))

    def test_snapshot_is_per_date(self):
        risk_state.snapshot_start_of_day_equity(100000.0, day="2026-06-16")
        self.assertEqual(risk_state.start_of_day_equity(day="2026-06-17"), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
