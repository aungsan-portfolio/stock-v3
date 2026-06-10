"""
test_logic.py — Unit tests for the production-refactor logic.

Pure stdlib `unittest` (no extra dependency). Run with either:

    python -m unittest test_logic -v
    python test_logic.py

These tests exercise the refactored logic only; they do not change it. Network
and trained-model files are avoided by feeding synthetic OHLCV data and stubbing
the per-symbol model predictions, so the suite is deterministic and offline.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import config
from errors import ModelNotAvailableError


# ── Synthetic data helpers ───────────────────────────────────────────
def make_ohlcv(n: int = 200, seed: int = 7) -> pd.DataFrame:
    """Deterministic random-walk OHLCV with a DatetimeIndex.

    Long enough to survive build_features() (which drops the first ~SMA_LONG
    rows) and make_labels()/next-day-return alignment.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0005, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1.0 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.005, size=n)))
    open_ = close * (1.0 + rng.normal(0, 0.003, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ── 1. apply_position_rule_with_hold — long-only behavior ────────────
class TestPositionRuleLongOnly(unittest.TestCase):
    def setUp(self):
        from predictor import apply_position_rule_with_hold
        self.step = apply_position_rule_with_hold

    def test_flat_buy_opens_long(self):
        pos, executed, note, held = self.step(0, "BUY", False, 0, 1)
        self.assertEqual(pos, 1)
        self.assertTrue(executed)
        self.assertEqual(held, 1)
        self.assertEqual(note, "open-long")

    def test_buy_while_long_is_noop_hold(self):
        pos, executed, note, held = self.step(1, "BUY", False, 3, 1)
        self.assertEqual(pos, 1)
        self.assertFalse(executed)
        self.assertEqual(held, 4)  # keeps incrementing
        self.assertIn("already long", note)

    def test_flat_sell_long_only_is_skipped(self):
        pos, executed, note, held = self.step(0, "SELL", False, 0, 1)
        self.assertEqual(pos, 0)
        self.assertFalse(executed)
        self.assertEqual(held, 0)
        self.assertIn("long-only", note)

    def test_sell_closes_long_when_hold_satisfied(self):
        # min_hold=1, already held 1 bar → SELL closes to flat.
        pos, executed, note, held = self.step(1, "SELL", False, 1, 1)
        self.assertEqual(pos, 0)
        self.assertTrue(executed)
        self.assertEqual(held, 0)
        self.assertEqual(note, "close-long")

    def test_hold_signal_increments_when_in_position(self):
        pos, executed, note, held = self.step(1, "HOLD", False, 2, 1)
        self.assertEqual(pos, 1)
        self.assertFalse(executed)
        self.assertEqual(held, 3)
        self.assertEqual(note, "hold")

    def test_hold_signal_flat_stays_flat(self):
        pos, executed, note, held = self.step(0, "HOLD", False, 0, 1)
        self.assertEqual(pos, 0)
        self.assertFalse(executed)
        self.assertEqual(held, 0)
        self.assertEqual(note, "flat")

    def test_long_only_never_opens_short_over_a_sequence(self):
        pos, held = 0, 0
        for sig in ["SELL", "HOLD", "SELL", "BUY", "SELL", "SELL"]:
            pos, _, _, held = self.step(pos, sig, False, held, 1)
            self.assertGreaterEqual(pos, 0, "long-only must never go short")


# ── 2. apply_position_rule_with_hold — short-enabled behavior ────────
class TestPositionRuleShortEnabled(unittest.TestCase):
    def setUp(self):
        from predictor import apply_position_rule_with_hold
        self.step = apply_position_rule_with_hold

    def test_flat_sell_opens_short_when_allowed(self):
        pos, executed, note, held = self.step(0, "SELL", True, 0, 1)
        self.assertEqual(pos, -1)
        self.assertTrue(executed)
        self.assertEqual(held, 1)
        self.assertEqual(note, "open-short")

    def test_sell_while_short_is_noop_hold(self):
        pos, executed, note, held = self.step(-1, "SELL", True, 2, 1)
        self.assertEqual(pos, -1)
        self.assertFalse(executed)
        self.assertEqual(held, 3)
        self.assertIn("already short", note)

    def test_buy_covers_short_when_hold_satisfied(self):
        pos, executed, note, held = self.step(-1, "BUY", True, 1, 1)
        self.assertEqual(pos, 0)
        self.assertTrue(executed)
        self.assertEqual(held, 0)
        self.assertEqual(note, "cover-short")

    def test_no_single_bar_flip_short_to_long(self):
        # Covering a short returns to flat, never directly to +1.
        pos, _, _, _ = self.step(-1, "BUY", True, 5, 1)
        self.assertEqual(pos, 0)


# ── 3. min_hold bars guard ───────────────────────────────────────────
class TestMinHoldGuard(unittest.TestCase):
    def setUp(self):
        from predictor import apply_position_rule_with_hold
        self.step = apply_position_rule_with_hold

    def test_opposite_signal_ignored_until_min_hold(self):
        min_hold = 3
        pos, _, _, held = self.step(0, "BUY", False, 0, min_hold)
        self.assertEqual((pos, held), (1, 1))

        # Two early SELLs must NOT close (held 1→2→3, all < or == boundary).
        pos, executed, note, held = self.step(pos, "SELL", False, held, min_hold)
        self.assertFalse(executed)
        self.assertEqual((pos, held), (1, 2))
        self.assertIn("min_hold 2/3", note)

        pos, executed, note, held = self.step(pos, "SELL", False, held, min_hold)
        self.assertFalse(executed)
        self.assertEqual((pos, held), (1, 3))

        # Now held == min_hold → next SELL closes.
        pos, executed, note, held = self.step(pos, "SELL", False, held, min_hold)
        self.assertTrue(executed)
        self.assertEqual((pos, held), (0, 0))
        self.assertEqual(note, "close-long")

    def test_position_held_exactly_min_hold_bars(self):
        """Open at bar 0, the close must execute on the bar where held==min_hold."""
        min_hold = 5
        pos, held = 0, 0
        pos, _, _, held = self.step(pos, "BUY", False, held, min_hold)  # bar 0: open
        close_bar = None
        for bar in range(1, 12):
            pos, executed, _, held = self.step(pos, "SELL", False, held, min_hold)
            if executed:
                close_bar = bar
                break
        self.assertEqual(close_bar, min_hold)

    def test_min_hold_floor_is_one(self):
        # min_hold <= 0 is clamped to 1 so a held position can still be closed.
        pos, executed, _, _ = self.step(1, "SELL", False, 1, 0)
        self.assertTrue(executed)


# ── 4 & 5. Ensemble: model availability → action ─────────────────────
class TestEnsembleModelAvailability(unittest.TestCase):
    def setUp(self):
        from predictor import (
            enough_ml_models, ml_model_count, weighted_blend, _safe_score,
        )
        self.enough = enough_ml_models
        self.count = ml_model_count
        self.blend = weighted_blend
        self.safe = _safe_score

    # --- building blocks ---
    def test_both_missing_not_enough(self):
        self.assertFalse(self.enough(False, False))

    def test_one_present_is_enough_with_default_threshold(self):
        self.assertEqual(config.MIN_ML_MODELS_FOR_SIGNAL, 1)
        self.assertTrue(self.enough(True, False))
        self.assertTrue(self.enough(False, True))

    def test_safe_score_marks_missing_model_as_not_ok(self):
        def raises():
            raise ModelNotAvailableError("no model")
        score, ok = self.safe(raises)
        self.assertFalse(ok)
        self.assertEqual(score, 0.5)

    def test_safe_score_marks_finite_score_ok(self):
        score, ok = self.safe(lambda: 0.83)
        self.assertTrue(ok)
        self.assertAlmostEqual(score, 0.83)

    def test_blend_ignores_missing_ml_weights(self):
        # With both ML missing the result must equal the technical score.
        result = self.blend(0.95, False, 0.95, False, 0.20)
        self.assertAlmostEqual(result, 0.20, places=6)

    # --- 4. integration: both ML models missing → forced HOLD ---
    def test_predict_all_forces_hold_when_both_models_missing(self):
        from predictor import Predictor
        with mock.patch("predictor.fetch_ohlcv", return_value=make_ohlcv()):
            p = Predictor()
            mock.patch.object(
                p.rf, "predict",
                side_effect=ModelNotAvailableError("no rf"),
            ).start()
            mock.patch.object(
                p.lstm, "predict",
                side_effect=ModelNotAvailableError("no lstm"),
            ).start()
            signals = p.predict_all(symbols=["TEST"])

        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.action, "HOLD")
        self.assertEqual(sig.confidence, 0.5)
        self.assertIn("forced HOLD", sig.reason)

    # --- 5. integration: one ML model available → real ensemble ---
    def test_predict_all_uses_ensemble_when_one_model_available(self):
        from predictor import Predictor
        with mock.patch("predictor.fetch_ohlcv", return_value=make_ohlcv()):
            p = Predictor()
            mock.patch.object(p.rf, "predict", return_value=0.9).start()
            mock.patch.object(
                p.lstm, "predict",
                side_effect=ModelNotAvailableError("no lstm"),
            ).start()
            signals = p.predict_all(symbols=["TEST"])

        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.rf_score, 0.9)
        # Not the forced-HOLD path: an ensemble was actually computed.
        self.assertIn("ensemble=", sig.reason)
        self.assertNotIn("forced HOLD", sig.reason)
        # LSTM marked uncertain/missing in the reason string.
        self.assertIn("LSTM=0.50?", sig.reason)

    def tearDown(self):
        mock.patch.stopall()


# ── 6. backtest report/CSV write behavior ────────────────────────────
class TestBacktestReportWrite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reports = Path(self.tmp.name)

    def tearDown(self):
        mock.patch.stopall()
        self.tmp.cleanup()

    def test_writes_empty_report_when_no_symbol_has_data(self):
        from backtest import run_backtest
        mock.patch.object(config, "REPORTS_DIR", self.reports).start()
        mock.patch(
            "backtest.fetch_ohlcv",
            side_effect=ValueError("no data"),
        ).start()

        result = run_backtest(symbols=["NODATA"], verbose=False)

        metrics_path = self.reports / "backtest_metrics.json"
        trades_path = self.reports / "backtest_trades.csv"
        self.assertTrue(metrics_path.exists())
        self.assertTrue(trades_path.exists())
        self.assertEqual(result["symbols_tested"], 0)
        # metrics JSON is valid and round-trips.
        on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["symbols_tested"], 0)
        self.assertEqual(on_disk["backtest_model"], "RF_TECHNICAL_WALK_FORWARD")
        # empty trades file when there are no rows.
        self.assertEqual(trades_path.read_text(encoding="utf-8"), "")

    def test_writes_populated_csv_when_rows_exist(self):
        from backtest import run_backtest
        df = make_ohlcv(n=600, seed=11)
        mock.patch.object(config, "REPORTS_DIR", self.reports).start()
        mock.patch("backtest.fetch_ohlcv", return_value=df).start()

        result = run_backtest(
            symbols=["SYNTH"], train_min=120, step=30,
            verbose=False, include_lstm=False,
        )

        metrics_path = self.reports / "backtest_metrics.json"
        trades_path = self.reports / "backtest_trades.csv"
        self.assertTrue(metrics_path.exists())
        self.assertTrue(trades_path.exists())
        self.assertGreaterEqual(result["symbols_tested"], 1)

        trades = pd.read_csv(trades_path)
        self.assertGreater(len(trades), 0)
        # Trade ledger carries the expected schema columns.
        for col in ("symbol", "signal", "new_position", "order_executed", "equity"):
            self.assertIn(col, trades.columns)
        # min_hold reported in metrics equals config value.
        self.assertEqual(result["min_hold_bars"], config.MIN_HOLD_BARS)


# ── 7. config.MIN_HOLD_BARS = ML_HORIZON ─────────────────────────────
class TestConfigMinHold(unittest.TestCase):
    def test_min_hold_bars_equals_ml_horizon(self):
        self.assertEqual(config.MIN_HOLD_BARS, config.ML_HORIZON)

    def test_min_hold_bars_is_positive_int(self):
        self.assertIsInstance(config.MIN_HOLD_BARS, int)
        self.assertGreaterEqual(config.MIN_HOLD_BARS, 1)


# ── 8. import smoke tests ────────────────────────────────────────────
class TestImportSmoke(unittest.TestCase):
    def test_core_modules_import(self):
        import config            # noqa: F401
        import errors            # noqa: F401
        import data_manager      # noqa: F401
        import logging_setup     # noqa: F401

    def test_predictor_imports_and_exposes_helper(self):
        import predictor
        self.assertTrue(hasattr(predictor, "apply_position_rule_with_hold"))
        self.assertTrue(hasattr(predictor, "weighted_blend"))
        self.assertTrue(hasattr(predictor, "Predictor"))

    def test_backtest_imports_cleanly(self):
        # This used to raise ImportError (missing apply_position_rule_with_hold).
        import backtest
        self.assertTrue(hasattr(backtest, "run_backtest"))

    def test_main_imports_cleanly(self):
        import main
        self.assertTrue(hasattr(main, "main"))

    def test_ibkr_bridge_imports_cleanly(self):
        # ib_insync's eventkit dependency touches the asyncio event loop at
        # import time; under Python 3.12+ a loop must already be set in the
        # current thread (the real `paper` entrypoint does this in main.py).
        # This is an environment quirk, not part of the refactored logic.
        import asyncio
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        import ibkr_bridge
        self.assertTrue(hasattr(ibkr_bridge, "IBKRBridge"))

    def test_engines_expose_shared_rf_builder(self):
        import ai_engine
        self.assertTrue(hasattr(ai_engine, "build_rf"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
