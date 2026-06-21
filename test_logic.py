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


# ── 2b. apply_position_rule_with_hold — hard-stop backstop ───────────
class TestHardStopBypass(unittest.TestCase):
    def setUp(self):
        from predictor import apply_position_rule_with_hold
        self.step = apply_position_rule_with_hold

    def test_hard_stop_closes_long_bypassing_min_hold(self):
        # Long held only 1 bar (min_hold=5) but price fell 5% (> 3% stop) →
        # the hard stop must close it immediately despite the hold guard.
        pos, executed, note, held = self.step(
            1, "HOLD", False, 1, 5,
            entry_price=100.0, current_price=95.0, hard_stop_pct=0.03,
        )
        self.assertEqual(pos, 0)
        self.assertTrue(executed)
        self.assertEqual(held, 0)
        self.assertTrue(note.startswith("hard-stop"))

    def test_hard_stop_bypasses_even_a_buy_signal(self):
        # A BUY arriving on a deeply underwater long still triggers the stop.
        pos, executed, note, _ = self.step(
            1, "BUY", False, 10, 5,
            entry_price=50.0, current_price=40.0, hard_stop_pct=0.03,
        )
        self.assertEqual(pos, 0)
        self.assertTrue(executed)
        self.assertTrue(note.startswith("hard-stop"))

    def test_no_bypass_when_loss_within_threshold(self):
        # Down 2% only (< 3% stop) → normal min_hold rules apply, no hard stop.
        pos, executed, note, held = self.step(
            1, "HOLD", False, 1, 5,
            entry_price=100.0, current_price=98.0, hard_stop_pct=0.03,
        )
        self.assertEqual(pos, 1)
        self.assertFalse(executed)
        self.assertEqual(held, 2)
        self.assertFalse(note.startswith("hard-stop"))

    def test_no_bypass_when_hard_stop_pct_is_none(self):
        # Disabled hard stop → falls through to the ordinary hold logic.
        pos, executed, note, _ = self.step(
            1, "HOLD", False, 1, 5,
            entry_price=100.0, current_price=50.0, hard_stop_pct=None,
        )
        self.assertEqual(pos, 1)
        self.assertFalse(executed)
        self.assertFalse(note.startswith("hard-stop"))

    def test_no_bypass_without_prices(self):
        # Missing entry/current price → hard stop cannot evaluate, no bypass.
        pos, executed, _, _ = self.step(
            1, "HOLD", False, 1, 5,
            entry_price=None, current_price=None, hard_stop_pct=0.03,
        )
        self.assertEqual(pos, 1)
        self.assertFalse(executed)

    def test_hard_stop_does_not_touch_short_positions(self):
        # Backstop is long-only: an adverse short move is not force-closed here.
        pos, executed, _, _ = self.step(
            -1, "HOLD", True, 1, 5,
            entry_price=100.0, current_price=200.0, hard_stop_pct=0.03,
        )
        self.assertEqual(pos, -1)
        self.assertFalse(executed)

    def test_backward_compatible_without_hard_stop_args(self):
        # Existing 5-arg callers keep working unchanged (hard stop defaults off).
        pos, executed, note, held = self.step(0, "BUY", False, 0, 1)
        self.assertEqual((pos, executed, note, held), (1, True, "open-long", 1))


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
        # A single AVAILABLE, positively-weighted model is enough. RF qualifies;
        # a zero-weighted LSTM does not — see the Item-6b voter tests below.
        self.assertTrue(self.enough(True, False))

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

    # --- Item 6b: WEIGHT_LSTM = 0 (LSTM val AUC sub-chance) ---
    def _expected_rf_tech(self, rf, tech):
        # weighted_blend renormalizes over the present weights; with LSTM at 0
        # only RF + technical survive, so the blend is their ratio-weighted mean.
        wr, wt = config.WEIGHT_RF, config.WEIGHT_TECHNICAL
        return (rf * wr + tech * wt) / (wr + wt)

    def test_lstm_weight_is_zero(self):
        # Pins the Item 6b decision: LSTM contributes no weight to the ensemble.
        self.assertEqual(config.WEIGHT_LSTM, 0.0)

    def test_blend_drops_lstm_even_when_present(self):
        # Even with lstm_ok=True, a zero LSTM weight must remove it from the math:
        # the blend is identical regardless of the LSTM score, and equals the
        # RF+technical renormalized mean. (RF and technical paths stay intact.)
        expected = self._expected_rf_tech(0.80, 0.40)
        low_lstm = self.blend(0.80, True, 0.00, True, 0.40)
        high_lstm = self.blend(0.80, True, 1.00, True, 0.40)
        self.assertAlmostEqual(low_lstm, high_lstm, places=6)
        self.assertAlmostEqual(low_lstm, expected, places=6)

    def test_blend_rf_and_technical_renormalize(self):
        # RF present, LSTM absent: confidence is the RF:technical renormalized
        # mean (weights need not sum to 1.0). Explicit numeric guard.
        result = self.blend(0.80, True, 0.99, False, 0.40)
        self.assertAlmostEqual(result, (0.80 * 0.40 + 0.40 * 0.25) / 0.65, places=6)

    # Item-6b safety tighten: a zero-weighted model is NOT an availability voter.
    def test_lstm_only_not_enough_when_lstm_weight_zero(self):
        # LSTM-only (RF missing) with WEIGHT_LSTM=0 must FAIL MIN_ML_MODELS_FOR_SIGNAL
        # — a zero-weight model adds nothing to the blend, so letting it satisfy the
        # voter would amount to trading on the technical score alone.
        self.assertEqual(config.WEIGHT_LSTM, 0.0)
        self.assertEqual(config.MIN_ML_MODELS_FOR_SIGNAL, 1)
        self.assertEqual(self.count(False, True), 0)
        self.assertFalse(self.enough(False, True))

    def test_rf_only_is_enough_via_positive_weight(self):
        # RF carries a positive weight, so an RF-only symbol still trades.
        self.assertGreater(config.WEIGHT_RF, 0)
        self.assertEqual(self.count(True, False), 1)
        self.assertTrue(self.enough(True, False))

    def test_both_missing_still_not_enough(self):
        self.assertEqual(self.count(False, False), 0)
        self.assertFalse(self.enough(False, False))

    def test_rf_and_lstm_enough_through_rf_weight(self):
        # Both available: RF's nonzero weight satisfies the voter on its own; the
        # zero-weighted LSTM is not counted, so the count is 1 (RF only).
        self.assertEqual(self.count(True, True), 1)
        self.assertTrue(self.enough(True, True))

    def test_predict_all_forces_hold_when_only_zero_weight_lstm_available(self):
        # Integration: RF missing, LSTM available but zero-weighted → not enough ML
        # models → forced HOLD (never reaches the blend / gate). Guards the tighten
        # end-to-end through predict_all.
        from predictor import Predictor
        with mock.patch("predictor.fetch_ohlcv", return_value=make_ohlcv()):
            p = Predictor()
            mock.patch.object(
                p.rf, "predict",
                side_effect=ModelNotAvailableError("no rf"),
            ).start()
            mock.patch.object(p.lstm, "predict", return_value=0.9).start()
            signals = p.predict_all(symbols=["TEST"])

        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertEqual(sig.action, "HOLD")
        self.assertEqual(sig.confidence, 0.5)
        self.assertIn("forced HOLD", sig.reason)

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


# ── 6b. Feature normalization (data_manager.build_features) ──────────
class TestFeatureNormalization(unittest.TestCase):
    def setUp(self):
        from data_manager import build_features, get_feature_columns
        self.feat = build_features(make_ohlcv(n=300, seed=3))
        self.cols = get_feature_columns()

    def test_raw_ema_dropped_dist_ema_added(self):
        # Raw, price-scaled `ema` must no longer be a model feature; the
        # close-normalized `dist_ema` replaces it.
        self.assertNotIn("ema", self.cols)
        self.assertIn("dist_ema", self.cols)
        self.assertIn("dist_ema", self.feat.columns)

    def test_dist_ema_is_close_normalized(self):
        expected = (self.feat["Close"] - self.feat["ema"]) / self.feat["Close"]
        np.testing.assert_allclose(
            self.feat["dist_ema"].values, expected.values, rtol=1e-9, atol=1e-12
        )

    def test_sma_cross_is_close_normalized(self):
        expected = (self.feat["sma_short"] - self.feat["sma_long"]) / self.feat["Close"]
        np.testing.assert_allclose(
            self.feat["sma_cross"].values, expected.values, rtol=1e-9, atol=1e-12
        )

    def test_macd_family_is_close_normalized(self):
        # All three MACD features are small (divided by close), not raw dollars.
        for col in ("macd", "macd_sig", "macd_hist"):
            self.assertIn(col, self.feat.columns)
            self.assertLess(
                self.feat[col].abs().max(), 1.0,
                f"{col} should be close-normalized, not raw price magnitude",
            )

    def test_macd_hist_equals_macd_minus_signal(self):
        np.testing.assert_allclose(
            self.feat["macd_hist"].values,
            (self.feat["macd"] - self.feat["macd_sig"]).values,
            rtol=1e-9, atol=1e-12,
        )

    def test_all_feature_columns_present_and_finite(self):
        for col in self.cols:
            self.assertIn(col, self.feat.columns)
        self.assertTrue(np.isfinite(self.feat[self.cols].values).all())


# ── 6c. make_labels MIN_PROFIT_MARGIN threshold ──────────────────────
class TestMakeLabelsThreshold(unittest.TestCase):
    def setUp(self):
        from data_manager import make_labels
        self.make_labels = make_labels

    def _df(self, closes):
        idx = pd.bdate_range("2021-01-01", periods=len(closes))
        return pd.DataFrame({"Close": np.asarray(closes, dtype=float)}, index=idx)

    def test_small_gain_below_margin_is_label_zero(self):
        # +0.1% over the horizon, margin 0.3% → not a BUY (must be 0).
        df = self._df([100.0, 100.1, 100.1])
        labels = self.make_labels(df, horizon=1, min_profit_margin=0.003)
        self.assertEqual(labels.iloc[0], 0.0)

    def test_gain_above_margin_is_label_one(self):
        # +1% over the horizon, margin 0.3% → BUY (label 1).
        df = self._df([100.0, 101.0, 101.0])
        labels = self.make_labels(df, horizon=1, min_profit_margin=0.003)
        self.assertEqual(labels.iloc[0], 1.0)

    def test_future_unknown_rows_are_nan(self):
        df = self._df([100.0, 101.0, 102.0])
        labels = self.make_labels(df, horizon=1, min_profit_margin=0.003)
        self.assertTrue(np.isnan(labels.iloc[-1]))

    def test_default_margin_comes_from_config(self):
        # A move exactly at +MIN_PROFIT_MARGIN is NOT > margin → label 0;
        # a clearly larger move is label 1. Confirms config wiring.
        m = config.MIN_PROFIT_MARGIN
        df = self._df([100.0, 100.0 * (1.0 + m), 100.0 * (1.0 + m + 0.01)])
        labels = self.make_labels(df, horizon=1)
        self.assertEqual(labels.iloc[0], 0.0)   # exactly at margin, not above
        self.assertEqual(labels.iloc[1], 1.0)   # above margin

    def test_higher_margin_labels_fewer_positives(self):
        df = make_ohlcv(n=200, seed=5)
        low = self.make_labels(df, horizon=5, min_profit_margin=0.0)
        high = self.make_labels(df, horizon=5, min_profit_margin=0.05)
        self.assertGreaterEqual(
            int(low.sum()), int(high.sum()),
            "a higher profit margin must not increase the positive count",
        )


# ── 6d. make_labels triple-barrier (path-aware) mode ─────────────────
class TestTripleBarrierLabels(unittest.TestCase):
    """Opt-in path-aware label. entry=Close[t]; tp_pct=0.015 -> upper=101.5,
    stop_pct=0.004 -> lower=99.6 when entry=100. Stays binary {0,1,NaN}."""

    def setUp(self):
        from data_manager import make_labels
        self.make_labels = make_labels

    def _df(self, bars):
        # bars: list of (high, low, close); Open mirrors Close (unused by labels).
        idx = pd.bdate_range("2021-01-01", periods=len(bars))
        close = [b[2] for b in bars]
        return pd.DataFrame(
            {
                "Open": close,
                "High": [b[0] for b in bars],
                "Low": [b[1] for b in bars],
                "Close": close,
                "Volume": [1_000_000.0] * len(bars),
            },
            index=idx,
        )

    def _labels(self, bars, horizon=2):
        return self.make_labels(
            self._df(bars), horizon=horizon, mode="triple_barrier",
            tp_pct=0.015, stop_pct=0.004,
        )

    def test_tp_before_stop_is_one(self):
        # bar1 High 102 >= 101.5 (TP) and Low 100 > 99.6 (no stop) -> 1.0
        labels = self._labels([(100, 100, 100), (102.0, 100.0, 101.0), (101, 100, 100.5)])
        self.assertEqual(labels.iloc[0], 1.0)

    def test_stop_before_tp_is_zero(self):
        # bar1 Low 99 <= 99.6 (stop) and High 100.5 < 101.5 (no TP) -> 0.0
        labels = self._labels([(100, 100, 100), (100.5, 99.0, 99.5), (101, 100, 100.5)])
        self.assertEqual(labels.iloc[0], 0.0)

    def test_same_bar_tie_resolves_to_stop(self):
        # bar1 hits BOTH barriers -> pessimistic stop -> 0.0
        labels = self._labels([(100, 100, 100), (102.0, 99.0, 100.0), (101, 100, 100.5)])
        self.assertEqual(labels.iloc[0], 0.0)

    def test_neither_barrier_within_horizon_is_zero(self):
        labels = self._labels([(100, 100, 100), (101.0, 99.8, 100.2), (101.0, 99.8, 100.3)])
        self.assertEqual(labels.iloc[0], 0.0)

    def test_future_unknown_rows_are_nan(self):
        labels = self._labels([(100, 100, 100), (102, 100, 101), (101, 100, 100.5)])
        self.assertTrue(np.isnan(labels.iloc[-1]))
        self.assertTrue(np.isnan(labels.iloc[-2]))

    def test_only_binary_values_and_nan(self):
        df = make_ohlcv(n=120, seed=3)
        labels = self.make_labels(df, horizon=5, mode="triple_barrier")
        valid = labels.dropna().unique()
        self.assertTrue(set(valid).issubset({0.0, 1.0}),
                        f"triple-barrier must stay binary, got {valid}")

    def test_requires_high_low_columns(self):
        df = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0]},
            index=pd.bdate_range("2021-01-01", periods=3),
        )
        with self.assertRaises(ValueError):
            self.make_labels(df, horizon=1, mode="triple_barrier")

    def test_default_mode_stays_binary(self):
        # With LABEL_MODE default 'binary', a +1% endpoint move over horizon=1 is 1.0
        # even though High/Low are present (binary path ignores them).
        labels = self.make_labels(
            self._df([(100, 100, 100), (101.0, 100.0, 101.0), (101, 100, 101)]),
            horizon=1,
        )
        self.assertEqual(labels.iloc[0], 1.0)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            self.make_labels(self._df([(100, 100, 100), (101, 100, 101)]),
                             horizon=1, mode="nonsense")


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
