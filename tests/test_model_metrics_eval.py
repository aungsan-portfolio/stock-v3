"""
test_model_metrics_eval.py — Item 5 edge-detection metrics.

Covers:
  * eval_metrics: pooled AUC / PR-AUC / precision-at-threshold / Brier helpers,
    including the guarded single-class and empty-tail cases.
  * model_metrics: the new RF/LSTM metric keys persist additively and round-trip,
    and the OPTIONAL ROC-AUC gate floor blocks/allows correctly while staying
    inert at its default (0.0).

Pure stdlib `unittest`, fully offline and deterministic. Run with:
    python -m unittest test_model_metrics_eval -v
"""
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config
import eval_metrics
import model_metrics


NOW = dt.date(2026, 6, 16)


# ── 1. eval_metrics helpers ──────────────────────────────────────────
class TestEvalMetrics(unittest.TestCase):
    def test_safe_auc_perfectly_separable_is_one(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(eval_metrics.safe_auc(y, s), 1.0, places=6)

    def test_safe_auc_single_class_is_none(self):
        self.assertIsNone(eval_metrics.safe_auc(np.ones(5), np.linspace(0, 1, 5)))
        self.assertIsNone(eval_metrics.safe_auc(np.array([]), np.array([])))

    def test_precision_at_threshold_basic(self):
        y = np.array([1, 0, 1, 0])
        s = np.array([0.9, 0.8, 0.7, 0.2])
        prec, n_at = eval_metrics.precision_at_threshold(y, s, 0.65)
        self.assertEqual(n_at, 3)               # 0.9, 0.8, 0.7 clear 0.65
        self.assertAlmostEqual(prec, 2.0 / 3.0) # two of those three are y==1

    def test_precision_at_threshold_empty_tail_is_none(self):
        y = np.array([1, 0, 1])
        s = np.array([0.1, 0.2, 0.3])
        prec, n_at = eval_metrics.precision_at_threshold(y, s, 0.65)
        self.assertEqual(n_at, 0)
        self.assertIsNone(prec)

    def test_pooled_metrics_uses_buy_threshold(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        with mock.patch.object(config, "BUY_THRESHOLD", 0.65):
            m = eval_metrics.pooled_classification_metrics(y, s)
        self.assertAlmostEqual(m["auc"], 1.0, places=6)
        self.assertEqual(m["n_at_buy"], 2)
        self.assertAlmostEqual(m["precision_at_buy"], 1.0)

    def test_round4_preserves_none(self):
        self.assertIsNone(eval_metrics.round4(None))
        self.assertEqual(eval_metrics.round4(0.123456), 0.1235)


# ── 2. persistence of the new metric keys ────────────────────────────
class _MetricsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model_metrics.json"
        for name, value in {
            "MODEL_METRICS_FILE": self.path,
            "MODEL_GATE_ENABLED": True,
            "MODEL_MIN_RF_ACCURACY": 0.50,
            "MODEL_MIN_RF_F1": 0.0,
            "MODEL_MIN_RF_AUC": 0.0,
            "MODEL_MAX_AGE_DAYS": 30,
        }.items():
            mock.patch.object(config, name, value).start()
        self.addCleanup(mock.patch.stopall)


class TestNewKeysPersist(_MetricsBase):
    def test_rf_new_keys_round_trip(self):
        model_metrics.save_rf_metrics(
            {"AAPL": {
                "test_acc": 0.55, "test_f1": 0.5, "oob_score": 0.54,
                "auc": 0.62, "pr_auc": 0.58, "brier": 0.24,
                "precision_at_buy": 0.7, "holdout_auc": 0.6, "n_samples": 800,
            }},
            trained_at="2026-06-16",
        )
        rf = model_metrics.load_all()["rf"]["AAPL"]
        self.assertAlmostEqual(rf["accuracy"], 0.55)   # legacy key intact
        self.assertAlmostEqual(rf["auc"], 0.62)
        self.assertAlmostEqual(rf["pr_auc"], 0.58)
        self.assertAlmostEqual(rf["precision_at_buy"], 0.7)
        self.assertAlmostEqual(rf["holdout_auc"], 0.6)

    def test_missing_new_keys_persist_as_none(self):
        # An older-shaped results dict (no auc) must persist auc=None, not crash.
        model_metrics.save_rf_metrics(
            {"MSFT": {"test_acc": 0.51, "test_f1": 0.5, "oob_score": 0.5, "n_samples": 500}},
            trained_at="2026-06-16",
        )
        rf = model_metrics.load_all()["rf"]["MSFT"]
        self.assertIsNone(rf["auc"])
        self.assertIsNone(rf["holdout_auc"])

    def test_lstm_val_auc_round_trips(self):
        model_metrics.save_lstm_metrics(
            {"AAPL": {"best_val_loss": 0.66, "best_val_auc": 0.57, "n_train_seq": 300}},
            trained_at="2026-06-16",
        )
        lstm = model_metrics.load_all()["lstm"]["AAPL"]
        self.assertAlmostEqual(lstm["best_val_loss"], 0.66)
        self.assertAlmostEqual(lstm["best_val_auc"], 0.57)

    def test_rf_and_lstm_still_coexist(self):
        model_metrics.save_rf_metrics(
            {"AAPL": {"test_acc": 0.55, "test_f1": 0.5, "oob_score": 0.5,
                      "auc": 0.6, "n_samples": 700}}, trained_at="2026-06-16")
        model_metrics.save_lstm_metrics(
            {"AAPL": {"best_val_loss": 0.6, "best_val_auc": 0.55, "n_train_seq": 200}},
            trained_at="2026-06-16")
        data = model_metrics.load_all()
        self.assertIn("AAPL", data["rf"])
        self.assertIn("AAPL", data["lstm"])


# ── 3. optional AUC gate floor ───────────────────────────────────────
class TestAUCFloorGate(_MetricsBase):
    def _save(self, acc=0.60, auc=0.60):
        model_metrics.save_rf_metrics(
            {"AAPL": {"test_acc": acc, "test_f1": 0.6, "oob_score": acc,
                      "auc": auc, "n_samples": 800}},
            trained_at="2026-06-16",
        )

    def test_disabled_by_default_auc_floor_zero(self):
        # Default MODEL_MIN_RF_AUC=0.0 -> AUC never blocks, even at coin-flip.
        self._save(acc=0.60, auc=0.50)
        self.assertTrue(model_metrics.evaluate_gate("AAPL", now=NOW).ok)

    def test_auc_below_floor_blocks(self):
        mock.patch.object(config, "MODEL_MIN_RF_AUC", 0.55).start()
        self._save(acc=0.60, auc=0.52)
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "below_threshold")

    def test_auc_above_floor_passes(self):
        mock.patch.object(config, "MODEL_MIN_RF_AUC", 0.55).start()
        self._save(acc=0.60, auc=0.60)
        self.assertTrue(model_metrics.evaluate_gate("AAPL", now=NOW).ok)

    def test_auc_missing_blocks_when_floor_enabled(self):
        mock.patch.object(config, "MODEL_MIN_RF_AUC", 0.55).start()
        # save without auc -> persisted None -> fail-closed under an enabled floor
        model_metrics.save_rf_metrics(
            {"AAPL": {"test_acc": 0.60, "test_f1": 0.6, "oob_score": 0.6, "n_samples": 800}},
            trained_at="2026-06-16",
        )
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)


if __name__ == "__main__":
    unittest.main()
