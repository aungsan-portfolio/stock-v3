"""
test_permutation_test.py — Item 2 shuffled-label null test.

Covers:
  * benjamini_hochberg: textbook step-up FDR behavior + edge cases.
  * ai_engine.cv_pooled_auc: high AUC on a separable signal, ~0.5 on noise.
  * permutation_test.evaluate_symbol: a real signal beats its shuffled-label null
    (small p, real_auc > null_mean); pure noise does not (real_auc ~ null_mean,
    not FDR-significant). _build_xy is mocked so NO network access occurs.

Pure stdlib `unittest`, deterministic, offline. Run with:
    python -m unittest test_permutation_test -v
"""
import unittest
from unittest import mock

import numpy as np

import permutation_test
from ai_engine import cv_pooled_auc


def _signal_xy(n=500, seed=0):
    """A learnable dataset: label driven by the first feature (+ noise)."""
    rng = np.random.default_rng(seed)
    f = rng.normal(size=n)
    noise = rng.normal(size=(n, 3))
    X = np.column_stack([f, noise])
    prob = 1.0 / (1.0 + np.exp(-3.0 * f))
    y = (rng.uniform(size=n) < prob).astype(int)
    return X, y


def _noise_xy(n=500, seed=1):
    """Pure noise: features carry no information about the label."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = rng.integers(0, 2, size=n)
    return X, y


class TestBenjaminiHochberg(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(permutation_test.benjamini_hochberg([], 0.05), [])

    def test_all_significant(self):
        sig = permutation_test.benjamini_hochberg([0.001, 0.002, 0.003], 0.05)
        self.assertEqual(sig, [True, True, True])

    def test_none_significant(self):
        sig = permutation_test.benjamini_hochberg([0.4, 0.5, 0.6], 0.05)
        self.assertEqual(sig, [False, False, False])

    def test_partial_rejection_preserves_order(self):
        # sorted thresholds q*k/m = 0.0125,0.025,0.0375,0.05; only p=0.001 clears.
        sig = permutation_test.benjamini_hochberg([0.001, 0.2, 0.3, 0.4], 0.05)
        self.assertEqual(sig, [True, False, False, False])

    def test_step_up_rescues_middle(self):
        # Largest k with p_(k) <= q*k/m rejects ALL ranks <= k, even a middle p
        # that individually exceeds its own threshold.
        sig = permutation_test.benjamini_hochberg([0.01, 0.024, 0.02], 0.05)
        self.assertEqual(sig, [True, True, True])


class TestCvPooledAuc(unittest.TestCase):
    def test_signal_scores_high(self):
        X, y = _signal_xy()
        auc = cv_pooled_auc(X, y, horizon=5)
        self.assertIsNotNone(auc)
        self.assertGreater(auc, 0.65)

    def test_noise_scores_near_half(self):
        X, y = _noise_xy()
        auc = cv_pooled_auc(X, y, horizon=5)
        self.assertIsNotNone(auc)
        self.assertGreater(auc, 0.40)
        self.assertLess(auc, 0.60)

    def test_too_short_returns_none(self):
        self.assertIsNone(cv_pooled_auc(np.zeros((3, 2)), np.array([0, 1, 0]), horizon=5))


class TestEvaluateSymbol(unittest.TestCase):
    def test_signal_beats_null(self):
        with mock.patch.object(permutation_test, "_build_xy", return_value=_signal_xy()):
            r = permutation_test.evaluate_symbol("SIG", n_shuffles=30, horizon=5, seed=0)
        self.assertIsNotNone(r)
        self.assertGreater(r["real_auc"], r["null_mean"])
        self.assertGreater(r["real_auc"], r["null_p95"])
        self.assertLessEqual(r["p_value"], 0.05)

    def test_noise_does_not_beat_null(self):
        with mock.patch.object(permutation_test, "_build_xy", return_value=_noise_xy()):
            r = permutation_test.evaluate_symbol("NOISE", n_shuffles=30, horizon=5, seed=0)
        self.assertIsNotNone(r)
        # real AUC sits inside the null spread -> not significant.
        self.assertLessEqual(r["real_auc"], r["null_p95"])
        self.assertGreater(r["p_value"], 0.05)

    def test_single_class_returns_none(self):
        X = np.random.default_rng(0).normal(size=(300, 3))
        y = np.ones(300, dtype=int)
        with mock.patch.object(permutation_test, "_build_xy", return_value=(X, y)):
            r = permutation_test.evaluate_symbol("ONE", n_shuffles=5, horizon=5, seed=0)
        self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()
