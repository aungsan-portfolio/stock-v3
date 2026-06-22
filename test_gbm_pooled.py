"""
test_gbm_pooled.py — offline tests for the pooled cross-symbol GBM edge test.

Pure stdlib ``unittest``, deterministic, no network, no IBKR. Covers:
  * leakage-safe panel CV mechanics: date-axis fold bounds, purge drops exactly the
    right boundary dates, and the split is by DATE not row order;
  * statistical sanity: a learnable signal panel scores high & beats its null, pure
    noise does not;
  * read-only safety: importing the module pulls no ib_insync/ibkr_bridge (in-process
    AND in a fresh interpreter), the source has no order/live-switch/model-write
    tokens, and a --smoke run leaves the served model files untouched;
  * the GBM factory + RF-baseline loader behave as specified.

    python -m unittest test_gbm_pooled -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import config
import gbm_pooled_ab_test as gbm
from test_features_regime import make_ohlcv, make_market

# In-process: importing the experiment module must not pull the IBKR path.
_IN_PROCESS_MODULES = set(sys.modules)


def _light_gbm():
    """Fast, deterministic GBM for tests (small/quick vs the production factory)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=80, learning_rate=0.1, min_samples_leaf=20, max_leaf_nodes=15,
        early_stopping=False, random_state=0,
    )


def _panel(n_syms=3, n_dates=500, signal=True, seed=0) -> dict:
    """A pooled panel sharing ONE business-day calendar across symbols. Rows are
    laid out symbol-by-symbol (all of SYM0, then SYM1, ...) so a row-order split
    would be wrong — the date-axis split must interleave them."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates).to_numpy()
    X_parts, y_parts, d_parts, s_parts = [], [], [], []
    for k in range(n_syms):
        f = rng.normal(size=n_dates)
        noise = rng.normal(size=(n_dates, 3))
        X = np.column_stack([f, noise])
        if signal:
            prob = 1.0 / (1.0 + np.exp(-3.0 * f))
            y = (rng.uniform(size=n_dates) < prob).astype(int)
        else:
            y = rng.integers(0, 2, size=n_dates)
        X_parts.append(X)
        y_parts.append(y)
        d_parts.append(idx)
        s_parts.append(np.full(n_dates, f"SYM{k}", dtype=object))
    return {
        "X": np.vstack(X_parts),
        "y": np.concatenate(y_parts),
        "dates": np.concatenate(d_parts),
        "symbols": np.concatenate(s_parts),
    }


class TestDateFoldBounds(unittest.TestCase):
    def test_contiguous_increasing_and_covers_tail(self):
        uniq = np.arange(100)
        bounds = gbm._date_fold_bounds(uniq, 4)
        self.assertEqual(len(bounds), 4)
        los = [int(b[0]) for b in bounds]
        self.assertEqual(los, [20, 40, 60, 80])           # test_size = 100//5 = 20
        # fold k's hi equals fold k+1's lo (contiguous, non-overlapping)
        self.assertEqual(int(bounds[0][1]), 40)
        self.assertEqual(int(bounds[1][1]), 60)
        self.assertEqual(int(bounds[2][1]), 80)
        self.assertIsNone(bounds[3][1])                    # final block runs to the end

    def test_too_short_returns_empty(self):
        self.assertEqual(gbm._date_fold_bounds(np.arange(3), 5), [])


class TestPanelCvLeakageMechanics(unittest.TestCase):
    def test_purge_drops_expected_boundary_dates(self):
        panel = _panel(n_syms=2, n_dates=200, signal=False, seed=1)
        uniq = np.unique(panel["dates"])
        purge = 5
        folds = list(gbm._iter_folds(panel["dates"], n_splits=4, purge=purge))
        self.assertTrue(folds)
        for train_mask, _test_mask, test_lo, purge_date in folds:
            idx_lo = int(np.searchsorted(uniq, test_lo))
            self.assertEqual(purge_date, uniq[max(0, idx_lo - purge)])
            if train_mask.any():
                # no training date reaches into the purge gap / test period
                self.assertLess(panel["dates"][train_mask].max(), purge_date)
            gap = int(np.sum((uniq >= purge_date) & (uniq < test_lo)))
            self.assertEqual(gap, min(purge, idx_lo))

    def test_purge_zero_keeps_boundary(self):
        panel = _panel(n_syms=2, n_dates=200, signal=False, seed=1)
        for _tr, _te, test_lo, purge_date in gbm._iter_folds(panel["dates"], 4, 0):
            self.assertEqual(purge_date, test_lo)   # no gap when purge=0

    def test_split_is_by_date_not_row_order(self):
        # Rows are SYM0 (all dates) then SYM1 (all dates). A row-order split would
        # make the first train block all-SYM0; a date split must contain BOTH.
        panel = _panel(n_syms=2, n_dates=300, signal=False, seed=3)
        train_mask, _test_mask, _lo, _pd = next(
            gbm._iter_folds(panel["dates"], n_splits=3, purge=0))
        self.assertEqual(set(panel["symbols"][train_mask]), {"SYM0", "SYM1"})


class TestPanelCvStatistics(unittest.TestCase):
    def test_signal_panel_scores_high_and_beats_null(self):
        panel = _panel(n_syms=3, n_dates=500, signal=True, seed=0)
        m = gbm._panel_metrics(panel["X"], panel["y"], panel["dates"],
                               panel["symbols"], horizon=5, n_splits=3, purge=5,
                               factory=_light_gbm)
        self.assertIsNotNone(m)
        self.assertGreater(m["pooled_auc"], 0.65)
        null = gbm._null_pvalue(panel, m["pooled_auc"], n_shuffles=24, horizon=5,
                                n_splits=3, purge=5, factory=_light_gbm, seed=0,
                                scope="global")
        self.assertLessEqual(null["p_value"], 0.05)
        self.assertGreater(m["pooled_auc"], null["null_mean"])

    def test_noise_panel_near_half_and_not_significant(self):
        panel = _panel(n_syms=3, n_dates=500, signal=False, seed=1)
        m = gbm._panel_metrics(panel["X"], panel["y"], panel["dates"],
                               panel["symbols"], horizon=5, n_splits=3, purge=5,
                               factory=_light_gbm)
        self.assertIsNotNone(m)
        self.assertGreater(m["pooled_auc"], 0.40)
        self.assertLess(m["pooled_auc"], 0.60)
        null = gbm._null_pvalue(panel, m["pooled_auc"], n_shuffles=12, horizon=5,
                                n_splits=3, purge=5, factory=_light_gbm, seed=0,
                                scope="global")
        self.assertGreater(null["p_value"], 0.05)


class TestGbmFactory(unittest.TestCase):
    def test_is_hgb_with_balanced_weight_and_seed(self):
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = gbm._gbm_factory()
        self.assertIsInstance(clf, HistGradientBoostingClassifier)
        self.assertEqual(clf.class_weight, "balanced")
        self.assertEqual(clf.random_state, config.RANDOM_STATE)


class TestRfBaselineLoader(unittest.TestCase):
    def test_tolerates_symbol_missing_auc(self):
        data = {"rf": {"AAA": {"auc": 0.55, "accuracy": 0.5},
                       "BBB": {"accuracy": 0.5}}}      # BBB has no auc
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model_metrics.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            with mock.patch.object(config, "MODEL_METRICS_FILE", p):
                out = gbm._load_rf_baseline(["AAA", "BBB", "CCC"])
        self.assertEqual(out["n_with_auc"], 1)
        self.assertEqual(out["mean_auc"], 0.55)
        self.assertNotIn("BBB", out["per_symbol"])

    def test_missing_file_is_safe(self):
        with mock.patch.object(config, "MODEL_METRICS_FILE",
                               Path(tempfile.gettempdir()) / "no_such_metrics_xyz.json"):
            out = gbm._load_rf_baseline(["AAA"])
        self.assertIsNone(out["mean_auc"])
        self.assertEqual(out["n_with_auc"], 0)


class TestReadOnlySafety(unittest.TestCase):
    def test_no_ibkr_modules_in_process(self):
        for banned in ("ib_insync", "ibkr_bridge"):
            self.assertNotIn(banned, _IN_PROCESS_MODULES,
                             f"{banned} must not be imported by gbm_pooled_ab_test")

    def test_no_ibkr_modules_fresh_interpreter(self):
        code = (
            "import gbm_pooled_ab_test, sys\n"
            "bad = [m for m in ('ib_insync', 'ibkr_bridge') if m in sys.modules]\n"
            "assert not bad, bad\n"
            "print('OK')\n"
        )
        proc = subprocess.run([sys.executable, "-c", code],
                              cwd=str(Path(__file__).resolve().parent),
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK", proc.stdout)

    def test_source_has_no_order_or_live_switch_tokens(self):
        src = Path(gbm.__file__).read_text(encoding="utf-8")
        for token in ("placeOrder", "COACH_LIVE_TRADING_ENABLED",
                      "REQUIRE_PAPER_PORT", "LIVE_ACCOUNT_ID", "IBKR_PORT"):
            self.assertNotIn(token, src, f"must not reference {token}")

    def test_source_has_no_model_write_tokens(self):
        src = Path(gbm.__file__).read_text(encoding="utf-8")
        # NOTE: do not scan for ".joblib"/"rf_models" — the docstring legitimately
        # NAMES the served files it promises NOT to touch. Scan write VECTORS only.
        for token in ("ML_MODELS_FILE", "LSTM_CKPT_FILE", "joblib.dump",
                      "_atomic_dump", "torch.save"):
            self.assertNotIn(token, src, f"must not write models via {token}")

    def test_smoke_run_is_offline_and_leaves_models_untouched(self):
        served = [config.ML_MODELS_FILE, config.LSTM_CKPT_FILE, config.MODEL_METRICS_FILE]
        before = {str(p): (Path(p).stat().st_mtime_ns if Path(p).exists() else None)
                  for p in served}

        def fake_fetch(symbol, *a, **k):
            return make_ohlcv(n=400, seed=abs(hash(symbol)) % 9999)

        def fake_market(*a, **k):
            return make_market(make_ohlcv(n=400, seed=5).index, seed=11)

        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(gbm, "fetch_ohlcv", side_effect=fake_fetch), \
                 mock.patch.object(gbm, "fetch_market_benchmark", side_effect=fake_market), \
                 mock.patch.object(config, "REPORTS_DIR", Path(tmp)):
                rc = gbm.main(["--smoke", "--rf-baseline", "none"])
            self.assertEqual(rc, 0)
            self.assertTrue((Path(tmp) / "gbm_pooled_ab_report_smoke.md").exists())
            self.assertTrue((Path(tmp) / "gbm_pooled_ab_report_smoke.json").exists())

        after = {str(p): (Path(p).stat().st_mtime_ns if Path(p).exists() else None)
                 for p in served}
        self.assertEqual(before, after, "served model files must not change")


if __name__ == "__main__":
    unittest.main(verbosity=2)
