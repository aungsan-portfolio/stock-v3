"""
test_gbm_tradeability.py — offline tests for the Arm B tradeability evaluation.

Pure stdlib ``unittest``, deterministic, no network, no IBKR. Covers:
  * metric primitives: profit factor, max drawdown, expectancy, and that a cost in
    bps shifts every per-trade return by exactly bps/1e4;
  * trade selection: threshold mask, top-k-per-day with deterministic tie-break and
    short-day handling, and that top-k uses only same-day probabilities;
  * REUSE / equivalence: build_tradeable_panel produces (X, y, dates, symbols)
    byte-identical to gbm_pooled_ab_test._build_panel for Arm B, and
    panel_oos_records yields the SAME (label, prob, symbol) as the A/B test's
    _panel_pooled_preds — so the predictions graded here are Arm B's predictions;
  * leakage/consistency: forward return == Close[t+H]/Close[t]-1 and, in binary
    mode, label == (fwd_ret > MIN_PROFIT_MARGIN) on every kept row;
  * read-only safety: no ib_insync/ibkr_bridge import (in-process AND fresh
    interpreter), no order/live-switch/model-write tokens in source, and a --smoke
    run writes ONLY the two report files, leaves the served models untouched, is
    deterministic, and prints the safety line.

    python -m unittest test_gbm_tradeability -v
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import config
import eval_metrics
import gbm_pooled_ab_test as gbm
import gbm_tradeability as gtr
from test_features_regime import make_ohlcv, make_market

# In-process: importing the module must not pull the IBKR path.
_IN_PROCESS_MODULES = set(sys.modules)


def _light_gbm():
    """Fast, deterministic GBM for tests (mirrors test_gbm_pooled._light_gbm)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=80, learning_rate=0.1, min_samples_leaf=20, max_leaf_nodes=15,
        early_stopping=False, random_state=0,
    )


def _signal_panel(n_syms=3, n_dates=500, seed=0) -> dict:
    """A learnable pooled panel with extra fwd_ret/entry/exit columns carried
    through the CV (values are arbitrary — they must survive selection intact)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates).to_numpy()
    X_p, y_p, d_p, s_p, f_p, e_p, x_p = [], [], [], [], [], [], []
    for k in range(n_syms):
        f = rng.normal(size=n_dates)
        X = np.column_stack([f, rng.normal(size=(n_dates, 3))])
        prob = 1.0 / (1.0 + np.exp(-3.0 * f))
        y = (rng.uniform(size=n_dates) < prob).astype(int)
        entry = 100.0 + rng.normal(size=n_dates)
        fwd = rng.normal(0.0, 0.02, size=n_dates)
        X_p.append(X); y_p.append(y); d_p.append(idx)
        s_p.append(np.full(n_dates, f"SYM{k}", dtype=object))
        f_p.append(fwd); e_p.append(entry); x_p.append(entry * (1.0 + fwd))
    return {
        "X": np.vstack(X_p), "y": np.concatenate(y_p), "dates": np.concatenate(d_p),
        "symbols": np.concatenate(s_p), "fwd_ret": np.concatenate(f_p),
        "entry": np.concatenate(e_p), "exit": np.concatenate(x_p),
    }


class _FlagRestoreMixin:
    """Snapshot/restore the Item-3 feature flags so panel builders (which mutate
    config via _set_flags) cannot leak global state across tests."""
    def setUp(self):
        super().setUp()
        self._flag_snapshot = {
            n: getattr(config, n) for n in
            ("USE_REGIME_FEATURES", "USE_MARKET_RELATIVE_FEATURES",
             "USE_CANDLESTICK_FEATURES")
        }

    def tearDown(self):
        for n, v in self._flag_snapshot.items():
            setattr(config, n, v)
        super().tearDown()


# ─────────────────────────── metric primitives ───────────────────────────
class TestMetricPrimitives(unittest.TestCase):
    def test_profit_factor_basic(self):
        pf = gtr._profit_factor(np.array([0.1, -0.05, 0.2, -0.1]))
        self.assertAlmostEqual(pf, 0.3 / 0.15)

    def test_profit_factor_no_losses_is_none(self):
        self.assertIsNone(gtr._profit_factor(np.array([0.1, 0.2, 0.0])))

    def test_profit_factor_no_gains_is_zero(self):
        self.assertEqual(gtr._profit_factor(np.array([-0.1, -0.2])), 0.0)

    def test_max_drawdown_curve(self):
        # equity [0,0.1,0.05,0.25,0.15]; peak [0,0.1,0.1,0.25,0.25]; max dd 0.1
        dd = gtr._max_drawdown(np.array([0.1, -0.05, 0.2, -0.1]))
        self.assertAlmostEqual(dd, 0.1)

    def test_max_drawdown_leading_loss_counts(self):
        # first trade negative -> drawdown below starting capital
        self.assertAlmostEqual(gtr._max_drawdown(np.array([-0.2, 0.05])), 0.2)

    def test_max_drawdown_all_gains_zero(self):
        self.assertEqual(gtr._max_drawdown(np.array([0.1, 0.1, 0.1])), 0.0)

    def test_max_drawdown_empty(self):
        self.assertEqual(gtr._max_drawdown(np.array([])), 0.0)


class TestStrategyMetrics(unittest.TestCase):
    def setUp(self):
        self.gross = np.array([0.1, -0.05, 0.2, -0.1])
        self.label = np.array([1, 0, 1, 0])
        self.dates = pd.bdate_range("2020-01-01", periods=4).to_numpy()
        self.syms = np.array(["A", "B", "C", "D"], dtype=object)

    def test_zero_cost_block(self):
        m = gtr.strategy_metrics(self.gross, self.label, self.dates, self.syms,
                                 base_pos_rate=0.5, cost_bps=0.0, horizon=5)
        self.assertEqual(m["n_trades"], 4)
        self.assertAlmostEqual(m["hit_rate"], 0.5)
        self.assertAlmostEqual(m["avg_return"], 0.0375)
        self.assertAlmostEqual(m["median_return"], 0.025)
        self.assertAlmostEqual(m["profit_factor"], 2.0)
        self.assertAlmostEqual(m["expectancy"], 0.0375)
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["lift"], 1.0)
        self.assertEqual(m["n_losses"], 2)

    def test_cost_shifts_every_return_by_bps(self):
        m0 = gtr.strategy_metrics(self.gross, self.label, self.dates, self.syms,
                                  0.5, 0.0, 5)
        m5 = gtr.strategy_metrics(self.gross, self.label, self.dates, self.syms,
                                  0.5, 5.0, 5)
        # 5 bps = 0.0005 deducted from the mean
        self.assertAlmostEqual(m5["avg_return"], m0["avg_return"] - 0.0005, places=6)
        # and from the total over 4 trades
        self.assertAlmostEqual(m5["total_net_return"],
                               m0["total_net_return"] - 4 * 0.0005, places=6)

    def test_empty_selection_is_all_none(self):
        m = gtr.strategy_metrics(np.array([]), np.array([]), np.array([]),
                                 np.array([], dtype=object), 0.5, 5.0, 5)
        self.assertEqual(m["n_trades"], 0)
        self.assertIsNone(m["expectancy"])
        self.assertIsNone(m["profit_factor"])
        self.assertIsNone(m["max_drawdown"])

    def test_annualization_scales_by_horizon(self):
        m = gtr.strategy_metrics(self.gross, self.label, self.dates, self.syms,
                                 0.5, 0.0, horizon=5)
        self.assertAlmostEqual(m["ann_return_approx"],
                               round(0.0375 * (252 / 5), 4))


# ─────────────────────────── trade selection ───────────────────────────
class TestSelection(unittest.TestCase):
    def test_threshold_mask(self):
        prob = np.array([0.4, 0.55, 0.6, 0.7])
        np.testing.assert_array_equal(gtr.select_threshold(prob, 0.55),
                                      np.array([False, True, True, True]))

    def test_topk_picks_highest_per_day(self):
        # day d0 has 3 rows (probs .9,.5,.7), day d1 has 2 rows (.6,.8); k=2
        dates = np.array(["d0", "d0", "d0", "d1", "d1"])
        prob = np.array([0.9, 0.5, 0.7, 0.6, 0.8])
        syms = np.array(["A", "B", "C", "D", "E"], dtype=object)
        mask = gtr.select_topk_per_day(prob, dates, syms, k=2)
        # d0 -> 0.9 (A) and 0.7 (C); d1 -> both (only 2 rows)
        np.testing.assert_array_equal(mask, np.array([True, False, True, True, True]))

    def test_topk_short_day_takes_all(self):
        dates = np.array(["d0", "d0"])
        prob = np.array([0.9, 0.5])
        syms = np.array(["A", "B"], dtype=object)
        mask = gtr.select_topk_per_day(prob, dates, syms, k=5)
        np.testing.assert_array_equal(mask, np.array([True, True]))

    def test_topk_deterministic_symbol_tiebreak(self):
        # identical probs -> tie broken by symbol ascending; B before C
        dates = np.array(["d0", "d0", "d0"])
        prob = np.array([0.6, 0.6, 0.6])
        syms = np.array(["C", "B", "A"], dtype=object)
        mask = gtr.select_topk_per_day(prob, dates, syms, k=2)
        # picks A and B (lowest symbols), not C
        np.testing.assert_array_equal(mask, np.array([False, True, True]))

    def test_topk_zero_selects_nothing(self):
        dates = np.array(["d0", "d0"])
        prob = np.array([0.9, 0.5])
        syms = np.array(["A", "B"], dtype=object)
        self.assertFalse(gtr.select_topk_per_day(prob, dates, syms, k=0).any())


# ─────────────────── REUSE: equivalence with Arm B machinery ───────────────────
class TestPanelEquivalence(_FlagRestoreMixin, unittest.TestCase):
    def _patched(self, n=400):
        idx = pd.bdate_range("2020-01-01", periods=n)
        spy = make_market(idx, seed=11)

        def fake_fetch(symbol, *a, **k):
            return make_ohlcv(n=n, seed=abs(hash(symbol)) % 9999)

        def fake_market(*a, **k):
            return spy

        return fake_fetch, fake_market, spy

    def test_shared_columns_match_build_panel(self):
        fake_fetch, fake_market, spy = self._patched()
        syms = ["AAA", "BBB", "CCC"]
        with mock.patch.object(gbm, "fetch_ohlcv", side_effect=fake_fetch), \
             mock.patch.object(gtr, "fetch_ohlcv", side_effect=fake_fetch):
            ref = gbm._build_panel(syms, market_rel=True, candles=False,
                                   mode="binary", horizon=5, spy_df=spy)
            mine = gtr.build_tradeable_panel(syms, candles=False, mode="binary",
                                             horizon=5, spy_df=spy)
        self.assertIsNotNone(ref)
        self.assertIsNotNone(mine)
        self.assertEqual(mine["feature_cols"], ref["feature_cols"])
        self.assertEqual(len(ref["feature_cols"]), 18)  # Arm B = base14 + 4 rel
        np.testing.assert_array_equal(mine["X"], ref["X"])
        np.testing.assert_array_equal(mine["y"], ref["y"])
        np.testing.assert_array_equal(mine["dates"], ref["dates"])
        np.testing.assert_array_equal(mine["symbols"], ref["symbols"])

    def test_forward_return_and_label_consistency(self):
        fake_fetch, fake_market, spy = self._patched()
        syms = ["AAA", "BBB"]
        with mock.patch.object(gtr, "fetch_ohlcv", side_effect=fake_fetch):
            panel = gtr.build_tradeable_panel(syms, candles=False, mode="binary",
                                              horizon=5, spy_df=spy)
        # fwd_ret == exit/entry - 1 exactly
        np.testing.assert_allclose(panel["fwd_ret"],
                                   panel["exit"] / panel["entry"] - 1.0, rtol=0, atol=0)
        # binary label == (fwd_ret > MIN_PROFIT_MARGIN) on every kept row (proves the
        # forward return / price columns align with the labels — no look-ahead drift)
        expect = (panel["fwd_ret"] > config.MIN_PROFIT_MARGIN).astype(int)
        np.testing.assert_array_equal(panel["y"], expect)
        self.assertTrue(np.isfinite(panel["entry"]).all())
        self.assertTrue(np.isfinite(panel["exit"]).all())
        self.assertTrue(np.isfinite(panel["fwd_ret"]).all())


class TestOosRecordEquivalence(unittest.TestCase):
    def test_labels_probs_symbols_match_ab_pooled_preds(self):
        panel = _signal_panel(n_syms=3, n_dates=500, seed=0)
        ref = gbm._panel_pooled_preds(panel["X"], panel["y"], panel["dates"],
                                      panel["symbols"], horizon=5, n_splits=3,
                                      purge=5, factory=_light_gbm)
        oos = gtr.panel_oos_records(panel, horizon=5, n_splits=3, purge=5,
                                    factory=_light_gbm)
        self.assertIsNotNone(ref)
        self.assertIsNotNone(oos)
        ref_y, ref_p, ref_s = ref
        np.testing.assert_array_equal(oos["label"], ref_y)
        np.testing.assert_allclose(oos["prob"], ref_p, rtol=0, atol=0)
        np.testing.assert_array_equal(oos["symbol"], ref_s)
        # carried columns are aligned and complete
        self.assertEqual(len(oos["fwd_ret"]), len(ref_y))
        self.assertEqual(len(oos["date"]), len(ref_y))
        self.assertTrue((oos["fold"] >= 0).all())

    def test_runs_are_identical(self):
        panel = _signal_panel(n_syms=2, n_dates=300, seed=2)
        a = gtr.panel_oos_records(panel, 5, 3, 5, _light_gbm)
        b = gtr.panel_oos_records(panel, 5, 3, 5, _light_gbm)
        np.testing.assert_allclose(a["prob"], b["prob"], rtol=0, atol=0)
        np.testing.assert_array_equal(a["label"], b["label"])


# ─────────────────── sweep over a synthetic OOS pool ───────────────────
class TestRunSweep(unittest.TestCase):
    def _oos(self):
        panel = _signal_panel(n_syms=3, n_dates=500, seed=1)
        return gtr.panel_oos_records(panel, 5, 3, 5, _light_gbm)

    def test_sweep_shapes_and_cost_levels(self):
        oos = self._oos()
        sweep = gtr.run_sweep(oos, (0.55, 0.65), (1, 3), (0.0, 5.0, 10.0), horizon=5)
        # 2 thresholds x 3 costs, 2 ks x 3 costs, 1 baseline x 3 costs
        self.assertEqual(len(sweep["threshold"]), 6)
        self.assertEqual(len(sweep["top_k"]), 6)
        self.assertEqual(len(sweep["baseline"]), 3)
        for r in sweep["threshold"] + sweep["top_k"] + sweep["baseline"]:
            self.assertIn(r["cost_bps"], (0.0, 5.0, 10.0))
            self.assertIn("expectancy", r)

    def test_baseline_trades_every_row(self):
        oos = self._oos()
        sweep = gtr.run_sweep(oos, (0.55,), (1,), (0.0,), horizon=5)
        self.assertEqual(sweep["baseline"][0]["n_trades"], len(oos["prob"]))
        self.assertEqual(sweep["baseline"][0]["selection"], "all")

    def test_higher_cost_never_raises_expectancy(self):
        oos = self._oos()
        sweep = gtr.run_sweep(oos, (0.55,), (), (0.0, 10.0), horizon=5)
        by_cost = {r["cost_bps"]: r for r in sweep["threshold"]}
        if by_cost[0.0]["expectancy"] is not None:
            self.assertGreaterEqual(by_cost[0.0]["expectancy"],
                                    by_cost[10.0]["expectancy"])


class TestVerdict(unittest.TestCase):
    """The verdict must be OUTLIER-ROBUST: an edge counts only when the median net
    return and precision-lift beat the trade-everything baseline, not just the
    fat-tail mean."""
    @staticmethod
    def _row(selection, param, cost, expectancy, median, lift, n=100):
        return {"selection": selection, "param": param, "cost_bps": float(cost),
                "expectancy": expectancy, "median_return": median, "lift": lift,
                "n_trades": n, "hit_rate": 0.5, "profit_factor": 1.2}

    def test_high_mean_but_median_at_baseline_is_no_robust_edge(self):
        # top_k=1 archetype: huge mean, median ~ baseline, lift < 1 -> NOT an edge
        sweep = {
            "baseline": [self._row("all", "all", 5.0, 0.006, 0.003, 1.0)],
            "threshold": [],
            "top_k": [self._row("top_k", 1, 5.0, 0.041, 0.0027, 0.991)],
        }
        v = gtr._verdict(sweep)
        self.assertIn("NO ROBUST EDGE", v)
        self.assertIn("OUTLIER-DRIVEN", v)

    def test_median_and_lift_beat_baseline_is_selection_edge(self):
        sweep = {
            "baseline": [self._row("all", "all", 5.0, 0.006, 0.003, 1.0)],
            "threshold": [self._row("threshold", 0.7, 5.0, 0.0101, 0.0088, 1.116),
                          self._row("threshold", 0.75, 5.0, 0.0118, 0.0129, 1.176)],
            "top_k": [],
        }
        v = gtr._verdict(sweep)
        self.assertIn("SELECTION EDGE", v)
        self.assertIn("0.75", v)  # best by median is the 0.75 bucket
        self.assertIn("NOT a go-live signal", v)

    def test_all_negative_is_no_tradeable_edge(self):
        sweep = {
            "baseline": [self._row("all", "all", 5.0, -0.001, -0.002, 0.9)],
            "threshold": [self._row("threshold", 0.65, 5.0, -0.002, -0.003, 0.95)],
            "top_k": [],
        }
        self.assertIn("NO TRADEABLE EDGE", gtr._verdict(sweep))

    def test_no_costed_selection_is_inconclusive(self):
        sweep = {  # only 0 bps rows -> nothing at >=5 bps
            "baseline": [self._row("all", "all", 0.0, 0.01, 0.005, 1.0)],
            "threshold": [self._row("threshold", 0.65, 0.0, 0.01, 0.008, 1.1)],
            "top_k": [],
        }
        self.assertIn("INCONCLUSIVE", gtr._verdict(sweep))

    def test_outlier_flag_detects_mean_dwarfing_median(self):
        self.assertTrue(gtr._is_outlier_driven(
            {"expectancy": 0.041, "median_return": 0.0027}))
        self.assertFalse(gtr._is_outlier_driven(
            {"expectancy": 0.0118, "median_return": 0.0129}))


# ─────────────────── end-to-end offline run ───────────────────
class TestEndToEndOffline(_FlagRestoreMixin, unittest.TestCase):
    def _fakes(self, n=400):
        idx = pd.bdate_range("2020-01-01", periods=n)
        spy = make_market(idx, seed=11)

        def fake_fetch(symbol, *a, **k):
            return make_ohlcv(n=n, seed=abs(hash(symbol)) % 9999)

        def fake_market(*a, **k):
            return spy
        return fake_fetch, fake_market

    def test_smoke_writes_two_files_safely_and_deterministically(self):
        served = [config.ML_MODELS_FILE, config.LSTM_CKPT_FILE,
                  config.MODEL_METRICS_FILE]
        before = {str(p): (Path(p).stat().st_mtime_ns if Path(p).exists() else None)
                  for p in served}
        fake_fetch, fake_market = self._fakes()

        def _run(tmp):
            with mock.patch.object(gtr, "fetch_ohlcv", side_effect=fake_fetch), \
                 mock.patch.object(gtr, "fetch_market_benchmark", side_effect=fake_market), \
                 mock.patch.object(config, "REPORTS_DIR", Path(tmp)):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = gtr.main(["--smoke"])
                return rc, buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp:
            rc, out = _run(tmp)
            self.assertEqual(rc, 0)
            md = Path(tmp) / "gbm_tradeability_report.md"
            js = Path(tmp) / "gbm_tradeability_report.json"
            self.assertTrue(md.exists())
            self.assertTrue(js.exists())
            # ONLY the two report files were written
            written = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(written,
                             ["gbm_tradeability_report.json", "gbm_tradeability_report.md"])
            self.assertIn("No IBKR connection was made, no orders placed, no model saved.",
                          out)
            doc = json.loads(js.read_text(encoding="utf-8"))
            self.assertEqual(doc["meta"]["arm"], "B")
            self.assertEqual(len(doc["meta"]["feature_cols"]), 18)
            self.assertIn("threshold_sweep", doc)
            self.assertIn("top_k_sweep", doc)
            self.assertIn("oos_predictions", doc)
            first_json = js.read_text(encoding="utf-8")

            # determinism: a second run yields byte-identical JSON (no timestamp in it)
            rc2, _ = _run(tmp)
            self.assertEqual(rc2, 0)
            self.assertEqual(js.read_text(encoding="utf-8"), first_json)

        after = {str(p): (Path(p).stat().st_mtime_ns if Path(p).exists() else None)
                 for p in served}
        self.assertEqual(before, after, "served model files must not change")


# ─────────────────── read-only safety ───────────────────
class TestReadOnlySafety(unittest.TestCase):
    def test_no_ibkr_modules_in_process(self):
        for banned in ("ib_insync", "ibkr_bridge"):
            self.assertNotIn(banned, _IN_PROCESS_MODULES,
                             f"{banned} must not be imported by gbm_tradeability")

    def test_no_ibkr_modules_fresh_interpreter(self):
        code = (
            "import gbm_tradeability, sys\n"
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
        src = Path(gtr.__file__).read_text(encoding="utf-8")
        for token in ("placeOrder", "COACH_LIVE_TRADING_ENABLED",
                      "REQUIRE_PAPER_PORT", "LIVE_ACCOUNT_ID", "IBKR_PORT"):
            self.assertNotIn(token, src, f"must not reference {token}")

    def test_source_has_no_model_write_tokens(self):
        # Scan write VECTORS only — the docstring legitimately NAMES the served
        # files it promises NOT to touch (rf_models.joblib etc.).
        src = Path(gtr.__file__).read_text(encoding="utf-8")
        for token in ("ML_MODELS_FILE", "LSTM_CKPT_FILE", "joblib.dump",
                      "_atomic_dump", "torch.save"):
            self.assertNotIn(token, src, f"must not write models via {token}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
