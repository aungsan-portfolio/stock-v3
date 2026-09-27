"""
tools/evaluate_september_forward.py — Forward Out-of-Sample Evaluation for Frozen Aug-29 RF Models.

Evaluates the frozen on-disk RF models (trained 2026-08-29) on September 2026 out-of-sample data
(2026-09-01 to 2026-09-26) to diagnose staleness and degradation.
Read-only: Zero broker connection, zero orders placed, zero model modification.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import config
import eval_metrics
from data_manager import fetch_ohlcv, build_features, make_labels, get_feature_columns

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("september_eval")


def run_september_eval():
    print("=" * 80)
    print("SEPTEMBER FORWARD EVALUATION (FROZEN AUG-29 RF MODELS)")
    print("=" * 80)
    print(f"Model store path : {config.ML_MODELS_FILE}")
    print(f"Evaluation window: 2026-09-01 -> 2026-09-26")
    print(f"Horizon          : {config.ML_HORIZON} bars | Buy threshold: {config.BUY_THRESHOLD}")
    print(f"Label mode       : {getattr(config, 'LABEL_MODE', 'binary')}")
    print("=" * 80)

    if not config.ML_MODELS_FILE.exists():
        print(f"ERROR: Model file not found at {config.ML_MODELS_FILE}")
        sys.exit(1)

    rf_models = joblib.load(config.ML_MODELS_FILE)
    feature_cols = get_feature_columns()

    rows = []
    pooled_y = []
    pooled_prob = []
    pooled_pred = []

    for sym in config.WATCHLIST:
        if sym not in rf_models:
            print(f"[SKIP] {sym:<6}: No model in rf_models.joblib")
            rows.append({
                "symbol": sym,
                "n_bars": 0,
                "auc": None,
                "prec_at_buy": None,
                "n_at_buy": 0,
                "f1": None,
                "brier": None,
                "pos_rate": None,
                "status": "NO_MODEL"
            })
            continue

        clf = rf_models[sym]

        try:
            df = fetch_ohlcv(sym)
            feat = build_features(df, cfg=config)
            labels = make_labels(feat)

            sep_mask = (feat.index >= "2026-09-01") & labels.notna()
            n_bars = int(sep_mask.sum())

            if n_bars == 0:
                print(f"[SKIP] {sym:<6}: 0 realized September bars")
                rows.append({
                    "symbol": sym, "n_bars": 0, "auc": None, "prec_at_buy": None,
                    "n_at_buy": 0, "f1": None, "brier": None, "pos_rate": None,
                    "status": "NO_SEP_BARS"
                })
                continue

            X_sep = feat.loc[sep_mask, feature_cols].values
            y_sep = labels.loc[sep_mask].values.astype(int)

            prob = eval_metrics.proba_pos(clf, X_sep)
            pred = clf.predict(X_sep)

            auc = eval_metrics.safe_auc(y_sep, prob)
            prec_at_buy, n_at_buy = eval_metrics.precision_at_threshold(y_sep, prob, config.BUY_THRESHOLD)
            f1 = float(f1_score(y_sep, pred, zero_division=0))
            brier = eval_metrics.safe_brier(y_sep, prob)
            pos_rate = float(np.mean(y_sep))

            pooled_y.extend(y_sep.tolist())
            pooled_prob.extend(prob.tolist())
            pooled_pred.extend(pred.tolist())

            rows.append({
                "symbol": sym,
                "n_bars": n_bars,
                "auc": auc,
                "prec_at_buy": prec_at_buy,
                "n_at_buy": n_at_buy,
                "f1": f1,
                "brier": brier,
                "pos_rate": pos_rate,
                "status": "OK"
            })

            auc_str = f"{auc:.4f}" if auc is not None else "N/A (1-class)"
            prec_str = f"{prec_at_buy:.4f} (n={n_at_buy})" if prec_at_buy is not None else f"N/A (n={n_at_buy})"
            print(f"  {sym:<6} | n={n_bars:<2} | AUC={auc_str:<14} | Prec@Buy={prec_str:<18} | F1={f1:.4f} | PosRate={pos_rate:.3f}")

        except Exception as e:
            print(f"[ERROR] {sym:<6}: {e}")
            rows.append({
                "symbol": sym, "n_bars": 0, "auc": None, "prec_at_buy": None,
                "n_at_buy": 0, "f1": None, "brier": None, "pos_rate": None,
                "status": f"ERROR: {e}"
            })

    print("=" * 80)
    print("SUMMARY RESULTS TABLE (MARKDOWN FORMAT)")
    print("=" * 80)
    print("| Symbol | n_sep_bars | AUC | precision_at_buy | F1 | Brier | positive_rate |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] == "OK":
            auc_disp = f"{r['auc']:.4f}" if r["auc"] is not None else "N/A"
            prec_disp = f"{r['prec_at_buy']:.4f} (n={r['n_at_buy']})" if r["prec_at_buy"] is not None else f"None (n={r['n_at_buy']})"
            f1_disp = f"{r['f1']:.4f}" if r["f1"] is not None else "N/A"
            brier_disp = f"{r['brier']:.4f}" if r["brier"] is not None else "N/A"
            pos_disp = f"{r['pos_rate']:.3f}" if r["pos_rate"] is not None else "N/A"
            print(f"| {r['symbol']:<6} | {r['n_bars']:<10} | {auc_disp:<5} | {prec_disp:<16} | {f1_disp:<4} | {brier_disp:<5} | {pos_disp:<13} |")
        else:
            print(f"| {r['symbol']:<6} | 0          | N/A | None (n=0)       | N/A  | N/A   | N/A           |")

    # Pooled metrics
    if pooled_y:
        py = np.array(pooled_y)
        pprob = np.array(pooled_prob)
        ppred = np.array(pooled_pred)

        pooled_auc = eval_metrics.safe_auc(py, pprob)
        pooled_prec, pooled_n_at = eval_metrics.precision_at_threshold(py, pprob, config.BUY_THRESHOLD)
        pooled_f1 = float(f1_score(py, ppred, zero_division=0))
        pooled_brier = eval_metrics.safe_brier(py, pprob)
        pooled_pos = float(np.mean(py))

        print("=" * 80)
        print("POOLED / MACRO AGGREGATE (Across all 20 symbols in September)")
        print("=" * 80)
        print(f"Total evaluated observations: {len(py)}")
        print(f"Pooled AUC                 : {pooled_auc:.4f}" if pooled_auc is not None else "Pooled AUC: N/A")
        print(f"Pooled Precision@Buy       : {pooled_prec:.4f} (n={pooled_n_at})" if pooled_prec is not None else f"Pooled Precision@Buy: None (n={pooled_n_at})")
        print(f"Pooled F1 Score            : {pooled_f1:.4f}")
        print(f"Pooled Brier Score         : {pooled_brier:.4f}")
        print(f"Pooled Positive Rate       : {pooled_pos:.4f}")
        print("=" * 80)


if __name__ == "__main__":
    run_september_eval()
