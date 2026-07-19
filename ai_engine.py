"""
ai_engine.py — Random Forest signal engine. One RF classifier per symbol.

Production hardening:
- Atomic model swap on disk (tmp file + os.replace) to prevent corrupt joblib
  files if the process is killed mid-write.
- Thread-safe in-memory model dict (RLock) for safe concurrent predict + train.
- Graceful per-symbol failure: one bad symbol never aborts the whole batch.
- Time-based train/test split (no shuffling) for evaluation.
- Final production model is refit on all valid historical rows after evaluation,
  so live predictions use the latest available data.
"""
import logging
import os
import threading
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit

import config
import eval_metrics
import model_metrics
from data_manager import (
    fetch_ohlcv,
    build_features,
    make_labels,
    get_feature_columns,
)
from errors import ModelNotAvailableError

logger = logging.getLogger(__name__)
FEATURE_COLS = get_feature_columns()


def _atomic_dump(obj, path) -> None:
    """Write joblib file atomically: write to tmp, fsync, then os.replace."""
    path = str(path)
    tmp = f"{path}.tmp"
    joblib.dump(obj, tmp)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(tmp, path)


def _compatible_rf_models(models: dict, expected_features: int) -> dict:
    compatible = {
        symbol: model
        for symbol, model in models.items()
        if getattr(model, "n_features_in_", None) == expected_features
    }
    skipped = len(models) - len(compatible)
    if skipped:
        logger.warning(
            "Skipped %d RF model(s) with incompatible feature width (expected %d)",
            skipped,
            expected_features,
        )
    return compatible


def _subsample_train_indices(n: int, horizon: int) -> np.ndarray:
    """Decorrelate overlapping horizon labels by keeping every ``horizon``-th
    training row. Labels built with a forward horizon share future bars with
    their neighbors, so adjacent rows are not independent; striding by the
    horizon removes most of that overlap. Applied to TRAINING rows only —
    evaluation/test rows are always kept intact.
    """
    horizon = max(1, int(horizon))
    if n <= 0:
        return np.empty(0, dtype=int)
    return np.arange(0, n, horizon, dtype=int)


def build_rf(oob_score: bool = False, cfg=None) -> RandomForestClassifier:
    """Canonical RF factory. Shared by training and the backtest folds so both
    use identical hyperparameters (single source of truth)."""
    cfg = cfg or config.get_settings()
    return RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_depth=cfg.RF_MAX_DEPTH,
        oob_score=oob_score,
        bootstrap=True,
        class_weight="balanced",
        n_jobs=-1,
        random_state=cfg.RANDOM_STATE,
    )


def cv_pooled_auc(X, y, horizon: Optional[int] = None, n_splits: int = 5, cfg=None) -> Optional[float]:
    """Pooled out-of-sample ROC-AUC from a walk-forward TimeSeriesSplit."""
    cfg = cfg or config.get_settings()
    horizon = int(horizon or cfg.ML_HORIZON)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    if len(X) < n_splits + 1:
        return None

    pooled_y: list = []
    pooled_p: list = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, te_idx in tscv.split(X):
        sub = _subsample_train_indices(len(tr_idx), horizon)
        X_tr, y_tr = X[tr_idx[sub]], y[tr_idx[sub]]
        X_te, y_te = X[te_idx], y[te_idx]
        if len(X_tr) == 0 or len(np.unique(y_tr)) < 2 or len(X_te) == 0:
            continue
        clf = build_rf(cfg=cfg)
        clf.fit(X_tr, y_tr)
        pp = eval_metrics.proba_pos(clf, X_te)
        if pp is not None:
            pooled_y.append(y_te)
            pooled_p.append(pp)

    if not pooled_y:
        return None
    return eval_metrics.safe_auc(np.concatenate(pooled_y), np.concatenate(pooled_p))


class StockRFEngine:
    def __init__(self, settings=None) -> None:
        self.cfg = settings or config.get_settings()
        self.feature_cols = get_feature_columns(self.cfg)
        self.models: Dict[str, RandomForestClassifier] = {}
        self._lock = threading.RLock()
        self._loaded: bool = False

    def _log_loaded_once(self, n: int) -> None:
        if not self._loaded:
            logger.info("RF models loaded (n=%d)", n)
            self._loaded = True

    def train(self, symbols: Optional[list] = None, verbose: bool = True) -> dict:
        symbols = symbols or self.cfg.WATCHLIST
        new_models: Dict[str, RandomForestClassifier] = {}
        results: dict = {}

        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol)
                feat = build_features(df, cfg=self.cfg)
                labels = make_labels(feat)

                valid_idx = labels.dropna().index
                X = feat.loc[valid_idx, self.feature_cols].values
                y = labels.loc[valid_idx].values.astype(int)

                if len(X) < 100:
                    logger.warning("Not enough data for %s (n=%d)", symbol, len(X))
                    continue
                if len(np.unique(y)) < 2:
                    logger.warning("Only one class in labels for %s — skipping", symbol)
                    continue

                # ── Walk-forward cross-validation (no shuffling, no leakage) ──
                horizon = int(self.cfg.ML_HORIZON)
                n_splits = 5
                fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []
                n_train_eval = 0
                n_test_eval = 0

                if len(X) < n_splits + 1:
                    logger.warning("Not enough rows for CV on %s (n=%d)", symbol, len(X))
                    continue

                pooled_y: list = []
                pooled_p: list = []

                tscv = TimeSeriesSplit(n_splits=n_splits)
                for tr_idx, te_idx in tscv.split(X):
                    sub = _subsample_train_indices(len(tr_idx), horizon)
                    tr_idx_sub = tr_idx[sub]
                    X_tr, y_tr = X[tr_idx_sub], y[tr_idx_sub]
                    X_te, y_te = X[te_idx], y[te_idx]

                    if len(X_tr) == 0 or len(np.unique(y_tr)) < 2 or len(X_te) == 0:
                        continue

                    fold_clf = build_rf(cfg=self.cfg)
                    fold_clf.fit(X_tr, y_tr)
                    pred = fold_clf.predict(X_te)

                    fold_acc.append(accuracy_score(y_te, pred))
                    fold_prec.append(precision_score(y_te, pred, zero_division=0))
                    fold_rec.append(recall_score(y_te, pred, zero_division=0))
                    fold_f1.append(f1_score(y_te, pred, zero_division=0))
                    n_train_eval += len(X_tr)
                    n_test_eval += len(X_te)

                    pp = eval_metrics.proba_pos(fold_clf, X_te)
                    if pp is not None:
                        pooled_y.append(y_te)
                        pooled_p.append(pp)

                if not fold_acc:
                    logger.warning("No usable CV folds for %s — skipping", symbol)
                    continue

                if pooled_y:
                    pooled = eval_metrics.pooled_classification_metrics(
                        np.concatenate(pooled_y), np.concatenate(pooled_p)
                    )
                else:
                    pooled = {"auc": None, "pr_auc": None, "brier": None,
                              "precision_at_buy": None, "n_at_buy": 0}

                holdout = {"holdout_auc": None, "holdout_precision_at_buy": None,
                           "n_holdout": 0}
                ratio = float(getattr(self.cfg, "MODEL_HOLDOUT_RATIO", 0.0) or 0.0)
                if 0.0 < ratio < 1.0:
                    cut = int(len(X) * (1.0 - ratio))
                    if cut >= 50 and (len(X) - cut) >= 20:
                        h_sub = _subsample_train_indices(cut, horizon)
                        X_h, y_h = X[:cut][h_sub], y[:cut][h_sub]
                        X_ho, y_ho = X[cut:], y[cut:]
                        if len(np.unique(y_h)) >= 2:
                            graded = build_rf(cfg=self.cfg)
                            graded.fit(X_h, y_h)
                            pp_ho = eval_metrics.proba_pos(graded, X_ho)
                            if pp_ho is not None:
                                hm = eval_metrics.pooled_classification_metrics(y_ho, pp_ho)
                                holdout = {
                                    "holdout_auc": hm["auc"],
                                    "holdout_precision_at_buy": hm["precision_at_buy"],
                                    "n_holdout": int(len(y_ho)),
                                }

                final_sub = _subsample_train_indices(len(X), horizon)
                X_final, y_final = X[final_sub], y[final_sub]
                if len(np.unique(y_final)) < 2:
                    X_final, y_final = X, y
                final_clf = build_rf(oob_score=True, cfg=self.cfg)
                final_clf.fit(X_final, y_final)

                new_models[symbol] = final_clf
                results[symbol] = {
                    "oob_score": round(float(getattr(final_clf, "oob_score_", float("nan"))), 4),
                    "test_acc": round(float(np.mean(fold_acc)), 4),
                    "test_precision": round(float(np.mean(fold_prec)), 4),
                    "test_recall": round(float(np.mean(fold_rec)), 4),
                    "test_f1": round(float(np.mean(fold_f1)), 4),
                    "auc": eval_metrics.round4(pooled["auc"]),
                    "pr_auc": eval_metrics.round4(pooled["pr_auc"]),
                    "brier": eval_metrics.round4(pooled["brier"]),
                    "precision_at_buy": eval_metrics.round4(pooled["precision_at_buy"]),
                    "n_at_buy": int(pooled["n_at_buy"]),
                    "holdout_auc": eval_metrics.round4(holdout["holdout_auc"]),
                    "holdout_precision_at_buy": eval_metrics.round4(holdout["holdout_precision_at_buy"]),
                    "n_holdout": int(holdout["n_holdout"]),
                    "cv_folds": int(len(fold_acc)),
                    "n_train_eval": int(n_train_eval),
                    "n_test_eval": int(n_test_eval),
                    "n_train_final": int(len(X_final)),
                    "n_samples": int(len(X)),
                    "positive_rate": round(float(np.mean(y)), 4),
                }
                if verbose:
                    auc = results[symbol]["auc"]
                    logger.info(
                        "%-6s acc=%.3f f1=%.3f auc=%s oob=%.3f n=%d",
                        symbol,
                        results[symbol]["test_acc"],
                        results[symbol]["test_f1"],
                        f"{auc:.3f}" if auc is not None else "n/a",
                        results[symbol]["oob_score"],
                        len(X),
                    )
            except Exception:
                logger.exception("Failed to train RF for %s", symbol)

        if not new_models:
            logger.error("RF training produced no models — keeping existing on-disk file")
            return results

        existing = {}
        try:
            if self.cfg.ML_MODELS_FILE.exists():
                loaded = joblib.load(self.cfg.ML_MODELS_FILE)
                existing = _compatible_rf_models(loaded, len(self.feature_cols))
        except Exception:
            logger.warning("Could not load existing RF models for merge", exc_info=True)
        merged = dict(existing)
        merged.update(new_models)
        logger.info("RF models merged: %d new + %d existing = %d total",
                      len(new_models), len(existing), len(merged))

        with self._lock:
            self.models = merged

        self.cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_dump(self.models, self.cfg.ML_MODELS_FILE)
        logger.info("RF models saved → %s (n=%d)", self.cfg.ML_MODELS_FILE, len(new_models))
        model_metrics.save_rf_metrics({s: results[s] for s in new_models if s in results})
        return results

    def load(self) -> bool:
        if not self.cfg.ML_MODELS_FILE.exists():
            return False
        try:
            loaded = joblib.load(self.cfg.ML_MODELS_FILE)
            compatible = _compatible_rf_models(loaded, len(self.feature_cols))
            with self._lock:
                self.models = compatible
            logger.info("RF models loaded (n=%d)", len(self.models))
            return True
        except Exception:
            logger.exception("Failed to load RF models")
            return False

    def predict(self, symbol: str, df: Optional[pd.DataFrame] = None) -> float:
        if not self.models and not self.load():
            raise RuntimeError("RF models not trained. Run train() first.")
        if df is None:
            df = fetch_ohlcv(symbol)

        feat = build_features(df, cfg=self.cfg)
        if len(feat) < 1:
            raise ValueError(f"Not enough features for {symbol}")

        x = feat[self.feature_cols].iloc[-1:].values
        with self._lock:
            clf = self.models.get(symbol)
        if clf is None:
            raise ModelNotAvailableError(f"No RF model for {symbol}")

        proba = clf.predict_proba(x)[0]
        classes = list(clf.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.0
