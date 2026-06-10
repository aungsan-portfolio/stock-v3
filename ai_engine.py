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

import config
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


def build_rf(oob_score: bool = False) -> RandomForestClassifier:
    """Canonical RF factory. Shared by training and the backtest folds so both
    use identical hyperparameters (single source of truth)."""
    return RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        oob_score=oob_score,
        bootstrap=True,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )


class StockRFEngine:
    def __init__(self) -> None:
        self.models: Dict[str, RandomForestClassifier] = {}
        self._lock = threading.RLock()

    def train(self, symbols: Optional[list] = None, verbose: bool = True) -> dict:
        symbols = symbols or config.WATCHLIST
        new_models: Dict[str, RandomForestClassifier] = {}
        results: dict = {}

        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol)
                feat = build_features(df)
                labels = make_labels(feat)

                valid_idx = labels.dropna().index
                X = feat.loc[valid_idx, FEATURE_COLS].values
                y = labels.loc[valid_idx].values.astype(int)

                if len(X) < 100:
                    logger.warning("Not enough data for %s (n=%d)", symbol, len(X))
                    continue
                if len(np.unique(y)) < 2:
                    logger.warning("Only one class in labels for %s — skipping", symbol)
                    continue

                split = int(len(X) * (1 - config.ML_TEST_RATIO))
                X_tr, X_te = X[:split], X[split:]
                y_tr, y_te = y[:split], y[split:]

                if len(X_te) == 0 or len(np.unique(y_tr)) < 2:
                    logger.warning("Invalid RF split for %s — skipping", symbol)
                    continue

                eval_clf = build_rf(oob_score=True)
                eval_clf.fit(X_tr, y_tr)
                pred = eval_clf.predict(X_te)

                # Refit the saved production model on all valid rows after evaluation.
                final_clf = build_rf(oob_score=True)
                final_clf.fit(X, y)

                new_models[symbol] = final_clf
                results[symbol] = {
                    "oob_score": round(float(getattr(final_clf, "oob_score_", float("nan"))), 4),
                    "test_acc": round(float(accuracy_score(y_te, pred)), 4),
                    "test_precision": round(float(precision_score(y_te, pred, zero_division=0)), 4),
                    "test_recall": round(float(recall_score(y_te, pred, zero_division=0)), 4),
                    "test_f1": round(float(f1_score(y_te, pred, zero_division=0)), 4),
                    "n_train_eval": int(len(X_tr)),
                    "n_test_eval": int(len(X_te)),
                    "n_samples": int(len(X)),
                    "positive_rate": round(float(np.mean(y)), 4),
                }
                if verbose:
                    logger.info(
                        "%-6s acc=%.3f f1=%.3f oob=%.3f n=%d",
                        symbol,
                        results[symbol]["test_acc"],
                        results[symbol]["test_f1"],
                        results[symbol]["oob_score"],
                        len(X),
                    )
            except Exception:
                logger.exception("Failed to train RF for %s", symbol)

        if not new_models:
            logger.error("RF training produced no models — keeping existing on-disk file")
            return results

        with self._lock:
            self.models = new_models

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_dump(self.models, config.ML_MODELS_FILE)
        logger.info("RF models saved → %s (n=%d)", config.ML_MODELS_FILE, len(new_models))
        return results

    def load(self) -> bool:
        if not config.ML_MODELS_FILE.exists():
            return False
        try:
            loaded = joblib.load(config.ML_MODELS_FILE)
            with self._lock:
                self.models = loaded
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

        feat = build_features(df)
        if len(feat) < 1:
            raise ValueError(f"Not enough features for {symbol}")

        x = feat[FEATURE_COLS].iloc[-1:].values
        with self._lock:
            clf = self.models.get(symbol)
        if clf is None:
            raise ModelNotAvailableError(f"No RF model for {symbol}")

        proba = clf.predict_proba(x)[0]
        classes = list(clf.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.0
