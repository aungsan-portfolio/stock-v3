"""
eval_metrics.py — shared edge-detection metrics (ROC-AUC / PR-AUC / Brier /
precision-at-threshold) for pooled out-of-sample predictions.

Why this module exists
----------------------
0.5-threshold accuracy is invariant to where a model is confident, but the bot
only ACTS on the high-confidence tail (>= config.BUY_THRESHOLD). The metric that
matches how it trades is ranking/threshold quality — AUC and precision-at-the-
BUY-threshold — not global accuracy. These helpers compute exactly that.

It is deliberately kept separate from model_metrics.py: model_metrics.py is the
fail-closed signal gate and is intentionally stdlib + config only (no heavy ML
imports). The sklearn-dependent math lives here so both ai_engine (training) and
permutation_test (the null check) compute the SAME numbers from one source.

Every helper is guard-railed for the degenerate single-class case that
roc_auc_score / average_precision_score raise on: it returns None instead of
throwing, so a symbol with a one-class fold/holdout never aborts training.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

import config

logger = logging.getLogger(__name__)


def round4(x: Optional[float]) -> Optional[float]:
    """Round to 4 dp, preserving None (so 'not computed' stays distinct from 0.0)."""
    return round(float(x), 4) if x is not None else None


def safe_auc(y_true, y_score) -> Optional[float]:
    """ROC-AUC, or None when undefined (empty / single-class y_true)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        logger.debug("safe_auc failed", exc_info=True)
        return None


def safe_pr_auc(y_true, y_score) -> Optional[float]:
    """Average precision (PR-AUC), or None when undefined."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        logger.debug("safe_pr_auc failed", exc_info=True)
        return None


def safe_brier(y_true, y_score) -> Optional[float]:
    """Brier score (calibration error), or None when undefined."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if len(y_true) == 0:
        return None
    try:
        return float(brier_score_loss(y_true, y_score))
    except Exception:
        logger.debug("safe_brier failed", exc_info=True)
        return None


def precision_at_threshold(y_true, y_score, threshold) -> Tuple[Optional[float], int]:
    """Precision among rows whose score >= ``threshold``.

    This is the metric that matches the live trade trigger (the bot only acts
    above BUY_THRESHOLD). Returns (precision, n_at) where ``precision`` is None
    when no row clears the threshold (n_at == 0) — an empty confident tail is
    reported consistently as None, never a fake 0.0 or a divide-by-zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = y_score >= float(threshold)
    n_at = int(mask.sum())
    if n_at == 0:
        return None, 0
    tp = float(((y_true == 1.0) & mask).sum())
    return tp / float(n_at), n_at


def pooled_classification_metrics(y_true, y_score, threshold: Optional[float] = None) -> dict:
    """Edge-detection metrics for pooled out-of-sample predictions.

    ``threshold`` defaults to config.BUY_THRESHOLD so ``precision_at_buy`` reflects
    the real trade trigger. Single-class inputs yield None AUC/PR-AUC (never raise).
    Returns a dict with keys: auc, pr_auc, brier, precision_at_buy, n_at_buy.
    """
    if threshold is None:
        threshold = float(config.BUY_THRESHOLD)
    prec, n_at = precision_at_threshold(y_true, y_score, threshold)
    return {
        "auc": safe_auc(y_true, y_score),
        "pr_auc": safe_pr_auc(y_true, y_score),
        "brier": safe_brier(y_true, y_score),
        "precision_at_buy": prec,
        "n_at_buy": n_at,
    }


def proba_pos(clf, X) -> Optional[np.ndarray]:
    """Class-1 probabilities from a fitted classifier, or None if it has no
    positive class (a degenerate single-class fit). Avoids the classic
    predict_proba[:, 1] bug when ``clf.classes_`` is not [0, 1]."""
    classes = list(getattr(clf, "classes_", []))
    if 1 not in classes:
        return None
    return clf.predict_proba(X)[:, classes.index(1)].astype(float)
