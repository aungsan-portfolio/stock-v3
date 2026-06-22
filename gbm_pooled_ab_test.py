"""
gbm_pooled_ab_test.py — read-only POOLED cross-symbol Gradient-Boosting edge test.

Decisive question: the per-symbol RandomForest models are a coin-flip (mean AUC
~0.507; permutation null finds 0/8 symbols significant). The leading hypothesis is
per-symbol DATA STARVATION — each symbol trains on only ~1,200 rows / ~300 positive
examples, too few for a tree model to find a weak signal. This experiment POOLS the
feature/label panel across the whole symbol universe (~108 symbols × ~1,200 rows ≈
130k rows) and trains ONE HistGradientBoosting model. If any learnable
cross-sectional structure exists, a pooled GBM is far likelier to surface it than
108 starved per-symbol RFs.

Arms (both scored with the SAME leakage-safe panel CV + shuffled-label null):
    A : pooled GBM, base feature set (regime/market-rel/candlestick OFF; --candles
        on to include the 16 candlestick columns)
    B : pooled GBM, base + 4 SPY relative-strength features (rel_ret_short/long,
        rs_slope, mkt_trend) — wires fetch_market_benchmark() -> market_df.
Baseline: the existing per-symbol RF AUCs (read from model_metrics.json, or
--rf-baseline recompute to re-grade them on the same base features in-run).

LEAKAGE-SAFE PANEL WALK-FORWARD CV — the make-or-break correctness piece. All
symbols share one calendar, so a naive TimeSeriesSplit on concatenated rows would
train on symbol B's FUTURE dates to predict symbol A's PAST. Instead we split on the
GLOBAL DATE axis and PURGE a gap of ML_HORIZON trading days before each test fold
(labels look forward exactly ML_HORIZON bars, so a train row within that gap shares
future bars with the test period). cv_pooled_auc (ai_engine.py) has NO purge — fine
for one symbol (5 boundary rows) but materially leaky on a 130k-row panel
(~n_symbols × 5 rows/fold), which is why this module adds an explicit date-purge.

SAFETY (same contract as permutation_test.py / regime_ab_test.py): this module NEVER
connects to IBKR, imports ib_insync/ibkr_bridge, places/cancels/simulates orders,
trains or SAVES any production model (rf_models.joblib / lstm_checkpoint.pt /
model_metrics.json are never written — model_metrics.json is read-only here), or
changes any trading / prediction / risk / live-config logic. It only reads price
data, trains transient in-memory models, and writes a report to REPORTS_DIR.

Run:
    python -X utf8 gbm_pooled_ab_test.py --smoke           # 3 symbols, seconds
    python -X utf8 gbm_pooled_ab_test.py                   # watchlist, arms A+B
    python -X utf8 gbm_pooled_ab_test.py --all --n-shuffles 20   # full universe
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)

# Imported AFTER config is in hand. None of these capture a feature flag in a way
# that matters here: build_features()/get_feature_columns() read config at CALL
# time, _subsample_train_indices/cv_pooled_auc/build_rf operate on raw (X, y)
# arrays, and we never use ai_engine.FEATURE_COLS.
from ai_engine import _subsample_train_indices  # noqa: E402
import eval_metrics  # noqa: E402
from data_manager import (  # noqa: E402
    fetch_ohlcv,
    build_features,
    make_labels,
    get_feature_columns,
    fetch_market_benchmark,
)

# Per-symbol minimum usable rows before a symbol contributes to the pool.
_MIN_SYMBOL_ROWS = 50


# ─────────────────────────── feature flags ───────────────────────────
def _set_flags(market_rel: bool, candles: bool) -> List[str]:
    """Set the Item-3/candlestick flags and return the resulting column list.

    Regime features stay OFF (a separate, already-exhausted lever). build_features
    and get_feature_columns read these at call time, so toggling here is enough.
    """
    config.USE_REGIME_FEATURES = False
    config.USE_MARKET_RELATIVE_FEATURES = bool(market_rel)
    config.USE_CANDLESTICK_FEATURES = bool(candles)
    return get_feature_columns()


# ─────────────────────────── universe ───────────────────────────
def _symbols_from_metrics() -> List[str]:
    """Full universe = the rf keys already present in model_metrics.json (READ-only,
    no network). Falls back to the watchlist if the file is missing/unreadable."""
    try:
        d = json.loads(Path(config.MODEL_METRICS_FILE).read_text(encoding="utf-8"))
        syms = sorted(d.get("rf", {}).keys())
        return syms or list(config.WATCHLIST)
    except Exception:
        logger.debug("could not read %s; using watchlist", config.MODEL_METRICS_FILE)
        return list(config.WATCHLIST)


def _resolve_universe(args) -> List[str]:
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.all:
        syms = _symbols_from_metrics()
    else:
        syms = list(config.WATCHLIST)
    if args.limit and args.limit > 0:
        syms = syms[: args.limit]
    return syms


def _load_rf_baseline(symbols: List[str]) -> dict:
    """Per-symbol RF AUCs from model_metrics.json (READ-only). Tolerates symbols
    with no `auc` field (e.g. ABSI) — they are simply excluded from the mean."""
    try:
        d = json.loads(Path(config.MODEL_METRICS_FILE).read_text(encoding="utf-8"))
    except Exception:
        return {"per_symbol": {}, "mean_auc": None, "median_auc": None,
                "n_with_auc": 0, "n_symbols": len(symbols), "source": "unavailable"}
    rf = d.get("rf", {})
    aucs = {s: float(rf[s]["auc"]) for s in symbols
            if s in rf and rf[s].get("auc") is not None}
    vals = list(aucs.values())
    return {
        "per_symbol": {s: round(v, 4) for s, v in aucs.items()},
        "mean_auc": round(float(np.mean(vals)), 4) if vals else None,
        "median_auc": round(float(np.median(vals)), 4) if vals else None,
        "n_with_auc": len(vals),
        "n_symbols": len(symbols),
        "source": "model_metrics.json (binary-trained RF)",
    }


def _rf_baseline_recompute(base_panel: dict, horizon: int, n_splits: int) -> dict:
    """Apples-to-apples baseline: re-grade the per-symbol RF (cv_pooled_auc) on the
    SAME base-feature rows used by arm A. Slower; opt-in via --rf-baseline."""
    from ai_engine import cv_pooled_auc  # local import; uses raw (X, y) only
    X, y, symbols = base_panel["X"], base_panel["y"], base_panel["symbols"]
    aucs = {}
    for s in np.unique(symbols):
        m = symbols == s
        a = cv_pooled_auc(X[m], y[m], horizon=horizon, n_splits=n_splits)
        if a is not None:
            aucs[str(s)] = round(float(a), 4)
    vals = list(aucs.values())
    return {
        "per_symbol": aucs,
        "mean_auc": round(float(np.mean(vals)), 4) if vals else None,
        "median_auc": round(float(np.median(vals)), 4) if vals else None,
        "n_with_auc": len(vals),
        "n_symbols": int(len(np.unique(symbols))),
        "source": "recompute: per-symbol cv_pooled_auc on base features",
    }


# ─────────────────────────── panel build ───────────────────────────
def _build_panel(symbols: List[str], market_rel: bool, candles: bool, mode: str,
                 horizon: int, spy_df) -> Optional[dict]:
    """Pool (features, label, date, symbol) across all symbols into flat arrays.

    Read-only: fetch_ohlcv only. Per symbol we drop NaN-label rows and any
    non-finite feature rows, and skip symbols with < _MIN_SYMBOL_ROWS usable rows.
    """
    cols = _set_flags(market_rel, candles)
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    d_parts: List[np.ndarray] = []
    s_parts: List[np.ndarray] = []
    used, skipped = [], []
    for sym in symbols:
        try:
            df = fetch_ohlcv(sym)
            feat = build_features(df, market_df=spy_df if market_rel else None)
            labels = make_labels(feat, horizon=horizon, mode=mode)
        except Exception:
            logger.debug("panel build skipped %s", sym, exc_info=True)
            skipped.append(sym)
            continue
        valid = labels.dropna().index
        if len(valid) < _MIN_SYMBOL_ROWS:
            skipped.append(sym)
            continue
        Xs = feat.loc[valid, cols].to_numpy(dtype=float)
        ys = labels.loc[valid].to_numpy().astype(int)
        ds = valid.to_numpy()  # datetime64[ns]
        finite = np.isfinite(Xs).all(axis=1)
        Xs, ys, ds = Xs[finite], ys[finite], ds[finite]
        if len(Xs) < _MIN_SYMBOL_ROWS:
            skipped.append(sym)
            continue
        X_parts.append(Xs)
        y_parts.append(ys)
        d_parts.append(ds)
        s_parts.append(np.full(len(Xs), sym, dtype=object))
        used.append(sym)

    if not X_parts:
        return None
    return {
        "X": np.vstack(X_parts),
        "y": np.concatenate(y_parts),
        "dates": np.concatenate(d_parts),
        "symbols": np.concatenate(s_parts),
        "feature_cols": list(cols),
        "n_symbols_used": len(used),
        "n_symbols_skipped": len(skipped),
        "symbols_used": used,
        "symbols_skipped": skipped,
    }


# ─────────────────────────── estimator ───────────────────────────
def _gbm_factory() -> "object":
    """Canonical pooled-GBM factory (single source of truth). sklearn's
    HistGradientBoostingClassifier — already a dependency, no new package. Defensive
    levers (small learning_rate, large min_samples_leaf, L2, early stopping) guard
    against fitting noise on the big pooled set; class_weight mirrors build_rf."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=config.RANDOM_STATE,
    )


# ─────────────── leakage-safe panel walk-forward CV ───────────────
def _date_fold_bounds(uniq: np.ndarray, n_splits: int) -> List[Tuple]:
    """TimeSeriesSplit-style contiguous test blocks on the unique-date axis.

    Returns [(test_lo, test_hi), ...]; test_hi is None for the final block (runs to
    the end). Blocks are contiguous and non-overlapping: fold k's test_hi equals
    fold k+1's test_lo. Train (everything before the purged cutoff) is an expanding
    window, exactly like sklearn's TimeSeriesSplit.
    """
    n = len(uniq)
    n_splits = max(2, int(n_splits))
    if n < n_splits + 1:
        return []
    test_size = n // (n_splits + 1)
    if test_size < 1:
        return []
    bounds: List[Tuple] = []
    for k in range(1, n_splits + 1):
        lo = k * test_size
        if lo >= n:
            break
        hi = (k + 1) * test_size
        bounds.append((uniq[lo], uniq[hi] if (k < n_splits and hi < n) else None))
    return bounds


def _iter_folds(dates: np.ndarray, n_splits: int, purge: int):
    """Yield (train_mask, test_mask, test_lo, purge_date) per walk-forward fold.

    Split is on the global date axis; train rows are those strictly before
    ``purge_date = uniq[idx_of(test_lo) - purge]`` so no train label window (which
    looks forward ``purge`` bars) overlaps the test period.
    """
    dates = np.asarray(dates)
    uniq = np.unique(dates)  # sorted ascending
    for test_lo, test_hi in _date_fold_bounds(uniq, n_splits):
        if test_hi is None:
            test_mask = dates >= test_lo
        else:
            test_mask = (dates >= test_lo) & (dates < test_hi)
        idx_lo = int(np.searchsorted(uniq, test_lo))
        purge_date = uniq[max(0, idx_lo - int(purge))]
        train_mask = dates < purge_date
        yield train_mask, test_mask, test_lo, purge_date


def _panel_pooled_preds(X, y, dates, symbols, horizon, n_splits, purge,
                        factory: Callable):
    """Pooled out-of-sample (y_true, y_score, symbol) across the date-axis folds,
    or None if no fold produced predictions."""
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    dates = np.asarray(dates)
    symbols = np.asarray(symbols)
    pooled_y, pooled_p, pooled_s = [], [], []
    for train_mask, test_mask, _lo, _pd in _iter_folds(dates, n_splits, purge):
        if not train_mask.any() or not test_mask.any():
            continue
        ytr_full = y[train_mask]
        if len(np.unique(ytr_full)) < 2:
            continue
        # Decorrelate overlapping forward labels: stride TRAIN rows in time order.
        order = np.argsort(dates[train_mask], kind="stable")
        keep = order[_subsample_train_indices(len(order), horizon)]
        Xtr = X[train_mask][keep]
        ytr = ytr_full[keep]
        if len(np.unique(ytr)) < 2:
            continue
        clf = factory()
        clf.fit(Xtr, ytr)
        pp = eval_metrics.proba_pos(clf, X[test_mask])
        if pp is None:
            continue
        pooled_y.append(y[test_mask])
        pooled_p.append(pp)
        pooled_s.append(symbols[test_mask])
    if not pooled_y:
        return None
    return (np.concatenate(pooled_y), np.concatenate(pooled_p),
            np.concatenate(pooled_s))


def _panel_auc(X, y, dates, symbols, horizon, n_splits, purge,
               factory: Callable) -> Optional[float]:
    """Pooled panel CV-AUC only (used by the null loop)."""
    pr = _panel_pooled_preds(X, y, dates, symbols, horizon, n_splits, purge, factory)
    if pr is None:
        return None
    yt, pp, _ = pr
    return eval_metrics.safe_auc(yt, pp)


def _panel_metrics(X, y, dates, symbols, horizon, n_splits, purge,
                   factory: Callable) -> Optional[dict]:
    """Full pooled metrics: AUC, per-symbol AUC, precision@BUY."""
    pr = _panel_pooled_preds(X, y, dates, symbols, horizon, n_splits, purge, factory)
    if pr is None:
        return None
    yt, pp, ss = pr
    auc = eval_metrics.safe_auc(yt, pp)
    if auc is None:
        return None
    per_sym = {}
    for s in np.unique(ss):
        a = eval_metrics.safe_auc(yt[ss == s], pp[ss == s])
        if a is not None:
            per_sym[str(s)] = round(float(a), 4)
    prec, n_at = eval_metrics.precision_at_threshold(yt, pp, config.BUY_THRESHOLD)
    return {
        "pooled_auc": round(float(auc), 4),
        "n_oos": int(len(yt)),
        "per_symbol_auc": per_sym,
        "precision_at_buy": eval_metrics.round4(prec),
        "n_at_buy": int(n_at),
    }


def _shuffle_y(y: np.ndarray, symbols: np.ndarray, dates: np.ndarray,
               scope: str, rng) -> np.ndarray:
    """Null label permutation. global: break all structure. within_symbol: preserve
    per-symbol base rates (isolate feature->label skill from base-rate exploitation).
    within_date: preserve each day's cross-section."""
    if scope == "within_symbol":
        groups = symbols
    elif scope == "within_date":
        groups = dates
    else:
        return rng.permutation(y)
    out = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = rng.permutation(y[idx])
    return out


def _null_pvalue(panel: dict, real_auc: float, n_shuffles: int, horizon: int,
                 n_splits: int, purge: int, factory: Callable, seed: int,
                 scope: str) -> dict:
    """Shuffled-label null distribution + add-one-smoothed p-value (P(null>=real))."""
    X, y = panel["X"], panel["y"]
    dates, symbols = panel["dates"], panel["symbols"]
    rng = np.random.default_rng(seed)
    null: List[float] = []
    for _ in range(n_shuffles):
        ys = _shuffle_y(y, symbols, dates, scope, rng)
        a = _panel_auc(X, ys, dates, symbols, horizon, n_splits, purge, factory)
        if a is not None:
            null.append(a)
    if not null:
        return {"null_mean": None, "null_p95": None, "n_null": 0, "p_value": None}
    arr = np.asarray(null, dtype=float)
    p = (1.0 + float((arr >= real_auc).sum())) / (1.0 + len(arr))
    return {
        "null_mean": round(float(arr.mean()), 4),
        "null_p95": round(float(np.percentile(arr, 95)), 4),
        "n_null": int(len(arr)),
        "p_value": round(float(p), 4),
    }


def _dist(vals: List[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None,
                "min": None, "max": None, "n_above_0_55": 0}
    a = np.asarray(vals, dtype=float)
    return {
        "n": int(len(a)),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "n_above_0_55": int((a > 0.55).sum()),
    }


def run_arm(name: str, mode: str, panel: dict, n_shuffles: int, horizon: int,
            n_splits: int, purge: int, seed: int, null_scope: str) -> Optional[dict]:
    """Real pooled CV + shuffled-label null for one arm/mode."""
    factory = _gbm_factory
    m = _panel_metrics(panel["X"], panel["y"], panel["dates"], panel["symbols"],
                       horizon, n_splits, purge, factory)
    if m is None:
        return None
    out = {
        "arm": name,
        "label_mode": mode,
        "n_symbols": panel["n_symbols_used"],
        "n_symbols_skipped": panel["n_symbols_skipped"],
        "n_samples": int(len(panel["y"])),
        "n_features": int(panel["X"].shape[1]),
        "feature_cols": panel["feature_cols"],
        "positive_rate": round(float(np.mean(panel["y"])), 4),
        "null_scope": null_scope,
    }
    out.update(m)
    out["per_symbol_auc_dist"] = _dist(list(m["per_symbol_auc"].values()))
    out.update(_null_pvalue(panel, m["pooled_auc"], n_shuffles, horizon, n_splits,
                            purge, factory, seed, null_scope))
    return out


def _verdict(arm: dict, rf_mean: Optional[float]) -> str:
    p = arm.get("p_value")
    auc = arm.get("pooled_auc")
    if p is None or auc is None:
        return "INCONCLUSIVE: pooled AUC or null p-value unavailable."
    beats_rf = (rf_mean is None) or (auc >= rf_mean + 0.02)
    rf_txt = f"RF baseline {rf_mean}" if rf_mean is not None else "RF baseline n/a"
    if p <= 0.05 and auc >= 0.55 and beats_rf:
        return (f"EDGE: pooled AUC {auc} beats the shuffled null (p={p}) and "
                f"{rf_txt}. Worth a guarded follow-up (still offline).")
    if p <= 0.05 and (auc >= 0.52 or beats_rf):
        return (f"WEAK: pooled AUC {auc} clears the null (p={p}) but the lift over "
                f"{rf_txt} is small. Not actionable yet.")
    return (f"NO EDGE: pooled AUC {auc} (p={p}) vs {rf_txt}. Pooling does not "
            f"recover edge from this feature/label set.")


# ─────────────────────────── report ───────────────────────────
def _write_report(summaries: List[dict], rf_baseline: dict, meta: dict,
                  suffix: str) -> "config.Path":
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"_{suffix}" if suffix else ""
    md_path = config.REPORTS_DIR / f"gbm_pooled_ab_report{stamp}.md"
    json_path = config.REPORTS_DIR / f"gbm_pooled_ab_report{stamp}.json"

    rf_mean = rf_baseline.get("mean_auc")
    L: List[str] = []
    L.append("# Pooled cross-symbol GBM A/B (offline edge test)")
    L.append("")
    L.append(f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_")
    L.append("")
    L.append("Read-only / offline ML evaluation. No IBKR, no orders, no model saved "
             "(rf_models.joblib / lstm_checkpoint.pt / model_metrics.json untouched).")
    L.append("Leakage-safe panel walk-forward CV: split on the global DATE axis with "
             f"a purge gap of {meta['purge']} bars (ML_HORIZON); pooled GBM = sklearn "
             "HistGradientBoostingClassifier.")
    L.append("")
    L.append("## Setup")
    L.append("")
    L.append(f"- Universe : {meta['n_symbols']} symbols requested "
             f"({meta['universe_src']})")
    L.append(f"- Label modes : {meta['modes']} | arms : {meta['arms']}")
    L.append(f"- CV splits : {meta['n_splits']} | purge : {meta['purge']} | "
             f"null draws : {meta['n_shuffles']} (scope {meta['null_scope']})")
    L.append(f"- Candlestick features : {'on' if meta['candles'] else 'off'} | "
             f"seed : {meta['seed']}")
    L.append("")
    L.append(f"### Per-symbol RF baseline ({rf_baseline.get('source')})")
    L.append("")
    L.append(f"- Mean AUC **{rf_mean}** | median {rf_baseline.get('median_auc')} | "
             f"symbols with AUC: {rf_baseline.get('n_with_auc')}/"
             f"{rf_baseline.get('n_symbols')}")
    L.append("")
    for s in summaries:
        d = s["per_symbol_auc_dist"]
        L.append(f"## {s['label_mode']} — arm {s['arm']} "
                 f"({s['n_features']} features, {s['n_symbols']} symbols, "
                 f"{s['n_samples']} rows)")
        L.append("")
        L.append(f"- **Pooled CV-AUC : {s['pooled_auc']}** "
                 f"(null mean {s['null_mean']}, p95 {s['null_p95']}, "
                 f"p-value **{s['p_value']}**, {s['n_null']} draws)")
        L.append(f"- Precision@BUY({config.BUY_THRESHOLD}) : {s['precision_at_buy']} "
                 f"on {s['n_at_buy']} signals | positive rate {s['positive_rate']}")
        L.append(f"- Per-symbol AUC : mean {d['mean']} median {d['median']} "
                 f"[{d['min']}..{d['max']}], {d['n_above_0_55']}/{d['n']} > 0.55")
        L.append("")
        L.append(f"**Verdict — {_verdict(s, rf_mean)}**")
        L.append("")

    md_path.write_text("\n".join(L), encoding="utf-8")
    json_path.write_text(json.dumps(
        {"meta": meta, "rf_baseline": rf_baseline, "arms": summaries}, indent=2),
        encoding="utf-8")
    return md_path


# ─────────────────────────── main ───────────────────────────
def main(argv: Optional[List[str]] = None) -> int:
    from logging_setup import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Pooled cross-symbol GBM A/B (offline; no IBKR, no orders, no model saved)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: config.WATCHLIST)")
    parser.add_argument("--all", action="store_true",
                        help="Use the full universe (rf keys in model_metrics.json)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the resolved universe to the first N symbols")
    parser.add_argument("--modes", type=str, default="binary",
                        help="Comma-separated label modes (binary,triple_barrier)")
    parser.add_argument("--arms", type=str, default="A,B",
                        help="A=base, B=base+SPY-relative")
    parser.add_argument("--rf-baseline", type=str, default="metrics",
                        choices=["metrics", "recompute", "both", "none"],
                        help="Per-symbol RF baseline source")
    parser.add_argument("--n-shuffles", type=int, default=None,
                        help="Null draws per arm (default: config.PERMUTATION_N_SHUFFLES)")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--purge", type=int, default=None,
                        help="Purge gap in bars (default: config.ML_HORIZON)")
    parser.add_argument("--null-scope", type=str, default="global",
                        choices=["global", "within_symbol", "within_date"])
    parser.add_argument("--candles", type=str, default="off",
                        choices=["on", "off"], help="Include candlestick features")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast sanity: 3 symbols, 3 shuffles, 2 splits, binary")
    args = parser.parse_args(argv)

    symbols = _resolve_universe(args)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    n_shuffles = int(args.n_shuffles or config.PERMUTATION_N_SHUFFLES)
    n_splits = max(2, int(args.n_splits))
    purge = int(args.purge if args.purge is not None else config.ML_HORIZON)
    candles = args.candles == "on"
    null_scope = args.null_scope
    suffix = args.suffix
    universe_src = ("explicit" if args.symbols else
                    "model_metrics.json rf keys" if args.all else "watchlist")

    if args.smoke:
        symbols = symbols[:3]
        modes = ["binary"]
        n_shuffles = 3
        n_splits = 2
        suffix = suffix or "smoke"

    horizon = int(config.ML_HORIZON)
    seed = int(config.RANDOM_STATE)

    print("=== Pooled cross-symbol GBM A/B (offline; no IBKR, no orders, no model saved) ===")
    print(f"Symbols: {len(symbols)} ({universe_src}) | modes: {modes} | arms: {arms}")
    print(f"CV splits: {n_splits} | purge: {purge} | null draws: {n_shuffles} "
          f"(scope {null_scope}) | candles: {candles} | seed: {seed}")

    spy_df = None
    if "B" in arms:
        try:
            spy_df = fetch_market_benchmark()
        except Exception:
            logger.warning("could not fetch %s benchmark; arm B will use neutral "
                           "market features", config.MARKET_BENCHMARK_SYMBOL)

    rf_metrics = _load_rf_baseline(symbols)

    summaries: List[dict] = []
    rf_recompute: Optional[dict] = None
    for mode in modes:
        base_panel = None
        for arm in arms:
            market_rel = (arm == "B")
            print(f"\n--- {mode} | arm {arm} ({'base+market' if market_rel else 'base'}) "
                  f"--- building panel...", flush=True)
            panel = _build_panel(symbols, market_rel, candles, mode, horizon, spy_df)
            if panel is None:
                print(f"  arm {arm} skipped (no usable rows)", flush=True)
                continue
            if arm == "A":
                base_panel = panel
            print(f"  panel: {panel['n_symbols_used']} symbols, "
                  f"{len(panel['y'])} rows, {panel['X'].shape[1]} features "
                  f"({panel['n_symbols_skipped']} skipped); scoring...", flush=True)
            res = run_arm(arm, mode, panel, n_shuffles, horizon, n_splits, purge,
                          seed, null_scope)
            if res is None:
                print(f"  arm {arm} produced no AUC (single-class / too short)", flush=True)
                continue
            summaries.append(res)
            print(f"  arm {arm}: pooled AUC={res['pooled_auc']} "
                  f"p={res['p_value']} (null {res['null_mean']})", flush=True)

        if args.rf_baseline in ("recompute", "both") and base_panel is not None and rf_recompute is None:
            print("  recomputing per-symbol RF baseline (cv_pooled_auc)...", flush=True)
            rf_recompute = _rf_baseline_recompute(base_panel, horizon, n_splits)

    rf_baseline = (rf_recompute if (args.rf_baseline in ("recompute", "both")
                                    and rf_recompute is not None) else rf_metrics)
    if args.rf_baseline == "none":
        rf_baseline = {"per_symbol": {}, "mean_auc": None, "median_auc": None,
                       "n_with_auc": 0, "n_symbols": len(symbols), "source": "disabled"}

    meta = {
        "n_symbols": len(symbols), "universe_src": universe_src, "modes": modes,
        "arms": arms, "n_splits": n_splits, "purge": purge, "n_shuffles": n_shuffles,
        "null_scope": null_scope, "candles": candles, "seed": seed, "horizon": horizon,
    }
    report = _write_report(summaries, rf_baseline, meta, suffix)

    print("\n=== SUMMARY ===")
    print(f"Per-symbol RF baseline mean AUC: {rf_baseline.get('mean_auc')} "
          f"({rf_baseline.get('source')})")
    for s in summaries:
        print(f"[{s['label_mode']}/{s['arm']}] pooled AUC {s['pooled_auc']} "
              f"(p={s['p_value']}) — {_verdict(s, rf_baseline.get('mean_auc'))}")
    print(f"\nReport written to {report}")
    print("No IBKR connection was made, no orders placed, no model saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
