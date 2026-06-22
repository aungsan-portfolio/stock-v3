"""
gbm_tradeability.py — read-only OFFLINE tradeability evaluation of Arm B only.

Context: gbm_pooled_ab_test.py showed the pooled cross-symbol GBM with Arm B
features (base 14 + 4 SPY relative-strength = 18 cols, binary label) has a WEAK
but real ranking signal — pooled CV-AUC ~0.516 (p~0.01-0.05), precision@BUY(0.65)
~0.547 vs a ~0.496 base positive rate. Statistical significance is necessary but
NOT sufficient: a tiny AUC lift can still lose money once you only trade the
high-probability tail and pay costs. This module answers the next question and
ONLY that question:

    Do Arm B's leakage-safe out-of-sample probabilities produce positive trading
    EXPECTANCY after thresholding / top-k selection and basic transaction costs?

It REUSES Arm B's exact leakage-safe panel walk-forward CV (gbm_pooled_ab_test.
_iter_folds + _gbm_factory + _subsample_train_indices + eval_metrics.proba_pos,
seed config.RANDOM_STATE) so the predictions graded here are byte-for-byte the
same predictions the A/B test scored — it just additionally carries the realized
forward return and entry/exit reference price for every out-of-sample row, then
runs a pure, deterministic trade-selection + cost sweep over them.

Tradeable return = the H-bar buy-and-hold endpoint return Close[t+H]/Close[t]-1
(H = config.ML_HORIZON). In binary label mode this is EXACTLY the quantity that
defines the label (label==1  <=>  fwd_ret > config.MIN_PROFIT_MARGIN), so the
evaluation is self-consistent and leakage-safe: the same horizon that the label
looks forward is the holding period the return measures, and the CV purge already
removes train rows whose label window overlaps the test period.

SAFETY (identical contract to gbm_pooled_ab_test / permutation_test): this module
NEVER connects to IBKR, imports ib_insync/ibkr_bridge, places/cancels/simulates
orders, trains or SAVES any production model (rf_models.joblib /
lstm_checkpoint.pt / model_metrics.json are never written), or changes any
trading / prediction / risk / live-config logic. It only reads price data, trains
transient in-memory GBMs for the walk-forward folds, and writes the two report
files under REPORTS_DIR.

Run:
    python -X utf8 gbm_tradeability.py --smoke          # 3 symbols, seconds
    python -X utf8 gbm_tradeability.py                  # watchlist
    python -X utf8 gbm_tradeability.py --all            # full universe (108 syms)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# Reuse Arm B's exact CV / estimator / flag machinery (single source of truth).
# Importing gbm_pooled_ab_test pulls in NONE of the IBKR/order path (it is itself
# a read-only offline module — enforced by its own test suite).
import gbm_pooled_ab_test as gbm  # noqa: E402
import eval_metrics  # noqa: E402
from ai_engine import _subsample_train_indices  # noqa: E402
from data_manager import (  # noqa: E402
    fetch_ohlcv,
    build_features,
    make_labels,
    fetch_market_benchmark,
)

# Defaults for the trade-selection sweep (overridable on the CLI).
DEFAULT_THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75)
DEFAULT_TOPK = (1, 3, 5, 10)
DEFAULT_COSTS_BPS = (0.0, 5.0, 10.0)
TRADING_DAYS_PER_YEAR = 252


# ──────────────────── Arm B panel WITH forward returns ────────────────────
def build_tradeable_panel(symbols: List[str], candles: bool, mode: str,
                          horizon: int, spy_df) -> Optional[dict]:
    """Pool (features, label, date, symbol, fwd_ret, entry, exit) across symbols.

    This mirrors gbm_pooled_ab_test._build_panel EXACTLY for the (X, y, dates,
    symbols) it produces — same Arm B flags (market-relative ON), same per-symbol
    NaN-label drop, same finite-feature mask, same _MIN_SYMBOL_ROWS floor, same
    row order — so the CV folds and predictions are identical to Arm B's. It only
    ADDS three aligned columns:

        entry  = Close[t]                 (build_features() output close)
        exit   = Close[t + horizon]
        fwd_ret = exit / entry - 1.0      (H-bar buy-and-hold endpoint return)

    For valid (non-NaN-label) rows the forward close always exists in BOTH label
    modes (binary labels are NaN exactly when Close[t+H] is unknown; triple-
    barrier labels are defined only for t < n-horizon), so entry/exit/fwd_ret are
    finite on every kept row. A defensive finite-fwd guard is applied anyway and
    any dropped row is logged.
    """
    cols = gbm._set_flags(market_rel=True, candles=candles)  # Arm B := market-rel ON
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    d_parts: List[np.ndarray] = []
    s_parts: List[np.ndarray] = []
    e_parts: List[np.ndarray] = []
    x_parts: List[np.ndarray] = []
    f_parts: List[np.ndarray] = []
    used, skipped = [], []
    for sym in symbols:
        try:
            df = fetch_ohlcv(sym)
            feat = build_features(df, market_df=spy_df)
            labels = make_labels(feat, horizon=horizon, mode=mode)
        except Exception:
            logger.debug("panel build skipped %s", sym, exc_info=True)
            skipped.append(sym)
            continue
        valid = labels.dropna().index
        if len(valid) < gbm._MIN_SYMBOL_ROWS:
            skipped.append(sym)
            continue
        Xs = feat.loc[valid, cols].to_numpy(dtype=float)
        ys = labels.loc[valid].to_numpy().astype(int)
        ds = valid.to_numpy()  # datetime64[ns]
        close = feat["Close"]
        entry = close.loc[valid].to_numpy(dtype=float)
        exitp = close.shift(-horizon).loc[valid].to_numpy(dtype=float)
        # Identical finite mask to _build_panel (features only) — keeps row sets
        # aligned with Arm B. fwd is finite on these rows; guard below is belt-and-braces.
        finite = np.isfinite(Xs).all(axis=1)
        Xs, ys, ds = Xs[finite], ys[finite], ds[finite]
        entry, exitp = entry[finite], exitp[finite]
        if len(Xs) < gbm._MIN_SYMBOL_ROWS:
            skipped.append(sym)
            continue
        fwd = exitp / entry - 1.0
        ok = np.isfinite(fwd) & np.isfinite(entry) & np.isfinite(exitp)
        if not ok.all():
            logger.warning("%s: dropped %d row(s) with non-finite forward return",
                           sym, int((~ok).sum()))
            Xs, ys, ds = Xs[ok], ys[ok], ds[ok]
            entry, exitp, fwd = entry[ok], exitp[ok], fwd[ok]
        if len(Xs) < gbm._MIN_SYMBOL_ROWS:
            skipped.append(sym)
            continue
        X_parts.append(Xs)
        y_parts.append(ys)
        d_parts.append(ds)
        s_parts.append(np.full(len(Xs), sym, dtype=object))
        e_parts.append(entry)
        x_parts.append(exitp)
        f_parts.append(fwd)
        used.append(sym)

    if not X_parts:
        return None
    return {
        "X": np.vstack(X_parts),
        "y": np.concatenate(y_parts),
        "dates": np.concatenate(d_parts),
        "symbols": np.concatenate(s_parts),
        "entry": np.concatenate(e_parts),
        "exit": np.concatenate(x_parts),
        "fwd_ret": np.concatenate(f_parts),
        "feature_cols": list(cols),
        "n_symbols_used": len(used),
        "n_symbols_skipped": len(skipped),
        "symbols_used": used,
        "symbols_skipped": skipped,
    }


# ──────────────── OOS predictions over Arm B's walk-forward CV ────────────────
def panel_oos_records(panel: dict, horizon: int, n_splits: int, purge: int,
                      factory: Callable) -> Optional[dict]:
    """Out-of-sample (date, symbol, prob, label, fwd_ret, entry, exit, fold) for
    every test row across the leakage-safe date-axis folds, or None if no fold
    produced predictions.

    The fit/predict path is IDENTICAL to gbm_pooled_ab_test._panel_pooled_preds —
    same fold iterator, same horizon train-row striding, same GBM factory, same
    class-1 probability extraction — so the (label, prob, symbol) it yields equal
    Arm B's pooled predictions; it merely carries the date/return/price columns
    through alongside.
    """
    X = np.asarray(panel["X"])
    y = np.asarray(panel["y"]).astype(int)
    dates = np.asarray(panel["dates"])
    symbols = np.asarray(panel["symbols"])
    fwd = np.asarray(panel["fwd_ret"], dtype=float)
    entry = np.asarray(panel["entry"], dtype=float)
    exitp = np.asarray(panel["exit"], dtype=float)

    o_y, o_p, o_s, o_d, o_f, o_e, o_x, o_fold = [], [], [], [], [], [], [], []
    fold_id = 0
    for train_mask, test_mask, _lo, _pd in gbm._iter_folds(dates, n_splits, purge):
        if not train_mask.any() or not test_mask.any():
            continue
        ytr_full = y[train_mask]
        if len(np.unique(ytr_full)) < 2:
            continue
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
        o_y.append(y[test_mask])
        o_p.append(pp)
        o_s.append(symbols[test_mask])
        o_d.append(dates[test_mask])
        o_f.append(fwd[test_mask])
        o_e.append(entry[test_mask])
        o_x.append(exitp[test_mask])
        o_fold.append(np.full(int(test_mask.sum()), fold_id, dtype=int))
        fold_id += 1
    if not o_y:
        return None
    return {
        "label": np.concatenate(o_y).astype(int),
        "prob": np.concatenate(o_p).astype(float),
        "symbol": np.concatenate(o_s),
        "date": np.concatenate(o_d),
        "fwd_ret": np.concatenate(o_f).astype(float),
        "entry": np.concatenate(o_e).astype(float),
        "exit": np.concatenate(o_x).astype(float),
        "fold": np.concatenate(o_fold).astype(int),
        "n_folds": fold_id,
    }


# ─────────────────────────── trade selection ───────────────────────────
def select_threshold(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of rows whose probability clears ``threshold`` (>=)."""
    return np.asarray(prob, dtype=float) >= float(threshold)


def select_topk_per_day(prob: np.ndarray, dates: np.ndarray, symbols: np.ndarray,
                        k: int) -> np.ndarray:
    """Boolean mask of the k highest-probability rows on EACH unique date.

    Deterministic tie-break: sort by probability descending, then symbol ascending,
    so the same panel always yields the same selection (no RNG, no dict-order
    dependence). Days with fewer than k rows contribute all of their rows. This is
    a same-day cross-sectional ranking — it uses only that day's probabilities, so
    it introduces no look-ahead.
    """
    prob = np.asarray(prob, dtype=float)
    dates = np.asarray(dates)
    symbols = np.asarray(symbols, dtype=object)
    k = int(k)
    mask = np.zeros(len(prob), dtype=bool)
    if k <= 0:
        return mask
    for d in np.unique(dates):
        idx = np.where(dates == d)[0]
        if len(idx) <= k:
            mask[idx] = True
            continue
        # rank by (-prob, symbol) for a deterministic, RNG-free order
        sym_d = symbols[idx].astype(str)
        order = sorted(range(len(idx)), key=lambda j: (-prob[idx[j]], sym_d[j]))
        mask[idx[[order[j] for j in range(k)]]] = True
    return mask


# ─────────────────────────── metric primitives ───────────────────────────
def _profit_factor(net: np.ndarray) -> Optional[float]:
    """sum(gains) / |sum(losses)|. None when there are no losses (undefined /
    infinite) — reported alongside n_losses so the caller can interpret it."""
    net = np.asarray(net, dtype=float)
    gains = net[net > 0].sum()
    losses = -net[net < 0].sum()
    if losses <= 0:
        return None
    return float(gains / losses)


def _max_drawdown(net_ordered: np.ndarray) -> float:
    """Max drawdown (>=0, in return units) of the additive equity curve formed by
    cumulatively summing per-trade net returns in the given (chronological) order.

    A leading 0 is prepended so a first losing trade counts as a drawdown below
    starting capital. This assumes 1 unit deployed per trade and IGNORES position
    overlap / capital limits — a relative DD proxy, not a live-capital simulation.
    """
    net_ordered = np.asarray(net_ordered, dtype=float)
    if len(net_ordered) == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(net_ordered)])
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def strategy_metrics(gross_ret: np.ndarray, label: np.ndarray, dates: np.ndarray,
                     symbols: np.ndarray, base_pos_rate: float, cost_bps: float,
                     horizon: int) -> dict:
    """Full tradeability metric block for one selected set of trades at one cost.

    ``cost_bps`` is deducted from every trade's gross return as a TOTAL round-trip
    cost (entry+exit+slippage combined): net = gross - cost_bps/1e4. Trades are
    ordered by (date, symbol) for the drawdown equity curve. ``base_pos_rate`` is
    the OOS positive-label rate over ALL predictions (so precision lift is vs the
    unconditional base rate). Returns JSON-safe scalars (None where undefined).
    """
    gross = np.asarray(gross_ret, dtype=float)
    label = np.asarray(label, dtype=int)
    n = int(len(gross))
    if n == 0:
        return {
            "n_trades": 0, "hit_rate": None, "avg_return": None,
            "median_return": None, "profit_factor": None, "n_losses": 0,
            "expectancy": None, "max_drawdown": None, "precision": None,
            "base_pos_rate": eval_metrics.round4(base_pos_rate), "lift": None,
            "avg_win": None, "avg_loss": None, "total_net_return": None,
            "return_std": None, "sharpe_per_trade": None, "ann_return_approx": None,
        }
    cost = float(cost_bps) / 1e4
    net = gross - cost

    # chronological order for the equity / drawdown curve
    order = np.lexsort((np.asarray(symbols, dtype=str), np.asarray(dates)))
    net_ordered = net[order]

    wins = net[net > 0]
    losses = net[net < 0]
    hit_rate = float((net > 0).mean())
    avg_ret = float(net.mean())
    precision = float((label == 1).mean())
    base = float(base_pos_rate) if base_pos_rate else None
    lift = (precision / base) if base else None
    std = float(net.std(ddof=1)) if n > 1 else 0.0
    return {
        "n_trades": n,
        "hit_rate": eval_metrics.round4(hit_rate),
        "avg_return": eval_metrics.round4(avg_ret),
        "median_return": eval_metrics.round4(float(np.median(net))),
        "profit_factor": eval_metrics.round4(_profit_factor(net)),
        "n_losses": int((net < 0).sum()),
        "expectancy": eval_metrics.round4(avg_ret),  # per-trade expectancy = mean net
        "max_drawdown": eval_metrics.round4(_max_drawdown(net_ordered)),
        "precision": eval_metrics.round4(precision),
        "base_pos_rate": eval_metrics.round4(base),
        "lift": eval_metrics.round4(lift),
        "avg_win": eval_metrics.round4(float(wins.mean())) if len(wins) else None,
        "avg_loss": eval_metrics.round4(float(losses.mean())) if len(losses) else None,
        "total_net_return": eval_metrics.round4(float(net.sum())),
        "return_std": eval_metrics.round4(std),
        "sharpe_per_trade": eval_metrics.round4(avg_ret / std) if std > 0 else None,
        # Rough non-compounded annualization: a trade ties up capital for ``horizon``
        # bars, so ~252/horizon independent trades fit a year. Ignores overlap /
        # capital constraints — directional only.
        "ann_return_approx": eval_metrics.round4(
            avg_ret * (TRADING_DAYS_PER_YEAR / max(1, int(horizon)))),
    }


# ─────────────────────────── sweep driver ───────────────────────────
def _oos_summary(oos: dict, horizon: int) -> dict:
    """Distribution summary of the raw OOS predictions (the pre-selection pool)."""
    prob = oos["prob"]
    fwd = oos["fwd_ret"]
    label = oos["label"]
    auc = eval_metrics.safe_auc(label, prob)
    return {
        "n_oos": int(len(prob)),
        "n_folds": int(oos["n_folds"]),
        "n_symbols": int(len(np.unique(oos["symbol"]))),
        "horizon_bars": int(horizon),
        "base_pos_rate": eval_metrics.round4(float(label.mean())),
        "pooled_auc": eval_metrics.round4(auc),
        "prob_min": eval_metrics.round4(float(prob.min())),
        "prob_mean": eval_metrics.round4(float(prob.mean())),
        "prob_max": eval_metrics.round4(float(prob.max())),
        "fwd_ret_mean": eval_metrics.round4(float(fwd.mean())),
        "fwd_ret_median": eval_metrics.round4(float(np.median(fwd))),
        "date_min": str(np.min(oos["date"]))[:10],
        "date_max": str(np.max(oos["date"]))[:10],
    }


def run_sweep(oos: dict, thresholds, topk, costs_bps, horizon: int) -> dict:
    """Run the threshold and top-k selections, plus a trade-EVERYTHING baseline,
    each across all cost levels.

    The baseline is the return-space analogue of the label base rate: it is the
    unconditional expectancy of buying every OOS signal. A threshold / top-k rule
    only adds tradeable value if it BEATS this baseline at the same cost — in a
    rising market the absolute expectancy can be positive from drift alone even
    when the model's ranking is uninformative (AUC <= 0.5).
    """
    prob = oos["prob"]
    fwd = oos["fwd_ret"]
    label = oos["label"]
    dates = oos["date"]
    symbols = oos["symbol"]
    base = float(label.mean())

    def _block(mask, name, param):
        idx = np.where(mask)[0]
        rows = []
        for c in costs_bps:
            m = strategy_metrics(fwd[idx], label[idx], dates[idx], symbols[idx],
                                 base, c, horizon)
            m["selection"] = name
            m["param"] = param
            m["cost_bps"] = float(c)
            m["outlier_driven"] = _is_outlier_driven(m)
            rows.append(m)
        return rows

    baseline_rows = _block(np.ones(len(prob), dtype=bool), "all", "all")
    threshold_rows = []
    for t in thresholds:
        threshold_rows.extend(_block(select_threshold(prob, t), "threshold", float(t)))
    topk_rows = []
    for k in topk:
        topk_rows.extend(
            _block(select_topk_per_day(prob, dates, symbols, k), "top_k", int(k)))
    return {"baseline": baseline_rows, "threshold": threshold_rows,
            "top_k": topk_rows}


# ─────────────────────────── report ───────────────────────────
def _fmt(v) -> str:
    return "n/a" if v is None else (f"{v}")


def _metric_table(rows: List[dict], param_label: str) -> List[str]:
    """Markdown table of one selection block (threshold or top-k) across costs."""
    L = [
        f"| {param_label} | cost(bps) | trades | hit | avg ret | med ret | "
        "PF | expect | maxDD | prec | lift | ann~ | flag |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        flag = "outlier" if r.get("outlier_driven") else ""
        L.append(
            f"| {_fmt(r['param'])} | {_fmt(r['cost_bps'])} | {_fmt(r['n_trades'])} | "
            f"{_fmt(r['hit_rate'])} | {_fmt(r['avg_return'])} | {_fmt(r['median_return'])} | "
            f"{_fmt(r['profit_factor'])} | {_fmt(r['expectancy'])} | {_fmt(r['max_drawdown'])} | "
            f"{_fmt(r['precision'])} | {_fmt(r['lift'])} | {_fmt(r['ann_return_approx'])} | {flag} |")
    return L


def _is_outlier_driven(row: dict) -> bool:
    """A selection's mean expectancy is outlier-driven when it dwarfs the median
    net return — a few fat-tailed winners, not a centered edge. Mean expectancy is
    deceptively large for concentrated rules (e.g. top-1/day picks the single
    highest-prob, often most-volatile, name). Threshold = |mean| > 4x |median|."""
    e, m = row.get("expectancy"), row.get("median_return")
    if e is None or m is None:
        return False
    return abs(e) > 4.0 * abs(m)


def _is_robust_edge(row: dict, base: dict) -> bool:
    """An edge is ROBUST only if, vs the trade-everything baseline at the SAME cost,
    BOTH the central tendency and the label precision improve — not just the
    outlier-sensitive mean:
      * positive net expectancy AND mean > baseline mean, AND
      * median net return > baseline median (centered, not fat-tail, lift), AND
      * precision lift > 1.0 (the picked rows beat the base positive rate).
    This deliberately rejects the high-mean / median~=baseline / lift<1 case that a
    naive 'max expectancy' read would crown."""
    if base is None or row.get("expectancy") is None:
        return False
    if not (row["expectancy"] > 0 and base.get("expectancy") is not None
            and row["expectancy"] > base["expectancy"]):
        return False
    rm, bm = row.get("median_return"), base.get("median_return")
    if rm is None or bm is None or not (rm > bm):
        return False
    return (row.get("lift") or 0.0) > 1.0


def _verdict(sweep: dict) -> str:
    """Headline read at realistic (>=5 bps) cost. Asks the outlier-robust question:
    do selections clear the trade-EVERYTHING baseline on the MEDIAN and on label
    precision (not just the fat-tail mean)? The 'best' is ranked by median net
    return, so a high-mean/median~=baseline rule cannot win."""
    base_by_cost = {r["cost_bps"]: r for r in sweep["baseline"]}
    rows = sweep["threshold"] + sweep["top_k"]
    costed = [r for r in rows if r["cost_bps"] >= 5.0 and r["n_trades"] >= 30
              and r["expectancy"] is not None]
    if not costed:
        return ("INCONCLUSIVE: no selection produced >=30 trades with a defined "
                "expectancy at >=5 bps cost.")
    pos = [r for r in costed if r["expectancy"] > 0]
    robust = [r for r in costed if _is_robust_edge(r, base_by_cost.get(r["cost_bps"]))]

    def _tag(r):
        b = base_by_cost.get(r["cost_bps"])
        bm = b["median_return"] if b else None
        be = b["expectancy"] if b else None
        flag = " [OUTLIER-DRIVEN: mean>>median]" if _is_outlier_driven(r) else ""
        return (f"{r['selection']}={r['param']} @ {r['cost_bps']}bps: "
                f"median {r['median_return']} (baseline {bm}), mean {r['expectancy']} "
                f"(baseline {be}), hit {r['hit_rate']}, PF {r['profit_factor']}, "
                f"lift {r['lift']}, {r['n_trades']} trades{flag}")

    if not pos:
        worst_to_best = max(costed, key=lambda r: r["expectancy"])
        return (f"NO TRADEABLE EDGE: every selection has <=0 net expectancy once "
                f"costs are paid (best {_tag(worst_to_best)}). The weak AUC lift "
                f"does not survive selection + costs.")
    if not robust:
        best_mean = max(pos, key=lambda r: r["expectancy"])
        return (f"NO ROBUST EDGE: {len(pos)}/{len(costed)} selections are net-"
                f"positive, but NONE beats the trade-everything baseline on BOTH "
                f"median return and precision-lift — the positive means are "
                f"outlier-/drift-driven, not a centered ranking edge. Highest mean: "
                f"{_tag(best_mean)}. Not actionable.")
    best = max(robust, key=lambda r: (r["median_return"], r["expectancy"]))
    frac = len(robust) / len(costed)
    tier = "SELECTION EDGE" if frac >= 0.5 else "MARGINAL SELECTION EDGE"
    return (f"{tier}: {len(robust)}/{len(costed)} costed selections beat the all-"
            f"signal baseline on median return AND precision-lift; best by median is "
            f"{_tag(best)}. Weak but centered — worth a guarded, still-offline "
            f"follow-up (robustness across regimes, position overlap, capacity at "
            f"the top-prob names). NOT a go-live signal.")


def _write_report(oos_summary: dict, sweep: dict, meta: dict,
                  oos: dict, max_export_rows: int, suffix: str) -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = config.REPORTS_DIR / "gbm_tradeability_report.md"
    json_path = config.REPORTS_DIR / "gbm_tradeability_report.json"

    L: List[str] = []
    L.append("# Arm B tradeability evaluation (offline)")
    L.append("")
    L.append(f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_")
    L.append("")
    L.append("Read-only / offline. No IBKR, no orders, no model saved "
             "(rf_models.joblib / lstm_checkpoint.pt / model_metrics.json untouched).")
    L.append("Reuses Arm B's leakage-safe panel walk-forward CV (date-axis split, "
             f"purge {meta['purge']} bars = ML_HORIZON; pooled sklearn "
             "HistGradientBoosting, seed "
             f"{meta['seed']}). Tradeable return = Close[t+{meta['horizon']}]/"
             "Close[t]-1 (H-bar hold); in binary mode this is exactly the quantity "
             "the label thresholds, so the eval is self-consistent.")
    L.append("Costs are deducted per trade as a TOTAL round-trip cost "
             "(net = gross - bps/1e4).")
    L.append("")
    L.append("> Reproducibility: the pipeline is deterministic GIVEN the price data "
             "(offline-tested), but it pulls live yfinance auto-adjusted history, so "
             "re-running on another day — or after a dividend/split re-adjusts a "
             "symbol's history — shifts the exact figures (the AUC and high-prob tail "
             "move a little). The numbers below are a snapshot of the data as of the "
             "generation time; read the conclusions, not the last decimal.")
    L.append("")
    L.append("## Setup")
    L.append("")
    L.append(f"- Arm : **B only** (base 14 + SPY relative-strength = "
             f"{len(meta['feature_cols'])} features) | label mode : {meta['mode']}")
    L.append(f"- Universe : {meta['n_symbols_requested']} requested "
             f"({meta['universe_src']}), {oos_summary['n_symbols']} used")
    L.append(f"- CV splits : {meta['n_splits']} | purge : {meta['purge']} | "
             f"horizon : {meta['horizon']} bars | candlesticks : "
             f"{'on' if meta['candles'] else 'off'}")
    L.append(f"- Thresholds : {meta['thresholds']} | top-k : {meta['topk']} | "
             f"costs(bps) : {meta['costs_bps']}")
    L.append("")
    L.append("## Out-of-sample prediction pool")
    L.append("")
    L.append(f"- Rows : **{oos_summary['n_oos']}** across {oos_summary['n_folds']} "
             f"walk-forward folds, {oos_summary['date_min']} → {oos_summary['date_max']}")
    L.append(f"- Pooled AUC : {oos_summary['pooled_auc']} | base positive rate : "
             f"{oos_summary['base_pos_rate']}")
    L.append(f"- Prob [min/mean/max] : {oos_summary['prob_min']} / "
             f"{oos_summary['prob_mean']} / {oos_summary['prob_max']}")
    L.append(f"- Forward return mean / median : {oos_summary['fwd_ret_mean']} / "
             f"{oos_summary['fwd_ret_median']}")
    L.append("")
    L.append("## Baseline — trade every signal")
    L.append("")
    L.append("_Return-space base rate: a threshold / top-k rule only adds value if "
             "it beats this at the same cost._")
    L.append("")
    L.extend(_metric_table(sweep["baseline"], "all"))
    L.append("")
    L.append("## Threshold sweep")
    L.append("")
    L.extend(_metric_table(sweep["threshold"], "thr"))
    L.append("")
    L.append("## Top-k per day")
    L.append("")
    L.extend(_metric_table(sweep["top_k"], "k"))
    L.append("")
    L.append(f"**Verdict — {_verdict(sweep)}**")
    L.append("")
    L.append("_Legend: PF=profit factor (n/a = no losing trades), expect=per-trade "
             "expectancy (mean net return), maxDD=max drawdown of the cumulative "
             "per-trade equity curve, prec=precision vs label, lift=precision/"
             "base-rate, ann~=rough non-compounded annualized (avg×252/horizon, "
             "ignores overlap), flag='outlier' when mean expectancy > 4x |median| "
             "(a few fat-tailed winners, not a centered edge — treat the mean with "
             "suspicion and read the median). Full per-row OOS predictions are in "
             "the .json under `oos_predictions` (columnar)._")

    md_path.write_text("\n".join(L), encoding="utf-8")

    # Columnar OOS-prediction export (rounded, optionally capped).
    n_total = int(len(oos["prob"]))
    n_exp = n_total if max_export_rows <= 0 else min(n_total, int(max_export_rows))
    predictions = {
        "n_total": n_total,
        "n_exported": n_exp,
        "capped": n_exp < n_total,
        "columns": ["date", "symbol", "prob", "label", "fwd_ret", "entry", "exit", "fold"],
        "date": [str(d)[:10] for d in oos["date"][:n_exp]],
        "symbol": [str(s) for s in oos["symbol"][:n_exp]],
        "prob": [round(float(v), 4) for v in oos["prob"][:n_exp]],
        "label": [int(v) for v in oos["label"][:n_exp]],
        "fwd_ret": [round(float(v), 6) for v in oos["fwd_ret"][:n_exp]],
        "entry": [round(float(v), 4) for v in oos["entry"][:n_exp]],
        "exit": [round(float(v), 4) for v in oos["exit"][:n_exp]],
        "fold": [int(v) for v in oos["fold"][:n_exp]],
    }
    json_path.write_text(json.dumps({
        "meta": meta,
        "oos_summary": oos_summary,
        "threshold_sweep": sweep["threshold"],
        "top_k_sweep": sweep["top_k"],
        "verdict": _verdict(sweep),
        "oos_predictions": predictions,
    }, indent=2), encoding="utf-8")
    return md_path


# ─────────────────────────── main ───────────────────────────
def _resolve_universe(args) -> tuple:
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        src = "explicit"
    elif args.all:
        syms = gbm._symbols_from_metrics()
        src = "model_metrics.json rf keys"
    else:
        syms = list(config.WATCHLIST)
        src = "watchlist"
    if args.limit and args.limit > 0:
        syms = syms[: args.limit]
    return syms, src


def _parse_floats(s: str, default) -> tuple:
    if not s:
        return tuple(default)
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _parse_ints(s: str, default) -> tuple:
    if not s:
        return tuple(default)
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main(argv: Optional[List[str]] = None) -> int:
    from logging_setup import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Arm B tradeability evaluation (offline; no IBKR, no orders, no model saved)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: config.WATCHLIST)")
    parser.add_argument("--all", action="store_true",
                        help="Use the full universe (rf keys in model_metrics.json)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Cap the resolved universe to the first N symbols")
    parser.add_argument("--mode", type=str, default="binary",
                        choices=["binary", "triple_barrier"],
                        help="Label mode (Arm B was validated on binary)")
    parser.add_argument("--n-splits", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--purge", type=int, default=None,
                        help="Purge gap in bars (default: config.ML_HORIZON)")
    parser.add_argument("--candles", type=str, default="off",
                        choices=["on", "off"], help="Include candlestick features")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated prob thresholds (default 0.55..0.75)")
    parser.add_argument("--topk", type=str, default=None,
                        help="Comma-separated top-k per day (default 1,3,5,10)")
    parser.add_argument("--costs-bps", type=str, default=None,
                        help="Comma-separated round-trip costs in bps (default 0,5,10)")
    parser.add_argument("--max-export-rows", type=int, default=0,
                        help="Cap raw OOS rows written to JSON (0 = all)")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast sanity: 3 symbols, 2 splits")
    args = parser.parse_args(argv)

    symbols, universe_src = _resolve_universe(args)
    mode = args.mode
    n_splits = max(2, int(args.n_splits))
    purge = int(args.purge if args.purge is not None else config.ML_HORIZON)
    candles = args.candles == "on"
    thresholds = _parse_floats(args.thresholds, DEFAULT_THRESHOLDS)
    topk = _parse_ints(args.topk, DEFAULT_TOPK)
    costs_bps = _parse_floats(args.costs_bps, DEFAULT_COSTS_BPS)

    if args.smoke:
        symbols = symbols[:3]
        n_splits = 2

    horizon = int(config.ML_HORIZON)
    seed = int(config.RANDOM_STATE)

    print("=== Arm B tradeability evaluation (offline; no IBKR, no orders, no model saved) ===")
    print(f"Symbols: {len(symbols)} ({universe_src}) | mode: {mode} | "
          f"CV splits: {n_splits} | purge: {purge} | horizon: {horizon} | seed: {seed}")
    print(f"Thresholds: {thresholds} | top-k: {topk} | costs(bps): {costs_bps}")

    spy_df = None
    try:
        spy_df = fetch_market_benchmark()
    except Exception:
        logger.warning("could not fetch %s benchmark; Arm B relative-strength "
                       "features fall back to neutral 0.0", config.MARKET_BENCHMARK_SYMBOL)

    print("Building Arm B panel (with forward returns)...", flush=True)
    panel = build_tradeable_panel(symbols, candles, mode, horizon, spy_df)
    if panel is None:
        print("No usable rows — aborting (no report written).")
        print("No IBKR connection was made, no orders placed, no model saved.")
        return 1
    print(f"  panel: {panel['n_symbols_used']} symbols, {len(panel['y'])} rows, "
          f"{panel['X'].shape[1]} features ({panel['n_symbols_skipped']} skipped)",
          flush=True)

    print("Generating leakage-safe OOS predictions (Arm B CV)...", flush=True)
    oos = panel_oos_records(panel, horizon, n_splits, purge, gbm._gbm_factory)
    if oos is None:
        print("No OOS predictions produced (single-class / too short) — aborting.")
        print("No IBKR connection was made, no orders placed, no model saved.")
        return 1

    oos_summary = _oos_summary(oos, horizon)
    print(f"  OOS: {oos_summary['n_oos']} rows, AUC {oos_summary['pooled_auc']}, "
          f"base rate {oos_summary['base_pos_rate']}", flush=True)

    sweep = run_sweep(oos, thresholds, topk, costs_bps, horizon)

    meta = {
        "arm": "B",
        "feature_cols": panel["feature_cols"],
        "mode": mode,
        "n_symbols_requested": len(symbols),
        "universe_src": universe_src,
        "n_splits": n_splits,
        "purge": purge,
        "horizon": horizon,
        "candles": candles,
        "seed": seed,
        "thresholds": list(thresholds),
        "topk": list(topk),
        "costs_bps": list(costs_bps),
    }
    report = _write_report(oos_summary, sweep, meta, oos, args.max_export_rows, args.suffix)

    print("\n=== SUMMARY ===")
    print(f"OOS rows {oos_summary['n_oos']} | AUC {oos_summary['pooled_auc']} | "
          f"base rate {oos_summary['base_pos_rate']}")
    print(_verdict(sweep))
    print(f"\nReport written to {report}")
    print("No IBKR connection was made, no orders placed, no model saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
