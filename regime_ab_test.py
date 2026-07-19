"""
regime_ab_test.py — read-only A/B of USE_REGIME_FEATURES (OFF vs ON).

Decisive question: does WIDENING the model feature vector with the Item-3 regime
features (realized_vol, vol_regime, mom_short/long, ts_rank, dist_high) recover
any measurable edge, or is the 20-feature model still a coin-flip like the frozen
14-feature one?

For each symbol we fetch OHLCV ONCE and build BOTH feature matrices from it:
    arm OFF : the frozen 14-feature set (USE_REGIME_FEATURES = False)
    arm ON  : the 20-feature set        (USE_REGIME_FEATURES = True)
The two arms are evaluated on the IDENTICAL set of rows/dates (the regime-ON
valid index, which is the shorter one because the 252-bar regime warmup drops
more leading rows). So the ONLY thing that varies between arms is the feature
set — same dates, same labels, same null permutations (paired). For each arm we
report the pooled walk-forward CV-AUC, a shuffled-label null distribution + add-
one p-value with Benjamini-Hochberg FDR, and the ai_engine pre-refit holdout AUC
(the same number a real retrain would print — computed here WITHOUT saving).

SAFETY (same contract as permutation_test.py): this module NEVER connects to
IBKR, places/cancels orders, trains or SAVES any production model
(rf_models.joblib / model_metrics.json are never touched), or changes any
trading / prediction / risk logic. It only reads price data, scores in-memory RF
models, and writes a report to REPORTS_DIR. USE_MARKET_RELATIVE_FEATURES is held
OFF throughout (it needs market_df wired into the engines first).

Run:
    python -X utf8 regime_ab_test.py                      # watchlist, both modes
    python -X utf8 regime_ab_test.py --smoke              # fast sanity (2 sym, 3 shuffles)
    python -X utf8 regime_ab_test.py --modes binary --n-shuffles 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from typing import List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# Hold the unwired market-relative features OFF for the whole experiment.
config.USE_MARKET_RELATIVE_FEATURES = False

# Imported AFTER config is in hand; none of these capture the regime flag at
# import time in a way that matters — build_features() / get_feature_columns()
# both read config.USE_REGIME_FEATURES at CALL time, and cv_pooled_auc/build_rf
# operate on raw (X, y) arrays.
from ai_engine import build_rf, cv_pooled_auc, _subsample_train_indices  # noqa: E402
import eval_metrics  # noqa: E402
from data_manager import (  # noqa: E402
    fetch_ohlcv,
    build_features,
    make_labels,
    get_feature_columns,
)
from permutation_test import benjamini_hochberg  # noqa: E402


def _set_regime(on: bool) -> List[str]:
    """Flip the regime flag and return the resulting feature-column list."""
    config.USE_REGIME_FEATURES = bool(on)
    return get_feature_columns()


def _holdout_auc(X: np.ndarray, y: np.ndarray, horizon: int) -> Optional[float]:
    """ai_engine's pre-refit walk-forward holdout AUC — replicated read-only.

    Grades the production RF RECIPE (build_rf + ML_HORIZON striding) on a final
    unseen tail of fraction MODEL_HOLDOUT_RATIO. Identical math to
    StockRFEngine.train (ai_engine.py), but nothing is saved to disk.
    """
    ratio = float(getattr(config, "MODEL_HOLDOUT_RATIO", 0.0) or 0.0)
    if not (0.0 < ratio < 1.0):
        return None
    cut = int(len(X) * (1.0 - ratio))
    if cut < 50 or (len(X) - cut) < 20:
        return None
    h_sub = _subsample_train_indices(cut, horizon)
    X_h, y_h = X[:cut][h_sub], y[:cut][h_sub]
    X_ho, y_ho = X[cut:], y[cut:]
    if len(np.unique(y_h)) < 2:
        return None
    graded = build_rf()
    graded.fit(X_h, y_h)
    pp = eval_metrics.proba_pos(graded, X_ho)
    if pp is None:
        return None
    return eval_metrics.safe_auc(y_ho, pp)


def _null_pvalue(
    X: np.ndarray, y: np.ndarray, real_auc: float, n_shuffles: int,
    horizon: int, seed: int,
) -> dict:
    """Shuffled-label null distribution for one (X, y). The rng is reseeded from
    ``seed`` so OFF and ON arms (same y after alignment) see the IDENTICAL
    permutation sequence — a paired null comparison."""
    rng = np.random.default_rng(seed)
    null: List[float] = []
    for _ in range(n_shuffles):
        a = cv_pooled_auc(X, rng.permutation(y), horizon=horizon)
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


def _eval_arm(
    X: np.ndarray, y: np.ndarray, n_shuffles: int, horizon: int, seed: int,
) -> Optional[dict]:
    """Full read-only evaluation of one arm on aligned (X, y)."""
    if len(X) < 100 or len(np.unique(y)) < 2:
        return None
    real_auc = cv_pooled_auc(X, y, horizon=horizon)
    if real_auc is None:
        return None
    out = {
        "real_auc": round(float(real_auc), 4),
        "holdout_auc": eval_metrics.round4(_holdout_auc(X, y, horizon)),
        "n_features": int(X.shape[1]),
        "n_samples": int(len(X)),
        "positive_rate": round(float(np.mean(y)), 4),
    }
    out.update(_null_pvalue(X, y, real_auc, n_shuffles, horizon, seed))
    return out


def evaluate_symbol(symbol: str, mode: str, n_shuffles: int, horizon: int,
                    seed: int) -> Optional[dict]:
    """Paired OFF/ON evaluation for one symbol under one label mode."""
    df = fetch_ohlcv(symbol)

    # Build BOTH feature frames from the SAME OHLCV.
    cols_off = _set_regime(False)
    feat_off = build_features(df)
    cols_on = _set_regime(True)
    feat_on = build_features(df)

    # Labels are identical at shared dates (depend only on OHLCV). Align both
    # arms to the regime-ON valid index (the shorter one, longer warmup).
    labels_on = make_labels(feat_on, horizon=horizon, mode=mode)
    common = labels_on.dropna().index
    common = common.intersection(feat_off.index)
    if len(common) < 100:
        logger.debug("%s/%s skipped (aligned n=%d)", symbol, mode, len(common))
        return None

    y = labels_on.loc[common].to_numpy().astype(int)
    X_off = feat_off.loc[common, cols_off].to_numpy(dtype=float)
    X_on = feat_on.loc[common, cols_on].to_numpy(dtype=float)

    off = _eval_arm(X_off, y, n_shuffles, horizon, seed)
    on = _eval_arm(X_on, y, n_shuffles, horizon, seed)
    if off is None or on is None:
        return None

    # Natural (unaligned) regime-OFF real AUC — cross-check vs the existing
    # baseline permutation report (full 14-feature row set, no null).
    lab_off_nat = make_labels(feat_off, horizon=horizon, mode=mode)
    v_nat = lab_off_nat.dropna().index
    X_off_nat = feat_off.loc[v_nat, cols_off].to_numpy(dtype=float)
    y_nat = lab_off_nat.loc[v_nat].to_numpy().astype(int)
    nat_auc = cv_pooled_auc(X_off_nat, y_nat, horizon=horizon) if len(X_off_nat) >= 100 else None

    return {
        "symbol": symbol,
        "aligned_n": int(len(common)),
        "off": off,
        "on": on,
        "delta_real_auc": round(on["real_auc"] - off["real_auc"], 4),
        "delta_holdout_auc": (
            round(on["holdout_auc"] - off["holdout_auc"], 4)
            if (on["holdout_auc"] is not None and off["holdout_auc"] is not None) else None
        ),
        "off_natural_real_auc": eval_metrics.round4(nat_auc),
        "off_natural_n": int(len(X_off_nat)),
    }


def _fdr(results: List[dict], arm: str, q: float) -> List[str]:
    """Tag results[*][arm]['significant_fdr'] and return the survivor symbols."""
    pvals = [r[arm]["p_value"] for r in results]
    sig = benjamini_hochberg(pvals, q)
    survivors = []
    for r, s in zip(results, sig):
        r[arm]["significant_fdr"] = bool(s)
        if s:
            survivors.append(r["symbol"])
    return survivors


def run_mode(symbols: List[str], mode: str, n_shuffles: int, horizon: int,
             seed: int, q: float) -> dict:
    print(f"\n--- label mode: {mode} | symbols: {len(symbols)} | "
          f"null draws/arm: {n_shuffles} ---", flush=True)
    results: List[dict] = []
    for i, sym in enumerate(symbols):
        try:
            r = evaluate_symbol(sym, mode, n_shuffles, horizon, seed + i)
        except Exception:
            logger.exception("regime A/B failed for %s/%s", sym, mode)
            continue
        if r is None:
            print(f"  {sym:<6} skipped", flush=True)
            continue
        results.append(r)
        print(
            f"  {sym:<6} OFF real={r['off']['real_auc']:.3f} p={r['off']['p_value']:.3f} | "
            f"ON real={r['on']['real_auc']:.3f} p={r['on']['p_value']:.3f} | "
            f"Δreal={r['delta_real_auc']:+.3f} | "
            f"hold OFF={r['off']['holdout_auc']} ON={r['on']['holdout_auc']} | "
            f"n={r['aligned_n']}",
            flush=True,
        )

    surv_off = _fdr(results, "off", q) if results else []
    surv_on = _fdr(results, "on", q) if results else []

    def _mean(arm, key):
        vals = [r[arm][key] for r in results if r[arm].get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "label_mode": mode,
        "n_symbols_tested": len(results),
        "n_shuffles": n_shuffles,
        "fdr_q": q,
        "mean_real_auc_off": _mean("off", "real_auc"),
        "mean_real_auc_on": _mean("on", "real_auc"),
        "mean_holdout_auc_off": _mean("off", "holdout_auc"),
        "mean_holdout_auc_on": _mean("on", "holdout_auc"),
        "mean_delta_real_auc": (
            round(float(np.mean([r["delta_real_auc"] for r in results])), 4) if results else None
        ),
        "survivors_off": surv_off,
        "survivors_on": surv_on,
        "n_significant_off": len(surv_off),
        "n_significant_on": len(surv_on),
        "per_symbol": results,
    }


def _verdict(mode_summary: dict) -> str:
    s = mode_summary
    on_surv = s["n_significant_on"]
    off_surv = s["n_significant_off"]
    d = s["mean_delta_real_auc"]
    if on_surv > off_surv and on_surv > 0:
        return (f"EDGE SIGNAL: regime ON beats the shuffled null on {on_surv} symbol(s) "
                f"(OFF: {off_surv}); mean Δreal_auc {d:+.4f}. Worth a guarded retrain.")
    if d is not None and d >= 0.01:
        return (f"WEAK LIFT: no FDR survivor, but mean real-AUC rises {d:+.4f} with regime ON. "
                f"Borderline — more symbols / shuffles before trusting it.")
    return (f"NO EDGE: regime ON does not beat its shuffled null (survivors ON={on_surv}, "
            f"OFF={off_surv}); mean Δreal_auc {d:+.4f}. Keep USE_REGIME_FEATURES=False.")


def _write_report(summaries: List[dict], suffix: str) -> "config.Path":
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"_{suffix}" if suffix else ""
    md_path = config.REPORTS_DIR / f"regime_ab_report{stamp}.md"
    json_path = config.REPORTS_DIR / f"regime_ab_report{stamp}.json"

    L: List[str] = []
    L.append("# Regime-feature A/B (USE_REGIME_FEATURES OFF vs ON)")
    L.append("")
    L.append(f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_")
    L.append("")
    L.append("Read-only. No IBKR, no orders, no model saved (rf_models.joblib / "
             "model_metrics.json untouched). USE_MARKET_RELATIVE_FEATURES held OFF.")
    L.append("Both arms are scored on the IDENTICAL aligned rows/dates and the same "
             "paired shuffled-label null, so only the feature set differs.")
    L.append("")
    for s in summaries:
        L.append(f"## Label mode: `{s['label_mode']}`")
        L.append("")
        L.append(f"- Symbols tested : {s['n_symbols_tested']}")
        L.append(f"- Null draws / arm : {s['n_shuffles']} | FDR q : {s['fdr_q']}")
        L.append(f"- Mean real CV-AUC : OFF **{s['mean_real_auc_off']}** → "
                 f"ON **{s['mean_real_auc_on']}** (Δ {s['mean_delta_real_auc']:+})")
        L.append(f"- Mean holdout AUC : OFF {s['mean_holdout_auc_off']} → "
                 f"ON {s['mean_holdout_auc_on']}")
        L.append(f"- FDR survivors : OFF {s['n_significant_off']} "
                 f"{s['survivors_off']} | ON {s['n_significant_on']} {s['survivors_on']}")
        L.append("")
        L.append(f"**Verdict — {_verdict(s)}**")
        L.append("")
        L.append("| Symbol | n | OFF real | ON real | Δreal | OFF p | ON p | "
                 "OFF FDR | ON FDR | OFF hold | ON hold |")
        L.append("|--------|--:|---------:|--------:|------:|------:|-----:|:------:|:------:|---------:|--------:|")
        for r in sorted(s["per_symbol"], key=lambda x: x["on"]["p_value"]):
            L.append(
                f"| {r['symbol']} | {r['aligned_n']} | {r['off']['real_auc']:.3f} | "
                f"{r['on']['real_auc']:.3f} | {r['delta_real_auc']:+.3f} | "
                f"{r['off']['p_value']:.3f} | {r['on']['p_value']:.3f} | "
                f"{'yes' if r['off'].get('significant_fdr') else 'no'} | "
                f"{'yes' if r['on'].get('significant_fdr') else 'no'} | "
                f"{r['off']['holdout_auc']} | {r['on']['holdout_auc']} |"
            )
        L.append("")

    md_path.write_text("\n".join(L), encoding="utf-8")
    json_path.write_text(json.dumps({"modes": summaries}, indent=2), encoding="utf-8")
    return md_path


def main(argv: Optional[List[str]] = None) -> int:
    from logging_setup import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Read-only A/B of USE_REGIME_FEATURES (no IBKR, no orders, no model saved)"
    )
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: config.WATCHLIST)")
    parser.add_argument("--modes", type=str, default="binary,triple_barrier",
                        help="Comma-separated label modes to test")
    parser.add_argument("--n-shuffles", type=int, default=None,
                        help="Null draws per arm (default: config.PERMUTATION_N_SHUFFLES)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Report filename suffix (regime_ab_report_<suffix>.md)")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast sanity run: 2 symbols, 3 shuffles, binary only")
    args = parser.parse_args(argv)

    symbols = ([s.strip().upper() for s in args.symbols.split(",") if s.strip()]
               if args.symbols else list(config.WATCHLIST))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    n_shuffles = int(args.n_shuffles or config.PERMUTATION_N_SHUFFLES)
    suffix = args.suffix

    if args.smoke:
        symbols = symbols[:2]
        modes = ["binary"]
        n_shuffles = 3
        suffix = suffix or "smoke"

    horizon = int(config.ML_HORIZON)
    seed = int(config.RANDOM_STATE)
    q = float(config.PERMUTATION_FDR_Q)

    print("=== Regime-feature A/B (read-only; no IBKR, no orders, no model saved) ===")
    print(f"Symbols: {symbols}")
    print(f"Modes: {modes} | null draws/arm: {n_shuffles} | seed: {seed} | horizon: {horizon}")

    summaries = [run_mode(symbols, m, n_shuffles, horizon, seed, q) for m in modes]
    report = _write_report(summaries, suffix)

    print("\n=== SUMMARY ===")
    for s in summaries:
        print(f"[{s['label_mode']}] mean real-AUC OFF {s['mean_real_auc_off']} -> "
              f"ON {s['mean_real_auc_on']} (Δ {s['mean_delta_real_auc']:+}) | "
              f"FDR survivors OFF {s['n_significant_off']} / ON {s['n_significant_on']}")
        print(f"    {_verdict(s)}")
    print(f"\nReport written to {report}")
    print("No IBKR connection was made, no orders placed, no model saved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
