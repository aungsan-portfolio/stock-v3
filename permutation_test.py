"""
permutation_test.py — read-only shuffled-label null test.

The decisive answer to "is there any edge, or is it luck?": for each symbol we
compute the REAL pooled walk-forward CV-AUC, then a NULL distribution of the same
AUC on many label-shuffled copies. If the real AUC does not beat the symbol's own
shuffled-label distribution, the model has learned nothing on that symbol. Across
many symbols we control multiple-testing with Benjamini-Hochberg FDR, so a few
"good" symbols out of a hundred are not mistaken for skill.

This module NEVER connects to IBKR, places orders, trains/saves production models,
or changes any trading/prediction/risk logic. It only reads price data, scores
in-memory RF models, and writes a report to REPORTS_DIR (mirrors model_doctor.py).

Run:
    python -X utf8 permutation_test.py                 # small default sample
    python -X utf8 permutation_test.py --all           # full universe (slow)
    python -X utf8 permutation_test.py --symbols AAPL,MSFT --n-shuffles 100
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np

import config
from ai_engine import cv_pooled_auc
from data_manager import fetch_ohlcv, build_features, make_labels, get_feature_columns

logger = logging.getLogger(__name__)
FEATURE_COLS = get_feature_columns()

PERMUTATION_REPORT_MD = config.REPORTS_DIR / "permutation_test_report.md"
PERMUTATION_REPORT_JSON = config.REPORTS_DIR / "permutation_test_report.json"


def benjamini_hochberg(pvals: List[float], q: float) -> List[bool]:
    """Benjamini-Hochberg step-up FDR. Returns a parallel list of booleans:
    True where the hypothesis is rejected (significant) at false-discovery rate q.

    Find the largest rank k (1-based, p ascending) with p_(k) <= q*k/m, then
    reject all hypotheses with rank <= k.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    k_max = 0
    for rank, idx in enumerate(order, start=1):
        if pvals[idx] <= q * rank / m:
            k_max = rank
    significant = [False] * m
    for rank, idx in enumerate(order, start=1):
        if rank <= k_max:
            significant[idx] = True
    return significant


def _build_xy(symbol: str):
    """Real feature matrix + labels for a symbol (same path training uses)."""
    df = fetch_ohlcv(symbol)
    feat = build_features(df)
    labels = make_labels(feat)
    valid = labels.dropna().index
    X = feat.loc[valid, FEATURE_COLS].to_numpy(dtype=float)
    y = labels.loc[valid].to_numpy().astype(int)
    return X, y


def evaluate_symbol(
    symbol: str, n_shuffles: int, horizon: int, seed: int
) -> Optional[dict]:
    """Real vs shuffled-label pooled CV-AUC for one symbol. None if unusable."""
    X, y = _build_xy(symbol)
    if len(X) < 100 or len(np.unique(y)) < 2:
        logger.debug("%s skipped (n=%d, classes=%d)", symbol, len(X), len(np.unique(y)))
        return None

    real_auc = cv_pooled_auc(X, y, horizon=horizon)
    if real_auc is None:
        return None

    rng = np.random.default_rng(seed)
    null: List[float] = []
    for _ in range(n_shuffles):
        a = cv_pooled_auc(X, rng.permutation(y), horizon=horizon)
        if a is not None:
            null.append(a)
    if not null:
        return None

    null_arr = np.asarray(null, dtype=float)
    # Add-one smoothed p-value: P(null AUC >= real AUC). Never exactly 0.
    p_value = (1.0 + float((null_arr >= real_auc).sum())) / (1.0 + len(null_arr))
    return {
        "symbol": symbol,
        "real_auc": round(float(real_auc), 4),
        "null_mean": round(float(null_arr.mean()), 4),
        "null_p95": round(float(np.percentile(null_arr, 95)), 4),
        "n_null": int(len(null_arr)),
        "p_value": round(float(p_value), 4),
        "n_samples": int(len(X)),
        "positive_rate": round(float(np.mean(y)), 4),
    }


def _write_report(summary: dict) -> "config.Path":
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Permutation / Shuffled-Label Null Test")
    lines.append("")
    lines.append(f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_")
    lines.append("")
    lines.append("Read-only. No IBKR connection, no orders, no models trained or saved.")
    lines.append("Each symbol's REAL walk-forward CV-AUC is compared against its own")
    lines.append("shuffled-label null distribution; multiple-testing is controlled with")
    lines.append("Benjamini-Hochberg FDR. A symbol is 'significant' only if its real AUC")
    lines.append("beats luck after that correction.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Label mode : `{summary['label_mode']}`")
    lines.append(f"- Symbols tested : {summary['n_symbols_tested']}")
    if summary["dropped_for_sample"]:
        lines.append(f"- Dropped (sample cap) : {summary['dropped_for_sample']} "
                     "(use `--all` to test everything)")
    lines.append(f"- Null draws / symbol : {summary['n_shuffles']}")
    lines.append(f"- FDR q : {summary['fdr_q']}")
    lines.append(f"- Mean real AUC : {summary['mean_real_auc']}")
    lines.append(f"- **Significant after FDR : {summary['n_significant']} / "
                 f"{summary['n_symbols_tested']}**")
    lines.append("")
    if summary["survivors"]:
        lines.append("Symbols that beat their shuffled-label null (FDR-significant):")
        lines.append("")
        lines.append("  " + ", ".join(summary["survivors"]))
    else:
        lines.append("**No symbol beat its shuffled-label null after FDR correction** — "
                     "consistent with no measurable edge from this feature/label set.")
    lines.append("")
    lines.append("## Per-symbol")
    lines.append("")
    lines.append("| Symbol | real AUC | null mean | null p95 | p-value | FDR sig | n |")
    lines.append("|--------|---------:|----------:|---------:|--------:|:-------:|--:|")
    for r in sorted(summary["per_symbol"], key=lambda x: x["p_value"]):
        lines.append(
            f"| {r['symbol']} | {r['real_auc']:.3f} | {r['null_mean']:.3f} | "
            f"{r['null_p95']:.3f} | {r['p_value']:.3f} | "
            f"{'yes' if r.get('significant_fdr') else 'no'} | {r['n_samples']} |"
        )
    lines.append("")
    PERMUTATION_REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    PERMUTATION_REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return PERMUTATION_REPORT_MD


def run_permutation_test(
    symbols: Optional[List[str]] = None,
    n_shuffles: Optional[int] = None,
    sample: Optional[int] = None,
    seed: Optional[int] = None,
) -> int:
    """Run the shuffled-label null test and write a report. No IBKR, no orders."""
    print("\n=== Permutation / shuffled-label null test (read-only; no IBKR, no orders) ===")
    n_shuffles = int(n_shuffles or config.PERMUTATION_N_SHUFFLES)
    seed = int(seed if seed is not None else config.RANDOM_STATE)
    horizon = int(config.ML_HORIZON)

    if symbols is None:
        symbols = list(config.WATCHLIST)
    if sample is None:
        sample = int(getattr(config, "PERMUTATION_SAMPLE_SYMBOLS", 0) or 0)

    dropped = 0
    if sample and 0 < sample < len(symbols):
        dropped = len(symbols) - sample
        symbols = symbols[:sample]

    scope = f"sampled, dropped {dropped} (use --all for everything)" if dropped else "all"
    print(f"Symbols: {len(symbols)} ({scope})")
    print(f"Null draws/symbol: {n_shuffles} | label mode: {config.LABEL_MODE} | "
          f"FDR q: {config.PERMUTATION_FDR_Q}")
    print("This trains in-memory RF models only (never saved); it may take a few minutes.\n")

    results: List[dict] = []
    for i, sym in enumerate(symbols):
        try:
            r = evaluate_symbol(sym, n_shuffles, horizon, seed + i)
        except Exception:
            logger.exception("permutation test failed for %s", sym)
            continue
        if r is not None:
            results.append(r)
            logger.info(
                "%-6s real_auc=%.3f null_mean=%.3f p=%.3f",
                sym, r["real_auc"], r["null_mean"], r["p_value"],
            )

    q = float(config.PERMUTATION_FDR_Q)
    sig = benjamini_hochberg([r["p_value"] for r in results], q)
    for r, s in zip(results, sig):
        r["significant_fdr"] = bool(s)
    survivors = [r["symbol"] for r in results if r["significant_fdr"]]

    summary = {
        "label_mode": config.LABEL_MODE,
        "n_symbols_tested": len(results),
        "n_shuffles": n_shuffles,
        "fdr_q": q,
        "n_significant": len(survivors),
        "survivors": survivors,
        "dropped_for_sample": dropped,
        "mean_real_auc": (
            round(float(np.mean([r["real_auc"] for r in results])), 4) if results else None
        ),
        "per_symbol": results,
    }
    report = _write_report(summary)

    print(f"\nTested {len(results)} symbol(s). "
          f"Significant after FDR (q={q}): {len(survivors)}")
    if survivors:
        print("  Survivors: " + ", ".join(survivors))
    else:
        print("  No symbol beat its shuffled-label null — consistent with no edge.")
    print(f"\nReport written to {report}")
    print("No IBKR connection was made and no orders were placed.")
    return 0


if __name__ == "__main__":
    from logging_setup import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser(
        description="Shuffled-label permutation null test (read-only; no IBKR, no orders)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Test the full universe (ignore PERMUTATION_SAMPLE_SYMBOLS)",
    )
    parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols to test (overrides the watchlist)",
    )
    parser.add_argument("--n-shuffles", type=int, default=None, help="Null draws per symbol")
    args = parser.parse_args()

    syms = None
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    raise SystemExit(
        run_permutation_test(
            symbols=syms,
            n_shuffles=args.n_shuffles,
            sample=0 if args.all else None,
        )
    )
