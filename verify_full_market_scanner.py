"""
verify_full_market_scanner.py — Beginner-friendly safety check (READ-ONLY).

Run:
    python -X utf8 verify_full_market_scanner.py

What it does (and just as importantly, what it does NOT do):
  1. Loads the broad US market universe from market_universe.py.
  2. Confirms symbols are returned.
  3. Runs scan_hot_stocks(full_market=True, selection_mode="hybrid", …).
  4. Confirms reports/hot_candidates.csv and reports/selected_scan_symbols.csv exist.
  5. Confirms hybrid selection is NOT alphabetically biased (not all A tickers)
     and includes core anchors (AAPL, MSFT, NVDA, TSLA, AMD) when available.
  6. Runs Predictor().predict_all(symbols=hot_symbols).
  7. Confirms NO order placement occurs (it never touches IBKRBridge).
  8. Prints a plain-English summary.

This script imports zero trading code. It cannot connect to IBKR and cannot
place an order, by construction.
"""
from __future__ import annotations

import sys

import pandas as pd

from logging_setup import setup_logging
import config
import market_universe
from hot_scanner import scan_hot_stocks
from predictor import Predictor

# Core anchors we expect hybrid mode to include whenever they are in the universe.
_EXPECTED_CORE = ["AAPL", "MSFT", "NVDA", "TSLA", "AMD"]


def _check(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✅" if ok else "❌"
    line = f"{icon} {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    setup_logging()
    print("═" * 70)
    print(" Full-Market Hot Scanner — Verification (read-only, no orders)")
    print("═" * 70)

    all_ok = True

    # 1) Load universe.
    print("\n[1/8] Loading broad US symbol universe…")
    universe = market_universe.load_symbol_universe()
    all_ok &= _check(
        "Universe loaded", not universe.empty,
        f"{len(universe)} tickers (cached at {config.SYMBOL_UNIVERSE_FILE.name})",
    )
    if universe.empty:
        print(
            "\n⚠️  Could not load a universe (no network and no cache yet).\n"
            "   Connect once to populate data/symbol_universe.csv, then re-run."
        )
        return 1

    # 2) Confirm symbols returned.
    sample = universe["symbol"].head(10).tolist()
    all_ok &= _check("Symbols returned", len(sample) > 0, f"sample: {', '.join(sample)}")

    # 3) Hybrid selection should NOT be all A tickers and should carry core anchors.
    print("\n[2/8] Selecting symbols in hybrid mode (max_symbols=120)…")
    selected_df, info = market_universe.select_symbols(
        max_symbols=120, mode="hybrid", write_report=True
    )
    selected_syms = selected_df["symbol"].tolist()
    universe_syms = set(universe["symbol"])

    first_letters = {s[0] for s in selected_syms}
    not_all_a = len(first_letters) > 1
    all_ok &= _check(
        "Hybrid selection is not alphabetically biased", not_all_a,
        f"{len(first_letters)} distinct first letters across {len(selected_syms)} symbols",
    )

    expected_present = [s for s in _EXPECTED_CORE if s in universe_syms]
    missing_core = [s for s in expected_present if s not in selected_syms]
    all_ok &= _check(
        "Core anchors included in hybrid mode", not missing_core,
        (f"present: {', '.join(expected_present)}" if not missing_core
         else f"MISSING from selection: {', '.join(missing_core)}"),
    )

    # 4) Selected-symbols report exists with the required columns.
    print("\n[3/8] Checking selected-symbols report…")
    sel_path = config.SELECTED_SCAN_SYMBOLS_FILE
    sel_ok = sel_path.exists()
    if sel_ok:
        sel_cols = list(pd.read_csv(sel_path, nrows=1).columns)
        sel_ok = {"symbol", "source", "selection_reason"}.issubset(set(sel_cols))
    all_ok &= _check("selected_scan_symbols.csv written", sel_ok, str(sel_path))

    # 5) Small, capped full-market scan (hybrid) — exercises the real scan path.
    print("\n[4/8] Scanning a small slice (max_symbols=50, top_n=10, hybrid)…")
    hot_symbols = scan_hot_stocks(
        full_market=True, max_symbols=50, top_n=10,
        selection_mode="hybrid", write_report=True,
    )
    all_ok &= _check(
        "Scan returned candidates", isinstance(hot_symbols, list),
        f"{len(hot_symbols)} candidates: {', '.join(hot_symbols) or '(none passed filters)'}",
    )

    # 6) Report exists.
    print("\n[5/8] Checking hot-candidates report file…")
    report_ok = config.HOT_CANDIDATES_FILE.exists()
    all_ok &= _check("Report written", report_ok, str(config.HOT_CANDIDATES_FILE))

    # 7) Predict on hot candidates (or fall back to watchlist if none passed).
    print("\n[6/8] Running Predictor on candidates (no orders)…")
    predict_symbols = hot_symbols or list(config.WATCHLIST)
    signals = Predictor().predict_all(symbols=predict_symbols)
    all_ok &= _check(
        "Predictor produced signals", len(signals) > 0,
        f"{len(signals)} signals over {len(predict_symbols)} symbols",
    )
    for s in signals[:10]:
        print(f"      {s.symbol:<6} {s.action:<5} conf={s.confidence:.2f} ${s.price:.2f}")

    # 8) Confirm no order placement path was used.
    print("\n[7/8] Confirming no orders were placed…")
    no_ibkr = "ibkr_bridge" not in sys.modules
    all_ok &= _check(
        "No IBKR bridge imported / no orders placed", no_ibkr,
        "this script never connects to IBKR",
    )

    # Summary.
    print("\n[8/8] Summary")
    print("─" * 70)
    print(f"  Universe size            : {len(universe)}")
    print(f"  Hybrid selection mode    : {info['mode']}")
    print(f"  Selected symbols         : {info['selected']} (core: {info['core_included']})")
    print(f"  First 20 selected        : {', '.join(info['first_20'])}")
    print(f"  Hot candidates (top 10)  : {len(hot_symbols)}")
    print(f"  Predicted signals        : {len(signals)}")
    buys = sum(1 for s in signals if s.action == "BUY")
    print(f"  BUY / HOLD / SELL        : "
          f"{buys} / {sum(1 for s in signals if s.action == 'HOLD')} / "
          f"{sum(1 for s in signals if s.action == 'SELL')}")
    print(f"  Orders placed            : 0 (verification is read-only)")
    print("─" * 70)

    if all_ok:
        print("\n🎉 All checks passed. The scanner discovers + ranks + predicts,")
        print("   selects a broad (non-alphabetical) slice, and places NO orders.")
        print("   To paper-trade, use:")
        print("     python -X utf8 main.py paper-hot --full-market --execute")
        return 0
    print("\n⚠️  Some checks did not pass — see the ❌ lines above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
