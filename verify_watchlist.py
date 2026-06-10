"""
verify_watchlist.py — Production verification layer (read-only, no orders).

Runs the full WATCHLIST through the live prediction + backtest paths and checks
the production-safety invariants, then writes a machine-readable summary JSON.
It never places orders and never modifies trading logic — it only observes.

Checks performed
----------------
1. The whole WATCHLIST is predicted and backtested without an uncaught crash.
2. Safety invariant: any symbol with no trained ML model (neither RF nor LSTM)
   must resolve to action == "HOLD" (technical-only trading stays disabled).
3. Reports are written to REPORTS_DIR (metrics JSON + trades CSV).
4. A verification summary JSON is emitted.

Usage
-----
    python verify_watchlist.py                       # predict + full backtest
    python verify_watchlist.py --skip-backtest       # faster, predict only
    python verify_watchlist.py --train-min 252 --step 21
    python verify_watchlist.py --out reports/verification_summary.json

Exit codes
----------
    0  all checks passed
    1  a safety invariant was violated, or verification failed
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List

import config
from logging_setup import setup_logging

logger = logging.getLogger("verify")


def _model_inventory(symbols: List[str]) -> Dict[str, List[str]]:
    """Which symbols currently have an RF model, an LSTM model, or neither."""
    from ai_engine import StockRFEngine
    from lstm_engine import StockLSTMEngine

    rf = StockRFEngine()
    rf.load()
    lstm = StockLSTMEngine()
    lstm.load()

    rf_syms = set(rf.models.keys())
    lstm_syms = set(lstm.models.keys())
    ml_missing = [s for s in symbols if s not in rf_syms and s not in lstm_syms]
    return {
        "rf": sorted(rf_syms & set(symbols)),
        "lstm": sorted(lstm_syms & set(symbols)),
        "ml_missing": ml_missing,
    }


def _verify_predict(symbols: List[str], ml_missing: List[str]) -> dict:
    """Predict the whole watchlist and check the model-missing → HOLD invariant."""
    from predictor import Predictor

    signals = Predictor().predict_all(symbols=symbols)
    by_symbol = {s.symbol: s for s in signals}

    actions = Counter(s.action for s in signals)
    forced_hold = [s.symbol for s in signals if "forced HOLD" in s.reason]
    no_signal = [s for s in symbols if s not in by_symbol]  # e.g. data fetch failed

    violations: List[dict] = []

    # (a) symbols with no ML model on disk must be HOLD when a signal exists.
    for sym in ml_missing:
        s = by_symbol.get(sym)
        if s is not None and s.action != "HOLD":
            violations.append({
                "symbol": sym,
                "action": s.action,
                "reason": s.reason,
                "rule": "ml_missing_must_hold",
            })

    # (b) any signal explicitly flagged as forced HOLD must actually be HOLD.
    for s in signals:
        if "forced HOLD" in s.reason and s.action != "HOLD":
            violations.append({
                "symbol": s.symbol,
                "action": s.action,
                "reason": s.reason,
                "rule": "forced_hold_must_hold",
            })

    return {
        "n_signals": len(signals),
        "actions": {a: int(actions.get(a, 0)) for a in ("BUY", "SELL", "HOLD")},
        "forced_hold_symbols": forced_hold,
        "no_signal_symbols": no_signal,
        "safety_violations": violations,
    }


def _verify_backtest(symbols: List[str], train_min: int, step: int) -> dict:
    """Run the walk-forward backtest over all symbols and confirm reports exist."""
    from backtest import run_backtest

    results = run_backtest(
        symbols=symbols, train_min=train_min, step=step,
        verbose=False, include_lstm=False,
    )
    metrics_path = config.REPORTS_DIR / "backtest_metrics.json"
    trades_path = config.REPORTS_DIR / "backtest_trades.csv"
    return {
        "ran": True,
        "model": results.get("backtest_model"),
        "symbols_tested": int(results.get("symbols_tested", 0)),
        "total_trades": int(results.get("total_trades", 0)),
        "min_hold_bars": int(results.get("min_hold_bars", config.MIN_HOLD_BARS)),
        "reports": {
            "metrics_path": str(metrics_path),
            "trades_path": str(trades_path),
            "metrics_exists": metrics_path.exists(),
            "trades_exists": trades_path.exists(),
        },
    }


def run_verification(
    symbols: List[str],
    train_min: int,
    step: int,
    skip_backtest: bool,
) -> dict:
    summary: dict = {
        "watchlist": symbols,
        "config": {
            "MIN_HOLD_BARS": config.MIN_HOLD_BARS,
            "ML_HORIZON": config.ML_HORIZON,
            "MIN_ML_MODELS_FOR_SIGNAL": config.MIN_ML_MODELS_FOR_SIGNAL,
            "ALLOW_SHORT": config.ALLOW_SHORT,
        },
        "models": {},
        "predict": {},
        "backtest": {},
        "errors": [],
        "ok": False,
    }

    try:
        summary["models"] = _model_inventory(symbols)
    except Exception as exc:  # never crash — record and continue
        logger.exception("Model inventory failed")
        summary["errors"].append(f"model_inventory: {exc}")
        summary["models"] = {"rf": [], "lstm": [], "ml_missing": list(symbols)}

    ml_missing = summary["models"].get("ml_missing", [])

    try:
        summary["predict"] = _verify_predict(symbols, ml_missing)
    except Exception as exc:
        logger.exception("Predict verification failed")
        summary["errors"].append(f"predict: {exc}")

    if skip_backtest:
        summary["backtest"] = {"ran": False, "skipped": True}
    else:
        try:
            summary["backtest"] = _verify_backtest(symbols, train_min, step)
        except Exception as exc:
            logger.exception("Backtest verification failed")
            summary["errors"].append(f"backtest: {exc}")
            summary["backtest"] = {"ran": False, "skipped": False}

    # ── Pass/fail decision ──
    violations = summary.get("predict", {}).get("safety_violations", [])
    backtest_ok = skip_backtest or (
        summary["backtest"].get("ran", False)
        and summary["backtest"].get("reports", {}).get("metrics_exists", False)
    )
    summary["ok"] = (not violations) and (not summary["errors"]) and backtest_ok
    return summary


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="WATCHLIST production verification")
    parser.add_argument("--train-min", type=int, default=252)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--skip-backtest", action="store_true",
                        help="Predict-only verification (faster)")
    parser.add_argument("--out", default=str(config.REPORTS_DIR / "verification_summary.json"))
    parser.add_argument("--symbols", nargs="*", default=None,
                        help="Override WATCHLIST (default: config.WATCHLIST)")
    args = parser.parse_args()

    symbols = args.symbols or list(config.WATCHLIST)
    summary = run_verification(symbols, args.train_min, args.step, args.skip_backtest)

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ── Human-readable summary ──
    pred = summary.get("predict", {})
    bt = summary.get("backtest", {})
    print("\n══ WATCHLIST Verification ══")
    print(f"  Symbols           : {len(symbols)}")
    print(f"  ML models present : RF={len(summary['models'].get('rf', []))} "
          f"LSTM={len(summary['models'].get('lstm', []))}")
    print(f"  ML missing        : {summary['models'].get('ml_missing', [])}")
    if pred:
        print(f"  Signals           : {pred.get('n_signals', 0)}  {pred.get('actions', {})}")
        print(f"  Forced HOLD       : {pred.get('forced_hold_symbols', [])}")
        print(f"  No-signal symbols : {pred.get('no_signal_symbols', [])}")
        print(f"  Safety violations : {len(pred.get('safety_violations', []))}")
    if not args.skip_backtest:
        print(f"  Backtest          : tested={bt.get('symbols_tested', 0)} "
              f"orders={bt.get('total_trades', 0)} reports="
              f"{bt.get('reports', {}).get('metrics_exists', False)}")
    if summary["errors"]:
        print(f"  Errors            : {summary['errors']}")
    print(f"\n  RESULT: {'✅ OK' if summary['ok'] else '❌ FAILED'}")
    print(f"  Summary → {out_path}")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
