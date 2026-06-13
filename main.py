"""
main.py — Stock Prediction Engine entry point.

Commands:
  python main.py train                    # Train RF + LSTM models
  python main.py predict                  # Print today's signals
  python main.py backtest                 # Walk-forward RF+Technical backtest
  python main.py backtest --include-lstm  # Slower RF+LSTM+Technical backtest
  python main.py paper                    # Predict + execute on IBKR paper
  python main.py paper --dry-run          # Predict only, no orders placed

Full-market hot scanner (discovery only — see README "Full Market Hot Scanner"):
  python main.py scan-hot                 # Rank watchlist hot candidates
  python main.py scan-hot --full-market   # Rank broad-market hot candidates
  python main.py predict-hot              # Scan, then predict candidates
  python main.py train-hot                # Scan, then train candidates
  python main.py paper-hot                # Scan + predict, DRY-RUN (no orders)
  python main.py paper-hot --full-market --execute  # Place PAPER orders only

Threshold tuning (read-only — never connects to IBKR, never places orders):
  python main.py threshold-report                 # Sweep BUY thresholds on WATCHLIST
  python main.py threshold-report --full-market    # Sweep on broad-market candidates
  python main.py threshold-report --full-market --max-symbols 100 --top-n 30

Exit codes:
  0  success
  1  IBKR connection failure
  2  ib_insync not installed
  3  unknown command / argument error
  4  RF training produced no models
  5  LSTM training produced no models
"""
import argparse
import csv
import logging
import sys

import config
from logging_setup import setup_logging
from ai_engine import StockRFEngine
from lstm_engine import StockLSTMEngine
from predictor import Predictor
from backtest import run_backtest
from hot_scanner import scan_hot_stocks

# Candidate BUY thresholds swept by `threshold-report`. SELL_THRESHOLD is held
# fixed at its configured value; only the BUY cut moves across this grid.
THRESHOLD_GRID = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]

setup_logging()
logger = logging.getLogger(__name__)


def cmd_train(_args) -> int:
    logger.info("═══ Training RF models ═══")
    rf_results = StockRFEngine().train(verbose=True)
    print("\n✅ RF Training Results:")
    if not rf_results:
        print("  ❌ No RF models were trained. Check data source / labels / WATCHLIST.")
        return 4
    for symbol, m in rf_results.items():
        print(
            f"  {symbol:<6}  oob={m['oob_score']:.3f}  "
            f"acc={m['test_acc']:.3f}  f1={m['test_f1']:.3f}  "
            f"n={m['n_samples']}"
        )

    logger.info("═══ Training LSTM models ═══")
    lstm_results = StockLSTMEngine().train(verbose=True)
    print("\n✅ LSTM Training Results:")
    if not lstm_results:
        print("  ❌ No LSTM models were trained. Check data source / labels / WATCHLIST.")
        return 5
    for symbol, m in lstm_results.items():
        print(
            f"  {symbol:<6}  best_val_loss={m['best_val_loss']:.4f}  "
            f"train_seq={m.get('n_train_seq', 0)}  val_seq={m.get('n_val_seq', 0)}"
        )

    print("\n🎯 All models saved.")
    return 0


def cmd_predict(_args) -> int:
    signals = Predictor().predict_all()
    print(f"\n{'Symbol':<8} {'Action':<6} {'Conf':>6} {'Price':>10}  Reason")
    print("─" * 75)
    for s in signals:
        icon = "🟢" if s.action == "BUY" else "🔴" if s.action == "SELL" else "⚪"
        print(
            f"{icon} {s.symbol:<6}  {s.action:<6}  "
            f"{s.confidence:>5.2f}  ${s.price:>9.2f}  {s.reason}"
        )
    b = sum(1 for s in signals if s.action == "BUY")
    se = sum(1 for s in signals if s.action == "SELL")
    h = sum(1 for s in signals if s.action == "HOLD")
    print(f"\nSummary: 🟢 BUY={b}  🔴 SELL={se}  ⚪ HOLD={h}")
    return 0


def cmd_backtest(args) -> int:
    logger.info(
        "═══ Walk-forward backtest (train_min=%d step=%d include_lstm=%s) ═══",
        args.train_min, args.step, args.include_lstm,
    )
    results = run_backtest(
        train_min=args.train_min,
        step=args.step,
        include_lstm=args.include_lstm,
    )
    if results.get("symbols_tested", 0) == 0:
        print("\n⚠️  Backtest produced no metrics. Check data/length.")
        return 0

    print("\n📊 Backtest Results:")
    print(f"  Model          : {results['backtest_model']}")
    print(f"  Symbols tested : {results['symbols_tested']}")
    print(f"  Orders         : {results['total_trades']}")
    print(f"  Bars simulated : {results['total_bars']}")
    print(f"  Avg return     : {results['avg_total_return']*100:.2f}%")
    print(f"  Avg Sharpe     : {results['avg_sharpe']:.2f}")
    print(f"  Avg win rate   : {results['avg_win_rate']*100:.1f}%")
    print(f"  Avg max DD     : {results['avg_max_drawdown']*100:.2f}%")
    print(f"  Cost/order     : {results['cost_per_order']*100:.3f}%")

    print(
        f"\n{'Symbol':<8} {'Return':>8} {'Sharpe':>8} "
        f"{'Win%':>7} {'MaxDD':>8} {'Orders':>7} {'Active':>7}"
    )
    print("─" * 70)
    for sym, m in results["per_symbol"].items():
        print(
            f"{sym:<8} {m['total_return']*100:>7.2f}%  {m['sharpe_ratio']:>7.2f}  "
            f"{m['win_rate_active_days']*100:>6.1f}%  {m['max_drawdown']*100:>7.2f}%  "
            f"{m['n_orders']:>6}  {m['n_active_days']:>6}"
        )
    print(f"\n💾 Report → {config.REPORTS_DIR / 'backtest_metrics.json'}")
    print(f"💾 Trades → {config.REPORTS_DIR / 'backtest_trades.csv'}")
    return 0


def _place_paper_orders(signals) -> int:
    """Place signals on the IBKR PAPER account via the existing IBKRBridge.

    All existing risk controls (paper-port lock, long-only, position/trade caps,
    duplicate-order guard) live inside IBKRBridge and are intentionally NOT
    bypassed here. Returns a process exit code.
    """
    # Windows asyncio policy fix for ib_insync.
    import asyncio
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

    # Python 3.14+ no longer auto-creates an event loop on first access, but
    # eventkit (ib_insync dependency) calls get_event_loop() at import time.
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    try:
        from ibkr_bridge import IBKRBridge
    except ImportError:
        print("❌ ib_insync not installed. Run: pip install ib_insync")
        return 2

    bridge = IBKRBridge()
    if not bridge.connect():
        print(
            "❌ Could not connect to IBKR TWS.\n"
            "   Ensure TWS/Gateway is running on port 7497 (paper).\n"
            "   Settings → API → Enable Socket Clients → Port 7497"
        )
        return 1

    try:
        result = bridge.execute_all(signals)
        print(f"\n✅ Orders accepted : {result['placed']}")
        print(f"   Skipped         : {result['skipped']}")
        print(f"   Total signals   : {result['total']}")
    finally:
        bridge.disconnect()
    return 0


def cmd_paper(args) -> int:
    signals = Predictor().predict_all()

    print(f"\n{'Symbol':<8} {'Action':<6} {'Conf':>6} {'Price':>10}")
    print("─" * 40)
    for s in signals:
        icon = "🟢" if s.action == "BUY" else "🔴" if s.action == "SELL" else "⚪"
        print(f"{icon} {s.symbol:<6}  {s.action:<6}  {s.confidence:>5.2f}  ${s.price:>9.2f}")

    if args.dry_run:
        print("\n⚠️  Dry-run mode — no orders placed.")
        return 0

    return _place_paper_orders(signals)


# ── Full-market hot scanner commands ─────────────────────────────────────────
def _get_hot_symbols(args) -> list:
    """Run the hot scanner and return the ranked candidate symbols.

    Discovery/ranking only — fetches price data and writes the report, never
    places an order.
    """
    full_market = bool(getattr(args, "full_market", False))
    max_symbols = getattr(args, "max_symbols", None)
    top_n = getattr(args, "top_n", None)
    selection_mode = getattr(args, "selection_mode", None)
    return scan_hot_stocks(
        full_market=full_market,
        max_symbols=max_symbols,
        top_n=top_n,
        write_report=True,
        selection_mode=selection_mode,
    )


def cmd_scan_hot(args) -> int:
    hot_symbols = _get_hot_symbols(args)
    if not hot_symbols:
        print("\n⚠️  No hot candidates found.")
        return 0
    print(f"\n🔥 Top {len(hot_symbols)} hot candidates:")
    print("   " + ", ".join(hot_symbols))
    print("\nℹ️  These are momentum/volume candidates only — NOT buy signals.")
    print("   Run 'predict-hot' to get BUY/HOLD/SELL from the model ensemble.")
    return 0


def _print_signals(signals) -> None:
    print(f"\n{'Symbol':<8} {'Action':<6} {'Conf':>6} {'Price':>10}  Reason")
    print("─" * 75)
    for s in signals:
        icon = "🟢" if s.action == "BUY" else "🔴" if s.action == "SELL" else "⚪"
        print(
            f"{icon} {s.symbol:<6}  {s.action:<6}  "
            f"{s.confidence:>5.2f}  ${s.price:>9.2f}  {s.reason}"
        )
    b = sum(1 for s in signals if s.action == "BUY")
    se = sum(1 for s in signals if s.action == "SELL")
    h = sum(1 for s in signals if s.action == "HOLD")
    print(f"\nSummary: 🟢 BUY={b}  🔴 SELL={se}  ⚪ HOLD={h}")


def cmd_predict_hot(args) -> int:
    hot_symbols = _get_hot_symbols(args)
    if not hot_symbols:
        print("\n⚠️  No hot candidates to predict.")
        return 0
    signals = Predictor().predict_all(symbols=hot_symbols)
    _print_signals(signals)
    return 0


def cmd_train_hot(args) -> int:
    hot_symbols = _get_hot_symbols(args)
    if not hot_symbols:
        print("\n⚠️  No hot candidates to train.")
        return 0

    logger.info("═══ Training RF models on %d hot candidates ═══", len(hot_symbols))
    rf_results = StockRFEngine().train(symbols=hot_symbols, verbose=True)
    print("\n✅ RF Training Results:")
    if not rf_results:
        print("  ❌ No RF models were trained for hot candidates.")
    else:
        for symbol, m in rf_results.items():
            print(
                f"  {symbol:<6}  oob={m['oob_score']:.3f}  "
                f"acc={m['test_acc']:.3f}  f1={m['test_f1']:.3f}  n={m['n_samples']}"
            )

    logger.info("═══ Training LSTM models on %d hot candidates ═══", len(hot_symbols))
    lstm_results = StockLSTMEngine().train(symbols=hot_symbols, verbose=True)
    print("\n✅ LSTM Training Results:")
    if not lstm_results:
        print("  ❌ No LSTM models were trained for hot candidates.")
    else:
        for symbol, m in lstm_results.items():
            print(
                f"  {symbol:<6}  best_val_loss={m['best_val_loss']:.4f}  "
                f"train_seq={m.get('n_train_seq', 0)}  val_seq={m.get('n_val_seq', 0)}"
            )

    print("\n🎯 Hot-candidate training done.")
    return 0


def cmd_paper_hot(args) -> int:
    hot_symbols = _get_hot_symbols(args)
    if not hot_symbols:
        print("\n⚠️  No hot candidates — nothing to predict or place.")
        return 0

    signals = Predictor().predict_all(symbols=hot_symbols)
    _print_signals(signals)

    if not bool(getattr(args, "execute", False)):
        print("\n⚠️  Dry-run mode (default) — no orders placed.")
        print("   Pass --execute to place PAPER orders through the existing IBKRBridge.")
        return 0

    buys = [s for s in signals if s.action in ("BUY", "SELL")]
    if not buys:
        print("\n⚠️  No actionable BUY/SELL signals — no orders placed.")
        return 0

    print("\n🟢 --execute set: placing PAPER orders (existing risk controls apply)…")
    return _place_paper_orders(signals)


# ── Threshold analysis (read-only) ───────────────────────────────────────────
def _classify_at(confidence: float, buy_threshold: float) -> str:
    """Re-derive BUY/HOLD/SELL for a candidate BUY threshold.

    Mirrors predictor.action_from_confidence but with a swept BUY cut. The live
    config.SELL_THRESHOLD is intentionally left unchanged — only the BUY side moves.
    """
    if confidence >= buy_threshold:
        return "BUY"
    if confidence <= config.SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def cmd_threshold_report(args) -> int:
    """Score symbols once, then report BUY/HOLD/SELL counts across a grid of
    candidate BUY thresholds.

    Strictly read-only: it never connects to IBKR, never places orders, and never
    edits config.py. Its only side effect is writing reports/threshold_report.csv.
    """
    full_market = bool(getattr(args, "full_market", False))
    if full_market:
        symbols = scan_hot_stocks(
            full_market=True,
            max_symbols=getattr(args, "max_symbols", None),
            top_n=getattr(args, "top_n", None),
            write_report=False,  # discovery only; don't clobber hot_candidates.csv
            selection_mode=getattr(args, "selection_mode", None),
        )
        if not symbols:
            print("\n⚠️  No full-market candidates found — nothing to analyze.")
            return 0
    else:
        symbols = None  # Predictor defaults to config.WATCHLIST

    signals = Predictor().predict_all(symbols=symbols)
    if not signals:
        print("\n⚠️  No signals produced — nothing to analyze.")
        return 0

    # predict_all already returns signals sorted by confidence (desc), so the
    # first BUYs at any threshold are the strongest candidates.
    print(
        f"\nThreshold analysis on {len(signals)} symbol(s) "
        f"(SELL_THRESHOLD={config.SELL_THRESHOLD:.2f}, held fixed)"
    )
    print(f"Current configured BUY_THRESHOLD = {config.BUY_THRESHOLD:.2f}  (marked '*' below)")
    print(f"\n{'BuyThr':>7} {'BUY':>5} {'HOLD':>5} {'SELL':>5}  Top BUY candidates")
    print("─" * 78)
    for thr in THRESHOLD_GRID:
        actions = [(s, _classify_at(s.confidence, thr)) for s in signals]
        buys = [s for s, a in actions if a == "BUY"]
        holds = sum(1 for _, a in actions if a == "HOLD")
        sells = sum(1 for _, a in actions if a == "SELL")
        top = ", ".join(f"{s.symbol}({s.confidence:.2f})" for s in buys[:5])
        if len(buys) > 5:
            top += f", +{len(buys) - 5} more"
        marker = "*" if abs(thr - config.BUY_THRESHOLD) < 1e-9 else " "
        print(f"{thr:>6.2f}{marker} {len(buys):>4} {holds:>5} {sells:>5}  {top}")

    # ── Write per-symbol CSV (confidence, scores, reason, action per threshold) ──
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.REPORTS_DIR / "threshold_report.csv"
    thr_cols = [f"action@{thr:.2f}" for thr in THRESHOLD_GRID]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["symbol", "confidence", "rf_score", "lstm_score", "tech_score", "price"]
            + thr_cols + ["reason"]
        )
        for s in signals:
            writer.writerow(
                [
                    s.symbol,
                    f"{s.confidence:.4f}",
                    f"{s.rf_score:.4f}",
                    f"{s.lstm_score:.4f}",
                    f"{s.tech_score:.4f}",
                    f"{s.price:.2f}",
                ]
                + [_classify_at(s.confidence, thr) for thr in THRESHOLD_GRID]
                + [s.reason]
            )

    print(f"\n💾 Report → {csv_path}")
    print(
        "\nℹ️  Read-only analysis — config.py was NOT modified. To change the live\n"
        "   threshold, edit BUY_THRESHOLD in config.py yourself."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stock Prediction Engine — IBKR Paper Trading")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Train RF + LSTM models")
    sub.add_parser("predict", help="Show today's signals")

    bt = sub.add_parser("backtest", help="Walk-forward backtest")
    bt.add_argument("--train-min", type=int, default=252)
    bt.add_argument("--step", type=int, default=21)
    bt.add_argument("--include-lstm", action="store_true", help="Slow full-ensemble RF+LSTM+Technical backtest")

    pp = sub.add_parser("paper", help="Execute signals on IBKR paper")
    pp.add_argument("--dry-run", action="store_true", help="Preview signals without placing orders")

    # ── Full-market hot scanner subcommands ──────────────────────────────────
    def _add_hot_flags(p, with_execute: bool = False) -> None:
        p.add_argument(
            "--full-market", action="store_true",
            help="Discover candidates from the broad US market (Nasdaq Trader directory)",
        )
        p.add_argument(
            "--max-symbols", type=int, default=None,
            help="Override FULL_MARKET_MAX_SYMBOLS_TO_CHECK for this run",
        )
        p.add_argument(
            "--top-n", type=int, default=None,
            help="Override HOT_SCAN_TOP_N for this run",
        )
        p.add_argument(
            "--selection-mode", choices=["hybrid", "random", "rotation", "alphabetical"],
            default=None,
            help=(
                "How to pick which symbols to scan from the full universe "
                "(default: config.FULL_MARKET_SELECTION_MODE). Only affects --full-market."
            ),
        )
        if with_execute:
            p.add_argument(
                "--execute", action="store_true",
                help="Place PAPER orders via IBKRBridge (default is dry-run, no orders)",
            )

    _add_hot_flags(sub.add_parser("scan-hot", help="Rank hot candidates (no orders)"))
    _add_hot_flags(sub.add_parser("predict-hot", help="Scan + predict candidates (no orders)"))
    _add_hot_flags(sub.add_parser("train-hot", help="Scan + train models for candidates"))
    _add_hot_flags(sub.add_parser("paper-hot", help="Scan + predict; --execute for PAPER orders"), with_execute=True)

    # Read-only BUY-threshold sweep. Shares the hot-scanner flags so --full-market
    # / --max-symbols / --top-n select the symbol set, but never places orders.
    _add_hot_flags(sub.add_parser("threshold-report", help="Sweep BUY thresholds (read-only, no orders)"))

    args = parser.parse_args()
    if args.command == "train":
        return cmd_train(args)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "backtest":
        return cmd_backtest(args)
    if args.command == "paper":
        return cmd_paper(args)
    if args.command == "scan-hot":
        return cmd_scan_hot(args)
    if args.command == "predict-hot":
        return cmd_predict_hot(args)
    if args.command == "train-hot":
        return cmd_train_hot(args)
    if args.command == "paper-hot":
        return cmd_paper_hot(args)
    if args.command == "threshold-report":
        return cmd_threshold_report(args)

    parser.print_help()
    return 3


if __name__ == "__main__":
    sys.exit(main())
