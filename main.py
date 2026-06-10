"""
main.py — Stock Prediction Engine entry point.

Commands:
  python main.py train                    # Train RF + LSTM models
  python main.py predict                  # Print today's signals
  python main.py backtest                 # Walk-forward RF+Technical backtest
  python main.py backtest --include-lstm  # Slower RF+LSTM+Technical backtest
  python main.py paper                    # Predict + execute on IBKR paper
  python main.py paper --dry-run          # Predict only, no orders placed

Exit codes:
  0  success
  1  IBKR connection failure
  2  ib_insync not installed
  3  unknown command / argument error
  4  RF training produced no models
  5  LSTM training produced no models
"""
import argparse
import logging
import sys

import config
from logging_setup import setup_logging
from ai_engine import StockRFEngine
from lstm_engine import StockLSTMEngine
from predictor import Predictor
from backtest import run_backtest

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

    # Windows asyncio policy fix for ib_insync.
    if sys.platform == "win32":
        import asyncio
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass

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

    args = parser.parse_args()
    if args.command == "train":
        return cmd_train(args)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "backtest":
        return cmd_backtest(args)
    if args.command == "paper":
        return cmd_paper(args)

    parser.print_help()
    return 3


if __name__ == "__main__":
    sys.exit(main())
