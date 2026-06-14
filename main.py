"""
main.py â€” Stock Prediction Engine entry point.

Commands:
  python main.py train                    # Train RF + LSTM models
  python main.py predict                  # Print today's signals
  python main.py backtest                 # Walk-forward RF+Technical backtest
  python main.py backtest --include-lstm  # Slower RF+LSTM+Technical backtest
  python main.py paper                    # Predict + execute on IBKR paper
  python main.py paper --dry-run          # Predict only, no orders placed

Full-market hot scanner (discovery only â€” see README "Full Market Hot Scanner"):
  python main.py scan-hot                 # Rank watchlist hot candidates
  python main.py scan-hot --full-market   # Rank broad-market hot candidates
  python main.py predict-hot              # Scan, then predict candidates
  python main.py train-hot                # Scan, then train candidates
  python main.py paper-hot                # Scan + predict, DRY-RUN (no orders)
  python main.py paper-hot --full-market --execute  # Place PAPER orders only

Threshold tuning (read-only â€” never connects to IBKR, never places orders):
  python main.py threshold-report                 # Sweep BUY thresholds on WATCHLIST
  python main.py threshold-report --full-market    # Sweep on broad-market candidates
  python main.py threshold-report --full-market --max-symbols 100 --top-n 30

Model readiness (no IBKR connection, no orders â€” diagnostics + maintenance only):
  python main.py model-doctor                      # Coverage/freshness for report candidates
  python main.py model-refresh --from-report --top-n 30  # Train/update + MERGE report candidates

Guided Paper Trading Coach (beginner-friendly; preview-first, never auto-buys):
  python main.py coach                    # Lessons on WATCHLIST (no IBKR, no orders)
  python main.py coach-hot                # Lessons on reports/hot_candidates.csv (no IBKR)
  python main.py paper-coach              # Connect paper, PREVIEW one trade (no orders)
  python main.py paper-coach --confirm    # Refuses: chart check required
  python main.py paper-coach --confirm --chart-checked  # Place at most ONE paper order

Guided Daily Trading Coach (full-market scan; previews + multi-trade PAPER practice):
  python main.py daily-coach              # Full-market scan, preview top 3 (no orders)
  python main.py daily-coach --max-trades 3            # Preview up to 3 (no orders)
  python main.py daily-coach --confirm --max-trades 3  # Refuses: chart check required
  python main.py daily-coach --confirm --chart-checked --max-trades 3  # Up to 3 PAPER trades
  (PAPER ONLY. Live trading is hard-blocked. See README "Guided Daily Trading Coach".)

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
import hot_scanner
from hot_scanner import scan_hot_stocks
import model_doctor
from trade_coach import (
    build_trade_lesson, build_trade_preview, print_trade_lesson, print_trade_preview,
    write_trade_note, select_coach_candidates,
    assert_paper_trading_only, LiveTradingDisabledError,
    assess_chart_status, evaluate_daily_candidate, print_daily_candidate,
    build_daily_coach_summary, print_daily_coach_summary, write_daily_coach_summary,
    build_ipo_watch_section,
)

# Candidate BUY thresholds swept by `threshold-report`. SELL_THRESHOLD is held
# fixed at its configured value; only the BUY cut moves across this grid.
THRESHOLD_GRID = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70]

setup_logging()
logger = logging.getLogger(__name__)


def cmd_train(_args) -> int:
    logger.info("â•â•â• Training RF models â•â•â•")
    rf_results = StockRFEngine().train(verbose=True)
    print("\nâœ… RF Training Results:")
    if not rf_results:
        print("  âŒ No RF models were trained. Check data source / labels / WATCHLIST.")
        return 4
    for symbol, m in rf_results.items():
        print(
            f"  {symbol:<6}  oob={m['oob_score']:.3f}  "
            f"acc={m['test_acc']:.3f}  f1={m['test_f1']:.3f}  "
            f"n={m['n_samples']}"
        )

    logger.info("â•â•â• Training LSTM models â•â•â•")
    lstm_results = StockLSTMEngine().train(verbose=True)
    print("\nâœ… LSTM Training Results:")
    if not lstm_results:
        print("  âŒ No LSTM models were trained. Check data source / labels / WATCHLIST.")
        return 5
    for symbol, m in lstm_results.items():
        print(
            f"  {symbol:<6}  best_val_loss={m['best_val_loss']:.4f}  "
            f"train_seq={m.get('n_train_seq', 0)}  val_seq={m.get('n_val_seq', 0)}"
        )

    print("\nðŸŽ¯ All models saved.")
    return 0


def cmd_predict(_args) -> int:
    signals = Predictor().predict_all()
    print(f"\n{'Symbol':<8} {'Action':<6} {'Conf':>6} {'Price':>10}  Reason")
    print("â”€" * 75)
    for s in signals:
        icon = "ðŸŸ¢" if s.action == "BUY" else "ðŸ”´" if s.action == "SELL" else "âšª"
        print(
            f"{icon} {s.symbol:<6}  {s.action:<6}  "
            f"{s.confidence:>5.2f}  ${s.price:>9.2f}  {s.reason}"
        )
    b = sum(1 for s in signals if s.action == "BUY")
    se = sum(1 for s in signals if s.action == "SELL")
    h = sum(1 for s in signals if s.action == "HOLD")
    print(f"\nSummary: ðŸŸ¢ BUY={b}  ðŸ”´ SELL={se}  âšª HOLD={h}")
    return 0


def cmd_backtest(args) -> int:
    logger.info(
        "â•â•â• Walk-forward backtest (train_min=%d step=%d include_lstm=%s) â•â•â•",
        args.train_min, args.step, args.include_lstm,
    )
    results = run_backtest(
        train_min=args.train_min,
        step=args.step,
        include_lstm=args.include_lstm,
    )
    if results.get("symbols_tested", 0) == 0:
        print("\nâš ï¸  Backtest produced no metrics. Check data/length.")
        return 0

    print("\nðŸ“Š Backtest Results:")
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
    print("â”€" * 70)
    for sym, m in results["per_symbol"].items():
        print(
            f"{sym:<8} {m['total_return']*100:>7.2f}%  {m['sharpe_ratio']:>7.2f}  "
            f"{m['win_rate_active_days']*100:>6.1f}%  {m['max_drawdown']*100:>7.2f}%  "
            f"{m['n_orders']:>6}  {m['n_active_days']:>6}"
        )
    print(f"\nðŸ’¾ Report â†’ {config.REPORTS_DIR / 'backtest_metrics.json'}")
    print(f"ðŸ’¾ Trades â†’ {config.REPORTS_DIR / 'backtest_trades.csv'}")
    return 0


def _ibkr_setup_asyncio() -> None:
    """Prepare the asyncio event loop for ib_insync on this platform.

    Mirrors the setup previously inlined in _place_paper_orders so the coach
    commands can reuse it without duplicating the Windows / Python 3.14 fixes.
    """
    import asyncio
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _connect_bridge():
    """Connect to the IBKR PAPER account via the existing IBKRBridge.

    Returns (bridge, exit_code). On success bridge is the connected IBKRBridge
    and exit_code is 0; on failure bridge is None and exit_code is non-zero.
    The paper-port lock and every other risk control live inside IBKRBridge and
    are never bypassed here. Used by the read-only `paper-coach` preview as well
    as order placement.
    """
    _ibkr_setup_asyncio()
    try:
        from ibkr_bridge import IBKRBridge
    except ImportError:
        print("ib_insync not installed. Run: pip install ib_insync")
        return None, 2

    bridge = IBKRBridge()
    if not bridge.connect():
        print(
            "Could not connect to IBKR TWS.\n"
            "   Ensure TWS/Gateway is running on port 7497 (paper).\n"
            "   Settings -> API -> Enable Socket Clients -> Port 7497"
        )
        return None, 1
    return bridge, 0


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
        print("âŒ ib_insync not installed. Run: pip install ib_insync")
        return 2

    bridge = IBKRBridge()
    if not bridge.connect():
        print(
            "âŒ Could not connect to IBKR TWS.\n"
            "   Ensure TWS/Gateway is running on port 7497 (paper).\n"
            "   Settings â†’ API â†’ Enable Socket Clients â†’ Port 7497"
        )
        return 1

    try:
        result = bridge.execute_all(signals)
        print(f"\nâœ… Orders accepted : {result['placed']}")
        print(f"   Skipped         : {result['skipped']}")
        print(f"   Total signals   : {result['total']}")
    finally:
        bridge.disconnect()
    return 0


def cmd_paper(args) -> int:
    signals = Predictor().predict_all()

    print(f"\n{'Symbol':<8} {'Action':<6} {'Conf':>6} {'Price':>10}")
    print("â”€" * 40)
    for s in signals:
        icon = "ðŸŸ¢" if s.action == "BUY" else "ðŸ”´" if s.action == "SELL" else "âšª"
        print(f"{icon} {s.symbol:<6}  {s.action:<6}  {s.confidence:>5.2f}  ${s.price:>9.2f}")

    if args.dry_run:
        print("\nâš ï¸  Dry-run mode â€” no orders placed.")
        return 0

    return _place_paper_orders(signals)


# â”€â”€ Full-market hot scanner commands â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _get_hot_symbols(args) -> list:
    """Run the hot scanner and return the ranked candidate symbols.

    Discovery/ranking only â€” fetches price data and writes the report, never
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
        print("\nâš ï¸  No hot candidates found.")
        return 0
    print(f"\nðŸ”¥ Top {len(hot_symbols)} hot candidates:")
    print("   " + ", ".join(hot_symbols))
    print("\nâ„¹ï¸  These are momentum/volume candidates only â€” NOT buy signals.")
    print("   Run 'predict-hot' to get BUY/HOLD/SELL from the model ensemble.")
    return 0


def _print_signals(signals) -> None:
    print(f"\n{'Symbol':<8} {'Action':<6} {'Conf':>6} {'Price':>10}  Reason")
    print("â”€" * 75)
    for s in signals:
        icon = "ðŸŸ¢" if s.action == "BUY" else "ðŸ”´" if s.action == "SELL" else "âšª"
        print(
            f"{icon} {s.symbol:<6}  {s.action:<6}  "
            f"{s.confidence:>5.2f}  ${s.price:>9.2f}  {s.reason}"
        )
    b = sum(1 for s in signals if s.action == "BUY")
    se = sum(1 for s in signals if s.action == "SELL")
    h = sum(1 for s in signals if s.action == "HOLD")
    print(f"\nSummary: ðŸŸ¢ BUY={b}  ðŸ”´ SELL={se}  âšª HOLD={h}")


def cmd_predict_hot(args) -> int:
    hot_symbols = _get_hot_symbols(args)
    if not hot_symbols:
        print("\nâš ï¸  No hot candidates to predict.")
        return 0
    signals = Predictor().predict_all(symbols=hot_symbols)
    _print_signals(signals)
    return 0


def cmd_train_hot(args) -> int:
    hot_symbols = _get_hot_symbols(args)
    if not hot_symbols:
        print("\nâš ï¸  No hot candidates to train.")
        return 0

    logger.info("â•â•â• Training RF models on %d hot candidates â•â•â•", len(hot_symbols))
    rf_results = StockRFEngine().train(symbols=hot_symbols, verbose=True)
    print("\nâœ… RF Training Results:")
    if not rf_results:
        print("  âŒ No RF models were trained for hot candidates.")
    else:
        for symbol, m in rf_results.items():
            print(
                f"  {symbol:<6}  oob={m['oob_score']:.3f}  "
                f"acc={m['test_acc']:.3f}  f1={m['test_f1']:.3f}  n={m['n_samples']}"
            )

    logger.info("â•â•â• Training LSTM models on %d hot candidates â•â•â•", len(hot_symbols))
    lstm_results = StockLSTMEngine().train(symbols=hot_symbols, verbose=True)
    print("\nâœ… LSTM Training Results:")
    if not lstm_results:
        print("  âŒ No LSTM models were trained for hot candidates.")
    else:
        for symbol, m in lstm_results.items():
            print(
                f"  {symbol:<6}  best_val_loss={m['best_val_loss']:.4f}  "
                f"train_seq={m.get('n_train_seq', 0)}  val_seq={m.get('n_val_seq', 0)}"
            )

    print("\nðŸŽ¯ Hot-candidate training done.")
    return 0


def cmd_paper_hot(args) -> int:
    hot_symbols = _get_hot_symbols(args)
    if not hot_symbols:
        print("\nâš ï¸  No hot candidates â€” nothing to predict or place.")
        return 0

    signals = Predictor().predict_all(symbols=hot_symbols)
    _print_signals(signals)

    if not bool(getattr(args, "execute", False)):
        print("\nâš ï¸  Dry-run mode (default) â€” no orders placed.")
        print("   Pass --execute to place PAPER orders through the existing IBKRBridge.")
        return 0

    buys = [s for s in signals if s.action in ("BUY", "SELL")]
    if not buys:
        print("\nâš ï¸  No actionable BUY/SELL signals â€” no orders placed.")
        return 0

    print("\nðŸŸ¢ --execute set: placing PAPER orders (existing risk controls apply)â€¦")
    return _place_paper_orders(signals)


# â”€â”€ Threshold analysis (read-only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _classify_at(confidence: float, buy_threshold: float) -> str:
    """Re-derive BUY/HOLD/SELL for a candidate BUY threshold.

    Mirrors predictor.action_from_confidence but with a swept BUY cut. The live
    config.SELL_THRESHOLD is intentionally left unchanged â€” only the BUY side moves.
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
            print("\nâš ï¸  No full-market candidates found â€” nothing to analyze.")
            return 0
    else:
        symbols = None  # Predictor defaults to config.WATCHLIST

    signals = Predictor().predict_all(symbols=symbols)
    if not signals:
        print("\nâš ï¸  No signals produced â€” nothing to analyze.")
        return 0

    # predict_all already returns signals sorted by confidence (desc), so the
    # first BUYs at any threshold are the strongest candidates.
    print(
        f"\nThreshold analysis on {len(signals)} symbol(s) "
        f"(SELL_THRESHOLD={config.SELL_THRESHOLD:.2f}, held fixed)"
    )
    print(f"Current configured BUY_THRESHOLD = {config.BUY_THRESHOLD:.2f}  (marked '*' below)")
    print(f"\n{'BuyThr':>7} {'BUY':>5} {'HOLD':>5} {'SELL':>5}  Top BUY candidates")
    print("â”€" * 78)
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

    # â”€â”€ Write per-symbol CSV (confidence, scores, reason, action per threshold) â”€â”€
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

    print(f"\nðŸ’¾ Report â†’ {csv_path}")
    print(
        "\nâ„¹ï¸  Read-only analysis â€” config.py was NOT modified. To change the live\n"
        "   threshold, edit BUY_THRESHOLD in config.py yourself."
    )
    return 0


# â”€â”€ Guided Paper Trading Coach (read-mostly, beginner-friendly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _print_coach_intro() -> None:
    print("\n=== Guided Paper Trading Coach ===")
    print("A learning tool: it explains signals and previews PAPER trades.")
    print("It never auto-buys. A paper order needs BOTH --confirm AND --chart-checked.")


def _read_hot_candidate_symbols():
    """Read symbols from reports/hot_candidates.csv (first column).

    Returns a list of symbols, or None when the file does not exist yet.
    """
    path = config.HOT_CANDIDATES_FILE
    if not path.exists():
        return None
    symbols = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip().upper()
            if sym:
                symbols.append(sym)
    return symbols


def cmd_coach(_args) -> int:
    """Beginner lessons on the WATCHLIST. No IBKR connection, no orders."""
    _print_coach_intro()
    signals = Predictor().predict_all()  # defaults to config.WATCHLIST
    if not signals:
        print("\nNo signals produced â€” check trained models / data source.")
        return 0

    _print_signals(signals)
    for s in signals:
        print_trade_lesson(build_trade_lesson(s))

    candidates = select_coach_candidates(
        signals, max_n=int(getattr(config, "COACH_MAX_NEW_TRADES_PER_RUN", 1))
    )
    if not candidates:
        print("\nNo BUY candidate above the confidence floor today. HOLD is a safe, valid choice.")
    else:
        print(
            f"\nNotional preview (uses PAPER_CAPITAL=${config.PAPER_CAPITAL:,.0f}; "
            "run `paper-coach` for live paper-account numbers):"
        )
        for c in candidates:
            preview = build_trade_preview(
                c, cash=float(config.PAPER_CAPITAL), current_positions={}, open_orders=[]
            )
            print_trade_preview(preview)
            write_trade_note(build_trade_lesson(c), action_taken="preview_only", preview=preview)

    print("\nNo IBKR connection was made and no orders were placed.")
    print("Next step: python -X utf8 main.py paper-coach")
    return 0


def cmd_coach_hot(_args) -> int:
    """Beginner lessons on the latest hot_candidates.csv. No IBKR, no orders."""
    _print_coach_intro()
    symbols = _read_hot_candidate_symbols()
    if symbols is None:
        print(f"\nNo hot candidates file at {config.HOT_CANDIDATES_FILE}.")
        print("Run `python -X utf8 main.py scan-hot` (or `scan-hot --full-market`) first.")
        return 0
    if not symbols:
        print("\nHot candidates file is empty â€” re-run `scan-hot` first.")
        return 0

    signals = Predictor().predict_all(symbols=symbols)
    if not signals:
        print("\nNo signals produced for the hot candidates.")
        return 0

    candidates = select_coach_candidates(signals, max_n=3)
    if not candidates:
        print("\nNo BUY candidate among the hot list crossed the confidence floor.")
        print("Showing the strongest signals for study instead:")
        candidates = signals[:3]

    print(f"\nTop {len(candidates)} candidate(s) from the hot list:")
    for c in candidates:
        print_trade_lesson(build_trade_lesson(c))
        preview = build_trade_preview(
            c, cash=float(config.PAPER_CAPITAL), current_positions={}, open_orders=[]
        )
        print(
            f"\nNotional preview (PAPER_CAPITAL=${config.PAPER_CAPITAL:,.0f}; "
            "run `paper-coach` for live numbers):"
        )
        print_trade_preview(preview)
        write_trade_note(build_trade_lesson(c), action_taken="preview_only", preview=preview)

    print("\nNo IBKR connection was made and no orders were placed.")
    return 0


# â”€â”€ Model Doctor / Model Refresh (read / maintenance only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def cmd_model_doctor(args) -> int:
    """Inspect RF/LSTM coverage + freshness for the report candidates.

    Read-only: no IBKR connection, no orders. Writes reports/model_doctor_report.md.
    """
    return model_doctor.run_doctor(top_n=getattr(args, "top_n", None))


def cmd_model_refresh(args) -> int:
    """Train/update models for the exact report candidates, preserving existing
    models (merge + backup). No IBKR connection, no orders, no live trading.
    """
    if not bool(getattr(args, "from_report", False)):
        print(
            "model-refresh currently supports only --from-report "
            "(symbols from reports/hot_candidates.csv)."
        )
        print("Run: python -X utf8 main.py model-refresh --from-report --top-n 30")
        return 3
    top_n = int(getattr(args, "top_n", 30) or 30)
    return model_doctor.run_refresh(top_n=top_n)


def cmd_paper_coach(args) -> int:
    """Preview (and, only with --confirm --chart-checked, place) ONE paper trade.

    Default behavior is preview only. An order is placed only when ALL of these
    hold: --confirm AND --chart-checked are passed, the candidate is a BUY at or
    above BUY_THRESHOLD, no existing position in the symbol, and no working order
    in the symbol. Every existing IBKRBridge risk control still applies.
    """
    _print_coach_intro()
    confirm = bool(getattr(args, "confirm", False))
    chart_checked = bool(getattr(args, "chart_checked", False))

    signals = Predictor().predict_all()  # defaults to config.WATCHLIST
    if not signals:
        print("\nNo signals produced â€” nothing to preview.")
        return 0

    bridge, code = _connect_bridge()
    if bridge is None:
        return code

    try:
        cash = bridge.get_cash()
        positions = {
            p.contract.symbol.upper(): float(p.position)
            for p in bridge.ib.positions()
            if float(p.position) != 0.0
        }
        working = bridge.working_order_symbols()
        print(
            f"\nPaper account: cash=${cash:,.2f}  open_positions={len(positions)}  "
            f"working_orders={len(working)}"
        )

        # Best BUY candidate that is not already held or pending.
        candidates = select_coach_candidates(
            signals, max_n=int(getattr(config, "COACH_MAX_NEW_TRADES_PER_RUN", 1))
        )
        candidates = [
            c for c in candidates
            if c.symbol.upper() not in positions and c.symbol.upper() not in working
        ]
        if not candidates:
            print(
                "\nNo eligible BUY candidate (none above the floor, or all already held / pending)."
            )
            print("No order placed.")
            return 0

        candidate = candidates[0]
        lesson = build_trade_lesson(candidate, position_info=positions, open_order_info=working)
        preview = build_trade_preview(
            candidate, cash=cash, current_positions=positions, open_orders=list(working)
        )
        print_trade_lesson(lesson)
        print_trade_preview(preview)

        # â”€â”€ Preview only (no --confirm) â”€â”€
        if not confirm:
            write_trade_note(lesson, action_taken="preview_only", preview=preview)
            print(
                "\nNo order placed. To place a paper order, rerun with --confirm "
                "after checking chart."
            )
            return 0

        # â”€â”€ --confirm but no --chart-checked: refuse â”€â”€
        if not chart_checked:
            write_trade_note(
                lesson, action_taken="refused", preview=preview,
                order_result={"reason": "Chart check required (missing --chart-checked)."},
            )
            print("\nChart check required before paper execution.")
            return 0

        # â”€â”€ --confirm AND --chart-checked: validate, then place at most ONE order â”€â”€
        sym = candidate.symbol.upper()
        reasons = []
        if candidate.action.upper() != "BUY":
            reasons.append(f"action is {candidate.action}, not BUY")
        if float(candidate.confidence) < float(config.BUY_THRESHOLD):
            reasons.append(
                f"confidence {candidate.confidence:.2f} < BUY_THRESHOLD {config.BUY_THRESHOLD:.2f}"
            )
        if sym in positions:
            reasons.append("already holding this symbol")
        if bridge.has_working_order(sym):
            reasons.append("a working order already exists for this symbol")
        if not preview.get("tradeable"):
            reasons.append(preview.get("skip_reason") or "preview not tradeable")

        if reasons:
            write_trade_note(
                lesson, action_taken="refused", preview=preview,
                order_result={"reason": "; ".join(reasons)},
            )
            print("\nRefused to place order:")
            for r in reasons:
                print(f"  - {r}")
            return 0

        print(
            f"\nPlacing at most 1 PAPER order for {sym} through the existing "
            "IBKRBridge (all risk controls apply)..."
        )
        accepted = bridge.execute_signal(candidate)
        order_result = {"symbol": sym, "accepted": bool(accepted)}
        if accepted:
            print(f"PAPER order accepted for {sym}. Existing risk controls were applied.")
            write_trade_note(
                lesson, action_taken="paper_order_placed", preview=preview,
                order_result=order_result,
            )
        else:
            print(
                f"Order for {sym} was NOT accepted (risk guard, duplicate, or pricing). See logs."
            )
            write_trade_note(
                lesson, action_taken="refused", preview=preview,
                order_result={
                    "reason": "IBKRBridge declined the order (risk guard / pricing).",
                    **order_result,
                },
            )
        return 0
    finally:
        bridge.disconnect()


# â”€â”€ Guided Daily Trading Coach (full-market scan, multi-trade PAPER practice) â”€
def _quiet_daily_coach_logs(verbose: bool):
    """Raise the console log level of noisy modules unless --verbose was passed.

    Suppresses INFO chatter from predictor/data_manager/ai_engine/ibkr_bridge so
    a beginner sees a clean summary. Returns a list of (logger, previous_level)
    to restore later. When verbose is True this is a no-op (detailed logs shown).
    This only changes console verbosity; no trading logic or risk control changes.
    """
    _noisy = ("predictor", "data_manager", "ai_engine", "ibkr_bridge")
    if verbose:
        return []
    previous = []
    for name in _noisy:
        lg = logging.getLogger(name)
        previous.append((lg, lg.level))
        lg.setLevel(logging.WARNING)
    return previous


def _restore_daily_coach_logs(previous) -> None:
    for lg, level in previous:
        lg.setLevel(level)


def _resolve_daily_n_candidates(args) -> "tuple[int, int, bool]":
    """Resolve how many candidates daily-coach considers this run.

    Returns (n, hard_cap, clamped). ``n`` is the number of top candidates to
    preview / the upper bound on paper orders to place. It is taken from
    --max-trades when given (else COACH_DEFAULT_PREVIEW_CANDIDATES) and is
    ALWAYS clamped to [0, COACH_MAX_PAPER_TRADES_PER_RUN] so --max-trades can
    only lower the safe cap, never raise it.
    """
    hard_cap = int(getattr(config, "COACH_MAX_PAPER_TRADES_PER_RUN", 3))
    requested = getattr(args, "max_trades", None)
    if requested is None:
        requested = int(getattr(config, "COACH_DEFAULT_PREVIEW_CANDIDATES", 3))
    n = int(requested)
    clamped = False
    if n < 0:
        n = 0
    if n > hard_cap:
        n = hard_cap
        clamped = True
    return n, hard_cap, clamped


def _ipo_watch_fetch(symbol: str):
    """Read-only OHLCV fetch for the IPO / New Listing watch section.

    Deliberately bypasses data_manager.fetch_ohlcv's MIN_HISTORY_DAYS gate,
    because a brand-new listing legitimately has very little history and this
    section must still display it. Display-only: no caching into the trading
    pipeline, no training, no trading, no orders. Returns a DataFrame or None.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(str(symbol).upper().strip())
        df = ticker.history(period="max", interval="1d", auto_adjust=True)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _ipo_watch_price_lookup(symbol: str):
    """Read-only last-price lookup for the IPO / New Listing watch section.

    Watch-only display. Never trains, never trades, never places an order.
    Returns None if no data.
    """
    df = _ipo_watch_fetch(symbol)
    if df is None or df.empty:
        return None
    try:
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def _ipo_watch_history_lookup(symbol: str):
    """Read-only count of available daily history bars for a watch symbol.

    Used only to flag insufficient model history (vs MIN_HISTORY_DAYS) so the
    coach can mark a new listing as watch-only. Returns None if unavailable.
    """
    df = _ipo_watch_fetch(symbol)
    if df is None or df.empty:
        return None
    try:
        return int(len(df))
    except Exception:
        return None


def _print_and_write_ipo_watch_section(report_path: str) -> None:
    """Build, print, and append the watch-only IPO / New Listing section.

    Always shown. Read-only display: no training, no trading, no orders.
    """
    section = build_ipo_watch_section(
        price_lookup=_ipo_watch_price_lookup,
        history_lookup=_ipo_watch_history_lookup,
    )
    print()
    for ln in section:
        print(ln)
    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("\n## IPO / New Listing Watch\n\n```\n")
            f.write("\n".join(section))
            f.write("\n```\n")
    except Exception:
        pass


def _daily_coach_preview(candidates, n_candidates: int, confirm: bool, chart_checked: bool) -> int:
    """Preview-only path: build lessons/previews, print, write notes. No orders."""
    print(
        f"\nNotional preview (PAPER_CAPITAL=${config.PAPER_CAPITAL:,.0f}; "
        "no IBKR connection, no orders):"
    )
    evaluations = []
    for i, c in enumerate(candidates, 1):
        preview = build_trade_preview(
            c, cash=float(config.PAPER_CAPITAL), current_positions={}, open_orders=[]
        )
        status, detail = assess_chart_status(c)
        ev = evaluate_daily_candidate(c, preview, status, detail, positions={}, working=set())
        print_daily_candidate(ev, index=i)
        write_trade_note(build_trade_lesson(c), action_taken="preview_only", preview=preview)
        evaluations.append(ev)

    valid = [e for e in evaluations if e["accepted"]]
    skipped = [e for e in evaluations if not e["accepted"]]

    print("\nâ”€â”€ Execution summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print(f"  Requested max trades : {n_candidates}")
    print(f"  Valid candidates     : {len(valid)}")
    print("  Orders placed        : 0  (preview only)")
    if skipped:
        print("  Skipped candidates:")
        for e in skipped:
            print(f"    - {e['symbol']}: {e['skip_reason']}")
    else:
        print("  Skipped candidates   : none")

    if confirm and not chart_checked:
        print("\nâ›” Chart check required before paper execution.")
        print("   Re-run with BOTH --confirm AND --chart-checked to place PAPER trades.")
    else:
        print(
            "\n  To place up to N PAPER trades, re-run:\n"
            "   python -X utf8 main.py daily-coach --confirm --chart-checked --max-trades N"
        )
    print("\n  Reminder: This is paper trading practice.")
    print("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    return 0


def _daily_coach_execute(candidates, n_candidates: int) -> int:
    """Execution path (--confirm AND --chart-checked): place up to N PAPER trades.

    Connects to the IBKR PAPER account through the existing IBKRBridge and never
    bypasses it. In addition to the bridge's own guards, this loop enforces the
    run-level caps that the bridge's single-order path does not check on its own:
    --max-trades (n_candidates), MAX_OPEN_POSITIONS, and MAX_DAILY_TRADES.
    """
    import risk_state

    # Belt-and-suspenders: re-assert paper-only right before we connect.
    try:
        assert_paper_trading_only()
    except LiveTradingDisabledError as exc:
        print(f"\nâ›” {exc}")
        return 3

    bridge, code = _connect_bridge()
    if bridge is None:
        return code

    try:
        cash = bridge.get_cash()
        positions = {
            p.contract.symbol.upper(): float(p.position)
            for p in bridge.ib.positions()
            if float(p.position) != 0.0
        }
        working = bridge.working_order_symbols()
        print(
            f"\nPaper account: cash=${cash:,.2f}  open_positions={len(positions)}  "
            f"working_orders={len(working)}"
        )

        max_open = int(getattr(config, "MAX_OPEN_POSITIONS", 3))
        occupied = set(positions) | set(working)

        # Evaluate every candidate against the static per-symbol gates first.
        evaluations = []
        for c in candidates:
            preview = build_trade_preview(
                c, cash=cash, current_positions=positions, open_orders=list(working)
            )
            status, detail = assess_chart_status(c)
            ev = evaluate_daily_candidate(
                c, preview, status, detail, positions=positions, working=working
            )
            evaluations.append((c, preview, ev))

        valid_count = sum(1 for _, _, ev in evaluations if ev["accepted"])
        placed = 0
        orders: list = []
        skipped_log: list = []

        for i, (c, preview, ev) in enumerate(evaluations, 1):
            sym = ev["symbol"]
            final = ev
            do_place = False

            if not ev["accepted"]:
                pass  # static gate already produced a skip reason
            elif placed >= n_candidates:
                final = {**ev, "accepted": False,
                         "skip_reason": f"--max-trades {n_candidates} already satisfied ({placed} placed)"}
            elif len(occupied) >= max_open:
                final = {**ev, "accepted": False,
                         "skip_reason": f"MAX_OPEN_POSITIONS ({max_open}) reached"}
            elif not risk_state.can_open_more():
                final = {**ev, "accepted": False,
                         "skip_reason": f"MAX_DAILY_TRADES ({config.MAX_DAILY_TRADES}) reached"}
            else:
                do_place = True

            print_daily_candidate(final, index=i)

            if not do_place:
                skipped_log.append((sym, final["skip_reason"]))
                continue

            print(
                f"\nðŸŸ¢ Placing PAPER order for {sym} through the existing IBKRBridge "
                "(all risk controls apply)â€¦"
            )
            lesson = build_trade_lesson(c, position_info=positions, open_order_info=working)
            accepted = bridge.execute_signal(c)
            if accepted:
                placed += 1
                occupied.add(sym)
                orders.append(sym)
                print(f"âœ… PAPER order accepted for {sym}.")
                write_trade_note(
                    lesson, action_taken="paper_order_placed", preview=preview,
                    order_result={"symbol": sym, "accepted": True},
                )
            else:
                reason = "IBKRBridge declined the order (risk guard / duplicate / pricing)"
                skipped_log.append((sym, reason))
                print(f"âš ï¸  Order for {sym} was NOT accepted ({reason}). See logs.")
                write_trade_note(
                    lesson, action_taken="refused", preview=preview,
                    order_result={"reason": reason, "symbol": sym},
                )

        print("\nâ”€â”€ Execution summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        print(f"  Requested max trades : {n_candidates}")
        print(f"  Valid candidates     : {valid_count}")
        print(
            f"  Orders placed        : {placed}"
            + (f"  ({', '.join(orders)})" if orders else "")
        )
        if skipped_log:
            print("  Skipped candidates:")
            for sym, reason in skipped_log:
                print(f"    - {sym}: {reason}")
        else:
            print("  Skipped candidates   : none")
        print("\n  Reminder: This is paper trading practice.")
        print("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
        return 0
    finally:
        bridge.disconnect()


def cmd_daily_coach(args) -> int:
    """Full-market scan, preview top candidates, optionally place up to N PAPER trades.

    Behavior:
      * default / --max-trades N           : preview only, no orders.
      * --confirm without --chart-checked  : refuses (chart check required).
      * --confirm AND --chart-checked      : may place up to N PAPER trades,
        respecting --max-trades, MAX_OPEN_POSITIONS, MAX_DAILY_TRADES, and every
        per-symbol gate. PAPER ONLY -- live trading is hard-blocked.

    By default, INFO log spam from predictor/data_manager/ai_engine/ibkr_bridge is
    suppressed so a beginner sees a clean summary. Pass --verbose to show logs.
    """
    verbose = bool(getattr(args, "verbose", False))
    _quieted = _quiet_daily_coach_logs(verbose)
    try:
        return _run_daily_coach(args, verbose)
    finally:
        _restore_daily_coach_logs(_quieted)


def _run_daily_coach(args, verbose: bool) -> int:
    print("\n=== Guided Daily Trading Coach (PAPER ONLY) ===")
    if bool(getattr(args, "from_report", False)):
        print("Uses the saved hot-candidate report, previews the best BUY candidates, and - only with")
    else:
        print("Scans the full market, previews the best BUY candidates, and - only with")
    print(
        "BOTH --confirm AND --chart-checked - may place up to "
        f"{int(getattr(config, 'COACH_MAX_PAPER_TRADES_PER_RUN', 3))} PAPER trades. "
        "Live trading is disabled."
    )

    # -- Hard live-trading block (refuse before doing anything else) --
    try:
        assert_paper_trading_only()
    except LiveTradingDisabledError as exc:
        print(f"\n[BLOCKED] {exc}")
        return 3

    confirm = bool(getattr(args, "confirm", False))
    chart_checked = bool(getattr(args, "chart_checked", False))
    n_candidates, hard_cap, clamped = _resolve_daily_n_candidates(args)
    if clamped:
        print(
            f"\n[WARN] --max-trades exceeds the safe cap "
            f"COACH_MAX_PAPER_TRADES_PER_RUN={hard_cap}; clamping to {hard_cap}."
        )

    min_conf = float(getattr(config, "COACH_MIN_CONFIDENCE_FOR_CANDIDATE", 0.65))

    from_report = bool(getattr(args, "from_report", False))
    if from_report:
        # Use the already-trained saved report; never re-scan, never overwrite it.
        # This keeps daily-coach aligned with the symbols model-refresh trained.
        hot_symbols = _read_hot_candidate_symbols()
        if not hot_symbols:
            print("No hot candidate report found. Run scan-hot first.")
            return 0
        print("Using saved hot candidates from reports/hot_candidates.csv")
        top_n = getattr(args, "top_n", None)
        if top_n is not None and top_n > 0:
            hot_symbols = hot_symbols[:top_n]
        scanned = len(hot_symbols)
    else:
        # -- Full-market scan (discovery only; never places orders) --
        hot_symbols = scan_hot_stocks(
            full_market=True,
            max_symbols=getattr(args, "max_symbols", None),
            top_n=getattr(args, "top_n", None),
            write_report=True,
            selection_mode=getattr(args, "selection_mode", None),
        )
        scanned = int(hot_scanner.LAST_SCAN_STATS.get("scanned", 0))

    if not hot_symbols:
        lines = build_daily_coach_summary(scanned, [], [], min_conf)
        print_daily_coach_summary(lines)
        report = write_daily_coach_summary(lines)
        print(f"\nReport written to {report}")
        _print_and_write_ipo_watch_section(report)
        return 0

    signals = Predictor().predict_all(symbols=hot_symbols)
    if not signals:
        lines = build_daily_coach_summary(scanned, [], [], min_conf)
        print_daily_coach_summary(lines)
        report = write_daily_coach_summary(lines)
        print(f"\nReport written to {report}")
        _print_and_write_ipo_watch_section(report)
        return 0

    candidates = select_coach_candidates(signals, max_n=n_candidates)
    eligible_buys = [
        s for s in signals
        if s.action == "BUY" and float(s.confidence) >= min_conf
    ]

    # -- Model readiness note (clearer message only; no trading change) --
    # Candidates with neither RF nor LSTM are forced to HOLD by the predictor.
    # If that's the majority, surface a clear next step instead of silently
    # returning no BUY candidates. This does not force or skip any trade.
    n_missing = sum(1 for s in signals if "models missing" in s.reason)
    if signals and n_missing >= len(signals) / 2:
        print(
            f"\n[MODEL] {n_missing}/{len(signals)} candidates have no ML model "
            "(forced HOLD)."
        )
        print(
            "Model coverage is low. Run model-refresh --from-report before "
            "expecting reliable BUY signals."
        )

    # -- Always print/write the clean beginner-friendly summary first --
    summary_lines = build_daily_coach_summary(
        scanned, signals, eligible_buys, min_conf, top_n=5
    )
    print_daily_coach_summary(summary_lines)
    report = write_daily_coach_summary(summary_lines)
    print(f"\nReport written to {report}")

    # -- Always show the watch-only IPO / New Listing section --
    # Read-only display: never trains, trades, or places orders. A watch symbol
    # only becomes an official BUY candidate if it independently passes the
    # normal model/risk rules in the candidate flow above.
    _print_and_write_ipo_watch_section(report)

    if not candidates:
        # No actionable BUY candidate; the summary above already explains why.
        return 0

    # Detailed per-candidate previews / execution path (unchanged trading logic).
    if not (confirm and chart_checked):
        return _daily_coach_preview(candidates, n_candidates, confirm, chart_checked)
    return _daily_coach_execute(candidates, n_candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stock Prediction Engine â€” IBKR Paper Trading")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train", help="Train RF + LSTM models")
    sub.add_parser("predict", help="Show today's signals")

    bt = sub.add_parser("backtest", help="Walk-forward backtest")
    bt.add_argument("--train-min", type=int, default=252)
    bt.add_argument("--step", type=int, default=21)
    bt.add_argument("--include-lstm", action="store_true", help="Slow full-ensemble RF+LSTM+Technical backtest")

    pp = sub.add_parser("paper", help="Execute signals on IBKR paper")
    pp.add_argument("--dry-run", action="store_true", help="Preview signals without placing orders")

    # â”€â”€ Full-market hot scanner subcommands â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

    # â”€â”€ Guided Paper Trading Coach subcommands â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    sub.add_parser("coach", help="Beginner lessons on WATCHLIST signals (no IBKR, no orders)")
    sub.add_parser("coach-hot", help="Beginner lessons on hot_candidates.csv (no IBKR, no orders)")
    pc = sub.add_parser(
        "paper-coach",
        help="Paper preview; place ONE order only with --confirm AND --chart-checked",
    )
    pc.add_argument(
        "--confirm", action="store_true",
        help="Required to place a paper order (still also needs --chart-checked)",
    )
    pc.add_argument(
        "--chart-checked", action="store_true",
        help="Acknowledge you manually checked the chart before any execution",
    )

    # â”€â”€ Model Doctor / Model Refresh (read / maintenance only; no IBKR) â”€â”€â”€â”€â”€â”€
    md = sub.add_parser(
        "model-doctor",
        help="Inspect RF/LSTM coverage + freshness for report candidates (no IBKR, no orders)",
    )
    md.add_argument(
        "--top-n", type=int, default=None,
        help="Only inspect the first N candidates from hot_candidates.csv (default: all)",
    )
    mr = sub.add_parser(
        "model-refresh",
        help="Train/update models for report candidates, preserving existing models (no IBKR, no orders)",
    )
    mr.add_argument(
        "--from-report", action="store_true",
        help="Load symbols from reports/hot_candidates.csv (required source)",
    )
    mr.add_argument(
        "--top-n", type=int, default=30,
        help="Train/update the first N report candidates (default: 30)",
    )

    # â”€â”€ Guided Daily Trading Coach (full-market scan, multi-trade PAPER) â”€â”€â”€â”€
    dc = sub.add_parser(
        "daily-coach",
        help=(
            "Full-market scan + preview; place up to COACH_MAX_PAPER_TRADES_PER_RUN "
            "PAPER trades only with --confirm AND --chart-checked"
        ),
    )
    dc.add_argument(
        "--confirm", action="store_true",
        help="Required to place PAPER orders (still also needs --chart-checked)",
    )
    dc.add_argument(
        "--chart-checked", action="store_true",
        help="Acknowledge you manually checked the chart before any execution",
    )
    dc.add_argument(
        "--max-trades", type=int, default=None,
        help=(
            "How many top candidates to preview / max PAPER trades to place. "
            "Clamped to COACH_MAX_PAPER_TRADES_PER_RUN; cannot raise the cap."
        ),
    )
    dc.add_argument(
        "--from-report", action="store_true",
        help="Use the saved reports/hot_candidates.csv symbols instead of running a "
             "fresh full-market scan. Does not overwrite the report (keeps trained "
             "coverage aligned with model-refresh).",
    )
    dc.add_argument(
        "--verbose", action="store_true",
        help="Show detailed INFO logs (predictor/data_manager/ai_engine/ibkr_bridge). "
             "By default these are suppressed for a clean beginner-friendly summary.",
    )
    # Share the full-market selection knobs so power users can tune the scan.
    dc.add_argument("--max-symbols", type=int, default=None,
                    help="Override FULL_MARKET_MAX_SYMBOLS_TO_CHECK for this run")
    dc.add_argument("--top-n", type=int, default=None,
                    help="Override HOT_SCAN_TOP_N for this run")
    dc.add_argument(
        "--selection-mode", choices=["hybrid", "random", "rotation", "alphabetical"],
        default=None,
        help="Symbol-selection strategy for the full-market scan (default: config).",
    )

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
    if args.command == "coach":
        return cmd_coach(args)
    if args.command == "coach-hot":
        return cmd_coach_hot(args)
    if args.command == "paper-coach":
        return cmd_paper_coach(args)
    if args.command == "model-doctor":
        return cmd_model_doctor(args)
    if args.command == "model-refresh":
        return cmd_model_refresh(args)
    if args.command == "daily-coach":
        return cmd_daily_coach(args)

    parser.print_help()
    return 3


if __name__ == "__main__":
    sys.exit(main())

