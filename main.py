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

Live-trading readiness (read-only scorecard + emergency kill-switch):
  python main.py live-readiness           # Go-live scorecard (config + capability flags)
  python main.py live-readiness --connect # Also check PAPER account for unprotected positions
  python main.py panic-flatten            # DRY-RUN: show orders/positions it would flatten
  python main.py panic-flatten --confirm  # Cancel all orders + market-close all PAPER positions
  (See reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md. Still PAPER ONLY.)

Supervised scheduler (Path A; one-shot, market-hours gated; PAPER ONLY):
  python -X utf8 main.py run-scheduled    # Run `paper` only inside US market hours (plan/dry-run)
  python -X utf8 main.py run-scheduled --execute  # Forward to the paper order path (still gated;
                                          # needs config.SCHEDULER_DRY_RUN_DEFAULT=False to place)
  (Single-shot: never loops. Use Windows Task Scheduler + run_scheduled.bat to recur.)

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
import json
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
import permutation_test
import expectancy
import forward_test  # read-only forward paper-test report (no order path, no IBKR)
import coach_i18n  # M5A: DISPLAY-ONLY language layer (no order path, no decisions)
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


def _run_paper_startup_checks(bridge) -> dict:
    """Shared PAPER startup checks before connected paper commands use orders.

    The reconciliation path is broker-truth, may repair unprotected longs through
    the existing protective-stop repair, and runs the read-only forward-test exit
    sweep inside reconcile_startup_state(). It never opens a position.
    """
    bridge.snapshot_start_of_day_equity()
    return bridge.reconcile_startup_state()


def _print_startup_reconciliation(recon: dict) -> None:
    snap = recon["snapshot"]
    protect = recon["protect"]
    print(f"Startup reconciliation (broker = source of truth): "
          f"{len(snap['open_positions'])} long(s), "
          f"{snap['n_working_orders']} working order(s); clean={snap['clean']}")
    if protect.get("repaired"):
        print(f"Repaired protective GTC stop(s) for: {protect['repaired']}")
    if protect.get("failed"):
        print(f"Startup protection FAILED for {protect['failed']} "
              f"- NEW entries halted; closes/flatten still allowed.")
    if snap["duplicate_refs"]:
        print(f"Duplicate orderRef(s) at broker: {snap['duplicate_refs']}")
    if snap["orphan_exits"]:
        orphan_syms = sorted({o.get("symbol", "") for o in snap["orphan_exits"]})
        print(f"Orphan exit order(s) with no covering long: {orphan_syms}")


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
        recon = _run_paper_startup_checks(bridge)
        _print_startup_reconciliation(recon)

        result = bridge.execute_all(signals)
        print(f"\nâœ… Orders accepted : {result['placed']}")
        print(f"   Skipped         : {result['skipped']}")
        print(f"   Total signals   : {result['total']}")
    finally:
        # Phase 5B-3: graceful shutdown -- preserve every resting GTC protective
        # stop, repair any unprotected long via the existing Phase-3 path (or
        # halt), and mark the disconnect intentional so the watchdog reads it as
        # a clean close. Never opens a new entry; never cancels a protective stop.
        bridge.graceful_shutdown()
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

    # ----------------------------------------------------------------------------
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

    print(f"\n[REPORT] Report -> {csv_path}")
    print(
        "\n[INFO] Read-only analysis -- config.py was NOT modified. To change the live\n"
        "   threshold, edit BUY_THRESHOLD in config.py yourself."
    )
    return 0


# â”€â”€ Guided Paper Trading Coach (read-mostly, beginner-friendly) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _print_coach_intro(lang=None) -> None:
    """Print the coach banner. ``lang`` is DISPLAY-ONLY (M5A): the critical safety
    line stays BILINGUAL (English + Burmese) and keeps the literal CLI tokens."""
    lang = coach_i18n.resolve_language(lang)
    print("\n=== " + coach_i18n.t("coach_intro_title", lang) + " ===")
    print(coach_i18n.t("coach_intro_learn", lang))
    print(coach_i18n.t("coach_intro_safety", lang))


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


def cmd_coach(args) -> int:
    """Beginner lessons on the WATCHLIST. No IBKR connection, no orders."""
    lang = coach_i18n.resolve_language(getattr(args, "language", None))
    _print_coach_intro(lang)
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

    print("\n" + coach_i18n.t("no_ibkr_no_orders", lang))
    print("Next step: python -X utf8 main.py paper-coach")
    return 0


def cmd_coach_hot(args) -> int:
    """Beginner lessons on the latest hot_candidates.csv. No IBKR, no orders."""
    lang = coach_i18n.resolve_language(getattr(args, "language", None))
    _print_coach_intro(lang)
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

    print("\n" + coach_i18n.t("no_ibkr_no_orders", lang))
    return 0


# â”€â”€ Model Doctor / Model Refresh (read / maintenance only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def cmd_model_doctor(args) -> int:
    """Inspect RF/LSTM coverage + freshness for the report candidates.

    Read-only: no IBKR connection, no orders. Writes reports/model_doctor_report.md.
    """
    return model_doctor.run_doctor(top_n=getattr(args, "top_n", None))


def cmd_permutation_test(args) -> int:
    """Shuffled-label null test: does any symbol's real CV-AUC beat luck?

    Read-only research: no IBKR connection, no orders, no models saved. Writes
    reports/permutation_test_report.md (+ .json).
    """
    symbols = None
    raw = getattr(args, "symbols", None)
    if raw:
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]
    return permutation_test.run_permutation_test(
        symbols=symbols,
        n_shuffles=getattr(args, "n_shuffles", None),
        sample=0 if getattr(args, "all", False) else None,
    )


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

        _print_startup_reconciliation(_run_paper_startup_checks(bridge))
        print(
            f"\nPlacing at most 1 PAPER order for {sym} through the existing "
            "IBKRBridge (all risk controls apply)..."
        )
        accepted = bridge.execute_signal(candidate, coach=True)
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
        _print_startup_reconciliation(_run_paper_startup_checks(bridge))
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
            accepted = bridge.execute_signal(c, coach=True)
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


def _bt_summary_pct(x) -> str:
    """Format a fraction (0.05) as a percent string (+5.00%). Returns 'n/a' if None."""
    if x is None:
        return "n/a"
    return f"{x * 100:+.2f}%"


def _bt_parse_date(s: str):
    """Parse YYYY-MM-DD to a date, or None on failure (read-only, never raises)."""
    try:
        from datetime import date
        y, m, d = (int(p) for p in str(s).split("-"))
        return date(y, m, d)
    except Exception:
        return None


def _reconstruct_backtest_trades(rows):
    """Canonical closed-trade reconstruction shared with the expectancy report.

    Delegates to ``expectancy.reconstruct_closed_trades`` + ``compute_metrics`` so
    ``backtest-summary`` and ``expectancy-report`` reconcile on the SAME ledger:
    round-trips are paired WITHIN each ``(symbol, fold_start)`` group (never across
    symbols or walk-forward folds). Positions that span a fold/period boundary are
    excluded with a reason (``open_at_period_end`` / ``orphan_exit``) instead of being
    mis-paired by a global ``zip`` (the old behavior, which paired one symbol's entry
    with an unrelated symbol's exit and inflated the trade count / profit factor).

    Read-only and pure. R-multiple / proxy risk is NOT applied here (backtest-summary
    reports PnL only); the R metrics live in ``expectancy-report``.

    Returns ``(included_trades, metrics)`` where ``included_trades`` is the list of
    PnL-included ``ClosedTrade`` round-trips and ``metrics`` is the
    ``expectancy.compute_metrics`` dict.
    """
    trades = expectancy.reconstruct_closed_trades(rows)
    metrics = expectancy.compute_metrics(trades)
    included = [t for t in trades if t.pnl_included]
    return included, metrics


def cmd_backtest_summary(_args) -> int:
    """Read-only summary of the most recent backtest artifacts.

    Reads reports/backtest_metrics.json and reports/backtest_trades.csv.
    Does NOT run a backtest, train models, connect to IBKR, or place orders.
    Writes reports/backtest_summary.md.
    """
    metrics_path = config.REPORTS_DIR / "backtest_metrics.json"
    trades_path = config.REPORTS_DIR / "backtest_trades.csv"

    # ---- Load metrics (optional) -------------------------------------------
    metrics = {}
    if metrics_path.exists():
        try:
            with metrics_path.open("r", encoding="utf-8") as fh:
                metrics = json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive read-only
            print(f"Could not read {metrics_path.name}: {exc}")
    else:
        print(f"No metrics file at {metrics_path} (continuing).")

    symbols_tested = metrics.get("symbols_tested")
    model_name = metrics.get("backtest_model", "unknown")
    include_lstm = metrics.get("include_lstm", False)
    max_drawdown = metrics.get("avg_max_drawdown")

    # ---- Load trades (optional) --------------------------------------------
    rows = []
    if trades_path.exists():
        try:
            with trades_path.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        except Exception as exc:  # pragma: no cover - defensive read-only
            print(f"Could not read {trades_path.name}: {exc}")
    else:
        print(f"No trades file at {trades_path} (continuing).")

    # Actual simulated orders are those where order_executed == True.
    executed = [r for r in rows if str(r.get("order_executed")).strip().lower() == "true"]
    entries = [r for r in executed if r.get("old_position") == "0" and r.get("new_position") == "1"]
    exits = [r for r in executed if r.get("old_position") == "1" and r.get("new_position") == "0"]
    open_positions_left = len(entries) - len(exits)

    # ---- Reconstruct closed trades (canonical, shared with expectancy-report)
    # Round-trips are paired WITHIN each (symbol, fold_start) group, NOT with a
    # global zip across the whole file. This reconciles backtest-summary with
    # expectancy-report: the same ledger yields the same closed-trade count,
    # win/loss split, and profit factor in both reports. Positions that span a
    # fold/period boundary are excluded with a reason rather than mis-paired.
    included, metrics = _reconstruct_backtest_trades(rows)
    closed_trades = []
    for t in included:
        ed = _bt_parse_date(t.entry_date)
        xd = _bt_parse_date(t.exit_date)
        closed_trades.append({
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "return": t.realized_pnl,
            "hold_days": (xd - ed).days if (ed and xd) else None,
        })

    n_closed = metrics["n_trades_included"]
    wins_n = metrics["wins"]
    losses_n = metrics["losses"]
    scratch_n = metrics["scratch"]
    win_rate = metrics["win_rate"]
    avg_win = metrics["avg_win"]
    avg_loss = metrics["avg_loss"]
    pf_display = metrics["profit_factor_display"]
    total_pnl = metrics["total_realized_pnl"]
    excluded_by_reason = metrics["excluded_by_reason"]

    best = max(closed_trades, key=lambda t: t["return"]) if closed_trades else None
    worst = min(closed_trades, key=lambda t: t["return"]) if closed_trades else None

    hold_days_vals = [t["hold_days"] for t in closed_trades if t["hold_days"] is not None]
    avg_hold_days = (sum(hold_days_vals) / len(hold_days_vals)) if hold_days_vals else None

    # ---- Beginner-friendly notes -------------------------------------------
    notes = []
    if symbols_tested == 1:
        notes.append("This is a SPY-only backtest (symbols_tested = 1), not a broad test.")
    notes.append("This is NOT the full-market daily-coach backtest.")
    if avg_hold_days is not None and avg_hold_days >= 14:
        notes.append(
            f"This is NOT a quick in/out strategy — trades are held ~{avg_hold_days:.0f} days "
            "(weeks to months) on average."
        )
    notes.append("Live trading is NOT recommended from this result alone.")

    def fmt_trade(t):
        if not t:
            return "n/a"
        hold = f", held {t['hold_days']}d" if t.get("hold_days") is not None else ""
        return f"{_bt_summary_pct(t['return'])} ({t['entry_date']} -> {t['exit_date']}{hold})"

    decision = "Useful for learning, but not live-ready yet."

    # ---- Print summary ------------------------------------------------------
    print("\nBacktest Summary")
    print(f"Symbols tested: {symbols_tested if symbols_tested is not None else 'unknown'}")
    print(f"Model: {model_name}")
    print(f"LSTM included: {include_lstm}")
    print(f"Executed orders: {len(executed)}")
    print(f"Entries: {len(entries)}")
    print(f"Exits: {len(exits)}")
    print(f"Open positions left: {open_positions_left}")
    print(f"Closed trades: {n_closed}")
    if excluded_by_reason:
        ex = ", ".join(f"{k}={v}" for k, v in sorted(excluded_by_reason.items()))
        print(f"Excluded (not a clean round-trip within symbol/fold): {ex}")
    print(f"Wins / Losses / Scratch: {wins_n} / {losses_n} / {scratch_n}")
    print(f"Win rate: {win_rate * 100:.1f}%" if win_rate is not None else "Win rate: n/a")
    print(f"Average win: {_bt_summary_pct(avg_win)}")
    print(f"Average loss: {_bt_summary_pct(avg_loss)}")
    print(f"Profit factor (approx): {pf_display}")
    print(f"Best trade: {fmt_trade(best)}")
    print(f"Worst trade: {fmt_trade(worst)}")
    print(f"Max drawdown: {_bt_summary_pct(max_drawdown)}")
    print("\nWhat this means:")
    for n in notes:
        print(f"  - {n}")
    print(f"\nDecision:\n{decision}")

    # ---- Write report -------------------------------------------------------
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.REPORTS_DIR / "backtest_summary.md"

    def md_trade(t):
        return fmt_trade(t)

    pf_str = pf_display
    excluded_str = (
        ", ".join(f"`{k}`={v}" for k, v in sorted(excluded_by_reason.items()))
        if excluded_by_reason else "none"
    )
    lines = [
        "# Backtest Summary",
        "",
        "_Read-only summary of existing backtest artifacts. No backtest was run, "
        "no models trained, no IBKR connection, no orders placed._",
        "",
        "_Closed trades are reconstructed as long round-trips paired WITHIN each "
        "`(symbol, fold_start)` group — the same canonical logic as "
        "`expectancy-report`, so the two reports reconcile._",
        "",
        f"- **Symbols tested:** {symbols_tested if symbols_tested is not None else 'unknown'}",
        f"- **Model:** {model_name}",
        f"- **LSTM included:** {include_lstm}",
        f"- **Executed orders:** {len(executed)}",
        f"- **Entries:** {len(entries)}",
        f"- **Exits:** {len(exits)}",
        f"- **Open positions left:** {open_positions_left}",
        f"- **Closed trades:** {n_closed}",
        f"- **Excluded (not a clean round-trip within symbol/fold):** {excluded_str}",
        f"- **Wins / Losses / Scratch:** {wins_n} / {losses_n} / {scratch_n}",
        f"- **Win rate:** {win_rate * 100:.1f}%" if win_rate is not None else "- **Win rate:** n/a",
        f"- **Total realized PnL (return units):** {total_pnl}",
        f"- **Average win:** {_bt_summary_pct(avg_win)}",
        f"- **Average loss:** {_bt_summary_pct(avg_loss)}",
        f"- **Profit factor (approx):** {pf_str}",
        f"- **Best trade:** {md_trade(best)}",
        f"- **Worst trade:** {md_trade(worst)}",
        f"- **Max drawdown:** {_bt_summary_pct(max_drawdown)}",
        "",
        "## What this means",
        "",
    ]
    lines += [f"- {n}" for n in notes]
    lines += ["", "## Decision", "", decision, ""]

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"\nReport written -> {out_path}")
    print("No IBKR connection was made and no orders were placed.")
    return 0


def cmd_expectancy_report(args) -> int:
    """Read-only expectancy / R-multiple report from the backtest ledger (M4-core).

    Reads reports/backtest_trades.csv (the offline walk-forward simulation
    ledger). Does NOT run a backtest, train models, connect to IBKR, or place
    orders. Writes reports/expectancy_metrics.json + reports/expectancy_report.md.

    R-multiple and expectancy_R are NOT computed by default: the backtest ledger
    has no true Minervini 1R initial risk, so every trade is excluded from the R
    metrics (reason `unavailable_risk`). Pass --proxy-risk for a clearly-labeled
    hard-stop PROXY R (never presented as true Minervini R).
    """
    lang = coach_i18n.resolve_language(getattr(args, "language", None))
    trades_path = getattr(config, "EXPECTANCY_BACKTEST_TRADES",
                          config.REPORTS_DIR / "backtest_trades.csv")
    json_path = getattr(config, "EXPECTANCY_REPORT_JSON",
                        config.REPORTS_DIR / "expectancy_metrics.json")
    md_path = getattr(config, "EXPECTANCY_REPORT_MD",
                      config.REPORTS_DIR / "expectancy_report.md")
    enable_proxy = bool(getattr(args, "proxy_risk", False)) or \
        bool(getattr(config, "EXPECTANCY_ENABLE_PROXY_RISK", False))

    if not trades_path.exists():
        print(f"No backtest ledger found at {trades_path} (continuing with an empty report).")
        print("Run `python main.py backtest` first to generate reports/backtest_trades.csv.")

    report = expectancy.generate_report(trades_path, enable_proxy_risk=enable_proxy)
    jp, mp = expectancy.write_report(report, json_path=json_path, md_path=md_path, lang=lang)

    # DISPLAY-ONLY localization (M5A): labels are translated; numbers, file paths,
    # reason codes, and risk mode keys are passed through verbatim.
    m = report["metrics"]
    r = m["r"]
    risk_suffix = (f" (proxy_pct={report['proxy_pct']})" if report["risk_mode"] != "none" else "")
    print("\n" + coach_i18n.t("cli_exp_header", lang))
    print(coach_i18n.t("cli_exp_source", lang, path=report["source_file"]))
    print(coach_i18n.t("cli_exp_rows", lang, n=report["n_rows_read"]))
    print(coach_i18n.t("cli_exp_risk_mode", lang, mode=report["risk_mode"], suffix=risk_suffix))
    print(coach_i18n.t("cli_exp_trades", lang,
                       inc=m["n_trades_included"], exc=m["n_trades_excluded"]))
    print(coach_i18n.t("cli_exp_total_pnl", lang, v=m["total_realized_pnl"]))
    print(coach_i18n.t("cli_exp_wls", lang, w=m["wins"], l=m["losses"], s=m["scratch"]))
    wr = m["win_rate"]
    wr_str = f"{wr * 100:.1f}%" if wr is not None else "n/a"
    print(coach_i18n.t("cli_exp_win_rate", lang, v=wr_str))
    print(coach_i18n.t("cli_exp_profit_factor", lang, v=m["profit_factor_display"]))
    if r["expectancy_R"] is None:
        print(coach_i18n.t("cli_exp_expectancy_na", lang))
    else:
        print(coach_i18n.t("cli_exp_expectancy_val", lang,
                           v=f"{r['expectancy_R']:.4f}", n=r["n_r_included"]))
    if m["excluded_by_reason"]:
        pairs = ", ".join(f"{k}={v}" for k, v in sorted(m["excluded_by_reason"].items()))
        print(coach_i18n.t("cli_exp_excluded", lang, pairs=pairs))
    print("\n" + coach_i18n.t("cli_exp_reports_written", lang, jp=jp))
    print(f"                   {mp}")
    print(coach_i18n.t("no_ibkr_no_orders", lang))
    return 0


def cmd_forward_test_report(args) -> int:
    """Read-only forward PAPER-TEST report from the paper-fill journal.

    Reads reports/paper_trades.jsonl (the append-only journal of REAL paper fills
    written during `python main.py paper`). Does NOT connect to IBKR, train
    models, or place orders. Writes reports/forward_test_metrics.json +
    reports/forward_test_report.md.

    PnL is in US dollars; R-multiple uses BROKER PROTECTIVE-STOP risk (entry vs.
    the real resting stop), never a Minervini setup R. Win rate / expectancy_R /
    profit factor share expectancy.compute_metrics with the backtest report.
    """
    journal_path = getattr(config, "FORWARD_TEST_JOURNAL",
                           config.REPORTS_DIR / "paper_trades.jsonl")
    json_path = getattr(config, "FORWARD_TEST_REPORT_JSON",
                        config.REPORTS_DIR / "forward_test_metrics.json")
    md_path = getattr(config, "FORWARD_TEST_REPORT_MD",
                      config.REPORTS_DIR / "forward_test_report.md")
    session = getattr(args, "session", None)
    since = getattr(args, "since", None)

    if not journal_path.exists():
        print(f"No paper-fill journal found at {journal_path} (continuing with an empty report).")
        print("Run `python main.py paper` to generate reports/paper_trades.jsonl.")

    report = forward_test.generate_report(journal_path, session=session, since=since)
    jp, mp = forward_test.write_report(report, json_path=json_path, md_path=md_path)

    m = report["metrics"]
    r = m["r"]
    dd = report["drawdown"]
    wr = m["win_rate"]
    wr_str = f"{wr * 100:.1f}%" if wr is not None else "n/a"
    ddp = dd["max_drawdown_pct"]
    ddp_str = "n/a" if ddp is None else f"{ddp * 100:.1f}%"

    print("\n── Forward Paper-Test Report ──")
    print(f"Source:            {report['source_file']}")
    print(f"Events read:       {report['n_events_read']}")
    print(f"Session / Since:   {report['session_filter'] or 'all'} / {report['since_filter'] or 'all'}")
    print(f"Closed / Excluded: {m['n_trades_included']} / {m['n_trades_excluded']}")
    print(f"Total realized $:  {m['total_realized_pnl']}")
    print(f"Win / Loss / Scr:  {m['wins']} / {m['losses']} / {m['scratch']}")
    print(f"Win rate:          {wr_str}")
    print(f"Profit factor:     {m['profit_factor_display']}")
    if r["expectancy_R"] is None:
        print("Expectancy (R):    n/a (no trade had a valid broker protective-stop risk)")
    else:
        print(f"Expectancy (R):    {r['expectancy_R']:.4f}R "
              f"(n={r['n_r_included']}, broker protective-stop risk)")
    print(f"Max drawdown:      {dd['max_drawdown_usd']} ({ddp_str})")
    print(f"Sharpe (daily):    {report['sharpe']}")
    breaches = [d["date"] for d in report["daily"] if d["daily_loss_breach"]]
    if breaches:
        print(f"Daily-loss breaches: {', '.join(breaches)}")
    print(f"\nReports written:   {jp}")
    print(f"                   {mp}")
    print("(read-only — no IBKR, no orders)")
    return 0


# -- Live-trading readiness self-check (read-only) ----------------------------
def _load_bridge_module():
    """Import ibkr_bridge after preparing the asyncio loop, or return None.

    Reading the bridge's live-readiness capability flags does NOT require a
    connection, but importing ib_insync (eventkit) touches the event loop at
    import time, so prepare it first. Returns the module or None if ib_insync
    is unavailable.
    """
    _ibkr_setup_asyncio()
    try:
        import ibkr_bridge
        return ibkr_bridge
    except Exception:
        return None


def _collect_readiness_gates(bridge_mod):
    """Build the offline go-live scorecard from config + bridge capability flags.

    Returns a list of gate dicts: {phase, title, status, detail}. status is one
    of "PASS" / "FAIL" / "MANUAL". Read-only; never connects or places orders.
    """
    def cap(name):
        # Capability flags default to False/absent => not implemented yet.
        return bool(getattr(bridge_mod, name, False)) if bridge_mod else False

    def gate(phase, title, ok, detail):
        return {"phase": phase, "title": title, "status": "PASS" if ok else "FAIL", "detail": detail}

    # -- Phase 1 automated gates (flip FAIL->PASS only because the logic exists) --
    p1_gates = []
    try:
        import model_metrics
        p1_gates.append(gate(
            "P1", "Model performance gate (RF accuracy floor) implemented",
            bool(getattr(model_metrics, "SUPPORTS_MODEL_PERFORMANCE_GATE", False)),
            f"model_metrics.SUPPORTS_MODEL_PERFORMANCE_GATE; floor="
            f"{getattr(config, 'MODEL_MIN_RF_ACCURACY', None)}"))
        p1_gates.append(gate(
            "P1", "Model staleness gate (max age) implemented",
            bool(getattr(model_metrics, "SUPPORTS_MODEL_STALENESS_GATE", False)),
            f"model_metrics.SUPPORTS_MODEL_STALENESS_GATE; max_age_days="
            f"{getattr(config, 'MODEL_MAX_AGE_DAYS', None)}"))
    except Exception as exc:
        p1_gates.append(gate("P1", "Model performance/staleness gate implemented",
                             False, f"model_metrics unavailable: {exc}"))
    try:
        import backtest
        parity = backtest.check_signal_parity()
        p1_gates.append(gate(
            "P1", "Backtest/live signal-construction parity",
            bool(parity.get("parity_ok")),
            f"shared helpers={parity.get('shared_weighted_blend')}/"
            f"{parity.get('shared_action_fn')}/{parity.get('shared_position_rule')}, "
            f"numeric_match={parity.get('numeric_blend_match')}"))
    except Exception as exc:
        p1_gates.append(gate("P1", "Backtest/live signal-construction parity",
                             False, f"parity check unavailable: {exc}"))

    gates = p1_gates + [
        gate("P2", "Fill-driven order confirmation (not 'accepted')",
             cap("SUPPORTS_FILL_VERIFICATION"),
             "ibkr_bridge.SUPPORTS_FILL_VERIFICATION"),
        gate("P2", "Partial-fill handling (size children from filled qty)",
             cap("SUPPORTS_PARTIAL_FILL_HANDLING"),
             "ibkr_bridge.SUPPORTS_PARTIAL_FILL_HANDLING"),
        gate("P2", "Protective-child verification (stop is live or flatten)",
             cap("SUPPORTS_PROTECTIVE_CHILD_VERIFY"),
             "ibkr_bridge.SUPPORTS_PROTECTIVE_CHILD_VERIFY"),
        gate("P3", "Server-side GTC/OCA hard stop per entry",
             cap("SUPPORTS_SERVER_SIDE_GTC_STOP"),
             "ibkr_bridge.SUPPORTS_SERVER_SIDE_GTC_STOP"),
        gate("P3", "Daily-loss kill-switch wired into order gate",
             cap("SUPPORTS_DAILY_LOSS_KILLSWITCH"),
             "ibkr_bridge.SUPPORTS_DAILY_LOSS_KILLSWITCH (risk_state.loss_breached)"),
        gate("P4", "Real-time data guard (reject delayed for live)",
             cap("SUPPORTS_REALTIME_DATA_GUARD"),
             "ibkr_bridge.SUPPORTS_REALTIME_DATA_GUARD"),
        gate("P4", "Real-time market-data type (IBKR_MARKET_DATA_TYPE==1)",
             int(getattr(config, "IBKR_MARKET_DATA_TYPE", 3)) == 1,
             f"config.IBKR_MARKET_DATA_TYPE={getattr(config, 'IBKR_MARKET_DATA_TYPE', None)} (3=delayed)"),
        gate("P4", "Market-hours / holiday gate",
             cap("SUPPORTS_MARKET_HOURS_GATE"),
             "ibkr_bridge.SUPPORTS_MARKET_HOURS_GATE"),
        gate("P4", "Historical-close pricing disabled for orders",
             not bool(getattr(config, "ALLOW_HISTORICAL_PRICE_FOR_ORDERS", False)),
             f"config.ALLOW_HISTORICAL_PRICE_FOR_ORDERS={getattr(config, 'ALLOW_HISTORICAL_PRICE_FOR_ORDERS', None)}"),
        gate("P5", "Startup reconciliation (broker = source of truth)",
             cap("SUPPORTS_STARTUP_RECONCILIATION"),
             "ibkr_bridge.SUPPORTS_STARTUP_RECONCILIATION"),
        gate("P6", "Account-type assertion (paper DU / live U)",
             cap("SUPPORTS_ACCOUNT_TYPE_ASSERTION"),
             "ibkr_bridge.SUPPORTS_ACCOUNT_TYPE_ASSERTION"),
        gate("P6", "Explicit LIVE_ACCOUNT_ID configured",
             bool(getattr(config, "LIVE_ACCOUNT_ID", None)),
             f"config.LIVE_ACCOUNT_ID={getattr(config, 'LIVE_ACCOUNT_ID', None)}"),
    ]

    manual = [
        {"phase": "P1", "title": "Backtest --include-lstm EDGE comparison vs live",
         "status": "MANUAL", "detail": "run `backtest --include-lstm`; compare edge/frequency to live"},
        {"phase": "P1", "title": "Forward PAPER test 1-3 months matches backtest",
         "status": "MANUAL", "detail": "operator sign-off required"},
        {"phase": "P3", "title": "TWS shows a resting GTC stop for every open long",
         "status": "MANUAL", "detail": "visual check in TWS after an entry"},
    ]
    return gates + manual


def _readiness_broker_gates():
    """Optional --connect gates: inspect live PAPER broker state (read-only).

    Reports any open long without a working protective SELL (unprotected_longs).
    Connects via the paper-locked bridge and never places an order.
    """
    import live_invariants
    bridge, code = _connect_bridge()
    if bridge is None:
        return [{"phase": "BRK", "title": "Connect to PAPER account", "status": "FAIL",
                 "detail": f"could not connect (exit {code})"}]
    try:
        positions = live_invariants.adapt_positions(bridge.ib.positions())
        bridge.ib.reqAllOpenOrders()
        bridge.ib.sleep(1)
        working = live_invariants.adapt_working_orders(bridge.ib.openTrades())
        unprotected = live_invariants.unprotected_longs(positions, working)
        n_long = sum(1 for p in positions if p["qty"] > 0)
        return [{
            "phase": "BRK",
            "title": "Every open long has a working protective stop",
            "status": "PASS" if not unprotected else "FAIL",
            "detail": (f"{n_long} long position(s); unprotected: "
                       + (", ".join(unprotected) if unprotected else "none")),
        }]
    finally:
        bridge.disconnect()


def cmd_live_readiness(args) -> int:
    """Read-only go-live scorecard. Inspects config + bridge capability flags
    (and, with --connect, the live PAPER broker state). NEVER places orders.

    Exit code: 0 only when all AUTOMATED live-readiness gates PASS (manual gates
    are listed separately for operator sign-off). Today this MUST report
    NOT READY -- that is the correct Phase 0 baseline.
    """
    print("\n=== Live-Trading Readiness Self-Check (read-only) ===")
    print("Plan: reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md")
    print("This bot is PAPER-ONLY. This check does not enable live trading.\n")

    bridge_mod = _load_bridge_module()
    if bridge_mod is None:
        print("[WARN] ib_insync/ibkr_bridge unavailable -- capability flags assumed not implemented.\n")

    gates = _collect_readiness_gates(bridge_mod)
    if bool(getattr(args, "connect", False)):
        gates += _readiness_broker_gates()

    icon = {"PASS": "[PASS]", "FAIL": "[FAIL]", "MANUAL": "[MANL]"}
    print(f"{'Gate':<6} {'Status':<7} Title")
    print("-" * 78)
    for g in gates:
        print(f"{g['phase']:<6} {icon.get(g['status'], g['status']):<7} {g['title']}")
        print(f"{'':<14}-> {g['detail']}")

    n_pass = sum(1 for g in gates if g["status"] == "PASS")
    n_fail = sum(1 for g in gates if g["status"] == "FAIL")
    n_manual = sum(1 for g in gates if g["status"] == "MANUAL")

    print("\n-- Summary --")
    print(f"  Automated PASS : {n_pass}")
    print(f"  Automated FAIL : {n_fail}")
    print(f"  Manual gates   : {n_manual}  (require operator sign-off)")

    ready = n_fail == 0
    if ready:
        print("\n[OK] All automated gates pass. Complete the manual gates before going live.")
    else:
        print(f"\n[NOT READY] {n_fail} automated gate(s) still failing. Live trading must stay disabled.")
    print("No IBKR orders were placed.")
    return 0 if ready else 1


# -- Emergency kill-switch: cancel orders + flatten positions (PAPER) ----------
def cmd_panic_flatten(args) -> int:
    """Emergency: cancel ALL working orders and market-close ALL positions on the
    PAPER account. Dry-run by default; only with --confirm does it place orders.

    Paper-port-locked through IBKRBridge.connect() like every other order path.
    """
    confirm = bool(getattr(args, "confirm", False))
    print("\n=== PANIC FLATTEN (emergency kill-switch) ===")
    print("Cancels ALL working orders and MARKET-CLOSES ALL positions (PAPER account).")
    if not confirm:
        print("DRY-RUN (default): nothing will be placed. Re-run with --confirm to execute.\n")
    else:
        print("--confirm set: orders WILL be cancelled and positions WILL be flattened.\n")

    bridge, code = _connect_bridge()
    if bridge is None:
        return code
    try:
        plan = bridge.flatten_all(confirm=confirm)

        orders = plan.get("orders_to_cancel", [])
        positions = plan.get("positions_to_flatten", [])
        print(f"Working orders to cancel : {len(orders)}")
        for o in orders:
            print(f"  - {o['action']} {o['symbol']} {o['order_type']} x{o['qty']:.0f} "
                  f"(id={o['order_id']}, status={o['status']})")
        print(f"Positions to flatten     : {len(positions)}")
        for p in positions:
            print(f"  - {p['action']} {p['symbol']} x{abs(p['qty']):.0f}")

        if not confirm:
            print("\nNo orders placed (dry-run). To execute:")
            print("   python -X utf8 main.py panic-flatten --confirm")
        else:
            print(f"\nCancelled orders : {plan.get('cancelled', 0)}")
            print(f"Flattened        : {plan.get('flattened', 0)}")
            for r in plan.get("flatten_results", []):
                print(f"  - {r['action']} {r['symbol']} x{r['qty']} -> status={r['status']} "
                      f"filled={r['filled']:.0f} avg={r['avg_fill']:.2f}")
            print("\nVerify in TWS that positions are flat and no working orders remain.")
        return 0
    finally:
        bridge.disconnect()


# -- Backtest/live signal parity + model-gate config (read-only) --------------
def cmd_signal_parity(_args) -> int:
    """Read-only: print backtest/live signal-construction parity and the model
    performance/staleness gate configuration. No IBKR, no orders, no training.

    Exit code: 0 if construction parity holds, 1 otherwise.
    """
    import backtest
    import model_metrics

    print("\n=== Backtest / Live Signal Parity (read-only) ===")
    parity = backtest.check_signal_parity()
    ok = bool(parity.get("parity_ok"))
    print(f"  Shared weighted_blend        : {parity['shared_weighted_blend']}")
    print(f"  Shared action_from_confidence: {parity['shared_action_fn']}")
    print(f"  Shared position rule         : {parity['shared_position_rule']}")
    print(f"  Numeric blend match          : {parity['numeric_blend_match']}")
    print(f"  Ensemble weights             : {parity['weights']}")
    print(f"  BUY/SELL thresholds          : {parity['buy_threshold']} / {parity['sell_threshold']}")
    print(f"  MIN_ML_MODELS_FOR_SIGNAL     : {parity['min_ml_models_for_signal']}")
    print(f"  Backtest include_lstm default: {parity['include_lstm_default']}")
    print(f"  -> {parity['note']}")

    print("\n=== Model Performance / Staleness Gate ===")
    print(f"  Gate enabled        : {getattr(config, 'MODEL_GATE_ENABLED', True)}")
    print(f"  Min RF accuracy     : {getattr(config, 'MODEL_MIN_RF_ACCURACY', None)}")
    print(f"  Min RF F1 (0=off)   : {getattr(config, 'MODEL_MIN_RF_F1', None)}")
    print(f"  Max model age (days): {getattr(config, 'MODEL_MAX_AGE_DAYS', None)}")
    print(f"  Metrics file        : {getattr(config, 'MODEL_METRICS_FILE', None)}")
    data = model_metrics.load_all()
    n_rf = len(data.get("rf", {}) or {})
    n_lstm = len(data.get("lstm", {}) or {})
    print(f"  Persisted metrics   : RF={n_rf} symbol(s), LSTM={n_lstm} symbol(s)")
    if n_rf == 0 and n_lstm == 0:
        print("  [WARN] No persisted metrics yet -> every symbol is force-HOLD until you run `train`.")

    if ok:
        print("\n[OK] Backtest and live construct signals from the same source of truth.")
        print("     For full EDGE parity also run: python -X utf8 main.py backtest --include-lstm")
    else:
        print("\n[MISMATCH] Backtest and live signal construction diverged -- investigate before trusting backtests.")
    print("No IBKR connection was made and no orders were placed.")
    return 0 if ok else 1


# -- Phase 5B-1: supervised scheduler / market-hours runner (Path A) ----------
def cmd_run_scheduled(args) -> int:
    """Supervised one-shot scheduler entry (PAPER ONLY). Decides whether a
    scheduled run is allowed right now (paper-safety + SCHEDULER_ENABLED +
    US regular-hours gate) and, if so, runs the EXISTING one-shot `paper` command.

    It adds NO trading logic, NO daemon loop, and NO gate bypass -- every existing
    safety control runs inside cmd_paper. Plan/dry-run by default: it places NO
    orders unless an operator BOTH passes --execute AND has deliberately set
    config.SCHEDULER_DRY_RUN_DEFAULT=False (and even then it is paper-locked).

    Exit code: 0 for a cleanly blocked run (market closed / disabled is a normal
    no-op), otherwise the underlying command's exit code.
    """
    import scheduler_runner

    execute = bool(getattr(args, "execute", False))
    print("\n=== Supervised Scheduler Run (Path A, PAPER ONLY) ===")
    print("Decides if a scheduled run is allowed (paper-safety + enabled + market hours),")
    print("then runs the existing one-shot `paper` command. Never loops; never places live orders.")

    result = scheduler_runner.run_scheduled(execute=execute)
    decision = result["decision"]
    print("\n" + scheduler_runner.summary_line(decision))

    if not result["ran"]:
        print(f"[SKIP] Scheduled run blocked: {decision.reason} -- {decision.detail}")
        print("No orders placed. No IBKR connection made.")
        return result["exit_code"]

    mode = "DRY-RUN / PLAN (no orders)" if result["dry_run"] else \
        "EXECUTE (paper order path; still fully gated + paper-locked)"
    print(f"[RUN] Scheduled run allowed -> {mode}")
    print(f"Underlying `paper` command exit code: {result['exit_code']}")
    return result["exit_code"]


def cmd_daytrade_scan(args) -> int:
    from strategies.scanner.premarket_scanner import scan
    import time
    
    rich_available = False
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.live import Live
        rich_available = True
        console = Console()
    except ImportError:
        pass

    def _display_scan(candidates, live=False):
        if not candidates:
            if not live:
                print("  No candidates found.\n")
            return None
            
        if rich_available:
            table = Table(title="📈 Market Scanner", show_header=True, header_style="bold green")
            table.add_column("Symbol", style="cyan", no_wrap=True)
            table.add_column("Gap%", justify="right", style="green")
            table.add_column("Sess Vol", justify="right")
            table.add_column("RVol", justify="right", style="yellow")
            table.add_column("Score", justify="right", style="magenta")
            table.add_column("Sentiment", justify="right", style="magenta")
            table.add_column("Catalyst", justify="right", style="blue")
            table.add_column("Reason", style="default")
            
            for c in candidates:
                row_style = "bold green" if c.score >= 80 else "bold yellow" if c.score >= 40 else "bold red"
                table.add_row(
                    c.symbol,
                    f"{c.gap_pct:+.1f}",
                    f"{c.premarket_volume:,}",
                    f"{c.relative_volume:.1f}",
                    f"{c.score:.1f}",
                    f"{c.sentiment_score:+.2f}" if hasattr(c, "sentiment_score") else "",
                    c.catalyst_type or "",
                    c.reason,
                    style=row_style,
                )
            if not live:
                console.print(table)
                print("\n")
            return table
        else:
            from strategies.session import session_status
            print(f"\n{'='*60}")
            print("  MARKET SCANNER")
            print(f"  Session: {session_status()}")
            print(f"{'='*60}\n")
            from strategies.session import is_market_open
            volume_label = "Sess Vol" if is_market_open() else "PM Vol"
            print(f"  {'Symbol':<8} {'Gap%':>7} {volume_label:>10} {'RVol':>5} {'Score':>5} {'Sent%':>5} {'Catalyst':<10} Reason")
            print(f"  {'-'*8} {'-'*7} {'-'*10} {'-'*5} {'-'*5} {'-'*5} {'-'*10} {'-'*30}")
            for c in candidates:
                sent_str = f"{c.sentiment_score:>+5.2f}" if hasattr(c, 'sentiment_score') else "  N/A"
                cat_str = c.catalyst_type if hasattr(c, 'catalyst_type') and c.catalyst_type else "None"
                print(
                    f"  {c.symbol:<8} {c.gap_pct:>+7.1f} {c.premarket_volume:>10,} "
                    f"{c.relative_volume:>5.1f} {c.score:>5.1f} {sent_str} {cat_str:<10} {c.reason}"
                )
            print()
            return None

    if getattr(args, 'live', False) and rich_available:
        with Live(console, refresh_per_second=1, screen=False) as live_context:
            while True:
                try:
                    candidates = scan(max_candidates=args.top_n)
                    table = _display_scan(candidates, live=True)
                    if table:
                        live_context.update(table)
                    time.sleep(3)
                except KeyboardInterrupt:
                    break
    else:
        candidates = scan(max_candidates=args.top_n)
        _display_scan(candidates, live=False)
    return 0


def cmd_daytrade_signals(args) -> int:
    raw_symbols = args.symbols or list(getattr(config, "DAYTRADE_WATCHLIST", []))
    symbols = []
    for s in raw_symbols:
        symbols.extend([x.strip().upper() for x in s.split(",") if x.strip()])
    from strategies.session import session_status
    print(f"\n{'='*60}")
    print("  STRATEGY SIGNALS (DAY TRADING)")
    print(f"  Session: {session_status()}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"{'='*60}\n")

    from alpaca_bridge import AlpacaBridge
    bridge = AlpacaBridge()
    try:
        from strategies.orchestrator import evaluate_symbols_parallel, apply_portfolio_correlation
        all_signals = evaluate_symbols_parallel(symbols, bridge=bridge)

        mock_positions = getattr(args, 'mock_positions', None)
        open_positions = mock_positions if mock_positions else []

        if not mock_positions:
            try:
                if bridge.connect():
                    open_positions = bridge.ib.positions()
            except Exception:
                pass

        all_signals = apply_portfolio_correlation(all_signals, open_positions, bridge=bridge)

        if not all_signals:
            print("  No signals generated.\n")
            return 0

        all_signals.sort(key=lambda s: s.confidence, reverse=True)

        for sig in all_signals:
            rr = sig.reward_risk_ratio
            print(f"  {sig.side:<4} {sig.symbol:<8} [{sig.strategy}]")
            if getattr(args, 'verbose', False):
                raw_conf = sig.metadata.get('raw_confidence', 'N/A')
                raw_conf_str = f"{raw_conf:.2f}" if isinstance(raw_conf, float) else str(raw_conf)
                print(f"       Confidence: {sig.confidence:.2f} (Raw: {raw_conf_str})")
                
                corr_pen = sig.metadata.get('correlation_penalty', 1.0)
                max_corr = sig.metadata.get('max_correlation', 0.0)
                comp_syms = sig.metadata.get('portfolio_overlap_symbols', [])
                if comp_syms:
                    print(f"       Correlation penalty: {corr_pen:.2f}")
                    print(f"       Max corr: {max_corr:.2f} vs positions: {', '.join(comp_syms)}")
            else:
                print(f"       Confidence: {sig.confidence:.2f}")
            print(f"       Entry: ${sig.entry_price:.2f}  Stop: ${sig.stop_price:.2f}  Target: ${sig.target_price:.2f}")
            print(f"       Risk/share: ${sig.risk_per_share:.2f}  R:R = 1:{rr:.1f}")
            if sig.pattern_name:
                print(f"       Pattern: {sig.pattern_name}")
            print(f"       Reason: {sig.reason}")
        print(f"  Total signals: {len(all_signals)}\n")
    finally:
        bridge.disconnect()
    return 0


def cmd_daytrade_threshold_report(args) -> int:
    strategy_name = args.strategy.upper()
    raw_symbols = args.symbols or list(getattr(config, "DAYTRADE_WATCHLIST", []))
    symbols = []
    for s in raw_symbols:
        symbols.extend([x.strip().upper() for x in s.split(",") if x.strip()])
    lookback = args.lookback

    print(f"\n{'='*60}")
    print("  DAY TRADING THRESHOLD SWEEP REPORT")
    print(f"  Strategy: {strategy_name}")
    print(f"  Symbols: {', '.join(symbols)} ({len(symbols)} symbols)")
    print(f"  Lookback: {lookback} days")
    print(f"{'='*60}\n")

    from strategies.backtester import run_portfolio_backtest
    from strategies.constants import StrategyName
    
    name_map = {
        "ORB": StrategyName.ORB,
        "VWAP_BOUNCE": StrategyName.VWAP_BOUNCE,
        "GAP_AND_GO": StrategyName.GAP_AND_GO,
        "MOMENTUM_SCALP": StrategyName.MOMENTUM_SCALP,
        "CANDLESTICK": StrategyName.CANDLESTICK
    }
    mapped_strategy = name_map.get(strategy_name)
    if not mapped_strategy:
        print(f"  ERROR: Unsupported strategy '{strategy_name}'")
        return 1

    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(mapped_strategy)
    current_thresh = float(getattr(strat_cfg, "confidence_min", 0.65) if strat_cfg else 0.65)

    THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    
    results = []
    print(f" Running threshold grid simulation on {len(symbols)} symbols...")
    for thr in THRESHOLD_GRID:
        res = run_portfolio_backtest(
            symbols=symbols,
            strategy_name=strategy_name,
            lookback_days=lookback,
            override_min_confidence=thr,
            plot=False
        )
        results.append((thr, res))

    print("\n Thresh   Trades   Wins   Losses   Win Rate    Avg Win    Avg Loss    ProfFact    Net PnL   Status")
    print("-" * 105)
    
    csv_rows = [["threshold", "trades", "wins", "losses", "win_rate", "avg_win", "avg_loss", "profit_factor", "net_pnl", "is_current", "status"]]
    
    for thr, res in results:
        is_curr = abs(thr - current_thresh) < 1e-9
        marker = "*" if is_curr else " "
        win_rate_str = f"{res['win_rate']:.1f}%"
        pnl_str = f"${res['net_pnl']:.2f}"
        avg_win_str = f"${res['avg_win']:.2f}"
        avg_loss_str = f"${res['avg_loss']:.2f}"
        pf_str = f"{res['profit_factor']:.2f}" if res['profit_factor'] != float('inf') else "inf"
        
        status_str = "OK"
        if res['total_trades'] < 30:
            status_str = "Low Vol (!)"
            
        print(f"  {thr:>4.2f}{marker}   {res['total_trades']:>6}   {res['wins']:>4}   {res['losses']:>6}   {win_rate_str:>8}   {avg_win_str:>9}   {avg_loss_str:>9}   {pf_str:>9}   {pnl_str:>8}   {status_str}")
        
        csv_rows.append([
            f"{thr:.2f}",
            str(res['total_trades']),
            str(res['wins']),
            str(res['losses']),
            win_rate_str,
            avg_win_str,
            avg_loss_str,
            pf_str,
            pnl_str,
            "true" if is_curr else "false",
            status_str
        ])

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.REPORTS_DIR / f"daytrade_threshold_report_{strategy_name.lower()}.csv"
    
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
        
    print(f"\n[REPORT] Report -> {csv_path}")
    print(
        "\n[INFO] Read-only analysis -- config.py was NOT modified. To change the live\n"
        "   threshold, edit STRATEGY_SETTINGS in config.py yourself.\n"
    )
    return 0


def cmd_daytrade_paper(args) -> int:
    from strategies.session import session_status, is_market_open
    dry_run = args.dry_run
    raw_symbols = args.symbols or list(getattr(config, "DAYTRADE_WATCHLIST", []))
    symbols = []
    for s in raw_symbols:
        symbols.extend([x.strip().upper() for x in s.split(",") if x.strip()])

    print(f"\n{'='*60}")
    print(f"  DAY TRADING PAPER {'(DRY RUN)' if dry_run else '(LIVE PAPER)'}")
    print(f"  Session: {session_status()}")
    print(f"{'='*60}\n")

    if not dry_run and not is_market_open():
        print(f"  ERROR: Market is not open. Session: {session_status()}")
        print("  Refusing to place new paper orders outside regular market hours.\n")
        sys.exit(3)

    from alpaca_bridge import AlpacaBridge
    bridge = AlpacaBridge()

    if not dry_run:
        if not bridge.connect():
            print("  ERROR: Cannot connect to Alpaca paper broker.\n")
            sys.exit(1)

    try:
        from strategies.orchestrator import evaluate_and_execute
        from strategies.error_handler import TradingErrorHandler
        error_handler = TradingErrorHandler(bridge=bridge)
        
        evaluate_and_execute(
            watchlist=symbols, 
            bridge=bridge, 
            live_paper=not dry_run,
            error_handler=error_handler
        )

        if dry_run:
            print("  This was a DRY RUN. No orders were placed.")
            print("  Run with --live-paper to place Alpaca paper orders.\n")

    finally:
        bridge.disconnect()
    return 0


def cmd_daytrade_flatten(args) -> int:
    from alpaca_bridge import AlpacaBridge
    print(f"\n{'='*60}")
    print("  EMERGENCY FLATTEN ALL (DAY TRADING)")
    print(f"{'='*60}\n")

    bridge = AlpacaBridge()
    if not bridge.connect():
        print("  ERROR: Cannot connect to Alpaca paper broker.\n")
        sys.exit(1)

    try:
        positions = bridge.ib.positions()
        orders = bridge.ib.openTrades()
        print(f"  Open positions: {len(positions)}")
        print(f"  Working orders: {len(orders)}")

        if not positions and not orders:
            print("  Nothing to flatten.\n")
            return 0

        if not args.confirm:
            print("\n  Add --confirm to actually cancel orders and flatten positions.\n")
            return 0

        bridge.flatten_all()
        print("  All positions flattened and orders cancelled.\n")

    finally:
        bridge.disconnect()
    return 0


def cmd_daytrade_trade(args) -> int:
    from strategies.base import TradeSignal
    from strategies.command_parser import CommandParser
    from strategies.session import is_market_open
    from strategies.trade_journal import today_pnl
    from strategies.order_manager import execute_signal
    from strategies.trade_journal import day_trades_in_last_5_days
    
    dry_run = not args.live_paper
    text = args.command_text
    
    print(f"\n{'='*60}")
    print(f"  MANUAL TRADE COMMAND {'(DRY RUN)' if dry_run else '(LIVE PAPER)'}")
    print(f"  Command: '{text}'")
    print(f"{'='*60}\n")
    
    parser = CommandParser()
    cmd = parser.parse(text)
    
    if not cmd.is_valid:
        print(f"  ERROR: {cmd.error_msg}")
        sys.exit(1)
        
    print(f"  Parsed Intent: {cmd.action}", end="")
    if cmd.quantity:
        print(f" | Qty: {cmd.quantity}", end="")
    if cmd.symbol:
        print(f" | Symbol: {cmd.symbol}", end="")
    if cmd.limit_price:
        print(f" | Limit: {cmd.limit_price}", end="")
    print("\n")
    
    from alpaca_bridge import AlpacaBridge
    bridge = AlpacaBridge()
    if not dry_run:
        if not bridge.connect():
            print("  ERROR: Cannot connect to Alpaca paper broker.\n")
            sys.exit(1)
            
    try:
        if cmd.action == "FLATTEN":
            print("  Executing: Flattening all positions and orders...")
            if not dry_run:
                bridge.flatten_all()
                print("  Success: Flatten complete.")
                
        elif cmd.action == "CANCEL":
            print("  Executing: Cancelling all open orders...")
            if not dry_run:
                order_count = len(bridge.ib.openTrades())
                bridge._client.cancel_all_orders()
                print(f"  Success: Cancelled up to {order_count} eligible orders.")
                
        elif cmd.action in ("BUY", "SELL", "SHORT"):
            if not cmd.quantity or not cmd.symbol:
                print("  ERROR: Quantity or symbol missing.")
                sys.exit(1)

            if not dry_run and not is_market_open():
                print("  ERROR: Manual orders are only allowed during regular market hours.")
                sys.exit(1)

            if cmd.action == "SELL":
                exit_qty = cmd.quantity
                if not dry_run:
                    long_qty = int(max(0, bridge.get_position(cmd.symbol)))
                    if long_qty <= 0:
                        print(f"  ERROR: No long {cmd.symbol} position is available to sell.")
                        sys.exit(1)
                    exit_qty = min(exit_qty, long_qty)
                    
                    if cmd.limit_price:
                        from alpaca.trading.requests import LimitOrderRequest
                        from alpaca.trading.enums import OrderSide, TimeInForce
                        order = bridge._client.submit_order(LimitOrderRequest(
                            symbol=cmd.symbol.upper().strip(),
                            qty=exit_qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC,
                            limit_price=round(cmd.limit_price, 2)
                        ))
                    else:
                        from alpaca.trading.requests import MarketOrderRequest
                        from alpaca.trading.enums import OrderSide, TimeInForce
                        order = bridge._client.submit_order(MarketOrderRequest(
                            symbol=cmd.symbol.upper().strip(),
                            qty=exit_qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        ))
                    print(
                        f"  Result: PLACED | shares={exit_qty} | "
                        f"Exit order {order.id}"
                    )
                else:
                    print(
                        f"  Result: DRY_RUN | shares={exit_qty} | "
                        "Would reduce an existing long position"
                    )
                if dry_run:
                    print("  (This was a DRY RUN. Use --live-paper to execute.)\n")
                return 0

            entry_side = "SELL" if cmd.action == "SHORT" else "BUY"
            allow_short = getattr(config, "ALLOW_SHORT", False)
            if entry_side == "SELL" and not allow_short:
                print("  ERROR: Short entries are disabled.")
                sys.exit(1)

            if cmd.limit_price:
                entry_price = cmd.limit_price
            elif bridge._connected:
                entry_price = bridge.market_price(cmd.symbol)
            else:
                from strategies.intraday_data import fetch_intraday
                df = fetch_intraday(cmd.symbol)
                entry_price = None if df.empty else float(df["close"].iloc[-1])
            if not entry_price or entry_price <= 0:
                print("  ERROR: Could not determine a valid entry price.")
                sys.exit(1)

            fallback_stop = getattr(config, "DAYTRADE_FALLBACK_STOP_PCT", 0.02)
            risk_per_share = entry_price * fallback_stop
            default_rr = getattr(config, "DEFAULT_TARGET_RR_RATIO", 2.0)
            if entry_side == "BUY":
                stop_price = entry_price - risk_per_share
                target_price = entry_price + risk_per_share * default_rr
            else:
                stop_price = entry_price + risk_per_share
                target_price = entry_price - risk_per_share * default_rr

            signal = TradeSignal(
                symbol=cmd.symbol,
                strategy="MANUAL",
                side=entry_side,
                confidence=1.0,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                atr=0.0,
                risk_per_share=risk_per_share,
                reason="Manual command routed through risk controls",
            )
            is_connected = getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)
            equity = bridge.get_net_liquidation() if is_connected else config.PDT_MIN_EQUITY
            current_pnl = today_pnl()
            result = execute_signal(
                signal=signal,
                bridge=bridge,
                equity=equity,
                current_pnl=current_pnl,
                day_trades_last_5_days=day_trades_in_last_5_days(),
                dry_run=dry_run,
                requested_shares=cmd.quantity,
                entry_limit_price=cmd.limit_price,
            )
            print(
                f"  Result: {result['status']} | shares={result['shares']} | "
                f"{result['reason']}"
            )
            if result["status"] == "REJECTED":
                sys.exit(2)
                
        if dry_run:
            print("  (This was a DRY RUN. Use --live-paper to execute.)\n")
            
    finally:
        bridge.disconnect()
    return 0


def cmd_daytrade_status(args) -> int:
    from strategies.session import session_status, is_market_open, minutes_until_close
    from strategies.trade_journal import today_trade_count, today_pnl
    status = session_status()
    
    rich_available = False
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.align import Align
        rich_available = True
        console = Console()
    except ImportError:
        pass

    from alpaca_bridge import AlpacaBridge
    bridge = None
    if args.connect:
        bridge = AlpacaBridge()
        if not bridge.connect():
            print("  ERROR: Cannot connect to Alpaca paper broker.\n")
            sys.exit(1)
            
    try:
        if rich_available:
            account_text = (
                f"Session:      {status}\n"
            )
            if is_market_open():
                account_text += f"Minutes to close: {minutes_until_close():.0f}\n"
            account_text += (
                f"Trades Today: {today_trade_count()}\n"
                f"Today PnL:    ${today_pnl():,.2f}\n"
            )
            
            if bridge:
                try:
                    account_text += (
                        f"\nALPACA Equity:    ${bridge.get_net_liquidation():,.2f}\n"
                        f"Open Positions: {bridge.open_position_count()}\n"
                        f"Working Orders: {len(bridge.ib.openTrades())}\n"
                    )
                except Exception as exc:
                    account_text += f"\nALPACA state unavailable: {exc}\n"
            
            account_panel = Panel(
                Align.center(account_text, vertical="middle"),
                title="📊 Account Overview",
                border_style="cyan"
            )
            
            strategy_text = (
                f"ORB:           {'ON' if getattr(config, 'STRATEGY_ORB_ENABLED', True) else 'OFF'}\n"
                f"VWAP Bounce:   {'ON' if getattr(config, 'STRATEGY_VWAP_BOUNCE_ENABLED', True) else 'OFF'}\n"
                f"Gap-and-Go:    {'ON' if getattr(config, 'STRATEGY_GAP_AND_GO_ENABLED', True) else 'OFF'}\n"
                f"Momentum:      {'ON' if getattr(config, 'STRATEGY_MOMENTUM_SCALP_ENABLED', True) else 'OFF'}\n"
                f"Candlestick:   {'ON' if getattr(config, 'STRATEGY_CANDLESTICK_ENABLED', True) else 'OFF'}\n"
            )
            strategy_panel = Panel(
                Align.center(strategy_text, vertical="middle"),
                title="📈 Strategies Enabled",
                border_style="magenta"
            )
            
            sizing_method = getattr(config, "SIZING_METHOD", "risk_percent")
            max_risk_pct = getattr(config, "MAX_RISK_PER_TRADE_PCT", 1.0)
            max_daily_loss_dollars = getattr(config, "MAX_DAILY_LOSS_DOLLARS", 1000.0)
            max_daily_loss_pct = getattr(config, "MAX_DAILY_LOSS_PCT", 2.0)
            max_trades_per_day = getattr(config, "MAX_TRADES_PER_DAY", 10)
            max_open_positions = getattr(config, "MAX_OPEN_POSITIONS", 3)
            use_atr = getattr(config, "TRAILING_STOP_USE_ATR", True)
            atr_mult = getattr(config, "TRAILING_STOP_ATR_MULTIPLE", 1.5)
            fallback_pct = getattr(config, "TRAILING_STOP_FALLBACK_PCT", 0.02)
            
            risk_text = (
                f"Max risk/trade:  {max_risk_pct}%\n"
                f"Max daily loss:  ${max_daily_loss_dollars} / {max_daily_loss_pct}%\n"
                f"Max trades/day:  {max_trades_per_day}\n"
                f"Max positions:   {max_open_positions}\n"
                f"Trailing stop:   {'ON' if use_atr else 'OFF'} (ATRx{atr_mult})\n"
                f"Sizing method:   {sizing_method}\n"
            )
            risk_panel = Panel(
                Align.center(risk_text, vertical="middle"),
                title="🛡️ Risk Settings",
                border_style="yellow"
            )
            
            console.print(account_panel)
            console.print(strategy_panel)
            console.print(risk_panel)
            
            try:
                from strategies.trailing_stop import manager
                active_stops = [s for s in manager.states.values() if s.active]
                if active_stops:
                    stop_text = ""
                    for s in active_stops:
                        trail_type = f"ATRx{s.trail_multiple}" if s.atr > 0 else f"{fallback_pct*100:.1f}%"
                        side_label = "LONG" if s.side == "BUY" else "SHORT"
                        peak_label = "peak" if s.side == "BUY" else "trough"
                        stop_text += f"{s.symbol:<6} [{side_label:<5}] entry ${s.entry_price:.2f}  {peak_label} ${s.peak_price:.2f}  stop ${s.stop_price:.2f} ({trail_type})\n"
                    
                    stop_panel = Panel(
                        stop_text.strip(),
                        title="🛑 Active Trailing Stops",
                        border_style="red"
                    )
                    console.print(stop_panel)
            except Exception:
                pass
            
        else:
            print(f"\n{'='*60}")
            print("  DAY TRADING STATUS")
            print(f"{'='*60}\n")

            print(f"  Session:        {status}")

            if is_market_open():
                print(f"  Minutes to close: {minutes_until_close():.0f}")

            print(f"  Trades today:   {today_trade_count()}")
            print(f"  Today PnL:      ${today_pnl():,.2f}")
            print()

            print("  Strategies enabled:")
            print(f"    ORB:           {'ON' if getattr(config, 'STRATEGY_ORB_ENABLED', True) else 'OFF'}")
            print(f"    VWAP Bounce:   {'ON' if getattr(config, 'STRATEGY_VWAP_BOUNCE_ENABLED', True) else 'OFF'}")
            print(f"    Gap-and-Go:    {'ON' if getattr(config, 'STRATEGY_GAP_AND_GO_ENABLED', True) else 'OFF'}")
            print(f"    Momentum:      {'ON' if getattr(config, 'STRATEGY_MOMENTUM_SCALP_ENABLED', True) else 'OFF'}")
            print(f"    Candlestick:   {'ON' if getattr(config, 'STRATEGY_CANDLESTICK_ENABLED', True) else 'OFF'}")
            print()

            sizing_method = getattr(config, "SIZING_METHOD", "risk_percent")
            max_risk_pct = getattr(config, "MAX_RISK_PER_TRADE_PCT", 1.0)
            max_daily_loss_dollars = getattr(config, "MAX_DAILY_LOSS_DOLLARS", 1000.0)
            max_daily_loss_pct = getattr(config, "MAX_DAILY_LOSS_PCT", 2.0)
            max_trades_per_day = getattr(config, "MAX_TRADES_PER_DAY", 10)
            max_open_positions = getattr(config, "MAX_OPEN_POSITIONS", 3)
            use_atr = getattr(config, "TRAILING_STOP_USE_ATR", True)
            fallback_pct = getattr(config, "TRAILING_STOP_FALLBACK_PCT", 0.02)
            pdt_enabled = getattr(config, "PDT_ENABLED", True)

            print("  Risk settings:")
            print(f"    Max risk/trade:  {max_risk_pct}%")
            print(f"    Max daily loss:  ${max_daily_loss_dollars} / {max_daily_loss_pct}%")
            print(f"    Max trades/day:  {max_trades_per_day}")
            print(f"    Max positions:   {max_open_positions}")
            print(f"    PDT guard:       {'ON' if pdt_enabled else 'OFF'}")
            print(f"    Sizing method:   {sizing_method}")
            print()

            if bridge:
                print(f"  ALPACA equity:     ${bridge.get_net_liquidation():,.2f}")
                print(f"  Open positions:  {bridge.open_position_count()}")
                print(f"  Working orders:  {len(bridge.ib.openTrades())}")
                print()
                
            try:
                from strategies.trailing_stop import manager
                active_stops = [s for s in manager.states.values() if s.active]
                if active_stops:
                    print("  Active Trailing Stops:")
                    for s in active_stops:
                        trail_type = f"ATRx{s.trail_multiple}" if s.atr > 0 else f"{fallback_pct*100:.1f}%"
                        print(f"    {s.symbol:<8} entry ${s.entry_price:.2f}  peak ${s.peak_price:.2f}  stop ${s.stop_price:.2f}  ATR {s.atr:.2f}  mode {trail_type}")
                    print()
            except Exception:
                pass
                
    finally:
        if bridge:
            bridge.disconnect()
    return 0


def cmd_daytrade_bot(args) -> int:
    import time
    import signal as sig_module
    from alpaca_bridge import AlpacaBridge
    from strategies.scanner.premarket_scanner import scan
    from strategies.orchestrator import evaluate_and_execute
    from strategies.session import is_market_open, session_status, now_eastern
    
    print(f"\n{'='*60}")
    print("  AUTOMATED DAY TRADING BOT STARTED")
    print(f"{'='*60}\n")
    
    bridge = AlpacaBridge()
    if not bridge.connect():
        print("  ERROR: Cannot connect to Alpaca paper broker.\n")
        sys.exit(1)
            
    watchlist = list(getattr(config, "DAYTRADE_WATCHLIST", []))
    last_scan_time = 0
    _already_flattened = False

    def handle_graceful_exit(signum, frame):
        nonlocal _already_flattened
        if not _already_flattened:
            _already_flattened = True
            print(f"\n[SIGNAL] Received signal {signum}. Triggering emergency flatten...")
            try:
                bridge.flatten_all()
            except Exception as e:
                print(f"Error flattening positions during signal handler: {e}")
        sys.exit(0)

    # Register handlers
    sig_module.signal(sig_module.SIGTERM, handle_graceful_exit)
    sig_module.signal(sig_module.SIGINT, handle_graceful_exit)

    try:
        from strategies.error_handler import TradingErrorHandler
        error_handler = TradingErrorHandler(bridge=bridge)

        while True:
            current_time = time.time()
            if not is_market_open():
                from strategies.session import is_postmarket
                if is_postmarket():
                    print(f"[{session_status()}] Market closed. EOD reached, exiting cleanly.")
                    print("Flattening all positions before exit...")
                    _already_flattened = True
                    bridge.flatten_all()
                    sys.exit(0)

                print(f"[{session_status()}] Market closed. Waiting...")
                time.sleep(60)
                continue

            use_dynamic_scanner = getattr(config, "USE_DYNAMIC_SCANNER", True)
            if use_dynamic_scanner:
                if current_time - last_scan_time > 300:
                    print("Running periodic scanner to update watchlist...")
                    candidates = scan(max_candidates=10)
                    if candidates:
                        watchlist = [c.symbol for c in candidates]
                    last_scan_time = current_time
                    print(f"Updated watchlist: {watchlist}")
            else:
                watchlist = list(getattr(config, "DAYTRADE_WATCHLIST", []))

            # Union watchlist with open positions to ensure strategy-level exits track held symbols
            open_symbols = []
            try:
                if hasattr(bridge, "ib") and hasattr(bridge.ib, "positions"):
                    open_positions = bridge.ib.positions()
                    open_symbols = [p.contract.symbol for p in open_positions]
                elif hasattr(bridge, "positions_plain"):
                    open_symbols = [p["symbol"] for p in bridge.positions_plain()]
            except Exception as exc:
                print(f"Error fetching open positions for watchlist union: {exc}")
            
            if open_symbols:
                watchlist = list(set(watchlist) | set(open_symbols))
                print(f"Watchlist union with open positions: {watchlist}")

            earnings_cal = getattr(config, "EARNINGS_CALENDAR", {})
            blackout_days = getattr(config, "EARNINGS_BLACKOUT_DAYS", 2)
            if earnings_cal:
                from datetime import datetime, timedelta
                today = now_eastern().date()
                blacked_out = []
                for sym, date_str in earnings_cal.items():
                    try:
                        earnings_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        blackout_start = earnings_date - timedelta(days=blackout_days)
                        if blackout_start <= today <= earnings_date:
                            blacked_out.append(sym)
                    except ValueError:
                        pass
                if blacked_out:
                    watchlist = [s for s in watchlist if s not in blacked_out]
                    print(f"  [EARNINGS BLACKOUT] Removed {blacked_out} from watchlist")

            evaluate_and_execute(
                watchlist=watchlist,
                bridge=bridge,
                live_paper=getattr(args, 'live_paper', False),
                error_handler=error_handler,
            )

            print("Sleeping for 30 seconds before next evaluation...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\nStopping bot...")
        if not _already_flattened:
            _already_flattened = True
            try:
                bridge.flatten_all()
            except Exception as e:
                print(f"Error flattening positions during KeyboardInterrupt: {e}")
        sys.exit(0)
    except Exception as exc:
        print(f"\n🚨 CRITICAL ERROR: Unhandled exception in bot loop: {exc}")
        if not _already_flattened:
            _already_flattened = True
            try:
                bridge.flatten_all()
            except Exception as e:
                print(f"Error flattening positions during loop crash: {e}")
        sys.exit(1)
    finally:
        bridge.disconnect()
    return 0


def cmd_daytrade_train(args) -> int:
    from strategies.models.trainer import ModelTrainer
    trainer = ModelTrainer()
    print("Training Day Trading ML model...")
    metrics = trainer.train(feature_csv_pattern=args.feature_file)
    if "error" in metrics:
        print(f"Error: {metrics['error']}")
        return 1
    else:
        print(f"Success! Metrics: {metrics}")
        return 0


def main() -> int:

    parser = argparse.ArgumentParser(description="Stock Prediction Engine â€” IBKR Paper Trading")
    sub = parser.add_subparsers(dest="command")

    # M5A: DISPLAY-ONLY language flag for safe, read-only coach/report commands.
    # It localizes labels/headings only - never metrics, decisions, sizing, the
    # order path, JSON keys, numbers, or file paths. Intentionally NOT added to
    # any order-capable command (paper / paper-coach / daily-coach / paper-hot).
    def _add_language_flag(p) -> None:
        p.add_argument(
            "--language", choices=["en", "my"], default=None,
            help="Display language (en=English default, my=Burmese). DISPLAY ONLY: "
                 "never affects metrics, decisions, sizing, orders, JSON keys, "
                 "numbers, or file paths. Falls back to "
                 "config.MINERVINI_COACH_LANGUAGE, then English.",
        )

    # Daytrading subcommands
    p_dt_scan = sub.add_parser("daytrade-scan", help="Premarket/open-session scanner (Day Trading)")
    p_dt_scan.add_argument("--top-n", type=int, default=10)
    p_dt_scan.add_argument("--live", action="store_true", help="Auto-refresh scanner results")

    p_dt_signals = sub.add_parser("daytrade-signals", help="Evaluate strategy signals on symbols (Day Trading)")
    p_dt_signals.add_argument("--symbols", nargs="+", default=None)
    p_dt_signals.add_argument("--verbose", action="store_true", help="Show verbose output")
    p_dt_signals.add_argument("--mock-positions", nargs="+", default=None, help="Mock positions for testing correlation")

    p_dt_paper = sub.add_parser("daytrade-paper", help="Paper trade via Alpaca (Day Trading)")
    p_dt_paper.add_argument("--dry-run", action="store_true", help="Preview paper trades without placing orders")
    p_dt_paper.add_argument("--live-paper", dest="dry_run", action="store_false", help="Place orders in the Alpaca paper account")
    p_dt_paper.set_defaults(dry_run=True)
    p_dt_paper.add_argument("--symbols", nargs="+", default=None)

    p_dt_trade = sub.add_parser("daytrade-trade", help="Execute manual NLP text commands (Day Trading)")
    p_dt_trade.add_argument("command_text", help="Command text, e.g. 'Buy 100 AAPL'")
    p_dt_trade.add_argument("--live-paper", action="store_true", help="Place the order (otherwise dry run)")

    p_dt_flatten = sub.add_parser("daytrade-flatten", help="Emergency flatten all positions (Day Trading)")
    p_dt_flatten.add_argument("--confirm", action="store_true")

    p_dt_status = sub.add_parser("daytrade-status", help="Show daytrading session and account status")
    p_dt_status.add_argument("--connect", action="store_true", help="Query Alpaca paper account")

    p_dt_bot = sub.add_parser("daytrade-bot", help="Run automated daytrading bot loop")
    p_dt_bot.add_argument("--live-paper", action="store_true", help="Place paper orders in loop")

    p_dt_train = sub.add_parser("daytrade-train", help="Train ML model for day trading signals")
    p_dt_train.add_argument("--feature-file", default=None, help="Override feature CSV path")

    p_dt_thresh = sub.add_parser("daytrade-threshold-report", help="Run threshold sweep analysis for Day Trading strategies")
    p_dt_thresh.add_argument("--strategy", required=True, choices=["ORB", "VWAP_BOUNCE", "GAP_AND_GO", "MOMENTUM_SCALP", "CANDLESTICK"], help="Strategy to sweep")
    p_dt_thresh.add_argument("--symbols", nargs="+", default=None, help="Symbols to test (space or comma-separated)")
    p_dt_thresh.add_argument("--lookback", type=int, default=5, help="Number of lookback days")

    sub.add_parser("train", help="Train RF + LSTM models")
    sub.add_parser("predict", help="Show today's signals")

    bt = sub.add_parser("backtest", help="Walk-forward backtest")
    bt.add_argument("--train-min", type=int, default=252)
    bt.add_argument("--step", type=int, default=21)
    bt.add_argument("--include-lstm", action="store_true", help="Slow full-ensemble RF+LSTM+Technical backtest")

    sub.add_parser(
        "backtest-summary",
        help="Read-only summary of reports/backtest_* (no backtest, no IBKR, no orders)",
    )

    # -- M4-core: read-only expectancy / R-multiple report (no IBKR, no orders) --
    er = sub.add_parser(
        "expectancy-report",
        help="Read-only expectancy / R-multiple report from reports/backtest_trades.csv "
             "(no backtest, no IBKR, no orders)",
    )
    er.add_argument(
        "--proxy-risk", action="store_true",
        help="Compute a clearly-LABELED hard-stop PROXY R-multiple/expectancy "
             "(default off). This is NOT true Minervini 1R; backtest trades have "
             "no real setup stop.",
    )
    _add_language_flag(er)  # display-only; JSON/metrics/paths unchanged

    # -- Forward paper-test report (read-only; reads reports/paper_trades.jsonl) --
    ftr = sub.add_parser(
        "forward-test-report",
        help="Read-only forward paper-test report from reports/paper_trades.jsonl "
             "(no IBKR, no orders)",
    )
    ftr.add_argument(
        "--session", default=None,
        help="Only include journal events stamped with this session label",
    )
    ftr.add_argument(
        "--since", default=None,
        help="Only include trades whose exit (or entry, for open positions) is "
             "on/after this date (YYYY-MM-DD)",
    )

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
    _add_language_flag(
        sub.add_parser("coach", help="Beginner lessons on WATCHLIST signals (no IBKR, no orders)")
    )
    _add_language_flag(
        sub.add_parser("coach-hot", help="Beginner lessons on hot_candidates.csv (no IBKR, no orders)")
    )
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

    # -- Live-trading readiness self-check (read-only; never places orders) --
    lr = sub.add_parser(
        "live-readiness",
        help="Read-only go-live scorecard (config + capability flags; no orders)",
    )
    lr.add_argument(
        "--connect", action="store_true",
        help="Also connect to the PAPER account (read-only) to check for "
             "unprotected open positions",
    )

    # -- Emergency kill-switch (dry-run unless --confirm) --
    pf = sub.add_parser(
        "panic-flatten",
        help="Emergency: cancel all working orders + market-close all positions "
             "(PAPER; dry-run unless --confirm)",
    )
    pf.add_argument(
        "--confirm", action="store_true",
        help="Actually cancel orders and flatten positions (default is dry-run)",
    )

    # -- Backtest/live signal parity + model-gate config (read-only) --
    sub.add_parser(
        "signal-parity",
        help="Show backtest/live signal-construction parity + model-gate config "
             "(read-only, no IBKR, no orders)",
    )

    # -- Shuffled-label permutation null test (read-only research) --
    pt = sub.add_parser(
        "permutation-test",
        help="Shuffled-label null test: does any symbol's real CV-AUC beat luck? "
             "(read-only, no IBKR, no orders, no models saved)",
    )
    pt.add_argument(
        "--all", action="store_true",
        help="Test the full universe (ignore PERMUTATION_SAMPLE_SYMBOLS)",
    )
    pt.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols to test (overrides the watchlist)",
    )
    pt.add_argument("--n-shuffles", type=int, default=None, help="Null draws per symbol")

    # -- Phase 5B-1: supervised scheduler / market-hours runner (Path A) --
    rs = sub.add_parser(
        "run-scheduled",
        help="Supervised one-shot scheduler: run the `paper` command only inside "
             "US market hours (PAPER ONLY; plan/dry-run by default, never loops)",
    )
    rs.add_argument(
        "--execute", action="store_true",
        help="Forward to the (paper) order path instead of plan/dry-run. Still "
             "requires config.SCHEDULER_DRY_RUN_DEFAULT=False to place orders, and "
             "stays paper-port-locked. Never enables live trading.",
    )

    args = parser.parse_args()
    if args.command == "daytrade-scan":
        return cmd_daytrade_scan(args)
    if args.command == "daytrade-signals":
        return cmd_daytrade_signals(args)
    if args.command == "daytrade-paper":
        return cmd_daytrade_paper(args)
    if args.command == "daytrade-trade":
        return cmd_daytrade_trade(args)
    if args.command == "daytrade-flatten":
        return cmd_daytrade_flatten(args)
    if args.command == "daytrade-status":
        return cmd_daytrade_status(args)
    if args.command == "daytrade-bot":
        return cmd_daytrade_bot(args)
    if args.command == "daytrade-train":
        return cmd_daytrade_train(args)
    if args.command == "daytrade-threshold-report":
        return cmd_daytrade_threshold_report(args)

    if args.command == "train":
        return cmd_train(args)
    if args.command == "predict":
        return cmd_predict(args)
    if args.command == "backtest":
        return cmd_backtest(args)
    if args.command == "backtest-summary":
        return cmd_backtest_summary(args)
    if args.command == "expectancy-report":
        return cmd_expectancy_report(args)
    if args.command == "forward-test-report":
        return cmd_forward_test_report(args)
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
    if args.command == "live-readiness":
        return cmd_live_readiness(args)
    if args.command == "panic-flatten":
        return cmd_panic_flatten(args)
    if args.command == "signal-parity":
        return cmd_signal_parity(args)
    if args.command == "permutation-test":
        return cmd_permutation_test(args)
    if args.command == "run-scheduled":
        return cmd_run_scheduled(args)

    parser.print_help()
    return 3


if __name__ == "__main__":
    sys.exit(main())
