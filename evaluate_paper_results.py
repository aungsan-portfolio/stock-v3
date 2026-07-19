#!/usr/bin/env python3
"""
evaluate_paper_results.py - Quantitative Evaluation Tool for Daytrading Dry Runs.

Calculates the 5 Production Promotion Gates from a JSONL trade journal:
  1. Statistical Sample Size (N >= 30)
  2. Profit Factor Gate (PF >= 1.25)
  3. Win Rate / RR Consistency (Win Rate >= 40%)
  4. Drawdown Tolerance (Max Drawdown <= 5.0%)
  5. Slippage Audit (Avg Slippage <= 0.05%)
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Default initial mock capital for drawdown calculations
DEFAULT_CAPITAL = 10000.0


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trade journal against Production Gates.")
    parser.add_argument(
        "--journal",
        type=str,
        default="reports/daytrade_journal.jsonl",
        help="Path to the JSONL trade journal file."
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help="Mock initial capital for drawdown tracking."
    )
    return parser.parse_args()


def load_records(journal_path: Path) -> list:
    if not journal_path.exists():
        print(f"Error: Journal file not found at {journal_path}", file=sys.stderr)
        return []
    records = []
    with open(journal_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def calculate_metrics(records: list, initial_capital: float) -> dict:
    trades = []
    fills = []

    for r in records:
        event_type = r.get("event_type")
        # Trade logs
        if event_type == "TRADE" and r.get("exit_price") is not None:
            trades.append(r)
        # Fill logs
        elif event_type == "FILL" or r.get("type") == "FILL":
            fills.append(r)

    # Sort trades chronologically
    trades.sort(key=lambda t: t.get("timestamp", ""))

    # 1. Total Trades
    n_trades = len(trades)

    # 2. Win Rate
    winning_trades = [t for t in trades if float(t.get("pnl", 0.0) or 0.0) > 0.0]
    win_rate = winning_trades_ratio = winning_trades_pct = 0.0
    if n_trades > 0:
        winning_trades_ratio = len(winning_trades) / n_trades
        win_rate = winning_trades_pct = winning_trades_ratio * 100.0

    # 3. Profit Factor
    gross_profit = sum(float(t.get("pnl", 0.0) or 0.0) for t in trades if float(t.get("pnl", 0.0) or 0.0) > 0.0)
    gross_loss = sum(abs(float(t.get("pnl", 0.0) or 0.0)) for t in trades if float(t.get("pnl", 0.0) or 0.0) < 0.0)
    
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 1.0

    # 4. Maximum Drawdown
    equity = initial_capital
    peak = initial_capital
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    for t in trades:
        pnl = float(t.get("pnl", 0.0) or 0.0)
        equity += pnl
        if equity > peak:
            peak = equity
        dd_dollars = peak - equity
        dd_pct = (dd_dollars / peak) * 100.0 if peak > 0 else 0.0
        
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_dollars = dd_dollars

    # 5. Slippage Audit
    slippages = []
    for f in fills:
        fill_p = float(f.get("fill_price", 0.0) or 0.0)
        exp_p = float(f.get("expected_price", 0.0) or 0.0)
        if exp_p > 0:
            slip_pct = (abs(fill_p - exp_p) / exp_p) * 100.0
            slippages.append(slip_pct)
    
    avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_dd_pct": max_dd_pct,
        "max_dd_dollars": max_dd_dollars,
        "avg_slippage": avg_slippage,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "final_equity": equity
    }


def print_report(m: dict):
    # Pass/Fail evaluation
    gate_n = m["n_trades"] >= 30
    gate_pf = m["profit_factor"] >= 1.25
    gate_wr = m["win_rate"] >= 40.0
    gate_dd = m["max_dd_pct"] <= 5.0
    gate_slip = m["avg_slippage"] <= 0.05

    all_passed = gate_n and gate_pf and gate_wr and gate_dd and gate_slip

    print("======================================================================")
    print("           PRODUCTION PROMOTION GATES EVALUATION REPORT               ")
    print("======================================================================")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Closed Trades Evaluated: {m['n_trades']}")
    print(f"Final Mock Equity: ${m['final_equity']:.2f}")
    print(f"Gross Profits: ${m['gross_profit']:.2f} | Gross Losses: ${m['gross_loss']:.2f}")
    print("----------------------------------------------------------------------")
    print("Gate Verification Matrix:")
    print("----------------------------------------------------------------------")
    
    def format_status(passed: bool) -> str:
        return "✅ PASSED" if passed else "❌ FAILED"

    prelim = " (⚠️ Preliminary - sample size N < 30)" if not gate_n else ""

    print(f"1. Statistical Sample Size (N >= 30):   {format_status(gate_n)} (N = {m['n_trades']})")
    
    pf_val = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞"
    print(f"2. Profit Factor Gate (PF >= 1.25):     {format_status(gate_pf)} (PF = {pf_val}){prelim}")
    print(f"3. Win Rate Gate (WR >= 40%):           {format_status(gate_wr)} (WR = {m['win_rate']:.1f}%){prelim}")
    print(f"4. Drawdown Tolerance (Max DD <= 5.0%): {format_status(gate_dd)} (Max DD = {m['max_dd_pct']:.2f}% / ${m['max_dd_dollars']:.2f}){prelim}")
    print(f"5. Slippage Audit (Avg Slip <= 0.05%):  {format_status(gate_slip)} (Avg Slip = {m['avg_slippage']:.3f}%){prelim}")
    print("----------------------------------------------------------------------")
    
    if all_passed:
        print("🎉 STATUS: ALL GATES PASSED! Safe to promote system to LIVE TRADING.")
        sys.exit(0)
    else:
        print("⚠️ STATUS: GATES NOT YET SATISFIED. Continue paper trading.")
        sys.exit(1)


def main():
    args = parse_args()
    journal_path = Path(args.journal)
    
    if not journal_path.exists():
        # Look for default CSV path or configure default JSONL
        # config.DAYTRADE_TRADE_JOURNAL_FILE might be config/DAYTRADE_TRADE_JOURNAL_FILE
        # Let's fallback to search inside workspace config settings
        import config
        journal_path = config.DAYTRADE_TRADE_JOURNAL_FILE

    records = load_records(journal_path)
    if not records:
        print("No trade records found to evaluate.")
        sys.exit(1)

    metrics = calculate_metrics(records, args.capital)
    print_report(metrics)


if __name__ == "__main__":
    main()
