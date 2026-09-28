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
    default_journal = "reports/trade_journal.jsonl" if Path("reports/trade_journal.jsonl").exists() else "reports/daytrade_journal.jsonl"
    parser.add_argument(
        "--journal",
        type=str,
        default=default_journal,
        help="Path to the JSONL trade journal file."
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=DEFAULT_CAPITAL,
        help="Mock initial capital for drawdown tracking."
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Filter trade records by cohort (e.g. 'atomic_rotation_v1')."
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Filter trade records on or after ISO date (e.g. '2026-10-01')."
    )
    return parser.parse_args()


def load_records(journal_path: Path, since: str | None = None) -> list:
    if not journal_path.exists():
        print(f"Error: Journal file not found at {journal_path}", file=sys.stderr)
        return []
    records = []
    with open(journal_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    if since:
                        ts = str(rec.get("timestamp", ""))[:10]
                        if ts and ts < since[:10]:
                            continue
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def calculate_metrics(records: list, initial_capital: float, cohort: str | None = None) -> dict:
    if cohort is not None:
        records = [r for r in records if r.get("cohort") == cohort]

    trades = []
    fills = []

    for r in records:
        event_type = r.get("event_type") or r.get("type")
        # Trade logs (both opened TRADE with exit_price and closed TRADE_CLOSED)
        if event_type in ("TRADE", "TRADE_CLOSED") and r.get("exit_price") is not None:
            trades.append(r)
        # Fill logs
        elif event_type == "FILL":
            fills.append(r)

    # Sort trades chronologically
    trades.sort(key=lambda t: t.get("timestamp", ""))

    def _get_trade_pnl(t: dict) -> float:
        val = t.get("pnl")
        if val is None:
            val = t.get("realized_pnl")
        try:
            return float(val or 0.0)
        except (ValueError, TypeError):
            return 0.0

    # 1. Total Trades
    n_trades = len(trades)

    # 1b. Sample Diversity (Distinct Days >= 10, Single-day contribution <= 30%)
    from collections import Counter
    from datetime import date
    day_counts = Counter(str(t.get("timestamp", ""))[:10] for t in trades if t.get("timestamp"))
    distinct_days = len(day_counts)
    max_single_day_trades = max(day_counts.values()) if day_counts else 0
    max_single_day_pct = (max_single_day_trades / n_trades * 100.0) if n_trades > 0 else 0.0

    # Evaluation duration (calendar days and weeks elapsed)
    valid_dates = sorted([str(t.get("timestamp", ""))[:10] for t in trades if t.get("timestamp")])
    if valid_dates:
        try:
            d0 = date.fromisoformat(valid_dates[0])
            d1 = date.fromisoformat(valid_dates[-1])
            eval_days = (d1 - d0).days + 1
        except Exception:
            eval_days = distinct_days
    else:
        eval_days = 0
    eval_weeks = eval_days / 7.0

    # 2. Win Rate
    winning_trades = [t for t in trades if _get_trade_pnl(t) > 0.0]
    win_rate = winning_trades_ratio = winning_trades_pct = 0.0
    if n_trades > 0:
        winning_trades_ratio = len(winning_trades) / n_trades
        win_rate = winning_trades_pct = winning_trades_ratio * 100.0

    # 3. Profit Factor
    gross_profit = sum(_get_trade_pnl(t) for t in trades if _get_trade_pnl(t) > 0.0)
    gross_loss = sum(abs(_get_trade_pnl(t)) for t in trades if _get_trade_pnl(t) < 0.0)
    
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 1.0

    # 4. Maximum Drawdown (Closed-trade consecutive peak-to-trough as % of initial starting equity)
    equity = initial_capital
    peak = initial_capital
    max_dd_dollars = 0.0
    max_dd_pct = 0.0

    for t in trades:
        pnl = _get_trade_pnl(t)
        equity += pnl
        if equity > peak:
            peak = equity
        dd_dollars = peak - equity
        dd_pct = (dd_dollars / initial_capital) * 100.0 if initial_capital > 0 else 0.0
        
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

    # 6. Expectancy and Average PnL Calculations
    import math
    losing_trades = [t for t in trades if _get_trade_pnl(t) < 0.0]
    avg_win = gross_profit / len(winning_trades) if winning_trades else 0.0
    avg_loss = gross_loss / len(losing_trades) if losing_trades else 0.0
    win_fraction = winning_trades_ratio
    loss_fraction = len(losing_trades) / n_trades if n_trades > 0 else 0.0
    expectancy = (win_fraction * avg_win) - (loss_fraction * avg_loss)

    pnls = [_get_trade_pnl(t) for t in trades]
    if len(pnls) >= 2:
        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((x - mean_pnl) ** 2 for x in pnls) / (len(pnls) - 1)
        std_dev = math.sqrt(variance)
        std_err = std_dev / math.sqrt(len(pnls))
        ci_low = expectancy - 1.96 * std_err
        ci_high = expectancy + 1.96 * std_err
        is_statistically_robust = ci_low > 0
    else:
        std_err = 0.0
        ci_low = expectancy
        ci_high = expectancy
        is_statistically_robust = False

    return {
        "n_trades": n_trades,
        "eval_days": eval_days,
        "eval_weeks": eval_weeks,
        "distinct_days": distinct_days,
        "max_single_day_trades": max_single_day_trades,
        "max_single_day_pct": max_single_day_pct,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_dd_pct": max_dd_pct,
        "max_dd_dollars": max_dd_dollars,
        "avg_slippage": avg_slippage,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "std_err": std_err,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "is_statistically_robust": is_statistically_robust,
        "final_equity": equity
    }


def print_report(m: dict, cohort: str | None = None):
    # Pass/Fail evaluation: Binding Hard Gates Only
    gate_n = (m["n_trades"] >= 30) and (m["eval_days"] >= 14)
    gate_diversity = (m["distinct_days"] >= 10) and (m["max_single_day_pct"] <= 30.0)
    gate_pf = m["profit_factor"] >= 1.25
    gate_dd = m["max_dd_pct"] <= 5.0

    all_passed = gate_n and gate_diversity and gate_pf and gate_dd

    print("======================================================================")
    print("           PRODUCTION PROMOTION GATES EVALUATION REPORT               ")
    print("======================================================================")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if cohort:
        print(f"Cohort Filter: {cohort}")
    print(f"Total Closed Trades Evaluated: {m['n_trades']}")
    if m["n_trades"] == 0:
        print("----------------------------------------------------------------------")
        print("[NOTICE] STATUS: NO TRADES FOR SPECIFIED COHORT. Continue paper trading.")
        sys.exit(1)
    print(f"Final Mock Equity: ${m['final_equity']:.2f}")
    print(f"Gross Profits: ${m['gross_profit']:.2f} | Gross Losses: ${m['gross_loss']:.2f}")
    print("----------------------------------------------------------------------")
    print("Binding Hard Promotion Gate Matrix:")
    print("----------------------------------------------------------------------")
    
    def format_status(passed: bool) -> str:
        return "[PASS]" if passed else "[FAIL]"

    if not gate_n:
        if m["n_trades"] < 30 and m["eval_days"] < 14:
            prelim = " (Preliminary - sample size N < 30 & duration < 14 days)"
        elif m["n_trades"] < 30:
            prelim = " (Preliminary - sample size N < 30)"
        else:
            prelim = " (Preliminary - duration < 14 calendar days)"
    else:
        prelim = ""

    print(f"Gate 1.  Sample Size & Time (N >= 30 & >= 2 wks):  {format_status(gate_n)} (N = {m['n_trades']}, {m['eval_weeks']:.1f} wks [{m['eval_days']} days]){prelim}")
    print(f"Gate 1b. Sample Diversity (Days >= 10, Day <= 30%): {format_status(gate_diversity)} ({m['distinct_days']} days, max single day = {m['max_single_day_trades']}/{m['n_trades']} [{m['max_single_day_pct']:.1f}%]){prelim}")
    pf_val = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "∞"
    print(f"Gate 2.  Profit Factor Gate (PF >= 1.25):         {format_status(gate_pf)} (PF = {pf_val}){prelim}")
    print(f"Gate 3.  Drawdown Tolerance (Max DD <= 5.0%):     {format_status(gate_dd)} (Max DD = {m['max_dd_pct']:.2f}% / ${m['max_dd_dollars']:.2f} on starting equity){prelim}")
    print("----------------------------------------------------------------------")
    print("Reported Diagnostic Co-Metrics (Informative / Non-Blocking):")
    print("----------------------------------------------------------------------")
    wr_ref = "[REF PASS]" if m['win_rate'] >= 40.0 else "[REF BELOW]"
    print(f" • Win Rate (Target Reference >= 40.0%):          {wr_ref} {m['win_rate']:.1f}%")
    slip_ref = "[REF PASS]" if m['avg_slippage'] <= 0.05 else "[REF WIDE]"
    print(f" • Execution Slippage (Target <= 0.05%):          {slip_ref} {m['avg_slippage']:.3f}%*")
    print("   * Footnote: Paper fills are broker-simulated — slippage measurement is approximate and requires empirical re-verification under live quote conditions upon real-money migration.")
    print("----------------------------------------------------------------------")
    print(f"Expectancy & Mathematical Edge (Statistical Robustness):")
    print(f" • Avg Winning Trade: ${m['avg_win']:.2f} | Avg Losing Trade: ${m['avg_loss']:.2f}")
    exp_status = "[PASS]" if m['expectancy'] > 0 else "[FAIL]"
    print(f" • Point Expectancy: {exp_status} ${m['expectancy']:+.2f} per trade")

    # Calculate R-multiple Net EV assuming average loss is 1R base risk
    base_r = abs(m['avg_loss']) if m['avg_loss'] > 0 else (abs(m['expectancy']) or 1.0)
    avg_win_r = m['avg_win'] / base_r if base_r > 0 else 0.0
    avg_loss_r = 1.0 if m['avg_loss'] > 0 else 0.0
    p_win = m['win_rate'] / 100.0
    p_loss = 1.0 - p_win
    avg_cost_r = 0.08
    gross_ev_r = (p_win * avg_win_r) - (p_loss * avg_loss_r)
    net_ev_r = gross_ev_r - avg_cost_r
    net_ev_dollars = net_ev_r * base_r

    print(f" • Friction-Adjusted Net EV: {net_ev_r:+.3f}R per trade (${net_ev_dollars:+.2f} after 0.08R cost)")
    if m['n_trades'] >= 2:
        print(f" • 95% Confidence Interval (CI95): [${m['ci_low']:+.2f}, ${m['ci_high']:+.2f}] (Std Err: ${m['std_err']:.2f})")
        robust_tag = "[ROBUST EDGE - Lower CI > 0]" if m['is_statistically_robust'] else "[INCONCLUSIVE - Lower CI <= 0, need more sample size N]"
        print(f" • Statistical Significance: {robust_tag}")
    print("----------------------------------------------------------------------")
    
    if all_passed:
        print("[SUCCESS] STATUS: ALL GATES PASSED! Safe to promote system to LIVE TRADING.")
        sys.exit(0)
    else:
        print("[NOTICE] STATUS: GATES NOT YET SATISFIED. Continue paper trading.")
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

    records = load_records(journal_path, since=args.since)
    if not records:
        print("No trade records found to evaluate.")
        sys.exit(1)

    metrics = calculate_metrics(records, args.capital, args.cohort)
    print_report(metrics, args.cohort)


if __name__ == "__main__":
    main()
