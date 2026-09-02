#!/usr/bin/env python3
"""
evaluate_forward_cohort.py - Quantitative Forward Cohort Tracker (Phase 2 Baseline).

Isolates modern Phase 2 trades (August 17, 2026 onwards) from legacy sandbox noise
and evaluates the 5 Promotion Gates against .agents/AGENTS.md Rule #3:
  1. Statistical Sample Size (N >= 30)
  2. Profit Factor Gate (PF >= 1.25)
  3. Win Rate Gate (WR >= 40%)
  4. Max Drawdown Gate (Max DD <= 5.0%)
  5. Slippage Audit (Avg Slippage <= 0.05%)
"""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Configure UTF-8 for Windows console output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PHASE2_BASELINE_DATE = "2026-08-17T00:00:00"

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 Forward Cohort against Promotion Gates.")
    parser.add_argument(
        "--journal",
        type=str,
        default="reports/trade_journal.jsonl",
        help="Path to the JSONL trade journal file."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=PHASE2_BASELINE_DATE,
        help="Cutoff timestamp for forward cohort (default: 2026-08-17)."
    )
    return parser.parse_args()


def load_forward_trades(journal_path: Path, start_date: str):
    if not journal_path.exists():
        print(f"Error: Journal file not found at {journal_path}", file=sys.stderr)
        return []

    lines = journal_path.read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(l) for l in lines if l.strip()]

    # Parse and reconstruct round-trip trades from fills
    fills = []
    for r in records:
        ts = r.get("timestamp", "")
        if ts < start_date:
            continue
        
        side = r.get("side")
        sym = r.get("symbol")
        px = r.get("price") or r.get("entry_price") or r.get("fill_price")
        qty = r.get("shares") or r.get("qty") or r.get("quantity") or 0
        strat = r.get("strategy", "UNKNOWN")
        slip = r.get("slippage") or r.get("slippage_pct") or 0.0
        
        if side and sym and px:
            fills.append({
                "timestamp": ts,
                "symbol": sym,
                "side": side.upper(),
                "price": float(px),
                "qty": float(qty) if qty else 1.0,
                "strategy": strat,
                "slippage": float(slip)
            })

    positions = defaultdict(list)
    closed_trades = []

    for f in fills:
        sym = f["symbol"]
        side = f["side"]
        qty = f["qty"]
        px = f["price"]
        strat = f["strategy"]
        ts = f["timestamp"]
        
        if side == "BUY":
            positions[sym].append(f)
        elif side == "SELL":
            if positions[sym]:
                entry = positions[sym].pop(0)
                matched_qty = min(qty, entry["qty"])
                pnl = (px - entry["price"]) * matched_qty
                pnl_pct = ((px - entry["price"]) / entry["price"]) * 100 if entry["price"] > 0 else 0
                closed_trades.append({
                    "symbol": sym,
                    "entry_time": entry["timestamp"],
                    "exit_time": ts,
                    "entry_price": entry["price"],
                    "exit_price": px,
                    "qty": matched_qty,
                    "strategy": strat if strat != "UNKNOWN" else entry["strategy"],
                    "pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "slippage": f["slippage"] or entry["slippage"]
                })

    return closed_trades


def evaluate_cohort(trades: list):
    print("=" * 70)
    print("  PHASE 2 FORWARD COHORT EVALUATION (August 17, 2026 — Present)")
    print("=" * 70)

    n_trades = len(trades)
    print(f"\n--- Cohort Sample Size: N = {n_trades} closed trades (Target: N >= 30) ---")

    if n_trades == 0:
        print("No closed trades recorded in Phase 2 forward window yet.")
        return

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    breakevens = [t for t in trades if t["pnl"] == 0]

    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    net_pnl = sum(t["pnl"] for t in trades)

    win_rate = (len(wins) / n_trades) * 100.0 if n_trades > 0 else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0
    expectancy = ((len(wins) / n_trades) * avg_win) - ((len(losses) / n_trades) * avg_loss)

    # Max Drawdown Calculation
    cum_pnl = 0.0
    peak_pnl = 0.0
    max_dd = 0.0
    for t in trades:
        cum_pnl += t["pnl"]
        if cum_pnl > peak_pnl:
            peak_pnl = cum_pnl
        dd = peak_pnl - cum_pnl
        if dd > max_dd:
            max_dd = dd

    mock_equity = 98404.67
    max_dd_pct = (max_dd / mock_equity) * 100.0 if mock_equity > 0 else 0.0
    avg_slippage = (sum(t["slippage"] for t in trades) / n_trades) if n_trades > 0 else 0.0

    print(f"  Gross Profit:       ${gross_profit:+.2f}")
    print(f"  Gross Loss:        -${gross_loss:.2f}")
    print(f"  Net PnL:            ${net_pnl:+.2f}")
    print(f"  Profit Factor (PF): {profit_factor:.2f}")
    print(f"  Win Rate:           {win_rate:.1f}% ({len(wins)}W / {len(losses)}L / {len(breakevens)}BE)")
    print(f"  Avg Win / Avg Loss: ${avg_win:.2f} / -${avg_loss:.2f} (Payoff: {payoff_ratio:.2f}x)")
    print(f"  Expectancy / Trade: ${expectancy:+.2f}")
    print(f"  Max Drawdown:       ${max_dd:.2f} ({max_dd_pct:.2f}% of equity)")

    # ── Strategy Breakdown ──
    print(f"\n--- Strategy Performance Breakdown ---")
    by_strat = defaultdict(list)
    for t in trades:
        by_strat[t["strategy"]].append(t)

    for strat, strades in sorted(by_strat.items()):
        swins = [t for t in strades if t["pnl"] > 0]
        sloss = [t for t in strades if t["pnl"] < 0]
        sgp = sum(t["pnl"] for t in swins)
        sgl = abs(sum(t["pnl"] for t in sloss))
        spf = (sgp / sgl) if sgl > 0 else (999.0 if sgp > 0 else 0.0)
        swr = (len(swins) / len(strades)) * 100.0
        snet = sum(t["pnl"] for t in strades)
        print(f"  {strat:<20}: N={len(strades):<2} | WinRate={swr:5.1f}% | PF={spf:5.2f} | Net=${snet:+7.2f}")

    # ── Workhorse Strategy Spotlight (MOMENTUM_SCALP) ──
    print(f"\n--- Core Workhorse Focus (MOMENTUM_SCALP) ---")
    m_trades = [t for t in trades if t["strategy"] == "MOMENTUM_SCALP"]
    if m_trades:
        m_n = len(m_trades)
        m_wins = [t for t in m_trades if t["pnl"] > 0]
        m_loss = [t for t in m_trades if t["pnl"] < 0]
        m_gp = sum(t["pnl"] for t in m_wins)
        m_gl = abs(sum(t["pnl"] for t in m_loss))
        m_net = sum(t["pnl"] for t in m_trades)
        m_wr = (len(m_wins) / m_n) * 100.0
        m_pf = (m_gp / m_gl) if m_gl > 0 else (999.0 if m_gp > 0 else 0.0)
        m_avg_win = (m_gp / len(m_wins)) if m_wins else 0.0
        m_avg_loss = (m_gl / len(m_loss)) if m_loss else 0.0
        m_exp = ((len(m_wins) / m_n) * m_avg_win) - ((len(m_loss) / m_n) * m_avg_loss)
        
        # Real-money friction model (approx $0.005/share commission + 0.02% spread round-trip)
        est_shares = sum(t["qty"] for t in m_trades) * 2 # entry + exit
        est_volume_dollars = sum(t["qty"] * t["entry_price"] for t in m_trades) * 2
        est_friction = (est_shares * 0.005) + (est_volume_dollars * 0.0002)
        m_net_frictional = m_net - est_friction

        print(f"  Trades: N = {m_n} ({m_n/n_trades*100:.0f}% of cohort volume)")
        print(f"  Win Rate:              {m_wr:.1f}% ({len(m_wins)}W / {len(m_loss)}L)")
        print(f"  Gross P/L:             +${m_gp:.2f} / -${m_gl:.2f}")
        print(f"  Gross Net PnL:         ${m_net:+.2f}")
        print(f"  Est. Friction Cost:   -${est_friction:.2f} (commissions + spread model)")
        print(f"  Friction-Adj Net PnL:  ${m_net_frictional:+.2f}")
        print(f"  Workhorse PF:          {m_pf:.2f} (Break-even: 1.00)")
        print(f"  Workhorse Exp/Tr:      ${m_exp:+.2f}")
    else:
        print("  No MOMENTUM_SCALP trades in forward window.")

    # ── Outlier-Trimmed Robustness Audit ──
    print(f"\n--- Outlier-Trimmed Robustness (Excluding Top Outlier Win) ---")
    if len(wins) > 1:
        sorted_pnls = sorted([t["pnl"] for t in trades], reverse=True)
        top_win = sorted_pnls[0]
        trimmed_pnls = sorted_pnls[1:]
        t_wins = [p for p in trimmed_pnls if p > 0]
        t_loss = [p for p in trimmed_pnls if p < 0]
        t_gp = sum(t_wins)
        t_gl = abs(sum(t_loss))
        t_net = sum(trimmed_pnls)
        t_pf = (t_gp / t_gl) if t_gl > 0 else (999.0 if t_gp > 0 else 0.0)
        t_exp = (t_net / len(trimmed_pnls)) if trimmed_pnls else 0.0
        print(f"  Top Outlier Excluded: ${top_win:+.2f}")
        print(f"  Trimmed Net PnL:      ${t_net:+.2f}")
        print(f"  Trimmed PF:           {t_pf:.2f}")
        print(f"  Trimmed Exp/Trade:    ${t_exp:+.2f}")
    else:
        print("  Sample too small for outlier trimming.")

    # ── Milestone Progression & Trajectory ──
    print(f"\n--- Milestone Progression Checkpoints ---")
    milestones = [10, 15, 20, 25, 30]
    next_m = next((m for m in milestones if m > n_trades), 30)
    pct = (n_trades / 30.0) * 100.0
    bar_len = 30
    filled_len = int(bar_len * n_trades // 30)
    bar = "█" * filled_len + "░" * (bar_len - filled_len)
    print(f"  Progress: [{bar}] {n_trades}/30 ({pct:.1f}%)")
    print(f"  Next Milestone: N = {next_m} ({next_m - n_trades} trades to checkpoint)")
    print(f"  Audit Rule: Maintain fixed config across all 30 trades (No population contamination)")

    # ── Promotion Gates Audit ──
    print(f"\n--- Quantitative Promotion Gates (Rule #3 Audit) ---")
    gate_n = n_trades >= 30
    gate_pf = profit_factor >= 1.25 and gross_loss > 0
    gate_wr = win_rate >= 40.0
    gate_dd = max_dd_pct <= 5.0
    gate_slip = avg_slippage <= 0.05

    def format_gate(name, passed, current, target):
        if not gate_n and name != "1. Statistical Validity":
            icon = "[PROVISIONAL]" if passed else "[PENDING]    "
            return f"  {icon} | {name:<24}: {current} (Target: {target})"
        icon = "[PASS]       " if passed else "[PENDING]    "
        return f"  {icon} | {name:<24}: {current} (Target: {target})"

    print(format_gate("1. Statistical Validity", gate_n, f"N = {n_trades}", "N >= 30"))
    print(format_gate("2. Profit Factor", gate_pf, f"PF = {profit_factor:.2f}", "PF >= 1.25"))
    print(format_gate("3. Win Rate Consistency", gate_wr, f"WR = {win_rate:.1f}%", "WR >= 40.0%"))
    print(format_gate("4. Drawdown Tolerance", gate_dd, f"Max DD = {max_dd_pct:.2f}%", "Max DD <= 5.0%"))
    print(format_gate("5. Slippage Control", gate_slip, f"Slippage = {avg_slippage:.4f}%", "Slippage <= 0.05%"))

    all_passed = gate_n and gate_pf and gate_wr and gate_dd and gate_slip
    print("\n" + "=" * 70)
    if all_passed:
        print("  FINAL STATUS: [PROMOTION ELIGIBLE] -- ALL 5 PRODUCTION GATES PASSED")
    else:
        print(f"  FINAL STATUS: [ACCUMULATING FORWARD SAMPLE] -- ({n_trades}/30 trades completed)")
    print("=" * 70)


def main():
    args = parse_args()
    journal_path = Path(args.journal)
    trades = load_forward_trades(journal_path, args.start_date)
    evaluate_cohort(trades)


if __name__ == "__main__":
    main()
