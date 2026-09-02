# analyze_missed_opportunities.py
"""
TRANSITIONING texture ကြောင့် Reject ဖြစ်သော Signal များသည်
တကယ်သာ Trade ဝင်ခဲ့ပါက TP ထိမည်လား SL ထိမည်လားကို
Historical Intraday Candles ဖြင့် Walk-Forward Simulation ပြုလုပ်ပေးသော Script
"""

import os
import glob
import re
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

LOG_FILES = sorted(glob.glob("logs/stock_engine.log*") + glob.glob("logs/daytrade_current.log"))

REJECT_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Skipped (\w+): strategy (\S+) not allowed in market texture 'TRANSITIONING'"
)

SIZER_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[SIZER\] (\w+) \| side=(\w+) entry=([\d.]+) stop=([\d.]+)"
)

def load_sizer_map():
    sizer_map = defaultdict(list)
    for log_path in LOG_FILES:
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = SIZER_PATTERN.search(line)
                    if m:
                        ts, sym, side, entry, stop = m.groups()
                        try:
                            sizer_map[sym].append({
                                "ts": datetime.fromisoformat(ts),
                                "side": side,
                                "entry": float(entry),
                                "stop": float(stop),
                            })
                        except Exception:
                            pass
        except Exception:
            pass
    return sizer_map

def extract_rejected_signals():
    sizer_map = load_sizer_map()
    rejected_signals = []

    for log_path in LOG_FILES:
        if not os.path.exists(log_path):
            continue
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m = REJECT_PATTERN.search(line)
                    if not m:
                        continue
                    try:
                        reject_ts = datetime.fromisoformat(m.group(1))
                    except Exception:
                        continue
                    symbol = m.group(2)
                    strategy = m.group(3)

                    # Find nearest sizer entry
                    best_sizer = None
                    for s in sizer_map.get(symbol, []):
                        diff = abs((s["ts"] - reject_ts).total_seconds())
                        if diff <= 60:
                            best_sizer = s
                            break

                    if not best_sizer:
                        continue

                    entry = best_sizer["entry"]
                    stop = best_sizer["stop"]
                    side = best_sizer["side"]

                    risk = abs(entry - stop)
                    if risk <= 0:
                        continue
                    target = entry + (risk * 2.0) if side == "BUY" else entry - (risk * 2.0)

                    rejected_signals.append({
                        "ts": reject_ts,
                        "symbol": symbol,
                        "strategy": strategy,
                        "side": side,
                        "entry": entry,
                        "stop": stop,
                        "target": round(target, 2),
                        "risk": risk,
                    })
        except Exception:
            pass

    return rejected_signals

def run_intraday_walkforward():
    signals = extract_rejected_signals()
    if not signals:
        print("No rejected signals found.")
        return

    # Deduplicate signals by (symbol, timestamp rounded to 5m, side)
    unique_signals = []
    seen = set()
    for s in signals:
        key = (s["symbol"], s["ts"].strftime("%Y-%m-%d %H:%M"), s["side"])
        if key not in seen:
            seen.add(key)
            unique_signals.append(s)

    print(f"\nEvaluating {len(unique_signals)} unique rejected signals with Forward Candle Simulation...\n")

    # Group by symbol to batch fetch bars
    by_symbol = defaultdict(list)
    for s in unique_signals:
        by_symbol[s["symbol"]].append(s)

    results = []

    for sym, sym_signals in by_symbol.items():
        try:
            # Download 5m candles for the past 10 days
            df = yf.download(sym, period="10d", interval="5m", progress=False, auto_adjust=False)
            if df.empty:
                continue

            # Flatten multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            # Ensure tz-naive datetime index in local session
            if df.index.tz is not None:
                # Convert to US/Eastern or naive
                df.index = df.index.tz_convert("America/New_York").tz_localize(None)

            for sig in sym_signals:
                entry_ts = sig["ts"]
                entry_px = sig["entry"]
                stop_px = sig["stop"]
                target_px = sig["target"]
                side = sig["side"]

                # Convert entry_ts (ICT / UTC+7) to ET (UTC-4) -> subtract 11 hours
                # In EDT: ICT is UTC+7, EDT is UTC-4 -> diff is 11 hours
                et_entry = entry_ts - timedelta(hours=11)

                # Slice forward bars on the same day after entry
                forward_bars = df[(df.index >= et_entry) & (df.index.date == et_entry.date())]

                if forward_bars.empty:
                    # Try finding closest bar
                    continue

                outcome = "EOD_FLAT"
                exit_price = float(forward_bars["close"].iloc[-1])
                exit_reason = "EOD_FLATTEN"

                for bar_idx, bar in forward_bars.iterrows():
                    high = float(bar["high"])
                    low = float(bar["low"])

                    if side == "BUY":
                        # Check stop hit first (conservative: worst-case check)
                        if low <= stop_px:
                            outcome = "SL_HIT"
                            exit_price = stop_px
                            exit_reason = "STOP_LOSS"
                            break
                        elif high >= target_px:
                            outcome = "TP_HIT"
                            exit_price = target_px
                            exit_reason = "TAKE_PROFIT"
                            break
                    else:  # SELL
                        if high >= stop_px:
                            outcome = "SL_HIT"
                            exit_price = stop_px
                            exit_reason = "STOP_LOSS"
                            break
                        elif low <= target_px:
                            outcome = "TP_HIT"
                            exit_price = target_px
                            exit_reason = "TAKE_PROFIT"
                            break

                pnl_pct = ((exit_price - entry_px) / entry_px * 100.0) if side == "BUY" else ((entry_px - exit_price) / entry_px * 100.0)

                results.append({
                    "ts": entry_ts.strftime("%Y-%m-%d %H:%M"),
                    "symbol": sym,
                    "side": side,
                    "entry": entry_px,
                    "stop": stop_px,
                    "target": target_px,
                    "outcome": outcome,
                    "exit_price": round(exit_price, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "exit_reason": exit_reason,
                })
        except Exception as e:
            print(f"Error evaluating {sym}: {e}")

    tp_hits = [r for r in results if r["outcome"] == "TP_HIT"]
    sl_hits = [r for r in results if r["outcome"] == "SL_HIT"]
    eod_flats = [r for r in results if r["outcome"] == "EOD_FLAT"]
    total = len(results)

    print(f"\n{'='*70}")
    print(f"  MISSED OPPORTUNITY SIMULATION REPORT (TRANSITIONING REJECTS)")
    print(f"{'='*70}")
    print(f"  Total Simulated Signals : {total}")
    if total > 0:
        print(f"  TP Hit (Profitable)     : {len(tp_hits):>3} ({len(tp_hits)/total*100:>5.1f}%)")
        print(f"  SL Hit (Losses Prevented): {len(sl_hits):>3} ({len(sl_hits)/total*100:>5.1f}%)")
        print(f"  EOD Flatten (Chop/No TP): {len(eod_flats):>3} ({len(eod_flats)/total*100:>5.1f}%)")

        win_loss_total = len(tp_hits) + len(sl_hits)
        if win_loss_total > 0:
            effective_wr = len(tp_hits) / win_loss_total * 100
            print(f"  Effective Win Rate      : {effective_wr:>5.1f}% (TP vs SL)")

        avg_pnl = sum(r["pnl_pct"] for r in results) / total
        print(f"  Average Simulated PnL   : {avg_pnl:>+5.2f}%")

    print(f"\n  {'Time':<17} {'Sym':<6} {'Side':<5} {'Entry':>7} {'Stop':>7} {'Target':>8} {'Outcome':<10} {'Exit Px':>8} {'PnL%':>7}")
    print(f"  {'-'*78}")
    for r in sorted(results, key=lambda x: x["ts"])[-30:]:
        print(f"  {r['ts']:<17} {r['symbol']:<6} {r['side']:<5} "
              f"{r['entry']:>7.2f} {r['stop']:>7.2f} {r['target']:>8.2f} "
              f"{r['outcome']:<10} {r['exit_price']:>8.2f} {r['pnl_pct']:>+6.2f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    run_intraday_walkforward()
