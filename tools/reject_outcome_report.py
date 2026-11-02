"""
reject_outcome_report.py -- What would rejected (and placed) signals have done?

Answers: "Is gate X saving money or costing money?" using real consolidated 1-minute
bars from yfinance (not IEX), evaluated after the session.

Sources (auto-detected, pick with --source):
  rejections  data/exports/rejections_YYYYMMDD.csv  (written by the patched orchestrator;
              has stage, reason, RSI, gate)
  signals     data/exports/signals_YYYYMMDD.csv     (already written by build 5a38693;
              no stage/RSI -> bucketed by side + confidence band). Works for past sessions.

Simulation (deliberately conservative):
  * Entry = signal entry_price as a limit. Filled only if price trades through it within
    --fill-window-min minutes after the decision (BUY: low <= entry, SELL: high >= entry).
  * After fill, walk 1m bars: stop checked BEFORE target in the same bar (tie = loss).
  * Not resolved by --eod (default 15:55 ET) -> exit at that bar's close.
  * R = PnL / |entry - stop|. No commissions/slippage beyond the tie rule.
  * Repeated signals for the same symbol+side(+stage) within --dedupe-min minutes count once.

Usage (Windows, from the project root):
  python tools\\reject_outcome_report.py --date 20261102 --source signals
  python tools\\reject_outcome_report.py --date 20261103            # uses rejections_ if present

Outputs next to the input: outcome_<source>_YYYYMMDD.csv and outcome_summary_<source>_YYYYMMDD.csv
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from datetime import datetime, timedelta, time as dtime
from typing import Dict, Optional

import pandas as pd

ET_TZ = "America/New_York"


# ── Pure simulation (unit-testable, no network) ──────────────────────────────
def simulate_trade(
    side: str,
    entry: float,
    stop: float,
    target: float,
    bars: pd.DataFrame,
    fill_window_bars: int = 2,
) -> Dict[str, object]:
    """bars: 1m OHLC (columns High/Low/Close), index = bar start, already filtered to
    bars starting at/after the decision time and before EOD."""
    side = side.upper().strip()
    is_long = side in ("BUY", "LONG")
    risk = abs(entry - stop)
    if risk <= 0 or bars is None or bars.empty:
        return {"outcome": "NO_DATA", "r": float("nan"), "fill_time": None, "exit_time": None}

    fill_idx = None
    for i in range(min(fill_window_bars, len(bars))):
        row = bars.iloc[i]
        if (is_long and row["Low"] <= entry) or (not is_long and row["High"] >= entry):
            fill_idx = i
            break
    if fill_idx is None:
        return {"outcome": "NO_FILL", "r": 0.0, "fill_time": None, "exit_time": None}

    reward_r = abs(target - entry) / risk
    for i in range(fill_idx, len(bars)):
        row = bars.iloc[i]
        hit_stop = (row["Low"] <= stop) if is_long else (row["High"] >= stop)
        hit_target = (row["High"] >= target) if is_long else (row["Low"] <= target)
        if hit_stop:
            return {"outcome": "STOP", "r": -1.0, "fill_time": bars.index[fill_idx], "exit_time": bars.index[i]}
        if hit_target:
            return {"outcome": "TARGET", "r": reward_r, "fill_time": bars.index[fill_idx], "exit_time": bars.index[i]}

    last_close = float(bars.iloc[-1]["Close"])
    pnl = (last_close - entry) if is_long else (entry - last_close)
    return {"outcome": "EOD", "r": pnl / risk, "fill_time": bars.index[fill_idx], "exit_time": bars.index[-1]}


def bucket_label(row: pd.Series, source: str, new_gate: float, new_rsi_cap: float) -> str:
    side = str(row.get("side", "")).upper()
    conf = _f(row.get("confidence"))
    if source == "signals":
        if side in ("SELL", "SHORT"):
            return "SELL (all)"
        if conf is None:
            return "BUY conf=?"
        if conf >= 0.60:
            return "BUY conf>=0.60 (passed old gate)"
        if conf >= new_gate:
            return f"BUY {new_gate:.2f}<=conf<0.60 (new gate only)"
        return f"BUY conf<{new_gate:.2f}"

    stage = str(row.get("stage", ""))
    if stage == "STAGE_CONFIDENCE":
        if side in ("SELL", "SHORT"):
            return "CONFIDENCE reject SELL"
        if conf is not None and conf >= new_gate:
            return f"CONFIDENCE reject BUY conf>={new_gate:.2f}"
        return f"CONFIDENCE reject BUY conf<{new_gate:.2f}"
    if stage == "STAGE_RSI_FILTER":
        rsi = _f(row.get("rsi"))
        if rsi is None:
            return "RSI reject (rsi=?)"
        return f"RSI reject {'<' if rsi < new_rsi_cap else '>='}{new_rsi_cap:.0f}"
    return stage or "UNKNOWN"


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


# ── Data loading ─────────────────────────────────────────────────────────────
def load_input(data_dir: str, date_str: str, source: str) -> tuple[pd.DataFrame, str, str]:
    export_dir = os.path.join(data_dir, "exports")
    rej = os.path.join(export_dir, f"rejections_{date_str}.csv")
    sig = os.path.join(export_dir, f"signals_{date_str}.csv")
    if source == "auto":
        source = "rejections" if os.path.exists(rej) else "signals"
    path = rej if source == "rejections" else sig
    if not os.path.exists(path):
        sys.exit(f"Input not found: {path}")
    df = pd.read_csv(path)
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    df["decision_ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_convert(ET_TZ)
    df = df.dropna(subset=["decision_ts"])
    for c in ("entry_price", "stop_price", "target_price"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["entry_price", "stop_price", "target_price"])
    df = df[(df["entry_price"] > 0) & (df["stop_price"] > 0) & (df["target_price"] > 0)]
    return df.sort_values("decision_ts").reset_index(drop=True), source, export_dir


def dedupe(df: pd.DataFrame, minutes: int, source: str) -> pd.DataFrame:
    keys = ["symbol", "side"] + (["stage"] if source == "rejections" and "stage" in df.columns else [])
    keep, last_seen = [], {}
    for idx, row in df.iterrows():
        k = tuple(row[c] for c in keys)
        prev = last_seen.get(k)
        if prev is None or (row["decision_ts"] - prev) >= timedelta(minutes=minutes):
            keep.append(idx)
            last_seen[k] = row["decision_ts"]
    return df.loc[keep].reset_index(drop=True)


_bar_cache: Dict[str, pd.DataFrame] = {}


def get_bars(symbol: str, day: datetime) -> pd.DataFrame:
    if symbol in _bar_cache:
        return _bar_cache[symbol]
    import yfinance as yf  # consolidated prices; 1m history covers roughly the last 30 days
    start = day.strftime("%Y-%m-%d")
    end = (day + timedelta(days=1)).strftime("%Y-%m-%d")
    bars = yf.Ticker(symbol).history(start=start, end=end, interval="1m", prepost=False, auto_adjust=False)
    if not bars.empty:
        bars.index = bars.index.tz_convert(ET_TZ)
    _bar_cache[symbol] = bars
    return bars


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="ET session date YYYYMMDD")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--source", choices=["auto", "rejections", "signals"], default="auto")
    ap.add_argument("--fill-window-min", type=int, default=2)
    ap.add_argument("--dedupe-min", type=int, default=45, help="match REENTRY_COOLDOWN_MINUTES")
    ap.add_argument("--eod", default="15:55", help="ET time to force-exit open simulations")
    ap.add_argument("--new-gate", type=float, default=0.58)
    ap.add_argument("--new-rsi-cap", type=float, default=75.0)
    args = ap.parse_args()

    day = datetime.strptime(args.date, "%Y%m%d")
    eod_t = dtime.fromisoformat(args.eod)
    df, source, out_dir = load_input(args.data_dir, args.date, args.source)
    df = dedupe(df, args.dedupe_min, source)
    print(f"[{source}] {len(df)} unique signals after {args.dedupe_min}m de-dup")

    rows = []
    for _, r in df.iterrows():
        sym = str(r["symbol"]).upper().strip()
        try:
            bars = get_bars(sym, day)
        except Exception as exc:
            print(f"  {sym}: bar download failed ({exc})")
            bars = pd.DataFrame()
        if not bars.empty:
            start = r["decision_ts"].ceil("min")
            eod_ts = start.normalize() + timedelta(hours=eod_t.hour, minutes=eod_t.minute)
            window = bars[(bars.index >= start) & (bars.index <= eod_ts)]
        else:
            window = bars
        sim = simulate_trade(str(r["side"]), float(r["entry_price"]), float(r["stop_price"]),
                             float(r["target_price"]), window, args.fill_window_min)
        out = r.to_dict()
        out.update(sim)
        out["bucket"] = bucket_label(r, source, args.new_gate, args.new_rsi_cap)
        rows.append(out)

    res = pd.DataFrame(rows)
    if res.empty:
        sys.exit("No rows to evaluate.")
    filled = res[res["outcome"].isin(["STOP", "TARGET", "EOD"])]
    summary = (
        res.groupby("bucket")
        .agg(signals=("outcome", "size"),
             no_fill=("outcome", lambda s: int((s == "NO_FILL").sum())),
             no_data=("outcome", lambda s: int((s == "NO_DATA").sum())))
        .join(filled.groupby("bucket").agg(
            filled=("r", "size"),
            win_rate=("r", lambda s: float((s > 0).mean())),
            avg_r=("r", "mean"),
            total_r=("r", "sum")), how="left")
        .fillna({"filled": 0})
        .sort_values("total_r", ascending=False)
    )

    detail_path = os.path.join(out_dir, f"outcome_{source}_{args.date}.csv")
    summary_path = os.path.join(out_dir, f"outcome_summary_{source}_{args.date}.csv")
    res.to_csv(detail_path, index=False)
    summary.to_csv(summary_path)
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summary.round(3).to_string())
    print(f"\nDetail : {detail_path}\nSummary: {summary_path}")
    print("Small samples mislead: wait for 30+ filled signals per bucket before changing a gate.")


if __name__ == "__main__":
    main()
