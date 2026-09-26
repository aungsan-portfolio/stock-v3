#!/usr/bin/env python3
"""
tools/backfill_training_signals.py

Bootstrap ML training data from existing trade journal records.
Converts historical closed trades into synthetic signal feature rows
so that `python main.py daytrade-train` can run immediately.

Usage:
    python tools/backfill_training_signals.py
    python tools/backfill_training_signals.py --journal reports/trade_journal.jsonl
    python tools/backfill_training_signals.py --out data/exports/signals_synthetic.csv
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import pytz
    _ET = pytz.timezone("America/New_York")
except ImportError:
    _ET = None

SYNTHETIC_COLUMNS = [
    "timestamp",
    "timestamp_et",
    "symbol",
    "trigger_price",
    "entry_price",
    "stop_price",
    "target_price",
    "side",
    "confidence",
    "raw_confidence",
    "correlation_penalty",
    "max_correlation",
    "risk_per_share",
    "rr_ratio",
    "regime_score",
    "regime_mode",
    "regime_texture",
    "spy_pct",
    "qqq_pct",
    "soxx_pct",
    "sizing_pct",
    "halt_entries",
    "market_texture",
    "would_have_entered",
    "signal_type",
    "strategy",
    "signal_id",
    # Training label
    "outcome_profitable",   # 1 = profitable trade, 0 = loss
    "outcome_pnl_pct",      # realized PnL % for label strength
]


def _parse_args():
    p = argparse.ArgumentParser(description="Backfill training signals from journal.")
    p.add_argument(
        "--journal",
        default="reports/trade_journal.jsonl",
        help="Path to trade_journal.jsonl",
    )
    p.add_argument(
        "--out",
        default="data/exports/signals_synthetic.csv",
        help="Output CSV path",
    )
    p.add_argument(
        "--min-records",
        type=int,
        default=30,
        help="Warn if fewer than N records found",
    )
    return p.parse_args()


def load_journal_records(journal_path: str) -> list[dict]:
    """Load all JSONL records from the trade journal."""
    path = Path(journal_path)
    if not path.exists():
        print(f"[ERROR] Journal not found: {journal_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] Line {lineno} skipped (JSON error): {exc}")

    print(f"[INFO] Loaded {len(records)} raw journal records from {journal_path}")
    return records


def journal_record_to_signal_row(rec: dict, idx: int = 0) -> dict | None:
    """
    Convert one journal record into a synthetic signal feature row.
    Returns None if the record lacks required fields.
    """
    # Accept TRADE_CLOSED, EXIT, CLOSE, or records with explicit realized_pnl
    event_type = str(rec.get("event_type", rec.get("type", ""))).upper()
    has_pnl = rec.get("realized_pnl") is not None or rec.get("pnl") is not None
    is_closed = (
        event_type in ("TRADE_CLOSED", "EXIT", "CLOSE", "FILL_EXIT", "STOP_TRIGGERED",
                       "EMERGENCY_FLATTEN", "TARGET_HIT", "LOSS_EXIT", "PROFIT_EXIT")
        or (has_pnl and rec.get("exit_price") is not None)
    )

    if not is_closed:
        return None

    entry_px = rec.get("entry_price") or rec.get("avg_entry_price") or rec.get("fill_price")
    exit_px  = rec.get("exit_price")  or rec.get("avg_exit_price")
    symbol   = rec.get("symbol")

    if not all([entry_px, exit_px, symbol]):
        return None

    try:
        entry_px = float(entry_px)
        exit_px  = float(exit_px)
    except (TypeError, ValueError):
        return None

    if entry_px <= 0:
        return None

    side = str(rec.get("side", "BUY")).upper()
    if side not in ("BUY", "SELL"):
        side = "BUY"

    if side == "BUY":
        pnl_pct = (exit_px - entry_px) / entry_px * 100.0
    else:
        pnl_pct = (entry_px - exit_px) / entry_px * 100.0

    raw_pnl = rec.get("realized_pnl")
    if raw_pnl is None:
        raw_pnl = rec.get("pnl")

    if raw_pnl is not None:
        outcome_profit = 1 if float(raw_pnl) > 0 else 0
    else:
        outcome_profit = 1 if pnl_pct > 0 else 0

    # Stop price and target price synthesis if missing
    stop_px = float(rec.get("stop_price") or 0.0)
    target_px = float(rec.get("target_price") or 0.0)
    if stop_px <= 0.0:
        stop_px = round(entry_px * 0.99 if side == "BUY" else entry_px * 1.01, 2)
    if target_px <= 0.0:
        target_px = round(entry_px * 1.02 if side == "BUY" else entry_px * 0.98, 2)

    risk_per_share = abs(entry_px - stop_px)
    dist_target = abs(target_px - entry_px)
    rr_ratio = round(dist_target / risk_per_share, 2) if risk_per_share > 1e-6 else 2.0

    raw_ts = rec.get("opened_at") or rec.get("timestamp") or rec.get("ts", "")
    ts_et = raw_ts
    if _ET and raw_ts:
        try:
            dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            ts_et = dt.astimezone(_ET).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    strat = rec.get("strategy") or rec.get("signal_type") or "DAYTRADE_SCALP"
    if str(strat).strip() == "UNKNOWN":
        strat = "DAYTRADE_SCALP"

    sig_id = rec.get("signal_id") or f"synth_sig_{idx:04d}"

    return {
        "timestamp":           raw_ts,
        "timestamp_et":        ts_et,
        "symbol":              symbol.upper().strip(),
        "trigger_price":       entry_px,
        "entry_price":         entry_px,
        "stop_price":          stop_px,
        "target_price":        target_px,
        "side":                side,
        "confidence":          0.70,
        "raw_confidence":      0.70,
        "correlation_penalty": 1.0,
        "max_correlation":     0.0,
        "risk_per_share":      round(risk_per_share, 4),
        "rr_ratio":            rr_ratio,
        "regime_score":        rec.get("regime_score", -1),
        "regime_mode":         rec.get("regime_mode",  "CAUTION"),
        "regime_texture":      rec.get("regime_texture", "TRENDING"),
        "spy_pct":             rec.get("spy_pct",  -0.50),
        "qqq_pct":             rec.get("qqq_pct",  -0.70),
        "soxx_pct":            rec.get("soxx_pct", -1.00),
        "sizing_pct":          rec.get("sizing_pct", "1.00"),
        "halt_entries":        rec.get("halt_entries", False),
        "market_texture":      rec.get("market_texture", "TRENDING"),
        "would_have_entered":  True,    # Historical trades all entered
        "signal_type":         strat,
        "strategy":            strat,
        "signal_id":           sig_id,
        "outcome_profitable":  outcome_profit,
        "outcome_pnl_pct":     round(pnl_pct, 4),
    }


def write_synthetic_csv(rows: list[dict], out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SYNTHETIC_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Wrote {len(rows)} synthetic signal rows -> {out_path}")


def main() -> None:
    args    = _parse_args()
    records = load_journal_records(args.journal)
    rows    = []
    for idx, rec in enumerate(records, 1):
        row = journal_record_to_signal_row(rec, idx=idx)
        if row:
            rows.append(row)

    if not rows:
        print("[ERROR] No convertible records found. "
              "Ensure journal contains closed trades.", file=sys.stderr)
        sys.exit(1)

    if len(rows) < args.min_records:
        print(f"[WARN] Only {len(rows)} rows generated "
              f"(--min-records={args.min_records}). "
              f"Model quality may be low.")

    write_synthetic_csv(rows, args.out)
    print(f"[DONE] Signal backfill complete. Run: python main.py daytrade-train")


if __name__ == "__main__":
    main()
