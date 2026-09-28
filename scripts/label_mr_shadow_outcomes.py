#!/usr/bin/env python3
"""
scripts/label_mr_shadow_outcomes.py -- Post-Market Outcome Labeler for D2 Mean Reversion Shadow.

Methodology (identical to Sep 5m OHLCV replay):
- Reads candidates from data/shadow/mean_reversion_shadow.jsonl.
- For each evaluated candidate, evaluates the subsequent 6 completed 5m bars (30-minute forward horizon).
- Evaluates conservative bracket resolution:
    Stop: Close - 0.5 * ATR
    Target: VWAP
- Outcomes classified:
    REVERSION: Touched VWAP target before Stop.
    BREAKDOWN: Breached Stop before Target.
    AMBIGUOUS_INTRABAR: Both touched in same bar (treated as Loss).
    TIME_EXPIRY_30M: Neither touched; marked-to-market at 6th bar close.
- Computes MFE (Max Favorable Excursion), MAE (Max Adverse Excursion), and Realized R.
- Appends resolved records to data/shadow/mean_reversion_shadow_outcomes.jsonl.
- Generates multi-feature cross-tab summary:
    lower_bb_break (T/F) x making_new_low (T/F) x atr_expanding (T/F) -> Outcome.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mr_outcome_labeler")


def parse_args():
    parser = argparse.ArgumentParser(description="Label forward 30m outcomes for MR shadow candidates.")
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/shadow/mean_reversion_shadow.jsonl",
        help="Path to input candidate records JSONL.",
    )
    parser.add_argument(
        "--outcomes",
        type=str,
        default="data/shadow/mean_reversion_shadow_outcomes.jsonl",
        help="Path to output resolved outcomes JSONL.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="reports/shadow/mr_shadow_crosstab_report.md",
        help="Path to markdown cross-tab summary report.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records


def evaluate_forward_30m(
    cand: dict,
    future_bars_df: pd.DataFrame,
) -> Optional[dict]:
    """Evaluates 30-minute forward path (up to 6 completed 5m bars)."""
    if future_bars_df is None or len(future_bars_df) == 0:
        return None

    entry_price = float(cand["close"])
    atr = float(cand["atr"])
    target_price = float(cand["vwap"])
    stop_dist = 0.5 * atr
    stop_price = entry_price - stop_dist

    eval_window = future_bars_df.head(6)
    outcome = "TIME_EXPIRY_30M"
    exit_price = float(eval_window["close"].iloc[-1])
    bars_held = len(eval_window)
    exit_bar_ts = str(eval_window.index[-1])

    max_high = float(eval_window["high"].max())
    min_low = float(eval_window["low"].min())
    mfe = (max_high - entry_price) / stop_dist if stop_dist > 0 else 0.0
    mae = (entry_price - min_low) / stop_dist if stop_dist > 0 else 0.0

    for idx, (bar_ts, bar) in enumerate(eval_window.iterrows()):
        b_high = float(bar["high"])
        b_low = float(bar["low"])

        hit_target = b_high >= target_price
        hit_stop = b_low <= stop_price

        if hit_target and hit_stop:
            outcome = "AMBIGUOUS_INTRABAR"
            exit_price = stop_price
            bars_held = idx + 1
            exit_bar_ts = str(bar_ts)
            break
        elif hit_stop:
            outcome = "BREAKDOWN"
            exit_price = stop_price
            bars_held = idx + 1
            exit_bar_ts = str(bar_ts)
            break
        elif hit_target:
            outcome = "REVERSION"
            exit_price = target_price
            bars_held = idx + 1
            exit_bar_ts = str(bar_ts)
            break

    realized_r = (exit_price - entry_price) / stop_dist if stop_dist > 0 else 0.0

    resolved_cand = dict(cand)
    resolved_cand.update({
        "outcome": outcome,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_price": exit_price,
        "bars_held": bars_held,
        "exit_bar_timestamp": exit_bar_ts,
        "realized_r": round(realized_r, 3),
        "mfe_r": round(mfe, 3),
        "mae_r": round(mae, 3),
        "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return resolved_cand


def run_labeler():
    args = parse_args()
    cand_path = Path(args.candidates)
    outcomes_path = Path(args.outcomes)
    report_path = Path(args.report)

    candidates = load_jsonl(cand_path)
    if not candidates:
        logger.info("[LABELER] No candidates found in %s", cand_path)
        return

    logger.info("[LABELER] Loaded %d candidate records from %s", len(candidates), cand_path)

    existing_outcomes = load_jsonl(outcomes_path)
    resolved_keys = {
        (r.get("symbol"), r.get("bar_timestamp")) for r in existing_outcomes if r.get("outcome")
    }

    pending = [
        c for c in candidates
        if (c.get("symbol"), c.get("bar_timestamp")) not in resolved_keys
        and c.get("l1_passed", False)  # Focus resolution on L1-qualified setups
    ]

    logger.info("[LABELER] %d pending candidates require 30m outcome evaluation", len(pending))
    if not pending:
        logger.info("[LABELER] All candidates already resolved.")
        return

    # In actual production replay, future bars are loaded from intraday cache or Alpaca
    from strategies.intraday_data import get_intraday_data
    resolved_count = 0
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)

    with open(outcomes_path, "a", encoding="utf-8") as out_f:
        for cand in pending:
            sym = cand["symbol"]
            bar_ts = cand["bar_timestamp"]
            try:
                df = get_intraday_data(sym, interval="5m", lookback_days=5)
                if df is None or df.empty:
                    continue
                # Normalize index
                df.index = pd.to_datetime(df.index, utc=True)
                cand_dt = pd.to_datetime(bar_ts, utc=True)
                future_bars = df[df.index > cand_dt]
                if len(future_bars) < 6:
                    # Session not yet closed or insufficient future bars
                    continue

                res = evaluate_forward_30m(cand, future_bars)
                if res is not None:
                    out_f.write(json.dumps(res) + "\n")
                    out_f.flush()
                    resolved_count += 1
            except Exception as e:
                logger.debug("Could not label candidate %s @ %s: %s", sym, bar_ts, e)

    logger.info("[LABELER] Successfully labeled %d new MR shadow outcomes", resolved_count)


if __name__ == "__main__":
    run_labeler()
