#!/usr/bin/env python3
"""
tools/reconcile_validation_ledger.py -- Historical Ledger Remediation Tool.

Isolates validation lane fills and exits from the production trade journal,
generates a clean production trade journal view, and backfills validation
fills and exits into the paper validation ledger.

Invariants:
  1. The raw original reports/trade_journal.jsonl is NEVER modified in-place
     to preserve forensic audit trails.
  2. A segregated clean view reports/trade_journal_production_clean.jsonl is created.
  3. All matched validation fills, exits, slippage, and P&L are backfilled into
     data/paper_validation_trades.jsonl atomically.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def reconcile_ledgers(
    journal_path: Path,
    validation_path: Path,
    clean_output_path: Path,
    apply: bool = False,
    overwrite: bool = True,
) -> Dict[str, Any]:
    """
    Reconciles journal records and updates validation ledger.

    Returns summary metrics of the reconciliation.
    """
    if not journal_path.exists():
        raise FileNotFoundError(f"Journal not found at {journal_path}")
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation ledger not found at {validation_path}")

    # Load validation ledger records
    val_records: List[Dict[str, Any]] = []
    with open(validation_path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                try:
                    val_records.append(json.loads(line_s))
                except Exception as e:
                    logger.warning("Skipping corrupted validation line: %s", e)

    # Load production journal records
    journal_records: List[Dict[str, Any]] = []
    with open(journal_path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            if line_s:
                try:
                    journal_records.append(json.loads(line_s))
                except Exception as e:
                    logger.warning("Skipping corrupted journal line: %s", e)

    entry_order_map: Dict[str, Dict[str, Any]] = {
        str(r.get("entry_order_id") or "").strip(): r
        for r in val_records
        if r.get("entry_order_id")
    }

    open_positions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    val_entry_fills = 0
    val_exit_fills = 0
    production_records: List[Dict[str, Any]] = []
    segregated_journal_records: List[Dict[str, Any]] = []

    for j in journal_records:
        oid = str(j.get("order_id") or "").strip()
        client_ref = str(j.get("client_order_id") or "").strip()
        ts = str(j.get("timestamp") or "")
        account_id = str(j.get("account_id") or "")
        is_val = False

        if (oid and oid in entry_order_map) or client_ref.startswith("val_"):
            is_val = True
            val_entry_fills += 1
            v = entry_order_map.get(oid)
            if v:
                v["entry_fill_price"] = j.get("fill_price")
                v["entry_price"] = j.get("fill_price")
                v["status"] = "FILLED"
                v["event_type"] = "TRADE"
                if "slippage" in j and j["slippage"] is not None:
                    v["slippage_bps"] = round(float(j["slippage"]) * 10000.0, 2)
                open_positions[str(v.get("symbol") or "").upper()].append(v)
        elif ts.startswith("2026-09-25") and account_id == "PA3UODIC5JTA":
            sym = str(j.get("symbol") or "").upper().strip()
            qty = int(j.get("qty", 1) or 1)
            if sym in open_positions and open_positions[sym]:
                is_val = True
                val_exit_fills += 1
                for _ in range(qty):
                    if open_positions[sym]:
                        v = open_positions[sym].pop(0)
                        v["exit_order_id"] = oid
                        v["exit_fill_price"] = j.get("fill_price")
                        v["exit_price"] = j.get("fill_price")
                        v["status"] = "CLOSED"
                        v["event_type"] = "TRADE_CLOSED"
                        v["exit_reason"] = "BRACKET_EXIT"
                        en_px = float(v.get("entry_fill_price") or v.get("decision_price") or 0.0)
                        ex_px = float(j.get("fill_price") or 0.0)
                        pos_side = str(v.get("side") or "BUY").upper()
                        if en_px > 0:
                            pnl = (ex_px - en_px) if pos_side in ("BUY", "LONG") else (en_px - ex_px)
                            v["realized_pnl"] = round(pnl, 4)
                            v["pnl"] = round(pnl, 4)

        if is_val:
            segregated_journal_records.append(j)
        else:
            production_records.append(j)

    closed_val_trades = [v for v in val_records if v.get("status") == "CLOSED"]
    filled_val_trades = [v for v in val_records if v.get("status") == "FILLED"]
    open_val_trades = [v for v in val_records if v.get("status") in ("SUBMITTED", "FILLED")]

    total_pnl = sum(float(v.get("realized_pnl", 0.0) or 0.0) for v in closed_val_trades)
    winning_trades = [v for v in closed_val_trades if float(v.get("realized_pnl", 0.0) or 0.0) > 0]
    win_rate = (len(winning_trades) / len(closed_val_trades) * 100.0) if closed_val_trades else 0.0

    summary = {
        "journal_total_records": len(journal_records),
        "clean_production_records": len(production_records),
        "segregated_validation_records": len(segregated_journal_records),
        "val_entry_fills": val_entry_fills,
        "val_exit_fills": val_exit_fills,
        "val_total_records": len(val_records),
        "val_closed_trades": len(closed_val_trades),
        "val_open_trades": len(filled_val_trades),
        "val_total_pnl": round(total_pnl, 4),
        "val_win_rate_pct": round(win_rate, 2),
        "apply": apply,
    }

    if apply:
        if clean_output_path.exists() and not overwrite:
            raise FileExistsError(f"Clean output {clean_output_path} already exists and overwrite=False")

        clean_tmp = clean_output_path.with_suffix(".tmp")
        with open(clean_tmp, "w", encoding="utf-8") as f:
            for rec in production_records:
                f.write(json.dumps(rec, default=str) + "\n")
        clean_tmp.replace(clean_output_path)
        logger.info("[RECONCILE APPLIED] Wrote clean production journal: %s", clean_output_path)

        val_tmp = validation_path.with_suffix(".tmp")
        with open(val_tmp, "w", encoding="utf-8") as f:
            for rec in val_records:
                f.write(json.dumps(rec, default=str) + "\n")
        val_tmp.replace(validation_path)
        logger.info("[RECONCILE APPLIED] Updated validation ledger: %s", validation_path)
    else:
        logger.info("[DRY RUN] No files modified. Run with --apply to write outputs.")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Reconcile validation lane entries/exits from production journal.")
    parser.add_argument("--journal", type=str, default="reports/trade_journal.jsonl", help="Path to raw trade journal.")
    parser.add_argument("--validation-ledger", type=str, default="data/paper_validation_trades.jsonl", help="Path to paper validation trades.")
    parser.add_argument("--clean-output", type=str, default="reports/trade_journal_production_clean.jsonl", help="Clean production view path.")
    parser.add_argument("--apply", action="store_true", help="Apply changes and generate files.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite clean output if it exists.")

    args = parser.parse_args()

    summary = reconcile_ledgers(
        journal_path=Path(args.journal),
        validation_path=Path(args.validation_ledger),
        clean_output_path=Path(args.clean_output),
        apply=args.apply,
        overwrite=not args.no_overwrite,
    )

    print("\n" + "=" * 60)
    print("  LEDGER RECONCILIATION SUMMARY")
    print("=" * 60)
    print(f"  Mode:                         {'APPLIED' if summary['apply'] else 'DRY-RUN'}")
    print(f"  Total Journal Records:        {summary['journal_total_records']}")
    print(f"  Preserved Production Records: {summary['clean_production_records']}")
    print(f"  Segregated Validation Fills:  {summary['segregated_validation_records']}")
    print(f"    - Entry Fills:              {summary['val_entry_fills']}")
    print(f"    - Exit Fills:               {summary['val_exit_fills']}")
    print(f"  Validation Ledger Trades:     {summary['val_total_records']}")
    print(f"    - Closed Trades (N):        {summary['val_closed_trades']}")
    print(f"    - Still Open Trades:        {summary['val_open_trades']}")
    print(f"    - Win Rate:                 {summary['val_win_rate_pct']}%")
    print(f"    - Realized Net P&L:         ${summary['val_total_pnl']:+.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
