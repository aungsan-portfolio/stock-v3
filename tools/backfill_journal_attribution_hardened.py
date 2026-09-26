"""
tools/backfill_journal_attribution_hardened.py
Hardened forensic attribution recovery tool.
- Supports --dry-run (Report-Only) mode.
- Strict 6-field mock contract fingerprint for quarantine.
- Supports se3- client_order_id prefix parsing (Attempt 0).
- Supports direct order_id and session-date proximity matching (Attempt 1 & 2).
- Supports July 22 archive lookup (Attempt 3).
- Preserves raw source files with cryptographic audit integrity.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

MOCK_CONTRACT = {
    "symbol": "AAPL",
    "qty": 100.0,
    "entry_price": 150.0,
    "exit_price": 140.0,
    "pnl": -1000.0,
    "notes": "Emergency flatten exit",
}

STRATEGY_PREFIX_MAP = {
    "mome": "MOMENTUM_SCALP",
    "vwap": "VWAP_BOUNCE",
    "gapa": "GAP_AND_GO",
    "gap": "GAP_AND_GO",
    "orb": "ORB",
    "cand": "CANDLESTICK",
    "mean": "MEAN_REVERSION",
}


def is_synthetic_leak(rec) -> bool:
    """Exact 6-field mock contract fingerprint validation."""
    try:
        pnl = rec.get("realized_pnl") if rec.get("realized_pnl") is not None else rec.get("pnl")
        if pnl is None:
            return False
        return (
            rec.get("symbol") == MOCK_CONTRACT["symbol"]
            and float(rec.get("qty", 0.0) or 0.0) == MOCK_CONTRACT["qty"]
            and float(rec.get("entry_price", 0.0) or 0.0) == MOCK_CONTRACT["entry_price"]
            and float(rec.get("exit_price", 0.0) or 0.0) == MOCK_CONTRACT["exit_price"]
            and float(pnl) == MOCK_CONTRACT["pnl"]
            and str(rec.get("notes") or "").strip() == MOCK_CONTRACT["notes"]
        )
    except (TypeError, ValueError):
        return False


def parse_strategy_from_client_id(cid: str) -> str:
    """Extract strategy from se3-{SYMBOL}-{SIDE}-{STRAT}-{DATE}-{DIGEST}."""
    if not cid or not isinstance(cid, str):
        return "UNKNOWN"
    parts = cid.split("-")
    if len(parts) >= 6 and parts[0] == "se3":
        code = parts[3].lower().strip()
        return STRATEGY_PREFIX_MAP.get(code, "UNKNOWN")
    return "UNKNOWN"


def parse_iso(ts_str):
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except Exception:
        return None


def run_attribution_pipeline(
    journal_path: str = "reports/trade_journal_production_clean.jsonl",
    gate_path: str = "data/trade_gate_decision.jsonl",
    archive_jul22_path: str = "reports/trade_journal_archive_jul22.jsonl",
    output_path: str = "reports/trade_journal_attribution_recovered.jsonl",
    dry_run: bool = True,
):
    print("=" * 75)
    print(f"  FORENSIC ATTRIBUTION RECOVERY & AUDIT (Mode: {'REPORT-ONLY (DRY-RUN)' if dry_run else 'APPLY-WRITE'})")
    print("=" * 75)

    # Source File Cryptographic Verification
    src_bytes = Path(journal_path).read_bytes()
    src_sha256 = hashlib.sha256(src_bytes).hexdigest()
    print(f"[*] Source Journal Path:    {journal_path}")
    print(f"[*] Source SHA-256 Digest:  {src_sha256}")
    print(f"[*] Source File Size:       {len(src_bytes):,} bytes")

    # 1. Index GATE PASS decisions
    pass_by_sym_date = defaultdict(list)
    pass_by_order_id = {}

    if Path(gate_path).exists():
        with open(gate_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("decision") == "PASS":
                    dt = parse_iso(rec.get("timestamp"))
                    sym = rec.get("symbol", "").upper().strip()
                    strat = rec.get("strategy", "").replace("StrategyName.", "").strip()
                    meta = rec.get("metadata") or {}
                    parent_oid = meta.get("parent_order_id") or rec.get("order_id")
                    item = {
                        "symbol": sym,
                        "strategy": strat,
                        "timestamp": dt,
                        "date": dt.strftime("%Y-%m-%d") if dt else "",
                        "entry_price": meta.get("entry_price") or meta.get("price"),
                        "shares": meta.get("shares"),
                        "parent_order_id": parent_oid,
                    }
                    if parent_oid:
                        pass_by_order_id[parent_oid] = item
                    if dt and sym:
                        pass_by_sym_date[(sym, dt.strftime("%Y-%m-%d"))].append(item)

        print(f"[*] Indexed {len(pass_by_order_id)} PASS order IDs, {len(pass_by_sym_date)} (symbol, date) gate buckets.")

    # 2. Index July 22 Archive by symbol, entry_price, qty for legacy trades
    archive_jul22_map = {}
    if Path(archive_jul22_path).exists():
        with open(archive_jul22_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pnl = rec.get("realized_pnl") if rec.get("realized_pnl") is not None else rec.get("pnl")
                strat = rec.get("strategy")
                if pnl is not None and strat and strat != "UNKNOWN":
                    sym = rec.get("symbol", "").upper().strip()
                    entry_p = float(rec.get("entry_price", 0.0) or 0.0)
                    qty = float(rec.get("qty", 0.0) or 0.0)
                    archive_jul22_map[(sym, round(entry_p, 2), round(qty, 2))] = strat

        print(f"[*] Indexed {len(archive_jul22_map)} non-UNKNOWN historical archive trades from {archive_jul22_path}.")

    # 3. Process Journal
    total_records = 0
    quarantined_count = 0
    recovered_by_client_id = 0
    recovered_by_order_id = 0
    recovered_by_gate_session = 0
    recovered_by_archive = 0
    already_attributed = 0
    unresolved_count = 0

    output_lines = []
    audit_log = []

    with open(journal_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            total_records += 1
            rec = json.loads(line)

            pnl = rec.get("realized_pnl") if rec.get("realized_pnl") is not None else rec.get("pnl")
            sym = rec.get("symbol", "").upper().strip()
            strat = rec.get("strategy") or rec.get("strategy_name") or "UNKNOWN"
            strat = str(strat).replace("StrategyName.", "")
            order_id = rec.get("order_id")
            cid = rec.get("client_order_id")
            j_ts = parse_iso(rec.get("timestamp"))
            j_date = j_ts.strftime("%Y-%m-%d") if j_ts else ""
            entry_price = float(rec.get("entry_price", 0.0) or 0.0)
            qty = float(rec.get("qty", 0.0) or 0.0)

            # Check Strict Quarantine Fingerprint
            if is_synthetic_leak(rec):
                rec["is_quarantined"] = True
                rec["quarantine_reason"] = "SYNTHETIC_TEST_LEAK_CONFIRMED"
                rec["quarantine_contract"] = MOCK_CONTRACT
                quarantined_count += 1
                output_lines.append(rec)
                audit_log.append(
                    f"  [QUARANTINE] Line {line_no:4d}: {sym} on {j_date} (PnL={pnl}, Qty={qty}, Entry={entry_price}, Exit={rec.get('exit_price')}) -> QUARANTINED"
                )
                continue

            rec["is_quarantined"] = False

            # Closed Trade Attribution Recovery
            if pnl is not None:
                if strat not in ("UNKNOWN", "RECONCILED", None, ""):
                    already_attributed += 1
                    output_lines.append(rec)
                    continue

                # Attempt 0: Parse se3- client_order_id prefix
                cid_strat = parse_strategy_from_client_id(cid)
                if cid_strat != "UNKNOWN":
                    rec["strategy"] = cid_strat
                    rec["attribution_recovered"] = True
                    rec["attribution_method"] = "CLIENT_ORDER_ID_PREFIX"
                    recovered_by_client_id += 1
                    output_lines.append(rec)
                    audit_log.append(f"  [RECOVER:CID] Line {line_no:4d}: {sym} on {j_date} (cid={cid}) -> {cid_strat}")
                    continue

                # Attempt 1: Direct order_id match from gate PASS
                if order_id and order_id in pass_by_order_id:
                    matched = pass_by_order_id[order_id]
                    rec["strategy"] = matched["strategy"]
                    rec["attribution_recovered"] = True
                    rec["attribution_method"] = "DIRECT_ORDER_ID"
                    recovered_by_order_id += 1
                    output_lines.append(rec)
                    audit_log.append(f"  [RECOVER:ORDER_ID] Line {line_no:4d}: {sym} on {j_date} -> {matched['strategy']}")
                    continue

                # Attempt 2: Gate Session Date + Symbol match
                bucket = pass_by_sym_date.get((sym, j_date), [])
                if bucket:
                    best_match = None
                    min_diff = float("inf")
                    for cand in bucket:
                        if cand["entry_price"] and entry_price:
                            diff = abs(float(cand["entry_price"]) - float(entry_price))
                        else:
                            diff = abs((cand["timestamp"] - j_ts).total_seconds()) if cand["timestamp"] and j_ts else 0.0
                        if diff < min_diff:
                            min_diff = diff
                            best_match = cand

                    if best_match and min_diff < 5.0:  # within $5.00 price variance
                        rec["strategy"] = best_match["strategy"]
                        rec["attribution_recovered"] = True
                        rec["attribution_method"] = "SESSION_DATE_TIEBREAK"
                        recovered_by_gate_session += 1
                        output_lines.append(rec)
                        audit_log.append(f"  [RECOVER:GATE_SESSION] Line {line_no:4d}: {sym} on {j_date} -> {best_match['strategy']}")
                        continue

                # Attempt 3: Archive July 22 lookup by (symbol, entry_price, qty)
                archive_key = (sym, round(entry_price, 2), round(qty, 2))
                if archive_key in archive_jul22_map:
                    arch_strat = archive_jul22_map[archive_key]
                    rec["strategy"] = arch_strat
                    rec["attribution_recovered"] = True
                    rec["attribution_method"] = "ARCHIVE_EXACT_MATCH"
                    recovered_by_archive += 1
                    output_lines.append(rec)
                    audit_log.append(f"  [RECOVER:ARCHIVE] Line {line_no:4d}: {sym} on {j_date} (Entry={entry_price}, Qty={qty}) -> {arch_strat}")
                    continue

                unresolved_count += 1
                audit_log.append(f"  [UNRESOLVED] Line {line_no:4d}: {sym} on {j_date} (PnL={pnl:.2f}, Qty={qty}) -> Remained UNKNOWN")

            output_lines.append(rec)

    # 4. Print Forensic Checklist Summary
    print("\n" + "=" * 75)
    print("  DRY-RUN FORENSIC CHECKLIST REVIEW")
    print("=" * 75)
    print(f"Total Journal Records Evaluated:           {total_records}")
    print(f"Quarantined Synthetic Records:             {quarantined_count}  (Expected: exactly 2)")
    print(f"Already Correctly Attributed Trades:       {already_attributed}")
    print(f"Recovered via se3- Client Order ID:        {recovered_by_client_id}")
    print(f"Recovered via Gate Direct Order ID:        {recovered_by_order_id}")
    print(f"Recovered via Gate Session Date Proximity: {recovered_by_gate_session}")
    print(f"Recovered via Historical July 22 Archive:  {recovered_by_archive}")
    print(f"Remaining Unresolved UNKNOWN Trades:       {unresolved_count}")
    print("-" * 75)

    print("\nDetailed Audit Trail Entries:")
    for entry in audit_log:
        print(entry)

    # Verify Source Immutability
    post_bytes = Path(journal_path).read_bytes()
    post_sha256 = hashlib.sha256(post_bytes).hexdigest()
    assert src_sha256 == post_sha256, "CRITICAL ERROR: Source file was mutated during dry run!"
    print("\n[*] Cryptographic Provenance: Source journal verified 100% UNTOUCHED (SHA-256 match).")

    if dry_run:
        print("\n[!] REPORT-ONLY MODE COMPLETE. ZERO BYTES WRITTEN TO DISK.")
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            for r in output_lines:
                fout.write(json.dumps(r) + "\n")
        print(f"\n[OK] SUCCESS: Clean audited journal written to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply and write output file")
    args = parser.parse_args()
    run_attribution_pipeline(dry_run=not args.apply)
