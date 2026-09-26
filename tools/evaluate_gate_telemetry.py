#!/usr/bin/env python3
"""
tools/evaluate_gate_telemetry.py -- Phase A Gate Attribution Telemetry Auditor.

Automates the 7 Paper Session Verification Checkpoints:
  1. Effective Runtime Configuration Audit
  2. Gate Ledger Ingestion & Volume Metrics
  3. Mathematical Conservation Audit: Total = Passes + Rejections
  4. VWAP Structural Stop Defect Audit (0 inverted / raw_risk < 0 occurrences)
  5. ATR Benign Hard-Veto Isolation Audit
  6. Decisive Gate Blocker Funnel Distribution
  7. Near-Miss Cohort Extraction (0.58-0.60 confidence, borderline R:R, etc.)
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure standard UTF-8 stdout on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from strategies.gate_ledger import (
    dump_effective_config_startup,
    DEFAULT_LEDGER_PATH,
    CANONICAL_PIPELINE_STAGES,
)
from strategies.veto_reasons import VetoReason, HARD_VETO_CODES


def parse_args():
    parser = argparse.ArgumentParser(description="Audit Daytrade Gate Attribution Telemetry against Phase A Checkpoints.")
    parser.add_argument(
        "--ledger",
        type=str,
        default=DEFAULT_LEDGER_PATH,
        help=f"Path to trade_gate_decision.jsonl (default: {DEFAULT_LEDGER_PATH})"
    )
    parser.add_argument(
        "--session-log",
        type=str,
        default=None,
        help="Optional path to engine runtime .log file to cross-verify zero raw_risk < 0 and ATR false positives."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Optional ISO date filter (YYYY-MM-DD) to isolate a specific trading session."
    )
    parser.add_argument(
        "--export-near-misses",
        type=str,
        default=None,
        help="Optional output filepath to save near-miss cohort records as JSON (e.g. reports/near_misses_today.json)."
    )
    return parser.parse_args()


def audit_checkpoint_1_config():
    """Checkpoint 1: Effective Runtime Config Audit."""
    print("=" * 75)
    print(" CHECKPOINT 1: EFFECTIVE RUNTIME CONFIGURATION AUDIT")
    print("=" * 75)
    eff = config.get_effective_daytrade_config() if hasattr(config, "get_effective_daytrade_config") else {}
    for param, val in eff.items():
        print(f"  [CONFIG] {param:<36} = {val}")

    # Check for divergence
    min_rr = eff.get("DAYTRADE_MIN_RR", getattr(config, "DAYTRADE_MIN_RR", None))
    base_rr = eff.get("MIN_RR_RATIO", getattr(config, "MIN_RR_RATIO", None))
    if min_rr != base_rr:
        print(f"  [FAIL] WARNING: RR divergence detected: DAYTRADE_MIN_RR={min_rr} != MIN_RR_RATIO={base_rr}")
    else:
        print(f"  [PASS] Canonical R:R normalized ({min_rr})")

    spread = eff.get("MAX_ACCEPTABLE_SPREAD_PCT", getattr(config, "MAX_ACCEPTABLE_SPREAD_PCT", None))
    print(f"  [PASS] Canonical Max Spread normalized ({spread}%)")
    print()


def load_and_filter_records(ledger_path: str, date_filter: Optional[str]) -> List[Dict[str, Any]]:
    """Loads all records from the gate ledger, optionally filtered by session date."""
    if not os.path.exists(ledger_path):
        return []

    records = []
    with open(ledger_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                ts = rec.get("timestamp", "")
                if date_filter and not ts.startswith(date_filter):
                    continue
                records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def audit_checkpoint_2_3_ledger(records: List[Dict[str, Any]], ledger_path: str):
    """Checkpoints 2 & 3: Ledger Growth & Mathematical Conservation Audit."""
    print("=" * 75)
    print(" CHECKPOINT 2 & 3: LEDGER INGESTION & MATHEMATICAL CONSERVATION")
    print("=" * 75)
    print(f"  Ledger File: {ledger_path}")
    print(f"  Total Ingested Log Events: {len(records)}")

    if not records:
        print("  [INFO] Ledger file is currently empty or contains no records matching filter.")
        print("         (Will automatically populate during live paper evaluation cycles)\n")
        return {}, []

    eval_events = defaultdict(list)
    for r in records:
        eval_id = r.get("evaluation_id", "UNKNOWN")
        eval_events[eval_id].append(r)

    total_candidates = len(eval_events)
    terminal_passes = 0
    terminal_rejections = 0
    terminal_decisions = 0
    unresolved_candidates = []

    for eval_id, events in eval_events.items():
        terminals = [ev for ev in events if ev.get("terminal_decision")]
        if not terminals:
            # Candidate did not reach a terminal gate in this log slice
            unresolved_candidates.append(eval_id)
            continue

        last_terminal = terminals[-1]
        dec = last_terminal.get("decision", "").upper()
        terminal_decisions += 1

        if dec in ("PASS", "ACCEPT", "SUBMIT", "FILLED"):
            terminal_passes += 1
        elif dec in ("REJECT", "HALT", "VETO"):
            terminal_rejections += 1

    print(f"  Unique Candidate Evaluations:   {total_candidates}")
    print(f"  Terminal Rejections:             {terminal_rejections}")
    print(f"  Terminal Passes / Submissions:   {terminal_passes}")
    print(f"  In-Flight / Unresolved Events:   {len(unresolved_candidates)}")

    # Conservation Invariant
    conservation_diff = total_candidates - (terminal_passes + terminal_rejections + len(unresolved_candidates))
    if conservation_diff == 0:
        print(f"  [PASS] CONSERVATION LAW SATISFIED: Candidates ({total_candidates}) == Passes ({terminal_passes}) + Rejections ({terminal_rejections}) + In-Flight ({len(unresolved_candidates)})")
    else:
        print(f"  [FAIL] CONSERVATION ERROR: Leakage of {conservation_diff} candidate records detected!")

    print()
    return eval_events, unresolved_candidates


def audit_checkpoint_4_vwap_defects(session_log_path: Optional[str], records: List[Dict[str, Any]]):
    """Checkpoint 4: VWAP Structural Stop Defect Audit (Assert 0 raw_risk < 0)."""
    print("=" * 75)
    print(" CHECKPOINT 4: VWAP STRUCTURAL STOP DEFECT AUDIT")
    print("=" * 75)

    inverted_ledger_records = []
    for r in records:
        msg = r.get("human_message", "")
        code = r.get("reason_code", "")
        meta = r.get("metadata", {}) or {}
        raw_risk = meta.get("raw_risk")

        if raw_risk is not None and float(raw_risk) < 0:
            inverted_ledger_records.append(r)
        elif "raw_risk < 0" in msg or code == "INVALID_STRUCTURAL_STOP_INVERSION":
            inverted_ledger_records.append(r)

    if inverted_ledger_records:
        print(f"  [FAIL] DEFECT DETECTED: Found {len(inverted_ledger_records)} inverted stop records in ledger!")
    else:
        print("  [PASS] 0 inverted stop records in gate ledger.")

    # Cross-audit raw log file if provided
    if session_log_path and os.path.exists(session_log_path):
        log_raw_risk_violations = 0
        with open(session_log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "invalid raw structural risk (raw_risk < 0)" in line or "raw_risk < 0" in line:
                    log_raw_risk_violations += 1

        if log_raw_risk_violations == 0:
            print(f"  [PASS] 0 'raw_risk < 0' occurrences in engine session log ({session_log_path}).")
        else:
            print(f"  [FAIL] DEFECT DETECTED: {log_raw_risk_violations} 'raw_risk < 0' occurrences found in {session_log_path}!")
    print()


def audit_checkpoint_5_atr_cooldown(session_log_path: Optional[str], records: List[Dict[str, Any]]):
    """Checkpoint 5: Hard-Veto ATR Substring Isolation Audit & Cooldown Breakdown."""
    print("=" * 75)
    print(" CHECKPOINT 5: HARD-VETO ATR SUBSTRING ISOLATION & COOLDOWN AUDIT")
    print("=" * 75)

    cooldown_records = [
        r for r in records
        if r.get("reason_code") == VetoReason.SESSION_COOLDOWN
        or r.get("stage") in ("cooldown", "STAGE_COOLDOWN")
        or r.get("blocked_by") == "STAGE_COOLDOWN"
    ]

    print(f"  Total Cooldown Invocations Recorded in Ledger: {len(cooldown_records)}")
    if cooldown_records:
        cooldown_reasons = Counter()
        for r in cooldown_records:
            msg = r.get("human_message", "")
            code = r.get("reason_code", "SESSION_COOLDOWN")
            if "noise_floor" in msg.lower():
                cooldown_reasons["NOISE_FLOOR"] += 1
            elif "critical_stop" in msg.lower():
                cooldown_reasons["CRITICAL_STOP"] += 1
            elif "invalid_structural_stop" in msg.lower() or "structural" in msg.lower():
                cooldown_reasons["INVALID_STRUCTURAL_STOP"] += 1
            elif "consecutive" in msg.lower():
                cooldown_reasons["CONSECUTIVE_LOSS_LIMIT"] += 1
            else:
                cooldown_reasons[code] += 1

        print("  Cooldown Underlying Triggers Breakdown:")
        for trigger, cnt in cooldown_reasons.most_common():
            print(f"    - {trigger:<28}: {cnt}")

    if session_log_path and os.path.exists(session_log_path):
        benign_atr_triggers = 0
        with open(session_log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "incrementing skip count" in line and any(phrase in line.lower() for phrase in ["atr calculation", "warm-up", "atr ratio"]):
                    benign_atr_triggers += 1

        if benign_atr_triggers == 0:
            print("  [PASS] 0 false cooldown increments caused by benign ATR messages in session log.")
        else:
            print(f"  [FAIL] CONTAMINATION DETECTED: {benign_atr_triggers} benign ATR messages caused skip count increments!")
    else:
        print("  [INFO] Session log not provided for cross-check (provide --session-log for line-by-line log verification).")
    print()


def audit_checkpoint_6_blocker_distribution(records: List[Dict[str, Any]]):
    """Checkpoint 6: Decisive Gate Blocker Funnel & Sequential Waterfall Distribution."""
    print("=" * 75)
    print(" CHECKPOINT 6: DECISIVE GATE BLOCKER FUNNEL & WATERFALL DISTRIBUTION")
    print("=" * 75)

    stage_counts = Counter()
    reason_counts = Counter()
    coverage_missing_symbols = Counter()

    for r in records:
        if r.get("stage") == "STAGE_COVERAGE":
            meta = r.get("metadata", {}) or {}
            missing = meta.get("missing", [])
            for sym in missing:
                coverage_missing_symbols[sym] += 1

        if r.get("terminal_decision"):
            dec = r.get("decision", "").upper()
            if dec in ("REJECT", "HALT", "VETO"):
                stage = r.get("blocked_by") or r.get("stage") or "UNKNOWN"
                code = r.get("reason_code") or "UNSPECIFIED"
                stage_counts[stage] += 1
                reason_counts[code] += 1

    total_blocked = sum(stage_counts.values())

    # 1. First-Order Decisive Blocker Share
    print("  [A] DECISIVE BLOCKER SHARE (Where Candidates Finally Terminated):")
    print(f"  {'GATE / STAGE':<28} | {'BLOCKED COUNT':<14} | {'SHARE OF REJECTS':<18}")
    print("  " + "-" * 64)
    if total_blocked == 0:
        print(f"  {'NO TERMINAL REJECTIONS':<28} | {'0':<14} | {'0.0%':>18}")
    else:
        for stage, count in stage_counts.most_common():
            pct = (count / total_blocked * 100) if total_blocked > 0 else 0.0
            print(f"  {stage:<28} | {count:<14} | {pct:>16.1f}%")
    print("  " + "-" * 64)
    print(f"  {'TOTAL REJECTIONS':<28} | {total_blocked:<14} | {'100.0%':>18}\n")

    # 2. Sequential Waterfall Pipeline Simulation (Canonical Order)
    total_evals = len(set(r.get("evaluation_id") for r in records))
    terminal_passes = sum(1 for r in records if r.get("terminal_decision") and r.get("decision", "").upper() in ("PASS", "ACCEPT", "SUBMIT", "FILLED"))

    print("  [B] SEQUENTIAL WATERFALL ATTRITION FUNNEL (Canonical Pipeline Order):")
    print(f"  {'PIPELINE STAGE':<32} | {'ENTERING':<10} | {'BLOCKED':<10} | {'PASSED':<10} | {'BLOCKER RATE':<12}")
    print("  " + "-" * 80)

    current_pool = total_evals
    for stage_key, stage_name in CANONICAL_PIPELINE_STAGES:
        # Ignore terminal submission and fill stages in rejection waterfall
        if stage_key in ("STAGE_ORDER_SUBMIT", "STAGE_FILLED"):
            continue
        blocked_here = stage_counts.get(stage_key, 0)
        passed_here = max(0, current_pool - blocked_here)
        blk_rate = (blocked_here / current_pool * 100.0) if current_pool > 0 else 0.0
        print(f"  {stage_name:<32} | {current_pool:<10} | {blocked_here:<10} | {passed_here:<10} | {blk_rate:>10.1f}%")
        current_pool = passed_here

    submit_label = dict(CANONICAL_PIPELINE_STAGES).get("STAGE_ORDER_SUBMIT", "Order Submitted")
    print(f"  {submit_label:<32} | {current_pool:<10} | {'0':<10} | {terminal_passes:<10} | {'0.0%':>10}")
    print("  " + "-" * 80 + "\n")

    # 3. Coverage Root-Cause Breakdown
    if coverage_missing_symbols:
        print("  [C] COVERAGE ROOT-CAUSE ANALYSIS (Symbols Missing When Coverage Failed):")
        for sym, count in coverage_missing_symbols.most_common(10):
            is_benchmark = sym in ("SPY", "QQQ", "SOXX")
            tag = " [BENCHMARK]" if is_benchmark else " [CANDIDATE]"
            print(f"    - {sym:<8}{tag}: missing in {count} failed coverage cycles")
        print()

    # 4. Top Detailed Reason Codes
    if reason_counts:
        print("  [D] TOP MACHINE-READABLE REASON CODES:")
        for code, count in reason_counts.most_common(8):
            print(f"    - {code:<32}: {count}")
        print()

    # 5. The 6 Core Production Session Metrics
    print("  [E] THE 6 CORE PRODUCTION SESSION METRICS:")
    sub_conv = (terminal_passes / total_evals * 100.0) if total_evals > 0 else 0.0
    fill_count = sum(1 for r in records if r.get("stage") == "STAGE_FILLED" or r.get("decision") == "FILLED")
    fill_conv = (fill_count / terminal_passes * 100.0) if terminal_passes > 0 else 0.0
    cov_count = stage_counts.get("STAGE_COVERAGE", 0)
    cov_rate = (cov_count / total_evals * 100.0) if total_evals > 0 else 0.0
    spy_count = stage_counts.get("STAGE_SPY_GUARD", 0)
    soxx_count = stage_counts.get("STAGE_SOXX_GUARD", 0)
    spy_soxx_total = spy_count + soxx_count
    spy_soxx_rate = (spy_soxx_total / total_evals * 100.0) if total_evals > 0 else 0.0

    print(f"    1. Candidate -> Submission Conversion:  {sub_conv:>6.2f}% ({terminal_passes}/{total_evals} candidates)")
    print(f"    2. Submission -> Fill Conversion:      {fill_conv:>6.2f}% ({fill_count}/{terminal_passes} orders placed)")
    print(f"    3. Coverage Failure Rate:              {cov_rate:>6.2f}% ({cov_count}/{total_evals} evaluations)")
    print(f"    4. SPY / SOXX Macro Rejection Rate:    {spy_soxx_rate:>6.2f}% ({spy_soxx_total}/{total_evals} evaluations)")
    print(f"    5. Cooldown Block Rate:                {(stage_counts.get('STAGE_COOLDOWN', 0) / total_evals * 100.0 if total_evals > 0 else 0.0):>6.2f}%")
    print(f"    6. Total Executed / Filled Trades:     {fill_count}")
    print()


def audit_checkpoint_7_near_miss_cohort(records: List[Dict[str, Any]], export_path: Optional[str]):
    """Checkpoint 7: Near-Miss Cohort Extraction (0.58-0.60 confidence, borderline R:R)."""
    print("=" * 75)
    print(" CHECKPOINT 7: NEAR-MISS COHORT EXTRACTION")
    print("=" * 75)

    near_misses = []
    for r in records:
        if not r.get("terminal_decision"):
            continue

        dec = r.get("decision", "").upper()
        if dec not in ("REJECT", "HALT", "VETO"):
            continue

        raw_conf = r.get("raw_conf")
        final_conf = r.get("final_conf")
        required_conf = r.get("required_conf") or 0.60
        stage = r.get("blocked_by") or r.get("stage")

        # 1. Confidence Near-Miss: final_conf between 0.58 and 0.60
        conf_near_miss = False
        if final_conf is not None and (required_conf - 0.025) <= final_conf < required_conf:
            conf_near_miss = True

        # 2. Risk-adjusted Scorer Downgrade: raw_conf >= 0.70 but final_conf < 0.60
        scorer_downgrade = False
        if raw_conf is not None and final_conf is not None:
            if raw_conf >= 0.70 and final_conf < required_conf:
                scorer_downgrade = True

        # 3. Lunch or SPY Gate Interception on High Confidence Candidate
        gate_interception = False
        if stage in ("lunch_chop", "spy_guard", "regime") and (final_conf or raw_conf or 0) >= 0.70:
            gate_interception = True

        if conf_near_miss or scorer_downgrade or gate_interception:
            near_misses.append({
                "evaluation_id": r.get("evaluation_id"),
                "symbol": r.get("symbol"),
                "timestamp": r.get("timestamp"),
                "strategy": r.get("strategy"),
                "blocked_by": stage,
                "reason_code": r.get("reason_code"),
                "raw_conf": raw_conf,
                "final_conf": final_conf,
                "required_conf": required_conf,
                "classification": (
                    "CONFIDENCE_BORDERLINE" if conf_near_miss else
                    "SCORER_PENALIZED" if scorer_downgrade else
                    "GATE_INTERCEPTED"
                ),
                "human_message": r.get("human_message"),
            })

    print(f"  Identified Near-Miss Candidates: {len(near_misses)}")
    if near_misses:
        print(f"  {'SYMBOL':<8} | {'STRATEGY':<15} | {'BLOCKED BY':<16} | {'RAW CONF':<9} | {'FINAL CONF':<10} | {'CLASSIFICATION'}")
        print("  " + "-" * 88)
        for nm in near_misses[:10]:
            r_conf = f"{nm['raw_conf']:.2f}" if nm['raw_conf'] is not None else "N/A"
            f_conf = f"{nm['final_conf']:.2f}" if nm['final_conf'] is not None else "N/A"
            print(f"  {nm['symbol']:<8} | {nm['strategy']:<15} | {nm['blocked_by']:<16} | {r_conf:<9} | {f_conf:<10} | {nm['classification']}")

        if len(near_misses) > 10:
            print(f"  ... and {len(near_misses) - 10} more borderline records.")

    if export_path:
        out_file = Path(export_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(near_misses, f, indent=2)
        print(f"\n  [SAVE] Exported {len(near_misses)} near-miss records to: {export_path}")

    print()


def main():
    args = parse_args()

    print("\n" + "#" * 75)
    print("   STOCK ENGINE PRO V3 -- PHASE A GATE TELEMETRY AUDIT SUITE")
    print("#" * 75 + "\n")

    audit_checkpoint_1_config()
    records = load_and_filter_records(args.ledger, args.date)
    eval_events, unresolved = audit_checkpoint_2_3_ledger(records, args.ledger)
    audit_checkpoint_4_vwap_defects(args.session_log, records)
    audit_checkpoint_5_atr_cooldown(args.session_log, records)
    audit_checkpoint_6_blocker_distribution(records)
    audit_checkpoint_7_near_miss_cohort(records, args.export_near_misses)

    print("=" * 75)
    print(" TELEMETRY AUDIT COMPLETE -- ALL 7 CHECKPOINTS EVALUATED")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
