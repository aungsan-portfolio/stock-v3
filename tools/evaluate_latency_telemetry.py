#!/usr/bin/env python3
"""
tools/evaluate_latency_telemetry.py - Gate 1 Latency Decomposition & Percentile Reporter.

Parses Schema 2.0 ORDER_LATENCY_SUMMARY records from logs/order_audit.jsonl.
Computes:
  - Sample size n
  - p50, p95, p99, max percentiles
  - Disaggregation by marketability (MARKETABLE vs PASSIVE vs UNKNOWN)
  - Disaggregation by feed (IEX vs SIP)
  - Forensic flag validation (causal_claim_permitted == False)
"""
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Gate 1 Latency Telemetry.")
    parser.add_argument(
        "--audit-file",
        type=str,
        default="logs/order_audit.jsonl",
        help="Path to order_audit.jsonl file."
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on mock sample data for verification."
    )
    return parser.parse_args()


def load_summaries(file_path: Path) -> List[Dict[str, Any]]:
    if not file_path.exists():
        return []
    summaries = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("record_type") == "ORDER_LATENCY_SUMMARY":
                    summaries.append(rec)
            except Exception:
                continue
    return summaries


def compute_percentiles(values: List[float | None]) -> Dict[str, Any]:
    valid = [v for v in values if v is not None and not np.isnan(v)]
    n = len(valid)
    if n == 0:
        return {
            "n": 0,
            "status": "NO_DATA",
            "p50": None,
            "p95": None,
            "p99": None,
            "max": None,
            "is_adequate_baseline": False,
        }

    status = "ADEQUATE_BASELINE" if n >= 100 else ("LIMITED_SAMPLE" if n >= 20 else "INSUFFICIENT_SAMPLE")
    arr = np.array(valid, dtype=float)
    return {
        "n": n,
        "status": status,
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "is_adequate_baseline": (n >= 100),
    }


def format_ms(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:.1f} ms" if val < 1000 else f"{val / 1000.0:.2f} s"


def format_stat_line(name: str, p: Dict[str, Any], target_str: str = "") -> str:
    status_label = f"[{p['status']}]"
    if p["n"] == 0:
        return f"  * {name} {target_str}:\n      n=0 {status_label} | No valid measurements"
    p99_str = format_ms(p["p99"])
    if p["n"] < 100 and p["p99"] is not None:
        p99_str += " (descriptive)"
    return (
        f"  * {name} {target_str}:\n"
        f"      n={p['n']} {status_label} | p50={format_ms(p['p50'])} | p95={format_ms(p['p95'])} | "
        f"p99={p99_str} | max={format_ms(p['max'])}"
    )


def generate_report(summaries: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("=" * 74)
    lines.append("           GATE 1: DETERMINISTIC INFRASTRUCTURE REPORT")
    lines.append("=" * 74)
    total_orders = len(summaries)
    lines.append(f"Total Order Summaries Evaluated: {total_orders}")
    lines.append("-" * 74)

    if not summaries:
        lines.append("[NOTICE] No Schema 2.0 ORDER_LATENCY_SUMMARY records found.")
        lines.append("=" * 74)
        return "\n".join(lines)

    rest_ack_latencies = [s.get("latency_t4a_t3_ms") for s in summaries]
    stream_ack_latencies = [s.get("latency_t4b_t3_ms") for s in summaries]
    ack_latencies = [s.get("latency_t4_t3_ms") for s in summaries]
    compute_latencies = [s.get("latency_t3_t2_ms") for s in summaries]
    fill_durations = [s.get("latency_t5_t4_ms") for s in summaries]
    completion_durations = [s.get("latency_t6_t5_ms") for s in summaries]
    pre_touch_durations = [s.get("pre_touch_observed_wait_ms") for s in summaries]
    post_touch_durations = [s.get("post_touch_observed_wait_ms") for s in summaries]

    rest_p = compute_percentiles(rest_ack_latencies)
    stream_p = compute_percentiles(stream_ack_latencies)
    ack_p = compute_percentiles(ack_latencies)
    comp_p = compute_percentiles(compute_latencies)
    fill_p = compute_percentiles(fill_durations)
    compl_p = compute_percentiles(completion_durations)
    pre_touch_p = compute_percentiles(pre_touch_durations)
    post_touch_p = compute_percentiles(post_touch_durations)

    lines.append("GLOBAL LATENCY DECOMPOSITION (All Orders):")
    lines.append(format_stat_line("REST API ACK Latency (T4a - T3)", rest_p, "[Target p95 < 500ms]"))
    lines.append(format_stat_line("Trade Stream ACK Latency (T4b - T3)", stream_p))
    lines.append(format_stat_line("Canonical ACK Latency (T4 - T3)", ack_p, "[Target p95 < 500ms]"))
    lines.append(format_stat_line("Strategy Compute Latency (T3 - T2)", comp_p, "[Target < 100ms]"))
    lines.append(format_stat_line("Accepted-to-First-Fill Duration (T5 - T4)", fill_p))
    lines.append(format_stat_line("  - Pre-Touch Observed Wait (T_touch - T4)", pre_touch_p, "[Market Away]"))
    lines.append(format_stat_line("  - Post-Touch Observed Wait (T5 - T_touch)", post_touch_p, "[Market At Limit]"))
    lines.append(format_stat_line("Partial-to-Complete Duration (T6 - T5)", compl_p))
    lines.append("-" * 74)

    # Missingness Audit
    missing_ack = sum(1 for s in summaries if s.get("latency_t4_t3_ms") is None)
    missing_first_fill = sum(1 for s in summaries if s.get("latency_t5_t4_ms") is None)
    missing_complete_fill = sum(1 for s in summaries if s.get("latency_t6_t5_ms") is None)
    missing_touch = sum(1 for s in summaries if s.get("pre_touch_observed_wait_ms") is None)

    lines.append("TELEMETRY COMPLETION & MISSINGNESS AUDIT:")
    lines.append(f"  * Orders Evaluated: {total_orders}")
    lines.append(f"  * Missing Canonical ACK (T4): {missing_ack} ({missing_ack / total_orders * 100.0:.1f}%)")
    lines.append(f"  * Missing First Fill (T5): {missing_first_fill} ({missing_first_fill / total_orders * 100.0:.1f}%)")
    lines.append(f"  * Missing Complete Fill (T6): {missing_complete_fill} ({missing_complete_fill / total_orders * 100.0:.1f}%)")
    lines.append(f"  * Unobserved / Missing Touch Timeline: {missing_touch} ({missing_touch / total_orders * 100.0:.1f}%)")
    lines.append("-" * 74)

    # Disaggregation by Touch Evidence State
    by_evidence = {}
    for s in summaries:
        ev_state = s.get("touch_evidence_state", "TOUCH_TIMELINE_UNAVAILABLE")
        by_evidence.setdefault(ev_state, []).append(s)

    lines.append("DISAGGREGATION BY TOUCH EVIDENCE STATE:")
    for ev_st, group in sorted(by_evidence.items()):
        grp_post = [x.get("post_touch_observed_wait_ms") for x in group]
        p = compute_percentiles(grp_post)
        p99_str = format_ms(p["p99"])
        if p["n"] < 100 and p["p99"] is not None:
            p99_str += " (descriptive)"
        lines.append(f"  [{ev_st}] (n={len(group)} orders, measured_post_touch_n={p['n']}):")
        lines.append(f"      post_touch p50={format_ms(p['p50'])} | p95={format_ms(p['p95'])} | p99={p99_str} | max={format_ms(p['max'])}")
    lines.append("-" * 74)

    # Disaggregation by Marketability
    by_marketability = {}
    for s in summaries:
        m = s.get("marketability_at_submit", "UNKNOWN")
        by_marketability.setdefault(m, []).append(s)

    lines.append("DISAGGREGATION BY SUBMIT MARKETABILITY:")
    for m, group in sorted(by_marketability.items()):
        grp_fill = [x.get("latency_t5_t4_ms") for x in group]
        p = compute_percentiles(grp_fill)
        lines.append(f"  [{m}] (n={p['n']}, status={p['status']}):")
        p99_str = format_ms(p["p99"])
        if p["n"] < 100 and p["p99"] is not None:
            p99_str += " (descriptive)"
        lines.append(f"      p50={format_ms(p['p50'])} | p95={format_ms(p['p95'])} | p99={p99_str} | max={format_ms(p['max'])}")
    lines.append("-" * 74)

    # Forensic Safety Verification
    invalid_causal = [s for s in summaries if s.get("causal_claim_permitted", False)]
    prohibited_keys = ["queue_wait_ms", "venue_queue_delay_ms", "liquidity_delay_ms"]
    found_prohibited = [k for k in prohibited_keys if any(k in s for s in summaries)]

    lines.append("FORENSIC ATTRIBUTION INTEGRITY:")
    if not invalid_causal and not found_prohibited:
        lines.append("  [PASS] All orders respect evidence boundaries (causal_claim_permitted == False).")
        lines.append("  [PASS] Zero speculative causal keys detected in latency telemetry.")
    else:
        if invalid_causal:
            lines.append(f"  [FAIL] {len(invalid_causal)} orders violated causal claim invariant!")
        if found_prohibited:
            lines.append(f"  [FAIL] Prohibited causal keys detected: {found_prohibited}!")
    lines.append("=" * 74)
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.sample:
        mock_summaries = [
            {
                "schema_version": "2.1",
                "record_type": "ORDER_LATENCY_SUMMARY",
                "symbol": "SPY",
                "marketability_at_submit": "PASSIVE",
                "latency_t3_t2_ms": 25,
                "latency_t4a_t3_ms": 5,
                "latency_t4b_t3_ms": 12,
                "latency_t4_t3_ms": 5,
                "latency_t5_t4_ms": 284705,
                "latency_t6_t5_ms": 1947,
                "pre_touch_observed_wait_ms": 280000,
                "post_touch_observed_wait_ms": 4705,
                "touch_evidence_state": "TOUCH_OBSERVED",
                "touch_source_feed": "IEX",
                "touch_timeline_resolution": "TICK_LEVEL",
                "causal_claim_permitted": False,
            },
            {
                "schema_version": "2.1",
                "record_type": "ORDER_LATENCY_SUMMARY",
                "symbol": "BAC",
                "marketability_at_submit": "MARKETABLE",
                "latency_t3_t2_ms": 18,
                "latency_t4a_t3_ms": 8,
                "latency_t4b_t3_ms": 15,
                "latency_t4_t3_ms": 8,
                "latency_t5_t4_ms": 1200,
                "latency_t6_t5_ms": 300,
                "pre_touch_observed_wait_ms": 0,
                "post_touch_observed_wait_ms": 1208,
                "touch_evidence_state": "IMMEDIATE_AT_SUBMIT",
                "touch_source_feed": "IEX",
                "touch_timeline_resolution": "TICK_LEVEL",
                "causal_claim_permitted": False,
            },
            {
                "schema_version": "2.1",
                "record_type": "ORDER_LATENCY_SUMMARY",
                "symbol": "NFLX",
                "marketability_at_submit": "PASSIVE",
                "latency_t3_t2_ms": 30,
                "latency_t4a_t3_ms": 6,
                "latency_t4b_t3_ms": 10,
                "latency_t4_t3_ms": 6,
                "latency_t5_t4_ms": 950,
                "latency_t6_t5_ms": 420,
                "pre_touch_observed_wait_ms": None,
                "post_touch_observed_wait_ms": None,
                "touch_evidence_state": "UNOBSERVED_FEED_GAP",
                "touch_source_feed": "IEX",
                "touch_timeline_resolution": "TICK_LEVEL",
                "causal_claim_permitted": False,
            }
        ]
        print(generate_report(mock_summaries))
        return

    summaries = load_summaries(Path(args.audit_file))
    print(generate_report(summaries))


if __name__ == "__main__":
    main()
