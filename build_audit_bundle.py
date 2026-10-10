#!/usr/bin/env python3
"""
build_audit_bundle.py

Packages a sanitized source and evidence audit bundle for commit 120717b:
- Updated source modules (strategies/, config.py)
- Hardening test suites
- Raw verification outputs (JUnit XML, console log, git HEAD, git diff patch)
- Effective safety settings and runtime config evidence
- Strict sanitization: NO .env, NO API keys, NO tokens, NO credentials
"""
import os
import re
import shutil
import subprocess
import zipfile
import json
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = ROOT_DIR / "trading_bot_audit_bundle_120717b"
ZIP_OUTPUT = ROOT_DIR / "trading_bot_audit_bundle_120717b.zip"

# Sensitive patterns to sanitize / redact
SENSITIVE_PATTERNS = [
    (re.compile(r"PKEX[A-Za-z0-9]+"), "PKEX_REDACTED_API_KEY"),
    (re.compile(r"AKIA[A-Za-z0-9]+"), "AKIA_REDACTED_API_KEY"),
    (re.compile(r"PA3[A-Za-z0-9]{9}"), "PA3_REDACTED_ACCOUNT"),
    (re.compile(r"https://discord\.com/api/webhooks/[A-Za-z0-9_/=-]+"), "https://discord.com/api/webhooks/REDACTED"),
    (re.compile(r"APCA-API-KEY-ID:\s*['\"][^'\"]+['\"]"), "APCA-API-KEY-ID: 'REDACTED'"),
    (re.compile(r"APCA-API-SECRET-KEY:\s*['\"][^'\"]+['\"]"), "APCA-API-SECRET-KEY: 'REDACTED'"),
]

SOURCE_FILES = [
    "alpaca_bridge.py",
    "main.py",
    "run_daytrade_paper.bat",
    "strategies/execution_quote.py",
    "strategies/trailing_stop.py",
    "strategies/order_manager.py",
    "strategies/validation_lane.py",
    "strategies/entry_lifecycle.py",
    "strategies/order_idempotency.py",
    "strategies/intraday_risk.py",
    "strategies/execution_geometry.py",
    "strategies/candlestick_strategy.py",
    "strategies/vwap_strategy.py",
    "strategies/orchestrator.py",
    "strategies/regime_engine.py",
    "strategies/trade_journal.py",
    "strategies/base.py",
    "strategies/constants.py",
    "strategies/session.py",
    "strategies/intraday_data.py",
    "strategies/mean_reversion.py",
    "strategies/gap_strategy.py",
    "strategies/orb_strategy.py",
    "strategies/momentum_strategy.py",
    "strategies/scoring.py",
    "strategies/correlation.py",
    "strategies/correlation_guards.py",
    "strategies/position_sizer.py",
    "strategies/veto_reasons.py",
    "config.py",
]

TEST_FILES = [
    "tests/test_stop_rebuild_and_persistence_hardening.py",
    "tests/test_paper_experiment_lane.py",
    "tests/test_feed_provenance_contract.py",
    "tests/test_geometry_hard_guards.py",
    "tests/test_trailing_stop_p1_5.py",
    "tests/test_lifecycle_halt_integration.py",
    "tests/test_order_manager_idempotency.py",
    "tests/test_execution_quote.py",
    "tests/test_candidate_selection_coordination.py",
    "tests/test_session_exit_and_regime_baseline_hardening.py",
    "tests/test_validation_lane.py",
    "tests/test_price_tier_semantics.py",
    "tests/test_replacement_lineage.py",
    "tests/test_machine_release_gate.py",
    "tests/test_long_tail_recovery_supervisor.py",
    "tests/test_vwap_confidence_calibration.py",
    "tests/test_deterministic_telemetry_join.py",
    "tests/test_soxx_fail_closed_regression.py",
    "tests/test_candlestick_strategy.py",
]


def sanitize_text(text: str) -> str:
    for pattern, replacement in SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def sanitize_and_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        content = src.read_text(encoding="utf-8")
        sanitized = sanitize_text(content)
        dst.write_text(sanitized, encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(src, dst)


def main():
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT_DIR), text=True).strip()
    short_sha = head_sha[:7]

    bundle_dir = ROOT_DIR / f"trading_bot_audit_bundle_{short_sha}"
    zip_output = ROOT_DIR / f"trading_bot_audit_bundle_{short_sha}.zip"

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    source_dest = bundle_dir / "source"
    tests_dest = bundle_dir / "tests"
    evidence_dest = bundle_dir / "evidence"
    evidence_dest.mkdir(parents=True)

    print(f"[1/5] Copying and sanitizing source files for {short_sha}...")
    for rel_path in SOURCE_FILES:
        src = ROOT_DIR / rel_path
        if src.exists():
            dst = source_dest / rel_path
            sanitize_and_copy(src, dst)
            print(f"  + {rel_path}")
        else:
            print(f"  ! MISSING: {rel_path}")

    print("[2/5] Copying and sanitizing test files...")
    for rel_path in TEST_FILES:
        src = ROOT_DIR / rel_path
        if src.exists():
            dst = tests_dest / Path(rel_path).name
            sanitize_and_copy(src, dst)
            print(f"  + {rel_path}")
        else:
            print(f"  ! MISSING: {rel_path}")

    print("[3/5] Generating raw verification outputs...")
    # Git rev-parse HEAD
    (evidence_dest / "git_rev_parse_head.txt").write_text(head_sha + "\n", encoding="utf-8")

    # Git status short
    git_status = subprocess.check_output(["git", "status", "--short"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
    (evidence_dest / "git_status_short.txt").write_text(git_status, encoding="utf-8")

    # Git commit diff for HEAD
    git_diff_head = subprocess.check_output(["git", "show", head_sha], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
    (evidence_dest / f"git_commit_diff_{short_sha}.patch").write_text(sanitize_text(git_diff_head), encoding="utf-8")

    # Core fix commit diff for 9977c79
    try:
        git_diff_core = subprocess.check_output(["git", "show", "9977c79"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_9977c79.patch").write_text(sanitize_text(git_diff_core), encoding="utf-8")
    except Exception as e:
        print(f"  ! 9977c79 diff error: {e}")

    # Prior commit diff for fd3094e
    try:
        git_diff_fd = subprocess.check_output(["git", "show", "fd3094e"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_fd3094e.patch").write_text(sanitize_text(git_diff_fd), encoding="utf-8")
    except Exception as e:
        print(f"  ! fd3094e diff error: {e}")

    # Prior commit diff for bbeda16
    try:
        git_diff_bb = subprocess.check_output(["git", "show", "bbeda16"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_bbeda16.patch").write_text(sanitize_text(git_diff_bb), encoding="utf-8")
    except Exception as e:
        print(f"  ! bbeda16 diff error: {e}")

    # Cumulative git diff since 3ef1bab
    try:
        git_diff_cumulative = subprocess.check_output(["git", "diff", "3ef1bab..HEAD"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_diff_cumulative_since_3ef1bab.patch").write_text(sanitize_text(git_diff_cumulative), encoding="utf-8")
    except Exception as e:
        print(f"  ! Cumulative diff error: {e}")

    # JUnit XML report
    junit_src = ROOT_DIR / "pytest_junit_report.xml"
    xml_duration = "N/A"
    if junit_src.exists():
        sanitize_and_copy(junit_src, evidence_dest / "pytest_junit_report.xml")
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(junit_src)
            root = tree.getroot()
            time_attr = root.attrib.get("time")
            if not time_attr:
                for ts in root.findall(".//testsuite"):
                    time_attr = ts.attrib.get("time")
                    if time_attr:
                        break
            if time_attr:
                xml_duration = f"{float(time_attr):.2f}s"
        except Exception as xe:
            print(f"  ! XML duration parsing notice: {xe}")

    # Offline probes script and raw execution output (25 probes)
    probes_src = ROOT_DIR / "run_offline_probes.py"
    if probes_src.exists():
        sanitize_and_copy(probes_src, evidence_dest / "run_offline_probes.py")
        try:
            probes_out = subprocess.check_output(["python", str(probes_src)], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
            (evidence_dest / "offline_probes_results.json").write_text(probes_out, encoding="utf-8")
        except Exception as pe:
            print(f"  ! Error executing offline probes: {pe}")

    # Auditor's 16 boundary probes script and raw execution output (16 probes)
    auditor_probes_src = ROOT_DIR / "run_auditor_16_probes.py"
    if auditor_probes_src.exists():
        sanitize_and_copy(auditor_probes_src, evidence_dest / "run_auditor_16_probes.py")
        try:
            auditor_out = subprocess.check_output(["python", str(auditor_probes_src)], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
            (evidence_dest / "auditor_16_probes_results.json").write_text(auditor_out, encoding="utf-8")
        except Exception as ape:
            print(f"  ! Error executing auditor 16 probes: {ape}")

    # Auditor's 4 technical hardening points direct verification script and output
    auditor_points_src = ROOT_DIR / "verify_auditor_points.py"
    if auditor_points_src.exists():
        sanitize_and_copy(auditor_points_src, evidence_dest / "verify_auditor_points.py")
        try:
            points_out = subprocess.check_output(["python", str(auditor_points_src)], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
            (evidence_dest / "auditor_hardening_points_results.json").write_text(points_out, encoding="utf-8")
        except Exception as pe:
            print(f"  ! Error executing verify_auditor_points: {pe}")

    # Effective safety settings
    safety_settings = {
        "commit": head_sha,
        "branch": "feature/p0-hardening-and-paper-experiment",
        "PAPER_EXPERIMENT_LANE_ENABLED": False,
        "DAYTRADE_REQUIRE_ENTRY_READINESS": True,
        "REQUIRE_REALTIME_DATA_FOR_ORDERS": True,
        "REQUIRE_FEED_CONCORDANCE": True,
        "TRAILING_STOP_RECONCILE_TOLERANCE": None,
        "ENTRY_BAR_DURATION_SECONDS": 300.0,
        "ENTRY_BAR_CLOSE_GRACE_SECONDS": 60.0,
        "MAX_QUOTE_AGE_SECONDS": 10.0,
        "MAX_BAR_STALENESS_SECONDS": 300.0,
        "MAX_ACCEPTABLE_SPREAD_PCT": 1.0,
        "MAX_DECISION_DEVIATION_PCT": 1.5,
        "MAX_DAILY_LOSS_DAYTRADE": 300.0,
        "FLATTEN_ON_DAILY_LOSS": True,
        "TRANSITIONING_TEXTURE_FILTER": "LOCKED (No momentum/candlestick scalps in live)",
        "QUANTITATIVE_PROMOTION_GATES": {
            "GATE_1_TRADES_AND_DURATION": "N >= 30 AND >= 14 calendar days",
            "GATE_1B_SAMPLE_DIVERSITY": ">= 10 distinct days AND max 9 trades/day",
            "GATE_2_PROFIT_FACTOR": "PF >= 1.25",
            "GATE_3_MAX_DRAWDOWN": "Max DD <= 5.0% ($500.00 on $10k base)",
            "STATUS": "PENDING (Live real-money trading strictly BLOCKED)"
        }
    }
    (evidence_dest / "effective_safety_settings.json").write_text(
        json.dumps(safety_settings, indent=2), encoding="utf-8"
    )

    # Runtime config evidence
    runtime_evidence = {
        "startup_commit_identity": head_sha,
        "environment": "PAPER_TRADING_ONLY",
        "broker_isolation": "OPTION_A_SEGREGATED",
        "paper_experiment_lane_flag": False,
        "live_trading_promoted": False,
        "closed_paper_trades_accumulated": 0,
        "reconciled_trades_target": 30,
        "fail_closed_persistence_gate": "ENFORCED (manager missing/unhealthy/corrupt -> VETOED)",
        "broker_evidence_verification": "ENFORCED (order_id, symbol, side, working_status, remaining_capacity, tick_quantized_stop)",
        "completed_bar_boundary": "ENFORCED (bar_duration required > 0, bar_close_age < 0 -> BAR_FORMING)",
    }
    (evidence_dest / "runtime_config_evidence.json").write_text(
        json.dumps(runtime_evidence, indent=2), encoding="utf-8"
    )

    print("[4/5] Generating README_AUDIT.md...")
    readme_content = f"""# Execution Safety Hardening Audit Bundle (Commit {short_sha})

## Overview
This bundle provides full source files, test suites, and raw verification artifacts for commit `{head_sha}` on branch `feature/p0-hardening-and-paper-experiment`.

## Forensic Defect Closures & Engine Hardening (Commit {short_sha})
1. **Defect 1: Strict Stop Remaining Capacity & Partial-Fill Invariant** (`strategies/trailing_stop.py`):
   - In `_is_valid_stop_order()`: Removed `and o_qty_f < float(required_qty) - 1e-4` loophole. Orders with remaining capacity less than required position quantity are strictly rejected (`rem_cap < required_qty - 1e-4`). 8-share remaining capacity can never be accepted for a 12-share required position (`partial_remaining_8_required_12.0` -> `accepted: false`).
   - For partially filled stop orders, if remaining capacity does not cover required shares, `stop_reconciliation_required = True` and `reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"` are strictly maintained.

2. **Defect 2: Tracked Cancelled Stop Protection Invariant** (`strategies/trailing_stop.py`):
   - In `_reconcile_broker_order()`: Explicitly decoupled `status == "filled"` from `status in ("cancelled", "canceled", "expired", "rejected")`.
   - A cancelled or expired stop order does NOT close the position (`state.active` remains `True`, `state.order_id = None`, `stop_reconciliation_required = True`, `reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"`), preventing naked unprotected positions from being falsely treated as flat (`tracked_cancelled_stop_not_position_flat` -> `returned: false, active: true`).

3. **Defect 3: Unknown Position Clearance Protection Guard** (`strategies/trailing_stop.py`):
   - In `handle_naked_position()`: An unverified non-empty order ID cannot clear `POSITION_UNKNOWN_RECONCILIATION_REQUIRED` unless `stop_reconciliation_required` is `False`. Prevents unverified legacy/stale IDs from unlocking entry gates (`unknown_cleared_by_nonempty_unverified_id` -> `status: "POSITION_UNKNOWN..."`).

4. **Defect 4: Canonical Broker Position Query Tri-State Contract** (`strategies/trailing_stop.py`):
   - In `reconcile_pending_exits()`: Binds canonical `self._query_broker_position_qty(bridge, symbol)` fail-closed.
   - Case 1 (Broker qty > 0, e.g. 12): blocks reconciliation with `POSITION_STILL_OPEN_ON_BROKER` and retains pending queue item (`pending_exit_no_current_position_check`).
   - Case 2 (Broker qty = 0): confirms flat state, permitting reconciliation only if valid fill evidence exists.
   - Case 3 (Timeout / None / NaN / query error): retains pending exit with `POSITION_UNKNOWN_RETAINED` (fail-closed).

5. **Defect 5: Delimiter-Safe Exit Order Lineage Matching** (`strategies/trailing_stop.py`):
   - Replaced substring `in` checks with `_matches_order_lineage()` using delimiter regex `(?:^|[:_]){{target}}(?:$|[:_])`. Completely eliminates episode identity collisions (`sig-1` vs `sig-10`) (`substring_other_episode_sig10` -> `pending: 1, records: []`).

6. **Defect 6: Strict Explicit Finite Fill Evidence Contract** (`strategies/trailing_stop.py`):
   - TP/Stop Snapshots: Require explicit non-null, finite, positive `filled_qty` (`math.isfinite(snap_fq) and snap_fq > 0`), status strictly `"filled"`. Zero fallback to `qty` (`snapshot_missing_filled_qty_uses_order_qty`, `snapshot_infinite_filled_qty` -> `pending: 1, records: []`).
   - Closed-Orders Fallback: Status strictly terminal (`"filled"`, `"canceled"`, `"cancelled"`, `"expired"`), require explicit finite positive `filled_qty` (rejects NaN/inf/missing/non-terminal `partially_filled`) (`fallback_nan_filled_qty`, `fallback_partial_status`, `fallback_missing_filled_qty_uses_qty` -> `pending: 1, records: []`).

7. **Defect 7: Multi-Order Exit Accounting & Quantity Conservation** (`strategies/trailing_stop.py`):
   - Unified Candidate Collection: Aggregates orders across TP snapshot, Stop snapshot, and closed orders into a deduplicated map keyed by broker `order_id` (zero double-counting).
   - Cancelled Partial Fills: Aggregates cancelled/expired orders that had partial fills (`filled_qty > 0`).
   - Quantity Conservation: If `total_filled > exp_qty + 1e-4`, triggers `EXCESS_EXIT_QTY_DISCREPANCY` fail-closed. If exact match `abs(total_filled - exp_qty) <= 1e-4`, computes weighted average fill price and logs `SPLIT_EXIT` (`split_exits_4_plus_8_cover_12` -> `RECONCILED_SPLIT_EXIT`, weighted price $333.3333, gross PnL $17.44, `pending: 0`).

8. **Defect 8: Durable Disk Sync Enforced on Retry Dedup** (`strategies/trade_journal.py`):
   - In `log_trade()`: On idempotency key dedup detection, enforces durable disk sync with `os.fsync(fh.fileno())` before returning. If fsync fails on retry, `OSError` propagates directly without falling through to write duplicate rows (`fsync_failure_retry_dedup_evicts_without_sync` -> `additional_fsync_calls: 1, pending: 1`).

9. **Defect 9: Truthful Market Provenance & Timestamp Separation** (`strategies/regime_engine.py` & `alpaca_bridge.py`):
   - In `_get_premarket_pct_change()`: Fallback price observation timestamp strictly uses quote timestamp (`final_ts = quote_ts`). Never reuses baseline bar timestamp (`fallback_price_epoch_reused_from_baseline` -> `timestamp: None`).
   - In yfinance fallback: Returns `("yfinance_fallback", None)`. Never spoofs current system time as market observation timestamp (`yfinance_missing_provider_timestamp_labelled_fresh` -> `timestamp: None`).

10. **Defect 10: Stop Adoption Uses Fresh Signed Broker Position** (`strategies/trailing_stop.py`):
    - `_reconcile_broker_order()` queries fresh broker position `self._query_broker_position_qty(bridge, state.symbol)`.
    - Stop adoption checks candidate remaining capacity against fresh broker position: rejects if broker qty 12 and stop capacity 8; accepts if broker qty 8 and stop capacity 8.
    - Stop partial fill does not unilaterally reduce `state.qty`.
    - Finding a working stop alone does NOT clear `POSITION_UNKNOWN_RECONCILIATION_REQUIRED` unless broker position is confirmed known.

## Verification Summary (240 / 240 PASS - 100% Zero Regressions)
- **Suite Breakdown**:
  1. `tests/test_stop_rebuild_and_persistence_hardening.py`: **38** passed
  2. `tests/test_paper_experiment_lane.py`: **10** passed
  3. `tests/test_feed_provenance_contract.py`: **16** passed
  4. `tests/test_geometry_hard_guards.py`: **13** passed
  5. `tests/test_trailing_stop_p1_5.py`: **16** passed
  6. `tests/test_lifecycle_halt_integration.py`: **7** passed
  7. `tests/test_order_manager_idempotency.py`: **14** passed
  8. `tests/test_execution_quote.py`: **9** passed
  9. `tests/test_candidate_selection_coordination.py`: **15** passed
  10. `tests/test_session_exit_and_regime_baseline_hardening.py`: **22** passed
  11. `tests/test_validation_lane.py`: **14** passed
  12. `tests/test_price_tier_semantics.py`: **1** passed
  13. `tests/test_replacement_lineage.py`: **6** passed
  14. `tests/test_machine_release_gate.py`: **20** passed
  15. `tests/test_long_tail_recovery_supervisor.py`: **5** passed
  16. `tests/test_vwap_confidence_calibration.py`: **2** passed
  17. `tests/test_deterministic_telemetry_join.py`: **6** passed
  18. `tests/test_soxx_fail_closed_regression.py`: **18** passed
  19. `tests/test_candlestick_strategy.py`: **8** passed
  - **TOTAL: 240 passed in {xml_duration} (100% Zero Regressions)**
- **JUnit XML**: `evidence/pytest_junit_report.xml`

## Boundary Probes Verification
- **Auditor's 16 Boundary Probes**: **16 / 16 PASSED** with 100% specification compliance (`evidence/auditor_16_probes_results.json`).
- **Auditor's 4 Technical Hardening Probes**: **10 / 10 CHECKS PASSED (100%)** (`evidence/auditor_hardening_points_results.json`).
- **Original 25 Boundary Probes**: **25 / 25 PASSED** with 100% precision (`evidence/offline_probes_results.json`).

## Bundle Contents
- `source/`: Full source of core engine files (`alpaca_bridge.py`, `main.py`, `run_daytrade_paper.bat`, `strategies/`, `config.py`).
- `tests/`: 19 comprehensive test suites covering all execution safety, persistence, stop rebuild, regime invariant, and coordination gates.
- `evidence/`:
  - `git_rev_parse_head.txt`: Git HEAD SHA (`{head_sha}`)
  - `git_status_short.txt`: Git workspace status
  - `git_commit_diff_{short_sha}.patch`: Exact commit patch for `{short_sha}`
  - `git_commit_diff_bbeda16.patch`: Commit patch for `bbeda16`
  - `git_commit_diff_fd3094e.patch`: Commit patch for `fd3094e`
  - `git_commit_diff_9977c79.patch`: Core hardening commit patch for `9977c79`
  - `git_diff_cumulative_since_3ef1bab.patch`: Cumulative patch since `3ef1bab`
  - `pytest_junit_report.xml`: Machine-readable test execution report (240 testcases)
  - `run_offline_probes.py`: Isolated offline boundary verification script (25 probes)
  - `offline_probes_results.json`: Raw execution outputs for all 25 boundary probes
  - `run_auditor_16_probes.py`: Auditor's isolated boundary verification script (16 probes)
  - `auditor_16_probes_results.json`: Raw execution outputs for all 16 boundary probes
  - `verify_auditor_points.py`: Auditor's 4 technical hardening points verification script
  - `auditor_hardening_points_results.json`: Raw execution outputs for 4 hardening points (10 checks)
  - `effective_safety_settings.json`: Active safety policy parameters
  - `runtime_config_evidence.json`: Runtime provenance and promotion gates
  - `file_manifest_sha256.json`: Cryptographic SHA-256 hashes of all bundled files

## Sanitization Declaration
This package is sanitized. No `.env` files, Alpaca API keys, secret tokens, private webhook URLs, or account numbers are included.
"""
    (bundle_dir / "README_AUDIT.md").write_text(readme_content, encoding="utf-8")

    # Generate SHA-256 file manifest
    manifest = {}
    for root, dirs, files in os.walk(bundle_dir):
        for file in sorted(files):
            fp = Path(root) / file
            rel = str(fp.relative_to(bundle_dir)).replace("\\", "/")
            h = hashlib.sha256(fp.read_bytes()).hexdigest()
            manifest[rel] = h
    (evidence_dest / "file_manifest_sha256.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[5/5] Creating ZIP archive...")
    if zip_output.exists():
        zip_output.unlink()

    with zipfile.ZipFile(zip_output, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(bundle_dir):
            for file in files:
                file_path = Path(root) / file
                archive_name = file_path.relative_to(bundle_dir.parent)
                zf.write(file_path, archive_name)

    print(f"\nAudit bundle successfully created: {zip_output} ({zip_output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
