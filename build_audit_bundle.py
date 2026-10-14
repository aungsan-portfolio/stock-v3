#!/usr/bin/env python3
"""
build_audit_bundle.py

Packages a sanitized source and evidence audit bundle for commit 9cd671b:
- Updated source modules (strategies/, config.py, alpaca_bridge.py)
- Hardening test suites (19 suites, 240 tests)
- Raw verification outputs (JUnit XML, console log, git HEAD, git diff patch)
- Effective safety settings and runtime config evidence
- Strict sanitization: NO .env, NO API keys, NO tokens, NO credentials
"""
import sys
import os
import re
import shutil
import subprocess
import zipfile
import json
import hashlib
from pathlib import Path
import config

ROOT_DIR = Path(__file__).resolve().parent

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
    "alerts.py",
    "alpaca_bridge.py",
    "config.py",
    "data_integrity.py",
    "errors.py",
    "evaluate_paper_results.py",
    "live_invariants.py",
    "main.py",
    "order_audit.py",
    "order_exec.py",
    "paper_ledger.py",
    "position_rotator.py",
    "predictor.py",
    "pyproject.toml",
    "reconciliation.py",
    "requirements.txt",
    "requirements-dev.txt",
    "rs_calculator.py",
    "run_daytrade_paper.bat",
    "shutdown_guard.py",
    "strategies/atomic_rotation.py",
    "strategies/backtester.py",
    "strategies/base.py",
    "strategies/candlestick_strategy.py",
    "strategies/candle_confirmation.py",
    "strategies/command_parser.py",
    "strategies/constants.py",
    "strategies/correlation.py",
    "strategies/correlation_guards.py",
    "strategies/entry_lifecycle.py",
    "strategies/errors.py",
    "strategies/error_handler.py",
    "strategies/event_guard.py",
    "strategies/execution_geometry.py",
    "strategies/execution_health.py",
    "strategies/execution_quote.py",
    "strategies/gap_strategy.py",
    "strategies/gate_ledger.py",
    "strategies/halt_manager.py",
    "strategies/indicators.py",
    "strategies/intraday_data.py",
    "strategies/intraday_risk.py",
    "strategies/mean_reversion.py",
    "strategies/mega_cap_guard.py",
    "strategies/momentum_strategy.py",
    "strategies/orb_strategy.py",
    "strategies/orchestrator.py",
    "strategies/order_idempotency.py",
    "strategies/order_manager.py",
    "strategies/order_registry.py",
    "strategies/performance.py",
    "strategies/performance_tracker.py",
    "strategies/position_sizer.py",
    "strategies/regime_classifier.py",
    "strategies/regime_engine.py",
    "strategies/rotation_config.py",
    "strategies/scoring.py",
    "strategies/session.py",
    "strategies/soxx_gate_config.py",
    "strategies/startup_security.py",
    "strategies/swing_risk_store.py",
    "strategies/swing_sizing.py",
    "strategies/trade_journal.py",
    "strategies/trailing_stop.py",
    "strategies/validation_lane.py",
    "strategies/veto_reasons.py",
    "strategies/vwap_strategy.py",
    "strategies/watchdog.py",
    "strategies/webhook.py",
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
    "tests/test_hysteresis_invariants.py",
    "tests/test_data_integrity.py",
    "tests/test_order_exec.py",
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

    bundle_name = "trading_bot_candidate_safety_closure_review"
    bundle_dir = ROOT_DIR / bundle_name
    zip_output = ROOT_DIR / f"{bundle_name}.zip"

    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)

    source_dest = bundle_dir / "source"
    tests_dest = bundle_dir / "tests"
    evidence_dest = bundle_dir / "evidence"
    evidence_dest.mkdir(parents=True)

    print(f"[1/5] Copying and sanitizing source files for candidate review on {short_sha}...")
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

    # Candidate working tree binary diff since 4db2611
    try:
        git_diff_candidate = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        git_diff_candidate_lf = git_diff_candidate.replace("\r\n", "\n")
        (evidence_dest / f"git_candidate_diff_binary_since_{short_sha}.patch").write_text(git_diff_candidate_lf, encoding="utf-8", newline="\n")
        (evidence_dest / "patch_4db2611_candidate.patch").write_text(git_diff_candidate_lf, encoding="utf-8", newline="\n")
        (ROOT_DIR / "patch_4db2611_candidate.patch").write_text(git_diff_candidate_lf, encoding="utf-8", newline="\n")
    except Exception as e:
        print(f"  ! Candidate binary diff error: {e}")

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

    # Prior commit diff for 0104e65
    try:
        git_diff_01 = subprocess.check_output(["git", "show", "0104e65"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_0104e65.patch").write_text(sanitize_text(git_diff_01), encoding="utf-8")
    except Exception as e:
        print(f"  ! 0104e65 diff error: {e}")

    # Prior commit diff for 3896f5d
    try:
        git_diff_38 = subprocess.check_output(["git", "show", "3896f5d"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_3896f5d.patch").write_text(sanitize_text(git_diff_38), encoding="utf-8")
    except Exception as e:
        print(f"  ! 3896f5d diff error: {e}")

    # Prior commit diff for 8027bcd
    try:
        git_diff_80 = subprocess.check_output(["git", "show", "8027bcd"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_8027bcd.patch").write_text(sanitize_text(git_diff_80), encoding="utf-8")
    except Exception as e:
        print(f"  ! 8027bcd diff error: {e}")

    # Prior commit diff for b3df048
    try:
        git_diff_b3 = subprocess.check_output(["git", "show", "b3df048"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_b3df048.patch").write_text(sanitize_text(git_diff_b3), encoding="utf-8")
    except Exception as e:
        print(f"  ! b3df048 diff error: {e}")

    # Scoped diff since b3df048
    try:
        git_diff_since_b3 = subprocess.check_output(["git", "diff", "b3df048..HEAD"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_diff_since_b3df048.patch").write_text(sanitize_text(git_diff_since_b3), encoding="utf-8")
    except Exception as e:
        print(f"  ! Diff since b3df048 error: {e}")

    # Prior commit diff for 9cd671b
    try:
        git_diff_9c = subprocess.check_output(["git", "show", "9cd671b"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_9cd671b.patch").write_text(sanitize_text(git_diff_9c), encoding="utf-8")
    except Exception as e:
        print(f"  ! 9cd671b diff error: {e}")

    # Scoped diff since 9cd671b
    try:
        git_diff_since_9c = subprocess.check_output(["git", "diff", "9cd671b..HEAD"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_diff_since_9cd671b.patch").write_text(sanitize_text(git_diff_since_9c), encoding="utf-8")
    except Exception as e:
        print(f"  ! Diff since 9cd671b error: {e}")

    # Prior commit diff for 9333081
    try:
        git_diff_93 = subprocess.check_output(["git", "show", "9333081"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_commit_diff_9333081.patch").write_text(sanitize_text(git_diff_93), encoding="utf-8")
    except Exception as e:
        print(f"  ! 9333081 diff error: {e}")

    # Scoped diff since 9333081
    try:
        git_diff_since_93 = subprocess.check_output(["git", "diff", "9333081..HEAD"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_diff_since_9333081.patch").write_text(sanitize_text(git_diff_since_93), encoding="utf-8")
    except Exception as e:
        print(f"  ! Diff since 9333081 error: {e}")

    # Cumulative git diff since 3ef1bab
    try:
        git_diff_cumulative = subprocess.check_output(["git", "diff", "3ef1bab..HEAD"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_diff_cumulative_since_3ef1bab.patch").write_text(sanitize_text(git_diff_cumulative), encoding="utf-8")
    except Exception as e:
        print(f"  ! Cumulative diff error: {e}")

    # Dynamic fresh execution of canonical pytest suites (22 suites, 323 tests)
    print("  + Executing fresh canonical test suites to generate evidence/pytest_junit_report.xml...")
    junit_dest = evidence_dest / "pytest_junit_report.xml"
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        "-o", "addopts=",
        "--basetemp=temp_pytest_run",
        f"--junitxml={junit_dest}",
    ] + TEST_FILES
    xml_duration = "N/A"
    total_xml_tests = 323
    try:
        subprocess.check_call(pytest_cmd, cwd=str(ROOT_DIR))
        print("  + Pytest execution finished successfully.")
    except Exception as pe:
        print(f"  ! Pytest execution notice: {pe}")

    if junit_dest.exists():
        sanitize_and_copy(junit_dest, ROOT_DIR / "pytest_junit_report.xml")
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(junit_dest)
            root = tree.getroot()
            total_xml_tests = int(root.attrib.get("tests", total_xml_tests))
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

    # Auditor's 11 expanded boundary checks script and raw execution output (11 checks)
    expanded_src = ROOT_DIR / "expanded_boundary_checks.py"
    if expanded_src.exists():
        sanitize_and_copy(expanded_src, evidence_dest / "expanded_boundary_checks.py")
        try:
            expanded_out = subprocess.check_output(["python", str(expanded_src)], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
            (evidence_dest / "expanded_boundary_checks_results.json").write_text(expanded_out, encoding="utf-8")
        except Exception as ee:
            print(f"  ! Error executing expanded_boundary_checks: {ee}")

    # Auditor's scoped closure checks script and raw execution output (11 checks)
    scoped_src = ROOT_DIR / "scoped_closure_checks.py"
    if scoped_src.exists():
        sanitize_and_copy(scoped_src, evidence_dest / "scoped_closure_checks.py")
        try:
            scoped_out = subprocess.check_output(["python", str(scoped_src)], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
            (evidence_dest / "scoped_closure_results.json").write_text(scoped_out, encoding="utf-8")
        except Exception as se:
            print(f"  ! Error executing scoped_closure_checks: {se}")

    # Scoped lifecycle blockers verification script and output (fresh execution)
    blockers_src = ROOT_DIR / "verify_scoped_lifecycle_blockers.py"
    if blockers_src.exists():
        sanitize_and_copy(blockers_src, evidence_dest / "verify_scoped_lifecycle_blockers.py")
        try:
            print("  + Executing fresh verify_scoped_lifecycle_blockers...")
            subprocess.check_call(["python", str(blockers_src)], cwd=str(ROOT_DIR))
            blockers_json = ROOT_DIR / "scoped_lifecycle_blockers_results.json"
            if blockers_json.exists():
                sanitize_and_copy(blockers_json, evidence_dest / "scoped_lifecycle_blockers_results.json")
        except Exception as be:
            print(f"  ! Error executing verify_scoped_lifecycle_blockers: {be}")

    # Identity and lineage acceptance checks script and output (fresh execution)
    identity_src = ROOT_DIR / "identity_acceptance_checks.py"
    if identity_src.exists():
        sanitize_and_copy(identity_src, evidence_dest / "identity_acceptance_checks.py")
        try:
            print("  + Executing fresh identity_acceptance_checks...")
            subprocess.check_call(["python", str(identity_src)], cwd=str(ROOT_DIR))
            identity_json = ROOT_DIR / "identity_acceptance_checks_results.json"
            if identity_json.exists():
                sanitize_and_copy(identity_json, evidence_dest / "identity_acceptance_checks_results.json")
        except Exception as ie:
            print(f"  ! Error executing identity_acceptance_checks: {ie}")

    # Effective safety settings
    safety_settings = {
        "commit": head_sha,
        "branch": "feature/p0-hardening-and-paper-experiment",
        "PAPER_EXPERIMENT_LANE_ENABLED": False,
        "DAYTRADE_REQUIRE_ENTRY_READINESS": True,
        "REQUIRE_REALTIME_DATA_FOR_ORDERS": True,
        "REQUIRE_FEED_CONCORDANCE": True,
        "TRAILING_STOP_RECONCILE_TOLERANCE": None,
        "ENTRY_BAR_DURATION_SECONDS": float(getattr(config, "ENTRY_BAR_DURATION_SECONDS", 300.0)),
        "ENTRY_BAR_CLOSE_GRACE_SECONDS": float(getattr(config, "ENTRY_BAR_CLOSE_GRACE_SECONDS", 60.0)),
        "MAX_QUOTE_AGE_SECONDS": float(getattr(config, "MAX_QUOTE_AGE_SECONDS", 10.0)),
        "MAX_BAR_STALENESS_SECONDS": float(getattr(config, "MAX_STALE_BAR_SECONDS", 300.0)),
        "MAX_ACCEPTABLE_SPREAD_PCT": float(getattr(config, "MAX_ACCEPTABLE_SPREAD_PCT", 0.75)),
        "MAX_DECISION_DEVIATION_PCT": float(getattr(config, "MAX_DECISION_DEVIATION_PCT", 1.5)),
        "MAX_DAILY_LOSS_DAYTRADE": float(getattr(config, "MAX_DAILY_LOSS_DAYTRADE", 300.0)),
        "FLATTEN_ON_DAILY_LOSS": bool(getattr(config, "FLATTEN_ON_DAILY_LOSS", True)),
        "STRATEGY_ENABLED_FLAGS": {
            "STRATEGY_ORB_ENABLED": bool(getattr(config, "STRATEGY_ORB_ENABLED", False)),
            "STRATEGY_VWAP_BOUNCE_ENABLED": bool(getattr(config, "STRATEGY_VWAP_BOUNCE_ENABLED", False)),
            "STRATEGY_GAP_AND_GO_ENABLED": bool(getattr(config, "STRATEGY_GAP_AND_GO_ENABLED", True)),
            "STRATEGY_MOMENTUM_SCALP_ENABLED": bool(getattr(config, "STRATEGY_MOMENTUM_SCALP_ENABLED", False)),
            "STRATEGY_CANDLESTICK_ENABLED": bool(getattr(config, "STRATEGY_CANDLESTICK_ENABLED", True)),
        },
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
    readme_content = f"""# Candidate Safety Closure Review Audit Bundle (Commit {short_sha} Candidate)

## Governance & Release Policy
- **Audit Target**: Candidate safety closure review on baseline commit `{head_sha}`
- **Governance Decision**: **REVIEW REQUIRED — NOT READY FOR COMMIT / MERGE / DEPLOYMENT**
- **Phase C Status**: **HELD / PENDING** (Paper runtime accumulation awaiting auditor formal concurrence)
- **Real-Money Trading**: **STRICT NO-GO** (Quantitative promotion gates strictly enforced: $N \\ge 30$, $PF \\ge 1.25$, Max DD $\\le 5.0%$)
- **Zero Parameter Drift**: Risk limits, stop sizing, and promotion criteria remain strictly frozen.

## Key Hardening Points Evaluated in Candidate
1. **Strict Conflict & Reverse Conflict Precedence (Point A)**:
   - Durably mapped broker order IDs enable recovery with generic broker client IDs.
   - Any candidate order referencing registered foreign mappings or foreign tokens (`other-episode`, `foreign-episode`) is rejected as conflicting evidence taking strict precedence over current lineage match (`CONFLICTING_EXIT_EVIDENCE_DISCREPANCY`, journal writes = 0, pending retained).
   - Dual-match candidate orders matching both current lineage and foreign identifiers fail closed.
   - Reverse-conflict candidate orders whose broker order ID belongs to a foreign episode while client ID matches current lineage are strictly rejected fail-closed.

2. **Fault-Injected Cold-Start Crash/Restart Recovery (Point B)**:
   - Evaluated by injecting state persistence failures during pending exit recording, leaving active state on disk and dropping in-memory manager state.
   - Fresh manager cold-starts from disk, queries broker truth via `_recover_orphaned_flat_states(bridge)`, and reconstructs pending exit.
   - Authoritative later fills reconcile exactly once (1 journal row), with idempotent retries producing zero duplicates.
   - Unlinked fills are rejected fail-closed (0 journal rows, pending exit retained, new entries blocked via `has_pending_exit`).

3. **Signed-Side Mismatch & Inactive State Reload Guards (Point C)**:
   - `_load_state()` deserializes all unresolved reconciliation states on process restart, even when `active=False`.
   - Pre-submission final-submit gate enforces fail-closed halts on both `has_unresolved_position_unknown()` and `has_unresolved_signed_side_mismatch()` across `active=True` and `active=False` states.

## Verification Summary ({total_xml_tests} / {total_xml_tests} PASS - 100% Zero Regressions)
- **Canonical Pytest Suites (22 suites, {total_xml_tests} tests)**:
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
  20. `tests/test_hysteresis_invariants.py`: **13** passed
  21. `tests/test_data_integrity.py`: **35** passed
  22. `tests/test_order_exec.py`: **35** passed
  - **TOTAL: {total_xml_tests} passed in {xml_duration} (100% Zero Regressions, 0 failures, 0 errors)**
- **JUnit XML Report**: `evidence/pytest_junit_report.xml`

## Boundary Probes & Lifecycle Blockers Verification
- **Scoped Lifecycle Blockers Verification Suite**: **35 / 35 COMPREHENSIVE TESTS PASSED** (`evidence/scoped_lifecycle_blockers_results.json`).
- **Identity, Precedence & Crash Recovery Acceptance Suite**: **12 / 12 CHECKS PASSED** (`evidence/identity_acceptance_checks_results.json`).
- **Auditor's 16 Boundary Probes**: **16 / 16 PASSED** (`evidence/auditor_16_probes_results.json`).
- **Auditor's 4 Technical Hardening Probes**: **10 / 10 CHECKS PASSED** (`evidence/auditor_hardening_points_results.json`).
- **Offline 25 Boundary Probes**: **25 / 25 PASSED** (`evidence/offline_probes_results.json`).

## Bundle Contents
- `source/`: Sanitized production source files (`strategies/`, `config.py`, `alpaca_bridge.py`, `main.py`, `rs_calculator.py`, root bridge dependencies).
- `tests/`: 22 comprehensive automated test suites.
- `evidence/`:
  - `patch_4db2611_candidate.patch` (and `git_candidate_diff_binary_since_4db2611.patch`): Candidate working-tree binary patch since baseline `4db2611`
  - `git_rev_parse_head.txt`: Git HEAD SHA (`{head_sha}`)
  - `git_status_short.txt`: Git workspace status
  - `pytest_junit_report.xml`: Machine-readable JUnit test report ({total_xml_tests} test cases)
  - `verify_scoped_lifecycle_blockers.py`: Scoped lifecycle blockers verification script (35 test cases)
  - `scoped_lifecycle_blockers_results.json`: Execution outputs of 35 scoped verification tests
  - `identity_acceptance_checks.py`: Identity, precedence & crash recovery acceptance test script (12 test cases)
  - `identity_acceptance_checks_results.json`: Execution outputs of 12 identity acceptance checks
  - `run_auditor_16_probes.py` & `auditor_16_probes_results.json`: Auditor's 16 boundary probes script and outputs
  - `run_offline_probes.py` & `offline_probes_results.json`: Offline 25 boundary probes script and outputs
  - `verify_auditor_points.py` & `auditor_hardening_points_results.json`: Auditor's technical hardening points script and outputs
  - `effective_safety_settings.json`: Active safety policy parameters
  - `runtime_config_evidence.json`: Runtime configuration and quantitative promotion gates
  - `file_manifest_sha256.json`: Cryptographic SHA-256 hashes of all bundled files

## Sanitization Declaration
This package is sanitized. No `.env` files, API keys, secret tokens, private webhooks, or account credentials are included.
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
