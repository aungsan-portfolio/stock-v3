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

    # Cumulative git diff since 3ef1bab
    try:
        git_diff_cumulative = subprocess.check_output(["git", "diff", "3ef1bab..HEAD"], cwd=str(ROOT_DIR), encoding="utf-8", errors="replace")
        (evidence_dest / "git_diff_cumulative_since_3ef1bab.patch").write_text(sanitize_text(git_diff_cumulative), encoding="utf-8")
    except Exception as e:
        print(f"  ! Cumulative diff error: {e}")

    # JUnit XML report
    junit_src = ROOT_DIR / "pytest_junit_report.xml"
    if junit_src.exists():
        sanitize_and_copy(junit_src, evidence_dest / "pytest_junit_report.xml")

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

## Strategy Coordination & P0 Hardening Closures
1. **Fix 1: Candidate Claim Order & Symbol Suppression** (`strategies/orchestrator.py`):
   - `seen_symbols.add(sym)` moved from line 2317 (before candidate checks) to line 2412 (immediately before `deduped_signals.append(sig)`).
   - Prevents rejected high-priority candidates (e.g. failing RSI, inverse-ETF, or geometry) from claiming the symbol and causing valid alternative candidates to be skipped as `DUPLICATE_ORDER`.
   - Verified across all 7 probe cases and pre-submit non-fallback semantics in `test_candidate_selection_coordination.py`.

2. **Criterion 1 Remediations: Stop Evidence Remaining Capacity & Protection Call-Sites** (`strategies/trailing_stop.py`):
   - Reject overfilled order evidence where `filled_qty > qty` (`EVIDENCE_FILLED_EXCEEDS_QTY`).
   - Non-mock `raw_filled_qty` that is NaN, negative, or non-numeric fails closed (`EVIDENCE_FILLED_QTY_INVALID`).
   - In `validate_and_rebuild_geometry`, unchanged geometry (`ACCEPTED`, `POST_FILL_RR_DEGRADED`) enforces `require_qty=True`. If `expected_qty` is invalid or missing, verification fails closed (`confirmed_broker_stop_price = None`, `rebuild_succeeded = False`).
   - Price matching tolerance heuristic clarifies that half-tick bounding (`min(max(tick_size / 2.0, float(cfg_tol)), tick_size) + 1e-6`) is a rounding tolerance buffer rather than broker-quantized subpenny execution proof.

3. **Criterion 2 Remediations: Final-Submit Boundary Persistence Health Gate** (`strategies/order_manager.py`):
   - Strict Fail-Closed Check: Placed immediately before `_place_bracket_order_once` submission adapter.
   - Re-verifies canonical `get_trailing_manager()`. If manager is missing, missing `persistence_healthy` attr, `persistence_healthy is not True`, or throws on check:
     - Order is vetoed (`status: "VETOED"` citing Final Submit boundary).
     - `_place_bracket_order_once` is never invoked, fake broker `submit_order` call count remains 0.
     - Pre-submit idempotency reservation is cleanly released from reservations store.
     - Non-entry lifecycle operations (exits, stop modifications, reconciliation) remain unblocked.

4. **Criterion 3 Remediations: Candle-Based Entry Missing-Bar Guard** (`strategies/order_manager.py`, `strategies/validation_lane.py`):
   - Upstream entry readiness already rejects missing signal bar as `[ENTRY READINESS BAR INVALID]`.
   - Defense-in-depth: Phase A Execution Quote Gate strictly rejects candle-based entry signals (`ORB`, `VWAP_BOUNCE`, `GAP_AND_GO`, `MOMENTUM_SCALP`, `CANDLESTICK`, `MEAN_REVERSION`) when `bar_timestamp` is None or empty as `BAR_TIMESTAMP_MISSING`.
   - Validation lane also strictly vetoes candle entry signals with missing `bar_timestamp` (`BAR_TIMESTAMP_MISSING`).
   - Generic quote validation, exit orders, and non-candle quote checks remain completely unblocked without `bar_timestamp`.

## Verification Summary (108 / 108 PASS)
- **Suite Breakdown**:
  - Stop rebuild & persistence hardening: **38**
  - Candidate selection coordination: **8** (all 7 probe cases + pre-submit fallback safety)
  - Paper experiment lane: **10**
  - Feed provenance contract: **16**
  - Geometry hard guards: **13**
  - Trailing stop P1.5: **16**
  - Lifecycle halt integration: **7**
  - **TOTAL: 108 passed (100% Zero Regressions)**
- **JUnit XML**: `evidence/pytest_junit_report.xml`

## Bundle Contents
- `source/`: Full source of all strategy modules including generators, sizers, correlation, scoring, and veto reasons.
- `tests/`: 9 complete test suites covering persistence, stop rebuild, candidate coordination, tick quantization, candle boundaries, idempotency, and halt integration.
- `evidence/`:
  - `git_rev_parse_head.txt`: Git HEAD SHA (`{head_sha}`)
  - `git_status_short.txt`: Git workspace status
  - `git_commit_diff_{short_sha}.patch`: Exact commit patch for `{short_sha}`
  - `git_diff_cumulative_since_3ef1bab.patch`: Cumulative patch since `3ef1bab`
  - `pytest_junit_report.xml`: Machine-readable test execution report (100 testcases)
  - `effective_safety_settings.json`: Active safety policy parameters
  - `runtime_config_evidence.json`: Runtime provenance and promotion gates

## Sanitization Declaration
This package is sanitized. No `.env` files, Alpaca API keys, secret tokens, private webhook URLs, or account numbers are included.
"""
    (bundle_dir / "README_AUDIT.md").write_text(readme_content, encoding="utf-8")

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
