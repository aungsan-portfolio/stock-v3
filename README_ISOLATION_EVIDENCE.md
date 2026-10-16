# Step 1: Test / Runtime Audit Isolation — Verification & Evidence Report

**Document Version**: 2.1 (Hardened Single Revision — Fully Reconciled)  
**Date**: 2026-10-16  
**Status**: COMPLETE -- Ready for Acceptance Review  
**Git Baseline Commit**: `c70f283` (pre-Step-1 clean state)  
**Final Hardened Commits**:
- `a73614c` (Step 1 Guard Hardening, Rename/Replace checks, io.open, bytes paths, root overlap)
- `a318821` (Maintenance Preservation: GAP veto logging, stale bar test fixture)

---

## 1. Executive Summary

This revision addresses and resolves all 4 confirmed blockers and all 3 missing acceptance evidence gaps identified in the independent acceptance review of Step 1 (Test / Runtime Audit Isolation).

Every confirmed probe bypass vector has been hardened with fail-closed production write guards. Full double-run test execution across **65 selected tests** (including `tests/test_validation_lane.py` and trailing stop regression suites) passed with zero failures and zero mutations across all 1,704 tracked production files. Prior maintenance changes are 100% preserved.

---

## 2. Confirmed Blockers Resolved

### Blocker 1: Restored `return "unrelated"` in `strategies/trailing_stop.py`
- **Root Cause**: An earlier patch omitted `return "unrelated"` from `_classify_held_order()`, causing unmatched orders to fall through to `None`.
- **Resolution**: Restored `return "unrelated"` as the final return in `strategies/trailing_stop.py::_classify_held_order()`.
- **Verification**: Added `test_classify_held_order_regression()` verifying that unmatched orders, arbitrary `client_order_id` values, empty orders, and `None` strictly evaluate to `"unrelated"`. Verified all 18 trailing stop safety regression tests (`test_trailing_stop_p1_5.py` and `test_trailing_stop_gap_through.py`) pass cleanly.

### Blocker 2: Both `src` and `dst` Checked for Mutations (`rename`/`replace`)
- **Root Cause**: `_guarded_os_rename`, `_guarded_os_replace`, `_guarded_path_rename`, and `_guarded_path_replace` only verified destination paths, allowing production sources to be moved or deleted into isolated directories.
- **Resolution**: Updated all 4 hooks to check both source (`src`/`self`) and destination (`dst`/`target`). Any mutation targeting or originating from protected production paths raises `TestIsolationViolationError` and records a ledger entry before filesystem execution.
- **Verification**: Probes `'rename production source to isolated destination'` and `'replace production source to isolated destination'` both successfully block (`blocked=True`, `ledger_count > 0`).

### Blocker 3: Guarded `io.open` and Normalized `bytes` Paths
- **Root Cause**: `io.open` bypassed `builtins.open`, and raw `bytes` paths passed to `open()` bypassed path resolution because path coercion failed open.
- **Resolution**:
  - Implemented `_normalize_target_path(target)` using `os.fsdecode()`, supporting `bytes`, `bytearray`, `str`, `Path`, and `os.PathLike`. File descriptors (`int`) are recognized and exempt.
  - Guarded `io.open` and `_io.open` in `config.install_production_write_guard()`.
  - Fail-closed invariant: Any unresolvable path matching protected directory markers (`/data/`, `/logs/`, `/reports/`) is unconditionally blocked.
- **Verification**: Probes `'io.open direct production write'` and `'builtins.open bytes production path'` both successfully block (`blocked=True`, `ledger_count > 0`).

### Blocker 4: Test Isolation Root Validation
- **Root Cause**: `init_isolated_test_env` accepted roots nested within `BASE_DIR / data`, creating unintended write exemptions inside production directories.
- **Resolution**: Created `validate_test_isolation_root(root)` which resolves paths and strictly rejects any candidate test root overlapping production trees in either direction (ancestor, descendant, or equal to `BASE_DIR`, `BASE_DIR/data`, `BASE_DIR/logs`, `BASE_DIR/reports`). Validation occurs **before** directory creation or environment variable mutation.
- **Verification**: Added `test_guard_rejects_overlapping_production_roots` and probe `'test root nested in production'` which raises `ValueError` and prevents initialization.

---

## 3. Acceptance Evidence Gaps Closed

### 1. Multi-Process Isolation Test (`test_multiprocess_workers_distinct_isolated_roots`)
- Added `test_multiprocess_workers_distinct_isolated_roots` spawning two independent Python worker subprocesses with distinct `STOCK_ENGINE_TEST_ROOT` directories.
- Verified both processes execute state operations (`EntryIdempotencyStore.reserve()`, `order_audit.log_event()`), creating separate state and lock files in their respective roots with zero lock contention, zero state collision, and zero pollution of production trees.
- Clarification: Formally classified as a **multi-process isolation test** across distinct OS processes.

### 2. Child-Process Unexpected Violation Propagation
- Added `test_subprocess_unexpected_violation_propagates_and_fails`.
- Demonstrated that if a child process attempts an unhandled illegal write to production directories, the process crashes with a non-zero exit code and `TestIsolationViolationError` in `stderr`, guaranteeing that child violations fail parent runners and test orchestrators.

### 3. Manifest Comparator Error-Token Rejection
- Hardened `tools/compare_manifests.py` to validate:
  - `sha256`: Strictly 64 hexadecimal characters `[0-9a-fA-F]`.
  - `size_bytes`: Non-negative integer.
- If any manifest contains `"ERROR: Permission denied"`, unreadable tokens, or malformed hashes, the comparator records them under `[MANIFEST INTEGRITY ERRORS]` and unconditionally returns `False` (`RESULT: FAIL`).
- Added `test_compare_manifests_rejects_error_tokens_and_malformed_hashes` verifying that matching ERROR strings fail comparison.

### 4. Forward Patch Application on Baseline `c70f283`
- Verified via `git apply --check` that `step1_isolation_scoped.patch` applies **forward** onto a clean checkout of baseline `c70f283` with zero errors, zero offsets, and zero fuzz.

### 5. Maintenance Changes Preserved
- Confirmed that GAP continuation veto and emitted signal logging in `strategies/gap_strategy.py` are preserved.
- Confirmed that stale-bar grace period mock in `tests/test_p1_6_remediation.py` is preserved and passing.
- Confirmed that `use_batch` and batch breadth fallback in `strategies/regime_engine.py` are preserved.

---

## 4. Empirical Test Results

### Double Test Run Outputs (Raw files included in bundle)
- **Run 1**: 65 passed in 31.73s (exit code 0) -> `test_run_summary_1.txt`
- **Run 2**: 65 passed in 30.60s (exit code 0) -> `test_run_summary_2.txt`

Suites included:
1. `tests/test_runtime_audit_isolation.py` (23 tests -- all passing)
2. `tests/test_journal_isolation.py` (10 tests -- all passing)
3. `tests/test_validation_lane.py` (14 tests -- all passing)
4. `tests/test_trailing_stop_p1_5.py` (15 tests -- all passing)
5. `tests/test_trailing_stop_gap_through.py` (3 tests -- all passing)
**Total: 65 tests passed, 0 failures, 0 regressions.**

### Cryptographic Manifest Comparison (`manifest_diff_report.txt`)
- Before Manifest: 1,704 files tracked (`audit_manifest_before.json`)
- After Manifest: 1,704 files tracked (`audit_manifest_after.json`)
- Excluded Paths: `data/cache`
- Integrity Errors: 0
- Files Added: 0
- Files Removed: 0
- Files Modified: 0
- **RESULT: PASS -- 100% Zero Mutations Across All Tracked Production Files.**

### User Probe Verification (`guard_probe_results.json`)
All 6 user probe vectors pass with `blocked=True`:
1. `normal builtins.open production write`: blocked=True
2. `rename production source to isolated destination`: blocked=True
3. `replace production source to isolated destination`: blocked=True
4. `io.open direct production write`: blocked=True
5. `builtins.open bytes production path`: blocked=True
6. `test root nested in production`: blocked=True

---

## 5. Git Diff & Packaging Details

- **Encoding**: Pure UTF-8 binary export.
- **Baseline Git Commit**: `c70f283`
- **Hardened Commits**: `a73614c`, `a318821`
- **Supplied Patches**:
  - `step1_isolation_scoped.patch`: Scoped exclusively to the 16 core audit isolation files (73,271 bytes, UTF-8).
  - `step1_isolation_full.patch`: Full diff including maintenance and breadth telemetry (87,708 bytes, UTF-8).
  - `step1_isolation.patch`: Standard UTF-8 patch for drop-in application.
