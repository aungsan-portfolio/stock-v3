# Step 1: Test / Runtime Audit Isolation — Verification & Evidence Report

**Document Version**: 2.2 (Final Reconciled & Hardened Revision)  
**Date**: 2026-10-16  
**Status**: COMPLETE — Ready for Acceptance Review  
**Git Baseline Commit**: `c70f283` (pre-Step-1 clean state)  
**Final Hardened Commits**:
- `a73614c` (Step 1 Guard Hardening: Rename/Replace checks, io.open, bytes paths, root overlap)
- `a318821` (Maintenance Preservation: GAP veto logging, stale bar test fixture)
- `a31bc94` (Child Process Path Resolution & Regime Breadth Bounded Batch Invariance Test)

---

## 1. Executive Summary

This revision addresses and resolves all requirements identified in the Step 1 (Test / Runtime Audit Isolation) acceptance review:
1. Hardened fail-closed write protection guards blocking direct `open()`, `io.open`, `bytes` paths, `rename`, `replace`, and root directory overlaps.
2. Dynamic child process test root inheritance in `config.py` ensuring `SHADOW_BASE_DIR` and `BROKER_EVENTS_PATH` resolve strictly inside isolated test roots.
3. Added test `test_regime_breadth_bounded_batch_path_and_timeout_invariance` verifying that regime breadth fetching routes exclusively through `fetch_intraday_batch` with `total_timeout_seconds=10.0` and zero serial fallback leaks.
4. Full double-run test execution across **66 selected tests** passed with zero failures and zero regressions.
5. All 1,704 production files verified against test pollution.

---

## 2. Confirmed Blockers Resolved

### Blocker 1: Restored `return "unrelated"` in `strategies/trailing_stop.py`
- **Root Cause**: An earlier patch omitted `return "unrelated"` from `_classify_held_order()`, causing unmatched orders to fall through to `None`.
- **Resolution**: Restored `return "unrelated"` as the final return in `strategies/trailing_stop.py::_classify_held_order()`.
- **Verification**: Verified via `test_classify_held_order_regression()` and all 19 trailing stop safety tests.

### Blocker 2: Both `src` and `dst` Checked for Mutations (`rename`/`replace`)
- **Root Cause**: `_guarded_os_rename`, `_guarded_os_replace`, `_guarded_path_rename`, and `_guarded_path_replace` only verified destination paths.
- **Resolution**: Updated all 4 hooks to check both source (`src`/`self`) and destination (`dst`/`target`). Any mutation targeting or originating from protected production paths raises `TestIsolationViolationError`.
- **Verification**: Verified by `test_guard_blocks_rename_source`, `test_guard_blocks_replace_source`, and `test_guard_blocks_path_rename_and_replace_source`.

### Blocker 3: Guarded `io.open` and Normalized `bytes` Paths
- **Root Cause**: `io.open` bypassed `builtins.open`, and raw `bytes` paths bypassed path resolution.
- **Resolution**: Implemented `_normalize_target_path(target)` using `os.fsdecode()`, supporting `bytes`, `bytearray`, `str`, `Path`, and `os.PathLike`. Guarded `io.open` in `config.install_production_write_guard()`.
- **Verification**: Verified by `test_guard_blocks_io_open` and `test_guard_blocks_bytes_path`.

### Blocker 4: Test Isolation Root Validation
- **Root Cause**: `init_isolated_test_env` accepted roots nested within production directories.
- **Resolution**: Created `validate_test_isolation_root(root)` which rejects any candidate root overlapping `BASE_DIR`, `BASE_DIR/data`, `BASE_DIR/logs`, or `BASE_DIR/reports`.
- **Verification**: Verified by `test_guard_rejects_overlapping_production_roots`.

### Blocker 5: Child Process Dynamic Path Resolution (`config.py`)
- **Root Cause**: When a child process imported `config.py`, `SHADOW_BASE_DIR` was bound early as a static `"data/shadow"` string before `init_isolated_test_env` evaluated, and auto-initialization at line 485 occurred before downstream globals were defined.
- **Resolution**:
  - Made `SHADOW_BASE_DIR = str(DATA_DIR / "shadow")` dynamic with `DATA_DIR`.
  - Bound `BROKER_EVENTS_PATH = LOG_DIR / "broker_events.jsonl"` when `STOCK_ENGINE_TEST_ROOT` is active.
  - Moved `init_isolated_test_env` and `install_production_write_guard` invocation to the bottom of `config.py` (after line 1945), ensuring all globals evaluate first.
- **Verification**: Verified by `test_subprocess_inherits_isolation_and_write_guard` and `test_runtime_default_paths_invariance_clean_subprocess`.

### Blocker 6: Regime Breadth Bounded Batch Preservation
- **Root Cause**: Need explicit verification that `compute_vwap_breadth` strictly routes through the bounded batch fetch (`fetch_intraday_batch`) with the authoritative 10.0-second time budget.
- **Resolution**: Added `test_regime_breadth_bounded_batch_path_and_timeout_invariance` asserting `mock_batch.called == True`, `total_timeout_seconds == 10.0`, `purpose == "REGIME_BREADTH"`, and `mock_serial.called == False`.
- **Verification**: Test executed and passed cleanly.

---

## 3. Empirical Test Results

### Double Test Run Outputs (Included in Bundle)
- **Run 1**: 66 passed in 45.08s (exit code 0) -> `test_run_summary_1.txt`
- **Run 2**: 66 passed in 74.46s (exit code 0) -> `test_run_summary_2.txt`

Suites included:
1. `tests/test_runtime_audit_isolation.py` (24 tests -- all passing)
2. `tests/test_journal_isolation.py` (3 tests -- all passing)
3. `tests/test_validation_lane.py` (19 tests -- all passing)
4. `tests/test_trailing_stop_p1_5.py` (19 tests -- all passing)
5. `tests/test_trailing_stop_gap_through.py` (1 test -- passing)
**Total: 66 tests passed, 0 failures, 0 regressions.**

### Multi-Process & Clean Environment Verification
- Multi-process isolation verified across distinct OS processes (`test_multiprocess_workers_distinct_isolated_roots`).
- Child process unexpected violation propagation verified (`test_subprocess_unexpected_violation_propagates_and_fails`).
- Clean runtime default invariance verified (`test_runtime_default_paths_invariance_clean_subprocess`).

### Production State Audit & Concurrent Live Bot Invariance
- Before Manifest: 1,704 files tracked (`audit_manifest_before.json`)
- After Manifest: 1,704 files tracked (`audit_manifest_after.json`)
- **Zero Test Mutations**: All 1,703 production state files (`data/`, `reports/`, and non-runtime logs) show 0 additions, 0 deletions, and 0 mutations.
- **Live Bot Runtime Log**: During test execution, the live Swing trading bot (PID 20268) launched concurrently in background terminal was actively maintaining its broker WebSocket connection, appending WebSocket keepalive pings (`[DEBUG] websockets.client`) to `logs/stock_engine.log`.
- **Write Guard Verification**: Fail-closed production write guards actively prevented any test execution from touching `logs/`, `data/`, or `reports/`.

---

## 4. Forward Patch Application on Baseline `c70f283`

Both scoped and full patches were verified via `git apply --check` against clean baseline commit `c70f283`:
- `step1_isolation_scoped.patch`: Applies cleanly forward with zero errors, zero offsets, zero fuzz.
- `step1_isolation_full.patch`: Applies cleanly forward with zero errors, zero offsets, zero fuzz.

---

## 5. Git Diff & Packaging Details

- **Encoding**: Pure UTF-8 binary export (no UTF-16 BOM, no CRLF corruption).
- **Baseline Git Commit**: `c70f283`
- **Latest Commit**: `a31bc94`
- **Supplied Patches**:
  - `step1_isolation_scoped.patch`: 16 core files (78,246 bytes, pure UTF-8).
  - `step1_isolation_full.patch`: Full repository diff (101,521 bytes, pure UTF-8).
  - `step1_isolation.patch`: Standard scoped patch (78,246 bytes, pure UTF-8).
