"""
tests/conftest.py — Global Test Configuration & Production Audit Isolation.

Guarantees:
1. Production directories (data/, logs/, reports/) and persistent state files
   (order_audit.jsonl, broker_events.jsonl, trade_journal.jsonl, paper_trades.jsonl,
   daily_risk_state.json, order_idempotency_state.json, trailing_state.json,
   trade_gate_decision.jsonl, paper_validation_trades.jsonl, and process locks)
   are strictly isolated during automated pytest runs.
2. An isolated temporary test root is allocated before application test modules
   are executed, re-pointing config.DATA_DIR, config.LOG_DIR, and config.REPORTS_DIR.
3. A strict write-protection guard fails closed with TestIsolationViolationError
   (a BaseException subclass that cannot be swallowed by `except Exception:`)
   if any test or runtime component attempts to write, modify, append, or delete
   files inside real production directories.
4. Concurrent and subprocess isolation: worker-isolated directories support
   xdist, and child processes automatically inherit write protection.
5. All violations are tracked in a session ledger, unconditionally failing
   individual tests in teardown and the entire test run in sessionfinish.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Set

import pytest

# Determine the absolute workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Create an isolated temporary test root for this test session / worker
_worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
_prefix = f"se_test_isolation_{_worker_id}_" if _worker_id else "se_test_isolation_"
_SESSION_TEST_ROOT = Path(tempfile.mkdtemp(prefix=_prefix)).resolve()

os.environ["STOCK_ENGINE_TEST_ROOT"] = str(_SESSION_TEST_ROOT)
os.environ["TEST_RUNNING"] = "1"

# Initialize isolated directories
(_SESSION_TEST_ROOT / "data").mkdir(parents=True, exist_ok=True)
(_SESSION_TEST_ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)
(_SESSION_TEST_ROOT / "logs").mkdir(parents=True, exist_ok=True)
(_SESSION_TEST_ROOT / "logs" / "incidents").mkdir(parents=True, exist_ok=True)
(_SESSION_TEST_ROOT / "reports").mkdir(parents=True, exist_ok=True)

# Copy static read-only seed data if present in production data
for static_name in ["symbol_universe.csv"]:
    src_file = WORKSPACE_ROOT / "data" / static_name
    if src_file.exists():
        try:
            shutil.copy2(src_file, _SESSION_TEST_ROOT / "data" / static_name)
        except Exception:
            pass

# Import config, initialize the isolated test environment, and install the write guard
import config
from config import TestIsolationViolationError, SESSION_VIOLATIONS

config.init_isolated_test_env(_SESSION_TEST_ROOT)
config.install_production_write_guard()


@pytest.fixture(scope="session")
def test_isolation_root() -> Path:
    """Returns the isolated test root directory for the active test session."""
    return _SESSION_TEST_ROOT


@pytest.fixture(autouse=True)
def enforce_test_isolation_roots(monkeypatch):
    """Autouse fixture ensuring config paths remain pinned to isolated test root."""
    config.init_isolated_test_env(_SESSION_TEST_ROOT)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Fail test unconditionally if any unauthorized production write occurred."""
    outcome = yield
    if config.SESSION_VIOLATIONS:
        violations = list(config.SESSION_VIOLATIONS)
        config.SESSION_VIOLATIONS.clear()
        pytest.fail(
            f"TEST ISOLATION VIOLATION: Test '{item.name}' attempted unauthorized "
            f"mutation on protected production path: {violations}"
        )


def pytest_sessionfinish(session, exitstatus):
    """Fail the session if any unhandled production write violations remain recorded."""
    if config.SESSION_VIOLATIONS:
        session.exitstatus = 1
