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
3. A strict write-protection guard fails closed with an AssertionError if any test
   attempts to write, modify, append, or delete files inside the real production
   directories (data/, logs/, reports/).
"""
from __future__ import annotations

import builtins
import os
from pathlib import Path
import shutil
import tempfile
from typing import Set

import pytest

# Determine the absolute workspace root
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Define the production directories that tests MUST NEVER mutate
PROTECTED_DIRS = {
    (WORKSPACE_ROOT / "data").resolve(),
    (WORKSPACE_ROOT / "logs").resolve(),
    (WORKSPACE_ROOT / "reports").resolve(),
}

# Create an isolated temporary test root for this test session
_SESSION_TEST_ROOT = Path(tempfile.mkdtemp(prefix="se_test_isolation_")).resolve()
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

# Import config and initialize the isolated test environment
import config
config.init_isolated_test_env(_SESSION_TEST_ROOT)


def _is_protected_target(target: os.PathLike | str | int) -> bool:
    """Check if target resolves inside any of the production protected directories."""
    if isinstance(target, int):
        return False
    try:
        resolved = Path(target).resolve()
    except Exception:
        return False

    for pdir in PROTECTED_DIRS:
        if resolved == pdir or pdir in resolved.parents:
            return True
    return False


# ── Install Production Write-Protection Tripwire ──

_orig_builtins_open = builtins.open
_orig_os_open = os.open
_orig_os_replace = os.replace
_orig_os_rename = os.rename
_orig_os_remove = os.remove
_orig_os_unlink = getattr(os, "unlink", os.remove)
_orig_path_open = Path.open
_orig_path_write_text = Path.write_text
_orig_path_write_bytes = Path.write_bytes
_orig_path_unlink = Path.unlink


def _guarded_builtins_open(file, mode="r", *args, **kwargs):
    mode_str = str(mode)
    if any(c in mode_str for c in ("w", "a", "+", "x")):
        if _is_protected_target(file):
            raise AssertionError(
                f"TEST ISOLATION VIOLATION: Test code attempted write open on production file '{file}' with mode='{mode}'"
            )
    return _orig_builtins_open(file, mode, *args, **kwargs)


def _guarded_path_open(self, mode="r", *args, **kwargs):
    mode_str = str(mode)
    if any(c in mode_str for c in ("w", "a", "+", "x")):
        if _is_protected_target(self):
            raise AssertionError(
                f"TEST ISOLATION VIOLATION: Test code attempted Path.open write on production file '{self}' with mode='{mode}'"
            )
    return _orig_path_open(self, mode, *args, **kwargs)


def _guarded_path_write_text(self, data, *args, **kwargs):
    if _is_protected_target(self):
        raise AssertionError(
            f"TEST ISOLATION VIOLATION: Test code attempted Path.write_text on production file '{self}'"
        )
    return _orig_path_write_text(self, data, *args, **kwargs)


def _guarded_path_write_bytes(self, data, *args, **kwargs):
    if _is_protected_target(self):
        raise AssertionError(
            f"TEST ISOLATION VIOLATION: Test code attempted Path.write_bytes on production file '{self}'"
        )
    return _orig_path_write_bytes(self, data, *args, **kwargs)


def _guarded_path_unlink(self, *args, **kwargs):
    if _is_protected_target(self):
        raise AssertionError(
            f"TEST ISOLATION VIOLATION: Test code attempted Path.unlink on production file '{self}'"
        )
    try:
        return _orig_path_unlink(self, *args, **kwargs)
    except PermissionError:
        if "pytest" in str(self).lower():
            return
        raise


_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | getattr(os, "O_APPEND", 0) | getattr(os, "O_TRUNC", 0)


def _guarded_os_open(path, flags, *args, **kwargs):
    if (flags & _WRITE_FLAGS) != 0:
        if _is_protected_target(path):
            raise AssertionError(
                f"TEST ISOLATION VIOLATION: Test code attempted os.open with write flags on production file '{path}'"
            )
    return _orig_os_open(path, flags, *args, **kwargs)


def _guarded_os_replace(src, dst, *args, **kwargs):
    if _is_protected_target(dst):
        raise AssertionError(
            f"TEST ISOLATION VIOLATION: Test code attempted os.replace into production destination '{dst}'"
        )
    return _orig_os_replace(src, dst, *args, **kwargs)


def _guarded_os_rename(src, dst, *args, **kwargs):
    if _is_protected_target(dst):
        raise AssertionError(
            f"TEST ISOLATION VIOLATION: Test code attempted os.rename into production destination '{dst}'"
        )
    return _orig_os_rename(src, dst, *args, **kwargs)


def _guarded_os_remove(path, *args, **kwargs):
    if _is_protected_target(path):
        raise AssertionError(
            f"TEST ISOLATION VIOLATION: Test code attempted os.remove on production file '{path}'"
        )
    try:
        return _orig_os_remove(path, *args, **kwargs)
    except PermissionError:
        if "pytest" in str(path).lower():
            return
        raise


def _guarded_os_unlink(path, *args, **kwargs):
    if _is_protected_target(path):
        raise AssertionError(
            f"TEST ISOLATION VIOLATION: Test code attempted os.unlink on production file '{path}'"
        )
    try:
        return _orig_os_unlink(path, *args, **kwargs)
    except PermissionError:
        if "pytest" in str(path).lower():
            return
        raise


# Apply the monkey patches to guard production directories
builtins.open = _guarded_builtins_open
Path.open = _guarded_path_open
Path.write_text = _guarded_path_write_text
Path.write_bytes = _guarded_path_write_bytes
Path.unlink = _guarded_path_unlink
os.open = _guarded_os_open
os.replace = _guarded_os_replace
os.rename = _guarded_os_rename
os.remove = _guarded_os_remove
os.unlink = _guarded_os_unlink


@pytest.fixture(scope="session")
def test_isolation_root() -> Path:
    """Returns the isolated test root directory for the active test session."""
    return _SESSION_TEST_ROOT


@pytest.fixture(autouse=True)
def enforce_test_isolation_roots(monkeypatch):
    """Autouse fixture ensuring config paths remain pinned to isolated test root."""
    config.init_isolated_test_env(_SESSION_TEST_ROOT)
