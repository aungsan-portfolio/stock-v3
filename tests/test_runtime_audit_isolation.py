"""
tests/test_runtime_audit_isolation.py — Verification of Step 1 Test/Runtime Audit Isolation.

Verifies:
1. Dynamic config path resolution redirects DATA_DIR, LOG_DIR, and REPORTS_DIR to the isolated test root.
2. order_audit.log_event appends exclusively to isolated logs/order_audit.jsonl.
3. EntryIdempotencyStore defaults to isolated data/order_idempotency_state.json.
4. DynamicTrailingStopManager defaults to isolated data/trailing_state.json.
5. log_gate_decision appends exclusively to isolated data/trade_gate_decision.jsonl.
6. log_trade appends exclusively to isolated reports/trade_journal.jsonl.
7. log_validation_record appends exclusively to isolated data/paper_validation_trades.jsonl.
8. risk_state.record_trade writes exclusively to isolated logs/daily_risk_state.json.
9. Guard tripwire raises TestIsolationViolationError (BaseException) when write operations target production directories.
10. Guard tripwire CANNOT be swallowed by generic `except Exception:`.
11. Subprocesses automatically inherit STOCK_ENGINE_TEST_ROOT and the fail-closed write guard.
12. Clean subprocess without STOCK_ENGINE_TEST_ROOT preserves 100% of runtime default production paths.
13. Concurrent thread operations write safely to isolated stores without lock/state contention.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest

import config
from config import TestIsolationViolationError, SESSION_VIOLATIONS
import order_audit
from strategies.order_idempotency import EntryIdempotencyStore, OrderReservation
from strategies.trailing_stop import DynamicTrailingStopManager
from strategies.gate_ledger import log_gate_decision
from strategies.trade_journal import log_trade
from strategies.validation_lane import ValidationTradeRecord, log_validation_record
import risk_state


@contextmanager
def capture_expected_violation():
    """Captures and clears an expected TestIsolationViolationError during guard testing."""
    initial = len(config.SESSION_VIOLATIONS)
    caught = False
    try:
        yield
    except TestIsolationViolationError:
        caught = True
    if not caught:
        raise AssertionError("Expected TestIsolationViolationError was not raised!")
    del config.SESSION_VIOLATIONS[initial:]


def test_isolated_config_paths(test_isolation_root):
    """Verify that all core mutable directories point inside the isolated test root."""
    assert test_isolation_root == Path(config.DATA_DIR).parent
    assert Path(config.DATA_DIR).is_relative_to(test_isolation_root)
    assert Path(config.LOG_DIR).is_relative_to(test_isolation_root)
    assert Path(config.REPORTS_DIR).is_relative_to(test_isolation_root)

    repo_root = Path(__file__).resolve().parent.parent
    assert Path(config.DATA_DIR) != repo_root / "data"
    assert Path(config.LOG_DIR) != repo_root / "logs"
    assert Path(config.REPORTS_DIR) != repo_root / "reports"


def test_order_audit_isolation(test_isolation_root):
    """Verify order_audit writes exclusively to the test root."""
    order_audit.log_event(
        "test_isolation_stage",
        symbol="ISOLATION_TEST",
        side="BUY",
        order_id="test_ord_123",
    )
    isolated_audit = test_isolation_root / "logs" / "order_audit.jsonl"
    assert isolated_audit.exists()
    content = isolated_audit.read_text(encoding="utf-8")
    assert "ISOLATION_TEST" in content


def test_order_idempotency_isolation(test_isolation_root):
    """Verify EntryIdempotencyStore defaults to the test root."""
    store = EntryIdempotencyStore()
    assert Path(store.storage_path).is_relative_to(test_isolation_root)
    assert store.storage_path == test_isolation_root / "data" / "order_idempotency_state.json"


def test_trailing_stop_isolation(test_isolation_root):
    """Verify DynamicTrailingStopManager defaults to the test root."""
    mgr = DynamicTrailingStopManager()
    assert Path(mgr.STATE_FILE).is_relative_to(test_isolation_root)
    assert Path(mgr.STATE_FILE) == test_isolation_root / "data" / "trailing_state.json"


def test_gate_ledger_isolation(test_isolation_root):
    """Verify log_gate_decision writes exclusively to the test root."""
    log_gate_decision(
        evaluation_id="eval_iso_test_001",
        symbol="ISO_TEST",
        bar="5m",
        stage="STAGE_CONFIDENCE",
        decision="PASS",
        human_message="Test gate isolation",
    )
    isolated_gate = test_isolation_root / "data" / "trade_gate_decision.jsonl"
    assert isolated_gate.exists()
    content = isolated_gate.read_text(encoding="utf-8")
    assert "eval_iso_test_001" in content


def test_trade_journal_isolation(test_isolation_root):
    """Verify log_trade writes exclusively to the test root."""
    log_trade(
        symbol="ISO_JRN",
        side="BUY",
        strategy="ORB",
        qty=1,
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        reason="test_journal_isolation",
    )
    isolated_journal = test_isolation_root / "reports" / "trade_journal.jsonl"
    assert isolated_journal.exists()
    content = isolated_journal.read_text(encoding="utf-8")
    assert "ISO_JRN" in content


def test_validation_lane_isolation(test_isolation_root):
    """Verify log_validation_record writes exclusively to the test root."""
    rec = ValidationTradeRecord(
        validation_id="val_iso_001",
        timestamp="2026-10-16T00:00:00Z",
        symbol="ISO_VAL",
        strategy="MOMENTUM_SCALP",
        side="BUY",
        confidence=0.65,
        regime="BULLISH",
        market_texture="TRENDING",
        terminal_gate="STAGE_CONFIDENCE",
        rejection_reason="CONFIDENCE_BELOW_FLOOR",
        decision_price=50.0,
        decision_feed="alpaca_sip",
        bar_timestamp="2026-10-16T00:00:00Z",
        quote_mid=50.0,
        quote_feed="alpaca_sip",
        quote_timestamp=1700000000.0,
        stop_price=48.0,
        target_price=54.0,
        hard_safety_pass=True,
        policy_rejected=True,
    )
    log_validation_record(rec)
    isolated_val = test_isolation_root / "data" / "paper_validation_trades.jsonl"
    assert isolated_val.exists()
    content = isolated_val.read_text(encoding="utf-8")
    assert "val_iso_001" in content


def test_risk_state_isolation(test_isolation_root):
    """Verify risk_state writes exclusively to the test root."""
    count = risk_state.record_trade("2099-01-01")
    assert count >= 1
    isolated_risk = test_isolation_root / "logs" / "daily_risk_state.json"
    assert isolated_risk.exists()
    content = isolated_risk.read_text(encoding="utf-8")
    assert "2099-01-01" in content


def test_write_guard_fails_closed_base_exception():
    """Verify write-protection tripwire is a BaseException and catches all write methods."""
    assert issubclass(TestIsolationViolationError, BaseException)
    assert not issubclass(TestIsolationViolationError, Exception)

    repo_root = Path(__file__).resolve().parent.parent
    prod_test_file = repo_root / "logs" / "illegal_test_write.tmp"
    prod_test_dir = repo_root / "logs" / "illegal_dir"

    with capture_expected_violation():
        with open(prod_test_file, "w") as f:
            f.write("should_fail")

    with capture_expected_violation():
        prod_test_file.write_text("should_fail")

    with capture_expected_violation():
        prod_test_file.unlink()

    with capture_expected_violation():
        os.open(str(prod_test_file), os.O_WRONLY | os.O_CREAT)

    with capture_expected_violation():
        os.remove(str(prod_test_file))

    with capture_expected_violation():
        os.mkdir(str(prod_test_dir))


def test_write_guard_cannot_be_swallowed_by_generic_except_exception():
    """Verify generic `except Exception:` CANNOT swallow TestIsolationViolationError."""
    repo_root = Path(__file__).resolve().parent.parent
    prod_test_file = repo_root / "logs" / "swallow_test.tmp"
    swallowed = False

    with capture_expected_violation():
        try:
            with open(prod_test_file, "w") as f:
                f.write("should_fail")
        except Exception:
            swallowed = True

    assert not swallowed, "Write guard error was swallowed by `except Exception:`!"


def test_subprocess_inherits_isolation_and_write_guard(test_isolation_root):
    """Verify child processes inherit STOCK_ENGINE_TEST_ROOT and the write protection guard."""
    code = """
import os, sys
from pathlib import Path
import config

test_root = Path(os.environ["STOCK_ENGINE_TEST_ROOT"]).resolve()
assert Path(config.DATA_DIR).is_relative_to(test_root), f"DATA_DIR {config.DATA_DIR} not in {test_root}"
assert Path(config.LOG_DIR).is_relative_to(test_root), f"LOG_DIR {config.LOG_DIR} not in {test_root}"
assert Path(config.REPORTS_DIR).is_relative_to(test_root), f"REPORTS_DIR {config.REPORTS_DIR} not in {test_root}"

# Attempting to write to production logs must raise TestIsolationViolationError
prod_target = Path(config.BASE_DIR) / "logs" / "subprocess_leak.tmp"
caught = False
try:
    with open(prod_target, "w") as f:
        f.write("leak")
except config.TestIsolationViolationError:
    caught = True

assert caught, "Subprocess did not enforce TestIsolationViolationError!"
sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, f"Child process failed: {result.stderr}\n{result.stdout}"


def test_runtime_default_paths_invariance_clean_subprocess():
    """Verify that in a clean environment without STOCK_ENGINE_TEST_ROOT, all runtime paths are unchanged."""
    clean_env = os.environ.copy()
    clean_env.pop("STOCK_ENGINE_TEST_ROOT", None)
    clean_env.pop("TEST_RUNNING", None)
    clean_env.pop("PYTEST_CURRENT_TEST", None)

    code = """
import logging
logging.disable(logging.CRITICAL)
import sys
from pathlib import Path
import config
import order_audit
import strategies.order_idempotency as idempotency
import strategies.trailing_stop as trailing
import strategies.gate_ledger as gate
import strategies.validation_lane as val
import risk_state
import alpaca_bridge
import main

base = Path(config.BASE_DIR).resolve()
assert Path(config.DATA_DIR).resolve() == base / "data", f"DATA_DIR mismatch: {config.DATA_DIR}"
assert Path(config.LOG_DIR).resolve() == base / "logs", f"LOG_DIR mismatch: {config.LOG_DIR}"
assert Path(config.REPORTS_DIR).resolve() == base / "reports", f"REPORTS_DIR mismatch: {config.REPORTS_DIR}"
assert Path(config.HALT_LOCK_FILE).resolve() == base / "data" / "HALT.lock"
assert Path(config.FLATTEN_LOCK_FILE).resolve() == base / "data" / "FLATTEN.lock"

# Module default stores
assert Path(order_audit.get_audit_file()).resolve() == base / "logs" / "order_audit.jsonl"
store = idempotency.EntryIdempotencyStore()
assert Path(store.storage_path).resolve() == base / "data" / "order_idempotency_state.json"
mgr = trailing.DynamicTrailingStopManager()
assert Path(mgr.STATE_FILE).resolve() == base / "data" / "trailing_state.json"
assert Path(gate.get_gate_ledger_path()).resolve() == base / "data" / "trade_gate_decision.jsonl"
assert Path(val.get_validation_ledger_path()).resolve() == base / "data" / "paper_validation_trades.jsonl"
assert Path(risk_state._get_state_path()).resolve() == base / "logs" / "daily_risk_state.json"
assert Path(config.BROKER_EVENTS_PATH).resolve() == base / "logs" / "broker_events.jsonl"
assert Path(alpaca_bridge.AlpacaBridge.get_broker_events_path(None)).resolve() == base / "logs" / "broker_events.jsonl"

# Write guard must NOT be active in standard production runtime
assert config._GUARD_INSTALLED is False, "Production write guard must be inactive in clean runtime"
sys.exit(0)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert result.returncode == 0, f"Clean runtime regression detected: {result.stderr}\n{result.stdout}"


def test_concurrent_workers_isolation(test_isolation_root):
    """Verify concurrent thread execution writes safely to isolated paths without contention."""
    errors = []
    store = EntryIdempotencyStore()

    def worker_action(thread_id: int):
        try:
            reserved, reason = store.reserve(f"TH_{thread_id}", side="BUY")
            assert reserved, f"Failed to reserve slot: {reason}"
            order_audit.log_event("thread_audit", symbol=f"TH_{thread_id}", order_id=f"cid_th_{thread_id}")
            log_gate_decision(
                evaluation_id=f"eval_th_{thread_id}",
                symbol=f"TH_{thread_id}",
                bar="5m",
                stage="STAGE_CONFIDENCE",
                decision="PASS",
                human_message="concurrent test",
            )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker_action, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent workers experienced errors: {errors}"
    assert (test_isolation_root / "data" / "order_idempotency_state.json").exists()
    assert (test_isolation_root / "logs" / "order_audit.jsonl").exists()
    assert (test_isolation_root / "data" / "trade_gate_decision.jsonl").exists()
