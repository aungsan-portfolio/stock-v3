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
import io
import json
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
from strategies.trailing_stop import DynamicTrailingStopManager, _classify_held_order
from strategies.gate_ledger import log_gate_decision
from strategies.trade_journal import log_trade
from strategies.validation_lane import ValidationTradeRecord, log_validation_record
import risk_state
from tools.compare_manifests import compare_manifests


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
assert Path(config.SHADOW_BASE_DIR).resolve().is_relative_to(test_root), f"SHADOW_BASE_DIR {config.SHADOW_BASE_DIR} not in {test_root}"
assert Path(config.BROKER_EVENTS_PATH).resolve().is_relative_to(test_root), f"BROKER_EVENTS_PATH {config.BROKER_EVENTS_PATH} not in {test_root}"

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
    clean_env.pop("BROKER_EVENTS_PATH", None)
    clean_env.pop("SHADOW_BASE_DIR", None)

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


def test_guard_blocks_rename_source(test_isolation_root):
    """Verify os.rename fails closed when the source is inside a protected production directory."""
    repo_root = Path(config.BASE_DIR).resolve()
    fake_prod_source = repo_root / "logs" / "audit_rename_source.tmp"
    iso_dest = test_isolation_root / "logs" / "moved.tmp"
    with capture_expected_violation():
        os.rename(fake_prod_source, iso_dest)


def test_guard_blocks_replace_source(test_isolation_root):
    """Verify os.replace fails closed when the source is inside a protected production directory."""
    repo_root = Path(config.BASE_DIR).resolve()
    fake_prod_source = repo_root / "logs" / "audit_replace_source.tmp"
    iso_dest = test_isolation_root / "logs" / "moved.tmp"
    with capture_expected_violation():
        os.replace(fake_prod_source, iso_dest)


def test_guard_blocks_path_rename_and_replace_source(test_isolation_root):
    """Verify Path.rename and Path.replace fail closed when source is a protected production path."""
    repo_root = Path(config.BASE_DIR).resolve()
    fake_prod_source = repo_root / "logs" / "path_rename_source.tmp"
    iso_dest = test_isolation_root / "logs" / "moved.tmp"
    with capture_expected_violation():
        fake_prod_source.rename(iso_dest)
    with capture_expected_violation():
        fake_prod_source.replace(iso_dest)


def test_guard_blocks_io_open():
    """Verify io.open fails closed on protected production paths."""
    repo_root = Path(config.BASE_DIR).resolve()
    fake_prod_file = repo_root / "logs" / "io_open_probe.tmp"
    with capture_expected_violation():
        with io.open(fake_prod_file, "w") as f:
            f.write("probe")


def test_guard_blocks_bytes_path():
    """Verify builtins.open fails closed when given a bytes production path."""
    repo_root = Path(config.BASE_DIR).resolve()
    fake_prod_bytes = os.fsencode(str(repo_root / "logs" / "bytes_probe.tmp"))
    with capture_expected_violation():
        with open(fake_prod_bytes, "w") as f:
            f.write("probe")


def test_guard_rejects_overlapping_production_roots(tmp_path):
    """Verify validate_test_isolation_root rejects production trees in both directions (ancestor, descendant, equal)."""
    repo_root = Path(config.BASE_DIR).resolve()

    # Equal to BASE_DIR or ancestor
    with pytest.raises(ValueError, match="INVALID_TEST_ROOT"):
        config.validate_test_isolation_root(repo_root)
    with pytest.raises(ValueError, match="INVALID_TEST_ROOT"):
        config.validate_test_isolation_root(repo_root.parent)

    # Protected subdirectories and descendants
    for sub in ["data", "logs", "reports"]:
        prod_sub = repo_root / sub
        with pytest.raises(ValueError, match="INVALID_TEST_ROOT"):
            config.validate_test_isolation_root(prod_sub)
        with pytest.raises(ValueError, match="INVALID_TEST_ROOT"):
            config.validate_test_isolation_root(prod_sub / "nested_test_dir")
        # Bytes format
        with pytest.raises(ValueError, match="INVALID_TEST_ROOT"):
            config.validate_test_isolation_root(os.fsencode(str(prod_sub / "bytes_test")))

    # Valid disposable root passes cleanly
    valid_root = tmp_path / "disposable_root"
    validated = config.validate_test_isolation_root(valid_root)
    assert validated == valid_root.resolve()


def test_classify_held_order_regression():
    """Verify _classify_held_order returns 'unrelated' for unmatched orders, None, and arbitrary orderRefs."""
    class DummyOrder:
        def __init__(self, order_type="", client_order_id=""):
            self.order_type = order_type
            self.client_order_id = client_order_id

    # None and empty
    assert _classify_held_order(None) == "unrelated"
    assert _classify_held_order(DummyOrder()) == "unrelated"

    # Stop orders
    assert _classify_held_order(DummyOrder(order_type="stop")) == "stop"
    assert _classify_held_order(DummyOrder(order_type="stop_limit")) == "stop"
    assert _classify_held_order(DummyOrder(order_type="trailing_stop")) == "stop"

    # In-flight exit orders
    assert _classify_held_order(DummyOrder(order_type="market", client_order_id="dt_exit_001")) == "exit"
    assert _classify_held_order(DummyOrder(order_type="limit", client_order_id="val_exit_002")) == "exit"
    assert _classify_held_order(DummyOrder(order_type="market", client_order_id="parent:child_exit")) == "exit"

    # Unmatched / unrelated orders MUST return "unrelated" (not None)
    assert _classify_held_order(DummyOrder(order_type="limit", client_order_id="arbitrary_unrelated_order")) == "unrelated"
    assert _classify_held_order(DummyOrder(order_type="market", client_order_id="swing_unrelated_123")) == "unrelated"


def test_multiprocess_workers_distinct_isolated_roots(tmp_path):
    """Verify distinct worker subprocesses operate in isolated roots without collision or contention."""
    worker_0_root = tmp_path / "worker_0"
    worker_1_root = tmp_path / "worker_1"

    worker_code = """
import os, sys
from pathlib import Path
import config
from strategies.order_idempotency import EntryIdempotencyStore
import order_audit

worker_id = sys.argv[1]
root_dir = Path(os.environ["STOCK_ENGINE_TEST_ROOT"]).resolve()
assert Path(config.DATA_DIR).resolve().is_relative_to(root_dir)

store = EntryIdempotencyStore()
reserved, reason = store.reserve(f"SYM_W{worker_id}", side="BUY")
assert reserved, f"Worker {worker_id} reservation failed: {reason}"
order_audit.log_event("worker_event", symbol=f"SYM_W{worker_id}", order_id=f"cid_w{worker_id}")

assert (root_dir / "data" / "order_idempotency_state.json").exists()
assert (root_dir / "logs" / "order_audit.jsonl").exists()
sys.exit(0)
"""
    env0 = os.environ.copy()
    env0["STOCK_ENGINE_TEST_ROOT"] = str(worker_0_root)
    env1 = os.environ.copy()
    env1["STOCK_ENGINE_TEST_ROOT"] = str(worker_1_root)

    p0 = subprocess.Popen([sys.executable, "-c", worker_code, "0"], env=env0, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p1 = subprocess.Popen([sys.executable, "-c", worker_code, "1"], env=env1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    out0, err0 = p0.communicate(timeout=15)
    out1, err1 = p1.communicate(timeout=15)

    assert p0.returncode == 0, f"Worker 0 failed: {err0}\n{out0}"
    assert p1.returncode == 0, f"Worker 1 failed: {err1}\n{out1}"

    # Verify worker 0 artifacts
    w0_idemp = (worker_0_root / "data" / "order_idempotency_state.json").read_text(encoding="utf-8")
    assert "SYM_W0" in w0_idemp
    assert "SYM_W1" not in w0_idemp

    # Verify worker 1 artifacts
    w1_idemp = (worker_1_root / "data" / "order_idempotency_state.json").read_text(encoding="utf-8")
    assert "SYM_W1" in w1_idemp
    assert "SYM_W0" not in w1_idemp


def test_subprocess_unexpected_violation_propagates_and_fails(tmp_path):
    """Verify unhandled guard violation in a child process crashes child with non-zero exit code and propagates error."""
    child_root = tmp_path / "child_fail_test"
    child_code = """
import os, sys
from pathlib import Path
import config

# Unhandled attempt to write to production logs must crash child process
prod_target = Path(config.BASE_DIR) / "logs" / "unhandled_child_leak.log"
with open(prod_target, "w") as f:
    f.write("unhandled leak")
sys.exit(0)
"""
    child_env = os.environ.copy()
    child_env["STOCK_ENGINE_TEST_ROOT"] = str(child_root)

    result = subprocess.run(
        [sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
        env=child_env,
    )
    # The child process must FAIL (non-zero return code)
    assert result.returncode != 0, "Child process unexpectedly succeeded despite illegal write!"
    assert "TestIsolationViolationError" in result.stderr, f"Violation error not in stderr: {result.stderr}"


def test_compare_manifests_rejects_error_tokens_and_malformed_hashes(tmp_path):
    """Verify compare_manifests fails closed and records integrity errors for ERROR strings and invalid digests."""
    # Case 1: Matching ERROR strings
    err_manifest = {"files": {"logs/unreadable.log": {"size_bytes": 10, "sha256": "ERROR: Permission denied"}}}
    m1 = tmp_path / "m1_before.json"
    m2 = tmp_path / "m1_after.json"
    m1.write_text(json.dumps(err_manifest), encoding="utf-8")
    m2.write_text(json.dumps(err_manifest), encoding="utf-8")
    assert compare_manifests(m1, m2) is False

    # Case 2: Malformed digest (not 64 hex characters)
    bad_hash_manifest = {"files": {"logs/bad.log": {"size_bytes": 10, "sha256": "invalid_sha256_too_short"}}}
    m3 = tmp_path / "m2_before.json"
    m4 = tmp_path / "m2_after.json"
    m3.write_text(json.dumps(bad_hash_manifest), encoding="utf-8")
    m4.write_text(json.dumps(bad_hash_manifest), encoding="utf-8")
    assert compare_manifests(m3, m4) is False

    # Case 3: Invalid negative size
    neg_size_manifest = {"files": {"logs/neg.log": {"size_bytes": -5, "sha256": "a" * 64}}}
    m5 = tmp_path / "m3_before.json"
    m6 = tmp_path / "m3_after.json"
    m5.write_text(json.dumps(neg_size_manifest), encoding="utf-8")
    m6.write_text(json.dumps(neg_size_manifest), encoding="utf-8")
    assert compare_manifests(m5, m6) is False


def test_regime_breadth_bounded_batch_path_and_timeout_invariance():
    """Verify compute_vwap_breadth routes through bounded fetch_intraday_batch with 10.0s timeout and no serial leaks."""
    from unittest.mock import patch, MagicMock
    import strategies.regime_engine as re

    old_batch = os.environ.get("REGIME_BREADTH_BATCH_ENABLED")
    os.environ["REGIME_BREADTH_BATCH_ENABLED"] = "true"
    try:
        with patch("strategies.intraday_data.fetch_intraday_batch") as mock_batch, \
             patch("strategies.intraday_data.fetch_intraday") as mock_serial:

            mock_batch.return_value = ({}, {"coverage": 0.0})
            symbols = ["AAPL", "MSFT", "NVDA"]
            mock_bridge = MagicMock()
            mock_bridge.is_connected = True

            breadth_pct, mode, details = re.compute_vwap_breadth(symbols=symbols, bridge=mock_bridge)

            assert mock_batch.called, "Batch fetch must be called when REGIME_BREADTH_BATCH_ENABLED is active"
            _, kwargs = mock_batch.call_args
            assert kwargs.get("total_timeout_seconds") == 10.0, "Timeout budget must strictly be 10.0s"
            assert kwargs.get("purpose") == "REGIME_BREADTH", "Purpose must strictly be REGIME_BREADTH"
            assert not mock_serial.called, "Serial fetch_intraday must NOT be called when use_batch is active (zero leak invariant)"
    finally:
        if old_batch is None:
            os.environ.pop("REGIME_BREADTH_BATCH_ENABLED", None)
        else:
            os.environ["REGIME_BREADTH_BATCH_ENABLED"] = old_batch


