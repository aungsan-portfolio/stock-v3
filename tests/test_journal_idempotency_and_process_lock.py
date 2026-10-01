"""
tests/test_journal_idempotency_and_process_lock.py -- Verification Suite for:
  T1: Process lock - first instance succeeds
  T2: Process lock - second instance rejected
  T3: Lock release & stale-owner recovery
  T4: Case A - exact duplicate replay (order qty=1, exec-A=1, exec-A=1 -> local qty=1)
  T5: Case B - legitimate partial fills (order qty=1, exec-A=0.4, exec-B=0.6 -> local qty=1.0)
  T6: Case C - distinct executions on larger order (order qty=2, exec-A=1, exec-B=1 -> local qty=2.0)
  T7: Sep-30 LLY replay (in-memory fixture reproducing duplicate incident -> clean reconciliation)
  T8: Missing execution_id contract (missing exec_id does not blindly dedupe)
  T9: Concurrency & write-time idempotency test
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import os
import socket
import threading
from decimal import Decimal

import pytest

from strategies.trade_journal import (
    OrderFillState,
    load_journal_order_fill_state,
    log_fill,
    read_journal,
    canonical_decimal,
)
from strategies.startup_security import (
    acquire_atomic_process_lock,
    release_atomic_process_lock,
    ProcessLockMetadata,
    StartupSafetyRejected,
)


# =============================================================================
# T1 - T3: Process Exclusivity & Lock Lifecycle
# =============================================================================

def test_t1_process_lock_first_instance_succeeds(tmp_path):
    """
    T1: First daytrade instance acquires the lock successfully.
    """
    lock_path = tmp_path / "daytrade_bot_t1.lock"
    meta = ProcessLockMetadata(
        pid=os.getpid(),
        hostname=socket.gethostname(),
        account="PA3UODIC5JTA",
        mode="daytrade",
        branch="feature/batch-fetch-engine",
        release_sha="2d07950",
        started_at_utc="2026-10-01T12:00:00Z",
    )
    fd = acquire_atomic_process_lock(lock_path, meta)
    try:
        assert fd >= 0
        assert lock_path.exists()
        with open(lock_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["pid"] == os.getpid()
        assert data["account"] == "PA3UODIC5JTA"
    finally:
        release_atomic_process_lock(lock_path, fd)
    assert not lock_path.exists()


def test_t2_process_lock_second_instance_rejected(tmp_path):
    """
    T2: Second daytrade instance is rejected with StartupSafetyRejected (STARTUP_LOCK_COLLISION)
    while first instance is actively running.
    """
    lock_path = tmp_path / "daytrade_bot_t2.lock"
    meta1 = ProcessLockMetadata(
        pid=os.getpid(),
        hostname=socket.gethostname(),
        account="PA3UODIC5JTA",
        mode="daytrade",
        branch="feature/batch-fetch-engine",
        release_sha="2d07950",
        started_at_utc="2026-10-01T12:00:00Z",
    )
    fd1 = acquire_atomic_process_lock(lock_path, meta1)
    try:
        assert fd1 >= 0
        meta2 = ProcessLockMetadata(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            account="PA3UODIC5JTA",
            mode="daytrade",
            branch="feature/batch-fetch-engine",
            release_sha="2d07950",
            started_at_utc="2026-10-01T12:00:01Z",
        )
        with pytest.raises(StartupSafetyRejected) as exc_info:
            acquire_atomic_process_lock(lock_path, meta2)
        assert "STARTUP_LOCK_COLLISION" in str(exc_info.value)
    finally:
        release_atomic_process_lock(lock_path, fd1)


def test_t3_lock_release_and_stale_owner_recovery(tmp_path):
    """
    T3:
    Part 1: Normal lock release cleanly unlinks lock file.
    Part 2: If previous instance exited abnormally (dead PID), a new legitimate
    instance cleans the stale lock and acquires successfully.
    """
    lock_path = tmp_path / "daytrade_bot_t3.lock"

    # Part 1: Normal acquisition and release
    meta = ProcessLockMetadata(
        pid=os.getpid(),
        hostname=socket.gethostname(),
        account="PA3UODIC5JTA",
        mode="daytrade",
        branch="feature/batch-fetch-engine",
        release_sha="2d07950",
        started_at_utc="2026-10-01T12:00:00Z",
    )
    fd = acquire_atomic_process_lock(lock_path, meta)
    assert lock_path.exists()
    release_atomic_process_lock(lock_path, fd)
    assert not lock_path.exists()

    # Part 2: Stale lock recovery (simulate dead PID 9999999)
    stale_meta = {
        "pid": 9999999,
        "hostname": socket.gethostname(),
        "account": "PA3UODIC5JTA",
        "mode": "daytrade",
        "branch": "feature/batch-fetch-engine",
        "release_sha": "2d07950",
        "started_at_utc": "2026-09-30T12:00:00Z",
    }
    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(stale_meta, f)
    assert lock_path.exists()

    # New legitimate instance should detect stale dead PID, clean it, and acquire
    fd_new = acquire_atomic_process_lock(lock_path, meta)
    try:
        assert fd_new >= 0
        with open(lock_path, "r", encoding="utf-8") as f:
            recovered_data = json.load(f)
        assert recovered_data["pid"] == os.getpid()
    finally:
        release_atomic_process_lock(lock_path, fd_new)
    assert not lock_path.exists()


# =============================================================================
# T4 - T6: Idempotent Fill Handling (Cases A, B, C)
# =============================================================================

def test_t4_case_a_exact_duplicate_replay():
    """
    CASE A: Exact replay
      Order qty = 1
      exec-A = 1
      exec-A = 1 (repeated)
    Expected:
      local qty = 1
      local notional = 1 fill only
    """
    state = OrderFillState(broker_order_id="order_case_a")

    # Slice 1: 1 share @ $100
    res1 = state.add_fill(qty=Decimal("1"), price=Decimal("100.0"), exec_id="exec-A")
    assert res1 is True
    assert state.logged_qty == Decimal("1")
    assert state.logged_notional == Decimal("100.0")

    # Duplicate slice 1 replayed
    res2 = state.add_fill(qty=Decimal("1"), price=Decimal("100.0"), exec_id="exec-A")
    assert res2 is False
    assert state.logged_qty == Decimal("1")
    assert state.logged_notional == Decimal("100.0")


def test_t5_case_b_legitimate_partial_fills():
    """
    CASE B: Legitimate partial fills
      Order qty = 1
      exec-A = 0.4
      exec-B = 0.6
    Expected:
      local qty = 1.0
      local notional accumulated
    """
    state = OrderFillState(broker_order_id="order_case_b")

    res1 = state.add_fill(qty=Decimal("0.4"), price=Decimal("100.0"), exec_id="exec-A")
    assert res1 is True
    assert state.logged_qty == Decimal("0.4")
    assert state.logged_notional == Decimal("40.0")

    res2 = state.add_fill(qty=Decimal("0.6"), price=Decimal("105.0"), exec_id="exec-B")
    assert res2 is True
    assert state.logged_qty == Decimal("1.0")
    assert state.logged_notional == Decimal("103.0")
    assert len(state.execution_ids) == 2


def test_t6_case_c_distinct_executions_larger_order():
    """
    CASE C: Distinct executions on a larger order
      Order qty = 2
      exec-A = 1
      exec-B = 1
    Expected:
      local qty = 2.0
    """
    state = OrderFillState(broker_order_id="order_case_c")

    res1 = state.add_fill(qty=Decimal("1"), price=Decimal("50.0"), exec_id="exec-A")
    assert res1 is True
    assert state.logged_qty == Decimal("1")

    res2 = state.add_fill(qty=Decimal("1"), price=Decimal("51.0"), exec_id="exec-B")
    assert res2 is True
    assert state.logged_qty == Decimal("2")
    assert state.logged_notional == Decimal("101.0")

    # Replay of exec-A should still be rejected
    res3 = state.add_fill(qty=Decimal("1"), price=Decimal("50.0"), exec_id="exec-A")
    assert res3 is False
    assert state.logged_qty == Decimal("2")


# =============================================================================
# T7 - T8: Sep-30 LLY Replay & Missing Execution ID Contract
# =============================================================================

def test_t7_sep30_lly_replay_reconciliation_clean():
    """
    T7: Exact Sep-30 LLY Replay Case (reproducing the exact incident in-memory)
    Input:
      broker_order_id = f31f5072-7ec3-4536-b53c-e811cac5f46e
      qty = 1
      execution_id = SYNTH:ALPACA:f31f5072-7ec3-4536-b53c-e811cac5f46e:0-1
      execution_id = SYNTH:ALPACA:f31f5072-7ec3-4536-b53c-e811cac5f46e:0-1 (repeated)
    Expected:
      local qty = 1 (NOT 2)
      broker qty = 1
      delta_qty = 0 -> clean reconciliation
    """
    record_1 = {
        "timestamp": "2026-09-30T16:12:53.679748+00:00",
        "journal_write_timestamp_utc": "2026-09-30T16:13:23.057477+00:00",
        "type": "FILL",
        "event_type": "FILL",
        "fill_id": "SYNTH:ALPACA:f31f5072-7ec3-4536-b53c-e811cac5f46e:0-1",
        "execution_id": "SYNTH:ALPACA:f31f5072-7ec3-4536-b53c-e811cac5f46e:0-1",
        "order_id": "f31f5072-7ec3-4536-b53c-e811cac5f46e",
        "broker_order_id": "f31f5072-7ec3-4536-b53c-e811cac5f46e",
        "symbol": "LLY",
        "side": "BUY",
        "qty": 1,
        "fill_price": 1186.67,
        "account_id": "PA3UODIC5JTA",
    }
    record_2 = dict(record_1)
    record_2["journal_write_timestamp_utc"] = "2026-09-30T16:13:23.300826+00:00"

    journal_records = [record_1, record_2]

    # Load state from memory fixture (never touching raw files on disk)
    states = load_journal_order_fill_state(
        journal_records,
        account_id="PA3UODIC5JTA",
        trading_date="2026-09-30",
    )

    key = ("PA3UODIC5JTA", "2026-09-30", "f31f5072-7ec3-4536-b53c-e811cac5f46e", "LLY", "BUY")
    assert key in states
    fill_state = states[key]

    # Invariant: Must be 1, NOT 2
    assert fill_state.logged_qty == Decimal("1")
    assert fill_state.logged_notional == Decimal("1186.67")

    # Reconciliation simulation against Alpaca broker truth
    broker_cum_qty = Decimal("1")
    delta_qty = broker_cum_qty - fill_state.logged_qty
    # Invariant: delta_qty must be 0 (no negative anomaly, no duplicate error)
    assert delta_qty == Decimal("0")


def test_t8_missing_execution_id_contract():
    """
    T8: Contract for execution ID handling:
      - same scoped execution_id -> count once
      - different execution_id -> count independently
      - missing execution_id -> do NOT blindly dedupe (provenance rule)
    """
    state = OrderFillState(broker_order_id="order_contract_test")

    # 1. Missing execution_id (None or empty) -> accumulates independently
    r1 = state.add_fill(qty=Decimal("1"), price=Decimal("10.0"), exec_id=None)
    assert r1 is True
    assert state.logged_qty == Decimal("1")

    r2 = state.add_fill(qty=Decimal("1"), price=Decimal("10.0"), exec_id="")
    assert r2 is True
    assert state.logged_qty == Decimal("2")

    # 2. Distinct execution IDs -> accumulate independently
    r3 = state.add_fill(qty=Decimal("1"), price=Decimal("10.0"), exec_id="EXEC_001")
    assert r3 is True
    assert state.logged_qty == Decimal("3")

    r4 = state.add_fill(qty=Decimal("1"), price=Decimal("10.0"), exec_id="EXEC_002")
    assert r4 is True
    assert state.logged_qty == Decimal("4")

    # 3. Same execution ID -> deduplicated (count once)
    r5 = state.add_fill(qty=Decimal("1"), price=Decimal("10.0"), exec_id="EXEC_001")
    assert r5 is False
    assert state.logged_qty == Decimal("4")


# =============================================================================
# T9: Concurrency & Write-Time Deduplication
# =============================================================================

def test_t9_concurrency_and_write_time_deduplication(tmp_path):
    """
    T9: Calling log_fill across multiple threads or concurrent operations with
    the same execution_id appends exactly once.
    """
    journal_file = tmp_path / "concurrent_trade_journal.jsonl"
    exec_id = "SYNTH:ALPACA:concurrent_order_1:0-1"

    def do_write():
        log_fill(
            symbol="LLY",
            side="BUY",
            qty=1,
            fill_price=1186.67,
            expected_price=1185.00,
            slippage=0.001,
            fill_latency_ms=250.0,
            journal_file=journal_file,
            order_id="concurrent_order_1",
            broker_order_id="concurrent_order_1",
            execution_id=exec_id,
            account_id="PA3UODIC5JTA",
        )

    threads = [threading.Thread(target=do_write) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = read_journal(journal_file)
    # Invariant: Even with 5 concurrent threads, only 1 record written
    assert len(records) == 1
    assert records[0]["fill_id"] == exec_id
