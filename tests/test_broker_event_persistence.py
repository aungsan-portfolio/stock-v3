"""
tests/test_broker_event_persistence.py -- Concurrency and persistence tests for broker trade update events.

Covers:
1. Default path persistence
2. Configured custom path persistence
3. Missing parent directory creation
4. Existing file append behavior
5. Two sequential updates produce two complete lines
6. Concurrent updates do not interleave JSON records
7. Non-serializable field is handled without callback failure
8. Unwritable path produces observable error log without callback failure
9. Persisted data does not expose credentials or authentication headers
10. Duplicate broker event policy is explicitly verified
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import threading
from unittest.mock import MagicMock
import pytest

import config as config_module
from alpaca_bridge import AlpacaBridge


class MockOrder:
    def __init__(self, order_id: str, symbol: str, status: str, filled_qty: float, price: float):
        self.id = order_id
        self.symbol = symbol
        self.status = status
        self.filled_qty = filled_qty
        self.filled_avg_price = price


class MockUpdate:
    def __init__(
        self,
        event: str = "fill",
        order_id: str = "ord_1001",
        execution_id: str = "exec_1001",
        order_status: str = "filled",
        filled_qty: float = 10.0,
        price: float = 150.25,
        symbol: str = "AAPL",
    ):
        self.event = event
        self.order_id = order_id
        self.execution_id = execution_id
        self.order_status = order_status
        self.position_qty = filled_qty
        self.price = price
        self.order = MockOrder(order_id, symbol, order_status, filled_qty, price)


def _create_mock_bridge() -> AlpacaBridge:
    bridge = AlpacaBridge.__new__(AlpacaBridge)
    bridge._trade_stream_lock = threading.RLock()
    bridge._seen_trade_update_keys = set()
    bridge._seen_trade_update_order = deque(maxlen=1000)
    bridge.on_entry_order_update = MagicMock()
    return bridge


def test_default_path_persistence(tmp_path, monkeypatch):
    """Requirement 1: Default path persistence fallback when BROKER_EVENTS_PATH is None."""
    default_target = tmp_path / "logs" / "broker_events.jsonl"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", None, raising=False)

    bridge = _create_mock_bridge()
    u = MockUpdate(order_id="ord_default", execution_id="ex_default", symbol="AAPL", filled_qty=10.0, price=180.00)
    bridge._dispatch_trade_update(u)

    assert default_target.exists()
    lines = default_target.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["order_id"] == "ord_default"
    assert record["symbol"] == "AAPL"


def test_configured_custom_path_persistence(tmp_path, monkeypatch):
    custom_path = tmp_path / "custom_broker_events.jsonl"
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(custom_path), raising=False)

    bridge = _create_mock_bridge()
    u = MockUpdate(order_id="ord_1", execution_id="ex_1", symbol="NVDA", filled_qty=5.0, price=120.50)
    bridge._dispatch_trade_update(u)

    assert custom_path.exists()
    lines = custom_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["order_id"] == "ord_1"
    assert data["symbol"] == "NVDA"
    assert data["execution_id"] == "ex_1"
    assert data["filled_qty"] == 5.0
    assert data["price"] == 120.50
    assert "ts" in data


def test_missing_parent_directory_creation(tmp_path, monkeypatch):
    nested_path = tmp_path / "nested" / "audit_dir" / "events.jsonl"
    assert not nested_path.parent.exists()
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(nested_path), raising=False)

    bridge = _create_mock_bridge()
    bridge._dispatch_trade_update(MockUpdate(order_id="ord_nested"))

    assert nested_path.exists()
    assert len(nested_path.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_existing_file_append_behavior(tmp_path, monkeypatch):
    target_path = tmp_path / "events.jsonl"
    target_path.write_text('{"initial": "seed"}\n', encoding="utf-8")
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(target_path), raising=False)

    bridge = _create_mock_bridge()
    bridge._dispatch_trade_update(MockUpdate(order_id="ord_append"))

    lines = target_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"initial": "seed"}
    assert json.loads(lines[1])["order_id"] == "ord_append"


def test_two_sequential_updates_produce_two_lines(tmp_path, monkeypatch):
    target_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(target_path), raising=False)

    bridge = _create_mock_bridge()
    bridge._dispatch_trade_update(MockUpdate(order_id="ord_seq_1", execution_id="ex_seq_1"))
    bridge._dispatch_trade_update(MockUpdate(order_id="ord_seq_2", execution_id="ex_seq_2"))

    lines = target_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["order_id"] == "ord_seq_1"
    assert json.loads(lines[1])["order_id"] == "ord_seq_2"


def test_concurrent_updates_do_not_interleave(tmp_path, monkeypatch):
    target_path = tmp_path / "concurrent_events.jsonl"
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(target_path), raising=False)

    bridge = _create_mock_bridge()
    num_updates = 30

    def dispatch_worker(i):
        u = MockUpdate(order_id=f"ord_{i:03d}", execution_id=f"ex_{i:03d}", price=100.0 + i)
        bridge._dispatch_trade_update(u)

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(dispatch_worker, range(num_updates)))

    assert target_path.exists()
    lines = target_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == num_updates

    # Verify every line is valid JSON without any character interleaving
    order_ids = set()
    for line in lines:
        record = json.loads(line)
        order_ids.add(record["order_id"])

    assert len(order_ids) == num_updates


def test_unserializable_field_does_not_crash_callback(tmp_path, monkeypatch, caplog):
    target_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(target_path), raising=False)

    bridge = _create_mock_bridge()
    u = MockUpdate(order_id="ord_valid")

    with monkeypatch.context() as m:
        m.setattr(json, "dumps", MagicMock(side_effect=TypeError("Non-serializable field")))
        with caplog.at_level(logging.ERROR):
            # Must not raise an exception
            bridge._dispatch_trade_update(u)

    assert "Failed to persist broker trade-update event" in caplog.text


def test_unwritable_path_produces_observable_error_log(tmp_path, monkeypatch, caplog):
    # Pass a path pointing to a directory to induce write failure
    dir_path = tmp_path / "is_a_directory"
    dir_path.mkdir()
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(dir_path), raising=False)

    bridge = _create_mock_bridge()
    u = MockUpdate(order_id="ord_fail")

    with caplog.at_level(logging.ERROR):
        # Must handle cleanly and log error
        bridge._dispatch_trade_update(u)

    assert "Failed to persist broker trade-update event" in caplog.text


def test_persisted_data_does_not_expose_credentials(tmp_path, monkeypatch):
    target_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(target_path), raising=False)

    bridge = _create_mock_bridge()
    bridge._dispatch_trade_update(MockUpdate(order_id="ord_clean"))

    line = target_path.read_text(encoding="utf-8").strip()
    record = json.loads(line)

    allowed_keys = {"ts", "event", "status", "order_id", "symbol", "filled_qty", "price", "execution_id"}
    assert set(record.keys()).issubset(allowed_keys)
    # Check no sensitive keys present
    forbidden_tokens = ["key", "secret", "token", "auth", "password"]
    for k in record.keys():
        for token in forbidden_tokens:
            assert token not in k.lower()


def test_duplicate_broker_event_policy(tmp_path, monkeypatch):
    target_path = tmp_path / "events.jsonl"
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(target_path), raising=False)

    bridge = _create_mock_bridge()
    u = MockUpdate(order_id="ord_dup", execution_id="ex_dup")

    # First dispatch
    bridge._dispatch_trade_update(u)
    # Duplicate dispatch with identical event_key
    bridge._dispatch_trade_update(u)

    # 1. Audit trail: Raw event is logged both times for audit record
    lines = target_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # 2. Downstream handler: Duplicate was filtered by _remember_trade_update
    assert bridge.on_entry_order_update.call_count == 1


def test_multi_process_path_isolation_daytrade_vs_swing(tmp_path):
    """Multi-process safety: Daytrade and Swing run in separate processes with isolated log paths."""
    import subprocess
    import sys

    daytrade_log = tmp_path / "daytrade_broker_events.jsonl"
    swing_log = tmp_path / "swing_broker_events.jsonl"

    script_template = (
        "import sys, os, json, threading\n"
        "from pathlib import Path\n"
        "from collections import deque\n"
        "from unittest.mock import MagicMock\n"
        "sys.path.insert(0, os.getcwd())\n"
        "import config as config_module\n"
        "from alpaca_bridge import AlpacaBridge\n"
        "class MockOrder:\n"
        "    def __init__(self, order_id, symbol):\n"
        "        self.id = order_id\n"
        "        self.symbol = symbol\n"
        "class MockUpdate:\n"
        "    def __init__(self, order_id, symbol):\n"
        "        self.event = 'fill'\n"
        "        self.order_id = order_id\n"
        "        self.execution_id = 'ex_' + order_id\n"
        "        self.order_status = 'filled'\n"
        "        self.position_qty = 10.0\n"
        "        self.price = 100.0\n"
        "        self.order = MockOrder(order_id, symbol)\n"
        "config_module.BROKER_EVENTS_PATH = '{path}'\n"
        "bridge = AlpacaBridge.__new__(AlpacaBridge)\n"
        "bridge._trade_stream_lock = threading.RLock()\n"
        "bridge._seen_trade_update_keys = set()\n"
        "bridge._seen_trade_update_order = deque(maxlen=1000)\n"
        "bridge.on_entry_order_update = MagicMock()\n"
        "for i in range(15):\n"
        "    bridge._dispatch_trade_update(MockUpdate(order_id='{prefix}_' + str(i), symbol='{symbol}'))\n"
    )

    daytrade_code = script_template.format(path=str(daytrade_log).replace("\\", "/"), prefix="DT", symbol="NVDA")
    swing_code = script_template.format(path=str(swing_log).replace("\\", "/"), prefix="SW", symbol="SPY")

    p1 = subprocess.Popen([sys.executable, "-c", daytrade_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    p2 = subprocess.Popen([sys.executable, "-c", swing_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    out1, err1 = p1.communicate(timeout=30)
    out2, err2 = p2.communicate(timeout=30)

    assert p1.returncode == 0, f"Daytrade process failed: {err1}"
    assert p2.returncode == 0, f"Swing process failed: {err2}"

    assert daytrade_log.exists()
    assert swing_log.exists()

    dt_lines = daytrade_log.read_text(encoding="utf-8").strip().splitlines()
    sw_lines = swing_log.read_text(encoding="utf-8").strip().splitlines()

    assert len(dt_lines) == 15
    assert len(sw_lines) == 15

    for line in dt_lines:
        rec = json.loads(line)
        assert rec["symbol"] == "NVDA"
        assert rec["order_id"].startswith("DT_")

    for line in sw_lines:
        rec = json.loads(line)
        assert rec["symbol"] == "SPY"
        assert rec["order_id"].startswith("SW_")


def test_shared_broker_event_path_rejected_in_multi_process_canary(monkeypatch):
    """Verify that resolving shared broker_events.jsonl raises RuntimeError when MULTI_PROCESS_CANARY is set."""
    monkeypatch.setenv("MULTI_PROCESS_CANARY", "1")
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", "logs/broker_events.jsonl", raising=False)
    bridge = AlpacaBridge.__new__(AlpacaBridge)
    with pytest.raises(RuntimeError, match="Shared broker event path is not permitted"):
        bridge.get_broker_events_path()


