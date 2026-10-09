"""
tests/test_deterministic_telemetry_join.py

Unit test suite verifying deterministic telemetry foreign-key linkage:
1. TEST 1: execute_signal PLACED result contains non-null parent_order_id, client_order_id, stop_order_id, tp_order_id.
2. TEST 2: Gate ledger STAGE_ORDER_SUBMIT metadata records the exact same IDs.
3. TEST 3: Broker event stream records same order_id, client_order_id, and strictly separates observed_ts from event_ts.
4. TEST 4: Child leg events contain order_role in (STOP_LOSS, TAKE_PROFIT) and link to parent_order_id.
5. TEST 5: Complete deterministic join without time-window proximity heuristics.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from strategies.base import TradeSignal
from strategies.constants import StrategyName
from strategies.order_manager import execute_signal
from alpaca_bridge import AlpacaBridge


def _mock_trade_signal():
    return TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=150.00,
        stop_price=148.00,
        target_price=154.00,
        confidence=0.85,
        atr=1.5,
        risk_per_share=2.0,
        reason="Test breakout",
        strategy=StrategyName.ORB,
        signal_id="sig_test_aapl_001",
        bar_timestamp="2026-09-21T13:30:00Z",
    )


def test_placed_result_contains_all_deterministic_ids():
    """TEST 1: PLACED result dictionary contains parent, client, stop, and tp IDs."""
    from strategies.order_idempotency import idempotency_manager
    idempotency_manager.clear()
    sig = _mock_trade_signal()
    bridge = MagicMock()
    bridge.is_connected = True
    bridge.is_paper = True
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.place_bracket_order.return_value = {
        "order_id": "parent-12345",
        "stop_order_id": "stop-67890",
        "tp_order_id": "tp-99999",
        "order_ids": ["parent-12345", "stop-67890", "tp-99999"],
        "status": "Submitted",
    }

    try:
        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            with patch("strategies.order_manager.validate_geometry", return_value=(True, "", 2.0, 4.0, 2.0)):
                with patch("strategies.execution_quote.validate_execution_quote") as mock_vq:
                    mock_vq.return_value = MagicMock(valid=True, quote=MagicMock(ask=150.00, bid=149.95, last=150.00, timestamp=datetime.now(timezone.utc)))
                    with patch("config.BRACKET_ORDER_ENABLED", True), patch("config.SPY_TREND_GUARD_ENABLED", False):
                        import time
                        from strategies.base import EntryReadiness
                        readiness = EntryReadiness(
                            entry_allowed=True,
                            suppression_reason=None,
                            cycle_id="cycle_01",
                            bar_ts="2026-09-21T13:30:00Z",
                            created_at=time.time(),
                            expires_at=time.time() + 300.0,
                        )
                        res = execute_signal(
                            signal=sig,
                            bridge=bridge,
                            equity=100000.0,
                            current_pnl=0.0,
                            dry_run=False,
                            entry_readiness=readiness,
                        )

        assert res["status"] == "PLACED", f"res was: {res}"
        assert res["parent_order_id"] == "parent-12345"
        assert "se3-aapl-b-orb" in res["client_order_id"].lower()
        assert res["stop_order_id"] == "stop-67890"
        assert res["tp_order_id"] == "tp-99999"
    finally:
        from strategies.entry_lifecycle import entry_lifecycle
        entry_lifecycle.finalize_entry("parent-12345", reason="test_cleanup")


def test_gate_ledger_metadata_contains_deterministic_ids(tmp_path):
    """TEST 2: Gate ledger STAGE_ORDER_SUBMIT metadata records the exact same IDs."""
    from strategies.gate_ledger import log_gate_decision

    test_ledger = tmp_path / "test_ledger.jsonl"
    result = {
        "status": "PLACED",
        "shares": 10,
        "parent_order_id": "parent-12345",
        "client_order_id": "client-abcde",
        "stop_order_id": "stop-67890",
        "tp_order_id": "tp-99999",
    }

    log_gate_decision(
        evaluation_id="eval_001",
        symbol="AAPL",
        bar="2026-09-21T13:30:00Z",
        strategy="ORB",
        stage="STAGE_ORDER_SUBMIT",
        decision="PASS",
        reason_code="ORDER_SUBMITTED",
        metadata={
            "order_status": result.get("status"),
            "shares": result.get("shares"),
            "parent_order_id": result.get("parent_order_id"),
            "client_order_id": result.get("client_order_id"),
            "stop_order_id": result.get("stop_order_id"),
            "tp_order_id": result.get("tp_order_id"),
        },
        filepath=str(test_ledger),
    )

    with open(test_ledger, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())

    assert rec["decision"] == "PASS"
    meta = rec["metadata"]
    assert meta["parent_order_id"] == "parent-12345"
    assert meta["client_order_id"] == "client-abcde"
    assert meta["stop_order_id"] == "stop-67890"
    assert meta["tp_order_id"] == "tp-99999"


def test_broker_event_timestamps_and_role_separation(tmp_path, monkeypatch):
    """TEST 3 & 4: Broker event stream separates observed_ts from event_ts and records roles."""
    import config as config_module
    custom_broker_log = tmp_path / "broker_events.jsonl"
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(custom_broker_log), raising=False)

    bridge = AlpacaBridge(mode="daytrade")
    bridge.register_bracket_order_roles(
        parent_id="parent-12345",
        stop_leg_id="stop-67890",
        tp_leg_id="tp-99999",
        client_order_id="client-abcde",
    )

    # 1. Simulate Parent Order New Update (with authoritative broker timestamp)
    mock_order_parent = MagicMock()
    mock_order_parent.id = "parent-12345"
    mock_order_parent.symbol = "AAPL"
    mock_order_parent.status = "new"
    mock_order_parent.filled_qty = "0.0"
    mock_order_parent.filled_avg_price = "0.0"
    mock_order_parent.client_order_id = "client-abcde"

    mock_update_parent = MagicMock()
    mock_update_parent.event = "new"
    mock_update_parent.order = mock_order_parent
    mock_update_parent.timestamp = datetime(2026, 9, 21, 13, 30, 1, 123456, tzinfo=timezone.utc)
    mock_update_parent.execution_id = "exec-p-1"

    bridge._dispatch_trade_update(mock_update_parent)

    # 2. Simulate Stop-Loss Leg Fill Update (authoritative broker timestamp)
    mock_order_stop = MagicMock()
    mock_order_stop.id = "stop-67890"
    mock_order_stop.symbol = "AAPL"
    mock_order_stop.status = "filled"
    mock_order_stop.filled_qty = "10.0"
    mock_order_stop.filled_avg_price = "148.00"
    mock_order_stop.client_order_id = "client-abcde"

    mock_update_stop = MagicMock()
    mock_update_stop.event = "fill"
    mock_update_stop.order = mock_order_stop
    mock_update_stop.timestamp = datetime(2026, 9, 21, 13, 35, 10, 654321, tzinfo=timezone.utc)
    mock_update_stop.execution_id = "exec-s-1"

    bridge._dispatch_trade_update(mock_update_stop)

    # Verify written events
    assert custom_broker_log.exists()
    lines = [json.loads(line) for line in custom_broker_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2

    # Parent event assertions
    p_ev = lines[0]
    assert p_ev["order_id"] == "parent-12345"
    assert p_ev["client_order_id"] == "client-abcde"
    assert p_ev["order_role"] == "ENTRY"
    assert p_ev["parent_order_id"] == "parent-12345"
    assert p_ev["event_ts"] == "2026-09-21T13:30:01.123456+00:00"
    assert p_ev["observed_ts"] is not None
    assert p_ev["event_ts"] != p_ev["observed_ts"]  # Strictly separated!

    # Child stop-loss event assertions
    s_ev = lines[1]
    assert s_ev["order_id"] == "stop-67890"
    assert s_ev["client_order_id"] == "client-abcde"
    assert s_ev["order_role"] == "STOP_LOSS"
    assert s_ev["parent_order_id"] == "parent-12345"
    assert s_ev["event_ts"] == "2026-09-21T13:35:10.654321+00:00"


def test_deterministic_primary_key_join():
    """TEST 5: Verify 100% deterministic key join without timestamp heuristics."""
    gate_record = {
        "evaluation_id": "eval-uuid-999",
        "symbol": "AAPL",
        "metadata": {
            "parent_order_id": "parent-12345",
            "client_order_id": "client-abcde",
            "stop_order_id": "stop-67890",
            "tp_order_id": "tp-99999",
        }
    }

    broker_events = [
        {
            "order_id": "parent-12345",
            "client_order_id": "client-abcde",
            "event": "fill",
            "order_role": "ENTRY",
            "parent_order_id": "parent-12345",
            "filled_qty": 10.0,
            "price": 150.0,
        },
        {
            "order_id": "tp-99999",
            "client_order_id": "client-abcde",
            "event": "fill",
            "order_role": "TAKE_PROFIT",
            "parent_order_id": "parent-12345",
            "filled_qty": 10.0,
            "price": 154.0,
        },
    ]

    # Direct Primary/Foreign Key Join:
    target_parent_id = gate_record["metadata"]["parent_order_id"]

    matched_entry = [e for e in broker_events if e["parent_order_id"] == target_parent_id and e["order_role"] == "ENTRY"]
    matched_exit = [e for e in broker_events if e["parent_order_id"] == target_parent_id and e["order_role"] in ("STOP_LOSS", "TAKE_PROFIT")]

    assert len(matched_entry) == 1
    assert matched_entry[0]["filled_qty"] == 10.0
    assert len(matched_exit) == 1
    assert matched_exit[0]["order_role"] == "TAKE_PROFIT"
    assert matched_exit[0]["filled_qty"] == 10.0
    # ZERO timestamp heuristics used!


def test_restart_resilient_order_role_recovery(tmp_path, monkeypatch):
    """
    TEST 6: Restart-Resilient Role Mapping Recovery.
    Verifies that:
      1. Order roles registered by bridge instance 1 are persisted to disk.
      2. When process restarts (bridge instance 1 destroyed, bridge instance 2 instantiated),
         child leg events arriving from broker WebSocket still deterministically resolve
         order_role ('STOP_LOSS' / 'TAKE_PROFIT') and parent_order_id.
      3. Secondary ledger recovery: if registry file is wiped, STAGE_ORDER_SUBMIT in gate ledger
         recovers the deterministic parent-child linkage.
    """
    import config as config_module
    events_log = tmp_path / "broker_events.jsonl"
    monkeypatch.setattr(config_module, "BROKER_EVENTS_PATH", str(events_log), raising=False)

    # 1. First Process Instance Submits and Registers Bracket
    bridge1 = AlpacaBridge(mode="daytrade")
    bridge1.register_bracket_order_roles(
        parent_id="P_RESTART_100",
        stop_leg_id="S_RESTART_200",
        tp_leg_id="T_RESTART_300",
        client_order_id="C_RESTART_001",
    )

    # Verify persistent role file exists
    role_reg_file = tmp_path / "broker_events_order_roles.json"
    assert role_reg_file.exists(), "Order role registry must be persisted to disk"
    persisted_roles = json.loads(role_reg_file.read_text(encoding="utf-8"))
    assert "S_RESTART_200" in persisted_roles
    assert persisted_roles["S_RESTART_200"]["role"] == "STOP_LOSS"
    assert persisted_roles["S_RESTART_200"]["parent_order_id"] == "P_RESTART_100"

    # 2. SIMULATE PROCESS RESTART: Destroy bridge1
    del bridge1

    # 3. New Process Instance Starts Up
    bridge2 = AlpacaBridge(mode="daytrade")

    # Inbound trade update arrives from broker WebSocket for the child stop-loss leg:
    mock_order_stop = MagicMock()
    mock_order_stop.id = "S_RESTART_200"
    mock_order_stop.symbol = "AAPL"
    mock_order_stop.status = "filled"
    mock_order_stop.filled_qty = "50.0"
    mock_order_stop.filled_avg_price = "148.50"
    mock_order_stop.client_order_id = None  # Broker often omits client_order_id on child legs!

    mock_update_stop = MagicMock()
    mock_update_stop.event = "fill"
    mock_update_stop.order = mock_order_stop
    mock_update_stop.timestamp = datetime(2026, 9, 21, 14, 0, 0, tzinfo=timezone.utc)
    mock_update_stop.execution_id = "exec_stop_restart_01"

    bridge2._dispatch_trade_update(mock_update_stop)

    # Verify child event in broker_events.jsonl has deterministic role and parent_order_id
    lines = events_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["order_id"] == "S_RESTART_200"
    assert rec["order_role"] == "STOP_LOSS"
    assert rec["parent_order_id"] == "P_RESTART_100"
    assert rec["client_order_id"] == "C_RESTART_001"

    # 4. SECONDARY RECOVERY TEST: Even if order_roles.json is deleted, recover from gate ledger
    role_reg_file.unlink()  # simulate file loss
    assert not role_reg_file.exists()

    # Create mock gate ledger with STAGE_ORDER_SUBMIT
    ledger_file = tmp_path / "gate_ledger.jsonl"
    ledger_record = {
        "evaluation_id": "eval_recovery_999",
        "symbol": "TSLA",
        "bar": "2026-09-21T14:30:00Z",
        "strategy": "ORB",
        "stage": "STAGE_ORDER_SUBMIT",
        "decision": "PASS",
        "terminal_decision": True,
        "metadata": {
            "parent_order_id": "P_LEDGER_555",
            "client_order_id": "C_LEDGER_777",
            "stop_order_id": "S_LEDGER_888",
            "tp_order_id": "T_LEDGER_999",
        },
    }
    ledger_file.write_text(json.dumps(ledger_record) + "\n", encoding="utf-8")
    monkeypatch.setattr("strategies.gate_ledger.DEFAULT_LEDGER_PATH", str(ledger_file))

    # Fresh bridge instance recovers from gate ledger
    bridge3 = AlpacaBridge(mode="daytrade")
    info_stop = bridge3.get_order_role_info("S_LEDGER_888")
    assert info_stop["role"] == "STOP_LOSS"
    assert info_stop["parent_order_id"] == "P_LEDGER_555"
    assert info_stop["client_order_id"] == "C_LEDGER_777"

    info_tp = bridge3.get_order_role_info("T_LEDGER_999")
    assert info_tp["role"] == "TAKE_PROFIT"
    assert info_tp["parent_order_id"] == "P_LEDGER_555"


def test_os_process_restart_order_role_recovery(tmp_path):
    """
    TEST 7: True OS-Level Process Boundary Restart Verification.
    Executes two sequential OS subprocesses:
      Process 1: Submits bracket order, persists registry to disk, exits with code 0.
      Process 2: Fresh Python process starts, instantiates AlpacaBridge,
                 receives child trade update, resolves STOP_LOSS + parent_order_id + client_order_id.
    """
    import subprocess
    import sys

    events_log = tmp_path / "os_broker_events.jsonl"
    events_log_str = str(events_log).replace("\\", "/")

    p1_code = (
        "import sys, os\n"
        "sys.path.insert(0, os.getcwd())\n"
        "import config as config_module\n"
        f"config_module.BROKER_EVENTS_PATH = '{events_log_str}'\n"
        "from alpaca_bridge import AlpacaBridge\n"
        "bridge = AlpacaBridge(mode='daytrade')\n"
        "bridge.register_bracket_order_roles(\n"
        "    parent_id='P_OS_101',\n"
        "    stop_leg_id='S_OS_202',\n"
        "    tp_leg_id='T_OS_303',\n"
        "    client_order_id='C_OS_404',\n"
        ")\n"
        "sys.exit(0)\n"
    )

    p2_code = (
        "import sys, os\n"
        "from unittest.mock import MagicMock\n"
        "from datetime import datetime, timezone\n"
        "sys.path.insert(0, os.getcwd())\n"
        "import config as config_module\n"
        f"config_module.BROKER_EVENTS_PATH = '{events_log_str}'\n"
        "from alpaca_bridge import AlpacaBridge\n"
        "class MockOrder:\n"
        "    def __init__(self, order_id, symbol):\n"
        "        self.id = order_id\n"
        "        self.symbol = symbol\n"
        "        self.status = 'filled'\n"
        "        self.filled_qty = '15.0'\n"
        "        self.filled_avg_price = '210.50'\n"
        "        self.client_order_id = None\n"
        "class MockUpdate:\n"
        "    def __init__(self, order_id, symbol):\n"
        "        self.event = 'fill'\n"
        "        self.order_id = order_id\n"
        "        self.execution_id = 'ex_os_stop_01'\n"
        "        self.order_status = 'filled'\n"
        "        self.position_qty = 15.0\n"
        "        self.price = 210.50\n"
        "        self.timestamp = datetime(2026, 9, 21, 14, 30, 0, tzinfo=timezone.utc)\n"
        "        self.order = MockOrder(order_id, symbol)\n"
        "bridge = AlpacaBridge(mode='daytrade')\n"
        "bridge._dispatch_trade_update(MockUpdate(order_id='S_OS_202', symbol='TSLA'))\n"
        "sys.exit(0)\n"
    )

    # Run Process 1 to completion
    p1 = subprocess.Popen([sys.executable, "-c", p1_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out1, err1 = p1.communicate(timeout=60)
    assert p1.returncode == 0, f"Process 1 failed: {err1}"

    # Run Process 2 to completion
    p2 = subprocess.Popen([sys.executable, "-c", p2_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out2, err2 = p2.communicate(timeout=60)
    assert p2.returncode == 0, f"Process 2 failed: {err2}"

    # Verify event stream output from Process 2
    assert events_log.exists()
    lines = events_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["order_id"] == "S_OS_202"
    assert rec["order_role"] == "STOP_LOSS"
    assert rec["parent_order_id"] == "P_OS_101"
    assert rec["client_order_id"] == "C_OS_404"


