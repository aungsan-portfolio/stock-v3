"""
tests/test_chaos_manual_halt.py -- Chaos drill acceptance test for MANUAL_HALT incident dump.
"""
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from observability.events import LatchTrigger, ClearReason
from observability.flight_recorder import attach_flight_recorder
from observability.incidents import IncidentBuilder
from observability.state_machine import get_latch_state_machine
from strategies.halt_manager import halt_manager, LATCHING_HALT_PROTECTION_FAILURE


def test_chaos_drill_manual_halt_incident_dump():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 1. Attach Flight Recorder
        handler = attach_flight_recorder(capacity=100)
        logger = logging.getLogger("chaos_test")
        logger.info("Pre-incident normal trade poll event 1")
        logger.info("Pre-incident normal trade poll event 2")

        # 2. Configure State Machine to use temp incident dir
        incident_base = tmp_path / "incidents"
        incident_builder = IncidentBuilder(incident_base_dir=str(incident_base))
        sm = get_latch_state_machine()
        sm.run_id = "chaos_run_01"
        sm.incident_builder = incident_builder
        sm.clear(ClearReason.MANUAL_RESET)
        halt_manager.reset_all()

        # 3. Mock Bridge
        mock_bridge = MagicMock()
        mock_acc = MagicMock()
        mock_acc.equity = 99120.0
        mock_acc.cash = 50000.0
        mock_acc.buying_power = 100000.0
        mock_bridge.get_account.return_value = mock_acc
        mock_bridge.get_positions.return_value = []
        mock_bridge.get_working_orders.return_value = []
        mock_bridge._start_of_day_equity = 100000.0

        # 4. Trigger MANUAL_HALT Chaos Drill
        logger.warning("Triggering simulated MANUAL_HALT chaos drill...")
        halt_manager.add("MANUAL_HALT", bridge=mock_bridge)

        # 5. Verify Engine Safety Gates
        assert halt_manager.entries_allowed() is False
        assert sm.is_latched is True
        assert sm.trigger_source == LatchTrigger.MANUAL_HALT

        # 6. Verify Complete 5-File Incident Bundle on Disk
        assert sm._latest_incident_dir is not None
        bundle_dir = Path(sm._latest_incident_dir)
        assert bundle_dir.exists()

        f_latch = bundle_dir / "latch_event.json"
        f_snapshot = bundle_dir / "snapshot.jsonl"
        f_account = bundle_dir / "account_state.json"
        f_broker = bundle_dir / "broker_events.jsonl"
        f_manifest = bundle_dir / "manifest.json"

        assert f_latch.exists(), "latch_event.json missing"
        assert f_snapshot.exists(), "snapshot.jsonl missing"
        assert f_account.exists(), "account_state.json missing"
        assert f_broker.exists(), "broker_events.jsonl missing"
        assert f_manifest.exists(), "manifest.json missing"

        # 7. Verify JSON validity of all files
        latch_data = json.loads(f_latch.read_text(encoding="utf-8"))
        assert latch_data["event"] == "PROTECTION_LATCH_SET"
        assert latch_data["trigger_source"] == "MANUAL_HALT"
        assert latch_data["run_id"] == "chaos_run_01"
        assert latch_data["equity"] == 99120.0

        snapshot_lines = f_snapshot.read_text(encoding="utf-8").strip().splitlines()
        assert len(snapshot_lines) >= 2

        manifest_data = json.loads(f_manifest.read_text(encoding="utf-8"))
        assert manifest_data["bundle_version"] == 1
        assert manifest_data["observability_version"] == 1

        account_data = json.loads(f_account.read_text(encoding="utf-8"))
        assert account_data["equity"] == 99120.0

        # 8. Verify Clear on Clean Startup / Reset
        halt_manager.reset_all()
        assert halt_manager.entries_allowed() is True
        assert sm.is_latched is False
