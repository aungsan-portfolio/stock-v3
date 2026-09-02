"""
tests/test_observability.py -- Unit tests for Scoped P0 Observability components.
"""
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from observability.events import (
    LatchTrigger,
    ClearReason,
    build_latch_set_event,
    build_latch_clear_event,
    build_latch_active_heartbeat,
)
from observability.flight_recorder import FlightRecorderHandler
from observability.incidents import IncidentBuilder, BUNDLE_VERSION, OBSERVABILITY_VERSION
from observability.state_machine import LatchStateMachine


class TestFlightRecorder:
    def test_ring_buffer_capacity_capping(self):
        handler = FlightRecorderHandler(capacity=10)
        logger = logging.getLogger("test_fr")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        for i in range(25):
            logger.info("Message %d", i)

        events = handler.snapshot()
        assert len(events) == 10
        assert "Message 24" in events[-1]["msg"]
        assert "Message 15" in events[0]["msg"]

    def test_flight_recorder_dump(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            handler = FlightRecorderHandler(capacity=10)
            logger = logging.getLogger("test_fr_dump")
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)

            logger.info("Hello Flight Recorder")
            dump_file = tmp_path / "snapshot.jsonl"
            handler.dump(str(dump_file))

            assert dump_file.exists()
            lines = dump_file.read_text(encoding="utf-8").strip().splitlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert "Hello Flight Recorder" in record["msg"]
            assert record["lvl"] == "INFO"


class TestLatchStateMachine:
    def test_edge_triggered_latch_set(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            incident_builder = IncidentBuilder(incident_base_dir=str(tmp_path / "incidents"))
            sm = LatchStateMachine(run_id="test_run_123", incident_builder=incident_builder)

            # 1. First trigger -> Transition ARMED -> LATCHED
            first_call = sm.latch(LatchTrigger.UNPROTECTED_POSITION, symbol="PLTR", parent_order_id="ord-01")
            assert first_call is True
            assert sm.is_latched is True
            assert sm.trigger_source == LatchTrigger.UNPROTECTED_POSITION

            # 2. Second trigger -> Repeat detection is a no-op (no duplicate SET)
            second_call = sm.latch(LatchTrigger.UNPROTECTED_POSITION, symbol="PLTR", parent_order_id="ord-01")
            assert second_call is False
            assert sm.is_latched is True

    def test_change_request_1_reject_unknown_trigger(self):
        """CR-1: UNKNOWN trigger is strictly forbidden at runtime."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            incident_builder = IncidentBuilder(incident_base_dir=str(tmp_path / "incidents"))
            sm = LatchStateMachine(run_id="test_run_123", incident_builder=incident_builder)

            with pytest.raises(ValueError, match="UNKNOWN is strictly forbidden"):
                sm.latch(LatchTrigger.UNKNOWN, symbol="TEST")

    def test_latch_clear_transition(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            incident_builder = IncidentBuilder(incident_base_dir=str(tmp_path / "incidents"))
            sm = LatchStateMachine(run_id="test_run_123", incident_builder=incident_builder)

            sm.latch(LatchTrigger.DAILY_LOSS_LIMIT)
            assert sm.is_latched is True

            cleared = sm.clear(ClearReason.STARTUP_RECONCILIATION_CLEAN, reconcile={"clean": True})
            assert cleared is True
            assert sm.is_latched is False

    def test_latch_heartbeat_emission(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            incident_builder = IncidentBuilder(incident_base_dir=str(tmp_path / "incidents"))
            sm = LatchStateMachine(run_id="test_run_123", incident_builder=incident_builder)

            sm.latch(LatchTrigger.ORDER_REJECT)
            # Due immediately when interval=0.0
            assert sm.heartbeat_if_due(interval_seconds=0.0) is True


class TestIncidentBundleBuilder:
    def test_5_file_bundle_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            incident_base = tmp_path / "incidents"
            broker_events_path = tmp_path / "broker_events.jsonl"
            broker_events_path.write_text('{"event": "fill", "order_id": "ord-01"}\n', encoding="utf-8")

            builder = IncidentBuilder(
                incident_base_dir=str(incident_base),
                broker_events_path=str(broker_events_path),
            )

            mock_bridge = MagicMock()
            mock_acc = MagicMock()
            mock_acc.equity = 98450.0
            mock_acc.cash = 98450.0
            mock_acc.buying_power = 196900.0
            mock_bridge.get_account.return_value = mock_acc
            mock_bridge.get_positions.return_value = []
            mock_bridge.get_working_orders.return_value = []

            latch_event = build_latch_set_event(
                run_id="test_bundle_run",
                trigger_source=LatchTrigger.UNPROTECTED_POSITION,
                symbol="PLTR",
                parent_order_id="1120bef1-PLTR-0031",
                stop_order_id="1120bef1-PLTR-0032",
                filled_qty=21.0,
                equity=98450.0,
            )

            bundle_dir = builder.dump_bundle(latch_event=latch_event, bridge=mock_bridge)
            assert bundle_dir is not None
            assert bundle_dir.exists()

            # Verify all 5 files exist
            f_latch = bundle_dir / "latch_event.json"
            f_snapshot = bundle_dir / "snapshot.jsonl"
            f_account = bundle_dir / "account_state.json"
            f_broker = bundle_dir / "broker_events.jsonl"
            f_manifest = bundle_dir / "manifest.json"

            assert f_latch.exists()
            assert f_snapshot.exists()
            assert f_account.exists()
            assert f_broker.exists()
            assert f_manifest.exists()

            # Check manifest version metadata (Change Request #3)
            manifest_data = json.loads(f_manifest.read_text(encoding="utf-8"))
            assert manifest_data["bundle_version"] == BUNDLE_VERSION
            assert manifest_data["observability_version"] == OBSERVABILITY_VERSION
            assert manifest_data["run_id"] == "test_bundle_run"

            # Check account state data
            account_data = json.loads(f_account.read_text(encoding="utf-8"))
            assert account_data["equity"] == 98450.0

    def test_change_request_2_incident_builder_never_raises(self):
        """CR-2: dump_bundle must never raise even if filesystem write errors occur."""
        builder = IncidentBuilder(incident_base_dir="/invalid/non_existent_read_only_path/incidents")
        latch_event = {"run_id": "err_test"}

        with patch.object(builder, "create_incident_dir", side_effect=PermissionError("Mock Permission Denied")):
            # Must return None and not raise
            res = builder.dump_bundle(latch_event=latch_event)
            assert res is None
