"""
test_daily_budget_and_broker_anchor.py - Verification suite for broker calendar-day anchor,
preflight daily-budget gate, provenance stamping, and pytest risk-state isolation.
"""
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from alpaca_bridge import AlpacaBridge
from preflight import (
    GateResult,
    PreflightDecision,
    check_daily_loss_budget,
    run_preflight_audit,
)


class MockAccount:
    def __init__(self, equity=98391.30, last_equity=98391.30, status="ACTIVE", account_number="PA3UODIC5JTA"):
        self.equity = equity
        self.last_equity = last_equity
        self.cash = equity
        self.status = status
        self.account_number = account_number
        self.id = "mock-id-12345"


@pytest.fixture
def isolated_bridge():
    """Fixture providing AlpacaBridge isolated in a safe TemporaryDirectory."""
    with tempfile.TemporaryDirectory() as td:
        bridge = AlpacaBridge(mode="daytrade")
        bridge._connected = True
        bridge._account_id = "PA3UODIC5JTA"
        bridge._risk_state_dir = td
        bridge._client = MagicMock()
        bridge._client.get_account.return_value = MockAccount()
        yield bridge


def test_broker_last_equity_anchor(isolated_bridge):
    """Test 1: Start-of-day equity strictly anchors to broker last_equity (not nominal 100k)."""
    with patch("strategies.session.now_eastern") as mock_dt:
        mock_dt.return_value.date.return_value = "2026-09-28"
        sod = isolated_bridge.snapshot_start_of_day_equity()
        assert sod == 98391.30
        assert isolated_bridge._start_of_day_equity == 98391.30

        isolated_bridge.get_net_liquidation = MagicMock(return_value=98391.30)
        pnl = isolated_bridge.account_daily_pnl()
        assert pnl == 0.0


def test_reject_legacy_nominal_without_provenance(isolated_bridge):
    """Test 2: Legacy risk state file with nominal $100k but NO provenance stamp is rejected."""
    state_file = os.path.join(isolated_bridge._risk_state_dir, "daytrade_risk_state_PA3UODIC5JTA.json")
    legacy_data = {
        "account_id": "PA3UODIC5JTA",
        "date": "2026-09-28",
        "start_of_day_equity": 100000.0,
        "suspended": False,
        # Notice: NO provenance block!
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(legacy_data, f)

    with patch("strategies.session.now_eastern") as mock_dt:
        mock_dt.return_value.date.return_value = "2026-09-28"
        sod = isolated_bridge.snapshot_start_of_day_equity()
        # Legacy nominal $100,000 must be REJECTED in favor of broker last_equity ($98,391.30)
        assert sod == 98391.30
        assert isolated_bridge._start_of_day_equity == 98391.30

        # Verify the file was overwritten with provenance stamp
        with open(state_file, "r", encoding="utf-8") as f:
            updated_data = json.load(f)
        assert updated_data["start_of_day_equity"] == 98391.30
        assert "provenance" in updated_data
        assert updated_data["provenance"]["source"] == "broker_last_equity"


def test_pytest_risk_state_diversion():
    """Test 3: Pytest guard diverts risk state directory away from production data directory."""
    default_bridge = AlpacaBridge(mode="daytrade")
    # _risk_state_dir is None by default
    assert default_bridge._risk_state_dir is None

    # In pytest context, _get_risk_state_dir() must NOT point to repo's data directory
    resolved_dir = default_bridge._get_risk_state_dir()
    repo_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    assert os.path.abspath(resolved_dir) != os.path.abspath(repo_data_dir)
    assert "stock_engine_test_risk_state" in resolved_dir or "temp" in resolved_dir.lower()


def test_preflight_daily_budget_gate_pass(isolated_bridge):
    """Test 4: Preflight Gate 8 passes when broker calendar-day PnL is healthy."""
    with patch.object(isolated_bridge, "account_daily_pnl", return_value=0.0), \
         patch.object(isolated_bridge, "get_net_liquidation", return_value=98391.30):

        gate_res = check_daily_loss_budget(isolated_bridge)
        assert gate_res.passed is True
        assert gate_res.status == "PASS"
        assert "healthy" in gate_res.message
        assert gate_res.category == "CRITICAL"


def test_preflight_daily_budget_gate_fail_closed(isolated_bridge):
    """Test 5: Preflight Gate 8 fails and triggers NO_GO when calendar-day loss breaches budget."""
    with patch.object(isolated_bridge, "account_daily_pnl", return_value=-350.0), \
         patch.object(isolated_bridge, "get_net_liquidation", return_value=98391.30):

        gate_res = check_daily_loss_budget(isolated_bridge)
        assert gate_res.passed is False
        assert gate_res.status == "FAIL"
        assert "BREACHED" in gate_res.message

        with patch("preflight.check_historical_data_freshness", return_value=GateResult("Data", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_intraday_data_readiness", return_value=GateResult("Intraday", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_broker_reconciliation", return_value=GateResult("Recon", "CRITICAL", True, "clean", {}, status="PASS")), \
             patch("preflight.check_position_protection", return_value=GateResult("Protect", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_portfolio_exposure", return_value=GateResult("Exposure", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_event_and_models", return_value=GateResult("Event", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_trade_stream_readiness", return_value=GateResult("Stream", "CRITICAL", True, "ok", {}, status="PASS")):

            report = run_preflight_audit(bridge=isolated_bridge, save_report=False)
            assert report.decision == PreflightDecision.NO_GO
            assert report.status == "NO_GO"
            assert report.entry_allowed is False
            assert report.critical_fail >= 1


def test_clean_startup_rearm_on_new_date(isolated_bridge):
    """Test 6: Day rollover automatically re-arms suspended status and re-anchors to broker last_equity."""
    state_file = os.path.join(isolated_bridge._risk_state_dir, "daytrade_risk_state_PA3UODIC5JTA.json")
    yesterday_state = {
        "account_id": "PA3UODIC5JTA",
        "date": "2026-09-28",
        "start_of_day_equity": 98391.30,
        "suspended": True,
        "provenance": {"source": "broker_last_equity", "captured_at": "2026-09-28T12:00:00Z"},
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(yesterday_state, f)

    # Fast forward to next date: 2026-09-29
    with patch("strategies.session.now_eastern") as mock_dt:
        mock_dt.return_value.date.return_value = "2026-09-29"

        isolated_bridge._daytrade_suspended = True
        sod = isolated_bridge.snapshot_start_of_day_equity()

        # New date must re-anchor to last_equity and clear suspended flag
        assert sod == 98391.30
        assert isolated_bridge._daytrade_suspended is False

        with open(state_file, "r", encoding="utf-8") as f:
            updated_data = json.load(f)
        assert updated_data["date"] == "2026-09-29"
        assert updated_data["suspended"] is False
        assert updated_data["provenance"]["source"] == "broker_last_equity"
