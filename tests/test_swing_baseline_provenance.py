"""
Comprehensive Verification Suite: Swing Baseline Provenance and False Halt Recovery
Validates Start-of-Day (SOD) baseline lifecycle, provisional vs verified state,
mathematical false-halt proof, genuine breach latching, and clean session rollover.
"""

from decimal import Decimal
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from alpaca_bridge import AlpacaBridge, EntryGateMock


@pytest.fixture
def test_env():
    """Provides an isolated temporary directory and fake account ID for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        acct_id = "PA_TEST_SWING_PROV_999"
        yield tmp_path, acct_id


def test_oct_15_incident_phantom_halt_cleared(test_env):
    """
    Simulates the exact Oct 15 incident:
    - Stale provisional baseline of $100,607.73 (from pre-settlement Oct 13)
    - Phantom daily loss halt written at -$310.00
    - Broker provides verified Oct 14 close of $99,745.55
    - Current equity is $99,600.62 (true P&L is -$144.93, within $300 limit)
    - Result: False halt is cleared, halt file removed, baseline verified.
    """
    tmp_path, acct_id = test_env
    from strategies.session import now_eastern
    session_date = str(now_eastern().date())

    stale_state = {
        "account_id": acct_id,
        "date": session_date,
        "session_date_et": session_date,
        "start_of_day_equity": 100607.73,
        "status": "PROVISIONAL",
        "suspended": False,
        "provenance": {
            "source": "broker_last_equity",
            "status": "PROVISIONAL",
            "settlement_confirmed": False,
            "captured_at": "2026-10-15T07:35:49.000000+00:00"
        }
    }

    halt_file = tmp_path / f"swing_daily_loss_halt_{acct_id}.json"
    halt_file.write_text(json.dumps({
        "account_id": acct_id,
        "session_date_et": session_date,
        "daily_pnl": "-310.00",
        "daily_loss_limit": "300.00",
        "baseline_equity": "100607.73",
        "halted": True,
        "reason": "DAILY_LOSS_LIMIT",
        "created_at_utc": "2026-10-15T08:00:00+00:00"
    }), encoding="utf-8")

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = acct_id
    bridge._risk_state_dir = str(tmp_path)
    bridge.entry_gate._test_halt_dir = tmp_path
    bridge.entry_gate.halt("DAILY_LOSS_LIMIT")

    mock_acct = MagicMock(
        account_number=acct_id,
        last_equity="99745.55",
        equity="99600.62"
    )

    # Reconcile session baseline
    new_sod = bridge._restore_session_baseline(mock_acct, stale_state, session_date, acct_id)
    assert new_sod == 99745.55
    assert bridge._start_of_day_equity == 99745.55

    # Phantom halt must be cleared because under verified baseline ($99,745.55),
    # true PnL at halt was ($100,607.73 - $310.00) - $99,745.55 = +$552.18,
    # and current PnL is $99,600.62 - $99,745.55 = -$144.93, well within -$300 limit.
    assert not halt_file.exists()
    assert not bridge.entry_gate.halted
    assert "DAILY_LOSS_LIMIT" not in bridge.entry_gate.reasons

    # Check updated risk state file
    saved_state_file = tmp_path / f"daytrade_risk_state_{acct_id}.json"
    assert saved_state_file.exists()
    with open(saved_state_file, "r", encoding="utf-8") as f:
        saved_state = json.load(f)
    assert saved_state["status"] == "VERIFIED"
    assert saved_state["start_of_day_equity"] == 99745.55
    assert saved_state["provenance"]["settlement_confirmed"] is True
    assert saved_state["provenance"]["prior_session_close"] == 99745.55


def test_repeated_reconnects_preserve_verified_baseline(test_env):
    """
    Once a baseline is marked VERIFIED, repeated reconnects and restarts must
    strictly preserve the verified baseline without mutation or drift.
    """
    tmp_path, acct_id = test_env
    from strategies.session import now_eastern
    session_date = str(now_eastern().date())

    verified_state = {
        "account_id": acct_id,
        "date": session_date,
        "session_date_et": session_date,
        "start_of_day_equity": 99745.55,
        "status": "VERIFIED",
        "suspended": False,
        "provenance": {
            "source": "broker_last_equity",
            "status": "VERIFIED",
            "settlement_confirmed": True,
            "prior_session_close": 99745.55,
            "captured_at": "2026-10-15T08:30:00.000000+00:00",
            "verified_at": "2026-10-15T08:30:00.000000+00:00"
        }
    }

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = acct_id
    bridge._risk_state_dir = str(tmp_path)
    bridge.entry_gate._test_halt_dir = tmp_path

    # Simulate broker last_equity transiently reporting a slightly different number
    mock_acct = MagicMock(
        account_number=acct_id,
        last_equity="99750.00",
        equity="99650.00"
    )

    sod_1 = bridge._restore_session_baseline(mock_acct, verified_state, session_date, acct_id)
    assert sod_1 == 99745.55
    assert bridge._start_of_day_equity == 99745.55

    # Simulate subsequent snapshot check
    bridge._client = MagicMock()
    bridge._client.get_account.return_value = mock_acct
    with patch("alpaca_bridge.AlpacaBridge._load_daytrade_risk_state", return_value=verified_state):
        sod_2 = bridge.snapshot_start_of_day_equity()
        assert sod_2 == 99745.55


def test_genuine_loss_then_equity_recovery_preserves_halt(test_env):
    """
    If a genuine breach occurred under the verified baseline (e.g. -$350 loss <= -$300 limit),
    a subsequent recovery of equity to $99,800.00 MUST NOT clear the halt latch!
    """
    tmp_path, acct_id = test_env
    from strategies.session import now_eastern
    session_date = str(now_eastern().date())
    verified_baseline = 99745.55

    halt_file = tmp_path / f"swing_daily_loss_halt_{acct_id}.json"
    halt_file.write_text(json.dumps({
        "account_id": acct_id,
        "session_date_et": session_date,
        "daily_pnl": "-350.00",
        "daily_loss_limit": "300.00",
        "baseline_equity": str(verified_baseline),
        "halted": True,
        "reason": "DAILY_LOSS_LIMIT",
        "created_at_utc": "2026-10-15T10:00:00+00:00"
    }), encoding="utf-8")

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = acct_id
    bridge._risk_state_dir = str(tmp_path)
    bridge.entry_gate._test_halt_dir = tmp_path
    bridge.entry_gate.halt("DAILY_LOSS_LIMIT")

    # Current equity has bounced back up above starting equity ($99,800.00 > $99,745.55)
    mock_acct = MagicMock(
        account_number=acct_id,
        last_equity=str(verified_baseline),
        equity="99800.00"
    )

    state = {
        "account_id": acct_id,
        "date": session_date,
        "start_of_day_equity": verified_baseline,
        "status": "VERIFIED",
        "provenance": {"source": "broker_last_equity", "status": "VERIFIED"}
    }

    bridge._restore_session_baseline(mock_acct, state, session_date, acct_id)

    # Invariant: Halt must be preserved because it was evaluated against verified baseline!
    assert halt_file.exists()
    assert bridge.entry_gate.halted
    assert "DAILY_LOSS_LIMIT" in bridge.entry_gate.reasons


def test_baseline_unverified_distinction(test_env):
    """
    Non-positive SOD equity halts entry with BASELINE_UNVERIFIED,
    and strictly does NOT create a DAILY_LOSS_LIMIT halt file.
    """
    tmp_path, acct_id = test_env
    gate = EntryGateMock()
    gate._test_halt_dir = tmp_path

    blocked, reason = gate.daily_loss_blocked(
        current_equity=99000.0,
        session_start_equity=0.0,  # Invalid baseline
        account_id=acct_id,
        session_date_et="2026-10-15"
    )

    assert blocked is True
    assert "BASELINE_UNVERIFIED" in reason
    assert "BASELINE_UNVERIFIED" in gate.reasons

    # Verify no swing_daily_loss_halt file was created
    halt_file = tmp_path / f"swing_daily_loss_halt_{acct_id}.json"
    assert not halt_file.exists()


def test_next_session_rollover_cleans_up_previous_halt(test_env):
    """
    When session date rolls over from 2026-10-14 to 2026-10-15:
    - Previous session's halt file is cleaned up.
    - DAILY_LOSS_LIMIT reason is discarded.
    """
    tmp_path, acct_id = test_env
    yesterday = "2026-10-14"
    today = "2026-10-15"

    halt_file = tmp_path / f"swing_daily_loss_halt_{acct_id}.json"
    halt_file.write_text(json.dumps({
        "account_id": acct_id,
        "session_date_et": yesterday,
        "daily_pnl": "-320.00",
        "daily_loss_limit": "300.00",
        "baseline_equity": "100000.00",
        "halted": True,
        "reason": "DAILY_LOSS_LIMIT",
        "created_at_utc": "2026-10-14T15:00:00+00:00"
    }), encoding="utf-8")

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = acct_id
    bridge._risk_state_dir = str(tmp_path)
    bridge.entry_gate._test_halt_dir = tmp_path
    bridge.entry_gate.halt("DAILY_LOSS_LIMIT")

    mock_acct = MagicMock(
        account_number=acct_id,
        last_equity="99745.55",
        equity="99745.55"
    )

    # Rollover check in _reconcile_daily_loss_halt_if_needed
    bridge._reconcile_daily_loss_halt_if_needed(
        account_id=acct_id,
        et_date=today,
        acct=mock_acct,
        verified_last_eq=99745.55
    )

    assert not halt_file.exists()
    assert "DAILY_LOSS_LIMIT" not in bridge.entry_gate.reasons


def test_clearing_false_halt_preserves_unrelated_halts(test_env):
    """
    If multiple halt reasons exist (e.g. DAILY_LOSS_LIMIT AND TRADE_STREAM_DISCONNECTED),
    clearing a false loss halt must ONLY discard DAILY_LOSS_LIMIT, leaving the other
    halt active and the engine halted.
    """
    tmp_path, acct_id = test_env
    from strategies.session import now_eastern
    session_date = str(now_eastern().date())

    halt_file = tmp_path / f"swing_daily_loss_halt_{acct_id}.json"
    halt_file.write_text(json.dumps({
        "account_id": acct_id,
        "session_date_et": session_date,
        "daily_pnl": "-310.00",
        "daily_loss_limit": "300.00",
        "baseline_equity": "100607.73",
        "halted": True,
        "reason": "DAILY_LOSS_LIMIT",
        "created_at_utc": "2026-10-15T08:00:00+00:00"
    }), encoding="utf-8")

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = acct_id
    bridge._risk_state_dir = str(tmp_path)
    bridge.entry_gate._test_halt_dir = tmp_path

    # Inject two halt reasons
    bridge.entry_gate.halt("DAILY_LOSS_LIMIT")
    bridge.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
    assert bridge.entry_gate.halted is True
    assert len(bridge.entry_gate.reasons) == 2

    mock_acct = MagicMock(
        account_number=acct_id,
        last_equity="99745.55",
        equity="99600.62"
    )

    bridge._reconcile_daily_loss_halt_if_needed(
        account_id=acct_id,
        et_date=session_date,
        acct=mock_acct,
        verified_last_eq=99745.55,
        prior_stale_baseline=100607.73
    )

    # False halt file is deleted
    assert not halt_file.exists()
    # DAILY_LOSS_LIMIT is discarded
    assert "DAILY_LOSS_LIMIT" not in bridge.entry_gate.reasons
    # TRADE_STREAM_DISCONNECTED is PRESERVED!
    assert "TRADE_STREAM_DISCONNECTED" in bridge.entry_gate.reasons
    assert bridge.entry_gate.halted is True
    assert bridge.entry_gate.halt_reason == "TRADE_STREAM_DISCONNECTED"
