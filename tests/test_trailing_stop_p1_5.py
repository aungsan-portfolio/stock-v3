"""
tests/test_trailing_stop_p1_5.py — Unit & Invariant Tests for Trailing Stop P1.5 Production Hardening.
"""
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from strategies.base import TradeSignal
from strategies.trailing_stop import (
    DynamicTrailingStopManager,
    TrailingStopState,
    check_time_based_exit,
    get_trailing_manager,
)


@pytest.fixture
def clean_manager(monkeypatch):
    """Fixture providing an isolated DynamicTrailingStopManager with temp state file."""
    with tempfile.TemporaryDirectory() as td:
        state_file = Path(td) / "trailing_state.json"
        monkeypatch.setattr("strategies.trailing_stop.STATE_FILE", str(state_file))
        mgr = DynamicTrailingStopManager()
        mgr.states = {}
        yield mgr, str(state_file)


class TestAtomicStatePersistence:
    def test_atomic_save_creates_valid_json(self, clean_manager):
        mgr, state_file = clean_manager
        state = TrailingStopState(
            symbol="NVDA",
            side="BUY",
            entry_price=100.0,
            peak_price=105.0,
            atr=2.0,
            stop_price=97.0,
            trail_multiple=1.5,
            last_updated=time.time(),
            qty=10.0,
        )
        mgr.states["NVDA"] = state
        mgr._save_state()

        assert os.path.exists(state_file)
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "NVDA" in data
        assert data["NVDA"]["entry_price"] == 100.0

    def test_tmp_file_cleaned_on_exception(self, clean_manager, monkeypatch):
        mgr, state_file = clean_manager
        # Force json.dump to raise exception
        monkeypatch.setattr(json, "dump", MagicMock(side_effect=RuntimeError("Disk full simulation")))
        mgr._save_state()
        # No orphaned .tmp files should exist in dir
        tmp_files = list(Path(state_file).parent.glob("*.tmp*"))
        assert len(tmp_files) == 0


class TestResetAndLifecycle:
    def test_soft_reset_preserves_telemetry(self, clean_manager):
        mgr, _ = clean_manager
        state = TrailingStopState(
            symbol="MSFT",
            side="BUY",
            entry_price=400.0,
            peak_price=410.0,
            atr=5.0,
            stop_price=392.5,
            trail_multiple=1.5,
            last_updated=time.time(),
            active=True,
        )
        mgr.states["MSFT"] = state
        mgr.soft_reset("MSFT")

        assert "MSFT" in mgr.states
        assert mgr.states["MSFT"].active is False

    def test_hard_reset_deletes_state_completely(self, clean_manager):
        mgr, _ = clean_manager
        state = TrailingStopState(
            symbol="AMZN",
            side="BUY",
            entry_price=200.0,
            peak_price=205.0,
            atr=3.0,
            stop_price=195.5,
            trail_multiple=1.5,
            last_updated=time.time(),
            active=True,
        )
        mgr.states["AMZN"] = state
        mgr.hard_reset("AMZN")

        assert "AMZN" not in mgr.states

    def test_reset_alias_calls_hard_reset(self, clean_manager):
        mgr, _ = clean_manager
        state = TrailingStopState(
            symbol="QQQ",
            side="BUY",
            entry_price=500.0,
            peak_price=505.0,
            atr=4.0,
            stop_price=494.0,
            trail_multiple=1.5,
            last_updated=time.time(),
            active=True,
        )
        mgr.states["QQQ"] = state
        mgr.reset("QQQ")

        assert "QQQ" not in mgr.states

    def test_inactive_reentry_creates_fresh_lifecycle(self, clean_manager):
        """CRITICAL: Re-entering a soft-reset inactive symbol must NOT stay inactive!"""
        mgr, _ = clean_manager
        old_state = TrailingStopState(
            symbol="AAPL",
            side="BUY",
            entry_price=150.0,
            peak_price=160.0,
            atr=2.0,
            stop_price=147.0,
            trail_multiple=1.5,
            last_updated=time.time() - 3600,
            active=False,  # Inactive from prior closed trade
        )
        mgr.states["AAPL"] = old_state

        from strategies.constants import StrategyName
        sig = TradeSignal(
            symbol="AAPL",
            strategy=StrategyName.ORB,
            side="BUY",
            confidence=0.8,
            entry_price=200.0,
            stop_price=194.75,
            target_price=210.0,
            atr=3.5,
            risk_per_share=5.25,
            reason="Test breakout",
        )

        new_state = mgr.apply_confirmed_fill(
            signal=sig,
            average_fill_price=200.0,
            cumulative_qty=5.0,
            stop_order_id="new-stop-id",
        )

        assert new_state.active is True
        assert new_state.entry_price == 200.0
        assert new_state.peak_price == 200.0
        assert new_state.atr == 3.5
        assert new_state.order_id == "new-stop-id"
        assert new_state.remediation_in_progress is False
        assert new_state.last_remediation_time == 0.0
        assert new_state.remediation_attempts == 0
        assert new_state.escalation_alerted is False
        assert new_state.entry_time > 0.0

    def test_ensure_initialized_recovers_persisted_entry_time(self, clean_manager):
        """CRITICAL: ensure_initialized must recover original entry_time across bot restarts!"""
        mgr, _ = clean_manager
        orig_time = time.time() - 900.0  # 15 minutes ago
        saved_state = TrailingStopState(
            symbol="TSLA",
            side="BUY",
            entry_price=250.0,
            peak_price=252.0,
            atr=4.0,
            stop_price=244.0,
            trail_multiple=1.5,
            last_updated=time.time(),
            entry_time=orig_time,
            active=False,
        )
        mgr.states["TSLA"] = saved_state

        # Re-initialize on restart
        reconciled = mgr.ensure_initialized(
            symbol="TSLA",
            side="BUY",
            avg_cost=250.0,
            open_orders=[],
            current_price=251.0,
            qty=10.0,
        )
        assert reconciled is not None
        assert reconciled.entry_time == orig_time  # Clock preserved, not reset!


class TestDynamicATRRefresh:
    def test_update_stop_smooths_and_bounds_atr(self, clean_manager):
        mgr, _ = clean_manager
        state = TrailingStopState(
            symbol="AAPL",
            side="BUY",
            entry_price=200.0,
            peak_price=200.0,
            atr=2.0,  # Old ATR
            stop_price=197.0,
            trail_multiple=1.5,
            last_updated=0.0,
            active=True,
        )
        mgr.states["AAPL"] = state

        # Raw ATR leaps from 2.0 to 5.0 (+150%, way above max_change_pct=25%)
        # Bounded ATR = 2.0 * 1.25 = 2.50
        # Refreshed ATR with alpha=0.25: 2.0 * 0.75 + 2.50 * 0.25 = 1.50 + 0.625 = 2.125
        mgr.update_stop("AAPL", current_price=200.5, current_atr=5.0, dry_run=True)
        assert math.isclose(mgr.states["AAPL"].atr, 2.125, rel_tol=1e-4)

    def test_atr_persistence_when_stop_does_not_move(self, clean_manager):
        mgr, state_file = clean_manager
        state = TrailingStopState(
            symbol="MSFT",
            side="BUY",
            entry_price=400.0,
            peak_price=400.0,
            atr=4.0,
            stop_price=394.0,
            trail_multiple=1.5,
            last_updated=0.0,
            active=True,
        )
        mgr.states["MSFT"] = state
        mgr._save_state()

        # Update ATR when price has not moved enough to update stop
        mgr.update_stop("MSFT", current_price=400.0, current_atr=4.8, dry_run=True)

        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["MSFT"]["atr"] > 4.0


class TestDynamicCooldown:
    def test_high_volatility_symbol_uses_15s_cooldown(self, clean_manager):
        mgr, _ = clean_manager
        # NVDA has atr_pct = 3.0 / 120.0 = 2.5% >= 0.8% threshold
        state = TrailingStopState(
            symbol="NVDA",
            side="BUY",
            entry_price=120.0,
            peak_price=120.0,
            atr=3.0,
            stop_price=115.5,
            trail_multiple=1.5,
            last_updated=time.time() - 20.0,  # 20s ago (>15s but <30s)
            active=True,
        )
        mgr.states["NVDA"] = state

        res = mgr.update_stop("NVDA", current_price=124.0, dry_run=True)
        assert res is not None
        assert res.peak_price == 124.0

    def test_low_volatility_symbol_respects_30s_cooldown(self, clean_manager):
        mgr, _ = clean_manager
        # JNJ has atr_pct = 0.50 / 160.0 = 0.31% < 0.8% threshold
        state = TrailingStopState(
            symbol="JNJ",
            side="BUY",
            entry_price=160.0,
            peak_price=160.0,
            atr=0.50,
            stop_price=159.0,
            trail_multiple=1.5,
            last_updated=time.time() - 20.0,  # 20s ago (<30s base cooldown)
            active=True,
        )
        mgr.states["JNJ"] = state

        res = mgr.update_stop("JNJ", current_price=161.0, dry_run=True)
        assert res is not None
        assert res.peak_price == 160.0  # Cooldown blocked update


class TestTimeBasedExit:
    def test_stagnant_trade_held_over_25m_triggers_exit(self):
        now = time.time()
        entry_time = now - (26 * 60)
        triggered, reason = check_time_based_exit(
            symbol="QQQ",
            side="BUY",
            entry_price=500.0,
            current_price=500.20,  # +0.04% (stagnant near breakeven)
            entry_timestamp=entry_time,
            peak_price=500.30,
        )
        assert triggered is True
        assert "stagnant near breakeven" in reason

    def test_giveback_trade_held_over_25m_triggers_exit(self):
        now = time.time()
        entry_time = now - (30 * 60)
        # Entry 500, peaked at 503.0 (+0.60% MFE >= 0.30%), current 500.10 (+0.02% near breakeven)
        # Giveback = (503 - 500.10) / 503 = 0.57% >= 25% of peak gain
        triggered, reason = check_time_based_exit(
            symbol="AMZN",
            side="BUY",
            entry_price=500.0,
            current_price=500.10,
            entry_timestamp=entry_time,
            peak_price=503.0,
        )
        assert triggered is True
        assert "gave back peak profits" in reason

    def test_trade_holding_gains_does_not_trigger(self):
        now = time.time()
        entry_time = now - (30 * 60)
        triggered, reason = check_time_based_exit(
            symbol="QQQ",
            side="BUY",
            entry_price=500.0,
            current_price=504.00,  # +0.80% (holding gain outside band)
            entry_timestamp=entry_time,
            peak_price=504.50,
        )
        assert triggered is False
        assert "let run" in reason
