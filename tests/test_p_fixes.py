"""
tests/test_p_fixes.py — Dedicated Unit Tests for P1 to P4 Invariant Fixes

P1: Trailing stop state file partitioning by mode and account_id
P2: DAYTRADE_EARNINGS_BLACKOUT=False and SWING_EARNINGS_BLACKOUT=True
P3: Future-only earnings date check (0 <= days <= b_days) with fail-open warning logging
P4: Shadow cache quote staleness guard (age <= 2.0s)
"""
import os
import json
import time
from datetime import datetime, date, timedelta
from unittest.mock import MagicMock, patch

import pytest
import config
from alpaca_bridge import AlpacaBridge
from strategies.trailing_stop import (
    get_trailing_state_file,
    get_trailing_manager,
    DynamicTrailingStopManager,
)


def test_p1_state_file_partitioning():
    """Verify that Day and Swing managers use separate state files and do not collide."""
    day_file = get_trailing_state_file(mode="daytrade", account_id="PA3UODIC5JTA")
    swing_file = get_trailing_state_file(mode="swing", account_id="PA3R71I7JFDZ")

    assert "trailing_state_daytrade_PA3UODIC5JTA.json" in str(day_file)
    assert "trailing_state_swing_PA3R71I7JFDZ.json" in str(swing_file)
    assert day_file != swing_file

    day_mgr = get_trailing_manager(mode="daytrade", account_id="PA3UODIC5JTA")
    swing_mgr = get_trailing_manager(mode="swing", account_id="PA3R71I7JFDZ")

    assert day_mgr is not swing_mgr
    assert day_mgr.STATE_FILE != swing_mgr.STATE_FILE


def test_p1_swing_reconcile_does_not_purge_day_pending_exit(tmp_path):
    """Verify that Swing reconcile does not touch or purge Day bot pending exits."""
    day_state_file = tmp_path / "day_state.json"
    swing_state_file = tmp_path / "swing_state.json"

    day_mgr = DynamicTrailingStopManager(state_file=day_state_file)
    swing_mgr = DynamicTrailingStopManager(state_file=swing_state_file)

    # Queue a pending exit in Day Bot manager
    episode_id = "NVDA:day_ep_1"
    day_mgr.pending_exits[episode_id] = {
        "episode_id": episode_id,
        "symbol": "NVDA",
        "side": "BUY",
        "qty": 10.0,
        "entry_price": 120.0,
        "target_price": 122.0,
        "stop_price": 118.0,
        "entry_time": time.time(),
        "registered_at": time.time(),
        "attempts": 0,
    }
    day_mgr._save_state()
    assert day_mgr.has_pending_exit("NVDA") is True

    # Swing Bridge with no open positions
    mock_swing_bridge = MagicMock()
    mock_swing_bridge.positions_plain.return_value = []
    mock_swing_bridge.working_orders_plain.return_value = []
    mock_swing_bridge.get_positions.return_value = []
    mock_swing_bridge.get_position.return_value = 0.0
    mock_swing_bridge.get_trailing_stop_manager.return_value = swing_mgr

    # Swing reconcile runs on Swing manager
    swing_mgr.reconcile_pending_exits(mock_swing_bridge)

    # Day Bot pending exit must still exist completely untouched!
    assert day_mgr.has_pending_exit("NVDA") is True
    assert episode_id in day_mgr.pending_exits
    assert len(swing_mgr.pending_exits) == 0


def test_p2_mode_specific_earnings_blackout_flags():
    """Verify P2 flags in config: Day trade disabled, Swing enabled."""
    assert config.DAYTRADE_EARNINGS_BLACKOUT is False
    assert config.SWING_EARNINGS_BLACKOUT is True
    assert config.EARNINGS_BLACKOUT is True


def test_p2_and_p3_swing_blocks_future_earnings_and_allows_past_earnings():
    """Verify Swing execute_all blocks imminent future earnings, but allows past earnings."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge.get_open_positions = MagicMock(return_value=[])
    bridge.get_position = MagicMock(return_value=None)
    bridge.rotator = MagicMock()
    bridge.rotator.state.is_locked.return_value = False

    cal = {
        "META": "2026-10-29",  # today: blocked
        "V": "2026-10-27",     # 2 days ago: allowed
    }

    mock_signals = [
        MagicMock(symbol="META", action="BUY", confidence=0.8, price=500.0, target_weight=0.05, shadow_mode=False),
        MagicMock(symbol="V", action="BUY", confidence=0.8, price=280.0, target_weight=0.05, shadow_mode=False),
    ]

    executed_symbols = []
    def fake_execute(sig, **kwargs):
        executed_symbols.append(sig.symbol)
        return {"status": "SUBMITTED"}

    with patch("strategies.session.now_eastern") as mock_now, \
         patch.object(config, "EARNINGS_CALENDAR", cal), \
         patch.object(config, "SWING_EARNINGS_BLACKOUT", True), \
         patch.object(config, "EARNINGS_BLACKOUT_DAYS", 2), \
         patch.object(bridge, "execute_signal", side_effect=fake_execute), \
         patch.object(bridge, "positions_plain", return_value=[]), \
         patch.object(bridge, "working_orders_plain", return_value=[]):

        mock_now.return_value = datetime(2026, 10, 29, 10, 0, 0)
        res = bridge.execute_all(mock_signals)

        # META must be rejected with EARNINGS_BLACKOUT
        rejections = res.get("rejections", [])
        meta_rej = [r for r in rejections if r.get("symbol") == "META"]
        assert len(meta_rej) == 1
        assert meta_rej[0]["reason"] == "EARNINGS_BLACKOUT"

        # V (past earnings) must NOT be rejected by EARNINGS_BLACKOUT!
        v_rej = [r for r in rejections if r.get("symbol") == "V" and r.get("reason") == "EARNINGS_BLACKOUT"]
        assert len(v_rej) == 0
        assert "V" in executed_symbols


def test_p3_earnings_blackout_fail_open_on_error():
    """Verify that if earnings calendar has invalid data or raises an exception, it fails open (does not block)."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge.get_open_positions = MagicMock(return_value=[])
    bridge.get_position = MagicMock(return_value=None)
    bridge.rotator = MagicMock()
    bridge.rotator.state.is_locked.return_value = False

    # Bad date format
    cal = {"AAPL": "NOT_A_DATE"}
    mock_signals = [
        MagicMock(symbol="AAPL", action="BUY", confidence=0.8, price=200.0, target_weight=0.05, shadow_mode=False),
    ]

    executed_symbols = []
    def fake_execute(sig, **kwargs):
        executed_symbols.append(sig.symbol)
        return {"status": "SUBMITTED"}

    with patch.object(config, "EARNINGS_CALENDAR", cal), \
         patch.object(config, "SWING_EARNINGS_BLACKOUT", True), \
         patch.object(bridge, "execute_signal", side_effect=fake_execute), \
         patch.object(bridge, "positions_plain", return_value=[]), \
         patch.object(bridge, "working_orders_plain", return_value=[]):

        res = bridge.execute_all(mock_signals)
        # Should NOT be blocked by earnings blackout (fail-open)
        rejections = res.get("rejections", [])
        aapl_rej = [r for r in rejections if r.get("symbol") == "AAPL" and r.get("reason") == "EARNINGS_BLACKOUT"]
        assert len(aapl_rej) == 0
        assert "AAPL" in executed_symbols


def test_p4_shadow_cache_quote_staleness_guard():
    """Verify that shadow cache quote is used if age <= 2.0s, but rejected if age > 2.0s."""
    from strategies.orchestrator import _evaluate_mean_reversion
    import pandas as pd
    import numpy as np

    # Build dummy df
    dates = pd.date_range("2026-10-29 09:30", periods=50, freq="1min")
    df = pd.DataFrame({
        "open": np.linspace(100, 105, 50),
        "high": np.linspace(101, 106, 50),
        "low": np.linspace(99, 104, 50),
        "close": np.linspace(100, 105, 50),
        "volume": [1000] * 50,
    }, index=dates)

    mock_bridge = MagicMock()
    mock_bridge.get_quote.return_value = MagicMock(bid=104.9, ask=105.1, price=105.0)

    # 1. Fresh shadow quote (age = 0.5s <= 2.0s)
    fresh_shadow_q = MagicMock(bid=104.95, ask=105.05, price=105.0, timestamp=time.time() - 0.5)
    mock_mgr_fresh = MagicMock()
    mock_mgr_fresh.get_latest_quote.return_value = fresh_shadow_q

    with patch("strategies.shadow_stream_manager.get_shadow_stream_manager", return_value=mock_mgr_fresh), \
         patch.object(config, "STRATEGY_MEAN_REVERSION_ENABLED", True), \
         patch.dict(config.MARKET_REGIME_SETTINGS, {"SIDEWAYS": {"allowed_strategies": ["MEAN_REVERSION"]}}):
        sig = _evaluate_mean_reversion(
            symbol="TEST",
            df=df,
            market_texture="SIDEWAYS",
            scorer=MagicMock(),
            bridge=mock_bridge,
        )
        # Bridge REST get_quote should NOT be called because fresh stream quote was used
        assert mock_bridge.get_quote.call_count == 0

    # 2. Stale shadow quote (age = 5.0s > 2.0s)
    stale_shadow_q = MagicMock(bid=104.95, ask=105.05, price=105.0, timestamp=time.time() - 5.0)
    mock_mgr_stale = MagicMock()
    mock_mgr_stale.get_latest_quote.return_value = stale_shadow_q

    mock_bridge.get_quote.reset_mock()
    with patch("strategies.shadow_stream_manager.get_shadow_stream_manager", return_value=mock_mgr_stale), \
         patch.object(config, "STRATEGY_MEAN_REVERSION_ENABLED", True), \
         patch.dict(config.MARKET_REGIME_SETTINGS, {"SIDEWAYS": {"allowed_strategies": ["MEAN_REVERSION"]}}):
        sig = _evaluate_mean_reversion(
            symbol="TEST",
            df=df,
            market_texture="SIDEWAYS",
            scorer=MagicMock(),
            bridge=mock_bridge,
        )
        # Stale stream quote rejected; bridge REST get_quote MUST be called!
        assert mock_bridge.get_quote.call_count == 1
