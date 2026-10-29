"""
tests/test_c_strategy_flags.py — Verification of Phase C Strategy Feature Flags C1–C7.
"""
import pytest
from datetime import datetime, date, time as dt_time, timedelta
from unittest.mock import MagicMock, patch
import pytz
import pandas as pd

import config
from strategies.event_guard import evaluate_event
from strategies.mean_reversion import MeanReversionStrategy
from strategies.swing_sizing import SECTOR_EXPOSURE_CAP_PCT


def test_c1_earnings_blackout_flag_and_window():
    """Verify C1 EARNINGS_BLACKOUT blocks entries within 2 days of earnings."""
    assert config.EARNINGS_BLACKOUT is True
    assert config.EARNINGS_BLACKOUT_DAYS == 2
    assert config.EARNINGS_EXIT_MODE == "hold"
    assert "META" in config.EARNINGS_CALENDAR


def test_c2_macro_event_blackout_fomc_window():
    """Verify C2 MACRO_EVENT_BLACKOUT blocks entries during scheduled FOMC blackout (13:45–15:00 ET)."""
    assert config.MACRO_EVENT_BLACKOUT is True
    et_tz = pytz.timezone("America/New_York")

    # Time during FOMC blackout on 2026-10-28 at 14:00 ET
    fomc_time = datetime(2026, 10, 28, 14, 0, 0, tzinfo=et_tz)
    decision = evaluate_event(fomc_time)
    assert decision.block_entries is True
    assert "FOMC_RATE_DECISION" in decision.tag

    # Time outside blackout on 2026-10-28 at 10:30 ET
    normal_time = datetime(2026, 10, 28, 10, 30, 0, tzinfo=et_tz)
    decision_normal = evaluate_event(normal_time)
    assert decision_normal.block_entries is False


def test_c3_mr_stop_atr_mult_and_breakeven_parameters():
    """Verify C3 MR_STOP_ATR_MULT is 1.25 and breakeven threshold is 1.5 ATR with 0.1 ATR lock."""
    assert config.MR_STOP_ATR_MULT == 1.25
    assert config.MR_BREAKEVEN_PROFIT_ATR == 1.5
    assert config.MR_BREAKEVEN_LOCK_ATR == 0.1


def test_c4_mr_rs_filter_parameters():
    """Verify C4 MR_RS_FILTER is True and threshold is -1.0%."""
    assert config.MR_RS_FILTER is True
    assert config.MR_RS_FILTER_THRESHOLD == -1.0


def test_c5_sector_trend_override_parameters():
    """Verify C5 SECTOR_TREND_OVERRIDE is True with SOXX threshold 1.5% and breadth 60.0%."""
    assert config.SECTOR_TREND_OVERRIDE is True
    assert config.SOXX_TREND_THRESHOLD == 1.5
    assert config.SOXX_BREADTH_THRESHOLD == 60.0


def test_c6_signal_hysteresis_in_swing_bridge():
    """Verify C6 SIGNAL_HYSTERESIS requires 2 consecutive cycles for SELL exit."""
    assert config.SIGNAL_HYSTERESIS is True
    assert config.SIGNAL_HYSTERESIS_CYCLES == 2

    from alpaca_bridge import AlpacaBridge
    bridge = AlpacaBridge(mode="swing", dry_run=True)

    sig_sell = MagicMock()
    sig_sell.symbol = "AAPL"
    sig_sell.action = "SELL"
    sig_sell.confidence = 0.80
    sig_sell.shadow_mode = False

    with patch.object(bridge, "_evaluate_and_execute_rotation", return_value=set()):
        with patch.object(bridge, "get_open_positions", return_value={"AAPL": MagicMock()}):
            with patch.object(bridge, "working_order_symbols", return_value=set()):
                with patch.object(bridge, "get_position", return_value=10):
                    with patch.object(bridge, "_cancel_working_entry_orders"):
                        # Cycle 1: first SELL signal observed -> skipped, awaiting confirmation
                        res1 = bridge.execute_all([sig_sell])
                        assert res1["skipped"] == 1
                        assert bridge._sell_signal_cycles.get("AAPL") == 1

                        # Cycle 2: second consecutive SELL signal -> executed
                        with patch.object(bridge, "execute_signal", return_value=True) as mock_exec:
                            res2 = bridge.execute_all([sig_sell])
                            assert mock_exec.call_count == 1
                            assert res2["placed"] == 1


def test_c7_cluster_notional_cap():
    """Verify C7 CLUSTER_NOTIONAL_CAP enforces 15% equity per sector."""
    assert config.CLUSTER_NOTIONAL_CAP == 0.15
    assert SECTOR_EXPOSURE_CAP_PCT == 0.15
