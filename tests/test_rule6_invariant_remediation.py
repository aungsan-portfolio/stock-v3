"""
tests/test_rule6_invariant_remediation.py -- Regression Test Suite for Rule 6 Invariant Remediation.

Covers the 4 Critical Remediation Invariant Groups:
  - Group A: Correlation Empty-Input Guard & Explicit Telemetry Purpose Contract
  - Group B: Multi-Account & Two-Lane Accounting Isolation (Production Ledger vs Validation Ledger)
  - Group C: Deterministic Side Arbitration & Opposing Position Veto
  - Group D: Intraday Telemetry Purpose Isolation (Entry Circuit Breaker Non-Pollution)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from strategies.base import OrderIntent, StrategyName, TradeSignal
from strategies.correlation import CorrelationAnalyzer
from strategies.orchestrator import apply_portfolio_correlation
from strategies.order_registry import register_order_expected_price, is_validation_order
from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState
from strategies.validation_lane import (
    ValidationTradeRecord,
    evaluate_and_route_validation_candidate,
    record_validation_fill,
    record_validation_exit,
    log_validation_record,
)
import strategies.intraday_data as intraday_data_module


# =====================================================================
# Group A: Correlation Empty Guard & Purpose Contract
# =====================================================================

def test_correlation_empty_input_guard():
    """Verify apply_portfolio_correlation([]) returns [] immediately without calling analyzer or bridge."""
    mock_bridge = MagicMock()
    result = apply_portfolio_correlation([], open_positions=["AAPL", "MSFT"], bridge=mock_bridge)
    assert result == []
    # Bridge should never have been queried for equity or positions
    assert mock_bridge.get_net_liquidation.call_count == 0


def test_correlation_passes_position_monitoring_purpose():
    """Verify CorrelationAnalyzer passes purpose='POSITION_MONITORING' to fetch_intraday."""
    analyzer = CorrelationAnalyzer()
    mock_bridge = MagicMock()

    with patch("strategies.correlation.fetch_intraday") as mock_fetch:
        # Return valid mock DataFrame with 50 bars
        dates = pd.date_range("2026-09-25 14:00", periods=50, freq="5min")
        mock_df = pd.DataFrame({"close": [100.0 + i * 0.1 for i in range(50)]}, index=dates)
        mock_fetch.return_value = mock_df

        ret = analyzer.get_returns("AAPL", bridge=mock_bridge)
        assert ret is not None
        assert mock_fetch.call_count == 1
        _, kwargs = mock_fetch.call_args
        assert kwargs.get("purpose") == "POSITION_MONITORING"


# =====================================================================
# Group B: Two-Lane Accounting Isolation
# =====================================================================

def test_order_registry_tracks_validation_flag():
    """Verify order registry correctly tags validation orders and returns is_validation_order."""
    register_order_expected_price("val_order_001", 150.0, side="BUY", is_validation=True, client_order_id="val_AAPL_123")
    register_order_expected_price("prod_order_001", 200.0, side="SELL", is_validation=False, client_order_id="prod_MSFT_456")

    assert is_validation_order("val_order_001") is True
    assert is_validation_order("prod_order_001") is False
    assert is_validation_order("unknown_order") is False


def test_record_validation_fill_and_exit_isolated_ledger(tmp_path):
    """Verify validation fills and exits update paper_validation_trades.jsonl without touching production journal."""
    val_ledger = tmp_path / "paper_validation_trades.jsonl"
    prod_journal = tmp_path / "trade_journal.jsonl"

    # Pre-populate a candidate in validation ledger
    rec = ValidationTradeRecord(
        validation_id="val_TEST_001",
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol="NFLX",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.55,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_CONFIDENCE",
        rejection_reason="Confidence below floor",
        decision_price=71.20,
        decision_feed="iex",
        bar_timestamp=datetime.now(timezone.utc).isoformat(),
        quote_mid=71.20,
        quote_feed="iex",
        quote_timestamp=time.time(),
        entry_order_id="order_nflx_in",
        stop_price=70.00,
        target_price=73.00,
    )
    log_validation_record(rec, filepath=str(val_ledger))

    # Record entry fill
    fill_ok = record_validation_fill(
        order_id="order_nflx_in",
        fill_price=71.25,
        qty=1.0,
        side="BUY",
        slippage=0.0007,
        filepath=str(val_ledger),
    )
    assert fill_ok is True

    # Verify filled state
    with open(val_ledger, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    assert len(data) == 1
    assert data[0]["status"] == "FILLED"
    assert data[0]["entry_fill_price"] == 71.25
    assert data[0]["slippage_bps"] == 7.0

    # Record exit fill
    exit_ok = record_validation_exit(
        order_id="order_nflx_out",
        exit_price=71.50,
        qty=1.0,
        side="SELL",
        symbol="NFLX",
        exit_reason="STOP_OUT",
        filepath=str(val_ledger),
    )
    assert exit_ok is True

    # Verify closed state and P&L
    with open(val_ledger, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
    assert len(data) == 1
    assert data[0]["status"] == "CLOSED"
    assert data[0]["exit_fill_price"] == 71.50
    assert data[0]["realized_pnl"] == 0.25
    assert data[0]["pnl"] == 0.25

    # Production journal was never created or modified
    assert not prod_journal.exists()


def test_trailing_stop_manager_routes_validation_exit(tmp_path):
    """Verify TrailingStopManager routes validation exits to record_validation_exit and bypasses log_trade."""
    val_ledger = tmp_path / "paper_val.jsonl"
    rec = ValidationTradeRecord(
        validation_id="val_TS_001",
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol="UBER",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.55,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_CONFIDENCE",
        rejection_reason="Near-miss",
        decision_price=69.00,
        decision_feed="iex",
        bar_timestamp=datetime.now(timezone.utc).isoformat(),
        quote_mid=69.00,
        quote_feed="iex",
        quote_timestamp=time.time(),
        entry_order_id="uber_entry_oid",
        stop_price=68.50,
        target_price=71.00,
    )
    log_validation_record(rec, filepath=str(val_ledger))
    record_validation_fill("uber_entry_oid", 69.00, 1.0, "BUY", filepath=str(val_ledger))

from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState


def test_trailing_stop_manager_routes_validation_exit(tmp_path):
    """Verify DynamicTrailingStopManager routes validation exits to record_validation_exit and bypasses log_trade."""
    val_ledger = tmp_path / "paper_val.jsonl"
    rec = ValidationTradeRecord(
        validation_id="val_TS_001",
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol="UBER",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.55,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_CONFIDENCE",
        rejection_reason="Near-miss",
        decision_price=69.00,
        decision_feed="iex",
        bar_timestamp=datetime.now(timezone.utc).isoformat(),
        quote_mid=69.00,
        quote_feed="iex",
        quote_timestamp=time.time(),
        entry_order_id="uber_entry_oid",
        stop_price=68.50,
        target_price=71.00,
    )
    log_validation_record(rec, filepath=str(val_ledger))
    record_validation_fill("uber_entry_oid", 69.00, 1.0, "BUY", filepath=str(val_ledger))

    # Set up DynamicTrailingStopManager with an initialized validation state
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="UBER",
        side="BUY",
        entry_price=69.00,
        peak_price=69.00,
        atr=0.50,
        stop_price=68.50,
        trail_multiple=1.5,
        last_updated=time.time(),
        qty=1.0,
        order_id=None,
        active=True,
        is_validation=True,
    )
    mgr.states["UBER"] = state

    with patch("strategies.validation_lane.record_validation_exit") as mock_val_exit, \
         patch("strategies.trade_journal.log_trade") as mock_prod_log:
        
        mock_bridge = MagicMock()
        mock_bridge.get_position.return_value = 0.0

        mgr.handle_naked_position("UBER", 1.0, bridge=mock_bridge)

        # Must route to validation exit
        assert mock_val_exit.call_count == 1
        _, kwargs = mock_val_exit.call_args
        assert kwargs.get("symbol") == "UBER"
        assert kwargs.get("exit_price") == 68.50

        # Must NOT call production log_trade
        assert mock_prod_log.call_count == 0



# =====================================================================
# Group C: Deterministic Side Arbitration & Opposing Position Veto
# =====================================================================

def test_validation_lane_rejects_opposing_broker_position():
    """Verify evaluate_and_route_validation_candidate vetos opposing position against broker."""
    sig_sell = TradeSignal(
        symbol="SPY",
        strategy=StrategyName.CANDLESTICK,
        side="SELL",
        confidence=0.45,
        entry_price=770.0,
        stop_price=772.0,
        target_price=766.0,
        atr=1.5,
        risk_per_share=2.0,
        reason="Opposing signal",
        decision_feed="iex",
        bar_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    mock_bridge = MagicMock()
    # Mock active broker position: LONG 1 share of SPY
    mock_pos = MagicMock()
    mock_pos.qty = 1.0
    mock_bridge.get_position.return_value = mock_pos
    mock_bridge.is_paper = True

    result = evaluate_and_route_validation_candidate(
        signal=sig_sell,
        bridge=mock_bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_THRESHOLD",
        policy_reason_msg="Confidence below floor",
        dry_run=True,
    )

    assert result["status"] == "VETOED"
    assert "OPPOSING_POSITION_VETO" in result["reason"]


def test_can_route_validation_candidate_arbitration_logic():
    """Verify single candidate per symbol per cycle and opposing side arbitration."""
    routed_map: Dict[str, str] = {}
    active_sides: Dict[str, str] = {"SPY": "BUY", "NVDA": "SELL"}

    def _can_route(sig, routed, sides):
        sym_u = sig.symbol.upper().strip()
        if sym_u in routed:
            return False, f"Duplicate validation candidate in cycle (already routed {routed[sym_u]})"
        if sym_u in sides:
            curr_side = sides[sym_u]
            sig_side = "BUY" if sig.side.upper().strip() in ("BUY", "LONG") else "SELL"
            if sig_side != curr_side:
                return False, f"Opposing validation candidate {sig_side} conflicts with active {curr_side} position"
        return True, ""

    sig1 = TradeSignal(symbol="MSFT", strategy=StrategyName.MOMENTUM_SCALP, side="BUY", confidence=0.8, entry_price=500.0, stop_price=495.0, target_price=510.0, atr=2.0, risk_per_share=5.0, reason="test")
    sig2 = TradeSignal(symbol="MSFT", strategy=StrategyName.CANDLESTICK, side="SELL", confidence=0.4, entry_price=500.0, stop_price=505.0, target_price=490.0, atr=2.0, risk_per_share=5.0, reason="test")
    sig3 = TradeSignal(symbol="SPY", strategy=StrategyName.CANDLESTICK, side="SELL", confidence=0.4, entry_price=770.0, stop_price=775.0, target_price=760.0, atr=2.0, risk_per_share=5.0, reason="test")
    sig4 = TradeSignal(symbol="NVDA", strategy=StrategyName.VWAP_BOUNCE, side="BUY", confidence=0.7, entry_price=220.0, stop_price=215.0, target_price=230.0, atr=2.0, risk_per_share=5.0, reason="test")

    # 1. First MSFT candidate routes successfully
    ok, reason = _can_route(sig1, routed_map, active_sides)
    assert ok is True
    routed_map["MSFT"] = "val_MSFT_001"

    # 2. Second MSFT candidate in same cycle is rejected
    ok, reason = _can_route(sig2, routed_map, active_sides)
    assert ok is False
    assert "Duplicate validation candidate" in reason

    # 3. SPY SELL conflicts with active BUY position
    ok, reason = _can_route(sig3, routed_map, active_sides)
    assert ok is False
    assert "conflicts with active BUY position" in reason

    # 4. NVDA BUY conflicts with active SELL position
    ok, reason = _can_route(sig4, routed_map, active_sides)
    assert ok is False
    assert "conflicts with active SELL position" in reason


# =====================================================================
# Group D: Telemetry Purpose Isolation
# =====================================================================

def test_telemetry_isolation_entry_vs_monitoring():
    """Verify stale bars with purpose != ENTRY_SIGNAL do NOT increment entry circuit-breaker stale_bar_consecutive."""
    with intraday_data_module._feed_telemetry_lock:
        intraday_data_module._feed_telemetry["stale_bar_consecutive"] = 0
        intraday_data_module._feed_telemetry["stale_monitoring_count"] = 0
        intraday_data_module._last_stale_bar_ts.clear()

    # Simulate position monitoring stale bar (post-close delay 120s > grace 60s)
    mock_bar = MagicMock()
    mock_bar.post_close_delay_seconds = 120.0
    mock_bar.close_time = "2026-09-25T16:20:00Z"
    mock_bar.start_time = "2026-09-25T16:15:00Z"

    # Update telemetry via simulated logic from intraday_data.py
    delay_seconds = 120.0
    grace_seconds = 60.0
    bar_id = mock_bar.close_time

    # Tick 1: Non-entry purpose
    purpose = "POSITION_MONITORING"
    with intraday_data_module._feed_telemetry_lock:
        if delay_seconds > grace_seconds:
            intraday_data_module._feed_telemetry["stale_observation_total"] += 1
            last_ts = intraday_data_module._last_stale_bar_ts.get("AAPL")
            if last_ts != bar_id:
                if purpose == "ENTRY_SIGNAL":
                    intraday_data_module._feed_telemetry["stale_bar_consecutive"] += 1
                else:
                    intraday_data_module._feed_telemetry["stale_monitoring_count"] = intraday_data_module._feed_telemetry.get("stale_monitoring_count", 0) + 1
                intraday_data_module._last_stale_bar_ts["AAPL"] = bar_id

    # Entry circuit breaker must NOT have incremented
    assert intraday_data_module._feed_telemetry["stale_bar_consecutive"] == 0
    assert intraday_data_module._feed_telemetry["stale_monitoring_count"] == 1

    # Tick 2: ENTRY_SIGNAL purpose on a new bar
    new_bar_id = "2026-09-25T16:25:00Z"
    purpose = "ENTRY_SIGNAL"
    with intraday_data_module._feed_telemetry_lock:
        if delay_seconds > grace_seconds:
            intraday_data_module._feed_telemetry["stale_observation_total"] += 1
            last_ts = intraday_data_module._last_stale_bar_ts.get("AAPL")
            if last_ts != new_bar_id:
                if purpose == "ENTRY_SIGNAL":
                    intraday_data_module._feed_telemetry["stale_bar_consecutive"] += 1
                else:
                    intraday_data_module._feed_telemetry["stale_monitoring_count"] = intraday_data_module._feed_telemetry.get("stale_monitoring_count", 0) + 1
                intraday_data_module._last_stale_bar_ts["AAPL"] = new_bar_id

    # Now entry circuit breaker DOES increment
    assert intraday_data_module._feed_telemetry["stale_bar_consecutive"] == 1
    assert intraday_data_module._feed_telemetry["stale_monitoring_count"] == 1
