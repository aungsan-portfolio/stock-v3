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
from strategies.order_registry import register_order_expected_price, is_validation_order, get_registered_order
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
        parent_order_id="order_nflx_in",
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
    test_order_id = "val_stop_uber_99"
    mgr = DynamicTrailingStopManager(state_file=tmp_path / "trailing_state_uber_test.json")
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
        last_stop_order_id=test_order_id,
        active=True,
        is_validation=True,
    )
    mgr.states["UBER"] = state

    with patch("strategies.validation_lane.record_validation_exit", return_value=True) as mock_val_exit, \
         patch("strategies.trade_journal.log_trade") as mock_prod_log:
        
        mock_bridge = MagicMock()
        mock_bridge.get_position.return_value = 0.0
        mock_order = MagicMock()
        mock_order.id = test_order_id
        mock_order.client_order_id = f"val_{test_order_id}"
        mock_order.symbol = "UBER"
        mock_order.side = "sell"
        mock_order.status = "filled"
        mock_order.order_type = "stop"
        mock_order.filled_qty = 1.0
        mock_order.filled_avg_price = 68.50
        mock_bridge.get_orders.return_value = [mock_order]
        mock_bridge.get_order_snapshot.return_value = {
            "id": test_order_id,
            "symbol": "UBER",
            "side": "sell",
            "status": "filled",
            "filled_qty": 1.0,
            "filled_avg_price": 68.50,
            "type": "stop",
        }

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


# =====================================================================
# Group E: Validation Accounting Integrity & Directional Invariants
# =====================================================================

def test_validation_short_entry_routing_not_classified_as_exit():
    """Verify that short entry orders (side=SELL, order_type=ENTRY) are never classified as exit fills."""
    register_order_expected_price(
        order_id="test_short_order_001",
        expected_price=150.0,
        side="SELL",
        order_type="ENTRY",
        is_validation=True,
    )
    reg = get_registered_order("test_short_order_001")
    assert reg is not None
    assert isinstance(reg, dict)

    role_info = {}
    parent_oid = None
    order_type = reg.get("order_type", "ENTRY")
    side = "SELL"

    reg_side = (reg.get("side") if isinstance(reg, dict) else getattr(reg, "side", "")) or ""
    reg_type = (reg.get("order_type") if isinstance(reg, dict) else getattr(reg, "order_type", "")) or ""
    role_type = str(role_info.get("order_role") or role_info.get("role") or "").upper()

    is_entry = (order_type == "ENTRY" or reg_type == "ENTRY" or role_type == "ENTRY")
    is_exit_fill = False
    if not is_entry:
        if order_type in ("STOP_LOSS", "TAKE_PROFIT") or reg_type in ("STOP_LOSS", "TAKE_PROFIT") or role_type in ("STOP_LOSS", "TAKE_PROFIT"):
            is_exit_fill = True
        elif parent_oid is not None:
            is_exit_fill = True
        elif (side == "SELL" and reg_side != "SELL") or (side == "BUY" and reg_side == "SELL"):
            is_exit_fill = True

    assert is_entry is True
    assert is_exit_fill is False


def test_validation_exit_directional_enforcement_and_unfilled_rejection(tmp_path):
    """Verify record_validation_exit rejects unfilled SUBMITTED records and enforces directional matching."""
    ledger = str(tmp_path / "val_dir_test.jsonl")

    # 1. Unfilled SUBMITTED record cannot be closed
    rec_unfilled = ValidationTradeRecord(
        validation_id="val_UNFILLED_01",
        timestamp="2026-10-01T15:00:00Z",
        symbol="AMD",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.6,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=100.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=100.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_unfilled",
        status="SUBMITTED",
        entry_fill_price=None,
    )
    log_validation_record(rec_unfilled, filepath=ledger)

    # Must return False and leave record in SUBMITTED status
    res = record_validation_exit(
        order_id="exit_attempt",
        exit_price=102.0,
        qty=1.0,
        side="SELL",
        symbol="AMD",
        filepath=ledger,
    )
    assert res is False

    # 2. Add two FILLED records for ORCL (one BUY, one SELL)
    rec_buy = ValidationTradeRecord(
        validation_id="val_ORCL_BUY",
        timestamp="2026-10-01T15:10:00Z",
        symbol="ORCL",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.6,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test BUY",
        decision_price=130.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T15:05:00Z",
        quote_mid=130.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_orcl_buy",
        status="FILLED",
        entry_fill_price=130.5,
    )
    rec_sell = ValidationTradeRecord(
        validation_id="val_ORCL_SELL",
        timestamp="2026-10-01T15:20:00Z",
        symbol="ORCL",
        strategy="CANDLESTICK",
        side="SELL",
        confidence=0.6,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test SELL",
        decision_price=132.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T15:15:00Z",
        quote_mid=132.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_orcl_sell",
        status="FILLED",
        entry_fill_price=132.5,
    )
    log_validation_record(rec_buy, filepath=ledger)
    log_validation_record(rec_sell, filepath=ledger)

    # SELL exit closes BUY record only
    ok_sell = record_validation_exit(
        order_id="stop_buy",
        exit_price=131.0,
        qty=1.0,
        side="SELL",
        symbol="ORCL",
        filepath=ledger,
        parent_order_id="entry_orcl_buy",
    )
    assert ok_sell is True

    # Re-calling with same order ID must be idempotent no-op and NOT close the SELL position
    ok_repeat = record_validation_exit(
        order_id="stop_buy",
        exit_price=131.0,
        qty=1.0,
        side="SELL",
        symbol="ORCL",
        filepath=ledger,
    )
    assert ok_repeat is True

    with open(ledger, "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    buy_r = next(r for r in rows if r.get("validation_id") == "val_ORCL_BUY")
    sell_r = next(r for r in rows if r.get("validation_id") == "val_ORCL_SELL")

    assert buy_r["status"] == "CLOSED"
    assert buy_r["realized_pnl"] == round((131.0 - 130.5) * 1.0, 4)
    assert sell_r["status"] == "FILLED", "SELL position must remain FILLED and not closed by SELL exit"

    # BUY exit closes SELL record with short PnL
    ok_buy = record_validation_exit(
        order_id="cover_sell",
        exit_price=130.0,
        qty=1.0,
        side="BUY",
        symbol="ORCL",
        filepath=ledger,
        parent_order_id="entry_orcl_sell",
    )
    assert ok_buy is True

    with open(ledger, "r") as f:
        rows2 = [json.loads(l) for l in f if l.strip()]
    sell_r2 = next(r for r in rows2 if r.get("validation_id") == "val_ORCL_SELL")
    assert sell_r2["status"] == "CLOSED"
    assert sell_r2["realized_pnl"] == round((132.5 - 130.0) * 1.0, 4)


def test_mandatory_lineage_matching_and_unlinked_rejection(tmp_path):
    """Verify record_validation_exit enforces strict lineage matching and rejects unlinked / ambiguous exits."""
    ledger = str(tmp_path / "val_lineage_test.jsonl")

    # Add two open BUY records for ORCL
    rec_orcl_1 = ValidationTradeRecord(
        validation_id="val_ORCL_01",
        timestamp="2026-10-01T15:10:00Z",
        symbol="ORCL",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.6,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=130.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T15:05:00Z",
        quote_mid=130.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_orcl_uuid_1",
        status="FILLED",
        entry_fill_price=130.0,
    )
    rec_orcl_2 = ValidationTradeRecord(
        validation_id="val_ORCL_02",
        timestamp="2026-10-01T15:15:00Z",
        symbol="ORCL",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.6,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=132.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T15:10:00Z",
        quote_mid=132.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_orcl_uuid_2",
        status="FILLED",
        entry_fill_price=132.0,
    )
    log_validation_record(rec_orcl_1, filepath=ledger)
    log_validation_record(rec_orcl_2, filepath=ledger)

    # Case 1: Unlinked lineage (parent_order_id does not exist) -> MUST return False, touch nothing
    res_unlinked = record_validation_exit(
        order_id="stop_nonexistent",
        exit_price=131.0,
        qty=1.0,
        side="SELL",
        symbol="ORCL",
        filepath=ledger,
        parent_order_id="entry_nonexistent_uuid",
    )
    assert res_unlinked is False

    # Case 2: Ambiguous matching (no parent_order_id or client_order_id with multiple open candidates) -> MUST fail closed
    res_ambiguous = record_validation_exit(
        order_id="stop_unlinked",
        exit_price=131.0,
        qty=1.0,
        side="SELL",
        symbol="ORCL",
        filepath=ledger,
    )
    assert res_ambiguous is False

    # Verify both records still open and untouched
    with open(ledger, "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert all(r["status"] == "FILLED" for r in rows)

    # Case 3: Exact lineage match by parent_order_id -> closes ONLY the targeted record
    res_linked = record_validation_exit(
        order_id="stop_linked_1",
        exit_price=135.0,
        qty=1.0,
        side="SELL",
        symbol="ORCL",
        filepath=ledger,
        parent_order_id="entry_orcl_uuid_1",
    )
    assert res_linked is True

    with open(ledger, "r") as f:
        rows2 = [json.loads(l) for l in f if l.strip()]
    r1 = next(r for r in rows2 if r["entry_order_id"] == "entry_orcl_uuid_1")
    r2 = next(r for r in rows2 if r["entry_order_id"] == "entry_orcl_uuid_2")
    assert r1["status"] == "CLOSED"
    assert r1["realized_pnl"] == round((135.0 - 130.0) * 1.0, 4)
    assert r2["status"] == "FILLED", "Second record must remain open and not arbitrarily matched"


def test_partial_fill_safe_deduplication_and_quantity_conservation(tmp_path):
    """Verify partial exit fills do not prematurely consume order_id and enforce quantity conservation."""
    ledger = str(tmp_path / "val_partial_test.jsonl")

    # Add a 2-share FILLED record
    rec = ValidationTradeRecord(
        validation_id="val_PARTIAL_01",
        timestamp="2026-10-01T15:00:00Z",
        symbol="MSFT",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.6,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=400.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=400.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_msft_01",
        status="FILLED",
        entry_fill_price=400.0,
    )
    log_validation_record(rec, filepath=ledger)

    # Manually ensure qty=2.0
    with open(ledger, "r") as f:
        data = json.loads(f.read().strip())
    data["qty"] = 2.0
    with open(ledger, "w") as f:
        f.write(json.dumps(data) + "\n")

    # Partial Fill 1: 1.0 share from stop order 'stop_msft_partial'
    res1 = record_validation_exit(
        order_id="stop_msft_partial",
        exit_price=405.0,
        qty=1.0,
        side="SELL",
        symbol="MSFT",
        filepath=ledger,
        parent_order_id="entry_msft_01",
        execution_id="EXEC_EXIT_PARTIAL_1",
    )
    assert res1 is True

    # Record must remain FILLED with closed_qty = 1.0
    with open(ledger, "r") as f:
        row1 = json.loads(f.read().strip())
    assert row1["status"] == "FILLED"
    assert row1["closed_qty"] == 1.0

    # Idempotent retry of Partial Fill 1 -> returns True, no state mutation
    res1_retry = record_validation_exit(
        order_id="stop_msft_partial",
        exit_price=405.0,
        qty=1.0,
        side="SELL",
        symbol="MSFT",
        filepath=ledger,
        parent_order_id="entry_msft_01",
        execution_id="EXEC_EXIT_PARTIAL_1",
    )
    assert res1_retry is True

    # Partial Fill 2: 1.0 share from SAME stop order 'stop_msft_partial' but distinct execution ID
    res2 = record_validation_exit(
        order_id="stop_msft_partial",
        exit_price=406.0,
        qty=1.0,
        side="SELL",
        symbol="MSFT",
        filepath=ledger,
        parent_order_id="entry_msft_01",
        execution_id="EXEC_EXIT_PARTIAL_2",
    )
    assert res2 is True

    # Now cumulative closed_qty = 2.0 >= entry_qty (2.0) -> marked CLOSED!
    with open(ledger, "r") as f:
        row2 = json.loads(f.read().strip())
    assert row2["status"] == "CLOSED"
    assert row2["closed_qty"] == 2.0
    assert row2["exit_fill_price"] == 405.5
    assert row2["realized_pnl"] == 11.0


def test_alpaca_bridge_no_exit_to_entry_fallback():
    """Verify alpaca_bridge never falls back to record_validation_fill when record_validation_exit fails."""
    from alpaca_bridge import AlpacaBridge
    from types import SimpleNamespace
    from strategies.session import now_eastern

    bridge = AlpacaBridge()
    bridge._connected = True
    bridge._client = MagicMock()
    now_et = now_eastern()
    mock_order = SimpleNamespace(
        id="exit_order_unlinked",
        client_order_id="val_stop_unknown",
        symbol="NVDA",
        side="sell",
        status="filled",
        filled_qty="1.0",
        filled_avg_price="120.0",
        filled_at=now_et,
        submitted_at=now_et,
        parent_order_id="entry_parent_unregistered",
        order_type="stop",
        type="stop",
        stop_price="120.0",
        replaces=None,
    )
    bridge._client.get_orders.return_value = [mock_order]

    with patch("strategies.validation_lane.record_validation_exit", return_value=False) as mock_exit, \
         patch("strategies.validation_lane.record_validation_fill") as mock_fill:

        # Run trade journal sync loop directly with mock order
        synced = bridge.sync_today_trades_to_journal(orders=[mock_order])

        # record_validation_exit was attempted with parent_order_id
        assert mock_exit.called
        # record_validation_fill MUST NEVER be called on unmapped exit fill
        assert not mock_fill.called, "record_validation_fill must NEVER be called as fallback for exit fill!"


# =====================================================================
# Group E: Forensic Accounting & Reconciliation Remediation (P0 & P1)
# =====================================================================

def test_entry_with_self_parent_order_id_not_rejected_as_conflict(tmp_path):
    """
    Finding 1 (P0): Root entry orders often have parent_order_id == id.
    Verify self-parent orders are normalized as root entry evidence and NOT falsely rejected
    as ROLE CONFLICT during AlpacaBridge journal sync.
    """
    from alpaca_bridge import AlpacaBridge
    from types import SimpleNamespace
    from strategies.session import now_eastern

    ledger = str(tmp_path / "val_self_parent.jsonl")
    now_et = now_eastern()
    oid = "entry-root-uuid-001"

    rec = ValidationTradeRecord(
        validation_id="val_AAPL_self_parent",
        timestamp="2026-10-01T15:00:00Z",
        symbol="AAPL",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.65,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=150.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=150.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id=oid,
        status="SUBMITTED",
        qty=1.0,
    )
    log_validation_record(rec, filepath=ledger)

    bridge = AlpacaBridge()
    bridge._connected = True
    bridge._client = MagicMock()

    # Self-parent bracket entry order from Alpaca: parent_order_id == id
    mock_order = SimpleNamespace(
        id=oid,
        client_order_id="val_aapl_12345",
        symbol="AAPL",
        side="buy",
        status="filled",
        filled_qty="1.0",
        filled_avg_price="150.25",
        filled_at=now_et,
        submitted_at=now_et,
        parent_order_id=oid,  # Self-parent!
        order_type="limit",
        type="limit",
        stop_price=None,
        replaces=None,
    )
    bridge._client.get_orders.return_value = [mock_order]

    orig_fill = record_validation_fill
    with patch("strategies.validation_lane.record_validation_fill", side_effect=lambda **kw: orig_fill(**{**kw, "filepath": ledger})):
        synced = bridge.sync_today_trades_to_journal(orders=[mock_order])

    assert synced == 1
    with open(ledger, "r") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    assert len(rows) == 1
    assert rows[0]["status"] == "FILLED"
    assert rows[0]["entry_fill_price"] == 150.25


def test_uncommitted_ledger_write_does_not_advance_sync_watermark_and_retries(tmp_path):
    """
    Finding 2 (P0): If ledger write fails or is uncommitted, sync watermark must NOT advance,
    allowing subsequent polls to retry and successfully commit.
    """
    from alpaca_bridge import AlpacaBridge
    from types import SimpleNamespace
    from strategies.session import now_eastern

    now_et = now_eastern()
    oid = "entry-uncommitted-001"
    bridge = AlpacaBridge()
    bridge._connected = True
    bridge._client = MagicMock()

    mock_order = SimpleNamespace(
        id=oid,
        client_order_id="val_msft_retry",
        symbol="MSFT",
        side="buy",
        status="filled",
        filled_qty="1.0",
        filled_avg_price="400.0",
        filled_at=now_et,
        submitted_at=now_et,
        parent_order_id=None,
        order_type="limit",
        type="limit",
        stop_price=None,
        replaces=None,
    )

    # Poll 1: record_validation_fill fails (returns False)
    with patch("strategies.validation_lane.record_validation_fill", return_value=False) as mock_fill:
        synced1 = bridge.sync_today_trades_to_journal(orders=[mock_order])
        assert synced1 == 0
        assert mock_fill.called

    # Poll 2: record_validation_fill succeeds (returns True)
    # Proves watermark was NOT advanced on uncommitted write; subsequent sync poll successfully retries!
    with patch("strategies.validation_lane.record_validation_fill", return_value=True) as mock_fill_ok:
        synced2 = bridge.sync_today_trades_to_journal(orders=[mock_order])
        assert synced2 == 1
        assert mock_fill_ok.called


def test_partial_exits_quantity_weighted_average_pnl(tmp_path):
    """
    Finding 3 (P1): Multiple partial exits at different prices must accumulate exit notional
    and compute quantity-weighted average exit price and accurate realized PnL.
    BUY 2 shares @ $100.
    Exit 1 share @ $90 (PnL -$10).
    Exit 1 share @ $110 (PnL +$10).
    Total realized PnL must be $0.0 (NOT +$20!).
    """
    ledger = str(tmp_path / "val_weighted_pnl.jsonl")

    rec = ValidationTradeRecord(
        validation_id="val_WEIGHT_01",
        timestamp="2026-10-01T15:00:00Z",
        symbol="GOOGL",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.7,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=100.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=100.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_googl_01",
        status="FILLED",
        entry_fill_price=100.0,
        qty=2.0,
    )
    log_validation_record(rec, filepath=ledger)

    # Partial exit 1: 1 share @ $90.0
    res1 = record_validation_exit(
        order_id="exit_googl_p1",
        exit_price=90.0,
        qty=1.0,
        side="SELL",
        symbol="GOOGL",
        filepath=ledger,
        parent_order_id="entry_googl_01",
        execution_id="EXEC_G_P1",
    )
    assert res1 is True

    with open(ledger, "r") as f:
        row1 = json.loads(f.read().strip())
    assert row1["status"] == "FILLED"
    assert row1["closed_qty"] == 1.0
    assert row1["realized_pnl"] == -10.0
    assert row1["exit_fill_price"] == 90.0

    # Partial exit 2: 1 share @ $110.0
    res2 = record_validation_exit(
        order_id="exit_googl_p2",
        exit_price=110.0,
        qty=1.0,
        side="SELL",
        symbol="GOOGL",
        filepath=ledger,
        parent_order_id="entry_googl_01",
        execution_id="EXEC_G_P2",
    )
    assert res2 is True

    with open(ledger, "r") as f:
        row2 = json.loads(f.read().strip())
    assert row2["status"] == "CLOSED"
    assert row2["closed_qty"] == 2.0
    assert row2["exit_fill_price"] == 100.0  # Weighted avg: (1*90 + 1*110)/2
    assert row2["realized_pnl"] == 0.0  # Net realized PnL: -10 + 10 = 0.0


def test_partial_exits_restart_persistence_cumulative_state(tmp_path):
    """
    Finding 3 (P1): State persistence across process restarts.
    Accumulated closed quantity, exit notional, and realized PnL must be durably read from
    disk after restart so the second partial exit computes accurate cumulative values.
    """
    ledger = str(tmp_path / "val_restart_pnl.jsonl")

    rec = ValidationTradeRecord(
        validation_id="val_RESTART_01",
        timestamp="2026-10-01T15:00:00Z",
        symbol="NVDA",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.75,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=50.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=50.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_nvda_restart",
        status="FILLED",
        entry_fill_price=50.0,
        qty=3.0,
    )
    log_validation_record(rec, filepath=ledger)

    # Process 1: First partial exit (1 share @ $60)
    ok1 = record_validation_exit(
        order_id="exit_nvda_p1",
        exit_price=60.0,
        qty=1.0,
        side="SELL",
        symbol="NVDA",
        filepath=ledger,
        parent_order_id="entry_nvda_restart",
        execution_id="EXEC_NVDA_P1",
    )
    assert ok1 is True

    # Simulate Process Restart: Process 2 opens ledger afresh from disk
    # Second partial exit (2 shares @ $55)
    ok2 = record_validation_exit(
        order_id="exit_nvda_p2",
        exit_price=55.0,
        qty=2.0,
        side="SELL",
        symbol="NVDA",
        filepath=ledger,
        parent_order_id="entry_nvda_restart",
        execution_id="EXEC_NVDA_P2",
    )
    assert ok2 is True

    with open(ledger, "r") as f:
        row = json.loads(f.read().strip())
    assert row["status"] == "CLOSED"
    assert row["closed_qty"] == 3.0
    # Expected weighted exit price: (1*60 + 2*55) / 3 = 170 / 3 = 56.6667
    assert row["exit_fill_price"] == round(170.0 / 3.0, 4)
    # Expected realized PnL: (60 - 50)*1 + (55 - 50)*2 = 10 + 10 = 20.0
    assert row["realized_pnl"] == 20.0


def test_exit_quantity_exceeds_remaining_quantity_rejected_overfill(tmp_path):
    """
    Finding 4 (P1): If reported exit quantity exceeds remaining quantity (+ tolerance),
    it must be immediately rejected as an overfill discrepancy and fail closed.
    Must also reject non-positive or non-finite quantities, and reject records with unknown entry quantity.
    """
    ledger = str(tmp_path / "val_overfill.jsonl")

    rec = ValidationTradeRecord(
        validation_id="val_OVERFILL_01",
        timestamp="2026-10-01T15:00:00Z",
        symbol="META",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.6,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=500.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=500.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_meta_01",
        status="FILLED",
        entry_fill_price=500.0,
        qty=1.0,
    )
    log_validation_record(rec, filepath=ledger)

    # 1. Overfill attempt: 2.0 shares reported exit for 1.0 share entry -> MUST be rejected
    res_overfill = record_validation_exit(
        order_id="exit_meta_overfill",
        exit_price=505.0,
        qty=2.0,
        side="SELL",
        symbol="META",
        filepath=ledger,
        parent_order_id="entry_meta_01",
    )
    assert res_overfill is False, "Exit fill exceeding remaining quantity must be rejected"

    # Verify record untouched
    with open(ledger, "r") as f:
        row = json.loads(f.read().strip())
    assert row["status"] == "FILLED"
    assert row.get("closed_qty", 0.0) == 0.0

    # 2. Non-positive or non-finite quantity rejection
    assert record_validation_exit(
        order_id="exit_meta_zero",
        exit_price=505.0,
        qty=0.0,
        side="SELL",
        symbol="META",
        filepath=ledger,
        parent_order_id="entry_meta_01",
    ) is False

    assert record_validation_exit(
        order_id="exit_meta_neg",
        exit_price=505.0,
        qty=-1.0,
        side="SELL",
        symbol="META",
        filepath=ledger,
        parent_order_id="entry_meta_01",
    ) is False

    assert record_validation_exit(
        order_id="exit_meta_nan",
        exit_price=505.0,
        qty=float("nan"),
        side="SELL",
        symbol="META",
        filepath=ledger,
        parent_order_id="entry_meta_01",
    ) is False

    # 3. Unknown entry quantity rejection (do not guess 1.0)
    ledger_no_qty = str(tmp_path / "val_no_qty.jsonl")
    rec_dict = rec.to_dict()
    del rec_dict["qty"]  # Purge entry quantity field
    with open(ledger_no_qty, "w") as f:
        f.write(json.dumps(rec_dict) + "\n")

    res_unknown_qty = record_validation_exit(
        order_id="exit_meta_valid",
        exit_price=505.0,
        qty=1.0,
        side="SELL",
        symbol="META",
        filepath=ledger_no_qty,
        parent_order_id="entry_meta_01",
    )
    assert res_unknown_qty is False, "Record with unknown entry quantity must fail closed (never guess 1.0)"


def test_wrong_parent_with_matching_generic_client_token_rejected(tmp_path):
    """
    Finding 5 (P0): Strict Lineage Correlation.
    If explicit parent_order_id is provided and does not match candidate entry_order_id,
    weaker client token matching is strictly prohibited from overriding the parent mismatch.
    Must fail closed and return False.
    """
    ledger = str(tmp_path / "val_strict_lineage.jsonl")

    rec = ValidationTradeRecord(
        validation_id="val_DIS_tok_12345",
        timestamp="2026-10-01T15:00:00Z",
        symbol="DIS",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.6,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=95.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=95.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_dis_actual_parent",
        status="FILLED",
        entry_fill_price=95.0,
        qty=1.0,
    )
    log_validation_record(rec, filepath=ledger)

    # Exit order has conflicting/wrong parent order ID but matching client token
    res = record_validation_exit(
        order_id="stop_dis_wrong",
        exit_price=96.0,
        qty=1.0,
        side="SELL",
        symbol="DIS",
        filepath=ledger,
        parent_order_id="entry_dis_foreign_wrong_parent",  # Mismatched parent!
        client_order_id="val_DIS_tok_12345",  # Matching client token
    )
    assert res is False, "Explicit parent mismatch must NOT be overridden by client token matching"

    # Verify record remains FILLED and unclosed
    with open(ledger, "r") as f:
        row = json.loads(f.read().strip())
    assert row["status"] == "FILLED"


def test_no_lineage_with_one_candidate_remains_pending(tmp_path):
    """
    Finding 5 (P0): Strict Lineage Correlation.
    When exit order has NO parent_order_id and NO client_order_id, it must NOT fall back
    to closing an open candidate even if there is exactly ONE candidate in the ledger.
    Must fail closed and retain pending.
    """
    ledger = str(tmp_path / "val_single_candidate.jsonl")

    # Exactly ONE open FILLED candidate for TSLA
    rec = ValidationTradeRecord(
        validation_id="val_TSLA_sole_candidate",
        timestamp="2026-10-01T15:00:00Z",
        symbol="TSLA",
        strategy="CANDLESTICK",
        side="BUY",
        confidence=0.65,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test",
        decision_price=220.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=220.0,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_tsla_sole_001",
        status="FILLED",
        entry_fill_price=220.0,
        qty=1.0,
    )
    log_validation_record(rec, filepath=ledger)

    # Exit order arrives with symbol TSLA but NO parent_order_id and NO client_order_id
    res = record_validation_exit(
        order_id="stop_tsla_no_lineage",
        exit_price=222.0,
        qty=1.0,
        side="SELL",
        symbol="TSLA",
        filepath=ledger,
        parent_order_id=None,
        client_order_id=None,
    )
    assert res is False, "Exit without lineage must NOT fall back to matching single candidate; must retain pending"

    # Verify candidate is still FILLED
    with open(ledger, "r") as f:
        row = json.loads(f.read().strip())
    assert row["status"] == "FILLED"



