"""
tests/test_validation_fill_replay_and_partial_fills.py -- Unit tests for Lane 1:
Validation fill replay prevention, partial-fill safety, composite dedupe key,
dedupe-on-read projection, and process exclusivity.
"""
import os
import json
import time
from decimal import Decimal
from datetime import datetime, timezone
import pytest

from strategies.validation_lane import (
    build_fill_composite_key,
    record_validation_fill,
    record_validation_exit,
    dedupe_on_read,
    load_validation_fill_states,
    load_seen_validation_order_ids,
    ValidationTradeRecord,
    log_validation_record,
)
from strategies.startup_security import (
    acquire_atomic_process_lock,
    release_atomic_process_lock,
    ProcessLockMetadata,
    StartupSafetyRejected,
)


def test_validation_fill_composite_key():
    """Verify deterministic composite key generation."""
    k1 = build_fill_composite_key("ord_123", "BUY", 100.50, 1.0, "2026-10-01T15:00:00Z")
    k2 = build_fill_composite_key("ord_123", "buy", 100.50, 1, "2026-10-01T15:00:00Z")
    assert k1 == k2
    assert "ord_123" in k1
    assert "BUY" in k1
    assert "100.5000" in k1

    # Primary execution_id precedence
    k_exec = build_fill_composite_key("ord_123", "BUY", 100.50, 1.0, execution_id="EXEC_VENUE_999")
    assert k_exec == "EXEC_VENUE_999"


def test_validation_fill_idempotent_replay(tmp_path):
    """Verify that replaying the exact same fill payload does not duplicate fills or corrupt state."""
    ledger = str(tmp_path / "val_trades.jsonl")

    # Initial candidate record
    rec = ValidationTradeRecord(
        validation_id="val_AMD_1790864442",
        timestamp="2026-10-01T14:20:42.734337+00:00",
        symbol="AMD",
        strategy="MEAN_REVERSION",
        side="BUY",
        confidence=0.5824,
        regime="RISK_ON",
        market_texture="SIDEWAYS",
        terminal_gate="STAGE_CONFIDENCE",
        rejection_reason="Confidence 0.5824 < 0.6000",
        decision_price=604.43,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:15:00+00:00",
        quote_mid=604.225,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="d6291fe4-6986-4c0b-a71e-5d7036056497",
        status="SUBMITTED",
    )
    log_validation_record(rec, filepath=ledger)

    # First fill
    fill_ok_1 = record_validation_fill(
        order_id="d6291fe4-6986-4c0b-a71e-5d7036056497",
        fill_price=604.36,
        qty=1.0,
        side="BUY",
        timestamp="2026-10-01T14:21:00Z",
        slippage=-0.0001,
        symbol="AMD",
        execution_id="EXEC_AMD_BUY_001",
        filepath=ledger,
    )
    assert fill_ok_1 is True

    # Replay identical fill (simulating subsequent poll cycle)
    fill_ok_2 = record_validation_fill(
        order_id="d6291fe4-6986-4c0b-a71e-5d7036056497",
        fill_price=604.36,
        qty=1.0,
        side="BUY",
        timestamp="2026-10-01T14:21:00Z",
        slippage=-0.0001,
        symbol="AMD",
        execution_id="EXEC_AMD_BUY_001",
        filepath=ledger,
    )
    # Must be handled cleanly as an idempotent no-op
    assert fill_ok_2 is True

    # Verify ledger only has 1 record with status FILLED and qty 1.0
    trades = dedupe_on_read(ledger)
    assert len(trades) == 1
    assert trades[0]["status"] == "FILLED"
    assert trades[0]["entry_fill_price"] == 604.36
    assert trades[0]["qty"] == 1.0


def test_validation_fill_partial_fills_safe(tmp_path):
    """Verify partial fills from broker venue are correctly aggregated without dropping trades."""
    ledger = str(tmp_path / "val_partial.jsonl")

    rec = ValidationTradeRecord(
        validation_id="val_UBER_123",
        timestamp="2026-10-01T15:00:00Z",
        symbol="UBER",
        strategy="MOMENTUM_SCALP",
        side="BUY",
        confidence=0.59,
        regime="RISK_ON",
        market_texture="TRENDING",
        terminal_gate="STAGE_CONFIDENCE",
        rejection_reason="Confidence below floor",
        decision_price=80.00,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=80.00,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="ord_uber_001",
        status="SUBMITTED",
    )
    log_validation_record(rec, filepath=ledger)

    # Partial fill 1: 0.5 shares @ $80.00
    ok1 = record_validation_fill(
        order_id="ord_uber_001",
        fill_price=80.00,
        qty=0.5,
        side="BUY",
        timestamp="2026-10-01T15:00:10Z",
        symbol="UBER",
        execution_id="EXEC_UBER_P1",
        filepath=ledger,
    )
    assert ok1 is True

    # Partial fill 2: 0.5 shares @ $80.10
    ok2 = record_validation_fill(
        order_id="ord_uber_001",
        fill_price=80.10,
        qty=0.5,
        side="BUY",
        timestamp="2026-10-01T15:00:20Z",
        symbol="UBER",
        execution_id="EXEC_UBER_P2",
        filepath=ledger,
    )
    assert ok2 is True

    trades = dedupe_on_read(ledger)
    assert len(trades) == 1
    # Total qty must aggregate to 1.0, weighted avg entry price must be 80.05
    assert trades[0]["qty"] == 1.0
    assert abs(trades[0]["entry_fill_price"] - 80.05) < 1e-4


def test_load_validation_fill_states_reconciler_idempotency(tmp_path):
    """Verify that load_validation_fill_states provides accurate fill state to Alpaca reconciler."""
    ledger = str(tmp_path / "val_recon.jsonl")

    # Ingest completed AMD trade
    rec = ValidationTradeRecord(
        validation_id="val_AMD_001",
        timestamp="2026-10-01T14:20:42Z",
        symbol="AMD",
        strategy="MEAN_REVERSION",
        side="BUY",
        confidence=0.58,
        regime="RISK_ON",
        market_texture="SIDEWAYS",
        terminal_gate="STAGE_CONFIDENCE",
        rejection_reason="Test",
        decision_price=604.43,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:15:00Z",
        quote_mid=604.225,
        quote_feed="iex",
        quote_timestamp=1790864438.0,
        entry_order_id="entry_amd_123",
        status="SUBMITTED",
    )
    log_validation_record(rec, filepath=ledger)

    record_validation_fill(
        order_id="entry_amd_123",
        fill_price=604.36,
        qty=1.0,
        side="BUY",
        symbol="AMD",
        timestamp="2026-10-01T14:21:00Z",
        execution_id="EXEC_AMD_IN",
        filepath=ledger,
    )

    record_validation_exit(
        order_id="exit_amd_456",
        exit_price=601.72,
        qty=1.0,
        side="SELL",
        symbol="AMD",
        timestamp="2026-10-01T14:23:00Z",
        exit_reason="STOP_LOSS",
        execution_id="EXEC_AMD_OUT",
        parent_order_id="entry_amd_123",
        filepath=ledger,
    )

    # Load states
    states = load_validation_fill_states(account_id="PA3UODIC5JTA", filepath=ledger)
    assert len(states) == 2

    # Entry order fill state check
    entry_key = ("PA3UODIC5JTA", "2026-10-01", "entry_amd_123", "AMD", "BUY")
    assert entry_key in states
    assert states[entry_key].logged_qty == Decimal("1")
    assert states[entry_key].logged_notional == Decimal("604.36")

    # Exit order fill state check
    exit_key = ("PA3UODIC5JTA", "2026-10-01", "exit_amd_456", "AMD", "SELL")
    assert exit_key in states
    assert states[exit_key].logged_qty == Decimal("1")
    assert states[exit_key].logged_notional == Decimal("601.72")

    # Reconciler idempotency test: broker reports cum_qty = 1 for entry_amd_123
    broker_cum_qty = Decimal("1")
    already_logged = states[entry_key].logged_qty
    delta_qty = broker_cum_qty - already_logged
    assert delta_qty == Decimal("0")  # Idempotent no-op satisfied!


def test_load_seen_validation_order_ids(tmp_path):
    """Verify startup order ID extractor for fast lookups."""
    ledger = str(tmp_path / "seen.jsonl")
    with open(ledger, "w") as f:
        f.write(json.dumps({"entry_order_id": "oid_1", "exit_order_id": "oid_2"}) + "\n")
        f.write(json.dumps({"entry_order_id": "oid_3"}) + "\n")

    seen = load_seen_validation_order_ids(ledger)
    assert seen == {"oid_1", "oid_2", "oid_3"}
