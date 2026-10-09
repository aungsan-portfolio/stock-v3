import json
import pytest
from pathlib import Path
from strategies.trade_journal import log_trade, log_fill

def test_log_trade_price_tier_derivation_semantics(tmp_path):
    """Verify that log_trade derives price_tier with exact schema parity to log_fill."""
    journal_file = tmp_path / "test_journal.jsonl"

    # 1. Tier: >10
    log_trade(
        symbol="AAPL",
        side="BUY",
        strategy="ORB",
        qty=10,
        entry_price=150.0,
        stop_price=145.0,
        target_price=160.0,
        journal_file=journal_file,
    )

    # 2. Tier: 5-10
    log_trade(
        symbol="F",
        side="BUY",
        strategy="ORB",
        qty=100,
        entry_price=9.50,
        stop_price=9.00,
        target_price=10.50,
        journal_file=journal_file,
    )

    # 3. Tier: <5
    log_trade(
        symbol="PENNY",
        side="BUY",
        strategy="ORB",
        qty=500,
        entry_price=3.25,
        stop_price=3.00,
        target_price=3.75,
        journal_file=journal_file,
    )

    # 4. Explicit override in kwargs
    log_trade(
        symbol="CUSTOM",
        side="BUY",
        strategy="ORB",
        qty=10,
        entry_price=25.0,
        stop_price=24.0,
        target_price=27.0,
        price_tier="TIER_OVERRIDE",
        journal_file=journal_file,
    )

    # Read records and assert price_tier semantics
    lines = journal_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4

    rec1 = json.loads(lines[0])
    assert rec1["symbol"] == "AAPL"
    assert rec1["price_tier"] == ">10"

    rec2 = json.loads(lines[1])
    assert rec2["symbol"] == "F"
    assert rec2["price_tier"] == "5-10"

    rec3 = json.loads(lines[2])
    assert rec3["symbol"] == "PENNY"
    assert rec3["price_tier"] == "<5"

    rec4 = json.loads(lines[3])
    assert rec4["symbol"] == "CUSTOM"
    assert rec4["price_tier"] == "TIER_OVERRIDE"
