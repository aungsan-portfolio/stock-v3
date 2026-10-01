"""tests/test_budget_quarantine.py -- Targeted tests for Lane 6 (Validation P&L isolation and budget quarantine)."""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock

from strategies.validation_lane import (
    ValidationTradeRecord,
    log_validation_record,
    get_validation_daily_pnl,
    get_production_risk_equity,
)


def test_validation_records_have_quarantine_marker():
    """Validation records must have quarantined_from_production_budget=True by default and in ledger."""
    rec = ValidationTradeRecord(
        validation_id="val_TEST_1",
        timestamp="2026-10-01T15:00:00Z",
        symbol="AAPL",
        strategy="MEAN_REVERSION",
        side="BUY",
        confidence=0.65,
        regime="RISK_ON",
        market_texture="SIDEWAYS",
        terminal_gate="STAGE_LUNCH_CHOP",
        rejection_reason="Test rejection",
        decision_price=100.0,
        decision_feed="iex",
        bar_timestamp="2026-10-01T14:55:00Z",
        quote_mid=100.0,
        quote_feed="iex",
        quote_timestamp=1000.0,
    )
    rec_dict = rec.to_dict()
    assert rec_dict.get("quarantined_from_production_budget") is True


def test_production_risk_budget_isolated_from_validation_losses():
    """Inject mock validation trades (-$10.00, -$50.00): production budget remains $300.00 (0 deduction)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = os.path.join(tmpdir, "test_validation.jsonl")

        # Create two closed validation trades on 2026-10-01: -$10.00 and -$50.00
        rec1 = {
            "validation_id": "val_1",
            "timestamp": "2026-10-01T14:00:00Z",
            "symbol": "AMD",
            "realized_pnl": -10.00,
            "status": "CLOSED",
            "quarantined_from_production_budget": True,
        }
        rec2 = {
            "validation_id": "val_2",
            "timestamp": "2026-10-01T15:00:00Z",
            "symbol": "AAPL",
            "realized_pnl": -50.00,
            "status": "CLOSED",
            "quarantined_from_production_budget": True,
        }
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(rec1) + "\n")
            f.write(json.dumps(rec2) + "\n")

        # Test daily validation P&L calculation
        val_pnl = get_validation_daily_pnl(session_date="2026-10-01", filepath=ledger_path)
        assert pytest.approx(val_pnl, 0.001) == -60.00

        # Mock broker: account started with 10,000, now at 9,940 due to -$60 validation trades
        bridge_mock = MagicMock()
        bridge_mock.get_net_liquidation.return_value = 9940.00
        bridge_mock.account_daily_pnl.return_value = -60.00

        # Production risk equity computation with validation isolated
        from unittest.mock import patch
        with patch("strategies.validation_lane.DEFAULT_VALIDATION_LEDGER_PATH", ledger_path), \
             patch("strategies.validation_lane.get_validation_daily_pnl", return_value=-60.00):

            # 1. With include_validation=False (Production default)
            gov_equity, prod_pnl = get_production_risk_equity(bridge_mock, include_validation=False)
            assert pytest.approx(gov_equity, 0.01) == 10000.00  # 9940 - (-60) = 10000
            assert pytest.approx(prod_pnl, 0.01) == 0.00        # -60 - (-60) = 0.00

            # Assert production daily loss budget ($300 limit) has zero deduction
            max_daily_loss = 300.00
            remaining_budget = max_daily_loss + prod_pnl
            assert pytest.approx(remaining_budget, 0.01) == 300.00

            # 2. With include_validation=True (Diagnostic / raw account mode)
            raw_equity, raw_pnl = get_production_risk_equity(bridge_mock, include_validation=True)
            assert pytest.approx(raw_equity, 0.01) == 9940.00
            assert pytest.approx(raw_pnl, 0.01) == -60.00
