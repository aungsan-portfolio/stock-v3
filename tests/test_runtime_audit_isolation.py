"""
tests/test_runtime_audit_isolation.py — Verification of Step 1 Test/Runtime Audit Isolation.

Verifies:
1. Dynamic config path resolution redirects DATA_DIR, LOG_DIR, and REPORTS_DIR to the isolated test root.
2. order_audit.log_event appends exclusively to isolated logs/order_audit.jsonl.
3. EntryIdempotencyStore defaults to isolated data/order_idempotency_state.json.
4. DynamicTrailingStopManager defaults to isolated data/trailing_state.json.
5. log_gate_decision appends exclusively to isolated data/trade_gate_decision.jsonl.
6. log_trade appends exclusively to isolated reports/trade_journal.jsonl.
7. log_validation_record appends exclusively to isolated data/paper_validation_trades.jsonl.
8. risk_state.record_trade writes exclusively to isolated logs/daily_risk_state.json.
9. Guard tripwire fails closed with AssertionError when write operations target production directories.
"""
from pathlib import Path
import pytest

import config
import order_audit
from strategies.order_idempotency import EntryIdempotencyStore, OrderReservation
from strategies.trailing_stop import DynamicTrailingStopManager
from strategies.gate_ledger import log_gate_decision
from strategies.trade_journal import log_trade
from strategies.validation_lane import ValidationTradeRecord, log_validation_record
import risk_state


def test_isolated_config_paths(test_isolation_root):
    """Verify that all core mutable directories point inside the isolated test root."""
    assert test_isolation_root == Path(config.DATA_DIR).parent
    assert Path(config.DATA_DIR).is_relative_to(test_isolation_root)
    assert Path(config.LOG_DIR).is_relative_to(test_isolation_root)
    assert Path(config.REPORTS_DIR).is_relative_to(test_isolation_root)

    repo_root = Path(__file__).resolve().parent.parent
    assert Path(config.DATA_DIR) != repo_root / "data"
    assert Path(config.LOG_DIR) != repo_root / "logs"
    assert Path(config.REPORTS_DIR) != repo_root / "reports"


def test_order_audit_isolation(test_isolation_root):
    """Verify order_audit writes exclusively to the test root."""
    order_audit.log_event(
        "test_isolation_stage",
        symbol="ISOLATION_TEST",
        side="BUY",
        order_id="test_ord_123",
    )
    isolated_audit = test_isolation_root / "logs" / "order_audit.jsonl"
    assert isolated_audit.exists()
    content = isolated_audit.read_text(encoding="utf-8")
    assert "ISOLATION_TEST" in content


def test_order_idempotency_isolation(test_isolation_root):
    """Verify EntryIdempotencyStore defaults to the test root."""
    store = EntryIdempotencyStore()
    assert Path(store.storage_path).is_relative_to(test_isolation_root)
    assert store.storage_path == test_isolation_root / "data" / "order_idempotency_state.json"


def test_trailing_stop_isolation(test_isolation_root):
    """Verify DynamicTrailingStopManager defaults to the test root."""
    mgr = DynamicTrailingStopManager()
    assert Path(mgr.STATE_FILE).is_relative_to(test_isolation_root)
    assert Path(mgr.STATE_FILE) == test_isolation_root / "data" / "trailing_state.json"


def test_gate_ledger_isolation(test_isolation_root):
    """Verify log_gate_decision writes exclusively to the test root."""
    log_gate_decision(
        evaluation_id="eval_iso_test_001",
        symbol="ISO_TEST",
        bar="5m",
        stage="STAGE_CONFIDENCE",
        decision="PASS",
        human_message="Test gate isolation",
    )
    isolated_gate = test_isolation_root / "data" / "trade_gate_decision.jsonl"
    assert isolated_gate.exists()
    content = isolated_gate.read_text(encoding="utf-8")
    assert "eval_iso_test_001" in content


def test_trade_journal_isolation(test_isolation_root):
    """Verify log_trade writes exclusively to the test root."""
    log_trade(
        symbol="ISO_JRN",
        side="BUY",
        strategy="ORB",
        qty=1,
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        reason="test_journal_isolation",
    )
    isolated_journal = test_isolation_root / "reports" / "trade_journal.jsonl"
    assert isolated_journal.exists()
    content = isolated_journal.read_text(encoding="utf-8")
    assert "ISO_JRN" in content


def test_validation_lane_isolation(test_isolation_root):
    """Verify log_validation_record writes exclusively to the test root."""
    rec = ValidationTradeRecord(
        validation_id="val_iso_001",
        timestamp="2026-10-16T00:00:00Z",
        symbol="ISO_VAL",
        strategy="MOMENTUM_SCALP",
        side="BUY",
        confidence=0.65,
        regime="BULLISH",
        market_texture="TRENDING",
        terminal_gate="STAGE_CONFIDENCE",
        rejection_reason="CONFIDENCE_BELOW_FLOOR",
        decision_price=50.0,
        decision_feed="alpaca_sip",
        bar_timestamp="2026-10-16T00:00:00Z",
        quote_mid=50.0,
        quote_feed="alpaca_sip",
        quote_timestamp=1700000000.0,
        stop_price=48.0,
        target_price=54.0,
        hard_safety_pass=True,
        policy_rejected=True,
    )
    log_validation_record(rec)
    isolated_val = test_isolation_root / "data" / "paper_validation_trades.jsonl"
    assert isolated_val.exists()
    content = isolated_val.read_text(encoding="utf-8")
    assert "val_iso_001" in content


def test_risk_state_isolation(test_isolation_root):
    """Verify risk_state writes exclusively to the test root."""
    count = risk_state.record_trade("2099-01-01")
    assert count >= 1
    isolated_risk = test_isolation_root / "logs" / "daily_risk_state.json"
    assert isolated_risk.exists()
    content = isolated_risk.read_text(encoding="utf-8")
    assert "2099-01-01" in content


def test_write_guard_fails_closed_on_production_directory():
    """Verify write-protection tripwire immediately raises AssertionError on production target."""
    repo_root = Path(__file__).resolve().parent.parent
    prod_test_file = repo_root / "logs" / "illegal_test_write.tmp"

    with pytest.raises(AssertionError, match="TEST ISOLATION VIOLATION"):
        with open(prod_test_file, "w") as f:
            f.write("should_fail")

    with pytest.raises(AssertionError, match="TEST ISOLATION VIOLATION"):
        prod_test_file.write_text("should_fail")
