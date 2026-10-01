"""tests/test_evaluate_paper_results_quarantine.py -- Verify test artifact IDs and commit quarantine in evaluate_paper_results."""

import json
import pytest
from pathlib import Path
from evaluate_paper_results import load_records, TEST_ARTIFACT_ORDER_IDS, QUARANTINED_COMMITS


def test_test_artifact_order_ids_and_commit_quarantine(tmp_path: Path):
    journal = tmp_path / "test_journal.jsonl"
    records = [
        # 1. Clean forward trade (should be included)
        {
            "event_type": "TRADE",
            "order_id": "clean-live-001",
            "client_order_id": "dt_vwap_001",
            "symbol": "AAPL",
            "git_commit": "v0.4.0",
            "pnl": 50.0,
            "exit_price": 150.0,
            "timestamp": "2026-10-02T14:00:00+00:00",
        },
        # 2. Test artifact: broker-parent-999 (should be excluded)
        {
            "event_type": "TRADE",
            "order_id": "broker-parent-999",
            "symbol": "AAPL",
            "pnl": 10.0,
            "exit_price": 101.0,
            "timestamp": "2026-09-25T17:02:43+00:00",
        },
        # 3. Test artifact: p_555 (should be excluded)
        {
            "event_type": "TRADE",
            "order_id": "ord_1",
            "parent_order_id": "p_555",
            "symbol": "MSFT",
            "pnl": -20.0,
            "exit_price": 300.0,
            "timestamp": "2026-09-26T17:00:00+00:00",
        },
        # 4. Test artifact: parent-ord-888 (should be excluded)
        {
            "event_type": "TRADE",
            "order_id": "ord_2",
            "client_order_id": "test_parent-ord-888_leg",
            "symbol": "GOOGL",
            "pnl": 15.0,
            "exit_price": 120.0,
            "timestamp": "2026-09-27T17:00:00+00:00",
        },
        # 5. Test artifact: parent-12345 (should be excluded)
        {
            "event_type": "TRADE",
            "order_id": "parent-12345",
            "symbol": "AMZN",
            "pnl": 5.0,
            "exit_price": 180.0,
            "timestamp": "2026-09-28T17:00:00+00:00",
        },
        # 6. Quarantined commit: 75d0234 (should be excluded)
        {
            "event_type": "TRADE",
            "order_id": "hist-001",
            "symbol": "LLY",
            "git_commit": "75d0234",
            "pnl": -0.35,
            "exit_price": 900.0,
            "timestamp": "2026-09-30T17:00:00+00:00",
        },
        # 7. Quarantined commit: 2d07950 (should be excluded)
        {
            "event_type": "TRADE",
            "order_id": "hist-002",
            "symbol": "AMD",
            "git_commit": "2d07950",
            "pnl": -2.64,
            "exit_price": 160.0,
            "timestamp": "2026-10-01T17:00:00+00:00",
        },
        # 8. Quarantined flag: quarantined_from_production_budget=True (should be excluded)
        {
            "event_type": "TRADE",
            "order_id": "hist-003",
            "symbol": "NVDA",
            "quarantined_from_production_budget": True,
            "pnl": -10.0,
            "exit_price": 120.0,
            "timestamp": "2026-10-01T18:00:00+00:00",
        },
        # 9. Clean forward trade 2 (should be included)
        {
            "event_type": "TRADE",
            "order_id": "clean-live-002",
            "client_order_id": "dt_mr_002",
            "symbol": "TSLA",
            "git_commit": "v0.4.0",
            "pnl": 45.0,
            "exit_price": 250.0,
            "timestamp": "2026-10-02T15:00:00+00:00",
        },
    ]

    with open(journal, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    loaded = load_records(journal)
    assert len(loaded) == 2
    assert [r["order_id"] for r in loaded] == ["clean-live-001", "clean-live-002"]
