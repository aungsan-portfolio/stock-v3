"""Offline tests for the read-only WMT reservation broker inspector."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from scripts import inspect_wmt_reservation as inspector


@pytest.fixture
def sample(tmp_path):
    first, second = str(uuid4()), str(uuid4())
    path = tmp_path / "order_idempotency_state.json"
    path.write_text(json.dumps({"WMT_BUY": {"key": "WMT_BUY", "symbol": "WMT", "side": "BUY",
        "parent_order_id": first, "created_at": 12345, "status": "ACKNOWLEDGED"}}), encoding="utf8")
    broker = MagicMock()
    broker.get_account.return_value = SimpleNamespace(account_number=inspector.EXPECTED_ACCOUNT)
    broker.get_all_positions.return_value = []
    broker.get_orders.return_value = []
    orders = {
        first: SimpleNamespace(id=first, symbol="WMT", side="buy", status="replaced", replaced_by=second, replaces=None, legs=[]),
        second: SimpleNamespace(id=second, symbol="WMT", side="buy", status="canceled", replaced_by=None, replaces=first, legs=[]),
    }
    broker.get_order_by_id.side_effect = lambda oid, *args: orders[oid]
    return SimpleNamespace(path=path, broker=broker, orders=orders, first=first, second=second)


def test_complete_read_only_history_follows_replacements_and_preserves_state(sample):
    original = sample.path.read_bytes()
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 0
    assert report["paper_account_verified"] is True
    assert report["positions"]["items"] == report["open_orders"]["items"] == []
    assert [o["lookup_id"] for o in report["order_lineage"]] == [sample.first, sample.second]
    assert report["reservation"]["parent_order_id"] == sample.first
    assert report["deletion_authorized"] is False
    assert sample.path.read_bytes() == original
    sample.broker.cancel_order_by_id.assert_not_called()
    sample.broker.submit_order.assert_not_called()
    sample.broker.close_position.assert_not_called()


def test_account_mismatch_never_queries_positions_or_orders(sample):
    sample.broker.get_account.return_value.account_number = "WRONG_ACCOUNT"
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and not report["paper_account_verified"]
    assert report["errors"][0]["step"] == "verify_account"
    sample.broker.get_all_positions.assert_not_called()
    sample.broker.get_orders.assert_not_called()
    sample.broker.get_order_by_id.assert_not_called()


@pytest.mark.parametrize("problem", ["missing", "bad_json", "missing_id", "partial_id", "wrong_side"])
def test_invalid_local_reservation_remains_unknown_and_never_queries_broker(sample, problem):
    if problem == "missing":
        sample.path.unlink()
    elif problem == "bad_json":
        sample.path.write_text("{", encoding="utf8")
    else:
        doc = json.loads(sample.path.read_text())
        if problem == "missing_id":
            doc["WMT_BUY"].pop("parent_order_id")
        elif problem == "partial_id":
            doc["WMT_BUY"]["parent_order_id"] = sample.first[:8]
        else:
            doc["WMT_BUY"]["side"] = "SELL"
        sample.path.write_text(json.dumps(doc), encoding="utf8")
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and report["reservation"]["status"] == "unknown"
    assert report["deletion_authorized"] is False
    sample.broker.get_account.assert_not_called()


@pytest.mark.parametrize("method", ["get_all_positions", "get_orders", "get_order_by_id"])
def test_broker_errors_are_unknown_and_exception_messages_are_not_leaked(sample, method):
    getattr(sample.broker, method).side_effect = RuntimeError("secret-header-example")
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and report["errors"] and not report["deletion_authorized"]
    assert "secret-header-example" not in json.dumps(report)


def test_full_open_order_page_is_not_a_complete_snapshot(sample):
    sample.broker.get_orders.return_value = [SimpleNamespace(id=str(uuid4()), symbol="WMT", status="new", legs=[]) for _ in range(500)]
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and report["open_orders"]["status"] == "possibly_truncated"


def test_malformed_open_order_or_child_is_unknown_not_verified(sample):
    sample.broker.get_orders.return_value = [SimpleNamespace(id=str(uuid4()), symbol="WMT", status="new",
        legs=[SimpleNamespace(id=None, symbol="WMT", status="new")])]
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and report["open_orders"]["status"] == "unknown"


@pytest.mark.parametrize("qty", [None, "nan", "0"])
def test_invalid_position_quantity_is_unknown_not_verified(sample, qty):
    sample.broker.get_all_positions.return_value = [SimpleNamespace(symbol="WMT", side="long", qty=qty)]
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and report["positions"]["status"] == "unknown"


@pytest.mark.parametrize("field,value", [("symbol", "AAPL"), ("side", "sell"), ("status", None)])
def test_mismatched_parent_order_never_approves_observation(sample, field, value):
    setattr(sample.orders[sample.first], field, value)
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and not report["deletion_authorized"]
    assert any(error["step"] == "order_lookup" for error in report["errors"])


def test_missing_linked_order_is_unknown(sample):
    del sample.orders[sample.second]
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and len(report["order_lineage"]) == 1
    assert any(error.get("order_id") == sample.second for error in report["errors"])


def test_traversal_limit_is_incomplete(sample, monkeypatch):
    monkeypatch.setattr(inspector, "MAX_LINKED_ORDERS", 1)
    report, code = inspector.inspect(sample.broker, sample.path)
    assert code == 2 and any(error["error_type"] == "order_limit_exceeded" for error in report["errors"])
