"""Read-only order inspection regressions; no broker connection is made."""
from types import SimpleNamespace
from unittest.mock import MagicMock

from scripts import inspect_spy_order_history as audit


def _client():
    client = MagicMock()
    client.get_account.return_value = SimpleNamespace(account_number=audit.EXPECTED_ACCOUNT)
    client.get_all_positions.return_value = []
    client.get_orders.return_value = []
    client.get_order_by_id.side_effect = lambda oid, request: SimpleNamespace(
        id=oid, symbol="SPY", status="canceled", legs=[]
    )
    return client


def test_wrong_account_fails_before_any_broker_read_or_mutation():
    client = _client()
    client.get_account.return_value.account_number = "wrong-account"
    report, code = audit.inspect_orders(client)
    assert code == 2 and report["paper_account_verified"] is False
    client.get_all_positions.assert_not_called()
    client.get_orders.assert_not_called()
    client.get_order_by_id.assert_not_called()
    client.cancel_orders.assert_not_called()
    client.close_all_positions.assert_not_called()


def test_exact_account_allows_readonly_snapshots_and_nested_order_lookups():
    client = _client()
    report, code = audit.inspect_orders(client)
    assert code == 0 and report["paper_account_verified"] is True
    assert report["positions"]["status"] == "verified" and report["positions"]["items"] == []
    assert report["positions"]["observed_at_utc"].endswith("+00:00")
    assert report["open_orders"]["status"] == "verified" and report["open_orders"]["items"] == []
    assert report["open_orders"]["observed_at_utc"].endswith("+00:00")
    assert set(o["lookup_id"] for o in report["orders"]) == set(audit.ORDER_IDS.values())
    assert all(call.args[1].nested is True for call in client.get_order_by_id.call_args_list)
    client.cancel_orders.assert_not_called()
    client.close_all_positions.assert_not_called()


def test_failed_snapshot_is_unknown_not_flat_and_exception_text_is_not_printed():
    client = _client()
    client.get_all_positions.side_effect = TimeoutError("private-secret")
    client.get_orders.return_value = None
    report, code = audit.inspect_orders(client)
    assert code == 2
    assert report["positions"]["status"] == "unknown"
    assert report["open_orders"]["status"] == "unknown"
    assert "private-secret" not in str(report)
    assert {e["step"] for e in report["errors"]} == {"positions", "open_orders"}


def test_order_response_id_must_match_and_missing_order_is_error():
    client = _client()
    ids = list(audit.ORDER_IDS.values())
    client.get_order_by_id.side_effect = lambda oid, request: (
        SimpleNamespace(id="unexpected", symbol="SPY") if oid == ids[0]
        else None if oid == ids[1] else SimpleNamespace(id=oid, symbol="SPY", legs=[])
    )
    report, code = audit.inspect_orders(client)
    assert code == 2
    assert {e["order_id"] for e in report["errors"]} == set(ids[:2])


def test_replacement_lineage_is_followed_once_without_order_writes():
    client = _client()
    root_id = audit.ORDER_IDS["unknown_order"]
    replacement = "11111111-1111-4111-8111-111111111111"

    def lookup(oid, request):
        return SimpleNamespace(id=oid, symbol="SPY", legs=[],
                               replaced_by=replacement if oid == root_id else None,
                               replaces=root_id if oid == replacement else None)

    client.get_order_by_id.side_effect = lookup
    report, code = audit.inspect_orders(client)
    assert code == 0
    assert sum(o["lookup_id"] == replacement for o in report["orders"]) == 1
    client.replace_order_by_id.assert_not_called()
    client.cancel_order_by_id.assert_not_called()


def test_credentials_use_only_complete_daytrade_pairs_without_generic_fallback(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("DAYTRADE_APCA_API_KEY_ID=file-key\nDAYTRADE_APCA_API_SECRET_KEY=file-secret\n")
    key, secret = audit.load_credentials({"DAYTRADE_APCA_API_KEY_ID": "partial-env", "APCA_API_SECRET_KEY": "generic"}, env_file)
    assert (key, secret) == ("file-key", "file-secret")


def test_more_than_or_equal_to_limit_open_orders_is_not_verified():
    client = _client()
    client.get_orders.return_value = [SimpleNamespace(legs=[]) for _ in range(500)]
    report, code = audit.inspect_orders(client)
    assert code == 2 and report["open_orders"]["status"] == "possibly_truncated"
