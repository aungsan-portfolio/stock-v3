"""Inspect a specific SPY paper-trading order lifecycle without changing it.

Run from the project root: python scripts/inspect_spy_order_history.py

Only DAYTRADE_APCA_API_KEY_ID and DAYTRADE_APCA_API_SECRET_KEY are used.
The script checks the paper account number before looking up any orders.
Output is a JSON document of broker-reported order fields and errors; it is
not proof of continuous protection or of the reason an order was canceled.
Credentials are never included in the report. Exit 2 means the account could
not be verified or at least one order could not be inspected.
"""
from __future__ import annotations

from collections import deque
from datetime import date, datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrderByIdRequest, GetOrdersRequest
from requests.adapters import HTTPAdapter


EXPECTED_ACCOUNT = "PA3UODIC5JTA"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORDER_IDS = {
    "entry": "5336ec5f-6dce-4dd6-b551-5b0599dae0a9",
    "original_stop": "250205a0-8b8f-4afa-bad4-d3ec85426657",
    "original_take_profit": "20010164-d466-4932-b39e-cd2078dbf960",
    "unknown_order": "a11cad38-7b71-4a01-b573-1bfa50d3d226",
    "closing_order": "96247c30-505b-4898-a100-a1ab8ad74696",
}
FIELDS = (
    "id", "client_order_id", "symbol", "side", "order_type", "type",
    "order_class", "status", "qty", "filled_qty", "filled_avg_price",
    "limit_price", "stop_price", "time_in_force", "created_at", "submitted_at",
    "filled_at", "canceled_at", "expired_at", "failed_at", "replaced_at",
    "replaced_by", "replaces", "parent_order_id",
)
MAX_ORDERS = 50


class BoundedTimeoutAdapter(HTTPAdapter):
    """Apply finite connect/read timeouts even when the SDK passes None."""

    def send(self, request: Any, **kwargs: Any) -> Any:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (5.0, 10.0)
        return super().send(request, **kwargs)


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _order_fields(order: Any) -> dict[str, Any]:
    result = {name: _text(_get(order, name)) for name in FIELDS}
    result["legs"] = [
        {name: _text(_get(leg, name)) for name in FIELDS}
        for leg in (_get(order, "legs") or [])
    ]
    return result


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse only the two allowed keys, without printing or returning other values."""
    keys = {"DAYTRADE_APCA_API_KEY_ID", "DAYTRADE_APCA_API_SECRET_KEY"}
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if name in keys:
            values[name] = value.strip().strip("\"").strip("'")
    return values


def load_credentials(env: Any = None, env_file: Path | None = None) -> tuple[str, str]:
    """Use a complete daytrade pair from one source; never mix sources."""
    env = os.environ if env is None else env
    key_name = "DAYTRADE_APCA_API_KEY_ID"
    secret_name = "DAYTRADE_APCA_API_SECRET_KEY"
    key, secret = env.get(key_name), env.get(secret_name)
    if key and secret and "YOUR_" not in key and "YOUR_" not in secret:
        return key, secret
    values = _parse_env_file(env_file or PROJECT_ROOT / ".env")
    key, secret = values.get(key_name), values.get(secret_name)
    if not key or not secret or "YOUR_" in key or "YOUR_" in secret:
        raise ValueError("daytrade credential pair is unavailable")
    return key, secret


def inspect_orders(client: Any) -> tuple[dict[str, Any], int]:
    """Verify account first, then perform only read-only, nested order lookups."""
    report: dict[str, Any] = {
        "paper_account_verified": False,
        "expected_account_suffix": EXPECTED_ACCOUNT[-4:],
        "requested_orders": ORDER_IDS.copy(),
        "positions": {"status": "unknown", "items": []},
        "open_orders": {"status": "unknown", "items": []},
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "orders": [],
        "errors": [],
        "note": "Order events alone do not establish continuous protection or cancel cause.",
    }
    try:
        account = client.get_account()
    except Exception as exc:
        # SDK error messages may contain headers or other sensitive data.
        report["errors"].append({"step": "verify_account", "error_type": type(exc).__name__})
        return report, 2
    if str(_get(account, "account_number") or "").strip() != EXPECTED_ACCOUNT:
        report["errors"].append({"step": "verify_account", "error_type": "account_mismatch_or_missing"})
        return report, 2
    report["paper_account_verified"] = True

    # A successful empty result proves the queried snapshot is empty. A failed
    # or truncated query remains unknown rather than being labeled flat.
    try:
        positions = client.get_all_positions()
        if not isinstance(positions, (list, tuple)):
            raise ValueError("missing positions response")
        report["positions"] = {
            "status": "verified",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "items": [
                {name: _text(_get(p, name)) for name in
                 ("symbol", "qty", "side", "avg_entry_price", "current_price", "market_value", "unrealized_pl")}
                for p in positions
            ],
        }
    except Exception as exc:
        report["errors"].append({"step": "positions", "error_type": type(exc).__name__})
    try:
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True))
        if not isinstance(orders, (list, tuple)):
            raise ValueError("missing open orders response")
        report["open_orders"] = {
            "status": "verified" if len(orders) < 500 else "possibly_truncated",
            "observed_at_utc": datetime.now(timezone.utc).isoformat(),
            "items": [_order_fields(o) for o in orders],
        }
        if len(orders) >= 500:
            report["errors"].append({"step": "open_orders", "error_type": "response_may_be_truncated"})
    except Exception as exc:
        report["errors"].append({"step": "open_orders", "error_type": type(exc).__name__})

    pending = deque(ORDER_IDS.values())
    visited: set[str] = set()
    while pending and len(visited) < MAX_ORDERS:
        order_id = str(pending.popleft())
        if order_id in visited:
            continue
        visited.add(order_id)
        try:
            order = client.get_order_by_id(order_id, GetOrderByIdRequest(nested=True))
            if order is None or _get(order, "id") is None:
                raise ValueError("missing order response")
            if str(_get(order, "id")).lower() != order_id.lower():
                raise ValueError("broker order id does not match lookup")
            if str(_get(order, "symbol") or "").upper() != "SPY":
                raise ValueError("order symbol does not match SPY")
            entry = _order_fields(order)
            entry["lookup_id"] = order_id
            legs = _get(order, "legs") or []
            entry["leg_ids"] = [_text(_get(leg, "id")) for leg in legs if _get(leg, "id") is not None]
            report["orders"].append(entry)
            # Linked orders and legs are queried independently so their exact
            # status/role and replacement lineage remain visible in the output.
            for linked in [*entry["leg_ids"], entry.get("replaced_by"), entry.get("replaces"), entry.get("parent_order_id")]:
                if linked and linked not in visited:
                    pending.append(linked)
        except Exception as exc:
            report["errors"].append({"step": "order_lookup", "order_id": order_id, "error_type": type(exc).__name__})
    if pending:
        report["errors"].append({"step": "order_lookup", "error_type": "order_limit_exceeded"})
    return report, 2 if report["errors"] else 0


def main() -> int:
    try:
        key, secret = load_credentials()
        # No live URL override and no calls to any mutating broker methods.
        client = TradingClient(key, secret, paper=True)
        # Bound each SDK request and avoid SDK retries that can extend a run.
        if getattr(client, "_session", None) is None:
            raise RuntimeError("SDK HTTP session is unavailable")
        client._session.mount("https://", BoundedTimeoutAdapter(max_retries=0))
        client._retry = 0
        report, code = inspect_orders(client)
    except Exception as exc:
        report = {"paper_account_verified": False, "errors": [{"step": "setup", "error_type": type(exc).__name__}]}
        code = 2
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
