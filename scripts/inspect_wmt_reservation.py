"""Read-only inspection of WMT_BUY reservation and Alpaca paper broker state.

Run from the project root: python scripts/inspect_wmt_reservation.py
Uses only daytrade paper credentials. It does not modify the state file, place,
cancel or change orders, or decide that a reservation is safe to delete.
Exit 2 means at least one required observation is unknown or incomplete.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any
from uuid import UUID

# Permit execution both as a script on Windows and as an imported test module.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrderByIdRequest, GetOrdersRequest
from scripts.inspect_spy_order_history import (
    BoundedTimeoutAdapter, EXPECTED_ACCOUNT, _get, _order_fields, _text,
    load_credentials,
)

STATE_PATH = ROOT / "data" / "order_idempotency_state.json"
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_LINKED_ORDERS = 30


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _order_id(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("missing order ID")
    return str(UUID(value.strip()))


def _read_reservation(path: Path) -> dict[str, Any]:
    # Limit before parsing, and include only explicitly selected fields in output.
    with path.open("rb") as fh:
        raw = fh.read(MAX_STATE_BYTES + 1)
    if len(raw) > MAX_STATE_BYTES:
        raise ValueError("reservation state too large")
    document = json.loads(raw)
    if not isinstance(document, dict) or not isinstance(document.get("WMT_BUY"), dict):
        raise ValueError("WMT_BUY reservation is missing or invalid")
    entry = document["WMT_BUY"]
    if str(entry.get("symbol", "")).upper().strip() != "WMT" or str(entry.get("side", "")).upper().strip() != "BUY":
        raise ValueError("reservation symbol or side does not match")
    status = entry.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("reservation status missing")
    return {"key": "WMT_BUY", "symbol": "WMT", "side": "BUY",
            "status": status, "parent_order_id": _order_id(entry.get("parent_order_id")),
            "created_at": _text(entry.get("created_at"))}


def inspect(client: Any, state_path: Path = STATE_PATH) -> tuple[dict[str, Any], int]:
    """Return broker observations; success does not authorize state deletion."""
    report: dict[str, Any] = {
        "paper_account_verified": False, "expected_account_suffix": EXPECTED_ACCOUNT[-4:],
        "reservation": {"status": "unknown"},
        "positions": {"status": "unknown", "items": []},
        "open_orders": {"status": "unknown", "items": []},
        "order_lineage": [], "errors": [], "started_at_utc": _utc_now(),
        "deletion_authorized": False,
        "note": "Read-only observations are sequential, not an atomic snapshot; manual review is required. No deletion is authorized.",
    }
    try:
        reservation = _read_reservation(state_path)
        report["reservation"] = reservation
    except Exception as exc:
        report["errors"].append({"step": "reservation", "error_type": type(exc).__name__})
        return report, 2
    try:
        account = client.get_account()
        if str(_get(account, "account_number") or "").strip() != EXPECTED_ACCOUNT:
            raise ValueError("account mismatch")
    except Exception as exc:
        # Do not include exception messages, which may contain credentials.
        report["errors"].append({"step": "verify_account", "error_type": type(exc).__name__})
        return report, 2
    report["paper_account_verified"] = True

    try:
        positions = client.get_all_positions()
        if not isinstance(positions, (list, tuple)):
            raise ValueError("invalid positions")
        for position in positions:
            symbol = _text(_get(position, "symbol"))
            side = (_text(_get(position, "side")) or "").lower()
            qty = float(_get(position, "qty"))
            if not symbol or side not in {"long", "short"} or not math.isfinite(qty) or qty == 0:
                raise ValueError("incomplete position evidence")
        report["positions"] = {"status": "verified", "observed_at_utc": _utc_now(),
            "items": [{name: _text(_get(p, name)) for name in ("symbol", "qty", "side", "avg_entry_price")}
                      for p in positions]}
    except Exception as exc:
        report["errors"].append({"step": "positions", "error_type": type(exc).__name__})

    try:
        orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500, nested=True))
        if not isinstance(orders, (list, tuple)):
            raise ValueError("invalid open orders")
        for order in orders:
            for item in [order, *(_get(order, "legs") or [])]:
                _order_id(_text(_get(item, "id")))
                if not _text(_get(item, "symbol")) or not _text(_get(item, "status")):
                    raise ValueError("incomplete open-order evidence")
        report["open_orders"] = {"status": "verified" if len(orders) < 500 else "possibly_truncated",
                                 "observed_at_utc": _utc_now(), "items": [_order_fields(o) for o in orders]}
        if len(orders) >= 500:
            report["errors"].append({"step": "open_orders", "error_type": "response_may_be_truncated"})
    except Exception as exc:
        report["errors"].append({"step": "open_orders", "error_type": type(exc).__name__})

    pending = deque([reservation["parent_order_id"]])
    visited: set[str] = set()
    while pending and len(visited) < MAX_LINKED_ORDERS:
        oid = pending.popleft()
        if oid in visited:
            continue
        visited.add(oid)
        try:
            order = client.get_order_by_id(oid, GetOrderByIdRequest(nested=True))
            if order is None or _order_id(_text(_get(order, "id"))) != oid:
                raise ValueError("missing or mismatched order")
            if str(_get(order, "symbol") or "").upper().strip() != "WMT":
                raise ValueError("unexpected symbol")
            side = (_text(_get(order, "side")) or "").lower().strip()
            if side != "buy" or not _text(_get(order, "status")):
                raise ValueError("missing or mismatched order metadata")
            data = _order_fields(order)
            data["lookup_id"] = oid
            report["order_lineage"].append(data)
            for field in ("replaces", "replaced_by"):
                link = _get(order, field)
                if link:
                    linked_id = _order_id(_text(link))
                    if linked_id not in visited:
                        pending.append(linked_id)
        except Exception as exc:
            report["errors"].append({"step": "order_lookup", "order_id": oid, "error_type": type(exc).__name__})
    if pending:
        report["errors"].append({"step": "order_lookup", "error_type": "order_limit_exceeded"})
    return report, 2 if report["errors"] else 0


def main() -> int:
    try:
        key, secret = load_credentials()
        client = TradingClient(key, secret, paper=True)
        if getattr(client, "_session", None) is None:
            raise RuntimeError("SDK HTTP session unavailable")
        client._session.mount("https://", BoundedTimeoutAdapter(max_retries=0))
        client._retry = 0
        report, code = inspect(client)
    except Exception as exc:
        report = {"paper_account_verified": False, "deletion_authorized": False,
                  "errors": [{"step": "setup", "error_type": type(exc).__name__}]}
        code = 2
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
