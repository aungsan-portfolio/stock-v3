"""
order_registry.py -- In-memory thread-safe registry tracking expected signal quotes & submit timestamps.

Used by trade_journal and alpaca_bridge to calculate exact signed slippage and execution latency.
"""
import time
import threading
from typing import Optional, Dict, Any

_lock = threading.Lock()
_ORDER_REGISTRY: Dict[str, Dict[str, Any]] = {}

def register_order_expected_price(
    order_id: str,
    expected_price: float,
    side: str,
    order_type: str = "ENTRY",
    submit_ts: Optional[float] = None,
    symbol: Optional[str] = None,
    strategy: Optional[str] = None,
    signal_id: Optional[str] = None,
    is_validation: bool = False,
    client_order_id: Optional[str] = None,
) -> None:
    if not order_id:
        return
    order_key = str(order_id).strip()
    with _lock:
        _ORDER_REGISTRY[order_key] = {
            "expected_price": float(expected_price),
            "side": str(side).upper(),
            "order_type": str(order_type).upper(),
            "submit_ts": float(submit_ts if submit_ts is not None else time.time()),
            "symbol": str(symbol).upper() if symbol else "",
            "strategy": str(strategy) if strategy else "UNKNOWN",
            "signal_id": str(signal_id) if signal_id else None,
            "is_validation": bool(is_validation),
            "client_order_id": str(client_order_id) if client_order_id else None,
        }

def get_registered_order(order_id: str) -> Optional[Dict[str, Any]]:
    if not order_id:
        return None
    order_key = str(order_id).strip()
    with _lock:
        return _ORDER_REGISTRY.get(order_key)


def is_validation_order(order_id: str) -> bool:
    """Return True if the order is tagged as a validation lane order."""
    reg = get_registered_order(order_id)
    if reg:
        return bool(reg.get("is_validation", False))
    return False

