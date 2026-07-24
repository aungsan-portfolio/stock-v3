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
    symbol: Optional[str] = None
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
        }

def get_registered_order(order_id: str) -> Optional[Dict[str, Any]]:
    if not order_id:
        return None
    order_key = str(order_id).strip()
    with _lock:
        return _ORDER_REGISTRY.get(order_key)
