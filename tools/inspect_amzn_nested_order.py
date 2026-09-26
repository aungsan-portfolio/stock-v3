"""
tools/inspect_amzn_nested_order.py — Inspect AMZN Order Structure and Nested Legs on Alpaca Paper Broker.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca_bridge import AlpacaBridge
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus

logging.basicConfig(level=logging.WARNING)


def inspect():
    bridge = AlpacaBridge(mode="swing")
    if not bridge.connect():
        print("ERROR: Could not connect to Alpaca Paper Broker.")
        return

    print("=" * 70)
    print("  AUTHORITATIVE BROKER INSPECTION: AMZN & PORTFOLIO NESTED ORDERS")
    print("=" * 70)

    # 1. Inspect Position
    pos = bridge._client.get_open_position("AMZN") if hasattr(bridge._client, "get_open_position") else None
    print(f"\n[AMZN POSITION ON BROKER]")
    if pos:
        print(f"  Symbol: {pos.symbol}")
        print(f"  Qty: {pos.qty} (shares)")
        print(f"  Avg Entry Price: ${float(pos.avg_entry_price):.2f}")
        print(f"  Current Price: ${float(pos.current_price):.2f}")
        print(f"  Market Value: ${float(pos.market_value):.2f}")
        print(f"  Unrealized PnL: ${float(pos.unrealized_pl):.2f} ({float(pos.unrealized_plpc)*100:.2f}%)")
    else:
        print("  None found!")

    # 2. Inspect Open Working Orders with nested=True
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
    open_orders = bridge._client.get_orders(filter=req)
    print(f"\n[ALL OPEN ORDERS ON BROKER (Count: {len(open_orders)})]")
    for o in open_orders:
        sym = getattr(o, "symbol", "")
        raw_type = getattr(o, "order_type", getattr(o, "type", ""))
        otype = str(getattr(raw_type, "value", raw_type)).lower().replace("ordertype.", "")
        legs = getattr(o, "legs", None)
        print(f"\n  • Order ID: {o.id}")
        print(f"    Client Order ID: {o.client_order_id}")
        print(f"    Symbol: {sym} | Side: {o.side} | Type: {otype} | Qty: {o.qty}")
        print(f"    Status: {o.status} | Time in Force: {o.time_in_force}")
        print(f"    Stop Price: {getattr(o, 'stop_price', None)} | Limit Price: {getattr(o, 'limit_price', None)}")
        print(f"    Parent Order ID: {getattr(o, 'parent_order_id', None)}")
        print(f"    Legs Count: {len(legs) if legs else 0}")
        if legs:
            for i, leg in enumerate(legs, 1):
                leg_type = str(getattr(getattr(leg, "order_type", getattr(leg, "type", "")), "value", getattr(leg, "order_type", getattr(leg, "type", ""))))
                print(f"      - Leg #{i} [ID: {leg.id}]: Type={leg_type}, Side={leg.side}, Qty={leg.qty}, Status={leg.status}, StopPrice={getattr(leg, 'stop_price', None)}, LimitPrice={getattr(leg, 'limit_price', None)}")

    # 3. Specifically inspect recent closed/filled orders for AMZN
    req_closed = GetOrdersRequest(status=QueryOrderStatus.CLOSED, nested=True, symbols=["AMZN"], limit=10)
    closed_orders = bridge._client.get_orders(filter=req_closed)
    print(f"\n[RECENT CLOSED ORDERS FOR AMZN (Count: {len(closed_orders)})]")
    for o in closed_orders:
        raw_type = getattr(o, "order_type", getattr(o, "type", ""))
        otype = str(getattr(raw_type, "value", raw_type)).lower().replace("ordertype.", "")
        legs = getattr(o, "legs", None)
        print(f"  • Order ID: {o.id} | ClientID: {o.client_order_id} | Side: {o.side} | Type: {otype} | Qty: {o.qty} (Filled: {o.filled_qty}) | Status: {o.status}")
        if legs:
            for i, leg in enumerate(legs, 1):
                leg_type = str(getattr(getattr(leg, "order_type", getattr(leg, "type", "")), "value", getattr(leg, "order_type", getattr(leg, "type", ""))))
                print(f"      - Leg #{i} [ID: {leg.id}]: Type={leg_type}, Side={leg.side}, Qty={leg.qty}, Status={leg.status}, StopPrice={getattr(leg, 'stop_price', None)}")


if __name__ == "__main__":
    inspect()
