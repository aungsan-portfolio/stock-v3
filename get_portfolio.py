"""
get_portfolio.py — Output portfolio data as JSON for dashboard.
Usage: python -X utf8 get_portfolio.py
Output: JSON with cash, net_liquidation, positions[], orders[]
"""
import json
import sys
import os
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

try:
    from ibkr_bridge import IBKRBridge
    ib = IBKRBridge()
    ib.connect()

    cash = ib.get_cash()
    nlv = ib.get_net_liquidation()

    positions = []
    try:
        for p in ib.ib.positions():
            if float(p.position) == 0:
                continue
            qty = float(p.position)
            avg_cost = float(p.avgCost) / abs(qty) if abs(qty) > 0 else 0
            mkt_val = float(p.marketValue)
            upnl = float(p.unrealizedPNL)
            positions.append({
                "symbol": p.contract.symbol,
                "qty": qty,
                "avg_cost": round(avg_cost, 2),
                "market_value": round(mkt_val, 2),
                "unrealized_pnl": round(upnl, 2)
            })
    except Exception as e:
        pass

    orders = []
    try:
        for o in ib.ib.reqOpenOrders():
            orders.append({
                "symbol": o.contract.symbol,
                "action": o.action,
                "qty": float(o.totalQuantity),
                "type": o.orderType,
                "limit_price": getattr(o, "lmtPrice", ""),
                "status": o.orderState.status
            })
    except Exception:
        pass

    ib.disconnect()

    print(json.dumps({
        "cash": round(cash, 2),
        "net_liquidation": round(nlv, 2),
        "positions_count": len(positions),
        "positions": positions,
        "orders_count": len(orders),
        "orders": orders
    }))

except Exception as e:
    print(json.dumps({
        "cash": 0,
        "net_liquidation": 0,
        "positions_count": 0,
        "positions": [],
        "orders_count": 0,
        "orders": [],
        "error": str(e)
    }))
