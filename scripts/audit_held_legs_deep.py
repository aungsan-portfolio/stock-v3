import os
import json
from dotenv import load_dotenv
load_dotenv('.env')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, GetOrderByIdRequest
from alpaca.trading.enums import QueryOrderStatus

key = os.getenv('SWING_APCA_API_KEY_ID') or os.getenv('APCA_API_KEY_ID')
sec = os.getenv('SWING_APCA_API_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY')
client = TradingClient(key, sec, paper=True)

symbols = ['TSLA', 'QQQ', 'JPM', 'BAC', 'MSFT']
pos_map = {p.symbol: p for p in client.get_all_positions()}

print("=== SWING POSITIONS AUDIT ===")
for sym in symbols:
    p = pos_map.get(sym)
    if p:
        print(f"{sym}: qty={p.qty}, entry_px={p.avg_entry_price}, current_px={p.current_price}, market_val={p.market_value}")
    else:
        print(f"{sym}: NO POSITION")

stop_order_ids = {
    'TSLA': '33d02017-3747-499a-83e7-307dbcc5c6b8',
    'QQQ': '73adb4af-08c9-4101-8b04-851d8d054079',
    'JPM': '4505f4b4-6e69-473e-818a-a8d185410235',
    'BAC': 'c3cce6b2-ca2f-469e-9b1f-7d222eb96171',
    'MSFT': 'c37ed236-ad39-464d-92f3-ace65f17b0cd',
}

print("\n=== DEEP STOP LEG & PARENT AUDIT ===")
results = []
for sym, oid in stop_order_ids.items():
    res = {"symbol": sym, "stop_order_id": oid}
    try:
        ord_obj = client.get_order_by_id(oid, GetOrderByIdRequest(nested=True))
        res["leg_symbol"] = str(ord_obj.symbol)
        res["leg_side"] = str(ord_obj.side)
        res["leg_type"] = str(ord_obj.order_type)
        res["leg_class"] = str(ord_obj.order_class)
        res["leg_status"] = str(ord_obj.status)
        res["leg_qty"] = str(ord_obj.qty)
        res["leg_filled_qty"] = str(ord_obj.filled_qty)
        res["leg_stop_price"] = str(ord_obj.stop_price)
        res["leg_limit_price"] = str(ord_obj.limit_price)
        res["leg_tif"] = str(ord_obj.time_in_force)
        res["leg_created_at"] = str(ord_obj.created_at)
        res["leg_updated_at"] = str(ord_obj.updated_at)
        res["parent_order_id"] = str(ord_obj.parent_order_id)
        res["replaces"] = str(getattr(ord_obj, "replaces", None))
        res["replaced_by"] = str(getattr(ord_obj, "replaced_by", None))

        parent_info = None
        if ord_obj.parent_order_id:
            parent = client.get_order_by_id(ord_obj.parent_order_id, GetOrderByIdRequest(nested=True))
            parent_info = {
                "parent_id": str(parent.id),
                "parent_symbol": str(parent.symbol),
                "parent_side": str(parent.side),
                "parent_type": str(parent.order_type),
                "parent_class": str(parent.order_class),
                "parent_status": str(parent.status),
                "parent_qty": str(parent.qty),
                "parent_filled_qty": str(parent.filled_qty),
                "parent_filled_avg_price": str(parent.filled_avg_price),
                "parent_filled_at": str(parent.filled_at),
                "legs": []
            }
            if hasattr(parent, 'legs') and parent.legs:
                for l in parent.legs:
                    parent_info["legs"].append({
                        "id": str(l.id),
                        "type": str(l.order_type),
                        "side": str(l.side),
                        "status": str(l.status),
                        "qty": str(l.qty),
                        "stop_price": str(l.stop_price),
                        "limit_price": str(l.limit_price),
                    })
        res["parent"] = parent_info
        results.append(res)
    except Exception as e:
        res["error"] = str(e)
        results.append(res)

with open("reports/held_stop_legs_deep_audit.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

for r in results:
    print(f"\n[{r['symbol']}] Stop Leg ID: {r['stop_order_id']}")
    print(f"  Status: {r.get('leg_status')} | Qty: {r.get('leg_qty')} | Stop Price: ${r.get('leg_stop_price')} | TIF: {r.get('leg_tif')}")
    p = r.get("parent")
    if p:
        print(f"  Parent: {p.get('parent_id')} | Status: {p.get('parent_status')} | Qty: {p.get('parent_qty')} | Filled Qty: {p.get('parent_filled_qty')} | Filled At: {p.get('parent_filled_at')}")
        print(f"  Bracket Legs Count: {len(p.get('legs', []))}")
        for idx, leg in enumerate(p.get('legs', [])):
            print(f"    - Leg {idx+1}: id={leg.get('id')} type={leg.get('type')} status={leg.get('status')} stop={leg.get('stop_price')} limit={leg.get('limit_price')}")
    else:
        print("  Parent: None (Standalone Order)")
