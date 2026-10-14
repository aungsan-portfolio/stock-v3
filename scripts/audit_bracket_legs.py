import os
from dotenv import load_dotenv

load_dotenv()
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrderByIdRequest

def inspect_legs():
    s_key = os.getenv("SWING_APCA_API_KEY_ID")
    s_sec = os.getenv("SWING_APCA_API_SECRET_KEY")
    s_tc = TradingClient(s_key, s_sec, paper=True)

    parent_ids = {
        "TSLA": "4de0c453-bbad-4cae-bf4c-1cf59c262e2c",
        "QQQ": "fa18f48c-f6bd-4fa2-a5b9-586ff67cdd33",
        "JPM": "4e163c09-7daa-494c-b6f9-145edbec3c6e",
        "BAC": "64fd2b39-0d3b-4931-b8c5-195548eff129"
    }

    print("==================================================")
    print("SWING BRACKET CHILD LEGS AUDIT")
    print("==================================================")

    for sym, pid in parent_ids.items():
        req = GetOrderByIdRequest(nested=True)
        p = s_tc.get_order_by_id(pid, req)
        print(f"\nSymbol: {sym} | Parent Order ID: {p.id}")
        print(f"  Parent Status: {p.status} | Side: {p.side} | Qty: {p.qty} | Filled: {p.filled_qty}")
        legs = getattr(p, "legs", []) or []
        print(f"  Nested Legs: {len(legs)}")
        for leg in legs:
            print(f"    Leg ID:      {leg.id}")
            print(f"    Order Class: {getattr(leg, 'order_class', 'N/A')}")
            print(f"    Order Type:  {getattr(leg, 'order_type', 'N/A')}")
            print(f"    Side:        {leg.side}")
            print(f"    Qty:         {leg.qty} (Filled: {leg.filled_qty})")
            print(f"    Status:      {leg.status}")
            print(f"    Stop Price:  {leg.stop_price}")
            print(f"    Limit Price: {leg.limit_price}")

if __name__ == "__main__":
    inspect_legs()
