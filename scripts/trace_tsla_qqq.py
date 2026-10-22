import os
import json
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv('.env')
key = os.getenv('SWING_APCA_API_KEY_ID')
sec = os.getenv('SWING_APCA_API_SECRET_KEY')
client = TradingClient(key, sec, paper=True)

# Fetch all filled orders for TSLA
orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=['TSLA'], limit=100))
filled_orders = [o for o in orders if str(o.status) == "OrderStatus.FILLED"]
# Sort by filled_at or created_at
filled_orders.sort(key=lambda x: x.filled_at or x.created_at)

print("=== ALL TSLA FILLED ORDERS CHRONOLOGICALLY ===")
cum_shares = 0
for o in filled_orders:
    qty = float(o.filled_qty)
    side = str(o.side).replace("OrderSide.", "")
    if side == "BUY":
        cum_shares += qty
    else:
        cum_shares -= qty
    print(f"ID: {str(o.id)[:8]} | Side: {side:4} | Qty: {qty:4.0f} | Price: {float(o.filled_avg_price):.2f} | Created: {o.created_at} | Filled: {o.filled_at} | Cum: {cum_shares:4.0f} | ClientID: {o.client_order_id}")

print("\n=== ALL QQQ FILLED ORDERS CHRONOLOGICALLY ===")
qqq_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=['QQQ'], limit=100))
qqq_filled = [o for o in qqq_orders if str(o.status) == "OrderStatus.FILLED"]
qqq_filled.sort(key=lambda x: x.filled_at or x.created_at)
cum_qqq = 0
for o in qqq_filled:
    qty = float(o.filled_qty)
    side = str(o.side).replace("OrderSide.", "")
    if side == "BUY":
        cum_qqq += qty
    else:
        cum_qqq -= qty
    print(f"ID: {str(o.id)[:8]} | Side: {side:4} | Qty: {qty:4.0f} | Price: {float(o.filled_avg_price):.4f} | Created: {o.created_at} | Filled: {o.filled_at} | Cum: {cum_qqq:4.0f} | ClientID: {o.client_order_id}")
