import os
import json
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
import urllib.request

load_dotenv('.env')
key = os.getenv('SWING_APCA_API_KEY_ID')
sec = os.getenv('SWING_APCA_API_SECRET_KEY')
headers = {'APCA-API-KEY-ID': key, 'APCA-API-SECRET-KEY': sec}
client = TradingClient(key, sec, paper=True)

print("="*60)
print("1. ALL TSLA ORDERS")
print("="*60)
req_tsla = GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=['TSLA'], limit=100)
tsla_orders = client.get_orders(req_tsla)
tsla_orders_sorted = sorted(tsla_orders, key=lambda x: x.created_at)
for o in tsla_orders_sorted:
    print(f"{o.created_at.strftime('%Y-%m-%d %H:%M:%S')} | ID: {str(o.id)} | Side: {str(o.side):4} | Status: {str(o.status):20} | Qty: {o.qty} | Filled: {o.filled_qty} @ {o.filled_avg_price} | Type: {str(o.order_type):10} | Stop: {o.stop_price} | Limit: {o.limit_price} | ClientID: {o.client_order_id}")

print("\n" + "="*60)
print("2. ALL QQQ ORDERS")
print("="*60)
req_qqq = GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=['QQQ'], limit=100)
qqq_orders = client.get_orders(req_qqq)
qqq_orders_sorted = sorted(qqq_orders, key=lambda x: x.created_at)
for o in qqq_orders_sorted:
    print(f"{o.created_at.strftime('%Y-%m-%d %H:%M:%S')} | ID: {str(o.id)} | Side: {str(o.side):4} | Status: {str(o.status):20} | Qty: {o.qty} | Filled: {o.filled_qty} @ {o.filled_avg_price} | Type: {str(o.order_type):10} | Stop: {o.stop_price} | Limit: {o.limit_price} | ClientID: {o.client_order_id}")

print("\n" + "="*60)
print("3. PORTFOLIO HISTORY / EQUITY CURVE / MAX DRAWDOWN")
print("="*60)
req_port = urllib.request.Request("https://paper-api.alpaca.markets/v2/account/portfolio/history?period=3M&timeframe=1D", headers=headers)
with urllib.request.urlopen(req_port) as resp:
    port_hist = json.loads(resp.read().decode())
    equity_vals = [e for e in port_hist.get("equity", []) if e is not None]
    timestamps = port_hist.get("timestamp", [])
    print(f"Equity datapoints: {len(equity_vals)}")
    if equity_vals:
        peak = equity_vals[0]
        max_dd_dollars = 0.0
        max_dd_pct = 0.0
        peak_date = None
        trough_date = None
        for t, eq in zip(timestamps, equity_vals):
            if eq > peak:
                peak = eq
            dd_dollars = peak - eq
            dd_pct = dd_dollars / peak * 100.0 if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_dollars = dd_dollars
        print(f"Initial Equity: ${equity_vals[0]:,.2f}")
        print(f"Current/Ending Equity: ${equity_vals[-1]:,.2f}")
        print(f"Peak Equity: ${max(equity_vals):,.2f}")
        print(f"Trough Equity: ${min(equity_vals):,.2f}")
        print(f"Max Drawdown ($): ${max_dd_dollars:,.2f}")
        print(f"Max Drawdown (%): {max_dd_pct:.2f}%")

print("\n" + "="*60)
print("4. JPM BRACKET ORDER AND LEGS")
print("="*60)
# Look for JPM parent order
jpm_parents = [o for o in client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=['JPM'], limit=50)) if o.side.name == 'BUY']
for p in sorted(jpm_parents, key=lambda x: x.created_at, reverse=True)[:3]:
    req = urllib.request.Request(f"https://paper-api.alpaca.markets/v2/orders/{p.id}", headers=headers)
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode())
        print(f"JPM Parent Buy: ID={d.get('id')}, Date={d.get('created_at')}, Status={d.get('status')}, Class={d.get('order_class')}, GroupID={d.get('group_id')}")
        legs = d.get('legs')
        if legs:
            for l in legs:
                print(f"  -> Child Leg: ID={l.get('id')}, Type={l.get('order_type')}, Status={l.get('status')}, GroupID={l.get('group_id')}, GroupRole={l.get('group_role')}, StopPx={l.get('stop_price')}, LimitPx={l.get('limit_price')}")

print("\n" + "="*60)
print("5. MSFT REPLACEMENT / AUDIT HISTORY")
print("="*60)
msft_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=['MSFT'], limit=50))
for o in sorted(msft_orders, key=lambda x: x.created_at):
    req = urllib.request.Request(f"https://paper-api.alpaca.markets/v2/orders/{o.id}", headers=headers)
    with urllib.request.urlopen(req) as resp:
        d = json.loads(resp.read().decode())
        print(f"MSFT: {d.get('created_at')} | ID: {d.get('id')} | Side: {d.get('side')} | Status: {d.get('status')} | Class: {d.get('order_class')} | Replaces: {d.get('replaces')} | ReplacedBy: {d.get('replaced_by')} | StopPx: {d.get('stop_price')} | LimitPx: {d.get('limit_price')}")
