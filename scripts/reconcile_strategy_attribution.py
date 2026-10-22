import os
import json
import urllib.request
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

load_dotenv(".env")
key = os.getenv("SWING_APCA_API_KEY_ID")
sec = os.getenv("SWING_APCA_API_SECRET_KEY")
headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
client = TradingClient(key, sec, paper=True)

print("="*80)
print("1. SYNCHRONIZED REAL-TIME ACCOUNT EQUITY & BALANCE SNAPSHOT")
print("="*80)
acct = client.get_account()
print(f"Account Number:       {acct.account_number}")
print(f"Account Status:       {acct.status}")
print(f"Currency:             {acct.currency}")
print(f"Cash:                 ${float(acct.cash):,.2f}")
print(f"Long Market Value:    ${float(acct.long_market_value):,.2f}")
print(f"Short Market Value:   ${float(acct.short_market_value):,.2f}")
print(f"Equity:               ${float(acct.equity):,.2f}")
print(f"Last Equity (Close):  ${float(acct.last_equity):,.2f}")
print(f"Buying Power:         ${float(acct.buying_power):,.2f}")
print(f"Initial Margin:       ${float(acct.initial_margin):,.2f}")
print(f"Maintenance Margin:   ${float(acct.maintenance_margin):,.2f}")

positions = client.get_all_positions()
print("\nOpen Positions Mark-to-Market Sum:")
calc_lmv = 0.0
calc_unrealized = 0.0
for p in positions:
    mv = float(p.market_value)
    upl = float(p.unrealized_pl)
    calc_lmv += mv
    calc_unrealized += upl
    print(f"  {p.symbol:5} | Qty: {p.qty:>4} | Entry: ${float(p.avg_entry_price):>7.2f} | Current: ${float(p.current_price):>7.2f} | MktVal: ${mv:>10.2f} | Unrealized: ${upl:>+8.2f}")

print(f"\nSum of Long Market Value from Positions: ${calc_lmv:,.2f}")
print(f"Cash + Long Market Value = ${float(acct.cash) + calc_lmv:,.2f} (matches Equity: ${float(acct.equity):,.2f})")
print(f"Difference between Last Close Equity (${float(acct.last_equity):,.2f}) and Real-time Equity (${float(acct.equity):,.2f}) is: ${float(acct.equity) - float(acct.last_equity):+,.2f}")

print("\n" + "="*80)
print("2. INTRADAY PORTFOLIO HISTORY / EQUITY CURVE (HIGHER RESOLUTION)")
print("="*80)
# Let's query 1H or 15Min portfolio history
try:
    req_port = urllib.request.Request("https://paper-api.alpaca.markets/v2/account/portfolio/history?period=1M&timeframe=1H", headers=headers)
    with urllib.request.urlopen(req_port) as resp:
        ph_1h = json.loads(resp.read().decode())
        eq_1h = [e for e in ph_1h.get("equity", []) if e is not None]
        ts_1h = ph_1h.get("timestamp", [])
        if eq_1h:
            peak_1h = eq_1h[0]
            max_dd_dollars_1h = 0.0
            max_dd_pct_1h = 0.0
            for t, eq in zip(ts_1h, eq_1h):
                if eq > peak_1h:
                    peak_1h = eq
                dd = peak_1h - eq
                pct = dd / peak_1h * 100.0 if peak_1h > 0 else 0.0
                if pct > max_dd_pct_1h:
                    max_dd_pct_1h = pct
                    max_dd_dollars_1h = dd
            print(f"1-Hour Resolution (Past 1 Month): {len(eq_1h)} points")
            print(f"  Peak Equity:     ${max(eq_1h):,.2f}")
            print(f"  Trough Equity:   ${min(eq_1h):,.2f}")
            print(f"  Ending Equity:   ${eq_1h[-1]:,.2f}")
            print(f"  Max Drawdown $:  ${max_dd_dollars_1h:,.2f}")
            print(f"  Max Drawdown %:  {max_dd_pct_1h:.2f}%")
except Exception as e:
    print(f"Error fetching 1H portfolio history: {e}")

print("\n" + "="*80)
print("3. STRATEGY ATTRIBUTION (CLIENT ORDER ID ANALYSIS)")
print("="*80)
all_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
print(f"Total account orders: {len(all_orders)}")

dt_orders = []
swing_swg_orders = []
swing_se3_orders = []
uuid_orders = []
other_orders = []

for o in all_orders:
    cid = str(o.client_order_id or "")
    if cid.startswith("dt_"):
        dt_orders.append(o)
    elif cid.startswith("swg_"):
        swing_swg_orders.append(o)
    elif cid.startswith("se3-"):
        swing_se3_orders.append(o)
    elif "-" in cid and len(cid) == 36: # UUID format
        uuid_orders.append(o)
    else:
        other_orders.append(o)

print(f"  dt_* (Daytrade Bot Orders):          {len(dt_orders)}")
print(f"  swg_* (Modern Swing Bot Orders):     {len(swing_swg_orders)}")
print(f"  se3-* (Prototype Swing Bot Orders):  {len(swing_se3_orders)}")
print(f"  UUID (Child bracket/stop legs/SDK):  {len(uuid_orders)}")
print(f"  Other / Manual / Unclassified:       {len(other_orders)}")

print("\n" + "="*80)
print("4. EXECUTION ACTIVITIES (/v2/account/activities/FILL)")
print("="*80)
req_act = urllib.request.Request("https://paper-api.alpaca.markets/v2/account/activities/FILL?direction=desc&page_size=100", headers=headers)
with urllib.request.urlopen(req_act) as resp:
    fills = json.loads(resp.read().decode())
    print(f"Recent Fill Activities Fetched: {len(fills)}")
    if fills:
        print("Sample of 5 most recent Execution Fills:")
        for f in fills[:5]:
            print(f"  Activity ID: {f.get('id')} | Order ID: {f.get('order_id')[:8]} | Sym: {f.get('symbol'):5} | Side: {f.get('side'):4} | Qty: {f.get('qty')} @ ${float(f.get('price')):.4f} | Time: {f.get('transaction_time')}")
