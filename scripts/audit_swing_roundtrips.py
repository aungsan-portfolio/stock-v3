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

# 1. Alpaca Portfolio History / Equity Curve & Peak-to-Trough Drawdown
print("=" * 80)
print("SECTION 1: ACCOUNT EQUITY CURVE & MAXIMUM DRAWDOWN AUDIT (GATE 3)")
print("=" * 80)
req_port = urllib.request.Request("https://paper-api.alpaca.markets/v2/account/portfolio/history?period=3M&timeframe=1D", headers=headers)
with urllib.request.urlopen(req_port) as resp:
    port_hist = json.loads(resp.read().decode())
    equity_vals = [e for e in port_hist.get("equity", []) if e is not None]
    timestamps = port_hist.get("timestamp", [])
    
    peak = equity_vals[0] if equity_vals else 0.0
    max_dd_dollars = 0.0
    max_dd_pct = 0.0
    peak_idx = 0
    trough_idx = 0
    
    for idx, (t, eq) in enumerate(zip(timestamps, equity_vals)):
        if eq > peak:
            peak = eq
            peak_idx = idx
        dd_dollars = peak - eq
        dd_pct = (dd_dollars / peak) * 100.0 if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_dollars = dd_dollars
            trough_idx = idx

    print(f"Data Points: {len(equity_vals)} daily closes")
    print(f"Initial Account Equity:  ${equity_vals[0]:,.2f}")
    print(f"Current Account Equity:  ${equity_vals[-1]:,.2f}")
    print(f"Historical Peak Equity:  ${max(equity_vals):,.2f} (Index {peak_idx})")
    print(f"Historical Trough Equity:${min(equity_vals):,.2f} (Index {trough_idx})")
    print(f"Maximum Drawdown ($):    ${max_dd_dollars:,.2f}")
    print(f"Maximum Drawdown (%):    {max_dd_pct:.2f}%")
    print(f"Gate 3 Threshold:        <= 5.00%")
    print(f"Gate 3 Status:           {'PASS' if max_dd_pct <= 5.0 else 'FAIL'}")

# 2. Extract All Orders and Sort by filled_at
print("\n" + "=" * 80)
print("SECTION 2: RECONCILED FIFO ROUND-TRIP TRADE LEDGER (SORTED BY FILLED_AT)")
print("=" * 80)
client = TradingClient(key, sec, paper=True)
all_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
filled_orders = [o for o in all_orders if str(o.status) == "OrderStatus.FILLED"]
print(f"Total account orders: {len(all_orders)} | Total filled orders: {len(filled_orders)}")

# Sort by actual fill timestamp
filled_orders.sort(key=lambda x: (x.filled_at or x.created_at, x.created_at))

positions_fifo = {}
round_trips = []

for o in filled_orders:
    sym = o.symbol
    qty = float(o.filled_qty or o.qty)
    px = float(o.filled_avg_price)
    ts = o.filled_at or o.created_at
    side = str(o.side).replace("OrderSide.", "").upper()
    
    if side == "BUY":
        positions_fifo.setdefault(sym, []).append({
            "order_id": str(o.id),
            "client_order_id": str(o.client_order_id),
            "qty": qty,
            "price": px,
            "timestamp": ts,
        })
    elif side == "SELL":
        rem_sell_qty = qty
        while rem_sell_qty > 0 and positions_fifo.get(sym):
            buy_lot = positions_fifo[sym][0]
            matched_qty = min(buy_lot["qty"], rem_sell_qty)
            
            cost = matched_qty * buy_lot["price"]
            proceeds = matched_qty * px
            pnl = proceeds - cost
            pnl_pct = (px - buy_lot["price"]) / buy_lot["price"] * 100.0
            duration_days = (ts - buy_lot["timestamp"]).total_seconds() / 86400.0
            
            round_trips.append({
                "symbol": sym,
                "qty": matched_qty,
                "entry_order_id": buy_lot["order_id"],
                "entry_time": buy_lot["timestamp"].isoformat(),
                "entry_price": buy_lot["price"],
                "exit_order_id": str(o.id),
                "exit_time": ts.isoformat(),
                "exit_price": px,
                "exit_type": str(o.order_type).replace("OrderType.", ""),
                "stop_price": float(o.stop_price) if o.stop_price else None,
                "limit_price": float(o.limit_price) if o.limit_price else None,
                "cost_basis": cost,
                "proceeds": proceeds,
                "realized_pnl": pnl,
                "pnl_pct": pnl_pct,
                "duration_days": duration_days,
            })
            
            buy_lot["qty"] -= matched_qty
            rem_sell_qty -= matched_qty
            if buy_lot["qty"] <= 1e-6:
                positions_fifo[sym].pop(0)

print(f"\nTotal Verified Closed Round-Trip Trades Reconstructed: N = {len(round_trips)}")
win_count = sum(1 for r in round_trips if r["realized_pnl"] > 0)
loss_count = sum(1 for r in round_trips if r["realized_pnl"] < 0)
total_pnl = sum(r["realized_pnl"] for r in round_trips)
gross_profit = sum(r["realized_pnl"] for r in round_trips if r["realized_pnl"] > 0)
gross_loss = abs(sum(r["realized_pnl"] for r in round_trips if r["realized_pnl"] < 0))
pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

print(f"Wins: {win_count}, Losses: {loss_count}, Win Rate: {win_count / len(round_trips) * 100:.2f}%")
print(f"Gross Profit: +${gross_profit:,.2f}, Gross Loss: -${gross_loss:,.2f}, Net Realized PnL: ${total_pnl:,.2f}, Profit Factor: {pf:.2f}")

print("\n" + "=" * 80)
print("SECTION 3: REMAINING OPEN POSITIONS (FIFO RESIDUAL LOTS VS BROKER POSITION TRUTH)")
print("=" * 80)
broker_positions = client.get_all_positions()
print("Broker Open Positions:")
for bp in broker_positions:
    print(f"  {bp.symbol}: {bp.qty} shares @ ${float(bp.avg_entry_price):.2f} (Current Px: ${float(bp.current_price):.2f}, Unrealized PnL: ${float(bp.unrealized_pl):.2f})")

print("\nFIFO Residual Open Lots:")
for sym, lots in sorted(positions_fifo.items()):
    active_lots = [l for l in lots if l["qty"] > 1e-6]
    if active_lots:
        tot_q = sum(l["qty"] for l in active_lots)
        print(f"  {sym}: Total {tot_q:.0f} shares across {len(active_lots)} lot(s):")
        for l in active_lots:
            print(f"     -> Lot Qty: {l['qty']:.0f} @ ${l['price']:.2f} | Entry Order ID: {l['order_id']} | Time: {l['timestamp']}")

print("\n" + "=" * 80)
print("SECTION 4: COMPLETE TSLA & QQQ RECONCILIATION")
print("=" * 80)
print("--- TSLA Round-Trips ---")
tsla_trades = [r for r in round_trips if r["symbol"] == "TSLA"]
for idx, r in enumerate(tsla_trades, 1):
    print(f"TSLA #{idx:02d} | Qty: {r['qty']:2.0f} | Entry: {r['entry_time'][:16]} @ ${r['entry_price']:6.2f} (ID: {r['entry_order_id'][:8]}) | Exit: {r['exit_time'][:16]} @ ${r['exit_price']:6.2f} ({r['exit_type']:6}, ID: {r['exit_order_id'][:8]}) | PnL: ${r['realized_pnl']:+7.2f} ({r['pnl_pct']:+5.2f}%) | Days: {r['duration_days']:4.1f}")

print("\n--- QQQ Round-Trips ---")
qqq_trades = [r for r in round_trips if r["symbol"] == "QQQ"]
for idx, r in enumerate(qqq_trades, 1):
    print(f"QQQ #{idx:02d}  | Qty: {r['qty']:2.0f} | Entry: {r['entry_time'][:16]} @ ${r['entry_price']:6.2f} (ID: {r['entry_order_id'][:8]}) | Exit: {r['exit_time'][:16]} @ ${r['exit_price']:6.2f} ({r['exit_type']:6}, ID: {r['exit_order_id'][:8]}) | PnL: ${r['realized_pnl']:+7.2f} ({r['pnl_pct']:+5.2f}%) | Days: {r['duration_days']:4.1f}")

print("\n" + "=" * 80)
print("SECTION 5: ALL RECENT ROUND-TRIP TRADES (FULL LIST)")
print("=" * 80)
for idx, r in enumerate(round_trips, 1):
    print(f"#{idx:02d} | {r['symbol']:5} | Qty: {r['qty']:4.0f} | Entry: {r['entry_time'][:10]} @ ${r['entry_price']:7.2f} ({r['entry_order_id'][:8]}) | Exit: {r['exit_time'][:10]} @ ${r['exit_price']:7.2f} ({r['exit_type']:6}, {r['exit_order_id'][:8]}) | PnL: ${r['realized_pnl']:+8.2f} ({r['pnl_pct']:+6.2f}%) | Days: {r['duration_days']:4.1f}")
