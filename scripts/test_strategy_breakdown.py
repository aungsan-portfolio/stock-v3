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

# Also check timeframe parameters for portfolio history to see intraday
# Alpaca portfolio history allowed timeframes: 1Min, 5Min, 15Min, 1H, 1D
# But period and timeframe must be compatible. For 1M period, timeframe='1D' or '1H'.
# Let's test with period=1M&timeframe=1H or period=7D&timeframe=15Min.
for period, tf in [("1M", "1D"), ("7D", "15Min"), ("1M", "1H")]:
    try:
        url = f"https://paper-api.alpaca.markets/v2/account/portfolio/history?period={period}&timeframe={tf}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            eqs = [e for e in data.get("equity", []) if e is not None]
            if eqs:
                p = eqs[0]
                m_dd = 0.0
                for e in eqs:
                    if e > p: p = e
                    dd = (p - e) / p * 100.0 if p > 0 else 0
                    if dd > m_dd: m_dd = dd
                print(f"Portfolio History ({period}, {tf}): {len(eqs)} points | Max DD: {m_dd:.2f}% | Peak: ${max(eqs):,.2f} | Trough: ${min(eqs):,.2f}")
    except Exception as ex:
        print(f"Error ({period}, {tf}): {ex}")

# Now let's do Strategy Attribution on Closed Round-Trips
all_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500))
filled_orders = [o for o in all_orders if str(o.status) == "OrderStatus.FILLED"]
filled_orders.sort(key=lambda x: (x.filled_at or x.created_at, x.created_at))

# Trace each round-trip and attribute strategy by the ENTRY client_order_id
positions_fifo = {}
round_trips = []

for o in filled_orders:
    sym = o.symbol
    qty = float(o.filled_qty or o.qty)
    px = float(o.filled_avg_price)
    ts = o.filled_at or o.created_at
    side = str(o.side).replace("OrderSide.", "").upper()
    cid = str(o.client_order_id or "")
    
    if side == "BUY":
        positions_fifo.setdefault(sym, []).append({
            "order_id": str(o.id),
            "client_order_id": cid,
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
            
            # Determine strategy attribution
            entry_cid = buy_lot["client_order_id"]
            if entry_cid.startswith("dt_"):
                strategy = "DAYTRADE"
            elif entry_cid.startswith("swg_"):
                strategy = "SWING_MODERN"
            elif entry_cid.startswith("se3-"):
                strategy = "SWING_PROTOTYPE"
            else:
                strategy = "SWING_OTHER"
            
            round_trips.append({
                "symbol": sym,
                "strategy": strategy,
                "entry_cid": entry_cid,
                "exit_cid": cid,
                "qty": matched_qty,
                "entry_id": buy_lot["order_id"],
                "entry_time": buy_lot["timestamp"].isoformat(),
                "entry_price": buy_lot["price"],
                "exit_id": str(o.id),
                "exit_time": ts.isoformat(),
                "exit_price": px,
                "exit_type": str(o.order_type).replace("OrderType.", ""),
                "realized_pnl": pnl,
                "pnl_pct": pnl_pct,
                "duration_days": duration_days,
            })
            
            buy_lot["qty"] -= matched_qty
            rem_sell_qty -= matched_qty
            if buy_lot["qty"] <= 1e-6:
                positions_fifo[sym].pop(0)

print("\n" + "="*80)
print("STRATEGY ATTRIBUTION BREAKDOWN")
print("="*80)

def calc_stats(trades, name):
    if not trades:
        print(f"\n[{name}] No trades")
        return
    wins = sum(1 for t in trades if t["realized_pnl"] > 0)
    losses = sum(1 for t in trades if t["realized_pnl"] < 0)
    gp = sum(t["realized_pnl"] for t in trades if t["realized_pnl"] > 0)
    gl = abs(sum(t["realized_pnl"] for t in trades if t["realized_pnl"] < 0))
    net = gp - gl
    pf = gp / gl if gl > 0 else 0.0
    wr = wins / len(trades) * 100.0
    print(f"\n--- {name} (N = {len(trades)}) ---")
    print(f"  Wins: {wins} | Losses: {losses} | Win Rate: {wr:.1f}%")
    print(f"  Gross Profit: +${gp:,.2f} | Gross Loss: -${gl:,.2f} | Net Realized PnL: ${net:+,.2f}")
    print(f"  Profit Factor: {pf:.2f}")

calc_stats(round_trips, "TOTAL ACCOUNT-LEVEL (ALL TRADES)")

dt_trades = [t for t in round_trips if t["strategy"] == "DAYTRADE"]
calc_stats(dt_trades, "DAYTRADE BOT LEGACY COHORT (dt_*)")

swing_all = [t for t in round_trips if t["strategy"] in ("SWING_MODERN", "SWING_PROTOTYPE", "SWING_OTHER")]
calc_stats(swing_all, "COMBINED SWING TRADES (swg_* + se3-* + other)")

swing_modern = [t for t in round_trips if t["strategy"] == "SWING_MODERN"]
calc_stats(swing_modern, "MODERN SWING BOT COHORT (swg_*)")

swing_proto = [t for t in round_trips if t["strategy"] == "SWING_PROTOTYPE"]
calc_stats(swing_proto, "PROTOTYPE SWING BOT COHORT (se3-*)")
