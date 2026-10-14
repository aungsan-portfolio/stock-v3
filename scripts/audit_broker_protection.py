import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest, QueryOrderStatus

def audit():
    now_utc = datetime.now(timezone.utc).isoformat()
    print(f"==================================================")
    print(f"BROKER PROTECTION AUDIT — {now_utc}")
    print(f"==================================================")

    # 1. Daytrade Account
    d_key = os.getenv("DAYTRADE_APCA_API_KEY_ID")
    d_sec = os.getenv("DAYTRADE_APCA_API_SECRET_KEY")
    d_tc = TradingClient(d_key, d_sec, paper=True)
    d_acct = d_tc.get_account()
    d_pos = d_tc.get_all_positions()
    d_ords = d_tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True))

    print(f"\n[DAYTRADE ACCOUNT]")
    print(f"  Account ID:     {d_acct.account_number}")
    print(f"  Status:         {d_acct.status}")
    print(f"  Equity:         ${float(d_acct.equity):.2f}")
    print(f"  Positions:      {len(d_pos)}")
    print(f"  Open Orders:    {len(d_ords)}")

    # 2. Swing Account
    s_key = os.getenv("SWING_APCA_API_KEY_ID")
    s_sec = os.getenv("SWING_APCA_API_SECRET_KEY")
    s_tc = TradingClient(s_key, s_sec, paper=True)
    s_acct = s_tc.get_account()
    s_pos = s_tc.get_all_positions()
    s_ords = s_tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True))

    print(f"\n[SWING ACCOUNT]")
    print(f"  Account ID:     {s_acct.account_number}")
    print(f"  Status:         {s_acct.status}")
    print(f"  Equity:         ${float(s_acct.equity):.2f}")
    print(f"  Positions:      {len(s_pos)}")
    print(f"  Open Orders:    {len(s_ords)}")

    print(f"\n[SWING DETAILED POSITION PROTECTION AUDIT]")
    pos_map = {p.symbol: p for p in s_pos}
    ord_map = {}
    for o in s_ords:
        ord_map.setdefault(o.symbol, []).append(o)

    all_protected = True
    for sym, pos in pos_map.items():
        qty = float(pos.qty)
        ords = ord_map.get(sym, [])
        print(f"\n  Symbol: {sym}")
        print(f"    Current Qty:       {qty}")
        print(f"    Side:              {pos.side}")
        print(f"    Current Price:     ${float(pos.current_price):.2f}")
        print(f"    Unrealized PnL:    ${float(pos.unrealized_pl):.2f} ({float(pos.unrealized_plpc)*100:.2f}%)")
        print(f"    Open Broker Orders: {len(ords)}")

        covered_qty = 0.0
        for o in ords:
            o_qty = float(o.qty or 0)
            print(f"      -> Order ID:    {o.id}")
            print(f"         Class:       {o.order_class}")
            print(f"         Type:        {o.order_type}")
            print(f"         Side:        {o.side}")
            print(f"         Qty:         {o_qty}")
            print(f"         Status:      {o.status}")
            print(f"         Limit Price: {o.limit_price}")
            print(f"         Stop Price:  {o.stop_price}")
            if str(o.side).upper().endswith("SELL"):
                covered_qty += o_qty

        if covered_qty >= qty:
            print(f"    Protection Status: FULLY COVERED ({covered_qty}/{qty} covered)")
        else:
            print(f"    Protection Status: UNCOVERED DEFICIT ({covered_qty}/{qty} covered)")
            all_protected = False

    print(f"\nOverall Swing Account Protection Certified: {all_protected}")
    return all_protected

if __name__ == "__main__":
    audit()
