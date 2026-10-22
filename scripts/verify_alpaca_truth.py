"""
Script: scripts/verify_alpaca_truth.py
Purpose: Directly verify Alpaca Paper account truth (positions=0, working orders=0)
Authentication: Uses DAYTRADE_APCA_API_KEY_ID and DAYTRADE_APCA_API_SECRET_KEY from .env
"""
import os
import sys
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus

def main():
    load_dotenv(".env")
    key = os.getenv("DAYTRADE_APCA_API_KEY_ID")
    sec = os.getenv("DAYTRADE_APCA_API_SECRET_KEY")
    url = os.getenv("DAYTRADE_APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not key or not sec:
        print("[ERROR] DAYTRADE_APCA_API_KEY_ID or DAYTRADE_APCA_API_SECRET_KEY missing in .env", file=sys.stderr)
        return 1

    masked_key = key[:4] + "..." + key[-4:] if len(key) >= 8 else "***"
    print(f"Connecting to Alpaca Paper endpoint: {url} with key {masked_key}")

    client = TradingClient(key, sec, paper=True)
    account = client.get_account()
    print("=" * 60)
    print("ALPACAPAPER BROKER TRUTH VERIFICATION")
    print("=" * 60)
    # Primary account invariants audit
    status_ok = (account.status == "ACTIVE" or str(account.status) == "AccountStatus.ACTIVE")
    trading_ok = not account.trading_blocked
    bp_val = float(account.buying_power)
    dt_bp_raw = getattr(account, "daytrading_buying_power", None)
    if dt_bp_raw is not None:
        dt_bp_val = f"${float(dt_bp_raw):,.2f}"
    else:
        dt_bp_val = "None (Deprecated & Retired from Alpaca Trading API schema on July 6, 2026; Trading API uses account.buying_power)"

    print(f"Account ID:           {account.id}")
    print(f"Account Number:       {account.account_number}")
    print(f"Account Status:       {account.status} {'[OK]' if status_ok else '[FAILED]'}")
    print(f"Trading Blocked:      {account.trading_blocked} {'[OK]' if trading_ok else '[BLOCKED]'}")
    print(f"Transfers Blocked:    {account.transfers_blocked}")
    print(f"Account Blocked:      {account.account_blocked}")
    print(f"Trade Suspended User: {account.trade_suspended_by_user}")
    print(f"Currency:             {account.currency}")
    print(f"Multiplier:           {account.multiplier}")
    print(f"Buying Power:         ${bp_val:,.2f} (Broker margin ceiling; NOT bot sizing target. Bot risk caps remain authoritative)")
    regt_bp = float(account.regt_buying_power) if account.regt_buying_power is not None else 0.0
    print(f"Reg-T Buying Power:   ${regt_bp:,.2f}")
    print(f"Daytrading Buying Pwr:{dt_bp_val}")
    print(f"Cash Balance:         ${float(account.cash):,.2f}")
    print(f"Portfolio Value:      ${float(account.portfolio_value):,.2f}")
    print(f"Current Equity:       ${float(account.equity):,.2f}")
    print("-" * 60)

    # Positions check
    positions = client.get_all_positions()
    print(f"Open Positions Count: {len(positions)}")
    for pos in positions:
        print(f"  - Symbol: {pos.symbol}, Side: {pos.side}, Qty: {pos.qty}, Current Price: {pos.current_price}, Unrealized P&L: {pos.unrealized_pl}")

    # Working / Open orders check
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, nested=True)
    open_orders = client.get_orders(filter=req)
    print(f"Working Orders Count: {len(open_orders)}")
    for o in open_orders:
        print(f"  - Order ID: {o.id}, Symbol: {o.symbol}, Side: {o.side}, Type: {o.type}, Status: {o.status}")

    print("-" * 60)
    is_clean = len(positions) == 0 and len(open_orders) == 0
    all_passed = status_ok and trading_ok and (bp_val > 0) and is_clean

    print("=" * 60)
    print("TELEMETRY INVARIANTS AUDIT SUMMARY:")
    print(f"  1. AccountStatus.ACTIVE:   {'PASS' if status_ok else 'FAIL'}")
    print(f"  2. Trading Not Blocked:    {'PASS' if trading_ok else 'FAIL'}")
    print(f"  3. Buying Power Available: {'PASS ($' + f'{bp_val:,.2f})' if bp_val > 0 else 'FAIL'}")
    print(f"  4. Positions = 0 (Flat):   {'PASS' if len(positions) == 0 else 'FAIL'}")
    print(f"  5. Working Orders = 0:     {'PASS' if len(open_orders) == 0 else 'FAIL'}")
    print("-" * 60)

    if all_passed:
        print("VERIFICATION RESULT: ALL 5 TELEMETRY INVARIANTS PASSED (CLEAN & FLAT)")
        print("=" * 60)
        return 0
    else:
        print("VERIFICATION RESULT: INVARIANT FAILURE DETECTED")
        print("=" * 60)
        return 2

if __name__ == "__main__":
    sys.exit(main())
