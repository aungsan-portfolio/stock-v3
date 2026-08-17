import os
from alpaca.trading.client import TradingClient

env = {}
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'").strip('"')

print("Found keys in .env:", list(env.keys()))

# Test Daytrade
dt_k = env.get("DAYTRADE_APCA_API_KEY_ID", "")
dt_s = env.get("DAYTRADE_APCA_API_SECRET_KEY", "")
try:
    c_dt = TradingClient(dt_k, dt_s, paper=True)
    acct_dt = c_dt.get_account()
    print(f"✅ Daytrade Account Connected: {acct_dt.account_number} (Status: {acct_dt.status}, Equity: ${float(acct_dt.equity):.2f})")
except Exception as e:
    print(f"❌ Daytrade Account Error: {e}")

# Test Swing
sw_k = env.get("SWING_APCA_API_KEY_ID", "")
sw_s = env.get("SWING_APCA_API_SECRET_KEY", "")
try:
    c_sw = TradingClient(sw_k, sw_s, paper=True)
    acct_sw = c_sw.get_account()
    print(f"✅ Swing Account Connected: {acct_sw.account_number} (Status: {acct_sw.status}, Equity: ${float(acct_sw.equity):.2f})")
except Exception as e:
    print(f"❌ Swing Account Error: {e}")
