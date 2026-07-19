"""flatten_vti.py — Market-sell the open VTI position to flatten it on Alpaca."""
from ibkr_bridge import IBKRBridge

bridge = IBKRBridge()
if not bridge.connect():
    print("Could not connect to Alpaca")
    exit(1)

target = "VTI"
qty = 0
for p in bridge.ib.positions():
    if p.contract.symbol.upper() == target:
        qty = int(p.position)

print(f"{target} position qty = {qty}")
if qty > 0:
    # Use bridge's custom close_position method
    price = bridge.get_price(target)
    trade = bridge._close_position(target, "SELL", qty, price, "Manual flatten VTI")
    bridge.ib.sleep(2)
    print(f"SELL {qty} {target} -> status={trade.status} "
          f"filled={trade.filled} avgFill={trade.avg_fill_price}")
    bridge.ib.sleep(2)
else:
    print("Nothing to flatten.")

print("\n=== Positions after ===")
for p in bridge.ib.positions():
    print(f"  {p.contract.symbol}: {p.position}")

bridge.disconnect()
