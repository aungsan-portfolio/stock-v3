"""check_positions.py — Show current positions and any remaining open orders on Alpaca."""
from ibkr_bridge import IBKRBridge

bridge = IBKRBridge()
if not bridge.connect():
    print("Could not connect to Alpaca")
    exit(1)

print("=== Positions ===")
positions = bridge.ib.positions()
if not positions:
    print("  (none)")
for p in positions:
    print(f"  {p.contract.symbol}: {p.position} @ avgCost={p.avgCost}")

print("\n=== Open orders ===")
ot = bridge.ib.openTrades()
if not ot:
    print("  (none)")
for t in ot:
    print(f"  {t.order.action} {t.contract.symbol} {t.order.orderType} "
          f"x{t.order.totalQuantity} ref={t.order.orderRef} status={t.orderStatus.status}")

bridge.disconnect()
