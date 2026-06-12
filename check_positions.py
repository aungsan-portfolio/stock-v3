"""check_positions.py — Show current positions and any remaining open orders."""
import asyncio
import sys

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB
import config

ib = IB()
ib.RequestTimeout = 30
ib.connect(
    host=config.IBKR_HOST,
    port=config.IBKR_PORT,
    clientId=config.CLIENT_ID_CHECK,
    timeout=30,
)
ib.sleep(1)

print("=== Positions ===")
positions = ib.positions()
if not positions:
    print("  (none)")
for p in positions:
    print(f"  {p.contract.symbol}: {p.position} @ avgCost={p.avgCost}")

ib.reqAllOpenOrders()
ib.sleep(2)
print("\n=== Open orders ===")
ot = ib.openTrades()
if not ot:
    print("  (none)")
for t in ot:
    print(f"  {t.order.action} {t.contract.symbol} {t.order.orderType} "
          f"x{t.order.totalQuantity} id={t.order.orderId} status={t.orderStatus.status}")

ib.disconnect()
