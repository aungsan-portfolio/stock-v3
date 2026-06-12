"""flatten_vti.py — Market-sell the open VTI position to flatten it."""
import asyncio
import sys

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Stock, MarketOrder
import config

ib = IB()
ib.RequestTimeout = 30
ib.connect(
    host=config.IBKR_HOST,
    port=config.IBKR_PORT,
    clientId=config.CLIENT_ID_FLATTEN,
    timeout=30,
)
ib.sleep(1)

target = "VTI"
qty = 0
for p in ib.positions():
    if p.contract.symbol.upper() == target:
        qty = int(p.position)

print(f"{target} position qty = {qty}")
if qty > 0:
    contract = Stock(target, "SMART", "USD")
    ib.qualifyContracts(contract)
    order = MarketOrder("SELL", qty)
    trade = ib.placeOrder(contract, order)
    ib.sleep(2)
    print(f"SELL {qty} {target} -> status={trade.orderStatus.status} "
          f"filled={trade.orderStatus.filled} avgFill={trade.orderStatus.avgFillPrice}")
    ib.sleep(2)
else:
    print("Nothing to flatten.")

print("\n=== Positions after ===")
for p in ib.positions():
    print(f"  {p.contract.symbol}: {p.position}")

ib.disconnect()
