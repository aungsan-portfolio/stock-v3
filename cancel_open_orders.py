"""
cancel_open_orders.py — Cancel all open/working orders on the IBKR paper account.
Use before re-running `paper` after a sizing change.
"""
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
    clientId=config.CLIENT_ID_CANCEL,
    timeout=30,
)

# reqAllOpenOrders surfaces orders placed by any client id, not just this session.
ib.reqAllOpenOrders()
ib.sleep(2)

open_trades = ib.openTrades()
print(f"Open working orders: {len(open_trades)}")
for t in open_trades:
    print(
        f"  cancel -> {t.order.action} {t.contract.symbol} "
        f"{t.order.orderType} x{t.order.totalQuantity} id={t.order.orderId} "
        f"status={t.orderStatus.status}"
    )
    ib.cancelOrder(t.order)

ib.sleep(2)

remaining = ib.openTrades()
print(f"\nRemaining open orders after cancel: {len(remaining)}")
for t in remaining:
    print(f"  still open: {t.contract.symbol} id={t.order.orderId} status={t.orderStatus.status}")

ib.disconnect()
