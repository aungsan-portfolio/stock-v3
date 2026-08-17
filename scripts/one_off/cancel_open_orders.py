import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
"""
cancel_open_orders.py — Cancel all open/working orders on the Alpaca paper account.
Use before re-running `paper` after a sizing change.
"""
from ibkr_bridge import IBKRBridge

bridge = IBKRBridge()
if not bridge.connect():
    print("Could not connect to Alpaca")
    exit(1)

open_trades = bridge.ib.openTrades()
print(f"Open working orders: {len(open_trades)}")
for t in open_trades:
    print(
        f"  cancel -> {t.order.action} {t.contract.symbol} "
        f"{t.order.orderType} x{t.order.totalQuantity} ref={t.order.orderRef} "
        f"status={t.orderStatus.status}"
    )
    # Cancel the working order on Alpaca
    bridge._cancel_symbol_working_orders(t.contract.symbol)

bridge.ib.sleep(2)

remaining = bridge.ib.openTrades()
print(f"\nRemaining open orders after cancel: {len(remaining)}")
for t in remaining:
    print(f"  still open: {t.contract.symbol} ref={t.order.orderRef} status={t.orderStatus.status}")

bridge.disconnect()

