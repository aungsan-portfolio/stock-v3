import asyncio
from datetime import datetime

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from ib_insync import IB, Stock, LimitOrder, StopOrder

HOST = "127.0.0.1"
PORT = 7497
CLIENT_ID = 101

SYMBOL = "VTI"
TAKE_PROFIT_PRICE = 413.78
STOP_LOSS_PRICE = 341.82

ib = IB()

try:
    ib.connect(HOST, PORT, clientId=CLIENT_ID, timeout=15)

    contract = Stock(SYMBOL, "SMART", "USD")
    ib.qualifyContracts(contract)

    positions = [p for p in ib.positions() if p.contract.symbol == SYMBOL]
    qty = sum(p.position for p in positions)

    if qty <= 0:
        raise SystemExit("No long VTI position found. Nothing to protect.")

    qty = int(qty)

    oca_group = f"VTI_PROTECT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    take_profit = LimitOrder("SELL", qty, TAKE_PROFIT_PRICE)
    stop_loss = StopOrder("SELL", qty, STOP_LOSS_PRICE)

    for order in (take_profit, stop_loss):
        order.tif = "GTC"
        order.ocaGroup = oca_group
        order.ocaType = 1
        order.transmit = True

    tp_trade = ib.placeOrder(contract, take_profit)
    sl_trade = ib.placeOrder(contract, stop_loss)

    ib.sleep(2)

    print(f"Protected VTI position: {qty} shares")
    print(f"Take-profit: SELL LMT x{qty} @ {TAKE_PROFIT_PRICE} GTC")
    print(f"Stop-loss:   SELL STP x{qty} @ {STOP_LOSS_PRICE} GTC")
    print(f"OCA group:   {oca_group}")
    print("TP status:", tp_trade.orderStatus.status)
    print("SL status:", sl_trade.orderStatus.status)

finally:
    if ib.isConnected():
        ib.disconnect()
    loop.close()
