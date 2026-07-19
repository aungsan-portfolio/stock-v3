"""protect_vti_gtc.py — Place Stop Loss and Take Profit orders on Alpaca for existing VTI position."""
from datetime import datetime
from ibkr_bridge import IBKRBridge

SYMBOL = "VTI"
TAKE_PROFIT_PRICE = 413.78
STOP_LOSS_PRICE = 341.82

bridge = IBKRBridge()

try:
    if not bridge.connect():
        print("Could not connect to Alpaca")
        exit(1)

    qty = int(bridge.get_position(SYMBOL))
    if qty <= 0:
        print(f"No long {SYMBOL} position found. Nothing to protect.")
        exit(0)

    # Submit separate limit and stop orders for protection on Alpaca
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, StopOrderRequest

    print(f"Submitting protection for {SYMBOL}: {qty} shares")
    
    # Take Profit
    tp_order = bridge._client.submit_order(LimitOrderRequest(
        symbol=SYMBOL, qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        limit_price=round(TAKE_PROFIT_PRICE, 2)
    ))
    
    # Stop Loss
    sl_order = bridge._client.submit_order(StopOrderRequest(
        symbol=SYMBOL, qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        stop_price=round(STOP_LOSS_PRICE, 2)
    ))

    print(f"Protected {SYMBOL} position: {qty} shares")
    print(f"Take-profit (LMT): {tp_order.id} @ {TAKE_PROFIT_PRICE}")
    print(f"Stop-loss   (STP): {sl_order.id} @ {STOP_LOSS_PRICE}")

finally:
    bridge.disconnect()
