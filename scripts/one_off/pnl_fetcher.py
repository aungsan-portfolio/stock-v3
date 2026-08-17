import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

logger = logging.getLogger("pnl_fetcher")

def _enum_value(value):
    """Return clean enum/string value."""
    if hasattr(value, "value"):
        return value.value
    return str(value)

def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

def fetch_details():
    from alpaca_bridge import AlpacaBridge

    bridge = AlpacaBridge()
    bridge.connect()

    try:
        account = bridge._client.get_account()

        equity = _to_float(getattr(account, "equity", None))
        last_equity = _to_float(getattr(account, "last_equity", None))
        day_pnl = equity - last_equity

        day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0.0

        logger.info("=== ALPACA ACCOUNT SUMMARY ===")
        logger.info(f"Account ID: {getattr(account, 'id', 'N/A')}")
        logger.info(f"Current Equity: ${equity:,.2f}")
        logger.info(f"Start-of-Day / Last Equity: ${last_equity:,.2f}")
        logger.info(f"Day PnL: ${day_pnl:+,.2f} ({day_pnl_pct:+.2f}%)")

        # Query recent orders using raw Alpaca client
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            req = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                limit=100,
            )

            orders = bridge._client.get_orders(req)

            logger.info("")
            logger.info(f"=== RECENT ORDERS ({len(orders)} total fetched) ===")

            for o in orders:
                created_at = getattr(o, "created_at", None)
                o_date = str(created_at)[:19] if created_at else ""

                symbol = str(getattr(o, "symbol", ""))
                status = _enum_value(getattr(o, "status", ""))
                side = _enum_value(getattr(o, "side", ""))
                order_type = _enum_value(getattr(o, "type", ""))

                qty = str(getattr(o, "qty", ""))
                filled_qty = _to_float(getattr(o, "filled_qty", None))
                filled_avg = _to_float(getattr(o, "filled_avg_price", None))

                order_id = str(getattr(o, "id", ""))
                order_id_short = f"{order_id[:8]}..." if order_id else "N/A"

                logger.info(
                    f"[{o_date}] "
                    f"{symbol:5s} | "
                    f"{side.upper():4s} | "
                    f"{order_type.upper():12s} | "
                    f"Status: {status:10s} | "
                    f"Qty: {qty:>6s} | "
                    f"Filled: {filled_qty:5.1f} @ ${filled_avg:,.2f} | "
                    f"ID: {order_id_short}"
                )

        except Exception as exc:
            logger.exception(f"Error listing orders: {exc}")

    finally:
        try:
            bridge.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    fetch_details()

