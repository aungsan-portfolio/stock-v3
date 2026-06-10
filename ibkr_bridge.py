"""
ibkr_bridge.py — IBKR Paper Trading bridge via ib_insync.
TWS/Gateway must be running with API enabled on port 7497 (paper).

Production hardening:
- Long-only by default. SELL closes existing longs; never opens accidental shorts.
- Duplicate-entry guard: BUY skipped when already long; SELL skipped when flat.
- Bracket children only on opening trades (closing trades use plain LimitOrder).
- Snapshot price fetch waits for the snapshot to populate (event-driven, not
  fixed sleep). Falls back to bid/ask/close, never NaN.
- qualifyContracts() result captured (the contract is mutated in place but we
  also return the qualified contract from a single source of truth).
"""
import logging
import math
from typing import List, Optional

from ib_insync import IB, Stock, LimitOrder, Contract

import config
from predictor import Signal

logger = logging.getLogger(__name__)

ACCEPTED_ORDER_STATUSES = {"PendingSubmit", "PreSubmitted", "Submitted", "Filled", "ApiPending"}
BAD_ORDER_STATUSES = {"Cancelled", "ApiCancelled", "Inactive", "Rejected"}


def _order_status_name(trade) -> str:
    return str(getattr(getattr(trade, "orderStatus", None), "status", "UNKNOWN"))


def _is_order_accepted(status: str) -> bool:
    status = str(status)
    if status in BAD_ORDER_STATUSES:
        return False
    return status in ACCEPTED_ORDER_STATUSES


class IBKRBridge:
    def __init__(self) -> None:
        self.ib = IB()
        self._contract_cache: dict = {}

    # ── Connection ───────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            self.ib.connect(
                host=config.IBKR_HOST,
                port=config.IBKR_PORT,
                clientId=config.IBKR_CLIENT_ID,
            )
            logger.info("Connected to IBKR | account=%s", self.ib.managedAccounts())
            return True
        except Exception:
            logger.exception("Failed to connect to IBKR TWS")
            return False

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    # ── Account state ────────────────────────────────────────────
    def get_cash(self) -> float:
        for av in self.ib.accountValues():
            if av.tag == "AvailableFunds" and av.currency == "USD":
                try:
                    return float(av.value)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def get_position(self, symbol: str) -> float:
        symbol = symbol.upper().strip()
        for pos in self.ib.positions():
            if pos.contract.symbol.upper() == symbol:
                try:
                    return float(pos.position)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    # ── Contracts & pricing ──────────────────────────────────────
    def _contract(self, symbol: str) -> Contract:
        symbol = symbol.upper().strip()
        cached = self._contract_cache.get(symbol)
        if cached is not None:
            return cached
        contract = Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"Could not qualify contract for {symbol}")
        self._contract_cache[symbol] = qualified[0]
        return qualified[0]

    def get_price(self, symbol: str, timeout: float = 5.0) -> float:
        contract = self._contract(symbol)
        ticker = self.ib.reqMktData(contract, "", True, False)

        def _cancel_snapshot() -> None:
            try:
                self.ib.cancelMktData(contract)
            except Exception:
                pass

        def _finite_positive(value) -> Optional[float]:
            try:
                fv = float(value)
                if fv > 0 and not math.isnan(fv):
                    return fv
            except (TypeError, ValueError):
                return None
            return None

        # Wait for snapshot to populate; ib_insync.sleep yields to the event loop.
        for _ in range(max(1, int(timeout * 4))):
            self.ib.sleep(0.25)

            for attr in ("last", "close"):
                price = _finite_positive(getattr(ticker, attr, None))
                if price is not None:
                    _cancel_snapshot()
                    return price

            # In ib_insync marketPrice is a method, not a numeric attribute.
            market_price_fn = getattr(ticker, "marketPrice", None)
            if callable(market_price_fn):
                try:
                    price = _finite_positive(market_price_fn())
                    if price is not None:
                        _cancel_snapshot()
                        return price
                except Exception:
                    pass

            bid = _finite_positive(getattr(ticker, "bid", None))
            ask = _finite_positive(getattr(ticker, "ask", None))
            if bid is not None and ask is not None:
                _cancel_snapshot()
                return (bid + ask) / 2.0

        _cancel_snapshot()
        return 0.0

    # ── Quantity / pricing helpers ───────────────────────────────
    def _calc_quantity(self, price: float, cash: float) -> int:
        if price <= 0:
            return 0
        max_value = cash * config.MAX_POSITION_PCT
        return max(int(max_value / price), 0)

    def _limit_price(self, action: str, price: float) -> float:
        offset = float(config.LIMIT_ORDER_OFFSET_PCT)
        if action == "BUY":
            return round(price * (1 + offset), 2)
        return round(price * (1 - offset), 2)

    # ── Order placement ──────────────────────────────────────────
    def _place_limit_order(self, symbol: str, action: str, qty: int, price: float, note: str) -> bool:
        if qty <= 0:
            logger.warning("Invalid qty=%s for %s %s", qty, action, symbol)
            return False

        contract = self._contract(symbol)
        limit_price = self._limit_price(action, price)
        order = LimitOrder(action, qty, limit_price)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)

        status = _order_status_name(trade)
        accepted = _is_order_accepted(status)
        logger.info(
            "%s | %s %s x%d limit=%.2f | status=%s accepted=%s",
            note, action, symbol, qty, limit_price, status, accepted,
        )
        return accepted

    def _place_open_bracket(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
    ) -> bool:
        contract = self._contract(symbol)

        if action == "BUY":
            stop_price = round(price * (1 - config.STOP_LOSS_PCT), 2)
            profit_price = round(price * (1 + config.TAKE_PROFIT_PCT), 2)
        elif action == "SELL":
            stop_price = round(price * (1 + config.STOP_LOSS_PCT), 2)
            profit_price = round(price * (1 - config.TAKE_PROFIT_PCT), 2)
        else:
            raise ValueError(f"Unsupported bracket action: {action}")

        bracket = self.ib.bracketOrder(
            action=action,
            quantity=qty,
            limitPrice=self._limit_price(action, price),
            takeProfitPrice=profit_price,
            stopLossPrice=stop_price,
        )

        trades = [self.ib.placeOrder(contract, order) for order in bracket]
        self.ib.sleep(1)
        statuses = [_order_status_name(t) for t in trades]
        accepted = all(_is_order_accepted(status) for status in statuses)
        logger.info(
            "Open bracket | %s %s x%d @ %.2f | SL=%.2f TP=%.2f | conf=%.2f | statuses=%s accepted=%s",
            action, symbol, qty, price, stop_price, profit_price, confidence, statuses, accepted,
        )
        return accepted

    # ── Public API ───────────────────────────────────────────────
    def execute_signal(self, signal: Signal) -> bool:
        symbol = signal.symbol.upper().strip()
        action = signal.action.upper().strip()
        allow_short = bool(config.ALLOW_SHORT)
        min_cash = float(config.MIN_TRADE_CASH)

        if action == "HOLD":
            return False
        if action not in {"BUY", "SELL"}:
            logger.warning("Unsupported signal action for %s: %s", symbol, signal.action)
            return False

        position = self.get_position(symbol)
        price = self.get_price(symbol)
        if price <= 0:
            logger.warning("Could not get valid price for %s", symbol)
            return False

        if action == "BUY":
            if position > 0:
                logger.info("Already long %s position=%.2f — skipping BUY", symbol, position)
                return False
            if position < 0:
                qty = int(abs(position))
                return self._place_limit_order(symbol, "BUY", qty, price, "Close short")

            cash = self.get_cash()
            if cash < min_cash:
                logger.warning("Insufficient cash for BUY %s: $%.2f", symbol, cash)
                return False
            qty = self._calc_quantity(price, cash)
            if qty == 0:
                logger.warning("Qty=0 for BUY %s price=%.2f cash=%.2f", symbol, price, cash)
                return False
            return self._place_open_bracket(symbol, "BUY", qty, price, signal.confidence)

        # action == "SELL"
        if position > 0:
            qty = int(position)
            return self._place_limit_order(symbol, "SELL", qty, price, "Close long")
        if position < 0:
            logger.info("Already short %s position=%.2f — skipping SELL", symbol, position)
            return False
        if not allow_short:
            logger.info("No long in %s — skipping SELL (ALLOW_SHORT=False)", symbol)
            return False

        cash = self.get_cash()
        if cash < min_cash:
            logger.warning("Insufficient cash for short SELL %s: $%.2f", symbol, cash)
            return False
        qty = self._calc_quantity(price, cash)
        if qty == 0:
            logger.warning("Qty=0 for short SELL %s price=%.2f cash=%.2f", symbol, price, cash)
            return False
        return self._place_open_bracket(symbol, "SELL", qty, price, signal.confidence)

    def execute_all(self, signals: List[Signal]) -> dict:
        open_symbols = {
            p.contract.symbol.upper()
            for p in self.ib.positions()
            if float(p.position) != 0.0
        }
        planned_new_symbols: set = set()
        placed, skipped = 0, 0
        allow_short = bool(config.ALLOW_SHORT)

        for signal in signals:
            symbol = signal.symbol.upper().strip()
            action = signal.action.upper().strip()

            if action == "HOLD":
                skipped += 1
                continue

            current_position = self.get_position(symbol)
            opens_new = (
                (action == "BUY" and current_position == 0)
                or (action == "SELL" and current_position == 0 and allow_short)
            )

            if opens_new:
                planned_total = len(open_symbols | planned_new_symbols)
                if planned_total >= config.MAX_OPEN_POSITIONS:
                    logger.warning(
                        "Max positions (%d) reached — skipping %s",
                        config.MAX_OPEN_POSITIONS, symbol,
                    )
                    skipped += 1
                    continue

            if self.execute_signal(signal):
                placed += 1
                if opens_new:
                    planned_new_symbols.add(symbol)
            else:
                skipped += 1

        return {"placed": placed, "skipped": skipped, "total": len(signals)}
