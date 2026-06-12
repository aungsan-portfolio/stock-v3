"""
ibkr_bridge.py — IBKR Paper Trading bridge via ib_insync.
TWS/Gateway must be running with API enabled on port 7497 (paper).

Production hardening:
- Paper-only lock by default; refuses non-paper ports unless config opts out.
- Long-only by default. SELL closes existing longs; never opens accidental shorts.
- Duplicate-entry guard: pending/working orders count as occupied symbols.
- Opening trades use either a fixed TP/SL bracket or a parent limit + trailing stop.
- Snapshot price fetch waits for the snapshot to populate (event-driven, not
  fixed sleep). Historical close fallback is blocked for order placement by default.
- qualifyContracts() result captured (the contract is mutated in place but we
  also return the qualified contract from a single source of truth).
"""
import logging
import math
from typing import List, Optional

from ib_insync import IB, Stock, LimitOrder, Contract, Order

import config
import risk_state
from predictor import Signal

logger = logging.getLogger(__name__)

ACCEPTED_ORDER_STATUSES = {"PendingSubmit", "PreSubmitted", "Submitted", "Filled", "ApiPending"}
BAD_ORDER_STATUSES = {"Cancelled", "ApiCancelled", "Inactive", "Rejected"}
WORKING_ORDER_STATUSES = {"PendingSubmit", "ApiPending", "PreSubmitted", "Submitted"}


def _order_status_name(trade) -> str:
    return str(getattr(getattr(trade, "orderStatus", None), "status", "UNKNOWN"))


def _is_order_accepted(status: str) -> bool:
    status = str(status)
    if status in BAD_ORDER_STATUSES:
        return False
    return status in ACCEPTED_ORDER_STATUSES


def _is_order_working(status: str) -> bool:
    return str(status) in WORKING_ORDER_STATUSES


class IBKRBridge:
    def __init__(self) -> None:
        self.ib = IB()
        self._contract_cache: dict = {}

    # ── Connection ───────────────────────────────────────────────
    def connect(self) -> bool:
        if bool(getattr(config, "REQUIRE_PAPER_PORT", True)):
            paper_port = int(getattr(config, "PAPER_IBKR_PORT", 7497))
            if int(config.IBKR_PORT) != paper_port:
                logger.error(
                    "Refusing to connect: IBKR_PORT=%s is not paper port %s",
                    config.IBKR_PORT,
                    paper_port,
                )
                return False

        try:
            self.ib.connect(
                host=config.IBKR_HOST,
                port=config.IBKR_PORT,
                clientId=getattr(config, "CLIENT_ID_BOT", config.IBKR_CLIENT_ID),
            )
            # Fall back to delayed (15-min) data when the account lacks a
            # real-time market-data subscription. 3 = DELAYED, 4 = DELAYED_FROZEN.
            try:
                self.ib.reqMarketDataType(config.IBKR_MARKET_DATA_TYPE)
            except Exception:
                logger.warning("Could not set market data type", exc_info=True)
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

    def working_order_symbols(self) -> set:
        """Symbols with working orders, across all visible client ids."""
        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(1)
        except Exception:
            logger.warning("Could not refresh open orders", exc_info=True)

        symbols = set()
        for trade in self.ib.openTrades():
            status = _order_status_name(trade)
            if _is_order_working(status):
                symbol = str(getattr(trade.contract, "symbol", "")).upper().strip()
                if symbol:
                    symbols.add(symbol)
        return symbols

    def has_working_order(self, symbol: str, action: Optional[str] = None) -> bool:
        """Return True when IBKR already has a working order for this symbol.

        This prevents duplicate entries when a previous limit/bracket/trailing
        order is still PendingSubmit/PreSubmitted/Submitted but has not yet
        become a position. reqAllOpenOrders is used so orders from other
        clientIds are visible too.
        """
        symbol = symbol.upper().strip()
        action = action.upper().strip() if action else None

        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(1)
        except Exception:
            logger.warning("Could not refresh open orders", exc_info=True)

        for trade in self.ib.openTrades():
            contract_symbol = str(getattr(trade.contract, "symbol", "")).upper().strip()
            order_action = str(getattr(trade.order, "action", "")).upper().strip()
            status = _order_status_name(trade)

            if contract_symbol != symbol:
                continue
            if not _is_order_working(status):
                continue
            if action is not None and order_action != action:
                continue
            return True

        return False

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

    def get_price(self, symbol: str, timeout: float = 5.0, allow_historical: bool = True) -> float:
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

            for attr in ("last", "close", "delayedLast", "delayedClose"):
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

            bid = _finite_positive(getattr(ticker, "bid", None)) \
                or _finite_positive(getattr(ticker, "delayedBid", None))
            ask = _finite_positive(getattr(ticker, "ask", None)) \
                or _finite_positive(getattr(ticker, "delayedAsk", None))
            if bid is not None and ask is not None:
                _cancel_snapshot()
                return (bid + ask) / 2.0

        _cancel_snapshot()

        if not allow_historical:
            logger.warning("No live/delayed snapshot price for %s; historical fallback disabled", symbol)
            return 0.0

        # Fallback: last daily close from historical data. Works without a
        # real-time subscription and when delayed snapshots fail to populate
        # (e.g. outside regular trading hours). This should not be used for
        # order placement unless config explicitly allows it.
        hist_price = self._historical_close(contract)
        if hist_price is not None:
            return hist_price

        return 0.0

    def _historical_close(self, contract) -> Optional[float]:
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="5 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        except Exception:
            logger.warning("Historical data request failed for %s", contract.symbol, exc_info=True)
            return None
        for bar in reversed(bars or []):
            try:
                close = float(bar.close)
                if close > 0 and not math.isnan(close):
                    return close
            except (TypeError, ValueError):
                continue
        return None

    # ── Quantity / pricing helpers ───────────────────────────────
    def _calc_quantity(self, price: float, cash: float) -> int:
        if price <= 0:
            return 0
        max_value = cash * config.MAX_POSITION_PCT
        cap = getattr(config, "MAX_TRADE_VALUE", None)
        if cap is not None:
            max_value = min(max_value, float(cap))
        return max(int(max_value / price), 0)

    def _limit_price(self, action: str, price: float) -> float:
        offset = float(config.LIMIT_ORDER_OFFSET_PCT)
        if action == "BUY":
            return round(price * (1 + offset), 2)
        return round(price * (1 - offset), 2)

    def _initial_stop_price(self, action: str, price: float) -> float:
        if action == "BUY":
            return round(price * (1 - config.STOP_LOSS_PCT), 2)
        return round(price * (1 + config.STOP_LOSS_PCT), 2)

    def _take_profit_price(self, action: str, price: float) -> float:
        if action == "BUY":
            return round(price * (1 + config.TAKE_PROFIT_PCT), 2)
        return round(price * (1 - config.TAKE_PROFIT_PCT), 2)

    def _validate_bracket_prices(self, action: str, limit_price: float, stop_price: float, profit_price: float) -> bool:
        if action == "BUY":
            return stop_price < limit_price < profit_price
        if action == "SELL":
            return profit_price < limit_price < stop_price
        return False

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

    def _place_open_trailing_exit(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
    ) -> bool:
        """Open with a limit parent and a trailing stop child.

        This matches the desired behavior: be wrong small via the initial stop,
        but let correct trades keep running until price reverses by the trailing
        percent from its high. Kept long/short-aware, although shorts are disabled
        by default in config.
        """
        contract = self._contract(symbol)
        limit_price = self._limit_price(action, price)
        stop_price = self._initial_stop_price(action, price)
        trailing_pct = round(float(config.TRAILING_STOP_PCT) * 100.0, 4)

        parent = LimitOrder(action, qty, limit_price)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False

        exit_action = "SELL" if action == "BUY" else "BUY"
        trailing_stop = Order(
            action=exit_action,
            orderType="TRAIL",
            totalQuantity=qty,
            parentId=parent.orderId,
            trailingPercent=trailing_pct,
            trailStopPrice=stop_price,
            transmit=True,
        )
        trailing_stop.orderId = self.ib.client.getReqId()

        trades = [
            self.ib.placeOrder(contract, parent),
            self.ib.placeOrder(contract, trailing_stop),
        ]
        self.ib.sleep(1)
        statuses = [_order_status_name(t) for t in trades]
        accepted = all(_is_order_accepted(status) for status in statuses)
        logger.info(
            "Open trailing | %s %s x%d limit=%.2f | initial_stop=%.2f trail=%.4f%% | conf=%.2f | statuses=%s accepted=%s",
            action, symbol, qty, limit_price, stop_price, trailing_pct, confidence, statuses, accepted,
        )
        return accepted

    def _place_open_bracket(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
    ) -> bool:
        if bool(getattr(config, "USE_TRAILING_EXIT", False)):
            return self._place_open_trailing_exit(symbol, action, qty, price, confidence)

        contract = self._contract(symbol)
        limit_price = self._limit_price(action, price)
        stop_price = self._initial_stop_price(action, price)
        profit_price = self._take_profit_price(action, price)

        if not self._validate_bracket_prices(action, limit_price, stop_price, profit_price):
            logger.warning(
                "Invalid bracket prices | %s %s limit=%.2f stop=%.2f profit=%.2f",
                action, symbol, limit_price, stop_price, profit_price,
            )
            return False

        bracket = self.ib.bracketOrder(
            action=action,
            quantity=qty,
            limitPrice=limit_price,
            takeProfitPrice=profit_price,
            stopLossPrice=stop_price,
        )

        trades = [self.ib.placeOrder(contract, order) for order in bracket]
        self.ib.sleep(1)
        statuses = [_order_status_name(t) for t in trades]
        accepted = all(_is_order_accepted(status) for status in statuses)
        logger.info(
            "Open bracket | %s %s x%d limit=%.2f | SL=%.2f TP=%.2f | conf=%.2f | statuses=%s accepted=%s",
            action, symbol, qty, limit_price, stop_price, profit_price, confidence, statuses, accepted,
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

        if self.has_working_order(symbol):
            logger.info("Working order already exists for %s — skipping %s", symbol, action)
            return False

        position = self.get_position(symbol)
        price = self.get_price(
            symbol,
            allow_historical=bool(getattr(config, "ALLOW_HISTORICAL_PRICE_FOR_ORDERS", False)),
        )
        if price <= 0:
            logger.warning("Could not get valid order price for %s", symbol)
            return False

        if action == "BUY":
            if position > 0:
                logger.info("Already long %s position=%.2f — skipping BUY", symbol, position)
                return False
            if position < 0:
                qty = int(abs(position))
                return self._place_limit_order(symbol, "BUY", qty, price, "Close short")

            if not risk_state.can_open_more():
                logger.warning("Max daily trades (%d) reached — skipping BUY %s", config.MAX_DAILY_TRADES, symbol)
                return False

            cash = self.get_cash()
            if cash < min_cash:
                logger.warning("Insufficient cash for BUY %s: $%.2f", symbol, cash)
                return False
            qty = self._calc_quantity(price, cash)
            if qty == 0:
                logger.warning("Qty=0 for BUY %s price=%.2f cash=%.2f", symbol, price, cash)
                return False
            accepted = self._place_open_bracket(symbol, "BUY", qty, price, signal.confidence)
            if accepted:
                trades = risk_state.record_trade()
                logger.info("Recorded daily trade #%d for %s", trades, symbol)
            return accepted

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

        if not risk_state.can_open_more():
            logger.warning("Max daily trades (%d) reached — skipping short SELL %s", config.MAX_DAILY_TRADES, symbol)
            return False

        cash = self.get_cash()
        if cash < min_cash:
            logger.warning("Insufficient cash for short SELL %s: $%.2f", symbol, cash)
            return False
        qty = self._calc_quantity(price, cash)
        if qty == 0:
            logger.warning("Qty=0 for short SELL %s price=%.2f cash=%.2f", symbol, price, cash)
            return False
        accepted = self._place_open_bracket(symbol, "SELL", qty, price, signal.confidence)
        if accepted:
            trades = risk_state.record_trade()
            logger.info("Recorded daily trade #%d for %s", trades, symbol)
        return accepted

    def execute_all(self, signals: List[Signal]) -> dict:
        open_symbols = {
            p.contract.symbol.upper()
            for p in self.ib.positions()
            if float(p.position) != 0.0
        }
        working_symbols = self.working_order_symbols()
        occupied_symbols = open_symbols | working_symbols
        planned_new_symbols: set = set()
        placed, skipped = 0, 0
        allow_short = bool(config.ALLOW_SHORT)

        for signal in signals:
            symbol = signal.symbol.upper().strip()
            action = signal.action.upper().strip()

            if action == "HOLD":
                skipped += 1
                continue

            if symbol in working_symbols:
                logger.info("Working order already exists for %s — skipping", symbol)
                skipped += 1
                continue

            current_position = self.get_position(symbol)
            opens_new = (
                (action == "BUY" and current_position == 0)
                or (action == "SELL" and current_position == 0 and allow_short)
            )

            if opens_new:
                planned_total = len(occupied_symbols | planned_new_symbols)
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
