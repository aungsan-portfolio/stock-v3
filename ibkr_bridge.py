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
import datetime as _dt
import logging
import math
from typing import List, Optional

from ib_insync import IB, Stock, LimitOrder, MarketOrder, Contract, Order

import config
import live_invariants
import order_audit
import order_exec
import risk_state
from predictor import Signal

logger = logging.getLogger(__name__)

# "Working" = live at the broker but not yet a position; used by the duplicate-
# entry / open-order guards. Fill vs. acceptance classification now lives in
# order_exec (single source of truth) so "accepted" can never be mistaken for a
# fill (H4).
WORKING_ORDER_STATUSES = {"PendingSubmit", "ApiPending", "PreSubmitted", "Submitted"}

# ── Live-readiness capability flags ───────────────────────────────────────────
# These advertise which live-trading safety capabilities are ACTUALLY IMPLEMENTED
# in this bridge. They start False and each is flipped to True ONLY by the phase
# that builds the backing logic (see reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md).
# The `live-readiness` command reads them to produce an honest go-live scorecard.
# Flipping one True without its implementation is a deliberate footgun — do not.
SUPPORTS_FILL_VERIFICATION       = True    # Phase 2 (H4, H5): wait for real fill, not "accepted"
SUPPORTS_PARTIAL_FILL_HANDLING   = True    # Phase 2 (H6): size children from actual filled qty
SUPPORTS_PROTECTIVE_CHILD_VERIFY = True    # Phase 2 (H7): confirm stop child is live or flatten
SUPPORTS_SERVER_SIDE_GTC_STOP    = False   # Phase 3 (C2, H19): resting GTC/OCA hard stop per entry
SUPPORTS_DAILY_LOSS_KILLSWITCH   = False   # Phase 3 (H1): loss_breached() wired into the order gate
SUPPORTS_REALTIME_DATA_GUARD     = False   # Phase 4 (H12, H13): require real-time, reject delayed
SUPPORTS_MARKET_HOURS_GATE       = False   # Phase 4 (H15): refuse orders outside RTH/holidays
SUPPORTS_STARTUP_RECONCILIATION  = False   # Phase 5 (H18): broker = source of truth on startup
SUPPORTS_ACCOUNT_TYPE_ASSERTION  = False   # Phase 6: assert paper(DU)/live(U) account, not just port


def _order_status_name(trade) -> str:
    return str(getattr(getattr(trade, "orderStatus", None), "status", "UNKNOWN"))


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

    # ── Fill-driven order primitives (Phase 2: H4-H9) ────────────────────────
    @staticmethod
    def _today() -> str:
        return _dt.date.today().isoformat()

    def _order_wait(self, poll: float) -> None:
        """Yield to the event loop so fills/acks can arrive. Prefers the
        event-driven waitOnUpdate (returns early on any update) and falls back to
        sleep. Never raises into the order path."""
        try:
            self.ib.waitOnUpdate(timeout=poll)
        except Exception:
            try:
                self.ib.sleep(poll)
            except Exception:
                pass

    def _await_order_outcome(self, trade, timeout: Optional[float] = None,
                             poll: Optional[float] = None) -> order_exec.OrderResult:
        """Wait for `trade` to reach a terminal state, then classify the REAL
        outcome (H4/H5). Replaces the old fixed `ib.sleep(1)` + "accepted" check.

        Bounded by `ORDER_FILL_TIMEOUT_SECONDS`; a trade still working when the
        budget is exhausted is returned WORKING/TIMEOUT (or PARTIALLY_FILLED if
        some shares filled). Iteration-bounded so it is deterministic offline.
        """
        if timeout is None:
            timeout = float(getattr(config, "ORDER_FILL_TIMEOUT_SECONDS", 8.0))
        if poll is None:
            poll = float(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        steps = max(1, int(math.ceil(timeout / poll))) if poll > 0 else 1
        for _ in range(steps):
            if order_exec.is_terminal(trade):
                break
            self._order_wait(poll)
        if order_exec.is_terminal(trade):
            return order_exec.classify_trade(trade)
        return order_exec.outcome_on_timeout(trade)

    def _working_orders_plain(self) -> list:
        """Refresh open orders from the broker and adapt to plain dicts (with
        parent_id/qty/status) for the pure order_exec verifiers."""
        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        except Exception:
            logger.warning("Could not refresh open orders", exc_info=True)
        return live_invariants.adapt_working_orders(self.ib.openTrades())

    def _cancel_symbol_working_orders(self, symbol: str) -> int:
        """Cancel ALL working orders for `symbol` (any side / type). Used before
        an emergency flatten so no resting exit child can fire AFTER we go flat
        and open an accidental short. Returns the number cancelled."""
        symbol = symbol.upper().strip()
        n = 0
        for t in self.ib.openTrades():
            if str(getattr(t.contract, "symbol", "")).upper().strip() != symbol:
                continue
            try:
                self.ib.cancelOrder(t.order)
                n += 1
            except Exception:
                logger.warning("Failed to cancel working order for %s", symbol, exc_info=True)
        if n:
            self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        return n

    def _market_close_symbol(self, symbol: str) -> dict:
        """Emergency-flatten one symbol (H7 fallback when exit protection cannot
        be made safe). Cancels EVERY working order for the symbol first -- so no
        resting child can execute after we go flat (which would open an accidental
        short) -- then market-closes the whole position."""
        symbol = symbol.upper().strip()
        self._cancel_symbol_working_orders(symbol)
        position = self.get_position(symbol)
        qty = int(abs(position))
        if qty <= 0:
            return {"symbol": symbol, "qty": 0, "outcome": "flat"}
        action = "SELL" if position > 0 else "BUY"
        try:
            contract = self._contract(symbol)
            trade = self.ib.placeOrder(contract, MarketOrder(action, qty))
            res = self._await_order_outcome(trade)
        except Exception:
            logger.error("Emergency close failed for %s", symbol, exc_info=True)
            return {"symbol": symbol, "qty": qty, "outcome": "error"}
        order_audit.log_event(
            order_audit.STAGE_FLATTEN, phase="unprotected_close", symbol=symbol,
            action=action, qty=qty, outcome=res.outcome, filled=res.filled,
        )
        logger.warning("Emergency close %s %s x%d -> %s (filled=%.0f)",
                       action, symbol, qty, res.outcome, res.filled)
        return {"symbol": symbol, "qty": qty, "action": action,
                "outcome": res.outcome, "filled": res.filled}

    def _cancel_oversized_exit_children(self, symbol: str, exit_action: str, filled: float) -> int:
        """Cancel EVERY working exit (``exit_action``) order for `symbol` whose
        quantity exceeds the actual filled qty -- protective stop, trailing stop,
        AND take-profit LIMIT alike (H6). After this no resting child can sell
        more than we hold, so a partial fill cannot leave an accidental-short
        trap. Returns the number cancelled."""
        symbol = symbol.upper().strip()
        exit_action = exit_action.upper().strip()
        cancelled = 0
        for t in self.ib.openTrades():
            if str(getattr(t.contract, "symbol", "")).upper().strip() != symbol:
                continue
            if str(getattr(t.order, "action", "")).upper().strip() != exit_action:
                continue
            cqty = float(getattr(t.order, "totalQuantity", 0) or 0)
            if order_exec.child_needs_resize(cqty, filled):  # cqty > filled
                try:
                    self.ib.cancelOrder(t.order)
                    cancelled += 1
                except Exception:
                    logger.warning("Failed to cancel oversized exit child for %s", symbol, exc_info=True)
        if cancelled:
            self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        return cancelled

    def _place_protective_stop(self, symbol: str, filled: float, stop_price: float,
                               exit_action: str = "SELL") -> bool:
        """Place a protective STP sized to the ACTUAL filled qty (H6). Returns
        True only when a correctly-sized protective stop is confirmed working."""
        symbol = symbol.upper().strip()
        target_qty = order_exec.protective_exit_qty(filled)
        if target_qty <= 0:
            return False
        try:
            contract = self._contract(symbol)
            stop = Order(action=exit_action, orderType="STP", totalQuantity=target_qty,
                         auxPrice=round(float(stop_price), 2), transmit=True)
            stop.orderRef = order_exec.deterministic_order_ref(self._today(), symbol, exit_action + "_STOP")
            self.ib.placeOrder(contract, stop)
            self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        except Exception:
            logger.error("Failed to place protective stop for %s", symbol, exc_info=True)
            return False
        working = self._working_orders_plain()
        verdict = order_exec.verify_protective_child(symbol, 0, target_qty, working)
        order_audit.log_event(
            order_audit.STAGE_STOP_PLACED, symbol=symbol, action=exit_action,
            qty=target_qty, stop_price=round(float(stop_price), 2),
            ok=verdict["ok"], reason=verdict["reason"],
        )
        return bool(verdict["ok"])

    def _verify_or_protect(self, symbol: str, parent_order, result: order_exec.OrderResult,
                           stop_price: float, exit_action: str = "SELL") -> order_exec.OrderResult:
        """After a long parent OPEN fills (full or partial), guarantee ONE of two
        safe end states (H6/H7):

          (a) a live protective stop covers the ACTUAL filled qty AND no resting
              exit child can sell more than we hold (no accidental short), or
          (b) the position is flat (emergency-flattened; result marked aborted).

        This covers BOTH the protective stop/trailing child AND the take-profit
        LIMIT child: on a partial fill every oversized exit order is cancelled,
        then a stop sized to the filled qty is (re)placed and verified, and a
        final safety gate refuses any leftover oversized exit. Mutates `result`.
        """
        filled = order_exec.protective_exit_qty(result.filled)
        parent_id = getattr(parent_order, "orderId", 0)

        # 1) On a partial fill, stop chasing the remainder so the held qty is final.
        if result.outcome == order_exec.PARTIALLY_FILLED:
            try:
                self.ib.cancelOrder(parent_order)
                self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
            except Exception:
                logger.warning("Could not cancel partial parent for %s", symbol, exc_info=True)

        # 2) Cancel EVERY oversized exit child (stop / trailing / take-profit limit)
        #    so nothing resting can sell more than the filled qty.
        self._cancel_oversized_exit_children(symbol, exit_action, filled)

        # 3) Ensure a protective stop covers the actual filled qty.
        working = self._working_orders_plain()
        verdict = order_exec.verify_protective_child(symbol, parent_id, filled, working)
        if not verdict["ok"]:
            if self._place_protective_stop(symbol, filled, stop_price, exit_action):
                working = self._working_orders_plain()
                verdict = order_exec.verify_protective_child(symbol, parent_id, filled, working)

        # 4) Safety gate: a covering stop exists AND no exit child oversells.
        oversized = order_exec.oversized_exit_children(symbol, exit_action, filled, working)
        if (not verdict["ok"]) or oversized:
            reason = verdict["reason"] if not verdict["ok"] else "oversized_exit_child"
            order_audit.log_event(
                order_audit.STAGE_STOP_CONFIRMED, symbol=symbol, ok=False,
                reason=reason, n_oversized=len(oversized), action="emergency_flatten",
            )
            logger.error("Unsafe exit protection for filled %s (stop_ok=%s, oversized=%d) -> emergency flatten",
                         symbol, verdict["ok"], len(oversized))
            self._market_close_symbol(symbol)
            result.protective_ok = False
            result.aborted = True
            return result

        result.protective_ok = True
        order_audit.log_event(
            order_audit.STAGE_STOP_CONFIRMED, symbol=symbol, ok=True,
            qty=filled, stop_price=round(float(stop_price), 2),
        )
        return result

    def _finalize_open(self, symbol: str, action: str, parent_trade, result: order_exec.OrderResult,
                       stop_price: float, intended_qty: int) -> order_exec.OrderResult:
        """Audit the verified open outcome and, for a real long fill, verify the
        protective child stop (H6/H7). Returns the (possibly aborted) result."""
        if result.outcome == order_exec.FILLED:
            order_audit.log_event(
                order_audit.STAGE_FILLED, symbol=symbol, action=action, filled=result.filled,
                avg_fill_price=result.avg_fill_price, intended_qty=intended_qty,
            )
        elif result.outcome == order_exec.PARTIALLY_FILLED:
            order_audit.log_event(
                order_audit.STAGE_PARTIAL, symbol=symbol, action=action, filled=result.filled,
                remaining=result.remaining, avg_fill_price=result.avg_fill_price, intended_qty=intended_qty,
            )
        else:
            order_audit.log_event(
                order_audit.STAGE_REJECTED if result.outcome == order_exec.REJECTED else order_audit.STAGE_ACK,
                symbol=symbol, action=action, outcome=result.outcome, status=result.status,
                intended_qty=intended_qty,
            )
            logger.info("Open %s %s x%d -> %s (no fill)", action, symbol, intended_qty, result.outcome)
            return result

        # A real (full/partial) fill. Long entries must be backed by a live stop.
        if action == "BUY":
            exit_action = "SELL"
            result = self._verify_or_protect(symbol, parent_trade.order, result, stop_price, exit_action)
        logger.info(
            "Open %s %s -> %s filled=%.0f avg=%.2f protective_ok=%s aborted=%s",
            action, symbol, result.outcome, result.filled, result.avg_fill_price,
            result.protective_ok, result.aborted,
        )
        return result

    # ── Order placement ──────────────────────────────────────────
    def _close_position(self, symbol: str, action: str, qty: int, price: float,
                        note: str) -> order_exec.OrderResult:
        """Robustly close `qty` shares (H8): marketable-limit first, then escalate
        to a market order, confirming each attempt by fill until remaining == 0 or
        attempts are exhausted. Replaces the old single bare-limit + one-check
        close. Never silently treats an unconfirmed close as flat."""
        symbol = symbol.upper().strip()
        if qty <= 0:
            logger.warning("Invalid close qty=%s for %s %s", qty, action, symbol)
            return order_exec.OrderResult(outcome=order_exec.REJECTED, status="bad_qty")

        contract = self._contract(symbol)
        order_ref = order_exec.deterministic_order_ref(self._today(), symbol, action)
        max_attempts = max(1, int(getattr(config, "CLOSE_MAX_ATTEMPTS", 3)))
        remaining = int(qty)
        total_filled = 0.0
        last = order_exec.OrderResult(outcome=order_exec.WORKING)

        for attempt in range(1, max_attempts + 1):
            if attempt == 1 and price > 0:
                limit_price = self._limit_price(action, price)
                order = LimitOrder(action, remaining, limit_price)
                kind, px = "close_limit", limit_price
            else:
                order = MarketOrder(action, remaining)  # escalate to certainty of exit
                kind, px = "close_market", 0.0
            order.orderRef = order_ref
            order_audit.log_event(
                order_audit.STAGE_SUBMIT, kind=kind, note=note, symbol=symbol,
                action=action, qty=remaining, attempt=attempt, limit_price=px,
            )
            trade = self.ib.placeOrder(contract, order)
            res = self._await_order_outcome(trade)
            total_filled += res.filled
            last = res

            decision = order_exec.close_followup(res, attempt, max_attempts)
            stage = (order_audit.STAGE_FILLED if res.outcome == order_exec.FILLED
                     else order_audit.STAGE_PARTIAL if res.outcome == order_exec.PARTIALLY_FILLED
                     else order_audit.STAGE_REJECTED)
            order_audit.log_event(
                stage, kind=kind, note=note, symbol=symbol, action=action, attempt=attempt,
                outcome=res.outcome, filled=res.filled, remaining=res.remaining, decision=decision,
            )
            logger.info("%s | %s %s attempt=%d -> %s filled=%.0f remaining=%.0f (%s)",
                        note, action, symbol, attempt, res.outcome, res.filled, res.remaining, decision)

            if decision == "done":
                break
            if decision == "giveup":
                logger.error("Close %s %s NOT flat after %d attempts (remaining=%.0f) -> ALERT",
                             action, symbol, attempt, res.remaining)
                break
            remaining = max(int(round(res.remaining)), 0)
            if remaining <= 0:
                break

        last.filled = total_filled
        return last

    def _place_open_trailing_exit(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
    ) -> order_exec.OrderResult:
        """Open with a limit parent and a trailing-stop child, then CONFIRM the
        parent fill before treating the entry as real (H4/H5) and verify the
        trailing child is live (H7). Long/short-aware; shorts are disabled by
        default in config and skip the long-only protective check.
        """
        contract = self._contract(symbol)
        limit_price = self._limit_price(action, price)
        stop_price = self._initial_stop_price(action, price)
        trailing_pct = round(float(config.TRAILING_STOP_PCT) * 100.0, 4)

        parent = LimitOrder(action, qty, limit_price)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False
        parent.orderRef = order_exec.deterministic_order_ref(self._today(), symbol, action)

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

        order_audit.log_event(
            order_audit.STAGE_SUBMIT, kind="open_trailing", symbol=symbol, action=action,
            qty=qty, limit_price=limit_price, initial_stop=stop_price, trail_pct=trailing_pct,
            order_ref=parent.orderRef,
        )
        parent_trade = self.ib.placeOrder(contract, parent)
        self.ib.placeOrder(contract, trailing_stop)
        result = self._await_order_outcome(parent_trade)
        return self._finalize_open(symbol, action, parent_trade, result, stop_price, qty)

    def _place_open_bracket(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
    ) -> order_exec.OrderResult:
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
            return order_exec.OrderResult(outcome=order_exec.REJECTED, status="bad_prices")

        bracket = self.ib.bracketOrder(
            action=action,
            quantity=qty,
            limitPrice=limit_price,
            takeProfitPrice=profit_price,
            stopLossPrice=stop_price,
        )
        order_ref = order_exec.deterministic_order_ref(self._today(), symbol, action)
        bracket[0].orderRef = order_ref  # tag the parent for idempotency/reconcile

        order_audit.log_event(
            order_audit.STAGE_SUBMIT, kind="open_bracket", symbol=symbol, action=action,
            qty=qty, limit_price=limit_price, stop_price=stop_price, profit_price=profit_price,
            order_ref=order_ref,
        )
        trades = [self.ib.placeOrder(contract, order) for order in bracket]
        parent_trade = trades[0]
        result = self._await_order_outcome(parent_trade)
        return self._finalize_open(symbol, action, parent_trade, result, stop_price, qty)

    # ── Public API ───────────────────────────────────────────────
    def _duplicate_ref_working(self, symbol: str, action: str) -> bool:
        """Idempotency guard (H9): refuse to place an OPEN whose deterministic
        orderRef (date:symbol:action) is already working at the broker -- e.g. a
        duplicate after a restart / mid-flight. Complements has_working_order with
        intent-level matching and lays the groundwork for Phase 5 reconciliation.
        """
        ref = order_exec.deterministic_order_ref(self._today(), symbol, action)
        try:
            working = self._working_orders_plain()
        except Exception:
            return False
        return order_exec.has_order_ref(ref, working)

    def _record_if_filled(self, symbol: str, action: str, result: order_exec.OrderResult) -> None:
        """Record a daily trade ONLY for a real, held fill (H4/H6) -- never on a
        bare acknowledgement, rejection, timeout, or emergency-aborted entry."""
        if not order_exec.should_record_trade(result):
            return
        trades = risk_state.record_trade()
        order_audit.log_event(
            order_audit.STAGE_TRADE_RECORDED, symbol=symbol, action=action,
            qty=order_exec.protective_exit_qty(result.filled),
            avg_fill_price=result.avg_fill_price, daily_trade_count=trades, on="fill",
        )
        logger.info("Recorded daily trade #%d for %s (filled=%.0f)", trades, symbol, result.filled)

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
                # Cover an existing short: a robust, fill-confirmed close (H8).
                return self._close_position(symbol, "BUY", qty, price, "Close short").has_fill

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
            if self._duplicate_ref_working(symbol, "BUY"):
                logger.info("Deterministic orderRef already working for %s BUY — skipping (idempotency)", symbol)
                return False
            result = self._place_open_bracket(symbol, "BUY", qty, price, signal.confidence)
            self._record_if_filled(symbol, "BUY", result)
            return result.occupies_slot

        # action == "SELL"
        if position > 0:
            qty = int(position)
            # Close an existing long: a robust, fill-confirmed close (H8).
            return self._close_position(symbol, "SELL", qty, price, "Close long").has_fill
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
        if self._duplicate_ref_working(symbol, "SELL"):
            logger.info("Deterministic orderRef already working for %s SELL — skipping (idempotency)", symbol)
            return False
        result = self._place_open_bracket(symbol, "SELL", qty, price, signal.confidence)
        self._record_if_filled(symbol, "SELL", result)
        return result.occupies_slot

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

    # ── Emergency flatten (kill-switch) ──────────────────────────────────────
    def flatten_all(self, confirm: bool = False) -> dict:
        """Emergency kill-switch: cancel every working order and market-close
        every open position. PAPER-port-locked via connect() like all order
        paths.

        confirm=False (default) is a DRY-RUN: it reports exactly what it would
        cancel/flatten and places nothing. confirm=True actually cancels orders
        and submits MarketOrders to flatten. Returns a structured plan/result
        dict so callers (the `panic-flatten` command, later graceful shutdown)
        can print and verify it.

        Uses market orders on purpose: in an emergency, certainty of exit beats
        price. Handles longs (SELL to close) and any shorts (BUY to cover).
        """
        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(1)
        except Exception:
            logger.warning("Could not refresh open orders before flatten", exc_info=True)

        open_trades = list(self.ib.openTrades())
        positions = [p for p in self.ib.positions() if float(p.position) != 0.0]

        plan = {
            "confirm": bool(confirm),
            "orders_to_cancel": [
                {
                    "symbol": str(getattr(t.contract, "symbol", "")),
                    "action": str(getattr(t.order, "action", "")),
                    "order_type": str(getattr(t.order, "orderType", "")),
                    "qty": float(getattr(t.order, "totalQuantity", 0) or 0),
                    "order_id": int(getattr(t.order, "orderId", 0) or 0),
                    "status": _order_status_name(t),
                }
                for t in open_trades
                if _is_order_working(_order_status_name(t))
            ],
            "positions_to_flatten": [
                {
                    "symbol": str(p.contract.symbol),
                    "qty": float(p.position),
                    "action": "SELL" if float(p.position) > 0 else "BUY",
                }
                for p in positions
            ],
            "cancelled": 0,
            "flattened": 0,
            "flatten_results": [],
        }

        order_audit.log_event(
            order_audit.STAGE_FLATTEN, phase="plan", confirm=bool(confirm),
            n_orders=len(plan["orders_to_cancel"]), n_positions=len(plan["positions_to_flatten"]),
        )

        if not confirm:
            return plan  # dry-run: nothing placed

        # 1) Cancel all working orders (protective children included; they are
        #    redundant once we market-close the underlying position).
        for t in open_trades:
            if not _is_order_working(_order_status_name(t)):
                continue
            try:
                self.ib.cancelOrder(t.order)
                plan["cancelled"] += 1
            except Exception:
                logger.warning("Failed to cancel order id=%s", getattr(t.order, "orderId", "?"), exc_info=True)
        self.ib.sleep(1)

        # 2) Market-close every open position.
        for p in positions:
            symbol = str(p.contract.symbol).upper().strip()
            qty = int(abs(float(p.position)))
            if qty <= 0:
                continue
            action = "SELL" if float(p.position) > 0 else "BUY"
            try:
                contract = self._contract(symbol)
                trade = self.ib.placeOrder(contract, MarketOrder(action, qty))
                # Confirm the emergency close by fill, not a fixed sleep (H8).
                res = self._await_order_outcome(trade)
                result = {
                    "symbol": symbol, "action": action, "qty": qty, "status": res.status,
                    "outcome": res.outcome, "filled": res.filled, "avg_fill": res.avg_fill_price,
                }
                plan["flatten_results"].append(result)
                plan["flattened"] += 1
                order_audit.log_event(order_audit.STAGE_FLATTEN, phase="executed", **result)
                logger.info("Flatten %s %s x%d -> %s filled=%.0f", action, symbol, qty, res.outcome, res.filled)
            except Exception:
                logger.warning("Failed to flatten %s", symbol, exc_info=True)
        self.ib.sleep(1)
        return plan
