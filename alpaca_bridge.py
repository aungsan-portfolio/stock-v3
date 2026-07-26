"""
alpaca_bridge.py — Alpaca Paper Trading bridge via alpaca-py.
Duck-type replacement for IBKRBridge. Exposes all pricing, account,
placement, and reconciliation methods expected by Pro V3.
"""
import datetime as _dt
import logging
import math
import os
import time
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest, MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import config as config_module
import order_exec
import live_invariants
from data_integrity import QuoteDecision
import reconciliation
import order_audit
import alerts
import paper_ledger
import shutdown_guard
from predictor import Signal

logger = logging.getLogger(__name__)

# Alpaca status to IBKR status mappings
status_map = {
    "new": "Submitted",
    "accepted": "Submitted",
    "partially_filled": "Submitted",
    "filled": "Filled",
    "canceled": "Cancelled",
    "rejected": "Rejected",
    "expired": "Inactive"
}


class _Contract:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper().strip()


class _Position:
    def __init__(self, symbol: str, qty: float, avg_cost: float, market_value: float = 0.0, unrealized_pnl: float = 0.0):
        self.contract = _Contract(symbol)
        self.position = qty
        self.avgCost = avg_cost
        self.marketValue = market_value
        self.unrealizedPNL = unrealized_pnl


class _Bar:
    def __init__(self, b):
        self.date = b.timestamp
        self.open = float(b.open)
        self.high = float(b.high)
        self.low = float(b.low)
        self.close = float(b.close)
        self.volume = float(b.volume)


class MockPosition:
    def __init__(self, symbol: str, qty: float, avg_cost: float, market_value: float, unrealized_pnl: float):
        self.contract = _Contract(symbol)
        self.position = qty
        self.avgCost = avg_cost
        self.marketValue = market_value
        self.unrealizedPNL = unrealized_pnl


class MockOrder:
    def __init__(self, order_id: str, symbol: str, action: str, qty: float, order_type: str, limit_price: float, status: str, client_order_id: str):
        self.id = order_id
        self.contract = _Contract(symbol)
        self.action = action
        self.totalQuantity = qty
        self.orderType = order_type
        self.lmtPrice = limit_price
        self.orderState = type("State", (), {"status": status})()
        self.orderRef = client_order_id
        self.order = self
        self.orderStatus = type("OrderStatus", (), {
            "status": status,
            "filled": qty if status == "Filled" else 0.0,
            "remaining": 0.0 if status == "Filled" else qty,
            "avgFillPrice": limit_price
        })()


class MockIB:
    def __init__(self, bridge):
        self.bridge = bridge

    def positions(self) -> List[MockPosition]:
        self.bridge._require_connection()
        try:
            al_positions = self.bridge._client.get_all_positions()
            res = []
            for p in al_positions:
                qty = float(p.qty)
                avg_cost = float(p.avg_entry_price)
                mkt_val = float(p.market_value)
                upnl = float(p.unrealized_pl)
                res.append(MockPosition(p.symbol, qty, avg_cost, mkt_val, upnl))
            return res
        except Exception as exc:
            logger.error("MockIB.positions() failed: %s", exc)
            self.bridge._conn_health.mark_unhealthy(f"MockIB.positions error: {exc}")
            raise RuntimeError(f"Alpaca API outage while fetching positions: {exc}")

    def openTrades(self) -> List[MockOrder]:
        try:
            al_orders = self.bridge._get_active_orders()
            res = []
            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated"}
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() not in active_statuses:
                    continue
                action = "BUY" if o.side.value.upper() == "BUY" else "SELL"
                qty = float(o.qty)
                status = status_map.get(raw_status.lower(), "Submitted")
                lmt_price = float(o.limit_price) if o.limit_price is not None else 0.0
                order_type = "LMT" if o.type.value.upper() == "LIMIT" else "MKT"
                res.append(MockOrder(str(o.id), o.symbol, action, qty, order_type, lmt_price, status, o.client_order_id))
            return res
        except Exception as exc:
            logger.error("MockIB.openTrades() failed: %s", exc)
            return []

    def reqOpenOrders(self) -> List[MockOrder]:
        # Returns raw order objects for get_portfolio.py
        return [trade.order for trade in self.openTrades()]

    def reqAllOpenOrders(self) -> None:
        pass

    def sleep(self, sec: float) -> None:
        time.sleep(sec)


class ConnectionHealth:
    def __init__(self):
        self.is_healthy = True

    def mark_healthy(self, reason=None):
        self.is_healthy = True

    def mark_unhealthy(self, reason=None):
        self.is_healthy = False


class EntryGateMock:
    def __init__(self):
        self.halted = False

    def halt(self):
        self.halted = True

    def risk_sized_qty(self, symbol, signal, price, qty):
        # By default, pass-through without Minervini stage resizing
        return qty

    def entry_blocked(self, symbol, value, signal, price):
        return False, ""


class AlpacaBridge:
    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or config_module.get_settings()
        self._client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None
        self._connected = False
        self._conn_health = ConnectionHealth()
        self.ib = MockIB(self)
        self.entry_gate = EntryGateMock()

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _c(self, name, default=None):
        return getattr(self._cfg, name, default)

    def _require_connection(self):
        if not self._connected or self._client is None:
            raise RuntimeError("AlpacaBridge is not connected")

    # ── Connection ───────────────────────────────────────────────
    def connect(self) -> bool:
        key = os.environ.get("APCA_API_KEY_ID", "")
        secret = os.environ.get("APCA_API_SECRET_KEY", "")
        if not key or not secret:
            logger.error("Alpaca credentials missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars")
            self._conn_health.mark_unhealthy("Credentials missing")
            return False

        endpoint = os.environ.get("APCA_API_BASE_URL", self._c("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")).rstrip("/")
        if endpoint != "https://paper-api.alpaca.markets":
            logger.error("Safety block: APCA_API_BASE_URL must be paper endpoint (https://paper-api.alpaca.markets)")
            self._conn_health.mark_unhealthy("Non-paper base URL detected")
            raise RuntimeError("Live trading is disabled. Endpoint must be https://paper-api.alpaca.markets")

        try:
            self._client = TradingClient(key, secret, paper=True)
            self._data_client = StockHistoricalDataClient(key, secret)

            # Validate account status
            acct = self._client.get_account()
            if getattr(acct, "status", None) != "ACTIVE" or getattr(acct, "account_blocked", False) or getattr(acct, "trading_blocked", False):
                logger.error("Alpaca account is not active or is blocked. Status: %s", getattr(acct, "status", "UNKNOWN"))
                self._conn_health.mark_unhealthy("Account blocked")
                return False

            logger.info("Connected to Alpaca Paper Account | Status: %s | Equity: %s", acct.status, acct.equity)
            from strategies.session import now_eastern
            et_date = str(now_eastern().date())
            state = self._load_daytrade_risk_state()
            if state.get("date") == et_date:
                self._start_of_day_equity = float(state.get("start_of_day_equity", acct.equity))
                self._daytrade_suspended = bool(state.get("suspended", False))
                logger.info("Restored start-of-day equity: $%.2f, suspended: %s", self._start_of_day_equity, self._daytrade_suspended)
            else:
                self._start_of_day_equity = float(acct.equity)
                self._daytrade_suspended = False
                self._save_daytrade_risk_state(et_date, self._start_of_day_equity, False)
                logger.info("Initialized new day start-of-day equity: $%.2f", self._start_of_day_equity)
            self._session_start_time = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=24)
            self._connected = True
            self._conn_health.mark_healthy()
            return True
        except Exception as exc:
            logger.exception("Failed to connect to Alpaca API")
            self._conn_health.mark_unhealthy(str(exc))
            return False

    def disconnect(self) -> None:
        self._client = None
        self._data_client = None
        self._connected = False
        logger.info("Disconnected from Alpaca")

    def _load_daytrade_risk_state(self) -> dict:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'daytrade_risk_state.json')
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                import json
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load daytrade risk state: %s", e)
            return {}

    def _save_daytrade_risk_state(self, date_str: str, equity: float, suspended: bool):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'daytrade_risk_state.json')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                import json
                json.dump({
                    "date": date_str,
                    "start_of_day_equity": equity,
                    "suspended": suspended
                }, f, indent=2)
        except Exception as e:
            logger.error("Failed to save daytrade risk state: %s", e)

    def _get_active_orders(self) -> list:
        self._require_connection()
        try:
            after_time = getattr(self, "_session_start_time", None)
            if after_time is None:
                after_time = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=24)
            return self._client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                after=after_time,
                limit=500,
                direction="desc"
            ))
        except Exception as exc:
            logger.error("Failed to query active orders from broker: %s", exc)
            return []

    # ── Account Services ─────────────────────────────────────────
    def get_cash(self) -> float:
        self._require_connection()
        try:
            return float(self._client.get_account().cash)
        except Exception as exc:
            logger.error("get_cash failed: %s", exc)
            return 0.0

    def get_net_liquidation(self) -> float:
        self._require_connection()
        try:
            return float(self._client.get_account().equity)
        except Exception as exc:
            logger.error("get_net_liquidation failed: %s", exc)
            return 0.0

    def account_daily_pnl(self) -> float:
        if getattr(self, "_start_of_day_equity", 0.0) <= 0.0:
            return 0.0
        return self.get_net_liquidation() - self._start_of_day_equity

    def snapshot_start_of_day_equity(self) -> float:
        from strategies.session import now_eastern
        et_date = str(now_eastern().date())
        current_equity = self.get_net_liquidation()
        self._start_of_day_equity = current_equity
        self._save_daytrade_risk_state(et_date, current_equity, getattr(self, "_daytrade_suspended", False))
        return self._start_of_day_equity

    def get_position(self, symbol: str) -> float:
        self._require_connection()
        try:
            p = self._client.get_open_position(symbol.upper().strip())
            return float(p.qty)
        except Exception as exc:
            if "position does not exist" in str(exc).lower() or "404" in str(exc):
                return 0.0
            logger.error("get_position(%s) failed: %s", symbol, exc)
            return 0.0

    def working_order_symbols(self) -> set:
        try:
            al_orders = self._get_active_orders()
            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated"}
            symbols = set()
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() in active_statuses:
                    symbols.add(o.symbol.upper().strip())
            return symbols
        except Exception as exc:
            logger.error("working_order_symbols failed: %s", exc)
            return set()

    def has_working_order(self, symbol: str, action: Optional[str] = None) -> bool:
        try:
            al_orders = self._get_active_orders()
            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated"}
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() not in active_statuses:
                    continue
                if o.symbol.upper().strip() == symbol.upper().strip():
                    if action is None or o.side.value.upper() == action.upper():
                        return True
            return False
        except Exception as exc:
            logger.error("has_working_order failed: %s", exc)
            return False

    # ── Pricing Services ─────────────────────────────────────────
    def _contract(self, symbol: str) -> _Contract:
        return _Contract(symbol)

    def get_order_quote(self, symbol: str, timeout: float = 5.0) -> QuoteDecision:
        # Duck-type matches PricingService.get_order_quote using StockLatestQuoteRequest
        self._require_connection()
        symbol = symbol.upper().strip()
        try:
            quotes = self._data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol, feed="iex"))
            q = quotes.get(symbol)
            if q is None:
                return QuoteDecision(False, 0.0, "alpaca", "No latest quote returned")

            bid = float(getattr(q, "bid_price", 0) or 0)
            ask = float(getattr(q, "ask_price", 0) or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid or 0.0)

            if mid <= 0:
                return QuoteDecision(False, 0.0, "alpaca", "Quote mid-price is zero")

            # Check spread
            spread_pct = (ask - bid) / mid if mid > 0 else 0.0
            max_spread = float(self._c("MAX_QUOTE_SPREAD_PCT", 0.02))
            if spread_pct > max_spread:
                return QuoteDecision(False, mid, "alpaca", f"Spread {spread_pct:.4f} exceeds max {max_spread:.4f}")

            return QuoteDecision(True, mid, "alpaca", "ok")
        except Exception as exc:
            logger.error("get_order_quote failed for %s: %s", symbol, exc)
            return QuoteDecision(False, 0.0, "alpaca", str(exc))

    def get_price(self, symbol: str, timeout: float = 5.0, allow_historical: bool = True) -> float:
        decision = self.get_order_quote(symbol, timeout=timeout)
        if decision.ok and decision.price > 0:
            return decision.price
        if allow_historical:
            # Fallback to historical close
            bars = self.fetch_historical_data(symbol, durationStr="5 D", barSizeSetting="1 day")
            if bars:
                return bars[-1].close
        return 0.0

    def fetch_historical_data(self, symbol: str, durationStr: str = "5 D", barSizeSetting: str = "1 day", useRTH: bool = True, whatToShow: str = "TRADES") -> List[_Bar]:
        self._require_connection()
        try:
            tf_map = {
                "1 day": TimeFrame(1, TimeFrameUnit.Day),
                "1 hour": TimeFrame(1, TimeFrameUnit.Hour),
                "5 mins": TimeFrame(5, TimeFrameUnit.Minute),
                "1 min": TimeFrame(1, TimeFrameUnit.Minute)
            }
            tf = tf_map.get(barSizeSetting.lower(), TimeFrame(1, TimeFrameUnit.Day))
            parts = durationStr.split()
            n = int(parts[0]) if parts else 5
            unit = parts[1].upper() if len(parts) > 1 else "D"
            days = n * {"D": 1, "W": 7, "M": 30, "Y": 365}.get(unit[0], 1)

            end = _dt.datetime.now(_dt.timezone.utc)
            start = end - _dt.timedelta(days=days)

            bars = self._data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol.upper().strip(),
                timeframe=tf,
                start=start,
                end=end,
                feed="iex"
            ))
            raw_bars = bars.data.get(symbol.upper().strip(), [])
            return [_Bar(b) for b in raw_bars]
        except Exception as exc:
            logger.error("fetch_historical_data failed for %s: %s", symbol, exc)
            return []

    # ── Sizing / Limit helpers ───────────────────────────────────
    def _calc_quantity(self, price: float, cash: float) -> int:
        if price <= 0:
            return 0
        max_value = cash * float(self._c("MAX_POSITION_PCT", 0.02))
        cap = self._c("MAX_TRADE_VALUE", None)
        if cap is not None:
            max_value = min(max_value, float(cap))
        return max(int(max_value / price), 0)

    def _limit_price(self, action: str, price: float) -> float:
        offset = float(self._c("LIMIT_ORDER_OFFSET_PCT", 0.001))
        return round(price * (1 + offset), 2) if action == "BUY" else round(price * (1 - offset), 2)

    def _initial_stop_price(self, action: str, price: float) -> float:
        sp = float(self._c("STOP_LOSS_PCT", 0.008))
        return round(price * (1 - sp), 2) if action == "BUY" else round(price * (1 + sp), 2)

    def _take_profit_price(self, action: str, price: float) -> float:
        tp = float(self._c("TAKE_PROFIT_PCT", 0.015))
        return round(price * (1 + tp), 2) if action == "BUY" else round(price * (1 - tp), 2)

    # ── Placement & Execution ────────────────────────────────────
    def _place_open_bracket(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
        strategy_name: str = "", custom_stop_price: Optional[float] = None,
        custom_target_price: Optional[float] = None
    ) -> order_exec.OrderResult:
        self._require_connection()
        limit_price = self._limit_price(action, price)
        stop_price = custom_stop_price if custom_stop_price is not None else self._initial_stop_price(action, price)
        take_profit_price = custom_target_price if custom_target_price is not None else self._take_profit_price(action, price)
        
        strat_tag = str(strategy_name or "UNKNOWN").replace("StrategyName.", "").upper()
        ts_suffix = str(int(time.time()))[-5:]
        client_order_id = f"dt_{strat_tag[:10]}_{symbol}_{action}_{ts_suffix}"[:48]

        order_audit.log_event(
            order_audit.STAGE_SUBMIT, kind="open", symbol=symbol, action=action,
            qty=qty, limit_price=limit_price, confidence=confidence, order_ref=client_order_id,
        )

        try:
            order = self._client.submit_order(LimitOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.GTC,
                limit_price=round(limit_price, 2),
                order_class=OrderClass.BRACKET,
                client_order_id=client_order_id,
                take_profit=TakeProfitRequest(limit_price=round(take_profit_price, 2)),
                stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
            ))
            
            parent_id = str(order.id)
            stop_leg_id = None
            try:
                from alpaca.trading.requests import GetOrderByIdRequest
                nested_order = self._client.get_order_by_id(parent_id, GetOrderByIdRequest(nested=True))
                legs = getattr(nested_order, "legs", []) or []
                for leg in legs:
                    leg_type = str(getattr(leg, "order_type", getattr(leg, "type", ""))).lower()
                    leg_stop = getattr(leg, "stop_price", None)
                    if "stop" in leg_type or leg_stop is not None:
                        stop_leg_id = str(leg.id)
                        logger.info(f"Extracted stop leg ID {stop_leg_id} for parent bracket order {parent_id}")
                        break
            except Exception as leg_exc:
                logger.warning(f"Could not extract nested child leg ID for {parent_id}: {leg_exc}")

            result = self._await_order_outcome(parent_id)
            if stop_leg_id:
                setattr(result, "stop_order_id", stop_leg_id)
            return self._finalize_open(symbol, action, result, qty)
        except Exception as exc:
            logger.error("submit_order failed: %s", exc)
            return order_exec.OrderResult(outcome=order_exec.REJECTED, status="Rejected", filled=0.0, remaining=float(qty))

    def _await_order_outcome(self, order_id: str, timeout: float = 10.0) -> order_exec.OrderResult:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                o = self._client.get_order_by_id(order_id)
                status = getattr(o, "status", None)
                if status is not None:
                    status = status.value if hasattr(status, "value") else str(status)
                status = str(status).lower()

                if status == "filled":
                    return order_exec.OrderResult(
                        outcome=order_exec.FILLED,
                        status="Filled",
                        filled=float(o.filled_qty),
                        remaining=0.0,
                        avg_fill_price=float(o.filled_avg_price or o.limit_price or 0.0),
                        protective_ok=True,
                        aborted=False
                    )
                elif status in {"canceled", "rejected", "expired"}:
                    outcome = order_exec.REJECTED if status == "rejected" else order_exec.CANCELLED
                    return order_exec.OrderResult(
                        outcome=outcome,
                        status=status.capitalize(),
                        filled=0.0,
                        remaining=float(o.qty),
                        avg_fill_price=0.0,
                        protective_ok=False,
                        aborted=False
                    )
            except Exception as exc:
                logger.warning("Error getting order outcome: %s", exc)
            time.sleep(0.5)

        # Timeout handling: treat as working
        try:
            o = self._client.get_order_by_id(order_id)
            raw_status = getattr(o, "status", "new")
            if hasattr(raw_status, "value"):
                raw_status = raw_status.value
            mapped_status = status_map.get(raw_status.lower(), "Submitted")
            filled = float(o.filled_qty or 0)
            qty = float(o.qty or 0)
            return order_exec.OrderResult(
                outcome=order_exec.PARTIALLY_FILLED if filled > 0 else order_exec.SUBMITTED,
                status=mapped_status,
                filled=filled,
                remaining=qty - filled,
                avg_fill_price=float(o.filled_avg_price or 0.0),
                protective_ok=True,
                aborted=False
            )
        except Exception:
            return order_exec.OrderResult(outcome=order_exec.TIMEOUT, status="Timeout", filled=0.0, remaining=0.0)

    def _finalize_open(self, symbol: str, action: str, result: order_exec.OrderResult, intended_qty: int) -> order_exec.OrderResult:
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
            alerts.emit(alerts.EVENT_PARTIAL_FILL, symbol=symbol, action=action,
                        filled=result.filled, remaining=result.remaining,
                        intended_qty=intended_qty)
        else:
            order_audit.log_event(
                order_audit.STAGE_REJECTED if result.outcome == order_exec.REJECTED else order_audit.STAGE_ACK,
                symbol=symbol, action=action, outcome=result.outcome, status=result.status,
                intended_qty=intended_qty,
            )
            if result.outcome == order_exec.REJECTED:
                alerts.emit(alerts.EVENT_ORDER_REJECTED, symbol=symbol, action=action,
                            status=result.status, intended_qty=intended_qty)
            logger.info("Open %s %s x%d -> %s (no fill)", action, symbol, intended_qty, result.outcome)
            return result

        logger.info(
            "Open %s %s -> %s filled=%.0f avg=%.2f protective_ok=%s aborted=%s",
            action, symbol, result.outcome, result.filled, result.avg_fill_price,
            result.protective_ok, result.aborted,
        )
        return result

    def _close_position(self, symbol: str, action: str, qty: int, price: float, note: str) -> order_exec.OrderResult:
        self._require_connection()
        client_order_id = order_exec.deterministic_order_ref(self._today(), symbol, action)
        order_audit.log_event(
            order_audit.STAGE_SUBMIT, kind="close", symbol=symbol, action=action,
            qty=qty, limit_price=price, note=note, order_ref=client_order_id,
        )

        try:
            # Cancel working limit/stop orders first and poll for cancellation confirmation
            if not self._cancel_symbol_working_orders(symbol, timeout_seconds=10.0):
                logger.error("Aborting close_position for %s: working order cancellation unconfirmed/timed out", symbol)
                return order_exec.OrderResult(outcome=order_exec.REJECTED, status="CancellationTimeout", filled=0.0, remaining=float(qty))
            
            # Submit market order to close position
            order = self._client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                client_order_id=client_order_id,
            ))
            
            result = self._await_order_outcome(str(order.id))
            logger.info("Close position %s x%d -> %s filled=%.0f avg=%.2f",
                        symbol, qty, result.outcome, result.filled, result.avg_fill_price)
            return result
        except Exception as exc:
            logger.error("close_position failed for %s: %s", symbol, exc)
            return order_exec.OrderResult(outcome=order_exec.REJECTED, status="Rejected", filled=0.0, remaining=float(qty))

    def _cancel_symbol_working_orders(self, symbol: str, timeout_seconds: float = 10.0) -> bool:
        sym_upper = symbol.upper().strip()
        active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated"}
        
        try:
            al_orders = self._get_active_orders()
            matching_ids = [
                str(o.id) for o in al_orders
                if o.symbol.upper().strip() == sym_upper
                and (o.status.value if hasattr(o.status, "value") else str(o.status)).lower() in active_statuses
            ]
            if not matching_ids:
                return True

            for oid in matching_ids:
                try:
                    self._client.cancel_order_by_id(oid)
                except Exception as c_exc:
                    logger.warning("Cancel request for order %s (%s) error: %s", oid, symbol, c_exc)

            # Poll for cancellation confirmation with 10s timeout
            start_time = time.time()
            while time.time() - start_time < timeout_seconds:
                remaining_orders = [
                    o for o in self._get_active_orders()
                    if o.symbol.upper().strip() == sym_upper
                    and (o.status.value if hasattr(o.status, "value") else str(o.status)).lower() in active_statuses
                ]
                if not remaining_orders:
                    return True
                time.sleep(0.5)

            logger.error("Timed out waiting for working order cancellation for %s after %.1fs", symbol, timeout_seconds)
            alerts.send_alert(f"CRITICAL: Timed out canceling working orders for {symbol} before position close", level="CRITICAL")
            return False
        except Exception as exc:
            logger.error("cancel_symbol_working_orders failed for %s: %s", symbol, exc)
            return False

    def _duplicate_ref_working(self, symbol: str, action: str) -> bool:
        ref = order_exec.deterministic_order_ref(self._today(), symbol, action)
        try:
            al_orders = self._get_active_orders()
            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated"}
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() not in active_statuses:
                    continue
                if getattr(o, "client_order_id", None) == ref:
                    return True
            return False
        except Exception:
            return False

    def _record_if_filled(self, symbol: str, action: str, result: order_exec.OrderResult) -> None:
        pass

    # ── Reconciliation & Flattening ──────────────────────────────
    def get_positions(self) -> list:
        """Return position objects (MockPosition) compatible with IBKR/OrderManager interface."""
        return self.ib.positions()

    def get_open_positions(self) -> list:
        """Alias for get_positions for OrderManager risk check compatibility."""
        return self.get_positions()

    def positions_plain(self) -> list:
        # Returns list of plain dicts for live_invariants and reconciliation
        self._require_connection()
        try:
            al_positions = self._client.get_all_positions()
            return [{"symbol": p.symbol, "qty": float(p.qty)} for p in al_positions]
        except Exception as exc:
            logger.error("positions_plain failed: %s", exc)
            self._conn_health.mark_unhealthy(f"positions_plain error: {exc}")
            raise RuntimeError(f"Alpaca API outage while fetching positions: {exc}")

    def working_orders_plain(self) -> list:
        # Returns list of plain dicts for live_invariants and reconciliation
        try:
            al_orders = self._get_active_orders()
            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated"}
            res = []
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() not in active_statuses:
                    continue
                action = "BUY" if o.side.value.upper() == "BUY" else "SELL"
                status = status_map.get(raw_status.lower(), "Submitted")
                order_type = "LMT" if o.type.value.upper() == "LIMIT" else "MKT"
                res.append({
                    "symbol": o.symbol,
                    "action": action,
                    "order_type": order_type,
                    "order_ref": o.client_order_id,
                    "qty": float(o.qty),
                    "status": status,
                    "tif": "GTC"
                })
            return res
        except Exception as exc:
            logger.error("working_orders_plain failed: %s", exc)
            return []

    def reconcile_startup_state(self) -> dict:
        positions = self.positions_plain()
        working = self.working_orders_plain()
        snapshot = reconciliation.build_snapshot(positions, working)

        order_audit.log_event(
            order_audit.STAGE_RECONCILE, phase="startup",
            source="broker", **reconciliation.audit_fields(snapshot),
        )
        logger.info("Startup reconciliation | %s", reconciliation.summary_line(snapshot))

        if snapshot["unprotected_longs"]:
            alerts.emit(alerts.EVENT_RECONCILE_UNPROTECTED_LONG,
                        message="startup: long(s) with no resting GTC protective stop",
                        symbols=list(snapshot["unprotected_longs"]))

        # Sequential check instead of parallel
        protect = {"checked": 0, "unprotected": [], "repaired": [], "failed": []}
        if snapshot["unprotected_longs"]:
            protect = self.ensure_protective_stops()

        return {
            "snapshot": snapshot,
            "protect": protect,
            "halt_new_entries": bool(self.entry_gate.halted),
            "clean": bool(snapshot["clean"]),
        }

    def ensure_protective_stops(self) -> dict:
        report = {"checked": 0, "unprotected": [], "repaired": [], "failed": []}
        self._require_connection()
        try:
            positions = self._client.get_all_positions()
        except Exception as exc:
            logger.warning("Could not read positions for startup protection scan: %s", exc)
            self.entry_gate.halt()
            report["failed"].append("PORTFOLIO_FETCH_ERROR")
            return report

        working = self.working_orders_plain()
        for p in positions:
            symbol = p.symbol.upper().strip()
            qty = float(p.qty)
            if qty <= 0:
                continue
            report["checked"] += 1
            if order_exec.has_gtc_protective_stop(symbol, qty, working):
                continue
            report["unprotected"].append(symbol)

            # Settle average cost
            basis = float(p.avg_entry_price)
            if basis <= 0:
                self.entry_gate.halt()
                report["failed"].append(symbol)
                logger.error("Startup: unprotected long %s with no valid avgCost -> HALT", symbol)
                continue

            # Submit limit and stop orders manually for protection on Alpaca
            # In Phase 1, we place simple stop order at initial stop price
            stop_price = self._initial_stop_price("SELL", basis)
            try:
                from alpaca.trading.requests import StopOrderRequest
                self._client.submit_order(StopOrderRequest(
                    symbol=symbol, qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=round(stop_price, 2)
                ))
                report["repaired"].append(symbol)
                logger.warning("Startup: repaired GTC protection stop for %s x%d at %.2f", symbol, qty, stop_price)
            except Exception as exc:
                self.entry_gate.halt()
                report["failed"].append(symbol)
                logger.error("Startup: could not protect long %s -> HALT: %s", symbol, exc)
        return report

    def flatten_all(self, confirm: bool = False) -> dict:
        self._require_connection()
        logger.info("Flattening all positions and orders on Alpaca paper account")
        try:
            # Cancel all orders
            try:
                self._client.cancel_orders()
            except Exception as e:
                logger.warning("cancel_orders failed: %s. Canceling one by one.", e)
                active_orders = self._get_active_orders()
                for o in active_orders:
                    try:
                        self._client.cancel_order_by_id(str(o.id))
                    except Exception as inner_e:
                        logger.error("Failed to cancel order %s: %s", o.id, inner_e)
            # Close all positions
            positions = self._client.get_all_positions()
            for p in positions:
                self._client.close_position(p.symbol)
            return {"status": "success", "positions_closed": len(positions)}
        except Exception as exc:
            logger.error("flatten_all failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    def graceful_shutdown(self) -> dict:
        self._require_connection()
        logger.info("Graceful shutdown initiated")
        try:
            self.flatten_all()
            self.disconnect()
            return {"disconnected": True}
        except Exception as exc:
            logger.error("graceful_shutdown failed: %s", exc)
            return {"disconnected": False}

    def place_bracket_order(self, symbol: str, side: str, qty: int, entry_price: float, stop_price: float, target_price: float) -> dict:
        self._require_connection()
        order = self._client.submit_order(LimitOrderRequest(
            symbol=symbol.upper().strip(),
            qty=int(qty),
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=round(entry_price, 2),
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(target_price, 2)),
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
        ))
        parent_id = str(order.id)
        order_ids = [parent_id]
        stop_leg_id = None
        try:
            from alpaca.trading.requests import GetOrderByIdRequest
            nested_order = self._client.get_order_by_id(parent_id, GetOrderByIdRequest(nested=True))
            legs = getattr(nested_order, "legs", []) or []
            for leg in legs:
                leg_id = str(leg.id)
                order_ids.append(leg_id)
                leg_type = str(getattr(leg, "order_type", getattr(leg, "type", ""))).lower()
                leg_stop = getattr(leg, "stop_price", None)
                if "stop" in leg_type or leg_stop is not None:
                    stop_leg_id = leg_id
                    logger.info(f"Extracted stop leg ID {stop_leg_id} for parent bracket order {parent_id}")
        except Exception as leg_exc:
            logger.warning(f"Could not extract nested child leg ID for {parent_id}: {leg_exc}")

        return {
            "order_id": parent_id,
            "stop_order_id": stop_leg_id,
            "order_ids": order_ids,
            "status": "Submitted"
        }

    def place_stop_order(self, symbol: str, side: str, qty: float, stop_price: float) -> dict:
        self._require_connection()
        from alpaca.trading.requests import StopOrderRequest
        order = self._client.submit_order(StopOrderRequest(
            symbol=symbol.upper().strip(),
            qty=int(qty),
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2)
        ))
        return {"order_id": str(order.id)}

    def modify_stop_order(self, order_id: str, new_stop: float) -> Optional[str]:
        self._require_connection()
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            req = ReplaceOrderRequest(stop_price=round(new_stop, 2))
            new_order = self._client.replace_order_by_id(order_id, req)
            new_id = str(getattr(new_order, "id", order_id))
            logger.info(f"Replaced stop order {order_id} -> new order ID {new_id} at stop price ${new_stop:.2f}")
            return new_id
        except Exception as exc:
            logger.error("Failed to modify stop order %s: %s", order_id, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        self._require_connection()
        try:
            self._client.cancel_order_by_id(order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    def sync_today_trades_to_journal(self) -> int:
        """Fetch today's filled orders from Alpaca and sync into trade_journal.jsonl."""
        if not self._connected or not self._client:
            return 0
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            from strategies.session import now_eastern
            from strategies.trade_journal import log_fill, read_journal
            import pytz

            today_date = now_eastern().date()
            existing_records = read_journal()
            existing_order_ids = {str(r.get("execution_id") or r.get("order_id") or "") for r in existing_records}

            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)
            orders = self._client.get_orders(req)
            synced_count = 0

            for o in sorted(orders, key=lambda x: x.filled_at if x.filled_at else _dt.datetime.min.replace(tzinfo=_dt.timezone.utc)):
                if o.filled_at is None or not o.filled_qty or float(o.filled_qty) <= 0:
                    continue
                filled_utc = o.filled_at
                if filled_utc.tzinfo is None:
                    filled_utc = filled_utc.replace(tzinfo=_dt.timezone.utc)
                
                et_tz = pytz.timezone("US/Eastern")
                filled_et = filled_utc.astimezone(et_tz)
                if filled_et.date() != today_date:
                    continue

                order_id_str = str(o.id)
                if order_id_str in existing_order_ids:
                    continue

                client_ref = getattr(o, "client_order_id", "") or ""
                parts = client_ref.split("_")
                strat = parts[1] if len(parts) >= 2 and parts[0] == "dt" else "UNKNOWN"

                side = o.side.value.upper() if hasattr(o.side, "value") else str(o.side).upper()
                fill_price = float(o.filled_avg_price or 0.0)
                qty = int(float(o.filled_qty))

                from strategies.order_registry import get_registered_order
                reg = get_registered_order(order_id_str)
                
                order_type = "ENTRY"
                if reg:
                    expected_price = reg["expected_price"]
                    order_type = reg.get("order_type", "ENTRY")
                else:
                    if getattr(o, "stop_price", None) and float(o.stop_price) > 0:
                        expected_price = float(o.stop_price)
                        order_type = "STOP_LOSS"
                    elif getattr(o, "limit_price", None) and float(o.limit_price) > 0:
                        expected_price = float(o.limit_price)
                        order_type = "TAKE_PROFIT" if side == "SELL" else "ENTRY"
                    else:
                        expected_price = fill_price

                # Signed Slippage: Positive (+) = Adverse, Negative (-) = Favorable
                if expected_price > 0:
                    if side == "BUY":
                        slippage = (fill_price - expected_price) / expected_price
                    else:
                        slippage = (expected_price - fill_price) / expected_price
                else:
                    slippage = 0.0

                # Execution Latency: Only for ENTRY / immediate execution. Mark None (N/A) for stop legs
                submitted_at = getattr(o, "submitted_at", None) or getattr(o, "created_at", None)
                fill_latency_ms = None
                if submitted_at and getattr(o, "filled_at", None):
                    try:
                        duration_sec = (o.filled_at - submitted_at).total_seconds()
                        if order_type == "ENTRY" or duration_sec <= 10.0:
                            fill_latency_ms = max(0.0, duration_sec * 1000.0)
                    except Exception:
                        fill_latency_ms = None

                tier = "5-10" if (5.0 <= fill_price < 10.0) else (">10" if fill_price >= 10.0 else "<5")
                log_fill(
                    symbol=o.symbol,
                    side=side,
                    qty=qty,
                    fill_price=fill_price,
                    expected_price=expected_price,
                    slippage=slippage,
                    fill_latency_ms=fill_latency_ms,
                    order_id=order_id_str,
                    execution_id=order_id_str,
                    timestamp=filled_utc.isoformat(),
                    strategy=strat,
                    price_tier=tier,
                )
                if side == "SELL":
                    try:
                        from strategies.intraday_risk import register_symbol_loss
                        from strategies.trailing_stop import get_trailing_manager
                        from strategies.trade_journal import get_today_closed_trades

                        is_loss = True  # Default fail-safe to True
                        mgr = get_trailing_manager()
                        state = mgr.states.get(o.symbol)

                        if state and state.entry_price > 0:
                            if fill_price >= state.entry_price:
                                is_loss = False
                        else:
                            closed_trades = [t for t in get_today_closed_trades() if t.get("symbol") == o.symbol]
                            if closed_trades and float(closed_trades[-1].get("realized_pnl", 0.0) or 0.0) >= 0:
                                is_loss = False

                        if is_loss:
                            register_symbol_loss(o.symbol, filled_utc)
                    except Exception:
                        try:
                            from strategies.intraday_risk import register_symbol_loss
                            register_symbol_loss(o.symbol, filled_utc)
                        except Exception:
                            pass
                existing_order_ids.add(order_id_str)
                synced_count += 1
            return synced_count
        except Exception as exc:
            logger.warning("sync_today_trades_to_journal failed: %s", exc)
            return 0

    def open_position_count(self) -> int:
        self._require_connection()
        try:
            positions = self._client.get_all_positions()
            return len([p for p in positions if float(p.qty) != 0])
        except Exception as exc:
            logger.error("Failed to get open_position_count: %s", exc)
            return 0

    def has_position(self, symbol: str) -> bool:
        self._require_connection()
        try:
            qty = self.get_position(symbol)
            return abs(qty) > 0
        except Exception:
            return False

    def close_position(self, symbol: str) -> bool:
        self._require_connection()
        try:
            self._client.close_position(symbol.upper().strip())
            return True
        except Exception as exc:
            logger.error("Failed to close position for %s: %s", symbol, exc)
            return False

    def market_price(self, symbol: str) -> Optional[float]:
        self._require_connection()
        try:
            quotes = self._data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol.upper().strip()))
            q = quotes.get(symbol.upper().strip())
            if q is not None:
                bid = float(getattr(q, "bid_price", 0) or 0)
                ask = float(getattr(q, "ask_price", 0) or 0)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2
                return ask or bid or None
            return None
        except Exception as exc:
            logger.error("Failed to get market price for %s: %s", symbol, exc)
            return None

    def execute_signal(self, signal: Signal, coach: bool = False) -> bool:
        symbol = signal.symbol.upper().strip()

        action = signal.action.upper().strip()
        allow_short = bool(self._c("ALLOW_SHORT", False))
        min_cash = float(self._c("MIN_TRADE_CASH", 100.0))

        if action == "HOLD":
            return False
        if action not in {"BUY", "SELL"}:
            logger.warning("Unsupported signal action: %s", signal.action)
            return False

        if self.has_working_order(symbol):
            logger.info("Working order already exists for %s — skipping %s", symbol, action)
            return False

        position = self.get_position(symbol)
        price = self.get_price(symbol, allow_historical=True)
        if price <= 0:
            logger.warning("Could not get valid order price for %s", symbol)
            return False

        if action == "BUY":
            if position > 0:
                logger.info("Already long %s — skipping BUY", symbol)
                return False
            if position < 0:
                qty = int(abs(position))
                return self._close_position(symbol, "BUY", qty, price, "Close short").has_fill

            cash = self.get_cash()
            if cash < min_cash:
                logger.warning("Insufficient cash for BUY %s: $%.2f", symbol, cash)
                return False
            qty = self._calc_quantity(price, cash)
            if qty == 0:
                logger.warning("Qty=0 for BUY %s", symbol)
                return False

            strat_name = getattr(signal, "strategy", getattr(signal, "strategy_name", ""))
            stop_price = getattr(signal, "stop_price", getattr(signal, "stop_loss", None))
            target_price = getattr(signal, "target_price", getattr(signal, "take_profit", None))
            if coach and (stop_price is None or target_price is None or stop_price <= 0 or target_price <= 0):
                logger.warning("Rejecting execute_signal for %s: missing explicit stop_loss/target_price geometry (fail-closed)", symbol)
                return False

            result = self._place_open_bracket(
                symbol, "BUY", qty, price, signal.confidence,
                strategy_name=strat_name,
                custom_stop_price=stop_price,
                custom_target_price=target_price
            )
            paper_ledger.record_entry(
                signal, result,
                order_ref=order_exec.deterministic_order_ref(self._today(), symbol, "BUY"),
            )
            return result.occupies_slot

        # SELL
        if position > 0:
            qty = int(position)
            return self._close_position(symbol, "SELL", qty, price, "Close long").has_fill
        if position < 0:
            logger.info("Already short %s — skipping SELL", symbol)
            return False
        if not allow_short:
            logger.info("No long in %s — skipping SELL (ALLOW_SHORT=False)", symbol)
            return False

        cash = self.get_cash()
        if cash < min_cash:
            logger.warning("Insufficient cash for short SELL %s", symbol)
            return False
        qty = self._calc_quantity(price, cash)
        if qty == 0:
            logger.warning("Qty=0 for short SELL %s", symbol)
            return False

        result = self._place_open_bracket(symbol, "SELL", qty, price, signal.confidence)
        return result.occupies_slot

    def execute_all(self, signals: List[Signal]) -> dict:
        open_symbols = {
            p.symbol.upper()
            for p in self._client.get_all_positions()
            if float(p.qty) != 0.0
        }
        working_symbols = self.working_order_symbols()
        occupied_symbols = open_symbols | working_symbols
        planned_new_symbols: set = set()
        placed, skipped = 0, 0
        allow_short = bool(self._c("ALLOW_SHORT", False))

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
                max_open = int(getattr(config_module, "MAX_OPEN_POSITIONS"))
                planned_total = len(occupied_symbols | planned_new_symbols)
                if planned_total >= max_open:
                    logger.warning("Max positions (%d) reached — skipping %s", max_open, symbol)
                    skipped += 1
                    continue

            if self.execute_signal(signal):
                placed += 1
                if opens_new:
                    planned_new_symbols.add(symbol)
            else:
                skipped += 1

        return {"placed": placed, "skipped": skipped, "total": len(signals)}

    @staticmethod
    def _today() -> str:
        return _dt.date.today().strftime("%Y%m%d")
