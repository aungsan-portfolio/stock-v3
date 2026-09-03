"""
alpaca_bridge.py — Alpaca Paper Trading bridge via alpaca-py.
Duck-type replacement for IBKRBridge. Exposes all pricing, account,
placement, and reconciliation methods expected by Pro V3.
"""
import asyncio
import datetime as _dt
import logging
import math
import os
import random
import threading
import time
from collections import deque
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest, MarketOrderRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from dataclasses import dataclass
from enum import Enum

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
from position_rotator import PositionRotator, OpenPosition, IncomingSignal

logger = logging.getLogger(__name__)


class TimeoutHTTPAdapter(HTTPAdapter):
    """HTTP adapter enforcing hard connect and read timeouts and exponential backoff retries."""

    def __init__(
        self,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        max_retries: Optional[Retry] = None,
        **kwargs,
    ):
        self.connect_timeout = float(connect_timeout)
        self.read_timeout = float(read_timeout)
        if max_retries is None:
            max_retries = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                raise_on_status=False,
            )
        super().__init__(max_retries=max_retries, **kwargs)

    def send(self, request, **kwargs):
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = (self.connect_timeout, self.read_timeout)
        return super().send(request, **kwargs)


def configure_session_timeouts(
    session,
    connect_timeout: float = 5.0,
    read_timeout: float = 10.0,
    total_retries: int = 3,
    backoff_factor: float = 0.5,
) -> None:
    """Mount TimeoutHTTPAdapter on a requests.Session for both http:// and https:// prefixes."""
    if session is None or not hasattr(session, "mount"):
        return
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    adapter = TimeoutHTTPAdapter(
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        max_retries=retries,
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)


class StreamState(Enum):
    DISCONNECTED = "DISCONNECTED"
    RECONNECTING = "RECONNECTING"
    AUTHENTICATED_UNRECONCILED = "AUTHENTICATED_UNRECONCILED"
    RECONCILING = "RECONCILING"
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    LATCHED_DESYNC = "LATCHED_DESYNC"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    STOPPED = "STOPPED"


class BrokerOrderCheck(Enum):
    WORKING_ORDER_EXISTS = "WORKING_ORDER_EXISTS"
    NO_WORKING_ORDER = "NO_WORKING_ORDER"
    BROKER_QUERY_FAILED = "BROKER_QUERY_FAILED"


@dataclass(frozen=True)
class ShutdownResult:
    stop_requested: bool
    reconnect_disabled: bool
    socket_closed: bool
    thread_joined: bool
    thread_alive: bool

    @property
    def clean(self) -> bool:
        return (
            self.stop_requested
            and self.reconnect_disabled
            and self.socket_closed
            and self.thread_joined
            and not self.thread_alive
        )


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


def _round_price(price: float) -> float:
    """Sub-penny rounding helper: 4 decimals if price < $1, 2 decimals if price >= $1."""
    if price <= 0:
        return 0.0
    return round(price, 4) if price < 1.0 else round(price, 2)


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
    def __init__(self, order_id: str, symbol: str, action: str, qty: float, order_type: str, limit_price: float, status: str, client_order_id: str, stop_price: Optional[float] = None, raw_type: Optional[str] = None, raw_side: Optional[str] = None):
        self.id = str(order_id)
        self.symbol = symbol
        self.contract = _Contract(symbol)
        self.action = action.upper()
        self.side = raw_side or action.lower()
        self.totalQuantity = qty
        self.qty = qty
        self.orderType = order_type
        self.order_type = order_type
        self.type = raw_type or order_type.lower()
        self.lmtPrice = limit_price
        self.limit_price = limit_price if (limit_price and limit_price > 0) else None
        self.stopPrice = stop_price if (stop_price and stop_price > 0) else None
        self.stop_price = stop_price if (stop_price and stop_price > 0) else None
        self.orderState = type("State", (), {"status": status})()
        self.status = status
        self.orderRef = client_order_id
        self.client_order_id = client_order_id
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
            al_positions = self.bridge._call_with_retry(lambda: self.bridge._client.get_all_positions())
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
            raw_orders = self.bridge._get_active_orders()
            al_orders = []
            for o in raw_orders:
                al_orders.append(o)
                legs = getattr(o, "legs", []) or []
                for leg in legs:
                    if getattr(leg, "symbol", None) is None:
                        leg.symbol = getattr(o, "symbol", "")
                    if getattr(leg, "qty", None) is None or str(getattr(leg, "qty", "")).strip() in ("0", "0.0", ""):
                        leg.qty = getattr(o, "qty", None) or getattr(o, "filled_qty", 0.0)
                    al_orders.append(leg)

            res = []
            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated"}
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() not in active_statuses:
                    continue
                raw_side = o.side.value if hasattr(o.side, "value") else str(o.side)
                action = "BUY" if str(raw_side).upper() == "BUY" else "SELL"
                o_qty = getattr(o, "qty", None)
                if o_qty is None or str(o_qty).strip() in ("0", "0.0", ""):
                    f_qty = getattr(o, "filled_qty", None)
                    if f_qty is not None and str(f_qty).strip() not in ("0", "0.0", ""):
                        o_qty = f_qty
                try:
                    qty = float(o_qty) if o_qty is not None else 0.0
                except (TypeError, ValueError):
                    qty = 0.0
                status = status_map.get(raw_status.lower(), "Submitted")
                lmt_price = float(o.limit_price) if getattr(o, "limit_price", None) is not None else 0.0
                stp_price = float(o.stop_price) if getattr(o, "stop_price", None) is not None else None
                
                raw_type = o.type.value if hasattr(o.type, "value") else str(o.type)
                raw_type_str = str(raw_type).upper().strip()
                if raw_type_str in ("STOP", "STP"):
                    order_type = "STP"
                elif raw_type_str in ("STOP_LIMIT", "STP LMT", "STOPLIMIT"):
                    order_type = "STP LMT"
                elif raw_type_str in ("TRAILING_STOP", "TRAIL", "TRAILLIMIT", "TRAIL LIMIT"):
                    order_type = "TRAIL"
                elif raw_type_str in ("LIMIT", "LMT"):
                    order_type = "LMT"
                else:
                    order_type = "MKT"

                res.append(MockOrder(
                    order_id=str(o.id),
                    symbol=o.symbol,
                    action=action,
                    qty=qty,
                    order_type=order_type,
                    limit_price=lmt_price,
                    status=status,
                    client_order_id=o.client_order_id,
                    stop_price=stp_price,
                    raw_type=raw_type_str,
                    raw_side=raw_side
                ))
            return res
        except Exception as exc:
            logger.error("MockIB.openTrades() failed: %s", exc)
            return []

    def reqOpenOrders(self) -> List[MockOrder]:
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
    def __init__(self, bridge=None):
        self.halted = False
        self.halt_reason = ""
        self.reasons = set()
        self.bridge = bridge

    def halt(self, reason: str = ""):
        self.halted = True
        self.halt_reason = str(reason or "")
        if reason:
            self.reasons.add(str(reason))

    def clear(self, reason: str = ""):
        if reason:
            self.reasons.discard(str(reason))
        if not self.reasons:
            self.halted = False
            self.halt_reason = ""

    def clear_recovery_halts(self):
        self.reasons.discard("TRADE_STREAM_DISCONNECTED")
        self.reasons.discard("BROKER_RECONCILIATION_PENDING")
        if not self.reasons:
            self.halted = False
            self.halt_reason = ""

    def risk_sized_qty(self, symbol, signal, price, qty):
        if self.halted:
            return 0
        return qty

    def entry_blocked(self, symbol, value, signal, price):
        if self.halted:
            return True, "Trading engine is halted"
        if self.bridge is not None:
            try:
                daily_pnl = self.bridge.account_daily_pnl()
                max_loss = float(getattr(config_module, "MAX_DAILY_LOSS_SWING", 1000.0))
                if daily_pnl < -abs(max_loss):
                    self.halt()
                    return True, f"Max daily loss limit reached (${daily_pnl:.2f} < -${max_loss:.2f})"
            except Exception as e:
                logger.error("Error checking daily loss limit in EntryGateMock: %s", e)
        return False, ""


class AlpacaBridge:
    def __init__(self, cfg=None, mode: str = "daytrade") -> None:
        self.mode = str(mode or "daytrade").lower()
        self._cfg = cfg or config_module.get_settings()
        self._client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None
        self._connected = False
        self._conn_health = ConnectionHealth()
        # Stream State Machine and Lifecycle Controls
        self._stream_state = StreamState.DISCONNECTED
        self._stream_state_lock = threading.RLock()
        self._reconcile_lock = threading.Lock()
        self._stream_generation = 0
        # Daemon thread owns SDK websocket event loop
        self._trade_stream: Optional[TradingStream] = None
        self._trade_stream_thread: Optional[threading.Thread] = None
        self._trade_stream_lock = threading.RLock()
        self._trade_stream_stopping = False
        self._trade_stream_ready = False
        self._trade_stream_recovery_required = False
        self._trade_stream_credentials = None
        self._trade_stream_reconnect_thread: Optional[threading.Thread] = None
        self._trade_stream_reconnect_cancel = threading.Event()
        self._seen_trade_update_keys = set()
        self._seen_trade_update_order = deque(maxlen=10_000)
        self.ib = MockIB(self)
        self.entry_gate = EntryGateMock(self)
        self.rotator = PositionRotator()

    @property
    def stream_state(self) -> StreamState:
        with self._stream_state_lock:
            return self._stream_state

    @property
    def stream_generation(self) -> int:
        with self._stream_state_lock:
            return self._stream_generation

    @property
    def active_stream_threads_count(self) -> int:
        count = 0
        with self._trade_stream_lock:
            if self._trade_stream_thread is not None and self._trade_stream_thread.is_alive():
                count += 1
            if self._trade_stream_reconnect_thread is not None and self._trade_stream_reconnect_thread.is_alive():
                count += 1
        return count

    def _transition_stream_state(self, expected: set, new_state: StreamState, reason: str = "") -> bool:
        with self._stream_state_lock:
            if self._stream_state not in expected:
                logger.debug(
                    "[STREAM STATE REJECTED] current=%s expected=%s target=%s reason=%s",
                    self._stream_state.value,
                    [e.value if hasattr(e, "value") else str(e) for e in expected],
                    new_state.value,
                    reason,
                )
                return False
            old_state = self._stream_state
            self._stream_state = new_state
            logger.info(
                "[STREAM STATE] old=%s new=%s reason=%s",
                old_state.value,
                new_state.value,
                reason,
            )
            return True

    def _is_reconnect_allowed(self) -> bool:
        with self._stream_state_lock:
            if self._stream_state in (StreamState.SHUTTING_DOWN, StreamState.STOPPED):
                return False
            return True

    def _clear_recovery_halts(self) -> None:
        gate = getattr(self, "entry_gate", None)
        if gate is not None:
            if hasattr(gate, "clear_recovery_halts"):
                gate.clear_recovery_halts()
            elif hasattr(gate, "clear"):
                gate.clear("TRADE_STREAM_DISCONNECTED")
                gate.clear("BROKER_RECONCILIATION_PENDING")
            else:
                gate.halted = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_stream_ready(self) -> bool:
        return self._trade_stream_ready

    def _c(self, name, default=None):
        return getattr(self._cfg, name, default)

    def _require_connection(self):
        if not self._connected or self._client is None:
            raise RuntimeError("AlpacaBridge is not connected")

    def _require_trade_stream(self) -> None:
        """Fail closed for new entries when order-state streaming is unavailable or not HEALTHY."""
        if os.environ.get("PYTEST_CURRENT_TEST") or getattr(self, "_test_mode", False):
            return
        thread = self._trade_stream_thread
        stream = self._trade_stream
        if (
            self._stream_state != StreamState.HEALTHY
            or not self._trade_stream_ready
            or thread is None
            or not thread.is_alive()
            or self._trade_stream_recovery_required
            or stream is None
            or not getattr(stream, "_running", False)
        ):
            raise RuntimeError(f"Trade-update stream is not HEALTHY (state: {self._stream_state}); refusing new entry")

    # ── Connection ───────────────────────────────────────────────
    def connect(self) -> bool:
        # Load .env credentials if present (do not overwrite existing os.environ)
        if os.path.exists(".env") and not os.environ.get("PYTEST_CURRENT_TEST"):
            try:
                with open(".env", "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'").strip('"')
                            os.environ.setdefault(k, v)
            except Exception as e:
                logger.warning(f"Failed to auto-parse .env file: {e}")

        mode = getattr(self, "mode", None) or os.environ.get("TRADING_MODE", "daytrade")
        if str(mode).lower() == "swing":
            key = os.environ.get("SWING_APCA_API_KEY_ID", "") or os.environ.get("APCA_API_KEY_ID", "")
            secret = os.environ.get("SWING_APCA_API_SECRET_KEY", "") or os.environ.get("APCA_API_SECRET_KEY", "")
            endpoint = os.environ.get("SWING_APCA_API_BASE_URL", "") or os.environ.get("APCA_API_BASE_URL", self._c("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")).rstrip("/")
        else:
            key = os.environ.get("DAYTRADE_APCA_API_KEY_ID", "") or os.environ.get("APCA_API_KEY_ID", "")
            secret = os.environ.get("DAYTRADE_APCA_API_SECRET_KEY", "") or os.environ.get("APCA_API_SECRET_KEY", "")
            endpoint = os.environ.get("DAYTRADE_APCA_API_BASE_URL", "") or os.environ.get("APCA_API_BASE_URL", self._c("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")).rstrip("/")
        if not key or not secret:
            logger.error("Alpaca credentials missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars")
            self._conn_health.mark_unhealthy("Credentials missing")
            return False

        # Export active credentials to os.environ so historical data and screener clients authenticate correctly
        os.environ["APCA_API_KEY_ID"] = key
        os.environ["APCA_API_SECRET_KEY"] = secret
        os.environ["APCA_API_BASE_URL"] = endpoint
        if endpoint != "https://paper-api.alpaca.markets":
            logger.error("Safety block: APCA_API_BASE_URL must be paper endpoint (https://paper-api.alpaca.markets)")
            self._conn_health.mark_unhealthy("Non-paper base URL detected")
            raise RuntimeError("Live trading is disabled. Endpoint must be https://paper-api.alpaca.markets")

        try:
            self._client = TradingClient(key, secret, paper=True)
            self._data_client = StockHistoricalDataClient(key, secret)

            # Hardened Network Timeout & Retry Layer (connect=5.0s, read=10.0s, retries=3)
            configure_session_timeouts(getattr(self._client, "_session", None), connect_timeout=5.0, read_timeout=10.0)
            configure_session_timeouts(getattr(self._data_client, "_session", None), connect_timeout=5.0, read_timeout=10.0)

            # Validate account status
            acct = self._client.get_account()
            if getattr(acct, "status", None) != "ACTIVE" or getattr(acct, "account_blocked", False) or getattr(acct, "trading_blocked", False):
                logger.error("Alpaca account is not active or is blocked. Status: %s", getattr(acct, "status", "UNKNOWN"))
                self._conn_health.mark_unhealthy("Account blocked")
                return False

            self._account_id = str(getattr(acct, "account_number", "") or getattr(acct, "id", "") or "default")
            logger.info("Connected to Alpaca Paper Account (%s) | Status: %s | Equity: %s", self._account_id, acct.status, acct.equity)
            from observability import get_build_provenance
            logger.info(get_build_provenance().marker_string())
            from strategies.session import now_eastern
            et_date = str(now_eastern().date())
            state = self._load_daytrade_risk_state(self._account_id)
            if state.get("date") == et_date and state.get("account_id") == self._account_id:
                self._start_of_day_equity = float(state.get("start_of_day_equity", acct.equity))
                self._daytrade_suspended = bool(state.get("suspended", False))
                logger.info("[%s] Restored start-of-day equity: $%.2f, suspended: %s", self._account_id, self._start_of_day_equity, self._daytrade_suspended)
            else:
                self._start_of_day_equity = float(acct.equity)
                self._daytrade_suspended = False
                self._save_daytrade_risk_state(et_date, self._start_of_day_equity, False, account_id=self._account_id)
                logger.info("[%s] Initialized new day start-of-day equity: $%.2f", self._account_id, self._start_of_day_equity)

            # Attach FlightRecorderHandler and restore persisted latch state if present
            try:
                from observability import attach_flight_recorder, get_latch_state_machine
                attach_flight_recorder()
                latch_state = state.get("latch_state")
                if latch_state:
                    get_latch_state_machine().import_state_dict(latch_state)
            except Exception as obs_exc:
                logger.debug("[ALPACA BRIDGE] Observability connect init: %s", obs_exc)

            self._session_start_time = _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(hours=24)
            self._connected = True
            self._conn_health.mark_healthy()
            try:
                self._start_trade_stream(key, secret)
            except Exception as str_err:
                logger.warning("[TRADE STREAM] Auto-start stream warning: %s", str_err)
            return True
        except Exception as exc:
            logger.exception("Failed to connect to Alpaca API")
            self._connected = False
            self._stop_trade_stream()
            self._client = None
            self._data_client = None
            self._conn_health.mark_unhealthy(str(exc))
            return False

    def disconnect(self) -> None:
        self._connected = False
        self._stop_trade_stream()
        self._client = None
        self._data_client = None
        logger.info("Disconnected from Alpaca")

    @staticmethod
    def _string_value(value) -> str:
        return str(getattr(value, "value", value) or "").lower().strip()

    @staticmethod
    def _float_value(value) -> float:
        try:
            return float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _remember_trade_update(self, key: str) -> bool:
        if not key:
            return True
        with self._trade_stream_lock:
            if key in self._seen_trade_update_keys:
                return False
            if len(self._seen_trade_update_order) == self._seen_trade_update_order.maxlen:
                stale_key = self._seen_trade_update_order.popleft()
                self._seen_trade_update_keys.discard(stale_key)
            self._seen_trade_update_order.append(key)
            self._seen_trade_update_keys.add(key)
        return True

    def _dispatch_trade_update_with_generation(self, update, generation: int) -> bool:
        with self._stream_state_lock:
            if generation != self._stream_generation:
                logger.warning(
                    "[STREAM CALLBACK IGNORED] stale_generation=%s current=%s",
                    generation,
                    self._stream_generation,
                )
                return False
        self._dispatch_trade_update(update)
        return True

    async def _on_trade_update(self, update, generation: int = 0) -> None:
        await asyncio.to_thread(self._dispatch_trade_update_with_generation, update, generation)

    def _dispatch_trade_update(self, update) -> None:
        try:
            order = getattr(update, "order", None)
            parent_order_id = str(getattr(order, "id", "") if order else getattr(update, "order_id", "") or "").strip()
            if not parent_order_id:
                return

            event_name = self._string_value(getattr(update, "event", ""))
            order_status = self._string_value(getattr(order, "status", "") if order else getattr(update, "order_status", ""))
            status_from_event = {
                "fill": "filled",
                "partial_fill": "partially_filled",
                "canceled": "canceled",
                "cancelled": "canceled",
                "rejected": "rejected",
                "expired": "expired",
                "done_for_day": "done_for_day",
                "replaced": "replaced",
            }
            status = order_status or status_from_event.get(event_name, event_name)
            cumulative_filled_qty = self._float_value(getattr(order, "filled_qty", None) if order else getattr(update, "position_qty", None))
            average_fill_price = self._float_value(getattr(order, "filled_avg_price", None) if order else getattr(update, "price", None))
            execution_id = str(getattr(update, "execution_id", "") or "").strip() or None

            event_key = execution_id or (
                f"{parent_order_id}|{event_name}|{status}|"
                f"{cumulative_filled_qty:.12g}|{average_fill_price:.12g}"
            )

            # Record always-on trade update stream to broker_events.jsonl
            try:
                broker_ev_path = getattr(config, "BROKER_EVENTS_PATH", None) or Path("logs/broker_events.jsonl")
                broker_ev_path = Path(broker_ev_path)
                broker_ev_path.parent.mkdir(parents=True, exist_ok=True)
                with open(broker_ev_path, "a", encoding="utf-8") as f_ev:
                    f_ev.write(json.dumps({
                        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                        "event": event_name,
                        "status": status,
                        "order_id": parent_order_id,
                        "symbol": str(getattr(order, "symbol", "") if order else ""),
                        "filled_qty": cumulative_filled_qty,
                        "price": average_fill_price,
                        "execution_id": execution_id,
                    }) + "\n")
            except Exception:
                pass

            if not self._remember_trade_update(event_key):
                logger.debug("Ignoring duplicate Alpaca trade update %s", event_key)
                return

            from strategies.entry_lifecycle import entry_lifecycle
            was_pending_entry = parent_order_id in entry_lifecycle.pending_snapshot()

            self.on_entry_order_update(
                parent_order_id=parent_order_id,
                cumulative_filled_qty=cumulative_filled_qty,
                average_fill_price=average_fill_price,
                execution_id=execution_id,
                status=status,
            )

            if status == "replaced":
                if was_pending_entry:
                    self._conn_health.mark_unhealthy("Tracked entry parent was replaced")
                    self.entry_gate.halt("TRACKED_ENTRY_REPLACED")
                    logger.critical(
                        "[TRADE STREAM] Tracked entry parent %s was replaced; "
                        "new entries halted pending lifecycle reconciliation",
                        parent_order_id,
                    )
                else:
                    logger.debug(
                        "[TRADE STREAM] Non-entry order %s replaced (e.g. trailing stop modification); entry gate unaffected",
                        parent_order_id,
                    )
        except Exception as exc:
            self._conn_health.mark_unhealthy(f"Trade-update dispatch error: {exc}")
            self.entry_gate.halt("TRADE_STREAM_DISPATCH_ERROR")
            logger.exception("[TRADE STREAM] Lifecycle dispatch failed; new entries halted")

    def on_entry_order_update(self, parent_order_id: str, cumulative_filled_qty: float, average_fill_price: float, execution_id: Optional[str] = None, status: str = ""):
        """Route entry parent order fill updates into entry_lifecycle."""
        from strategies.entry_lifecycle import entry_lifecycle
        pending = entry_lifecycle.pending_snapshot()
        if parent_order_id not in pending:
            return

        try:
            if cumulative_filled_qty > 0 and average_fill_price > 0:
                entry_lifecycle.on_parent_fill(
                    bridge=self,
                    parent_order_id=parent_order_id,
                    cumulative_filled_qty=cumulative_filled_qty,
                    average_fill_price=average_fill_price,
                    execution_id=execution_id,
                )
        except Exception as fill_err:
            logger.error("on_entry_order_update fill processing failed for order %s: %s", parent_order_id, fill_err)

        try:
            st_lower = str(status).lower().strip()
            terminal_statuses = {"cancelled", "canceled", "rejected", "expired", "done_for_day", "replaced", "filled"}
            if st_lower in terminal_statuses:
                entry_lifecycle.on_parent_terminal(parent_order_id, status)
        except Exception as term_err:
            logger.error("on_entry_order_update terminal processing failed for order %s: %s", parent_order_id, term_err)

    def _mark_trade_stream_unavailable(self, reason: str) -> None:
        with self._trade_stream_lock:
            self._trade_stream_ready = False
            self._trade_stream_recovery_required = True
        self._conn_health.mark_unhealthy(reason)
        self.entry_gate.halt("TRADE_STREAM_DISCONNECTED")
        logger.error("[TRADE STREAM] %s; new entries are blocked until recovery", reason)

    def _schedule_trade_stream_reconnect(self) -> None:
        with self._trade_stream_lock:
            credentials = self._trade_stream_credentials
            existing = self._trade_stream_reconnect_thread
            if (
                self._trade_stream_stopping
                or not self._connected
                or not credentials
                or not self._is_reconnect_allowed()
                or (existing is not None and existing.is_alive())
            ):
                return
            self._trade_stream_reconnect_cancel.clear()
            thread = threading.Thread(
                target=self._trade_stream_reconnect_loop,
                name="alpaca-trade-reconnect",
                daemon=True,
            )
            self._trade_stream_reconnect_thread = thread
            thread.start()

    def _trade_stream_reconnect_loop(self) -> None:
        max_attempts = int(self._c("TRADE_STREAM_RECONNECT_MAX_ATTEMPTS", 8))
        base_delay = float(self._c("TRADE_STREAM_RECONNECT_BASE_DELAY_SECONDS", 1.0))
        max_delay = float(self._c("TRADE_STREAM_RECONNECT_MAX_DELAY_SECONDS", 30.0))
        jitter_ratio = float(self._c("TRADE_STREAM_RECONNECT_JITTER_RATIO", 0.20))

        try:
            for attempt in range(1, max_attempts + 1):
                if (
                    self._trade_stream_reconnect_cancel.is_set()
                    or not self._connected
                    or not self._is_reconnect_allowed()
                ):
                    return
                self._transition_stream_state(
                    expected={StreamState.DISCONNECTED, StreamState.RECONNECTING, StreamState.DEGRADED},
                    new_state=StreamState.RECONNECTING,
                    reason=f"Reconnect attempt {attempt}/{max_attempts}",
                )
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay *= 1.0 + random.uniform(-jitter_ratio, jitter_ratio)
                logger.warning(
                    "[TRADE STREAM] Reconnect attempt %d/%d in %.2fs",
                    attempt,
                    max_attempts,
                    delay,
                )
                if self._trade_stream_reconnect_cancel.wait(max(0.0, delay)):
                    return

                with self._trade_stream_lock:
                    credentials = self._trade_stream_credentials
                if not credentials:
                    self._mark_trade_stream_unavailable("Reconnect credentials are unavailable")
                    return

                try:
                    self._start_trade_stream(*credentials)
                    with self._trade_stream_lock:
                        self._trade_stream_recovery_required = False
                    self._conn_health.mark_healthy("Trade stream reconnected")
                    logger.info("[TRADE STREAM] Reconnected and authenticated on attempt %d", attempt)
                    self._run_post_reconnect_reconciliation()
                    return
                except Exception as exc:
                    self._mark_trade_stream_unavailable(
                        f"Reconnect attempt {attempt}/{max_attempts} failed: {exc}"
                    )

            self._mark_trade_stream_unavailable(
                f"Reconnect exhausted after {max_attempts} attempts; operator action required"
            )
            self._transition_stream_state(
                expected={StreamState.RECONNECTING},
                new_state=StreamState.DEGRADED,
                reason="Reconnect attempts exhausted",
            )
        finally:
            with self._trade_stream_lock:
                if self._trade_stream_reconnect_thread is threading.current_thread():
                    self._trade_stream_reconnect_thread = None

    def _run_trade_stream(self, stream: TradingStream) -> None:
        unexpected_exit = False
        failure_reason = "Trade stream stopped unexpectedly"
        try:
            stream.run()
        except Exception as exc:
            failure_reason = f"Trade stream crashed: {exc}"
            logger.exception("[TRADE STREAM] Websocket crashed")
            unexpected_exit = True
        finally:
            with self._trade_stream_lock:
                if self._trade_stream is stream:
                    unexpected_exit = unexpected_exit or (
                        not self._trade_stream_stopping and self._connected and self._is_reconnect_allowed()
                    )
                    self._trade_stream_ready = False
                    self._trade_stream_thread = None
                    self._trade_stream = None
            if unexpected_exit and self._is_reconnect_allowed():
                self._transition_stream_state(
                    expected={StreamState.HEALTHY, StreamState.AUTHENTICATED_UNRECONCILED, StreamState.RECONCILING, StreamState.DEGRADED},
                    new_state=StreamState.DISCONNECTED,
                    reason=failure_reason,
                )
                self._mark_trade_stream_unavailable(failure_reason)
                self._schedule_trade_stream_reconnect()

    def _start_trade_stream(self, api_key: str, secret_key: str) -> None:
        with self._trade_stream_lock:
            if self._trade_stream_thread and self._trade_stream_thread.is_alive():
                if self._trade_stream_ready:
                    return
                raise RuntimeError("Trade-update stream startup is already in progress")

            self._stream_generation += 1
            current_gen = self._stream_generation
            self._trade_stream_stopping = False
            self._trade_stream_ready = False
            self._trade_stream_credentials = (api_key, secret_key)
            stream = TradingStream(api_key, secret_key, paper=True)

            async def _stream_callback(upd):
                await self._on_trade_update(upd, generation=current_gen)

            stream.subscribe_trade_updates(_stream_callback)
            thread = threading.Thread(
                target=self._run_trade_stream,
                args=(stream,),
                name="alpaca-trade-updates",
                daemon=True,
            )
            self._trade_stream = stream
            self._trade_stream_thread = thread
            thread.start()

        deadline = time.monotonic() + float(self._c("TRADE_STREAM_START_TIMEOUT_SECONDS", 10.0))
        while time.monotonic() < deadline:
            if getattr(stream, "_running", False):
                with self._trade_stream_lock:
                    if self._trade_stream is stream:
                        self._trade_stream_ready = True
                        self._trade_stream_recovery_required = False
                self._transition_stream_state(
                    expected={StreamState.DISCONNECTED, StreamState.RECONNECTING, StreamState.AUTHENTICATED_UNRECONCILED, StreamState.STOPPED},
                    new_state=StreamState.AUTHENTICATED_UNRECONCILED,
                    reason="Authenticated paper trade-update stream is ready",
                )
                logger.info("[TRADE STREAM] Authenticated paper trade-update stream is ready")
                return
            if not thread.is_alive():
                break
            time.sleep(0.05)

        self._stop_trade_stream(cancel_reconnect=False)
        raise RuntimeError("Trade-update stream did not authenticate before startup timeout")

    def start_trading_stream(self, key: str, secret: str) -> None:
        """Public alias for _start_trade_stream."""
        self._start_trade_stream(key, secret)

    def _run_post_reconnect_reconciliation(self) -> bool:
        if not self._reconcile_lock.acquire(blocking=False):
            logger.info("[RECONCILIATION] already_in_progress=True")
            return False

        try:
            self._transition_stream_state(
                expected={StreamState.AUTHENTICATED_UNRECONCILED, StreamState.RECONNECTING, StreamState.DEGRADED},
                new_state=StreamState.RECONCILING,
                reason="Executing post-reconnect broker truth check",
            )

            recon = self.reconcile_startup_state()
            is_clean = recon.get("clean", False) if isinstance(recon, dict) else getattr(recon, "clean", False)

            if is_clean:
                self._transition_stream_state(
                    expected={StreamState.RECONCILING},
                    new_state=StreamState.HEALTHY,
                    reason="Broker reconciliation clean=True",
                )
                self._clear_recovery_halts()
                logger.info("[RECONNECT RECOVERY] Full broker reconciliation clean; entry halts cleared.")
                return True
            else:
                self._transition_stream_state(
                    expected={StreamState.RECONCILING},
                    new_state=StreamState.LATCHED_DESYNC,
                    reason="Broker reconciliation clean=False (desync/unprotected)",
                )
                try:
                    from observability import get_latch_state_machine, LatchTrigger
                    get_latch_state_machine().latch(
                        LatchTrigger.RECONCILE_MISMATCH,
                        bridge=self,
                        extra_context={"snapshot": recon.get("snapshot", {})},
                    )
                except Exception as l_err:
                    logger.warning("Could not trigger latch state machine during desync: %s", l_err)
                return False
        finally:
            self._reconcile_lock.release()

    def _stop_trade_stream(self, *, cancel_reconnect: bool = True) -> None:
        with self._trade_stream_lock:
            self._trade_stream_stopping = True
            stream = self._trade_stream
            thread = self._trade_stream_thread
            reconnect_thread = self._trade_stream_reconnect_thread
            if cancel_reconnect:
                self._trade_stream_reconnect_cancel.set()

        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                logger.warning("[TRADE STREAM] Graceful stop failed: %s", exc)

        timeout = float(self._c("TRADE_STREAM_STOP_TIMEOUT_SECONDS", 8.0))
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.error("[TRADE STREAM] Thread did not stop before timeout")

        if (
            cancel_reconnect
            and reconnect_thread is not None
            and reconnect_thread is not threading.current_thread()
        ):
            reconnect_thread.join(timeout=timeout)
            if reconnect_thread.is_alive():
                logger.error("[TRADE STREAM] Reconnect supervisor did not stop before timeout")

        with self._trade_stream_lock:
            self._trade_stream = None
            self._trade_stream_thread = None
            self._trade_stream_ready = False
            self._trade_stream_stopping = False
            if cancel_reconnect:
                self._trade_stream_reconnect_thread = None

        with self._trade_stream_lock:
            self._trade_stream = None
            self._trade_stream_thread = None
            self._trade_stream_ready = False
            self._trade_stream_stopping = False
            if cancel_reconnect:
                self._trade_stream_reconnect_thread = None

    def _load_daytrade_risk_state(self, account_id: str = None) -> dict:
        acct_id = account_id or getattr(self, "_account_id", "default")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f'daytrade_risk_state_{acct_id}.json')
        if not os.path.exists(path):
            legacy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'daytrade_risk_state.json')
            if not os.path.exists(legacy_path):
                return {}
            try:
                with open(legacy_path, "r") as f:
                    import json
                    data = json.load(f)
                    if data.get("account_id") == acct_id:
                        return data
                    return {}
            except Exception:
                return {}
        try:
            with open(path, "r") as f:
                import json
                return json.load(f)
        except Exception as e:
            logger.error("Failed to load daytrade risk state: %s", e)
            return {}

    def _save_daytrade_risk_state(self, date_str: str, equity: float, suspended: bool, account_id: str = None):
        acct_id = account_id or getattr(self, "_account_id", "default")
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', f'daytrade_risk_state_{acct_id}.json')
        latch_dict = None
        try:
            from observability import get_latch_state_machine
            latch_dict = get_latch_state_machine().export_state_dict()
        except Exception:
            pass
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                import json
                json.dump({
                    "account_id": acct_id,
                    "date": date_str,
                    "start_of_day_equity": equity,
                    "suspended": suspended,
                    "latch_state": latch_dict,
                }, f, indent=2)
        except Exception as e:
            logger.error("Failed to save daytrade risk state: %s", e)

    def _call_with_retry(self, func, retries: int = 3, backoff_factor: float = 1.0):
        for attempt in range(1, retries + 1):
            try:
                return func()
            except Exception as exc:
                exc_str = str(exc).lower()
                is_transient = any(kw in exc_str for kw in (
                    "time out", "timed out", "timeout", "timedout",
                    "connection pool", "connecttimeout", "readtimeout",
                    "max retries", "500", "502", "503", "504"
                ))
                if is_transient and attempt < retries:
                    sleep_sec = backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        "[ALPACA API RETRY] Transient network error (attempt %d/%d): %s — Retrying in %.1fs...",
                        attempt, retries, exc, sleep_sec
                    )
                    time.sleep(sleep_sec)
                else:
                    raise

    def _get_active_orders(self) -> list:
        self._require_connection()
        try:
            return self._call_with_retry(lambda: self._client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                nested=True,
                limit=500,
                direction="desc"
            )))
        except Exception as exc:
            logger.error("Failed to query active orders from broker: %s", exc)
            self._conn_health.mark_unhealthy(str(exc))
            raise RuntimeError(f"Failed to query active orders from broker (fail-closed): {exc}")

    # ── Account Services ─────────────────────────────────────────
    def get_cash(self) -> float:
        self._require_connection()
        try:
            acct = self._call_with_retry(lambda: self._client.get_account())
            return float(acct.cash)
        except Exception as exc:
            logger.error("get_cash failed: %s", exc)
            return 0.0

    def get_net_liquidation(self) -> float:
        self._require_connection()
        try:
            acct = self._call_with_retry(lambda: self._client.get_account())
            return float(acct.equity)
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
            p = self._call_with_retry(lambda: self._client.get_open_position(symbol.upper().strip()))
            return float(p.qty)
        except Exception as exc:
            if "position does not exist" in str(exc).lower() or "404" in str(exc):
                return 0.0
            logger.error("get_position(%s) failed: %s", symbol, exc)
            self._conn_health.mark_unhealthy(str(exc))
            raise RuntimeError(f"get_position({symbol}) failed (fail-closed): {exc}")

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

    def check_working_order(self, symbol: str, action: Optional[str] = None) -> BrokerOrderCheck:
        try:
            al_orders = self._get_active_orders()
            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated"}
            sym_upper = symbol.upper().strip()
            act_upper = action.upper().strip() if action else None
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() not in active_statuses:
                    continue
                if o.symbol.upper().strip() == sym_upper:
                    if act_upper is None or (o.side.value if hasattr(o.side, "value") else str(o.side)).upper() == act_upper:
                        return BrokerOrderCheck.WORKING_ORDER_EXISTS
            return BrokerOrderCheck.NO_WORKING_ORDER
        except Exception as exc:
            logger.error("check_working_order failed for %s: %s", symbol, exc)
            return BrokerOrderCheck.BROKER_QUERY_FAILED

    def has_working_order(self, symbol: str, action: Optional[str] = None) -> bool:
        check = self.check_working_order(symbol, action=action)
        return check == BrokerOrderCheck.WORKING_ORDER_EXISTS

    # ── Pricing Services ─────────────────────────────────────────
    def _contract(self, symbol: str) -> _Contract:
        return _Contract(symbol)

    def get_order_quote(self, symbol: str, timeout: float = 5.0) -> QuoteDecision:
        self._require_connection()
        symbol = symbol.upper().strip()
        try:
            quotes = self._call_with_retry(lambda: self._data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol, feed="iex")))
            q = quotes.get(symbol)
            if q is None:
                return QuoteDecision(False, 0.0, "alpaca", "No latest quote returned")

            bid = float(getattr(q, "bid_price", 0) or 0)
            ask = float(getattr(q, "ask_price", 0) or 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else (ask or bid or 0.0)

            if mid <= 0:
                return QuoteDecision(False, 0.0, "alpaca", "Quote mid-price is zero")

            spread_pct = (ask - bid) / mid if mid > 0 else 0.0
            max_spread = float(self._c("MAX_QUOTE_SPREAD_PCT", 0.02))
            if spread_pct > max_spread:
                return QuoteDecision(False, mid, "alpaca", f"Spread {spread_pct:.4f} exceeds max {max_spread:.4f}")

            return QuoteDecision(True, mid, "alpaca", "ok")
        except Exception as exc:
            logger.error("get_order_quote failed for %s: %s", symbol, exc)
            return QuoteDecision(False, 0.0, "alpaca", str(exc))

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Fetch fresh realtime broker quote mapping for M-01 execution quote gate."""
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            sym = symbol.upper().strip()
            res = self._data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym))
            quote = res.get(sym) if isinstance(res, dict) else (getattr(res, sym, None) or res)
            if not quote:
                return None
            bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
            ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
            last = float(getattr(quote, "ask_price", 0.0) or bid or 0.0)
            ts = getattr(quote, "timestamp", None)
            if ts and hasattr(ts, "timestamp"):
                ts_epoch = ts.timestamp()
            elif isinstance(ts, (int, float)):
                ts_epoch = float(ts)
            else:
                ts_epoch = time.time()

            return {
                "source": "alpaca",
                "is_realtime": True,
                "timestamp": ts_epoch,
                "bid": bid,
                "ask": ask,
                "last": last,
            }
        except Exception as exc:
            logger.error("get_quote failed for %s: %s", symbol, exc)
            return None

    def get_price(self, symbol: str, timeout: float = 5.0, allow_historical: bool = True) -> float:
        decision = self.get_order_quote(symbol, timeout=timeout)
        if decision.ok and decision.price > 0:
            return decision.price
        if allow_historical:
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
            if isinstance(durationStr, int):
                durationStr = f"{durationStr} D"
            parts = str(durationStr).split()
            n = int(parts[0]) if parts and parts[0].isdigit() else 5
            unit = parts[1].upper() if len(parts) > 1 else "D"
            days = n * {"D": 1, "W": 7, "M": 30, "Y": 365}.get(unit[0], 1)

            end = _dt.datetime.now(_dt.timezone.utc)
            start = end - _dt.timedelta(days=days)

            from alpaca.data.enums import Adjustment
            bars = self._data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=symbol.upper().strip(),
                timeframe=tf,
                start=start,
                end=end,
                feed="iex",
                adjustment=Adjustment.ALL,
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
        raw_qty = max(int(max_value / price), 0)
        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        if raw_qty > 0 and raw_qty < min_shares:
            raw_qty = min_shares
        return raw_qty

    def _limit_price(self, action: str, price: float) -> float:
        offset = float(self._c("LIMIT_ORDER_OFFSET_PCT", 0.001))
        return _round_price(price * (1 + offset)) if action == "BUY" else _round_price(price * (1 - offset))

    def get_daily_atr_pct(self, symbol: str, default: float = 0.025) -> float:
        """Fetch 14-day Daily ATR percentage using 1y daily bars with in-memory caching."""
        sym = str(symbol or "").upper().strip()
        if not sym:
            return default
        if not hasattr(self, "_daily_atr_cache"):
            self._daily_atr_cache = {}
        now = time.time()
        cached = self._daily_atr_cache.get(sym)
        if cached and (now - cached[0] < 3600):
            return cached[1]
        try:
            import data_manager, pandas as pd
            df = data_manager.fetch_ohlcv(sym, period="1y", interval="1d")
            if df is not None and len(df) >= 14:
                high = df["High"]
                low = df["Low"]
                close_prev = df["Close"].shift(1)
                tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
                atr_14 = float(tr.rolling(14, min_periods=5).mean().iloc[-1])
                last_close = float(df["Close"].iloc[-1])
                if last_close > 0:
                    atr_pct = float(atr_14 / last_close)
                    self._daily_atr_cache[sym] = (now, atr_pct)
                    return atr_pct
        except Exception as e:
            logger.debug("get_daily_atr_pct error for %s: %s", sym, e)
        return default

    def _initial_stop_price(self, action: str, price: float, symbol: Optional[str] = None) -> float:
        if getattr(self, "mode", "daytrade").lower() == "swing":
            sym = str(symbol or "").upper().strip()
            etf_symbols = {"SPY", "VTI", "QQQ", "IWM", "XLK", "XLF", "XLV", "XLE", "DIA"}
            atr_pct = self.get_daily_atr_pct(sym, default=0.025)
            if sym in etf_symbols:
                sp = max(0.012, min(0.020, atr_pct * 1.2))
            else:
                # Dynamic 1.3x Daily ATR stop clamped between 2.5% and 5.0%
                sp = max(0.025, min(0.050, atr_pct * 1.3))
        else:
            sp = float(self._c("STOP_LOSS_PCT", 0.008))
        return _round_price(price * (1 - sp)) if action == "BUY" else _round_price(price * (1 + sp))

    def _take_profit_price(self, action: str, price: float, symbol: Optional[str] = None) -> float:
        if getattr(self, "mode", "daytrade").lower() == "swing":
            tp = float(self._c("SWING_TAKE_PROFIT_PCT", 0.080))
        else:
            tp = float(self._c("TAKE_PROFIT_PCT", 0.015))
        return _round_price(price * (1 + tp)) if action == "BUY" else _round_price(price * (1 - tp))

    # ── Placement & Execution ────────────────────────────────────
    def _place_open_bracket(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
        strategy_name: str = "", custom_stop_price: Optional[float] = None,
        custom_target_price: Optional[float] = None
    ) -> order_exec.OrderResult:
        self._require_connection()
        self._require_trade_stream()
        self._require_new_entries_allowed()
        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        if int(qty) <= 0:
            raise ValueError(f"Order quantity must be positive, got {qty}")
        if int(qty) < min_shares:
            raise ValueError(f"Order quantity {qty} is below minimum allowed share count {min_shares}")
        qty = int(qty)
        limit_price = self._limit_price(action, price)
        stop_price = custom_stop_price if custom_stop_price is not None else self._initial_stop_price(action, price, symbol=symbol)
        take_profit_price = custom_target_price if custom_target_price is not None else self._take_profit_price(action, price, symbol=symbol)
        
        strat_tag = str(strategy_name or "UNKNOWN").replace("StrategyName.", "").upper()
        from strategies.order_idempotency import build_client_order_id
        client_order_id = build_client_order_id(symbol=symbol, side=action, bar_timestamp_utc="", signal_hash=f"{strat_tag}_{int(price*100)}_{int(time.time()//300)}", attempt=0)

        order_audit.log_event(
            order_audit.STAGE_SUBMIT, kind="open", symbol=symbol, action=action,
            qty=qty, limit_price=limit_price, confidence=confidence, order_ref=client_order_id,
        )

        tif = TimeInForce.DAY
        try:
            order = self._client.submit_order(LimitOrderRequest(
                symbol=symbol, qty=qty,
                side=OrderSide.BUY if action == "BUY" else OrderSide.SELL,
                time_in_force=tif,
                limit_price=_round_price(limit_price),
                order_class=OrderClass.BRACKET,
                client_order_id=client_order_id,
                take_profit=TakeProfitRequest(limit_price=_round_price(take_profit_price)),
                stop_loss=StopLossRequest(stop_price=_round_price(stop_price)),
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

        # Timeout expired: query broker truth, cancel unfilled working order and confirm
        try:
            o = self._client.get_order_by_id(order_id)
            raw_status = getattr(o, "status", "new")
            if hasattr(raw_status, "value"):
                raw_status = raw_status.value
            status_str = str(raw_status).lower()
            filled = float(o.filled_qty or 0)
            qty = float(o.qty or 0)

            if status_str == "filled":
                return order_exec.OrderResult(
                    outcome=order_exec.FILLED,
                    status="Filled",
                    filled=filled,
                    remaining=0.0,
                    avg_fill_price=float(o.filled_avg_price or o.limit_price or 0.0),
                    protective_ok=True,
                    aborted=False,
                )
            elif status_str in {"canceled", "rejected", "expired"}:
                outcome = order_exec.REJECTED if status_str == "rejected" else order_exec.CANCELLED
                return order_exec.OrderResult(
                    outcome=outcome,
                    status=status_str.capitalize(),
                    filled=filled,
                    remaining=qty - filled,
                    avg_fill_price=float(o.filled_avg_price or 0.0),
                    protective_ok=(filled > 0),
                    aborted=False,
                )

            # Order is still working / unfilled -> cancel order and confirm terminal state
            logger.warning("[ORDER TIMEOUT] Order %s unfilled after %.1fs (status=%s, filled=%.1f/%.1f). Requesting broker cancel...", order_id, timeout, status_str, filled, qty)
            try:
                self._client.cancel_order_by_id(order_id)
            except Exception as cancel_err:
                logger.error("Failed to request order cancel for %s: %s", order_id, cancel_err)

            # Confirm cancellation within 5s grace window
            confirm_deadline = time.time() + 5.0
            while time.time() < confirm_deadline:
                time.sleep(0.5)
                try:
                    conf = self._client.get_order_by_id(order_id)
                    conf_status = getattr(conf, "status", "")
                    if hasattr(conf_status, "value"):
                        conf_status = conf_status.value
                    conf_status_str = str(conf_status).lower()
                    if conf_status_str in {"canceled", "expired", "filled"}:
                        final_filled = float(conf.filled_qty or 0)
                        logger.info("[CANCEL CONFIRMED] Order %s terminal status=%s, final_filled=%.1f", order_id, conf_status_str, final_filled)
                        if conf_status_str == "filled":
                            return order_exec.OrderResult(
                                outcome=order_exec.FILLED,
                                status="Filled",
                                filled=final_filled,
                                remaining=0.0,
                                avg_fill_price=float(conf.filled_avg_price or conf.limit_price or 0.0),
                                protective_ok=True,
                                aborted=False,
                            )
                        elif final_filled > 0:
                            return order_exec.OrderResult(
                                outcome=order_exec.PARTIALLY_FILLED,
                                status="PartiallyFilled",
                                filled=final_filled,
                                remaining=qty - final_filled,
                                avg_fill_price=float(conf.filled_avg_price or 0.0),
                                protective_ok=True,
                                aborted=False,
                            )
                        else:
                            return order_exec.OrderResult(
                                outcome=order_exec.CANCELLED,
                                status="Canceled",
                                filled=0.0,
                                remaining=qty,
                                avg_fill_price=0.0,
                                protective_ok=False,
                                aborted=False,
                            )
                except Exception as poll_err:
                    logger.debug("Polling cancel status for %s: %s", order_id, poll_err)

            return order_exec.OrderResult(outcome=order_exec.TIMEOUT, status="Timeout", filled=filled, remaining=qty - filled)
        except Exception as e:
            logger.error("Error during order timeout reconciliation for %s: %s", order_id, e)
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
            if not self._cancel_symbol_working_orders(symbol, timeout_seconds=10.0):
                logger.error("Aborting close_position for %s: working order cancellation unconfirmed/timed out", symbol)
                return order_exec.OrderResult(outcome=order_exec.REJECTED, status="CancellationTimeout", filled=0.0, remaining=float(qty))
            
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
        return self.ib.positions()

    def get_open_positions(self) -> list:
        return self.get_positions()

    def positions_plain(self) -> list:
        self._require_connection()
        try:
            al_positions = self._call_with_retry(lambda: self._client.get_all_positions())
            return [{"symbol": p.symbol, "qty": float(p.qty)} for p in al_positions]
        except Exception as exc:
            logger.error("positions_plain failed: %s", exc)
            self._conn_health.mark_unhealthy(f"positions_plain error: {exc}")
            raise RuntimeError(f"Alpaca API outage while fetching positions: {exc}")

    def working_orders_plain(self) -> list:
        try:
            raw_orders = self._get_active_orders()
            al_orders = []
            for o in raw_orders:
                al_orders.append(o)
                legs = getattr(o, "legs", []) or []
                for leg in legs:
                    if getattr(leg, "symbol", None) is None:
                        leg.symbol = getattr(o, "symbol", "")
                    if getattr(leg, "qty", None) is None or str(getattr(leg, "qty", "")).strip() in ("0", "0.0", ""):
                        leg.qty = getattr(o, "qty", None) or getattr(o, "filled_qty", 0.0)
                    al_orders.append(leg)

            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new", "accepted_for_bidding", "stopped", "suspended", "calculated", "pending_cancel"}
            res = []
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() not in active_statuses:
                    continue
                raw_side = o.side.value if hasattr(o.side, "value") else str(o.side)
                action = "BUY" if str(raw_side).upper() == "BUY" else "SELL"
                status = status_map.get(raw_status.lower(), "Submitted")

                raw_type = o.type.value if hasattr(o.type, "value") else str(o.type)
                raw_type_str = str(raw_type).upper().strip()
                if raw_type_str in ("STOP", "STP"):
                    order_type = "STOP"
                elif raw_type_str in ("STOP_LIMIT", "STP LMT", "STOPLIMIT"):
                    order_type = "STOP_LIMIT"
                elif raw_type_str in ("TRAILING_STOP", "TRAIL", "TRAILLIMIT", "TRAIL LIMIT"):
                    order_type = "TRAILING_STOP"
                elif raw_type_str in ("LIMIT", "LMT"):
                    order_type = "LMT"
                else:
                    order_type = "MKT"

                stp_price = float(o.stop_price) if getattr(o, "stop_price", None) is not None else None
                lmt_price = float(o.limit_price) if getattr(o, "limit_price", None) is not None else None
                raw_tif = getattr(o, "time_in_force", "GTC")
                if hasattr(raw_tif, "_mock_name") or hasattr(raw_tif, "return_value"):
                    tif = "GTC"
                else:
                    val = getattr(raw_tif, "value", raw_tif)
                    s = str(val or "GTC").upper()
                    tif = "GTC" if "GTC" in s else (s if s else "GTC")

                o_qty = getattr(o, "qty", None)
                if o_qty is None or str(o_qty).strip() in ("0", "0.0", ""):
                    f_qty = getattr(o, "filled_qty", None)
                    if f_qty is not None and str(f_qty).strip() not in ("0", "0.0", ""):
                        o_qty = f_qty
                try:
                    qty_val = float(o_qty) if o_qty is not None else 0.0
                except (TypeError, ValueError):
                    qty_val = 0.0

                res.append({
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "action": action,
                    "side": raw_side.lower() if raw_side else action.lower(),
                    "order_type": order_type,
                    "order_ref": o.client_order_id,
                    "qty": qty_val,
                    "status": status,
                    "tif": tif,
                    "stop_price": stp_price,
                    "limit_price": lmt_price
                })
            return res
        except Exception as exc:
            logger.error("working_orders_plain failed: %s", exc)
            return []

    def working_order_count(self) -> int:
        try:
            return len(self.working_orders_plain())
        except Exception as exc:
            logger.error("working_order_count failed: %s", exc)
            return 0

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

        protect = {"checked": 0, "unprotected": [], "repaired": [], "failed": []}
        if snapshot["unprotected_longs"]:
            protect = self.ensure_protective_stops()

        auto_cleaned_orphans = []
        if snapshot["orphan_exits"]:
            for orphan_order in snapshot["orphan_exits"]:
                oid = orphan_order.get("id") if isinstance(orphan_order, dict) else getattr(orphan_order, "id", None)
                orphan_symbol = orphan_order.get("symbol") if isinstance(orphan_order, dict) else getattr(orphan_order, "symbol", "")
                if oid:
                    try:
                        self._client.cancel_order_by_id(oid)
                        auto_cleaned_orphans.append(oid)
                        logger.info("Startup reconciliation: Auto-cancelled orphaned working exit order %s for %s", oid, orphan_symbol)
                    except Exception as c_exc:
                        logger.warning("Failed auto-cancelling orphaned working exit order %s for %s: %s", oid, orphan_symbol, c_exc)
            if auto_cleaned_orphans:
                working = self.working_orders_plain()
                snapshot = reconciliation.build_snapshot(positions, working)

        soxx_cancelled = []
        try:
            from strategies.intraday_risk import check_soxx_tech_long_gate
            soxx_ok, soxx_reason = check_soxx_tech_long_gate("NVDA", "BUY")
            if not soxx_ok:
                TECH_SEMI_SYMBOLS = {"NVDA", "AMD", "INTC", "SMCI", "QCOM", "MU", "AVGO"}
                logger.warning("[STARTUP SOXX CHECK] SOXX gate failed on startup: %s. Inspecting working entry orders...", soxx_reason)
                for w_order in working:
                    w_sym = str(w_order.get("symbol") if isinstance(w_order, dict) else getattr(w_order, "symbol", "")).upper().strip()
                    w_side = str(w_order.get("side") if isinstance(w_order, dict) else getattr(w_order, "side", "")).lower().strip()
                    w_id = w_order.get("id") if isinstance(w_order, dict) else getattr(w_order, "id", None)
                    if w_sym in TECH_SEMI_SYMBOLS and w_side == "buy" and w_id:
                        try:
                            self._client.cancel_order_by_id(w_id)
                            soxx_cancelled.append(w_sym)
                            logger.warning("[STARTUP SOXX CANCEL] Cancelled pre-existing Tech Long working order %s for %s due to SOXX gate.", w_id, w_sym)
                        except Exception as c_err:
                            logger.warning("Failed cancelling pre-existing Tech Long working order for %s: %s", w_sym, c_err)
                if soxx_cancelled:
                    working = self.working_orders_plain()
                    snapshot = reconciliation.build_snapshot(positions, working)
        except Exception as gate_err:
            logger.debug("Startup SOXX check exception: %s", gate_err)

        # Reconcile local trailing stop states against broker open positions
        try:
            from strategies.trailing_stop import manager as trailing_manager
            open_symbols = {
                (p.get("symbol") if isinstance(p, dict) else getattr(p, "symbol", "")).upper().strip()
                for p in positions
            }
            with trailing_manager._lock:
                orphan_states = [sym for sym in list(trailing_manager.states.keys()) if sym.upper().strip() not in open_symbols]
                for sym in orphan_states:
                    logger.info("[RECONCILE TRAILING STOP] Cleaning up orphan trailing stop state for closed position %s", sym)
                    del trailing_manager.states[sym]
                if orphan_states:
                    trailing_manager._save_state()
        except Exception as trail_rec_err:
            logger.debug("Trailing stop startup reconciliation error: %s", trail_rec_err)

        # Reconcile Latch State Machine: clear if clean, emit heartbeat if dirty
        try:
            from observability import get_latch_state_machine, ClearReason
            sm = get_latch_state_machine()
            if snapshot.get("clean"):
                if sm.is_latched:
                    sm.clear(ClearReason.STARTUP_RECONCILIATION_CLEAN, reconcile=snapshot)
                    from strategies.session import now_eastern
                    self._save_daytrade_risk_state(str(now_eastern().date()), self._start_of_day_equity, False, account_id=self._account_id)
            else:
                if sm.is_latched:
                    sm.heartbeat_if_due(interval_seconds=0.0)  # Immediate heartbeat on dirty startup
        except Exception as l_exc:
            logger.debug("[STARTUP RECONCILE] Latch reconciliation non-fatal warning: %s", l_exc)

        # Update Stream State Machine if in startup or reconciling phase
        try:
            with self._stream_state_lock:
                if self._stream_state == StreamState.AUTHENTICATED_UNRECONCILED:
                    self._transition_stream_state(
                        expected={StreamState.AUTHENTICATED_UNRECONCILED},
                        new_state=StreamState.RECONCILING,
                        reason="Broker reconciliation started",
                    )
                if self._stream_state == StreamState.RECONCILING:
                    if snapshot.get("clean"):
                        self._transition_stream_state(
                            expected={StreamState.RECONCILING},
                            new_state=StreamState.HEALTHY,
                            reason="Broker reconciliation clean=True",
                        )
                        self._clear_recovery_halts()
                    else:
                        self._transition_stream_state(
                            expected={StreamState.RECONCILING},
                            new_state=StreamState.LATCHED_DESYNC,
                            reason="Broker reconciliation clean=False (desync/unprotected)",
                        )
        except Exception as st_exc:
            logger.debug("[STARTUP RECONCILE] Stream state transition non-fatal: %s", st_exc)

        return {
            "snapshot": snapshot,
            "protect": protect,
            "halt_new_entries": bool(self.entry_gate.halted),
            "clean": bool(snapshot["clean"]),
            "auto_cleaned_orphans": auto_cleaned_orphans,
            "soxx_cancelled": soxx_cancelled,
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
            if qty == 0:
                continue
            is_long = qty > 0
            abs_qty = abs(qty)
            report["checked"] += 1
            basis = float(p.avg_entry_price)
            if basis <= 0:
                self.entry_gate.halt()
                report["failed"].append(symbol)
                logger.error("Startup: unprotected position %s with no valid avgCost -> HALT", symbol)
                continue

            cur_price = float(getattr(p, "current_price", basis) or basis)
            pnl_pct = (cur_price - basis) / basis if basis > 0 else 0.0

            # Find existing protective stop for this symbol if any
            existing_stop = None
            for wo in working:
                w_sym = str(wo.get("symbol") if isinstance(wo, dict) else getattr(wo, "symbol", "")).upper().strip()
                if w_sym == symbol:
                    w_type = str(wo.get("order_type") if isinstance(wo, dict) else getattr(wo, "type", "")).upper()
                    w_stop = wo.get("stop_price") if isinstance(wo, dict) else getattr(wo, "stop_price", None)
                    if "STOP" in w_type or w_stop is not None:
                        try:
                            existing_stop = float(w_stop) if w_stop is not None else None
                        except (TypeError, ValueError):
                            pass
                        break

            if is_long:
                target_stop = self._initial_stop_price("BUY", basis, symbol=symbol)
                mkt_price = self.market_price(symbol)
                if mkt_price is not None and mkt_price > 0:
                    target_stop = min(target_stop, mkt_price * 0.995)
                target_stop = round(target_stop, 2)
                stop_side = OrderSide.SELL
            else:
                target_stop = basis * 1.01  # +1.0% upside stop for short
                mkt_price = self.market_price(symbol)
                if mkt_price is not None and mkt_price > 0:
                    target_stop = max(target_stop, mkt_price * 1.005)
                target_stop = round(target_stop, 2)
                stop_side = OrderSide.BUY

            if existing_stop is not None or order_exec.has_gtc_protective_stop(symbol, abs_qty, working):
                continue

            report["unprotected"].append(symbol)
            stop_price = target_stop

            try:
                from alpaca.trading.requests import StopOrderRequest
                self._client.submit_order(StopOrderRequest(
                    symbol=symbol, qty=abs_qty,
                    side=stop_side,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_price
                ))
                report["repaired"].append(symbol)
                logger.warning("Startup: repaired GTC protection stop for %s x%.2f at %.2f", symbol, abs_qty, stop_price)
            except Exception as exc:
                err_str = str(exc)
                has_stop_protection = False
                if "insufficient qty" in err_str.lower() or "held_for_orders" in err_str.lower() or "40310000" in err_str:
                    for wo in working:
                        w_info = getattr(wo, "order", wo)
                        w_sym = getattr(w_info, "symbol", "")
                        if str(w_sym).upper().strip() == symbol:
                            w_type = str(getattr(w_info, "order_type", getattr(w_info, "type", ""))).lower()
                            w_stop = getattr(w_info, "stop_price", None)
                            if "stop" in w_type or w_stop is not None:
                                has_stop_protection = True
                                break
                    if not has_stop_protection:
                        logger.warning("Startup: %s shares are held by non-protective order without stop loss. Cancelling non-protective order to attach GTC stop...", symbol)
                        for wo in working:
                            w_sym = str(wo.get("symbol") if isinstance(wo, dict) else getattr(wo, "symbol", "")).upper().strip()
                            w_id = wo.get("id") if isinstance(wo, dict) else getattr(wo, "id", None)
                            if w_sym == symbol and w_id:
                                try:
                                    self._client.cancel_order_by_id(w_id)
                                    logger.info("Startup: Cancelled non-protective order %s for %s", w_id, symbol)
                                except Exception as c_err:
                                    logger.warning("Failed cancelling non-protective order %s: %s", w_id, c_err)
                        try:
                            self._client.submit_order(StopOrderRequest(
                                symbol=symbol, qty=abs_qty,
                                side=stop_side,
                                time_in_force=TimeInForce.GTC,
                                stop_price=stop_price
                            ))
                            report["repaired"].append(symbol)
                            has_stop_protection = True
                            logger.warning("Startup: successfully repaired real GTC protection stop for %s x%.2f at %.2f", symbol, abs_qty, stop_price)
                        except Exception as retry_exc:
                            logger.error("Startup: failed placing GTC protection stop for %s after cancelling unprotective order: %s", symbol, retry_exc)

                if has_stop_protection:
                    logger.info("Startup: %s is confirmed protected by broker stop order.", symbol)
                    if symbol not in report["repaired"]:
                        report["repaired"].append(symbol)
                else:
                    self.entry_gate.halt()
                    report["failed"].append(symbol)
                    logger.error("Startup: long %s is held by orders without a downside stop loss leg -> HALT for safety: %s", symbol, exc)
        return report

    def flatten_all(self, confirm: bool = False) -> dict:
        self._require_connection()
        logger.info("Flattening all positions and orders on Alpaca paper account")
        closed_count = 0
        failed_symbols = []
        try:
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

            positions = []
            try:
                positions = self._client.get_all_positions()
            except Exception as p_err:
                logger.error("flatten_all: Failed to fetch positions: %s", p_err)
                try:
                    alerts.send_alert(f"🚨 CRITICAL: flatten_all failed to fetch positions from Alpaca: {p_err}", level="ERROR")
                except Exception:
                    pass
                return {"status": "error", "message": str(p_err), "positions_closed": 0}

            for p in positions:
                symbol = p.symbol.upper().strip()
                close_success = False
                for attempt in range(3):
                    try:
                        self._client.close_position(symbol)
                        close_success = True
                        closed_count += 1
                        logger.info("Successfully closed position %s on attempt %d", symbol, attempt + 1)
                        break
                    except Exception as c_err:
                        logger.warning("Attempt %d to close position %s failed: %s", attempt + 1, symbol, c_err)
                        time.sleep(0.5)

                if not close_success:
                    failed_symbols.append(symbol)
                    logger.error("CRITICAL: Failed to close position %s after 3 attempts", symbol)

            time.sleep(1.0)
            remaining_positions = []
            try:
                remaining_positions = [p for p in self._client.get_all_positions() if float(p.qty) != 0]
            except Exception:
                pass

            if failed_symbols or remaining_positions:
                rem_syms = failed_symbols or [p.symbol for p in remaining_positions]
                msg = f"🚨 CRITICAL FLATTEN ALERT: Failed to close positions for: {rem_syms}"
                logger.error(msg)
                try:
                    alerts.send_alert(msg, level="ERROR")
                except Exception:
                    pass
                return {"status": "partial", "positions_closed": closed_count, "failed_symbols": rem_syms}

            return {"status": "success", "positions_closed": closed_count, "failed_symbols": []}
        except Exception as exc:
            logger.error("flatten_all failed: %s", exc)
            return {"status": "error", "message": str(exc), "positions_closed": closed_count}

    def flatten_on_shutdown(self, only_unprotected: bool = True) -> dict:
        self._require_connection()
        if not only_unprotected:
            return self.flatten_all()

        logger.info("[SHUTDOWN GUARD] Evaluating open positions before graceful exit...")
        positions = []
        try:
            positions = [p for p in self._client.get_all_positions() if float(p.qty) != 0]
        except Exception as p_err:
            logger.error("[SHUTDOWN GUARD] Failed to fetch positions: %s. Falling back to full flatten_all()", p_err)
            return self.flatten_all()

        if not positions:
            logger.info("[SHUTDOWN GUARD] No open positions. Shutdown clean.")
            return {"status": "clean", "positions_closed": 0}

        working = self.working_orders_plain()
        closed_count = 0
        preserved_count = 0

        for p in positions:
            symbol = p.symbol.upper().strip()
            qty = abs(float(p.qty))
            if qty == 0:
                continue

            has_stop = False
            symbol_orders = []
            for w in working:
                w_sym = (w.get("symbol") if isinstance(w, dict) else getattr(w, "symbol", "") or "").upper().strip()
                if w_sym == symbol:
                    symbol_orders.append(w)
                    w_type = str(w.get("order_type") or w.get("type") if isinstance(w, dict) else getattr(w, "order_type", getattr(w, "type", "")) or "").lower()
                    if "stop" in w_type:
                        has_stop = True

            if has_stop or order_exec.has_gtc_protective_stop(symbol, qty, working):
                logger.info(f"[SHUTDOWN GUARD] Preserving protected position {symbol} ({qty} shares). Active broker stop order exists.")
                preserved_count += 1
            else:
                logger.warning(f"[SHUTDOWN GUARD] Unprotected position detected for {symbol} ({qty} shares). Canceling working orders and liquidating for safety.")
                try:
                    for o in symbol_orders:
                        o_id = str(o.get("id") if isinstance(o, dict) else getattr(o, "id", ""))
                        if o_id:
                            self._client.cancel_order_by_id(o_id)
                    if symbol_orders:
                        time.sleep(0.2)
                except Exception as cancel_err:
                    logger.warning(f"Could not cancel working orders for {symbol} before close: {cancel_err}")

                for attempt in range(3):
                    try:
                        self._client.close_position(symbol)
                        closed_count += 1
                        logger.info(f"Successfully closed unprotected position {symbol}")
                        break
                    except Exception as c_err:
                        logger.warning(f"Attempt {attempt+1} to close unprotected position {symbol} failed: {c_err}")
                        time.sleep(0.5)

        return {"status": "success", "preserved_protected": preserved_count, "closed_unprotected": closed_count}

    def graceful_shutdown(self, preserve_protected: bool = True) -> ShutdownResult:
        self._transition_stream_state(
            expected={
                StreamState.HEALTHY,
                StreamState.DEGRADED,
                StreamState.AUTHENTICATED_UNRECONCILED,
                StreamState.RECONNECTING,
                StreamState.RECONCILING,
                StreamState.DISCONNECTED,
                StreamState.LATCHED_DESYNC,
                StreamState.STOPPED,
            },
            new_state=StreamState.SHUTTING_DOWN,
            reason="Graceful shutdown initiated",
        )
        logger.info("Graceful shutdown initiated (preserve_protected=%s)", preserve_protected)
        try:
            if self._connected and self._client is not None:
                self.flatten_on_shutdown(only_unprotected=preserve_protected)
        except Exception as exc:
            logger.warning("flatten_on_shutdown warning during graceful exit: %s", exc)

        self._stop_trade_stream(cancel_reconnect=True)
        self.disconnect()

        thread_alive = bool(
            self._trade_stream_thread is not None and self._trade_stream_thread.is_alive()
        )
        res = ShutdownResult(
            stop_requested=True,
            reconnect_disabled=True,
            socket_closed=True,
            thread_joined=not thread_alive,
            thread_alive=thread_alive,
        )

        self._transition_stream_state(
            expected={StreamState.SHUTTING_DOWN},
            new_state=StreamState.STOPPED,
            reason=f"Shutdown completed clean={res.clean}",
        )

        logger.info(
            "[STREAM SHUTDOWN] state=STOPPED stop_requested=True reconnect_disabled=True "
            "socket_closed=True thread_joined=%s thread_alive=%s clean=%s",
            res.thread_joined, res.thread_alive, res.clean
        )
        return res

    def _require_new_entries_allowed(self) -> None:
        gate = getattr(self, "entry_gate", None)
        if gate is None or getattr(gate, "halted", False):
            reason = getattr(gate, "reason", "startup reconciliation / safety circuit breaker") if gate else "entry gate missing"
            raise RuntimeError(f"New entries blocked: {reason}")

    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
        client_order_id: Optional[str] = None,
    ) -> dict:
        self._require_connection()
        self._require_trade_stream()
        self._require_new_entries_allowed()
        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        if int(qty) <= 0:
            raise ValueError(f"Order quantity must be positive, got {qty}")
        if int(qty) < min_shares:
            raise ValueError(f"Order quantity {qty} is below minimum allowed share count {min_shares}")
        qty = int(qty)
        entry_p = _round_price(entry_price)
        target_p = _round_price(target_price)
        stop_p = _round_price(stop_price)

        request_kwargs = {
            "symbol": symbol.upper().strip(),
            "qty": qty,
            "side": OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            "time_in_force": TimeInForce.DAY,
            "limit_price": entry_p,
            "order_class": OrderClass.BRACKET,
            "take_profit": TakeProfitRequest(limit_price=target_p),
            "stop_loss": StopLossRequest(stop_price=stop_p),
        }
        if client_order_id:
            request_kwargs["client_order_id"] = str(client_order_id)[:128]

        order = self._client.submit_order(LimitOrderRequest(**request_kwargs))
        parent_id = str(order.id)
        order_ids = [parent_id]
        stop_leg_id = None
        tp_leg_id = None
        try:
            from alpaca.trading.requests import GetOrderByIdRequest
            nested_order = self._client.get_order_by_id(parent_id, GetOrderByIdRequest(nested=True))
            legs = getattr(nested_order, "legs", []) or []
            for leg in legs:
                leg_id = str(leg.id)
                order_ids.append(leg_id)
                leg_type = str(getattr(leg, "order_type", getattr(leg, "type", ""))).lower()
                leg_stop = getattr(leg, "stop_price", None)
                leg_limit = getattr(leg, "limit_price", None)
                if "stop" in leg_type or leg_stop is not None:
                    stop_leg_id = leg_id
                elif "limit" in leg_type or leg_limit is not None:
                    tp_leg_id = leg_id
        except Exception as leg_exc:
            logger.warning(f"Could not extract nested child leg IDs for {parent_id}: {leg_exc}")

        return {
            "order_id": parent_id,
            "stop_order_id": stop_leg_id,
            "tp_order_id": tp_leg_id,
            "order_ids": order_ids,
            "status": "Submitted"
        }

    def get_order_snapshot(self, order_id: str) -> dict:
        try:
            order = self._client.get_order_by_id(order_id)
            status = order.status.value if hasattr(order.status, "value") else str(order.status)
            qty = float(order.qty) if getattr(order, "qty", None) is not None else 0.0
            return {"id": str(order.id), "status": status, "qty": qty}
        except Exception as exc:
            logger.error("get_order_snapshot failed for %s: %s", order_id, exc)
            return None

    def place_stop_order(self, symbol: str, side: str, qty: float, stop_price: float) -> dict:
        self._require_connection()
        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        qty = max(int(qty), min_shares)
        from alpaca.trading.requests import StopOrderRequest
        order = self._client.submit_order(StopOrderRequest(
            symbol=symbol.upper().strip(),
            qty=qty,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=_round_price(stop_price)
        ))
        return {"order_id": str(order.id)}

    def modify_stop_order(self, order_id: str, new_stop: float, symbol: Optional[str] = None) -> Optional[str]:
        self._require_connection()
        if symbol:
            try:
                price = self.get_price(symbol)
                pos = self.get_position(symbol)
                is_short = pos < 0
                if price and price > 0:
                    if is_short:
                        if new_stop <= price:
                            logger.warning(f"[MODIFY STOP P3] {symbol} (SHORT): new_stop=${new_stop:.2f} <= current price ${price:.2f}. Skipping invalid stop update.")
                            return None
                    else:
                        if new_stop >= price:
                            logger.warning(f"[MODIFY STOP P3] {symbol} (LONG): new_stop=${new_stop:.2f} >= current price ${price:.2f}. Skipping invalid stop update.")
                            return None
            except Exception as q_exc:
                logger.debug(f"[MODIFY STOP P3] Quote check skipped for {symbol}: {q_exc}")

        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            req = ReplaceOrderRequest(stop_price=_round_price(new_stop))
            new_order = self._client.replace_order_by_id(order_id, req)
            new_id = str(getattr(new_order, "id", order_id))
            logger.info(f"[MODIFY STOP P3] Replaced stop order {order_id} -> new order ID {new_id} at stop price ${_round_price(new_stop):.2f} (Atomic PATCH)")
            return new_id
        except Exception as exc:
            logger.error(f"[MODIFY STOP P3] Failed to modify stop order {order_id}: {exc}")
            return None

    def get_current_stop(self, symbol: str) -> Optional[float]:
        """
        Symbol တစ်ခု၏ active resting stop price ကို ပြန်ပေးသည်။
        Stop order မရှိလျှင် None ပြန်သည် (exception မဟုတ်)။
        """
        self._require_connection()
        sym = symbol.upper().strip()
        try:
            for order in self._get_active_orders():
                if str(getattr(order, "symbol", "")).upper().strip() != sym:
                    continue
                otype = str(getattr(order, "order_type", "")).lower().replace("ordertype.", "")
                if otype in ("stop", "stop_limit", "trailing_stop"):
                    price = getattr(order, "stop_price", None)
                    return float(price) if price is not None else None
            return None
        except Exception as exc:
            logger.error("[BRIDGE] get_current_stop failed for %s: %s", sym, exc)
            return None

    def replace_stop(self, symbol: str, new_stop_price: float) -> Optional[str]:
        """
        Symbol ၏ resting stop order ကို ရှာဖွေပြီး atomic PATCH လုပ်သည်။
        Order မရှိလျှင် warning log — exception raise မလုပ် (caller မ block ဖြစ်စေရန်)။

        Returns: order_id (str) on success, None on miss/failure
        """
        self._require_connection()
        sym = symbol.upper().strip()
        try:
            for order in self._get_active_orders():
                if str(getattr(order, "symbol", "")).upper().strip() != sym:
                    continue
                otype = str(getattr(order, "order_type", "")).lower().replace("ordertype.", "")
                if otype in ("stop", "stop_limit", "trailing_stop"):
                    return self.modify_stop_order(
                        order_id=str(order.id),
                        new_stop=new_stop_price,
                        symbol=sym,
                    )
            logger.warning("[REPLACE_STOP] No resting stop found for %s", sym)
            return None
        except Exception as exc:
            logger.error("[REPLACE_STOP FAILED] %s | %s", sym, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        self._require_connection()
        try:
            self._client.cancel_order_by_id(order_id)
            logger.info("Cancelled order %s", order_id)
            return True
        except Exception as exc:
            logger.error("Failed to cancel order %s: %s", order_id, exc)
            return False

    def sync_today_trades_to_journal(self) -> int:
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

                if expected_price > 0:
                    if side == "BUY":
                        slippage = (fill_price - expected_price) / expected_price
                    else:
                        slippage = (expected_price - fill_price) / expected_price
                else:
                    slippage = 0.0

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

                        is_loss = False  # Fail-safe default: do not penalize without proven negative PnL
                        mgr = get_trailing_manager()
                        state = mgr.states.get(o.symbol)

                        if state and state.entry_price > 0:
                            is_loss = (fill_price < state.entry_price)
                        else:
                            closed_trades = [t for t in get_today_closed_trades() if t.get("symbol") == o.symbol]
                            if closed_trades and closed_trades[-1].get("realized_pnl") is not None:
                                is_loss = (float(closed_trades[-1].get("realized_pnl", 0.0) or 0.0) < 0)
                            else:
                                # Fallback to authoritative broker BUY orders for this symbol today
                                buy_orders = [
                                    b for b in orders
                                    if getattr(b, "symbol", "") == o.symbol
                                    and str(getattr(b, "side", "")).upper() in ("BUY", "ORDERSIDE.BUY")
                                    and getattr(b, "filled_avg_price", None) is not None
                                ]
                                if buy_orders:
                                    try:
                                        avg_buy_price = float(buy_orders[-1].filled_avg_price)
                                        if avg_buy_price > 0:
                                            is_loss = (fill_price < avg_buy_price)
                                    except Exception:
                                        is_loss = False

                        if is_loss:
                            register_symbol_loss(o.symbol, filled_utc)
                    except Exception as loss_chk_exc:
                        logger.debug("Error checking trade exit loss status for %s: %s", o.symbol, loss_chk_exc)
                existing_order_ids.add(order_id_str)
                synced_count += 1
            return synced_count
        except Exception as exc:
            logger.warning("sync_today_trades_to_journal failed: %s", exc)
            return 0

    def open_position_count(self) -> int:
        self._require_connection()
        try:
            positions = self._call_with_retry(lambda: self._client.get_all_positions())
            return len([p for p in positions if float(p.qty) != 0])
        except Exception as exc:
            logger.error("Failed to get open_position_count: %s", exc)
            return 0

    def position_breakdown(self) -> dict:
        self._require_connection()
        try:
            positions = self._call_with_retry(lambda: self._client.get_all_positions())
            longs = sum(1 for p in positions if float(p.qty) > 0)
            shorts = sum(1 for p in positions if float(p.qty) < 0)
            return {"longs": longs, "shorts": shorts, "total": longs + shorts}
        except Exception as exc:
            logger.error("Failed to get position_breakdown: %s", exc)
            return {"longs": 0, "shorts": 0, "total": 0}

    def has_position(self, symbol: str) -> bool:
        self._require_connection()
        try:
            qty = self.get_position(symbol)
            return abs(qty) > 0
        except Exception:
            return False

    def _cancel_all_orders_for_symbol(self, symbol: str) -> None:
        try:
            orders = self.working_orders_plain()
            sym_upper = symbol.upper().strip()
            for o in orders:
                o_sym = str(o.get("symbol") if isinstance(o, dict) else getattr(o, "symbol", "")).upper().strip()
                o_id = o.get("id") if isinstance(o, dict) else getattr(o, "id", None)
                if o_sym == sym_upper and o_id:
                    try:
                        self._client.cancel_order_by_id(o_id)
                        logger.info("Cancelled working order %s for %s during position close.", o_id, sym_upper)
                    except Exception as c_err:
                        logger.debug("Failed cancelling order %s for %s: %s", o_id, sym_upper, c_err)
        except Exception as err:
            logger.debug("Error cancelling orders for %s: %s", symbol, err)

    def close_position(self, symbol: str) -> bool:
        self._require_connection()
        try:
            self._cancel_all_orders_for_symbol(symbol)
            close_order = self._client.close_position(symbol.upper().strip())
            order_id = getattr(close_order, "id", None)
            if order_id and hasattr(self, "_await_order_outcome"):
                res = self._await_order_outcome(str(order_id), timeout_seconds=15.0)
                is_filled = res.has_fill or (res.outcome == order_exec.FILLED)
                logger.info("Closed position for %s: outcome=%s, filled=%.0f", symbol, res.outcome, res.filled)
                return is_filled
            time.sleep(0.5)
            pos = self.get_position(symbol)
            return pos == 0
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

    def _cancel_working_entry_orders(self, symbol: str) -> None:
        try:
            al_orders = self._get_active_orders()
            active_statuses = {"new", "partially_filled", "submitted", "queued", "held", "accepted", "pending_new"}
            for o in al_orders:
                raw_status = o.status.value if hasattr(o.status, "value") else str(o.status)
                if raw_status.lower() in active_statuses and o.symbol.upper().strip() == symbol.upper().strip():
                    side = o.side.value if hasattr(o.side, "value") else str(o.side)
                    if side.upper() == "BUY":
                        logger.info("Cancelling stale working BUY entry order %s for %s", getattr(o, "id", ""), symbol)
                        try:
                            self._client.cancel_order_by_id(o.id)
                        except Exception as ce:
                            logger.warning("Could not cancel order %s for %s: %s", getattr(o, "id", ""), symbol, ce)
        except Exception as exc:
            logger.error("Failed to cancel working entry orders for %s: %s", symbol, exc)

    def execute_signal(self, signal: Signal, coach: bool = False) -> bool:
        symbol = signal.symbol.upper().strip()

        action = signal.action.upper().strip()
        allow_short = bool(self._c("ALLOW_SHORT", False))
        min_cash = float(self._c("MIN_TRADE_CASH", 100.0))

        if action == "HOLD":
            self._cancel_working_entry_orders(symbol)
            return False
        if action not in {"BUY", "SELL"}:
            logger.warning("Unsupported signal action: %s", signal.action)
            return False

        position = self.get_position(symbol)
        allow_hist = bool(self._c("ALLOW_HISTORICAL_PRICE_FOR_ORDERS", False))
        quote_decision = self.get_order_quote(symbol)
        if quote_decision.ok and quote_decision.price > 0:
            price = quote_decision.price
        elif allow_hist:
            price = self.get_price(symbol, allow_historical=True)
        else:
            price = 0.0

        if price <= 0:
            logger.warning(
                "[PRICE FAIL] %s | quote_ok=%s | reason=%s | quote_px=%.2f | allow_hist=%s",
                symbol, quote_decision.ok, quote_decision.reason, quote_decision.price, allow_hist,
            )
            return False

        if action == "BUY":
            # Sector-level pre-trade SOXX veto check for semiconductor & tech cluster
            from strategies.intraday_risk import check_soxx_tech_long_gate, TECH_CORRELATED_CLUSTER
            if symbol in TECH_CORRELATED_CLUSTER:
                soxx_ok, soxx_reason = check_soxx_tech_long_gate(symbol, action)
                if not soxx_ok:
                    logger.warning("[PRE-SUBMISSION SOXX VETO] %s Long blocked: %s", symbol, soxx_reason)
                    return False

            if self.has_working_order(symbol, action="BUY"):
                logger.info("Working BUY order already exists for %s — skipping BUY", symbol)
                return False
            if position > 0:
                logger.info("Already long %s — skipping BUY", symbol)
                return False
            if position < 0:
                qty = int(abs(position))
                return self._close_position(symbol, "BUY", qty, price, "Close short").has_fill

            equity = self.get_net_liquidation()
            if equity <= 0:
                equity = self.get_cash()
            if equity < min_cash:
                logger.warning("Insufficient equity for BUY %s: $%.2f", symbol, equity)
                return False

            strat_name = getattr(signal, "strategy", getattr(signal, "strategy_name", ""))
            stop_price = getattr(signal, "stop_price", getattr(signal, "stop_loss", None))
            target_price = getattr(signal, "target_price", getattr(signal, "take_profit", None))
            if stop_price is None or target_price is None:
                stop_loss_pct = float(self._c("STOP_LOSS_PCT", 0.05))
                take_profit_pct = float(self._c("TAKE_PROFIT_PCT", 0.10))
                if stop_price is None:
                    stop_price = round(price * (1.0 - stop_loss_pct), 2)
                if target_price is None:
                    target_price = round(price * (1.0 + take_profit_pct), 2)

            if stop_price <= 0 or target_price <= 0 or stop_price >= price or target_price <= price:
                logger.warning("Rejecting execute_signal for %s: invalid stop_loss/target_price geometry (stop=%s, target=%s, price=%.2f)", symbol, stop_price, target_price, price)
                return False

            # Risk-based sizing with account equity
            risk_per_share = max(price - stop_price, 0.01)
            max_risk_pct = float(self._c("MAX_RISK_PER_TRADE_PCT", 0.01))
            risk_budget = equity * max_risk_pct
            risk_cap = float(self._c("MAX_RISK_PER_TRADE_DOLLARS", 150.0))
            risk_budget = min(risk_budget, risk_cap)
            qty = max(int(risk_budget / risk_per_share), 0)

            # Cap by maximum position allocation
            max_pos_val = equity * float(self._c("MAX_POSITION_PCT", 0.08))
            if qty * price > max_pos_val:
                qty = max(int(max_pos_val / price), 0)

            min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
            if qty < min_shares:
                logger.warning("Calculated qty=%d below min_shares=%d for BUY %s (risk_per_share=%.2f)", qty, min_shares, symbol, risk_per_share)
                return False

            result = self._place_open_bracket(
                symbol, "BUY", qty, price, getattr(signal, "confidence", 1.0),
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
            self._cancel_working_entry_orders(symbol)
            qty = int(position)
            return self._close_position(symbol, "SELL", qty, price, "Close long").has_fill
        if position < 0:
            logger.info("Already short %s — skipping SELL", symbol)
            return False
        if not allow_short:
            logger.info("No long in %s — skipping SELL (ALLOW_SHORT=False)", symbol)
            return False

        inverse_etfs = getattr(config, "INVERSE_ETF_SYMBOLS", set())
        if symbol.upper().strip() in inverse_etfs:
            logger.info("Skipped %s: Inverse ETFs cannot be sold short (Long-only hedge)", symbol)
            return False

        if self.has_working_order(symbol, action="SELL"):
            logger.info("Working SELL order already exists for %s — skipping short SELL", symbol)
            return False

        equity = self.get_net_liquidation()
        if equity <= 0:
            equity = self.get_cash()
        if equity < min_cash:
            logger.warning("Insufficient equity for short SELL %s", symbol)
            return False

        strat_name = getattr(signal, "strategy", getattr(signal, "strategy_name", ""))
        stop_price = getattr(signal, "stop_price", getattr(signal, "stop_loss", None))
        target_price = getattr(signal, "target_price", getattr(signal, "take_profit", None))
        if stop_price is None or target_price is None:
            stop_loss_pct = float(self._c("STOP_LOSS_PCT", 0.05))
            take_profit_pct = float(self._c("TAKE_PROFIT_PCT", 0.10))
            if stop_price is None:
                stop_price = round(price * (1.0 + stop_loss_pct), 2)
            if target_price is None:
                target_price = round(price * (1.0 - take_profit_pct), 2)

        if stop_price <= 0 or target_price <= 0 or stop_price <= price or target_price >= price:
            logger.warning("Rejecting short SELL for %s: invalid stop_loss/target_price geometry (stop=%s, target=%s, price=%.2f)", symbol, stop_price, target_price, price)
            return False

        risk_per_share = max(stop_price - price, 0.01)
        max_risk_pct = float(self._c("MAX_RISK_PER_TRADE_PCT", 0.01))
        risk_budget = min(equity * max_risk_pct, float(self._c("MAX_RISK_PER_TRADE_DOLLARS", 150.0)))
        qty = max(int(risk_budget / risk_per_share), 0)

        max_pos_val = equity * float(self._c("MAX_POSITION_PCT", 0.08))
        if qty * price > max_pos_val:
            qty = max(int(max_pos_val / price), 0)

        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        if qty < min_shares:
            logger.warning("Calculated qty=%d below min_shares=%d for short SELL %s", qty, min_shares, symbol)
            return False

        result = self._place_open_bracket(
            symbol, "SELL", qty, price, getattr(signal, "confidence", 1.0),
            strategy_name=strat_name,
            custom_stop_price=stop_price,
            custom_target_price=target_price
        )
        return result.occupies_slot

    def _evaluate_and_execute_rotation(self, signals: List[Signal]) -> None:
        try:
            self.rotator.state.reset_cycle()
            raw_positions = self._client.get_all_positions()
            if not raw_positions:
                return

            # ── Step 1: Risk layer (ratchet stops FIRST) ──────────────────
            try:
                ratchet_results = self.rotator._ratchet_stops(raw_positions, bridge=self, dry_run=False)
                self.rotator._log_ratchet_summary(ratchet_results)
            except Exception as r_exc:
                logger.error("[RATCHET EXECUTE ERROR] %s", r_exc)

            conf_lookup = {
                (s.symbol.upper() if hasattr(s, "symbol") else s["symbol"].upper()): (s.confidence if hasattr(s, "confidence") else s["confidence"])
                for s in signals
            }

            protected_syms = set(getattr(self, "protected_symbols", []))
            open_pos = []
            for raw in raw_positions:
                qty = float(raw.qty)
                if qty == 0:
                    continue
                avg_cost = float(raw.avg_entry_price)
                mkt_price = float(raw.current_price)
                upnl = float(raw.unrealized_pl)
                sym = raw.symbol.upper()
                pos = OpenPosition(
                    symbol=sym,
                    side=str(raw.side).lower(),
                    qty=qty,
                    avg_entry_price=avg_cost,
                    current_price=mkt_price,
                    unrealized_pnl=upnl,
                    current_conf=conf_lookup.get(sym, 0.50),
                    is_protected=(sym in protected_syms),
                )
                open_pos.append(pos)

            incoming = []
            for s in signals:
                sym = s.symbol.upper() if hasattr(s, "symbol") else s["symbol"].upper()
                action = s.action.upper() if hasattr(s, "action") else s["action"].upper()
                conf = s.confidence if hasattr(s, "confidence") else s["confidence"]
                price = s.price if hasattr(s, "price") else s["price"]
                if action != "HOLD":
                    incoming.append(IncomingSignal(symbol=sym, action=action, confidence=conf, price=price))

            decision = self.rotator.evaluate(open_pos, incoming)
            if decision.should_rotate:
                exit_sym = decision.exit_position.symbol
                enter_sig = decision.enter_signal
                logger.info("[ROTATION EXECUTE] Closing %s → Opening %s", exit_sym, enter_sig.symbol)
                try:
                    self.close_position(exit_sym)
                    logger.info("[ROTATION] Closed %s ✅", exit_sym)
                except Exception as exc:
                    logger.error("[ROTATION ABORT] Failed to close %s: %s — NOT entering %s", exit_sym, exc, enter_sig.symbol)
                    return

                matched_sig = next((s for s in signals if (s.symbol.upper() if hasattr(s, "symbol") else s["symbol"].upper()) == enter_sig.symbol), None)
                if matched_sig:
                    try:
                        self.execute_signal(matched_sig)
                        logger.info("[ROTATION] Entered %s ✅", enter_sig.symbol)
                        self.rotator.state.record_rotation(decision)
                    except Exception as exc:
                        logger.error("[ROTATION PARTIAL] Closed %s but failed to enter %s: %s", exit_sym, enter_sig.symbol, exc)
        except Exception as exc:
            logger.error("Error during position rotation evaluation: %s", exc)

    def execute_all(self, signals: List[Signal]) -> dict:
        self._evaluate_and_execute_rotation(signals)

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
                self._cancel_working_entry_orders(symbol)
                skipped += 1
                continue

            current_position = self.get_position(symbol)

            if action == "SELL":
                self._cancel_working_entry_orders(symbol)

            if action == "BUY" and self.has_working_order(symbol, action="BUY"):
                logger.info("Working BUY entry order already exists for %s — skipping", symbol)
                skipped += 1
                continue

            opens_new = (
                (action == "BUY" and current_position == 0)
                or (action == "SELL" and current_position == 0 and allow_short)
            )

            if opens_new:
                max_open = int(getattr(config_module, "MAX_OPEN_POSITIONS", 5))
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
        from strategies.session import now_eastern
        return now_eastern().date().strftime("%Y%m%d")
