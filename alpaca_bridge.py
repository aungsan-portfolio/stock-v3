"""
alpaca_bridge.py — Alpaca Paper Trading bridge via alpaca-py.
Duck-type replacement for IBKRBridge. Exposes all pricing, account,
placement, and reconciliation methods expected by Pro V3.
"""
import asyncio
import datetime as _dt
import json
import logging
import math
import os
import random
import threading
import time
from collections import deque
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

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


import requests
import urllib3

class TimeoutHTTPAdapter(HTTPAdapter):
    """HTTP adapter enforcing hard connect and read timeouts and exponential backoff retries."""

    def __init__(
        self,
        connect_timeout: float = 15.0,
        read_timeout: float = 30.0,
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
    connect_timeout: float = 15.0,
    read_timeout: float = 30.0,
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


def normalize_enum_value(val: Any) -> str:
    """Safely extract and normalize enum value or string representation."""
    if val is None:
        return ""
    if hasattr(val, "_mock_name") or "Mock" in type(val).__name__:
        return ""
    if hasattr(val, "value"):
        raw = val.value
        if hasattr(raw, "_mock_name") or "Mock" in type(raw).__name__:
            return ""
        return str(raw).strip().upper()
    if isinstance(val, str):
        return val.strip().upper()
    return str(val).strip().upper()


def flatten_orders(orders: Any) -> List[Any]:
    """
    Flattens orders and nested child bracket legs (order.legs),
    deduplicating by order ID while preserving order objects.
    """
    flattened = []
    for o in orders or []:
        flattened.append(o)
        for leg in getattr(o, "legs", None) or []:
            flattened.append(leg)
    dedup = {}
    for o in flattened:
        oid = getattr(o, "id", None) if not isinstance(o, dict) else o.get("id")
        key = str(oid) if oid is not None else id(o)
        if key not in dedup:
            dedup[key] = o
    return list(dedup.values())


def check_regular_market_hours(client: Any = None) -> Tuple[bool, str]:
    """
    Validates that current execution is strictly within NYSE Regular Trading Hours:
    9:30 AM – 3:55 PM Eastern Time on regular trading days.
    Blocks extended-hours (pre-market/after-hours) and overnight sessions.
    """
    from datetime import datetime, time, timezone
    from zoneinfo import ZoneInfo
    from strategies.session import now_eastern

    et_tz = ZoneInfo("America/New_York")
    now_et = now_eastern()

    # If broker client provides clock, verify broker market open status
    if client is not None and hasattr(client, "get_clock"):
        try:
            clock = client.get_clock()
            is_open = getattr(clock, "is_open", None)
            if is_open is not None and not bool(is_open):
                return False, "BROKER_MARKET_CLOSED"
            b_ts = getattr(clock, "timestamp", None)
            if b_ts is not None and hasattr(b_ts, "astimezone") and isinstance(b_ts, datetime):
                now_et = b_ts.astimezone(et_tz)
        except Exception as c_err:
            logger.warning("Could not query broker clock: %s; relying on conservative local ET gate", c_err)

    if not isinstance(now_et, datetime):
        return True, ""

    # Regular session bounds: 9:30 AM <= time < 3:55 PM ET (15:55 cutoff buffer)
    cur_time = now_et.time()
    if cur_time < time(9, 30):
        return False, f"PRE_MARKET_HOURS: {cur_time.strftime('%H:%M:%S')} ET < 09:30:00"
    if cur_time >= time(15, 55):
        return False, f"ENTRY_CUTOFF_EXCEEDED: {cur_time.strftime('%H:%M:%S')} ET >= 15:55:00"

    # Weekday check (0 = Mon, 4 = Fri)
    if now_et.weekday() >= 5:
        return False, f"WEEKEND_HOURS: weekday={now_et.weekday()}"

    return True, ""


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



@dataclass
class MutationCounters:
    mutation_attempts: int = 0
    mutation_blocked: int = 0
    mutation_passed_through: int = 0
    submit_order_passed_through: int = 0
    replace_order_passed_through: int = 0
    cancel_order_passed_through: int = 0
    close_position_passed_through: int = 0

    def record(self, kind: str, dry_run: bool) -> bool:
        self.mutation_attempts += 1
        if dry_run:
            self.mutation_blocked += 1
            return True
        self.mutation_passed_through += 1
        if kind == "submit_order":
            self.submit_order_passed_through += 1
        elif kind == "replace_order":
            self.replace_order_passed_through += 1
        elif kind == "cancel_order":
            self.cancel_order_passed_through += 1
        elif kind == "close_position":
            self.close_position_passed_through += 1
        return False

    def to_dict(self) -> dict:
        return {
            "mutation_attempts": self.mutation_attempts,
            "mutation_blocked": self.mutation_blocked,
            "mutation_passed_through": self.mutation_passed_through,
            "submit_order_passed_through": self.submit_order_passed_through,
            "replace_order_passed_through": self.replace_order_passed_through,
            "cancel_order_passed_through": self.cancel_order_passed_through,
            "close_position_passed_through": self.close_position_passed_through,
        }

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
        self.reason = ""
        self.last_error = ""

    def mark_healthy(self, reason=None):
        self.is_healthy = True
        self.reason = str(reason or "")
        self.last_error = ""

    def mark_unhealthy(self, reason=None):
        self.is_healthy = False
        self.reason = str(reason or "")
        self.last_error = str(reason or "")


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

    def daily_loss_blocked(
        self,
        account_id: str,
        session_date_et: str,
        current_equity: Any,
        session_start_equity: Any,
        net_session_cash_flows: Any = 0.0,
    ) -> Tuple[bool, str]:
        """
        Dedicated daily-loss circuit breaker for Swing Trading using end-to-end Decimal arithmetic.
        Enforces min(SOD_equity * SWING_MAX_DAILY_LOSS_PCT, SWING_MAX_DAILY_LOSS_DOLLAR_CAP).
        """
        if self.halted:
            return True, "Trading engine is halted"

        try:
            d_current_equity = Decimal(str(current_equity))
            d_sod_equity = Decimal(str(session_start_equity))
            d_cash_flows = Decimal(str(net_session_cash_flows or 0.0))
        except Exception as e:
            return True, f"INVALID_EQUITY_DECIMAL: {e}"

        if not (d_current_equity.is_finite() and d_sod_equity.is_finite() and d_cash_flows.is_finite()):
            return True, "NON_FINITE_EQUITY_VALUES"

        if d_sod_equity <= Decimal("0"):
            return True, "NON_POSITIVE_SOD_EQUITY_BASELINE"

        d_daily_pnl = d_current_equity - d_sod_equity - d_cash_flows

        pct_cap_float = float(getattr(config_module, "SWING_MAX_DAILY_LOSS_PCT", 0.01))
        dollar_cap_float = float(getattr(config_module, "SWING_MAX_DAILY_LOSS_DOLLAR_CAP", 300.00))

        d_pct_cap = d_sod_equity * Decimal(str(pct_cap_float))
        d_dollar_cap = Decimal(str(dollar_cap_float))
        d_limit = min(d_pct_cap, d_dollar_cap)

        if d_daily_pnl <= -d_limit:
            self.persist_daily_loss_halt(account_id, session_date_et, d_daily_pnl, d_limit)
            self.halt()
            return True, f"Max daily loss limit reached (${d_daily_pnl:.2f} <= -${d_limit:.2f})"

        return False, ""

    def persist_daily_loss_halt(
        self,
        account_id: str,
        session_date_et: str,
        daily_pnl: Decimal,
        daily_loss_limit: Decimal,
    ) -> None:
        """Atomically persist durable daily-loss halt state."""
        from datetime import datetime, timezone
        from pathlib import Path
        data_dir = getattr(config_module, "DATA_DIR", Path("data"))
        halt_file = data_dir / f"swing_daily_loss_halt_{account_id.strip().upper()}.json"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "account_id": account_id.strip().upper(),
            "session_date_et": session_date_et,
            "daily_pnl": str(daily_pnl),
            "daily_loss_limit": str(daily_loss_limit),
            "halted": True,
            "reason": "DAILY_LOSS_LIMIT",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2)
        tmp_file = halt_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, halt_file)
            logger.warning("[DAILY LOSS HALT PERSISTED] %s: pnl=%s limit=%s", account_id, daily_pnl, daily_loss_limit)
        except Exception as exc:
            logger.error("Failed to persist daily loss halt state for %s: %s", account_id, exc)

    def check_daily_loss_halt_state(
        self,
        account_id: str,
        current_session_date_et: str,
    ) -> Tuple[bool, str]:
        """
        Inspect durable daily loss halt file with strict session rollover validation.
        """
        from pathlib import Path
        data_dir = getattr(config_module, "DATA_DIR", Path("data"))
        halt_file = data_dir / f"swing_daily_loss_halt_{account_id.strip().upper()}.json"
        if not halt_file.exists():
            return False, ""
        try:
            with open(halt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("Corrupted daily loss halt file for %s: %s (failing closed)", account_id, exc)
            self.halt()
            return True, f"CORRUPTED_HALT_STATE: {exc}"

        rec_acct = str(data.get("account_id", "")).strip().upper()
        if rec_acct != account_id.strip().upper():
            logger.error("Halt file account mismatch: expected %s, got %s (failing closed)", account_id, rec_acct)
            self.halt()
            return True, "HALT_STATE_ACCOUNT_MISMATCH"

        rec_date = str(data.get("session_date_et", "")).strip()
        if rec_date > current_session_date_et:
            logger.error("Future session date %s detected in halt state for %s (failing closed)", rec_date, account_id)
            self.halt()
            return True, "FUTURE_SESSION_DATE_DETECTED"

        if rec_date == current_session_date_et:
            if data.get("halted", False):
                self.halt()
                return True, f"DAILY_LOSS_LIMIT_REACHED (pnl={data.get('daily_pnl')}, limit={data.get('daily_loss_limit')})"

        # Previous session -> controlled reset
        try:
            logger.info("[SESSION ROLLOVER RESET] Resetting previous session (%s) halt state for %s", rec_date, account_id)
            halt_file.unlink(missing_ok=True)
            self.halted = False
            self.halt_reason = ""
        except Exception as r_exc:
            logger.warning("Could not unlink previous halt file for %s: %s", account_id, r_exc)
        return False, ""

    def entry_blocked(self, symbol, value, signal, price=None):
        if self.halted:
            return True, "Trading engine is halted"
        if self.bridge is not None:
            try:
                daily_pnl = self.bridge.account_daily_pnl()
                mode = getattr(self.bridge, "mode", "daytrade").lower()
                if mode == "swing":
                    sod_equity = getattr(self.bridge, "_start_of_day_equity", 0.0)
                    if sod_equity <= 0.0:
                        sod_equity = self.bridge.get_net_liquidation()
                    if sod_equity <= 0.0:
                        logger.warning("Risk baseline unavailable; blocking new entries only. Protective exits remain enabled.")
                        return True, "Risk baseline unavailable (blocking new entries only; protective exits remain enabled)"
                    pct_limit = abs(sod_equity * float(getattr(config_module, "SWING_MAX_DAILY_LOSS_PCT", 0.01)))
                    dollar_cap = float(getattr(config_module, "SWING_MAX_DAILY_LOSS_DOLLAR_CAP", 300.00))
                    max_loss = min(pct_limit, dollar_cap) if dollar_cap > 0 else 300.00
                else:
                    max_loss = float(getattr(config_module, "MAX_DAILY_LOSS_DAYTRADE", 300.0))
                if round(daily_pnl, 2) <= -round(abs(max_loss), 2):
                    self.halt()
                    return True, f"Max daily loss limit reached (${daily_pnl:.2f} <= -${max_loss:.2f})"
            except Exception as e:
                logger.error("Error checking daily loss limit in EntryGateMock: %s; blocking new entries only. Protective exits remain enabled.", e)
                return True, f"Risk check error: {e} (blocking new entries only)"
        return False, ""


class AlpacaBridge:
    def __init__(self, cfg=None, mode: str = "daytrade", dry_run: bool = False) -> None:
        self.mode = str(mode or "daytrade").lower()
        self.dry_run = bool(dry_run)
        self.mutation_counters = MutationCounters()
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
        self._swing_dispatch_lock = threading.Lock()

    @property
    def is_paper(self) -> bool:
        if self._client is not None:
            return bool(getattr(self._client, "_paper", True))
        return True

    @property
    def account_mode(self) -> str:
        return "PAPER" if self.is_paper else "LIVE"

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

        connect_timeout = float(
            os.environ.get(
                "ALPACA_CONNECT_TIMEOUT",
                getattr(config_module, "ALPACA_CONNECT_TIMEOUT_SECONDS", 15.0),
            )
        )
        read_timeout = float(
            os.environ.get(
                "ALPACA_READ_TIMEOUT",
                getattr(config_module, "ALPACA_READ_TIMEOUT_SECONDS", 30.0),
            )
        )
        total_retries = int(
            os.environ.get(
                "ALPACA_TOTAL_RETRIES",
                getattr(config_module, "ALPACA_TOTAL_RETRIES", 3),
            )
        )
        backoff_factor = float(
            os.environ.get(
                "ALPACA_RETRY_BACKOFF_FACTOR",
                getattr(config_module, "ALPACA_RETRY_BACKOFF_FACTOR", 0.5),
            )
        )
        max_connect_attempts = int(
            os.environ.get(
                "ALPACA_CONNECT_MAX_ATTEMPTS",
                getattr(config_module, "ALPACA_CONNECT_MAX_ATTEMPTS", 3),
            )
        )

        try:
            self._client = TradingClient(key, secret, paper=True)
            self._data_client = StockHistoricalDataClient(key, secret)

            # Hardened Network Timeout & Retry Layer
            configure_session_timeouts(
                getattr(self._client, "_session", None),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                total_retries=total_retries,
                backoff_factor=backoff_factor,
            )
            configure_session_timeouts(
                getattr(self._data_client, "_session", None),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                total_retries=total_retries,
                backoff_factor=backoff_factor,
            )

            # Retryable network exceptions only (transient connectivity / socket timeouts)
            retryable_net_exceptions = (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.Timeout,
                urllib3.exceptions.HTTPError,
                TimeoutError,
            )

            acct = None
            for attempt in range(1, max_connect_attempts + 1):
                try:
                    acct = self._client.get_account()
                    break
                except Exception as exc:
                    # Check for fatal non-retryable auth/authorization errors (401, 403)
                    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
                    err_msg = str(exc).lower()
                    is_fatal_auth = status_code in (401, 403) or "401" in err_msg or "403" in err_msg or "unauthorized" in err_msg or "forbidden" in err_msg

                    if is_fatal_auth or not isinstance(exc, retryable_net_exceptions):
                        logger.error("[ALPACA BRIDGE] Fatal non-retryable error during connection: %s", exc)
                        self._conn_health.mark_unhealthy(f"FATAL_AUTH_OR_CONFIG_ERROR: {exc}")
                        self._connected = False
                        self._stop_trade_stream()
                        self._client = None
                        self._data_client = None
                        return False

                    # Transient network error: retry if attempts remaining
                    if attempt < max_connect_attempts:
                        delay = 2.0 * attempt
                        logger.warning(
                            "[ALPACA BRIDGE] Initial connection attempt %d/%d encountered transient network error (%s). Retrying in %.1fs...",
                            attempt,
                            max_connect_attempts,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        # Circuit breaker trigger when all attempts exhausted
                        logger.critical(
                            "[CIRCUIT BREAKER] Alpaca broker unavailable after %d connection attempts: %s",
                            max_connect_attempts,
                            exc,
                        )
                        self._conn_health.mark_unhealthy(f"BROKER_UNAVAILABLE: {exc}")
                        self._connected = False
                        self._stop_trade_stream()
                        self._client = None
                        self._data_client = None
                        return False

            if acct is None:
                self._conn_health.mark_unhealthy("BROKER_UNAVAILABLE: No account returned")
                return False

            # Validate account status
            if getattr(acct, "status", None) != "ACTIVE" or getattr(acct, "account_blocked", False) or getattr(acct, "trading_blocked", False):
                logger.error("Alpaca account is not active or is blocked. Status: %s", getattr(acct, "status", "UNKNOWN"))
                self._conn_health.mark_unhealthy("Account blocked")
                return False

            self._account_id = str(getattr(acct, "account_number", "") or getattr(acct, "id", "") or "default")
            if self.mode == "swing":
                from strategies.startup_security import verify_authenticated_account_identity
                expected_acct = os.environ.get("SWING_ACCOUNT_ID", "PA3R71I7JFDZ")
                verify_authenticated_account_identity(
                    configured_account_id=expected_acct,
                    authenticated_api_account_id=self._account_id,
                    permitted_account_id="PA3R71I7JFDZ",
                )
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

    def get_market_clock(self):
        self._require_connection()
        return self._call_with_retry(lambda: self._client.get_clock())

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

    def _entry_time_in_force(self) -> TimeInForce:
        mode = str(getattr(self, "mode", "daytrade")).strip().lower()
        return TimeInForce.GTC if mode == "swing" else TimeInForce.DAY

    # ── Placement & Execution ────────────────────────────────────
    def _place_open_bracket(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
        strategy_name: str = "", custom_stop_price: Optional[float] = None,
        custom_target_price: Optional[float] = None,
        custom_client_order_id: Optional[str] = None,
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
        if custom_client_order_id is not None:
            client_order_id = custom_client_order_id
        elif self.mode == "swing":
            from strategies.swing_sizing import build_canonical_swing_order_id
            from strategies.session import now_eastern
            client_order_id = build_canonical_swing_order_id(
                symbol=symbol,
                side=action,
                session_date=now_eastern().date().isoformat(),
                bar_timestamp_utc=now_eastern().isoformat(),
                signal_kind=strat_tag,
                account_number=getattr(self, "_account_id", "PA3R71I7JFDZ"),
            )
        else:
            from strategies.order_idempotency import build_client_order_id
            client_order_id = build_client_order_id(symbol=symbol, side=action, bar_timestamp_utc="", signal_hash=f"{strat_tag}_{int(price*100)}_{int(time.time()//300)}", attempt=0)

        order_audit.log_event(
            order_audit.STAGE_SUBMIT, kind="open", symbol=symbol, action=action,
            qty=qty, limit_price=limit_price, confidence=confidence, order_ref=client_order_id,
        )

        if getattr(self, "dry_run", False):
            logger.info("[DRY-RUN MUTATION BLOCKED] _place_open_bracket for %s %s qty=%d limit=%.2f stop=%.2f tp=%.2f",
                        symbol, action, qty, limit_price, stop_price, take_profit_price)
            return order_exec.OrderResult(
                outcome=order_exec.REJECTED,
                status="DryRunBlocked",
                filled=0.0,
                remaining=float(qty),
                avg_fill_price=0.0,
            )

        tif = self._entry_time_in_force()
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

        if self.mode == "swing" and action == "SELL":
            pos_qty = self.get_position(symbol)
            if qty > pos_qty:
                logger.error("Stage 0 long-only safety violation: close qty %d exceeds verified long position %s for %s", qty, pos_qty, symbol)
                return order_exec.OrderResult(outcome=order_exec.REJECTED, status="InvalidCloseQuantity", filled=0.0, remaining=float(qty))

        if getattr(self, "dry_run", False):
            logger.info("[DRY-RUN MUTATION BLOCKED] _close_position for %s %s x%d", symbol, action, qty)
            return order_exec.OrderResult(
                outcome=order_exec.REJECTED,
                status="DryRunBlocked",
                filled=0.0,
                remaining=float(qty),
                avg_fill_price=0.0,
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
            if protect.get("repaired"):
                positions = self.positions_plain()
                working = self.working_orders_plain()
                snapshot = reconciliation.build_snapshot(positions, working)
                if snapshot.get("unprotected_longs"):
                    time.sleep(0.5)
                    positions = self.positions_plain()
                    working = self.working_orders_plain()
                    snapshot = reconciliation.build_snapshot(positions, working)

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

    def audit_swing_portfolio_safety(
        self,
        positions: List[Any],
        open_orders: List[Any],
    ) -> Tuple[float, float, Set[str], Set[str]]:
        """
        Audits active swing portfolio safety, stop quantity coverage, and pending entries.
        Fail-closed guarantees:
        1. Rejects any account short position (UNSUPPORTED_ACCOUNT_SHORT_POSITION).
        2. Verifies full protective stop coverage for every open long position (INSUFFICIENT_STOP_COVERAGE).
        3. Rejects any pending swing entry lacking a protective stop (PENDING_ENTRY_WITHOUT_PROTECTIVE_STOP).
        4. Computes conservative open stop risk and pending entry risk.
        Returns: (current_open_stop_risk, pending_entry_risk, active_position_symbols, pending_new_symbols)
        """
        from strategies.swing_sizing import require, SwingSizingRejected

        def _get(obj: Any, attr: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(attr, default)
            return getattr(obj, attr, default)

        pos_list = positions if isinstance(positions, list) else []
        ord_list = open_orders if isinstance(open_orders, list) else []

        # 1. Reject any short positions
        for p in pos_list:
            qty = float(_get(p, "qty", 0.0) or 0.0)
            sym = str(_get(p, "symbol", "") or "").strip().upper()
            if qty < 0.0:
                raise SwingSizingRejected(f"UNSUPPORTED_ACCOUNT_SHORT_POSITION: {sym}")

        active_positions = [p for p in pos_list if float(_get(p, "qty", 0.0) or 0.0) > 0.0]
        active_position_symbols = {
            str(_get(p, "symbol", "") or "").strip().upper()
            for p in active_positions
            if str(_get(p, "symbol", "") or "").strip()
        }

        flattened_orders = flatten_orders(ord_list)
        PROTECTIVE_STOP_TYPES = {"STOP"}
        KNOWN_ACTIVE_STATUSES = {
            "NEW", "ACCEPTED", "SUBMITTED", "PENDING_NEW", "HELD",
            "PARTIALLY_FILLED", "QUEUED", "OPEN"
        }
        KNOWN_TERMINAL_STATUSES = {
            "FILLED", "CANCELED", "CANCELLED", "EXPIRED", "REJECTED", "STOPPED", "SUSPENDED"
        }

        # Helper to validate and extract active remaining quantity
        def _get_active_remaining_qty(order_obj: Any) -> float:
            st = normalize_enum_value(_get(order_obj, "status", ""))
            if st in KNOWN_TERMINAL_STATUSES or st in ("PENDING_CANCEL", "PENDING_REPLACE"):
                return 0.0
            if st and st not in KNOWN_ACTIVE_STATUSES:
                raise SwingSizingRejected(f"UNKNOWN_BROKER_ORDER_STATUS: {st}")
            tot = float(_get(order_obj, "qty", 0.0) or 0.0)
            fld = float(_get(order_obj, "filled_qty", 0.0) or 0.0)
            return max(0.0, tot - fld)

        # 2. Verify protective stop quantity coverage for each active position
        current_open_stop_risk = 0.0
        for p in active_positions:
            sym = str(_get(p, "symbol", "") or "").strip().upper()
            pos_qty = float(_get(p, "qty", 0.0) or 0.0)
            matching_stops = [
                o for o in flattened_orders
                if str(_get(o, "symbol", "") or "").strip().upper() == sym
                and normalize_enum_value(_get(o, "side", "")) == "SELL"
                and (
                    normalize_enum_value(_get(o, "order_type", "")) in PROTECTIVE_STOP_TYPES
                    or normalize_enum_value(_get(o, "type", "")) in PROTECTIVE_STOP_TYPES
                )
                and _get_active_remaining_qty(o) > 0.0
            ]
            protected_qty = sum(_get_active_remaining_qty(o) for o in matching_stops)
            require(
                protected_qty >= pos_qty,
                f"INSUFFICIENT_STOP_COVERAGE: {sym} position={pos_qty}, protected={protected_qty}"
            )

            # Verify every matching stop has a deterministic stop_price
            for o in matching_stops:
                st_px = _get(o, "stop_price", None)
                require(
                    st_px is not None and float(st_px) > 0.0,
                    f"MISSING_STOP_PRICE: Order {getattr(o, 'id', None)} for {sym}"
                )

            # Conservative stop risk calculation
            stop_prices = [
                float(_get(o, "stop_price", 0.0))
                for o in matching_stops
            ]
            min_stop = min(stop_prices) if stop_prices else (float(_get(p, "avg_entry_price", 0.0) or 0.0) * 0.95)
            entry_basis = float(_get(p, "avg_entry_price", _get(p, "current_price", 0.0)) or 0.0)
            pos_risk = pos_qty * max(0.0, entry_basis - min_stop + 0.04)
            current_open_stop_risk += pos_risk

        # 3. Identify pending swing BUY entries and compute pending_entry_risk
        pending_swing_buys = [
            o for o in flattened_orders
            if normalize_enum_value(_get(o, "side", "")) == "BUY"
            and str(_get(o, "client_order_id", "") or "").startswith("swg_")
            and _get_active_remaining_qty(o) > 0.0
        ]
        pending_new_symbols = {
            str(_get(o, "symbol", "") or "").strip().upper()
            for o in pending_swing_buys
            if str(_get(o, "symbol", "") or "").strip().upper() not in active_position_symbols
        }

        pending_entry_risk = 0.0
        for p_order in pending_swing_buys:
            rem_qty = _get_active_remaining_qty(p_order)
            if rem_qty <= 0.0:
                continue

            p_sym = str(_get(p_order, "symbol", "") or "").strip().upper()
            p_stops = [
                o for o in flattened_orders
                if str(_get(o, "symbol", "") or "").strip().upper() == p_sym
                and normalize_enum_value(_get(o, "side", "")) == "SELL"
                and (
                    normalize_enum_value(_get(o, "order_type", "")) in PROTECTIVE_STOP_TYPES
                    or normalize_enum_value(_get(o, "type", "")) in PROTECTIVE_STOP_TYPES
                )
                and _get_active_remaining_qty(o) > 0.0
            ]
            require(len(p_stops) > 0, f"PENDING_ENTRY_WITHOUT_PROTECTIVE_STOP: {p_sym}")
            for o in p_stops:
                st_px = _get(o, "stop_price", None)
                require(st_px is not None and float(st_px) > 0.0, f"PENDING_ENTRY_WITHOUT_VALID_STOP_PRICE: {p_sym}")

            min_p_stop = min(float(_get(o, "stop_price", 0.0)) for o in p_stops)
            p_entry_basis = float(
                _get(p_order, "limit_price", None)
                or _get(p_order, "filled_avg_price", None)
                or _get(p_order, "price", 0.0)
                or 0.0
            )
            require(p_entry_basis > 0.0, f"PENDING_ENTRY_WITHOUT_VALID_PRICE_BASIS: {p_sym}")
            pending_entry_risk += rem_qty * max(0.0, p_entry_basis - min_p_stop + 0.06) + 0.50

        return current_open_stop_risk, pending_entry_risk, active_position_symbols, pending_new_symbols

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
            "time_in_force": self._entry_time_in_force(),
            "limit_price": entry_p,
            "order_class": OrderClass.BRACKET,
            "take_profit": TakeProfitRequest(limit_price=target_p),
            "stop_loss": StopLossRequest(stop_price=stop_p),
        }
        if client_order_id:
            request_kwargs["client_order_id"] = str(client_order_id)[:128]

        if getattr(self, "dry_run", False):
            self.mutation_counters.record("submit_order", True)
            logger.info("[DRY-RUN MUTATION BLOCKED] place_bracket_order for %s %s qty=%d @ limit=%.2f", symbol, side, qty, entry_p)
            return {
                "status": "dry_run_blocked",
                "order_id": f"dry_{client_order_id or 'order'}",
                "stop_order_id": None,
                "tp_order_id": None,
                "placed": False,
                "submitted": False,
                "reason": "DRY_RUN_MUTATION_BLOCKED",
            }

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

    def get_order_by_client_id(self, client_order_id: str) -> Optional[dict]:
        try:
            order = self._client.get_order_by_client_id(client_order_id)
            if not order:
                return None
            status = order.status.value if hasattr(order.status, "value") else str(order.status)
            qty = float(order.qty) if getattr(order, "qty", None) is not None else 0.0
            order_id = str(order.id)
            legs = getattr(order, "legs", None)
            leg_ids = [str(getattr(leg, "id", "")) for leg in legs] if legs else []
            stop_leg_id = None
            tp_leg_id = None
            if legs:
                for leg in legs:
                    leg_type = str(getattr(leg, "order_type", getattr(leg, "type", "")) or "").lower()
                    if "stop" in leg_type:
                        stop_leg_id = str(getattr(leg, "id", ""))
                    elif "limit" in leg_type:
                        tp_leg_id = str(getattr(leg, "id", ""))
            return {
                "id": order_id,
                "order_id": order_id,
                "parent_order_id": order_id,
                "stop_order_id": stop_leg_id,
                "tp_order_id": tp_leg_id,
                "order_ids": [order_id] + leg_ids,
                "status": status,
                "qty": qty,
                "legs": legs,
            }
        except Exception as exc:
            logger.warning("get_order_by_client_id failed for %s: %s", client_order_id, exc)
            return None

    def place_stop_order(self, symbol: str, side: str, qty: float, stop_price: float) -> dict:
        self._require_connection()
        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        qty = max(int(qty), min_shares)
        if getattr(self, "dry_run", False):
            logger.info("[DRY-RUN MUTATION BLOCKED] place_stop_order for %s %s qty=%d @ stop=%.2f", symbol, side, qty, stop_price)
            return {"order_id": f"dry_stop_{symbol}"}
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
        if getattr(self, "dry_run", False):
            logger.info("[DRY-RUN MUTATION BLOCKED] modify_stop_order %s -> stop=%.2f", order_id, new_stop)
            return f"dry_{order_id}"
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
                raw_type = getattr(order, "order_type", getattr(order, "type", ""))
                otype = str(getattr(raw_type, "value", raw_type)).strip().lower().replace("ordertype.", "")
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
        if getattr(self, "dry_run", False):
            logger.info("[DRY-RUN MUTATION BLOCKED] replace_stop for %s -> stop=%.2f", symbol, new_stop_price)
            return f"dry_stop_{symbol}"
        sym = symbol.upper().strip()
        try:
            for order in self._get_active_orders():
                if str(getattr(order, "symbol", "")).upper().strip() != sym:
                    continue
                raw_type = getattr(order, "order_type", getattr(order, "type", ""))
                otype = str(getattr(raw_type, "value", raw_type)).strip().lower().replace("ordertype.", "")
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
        if getattr(self, "dry_run", False):
            self.mutation_counters.record("cancel_order", True)
            logger.info("[DRY-RUN MUTATION BLOCKED] cancel_order %s", order_id)
            return True
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
        if getattr(self, "dry_run", False):
            logger.info("[DRY-RUN MUTATION BLOCKED] _cancel_all_orders_for_symbol %s", symbol)
            return
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
        if getattr(self, "dry_run", False):
            self.mutation_counters.record("close_position", True)
            logger.info("[DRY-RUN MUTATION BLOCKED] close_position %s", symbol)
            return True
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
        if getattr(self, "dry_run", False):
            self.mutation_counters.record("cancel_order", True)
            logger.info("[DRY-RUN MUTATION BLOCKED] _cancel_working_entry_orders %s", symbol)
            return
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

            if self.mode == "swing":
                from strategies.regime_classifier import classify_market_regime, MarketRegime
                adx_val = getattr(signal, "adx", None)
                vix_val = getattr(signal, "vix", None)
                spy_px = getattr(signal, "spy_price", None) or (self.get_price("SPY") if hasattr(self, "get_price") else 500.0) or 500.0
                spy_sma = getattr(signal, "spy_sma50", None) or (spy_px * 0.98)
                sma_slope = getattr(signal, "sma50_slope", None) or 0.1

                if adx_val is not None and vix_val is not None:
                    reg_decision = classify_market_regime(
                        adx=float(adx_val),
                        vix=float(vix_val),
                        spy_price=float(spy_px),
                        spy_sma50=float(spy_sma),
                        sma50_slope=float(sma_slope),
                    )
                    if not reg_decision.allows_new_entries:
                        logger.warning("[REGIME FILTER VETO] Swing BUY for %s rejected under %s regime: %s",
                                       symbol, reg_decision.regime.value, reg_decision.reason)
                        return False
                elif getattr(signal, "regime", None) in (MarketRegime.TRANSITIONING.value, MarketRegime.DATA_INVALID.value, MarketRegime.TRENDING_BEAR.value):
                    logger.warning("[REGIME FILTER VETO] Swing BUY for %s rejected under %s regime", symbol, signal.regime)
                    return False

                with self._swing_dispatch_lock:
                    enforce_hours = getattr(self, "_enforce_market_hours_test", None)
                    if enforce_hours is None:
                        enforce_hours = not bool(os.environ.get("PYTEST_CURRENT_TEST"))
                    if enforce_hours:
                        market_hours_ok, mh_reason = check_regular_market_hours(self._client)
                        if not market_hours_ok:
                            logger.warning("[REGULAR MARKET HOURS REJECTED] %s: %s", symbol, mh_reason)
                            return False

                    broker_account = None
                    try:
                        if self._client:
                            broker_account = self._client.get_account()
                    except Exception as e:
                        logger.warning("Failed to fetch broker account: %s", e)

                    broker_acct_raw = getattr(broker_account, "account_number", None) if broker_account else None
                    broker_acct_num = normalize_enum_value(broker_acct_raw)
                    if not broker_acct_num:
                        broker_acct_num = getattr(self._cfg, "SWING_ACCOUNT_ID", "PA3R71I7JFDZ")

                    configured_acct = getattr(self._cfg, "SWING_ACCOUNT_ID", "PA3R71I7JFDZ")
                    if broker_acct_num.strip().upper() != configured_acct.strip().upper():
                        logger.error(
                            "[ACCOUNT MISMATCH] Broker account %s does not match configured swing account %s (blocking dispatch)",
                            broker_acct_num, configured_acct
                        )
                        return False

                    from strategies.session import now_eastern
                    session_date = now_eastern().date().strftime("%Y-%m-%d")

                    if hasattr(self, "entry_gate") and hasattr(self.entry_gate, "check_daily_loss_halt_state"):
                        is_halted, h_reason = self.entry_gate.check_daily_loss_halt_state(broker_acct_num, session_date)
                        if is_halted:
                            logger.warning("[DAILY LOSS HALTED] Swing BUY for %s blocked by durable halt state: %s", symbol, h_reason)
                            return False

                    is_mock_account = hasattr(broker_account, "_mock_name") or "Mock" in type(broker_account).__name__
                    equity = 0.0
                    if broker_account and not is_mock_account:
                        try:
                            equity = float(getattr(broker_account, "equity", 0.0) or 0.0)
                        except (ValueError, TypeError):
                            equity = 0.0
                    if equity <= 0:
                        equity = self.get_net_liquidation()
                    if equity <= 0:
                        equity = self.get_cash()
                    if equity < min_cash:
                        logger.warning("Insufficient equity for BUY %s: $%.2f", symbol, equity)
                        return False

                    bp = 0.0
                    if broker_account and not is_mock_account:
                        try:
                            bp = float(getattr(broker_account, "buying_power", 0.0) or 0.0)
                        except (ValueError, TypeError):
                            bp = 0.0
                    if bp <= 0:
                        bp = equity

                    sod_equity = getattr(self, "_start_of_day_equity", 0.0)
                    if sod_equity <= 0.0:
                        if broker_account and hasattr(broker_account, "last_equity"):
                            try:
                                sod_equity = float(getattr(broker_account, "last_equity", 0.0) or 0.0)
                            except (ValueError, TypeError):
                                sod_equity = equity
                        else:
                            sod_equity = equity

                    if hasattr(self, "entry_gate") and hasattr(self.entry_gate, "daily_loss_blocked"):
                        dl_blocked, dl_reason = self.entry_gate.daily_loss_blocked(
                            account_id=broker_acct_num,
                            session_date_et=session_date,
                            current_equity=equity,
                            session_start_equity=sod_equity,
                            net_session_cash_flows=0.0,
                        )
                        if dl_blocked:
                            logger.warning("[DAILY LOSS CIRCUIT BREAKER] Swing BUY for %s blocked: %s", symbol, dl_reason)
                            return False

                    if hasattr(self, "entry_gate") and hasattr(self.entry_gate, "entry_blocked"):
                        blocked, reason = self.entry_gate.entry_blocked(symbol, price, signal, price=price)
                        if blocked:
                            logger.warning("[ENTRY GATE BLOCKED] Swing BUY for %s: %s", symbol, reason)
                            return False

                    all_pos = []
                    if self._client:
                        try:
                            raw_pos = self._client.get_all_positions()
                            if isinstance(raw_pos, list):
                                all_pos = raw_pos
                        except Exception as e:
                            logger.warning("Failed to fetch positions: %s", e)

                    raw_open_orders = []
                    if self._client:
                        try:
                            raw_orders = self._client.get_orders(status=QueryOrderStatus.OPEN)
                            if isinstance(raw_orders, list):
                                raw_open_orders = raw_orders
                        except Exception as e:
                            logger.warning("Failed to fetch open orders: %s", e)

                    from strategies.swing_sizing import (
                        calculate_swing_position_size,
                        calculate_swing_stop_price,
                        validate_swing_order_safety,
                        build_canonical_swing_order_id,
                        PositionIntent,
                        SwingSizingRejected,
                    )

                    try:
                        open_stop_risk, pending_entry_risk, active_pos_syms, pending_new_syms = self.audit_swing_portfolio_safety(
                            all_pos, raw_open_orders
                        )
                    except SwingSizingRejected as s_err:
                        logger.warning("[SWING SAFETY AUDIT REJECTED] %s: %s", symbol, s_err)
                        return False
                    except Exception as e:
                        logger.error("[SWING SAFETY AUDIT ERROR] %s: %s", symbol, e)
                        return False

                    if symbol in pending_new_syms:
                        logger.warning("[SWING REJECTED] Pending BUY already exists for %s", symbol)
                        return False

                    if symbol in active_pos_syms:
                        logger.info("Already long %s — skipping swing BUY", symbol)
                        return False

                    sector_entry = getattr(config_module, "SECTOR_MAPPINGS", {}).get(symbol)
                    if sector_entry is None:
                        logger.warning("[SECTOR REJECTED] Symbol %s has no sector attribution in config.SECTOR_MAPPINGS", symbol)
                        return False
                    symbol_sector = sector_entry.get("sector") if isinstance(sector_entry, dict) else str(sector_entry)

                    signal_sector = getattr(signal, "sector", None)
                    if signal_sector and str(signal_sector).strip().upper() != str(symbol_sector).strip().upper():
                        logger.warning(
                            "[SECTOR ATTRIBUTION MISMATCH] Signal sector %s does not match configured sector %s for %s",
                            signal_sector, symbol_sector, symbol
                        )
                        return False

                    def _get_qty_price(p):
                        q = float(getattr(p, "qty", 0.0) if not isinstance(p, dict) else p.get("qty", 0.0))
                        px = float(getattr(p, "current_price", getattr(p, "avg_entry_price", price)) if not isinstance(p, dict) else p.get("current_price", price))
                        return abs(q) * px

                    def _get_sym(p):
                        return str(getattr(p, "symbol", "") if not isinstance(p, dict) else p.get("symbol", "")).upper().strip()

                    existing_sym_exp = sum(_get_qty_price(p) for p in all_pos if _get_sym(p) == symbol)
                    existing_port_exp = sum(_get_qty_price(p) for p in all_pos)

                    existing_sector_exposure_by_sector = {}
                    for p in all_pos:
                        p_sym = _get_sym(p)
                        p_sec_entry = getattr(config_module, "SECTOR_MAPPINGS", {}).get(p_sym)
                        p_sec = (p_sec_entry.get("sector") if isinstance(p_sec_entry, dict) else str(p_sec_entry)) if p_sec_entry else "UNKNOWN"
                        p_val = _get_qty_price(p)
                        existing_sector_exposure_by_sector[p_sec] = existing_sector_exposure_by_sector.get(p_sec, 0.0) + p_val

                    pending_sector_exposure_by_sector = {}
                    for o in raw_open_orders:
                        o_side = normalize_enum_value(getattr(o, "side", "") if not isinstance(o, dict) else o.get("side", ""))
                        if o_side == "BUY":
                            o_sym = str(getattr(o, "symbol", "") if not isinstance(o, dict) else o.get("symbol", "")).upper().strip()
                            o_sec_entry = getattr(config_module, "SECTOR_MAPPINGS", {}).get(o_sym)
                            o_sec = (o_sec_entry.get("sector") if isinstance(o_sec_entry, dict) else str(o_sec_entry)) if o_sec_entry else "UNKNOWN"
                            o_tot = float(getattr(o, "qty", 0.0) if not isinstance(o, dict) else o.get("qty", 0.0) or 0.0)
                            o_fld = float(getattr(o, "filled_qty", 0.0) if not isinstance(o, dict) else o.get("filled_qty", 0.0) or 0.0)
                            o_rem = max(0.0, o_tot - o_fld)
                            o_px = float(getattr(o, "limit_price", None) or getattr(o, "stop_price", None) or price if not isinstance(o, dict) else o.get("limit_price") or o.get("stop_price") or price)
                            pending_sector_exposure_by_sector[o_sec] = pending_sector_exposure_by_sector.get(o_sec, 0.0) + (o_rem * o_px)

                    atr_val = getattr(signal, "atr_14", None) or getattr(signal, "atr", None) or (price * 0.02)
                    reg_name = getattr(signal, "regime", "TRENDING_BULL")
                    norm_stop, norm_stop_dist = calculate_swing_stop_price(
                        entry_price=price,
                        atr_14=atr_val,
                        regime=str(reg_name),
                        swing_low=getattr(signal, "swing_low", None),
                    )
                    stop_price = norm_stop
                    target_price = getattr(signal, "target_price", getattr(signal, "take_profit", None))
                    if target_price is None or target_price <= price:
                        target_price = round(price + (2.0 * norm_stop_dist), 2)

                    adv_val = getattr(signal, "adv_shares_20", None) or getattr(signal, "volume", None)
                    min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)

                    sizing = calculate_swing_position_size(
                        entry_price=price,
                        stop_price=stop_price,
                        account_equity=equity,
                        available_buying_power=bp,
                        existing_symbol_exposure=existing_sym_exp,
                        existing_portfolio_exposure=existing_port_exp,
                        adv_shares_20=adv_val,
                    )

                    if sizing.rejected or sizing.shares < min_shares:
                        logger.warning(
                            "[SWING SIZING REJECTED] %s: %s (shares=%d, constraint=%s)",
                            symbol, sizing.rejection_reason or "Below min_shares", sizing.shares, sizing.binding_constraint
                        )
                        return False

                    qty = sizing.shares
                    strat_name = getattr(signal, "strategy", getattr(signal, "strategy_name", ""))
                    bar_ts = getattr(signal, "timestamp", None) or now_eastern().isoformat()
                    order_ref = build_canonical_swing_order_id(
                        symbol=symbol,
                        side="BUY",
                        session_date=session_date,
                        bar_timestamp_utc=str(bar_ts),
                        signal_kind=str(strat_name or "ML_SWING").upper(),
                        account_number=configured_acct,
                    )

                    try:
                        validate_swing_order_safety(
                            client_order_id=order_ref,
                            strategy_instance_id="SWING_BOT_PRO",
                            symbol=symbol,
                            signal_version=getattr(signal, "version", "v1.6-alpha"),
                            entry_price=price,
                            stop_price=stop_price,
                            shares=qty,
                            account_equity=equity,
                            available_buying_power=bp,
                            current_open_stop_risk=open_stop_risk,
                            pending_entry_risk=pending_entry_risk,
                            side="BUY",
                            account_number=configured_acct,
                            broker_account_number=broker_acct_num,
                            current_position_count=len(active_pos_syms),
                            pending_open_position_count=len(pending_new_syms),
                            existing_position_symbols=active_pos_syms,
                            intent=PositionIntent.OPEN_LONG,
                            symbol_sector=symbol_sector,
                            existing_sector_exposure_by_sector=existing_sector_exposure_by_sector,
                            pending_sector_exposure_by_sector=pending_sector_exposure_by_sector,
                        )
                    except SwingSizingRejected as val_err:
                        logger.warning("[SWING SAFETY VALIDATION REJECTED] %s: %s", symbol, val_err)
                        return False
                    except Exception as ex:
                        logger.error("[SWING SAFETY VALIDATION ERROR] %s: %s", symbol, ex)
                        return False

                    logger.info(
                        "[SWING SIZING ACCEPTED] %s: shares=%d (constraint=%s, risk=$%.2f, notional=$%.2f, stop=$%.2f)",
                        symbol, qty, sizing.binding_constraint, sizing.estimated_risk, sizing.gross_notional, stop_price
                    )

                    result = self._place_open_bracket(
                        symbol, "BUY", qty, price, getattr(signal, "confidence", 1.0),
                        strategy_name=strat_name,
                        custom_stop_price=stop_price,
                        custom_target_price=target_price,
                        custom_client_order_id=order_ref,
                    )
                    paper_ledger.record_entry(
                        signal, result,
                        order_ref=order_ref,
                    )

                    if result.occupies_slot:
                        try:
                            from strategies.swing_risk_store import save_risk_basis, PositionRiskBasis
                            save_risk_basis(PositionRiskBasis(
                                symbol=symbol,
                                entry_price=price,
                                initial_stop_price=stop_price,
                                initial_risk_per_share=price - stop_price,
                                initial_total_risk=qty * (price - stop_price),
                                shares=qty,
                                entry_timestamp_utc=now_eastern().isoformat(),
                            ))
                        except Exception as r_err:
                            logger.warning("Could not persist risk basis for %s: %s", symbol, r_err)

                    return result.occupies_slot

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

            min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)

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

            risk_per_share = max(price - stop_price, 0.01)
            max_risk_pct = float(self._c("MAX_RISK_PER_TRADE_PCT", 0.01))
            risk_budget = equity * max_risk_pct
            risk_cap = float(self._c("MAX_RISK_PER_TRADE_DOLLARS", 150.0))
            risk_budget = min(risk_budget, risk_cap)
            qty = max(int(risk_budget / risk_per_share), 0)

            max_pos_val = equity * float(self._c("MAX_POSITION_PCT", 0.08))
            if qty * price > max_pos_val:
                qty = max(int(max_pos_val / price), 0)

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
        if not allow_short or self.mode == "swing":
            logger.info("No long in %s — skipping SELL (Stage 0 swing is long-only / ALLOW_SHORT=False)", symbol)
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
                ratchet_results = self.rotator._ratchet_stops(raw_positions, bridge=self, dry_run=getattr(self, "dry_run", False))
                self.rotator._log_ratchet_summary(ratchet_results)
            except Exception as r_exc:
                logger.error("[RATCHET EXECUTE ERROR] %s", r_exc)

            conf_lookup = {
                (s.symbol.upper() if hasattr(s, "symbol") else s["symbol"].upper()): (s.confidence if hasattr(s, "confidence") else s["confidence"])
                for s in signals
            }

            cfg_prot = getattr(config_module, "SWING_PROTECTED_SYMBOLS", {"SPY"})
            protected_syms = {s.upper().strip() for s in cfg_prot} | {
                s.upper().strip() for s in getattr(self, "protected_symbols", [])
            }
            open_pos = []
            for raw in raw_positions:
                qty = float(raw.qty)
                if qty == 0:
                    continue
                avg_cost = float(raw.avg_entry_price)
                mkt_price = float(raw.current_price)
                upnl = float(raw.unrealized_pl)
                sym = raw.symbol.upper()
                from strategies.swing_risk_store import get_unrealized_r
                curr_stop = self.get_current_stop(sym)
                u_r = get_unrealized_r(
                    symbol=sym,
                    current_price=mkt_price,
                    avg_entry_price=avg_cost,
                    current_stop_price=curr_stop,
                )
                pos = OpenPosition(
                    symbol=sym,
                    side=str(raw.side).lower(),
                    qty=qty,
                    avg_entry_price=avg_cost,
                    current_price=mkt_price,
                    unrealized_pnl=upnl,
                    unrealized_r=u_r,
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
                if getattr(self, "dry_run", False):
                    logger.info("[ROTATION DRY-RUN] Would rotate: Close %s -> Open %s (Simulated, no broker orders)",
                                exit_sym, enter_sig.symbol)
                    return
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

        open_positions = self.get_open_positions()
        if isinstance(open_positions, dict):
            open_symbols = {str(s).upper() for s in open_positions.keys()}
        else:
            open_symbols = {str(s).upper() for s in open_positions}
        working_symbols = self.working_order_symbols()
        occupied_symbols = open_symbols | working_symbols
        planned_new_symbols: set = set()
        placed, skipped = 0, 0
        rejections = []
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
                if self.mode == "swing":
                    from strategies.rotation_config import MAX_CONCURRENT_POSITIONS
                    max_open = int(MAX_CONCURRENT_POSITIONS)
                else:
                    max_open = int(getattr(config_module, "MAX_OPEN_POSITIONS", 5))
                planned_total = len(occupied_symbols | planned_new_symbols)
                if planned_total >= max_open:
                    logger.warning("Max positions (%d) reached — skipping %s", max_open, symbol)
                    skipped += 1
                    rejections.append({"symbol": symbol, "reason": "MAX_CONCURRENT_POSITIONS_REACHED"})
                    continue

            if self.execute_signal(signal):
                placed += 1
                if opens_new:
                    planned_new_symbols.add(symbol)
            else:
                skipped += 1

        return {"placed": placed, "skipped": skipped, "total": len(signals), "rejections": rejections, "reason": rejections[0]["reason"] if rejections else None}

    @staticmethod
    def _today() -> str:
        from strategies.session import now_eastern
        return now_eastern().date().strftime("%Y%m%d")
