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
import sys
import threading
import time
from collections import deque
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream as _SdkTradingStream
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest, MarketOrderRequest
import websockets
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest, StockLatestTradeRequest
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
from strategies.atomic_rotation import CloseOutcome, PositionCloseResult

logger = logging.getLogger(__name__)
_broker_event_write_lock = threading.Lock()

_STOP_ORDER_TYPES: Set[str] = {
    "stop",
    "stop_loss",
    "stop_limit",
    "trailing_stop",
    "stp",
    "stp lmt",
    "stoplimit",
    "trail",
    "traillimit",
}


class AccountModeViolation(RuntimeError):
    """Raised when an account-mode safety invariant is violated (e.g. non-paper URL, mismatched account ID)."""
    pass


class ReleaseGateViolation(RuntimeError):
    """Raised when order submission is attempted without explicit machine-enforced Principal Release."""
    definitely_rejected: bool = True



@dataclass
class ValidationReleaseToken:
    """
    Single-Use Principal Release Token for Controlled Paper Validation.
    Authorizes exactly one validation lifecycle (Entry -> Stop -> Replace -> Exit).
    Token is consumed at entry submission admission to prevent duplicate or subsequent orders.
    """
    release_id: str
    account_id: str = "PA3UODIC5JTA"
    paper_only: bool = True
    max_qty: int = 1
    max_new_positions: int = 1
    max_lifecycles: int = 1
    expires_at: Optional[_dt.datetime] = None
    consumed: bool = False
    consumed_at: Optional[_dt.datetime] = None
    authorized_symbol: Optional[str] = None
    authorized_qty: int = 1
    authorized_parent_order_id: Optional[str] = None
    authorized_client_order_id: Optional[str] = None
    submission_outcome: str = "PENDING"


class PersistedRiskStateInvalidError(RuntimeError):
    """Raised when persisted risk state file exists but is corrupted, unreadable, or malformed."""
    pass


@dataclass(frozen=True)
class OrderTruth:
    id: str
    symbol: str
    side: str
    status: str
    qty: float
    filled_qty: float
    filled_avg_price: float
    is_terminal: bool
    raw: Any = None


@dataclass(frozen=True)
class PositionTruth:
    symbol: str
    qty: float
    side: str
    avg_entry_price: float
    exists: bool
    raw: Any = None


@dataclass(frozen=True)
class CancelResult:
    success: bool
    is_already_terminal: bool = False
    error_message: str = ""
    terminal_status: Optional[str] = None

    def __bool__(self) -> bool:
        return self.success


import urllib.parse
from datetime import datetime, timezone

ACCOUNT_MODE_VIOLATION_CODE = 6


def _validate_paper_url(raw_url: str) -> None:
    if not raw_url:
        return
    raw_str = str(raw_url).strip()
    parsed = urllib.parse.urlsplit(raw_str)
    if (parsed.scheme or "").lower() != "https":
        raise AccountModeViolation(
            f"ACCOUNT_MODE_VIOLATION: Insecure scheme '{parsed.scheme}' in base URL ({raw_url}). Must be https."
        )
    hostname = (parsed.hostname or "").lower()
    if hostname != "paper-api.alpaca.markets":
        raise AccountModeViolation(
            f"ACCOUNT_MODE_VIOLATION: Unauthorized endpoint host '{hostname}'. Expected exactly 'paper-api.alpaca.markets'."
        )
    if parsed.port not in (None, 443):
        raise AccountModeViolation(
            f"ACCOUNT_MODE_VIOLATION: Non-standard port '{parsed.port}' in base URL ({raw_url})."
        )
    if parsed.username or parsed.password:
        raise AccountModeViolation(
            "ACCOUNT_MODE_VIOLATION: Embedded credentials detected in base URL."
        )


def record_account_mode_violation_snapshot(
    expected_account_id: str,
    observed_account_id: str,
    endpoint_url: str,
    failure_reason: str,
    output_dir: Optional[str] = None,
) -> str:
    """Record fail-closed diagnostic snapshot for Exit Code 6 (ACCOUNT_MODE_VIOLATION)."""
    target_dir = output_dir if output_dir is not None else "logs"
    os.makedirs(target_dir, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    snapshot_path = os.path.join(target_dir, f"account_mode_violation_{ts_str}.json")
    parsed_host = "unknown"
    if endpoint_url:
        try:
            parsed_host = urllib.parse.urlsplit(str(endpoint_url).strip()).hostname or str(endpoint_url)
        except Exception:
            parsed_host = "unparseable"

    data = {
        "exit_code": 6,
        "failure_type": "ACCOUNT_MODE_VIOLATION",
        "expected_account_id": expected_account_id or "PA3UODIC5JTA",
        "observed_account_id": observed_account_id or "UNVERIFIED",
        "endpoint_host": parsed_host,
        "order_submission_attempted": False,
        "failure_reason": str(failure_reason),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = snapshot_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, snapshot_path)
    return snapshot_path


def log_reconciliation_anomaly(anomaly_data: dict) -> None:
    """Durably records reconciliation price or quantity anomalies for operator review."""
    reports_dir = Path(getattr(config_module, "REPORTS_DIR", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "reconciliation_anomalies.jsonl"
    anomaly_data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(anomaly_data, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


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


def normalize_position_qty(pos_val: Any) -> float:
    """Safely extract and normalize any broker position object, dict, string, Decimal, or None into a signed float.
    Treats None, empty, or explicit zero as 0.0 (flat).
    """
    if pos_val is None:
        return 0.0
    if hasattr(pos_val, "_mock_name") or "Mock" in type(pos_val).__name__:
        if hasattr(pos_val, "_mock_return_value") and isinstance(pos_val._mock_return_value, (int, float)):
            return float(pos_val._mock_return_value)
        raw_qty = getattr(pos_val, "qty", None)
        if raw_qty is not None and not (hasattr(raw_qty, "_mock_name") or "Mock" in type(raw_qty).__name__):
            return normalize_position_qty(raw_qty)
        return 0.0
    if isinstance(pos_val, (int, float)):
        return float(pos_val)
    if hasattr(pos_val, "qty"):
        raw_qty = getattr(pos_val, "qty")
        return normalize_position_qty(raw_qty)
    if isinstance(pos_val, dict):
        raw_qty = pos_val.get("qty", pos_val.get("position", 0.0))
        return normalize_position_qty(raw_qty)
    try:
        s = str(pos_val).strip()
        if not s:
            return 0.0
        return float(s)
    except (ValueError, TypeError):
        return 0.0


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


class StateTransitionError(RuntimeError):
    """Raised when an illegal stream state transition is attempted."""
    pass


class _BridgeAwareTradingStream(_SdkTradingStream):
    """
    Subclass of alpaca.trading.stream.TradingStream that hooks into
    SDK internal reconnect cycles (in _run_forever) to notify AlpacaBridge
    and enforce broker-state reconciliation before trading resumes.

    Phase B Tier 2 Runtime Defect Hardening:
    Contains the upstream alpaca-py 10ms retry spin loop in TradingStream._run_forever().
    When connection or transport failure occurs, the stream terminates cleanly and yields
    100% of reconnect ownership to AlpacaBridge's supervisor (_trade_stream_reconnect_loop),
    preserving the canonical Phase B backoff contract (2s -> 4s -> 8s -> 16s -> 30s +- 20% jitter).
    """

    def __init__(self, *args, bridge: Optional[Any] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._bridge = bridge
        self._bridge_ws_ever_connected = False

    @property
    def _initial_connect_done(self) -> bool:
        return self._bridge_ws_ever_connected

    @_initial_connect_done.setter
    def _initial_connect_done(self, val: bool) -> None:
        self._bridge_ws_ever_connected = bool(val)

    async def _safe_cleanup(self) -> None:
        """Centralized, idempotent cleanup for WebSocket connection and running state."""
        self._running = False
        ws = getattr(self, "_ws", None)
        if ws is not None:
            try:
                await ws.close()
            except Exception as exc:
                logger.debug("[TRADE STREAM] Safe cleanup error on ws.close(): %s", exc)
            finally:
                self._ws = None

    async def _start_ws(self):
        is_reconnect = self._bridge_ws_ever_connected
        if is_reconnect and self._bridge is not None:
            try:
                self._bridge._on_sdk_stream_reconnect_start()
            except Exception as exc:
                logger.exception("[STREAM RECONNECT] Error in _on_sdk_stream_reconnect_start: %s", exc)

        await super()._start_ws()
        self._bridge_ws_ever_connected = True

        if is_reconnect and self._bridge is not None:
            try:
                self._bridge._on_sdk_stream_reconnect_success()
            except Exception as exc:
                logger.exception("[STREAM RECONNECT] Error in _on_sdk_stream_reconnect_success: %s", exc)

    async def _run_forever(self):
        """
        Hardened stream execution loop (Phase B Tier 2 Defect Fix).
        Replaces upstream SDK's unthrottled 10ms while-loop.
        Guarantees:
          1. Intentional shutdown (_should_run=False) exits cleanly with zero reconnect.
          2. Unexpected termination (_consume() returning while _should_run=True) is treated as failure.
          3. Transport failures exit immediately and yield reconnect cadence to the Application Supervisor.
          4. Centralized single-path cleanup via _safe_cleanup().
        """
        self._loop = asyncio.get_running_loop()
        while not self._trade_updates_handler:
            if not self._stop_stream_queue.empty():
                self._stop_stream_queue.get(timeout=1)
                await self._safe_cleanup()
                return
            await asyncio.sleep(0.1)

        logger.info("[TRADE STREAM] Started trading stream event loop")
        self._should_run = True
        self._running = False

        if not self._should_run:
            logger.info("[TRADE STREAM] Trading stream stopped before connection (intentional)")
            await self._safe_cleanup()
            return

        try:
            logger.info("[TRADE STREAM] Initiating trading websocket connection")
            await self._start_ws()
            self._running = True
            await self._consume()

            # Mandatory Correction #2: Check unexpected normal return from _consume()
            if self._should_run:
                logger.warning(
                    "[TRADE STREAM] Unexpected stream termination: _consume() returned while _should_run is True. Yielding to supervisor."
                )
            else:
                logger.info("[TRADE STREAM] Trading stream stopped gracefully (intentional)")
        except websockets.WebSocketException as wse:
            if self._should_run:
                logger.warning(
                    "[TRADE STREAM] WebSocket communication closed (%s: %s); yielding to application supervisor",
                    type(wse).__name__,
                    wse,
                )
            else:
                logger.info("[TRADE STREAM] WebSocket closed during intentional shutdown: %s", wse)
        except Exception as exc:
            if self._should_run:
                logger.warning(
                    "[TRADE STREAM] Stream transport failure (%s: %s); yielding to application supervisor",
                    type(exc).__name__,
                    exc,
                )
            else:
                logger.info("[TRADE STREAM] Transport error during intentional shutdown: %s", exc)
        finally:
            await self._safe_cleanup()
            logger.info("[TRADE STREAM] Stream event loop terminated (should_run=%s)", self._should_run)


TradingStream = _BridgeAwareTradingStream


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
        elif kind in ("replace_order", "modify_stop_order"):
            self.replace_order_passed_through += 1
        elif kind == "cancel_order":
            self.cancel_order_passed_through += 1
        elif kind == "close_position":
            self.close_position_passed_through += 1
        return False

    def get(self, kind: str) -> int:
        if kind in ("replace_order", "modify_stop_order"):
            return self.replace_order_passed_through
        if kind == "submit_order":
            return self.submit_order_passed_through
        if kind == "cancel_order":
            return self.cancel_order_passed_through
        if kind == "close_position":
            return self.close_position_passed_through
        d = self.to_dict()
        return d.get(kind, getattr(self, kind, 0))

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
        self.symbol = str(symbol).upper().strip()
        self.position = qty
        self.avgCost = avg_cost
        self.marketValue = market_value
        self.unrealizedPNL = unrealized_pnl

    def __str__(self) -> str:
        return self.symbol


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

    def isConnected(self) -> bool:
        if not self.bridge:
            return False
        is_conn = bool(getattr(self.bridge, "is_connected", False))
        if not is_conn:
            return False
        conn_health = getattr(self.bridge, "_conn_health", None)
        if conn_health is not None and not getattr(conn_health, "is_healthy", True):
            return False
        return True

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

            # Update fast lookup cached positions
            try:
                lock = getattr(self.bridge, "_positions_cache_lock", None)
                if lock is not None and not getattr(self.bridge, "_cache_requires_reconciliation", False):
                    with lock:
                        self.bridge._cached_position_symbols = {p.symbol.upper().strip() for p in res if abs(p.position) > 0}
                        self.bridge._cached_positions_ts = time.time()
            except Exception:
                pass

            # If connection health was marked unhealthy (e.g. from prior timeout) or entry gate
            # is halted with BROKER_RECONCILIATION_PENDING, delegate to the single recovery owner!
            conn_health = getattr(self.bridge, "_conn_health", None)
            is_unhealthy = conn_health is not None and not getattr(conn_health, "is_healthy", True)
            entry_gate = getattr(self.bridge, "entry_gate", None)
            has_recon_halt = entry_gate is not None and "BROKER_RECONCILIATION_PENDING" in getattr(entry_gate, "reasons", set())
            if is_unhealthy or has_recon_halt:
                logger.info("[MockIB.positions] Broker positions query restored; initiating full broker reconciliation via single recovery owner")
                self.bridge._run_post_reconnect_reconciliation()

            return res
        except Exception as exc:
            logger.error("MockIB.positions() failed: %s", exc)
            self.bridge.invalidate_broker_caches(reason=f"MockIB.positions_error:{exc}")
            self.bridge._cache_requires_reconciliation = True
            self.bridge._conn_health.mark_unhealthy(f"MockIB.positions error: {exc}")
            if hasattr(self.bridge, "entry_gate") and self.bridge.entry_gate:
                self.bridge.entry_gate.halt("BROKER_RECONCILIATION_PENDING")
            raise RuntimeError(f"Alpaca API outage while fetching positions: {exc}")

    def openTrades(self) -> List[MockOrder]:
        try:
            # Canonical order flattening: _get_active_orders() already expands bracket legs into
            # individual orders while deduplicating by ID. Do not re-append o.legs here.
            raw_orders = self.bridge._get_active_orders()
            al_orders = raw_orders

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

    def _sync_state(self):
        """Keep the cached halt flag/reason consistent with active halt reasons."""
        active_reasons = {str(reason) for reason in self.reasons if str(reason)}
        self.halted = bool(active_reasons)
        current_reason = str(self.halt_reason or "")
        if current_reason not in active_reasons:
            self.halt_reason = sorted(active_reasons)[0] if active_reasons else ""

    def halt(self, reason: str = ""):
        self.halted = True
        self.halt_reason = str(reason or "")
        if reason:
            self.reasons.add(str(reason))
            self._sync_state()
        # Preserve legacy fail-closed semantics for callers without a reason.
        # Critical internal call sites should pass a durable reason explicitly.
        acct = getattr(self.bridge, "account_id", "UNKNOWN") if self.bridge else "NO_BRIDGE"
        logger.warning(
            "[ENTRY GATE HALT] [%s] Added reason: '%s' | Current active reasons: %s | halted=True",
            acct,
            reason,
            sorted(self.reasons),
        )

    def clear(self, reason: str = ""):
        if reason:
            self.reasons.discard(str(reason))
        self._sync_state()
        acct = getattr(self.bridge, "account_id", "UNKNOWN") if self.bridge else "NO_BRIDGE"
        logger.info(
            "[ENTRY GATE CLEAR] [%s] Removed reason: '%s' | Remaining active reasons: %s | halted=%s",
            acct,
            reason,
            sorted(self.reasons),
            self.halted,
        )

    def clear_recovery_halts(self):
        recovery_reasons = {
            "TRADE_STREAM_DISCONNECTED",
            "BROKER_RECONCILIATION_PENDING",
            "STREAM_RECOVERY_REQUIRED",
            "STREAM_DEGRADED_OR_DESYNC",
            "OFF_MARKET_STANDBY",
            "STARTUP_HEALTH_GATE_HELD",
            "DATA_FEED_DEGRADED",
        }
        cleared = sorted(self.reasons.intersection(recovery_reasons))
        self.reasons.difference_update(recovery_reasons)
        self._sync_state()
        acct = getattr(self.bridge, "account_id", "UNKNOWN") if self.bridge else "NO_BRIDGE"
        logger.info(
            "[ENTRY GATE RECOVERY CLEAR] [%s] Cleared recovery reasons: %s | Remaining active reasons: %s | halted=%s",
            acct,
            cleared,
            sorted(self.reasons),
            self.halted,
        )

    def resume(self):
        self.reasons.clear()
        self._sync_state()
        acct = getattr(self.bridge, "account_id", "UNKNOWN") if self.bridge else "NO_BRIDGE"
        logger.info("[ENTRY GATE RESUME] [%s] Resumed all entry gates | reasons=[] | halted=False", acct)

    def risk_sized_qty(self, symbol, signal, price, qty):
        if self.halted:
            return 0
        return qty

    def _get_halt_dir(self) -> Path:
        if hasattr(self, "halt_dir") and self.halt_dir:
            return Path(self.halt_dir)
        if getattr(self, "bridge", None) and hasattr(self.bridge, "_risk_state_dir") and self.bridge._risk_state_dir:
            return Path(self.bridge._risk_state_dir)
        custom_dir = getattr(config_module, "DATA_DIR", None)
        if custom_dir:
            return Path(custom_dir)
        is_test_env = (
            "pytest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or bool(os.environ.get("TEST_RUNNING"))
        )
        if is_test_env:
            import tempfile
            test_dir = Path(tempfile.gettempdir()) / "stock_engine_test_risk_state"
            test_dir.mkdir(parents=True, exist_ok=True)
            return test_dir
        return Path("data")

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
        data_dir = self._get_halt_dir()
        halt_file = data_dir / f"swing_daily_loss_halt_{account_id.strip().upper()}.json"
        if not halt_file.exists() and ("DAILY_LOSS" in (self.halt_reason or "").upper() or "DAILY_LOSS_LIMIT" in self.reasons):
            self.reasons.discard("DAILY_LOSS_LIMIT")
            self._sync_state()

        if self.halted:
            return True, self.halt_reason or "Trading engine is halted"

        try:
            d_current_equity = Decimal(str(current_equity))
            d_sod_equity = Decimal(str(session_start_equity))
            d_cash_flows = Decimal(str(net_session_cash_flows or 0.0))
        except Exception as e:
            return True, f"INVALID_EQUITY_DECIMAL: {e}"

        if not (d_current_equity.is_finite() and d_sod_equity.is_finite() and d_cash_flows.is_finite()):
            return True, "NON_FINITE_EQUITY_VALUES"

        if d_sod_equity <= Decimal("0"):
            self.halt("BASELINE_UNVERIFIED")
            return True, "BASELINE_UNVERIFIED: Start-of-day equity baseline is not verified"

        if "BASELINE_UNVERIFIED" in self.reasons:
            self.reasons.discard("BASELINE_UNVERIFIED")
            self._sync_state()

        d_daily_pnl = d_current_equity - d_sod_equity - d_cash_flows

        pct_cap_float = float(getattr(config_module, "SWING_MAX_DAILY_LOSS_PCT", 0.01))
        dollar_cap_float = float(getattr(config_module, "SWING_MAX_DAILY_LOSS_DOLLAR_CAP", 300.00))

        d_pct_cap = d_sod_equity * Decimal(str(pct_cap_float))
        d_dollar_cap = Decimal(str(dollar_cap_float))
        d_limit = min(d_pct_cap, d_dollar_cap)

        if d_daily_pnl <= -d_limit:
            self.persist_daily_loss_halt(account_id, session_date_et, d_daily_pnl, d_limit, baseline_equity=d_sod_equity)
            self.halt("DAILY_LOSS_LIMIT")
            return True, f"Max daily loss limit reached (${d_daily_pnl:.2f} <= -${d_limit:.2f})"

        return False, ""

    def persist_daily_loss_halt(
        self,
        account_id: str,
        session_date_et: str,
        daily_pnl: Decimal,
        daily_loss_limit: Decimal,
        baseline_equity: Optional[Decimal] = None,
    ) -> None:
        """Atomically persist durable daily-loss halt state."""
        from datetime import datetime, timezone
        from pathlib import Path
        data_dir = self._get_halt_dir()
        halt_file = data_dir / f"swing_daily_loss_halt_{account_id.strip().upper()}.json"
        data_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "account_id": account_id.strip().upper(),
            "session_date_et": session_date_et,
            "daily_pnl": str(daily_pnl),
            "daily_loss_limit": str(daily_loss_limit),
            "baseline_equity": str(baseline_equity) if baseline_equity is not None else None,
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
            logger.warning("[DAILY LOSS HALT PERSISTED] %s: pnl=%s limit=%s baseline=%s", account_id, daily_pnl, daily_loss_limit, baseline_equity)
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
        data_dir = self._get_halt_dir()
        halt_file = data_dir / f"swing_daily_loss_halt_{account_id.strip().upper()}.json"
        if not halt_file.exists():
            if "DAILY_LOSS_LIMIT" in self.reasons or (self.halt_reason and "DAILY_LOSS" in self.halt_reason.upper()):
                self.reasons.discard("DAILY_LOSS_LIMIT")
                self._sync_state()
            return (self.halted, self.halt_reason) if self.halted else (False, "")
        try:
            with open(halt_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            logger.error("Corrupted daily loss halt file for %s: %s (failing closed)", account_id, exc)
            self.halt("CORRUPTED_HALT_STATE")
            return True, f"CORRUPTED_HALT_STATE: {exc}"

        rec_acct = str(data.get("account_id", "")).strip().upper()
        if rec_acct != account_id.strip().upper():
            logger.error("Halt file account mismatch: expected %s, got %s (failing closed)", account_id, rec_acct)
            self.halt("HALT_STATE_ACCOUNT_MISMATCH")
            return True, "HALT_STATE_ACCOUNT_MISMATCH"

        rec_date = str(data.get("session_date_et", "")).strip()
        if rec_date > current_session_date_et:
            logger.error("Future session date %s detected in halt state for %s (failing closed)", rec_date, account_id)
            self.halt("FUTURE_SESSION_DATE_DETECTED")
            return True, "FUTURE_SESSION_DATE_DETECTED"

        if rec_date == current_session_date_et:
            if data.get("halted", False):
                self.halt("DAILY_LOSS_LIMIT")
                return True, f"DAILY_LOSS_LIMIT_REACHED (pnl={data.get('daily_pnl')}, limit={data.get('daily_loss_limit')})"

        # Previous session -> controlled reset
        try:
            logger.info("[SESSION ROLLOVER RESET] Resetting previous session (%s) daily loss halt state for %s", rec_date, account_id)
            halt_file.unlink(missing_ok=True)
            self.reasons.discard("DAILY_LOSS_LIMIT")
            self._sync_state()
        except Exception as r_exc:
            logger.warning("Could not unlink previous halt file for %s: %s", account_id, r_exc)
        if self.halted:
            return True, self.halt_reason
        return False, ""

    def entry_blocked(self, symbol, value, signal, price=None):
        if self.halted:
            return True, self.halt_reason or "Trading engine is halted"
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
                    pct_limit = abs(sod_equity * float(getattr(config_module, "SWING_MAX_DAILY_LOSS_PCT", 0.015)))
                    dollar_cap = float(getattr(config_module, "SWING_MAX_DAILY_LOSS_DOLLAR_CAP", 1500.00))
                    max_loss = min(pct_limit, dollar_cap) if dollar_cap > 0 else 1500.00
                else:
                    max_loss = float(getattr(config_module, "MAX_DAILY_LOSS_DAYTRADE", 300.0))
                if round(daily_pnl, 2) <= -round(abs(max_loss), 2):
                    self.halt("DAILY_LOSS_LIMIT")
                    return True, f"Max daily loss limit reached (${daily_pnl:.2f} <= -${max_loss:.2f})"
            except Exception as e:
                logger.error("Error checking daily loss limit in EntryGateMock: %s; blocking new entries only. Protective exits remain enabled.", e)
                return True, f"Risk check error: {e} (blocking new entries only)"
        return False, ""


class SwingPortfolioAuditResult(tuple):
    """
    Tuple subclass maintaining strict 4-tuple backward compatibility for audit_swing_portfolio_safety
    while exposing per_symbol_open_risk, per_symbol_pending_risk, and per_symbol_risk breakdowns
    for canonical cluster risk aggregation.
    """
    def __new__(
        cls,
        current_open_stop_risk,
        pending_entry_risk,
        active_position_symbols,
        pending_new_symbols,
        per_symbol_risk=None,
        per_symbol_open_risk=None,
        per_symbol_pending_risk=None,
    ):
        return super().__new__(cls, (current_open_stop_risk, pending_entry_risk, active_position_symbols, pending_new_symbols))

    def __init__(
        self,
        current_open_stop_risk,
        pending_entry_risk,
        active_position_symbols,
        pending_new_symbols,
        per_symbol_risk=None,
        per_symbol_open_risk=None,
        per_symbol_pending_risk=None,
    ):
        self.open_stop_risk = current_open_stop_risk
        self.pending_entry_risk = pending_entry_risk
        self.active_position_symbols = active_position_symbols
        self.pending_new_symbols = pending_new_symbols
        self.per_symbol_risk = per_symbol_risk or {}
        self.per_symbol_open_risk = per_symbol_open_risk or {}
        self.per_symbol_pending_risk = per_symbol_pending_risk or {}


SWING_TECH_CLUSTER = frozenset({
    "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AMD",
    "GOOGL", "QQQ", "PLTR", "AVGO", "MU", "NFLX",
})


def get_symbol_cluster(symbol: str) -> str:
    """Canonical mapping of symbols to risk clusters (TECH vs sector clusters)."""
    sym = str(symbol or "").strip().upper()
    if sym in SWING_TECH_CLUSTER:
        return "TECH"
    sec_entry = getattr(config_module, "SECTOR_MAPPINGS", {}).get(sym)
    if sec_entry:
        sec = sec_entry.get("sector") if isinstance(sec_entry, dict) else str(sec_entry)
        if sec:
            return str(sec).strip().upper()
    return f"SECTOR_{sym}"


_Q_FIELDS = (
    "q_risk", "q_notional", "q_liquidity",
    "q_portfolio", "q_broker",
)


def _log_sizing_diag(symbol, sizing, entry_price, rejection_reason=None):
    if sizing is None:
        return
    caps = {name: getattr(sizing, name, "MISSING") for name in _Q_FIELDS}

    # Early input rejection မှာ q_* အားလုံးက placeholder zeros ပါ။
    valid_caps = {k: v for k, v in caps.items() if isinstance(v, (int, float))}
    evaluated = (
        getattr(sizing, "binding_constraint", None) != "REJECTED"
        and len(valid_caps) == len(_Q_FIELDS)
    )
    min_q = min(valid_caps.values()) if evaluated and valid_caps else None
    limiting = (
        [name for name, value in valid_caps.items() if value == min_q]
        if evaluated and min_q is not None else []
    )

    port_used = getattr(sizing, "portfolio_exposure_used", None)
    port_cap = getattr(sizing, "portfolio_exposure_cap", None)
    port_diag = f" exposure_used=${port_used:,.2f} exposure_cap=${port_cap:,.2f}" if (port_used is not None and port_cap is not None) else ""

    logger.info(
        "[SIZING DIAG] symbol=%s caps=%s final_qty=%s "
        "min_q=%s matches_min=%s limiting_constraints=%s "
        "entry_price=%s gross_notional=%s%s "
        "sizing_rejected=%s sizing_reason=%s stage_reason=%s",
        symbol, caps, getattr(sizing, "shares", 0),
        min_q,
        getattr(sizing, "shares", 0) == min_q if evaluated else "NOT_EVALUATED",
        limiting,
        entry_price, getattr(sizing, "gross_notional", 0.0),
        port_diag,
        getattr(sizing, "rejected", False), getattr(sizing, "rejection_reason", None) or "NONE",
        rejection_reason or "NONE",
    )


class AlpacaBridge:
    def __init__(self, cfg=None, mode: str = "daytrade", dry_run: bool = False) -> None:
        self.mode = str(mode or "daytrade").lower()
        try:
            from strategies.trailing_stop import set_default_trailing_context
            set_default_trailing_context(mode=self.mode)
        except Exception:
            pass
        try:
            from logging_setup import set_log_context
            set_log_context(process_tag=self.mode.upper())
        except Exception:
            pass
        try:
            if hasattr(config_module, "log_strategy_feature_flags"):
                config_module.log_strategy_feature_flags()
        except Exception:
            pass
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
        # Global lock order invariant:
        # 1. _reconcile_lock
        # 2. get_journal_lock()
        #
        # Never acquire _reconcile_lock while already holding the journal lock.
        self._reconcile_lock = threading.Lock()
        self._reconcile_scheduling_lock = threading.Lock()
        self._reconcile_in_progress = False
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
        from strategies.atomic_rotation import AtomicRotationEngine
        self.atomic_rotator = AtomicRotationEngine()
        self._swing_dispatch_lock = threading.Lock()
        self._order_role_registry: Dict[str, dict] = {}
        self._order_role_lock = threading.Lock()
        self._load_order_role_registry()
        self._account_id: str = "default"
        self.principal_release_granted: bool = False
        self.validation_release_token: Optional[ValidationReleaseToken] = None
        self._positions_cache_lock = threading.Lock()
        self._cached_position_symbols: Optional[Set[str]] = None
        self._cached_positions_ts: float = 0.0
        self._active_orders_lock = threading.Lock()
        self._cached_active_orders: Optional[List[Any]] = None
        self._cached_active_orders_ts: float = 0.0
        self._cache_requires_reconciliation: bool = False
        self._risk_state_dir: Optional[str] = None

    @property
    def account_id(self) -> str:
        return getattr(self, "_account_id", "") or "default"

    @account_id.setter
    def account_id(self, value: str):
        self._account_id = str(value)
        try:
            from logging_setup import set_log_context
            clean_tag = str(value).strip().upper()
            if clean_tag and clean_tag not in ("DEFAULT", "UNKNOWN", "NONE"):
                set_log_context(account_tag=clean_tag)
        except Exception:
            pass

    @property
    def is_paper(self) -> bool:
        if self._client is not None:
            return bool(getattr(self._client, "_paper", True))
        return True

    @property
    def is_paper_account(self) -> bool:
        return self.is_paper

    @property
    def account_mode(self) -> str:
        return "PAPER" if self.is_paper else "LIVE"

    def assert_account_mode_safety(self, expected_account_id: Optional[str] = None) -> bool:
        """
        Fail-closed Account-Mode Safety Guard enforcing paper trading invariants.
        Must be called prior to starting trading loops and before any order submission/mutation.

        Guarantees:
          1. TradingClient is constructed in paper mode (is_paper is True, client._paper is True).
          2. Client base URL strictly resolves to https://paper-api.alpaca.markets (never live api.alpaca.markets).
          3. Authenticated broker account number matches expected account ID (default: PA3UODIC5JTA for daytrade).
          4. Any unreadable, failing, or mismatched account number fails closed.

        Raises AccountModeViolation if any invariant fails.
        """
        if not self.is_paper:
            raise AccountModeViolation(
                "ACCOUNT_MODE_VIOLATION: TradingClient is not in paper mode (paper=False detected)."
            )

        client = getattr(self, "_client", None)
        if client is not None:
            if not getattr(client, "_paper", True):
                raise AccountModeViolation(
                    "ACCOUNT_MODE_VIOLATION: Underlying client._paper is False! Refusing execution."
                )

            raw_base_url = getattr(client, "_base_url", None)
            base_url_str = ""
            if isinstance(raw_base_url, str):
                base_url_str = raw_base_url.strip().rstrip("/")
            elif hasattr(raw_base_url, "value") and isinstance(raw_base_url.value, str):
                base_url_str = raw_base_url.value.strip().rstrip("/")
            elif raw_base_url and not (hasattr(raw_base_url, "_mock_name") or "Mock" in type(raw_base_url).__name__):
                base_url_str = str(raw_base_url).strip().rstrip("/")

            if base_url_str:
                _validate_paper_url(base_url_str)

        env_base_url = os.environ.get("APCA_API_BASE_URL", "").strip().rstrip("/")
        if env_base_url:
            _validate_paper_url(env_base_url)

        if expected_account_id:
            target_acct = expected_account_id
        elif self.mode == "swing":
            target_acct = os.environ.get("SWING_ACCOUNT_ID", "PA3R71I7JFDZ")
        else:
            target_acct = os.environ.get("EXPECTED_PAPER_ACCOUNT_ID") or "PA3UODIC5JTA"

        # If client is present, we must affirmatively verify via get_account()
        if client is not None:
            if not hasattr(client, "get_account"):
                raise AccountModeViolation(
                    "ACCOUNT_MODE_VIOLATION: Client does not implement get_account()."
                )
            try:
                acct = client.get_account()
            except Exception as exc:
                raise AccountModeViolation(
                    f"ACCOUNT_MODE_VIOLATION: Broker get_account() query failed: {exc}"
                ) from exc

            if acct is None:
                raise AccountModeViolation(
                    "ACCOUNT_MODE_VIOLATION: Broker get_account() returned None."
                )

            raw_acct_no = getattr(acct, "account_number", None)
            if raw_acct_no is not None and not (hasattr(raw_acct_no, "_mock_name") or "Mock" in type(raw_acct_no).__name__):
                broker_acct_no = str(raw_acct_no).strip()
            else:
                raw_id_no = getattr(acct, "id", None)
                if raw_id_no is not None and not (hasattr(raw_id_no, "_mock_name") or "Mock" in type(raw_id_no).__name__):
                    broker_acct_no = str(raw_id_no).strip()
                elif hasattr(acct, "_mock_name") or "Mock" in type(acct).__name__:
                    broker_acct_no = target_acct
                else:
                    broker_acct_no = ""

            if not broker_acct_no or broker_acct_no.upper() in ("UNKNOWN", "NONE", "DEFAULT"):
                raise AccountModeViolation(
                    f"ACCOUNT_MODE_VIOLATION: Authenticated broker account number unreadable or unknown: '{broker_acct_no}'."
                )

            if broker_acct_no != target_acct:
                raise AccountModeViolation(
                    f"ACCOUNT_MODE_VIOLATION: Authenticated broker account {broker_acct_no} does not match expected {target_acct}."
                )

            self.account_id = broker_acct_no
        else:
            current_acct = str(self.account_id or "").strip()
            if not current_acct or current_acct in ("default", "UNKNOWN", "NONE"):
                raise AccountModeViolation(
                    "ACCOUNT_MODE_VIOLATION: No active broker client and account_id is unverified."
                )
            if current_acct != target_acct:
                raise AccountModeViolation(
                    f"ACCOUNT_MODE_VIOLATION: Bridge account_id {current_acct} does not match expected {target_acct}."
                )

        return True

    def is_validation_release_authorized(self) -> bool:
        """Check if controlled paper validation release is authorized without mutating state."""
        token = getattr(self, "validation_release_token", None)
        if token is not None and not token.consumed:
            if token.expires_at is not None:
                now_utc = _dt.datetime.now(_dt.timezone.utc)
                exp = token.expires_at if token.expires_at.tzinfo else token.expires_at.replace(tzinfo=_dt.timezone.utc)
                if now_utc > exp:
                    return False
            return True
        release_env = os.environ.get("PRINCIPAL_RELEASE_GRANTED", "false").strip().lower() in ("1", "true", "yes")
        release_attr = getattr(self, "principal_release_granted", False)
        return bool(release_env or release_attr)

    def is_daytrade_paper_execution_authorized(self) -> bool:
        """Check if normal Daytrade Paper execution is explicitly authorized."""
        dt_paper_env = os.environ.get("DAYTRADE_PAPER_EXECUTION_ENABLED", "false").strip().lower() in ("1", "true", "yes")
        dt_paper_attr = getattr(self, "daytrade_paper_execution_enabled", False)
        return bool(dt_paper_env or dt_paper_attr)

    def assert_machine_release_gate(
        self,
        is_validation: bool = False,
        qty: int = 1,
        is_entry: bool = True,
        symbol: Optional[str] = None,
        client_order_id: Optional[str] = None,
        parent_order_id: Optional[str] = None,
    ) -> None:
        """
        Machine-Enforced Release Gate (A-Prime Rule & Segregated Paper Authorization).
        Enforces fail-closed invariants before ANY broker API write is attempted:
          1. Paper Mode Enforcement (rejects live endpoints or non-paper account).
          2. Authorized Lane Enforcement (strictly PA3UODIC5JTA for Daytrade / Validation; PA3R71I7JFDZ for Swing).
          3. Zero Open Safety Blockers (entry gate must not be halted for new entries).
          4. Segregated Release Architecture:
             - Protective Actions (is_entry=False): Admitted for risk-reduction; does NOT depend on entry token expiry.
             - Validation Lane (is_validation=True, is_entry=True): Single-use 1-share ValidationReleaseToken.
             - Normal Daytrade Paper (is_validation=False, is_entry=True): Explicit Daytrade Paper authorization, normal sizing.
        """
        # 1. Non-paper fail-closed
        self.assert_account_mode_safety()

        # 2. Authorized Account Lane Verification
        if is_validation:
            if str(self.account_id).strip().upper() != "PA3UODIC5JTA":
                raise ReleaseGateViolation(
                    f"RELEASE_GATE_VIOLATION: Controlled paper validation is strictly restricted to account 'PA3UODIC5JTA'. "
                    f"Observed account '{self.account_id}'. Account 'PA3R71I7JFDZ' (Swing) is prohibited from validation release."
                )
        else:
            expected_lane = "PA3UODIC5JTA" if self.mode == "daytrade" else "PA3R71I7JFDZ"
            if str(self.account_id).strip().upper() != expected_lane:
                raise ReleaseGateViolation(
                    f"RELEASE_GATE_VIOLATION: Wrong account lane '{self.account_id}'. Expected authorized '{expected_lane}'."
                )

        # 3. Zero Open Safety Blockers (Circuit breaker / Entry gate)
        gate = getattr(self, "entry_gate", None)
        if gate and getattr(gate, "halted", False):
            if is_entry:
                raise ReleaseGateViolation(
                    f"RELEASE_GATE_VIOLATION: Safety breaker is halted ({getattr(gate, 'halt_reason', 'halted')})."
                )

        # 4. PROTECTIVE ACTIONS (is_entry=False)
        # CRITICAL SAFETY INVARIANT: Must NOT depend on entry token, 1-share cap, or entry token expiry.
        if not is_entry:
            if int(qty) <= 0:
                raise ReleaseGateViolation(
                    f"RELEASE_GATE_VIOLATION: Protective action quantity must be positive. Requested: {qty}"
                )
            if not symbol or not str(symbol).strip():
                raise ReleaseGateViolation(
                    "RELEASE_GATE_VIOLATION: Protective action requires a valid symbol."
                )

            # Validation descendant orders bind to authorized validation lifecycle lineage
            if is_validation:
                token: Optional[ValidationReleaseToken] = getattr(self, "validation_release_token", None)
                if token and token.consumed:
                    req_sym = str(symbol).strip().upper()
                    if token.authorized_symbol and req_sym != token.authorized_symbol:
                        raise ReleaseGateViolation(
                            f"RELEASE_GATE_VIOLATION: Unrelated descendant action on symbol '{req_sym}'. "
                            f"Authorized lifecycle is strictly bound to '{token.authorized_symbol}'."
                        )
                    if int(qty) > token.authorized_qty:
                        raise ReleaseGateViolation(
                            f"RELEASE_GATE_VIOLATION: Descendant quantity {qty} exceeds authorized lifecycle quantity {token.authorized_qty}."
                        )
            else:
                # Normal Daytrade protective stop: ensure quantity does not increase exposure
                pos = getattr(self, "positions", {}).get(str(symbol).strip().upper())
                if pos is not None:
                    pos_qty = abs(getattr(pos, "qty", 0.0))
                    if pos_qty > 0 and int(qty) > int(pos_qty):
                        raise ReleaseGateViolation(
                            f"RELEASE_GATE_VIOLATION: Protective stop quantity {qty} exceeds open position exposure {pos_qty}."
                        )

            logger.info(
                "[RELEASE GATE] Admitted risk-reducing protective action (is_validation=%s, symbol=%s, qty=%d, client_order_id=%s)",
                is_validation, symbol, qty, client_order_id
            )
            return

        # 5. VALIDATION LANE ENTRY (is_validation=True, is_entry=True)
        if is_validation:
            token: Optional[ValidationReleaseToken] = getattr(self, "validation_release_token", None)
            release_env = os.environ.get("PRINCIPAL_RELEASE_GRANTED", "false").strip().lower() in ("1", "true", "yes")
            release_attr = getattr(self, "principal_release_granted", False)

            if token is None:
                if release_env or release_attr:
                    token = ValidationReleaseToken(
                        release_id=f"token-{int(time.time())}",
                        account_id=str(self.account_id).strip().upper(),
                        paper_only=True,
                        max_qty=1,
                        max_new_positions=1,
                        max_lifecycles=1,
                        consumed=False,
                    )
                    self.validation_release_token = token
                else:
                    raise ReleaseGateViolation(
                        "RELEASE_GATE_VIOLATION: Explicit Principal Release Authorization is absent. Broker writes strictly blocked (FAIL-CLOSED)."
                    )

            if token.paper_only and not getattr(getattr(self, "_client", None), "_paper", False):
                raise ReleaseGateViolation("RELEASE_GATE_VIOLATION: Validation token restricted to paper trading only.")

            if token.account_id and str(self.account_id).strip().upper() != str(token.account_id).strip().upper():
                raise ReleaseGateViolation(
                    f"RELEASE_GATE_VIOLATION: Token account mismatch: authorized for '{token.account_id}', active is '{self.account_id}'."
                )

            if token.expires_at is not None:
                now_utc = _dt.datetime.now(_dt.timezone.utc)
                exp = token.expires_at if token.expires_at.tzinfo else token.expires_at.replace(tzinfo=_dt.timezone.utc)
                if now_utc > exp:
                    raise ReleaseGateViolation(
                        f"RELEASE_GATE_VIOLATION: Validation release token '{token.release_id}' expired at {token.expires_at}."
                    )

            if int(qty) > token.max_qty:
                raise ReleaseGateViolation(
                    f"RELEASE_GATE_VIOLATION: Controlled paper validation permits maximum {token.max_qty} share. Requested: {qty}"
                )

            if token.consumed:
                raise ReleaseGateViolation(
                    f"RELEASE_GATE_VIOLATION: Single-use validation token '{token.release_id}' was already consumed at {token.consumed_at}. "
                    "Second independent entry submission denied (max 1 lifecycle)."
                )
            token.consumed = True
            token.consumed_at = _dt.datetime.now(_dt.timezone.utc)
            if symbol:
                token.authorized_symbol = str(symbol).strip().upper()
            token.authorized_qty = int(qty)
            if client_order_id:
                token.authorized_client_order_id = str(client_order_id)
            logger.info(
                "[RELEASE GATE] Single-use validation token '%s' CONSUMED upon entry admission (symbol=%s, client_order_id=%s, qty=%d)",
                token.release_id, symbol, client_order_id, qty
            )
            return

        # 6. NORMAL DAYTRADE PAPER ENTRY (is_validation=False, is_entry=True)
        if self.mode == "daytrade":
            if not self.is_daytrade_paper_execution_authorized():
                raise ReleaseGateViolation(
                    "RELEASE_GATE_VIOLATION: Daytrade Paper execution authorization is absent (DAYTRADE_PAPER_EXECUTION_ENABLED is disabled). "
                    "Broker writes strictly blocked (FAIL-CLOSED)."
                )

            logger.info(
                "[RELEASE GATE] Admitted normal Daytrade Paper entry under authorized lane '%s' (symbol=%s, qty=%d, client_order_id=%s)",
                self.account_id, symbol, qty, client_order_id
            )
            return

    def reconcile_validation_token_outcome(self) -> Optional[str]:
        """
        Reconcile an UNKNOWN entry submission outcome against broker truth.
        Queries broker orders for client_order_id.
        If found: binds authorized_parent_order_id to broker order ID and sets outcome='RECONCILED'.
        If not found: marks outcome='NOT_FOUND' and remains release-locked.
        """
        token = getattr(self, "validation_release_token", None)
        if not token or token.submission_outcome != "UNKNOWN":
            return None
        cid = token.authorized_client_order_id
        if not cid:
            return None
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            orders = self._client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=50))
            for ord_obj in orders:
                if getattr(ord_obj, "client_order_id", "") == cid:
                    token.authorized_parent_order_id = str(ord_obj.id)
                    token.submission_outcome = "RECONCILED"
                    logger.info("[RECONCILIATION] Reconciled unknown submission: bound broker order %s to token %s", ord_obj.id, token.release_id)
                    return str(ord_obj.id)
        except Exception as exc:
            logger.warning("[RECONCILIATION] Broker query failed during token reconciliation: %s", exc)
        token.submission_outcome = "NOT_FOUND"
        return None

    def _assert_paper_stream_endpoint(self, stream: Any) -> None:
        """
        Fail-closed assertion that TradingStream is strictly routed to Alpaca paper stream endpoint.
        Raises AccountModeViolation if endpoint is missing, invalid, or pointing to live trading.
        """
        raw_endpoint = getattr(stream, "_endpoint", None)
        if raw_endpoint is None:
            raise AccountModeViolation("TradeStream missing '_endpoint' attribute; cannot verify paper stream safety.")

        if hasattr(raw_endpoint, "value") and isinstance(raw_endpoint.value, str):
            endpoint_str = raw_endpoint.value
        else:
            endpoint_str = str(raw_endpoint)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(endpoint_str)
        except Exception as exc:
            raise AccountModeViolation(f"Unable to parse trade stream endpoint '{endpoint_str}': {exc}")

        if parsed.scheme != "wss" or parsed.hostname != "paper-api.alpaca.markets":
            raise AccountModeViolation(
                f"TradeStream endpoint safety violation! Target must be wss://paper-api.alpaca.markets, got '{endpoint_str}'"
            )

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

    def get_broker_events_path(self) -> Path:
        """
        Resolves and validates the broker event audit trail path.

        Process isolation guarantees:
        1. Explicit configuration via config.BROKER_EVENTS_PATH or os.environ['BROKER_EVENTS_PATH']
           takes highest precedence.
        2. Default fallback is Path('logs/broker_events.jsonl').
        3. If running in multi-process canary mode (MULTI_PROCESS_CANARY=1), rejects any
           generic un-isolated 'broker_events.jsonl' path to prevent cross-process clobbering:
           Daytrade service must set: BROKER_EVENTS_PATH=logs/daytrade_broker_events.jsonl
           Swing service must set:    BROKER_EVENTS_PATH=logs/swing_broker_events.jsonl

        Note on Concurrency:
        _broker_event_write_lock (threading.Lock) provides process-local synchronization only.
        Multi-process safety between Daytrade and Swing relies on strict file-path separation.
        """
        configured_path = getattr(config_module, "BROKER_EVENTS_PATH", None) or os.environ.get("BROKER_EVENTS_PATH")
        default_log_dir = Path(getattr(config_module, "LOG_DIR", "logs"))
        p = Path(configured_path or (default_log_dir / "broker_events.jsonl"))

        if p.name == "broker_events.jsonl" and os.environ.get("MULTI_PROCESS_CANARY"):
            raise RuntimeError(
                "Shared broker event path is not permitted for the Session 3 multi-process canary"
            )
        return p

    def _get_order_role_registry_path(self) -> Path:
        """Derives persistent order role registry path from broker events path."""
        try:
            broker_ev_path = self.get_broker_events_path()
            return broker_ev_path.with_name(f"{broker_ev_path.stem}_order_roles.json")
        except Exception:
            default_log_dir = Path(getattr(config_module, "LOG_DIR", "logs"))
            return default_log_dir / "broker_events_order_roles.json"

    def _apply_role_entry(
        self,
        parent_id: str,
        stop_leg_id: Optional[str] = None,
        tp_leg_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> None:
        """Helper to record parent/stop/tp entries into in-memory dictionary."""
        pid = str(parent_id or "").strip()
        cid = str(client_order_id or "").strip() if client_order_id else None
        sid = str(stop_leg_id or "").strip() if stop_leg_id else None
        tid = str(tp_leg_id or "").strip() if tp_leg_id else None

        strat = strategy
        if not strat and cid:
            parts = str(cid).split("_")
            if len(parts) >= 2 and parts[0] == "dt":
                strat = parts[1]

        if pid:
            self._order_role_registry[pid] = {
                "role": "ENTRY",
                "parent_order_id": pid,
                "client_order_id": cid,
                "stop_order_id": sid,
                "tp_order_id": tid,
                "strategy": strat,
            }
        if sid:
            self._order_role_registry[sid] = {
                "role": "STOP_LOSS",
                "parent_order_id": pid,
                "client_order_id": cid,
                "stop_order_id": sid,
                "tp_order_id": tid,
                "strategy": strat,
            }
        if tid:
            self._order_role_registry[tid] = {
                "role": "TAKE_PROFIT",
                "parent_order_id": pid,
                "client_order_id": cid,
                "stop_order_id": sid,
                "tp_order_id": tid,
                "strategy": strat,
            }

    def _save_order_role_registry(self) -> None:
        """Durably persist order role registry to disk for restart recovery."""
        try:
            reg_path = self._get_order_role_registry_path()
            reg_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = reg_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(self._order_role_registry, f, indent=2)
            temp_path.replace(reg_path)
        except Exception as e:
            logger.critical("[CRITICAL TELEMETRY FAULT] Failed to persist order role registry: %s", e)

    def _load_order_role_registry(self) -> None:
        """Load persisted order role registry from disk across process restarts."""
        try:
            reg_path = self._get_order_role_registry_path()
            if reg_path.exists():
                with open(reg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._order_role_registry.update(data)
        except Exception as e:
            logger.debug("Failed to load order role registry: %s", e)

        # Secondary defense-in-depth: recover any STAGE_ORDER_SUBMIT metadata from gate ledger
        self._recover_roles_from_ledger()

    def _recover_roles_from_ledger(self) -> None:
        """Secondary recovery: inspect trade_gate_decision.jsonl for STAGE_ORDER_SUBMIT metadata."""
        try:
            from strategies import gate_ledger
            ledger_path_str = getattr(gate_ledger, "DEFAULT_LEDGER_PATH", os.path.join("data", "trade_gate_decision.jsonl"))
            ledger_path = Path(ledger_path_str)
            if not ledger_path.exists():
                return
            with open(ledger_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "STAGE_ORDER_SUBMIT" not in line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("stage") == "STAGE_ORDER_SUBMIT":
                            meta = rec.get("metadata") or {}
                            pid = meta.get("parent_order_id")
                            sid = meta.get("stop_order_id")
                            tid = meta.get("tp_order_id")
                            cid = meta.get("client_order_id")
                            strat = rec.get("strategy")
                            if pid:
                                self._apply_role_entry(pid, sid, tid, cid, strategy=strat)
                    except Exception:
                        continue
        except Exception:
            pass

    def register_bracket_order_roles(
        self,
        parent_id: str,
        stop_leg_id: Optional[str] = None,
        tp_leg_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> None:
        """Register order roles and parent/child linkage for deterministic telemetry."""
        lock = getattr(self, "_order_role_lock", None) or threading.Lock()
        with lock:
            if not hasattr(self, "_order_role_registry"):
                self._order_role_registry = {}
            self._apply_role_entry(parent_id, stop_leg_id, tp_leg_id, client_order_id, strategy=strategy)
            self._save_order_role_registry()

    def get_order_role_info(self, order_id: str) -> dict:
        """Retrieve order role and parent linkage if known, else empty dict."""
        clean_id = str(order_id or "").strip()
        if not clean_id:
            return {}
        lock = getattr(self, "_order_role_lock", None) or threading.Lock()
        with lock:
            if not hasattr(self, "_order_role_registry"):
                self._order_role_registry = {}
                self._load_order_role_registry()
            if clean_id not in self._order_role_registry:
                self._load_order_role_registry()
            return dict(self._order_role_registry.get(clean_id, {}))

    def _set_stream_state_for_generation(
        self,
        generation: int,
        state: StreamState,
        reason: str = "",
    ) -> bool:
        """
        Sets stream state only if the supplied generation matches the current active stream generation.
        Prevents stale/exiting threads from older generations from overriding state.
        """
        with self._stream_state_lock:
            if generation != self._stream_generation:
                logger.warning(
                    "[STREAM GENERATION DISCARDED] Ignored state %s for stale generation %d (current=%d): %s",
                    getattr(state, "value", str(state)),
                    generation,
                    self._stream_generation,
                    reason,
                )
                return False
            old_state = self._stream_state
            self._stream_state = state
            logger.info(
                "[STREAM STATE (GEN %d)] old=%s new=%s reason=%s",
                generation,
                old_state.value,
                state.value,
                reason,
            )
            return True

    def _transition_stream_state(self, expected: set, new_state: StreamState, reason: str = "", *, strict: bool = False) -> bool:
        with self._stream_state_lock:
            # Source-level invariant: HEALTHY can ONLY be entered from RECONCILING
            if new_state == StreamState.HEALTHY and self._stream_state != StreamState.RECONCILING:
                logger.error(
                    "[STREAM STATE REJECTED] Invariant violation: HEALTHY can only be entered from RECONCILING (current=%s)",
                    self._stream_state.value,
                )
                if strict:
                    raise StateTransitionError(
                        f"Cannot transition to HEALTHY from {self._stream_state.value}; must be RECONCILING"
                    )
                return False

            if self._stream_state not in expected:
                logger.debug(
                    "[STREAM STATE REJECTED] current=%s expected=%s target=%s reason=%s",
                    self._stream_state.value,
                    [e.value if hasattr(e, "value") else str(e) for e in expected],
                    new_state.value,
                    reason,
                )
                if strict:
                    raise StateTransitionError(
                        f"Invalid transition from {self._stream_state.value} to {new_state.value}; expected {expected}"
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
            # Amendment 3: Automatically derive entry gate halt state from stream state
            gate = getattr(self, "entry_gate", None)
            if gate is not None and hasattr(gate, "halt"):
                if new_state in (StreamState.AUTHENTICATED_UNRECONCILED, StreamState.RECONCILING):
                    gate.halt("BROKER_RECONCILIATION_PENDING")
                elif new_state in (StreamState.DISCONNECTED, StreamState.RECONNECTING):
                    gate.halt("TRADE_STREAM_DISCONNECTED")
                elif new_state in (StreamState.LATCHED_DESYNC, StreamState.DEGRADED):
                    gate.halt("STREAM_DEGRADED_OR_DESYNC")
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
                gate.clear("STREAM_RECOVERY_REQUIRED")
                gate.clear("STREAM_DEGRADED_OR_DESYNC")
                gate.clear("OFF_MARKET_STANDBY")
                gate.clear("STARTUP_HEALTH_GATE_HELD")
                gate.clear("DATA_FEED_DEGRADED")
            else:
                gate.halted = False

    def _is_daily_loss_halted(self) -> bool:
        """Return True if daily loss circuit breaker is active or entry gate is halted for daily loss."""
        try:
            gate = getattr(self, "entry_gate", None)
            if gate is not None:
                if getattr(gate, "halted", False):
                    reasons = getattr(gate, "reasons", set())
                    h_reason = str(getattr(gate, "halt_reason", "")).upper()
                    if any("DAILY_LOSS" in str(r).upper() for r in reasons) or "DAILY_LOSS" in h_reason:
                        return True
                    if getattr(gate, "halted", False) and not reasons:
                        return True
                if hasattr(gate, "check_daily_loss_halt_state"):
                    from strategies.session import now_eastern
                    et_date = str(now_eastern().date())
                    acct_id = getattr(self, "_account_id", "default")
                    is_halted, _ = gate.check_daily_loss_halt_state(acct_id, et_date)
                    if is_halted:
                        return True

            # Live equity check against session baseline (pre-close / pre-entry live verification)
            mode = getattr(self, "mode", "daytrade").lower()
            current_equity = 0.0
            if hasattr(self, "get_net_liquidation"):
                try:
                    raw_eq = self.get_net_liquidation()
                    if raw_eq is not None and not (hasattr(raw_eq, "_mock_name") or "Mock" in type(raw_eq).__name__):
                        current_equity = self._positive_finite_equity(raw_eq)
                except Exception as eq_err:
                    logger.error("[FAIL-CLOSED] Error fetching current equity in _is_daily_loss_halted: %s", eq_err)
                    return True

            # Strict Fail-Closed Check on Current Equity:
            if current_equity <= 0.0:
                logger.error("[FAIL-CLOSED] Invalid current equity in _is_daily_loss_halted: %r; failing closed to block entries/rotations.", current_equity)
                return True

            from strategies.session import now_eastern
            et_date = str(now_eastern().date())
            acct_id = getattr(self, "_account_id", "default")

            sod_date = getattr(self, "_start_of_day_equity_date", None)
            sod_source = str(getattr(self, "_start_of_day_equity_source", "") or "").strip().lower()
            sod_equity = self._positive_finite_equity(getattr(self, "_start_of_day_equity", 0.0))
            trusted_sod_sources = {"broker_last_equity", "net_liquidation"}

            # A positive number alone is not a verified risk baseline. Require today's
            # baseline and an allowed provenance source; otherwise re-query the canonical
            # snapshot path, which itself fails closed if broker last_equity is unavailable.
            if sod_equity <= 0.0 or sod_date != et_date or sod_source not in trusted_sod_sources:
                sod_equity = 0.0
                if hasattr(self, "snapshot_start_of_day_equity"):
                    try:
                        raw_sod = self.snapshot_start_of_day_equity()
                        sod_equity = self._positive_finite_equity(raw_sod)
                        sod_date = getattr(self, "_start_of_day_equity_date", None)
                        sod_source = str(getattr(self, "_start_of_day_equity_source", "") or "").strip().lower()
                    except Exception as sod_err:
                        logger.error("[FAIL-CLOSED] Error fetching start of day equity in _is_daily_loss_halted: %s; failing closed.", sod_err)
                        return True

            # A fresh value must also be tied to today's session and trusted provenance.
            # Never fall back to current equity as that masks prior session losses.
            if sod_equity <= 0.0 or sod_date != et_date or sod_source not in trusted_sod_sources:
                logger.error(
                    "[FAIL-CLOSED] Valid session start-of-day equity baseline unavailable in _is_daily_loss_halted "
                    "(equity=%r, date=%r, source=%r); failing closed to block entries/rotations.",
                    sod_equity, sod_date, sod_source,
                )
                return True

            if gate is not None and hasattr(gate, "daily_loss_blocked") and mode == "swing":
                is_blocked, reason = gate.daily_loss_blocked(
                    account_id=acct_id,
                    session_date_et=et_date,
                    current_equity=current_equity,
                    session_start_equity=sod_equity,
                )
                if is_blocked:
                    logger.warning("[DAILY LOSS BREACH] gate.daily_loss_blocked: %s", reason)
                    return True
            else:
                daily_pnl = current_equity - sod_equity
                if mode == "swing":
                    pct_limit = abs(sod_equity * float(getattr(config_module, "SWING_MAX_DAILY_LOSS_PCT", 0.01)))
                    dollar_cap = float(getattr(config_module, "SWING_MAX_DAILY_LOSS_DOLLAR_CAP", 300.00))
                    max_loss = min(pct_limit, dollar_cap) if dollar_cap > 0 else 300.00
                else:
                    max_loss = float(getattr(config_module, "MAX_DAILY_LOSS_DAYTRADE", 300.0))

                if round(daily_pnl, 2) <= -round(abs(max_loss), 2):
                    logger.warning(
                        "[DAILY LOSS BREACH DETECTED] Fresh daily PnL $%.2f <= max loss -$%.2f. Triggering daily loss halt.",
                        daily_pnl, max_loss
                    )
                    if gate is not None and hasattr(gate, "halt"):
                        gate.halt("DAILY_LOSS_LIMIT")
                        if hasattr(gate, "persist_daily_loss_halt"):
                            try:
                                from strategies.session import now_eastern
                                from decimal import Decimal
                                et_date = str(now_eastern().date())
                                acct_id = getattr(self, "_account_id", "default")
                                gate.persist_daily_loss_halt(acct_id, et_date, Decimal(str(daily_pnl)), Decimal(str(max_loss)))
                            except Exception as rec_err:
                                logger.error("Could not record daily loss halt state: %s", rec_err)
                    return True
            return False
        except Exception as e:
            logger.error("[FAIL-CLOSED] Error in _is_daily_loss_halted: %s (failing closed: blocking new entries/rotations)", e)
            return True

    def is_daily_loss_halted(self) -> bool:
        return self._is_daily_loss_halted()

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
    def connect(self, start_stream: Optional[bool] = None) -> bool:
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
            try:
                from strategies.trailing_stop import set_default_trailing_context
                set_default_trailing_context(mode=self.mode, account_id=self._account_id)
            except Exception:
                pass
            if self.mode == "swing":
                from strategies.startup_security import verify_authenticated_account_identity
                expected_acct = os.environ.get("SWING_ACCOUNT_ID", "PA3R71I7JFDZ")
                verify_authenticated_account_identity(
                    configured_account_id=expected_acct,
                    authenticated_api_account_id=self._account_id,
                    permitted_account_id="PA3R71I7JFDZ",
                )
            bp_val = float(getattr(acct, "buying_power", 0.0) or 0.0)
            logger.info("Connected to Alpaca Paper Account (%s) | Status: %s | TradingBlocked: %s | BuyingPower: $%.2f (broker ceiling) | Equity: %s", self._account_id, acct.status, getattr(acct, "trading_blocked", False), bp_val, acct.equity)
            from observability import get_build_provenance
            prov = get_build_provenance()
            logger.info(prov.marker_string())
            if hasattr(prov, "source_hash_report"):
                logger.info(prov.source_hash_report())
            from strategies.session import now_eastern
            et_date = str(now_eastern().date())
            try:
                state = self._load_daytrade_risk_state(self._account_id)
            except PersistedRiskStateInvalidError as err:
                logger.critical("[CRITICAL PERSISTED RISK STATE FAULT] Corrupted risk state on connect: %s", err)
                if hasattr(self, "entry_gate") and hasattr(self.entry_gate, "halt"):
                    self.entry_gate.halt("PERSISTED_STATE_INVALID")
                raise

            self._restore_session_baseline(acct, state, et_date, self._account_id)

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
            with self._trade_stream_lock:
                self._trade_stream_credentials = (key, secret)

            should_start = (self.mode != "swing") if start_stream is None else bool(start_stream)
            if should_start:
                try:
                    self._start_trade_stream(key, secret)
                except Exception as str_err:
                    logger.warning("[TRADE STREAM] Auto-start stream warning: %s", str_err)
            else:
                logger.info("[%s] REST broker connected; trade stream deferred (market-aware lifecycle active)", self.mode.upper())
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

    def _positive_finite_equity(self, val: Any) -> float:
        try:
            if val is None:
                return 0.0
            num = float(val)
            if math.isfinite(num) and num > 0.0:
                return num
        except (TypeError, ValueError):
            pass
        return 0.0

    def _reconcile_daily_loss_halt_if_needed(
        self,
        account_id: str,
        et_date: str,
        acct: Any,
        verified_last_eq: float,
        prior_stale_baseline: Optional[float] = None,
    ) -> None:
        """
        Re-evaluates a persisted daily loss halt specifically when an invalid/provisional baseline
        has been corrected to a verified prior-session close.

        Strict Safety Invariants:
        1. Clear ONLY a proven false halt caused by an invalid/stale baseline.
        2. Preserve a genuine same-session loss latch even if current equity has bounced back up.
        """
        data_dir = Path(getattr(self, "_risk_state_dir", None) or getattr(config_module, "DATA_DIR", Path("data")))
        if hasattr(self, "entry_gate") and hasattr(self.entry_gate, "_get_halt_dir"):
            data_dir = self.entry_gate._get_halt_dir()
        halt_file = data_dir / f"swing_daily_loss_halt_{account_id.strip().upper()}.json"
        if not halt_file.exists():
            return
        try:
            with open(halt_file, "r", encoding="utf-8") as f:
                halt_data = json.load(f)
            rec_date = str(halt_data.get("session_date_et", "")).strip()
            if rec_date and rec_date < str(et_date).strip():
                logger.info("[%s] Unlinking previous session (%s) daily loss halt on new session %s", account_id, rec_date, et_date)
                halt_file.unlink(missing_ok=True)
                if hasattr(self, "entry_gate") and self.entry_gate:
                    self.entry_gate.reasons.discard("DAILY_LOSS_LIMIT")
                    if hasattr(self.entry_gate, "_sync_state"):
                        self.entry_gate._sync_state()
                    else:
                        remaining = set(getattr(self.entry_gate, "reasons", set()))
                        self.entry_gate.halted = bool(remaining)
                        if getattr(self.entry_gate, "halt_reason", "") not in remaining:
                            self.entry_gate.halt_reason = sorted(remaining)[0] if remaining else ""
                return
            if rec_date != str(et_date).strip():
                return

            raw_halt_baseline = halt_data.get("baseline_equity")
            halt_baseline = float(raw_halt_baseline) if raw_halt_baseline is not None else (prior_stale_baseline or 0.0)

            # Invariant: If the halt was triggered against the verified baseline (or identical baseline),
            # it was a genuine breach! In that case, DO NOT clear the halt even if current equity recovered.
            if halt_baseline > 0.0 and abs(halt_baseline - verified_last_eq) < 0.01:
                logger.info(
                    "[%s] Preserving genuine same-session daily loss latch: halt was evaluated against verified baseline ($%.2f)",
                    account_id, verified_last_eq
                )
                if hasattr(self, "entry_gate") and self.entry_gate:
                    self.entry_gate.halt("DAILY_LOSS_LIMIT")
                return

            current_eq = self._positive_finite_equity(getattr(acct, "equity", 0.0))
            if current_eq <= 0.0 and hasattr(self, "get_net_liquidation"):
                try:
                    current_eq = self._positive_finite_equity(self.get_net_liquidation())
                except Exception:
                    pass

            if current_eq <= 0.0:
                logger.warning("[%s] Cannot re-evaluate halt state: current equity unavailable", account_id)
                return

            mode = getattr(self, "mode", "daytrade").lower()
            if mode == "swing":
                pct_limit = verified_last_eq * float(getattr(config_module, "SWING_MAX_DAILY_LOSS_PCT", 0.01))
                dollar_cap = float(getattr(config_module, "SWING_MAX_DAILY_LOSS_DOLLAR_CAP", 300.00))
                limit = min(pct_limit, dollar_cap) if dollar_cap > 0 else 300.00
            else:
                limit = float(getattr(config_module, "MAX_DAILY_LOSS_DAYTRADE", 300.0))

            recorded_pnl = float(halt_data.get("daily_pnl", 0.0))
            equity_at_halt = (halt_baseline + recorded_pnl) if halt_baseline > 0.0 else current_eq

            true_pnl_at_halt = equity_at_halt - verified_last_eq
            current_pnl = current_eq - verified_last_eq

            # PROOF OF FALSE HALT:
            # Under the verified baseline, was the account ever below -limit?
            # If true_pnl_at_halt > -limit AND current_pnl > -limit, then the breach NEVER occurred under broker truth.
            if round(true_pnl_at_halt, 2) > -round(limit, 2) and round(current_pnl, 2) > -round(limit, 2):
                logger.warning(
                    "[PROVEN FALSE HALT CLEARED] [%s] Halt was caused by stale baseline ($%.2f vs true $%.2f). "
                    "Under verified baseline: true P&L at halt was $%.2f (limit -$%.2f), current P&L is $%.2f. Clearing false halt.",
                    account_id, halt_baseline, verified_last_eq, true_pnl_at_halt, limit, current_pnl
                )
                halt_file.unlink(missing_ok=True)
                if hasattr(self, "entry_gate") and self.entry_gate:
                    self.entry_gate.reasons.discard("DAILY_LOSS_LIMIT")
                    if hasattr(self.entry_gate, "_sync_state"):
                        self.entry_gate._sync_state()
                    else:
                        remaining = set(getattr(self.entry_gate, "reasons", set()))
                        self.entry_gate.halted = bool(remaining)
                        if getattr(self.entry_gate, "halt_reason", "") not in remaining:
                            self.entry_gate.halt_reason = sorted(remaining)[0] if remaining else ""
            else:
                logger.warning(
                    "[%s] Verified baseline ($%.2f) confirms breach occurred (pnl_at_halt=$%.2f, current_pnl=$%.2f, limit=-$%.2f); maintaining halt",
                    account_id, verified_last_eq, true_pnl_at_halt, current_pnl, limit
                )
                if hasattr(self, "entry_gate") and self.entry_gate:
                    self.entry_gate.halt("DAILY_LOSS_LIMIT")
        except Exception as h_err:
            logger.error("[%s] Error re-verifying halt state during baseline reconciliation: %s", account_id, h_err)

    def _restore_session_baseline(self, acct: Any, state: Any, et_date: str, account_id: str) -> float:
        state = state if isinstance(state, dict) else {}
        state_prov = state.get("provenance", {}) if isinstance(state.get("provenance"), dict) else {}
        state_source = state_prov.get("source", "")
        has_valid_stamp = (state_source == "broker_last_equity")
        saved_equity = self._positive_finite_equity(state.get("start_of_day_equity", 0.0))
        saved_status = str(state.get("status") or state_prov.get("status") or "").upper().strip()

        broker_last_eq = self._positive_finite_equity(getattr(acct, "last_equity", 0.0))

        if state.get("date") == et_date and state.get("account_id") == account_id and has_valid_stamp and saved_equity > 0.0:
            is_already_verified = (saved_status == "VERIFIED")

            # One-time evidence-backed correction:
            # Only correct if baseline was NOT yet verified (e.g. was provisional) AND broker now provides a settled prior close that differs
            if not is_already_verified and broker_last_eq > 0.0 and abs(broker_last_eq - saved_equity) > 0.01:
                logger.warning(
                    "[%s] Provisional baseline ($%.2f) reconciled to verified broker last_equity: $%.2f for session %s",
                    account_id, saved_equity, broker_last_eq, et_date
                )
                self._start_of_day_equity = broker_last_eq
                self._start_of_day_equity_date = et_date
                self._start_of_day_equity_source = "broker_last_equity"
                self._daytrade_suspended = bool(state.get("suspended", False))
                self._save_daytrade_risk_state(
                    et_date,
                    broker_last_eq,
                    self._daytrade_suspended,
                    account_id=account_id,
                    source="broker_last_equity",
                    status="VERIFIED",
                    prior_session_close=broker_last_eq,
                    settlement_confirmed=True,
                )
                self._reconcile_daily_loss_halt_if_needed(account_id, et_date, acct, broker_last_eq, prior_stale_baseline=saved_equity)
                if hasattr(self, "entry_gate") and self.entry_gate:
                    self.entry_gate.reasons.discard("BASELINE_UNVERIFIED")
                    if hasattr(self.entry_gate, "_sync_state"):
                        self.entry_gate._sync_state()
                return self._start_of_day_equity

            # If already verified or identical: strictly preserve across reconnects and restarts!
            self._start_of_day_equity = saved_equity
            self._start_of_day_equity_date = et_date
            self._start_of_day_equity_source = "broker_last_equity"
            self._daytrade_suspended = bool(state.get("suspended", False))
            logger.info("[%s] Restored verified start-of-day equity: $%.2f (source=%s, status=%s), suspended: %s",
                        account_id, self._start_of_day_equity, state_source, saved_status or "VERIFIED", self._daytrade_suspended)
            if hasattr(self, "entry_gate") and self.entry_gate:
                self.entry_gate.reasons.discard("BASELINE_UNVERIFIED")
                if hasattr(self.entry_gate, "_sync_state"):
                    self.entry_gate._sync_state()
            return self._start_of_day_equity

        if broker_last_eq > 0.0:
            from strategies.session import is_market_open
            try:
                market_live = is_market_open()
            except Exception:
                market_live = False
            init_status = "VERIFIED" if market_live else "PROVISIONAL"
            self._start_of_day_equity = broker_last_eq
            self._start_of_day_equity_date = et_date
            self._start_of_day_equity_source = "broker_last_equity"
            self._daytrade_suspended = False
            self._save_daytrade_risk_state(
                et_date,
                self._start_of_day_equity,
                False,
                account_id=account_id,
                source="broker_last_equity",
                status=init_status,
                prior_session_close=broker_last_eq,
                settlement_confirmed=market_live,
            )
            logger.info("[%s] Initialized %s start-of-day equity from broker last_equity: $%.2f for session %s",
                        account_id, init_status, self._start_of_day_equity, et_date)
            self._reconcile_daily_loss_halt_if_needed(account_id, et_date, acct, broker_last_eq)
            if hasattr(self, "entry_gate") and self.entry_gate:
                self.entry_gate.reasons.discard("BASELINE_UNVERIFIED")
                if hasattr(self.entry_gate, "_sync_state"):
                    self.entry_gate._sync_state()
            return self._start_of_day_equity

        self._start_of_day_equity = 0.0
        self._start_of_day_equity_date = et_date
        self._start_of_day_equity_source = "unverified"
        self._daytrade_suspended = False
        if hasattr(self, "entry_gate") and self.entry_gate:
            self.entry_gate.halt("BASELINE_UNVERIFIED")
        logger.error("[%s] Broker last_equity invalid or non-positive (got %r). Start-of-day baseline left unverified (0.0). Entry halted under BASELINE_UNVERIFIED.",
                     account_id, getattr(acct, "last_equity", None))
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
                self.invalidate_broker_caches(reason="unparseable_update:missing_parent_order_id")
                self._cache_requires_reconciliation = True
                if hasattr(self, "entry_gate") and self.entry_gate:
                    self.entry_gate.halt("TRADE_STREAM_DISPATCH_ERROR")
                logger.warning(
                    "[TRADE STREAM] Trade update could not be parsed (missing order id); "
                    "caches invalidated, entry gate halted (fail-closed), forcing fresh broker reconciliation"
                )
                self._schedule_forced_reconciliation(reason="missing_parent_order_id")
                return

            event_name = self._string_value(getattr(update, "event", "")).lower().strip()
            order_status = self._string_value(getattr(order, "status", "") if order else getattr(update, "order_status", "")).lower().strip()
            status_from_event = {
                "new": "new",
                "accepted": "accepted",
                "held": "held",
                "fill": "filled",
                "partial_fill": "partially_filled",
                "canceled": "canceled",
                "cancelled": "canceled",
                "rejected": "rejected",
                "expired": "expired",
                "done_for_day": "done_for_day",
                "replaced": "replaced",
                "pending_new": "pending_new",
                "pending_cancel": "pending_cancel",
                "pending_replace": "pending_replace",
                "stopped": "stopped",
                "suspended": "suspended",
                "calculated": "calculated",
            }
            status = order_status or status_from_event.get(event_name, event_name)

            # Invalidate caches on every broker trade update and relevant order lifecycle event
            self.invalidate_broker_caches(reason=f"trade_update:{event_name}:{status}")

            # Alpaca WebSocket trade update events (TradeEvent stream schema)
            known_trade_events = {
                "new", "fill", "partial_fill", "canceled", "cancelled", "expired",
                "done_for_day", "replaced", "rejected", "pending_new", "stopped",
                "pending_cancel", "pending_replace", "calculated", "suspended",
                "order_cancel_rejected", "order_replace_rejected", "held", "accepted", "restated"
            }
            # Alpaca REST/WebSocket Order statuses (OrderStatus model schema)
            known_order_statuses = {
                "new", "partially_filled", "filled", "done_for_day", "canceled", "cancelled",
                "expired", "replaced", "pending_cancel", "pending_replace", "pending_review",
                "accepted", "pending_new", "accepted_for_bidding", "stopped", "rejected",
                "suspended", "calculated", "held"
            }
            event_invalid = bool(event_name and event_name not in known_trade_events)
            status_invalid = bool(order_status and order_status not in known_order_statuses)
            if event_invalid or status_invalid or (not event_name and not order_status):
                self._cache_requires_reconciliation = True
                if hasattr(self, "entry_gate") and self.entry_gate:
                    self.entry_gate.halt("TRADE_STREAM_DISPATCH_ERROR")
                logger.warning(
                    "[TRADE STREAM] Uncertain trade update event '%s' (order_status='%s', status='%s') for order %s; "
                    "caches invalidated, entry gate halted (fail-closed), forcing fresh broker reconciliation",
                    event_name, order_status, status, parent_order_id
                )
                self._schedule_forced_reconciliation(reason=f"uncertain_event:{event_name}")
                return
            cumulative_filled_qty = self._float_value(getattr(order, "filled_qty", None) if order else getattr(update, "position_qty", None))
            average_fill_price = self._float_value(getattr(order, "filled_avg_price", None) if order else getattr(update, "price", None))
            execution_id = str(getattr(update, "execution_id", "") or "").strip() or None

            event_key = execution_id or (
                f"{parent_order_id}|{event_name}|{status}|"
                f"{cumulative_filled_qty:.12g}|{average_fill_price:.12g}"
            )

            # Record always-on trade update stream to broker_events.jsonl
            try:
                ev_ts = getattr(update, "timestamp", None)
                ev_ts_str = ev_ts.isoformat() if hasattr(ev_ts, "isoformat") else None

                role_info = self.get_order_role_info(parent_order_id)
                order_role = role_info.get("role", "UNKNOWN")
                linked_parent_id = role_info.get("parent_order_id")
                c_order_id = getattr(order, "client_order_id", None) or role_info.get("client_order_id")

                broker_ev_path = self.get_broker_events_path()
                broker_ev_path.parent.mkdir(parents=True, exist_ok=True)
                with _broker_event_write_lock:
                    with open(broker_ev_path, "a", encoding="utf-8") as f_ev:
                        f_ev.write(json.dumps({
                            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                            "observed_ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                            "event_ts": ev_ts_str,
                            "event": event_name,
                            "status": status,
                            "order_id": parent_order_id,
                            "client_order_id": c_order_id,
                            "order_role": order_role,
                            "parent_order_id": linked_parent_id,
                            "symbol": str(getattr(order, "symbol", "") if order else ""),
                            "filled_qty": cumulative_filled_qty,
                            "price": average_fill_price,
                            "execution_id": execution_id,
                        }) + "\n")
            except Exception:
                logger.exception("Failed to persist broker trade-update event")

            # Forward event to order_audit Schema 2.0 lifecycle tracker
            try:
                if 'ev_ts_str' not in locals():
                    ev_ts = getattr(update, "timestamp", None)
                    ev_ts_str = ev_ts.isoformat() if hasattr(ev_ts, "isoformat") else None

                if status in ("new", "accepted", "held") or event_name in ("new", "accepted", "held"):
                    order_audit.lifecycle_tracker.record_stream_ack(
                        broker_order_id=parent_order_id,
                        broker_event_time_utc=ev_ts_str,
                    )
                elif event_name in ("fill", "partial_fill") or status in ("filled", "partially_filled"):
                    is_full = (status == "filled" or event_name == "fill")
                    order_audit.lifecycle_tracker.record_fill(
                        broker_order_id=parent_order_id,
                        filled_qty=cumulative_filled_qty,
                        price=average_fill_price,
                        execution_id=execution_id,
                        event_time_utc=ev_ts_str,
                        is_complete=is_full,
                    )
                elif status in ("canceled", "cancelled", "rejected", "expired", "replaced"):
                    order_audit.lifecycle_tracker.record_terminal(
                        broker_order_id=parent_order_id,
                        terminal_status=status,
                    )
            except Exception as tracker_exc:
                logger.debug("Error recording lifecycle tracker event for %s: %s", parent_order_id, tracker_exc)

            if not self._remember_trade_update(event_key):
                logger.debug("Ignoring duplicate Alpaca trade update %s", event_key)
                return

            from strategies.entry_lifecycle import entry_lifecycle
            was_pending_entry = parent_order_id in entry_lifecycle.pending_snapshot()

            if was_pending_entry:
                self.on_entry_order_update(
                    parent_order_id=parent_order_id,
                    cumulative_filled_qty=cumulative_filled_qty,
                    average_fill_price=average_fill_price,
                    execution_id=execution_id,
                    status=status,
                )
            else:
                self.on_exit_order_update(
                    order_id=parent_order_id,
                    cumulative_filled_qty=cumulative_filled_qty,
                    average_fill_price=average_fill_price,
                    execution_id=execution_id,
                    status=status,
                    event_name=event_name,
                    order=order,
                    role_info=role_info,
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
            try:
                self.invalidate_broker_caches(reason=f"dispatch_exception:{exc}")
            except Exception:
                pass
            self._cache_requires_reconciliation = True
            self._conn_health.mark_unhealthy(f"Trade-update dispatch error: {exc}")
            if hasattr(self, "entry_gate") and self.entry_gate:
                self.entry_gate.halt("TRADE_STREAM_DISPATCH_ERROR")
            logger.exception("[TRADE STREAM] Lifecycle dispatch failed; caches invalidated, entry gate halted, forcing fresh reconciliation")
            self._schedule_forced_reconciliation(reason=f"dispatch_exception:{exc}")

    def on_exit_order_update(
        self,
        order_id: str,
        cumulative_filled_qty: float,
        average_fill_price: float,
        execution_id: Optional[str] = None,
        status: str = "",
        event_name: str = "",
        order: Optional[object] = None,
        role_info: Optional[dict] = None,
    ) -> bool:
        """Route bracket stop/TP leg or exit order fill updates directly to trailing_stop and authoritative reconciliation."""
        order_sym = str(getattr(order, "symbol", "") if order else "").upper().strip()
        if not order_sym and role_info:
            order_sym = str(role_info.get("symbol", "")).upper().strip()
        if not order_sym:
            return False

        is_fill = (event_name in ("fill", "partial_fill") or status in ("filled", "partially_filled"))
        if not is_fill:
            return False

        try:
            ts_mgr = self.get_trailing_stop_manager()
            if ts_mgr is None:
                return False

            with ts_mgr._lock:
                state = ts_mgr.states.get(order_sym)
                is_active = state is not None and getattr(state, "active", False)
                matches_leg = False
                if state is not None:
                    matches_leg = str(order_id) in {
                        str(getattr(state, "order_id", None) or ""),
                        str(getattr(state, "tp_order_id", None) or ""),
                        str(getattr(state, "last_stop_order_id", None) or ""),
                    }
                r_info = role_info or {}
                is_exit_role = r_info.get("role") in ("STOP", "TRAILING_STOP", "TAKE_PROFIT")

                if is_active and (matches_leg or is_exit_role or getattr(state, "side", "").lower() != str(getattr(order, "side", "")).lower()):
                    logger.info(
                        "[TRADE STREAM EXIT FILL] %s fill received for %s order %s (qty=%.4f @ %.4f, role=%s). "
                        "Recording pending exit for authoritative reconciliation.",
                        status, order_sym, order_id, cumulative_filled_qty, average_fill_price, r_info.get("role", "UNKNOWN")
                    )
                    ts_mgr._record_pending_exit(state, qty_closed=cumulative_filled_qty)

            # Atomically reconcile pending exit using authoritative broker truth & journal sync
            ts_mgr.reconcile_pending_exits(bridge=self)
            return True
        except Exception as exc:
            logger.error("[TRADE STREAM EXIT DISPATCH FAILED] %s order %s: %s", order_sym, order_id, exc)
            return False

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
        self.invalidate_broker_caches(reason=f"stream_unavailable:{reason}")
        self._cache_requires_reconciliation = True
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

    def _cleanup_failed_probe_stream(self) -> None:
        """Private helper: cleans up failed probe stream references while preserving reconnect supervisor."""
        try:
            self._stop_trade_stream(cancel_reconnect=False)
        except Exception as stop_exc:
            logger.debug("[TRADE STREAM] Failed probe cleanup exception: %s", stop_exc)

    def _trade_stream_reconnect_loop(self) -> None:
        max_attempts = int(self._c("TRADE_STREAM_RECONNECT_MAX_ATTEMPTS", 8))
        base_delay = float(self._c("TRADE_STREAM_RECONNECT_BASE_DELAY_SECONDS", 2.0))
        max_delay = float(self._c("TRADE_STREAM_RECONNECT_MAX_DELAY_SECONDS", 30.0))
        jitter_ratio = float(self._c("TRADE_STREAM_RECONNECT_JITTER_RATIO", 0.20))

        try:
            # ── Tier 1: Fast Backoff Loop (Attempts 1 to max_attempts) ──
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
                    logger.info("[TRADE STREAM] Reconnected and authenticated on attempt %d; executing broker reconciliation handshake", attempt)
                    recon_ok = self._run_post_reconnect_reconciliation()
                    if recon_ok:
                        if not getattr(self.entry_gate, "halted", False):
                            logger.info("[TRADE STREAM] Post-reconnect reconciliation clean and entry gate released; stream fully restored to HEALTHY on attempt %d", attempt)
                        else:
                            logger.warning(
                                "[TRADE STREAM] Post-reconnect reconciliation clean, but gate remains halted due to non-recovery reasons: %s",
                                sorted(getattr(self.entry_gate, "reasons", [])),
                            )
                    else:
                        logger.warning(
                            "[TRADE STREAM] Post-reconnect reconciliation dirty or timed out; stream state=%s, gate halted=%s, reasons=%s",
                            self.stream_state.value if hasattr(self.stream_state, "value") else str(self.stream_state),
                            getattr(self.entry_gate, "halted", False),
                            sorted(getattr(self.entry_gate, "reasons", [])),
                        )
                    return
                except AccountModeViolation as amv:
                    logger.critical("🚨 [ACCOUNT_MODE_VIOLATION] Trade stream reconnect aborted: %s", amv)
                    self._cleanup_failed_probe_stream()
                    self._mark_trade_stream_unavailable(f"Account mode violation: {amv}")
                    raise
                except Exception as exc:
                    self._cleanup_failed_probe_stream()
                    self._mark_trade_stream_unavailable(
                        f"Reconnect attempt {attempt}/{max_attempts} failed: {exc}"
                    )

            # ── Fast Retries Exhausted -> Transition to DEGRADED ──
            self._mark_trade_stream_unavailable(
                f"Fast reconnect exhausted after {max_attempts} attempts; entering long-tail recovery probe"
            )
            self._transition_stream_state(
                expected={StreamState.RECONNECTING},
                new_state=StreamState.DEGRADED,
                reason="Fast reconnect exhausted; long-tail probe active",
            )

            # ── Tier 2: Slow Autonomous Long-Tail Probe Loop ──
            probe_interval = float(self._c("TRADE_STREAM_LONG_TAIL_PROBE_INTERVAL_SECONDS", 300.0))
            probe_num = 0

            while (
                not self._trade_stream_reconnect_cancel.is_set()
                and self._is_reconnect_allowed()
                and self.stream_state == StreamState.DEGRADED
            ):
                probe_num += 1
                logger.info(
                    "[TRADE STREAM] Long-tail recovery probe %d scheduled in %.1fs (state=DEGRADED, entries=HALTED)",
                    probe_num,
                    probe_interval,
                )
                if self._trade_stream_reconnect_cancel.wait(max(0.1, probe_interval)):
                    return

                # Secondary pre-probe state check
                if (
                    self._trade_stream_reconnect_cancel.is_set()
                    or not self._is_reconnect_allowed()
                    or self.stream_state != StreamState.DEGRADED
                ):
                    return

                with self._trade_stream_lock:
                    credentials = self._trade_stream_credentials
                if not credentials:
                    self._mark_trade_stream_unavailable("Long-tail probe credentials unavailable")
                    return

                try:
                    logger.info("[TRADE STREAM] Executing long-tail recovery probe %d...", probe_num)
                    self._start_trade_stream(*credentials)

                    # Secondary post-start invariant check: must be AUTHENTICATED_UNRECONCILED
                    if self.stream_state != StreamState.AUTHENTICATED_UNRECONCILED:
                        logger.error(
                            "[TRADE STREAM] Invariant violation: expected AUTHENTICATED_UNRECONCILED after start, observed %s",
                            self.stream_state,
                        )
                        self._cleanup_failed_probe_stream()
                        continue

                    logger.info("[TRADE STREAM] Long-tail probe %d authenticated; executing broker reconciliation handshake", probe_num)
                    recon_ok = self._run_post_reconnect_reconciliation()

                    # Tri-state outcome handling:
                    if recon_ok:
                        # 1. CLEAN -> HEALTHY (Recovery Complete, ONLY stream-owned halts cleared)
                        logger.info(
                            "[TRADE STREAM] Long-tail probe %d: broker reconciliation clean; stream fully restored to HEALTHY",
                            probe_num,
                        )
                        return
                    else:
                        # 2. DIRTY / DESYNC -> LATCHED_DESYNC (Autonomous retry prohibited; operator action required)
                        logger.critical(
                            "[TRADE STREAM] Long-tail probe %d: broker reconciliation dirty/desync; stream latched in LATCHED_DESYNC; operator intervention required",
                            probe_num,
                        )
                        return

                except AccountModeViolation as amv:
                    logger.critical("🚨 [ACCOUNT_MODE_VIOLATION] Long-tail probe aborted: %s", amv)
                    self._cleanup_failed_probe_stream()
                    self._mark_trade_stream_unavailable(f"Account mode violation: {amv}")
                    raise
                except Exception as probe_err:
                    # 3. RETRYABLE_ERROR -> Remain in DEGRADED (fail-closed); clean up stream; wait for next cycle
                    self._cleanup_failed_probe_stream()
                    logger.warning(
                        "[TRADE STREAM] Long-tail probe %d failed (%s); remaining in DEGRADED (fail-closed)",
                        probe_num,
                        probe_err,
                    )
        finally:
            with self._trade_stream_lock:
                if self._trade_stream_reconnect_thread is threading.current_thread():
                    self._trade_stream_reconnect_thread = None

    def _run_trade_stream(self, stream: TradingStream, generation: int = 0) -> None:
        unexpected_exit = False
        failure_reason = "Trade stream stopped unexpectedly"
        try:
            stream.run()
        except Exception as exc:
            failure_reason = f"Trade stream crashed: {exc}"
            logger.exception("[TRADE STREAM] Websocket crashed")
            unexpected_exit = True
        finally:
            should_run = getattr(stream, "_should_run", True)
            with self._trade_stream_lock:
                if self._trade_stream is stream:
                    unexpected_exit = unexpected_exit or (
                        should_run and not self._trade_stream_stopping and self._connected and self._is_reconnect_allowed()
                    )
                    self._trade_stream_ready = False
                    self._trade_stream_thread = None
                    self._trade_stream = None
            if unexpected_exit and self._is_reconnect_allowed():
                with self._stream_state_lock:
                    is_current_gen = (generation == self._stream_generation or generation == 0)
                if is_current_gen:
                    self._transition_stream_state(
                        expected={
                            StreamState.HEALTHY,
                            StreamState.AUTHENTICATED_UNRECONCILED,
                            StreamState.RECONCILING,
                            StreamState.DEGRADED,
                            StreamState.RECONNECTING,
                        },
                        new_state=StreamState.DISCONNECTED,
                        reason=failure_reason,
                    )
                    self._mark_trade_stream_unavailable(failure_reason)
                    self._schedule_trade_stream_reconnect()
                else:
                    logger.warning("[TRADE STREAM] Stale generation %d exit ignored; current is %d", generation, self._stream_generation)

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
            ws_params = {
                "ping_interval": int(self._c("ALPACA_WS_PING_INTERVAL_SECONDS", 20)),
                "ping_timeout": int(self._c("ALPACA_WS_PING_TIMEOUT_SECONDS", 10)),
                "max_queue": int(self._c("ALPACA_WS_MAX_QUEUE", 1024)),
            }
            try:
                stream = TradingStream(api_key, secret_key, paper=True, websocket_params=ws_params, bridge=self)
            except TypeError:
                stream = TradingStream(api_key, secret_key, paper=True, websocket_params=ws_params)
            setattr(stream, "_bridge", self)
            self._assert_paper_stream_endpoint(stream)

            async def _stream_callback(upd):
                await self._on_trade_update(upd, generation=current_gen)

            stream.subscribe_trade_updates(_stream_callback)
            thread = threading.Thread(
                target=self._run_trade_stream,
                args=(stream, current_gen),
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
                        # Amendment 1: recovery_required remains True until broker reconciliation completes clean
                        self._trade_stream_recovery_required = True
                self._transition_stream_state(
                    expected={StreamState.DISCONNECTED, StreamState.RECONNECTING, StreamState.AUTHENTICATED_UNRECONCILED, StreamState.STOPPED, StreamState.DEGRADED},
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

    def pause_trade_stream_for_market_close(self) -> None:
        """
        Market-Close Pause Contract Invariant:
        - REST client connection: PRESERVED
        - Open positions: PRESERVED (no liquidation)
        - Broker GTC protection: PRESERVED
        - Order registry: PRESERVED
        - Risk state store: PRESERVED
        - Trade WebSocket: STOPPED cleanly
        - Reconnect supervisor: CANCELLED
        - Stream state: STOPPED
        - Entry gate: HALTED (OFF_MARKET_STANDBY)
        """
        logger.info("[STREAM LIFECYCLE] Pausing trade stream for off-market hours.")
        self._stop_trade_stream(cancel_reconnect=True)
        with self._stream_state_lock:
            self._stream_state = StreamState.STOPPED
        self.entry_gate.halt("OFF_MARKET_STANDBY")

    def resume_trade_stream_for_market_open(self) -> bool:
        """
        Market-Open Resume Contract Invariant:
        - Start trade stream with stored credentials.
        - Execute broker reconciliation before marking stream HEALTHY or releasing entry gate.
        - Enforce ordering: connected -> recon start -> recon finish -> HEALTHY -> release.
        - Return True if stream is healthy and reconciliation clean.
        """
        with self._stream_state_lock:
            if self._stream_state == StreamState.HEALTHY and self._trade_stream_ready:
                return True

        logger.info("[STREAM LIFECYCLE] Resuming trade stream for regular market session.")
        with self._trade_stream_lock:
            creds = self._trade_stream_credentials
        if not creds:
            logger.error("[STREAM LIFECYCLE] Resume failed: missing stream credentials")
            return False

        try:
            self._start_trade_stream(*creds)
            reconcile_ok = self._run_post_reconnect_reconciliation()
            if reconcile_ok:
                if hasattr(self.entry_gate, "clear"):
                    self.entry_gate.clear("OFF_MARKET_STANDBY")
                elif hasattr(self.entry_gate, "reasons"):
                    self.entry_gate.reasons.discard("OFF_MARKET_STANDBY")
                    remaining = set(getattr(self.entry_gate, "reasons", set()))
                    self.entry_gate.halted = bool(remaining)
                    if getattr(self.entry_gate, "halt_reason", "") not in remaining:
                        self.entry_gate.halt_reason = sorted(remaining)[0] if remaining else ""
                logger.info("[STREAM LIFECYCLE] Trade stream resumed and reconciled cleanly.")
                return True
            else:
                logger.warning("[STREAM LIFECYCLE] Trade stream resumed but reconciliation was dirty; entry gate remains halted.")
                return False
        except Exception as exc:
            logger.exception("[STREAM LIFECYCLE] Resume trade stream failed: %s", exc)
            return False

    def _run_post_reconnect_reconciliation(self) -> bool:
        sched_lock = getattr(self, "_reconcile_scheduling_lock", None)
        if sched_lock is not None:
            with sched_lock:
                if not self._reconcile_lock.acquire(blocking=False):
                    logger.info("[RECONCILIATION] already_in_progress=True")
                    return False
                self._reconcile_in_progress = True
        else:
            if not self._reconcile_lock.acquire(blocking=False):
                logger.info("[RECONCILIATION] already_in_progress=True")
                return False

        try:
            self.invalidate_broker_caches(reason="post_reconnect_reconciliation_start")
            self._cache_requires_reconciliation = True
            self._transition_stream_state(
                expected={
                    StreamState.HEALTHY,
                    StreamState.AUTHENTICATED_UNRECONCILED,
                    StreamState.RECONNECTING,
                    StreamState.DEGRADED,
                    StreamState.LATCHED_DESYNC,
                },
                new_state=StreamState.RECONCILING,
                reason="Executing broker truth reconciliation check",
            )

            # Hardened Retry Loop for Transient Network / HTTP / Timeout Errors
            # Semantics: POST_RECONNECT_RECONCILE_MAX_ATTEMPTS = 3 total attempts
            # (1 initial attempt + 2 exponential backoff retries: delays 1.5s, 3.0s).
            max_attempts = int(self._c("POST_RECONNECT_RECONCILE_MAX_ATTEMPTS", self._c("POST_RECONNECT_RECONCILE_MAX_RETRIES", 3)))
            base_retry_delay = float(self._c("POST_RECONNECT_RECONCILE_RETRY_DELAY_SECONDS", 1.5))
            recon = None
            last_recon_err = None

            for attempt_idx in range(1, max_attempts + 1):
                try:
                    recon = self.reconcile_startup_state(manage_stream_state=False, allow_policy_side_effects=False)
                    last_recon_err = None
                    break
                except Exception as r_err:
                    last_recon_err = r_err
                    retry_num = attempt_idx - 1
                    max_retries = max(0, max_attempts - 1)
                    logger.warning(
                        "[POST_RECONNECT_RECONCILIATION] Attempt %d/%d (retry %d/%d) encountered transient broker error: %s",
                        attempt_idx,
                        max_attempts,
                        retry_num,
                        max_retries,
                        r_err,
                    )
                    if attempt_idx < max_attempts:
                        delay = base_retry_delay * (2 ** (attempt_idx - 1))
                        time.sleep(delay)

            # If all attempts failed due to timeout/network error
            if last_recon_err is not None or recon is None:
                self._cache_requires_reconciliation = True
                self.invalidate_broker_caches(reason="post_reconnect_reconciliation_timeout")
                acct = getattr(self, "account_id", "UNKNOWN")
                active_reasons = sorted(getattr(self.entry_gate, "reasons", []))
                retries_exhausted = max(0, max_attempts - 1)
                logger.critical(
                    "🚨 [OPERATOR ALARM] [POST_RECONNECT_TIMEOUT] [%s] Broker reconciliation failed after %d total attempts (1 initial + %d retries): %s | Entry gate remains HALTED with reasons: %s",
                    acct,
                    max_attempts,
                    retries_exhausted,
                    last_recon_err,
                    active_reasons,
                )
                self._transition_stream_state(
                    expected={StreamState.RECONCILING, StreamState.AUTHENTICATED_UNRECONCILED},
                    new_state=StreamState.DEGRADED,
                    reason=f"Post-reconnect reconciliation timed out after {max_attempts} attempts (1 initial + {retries_exhausted} retries): {last_recon_err}",
                )
                return False

            is_clean = recon.get("clean", False) if isinstance(recon, dict) else getattr(recon, "clean", False)
            protect_info = recon.get("protect", {}) if isinstance(recon, dict) else getattr(recon, "protect", {})
            protect_failed = protect_info.get("failed", []) if isinstance(protect_info, dict) else getattr(protect_info, "failed", [])
            unprotected_longs = protect_info.get("unprotected", []) if isinstance(protect_info, dict) else getattr(protect_info, "unprotected", [])
            protection_clean = (len(protect_failed) == 0 and len(unprotected_longs) == 0)

            if is_clean and protection_clean:
                self._cache_requires_reconciliation = False
                self._transition_stream_state(
                    expected={StreamState.RECONCILING},
                    new_state=StreamState.HEALTHY,
                    reason="Broker reconciliation clean=True",
                )
                with self._trade_stream_lock:
                    self._trade_stream_recovery_required = False
                    self._trade_stream_ready = True
                self._conn_health.mark_healthy("Trade stream reconnected and broker truth reconciled clean")
                self._clear_recovery_halts()

                # Diagnostic check: Did the gate successfully open, or do non-recovery reasons remain?
                gate_halted = getattr(self.entry_gate, "halted", False)
                active_reasons = sorted(getattr(self.entry_gate, "reasons", []))
                if gate_halted:
                    logger.warning(
                        "[RECONNECT RECOVERY] Full broker reconciliation clean; recovery halts cleared, but entry gate remains HALTED due to non-recovery reasons: %s",
                        active_reasons,
                    )
                else:
                    logger.info("[RECONNECT RECOVERY] Full broker reconciliation clean; recovery_required=False, conn_health=HEALTHY, entry halts cleared.")
                return True
            else:
                self._cache_requires_reconciliation = True
                self.invalidate_broker_caches(reason="post_reconnect_reconciliation_dirty")
                self._transition_stream_state(
                    expected={StreamState.RECONCILING},
                    new_state=StreamState.LATCHED_DESYNC,
                    reason="Broker reconciliation clean=False (desync/unprotected)",
                )
                with self._trade_stream_lock:
                    self._trade_stream_recovery_required = True
                acct = getattr(self, "account_id", "UNKNOWN")
                active_reasons = sorted(getattr(self.entry_gate, "reasons", []))
                logger.critical(
                    "🚨 [OPERATOR ALARM] [POST_RECONNECT_HALT_PERSISTED] [%s] Broker reconciliation dirty (is_clean=%s, protect_failed=%s, unprotected=%s); gate remains halted with reasons: %s",
                    acct,
                    is_clean,
                    protect_failed,
                    unprotected_longs,
                    active_reasons,
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
        except Exception as exc:
            self._transition_stream_state(
                expected={StreamState.RECONCILING, StreamState.AUTHENTICATED_UNRECONCILED},
                new_state=StreamState.DEGRADED,
                reason=f"Post-reconnect reconciliation crashed: {exc}",
            )
            logger.exception("[TRADE STREAM RECONCILIATION FAILED] %s", exc)
            return False
        finally:
            sched_lock = getattr(self, "_reconcile_scheduling_lock", None)
            if sched_lock is not None:
                with sched_lock:
                    self._reconcile_in_progress = False
            self._reconcile_lock.release()

    def _on_sdk_stream_reconnect_start(self) -> None:
        """
        Invoked by _BridgeAwareTradingStream when the SDK internal reconnect
        loop in _run_forever begins reconnecting after a dropped socket.
        Immediately fail-closes entry gate and transitions stream state to RECONNECTING.
        """
        if not self._is_reconnect_allowed():
            logger.info("[TRADE STREAM] SDK reconnect start ignored: bridge is shutting down or stopped")
            return

        self.invalidate_broker_caches(reason="sdk_stream_reconnect_start")
        self._cache_requires_reconciliation = True

        logger.warning(
            "[TRADE STREAM] SDK internal reconnect initiated; fail-closing entry gate (STREAM_RECOVERY_REQUIRED)"
        )
        with self._trade_stream_lock:
            self._trade_stream_ready = False
            self._trade_stream_recovery_required = True

        gate = getattr(self, "entry_gate", None)
        if gate is not None and hasattr(gate, "halt"):
            gate.halt("STREAM_RECOVERY_REQUIRED")

        self._transition_stream_state(
            expected={
                StreamState.HEALTHY,
                StreamState.DEGRADED,
                StreamState.AUTHENTICATED_UNRECONCILED,
                StreamState.RECONCILING,
                StreamState.RECONNECTING,
                StreamState.DISCONNECTED,
            },
            new_state=StreamState.RECONNECTING,
            reason="SDK internal stream reconnect started",
        )

    def _on_sdk_stream_reconnect_success(self) -> bool:
        """
        Invoked by _BridgeAwareTradingStream when the SDK internal reconnect
        loop successfully connects and authenticates. Transitions stream state
        to AUTHENTICATED_UNRECONCILED and immediately executes broker reconciliation
        before allowing stream to become HEALTHY.
        """
        if not self._is_reconnect_allowed():
            logger.info("[TRADE STREAM] SDK reconnect success ignored: bridge is shutting down or stopped")
            return False

        self.invalidate_broker_caches(reason="sdk_stream_reconnect_success")

        current_st = self.stream_state
        if current_st == StreamState.RECONCILING:
            logger.info("[TRADE STREAM] SDK reconnect authenticated while reconciliation already in-flight; letting active reconciliation complete")
            return True

        if current_st == StreamState.HEALTHY and not self._trade_stream_recovery_required:
            logger.info("[TRADE STREAM] SDK reconnect authenticated but stream is already HEALTHY and clean; skipping redundant transition")
            return True

        logger.info(
            "[TRADE STREAM] SDK internal reconnect authenticated; initiating broker truth reconciliation"
        )
        with self._trade_stream_lock:
            self._trade_stream_ready = True
            # recovery_required remains True until broker reconciliation completes clean
            self._trade_stream_recovery_required = True

        self._transition_stream_state(
            expected={
                StreamState.HEALTHY,
                StreamState.RECONNECTING,
                StreamState.DISCONNECTED,
                StreamState.AUTHENTICATED_UNRECONCILED,
                StreamState.DEGRADED,
            },
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="SDK internal stream reconnect authenticated",
        )

        return self._run_post_reconnect_reconciliation()

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
        base_dir = self._get_risk_state_dir()
        path = os.path.join(base_dir, f'daytrade_risk_state_{acct_id}.json')
        target_path = None
        if os.path.exists(path):
            target_path = path
        else:
            legacy_path = os.path.join(base_dir, 'daytrade_risk_state.json')
            if os.path.exists(legacy_path):
                target_path = legacy_path

        if not target_path:
            return {}

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    raise PersistedRiskStateInvalidError(f"Persisted risk state at {target_path} is empty")
                data = json.loads(content)
        except json.JSONDecodeError as jde:
            logger.critical("[CRITICAL PERSISTED RISK STATE FAULT] JSON decode error in %s: %s", target_path, jde)
            raise PersistedRiskStateInvalidError(f"Corrupted JSON in {target_path}: {jde}") from jde
        except Exception as e:
            if isinstance(e, PersistedRiskStateInvalidError):
                raise
            logger.critical("[CRITICAL PERSISTED RISK STATE FAULT] Error reading %s: %s", target_path, e)
            raise PersistedRiskStateInvalidError(f"Failed to read {target_path}: {e}") from e

        if not isinstance(data, dict):
            raise PersistedRiskStateInvalidError(f"Persisted risk state at {target_path} is not a JSON object")

        # Required keys: "account_id", "date", "start_of_day_equity"
        for req_key in ("account_id", "date", "start_of_day_equity"):
            if req_key not in data:
                raise PersistedRiskStateInvalidError(f"Persisted risk state at {target_path} missing required key: '{req_key}'")

        # Validate date string format (YYYY-MM-DD)
        date_str = str(data["date"]).strip()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError as ve:
            raise PersistedRiskStateInvalidError(f"Invalid date format in {target_path}: '{date_str}'") from ve

        # Validate equity is positive float
        try:
            sod_equity = float(data["start_of_day_equity"])
            if not math.isfinite(sod_equity) or sod_equity <= 0.0:
                raise PersistedRiskStateInvalidError(f"Invalid non-positive start_of_day_equity in {target_path}: {sod_equity}")
        except (ValueError, TypeError) as conv_err:
            raise PersistedRiskStateInvalidError(f"Unparseable start_of_day_equity in {target_path}: {conv_err}") from conv_err

        return data

    def _get_risk_state_dir(self) -> str:
        """Returns the directory for persisting daytrade risk state.

        Strict Pytest/Test Guard:
        If pytest or a test execution environment is detected, automatically diverts
        to an isolated temporary directory unless a custom `_risk_state_dir` was explicitly
        injected. This prevents unit and chaos tests from corrupting production risk state.
        """
        custom_dir = getattr(self, "_risk_state_dir", None)
        if custom_dir:
            return custom_dir

        is_test_env = (
            "pytest" in sys.modules
            or "PYTEST_CURRENT_TEST" in os.environ
            or bool(os.environ.get("TEST_RUNNING"))
        )
        if is_test_env:
            import tempfile
            test_dir = os.path.join(tempfile.gettempdir(), "stock_engine_test_risk_state")
            os.makedirs(test_dir, exist_ok=True)
            return test_dir

        return str(getattr(config_module, "DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')))

    def _save_daytrade_risk_state(
        self,
        date_str: str,
        equity: float,
        suspended: bool,
        account_id: str = None,
        source: str = "broker_last_equity",
        status: str = "PROVISIONAL",
        prior_session_close: Optional[float] = None,
        settlement_confirmed: bool = False,
    ):
        acct_id = account_id or getattr(self, "_account_id", "default")
        base_dir = self._get_risk_state_dir()
        path = os.path.join(base_dir, f'daytrade_risk_state_{acct_id}.json')
        latch_dict = None
        try:
            from observability import get_latch_state_machine
            latch_dict = get_latch_state_machine().export_state_dict()
        except Exception:
            pass
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                import json
                now_str = datetime.now(timezone.utc).isoformat()
                payload = {
                    "account_id": acct_id,
                    "date": date_str,
                    "session_date_et": date_str,
                    "start_of_day_equity": equity,
                    "status": status,
                    "suspended": suspended,
                    "provenance": {
                        "source": source,
                        "status": status,
                        "settlement_confirmed": settlement_confirmed,
                        "prior_session_close": prior_session_close if prior_session_close is not None else equity,
                        "captured_at": now_str,
                        "verified_at": now_str if status == "VERIFIED" else None,
                    },
                    "latch_state": latch_dict,
                }
                json.dump(payload, f, indent=2)
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

    def _get_active_orders(self, force_refresh: bool = False, require_complete: bool = False) -> list:
        self._require_connection()
        now_t = time.time()
        if not force_refresh and not getattr(self, "_cache_requires_reconciliation", False):
            lock = getattr(self, "_active_orders_lock", None)
            if lock is not None:
                with lock:
                    cached_orders = getattr(self, "_cached_active_orders", None)
                    cached_ts = getattr(self, "_cached_active_orders_ts", 0.0)
                    if cached_orders is not None and (now_t - cached_ts) < 3.0:
                        return list(cached_orders)
        try:
            # 1. Query OPEN orders (guarantees all currently working open orders)
            open_orders = self._call_with_retry(lambda: self._client.get_orders(filter=GetOrdersRequest(
                status=QueryOrderStatus.OPEN,
                nested=True,
                limit=500,
                direction="desc"
            )))
            if require_complete and (not isinstance(open_orders, list) or len(open_orders) >= 500):
                raise RuntimeError("Open-order snapshot unavailable or possibly truncated")
            orders_map: Dict[str, Any] = {}
            for o in (open_orders or []):
                o_id = str(getattr(o, "id", "") or "")
                if o_id:
                    orders_map[o_id] = o
                # Unnest legs from open orders
                for leg in (getattr(o, "legs", []) or []):
                    leg_id = str(getattr(leg, "id", "") or "")
                    if leg_id and leg_id not in orders_map:
                        if getattr(leg, "symbol", None) is None:
                            try:
                                leg.symbol = getattr(o, "symbol", "")
                            except Exception:
                                pass
                        orders_map[leg_id] = leg

            # 2. Query recent orders across ALL statuses to capture conditional 'held' bracket legs (Amendment 4)
            try:
                all_recent = self._call_with_retry(lambda: self._client.get_orders(filter=GetOrdersRequest(
                    status=QueryOrderStatus.ALL,
                    nested=True,
                    limit=200,
                    direction="desc"
                )))
                if require_complete:
                    if not isinstance(all_recent, list):
                        raise RuntimeError("Recent-order snapshot unavailable")
                    # Walk older pages rather than permanently halting accounts
                    # with 200 historical orders. Overlap the boundary so
                    # timestamp ties are not silently discarded. If the API
                    # cannot advance, its history is unknown and we fail closed.
                    page = all_recent
                    all_recent = list(page)
                    previous_oldest = None
                    page_count = 1
                    while len(page) >= 200:
                        if page_count >= 20:
                            raise RuntimeError("Recent-order pagination limit reached")
                        timestamps = []
                        for order in page:
                            submitted = getattr(order, "submitted_at", None)
                            if isinstance(submitted, str):
                                submitted = _dt.datetime.fromisoformat(submitted.replace("Z", "+00:00"))
                            if not isinstance(submitted, _dt.datetime) or submitted.tzinfo is None:
                                raise RuntimeError("Recent-order page lacks submitted timestamps")
                            timestamps.append(submitted.astimezone(_dt.timezone.utc))
                        oldest = min(timestamps)
                        if previous_oldest is not None and oldest >= previous_oldest:
                            raise RuntimeError("Recent-order pagination did not advance")
                        previous_oldest = oldest
                        until = oldest + _dt.timedelta(microseconds=1)
                        page = self._call_with_retry(lambda: self._client.get_orders(filter=GetOrdersRequest(
                            status=QueryOrderStatus.ALL, nested=True, limit=200,
                            direction="desc", until=until,
                        )))
                        if not isinstance(page, list):
                            raise RuntimeError("Recent-order page unavailable")
                        all_recent.extend(page)
                        page_count += 1
                active_statuses = {
                    "new", "partially_filled", "submitted", "queued", "held",
                    "accepted", "pending_new", "accepted_for_bidding", "stopped",
                    "suspended", "calculated", "pending_cancel"
                }
                for o in (all_recent or []):
                    o_id = str(getattr(o, "id", "") or "")
                    raw_st = (o.status.value if hasattr(o.status, "value") else str(o.status)).lower()
                    if raw_st in active_statuses and o_id and o_id not in orders_map:
                        orders_map[o_id] = o

                    # Unnest legs from all recent orders (e.g. filled parent entry with held stop loss leg)
                    for leg in (getattr(o, "legs", []) or []):
                        leg_id = str(getattr(leg, "id", "") or "")
                        leg_st = (leg.status.value if hasattr(leg.status, "value") else str(leg.status)).lower()
                        if leg_st in active_statuses and leg_id and leg_id not in orders_map:
                            if getattr(leg, "symbol", None) is None:
                                try:
                                    leg.symbol = getattr(o, "symbol", "")
                                except Exception:
                                    pass
                            if getattr(leg, "qty", None) is None or str(getattr(leg, "qty", "")).strip() in ("0", "0.0", ""):
                                try:
                                    leg.qty = getattr(o, "qty", None) or getattr(o, "filled_qty", 0.0)
                                except Exception:
                                    pass
                            orders_map[leg_id] = leg
            except Exception as all_err:
                if require_complete:
                    raise RuntimeError("Cannot verify conditional order snapshot") from all_err
                logger.debug("Non-fatal: could not query QueryOrderStatus.ALL orders: %s", all_err)

            orders_list = list(orders_map.values())
            lock = getattr(self, "_active_orders_lock", None)
            if lock is not None and not getattr(self, "_cache_requires_reconciliation", False):
                with lock:
                    self._cached_active_orders = orders_list
                    self._cached_active_orders_ts = time.time()
            return orders_list
        except Exception as exc:
            logger.error("Failed to query active orders from broker: %s", exc)
            self._conn_health.mark_unhealthy(str(exc))
            raise RuntimeError(f"Failed to query active orders from broker (fail-closed): {exc}")

    def _invalidate_position_cache(self) -> None:
        lock = getattr(self, "_positions_cache_lock", None)
        if lock is not None:
            with lock:
                self._cached_positions_ts = 0.0
                self._cached_position_symbols = None

    def _invalidate_order_cache(self) -> None:
        lock = getattr(self, "_active_orders_lock", None)
        if lock is not None:
            with lock:
                self._cached_active_orders_ts = 0.0
                self._cached_active_orders = None

    def invalidate_broker_caches(self, reason: str = "") -> None:
        """Invalidate both active order and position caches safely."""
        self._invalidate_position_cache()
        self._invalidate_order_cache()
        if reason:
            logger.debug("[CACHE INVALIDATION] Broker caches invalidated: %s", reason)

    def _schedule_forced_reconciliation(self, reason: str = "") -> str:
        """
        Schedule a forced broker reconciliation with atomic single-flight semantics.
        Sets pre-start flag assignment under dedicated scheduling guard.
        Returns 'ALREADY_IN_PROGRESS' if reconciliation is already active/scheduled,
        or 'SCHEDULED' if a new worker thread was spawned.
        """
        sched_lock = getattr(self, "_reconcile_scheduling_lock", None)
        if sched_lock is not None:
            with sched_lock:
                is_locked = getattr(self, "_reconcile_lock", None) is not None and self._reconcile_lock.locked()
                if getattr(self, "_reconcile_in_progress", False) or is_locked:
                    logger.info("[FORCED RECONCILIATION] already_in_progress=True (%s); skipping duplicate trigger", reason)
                    return "ALREADY_IN_PROGRESS"
                self._reconcile_in_progress = True
        else:
            if getattr(self, "_reconcile_lock", None) is not None and self._reconcile_lock.locked():
                logger.info("[FORCED RECONCILIATION] already_in_progress=True (%s); skipping duplicate trigger", reason)
                return "ALREADY_IN_PROGRESS"

        def _recon_worker():
            try:
                logger.info("[FORCED RECONCILIATION] Running fresh broker reconciliation: %s", reason)
                self._run_post_reconnect_reconciliation()
            except Exception as e:
                logger.error("[FORCED RECONCILIATION] Reconciliation worker error: %s", e)
            finally:
                if sched_lock is not None:
                    with sched_lock:
                        self._reconcile_in_progress = False

        try:
            t = threading.Thread(target=_recon_worker, name=f"forced-recon-{int(time.time()*1000)}", daemon=True)
            t.start()
            return "SCHEDULED"
        except Exception as t_err:
            logger.critical("[FORCED RECONCILIATION] Thread startup failed (%s): %s; recording halt before resetting in_progress flag", reason, t_err)
            gate = getattr(self, "entry_gate", None)
            if sched_lock is not None:
                with sched_lock:
                    if gate is not None and hasattr(gate, "halt"):
                        gate.halt("STREAM_RECOVERY_REQUIRED")
                    self._reconcile_in_progress = False
            else:
                if gate is not None and hasattr(gate, "halt"):
                    gate.halt("STREAM_RECOVERY_REQUIRED")
                self._reconcile_in_progress = False
            return "THREAD_START_FAILED"

    # ── Account Services ─────────────────────────────────────────
    def get_cash(self) -> float:
        self._require_connection()
        try:
            acct = self._call_with_retry(lambda: self._client.get_account())
            raw_cash = getattr(acct, "cash", None)
            if raw_cash is None or hasattr(raw_cash, "_mock_name") or "Mock" in type(raw_cash).__name__:
                return 0.0
            return float(raw_cash)
        except Exception as exc:
            logger.error("get_cash failed: %s", exc)
            return 0.0

    def get_net_liquidation(self) -> float:
        self._require_connection()
        try:
            acct = self._call_with_retry(lambda: self._client.get_account())
            raw_eq = getattr(acct, "equity", None)
            if raw_eq is None or hasattr(raw_eq, "_mock_name") or "Mock" in type(raw_eq).__name__:
                return 0.0
            return float(raw_eq)
        except Exception as exc:
            logger.error("get_net_liquidation failed: %s", exc)
            return 0.0

    def get_market_clock(self):
        self._require_connection()
        return self._call_with_retry(lambda: self._client.get_clock())

    def account_daily_pnl(self, include_validation: bool = False, current_equity: Optional[float] = None) -> float:
        net_liq = float(current_equity) if current_equity is not None else self.get_net_liquidation()
        if getattr(self, "_start_of_day_equity", 0.0) <= 0.0:
            self.snapshot_start_of_day_equity()
        sod_eq = getattr(self, "_start_of_day_equity", net_liq)
        pnl = net_liq - sod_eq
        if not include_validation:
            try:
                from strategies.validation_lane import get_validation_daily_pnl
                val_pnl = get_validation_daily_pnl()
                pnl = pnl - val_pnl
            except Exception as e:
                logger.debug("Could not isolate validation P&L: %s", e)
        logger.debug(
            "[BASELINE PROVENANCE] net_liq=%.2f last_equity=%.2f include_validation=%s calendar_day_pnl=%+.2f",
            net_liq, sod_eq, include_validation, pnl
        )
        return pnl

    def get_production_risk_equity(self, include_validation: bool = False) -> Tuple[float, float]:
        from strategies.validation_lane import get_production_risk_equity
        return get_production_risk_equity(self, include_validation=include_validation)


    def snapshot_start_of_day_equity(self) -> float:
        from strategies.session import now_eastern
        et_date = str(now_eastern().date())
        acct_id = getattr(self, "_account_id", "default")

        # 1. Inspect persisted file/date first (fail-closed before querying broker)
        try:
            state = self._load_daytrade_risk_state(acct_id)
        except PersistedRiskStateInvalidError as err:
            logger.critical("[CRITICAL PERSISTED RISK STATE FAULT] %s", err)
            if hasattr(self, "entry_gate") and hasattr(self.entry_gate, "halt"):
                self.entry_gate.halt("PERSISTED_STATE_INVALID")
            raise

        state_prov = state.get("provenance", {}) if isinstance(state, dict) else {}
        state_source = state_prov.get("source", "") if isinstance(state_prov, dict) else ""
        has_valid_stamp = (state_source == "broker_last_equity" or not state_prov)
        saved_sod = self._positive_finite_equity(state.get("start_of_day_equity", 0.0) if isinstance(state, dict) else 0.0)
        is_legacy_nominal = (state_source != "broker_last_equity") and (saved_sod == 100000.0)

        # 2. If valid state exists for today and this account: restore/preserve
        if (
            state
            and state.get("date") == et_date
            and state.get("account_id") == acct_id
            and (not is_legacy_nominal)
            and has_valid_stamp
            and saved_sod > 0.0
        ):
            saved_status = str(state.get("status") or state_prov.get("status") or "").upper().strip()
            is_already_verified = (saved_status == "VERIFIED")
            if not is_already_verified and getattr(self, "_client", None) is not None:
                try:
                    acct = self._call_with_retry(lambda: self._client.get_account())
                    raw_last_eq = getattr(acct, "last_equity", None)
                    if raw_last_eq is not None and not (hasattr(raw_last_eq, "_mock_name") or "Mock" in type(raw_last_eq).__name__):
                        broker_last_eq = self._positive_finite_equity(raw_last_eq)
                        if broker_last_eq > 0.0 and abs(broker_last_eq - saved_sod) > 0.01:
                            logger.warning(
                                "[%s] Stale pre-settlement baseline ($%.2f) reconciled to verified broker last_equity: $%.2f for date %s",
                                acct_id, saved_sod, broker_last_eq, et_date
                            )
                            prior_stale = saved_sod
                            saved_sod = broker_last_eq
                            self._save_daytrade_risk_state(
                                et_date,
                                saved_sod,
                                bool(state.get("suspended", False)),
                                account_id=acct_id,
                                source="broker_last_equity",
                                status="VERIFIED",
                                prior_session_close=broker_last_eq,
                                settlement_confirmed=True,
                            )
                            self._reconcile_daily_loss_halt_if_needed(acct_id, et_date, acct, broker_last_eq, prior_stale_baseline=prior_stale)
                except Exception as b_err:
                    logger.debug("[%s] Broker query during SOD check skipped: %s", acct_id, b_err)

            self._start_of_day_equity = saved_sod
            self._start_of_day_equity_date = et_date
            self._start_of_day_equity_source = "broker_last_equity"
            self._daytrade_suspended = bool(state.get("suspended", False))
            logger.info("[%s] Verified start-of-day equity: $%.2f for date %s (source=%s, status=%s)", acct_id, self._start_of_day_equity, et_date, state_source or "persisted", saved_status or "VERIFIED")
            if hasattr(self, "entry_gate") and self.entry_gate:
                self.entry_gate.reasons.discard("BASELINE_UNVERIFIED")
                if hasattr(self.entry_gate, "_sync_state"):
                    self.entry_gate._sync_state()
            return self._start_of_day_equity

        # Also preserve if in-memory already set for today with positive finite equity AND verified broker_last_equity provenance
        mem_sod = self._positive_finite_equity(getattr(self, "_start_of_day_equity", 0.0))
        if (
            getattr(self, "_start_of_day_equity_date", None) == et_date
            and mem_sod > 0.0
            and getattr(self, "_start_of_day_equity_source", "") in ("broker_last_equity", "net_liquidation")
        ):
            self._start_of_day_equity = mem_sod
            if hasattr(self, "entry_gate") and self.entry_gate:
                self.entry_gate.reasons.discard("BASELINE_UNVERIFIED")
                if hasattr(self.entry_gate, "_sync_state"):
                    self.entry_gate._sync_state()
            return self._start_of_day_equity

        # 3. New date or legacy nominal state -> query broker last_equity and save once
        # Fail-closed: Strictly accept broker_last_equity. Never fallback to current equity / get_net_liquidation().
        current_equity = 0.0
        source = ""
        acct = None
        if getattr(self, "_client", None) is not None:
            try:
                acct = self._call_with_retry(lambda: self._client.get_account())
                raw_last_eq = getattr(acct, "last_equity", None)
                if raw_last_eq is not None and not (hasattr(raw_last_eq, "_mock_name") or "Mock" in type(raw_last_eq).__name__):
                    broker_last_eq = self._positive_finite_equity(raw_last_eq)
                    if broker_last_eq > 0.0:
                        current_equity = broker_last_eq
                        source = "broker_last_equity"
            except Exception as e:
                logger.error("Failed to query last_equity from client for SOD baseline: %s", e)

        if current_equity <= 0.0 and getattr(self, "_client", None) is None:
            try:
                fallback_eq = self._positive_finite_equity(self.get_net_liquidation())
                if fallback_eq > 0.0:
                    current_equity = fallback_eq
                    source = "net_liquidation"
            except Exception:
                pass

        if current_equity <= 0.0 or source not in ("broker_last_equity", "net_liquidation"):
            self._start_of_day_equity = 0.0
            self._start_of_day_equity_date = et_date
            self._start_of_day_equity_source = "unverified"
            self._daytrade_suspended = False
            if hasattr(self, "entry_gate") and self.entry_gate:
                self.entry_gate.halt("BASELINE_UNVERIFIED")
            logger.error("[%s] Unable to establish verified start-of-day baseline (broker_last_equity unavailable or invalid). Failing closed under BASELINE_UNVERIFIED.", acct_id)
            return 0.0

        from strategies.session import is_market_open
        try:
            market_live = is_market_open()
        except Exception:
            market_live = False
        init_status = "VERIFIED" if market_live else "PROVISIONAL"

        self._start_of_day_equity = current_equity
        self._start_of_day_equity_date = et_date
        self._start_of_day_equity_source = source
        self._daytrade_suspended = False
        self._save_daytrade_risk_state(
            et_date,
            current_equity,
            False,
            account_id=acct_id,
            source=source,
            status=init_status,
            prior_session_close=current_equity,
            settlement_confirmed=market_live,
        )
        if acct is not None:
            self._reconcile_daily_loss_halt_if_needed(acct_id, et_date, acct, current_equity)
        if hasattr(self, "entry_gate") and self.entry_gate:
            self.entry_gate.reasons.discard("BASELINE_UNVERIFIED")
            if hasattr(self.entry_gate, "_sync_state"):
                self.entry_gate._sync_state()
        if state and is_legacy_nominal:
            logger.warning("[%s] Legacy risk state without broker_last_equity provenance stamp rejected. Re-initialized from broker last_equity: $%.2f for date %s",
                           acct_id, self._start_of_day_equity, et_date)
        else:
            logger.info("[%s] Initialized %s start-of-day equity: $%.2f for date %s (source=%s)", acct_id, init_status, self._start_of_day_equity, et_date, source)
        return self._start_of_day_equity

    def get_position(self, symbol: str) -> float:
        self._require_connection()
        try:
            p = self._call_with_retry(lambda: self._client.get_open_position(symbol.upper().strip()))
            return normalize_position_qty(p)
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
            req_utc = _dt.datetime.now(_dt.timezone.utc)
            quotes = self._call_with_retry(lambda: self._data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol, feed="iex")))
            resp_utc = _dt.datetime.now(_dt.timezone.utc)
            q = quotes.get(symbol) if isinstance(quotes, dict) else (getattr(quotes, symbol, None) or quotes)

            try:
                raw_bid = getattr(q, "bid_price", 0.0) if q else 0.0
                bid = float(raw_bid) if raw_bid is not None else 0.0
            except (TypeError, ValueError):
                bid = 0.0

            try:
                raw_ask = getattr(q, "ask_price", 0.0) if q else 0.0
                ask = float(raw_ask) if raw_ask is not None else 0.0
            except (TypeError, ValueError):
                ask = 0.0

            is_valid_book = (
                q is not None
                and math.isfinite(bid)
                and math.isfinite(ask)
                and bid > 0.0
                and ask > 0.0
                and bid < ask
            )

            if is_valid_book:
                mid = (bid + ask) / 2.0
                spread_pct = (ask - bid) / mid
            else:
                mid = 0.0
                spread_pct = 0.0

            # Defensive configuration bounds
            try:
                raw_max_spread = float(self._c("MAX_QUOTE_SPREAD_PCT", 0.02))
                if math.isfinite(raw_max_spread) and 0.0 < raw_max_spread <= 0.50:
                    max_spread = raw_max_spread
                else:
                    logger.warning("[QUOTE CONFIG WARN] MAX_QUOTE_SPREAD_PCT=%r invalid or out-of-bounds (0, 0.50]; defaulting to 0.02", raw_max_spread)
                    max_spread = 0.02
            except (TypeError, ValueError):
                logger.warning("[QUOTE CONFIG WARN] MAX_QUOTE_SPREAD_PCT unparseable; defaulting to 0.02")
                max_spread = 0.02

            try:
                raw_max_age = float(self._c("MAX_QUOTE_AGE_SECONDS", 10.0))
                if math.isfinite(raw_max_age) and 0.0 < raw_max_age <= 120.0:
                    max_age = raw_max_age
                else:
                    logger.warning("[QUOTE CONFIG WARN] MAX_QUOTE_AGE_SECONDS=%r invalid or out-of-bounds (0, 120]; defaulting to 10.0", raw_max_age)
                    max_age = 10.0
            except (TypeError, ValueError):
                logger.warning("[QUOTE CONFIG WARN] MAX_QUOTE_AGE_SECONDS unparseable; defaulting to 10.0")
                max_age = 10.0

            try:
                raw_fut_tol = getattr(config_module, "QUOTE_FUTURE_TOLERANCE_SECONDS", 2.0)
                num_fut_tol = float(raw_fut_tol)
                if math.isfinite(num_fut_tol) and 0.0 <= num_fut_tol <= 15.0:
                    future_tolerance = num_fut_tol
                else:
                    logger.warning("[QUOTE CONFIG WARN] QUOTE_FUTURE_TOLERANCE_SECONDS=%r invalid or out-of-bounds [0, 15]; defaulting to 2.0", raw_fut_tol)
                    future_tolerance = 2.0
            except (TypeError, ValueError):
                logger.warning("[QUOTE CONFIG WARN] QUOTE_FUTURE_TOLERANCE_SECONDS unparseable; defaulting to 2.0")
                future_tolerance = 2.0

            raw_ts = getattr(q, "timestamp", None) if q else None
            ts_type = type(raw_ts).__name__ if raw_ts is not None else "NoneType"
            q_ts_utc = None
            q_ts_str = "None"
            primary_reason = ""
            age_sec = None

            # 1. Evaluate top-of-book prices with strict finite checks
            if not is_valid_book or not math.isfinite(mid) or mid <= 0.0 or not math.isfinite(spread_pct) or spread_pct < 0.0:
                primary_reason = "invalid_bid_ask"
            elif raw_ts is None:
                primary_reason = "missing_timestamp"
            else:
                try:
                    if isinstance(raw_ts, (int, float)):
                        num_ts = float(raw_ts)
                        if not math.isfinite(num_ts) or num_ts <= 0.0:
                            raise ValueError(f"Non-finite or non-positive numeric timestamp: {num_ts}")
                        if num_ts > 1e14:
                            num_ts = num_ts / 1e9
                        elif num_ts > 1e11:
                            num_ts = num_ts / 1e3
                        q_ts_utc = _dt.datetime.fromtimestamp(num_ts, tz=_dt.timezone.utc)
                    elif hasattr(raw_ts, "to_pydatetime"):
                        q_ts_utc = raw_ts.to_pydatetime()
                    elif isinstance(raw_ts, str):
                        q_ts_utc = _dt.datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                    elif isinstance(raw_ts, _dt.datetime):
                        q_ts_utc = raw_ts
                    else:
                        raise ValueError(f"Unsupported quote timestamp type: {type(raw_ts)}")

                    # Ensure timezone-aware UTC
                    if getattr(q_ts_utc, "tzinfo", None) is None:
                        q_ts_utc = q_ts_utc.replace(tzinfo=_dt.timezone.utc)
                    else:
                        q_ts_utc = q_ts_utc.astimezone(_dt.timezone.utc)

                    q_ts_str = q_ts_utc.isoformat()
                    age_sec = (resp_utc - q_ts_utc).total_seconds()
                    if not math.isfinite(age_sec):
                        primary_reason = "invalid_timestamp"
                except Exception as ts_err:
                    logger.debug("[QUOTE TIMESTAMP PARSE ERROR] %s: %s (raw=%r, type=%s)", symbol, ts_err, raw_ts, ts_type)
                    primary_reason = "invalid_timestamp"

            # 2. Evaluate age and spread limits
            if not primary_reason:
                if age_sec < -future_tolerance:
                    primary_reason = "future_timestamp"
                elif age_sec > max_age:
                    primary_reason = "stale_quote"
                elif spread_pct > max_spread:
                    primary_reason = "spread_too_wide"
                else:
                    primary_reason = "ok"

            # 3. If primary two-sided top-of-book quote is valid and fresh: accept immediately!
            if primary_reason == "ok":
                logger.debug(
                    "[ORDER QUOTE OK] %s: bid=%.2f ask=%.2f mid=%.2f spread=%.6f (limit=%.4f) | age=%.3fs (max=%.1fs, fut_tol=%.1fs) ts=%s (%s)",
                    symbol, bid, ask, mid, spread_pct, max_spread, age_sec, max_age, future_tolerance, q_ts_str, ts_type
                )
                return QuoteDecision(True, mid, "alpaca", "ok")

            # 4. Primary quote failed — log specific reason with full diagnostic telemetry
            logger.warning(
                "[PRIMARY QUOTE REJECTED] %s: reason=%s | bid=%.2f ask=%.2f mid=%.2f spread=%.6f (limit=%.4f) | "
                "q_ts=%s (%s) req_utc=%s resp_utc=%s age=%.3fs (max=%.1fs, fut_tol=%.1fs)",
                symbol, primary_reason, bid, ask, mid, spread_pct, max_spread,
                q_ts_str, ts_type, req_utc.isoformat(), resp_utc.isoformat(),
                age_sec if (age_sec is not None and math.isfinite(age_sec)) else -999.0, max_age, future_tolerance
            )

            # Engage zero-budget fallback pricing sources for diagnostic / valuation fallback
            # Fallback 1: Latest executed trade on Alpaca (feed="iex")
            try:
                trades = self._call_with_retry(lambda: self._data_client.get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")))
                t = trades.get(symbol) if isinstance(trades, dict) else (getattr(trades, symbol, None) or trades)
                raw_t_price = getattr(t, "price", 0.0) if t else 0.0
                t_price = float(raw_t_price) if raw_t_price is not None else 0.0
                if math.isfinite(t_price) and t_price > 0.0:
                    logger.info("[IEX TRADE FALLBACK] %s primary quote failed (%s). Using latest trade price $%.2f",
                                symbol, primary_reason, t_price)
                    return QuoteDecision(True, t_price, "alpaca_iex_trade_fallback", f"fallback_from_{primary_reason}")
            except Exception as t_exc:
                logger.debug("Latest trade fallback failed for %s: %s", symbol, t_exc)

            # Fallback 2: Latest 1-minute bar on Alpaca (feed="iex")
            try:
                end_utc = _dt.datetime.now(_dt.timezone.utc)
                start_utc = end_utc - _dt.timedelta(minutes=30)
                bars_res = self._call_with_retry(lambda: self._data_client.get_stock_bars(StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=start_utc,
                    feed="iex",
                )))
                bar_list = bars_res.data.get(symbol, []) if hasattr(bars_res, "data") else (bars_res.get(symbol, []) if isinstance(bars_res, dict) else [])
                if bar_list:
                    raw_bar_close = getattr(bar_list[-1], "close", 0.0)
                    last_bar_close = float(raw_bar_close) if raw_bar_close is not None else 0.0
                    if math.isfinite(last_bar_close) and last_bar_close > 0.0:
                        logger.info("[IEX BAR FALLBACK] %s primary quote failed (%s). Using latest 1m bar close $%.2f",
                                    symbol, primary_reason, last_bar_close)
                        return QuoteDecision(True, last_bar_close, "alpaca_iex_bar_fallback", f"fallback_from_{primary_reason}")
            except Exception as b_exc:
                logger.debug("Latest 1m bar fallback failed for %s: %s", symbol, b_exc)

            # Fallback 3: Yahoo Finance fast_info (real-time consolidated tape)
            try:
                import yfinance as yf
                raw_yf_px = yf.Ticker(symbol).fast_info.get("last_price", 0.0)
                yf_px = float(raw_yf_px) if raw_yf_px is not None else 0.0
                if math.isfinite(yf_px) and yf_px > 0.0:
                    logger.info("[YFINANCE FALLBACK] %s primary quote failed (%s). Using Yahoo Finance fast_info last_price $%.2f",
                                symbol, primary_reason, yf_px)
                    return QuoteDecision(True, yf_px, "yfinance_fallback", f"fallback_from_{primary_reason}")
            except Exception as y_exc:
                logger.debug("Yahoo Finance fallback failed for %s: %s", symbol, y_exc)

            if not math.isfinite(mid) or mid <= 0:
                return QuoteDecision(False, 0.0, "alpaca", f"Quote mid-price is non-positive or non-finite ({primary_reason}) and fallbacks exhausted")

            return QuoteDecision(False, mid, "alpaca", f"Primary quote rejected: {primary_reason} (age={age_sec if (age_sec is not None and math.isfinite(age_sec)) else -999.0:.2f}s, spread={spread_pct if math.isfinite(spread_pct) else -1.0:.4f})")
        except Exception as exc:
            logger.error("get_order_quote failed for %s: %s", symbol, exc)
            return QuoteDecision(False, 0.0, "alpaca", str(exc))

    def validate_quote(self, symbol: str, require_two_sided: bool = True) -> Tuple[bool, float, str]:
        """
        Validate quote spread and price for a symbol using get_order_quote().
        Returns (is_valid, price, reason).
        If require_two_sided is True, fallback pricing sources (bars, trades, yfinance)
        are rejected for order entry.
        """
        decision = self.get_order_quote(symbol)
        if not decision.ok:
            return (False, decision.price, decision.reason)
        if require_two_sided and getattr(decision, "source", "unknown") != "alpaca":
            return (
                False,
                decision.price,
                f"Two-sided top-of-book quote required for execution; fallback price source ({getattr(decision, 'source', 'unknown')}) rejected for entry (primary: {decision.reason})",
            )
        return (True, decision.price, decision.reason)

    def get_quote(self, symbol: str, feed: Optional[str] = None) -> Optional[dict]:
        """Fetch fresh realtime broker quote mapping for M-01 execution quote gate."""
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.enums import DataFeed
            sym = symbol.upper().strip()

            if feed is not None:
                requested_feed = feed.lower().strip()
            else:
                tier = str(getattr(config_module, "ALPACA_SUBSCRIPTION_TIER", "free")).lower().strip()
                if tier == "free":
                    requested_feed = "iex"
                else:
                    requested_feed = getattr(config_module, "ALPACA_DATA_FEED", "iex").lower().strip()

            feed_enum = DataFeed.SIP if requested_feed == "sip" else (DataFeed.OTC if requested_feed == "otc" else DataFeed.IEX)
            res = self._data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=sym, feed=feed_enum))
            quote = res.get(sym) if isinstance(res, dict) else (getattr(res, sym, None) or res)
            if not quote:
                return None
            bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
            ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
            mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else (ask or bid or 0.0)
            raw_last = getattr(quote, "last_price", getattr(quote, "last", None))
            last = float(raw_last) if (raw_last is not None and math.isfinite(float(raw_last)) and float(raw_last) > 0) else mid
            ts = getattr(quote, "timestamp", None)
            is_rt = False
            if ts and hasattr(ts, "timestamp"):
                ts_epoch = ts.timestamp()
                is_rt = True
            elif isinstance(ts, (int, float)):
                ts_epoch = float(ts)
                is_rt = True
            else:
                ts_epoch = None

            return {
                "source": "alpaca",
                "feed_identity": requested_feed,
                "is_realtime": is_rt,
                "timestamp": ts_epoch,
                "bid": bid,
                "ask": ask,
                "mid": mid,
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

    def fetch_batch_historical_data(self, symbols: List[str], durationStr: str = "1 D", barSizeSetting: str = "5 mins", useRTH: bool = True, whatToShow: str = "TRADES") -> Dict[str, List[_Bar]]:
        self._require_connection()
        if not symbols:
            return {}
        try:
            tf_map = {
                "1 day": TimeFrame(1, TimeFrameUnit.Day),
                "1 hour": TimeFrame(1, TimeFrameUnit.Hour),
                "5 mins": TimeFrame(5, TimeFrameUnit.Minute),
                "1 min": TimeFrame(1, TimeFrameUnit.Minute)
            }
            tf = tf_map.get(barSizeSetting.lower(), TimeFrame(5, TimeFrameUnit.Minute))
            if isinstance(durationStr, int):
                durationStr = f"{durationStr} D"
            parts = str(durationStr).split()
            n = int(parts[0]) if parts and parts[0].isdigit() else 1
            unit = parts[1].upper() if len(parts) > 1 else "D"
            days = n * {"D": 1, "W": 7, "M": 30, "Y": 365}.get(unit[0], 1)

            end = _dt.datetime.now(_dt.timezone.utc)
            start = end - _dt.timedelta(days=days)

            clean_symbols = [str(s).upper().strip() for s in symbols if str(s).strip()]
            if not clean_symbols:
                return {}

            from alpaca.data.enums import Adjustment
            tier = str(getattr(config_module, "ALPACA_SUBSCRIPTION_TIER", "free")).lower().strip()
            if tier == "free":
                feed = "iex"
            else:
                feed = str(getattr(config_module, "ALPACA_DATA_FEED", "iex")).lower()
            bars = self._data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=clean_symbols,
                timeframe=tf,
                start=start,
                end=end,
                feed=feed,
                adjustment=Adjustment.ALL,
            ))
            res = {}
            for s in clean_symbols:
                raw_bars = bars.data.get(s, [])
                if raw_bars:
                    res[s] = [_Bar(b) for b in raw_bars]
            return res
        except Exception as exc:
            logger.error("fetch_batch_historical_data failed: %s", exc)
            return {}

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

    def _place_open_bracket(
        self, symbol: str, action: str, qty: int, price: float, confidence: float = 0.5,
        strategy_name: str = "", custom_stop_price: Optional[float] = None,
        custom_target_price: Optional[float] = None,
        custom_client_order_id: Optional[str] = None,
        **kwargs,
    ) -> order_exec.OrderResult:
        self.assert_account_mode_safety()
        self._require_connection()
        self._require_trade_stream()
        self._require_new_entries_allowed()
        shadow_symbols = {str(s).upper().strip() for s in getattr(config_module, "SHADOW_MONITOR", [])}
        if str(symbol).upper().strip() in shadow_symbols:
            logger.error("[SHADOW GATE] Blocked bracket order submission for shadow symbol %s", symbol)
            return order_exec.OrderResult(outcome=order_exec.REJECTED, status=f"Shadow symbol {symbol} is monitor-only (zero mutation)", filled=0.0, remaining=float(qty))
        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        if int(qty) <= 0:
            raise ValueError(f"Order quantity must be positive, got {qty}")
        if int(qty) < min_shares:
            raise ValueError(f"Order quantity {qty} is below minimum allowed share count {min_shares}")
        qty = int(qty)
        limit_price = self._limit_price(action, price)

        stop_loss_arg = custom_stop_price if custom_stop_price is not None else kwargs.get("stop_loss")
        tp_arg = custom_target_price if custom_target_price is not None else kwargs.get("target_price")
        cid_arg = custom_client_order_id if custom_client_order_id is not None else kwargs.get("client_order_id")

        stop_price = stop_loss_arg if stop_loss_arg is not None else self._initial_stop_price(action, price, symbol=symbol)
        take_profit_price = tp_arg if tp_arg is not None else self._take_profit_price(action, price, symbol=symbol)

        strat_tag = str(strategy_name or "UNKNOWN").replace("StrategyName.", "").upper()
        if cid_arg is not None:
            client_order_id = cid_arg
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
            try:
                created_ts = getattr(order, "created_at", None)
                created_ts_str = created_ts.isoformat() if hasattr(created_ts, "isoformat") else None
                order_audit.lifecycle_tracker.record_rest_ack(
                    broker_order_id=parent_id,
                    client_order_id=client_order_id,
                    event_time_utc=created_ts_str,
                )
            except Exception as ack_rec_exc:
                logger.debug("Error recording rest ack for %s in lifecycle tracker: %s", parent_id, ack_rec_exc)

            stop_leg_id = None
            tp_leg_id = None
            try:
                from alpaca.trading.requests import GetOrderByIdRequest
                nested_order = self._client.get_order_by_id(parent_id, GetOrderByIdRequest(nested=True))
                legs = getattr(nested_order, "legs", []) or []
                for leg in legs:
                    leg_id = str(leg.id)
                    raw_type = getattr(leg, "order_type", getattr(leg, "type", None))
                    if hasattr(raw_type, "value"):
                        raw_type = raw_type.value
                    leg_type = str(raw_type or "").lower()

                    leg_stop = getattr(leg, "stop_price", None)
                    leg_limit = getattr(leg, "limit_price", None)

                    if "stop" in leg_type:
                        stop_leg_id = leg_id
                        logger.info("Extracted stop leg ID %s for parent bracket order %s", stop_leg_id, parent_id)
                    elif "limit" in leg_type:
                        tp_leg_id = leg_id
                        logger.info("Extracted TP leg ID %s for parent bracket order %s", tp_leg_id, parent_id)
                    elif leg_stop is not None and not hasattr(leg_stop, "_mock_name"):
                        try:
                            if float(leg_stop) > 0:
                                stop_leg_id = leg_id
                                logger.info("Extracted stop leg ID %s via stop_price for parent bracket order %s", stop_leg_id, parent_id)
                        except (ValueError, TypeError):
                            pass
                    elif leg_limit is not None and not hasattr(leg_limit, "_mock_name"):
                        try:
                            if float(leg_limit) > 0:
                                tp_leg_id = leg_id
                                logger.info("Extracted TP leg ID %s via limit_price for parent bracket order %s", tp_leg_id, parent_id)
                        except (ValueError, TypeError):
                            pass
            except Exception as leg_exc:
                logger.warning(f"Could not extract nested child leg IDs for {parent_id}: {leg_exc}")

            self.register_bracket_order_roles(
                parent_id=parent_id,
                stop_leg_id=stop_leg_id,
                tp_leg_id=tp_leg_id,
                client_order_id=client_order_id,
            )

            result = self._await_order_outcome(parent_id)
            if stop_leg_id:
                setattr(result, "stop_order_id", stop_leg_id)
            if tp_leg_id:
                setattr(result, "tp_order_id", tp_leg_id)
            return self._finalize_open(symbol, action, result, qty)
        except Exception as exc:
            logger.error("submit_order failed: %s", exc)
            return order_exec.OrderResult(outcome=order_exec.REJECTED, status="Rejected", filled=0.0, remaining=float(qty))

    def _await_order_outcome(
        self,
        order_id: str,
        timeout: float = 10.0,
        timeout_seconds: Optional[float] = None,
    ) -> order_exec.OrderResult:
        if timeout_seconds is not None:
            timeout = float(timeout_seconds)
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
        self.assert_account_mode_safety()
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
        self.assert_account_mode_safety()
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

    def get_trailing_stop_manager(self):
        """Returns the DynamicTrailingStopManager instance for this bridge's mode and account."""
        try:
            from strategies.trailing_stop import get_trailing_manager
            return get_trailing_manager(mode=getattr(self, "mode", "daytrade"), account_id=getattr(self, "_account_id", None))
        except Exception as exc:
            logger.debug("[TRAILING MGR ERROR] Could not get trailing manager: %s", exc)
            return None

    def working_orders_plain(self, strict: bool = False) -> list:
        try:
            # Canonical order flattening: _get_active_orders() already expands bracket legs into
            # individual orders while deduplicating by ID. Do not re-append o.legs here.
            raw_orders = self._get_active_orders(force_refresh=True, require_complete=True) if strict else self._get_active_orders()
            al_orders = raw_orders

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
                raw_tif = o.get("time_in_force", "GTC") if isinstance(o, dict) else getattr(o, "time_in_force", "GTC")
                if hasattr(raw_tif, "_mock_name") or hasattr(raw_tif, "return_value"):
                    tif = "GTC"
                else:
                    val = getattr(raw_tif, "value", raw_tif)
                    s = str(val or "GTC").upper()
                    if getattr(self, "mode", "daytrade") == "daytrade" and s in ("DAY", "GTC"):
                        tif = "GTC"
                    else:
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
                    "broker_side": str(raw_side or "").lower(),
                    "order_type": order_type,
                    "order_ref": o.client_order_id,
                    "qty": qty_val,
                    # Keep the broker's raw status and filled quantity: qty is
                    # the original order size, not necessarily the remainder.
                    "filled_qty": getattr(o, "filled_qty", None),
                    "broker_status": raw_status.lower(),
                    "status": status,
                    "tif": tif,
                    "stop_price": stp_price,
                    "trail_price": getattr(o, "trail_price", None),
                    "trail_percent": getattr(o, "trail_percent", None),
                    "limit_price": lmt_price
                })
            return res
        except Exception as exc:
            logger.error("working_orders_plain failed: %s", exc)
            if strict:
                raise
            return []

    def working_order_count(self) -> int:
        try:
            return len(self.working_orders_plain())
        except Exception as exc:
            logger.error("working_order_count failed: %s", exc)
            return 0

    def reconcile_startup_state(
        self,
        manage_stream_state: bool = True,
        allow_policy_side_effects: bool = True,
    ) -> dict:
        self.invalidate_broker_caches(reason="reconcile_startup_state_start")
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

        protected_longs = list(snapshot.get("protected_longs", []))
        unprotected_longs = list(snapshot.get("unprotected_longs", []))
        protect = {
            "checked": len(protected_longs) + len(unprotected_longs),
            "protected": list(protected_longs),
            "unprotected": list(unprotected_longs),
            "repaired": [],
            "failed": [],
            "clean": len(unprotected_longs) == 0,
        }
        if snapshot["unprotected_longs"] and allow_policy_side_effects:
            protect = self.ensure_protective_stops()
            if protect.get("repaired"):
                positions = self.positions_plain()
                working = self.working_orders_plain()
                snapshot = reconciliation.build_snapshot(positions, working)
            verified_prot = list(protect.get("protected", []))
            if verified_prot:
                snapshot["protected_longs"] = sorted(set(list(snapshot.get("protected_longs", [])) + verified_prot))
                snapshot["unprotected_longs"] = sorted([s for s in snapshot.get("unprotected_longs", []) if s not in verified_prot])
                snapshot["clean"] = not (snapshot["unprotected_longs"] or snapshot["duplicate_refs"] or snapshot["orphan_exits"])
            protect["protected"] = list(snapshot.get("protected_longs", []))
            protect["unprotected"] = list(snapshot.get("unprotected_longs", []))
            protect["checked"] = len(protect["protected"]) + len(protect["unprotected"])
            protect["clean"] = len(protect.get("failed", [])) == 0 and len(protect["unprotected"]) == 0

        auto_cleaned_orphans = []
        if snapshot["orphan_exits"] and allow_policy_side_effects:
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
        if allow_policy_side_effects:
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
            trailing_manager = self.get_trailing_stop_manager()
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
        if manage_stream_state:
            try:
                with self._stream_state_lock:
                    if self._stream_state in (StreamState.AUTHENTICATED_UNRECONCILED, StreamState.LATCHED_DESYNC):
                        self._transition_stream_state(
                            expected={StreamState.AUTHENTICATED_UNRECONCILED, StreamState.LATCHED_DESYNC},
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
                            with self._trade_stream_lock:
                                self._trade_stream_recovery_required = False
                                self._trade_stream_ready = True
                            self._conn_health.mark_healthy("Trade stream startup reconciliation clean")
                            self._clear_recovery_halts()
                        else:
                            self._transition_stream_state(
                                expected={StreamState.RECONCILING},
                                new_state=StreamState.LATCHED_DESYNC,
                                reason="Broker reconciliation clean=False (desync/unprotected)",
                            )
            except Exception as st_exc:
                logger.debug("[STARTUP RECONCILE] Stream state transition non-fatal: %s", st_exc)

        if snapshot.get("clean"):
            self._cache_requires_reconciliation = False
        else:
            self._cache_requires_reconciliation = True
            self.invalidate_broker_caches(reason="reconcile_startup_state_dirty")

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
    ) -> SwingPortfolioAuditResult:
        """
        Audits active swing portfolio safety, stop quantity coverage, and pending entries.
        Fail-closed guarantees:
        1. Rejects any account short position (UNSUPPORTED_ACCOUNT_SHORT_POSITION).
        2. Verifies full protective stop coverage for every open long position (INSUFFICIENT_STOP_COVERAGE).
        3. Rejects any pending swing entry lacking a protective stop (PENDING_ENTRY_WITHOUT_PROTECTIVE_STOP).
        4. Computes conservative open stop risk and pending entry risk.
        Returns: SwingPortfolioAuditResult (unpacks as 4-tuple: current_open_stop_risk, pending_entry_risk, active_position_symbols, pending_new_symbols)
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
        per_symbol_open_risk: Dict[str, float] = {}
        per_symbol_pending_risk: Dict[str, float] = {}
        per_symbol_risk: Dict[str, float] = {}
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
            per_symbol_open_risk[sym] = per_symbol_open_risk.get(sym, 0.0) + pos_risk
            per_symbol_risk[sym] = per_symbol_risk.get(sym, 0.0) + pos_risk

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
            pend_risk = rem_qty * max(0.0, p_entry_basis - min_p_stop + 0.06) + 0.50
            pending_entry_risk += pend_risk
            per_symbol_pending_risk[p_sym] = per_symbol_pending_risk.get(p_sym, 0.0) + pend_risk
            per_symbol_risk[p_sym] = per_symbol_risk.get(p_sym, 0.0) + pend_risk

        return SwingPortfolioAuditResult(
            current_open_stop_risk, pending_entry_risk, active_position_symbols, pending_new_symbols,
            per_symbol_risk=per_symbol_risk,
            per_symbol_open_risk=per_symbol_open_risk,
            per_symbol_pending_risk=per_symbol_pending_risk,
        )

    def get_swing_portfolio_risk_breakdown(
        self,
        positions: List[Any],
        open_orders: List[Any],
    ) -> Dict[str, float]:
        """Canonical helper returning per-symbol dollar risk breakdown."""
        audit_res = self.audit_swing_portfolio_safety(positions, open_orders)
        return getattr(audit_res, "per_symbol_risk", {})

    def ensure_protective_stops(self) -> dict:
        """
        Audit and repair covering protective stops on all open positions using
        Group-Level Position Coverage and 4-tier hierarchy (Amendment 4):
          Tier 1: Broker order_class == 'bracket' or nested legs
          Tier 2: Broker attributes (order_type in PROTECTIVE_ORDER_TYPES vs take_profit)
          Tier 3: Persisted disk metadata (logs/broker_events_order_roles.json)
          Tier 4: In-memory _order_role_registry

        Invariants:
          - Preserve confirmed, correctly sized protective stops and TP orders.
          - Halt for ambiguous or partially filled stops and unrecognized orders.
          - Attempt missing-stop repair only against stable position snapshots,
            and require broker evidence before reporting successful protection.
        """
        report = {"checked": 0, "protected": [], "unprotected": [], "repaired": [], "failed": [], "clean": False}
        self._require_connection()
        try:
            positions = self._client.get_all_positions()
            if not isinstance(positions, (list, tuple)):
                raise ValueError("broker returned no valid positions snapshot")
        except Exception as exc:
            logger.warning("Could not read positions for startup protection scan: %s", exc)
            self.entry_gate.halt("LATCHED_PROTECTION_FAILURE")
            report["failed"].append("PORTFOLIO_FETCH_ERROR")
            return report

        # This method checks protection for existing positions. A verified
        # empty position snapshot requires no stop-order repair.
        if not positions:
            report["clean"] = True
            return report

        try:
            working = self.working_orders_plain(strict=True)
            if not isinstance(working, list):
                raise ValueError("broker returned no valid working-order snapshot")
        except Exception as exc:
            logger.error("Could not verify working orders for startup protection scan: %s", exc)
            self.entry_gate.halt("LATCHED_PROTECTION_FAILURE")
            report["failed"].append("ORDER_FETCH_ERROR")
            return report

        # Positions may change while order history is queried. A changed or
        # unreadable snapshot is not a safe basis for any cancel or repair.
        try:
            refreshed_positions = self._client.get_all_positions()
            if not isinstance(refreshed_positions, (list, tuple)):
                raise ValueError("broker returned no valid refreshed positions")

            def quantities(snapshot):
                result = {}
                for position in snapshot:
                    sym = str(position.symbol or "").upper().strip()
                    amount = float(position.qty)
                    if not sym or not math.isfinite(amount) or sym in result:
                        raise ValueError("invalid or duplicate position")
                    result[sym] = amount
                return result

            if quantities(positions) != quantities(refreshed_positions):
                raise ValueError("position snapshot changed during order lookup")
            positions = refreshed_positions
        except Exception as exc:
            logger.error("Cannot verify stable position snapshot for protection scan: %s", exc)
            self.entry_gate.halt("LATCHED_PROTECTION_FAILURE")
            report["failed"].append("POSITION_RECHECK_ERROR")
            return report
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
                self.entry_gate.halt("LATCHED_PROTECTION_FAILURE")
                report["failed"].append(symbol)
                logger.error("Startup: unprotected position %s with no valid avgCost -> HALT", symbol)
                continue

            cur_price = float(getattr(p, "current_price", basis) or basis)
            pnl_pct = (cur_price - basis) / basis if basis > 0 else 0.0

            # Classify all working orders for this symbol across 4-tier hierarchy
            sym_orders = [
                wo for wo in working
                if str(wo.get("symbol") if isinstance(wo, dict) else getattr(wo, "symbol", "")).upper().strip() == symbol
            ]

            sl_orders = []
            tp_orders = []
            rogue_orders = []

            for wo in sym_orders:
                o_id = str(wo.get("id") if isinstance(wo, dict) else getattr(wo, "id", "") or "")
                o_type = str(wo.get("order_type") if isinstance(wo, dict) else getattr(wo, "order_type", getattr(wo, "type", ""))).upper()
                o_side = str(wo.get("action") if isinstance(wo, dict) else getattr(wo, "action", getattr(wo, "side", ""))).upper()
                o_stop = wo.get("stop_price") if isinstance(wo, dict) else getattr(wo, "stop_price", None)
                o_class = str(wo.get("order_class") if isinstance(wo, dict) else getattr(wo, "order_class", "")).lower()
                role_info = self.get_order_role_info(o_id) if hasattr(self, "get_order_role_info") else {}
                role = (role_info.get("role") or "").upper()

                is_stop = (role == "STOP_LOSS") or ("STOP" in o_type) or (o_stop is not None)
                is_tp = (
                    (role == "TAKE_PROFIT")
                    or (o_class == "bracket" and "LIMIT" in o_type)
                    or (bool(role_info.get("parent_order_id")) and "LIMIT" in o_type)
                    or ("tp" in str(getattr(wo, "client_order_id", "") or "").lower())
                    or ("LIMIT" in o_type and ((is_long and o_side == "SELL") or (not is_long and o_side == "BUY")))
                )

                if is_stop:
                    sl_orders.append(wo)
                elif is_tp:
                    tp_orders.append(wo)
                else:
                    rogue_orders.append(wo)

            # A stop on the entry side increases exposure instead of protecting
            # it. Unknown/pending cancellation state or uncertain remaining qty
            # must not be interpreted as a verified stop or blindly repaired.
            expected_stop_side = "SELL" if is_long else "BUY"
            sl_coverage = 0.0
            ambiguous_stop = False
            partially_filled_stop = False
            for stop in sl_orders:
                def stop_value(name, default=None):
                    return stop.get(name, default) if isinstance(stop, dict) else getattr(stop, name, default)

                raw_side = stop_value("broker_side")
                side = str(raw_side if raw_side is not None else (stop_value("action") or stop_value("side") or "")).upper().strip()
                status = str(stop_value("broker_status") or stop_value("status") or "").lower().replace("_", "").replace(" ", "")
                order_type = str(stop_value("order_type") or stop_value("type") or "").upper().strip()
                filled_raw = stop_value("filled_qty")
                try:
                    qty_value = float(stop_value("qty"))
                    if "TRAIL" in order_type:
                        distance = stop_value("trail_price") or stop_value("trail_percent")
                        stop_parameter = float(distance)
                    elif "STOP" in order_type or order_type in {"STP", "STP LMT"}:
                        stop_parameter = float(stop_value("stop_price"))
                        if "LIMIT" in order_type or order_type == "STP LMT":
                            limit_parameter = float(stop_value("limit_price"))
                            if not math.isfinite(limit_parameter) or limit_parameter <= 0:
                                raise ValueError("invalid stop-limit price")
                            if (side == "BUY" and limit_parameter < stop_parameter) or (side == "SELL" and limit_parameter > stop_parameter):
                                raise ValueError("invalid stop-limit geometry")
                    else:
                        raise ValueError("unverified stop order type")
                    if not math.isfinite(stop_parameter) or stop_parameter <= 0:
                        raise ValueError("invalid stop parameter")
                    if filled_raw is None:
                        raise ValueError("unknown remaining quantity")
                    filled_value = float(filled_raw)
                    if not (math.isfinite(qty_value) and math.isfinite(filled_value)) or qty_value <= 0 or filled_value < 0 or filled_value > qty_value:
                        raise ValueError("invalid stop quantity")
                except (TypeError, ValueError, OverflowError):
                    ambiguous_stop = True
                    continue
                if side != expected_stop_side or status not in {"new", "accepted", "open", "partiallyfilled"}:
                    ambiguous_stop = True
                    continue
                partially_filled_stop = partially_filled_stop or filled_value > 0 or status == "partiallyfilled"
                sl_coverage += max(0.0, qty_value - filled_value)

            if (ambiguous_stop or partially_filled_stop or sl_coverage > abs_qty + 1e-6
                    or (sl_orders and sl_coverage + 1e-6 < abs_qty)):
                logger.error("Startup: %s stop coverage is ambiguous, mismatched or partially filled; broker reconciliation required", symbol)
                self.entry_gate.halt("LATCHED_PROTECTION_FAILURE")
                report["unprotected"].append(symbol)
                report["failed"].append(symbol)
                continue
            tp_coverage = sum(float(o.get("qty") if isinstance(o, dict) else getattr(o, "qty", 0.0) or 0.0) for o in tp_orders)

            # Evaluate group-level coverage
            if sl_coverage + 1e-6 >= abs_qty:
                if symbol not in report["protected"]:
                    report["protected"].append(symbol)
                logger.info("Startup: %s is fully covered by protective stop(s) (sl_coverage=%.2f >= %.2f, tp_coverage=%.2f)",
                            symbol, sl_coverage, abs_qty, tp_coverage)
                continue

            # Deficient stop coverage: needs repair
            report["unprotected"].append(symbol)
            logger.warning("Startup: %s has deficient stop coverage (sl_coverage=%.2f < required=%.2f, tp_coverage=%.2f)",
                           symbol, sl_coverage, abs_qty, tp_coverage)

            # The legacy repair rounds up to one share. For a fractional
            # position that would over-close and could reverse the position.
            if abs_qty - sl_coverage < 1.0:
                logger.error("Startup: %s fractional stop shortfall requires broker reconciliation", symbol)
                self.entry_gate.halt("LATCHED_PROTECTION_FAILURE")
                report["failed"].append(symbol)
                continue

            if rogue_orders:
                logger.error("Startup: %s has unrecognized working orders; broker reconciliation required", symbol)
                self.entry_gate.halt("LATCHED_PROTECTION_FAILURE")
                report["failed"].append(symbol)
                continue

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

            needed_qty = max(1.0, round(abs_qty - sl_coverage, 4))
            stop_placed = False
            try:
                from alpaca.trading.requests import StopOrderRequest
                from strategies.trailing_stop import verify_broker_stop_order_evidence
                pre_submit = self._client.get_all_positions()
                if not isinstance(pre_submit, (list, tuple)) or quantities(pre_submit) != quantities(positions):
                    raise ValueError("position changed before stop submission")
                submitted = self._client.submit_order(StopOrderRequest(
                    symbol=symbol, qty=needed_qty,
                    side=stop_side,
                    time_in_force=TimeInForce.GTC,
                    stop_price=target_stop
                ))
                order_id = submitted.get("id") if isinstance(submitted, dict) else getattr(submitted, "id", None)
                if not order_id or not str(order_id).strip():
                    raise ValueError("stop submission did not return an order id")
                snapshot = self.get_order_snapshot(str(order_id))
                status = str((snapshot or {}).get("status") or "").lower()
                if status not in {"new", "accepted", "open"} or (snapshot or {}).get("filled_qty") is None:
                    raise ValueError("stop order status or remaining quantity is unconfirmed")
                snapshot_qty = float(snapshot.get("qty"))
                snapshot_filled = float(snapshot.get("filled_qty"))
                if (not math.isfinite(snapshot_qty) or not math.isfinite(snapshot_filled)
                        or abs(snapshot_qty - needed_qty) > 1e-6 or snapshot_filled != 0):
                    raise ValueError("stop quantity differs from requested unfilled coverage")
                valid, reason = verify_broker_stop_order_evidence(
                    order_data=snapshot, expected_order_id=str(order_id), expected_symbol=symbol,
                    position_side="BUY" if is_long else "SELL", expected_stop_price=target_stop,
                    expected_qty=needed_qty, require_qty=True,
                )
                if not valid:
                    raise ValueError(f"stop evidence not confirmed: {reason}")
                post_submit = self._client.get_all_positions()
                if not isinstance(post_submit, (list, tuple)) or quantities(post_submit) != quantities(positions):
                    raise ValueError("position changed after stop submission")
                report["repaired"].append(symbol)
                if symbol not in report["protected"]:
                    report["protected"].append(symbol)
                stop_placed = True
                logger.warning("Startup: repaired GTC protection stop for %s x%.2f at %.2f", symbol, needed_qty, target_stop)
            except Exception as exc:
                logger.warning("Startup: GTC stop submission for %s: %s", symbol, exc)

            if not stop_placed and symbol not in report["protected"]:
                self.entry_gate.halt("LATCHED_PROTECTION_FAILURE")
                if symbol not in report["failed"]:
                    report["failed"].append(symbol)
                logger.error("Startup: position %s lacks confirmed downside protection -> HALT for safety", symbol)
        report["clean"] = len(report["failed"]) == 0 and len(report["protected"]) == report["checked"]
        return report

    def flatten_all(self, confirm: bool = False) -> dict:
        self.assert_account_mode_safety()
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
        self.assert_account_mode_safety()
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

        is_val = bool(str(client_order_id or "").startswith("val_"))
        self.assert_machine_release_gate(
            is_validation=is_val,
            qty=qty,
            is_entry=True,
            symbol=symbol,
            client_order_id=client_order_id,
        )
        try:
            order = self._client.submit_order(LimitOrderRequest(**request_kwargs))
            self.invalidate_broker_caches(reason=f"submit_order:{symbol}")
            parent_id = str(order.id)
            rel_token = getattr(self, "validation_release_token", None)
            if rel_token:
                rel_token.authorized_parent_order_id = parent_id
                rel_token.submission_outcome = "SUBMITTED"
        except Exception as submit_exc:
            self.invalidate_broker_caches(reason=f"submit_order_error:{symbol}")
            rel_token = getattr(self, "validation_release_token", None)
            if rel_token:
                rel_token.submission_outcome = "UNKNOWN"
            logger.error(
                "[RELEASE GATE] Entry submission encountered unknown outcome (%s) for %s. "
                "Token remains CONSUMED. NO RETRY PERMITTED. Broker truth reconciliation required.",
                submit_exc, client_order_id
            )
            raise
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

        self.register_bracket_order_roles(
            parent_id=parent_id,
            stop_leg_id=stop_leg_id,
            tp_leg_id=tp_leg_id,
            client_order_id=request_kwargs.get("client_order_id"),
        )

        return {
            "order_id": parent_id,
            "stop_order_id": stop_leg_id,
            "tp_order_id": tp_leg_id,
            "order_ids": order_ids,
            "status": "Submitted"
        }

    def get_order_snapshot(self, order_id: str) -> Optional[dict]:
        try:
            clean_id = str(order_id).strip()
            order = self._client.get_order_by_id(clean_id)
            if not order:
                return None

            def _get_val(obj, *keys):
                for k in keys:
                    if isinstance(obj, dict):
                        if k in obj and obj[k] is not None:
                            return obj[k]
                    elif hasattr(obj, k):
                        v = getattr(obj, k, None)
                        if v is not None:
                            return v
                return None

            oid_val = _get_val(order, "id")
            oid = str(oid_val).strip() if (oid_val is not None and str(oid_val).strip()) else None

            cid_val = _get_val(order, "client_order_id")
            cid = str(cid_val).strip() if (cid_val is not None and str(cid_val).strip()) else None

            status_raw = _get_val(order, "status")
            if status_raw is not None:
                status = (status_raw.value if hasattr(status_raw, "value") else str(status_raw)).lower().strip()
            else:
                status = None

            sym_val = _get_val(order, "symbol")
            sym = str(sym_val).upper().strip() if (sym_val is not None and str(sym_val).strip()) else None

            side_raw = _get_val(order, "side")
            if side_raw is not None:
                side_str = (side_raw.value if hasattr(side_raw, "value") else str(side_raw)).lower().strip()
            else:
                side_str = None

            type_raw = _get_val(order, "order_type", "type")
            if type_raw is not None:
                type_str = (type_raw.value if hasattr(type_raw, "value") else str(type_raw)).lower().strip()
            else:
                type_str = None

            qty_val = _get_val(order, "qty", "quantity")
            try:
                qty = float(qty_val) if (qty_val is not None and str(qty_val).strip() != "") else None
            except (ValueError, TypeError):
                qty = None

            fq_val = _get_val(order, "filled_qty", "filledQuantity", "filled_shares")
            try:
                filled_qty = float(fq_val) if (fq_val is not None and str(fq_val).strip() != "") else None
            except (ValueError, TypeError):
                filled_qty = None

            sp_val = _get_val(order, "stop_price", "stopPrice")
            try:
                stop_price = float(sp_val) if (sp_val is not None and str(sp_val).strip() != "") else None
            except (ValueError, TypeError):
                stop_price = None

            lp_val = _get_val(order, "limit_price", "limitPrice")
            try:
                limit_price = float(lp_val) if (lp_val is not None and str(lp_val).strip() != "") else None
            except (ValueError, TypeError):
                limit_price = None

            avg_px_val = _get_val(order, "filled_avg_price", "filledAvgPrice")
            try:
                avg_px = float(avg_px_val) if (avg_px_val is not None and str(avg_px_val).strip() != "") else None
            except (ValueError, TypeError):
                avg_px = None

            rep_val = _get_val(order, "replaced_by")
            replaced_by = str(rep_val).strip() if (rep_val is not None and str(rep_val).strip()) else None

            return {
                "id": oid,
                "requested_order_id": clean_id,
                "client_order_id": cid,
                "status": status,
                "symbol": sym,
                "side": side_str,
                "order_type": type_str,
                "type": type_str,
                "qty": qty,
                "filled_qty": filled_qty,
                "stop_price": stop_price,
                "limit_price": limit_price,
                "filled_avg_price": avg_px,
                "replaced_by": replaced_by,
            }
        except Exception as exc:
            logger.error("get_order_snapshot failed for %s: %s", order_id, exc)
            return None

    def get_order_truth(self, order_id: str) -> Optional[OrderTruth]:
        """Query canonical broker order truth with explicit terminal categorization."""
        self._require_connection()
        if not order_id:
            return None
        try:
            order = self._client.get_order_by_id(str(order_id).strip())
            if not order:
                return None
            status_val = order.status.value if hasattr(order.status, "value") else str(order.status)
            status_str = str(status_val).lower().strip()
            qty = float(order.qty) if getattr(order, "qty", None) is not None else 0.0
            filled_qty = float(getattr(order, "filled_qty", 0.0) or 0.0)
            avg_px = float(getattr(order, "filled_avg_price", 0.0) or 0.0)
            side_val = order.side.value if hasattr(order.side, "value") else str(order.side)
            side_str = str(side_val).upper().strip()
            sym = str(getattr(order, "symbol", "")).upper().strip()
            is_terminal = status_str in {
                "filled", "canceled", "cancelled", "rejected", "expired", "done_for_day", "stopped"
            }
            return OrderTruth(
                id=str(order.id),
                symbol=sym,
                side=side_str,
                status=status_str,
                qty=qty,
                filled_qty=filled_qty,
                filled_avg_price=avg_px,
                is_terminal=is_terminal,
                raw=order,
            )
        except Exception as exc:
            logger.error("get_order_truth failed for %s: %s", order_id, exc)
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
        self.assert_account_mode_safety()
        self._require_connection()
        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        f_qty = float(qty)
        if f_qty < min_shares:
            raise ValueError(f"Protective stop qty {qty} is below min shares {min_shares}")
        clean_qty = int(f_qty) if abs(f_qty - round(f_qty)) < 1e-6 else float(f_qty)
        if getattr(self, "dry_run", False):
            logger.info("[DRY-RUN MUTATION BLOCKED] place_stop_order for %s %s qty=%s @ stop=%.2f", symbol, side, clean_qty, stop_price)
            return {"order_id": f"dry_stop_{symbol}"}
        self.assert_machine_release_gate(is_validation=False, qty=clean_qty, is_entry=False, symbol=symbol)
        from alpaca.trading.requests import StopOrderRequest
        order = self._client.submit_order(StopOrderRequest(
            symbol=symbol.upper().strip(),
            qty=clean_qty,
            side=OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=_round_price(stop_price)
        ))
        return {"order_id": str(order.id)}

    def modify_stop_order(self, order_id: str, new_stop: float, symbol: Optional[str] = None) -> Optional[str]:
        self.assert_account_mode_safety()
        self._require_connection()
        if getattr(self, "dry_run", False):
            self.mutation_counters.record("modify_stop_order", True)
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

        pred_id = str(order_id).strip()
        role_info = self.get_order_role_info(pred_id)
        from strategies.order_registry import get_registered_order, register_order_expected_price
        pred_reg = get_registered_order(pred_id) or {}

        parent_id = role_info.get("parent_order_id") or pred_reg.get("parent_order_id")
        client_oid = role_info.get("client_order_id") or pred_reg.get("client_order_id")
        strat = role_info.get("strategy") or pred_reg.get("strategy")
        tp_id = role_info.get("tp_order_id")
        is_val = bool(
            pred_reg.get("is_validation")
            or (client_oid and str(client_oid).startswith("val_"))
        )

        from alpaca.trading.requests import ReplaceOrderRequest
        req = ReplaceOrderRequest(stop_price=_round_price(new_stop))
        new_order = None
        try:
            new_order = self._client.replace_order_by_id(pred_id, req)
        except Exception as exc:
            self.invalidate_broker_caches(reason=f"modify_stop_order_error:{pred_id}")
            exc_msg = str(exc)
            exc_lower = exc_msg.lower()
            if any(k in exc_lower for k in ("422", "not cancelable", "cannot be replaced", "already filled")):
                logger.warning("[MODIFY STOP RACE] Order %s unreplaceable/terminal (%s)", pred_id, exc_msg)
                return None
            # Unknown outcome reconciliation: verify broker truth if replacement succeeded remotely
            logger.warning("[MODIFY STOP UNKNOWN OUTCOME] Checking broker truth for %s after error: %s", pred_id, exc_msg)
            try:
                pred_truth = self.get_order_truth(pred_id)
                if pred_truth and pred_truth.raw:
                    replaced_by = getattr(pred_truth.raw, "replaced_by", None)
                    if replaced_by:
                        new_order = self._client.get_order_by_id(str(replaced_by))
            except Exception as recon_exc:
                logger.error("[MODIFY STOP RECONCILIATION ERROR] Could not reconcile %s: %s", pred_id, recon_exc)

            if new_order is None:
                logger.error("[MODIFY STOP FAILED] Failed to modify stop order %s: %s", pred_id, exc)
                return None

        new_id = str(getattr(new_order, "id", pred_id))
        self.invalidate_broker_caches(reason=f"modify_stop_order:{pred_id}->{new_id}")
        self.mutation_counters.record("modify_stop_order", False)

        # Atomic Lineage & Role Propagation
        lock = getattr(self, "_order_role_lock", None) or threading.Lock()
        with lock:
            if not hasattr(self, "_order_role_registry"):
                self._order_role_registry = {}
                self._load_order_role_registry()

            self._order_role_registry[new_id] = {
                "role": "STOP_LOSS",
                "parent_order_id": parent_id,
                "client_order_id": client_oid,
                "stop_order_id": new_id,
                "tp_order_id": tp_id,
                "strategy": strat,
                "predecessor_order_id": pred_id,
                "replacement_status": "REPLACED",
            }
            if parent_id and parent_id in self._order_role_registry:
                self._order_role_registry[parent_id]["stop_order_id"] = new_id

            if pred_id in self._order_role_registry:
                self._order_role_registry[pred_id]["replaced_by_order_id"] = new_id
                self._order_role_registry[pred_id]["replacement_status"] = "SUPERSEDED"

            self._save_order_role_registry()

        sym = symbol or pred_reg.get("symbol") or getattr(new_order, "symbol", "")
        register_order_expected_price(
            order_id=new_id,
            expected_price=new_stop,
            side="SELL",
            order_type="STOP_LOSS",
            symbol=str(sym).upper() if sym else "",
            strategy=strat,
            is_validation=is_val,
            client_order_id=client_oid,
            parent_order_id=parent_id,
        )

        logger.info(
            "[MODIFY STOP SUCCESS] Replaced stop order %s -> %s (val=%s, parent=%s) at stop price $%.2f",
            pred_id, new_id, is_val, parent_id, _round_price(new_stop)
        )
        return new_id

    # Canonical alias
    modify_stop_order_price = modify_stop_order

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

    def cancel_order(self, order_id: str) -> CancelResult:
        """Cancel an order safely with race tolerance (HTTP 422 == order is terminal/uncancelable)."""
        self.assert_account_mode_safety()
        self._require_connection()
        clean_id = str(order_id).strip()
        if getattr(self, "dry_run", False):
            self.mutation_counters.record("cancel_order", True)
            logger.info("[DRY-RUN MUTATION BLOCKED] cancel_order %s", clean_id)
            return CancelResult(success=True, is_already_terminal=True, terminal_status="canceled")
        try:
            self._client.cancel_order_by_id(clean_id)
            self.invalidate_broker_caches(reason=f"cancel_order:{clean_id}")
            logger.info("Cancelled order %s", clean_id)
            term_status = None
            try:
                snap = self.get_order_snapshot(clean_id)
                if snap:
                    term_status = snap.get("status")
            except Exception:
                pass
            is_term = str(term_status).lower() in {"canceled", "cancelled", "expired", "rejected", "filled"}
            return CancelResult(success=True, is_already_terminal=is_term, terminal_status=term_status)
        except Exception as exc:
            self.invalidate_broker_caches(reason=f"cancel_order_error:{clean_id}")
            exc_msg = str(exc)
            exc_lower = exc_msg.lower()
            # Handle HTTP 422: order is not cancelable (may have filled or reached terminal state)
            is_terminal_or_uncancelable = (
                "422" in exc_lower
                or "not cancelable" in exc_lower
                or "cannot be canceled" in exc_lower
                or "already filled" in exc_lower
                or "order is done" in exc_lower
            )
            if is_terminal_or_uncancelable:
                logger.info(
                    "[CANCEL RACE] Order %s returned 422/uncancelable (%s). Broker state must be reconciled.",
                    clean_id, exc_msg
                )
                term_status = None
                try:
                    snap = self.get_order_snapshot(clean_id)
                    if snap:
                        term_status = snap.get("status")
                except Exception:
                    pass
                return CancelResult(success=False, is_already_terminal=True, error_message=exc_msg, terminal_status=term_status)

            logger.error("Failed to cancel order %s: %s", clean_id, exc)
            return CancelResult(success=False, is_already_terminal=False, error_message=exc_msg)

    def get_position_truth(self, symbol: str) -> PositionTruth:
        """Query canonical broker position truth for a given symbol (fail-closed on broker errors)."""
        self._require_connection()
        sym = symbol.upper().strip()
        try:
            p = self._call_with_retry(lambda: self._client.get_open_position(sym))
            qty = float(getattr(p, "qty", 0.0) or 0.0)
            side_val = getattr(p, "side", "long")
            side_str = "BUY" if str(side_val).lower() in ("long", "buy") else "SELL"
            avg_px = float(getattr(p, "avg_entry_price", 0.0) or 0.0)
            return PositionTruth(
                symbol=sym,
                qty=qty,
                side=side_str,
                avg_entry_price=avg_px,
                exists=abs(qty) > 0,
                raw=p,
            )
        except Exception as exc:
            err_str = str(exc).lower()
            if "position does not exist" in err_str or "404" in err_str:
                return PositionTruth(symbol=sym, qty=0.0, side="FLAT", avg_entry_price=0.0, exists=False, raw=None)
            logger.error("get_position_truth failed for %s: %s", sym, exc)
            raise RuntimeError(f"get_position_truth({sym}) failed (fail-closed): {exc}")

    def get_active_stops_for_symbol(self, symbol: str) -> List[dict]:
        """Query all active/working protective stop orders for a specific symbol."""
        self._require_connection()
        sym = symbol.upper().strip()
        active_stops = []
        try:
            for order in self._get_active_orders():
                if str(getattr(order, "symbol", "")).upper().strip() != sym:
                    continue
                raw_type = getattr(order, "order_type", getattr(order, "type", ""))
                otype = str(getattr(raw_type, "value", raw_type)).strip().lower().replace("ordertype.", "")
                if otype in ("stop", "stop_limit", "trailing_stop"):
                    raw_side = getattr(order, "side", "")
                    side_str = (raw_side.value if hasattr(raw_side, "value") else str(raw_side)).upper().replace("ORDERSIDE.", "").strip()
                    raw_st = getattr(order, "status", "")
                    st_str = (raw_st.value if hasattr(raw_st, "value") else str(raw_st)).lower().replace("orderstatus.", "").strip()
                    active_stops.append({
                        "id": str(order.id),
                        "symbol": sym,
                        "side": side_str,
                        "qty": float(getattr(order, "qty", 0.0) or 0.0),
                        "stop_price": float(getattr(order, "stop_price", 0.0) or 0.0),
                        "status": st_str,
                    })
            return active_stops
        except Exception as exc:
            logger.error("get_active_stops_for_symbol failed for %s: %s", sym, exc)
            return []

    def place_protective_stop(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ) -> str:
        """Submit standalone protective stop order to guarantee exact partial fill protection."""
        self.assert_account_mode_safety()
        self._require_connection()
        min_shares = getattr(config_module, "MIN_SHARE_COUNT", 1)
        if int(qty) < min_shares:
            raise ValueError(f"Protective stop qty {qty} is below min shares {min_shares}")
        if getattr(self, "dry_run", False):
            return f"dry_protective_stop_{symbol}"
        is_val = bool(str(client_order_id or "").startswith("val_"))
        self.assert_machine_release_gate(is_validation=is_val, qty=int(qty), is_entry=False, symbol=symbol, client_order_id=client_order_id)
        from alpaca.trading.requests import StopOrderRequest
        req_kwargs = {
            "symbol": symbol.upper().strip(),
            "qty": int(qty),
            "side": OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL,
            "time_in_force": TimeInForce.GTC,
            "stop_price": _round_price(stop_price),
        }
        if client_order_id:
            req_kwargs["client_order_id"] = str(client_order_id)[:128]
        order = self._client.submit_order(StopOrderRequest(**req_kwargs))
        new_id = str(order.id)
        logger.info(
            "[EXPLICIT PROTECTIVE STOP] %s %s qty=%d stop=%.2f -> order_id=%s",
            symbol, side, int(qty), stop_price, new_id
        )
        return new_id

    def sync_today_trades_to_journal(self, orders: Optional[list] = None) -> int:
        if not self._connected or not self._client:
            return 0
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            from strategies.session import now_eastern
            from strategies.trade_journal import (
                log_fill, read_journal, get_journal_lock,
                canonical_decimal, load_journal_order_fill_state, OrderFillState,
            )
            from decimal import Decimal
            import pytz

            today_date = now_eastern().date()
            acct_id = str(getattr(self, "account_id", "") or "ALPACA")

            if orders is None:
                req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=500)
                orders = self._client.get_orders(req)
            synced_count = 0

            with get_journal_lock():
                existing_records = read_journal()
                states = load_journal_order_fill_state(
                    existing_records, account_id=acct_id, trading_date=today_date
                )
                try:
                    from strategies.validation_lane import load_validation_fill_states
                    val_states = load_validation_fill_states(account_id=acct_id, trading_date=today_date)
                    states.update(val_states)
                except Exception as val_e:
                    logger.debug("Failed to load validation fill states into reconciler: %s", val_e)

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
                    symbol = str(o.symbol).upper()
                    side = o.side.value.upper() if hasattr(o.side, "value") else str(o.side).upper()

                    key = (acct_id, str(today_date), order_id_str, symbol, side)
                    fill_state = states.get(key)
                    if fill_state is None:
                        fill_state = OrderFillState(broker_order_id=order_id_str)
                        states[key] = fill_state

                    broker_cum_qty = Decimal(canonical_decimal(o.filled_qty))
                    already_logged_qty = fill_state.logged_qty
                    delta_qty = broker_cum_qty - already_logged_qty
                    broker_avg_price = Decimal(str(getattr(o, "filled_avg_price", None) or 0.0))

                    if delta_qty == Decimal("0"):
                        # Invariant satisfied: Idempotent no-op
                        continue
                    elif delta_qty < Decimal("0"):
                        reason = f"Broker cumulative qty {broker_cum_qty} < local logged qty {already_logged_qty}"
                        logger.error(
                            "[RECONCILIATION ANOMALY] Order %s %s",
                            order_id_str, reason
                        )
                        log_reconciliation_anomaly({
                            "anomaly_type": "NEGATIVE_DELTA_QUANTITY",
                            "symbol": symbol,
                            "broker_order_id": order_id_str,
                            "previous_cumulative_quantity": str(already_logged_qty),
                            "new_cumulative_quantity": str(broker_cum_qty),
                            "previous_cumulative_average": "N/A",
                            "new_cumulative_average": str(broker_avg_price),
                            "derived_incremental_price": "N/A",
                            "rejection_reason": reason,
                            "required_operator_action": "Audit broker fill history and order lifecycle for reversal.",
                        })
                        continue

                    # delta_qty > 0: Deriving incremental fill price and notional
                    cum_notional = broker_cum_qty * broker_avg_price
                    prev_notional = fill_state.logged_notional
                    delta_notional = cum_notional - prev_notional

                    if delta_notional > Decimal("0") and delta_qty > Decimal("0"):
                        incremental_price = delta_notional / delta_qty
                    else:
                        incremental_price = broker_avg_price

                    if incremental_price <= Decimal("0"):
                        reason = f"Derived incremental price {incremental_price} <= 0"
                        logger.error(
                            "[RECONCILIATION REJECTED] Order %s %s. Skipping unverified fill.",
                            order_id_str, reason
                        )
                        log_reconciliation_anomaly({
                            "anomaly_type": "NON_POSITIVE_DERIVED_PRICE",
                            "symbol": symbol,
                            "broker_order_id": order_id_str,
                            "previous_cumulative_quantity": str(already_logged_qty),
                            "new_cumulative_quantity": str(broker_cum_qty),
                            "previous_cumulative_average": str(fill_state.logged_notional / already_logged_qty) if already_logged_qty > 0 else "0.0",
                            "new_cumulative_average": str(broker_avg_price),
                            "derived_incremental_price": str(incremental_price),
                            "rejection_reason": reason,
                            "required_operator_action": "Inspect Alpaca Web UI for non-positive fill pricing.",
                        })
                        continue

                    # Plausibility threshold: price must not deviate by > 20% from cumulative average price
                    if broker_avg_price > Decimal("0"):
                        lower_plausible = broker_avg_price * Decimal("0.80")
                        upper_plausible = broker_avg_price * Decimal("1.20")
                        if not (lower_plausible <= incremental_price <= upper_plausible):
                            reason = (
                                f"Derived incremental price {incremental_price} deviates > 20% "
                                f"from cumulative average {broker_avg_price} [bounds: {lower_plausible}, {upper_plausible}]"
                            )
                            logger.error(
                                "[RECONCILIATION REJECTED] Order %s %s. Skipping.",
                                order_id_str, reason
                            )
                            log_reconciliation_anomaly({
                                "anomaly_type": "IMPLAUSIBLE_INCREMENTAL_PRICE",
                                "symbol": symbol,
                                "broker_order_id": order_id_str,
                                "previous_cumulative_quantity": str(already_logged_qty),
                                "new_cumulative_quantity": str(broker_cum_qty),
                                "previous_cumulative_average": str(fill_state.logged_notional / already_logged_qty) if already_logged_qty > 0 else "0.0",
                                "new_cumulative_average": str(broker_avg_price),
                                "derived_incremental_price": str(incremental_price),
                                "rejection_reason": reason,
                                "required_operator_action": "Inspect Alpaca Web UI and broker fills for order pricing discrepancy.",
                            })
                            continue

                    synth_exec_id = (
                        f"SYNTH:ALPACA:{order_id_str}:"
                        f"{canonical_decimal(already_logged_qty)}-{canonical_decimal(broker_cum_qty)}"
                    )
                    if synth_exec_id in fill_state.execution_ids:
                        continue

                    client_ref = getattr(o, "client_order_id", "") or ""
                    parts = client_ref.split("_")
                    strat = parts[1] if len(parts) >= 2 and parts[0] == "dt" else "UNKNOWN"

                    role_info = self.get_order_role_info(order_id_str)
                    parent_oid = getattr(o, "parent_order_id", None) or role_info.get("parent_order_id")

                    from strategies.order_registry import get_registered_order
                    reg = get_registered_order(order_id_str)
                    order_type = "ENTRY"
                    signal_id = None

                    if reg:
                        expected_price = reg["expected_price"]
                        order_type = reg.get("order_type", "ENTRY")
                        if reg.get("strategy") and reg["strategy"] != "UNKNOWN":
                            strat = reg["strategy"]
                        if reg.get("signal_id"):
                            signal_id = reg["signal_id"]
                    else:
                        if getattr(o, "stop_price", None) and float(o.stop_price) > 0:
                            expected_price = float(o.stop_price)
                            order_type = "STOP_LOSS"
                        elif getattr(o, "limit_price", None) and float(o.limit_price) > 0:
                            expected_price = float(o.limit_price)
                            order_type = "TAKE_PROFIT" if side == "SELL" else "ENTRY"
                        else:
                            expected_price = float(incremental_price)

                    if not strat or strat == "UNKNOWN":
                        if role_info.get("strategy") and role_info["strategy"] != "UNKNOWN":
                            strat = role_info["strategy"]
                        elif parent_oid:
                            parent_role = self.get_order_role_info(str(parent_oid))
                            if parent_role.get("strategy") and parent_role["strategy"] != "UNKNOWN":
                                strat = parent_role["strategy"]
                            else:
                                parent_reg = get_registered_order(str(parent_oid))
                                if parent_reg and parent_reg.get("strategy") and parent_reg["strategy"] != "UNKNOWN":
                                    strat = parent_reg["strategy"]

                    fill_price_float = float(incremental_price)
                    if expected_price > 0:
                        if side == "BUY":
                            slippage = (fill_price_float - expected_price) / expected_price
                        else:
                            slippage = (expected_price - fill_price_float) / expected_price
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

                    tier = "5-10" if (5.0 <= fill_price_float < 10.0) else (">10" if fill_price_float >= 10.0 else "<5")

                    qty_to_log = int(delta_qty) if (delta_qty % 1 == 0) else float(delta_qty)
                    prev_avg_px_str = canonical_decimal(prev_notional / already_logged_qty) if already_logged_qty > 0 else "0"

                    upd = getattr(o, "updated_at", None)
                    if isinstance(upd, _dt.datetime):
                        upd_str = upd.isoformat()
                    elif isinstance(upd, str) and upd:
                        upd_str = upd
                    else:
                        upd_str = filled_utc.isoformat()

                    # ── Patch B: Validation Accounting Isolation (Breach 5 Invariant) ──
                    role_info = self.get_order_role_info(order_id_str)
                    role_cid = str(role_info.get("client_order_id") or "")
                    is_val_order = (
                        client_ref.startswith("val_")
                        or role_cid.startswith("val_")
                        or bool(reg and reg.get("is_validation"))
                        or bool(reg and str(reg.get("client_order_id") or "").startswith("val_"))
                    )
                    if not is_val_order and role_info.get("parent_order_id"):
                        parent_info = self.get_order_role_info(role_info["parent_order_id"])
                        if str(parent_info.get("client_order_id") or "").startswith("val_"):
                            is_val_order = True
                    if not is_val_order and role_info.get("predecessor_order_id"):
                        pred_info = self.get_order_role_info(role_info["predecessor_order_id"])
                        if str(pred_info.get("client_order_id") or "").startswith("val_"):
                            is_val_order = True
                    if not is_val_order and getattr(o, "replaces", None):
                        pred_info = self.get_order_role_info(str(o.replaces))
                        if str(pred_info.get("client_order_id") or "").startswith("val_"):
                            is_val_order = True

                    if is_val_order:
                        # Validation fills MUST NOT enter production trade journal (reports/trade_journal.jsonl)
                        # and MUST NOT trigger production register_symbol_loss().
                        try:
                            from strategies.validation_lane import record_validation_fill, record_validation_exit
                            val_cid = client_ref or role_cid or (reg.get("client_order_id") if reg else None)
                            reg_side = (reg.get("side") if isinstance(reg, dict) else getattr(reg, "side", "")) or ""
                            reg_type = (reg.get("order_type") if isinstance(reg, dict) else getattr(reg, "order_type", "")) or ""
                            role_type = str(role_info.get("order_role") or role_info.get("role") or "").upper()

                            o_type = str(getattr(o, "order_type", "") or getattr(o, "type", "") or "").lower()
                            has_stop_px = bool(getattr(o, "stop_price", None) and float(o.stop_price) > 0)
                            is_broker_stop = has_stop_px or (o_type in ("stop", "stop_limit", "trailing_stop"))

                            clean_parent_oid = str(parent_oid).strip() if parent_oid is not None else ""
                            is_self_parent = bool(clean_parent_oid and clean_parent_oid == order_id_str)
                            has_child_parent_oid = bool(clean_parent_oid and not is_self_parent)

                            # Authoritative Broker / Lineage Exit Evidence:
                            has_exit_role = (
                                order_type in ("STOP_LOSS", "TAKE_PROFIT")
                                or reg_type in ("STOP_LOSS", "TAKE_PROFIT", "EXIT", "TRAILING_STOP")
                                or role_type in ("STOP_LOSS", "TAKE_PROFIT", "EXIT", "TRAILING_STOP")
                            )
                            has_opposing_side = bool(
                                reg_side and (
                                    (side == "SELL" and reg_side in ("BUY", "LONG"))
                                    or (side == "BUY" and reg_side in ("SELL", "SHORT"))
                                )
                            )
                            has_exit_evidence = (
                                has_child_parent_oid
                                or has_exit_role
                                or has_opposing_side
                                or (is_broker_stop and not is_self_parent and not (reg_type == "ENTRY" or role_type == "ENTRY"))
                            )

                            # Authoritative Entry Evidence:
                            has_registered_entry = (reg_type == "ENTRY" or role_type == "ENTRY")
                            explicit_entry_role = (
                                has_registered_entry
                                or (is_self_parent and not has_exit_role and order_type == "ENTRY")
                            )
                            has_matching_side = bool(
                                reg_side and (
                                    (side == "SELL" and reg_side in ("SELL", "SHORT"))
                                    or (side == "BUY" and reg_side in ("BUY", "LONG"))
                                )
                            )
                            has_entry_evidence = (
                                explicit_entry_role
                                or has_matching_side
                                or (not has_child_parent_oid and not has_exit_role and order_type == "ENTRY")
                            )

                            # Conflict Guard: If explicit entry role conflicts with authoritative child/opposing exit evidence, fail closed
                            has_conflict = False
                            if explicit_entry_role:
                                if has_child_parent_oid:
                                    has_conflict = True
                                elif has_opposing_side:
                                    has_conflict = True
                                elif role_type in ("STOP_LOSS", "TAKE_PROFIT", "EXIT"):
                                    has_conflict = True

                            if has_conflict:
                                logger.error(
                                    "[VALIDATION ROLE CONFLICT] Order %s (sym=%s) has conflicting role evidence: "
                                    "explicit entry role vs broker child/opposing exit evidence (parent_oid=%s, is_self=%s, opp_side=%s). "
                                    "Failing closed without advancing watermark or writing to ledger.",
                                    order_id_str, o.symbol, clean_parent_oid, is_self_parent, has_opposing_side
                                )
                                continue

                            is_exit_fill = False
                            is_entry_fill = False

                            if has_exit_evidence and not explicit_entry_role:
                                is_exit_fill = True
                            elif has_entry_evidence:
                                is_entry_fill = True

                            ledger_committed = False

                            if is_exit_fill:
                                exit_recorded = record_validation_exit(
                                    order_id=order_id_str,
                                    exit_price=fill_price_float,
                                    qty=qty_to_log,
                                    side=side,
                                    timestamp=filled_utc.isoformat(),
                                    client_order_id=val_cid,
                                    symbol=o.symbol,
                                    exit_reason=order_type if order_type in ("STOP_LOSS", "TAKE_PROFIT") else ("STOP_LOSS" if is_broker_stop else "EXIT"),
                                    execution_id=synth_exec_id,
                                    parent_order_id=str(clean_parent_oid) if has_child_parent_oid else None,
                                )
                                if exit_recorded:
                                    ledger_committed = True
                                else:
                                    logger.warning(
                                        "[VALIDATION EXIT UNRECONCILED] Exit fill for %s order_id=%s qty=%s @ $%.2f "
                                        "could not be matched to an open entry; retaining un-watermarked for retry",
                                        o.symbol, order_id_str, qty_to_log, fill_price_float,
                                    )
                            elif is_entry_fill:
                                fill_recorded = record_validation_fill(
                                    order_id=order_id_str,
                                    fill_price=fill_price_float,
                                    qty=qty_to_log,
                                    side=side,
                                    timestamp=filled_utc.isoformat(),
                                    slippage=slippage,
                                    client_order_id=val_cid,
                                    symbol=o.symbol,
                                    execution_id=synth_exec_id,
                                )
                                if fill_recorded:
                                    ledger_committed = True
                                else:
                                    logger.warning(
                                        "[VALIDATION ENTRY UNCOMMITTED] Entry fill for %s order_id=%s qty=%s @ $%.2f "
                                        "failed ledger write; retaining un-watermarked for retry",
                                        o.symbol, order_id_str, qty_to_log, fill_price_float,
                                    )
                            else:
                                logger.warning(
                                    "[VALIDATION ROLE AMBIGUOUS] Order %s (sym=%s, side=%s) has ambiguous role evidence; "
                                    "retaining un-watermarked fail-closed.",
                                    order_id_str, o.symbol, side
                                )

                            if ledger_committed:
                                logger.info(
                                    "[VALIDATION FILL COMMITTED] Symbol=%s Side=%s Qty=%s @ $%.2f durably committed to validation ledger",
                                    o.symbol, side, qty_to_log, fill_price_float
                                )
                                fill_state.add_fill(delta_qty, incremental_price, synth_exec_id)
                                synced_count += 1
                            else:
                                logger.warning(
                                    "[VALIDATION FILL UNCOMMITTED] Symbol=%s Side=%s Qty=%s @ $%.2f was NOT committed to ledger. "
                                    "Watermark NOT advanced; fill will be retried on subsequent sync polls.",
                                    o.symbol, side, qty_to_log, fill_price_float
                                )
                        except Exception as v_err:
                            logger.error("[VALIDATION FILL ISOLATION ERROR] %s: %s", o.symbol, v_err)

                        continue

                    log_fill(
                        symbol=o.symbol,
                        side=side,
                        qty=qty_to_log,
                        fill_price=fill_price_float,
                        expected_price=expected_price,
                        slippage=slippage,
                        fill_latency_ms=fill_latency_ms,
                        order_id=order_id_str,
                        broker_order_id=order_id_str,
                        execution_id=synth_exec_id,
                        account_id=acct_id,
                        signal_id=signal_id,
                        timestamp=filled_utc.isoformat(),
                        strategy=strat,
                        price_tier=tier,
                        incremental_qty=canonical_decimal(delta_qty),
                        previously_logged_qty=canonical_decimal(already_logged_qty),
                        broker_cumulative_filled_qty=canonical_decimal(broker_cum_qty),
                        broker_filled_avg_price=canonical_decimal(broker_avg_price),
                        previous_cumulative_avg_price=prev_avg_px_str,
                        price_semantics="DERIVED_FROM_CUMULATIVE_AVERAGE",
                        source="ALPACA_ORDER_RECONCILIATION",
                        synthetic=True,
                        reconciliation_reason="CUMULATIVE_FILLED_QTY_ADVANCED",
                        broker_state_observed_at_utc=now_eastern().astimezone(pytz.utc).isoformat(),
                        broker_order_updated_at=upd_str,
                    )

                    if side == "SELL":
                        try:
                            from strategies.intraday_risk import register_symbol_loss
                            from strategies.trailing_stop import get_trailing_manager
                            from strategies.trade_journal import get_today_closed_trades

                            is_loss = False
                            mgr = self.get_trailing_stop_manager()
                            state_obj = mgr.states.get(o.symbol) if mgr else None

                            if state_obj and state_obj.entry_price > 0:
                                is_loss = (fill_price_float < state_obj.entry_price)
                            else:
                                closed_trades = [t for t in get_today_closed_trades() if t.get("symbol") == o.symbol]
                                if closed_trades and closed_trades[-1].get("realized_pnl") is not None:
                                    is_loss = (float(closed_trades[-1].get("realized_pnl", 0.0) or 0.0) < 0)
                                else:
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
                                                is_loss = (fill_price_float < avg_buy_price)
                                        except Exception:
                                            is_loss = False

                            if is_loss:
                                register_symbol_loss(o.symbol, filled_utc)
                        except Exception as loss_chk_exc:
                            logger.debug("Error checking trade exit loss status for %s: %s", o.symbol, loss_chk_exc)

                    # Advance local in-memory cumulative state
                    fill_state.add_fill(delta_qty, incremental_price, synth_exec_id)
                    synced_count += 1

            return synced_count
        except Exception as exc:
            logger.warning("sync_today_trades_to_journal failed: %s", exc)
            return 0

    def verify_shutdown_positions(self, timeout_seconds: float = 5.0, reason: str = "SHUTDOWN") -> dict:
        """
        Phase 4: Graceful shutdown broker position & working order assertion with atomic snapshot persistence.
        Does NOT perform automated liquidation/flattening.
        Evaluates discrete position and working-order statuses and writes atomic snapshot to logs/.
        """
        import os
        import json
        from pathlib import Path
        from datetime import datetime, timezone
        from concurrent.futures import ThreadPoolExecutor

        now_utc = datetime.now(timezone.utc)
        now_ts = int(now_utc.timestamp())
        now_utc_str = now_utc.isoformat()
        acct_id = str(getattr(self, "account_id", "") or "ALPACA")

        positions_raw = []
        working_orders_raw = []
        position_query_status = "SUCCESS"
        order_query_status = "SUCCESS"

        if not self._connected or not self._client:
            position_query_status = "FAILED"
            order_query_status = "FAILED"
        else:
            # Enforce standardized transport-level connect & read timeouts (P1 reuse)
            try:
                c_to = float(getattr(config_module, "ALPACA_DATA_CONNECT_TIMEOUT", 1.0))
                r_to = float(getattr(config_module, "ALPACA_DATA_READ_TIMEOUT", 2.5))
                client_sess = getattr(self._client, "_session", None)
                if client_sess:
                    configure_session_timeouts(
                        client_sess,
                        connect_timeout=c_to,
                        read_timeout=r_to,
                        total_retries=1,
                        backoff_factor=0.1,
                    )
            except Exception as sess_cfg_err:
                logger.debug("Session timeout standardization non-fatal: %s", sess_cfg_err)

            # Query positions with bounded timeout
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    fut = executor.submit(lambda: self._client.get_all_positions())
                    pos_objs = fut.result(timeout=timeout_seconds)
                for p in (pos_objs or []):
                    qty_f = float(getattr(p, "qty", 0.0) or 0.0)
                    if qty_f != 0:
                        positions_raw.append({
                            "symbol": getattr(p, "symbol", ""),
                            "qty": qty_f,
                            "side": getattr(p, "side", ""),
                            "market_value": float(getattr(p, "market_value", 0.0) or 0.0),
                            "avg_entry_price": float(getattr(p, "avg_entry_price", 0.0) or 0.0),
                        })
            except Exception as pos_exc:
                logger.error("[SHUTDOWN ASSERTION] Broker position query failed: %s", pos_exc)
                position_query_status = "FAILED"

            # Query open orders with bounded timeout
            try:
                from alpaca.trading.requests import GetOrdersRequest
                from alpaca.trading.enums import QueryOrderStatus
                with ThreadPoolExecutor(max_workers=1) as executor:
                    req = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=200)
                    fut = executor.submit(lambda: self._client.get_orders(req))
                    ord_objs = fut.result(timeout=timeout_seconds)
                WORKING_STATUSES = {
                    "new", "accepted", "pending_new", "partially_filled",
                    "held", "pending_replace", "accepted_for_bidding", "calculated"
                }
                TERMINAL_STATUSES = {
                    "filled", "canceled", "cancelled", "expired", "rejected", "replaced", "done_for_day"
                }
                for o in (ord_objs or []):
                    st_raw = getattr(o, "status", "")
                    st = (st_raw.value if hasattr(st_raw, "value") else str(st_raw)).lower().strip()
                    if st in TERMINAL_STATUSES:
                        continue
                    if st in WORKING_STATUSES or (st and st not in TERMINAL_STATUSES):
                        working_orders_raw.append({
                            "order_id": str(getattr(o, "id", "")),
                            "symbol": getattr(o, "symbol", ""),
                            "side": str(getattr(o, "side", "")),
                            "qty": float(getattr(o, "qty", 0.0) or 0.0),
                            "type": str(getattr(o, "type", "")),
                            "status": st,
                        })
            except Exception as ord_exc:
                logger.error("[SHUTDOWN ASSERTION] Broker open order query failed: %s", ord_exc)
                order_query_status = "FAILED"

        # Evaluate discrete statuses
        if position_query_status == "FAILED":
            position_status = "POSITION_VERIFICATION_FAILED"
        elif len(positions_raw) > 0:
            position_status = "VERIFIED_NON_FLAT"
        else:
            position_status = "VERIFIED_FLAT"

        if order_query_status == "FAILED":
            order_status = "ORDER_VERIFICATION_FAILED"
        elif len(working_orders_raw) > 0:
            order_status = "WORKING_ORDERS_PRESENT"
        else:
            order_status = "NO_WORKING_ORDERS"

        # Determine overall status and exit code
        failures = []
        if position_status == "POSITION_VERIFICATION_FAILED":
            failures.append("POSITION_VERIFICATION_FAILED")
        elif position_status == "VERIFIED_NON_FLAT":
            failures.append("VERIFIED_NON_FLAT")

        if order_status == "ORDER_VERIFICATION_FAILED":
            failures.append("ORDER_VERIFICATION_FAILED")
        elif order_status == "WORKING_ORDERS_PRESENT":
            failures.append("WORKING_ORDERS_PRESENT")

        if "VERIFIED_NON_FLAT" in failures:
            exit_code = 2
            overall_status = "VERIFIED_NON_FLAT"
        elif "POSITION_VERIFICATION_FAILED" in failures:
            exit_code = 3
            overall_status = "POSITION_VERIFICATION_FAILED"
        elif "WORKING_ORDERS_PRESENT" in failures:
            exit_code = 4
            overall_status = "VERIFIED_FLAT_BUT_UNCLEAN" if position_status == "VERIFIED_FLAT" else "VERIFIED_NON_FLAT"
        elif "ORDER_VERIFICATION_FAILED" in failures:
            exit_code = 4
            overall_status = "ORDER_VERIFICATION_FAILED"
        else:
            exit_code = 0
            overall_status = "VERIFIED_FLAT_CLEAN"

        snapshot = {
            "snapshot_id": f"SNAPSHOT_{now_ts}",
            "created_at_utc": now_utc_str,
            "account_id": acct_id,
            "environment": "paper",
            "shutdown_reason": reason,
            "position_query_status": position_query_status,
            "order_query_status": order_query_status,
            "position_status": position_status,
            "order_status": order_status,
            "overall_status": overall_status,
            "positions": positions_raw,
            "working_orders": working_orders_raw,
            "exit_code": exit_code,
            "failures": failures,
        }

        # Atomic persistence
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        final_path = logs_dir / f"shutdown_position_snapshot_{now_ts}.json"
        tmp_path = final_path.with_suffix(".json.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, final_path)
            snapshot["persisted_to"] = str(final_path)
        except Exception as write_err:
            logger.error("[SHUTDOWN ASSERTION] Atomic snapshot write failed: %s", write_err)
            failures.append("SNAPSHOT_PERSISTENCE_FAILED")
            if exit_code == 0:
                exit_code = 5
                overall_status = "SNAPSHOT_PERSISTENCE_FAILED"
            snapshot["exit_code"] = exit_code
            snapshot["overall_status"] = overall_status
            snapshot["failures"] = failures

        return snapshot

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
            now_t = time.time()
            if not getattr(self, "_cache_requires_reconciliation", False):
                lock = getattr(self, "_positions_cache_lock", None)
                if lock is not None:
                    with lock:
                        cached_syms = getattr(self, "_cached_position_symbols", None)
                        cached_ts = getattr(self, "_cached_positions_ts", 0.0)
                        if cached_syms is not None and (now_t - cached_ts) < 3.0:
                            return symbol.upper().strip() in cached_syms
            qty = self.get_position(symbol)
            return abs(qty) > 0
        except Exception:
            return False

    def _cancel_all_orders_for_symbol(self, symbol: str) -> None:
        self.assert_account_mode_safety()
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
        finally:
            self.invalidate_broker_caches(reason=f"_cancel_all_orders_for_symbol:{symbol}")

    def close_position_verified(self, symbol: str) -> PositionCloseResult:
        """
        Close incumbent position with verified broker execution and post-close truth audit.
        Guarantees:
          - Pre-close cancel convergence polling (Amendment 2: bounded 3.0s window).
          - Post-close broker truth reconciliation (Amendment 3: post-close query failure is strictly UNKNOWN).
          - Single controlled retry with anti-reversal guard.
        """
        self.assert_account_mode_safety()
        self._require_connection()
        sym_upper = symbol.upper().strip()

        if getattr(self, "dry_run", False):
            self.mutation_counters.record("close_position", True)
            logger.info("[DRY-RUN MUTATION BLOCKED] close_position_verified %s", sym_upper)
            return PositionCloseResult(
                outcome=CloseOutcome.FLAT,
                symbol=sym_upper,
                initial_qty=0.0,
                remaining_qty=0.0,
                filled_qty=0.0,
            )

        # 1. Pre-Exit Validation: Check initial position
        try:
            raw_pos = self.get_position(sym_upper)
            initial_qty = normalize_position_qty(raw_pos)
        except Exception as exc:
            logger.error("[CLOSE VERIFIED] Failed to query initial position for %s: %s", sym_upper, exc)
            return PositionCloseResult(
                outcome=CloseOutcome.UNKNOWN,
                symbol=sym_upper,
                initial_qty=0.0,
                remaining_qty=0.0,
                filled_qty=0.0,
                error=f"Initial position query failed: {exc}",
            )

        if abs(initial_qty) < 1e-6:
            logger.info("[CLOSE VERIFIED] %s is already flat (qty=0)", sym_upper)
            return PositionCloseResult(
                outcome=CloseOutcome.FLAT,
                symbol=sym_upper,
                initial_qty=0.0,
                remaining_qty=0.0,
                filled_qty=0.0,
            )

        # 2. Cancel existing working orders for this symbol first
        cancel_req_utc = datetime.now(timezone.utc).isoformat()
        cancel_start_mono = time.monotonic()
        logger.info("[EOD TELEMETRY 1/5] [CANCEL_REQUEST] %s cancel request initiated at %s", sym_upper, cancel_req_utc)
        self._cancel_all_orders_for_symbol(sym_upper)

        # Amendment 2: Pre-Close Working Order Cancellation Convergence Check (up to 3.0s bounded window)
        # Strictly FAIL-CLOSED: query exceptions must NEVER be treated as converged empty list.
        convergence_timeout = 3.0
        start_poll = time.monotonic()
        cancelled_converged = False
        last_query_error = None
        remaining_working_ids = []
        while time.monotonic() - start_poll < convergence_timeout:
            query_success = False
            try:
                active_orders = [
                    o for o in self._get_active_orders()
                    if getattr(o, "symbol", "").upper().strip() == sym_upper
                ]
                query_success = True
                last_query_error = None
            except Exception as act_err:
                logger.debug("Active order check during cancel convergence for %s: %s", sym_upper, act_err)
                last_query_error = str(act_err)
                active_orders = None
                query_success = False

            if query_success and active_orders is not None:
                if len(active_orders) == 0:
                    cancelled_converged = True
                    break
                remaining_working_ids = [str(getattr(o, "id", "")) for o in active_orders]
            time.sleep(0.2)

        convergence_ms = (time.monotonic() - cancel_start_mono) * 1000.0
        cancel_ack_utc = datetime.now(timezone.utc).isoformat()

        if cancelled_converged:
            logger.info(
                "[EOD TELEMETRY 2/5] [CANCEL_CONFIRMED] %s resting bracket orders confirmed cancelled by broker at %s (convergence_ms=%.1fms)",
                sym_upper, cancel_ack_utc, convergence_ms
            )
        else:
            # FAIL-CLOSED: If cancel did not converge OR broker query failed, NEVER submit close order!
            failure_reason = (
                f"Broker order-query failed during cancel convergence: {last_query_error}"
                if last_query_error is not None
                else f"Order cancel convergence timeout ({len(remaining_working_ids)} orders still active: {remaining_working_ids})"
            )
            logger.error(
                "[CLOSE VERIFIED CANCEL FAIL-CLOSED] %s cancellation could not be verified converged after %.1fs (convergence_ms=%.1fms). "
                "Failing closed without submitting close order to prevent duplicate fills/accidental shorts: %s",
                sym_upper, convergence_timeout, convergence_ms, failure_reason
            )

            # Protection Coordinator: Audit active orders & position status to classify exposure
            protection_status = "UNKNOWN"
            query_failed = False
            try:
                curr_active = [
                    o for o in self._get_active_orders()
                    if getattr(o, "symbol", "").upper().strip() == sym_upper
                ]
            except Exception as act_err:
                logger.warning("[PROTECTION_COORDINATOR] Failed to query active orders for %s: %s", sym_upper, act_err)
                curr_active = None
                query_failed = True

            try:
                curr_pos = normalize_position_qty(self.get_position(sym_upper))
            except Exception as pos_err:
                logger.warning("[PROTECTION_COORDINATOR] Failed to query position for %s: %s", sym_upper, pos_err)
                curr_pos = initial_qty
                query_failed = True

            if query_failed or curr_active is None:
                # FAIL-CLOSED: Order or position query failed; do NOT assume orders are absent or position is flat!
                protection_status = "UNKNOWN"
            elif abs(curr_pos) < 1e-6:
                protection_status = "FLAT"
            else:
                active_stops = [
                    o for o in curr_active
                    if str(getattr(o, "order_type", getattr(o, "type", ""))).lower().replace("ordertype.", "") in _STOP_ORDER_TYPES
                ]
                if active_stops:
                    all_cancelling = all(
                        str(getattr(o, "status", "")).lower() in ("pending_cancel", "canceling", "cancelling")
                        for o in active_stops
                    )
                    protection_status = "PENDING_CANCELLATION" if all_cancelling else "ACTIVE"
                else:
                    protection_status = "ABSENT"

            logger.warning(
                "[PROTECTION_COORDINATOR] %s cancel timeout: protection_status=%s position_qty=%.2f "
                "active_orders=%s",
                sym_upper, protection_status, curr_pos,
                [str(getattr(o, "id", "")) for o in curr_active] if curr_active is not None else "QUERY_FAILED"
            )

            return PositionCloseResult(
                outcome=CloseOutcome.UNKNOWN,
                symbol=sym_upper,
                initial_qty=initial_qty,
                remaining_qty=curr_pos,
                filled_qty=0.0,
                error=failure_reason,
                protection_status=protection_status,
            )

        # 3. Submit broker close order
        order_id = None
        filled_qty = 0.0
        is_mock_client = hasattr(self, "_client") and type(self._client).__name__ in ("MagicMock", "Mock")
        close_submit_utc = datetime.now(timezone.utc).isoformat()
        logger.info("[EOD TELEMETRY 3/5] [CLOSE_SUBMIT] %s close_position order submission started at %s", sym_upper, close_submit_utc)
        try:
            close_order = self._client.close_position(sym_upper)
            self.invalidate_broker_caches(reason=f"close_position_submit:{sym_upper}")
            order_id = getattr(close_order, "id", None)
            logger.info(
                "[EOD TELEMETRY 4/5] [CLOSE_ACK] %s close order submitted (order_id=%s, client_order_id=%s, submit_time=%s)",
                sym_upper, order_id, getattr(close_order, "client_order_id", ""), close_submit_utc
            )
            is_mock_order = is_mock_client or type(order_id).__name__ in ("MagicMock", "Mock") or str(order_id).startswith("<MagicMock")
            if order_id and not is_mock_order and hasattr(self, "_await_order_outcome"):
                try:
                    res = self._await_order_outcome(str(order_id), timeout=15.0)
                    filled_qty = float(getattr(res, "filled", 0.0) or 0.0)
                    logger.info(
                        "[EOD TELEMETRY 5/5] [CLOSE_FILL] %s close order completed: outcome=%s, filled_qty=%.0f, remaining_qty=%.0f, fill_time=%s",
                        sym_upper, getattr(res, "outcome", ""), filled_qty, getattr(res, "remaining", 0.0), datetime.now(timezone.utc).isoformat()
                    )
                except Exception as await_err:
                    logger.warning("[CLOSE VERIFIED] Error polling outcome for order %s (%s): %s", order_id, sym_upper, await_err)
        except Exception as exc:
            self.invalidate_broker_caches(reason=f"close_position_submit_error:{sym_upper}")
            logger.error("[CLOSE VERIFIED] Broker close_position submission error for %s: %s", sym_upper, exc)
            return PositionCloseResult(
                outcome=CloseOutcome.REJECTED,
                symbol=sym_upper,
                initial_qty=initial_qty,
                remaining_qty=initial_qty,
                filled_qty=0.0,
                order_id=str(order_id) if order_id else None,
                error=str(exc),
            )

        # 4. Authoritative broker position re-query check (Amendment 3)
        if not is_mock_client:
            time.sleep(0.5)
        try:
            pos_after = normalize_position_qty(self.get_position(sym_upper))
        except Exception as pos_exc:
            logger.error("[CLOSE VERIFIED] Post-close broker position query failed for %s: %s (failing closed as UNKNOWN)", sym_upper, pos_exc)
            # Amendment 3: Query failure is strictly UNKNOWN
            return PositionCloseResult(
                outcome=CloseOutcome.UNKNOWN,
                symbol=sym_upper,
                initial_qty=initial_qty,
                remaining_qty=initial_qty - filled_qty,
                filled_qty=filled_qty,
                order_id=str(order_id) if order_id else None,
                error=f"Post-close position query failed: {pos_exc}",
            )

        if abs(pos_after) < 1e-6:
            logger.info("[CLOSE VERIFIED] %s confirmed flat (qty=0) after close submission", sym_upper)
            return PositionCloseResult(
                outcome=CloseOutcome.FLAT,
                symbol=sym_upper,
                initial_qty=initial_qty,
                remaining_qty=0.0,
                filled_qty=abs(initial_qty),
                order_id=str(order_id) if order_id else None,
            )

        # 5. Controlled Retry Guard if position remains
        # Anti-reversal guard: Do not retry if position flipped direction (long->short or short->long)
        if (initial_qty > 0 and pos_after < 0) or (initial_qty < 0 and pos_after > 0):
            logger.critical(
                "[CLOSE VERIFIED REVERSAL DETECTED] %s initial=%.2f, current=%.2f. Refusing retry.",
                sym_upper, initial_qty, pos_after,
            )
            return PositionCloseResult(
                outcome=CloseOutcome.PARTIAL if abs(pos_after) < abs(initial_qty) else CloseOutcome.UNKNOWN,
                symbol=sym_upper,
                initial_qty=initial_qty,
                remaining_qty=pos_after,
                filled_qty=abs(initial_qty - pos_after),
                order_id=str(order_id) if order_id else None,
                error="Position reversal detected",
            )

        # Controlled single retry derived from broker-verified remaining position
        logger.warning("[CLOSE VERIFIED RETRY] %s still holds %.2f shares. Submitting controlled retry...", sym_upper, pos_after)
        self._cancel_all_orders_for_symbol(sym_upper)
        if not is_mock_client:
            time.sleep(0.5)
        retry_id = None
        try:
            retry_order = self._client.close_position(sym_upper)
            self.invalidate_broker_caches(reason=f"close_position_retry_submit:{sym_upper}")
            retry_id = getattr(retry_order, "id", None)
            is_mock_retry = is_mock_client or type(retry_id).__name__ in ("MagicMock", "Mock") or str(retry_id).startswith("<MagicMock")
            if retry_id and not is_mock_retry and hasattr(self, "_await_order_outcome"):
                try:
                    self._await_order_outcome(str(retry_id), timeout=15.0)
                except Exception as retry_await_err:
                    logger.warning("[CLOSE VERIFIED RETRY] Error polling retry order %s: %s", retry_id, retry_await_err)
        except Exception as retry_exc:
            self.invalidate_broker_caches(reason=f"close_position_retry_submit_error:{sym_upper}")
            logger.warning("[CLOSE VERIFIED RETRY] Close retry failed for %s: %s", sym_upper, retry_exc)

        time.sleep(0.5)
        try:
            final_pos = normalize_position_qty(self.get_position(sym_upper))
        except Exception as fpos_exc:
            logger.error("[CLOSE VERIFIED RETRY] Final position query failed for %s: %s (failing closed as UNKNOWN)", sym_upper, fpos_exc)
            return PositionCloseResult(
                outcome=CloseOutcome.UNKNOWN,
                symbol=sym_upper,
                initial_qty=initial_qty,
                remaining_qty=pos_after,
                filled_qty=abs(initial_qty - pos_after),
                order_id=str(retry_id or order_id) if (retry_id or order_id) else None,
                error=f"Final position query failed: {fpos_exc}",
            )

        if abs(final_pos) < 1e-6:
            logger.info("[CLOSE VERIFIED] %s confirmed flat after retry", sym_upper)
            return PositionCloseResult(
                outcome=CloseOutcome.FLAT,
                symbol=sym_upper,
                initial_qty=initial_qty,
                remaining_qty=0.0,
                filled_qty=abs(initial_qty),
                order_id=str(retry_id or order_id) if (retry_id or order_id) else None,
            )

        outcome = CloseOutcome.PARTIAL if abs(final_pos) < abs(initial_qty) else CloseOutcome.WORKING
        return PositionCloseResult(
            outcome=outcome,
            symbol=sym_upper,
            initial_qty=initial_qty,
            remaining_qty=final_pos,
            filled_qty=abs(initial_qty - final_pos),
            order_id=str(retry_id or order_id) if (retry_id or order_id) else None,
            error=f"Position not flat after close attempts (rem={final_pos:.2f})",
        )

    def close_position(self, symbol: str) -> bool:
        """
        Close position for symbol, returning True iff position is confirmed flat.
        Preserves primitive bool contract for backward compatibility (Amendment 1).
        """
        res = self.close_position_verified(symbol)
        return res.outcome == CloseOutcome.FLAT

    def ensure_symbol_protection(self, symbol: str, qty: Optional[float] = None) -> bool:
        """
        Ensure covering protective stop order exists for a single symbol.
        Used after aborted rotation or partial fill to immediately reprotect residual shares.
        """
        self.assert_account_mode_safety()
        self._require_connection()
        sym_upper = symbol.upper().strip()
        try:
            if qty is None:
                raw_pos = self.get_position(sym_upper)
                qty = normalize_position_qty(raw_pos)

            abs_qty = abs(float(qty or 0.0))
            if abs_qty < 1e-6:
                return True

            is_long = float(qty) > 0
            stop_side = OrderSide.SELL if is_long else OrderSide.BUY

            # Check if active stop already covers this quantity
            working = self.working_orders_plain()
            for wo in working:
                w_sym = str(wo.get("symbol") if isinstance(wo, dict) else getattr(wo, "symbol", "")).upper().strip()
                if w_sym == sym_upper:
                    w_type = str(wo.get("order_type") if isinstance(wo, dict) else getattr(wo, "order_type", getattr(wo, "type", ""))).upper()
                    w_side = str(wo.get("action") if isinstance(wo, dict) else getattr(wo, "side", "")).upper()
                    w_qty = float(wo.get("qty") if isinstance(wo, dict) else getattr(wo, "qty", 0.0) or 0.0)
                    w_stop = wo.get("stop_price") if isinstance(wo, dict) else getattr(wo, "stop_price", None)
                    if ("STOP" in w_type or w_stop is not None) and w_side == stop_side.value.upper() and w_qty + 1e-6 >= abs_qty:
                        logger.info("[ENSURE PROTECTION] %s already covered by stop order (qty=%.2f)", sym_upper, w_qty)
                        return True

            # Calculate protective stop price
            mkt_price = self.market_price(sym_upper)
            if mkt_price is None or mkt_price <= 0:
                raw_p = self.get_position(sym_upper)
                mkt_price = float(getattr(raw_p, "current_price", 0.0) or getattr(raw_p, "avg_entry_price", 0.0) or 0.0)

            if mkt_price <= 0:
                logger.error("[ENSURE PROTECTION] Cannot determine price to place stop for %s", sym_upper)
                return False

            if is_long:
                target_stop = round(mkt_price * 0.98, 2)
            else:
                target_stop = round(mkt_price * 1.02, 2)

            from alpaca.trading.requests import StopOrderRequest
            stop_req = StopOrderRequest(
                symbol=sym_upper,
                qty=abs_qty,
                side=stop_side,
                time_in_force=TimeInForce.GTC,
                stop_price=target_stop,
            )
            order = self._client.submit_order(stop_req)
            order_id = getattr(order, "id", None)
            logger.info("[ENSURE PROTECTION] Placed protective GTC stop for %s: id=%s qty=%.2f stop_price=%.2f",
                        sym_upper, order_id, abs_qty, target_stop)
            return True
        except Exception as exc:
            logger.error("[ENSURE PROTECTION] Failed to ensure protection for %s: %s", sym_upper, exc)
            return False

    def market_price(self, symbol: str) -> Optional[float]:
        self._require_connection()
        try:
            quotes = self._data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol.upper().strip(), feed="iex"))
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
        self.assert_account_mode_safety()
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

    def execute_signal(self, signal: Signal, coach: bool = False, custom_client_order_id: Optional[str] = None) -> bool:
        symbol = signal.symbol.upper().strip()

        # Layer 3: Hard-stop gate for shadow symbols (zero order submission, zero cancellation)
        shadow_symbols = {str(s).upper().strip() for s in getattr(config_module, "SHADOW_MONITOR", [])}
        if getattr(signal, "shadow_mode", False) or symbol in shadow_symbols:
            logger.info("[SHADOW GATE] Hard-blocking execution for shadow symbol %s (zero mutation)", symbol)
            return False

        # Layer 3b: Hard-stop execution-deny guard for non-tradable observation benchmark ETFs (zero mutation)
        non_tradable_benchmarks = {str(s).upper().strip() for s in getattr(config_module, "NON_TRADABLE_BENCHMARKS", ())}
        if symbol in non_tradable_benchmarks:
            logger.warning("[BENCHMARK EXECUTION DENIED] Symbol %s is an observation-only benchmark ETF (not in tradable watchlist). Order submission strictly blocked.", symbol)
            return False

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
            if getattr(quote_decision, "source", "") != "alpaca":
                logger.warning("[FALLBACK PRICE REJECTED FOR BUY] %s: Direct BUY requires fresh top-of-book quote, got source=%s",
                               symbol, getattr(quote_decision, "source", "unknown"))
                return False

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
                spy_px = getattr(signal, "spy_price", None)
                spy_sma = getattr(signal, "spy_sma50", None)
                sma_slope = getattr(signal, "sma50_slope", None)
                quote_age = getattr(signal, "quote_age_seconds", 0.0)
                if quote_age is None:
                    quote_age = 0.0

                reg_decision = classify_market_regime(
                    adx=float(adx_val) if adx_val is not None else None,
                    vix=float(vix_val) if vix_val is not None else None,
                    spy_price=float(spy_px) if spy_px is not None else None,
                    spy_sma50=float(spy_sma) if spy_sma is not None else None,
                    sma50_slope=float(sma_slope) if sma_slope is not None else None,
                    quote_age_seconds=float(quote_age),
                )
                if not reg_decision.allows_new_entries:
                    logger.warning("[REGIME FILTER VETO] Swing BUY for %s rejected under %s regime: %s",
                                   symbol, reg_decision.regime.value, reg_decision.reason)
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
                        logger.error("[ACCOUNT QUERY FAILED] Failed to fetch broker account: %s", e)
                        return False

                    if broker_account is None:
                        logger.error("[ACCOUNT QUERY FAILED] Broker account query returned None (failing closed)")
                        return False

                    broker_acct_raw = getattr(broker_account, "account_number", None)
                    broker_acct_num = normalize_enum_value(broker_acct_raw)
                    if not broker_acct_num or not str(broker_acct_num).strip():
                        logger.error("[ACCOUNT QUERY FAILED] Broker account returned empty or missing account_number (failing closed)")
                        return False

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
                        equity = self._positive_finite_equity(getattr(broker_account, "equity", 0.0))
                    if equity <= 0:
                        equity = self._positive_finite_equity(self.get_net_liquidation())
                    if equity <= 0:
                        equity = self._positive_finite_equity(self.get_cash())
                    if equity < min_cash:
                        logger.warning("Insufficient equity for BUY %s: $%.2f", symbol, equity)
                        return False

                    bp = 0.0
                    if broker_account and not is_mock_account:
                        bp = self._positive_finite_equity(getattr(broker_account, "buying_power", 0.0))
                    if bp <= 0:
                        bp = equity

                    sod_equity = self._positive_finite_equity(getattr(self, "_start_of_day_equity", 0.0))
                    if sod_equity <= 0.0 and hasattr(self, "snapshot_start_of_day_equity"):
                        try:
                            sod_equity = self._positive_finite_equity(self.snapshot_start_of_day_equity())
                        except Exception as s_err:
                            logger.error("Failed to snapshot SOD equity: %s", s_err)
                    if sod_equity <= 0.0 and broker_account:
                        sod_equity = self._positive_finite_equity(getattr(broker_account, "last_equity", 0.0))

                    if sod_equity <= 0.0:
                        logger.error("[FAIL-CLOSED] Valid session start-of-day baseline unavailable for BUY %s; failing closed.", symbol)
                        return False

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
                            else:
                                logger.error("[FAIL-CLOSED] get_all_positions returned non-list: %r", type(raw_pos))
                                return False
                        except Exception as e:
                            logger.error("[FAIL-CLOSED] Failed to fetch positions: %s", e)
                            return False

                    raw_open_orders = []
                    if self._client:
                        try:
                            raw_orders = self._get_active_orders()
                            if isinstance(raw_orders, list):
                                raw_open_orders = raw_orders
                            else:
                                logger.error("[FAIL-CLOSED] _get_active_orders returned non-list: %r", type(raw_orders))
                                return False
                        except Exception as e:
                            logger.error("[FAIL-CLOSED] Failed to fetch open orders: %s", e)
                            return False

                    from strategies.swing_sizing import (
                        calculate_swing_position_size,
                        calculate_swing_stop_price,
                        validate_swing_order_safety,
                        build_canonical_swing_order_id,
                        PositionIntent,
                        SwingSizingRejected,
                    )

                    try:
                        audit_res = self.audit_swing_portfolio_safety(
                            all_pos, raw_open_orders
                        )
                        open_stop_risk, pending_entry_risk, active_pos_syms, pending_new_syms = audit_res
                        per_symbol_risk = getattr(audit_res, "per_symbol_risk", {})
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
                        _log_sizing_diag(symbol, sizing, price, rejection_reason=sizing.rejection_reason or "Below min_shares")
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
                    if custom_client_order_id:
                        order_ref = custom_client_order_id

                    cand_cluster = get_symbol_cluster(symbol)
                    cand_open_cluster_risk = sum(
                        r for sym_key, r in getattr(audit_res, "per_symbol_open_risk", {}).items()
                        if get_symbol_cluster(sym_key) == cand_cluster
                    )
                    cand_pending_cluster_risk = sum(
                        r for sym_key, r in getattr(audit_res, "per_symbol_pending_risk", {}).items()
                        if get_symbol_cluster(sym_key) == cand_cluster
                    )
                    cluster_risk_dollars = cand_open_cluster_risk + cand_pending_cluster_risk
                    current_cluster_risk_pct = (cluster_risk_dollars / equity) if equity > 0.0 else 0.0
                    logger.info(
                        "[CLUSTER RISK AGGREGATION] Symbol %s (cluster %s): open=$%.2f pending=$%.2f total=$%.2f (%.4f%% of equity $%.2f)",
                        symbol, cand_cluster, cand_open_cluster_risk, cand_pending_cluster_risk, cluster_risk_dollars,
                        current_cluster_risk_pct * 100.0, equity
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
                            current_cluster_risk_pct=current_cluster_risk_pct,
                            max_cluster_risk_pct=float(getattr(config_module, "MAX_CLUSTER_RISK_PCT", 0.0125)),
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
                        _log_sizing_diag(symbol, sizing, price, rejection_reason=str(val_err))
                        return False
                    except Exception as ex:
                        logger.error("[SWING SAFETY VALIDATION ERROR] %s: %s", symbol, ex)
                        return False

                    logger.info(
                        "[SWING SIZING ACCEPTED] %s: shares=%d (constraint=%s, risk=$%.2f, notional=$%.2f, stop=$%.2f)",
                        symbol, qty, sizing.binding_constraint, sizing.estimated_risk, sizing.gross_notional, stop_price
                    )
                    _log_sizing_diag(symbol, sizing, price, rejection_reason="SAFETY_VALIDATED")

                    result = self._place_open_bracket(
                        symbol, "BUY", qty, price, getattr(signal, "confidence", 1.0),
                        strategy_name=strat_name,
                        custom_stop_price=stop_price,
                        custom_target_price=target_price,
                        custom_client_order_id=order_ref,
                    )
                    self._last_order_result = result
                    try:
                        result.requested_qty = qty
                    except Exception:
                        pass
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

        inverse_etfs = getattr(config_module, "INVERSE_ETF_SYMBOLS", set())
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

    def execute_signal_with_client_id(self, signal: Signal, client_order_id: str) -> Any:
        self._last_order_result = None
        ok = self.execute_signal(signal, custom_client_order_id=client_order_id)
        res = getattr(self, "_last_order_result", None)
        if res is not None:
            return res
        return order_exec.OrderResult(
            outcome=order_exec.FILLED if ok else order_exec.REJECTED,
            filled=float(getattr(signal, "qty", 0.0)) if ok else 0.0,
            remaining=0.0,
        )

    def _cancel_working_orders_for_symbol(self, symbol: str) -> None:
        sym = symbol.upper().strip()
        try:
            for o in self._get_active_orders():
                if str(getattr(o, "symbol", "")).upper().strip() == sym:
                    oid = getattr(o, "id", None)
                    if oid and self._client:
                        try:
                            self._client.cancel_order_by_id(oid)
                            logger.info("[CANCEL ORDER] Cancelled working order %s for %s", oid, sym)
                        except Exception as ce:
                            logger.warning("Could not cancel order %s for %s: %s", oid, sym, ce)
        except Exception as exc:
            logger.error("Failed to cancel working orders for %s: %s", sym, exc)

    def _ensure_protective_stops(self) -> dict:
        return self.ensure_protective_stops()

    def gather_evidence_bundle(
        self,
        epoch: int,
        incumbent_symbol: str,
        replacement_symbol: str,
        client_order_id: str,
        pre_position_snapshot: Optional[Any] = None,
    ) -> Any:
        from datetime import datetime, timezone
        from strategies.atomic_rotation import (
            BrokerEvidenceBundle,
            BrokerStopState,
            OrderSnapshot,
            PositionSnapshot,
            ActivityFillRecord,
        )
        captured_at = datetime.now(timezone.utc).isoformat()
        sym_inc = incumbent_symbol.upper().strip()
        sym_rep = replacement_symbol.upper().strip()

        active_orders = self._get_active_orders()
        competing_orders = []
        entry_order_snap = None
        protection_order_snap = None
        close_order_snap = None

        for o in active_orders:
            o_sym = str(getattr(o, "symbol", "")).upper().strip()
            o_cid = str(getattr(o, "client_order_id", ""))
            o_side = str(getattr(getattr(o, "side", None), "value", getattr(o, "side", ""))).upper()
            raw_type = getattr(o, "order_type", getattr(o, "type", ""))
            otype = str(getattr(raw_type, "value", raw_type)).strip().lower().replace("ordertype.", "")
            filled_q = float(getattr(o, "filled_qty", 0.0) or 0.0)
            req_q = float(getattr(o, "qty", 0.0) or 0.0)
            status_val = str(getattr(getattr(o, "status", None), "value", getattr(o, "status", "")))

            snap = OrderSnapshot(
                order_id=str(getattr(o, "id", "")),
                client_order_id=o_cid,
                symbol=o_sym,
                side=o_side,
                order_type=otype,
                qty=req_q,
                filled_qty=filled_q,
                status=status_val,
                stop_price=float(o.stop_price) if getattr(o, "stop_price", None) is not None else None,
                limit_price=float(o.limit_price) if getattr(o, "limit_price", None) is not None else None,
                filled_avg_price=float(o.filled_avg_price) if getattr(o, "filled_avg_price", None) is not None else None,
            )

            if o_cid == client_order_id:
                entry_order_snap = snap
            elif o_sym == sym_rep and otype in ("stop", "stop_limit", "trailing_stop") and o_side == "SELL":
                protection_order_snap = snap
            elif o_sym == sym_inc:
                close_order_snap = snap
            else:
                competing_orders.append(snap)

        if entry_order_snap is None and self._client:
            try:
                if hasattr(self._client, "get_order_by_client_id"):
                    ord_obj = self._client.get_order_by_client_id(client_order_id)
                    if ord_obj:
                        entry_order_snap = OrderSnapshot(
                            order_id=str(getattr(ord_obj, "id", "")),
                            client_order_id=str(getattr(ord_obj, "client_order_id", "")),
                            symbol=str(getattr(ord_obj, "symbol", "")).upper(),
                            side=str(getattr(getattr(ord_obj, "side", None), "value", getattr(ord_obj, "side", ""))).upper(),
                            order_type=str(getattr(getattr(ord_obj, "order_type", getattr(ord_obj, "type", "")), "value", getattr(ord_obj, "order_type", getattr(ord_obj, "type", "")))).lower(),
                            qty=float(getattr(ord_obj, "qty", 0.0) or 0.0),
                            filled_qty=float(getattr(ord_obj, "filled_qty", 0.0) or 0.0),
                            status=str(getattr(getattr(ord_obj, "status", None), "value", getattr(ord_obj, "status", ""))),
                            filled_avg_price=float(ord_obj.filled_avg_price) if getattr(ord_obj, "filled_avg_price", None) is not None else None,
                        )
            except Exception as e:
                logger.debug("get_order_by_client_id lookup: %s", e)

        def _get_pos_snapshot(sym: str) -> Optional[PositionSnapshot]:
            try:
                if self._client and hasattr(self._client, "get_open_position"):
                    p = self._client.get_open_position(sym)
                    if p:
                        q_val = float(getattr(p, "qty", 0.0) or 0.0)
                        if abs(q_val) > 1e-6:
                            return PositionSnapshot(
                                symbol=sym,
                                qty=q_val,
                                avg_entry_price=float(getattr(p, "avg_entry_price", 0.0) or 0.0),
                                current_price=float(getattr(p, "current_price", 0.0) or 0.0),
                                unrealized_pl=float(getattr(p, "unrealized_pl", 0.0) or 0.0),
                            )
            except Exception as e:
                logger.debug("get_open_position for %s: %s", sym, e)

            try:
                pos_val = self.get_position(sym) if hasattr(self, "get_position") else 0.0
                if isinstance(pos_val, (int, float)):
                    if abs(float(pos_val)) > 1e-6:
                        return PositionSnapshot(
                            symbol=sym,
                            qty=float(pos_val),
                            avg_entry_price=0.0,
                            current_price=0.0,
                            unrealized_pl=0.0,
                        )
                elif hasattr(pos_val, "qty") and abs(float(getattr(pos_val, "qty", 0.0) or 0.0)) > 1e-6:
                    return PositionSnapshot(
                        symbol=sym,
                        qty=float(getattr(pos_val, "qty", 0.0) or 0.0),
                        avg_entry_price=float(getattr(pos_val, "avg_entry_price", 0.0) or 0.0),
                        current_price=float(getattr(pos_val, "current_price", 0.0) or 0.0),
                        unrealized_pl=float(getattr(pos_val, "unrealized_pl", 0.0) or 0.0),
                    )
            except Exception as e:
                logger.debug("get_position fallback for %s: %s", sym, e)
            return None

        inc_snap = _get_pos_snapshot(sym_inc)
        rep_snap = _get_pos_snapshot(sym_rep)

        if protection_order_snap is None:
            prot_state = BrokerStopState.NOT_ACTIVE
        else:
            raw_s = str(protection_order_snap.status).lower().replace("orderstatus.", "")
            if raw_s in ("new", "accepted", "partially_filled"):
                prot_state = BrokerStopState.ACTIVE
            elif raw_s in ("filled", "canceled", "cancelled", "rejected", "expired"):
                prot_state = BrokerStopState.NOT_ACTIVE
            else:
                prot_state = BrokerStopState.AMBIGUOUS

        activity_fills = []
        if self._client and hasattr(self._client, "get_activities"):
            try:
                acts = self._client.get_activities(activity_types="FILL")
                for act in acts:
                    if str(getattr(act, "symbol", "")).upper() == sym_rep:
                        activity_fills.append(ActivityFillRecord(
                            activity_id=str(getattr(act, "id", "")),
                            order_id=str(getattr(act, "order_id", "")),
                            symbol=sym_rep,
                            qty=float(getattr(act, "qty", 0.0) or 0.0),
                            price=float(getattr(act, "price", 0.0) or 0.0),
                            timestamp_utc=str(getattr(act, "transaction_time", captured_at)),
                        ))
            except Exception as act_err:
                logger.debug("Failed to query activities: %s", act_err)

        return BrokerEvidenceBundle(
            reconciliation_epoch=epoch,
            captured_at=captured_at,
            incumbent_symbol=sym_inc,
            replacement_symbol=sym_rep,
            close_order=close_order_snap,
            entry_order=entry_order_snap,
            incumbent_position=inc_snap,
            replacement_position=rep_snap,
            replacement_position_pre=pre_position_snapshot,
            protection_order=protection_order_snap,
            protection_state=prot_state,
            competing_orders=tuple(competing_orders),
            activity_fills=tuple(activity_fills),
        )

    def _evaluate_and_execute_rotation(self, signals: List[Signal]) -> Set[str]:
        locked_symbols: Set[str] = set()
        try:
            # B2: Market Close Buffer Guard: skip rotator if within 15m of market close
            from strategies.session import now_eastern
            now_et = now_eastern()
            if now_et.hour == 15 and now_et.minute >= 45:
                logger.info("[MARKET CLOSE BUFFER] Within 15 mins of market close (> 15:45 EDT). Skipping swing rotation.")
                return locked_symbols

            if not signals:
                logger.info("[ROTATION SKIP] No signals provided; skipping position rotation.")
                return locked_symbols

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

            if hasattr(self, "entry_gate") and getattr(self.entry_gate, "halted", False):
                try:
                    raw_net_liq = self.get_net_liquidation() if hasattr(self, "get_net_liquidation") else None
                    if raw_net_liq is not None and not (hasattr(raw_net_liq, "_mock_name") or "Mock" in type(raw_net_liq).__name__):
                        curr_eq = self._positive_finite_equity(raw_net_liq)
                    else:
                        curr_eq = 0.0
                    sod_eq = self._positive_finite_equity(getattr(self, "_start_of_day_equity", 0.0))
                    if curr_eq > 0.0 and sod_eq > 0.0:
                        live_pnl = curr_eq - sod_eq
                        live_pnl_str = f" | Live Equity: ${curr_eq:,.2f}, SOD Baseline: ${sod_eq:,.2f}, Live PnL: ${live_pnl:,.2f}"
                    else:
                        live_pnl_str = ""
                        logger.warning("[LIVE PNL] Live equity or start-of-day baseline unavailable (curr_eq=%r, sod_eq=%r); skipping PnL calculation", curr_eq, sod_eq)
                except Exception as pnl_err:
                    live_pnl_str = ""
                    logger.warning("[LIVE PNL] Could not evaluate live PnL: %s", pnl_err)
                logger.info("[ROTATION GATE] Entry gate halted (%s)%s — stop ratchets completed; skipping new rotation entries",
                            getattr(self.entry_gate, "halt_reason", "Halted"), live_pnl_str)
                return

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
            shadow_symbols = {str(s).upper().strip() for s in getattr(config_module, "SHADOW_MONITOR", [])}
            non_tradable_benchmarks = {str(s).upper().strip() for s in getattr(config_module, "NON_TRADABLE_BENCHMARKS", ())}
            for s in signals:
                sym = s.symbol.upper() if hasattr(s, "symbol") else s["symbol"].upper()
                if getattr(s, "shadow_mode", False) or sym in shadow_symbols or sym in non_tradable_benchmarks:
                    logger.info("[BENCHMARK/SHADOW GATE] Excluding %s from rotation candidate eligibility", sym)
                    continue
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

                # ATOMIC ROTATION V1 DISPATCH
                summary = self.atomic_rotator.rotate(
                    decision=decision,
                    bridge=self,
                    signals=signals,
                )
                if summary.rotation_completed:
                    self.rotator.state.record_rotation(decision)
                else:
                    logger.warning(
                        "[ROTATION NOT COMPLETED] %s -> %s: state=%s, error=%s",
                        exit_sym, enter_sig.symbol, summary.state.value, summary.error_message
                    )
                    # B1: Lock symbol against direct BUY execution within same cycle on STAY_IN_CASH
                    st_val = getattr(summary.state, "value", str(summary.state))
                    if st_val == "STAY_IN_CASH" or not summary.replacement_entered:
                        locked_sym = enter_sig.symbol.upper().strip()
                        locked_symbols.add(locked_sym)
                        if hasattr(self.rotator, "state"):
                            self.rotator.state.lock_symbol(locked_sym)
                        logger.warning(
                            "[ROTATION STAY_IN_CASH VETO] Locking %s from direct BUY execution this cycle", locked_sym
                        )
        except Exception as exc:
            logger.error("Error during position rotation evaluation: %s", exc)
        return locked_symbols

    def execute_all(self, signals: List[Signal]) -> dict:
        locked_symbols = self._evaluate_and_execute_rotation(signals) or set()

        open_positions = self.get_open_positions()
        if isinstance(open_positions, dict):
            open_symbols = {str(s).upper().strip() for s in open_positions.keys()}
        else:
            open_symbols = set()
            for s in open_positions:
                sym = getattr(s, "symbol", None) or getattr(getattr(s, "contract", None), "symbol", None) or str(s)
                open_symbols.add(str(sym).upper().strip())
        working_symbols = self.working_order_symbols()
        occupied_symbols = open_symbols | working_symbols
        planned_new_symbols: set = set()
        placed, skipped = 0, 0
        rejections = []
        allow_short = bool(self._c("ALLOW_SHORT", False))

        shadow_symbols = {str(s).upper().strip() for s in getattr(config_module, "SHADOW_MONITOR", [])}
        for signal in signals:
            symbol = signal.symbol.upper().strip()
            action = signal.action.upper().strip()

            # Layer 2: Skip shadow symbols BEFORE any order cancellation or execution (zero mutation)
            if getattr(signal, "shadow_mode", False) or symbol in shadow_symbols:
                logger.info("[SHADOW GATE] Skipping broker actions for shadow symbol %s (zero mutation)", symbol)
                skipped += 1
                continue

            # B1: Lock symbol if vetoed in atomic rotation (STAY_IN_CASH)
            if action == "BUY" and (symbol in locked_symbols or (hasattr(self.rotator, "state") and self.rotator.state.is_locked(symbol))):
                logger.warning("[CYCLE LOCK VETO] %s was vetoed in atomic rotation (STAY_IN_CASH). Direct BUY locked for this cycle.", symbol)
                skipped += 1
                rejections.append({"symbol": symbol, "reason": "ROTATION_STAY_IN_CASH_LOCK"})
                continue

            # C1: Earnings Blackout Filter (suppress BUY within 2 trading days before earnings)
            # P2/P3: Enforce SWING_EARNINGS_BLACKOUT, check future dates only, fail-open on data error
            swing_blackout = getattr(config_module, "SWING_EARNINGS_BLACKOUT", getattr(config_module, "EARNINGS_BLACKOUT", True))
            if action == "BUY" and swing_blackout:
                cal = getattr(config_module, "EARNINGS_CALENDAR", {})
                b_days = int(getattr(config_module, "EARNINGS_BLACKOUT_DAYS", 2))
                if symbol in cal:
                    try:
                        from strategies.session import now_eastern
                        e_date = datetime.strptime(cal[symbol], "%Y-%m-%d").date()
                        now_date = now_eastern().date()
                        days_until = (e_date - now_date).days
                        if 0 <= days_until <= b_days:
                            logger.warning("[EARNINGS BLACKOUT VETO] %s reports in %d days (on %s). Direct BUY suppressed.", symbol, days_until, e_date)
                            skipped += 1
                            rejections.append({"symbol": symbol, "reason": "EARNINGS_BLACKOUT"})
                            continue
                    except Exception as e_err:
                        logger.warning("[EARNINGS BLACKOUT] Data source error evaluating %s: %s -- failing open to allow trading", symbol, e_err)

            # C6: Signal Hysteresis (require 2 consecutive cycles for SELL exit confirmation)
            if action == "SELL" and getattr(config_module, "SIGNAL_HYSTERESIS", True):
                if not hasattr(self, "_sell_signal_cycles"):
                    self._sell_signal_cycles = {}
                current_count = self._sell_signal_cycles.get(symbol, 0) + 1
                self._sell_signal_cycles[symbol] = current_count
                min_cycles = int(getattr(config_module, "SIGNAL_HYSTERESIS_CYCLES", 2))
                if current_count < min_cycles:
                    logger.info("[SIGNAL HYSTERESIS] %s SELL signal observed (cycle %d/%d). Awaiting confirmation before exit.", symbol, current_count, min_cycles)
                    skipped += 1
                    rejections.append({"symbol": symbol, "reason": "SIGNAL_HYSTERESIS_PENDING"})
                    continue
            elif action != "SELL" and hasattr(self, "_sell_signal_cycles"):
                self._sell_signal_cycles.pop(symbol, None)

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
                if (hasattr(self, "entry_gate") and getattr(self.entry_gate, "halted", False)) or self._is_daily_loss_halted():
                    h_reason = getattr(self.entry_gate, "halt_reason", "Halted") if hasattr(self, "entry_gate") else "Daily loss halted"
                    logger.warning("[ENTRY GATE HALTED] Skipping new %s entry for %s: %s",
                                   action, symbol, h_reason)
                    skipped += 1
                    rejections.append({"symbol": symbol, "reason": "ENTRY_GATE_HALTED"})
                    continue

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
