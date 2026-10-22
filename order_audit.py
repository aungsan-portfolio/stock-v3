"""
order_audit.py - Structured (JSONL) audit log of the order lifecycle.

Phase 0 of the live-trading readiness plan (see
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md). This module is intentionally
tiny, dependency-light, and SIDE-EFFECT-ONLY: it appends one JSON object per
order-lifecycle event to logs/order_audit.jsonl. It NEVER raises into the
caller, so adding `order_audit.log_event(...)` calls inside the order path
cannot change trading behavior - every call is wrapped so a logging failure is
swallowed (and best-effort echoed to the standard logger).

Later phases (fill verification, server-side stops) assert against this log to
confirm the order path actually reached `filled` / `stop_confirmed`, instead of
stopping at `accepted`.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
import socket
import time
import uuid

import config

logger = logging.getLogger(__name__)

# Process and Host instance identity for clock-domain safety
PROCESS_INSTANCE_ID = uuid.uuid4().hex
HOST_ID = socket.gethostname()

def get_audit_file() -> Path:
    """Dynamically resolve order audit file against config.LOG_DIR."""
    override = globals().get("_audit_file_override")
    if override is not None:
        return Path(override)
    return Path(config.LOG_DIR) / "order_audit.jsonl"

AUDIT_FILE = get_audit_file()

# Canonical lifecycle stages, ordered. A healthy live entry should progress all
# the way to `stop_confirmed`; today (Phase 0) the path stops at `accepted`.
STAGE_SUBMIT = "submit"            # order handed to placeOrder()
STAGE_ACK = "ack"                  # broker acknowledged (PreSubmitted/Submitted)
STAGE_FILLED = "filled"            # actually filled (Phase 2 will emit this)
STAGE_PARTIAL = "partial_fill"     # partially filled (Phase 2)
STAGE_REJECTED = "rejected"        # broker rejected/cancelled
STAGE_STOP_PLACED = "stop_placed"  # protective child placed (Phase 3)
STAGE_STOP_CONFIRMED = "stop_confirmed"  # protective child verified live (Phase 2/3)
STAGE_TRADE_RECORDED = "trade_recorded"  # risk_state.record_trade() called
STAGE_FLATTEN = "flatten"          # emergency/normal flatten action
STAGE_PROTECT = "protect"          # GTC/OCA protective exit set placed + verified (Phase 3)
STAGE_HALT = "halt"                # new-entry halt: daily-loss / drawdown / unprotected startup (Phase 3)
STAGE_SNAPSHOT = "equity_snapshot" # start-of-day NetLiquidation snapshot (Phase 3)
STAGE_QUOTE = "quote"              # order-price quote validated for data integrity (Phase 4)
STAGE_RECONCILE = "reconcile"      # startup broker-truth reconciliation snapshot (Phase 5A, H18)
STAGE_SCHEDULE = "schedule"        # supervised scheduler run decision: allowed/blocked + dispatch (Phase 5B-1)
STAGE_WATCHDOG = "watchdog"        # reconnect/connection-health event: connected/retried/disconnect/gave_up (Phase 5B-2)
STAGE_SHUTDOWN = "shutdown"        # graceful one-shot shutdown: plan/clean-disconnect, unprotected-long scan (Phase 5B-3)
STAGE_ALERT = "alert"              # operator alert emitted by alerts.py (Phase 5B-4); LOG/audit-only, never trades

# ── Schema 2.1 Constants & Lifecycle Definitions ───────────────────────────
# Canonical Contract Invariants:
# 1. Event time != observation time
# 2. Invalid / negative / cross-domain duration -> NULL -> never clamp to zero
# 3. Historical bar evidence -> APPROXIMATE -> never exact touch
SCHEMA_VERSION = "2.1"

RECORD_TYPE_EVENT = "ORDER_LIFECYCLE_EVENT"
RECORD_TYPE_SUMMARY = "ORDER_LATENCY_SUMMARY"

EVENT_CANDIDATE_ACCEPTED = "CANDIDATE_ACCEPTED"  # T2
EVENT_SUBMIT_DISPATCHED = "SUBMIT_DISPATCHED"    # T3
EVENT_BROKER_REST_ACK = "BROKER_REST_ACK"        # T4a
EVENT_BROKER_STREAM_ACK = "BROKER_STREAM_ACK"    # T4b
EVENT_BROKER_ACK = "BROKER_ACK"                  # T4 (canonical earliest local ack)
EVENT_QUOTE_TOUCH = "QUOTE_TOUCH"                # Phase B: First quote touch
EVENT_TRADE_TOUCH = "TRADE_TOUCH"                # Phase B: First trade touch
EVENT_PARTIAL_FILL = "PARTIAL_FILL"              # T5 (first) or subsequent
EVENT_COMPLETE_FILL = "COMPLETE_FILL"            # T6
EVENT_ORDER_CANCELED = "ORDER_CANCELED"
EVENT_ORDER_REJECTED = "ORDER_REJECTED"
EVENT_ORDER_EXPIRED = "ORDER_EXPIRED"
EVENT_ORDER_REPLACED = "ORDER_REPLACED"

MARKETABILITY_MARKETABLE = "MARKETABLE"
MARKETABILITY_PASSIVE = "PASSIVE"
MARKETABILITY_UNKNOWN_NO_QUOTE = "UNKNOWN_NO_QUOTE"
MARKETABILITY_UNKNOWN_NO_HISTORICAL_QUOTE = "UNKNOWN_NO_HISTORICAL_QUOTE"
MARKETABILITY_UNKNOWN_STALE_QUOTE = "UNKNOWN_STALE_QUOTE"
MARKETABILITY_UNKNOWN_SIDE = "UNKNOWN_SIDE"
MARKETABILITY_UNKNOWN = "UNKNOWN"

# Phase B Evidence States
TOUCH_STATE_OBSERVED = "TOUCH_OBSERVED"
TOUCH_STATE_IMMEDIATE = "IMMEDIATE_AT_SUBMIT"
TOUCH_STATE_UNOBSERVED_GAP = "UNOBSERVED_FEED_GAP"
TOUCH_STATE_CONTRADICTORY = "CONTRADICTORY_EVIDENCE"
TOUCH_STATE_UNAVAILABLE = "TOUCH_TIMELINE_UNAVAILABLE"

# Phase B Timeline Resolutions
TOUCH_RES_TICK = "TICK_LEVEL"
TOUCH_RES_1S_BAR = "1S_BAR"
TOUCH_RES_1M_BAR = "1M_BAR"
TOUCH_RES_UNAVAILABLE = "UNAVAILABLE"

# Feeds
FEED_IEX = "IEX"
FEED_SIP = "CONSOLIDATED_SIP"
FEED_HISTORICAL_BAR = "HISTORICAL_BAR_APPROX"

DEFAULT_MAX_QUOTE_AGE_MS = float(getattr(config, "MAX_QUOTE_AGE_MS", getattr(config, "MAX_QUOTE_AGE_SECONDS", 10.0) * 1000.0))


def capture_local_clock() -> dict:
    """Capture normalized dual-clock observation timestamps with clock-domain identity.

    Returns:
        dict with:
          - observed_time_utc: ISO-8601 UTC string
          - observed_mono_ns: int monotonic ns
          - process_instance_id: unique run ID of current Python process
          - host_id: hostname of the execution host
    """
    return {
        "observed_time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "observed_mono_ns": time.monotonic_ns(),
        "process_instance_id": PROCESS_INSTANCE_ID,
        "host_id": HOST_ID,
    }


def classify_marketability(
    side: str,
    order_type: str,
    limit_price: float | None,
    bid: float | None,
    ask: float | None,
    quote_age_ms: float | None = None,
    max_quote_age_ms: float | None = None,
) -> str:
    """Classify limit order marketability at submission without premature assumption.

    Returns one of MARKETABILITY_* states. Missing or stale quotes explicitly return UNKNOWN_*.
    """
    if max_quote_age_ms is None:
        max_quote_age_ms = DEFAULT_MAX_QUOTE_AGE_MS

    ot = str(order_type or "").lower().strip()
    if ot == "market":
        return MARKETABILITY_MARKETABLE

    if bid is None or ask is None:
        return MARKETABILITY_UNKNOWN_NO_QUOTE

    if quote_age_ms is None or quote_age_ms > max_quote_age_ms:
        return MARKETABILITY_UNKNOWN_STALE_QUOTE

    if limit_price is None:
        return MARKETABILITY_UNKNOWN

    s = str(side or "").lower().strip()
    if s in ("buy", "long"):
        return MARKETABILITY_MARKETABLE if float(limit_price) >= float(ask) else MARKETABILITY_PASSIVE
    elif s in ("sell", "short"):
        return MARKETABILITY_MARKETABLE if float(limit_price) <= float(bid) else MARKETABILITY_PASSIVE

    return MARKETABILITY_UNKNOWN_SIDE


class _TrackedOrderLifecycle:
    """In-memory state for an order progressing through T2->T6 with Phase B touch telemetry."""

    def __init__(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        order_type: str = "limit",
        requested_qty: float = 0.0,
        limit_price: float | None = None,
        decision_price: float | None = None,
        signal_id: str | None = None,
        signal_family_id: str | None = None,
        broker_order_id: str | None = None,
        feed_source: str = "IEX",
        session_id: str | None = None,
        max_quote_age_ms: float | None = None,
        setup_id: str | None = None,
    ):
        self.client_order_id = client_order_id
        self.broker_order_id = broker_order_id
        self.signal_id = signal_id
        self.signal_family_id = signal_family_id
        self.setup_id = setup_id
        self.symbol = symbol
        self.side = side.lower() if side else ""
        self.order_type = order_type.lower() if order_type else "limit"
        self.requested_qty = float(requested_qty)
        self.filled_qty = 0.0
        self.limit_price = float(limit_price) if limit_price is not None else None
        self.decision_price = float(decision_price) if decision_price is not None else None
        self.feed_source = feed_source
        self.session_id = session_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        self.max_quote_age_ms_applied: float = float(max_quote_age_ms if max_quote_age_ms is not None else DEFAULT_MAX_QUOTE_AGE_MS)

        self.process_instance_id: str = PROCESS_INSTANCE_ID
        self.host_id: str = HOST_ID

        self.marketability_at_submit: str = MARKETABILITY_UNKNOWN
        self.marketability_scope: str = "IEX_OBSERVED" if feed_source == "IEX" else "CONSOLIDATED_NBBO"
        self.nbbo_verified: bool = (feed_source != "IEX")

        # Submit-time quote observation (S) - explicitly separated from ACK (T4)
        self.s_quote_observed_utc: str | None = None
        self.s_quote_observed_mono_ns: int | None = None
        self.s_quote_proc: str | None = None
        self.s_quote_host: str | None = None

        # Timestamps and clock domains (UTC ISO string + monotonic ns + proc_id + host_id)
        self.t0_source_event_utc: str | None = None
        self.t1_ingest_utc: str | None = None

        self.t2_decision_utc: str | None = None
        self.t2_decision_mono_ns: int | None = None
        self.t2_proc: str | None = None
        self.t2_host: str | None = None

        self.t3_submit_utc: str | None = None
        self.t3_submit_mono_ns: int | None = None
        self.t3_proc: str | None = None
        self.t3_host: str | None = None

        # ACK separation: REST (T4a) vs Stream (T4b) vs Broker-Reported (T4e) vs Canonical (T4)
        self.t4a_rest_ack_utc: str | None = None
        self.t4a_rest_ack_mono_ns: int | None = None
        self.t4a_proc: str | None = None
        self.t4a_host: str | None = None

        self.t4b_stream_ack_utc: str | None = None
        self.t4b_stream_ack_mono_ns: int | None = None
        self.t4b_proc: str | None = None
        self.t4b_host: str | None = None

        self.t4e_broker_event_utc: str | None = None

        self.t4_ack_utc: str | None = None
        self.t4_ack_mono_ns: int | None = None
        self.t4_proc: str | None = None
        self.t4_host: str | None = None

        # Phase B: Quote & Trade Touch Telemetry
        self.first_quote_touch_utc: str | None = None
        self.first_quote_touch_mono_ns: int | None = None
        self.first_quote_touch_proc: str | None = None
        self.first_quote_touch_host: str | None = None

        self.first_trade_touch_utc: str | None = None
        self.first_trade_touch_mono_ns: int | None = None
        self.first_trade_touch_proc: str | None = None
        self.first_trade_touch_host: str | None = None

        self.first_touch_source: str | None = None  # "QUOTE", "TRADE", "SUBMIT_QUOTE"
        self.first_touch_observed_time_utc: str | None = None
        self.first_touch_observed_mono_ns: int | None = None
        self.first_touch_proc: str | None = None
        self.first_touch_host: str | None = None

        # Feed health & Touch Evidence State
        self.touch_source_feed: str = feed_source
        self.touch_timeline_resolution: str = TOUCH_RES_UNAVAILABLE
        self.touch_evidence_state: str = TOUCH_STATE_UNAVAILABLE
        self.feed_subscription_active: bool = False
        self.feed_connected: bool = False
        self.feed_reconnect_gap: bool = False
        self.auxiliary_bar_cross_observed: bool = False

        # Fill Timestamps (T5, T6)
        self.t5_first_fill_utc: str | None = None
        self.t5_first_fill_mono_ns: int | None = None
        self.t5_proc: str | None = None
        self.t5_host: str | None = None

        self.t6_complete_fill_utc: str | None = None
        self.t6_complete_fill_mono_ns: int | None = None
        self.t6_proc: str | None = None
        self.t6_host: str | None = None

        # Fills tracking
        self.first_fill_price: float | None = None
        self.average_fill_price: float | None = None
        self.seen_execution_ids: set[str] = set()
        self.fill_transitions_count: int = 0

        # Status & Forensic classification
        self.is_terminal: bool = False
        self.terminal_status: str | None = None
        self.classification_reason: str = "QUOTE_TRADE_TIMELINE_UNAVAILABLE"


class OrderLifecycleTracker:
    """Thread-safe lifecycle assembler emitting Schema 2.1 events & latency summaries."""

    def __init__(self):
        import threading
        self._lock = threading.RLock()
        self._orders_by_broker_id: dict[str, _TrackedOrderLifecycle] = {}
        self._orders_by_client_id: dict[str, _TrackedOrderLifecycle] = {}

    def start_order(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        order_type: str = "limit",
        requested_qty: float = 0.0,
        limit_price: float | None = None,
        decision_price: float | None = None,
        signal_id: str | None = None,
        signal_family_id: str | None = None,
        broker_order_id: str | None = None,
        feed_source: str = "IEX",
        session_id: str | None = None,
        max_quote_age_ms: float | None = None,
        setup_id: str | None = None,
        t2_decision_utc: str | None = None,
        t2_decision_mono_ns: int | None = None,
        t2_proc: str | None = None,
        t2_host: str | None = None,
    ) -> _TrackedOrderLifecycle:
        with self._lock:
            ord_obj = _TrackedOrderLifecycle(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                requested_qty=requested_qty,
                limit_price=limit_price,
                decision_price=decision_price,
                signal_id=signal_id,
                signal_family_id=signal_family_id,
                broker_order_id=broker_order_id,
                feed_source=feed_source,
                session_id=session_id,
                max_quote_age_ms=max_quote_age_ms,
                setup_id=setup_id,
            )
            ord_obj.t2_decision_utc = t2_decision_utc
            ord_obj.t2_decision_mono_ns = t2_decision_mono_ns
            ord_obj.t2_proc = t2_proc or (PROCESS_INSTANCE_ID if t2_decision_mono_ns else None)
            ord_obj.t2_host = t2_host or (HOST_ID if t2_decision_mono_ns else None)
            if client_order_id:
                self._orders_by_client_id[client_order_id] = ord_obj
            if broker_order_id:
                self._orders_by_broker_id[broker_order_id] = ord_obj
            return ord_obj

    def get_order(
        self,
        broker_order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> _TrackedOrderLifecycle | None:
        with self._lock:
            if broker_order_id and broker_order_id in self._orders_by_broker_id:
                return self._orders_by_broker_id[broker_order_id]
            if client_order_id and client_order_id in self._orders_by_client_id:
                return self._orders_by_client_id[client_order_id]
            return None

    def record_submission(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        limit_price: float | None,
        requested_qty: float,
        bid: float | None = None,
        ask: float | None = None,
        quote_age_ms: float | None = None,
        max_quote_age_ms: float | None = None,
        decision_price: float | None = None,
        signal_id: str | None = None,
        signal_family_id: str | None = None,
        feed_source: str = "IEX",
        broker_order_id: str | None = None,
        setup_id: str | None = None,
    ) -> _TrackedOrderLifecycle:
        clock = capture_local_clock()
        effective_max_quote_age_ms = max_quote_age_ms or DEFAULT_MAX_QUOTE_AGE_MS
        marketability = classify_marketability(
            side=side,
            order_type="limit",
            limit_price=limit_price,
            bid=bid,
            ask=ask,
            quote_age_ms=quote_age_ms,
            max_quote_age_ms=effective_max_quote_age_ms,
        )
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id, client_order_id=client_order_id)
            if not ord_obj:
                ord_obj = self.start_order(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side,
                    requested_qty=requested_qty,
                    limit_price=limit_price,
                    decision_price=decision_price,
                    signal_id=signal_id,
                    signal_family_id=signal_family_id,
                    broker_order_id=broker_order_id,
                    feed_source=feed_source,
                    max_quote_age_ms=effective_max_quote_age_ms,
                    setup_id=setup_id,
                )
            if getattr(ord_obj, "setup_id", None) is None and setup_id:
                ord_obj.setup_id = setup_id
            ord_obj.t3_submit_utc = clock["observed_time_utc"]
            ord_obj.t3_submit_mono_ns = clock["observed_mono_ns"]
            ord_obj.t3_proc = clock["process_instance_id"]
            ord_obj.t3_host = clock["host_id"]
            ord_obj.marketability_at_submit = marketability
            ord_obj.max_quote_age_ms_applied = effective_max_quote_age_ms

            # Submit-time quote observation (S)
            if bid is not None or ask is not None:
                ord_obj.s_quote_observed_utc = clock["observed_time_utc"]
                ord_obj.s_quote_observed_mono_ns = clock["observed_mono_ns"]
                ord_obj.s_quote_proc = clock["process_instance_id"]
                ord_obj.s_quote_host = clock["host_id"]

            # Canonical IMMEDIATE_AT_SUBMIT rule:
            # If order is marketable at submission, S is the first touch observation!
            if marketability == MARKETABILITY_MARKETABLE and ord_obj.s_quote_observed_mono_ns is not None:
                ord_obj.first_touch_observed_time_utc = ord_obj.s_quote_observed_utc
                ord_obj.first_touch_observed_mono_ns = ord_obj.s_quote_observed_mono_ns
                ord_obj.first_touch_proc = ord_obj.s_quote_proc
                ord_obj.first_touch_host = ord_obj.s_quote_host
                ord_obj.first_touch_source = "SUBMIT_QUOTE"
                ord_obj.touch_evidence_state = TOUCH_STATE_IMMEDIATE
                ord_obj.touch_timeline_resolution = TOUCH_RES_TICK

            if broker_order_id and not ord_obj.broker_order_id:
                ord_obj.broker_order_id = broker_order_id
                self._orders_by_broker_id[broker_order_id] = ord_obj

            self._emit_event(
                ord_obj=ord_obj,
                event_name=EVENT_SUBMIT_DISPATCHED,
                event_clock=clock,
                extra={
                    "marketability_at_submit": marketability,
                    "bid": bid,
                    "ask": ask,
                    "quote_age_ms": quote_age_ms,
                    "max_quote_age_ms_applied": effective_max_quote_age_ms,
                },
            )
            return ord_obj

    def record_rest_ack(
        self,
        broker_order_id: str,
        client_order_id: str | None = None,
        event_time_utc: str | None = None,
    ) -> _TrackedOrderLifecycle | None:
        clock = capture_local_clock()
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id, client_order_id=client_order_id)
            if not ord_obj:
                ord_obj = self.start_order(
                    client_order_id=client_order_id or broker_order_id,
                    symbol="",
                    side="",
                    broker_order_id=broker_order_id,
                )
            if broker_order_id and not ord_obj.broker_order_id:
                ord_obj.broker_order_id = broker_order_id
                self._orders_by_broker_id[broker_order_id] = ord_obj

            if event_time_utc and not ord_obj.t4e_broker_event_utc:
                ord_obj.t4e_broker_event_utc = event_time_utc

            if ord_obj.t4a_rest_ack_mono_ns is None:
                ord_obj.t4a_rest_ack_utc = clock["observed_time_utc"]
                ord_obj.t4a_rest_ack_mono_ns = clock["observed_mono_ns"]
                ord_obj.t4a_proc = clock["process_instance_id"]
                ord_obj.t4a_host = clock["host_id"]

            if ord_obj.t4_ack_mono_ns is None:
                ord_obj.t4_ack_utc = ord_obj.t4a_rest_ack_utc
                ord_obj.t4_ack_mono_ns = ord_obj.t4a_rest_ack_mono_ns
                ord_obj.t4_proc = ord_obj.t4a_proc
                ord_obj.t4_host = ord_obj.t4a_host

            self._emit_event(
                ord_obj=ord_obj,
                event_name=EVENT_BROKER_REST_ACK,
                event_clock=clock,
                event_time_utc=event_time_utc,
            )
            return ord_obj

    def record_stream_ack(
        self,
        broker_order_id: str,
        client_order_id: str | None = None,
        broker_event_time_utc: str | None = None,
    ) -> _TrackedOrderLifecycle | None:
        clock = capture_local_clock()
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id, client_order_id=client_order_id)
            if not ord_obj:
                ord_obj = self.start_order(
                    client_order_id=client_order_id or broker_order_id,
                    symbol="",
                    side="",
                    broker_order_id=broker_order_id,
                )
            if broker_order_id and not ord_obj.broker_order_id:
                ord_obj.broker_order_id = broker_order_id
                self._orders_by_broker_id[broker_order_id] = ord_obj

            if broker_event_time_utc and not ord_obj.t4e_broker_event_utc:
                ord_obj.t4e_broker_event_utc = broker_event_time_utc

            if ord_obj.t4b_stream_ack_mono_ns is None:
                ord_obj.t4b_stream_ack_utc = clock["observed_time_utc"]
                ord_obj.t4b_stream_ack_mono_ns = clock["observed_mono_ns"]
                ord_obj.t4b_proc = clock["process_instance_id"]
                ord_obj.t4b_host = clock["host_id"]

            if ord_obj.t4_ack_mono_ns is None:
                ord_obj.t4_ack_utc = ord_obj.t4b_stream_ack_utc
                ord_obj.t4_ack_mono_ns = ord_obj.t4b_stream_ack_mono_ns
                ord_obj.t4_proc = ord_obj.t4b_proc
                ord_obj.t4_host = ord_obj.t4b_host

            self._emit_event(
                ord_obj=ord_obj,
                event_name=EVENT_BROKER_STREAM_ACK,
                event_clock=clock,
                event_time_utc=broker_event_time_utc,
            )
            return ord_obj

    def record_ack(
        self,
        broker_order_id: str,
        client_order_id: str | None = None,
        event_time_utc: str | None = None,
    ) -> _TrackedOrderLifecycle | None:
        """Backward-compatible entry point for recording an ACK (defaults to REST ACK)."""
        return self.record_rest_ack(
            broker_order_id=broker_order_id,
            client_order_id=client_order_id,
            event_time_utc=event_time_utc,
        )

    def set_feed_status(
        self,
        broker_order_id: str,
        connected: bool = True,
        subscribed: bool = True,
        reconnect_gap: bool = False,
    ) -> None:
        """Update live feed health and subscription state for an order."""
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id)
            if ord_obj:
                ord_obj.feed_connected = connected
                ord_obj.feed_subscription_active = subscribed
                ord_obj.feed_reconnect_gap = reconnect_gap

    def record_quote_touch(
        self,
        broker_order_id: str,
        ask: float | None,
        bid: float | None,
        quote_time_utc: str | None = None,
        feed: str = "IEX",
    ) -> _TrackedOrderLifecycle | None:
        """Record the first time the market quote touches the order's limit price."""
        clock = capture_local_clock()
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id)
            if not ord_obj or ord_obj.is_terminal or ord_obj.t5_first_fill_mono_ns is not None:
                # Chronology guard: late touch arriving after first fill or terminal is NOT accepted
                return ord_obj

            if ord_obj.limit_price is None:
                return ord_obj

            is_touch = False
            s = ord_obj.side.lower()
            if s in ("buy", "long") and ask is not None and float(ask) <= ord_obj.limit_price:
                is_touch = True
            elif s in ("sell", "short") and bid is not None and float(bid) >= ord_obj.limit_price:
                is_touch = True

            if not is_touch:
                return ord_obj

            if ord_obj.first_quote_touch_mono_ns is None:
                ord_obj.first_quote_touch_utc = quote_time_utc or clock["observed_time_utc"]
                ord_obj.first_quote_touch_mono_ns = clock["observed_mono_ns"]
                ord_obj.first_quote_touch_proc = clock["process_instance_id"]
                ord_obj.first_quote_touch_host = clock["host_id"]

            if ord_obj.first_touch_observed_mono_ns is None:
                ord_obj.first_touch_observed_time_utc = ord_obj.first_quote_touch_utc
                ord_obj.first_touch_observed_mono_ns = ord_obj.first_quote_touch_mono_ns
                ord_obj.first_touch_proc = ord_obj.first_quote_touch_proc
                ord_obj.first_touch_host = ord_obj.first_quote_touch_host
                ord_obj.first_touch_source = "QUOTE"
                ord_obj.touch_evidence_state = TOUCH_STATE_OBSERVED
                ord_obj.touch_source_feed = feed
                ord_obj.touch_timeline_resolution = TOUCH_RES_TICK

                self._emit_event(
                    ord_obj=ord_obj,
                    event_name=EVENT_QUOTE_TOUCH,
                    event_clock=clock,
                    event_time_utc=quote_time_utc,
                    extra={"ask": ask, "bid": bid, "limit_price": ord_obj.limit_price},
                )
            return ord_obj

    def record_trade_touch(
        self,
        broker_order_id: str,
        price: float | None,
        size: float = 0.0,
        trade_time_utc: str | None = None,
        feed: str = "IEX",
    ) -> _TrackedOrderLifecycle | None:
        """Record the first time a market print trades at or through the order's limit price."""
        clock = capture_local_clock()
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id)
            if not ord_obj or ord_obj.is_terminal or ord_obj.t5_first_fill_mono_ns is not None:
                # Chronology guard: late touch arriving after first fill or terminal is NOT accepted
                return ord_obj

            if ord_obj.limit_price is None or price is None:
                return ord_obj

            is_touch = False
            s = ord_obj.side.lower()
            if s in ("buy", "long") and float(price) <= ord_obj.limit_price:
                is_touch = True
            elif s in ("sell", "short") and float(price) >= ord_obj.limit_price:
                is_touch = True

            if not is_touch:
                return ord_obj

            if ord_obj.first_trade_touch_mono_ns is None:
                ord_obj.first_trade_touch_utc = trade_time_utc or clock["observed_time_utc"]
                ord_obj.first_trade_touch_mono_ns = clock["observed_mono_ns"]
                ord_obj.first_trade_touch_proc = clock["process_instance_id"]
                ord_obj.first_trade_touch_host = clock["host_id"]

            if ord_obj.first_touch_observed_mono_ns is None:
                ord_obj.first_touch_observed_time_utc = ord_obj.first_trade_touch_utc
                ord_obj.first_touch_observed_mono_ns = ord_obj.first_trade_touch_mono_ns
                ord_obj.first_touch_proc = ord_obj.first_trade_touch_proc
                ord_obj.first_touch_host = ord_obj.first_trade_touch_host
                ord_obj.first_touch_source = "TRADE"
                ord_obj.touch_evidence_state = TOUCH_STATE_OBSERVED
                ord_obj.touch_source_feed = feed
                ord_obj.touch_timeline_resolution = TOUCH_RES_TICK

                self._emit_event(
                    ord_obj=ord_obj,
                    event_name=EVENT_TRADE_TOUCH,
                    event_clock=clock,
                    event_time_utc=trade_time_utc,
                    extra={"trade_price": price, "size": size, "limit_price": ord_obj.limit_price},
                )
            return ord_obj

    def record_bar_cross_auxiliary(
        self,
        broker_order_id: str,
        bar_high: float,
        bar_low: float,
        bar_time_utc: str | None = None,
    ) -> _TrackedOrderLifecycle | None:
        """Record auxiliary historical bar cross evidence.

        Invariant: Historical bars provide approximate range cross, but do NOT support
        exact quote/trade touch times. Exact first_touch fields remain unmutated!
        """
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id)
            if not ord_obj or ord_obj.limit_price is None:
                return ord_obj

            if bar_low <= ord_obj.limit_price <= bar_high:
                ord_obj.auxiliary_bar_cross_observed = True
                if ord_obj.touch_timeline_resolution == TOUCH_RES_UNAVAILABLE:
                    ord_obj.touch_timeline_resolution = TOUCH_RES_1M_BAR
            return ord_obj

    def record_fill(
        self,
        broker_order_id: str,
        filled_qty: float,
        price: float,
        execution_id: str | None = None,
        event_time_utc: str | None = None,
        is_complete: bool = False,
    ) -> _TrackedOrderLifecycle | None:
        clock = capture_local_clock()
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id)
            if not ord_obj:
                return None

            # Idempotency check: skip already seen execution_id
            if execution_id and execution_id in ord_obj.seen_execution_ids:
                return ord_obj
            if execution_id:
                ord_obj.seen_execution_ids.add(execution_id)

            ord_obj.filled_qty = max(ord_obj.filled_qty, float(filled_qty))
            ord_obj.fill_transitions_count += 1
            if ord_obj.average_fill_price is None or ord_obj.average_fill_price <= 0:
                ord_obj.average_fill_price = float(price) if price > 0 else None

            # First fill timestamp (T5)
            if ord_obj.t5_first_fill_mono_ns is None and ord_obj.filled_qty > 0:
                ord_obj.t5_first_fill_utc = event_time_utc or clock["observed_time_utc"]
                ord_obj.t5_first_fill_mono_ns = clock["observed_mono_ns"]
                ord_obj.t5_proc = clock["process_instance_id"]
                ord_obj.t5_host = clock["host_id"]
                ord_obj.first_fill_price = float(price) if price > 0 else None

                # Evidence state resolution upon first fill
                if ord_obj.first_touch_observed_mono_ns is None:
                    if ord_obj.feed_connected and ord_obj.feed_subscription_active and not ord_obj.feed_reconnect_gap:
                        ord_obj.touch_evidence_state = TOUCH_STATE_UNOBSERVED_GAP
                    else:
                        ord_obj.touch_evidence_state = TOUCH_STATE_UNAVAILABLE

                self._emit_event(
                    ord_obj=ord_obj,
                    event_name=EVENT_PARTIAL_FILL,
                    event_clock=clock,
                    event_time_utc=event_time_utc,
                    extra={"filled_qty": ord_obj.filled_qty, "fill_price": price, "is_first_fill": True},
                )
            else:
                self._emit_event(
                    ord_obj=ord_obj,
                    event_name=EVENT_PARTIAL_FILL,
                    event_clock=clock,
                    event_time_utc=event_time_utc,
                    extra={"filled_qty": ord_obj.filled_qty, "fill_price": price, "is_first_fill": False},
                )

            # Complete fill (T6)
            if is_complete or (ord_obj.requested_qty > 0 and ord_obj.filled_qty >= ord_obj.requested_qty):
                if ord_obj.t6_complete_fill_mono_ns is None:
                    ord_obj.t6_complete_fill_utc = event_time_utc or clock["observed_time_utc"]
                    ord_obj.t6_complete_fill_mono_ns = clock["observed_mono_ns"]
                    ord_obj.t6_proc = clock["process_instance_id"]
                    ord_obj.t6_host = clock["host_id"]
                ord_obj.is_terminal = True
                ord_obj.terminal_status = "filled"
                self._emit_event(
                    ord_obj=ord_obj,
                    event_name=EVENT_COMPLETE_FILL,
                    event_clock=clock,
                    event_time_utc=event_time_utc,
                    extra={"filled_qty": ord_obj.filled_qty, "avg_price": ord_obj.average_fill_price},
                )
                self.emit_latency_summary(ord_obj)

            return ord_obj

    def record_terminal(
        self,
        broker_order_id: str,
        terminal_status: str,
        reason: str | None = None,
    ) -> _TrackedOrderLifecycle | None:
        clock = capture_local_clock()
        with self._lock:
            ord_obj = self.get_order(broker_order_id=broker_order_id)
            if not ord_obj or ord_obj.is_terminal:
                return ord_obj

            ord_obj.is_terminal = True
            ord_obj.terminal_status = terminal_status
            ord_obj.classification_reason = reason or ord_obj.classification_reason
            ev_name = {
                "canceled": EVENT_ORDER_CANCELED,
                "cancelled": EVENT_ORDER_CANCELED,
                "rejected": EVENT_ORDER_REJECTED,
                "expired": EVENT_ORDER_EXPIRED,
                "replaced": EVENT_ORDER_REPLACED,
            }.get(terminal_status.lower(), "ORDER_TERMINAL")

            self._emit_event(
                ord_obj=ord_obj,
                event_name=ev_name,
                event_clock=clock,
                extra={"status": terminal_status, "reason": reason},
            )
            self.emit_latency_summary(ord_obj)
            return ord_obj

    def _emit_event(
        self,
        ord_obj: _TrackedOrderLifecycle,
        event_name: str,
        event_clock: dict,
        event_time_utc: str | None = None,
        extra: dict | None = None,
    ) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "record_type": RECORD_TYPE_EVENT,
            "session_id": ord_obj.session_id,
            "signal_id": ord_obj.signal_id,
            "signal_family_id": ord_obj.signal_family_id,
            "setup_id": getattr(ord_obj, "setup_id", None),
            "client_order_id": ord_obj.client_order_id,
            "broker_order_id": ord_obj.broker_order_id,
            "symbol": ord_obj.symbol,
            "side": ord_obj.side,
            "event": event_name,
            "event_time_utc": event_time_utc or event_clock["observed_time_utc"],
            "observed_time_utc": event_clock["observed_time_utc"],
            "observed_mono_ns": event_clock["observed_mono_ns"],
            "process_instance_id": event_clock.get("process_instance_id", PROCESS_INSTANCE_ID),
            "host_id": event_clock.get("host_id", HOST_ID),
            "feed": ord_obj.feed_source,
        }
        if extra:
            payload.update(extra)
        _write_audit_line(payload)

    def emit_latency_summary(self, ord_obj: _TrackedOrderLifecycle) -> dict:
        """Calculate durations with clock-domain protection and emit Schema 2.1 summary."""
        def _calc_ms(
            start_ns: int | None,
            end_ns: int | None,
            start_proc: str | None,
            end_proc: str | None,
            start_host: str | None,
            end_host: str | None,
            start_utc: str | None = None,
            end_utc: str | None = None,
        ) -> int | None:
            # Monotonic calculation is ONLY valid within identical process instance and identical host
            if (
                start_ns is not None
                and end_ns is not None
                and start_proc is not None
                and end_proc is not None
                and start_proc == end_proc
                and start_host == end_host
            ):
                diff_ns = end_ns - start_ns
                if diff_ns < 0:
                    # Contract invariant: Negative duration -> NULL -> never clamp to zero!
                    return None
                return int(diff_ns // 1_000_000)

            # Cross-process restart or cross-host: monotonic invalid, fallback to UTC delta
            if start_utc and end_utc:
                try:
                    t0 = _dt.datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
                    t1 = _dt.datetime.fromisoformat(end_utc.replace("Z", "+00:00"))
                    diff_s = (t1 - t0).total_seconds()
                    if diff_s < 0:
                        # Negative duration -> NULL
                        return None
                    return int(diff_s * 1000.0)
                except Exception:
                    return None
            return None

        lat_t3_t2 = _calc_ms(
            ord_obj.t2_decision_mono_ns, ord_obj.t3_submit_mono_ns,
            ord_obj.t2_proc, ord_obj.t3_proc,
            ord_obj.t2_host, ord_obj.t3_host,
            ord_obj.t2_decision_utc, ord_obj.t3_submit_utc,
        )
        lat_t4a_t3 = _calc_ms(
            ord_obj.t3_submit_mono_ns, ord_obj.t4a_rest_ack_mono_ns,
            ord_obj.t3_proc, ord_obj.t4a_proc,
            ord_obj.t3_host, ord_obj.t4a_host,
            ord_obj.t3_submit_utc, ord_obj.t4a_rest_ack_utc,
        )
        lat_t4b_t3 = _calc_ms(
            ord_obj.t3_submit_mono_ns, ord_obj.t4b_stream_ack_mono_ns,
            ord_obj.t3_proc, ord_obj.t4b_proc,
            ord_obj.t3_host, ord_obj.t4b_host,
            ord_obj.t3_submit_utc, ord_obj.t4b_stream_ack_utc,
        )
        lat_t4_t3 = _calc_ms(
            ord_obj.t3_submit_mono_ns, ord_obj.t4_ack_mono_ns,
            ord_obj.t3_proc, ord_obj.t4_proc,
            ord_obj.t3_host, ord_obj.t4_host,
            ord_obj.t3_submit_utc, ord_obj.t4_ack_utc,
        )
        lat_t5_t4 = _calc_ms(
            ord_obj.t4_ack_mono_ns, ord_obj.t5_first_fill_mono_ns,
            ord_obj.t4_proc, ord_obj.t5_proc,
            ord_obj.t4_host, ord_obj.t5_host,
            ord_obj.t4_ack_utc, ord_obj.t5_first_fill_utc,
        )
        lat_t6_t5 = _calc_ms(
            ord_obj.t5_first_fill_mono_ns, ord_obj.t6_complete_fill_mono_ns,
            ord_obj.t5_proc, ord_obj.t6_proc,
            ord_obj.t5_host, ord_obj.t6_host,
            ord_obj.t5_first_fill_utc, ord_obj.t6_complete_fill_utc,
        )

        # ── Gate 1 Phase B Decomposed Durations ─────────────────────────────
        # Canonical Rule:
        # IMMEDIATE_AT_SUBMIT:
        #   pre_touch_observed_wait_ms = 0
        #   post_touch_observed_wait_ms = T5_observed - S_quote_observed
        # TOUCH_OBSERVED:
        #   pre_touch_observed_wait_ms = first_touch - T4
        #   post_touch_observed_wait_ms = T5 - first_touch
        # All other states (UNOBSERVED_FEED_GAP, TOUCH_TIMELINE_UNAVAILABLE, etc.):
        #   pre_touch_observed_wait_ms = None
        #   post_touch_observed_wait_ms = None
        if ord_obj.touch_evidence_state == TOUCH_STATE_IMMEDIATE:
            pre_touch_observed_wait_ms = 0
            post_touch_observed_wait_ms = _calc_ms(
                ord_obj.s_quote_observed_mono_ns, ord_obj.t5_first_fill_mono_ns,
                ord_obj.s_quote_proc, ord_obj.t5_proc,
                ord_obj.s_quote_host, ord_obj.t5_host,
                ord_obj.s_quote_observed_utc, ord_obj.t5_first_fill_utc,
            )
        elif (
            ord_obj.touch_evidence_state == TOUCH_STATE_OBSERVED
            and (ord_obj.first_touch_observed_mono_ns is not None or ord_obj.first_touch_observed_time_utc is not None)
        ):
            pre_touch_observed_wait_ms = _calc_ms(
                ord_obj.t4_ack_mono_ns, ord_obj.first_touch_observed_mono_ns,
                ord_obj.t4_proc, ord_obj.first_touch_proc,
                ord_obj.t4_host, ord_obj.first_touch_host,
                ord_obj.t4_ack_utc, ord_obj.first_touch_observed_time_utc,
            )
            post_touch_observed_wait_ms = _calc_ms(
                ord_obj.first_touch_observed_mono_ns, ord_obj.t5_first_fill_mono_ns,
                ord_obj.first_touch_proc, ord_obj.t5_proc,
                ord_obj.first_touch_host, ord_obj.t5_host,
                ord_obj.first_touch_observed_time_utc, ord_obj.t5_first_fill_utc,
            )
        else:
            pre_touch_observed_wait_ms = None
            post_touch_observed_wait_ms = None

        # Classification
        classification = "UNATTRIBUTED_ACCEPTED_TO_FILL_WAIT"
        if ord_obj.marketability_at_submit == MARKETABILITY_MARKETABLE:
            classification = "MARKETABLE_FILL"
        elif ord_obj.marketability_at_submit == MARKETABILITY_PASSIVE:
            classification = "PASSIVE_OR_UNATTRIBUTED"

        summary = {
            "schema_version": SCHEMA_VERSION,
            "record_type": RECORD_TYPE_SUMMARY,
            "session_id": ord_obj.session_id,
            "signal_id": ord_obj.signal_id,
            "signal_family_id": ord_obj.signal_family_id,
            "setup_id": getattr(ord_obj, "setup_id", None),
            "client_order_id": ord_obj.client_order_id,
            "broker_order_id": ord_obj.broker_order_id,
            "symbol": ord_obj.symbol,
            "side": ord_obj.side,
            "order_type": ord_obj.order_type,
            "requested_qty": ord_obj.requested_qty,
            "filled_qty": ord_obj.filled_qty,
            "marketability_at_submit": ord_obj.marketability_at_submit,
            "marketability_scope": ord_obj.marketability_scope,
            "nbbo_verified": ord_obj.nbbo_verified,
            "max_quote_age_ms_applied": ord_obj.max_quote_age_ms_applied,

            "process_instance_id": ord_obj.process_instance_id,
            "host_id": ord_obj.host_id,

            # Observation Timestamps (UTC ISO)
            "t0_source_event_utc": ord_obj.t0_source_event_utc,
            "t1_ingest_utc": ord_obj.t1_ingest_utc,
            "t2_decision_utc": ord_obj.t2_decision_utc,
            "t3_submit_utc": ord_obj.t3_submit_utc,
            "s_quote_observed_utc": ord_obj.s_quote_observed_utc,
            "t4a_rest_ack_utc": ord_obj.t4a_rest_ack_utc,
            "t4b_stream_ack_utc": ord_obj.t4b_stream_ack_utc,
            "t4e_broker_event_utc": ord_obj.t4e_broker_event_utc,
            "t4_ack_utc": ord_obj.t4_ack_utc,
            "t5_first_fill_utc": ord_obj.t5_first_fill_utc,
            "t6_complete_fill_utc": ord_obj.t6_complete_fill_utc,

            # Touch Timestamps
            "first_quote_touch_utc": ord_obj.first_quote_touch_utc,
            "first_trade_touch_utc": ord_obj.first_trade_touch_utc,
            "first_touch_observed_time_utc": ord_obj.first_touch_observed_time_utc,
            "first_touch_source": ord_obj.first_touch_source,

            # Gate 1 Phase A Durations
            "latency_t3_t2_ms": lat_t3_t2,
            "latency_t4a_t3_ms": lat_t4a_t3,
            "latency_t4b_t3_ms": lat_t4b_t3,
            "latency_t4_t3_ms": lat_t4_t3,
            "latency_t5_t4_ms": lat_t5_t4,
            "latency_t6_t5_ms": lat_t6_t5,

            # Gate 1 Phase B Decomposed Durations (Locked Non-Causal Names)
            "pre_touch_observed_wait_ms": pre_touch_observed_wait_ms,
            "post_touch_observed_wait_ms": post_touch_observed_wait_ms,

            # Backward-compatibility aliases for Phase A consumers
            "passive_wait_ms": pre_touch_observed_wait_ms,
            "marketable_to_first_fill_ms": post_touch_observed_wait_ms,
            "first_marketable_observed_utc": ord_obj.first_touch_observed_time_utc,

            # Feed & Evidence Quality
            "touch_evidence_state": ord_obj.touch_evidence_state,
            "touch_source_feed": ord_obj.touch_source_feed,
            "touch_timeline_resolution": ord_obj.touch_timeline_resolution,
            "auxiliary_bar_cross_observed": ord_obj.auxiliary_bar_cross_observed,

            # Non-Causal Forensic Guardrails
            "execution_causation_established": False,
            "venue_queue_causation_established": False,
            "feed_omission_proven": False,
            "causal_claim_permitted": False,

            "classification": classification,
            "classification_reason": ord_obj.classification_reason or "QUOTE_TRADE_TIMELINE_UNAVAILABLE",
        }
        _write_audit_line(summary)
        return summary


# Global singleton instance
lifecycle_tracker = OrderLifecycleTracker()


def _write_audit_line(payload: dict) -> None:
    """Safely append one JSON line to audit file."""
    try:
        audit_file = get_audit_file()
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        with audit_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=_safe, sort_keys=True) + "\n")
    except Exception:
        try:
            logger.debug("order_audit._write_audit_line failed", exc_info=True)
        except Exception:
            pass


def log_event(stage: str, **fields) -> None:
    """Append one audit event. Fully backward-compatible. Never raises into caller."""
    try:
        record = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "stage": str(stage),
        }
        for key, value in fields.items():
            record[key] = _safe(value)
        _write_audit_line(record)
    except Exception:
        try:
            logger.debug("order_audit.log_event failed for stage=%s", stage, exc_info=True)
        except Exception:
            pass


def _safe(value):
    """Coerce a value into something json.dumps can handle without raising."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


def read_events(limit: int | None = None) -> list:
    """Read back audit events (most-recent-last). Returns [] if no file yet."""
    try:
        audit_file = get_audit_file()
        if not audit_file.exists():
            return []
        lines = audit_file.read_text(encoding="utf-8").splitlines()
        if limit is not None and limit >= 0:
            lines = lines[-limit:]
        events = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                events.append(json.loads(ln))
            except Exception:
                continue
        return events
    except Exception:
        return []

