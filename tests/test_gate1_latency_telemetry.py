"""
Unit tests for Gate 1: Deterministic Infrastructure Telemetry & Latency Decomposition.
Verifies:
  - Dual-clock normalization (UTC ISO wall clock + monotonic ns)
  - Order marketability classification state machine
  - Lifecycle state machine (T3 submit -> T4 ACK -> T5 first fill -> T6 complete fill)
  - Idempotency & duplicate event deduplication
  - Non-conflation of passive wait and execution latency
  - Null vs zero semantics (missing timestamps serialize to null, not 0)
"""
import pytest
import time
from datetime import datetime, timezone
import order_audit
from order_audit import (
    classify_marketability,
    capture_local_clock,
    OrderLifecycleTracker,
    MARKETABILITY_MARKETABLE,
    MARKETABILITY_PASSIVE,
    MARKETABILITY_UNKNOWN_NO_QUOTE,
    MARKETABILITY_UNKNOWN_STALE_QUOTE,
    MARKETABILITY_UNKNOWN_SIDE,
    MARKETABILITY_UNKNOWN,
    SCHEMA_VERSION,
)


def test_dual_clock_capture():
    clock1 = capture_local_clock()
    time.sleep(0.01)
    clock2 = capture_local_clock()

    assert "observed_time_utc" in clock1
    assert "observed_mono_ns" in clock1
    assert isinstance(clock1["observed_mono_ns"], int)
    assert clock1["observed_mono_ns"] > 0

    # Monotonic clock must strictly advance
    assert clock2["observed_mono_ns"] > clock1["observed_mono_ns"]
    elapsed_ms = (clock2["observed_mono_ns"] - clock1["observed_mono_ns"]) // 1_000_000
    assert elapsed_ms >= 8


def test_classify_marketability_buy():
    # BUY limit >= ask -> MARKETABLE
    assert classify_marketability("BUY", "limit", 759.15, bid=759.10, ask=759.15, quote_age_ms=50.0) == MARKETABILITY_MARKETABLE
    assert classify_marketability("buy", "limit", 759.20, bid=759.10, ask=759.15, quote_age_ms=50.0) == MARKETABILITY_MARKETABLE

    # BUY limit < ask -> PASSIVE
    assert classify_marketability("BUY", "limit", 759.12, bid=759.10, ask=759.15, quote_age_ms=50.0) == MARKETABILITY_PASSIVE


def test_classify_marketability_sell():
    # SELL limit <= bid -> MARKETABLE
    assert classify_marketability("SELL", "limit", 759.10, bid=759.10, ask=759.15, quote_age_ms=50.0) == MARKETABILITY_MARKETABLE
    assert classify_marketability("sell", "limit", 759.05, bid=759.10, ask=759.15, quote_age_ms=50.0) == MARKETABILITY_MARKETABLE

    # SELL limit > bid -> PASSIVE
    assert classify_marketability("SELL", "limit", 759.12, bid=759.10, ask=759.15, quote_age_ms=50.0) == MARKETABILITY_PASSIVE


def test_classify_marketability_missing_and_stale_quotes():
    # Missing bid or ask
    assert classify_marketability("BUY", "limit", 759.12, bid=None, ask=759.15, quote_age_ms=50.0) == MARKETABILITY_UNKNOWN_NO_QUOTE
    assert classify_marketability("BUY", "limit", 759.12, bid=759.10, ask=None, quote_age_ms=50.0) == MARKETABILITY_UNKNOWN_NO_QUOTE
    assert classify_marketability("BUY", "limit", 759.12, bid=None, ask=None, quote_age_ms=50.0) == MARKETABILITY_UNKNOWN_NO_QUOTE

    # Stale quote (> 10,000ms default)
    assert classify_marketability("BUY", "limit", 759.20, bid=759.10, ask=759.15, quote_age_ms=12000.0) == MARKETABILITY_UNKNOWN_STALE_QUOTE
    assert classify_marketability("BUY", "limit", 759.20, bid=759.10, ask=759.15, quote_age_ms=None) == MARKETABILITY_UNKNOWN_STALE_QUOTE

    # Market order is always MARKETABLE
    assert classify_marketability("BUY", "market", None, bid=759.10, ask=759.15) == MARKETABILITY_MARKETABLE


def test_lifecycle_progression_and_latency_decomposition():
    tracker = OrderLifecycleTracker()
    client_id = "test-client-order-001"
    broker_id = "broker-order-001"

    # 1. Submission (T3)
    ord_obj = tracker.record_submission(
        client_order_id=client_id,
        symbol="SPY",
        side="buy",
        limit_price=759.12,
        requested_qty=5.0,
        bid=759.10,
        ask=759.15,
        quote_age_ms=45.0,
        decision_price=759.18,
        feed_source="IEX",
        broker_order_id=broker_id,
    )
    assert ord_obj.marketability_at_submit == MARKETABILITY_PASSIVE
    assert ord_obj.t3_submit_mono_ns is not None
    t3_ns = ord_obj.t3_submit_mono_ns

    # Simulate 5ms ACK delay
    fake_ack_mono = t3_ns + 5_000_000
    ord_obj.t4_ack_mono_ns = fake_ack_mono
    ord_obj.t4_ack_utc = datetime.now(timezone.utc).isoformat()
    ord_obj.t4_proc = ord_obj.t3_proc
    ord_obj.t4_host = ord_obj.t3_host

    # 2. First partial fill (T5) - 2 shares
    fake_t5_mono = fake_ack_mono + 284_705_000_000  # 284.705s
    ord_obj.t5_first_fill_mono_ns = fake_t5_mono
    ord_obj.t5_first_fill_utc = datetime.now(timezone.utc).isoformat()
    ord_obj.t5_proc = ord_obj.t3_proc
    ord_obj.t5_host = ord_obj.t3_host
    ord_obj.filled_qty = 2.0
    ord_obj.seen_execution_ids.add("exec-1")

    # 3. Duplicate event replay (must be ignored)
    tracker.record_fill(
        broker_order_id=broker_id,
        filled_qty=2.0,
        price=759.12,
        execution_id="exec-1",
    )
    assert ord_obj.filled_qty == 2.0
    assert ord_obj.t5_first_fill_mono_ns == fake_t5_mono  # Preserved

    # 4. Final fill (T6) - 5 shares (1.947s later)
    fake_t6_mono = fake_t5_mono + 1_947_000_000
    ord_obj.t6_complete_fill_mono_ns = fake_t6_mono
    ord_obj.t6_complete_fill_utc = datetime.now(timezone.utc).isoformat()
    ord_obj.t6_proc = ord_obj.t3_proc
    ord_obj.t6_host = ord_obj.t3_host
    ord_obj.filled_qty = 5.0
    ord_obj.average_fill_price = 759.116
    ord_obj.is_terminal = True
    ord_obj.terminal_status = "filled"

    summary = tracker.emit_latency_summary(ord_obj)

    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["latency_t4_t3_ms"] == 5
    assert summary["latency_t5_t4_ms"] == 284705
    assert summary["latency_t6_t5_ms"] == 1947
    assert summary["causal_claim_permitted"] is False
    assert summary["classification"] == "PASSIVE_OR_UNATTRIBUTED"

    # Missing measurements must be null, never 0
    assert summary["t0_source_event_utc"] is None
    assert summary["t1_ingest_utc"] is None
    assert summary["passive_wait_ms"] is None
    assert summary["marketable_to_first_fill_ms"] is None


def test_clock_domain_protection_restart_fallback():
    """Verify that cross-process restarts invalidate monotonic diff and fallback to UTC timestamps."""
    tracker = OrderLifecycleTracker()
    ord_obj = tracker.start_order(
        client_order_id="restart-order-001",
        symbol="QQQ",
        side="buy",
    )

    # Submission in Process A
    ord_obj.t3_submit_utc = "2026-09-18T14:00:00.000000+00:00"
    ord_obj.t3_submit_mono_ns = 1_000_000_000
    ord_obj.t3_proc = "proc-instance-A"
    ord_obj.t3_host = "trading-box-1"

    # ACK observed in Process B (after bot restart)
    # Even if monotonic ns in Process B is completely arbitrary (e.g. 500_000_000 < 1_000_000_000)
    ord_obj.t4_ack_utc = "2026-09-18T14:00:00.015000+00:00"  # +15ms
    ord_obj.t4_ack_mono_ns = 500_000_000
    ord_obj.t4_proc = "proc-instance-B"  # Different process instance!
    ord_obj.t4_host = "trading-box-1"

    summary = tracker.emit_latency_summary(ord_obj)

    # Monotonic difference must NOT be computed (which would have been negative or nonsensical).
    # It must safely fall back to the UTC delta (15ms).
    assert summary["latency_t4_t3_ms"] == 15


def test_t4a_t4b_t4e_separation():
    """Verify separate tracking of REST ACK (T4a), Stream ACK (T4b), and Broker timestamp (T4e)."""
    tracker = OrderLifecycleTracker()
    broker_id = "brk-order-ack-test"

    # Submission at T3
    ord_obj = tracker.record_submission(
        client_order_id="test-ack-001",
        symbol="SPY",
        side="buy",
        limit_price=750.0,
        requested_qty=10.0,
        broker_order_id=broker_id,
    )
    t3_ns = ord_obj.t3_submit_mono_ns

    # REST ACK arrives first (T4a, +6ms)
    rest_clock_mono = t3_ns + 6_000_000
    ord_obj.t4a_rest_ack_mono_ns = rest_clock_mono
    ord_obj.t4a_rest_ack_utc = "2026-09-18T14:00:00.006000+00:00"
    ord_obj.t4a_proc = ord_obj.t3_proc
    ord_obj.t4a_host = ord_obj.t3_host
    ord_obj.t4e_broker_event_utc = "2026-09-18T14:00:00.004000+00:00"  # T4e from broker payload

    # Canonical T4 matches first observed (T4a)
    ord_obj.t4_ack_mono_ns = rest_clock_mono
    ord_obj.t4_ack_utc = ord_obj.t4a_rest_ack_utc
    ord_obj.t4_proc = ord_obj.t3_proc
    ord_obj.t4_host = ord_obj.t3_host

    # Stream ACK arrives later (T4b, +14ms)
    stream_clock_mono = t3_ns + 14_000_000
    ord_obj.t4b_stream_ack_mono_ns = stream_clock_mono
    ord_obj.t4b_stream_ack_utc = "2026-09-18T14:00:00.014000+00:00"
    ord_obj.t4b_proc = ord_obj.t3_proc
    ord_obj.t4b_host = ord_obj.t3_host

    summary = tracker.emit_latency_summary(ord_obj)

    assert summary["latency_t4a_t3_ms"] == 6
    assert summary["latency_t4b_t3_ms"] == 14
    assert summary["latency_t4_t3_ms"] == 6
    assert summary["t4e_broker_event_utc"] == "2026-09-18T14:00:00.004000+00:00"


def test_configurable_quote_freshness_and_scope():
    """Verify configurable max quote age and scope preservation."""
    # Under 2000ms max age: 2500ms is stale
    assert classify_marketability("BUY", "limit", 750.0, bid=749.0, ask=750.0, quote_age_ms=2500.0, max_quote_age_ms=2000.0) == MARKETABILITY_UNKNOWN_STALE_QUOTE
    # Under 5000ms max age: 2500ms is acceptable
    assert classify_marketability("BUY", "limit", 750.0, bid=749.0, ask=750.0, quote_age_ms=2500.0, max_quote_age_ms=5000.0) == MARKETABILITY_MARKETABLE

    tracker = OrderLifecycleTracker()
    ord_obj = tracker.record_submission(
        client_order_id="test-freshness-001",
        symbol="SPY",
        side="buy",
        limit_price=750.0,
        requested_qty=1.0,
        bid=749.0,
        ask=750.0,
        quote_age_ms=500.0,
        max_quote_age_ms=3000.0,
        feed_source="IEX",
    )
    assert ord_obj.max_quote_age_ms_applied == 3000.0
    assert ord_obj.marketability_scope == "IEX_OBSERVED"
    assert ord_obj.nbbo_verified is False

    summary = tracker.emit_latency_summary(ord_obj)
    assert summary["max_quote_age_ms_applied"] == 3000.0
    assert summary["marketability_scope"] == "IEX_OBSERVED"
    assert summary["nbbo_verified"] is False


def test_reporter_sample_quality_semantics():
    """Verify reporter statistical quality tags (INSUFFICIENT_SAMPLE, DESCRIPTIVE_ONLY, STATISTICALLY_VALID)."""
    from tools.evaluate_latency_telemetry import compute_percentiles

    # n = 0
    p0 = compute_percentiles([])
    assert p0["n"] == 0
    assert p0["status"] == "NO_DATA"
    assert p0["p50"] is None
    assert p0["is_adequate_baseline"] is False

    # n < 20 (e.g. n = 5)
    p_small = compute_percentiles([10.0, 20.0, 30.0, 40.0, 50.0])
    assert p_small["n"] == 5
    assert p_small["status"] == "INSUFFICIENT_SAMPLE"
    assert p_small["p50"] == 30.0
    assert p_small["is_adequate_baseline"] is False

    # 20 <= n < 100 (e.g. n = 25)
    p_med = compute_percentiles([float(x) for x in range(25)])
    assert p_med["n"] == 25
    assert p_med["status"] == "LIMITED_SAMPLE"
    assert p_med["is_adequate_baseline"] is False

    # n >= 100 (e.g. n = 120)
    p_large = compute_percentiles([float(x) for x in range(120)])
    assert p_large["n"] == 120
    assert p_large["status"] == "ADEQUATE_BASELINE"
    assert p_large["is_adequate_baseline"] is True
