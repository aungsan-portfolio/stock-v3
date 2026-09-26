"""
Unit tests for Gate 1 Phase B: Quote & Trade Touch Telemetry & Replay Fixture Layer.

Verifies:
  1. Quote touch detection for Buy limit (ask <= Plim) and Sell limit (bid >= Plim).
  2. Trade touch detection for Buy limit (trade <= Plim) and Sell limit (trade >= Plim).
  3. IMMEDIATE_AT_SUBMIT semantics:
     - Captures submit quote observation S.
     - pre_touch_observed_wait_ms = 0.
     - post_touch_observed_wait_ms = T5 - S (strictly from S, never conflated with T4).
  4. TOUCH_OBSERVED decomposed durations:
     - pre_touch_observed_wait_ms = T_touch - T4.
     - post_touch_observed_wait_ms = T5 - T_touch.
     - Sum equals latency_t5_t4_ms.
  5. UNOBSERVED_FEED_GAP:
     - Active feed, fill arrives without prior touch.
  6. Disconnected / Unavailable feed:
     - Resolves to TOUCH_TIMELINE_UNAVAILABLE, pre/post wait are None.
  7. Chronology guard:
     - Touch arriving after T5 is rejected and does not mutate first_touch.
  8. Auxiliary 1M bar cross:
     - Sets auxiliary_bar_cross_observed = True without populating exact touch timestamps.
  9. Clock-domain and negative duration protection:
     - Negative duration strictly returns None (never clamped to 0).
 10. Non-causal forensic integrity:
     - Zero prohibited causal keys (queue_wait_ms, venue_queue_delay_ms, liquidity_delay_ms).
     - causal_claim_permitted is strictly False.
"""
import pytest
from datetime import datetime, timezone
import order_audit
from order_audit import (
    OrderLifecycleTracker,
    MARKETABILITY_MARKETABLE,
    MARKETABILITY_PASSIVE,
    TOUCH_STATE_OBSERVED,
    TOUCH_STATE_IMMEDIATE,
    TOUCH_STATE_UNOBSERVED_GAP,
    TOUCH_STATE_UNAVAILABLE,
    TOUCH_RES_TICK,
    TOUCH_RES_1M_BAR,
    TOUCH_RES_UNAVAILABLE,
    SCHEMA_VERSION,
)


def test_quote_touch_detection_buy_and_sell():
    tracker = OrderLifecycleTracker()

    # Buy Limit Order: limit = 100.00
    ord_buy = tracker.start_order(
        client_order_id="test-buy-001",
        symbol="AAPL",
        side="buy",
        limit_price=100.00,
        broker_order_id="broker-buy-001",
    )
    # 1. Ask is above limit -> No touch
    tracker.record_quote_touch(broker_order_id="broker-buy-001", ask=100.05, bid=99.95)
    assert ord_buy.first_quote_touch_utc is None
    assert ord_buy.touch_evidence_state == TOUCH_STATE_UNAVAILABLE

    # 2. Ask touches limit -> Touch observed!
    tracker.record_quote_touch(broker_order_id="broker-buy-001", ask=100.00, bid=99.98)
    assert ord_buy.first_quote_touch_utc is not None
    assert ord_buy.first_touch_source == "QUOTE"
    assert ord_buy.touch_evidence_state == TOUCH_STATE_OBSERVED
    assert ord_buy.touch_timeline_resolution == TOUCH_RES_TICK

    # Sell Limit Order: limit = 200.00
    ord_sell = tracker.start_order(
        client_order_id="test-sell-001",
        symbol="MSFT",
        side="sell",
        limit_price=200.00,
        broker_order_id="broker-sell-001",
    )
    # 1. Bid is below limit -> No touch
    tracker.record_quote_touch(broker_order_id="broker-sell-001", ask=200.05, bid=199.95)
    assert ord_sell.first_quote_touch_utc is None
    assert ord_sell.touch_evidence_state == TOUCH_STATE_UNAVAILABLE

    # 2. Bid touches limit -> Touch observed!
    tracker.record_quote_touch(broker_order_id="broker-sell-001", ask=200.02, bid=200.00)
    assert ord_sell.first_quote_touch_utc is not None
    assert ord_sell.first_touch_source == "QUOTE"
    assert ord_sell.touch_evidence_state == TOUCH_STATE_OBSERVED


def test_trade_touch_detection_buy_and_sell():
    tracker = OrderLifecycleTracker()

    # Buy Limit Order: limit = 50.00
    ord_buy = tracker.start_order(
        client_order_id="trade-buy-001",
        symbol="AMD",
        side="buy",
        limit_price=50.00,
        broker_order_id="broker-tbuy-001",
    )
    # Print at 50.05 -> No touch for buy limit
    tracker.record_trade_touch(broker_order_id="broker-tbuy-001", price=50.05, size=100)
    assert ord_buy.first_trade_touch_utc is None

    # Print at 49.99 (trades through) -> Touch observed
    tracker.record_trade_touch(broker_order_id="broker-tbuy-001", price=49.99, size=200)
    assert ord_buy.first_trade_touch_utc is not None
    assert ord_buy.first_touch_source == "TRADE"
    assert ord_buy.touch_evidence_state == TOUCH_STATE_OBSERVED

    # Sell Limit Order: limit = 50.00
    ord_sell = tracker.start_order(
        client_order_id="trade-sell-001",
        symbol="AMD",
        side="sell",
        limit_price=50.00,
        broker_order_id="broker-tsell-001",
    )
    # Print at 49.95 -> No touch for sell limit
    tracker.record_trade_touch(broker_order_id="broker-tsell-001", price=49.95, size=100)
    assert ord_sell.first_trade_touch_utc is None

    # Print at 50.01 (trades through) -> Touch observed
    tracker.record_trade_touch(broker_order_id="broker-tsell-001", price=50.01, size=100)
    assert ord_sell.first_trade_touch_utc is not None
    assert ord_sell.first_touch_source == "TRADE"
    assert ord_sell.touch_evidence_state == TOUCH_STATE_OBSERVED


def test_immediate_at_submit_semantics():
    """Verify IMMEDIATE_AT_SUBMIT rule: S is captured, pre_touch=0, post_touch=T5 - S."""
    tracker = OrderLifecycleTracker()
    client_id = "imm-client-001"
    broker_id = "imm-broker-001"

    # Marketable buy order: limit 100.10 >= ask 100.05
    ord_obj = tracker.record_submission(
        client_order_id=client_id,
        symbol="NVDA",
        side="buy",
        limit_price=100.10,
        requested_qty=10.0,
        bid=100.00,
        ask=100.05,
        quote_age_ms=10.0,
        broker_order_id=broker_id,
    )

    assert ord_obj.marketability_at_submit == MARKETABILITY_MARKETABLE
    assert ord_obj.touch_evidence_state == TOUCH_STATE_IMMEDIATE
    assert ord_obj.first_touch_source == "SUBMIT_QUOTE"
    assert ord_obj.s_quote_observed_mono_ns is not None
    s_ns = ord_obj.s_quote_observed_mono_ns

    # Simulate ACK at S + 8ms
    ack_ns = s_ns + 8_000_000
    ord_obj.t4_ack_mono_ns = ack_ns
    ord_obj.t4_ack_utc = datetime.now(timezone.utc).isoformat()
    ord_obj.t4_proc = ord_obj.s_quote_proc
    ord_obj.t4_host = ord_obj.s_quote_host

    # Simulate Fill at S + 25ms (T5 - S = 25ms, T5 - T4 = 17ms)
    t5_ns = s_ns + 25_000_000
    ord_obj.t5_first_fill_mono_ns = t5_ns
    ord_obj.t5_first_fill_utc = datetime.now(timezone.utc).isoformat()
    ord_obj.t5_proc = ord_obj.s_quote_proc
    ord_obj.t5_host = ord_obj.s_quote_host
    ord_obj.filled_qty = 10.0
    ord_obj.is_terminal = True
    ord_obj.terminal_status = "filled"

    summary = tracker.emit_latency_summary(ord_obj)

    assert summary["pre_touch_observed_wait_ms"] == 0
    # post_touch must be measured from S (25ms), strictly NOT from T4 (17ms)
    assert summary["post_touch_observed_wait_ms"] == 25
    assert summary["latency_t4_t3_ms"] == 8
    assert summary["latency_t5_t4_ms"] == 17
    assert summary["touch_evidence_state"] == TOUCH_STATE_IMMEDIATE


def test_touch_observed_decomposition():
    """Verify TOUCH_OBSERVED decomposition: pre_touch = T_touch - T4, post_touch = T5 - T_touch."""
    tracker = OrderLifecycleTracker()
    client_id = "decomp-client-001"
    broker_id = "decomp-broker-001"

    # Passive buy order: limit 100.00 < ask 100.05
    ord_obj = tracker.record_submission(
        client_order_id=client_id,
        symbol="SPY",
        side="buy",
        limit_price=100.00,
        requested_qty=10.0,
        bid=99.95,
        ask=100.05,
        quote_age_ms=15.0,
        broker_order_id=broker_id,
    )
    assert ord_obj.marketability_at_submit == MARKETABILITY_PASSIVE
    t3_ns = ord_obj.t3_submit_mono_ns

    # ACK at T3 + 5ms
    t4_ns = t3_ns + 5_000_000
    ord_obj.t4_ack_mono_ns = t4_ns
    ord_obj.t4_ack_utc = datetime.now(timezone.utc).isoformat()
    ord_obj.t4_proc = ord_obj.t3_proc
    ord_obj.t4_host = ord_obj.t3_host

    # Touch observed at T4 + 120,000ms (120s later)
    touch_ns = t4_ns + 120_000_000_000
    ord_obj.first_quote_touch_mono_ns = touch_ns
    ord_obj.first_quote_touch_utc = datetime.now(timezone.utc).isoformat()
    ord_obj.first_quote_touch_proc = ord_obj.t3_proc
    ord_obj.first_quote_touch_host = ord_obj.t3_host

    ord_obj.first_touch_observed_mono_ns = touch_ns
    ord_obj.first_touch_observed_time_utc = ord_obj.first_quote_touch_utc
    ord_obj.first_touch_proc = ord_obj.t3_proc
    ord_obj.first_touch_host = ord_obj.t3_host
    ord_obj.first_touch_source = "QUOTE"
    ord_obj.touch_evidence_state = TOUCH_STATE_OBSERVED

    # Fill arrives at T_touch + 3,500ms (3.5s later)
    t5_ns = touch_ns + 3_500_000_000
    ord_obj.t5_first_fill_mono_ns = t5_ns
    ord_obj.t5_first_fill_utc = datetime.now(timezone.utc).isoformat()
    ord_obj.t5_proc = ord_obj.t3_proc
    ord_obj.t5_host = ord_obj.t3_host
    ord_obj.filled_qty = 10.0
    ord_obj.is_terminal = True
    ord_obj.terminal_status = "filled"

    summary = tracker.emit_latency_summary(ord_obj)

    assert summary["latency_t4_t3_ms"] == 5
    assert summary["latency_t5_t4_ms"] == 123500
    assert summary["pre_touch_observed_wait_ms"] == 120000
    assert summary["post_touch_observed_wait_ms"] == 3500
    assert summary["pre_touch_observed_wait_ms"] + summary["post_touch_observed_wait_ms"] == summary["latency_t5_t4_ms"]


def test_unobserved_feed_gap_resolution():
    """Verify that a connected healthy feed without touch before fill resolves to UNOBSERVED_FEED_GAP."""
    tracker = OrderLifecycleTracker()
    ord_obj = tracker.start_order(
        client_order_id="gap-001",
        symbol="TSLA",
        side="buy",
        limit_price=200.00,
        requested_qty=5.0,
        broker_order_id="broker-gap-001",
    )
    # Feed is connected, subscribed, no reconnect gap
    tracker.set_feed_status(
        broker_order_id="broker-gap-001",
        connected=True,
        subscribed=True,
        reconnect_gap=False,
    )
    assert ord_obj.feed_connected is True
    assert ord_obj.feed_subscription_active is True
    assert ord_obj.feed_reconnect_gap is False

    # Order fills without any quote or trade touch recorded
    tracker.record_fill(
        broker_order_id="broker-gap-001",
        filled_qty=5.0,
        price=200.00,
        execution_id="gap-exec-1",
        is_complete=True,
    )

    assert ord_obj.touch_evidence_state == TOUCH_STATE_UNOBSERVED_GAP
    summary = tracker.emit_latency_summary(ord_obj)
    assert summary["touch_evidence_state"] == TOUCH_STATE_UNOBSERVED_GAP
    assert summary["pre_touch_observed_wait_ms"] is None
    assert summary["post_touch_observed_wait_ms"] is None


def test_disconnected_feed_unavailable_resolution():
    """Verify that an inactive/disconnected feed resolves to TOUCH_TIMELINE_UNAVAILABLE."""
    tracker = OrderLifecycleTracker()
    ord_obj = tracker.start_order(
        client_order_id="unavail-001",
        symbol="TSLA",
        side="buy",
        limit_price=200.00,
        requested_qty=5.0,
        broker_order_id="broker-unavail-001",
    )
    # Feed is NOT connected
    tracker.set_feed_status(
        broker_order_id="broker-unavail-001",
        connected=False,
        subscribed=False,
    )

    # Order fills
    tracker.record_fill(
        broker_order_id="broker-unavail-001",
        filled_qty=5.0,
        price=200.00,
        execution_id="unavail-exec-1",
        is_complete=True,
    )

    assert ord_obj.touch_evidence_state == TOUCH_STATE_UNAVAILABLE
    summary = tracker.emit_latency_summary(ord_obj)
    assert summary["touch_evidence_state"] == TOUCH_STATE_UNAVAILABLE
    assert summary["pre_touch_observed_wait_ms"] is None
    assert summary["post_touch_observed_wait_ms"] is None


def test_event_chronology_guard():
    """Verify that late quote/trade touches arriving after T5 are rejected and do not mutate first_touch."""
    tracker = OrderLifecycleTracker()
    ord_obj = tracker.start_order(
        client_order_id="chrono-001",
        symbol="META",
        side="buy",
        limit_price=500.00,
        requested_qty=5.0,
        broker_order_id="broker-chrono-001",
    )

    # First fill occurs at T5
    tracker.record_fill(
        broker_order_id="broker-chrono-001",
        filled_qty=2.0,
        price=500.00,
        execution_id="chrono-exec-1",
        is_complete=False,
    )
    assert ord_obj.t5_first_fill_mono_ns is not None
    assert ord_obj.first_touch_observed_mono_ns is None

    # Late quote touch arrives AFTER fill
    tracker.record_quote_touch(
        broker_order_id="broker-chrono-001",
        ask=499.90,
        bid=499.80,
    )
    # Must be rejected by chronology guard
    assert ord_obj.first_quote_touch_utc is None
    assert ord_obj.first_touch_observed_mono_ns is None

    # Late trade touch arrives AFTER fill
    tracker.record_trade_touch(
        broker_order_id="broker-chrono-001",
        price=499.90,
        size=100,
    )
    assert ord_obj.first_trade_touch_utc is None
    assert ord_obj.first_touch_observed_mono_ns is None


def test_auxiliary_bar_cross_evidence():
    """Verify that historical bar cross sets auxiliary flag but leaves exact touch timestamps as None."""
    tracker = OrderLifecycleTracker()
    ord_obj = tracker.start_order(
        client_order_id="bar-001",
        symbol="AMZN",
        side="buy",
        limit_price=180.00,
        broker_order_id="broker-bar-001",
    )

    # 1-minute bar: Low 179.50, High 180.50 (crosses 180.00)
    tracker.record_bar_cross_auxiliary(
        broker_order_id="broker-bar-001",
        bar_high=180.50,
        bar_low=179.50,
        bar_time_utc="2026-09-18T14:22:00+00:00",
    )

    assert ord_obj.auxiliary_bar_cross_observed is True
    assert ord_obj.touch_timeline_resolution == TOUCH_RES_1M_BAR

    # Exact microsecond/millisecond touch timestamps MUST remain None
    assert ord_obj.first_quote_touch_utc is None
    assert ord_obj.first_trade_touch_utc is None
    assert ord_obj.first_touch_observed_time_utc is None


def test_negative_duration_null_protection():
    """Verify that clock anomalies yielding negative durations return None, never 0."""
    tracker = OrderLifecycleTracker()
    ord_obj = tracker.start_order(
        client_order_id="neg-001",
        symbol="GOOGL",
        side="buy",
        broker_order_id="broker-neg-001",
    )

    # Submit at T3 = 1000ms
    ord_obj.t3_submit_utc = "2026-09-18T14:00:01.000000+00:00"
    ord_obj.t3_submit_mono_ns = 1_000_000_000
    ord_obj.t3_proc = "proc-1"
    ord_obj.t3_host = "host-1"

    # Corrupted / reverse ACK timestamp at T4 = 900ms (diff = -100ms)
    ord_obj.t4_ack_utc = "2026-09-18T14:00:00.900000+00:00"
    ord_obj.t4_ack_mono_ns = 900_000_000
    ord_obj.t4_proc = "proc-1"
    ord_obj.t4_host = "host-1"

    summary = tracker.emit_latency_summary(ord_obj)
    # Contract rule: negative duration must return None, NEVER clamp to 0
    assert summary["latency_t4_t3_ms"] is None


def test_absence_of_prohibited_causal_keys():
    """Verify zero prohibited causal keys and strict non-causal boolean guards."""
    tracker = OrderLifecycleTracker()
    ord_obj = tracker.start_order(
        client_order_id="causal-guard-001",
        symbol="SPY",
        side="buy",
        broker_order_id="broker-cg-001",
    )
    summary = tracker.emit_latency_summary(ord_obj)

    # Prohibited keys
    prohibited_keys = ["queue_wait_ms", "venue_queue_delay_ms", "liquidity_delay_ms"]
    for key in prohibited_keys:
        assert key not in summary, f"Prohibited causal key '{key}' found in summary!"

    # Strict non-causal boolean flags
    assert summary["execution_causation_established"] is False
    assert summary["venue_queue_causation_established"] is False
    assert summary["feed_omission_proven"] is False
    assert summary["causal_claim_permitted"] is False
