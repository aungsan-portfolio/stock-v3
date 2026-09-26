"""
SPY Forensic Replay Test Fixture for Gate 1 Phase B (2026-09-18 Incident).

Order df1b029f-6e5c-4a6b-a1d3-a62ccc30ad76:
  - T3: 14:20:56.111507 (Submission)
  - T4: 14:20:56.116499 (REST ACK ~5ms)
  - T5: 14:25:40.821476 (First partial fill ~284,705ms)
  - T6: 14:25:42.768157 (Complete fill ~1,947ms)

Verifies:
  1. Historical baseline without tick feed correctly yields:
     - touch_evidence_state == TOUCH_STATE_UNAVAILABLE
     - pre_touch_observed_wait_ms is None
     - post_touch_observed_wait_ms is None
     - causal_claim_permitted is False
     - classification == UNATTRIBUTED_ACCEPTED_TO_FILL_WAIT
  2. Dual Counterfactual Replay proving mathematical indeterminacy:
     - Scenario A (Market away until 1s before fill): pre_touch = 283.7s, post_touch = 1.0s
     - Scenario B (Market at limit 1s after ACK): pre_touch = 1.0s, post_touch = 283.7s
     - Both produce identical T5 - T4 = 284.7s.
     - Proves that without tick touch telemetry, claiming "queue wait" is mathematically ungrounded.
  3. Auxiliary 1M bar cross does NOT fabricate microsecond touch timestamps.
  4. Complete feed gap counterfactual yields TOUCH_STATE_UNOBSERVED_GAP with null decomposed durations.
"""
import pytest
from datetime import datetime, timezone
import order_audit
from order_audit import (
    OrderLifecycleTracker,
    TOUCH_STATE_OBSERVED,
    TOUCH_STATE_IMMEDIATE,
    TOUCH_STATE_UNOBSERVED_GAP,
    TOUCH_STATE_UNAVAILABLE,
    TOUCH_RES_TICK,
    TOUCH_RES_1M_BAR,
    TOUCH_RES_UNAVAILABLE,
    SCHEMA_VERSION,
)


def _setup_base_spy_order(tracker: OrderLifecycleTracker) -> order_audit._TrackedOrderLifecycle:
    order_id = "df1b029f-6e5c-4a6b-a1d3-a62ccc30ad76"
    t3_str = "2026-09-18T14:20:56.111507+00:00"
    t4_str = "2026-09-18T14:20:56.116499+00:00"

    ord_obj = tracker.record_submission(
        client_order_id="spy_daytrade_20260918_01",
        symbol="SPY",
        side="buy",
        limit_price=759.12,
        requested_qty=5.0,
        bid=None,
        ask=None,
        quote_age_ms=None,
        broker_order_id=order_id,
        feed_source="IEX",
    )
    ord_obj.t3_submit_utc = t3_str
    tracker.record_rest_ack(
        broker_order_id=order_id,
        client_order_id="spy_daytrade_20260918_01",
        event_time_utc=t4_str,
    )
    ord_obj.t4_ack_utc = t4_str
    ord_obj.t4a_rest_ack_utc = t4_str

    # Clear monotonic markers for pure UTC timestamp replay
    ord_obj.t3_submit_mono_ns = None
    ord_obj.t4a_rest_ack_mono_ns = None
    ord_obj.t4_ack_mono_ns = None
    return ord_obj


def test_spy_historical_baseline_unattributed():
    """Historical baseline: Free IEX feed lacked tick touches -> strictly unattributed."""
    tracker = OrderLifecycleTracker()
    order_id = "df1b029f-6e5c-4a6b-a1d3-a62ccc30ad76"
    ord_obj = _setup_base_spy_order(tracker)

    t5_str = "2026-09-18T14:25:40.821476+00:00"
    t6_str = "2026-09-18T14:25:42.768157+00:00"

    tracker.record_fill(
        broker_order_id=order_id,
        filled_qty=2.0,
        price=759.12,
        execution_id="b4f6eaaa-cf3e-4c64-87a3-fe2457002b70",
        event_time_utc=t5_str,
        is_complete=False,
    )
    tracker.record_fill(
        broker_order_id=order_id,
        filled_qty=5.0,
        price=759.116,
        execution_id="702c96d0-6392-4a85-8204-ee4d1c2edfdb",
        event_time_utc=t6_str,
        is_complete=True,
    )
    ord_obj.t5_first_fill_utc = t5_str
    ord_obj.t6_complete_fill_utc = t6_str
    ord_obj.t5_first_fill_mono_ns = None
    ord_obj.t6_complete_fill_mono_ns = None

    summary = tracker.emit_latency_summary(ord_obj)

    # Macro durations
    assert summary["latency_t4_t3_ms"] in (4, 5)
    assert abs(summary["latency_t5_t4_ms"] - 284705) <= 1
    assert abs(summary["latency_t6_t5_ms"] - 1947) <= 1

    # Touch evidence state must be UNAVAILABLE
    assert summary["touch_evidence_state"] == TOUCH_STATE_UNAVAILABLE
    assert summary["pre_touch_observed_wait_ms"] is None
    assert summary["post_touch_observed_wait_ms"] is None

    # Forensic guards
    assert summary["classification"] == "UNATTRIBUTED_ACCEPTED_TO_FILL_WAIT"
    assert summary["classification_reason"] == "QUOTE_TRADE_TIMELINE_UNAVAILABLE"
    assert summary["causal_claim_permitted"] is False
    assert "QUEUE_WAIT" not in summary["classification"]


def test_spy_counterfactual_dual_hypothesis_replay():
    """Demonstrate mathematical indeterminacy: Scenario A vs Scenario B.

    Proves that without tick touch observations, an accepted-to-fill duration
    of 284.705s cannot be attributed to venue queue wait vs market-away wait.
    """
    tracker = OrderLifecycleTracker()
    t5_str = "2026-09-18T14:25:40.821476+00:00"

    # ── Scenario A: Market Away until 1s before fill ─────────────────────────
    ord_a = _setup_base_spy_order(tracker)
    ord_a.broker_order_id = "spy-scenario-a"
    # Market touches 759.12 at 14:25:39.821476 (+283,705ms after T4)
    touch_a_utc = "2026-09-18T14:25:39.821476+00:00"
    ord_a.first_touch_observed_time_utc = touch_a_utc
    ord_a.first_touch_source = "QUOTE"
    ord_a.touch_evidence_state = TOUCH_STATE_OBSERVED
    ord_a.touch_timeline_resolution = TOUCH_RES_TICK
    ord_a.t5_first_fill_utc = t5_str

    sum_a = tracker.emit_latency_summary(ord_a)
    assert abs(sum_a["pre_touch_observed_wait_ms"] - 283705) <= 1  # ~283.7s market-away
    assert sum_a["post_touch_observed_wait_ms"] == 1000            # ~1.0s queue/execution wait
    assert sum_a["pre_touch_observed_wait_ms"] + sum_a["post_touch_observed_wait_ms"] == sum_a["latency_t5_t4_ms"]

    # ── Scenario B: Market touches 1s after ACK, queues for 283.7s ──────────
    ord_b = _setup_base_spy_order(tracker)
    ord_b.broker_order_id = "spy-scenario-b"
    # Market touches 759.12 at 14:20:57.116499 (+1,000ms after T4)
    touch_b_utc = "2026-09-18T14:20:57.116499+00:00"
    ord_b.first_touch_observed_time_utc = touch_b_utc
    ord_b.first_touch_source = "QUOTE"
    ord_b.touch_evidence_state = TOUCH_STATE_OBSERVED
    ord_b.touch_timeline_resolution = TOUCH_RES_TICK
    ord_b.t5_first_fill_utc = t5_str

    sum_b = tracker.emit_latency_summary(ord_b)
    assert sum_b["pre_touch_observed_wait_ms"] == 1000             # ~1.0s market-away
    assert abs(sum_b["post_touch_observed_wait_ms"] - 283705) <= 1 # ~283.7s queue/execution wait
    assert sum_b["pre_touch_observed_wait_ms"] + sum_b["post_touch_observed_wait_ms"] == sum_b["latency_t5_t4_ms"]

    # ── Forensic Invariant Check ─────────────────────────────────────────────
    # Both scenarios have identical macro broker duration:
    assert sum_a["latency_t5_t4_ms"] == sum_b["latency_t5_t4_ms"]
    assert abs(sum_a["latency_t5_t4_ms"] - 284705) <= 1

    # But Scenario A implies post-touch wait was only 1.0s, whereas Scenario B implies 283.7s!
    # Without tick touches, choosing between A and B is impossible.
    # Therefore, classifying 2026-09-18 as "queue wait" was mathematically unproven.
    assert sum_a["causal_claim_permitted"] is False
    assert sum_b["causal_claim_permitted"] is False


def test_spy_auxiliary_bar_cross_behavior():
    """Verify that SPY 1-minute historical bar crossing limit price provides auxiliary evidence only."""
    tracker = OrderLifecycleTracker()
    order_id = "df1b029f-6e5c-4a6b-a1d3-a62ccc30ad76"
    ord_obj = _setup_base_spy_order(tracker)

    # 1-minute bar for SPY during the wait interval: Low 759.08, High 759.22 (crosses 759.12)
    tracker.record_bar_cross_auxiliary(
        broker_order_id=order_id,
        bar_high=759.22,
        bar_low=759.08,
        bar_time_utc="2026-09-18T14:22:00+00:00",
    )

    assert ord_obj.auxiliary_bar_cross_observed is True
    assert ord_obj.touch_timeline_resolution == TOUCH_RES_1M_BAR

    # Exact touch timestamps remain None
    assert ord_obj.first_quote_touch_utc is None
    assert ord_obj.first_trade_touch_utc is None
    assert ord_obj.first_touch_observed_time_utc is None

    # Latency decomposition does not fabricate touch durations
    summary = tracker.emit_latency_summary(ord_obj)
    assert summary["auxiliary_bar_cross_observed"] is True
    assert summary["touch_timeline_resolution"] == TOUCH_RES_1M_BAR
    assert summary["pre_touch_observed_wait_ms"] is None
    assert summary["post_touch_observed_wait_ms"] is None


def test_spy_unobserved_feed_gap_counterfactual():
    """If feed had been connected with tick resolution, fill without touch -> UNOBSERVED_FEED_GAP."""
    tracker = OrderLifecycleTracker()
    order_id = "df1b029f-6e5c-4a6b-a1d3-a62ccc30ad76"
    ord_obj = _setup_base_spy_order(tracker)

    # Assume SIP / connected feed was active
    tracker.set_feed_status(
        broker_order_id=order_id,
        connected=True,
        subscribed=True,
        reconnect_gap=False,
    )

    t5_str = "2026-09-18T14:25:40.821476+00:00"
    tracker.record_fill(
        broker_order_id=order_id,
        filled_qty=2.0,
        price=759.12,
        execution_id="b4f6eaaa-cf3e-4c64-87a3-fe2457002b70",
        event_time_utc=t5_str,
        is_complete=False,
    )

    assert ord_obj.touch_evidence_state == TOUCH_STATE_UNOBSERVED_GAP
    summary = tracker.emit_latency_summary(ord_obj)
    assert summary["touch_evidence_state"] == TOUCH_STATE_UNOBSERVED_GAP
    assert summary["pre_touch_observed_wait_ms"] is None
    assert summary["post_touch_observed_wait_ms"] is None
    assert summary["causal_claim_permitted"] is False
