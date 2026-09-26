"""
SPY Forensic Regression Test Fixture (2026-09-18 Incident).
Asserts exact forensic metrics for order df1b029f-6e5c-4a6b-a1d3-a62ccc30ad76:
  - ACK latency ~5ms (14:20:56.111 -> 14:20:56.116)
  - Accepted-to-first-fill duration ~284,705ms (14:20:56.116 -> 14:25:40.821)
  - Partial-to-complete duration ~1,947ms (14:25:40.821 -> 14:25:42.768)
  - Refuses to assert "queue wait" without quote evidence
  - Missing touch timeline strictly leaves touch metrics as null
  - causal_claim_permitted == False
"""
import pytest
from datetime import datetime, timezone
import order_audit
from order_audit import OrderLifecycleTracker, SCHEMA_VERSION


def test_spy_forensic_replay_accuracy():
    tracker = OrderLifecycleTracker()
    order_id = "df1b029f-6e5c-4a6b-a1d3-a62ccc30ad76"
    symbol = "SPY"
    side = "buy"

    # Exact UTC timestamps from broker_events.jsonl on 2026-09-18
    t3_str = "2026-09-18T14:20:56.111507+00:00"
    t4_str = "2026-09-18T14:20:56.116499+00:00"
    t5_str = "2026-09-18T14:25:40.821476+00:00"
    t6_str = "2026-09-18T14:25:42.768157+00:00"

    # 1. Initialize order at submission (historical: quote was unmeasured)
    ord_obj = tracker.record_submission(
        client_order_id="spy_daytrade_20260918_01",
        symbol=symbol,
        side=side,
        limit_price=759.12,
        requested_qty=5.0,
        bid=None,
        ask=None,
        quote_age_ms=None,
        broker_order_id=order_id,
        feed_source="IEX",
    )
    # Inject exact historical UTC timestamps
    ord_obj.t3_submit_utc = t3_str

    # 2. Broker ACK (T4)
    tracker.record_rest_ack(
        broker_order_id=order_id,
        client_order_id="spy_daytrade_20260918_01",
        event_time_utc=t4_str,
    )
    ord_obj.t4_ack_utc = t4_str
    ord_obj.t4a_rest_ack_utc = t4_str

    # 3. First partial fill (T5) - 2 shares @ 759.12
    tracker.record_fill(
        broker_order_id=order_id,
        filled_qty=2.0,
        price=759.12,
        execution_id="b4f6eaaa-cf3e-4c64-87a3-fe2457002b70",
        event_time_utc=t5_str,
        is_complete=False,
    )
    ord_obj.t5_first_fill_utc = t5_str

    # 4. Subsequent partial fill - 4 shares @ 759.12
    tracker.record_fill(
        broker_order_id=order_id,
        filled_qty=4.0,
        price=759.12,
        execution_id="53e90c0a-64d5-4323-8fdb-ba210117a7cf",
        event_time_utc="2026-09-18T14:25:41.803854+00:00",
        is_complete=False,
    )

    # 5. Complete fill (T6) - 5 shares @ 759.116
    summary = tracker.record_fill(
        broker_order_id=order_id,
        filled_qty=5.0,
        price=759.116,
        execution_id="702c96d0-6392-4a85-8204-ee4d1c2edfdb",
        event_time_utc=t6_str,
        is_complete=True,
    )
    ord_obj.t6_complete_fill_utc = t6_str

    # For historical replay without live process monotonic clocks, clear live monotonic ns:
    ord_obj.t3_submit_mono_ns = None
    ord_obj.t4a_rest_ack_mono_ns = None
    ord_obj.t4_ack_mono_ns = None
    ord_obj.t5_first_fill_mono_ns = None
    ord_obj.t6_complete_fill_mono_ns = None

    # Re-emit summary to evaluate calculated UTC fallback durations
    final_summary = tracker.emit_latency_summary(ord_obj)

    # ── Strict Forensic Assertions ──
    # T4 - T3: Broker ACK latency ~5ms (4.992ms rounded to 4 or 5)
    assert final_summary["latency_t4_t3_ms"] in (4, 5)
    assert final_summary["latency_t4a_t3_ms"] in (4, 5)

    # T5 - T4: Accepted to first fill ~284,705ms (284.705s)
    assert abs(final_summary["latency_t5_t4_ms"] - 284705) <= 1

    # T6 - T5: Partial to complete duration ~1,947ms (1.947s)
    assert abs(final_summary["latency_t6_t5_ms"] - 1947) <= 1

    # Forensic integrity: Must NOT claim queue wait without quote telemetry
    assert final_summary["causal_claim_permitted"] is False
    assert final_summary["first_marketable_observed_utc"] is None
    assert final_summary["passive_wait_ms"] is None
    assert final_summary["marketable_to_first_fill_ms"] is None
    assert final_summary["classification"] == "UNATTRIBUTED_ACCEPTED_TO_FILL_WAIT"
    assert final_summary["classification_reason"] == "QUOTE_TRADE_TIMELINE_UNAVAILABLE"
    assert "QUEUE_WAIT" not in final_summary["classification"]
