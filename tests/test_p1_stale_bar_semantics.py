"""
Unit and Property Tests for P1-B Stale-Bar Measurement Semantics and P1-A Reconnect Lifecycle Contracts.
Verifies:
1. Poll count vs Bar identity distinction:
   - 10 intra-bar polls on the SAME stale bar timestamp increments stale_bar_total by 10,
     but stale_bar_consecutive remains 1.
   - A subsequent distinct stale bar timestamp increments stale_bar_consecutive to 2.
   - A fresh completed bar resets stale_bar_consecutive to 0 while preserving stale_bar_total.
2. ExecutionHealthGate evaluates stale_bar_consecutive:
   - 38 total stale polls on 1 stale bar does NOT trip > 5 consecutive threshold.
   - 6 distinct consecutive stale bars trips ExecutionHealthGate (DEGRADED, allows_new_entries=False).
3. WebSocket Reconnect Lifecycle:
   - Amendment 1: _start_trade_stream() keeps recovery_required=True.
   - Amendment 2: Only clean reconciliation marks recovery_required=False and conn_health.mark_healthy.
   - Amendment 3: EntryGate automatically derives halts on stream state transitions.
   - Amendment 4: ExecutionHealthGate observes stream_state != HEALTHY and blocks entries.
4. Feed Provenance:
   - get_feed_provenance() authoritatively surfaces configured_feed, actual_feed, tier, capability.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

from strategies.intraday_data import (
    _validate_intraday_data,
    get_intraday_feed_telemetry,
    reset_intraday_feed_telemetry,
    get_feed_provenance,
    CompletedBar,
)
from strategies.execution_health import get_execution_health, HealthStatus
from alpaca_bridge import StreamState, EntryGateMock


import config


@pytest.fixture(autouse=True)
def reset_telemetry(monkeypatch):
    monkeypatch.setattr(config, "ENTRY_BAR_CLOSE_GRACE_SECONDS", 60.0)
    monkeypatch.setattr(config, "BAR_CLOSE_GRACE_SECONDS", 60.0)
    reset_intraday_feed_telemetry()
    yield
    reset_intraday_feed_telemetry()


class TestStaleBarMeasurementSemantics:
    """P1-B: Disentangling intra-bar polling ticks from true distinct bar staleness."""

    def _create_bar_df(self, bar_time: datetime, age_seconds: float) -> pd.DataFrame:
        """Create a mock 5-minute bar DataFrame."""
        return pd.DataFrame([
            {
                "date": bar_time,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 5000,
            }
        ]).set_index("date")

    def test_same_bar_intra_bar_polling_does_not_inflate_consecutive_stale_count(self):
        """
        Invariant: Repeated intra-bar polling on the SAME completed bar
        records observations in stale_bar_total, but stale_bar_consecutive remains 1.
        """
        bar_ts = datetime(2026, 9, 21, 14, 0, 0, tzinfo=timezone.utc)
        now_ts = bar_ts + timedelta(minutes=5, seconds=95)  # 95s delay > 60s grace -> STALE
        df = self._create_bar_df(bar_ts, age_seconds=95.0)

        # Simulate 10 intra-bar polls (e.g. every 20s over 3+ minutes) on the SAME bar
        with patch("strategies.session.is_market_open", return_value=True):
            for _ in range(10):
                with patch("strategies.intraday_data.select_last_completed_bar") as mock_sel:
                    mock_sel.return_value = CompletedBar(
                        row=df.iloc[0],
                        start_time=bar_ts,
                        close_time=bar_ts + timedelta(minutes=5),
                        post_close_delay_seconds=95.0,
                    )
                    res = _validate_intraday_data(df, "NFLX", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")
                    assert res.empty  # Stale bars must be skipped for entries

        tel = get_intraday_feed_telemetry()
        # Invariant checks:
        assert tel["poll_count"] == 10, "Total polls must equal 10"
        assert tel["stale_bar_total"] == 10, "Total stale observations must equal 10"
        assert tel["stale_bar_consecutive"] == 1, "Same bar repeated polling must NOT increment consecutive past 1"
        assert tel["stale_bar_skip_count"] == 1, "stale_bar_skip_count backwards compatibility mirror must equal 1"

    def test_new_distinct_stale_bar_increments_consecutive(self):
        """
        Invariant: When a NEW completed bar timestamp arrives and is stale,
        stale_bar_consecutive increments by 1.
        """
        bar1_ts = datetime(2026, 9, 21, 14, 0, 0, tzinfo=timezone.utc)
        bar2_ts = datetime(2026, 9, 21, 14, 5, 0, tzinfo=timezone.utc)
        df1 = self._create_bar_df(bar1_ts, 90.0)
        df2 = self._create_bar_df(bar2_ts, 85.0)

        with patch("strategies.session.is_market_open", return_value=True):
            # 3 polls on Bar 1
            for _ in range(3):
                with patch("strategies.intraday_data.select_last_completed_bar") as mock_sel:
                    mock_sel.return_value = CompletedBar(
                        row=df1.iloc[0],
                        start_time=bar1_ts,
                        close_time=bar1_ts + timedelta(minutes=5),
                        post_close_delay_seconds=90.0,
                    )
                    _validate_intraday_data(df1, "GOOGL", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")

            assert get_intraday_feed_telemetry()["stale_bar_consecutive"] == 1
            assert get_intraday_feed_telemetry()["stale_bar_total"] == 3

            # Now Bar 2 completes 5 minutes later, but is ALSO stale
            for _ in range(2):
                with patch("strategies.intraday_data.select_last_completed_bar") as mock_sel:
                    mock_sel.return_value = CompletedBar(
                        row=df2.iloc[0],
                        start_time=bar2_ts,
                        close_time=bar2_ts + timedelta(minutes=5),
                        post_close_delay_seconds=85.0,
                    )
                    _validate_intraday_data(df2, "GOOGL", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")

        tel = get_intraday_feed_telemetry()
        assert tel["stale_bar_total"] == 5
        assert tel["stale_bar_consecutive"] == 2, "Second distinct stale bar must increment consecutive to 2"

    def test_fresh_bar_resets_consecutive_to_zero(self):
        """
        Invariant: Arrival of a fresh completed bar resets stale_bar_consecutive = 0,
        while preserving cumulative stale_bar_total for telemetry auditing.
        """
        bar_stale = datetime(2026, 9, 21, 14, 0, 0, tzinfo=timezone.utc)
        bar_fresh = datetime(2026, 9, 21, 14, 5, 0, tzinfo=timezone.utc)
        df_stale = self._create_bar_df(bar_stale, 100.0)
        df_fresh = self._create_bar_df(bar_fresh, 15.0)

        with patch("strategies.session.is_market_open", return_value=True):
            # 1. Stale bar arrives
            with patch("strategies.intraday_data.select_last_completed_bar") as mock_sel:
                mock_sel.return_value = CompletedBar(
                    row=df_stale.iloc[0],
                    start_time=bar_stale,
                    close_time=bar_stale + timedelta(minutes=5),
                    post_close_delay_seconds=100.0,
                )
                _validate_intraday_data(df_stale, "SPY", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")

            assert get_intraday_feed_telemetry()["stale_bar_consecutive"] == 1
            assert get_intraday_feed_telemetry()["stale_bar_total"] == 1

            # 2. Fresh bar arrives (delay 15s <= grace 60s)
            with patch("strategies.intraday_data.select_last_completed_bar") as mock_sel:
                mock_sel.return_value = CompletedBar(
                    row=df_fresh.iloc[0],
                    start_time=bar_fresh,
                    close_time=bar_fresh + timedelta(minutes=5),
                    post_close_delay_seconds=15.0,
                )
                res_fresh = _validate_intraday_data(df_fresh, "SPY", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")
                assert not res_fresh.empty, "Fresh bar must be accepted"

        tel = get_intraday_feed_telemetry()
        assert tel["stale_bar_consecutive"] == 0, "Fresh bar MUST reset consecutive to 0"
        assert tel["stale_bar_skip_count"] == 0, "Backwards compatibility mirror MUST reset to 0"
        assert tel["stale_bar_total"] == 1, "stale_bar_total must preserve cumulative history"
        assert tel["stale_observation_total"] == 1, "stale_observation_total must preserve cumulative history"

    def test_same_bar_100_polls_consecutive_equals_one_then_next_stale_then_fresh(self):
        """
        Proof of Exact Semantics requested by User:
        - 100 intra-bar polls on SAME bar: consecutive == 1, stale_observation_total == 100
        - Next distinct stale bar: consecutive == 2, stale_observation_total == 101
        - Fresh completed bar: consecutive == 0, stale_observation_total preserved at 101
        """
        bar1_ts = datetime(2026, 9, 21, 14, 0, 0, tzinfo=timezone.utc)
        bar2_ts = datetime(2026, 9, 21, 14, 5, 0, tzinfo=timezone.utc)
        bar3_ts = datetime(2026, 9, 21, 14, 10, 0, tzinfo=timezone.utc)
        df1 = self._create_bar_df(bar1_ts, 95.0)
        df2 = self._create_bar_df(bar2_ts, 90.0)
        df3 = self._create_bar_df(bar3_ts, 20.0)

        with patch("strategies.session.is_market_open", return_value=True):
            # Phase 1: 100 polls on the SAME stale bar
            for _ in range(100):
                with patch("strategies.intraday_data.select_last_completed_bar") as mock_sel:
                    mock_sel.return_value = CompletedBar(
                        row=df1.iloc[0],
                        start_time=bar1_ts,
                        close_time=bar1_ts + timedelta(minutes=5),
                        post_close_delay_seconds=95.0,
                    )
                    _validate_intraday_data(df1, "NFLX", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")

            tel1 = get_intraday_feed_telemetry()
            assert tel1["stale_bar_consecutive"] == 1, "100 polls on same bar must yield consecutive == 1"
            assert tel1["stale_observation_total"] == 100, "100 polls on same bar must yield observation_total == 100"
            assert tel1["stale_bar_total"] == 100, "stale_bar_total alias must equal 100"
            assert "NFLX" in tel1["last_stale_bar_ts"]

            # Phase 2: Next distinct stale bar arrives
            with patch("strategies.intraday_data.select_last_completed_bar") as mock_sel:
                mock_sel.return_value = CompletedBar(
                    row=df2.iloc[0],
                    start_time=bar2_ts,
                    close_time=bar2_ts + timedelta(minutes=5),
                    post_close_delay_seconds=90.0,
                )
                _validate_intraday_data(df2, "NFLX", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")

            tel2 = get_intraday_feed_telemetry()
            assert tel2["stale_bar_consecutive"] == 2, "Second distinct stale bar must increment consecutive to 2"
            assert tel2["stale_observation_total"] == 101, "stale_observation_total must equal 101"

            # Phase 3: Fresh completed bar arrives
            with patch("strategies.intraday_data.select_last_completed_bar") as mock_sel:
                mock_sel.return_value = CompletedBar(
                    row=df3.iloc[0],
                    start_time=bar3_ts,
                    close_time=bar3_ts + timedelta(minutes=5),
                    post_close_delay_seconds=20.0,
                )
                res = _validate_intraday_data(df3, "NFLX", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")
                assert not res.empty

            tel3 = get_intraday_feed_telemetry()
            assert tel3["stale_bar_consecutive"] == 0, "Fresh bar must reset consecutive to 0"
            assert tel3["stale_observation_total"] == 101, "stale_observation_total must be preserved at 101"


class TestExecutionHealthGateWithStaleSemantics:
    """Verifies that ExecutionHealthGate threshold (> 5) correctly acts on consecutive distinct bars."""

    def test_health_gate_remains_healthy_with_high_total_but_low_consecutive(self):
        """
        Forensic Scenario: 38 intra-bar polling skips on 1 stale bar (like NFLX session)
        must NOT trip health gate when consecutive is 1 (<= 5).
        """
        bridge_mock = MagicMock()
        bridge_mock.is_connected = True
        bridge_mock.entry_gate.halted = False
        bridge_mock.last_rtt_latency_ms = 45.0
        bridge_mock.stream_state = StreamState.HEALTHY

        feed_tel = {
            "stale_bar_consecutive": 1,
            "stale_bar_total": 38,
            "stale_bar_skip_count": 1,
            "batch_timeout_count": 0,
        }

        health = get_execution_health(bridge=bridge_mock, bar_age_seconds=15.0, feed_telemetry=feed_tel)
        assert health.status == HealthStatus.HEALTHY
        assert health.allows_new_entries is True

    def test_health_gate_trips_degraded_when_consecutive_exceeds_five(self):
        """
        Invariant: When 6 distinct consecutive 5m bars are stale, health gate trips DEGRADED
        and blocks new entries.
        """
        bridge_mock = MagicMock()
        bridge_mock.is_connected = True
        bridge_mock.entry_gate.halted = False
        bridge_mock.last_rtt_latency_ms = 45.0
        bridge_mock.stream_state = StreamState.HEALTHY

        feed_tel = {
            "stale_bar_consecutive": 6,
            "stale_bar_total": 12,
            "stale_bar_skip_count": 6,
            "batch_timeout_count": 0,
        }

        health = get_execution_health(bridge=bridge_mock, bar_age_seconds=15.0, feed_telemetry=feed_tel)
        assert health.status == HealthStatus.DEGRADED
        assert health.allows_new_entries is False
        assert "6 > 5 limit" in health.summary


class TestWebSocketReconnectContractAmendments:
    """P1-A: Code-level verification of all 4 amendments for WebSocket Reconnect Contract."""

    def test_amendment_1_start_trade_stream_does_not_clear_recovery_required(self):
        """
        Amendment 1: When _start_trade_stream establishes connection,
        _trade_stream_ready=True, but _trade_stream_recovery_required MUST remain True.
        """
        from alpaca_bridge import AlpacaBridge
        bridge = AlpacaBridge(dry_run=True)
        bridge._trade_stream_credentials = ("key", "secret")

        mock_stream = MagicMock()
        mock_stream._running = True
        mock_stream._endpoint = "wss://paper-api.alpaca.markets/stream"

        with patch("alpaca_bridge.TradingStream", return_value=mock_stream), \
             patch.object(bridge, "_assert_paper_stream_endpoint"), \
             patch("alpaca_bridge.threading.Thread"):
            # Set initial recovery required state
            bridge._trade_stream_recovery_required = True
            bridge._start_trade_stream("key", "secret")

            assert bridge.is_stream_ready is True
            assert bridge.stream_state == StreamState.AUTHENTICATED_UNRECONCILED
            assert bridge._trade_stream_recovery_required is True, "Amendment 1 Violation: recovery_required was prematurely cleared!"

    def test_amendment_2_only_clean_reconciliation_clears_recovery_and_marks_healthy(self):
        """
        Amendment 2: recovery_required=False and conn_health.mark_healthy
        MUST only occur inside _run_post_reconnect_reconciliation upon clean check.
        """
        from alpaca_bridge import AlpacaBridge
        bridge = AlpacaBridge(dry_run=True)
        bridge._stream_state = StreamState.AUTHENTICATED_UNRECONCILED
        # A valid reconciliation must belong to a still-ready authenticated
        # transport, not merely a state value and a clean REST response.
        bridge._trade_stream = object()
        bridge._trade_stream_ready = True
        bridge._trade_stream_recovery_required = True

        with patch.object(bridge, "reconcile_startup_state", return_value={"clean": True}), \
             patch.object(bridge, "_clear_recovery_halts"):

            recon_ok = bridge._run_post_reconnect_reconciliation()
            assert recon_ok is True
            assert bridge.stream_state == StreamState.HEALTHY
            assert bridge._trade_stream_recovery_required is False
            assert bridge._conn_health.is_healthy is True

    def test_amendment_3_entry_gate_automatically_derives_halts_from_stream_state(self):
        """
        Amendment 3: Transitioning to AUTHENTICATED_UNRECONCILED or RECONCILING
        automatically halts the entry gate with BROKER_RECONCILIATION_PENDING.
        """
        from alpaca_bridge import AlpacaBridge
        bridge = AlpacaBridge(dry_run=True)
        bridge.entry_gate.resume()
        assert bridge.entry_gate.halted is False

        # Transition to AUTHENTICATED_UNRECONCILED
        bridge._transition_stream_state(
            expected={StreamState.DISCONNECTED},
            new_state=StreamState.AUTHENTICATED_UNRECONCILED,
            reason="Connected",
        )
        assert bridge.entry_gate.halted is True
        assert "BROKER_RECONCILIATION_PENDING" in bridge.entry_gate.reasons

        # Transition to RECONCILING
        bridge._transition_stream_state(
            expected={StreamState.AUTHENTICATED_UNRECONCILED},
            new_state=StreamState.RECONCILING,
            reason="Reconciling",
        )
        assert bridge.entry_gate.halted is True
        assert "BROKER_RECONCILIATION_PENDING" in bridge.entry_gate.reasons

    def test_amendment_4_execution_health_gate_secondary_safety_fence(self):
        """
        Amendment 4: ExecutionHealthGate directly observes bridge.stream_state
        and fails closed (allows_new_entries=False) whenever stream_state != HEALTHY.
        """
        bridge_mock = MagicMock()
        bridge_mock.is_connected = True
        bridge_mock.entry_gate.halted = False
        bridge_mock.last_rtt_latency_ms = 35.0

        for non_healthy in [
            StreamState.DISCONNECTED,
            StreamState.RECONNECTING,
            StreamState.AUTHENTICATED_UNRECONCILED,
            StreamState.RECONCILING,
            StreamState.LATCHED_DESYNC,
            StreamState.DEGRADED,
        ]:
            bridge_mock.stream_state = non_healthy
            health = get_execution_health(bridge=bridge_mock, bar_age_seconds=10.0)
            assert health.allows_new_entries is False, f"Failed closed invariant violated for {non_healthy}"
            assert health.status in (HealthStatus.DEGRADED, HealthStatus.BROKEN)


class TestFeedProvenanceP0:
    """P0: Authoritative Feed Provenance helper."""

    def test_feed_provenance_free_tier(self):
        with patch("config.ALPACA_SUBSCRIPTION_TIER", "free"), \
             patch("config.ALPACA_DATA_FEED", "iex"):
            fp = get_feed_provenance()
            assert fp["actual_feed"] == "iex"
            assert fp["subscription_tier"] == "free"
            assert "IEX_ONLY" in fp["feed_capability"]
            assert fp["bar_source"] == "alpaca_api"

    def test_feed_provenance_sip_tier(self):
        with patch("config.ALPACA_SUBSCRIPTION_TIER", "algo_trader_plus"), \
             patch("config.ALPACA_DATA_FEED", "sip"):
            fp = get_feed_provenance()
            assert fp["actual_feed"] == "sip"
            assert fp["subscription_tier"] == "algo_trader_plus"
            assert "SIP_CONSOLIDATED" in fp["feed_capability"]
