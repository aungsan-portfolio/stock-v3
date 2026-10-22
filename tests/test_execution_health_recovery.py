"""
test_execution_health_recovery.py — Comprehensive Unit & Fault-Injection Tests for Execution Health & Market-Data Recovery (P0)

Tests strictly enforce:
1. BrokerExecutionHealth evaluates websocket state, broker connection, entry gate, and RTT latency independently.
2. MarketDataHealth evaluates coverage (>= 95%), timeouts, batch duration, and bar freshness independently.
3. Decoupled channel separation: broker stream vs market data feed are not conflated.
4. Fail-closed on failures (stream disconnect, auth failure, 61.3% coverage, timeout).
5. Dynamic recovery verification: Health gate only re-opens when BOTH broker truth and data coverage are healthy.
"""
from unittest.mock import MagicMock
import pytest

from strategies.execution_health import (
    ExecutionHealthGate,
    HealthStatus,
    get_execution_health,
)


class TestExecutionHealthRecovery:

    @pytest.fixture
    def healthy_bridge(self):
        bridge = MagicMock()
        bridge.is_connected = True
        bridge.last_rtt_latency_ms = 45.0
        bridge.entry_gate.halted = False
        bridge.entry_gate.reasons = set()
        bridge.entry_gate.halt_reason = ""
        bridge.stream_state.value = "HEALTHY"
        return bridge

    def test_broker_disconnect_fails_closed(self, healthy_bridge):
        healthy_bridge.is_connected = False
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=15.0)

        assert report.status == HealthStatus.BROKEN
        assert report.allows_new_entries is False
        assert report.broker_health.allows_execution is False
        assert "Broker disconnected" in report.broker_health.reason

    def test_entry_gate_halted_fails_closed_with_sorted_reasons(self, healthy_bridge):
        healthy_bridge.entry_gate.halted = True
        healthy_bridge.entry_gate.reasons = {"BROKER_RECONCILIATION_PENDING", "TRADE_STREAM_DISCONNECTED"}
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=15.0)

        assert report.status == HealthStatus.BROKEN
        assert report.allows_new_entries is False
        assert report.broker_health.entry_gate_halted is True
        assert report.broker_health.entry_gate_reasons == ("BROKER_RECONCILIATION_PENDING", "TRADE_STREAM_DISCONNECTED")
        assert "BROKER_RECONCILIATION_PENDING" in report.reason

    def test_stream_reconnecting_degrades_broker_health(self, healthy_bridge):
        healthy_bridge.stream_state.value = "RECONNECTING"
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=15.0)

        assert report.status == HealthStatus.DEGRADED
        assert report.allows_new_entries is False
        assert report.broker_health.status == HealthStatus.DEGRADED
        assert "reconnect / broker reconciliation handshake pending" in report.broker_health.reason

    def test_stream_latched_desync_breaks_broker_health(self, healthy_bridge):
        healthy_bridge.stream_state.value = "LATCHED_DESYNC"
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=15.0)

        assert report.status == HealthStatus.BROKEN
        assert report.allows_new_entries is False
        assert report.broker_health.status == HealthStatus.BROKEN

    def test_broker_rtt_latency_elevated_degrades_broker_health(self, healthy_bridge):
        healthy_bridge.last_rtt_latency_ms = 620.0
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=15.0)

        assert report.status == HealthStatus.DEGRADED
        assert report.allows_new_entries is False
        assert report.broker_health.status == HealthStatus.DEGRADED
        assert "620.0ms > 500.0ms limit" in report.broker_health.reason

    def test_market_data_coverage_61_3_pct_fails_closed(self, healthy_bridge):
        """Coverage drop to 61.3% (below 95% threshold) must block new entries fail-closed."""
        feed_tel = {
            "last_batch_coverage": 0.613,  # 19/31 symbols
            "consecutive_fetch_timeouts": 0,
            "last_batch_duration_seconds": 4.5,
            "stale_bar_consecutive": 0,
        }
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=15.0, feed_telemetry=feed_tel)

        assert report.status == HealthStatus.DEGRADED
        assert report.allows_new_entries is False
        assert report.data_health.allows_signals is False
        assert report.broker_health.allows_execution is True  # Broker itself is healthy!
        assert "coverage 61.3% below required threshold 95.0%" in report.data_health.reason

    def test_market_data_coverage_recovery_restores_healthy(self, healthy_bridge):
        """When coverage returns to >= 95.0%, MarketDataHealth recovers cleanly."""
        feed_tel = {
            "last_batch_coverage": 1.0,  # 31/31 symbols
            "consecutive_fetch_timeouts": 0,
            "last_batch_duration_seconds": 3.2,
            "stale_bar_consecutive": 0,
        }
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=20.0, feed_telemetry=feed_tel)

        assert report.status == HealthStatus.HEALTHY
        assert report.allows_new_entries is True
        assert report.broker_health.status == HealthStatus.HEALTHY
        assert report.data_health.status == HealthStatus.HEALTHY
        assert report.recovery_ready is True

    def test_market_data_fetch_timeout_fails_closed(self, healthy_bridge):
        feed_tel = {
            "last_batch_coverage": 1.0,
            "consecutive_fetch_timeouts": 1,
            "last_batch_duration_seconds": 3.0,
            "stale_bar_consecutive": 0,
        }
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=15.0, feed_telemetry=feed_tel)

        assert report.status == HealthStatus.DEGRADED
        assert report.allows_new_entries is False
        assert report.data_health.allows_signals is False
        assert "timeouts detected" in report.data_health.reason

    def test_decoupled_channel_independence_broker_healthy_data_broken(self, healthy_bridge):
        """Broker is 100% healthy, but market data bar is stale (> 60s). Must block entries."""
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=75.0)

        assert report.status == HealthStatus.STALE
        assert report.allows_new_entries is False
        assert report.broker_health.allows_execution is True
        assert report.data_health.allows_signals is False
        assert report.recovery_ready is False

    def test_decoupled_channel_independence_data_healthy_broker_reconnecting(self, healthy_bridge):
        """Market data is 100% fresh, but trade stream is reconnecting. Must block entries."""
        healthy_bridge.stream_state.value = "RECONNECTING"
        feed_tel = {
            "last_batch_coverage": 1.0,
            "consecutive_fetch_timeouts": 0,
            "last_batch_duration_seconds": 2.0,
            "stale_bar_consecutive": 0,
        }
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=15.0, feed_telemetry=feed_tel)

        assert report.status == HealthStatus.DEGRADED
        assert report.allows_new_entries is False
        assert report.data_health.allows_signals is True
        assert report.broker_health.allows_execution is False
        assert report.recovery_ready is False

    def test_full_recovery_requires_both_channels_healthy(self, healthy_bridge):
        """Only when BOTH broker execution channel and market data feed are healthy is new entry allowed."""
        feed_tel = {
            "last_batch_coverage": 1.0,
            "consecutive_fetch_timeouts": 0,
            "last_batch_duration_seconds": 2.5,
            "stale_bar_consecutive": 0,
        }
        gate = ExecutionHealthGate()
        report = gate.evaluate(healthy_bridge, bar_age_seconds=25.0, feed_telemetry=feed_tel)

        assert report.status == HealthStatus.HEALTHY
        assert report.allows_new_entries is True
        assert report.broker_health.allows_execution is True
        assert report.data_health.allows_signals is True
        assert report.recovery_ready is True
