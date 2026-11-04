"""
execution_health.py — Fail-Closed Execution Health Gate for Day Trading (Phase A / P0 Hardening)

Evaluates the physical health of the execution environment (broker connectivity,
quote/bar data freshness, batch fetch latency, timeout occurrences, data coverage,
and startup reconciliation state) before permitting new order entries.

Strict Fail-Closed Contract:
    HEALTHY           -> allows_new_entries = True
    DEGRADED          -> allows_new_entries = False (NO confidence override in Phase A)
    STALE             -> allows_new_entries = False
    BROKEN            -> allows_new_entries = False

Channel Decoupling:
    - BrokerExecutionHealth: Evaluates trade-update WebSocket stream, broker REST connection,
      broker truth reconciliation state, and RTT latency.
    - MarketDataHealth: Evaluates data feed coverage (>= 95%), bar freshness, quote latency,
      consecutive fetch timeouts, and batch fetch duration.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import config as config_module

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    BROKEN = "broken"


@dataclass(frozen=True)
class BrokerExecutionHealth:
    """Independent health evaluation for broker connectivity and execution stream."""
    status: HealthStatus
    allows_execution: bool
    is_connected: bool
    stream_state: str
    reconciliation_clean: bool
    entry_gate_halted: bool
    entry_gate_reasons: Tuple[str, ...]
    rtt_latency_ms: Optional[float]
    reason: str


@dataclass(frozen=True)
class MarketDataHealth:
    """Independent health evaluation for market data ingestion and quote quality."""
    status: HealthStatus
    allows_signals: bool
    coverage: float
    required_coverage: float
    consecutive_fetch_timeouts: int
    last_batch_duration_seconds: float
    bar_age_seconds: Optional[float]
    stale_bar_consecutive: int
    reason: str


@dataclass(frozen=True)
class ExecutionHealthReport:
    status: HealthStatus
    allows_new_entries: bool
    reason: str
    broker_health: Optional[BrokerExecutionHealth] = None
    data_health: Optional[MarketDataHealth] = None
    recovery_ready: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: float = field(default_factory=time.time)

    @property
    def summary(self) -> str:
        return f"{self.status.value.upper()}: {self.reason}"

    @property
    def bar_age_seconds(self) -> Optional[float]:
        val = self.metrics.get("bar_age_seconds")
        return float(val) if val is not None else None

    @property
    def broker_latency_ms(self) -> Optional[float]:
        val = self.metrics.get("broker_rtt_latency_ms")
        return float(val) if val is not None else None


class ExecutionHealthGate:
    """
    Evaluates runtime telemetry to determine if the trading pipeline is healthy
    enough to accept new capital commitments.

    Strictly separates broker execution state from market data feed quality.
    """

    def __init__(
        self,
        max_bar_age_seconds: float = 60.0,
        max_batch_duration_seconds: float = 10.0,
        max_quote_age_seconds: float = 10.0,
    ):
        self.max_bar_age_seconds = max_bar_age_seconds
        self.max_batch_duration_seconds = max_batch_duration_seconds
        self.max_quote_age_seconds = max_quote_age_seconds

    def evaluate_broker_health(
        self,
        bridge: object,
        startup_reconciled: bool = True,
    ) -> BrokerExecutionHealth:
        """
        Evaluate broker connectivity, trade stream state, and reconciliation handshakes.
        """
        is_connected = False
        if hasattr(bridge, "is_connected"):
            is_connected = bool(bridge.is_connected)
        elif hasattr(bridge, "_connected"):
            is_connected = bool(bridge._connected)

        if not is_connected:
            return BrokerExecutionHealth(
                status=HealthStatus.BROKEN,
                allows_execution=False,
                is_connected=False,
                stream_state="DISCONNECTED",
                reconciliation_clean=False,
                entry_gate_halted=False,
                entry_gate_reasons=(),
                rtt_latency_ms=None,
                reason="Broker disconnected or connection unverified",
            )

        entry_gate = getattr(bridge, "entry_gate", None)
        entry_gate_halted = bool(entry_gate is not None and getattr(entry_gate, "halted", False) is True)
        raw_reasons = set(getattr(entry_gate, "reasons", set())) if entry_gate else set()
        gate_reasons = tuple(sorted(str(r) for r in raw_reasons))
        h_reason = str(getattr(entry_gate, "halt_reason", "") or "")

        if entry_gate_halted:
            reason_str = (
                f"Broker entry gate circuit breaker is halted (reasons: {list(gate_reasons)})"
                if gate_reasons
                else f"Broker entry gate circuit breaker is halted ({h_reason or 'unknown'})"
            )
            return BrokerExecutionHealth(
                status=HealthStatus.BROKEN,
                allows_execution=False,
                is_connected=True,
                stream_state="HALTED",
                reconciliation_clean=False,
                entry_gate_halted=True,
                entry_gate_reasons=gate_reasons,
                rtt_latency_ms=getattr(bridge, "last_rtt_latency_ms", None),
                reason=reason_str,
            )

        if not startup_reconciled:
            return BrokerExecutionHealth(
                status=HealthStatus.BROKEN,
                allows_execution=False,
                is_connected=True,
                stream_state="UNRECONCILED",
                reconciliation_clean=False,
                entry_gate_halted=False,
                entry_gate_reasons=gate_reasons,
                rtt_latency_ms=getattr(bridge, "last_rtt_latency_ms", None),
                reason="Startup state reconciliation is not yet verified clean",
            )

        # Check stream state as secondary safety fence
        raw_stream_state = getattr(bridge, "stream_state", None) or getattr(bridge, "_stream_state", None)
        stream_val = "HEALTHY"
        if raw_stream_state is not None:
            # Handle real StreamState enum, string, or explicit mock state
            candidate_val = getattr(raw_stream_state, "value", raw_stream_state)
            if isinstance(candidate_val, str) and not hasattr(candidate_val, "_mock_name"):
                stream_val = candidate_val.upper().strip()
            elif hasattr(raw_stream_state, "name") and not hasattr(raw_stream_state.name, "_mock_name"):
                stream_val = str(raw_stream_state.name).upper().strip()

            KNOWN_DEGRADED = {"DEGRADED", "RECONNECTING", "AUTHENTICATED_UNRECONCILED", "RECONCILING"}
            KNOWN_BROKEN = {"DISCONNECTED", "LATCHED_DESYNC", "STOPPED", "SHUTTING_DOWN"}
            if stream_val in KNOWN_DEGRADED:
                return BrokerExecutionHealth(
                    status=HealthStatus.DEGRADED,
                    allows_execution=False,
                    is_connected=True,
                    stream_state=stream_val,
                    reconciliation_clean=False,
                    entry_gate_halted=False,
                    entry_gate_reasons=gate_reasons,
                    rtt_latency_ms=getattr(bridge, "last_rtt_latency_ms", None),
                    reason=f"Trade-update stream is {stream_val} (reconnect / broker reconciliation handshake pending)",
                )
            elif stream_val in KNOWN_BROKEN:
                return BrokerExecutionHealth(
                    status=HealthStatus.BROKEN,
                    allows_execution=False,
                    is_connected=True,
                    stream_state=stream_val,
                    reconciliation_clean=False,
                    entry_gate_halted=False,
                    entry_gate_reasons=gate_reasons,
                    rtt_latency_ms=getattr(bridge, "last_rtt_latency_ms", None),
                    reason=f"Trade-update stream is {stream_val}",
                )

        rtt = getattr(bridge, "last_rtt_latency_ms", None)
        rtt_float = float(rtt) if rtt is not None and isinstance(rtt, (int, float)) else None
        if rtt_float is not None and rtt_float > 500.0:
            return BrokerExecutionHealth(
                status=HealthStatus.DEGRADED,
                allows_execution=False,
                is_connected=True,
                stream_state=stream_val,
                reconciliation_clean=True,
                entry_gate_halted=False,
                entry_gate_reasons=gate_reasons,
                rtt_latency_ms=rtt_float,
                reason=f"Broker RTT latency elevated ({rtt_float:.1f}ms > 500.0ms limit)",
            )

        return BrokerExecutionHealth(
            status=HealthStatus.HEALTHY,
            allows_execution=True,
            is_connected=True,
            stream_state=stream_val,
            reconciliation_clean=True,
            entry_gate_halted=False,
            entry_gate_reasons=gate_reasons,
            rtt_latency_ms=rtt_float,
            reason="Broker execution environment and stream state healthy",
        )

    def evaluate_market_data_health(
        self,
        bar_age_seconds: Optional[float] = None,
        feed_telemetry: Optional[Dict[str, Any]] = None,
    ) -> MarketDataHealth:
        """
        Evaluate data feed quality, coverage, timeout count, and bar freshness.
        """
        tel = feed_telemetry or {}
        timeouts = int(tel.get("consecutive_fetch_timeouts", 0) or 0)
        batch_dur = float(tel.get("last_batch_duration_s", 0.0) or tel.get("last_batch_duration_seconds", 0.0) or 0.0)
        stale_consecutive = tel.get("stale_bar_consecutive")
        if stale_consecutive is None:
            stale_consecutive = int(tel.get("stale_bar_skip_count", 0) or 0)
        else:
            stale_consecutive = int(stale_consecutive)

        req_coverage = float(getattr(config_module, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95))
        coverage_val = tel.get("last_batch_coverage")
        if coverage_val is None:
            coverage_val = tel.get("coverage")
        coverage = float(coverage_val) if coverage_val is not None else 1.0

        bar_age = float(bar_age_seconds) if bar_age_seconds is not None else None

        # 1. Bar staleness check (STALE)
        if bar_age is not None and bar_age > self.max_bar_age_seconds:
            return MarketDataHealth(
                status=HealthStatus.STALE,
                allows_signals=False,
                coverage=coverage,
                required_coverage=req_coverage,
                consecutive_fetch_timeouts=timeouts,
                last_batch_duration_seconds=batch_dur,
                bar_age_seconds=bar_age,
                stale_bar_consecutive=stale_consecutive,
                reason=f"Market data bar is stale ({bar_age:.1f}s > {self.max_bar_age_seconds:.1f}s max)",
            )

        # 2. Consecutive timeouts check (DEGRADED)
        if timeouts > 0:
            return MarketDataHealth(
                status=HealthStatus.DEGRADED,
                allows_signals=False,
                coverage=coverage,
                required_coverage=req_coverage,
                consecutive_fetch_timeouts=timeouts,
                last_batch_duration_seconds=batch_dur,
                bar_age_seconds=bar_age,
                stale_bar_consecutive=stale_consecutive,
                reason=f"Recent market data fetch timeouts detected (consecutive={timeouts})",
            )

        # 3. Batch duration elevated check (DEGRADED)
        if batch_dur > self.max_batch_duration_seconds:
            return MarketDataHealth(
                status=HealthStatus.DEGRADED,
                allows_signals=False,
                coverage=coverage,
                required_coverage=req_coverage,
                consecutive_fetch_timeouts=timeouts,
                last_batch_duration_seconds=batch_dur,
                bar_age_seconds=bar_age,
                stale_bar_consecutive=stale_consecutive,
                reason=f"Market data batch fetch latency elevated ({batch_dur:.2f}s > {self.max_batch_duration_seconds:.1f}s max)",
            )

        # 4. Elevated stale bars check (DEGRADED)
        if stale_consecutive > 5:
            return MarketDataHealth(
                status=HealthStatus.DEGRADED,
                allows_signals=False,
                coverage=coverage,
                required_coverage=req_coverage,
                consecutive_fetch_timeouts=timeouts,
                last_batch_duration_seconds=batch_dur,
                bar_age_seconds=bar_age,
                stale_bar_consecutive=stale_consecutive,
                reason=f"Elevated consecutive stale bars detected ({stale_consecutive} > 5 limit)",
            )

        # 5. Data Coverage check (DEGRADED if below required 95%)
        if coverage < (req_coverage - 1e-6):
            return MarketDataHealth(
                status=HealthStatus.DEGRADED,
                allows_signals=False,
                coverage=coverage,
                required_coverage=req_coverage,
                consecutive_fetch_timeouts=timeouts,
                last_batch_duration_seconds=batch_dur,
                bar_age_seconds=bar_age,
                stale_bar_consecutive=stale_consecutive,
                reason=f"Market data coverage {coverage*100.0:.1f}% below required threshold {req_coverage*100.0:.1f}%",
            )

        return MarketDataHealth(
            status=HealthStatus.HEALTHY,
            allows_signals=True,
            coverage=coverage,
            required_coverage=req_coverage,
            consecutive_fetch_timeouts=timeouts,
            last_batch_duration_seconds=batch_dur,
            bar_age_seconds=bar_age,
            stale_bar_consecutive=stale_consecutive,
            reason="Market data telemetry and coverage healthy",
        )

    def evaluate(
        self,
        bridge: object,
        bar_age_seconds: Optional[float] = None,
        feed_telemetry: Optional[Dict[str, Any]] = None,
        startup_reconciled: bool = True,
    ) -> ExecutionHealthReport:
        """
        Evaluate current execution health fail-closed, combining broker execution channel
        and market data ingestion channel.
        """
        broker_h = self.evaluate_broker_health(bridge=bridge, startup_reconciled=startup_reconciled)
        data_h = self.evaluate_market_data_health(bar_age_seconds=bar_age_seconds, feed_telemetry=feed_telemetry)

        metrics: Dict[str, Any] = {
            "bar_age_seconds": bar_age_seconds,
            "startup_reconciled": startup_reconciled,
            "broker_connected": broker_h.is_connected,
            "stream_state": broker_h.stream_state,
            "reconciliation_clean": broker_h.reconciliation_clean,
            "entry_gate_halted": broker_h.entry_gate_halted,
            "entry_gate_reasons": list(broker_h.entry_gate_reasons),
            "broker_rtt_latency_ms": broker_h.rtt_latency_ms,
            "coverage": data_h.coverage,
            "required_coverage": data_h.required_coverage,
            "consecutive_fetch_timeouts": data_h.consecutive_fetch_timeouts,
            "last_batch_duration_seconds": data_h.last_batch_duration_seconds,
            "stale_bar_consecutive": data_h.stale_bar_consecutive,
        }
        if feed_telemetry:
            metrics.update(feed_telemetry)

        # Hierarchy of severity: BROKEN > STALE > DEGRADED > HEALTHY
        overall_status = HealthStatus.HEALTHY
        primary_reason = "Execution environment and data telemetry healthy"

        if broker_h.status == HealthStatus.BROKEN:
            overall_status = HealthStatus.BROKEN
            primary_reason = broker_h.reason
        elif data_h.status == HealthStatus.BROKEN:
            overall_status = HealthStatus.BROKEN
            primary_reason = data_h.reason
        elif broker_h.status == HealthStatus.STALE:
            overall_status = HealthStatus.STALE
            primary_reason = broker_h.reason
        elif data_h.status == HealthStatus.STALE:
            overall_status = HealthStatus.STALE
            primary_reason = data_h.reason
        elif broker_h.status == HealthStatus.DEGRADED:
            overall_status = HealthStatus.DEGRADED
            primary_reason = broker_h.reason
        elif data_h.status == HealthStatus.DEGRADED:
            overall_status = HealthStatus.DEGRADED
            primary_reason = data_h.reason

        allows_new_entries = bool(broker_h.allows_execution and data_h.allows_signals)
        recovery_ready = bool(broker_h.allows_execution and data_h.allows_signals)

        return ExecutionHealthReport(
            status=overall_status,
            allows_new_entries=allows_new_entries,
            reason=primary_reason,
            broker_health=broker_h,
            data_health=data_h,
            recovery_ready=recovery_ready,
            metrics=metrics,
            evaluated_at=time.time(),
        )


# Global singleton gate instance
health_gate = ExecutionHealthGate()


def get_execution_health(
    bridge: object,
    bar_age_seconds: Optional[float] = None,
    feed_telemetry: Optional[Dict[str, Any]] = None,
    startup_reconciled: bool = True,
) -> ExecutionHealthReport:
    """Convenience functional access to ExecutionHealthGate evaluation."""
    return health_gate.evaluate(
        bridge=bridge,
        bar_age_seconds=bar_age_seconds,
        feed_telemetry=feed_telemetry,
        startup_reconciled=startup_reconciled,
    )
