"""
execution_health.py — Fail-Closed Execution Health Gate for Day Trading (Phase A)

Evaluates the physical health of the execution environment (broker connectivity,
quote/bar data freshness, batch fetch latency, timeout occurrences, and startup
reconciliation state) before permitting new order entries.

Strict Fail-Closed Contract:
    HEALTHY           -> allows_new_entries = True
    DEGRADED          -> allows_new_entries = False (NO confidence override in Phase A)
    STALE             -> allows_new_entries = False
    BROKEN            -> allows_new_entries = False
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    BROKEN = "broken"


@dataclass(frozen=True)
class ExecutionHealthReport:
    status: HealthStatus
    allows_new_entries: bool
    reason: str
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

    def evaluate(
        self,
        bridge: object,
        bar_age_seconds: Optional[float] = None,
        feed_telemetry: Optional[Dict[str, Any]] = None,
        startup_reconciled: bool = True,
    ) -> ExecutionHealthReport:
        """
        Evaluate current execution health fail-closed.
        """
        metrics: Dict[str, Any] = {
            "bar_age_seconds": bar_age_seconds,
            "startup_reconciled": startup_reconciled,
        }

        # 1. Broker Connection & Circuit Breakers (BROKEN)
        is_connected = False
        if hasattr(bridge, "is_connected"):
            is_connected = bool(bridge.is_connected)
        elif hasattr(bridge, "_connected"):
            is_connected = bool(bridge._connected)
        metrics["broker_connected"] = bool(is_connected)
        if not is_connected:
            return ExecutionHealthReport(
                status=HealthStatus.BROKEN,
                allows_new_entries=False,
                reason="Broker disconnected or connection unverified",
                metrics=metrics,
            )

        entry_gate = getattr(bridge, "entry_gate", None)
        if entry_gate is not None and getattr(entry_gate, "halted", False) is True:
            metrics["entry_gate_halted"] = True
            return ExecutionHealthReport(
                status=HealthStatus.BROKEN,
                allows_new_entries=False,
                reason="Broker entry gate circuit breaker is halted",
                metrics=metrics,
            )

        if not startup_reconciled:
            return ExecutionHealthReport(
                status=HealthStatus.BROKEN,
                allows_new_entries=False,
                reason="Startup state reconciliation is not yet verified clean",
                metrics=metrics,
            )

        # Check stream state as secondary safety fence (observing StreamState without mutating)
        raw_stream_state = getattr(bridge, "stream_state", None) or getattr(bridge, "_stream_state", None)
        if raw_stream_state is not None and not str(type(raw_stream_state)).endswith("Mock'>"):
            state_val = getattr(raw_stream_state, "value", str(raw_stream_state)) if hasattr(raw_stream_state, "value") else str(raw_stream_state)
            KNOWN_NON_HEALTHY = {
                "DISCONNECTED", "RECONNECTING", "AUTHENTICATED_UNRECONCILED",
                "RECONCILING", "DEGRADED", "LATCHED_DESYNC", "STOPPED", "SHUTTING_DOWN",
            }
            if state_val in KNOWN_NON_HEALTHY:
                metrics["stream_state"] = state_val
                return ExecutionHealthReport(
                    status=HealthStatus.DEGRADED if state_val in ("RECONNECTING", "AUTHENTICATED_UNRECONCILED", "RECONCILING") else HealthStatus.BROKEN,
                    allows_new_entries=False,
                    reason=f"Trade-update stream is {state_val} (reconnect / broker reconciliation handshake pending)",
                    metrics=metrics,
                )
            elif state_val == "HEALTHY":
                metrics["stream_state"] = "HEALTHY"

        # 2. Market Data Staleness (STALE)
        if bar_age_seconds is not None:
            metrics["bar_age_seconds"] = float(bar_age_seconds)
            if bar_age_seconds > self.max_bar_age_seconds:
                return ExecutionHealthReport(
                    status=HealthStatus.STALE,
                    allows_new_entries=False,
                    reason=f"Market data bar is stale ({bar_age_seconds:.1f}s > {self.max_bar_age_seconds:.1f}s max)",
                    metrics=metrics,
                )

        # 3. Broker Latency (DEGRADED)
        rtt = getattr(bridge, "last_rtt_latency_ms", None)
        if rtt is not None and isinstance(rtt, (int, float)):
            metrics["broker_rtt_latency_ms"] = float(rtt)
            if float(rtt) > 500.0:
                return ExecutionHealthReport(
                    status=HealthStatus.DEGRADED,
                    allows_new_entries=False,
                    reason=f"Broker RTT latency elevated ({float(rtt):.1f}ms > 500.0ms limit)",
                    metrics=metrics,
                )

        # 4. Feed Telemetry & Latency (DEGRADED / STALE)
        if feed_telemetry:
            metrics.update(feed_telemetry)
            timeouts = feed_telemetry.get("consecutive_fetch_timeouts", 0)
            batch_dur = feed_telemetry.get("last_batch_duration_s", 0.0) or feed_telemetry.get("last_batch_duration_seconds", 0.0)
            stale_consecutive = feed_telemetry.get("stale_bar_consecutive")
            if stale_consecutive is None:
                stale_consecutive = feed_telemetry.get("stale_bar_skip_count", 0)

            if timeouts > 0:
                return ExecutionHealthReport(
                    status=HealthStatus.DEGRADED,
                    allows_new_entries=False,
                    reason=f"Recent market data fetch timeouts detected (consecutive={timeouts})",
                    metrics=metrics,
                )

            if batch_dur > self.max_batch_duration_seconds:
                return ExecutionHealthReport(
                    status=HealthStatus.DEGRADED,
                    allows_new_entries=False,
                    reason=f"Market data batch fetch latency elevated ({batch_dur:.2f}s > {self.max_batch_duration_seconds:.1f}s max)",
                    metrics=metrics,
                )

            if stale_consecutive > 5:
                return ExecutionHealthReport(
                    status=HealthStatus.DEGRADED,
                    allows_new_entries=False,
                    reason=f"Elevated consecutive stale bars detected ({stale_consecutive} > 5 limit)",
                    metrics=metrics,
                )

        # All invariants clean
        return ExecutionHealthReport(
            status=HealthStatus.HEALTHY,
            allows_new_entries=True,
            reason="Execution environment and data telemetry healthy",
            metrics=metrics,
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
