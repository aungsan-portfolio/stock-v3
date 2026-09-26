"""
veto_reasons.py -- Machine-readable authoritative typed veto and rejection reason codes.
Canonical single source of truth for execution gates and telemetry attribution.
"""
from enum import Enum


class VetoReason(str, Enum):
    # Hard geometry / structural vetoes (eligible for hard-veto skip suppression)
    NOISE_FLOOR = "NOISE_FLOOR"
    CRITICAL_STOP = "CRITICAL_STOP"
    INVALID_STRUCTURAL_STOP = "INVALID_STRUCTURAL_STOP"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"

    # Execution quote / market micro-structure vetoes
    STALE_QUOTE = "STALE_QUOTE"
    MAX_CHASE = "MAX_CHASE"
    EXCESSIVE_SPREAD = "EXCESSIVE_SPREAD"
    INVERTED_QUOTE = "INVERTED_QUOTE"
    ZERO_POSITION_SIZE = "ZERO_POSITION_SIZE"

    # Macro / benchmark / trend guard vetoes
    SPY_BEARISH_5M = "SPY_BEARISH_5M"
    SOXX_TECH_LONG_VETO = "SOXX_TECH_LONG_VETO"
    COMMODITY_CORRELATION_VETO = "COMMODITY_CORRELATION_VETO"
    TECH_CLUSTER_CAP = "TECH_CLUSTER_CAP"
    COUNTER_TREND_SHORT_VETO = "COUNTER_TREND_SHORT_VETO"

    # Confidence & scoring gates
    CONFIDENCE_BELOW_FLOOR = "CONFIDENCE_BELOW_FLOOR"
    LUNCH_CHOP_ZONE = "LUNCH_CHOP_ZONE"
    RSI_OVERBOUGHT = "RSI_OVERBOUGHT"
    RSI_OVERSOLD = "RSI_OVERSOLD"

    # System / Infrastructure / Universe gates
    LOW_COVERAGE = "LOW_COVERAGE"
    REGIME_DATA_DEGRADED = "REGIME_DATA_DEGRADED"
    REGIME_RISK_OFF = "REGIME_RISK_OFF"
    ENTRY_CUTOFF_TIME = "ENTRY_CUTOFF_TIME"
    ENTRY_READINESS_FAILED = "ENTRY_READINESS_FAILED"
    SESSION_COOLDOWN = "SESSION_COOLDOWN"
    CONSECUTIVE_LOSS_LIMIT = "CONSECUTIVE_LOSS_LIMIT"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    IDEMPOTENCY_HELD = "IDEMPOTENCY_HELD"
    STRATEGY_DISABLED = "STRATEGY_DISABLED"


# Hard-veto codes that legitimately trigger symbol-level skip suppression
# Strictly excludes generic indicator/data logging containing 'atr'
HARD_VETO_CODES = frozenset({
    VetoReason.NOISE_FLOOR,
    VetoReason.CRITICAL_STOP,
    VetoReason.INVALID_STRUCTURAL_STOP,
})
