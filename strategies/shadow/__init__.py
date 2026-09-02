"""
strategies.shadow -- P1.6 Shadow Alpha Research & Simulation Package.

Strictly isolated, broker-free pipeline for shadow strategy evaluation,
excursion tracking, and conservative statistical expectancy auditing.
"""

from strategies.shadow.models import (
    TimeBucket,
    AlignmentStatus,
    ShadowOutcome,
    StreamType,
    ShadowCandidate,
    ShadowTrade,
)
from strategies.shadow.pricing import (
    PricingSource,
    QuoteSnapshot,
    resolve_entry_price,
    resolve_exit_price,
)
from strategies.shadow.storage import ShadowStorageManager
from strategies.shadow.orb_confirmed import ConfirmedORBStrategy
from strategies.shadow.lifecycle import ShadowLifecycleManager
from strategies.shadow.analytics import (
    ResearchGateStatus,
    ShadowAnalyticsReport,
    evaluate_shadow_metrics,
)

__all__ = [
    "TimeBucket",
    "AlignmentStatus",
    "ShadowOutcome",
    "StreamType",
    "ShadowCandidate",
    "ShadowTrade",
    "PricingSource",
    "QuoteSnapshot",
    "resolve_entry_price",
    "resolve_exit_price",
    "ShadowStorageManager",
    "ConfirmedORBStrategy",
    "ShadowLifecycleManager",
    "ResearchGateStatus",
    "ShadowAnalyticsReport",
    "evaluate_shadow_metrics",
]
