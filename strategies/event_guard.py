"""
strategies/event_guard.py -- Event-aware guardrail for scheduled high-impact market events.

Layers ON TOP of regime_engine + halt_manager without replacing them.
Guarantees fail-closed safety on scheduled macro events (e.g. Fed keynotes, major earnings).
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, time
import logging
from typing import Dict, Any, Optional
import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


@dataclass(frozen=True)
class EventDecision:
    throttle: float          # position size multiplier (0.0 = block)
    block_entries: bool      # hard halt entries this cycle
    atr_floor_mult: float    # widen ATR stop floor on event days
    tag: Optional[str]       # reason string for logging/journal
    affected_symbols: Optional[set] = None


# Hard-coded high-impact windows (times are ET) ---------------------
HIGH_IMPACT_EVENTS: Dict[str, Dict[str, Any]] = {
    # NVDA reports Wed Aug 26 after close -> Thursday morning gap risk
    "2026-08-27": {
        "reason": "NVDA_EARNINGS_GAP",
        "mode": "DEFENSIVE",
        "throttle": 0.5,
        "atr_floor_mult": 2.0,
        "affected_symbols": {"NVDA"},
    },
    # Jackson Hole: Warsh first keynote ~10:00 ET Friday
    "2026-08-28": {
        "reason": "JACKSON_HOLE_WARSH",
        "mode": "HEADLINE_BLACKOUT",
        "blackout": (time(9, 30), time(11, 15)),
        "throttle": 0.5,
        "atr_floor_mult": 2.0,
        "affected_symbols": set(),
    },
    # FOMC Decision: Oct 28, 2026 14:00 ET ± 15m to 15:00 ET press conference (C2)
    "2026-10-28": {
        "reason": "FOMC_RATE_DECISION",
        "mode": "HEADLINE_BLACKOUT",
        "blackout": (time(13, 45), time(15, 0)),
        "throttle": 0.5,
        "atr_floor_mult": 2.0,
        "affected_symbols": set(),
    },
    # NFP Employment Report: Nov 6, 2026 08:30 ET ± 15m (C2)
    "2026-11-06": {
        "reason": "NFP_EMPLOYMENT_REPORT",
        "mode": "HEADLINE_BLACKOUT",
        "blackout": (time(8, 15), time(9, 45)),
        "throttle": 0.5,
        "atr_floor_mult": 1.5,
        "affected_symbols": set(),
    },
    # CPI Inflation Report: Nov 13, 2026 08:30 ET ± 15m (C2)
    "2026-11-13": {
        "reason": "CPI_INFLATION_REPORT",
        "mode": "HEADLINE_BLACKOUT",
        "blackout": (time(8, 15), time(9, 45)),
        "throttle": 0.5,
        "atr_floor_mult": 1.5,
        "affected_symbols": set(),
    },
}


def evaluate_event(now_et: Optional[datetime] = None) -> EventDecision:
    """
    Single entry point for scheduled high-impact macro & earnings event risk.
    Returns a normalized EventDecision.
    """
    try:
        if now_et is None:
            now_et = datetime.now(ET)
        elif now_et.tzinfo is None:
            now_et = ET.localize(now_et)
        else:
            now_et = now_et.astimezone(ET)

        key = now_et.strftime("%Y-%m-%d")
        evt = HIGH_IMPACT_EVENTS.get(key)

        if not evt:
            return EventDecision(throttle=1.0, block_entries=False, atr_floor_mult=1.0, tag=None, affected_symbols=set())

        reason = evt.get("reason", "SCHEDULED_EVENT")
        affected = set(evt.get("affected_symbols", set()))

        # Full blackout window (e.g. Fed keynote / FOMC / CPI / NFP) -> hard block entries
        import config
        macro_blackout_enabled = getattr(config, "MACRO_EVENT_BLACKOUT", True)
        if "blackout" in evt and macro_blackout_enabled:
            start, end = evt["blackout"]
            current_time = now_et.time()
            if start <= current_time <= end:
                return EventDecision(
                    throttle=0.0,
                    block_entries=True,
                    atr_floor_mult=float(evt.get("atr_floor_mult", 2.0)),
                    tag=f"{reason}_BLACKOUT",
                    affected_symbols=affected,
                )

        # Defensive throttle for the rest of any flagged day
        if evt.get("mode") in ("DEFENSIVE", "HEADLINE_BLACKOUT"):
            throttle = float(evt.get("throttle", 0.5))
            atr_floor = float(evt.get("atr_floor_mult", 2.0))
            return EventDecision(
                throttle=throttle,
                block_entries=False,
                atr_floor_mult=atr_floor,
                tag=reason,
                affected_symbols=affected,
            )

        return EventDecision(throttle=1.0, block_entries=False, atr_floor_mult=1.0, tag=None, affected_symbols=set())
    except Exception as exc:
        logger.error("[EVENT GUARD] Evaluation error: %s -- failing CLOSED", exc)
        return EventDecision(
            throttle=0.0,
            block_entries=True,
            atr_floor_mult=2.0,
            tag="EVAL_EXCEPTION_BLACKOUT",
        )
