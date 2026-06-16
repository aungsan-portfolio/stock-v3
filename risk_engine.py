"""
risk_engine.py - Pure, offline portfolio-risk math for Phase 3 (H1, H3).

No IBKR / network / file dependency: every function takes plain numbers and
returns a plain bool/float, so the account circuit-breaker logic is fully
unit-testable offline. ibkr_bridge.py reads broker equity / positions and feeds
them in.

Scope (Phase 3):
  * account drawdown halt (H3 groundwork) -- block NEW entries when equity has
    fallen too far below the start-of-day snapshot;
  * per-symbol exposure cap (H3 groundwork) -- refuse a new position whose value
    would exceed a fraction of equity.

The daily-loss kill-switch (H1) itself lives in risk_state (it owns the
file-backed start-of-day snapshot); this module only adds the drawdown/exposure
checks. Full sector/correlation guard is intentionally OUT of Phase 3.
"""
from __future__ import annotations

import math

import config


def _f(value, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def drawdown_fraction(start_equity, current_equity) -> float:
    """Fraction by which equity has fallen below the start-of-day snapshot.

    0.0 when flat or up; positive when down. Returns 0.0 for an invalid/missing
    baseline (cannot claim a drawdown without a valid start).
    """
    start = _f(start_equity)
    cur = _f(current_equity)
    if start <= 0:
        return 0.0
    return max((start - cur) / start, 0.0)


def drawdown_halt_breached(start_equity, current_equity, max_dd_pct=None) -> bool:
    """True when the account drawdown halt should block NEW entries (H3).

    Conservative: returns False when the baseline is missing or the cap is
    disabled (<= 0), so it can never spuriously halt; it only ever blocks on a
    genuine, configured breach.
    """
    if max_dd_pct is None:
        max_dd_pct = getattr(config, "ACCOUNT_MAX_DRAWDOWN_PCT", 0.0)
    cap = _f(max_dd_pct)
    if cap <= 0:
        return False
    return drawdown_fraction(start_equity, current_equity) >= cap


def symbol_exposure_exceeded(new_value, existing_value, equity, max_pct=None) -> bool:
    """True when adding `new_value` would push this symbol's exposure above the
    per-symbol cap (H3 groundwork).

    Conservative: returns False when equity is unknown (<= 0) or the cap is
    disabled, so it never blocks without a real, configured breach.
    """
    if max_pct is None:
        max_pct = getattr(config, "MAX_SYMBOL_EXPOSURE_PCT", 0.0)
    cap = _f(max_pct)
    eq = _f(equity)
    if eq <= 0 or cap <= 0:
        return False
    total = _f(existing_value) + _f(new_value)
    return (total / eq) > cap
