"""
shutdown_guard.py - Phase 5B-3: graceful-shutdown planner for the ONE-SHOT bot.

See reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md (task 5.3). The Path A bot is
ONE-SHOT (connect -> execute -> disconnect -> exit); there is no daemon. A
"graceful shutdown" for a one-shot run therefore means exactly one thing: when we
tear the connection down we must leave the broker in a SAFE resting state -- every
open long still covered by its resting server-side GTC protective stop -- and we
must mark our own disconnect as intentional so the Phase-5B-2 watchdog does not
mis-read it as an unexpected mid-run drop.

This module is the PURE / offline half (like reconnect_watchdog's backoff,
live_invariants, and reconciliation.build_snapshot). It takes PLAIN data (the same
position / working-order dict shapes used everywhere else) and returns a plan
describing what shutdown WOULD do. It NEVER touches IBKR and -- by construction --
NEVER places, cancels, or modifies an order:

  * It does NOT open new entries (the plan's only verbs are "preserve" + "detect").
  * It PRESERVES every resting protective SELL stop; ``orders_to_cancel`` is always
    an empty list. Cancelling a resting GTC stop on the way out would strip a
    long's overnight protection -- the exact opposite of a graceful shutdown.
  * It only DETECTS unprotected longs so the caller (IBKRBridge.graceful_shutdown)
    can decide to run the EXISTING, already-tested protective-repair path
    (ensure_protective_stops) BEFORE disconnecting -- or, if it cannot price
    protection, halt rather than guess.

The protective-stop vocabulary is shared with ``live_invariants`` (single source
of truth for "what counts as a downside stop"), so this module adds no new notion
of protection.
"""
from __future__ import annotations

import live_invariants

# Reason codes recorded on the audit log / returned in the plan.
REASON_CLEAN = "clean"                  # nothing open, or every long already protected
REASON_UNPROTECTED_LONG = "unprotected_long"  # a long has no resting protective stop


def _is_gtc(order: dict) -> bool:
    return str(order.get("tif", "") or "").strip().upper() == "GTC"


def protective_stops(working_orders) -> list:
    """Every working protective SELL stop (stop/trailing family), across all
    symbols. This is the PRESERVE list: graceful shutdown must leave all of these
    resting at the broker untouched."""
    return [o for o in (working_orders or []) if live_invariants.is_protective_sell(o)]


def gtc_protective_stops(working_orders) -> list:
    """The subset of ``protective_stops`` that are GTC -- the only ones that keep
    protecting a long between the one-shot bot's runs (a DAY stop expires)."""
    return [o for o in protective_stops(working_orders) if _is_gtc(o)]


def open_longs(positions) -> list:
    """Sorted, de-duplicated symbols held LONG (qty > 0). Flat/short ignored."""
    out = []
    for pos in (positions or []):
        try:
            qty = float(pos.get("qty", 0.0))
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            continue
        sym = str(pos.get("symbol", "") or "").strip().upper()
        if sym:
            out.append(sym)
    return sorted(set(out))


def build_shutdown_plan(positions, working_orders) -> dict:
    """Build the PURE graceful-shutdown plan. Places/cancels nothing.

    Returns a dict describing the safe end state:
      * ``open_longs``                 -- all symbols held long
      * ``protected_longs``            -- longs already backed by a resting stop
      * ``unprotected_longs``          -- longs with NO resting protective SELL
                                          (caller must repair-or-halt before close)
      * ``protective_stops_preserved`` -- the resting protective SELL orders left
                                          untouched (PRESERVE list)
      * ``gtc_protective_stops``       -- the GTC subset of the preserve list
      * ``orders_to_cancel``           -- ALWAYS ``[]`` (shutdown cancels nothing)
      * ``opens_new_entries``          -- ALWAYS ``False`` (shutdown opens nothing)
      * ``safe``                       -- True iff there are no unprotected longs
      * ``reason``                     -- REASON_CLEAN / REASON_UNPROTECTED_LONG
    """
    positions = positions or []
    working_orders = working_orders or []

    longs = open_longs(positions)
    unprotected = live_invariants.unprotected_longs(positions, working_orders)
    protected = sorted(set(longs) - set(unprotected))
    preserved = protective_stops(working_orders)
    gtc = gtc_protective_stops(working_orders)

    safe = not unprotected
    return {
        "open_longs": longs,
        "protected_longs": protected,
        "unprotected_longs": unprotected,
        "protective_stops_preserved": preserved,
        "gtc_protective_stops": gtc,
        "n_open_longs": len(longs),
        "n_protective_stops": len(preserved),
        "n_gtc_protective_stops": len(gtc),
        "n_working_orders": len(working_orders),
        # Explicit, test-checkable invariants of a graceful shutdown.
        "orders_to_cancel": [],
        "opens_new_entries": False,
        "safe": bool(safe),
        "reason": REASON_CLEAN if safe else REASON_UNPROTECTED_LONG,
    }


def needs_protection_repair(plan: dict) -> bool:
    """True when at least one open long is unprotected, so the caller should run
    the existing ensure_protective_stops() repair BEFORE disconnecting."""
    return bool((plan or {}).get("unprotected_longs"))


def audit_fields(plan: dict) -> dict:
    """Flat, JSON-safe summary of a shutdown plan for the audit log."""
    plan = plan or {}
    return {
        "n_open_longs": plan.get("n_open_longs", 0),
        "n_protective_stops": plan.get("n_protective_stops", 0),
        "n_gtc_protective_stops": plan.get("n_gtc_protective_stops", 0),
        "unprotected_longs": list(plan.get("unprotected_longs", []) or []),
        "n_orders_to_cancel": len(plan.get("orders_to_cancel", []) or []),
        "opens_new_entries": bool(plan.get("opens_new_entries", False)),
        "safe": bool(plan.get("safe", False)),
        "reason": plan.get("reason", REASON_CLEAN),
    }


def summary_line(plan: dict) -> str:
    """One-line human summary for logs."""
    plan = plan or {}
    longs = plan.get("n_open_longs", 0)
    preserved = plan.get("n_protective_stops", 0)
    unprotected = plan.get("unprotected_longs", []) or []
    state = "SAFE" if plan.get("safe") else "NEEDS REPAIR"
    tail = "" if not unprotected else f"; unprotected={','.join(unprotected)}"
    return (f"graceful shutdown [{state}]: {longs} long(s), "
            f"{preserved} protective stop(s) preserved, 0 cancelled{tail}")
