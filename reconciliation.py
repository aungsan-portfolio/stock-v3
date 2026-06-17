"""
reconciliation.py - Pure, offline startup reconciliation (Phase 5A, H18).

Broker = source of truth. On startup the bot must rebuild its view of the world
from what the BROKER actually reports -- open positions and working orders --
never from a local file. This module is the PURE half of that: it takes PLAIN
data (the same list-of-dicts shapes `live_invariants` / `order_exec` already use)
and returns a deterministic "broker-truth snapshot" describing what is safe and
what is not. It has NO ib_insync / network dependency, so it is fully unit-
testable offline and is reused by `IBKRBridge.reconcile_startup_state()`.

It DOES NOT place, cancel, or modify any order. It only describes state. The
bridge decides what (if anything) to repair, and only ever through the existing
Phase-3 protective-repair path (`ensure_protective_stops`), which never opens a
new position.

Input shapes (same as live_invariants / order_exec, trivial to build in tests):
  position      = {"symbol": str, "qty": float}
  working_order = {"symbol": str, "action": "BUY"|"SELL", "order_type": str,
                   "order_ref": str, "qty": float, "tif": str, ...}

The snapshot classifies a long as PROTECTED only when a resting **GTC** stop
covers its actual qty -- the same invariant the order path enforces post-fill
(`order_exec.has_gtc_protective_stop`). A take-profit LIMIT or a non-GTC (DAY)
stop is NOT downside protection between the one-shot bot's runs.
"""
from __future__ import annotations

import live_invariants
import order_exec


def _norm(value) -> str:
    return str(value or "").strip().upper()


def _qty(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def long_positions(positions) -> list:
    """[(symbol, qty)] for every position whose qty > 0 (long-only invariant)."""
    out = []
    for pos in (positions or []):
        qty = _qty(pos.get("qty", 0.0))
        sym = _norm(pos.get("symbol"))
        if sym and qty > 0:
            out.append((sym, qty))
    return out


def protected_longs(positions, working_orders) -> list:
    """Sorted symbols that are LONG and backed by a resting GTC stop covering the
    held qty (broker-truth "safe long"). Mirrors the post-fill invariant."""
    out = [
        sym for sym, qty in long_positions(positions)
        if order_exec.has_gtc_protective_stop(sym, qty, working_orders)
    ]
    return sorted(set(out))


def unprotected_longs(positions, working_orders) -> list:
    """Sorted symbols that are LONG but have NO resting GTC stop covering the qty.

    GTC-aware on purpose: a DAY stop or a take-profit LIMIT does not protect the
    position between runs, so it is reported here for repair. This is the set the
    bridge hands to the existing Phase-3 repair path."""
    out = [
        sym for sym, qty in long_positions(positions)
        if not order_exec.has_gtc_protective_stop(sym, qty, working_orders)
    ]
    return sorted(set(out))


def working_order_refs(working_orders) -> list:
    """The (non-blank) orderRef of every working order, in input order."""
    refs = []
    for o in (working_orders or []):
        ref = str(o.get("order_ref", "") or "").strip()
        if ref:
            refs.append(ref)
    return refs


def duplicate_order_refs(working_orders) -> list:
    """Sorted deterministic orderRefs that appear on more than one working order
    (an idempotency violation -- a duplicate survived a restart, H9)."""
    return live_invariants.duplicate_order_refs(working_order_refs(working_orders))


def orphan_exit_orders(positions, working_orders) -> list:
    """Working SELL (exit-side) orders for a symbol the account is NOT long.

    A resting SELL with no covering long is an accidental-short trap: if it fires
    it sells shares we do not hold and opens a short. The held qty is taken from
    the broker positions (source of truth); any SELL whose symbol nets <= 0 is an
    orphan. Returns the offending working-order dicts (stop, trail, OR take-profit
    LIMIT alike -- all can over-sell). Empty list == none detected."""
    held = {}
    for pos in (positions or []):
        sym = _norm(pos.get("symbol"))
        if sym:
            held[sym] = held.get(sym, 0.0) + _qty(pos.get("qty", 0.0))

    orphans = []
    for o in (working_orders or []):
        if _norm(o.get("action")) != "SELL":
            continue
        sym = _norm(o.get("symbol"))
        if held.get(sym, 0.0) <= 0:
            orphans.append(o)
    return orphans


def build_snapshot(positions, working_orders) -> dict:
    """Build the broker-truth startup snapshot from PLAIN broker state.

    Pure: describes state, never mutates it. Keys:
      open_positions   : [(symbol, qty)] long positions (qty > 0)
      n_working_orders : count of working orders seen
      protected_longs  : longs covered by a resting GTC stop
      unprotected_longs: longs with NO covering GTC stop (repair candidates)
      duplicate_refs   : orderRefs appearing on >1 working order (idempotency)
      orphan_exits     : working SELL orders with no covering long (short trap)
      clean            : True when there is nothing to repair or alert on
    """
    longs = long_positions(positions)
    unprotected = unprotected_longs(positions, working_orders)
    dupes = duplicate_order_refs(working_orders)
    orphans = orphan_exit_orders(positions, working_orders)
    return {
        "open_positions": longs,
        "n_working_orders": len(working_orders or []),
        "protected_longs": protected_longs(positions, working_orders),
        "unprotected_longs": unprotected,
        "duplicate_refs": dupes,
        "orphan_exits": orphans,
        "clean": not (unprotected or dupes or orphans),
    }


def audit_fields(snapshot: dict) -> dict:
    """Flatten a snapshot into JSON-safe scalars/lists for the audit log."""
    orphan_syms = sorted({_norm(o.get("symbol")) for o in snapshot.get("orphan_exits", [])})
    return {
        "n_long": len(snapshot.get("open_positions", [])),
        "n_working_orders": int(snapshot.get("n_working_orders", 0)),
        "protected_longs": list(snapshot.get("protected_longs", [])),
        "unprotected_longs": list(snapshot.get("unprotected_longs", [])),
        "duplicate_refs": list(snapshot.get("duplicate_refs", [])),
        "orphan_exit_symbols": orphan_syms,
        "clean": bool(snapshot.get("clean", False)),
    }


def summary_line(snapshot: dict) -> str:
    """One-line human-readable summary for console / logs."""
    return (
        "broker-truth: longs={n_long} working_orders={n_wo} "
        "protected={prot} unprotected={unprot} duplicate_refs={dups} "
        "orphan_exits={orph} clean={clean}"
    ).format(
        n_long=len(snapshot.get("open_positions", [])),
        n_wo=int(snapshot.get("n_working_orders", 0)),
        prot=snapshot.get("protected_longs", []),
        unprot=snapshot.get("unprotected_longs", []),
        dups=snapshot.get("duplicate_refs", []),
        orph=sorted({_norm(o.get("symbol")) for o in snapshot.get("orphan_exits", [])}),
        clean=bool(snapshot.get("clean", False)),
    )
