"""
live_invariants.py - Pure, offline-checkable safety invariants for live trading.

Phase 0 of the live-trading readiness plan (see
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md). These functions take PLAIN data
(lists of dicts) and have NO IBKR/network dependency, so they can be unit-tested
offline and reused by:
  * the `live-readiness` self-check command (Phase 0), and
  * the startup reconciliation / unprotected-position scan (Phase 3 & 5).

Building the checkers now gives every later phase a deterministic testbed. The
checkers describe the *target* invariants; today's code does not yet satisfy all
of them (e.g. positions can be left without a server-side protective stop), which
is exactly what the `live-readiness` scorecard is meant to surface.

Input shapes (intentionally simple so they are trivial to construct in tests and
to adapt from ib_insync objects):
  position      = {"symbol": str, "qty": float}
  working_order = {"symbol": str, "action": "BUY"|"SELL", "order_type": str,
                   "order_ref": str (optional)}
"""
from __future__ import annotations

# ib_insync / IBKR order-type strings that act as a protective exit.
# TRAIL = trailing stop, STP = stop, "STP LMT" = stop-limit, MKT-on-stop, etc.
PROTECTIVE_ORDER_TYPES = {
    "STP", "STOP", "STP LMT", "STOP_LIMIT", "STOPLIMIT",
    "TRAIL", "TRAILING_STOP", "TRAIL LIMIT", "TRAILLIMIT", "MKT PRT",
}


def _norm(value) -> str:
    return str(value or "").strip().upper()


def is_protective_sell(order: dict) -> bool:
    """True when a working order is a protective SELL (stop/trailing) for a long.

    A long position is "protected" by a working SELL order whose type is a
    stop/trailing family order. A plain SELL LIMIT (take-profit) is NOT counted
    as downside protection here on purpose.
    """
    if _norm(order.get("action")) != "SELL":
        return False
    return _norm(order.get("order_type")) in PROTECTIVE_ORDER_TYPES


def protective_sells_for(symbol: str, working_orders) -> list:
    """All working protective SELL orders for a symbol."""
    sym = _norm(symbol)
    return [
        o for o in (working_orders or [])
        if _norm(o.get("symbol")) == sym and is_protective_sell(o)
    ]


def unprotected_longs(positions, working_orders) -> list:
    """Symbols that are LONG but have no working protective SELL order.

    This is the core live-trading invariant for Path A (server-side protection):
    every open long must be backed by a resting stop at the broker so the
    position is covered even while the one-shot bot is not running. Returns a
    sorted list of offending symbols (empty list == invariant holds).
    """
    offenders = []
    for pos in (positions or []):
        try:
            qty = float(pos.get("qty", 0.0))
        except (TypeError, ValueError):
            qty = 0.0
        if qty <= 0:
            continue  # flat or short; this long-only invariant ignores it
        sym = _norm(pos.get("symbol"))
        if not sym:
            continue
        if not protective_sells_for(sym, working_orders):
            offenders.append(sym)
    return sorted(set(offenders))


def duplicate_order_refs(order_refs) -> list:
    """Order-reference keys that appear more than once (idempotency violation).

    Phase 2 introduces a deterministic orderRef per intended order. Two working
    orders sharing a ref means a duplicate was placed. Returns sorted duplicates.
    """
    seen, dupes = set(), set()
    for ref in (order_refs or []):
        key = _norm(ref)
        if not key:
            continue
        if key in seen:
            dupes.add(key)
        seen.add(key)
    return sorted(dupes)


def adapt_positions(ib_positions) -> list:
    """Adapt ib_insync positions() output to the plain shape above.

    Defensive/duck-typed: only used by the `live-readiness --connect` path, never
    in the offline unit tests.
    """
    out = []
    for p in (ib_positions or []):
        try:
            out.append({
                "symbol": str(p.contract.symbol).upper().strip(),
                "qty": float(p.position),
            })
        except Exception:
            continue
    return out


def adapt_working_orders(ib_open_trades) -> list:
    """Adapt ib_insync openTrades() output to the plain shape above.

    The ``parent_id``, ``qty`` and ``status`` keys are additive (Phase 2): they
    let order_exec.verify_protective_child confirm a stop child is attached to
    the filled parent and covers the actual filled qty. Callers that only need
    the base shape (e.g. unprotected_longs) simply ignore them.
    """
    out = []
    for t in (ib_open_trades or []):
        try:
            order = t.order
            out.append({
                "symbol": str(t.contract.symbol).upper().strip(),
                "action": str(order.action).upper().strip(),
                "order_type": str(order.orderType).upper().strip(),
                "order_ref": str(getattr(order, "orderRef", "") or "").strip(),
                "parent_id": int(getattr(order, "parentId", 0) or 0),
                "qty": float(getattr(order, "totalQuantity", 0) or 0),
                "status": str(getattr(getattr(t, "orderStatus", None), "status", "") or "").strip(),
            })
        except Exception:
            continue
    return out
