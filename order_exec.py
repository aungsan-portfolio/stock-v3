"""
order_exec.py - Pure, offline order-execution logic for Phase 2 (H4-H9).

Phase 2 of the live-trading readiness plan (see
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md). This module has NO ib_insync /
network dependency: every function takes plain data or a *duck-typed* trade-like
object (anything exposing ``.orderStatus.status/filled/remaining/avgFillPrice``)
and returns plain values. That makes the order-robustness logic fully
unit-testable offline with fake Trade objects -- no live IBKR required.

ibkr_bridge.py uses these helpers to:

  * classify an order's REAL outcome -- FILLED / PARTIALLY_FILLED / WORKING /
    REJECTED / TIMEOUT -- instead of trusting "accepted" (H4/H5);
  * record a trade and size protective children from the ACTUAL filled qty and
    ``avgFillPrice`` (H6), never the intended qty;
  * verify the protective child stop is live after a parent fill, or flatten (H7);
  * decide close-order retry/escalation until ``remaining == 0`` (H8);
  * build a deterministic ``orderRef`` and detect duplicates on restart (H9).

The protective-order-type vocabulary is shared with ``live_invariants`` so there
is a single source of truth for "what counts as a downside stop".
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import live_invariants

# ── Explicit order outcomes ──────────────────────────────────────────────────
# "accepted"/"acknowledged" is deliberately NOT an outcome here: the whole point
# of Phase 2 is that broker acknowledgement is not a fill.
FILLED = "FILLED"                      # fully filled, remaining == 0
PARTIALLY_FILLED = "PARTIALLY_FILLED"  # some shares filled, remaining > 0 (or partial-then-cancelled)
WORKING = "WORKING"                    # live at broker, nothing filled yet
REJECTED = "REJECTED"                  # cancelled/rejected/inactive with no fill
TIMEOUT = "TIMEOUT"                    # we stopped waiting; still working, nothing filled

# Mirror ib_insync OrderStatus state sets so this module needs no ib import.
DONE_STATES = {"Filled", "Cancelled", "ApiCancelled"}
ACTIVE_STATES = {"Submitted", "PendingSubmit", "PreSubmitted", "ApiPending"}
BAD_STATES = {"Cancelled", "ApiCancelled", "Inactive", "Rejected"}
# Terminal = no more fills are coming. ib_insync's DoneStates omits the
# "Rejected"/"Inactive" strings, so include the bad states too: a rejected order
# must stop the await loop promptly and resolve as REJECTED, not TIMEOUT.
TERMINAL_STATES = DONE_STATES | BAD_STATES

_EPS = 1e-9


# ── Small, defensive coercion helpers ────────────────────────────────────────
def _f(value, default: float = 0.0) -> float:
    """Coerce to a finite float; return ``default`` on junk/NaN/inf."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _norm(value) -> str:
    return str(value or "").strip().upper()


def _id(value) -> str:
    """Normalise an order id (int/str/None) to a comparable string; '' for 0/None."""
    if value in (None, 0, "0", ""):
        return ""
    return str(value).strip()


def _order_status(trade):
    return getattr(trade, "orderStatus", None)


def status_of(trade) -> str:
    return str(getattr(_order_status(trade), "status", "") or "")


def filled_qty(trade) -> float:
    return max(_f(getattr(_order_status(trade), "filled", 0.0)), 0.0)


def remaining_qty(trade) -> float:
    return max(_f(getattr(_order_status(trade), "remaining", 0.0)), 0.0)


def avg_fill_price(trade) -> float:
    return max(_f(getattr(_order_status(trade), "avgFillPrice", 0.0)), 0.0)


def is_terminal(trade) -> bool:
    """True when no more fills are coming (Filled/Cancelled/Rejected/Inactive)."""
    return status_of(trade) in TERMINAL_STATES


# ── Outcome value object ─────────────────────────────────────────────────────
@dataclass
class OrderResult:
    """The verified result of an order attempt. ``outcome`` is one of the module
    constants. ``filled``/``remaining``/``avg_fill_price`` come from the broker's
    orderStatus -- the ACTUAL numbers Phase 2 acts on.

    ``protective_ok`` records whether a confirmed long entry is backed by a live
    protective child stop (H7). ``aborted`` means we entered but had to
    emergency-flatten (e.g. the stop could not be verified), so the position is
    NOT held -- it must neither record a trade nor occupy a slot.
    """
    outcome: str
    status: str = ""
    filled: float = 0.0
    remaining: float = 0.0
    avg_fill_price: float = 0.0
    protective_ok: bool = True
    aborted: bool = False

    @property
    def is_filled(self) -> bool:
        return self.outcome == FILLED and not self.aborted

    @property
    def is_partial(self) -> bool:
        return self.outcome == PARTIALLY_FILLED and not self.aborted

    @property
    def has_fill(self) -> bool:
        """Did any shares actually change hands (and we still hold them)?"""
        return self.filled > _EPS and not self.aborted

    @property
    def occupies_slot(self) -> bool:
        """Does this attempt tie up a position slot (held or resting)?

        FILLED/PARTIAL/WORKING/TIMEOUT all leave something at the broker (a
        position and/or a resting order), so they occupy a MAX_OPEN_POSITIONS
        slot. REJECTED frees it. An aborted (emergency-flattened) attempt holds
        nothing.
        """
        if self.aborted:
            return False
        return self.outcome in (FILLED, PARTIALLY_FILLED, WORKING, TIMEOUT)


def classify_trade(trade) -> OrderResult:
    """Classify a trade from a single orderStatus snapshot (H4).

    FILLED requires status 'Filled' OR (remaining == 0 AND filled > 0) and a
    non-bad status. A bad status (Cancelled/Rejected/Inactive) with a partial
    fill is PARTIALLY_FILLED; with no fill it is REJECTED. Any other state with a
    positive fill is PARTIALLY_FILLED; otherwise WORKING.
    """
    status = status_of(trade)
    f = filled_qty(trade)
    r = remaining_qty(trade)
    avg = avg_fill_price(trade)

    if status == "Filled" or (r <= _EPS and f > _EPS and status not in BAD_STATES):
        outcome = FILLED
    elif status in BAD_STATES:
        outcome = PARTIALLY_FILLED if f > _EPS else REJECTED
    elif f > _EPS:
        outcome = PARTIALLY_FILLED
    else:
        outcome = WORKING
    return OrderResult(outcome=outcome, status=status, filled=f, remaining=r, avg_fill_price=avg)


def outcome_on_timeout(trade) -> OrderResult:
    """Resolve a trade that did not reach a terminal state before we gave up.

    A positive partial fill is a real (partial) position -> PARTIALLY_FILLED.
    Nothing filled -> TIMEOUT (still resting at the broker, but unconfirmed; the
    caller must NOT record a trade for it).
    """
    res = classify_trade(trade)
    if res.outcome in (FILLED, REJECTED):
        return res  # a terminal status that slipped past the loop: keep it honest
    if res.filled > _EPS:
        res.outcome = PARTIALLY_FILLED
        return res
    return OrderResult(
        outcome=TIMEOUT, status=res.status, filled=0.0,
        remaining=res.remaining, avg_fill_price=0.0,
    )


# ── Record / sizing decisions (H6) ───────────────────────────────────────────
def should_record_trade(result: OrderResult) -> bool:
    """record_trade() may be called ONLY for a real, held fill (H4/H6).

    Accepted-but-unfilled, rejected, timed-out, and emergency-aborted attempts
    must never count toward the daily-trade cap.
    """
    return (
        result.outcome in (FILLED, PARTIALLY_FILLED)
        and result.filled > _EPS
        and not result.aborted
    )


def protective_exit_qty(parent_filled_qty) -> int:
    """The exact share count a protective child / close must use = ACTUAL filled
    qty, floored to whole shares (H6). Never the intended qty.

    Over-sizing a long's protective SELL is dangerous: if it triggers it would
    sell more than we hold and open an accidental short. So this is a floor, and
    callers resize an over-sized native bracket child down to it.
    """
    q = _f(parent_filled_qty, 0.0)
    if q <= 0:
        return 0
    return int(math.floor(q + _EPS))


# ── Deterministic orderRef / idempotency (H9) ────────────────────────────────
def deterministic_order_ref(day, symbol, action) -> str:
    """Stable per-intent key: ``YYYY-MM-DD:SYMBOL:ACTION``.

    The same intended order on the same day maps to the same ref, so a duplicate
    placed after a restart/mid-flight is detectable (see has_order_ref) and two
    working orders sharing a ref are an idempotency violation
    (live_invariants.duplicate_order_refs).
    """
    return f"{str(day).strip()}:{_norm(symbol)}:{_norm(action)}"


def order_refs_in_use(working_orders) -> set:
    """Set of non-blank orderRefs across the given working orders."""
    out = set()
    for o in (working_orders or []):
        ref = str(o.get("order_ref", "") or "").strip()
        if ref:
            out.add(ref)
    return out


def has_order_ref(order_ref, working_orders) -> bool:
    """True when an order carrying ``order_ref`` is already working at the broker.

    Used as the restart/mid-flight duplicate guard: if our deterministic ref is
    already live, do not place it again (H9).
    """
    target = str(order_ref or "").strip()
    if not target:
        return False
    return target in order_refs_in_use(working_orders)


# ── Protective-child verification (H7) ───────────────────────────────────────
def find_protective_children(symbol, working_orders) -> list:
    """All working protective SELL (stop/trailing) orders for ``symbol``.

    Protective-type vocabulary is shared with live_invariants.is_protective_sell
    (a plain SELL LIMIT take-profit is NOT downside protection).
    """
    sym = _norm(symbol)
    return [
        o for o in (working_orders or [])
        if _norm(o.get("symbol")) == sym and live_invariants.is_protective_sell(o)
    ]


def verify_protective_child(symbol, parent_id, filled_qty_value, working_orders) -> dict:
    """Confirm a filled long is backed by a live protective child stop (H7).

    Returns ``{"ok": bool, "reason": str, "child": dict|None}``.

    A child protects the position when it is a working protective SELL for the
    symbol whose quantity COVERS the actual filled qty (qty >= filled). A child
    whose ``parent_id`` matches the filled parent is preferred, but a standalone
    protective stop (e.g. one we re-placed after a partial fill, parentId 0) also
    counts. ``reason`` distinguishes a wholly missing stop from one that is too
    small to cover the fill, so the caller can flatten vs. resize.
    """
    filled = _f(filled_qty_value, 0.0)
    children = find_protective_children(symbol, working_orders)
    if not children:
        return {"ok": False, "reason": "no_protective_child", "child": None}

    matched = [o for o in children if _id(o.get("parent_id")) and _id(o.get("parent_id")) == _id(parent_id)]
    pool = matched or children

    covering = [o for o in pool if _f(o.get("qty", 0.0)) + _EPS >= filled]
    if filled <= 0 or not covering:
        # A stop exists but does not cover the filled qty (or filled unknown).
        return {"ok": False, "reason": "child_qty_below_filled", "child": pool[0]}
    return {"ok": True, "reason": "ok", "child": covering[0]}


def child_needs_resize(child_qty, filled_qty_value) -> bool:
    """True when an over-sized exit child must be shrunk to the filled qty (H6).
    Resizing DOWN to the real fill prevents an accidental short if the order
    triggers on a partial position.
    """
    return _f(child_qty, 0.0) > _f(filled_qty_value, 0.0) + _EPS


def exit_children(symbol, exit_action, working_orders) -> list:
    """All working orders for `symbol` on the EXIT side (``exit_action`` = SELL
    for a long). This is every resting order that would reduce/close the
    position -- protective stop, trailing stop, AND take-profit LIMIT alike --
    not just the stop family. Any of them can over-sell if its quantity exceeds
    what we actually hold, so all of them must be reconciled against the fill.
    """
    sym = _norm(symbol)
    act = _norm(exit_action)
    return [
        o for o in (working_orders or [])
        if _norm(o.get("symbol")) == sym and _norm(o.get("action")) == act
    ]


def oversized_exit_children(symbol, exit_action, filled_qty_value, working_orders) -> list:
    """Exit children whose quantity exceeds the actual filled qty (H6).

    A non-empty result is an accidental-short trap: a resting SELL (stop OR
    take-profit) could sell more than we hold and flip the position short if it
    triggers. The caller must cancel/resize them to the filled qty -- or flatten.
    """
    filled = _f(filled_qty_value, 0.0)
    return [
        o for o in exit_children(symbol, exit_action, working_orders)
        if _f(o.get("qty", 0.0)) > filled + _EPS
    ]


# ── Close-order retry / escalation (H8) ──────────────────────────────────────
def close_followup(result: OrderResult, attempt: int, max_attempts: int) -> str:
    """Decide what to do after a close attempt: 'done' / 'escalate' / 'giveup'.

    A close is only ``done`` when fully filled (remaining == 0). Otherwise, while
    attempts remain, ``escalate`` to a more aggressive order (marketable-limit ->
    market). When attempts are exhausted, ``giveup`` so the caller can alert -- a
    position that would not close is never silently treated as flat.
    """
    if result.outcome == FILLED and result.remaining <= _EPS:
        return "done"
    if attempt >= max_attempts:
        return "giveup"
    return "escalate"
