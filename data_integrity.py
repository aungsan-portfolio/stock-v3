"""
data_integrity.py - Pure, offline data-integrity guards for Phase 4 (H11-H16).

Phase 4 of the live-trading readiness plan (see
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md). Like order_exec.py and
risk_engine.py this module has NO ib_insync / network / file dependency: every
function takes plain numbers / datetimes and returns plain values, so the
order-price sanity logic and the market-hours calendar are fully unit-testable
offline. ibkr_bridge.py reads the raw IBKR ticker fields (and the wall clock)
and feeds them in.

Scope (Phase 4):
  * evaluate_order_quote (H11/H12/H13/H16) - decide the price an order may use
    from the raw quote fields:
      - reject the plain prior-session ``close``/``delayedClose`` as an order
        price (it is yesterday's close, stale);
      - when real-time is REQUIRED (live), reject delayed/close fields entirely;
      - reject a crossed/locked book (bid >= ask) and a too-wide spread;
      - otherwise use the real-time last (when inside the book) or a sane
        bid/ask midpoint; in paper mode the 15-min delayed equivalents are
        allowed so paper trading keeps working without a real-time subscription.
  * decision_price_ok (H14) - refuse to act on a signal when the live order
    price has moved too far from the price the decision was made on (a stale /
    gapped signal).
  * is_regular_hours (H15) - US equity regular-hours gate for NEW entries, with
    a dependency-free computed holiday + early-close calendar. The bridge passes
    a US/Eastern datetime; this module never reads the clock itself.

This module is paper-safe and additive: it never connects, never places an
order, and never enables live trading. It only ever says "this price/time is
(not) usable for a NEW entry".
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from datetime import date, time, timedelta

# ── Quote sources / reject reasons ───────────────────────────────────────────
SOURCE_REALTIME_LAST = "realtime_last"
SOURCE_REALTIME_MID = "realtime_mid"
SOURCE_DELAYED_LAST = "delayed_last"
SOURCE_DELAYED_MID = "delayed_mid"
SOURCE_NONE = "none"

REASON_OK = "ok"
REASON_NO_DATA = "no_data"               # nothing usable populated
REASON_DELAYED_BLOCKED = "delayed_blocked"  # real-time required but only delayed available
REASON_STALE_CLOSE = "stale_close"       # only a prior-session close is available
REASON_CROSSED = "crossed_market"        # bid >= ask (locked/crossed book)
REASON_WIDE_SPREAD = "wide_spread"       # (ask-bid)/mid above the cap

_EPS = 1e-9

# ── US equity regular-hours calendar constants ───────────────────────────────
# Defined up-front so the market-hours helpers below can reference them. All are
# US/Eastern wall-clock times; the bridge supplies ``now`` already in ET.
REGULAR_OPEN = time(9, 30)    # 09:30 ET regular session open
REGULAR_CLOSE = time(16, 0)   # 16:00 ET regular session close
EARLY_CLOSE = time(13, 0)     # 13:00 ET half-day (early-close) session close


# ── Defensive coercion (mirrors order_exec / risk_engine) ────────────────────
def _f(value, default: float = 0.0) -> float:
    """Coerce to a finite float; return ``default`` on junk/NaN/inf."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _pos(value):
    """Finite, strictly-positive float, or None. A price must be > 0 to be real;
    0/negative/NaN/None are all "not a usable price"."""
    v = _f(value, 0.0)
    return v if v > 0 else None


def _finite_positive(value) -> float:
    """Finite, strictly-positive float, or 0.0. Same intent as ``_pos`` but
    returns 0.0 instead of None so callers can simply compare ``<= 0``
    (used by decision_price_ok)."""
    v = _f(value, 0.0)
    return v if v > 0 else 0.0


# ── Quote sanity ─────────────────────────────────────────────────────────────
@dataclass
class QuoteDecision:
    """The verdict on a raw quote for ORDER use.

    ``ok`` is True only when ``price`` (> 0) is safe to put on an order; on a
    reject ``price`` is 0.0 and ``reason`` explains why (so the caller can audit
    and refuse). ``source`` records which field the price came from.
    """
    ok: bool
    price: float
    source: str
    reason: str


def is_crossed(bid, ask) -> bool:
    """True for a crossed/locked book: bid >= ask. Either is a bad/illiquid
    snapshot we must not price an order from."""
    b, a = _pos(bid), _pos(ask)
    if b is None or a is None:
        return False
    return b >= a - _EPS


def spread_fraction(bid, ask) -> float:
    """(ask - bid) / midpoint, or 0.0 when bid/ask are not both usable. Negative
    spreads (crossed) clamp to 0.0; use is_crossed for that case."""
    b, a = _pos(bid), _pos(ask)
    if b is None or a is None:
        return 0.0
    mid = (a + b) / 2.0
    if mid <= 0:
        return 0.0
    return max((a - b) / mid, 0.0)


def spread_too_wide(bid, ask, max_pct) -> bool:
    """True when the bid/ask spread exceeds the cap (an illiquid / unreliable
    quote). Disabled (never too wide) when the cap is <= 0."""
    cap = _f(max_pct)
    if cap <= 0:
        return False
    return spread_fraction(bid, ask) > cap + _EPS


def _price_from_book(last, bid, ask, max_spread_pct, last_src, mid_src):
    """Shared bid/ask -> usable-price logic for both the real-time and the
    delayed branches. Returns a QuoteDecision, or None when there is no usable
    book (so the caller can fall back to a lone last/other source)."""
    b, a = _pos(bid), _pos(ask)
    if b is None or a is None:
        return None
    if is_crossed(b, a):
        return QuoteDecision(False, 0.0, SOURCE_NONE, REASON_CROSSED)
    if spread_too_wide(b, a, max_spread_pct):
        return QuoteDecision(False, 0.0, SOURCE_NONE, REASON_WIDE_SPREAD)
    mid = (a + b) / 2.0
    lst = _pos(last)
    # Prefer the traded last ONLY when it sits inside the (sane) book; otherwise
    # the midpoint is the more defensible order price.
    if lst is not None and (b - _EPS) <= lst <= (a + _EPS):
        return QuoteDecision(True, round(lst, 4), last_src, REASON_OK)
    return QuoteDecision(True, round(mid, 4), mid_src, REASON_OK)


def evaluate_order_quote(*, last=None, bid=None, ask=None, close=None,
                         delayed_last=None, delayed_close=None,
                         delayed_bid=None, delayed_ask=None,
                         require_realtime: bool = False,
                         max_spread_pct: float = 0.02) -> QuoteDecision:
    """Decide the price (if any) an order may use from the raw ticker fields
    (H11/H12/H13/H16).

    Priority:
      1. Real-time book (bid/ask) -> last-inside-book or midpoint, after
         crossed/spread checks.
      2. Real-time last alone (a real trade, but no usable book).
      3. If real-time is REQUIRED, stop here: only delayed/close left ->
         delayed_blocked (or no_data).
      4. (paper) Delayed book -> delayed last-inside-book or midpoint.
      5. (paper) Delayed last alone.
      6. Only a prior-session close (close/delayedClose) is left -> stale_close;
         a daily close is never an order price.
    """
    # 1) real-time book
    book = _price_from_book(last, bid, ask, max_spread_pct,
                            SOURCE_REALTIME_LAST, SOURCE_REALTIME_MID)
    if book is not None:
        return book
    # 2) real-time last alone
    rt_last = _pos(last)
    if rt_last is not None:
        return QuoteDecision(True, round(rt_last, 4), SOURCE_REALTIME_LAST, REASON_OK)

    # 3) real-time required but none available
    if require_realtime:
        if (_pos(delayed_last) or _pos(delayed_bid) or _pos(delayed_ask)
                or _pos(close) or _pos(delayed_close)):
            return QuoteDecision(False, 0.0, SOURCE_NONE, REASON_DELAYED_BLOCKED)
        return QuoteDecision(False, 0.0, SOURCE_NONE, REASON_NO_DATA)

    # 4) delayed book (paper)
    d_book = _price_from_book(delayed_last, delayed_bid, delayed_ask, max_spread_pct,
                              SOURCE_DELAYED_LAST, SOURCE_DELAYED_MID)
    if d_book is not None:
        return d_book
    # 5) delayed last alone (paper)
    d_last = _pos(delayed_last)
    if d_last is not None:
        return QuoteDecision(True, round(d_last, 4), SOURCE_DELAYED_LAST, REASON_OK)

    # 6) only a prior-session close left -> too stale for an order
    if _pos(close) is not None or _pos(delayed_close) is not None:
        return QuoteDecision(False, 0.0, SOURCE_NONE, REASON_STALE_CLOSE)
    return QuoteDecision(False, 0.0, SOURCE_NONE, REASON_NO_DATA)


# ── Decision-vs-order price agreement (H14) ──────────────────────────────────
def price_deviation(reference, actual) -> float:
    """|actual - reference| / reference, or 0.0 when either is not a usable
    price (cannot claim a deviation without both)."""
    r, a = _pos(reference), _pos(actual)
    if r is None or a is None:
        return 0.0
    return abs(a - r) / r


def decision_price_ok(decision_price, order_price, max_dev_pct) -> bool:
    """Return True when decision price and order price agree enough.

    Missing decision_price should not spuriously block because older signals may
    not carry a reliable live decision price. Missing/zero order_price must fail
    closed because an order cannot be priced safely.
    """
    op = _finite_positive(order_price)
    if op <= 0:
        return False

    dp = _finite_positive(decision_price)
    if dp <= 0:
        return True

    pct = max(0.0, float(max_dev_pct or 0.0))
    if pct <= 0:
        return True
    return abs(op - dp) / dp <= pct

def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous algorithm). Used for Good Friday."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th given weekday (Mon=0..Sun=6) of a month, e.g. 3rd Monday."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last given weekday of a month (e.g. last Monday in May)."""
    nxt = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last = nxt - timedelta(days=1)
    offset = (last.weekday() - weekday) % 7
    return last - timedelta(days=offset)


def _observed(d: date) -> date:
    """Standard observance for a fixed-date holiday: Saturday -> Friday before,
    Sunday -> Monday after."""
    if d.weekday() == 5:      # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:      # Sunday
        return d + timedelta(days=1)
    return d


def us_market_holidays(year: int) -> set:
    """Set of full-closure US equity market holidays for ``year`` (observed)."""
    good_friday = _easter(year) - timedelta(days=2)
    holidays = {
        _observed(date(year, 1, 1)),               # New Year's Day
        _nth_weekday(year, 1, 0, 3),               # MLK Jr. Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),               # Presidents' Day (3rd Mon Feb)
        good_friday,                               # Good Friday
        _last_weekday(year, 5, 0),                 # Memorial Day (last Mon May)
        _nth_weekday(year, 9, 0, 1),               # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),              # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),             # Christmas Day
    }
    # Juneteenth became a market holiday in 2022.
    if year >= 2022:
        holidays.add(_observed(date(year, 6, 19)))
    # Independence Day.
    holidays.add(_observed(date(year, 7, 4)))
    return holidays


def us_market_early_closes(year: int) -> dict:
    """Map of {date: close time(13:00)} for half-day sessions in ``year``.

    Covers the common NYSE early closes: the day after Thanksgiving, Christmas
    Eve (when a weekday), and July 3 (when both it and the 4th are weekdays).
    """
    out = {}
    thanksgiving = _nth_weekday(year, 11, 3, 4)
    out[thanksgiving + timedelta(days=1)] = EARLY_CLOSE      # Black Friday
    xmas_eve = date(year, 12, 24)
    if xmas_eve.weekday() < 5:
        out[xmas_eve] = EARLY_CLOSE
    jul3, jul4 = date(year, 7, 3), date(year, 7, 4)
    if jul3.weekday() < 5 and jul4.weekday() < 5:
        out[jul3] = EARLY_CLOSE
    return out


def session_close(d: date) -> time:
    """The session close time for a trading day (16:00, or 13:00 on a half-day)."""
    return us_market_early_closes(d.year).get(d, REGULAR_CLOSE)


def is_trading_day(d: date) -> bool:
    """True when ``d`` is a weekday that is not a full-closure holiday."""
    if d.weekday() >= 5:
        return False
    return d not in us_market_holidays(d.year)


def is_regular_hours(now: _dt.datetime) -> bool:
    """True when ``now`` (a US/Eastern datetime) is inside the regular trading
    session: a trading day, at/after 09:30 and strictly before the close (16:00,
    or 13:00 on an early-close day). The bridge supplies ``now`` in ET; this
    function never reads the clock so it is deterministic in tests."""
    d = now.date()
    if not is_trading_day(d):
        return False
    t = now.time()
    return REGULAR_OPEN <= t < session_close(d)


def regular_hours_reason(now: _dt.datetime) -> str:
    """Human-readable reason for is_regular_hours, for logs/audit:
    'open' / 'weekend' / 'holiday' / 'before_open' / 'after_close'."""
    d = now.date()
    if d.weekday() >= 5:
        return "weekend"
    if d in us_market_holidays(d.year):
        return "holiday"
    t = now.time()
    if t < REGULAR_OPEN:
        return "before_open"
    if t >= session_close(d):
        return "after_close"
    return "open"
