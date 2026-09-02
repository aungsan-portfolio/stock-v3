"""
strategies.shadow.pricing -- Realistic Pricing & Execution Friction Model.

Enforces pricing hierarchy:
1. Fresh Quote (age <= 5s, uncrossed, positive) -> Ask (Long) / Bid (Short).
2. Stale Quote (> 5s), Crossed, or Missing -> Reference Close +- Synthetic Slippage.
Prevents spread double-counting by strictly segregating half-spread from slippage.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import pandas as pd

import config


class PricingSource(str, Enum):
    LIVE_ASK = "LIVE_ASK"
    LIVE_BID = "LIVE_BID"
    SYNTHETIC_CLOSE_BPS = "SYNTHETIC_CLOSE_BPS"


@dataclass(frozen=True)
class QuoteSnapshot:
    bid: float
    ask: float
    timestamp_utc: pd.Timestamp
    age_seconds: float
    is_valid: bool
    rejection_reason: str = ""


def validate_quote(
    quote: Optional[dict],
    now_utc: pd.Timestamp,
    max_age_seconds: float = 5.0,
    max_spread_pct: float = 0.015,  # 1.5% max spread
) -> QuoteSnapshot:
    """Validates real-time bid/ask quote geometry and freshness."""
    if not quote:
        return QuoteSnapshot(0.0, 0.0, now_utc, 999.0, False, "QUOTE_NONE")

    try:
        bid = float(quote.get("bid", 0.0) or quote.get("bid_price", 0.0) or 0.0)
        ask = float(quote.get("ask", 0.0) or quote.get("ask_price", 0.0) or 0.0)
    except (ValueError, TypeError):
        return QuoteSnapshot(0.0, 0.0, now_utc, 999.0, False, "NON_NUMERIC_QUOTE")

    if bid <= 0 or ask <= 0:
        return QuoteSnapshot(bid, ask, now_utc, 999.0, False, "NON_POSITIVE_PRICE")

    if bid > ask:
        return QuoteSnapshot(bid, ask, now_utc, 0.0, False, "CROSSED_QUOTE_BID_GT_ASK")

    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid if mid > 0 else 1.0
    if spread_pct > max_spread_pct:
        return QuoteSnapshot(bid, ask, now_utc, 0.0, False, f"EXCESSIVE_SPREAD_{spread_pct:.4f}")

    raw_ts = quote.get("timestamp") or quote.get("quote_timestamp")
    if raw_ts is not None:
        try:
            quote_ts = pd.Timestamp(raw_ts)
            if quote_ts.tzinfo is None:
                quote_ts = quote_ts.tz_localize("UTC")
            else:
                quote_ts = quote_ts.tz_convert("UTC")
            age = max(0.0, float((now_utc - quote_ts).total_seconds()))
        except Exception:
            quote_ts = now_utc
            age = 0.0
    else:
        quote_ts = now_utc
        age = 0.0

    if age > max_age_seconds:
        return QuoteSnapshot(bid, ask, quote_ts, age, False, f"STALE_QUOTE_{age:.1f}s")

    return QuoteSnapshot(bid, ask, quote_ts, age, True, "OK")


def resolve_entry_price(
    side: str,
    reference_close: float,
    quote: Optional[dict],
    now_utc: pd.Timestamp,
    tick_size: float = 0.01,
    slippage_bps: float = 5.0,
    max_quote_age_seconds: float = 5.0,
) -> Tuple[float, str, float, float, float, Optional[str], Optional[float]]:
    """Resolves hypothetical entry price following the strict pricing hierarchy.

    Returns:
        (entry_price, pricing_source, synthetic_cost, half_spread_cost, slippage_cost, quote_ts_iso, quote_age_seconds)
    """
    snapshot = validate_quote(quote, now_utc, max_age_seconds=max_quote_age_seconds)
    side_u = str(side or "BUY").upper().strip()

    if snapshot.is_valid:
        if side_u == "BUY":
            # Live ask contains spread friction naturally; do not add half-spread twice
            return (
                snapshot.ask,
                PricingSource.LIVE_ASK.value,
                0.0,
                0.0,
                0.0,
                snapshot.timestamp_utc.isoformat(),
                snapshot.age_seconds,
            )
        else:
            return (
                snapshot.bid,
                PricingSource.LIVE_BID.value,
                0.0,
                0.0,
                0.0,
                snapshot.timestamp_utc.isoformat(),
                snapshot.age_seconds,
            )

    # Synthetic fallback from reference close
    synthetic_cost = max(tick_size, reference_close * (slippage_bps / 10_000.0))
    if side_u == "BUY":
        entry_price = round(reference_close + synthetic_cost, 4)
    else:
        entry_price = round(reference_close - synthetic_cost, 4)

    return (
        entry_price,
        PricingSource.SYNTHETIC_CLOSE_BPS.value,
        round(synthetic_cost, 4),
        0.0,
        round(synthetic_cost, 4),
        snapshot.timestamp_utc.isoformat() if snapshot.timestamp_utc else None,
        snapshot.age_seconds if snapshot.age_seconds < 900 else None,
    )


def resolve_exit_price(
    side: str,
    reference_close: float,
    quote: Optional[dict],
    now_utc: pd.Timestamp,
    tick_size: float = 0.01,
    slippage_bps: float = 5.0,
    max_quote_age_seconds: float = 5.0,
) -> Tuple[float, str, float]:
    """Resolves hypothetical exit price following the strict pricing hierarchy.

    Returns:
        (exit_price, pricing_source, exit_synthetic_cost)
    """
    snapshot = validate_quote(quote, now_utc, max_age_seconds=max_quote_age_seconds)
    side_u = str(side or "BUY").upper().strip()

    if snapshot.is_valid:
        if side_u == "BUY":
            # Exiting a long position sells at bid
            return (snapshot.bid, PricingSource.LIVE_BID.value, 0.0)
        else:
            # Exiting a short position buys at ask
            return (snapshot.ask, PricingSource.LIVE_ASK.value, 0.0)

    # Synthetic fallback from reference close
    synthetic_cost = max(tick_size, reference_close * (slippage_bps / 10_000.0))
    if side_u == "BUY":
        exit_price = round(reference_close - synthetic_cost, 4)
    else:
        exit_price = round(reference_close + synthetic_cost, 4)

    return (exit_price, PricingSource.SYNTHETIC_CLOSE_BPS.value, round(synthetic_cost, 4))
