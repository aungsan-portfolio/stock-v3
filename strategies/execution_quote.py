"""
execution_quote.py -- M-01 Execution Quote Verification Module

Validates fresh broker quote, data provenance, spread, staleness, and decision-to-execution price deviation before order entry.
"""
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def _parse_bool(value: Any) -> Optional[bool]:
    """Strictly parse boolean from bool, int (0, 1), or case-insensitive string."""
    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False

    return None


@dataclass
class BrokerQuote:
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp: float
    source: str = "broker"
    is_realtime: bool = True

    def executable_price(self, side: str = "BUY") -> float:
        """Return ask for BUY, bid for SELL/SHORT."""
        s = side.upper().strip()
        if s in ("BUY", "LONG"):
            return self.ask if self.ask > 0 else self.last
        elif s in ("SELL", "SHORT"):
            return self.bid if self.bid > 0 else self.last
        return self.last if self.last > 0 else (self.ask or self.bid)


@dataclass
class QuoteValidationResult:
    valid: bool
    reason: str
    quote: Optional[BrokerQuote] = None

    @property
    def accepted(self) -> bool:
        return self.valid


ExecutionQuoteResult = QuoteValidationResult


def validate_execution_quote(
    bridge: Any,
    symbol: str,
    side: str = "BUY",
    decision_price: float = 0.0,
    max_spread_pct: float = 1.0,
    max_deviation_pct: float = 1.5,
    max_age_seconds: float = 10.0,
) -> QuoteValidationResult:
    """
    Validate that a fresh, real-time broker quote is available and within acceptable risk parameters.
    Fail-closed: Returns valid=False if quote is missing, stale, inverted, wide-spread, or deviates from decision price.
    """
    try:
        max_spread_pct = float(max_spread_pct)
        max_deviation_pct = float(max_deviation_pct)
        max_age_seconds = float(max_age_seconds)
        decision_price = float(decision_price)
    except (TypeError, ValueError, OverflowError) as exc:
        return QuoteValidationResult(
            valid=False,
            reason=f"Invalid quote-validation configuration: {exc}",
        )

    if (
        not math.isfinite(max_spread_pct)
        or not math.isfinite(max_deviation_pct)
        or not math.isfinite(max_age_seconds)
        or not math.isfinite(decision_price)
        or max_spread_pct < 0
        or max_deviation_pct < 0
        or max_age_seconds <= 0
        or decision_price < 0
    ):
        return QuoteValidationResult(
            valid=False,
            reason="Invalid quote-validation configuration",
        )

    get_quote_fn = getattr(bridge, "get_quote", None)
    if not callable(get_quote_fn):
        return QuoteValidationResult(
            valid=False,
            reason=f"Broker bridge does not implement get_quote() contract for {symbol}",
        )

    try:
        raw_quote = get_quote_fn(symbol)
    except Exception as exc:
        return QuoteValidationResult(
            valid=False,
            reason=f"Failed to fetch quote for {symbol}: {exc}",
        )

    if not raw_quote:
        return QuoteValidationResult(
            valid=False,
            reason=f"Broker returned empty/null quote for {symbol}",
        )

    # Standardize mapping or dataclass with fail-closed conversion
    try:
        if isinstance(raw_quote, dict):
            bid = float(raw_quote.get("bid", 0.0) or 0.0)
            ask = float(raw_quote.get("ask", 0.0) or 0.0)
            last = float(raw_quote.get("last", 0.0) or 0.0)
            ts = float(raw_quote.get("timestamp", 0.0) or 0.0)
            source = str(raw_quote.get("source", "broker"))
            is_realtime_raw = raw_quote.get("is_realtime", True)
        else:
            bid = float(getattr(raw_quote, "bid", 0.0) or 0.0)
            ask = float(getattr(raw_quote, "ask", 0.0) or 0.0)
            last = float(getattr(raw_quote, "last", 0.0) or 0.0)
            ts = float(getattr(raw_quote, "timestamp", 0.0) or 0.0)
            source = str(getattr(raw_quote, "source", "broker"))
            is_realtime_raw = getattr(raw_quote, "is_realtime", True)
    except (TypeError, ValueError, OverflowError) as exc:
        return QuoteValidationResult(
            valid=False,
            reason=f"Malformed broker quote for {symbol}: {exc}",
        )

    is_realtime = _parse_bool(is_realtime_raw)
    if is_realtime is None:
        return QuoteValidationResult(
            valid=False,
            reason=f"Invalid is_realtime flag for {symbol}: {is_realtime_raw!r}",
        )

    numeric_fields = {
        "bid": bid,
        "ask": ask,
        "last": last,
        "timestamp": ts,
    }
    invalid_fields = [
        name
        for name, value in numeric_fields.items()
        if not math.isfinite(value)
    ]
    if invalid_fields:
        return QuoteValidationResult(
            valid=False,
            reason=(
                f"Non-finite broker quote fields for {symbol}: "
                f"{', '.join(invalid_fields)}"
            ),
        )

    quote = BrokerQuote(
        symbol=symbol.upper().strip(),
        bid=bid,
        ask=ask,
        last=last,
        timestamp=ts,
        source=source,
        is_realtime=is_realtime,
    )

    if not is_realtime:
        return QuoteValidationResult(
            valid=False,
            reason=f"Quote for {symbol} is not real-time (source={source})",
            quote=quote,
        )

    now = time.time()
    age_seconds = now - ts
    future_tolerance_seconds = 2.0

    if ts <= 0:
        return QuoteValidationResult(
            valid=False,
            reason=f"Quote timestamp missing or invalid for {symbol}",
            quote=quote,
        )

    if age_seconds < -future_tolerance_seconds:
        return QuoteValidationResult(
            valid=False,
            reason=(
                f"Quote timestamp for {symbol} is "
                f"{abs(age_seconds):.1f}s in the future"
            ),
            quote=quote,
        )

    if age_seconds > max_age_seconds:
        return QuoteValidationResult(
            valid=False,
            reason=(
                f"Quote for {symbol} is stale "
                f"(age={age_seconds:.1f}s > "
                f"limit={max_age_seconds:.1f}s)"
            ),
            quote=quote,
        )

    if bid <= 0 or ask <= 0:
        return QuoteValidationResult(
            valid=False,
            reason=f"Invalid bid/ask prices for {symbol}: bid=${bid:.2f}, ask=${ask:.2f}",
            quote=quote,
        )

    if bid > ask:
        return QuoteValidationResult(
            valid=False,
            reason=f"Inverted order book for {symbol}: bid=${bid:.2f} > ask=${ask:.2f}",
            quote=quote,
        )

    mid = (bid + ask) / 2.0
    spread_dollars = ask - bid
    spread_pct = (spread_dollars / mid) * 100.0
    if spread_pct > max_spread_pct:
        age_str = f"{age_seconds:.3f}s"
        return QuoteValidationResult(
            valid=False,
            reason=(
                f"Spread too wide for {symbol}: {spread_pct:.2f}% > limit {max_spread_pct:.2f}% "
                f"[bid=${bid:.2f}, ask=${ask:.2f}, mid=${mid:.2f}, spread_dollars=${spread_dollars:.4f}, age={age_str}]"
            ),
            quote=quote,
        )

    if decision_price > 0:
        dev_pct = (abs(mid - decision_price) / decision_price) * 100.0
        if dev_pct > max_deviation_pct:
            return QuoteValidationResult(
                valid=False,
                reason=f"Price deviation for {symbol}: quote mid ${mid:.2f} vs decision ${decision_price:.2f} ({dev_pct:.2f}% > limit {max_deviation_pct:.2f}%)",
                quote=quote,
            )

    return QuoteValidationResult(valid=True, reason="Quote valid", quote=quote)
