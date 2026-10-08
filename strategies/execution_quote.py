"""
execution_quote.py -- M-01 Execution Quote Verification Module

Validates fresh broker quote, data provenance, spread, staleness, and decision-to-execution price deviation before order entry.
"""
import logging
import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

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


def _parse_timestamp_to_epoch(ts: Any) -> Optional[float]:
    """Strict fail-closed timestamp parsing to UTC epoch seconds."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        val = float(ts)
        return val if math.isfinite(val) and val > 0 else None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.timestamp()
    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return None
        try:
            val = float(s)
            return val if math.isfinite(val) and val > 0 else None
        except ValueError:
            pass
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            pass
        try:
            import pandas as pd
            pdt = pd.to_datetime(s, utc=True)
            return pdt.timestamp()
        except Exception:
            pass
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
    feed_identity: Optional[str] = None

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
    decision_feed: Optional[str] = None,
    bar_timestamp: Optional[Union[str, float, datetime, Any]] = None,
    max_bar_age_seconds: float = 300.0,
    require_feed_concordance: bool = False,
    quote: Optional[Any] = None,
    bar_duration_seconds: float = 0.0,
    max_entry_bar_close_age_seconds: Optional[float] = None,
) -> QuoteValidationResult:
    """
    Validate that a fresh, real-time broker quote is available and within acceptable risk parameters.
    Fail-closed: Returns valid=False if quote is missing, stale, inverted, wide-spread, or deviates from decision price.
    Also validates feed identity concordance, bar freshness, and cross-feed dislocation.
    """
    try:
        max_spread_pct = float(max_spread_pct)
        max_deviation_pct = float(max_deviation_pct)
        max_age_seconds = float(max_age_seconds)
        max_bar_age_seconds = float(max_bar_age_seconds)
        bar_duration_seconds = float(bar_duration_seconds)
        if max_entry_bar_close_age_seconds is not None:
            max_entry_bar_close_age_seconds = float(max_entry_bar_close_age_seconds)
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
        or not math.isfinite(max_bar_age_seconds)
        or not math.isfinite(bar_duration_seconds)
        or (max_entry_bar_close_age_seconds is not None and not math.isfinite(max_entry_bar_close_age_seconds))
        or not math.isfinite(decision_price)
        or max_spread_pct < 0
        or max_deviation_pct < 0
        or max_age_seconds <= 0
        or max_bar_age_seconds <= 0
        or bar_duration_seconds < 0
        or (max_entry_bar_close_age_seconds is not None and max_entry_bar_close_age_seconds <= 0)
        or decision_price < 0
    ):
        return QuoteValidationResult(
            valid=False,
            reason="Invalid quote-validation configuration",
        )

    if quote is not None:
        raw_quote = quote
    else:
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
            feed_id_raw = raw_quote.get("feed_identity")
        else:
            bid = float(getattr(raw_quote, "bid", 0.0) or 0.0)
            ask = float(getattr(raw_quote, "ask", 0.0) or 0.0)
            last = float(getattr(raw_quote, "last", 0.0) or 0.0)
            ts = float(getattr(raw_quote, "timestamp", 0.0) or 0.0)
            source = str(getattr(raw_quote, "source", "broker"))
            is_realtime_raw = getattr(raw_quote, "is_realtime", True)
            feed_id_raw = getattr(raw_quote, "feed_identity", None)
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

    feed_identity = str(feed_id_raw).lower().strip() if feed_id_raw is not None and str(feed_id_raw).strip() else None

    quote = BrokerQuote(
        symbol=symbol.upper().strip(),
        bid=bid,
        ask=ask,
        last=last,
        timestamp=ts,
        source=source,
        is_realtime=is_realtime,
        feed_identity=feed_identity,
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
            reason=f"QUOTE_TIMESTAMP_MISSING: Quote timestamp missing or invalid for {symbol}",
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
                f"QUOTE_STALE: Quote for {symbol} is stale "
                f"(age={age_seconds:.1f}s > "
                f"limit={max_age_seconds:.1f}s)"
            ),
            quote=quote,
        )

    # Validate decision bar freshness and ordering if bar_timestamp is provided
    bar_epoch = None
    if bar_timestamp is not None:
        bar_epoch = _parse_timestamp_to_epoch(bar_timestamp)
        if bar_epoch is None:
            return QuoteValidationResult(
                valid=False,
                reason=f"BAR_TIMESTAMP_MALFORMED: Could not parse bar timestamp for {symbol}: {bar_timestamp!r}",
                quote=quote,
            )
        bar_age = now - bar_epoch
        if bar_age < -future_tolerance_seconds:
            return QuoteValidationResult(
                valid=False,
                reason=f"BAR_TIMESTAMP_IN_FUTURE: Decision bar timestamp for {symbol} is in the future",
                quote=quote,
            )
        if bar_duration_seconds is not None:
            if not (math.isfinite(bar_duration_seconds) and bar_duration_seconds >= 0.0):
                return QuoteValidationResult(
                    valid=False,
                    reason=(
                        f"Invalid quote-validation configuration for {symbol}: "
                        f"bar_duration_seconds={bar_duration_seconds}"
                    ),
                    quote=quote,
                )
        if bar_duration_seconds > 0.0:
            bar_close_epoch = bar_epoch + bar_duration_seconds
            bar_close_age = now - bar_close_epoch
            effective_limit = (
                max_entry_bar_close_age_seconds
                if max_entry_bar_close_age_seconds is not None
                else max_bar_age_seconds
            )
            if not (math.isfinite(bar_close_age) and math.isfinite(effective_limit) and effective_limit >= 0.0):
                return QuoteValidationResult(
                    valid=False,
                    reason=(
                        f"Invalid quote-validation configuration for {symbol}: "
                        f"bar_close_age={bar_close_age}, effective_limit={effective_limit}"
                    ),
                    quote=quote,
                )
            if bar_close_age < 0.0:
                return QuoteValidationResult(
                    valid=False,
                    reason=(
                        f"BAR_FORMING: Decision bar for {symbol} has not closed yet "
                        f"(forming; closes in {abs(bar_close_age):.3f}s)"
                    ),
                    quote=quote,
                )
            if bar_close_age > effective_limit:
                return QuoteValidationResult(
                    valid=False,
                    reason=(
                        f"BAR_STALE: Decision bar for {symbol} is stale "
                        f"(close_age={bar_close_age:.1f}s > limit={effective_limit:.1f}s)"
                    ),
                    quote=quote,
                )
        else:
            if bar_age > max_bar_age_seconds:
                return QuoteValidationResult(
                    valid=False,
                    reason=(
                        f"BAR_STALE: Decision bar for {symbol} is stale "
                        f"(age={bar_age:.1f}s > limit={max_bar_age_seconds:.1f}s)"
                    ),
                    quote=quote,
                )
        if ts < bar_epoch - future_tolerance_seconds:
            return QuoteValidationResult(
                valid=False,
                reason=(
                    f"NEGATIVE_LATENCY_ERROR: Quote timestamp ({ts:.1f}) older than "
                    f"decision bar ({bar_epoch:.1f}) by > {future_tolerance_seconds:.1f}s"
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

    # Feed provenance normalization
    q_feed = (feed_identity or "").lower().strip()
    d_feed = (str(decision_feed).lower().strip()) if decision_feed is not None and str(decision_feed).strip() else ""

    # Required feed concordance enforces non-empty feed identities
    if require_feed_concordance:
        if not d_feed or not q_feed:
            return QuoteValidationResult(
                valid=False,
                reason=(
                    f"FEED_IDENTITY_MISSING: Missing feed identity for {symbol} "
                    f"(decision_feed={decision_feed!r}, quote_feed={feed_identity!r})"
                ),
                quote=quote,
            )

    # Decision price divergence guard
    if decision_price > 0:
        dev_pct = (abs(mid - decision_price) / decision_price) * 100.0
        if dev_pct > max_deviation_pct:
            if d_feed and q_feed and d_feed != q_feed:
                return QuoteValidationResult(
                    valid=False,
                    reason=(
                        f"PROVENANCE_FEED_DISCREPANCY_VETO: Price deviation for {symbol}: "
                        f"quote mid ${mid:.2f} ({q_feed}) vs decision ${decision_price:.2f} ({d_feed}) "
                        f"({dev_pct:.2f}% > limit {max_deviation_pct:.2f}%)"
                    ),
                    quote=quote,
                )
            else:
                feed_label = f" ({d_feed})" if d_feed else ""
                return QuoteValidationResult(
                    valid=False,
                    reason=(
                        f"PRICE_DRIFT_DEVIATION_VETO: Price deviation for {symbol}: "
                        f"quote mid ${mid:.2f}{feed_label} vs decision ${decision_price:.2f} "
                        f"({dev_pct:.2f}% > limit {max_deviation_pct:.2f}%)"
                    ),
                    quote=quote,
                )

    # Feed concordance check
    if require_feed_concordance:
        if d_feed != q_feed:
            return QuoteValidationResult(
                valid=False,
                reason=(
                    f"CROSS_FEED_PROVENANCE_DISLOCATION: Feed discordance for {symbol}: "
                    f"decision_feed={d_feed} vs quote_feed={q_feed}"
                ),
                quote=quote,
            )
    elif d_feed and q_feed and d_feed != q_feed:
        return QuoteValidationResult(
            valid=False,
            reason=(
                f"CROSS_FEED_PROVENANCE_DISLOCATION: Feed discordance for {symbol}: "
                f"decision_feed={d_feed} vs quote_feed={q_feed}"
            ),
            quote=quote,
        )

    return QuoteValidationResult(valid=True, reason="Quote valid", quote=quote)
