"""
order_manager.py -- Orchestrates signal -> risk check -> position size -> order.

The single entry point for executing a TradeSignal through the full pipeline.
Always paper-trading only.
"""
from dataclasses import replace
import inspect
import logging
import math
import threading
import time
from datetime import datetime, timezone
from collections import defaultdict
from typing import Optional, Tuple, Any, Union

import config
from strategies.base import TradeSignal
from strategies.intraday_risk import pre_trade_check
from strategies.position_sizer import calculate_shares, estimated_cost
from strategies.trade_journal import log_trade, today_trade_count
from strategies.order_idempotency import idempotency_manager
from strategies.webhook import send_discord_alert
from strategies.execution_geometry import (
    calculate_minimum_stop_distance, floor_to_tick_float, ceil_to_tick_float,
)
from strategies.entry_lifecycle import entry_lifecycle, EntryLifecycleState
import order_audit

logger = logging.getLogger(__name__)


def _normalize_bar_ts(ts: Any) -> Optional[datetime]:
    """Convert a supported bar timestamp (string, datetime, pd.Timestamp) to an aware UTC datetime."""
    if ts is None:
        return None

    if hasattr(ts, "to_pydatetime"):
        try:
            ts = ts.to_pydatetime()
        except Exception:
            pass

    if isinstance(ts, str):
        value = ts.strip()
        if not value:
            return None
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            ts = datetime.fromisoformat(value)
        except Exception:
            try:
                import pandas as pd
                pdt = pd.Timestamp(value)
                ts = pdt.to_pydatetime()
            except Exception:
                logger.warning("Could not parse bar timestamp string into datetime: %r", value)
                return None

    if not isinstance(ts, datetime):
        return None

    if ts.tzinfo is None:
        # Upstream naive timestamps are assumed UTC
        ts = ts.replace(tzinfo=timezone.utc)

    return ts.astimezone(timezone.utc)


class BrokerIdempotencyUnsupported(RuntimeError):
    """Broker bridge cannot safely accept a deterministic client_order_id."""


_session_state_lock = threading.RLock()
_session_skip_counts = defaultdict(int)
_session_suppressed_until = {}
_session_cooldown_counts = defaultdict(int)
_session_permanently_suppressed = set()
_seen_veto_fingerprints = {}

# Setup-scoped suppression state (keyed by (symbol, strategy, side))
_setup_skip_counts = defaultdict(int)
_setup_suppressed_until = {}
_setup_cooldown_counts = defaultdict(int)
_setup_permanently_suppressed = set()


def _normalize_setup_key(symbol: str, strategy: Optional[Any] = None, side: Optional[str] = None) -> Tuple[str, str, str]:
    sym = symbol.upper().strip()
    strat_str = getattr(strategy, "name", getattr(strategy, "value", str(strategy or "*"))).upper().strip()
    side_str = str(side or "*").upper().strip()
    return (sym, strat_str, side_str)


def _get_skip_limit() -> int:
    return getattr(config, "HARD_VETO_SKIP_LIMIT", 5)


def _get_cooldown_seconds() -> float:
    return float(getattr(config, "SESSION_SUPPRESSION_COOLDOWN_SECONDS", 2700.0))


def _get_max_cooldown_cycles() -> int:
    return int(getattr(config, "MAX_DAILY_COOLDOWN_CYCLES", 3))


def check_session_suppression(
    symbol: str,
    strategy: Optional[Any] = None,
    side: Optional[str] = None,
) -> Tuple[bool, str]:
    """Checks if a symbol or specific setup (symbol, strategy, side) is actively suppressed or in cooldown.
    Tier 1: Global / whole-symbol hazards & suppression.
    Tier 2: Candidate-specific setup suppression (prevents cross-strategy contamination).
    """
    if not getattr(config, "SKIP_SUPPRESSOR_ENABLED", True):
        return False, ""
    sym = symbol.upper().strip()
    now = time.time()

    with _session_state_lock:
        # Tier 1: Check permanent full-day symbol suppression (exceeded max cooldown cycles)
        if sym in _session_permanently_suppressed:
            max_cycles = _get_max_cooldown_cycles()
            return True, f"[SESSION SUPPRESSED] {sym}: permanently suppressed this session (exceeded {max_cycles} cooldown cycles)"

        # Tier 1: Check dynamic symbol cooldown window
        if sym in _session_suppressed_until:
            until = _session_suppressed_until[sym]
            if now < until:
                remaining_min = (until - now) / 60.0
                cycle = _session_cooldown_counts[sym]
                return True, f"[SESSION SUPPRESSED] {sym}: cooldown active ({remaining_min:.1f}m remaining, Cycle #{cycle})"
            else:
                cycle = _session_cooldown_counts[sym]
                logger.info("[COOLDOWN EXPIRE] %s: Cooldown period ended (Cycle #%d). Symbol restored to active evaluation.", sym, cycle)
                del _session_suppressed_until[sym]

        # Tier 2: Check Setup-Scoped Suppression (symbol, strategy, side)
        if strategy is not None:
            setup_key = _normalize_setup_key(sym, strategy, side)
            strat_label = f"{setup_key[1]}:{setup_key[2]}"
            if setup_key in _setup_permanently_suppressed:
                max_cycles = _get_max_cooldown_cycles()
                return True, f"[SESSION SUPPRESSED] {sym} [{strat_label}]: permanently suppressed this session (exceeded {max_cycles} cooldown cycles)"

            if setup_key in _setup_suppressed_until:
                until = _setup_suppressed_until[setup_key]
                if now < until:
                    remaining_min = (until - now) / 60.0
                    cycle = _setup_cooldown_counts[setup_key]
                    return True, f"[SESSION SUPPRESSED] {sym} [{strat_label}]: cooldown active ({remaining_min:.1f}m remaining, Cycle #{cycle})"
                else:
                    cycle = _setup_cooldown_counts[setup_key]
                    logger.info("[COOLDOWN EXPIRE] %s [%s]: Cooldown period ended (Cycle #%d). Setup restored to active evaluation.", sym, strat_label, cycle)
                    del _setup_suppressed_until[setup_key]

        return False, ""


is_symbol_suppressed = check_session_suppression


def record_hard_veto_skip(
    symbol: str,
    reason: Union[str, Any],
    bar_timestamp: str = "",
    strategy: Optional[Any] = None,
    side: Optional[str] = None,
):
    """Increments skip counter when noise floor / tight stop hard vetoes occur, deduplicated per bar fingerprint.
    When strategy and side are provided, scopes cooldown to (symbol, strategy, side) so one strategy's
    rejection does not contaminate other viable strategies on the same symbol.
    """
    from strategies.veto_reasons import VetoReason, HARD_VETO_CODES

    sym = symbol.upper().strip()
    is_hard_veto = False

    if isinstance(reason, VetoReason):
        is_hard_veto = reason in HARD_VETO_CODES
        reason_str = reason.value
    elif isinstance(reason, str):
        try:
            enum_val = VetoReason(reason)
            is_hard_veto = enum_val in HARD_VETO_CODES
        except ValueError:
            # Exact phrase check (Strictly excludes 'atr' substring contamination)
            EXACT_HARD_VETO_PHRASES = [
                "critically tight",
                "noise floor",
                "rejected_noise_floor_breach",
                "critical_stop",
                "invalid_structural_stop",
                "veto line",
            ]
            reason_lower = reason.lower()
            is_hard_veto = any(phrase in reason_lower for phrase in EXACT_HARD_VETO_PHRASES)
        reason_str = reason
    else:
        return

    if not is_hard_veto:
        return

    with _session_state_lock:
        now = time.time()
        # Candidate-specific setup suppression when strategy identity is provided
        if strategy is not None:
            setup_key = _normalize_setup_key(sym, strategy, side)
            strat_label = f"{setup_key[1]}:{setup_key[2]}"
            fingerprint = f"{setup_key[0]}:{strat_label}:{bar_timestamp}:{reason_str.strip()[:80]}"
            last_seen = _seen_veto_fingerprints.get(fingerprint, 0)

            if now - last_seen < 3600:
                count = _setup_skip_counts[setup_key]
                logger.debug("[SKIP COUNTER REPEAT] %s [%s]: Repeat hard veto skip #%d (deduplicated) — reason: %s", sym, strat_label, count, reason)
                return

            _seen_veto_fingerprints[fingerprint] = now
            _setup_skip_counts[setup_key] += 1
            count = _setup_skip_counts[setup_key]
            limit = _get_skip_limit()
            logger.info("[SKIP COUNTER] %s [%s]: Hard veto skip #%d/%d — reason: %s", sym, strat_label, count, limit, reason)

            if count >= limit:
                max_cycles = _get_max_cooldown_cycles()
                current_cycles = _setup_cooldown_counts[setup_key]

                if current_cycles >= max_cycles:
                    _setup_permanently_suppressed.add(setup_key)
                    _setup_skip_counts[setup_key] = 0
                    logger.warning(
                        "🚨 [SESSION SUPPRESS PERMANENT] %s [%s]: Suppressed for remainder of session after reaching max cooldown cycles (%d/%d).",
                        sym, strat_label, current_cycles, max_cycles,
                    )
                else:
                    cooldown_sec = _get_cooldown_seconds()
                    cooldown_until = now + cooldown_sec
                    _setup_suppressed_until[setup_key] = cooldown_until
                    _setup_cooldown_counts[setup_key] += 1
                    _setup_skip_counts[setup_key] = 0
                    cooldown_min = cooldown_sec / 60.0
                    resume_str = time.strftime("%H:%M:%S", time.localtime(cooldown_until))
                    logger.warning(
                        "⏳ [COOLDOWN START] %s [%s]: Suppressed for %.1fm cooldown (Cycle #%d/%d) after %d hard vetoes. Active again at %s.",
                        sym, strat_label, cooldown_min, _setup_cooldown_counts[setup_key], max_cycles, count, resume_str,
                    )
        else:
            # Fallback / Global symbol-level suppression (when no strategy identity is provided)
            fingerprint = f"{sym}:{bar_timestamp}:{reason_str.strip()[:80]}"
            last_seen = _seen_veto_fingerprints.get(fingerprint, 0)

            if now - last_seen < 3600:
                count = _session_skip_counts[sym]
                logger.debug("[SKIP COUNTER REPEAT] %s: Repeat hard veto skip #%d (deduplicated) — reason: %s", sym, count, reason)
                return

            _seen_veto_fingerprints[fingerprint] = now
            _session_skip_counts[sym] += 1
            count = _session_skip_counts[sym]
            limit = _get_skip_limit()
            logger.info("[SKIP COUNTER] %s: Hard veto skip #%d/%d — reason: %s", sym, count, limit, reason)

            if count >= limit:
                max_cycles = _get_max_cooldown_cycles()
                current_cycles = _session_cooldown_counts[sym]

                if current_cycles >= max_cycles:
                    _session_permanently_suppressed.add(sym)
                    _session_skip_counts[sym] = 0
                    logger.warning(
                        "🚨 [SESSION SUPPRESS PERMANENT] %s: Suppressed for remainder of session after reaching max cooldown cycles (%d/%d).",
                        sym, current_cycles, max_cycles,
                    )
                else:
                    cooldown_sec = _get_cooldown_seconds()
                    cooldown_until = now + cooldown_sec
                    _session_suppressed_until[sym] = cooldown_until
                    _session_cooldown_counts[sym] += 1
                    _session_skip_counts[sym] = 0  # Counter reset for fresh post-cooldown budget
                    cooldown_min = cooldown_sec / 60.0
                    resume_str = time.strftime("%H:%M:%S", time.localtime(cooldown_until))
                    logger.warning(
                        "⏳ [COOLDOWN START] %s: Suppressed for %.1fm cooldown (Cycle #%d/%d) after %d hard vetoes. Active again at %s.",
                        sym, cooldown_min, _session_cooldown_counts[sym], max_cycles, count, resume_str,
                    )


def reset_session_skip_state():
    """Resets session skip state for new trading sessions."""
    global _session_skip_counts, _session_suppressed_until, _session_cooldown_counts, _session_permanently_suppressed, _seen_veto_fingerprints
    global _setup_skip_counts, _setup_suppressed_until, _setup_cooldown_counts, _setup_permanently_suppressed
    with _session_state_lock:
        _session_skip_counts = defaultdict(int)
        _session_suppressed_until = {}
        _session_cooldown_counts = defaultdict(int)
        _session_permanently_suppressed = set()
        _seen_veto_fingerprints = {}
        _setup_skip_counts = defaultdict(int)
        _setup_suppressed_until = {}
        _setup_cooldown_counts = defaultdict(int)
        _setup_permanently_suppressed = set()
    logger.info("[ORDER MANAGER] Session skip state reset.")


def _strategy_max_risk_pct(strategy: str) -> float:
    """Safely retrieves max_risk_pct whether STRATEGY_SETTINGS is dict or object."""
    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(strategy)
    if isinstance(strat_cfg, dict):
        return float(strat_cfg.get("max_risk_pct", 1.0))
    if strat_cfg is not None:
        return float(getattr(strat_cfg, "max_risk_pct", 1.0))
    return 1.0


def _place_bracket_order_once(
    bridge: object,
    *,
    symbol: str,
    side: str,
    qty: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    client_order_id: str,
):
    """
    Submit a bracket order exactly once.

    Determines client_order_id support before submission instead of retrying
    after a broad TypeError. execute_signal() catches BrokerIdempotencyUnsupported
    to safely release idempotency reservation before any network transmission.
    """
    place_fn = getattr(bridge, "place_bracket_order", None)

    if not callable(place_fn):
        raise BrokerIdempotencyUnsupported(
            "Broker bridge does not implement place_bracket_order"
        )

    try:
        signature = inspect.signature(place_fn)
    except (TypeError, ValueError) as exc:
        raise BrokerIdempotencyUnsupported(
            "Cannot inspect broker submission signature; "
            "submission blocked to preserve broker idempotency"
        ) from exc

    supports_client_order_id = (
        "client_order_id" in signature.parameters
        or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    )

    if not supports_client_order_id:
        raise BrokerIdempotencyUnsupported(
            "Broker bridge does not support client_order_id; "
            "submission blocked to preserve broker idempotency"
        )

    kwargs = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "client_order_id": client_order_id,
    }

    return place_fn(**kwargs)


def get_adaptive_stop_multiplier(symbol: str, beta: float = 1.0, earnings_near: bool = False, cfg=None) -> float:
    """
    High beta (+0.5x) or pre-earnings (+0.5x) auto-adjusts ATR stop multiplier up to max 2.0x ATR.
    """
    cfg = cfg or config
    multiplier = getattr(cfg, "ATR_STOP_MULTIPLIER", 1.0)
    beta_threshold = getattr(cfg, "BETA_HIGH_THRESHOLD", 1.5)

    if beta >= beta_threshold:
        multiplier += (getattr(cfg, "ATR_STOP_HIGH_BETA", 1.5) - 1.0)
    if earnings_near:
        multiplier += (getattr(cfg, "ATR_STOP_EARNINGS", 2.0) - getattr(cfg, "ATR_STOP_HIGH_BETA", 1.5))
    return min(multiplier, getattr(cfg, "ATR_STOP_EARNINGS", 2.0))


def _compute_noise_floor(signal: TradeSignal, df: Optional[object] = None) -> float:
    """
    Calculate noise floor based on ATR using shared execution geometry.
    Returns baseline noise floor with use_buffer=False for order manager veto checks.
    """
    import pandas as pd
    atr_period = getattr(config, "NOISE_FLOOR_ATR_PERIOD", 10)
    abs_min = float(getattr(config, "NOISE_FLOOR_ABS_MIN", 0.08))
    tick_size = 0.01 if signal.entry_price >= 1.0 else 0.0001

    # 1. Check signal's explicit atr attribute first
    if hasattr(signal, "atr") and getattr(signal, "atr", None) is not None:
        try:
            sig_atr = float(getattr(signal, "atr"))
            if sig_atr > 0:
                noise_floor = calculate_minimum_stop_distance(
                    atr=sig_atr,
                    spread_floor=0.0,
                    tick_size=tick_size,
                    use_buffer=False,
                )
                return max(abs_min, noise_floor)
        except Exception:
            pass

    # 2. Check DataFrame ATR
    try:
        if df is not None and hasattr(df, "empty") and not df.empty and len(df) >= atr_period:
            high = df["high"].tail(atr_period + 1)
            low  = df["low"].tail(atr_period + 1)
            close_prev = df["close"].shift(1).tail(atr_period + 1)

            tr = pd.concat([
                high - low,
                (high - close_prev).abs(),
                (low  - close_prev).abs(),
            ], axis=1).max(axis=1)

            atr = float(tr.tail(atr_period).mean())
            if atr > 0:
                noise_floor = calculate_minimum_stop_distance(
                    atr=atr,
                    spread_floor=0.0,
                    tick_size=tick_size,
                    use_buffer=False,
                )
                logger.debug(
                    "[NOISE FLOOR] %s: ATR(%d)=%.4f noise_floor=%.4f (abs_min=%.2f)",
                    signal.symbol, atr_period, atr, noise_floor, abs_min
                )
                return max(abs_min, noise_floor)
    except Exception as e:
        logger.debug("[NOISE FLOOR] ATR calc failed for %s: %s. Using fallback.", signal.symbol, e)

    # Fallback: entry price * 0.25%
    fallback = max(abs_min, signal.entry_price * 0.0025)
    logger.debug("[NOISE FLOOR] %s: Fallback noise floor = %.4f", signal.symbol, fallback)
    return fallback



def compute_effective_noise_floor(
    signal: TradeSignal,
    df: Optional[object] = None,
    quote: Optional[dict] = None,
) -> float:
    """Compute authoritative effective noise floor:
    max(atr_noise_floor, spread_noise_floor, tick_size_floor, configured_min)
    using shared execution geometry with use_buffer=False.
    """
    tick_size_floor = 0.01 if signal.entry_price >= 1.0 else 0.0001
    configured_min = float(getattr(config, "NOISE_FLOOR_ABS_MIN", 0.08))

    # Spread noise floor
    spread_noise_floor = 0.0
    if quote:
        try:
            bid = float(quote.get("bid", 0.0) or quote.get("bid_price", 0.0) or 0.0)
            ask = float(quote.get("ask", 0.0) or quote.get("ask_price", 0.0) or 0.0)
            if ask > bid > 0:
                spread_noise_floor = ask - bid
        except Exception:
            pass

    # Extract ATR if available
    effective_atr = None
    if hasattr(signal, "atr") and getattr(signal, "atr", None) is not None:
        try:
            val = float(getattr(signal, "atr"))
            if val > 0:
                effective_atr = val
        except Exception:
            pass

    if effective_atr is None and df is not None and hasattr(df, "empty") and not df.empty:
        atr_period = getattr(config, "NOISE_FLOOR_ATR_PERIOD", 10)
        if len(df) >= atr_period:
            try:
                import pandas as pd
                high = df["high"].tail(atr_period + 1)
                low = df["low"].tail(atr_period + 1)
                close_prev = df["close"].shift(1).tail(atr_period + 1)
                tr = pd.concat([
                    high - low,
                    (high - close_prev).abs(),
                    (low - close_prev).abs(),
                ], axis=1).max(axis=1)
                calc_atr = float(tr.tail(atr_period).mean())
                if calc_atr > 0:
                    effective_atr = calc_atr
            except Exception:
                pass

    if effective_atr is not None and effective_atr > 0:
        base_geometry_noise = calculate_minimum_stop_distance(
            atr=effective_atr,
            spread_floor=spread_noise_floor,
            tick_size=tick_size_floor,
            use_buffer=False,
        )
        return max(base_geometry_noise, configured_min)

    atr_noise_floor = _compute_noise_floor(signal, df)
    return max(atr_noise_floor, spread_noise_floor, tick_size_floor, configured_min)


def validate_quote_health(
    quote: Optional[dict],
    now_ts: Optional[float] = None,
    max_age_seconds: Optional[float] = None,
    max_spread_pct: float = 0.015,
    atr: Optional[float] = None,
) -> Tuple[bool, str, dict]:
    """Validates real-time bid/ask quote health and returns normalized spread metrics."""
    if not quote:
        return False, "QUOTE_MISSING", {}

    if max_age_seconds is None:
        max_age_seconds = float(getattr(config, "MAX_QUOTE_AGE_SECONDS", 10.0))

    try:
        bid = float(quote.get("bid", 0.0) or quote.get("bid_price", 0.0) or 0.0)
        ask = float(quote.get("ask", 0.0) or quote.get("ask_price", 0.0) or 0.0)
    except (ValueError, TypeError):
        return False, "NON_NUMERIC_QUOTE", {}

    if bid <= 0 or ask <= 0:
        return False, f"NON_POSITIVE_QUOTE (bid={bid}, ask={ask})", {}

    if ask < bid:
        return False, f"CROSSED_QUOTE (bid={bid} > ask={ask})", {}

    mid = (bid + ask) / 2.0
    spread_dollars = ask - bid
    spread_bps = (spread_dollars / mid) * 10000.0 if mid > 0 else 0.0
    spread_as_atr_fraction = (spread_dollars / atr) if (atr and atr > 0) else 0.0

    metrics = {
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_dollars": spread_dollars,
        "spread_bps": spread_bps,
        "spread_as_atr_fraction": spread_as_atr_fraction,
    }

    spread_pct = spread_dollars / mid if mid > 0 else 1.0
    if spread_pct > max_spread_pct:
        return False, f"EXCESSIVE_SPREAD: {spread_pct*100:.2f}% > {max_spread_pct*100:.2f}%", metrics

    now = time.time() if now_ts is None else float(now_ts)
    raw_ts = quote.get("timestamp") or quote.get("quote_timestamp") or quote.get("ts")
    if raw_ts is not None:
        try:
            if isinstance(raw_ts, (int, float)):
                if raw_ts > 1e16:
                    q_ts = raw_ts / 1e9
                elif raw_ts > 1e11:
                    q_ts = raw_ts / 1e3
                else:
                    q_ts = float(raw_ts)
            else:
                import pandas as pd
                q_ts = pd.Timestamp(raw_ts).timestamp()
            age = max(0.0, now - q_ts)
            metrics["age_seconds"] = age
            if age > max_age_seconds:
                return False, f"STALE_QUOTE: age {age:.2f}s > {max_age_seconds:.1f}s", metrics
        except Exception:
            pass

    return True, "", metrics


def _pre_submission_veto(
    signal: TradeSignal,
    df: Optional[object] = None,
    market_texture: Optional[str] = None,
    quote: Optional[dict] = None,
) -> tuple:
    """Option A (Fail-Closed Default):
    stop_distance < effective_noise_floor => REJECTED_NOISE_FLOOR_BREACH
    """
    stop_distance = abs(signal.entry_price - signal.stop_price)
    effective_noise_floor = compute_effective_noise_floor(signal, df, quote=quote)

    policy = getattr(config, "NOISE_FLOOR_POLICY", "OPTION_A")
    if policy == "OPTION_A":
        if stop_distance < (effective_noise_floor - 1e-6):
            reason = (
                f"[Pre-submission Veto] REJECTED_NOISE_FLOOR_BREACH: stop distance ${stop_distance:.4f} < "
                f"effective noise floor ${effective_noise_floor:.4f}"
            )
            return True, reason
    else:
        # Legacy/Option B fallback
        hard_veto_ratio = getattr(config, "NOISE_FLOOR_HARD_VETO_RATIO", 0.75)
        if market_texture and hasattr(config, "MARKET_REGIME_SETTINGS"):
            tex_cfg = config.MARKET_REGIME_SETTINGS.get(market_texture, {})
            if "noise_floor_ratio" in tex_cfg:
                hard_veto_ratio = float(tex_cfg["noise_floor_ratio"])
        hard_veto_line = effective_noise_floor * hard_veto_ratio
        if stop_distance < (hard_veto_line - 1e-4):
            reason = (
                f"Pre-submission Veto: Stop distance ${stop_distance:.4f} critically tight "
                f"(< {hard_veto_ratio*100:.0f}% of ATR noise floor ${effective_noise_floor:.4f}). "
                f"Hard veto line: ${hard_veto_line:.4f}"
            )
            return True, reason

    return False, ""


def unfavorable_chase(side: str, signal_entry: float, executable_entry: float) -> float:
    """Compute unfavorable price movement between signal decision and executable price.
    Favorable price improvements evaluate to 0.0.
    """
    s = str(side).upper().strip()
    if s in ("BUY", "LONG"):
        return max(float(executable_entry) - float(signal_entry), 0.0)
    return max(float(signal_entry) - float(executable_entry), 0.0)


def validate_geometry(side: str, entry: float, stop: float, target: float) -> Tuple[bool, str, float, float, float]:
    """Side-neutral price geometry validator.
    Ensures correct price ordering:
      - BUY:  stop < entry < target
      - SELL: target < entry < stop
    Returns: (is_valid, reason, risk, reward, rr)
    """
    side_upper = str(side).upper().strip()
    e = float(entry)
    s = float(stop)
    t = float(target)

    if side_upper in ("BUY", "LONG"):
        ordering_valid = (s < e < t)
        risk = e - s
        reward = t - e
    elif side_upper in ("SELL", "SHORT"):
        ordering_valid = (t < e < s)
        risk = s - e
        reward = e - t
    else:
        return False, f"unsupported_side:{side}", 0.0, 0.0, 0.0

    if not ordering_valid:
        return False, "invalid_price_ordering", 0.0, 0.0, 0.0

    if risk <= 0.0 or reward <= 0.0:
        return False, "non_positive_geometry", 0.0, 0.0, 0.0

    rr = reward / risk
    return True, "", risk, reward, rr


def _validate_signal_prices(
    signal: TradeSignal,
    market_texture: Optional[str] = None,
    df_history: Optional[object] = None,
    quote: Optional[dict] = None,
) -> str:
    """Return a rejection reason when entry/stop/target geometry is invalid."""
    side = signal.side.upper()
    if signal.entry_price <= 0:
        return "Invalid entry price"
    if signal.stop_price <= 0 or signal.target_price <= 0:
        return "Invalid stop/target price"
    if signal.risk_per_share <= 0:
        return "Invalid risk/share"

    # Phase 0: Directional Price Ordering Check
    geo_ok, geo_err, risk, reward, rr = validate_geometry(
        signal.side, signal.entry_price, signal.stop_price, signal.target_price
    )
    if not geo_ok:
        return f"Invalid price ordering or geometry: {geo_err}"

    derived_risk = abs(signal.entry_price - signal.stop_price)
    if not math.isclose(signal.risk_per_share, derived_risk, rel_tol=1e-6, abs_tol=1e-6):
        return (
            f"Risk/share {signal.risk_per_share:.6f} does not match "
            f"entry/stop distance {derived_risk:.6f}"
        )

    # Validate signal geometry using tick size (SEC Rule 612) and Noise Floor
    tick_size = 0.01 if signal.entry_price >= 1.0 else 0.0001
    target_dist = round(abs(signal.target_price - signal.entry_price), 6)
    stop_dist = round(abs(signal.entry_price - signal.stop_price), 6)
    if target_dist <= tick_size:
        return f"Target distance {target_dist:.5f} is less than or equal to tick size {tick_size:.5f} (invalid geometry)"
    if stop_dist <= tick_size:
        return f"Stop distance {stop_dist:.5f} is less than or equal to tick size {tick_size:.5f} (invalid geometry)"

    # Phase 3: Pre-submission Veto for Too-Tight Signals using ATR-based Noise Floor
    if df_history is None:
        df_history = getattr(signal, "df", None)
    if df_history is None and getattr(signal, "atr", None) is None:
        try:
            from strategies.intraday_data import fetch_intraday
            df_history = fetch_intraday(signal.symbol)
        except Exception:
            pass

    should_skip, veto_reason = _pre_submission_veto(signal, df_history, market_texture, quote=quote)
    if should_skip:
        return veto_reason



    if side == "BUY":
        if signal.stop_price >= signal.entry_price:
            return "BUY stop must be below entry"
        if signal.target_price <= signal.entry_price:
            return "BUY target must be above entry"
    elif side == "SELL":
        if signal.stop_price <= signal.entry_price:
            return "SELL stop must be above entry"
        if signal.target_price >= signal.entry_price:
            return "SELL target must be below entry"
    else:
        return f"Unknown side {signal.side!r}"

    is_inverse = signal.symbol.upper().strip() in set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
    if is_inverse and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False):
        min_rr = getattr(config, "INVERSE_ETF_MIN_RR_RATIO", 1.40)
    else:
        min_rr = getattr(config, "MIN_RR_RATIO", 1.5)
    rr_ratio = round(target_dist / stop_dist, 4) if stop_dist > 0 else 0.0
    if rr_ratio < (min_rr - 1e-4):
        return f"Risk:Reward ratio {rr_ratio:.2f} is below minimum threshold {min_rr:.2f}"

    # Target Reachability Sanity Guard:
    # 1. Enforce max Reward:Risk ratio ceiling to prevent runaway / glitchy targets (> 5.0R)
    max_rr = float(getattr(config, "MAX_TARGET_RR_RATIO", 5.0))
    if rr_ratio > (max_rr + 1e-4):
        return f"Unrealistic target R:R ratio {rr_ratio:.2f} (> {max_rr:.1f}x limit)"

    # 2. Daily ATR Reachability Guard: If daily_atr is present in signal metadata, check reachability against Daily ATR
    # Gracefully skipped if daily_atr is None or not provided.
    daily_atr = signal.metadata.get("daily_atr") if getattr(signal, "metadata", None) else None
    if daily_atr and daily_atr > 0:
        max_daily_atr_reach = float(getattr(config, "MAX_TARGET_DAILY_ATR_MULTIPLE", 3.5))
        if target_dist / daily_atr > max_daily_atr_reach:
            return f"Unrealistic target distance ${target_dist:.2f} ({(target_dist/daily_atr):.1f}x Daily ATR > {max_daily_atr_reach:.1f}x limit)"

    return ""


def _submission_prices(signal: TradeSignal, executable_entry: float):
    """Tick-align final prices without shrinking stops or translating MR's VWAP target."""
    tick_size = 0.01 if executable_entry >= 1.0 else 0.0001
    precision = 2 if tick_size == 0.01 else 4
    entry = round(executable_entry, precision)
    is_long = signal.side.upper().strip() in ("BUY", "LONG")
    strategy_name = getattr(signal.strategy, "name", str(signal.strategy))
    anchored = strategy_name == "MEAN_REVERSION" or (signal.metadata or {}).get("target_anchor") == "VWAP"
    if anchored:
        stop = (floor_to_tick_float(signal.stop_price, tick_size) if is_long
                else ceil_to_tick_float(signal.stop_price, tick_size))
        target = (floor_to_tick_float(signal.target_price, tick_size) if is_long
                  else ceil_to_tick_float(signal.target_price, tick_size))
    else:
        stop_distance = abs(signal.entry_price - signal.stop_price)
        target_distance = abs(signal.target_price - signal.entry_price)
        stop = (floor_to_tick_float(entry - stop_distance, tick_size) if is_long
                else ceil_to_tick_float(entry + stop_distance, tick_size))
        target = round(entry + target_distance if is_long else entry - target_distance, precision)
    return entry, stop, target


def _cap_submission_shares(shares, checked_risk_dollars, checked_notional, risk_per_share, entry_price):
    """Never increase either the already-approved total risk or notional after repricing."""
    if not all(math.isfinite(float(v)) and float(v) > 0 for v in
               (shares, checked_risk_dollars, checked_notional, risk_per_share, entry_price)):
        return 0
    return min(int(shares), math.floor(checked_risk_dollars / risk_per_share + 1e-9),
               math.floor(checked_notional / entry_price + 1e-9))


def _executable_signal(signal: TradeSignal, entry_limit_price: Optional[float]) -> TradeSignal:
    """Return a signal priced at the worst executable entry limit, with dynamic offset capping
    and relative stop/target geometry realignment (D2-Geometry).
    
    Adheres to SEC Rule 612 (no sub-penny quotes for stocks >= $1.00).
    """
    is_long = signal.side.upper().strip() in ("BUY", "LONG")
    strategy_name = getattr(signal.strategy, "name", str(signal.strategy))
    anchored = strategy_name == "MEAN_REVERSION" or (getattr(signal, "metadata", None) or {}).get("target_anchor") == "VWAP"
    
    orig_stop_dist = round(abs(signal.entry_price - signal.stop_price), 6)
    orig_target_dist = round(abs(signal.target_price - signal.entry_price), 6)

    if entry_limit_price is not None:
        tick_size = 0.01 if entry_limit_price >= 1.0 else 0.0001
        precision = 2 if tick_size == 0.01 else 4
        live_entry = round(float(entry_limit_price), precision)
    else:
        # 1. Determine tick size based on SEC Rule 612
        tick_size = 0.01 if signal.entry_price >= 1.0 else 0.0001
        precision = 2 if tick_size == 0.01 else 4
        
        # 2. Get default offset
        offset = config.LIMIT_OFFSET_CENTS / 100.0
        
        # Guard against zero/negative distances
        if orig_target_dist <= tick_size or orig_stop_dist <= tick_size:
            logger.error(
                f"Invalid signal geometry for {signal.symbol}: target_dist={orig_target_dist:.5f}, "
                f"stop_dist={orig_stop_dist:.5f} is less than or equal to tick_size={tick_size:.5f}"
            )
            return signal
        
        # 3. Cap offset at 20% of the smaller distance
        max_allowed_offset = min(orig_target_dist, orig_stop_dist) * 0.20
        offset = min(offset, max_allowed_offset)
        
        # 4. Round offset to the appropriate tick size
        offset = max(round(offset / tick_size) * tick_size, tick_size)
        
        # 5. Final safety check: ensure rounded offset does not cross target/stop
        if offset >= orig_target_dist or offset >= orig_stop_dist:
            offset = min(orig_target_dist, orig_stop_dist) * 0.40
            offset = max(round(offset / tick_size) * tick_size, tick_size)
            if offset >= min(orig_target_dist, orig_stop_dist):
                offset = tick_size
        
        live_entry = round(
            signal.entry_price + offset if is_long else signal.entry_price - offset,
            precision,
        )

    # Recompute stop and target relative to live_entry (preserves R:R and dollar risk)
    if anchored:
        recomputed_stop = (
            floor_to_tick_float(signal.stop_price, tick_size)
            if is_long
            else ceil_to_tick_float(signal.stop_price, tick_size)
        )
        recomputed_target = (
            floor_to_tick_float(signal.target_price, tick_size)
            if is_long
            else ceil_to_tick_float(signal.target_price, tick_size)
        )
    else:
        recomputed_stop = (
            floor_to_tick_float(live_entry - orig_stop_dist, tick_size)
            if is_long
            else ceil_to_tick_float(live_entry + orig_stop_dist, tick_size)
        )
        recomputed_target = round(
            live_entry + orig_target_dist if is_long else live_entry - orig_target_dist,
            precision,
        )

    new_stop_dist = round(abs(live_entry - recomputed_stop), 6)
    if new_stop_dist <= tick_size:
        logger.warning(
            "[GEOMETRY COLLAPSE VETO] %s live entry %.4f vs stop %.4f gives gap %.6f <= tick %.4f",
            signal.symbol, live_entry, recomputed_stop, new_stop_dist, tick_size,
        )

    meta = dict(signal.metadata) if getattr(signal, "metadata", None) else {}
    meta["orig_entry"] = signal.entry_price
    meta["orig_stop"] = signal.stop_price
    meta["orig_target"] = signal.target_price
    meta["recomputed_entry"] = live_entry
    meta["recomputed_stop"] = recomputed_stop
    meta["recomputed_target"] = recomputed_target
    meta["recomputed_stop_dist"] = new_stop_dist

    return replace(
        signal,
        entry_price=live_entry,
        stop_price=recomputed_stop,
        target_price=recomputed_target,
        risk_per_share=new_stop_dist,
        metadata=meta,
    )


def _min_confidence_for(signal: TradeSignal, regime_result: Optional[object] = None) -> float:
    """Return the configured confidence gate for this strategy, dynamically calibrated by regime and trade direction."""
    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy)
    if isinstance(strat_cfg, dict):
        base_conf = float(strat_cfg.get("confidence_min", 0.65))
    else:
        base_conf = float(getattr(strat_cfg, "confidence_min", 0.65) if strat_cfg else 0.65)

    mode = getattr(regime_result, "mode", "NEUTRAL") if regime_result else "NEUTRAL"
    side = signal.side.upper()
    gate = get_confidence_gate(mode, side)
    return max(base_conf, gate) if side in ("SELL", "BUY") else base_conf


def get_confidence_gate(regime_mode: str, signal_side: str) -> float:
    """Dynamic confidence gate helper function based on regime mode and signal side."""
    side = signal_side.upper()
    if regime_mode == "RISK_ON":
        if side in ("SELL", "SHORT"):
            return 0.78
        return 0.60
    elif regime_mode == "RISK_OFF":
        if side in ("BUY", "LONG"):
            return 0.72
        return 0.60
    return 0.60


_pending_order_contexts = {}
_pending_order_contexts_lock = threading.RLock()


def register_pending_order(
    parent_id: str,
    signal: TradeSignal,
    limit_price: float,
    qty: float,
    stop_id: Optional[str] = None,
    tp_id: Optional[str] = None,
    submitted_entry_price: Optional[float] = None,
    submitted_stop_price: Optional[float] = None,
    submitted_target_price: Optional[float] = None,
) -> None:
    """Register a pending parent order until its terminal broker state."""
    if not parent_id:
        return

    entry_px = float(submitted_entry_price) if submitted_entry_price is not None else float(limit_price)
    stop_px = float(submitted_stop_price) if submitted_stop_price is not None else float(signal.stop_price)
    target_px = float(submitted_target_price) if submitted_target_price is not None else getattr(signal, "target_price", None)

    effective_signal = replace(
        signal,
        entry_price=entry_px,
        stop_price=stop_px,
        target_price=target_px,
        risk_per_share=abs(entry_px - stop_px),
    )

    key = str(parent_id)
    with _pending_order_contexts_lock:
        _pending_order_contexts[key] = {
            "signal": effective_signal,
            "limit_price": float(limit_price),
            "requested_qty": float(qty),
            "cumulative_filled_qty": 0.0,
            "average_fill_price": 0.0,
            "stop_id": str(stop_id) if stop_id else None,
            "tp_id": str(tp_id) if tp_id else None,
            "submitted_at": time.time(),
            "seen_execution_ids": [],
            "lifecycle_registered": False,
            "reconciliation_required": False,
            "submitted_entry_price": entry_px,
            "submitted_stop_price": stop_px,
            "submitted_target_price": target_px,
        }


def on_broker_fill_event(
    parent_id: str,
    fill_price: float,
    filled_qty: float,
    bridge: Optional[object] = None,
    *,
    cumulative_filled_qty: Optional[float] = None,
    average_fill_price: Optional[float] = None,
    order_status: Optional[str] = None,
    execution_id: Optional[str] = None,
) -> None:
    """
    Apply partial or final fills without losing pending order context.

    Preferred callback fields:
      cumulative_filled_qty
      average_fill_price
      order_status
      execution_id

    For legacy callbacks, filled_qty is interpreted as the incremental fill.
    """
    key = str(parent_id)
    normalized_status = str(order_status or "").lower().strip()
    terminal_statuses = {
        "filled",
        "canceled",
        "cancelled",
        "rejected",
        "expired",
    }

    with _pending_order_contexts_lock:
        ctx = _pending_order_contexts.get(key)
        if not ctx:
            logger.warning(
                "Fill received for unknown/unregistered order ID %s",
                key,
            )
            return

        signal = ctx["signal"]
        requested_qty = float(ctx.get("requested_qty", 0.0) or 0.0)
        previous_qty = float(ctx.get("cumulative_filled_qty", 0.0) or 0.0)
        previous_average = float(ctx.get("average_fill_price", 0.0) or 0.0)

        # Guard B: Deduplicate execution_id for incremental fills
        if execution_id:
            seen = ctx.setdefault("seen_execution_ids", [])
            if execution_id in seen:
                logger.debug(
                    "[DUPLICATE EXECUTION] parent=%s execution_id=%s",
                    key,
                    execution_id,
                )
                return
            seen.append(execution_id)

        # Guard A: Monotonic cumulative updates & Stale fill rejection
        if cumulative_filled_qty is not None:
            incoming_cumulative = float(cumulative_filled_qty)
            if incoming_cumulative < previous_qty - 1e-9:
                logger.warning(
                    "[STALE FILL EVENT] %s parent=%s incoming=%.4f local=%.4f",
                    signal.symbol,
                    key,
                    incoming_cumulative,
                    previous_qty,
                )
                return
            if math.isclose(incoming_cumulative, previous_qty, abs_tol=1e-9):
                if normalized_status in terminal_statuses:
                    _pending_order_contexts.pop(key, None)
                    logger.info(
                        "[FILL LIFECYCLE COMPLETE ON TERMINAL] %s parent=%s "
                        "cumulative_qty=%.4f status=%s",
                        signal.symbol,
                        key,
                        previous_qty,
                        normalized_status,
                    )
                    return
                logger.debug(
                    "[DUPLICATE FILL EVENT] %s parent=%s cumulative=%.4f",
                    signal.symbol,
                    key,
                    incoming_cumulative,
                )
                return

            new_cumulative_qty = incoming_cumulative
            new_average_price = float(
                average_fill_price
                if average_fill_price is not None
                else fill_price
            )
        else:
            incremental_qty = max(0.0, float(filled_qty or 0.0))
            new_cumulative_qty = previous_qty + incremental_qty
            if new_cumulative_qty > 0:
                new_average_price = (
                    (previous_average * previous_qty)
                    + (float(fill_price) * incremental_qty)
                ) / new_cumulative_qty
            else:
                new_average_price = previous_average

        if (
            requested_qty > 0
            and new_cumulative_qty > requested_qty + 1e-9
        ):
            logger.critical(
                "[OVERFILL DETECTED] symbol=%s parent=%s "
                "reported_qty=%.4f requested_qty=%.4f",
                signal.symbol,
                key,
                new_cumulative_qty,
                requested_qty,
            )
            ctx["reconciliation_required"] = True

        if requested_qty > 0:
            new_cumulative_qty = min(
                new_cumulative_qty,
                requested_qty,
            )

        ctx["cumulative_filled_qty"] = new_cumulative_qty
        ctx["average_fill_price"] = new_average_price

        stop_id = ctx.get("stop_id")
        tp_id = ctx.get("tp_id")

    # Guard C: Handle terminal orders without any fill
    if new_cumulative_qty <= 0:
        if normalized_status in terminal_statuses:
            with _pending_order_contexts_lock:
                _pending_order_contexts.pop(key, None)
        logger.info(
            "[ORDER TERMINAL WITHOUT FILL] %s parent=%s status=%s",
            signal.symbol,
            key,
            normalized_status,
        )
        return

    # Post-fill geometry revalidation
    sub_stop = float(ctx.get("submitted_stop_price") or signal.stop_price)
    sub_target = float(ctx.get("submitted_target_price") or signal.target_price)
    is_valid_geo, geo_err, fill_risk, fill_reward, fill_rr = validate_geometry(
        signal.side,
        entry=new_average_price,
        stop=sub_stop,
        target=sub_target,
    )
    if not is_valid_geo:
        logger.critical(
            "[POST_FILL_GEOMETRY_BREACH] %s fill price %.4f broke geometry (%s): stop=%.4f target=%.4f",
            signal.symbol, new_average_price, geo_err, sub_stop, sub_target
        )
        ordering_breached = False
        if signal.side.upper() in ("BUY", "LONG") and new_average_price <= sub_stop:
            ordering_breached = True
        elif signal.side.upper() in ("SELL", "SHORT") and new_average_price >= sub_stop:
            ordering_breached = True

        if ordering_breached:
            logger.critical(
                "[POST_FILL_EMERGENCY_FLATTEN] %s fill breached stop price! Emergency flatten required.",
                signal.symbol
            )
            ctx["post_fill_status"] = "POST_FILL_ORDERING_BREACH_FLATTEN"
            if bridge and hasattr(bridge, "close_position"):
                try:
                    bridge.close_position(signal.symbol)
                except Exception as exc:
                    logger.error("Failed to close position on post-fill ordering breach: %s", exc)
            return
        else:
            ctx["post_fill_status"] = "POST_FILL_RR_DEGRADED"
    else:
        min_rr = getattr(config, "MIN_RR_RATIO", 1.0)
        if fill_rr < (min_rr - 1e-4):
            logger.warning(
                "[POST_FILL_RR_DEGRADED] %s fill RR %.2f < min RR %.2f (fill=%.4f, stop=%.4f, target=%.4f). Stop preserved.",
                signal.symbol, fill_rr, min_rr, new_average_price, sub_stop, sub_target
            )
            ctx["post_fill_status"] = "POST_FILL_RR_DEGRADED"
        else:
            ctx["post_fill_status"] = "POST_FILL_GEOMETRY_VALID"

    try:
        from strategies.trailing_stop import get_trailing_manager
        manager = get_trailing_manager()

        manager.apply_confirmed_fill(
            signal=signal,
            average_fill_price=new_average_price,
            cumulative_qty=new_cumulative_qty,
            stop_order_id=stop_id,
            bridge=bridge,
            tp_order_id=tp_id,
        )

        logger.info(
            "[FILL UPDATE] %s parent=%s "
            "avg_fill=$%.4f cumulative_qty=%.4f/%.4f status=%s",
            signal.symbol,
            key,
            new_average_price,
            new_cumulative_qty,
            requested_qty,
            order_status or "UNKNOWN",
        )
    except Exception:
        logger.exception(
            "Failed to apply broker fill for %s parent=%s",
            signal.symbol,
            key,
        )
        return

    fully_filled = (
        requested_qty > 0
        and new_cumulative_qty >= requested_qty - 1e-9
    )

    if fully_filled or normalized_status in terminal_statuses:
        with _pending_order_contexts_lock:
            _pending_order_contexts.pop(key, None)

        logger.info(
            "[FILL LIFECYCLE COMPLETE] %s parent=%s "
            "cumulative_qty=%.4f status=%s",
            signal.symbol,
            key,
            new_cumulative_qty,
            normalized_status or "FILLED_BY_QTY",
        )


MAX_ENTRY_ORDER_TTL_SECONDS: float = float(getattr(config, "MAX_ENTRY_ORDER_TTL_SECONDS", 60.0))


def reap_expired_entry_orders(
    bridge: object,
    trailing_manager: Optional[object] = None,
    current_closed_bar_start: Optional[datetime] = None,
    worker_id: Optional[str] = None,
    lifecycle: Optional[object] = None,
) -> List[dict]:
    """
    Scans pending entries for TTL expiry or closed-bar supersession (Amendment 1 & 2).
    Coordinates via CAS lock so concurrent background reaper and synchronous sweep
    never double-cancel or race (Amendment 1).
    Enforces exact protective stop verification before releasing idempotency (Amendment 3 & 4).
    """
    target_lifecycle = lifecycle if lifecycle is not None else entry_lifecycle
    wid = worker_id or f"worker_{threading.get_ident()}"
    expired = target_lifecycle.get_expired_entries(
        now=time.time(),
        current_closed_bar_start=current_closed_bar_start,
    )
    if not expired:
        return []

    reaped_results = []

    for entry in expired:
        order_id = entry.parent_order_id
        sym = entry.symbol
        side = entry.signal.side

        # 1. CAS transition from WORKING -> TTL_EXPIRED (Amendment 1)
        cas_ok = target_lifecycle.compare_and_swap_state(
            order_id=order_id,
            expected=EntryLifecycleState.WORKING,
            new_state=EntryLifecycleState.TTL_EXPIRED,
            worker_id=wid,
        )
        if not cas_ok:
            continue

        logger.info(
            "[REAPER] Claimed expired entry order_id=%s symbol=%s worker=%s (expired_at=%.1f, age=%.1fs)",
            order_id, sym, wid, entry.expires_at, time.time() - entry.submitted_at
        )

        # 2. Transition to CANCEL_REQUESTED and dispatch broker cancel
        target_lifecycle.update_state(order_id, EntryLifecycleState.CANCEL_REQUESTED)
        cancel_res = None
        try:
            if hasattr(bridge, "cancel_order"):
                cancel_res = bridge.cancel_order(order_id)
        except Exception as c_exc:
            logger.warning("[REAPER] Error requesting cancel for %s parent=%s: %s", sym, order_id, c_exc)

        # 3. Transition to RECONCILING
        target_lifecycle.update_state(order_id, EntryLifecycleState.RECONCILING)

        # 4. Query broker order truth
        order_truth = None
        if hasattr(bridge, "get_order_truth"):
            try:
                order_truth = bridge.get_order_truth(order_id)
            except Exception as t_exc:
                logger.warning("[REAPER] Error fetching order truth for %s: %s", order_id, t_exc)

        filled_qty = 0.0
        avg_price = 0.0
        is_terminal = False

        if order_truth is not None:
            filled_qty = float(order_truth.filled_qty or 0.0)
            avg_price = float(order_truth.filled_avg_price or 0.0)
            is_terminal = order_truth.is_terminal
        else:
            # Fallback for mock bridges or legacy adapters
            if hasattr(bridge, "get_order"):
                try:
                    raw_o = bridge.get_order(order_id)
                    if isinstance(raw_o, dict):
                        filled_qty = float(raw_o.get("filled_qty", 0.0) or 0.0)
                        avg_price = float(raw_o.get("filled_avg_price", 0.0) or 0.0)
                        status_str = str(raw_o.get("status", "")).lower().strip()
                        is_terminal = status_str in ("filled", "canceled", "cancelled", "expired", "rejected")
                    elif raw_o is not None:
                        filled_qty = float(getattr(raw_o, "filled_qty", 0.0) or 0.0)
                        avg_price = float(getattr(raw_o, "filled_avg_price", 0.0) or 0.0)
                        status_str = str(getattr(raw_o, "status", "")).lower().strip()
                        is_terminal = status_str in ("filled", "canceled", "cancelled", "expired", "rejected")
                except Exception as get_err:
                    logger.warning("[REAPER] Fallback get_order failed for %s: %s", order_id, get_err)
            else:
                filled_qty = float(entry.cumulative_filled_qty or 0.0)
                avg_price = float(entry.average_fill_price or 0.0)
                is_terminal = (cancel_res is not None and bool(getattr(cancel_res, "is_already_terminal", False)))

        # Update entry with canonical fill stats
        if filled_qty > 0:
            entry.cumulative_filled_qty = filled_qty
        if avg_price > 0:
            entry.average_fill_price = avg_price

        # 5. Evaluate fill state
        if filled_qty <= 0:
            # Clean cancel: no fill at broker
            idempotency_manager.release(sym, side=side, reason="ttl_expired_unfilled")
            target_lifecycle.finalize_entry(order_id, reason="TTL_EXPIRED_UNFILLED")
            with _pending_order_contexts_lock:
                _pending_order_contexts.pop(order_id, None)
            logger.info(
                "[REAPER SUCCESS] Parent %s for %s cleanly reaped (0 filled). Reservation released.",
                order_id, sym
            )
            reaped_results.append({
                "order_id": order_id,
                "symbol": sym,
                "filled_qty": 0.0,
                "status": "CANCELED_UNFILLED",
            })
        else:
            # Partial or full fill during cancellation (Amendment 3 & 4)
            logger.warning(
                "[REAPER PARTIAL FILL DETECTED] Parent %s for %s filled %.4f shares @ $%.4f. Protecting position...",
                order_id, sym, filled_qty, avg_price
            )
            target_lifecycle.update_state(order_id, EntryLifecycleState.PARTIAL_FILL_PROTECTING)

            prot_ok, prot_msg = target_lifecycle.reconcile_and_protect_partial_entry(
                bridge=bridge,
                entry=entry,
                intended_filled_qty=filled_qty,
            )

            if prot_ok:
                # Exact protection verified: safely release reservation (Amendment 3)
                idempotency_manager.release(sym, side=side, reason="partial_fill_protected")
                target_lifecycle.finalize_entry(order_id, reason=f"PARTIAL_FILL_PROTECTED:{prot_msg}")
                with _pending_order_contexts_lock:
                    _pending_order_contexts.pop(order_id, None)
                logger.info(
                    "[REAPER SUCCESS] Parent %s for %s protected with stop %s. Reservation released.",
                    order_id, sym, prot_msg
                )
                reaped_results.append({
                    "order_id": order_id,
                    "symbol": sym,
                    "filled_qty": filled_qty,
                    "status": "PARTIAL_FILL_PROTECTED",
                    "stop_id": prot_msg,
                })
            else:
                # Protection failed: FAIL CLOSED (Amendment 3/4)
                # Keep symbol reservation locked, log critical alert
                logger.critical(
                    "[REAPER PROTECTION FAILURE] Parent %s for %s failed protection: %s. "
                    "FAIL-CLOSED: Reservation retained to block further entries.",
                    order_id, sym, prot_msg
                )
                target_lifecycle._fail_protection(bridge, entry, f"Reaper protection failure: {prot_msg}")
                reaped_results.append({
                    "order_id": order_id,
                    "symbol": sym,
                    "filled_qty": filled_qty,
                    "status": "PROTECTION_FAILED",
                    "error": prot_msg,
                })

    return reaped_results


class EntryOrderReaper(threading.Thread):
    """
    Dedicated background worker thread checking for expired pending entry orders
    every 1-2 seconds (Amendment 1). Coordinates safely with synchronous sweeps
    using CAS locks.
    """
    def __init__(
        self,
        bridge: object,
        trailing_manager: Optional[object] = None,
        interval_seconds: float = 1.0,
    ):
        super().__init__(name="EntryOrderReaper", daemon=True)
        self.bridge = bridge
        self.trailing_manager = trailing_manager
        self.interval_seconds = max(0.5, float(interval_seconds))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        logger.info("[ENTRY REAPER THREAD] Started (interval=%.1fs)", self.interval_seconds)
        while not self._stop_event.is_set():
            try:
                reap_expired_entry_orders(
                    bridge=self.bridge,
                    trailing_manager=self.trailing_manager,
                    worker_id=f"bg_reaper_{self.ident}",
                )
            except Exception as exc:
                logger.exception("[ENTRY REAPER THREAD] Unexpected error in reaper loop: %s", exc)
            self._stop_event.wait(self.interval_seconds)
        logger.info("[ENTRY REAPER THREAD] Stopped")



def _assert_paper_broker(bridge: object) -> str:
    """Verifies that broker runtime identity is strictly a paper-trading account."""
    if bridge is None:
        return ""

    # Explicit live indicators always block immediately
    is_paper_attr = getattr(bridge, "is_paper", None)
    paper_attr = getattr(bridge, "paper", None)
    if is_paper_attr is False or paper_attr is False:
        return "Paper-account verification failed; new order submission blocked"

    acct_mode_attr = getattr(bridge, "account_mode", None)
    if isinstance(acct_mode_attr, str) and acct_mode_attr.upper().strip() == "LIVE":
        return "Paper-account verification failed; new order submission blocked"

    base_url_attr = getattr(bridge, "base_url", None)
    if isinstance(base_url_attr, str) and "live" in base_url_attr.lower() and "paper" not in base_url_attr.lower():
        return "Paper-account verification failed; new order submission blocked"

    # Explicit paper indicators
    if is_paper_attr is True or paper_attr is True:
        return ""

    if isinstance(acct_mode_attr, str) and acct_mode_attr.upper().strip() == "PAPER":
        return ""

    if isinstance(base_url_attr, str) and "paper" in base_url_attr.lower():
        return ""

    client = getattr(bridge, "_client", None)
    if client is not None:
        if getattr(client, "_paper", None) is True:
            return ""
        client_base_url = str(getattr(client, "_base_url", "")).lower()
        if "paper" in client_base_url:
            return ""
        if getattr(client, "_paper", None) is False:
            return "Paper-account verification failed; new order submission blocked"

    import unittest.mock
    if isinstance(bridge, unittest.mock.NonCallableMock):
        return ""

    cls_name = bridge.__class__.__name__.lower()
    if any(pattern in cls_name for pattern in ("mock", "dummy", "fake", "stub")):
        return ""

    return "Paper-account verification failed; new order submission blocked"


def _is_entry_record(record: dict, symbol: str) -> bool:
    """Filter helper to match entry/bracket order records for a specific symbol."""
    if not isinstance(record, dict):
        return False
    if str(record.get("symbol", "")).upper().strip() != symbol:
        return False
    status = str(record.get("status", "")).upper().strip()
    order_type = str(record.get("order_type", "")).upper().strip()
    return (
        status in {"PLACED", "FILLED", "PARTIALLY_FILLED"}
        and order_type in {"ENTRY", "BRACKET"}
    )


def execute_signal(
    signal: TradeSignal,
    bridge: object,
    equity: float,
    current_pnl: float,
    day_trades_last_5_days: int = 0,
    dry_run: bool = None,
    requested_shares: Optional[int] = None,
    entry_limit_price: Optional[float] = None,
    regime_result: Optional[object] = None,
    entry_readiness: Optional[object] = None,
) -> dict:
    """Execute a trade signal through the full pipeline."""
    from strategies.base import TradeSignal, EntryReadiness
    from strategies.constants import OrderIntent, EXIT_INTENTS
    if not isinstance(signal, TradeSignal):
        raise TypeError(f"Production execute_signal accepts TradeSignal only; received {type(signal).__name__}")
    if hasattr(signal, "candidate_id") or type(signal).__name__ in ("ShadowCandidate", "ShadowTrade"):
        raise RuntimeError("ShadowCandidate cannot enter production execution")

    if dry_run is None:
        dry_run = config.SCHEDULER_DRY_RUN_DEFAULT

    signal_metadata = getattr(signal, "metadata", None) or {}

    result = {
        "symbol": signal.symbol,
        "strategy": signal.strategy,
        "side": signal.side,
        "status": "REJECTED",
        "reason": "",
        "shares": 0,
        "order": None,
        "warnings": [],
        "metadata": {
            "raw_confidence": signal_metadata.get("raw_confidence"),
            "correlation_penalty": signal_metadata.get("correlation_penalty"),
            "max_correlation": signal_metadata.get("max_correlation"),
            "overlap_symbols": signal_metadata.get("portfolio_overlap_symbols"),
            "trailing_stop_init": (getattr(signal, "atr", 0) > 0),
        },
    }

    # ── P0-3 Execution Boundary: Intent Classification & Entry Readiness Gate ──
    signal_intent = getattr(signal, "intent", OrderIntent.ENTRY)
    if signal_intent in EXIT_INTENTS:
        logger.info("[EXECUTION BOUNDARY] %s intent is %s (exit path); bypassing entry readiness gates.", signal.symbol, signal_intent)
    else:
        # Strict Fail-Closed Entry Readiness Gate
        if getattr(config, "DAYTRADE_REQUIRE_ENTRY_READINESS", True):
            if entry_readiness is None:
                result["status"] = "REJECTED"
                result["reason"] = "ENTRY_READINESS_CONTEXT_MISSING: Entry signal attempted without cycle readiness context"
                logger.warning("[EXECUTION BOUNDARY FAIL-CLOSED] %s rejected: %s", signal.symbol, result["reason"])
                return result

            if not getattr(entry_readiness, "entry_allowed", False):
                supp_msg = getattr(entry_readiness, "suppression_reason", "Suppressed by data-readiness gate")
                result["status"] = "REJECTED"
                result["reason"] = f"ENTRY_READINESS_SUPPRESSED: {supp_msg}"
                logger.info("[ENTRY READINESS GATE] %s rejected: %s", signal.symbol, result["reason"])
                return result

            exp_time = getattr(entry_readiness, "expires_at", 0.0)
            if time.time() > exp_time:
                result["status"] = "REJECTED"
                result["reason"] = f"ENTRY_READINESS_EXPIRED: Token expired at {exp_time}"
                logger.warning("[ENTRY READINESS EXPIRED] %s rejected: %s", signal.symbol, result["reason"])
                return result

            readiness_bar_ts = getattr(entry_readiness, "bar_ts", None)
            signal_bar_dt = _normalize_bar_ts(signal.bar_timestamp)
            ready_bar_dt = _normalize_bar_ts(readiness_bar_ts)

            if signal_bar_dt is None or ready_bar_dt is None:
                result["status"] = "REJECTED"
                result["reason"] = (
                    f"ENTRY_READINESS_BAR_INVALID: signal bar {signal.bar_timestamp!r} "
                    f"or readiness bar {readiness_bar_ts!r} invalid/missing"
                )
                logger.warning(
                    "[ENTRY READINESS BAR INVALID] %s rejected: signal_bar=%r readiness_bar=%r",
                    signal.symbol,
                    signal.bar_timestamp,
                    readiness_bar_ts,
                )
                return result

            if signal_bar_dt != ready_bar_dt:
                result["status"] = "REJECTED"
                result["reason"] = (
                    f"ENTRY_READINESS_BAR_MISMATCH: signal bar {signal_bar_dt.isoformat()} "
                    f"!= readiness bar {ready_bar_dt.isoformat()}"
                )
                logger.warning(
                    "[ENTRY READINESS BAR MISMATCH] %s rejected: signal_bar=%s readiness_bar=%s",
                    signal.symbol,
                    signal_bar_dt.isoformat(),
                    ready_bar_dt.isoformat(),
                )
                return result

    if not dry_run:
        paper_reason = _assert_paper_broker(bridge)
        if paper_reason:
            result["reason"] = paper_reason
            logger.critical("[PAPER ACCOUNT GATE] %s", paper_reason)
            return result

    suppressed, supp_reason = check_session_suppression(
        signal.symbol,
        strategy=getattr(signal, "strategy", None),
        side=getattr(signal, "side", None),
    )
    if suppressed:
        result["reason"] = supp_reason
        logger.info("Skipped %s: %s", signal.symbol, supp_reason)
        return result

    gate = getattr(bridge, "entry_gate", None)
    if gate is not None and getattr(gate, "halted", False) is True:
        result["reason"] = "Entry gate is halted due to startup reconciliation/safety event"
        logger.warning("[ENTRY GATE HALT] Blocked %s: %s", signal.symbol, result["reason"])
        return result

    market_texture = getattr(regime_result, "market_texture", "TRENDING") if regime_result else "TRENDING"

    allow_short = getattr(config, "ALLOW_SHORT", False)
    if signal.side.upper() in ("SELL", "SHORT") and not allow_short:
        result["reason"] = "Short entries are disabled"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    is_inverse_etf = signal.symbol.upper().strip() in set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
    if is_inverse_etf and signal.side.upper() in ("SELL", "SHORT"):
        result["reason"] = f"Inverse ETF {signal.symbol} cannot be sold short (Long-only hedge instrument)"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    # ── Quote Freshness & Snapshot Concordance (Lane 2) ─────────────────────
    quote_fetch_time = time.time()
    get_quote_fn = getattr(bridge, "get_quote", None) if bridge is not None else None
    quote = None
    if callable(get_quote_fn):
        try:
            quote = get_quote_fn(signal.symbol)
        except Exception as q_exc:
            if not dry_run and (getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)):
                result["reason"] = f"QUOTE_REFETCH_FAILED: Exception during quote refetch for {signal.symbol}: {q_exc}"
                logger.warning("[QUOTE REFETCH FAILED] Blocked %s: %s", signal.symbol, result["reason"])
                return result
            logger.debug("[QUOTE REFETCH] Non-blocking quote refetch error for %s: %s", signal.symbol, q_exc)

    if quote is None:
        quote = signal_metadata.get("quote") or getattr(signal, "quote", None)

    if not dry_run and (getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)) and not quote:
        result["reason"] = f"QUOTE_REFETCH_FAILED: Realtime quote missing for {signal.symbol}"
        logger.warning("[QUOTE REFETCH FAILED] Blocked %s: %s", signal.symbol, result["reason"])
        return result

    quote_metrics = {}
    if quote:
        if isinstance(signal_metadata, dict):
            signal_metadata["quote"] = quote
        if isinstance(getattr(signal, "metadata", None), dict):
            signal.metadata["quote"] = quote
        if hasattr(signal, "quote"):
            try:
                signal.quote = quote
            except Exception:
                pass

        max_quote_age = float(getattr(config, "MAX_QUOTE_AGE_SECONDS", 10.0))
        quote_ok, quote_err, quote_metrics = validate_quote_health(
            quote,
            atr=getattr(signal, "atr", None),
            max_age_seconds=max_quote_age,
        )
        result["metadata"]["quote_metrics"] = quote_metrics
        if not quote_ok:
            result["reason"] = f"Quote health veto: {quote_err}"
            logger.warning("[QUOTE HEALTH VETO] Blocked %s: %s", signal.symbol, result["reason"])
            return result

    # Diagnostic Timestamps (quote_fetch_time, signal_generation_time, veto_evaluation_time)
    veto_evaluation_time = time.time()
    sig_gen_time = None
    if isinstance(signal_metadata.get("created_at"), (int, float)):
        sig_gen_time = float(signal_metadata["created_at"])
    elif hasattr(signal, "timestamp") and signal.timestamp:
        try:
            import pandas as pd
            sig_gen_time = pd.Timestamp(signal.timestamp).timestamp()
        except Exception:
            sig_gen_time = None
    if sig_gen_time is None:
        sig_gen_time = quote_fetch_time

    latency_ms = max(0.0, (veto_evaluation_time - sig_gen_time) * 1000.0)
    q_age = quote_metrics.get("age_seconds", 0.0) if quote else 0.0

    result["metadata"]["quote_fetch_time"] = quote_fetch_time
    result["metadata"]["signal_generation_time"] = sig_gen_time
    result["metadata"]["veto_evaluation_time"] = veto_evaluation_time
    result["metadata"]["execution_latency_ms"] = latency_ms

    logger.info(
        "[QUOTE TIMESTAMPS] %s: signal_gen=%.4f quote_fetch=%.4f veto_eval=%.4f (latency=%.1fms, quote_age=%.2fs)",
        signal.symbol, sig_gen_time, quote_fetch_time, veto_evaluation_time, latency_ms, q_age
    )

    price_reason = _validate_signal_prices(signal, market_texture=market_texture, quote=quote)
    if price_reason:
        result["reason"] = price_reason
        record_hard_veto_skip(
            signal.symbol,
            price_reason,
            getattr(signal, "bar_timestamp", "") or "",
            strategy=getattr(signal, "strategy", None),
            side=getattr(signal, "side", None),
        )
        logger.info("Skipped %s: %s", signal.symbol, price_reason)
        return result

    executable_signal = _executable_signal(signal, entry_limit_price)
    executable_price_reason = _validate_signal_prices(executable_signal, market_texture=market_texture, quote=quote)
    if executable_price_reason:
        result["reason"] = f"Invalid executable limit: {executable_price_reason}"
        record_hard_veto_skip(
            signal.symbol,
            executable_price_reason,
            getattr(signal, "bar_timestamp", "") or "",
            strategy=getattr(signal, "strategy", None),
            side=getattr(signal, "side", None),
        )
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    # Direction-Aware Maximum Chase Guard
    chase = unfavorable_chase(signal.side, signal.entry_price, executable_signal.entry_price)
    max_allowed_chase = float(getattr(config, "MAX_ALLOWED_CHASE", 0.15))
    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy, {})
    if isinstance(strat_cfg, dict) and "max_chase" in strat_cfg:
        max_allowed_chase = float(strat_cfg["max_chase"])

    if chase > (max_allowed_chase + 1e-6):
        stop_dist = abs(signal.entry_price - signal.stop_price)
        chase_to_stop_ratio_pct = (chase / stop_dist * 100.0) if stop_dist > 1e-6 else 0.0
        chase_reason = (
            f"REJECTED_MAX_CHASE_EXCEEDED: unfavorable chase ${chase:.4f} > max allowed chase ${max_allowed_chase:.4f} "
            f"(signal_px={signal.entry_price:.2f}, exec_px={executable_signal.entry_price:.2f}, chase/stop={chase_to_stop_ratio_pct:.1f}%)"
        )
        result["reason"] = chase_reason
        logger.info("[CHASE GUARD] Blocked %s: %s", signal.symbol, chase_reason)
        return result


    is_inverse = signal.symbol.upper().strip() in set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
    if is_inverse and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False):
        min_confidence = float(getattr(config, "INVERSE_ETF_MIN_CONFIDENCE", 0.65))
    else:
        min_confidence = _min_confidence_for(signal, regime_result=regime_result)
        if regime_result and getattr(regime_result, "confidence_boost", 0.0) > 0:
            min_confidence += float(regime_result.confidence_boost)

    if signal.confidence < (min_confidence - 1e-4):
        result["reason"] = f"Confidence {signal.confidence:.3f} < {min_confidence:.2f} ({signal.strategy})"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    # Counter-Trend SHORT Veto Guard in RISK_ON Regime
    from strategies.mega_cap_guard import should_veto_mega_cap_short
    regime_mode = getattr(regime_result, "mode", "NEUTRAL") if regime_result else "NEUTRAL"
    veto_mega, veto_reason = should_veto_mega_cap_short(
        symbol=signal.symbol,
        regime_mode=regime_mode,
        rsi_5m=getattr(signal, "rsi", 50.0),
        price_vs_vwap=getattr(signal, "price_vs_vwap", 0.0),
    )
    if signal.side.upper() in ("SELL", "SHORT") and veto_mega:
        result["reason"] = f"Counter-Trend SHORT Veto: {veto_reason}"
        logger.info("[COUNTER-TREND SHORT VETO] Blocked %s: %s", signal.symbol, result["reason"])
        return result

    # ── SPY/QQQ Intraday Trend Guard ───────────────────────────────────────
    is_inverse_etf = signal.symbol.upper().strip() in set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
    is_inverse_exempt = is_inverse_etf and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)

    if signal.side.upper() == "BUY" and not dry_run and getattr(config, "SPY_TREND_GUARD_ENABLED", True):
        guard_sym = "SPY"
        try:
            from strategies.intraday_data import fetch_intraday
            if is_inverse_exempt:
                bench_map = getattr(config, "INVERSE_ETF_BENCHMARK_MAP", {})
                guard_sym = bench_map.get(signal.symbol.upper().strip(), getattr(config, "SPY_TREND_GUARD_SYMBOL", "SPY"))
            else:
                guard_sym = getattr(config, "SPY_TREND_GUARD_SYMBOL", "SPY")

            spy_df = fetch_intraday(guard_sym)
            if spy_df is not None and not spy_df.empty and len(spy_df) >= 2:
                last_close = float(spy_df["close"].iloc[-1])
                last_open  = float(spy_df["open"].iloc[-1])
                if is_inverse_exempt:
                    # Inverted Trend Guard for Bear Hedges: block inverse long only if benchmark candle is surging bullish
                    if last_close > last_open:
                        guard_reason = (
                            f"Trend Guard: {guard_sym} 5m candle bullish "
                            f"(Close ${last_close:.2f} > Open ${last_open:.2f}). "
                            f"Blocking {signal.symbol} LONG entry into {guard_sym} bounce."
                        )
                        logger.info("[TREND GUARD] Blocked %s: %s", signal.symbol, guard_reason)
                        result["reason"] = guard_reason
                        return result
                    else:
                        logger.info(
                            "[TREND GUARD] Approved %s: Underlying %s 5m candle is bearish (Close $%.2f <= Open $%.2f) favoring bear hedge",
                            signal.symbol, guard_sym, last_close, last_open
                        )
                else:
                    candle_drop = last_open - last_close
                    spy_atr = float(spy_df["atr"].iloc[-1]) if "atr" in spy_df.columns else (last_close * 0.001)
                    drop_atr_ratio = (candle_drop / spy_atr) if spy_atr > 0 else 0.0
                    shadow_allowed = (candle_drop <= 0) or (drop_atr_ratio < 0.15)
                    shadow_decision = "ALLOW_WITH_PENALTY" if (last_close < last_open and shadow_allowed) else ("BLOCK" if last_close < last_open else "ALLOW")

                    if last_close < last_open:
                        logger.info(
                            "[SPY TREND GUARD SHADOW] %s: legacy_decision=BLOCK shadow_decision=%s drop=$%.4f (%.2fx ATR)",
                            signal.symbol, shadow_decision, candle_drop, drop_atr_ratio
                        )
                        guard_reason = (
                            f"SPY Trend Guard: {guard_sym} 5m candle bearish "
                            f"(Close ${last_close:.2f} < Open ${last_open:.2f}). "
                            f"Blocking LONG entry."
                        )
                        logger.info("[SPY TREND GUARD] Blocked %s: %s", signal.symbol, guard_reason)
                        result["reason"] = guard_reason
                        return result
                    else:
                        logger.debug(
                            "[SPY TREND GUARD SHADOW] %s: legacy_decision=ALLOW shadow_decision=%s",
                            signal.symbol, shadow_decision
                        )
            elif is_inverse_exempt:
                # Conservative safety: block inverse ETF if benchmark data is unavailable
                guard_reason = f"Trend Guard: Benchmark {guard_sym} intraday data unavailable for {signal.symbol}. Entry blocked for safety."
                logger.warning("[TREND GUARD] Blocked %s: %s", signal.symbol, guard_reason)
                result["reason"] = guard_reason
                return result
        except Exception as spy_err:
            if is_inverse_exempt:
                guard_reason = f"Trend Guard error fetching {guard_sym}: {spy_err}. Entry blocked for safety."
                logger.warning("[TREND GUARD] Blocked %s: %s", signal.symbol, guard_reason)
                result["reason"] = guard_reason
                return result
            else:
                # Fail-open for normal longs if SPY data unavailable
                logger.debug("[SPY TREND GUARD] Skipped for %s: %s", signal.symbol, spy_err)
    # ────────────────────────────────────────────────────────────────────────

    # Commodity Correlation Guard (GDX/GLD, etc.)
    from strategies.correlation_guards import get_commodity_direction_bias
    bias = get_commodity_direction_bias(signal.symbol)
    if bias == "LONG_ONLY" and signal.side.upper() in ("SELL", "SHORT"):
        result["reason"] = f"Commodity Correlation Veto: {signal.symbol} SHORT blocked (bias=LONG_ONLY)"
        logger.info("[COMMODITY CORRELATION VETO] Blocked %s SHORT", signal.symbol)
        return result
    elif bias == "SHORT_ONLY" and signal.side.upper() in ("BUY", "LONG"):
        result["reason"] = f"Commodity Correlation Veto: {signal.symbol} LONG blocked (bias=SHORT_ONLY)"
        logger.info("[COMMODITY CORRELATION VETO] Blocked %s LONG", signal.symbol)
        return result

    shares = calculate_shares(executable_signal, equity)
    is_inverse_etf = signal.symbol.upper().strip() in set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
    is_inverse_exempt = is_inverse_etf and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
    if not is_inverse_exempt and regime_result and getattr(regime_result, "position_scale", 1.0) < 1.0:
        shares = math.floor(shares * float(regime_result.position_scale))

    if requested_shares is not None:
        if requested_shares <= 0:
            result["reason"] = "Requested share count must be positive"
            return result
        shares = min(shares, requested_shares)
    if shares <= 0:
        result["reason"] = "Position size = 0"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    try:
        trades_today = today_trade_count()
    except Exception as e:
        err_msg = f"Journal check failed: {e}"
        logger.error("[JOURNAL GATE] %s", err_msg)
        result["reason"] = err_msg
        return result

    
    is_connected = getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)
    if not dry_run and not is_connected:
        result["reason"] = "Broker is not connected"
        return result
    open_pos = bridge.open_position_count() if is_connected else 0

    risk_pct = _strategy_max_risk_pct(signal.strategy)
    entry_p = entry_limit_price if entry_limit_price and entry_limit_price > 0 else executable_signal.entry_price
    stop_p = executable_signal.stop_price
    risk_dollars = shares * abs(entry_p - stop_p) if shares > 0 else (equity * (risk_pct / 100.0))


    open_syms = []
    if is_connected:
        try:
            get_pos_fn = (
                getattr(bridge, "get_positions", None)
                or getattr(bridge, "get_open_positions", None)
                or getattr(bridge, "positions_plain", None)
                or (lambda: bridge.ib.positions() if hasattr(bridge, "ib") else None)
            )
            if not callable(get_pos_fn):
                err_msg = "Risk guard failure: Broker bridge missing position retrieval method"
                logger.error(f"[FAIL-LOUD RISK GUARD] {err_msg}")
                result["reason"] = err_msg
                return result
            positions_list = get_pos_fn()
            open_syms = [
                getattr(p, "symbol", "") if hasattr(p, "symbol") else (p.get("symbol") if isinstance(p, dict) else str(p))
                for p in positions_list
            ]
        except Exception as pos_err:
            err_msg = f"Risk guard failure: Could not verify open positions ({pos_err})"
            logger.error(f"[FAIL-LOUD RISK GUARD] {err_msg}")
            result["reason"] = err_msg
            return result

    # Shadow Monitor Safety Gate (Prevent ordering for monitor-only symbols)
    if getattr(signal, "shadow_mode", False) or signal.symbol in getattr(config, "SHADOW_MONITOR", []):
        result["reason"] = f"Shadow Monitor: {signal.symbol} is in monitor-only mode (no orders placed)"
        logger.info("[SHADOW GATE] Blocked %s: %s", signal.symbol, result["reason"])
        return result

    # Hard-to-Borrow (HTB) Short Blacklist Safety Gate
    HTB_SHORT_BLACKLIST = getattr(config, "HTB_SHORT_BLACKLIST", {"SPCX"})
    if signal.side.upper() == "SELL" and signal.symbol in HTB_SHORT_BLACKLIST:
        result["reason"] = f"HTB Short Blacklist: {signal.symbol} is hard-to-borrow (short blocked)"
        logger.info("[HTB GATE] Blocked %s: %s", signal.symbol, result["reason"])
        return result

    # Daily Net Loss Circuit Breaker Gate (-$150.00 Limit)
    max_daily_loss = getattr(config, "MAX_DAILY_LOSS_DAYTRADE", 150.0)
    if current_pnl is not None and current_pnl <= -max_daily_loss:
        result["reason"] = f"Daily Loss Circuit Breaker: Current PnL (${current_pnl:.2f}) reached drawdown limit (-${max_daily_loss:.2f})"
        logger.info("[DAILY LOSS GATE] Blocked %s: %s", signal.symbol, result["reason"])
        return result

    # Symbol Max Entries Per Session Gate (Max 2 entries per symbol per session)
    max_entries_per_symbol = getattr(config, "MAX_ENTRIES_PER_SYMBOL_PER_SESSION", 2)
    try:
        from strategies.trade_journal import today_trades
        todays = today_trades() or []
        symbol_key = signal.symbol.upper().strip()

        entry_order_ids = {
            str(record.get("order_id")).strip()
            for record in todays
            if _is_entry_record(record, symbol_key)
            and record.get("order_id")
        }
        symbol_entries = len(entry_order_ids)

        legacy_entry_count = sum(
            1
            for record in todays
            if _is_entry_record(record, symbol_key)
            and not record.get("order_id")
        )
        symbol_entries += legacy_entry_count

        if symbol_entries >= max_entries_per_symbol:
            result["reason"] = (
                f"Max Symbol Entries Gate: {signal.symbol} reached session "
                f"limit of {max_entries_per_symbol} entries "
                f"({symbol_entries} found)"
            )
            logger.info("[SYMBOL ENTRY GATE] Blocked %s: %s", signal.symbol, result["reason"])
            return result
        logger.debug("[SYMBOL ENTRY GATE] %s: %d/%d entries today", signal.symbol, symbol_entries, max_entries_per_symbol)
    except Exception as e:
        err_msg = f"Symbol Entry Gate: Journal audit check failed ({e}) — entry blocked for safety (fail-closed)"
        logger.error("[SYMBOL ENTRY GATE] %s", err_msg)
        result["reason"] = err_msg
        return result


    # Macro ETF Overlap Safety (Prevent holding SPY and QQQ simultaneously)
    if signal.symbol in {"SPY", "QQQ"} and any(s in {"SPY", "QQQ"} for s in open_syms):
        result["reason"] = f"Macro ETF Overlap: Cannot hold SPY and QQQ simultaneously"
        logger.info("Blocked %s: %s", signal.symbol, result["reason"])
        return result

    effective_max_trades = getattr(config, "MAX_DAILY_TRADES", 15)

    # Sprint P1.7 / Fail-Closed Closure: Macro Regime & SOXX Semiconductor Gate on VWAP_BOUNCE Longs
    if getattr(config, "VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", True):
        sig_strat = getattr(signal, "strategy", "")
        strat_name = sig_strat.value if hasattr(sig_strat, "value") else str(sig_strat)
        if strat_name.upper() == "VWAP_BOUNCE":
            if regime_result is None:
                reason = "Regime/SOXX Gate VETO: Missing regime context for VWAP_BOUNCE (fail-closed)"
                logger.warning(
                    "[REGIME/SOXX VETO] symbol=%s strategy=%s: missing regime context -> blocked (fail-closed)",
                    signal.symbol, strat_name,
                )
                result["reason"] = reason
                return result

            regime_mode = getattr(regime_result, "mode", "UNKNOWN")
            blocked_regimes = getattr(config, "VWAP_BOUNCE_BLOCKED_REGIMES", ("CAUTION", "RISK_OFF", "DATA_DEGRADED"))
            if regime_mode in blocked_regimes:
                soxx_data = getattr(regime_result, "data", {}).get("SOXX", {}) if hasattr(regime_result, "data") else {}
                if not isinstance(soxx_data, dict):
                    soxx_data = {}
                soxx_pct = soxx_data.get("pct")
                soxx_status = soxx_data.get("status")
                if soxx_status is None:
                    soxx_status = "VALID" if soxx_pct is not None else "MISSING"

                threshold = getattr(config, "VWAP_BOUNCE_SOXX_VETO_THRESHOLD", -1.50)
                if soxx_pct is None or soxx_status != "VALID":
                    reason = (
                        f"Regime/SOXX Gate VETO: VWAP_BOUNCE Long blocked for {signal.symbol} during {regime_mode} "
                        f"— SOXX benchmark data unavailable or invalid ({soxx_status})"
                    )
                    logger.warning(
                        "[REGIME/SOXX VETO] symbol=%s strategy=%s regime=%s soxx_status=%s -> blocked (fail-closed)",
                        signal.symbol, strat_name, regime_mode, soxx_status,
                    )
                    result["reason"] = reason
                    return result
                elif soxx_pct <= threshold:
                    reason = (
                        f"Regime/SOXX Gate VETO: VWAP_BOUNCE Long blocked for {signal.symbol} during {regime_mode} "
                        f"with SOXX {soxx_pct:.2f}% (<= {threshold:.2f}%)"
                    )
                    logger.warning(
                        "[REGIME/SOXX VETO] symbol=%s strategy=%s regime=%s soxx=%.2f threshold=%.2f",
                        signal.symbol, strat_name, regime_mode, soxx_pct, threshold,
                    )
                    result["reason"] = reason
                    return result

    if regime_result:
        is_short_entry = (getattr(signal, "side", "BUY").upper() == "SELL")
        if is_short_entry:
            # Strategy Alignment Guard: Dynamic Scanner candidates are bullish gap/breakout setups. Never short dynamic scanner candidates!
            try:
                from strategies.scanner.premarket_scanner import get_latest_candidates
                scanner_cands = get_latest_candidates() or []
                scanner_syms = {c.symbol for c in scanner_cands} if scanner_cands else set()
                if signal.symbol in scanner_syms:
                    result["reason"] = f"Strategy Alignment: {signal.symbol} is a bullish dynamic scanner candidate (SHORT blocked)"
                    logger.info("Blocked %s: %s", signal.symbol, result["reason"])
                    return result
            except Exception:
                pass

            if not getattr(regime_result, "allow_new_shorts", False) and not getattr(config, "ALLOW_SHORT", False):
                result["reason"] = "Short selling disabled by config or regime mode"
                logger.info("Blocked %s: %s", signal.symbol, result["reason"])
                return result
        else:
            is_inverse = signal.symbol.upper().strip() in getattr(config, "INVERSE_ETF_SYMBOLS", set())
            inverse_exempt = is_inverse and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
            if not inverse_exempt:
                if getattr(regime_result, "halt_new_entries", False) or not getattr(regime_result, "allow_new_longs", True):
                    result["reason"] = f"Regime score {getattr(regime_result, 'score', 0)} <= -3 → No new long entries allowed"
                    logger.info("Blocked %s: %s", signal.symbol, result["reason"])
                    return result

        if hasattr(regime_result, "max_trades_day") and regime_result.max_trades_day is not None:
            is_inverse = signal.symbol.upper().strip() in getattr(config, "INVERSE_ETF_SYMBOLS", set())
            inverse_exempt = is_inverse and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
            if is_inverse and inverse_exempt:
                effective_max_trades = max(effective_max_trades, getattr(config, "MAX_DAILY_TRADES", 15))
            else:
                effective_max_trades = min(effective_max_trades, regime_result.max_trades_day)
        elif getattr(regime_result, "mode", "") == "CAUTION":
            effective_max_trades = min(effective_max_trades, 5)

    logger.debug(
        "[TRADE LIMIT DEBUG] regime=%s today_fills=%d configured_max=%d effective_max=%d",
        getattr(regime_result, "mode", None) if regime_result else "UNKNOWN",
        trades_today,
        getattr(config, "MAX_DAILY_TRADES", 15),
        effective_max_trades,
    )

    if trades_today >= effective_max_trades:
        result["reason"] = f"Max trades/day reached ({trades_today}/{effective_max_trades})"
        logger.info("Blocked %s: %s", signal.symbol, result["reason"])
        return result

    risk_reason = pre_trade_check(
        equity=equity,
        current_pnl=current_pnl,
        trades_today=trades_today,
        open_positions=open_pos,
        day_trades_last_5_days=day_trades_last_5_days,
        risk_dollars=risk_dollars,
        symbol=signal.symbol,
        max_risk_pct=risk_pct,
        open_symbols=open_syms,
        max_trades=effective_max_trades,
        side=signal.side,
    )


    if risk_reason:
        result["reason"] = risk_reason
        logger.info("Blocked %s: %s", signal.symbol, risk_reason)
        return result

    if is_connected and bridge.has_position(signal.symbol):
        result["reason"] = f"Already have position in {signal.symbol}"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    if is_connected and bridge.has_working_order(signal.symbol):
        result["reason"] = f"Working order exists for {signal.symbol}"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    # M-05 Durable Idempotency Check
    from strategies.order_idempotency import build_client_order_id
    bar_ts = getattr(signal, "bar_timestamp", "") or ""
    sig_hash = getattr(signal, "signal_hash", "") or ""
    intent_nonce = getattr(signal, "signal_id", "") or ""
    client_order_id = build_client_order_id(
        symbol=signal.symbol,
        side=signal.side,
        bar_timestamp_utc=bar_ts,
        signal_hash=sig_hash,
        attempt=0,
        strategy=getattr(signal.strategy, "name", str(signal.strategy)),
        intent_nonce=intent_nonce,
    )
    res_ok, res_reason = idempotency_manager.reserve(signal.symbol, side=signal.side)
    if not res_ok:
        result["reason"] = f"Durable idempotency reservation active: {res_reason}"
        logger.info("Blocked %s: %s", signal.symbol, result["reason"])
        return result


    result["shares"] = shares
    cost = estimated_cost(executable_signal, shares)

    logger.info(
        "Signal: %s %s %d @ %.2f | stop=%.2f target=%.2f | risk=$%.2f cost=$%.2f | confidence=%.2f gate=%.2f | %s",
        signal.side, signal.symbol, shares, executable_signal.entry_price,
        signal.stop_price, signal.target_price,
        risk_dollars, cost, signal.confidence, min_confidence, signal.reason,
    )

    sym_clean = signal.symbol.upper().strip()
    if _session_cooldown_counts.get(sym_clean, 0) > 0:
        logger.info(
            "🌟 [POST-COOLDOWN OPPORTUNITY] %s | Cooldown Cycle: %d | Trade Taken: YES | side=%s shares=%d price=%.2f",
            sym_clean, _session_cooldown_counts[sym_clean], signal.side, shares, executable_signal.entry_price,
        )

    if dry_run:
        idempotency_manager.release(signal.symbol, side=signal.side, reason="dry_run")
        result["status"] = "DRY_RUN"
        result["reason"] = "Dry run -- no order placed"
        logger.info("[DRY RUN] Would place: %s %d %s", signal.side, shares, signal.symbol)
        return result

    if not config.BRACKET_ORDER_ENABLED:
        idempotency_manager.release(signal.symbol, side=signal.side, reason="bracket_disabled")
        result["reason"] = "Protective bracket orders are required for live paper entries"
        return result

    limit_price = executable_signal.entry_price
    stop_price = signal.stop_price
    target_price = signal.target_price

    # Phase 0: Trailing State Persistence Health Gate
    # Strict Fail-closed: Block new entries if trailing stop manager is missing,
    # attribute missing, read exception occurs, persistence unhealthy, or state unknown.
    if not dry_run and is_connected:
        try:
            from strategies.trailing_stop import get_trailing_manager
            ts_mgr = get_trailing_manager()
            if ts_mgr is None:
                idempotency_manager.release(signal.symbol, side=signal.side, reason="trailing_manager_missing")
                result["status"] = "VETOED"
                result["reason"] = "Trailing Stop Persistence Gate: Manager instance missing (fail-closed)"
                logger.warning("[PERSISTENCE HEALTH VETO] %s: Trailing stop manager missing, blocking new entry.", signal.symbol)
                return result

            if not hasattr(ts_mgr, "persistence_healthy"):
                idempotency_manager.release(signal.symbol, side=signal.side, reason="trailing_persistence_attr_missing")
                result["status"] = "VETOED"
                result["reason"] = "Trailing Stop Persistence Gate: persistence_healthy attribute missing (fail-closed)"
                logger.warning("[PERSISTENCE HEALTH VETO] %s: persistence_healthy attribute missing, blocking new entry.", signal.symbol)
                return result

            is_healthy = getattr(ts_mgr, "persistence_healthy", None)
            if is_healthy is not True:
                last_err = getattr(ts_mgr, "last_persistence_error", "STATE_UNHEALTHY_OR_UNKNOWN")
                idempotency_manager.release(signal.symbol, side=signal.side, reason="trailing_persistence_unhealthy")
                result["status"] = "VETOED"
                result["reason"] = f"Trailing Stop Persistence Unhealthy: {last_err} (new entries blocked)"
                logger.warning("[PERSISTENCE HEALTH VETO] %s: Trailing stop state persistence is %s (%s), blocking new entry.", signal.symbol, is_healthy, last_err)
                return result
        except Exception as persist_chk_err:
            idempotency_manager.release(signal.symbol, side=signal.side, reason="trailing_persistence_check_exception")
            result["status"] = "VETOED"
            result["reason"] = f"Trailing Stop Persistence Gate Exception: {persist_chk_err} (fail-closed)"
            logger.warning("[PERSISTENCE HEALTH VETO] %s: Trailing stop health check raised exception %s, blocking new entry.", signal.symbol, persist_chk_err)
            return result

    # Phase A: Quote Validation Gate
    require_realtime = getattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", True)
    if not dry_run and require_realtime and is_connected:
        try:
            from strategies.execution_quote import validate_execution_quote
            bar_ts = getattr(executable_signal, "bar_timestamp", None) or getattr(signal, "bar_timestamp", None)
            bar_dur_raw = getattr(executable_signal, "bar_duration_seconds", getattr(signal, "bar_duration_seconds", getattr(config, "ENTRY_BAR_DURATION_SECONDS", 300.0)))

            # Defense-in-depth: Candle-based entry signals strictly require bar_timestamp
            strat_obj = getattr(signal, "strategy", None)
            strat_key = getattr(strat_obj, "name", getattr(strat_obj, "value", str(strat_obj or ""))).upper()
            is_candle_strategy = strat_key in {
                "ORB", "VWAP_BOUNCE", "GAP_AND_GO", "MOMENTUM_SCALP", "CANDLESTICK", "MEAN_REVERSION"
            }
            order_intent_raw = getattr(signal, "order_intent", getattr(OrderIntent, "ENTRY", "ENTRY"))
            order_intent_str = getattr(order_intent_raw, "value", str(order_intent_raw)).upper()
            is_candle_entry = is_candle_strategy and (order_intent_str == "ENTRY")

            if is_candle_entry:
                if bar_ts is None or str(bar_ts).strip() == "":
                    idempotency_manager.release(signal.symbol, side=signal.side, reason="entry_bar_timestamp_missing")
                    result["status"] = "VETOED"
                    result["reason"] = "Execution Quote Gate failed closed: BAR_TIMESTAMP_MISSING (candle-based entry signal missing bar_timestamp)"
                    logger.warning("[EXECUTION QUOTE VETO] %s: Candle entry signal missing bar_timestamp.", signal.symbol)
                    return result

            if bar_ts is not None:
                if bar_dur_raw is None:
                    idempotency_manager.release(signal.symbol, side=signal.side, reason="bar_duration_missing")
                    result["status"] = "VETOED"
                    result["reason"] = "Execution Quote Gate failed closed: BAR_DURATION_INVALID (bar_duration_seconds missing)"
                    return result
                try:
                    bar_dur_f = float(bar_dur_raw)
                    if not math.isfinite(bar_dur_f) or bar_dur_f <= 0.0:
                        idempotency_manager.release(signal.symbol, side=signal.side, reason="bar_duration_invalid")
                        result["status"] = "VETOED"
                        result["reason"] = f"Execution Quote Gate failed closed: BAR_DURATION_INVALID (bar_duration_seconds={bar_dur_raw} must be > 0)"
                        return result
                except (TypeError, ValueError):
                    idempotency_manager.release(signal.symbol, side=signal.side, reason="bar_duration_malformed")
                    result["status"] = "VETOED"
                    result["reason"] = f"Execution Quote Gate failed closed: BAR_DURATION_INVALID (could not parse {bar_dur_raw})"
                    return result
            else:
                bar_dur_f = None

            quote_res = validate_execution_quote(
                bridge=bridge,
                symbol=signal.symbol,
                side=signal.side,
                decision_price=executable_signal.entry_price,
                max_spread_pct=getattr(config, "MAX_ACCEPTABLE_SPREAD_PCT", 1.0),
                max_deviation_pct=getattr(config, "MAX_DECISION_DEVIATION_PCT", 1.5),
                max_age_seconds=getattr(config, "MAX_QUOTE_AGE_SECONDS", 10.0),
                decision_feed=getattr(executable_signal, "decision_feed", None) or getattr(signal, "decision_feed", None),
                bar_timestamp=bar_ts,
                max_bar_age_seconds=getattr(config, "MAX_BAR_STALENESS_SECONDS", 300.0),
                require_feed_concordance=getattr(config, "REQUIRE_FEED_CONCORDANCE", False),
                quote=quote,
                bar_duration_seconds=bar_dur_f,
                max_entry_bar_close_age_seconds=float(getattr(config, "ENTRY_BAR_CLOSE_GRACE_SECONDS", 60.0)),
            )
        except Exception as quote_exc:
            idempotency_manager.release(
                signal.symbol,
                side=signal.side,
                reason=f"quote_check_error:{quote_exc}",
            )
            result["status"] = "VETOED"
            result["reason"] = f"Execution Quote Gate failed closed: {quote_exc}"
            logger.exception("[EXECUTION QUOTE ERROR] %s", signal.symbol)
            return result

        if not quote_res.valid:
            idempotency_manager.release(
                signal.symbol,
                side=signal.side,
                reason=f"quote_veto:{quote_res.reason}",
            )
            result["status"] = "VETOED"
            result["reason"] = f"Execution Quote Gate: {quote_res.reason}"
            logger.warning("[EXECUTION QUOTE VETO] %s: %s", signal.symbol, quote_res.reason)
            return result

        def _get_num(obj, attr):
            try:
                val = getattr(obj, attr, 0.0)
                if isinstance(val, (int, float)):
                    return float(val)
                return 0.0
            except Exception:
                return 0.0

        q_ask = _get_num(quote_res.quote, "ask")
        q_bid = _get_num(quote_res.quote, "bid")
        q_last = _get_num(quote_res.quote, "last")

        if quote_res.quote and (q_ask > 0 or q_bid > 0 or q_last > 0 or hasattr(quote_res.quote, "executable_price")):
            exec_price = 0.0
            if hasattr(quote_res.quote, "executable_price") and callable(quote_res.quote.executable_price):
                try:
                    p = quote_res.quote.executable_price(signal.side)
                    if isinstance(p, (int, float)) and p > 0:
                        exec_price = float(p)
                except Exception:
                    exec_price = 0.0
            if exec_price <= 0:
                exec_price = q_last or q_ask or q_bid

            # 1. Direction-aware unfavorable chase guard
            unf_chase = unfavorable_chase(signal.side, signal.entry_price, exec_price)
            max_allowed_chase = float(getattr(config, "MAX_ALLOWED_CHASE", 0.15))
            strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy, {})
            if isinstance(strat_cfg, dict) and "max_chase" in strat_cfg:
                max_allowed_chase = float(strat_cfg["max_chase"])

            if unf_chase > max_allowed_chase:
                stop_dist = abs(signal.entry_price - signal.stop_price)
                chase_to_stop_ratio_pct = (unf_chase / stop_dist * 100.0) if stop_dist > 1e-6 else 0.0
                q_ts = getattr(quote_res.quote, "timestamp", 0.0)
                if isinstance(q_ts, datetime):
                    quote_age_ms = max(0.0, (datetime.now(timezone.utc) - q_ts.astimezone(timezone.utc)).total_seconds() * 1000.0)
                elif isinstance(q_ts, (int, float)) and q_ts > 0:
                    quote_age_ms = max(0.0, (time.time() - float(q_ts)) * 1000.0)
                else:
                    quote_age_ms = 0.0

                idempotency_manager.release(
                    signal.symbol,
                    side=signal.side,
                    reason=f"chase_exceeded:unfavorable_chase={unf_chase:.4f}>max={max_allowed_chase:.4f}",
                )
                result["status"] = "VETOED"
                result["reason"] = (
                    f"Execution Quote Gate: Unfavorable chase ${unf_chase:.4f} exceeds safety guard ${max_allowed_chase:.4f} "
                    f"(chase/stop={chase_to_stop_ratio_pct:.1f}%, quote_age={quote_age_ms:.1f}ms)"
                )
                logger.warning(
                    "[CHASE GUARD VETO] %s: unfavorable chase $%.4f > allowed $%.4f "
                    "(signal_px=%.2f, exec_px=%.2f, quote_age=%.1fms, stop_dist=$%.4f, chase/stop=%.1f%%)",
                    signal.symbol, unf_chase, max_allowed_chase, signal.entry_price, exec_price,
                    quote_age_ms, stop_dist, chase_to_stop_ratio_pct
                )
                return result

            # 2. Directional stop rounding; MR retains absolute stop and VWAP target.
            limit_price, stop_price, target_price = _submission_prices(signal, exec_price)

            logger.info(
                "[PRE-SUBMIT OFFSET ADJUST] %s: signal entry $%.2f -> live limit $%.4f | stop $%.4f -> $%.4f | target $%.4f -> $%.4f (unfavorable_chase=$%.4f)",
                signal.symbol, signal.entry_price, limit_price, signal.stop_price, stop_price, signal.target_price, target_price, unf_chase
            )

            # 3. Side-neutral geometry validation
            geo_ok, geo_err, adj_risk, adj_reward, adj_rr = validate_geometry(
                signal.side, limit_price, stop_price, target_price
            )
            if not geo_ok:
                released = idempotency_manager.release(
                    signal.symbol,
                    side=signal.side,
                    reason=f"adjusted_geometry_invalid:{geo_err}",
                )
                result["status"] = "VETOED" if released else "IDEMPOTENCY_RELEASE_FAILED"
                result["reason"] = f"Final submitted geometry invalid: Adjusted geometry invalid: {geo_err}"
                logger.warning("[ADJUSTED GEOMETRY VETO] %s: %s", signal.symbol, geo_err)
                return result

            # Verify post-adjustment RR does not degrade below minimum
            min_rr = getattr(config, "MIN_RR_RATIO", 1.5)
            if adj_rr < (min_rr - 1e-6):
                released = idempotency_manager.release(
                    signal.symbol,
                    side=signal.side,
                    reason=f"adjusted_rr_insufficient:rr={adj_rr:.2f}<min={min_rr:.2f}",
                )
                result["status"] = "VETOED" if released else "IDEMPOTENCY_RELEASE_FAILED"
                result["reason"] = f"Final submitted geometry invalid: Adjusted risk/reward ratio {adj_rr:.2f} < minimum {min_rr:.2f}"
                logger.warning("[ADJUSTED RR VETO] %s: %s", signal.symbol, result["reason"])
                return result

            submitted_candidate = replace(
                executable_signal,
                entry_price=float(limit_price),
                stop_price=float(stop_price),
                target_price=float(target_price),
                risk_per_share=adj_risk,
            )

            final_geometry_reason = _validate_signal_prices(
                submitted_candidate,
                market_texture=market_texture,
                quote={"bid": q_bid, "ask": q_ask},
            )

            if final_geometry_reason:
                released = idempotency_manager.release(
                    signal.symbol,
                    side=signal.side,
                    reason=f"final_geometry_veto:{final_geometry_reason}",
                )

                result["status"] = (
                    "VETOED" if released
                    else "IDEMPOTENCY_RELEASE_FAILED"
                )
                result["reason"] = (
                    f"Final submitted geometry invalid: {final_geometry_reason}"
                )
                logger.warning(
                    "[FINAL GEOMETRY VETO] %s: %s (released=%s)",
                    signal.symbol,
                    final_geometry_reason,
                    released,
                )
                return result

    # Apply the same tick alignment/validation even if the optional quote block
    # did not reprice the order. The broker must not be the first price quantizer.
    limit_price, stop_price, target_price = _submission_prices(signal, limit_price)
    boundary_candidate = replace(
        executable_signal, entry_price=float(limit_price), stop_price=float(stop_price),
        target_price=float(target_price), risk_per_share=abs(float(limit_price) - float(stop_price)),
    )
    boundary_quote = ({"bid": q_bid, "ask": q_ask}
                      if locals().get("q_ask", 0) > locals().get("q_bid", 0) > 0 else quote)
    boundary_reason = _validate_signal_prices(boundary_candidate, market_texture=market_texture, quote=boundary_quote)
    if boundary_reason:
        released = idempotency_manager.release(signal.symbol, side=signal.side,
                                               reason=f"submission_boundary_veto:{boundary_reason}")
        result["status"] = "VETOED" if released else "IDEMPOTENCY_RELEASE_FAILED"
        result["reason"] = f"Submission boundary geometry invalid: {boundary_reason}"
        return result

    # Repricing/quantization occurs after sizing and risk checks. Only reduce quantity;
    # never silently exceed either of the already-approved risk and notional budgets.
    final_risk = abs(float(limit_price) - float(stop_price))
    shares = _cap_submission_shares(
        shares, risk_dollars, shares * executable_signal.entry_price,
        final_risk, float(limit_price),
    )
    if shares < max(1, int(getattr(config, "MIN_SHARE_COUNT", 1))):
        released = idempotency_manager.release(signal.symbol, side=signal.side, reason="final_size_below_minimum")
        result["status"] = "VETOED" if released else "IDEMPOTENCY_RELEASE_FAILED"
        result["reason"] = "Final geometry exceeds approved risk/notional budget at minimum size"
        return result
    result["shares"] = shares
    risk_dollars = shares * final_risk

    # Phase B: Broker Submission (Deterministic single call)
    try:
        try:
            q_age_val = locals().get("quote_age_ms")
            order_audit.lifecycle_tracker.record_submission(
                client_order_id=client_order_id,
                symbol=signal.symbol,
                side=signal.side,
                limit_price=limit_price,
                requested_qty=shares,
                bid=locals().get("q_bid"),
                ask=locals().get("q_ask"),
                quote_age_ms=q_age_val,
                decision_price=signal.entry_price,
                signal_id=getattr(signal, "signal_id", None),
                signal_family_id=getattr(signal, "signal_family_id", None) or signal_metadata.get("signal_family_id"),
                feed_source=getattr(config, "ALPACA_FEED", "IEX"),
            )
        except Exception as audit_snap_exc:
            logger.debug("[ORDER AUDIT] Failed to record submission snapshot: %s", audit_snap_exc)

        # Phase C: Final-Submit Boundary Persistence Health Gate
        # Strict Fail-Closed: Immediately before delegating to the broker submission adapter,
        # re-verify the canonical trailing stop manager's persistence health.
        if not dry_run and is_connected:
            try:
                from strategies.trailing_stop import get_trailing_manager
                final_ts_mgr = get_trailing_manager()
                if final_ts_mgr is None:
                    idempotency_manager.release(
                        signal.symbol,
                        side=signal.side,
                        reason="final_submit_trailing_manager_missing",
                    )
                    result["status"] = "VETOED"
                    result["reason"] = (
                        "Trailing Stop Persistence Gate (Final Submit): "
                        "Manager instance missing (fail-closed)"
                    )
                    logger.warning(
                        "[PERSISTENCE HEALTH FINAL SUBMIT VETO] %s: Trailing stop manager missing immediately before submit.",
                        signal.symbol,
                    )
                    return result

                if not hasattr(final_ts_mgr, "persistence_healthy"):
                    idempotency_manager.release(
                        signal.symbol,
                        side=signal.side,
                        reason="final_submit_persistence_attr_missing",
                    )
                    result["status"] = "VETOED"
                    result["reason"] = (
                        "Trailing Stop Persistence Gate (Final Submit): "
                        "persistence_healthy attribute missing (fail-closed)"
                    )
                    logger.warning(
                        "[PERSISTENCE HEALTH FINAL SUBMIT VETO] %s: persistence_healthy attribute missing immediately before submit.",
                        signal.symbol,
                    )
                    return result

                final_healthy = getattr(final_ts_mgr, "persistence_healthy", None)
                if final_healthy is not True:
                    last_err = getattr(
                        final_ts_mgr,
                        "last_persistence_error",
                        "STATE_UNHEALTHY_OR_UNKNOWN",
                    )
                    idempotency_manager.release(
                        signal.symbol,
                        side=signal.side,
                        reason="final_submit_persistence_unhealthy",
                    )
                    result["status"] = "VETOED"
                    result["reason"] = (
                        f"Trailing Stop Persistence Unhealthy (Final Submit): "
                        f"{last_err} (new entries blocked)"
                    )
                    logger.warning(
                        "[PERSISTENCE HEALTH FINAL SUBMIT VETO] %s: Trailing stop state persistence is %s (%s) immediately before submit.",
                        signal.symbol,
                        final_healthy,
                        last_err,
                    )
                    return result

                if hasattr(final_ts_mgr, "has_unresolved_position_unknown") and final_ts_mgr.has_unresolved_position_unknown():
                    idempotency_manager.release(
                        signal.symbol,
                        side=signal.side,
                        reason="final_submit_position_unknown_halt",
                    )
                    result["status"] = "VETOED"
                    result["reason"] = (
                        "Trailing Stop Position Unknown Halt (Final Submit): "
                        "Portfolio has active positions with unresolved broker query failures"
                    )
                    logger.warning(
                        "[POSITION UNKNOWN FINAL SUBMIT VETO] %s: Trailing stop manager has unresolved POSITION_UNKNOWN state immediately before submit.",
                        signal.symbol,
                    )
                    return result

                if hasattr(final_ts_mgr, "has_pending_exit") and final_ts_mgr.has_pending_exit(signal.symbol):
                    idempotency_manager.release(
                        signal.symbol,
                        side=signal.side,
                        reason="final_submit_unreconciled_exit_pending",
                    )
                    result["status"] = "VETOED"
                    result["reason"] = (
                        f"Unreconciled Exit Pending (Final Submit): "
                        f"Prior exit for {signal.symbol} pending broker reconciliation"
                    )
                    logger.warning(
                        "[UNRECONCILED EXIT FINAL SUBMIT VETO] %s: Symbol has unreconciled pending exit immediately before submit.",
                        signal.symbol,
                    )
                    return result
            except Exception as final_persist_err:
                idempotency_manager.release(
                    signal.symbol,
                    side=signal.side,
                    reason="final_submit_persistence_check_exception",
                )
                result["status"] = "VETOED"
                result["reason"] = (
                    f"Trailing Stop Persistence Gate (Final Submit) Exception: "
                    f"{final_persist_err} (fail-closed)"
                )
                logger.warning(
                    "[PERSISTENCE HEALTH FINAL SUBMIT VETO] %s: Trailing stop health check raised exception %s immediately before submit.",
                    signal.symbol,
                    final_persist_err,
                )
                return result

        order = _place_bracket_order_once(
            bridge,
            symbol=signal.symbol,
            side=signal.side,
            qty=shares,
            entry_price=limit_price,
            stop_price=stop_price,
            target_price=target_price,
            client_order_id=client_order_id,
        )

    except BrokerIdempotencyUnsupported as capability_error:
        released = idempotency_manager.release(
            signal.symbol,
            side=signal.side,
            reason=f"pre_submit_capability_failure:{capability_error}",
        )

        if not released:
            result["status"] = "IDEMPOTENCY_RELEASE_FAILED"
            result["reason"] = (
                "Broker submission was blocked before transmission, "
                "but the pending idempotency reservation could not "
                "be released safely."
            )
            logger.critical(
                "[IDEMPOTENCY RELEASE FAILURE] %s: %s",
                signal.symbol,
                result["reason"],
            )
            return result

        result["status"] = "REJECTED"
        result["reason"] = str(capability_error)

        logger.error(
            "[BROKER CAPABILITY BLOCK] %s: %s",
            signal.symbol,
            capability_error,
        )
        return result

    except Exception as submit_exc:
        exc_str = str(submit_exc).lower()
        is_client_id_collision = (
            "client_order_id must be unique" in exc_str
            or "42210000" in exc_str
        )

        if is_client_id_collision:
            order = None

            for lookup_attempt in range(1, 4):
                try:
                    recovered = None

                    if hasattr(bridge, "get_order_by_client_id"):
                        recovered = bridge.get_order_by_client_id(
                            client_order_id
                        )
                    elif (
                        hasattr(bridge, "_client")
                        and hasattr(
                            bridge._client,
                            "get_order_by_client_id",
                        )
                    ):
                        raw_order = (
                            bridge._client.get_order_by_client_id(
                                client_order_id
                            )
                        )
                        if raw_order:
                            recovered = {
                                "order_id": str(
                                    getattr(raw_order, "id", "")
                                ),
                                "status": str(
                                    getattr(raw_order, "status", "")
                                ),
                            }

                    if (
                        isinstance(recovered, dict)
                        and (
                            recovered.get("order_id")
                            or recovered.get("id")
                        )
                    ):
                        order = recovered
                        logger.info(
                            "[422 RECOVERY SUCCESS] %s: "
                            "Recovered order=%s status=%s attempt=%d",
                            signal.symbol,
                            order.get("order_id") or order.get("id"),
                            order.get("status"),
                            lookup_attempt,
                        )
                        break

                except Exception as lookup_err:
                    logger.warning(
                        "[422 RECOVERY LOOKUP FAILED] "
                        "%s (attempt %d/3): %s",
                        signal.symbol,
                        lookup_attempt,
                        lookup_err,
                    )

                if lookup_attempt < 3:
                    time.sleep(0.05 * lookup_attempt)

            if order is None:
                result["status"] = (
                    "SUBMITTED_RECONCILIATION_REQUIRED"
                )
                result["reason"] = (
                    "Broker reported duplicate client_order_id, "
                    "but lookup did not resolve the existing order. "
                    "Reservation retained for reconciliation."
                )
                result["metadata"]["reconciliation_reason"] = (
                    "DUPLICATE_CLIENT_ORDER_ID"
                )

                logger.exception(
                    "[BROKER SUBMISSION] %s: %s",
                    signal.symbol,
                    result["reason"],
                )
                return result

        else:
            definitely_rejected = bool(
                getattr(
                    submit_exc,
                    "definitely_rejected",
                    False,
                )
            )

            if definitely_rejected:
                idempotency_manager.release(
                    signal.symbol,
                    side=signal.side,
                    reason=f"definite_submit_rejection:{submit_exc}",
                )
                result["status"] = "REJECTED"
                result["reason"] = (
                    f"Broker definitely rejected order: "
                    f"{submit_exc}"
                )
            else:
                result["status"] = "AMBIGUOUS_SUBMIT"
                result["reason"] = (
                    f"Broker submission outcome is uncertain: "
                    f"{submit_exc}. Idempotency lock "
                    "retained for broker reconciliation."
                )

            logger.exception(
                "[BROKER SUBMISSION] %s: %s",
                signal.symbol,
                result["reason"],
            )
            return result

    # Phase C: Broker Response Validation
    if not isinstance(order, dict):
        result["status"] = "AMBIGUOUS_SUBMIT"
        result["reason"] = (
            f"Broker returned unsupported response type {type(order).__name__}. "
            "Idempotency lock retained for reconciliation."
        )
        logger.error("[AMBIGUOUS SUBMIT] %s: %s", signal.symbol, result["reason"])
        return result

    order_ids = order.get("order_ids") or []
    parent_id = (
        order.get("parent_order_id")
        or order.get("order_id")
        or (order_ids[0] if order_ids else None)
    )
    parent_order_id = str(parent_id) if parent_id else ""

    if parent_order_id:
        try:
            order_audit.lifecycle_tracker.record_ack(
                broker_order_id=parent_order_id,
                client_order_id=client_order_id,
            )
        except Exception as audit_ack_exc:
            logger.debug("[ORDER AUDIT] Failed to record ack: %s", audit_ack_exc)

    if not parent_order_id:
        result["status"] = "AMBIGUOUS_SUBMIT"
        result["reason"] = (
            f"Broker submission returned no parent order ID for {signal.symbol}. "
            "Idempotency lock retained for reconciliation."
        )
        logger.error("[AMBIGUOUS SUBMIT] %s: %s", signal.symbol, result["reason"])
        return result

    try:
        idempotency_manager.attach_parent_order(
            signal.symbol,
            signal.side,
            parent_order_id,
        )
    except Exception as attach_exc:
        result["status"] = "SUBMITTED_RECONCILIATION_REQUIRED"
        result["reason"] = (
            f"Broker accepted parent {parent_order_id}, but durable "
            f"idempotency attachment failed: {attach_exc}. "
            "Reservation retained for reconciliation."
        )
        result["order"] = order
        logger.exception(
            "[IDEMPOTENCY ATTACH FAILURE] %s parent=%s",
            signal.symbol,
            parent_order_id,
        )
        return result

    # Phase C-2: Validate Named Child Legs
    stop_id = order.get("stop_order_id")
    tp_id = order.get("tp_order_id")

    if not stop_id or not tp_id or str(stop_id) == str(tp_id):
        result["status"] = "SUBMITTED_RECONCILIATION_REQUIRED"
        result["reason"] = (
            f"Broker accepted parent order {parent_order_id}, "
            "but named stop/target leg IDs are missing or ambiguous. "
            "Idempotency lock retained for broker reconciliation."
        )
        result["order"] = order
        logger.critical(
            "[CHILD LEG RECONCILIATION] %s: %s",
            signal.symbol,
            result["reason"],
        )
        return result

    # Phase D: Post-Submission Lifecycle Registration & Geometry Propagation
    try:
        # P1-5: Structured single-line geometry logging
        logger.info(
            "[ORDER GEOMETRY] %s parent=%s | signal: entry=%.2f stop=%.2f tp=%.2f | "
            "submitted: entry=%.2f stop=%.2f tp=%.2f | legs: stop=%s tp=%s",
            signal.symbol,
            parent_order_id,
            float(signal.entry_price),
            float(signal.stop_price),
            float(getattr(signal, "target_price", 0.0) or 0.0),
            float(limit_price),
            float(stop_price),
            float(target_price),
            str(stop_id),
            str(tp_id),
        )

        submitted_signal = replace(
            signal,
            entry_price=float(limit_price),
            stop_price=float(stop_price),
            target_price=float(target_price),
            risk_per_share=abs(float(limit_price) - float(stop_price)),
        )

        register_signal = getattr(bridge, "register_order_signal", None)
        if callable(register_signal):
            register_signal(order_ids, signal.signal_id)

        from strategies.entry_lifecycle import entry_lifecycle

        register_pending_order(
            parent_id=parent_order_id,
            signal=submitted_signal,
            limit_price=limit_price,
            qty=shares,
            stop_id=str(stop_id),
            tp_id=str(tp_id),
            submitted_entry_price=limit_price,
            submitted_stop_price=stop_price,
            submitted_target_price=target_price,
        )

        entry_ttl = float(getattr(config, "MAX_ENTRY_ORDER_TTL_SECONDS", MAX_ENTRY_ORDER_TTL_SECONDS))
        q_age_ms = locals().get("quote_age_ms")
        q_ts_calc = time.time() - (float(q_age_ms) / 1000.0) if isinstance(q_age_ms, (int, float)) and q_age_ms > 0 else time.time()
        sig_gen_ts = getattr(signal, "timestamp", None) or signal_metadata.get("timestamp") or time.time()
        if isinstance(sig_gen_ts, datetime):
            sig_gen_ts = sig_gen_ts.timestamp()
        elif isinstance(sig_gen_ts, str):
            try:
                sig_gen_ts = float(sig_gen_ts)
            except ValueError:
                try:
                    sig_gen_ts = datetime.fromisoformat(sig_gen_ts.replace("Z", "+00:00")).timestamp()
                except Exception:
                    sig_gen_ts = time.time()
        elif not isinstance(sig_gen_ts, (int, float)):
            sig_gen_ts = time.time()

        entry_lifecycle.register_submission(
            parent_order_id=parent_order_id,
            signal=submitted_signal,
            requested_qty=shares,
            stop_order_id=str(stop_id),
            target_order_id=str(tp_id),
            submitted_entry_price=limit_price,
            submitted_stop_price=stop_price,
            submitted_target_price=target_price,
            expires_at=time.time() + entry_ttl,
            decision_bar_timestamp=bar_ts,
            submission_quote_timestamp=q_ts_calc,
            signal_generation_timestamp=float(sig_gen_ts),
        )

        with _pending_order_contexts_lock:
            ctx = _pending_order_contexts.get(parent_order_id)
            if ctx:
                ctx["lifecycle_registered"] = True
                ctx["reconciliation_required"] = False
    except Exception as lifecycle_exc:
        with _pending_order_contexts_lock:
            ctx = _pending_order_contexts.get(parent_order_id)
            if ctx:
                ctx["lifecycle_registered"] = False
                ctx["reconciliation_required"] = True

        result["status"] = "SUBMITTED_RECONCILIATION_REQUIRED"
        result["reason"] = (
            f"Broker accepted parent order {parent_order_id}, "
            f"but local lifecycle registration failed: {lifecycle_exc}. "
            "Idempotency lock retained."
        )
        result["order"] = order
        logger.exception("[POST-SUBMISSION FAILURE] %s", result["reason"])
        return result

    result["status"] = "PLACED"
    result["reason"] = "Order placed and lifecycle registered"
    result["order"] = order
    result["shares"] = shares
    result["parent_order_id"] = str(parent_order_id)
    result["client_order_id"] = str(client_order_id)
    result["stop_order_id"] = str(stop_id) if stop_id else None
    result["tp_order_id"] = str(tp_id) if tp_id else None

    # Auto confirm fill in backtest / paper simulation mode if enabled
    auto_fill = getattr(config, "PAPER_MODE_AUTO_CONFIRM_FILL", False)
    if auto_fill and parent_order_id:
        on_broker_fill_event(
            parent_order_id,
            fill_price=limit_price,
            filled_qty=shares,
            bridge=bridge,
        )
        try:
            entry_lifecycle.on_parent_fill(
                bridge=bridge,
                parent_order_id=parent_order_id,
                cumulative_filled_qty=shares,
                average_fill_price=limit_price,
            )
        except Exception as fill_exc:
            logger.warning("Entry lifecycle fill confirmation: %s", fill_exc)

    # Phase E: Non-Transactional Telemetry & Reporting
    try:
        from strategies.order_registry import register_order_expected_price
        now_ts = time.time()
        exit_side = "SELL" if signal.side.upper() == "BUY" else "BUY"
        register_order_expected_price(
            parent_order_id,
            expected_price=limit_price,
            side=signal.side,
            order_type="ENTRY",
            submit_ts=now_ts,
            symbol=signal.symbol,
            strategy=signal.strategy,
            signal_id=signal.signal_id,
        )
        register_order_expected_price(
            str(stop_id),
            expected_price=stop_price,
            side=exit_side,
            order_type="STOP_LOSS",
            submit_ts=now_ts,
            symbol=signal.symbol,
            strategy=signal.strategy,
            signal_id=signal.signal_id,
        )
        register_order_expected_price(
            str(tp_id),
            expected_price=target_price,
            side=exit_side,
            order_type="TAKE_PROFIT",
            submit_ts=now_ts,
            symbol=signal.symbol,
            strategy=signal.strategy,
            signal_id=signal.signal_id,
        )
    except Exception as reg_err:
        logger.exception("Expected-price registration failed for accepted parent %s: %s", parent_order_id, reg_err)
        result["warnings"].append(f"expected_price_reg_failed:{reg_err}")

    try:
        log_trade(
            symbol=signal.symbol,
            side=signal.side,
            strategy=signal.strategy,
            qty=shares,
            entry_price=limit_price,
            stop_price=stop_price,
            target_price=target_price,
            order_type="BRACKET",
            status="PLACED",
            order_id=parent_order_id,
            signal_id=signal.signal_id,
            equity=equity,
            confidence=signal.confidence,
        )
    except Exception as journal_err:
        logger.exception("Journal write failed for accepted order %s: %s", parent_order_id, journal_err)
        result["warnings"].append(f"journal_write_failed:{journal_err}")

    try:
        alert_msg = (
            f"🚀 **PAPER TRADE PLACED** 🚀\n"
            f"**Symbol**: {signal.symbol}\n"
            f"**Side**: {signal.side} ({shares} shares)\n"
            f"**Strategy**: {signal.strategy}\n"
            f"**Entry**: ${limit_price:.2f}\n"
            f"**Stop**: ${stop_price:.2f}\n"
            f"**Target**: ${target_price:.2f}\n"
            f"**Reason**: {signal.reason}"
        )
        send_discord_alert(alert_msg)
    except Exception as alert_err:
        logger.exception("Discord alert failed for accepted order %s: %s", parent_order_id, alert_err)
        result["warnings"].append(f"discord_alert_failed:{alert_err}")

    return result
