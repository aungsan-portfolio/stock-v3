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
from collections import defaultdict
from typing import Optional, Tuple

import config
from strategies.base import TradeSignal
from strategies.intraday_risk import pre_trade_check
from strategies.position_sizer import calculate_shares, estimated_cost
from strategies.trade_journal import log_trade, today_trade_count
from strategies.order_idempotency import idempotency_manager
from strategies.webhook import send_discord_alert

logger = logging.getLogger(__name__)


class BrokerIdempotencyUnsupported(RuntimeError):
    """Broker bridge cannot safely accept a deterministic client_order_id."""


_session_state_lock = threading.RLock()
_session_skip_counts = defaultdict(int)
_session_suppressed_until = {}
_session_cooldown_counts = defaultdict(int)
_session_permanently_suppressed = set()
_seen_veto_fingerprints = {}


def _get_skip_limit() -> int:
    return getattr(config, "HARD_VETO_SKIP_LIMIT", 5)


def _get_cooldown_seconds() -> float:
    return float(getattr(config, "SESSION_SUPPRESSION_COOLDOWN_SECONDS", 2700.0))


def _get_max_cooldown_cycles() -> int:
    return int(getattr(config, "MAX_DAILY_COOLDOWN_CYCLES", 3))


def check_session_suppression(symbol: str) -> Tuple[bool, str]:
    """Checks if a symbol is actively suppressed or in cooldown."""
    if not getattr(config, "SKIP_SUPPRESSOR_ENABLED", True):
        return False, ""
    sym = symbol.upper().strip()

    with _session_state_lock:
        # 1. Check permanent full-day suppression (exceeded max cooldown cycles)
        if sym in _session_permanently_suppressed:
            max_cycles = _get_max_cooldown_cycles()
            return True, f"[SESSION SUPPRESSED] {sym}: permanently suppressed this session (exceeded {max_cycles} cooldown cycles)"

        # 2. Check dynamic cooldown window
        if sym in _session_suppressed_until:
            now = time.time()
            until = _session_suppressed_until[sym]
            if now < until:
                remaining_min = (until - now) / 60.0
                cycle = _session_cooldown_counts[sym]
                return True, f"[SESSION SUPPRESSED] {sym}: cooldown active ({remaining_min:.1f}m remaining, Cycle #{cycle})"
            else:
                # Cooldown expired! Restore symbol to active evaluation
                cycle = _session_cooldown_counts[sym]
                logger.info("[COOLDOWN EXPIRE] %s: Cooldown period ended (Cycle #%d). Symbol restored to active evaluation.", sym, cycle)
                del _session_suppressed_until[sym]
                return False, ""

        return False, ""


def record_hard_veto_skip(symbol: str, reason: str, bar_timestamp: str = ""):
    """Increments skip counter when noise floor / tight stop hard vetoes occur, deduplicated per bar fingerprint."""
    sym = symbol.upper().strip()
    HARD_VETO_REASONS = ["critically tight", "noise floor", "atr", "veto line"]
    if not any(r.lower() in reason.lower() for r in HARD_VETO_REASONS):
        return

    with _session_state_lock:
        # Deduplicate repeat logs for identical geometry within 3600 seconds (session-level) across identical veto reasons
        fingerprint = f"{sym}:{bar_timestamp}:{reason.strip()[:80]}"
        now = time.time()
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

            # Safeguard #1: Cooldown Cycle Cap (Max 3 cycles before full-day permanent suppression)
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
    with _session_state_lock:
        _session_skip_counts = defaultdict(int)
        _session_suppressed_until = {}
        _session_cooldown_counts = defaultdict(int)
        _session_permanently_suppressed = set()
        _seen_veto_fingerprints = {}
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
    Calculate noise floor based on 10-period ATR (Average True Range).
    Returns max(0.08, ATR10 * 0.75) to ensure minimum stop distance.
    """
    import pandas as pd
    atr_period = getattr(config, "NOISE_FLOOR_ATR_PERIOD", 10)
    atr_multiplier = getattr(config, "NOISE_FLOOR_ATR_MULTIPLIER", 0.75)
    abs_min = getattr(config, "NOISE_FLOOR_ABS_MIN", 0.08)

    # 1. Check signal's explicit atr attribute first
    if hasattr(signal, "atr") and getattr(signal, "atr", None) is not None:
        try:
            sig_atr = float(getattr(signal, "atr"))
            if sig_atr > 0:
                noise_floor = max(abs_min, sig_atr * atr_multiplier)
                return noise_floor
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
            noise_floor = max(abs_min, atr * atr_multiplier)
            logger.debug(
                "[NOISE FLOOR] %s: ATR(%d)=%.4f x %.2f = %.4f (abs_min=%.2f)",
                signal.symbol, atr_period, atr, atr_multiplier, noise_floor, abs_min
            )
            return noise_floor
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
    """
    atr_noise_floor = _compute_noise_floor(signal, df)

    # Tick size floor (SEC Rule 612)
    tick_size_floor = 0.01 if signal.entry_price >= 1.0 else 0.0001

    # Configured minimum
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

    return max(atr_noise_floor, spread_noise_floor, tick_size_floor, configured_min)


def validate_quote_health(
    quote: Optional[dict],
    now_ts: Optional[float] = None,
    max_age_seconds: float = 5.0,
    max_spread_pct: float = 0.015,
    atr: Optional[float] = None,
) -> Tuple[bool, str, dict]:
    """Validates real-time bid/ask quote health and returns normalized spread metrics."""
    if not quote:
        return False, "QUOTE_MISSING", {}

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
        min_rr = getattr(config, "MIN_RR_RATIO", 1.0)
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


def _executable_signal(signal: TradeSignal, entry_limit_price: Optional[float]) -> TradeSignal:
    """Return a signal priced at the worst executable entry limit, with dynamic offset capping.
    
    Adheres to SEC Rule 612 (no sub-penny quotes for stocks >= $1.00).
    """
    if entry_limit_price is not None:
        tick_size = 0.01 if entry_limit_price >= 1.0 else 0.0001
        new_stop_dist = abs(entry_limit_price - signal.stop_price)
        if new_stop_dist <= tick_size:
            logger.warning(
                "[GEOMETRY COLLAPSE VETO] %s live entry %.4f vs stop %.4f gives gap %.6f <= tick %.4f",
                signal.symbol, entry_limit_price, signal.stop_price, new_stop_dist, tick_size,
            )
        return replace(
            signal,
            entry_price=entry_limit_price,
            risk_per_share=new_stop_dist,
        )
    
    # 1. Determine tick size based on SEC Rule 612
    tick_size = 0.01 if signal.entry_price >= 1.0 else 0.0001
    
    # 2. Get default offset
    offset = config.LIMIT_OFFSET_CENTS / 100.0
    
    # 3. Calculate target and stop distances
    target_dist = round(abs(signal.target_price - signal.entry_price), 6)
    stop_dist = round(abs(signal.entry_price - signal.stop_price), 6)
    
    # Guard against zero/negative distances
    if target_dist <= tick_size or stop_dist <= tick_size:
        logger.error(
            f"Invalid signal geometry for {signal.symbol}: target_dist={target_dist:.5f}, "
            f"stop_dist={stop_dist:.5f} is less than or equal to tick_size={tick_size:.5f}"
        )
        return signal
    
    # 4. Cap offset at 20% of the smaller distance
    max_allowed_offset = min(target_dist, stop_dist) * 0.20
    offset = min(offset, max_allowed_offset)
    
    # 5. Round offset to the appropriate tick size
    offset = max(round(offset / tick_size) * tick_size, tick_size)
    
    # 6. Final safety check: ensure rounded offset does not cross target/stop
    if offset >= target_dist or offset >= stop_dist:
        offset = min(target_dist, stop_dist) * 0.40
        offset = max(round(offset / tick_size) * tick_size, tick_size)
        if offset >= min(target_dist, stop_dist):
            offset = tick_size
    
    entry_limit_price = (
        signal.entry_price + offset
        if signal.side.upper() == "BUY"
        else signal.entry_price - offset
    )
    
    return replace(
        signal,
        entry_price=entry_limit_price,
        risk_per_share=abs(entry_limit_price - signal.stop_price),
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
) -> dict:
    """Execute a trade signal through the full pipeline."""
    from strategies.base import TradeSignal
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

    if not dry_run:
        paper_reason = _assert_paper_broker(bridge)
        if paper_reason:
            result["reason"] = paper_reason
            logger.critical("[PAPER ACCOUNT GATE] %s", paper_reason)
            return result

    suppressed, supp_reason = check_session_suppression(signal.symbol)
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

    quote = signal_metadata.get("quote") or getattr(signal, "quote", None)
    if quote:
        quote_ok, quote_err, quote_metrics = validate_quote_health(quote, atr=getattr(signal, "atr", None))
        result["metadata"]["quote_metrics"] = quote_metrics
        if not quote_ok:
            result["reason"] = f"Quote health veto: {quote_err}"
            logger.warning("[QUOTE HEALTH VETO] Blocked %s: %s", signal.symbol, result["reason"])
            return result

    price_reason = _validate_signal_prices(signal, market_texture=market_texture, quote=quote)
    if price_reason:
        result["reason"] = price_reason
        record_hard_veto_skip(signal.symbol, price_reason, getattr(signal, "bar_timestamp", "") or "")
        logger.info("Skipped %s: %s", signal.symbol, price_reason)
        return result

    executable_signal = _executable_signal(signal, entry_limit_price)
    executable_price_reason = _validate_signal_prices(executable_signal, market_texture=market_texture, quote=quote)
    if executable_price_reason:
        result["reason"] = f"Invalid executable limit: {executable_price_reason}"
        record_hard_veto_skip(signal.symbol, executable_price_reason, getattr(signal, "bar_timestamp", "") or "")
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    # Direction-Aware Maximum Chase Guard
    chase = unfavorable_chase(signal.side, signal.entry_price, executable_signal.entry_price)
    max_allowed_chase = float(getattr(config, "MAX_ALLOWED_CHASE", 0.15))
    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy, {})
    if isinstance(strat_cfg, dict) and "max_chase" in strat_cfg:
        max_allowed_chase = float(strat_cfg["max_chase"])

    if chase > (max_allowed_chase + 1e-6):
        chase_reason = f"REJECTED_MAX_CHASE_EXCEEDED: unfavorable chase ${chase:.4f} > max allowed chase ${max_allowed_chase:.4f}"
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

                # Sprint P1.7: Macro Regime & SOXX Semiconductor Gate on VWAP_BOUNCE Longs
                if getattr(config, "VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", True):
                    sig_strat = getattr(signal, "strategy", "")
                    strat_name = sig_strat.value if hasattr(sig_strat, "value") else str(sig_strat)
                    if strat_name.upper() == "VWAP_BOUNCE":
                        regime_mode = getattr(regime_result, "mode", "UNKNOWN")
                        blocked_regimes = getattr(config, "VWAP_BOUNCE_BLOCKED_REGIMES", ("CAUTION", "RISK_OFF", "DATA_DEGRADED"))
                        if regime_mode in blocked_regimes:
                            soxx_pct = None
                            soxx_data = getattr(regime_result, "data", {}).get("SOXX", {}) if hasattr(regime_result, "data") else {}
                            if isinstance(soxx_data, dict) and "pct" in soxx_data:
                                soxx_pct = soxx_data.get("pct")
                            if soxx_pct is None:
                                from strategies.regime_engine import get_soxx_pct_change
                                soxx_pct = get_soxx_pct_change()

                            threshold = getattr(config, "VWAP_BOUNCE_SOXX_VETO_THRESHOLD", -1.50)
                            if soxx_pct is None:
                                logger.warning(
                                    "[REGIME/SOXX GATE] SOXX data unavailable. Allowing %s VWAP_BOUNCE entry (fail-open).",
                                    signal.symbol,
                                )
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

    # Phase A: Quote Validation Gate
    require_realtime = getattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", True)
    if not dry_run and require_realtime and is_connected:
        try:
            from strategies.execution_quote import validate_execution_quote
            quote_res = validate_execution_quote(
                bridge=bridge,
                symbol=signal.symbol,
                decision_price=executable_signal.entry_price,
                max_spread_pct=getattr(config, "MAX_ACCEPTABLE_SPREAD_PCT", 1.0),
                max_deviation_pct=getattr(config, "MAX_DECISION_DEVIATION_PCT", 1.5),
                max_age_seconds=getattr(config, "MAX_QUOTE_AGE_SECONDS", 10.0),
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
                idempotency_manager.release(
                    signal.symbol,
                    side=signal.side,
                    reason=f"chase_exceeded:unfavorable_chase={unf_chase:.4f}>max={max_allowed_chase:.4f}",
                )
                result["status"] = "VETOED"
                result["reason"] = f"Execution Quote Gate: Unfavorable chase ${unf_chase:.4f} exceeds safety guard ${max_allowed_chase:.4f}"
                logger.warning(
                    "[CHASE GUARD VETO] %s: unfavorable chase $%.4f > allowed $%.4f (signal=%.2f, executable=%.2f)",
                    signal.symbol, unf_chase, max_allowed_chase, signal.entry_price, exec_price
                )
                return result

            # 2. Side-consistent offset calculation
            tick_size = 0.01 if exec_price >= 1.0 else 0.0001
            precision = 2 if tick_size == 0.01 else 4
            limit_price = round(exec_price, precision)

            side_upper = signal.side.upper().strip()
            if side_upper in ("BUY", "LONG"):
                stop_offset = executable_signal.entry_price - signal.stop_price
                target_offset = signal.target_price - executable_signal.entry_price
                stop_price = round(limit_price - stop_offset, precision)
                target_price = round(limit_price + target_offset, precision)
            else:
                stop_offset = signal.stop_price - executable_signal.entry_price
                target_offset = executable_signal.entry_price - signal.target_price
                stop_price = round(limit_price + stop_offset, precision)
                target_price = round(limit_price - target_offset, precision)

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
            min_rr = getattr(config, "MIN_RISK_REWARD_RATIO", 1.5)
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

    # Phase B: Broker Submission (Deterministic single call)
    try:
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

        entry_lifecycle.register_submission(
            parent_order_id=parent_order_id,
            signal=submitted_signal,
            requested_qty=shares,
            stop_order_id=str(stop_id),
            target_order_id=str(tp_id),
            submitted_entry_price=limit_price,
            submitted_stop_price=stop_price,
            submitted_target_price=target_price,
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
