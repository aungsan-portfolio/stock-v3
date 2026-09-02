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
from errors import BrokerOrderRejected, BrokerSubmissionAmbiguous, BrokerSubmissionError
from strategies.base import TradeSignal
from strategies.intraday_risk import pre_trade_check
from strategies.position_sizer import calculate_shares, estimated_cost, estimated_risk
from strategies.trade_journal import log_fill, log_trade, today_trade_count
from strategies.order_idempotency import idempotency_manager
from strategies.webhook import send_discord_alert

logger = logging.getLogger(__name__)

_session_skip_counts = defaultdict(int)
_session_suppressed = set()


def _get_skip_limit() -> int:
    return getattr(config, "HARD_VETO_SKIP_LIMIT", 5)


def check_session_suppression(symbol: str) -> Tuple[bool, str]:
    """Checks if a symbol has been hard-vetoed >= skip limit times in a session."""
    if not getattr(config, "SKIP_SUPPRESSOR_ENABLED", True):
        return False, ""
    sym = symbol.upper().strip()
    limit = _get_skip_limit()
    if sym in _session_suppressed:
        return True, f"[SESSION SUPPRESSED] {sym}: suppressed this session (skip count >= {limit})"
    return False, ""


_seen_veto_fingerprints = {}


def record_hard_veto_skip(symbol: str, reason: str, bar_timestamp: str = ""):
    """Increments skip counter when noise floor / tight stop hard vetoes occur, deduplicated per bar fingerprint."""
    sym = symbol.upper().strip()
    HARD_VETO_REASONS = ["critically tight", "noise floor", "atr", "veto line"]
    if not any(r.lower() in reason.lower() for r in HARD_VETO_REASONS):
        return

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
        _session_suppressed.add(sym)
        logger.warning("[SESSION SUPPRESS] %s: Suppressed for remainder of session after %d hard vetoes.", sym, count)


def reset_session_skip_state():
    """Resets session skip state for new trading sessions."""
    global _session_skip_counts, _session_suppressed, _seen_veto_fingerprints
    _session_skip_counts = defaultdict(int)
    _session_suppressed = set()
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
    after a broad TypeError.
    """
    place_fn = getattr(bridge, "place_bracket_order", None)
    if not callable(place_fn):
        raise AttributeError("Broker bridge does not implement place_bracket_order")

    kwargs = {
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
    }

    try:
        signature = inspect.signature(place_fn)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot inspect broker submission signature; "
            "submission blocked to prevent duplicate order"
        ) from exc

    supports_client_order_id = (
        "client_order_id" in signature.parameters
        or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
    )

    if supports_client_order_id:
        kwargs["client_order_id"] = client_order_id

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



def _pre_submission_veto(signal: TradeSignal, df: Optional[object] = None, market_texture: Optional[str] = None) -> tuple:
    """
    ATR-based pre-submission veto.
    Hard Veto:  stop_distance < noise_floor * hard_veto_ratio -> block entry
    Soft Warn:  stop_distance < noise_floor                   -> warn, allow entry
    Returns: (should_skip: bool, reason: str)
    """
    stop_distance = abs(signal.entry_price - signal.stop_price)
    noise_floor   = _compute_noise_floor(signal, df)

    # Dynamically select hard veto ratio based on active market texture
    hard_veto_ratio = getattr(config, "NOISE_FLOOR_HARD_VETO_RATIO", 0.75)
    if market_texture and hasattr(config, "MARKET_REGIME_SETTINGS"):
        tex_cfg = config.MARKET_REGIME_SETTINGS.get(market_texture, {})
        if "noise_floor_ratio" in tex_cfg:
            hard_veto_ratio = float(tex_cfg["noise_floor_ratio"])

    hard_veto_line  = noise_floor * hard_veto_ratio

    if stop_distance < (hard_veto_line - 1e-4):
        reason = (
            f"Pre-submission Veto: Stop distance ${stop_distance:.4f} critically tight "
            f"(< {hard_veto_ratio*100:.0f}% of ATR noise floor ${noise_floor:.4f}). "
            f"Hard veto line: ${hard_veto_line:.4f}"
        )
        return True, reason

    if stop_distance < noise_floor:
        logger.debug(
            "[SOFT-WARN] %s: Stop $%.4f slightly below noise floor $%.4f "
            "(%.1f%% of floor) -- allowing entry",
            signal.symbol,
            stop_distance,
            noise_floor,
            (stop_distance / noise_floor) * 100 if noise_floor > 0 else 100
        )

    return False, ""



def _validate_signal_prices(signal: TradeSignal, market_texture: Optional[str] = None, df_history: Optional[object] = None) -> str:
    """Return a rejection reason when entry/stop/target geometry is invalid."""
    side = signal.side.upper()
    if signal.entry_price <= 0:
        return "Invalid entry price"
    if signal.stop_price <= 0 or signal.target_price <= 0:
        return "Invalid stop/target price"
    if signal.risk_per_share <= 0:
        return "Invalid risk/share"

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

    should_skip, veto_reason = _pre_submission_veto(signal, df_history, market_texture)
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

    from dataclasses import replace
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

        if requested_qty > 0:
            new_cumulative_qty = min(new_cumulative_qty, requested_qty)

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

    price_reason = _validate_signal_prices(signal, market_texture=market_texture)
    if price_reason:
        result["reason"] = price_reason
        record_hard_veto_skip(signal.symbol, price_reason, getattr(signal, "bar_timestamp", "") or "")
        logger.info("Skipped %s: %s", signal.symbol, price_reason)
        return result

    executable_signal = _executable_signal(signal, entry_limit_price)
    executable_price_reason = _validate_signal_prices(executable_signal, market_texture=market_texture)
    if executable_price_reason:
        result["reason"] = f"Invalid executable limit: {executable_price_reason}"
        record_hard_veto_skip(signal.symbol, executable_price_reason, getattr(signal, "bar_timestamp", "") or "")
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
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
                    if last_close < last_open:
                        guard_reason = (
                            f"SPY Trend Guard: {guard_sym} 5m candle bearish "
                            f"(Close ${last_close:.2f} < Open ${last_open:.2f}). "
                            f"Blocking LONG entry."
                        )
                        logger.info("[SPY TREND GUARD] Blocked %s: %s", signal.symbol, guard_reason)
                        result["reason"] = guard_reason
                        return result
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
        symbol_entries = sum(
            1 for t in todays
            if isinstance(t, dict) and t.get("symbol") == signal.symbol
        )
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
    client_order_id = build_client_order_id(
        symbol=signal.symbol,
        side=signal.side,
        bar_timestamp_utc=bar_ts,
        signal_hash=sig_hash,
        attempt=0,
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

        if quote_res.quote and quote_res.quote.last > 0:
            offset_stop = executable_signal.entry_price - signal.stop_price
            offset_target = signal.target_price - executable_signal.entry_price
            limit_price = quote_res.quote.last
            stop_price = round(limit_price - offset_stop, 2)
            target_price = round(limit_price + offset_target, 2)

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
    except Exception as submit_exc:
        definitely_rejected = bool(getattr(submit_exc, "definitely_rejected", False))
        if definitely_rejected:
            idempotency_manager.release(
                signal.symbol,
                side=signal.side,
                reason=f"definite_submit_rejection:{submit_exc}",
            )
            result["status"] = "REJECTED"
            result["reason"] = f"Broker definitely rejected order: {submit_exc}"
        else:
            result["status"] = "AMBIGUOUS_SUBMIT"
            result["reason"] = (
                f"Broker submission outcome is uncertain: {submit_exc}. "
                "Idempotency lock retained for broker reconciliation."
            )
        logger.exception("[BROKER SUBMISSION] %s: %s", signal.symbol, result["reason"])
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

        from dataclasses import replace
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
        )
        register_order_expected_price(
            str(stop_id),
            expected_price=stop_price,
            side=exit_side,
            order_type="STOP_LOSS",
            submit_ts=now_ts,
            symbol=signal.symbol,
        )
        register_order_expected_price(
            str(tp_id),
            expected_price=target_price,
            side=exit_side,
            order_type="TAKE_PROFIT",
            submit_ts=now_ts,
            symbol=signal.symbol,
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
