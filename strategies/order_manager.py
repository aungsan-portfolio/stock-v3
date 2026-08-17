"""
order_manager.py -- Orchestrates signal -> risk check -> position size -> order.

The single entry point for executing a TradeSignal through the full pipeline.
Always paper-trading only.
"""
from dataclasses import replace
import logging
import math
from typing import Optional, Tuple
import time

import config
from strategies.base import TradeSignal
from strategies.intraday_risk import pre_trade_check
from strategies.position_sizer import calculate_shares, estimated_cost, estimated_risk
from strategies.trade_journal import log_fill, log_trade, today_trade_count
from strategies.webhook import send_discord_alert

logger = logging.getLogger(__name__)

from collections import defaultdict

_recent_orders = {}
IDEMPOTENCY_TTL = 60  # seconds

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
    fingerprint = f"{sym}:{reason.strip()[:80]}"
    now = time.time()
    last_seen = _seen_veto_fingerprints.get(fingerprint, 0)
    
    if now - last_seen < 3600:
        count = _session_skip_counts[sym]
        limit = _get_skip_limit()
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
    _session_suppressed = set()
    logger.info("[ORDER MANAGER] Session skip state reset.")


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

    if stop_distance < hard_veto_line:
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

    min_rr = getattr(config, "MIN_RR_RATIO", 1.0)
    rr_ratio = target_dist / stop_dist if stop_dist > 0 else 0.0
    if rr_ratio < min_rr:
        return f"Risk:Reward ratio {rr_ratio:.2f} is below minimum threshold {min_rr:.2f}"

    return ""


def _executable_signal(signal: TradeSignal, entry_limit_price: Optional[float]) -> TradeSignal:
    """Return a signal priced at the worst executable entry limit, with dynamic offset capping.
    
    Adheres to SEC Rule 612 (no sub-penny quotes for stocks >= $1.00).
    """
    if entry_limit_price is not None:
        return replace(
            signal,
            entry_price=entry_limit_price,
            risk_per_share=abs(entry_limit_price - signal.stop_price),
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
    return 0.63


_pending_order_contexts = {}


def register_pending_order(parent_id: str, signal: TradeSignal, limit_price: float, qty: float, stop_id: Optional[str] = None, tp_id: Optional[str] = None):
    """Register pending order before fill confirmation."""
    if not parent_id:
        return
    _pending_order_contexts[parent_id] = {
        "signal": signal,
        "limit_price": limit_price,
        "qty": qty,
        "stop_id": stop_id,
        "tp_id": tp_id,
        "submitted_at": time.time(),
    }


def on_broker_fill_event(parent_id: str, fill_price: float, filled_qty: float, bridge: Optional[object] = None):
    """Callback invoked when broker confirms execution fill for a pending order."""
    ctx = _pending_order_contexts.pop(parent_id, None)
    if not ctx:
        logger.warning("Fill received for unknown/unregistered order ID %s", parent_id)
        return

    signal = ctx["signal"]
    stop_id = ctx.get("stop_id")
    tp_id = ctx.get("tp_id")

    try:
        from strategies.trailing_stop import manager
        manager.initialize_position(
            signal=signal,
            fill_price=fill_price,
            qty=filled_qty,
            stop_order_id=stop_id,
            bridge=bridge,
            tp_order_id=tp_id,
        )
        logger.info(
            "[FILL CONFIRMED] Initialized position state for %s: fill=$%.2f, qty=%d",
            signal.symbol, fill_price, filled_qty
        )
    except Exception as e:
        logger.error("Failed to initialize position state on fill event for %s: %s", signal.symbol, e)



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
    if dry_run is None:
        dry_run = config.SCHEDULER_DRY_RUN_DEFAULT

    result = {
        "symbol": signal.symbol,
        "strategy": signal.strategy,
        "side": signal.side,
        "status": "REJECTED",
        "reason": "",
        "shares": 0,
        "order": None,
    }
    cache_key = f"{signal.symbol.upper().strip()}_{signal.side.upper().strip()}"

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

    price_reason = _validate_signal_prices(signal, market_texture=market_texture)
    if price_reason:
        result["reason"] = price_reason
        record_hard_veto_skip(signal.symbol, price_reason)
        logger.info("Skipped %s: %s", signal.symbol, price_reason)
        return result

    allow_short = getattr(config, "ALLOW_SHORT", False)
    if signal.side.upper() == "SELL" and not allow_short:
        result["reason"] = "Short entries are disabled"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    executable_signal = _executable_signal(signal, entry_limit_price)
    executable_price_reason = _validate_signal_prices(executable_signal, market_texture=market_texture)
    if executable_price_reason:
        result["reason"] = f"Invalid executable limit: {executable_price_reason}"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result


    min_confidence = _min_confidence_for(signal, regime_result=regime_result)
    if regime_result and getattr(regime_result, "confidence_boost", 0.0) > 0:
        min_confidence += float(regime_result.confidence_boost)

    if signal.confidence < min_confidence:
        result["reason"] = f"Confidence {signal.confidence:.2f} < {min_confidence:.2f} ({signal.strategy})"
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
    if signal.side.upper() == "BUY" and not dry_run and getattr(config, "SPY_TREND_GUARD_ENABLED", True):
        try:
            from strategies.intraday_data import fetch_intraday
            guard_sym = getattr(config, "SPY_TREND_GUARD_SYMBOL", "SPY")
            spy_df = fetch_intraday(guard_sym)
            if spy_df is not None and not spy_df.empty and len(spy_df) >= 2:
                last_close = float(spy_df["close"].iloc[-1])
                last_open  = float(spy_df["open"].iloc[-1])
                if last_close < last_open:
                    guard_reason = (
                        f"SPY Trend Guard: {guard_sym} 5m candle bearish "
                        f"(Close ${last_close:.2f} < Open ${last_open:.2f}). "
                        f"Blocking LONG entry."
                    )
                    logger.info("[SPY TREND GUARD] Blocked %s: %s", signal.symbol, guard_reason)
                    result["reason"] = guard_reason
                    return result
        except Exception as spy_err:
            # Fail-open: do not block trade if SPY data unavailable
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
    if regime_result and getattr(regime_result, "position_scale", 1.0) < 1.0:
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

    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy)
    risk_pct = strat_cfg.max_risk_pct if strat_cfg else 1.0
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
            if getattr(regime_result, "halt_new_entries", False) or not getattr(regime_result, "allow_new_longs", True):
                result["reason"] = f"Regime score {getattr(regime_result, 'score', 0)} <= -3 → No new long entries allowed"
                logger.info("Blocked %s: %s", signal.symbol, result["reason"])
                return result

        if hasattr(regime_result, "max_trades_day") and regime_result.max_trades_day is not None:
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
    from strategies.order_idempotency import build_client_order_id, idempotency_manager
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
        _recent_orders[cache_key] = time.time()
        return result

    if not config.BRACKET_ORDER_ENABLED:
        idempotency_manager.release(signal.symbol, side=signal.side, reason="bracket_disabled")
        result["reason"] = "Protective bracket orders are required for live paper entries"
        return result

    try:
        limit_price = executable_signal.entry_price
        stop_price = signal.stop_price
        target_price = signal.target_price

        # M-01 Fresh Execution Quote Gate
        require_realtime = getattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", True)
        if not dry_run and require_realtime and is_connected:
            from strategies.execution_quote import validate_execution_quote
            quote_res = validate_execution_quote(
                bridge=bridge,
                symbol=signal.symbol,
                decision_price=executable_signal.entry_price,
                max_spread_pct=getattr(config, "MAX_ACCEPTABLE_SPREAD_PCT", 1.0),
                max_deviation_pct=getattr(config, "MAX_DECISION_DEVIATION_PCT", 1.5),
                max_age_seconds=getattr(config, "MAX_QUOTE_AGE_SECONDS", 10.0),
            )
            if not quote_res.valid:
                idempotency_manager.release(signal.symbol, side=signal.side, reason=f"quote_veto:{quote_res.reason}")
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
        try:
            order = bridge.place_bracket_order(
                symbol=signal.symbol,
                side=signal.side,
                qty=shares,
                entry_price=limit_price,
                stop_price=stop_price,
                target_price=target_price,
                client_order_id=client_order_id,
            )
        except TypeError:
            order = bridge.place_bracket_order(
                symbol=signal.symbol,
                side=signal.side,
                qty=shares,
                entry_price=limit_price,
                stop_price=stop_price,
                target_price=target_price,
            )
        order_ids = order.get("order_ids") or []
        register_signal = getattr(bridge, "register_order_signal", None)
        if callable(register_signal):
            register_signal(order_ids, signal.signal_id)

        parent_id = order_ids[0] if order_ids else order.get("order_id")
        parent_order_id = str(parent_id) if parent_id else ""

        # Attach parent order ID to durable idempotency manager
        if parent_order_id:
            idempotency_manager.attach_parent_order(signal.symbol, signal.side, parent_order_id)
        else:
            # H-2: Ambiguous submit - Keep idempotency lock and return AMBIGUOUS_SUBMIT for broker reconciliation
            result["status"] = "AMBIGUOUS_SUBMIT"
            result["reason"] = f"Broker submission returned no parent_order_id for {signal.symbol} -- keeping idempotency lock for broker reconciliation"
            logger.error("[AMBIGUOUS SUBMIT] %s: %s", signal.symbol, result["reason"])
            return result

        # Register expected prices for signed slippage and execution latency tracking
        try:
            from strategies.order_registry import register_order_expected_price
            now_ts = time.time()
            parent_id = order_ids[0] if order_ids else order.get("order_id")
            stop_id = order.get("stop_order_id") or (order_ids[2] if len(order_ids) >= 3 else (order_ids[1] if len(order_ids) >= 2 else None))
            tp_id = order.get("tp_order_id") or (order_ids[1] if len(order_ids) >= 2 else None)
            exit_side = "SELL" if signal.side.upper() == "BUY" else "BUY"

            if parent_id:
                register_order_expected_price(parent_id, expected_price=limit_price, side=signal.side, order_type="ENTRY", submit_ts=now_ts, symbol=signal.symbol)
            if stop_id:
                register_order_expected_price(stop_id, expected_price=stop_price, side=exit_side, order_type="STOP_LOSS", submit_ts=now_ts, symbol=signal.symbol)
            if tp_id and tp_id != stop_id:
                register_order_expected_price(tp_id, expected_price=target_price, side=exit_side, order_type="TAKE_PROFIT", submit_ts=now_ts, symbol=signal.symbol)
        except Exception as reg_err:
            logger.debug(f"Failed to register order expected price: {reg_err}")

        parent_order_id = str(parent_id) if parent_id else ""
        from strategies.entry_lifecycle import entry_lifecycle
        try:
            register_pending_order(
                parent_id=parent_order_id,
                signal=signal,
                limit_price=limit_price,
                qty=shares,
                stop_id=str(stop_id) if stop_id else None,
                tp_id=str(tp_id) if tp_id else None,
            )
            if stop_id and tp_id and stop_id != tp_id:
                entry_lifecycle.register_submission(
                    parent_order_id=parent_order_id,
                    signal=signal,
                    requested_qty=shares,
                    stop_order_id=str(stop_id),
                    target_order_id=str(tp_id),
                )
            result["status"] = "PLACED"
            result["reason"] = "Order placed"
            result["order"] = order
        except Exception as reg_exc:
            # H-3: Lifecycle registration failure must be fail-closed (AMBIGUOUS_LIFECYCLE)
            err_msg = f"Lifecycle registration failed: {reg_exc}"
            logger.error("[AMBIGUOUS LIFECYCLE] %s: %s", signal.symbol, err_msg)
            result["status"] = "AMBIGUOUS_LIFECYCLE"
            result["reason"] = err_msg
            return result

        # C-02: Auto confirm fill default is False in production paper mode
        auto_fill = getattr(config, "PAPER_MODE_AUTO_CONFIRM_FILL", False)
        if auto_fill and parent_order_id:
            on_broker_fill_event(parent_order_id, fill_price=limit_price, filled_qty=shares, bridge=bridge)
            try:
                if stop_id and tp_id and stop_id != tp_id:
                    entry_lifecycle.on_parent_fill(
                        bridge=bridge,
                        parent_order_id=parent_order_id,
                        cumulative_filled_qty=shares,
                        average_fill_price=limit_price,
                    )
            except Exception as fill_exc:
                logger.warning("Entry lifecycle fill confirmation: %s", fill_exc)

        # Update idempotency cache
        _recent_orders[cache_key] = time.time()

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
        return result

    except Exception as exc:
        idempotency_manager.release(signal.symbol, side=signal.side, reason=f"submit_exception:{exc}")
        result["reason"] = f"Order failed: {exc}"
        logger.error("Order failed for %s: %s", signal.symbol, exc)

    result["metadata"] = {
        "raw_confidence": signal.metadata.get("raw_confidence"),
        "correlation_penalty": signal.metadata.get("correlation_penalty"),
        "max_correlation": signal.metadata.get("max_correlation"),
        "overlap_symbols": signal.metadata.get("portfolio_overlap_symbols"),
        "trailing_stop_init": getattr(signal, "atr", 0) > 0,
    }

    return result
