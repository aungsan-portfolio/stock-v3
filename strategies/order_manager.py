"""
order_manager.py -- Orchestrates signal -> risk check -> position size -> order.

The single entry point for executing a TradeSignal through the full pipeline.
Always paper-trading only.
"""
from dataclasses import replace
import logging
import math
from typing import Optional
import time

import config
from strategies.base import TradeSignal
from strategies.intraday_risk import pre_trade_check
from strategies.position_sizer import calculate_shares, estimated_cost, estimated_risk
from strategies.trade_journal import log_fill, log_trade, today_trade_count
from strategies.webhook import send_discord_alert

logger = logging.getLogger(__name__)

_recent_orders = {}
IDEMPOTENCY_TTL = 60  # seconds


def _compute_noise_floor(signal: TradeSignal, df: Optional[object] = None) -> float:
    """
    Calculate noise floor based on 10-period ATR (Average True Range).
    Returns max(0.08, ATR10 * 0.75) to ensure minimum stop distance.
    """
    import pandas as pd
    atr_period = getattr(config, "NOISE_FLOOR_ATR_PERIOD", 10)
    atr_multiplier = getattr(config, "NOISE_FLOOR_ATR_MULTIPLIER", 0.75)
    abs_min = getattr(config, "NOISE_FLOOR_ABS_MIN", 0.08)

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


def _pre_submission_veto(signal: TradeSignal, df: Optional[object] = None) -> tuple:
    """
    ATR-based pre-submission veto.
    Hard Veto:  stop_distance < noise_floor * 0.60  -> block entry
    Soft Warn:  stop_distance < noise_floor          -> warn, allow entry
    Returns: (should_skip: bool, reason: str)
    """
    stop_distance   = abs(signal.entry_price - signal.stop_price)
    noise_floor     = _compute_noise_floor(signal, df)
    hard_veto_ratio = getattr(config, "NOISE_FLOOR_HARD_VETO_RATIO", 0.60)
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


def _validate_signal_prices(signal: TradeSignal) -> str:
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
    df_history = None
    try:
        from strategies.intraday_data import fetch_intraday
        df_history = fetch_intraday(signal.symbol)
    except Exception:
        pass

    should_skip, veto_reason = _pre_submission_veto(signal, df_history)
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

    price_reason = _validate_signal_prices(signal)
    if price_reason:
        result["reason"] = price_reason
        logger.info("Skipped %s: %s", signal.symbol, price_reason)
        return result

    allow_short = getattr(config, "ALLOW_SHORT", False)
    if signal.side.upper() == "SELL" and not allow_short:
        result["reason"] = "Short entries are disabled"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    executable_signal = _executable_signal(signal, entry_limit_price)
    executable_price_reason = _validate_signal_prices(executable_signal)
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

    # Mega-Cap Counter-Trend SHORT Veto Guard in RISK_ON Regime
    from strategies.mega_cap_guard import should_veto_mega_cap_short
    regime_mode = getattr(regime_result, "mode", "NEUTRAL") if regime_result else "NEUTRAL"
    veto_mega, veto_reason = should_veto_mega_cap_short(
        symbol=signal.symbol,
        regime_mode=regime_mode,
        rsi_5m=getattr(signal, "rsi", 50.0),
        price_vs_vwap=getattr(signal, "price_vs_vwap", 0.0),
    )
    if signal.side.upper() in ("SELL", "SHORT") and veto_mega:
        result["reason"] = f"Mega-Cap Counter-Trend Veto: {veto_reason}"
        logger.info("[MEGA-CAP SHORT VETO] Blocked %s: %s", signal.symbol, result["reason"])
        return result

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

    risk_dollars = estimated_risk(executable_signal, shares)
    trades_today = today_trade_count()
    
    is_connected = getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)
    if not dry_run and not is_connected:
        result["reason"] = "Broker is not connected"
        return result
    open_pos = bridge.open_position_count() if is_connected else 0

    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy)
    risk_pct = strat_cfg.max_risk_pct if strat_cfg else 1.0

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
        from strategies.trade_journal import get_journal_summary
        summary = get_journal_summary()
        recent_trades = summary.get("recent_trades", []) if isinstance(summary, dict) else []
        symbol_entries = sum(1 for t in recent_trades if isinstance(t, dict) and t.get("symbol") == signal.symbol)
        if symbol_entries >= max_entries_per_symbol:
            result["reason"] = f"Max Symbol Entries Gate: {signal.symbol} reached session limit of {max_entries_per_symbol} entries"
            logger.info("[SYMBOL ENTRY GATE] Blocked %s: %s", signal.symbol, result["reason"])
            return result
    except Exception:
        pass

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
            effective_max_trades = min(effective_max_trades, 50)

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

    # Idempotency Check
    cache_key = f"{signal.symbol}_{signal.side}"
    last_placed_time = _recent_orders.get(cache_key, 0)
    if time.time() - last_placed_time < IDEMPOTENCY_TTL:
        result["reason"] = f"Idempotency block: {signal.side} order placed within {IDEMPOTENCY_TTL}s"
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
        result["status"] = "DRY_RUN"
        result["reason"] = "Dry run -- no order placed"
        logger.info("[DRY RUN] Would place: %s %d %s", signal.side, shares, signal.symbol)
        # Update idempotency cache even in dry run to avoid repeating spams in dry-run logs
        _recent_orders[cache_key] = time.time()
        return result

    if not config.BRACKET_ORDER_ENABLED:
        result["reason"] = "Protective bracket orders are required for live paper entries"
        return result

    try:
        limit_price = executable_signal.entry_price
        stop_price = signal.stop_price
        target_price = signal.target_price

        # Fresh pre-submission quote check to minimize quote staleness
        get_latest_p = getattr(bridge, "get_latest_price", None)
        if callable(get_latest_p):
            try:
                fresh_p = get_latest_p(signal.symbol)
                if fresh_p and fresh_p > 0:
                    slippage = abs(fresh_p - executable_signal.entry_price) / executable_signal.entry_price
                    if slippage > 0.003: # 0.3% threshold
                        logger.warning(
                            f"[PRE-SUBMISSION VETO] Signal quote stale for {signal.symbol}: "
                            f"signal=${executable_signal.entry_price:.2f}, fresh=${fresh_p:.2f} "
                            f"(slippage={slippage*100:.2f}% > 0.30%). Trade aborted for safety."
                        )
                        result["status"] = "VETOED"
                        result["reason"] = f"Pre-submission quote slippage {slippage*100:.2f}% exceeded 0.3% threshold"
                        return result

                    offset_stop = executable_signal.entry_price - signal.stop_price
                    offset_target = signal.target_price - executable_signal.entry_price
                    limit_price = fresh_p
                    stop_price = round(fresh_p - offset_stop, 2)
                    target_price = round(fresh_p + offset_target, 2)
                    logger.info(f"Refreshed pre-submission quote for {signal.symbol}: limit=${limit_price:.2f}, stop=${stop_price:.2f}, target=${target_price:.2f}")
            except Exception as q_err:
                logger.debug(f"Pre-submission quote refresh skipped for {signal.symbol}: {q_err}")

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

        # Register expected prices for signed slippage and execution latency tracking
        try:
            from strategies.order_registry import register_order_expected_price
            now_ts = time.time()
            parent_id = order_ids[0] if order_ids else order.get("order_id")
            stop_id = order.get("stop_order_id") or (order_ids[2] if len(order_ids) >= 3 else (order_ids[1] if len(order_ids) >= 2 else None))
            tp_id = order.get("tp_order_id") or (order_ids[1] if len(order_ids) >= 2 else None)
            exit_side = "SELL" if signal.side.upper() == "BUY" else "BUY"

            if parent_id:
                register_order_expected_price(parent_id, expected_price=signal.entry_price, side=signal.side, order_type="ENTRY", submit_ts=now_ts, symbol=signal.symbol)
            if stop_id:
                register_order_expected_price(stop_id, expected_price=stop_price, side=exit_side, order_type="STOP_LOSS", submit_ts=now_ts, symbol=signal.symbol)
            if tp_id and tp_id != stop_id:
                register_order_expected_price(tp_id, expected_price=target_price, side=exit_side, order_type="TAKE_PROFIT", submit_ts=now_ts, symbol=signal.symbol)
        except Exception as reg_err:
            logger.debug(f"Failed to register order expected price: {reg_err}")

        result["status"] = "PLACED"
        result["reason"] = "Order placed"
        result["order"] = order
        
        # In Alpaca, orders execute on exchange, and we handle fills asynchronously
        # We also initialize trailing stop directly upon submission in paper-mode
        try:
            from strategies.trailing_stop import manager
            stop_order_id = order.get("stop_order_id") if isinstance(order, dict) else getattr(order, "stop_order_id", None)
            if not stop_order_id:
                stop_order_id = order_ids[2] if len(order_ids) >= 3 else (order_ids[1] if len(order_ids) >= 2 else None)
            tp_order_id = order.get("tp_order_id") if isinstance(order, dict) else getattr(order, "tp_order_id", None)
            if not tp_order_id and len(order_ids) >= 2:
                tp_order_id = order_ids[1]
            manager.initialize_position(signal, limit_price, qty=shares, stop_order_id=stop_order_id, bridge=bridge, tp_order_id=tp_order_id)
        except Exception as e:
            logger.error("Failed to init trailing stop for %s: %s", signal.symbol, e)
        
        # Update idempotency cache
        _recent_orders[cache_key] = time.time()

        parent_order_id = order_ids[0] if order_ids else order.get("order_id")
        log_trade(
            symbol=signal.symbol,
            side=signal.side,
            strategy=signal.strategy,
            qty=shares,
            entry_price=limit_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            notes=signal.reason,
            event_type="ORDER_SUBMITTED",
            order_id=parent_order_id,
            signal_id=signal.signal_id,
        )

        alert_msg = (
            f"🚀 **PAPER TRADE PLACED** 🚀\n"
            f"**Symbol**: {signal.symbol}\n"
            f"**Side**: {signal.side} ({shares} shares)\n"
            f"**Strategy**: {signal.strategy}\n"
            f"**Entry**: ${limit_price:.2f}\n"
            f"**Stop**: ${signal.stop_price:.2f}\n"
            f"**Target**: ${signal.target_price:.2f}\n"
            f"**Reason**: {signal.reason}"
        )
        send_discord_alert(alert_msg)

    except Exception as exc:
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
