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

    # Validate signal geometry using tick size (SEC Rule 612)
    tick_size = 0.01 if signal.entry_price >= 1.0 else 0.0001
    target_dist = round(abs(signal.target_price - signal.entry_price), 6)
    stop_dist = round(abs(signal.entry_price - signal.stop_price), 6)
    if target_dist <= tick_size:
        return f"Target distance {target_dist:.5f} is less than or equal to tick size {tick_size:.5f} (invalid geometry)"
    if stop_dist <= tick_size:
        return f"Stop distance {stop_dist:.5f} is less than or equal to tick size {tick_size:.5f} (invalid geometry)"

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


def _min_confidence_for(signal: TradeSignal) -> float:
    """Return the configured confidence gate for this strategy."""
    strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(signal.strategy)
    if isinstance(strat_cfg, dict):
        return float(strat_cfg.get("confidence_min", 0.65))
    return float(getattr(strat_cfg, "confidence_min", 0.65) if strat_cfg else 0.65)


def execute_signal(
    signal: TradeSignal,
    bridge: object,
    equity: float,
    current_pnl: float,
    day_trades_last_5_days: int = 0,
    dry_run: bool = None,
    requested_shares: Optional[int] = None,
    entry_limit_price: Optional[float] = None,
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

    min_confidence = _min_confidence_for(signal)
    if signal.confidence < min_confidence:
        result["reason"] = f"Confidence {signal.confidence:.2f} < {min_confidence:.2f} ({signal.strategy})"
        logger.info("Skipped %s: %s", signal.symbol, result["reason"])
        return result

    shares = calculate_shares(executable_signal, equity)
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

    risk_reason = pre_trade_check(
        equity=equity,
        current_pnl=current_pnl,
        trades_today=trades_today,
        open_positions=open_pos,
        day_trades_last_5_days=day_trades_last_5_days,
        risk_dollars=risk_dollars,
        max_risk_pct=risk_pct,
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

        order = bridge.place_bracket_order(
            symbol=signal.symbol,
            side=signal.side,
            qty=shares,
            entry_price=limit_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
        )
        order_ids = order.get("order_ids") or []
        register_signal = getattr(bridge, "register_order_signal", None)
        if callable(register_signal):
            register_signal(order_ids, signal.signal_id)

        result["status"] = "PLACED"
        result["reason"] = "Order placed"
        result["order"] = order
        
        # In Alpaca, orders execute on exchange, and we handle fills asynchronously
        # We also initialize trailing stop directly upon submission in paper-mode
        try:
            from strategies.trailing_stop import manager
            stop_order_id = order_ids[2] if len(order_ids) >= 3 else (order_ids[1] if len(order_ids) >= 2 else None)
            manager.initialize_position(signal, limit_price, qty=shares, stop_order_id=stop_order_id)
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
