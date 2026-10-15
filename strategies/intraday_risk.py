"""
intraday_risk.py -- Intraday risk checks.

Pure, offline risk functions. No broker dependency.
Every function takes plain values and returns bool/float.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import config
from strategies.errors import (
    DailyLossLimitError,
    PDTViolationError,
    FlattenRequiredError,
    RiskLimitBreachedError,
)
from strategies.trade_journal import get_today_closed_trades
from strategies.session import should_flatten, minutes_until_close

logger = logging.getLogger(__name__)


def check_daily_loss(current_pnl: float, equity: float) -> bool:
    if current_pnl is None:
        return False
    dollar_limit = float(getattr(config, "MAX_DAILY_LOSS_DAYTRADE", getattr(config, "MAX_DAILY_LOSS_DOLLARS", 250.0)))
    if current_pnl <= -dollar_limit:
        return False
    if equity > 0:
        loss_pct = abs(current_pnl) / equity * 100.0
        if current_pnl < 0 and loss_pct >= config.MAX_DAILY_LOSS_PCT:
            return False
    return True


def check_max_trades(trades_today: int, max_trades: Optional[int] = None) -> bool:
    limit = max_trades if max_trades is not None else getattr(config, "MAX_DAILY_TRADES", getattr(config, "MAX_TRADES_PER_DAY", 15))
    return trades_today < limit


def check_max_positions(open_positions: int) -> bool:
    return open_positions < config.MAX_OPEN_POSITIONS


def check_pdt(
    day_trades_last_5_days: int,
    equity: float,
) -> bool:
    if not config.PDT_ENABLED:
        return True
    if equity >= config.PDT_MIN_EQUITY:
        return True
    return day_trades_last_5_days < config.PDT_MAX_DAY_TRADES_5_DAYS


def check_flatten_zone() -> bool:
    return not should_flatten()


def check_trade_risk(risk_dollars: float, equity: float, max_risk_pct: float) -> bool:
    if equity <= 0:
        return False
    max_risk = equity * (max_risk_pct / 100.0)
    return risk_dollars <= max_risk


def get_consecutive_losses(journal_file=None) -> int:
    closed_trades = get_today_closed_trades(journal_file)
    consecutive_losses = 0
    for trade in reversed(closed_trades):
        if float(trade.get("realized_pnl", 0.0) or 0.0) < 0:
            consecutive_losses += 1
            continue
        break
    return consecutive_losses


_RECENT_SYMBOL_LOSS_TIMESTAMPS: dict = {}


def register_symbol_loss(symbol: str, timestamp: Optional[datetime] = None) -> None:
    """Register an in-memory loss timestamp for a symbol to enforce re-entry cooldown instantly."""
    ts = timestamp or datetime.now(timezone.utc)
    _RECENT_SYMBOL_LOSS_TIMESTAMPS[symbol.upper().strip()] = ts
    logger.info(f"Registered instant loss cooldown timestamp for {symbol} at {ts}")


def get_recent_symbol_loss_time(symbol: str, journal_file=None) -> Optional[datetime]:
    sym = symbol.upper().strip()
    mem_time = _RECENT_SYMBOL_LOSS_TIMESTAMPS.get(sym)
    
    symbol_trades = [trade for trade in get_today_closed_trades(journal_file) if trade.get("symbol") == sym]
    journal_time = None
    if symbol_trades:
        last_trade = symbol_trades[-1]
        if float(last_trade.get("realized_pnl", 0.0) or 0.0) < 0:
            closed_at = last_trade.get("closed_at")
            if isinstance(closed_at, datetime):
                journal_time = closed_at

    if mem_time and journal_time:
        if mem_time.tzinfo is None:
            mem_time = mem_time.replace(tzinfo=timezone.utc)
        if journal_time.tzinfo is None:
            journal_time = journal_time.replace(tzinfo=timezone.utc)
        return max(mem_time, journal_time)
    return mem_time or journal_time


def reset_risk_state() -> None:
    """Reset in-memory risk tracking state (loss timestamps, bypass cooldowns)."""
    global _RECENT_SYMBOL_LOSS_TIMESTAMPS, _bypass_cooldowns
    _RECENT_SYMBOL_LOSS_TIMESTAMPS.clear()
    _bypass_cooldowns.clear()


def get_symbol_consecutive_losses(symbol: str, journal_file=None) -> int:
    symbol_trades = [trade for trade in get_today_closed_trades(journal_file) if trade.get("symbol") == symbol]
    if not symbol_trades:
        return 0

    grouped_cycles = []
    current_group = []

    for trade in symbol_trades:
        if not current_group:
            current_group.append(trade)
            continue

        last = current_group[-1]
        same_entry = (
            (trade.get("entry_order_id") and trade.get("entry_order_id") == last.get("entry_order_id"))
            or (trade.get("signal_id") and trade.get("signal_id") == last.get("signal_id"))
        )

        if same_entry:
            current_group.append(trade)
        else:
            grouped_cycles.append(current_group)
            current_group = [trade]

    if current_group:
        grouped_cycles.append(current_group)

    consecutive_losses = 0
    for cycle in reversed(grouped_cycles):
        cycle_pnl = sum(float(t.get("realized_pnl", 0.0) or 0.0) for t in cycle)
        if cycle_pnl < 0:
            consecutive_losses += 1
        else:
            break

    return consecutive_losses


TECH_CORRELATED_CLUSTER = {"NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AMD", "GOOGL", "QQQ", "PLTR", "AVGO", "MU", "NFLX"}
SOXX_TECH_LONG_VETO_THRESHOLD = -0.75  # SOXX -0.75% threshold


def _get_soxx_realtime_pct_change(bridge=None) -> Optional[float]:
    """Fetch real-time % change for SOXX semiconductor ETF relative to prior session close.
    Surfaces dual baselines (prior-day close including overnight gap vs session open) in telemetry.
    """
    try:
        from strategies.regime_engine import get_soxx_pct_change
        cached = get_soxx_pct_change()
        if cached is not None:
            return cached
    except Exception:
        pass
    try:
        from strategies.regime_engine import _get_premarket_pct_change
        pct, source, ts = _get_premarket_pct_change("SOXX", bridge=bridge)
        if pct is not None:
            return pct
    except Exception:
        pass
    try:
        import yfinance as yf
        ticker = yf.Ticker("SOXX")
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            last = float(getattr(fast_info, "last_price", 0.0) or 0.0)
            prev = float(getattr(fast_info, "previous_close", 0.0) or 0.0)
            open_px = float(getattr(fast_info, "open", 0.0) or 0.0)
            if last > 0 and prev > 0:
                pct_close = round(((last - prev) / prev) * 100.0, 2)
                pct_open = round(((last - open_px) / open_px) * 100.0, 2) if open_px > 0 else None
                logger.info(
                    "[SOXX BASELINE TELEMETRY] yfinance fast_info: last=%.2f, prev_close=%.2f (pct_vs_close=%+.2f%%), open=%.2f (pct_vs_open=%s)",
                    last, prev, pct_close, open_px, f"{pct_open:+.2f}%" if pct_open is not None else "N/A"
                )
                return pct_close
    except Exception as e:
        logger.debug("[SOXX GATE] yfinance fast_info fallback failed: %s", e)
    try:
        import yfinance as yf
        soxx = yf.Ticker("SOXX")
        hist = soxx.history(period="5d", interval="1d")
        if len(hist) >= 2:
            prev_close = float(hist["Close"].iloc[-2])
            last_price = float(hist["Close"].iloc[-1])
            today_open = float(hist["Open"].iloc[-1]) if "Open" in hist.columns else 0.0
            if prev_close > 0 and last_price > 0:
                pct_close = round(((last_price - prev_close) / prev_close) * 100.0, 2)
                pct_open = round(((last_price - today_open) / today_open) * 100.0, 2) if today_open > 0 else None
                logger.info(
                    "[SOXX BASELINE TELEMETRY] yfinance daily history: last=%.2f, prev_close=%.2f (pct_vs_close=%+.2f%%), open=%.2f (pct_vs_open=%s)",
                    last_price, prev_close, pct_close, today_open, f"{pct_open:+.2f}%" if pct_open is not None else "N/A"
                )
                return pct_close
    except Exception as e:
        logger.debug("[SOXX GATE] Error fetching SOXX daily history: %s", e)
    return None


from rs_calculator import RelativeStrengthEvaluator
from strategies.soxx_gate_config import (
    SOXX_VETO_THRESHOLD,
    SOXX_GOVERNED_SYMBOLS,
    RS_BYPASS_COOLDOWN_CYCLES,
)

_rs_evaluator = RelativeStrengthEvaluator()
_bypass_cooldowns: dict[str, int] = {}


def check_soxx_tech_long_gate(
    symbol: str,
    side: str = "BUY",
    pred_conf: float = 0.55,
    symbol_df=None,
    soxx_df=None,
    bridge=None,
) -> Tuple[bool, str]:
    """
    Blocks BUY/Long entries on Tech/Semi symbols if SOXX ETF is weak (<= -0.75%),
    UNLESS the individual symbol demonstrates strong Relative Strength (RSI >= 55 or Alpha >= +1.5%).
    """
    if not getattr(config, "SOXX_GATE_ENABLED", True):
        return True, "SOXX gate: disabled"
    if side.upper() not in ("BUY", "LONG"):
        return True, "SOXX gate: N/A (short trade)"
    sym_upper = symbol.upper().strip()
    if sym_upper not in SOXX_GOVERNED_SYMBOLS and sym_upper not in {"NVDA", "AMD", "INTC", "SMCI", "QCOM", "MU", "AVGO"}:
        return True, "SOXX gate: N/A (non-tech symbol)"

    if bridge is not None:
        try:
            soxx_pct = _get_soxx_realtime_pct_change(bridge=bridge)
        except TypeError:
            soxx_pct = _get_soxx_realtime_pct_change()
    else:
        soxx_pct = _get_soxx_realtime_pct_change()
    if soxx_pct is None:
        fail_closed = getattr(config, "SOXX_GATE_FAIL_CLOSED", True)
        if fail_closed:
            logger.warning("[SOXX GATE] Could not fetch SOXX benchmark data. Failing closed (blocking %s entry).", symbol)
            return False, "SOXX gate: data unavailable (fail-closed)"
        else:
            logger.warning("[SOXX GATE] Could not fetch SOXX data. Allowing %s entry.", symbol)
            return True, "SOXX gate: data unavailable (pass)"

    # Convert pct change from percentage (e.g. -1.49) to float ratio (e.g. -0.0149)
    soxx_ratio = soxx_pct / 100.0

    if soxx_ratio <= SOXX_VETO_THRESHOLD:
        # Check RS Bypass if DataFrames are provided
        if symbol_df is not None and soxx_df is not None:
            remaining_cooldown = _bypass_cooldowns.get(sym_upper, 0)
            if remaining_cooldown > 0:
                _bypass_cooldowns[sym_upper] = remaining_cooldown - 1
                reason = f"SOXX Gate VETO: {sym_upper} RS Bypass on cooldown ({remaining_cooldown} cycles left). SOXX is {soxx_pct:.2f}%"
                logger.info("[SOXX GATE VETO] %s", reason)
                return False, reason

            try:
                rs_res = _rs_evaluator.evaluate(
                    symbol=sym_upper,
                    symbol_df=symbol_df,
                    soxx_df=soxx_df,
                    soxx_chg_pct=soxx_ratio,
                    pred_conf=pred_conf,
                )
                if rs_res.bypass_granted:
                    _bypass_cooldowns[sym_upper] = RS_BYPASS_COOLDOWN_CYCLES
                    reason = f"SOXX Gate BYPASS: {sym_upper} Long ALLOWED despite SOXX {soxx_pct:.2f}%. {rs_res.bypass_reason}"
                    logger.info("[SOXX GATE BYPASS] %s", reason)
                    return True, reason
            except Exception as exc:
                logger.warning("[SOXX GATE] RS Eval error: %s", exc)

        reason = (
            f"SOXX Pre-Trade Gate VETO: {symbol} Long blocked. "
            f"SOXX is {soxx_pct:.2f}% (threshold: {SOXX_TECH_LONG_VETO_THRESHOLD}%)"
        )
        logger.info("[SOXX GATE VETO] %s", reason)
        return False, reason

    return True, f"SOXX gate: passed ({soxx_pct:.2f}%)"


def check_correlated_cluster_exposure(open_symbols: list, new_symbol: str) -> Optional[str]:
    """Prevent sector risk concentration by capping simultaneous open positions in the Tech correlated cluster."""
    if not new_symbol or new_symbol.upper() not in TECH_CORRELATED_CLUSTER:
        return None
    
    max_tech_open = getattr(config, "MAX_CORRELATED_TECH_POSITIONS", 2)
    open_syms = [str(s).upper() for s in (open_symbols or [])]
    tech_count = sum(1 for s in open_syms if s in TECH_CORRELATED_CLUSTER)
    if tech_count >= max_tech_open:
        return f"Correlated Tech Cluster Cap reached ({tech_count}/{max_tech_open}): Blocked {new_symbol} to prevent sector concentration risk"
    return None


def pre_trade_check(
    equity: float,
    current_pnl: Optional[float],
    trades_today: int,
    open_positions: int,
    day_trades_last_5_days: int,
    risk_dollars: float,
    symbol: Optional[str] = None,
    max_risk_pct: float = 1.0,
    open_symbols: Optional[list] = None,
    max_trades: Optional[int] = None,
    side: str = "BUY",
    check_reentry_cooldown: bool = True,
) -> str:
    """Run all risk checks before placing a trade."""
    if current_pnl is None:
        return "Current broker PnL unavailable; refusing new entries"

    if symbol and open_symbols:
        cluster_err = check_correlated_cluster_exposure(open_symbols, symbol)
        if cluster_err:
            return cluster_err

    if symbol:
        soxx_ok, soxx_reason = check_soxx_tech_long_gate(symbol, side=side)
        if not soxx_ok:
            return soxx_reason

    if not check_daily_loss(current_pnl, equity):
        return f"Daily loss limit reached (PnL ${current_pnl:.2f})"

    if symbol:
        symbol_losses = get_symbol_consecutive_losses(symbol)
        max_symbol_losses = getattr(config, "MAX_SYMBOL_CONSECUTIVE_LOSSES", 2)
        if symbol_losses >= max_symbol_losses:
            return f"Consecutive loss limit reached for {symbol} ({symbol_losses}/{max_symbol_losses})"
    else:
        consecutive_losses = get_consecutive_losses()
        max_consecutive_losses = getattr(config, "MAX_CONSECUTIVE_LOSSES", 10)
        if consecutive_losses >= max_consecutive_losses:
            return f"Consecutive loss limit reached ({consecutive_losses}/{max_consecutive_losses})"

    effective_limit = max_trades if max_trades is not None else getattr(config, "MAX_DAILY_TRADES", getattr(config, "MAX_TRADES_PER_DAY", 15))
    if not check_max_trades(trades_today, max_trades=effective_limit):
        return f"Max trades/day reached ({trades_today}/{effective_limit})"

    if not check_max_positions(open_positions):
        return f"Max open positions reached ({open_positions}/{config.MAX_OPEN_POSITIONS})"

    if symbol and open_symbols and symbol.upper().strip() in [str(s).upper().strip() for s in open_symbols]:
        return f"Duplicate entry blocked: already holding {symbol}"

    if not check_pdt(day_trades_last_5_days, equity):
        return (
            f"PDT warning: {day_trades_last_5_days} day trades in 5 days, "
            f"equity ${equity:,.0f} < ${config.PDT_MIN_EQUITY:,.0f}"
        )

    if not check_flatten_zone():
        mins = minutes_until_close()
        return f"Flatten zone: {mins:.0f} min until close, no new entries"

    if symbol:
        symbol_losses = get_symbol_consecutive_losses(symbol)
        max_symbol_losses = getattr(config, "MAX_SYMBOL_CONSECUTIVE_LOSSES", 2)
        if symbol_losses >= max_symbol_losses:
            logger.info("Re-entry blocked for %s: consecutive loss limit reached (%d/%d)", symbol, symbol_losses, max_symbol_losses)
            return f"Symbol consecutive loss limit reached for {symbol} ({symbol_losses}/{max_symbol_losses})"

        if check_reentry_cooldown:
            last_loss_time = get_recent_symbol_loss_time(symbol)
            if last_loss_time is not None:
                cooldown_minutes = getattr(config, "REENTRY_COOLDOWN_MINUTES", 5)
                elapsed_minutes = (datetime.now(timezone.utc) - last_loss_time).total_seconds() / 60.0
                if elapsed_minutes < cooldown_minutes:
                    remaining_minutes = cooldown_minutes - elapsed_minutes
                    return (
                        f"Re-entry cooldown active for {symbol} "
                        f"({remaining_minutes:.1f} min left of {cooldown_minutes} min)"
                    )

    if not check_trade_risk(risk_dollars, equity, max_risk_pct):
        max_risk = equity * (max_risk_pct / 100.0)
        return f"Trade risk ${risk_dollars:.2f} > max ${max_risk:.2f}"

    return ""


def should_force_flatten(open_positions: int) -> bool:
    if open_positions <= 0:
        return False
    return should_flatten()
