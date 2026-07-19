"""
intraday_risk.py -- Intraday risk checks.

Pure, offline risk functions. No broker dependency.
Every function takes plain values and returns bool/float.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

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
    if current_pnl <= -config.MAX_DAILY_LOSS_DOLLARS:
        return False
    if equity > 0:
        loss_pct = abs(current_pnl) / equity * 100.0
        if current_pnl < 0 and loss_pct >= config.MAX_DAILY_LOSS_PCT:
            return False
    return True


def check_max_trades(trades_today: int) -> bool:
    return trades_today < config.MAX_TRADES_PER_DAY


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


def get_recent_symbol_loss_time(symbol: str, journal_file=None) -> Optional[datetime]:
    symbol_trades = [trade for trade in get_today_closed_trades(journal_file) if trade.get("symbol") == symbol]
    if not symbol_trades:
        return None

    last_trade = symbol_trades[-1]
    if float(last_trade.get("realized_pnl", 0.0) or 0.0) < 0:
        closed_at = last_trade.get("closed_at")
        if isinstance(closed_at, datetime):
            return closed_at
    return None


def pre_trade_check(
    equity: float,
    current_pnl: float,
    trades_today: int,
    open_positions: int,
    day_trades_last_5_days: int,
    risk_dollars: float,
    symbol: Optional[str] = None,
    max_risk_pct: float = 1.0,
) -> str:
    """Run all risk checks before placing a trade."""
    if current_pnl is None:
        return "Current broker PnL unavailable; refusing new entries"

    if not check_daily_loss(current_pnl, equity):
        return f"Daily loss limit reached (PnL ${current_pnl:.2f})"

    consecutive_losses = get_consecutive_losses()
    max_consecutive_losses = getattr(config, "MAX_CONSECUTIVE_LOSSES", 3)
    if consecutive_losses >= max_consecutive_losses:
        return f"Consecutive loss limit reached ({consecutive_losses}/{max_consecutive_losses})"

    if not check_max_trades(trades_today):
        return f"Max trades/day reached ({trades_today}/{config.MAX_TRADES_PER_DAY})"

    if not check_max_positions(open_positions):
        return f"Max open positions reached ({open_positions}/{config.MAX_OPEN_POSITIONS})"

    if not check_pdt(day_trades_last_5_days, equity):
        return (
            f"PDT warning: {day_trades_last_5_days} day trades in 5 days, "
            f"equity ${equity:,.0f} < ${config.PDT_MIN_EQUITY:,.0f}"
        )

    if not check_flatten_zone():
        mins = minutes_until_close()
        return f"Flatten zone: {mins:.0f} min until close, no new entries"

    if symbol:
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
