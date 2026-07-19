"""Pure risk/reward formula helpers for paper-trading practice.

No network calls, no broker calls, and no file writes live here. These helpers
are used for read-only coach previews and refusal gates only; they do not change
IBKRBridge order sizing.
"""
from __future__ import annotations

import math
from typing import Iterable


def _require_positive(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def _require_non_negative_int(name: str, value: int) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer >= 0") from None
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value


def _side(side: str) -> str:
    side = str(side).upper().strip()
    if side not in {"LONG", "SHORT"}:
        raise ValueError(f"side must be LONG or SHORT, got {side!r}")
    return side


def risk_per_share(entry_price, stop_price, side="LONG") -> float:
    """Return dollars risked per share.

    LONG: entry_price - stop_price
    SHORT: stop_price - entry_price
    """
    entry = _require_positive("entry_price", entry_price)
    stop = _require_positive("stop_price", stop_price)
    side = _side(side)
    risk = entry - stop if side == "LONG" else stop - entry
    if risk <= 0:
        raise ValueError(
            f"risk_per_share must be positive for {side}: "
            f"entry={entry}, stop={stop}"
        )
    return float(risk)


def reward_per_share(entry_price, target_price, side="LONG") -> float:
    """Return dollars rewarded per share.

    LONG: target_price - entry_price
    SHORT: entry_price - target_price
    """
    entry = _require_positive("entry_price", entry_price)
    target = _require_positive("target_price", target_price)
    side = _side(side)
    reward = target - entry if side == "LONG" else entry - target
    if reward <= 0:
        raise ValueError(
            f"reward_per_share must be positive for {side}: "
            f"entry={entry}, target={target}"
        )
    return float(reward)


def risk_reward_ratio(entry_price, stop_price, target_price, side="LONG") -> float:
    """Return reward-per-share divided by risk-per-share."""
    return reward_per_share(entry_price, target_price, side) / risk_per_share(
        entry_price, stop_price, side
    )


def required_target(entry_price, stop_price, required_r=2.0, side="LONG") -> float:
    """Return the target price required to achieve ``required_r``.

    LONG: entry + risk * required_r
    SHORT: entry - risk * required_r
    """
    entry = _require_positive("entry_price", entry_price)
    required_r = _require_positive("required_r", required_r)
    risk = risk_per_share(entry, stop_price, side)
    return float(entry + risk * required_r if _side(side) == "LONG" else entry - risk * required_r)


def account_risk_dollars(account_equity, risk_pct) -> float:
    """Return account_equity * risk_pct."""
    equity = _require_positive("account_equity", account_equity)
    risk_pct = _require_positive("risk_pct", risk_pct)
    if risk_pct > 1:
        raise ValueError(f"risk_pct must be <= 1, got {risk_pct!r}")
    return float(equity * risk_pct)


def shares_for_risk(
    account_equity,
    risk_pct,
    entry_price,
    stop_price,
    side="LONG",
    max_trade_value=None,
    max_position_pct=None,
) -> int:
    """Return integer shares sized by account risk and optional caps."""
    equity = _require_positive("account_equity", account_equity)
    entry = _require_positive("entry_price", entry_price)
    dollars = account_risk_dollars(equity, risk_pct)
    raw_shares = math.floor(dollars / risk_per_share(entry, stop_price, side))

    caps = [raw_shares]
    if max_trade_value is not None:
        max_trade_value = _require_positive("max_trade_value", max_trade_value)
        caps.append(math.floor(max_trade_value / entry))
    if max_position_pct is not None:
        max_position_pct = _require_positive("max_position_pct", max_position_pct)
        if max_position_pct > 1:
            raise ValueError(f"max_position_pct must be <= 1, got {max_position_pct!r}")
        caps.append(math.floor((equity * max_position_pct) / entry))

    return max(int(min(caps)), 0)


def planned_risk_dollars(shares, entry_price, stop_price, side="LONG") -> float:
    """Return shares * risk_per_share."""
    shares = _require_non_negative_int("shares", shares)
    return float(shares * risk_per_share(entry_price, stop_price, side))


def realized_pnl(entry_price, exit_price, shares, side="LONG") -> float:
    """Return realized PnL for a completed trade."""
    entry = _require_positive("entry_price", entry_price)
    exit_price = _require_positive("exit_price", exit_price)
    shares = _require_non_negative_int("shares", shares)
    side = _side(side)
    pnl_per_share = exit_price - entry if side == "LONG" else entry - exit_price
    return float(pnl_per_share * shares)


def r_multiple(entry_price, stop_price, exit_price, shares=1, side="LONG") -> float:
    """Return realized_pnl divided by planned_risk_dollars."""
    shares = _require_non_negative_int("shares", shares)
    if shares == 0:
        raise ValueError("shares must be > 0 for r_multiple")
    planned = planned_risk_dollars(shares, entry_price, stop_price, side)
    if planned <= 0:
        raise ValueError("planned risk must be positive for r_multiple")
    return float(realized_pnl(entry_price, exit_price, shares, side) / planned)


def _finite_values(r_multiples: Iterable[float]) -> list[float]:
    vals = []
    for r in r_multiples:
        try:
            r = float(r)
        except (TypeError, ValueError):
            continue
        if math.isfinite(r):
            vals.append(r)
    return vals


def expectancy_r(r_multiples) -> float:
    """Return the mean finite R value, or 0.0 if none exist."""
    vals = _finite_values(r_multiples)
    return float(sum(vals) / len(vals)) if vals else 0.0


def win_rate(r_multiples) -> float:
    """Return fraction of finite R values greater than 0, or 0.0 if empty."""
    vals = _finite_values(r_multiples)
    return float(sum(1 for r in vals if r > 0) / len(vals)) if vals else 0.0


def profit_factor_from_r(r_multiples) -> float:
    """Return sum(winning R) / abs(sum(losing R))."""
    vals = _finite_values(r_multiples)
    wins = sum(r for r in vals if r > 0)
    losses = sum(r for r in vals if r < 0)
    if wins > 0 and losses == 0:
        return float("inf")
    if wins <= 0:
        return 0.0
    return float(wins / abs(losses))


def atr_stop_price(entry_price, atr, atr_multiple=1.5, side="LONG") -> float:
    """Return an ATR-based stop price."""
    entry = _require_positive("entry_price", entry_price)
    atr = _require_positive("atr", atr)
    atr_multiple = _require_positive("atr_multiple", atr_multiple)
    side = _side(side)
    return float(entry - atr * atr_multiple if side == "LONG" else entry + atr * atr_multiple)

def trailing_stop_price(
    current_price,
    atr=None,
    atr_multiple=1.0,
    trailing_pct=None,
    side="LONG",
    highest_price=None,
    lowest_price=None,
    highest_price_since_entry=None,
    lowest_price_since_entry=None,
) -> float:
    """Return a trailing stop price.

    LONG:
      - ATR mode: current/highest price - atr * atr_multiple
      - Percent mode: highest price * (1 - trailing_pct)

    SHORT:
      - ATR mode: current/lowest price + atr * atr_multiple
      - Percent mode: lowest price * (1 + trailing_pct)

    At least one of atr or trailing_pct must be supplied.
    For SHORT percent trailing, lowest_price is required so the stop follows
    the most favorable low instead of the current price.
    """
    current = _require_positive("current_price", current_price)
    side = _side(side)

    # Backward-compatible aliases used by tests and trade terminology.
    if highest_price is None and highest_price_since_entry is not None:
        highest_price = highest_price_since_entry
    if lowest_price is None and lowest_price_since_entry is not None:
        lowest_price = lowest_price_since_entry

    if atr is None and trailing_pct is None:
        raise ValueError("Either atr or trailing_pct must be provided")

    if side == "LONG":
        anchor = (
            _require_positive("highest_price", highest_price)
            if highest_price is not None
            else current
        )

        if atr is not None:
            atr = _require_positive("atr", atr)
            atr_multiple = _require_positive("atr_multiple", atr_multiple)
            return float(anchor - atr * atr_multiple)

        trailing_pct = _require_positive("trailing_pct", trailing_pct)
        if trailing_pct > 1:
            raise ValueError(f"trailing_pct must be <= 1, got {trailing_pct!r}")
        return float(anchor * (1 - trailing_pct))

    # SHORT
    if trailing_pct is not None and lowest_price is None and atr is None:
        raise ValueError("lowest_price is required for SHORT percent trailing stop")

    anchor = (
        _require_positive("lowest_price", lowest_price)
        if lowest_price is not None
        else current
    )

    if atr is not None:
        atr = _require_positive("atr", atr)
        atr_multiple = _require_positive("atr_multiple", atr_multiple)
        return float(anchor + atr * atr_multiple)

    trailing_pct = _require_positive("trailing_pct", trailing_pct)
    if trailing_pct > 1:
        raise ValueError(f"trailing_pct must be <= 1, got {trailing_pct!r}")
    return float(anchor * (1 + trailing_pct))

