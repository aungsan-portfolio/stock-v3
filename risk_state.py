"""risk_state.py — Simple per-day safety state for paper trading.

This module is intentionally small and file-backed so the bot can enforce a
basic max-daily-trades cap across repeated runs of `python main.py paper`.
It does not connect to IBKR and does not place orders.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path

import config

logger = logging.getLogger(__name__)

_STATE_PATH = Path(config.LOG_DIR) / "daily_risk_state.json"


def _today() -> str:
    return _dt.date.today().isoformat()


def _load() -> dict:
    try:
        if not _STATE_PATH.exists():
            return {}
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read daily risk state; starting fresh", exc_info=True)
        return {}


def _save(state: dict) -> None:
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def trades_today(day: str | None = None) -> int:
    day = day or _today()
    return int(_load().get(day, {}).get("trades", 0))


def can_open_more(day: str | None = None) -> bool:
    return trades_today(day) < int(config.MAX_DAILY_TRADES)


def record_trade(day: str | None = None) -> int:
    day = day or _today()
    state = _load()
    state.setdefault(day, {})
    state[day]["trades"] = int(state[day].get("trades", 0)) + 1
    _save(state)
    return int(state[day]["trades"])


def loss_breached(start_equity: float, current_equity: float) -> bool:
    """Return True when the configured daily loss limit has been exceeded."""
    return (float(start_equity) - float(current_equity)) >= float(config.MAX_DAILY_LOSS_USD)
