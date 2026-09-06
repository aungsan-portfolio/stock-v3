"""
strategies/swing_risk_store.py — Persistent tracking of initial swing position risk basis.
Ensures non-drifting unrealized R calculation for rotator profit acceleration protection.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_RISK_STORE_PATH = Path("data/swing_position_risk.json")


@dataclass
class PositionRiskBasis:
    symbol: str
    entry_price: float
    initial_stop_price: float
    initial_risk_per_share: float
    initial_total_risk: float
    shares: int
    entry_timestamp_utc: str


def load_risk_store(path: Optional[Path] = None) -> Dict[str, PositionRiskBasis]:
    target_path = Path(path or DEFAULT_RISK_STORE_PATH)
    if not target_path.exists():
        return {}
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {
                sym.upper(): PositionRiskBasis(**item)
                for sym, item in data.items()
            }
    except Exception as exc:
        logger.warning("Could not read risk store from %s: %s", target_path, exc)
        return {}


def save_risk_basis(basis: PositionRiskBasis, path: Optional[Path] = None) -> None:
    target_path = Path(path or DEFAULT_RISK_STORE_PATH)
    store = load_risk_store(target_path)
    store[basis.symbol.upper()] = basis
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({sym: asdict(b) for sym, b in store.items()}, f, indent=2)
    temp_path.replace(target_path)


def remove_risk_basis(symbol: str, path: Optional[Path] = None) -> None:
    target_path = Path(path or DEFAULT_RISK_STORE_PATH)
    store = load_risk_store(target_path)
    if symbol.upper() in store:
        del store[symbol.upper()]
        temp_path = target_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({sym: asdict(b) for sym, b in store.items()}, f, indent=2)
        temp_path.replace(target_path)


def get_unrealized_r(
    symbol: str,
    current_price: float,
    avg_entry_price: float,
    current_stop_price: Optional[float] = None,
    path: Optional[Path] = None,
) -> float:
    """
    Computes true unrealized R based on persisted initial risk basis:
    R_unrealized = (P_mark - P_entry) / R_initial/share
    Guarantees that trailing stops ratcheting up do NOT compress R.
    """
    store = load_risk_store(path)
    basis = store.get(symbol.upper())
    if basis and basis.initial_risk_per_share > 0.0:
        return (current_price - basis.entry_price) / basis.initial_risk_per_share

    # Fallback to current stop distance if known
    if current_stop_price is not None and current_stop_price < avg_entry_price:
        stop_dist = avg_entry_price - current_stop_price
        if stop_dist > 0.0:
            return (current_price - avg_entry_price) / stop_dist

    # Fallback to 2% proxy distance if stop unknown
    proxy_dist = max(0.01, avg_entry_price * 0.02)
    return (current_price - avg_entry_price) / proxy_dist
