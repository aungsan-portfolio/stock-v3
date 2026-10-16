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

def get_default_risk_store_path() -> Path:
    try:
        import config
        return Path(getattr(config, "DATA_DIR", "data")) / "swing_position_risk.json"
    except Exception:
        return Path("data/swing_position_risk.json")


DEFAULT_RISK_STORE_PATH = get_default_risk_store_path()


@dataclass
class PositionRiskBasis:
    symbol: str
    entry_price: float
    initial_stop_price: float
    initial_risk_per_share: float
    initial_total_risk: float
    shares: int
    entry_timestamp_utc: str
    basis_source: str = "INITIAL_STORE"
    initial_r_available: bool = True
    safety_eligible: bool = True
    r_multiple_eligible: bool = True


def load_risk_store(path: Optional[Path] = None) -> Dict[str, PositionRiskBasis]:
    target_path = Path(path or get_default_risk_store_path())
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
    target_path = Path(path or get_default_risk_store_path())
    store = load_risk_store(target_path)
    store[basis.symbol.upper()] = basis
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({sym: asdict(b) for sym, b in store.items()}, f, indent=2)
    temp_path.replace(target_path)


def remove_risk_basis(symbol: str, path: Optional[Path] = None) -> None:
    target_path = Path(path or get_default_risk_store_path())
    store = load_risk_store(target_path)
    if symbol.upper() in store:
        del store[symbol.upper()]
        temp_path = target_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({sym: asdict(b) for sym, b in store.items()}, f, indent=2)
        temp_path.replace(target_path)


def reconstruct_risk_basis_from_broker(
    symbol: str,
    avg_entry_price: float,
    current_stop_price: float,
    shares: int,
    entry_timestamp_utc: Optional[str] = None,
) -> PositionRiskBasis:
    """
    Reconstructs risk basis from broker-authoritative position and resting stop.
    Explicitly flags that initial R basis is unavailable (since stop may have trailed),
    making it safety-eligible for portfolio heat calculation, but NOT eligible for
    profit acceleration R-multiple exits.
    """
    from datetime import datetime, timezone
    risk_per_share = max(0.01, avg_entry_price - current_stop_price)
    ts = entry_timestamp_utc or datetime.now(timezone.utc).isoformat()
    return PositionRiskBasis(
        symbol=symbol.upper().strip(),
        entry_price=avg_entry_price,
        initial_stop_price=current_stop_price,
        initial_risk_per_share=risk_per_share,
        initial_total_risk=shares * risk_per_share,
        shares=shares,
        entry_timestamp_utc=ts,
        basis_source="BROKER_CURRENT_STOP",
        initial_r_available=False,
        safety_eligible=True,
        r_multiple_eligible=False,
    )


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
        r_val = (current_price - basis.entry_price) / basis.initial_risk_per_share
    elif current_stop_price is not None and current_stop_price < avg_entry_price:
        # Fallback to current stop distance if known
        stop_dist = avg_entry_price - current_stop_price
        if stop_dist > 0.0:
            r_val = (current_price - avg_entry_price) / stop_dist
        else:
            proxy_dist = max(0.01, avg_entry_price * 0.02)
            r_val = (current_price - avg_entry_price) / proxy_dist
    else:
        # Fallback to 2% proxy distance if stop unknown (analytics only)
        proxy_dist = max(0.01, avg_entry_price * 0.02)
        r_val = (current_price - avg_entry_price) / proxy_dist

    # Sanity check: Swing positions rarely exceed 10.0R under normal conditions.
    # An extreme value indicates basis corruption or severe data skew.
    if abs(r_val) > 10.0:
        logger.error(
            "[RISK SANITY ANOMALY] %s computed abnormal unrealized R=%.2f "
            "(curr=$%.2f, entry=$%.2f). Possible risk basis corruption! "
            "Falling back to broker 2%% proxy distance for rotation safety.",
            symbol, r_val, current_price, (basis.entry_price if basis else avg_entry_price)
        )
        # Fail-safe defense: Never propagate extreme corrupted R to rotation logic.
        # Fall back to broker-authoritative 2% proxy distance so position can still be managed.
        proxy_dist = max(0.01, avg_entry_price * 0.02)
        r_val = (current_price - avg_entry_price) / proxy_dist

    return r_val


def get_unrealized_r_analytics(
    symbol: str,
    current_price: float,
    avg_entry_price: float,
    current_stop_price: Optional[float] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Analytics-only reporting helper with explicit provenance tagging.
    NEVER used for safety gates, risk bounds, or sizing decisions.
    """
    store = load_risk_store(path)
    basis = store.get(symbol.upper())
    if basis and basis.initial_risk_per_share > 0.0:
        r_val = (current_price - basis.entry_price) / basis.initial_risk_per_share
        return {
            "r_value": round(r_val, 4),
            "basis_source": basis.basis_source,
            "safety_eligible": basis.safety_eligible,
            "r_multiple_eligible": basis.r_multiple_eligible,
        }

    if current_stop_price is not None and current_stop_price < avg_entry_price:
        stop_dist = avg_entry_price - current_stop_price
        if stop_dist > 0.0:
            r_val = (current_price - avg_entry_price) / stop_dist
            return {
                "r_value": round(r_val, 4),
                "basis_source": "BROKER_CURRENT_STOP",
                "safety_eligible": True,
                "r_multiple_eligible": False,
            }

    proxy_dist = max(0.01, avg_entry_price * 0.02)
    r_val = (current_price - avg_entry_price) / proxy_dist
    return {
        "r_value": round(r_val, 4),
        "basis_source": "PROXY_2PCT",
        "safety_eligible": False,
        "r_multiple_eligible": False,
    }
