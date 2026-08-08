"""
mega_cap_guard.py -- Mega-Cap counter-trend veto rules.
"""
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

MEGA_CAPS = frozenset({"NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL"})


def should_veto_mega_cap_short(
    symbol: str,
    regime_mode: str,
    spy_change: float = 0.0,
    qqq_change: float = 0.0,
    rsi_5m: float = 50.0,
    price_vs_vwap: float = 0.0,
) -> Tuple[bool, str]:
    """
    Returns (should_veto: bool, reason: str)
    """
    sym = symbol.upper().strip()
    if sym not in MEGA_CAPS:
        return False, ""

    if regime_mode != "RISK_ON":
        return False, ""

    # Veto condition 1: Broad market momentum too strong
    if spy_change > 0.50 and qqq_change > 1.00:
        return True, f"Broad market too strong (SPY={spy_change:+.2f}%, QQQ={qqq_change:+.2f}%)"

    # Veto condition 2: RSI still mid-range / neutral (no weakness confirmed)
    if rsi_5m > 45.0:
        return True, f"RSI {rsi_5m:.1f} > 45.0 — no weakness confirmed"

    # Veto condition 3: Price trading above VWAP
    if price_vs_vwap > 0.0:
        return True, f"Price {price_vs_vwap:+.2f}% above VWAP — no short confirmation"

    return False, ""
