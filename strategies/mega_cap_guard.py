"""
mega_cap_guard.py -- Counter-trend SHORT veto rules for RISK_ON regime.
"""
import logging
from typing import Tuple, Dict, Any
import config

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
    Blocks counter-trend SHORT entries during RISK_ON market regime.
    """
    sym = symbol.upper().strip()
    apply_all = getattr(config, "APPLY_SHORT_VETO_TO_ALL_SYMBOLS", True)
    if not apply_all and sym not in MEGA_CAPS:
        return False, ""

    if regime_mode != "RISK_ON":
        return False, ""

    # Veto condition 1: Price trading above VWAP (counter-trend short in bull market)
    if price_vs_vwap > 0.0:
        return True, f"Price {price_vs_vwap:+.2f}% above VWAP — counter-trend short blocked"

    # Veto condition 2: RSI not deeply weak/oversold (user-calibrated threshold: 40.0)
    rsi_threshold = getattr(config, "SHORT_VETO_RSI_THRESHOLD", 40.0)
    if rsi_5m > rsi_threshold:
        return True, f"RSI {rsi_5m:.1f} > {rsi_threshold:.1f} — no breakdown weakness confirmed"

    # Veto condition 3: Broad market momentum too strong
    if spy_change > 0.50 and qqq_change > 1.00:
        return True, f"Broad market too strong (SPY={spy_change:+.2f}%, QQQ={qqq_change:+.2f}%)"

    return False, ""

