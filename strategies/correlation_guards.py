"""
correlation_guards.py -- Commodity and sector correlation guards.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

COMMODITY_ETF_MAP = {
    "GDX": "GLD",    # Gold Miners -> Gold ETF
    "GDXJ": "GLD",   # Junior Gold Miners -> Gold ETF
    "USO": "CL=F",   # Oil ETF -> Crude Oil
    "SLV": "GLD",    # Silver ETF -> Gold ETF
}


def get_commodity_direction_bias(symbol: str, market_data: Dict[str, Any] = None) -> str:
    """
    Returns: 'LONG_ONLY' | 'SHORT_ONLY' | 'BOTH_ALLOWED'
    """
    sym = symbol.upper().strip()
    reference = COMMODITY_ETF_MAP.get(sym)
    if not reference:
        return "BOTH_ALLOWED"

    ref_change = 0.0
    if market_data and reference in market_data:
        ref_change = float(market_data.get(reference, {}).get("change_pct", 0.0) or 0.0)
    else:
        try:
            from strategies.regime_engine import _get_premarket_pct_change
            res = _get_premarket_pct_change(reference)
            pct_val = res[0] if isinstance(res, tuple) else res
            ref_change = float(pct_val or 0.0)
        except Exception:
            ref_change = 0.0

    if ref_change > 1.0:
        logger.info(f"Commodity Guard: {sym} reference {reference} up {ref_change:+.2f}% (> +1.0%) -> LONG_ONLY")
        return "LONG_ONLY"
    elif ref_change < -1.0:
        logger.info(f"Commodity Guard: {sym} reference {reference} down {ref_change:+.2f}% (< -1.0%) -> SHORT_ONLY")
        return "SHORT_ONLY"

    return "BOTH_ALLOWED"
