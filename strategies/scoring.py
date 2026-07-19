"""
scoring.py -- Risk-Adjusted Signal Scoring Engine.
"""
import logging
import config

logger = logging.getLogger(__name__)


class RiskAdjustedSignalScorer:
    def __init__(self):
        pass

    def calculate_risk_adjusted_confidence(self, signal) -> float:
        """
        Adjusts the raw strategy confidence score based on:
        - Reward/Risk Ratio (capped at 2:1)
        - Volatility (ATR / Entry Price)
        - Market Regime (placeholder)
        """
        base_conf = signal.confidence
        
        # Risk-reward ratio adjustment (capped at 2:1)
        rr_ratio = signal.reward_risk_ratio
        rr_adjustment = min(rr_ratio / 2.0, 1.0)
        
        # Volatility adjustment (higher volatility = lower confidence)
        if signal.entry_price > 0:
            volatility_pct = signal.atr / signal.entry_price
        else:
            volatility_pct = 0.0
            
        penalty_mult = getattr(config, "VOLATILITY_PENALTY_MULTIPLIER", 4.0)
        vol_adjustment = max(0.5, 1.0 - (volatility_pct * penalty_mult))
        
        # Market regime adjustment
        regime_adjustment = self.get_market_regime_adjustment(signal.symbol)
        
        # Combined multiplier
        adjusted_conf = base_conf * rr_adjustment * vol_adjustment * regime_adjustment
        
        return min(adjusted_conf, 1.0)
    
    def get_market_regime_adjustment(self, symbol: str) -> float:
        """
        Placeholder for Phase 3 Market Regime detection.
        Returns 1.0 for now.
        """
        return 1.0
