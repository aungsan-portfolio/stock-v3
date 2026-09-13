"""
scoring.py -- Risk-Adjusted Signal Scoring Engine.
"""
import logging
import config

logger = logging.getLogger(__name__)


from dataclasses import dataclass, asdict


@dataclass
class ConfidenceBreakdown:
    base_confidence: float
    rr_adjustment: float
    volatility_pct: float
    vol_adjustment: float
    regime_adjustment: float
    risk_adjusted_confidence: float


class RiskAdjustedSignalScorer:
    def __init__(self):
        pass

    def score_signal(self, signal) -> float:
        """Alias method calling calculate_risk_adjusted_confidence for backtester compatibility."""
        return self.calculate_risk_adjusted_confidence(signal)

    def calculate_risk_adjusted_confidence(self, signal) -> float:
        """
        Adjusts the raw strategy confidence score based on:
        - Reward/Risk Ratio (capped at 2:1)
        - Volatility (ATR / Entry Price)
        - Market Regime (placeholder)
        """
        base_conf = float(getattr(signal, "confidence", 0.5))

        # Risk-reward ratio adjustment (capped at 2:1)
        rr_ratio = float(getattr(signal, "reward_risk_ratio", 1.0))
        rr_adjustment = min(rr_ratio / 2.0, 1.0)

        # Volatility adjustment (higher volatility = lower confidence)
        entry_price = float(getattr(signal, "entry_price", 0.0))
        atr = float(getattr(signal, "atr", 0.0))
        if entry_price > 0:
            volatility_pct = atr / entry_price
        else:
            volatility_pct = 0.0

        penalty_mult = float(getattr(config, "VOLATILITY_PENALTY_MULTIPLIER", 4.0))
        vol_adjustment = max(0.5, 1.0 - (volatility_pct * penalty_mult))

        # Market regime adjustment
        regime_adjustment = float(self.get_market_regime_adjustment(getattr(signal, "symbol", "")))

        # Combined multiplier
        adjusted_conf = base_conf * rr_adjustment * vol_adjustment * regime_adjustment
        final_conf = min(adjusted_conf, 1.0)

        # Persist detailed confidence breakdown in signal metadata
        if not hasattr(signal, "metadata") or signal.metadata is None:
            signal.metadata = {}

        breakdown = ConfidenceBreakdown(
            base_confidence=base_conf,
            rr_adjustment=rr_adjustment,
            volatility_pct=volatility_pct,
            vol_adjustment=vol_adjustment,
            regime_adjustment=regime_adjustment,
            risk_adjusted_confidence=final_conf,
        )
        signal.metadata["confidence_breakdown"] = asdict(breakdown)
        signal.metadata["base_confidence"] = base_conf
        signal.metadata["rr_adjustment"] = rr_adjustment
        signal.metadata["volatility_pct"] = volatility_pct
        signal.metadata["vol_adjustment"] = vol_adjustment
        signal.metadata["regime_adjustment"] = regime_adjustment
        signal.metadata["risk_adjusted_confidence"] = final_conf

        return final_conf

    def get_market_regime_adjustment(self, symbol: str) -> float:
        """
        Placeholder for Phase 3 Market Regime detection.
        Returns 1.0 for now.
        """
        return 1.0
