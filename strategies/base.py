"""
base.py -- Base signal dataclass and strategy interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import pandas as pd

from strategies.constants import StrategyName


@dataclass
class TradeSignal:
    symbol: str
    strategy: StrategyName
    side: str
    confidence: float
    entry_price: float
    stop_price: float
    target_price: float
    atr: float
    risk_per_share: float
    reason: str
    pattern_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def reward_risk_ratio(self) -> float:
        if self.risk_per_share <= 0:
            return 0.0
        reward = abs(self.target_price - self.entry_price)
        return reward / self.risk_per_share

    @property
    def is_valid(self) -> bool:
        """Basic price-geometry validation shared by all strategies."""
        if self.entry_price <= 0 or self.stop_price <= 0 or self.target_price <= 0:
            return False
        if self.risk_per_share <= 0:
            return False
        if self.reward_risk_ratio < 1.0:
            return False

        side = self.side.upper()
        if side == "BUY":
            return self.stop_price < self.entry_price < self.target_price
        if side == "SELL":
            return self.target_price < self.entry_price < self.stop_price
        return False


class BaseStrategy:
    """Interface every strategy must implement."""

    name: StrategyName = None

    def evaluate(self, symbol: str, df: pd.DataFrame) -> Optional[TradeSignal]:
        raise NotImplementedError
