"""
base.py -- Base signal dataclass and strategy interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import pandas as pd

from strategies.constants import StrategyName, OrderIntent, EXIT_INTENTS


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
    intent: OrderIntent = OrderIntent.ENTRY
    metadata: dict = field(default_factory=dict)
    signal_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    bar_timestamp: Optional[str] = None

    @property
    def signal_hash(self) -> str:
        """Deterministic SHA-256 digest of core signal attributes.
        Uses fixed 4-decimal formatting for floats and avoids wall-clock non-determinism.
        """
        import hashlib

        def _fmt(val: Optional[float]) -> str:
            if val is None:
                return "NONE"
            try:
                return f"{round(float(val), 4):.4f}"
            except (TypeError, ValueError):
                return "0.0000"

        strat = self.strategy.value if hasattr(self.strategy, "value") else str(self.strategy)
        bar_ts = str(self.bar_timestamp or "").strip()
        raw = "|".join([
            strat.upper(),
            self.symbol.upper().strip(),
            self.side.upper().strip(),
            bar_ts,
            _fmt(self.entry_price),
            _fmt(self.stop_price),
            _fmt(self.target_price),
            _fmt(self.atr),
            _fmt(self.risk_per_share),
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

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


@dataclass(frozen=True)
class EntryReadiness:
    """Cycle-bound execution context token required for entry order placement."""
    entry_allowed: bool
    suppression_reason: Optional[str]
    cycle_id: str
    bar_ts: str
    created_at: float
    expires_at: float

    def is_expired(self, now: Optional[float] = None) -> bool:
        import time
        t = time.time() if now is None else now
        return t > self.expires_at
