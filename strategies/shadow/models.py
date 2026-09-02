"""
strategies.shadow.models -- Core Data Models & Enums for P1.6 Shadow Alpha.
"""
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Optional


class TimeBucket(str, Enum):
    OPENING_NOISE = "OPENING_NOISE"  # 09:30 - 10:00 ET
    PRIMARY = "PRIMARY"              # 10:00 - 11:30 ET
    MIDDAY = "MIDDAY"                # 11:30 - 14:00 ET
    LATE_DAY = "LATE_DAY"            # 15:00 - 15:45 ET
    FLATTEN = "FLATTEN"              # 15:45 - 16:00 ET


class AlignmentStatus(str, Enum):
    ALIGNED = "ALIGNED"
    OPPOSED = "OPPOSED"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FALLBACK_MARKET_CONFIRMED = "FALLBACK_MARKET_CONFIRMED"


class ShadowOutcome(str, Enum):
    ACTIVE = "ACTIVE"
    TARGET_HIT = "TARGET_HIT"
    STOPPED_OUT = "STOPPED_OUT"
    TIME_EXIT = "TIME_EXIT"
    AMBIGUOUS_INTRABAR = "AMBIGUOUS_INTRABAR"


class StreamType(str, Enum):
    RESEARCH_ALL = "RESEARCH_ALL"
    PORTFOLIO_SIM = "PORTFOLIO_SIM"


def generate_deterministic_shadow_id(
    strategy_version: str,
    symbol: str,
    side: str,
    signal_bar_close_utc: str,
    session_date_et: str,
) -> str:
    """Generates a deterministic, reproducible SHA-256 ID for a shadow candidate."""
    identity = (
        f"{strategy_version}|{symbol.upper().strip()}|{side.upper().strip()}|"
        f"{signal_bar_close_utc}|{session_date_et}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class ShadowCandidate:
    """Immutable research candidate emitted by a shadow strategy.

    Physically distinct from production TradeSignal to guarantee type-level
    pipeline separation.
    """
    candidate_id: str
    strategy_name: str
    strategy_version: str
    symbol: str
    side: str
    signal_timestamp_utc: str
    signal_bar_start_utc: str
    signal_bar_close_utc: str
    session_date_et: str
    reference_price: float
    stop_price: float
    target_price: float
    risk_per_share: float
    opening_range_high: float
    opening_range_low: float
    relative_volume: float
    breakout_volume_ratio: float
    price_vs_vwap: float
    vwap_slope: float
    ema9: float
    ema20: float
    atr: float
    atr_pct: float
    spread_pct: float
    spy_alignment: AlignmentStatus
    qqq_alignment: AlignmentStatus
    sector_alignment: AlignmentStatus
    catalyst_type: Optional[str]
    premarket_gap_pct: float
    premarket_volume_ratio: float
    market_texture: str
    regime_mode: str
    confidence: float
    context: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        if not isinstance(self.context, MappingProxyType):
            object.__setattr__(self, "context", MappingProxyType(dict(self.context or {})))


@dataclass
class ShadowTrade:
    """Mutable lifecycle state and persistent record for a shadow trade."""
    shadow_trade_id: str
    strategy_name: str
    strategy_version: str
    symbol: str
    side: str
    stream_type: str = StreamType.RESEARCH_ALL.value

    # Timestamps & Time Window
    signal_timestamp_utc: str = ""
    signal_bar_start_utc: str = ""
    signal_bar_close_utc: str = ""
    entry_timestamp_utc: str = ""
    entry_time_bucket: str = TimeBucket.PRIMARY.value
    session_date_et: str = ""

    # Market Regime & Texture
    market_texture: str = "TRENDING"
    regime_mode: str = "RISK_ON"

    # Opening Range Context
    opening_range_high: float = 0.0
    opening_range_low: float = 0.0

    # Entry & Hypothetical Bracket Geometry
    entry_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    risk_per_share: float = 0.0
    quantity_model: int = 100

    # Pricing Source & Frictions
    entry_pricing_source: str = "SYNTHETIC_CLOSE_BPS"
    entry_quote_timestamp_utc: Optional[str] = None
    entry_quote_age_seconds: Optional[float] = None
    entry_reference_close: float = 0.0
    entry_synthetic_cost: float = 0.0
    entry_half_spread_cost: float = 0.0
    entry_slippage_cost: float = 0.0

    # Technical Indicators at Signal
    relative_volume: float = 0.0
    rvol_method: str = "PROXY"
    breakout_volume_ratio: float = 0.0
    price_vs_vwap: float = 0.0
    vwap_slope: float = 0.0
    ema9: float = 0.0
    ema20: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    spread_pct: float = 0.0

    # Benchmark Alignment & Catalyst
    spy_alignment: str = AlignmentStatus.ALIGNED.value
    qqq_alignment: str = AlignmentStatus.ALIGNED.value
    sector_alignment: str = AlignmentStatus.NOT_APPLICABLE.value
    catalyst_type: Optional[str] = None
    premarket_gap_pct: float = 0.0
    premarket_volume_ratio: float = 0.0
    confidence: float = 0.0

    # Lifecycle Excursions
    status: str = ShadowOutcome.ACTIVE.value
    maximum_favorable_excursion_r: float = 0.0
    maximum_adverse_excursion_r: float = 0.0
    mfe_is_upper_bound: bool = True
    mae_is_upper_bound: bool = True
    highest_price_seen: float = 0.0
    lowest_price_seen: float = 0.0
    bars_held: int = 0
    secondary_1_5r_reached: bool = False

    # Outcome & Financial Settlement
    exit_price: Optional[float] = None
    exit_timestamp_utc: Optional[str] = None
    exit_reason: Optional[str] = None
    exit_pricing_source: Optional[str] = None
    exit_synthetic_cost: float = 0.0
    exit_half_spread_cost: float = 0.0
    exit_slippage_cost: float = 0.0

    gross_pnl: float = 0.0
    total_cost: float = 0.0
    net_pnl: float = 0.0
    gross_result_r: float = 0.0
    cost_r: float = 0.0
    net_result_r: float = 0.0
    outcome_class: str = "PENDING"  # WIN, LOSS, BREAKEVEN
    ambiguous_intrabar: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "ShadowTrade":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
