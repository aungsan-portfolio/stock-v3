from enum import Enum

class StrategyName(str, Enum):
    MOMENTUM_SCALP = "MOMENTUM_SCALP"
    ORB = "ORB"
    VWAP_BOUNCE = "VWAP_BOUNCE"
    GAP_AND_GO = "GAP_AND_GO"
    CANDLESTICK = "CANDLESTICK"
