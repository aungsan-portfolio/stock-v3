"""
errors.py -- Custom exceptions for the day trading engine.
"""

class DayTradingError(Exception):
    """Base exception for all day trading engine errors."""


class DataUnavailableError(DayTradingError):
    """Raised when market data cannot be fetched or is stale."""


class InsufficientDataError(DayTradingError):
    """Raised when not enough bars are available for a calculation."""


class BrokerConnectionError(DayTradingError):
    """Raised when broker connection fails or times out."""


class OrderRejectedError(DayTradingError):
    """Raised when the broker rejects an order."""


class RiskLimitBreachedError(DayTradingError):
    """Raised when a risk rule blocks a trade."""


class PDTViolationError(RiskLimitBreachedError):
    """Raised when a trade would violate PDT rules."""


class DailyLossLimitError(RiskLimitBreachedError):
    """Raised when daily loss limit is reached."""


class FlattenRequiredError(RiskLimitBreachedError):
    """Raised when positions must be flattened before market close."""


class SessionClosedError(DayTradingError):
    """Raised when trying to trade outside market hours."""
