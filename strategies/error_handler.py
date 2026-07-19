"""
error_handler.py -- Centralized error recovery for the trading engine.
"""
import logging
import time

import config
from strategies.errors import (
    BrokerConnectionError,
    DailyLossLimitError,
    FlattenRequiredError,
    OrderRejectedError,
    RiskLimitBreachedError,
)

logger = logging.getLogger(__name__)


class TradingErrorHandler:
    """Centralized error counter and recovery dispatcher."""

    def __init__(self, bridge, max_consecutive_errors: int = None, max_reconnect_attempts: int = 3):
        self._bridge = bridge
        self._max_errors = max_consecutive_errors or getattr(config, "ERROR_CIRCUIT_BREAKER_THRESHOLD", 5)
        self._max_reconnect = max_reconnect_attempts
        self._consecutive_errors = 0
        self._reconnect_attempts = 0

    def record(self, exc: Exception) -> None:
        """Record one error. Dispatches to the right recovery path."""
        self._consecutive_errors += 1
        logger.exception("Execution error #%d: %s", self._consecutive_errors, exc)

        if isinstance(exc, (DailyLossLimitError, FlattenRequiredError)):
            logger.error("Risk limit breached — flattening and exiting.")
            self._flatten_and_exit(reason=type(exc).__name__)

        if isinstance(exc, BrokerConnectionError):
            self.maybe_reconnect()
            return

        if self._consecutive_errors >= self._max_errors:
            logger.error("Circuit breaker: %d consecutive errors — flattening and exiting.", self._consecutive_errors)
            self._flatten_and_exit(reason="circuit_breaker")

    def reset(self) -> None:
        """Call after a successful execution to reset the error counter."""
        if self._consecutive_errors > 0:
            logger.debug("Error counter reset after successful execution.")
        self._consecutive_errors = 0
        self._reconnect_attempts = 0

    def maybe_reconnect(self) -> None:
        """Try to reconnect to broker. Exits if max attempts exceeded."""
        if self._reconnect_attempts >= self._max_reconnect:
            logger.error("Max reconnect attempts (%d) exceeded — exiting.", self._max_reconnect)
            self._flatten_and_exit(reason="reconnect_failed")
            return

        self._reconnect_attempts += 1
        wait = getattr(config, "ERROR_RECONNECT_WAIT_SECONDS", 10)
        logger.warning("Reconnect attempt %d/%d in %ds...", self._reconnect_attempts, self._max_reconnect, wait)
        time.sleep(wait)

        if self._bridge.connect():
            logger.info("Reconnected to broker.")
            self._consecutive_errors = 0
        else:
            logger.error("Reconnect attempt %d failed.", self._reconnect_attempts)

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    def _flatten_and_exit(self, reason: str) -> None:
        """Best-effort flatten, then sys.exit(1)."""
        import sys

        logger.critical("Flattening all positions before exit. Reason: %s", reason)
        try:
            is_connected = getattr(self._bridge, "is_connected", False) or getattr(self._bridge, "_connected", False)
            if is_connected:
                self._bridge.flatten_all()
                logger.info("Flatten complete.")
        except Exception as e:
            logger.error("Flatten failed: %s", e)
        sys.exit(1)
