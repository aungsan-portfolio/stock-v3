"""
observability/flight_recorder.py -- In-memory ring buffer logging handler.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


class FlightRecorderHandler(logging.Handler):
    """
    Thread-safe ring buffer over the root logger.
    Captures the last N log events before any critical incident.
    """

    def __init__(self, capacity: int = 500) -> None:
        super().__init__(level=logging.DEBUG)
        self.capacity = capacity
        self.buf: deque[Dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record) if self.formatter else record.getMessage()
            entry = {
                "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "lvl": record.levelname,
                "mod": record.name,
                "msg": msg,
            }
            if record.exc_info and record.exc_text:
                entry["exc"] = record.exc_text
            with self._lock:
                self.buf.append(entry)
        except Exception:
            self.handleError(record)

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a copy of the current in-memory buffer."""
        with self._lock:
            return list(self.buf)

    def dump(self, path: str) -> None:
        """Write all events in the ring buffer to path in JSONL format."""
        events = self.snapshot()
        with open(path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")


_FLIGHT_RECORDER: Optional[FlightRecorderHandler] = None
_INIT_LOCK = threading.Lock()


def get_flight_recorder(capacity: int = 500) -> FlightRecorderHandler:
    global _FLIGHT_RECORDER
    if _FLIGHT_RECORDER is None:
        with _INIT_LOCK:
            if _FLIGHT_RECORDER is None:
                _FLIGHT_RECORDER = FlightRecorderHandler(capacity=capacity)
    return _FLIGHT_RECORDER


def attach_flight_recorder(capacity: int = 500) -> FlightRecorderHandler:
    """Attach the flight recorder singleton to the root logger if not already attached."""
    handler = get_flight_recorder(capacity=capacity)
    root_logger = logging.getLogger()
    if handler not in root_logger.handlers:
        root_logger.addHandler(handler)
    return handler
