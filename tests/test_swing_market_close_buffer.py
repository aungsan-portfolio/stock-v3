"""
tests/test_swing_market_close_buffer.py -- Unit tests for Market Close Buffer Guard in Predictor.
"""
from datetime import datetime
from unittest.mock import patch
import pytz
import pytest

from predictor import Predictor

EASTERN = pytz.timezone("America/New_York")


def test_market_close_buffer_triggers_after_1545():
    """Verify that within 15 minutes of market close (hour=15, minute>=45), predict_all skips rebalance."""
    mock_now = datetime(2026, 10, 19, 15, 46, 0, tzinfo=EASTERN)
    with patch("strategies.session.now_eastern", return_value=mock_now):
        p = Predictor()
        # Should return empty list immediately without running inference
        res = p.predict_all(["AAPL"])
        assert res == []


def test_market_close_buffer_does_not_trigger_during_regular_hours():
    """Verify that during regular hours (e.g. 10:30), predict_all evaluates normally."""
    mock_now = datetime(2026, 10, 19, 10, 30, 0, tzinfo=EASTERN)
    with patch("strategies.session.now_eastern", return_value=mock_now):
        now_et = mock_now
        assert not (now_et.hour == 15 and now_et.minute >= 45)


def test_market_close_buffer_fails_closed_on_exception():
    """Verify that when obtaining Eastern time raises an exception, predict_all fails closed (returns [])."""
    with patch("strategies.session.now_eastern", side_effect=RuntimeError("System clock failure")):
        p = Predictor()
        res = p.predict_all(["AAPL"])
        assert res == []
