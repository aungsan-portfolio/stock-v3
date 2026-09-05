"""
tests/test_execution_quote.py -- Targeted test suite for execution_quote hardening.
"""
import math
import time
from decimal import Decimal
import pytest

from strategies.execution_quote import (
    validate_execution_quote,
    _parse_bool,
    BrokerQuote,
    QuoteValidationResult,
)


class DummyBridge:
    def __init__(self, quote_data):
        self.quote_data = quote_data

    def get_quote(self, symbol):
        if isinstance(self.quote_data, Exception):
            raise self.quote_data
        return self.quote_data


def test_normal_fresh_quote_remains_valid():
    """Verify normal fresh quote succeeds and returns valid=True."""
    raw = {
        "symbol": "AAPL",
        "bid": 150.00,
        "ask": 150.05,
        "last": 150.02,
        "timestamp": time.time() - 0.5,
        "is_realtime": True,
    }
    res = validate_execution_quote(DummyBridge(raw), "AAPL", decision_price=150.00)
    assert res.valid is True
    assert res.accepted is True
    assert res.quote is not None
    assert res.quote.bid == 150.00
    assert res.quote.ask == 150.05


def test_malformed_numeric_quote_fails_closed():
    """Verify malformed unparseable numeric values return valid=False instead of raising."""
    for malformed_bid in ["N/A", object(), "invalid"]:
        raw = {
            "symbol": "AAPL",
            "bid": malformed_bid,
            "ask": 150.05,
            "last": 150.02,
            "timestamp": time.time(),
        }
        res = validate_execution_quote(DummyBridge(raw), "AAPL")
        assert res.valid is False
        assert "Malformed broker quote" in res.reason


def test_nan_bid_fails_closed():
    """Verify NaN bid or ask returns valid=False."""
    raw_nan_str = {
        "symbol": "AAPL",
        "bid": float("nan"),
        "ask": 150.05,
        "last": 150.02,
        "timestamp": time.time(),
    }
    res = validate_execution_quote(DummyBridge(raw_nan_str), "AAPL")
    assert res.valid is False
    assert "Non-finite broker quote fields" in res.reason
    assert "bid" in res.reason

    raw_nan_decimal = {
        "symbol": "AAPL",
        "bid": Decimal("NaN"),
        "ask": 150.05,
        "last": 150.02,
        "timestamp": time.time(),
    }
    res2 = validate_execution_quote(DummyBridge(raw_nan_decimal), "AAPL")
    assert res2.valid is False
    assert "Non-finite broker quote fields" in res2.reason or "Malformed" in res2.reason


def test_infinite_ask_fails_closed():
    """Verify Inf or -Inf values return valid=False."""
    raw = {
        "symbol": "AAPL",
        "bid": 150.00,
        "ask": float("inf"),
        "last": 150.02,
        "timestamp": time.time(),
    }
    res = validate_execution_quote(DummyBridge(raw), "AAPL")
    assert res.valid is False
    assert "Non-finite broker quote fields" in res.reason
    assert "ask" in res.reason


def test_future_timestamp_fails_closed():
    """Verify quote with timestamp significantly in the future is rejected."""
    raw = {
        "symbol": "AAPL",
        "bid": 150.00,
        "ask": 150.05,
        "last": 150.02,
        "timestamp": time.time() + 10.0,
    }
    res = validate_execution_quote(DummyBridge(raw), "AAPL")
    assert res.valid is False
    assert "in the future" in res.reason


def test_small_clock_skew_is_allowed():
    """Verify small clock skew (<= 2.0s in future) is accepted."""
    raw = {
        "symbol": "AAPL",
        "bid": 150.00,
        "ask": 150.05,
        "last": 150.02,
        "timestamp": time.time() + 0.8,
    }
    res = validate_execution_quote(DummyBridge(raw), "AAPL", decision_price=150.00)
    assert res.valid is True


def test_string_false_realtime_is_rejected():
    """Verify string 'false', '0', 'no' strictly parsed as not real-time."""
    for false_val in ["false", "False", "FALSE", "0", "no", 0, False]:
        raw = {
            "symbol": "AAPL",
            "bid": 150.00,
            "ask": 150.05,
            "last": 150.02,
            "timestamp": time.time(),
            "is_realtime": false_val,
        }
        res = validate_execution_quote(DummyBridge(raw), "AAPL")
        assert res.valid is False
        assert "not real-time" in res.reason

    raw_invalid = {
        "symbol": "AAPL",
        "bid": 150.00,
        "ask": 150.05,
        "last": 150.02,
        "timestamp": time.time(),
        "is_realtime": "maybe",
    }
    res_inv = validate_execution_quote(DummyBridge(raw_invalid), "AAPL")
    assert res_inv.valid is False
    assert "Invalid is_realtime flag" in res_inv.reason


def test_wide_spread_reason_contains_full_context():
    """Verify wide spread message includes bid, ask, mid, spread_dollars, age."""
    raw = {
        "symbol": "NVDA",
        "bid": 95.00,
        "ask": 105.00,
        "last": 100.00,
        "timestamp": time.time() - 1.25,
    }
    res = validate_execution_quote(DummyBridge(raw), "NVDA", max_spread_pct=1.0)
    assert res.valid is False
    assert "Spread too wide for NVDA" in res.reason
    assert "bid=$95.00" in res.reason
    assert "ask=$105.00" in res.reason
    assert "mid=$100.00" in res.reason
    assert "spread_dollars=$10.0000" in res.reason
    assert "age=" in res.reason


def test_invalid_config_arguments_fail_closed():
    """Verify invalid/non-finite/negative config parameters fail closed."""
    raw = {
        "symbol": "AAPL",
        "bid": 150.00,
        "ask": 150.05,
        "last": 150.02,
        "timestamp": time.time(),
    }
    bridge = DummyBridge(raw)

    assert validate_execution_quote(bridge, "AAPL", max_spread_pct=-1.0).valid is False
    assert validate_execution_quote(bridge, "AAPL", max_age_seconds=0.0).valid is False
    assert validate_execution_quote(bridge, "AAPL", max_age_seconds=float("nan")).valid is False
    assert validate_execution_quote(bridge, "AAPL", decision_price="invalid").valid is False
