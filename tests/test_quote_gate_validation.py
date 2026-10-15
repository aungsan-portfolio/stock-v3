"""
tests/test_quote_gate_validation.py — Strict Verification of Quote Gate & Freshness Telemetry.

Verifies:
1. Fresh narrow quote acceptance (normal age, e.g. 1.0s, spread <= 0.02).
2. Sub-second future clock skew acceptance (e.g. age_sec = -0.5s, -0.85s within future_tolerance).
3. Excessive future timestamp rejection (age_sec < -5.0s -> 'future_timestamp').
4. Stale quote rejection (age_sec > 10.0s -> 'stale_quote') and fallback attribution.
5. Wide spread rejection (spread > 0.0200 -> 'spread_too_wide') and fallback attribution.
6. Missing timestamp rejection (timestamp is None -> 'missing_timestamp').
7. Invalid bid/ask rejection (bid <= 0, ask <= 0, or bid >= ask -> 'invalid_bid_ask').
8. Timestamp format handling (tz-aware datetime, naive datetime, ISO strings, epoch s/ms/ns, to_pydatetime duck type).
9. Two-sided requirement enforcement in validate_quote (rejects trade/bar/yfinance fallbacks for order entry, preserves primary reason).
10. Alpaca SDK Quote object compatibility.
"""

import datetime
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from alpaca_bridge import AlpacaBridge, QuoteDecision


class MockAlpacaQuote:
    """Simulates alpaca.data.models.quotes.Quote object from alpaca-py."""
    def __init__(self, bid_price: float, ask_price: float, timestamp):
        self.bid_price = bid_price
        self.ask_price = ask_price
        self.bid_size = 100
        self.ask_size = 100
        self.timestamp = timestamp


@pytest.fixture
def mock_bridge():
    """Returns an AlpacaBridge in dry_run mode with mocked client and data client."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._data_client = MagicMock()
    return bridge


def test_fresh_narrow_quote_accepted(mock_bridge):
    """Test 1: Fresh narrow quote (spread <= 2%, age 1.0s) accepted as direct 'alpaca' quote."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    q = MockAlpacaQuote(
        bid_price=100.00,
        ask_price=100.04,
        timestamp=now_utc - datetime.timedelta(seconds=1.0),
    )
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"NVDA": q}

    decision = mock_bridge.get_order_quote("NVDA")
    assert decision.ok is True
    assert decision.source == "alpaca"
    assert decision.reason == "ok"
    assert abs(decision.price - 100.02) < 1e-4

    is_valid, price, reason = mock_bridge.validate_quote("NVDA", require_two_sided=True)
    assert is_valid is True
    assert abs(price - 100.02) < 1e-4
    assert reason == "ok"


def test_subsecond_future_clock_skew_accepted(mock_bridge):
    """Test 2: Sub-second future clock skew (e.g. -0.85s) within tolerance accepted."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    # Quote timestamp is 850ms in future relative to client machine clock
    q = MockAlpacaQuote(
        bid_price=250.00,
        ask_price=250.05,
        timestamp=now_utc + datetime.timedelta(milliseconds=850),
    )
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"AMZN": q}

    decision = mock_bridge.get_order_quote("AMZN")
    assert decision.ok is True
    assert decision.source == "alpaca"
    assert decision.reason == "ok"
    assert abs(decision.price - 250.025) < 1e-4

    is_valid, price, reason = mock_bridge.validate_quote("AMZN", require_two_sided=True)
    assert is_valid is True
    assert abs(price - 250.025) < 1e-4
    assert reason == "ok"


def test_excessive_future_timestamp_rejected(mock_bridge):
    """Test 3: Excessive future timestamp (> 5.0s ahead) rejected with 'future_timestamp'."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    q = MockAlpacaQuote(
        bid_price=100.00,
        ask_price=100.04,
        timestamp=now_utc + datetime.timedelta(seconds=10.0),
    )
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"AAPL": q}
    mock_bridge._data_client.get_stock_latest_trade.side_effect = Exception("No trades")
    mock_bridge._data_client.get_stock_bars.side_effect = Exception("No bars")

    with patch("yfinance.Ticker", side_effect=Exception("No yf")):
        decision = mock_bridge.get_order_quote("AAPL")
        assert decision.ok is False
        assert "future_timestamp" in decision.reason

        is_valid, price, reason = mock_bridge.validate_quote("AAPL", require_two_sided=True)
        assert is_valid is False
        assert "future_timestamp" in reason


def test_stale_quote_rejected_and_fallback_labeled(mock_bridge):
    """Test 4: Stale quote (> 10.0s old) rejected with 'stale_quote' and fallback labeled correctly."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    q = MockAlpacaQuote(
        bid_price=100.00,
        ask_price=100.04,
        timestamp=now_utc - datetime.timedelta(seconds=15.0),
    )
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"MSFT": q}

    # Simulate trade fallback succeeding
    trade_mock = SimpleNamespace(price=100.03)
    mock_bridge._data_client.get_stock_latest_trade.return_value = {"MSFT": trade_mock}

    decision = mock_bridge.get_order_quote("MSFT")
    assert decision.ok is True
    assert decision.source == "alpaca_iex_trade_fallback"
    assert decision.reason == "fallback_from_stale_quote"
    assert decision.price == 100.03

    # validate_quote with require_two_sided=True MUST reject execution with detailed reason
    is_valid, price, reason = mock_bridge.validate_quote("MSFT", require_two_sided=True)
    assert is_valid is False
    assert price == 100.03
    assert "alpaca_iex_trade_fallback" in reason
    assert "fallback_from_stale_quote" in reason

    # validate_quote with require_two_sided=False allows fallback
    is_valid_diag, price_diag, reason_diag = mock_bridge.validate_quote("MSFT", require_two_sided=False)
    assert is_valid_diag is True
    assert price_diag == 100.03
    assert reason_diag == "fallback_from_stale_quote"


def test_wide_spread_rejected(mock_bridge):
    """Test 5: Wide spread (> 2.0%) rejected with 'spread_too_wide'."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    # Bid=100, Ask=103 -> Mid=101.5 -> Spread = 3.0 / 101.5 = 2.955% > 2.0%
    q = MockAlpacaQuote(
        bid_price=100.00,
        ask_price=103.00,
        timestamp=now_utc - datetime.timedelta(seconds=1.0),
    )
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"JPM": q}

    # Simulate trade fallback succeeding
    trade_mock = SimpleNamespace(price=101.50)
    mock_bridge._data_client.get_stock_latest_trade.return_value = {"JPM": trade_mock}

    decision = mock_bridge.get_order_quote("JPM")
    assert decision.ok is True
    assert decision.source == "alpaca_iex_trade_fallback"
    assert decision.reason == "fallback_from_spread_too_wide"
    assert decision.price == 101.50

    is_valid, price, reason = mock_bridge.validate_quote("JPM", require_two_sided=True)
    assert is_valid is False
    assert "spread_too_wide" in reason


def test_missing_timestamp_rejected(mock_bridge):
    """Test 6: Missing timestamp rejected with 'missing_timestamp'."""
    q = MockAlpacaQuote(
        bid_price=100.00,
        ask_price=100.04,
        timestamp=None,
    )
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"BAC": q}
    mock_bridge._data_client.get_stock_latest_trade.side_effect = Exception("No trades")
    mock_bridge._data_client.get_stock_bars.side_effect = Exception("No bars")

    with patch("yfinance.Ticker", side_effect=Exception("No yf")):
        decision = mock_bridge.get_order_quote("BAC")
        assert decision.ok is False
        assert "missing_timestamp" in decision.reason


def test_invalid_bid_ask_rejected(mock_bridge):
    """Test 7: Invalid bid/ask prices rejected with 'invalid_bid_ask'."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    # 7a: Zero bid
    q_zero_bid = MockAlpacaQuote(0.0, 100.0, now_utc)
    # 7b: Zero ask
    q_zero_ask = MockAlpacaQuote(100.0, 0.0, now_utc)
    # 7c: Inverted/crossed book
    q_crossed = MockAlpacaQuote(101.0, 100.0, now_utc)

    mock_bridge._data_client.get_stock_latest_trade.side_effect = Exception("No trades")
    mock_bridge._data_client.get_stock_bars.side_effect = Exception("No bars")

    with patch("yfinance.Ticker", side_effect=Exception("No yf")):
        for q_bad in [q_zero_bid, q_zero_ask, q_crossed]:
            mock_bridge._data_client.get_stock_latest_quote.return_value = {"TEST": q_bad}
            decision = mock_bridge.get_order_quote("TEST")
            assert decision.ok is False
            assert "invalid_bid_ask" in decision.reason


def test_timestamp_formats_supported(mock_bridge):
    """Test 8: Various timestamp formats (ISO, naive datetime, epoch s/ms/ns, duck type) handled."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    epoch_sec = now_utc.timestamp()

    class PandasTimestampDuck:
        def __init__(self, dt):
            self._dt = dt
        def to_pydatetime(self):
            return self._dt

    valid_timestamps = [
        now_utc,  # tz-aware datetime
        now_utc.replace(tzinfo=None),  # naive datetime
        now_utc.isoformat(),  # ISO-8601 string
        now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),  # ISO string with Z
        epoch_sec,  # epoch float (seconds)
        int(epoch_sec * 1000),  # epoch int (milliseconds)
        int(epoch_sec * 1e9),  # epoch int (nanoseconds)
        PandasTimestampDuck(now_utc),  # duck type
    ]

    for ts in valid_timestamps:
        q = MockAlpacaQuote(bid_price=50.00, ask_price=50.02, timestamp=ts)
        mock_bridge._data_client.get_stock_latest_quote.return_value = {"FMT": q}
        decision = mock_bridge.get_order_quote("FMT")
        assert decision.ok is True, f"Failed for timestamp format: {ts!r} ({type(ts)})"
        assert decision.source == "alpaca"
        assert decision.reason == "ok"


def test_invalid_timestamp_handled(mock_bridge):
    """Test 9: Completely unparseable timestamp rejected with 'invalid_timestamp'."""
    q = MockAlpacaQuote(bid_price=50.00, ask_price=50.02, timestamp="not-a-valid-timestamp")
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"BAD": q}
    mock_bridge._data_client.get_stock_latest_trade.side_effect = Exception("No trades")
    mock_bridge._data_client.get_stock_bars.side_effect = Exception("No bars")

    with patch("yfinance.Ticker", side_effect=Exception("No yf")):
        decision = mock_bridge.get_order_quote("BAD")
        assert decision.ok is False
        assert "invalid_timestamp" in decision.reason


def test_nan_bid_ask_rejected(mock_bridge):
    """Test 10: NaN bid, ask, or both strictly rejected with 'invalid_bid_ask'."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    mock_bridge._data_client.get_stock_latest_trade.side_effect = Exception("No trades")
    mock_bridge._data_client.get_stock_bars.side_effect = Exception("No bars")

    nan_quotes = [
        MockAlpacaQuote(float("nan"), 100.0, now_utc),
        MockAlpacaQuote(100.0, float("nan"), now_utc),
        MockAlpacaQuote(float("nan"), float("nan"), now_utc),
    ]

    with patch("yfinance.Ticker", side_effect=Exception("No yf")):
        for q_nan in nan_quotes:
            mock_bridge._data_client.get_stock_latest_quote.return_value = {"NAN_SYM": q_nan}
            decision = mock_bridge.get_order_quote("NAN_SYM")
            assert decision.ok is False
            assert "invalid_bid_ask" in decision.reason


def test_inf_bid_ask_rejected(mock_bridge):
    """Test 11: +/- Infinity bid or ask strictly rejected with 'invalid_bid_ask'."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    mock_bridge._data_client.get_stock_latest_trade.side_effect = Exception("No trades")
    mock_bridge._data_client.get_stock_bars.side_effect = Exception("No bars")

    inf_quotes = [
        MockAlpacaQuote(float("inf"), 100.0, now_utc),
        MockAlpacaQuote(100.0, float("inf"), now_utc),
        MockAlpacaQuote(float("-inf"), 100.0, now_utc),
        MockAlpacaQuote(100.0, float("-inf"), now_utc),
    ]

    with patch("yfinance.Ticker", side_effect=Exception("No yf")):
        for q_inf in inf_quotes:
            mock_bridge._data_client.get_stock_latest_quote.return_value = {"INF_SYM": q_inf}
            decision = mock_bridge.get_order_quote("INF_SYM")
            assert decision.ok is False
            assert "invalid_bid_ask" in decision.reason


def test_nan_and_inf_trade_fallback_rejected(mock_bridge):
    """Test 12: Trade/Bar/YF fallbacks with NaN/Inf prices are rejected, not admitted."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    # Stale quote triggers fallbacks
    q_stale = MockAlpacaQuote(100.0, 100.04, now_utc - datetime.timedelta(seconds=20.0))
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"FALLBACK_SYM": q_stale}

    # Trade fallback returns NaN
    mock_bridge._data_client.get_stock_latest_trade.return_value = {"FALLBACK_SYM": SimpleNamespace(price=float("nan"))}
    # Bar fallback returns Inf
    mock_bridge._data_client.get_stock_bars.return_value = SimpleNamespace(data={"FALLBACK_SYM": [SimpleNamespace(close=float("inf"))]})

    with patch("yfinance.Ticker") as mock_yf:
        mock_yf.return_value.fast_info.get.return_value = float("nan")
        decision = mock_bridge.get_order_quote("FALLBACK_SYM")
        assert decision.ok is False
        assert "fallback" not in decision.source
        assert "Primary quote rejected: stale_quote" in decision.reason


def test_invalid_tolerance_and_config_bounded(mock_bridge):
    """Test 13: Non-finite or out-of-bounds config thresholds fall back defensively to safe defaults."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    q = MockAlpacaQuote(100.0, 100.04, now_utc)
    mock_bridge._data_client.get_stock_latest_quote.return_value = {"BOUND_SYM": q}

    # Config returns NaN for spread and age
    mock_bridge._c = MagicMock(side_effect=lambda key, default: float("nan"))

    with patch("config.QUOTE_FUTURE_TOLERANCE_SECONDS", float("nan")):
        decision = mock_bridge.get_order_quote("BOUND_SYM")
        assert decision.ok is True
        assert decision.reason == "ok"
        assert decision.source == "alpaca"


def test_quote_future_tolerance_boundary_enforcement(mock_bridge):
    """Test 14: Strict boundary check at QUOTE_FUTURE_TOLERANCE_SECONDS = 2.0s."""
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    mock_bridge._data_client.get_stock_latest_trade.side_effect = Exception("No trades")
    mock_bridge._data_client.get_stock_bars.side_effect = Exception("No bars")

    with patch("config.QUOTE_FUTURE_TOLERANCE_SECONDS", 2.0), patch("yfinance.Ticker", side_effect=Exception("No yf")):
        # Within tolerance: 1.5s in future -> Accepted
        q_pass = MockAlpacaQuote(100.0, 100.04, now_utc + datetime.timedelta(seconds=1.5))
        mock_bridge._data_client.get_stock_latest_quote.return_value = {"TOL_SYM": q_pass}
        decision_pass = mock_bridge.get_order_quote("TOL_SYM")
        assert decision_pass.ok is True
        assert decision_pass.reason == "ok"

        # Beyond tolerance: 2.5s in future -> Rejected with future_timestamp
        q_reject = MockAlpacaQuote(100.0, 100.04, now_utc + datetime.timedelta(seconds=2.5))
        mock_bridge._data_client.get_stock_latest_quote.return_value = {"TOL_SYM": q_reject}
        decision_reject = mock_bridge.get_order_quote("TOL_SYM")
        assert decision_reject.ok is False
        assert "future_timestamp" in decision_reject.reason

