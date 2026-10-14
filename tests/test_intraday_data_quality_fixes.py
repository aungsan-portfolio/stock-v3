"""
tests/test_intraday_data_quality_fixes.py -- Regression tests for the 5 intraday-data quality and concurrency fixes.

Covers:
 1. test_bounded_slot_release_on_queued_cancellation: Verifies that cancelling a queued request releases the semaphore permit.
 2. test_forming_bar_not_selected_before_close: Verifies that forming candle is excluded when close is in the future even with tolerance.
 3. test_bar_end_utc_alignment: Verifies bar_start_utc and bar_end_utc attrs reflect bar start and bar close (start + interval).
 4. test_breadth_telemetry_isolation_in_batch_fetch: Verifies REGIME_BREADTH purpose routes batch telemetry to _breadth_telemetry without polluting _feed_telemetry.
 5. test_breadth_no_serial_fallback_when_batch_enabled: Verifies calculate_regime_market_breadth does not do serial fetch_intraday fallback when use_batch is True.
"""
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from strategies.intraday_data import (
    _alpaca_call_slots,
    _breadth_telemetry,
    _breadth_telemetry_lock,
    _feed_telemetry,
    _feed_telemetry_lock,
    _sdk_get_stock_bars_soft_bounded,
    fetch_batch_intraday_alpaca,
    get_intraday_feed_telemetry,
    reset_intraday_feed_telemetry,
    select_last_completed_bar,
    clear_cache,
)
from strategies.regime_engine import compute_vwap_breadth


# ---------------------------------------------------------------------------
# Test 1: Concurrency slot leak on cancellation
# ---------------------------------------------------------------------------
def test_bounded_slot_release_on_queued_cancellation():
    """Verify that if a queued request is cancelled via timeout, the acquired slot is released."""
    initial_slots = _alpaca_call_slots._value

    # We mock client.get_stock_bars to simulate a slow or blocked worker
    block_event = threading.Event()

    def slow_worker(req):
        block_event.wait(timeout=2.0)
        return MagicMock()

    mock_client = MagicMock()
    mock_client.get_stock_bars.side_effect = slow_worker

    # Fill up worker slots to force next request into the queue, or directly test soft-bounded timeout
    with patch("strategies.intraday_data.ALPACA_FUTURE_TIMEOUT_SECONDS", 0.05):
        try:
            _sdk_get_stock_bars_soft_bounded(mock_client, MagicMock(), timeout=0.05)
        except Exception:
            pass
        finally:
            block_event.set()

    # Allow background threads to settle
    time.sleep(0.2)
    final_slots = _alpaca_call_slots._value
    assert final_slots == initial_slots, f"Slot leaked: initial={initial_slots}, final={final_slots}"


# ---------------------------------------------------------------------------
# Test 2: Forming candle not selected before close
# ---------------------------------------------------------------------------
def test_forming_bar_not_selected_before_close():
    """Candle starting at 15:20 with 5m interval closes at 15:25:00.
    At 15:24:59 (1s before close), it must NOT be selected, even if close_tolerance_seconds > 0.
    """
    ts = pd.Timestamp("2026-10-14 15:20:00", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [500],
        },
        index=pd.DatetimeIndex([ts]),
    )

    one_sec_before_close = pd.Timestamp("2026-10-14 15:24:59", tz="UTC")
    with pytest.raises(ValueError, match="No completed bars available"):
        select_last_completed_bar(
            df,
            interval_seconds=300,
            now_utc=one_sec_before_close,
            close_tolerance_seconds=5.0,  # Positive tolerance must not include future close
        )

    exact_close = pd.Timestamp("2026-10-14 15:25:00", tz="UTC")
    bar = select_last_completed_bar(
        df,
        interval_seconds=300,
        now_utc=exact_close,
        close_tolerance_seconds=0.0,
    )
    assert bar is not None
    assert bar.start_time == ts
    assert bar.close_time == exact_close


# ---------------------------------------------------------------------------
# Test 3: bar_start_utc and bar_end_utc timestamp alignment
# ---------------------------------------------------------------------------
def test_bar_end_utc_alignment():
    """Verify that bar_start_utc is bar start and bar_end_utc is bar close (start + 5m)."""
    clear_cache()
    ts = pd.Timestamp("2026-10-14 15:20:00", tz="UTC")
    raw_df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [500],
        },
        index=pd.DatetimeIndex([ts]),
    )

    mock_bars = MagicMock()
    mock_bars.df = raw_df

    with patch("strategies.intraday_data._get_alpaca_credentials", return_value=("mock_k", "mock_s")), \
         patch("strategies.intraday_data._get_alpaca_client") as mock_get_client, \
         patch("strategies.intraday_data._read_cache", return_value=None), \
         patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded", return_value=mock_bars), \
         patch("strategies.intraday_data._validate_intraday_data", side_effect=lambda df, *args, **kwargs: df):

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        results = fetch_batch_intraday_alpaca(["TESTSYM3"], interval="5m", lookback_days=1, purpose="ENTRY_SIGNAL")
        assert "TESTSYM3" in results
        df = results["TESTSYM3"]
        assert "bar_start_utc" in df.attrs
        assert "bar_end_utc" in df.attrs

        expected_start = ts.isoformat()
        expected_end = (ts + pd.Timedelta(minutes=5)).isoformat()
        assert df.attrs["bar_start_utc"] == expected_start
        assert df.attrs["bar_end_utc"] == expected_end


# ---------------------------------------------------------------------------
# Test 4: Breadth telemetry isolation in batch fetch
# ---------------------------------------------------------------------------
def test_breadth_telemetry_isolation_in_batch_fetch():
    """Verify that fetch_batch_intraday_alpaca with purpose='REGIME_BREADTH' updates _breadth_telemetry,
    leaving _feed_telemetry unchanged."""
    reset_intraday_feed_telemetry()
    clear_cache()

    raw_df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [500],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-10-14 15:20:00", tz="UTC")]),
    )

    mock_bars = MagicMock()
    mock_bars.df = raw_df

    with patch("strategies.intraday_data._get_alpaca_credentials", return_value=("mock_k", "mock_s")), \
         patch("strategies.intraday_data._get_alpaca_client") as mock_get_client, \
         patch("strategies.intraday_data._read_cache", return_value=None), \
         patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded", return_value=mock_bars), \
         patch("strategies.intraday_data._validate_intraday_data", side_effect=lambda df, *args, **kwargs: df):

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        fetch_batch_intraday_alpaca(["TESTSYM4"], interval="5m", lookback_days=1, purpose="REGIME_BREADTH")

        breadth_telem = get_intraday_feed_telemetry("REGIME_BREADTH")
        feed_telem = get_intraday_feed_telemetry("ENTRY_SIGNAL")

        assert breadth_telem["total_bars_fetched"] == 1
        assert breadth_telem["last_batch_duration_seconds"] > 0
        assert feed_telem["total_bars_fetched"] == 0
        assert feed_telem["last_batch_duration_seconds"] == 0.0


# ---------------------------------------------------------------------------
# Test 5: No serial fallback when use_batch is True in calculate_regime_market_breadth
# ---------------------------------------------------------------------------
def test_breadth_no_serial_fallback_when_batch_enabled():
    """Verify that calculate_regime_market_breadth does not call serial fetch_intraday when use_batch is True."""
    mock_bridge = MagicMock()
    mock_bridge.is_connected = True

    with patch.dict(os.environ, {"REGIME_BREADTH_BATCH_ENABLED": "true"}), \
         patch("strategies.intraday_data.fetch_intraday_batch", return_value=({}, {})) as mock_batch, \
         patch("strategies.intraday_data.fetch_intraday") as mock_serial:

        res = compute_vwap_breadth(["AAPL", "MSFT"], bridge=mock_bridge)
        assert mock_batch.called
        assert not mock_serial.called, "Serial fetch_intraday was called as fallback despite batch being enabled!"
