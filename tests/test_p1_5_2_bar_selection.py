"""
tests/test_p1_5_2_bar_selection.py -- Dedicated test suite for P1.5.2.1 completed-bar selection.

Covers all 14 mandated invariants:
 1. test_completed_bar_returned
 2. test_forming_bar_excluded
 3. test_all_bars_forming_raises
 4. test_exact_close_boundary_included
 5. test_one_second_before_close_excluded
 6. test_one_second_after_close_included
 7. test_multiple_forming_bars_skipped
 8. test_grace_boundary
 9. test_timezone_semantics
10. test_single_completed_bar
11. test_selected_timestamp_returned_correctly
12. test_unsorted_index_handled_safely
13. test_duplicate_timestamps_rejected
14. test_integration_with_freshness_validator
"""
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import config
from strategies.intraday_data import (
    CompletedBar,
    _normalize_utc_index,
    _normalize_utc_timestamp,
    _validate_intraday_data,
    select_last_completed_bar,
)


def _make_df(timestamps, closes=None):
    if closes is None:
        closes = [100.0 + i for i in range(len(timestamps))]
    idx = pd.DatetimeIndex(timestamps)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
            "atr": [1.5] * len(closes),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Test 1: Completed bar returned with valid row and timestamps
# ---------------------------------------------------------------------------
def test_completed_bar_returned():
    ts = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    df = _make_df([ts])
    now_utc = pd.Timestamp("2026-09-02 15:26:00", tz="UTC")

    result = select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)
    assert isinstance(result, CompletedBar)
    assert result.start_time == ts
    assert result.close_time == ts + pd.Timedelta(seconds=300)
    assert result.post_close_delay_seconds == 60.0
    assert result.row["close"] == 100.0


# ---------------------------------------------------------------------------
# Test 2: Forming bar excluded, prior completed bar returned
# ---------------------------------------------------------------------------
def test_forming_bar_excluded():
    ts1 = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    ts2 = pd.Timestamp("2026-09-02 15:25:00", tz="UTC")  # Forming at 15:27:25
    df = _make_df([ts1, ts2])
    now_utc = pd.Timestamp("2026-09-02 15:27:25", tz="UTC")

    result = select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)
    assert result.start_time == ts1
    assert result.close_time == pd.Timestamp("2026-09-02 15:25:00", tz="UTC")
    assert result.row["close"] == 100.0


# ---------------------------------------------------------------------------
# Test 3: All bars forming raises ValueError
# ---------------------------------------------------------------------------
def test_all_bars_forming_raises():
    ts = pd.Timestamp("2026-09-02 15:25:00", tz="UTC")
    df = _make_df([ts])
    now_utc = pd.Timestamp("2026-09-02 15:27:00", tz="UTC")  # Closes at 15:30:00

    with pytest.raises(ValueError, match="No completed bars available"):
        select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)


# ---------------------------------------------------------------------------
# Test 4: Exact close boundary included
# ---------------------------------------------------------------------------
def test_exact_close_boundary_included():
    ts = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    df = _make_df([ts])
    now_utc = pd.Timestamp("2026-09-02 15:25:00", tz="UTC")  # Exact close

    result = select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)
    assert result.start_time == ts
    assert result.close_time == now_utc
    assert result.post_close_delay_seconds == 0.0


# ---------------------------------------------------------------------------
# Test 5: One second before close excluded
# ---------------------------------------------------------------------------
def test_one_second_before_close_excluded():
    ts = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    df = _make_df([ts])
    now_utc = pd.Timestamp("2026-09-02 15:24:59", tz="UTC")  # 1s before close

    with pytest.raises(ValueError, match="No completed bars available"):
        select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc, close_tolerance_seconds=0.0)


# ---------------------------------------------------------------------------
# Test 6: One second after close included
# ---------------------------------------------------------------------------
def test_one_second_after_close_included():
    ts = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    df = _make_df([ts])
    now_utc = pd.Timestamp("2026-09-02 15:25:01", tz="UTC")  # 1s after close

    result = select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)
    assert result.start_time == ts
    assert result.post_close_delay_seconds == 1.0


# ---------------------------------------------------------------------------
# Test 7: Multiple forming bars skipped
# ---------------------------------------------------------------------------
def test_multiple_forming_bars_skipped():
    ts1 = pd.Timestamp("2026-09-02 15:15:00", tz="UTC")  # Closes 15:20
    ts2 = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")  # Closes 15:25
    ts3 = pd.Timestamp("2026-09-02 15:25:00", tz="UTC")  # Closes 15:30 (forming)
    ts4 = pd.Timestamp("2026-09-02 15:30:00", tz="UTC")  # Closes 15:35 (future)
    df = _make_df([ts1, ts2, ts3, ts4])
    now_utc = pd.Timestamp("2026-09-02 15:27:00", tz="UTC")

    result = select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)
    assert result.start_time == ts2
    assert result.close_time == pd.Timestamp("2026-09-02 15:25:00", tz="UTC")
    assert result.post_close_delay_seconds == 120.0


# ---------------------------------------------------------------------------
# Test 8: Grace boundary evaluation
# ---------------------------------------------------------------------------
def test_grace_boundary():
    ts = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    df = _make_df([ts])

    # 45s grace boundary: 15:25:45 is exactly 45s post close -> Fresh
    now_fresh = pd.Timestamp("2026-09-02 15:25:45", tz="UTC")
    bar_fresh = select_last_completed_bar(df, interval_seconds=300, now_utc=now_fresh)
    assert bar_fresh.post_close_delay_seconds == 45.0

    # 46s post close -> Stale for 45s grace
    now_stale = pd.Timestamp("2026-09-02 15:25:46", tz="UTC")
    bar_stale = select_last_completed_bar(df, interval_seconds=300, now_utc=now_stale)
    assert bar_stale.post_close_delay_seconds == 46.0


# ---------------------------------------------------------------------------
# Test 9: Timezone semantics (UTC, Eastern, and naive)
# ---------------------------------------------------------------------------
def test_timezone_semantics():
    # 1. Aware UTC
    ts_utc = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    norm_utc = _normalize_utc_timestamp(ts_utc)
    assert str(norm_utc.tz) == "UTC"

    # 2. Aware US/Eastern (11:20:00 EDT == 15:20:00 UTC)
    ts_et = pd.Timestamp("2026-09-02 11:20:00", tz="US/Eastern")
    norm_et = _normalize_utc_timestamp(ts_et)
    assert norm_et == ts_utc

    # 3. Naive with fallback
    ts_naive = pd.Timestamp("2026-09-02 11:20:00")
    with patch.object(config, "INTRADAY_DATA_SOURCE_TIMEZONE", "US/Eastern"):
        norm_naive = _normalize_utc_timestamp(ts_naive)
        assert norm_naive == ts_utc

    # 4. Naive without configured timezone raises ValueError
    with patch.object(config, "INTRADAY_DATA_SOURCE_TIMEZONE", None):
        with pytest.raises(ValueError, match="must be timezone-aware"):
            _normalize_utc_timestamp(ts_naive)


# ---------------------------------------------------------------------------
# Test 10: Single completed bar
# ---------------------------------------------------------------------------
def test_single_completed_bar():
    ts = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    df = _make_df([ts])
    now_utc = pd.Timestamp("2026-09-02 15:25:30", tz="UTC")

    result = select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)
    assert result.start_time == ts
    assert result.post_close_delay_seconds == 30.0


# ---------------------------------------------------------------------------
# Test 11: Selected timestamp returned correctly
# ---------------------------------------------------------------------------
def test_selected_timestamp_returned_correctly():
    ts_prev = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    ts_forming = pd.Timestamp("2026-09-02 15:25:00", tz="UTC")
    df = _make_df([ts_prev, ts_forming], closes=[369.0, 370.5])
    now_utc = pd.Timestamp("2026-09-02 15:27:25", tz="UTC")

    result = select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)
    assert result.start_time == ts_prev
    assert result.close_time == pd.Timestamp("2026-09-02 15:25:00", tz="UTC")
    assert result.row["close"] == 369.0
    assert result.row["close"] != 370.5


# ---------------------------------------------------------------------------
# Test 12: Unsorted index handled safely
# ---------------------------------------------------------------------------
def test_unsorted_index_handled_safely():
    ts1 = pd.Timestamp("2026-09-02 15:25:00", tz="UTC")
    ts2 = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")  # Unsorted!
    df = _make_df([ts1, ts2])
    now_utc = pd.Timestamp("2026-09-02 15:30:00", tz="UTC")

    with pytest.raises(ValueError, match="Intraday bar index must be monotonic increasing"):
        select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)


# ---------------------------------------------------------------------------
# Test 13: Duplicate timestamps rejected
# ---------------------------------------------------------------------------
def test_duplicate_timestamps_rejected():
    ts = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    df = _make_df([ts, ts])  # Duplicate!
    now_utc = pd.Timestamp("2026-09-02 15:30:00", tz="UTC")

    with pytest.raises(ValueError, match="Intraday bar index contains duplicate timestamps"):
        select_last_completed_bar(df, interval_seconds=300, now_utc=now_utc)


# ---------------------------------------------------------------------------
# Test 14: Integration test with freshness validator
# ---------------------------------------------------------------------------
def test_integration_with_freshness_validator():
    """Live September 2 audit scenario:
    15:20 completed bar
    15:25 forming bar
    Now is 15:27:25
    - POSITION_MONITORING: 15:20 bar selected, delay=145s <= 360s -> VALID
    - ENTRY_SIGNAL: 15:20 bar selected, delay=145s > 45s -> STALE (quarantined)
    - All bars forming: FUTURE-BAR-GUARD triggers fail-closed
    """
    ts_comp = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    ts_form = pd.Timestamp("2026-09-02 15:25:00", tz="UTC")
    df = _make_df([ts_comp, ts_form])

    now_utc = pd.Timestamp("2026-09-02 15:27:25", tz="UTC")

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("pandas.Timestamp.now", return_value=now_utc):

        # Case A: Position Monitoring accepts the 145s delay as fresh ATR
        res_pos = _validate_intraday_data(
            df.copy(), symbol="AVGO", interval="5m", validate_staleness=True, purpose="POSITION_MONITORING"
        )
        assert not res_pos.empty, "Position monitoring must accept valid completed bar during mid-bar polling"

        # Case B: Entry Signal rejects 145s delay because 45s grace exceeded for new entries
        res_entry = _validate_intraday_data(
            df.copy(), symbol="AVGO", interval="5m", validate_staleness=True, purpose="ENTRY_SIGNAL"
        )
        assert res_entry.empty, "Entry signal must reject completed bar when post-close delay > 45s"

    # Case C: All bars in the future fails closed under FUTURE-BAR-GUARD
    df_future = _make_df([ts_form])  # Only 15:25 bar when now is 15:20
    now_past = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    with patch("strategies.session.is_market_open", return_value=True), \
         patch("pandas.Timestamp.now", return_value=now_past):
        res_fut = _validate_intraday_data(
            df_future.copy(), symbol="AVGO", interval="5m", validate_staleness=True, purpose="ENTRY_SIGNAL"
        )
        assert res_fut.empty, "Future-dated bars must fail closed"


# ---------------------------------------------------------------------------
# Test 15: get_intraday_features truncates at completed bar
# ---------------------------------------------------------------------------
def test_get_intraday_features_truncates_at_completed_bar():
    from strategies.intraday_data import get_intraday_features

    ts_comp = pd.Timestamp("2026-09-02 15:20:00", tz="UTC")
    ts_form = pd.Timestamp("2026-09-02 15:25:00", tz="UTC")
    df = _make_df([ts_comp, ts_form], closes=[370.0, 372.0])

    now_utc = pd.Timestamp("2026-09-02 15:27:25", tz="UTC")

    with patch("strategies.intraday_data.fetch_intraday", return_value=df), \
         patch("pandas.Timestamp.now", return_value=now_utc):
        feat_df = get_intraday_features("AVGO", interval="5m", purpose="POSITION_MONITORING")

        assert not feat_df.empty
        # The last row of feat_df must be the completed bar (15:20), NOT the forming bar (15:25)
        assert feat_df.index[-1] == ts_comp
        assert feat_df["close"].iloc[-1] == 370.0

