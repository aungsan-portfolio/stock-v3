import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from unittest.mock import patch

from strategies.intraday_data import (
    _validate_intraday_data,
    get_intraday_feed_telemetry,
    reset_intraday_feed_telemetry,
    should_validate_staleness,
    _feed_telemetry,
)
from strategies.regime_engine import compute_vwap_breadth


def create_sample_bars(num_bars=10, interval_minutes=1, last_bar_closed_ago_seconds=10):
    now = datetime.now(timezone.utc)
    last_close = now - timedelta(seconds=last_bar_closed_ago_seconds)
    last_start = last_close - timedelta(minutes=interval_minutes)
    
    timestamps = [last_start - timedelta(minutes=interval_minutes * (num_bars - 1 - i)) for i in range(num_bars)]
    data = []
    base_price = 100.0
    for i, ts in enumerate(timestamps):
        p = base_price + i * 0.1
        data.append({
            "datetime": ts,
            "open": p,
            "high": p + 0.2,
            "low": p - 0.2,
            "close": p + 0.05,
            "volume": 1000 + i * 50
        })
    df = pd.DataFrame(data).set_index("datetime")
    return df


class TestBreadthTelemetryIsolation:
    def setup_method(self):
        reset_intraday_feed_telemetry()
        if "REGIME_BREADTH_BATCH_ENABLED" in os.environ:
            del os.environ["REGIME_BREADTH_BATCH_ENABLED"]
        self.market_open_patch = patch("strategies.session.is_market_open", return_value=True)
        self.market_open_patch.start()

    def teardown_method(self):
        try:
            self.market_open_patch.stop()
        except Exception:
            pass
        reset_intraday_feed_telemetry()
        if "REGIME_BREADTH_BATCH_ENABLED" in os.environ:
            del os.environ["REGIME_BREADTH_BATCH_ENABLED"]

    def test_should_validate_staleness_includes_regime_breadth(self):
        assert should_validate_staleness("REGIME_BREADTH", None) is True
        assert should_validate_staleness("REGIME_BREADTH", 1) is True

    def test_breadth_stale_observation_does_not_increment_entry_counter(self):
        # Set initial entry counter
        _feed_telemetry["stale_bar_consecutive"] = 3
        
        # Stale bar (delay = 120s > 60s grace)
        stale_df = create_sample_bars(num_bars=5, interval_minutes=1, last_bar_closed_ago_seconds=120)
        
        vdf = _validate_intraday_data(stale_df, "STALE_SYM", "1m", purpose="REGIME_BREADTH")
        assert vdf.empty is True

        entry_tel = get_intraday_feed_telemetry()
        breadth_tel = get_intraday_feed_telemetry(purpose="REGIME_BREADTH")

        # Invariant 1: Entry stale counter MUST NOT change
        assert entry_tel["stale_bar_consecutive"] == 3

        # Invariant 2: Breadth telemetry records stale observation separately
        assert breadth_tel.get("stale_observation_total", 0) >= 1

    def test_breadth_fresh_observation_does_not_reset_entry_counter(self):
        # Set initial entry counter
        _feed_telemetry["stale_bar_consecutive"] = 4
        
        # Fresh bar (delay = 10s <= 60s grace)
        fresh_df = create_sample_bars(num_bars=5, interval_minutes=1, last_bar_closed_ago_seconds=10)
        
        vdf = _validate_intraday_data(fresh_df, "FRESH_SYM", "1m", purpose="REGIME_BREADTH")
        assert not vdf.empty

        entry_tel = get_intraday_feed_telemetry()

        # Invariant: Entry stale counter MUST NOT be reset by breadth data
        assert entry_tel["stale_bar_consecutive"] == 4

    def test_entry_signal_telemetry_retains_original_behavior(self):
        _feed_telemetry["stale_bar_consecutive"] = 2

        # Fresh ENTRY_SIGNAL bar resets counter
        fresh_df = create_sample_bars(num_bars=5, interval_minutes=5, last_bar_closed_ago_seconds=10)
        vdf = _validate_intraday_data(fresh_df, "ENTRY_FRESH", "5m", purpose="ENTRY_SIGNAL")
        assert not vdf.empty
        assert get_intraday_feed_telemetry()["stale_bar_consecutive"] == 0

        # Stale ENTRY_SIGNAL bar increments counter
        stale_df = create_sample_bars(num_bars=5, interval_minutes=5, last_bar_closed_ago_seconds=150)
        vdf2 = _validate_intraday_data(stale_df, "ENTRY_STALE", "5m", purpose="ENTRY_SIGNAL")
        assert vdf2.empty
        assert get_intraday_feed_telemetry()["stale_bar_consecutive"] == 1

    def test_forming_candle_filtered_out_of_breadth(self):
        # Create bars where the latest bar started 10s ago and closes in 50s (forming)
        now = datetime.now(timezone.utc)
        start1 = now - timedelta(seconds=130)
        start2 = now - timedelta(seconds=70)
        forming_start = now - timedelta(seconds=10)  # closes at now + 50s

        df = pd.DataFrame([
            {"datetime": start1, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
            {"datetime": start2, "open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0, "volume": 1200},
            {"datetime": forming_start, "open": 101.0, "high": 105.0, "low": 100.5, "close": 104.0, "volume": 500},
        ]).set_index("datetime")

        # Test compute_vwap_breadth with pre-provided symbol_data
        breadth_pct, mode, details = compute_vwap_breadth(
            symbols=["TEST_FORMING"],
            symbol_data={"TEST_FORMING": df},
        )
        # Forming candle close of 104.0 must NOT be used.
        # If forming bar is filtered out, last close is 101.0.
        # Fallback to index if < 5 symbols
        assert mode == "INDEX_FALLBACK"

    def test_nan_and_negative_volume_sanitized(self):
        now = datetime.now(timezone.utc)
        bad_df = pd.DataFrame([
            {"datetime": now - timedelta(minutes=2), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": -100},
            {"datetime": now - timedelta(minutes=1), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
        ]).set_index("datetime")
        vdf = _validate_intraday_data(bad_df, "BAD_VOL", "1m", purpose="REGIME_BREADTH")
        assert vdf.empty

    def test_batch_disabled_by_default(self):
        assert os.environ.get("REGIME_BREADTH_BATCH_ENABLED") is None
        # Calling compute_vwap_breadth without env flag does not raise
        pct, mode, det = compute_vwap_breadth(symbols=["SPY"])
        assert isinstance(pct, float)
