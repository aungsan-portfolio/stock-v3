import os
import sys
sys.path.insert(0, os.path.abspath("."))
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import numpy as np

import config

from strategies.intraday_data import (
    _has_alpaca_credentials,
    _get_alpaca_credentials,
    fetch_intraday_alpaca,
    fetch_intraday_yfinance,
    fetch_intraday,
    DataProvider,
    MarketDataTransportTimeout,
    MarketDataCapacityError,
    _sdk_get_stock_bars_soft_bounded,
    _alpaca_call_slots,
    get_market_data_metrics,
    CachedAtr,
    update_position_atr_cache,
    get_position_atr_if_fresh,
    clear_position_atr_cache,
    TimeoutHTTPAdapter,
)
from strategies.orchestrator import (
    RecoveryPlan,
    build_recovery_plan,
    recover_missing_intraday,
    recover_unresolved_concurrently,
)
from strategies.watchdog import (
    touch_orchestrator_heartbeat,
    get_orchestrator_progress,
    reset_orchestrator_heartbeat,
    EODFlattenWatchdog,
)


def _create_mock_df(symbol: str, count: int = 10, interval_seconds: int = 300) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    last_close = now - timedelta(seconds=15)
    last_start = last_close - timedelta(seconds=interval_seconds)
    first_start = last_start - timedelta(seconds=interval_seconds * (count - 1))
    idx = pd.date_range(start=first_start, periods=count, freq=f"{interval_seconds}s", tz="UTC")
    df = pd.DataFrame({
        "open": np.linspace(100.0, 105.0, count),
        "high": np.linspace(101.0, 106.0, count),
        "low": np.linspace(99.0, 104.0, count),
        "close": np.linspace(100.5, 105.5, count),
        "volume": [1000.0] * count,
    }, index=idx)
    df.index.name = "datetime"
    return df


# ── 1. Credential Detection ──
def test_daytrade_prefixed_credentials_enable_alpaca(monkeypatch):
    """Verify that DAYTRADE_APCA_* keys enable Alpaca detection."""
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("TRADING_MODE", "daytrade")
    monkeypatch.setenv("DAYTRADE_APCA_API_KEY_ID", "PK_DAYTRADE_TEST")
    monkeypatch.setenv("DAYTRADE_APCA_API_SECRET_KEY", "SK_DAYTRADE_TEST")

    assert _has_alpaca_credentials() is True
    k, s = _get_alpaca_credentials()
    assert k == "PK_DAYTRADE_TEST"
    assert s == "SK_DAYTRADE_TEST"


# ── 2. Provider Non-Recursion ──
def test_provider_specific_fetch_does_not_recurse(monkeypatch):
    """Verify fetch_intraday_alpaca never calls yfinance, and fetch_intraday_yfinance never calls alpaca."""
    alpaca_called = []
    yf_called = []

    def fake_alpaca(*args, **kwargs):
        alpaca_called.append(True)
        return pd.DataFrame()

    def fake_yf(*args, **kwargs):
        yf_called.append(True)
        return pd.DataFrame()

    with patch("strategies.intraday_data.fetch_intraday_yfinance", side_effect=fake_yf), \
         patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded", return_value=MagicMock(df=pd.DataFrame())), \
         patch("strategies.intraday_data._read_cache", return_value=None):
        # Alpaca returns empty, must not invoke yfinance
        df = fetch_intraday_alpaca("AAPL", feed="iex")
        assert df.empty
        assert len(yf_called) == 0

    with patch("strategies.intraday_data.fetch_intraday_alpaca", side_effect=fake_alpaca):
        # yfinance raises exception or returns empty, must not invoke alpaca
        with patch("yfinance.download", side_effect=RuntimeError("Simulated YF Outage")):
            df = fetch_intraday_yfinance("AAPL")
            assert df.empty
            assert len(alpaca_called) == 0


# ── 3. Batch Recovery Precedes Single Recovery ──
def test_batch_recovery_precedes_single_symbol_recovery():
    """Verify recover_missing_intraday queries micro-batch first before single-symbol fallback."""
    plan = RecoveryPlan(
        held_symbols=("SPY",),
        candidate_symbols=("QQQ",),
        skipped_symbols=(),
    )
    call_order = []

    def fake_batch(symbols, *args, **kwargs):
        call_order.append("batch")
        # Return SPY, leaving QQQ unresolved
        return {"SPY": _create_mock_df("SPY")}

    def fake_concurrent(symbols, *args, **kwargs):
        call_order.append("concurrent")
        return {"QQQ": _create_mock_df("QQQ")}, {}

    with patch("strategies.intraday_data.fetch_batch_intraday_alpaca", side_effect=fake_batch), \
         patch("strategies.orchestrator.recover_unresolved_concurrently", side_effect=fake_concurrent):

        recovered, failures = recover_missing_intraday(plan)
        assert "SPY" in recovered
        assert "QQQ" in recovered
        assert call_order == ["batch", "concurrent"]


# ── 4 & 5 & 6. RecoveryPlan Partitioning ──
def test_held_positions_are_prioritized():
    """Verify that held positions are always placed in held_symbols regardless of candidate limit."""
    missing = ["AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA"]
    held = ["AAPL", "NVDA"]
    candidates = ["MSFT", "GOOG", "AMZN", "META", "TSLA"]

    plan = build_recovery_plan(
        missing_symbols=missing,
        held_symbols=held,
        ranked_candidates=candidates,
        candidate_limit=3,
    )
    assert set(plan.held_symbols) == {"AAPL", "NVDA"}
    assert len(plan.candidate_symbols) == 3
    assert set(plan.candidate_symbols) == {"MSFT", "GOOG", "AMZN"}
    assert set(plan.skipped_symbols) == {"META", "TSLA"}


def test_only_top_five_candidates_are_recovered():
    """Verify candidate recovery is capped at candidate_limit (default 5)."""
    missing = [f"SYM_{i}" for i in range(15)]
    candidates = [f"SYM_{i}" for i in range(15)]

    plan = build_recovery_plan(
        missing_symbols=missing,
        held_symbols=[],
        ranked_candidates=candidates,
        candidate_limit=5,
    )
    assert len(plan.candidate_symbols) == 5
    assert len(plan.skipped_symbols) == 10


def test_non_priority_symbols_fail_closed():
    """Verify that symbols in skipped_symbols are tagged non_priority_fail_closed."""
    plan = RecoveryPlan(
        held_symbols=(),
        candidate_symbols=(),
        skipped_symbols=("SYM_SKIP_1", "SYM_SKIP_2"),
    )
    recovered, failures = recover_missing_intraday(plan)
    assert len(recovered) == 0
    assert failures["SYM_SKIP_1"] == "non_priority_fail_closed"
    assert failures["SYM_SKIP_2"] == "non_priority_fail_closed"


# ── 7. Total Recovery Deadline Enforcement ──
def test_total_recovery_deadline_is_enforced():
    """Verify that if total deadline expires, unresolved symbols are cleanly cancelled."""
    def slow_fetch(sym, *args, **kwargs):
        time.sleep(1.0)
        return _create_mock_df(sym)

    # 100ms deadline with a 1s task
    deadline = time.monotonic() + 0.1
    with patch("strategies.intraday_data.fetch_intraday", side_effect=slow_fetch):
        recovered, failures = recover_unresolved_concurrently(
            ["AAPL", "MSFT"],
            deadline=deadline,
        )
        assert len(recovered) == 0
        assert "AAPL" in failures
        assert "MSFT" in failures
        assert any("deadline" in str(f) for f in failures.values())


# ── 8. Executor Queue Boundedness ──
def test_executor_queue_is_bounded():
    """Verify MarketDataCapacityError is raised when executor slots are saturated."""
    acquired_count = 0
    slots_to_hold = []
    while True:
        got = _alpaca_call_slots.acquire(blocking=False)
        if not got:
            break
        slots_to_hold.append(got)
        acquired_count += 1

    try:
        mock_client = MagicMock()
        mock_request = MagicMock()
        with pytest.raises(MarketDataCapacityError):
            _sdk_get_stock_bars_soft_bounded(mock_client, mock_request)
    finally:
        for _ in slots_to_hold:
            _alpaca_call_slots.release()


# ── 9. Soft Timeout Metric Tracking ──
def test_soft_timeout_records_running_future():
    """Verify that a slow SDK call triggers MarketDataTransportTimeout and tracks soft_timeout_total."""
    mock_client = MagicMock()
    def slow_sdk_call(*args, **kwargs):
        time.sleep(0.5)
        return MagicMock()

    mock_client.get_stock_bars.side_effect = slow_sdk_call
    mock_request = MagicMock()

    metrics_before = get_market_data_metrics()
    with pytest.raises(MarketDataTransportTimeout):
        _sdk_get_stock_bars_soft_bounded(mock_client, mock_request, timeout=0.05)

    metrics_after = get_market_data_metrics()
    assert metrics_after["soft_timeout_total"] > metrics_before["soft_timeout_total"]


# ── 10. Transport Connect & Read Timeout Adapter ──
def test_transport_connect_timeout_is_enforced():
    """Verify TimeoutHTTPAdapter sets connect and read timeouts on requests."""
    import requests
    adapter = TimeoutHTTPAdapter(connect_timeout=0.75, read_timeout=1.75)
    assert adapter.connect_timeout == 0.75
    assert adapter.read_timeout == 1.75

    req = requests.Request("GET", "https://paper-api.alpaca.markets/v2/clock").prepare()
    with patch("requests.adapters.HTTPAdapter.send") as super_send:
        super_send.return_value = MagicMock()
        adapter.send(req)
        assert super_send.call_count == 1
        assert super_send.call_args[1]["timeout"] == (0.75, 1.75)

        # Verify explicit timeout is preserved
        adapter.send(req, timeout=(3.0, 4.0))
        assert super_send.call_args[1]["timeout"] == (3.0, 4.0)


# ── 11 & 12. ATR Cache Freshness & Invalidation ──
def test_atr_cache_survives_five_minute_key_boundary():
    """Verify that get_position_atr_if_fresh returns valid ATR within max_age_seconds."""
    clear_position_atr_cache()
    now_utc = datetime.now(timezone.utc)
    update_position_atr_cache("AAPL", "5m", 1.85, now_utc)

    atr = get_position_atr_if_fresh("AAPL", interval="5m", max_age_seconds=360.0)
    assert atr == 1.85


def test_atr_cache_rejects_old_source_bar():
    """Verify that get_position_atr_if_fresh rejects source bars older than tolerance."""
    clear_position_atr_cache()
    old_utc = datetime.now(timezone.utc) - timedelta(seconds=1200)
    update_position_atr_cache("AAPL", "5m", 1.85, old_utc)

    atr = get_position_atr_if_fresh("AAPL", interval="5m", max_age_seconds=360.0)
    assert atr is None


# ── 13. Watchdog Stage Reporting ──
def test_watchdog_reports_exact_stalled_stage():
    """Verify that watchdog evaluation cycle captures and reports the exact stalled stage."""
    reset_orchestrator_heartbeat(time.time() - 130.0)
    touch_orchestrator_heartbeat(
        stage="primary_data_retrieval_completed",
        metadata={"requested_symbols": 17, "missing_symbols": 5},
    )
    reset_orchestrator_heartbeat(time.time() - 130.0)

    wd = EODFlattenWatchdog(stale_alarm_seconds=120.0, stale_alert_seconds=300.0)
    cycle_res = wd.check_and_execute_cycle()

    assert cycle_res["tier1_alarm"] is True
    assert cycle_res["stalled_stage"] == "primary_data_retrieval_completed"


# ── 14. Thread Leak Test ──
def test_timeout_storm_does_not_create_unbounded_threads():
    """Verify that repeated timeout recovery calls do not leak threads."""
    def count_recovery_threads():
        return sum(
            t.is_alive()
            for t in threading.enumerate()
            if t.name.startswith(("intraday-recovery", "alpaca-bars"))
        )

    def slow_fetch(sym, *args, **kwargs):
        time.sleep(0.05)
        return pd.DataFrame()

    baseline = count_recovery_threads()
    with patch("strategies.intraday_data.fetch_intraday", side_effect=slow_fetch):
        for _ in range(10):
            deadline = time.monotonic() + 0.02
            recover_unresolved_concurrently(["AAPL", "MSFT", "GOOG"], deadline=deadline)

    time.sleep(0.2)
    after_threads = count_recovery_threads()
    assert after_threads <= baseline + 8


# ── 15. Bounded Soft Timeout In-Flight Metric & Zero Thread Leak ──
def test_bounded_soft_timeout_tracks_in_flight_and_no_thread_spawn():
    """Verify soft-bounded SDK calls track in-flight and timed-out metrics without spawning threading.Thread."""
    from strategies.intraday_data import _sdk_get_stock_bars_soft_bounded, get_market_data_metrics, MarketDataTransportTimeout
    from unittest.mock import MagicMock

    block_event = threading.Event()
    release_event = threading.Event()

    def blocked_sdk(req):
        block_event.set()
        release_event.wait(timeout=2.0)
        return MagicMock()

    mock_client = MagicMock()
    mock_client.get_stock_bars.side_effect = blocked_sdk
    mock_req = MagicMock(symbol_or_symbols=["AAPL"], timeframe="5m", start="2026-10-15T09:30:00Z", end="2026-10-15T09:35:00Z", feed="iex")

    threads_before = threading.active_count()
    with patch("threading.Thread") as mock_thread_class:
        # Run with short timeout
        with pytest.raises(MarketDataTransportTimeout):
            _sdk_get_stock_bars_soft_bounded(mock_client, mock_req, timeout=0.05)

        # Thread class should NOT be called at all (eliminated in favor of add_done_callback)
        assert mock_thread_class.call_count == 0

    metrics_during = get_market_data_metrics()
    assert metrics_during["timed_out_in_flight"] >= 1
    assert metrics_during["in_flight_calls"] >= 1

    # Unblock the worker thread and wait for completion callback
    release_event.set()
    time.sleep(0.1)

    metrics_after = get_market_data_metrics()
    assert metrics_after["timed_out_in_flight"] == 0
    assert metrics_after["in_flight_calls"] == 0
    assert metrics_after["late_completion_total"] >= 1


# ── 16. Batch Fetch Reason Breakdown & Forming Bar Taxonomy ──
def test_batch_fetch_categorizes_reasons_and_telemetry():
    """Verify batch fetch populates telemetry with distinct reasons: sdk_timeout, no_completed_bars_yet, valid."""
    from strategies.intraday_data import (
        fetch_batch_intraday_alpaca,
        get_feed_telemetry,
        _feed_telemetry,
        _feed_telemetry_lock,
    )
    from unittest.mock import patch, MagicMock

    now_utc = pd.Timestamp("2026-10-15 13:32:00", tz="UTC")
    ts_forming = pd.Timestamp("2026-10-15 13:30:00", tz="UTC") # Closes at 13:35:00 > now_utc (forming)
    forming_df = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]
    }, index=pd.DatetimeIndex([ts_forming]))

    # Reset telemetry
    with _feed_telemetry_lock:
        _feed_telemetry["last_batch_timeout_count"] = 0
        _feed_telemetry["last_batch_forming_count"] = 0
        _feed_telemetry["last_batch_reasons"] = {}

    with patch("strategies.intraday_data._get_alpaca_credentials", return_value=("key", "sec")), \
         patch("strategies.intraday_data._get_alpaca_client"), \
         patch("strategies.intraday_data._read_cache", return_value=None), \
         patch("strategies.intraday_data._write_cache"), \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("pandas.Timestamp.now", return_value=now_utc):

        # Mock SDK: chunk 1 (AAPL) times out; chunk 2 (MSFT) returns forming bar
        def mock_sdk(client, req, timeout=None):
            syms = req.symbol_or_symbols
            if "AAPL" in syms:
                from strategies.intraday_data import MarketDataTransportTimeout
                raise MarketDataTransportTimeout("Simulated timeout")
            # MSFT has forming bar
            mock_res = MagicMock()
            forming_df_multi = forming_df.copy()
            forming_df_multi["symbol"] = "MSFT"
            forming_df_multi = forming_df_multi.set_index(["symbol", forming_df_multi.index])
            mock_res.df = forming_df_multi
            return mock_res

        with patch("strategies.intraday_data._sdk_get_stock_bars_soft_bounded", side_effect=mock_sdk), \
             patch("config.ALPACA_CHUNK_SIZE", 1):
            results = fetch_batch_intraday_alpaca(["AAPL", "MSFT"], interval="5m", lookback_days=1)

    telem = get_feed_telemetry()
    assert telem["last_batch_timeout_count"] == 1
    assert telem["last_batch_forming_count"] == 1
    assert telem["last_batch_reasons"]["AAPL"] == "sdk_timeout"
    assert telem["last_batch_reasons"]["MSFT"] == "no_completed_bars_yet"


# ── 17. Forming Candle at Market Open Does Not Quarantine ──
def test_forming_bar_market_open_no_quarantine():
    """Verify that a forming bar at market open returns empty df without 30m quarantine."""
    from strategies.intraday_data import _validate_intraday_data, is_quarantined
    from strategies.scanner.premarket_scanner import clear_quarantine

    clear_quarantine()
    ts_forming = pd.Timestamp("2026-10-15 13:30:00", tz="UTC") # 5m candle closing at 13:35:00
    df = pd.DataFrame({
        "open": [150.0], "high": [151.0], "low": [149.5], "close": [150.2], "volume": [2000]
    }, index=pd.DatetimeIndex([ts_forming]))

    now_forming = pd.Timestamp("2026-10-15 13:32:00", tz="UTC")
    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.intraday_data._in_market_open_transition", return_value=True), \
         patch("pandas.Timestamp.now", return_value=now_forming):
        vdf = _validate_intraday_data(df, "NVDA", "5m", validate_staleness=True, purpose="ENTRY_SIGNAL")
        assert vdf.empty, "Forming bar before close must fail closed for this cycle"
        assert not is_quarantined("NVDA"), "Forming candle must NOT trigger 30-minute quarantine"

