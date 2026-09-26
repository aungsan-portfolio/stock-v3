"""
test_p0_p1_patches.py -- Unit tests for P0 (Timestamp Normalization) and P1 (True Range ATR Fallback) patches.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from strategies.base import TradeSignal, OrderIntent
from strategies.constants import StrategyName
from strategies.order_manager import _normalize_bar_ts, execute_signal
from strategies.gap_strategy import _compute_atr, _safe_atr, GapAndGoStrategy


# ─────────────────────────────────────────────────────────────────────────────
# P0: Timestamp Normalization Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_bar_ts_space_vs_t_format():
    """Verify that timestamps with space vs 'T' normalize to the exact same UTC datetime."""
    ts_space = "2026-09-15 14:15:00+00:00"
    ts_iso = "2026-09-15T14:15:00+00:00"

    dt_space = _normalize_bar_ts(ts_space)
    dt_iso = _normalize_bar_ts(ts_iso)

    assert dt_space is not None
    assert dt_iso is not None
    assert dt_space == dt_iso
    assert dt_space.tzinfo == timezone.utc


def test_normalize_bar_ts_z_suffix():
    """Verify 'Z' suffix is parsed to UTC timezone-aware datetime."""
    ts_z = "2026-09-15T14:15:00Z"
    ts_offset = "2026-09-15T14:15:00+00:00"

    dt_z = _normalize_bar_ts(ts_z)
    dt_offset = _normalize_bar_ts(ts_offset)

    assert dt_z == dt_offset


def test_normalize_bar_ts_pandas_timestamp_and_datetime():
    """Verify pd.Timestamp and native datetime objects normalize identically to strings."""
    ts_str = "2026-09-15T14:15:00+00:00"
    pdt = pd.Timestamp(ts_str)
    pydt = datetime(2026, 9, 15, 14, 15, 0, tzinfo=timezone.utc)

    assert _normalize_bar_ts(ts_str) == _normalize_bar_ts(pdt)
    assert _normalize_bar_ts(pdt) == _normalize_bar_ts(pydt)


def test_normalize_bar_ts_none_and_empty():
    """Verify None and empty string return None gracefully."""
    assert _normalize_bar_ts(None) is None
    assert _normalize_bar_ts("") is None
    assert _normalize_bar_ts("   ") is None


def test_execution_boundary_readiness_no_mismatch_on_format_difference():
    """Verify execute_signal does NOT reject when bar times match in time but differ in format."""
    sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.75,
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        atr=2.0,
        risk_per_share=2.0,
        reason="VWAP bounce",
        intent=OrderIntent.ENTRY,
        bar_timestamp="2026-09-15 14:15:00+00:00",  # String with space from pandas
    )

    from strategies.base import EntryReadiness
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="c_test",
        bar_ts="2026-09-15T14:15:00+00:00",  # String with 'T' from datetime.isoformat()
        created_at=datetime.now(timezone.utc).timestamp(),
        expires_at=datetime.now(timezone.utc).timestamp() + 300.0,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0
    mock_bridge.open_position_count.return_value = 0
    mock_bridge.open_positions.return_value = []
    mock_bridge.working_orders.return_value = []
    mock_bridge.paper = True

    res = execute_signal(
        signal=sig,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        dry_run=True,
        entry_readiness=readiness,
    )

    # Must NOT be rejected for ENTRY_READINESS_BAR_MISMATCH!
    assert "ENTRY_READINESS_BAR_MISMATCH" not in res.get("reason", "")


def test_execution_boundary_readiness_rejects_actual_bar_time_difference():
    """Verify execute_signal STILL rejects when bar times are genuinely different."""
    sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.75,
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        atr=2.0,
        risk_per_share=2.0,
        reason="VWAP bounce",
        intent=OrderIntent.ENTRY,
        bar_timestamp="2026-09-15 14:10:00+00:00",  # 14:10
    )

    from strategies.base import EntryReadiness
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="c_test",
        bar_ts="2026-09-15T14:15:00+00:00",  # 14:15 (different bar)
        created_at=datetime.now(timezone.utc).timestamp(),
        expires_at=datetime.now(timezone.utc).timestamp() + 300.0,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0

    res = execute_signal(
        signal=sig,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        dry_run=True,
        entry_readiness=readiness,
    )

    assert res["status"] == "REJECTED"
    assert "ENTRY_READINESS_BAR_MISMATCH" in res["reason"]


# ─────────────────────────────────────────────────────────────────────────────
# P1: True Range ATR Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_atr_valid_ohlc_with_gap():
    """Verify _compute_atr computes authentic True Range including gaps."""
    # Create 20 synthetic bars where High-Low = 1.0, but bar gaps up by 2.0
    dates = pd.date_range("2026-09-15 09:30", periods=20, freq="5min", tz="UTC")
    data = []
    prev_c = 100.0
    for i in range(20):
        c = prev_c + 2.0  # Gap up
        h = c + 0.5
        l = c - 0.5
        data.append({"open": c, "high": h, "low": l, "close": c, "volume": 1000})
        prev_c = c

    df = pd.DataFrame(data, index=dates)
    atr = _compute_atr(df, period=14)

    assert atr > 0.01
    assert np.isclose(atr, 2.5, atol=0.01)


def test_compute_atr_insufficient_history():
    """Verify _compute_atr returns 0.0 when bars < period + 1."""
    dates = pd.date_range("2026-09-15 09:30", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 1000
    }, index=dates)

    atr = _compute_atr(df, period=14)
    assert atr == 0.0


def test_safe_atr_prefers_valid_intraday_atr():
    """Verify _safe_atr uses existing valid intraday ATR when present."""
    intraday_df = pd.DataFrame({"atr": [1.5, 2.0, 2.5]})
    full_df = pd.DataFrame()

    atr = _safe_atr(intraday_df=intraday_df, full_df=full_df, period=14)
    assert atr == 2.5


def test_safe_atr_falls_back_to_full_df_when_intraday_atr_is_nan():
    """Verify _safe_atr falls back to full_df when intraday ATR is NaN."""
    intraday_df = pd.DataFrame({"atr": [np.nan, np.nan, np.nan]})

    dates = pd.date_range("2026-09-14 09:30", periods=25, freq="5min", tz="UTC")
    full_df = pd.DataFrame({
        "open": [100.0 + i for i in range(25)],
        "high": [101.0 + i for i in range(25)],
        "low": [99.0 + i for i in range(25)],
        "close": [100.0 + i for i in range(25)],
        "volume": 1000,
    }, index=dates)

    atr = _safe_atr(intraday_df=intraday_df, full_df=full_df, period=14)
    assert atr > 0.01
    assert np.isclose(atr, 2.0, atol=0.01)


def test_execution_boundary_rejects_missing_signal_timestamp():
    """Verify execute_signal rejects when signal.bar_timestamp is None (fail-closed)."""
    sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.75,
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        atr=2.0,
        risk_per_share=2.0,
        reason="VWAP bounce",
        intent=OrderIntent.ENTRY,
        bar_timestamp=None,  # Missing timestamp
    )

    from strategies.base import EntryReadiness
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="c_test",
        bar_ts="2026-09-15T14:15:00+00:00",
        created_at=datetime.now(timezone.utc).timestamp(),
        expires_at=datetime.now(timezone.utc).timestamp() + 300.0,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0

    res = execute_signal(
        signal=sig,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        dry_run=True,
        entry_readiness=readiness,
    )

    assert res["status"] == "REJECTED"
    assert "ENTRY_READINESS_BAR_INVALID" in res["reason"]


def test_execution_boundary_rejects_unparseable_readiness_timestamp():
    """Verify execute_signal rejects when readiness token has an unparseable bar timestamp."""
    sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.75,
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        atr=2.0,
        risk_per_share=2.0,
        reason="VWAP bounce",
        intent=OrderIntent.ENTRY,
        bar_timestamp="2026-09-15T14:15:00+00:00",
    )

    from strategies.base import EntryReadiness
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="c_test",
        bar_ts="not-a-timestamp",  # Unparseable
        created_at=datetime.now(timezone.utc).timestamp(),
        expires_at=datetime.now(timezone.utc).timestamp() + 300.0,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0

    res = execute_signal(
        signal=sig,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        dry_run=True,
        entry_readiness=readiness,
    )

    assert res["status"] == "REJECTED"
    assert "ENTRY_READINESS_BAR_INVALID" in res["reason"]


def test_latest_session_does_not_mutate_full_history():
    """Verify _latest_session does not alter the length or contents of the input full_df."""
    dates = pd.date_range("2026-09-14 09:30", periods=50, freq="5min", tz="UTC")
    full_df = pd.DataFrame({
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000,
    }, index=dates)

    original_len = len(full_df)
    from strategies.gap_strategy import _latest_session
    session_df = _latest_session(full_df)

    assert len(full_df) == original_len
    assert len(session_df) <= original_len


def test_safe_atr_telemetry_source():
    """Verify _safe_atr correctly reports INTRADAY_CURRENT, FULL_HISTORY_FALLBACK, and UNAVAILABLE."""
    # 1. Valid intraday
    df_valid = pd.DataFrame({"atr": [2.5]})
    atr, src = _safe_atr(df_valid, pd.DataFrame(), period=14, return_source=True)
    assert src == "INTRADAY_CURRENT"
    assert atr == 2.5

    # 2. Fallback
    df_nan = pd.DataFrame({"atr": [np.nan]})
    dates = pd.date_range("2026-09-14 09:30", periods=20, freq="5min", tz="UTC")
    df_full = pd.DataFrame({
        "open": 100.0, "high": 102.0, "low": 98.0, "close": 101.0, "volume": 1000
    }, index=dates)
    atr_fb, src_fb = _safe_atr(df_nan, df_full, period=14, return_source=True)
    assert src_fb == "FULL_HISTORY_FALLBACK"
    assert atr_fb > 0.0

    # 3. Unavailable
    atr_un, src_un = _safe_atr(df_nan, pd.DataFrame(), period=14, return_source=True)
    assert src_un == "UNAVAILABLE"
    assert atr_un == 0.0


def test_candidate_signal_export_hook_has_valid_path_dependency():
    """Verify export_candidate_signals executes with Path and creates the export directory without crash."""
    import tempfile
    from pathlib import Path
    from strategies.orchestrator import export_candidate_signals
    from strategies.regime_engine import MarketRegimeResult, MODE_RISK_ON

    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.72,
        entry_price=120.0,
        stop_price=116.0,
        target_price=128.0,
        atr=2.0,
        risk_per_share=4.0,
        reason="TEST",
        bar_timestamp="2026-09-15T14:15:00+00:00",
    )
    regime = MarketRegimeResult(
        mode=MODE_RISK_ON, score=0, allow_new_longs=True, allow_new_shorts=False,
        halt_new_entries=False, requires_opening_buffer=False, position_scale=1.0,
        confidence_boost=0.0, market_texture="TRENDING", reasons=[],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        export_dir = str(Path(tmp_dir) / "test_exports")
        export_candidate_signals(signals=[sig], regime=regime, market_texture="TRENDING", export_dir=export_dir)

        exported_files = list(Path(export_dir).glob("signals_*.csv"))
        assert len(exported_files) == 1
        content = exported_files[0].read_text(encoding="utf-8")
        assert "NVDA" in content
        assert "MOMENTUM_SCALP" in content

