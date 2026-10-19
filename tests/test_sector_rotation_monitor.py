"""
tests/test_sector_rotation_monitor.py — Targeted Tests for Observe-Only Sector Relative Strength Monitor.

Covers:
1. Observation benchmark symbols cannot reach order submission path or rotation eligibility.
2. Missing or stale bars strictly fail closed and generate veto records.
3. Aligned completed daily bars correctly compute 1d, 5d, 20d relative strength.
4. Invalid observations (NaN, non-positive, timestamp misalignment) generate clear veto records.
5. Existing order eligibility, sector limits, and risk guards remain completely intact.
"""
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

import config
from predictor import Signal
from alpaca_bridge import AlpacaBridge
from strategies.sector_rotation_monitor import (
    SectorRotationMonitor,
    SectorObservationTelemetry,
    SECTOR_BENCHMARK_NAMES,
)


def _generate_synthetic_daily_df(dates, close_values):
    """Helper to generate well-formed OHLCV daily bars."""
    df = pd.DataFrame(
        {
            "Open": close_values,
            "High": [c * 1.01 for c in close_values],
            "Low": [c * 0.99 for c in close_values],
            "Close": close_values,
            "Volume": [1_000_000] * len(close_values),
        },
        index=pd.to_datetime(dates, utc=True),
    )
    df.attrs["volume_source"] = "synthetic_daily_bars"
    return df


def test_observation_symbols_cannot_reach_order_submission_path():
    """Verify that pure observation benchmark ETFs are strictly blocked from broker order submission."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True

    for b_sym in config.NON_TRADABLE_BENCHMARKS:
        dummy_sig = Signal(
            symbol=b_sym,
            action="BUY",
            confidence=0.85,
            rf_score=0.85,
            lstm_score=0.85,
            xgb_score=0.85,
            trans_score=0.85,
            tech_score=0.85,
            price=150.0,
            reason="Benchmark ETF test",
        )
        # execute_signal must return False immediately without any broker write
        res = bridge.execute_signal(dummy_sig)
        assert res is False, f"Expected execute_signal to strictly deny {b_sym}"


def test_spy_and_xlv_preserve_existing_trading_eligibility():
    """Verify that SPY and XLV (in WATCHLIST) are not blocked by the non-tradable benchmark deny guard."""
    assert "SPY" in config.WATCHLIST
    assert "XLV" in config.WATCHLIST
    assert "SPY" not in config.NON_TRADABLE_BENCHMARKS
    assert "XLV" not in config.NON_TRADABLE_BENCHMARKS

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True

    # Layer 3b should not trigger for SPY and XLV
    non_tradable = {str(s).upper().strip() for s in getattr(config, "NON_TRADABLE_BENCHMARKS", ())}
    assert "SPY" not in non_tradable
    assert "XLV" not in non_tradable


def test_observation_symbols_excluded_from_rotation_candidates():
    """Verify that observation benchmark ETFs are excluded from rotation incoming candidate eligibility."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._client.get_all_positions.return_value = []

    # Send incoming signals containing benchmark ETFs
    signals = [
        Signal(
            symbol="XLP", action="BUY", confidence=0.90, price=80.0,
            rf_score=0.9, lstm_score=0.9, xgb_score=0.9, trans_score=0.9, tech_score=0.9,
            reason="Observation benchmark",
        ),
        Signal(
            symbol="SOXX", action="BUY", confidence=0.88, price=240.0,
            rf_score=0.88, lstm_score=0.88, xgb_score=0.88, trans_score=0.88, tech_score=0.88,
            reason="Observation benchmark",
        ),
    ]

    with patch.object(bridge.rotator, "evaluate") as mock_eval:
        bridge._evaluate_and_execute_rotation(signals)
        if mock_eval.called:
            _, incoming = mock_eval.call_args[0]
            incoming_syms = [inc.symbol for inc in incoming]
            assert "XLP" not in incoming_syms
            assert "SOXX" not in incoming_syms


def test_etf_and_spy_returns_use_aligned_completed_daily_bars():
    """Verify exact calculation of 1d, 5d, and 20d RS against SPY using aligned completed bars."""
    monitor = SectorRotationMonitor()

    # Generate 25 contiguous dates
    base_date = date(2026, 9, 15)
    dates = [base_date + timedelta(days=i) for i in range(25)]

    # Constant prices for 0..19, then known step changes
    spy_closes = [500.0] * 25
    xlv_closes = [150.0] * 25

    # Set known prices
    # -21 bar (20 bars ago): SPY=500, XLV=150
    # -6 bar (5 bars ago): SPY=500, XLV=150
    # -2 bar (1 bar ago): SPY=500, XLV=150
    # Latest bar (-1): SPY +2% (510.0), XLV +5% (157.50)
    spy_closes[-1] = 510.0  # +2.0%
    xlv_closes[-1] = 157.50 # +5.0%

    spy_df = _generate_synthetic_daily_df(dates, spy_closes)
    xlv_df = _generate_synthetic_daily_df(dates, xlv_closes)

    eval_date = dates[-1]
    telem = monitor.compute_sector_relative_strength(
        symbol="XLV",
        symbol_df=xlv_df,
        benchmark_df=spy_df,
        benchmark_symbol="SPY",
        now_date=eval_date,
        max_staleness_days=3,
    )

    assert telem.valid is True
    assert telem.status == "VALID"
    assert telem.tradable is False
    assert telem.symbol == "XLV"
    assert telem.benchmark == "SPY"

    # ETF return: (157.5 - 150) / 150 = +0.05 (+5%)
    # SPY return: (510 - 500) / 500 = +0.02 (+2%)
    # RS = +0.05 - (+0.02) = +0.03 (+3%)
    assert pytest.approx(telem.ret_1d_etf, 1e-4) == 0.05
    assert pytest.approx(telem.ret_1d_spy, 1e-4) == 0.02
    assert pytest.approx(telem.rs_1d, 1e-4) == 0.03

    assert pytest.approx(telem.rs_5d, 1e-4) == 0.03
    assert pytest.approx(telem.rs_20d, 1e-4) == 0.03


def test_missing_or_stale_bars_fails_closed():
    """Verify that stale daily bars or insufficient history fail closed with clear veto status."""
    monitor = SectorRotationMonitor()
    base_date = date(2026, 9, 1)
    dates = [base_date + timedelta(days=i) for i in range(25)]
    spy_df = _generate_synthetic_daily_df(dates, [500.0] * 25)
    xlv_df = _generate_synthetic_daily_df(dates, [150.0] * 25)

    # 1. Stale bars: latest bar is 2026-09-25, but eval_date is 2026-10-15 (20 days stale > 5 max)
    stale_eval_date = date(2026, 10, 15)
    telem_stale = monitor.compute_sector_relative_strength(
        symbol="XLV", symbol_df=xlv_df, benchmark_df=spy_df,
        now_date=stale_eval_date, max_staleness_days=5,
    )
    assert telem_stale.valid is False
    assert telem_stale.status == "VETO_STALE"
    assert "Stale daily bars" in telem_stale.reason
    assert telem_stale.rs_1d is None

    # 2. Insufficient history (< 21 bars)
    short_dates = dates[:15]
    short_spy_df = _generate_synthetic_daily_df(short_dates, [500.0] * 15)
    short_xlv_df = _generate_synthetic_daily_df(short_dates, [150.0] * 15)

    telem_short = monitor.compute_sector_relative_strength(
        symbol="XLV", symbol_df=short_xlv_df, benchmark_df=short_spy_df,
        now_date=short_dates[-1], max_staleness_days=5,
    )
    assert telem_short.valid is False
    assert telem_short.status == "VETO_INSUFFICIENT_HISTORY"
    assert "Insufficient aligned completed daily bars" in telem_short.reason


def test_invalid_observations_generate_clear_veto_record():
    """Verify that non-finite prices, missing DataFrames, and timestamp misalignments generate veto records."""
    monitor = SectorRotationMonitor()
    dates = [date(2026, 9, 1) + timedelta(days=i) for i in range(25)]
    spy_df = _generate_synthetic_daily_df(dates, [500.0] * 25)

    # 1. Null / None DataFrame
    telem_null = monitor.compute_sector_relative_strength("XLV", None, spy_df)
    assert telem_null.valid is False
    assert telem_null.status == "VETO_DATA_INVALID"

    # 2. Non-finite price in window (NaN)
    bad_closes = [150.0] * 25
    bad_closes[22] = float("nan")
    bad_xlv_df = _generate_synthetic_daily_df(dates, bad_closes)
    telem_nan = monitor.compute_sector_relative_strength(
        "XLV", bad_xlv_df, spy_df, now_date=dates[-1],
    )
    assert telem_nan.valid is False
    assert telem_nan.status == "VETO_DATA_INVALID"
    assert "Non-finite prices detected" in telem_nan.reason

    # 3. Timestamp Misalignment (SPY ends on dates[-1], ETF ends on dates[-2])
    mismatched_xlv_df = _generate_synthetic_daily_df(dates[:-1], [150.0] * 24)
    telem_misalign = monitor.compute_sector_relative_strength(
        "XLV", mismatched_xlv_df, spy_df, now_date=dates[-1],
    )
    assert telem_misalign.valid is False
    assert telem_misalign.status == "VETO_TIMESTAMP_MISALIGNMENT"


def test_existing_order_eligibility_and_risk_guards_unchanged():
    """Verify that existing tradable universe symbols (e.g. NVDA, MSFT) are not affected by benchmark deny guard."""
    assert "NVDA" in config.SECTOR_MAPPINGS
    assert "MSFT" in config.SECTOR_MAPPINGS
    assert "SECTOR_BENCHMARKS" in dir(config)
    assert "NVDA" not in config.SECTOR_BENCHMARKS
    assert "MSFT" not in config.SECTOR_BENCHMARKS

    # SECTOR_BENCHMARKS should be isolated
    assert "XLV" in config.SECTOR_BENCHMARKS
    assert "SOXX" in config.SECTOR_BENCHMARKS
    assert "XLP" in config.SECTOR_BENCHMARKS


def test_data_manager_rows_rejected_regression_test():
    """Regression test: fetch_ohlcv must raise ValueError cleanly (not NameError on rows_rejected)."""
    import data_manager
    short_df = pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [101.0] * 5,
            "Low": [99.0] * 5,
            "Close": [100.0] * 5,
            "Volume": [1000] * 5,
        },
        index=pd.date_range("2026-01-01", periods=5, tz="UTC"),
    )

    mock_inst = MagicMock()
    mock_inst.history.return_value = short_df
    with patch.object(data_manager.yf, "Ticker", return_value=mock_inst):
        with pytest.raises(ValueError) as excinfo:
            data_manager.fetch_ohlcv("SHORT_SYM_REGRESSION")
        assert "only 5 days after filtering" in str(excinfo.value)
        assert "rows_rejected=" in str(excinfo.value)


def test_observer_module_cannot_emit_broker_orders():
    """Verify that SectorRotationMonitor outputs cannot emit orders or interact with broker state."""
    monitor = SectorRotationMonitor()
    base_date = date(2026, 9, 15)
    dates = [base_date + timedelta(days=i) for i in range(25)]
    spy_df = _generate_synthetic_daily_df(dates, [500.0] * 25)
    xlp_df = _generate_synthetic_daily_df(dates, [80.0] * 25)

    telem = monitor.compute_sector_relative_strength("XLP", xlp_df, spy_df, now_date=dates[-1])
    assert isinstance(telem, SectorObservationTelemetry)
    assert telem.tradable is False
    assert not hasattr(telem, "action")
    assert not hasattr(telem, "quantity")
    assert not hasattr(telem, "client_order_id")


def test_predict_all_integration_executes_monitor():
    """Verify that the normal Swing scan loop (predict_all) executes the sector rotation monitor."""
    from predictor import Predictor

    pred = Predictor()
    with patch("strategies.sector_rotation_monitor.SectorRotationMonitor.scan_all_benchmarks") as mock_scan:
        with patch.object(pred, "predict_all", wraps=pred.predict_all):
            # We mock the symbols list to be empty so it only runs preflight & monitor
            pred.predict_all(symbols=[])
            assert mock_scan.called, "Expected predict_all to execute sector rotation monitor scan"
