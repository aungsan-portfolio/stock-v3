"""
tests/test_p1_system.py -- Verification unit tests for Phase 2 (P1 System Consistency & Performance Fixes).
"""

import pytest
import pandas as pd
import numpy as np

from strategies.base import TradeSignal
from strategies.constants import StrategyName
from strategies.orchestrator import _prepare_closed_bars, _evaluate_on_df
from strategies.order_manager import _validate_signal_prices, execute_signal


def test_bar_cadence_strips_open_bar():
    """H-05: _prepare_closed_bars removes the active open bar when len(df) > 1."""
    dates = pd.date_range("2026-08-13 09:30", periods=5, freq="5min")
    df = pd.DataFrame({"close": [10.0, 10.5, 11.0, 11.5, 12.0]}, index=dates)
    
    prepared = _prepare_closed_bars(df)
    assert len(prepared) == 4
    assert prepared.iloc[-1]["close"] == 11.5


def test_journal_failure_is_fail_closed():
    """M-03: Symbol Entry Gate journal audit exception must reject entry (Fail-Closed)."""
    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.70,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        atr=0.5,
        risk_per_share=1.0,
        reason="Test journal gate error",
    )


    import strategies.order_manager as om
    # Mock today_trades to raise an exception
    def broken_today_trades(*args, **kwargs):
        raise RuntimeError("Journal database corrupted")


    import strategies.trade_journal as tj
    orig_fn = tj.today_trades
    try:
        tj.today_trades = broken_today_trades
        res = execute_signal(
            signal=sig,
            bridge=None,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True
        )
        assert res["status"] == "REJECTED"
        assert "Journal check failed" in res["reason"] or "fail-closed" in res["reason"]

    finally:
        tj.today_trades = orig_fn


def test_validate_prices_uses_provided_df_history():
    """H-08: _validate_signal_prices reuses passed df_history snapshot without throwing or re-fetching."""
    dates = pd.date_range("2026-08-13 09:30", periods=15, freq="5min")
    df_snapshot = pd.DataFrame({
        "open": np.full(15, 100.0),
        "high": np.full(15, 101.0),
        "low": np.full(15, 99.0),
        "close": np.full(15, 100.0),
        "volume": np.full(15, 10000),
    }, index=dates)

    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.70,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        atr=0.5,
        risk_per_share=1.0,
        reason="Test snapshot validation",
    )

    reason = _validate_signal_prices(sig, market_texture="TRENDING", df_history=df_snapshot)
    assert reason == ""
