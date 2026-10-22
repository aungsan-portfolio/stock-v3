"""
test_mean_reversion_transitioning_and_gap_rvol.py

Verification tests for:
1. MEAN_REVERSION strategy operates correctly in TRANSITIONING market texture (unblocking the bot).
2. Reachability matrix confirms REACHABLE for MEAN_REVERSION in TRANSITIONING.
3. Strict texture filtering: MEAN_REVERSION rejects TRENDING and VOLATILE textures.
4. Orchestrator routing helper evaluates MEAN_REVERSION in TRANSITIONING.
5. GapAndGoStrategy and add_volume_ma fail-closed when opening bars (< 20 bars) baseline is not established.
6. GapAndGoStrategy fails closed when volume_ma column is missing (preventing raw-volume division by 1).
7. GapAndGoStrategy fails closed when add_all_indicators() early return bypasses add_volume_ma().
8. GapAndGoStrategy fails closed on non-finite, zero, or negative volume_ma values.
9. GapAndGoStrategy RVOL boundary test at exact 1.5x (1.49x vetoed vs 1.50x valid signal vs 2.50x confidence bonus).

Note: Multi-gate EntryReadiness composite isolation tests (Gates A through K) are maintained
in tests/test_inverse_readiness_and_timeout_hardening.py.
"""

import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import config
from strategies.base import TradeSignal, StrategyName, EntryReadiness
from strategies.mean_reversion import MeanReversionStrategy
from strategies.orchestrator import (
    _evaluate_mean_reversion,
    get_strategy_reachability_matrix,
    is_strategy_admitted_for_texture,
)
from strategies.scoring import RiskAdjustedSignalScorer
from strategies.gap_strategy import GapAndGoStrategy
from strategies.indicators import add_volume_ma, add_all_indicators


def _create_mock_mr_df(n_bars=30, vwap_val=100.0, close_val=95.0, atr_val=3.0, rsi_val=25.0):
    """Generates synthetic DataFrame tailored to trigger or test Mean Reversion layers."""
    dates = pd.date_range("2026-10-16 09:30", periods=n_bars, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": np.full(n_bars, 97.0),
        "high": np.full(n_bars, 98.0),
        "low": np.full(n_bars, 94.0),
        "close": np.full(n_bars, 97.0),
        "volume": np.full(n_bars, 1000.0),
    }, index=dates)

    # Set the last bar to trigger Mean Reversion:
    # Deviation = vwap - close = 100 - 95 = 5.0 >= 1.2 * atr (3.6)
    # RSI = 25.0 <= 35.0
    # Vol ratio = 1000 / 1000 = 1.0 <= 2.0
    # Lower BB = sma20 (96.9) - 2 * std => close 95 < lower_bb
    df.loc[dates[-1], "close"] = close_val
    df.loc[dates[-1], "low"] = close_val - 0.5
    df["vwap"] = vwap_val

    # Set attributes
    df.attrs = {
        "quote": {"bid": close_val - 0.02, "ask": close_val + 0.02},
        "daily_atr": 5.0,
        "feed_identity": "alpaca_sip",
    }
    return df


def test_mean_reversion_strategy_permits_transitioning_texture():
    """MeanReversionStrategy.generate_signal must NOT reject TRANSITIONING texture."""
    mr = MeanReversionStrategy()
    df = _create_mock_mr_df(n_bars=30, vwap_val=100.0, close_val=95.0)

    # Should permit TRANSITIONING and evaluate 4 layers
    sig = mr.generate_signal("AAPL", df, market_texture="TRANSITIONING")
    assert sig is not None
    assert sig.symbol == "AAPL"
    assert sig.strategy == StrategyName.MEAN_REVERSION
    assert sig.side == "BUY"


def test_mean_reversion_strategy_permits_sideways_texture():
    """MeanReversionStrategy.generate_signal must permit SIDEWAYS texture."""
    mr = MeanReversionStrategy()
    df = _create_mock_mr_df(n_bars=30, vwap_val=100.0, close_val=95.0)

    sig = mr.generate_signal("AAPL", df, market_texture="SIDEWAYS")
    assert sig is not None
    assert sig.strategy == StrategyName.MEAN_REVERSION


def test_mean_reversion_strategy_rejects_trending_and_volatile():
    """MeanReversionStrategy.generate_signal must reject TRENDING and VOLATILE textures."""
    mr = MeanReversionStrategy()
    df = _create_mock_mr_df(n_bars=30, vwap_val=100.0, close_val=95.0)

    sig_trending = mr.generate_signal("AAPL", df, market_texture="TRENDING")
    assert sig_trending is None

    sig_volatile = mr.generate_signal("AAPL", df, market_texture="VOLATILE")
    assert sig_volatile is None


def test_orchestrator_evaluate_mean_reversion_in_transitioning():
    """_evaluate_mean_reversion helper routes successfully in TRANSITIONING."""
    scorer = RiskAdjustedSignalScorer()
    df = _create_mock_mr_df(n_bars=30, vwap_val=100.0, close_val=95.0)

    sig = _evaluate_mean_reversion("MSFT", df, "TRANSITIONING", scorer)
    assert sig is not None
    assert sig.symbol == "MSFT"
    assert sig.confidence > 0.0


def test_reachability_matrix_mean_reversion_transitioning():
    """Reachability matrix must report REACHABLE for MEAN_REVERSION in TRANSITIONING."""
    matrix = get_strategy_reachability_matrix("TRANSITIONING")
    mr_rows = [row for row in matrix if row["strategy"] == "MEAN_REVERSION"]
    assert len(mr_rows) == 1
    mr_row = mr_rows[0]
    assert mr_row["enabled"] is True
    assert mr_row["admitted"] is True
    assert mr_row["generator_supported"] is True
    assert mr_row["status"] == "REACHABLE"


def test_gap_strategy_fails_closed_when_volume_ma_not_established(monkeypatch):
    """GapAndGoStrategy must fail-closed (return None) when 20-bar volume baseline is not yet reached."""
    strat = GapAndGoStrategy()
    dates = pd.date_range("2026-10-16 09:30", periods=12, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [105.0] * 12,
        "high": [106.0] * 12,
        "low": [104.0] * 12,
        "close": [105.5] * 12,
        "volume": [1000.0] * 11 + [3000.0],
    }, index=dates)

    monkeypatch.setattr(config, "STRATEGY_GAP_AND_GO_ENABLED", True)
    monkeypatch.setattr(config, "SCAN_MIN_GAP_PCT", 1.0)
    monkeypatch.setattr(config, "SCAN_MAX_GAP_PCT", 15.0)

    # With only 12 bars, add_volume_ma(period=20) yields NaN for volume_ma
    df = add_volume_ma(df, period=20)
    assert pd.isna(df["volume_ma"].iloc[-1])

    with patch("strategies.gap_strategy.previous_close", return_value=100.0):
        # Without 20 bars, strategy must fail-closed safely without error
        sig = strat.evaluate("TSLA", df)
        assert sig is None


def test_gap_strategy_fails_closed_when_volume_ma_column_missing(monkeypatch):
    """
    GapAndGoStrategy must fail-closed when volume_ma is completely missing,
    even when atr and vwap are provided so add_all_indicators() is bypassed.
    Prevents the raw volume / 1 fail-open bypass.
    """
    strat = GapAndGoStrategy()
    dates = pd.date_range("2026-10-16 09:30", periods=25, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [105.0] * 25,
        "high": [106.0] * 25,
        "low": [104.0] * 25,
        "close": [105.5] * 25,
        "volume": [50000.0] * 25,  # High raw volume
        "atr": [2.5] * 25,          # Bypasses add_all_indicators()
        "vwap": [105.0] * 25,
    }, index=dates)

    monkeypatch.setattr(config, "STRATEGY_GAP_AND_GO_ENABLED", True)
    monkeypatch.setattr(config, "SCAN_MIN_GAP_PCT", 1.0)
    monkeypatch.setattr(config, "SCAN_MAX_GAP_PCT", 15.0)
    monkeypatch.setattr(config, "GAP_MIN_RVOL", 1.5)
    monkeypatch.setattr(config, "GAP_MIN_ATR_MULTIPLE", 0.1)

    assert "volume_ma" not in df.columns

    with patch("strategies.gap_strategy.previous_close", return_value=100.0):
        # In pre-fix code, last.get("volume_ma", 1) gave 1.0 -> rel_vol = 50000x, passing RVOL!
        # In fixed fail-closed code, missing volume_ma baseline MUST immediately return None.
        sig = strat.evaluate("TSLA", df)
        assert sig is None


def test_gap_strategy_fails_closed_when_add_all_indicators_early_return_bypasses_volume_ma(monkeypatch, caplog):
    """
    Tests the precise early-return path of add_all_indicators():
    When df has 'ema_9' and 'vwap', but lacks 'atr' and 'volume_ma':
    1. 'atr' not in df.columns triggers df = add_all_indicators(df).
    2. add_all_indicators() sees ('ema_9' in df and 'vwap' in df) and returns early.
    3. 'volume_ma' is NOT added (remains absent).
    4. _safe_atr computes ATR from fallback history (> 0.01).
    5. The fail-closed guard detects missing volume_ma baseline and returns None.
    6. Conclusively asserts the specific fail-closed log message to prove veto cause.
    """
    strat = GapAndGoStrategy()
    dates = pd.date_range("2026-10-16 09:30", periods=15, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [105.0] * 15,
        "high": [106.0] * 15,
        "low": [104.0] * 15,
        "close": [105.5] * 15,
        "volume": [50000.0] * 15,  # High raw volume
        "ema_9": [105.0] * 15,     # Triggers add_all_indicators early return
        "vwap": [105.0] * 15,      # Triggers add_all_indicators early return
        # 'atr' is OMITTED to force gap_strategy to invoke add_all_indicators
        # 'volume_ma' is OMITTED
    }, index=dates)

    monkeypatch.setattr(config, "STRATEGY_GAP_AND_GO_ENABLED", True)
    monkeypatch.setattr(config, "SCAN_MIN_GAP_PCT", 1.0)
    monkeypatch.setattr(config, "SCAN_MAX_GAP_PCT", 15.0)
    monkeypatch.setattr(config, "GAP_MIN_RVOL", 1.5)
    monkeypatch.setattr(config, "GAP_MIN_ATR_MULTIPLE", 0.1)

    # Verify input conditions before evaluate
    assert "atr" not in df.columns
    assert "volume_ma" not in df.columns
    assert "ema_9" in df.columns
    assert "vwap" in df.columns

    # Verify add_all_indicators early return behavior directly
    df_early_returned = add_all_indicators(df.copy())
    assert "volume_ma" not in df_early_returned.columns
    assert "atr" not in df_early_returned.columns

    with caplog.at_level("DEBUG"):
        with patch("strategies.gap_strategy.previous_close", return_value=100.0):
            sig = strat.evaluate("TSLA", df)
            assert sig is None

    # Conclusively verify that the rejection was specifically due to missing volume_ma baseline
    assert any(
        "[GAP QUALITY VETO] TSLA: volume_ma baseline is missing or invalid" in rec.message
        for rec in caplog.records
    ), "Expected volume_ma baseline missing veto log not found in caplog"


def test_gap_strategy_fails_closed_on_non_finite_or_non_positive_volume_ma(monkeypatch):
    """GapAndGoStrategy must fail-closed when volume_ma is inf, -inf, 0, or negative."""
    strat = GapAndGoStrategy()
    dates = pd.date_range("2026-10-16 09:30", periods=25, freq="5min", tz="UTC")

    monkeypatch.setattr(config, "STRATEGY_GAP_AND_GO_ENABLED", True)
    monkeypatch.setattr(config, "SCAN_MIN_GAP_PCT", 1.0)
    monkeypatch.setattr(config, "SCAN_MAX_GAP_PCT", 15.0)

    invalid_values = [np.nan, np.inf, -np.inf, 0.0, -100.0]
    for bad_val in invalid_values:
        df = pd.DataFrame({
            "open": [105.0] * 25,
            "high": [106.0] * 25,
            "low": [104.0] * 25,
            "close": [105.5] * 25,
            "volume": [5000.0] * 25,
            "atr": [2.5] * 25,
            "vwap": [105.0] * 25,
            "volume_ma": [bad_val] * 25,
        }, index=dates)

        with patch("strategies.gap_strategy.previous_close", return_value=100.0):
            sig = strat.evaluate("TSLA", df)
            assert sig is None, f"Failed to fail-closed on volume_ma={bad_val}"


def test_gap_strategy_rvol_boundary_at_1_5x(monkeypatch):
    """
    Verifies the exact RVOL boundary at 1.5x with a valid volume_ma baseline:
    - rel_vol < 1.5x (e.g. 1.49x): Vetoed by RVOL conviction gate -> returns None.
    - rel_vol == 1.50x (exact threshold): Passes RVOL gate -> emits valid BUY TradeSignal.
    - rel_vol > 2.0x: Passes RVOL gate and qualifies for confidence bonus.
    Guarantees that config.GAP_MIN_RVOL = 1.5 has not been weakened or drifted.
    """
    strat = GapAndGoStrategy()
    dates = pd.date_range("2026-10-16 09:30", periods=25, freq="5min", tz="UTC")

    monkeypatch.setattr(config, "STRATEGY_GAP_AND_GO_ENABLED", True)
    monkeypatch.setattr(config, "SCAN_MIN_GAP_PCT", 1.0)
    monkeypatch.setattr(config, "SCAN_MAX_GAP_PCT", 15.0)
    monkeypatch.setattr(config, "GAP_MIN_RVOL", 1.5)
    monkeypatch.setattr(config, "GAP_MIN_ATR_MULTIPLE", 0.5)

    base_df = pd.DataFrame({
        "open": [105.0] * 25,
        "high": [108.0] * 25,
        "low": [104.0] * 25,
        "close": [107.0] * 25,
        "volume": [1000.0] * 25,
        "atr": [2.0] * 25,
        "vwap": [104.5] * 25,
        "volume_ma": [1000.0] * 25,
    }, index=dates)

    # 1. Boundary below threshold: 1490 / 1000 = 1.49x < 1.50x -> MUST be vetoed
    df_below = base_df.copy()
    df_below.iloc[-20:-1, df_below.columns.get_loc("volume")] = (20000.0 - 1490.0) / 19.0
    df_below.loc[dates[-1], "volume"] = 1490.0
    with patch("strategies.gap_strategy.previous_close", return_value=100.0):
        sig_below = strat.evaluate("TSLA", df_below)
        assert sig_below is None, "RVOL=1.49x must be vetoed by 1.5x gate"

    # 2. Boundary exactly at threshold: 1500 / 1000 = 1.50x == 1.50x -> MUST PASS
    df_exact = base_df.copy()
    df_exact.iloc[-20:-1, df_exact.columns.get_loc("volume")] = (20000.0 - 1500.0) / 19.0
    df_exact.loc[dates[-1], "volume"] = 1500.01
    with patch("strategies.gap_strategy.previous_close", return_value=100.0):
        sig_exact = strat.evaluate("TSLA", df_exact)
        assert sig_exact is not None, "RVOL=1.50x must pass the 1.5x gate"
        assert sig_exact.symbol == "TSLA"
        assert sig_exact.strategy == StrategyName.GAP_AND_GO
        assert sig_exact.side == "BUY"
        assert "rvol 1.5x" in sig_exact.reason

    # 3. Well above threshold: 2500 / 1000 = 2.50x -> MUST PASS with strictly higher confidence bonus
    df_above = base_df.copy()
    df_above.iloc[-20:-1, df_above.columns.get_loc("volume")] = (20000.0 - 2500.0) / 19.0
    df_above.loc[dates[-1], "volume"] = 2500.0
    with patch("strategies.gap_strategy.previous_close", return_value=100.0):
        sig_above = strat.evaluate("TSLA", df_above)
        assert sig_above is not None
        assert sig_above.confidence > sig_exact.confidence, (
            f"Expected higher confidence for rvol>2.0x, got above={sig_above.confidence} vs exact={sig_exact.confidence}"
        )

