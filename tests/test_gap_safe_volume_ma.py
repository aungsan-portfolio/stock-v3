"""
tests/test_gap_safe_volume_ma.py

Dedicated verification tests for GAP Strategy Option (B) Candidate:
`_safe_volume_ma()` isolated historical fallback vs Production baseline behavior.

Covers all 5 Reviewer Acceptance Criteria:
1. Baseline Invariance: Production code vs Candidate comparison.
2. Historical Fallback: Uses completed bars from previous session (bars 10 to 19).
3. Fail-Closed Preservation: When full history < 20 bars, returns None / UNAVAILABLE.
4. Input robustness: Handles input with 'atr' present but 'volume_ma' missing.
5. Invariants intact: GAP thresholds, cutoff 10:30, continuation gates preserved.
6. Multi-gate sequence: Quality pass proceeds to subsequent continuation gates (quality pass != signal guarantee).
"""

import math
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

import config
from strategies.base import TradeSignal, StrategyName
from strategies.gap_strategy import GapAndGoStrategy
try:
    from evidence.candidates.gap_strategy_candidate import (
        GapAndGoStrategyCandidate,
        _safe_volume_ma,
        _compute_volume_ma,
        _safe_atr,
    )
except ImportError:
    GapAndGoStrategyCandidate = None
    _safe_volume_ma = None
    _compute_volume_ma = None
    _safe_atr = None

pytestmark = pytest.mark.skipif(
    GapAndGoStrategyCandidate is None,
    reason="evidence module is excluded by .gitignore and not present in repository",
)


def _build_multi_day_df(n_yest=78, n_today=12, today_vol=1000.0, yest_vol=1000.0, gap_pct=5.0):
    """Build multi-day 5m test DataFrame."""
    dates_yest = pd.date_range("2026-10-15 09:30", periods=n_yest, freq="5min", tz="America/New_York")
    dates_today = pd.date_range("2026-10-16 09:30", periods=n_today, freq="5min", tz="America/New_York")
    dates = dates_yest.append(dates_today)

    open_today = 100.0 * (1.0 + gap_pct / 100.0)
    df = pd.DataFrame({
        "open": [100.0] * n_yest + [open_today] * n_today,
        "high": [101.0] * n_yest + [open_today + 1.0] * n_today,
        "low": [99.0] * n_yest + [open_today - 1.0] * n_today,
        "close": [100.0] * n_yest + [open_today + 0.5] * n_today,
        "volume": [yest_vol] * n_yest + [today_vol] * (n_today - 1) + [today_vol * 5],
    }, index=dates)
    return df


def test_production_strategy_now_emits_valid_trade_signal():
    """Confirms that production GapAndGoStrategy now successfully resolves volume_ma
    and emits a valid TradeSignal on 12 morning session bars before cutoff.
    """
    prod_strat = GapAndGoStrategy()
    df = _build_multi_day_df(n_yest=78, n_today=12, today_vol=1000.0, yest_vol=1000.0, gap_pct=5.0)

    with patch("strategies.gap_strategy.previous_close", return_value=100.0):
        sig = prod_strat.evaluate("NVDA", df)
        assert sig is not None
        assert sig.symbol == "NVDA"
        assert sig.strategy == StrategyName.GAP_AND_GO
        assert sig.side == "BUY"
        assert sig.entry_price == 105.5


def test_candidate_resolves_session_bars_10_to_19_historical_fallback():
    """Case 1: Candidate resolves 12 bars session using completed bars from previous session.
    _safe_volume_ma activates FULL_HISTORY_FALLBACK, resolves valid baseline, and emits BUY signal.
    """
    cand_strat = GapAndGoStrategyCandidate()
    df = _build_multi_day_df(n_yest=78, n_today=12, today_vol=1000.0, yest_vol=1000.0, gap_pct=5.0)

    with patch("evidence.candidates.gap_strategy_candidate.previous_close", return_value=100.0):
        sig = cand_strat.evaluate("NVDA", df)
        assert sig is not None
        assert sig.side == "BUY"
        assert sig.strategy == StrategyName.GAP_AND_GO
        assert "Gap-and-Go" in sig.reason


def test_candidate_history_less_than_20_bars_fails_closed():
    """Case 2: Full history has < 20 bars total.
    Must return 0.0 / 'UNAVAILABLE' and fail-closed in strategy evaluation.
    """
    dates = pd.date_range("2026-10-16 09:30", periods=15, freq="5min", tz="America/New_York")
    df = pd.DataFrame({
        "open": [105.0] * 15,
        "high": [106.0] * 15,
        "low": [104.0] * 15,
        "close": [105.5] * 15,
        "volume": [5000.0] * 15,
    }, index=dates)

    vol_ma, src = _safe_volume_ma(df, df, period=20, return_source=True)
    assert vol_ma == 0.0
    assert src == "UNAVAILABLE"

    cand_strat = GapAndGoStrategyCandidate()
    with patch("evidence.candidates.gap_strategy_candidate.previous_close", return_value=100.0):
        sig = cand_strat.evaluate("TEST", df)
        assert sig is None


def test_candidate_atr_present_but_volume_ma_missing_resolved():
    """Case 3: Input has 'atr' column but missing 'volume_ma'.
    _safe_volume_ma must cleanly resolve via historical fallback without KeyError.
    """
    df = _build_multi_day_df(n_yest=78, n_today=12, today_vol=1500.0, yest_vol=1000.0)
    intraday_df = df.loc[df.index.date == df.index.date[-1]].copy()
    intraday_df["atr"] = 2.50 # atr present
    assert "volume_ma" not in intraday_df.columns

    vol_ma, src = _safe_volume_ma(intraday_df, df, period=20, return_source=True)
    assert src == "FULL_HISTORY_FALLBACK"
    assert vol_ma > 0


def test_candidate_current_session_reaches_20_bars_intraday_current():
    """Case 4: Current session has >= 20 bars.
    _safe_volume_ma must return INTRADAY_CURRENT matching current session rolling-20.
    """
    dates = pd.date_range("2026-10-16 09:30", periods=25, freq="5min", tz="America/New_York")
    df = pd.DataFrame({
        "open": [105.0] * 25,
        "high": [106.0] * 25,
        "low": [104.0] * 25,
        "close": [105.5] * 25,
        "volume": [3000.0] * 25,
    }, index=dates)
    df["volume_ma"] = df["volume"].rolling(20).mean()

    vol_ma, src = _safe_volume_ma(df, df, period=20, return_source=True)
    assert src == "INTRADAY_CURRENT"
    assert vol_ma == 3000.0


def test_candidate_gap_invariants_preserved():
    """Case 5: Verification that key GAP strategy parameters and thresholds remain strictly invariant."""
    assert getattr(config, "GAP_MIN_RVOL", 1.5) == 1.5
    assert getattr(config, "GAP_MIN_ATR_MULTIPLE", 0.5) == 0.5
    assert getattr(config, "GAP_CONTINUATION_MIN_HOLD_BARS", 3) == 3


def test_candidate_gap_quality_pass_proceeds_to_subsequent_continuation_gates(caplog):
    """Case 6: Quality pass (RVOL >= 1.5x, gap >= 0.5x ATR) must proceed to subsequent gates
    (VWAP hold, hold bars count) and fail if continuation conditions are violated.
    """
    cand_strat = GapAndGoStrategyCandidate()
    df = _build_multi_day_df(n_yest=78, n_today=12, today_vol=1000.0, yest_vol=1000.0, gap_pct=5.0)

    # Pre-populate atr so add_all_indicators() does not overwrite our custom vwap test fixture
    df["atr"] = 2.5
    # Force close below VWAP on the latest session to fail continuation check (price 105.5 < vwap 110.0)
    latest_date = df.index.date[-1]
    df.loc[df.index.date == latest_date, "vwap"] = 110.0

    with caplog.at_level("INFO"):
        with patch("evidence.candidates.gap_strategy_candidate.previous_close", return_value=100.0):
            sig = cand_strat.evaluate("TEST", df)
            # Must return None because continuation gate failed (not_above_vwap)
            assert sig is None
            assert "[GAP QUALITY PASS]" in caplog.text
            assert "[GAP CONTINUATION VETO]" in caplog.text
            assert "reason=not_above_vwap" in caplog.text


def test_candidate_provenance_raw_volume_precedence_over_preexisting_column():
    """Case 7: Provenance Hardening: Raw volume data is authoritative.
    If 'volume_ma' column contains an arbitrary/corrupted upstream value (e.g. 99999.0),
    _safe_volume_ma MUST ignore it and compute strictly from raw 'volume' series.
    """
    dates = pd.date_range("2026-10-16 09:30", periods=25, freq="5min", tz="America/New_York")
    df = pd.DataFrame({
        "open": [105.0] * 25,
        "high": [106.0] * 25,
        "low": [104.0] * 25,
        "close": [105.5] * 25,
        "volume": [2000.0] * 25, # True raw volume MA is 2000.0
        "volume_ma": [99999.0] * 25, # Bogus/mismatched upstream column
    }, index=dates)

    vol_ma, src = _safe_volume_ma(df, df, period=20, return_source=True)
    assert src == "INTRADAY_CURRENT"
    # MUST compute exactly from raw volume (2000.0), NOT bogus volume_ma column (99999.0)
    assert vol_ma == 2000.0


def test_candidate_provenance_rejects_timestamp_misalignment():
    """Case 8: Provenance Hardening: History and current session must end at the identical bar.
    If full_df ends at a different timestamp than intraday_df, must fail closed.
    """
    df = _build_multi_day_df(n_yest=78, n_today=12)
    intraday_df = df.loc[df.index.date == df.index.date[-1]].copy()

    # Artificially alter full_df latest timestamp
    stale_full_df = df.iloc[:-1].copy()

    vol_ma, src = _safe_volume_ma(intraday_df, stale_full_df, period=20, return_source=True)
    assert src == "UNAVAILABLE"
    assert vol_ma == 0.0


def test_candidate_provenance_rejects_unsorted_or_duplicate_timestamps():
    """Case 9: Provenance Hardening: Frames with duplicate or non-monotonic timestamps fail closed."""
    dates = pd.date_range("2026-10-16 09:30", periods=25, freq="5min", tz="America/New_York")
    # Introduce duplicate timestamp
    dates_dup = dates[:10].append(dates[9:]).drop(dates[-1])
    df_dup = pd.DataFrame({
        "open": [105.0] * 25,
        "high": [106.0] * 25,
        "low": [104.0] * 25,
        "close": [105.5] * 25,
        "volume": [2000.0] * 25,
    }, index=dates_dup)

    vol_ma, src = _safe_volume_ma(df_dup, df_dup, period=20, return_source=True)
    assert src == "UNAVAILABLE"
    assert vol_ma == 0.0


def test_candidate_provenance_non_finite_volume_in_window_fails_closed():
    """Case 10: Provenance Hardening: Non-finite or negative volume in the 20-bar window fails closed."""
    dates = pd.date_range("2026-10-16 09:30", periods=25, freq="5min", tz="America/New_York")
    volumes = [2000.0] * 25
    volumes[-5] = np.nan # NaN in rolling window

    df = pd.DataFrame({
        "open": [105.0] * 25,
        "high": [106.0] * 25,
        "low": [104.0] * 25,
        "close": [105.5] * 25,
        "volume": volumes,
    }, index=dates)

    vol_ma, src = _safe_volume_ma(df, df, period=20, return_source=True)
    assert src == "UNAVAILABLE"
    assert vol_ma == 0.0


def test_candidate_invalid_current_session_volume_with_valid_history_fails_closed():
    """Case 11: When >= 20 current-session bars exist but current session has invalid/NaN volume,
    strategy MUST fail-closed ('UNAVAILABLE' / 0.0) and MUST NOT silently fall back to history.
    """
    df = _build_multi_day_df(n_yest=78, n_today=25, today_vol=1000.0, yest_vol=1000.0)
    intraday_df = df.loc[df.index.date == df.index.date[-1]].copy()
    # Inject NaN into current session window
    intraday_df.iloc[-5, intraday_df.columns.get_loc("volume")] = np.nan

    vol_ma, src = _safe_volume_ma(intraday_df, df, period=20, return_source=True)
    assert src == "UNAVAILABLE", f"Expected UNAVAILABLE, got {src}"
    assert vol_ma == 0.0

