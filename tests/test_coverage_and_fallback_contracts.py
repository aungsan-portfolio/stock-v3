"""
tests/test_coverage_and_fallback_contracts.py
Source-level test verification of:
1. Fail-closed Entry Signal Suppression on coverage < 95%
2. Dual-Mode Regime Breadth Watchlist -> Index Fallback -> Neutral Fallback
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

import config
from strategies.regime_engine import compute_vwap_breadth
from strategies.orchestrator import evaluate_symbols_parallel


def _sample_intraday_df(is_above=True):
    now = pd.Timestamp.now(tz="UTC")
    dates = pd.date_range(now - pd.Timedelta(minutes=200), periods=40, freq="5min", tz="UTC")
    vwap_vals = np.linspace(100.0, 105.0, 40)
    close_vals = vwap_vals + (2.0 if is_above else -2.0)
    df = pd.DataFrame({
        "open": vwap_vals,
        "high": close_vals + 1.0,
        "low": vwap_vals - 1.0,
        "close": close_vals,
        "volume": np.full(40, 10000.0),
        "vwap": vwap_vals,
    }, index=dates)
    return df


class TestCoverageContracts:
    """Proves that Entry-Signal coverage < 95% fail-closes, preventing universe selection bias."""

    def test_entry_signal_coverage_below_threshold_suppresses_entries(self):
        # 10 symbols requested, only 8 return data => coverage = 80% < 95%
        symbols = [f"SYM{i}" for i in range(10)]
        mock_data = {f"SYM{i}": _sample_intraday_df() for i in range(8)}  # 80% coverage
        telemetry = {
            "requested": 10,
            "received": 8,
            "coverage": 0.80,
            "duration_ms": 100.0,
            "rest_subset_requested": 0,
            "quarantined_count": 0,
            "budget_exhausted": False,
        }

        with patch.object(config, "DAYTRADE_BATCH_FETCH_ENABLED", True), \
             patch("strategies.intraday_data.fetch_intraday_batch", return_value=(mock_data, telemetry)):
            result = evaluate_symbols_parallel(
                symbols=symbols,
                bridge=MagicMock(),
                market_texture="TRENDING",
            )

        assert result.entry_allowed is False, "Entry must be suppressed when coverage < 95%"
        assert "LOW_COVERAGE" in result.suppression_reason
        assert result.coverage == 0.80
        assert len(result.entry_signals) == 0

    def test_entry_signal_coverage_at_or_above_threshold_allowed(self):
        # 10 symbols requested, all 10 return data => coverage = 100% >= 95%
        symbols = [f"SYM{i}" for i in range(10)]
        mock_data = {f"SYM{i}": _sample_intraday_df() for i in range(10)}  # 100% coverage
        telemetry = {
            "requested": 10,
            "received": 10,
            "coverage": 1.0,
            "duration_ms": 100.0,
            "rest_subset_requested": 0,
            "quarantined_count": 0,
            "budget_exhausted": False,
        }

        with patch.object(config, "DAYTRADE_BATCH_FETCH_ENABLED", True), \
             patch("strategies.intraday_data.fetch_intraday_batch", return_value=(mock_data, telemetry)), \
             patch("strategies.orchestrator._evaluate_on_df", return_value=[]):
            result = evaluate_symbols_parallel(
                symbols=symbols,
                bridge=MagicMock(),
                market_texture="TRENDING",
            )

        assert result.entry_allowed is True, "Entry must be permitted when coverage >= 95%"
        assert result.coverage == 1.0
        assert result.suppression_reason is None


class TestRegimeBreadthFallbackContracts:
    """Proves Dual-Mode Watchlist -> Index -> Neutral Breadth fallback mechanisms."""

    def test_primary_watchlist_mode_used_when_symbols_gte_5(self):
        # 6 valid symbols provided => primary watchlist breadth
        valid_symbols = [f"SYM{i}" for i in range(6)]
        symbol_data = {
            f"SYM{i}": _sample_intraday_df(is_above=(i < 4))  # 4 above vwap, 2 below
            for i in range(6)
        }

        breadth_pct, mode, details = compute_vwap_breadth(valid_symbols, symbol_data=symbol_data)

        assert mode == "WATCHLIST_VWAP"
        assert details["evaluated"] == 6
        assert details["above_vwap"] == 4
        assert breadth_pct == round((4 / 6) * 100.0, 2)

    def test_fallback_to_index_breadth_when_symbols_lt_5(self):
        # Only 2 valid symbols (< 5) => must fall back to index breadth
        scarce_symbols = ["SYM1", "SYM2"]
        mock_data = {
            "spy_pct": 1.20,
            "qqq_pct": 0.85,
            "soxx_pct": -0.50,
            "iwm_pct": -0.30,
        }

        breadth_pct, mode, details = compute_vwap_breadth(scarce_symbols, symbol_data={}, data=mock_data)

        assert mode == "INDEX_FALLBACK"
        assert details["evaluated"] == 4
        assert details["positive_count"] == 2
        assert breadth_pct == 50.0  # 2 of 4 index benchmarks positive

    def test_neutral_fallback_when_index_benchmarks_also_fail(self):
        # When even index benchmarks return no data, fall back safely to 50.0% neutral
        breadth_pct, mode, details = compute_vwap_breadth(["SYM1"], symbol_data={}, data={})

        assert mode == "INDEX_FALLBACK"
        assert details["evaluated"] == 0
        assert breadth_pct == 50.0
