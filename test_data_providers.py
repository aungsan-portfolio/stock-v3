"""
test_data_providers.py — Unit tests for the multi-source data pipeline.

Tests the MultiSourceDataProvider, corporate-action detection, CSV caching,
fallback logic, and the backward-compatible fetch_ohlcv path.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, PropertyMock

import pandas as pd
import numpy as np

from data_providers import MultiSourceDataProvider, DataProviderError


# ── Fixtures ──────────────────────────────────────────────────────────

def _fake_ohlcv(n=260) -> pd.DataFrame:
    """Deterministic fake OHLCV with no anomalies."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Open": close * (1 + np.random.randn(n) * 0.002),
        "High": close * (1 + np.abs(np.random.randn(n)) * 0.01),
        "Low": close * (1 - np.abs(np.random.randn(n)) * 0.01),
        "Close": close,
        "Volume": np.random.randint(1_000_000, 10_000_000, n),
    }, index=dates)


def _split_anomaly_df() -> pd.DataFrame:
    """Fake OHLCV with a 50% overnight gap (simulates unadjusted split)."""
    df = _fake_ohlcv(260)
    # Inject a -50% gap on day 200
    df.iloc[200:, df.columns.get_loc("Close")] *= 0.5
    df.iloc[200:, df.columns.get_loc("Open")] *= 0.5
    df.iloc[200:, df.columns.get_loc("High")] *= 0.5
    df.iloc[200:, df.columns.get_loc("Low")] *= 0.5
    return df


# ── Tests: _clean_and_validate ────────────────────────────────────────

class TestCleanAndValidate:
    def test_normal_ohlcv_passes(self):
        df = _fake_ohlcv(260)
        prov = MultiSourceDataProvider()
        cleaned = prov._clean_and_validate("TEST", df)
        assert len(cleaned) >= 252, "Should pass MIN_HISTORY_DAYS"
        assert list(cleaned.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_too_short_raises(self):
        df = _fake_ohlcv(50)
        prov = MultiSourceDataProvider()
        try:
            prov._clean_and_validate("TEST", df)
            assert False, "Expected ValueError for short data"
        except ValueError:
            pass

    def test_missing_column_raises(self):
        df = _fake_ohlcv(260).drop(columns=["Volume"])
        prov = MultiSourceDataProvider()
        try:
            prov._clean_and_validate("TEST", df)
            assert False, "Expected ValueError for missing column"
        except ValueError:
            pass

    def test_split_anomaly_logged(self, caplog):
        """A severe price drop >35% logs a warning, but does not raise."""
        import logging
        caplog.set_level(logging.WARNING)
        df = _split_anomaly_df()
        prov = MultiSourceDataProvider()
        cleaned = prov._clean_and_validate("SPLIT", df)
        assert len(cleaned) >= 252
        assert any("severe price gap" in rec.message for rec in caplog.records)


# ── Tests: fetch with mocked yfinance ─────────────────────────────────

class TestFetch:
    def test_fetch_yfinance_ok(self):
        """Happy path: yfinance returns valid data."""
        with patch.object(MultiSourceDataProvider, "_fetch_yfinance",
                          return_value=_fake_ohlcv(260)) as mock_yf:
            prov = MultiSourceDataProvider()
            df = prov.fetch("SPY", force_refresh=True)
            assert len(df) >= 252
            mock_yf.assert_called_once()

    def test_fetch_fallback_ibkr(self):
        """When yfinance fails and IBKR connected, falls back."""
        normal = _fake_ohlcv(260)
        def fake_yf(sym, period, interval):
            raise ValueError("yfinance down")
        mock_ib = Mock()
        mock_ib.ib.isConnected.return_value = True
        mock_ib._contract.return_value.symbol = "SPY"

        with patch.object(MultiSourceDataProvider, "_fetch_yfinance", fake_yf):
            with patch.object(MultiSourceDataProvider, "_fetch_ibkr",
                              return_value=normal) as mock_ibkr:
                prov = MultiSourceDataProvider(ib_bridge=mock_ib)
                df = prov.fetch("SPY", force_refresh=True)
                assert len(df) >= 252
                mock_ibkr.assert_called_once()

    def test_fetch_all_fail_raises(self):
        """When every source fails, DataProviderError is raised."""
        def always_fail(sym, period, interval):
            raise ValueError("simulated failure")

        with patch.object(MultiSourceDataProvider, "_fetch_yfinance", always_fail):
            prov = MultiSourceDataProvider()  # no IBKR, no Polygon
            try:
                prov.fetch("NODATA", force_refresh=True)
                assert False, "Expected DataProviderError"
            except DataProviderError:
                pass

    def test_polygon_attempted_when_key_set(self):
        """When POLYGON_API_KEY is set, Polygon is attempted after yfinance fails."""
        normal = _fake_ohlcv(260)
        fake_errors = 0

        def fake_yf(sym, period, interval):
            nonlocal fake_errors
            fake_errors += 1
            raise ValueError("yfinance timeout")

        with patch.object(MultiSourceDataProvider, "_fetch_yfinance", fake_yf):
            with patch.object(MultiSourceDataProvider, "_fetch_polygon",
                              return_value=normal) as mock_poly:
                prov = MultiSourceDataProvider(polygon_api_key="test_key_123")
                df = prov.fetch("SPY", force_refresh=True)
                assert len(df) >= 252
                mock_poly.assert_called_once()
                assert fake_errors == 1


# ── Tests: CSV disk cache ─────────────────────────────────────────────

class TestCaching:
    def test_cache_write_then_read(self, tmp_path):
        """Data is written to CSV and read back identically."""
        from data_providers import CACHE_DIR
        original = CACHE_DIR

        try:
            # Override CACHE_DIR for the test
            import data_providers as dp
            dp.CACHE_DIR = tmp_path / "ohlcv_cache"
            dp.CACHE_DIR.mkdir(parents=True, exist_ok=True)

            with patch.object(MultiSourceDataProvider, "_fetch_yfinance",
                              return_value=_fake_ohlcv(260)):
                prov = MultiSourceDataProvider()
                df1 = prov.fetch("CACHE", force_refresh=True)
                # Second call without force_refresh reads from cache
                df2 = prov.fetch("CACHE", force_refresh=False)
                assert len(df2) >= 252
                # Close values should match
                assert abs(df1.Close.iloc[-1] - df2.Close.iloc[-1]) < 0.01
        finally:
            dp.CACHE_DIR = original


# ── Tests: backward-compatible fetch_ohlcv ────────────────────────────

class TestDataManagerCompat:
    def test_legacy_yfinance_path(self):
        """Without set_data_provider, fetch_ohlcv uses legacy yfinance path.

        This test verifies the branch-gate is backward compatible.
        We mock yfinance directly to avoid network calls.
        """
        fake = _fake_ohlcv(260)
        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker.return_value.history.return_value = fake
            import data_manager
            # Ensure no provider is set
            data_manager.set_data_provider(None)
            df = data_manager.fetch_ohlcv("TEST", force_refresh=True)
            assert len(df) >= 252
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_provider_mode(self):
        """When set_data_provider is called, fetch_ohlcv routes through provider."""
        fake = _fake_ohlcv(260)
        prov = MultiSourceDataProvider()
        with patch.object(prov, "fetch", return_value=fake) as mock_fetch:
            import data_manager
            data_manager.set_data_provider(prov)
            df = data_manager.fetch_ohlcv("ROUTE", force_refresh=True)
            assert len(df) >= 252
            mock_fetch.assert_called_once()
        # Reset
        data_manager.set_data_provider(None)
