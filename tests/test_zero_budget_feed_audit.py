"""
tests/test_zero_budget_feed_audit.py — Verification suite for Zero-Budget market data pipeline.
Audits:
1. Alpaca free tier IEX enforcement (no SIP requests on zero-budget path).
2. Clean fallback to Alpaca IEX daily bars when yfinance fails without bridge.
3. MockIB.isConnected() verified health check.
4. Data provider deduplication, OHLC geometric consistency, and volume provenance tagging.
5. Predictor safe VIX handling (no forgery of VIX, fail-closed regime classification).
6. SOXX benchmark alignment against prior-day close.
"""

import math
import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

import config
from alpaca.data.enums import DataFeed
from strategies.regime_classifier import MarketRegime


class TestZeroBudgetFeedAudit(unittest.TestCase):

    def setUp(self):
        self.orig_tier = getattr(config, "ALPACA_SUBSCRIPTION_TIER", "free")
        self.orig_feed = getattr(config, "ALPACA_DATA_FEED", "iex")
        config.ALPACA_SUBSCRIPTION_TIER = "free"
        config.ALPACA_DATA_FEED = "iex"

    def tearDown(self):
        config.ALPACA_SUBSCRIPTION_TIER = self.orig_tier
        config.ALPACA_DATA_FEED = self.orig_feed

    # ── 1. Alpaca Free Tier IEX Enforcement & Fallback Chain ───────────────────

    @patch("strategies.intraday_data._read_cache", return_value=None)
    @patch("strategies.intraday_data._has_alpaca_credentials", return_value=True)
    @patch("strategies.intraday_data.fetch_intraday_alpaca")
    @patch("strategies.intraday_data.fetch_intraday_yfinance")
    def test_intraday_data_free_tier_provider_chain(self, mock_yf, mock_alpaca, mock_creds, mock_cache):
        """When bridge is disconnected on free tier, providers must be [(ALPACA, 'iex'), (YFINANCE, None)] — NEVER 'sip'."""
        from strategies.intraday_data import fetch_intraday

        # Simulate Alpaca IEX returning empty, requiring yfinance fallback
        mock_alpaca.return_value = pd.DataFrame()
        valid_df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]
        }, index=pd.date_range("2026-10-14 09:30", periods=1, freq="5min", tz="UTC"))
        mock_yf.return_value = valid_df

        df = fetch_intraday("AAPL", interval="5m", lookback_days=1, bridge=None)

        # Alpaca was called with feed='iex'
        self.assertTrue(mock_alpaca.called)
        _, kwargs = mock_alpaca.call_args
        self.assertEqual(kwargs.get("feed"), "iex")

        # yfinance fallback was engaged because alpaca returned empty
        self.assertTrue(mock_yf.called)
        self.assertFalse(df.empty)

    @patch("strategies.intraday_data._read_cache", return_value=None)
    @patch("strategies.intraday_data._has_alpaca_credentials", return_value=True)
    @patch("strategies.intraday_data._get_alpaca_client")
    def test_fetch_daily_alpaca_free_tier_never_calls_sip(self, mock_client_getter, mock_creds, mock_cache):
        """On free tier, if get_stock_bars fails on IEX, it must NEVER attempt SIP."""
        from strategies.intraday_data import fetch_daily_alpaca

        client = MagicMock()
        mock_client_getter.return_value = client
        client.get_stock_bars.side_effect = Exception("IEX rate limit")

        df = fetch_daily_alpaca("AAPL", lookback_days=30)
        self.assertTrue(df.empty)

        # Only one attempt was made, with feed=DataFeed.IEX (no secondary SIP request)
        self.assertEqual(client.get_stock_bars.call_count, 1)
        req = client.get_stock_bars.call_args[0][0]
        self.assertEqual(req.feed, DataFeed.IEX)

    # ── 2. AlpacaBridge Free Tier IEX Enforcement ─────────────────────────────

    def test_alpaca_bridge_get_quote_forces_iex_on_free_tier(self):
        """On free tier, bridge.get_quote with default feed must resolve to IEX."""
        from alpaca_bridge import AlpacaBridge

        bridge = AlpacaBridge.__new__(AlpacaBridge)
        bridge._connected = True
        bridge._client = MagicMock()
        bridge._data_client = MagicMock()

        mock_quote = MagicMock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.10
        mock_quote.last_price = 150.05
        mock_quote.timestamp = 1728950000.0
        bridge._data_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

        # Call with default feed on free tier
        res = bridge.get_quote("AAPL")
        self.assertIsNotNone(res)

        # Must have been resolved to DataFeed.IEX
        req = bridge._data_client.get_stock_latest_quote.call_args[0][0]
        self.assertEqual(req.feed, DataFeed.IEX)

    def test_alpaca_bridge_fetch_batch_historical_data_forces_iex_on_free_tier(self):
        """On free tier, bridge.fetch_batch_historical_data must use feed='iex'."""
        from alpaca_bridge import AlpacaBridge

        bridge = AlpacaBridge.__new__(AlpacaBridge)
        bridge._connected = True
        bridge._client = MagicMock()
        bridge._data_client = MagicMock()
        bridge._data_client.get_stock_bars.return_value = MagicMock(data={})

        bridge.fetch_batch_historical_data(["AAPL", "MSFT"])

        req = bridge._data_client.get_stock_bars.call_args[0][0]
        self.assertEqual(req.feed, "iex")

    # ── 3. MockIB.isConnected() Verified Health Checks ─────────────────────────

    def test_mock_ib_is_connected_verified_health(self):
        """MockIB.isConnected() must verify both is_connected and _conn_health.is_healthy."""
        from alpaca_bridge import MockIB

        bridge = MagicMock()
        mock_ib = MockIB(bridge)

        # Case 1: Disconnected bridge -> False
        bridge.is_connected = False
        bridge._conn_health.is_healthy = True
        self.assertFalse(mock_ib.isConnected())

        # Case 2: Connected but unhealthy connection -> False
        bridge.is_connected = True
        bridge._conn_health.is_healthy = False
        self.assertFalse(mock_ib.isConnected())

        # Case 3: Connected and healthy -> True
        bridge.is_connected = True
        bridge._conn_health.is_healthy = True
        self.assertTrue(mock_ib.isConnected())

    # ── 4. MultiSourceDataProvider Fallback & Validation ──────────────────────

    @patch("yfinance.Ticker")
    @patch("strategies.intraday_data.fetch_daily_alpaca")
    def test_multi_source_provider_falls_back_to_alpaca_without_bridge(self, mock_alpaca, mock_ticker):
        """When yfinance fails and bridge is None, provider falls back cleanly to Alpaca IEX daily bars."""
        from data_providers import MultiSourceDataProvider

        mock_ticker.side_effect = Exception("YFinance rate limited 429")

        # Mock valid 250 daily bars from Alpaca (exceeding MIN_HISTORY_DAYS=240)
        dates = pd.date_range("2025-01-01", periods=250, freq="D", tz="UTC")
        mock_alpaca.return_value = pd.DataFrame({
            "open": [100.0] * 250,
            "high": [105.0] * 250,
            "low": [98.0] * 250,
            "close": [102.0] * 250,
            "volume": [500000] * 250,
        }, index=dates)

        provider = MultiSourceDataProvider(ib_bridge=None)
        df = provider.fetch("TESTSYM", period="1y", force_refresh=True)

        self.assertFalse(df.empty)
        self.assertEqual(len(df), 250)
        self.assertEqual(df.attrs.get("volume_source"), "alpaca_iex")

    def test_data_cleaning_and_validation_geometry_and_deduplication(self):
        """_clean_and_validate must deduplicate timestamps and drop rows with invalid OHLC geometry."""
        from data_providers import MultiSourceDataProvider

        provider = MultiSourceDataProvider()

        # Build 250 rows (exceeding MIN_HISTORY_DAYS=240)
        dates = pd.date_range("2025-01-01", periods=250, freq="D")
        df = pd.DataFrame({
            "Open": [100.0] * 250,
            "High": [105.0] * 250,
            "Low": [95.0] * 250,
            "Close": [101.0] * 250,
            "Volume": [10000.0] * 250,
        }, index=dates)

        # Inject invalid OHLC geometry into row 5 (High < Low)
        df.iloc[5, df.columns.get_loc("High")] = 90.0
        df.iloc[5, df.columns.get_loc("Low")] = 95.0

        # Inject duplicate index at end
        dup_row = df.iloc[-1:].copy()
        df = pd.concat([df, dup_row])

        cleaned = provider._clean_and_validate("TESTSYM", df)

        # 1 duplicate dropped, 1 invalid OHLC geometry dropped -> 249 rows
        self.assertEqual(len(cleaned), 249)
        self.assertTrue((cleaned["High"] >= cleaned["Low"]).all())
        self.assertFalse(cleaned.index.duplicated().any())
        self.assertEqual(cleaned.attrs.get("rows_received"), 251)
        self.assertEqual(cleaned.attrs.get("rows_rejected"), 2)
        self.assertIn("duplicate_timestamps", cleaned.attrs.get("rejection_reasons", {}))
        self.assertIn("geometric_inconsistency", cleaned.attrs.get("rejection_reasons", {}))
        self.assertAlmostEqual(cleaned.attrs.get("valid_coverage"), 249 / 251, places=3)

    # ── 5. Safe VIX Handling & Fail-Closed Regime ─────────────────────────────

    @patch("predictor.fetch_ohlcv")
    def test_safe_vix_handling_fails_closed_when_genuine_vix_unavailable(self, mock_fetch):
        """When VIX is unavailable, predictor must NOT forge res['vix'] and must fail-closed."""
        from predictor import get_macro_benchmark_indicators

        # Return 60 days of SPY data
        spy_dates = pd.date_range("2026-01-01", periods=60, freq="D")
        spy_df = pd.DataFrame({
            "Open": [500.0] * 60,
            "High": [505.0] * 60,
            "Low": [495.0] * 60,
            "Close": [500.0 + i * 0.1 for i in range(60)],
            "Volume": [1000000.0] * 60,
        }, index=spy_dates)

        def fetch_side_effect(sym, **kwargs):
            if sym == "SPY":
                return spy_df.copy()
            if sym in ("^VIX", "VIX"):
                return pd.DataFrame()  # VIX unavailable
            return pd.DataFrame()

        mock_fetch.side_effect = fetch_side_effect

        res = get_macro_benchmark_indicators()

        # VIX must NOT be overwritten by realized vol
        self.assertIsNone(res["vix"])
        self.assertEqual(res["vix_status"], "UNAVAILABLE")
        self.assertIsNone(res["vix_source"])

        # SPY 20d realized vol is tracked separately as diagnostic metric
        self.assertIsNotNone(res["spy_realized_vol_20d"])

        # Fail-closed regime: DATA_INVALID prohibits new entries
        self.assertEqual(res["regime"], MarketRegime.DATA_INVALID.value)
        self.assertFalse(res["allows_new_entries"])

    # ── 6. SOXX Benchmark Alignment Against Prior Close ───────────────────────

    def test_soxx_benchmark_calculation_uses_prior_close_not_open(self):
        """SOXX real-time calculation must evaluate against previous close, NOT today's open."""
        from strategies.intraday_risk import _get_soxx_realtime_pct_change

        # Mock yfinance Ticker fast_info
        mock_fast_info = MagicMock()
        mock_fast_info.previous_close = 200.0  # Yesterday close: $200.00
        mock_fast_info.last_price = 194.0      # Current price: $194.00 (-3.00%)

        with patch("strategies.regime_engine.get_soxx_pct_change", return_value=None), \
             patch("strategies.regime_engine._get_premarket_pct_change", return_value=(None, "data_unavailable", None)), \
             patch("yfinance.Ticker") as mock_ticker:

            mock_ticker.return_value.fast_info = mock_fast_info

            pct = _get_soxx_realtime_pct_change()

            # (194.0 - 200.0) / 200.0 * 100 = -3.00%
            self.assertEqual(pct, -3.00)


if __name__ == "__main__":
    unittest.main()
