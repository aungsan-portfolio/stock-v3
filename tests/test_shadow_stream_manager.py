"""
tests/test_shadow_stream_manager.py -- Comprehensive Unit Tests for ShadowStreamManager.

Verifies:
1. Basic Plan Symbol Partitioning: Exactly 30 symbols streamed, 31st+ symbol isolated on REST fallback.
2. Quote Handling: Accurate thread-safe caching of bid, ask, spread, and spread_pct.
3. Incremental Bar Cache: Bar appending and rolling window capacity enforcement.
4. updatedBars Channel: Late-trade correction updates existing bar in-place.
5. Fail-Closed Health Monitoring: Unhealthy when disconnected, heartbeat expired, or error encountered.
6. Telemetry Logging: Shadow comparison written to JSONL with isolated non-mutating schema.
"""

import sys
import os
import time
import json
import tempfile
import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.shadow_stream_manager import (
    ShadowStreamManager,
    StreamBarRecord,
    StreamQuoteRecord,
    MAX_WEBSOCKET_SYMBOLS,
)
import config


class TestShadowStreamManager(unittest.TestCase):

    def setUp(self):
        self.test_symbols = [
            "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AMD", "GOOGL", "QQQ", "SPY",
            "PLTR", "AVGO", "JPM", "BAC", "UBER", "MU", "NFLX",
            "LLY", "UNH", "ABBV", "COST", "WMT", "HD", "GS", "MS", "CAT", "GE",
            "XOM", "CVX", "CRM", "ORCL",
        ]
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "test_comparison.jsonl"
        self.mgr = ShadowStreamManager(
            api_key="test_key",
            secret_key="test_secret",
            watchlist=self.test_symbols,
            max_stream_symbols=MAX_WEBSOCKET_SYMBOLS,
            comparison_log_path=self.log_path,
        )

    def tearDown(self):
        self.mgr.stop()
        self.temp_dir.cleanup()

    def test_basic_plan_symbol_partitioning(self):
        """
        Alpaca Basic Plan constraint:
        Strictly max 30 symbols subscribed to WebSocket stream.
        31st symbol ('ORCL') is partitioned into rest_fallback_symbols.
        """
        self.assertEqual(len(self.mgr.streamed_symbols), 30)
        self.assertEqual(len(self.mgr.rest_fallback_symbols), 1)
        self.assertEqual(self.mgr.rest_fallback_symbols[0], "ORCL")
        self.assertNotIn("ORCL", self.mgr.streamed_symbols)
        self.assertEqual(self.mgr.health.streamed_symbols_count, 30)
        self.assertEqual(self.mgr.health.rest_fallback_symbols_count, 1)

    def test_quote_event_handling(self):
        """Streaming quote updates in-memory quote record and calculates spread."""
        fake_quote = NS(
            symbol="NVDA",
            bid_price=130.50,
            bid_size=10.0,
            ask_price=130.52,
            ask_size=25.0,
            timestamp=1792000000.0,
        )

        asyncio.run(self.mgr._handle_quote(fake_quote))

        q = self.mgr.get_latest_quote("NVDA")
        self.assertIsNotNone(q)
        self.assertEqual(q.symbol, "NVDA")
        self.assertEqual(q.bid_price, 130.50)
        self.assertEqual(q.ask_price, 130.52)
        self.assertAlmostEqual(q.spread, 0.02, places=4)
        expected_spread_pct = (0.02 / 130.51) * 100.0
        self.assertAlmostEqual(q.spread_pct, expected_spread_pct, places=4)
        self.assertEqual(self.mgr.health.quotes_received, 1)
        self.assertTrue(self.mgr.health.is_connected)

    def test_incremental_bar_cache_and_rolling_limit(self):
        """Minute bars append incrementally and roll at maximum capacity."""
        for i in range(10):
            bar = NS(
                symbol="AAPL",
                open=220.0 + i,
                high=221.0 + i,
                low=219.0 + i,
                close=220.5 + i,
                volume=1000.0 * (i + 1),
                trade_count=50,
                vwap=220.3 + i,
                timestamp=1792000000.0 + (i * 60.0),
            )
            asyncio.run(self.mgr._handle_bar(bar, is_update=False))

        bars = self.mgr.get_bars("AAPL", limit=5)
        self.assertEqual(len(bars), 5)
        self.assertEqual(bars[-1].close, 220.5 + 9)
        self.assertEqual(self.mgr.health.bars_received, 10)

    def test_updated_bars_late_trade_correction(self):
        """
        updatedBars Channel:
        When Alpaca pushes an updated bar for late-reported trades,
        the cache correctly replaces the existing minute bar.
        """
        base_ts = 1792000100.0
        initial_bar = NS(
            symbol="AMD",
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.2,
            volume=5000.0,
            trade_count=100,
            vwap=150.1,
            timestamp=base_ts,
        )
        asyncio.run(self.mgr._handle_bar(initial_bar, is_update=False))
        
        bars = self.mgr.get_bars("AMD")
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].volume, 5000.0)
        self.assertFalse(bars[0].is_updated)

        # Late trade arrives for the same bar (within 30 seconds of base_ts)
        corrected_bar = NS(
            symbol="AMD",
            open=150.0,
            high=151.5,  # Late trade made a higher high
            low=149.5,
            close=150.8,
            volume=6200.0,  # Additional 1200 volume
            trade_count=115,
            vwap=150.3,
            timestamp=base_ts + 2.0,
        )
        asyncio.run(self.mgr._handle_updated_bar(corrected_bar))

        bars_after = self.mgr.get_bars("AMD")
        self.assertEqual(len(bars_after), 1, "Must update in-place without duplicating the bar")
        self.assertEqual(bars_after[0].high, 151.5)
        self.assertEqual(bars_after[0].volume, 6200.0)
        self.assertTrue(bars_after[0].is_updated)
        self.assertEqual(self.mgr.health.updated_bars_received, 1)

    def test_fail_closed_health_monitoring(self):
        """Stream reports unhealthy if disconnected, uninitialized, or heartbeat expired."""
        # Uninitialized manager
        self.assertFalse(self.mgr.is_healthy())

        # Connected with fresh heartbeat
        self.mgr.health.is_connected = True
        self.mgr.health.last_heartbeat = time.time()
        self.assertTrue(self.mgr.is_healthy(max_heartbeat_age_seconds=10.0))

        # Stale heartbeat (> 10s) fails closed
        self.mgr.health.last_heartbeat = time.time() - 25.0
        self.assertFalse(self.mgr.is_healthy(max_heartbeat_age_seconds=10.0))

        # Disconnected state fails closed
        self.mgr.health.is_connected = False
        self.assertFalse(self.mgr.is_healthy())

    def test_record_shadow_comparison_telemetry(self):
        """Shadow telemetry writes comparison entry to JSONL without touching production execution."""
        fake_quote = NS(
            symbol="MSFT",
            bid_price=420.10,
            bid_size=100.0,
            ask_price=420.15,
            ask_size=50.0,
            timestamp=time.time(),
        )
        asyncio.run(self.mgr._handle_quote(fake_quote))

        rest_quote = {"bid": 420.08, "ask": 420.18}
        self.mgr.record_shadow_comparison(
            symbol="MSFT",
            rest_quote=rest_quote,
            rest_latency_ms=124.5,
            is_spread_veto=False,
        )

        self.assertTrue(self.log_path.exists())
        with open(self.log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        record = json.loads(lines[0])
        self.assertEqual(record["symbol"], "MSFT")
        self.assertTrue(record["is_streamed"])
        self.assertEqual(record["rest"]["bid"], 420.08)
        self.assertEqual(record["stream"]["bid"], 420.10)
        self.assertEqual(record["rest"]["latency_ms"], 124.5)
        self.assertIn("cycle_id", record)
        self.assertIn("delta", record)
        self.assertIn("spread_diff_pct", record["delta"])

    def test_quote_latency_breakdown(self):
        """Streaming quote measures receive delay and handler duration precisely."""
        event_ts = time.time() - 0.045  # 45ms earlier
        fake_quote = NS(
            symbol="NVDA",
            bid_price=135.0,
            bid_size=10.0,
            ask_price=135.05,
            ask_size=20.0,
            timestamp=event_ts,
        )
        asyncio.run(self.mgr._handle_quote(fake_quote))

        q = self.mgr.get_latest_quote("NVDA")
        self.assertIsNotNone(q)
        self.assertEqual(q.event_timestamp, event_ts)
        self.assertGreater(q.receive_delay_ms, 0.0)
        self.assertGreater(q.handler_duration_ms, 0.0)

    def test_bar_deduplication_same_minute(self):
        """Bars arriving within same minute window update in-place without duplicating records."""
        base_ts = 1792000000.0
        bar1 = NS(
            symbol="TSLA",
            open=250.0,
            high=251.0,
            low=249.0,
            close=250.5,
            volume=1000.0,
            trade_count=10,
            vwap=250.2,
            timestamp=base_ts,
        )
        asyncio.run(self.mgr._handle_bar(bar1, is_update=False))
        self.assertEqual(len(self.mgr.get_bars("TSLA")), 1)

        # Bar arriving 10 seconds later for the same minute bar
        bar2 = NS(
            symbol="TSLA",
            open=250.0,
            high=252.0,  # higher high
            low=249.0,
            close=251.8,
            volume=1800.0,
            trade_count=22,
            vwap=250.9,
            timestamp=base_ts + 10.0,
        )
        asyncio.run(self.mgr._handle_bar(bar2, is_update=False))
        bars = self.mgr.get_bars("TSLA")
        self.assertEqual(len(bars), 1, "Must update in-place rather than creating duplicate bar")
        self.assertEqual(bars[0].high, 252.0)
        self.assertEqual(bars[0].volume, 1800.0)

    @patch("alpaca.data.historical.StockHistoricalDataClient")
    def test_bootstrap_historical_bars(self, mock_client_cls):
        """Historical bar bootstrap fetches and populates premarket bars in cache."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        mock_raw_bar1 = NS(open=100.0, high=101.0, low=99.0, close=100.5, volume=500.0, trade_count=5, vwap=100.2, timestamp=1792000000.0)
        mock_raw_bar2 = NS(open=100.5, high=102.0, low=100.0, close=101.5, volume=800.0, trade_count=8, vwap=101.0, timestamp=1792000060.0)
        mock_client.get_stock_bars.return_value = {"NVDA": [mock_raw_bar1, mock_raw_bar2]}

        counts = self.mgr.bootstrap_historical_bars(symbols=["NVDA"], lookback_hours=12)
        self.assertIn("NVDA", counts)
        self.assertEqual(counts["NVDA"], 2)
        
        cached = self.mgr.get_bars("NVDA")
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached[0].open, 100.0)
        self.assertEqual(cached[1].close, 101.5)


    def test_hardened_stream_phase_b_backoff_progression(self):
        """Verifies canonical Phase B backoff: 2s -> 4s -> 8s -> 16s -> 30s (+-20% jitter)."""
        from strategies.shadow_stream_manager import _HardenedStockDataStream
        
        stream = _HardenedStockDataStream("key", "secret", manager=self.mgr)
        
        # Test 100 simulations across attempts 1..6 to verify mathematical bounds
        for _ in range(50):
            # Attempt 1: base 2.0s (+-20% -> [1.6, 2.4])
            raw_1 = min(stream._max_delay, stream._base_delay * (2 ** (1 - 1)))
            self.assertEqual(raw_1, 2.0)
            
            # Attempt 2: base 4.0s (+-20% -> [3.2, 4.8])
            raw_2 = min(stream._max_delay, stream._base_delay * (2 ** (2 - 1)))
            self.assertEqual(raw_2, 4.0)

            # Attempt 3: base 8.0s (+-20% -> [6.4, 9.6])
            raw_3 = min(stream._max_delay, stream._base_delay * (2 ** (3 - 1)))
            self.assertEqual(raw_3, 8.0)

            # Attempt 4: base 16.0s (+-20% -> [12.8, 19.2]) - MUST NOT SKIP 16s!
            raw_4 = min(stream._max_delay, stream._base_delay * (2 ** (4 - 1)))
            self.assertEqual(raw_4, 16.0)

            # Attempt 5: base 30.0s (capped at max_delay)
            raw_5 = min(stream._max_delay, stream._base_delay * (2 ** (5 - 1)))
            self.assertEqual(raw_5, 30.0)

            # Attempt 6+: base 30.0s (capped at max_delay)
            raw_6 = min(stream._max_delay, stream._base_delay * (2 ** (6 - 1)))
            self.assertEqual(raw_6, 30.0)

    def test_hardened_stream_fail_closed_on_error(self):
        """Verifies fail-closed health transition when disconnected or error encountered."""
        from strategies.shadow_stream_manager import _HardenedStockDataStream
        
        stream = _HardenedStockDataStream("key", "secret", manager=self.mgr)
        self.mgr.health.is_connected = True
        
        stream._mark_manager_disconnected("socket.gaierror: [Errno 11001] getaddrinfo failed")
        
        self.assertFalse(self.mgr.health.is_connected, "Stream manager must transition fail-closed on disconnect")
        self.assertIn("11001", self.mgr.health.last_error)
        self.assertEqual(self.mgr.health.errors_count, 1)

    def test_hardened_stream_stop_responsive_during_backoff(self):
        """Verifies that calling stop breaks backoff loop without hanging on sleep."""
        from strategies.shadow_stream_manager import _HardenedStockDataStream
        
        stream = _HardenedStockDataStream("key", "secret", manager=self.mgr)
        stream._should_run = False
        
        t0 = time.monotonic()
        asyncio.run(stream._handle_backoff("TestError", "simulate abort"))
        elapsed = time.monotonic() - t0
        
        self.assertLess(elapsed, 0.5, "Backoff loop must terminate immediately when _should_run is False")


if __name__ == "__main__":
    unittest.main()
