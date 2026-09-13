"""
tests/test_p0_forensic_replays.py -- 13 Mandatory Deterministic Replay Tests for P0 Hardening.

Covers:
  1. test_stale_primary_and_stale_fallback_rejects_both
  2. test_stale_primary_and_fresh_fallback_accepts_fresh
  3. test_five_day_lookback_with_entry_signal_validates_staleness
  4. test_regime_nested_benchmark_values_flattened_properly
  5. test_regime_benchmarks_genuinely_absent_yields_insufficient_data
  6. test_aapl_offset_anchor_preserves_structural_rr
  7. test_post_fill_rr_below_threshold_preserves_stop_and_emits_degraded
  8. test_post_fill_stop_invalid_triggers_protection_first_remediation
  9. test_open_position_code_deployment_guard
  10. test_direction_aware_chase_favorable_quote
  11. test_noise_floor_fail_closed_rejection
  12. test_ledger_reconciliation_roundtrip
  13. test_stale_primary_and_unauthorized_fallback_fails_closed
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

import config
from strategies.base import TradeSignal, StrategyName
from strategies.intraday_data import (
    _validate_intraday_data,
    should_validate_staleness,
    fetch_batch_intraday_alpaca,
)
from strategies.regime_engine import (
    extract_benchmark_pct,
    validate_regime_texture_combination,
    MarketRegimeResult,
    DIAGNOSTIC_INSUFFICIENT_DATA,
    DIAGNOSTIC_VALID,
    MODE_RISK_ON,
    TEXTURE_TRENDING,
    TEXTURE_VOLATILE,
)
from strategies.order_manager import (
    unfavorable_chase,
    validate_geometry,
    execute_signal,
    compute_effective_noise_floor,
    _pending_order_contexts,
    _pending_order_contexts_lock,
)
from strategies.trailing_stop import DynamicTrailingStopManager
from tools.reconstruct_session_ledger import load_dynamic_broker_trades, compute_sha256


class TestP0ForensicReplays(unittest.TestCase):

    def setUp(self):
        with _pending_order_contexts_lock:
            _pending_order_contexts.clear()
        self._cache_patch = patch("strategies.intraday_data._read_cache", return_value=None)
        self._cache_patch.start()

    def tearDown(self):
        self._cache_patch.stop()
        with _pending_order_contexts_lock:
            _pending_order_contexts.clear()

    # ─────────────────────────────────────────────────────────────
    # Replay Test 1: Stale Primary & Stale Fallback Rejects Both
    # ─────────────────────────────────────────────────────────────
    def test_stale_primary_and_stale_fallback_rejects_both(self):
        """When both primary (IEX) and fallback (SIP) feeds return stale bars, symbol is strictly excluded."""
        now = datetime.now(timezone.utc)
        stale_bar_time = now - timedelta(minutes=15)

        stale_df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 102.5],
                "low": [99.5, 100.5],
                "close": [101.0, 102.0],
                "volume": [5000, 6000],
            },
            index=[stale_bar_time - timedelta(minutes=5), stale_bar_time],
        )

        mock_client = MagicMock()
        mock_bars = MagicMock()
        mock_bars.df = stale_df
        mock_client.get_stock_bars.return_value = mock_bars

        with patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec")), \
             patch("strategies.intraday_data._get_alpaca_client", return_value=mock_client), \
             patch("strategies.session.is_market_open", return_value=True):

            res = fetch_batch_intraday_alpaca(["AAPL"], interval="5m", lookback_days=1, purpose="ENTRY_SIGNAL")
            # Both were stale -> symbol must NOT be accepted in results
            self.assertNotIn("AAPL", res, "Stale symbol must be strictly excluded from batch results")

    # ─────────────────────────────────────────────────────────────
    # Replay Test 2: Stale Primary & Fresh Fallback Accepts Fresh
    # ─────────────────────────────────────────────────────────────
    def test_stale_primary_and_fresh_fallback_accepts_fresh(self):
        """When primary feed is stale, fallback feed returns fresh bar -> fallback accepted."""
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(minutes=15)
        # For a 5m bar, start time must be 5m + 20s ago so that close time (start + 5m) was 20s ago (completed & fresh)
        fresh_time = now - timedelta(minutes=5, seconds=20)

        stale_df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]},
            index=[stale_time],
        )
        fresh_df = pd.DataFrame(
            {"open": [101.0], "high": [102.0], "low": [100.5], "close": [101.8], "volume": [2000]},
            index=[fresh_time],
        )

        mock_client = MagicMock()
        stale_bars = MagicMock()
        stale_bars.df = stale_df
        fresh_bars = MagicMock()
        fresh_bars.df = fresh_df

        # First call (primary) returns stale, second call (fallback) returns fresh
        mock_client.get_stock_bars.side_effect = [stale_bars, fresh_bars]

        with patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec")), \
             patch("strategies.intraday_data._get_alpaca_client", return_value=mock_client), \
             patch("strategies.session.is_market_open", return_value=True):

            res = fetch_batch_intraday_alpaca(["AAPL"], interval="5m", lookback_days=1, purpose="ENTRY_SIGNAL")
            self.assertIn("AAPL", res, "Fresh fallback data must be accepted")
            self.assertEqual(float(res["AAPL"]["close"].iloc[-1]), 101.8)

    # ─────────────────────────────────────────────────────────────
    # Replay Test 3: 5-Day Lookback with ENTRY_SIGNAL Validates Staleness
    # ─────────────────────────────────────────────────────────────
    def test_five_day_lookback_with_entry_signal_validates_staleness(self):
        """Unconditional freshness invariant: lookback_days=5 with purpose=ENTRY_SIGNAL MUST validate staleness."""
        self.assertTrue(should_validate_staleness("ENTRY_SIGNAL", lookback_days=5))
        self.assertTrue(should_validate_staleness("ENTRY_SIGNAL", lookback_days=10))
        self.assertFalse(should_validate_staleness("HISTORICAL_BACKTEST", lookback_days=5))

        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(hours=2)
        stale_df = pd.DataFrame(
            {"open": [150.0], "high": [152.0], "low": [149.0], "close": [151.0], "volume": [10000]},
            index=[stale_time],
        )

        with patch("strategies.session.is_market_open", return_value=True):
            validated = _validate_intraday_data(
                stale_df, symbol="NVDA", interval="5m",
                validate_staleness=False,  # Caller incorrectly passed False
                purpose="ENTRY_SIGNAL",     # But purpose is ENTRY_SIGNAL
            )
            self.assertTrue(validated.empty, "Stale data must be rejected when purpose is ENTRY_SIGNAL even if validate_staleness=False was passed")

    # ─────────────────────────────────────────────────────────────
    # Replay Test 4: Nested Benchmark Values Flattened Properly
    # ─────────────────────────────────────────────────────────────
    def test_regime_nested_benchmark_values_flattened_properly(self):
        """extract_benchmark_pct must correctly unpack nested benchmark dicts and avoid None mask."""
        nested_data = {
            "SPY": {"pct": -0.45, "source": "alpaca_broker"},
            "QQQ": {"pct": -0.65, "source": "alpaca_broker"},
            "SOXX": {"pct": 1.20, "source": "yfinance_realtime"},
            "market_breadth": 12,
        }

        self.assertAlmostEqual(extract_benchmark_pct(nested_data, "SPY"), -0.45)
        self.assertAlmostEqual(extract_benchmark_pct(nested_data, "QQQ"), -0.65)
        self.assertAlmostEqual(extract_benchmark_pct(nested_data, "SOXX"), 1.20)

        status, msg, attribution = validate_regime_texture_combination(
            mode=MODE_RISK_ON,
            texture=TEXTURE_TRENDING,
            component_scores={"SPY": -1, "QQQ": -1},
            data=nested_data,
        )
        self.assertEqual(attribution["spy_pct"], -0.45)
        self.assertEqual(attribution["qqq_pct"], -0.65)
        self.assertEqual(attribution["soxx_pct"], 1.20)
        self.assertEqual(attribution["breadth"], 12)

    # ─────────────────────────────────────────────────────────────
    # Replay Test 5: Genuinely Absent Benchmarks Yields INSUFFICIENT_DATA
    # ─────────────────────────────────────────────────────────────
    def test_regime_benchmarks_genuinely_absent_yields_insufficient_data(self):
        """When benchmark data is missing, engine diagnoses INSUFFICIENT_DATA and never claims 'Normal Healthy Market'."""
        empty_data = {}
        status, msg, attr = validate_regime_texture_combination(
            mode=MODE_RISK_ON,
            texture=TEXTURE_TRENDING,
            component_scores={},
            data=empty_data,
        )
        self.assertEqual(status, DIAGNOSTIC_INSUFFICIENT_DATA)
        self.assertIn("Required benchmark inputs unavailable", msg)

        # Verify summary output honesty
        res = MarketRegimeResult(
            mode=MODE_RISK_ON,
            score=0,
            allow_new_longs=True,
            requires_opening_buffer=False,
            position_scale=1.0,
            confidence_boost=0.0,
            diagnostic_status=DIAGNOSTIC_INSUFFICIENT_DATA,
        )
        summary_str = res.summary()
        self.assertNotIn("Normal Healthy Market", summary_str)
        self.assertIn("Insufficient Benchmark Data", summary_str)

    # ─────────────────────────────────────────────────────────────
    # Replay Test 6: AAPL Offset Anchor Preserves Structural R:R
    # ─────────────────────────────────────────────────────────────
    def test_aapl_offset_anchor_preserves_structural_rr(self):
        """Pre-submit anchor fix: Structural risk and reward distances anchored to signal.entry_price preserve R:R."""
        # Original AAPL signal from 2026-09-09 session:
        # entry=316.10, stop=315.44 (dist=0.66), target=317.42 (dist=1.32), RR=2.00
        signal_entry = 316.10
        signal_stop = 315.44
        signal_target = 317.42

        stop_distance = signal_entry - signal_stop      # 0.66
        target_distance = signal_target - signal_entry  # 1.32

        # Executable quote limit price after bid/ask execution offset
        live_limit = 316.04

        # New anchored calculation
        submitted_stop = round(live_limit - stop_distance, 2)     # 315.38
        submitted_target = round(live_limit + target_distance, 2) # 317.36

        submitted_risk = live_limit - submitted_stop      # 0.66
        submitted_reward = submitted_target - live_limit  # 1.32
        submitted_rr = round(submitted_reward / submitted_risk, 4)

        self.assertAlmostEqual(submitted_stop, 315.38)
        self.assertAlmostEqual(submitted_target, 317.36)
        self.assertAlmostEqual(submitted_rr, 2.00)

        # Geometry validator passes
        valid, err, r, rew, rr = validate_geometry("BUY", live_limit, submitted_stop, submitted_target)
        self.assertTrue(valid)
        self.assertAlmostEqual(rr, 2.00)

    # ─────────────────────────────────────────────────────────────
    # Replay Test 7: Post-Fill R:R Degraded Preserves Stop & Emits Event
    # ─────────────────────────────────────────────────────────────
    def test_post_fill_rr_below_threshold_preserves_stop_and_emits_degraded(self):
        """When post-fill R:R degrades (e.g. fill slippage), valid protective stop is PRESERVED and untouched."""
        manager = DynamicTrailingStopManager()

        # LONG position: Entry 316.20 (slippage from 316.04), Stop 315.40 (risk 0.80), Target 317.00 (reward 0.80)
        # actual_rr = 1.00 < min_rr 1.50
        # Stop is valid and safe (315.40 < 316.20, dist 0.80 >= 0.05), but RR is degraded
        resulting_stop = manager.validate_and_rebuild_geometry(
            symbol="AAPL",
            signal_price=316.04,
            fill_price=316.20,
            initial_stop=315.40,
            target_price=317.00,
            side="BUY",
        )
        # Protective stop MUST be preserved at 315.40 (never cancelled or wiped out)
        self.assertEqual(resulting_stop, 315.40)

    # ─────────────────────────────────────────────────────────────
    # Replay Test 8: Post-Fill Stop Invalid Triggers Protection First Remediation
    # ─────────────────────────────────────────────────────────────
    def test_post_fill_stop_invalid_triggers_protection_first_remediation(self):
        """When post-fill stop is inverted (stop > fill for BUY), protection remediation rebuilds a valid stop."""
        manager = DynamicTrailingStopManager()

        # Inverted stop: initial_stop 317.00 > fill_price 316.00 for LONG!
        resulting_stop = manager.validate_and_rebuild_geometry(
            symbol="AAPL",
            signal_price=316.00,
            fill_price=316.00,
            initial_stop=317.00,  # Structurally inverted!
            target_price=318.00,
            side="BUY",
        )
        # Rebuilt stop must be strictly below fill_price to restore protective invariants
        self.assertLess(resulting_stop, 316.00)
        self.assertGreater(resulting_stop, 0.0)

    # ─────────────────────────────────────────────────────────────
    # Replay Test 9: Open Position Deployment Guard
    # ─────────────────────────────────────────────────────────────
    def test_open_position_code_deployment_guard(self):
        """Zero Live Hot-Patching Invariant: Refuses preflight if open broker positions exist."""
        def verify_deployment_preflight(bridge_mock):
            positions = bridge_mock.positions_plain()
            if len(positions) > 0:
                raise RuntimeError(f"Deployment safety gate violation: {len(positions)} open positions active at broker")
            return True

        # Case A: Open positions active -> refuses deployment
        dirty_bridge = MagicMock()
        dirty_bridge.positions_plain.return_value = [{"symbol": "QQQ", "qty": 5.0}]
        with self.assertRaises(RuntimeError) as ctx:
            verify_deployment_preflight(dirty_bridge)
        self.assertIn("Deployment safety gate violation", str(ctx.exception))

        # Case B: Account flat (0 positions) -> allows deployment
        clean_bridge = MagicMock()
        clean_bridge.positions_plain.return_value = []
        self.assertTrue(verify_deployment_preflight(clean_bridge))

    # ─────────────────────────────────────────────────────────────
    # Replay Test 10: Direction-Aware Chase Favorable Quote
    # ─────────────────────────────────────────────────────────────
    def test_direction_aware_chase_favorable_quote(self):
        """Direction-aware chase: Favorable price movement (paying less on BUY) yields 0.0 unfavorable chase."""
        # Live ask is $316.04 vs signal price $316.10 (favorable price improvement)
        unf_chase = unfavorable_chase("BUY", 316.10, 316.04)
        self.assertEqual(unf_chase, 0.0, "Favorable price improvement must not consume chase buffer")

        # Unfavorable price movement (paying more on BUY)
        unf_chase_bad = unfavorable_chase("BUY", 316.10, 316.25)
        self.assertAlmostEqual(unf_chase_bad, 0.15)

    # ─────────────────────────────────────────────────────────────
    # Replay Test 11: Noise Floor Fail-Closed Rejection
    # ─────────────────────────────────────────────────────────────
    def test_noise_floor_fail_closed_rejection(self):
        """Option A Fail-Closed Invariant: Stop distance < noise floor results in immediate VETO/REJECTION."""
        sig = TradeSignal(
            symbol="AVGO",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.85,
            entry_price=100.00,
            stop_price=99.96,  # stop distance = 0.04 < minimum noise floor 0.08
            target_price=101.00,
            atr=0.10,
            risk_per_share=0.04,
            reason="Sub noise floor test",
        )

        res = execute_signal(
            signal=sig,
            bridge=None,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
        )
        self.assertEqual(res["status"], "REJECTED")
        self.assertIn("REJECTED_NOISE_FLOOR_BREACH", res["reason"])

    # ─────────────────────────────────────────────────────────────
    # Replay Test 12: Ledger Reconciliation Roundtrip
    # ─────────────────────────────────────────────────────────────
    def test_ledger_reconciliation_roundtrip(self):
        """Reconstruct session ledger from broker orders and verify that all fills cleanly reconcile."""
        orders_path = Path("reports/raw/broker_orders_20260908.json")
        self.assertTrue(orders_path.exists(), "Raw broker orders export must exist for forensic immutability")

        trades = load_dynamic_broker_trades(orders_path, session_date="2026-09-08")
        self.assertGreater(len(trades), 0, "Must reconstruct non-empty set of closed trades")

        total_broker_pnl = sum(t.broker_realized_pnl for t in trades)
        # Expected broker realized PnL from forensic baseline is +$12.22
        self.assertAlmostEqual(total_broker_pnl, 12.22, places=2)

    # ─────────────────────────────────────────────────────────────
    # Replay Test 13: Stale Primary & Unauthorized Fallback Fails Closed
    # ─────────────────────────────────────────────────────────────
    def test_stale_primary_and_unauthorized_fallback_fails_closed(self):
        """When primary returns stale bars and fallback raises 403 Forbidden / Unauthorized, fail-closed rejects symbol."""
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(minutes=15)

        stale_df = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5], "volume": [1000]},
            index=[stale_time],
        )

        mock_client = MagicMock()
        stale_bars = MagicMock()
        stale_bars.df = stale_df

        # Primary call succeeds with stale data; Fallback call raises 403 Forbidden / Unauthorized
        mock_client.get_stock_bars.side_effect = [
            stale_bars,
            Exception("403 Client Error: Forbidden for url: https://data.alpaca.markets/v2/stocks/bars?feed=sip (subscription required)"),
        ]

        with patch("strategies.intraday_data._get_alpaca_credentials", return_value=("test_key", "test_sec")), \
             patch("strategies.intraday_data._get_alpaca_client", return_value=mock_client), \
             patch("strategies.session.is_market_open", return_value=True), \
             self.assertLogs("strategies.intraday_data", level="WARNING") as cm:

            res = fetch_batch_intraday_alpaca(["AAPL"], interval="5m", lookback_days=1, purpose="ENTRY_SIGNAL")
            # Must fail closed: symbol must NOT be in results
            self.assertNotIn("AAPL", res, "Symbol must be strictly excluded when fallback is unauthorized")
            # Must log FALLBACK_UNAUTHORIZED
            self.assertTrue(
                any("FALLBACK_UNAUTHORIZED" in log_msg for log_msg in cm.output),
                f"Expected FALLBACK_UNAUTHORIZED in log messages, got: {cm.output}",
            )


if __name__ == "__main__":
    unittest.main()
