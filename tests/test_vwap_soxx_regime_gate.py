"""
test_vwap_soxx_regime_gate.py -- Unit tests for Sprint P1.7:
1. VWAP_BOUNCE Macro Regime & SOXX Semiconductor Gate
2. ORB Shadow Funnel Accounting & Resolution Adaptation
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

import config
from strategies.base import TradeSignal
from strategies.constants import StrategyName
import strategies.order_manager as om
from strategies.regime_engine import (
    MarketRegimeResult,
    MODE_CAUTION,
    MODE_RISK_ON,
    MODE_RISK_OFF,
    get_soxx_pct_change,
)
from strategies.shadow.orb_confirmed import (
    ConfirmedORBStrategy,
    get_confirmed_orb_strategy,
)


class TestVWAPSOXXRegimeGate(unittest.TestCase):
    def setUp(self):
        om.reset_session_skip_state()
        with om._pending_order_contexts_lock:
            om._pending_order_contexts.clear()

        self.bridge = MagicMock()
        self.bridge.is_connected = True
        self.bridge.open_position_count.return_value = 0
        self.bridge.get_positions.return_value = []
        self.bridge.has_position.return_value = False
        self.bridge.has_working_order.return_value = False

        self.orig_realtime = getattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", True)
        self.orig_spy_guard = getattr(config, "SPY_TREND_GUARD_ENABLED", True)
        config.REQUIRE_REALTIME_DATA_FOR_ORDERS = False
        config.SPY_TREND_GUARD_ENABLED = False

    def tearDown(self):
        config.REQUIRE_REALTIME_DATA_FOR_ORDERS = self.orig_realtime
        config.SPY_TREND_GUARD_ENABLED = self.orig_spy_guard
        om.reset_session_skip_state()
        with om._pending_order_contexts_lock:
            om._pending_order_contexts.clear()

    def _create_signal(self, symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY"):
        return TradeSignal(
            symbol=symbol,
            strategy=strategy,
            side=side,
            confidence=0.70,
            entry_price=713.0,
            stop_price=712.0,
            target_price=715.0,
            atr=1.0,
            risk_per_share=1.0,
            reason="Test signal",
        )

    def _create_regime(self, mode=MODE_CAUTION, soxx_pct=-1.64):
        return MarketRegimeResult(
            mode=mode,
            score=-1,
            allow_new_longs=True,
            requires_opening_buffer=False,
            position_scale=0.5,
            confidence_boost=0.0,
            data={"SOXX": {"pct": soxx_pct, "source": "test"}},
        )

    def test_1_blocks_vwap_bounce_long_in_caution_when_soxx_weak(self):
        """Test 1: Blocks VWAP_BOUNCE Long when regime == CAUTION and SOXX <= -1.50%."""
        sig = self._create_signal(symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = self._create_regime(mode=MODE_CAUTION, soxx_pct=-1.64)

        res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
        self.assertEqual(res["status"], "REJECTED")
        self.assertIn("Regime/SOXX Gate VETO", res["reason"])
        self.assertIn("QQQ", res["reason"])
        self.assertIn("-1.64%", res["reason"])

    def test_2_allows_vwap_bounce_long_in_risk_on(self):
        """Test 2: Allows VWAP_BOUNCE Long when regime == RISK_ON even if SOXX is slightly negative (-0.50%)."""
        sig = self._create_signal(symbol="AAPL", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = self._create_regime(mode=MODE_RISK_ON, soxx_pct=-0.50)

        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
            # Should not be blocked by the SOXX gate
            self.assertNotIn("Regime/SOXX Gate VETO", res.get("reason", ""))
            self.assertEqual(res["status"], "DRY_RUN")

    def test_3_allows_vwap_bounce_long_when_soxx_positive_in_caution(self):
        """Test 3: Allows VWAP_BOUNCE Long when SOXX is positive (+0.50%) during non-chip CAUTION."""
        sig = self._create_signal(symbol="AAPL", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = self._create_regime(mode=MODE_CAUTION, soxx_pct=+0.50)

        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
            self.assertNotIn("Regime/SOXX Gate VETO", res.get("reason", ""))
            self.assertEqual(res["status"], "DRY_RUN")

    def test_4_allows_non_vwap_strategies_during_caution_soxx_weak(self):
        """Test 4: Allows non-VWAP strategies (e.g. MOMENTUM_SCALP, GAP_AND_GO) during CAUTION and weak SOXX."""
        sig = self._create_signal(symbol="PLTR", strategy=StrategyName.MOMENTUM_SCALP, side="BUY")
        regime = self._create_regime(mode=MODE_CAUTION, soxx_pct=-2.20)

        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
            self.assertNotIn("Regime/SOXX Gate VETO", res.get("reason", ""))
            self.assertEqual(res["status"], "DRY_RUN")

    def test_5_exact_boundary_soxx_minus_1_50_blocks(self):
        """Test 5 (Boundary Test 6): Exact threshold SOXX == -1.50% MUST BLOCK."""
        sig = self._create_signal(symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = self._create_regime(mode=MODE_CAUTION, soxx_pct=-1.50)

        res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
        self.assertEqual(res["status"], "REJECTED")
        self.assertIn("Regime/SOXX Gate VETO", res["reason"])
        self.assertIn("-1.50%", res["reason"])

    def test_6_boundary_soxx_minus_1_49_passes(self):
        """Test 6: SOXX == -1.49% is just above the threshold, should NOT block."""
        sig = self._create_signal(symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = self._create_regime(mode=MODE_CAUTION, soxx_pct=-1.49)

        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
            self.assertNotIn("Regime/SOXX Gate VETO", res.get("reason", ""))
            self.assertEqual(res["status"], "DRY_RUN")

    def test_7_soxx_missing_none_fails_open_with_warning(self):
        """Test 7: When SOXX data is None/unavailable, fail-open (do not freeze strategy) and log warning."""
        sig = self._create_signal(symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = MarketRegimeResult(
            mode=MODE_CAUTION,
            score=-1,
            allow_new_longs=True,
            requires_opening_buffer=False,
            position_scale=0.5,
            confidence_boost=0.0,
            data={},  # No SOXX data
        )

        with patch("strategies.regime_engine.get_soxx_pct_change", return_value=None), \
             patch("strategies.order_manager.pre_trade_check", return_value=None), \
             self.assertLogs("strategies.order_manager", level="WARNING") as cm:
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
            self.assertNotIn("Regime/SOXX Gate VETO", res.get("reason", ""))
            self.assertEqual(res["status"], "DRY_RUN")
            self.assertTrue(any("SOXX data unavailable" in log_msg for log_msg in cm.output))
            self.assertTrue(any("SOXX data unavailable" in log_msg for log_msg in cm.output))


class TestORBShadowFunnelAccounting(unittest.TestCase):
    def setUp(self):
        self.strat = ConfirmedORBStrategy()
        self.strat.reset_cycle_stats()

    def test_orb_shadow_funnel_accounting_invariant_5m_bars(self):
        """Test that evaluated == accounted holds strictly when 5m bars are supplied."""
        # Create mock 5-minute bar DataFrame (09:30 through 10:15 ET)
        dates = pd.date_range("2026-09-04 09:30:00", "2026-09-04 10:15:00", freq="5min", tz="America/New_York")
        df_5m = pd.DataFrame({
            "open": [100.0 + i for i in range(len(dates))],
            "high": [101.0 + i for i in range(len(dates))],
            "low": [99.0 + i for i in range(len(dates))],
            "close": [100.5 + i for i in range(len(dates))],
            "volume": [10000 + i * 1000 for i in range(len(dates))],
            "vwap": [100.2 + i for i in range(len(dates))],
            "atr": [1.0] * len(dates),
            "ema9": [100.1 + i for i in range(len(dates))],
            "ema20": [99.5 + i for i in range(len(dates))],
        }, index=dates)

        with patch.object(config, "P1_6_ORB_SHADOW_ENABLED", True):
            # Evaluate multiple symbols (some with valid data, some with short/empty data)
            self.strat.evaluate_candidate("AAPL", df_5m)
            self.strat.evaluate_candidate("MSFT", df_5m.iloc[:2])  # Too short
            self.strat.evaluate_candidate("NVDA", pd.DataFrame())  # Empty

        stats = self.strat.get_cycle_stats()
        accounted = (
            stats.candidates
            + stats.outside_window
            + stats.incomplete_range
            + stats.no_breakout
            + stats.below_vwap
            + stats.negative_vwap_slope
            + stats.low_breakout_vol
            + stats.rvol_low
            + stats.stock_in_play_failed
            + stats.market_unaligned
            + stats.sector_unavailable
            + stats.overextended
            + stats.spread_failed
            + stats.zero_risk
        )
        self.assertEqual(stats.evaluated, 3)
        self.assertEqual(accounted, stats.evaluated, f"Funnel mismatch: evaluated={stats.evaluated} != accounted={accounted}")


if __name__ == "__main__":
    unittest.main()
