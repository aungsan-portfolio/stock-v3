"""
test_soxx_fail_closed_regression.py — Comprehensive Regression Suite for SOXX Fail-Closed Closure

Verifies:
1. Targeted fail-closed regression across data health and execution boundaries (10/10)
2. Valid SOXX scoring matrix preservation (5/5)
3. Absence of legacy fail-open paths and cached fallbacks
4. Cache clearing and non-reuse semantics
"""
import inspect
import os
import sys
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from strategies.base import TradeSignal
from strategies.constants import StrategyName
import strategies.order_manager as om
import strategies.regime_engine as re
from strategies.regime_engine import (
    MarketRegimeResult,
    MODE_CAUTION,
    MODE_RISK_ON,
    MODE_RISK_OFF,
    MODE_DATA_DEGRADED,
    _check_data_health,
    evaluate_market_regime,
    get_soxx_pct_change,
    clear_soxx_cache,
)


class TestSOXXFailClosedTargetedRegression(unittest.TestCase):
    """10-point targeted regression test suite for SOXX fail-closed contract."""

    def setUp(self):
        om.reset_session_skip_state()
        with om._pending_order_contexts_lock:
            om._pending_order_contexts.clear()

        self.orig_realtime = getattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", True)
        self.orig_spy_guard = getattr(config, "SPY_TREND_GUARD_ENABLED", True)
        self.orig_entry_readiness = getattr(config, "DAYTRADE_REQUIRE_ENTRY_READINESS", True)
        self.orig_health_cfg = dict(getattr(config, "DATA_HEALTH_SETTINGS", {}))

        config.REQUIRE_REALTIME_DATA_FOR_ORDERS = False
        config.SPY_TREND_GUARD_ENABLED = False
        config.DAYTRADE_REQUIRE_ENTRY_READINESS = False

        self.bridge = MagicMock()
        self.bridge.is_connected = True
        self.bridge.open_position_count.return_value = 0
        self.bridge.get_positions.return_value = []
        self.bridge.has_position.return_value = False
        self.bridge.has_working_order.return_value = False

        self._count_patch = patch("strategies.order_manager.today_trade_count", return_value=0)
        self._count_patch.start()
        clear_soxx_cache()

    def tearDown(self):
        self._count_patch.stop()
        config.REQUIRE_REALTIME_DATA_FOR_ORDERS = self.orig_realtime
        config.SPY_TREND_GUARD_ENABLED = self.orig_spy_guard
        config.DAYTRADE_REQUIRE_ENTRY_READINESS = self.orig_entry_readiness
        config.DATA_HEALTH_SETTINGS = self.orig_health_cfg
        om.reset_session_skip_state()
        with om._pending_order_contexts_lock:
            om._pending_order_contexts.clear()
        clear_soxx_cache()

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

    # ── Test 1: SOXX in critical_symbols and missing pct triggers DATA_DEGRADED fail-closed ──
    def test_01_soxx_missing_routes_to_data_degraded(self):
        def mock_get_pct(sym, bridge):
            now = time.time()
            if sym == "SPY":
                return -0.20, "mock", now
            elif sym == "QQQ":
                return -0.30, "mock", now
            elif sym == "SOXX":
                return None, "mock_unavailable", None
            return None, "unknown", None

        with patch("strategies.regime_engine._get_premarket_pct_change", side_effect=mock_get_pct):
            result = evaluate_market_regime(bridge=self.bridge)
            self.assertEqual(result.mode, MODE_DATA_DEGRADED)
            self.assertTrue(result.halt_new_entries)
            self.assertFalse(result.allow_new_longs)
            self.assertIn("SOXX benchmark data unavailable", result.reasons[0])
            self.assertEqual(result.data["SOXX"]["status"], "MISSING")

    # ── Test 2: SOXX stale timestamp triggers DATA_DEGRADED fail-closed ──
    def test_02_soxx_stale_timestamp_routes_to_data_degraded(self):
        now = time.time()
        def mock_get_pct(sym, bridge):
            if sym == "SPY":
                return -0.20, "mock", now
            elif sym == "QQQ":
                return -0.30, "mock", now
            elif sym == "SOXX":
                return -0.50, "mock", now - 300.0  # 300s old > 60s threshold
            return None, "unknown", None

        with patch("strategies.regime_engine._get_premarket_pct_change", side_effect=mock_get_pct):
            result = evaluate_market_regime(bridge=self.bridge)
            self.assertEqual(result.mode, MODE_DATA_DEGRADED)
            self.assertTrue(result.halt_new_entries)
            self.assertFalse(result.allow_new_longs)
            self.assertIn("SOXX benchmark data stale", result.reasons[0])
            self.assertEqual(result.data["SOXX"]["status"], "STALE")

    # ── Test 3: SOXX un-timestamped triggers DATA_DEGRADED fail-closed ──
    def test_03_soxx_untimestamped_routes_to_data_degraded(self):
        now = time.time()
        def mock_get_pct(sym, bridge):
            if sym == "SPY":
                return -0.20, "mock", now
            elif sym == "QQQ":
                return -0.30, "mock", now
            elif sym == "SOXX":
                return -0.50, "mock", None  # un-timestamped
            return None, "unknown", None

        with patch("strategies.regime_engine._get_premarket_pct_change", side_effect=mock_get_pct):
            result = evaluate_market_regime(bridge=self.bridge)
            self.assertEqual(result.mode, MODE_DATA_DEGRADED)
            self.assertTrue(result.halt_new_entries)
            self.assertIn("SOXX benchmark data un-timestamped", result.reasons[0])
            self.assertEqual(result.data["SOXX"]["status"], "STALE")

    # ── Test 4: SOXX cache is cleared before health evaluation and failed refresh does NOT reuse prior ──
    def test_04_cache_cleared_and_not_reused_on_failure(self):
        # First: seed cache
        with re._last_soxx_lock:
            re._last_soxx_pct = -1.25
        self.assertEqual(get_soxx_pct_change(), -1.25)

        # Second: evaluate when SOXX is unavailable
        def mock_get_pct(sym, bridge):
            if sym == "SOXX":
                return None, "unavailable", None
            return -0.10, "mock", time.time()

        with patch("strategies.regime_engine._get_premarket_pct_change", side_effect=mock_get_pct):
            _check_data_health(bridge=self.bridge)
            # Must be None, NOT -1.25!
            self.assertIsNone(get_soxx_pct_change())

    # ── Test 5: SOXX cache populated ONLY when health check evaluates SOXX as VALID ──
    def test_05_cache_populated_only_on_valid(self):
        now = time.time()
        def mock_get_pct(sym, bridge):
            return -0.85, "mock", now

        with patch("strategies.regime_engine._get_premarket_pct_change", side_effect=mock_get_pct):
            ok, reason, health_data = _check_data_health(bridge=self.bridge)
            self.assertTrue(ok)
            self.assertEqual(health_data["SOXX"]["status"], "VALID")
            self.assertEqual(get_soxx_pct_change(), -0.85)

    # ── Test 6: Regime scoring consumes validated SOXX snapshot (no second unvalidated fetch) ──
    def test_06_regime_scoring_consumes_validated_snapshot_without_second_fetch(self):
        fetch_calls = []
        now = time.time()
        def mock_get_pct(sym, bridge):
            fetch_calls.append(sym)
            return -1.15, "mock", now

        with patch("strategies.regime_engine._get_premarket_pct_change", side_effect=mock_get_pct):
            res = evaluate_market_regime(bridge=self.bridge)
            # Critical symbols are ["SPY", "QQQ", "SOXX"] -> each fetched exactly once!
            self.assertEqual(fetch_calls.count("SOXX"), 1)
            self.assertEqual(res.factor_scores.get("SOXX"), -1)
            self.assertEqual(res.data["SOXX"]["status"], "VALID")

    # ── Test 7: order_manager blocks VWAP_BOUNCE when regime_result is None (fail-closed) ──
    def test_07_order_manager_blocks_when_regime_result_none(self):
        sig = self._create_signal(symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=None)
            self.assertEqual(res["status"], "REJECTED")
            self.assertIn("Regime/SOXX Gate VETO", res["reason"])
            self.assertIn("Missing regime context", res["reason"])

    # ── Test 8: order_manager blocks VWAP_BOUNCE when SOXX is missing/None in CAUTION ──
    def test_08_order_manager_blocks_when_soxx_missing_in_caution(self):
        sig = self._create_signal(symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = MarketRegimeResult(
            mode=MODE_CAUTION,
            score=-1,
            allow_new_longs=True,
            requires_opening_buffer=False,
            position_scale=0.5,
            confidence_boost=0.0,
            data={"SOXX": {"pct": None, "status": "MISSING"}},
        )
        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
            self.assertEqual(res["status"], "REJECTED")
            self.assertIn("Regime/SOXX Gate VETO", res["reason"])
            self.assertIn("unavailable or invalid", res["reason"])

    # ── Test 9: order_manager blocks VWAP_BOUNCE when SOXX status is STALE in CAUTION ──
    def test_09_order_manager_blocks_when_soxx_stale_in_caution(self):
        sig = self._create_signal(symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = MarketRegimeResult(
            mode=MODE_CAUTION,
            score=-1,
            allow_new_longs=True,
            requires_opening_buffer=False,
            position_scale=0.5,
            confidence_boost=0.0,
            data={"SOXX": {"pct": -0.20, "status": "STALE"}},
        )
        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
            self.assertEqual(res["status"], "REJECTED")
            self.assertIn("Regime/SOXX Gate VETO", res["reason"])
            self.assertIn("unavailable or invalid", res["reason"])

    # ── Test 10: order_manager blocks VWAP_BOUNCE when SOXX <= -1.50% in CAUTION ──
    def test_10_order_manager_blocks_when_soxx_below_threshold(self):
        sig = self._create_signal(symbol="QQQ", strategy=StrategyName.VWAP_BOUNCE, side="BUY")
        regime = MarketRegimeResult(
            mode=MODE_CAUTION,
            score=-1,
            allow_new_longs=True,
            requires_opening_buffer=False,
            position_scale=0.5,
            confidence_boost=0.0,
            data={"SOXX": {"pct": -1.51, "status": "VALID"}},
        )
        with patch("strategies.order_manager.pre_trade_check", return_value=None):
            res = om.execute_signal(sig, bridge=self.bridge, equity=100000.0, current_pnl=0.0, dry_run=True, regime_result=regime)
            self.assertEqual(res["status"], "REJECTED")
            self.assertIn("Regime/SOXX Gate VETO", res["reason"])
            self.assertIn("-1.51%", res["reason"])


class TestValidSOXXScoringMatrix(unittest.TestCase):
    """5/5 verification of valid SOXX scoring matrix preservation."""

    def _eval_with_soxx(self, soxx_pct: float) -> MarketRegimeResult:
        now = time.time()
        def mock_get_pct(sym, bridge):
            if sym == "SPY":
                return -0.10, "mock", now
            elif sym == "QQQ":
                return -0.10, "mock", now
            elif sym == "SOXX":
                return soxx_pct, "mock", now
            return None, "mock", None

        with patch("strategies.regime_engine._get_premarket_pct_change", side_effect=mock_get_pct):
            return evaluate_market_regime(bridge=MagicMock())

    def test_matrix_1_severe_loss(self):
        # -2.01 -> -2
        res = self._eval_with_soxx(-2.01)
        self.assertEqual(res.factor_scores.get("SOXX"), -2)

    def test_matrix_2_moderate_loss(self):
        # -1.01 -> -1
        res = self._eval_with_soxx(-1.01)
        self.assertEqual(res.factor_scores.get("SOXX"), -1)

    def test_matrix_3_mild_loss(self):
        # -0.50 -> 0
        res = self._eval_with_soxx(-0.50)
        self.assertEqual(res.factor_scores.get("SOXX"), 0)

    def test_matrix_4_flat(self):
        # 0.00 -> 0
        res = self._eval_with_soxx(0.00)
        self.assertEqual(res.factor_scores.get("SOXX"), 0)

    def test_matrix_5_strong_gain(self):
        # +2.01 -> +1
        res = self._eval_with_soxx(2.01)
        self.assertEqual(res.factor_scores.get("SOXX"), 1)


class TestCodeIntegrityAndScopeDiscipline(unittest.TestCase):
    """Verifies that legacy fail-open paths and cached fallbacks are strictly absent."""

    def test_legacy_fail_open_strings_absent_in_order_manager(self):
        om_src = inspect.getsource(om)
        self.assertNotIn("Allowing %s VWAP_BOUNCE entry (fail-open)", om_src)
        self.assertNotIn("(fail-open)", om_src)

    def test_order_manager_cached_soxx_fallback_removed(self):
        om_src = inspect.getsource(om.execute_signal)
        self.assertNotIn("get_soxx_pct_change", om_src)

    def test_legacy_fail_open_string_absent_in_regime_engine(self):
        re_src = inspect.getsource(re.evaluate_market_regime)
        self.assertNotIn("SOXX_data_unavailable (optional sector signal, zero penalty)", re_src)
        self.assertNotIn("optional sector signal, zero penalty", re_src)


if __name__ == "__main__":
    unittest.main()
