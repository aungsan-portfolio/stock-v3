import unittest
import time
from datetime import datetime
from unittest.mock import MagicMock
import pytz
import config
from strategies.regime_engine import (
    evaluate_market_regime,
    MODE_RISK_ON,
    MODE_CAUTION,
    MODE_RISK_OFF,
    MODE_DATA_DEGRADED,
    MarketRegimeResult,
    RegimeStateTracker,
    TEXTURE_VOLATILE,
    TEXTURE_TRENDING,
    _check_data_health,
)
from strategies.session import is_past_opening_buffer


class TestRegimeEngine(unittest.TestCase):

    def _create_mock_bridge(self, spy_price=100.0, spy_prev=100.0, qqq_price=200.0, qqq_prev=200.0, soxx_price=300.0, soxx_prev=300.0, spy_ts=None, qqq_ts=None, soxx_ts=None):
        bridge = MagicMock()
        now_ts = time.time()
        
        def mock_get_price(s, allow_historical=True):
            if s == "SPY":
                return spy_price
            elif s == "QQQ":
                return qqq_price
            elif s == "SOXX":
                return soxx_price
            return 100.0

        bridge.get_price = mock_get_price

        def mock_fetch(sym, *args, **kwargs):
            if sym == "SPY":
                if spy_price is None: return []
                b1 = MagicMock(close=spy_prev, timestamp=spy_ts or now_ts)
                b2 = MagicMock(close=spy_price, timestamp=spy_ts or now_ts)
                return [b1, b2]
            elif sym == "QQQ":
                if qqq_price is None: return []
                b1 = MagicMock(close=qqq_prev, timestamp=qqq_ts or now_ts)
                b2 = MagicMock(close=qqq_price, timestamp=qqq_ts or now_ts)
                return [b1, b2]
            elif sym == "SOXX":
                if soxx_price is None: return []
                b1 = MagicMock(close=soxx_prev, timestamp=soxx_ts or now_ts)
                b2 = MagicMock(close=soxx_price, timestamp=soxx_ts or now_ts)
                return [b1, b2]
            return []

        bridge.fetch_historical_data = mock_fetch
        return bridge

    def test_risk_on_healthy_market(self):
        """Test that healthy market conditions produce RISK_ON mode."""
        bridge = self._create_mock_bridge(spy_price=100.0, spy_prev=99.8, qqq_price=200.0, qqq_prev=199.8, soxx_price=300.0, soxx_prev=299.8)
        res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=5)
        self.assertEqual(res.mode, MODE_RISK_ON)
        self.assertTrue(res.allow_new_longs)
        self.assertFalse(res.halt_new_entries)
        self.assertEqual(res.score, 0)
        self.assertEqual(res.position_scale, 1.0)
        self.assertEqual(res.confidence_boost, 0.0)

    def test_scanner_candidate_count_zero_vs_none(self):
        """Test that candidate_count=0 applies -2 penalty, whereas None applies 0 penalty."""
        bridge = self._create_mock_bridge(spy_price=100.0, spy_prev=100.0, qqq_price=200.0, qqq_prev=200.0)

        # candidate_count = 0 -> -2 penalty
        res_zero = evaluate_market_regime(bridge=bridge, scanner_candidate_count=0)
        self.assertEqual(res_zero.factor_scores.get("scanner"), -2)
        self.assertIn("Scanner Health Guard active", "".join(res_zero.reasons))

        # candidate_count = None -> no penalty
        res_none = evaluate_market_regime(bridge=bridge, scanner_candidate_count=None)
        self.assertNotIn("scanner", res_none.factor_scores)
        self.assertIn("scanner_candidate_count_unavailable", res_none.reasons)

    def test_risk_off_severe_selloff(self):
        """Test that severe selloff across SPY, QQQ, SOXX, and 0 candidates produces RISK_OFF mode."""
        # SPY down -2.0%, QQQ down -2.5%, SOXX down -3.33%
        bridge = self._create_mock_bridge(spy_price=98.0, spy_prev=100.0, qqq_price=195.0, qqq_prev=200.0, soxx_price=290.0, soxx_prev=300.0)

        # Score: SPY (-2) + QQQ (-2) + SOXX (-2) + Scanner (0 candidates = -2) = -8
        res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=0)
        self.assertEqual(res.mode, MODE_RISK_OFF)
        self.assertFalse(res.allow_new_longs)
        self.assertTrue(res.halt_new_entries)
        self.assertLessEqual(res.score, -4)
        self.assertEqual(res.position_scale, 0.0)

    def test_caution_mode_mixed_market(self):
        """Test mixed market conditions produce CAUTION mode with 09:45 buffer requirement."""
        # SPY down -0.9% (-2 pts), QQQ down -0.5% (-1 pt)
        bridge = self._create_mock_bridge(spy_price=99.1, spy_prev=100.0, qqq_price=199.0, qqq_prev=200.0, soxx_price=300.0, soxx_prev=300.0)

        res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=3)
        self.assertEqual(res.mode, MODE_CAUTION)
        self.assertTrue(res.requires_opening_buffer)
        self.assertEqual(res.position_scale, 0.5)
        self.assertEqual(res.confidence_boost, getattr(config, "CAUTION_CONFIDENCE_BOOST", 0.0))

    def test_opening_buffer_timezone_safe(self):
        """Test is_past_opening_buffer helper at various Eastern times."""
        tz = pytz.timezone("America/New_York")
        
        # 09:31 ET -> False
        dt_0931 = datetime(2026, 7, 30, 9, 31, 0, tzinfo=tz)
        self.assertFalse(is_past_opening_buffer(15, now=dt_0931))

        # 09:44:59 ET -> False
        dt_0944 = datetime(2026, 7, 30, 9, 44, 59, tzinfo=tz)
        self.assertFalse(is_past_opening_buffer(15, now=dt_0944))

        # 09:45:00 ET -> True
        dt_0945 = datetime(2026, 7, 30, 9, 45, 0, tzinfo=tz)
        self.assertTrue(is_past_opening_buffer(15, now=dt_0945))

        # 10:00:00 ET -> True
        dt_1000 = datetime(2026, 7, 30, 10, 0, 0, tzinfo=tz)
        self.assertTrue(is_past_opening_buffer(15, now=dt_1000))

    def test_data_health_gate_spy_missing_is_degraded(self):
        """P1: Missing SPY triggers DATA_DEGRADED mode with fail-closed entries halted."""
        bridge = self._create_mock_bridge(spy_price=0.0, qqq_price=200.0, qqq_prev=200.0)
        # Mock yfinance fallback also returning None
        import yfinance as yf
        orig_ticker = yf.Ticker
        try:
            yf.Ticker = lambda sym: MagicMock(fast_info=None)
            res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=5)
            self.assertEqual(res.mode, MODE_DATA_DEGRADED)
            self.assertTrue(res.halt_new_entries)
            self.assertFalse(res.allow_new_longs)
            self.assertFalse(res.allow_new_shorts)
            self.assertEqual(res.position_scale, 0.0)
            self.assertEqual(res.market_texture, TEXTURE_VOLATILE)
            self.assertIn("SPY benchmark data unavailable", "".join(res.reasons))
        finally:
            yf.Ticker = orig_ticker

    def test_data_health_gate_qqq_missing_is_degraded(self):
        """P1: Missing QQQ triggers DATA_DEGRADED mode with fail-closed entries halted."""
        bridge = self._create_mock_bridge(spy_price=100.0, spy_prev=100.0, qqq_price=0.0)
        import yfinance as yf
        orig_ticker = yf.Ticker
        try:
            yf.Ticker = lambda sym: MagicMock(fast_info=None)
            res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=5)
            self.assertEqual(res.mode, MODE_DATA_DEGRADED)
            self.assertTrue(res.halt_new_entries)
            self.assertFalse(res.allow_new_longs)
            self.assertFalse(res.allow_new_shorts)
            self.assertEqual(res.position_scale, 0.0)
            self.assertIn("QQQ benchmark data unavailable", "".join(res.reasons))
        finally:
            yf.Ticker = orig_ticker

    def test_data_health_gate_stale_data_is_degraded(self):
        """P1: Stale SPY timestamp (> 60s) triggers DATA_DEGRADED mode."""
        stale_ts = time.time() - 120.0  # 120s old > 60s threshold
        bridge = self._create_mock_bridge(spy_price=100.0, spy_prev=100.0, qqq_price=200.0, qqq_prev=200.0, spy_ts=stale_ts)

        res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=5)
        self.assertEqual(res.mode, MODE_DATA_DEGRADED)
        self.assertTrue(res.halt_new_entries)
        self.assertFalse(res.allow_new_longs)
        self.assertIn("SPY benchmark data stale", "".join(res.reasons))

    def test_data_health_gate_exception_fails_closed(self):
        """P1: Any unexpected exception in _check_data_health fails closed."""
        bridge = MagicMock()
        bridge.get_price.side_effect = RuntimeError("Mock network drop")
        bridge.fetch_historical_data.side_effect = RuntimeError("Mock network drop")

        import yfinance as yf
        orig_ticker = yf.Ticker
        try:
            yf.Ticker = lambda sym: MagicMock(fast_info=None)
            ok, reason, _ = _check_data_health(bridge=bridge)
            self.assertFalse(ok)
            res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=5)
            self.assertEqual(res.mode, MODE_DATA_DEGRADED)
            self.assertTrue(res.halt_new_entries)
        finally:
            yf.Ticker = orig_ticker

    def test_soxx_missing_does_not_trigger_degraded_and_no_penalty(self):
        """P1: Missing SOXX is non-penalizing (0 pts) and does NOT trigger DATA_DEGRADED."""
        bridge = self._create_mock_bridge(spy_price=100.0, spy_prev=100.0, qqq_price=200.0, qqq_prev=200.0, soxx_price=0.0)
        import yfinance as yf
        orig_ticker = yf.Ticker
        try:
            yf.Ticker = lambda sym: MagicMock(fast_info=None) if sym == "SOXX" else orig_ticker(sym)
            res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=5)
            self.assertEqual(res.mode, MODE_RISK_ON)
            self.assertEqual(res.score, 0)
            self.assertTrue(res.allow_new_longs)
            self.assertFalse(res.halt_new_entries)
            self.assertIn("SOXX_data_unavailable", "".join(res.reasons))
        finally:
            yf.Ticker = orig_ticker

    def test_soxx_symmetrized_scoring(self):
        """P1: SOXX strong (+2.5%) adds +1 pt symmetric score; weak (-2.5%) subtracts -2 pts."""
        # Strong SOXX (+2.5%)
        b_strong = self._create_mock_bridge(spy_price=100.0, spy_prev=100.0, qqq_price=200.0, qqq_prev=200.0, soxx_price=307.5, soxx_prev=300.0)
        res_strong = evaluate_market_regime(bridge=b_strong, scanner_candidate_count=5)
        self.assertEqual(res_strong.factor_scores.get("SOXX"), 1)

        # Weak SOXX (-2.5%)
        b_weak = self._create_mock_bridge(spy_price=100.0, spy_prev=100.0, qqq_price=200.0, qqq_prev=200.0, soxx_price=292.5, soxx_prev=300.0)
        res_weak = evaluate_market_regime(bridge=b_weak, scanner_candidate_count=5)
        self.assertEqual(res_weak.factor_scores.get("SOXX"), -2)

    def test_regime_state_tracker_hysteresis_cooldown(self):
        """P1: RegimeStateTracker requires N consecutive healthy bars to unlatch DATA_DEGRADED."""
        tracker = RegimeStateTracker(recovery_bars=3)
        self.assertFalse(tracker.is_degraded)
        self.assertEqual(tracker.consecutive_healthy, 0)

        degraded_res = MarketRegimeResult(
            mode=MODE_DATA_DEGRADED,
            score=0,
            allow_new_longs=False,
            requires_opening_buffer=True,
            position_scale=0.0,
            confidence_boost=0.0,
            allow_new_shorts=False,
            halt_new_entries=True,
        )

        healthy_res = MarketRegimeResult(
            mode=MODE_RISK_ON,
            score=0,
            allow_new_longs=True,
            requires_opening_buffer=False,
            position_scale=1.0,
            confidence_boost=0.0,
            allow_new_shorts=True,
            halt_new_entries=False,
        )

        # 1. First event is degraded -> latches degraded
        r1 = tracker.update(degraded_res)
        self.assertEqual(r1.mode, MODE_DATA_DEGRADED)
        self.assertTrue(r1.halt_new_entries)
        self.assertTrue(tracker.is_degraded)
        self.assertEqual(tracker.consecutive_healthy, 0)

        # 2. Fresh Bar 1 -> still latched in cool-down recovery
        r2 = tracker.update(healthy_res)
        self.assertEqual(r2.mode, MODE_DATA_DEGRADED)
        self.assertTrue(r2.halt_new_entries)
        self.assertEqual(tracker.consecutive_healthy, 1)
        self.assertTrue(tracker.is_degraded)

        # 3. Fresh Bar 2 -> still latched in cool-down recovery
        r3 = tracker.update(healthy_res)
        self.assertEqual(r3.mode, MODE_DATA_DEGRADED)
        self.assertTrue(r3.halt_new_entries)
        self.assertEqual(tracker.consecutive_healthy, 2)
        self.assertTrue(tracker.is_degraded)

        # 4. Fresh Bar 3 -> reaches threshold (3), unlatches and returns RISK_ON
        r4 = tracker.update(healthy_res)
        self.assertEqual(r4.mode, MODE_RISK_ON)
        self.assertTrue(r4.allow_new_longs)
        self.assertFalse(r4.halt_new_entries)
        self.assertFalse(tracker.is_degraded)

    def test_regime_state_tracker_reuse_preserves_state(self):
        """P1: Verify tracker instance maintains state correctly across multiple ticks."""
        tracker = RegimeStateTracker(recovery_bars=2)
        degraded_res = MarketRegimeResult(mode=MODE_DATA_DEGRADED, score=0, allow_new_longs=False, requires_opening_buffer=True, position_scale=0.0, confidence_boost=0.0, halt_new_entries=True)
        healthy_res = MarketRegimeResult(mode=MODE_RISK_ON, score=0, allow_new_longs=True, requires_opening_buffer=False, position_scale=1.0, confidence_boost=0.0, halt_new_entries=False)

        tracker.update(degraded_res)
        self.assertTrue(tracker.is_degraded)

        tracker.update(healthy_res)
        self.assertTrue(tracker.is_degraded)

        # Second healthy tick unlatches
        tracker.update(healthy_res)
        self.assertFalse(tracker.is_degraded)

        # Explicit reset clears latch
        tracker.update(degraded_res)
        self.assertTrue(tracker.is_degraded)
        tracker.reset()
        self.assertFalse(tracker.is_degraded)

    def test_downstream_boolean_invariants_across_all_modes(self):
        """P1: Verify halt_new_entries and allow_new_longs are accurate across all modes."""
        r_deg = MarketRegimeResult(mode=MODE_DATA_DEGRADED, score=0, allow_new_longs=False, requires_opening_buffer=True, position_scale=0.0, confidence_boost=0.0, halt_new_entries=True)
        self.assertTrue(r_deg.halt_new_entries or not r_deg.allow_new_longs)

        r_off = MarketRegimeResult(mode=MODE_RISK_OFF, score=-5, allow_new_longs=False, requires_opening_buffer=True, position_scale=0.0, confidence_boost=0.25, halt_new_entries=True)
        self.assertTrue(r_off.halt_new_entries or not r_off.allow_new_longs)

        r_on = MarketRegimeResult(mode=MODE_RISK_ON, score=0, allow_new_longs=True, requires_opening_buffer=False, position_scale=1.0, confidence_boost=0.0, halt_new_entries=False)
        self.assertFalse(r_on.halt_new_entries)
        self.assertTrue(r_on.allow_new_longs)


if __name__ == "__main__":
    unittest.main()

