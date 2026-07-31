import unittest
from datetime import datetime, time
from unittest.mock import MagicMock
import pytz
from strategies.regime_engine import (
    evaluate_market_regime,
    MODE_RISK_ON,
    MODE_CAUTION,
    MODE_RISK_OFF,
    MarketRegimeResult
)
from strategies.session import is_past_opening_buffer

class TestRegimeEngine(unittest.TestCase):

    def test_risk_on_healthy_market(self):
        """Test that healthy market conditions produce RISK_ON mode."""
        bridge = MagicMock()
        bridge.get_price.side_effect = lambda s, allow_historical=True: 100.0 if s == "SPY" else (200.0 if s == "QQQ" else 300.0)
        
        mock_spy_bar1 = MagicMock(close=99.8)
        mock_spy_bar2 = MagicMock(close=100.0)
        mock_qqq_bar1 = MagicMock(close=199.8)
        mock_qqq_bar2 = MagicMock(close=200.0)
        mock_soxx_bar1 = MagicMock(close=299.8)
        mock_soxx_bar2 = MagicMock(close=300.0)

        def mock_fetch(sym, *args, **kwargs):
            if sym == "SPY":
                return [mock_spy_bar1, mock_spy_bar2]
            elif sym == "QQQ":
                return [mock_qqq_bar1, mock_qqq_bar2]
            else:
                return [mock_soxx_bar1, mock_soxx_bar2]

        bridge.fetch_historical_data = mock_fetch

        res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=5)
        self.assertEqual(res.mode, MODE_RISK_ON)
        self.assertTrue(res.allow_new_longs)
        self.assertEqual(res.score, 0)
        self.assertEqual(res.position_scale, 1.0)
        self.assertEqual(res.confidence_boost, 0.0)

    def test_scanner_candidate_count_zero_vs_none(self):
        """Test that candidate_count=0 applies -2 penalty, whereas None applies 0 penalty."""
        bridge = MagicMock()
        bridge.get_price.side_effect = lambda s, allow_historical=True: 100.0
        bridge.fetch_historical_data = lambda s, *a, **k: [MagicMock(close=100.0), MagicMock(close=100.0)]

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
        bridge = MagicMock()
        bridge.get_price.side_effect = lambda s, allow_historical=True: 98.0 if s == "SPY" else (195.0 if s == "QQQ" else 290.0)
        
        # SPY down -2.0%, QQQ down -2.5%, SOXX down -3.3%
        mock_spy_bar1 = MagicMock(close=100.0)
        mock_spy_bar2 = MagicMock(close=98.0)
        mock_qqq_bar1 = MagicMock(close=200.0)
        mock_qqq_bar2 = MagicMock(close=195.0)
        mock_soxx_bar1 = MagicMock(close=300.0)
        mock_soxx_bar2 = MagicMock(close=290.0)

        def mock_fetch(sym, *args, **kwargs):
            if sym == "SPY":
                return [mock_spy_bar1, mock_spy_bar2]
            elif sym == "QQQ":
                return [mock_qqq_bar1, mock_qqq_bar2]
            else:
                return [mock_soxx_bar1, mock_soxx_bar2]

        bridge.fetch_historical_data = mock_fetch

        # Score: SPY (-2) + QQQ (-2) + SOXX (-2) + Scanner (0 candidates = -2) = -8
        res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=0)
        self.assertEqual(res.mode, MODE_RISK_OFF)
        self.assertFalse(res.allow_new_longs)
        self.assertLessEqual(res.score, -4)
        self.assertEqual(res.position_scale, 0.0)

    def test_caution_mode_mixed_market(self):
        """Test mixed market conditions produce CAUTION mode with 09:45 buffer requirement."""
        bridge = MagicMock()
        bridge.get_price.side_effect = lambda s, allow_historical=True: 99.1 if s == "SPY" else (199.0 if s == "QQQ" else 300.0)
        
        # SPY down -0.9% (-2 pts)
        mock_spy_bar1 = MagicMock(close=100.0)
        mock_spy_bar2 = MagicMock(close=99.1)
        mock_qqq_bar1 = MagicMock(close=200.0)
        mock_qqq_bar2 = MagicMock(close=199.0)

        def mock_fetch(sym, *args, **kwargs):
            if sym == "SPY":
                return [mock_spy_bar1, mock_spy_bar2]
            else:
                return [mock_qqq_bar1, mock_qqq_bar2]

        bridge.fetch_historical_data = mock_fetch

        res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=3)
        self.assertEqual(res.mode, MODE_CAUTION)
        self.assertTrue(res.allow_new_longs)
        self.assertTrue(res.requires_opening_buffer)
        self.assertEqual(res.position_scale, 0.5)

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

if __name__ == "__main__":
    unittest.main()
