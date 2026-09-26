"""
tests/test_phase_a_suppression_safety.py -- Specialized Test 1: Suppression Safety.
Verifies that 1,000 benign / non-hard-veto messages containing "ATR", "atr",
"invalid atr", "atr ratio", etc. NEVER contaminate the hard-veto counter or trigger cooldowns.
"""
import unittest
from strategies.order_manager import (
    record_hard_veto_skip,
    check_session_suppression,
    reset_session_skip_state,
    _session_skip_counts,
    _session_suppressed_until,
    _session_permanently_suppressed,
)
from strategies.veto_reasons import VetoReason


class TestPhaseASuppressionSafety(unittest.TestCase):

    def setUp(self):
        reset_session_skip_state()

    def tearDown(self):
        reset_session_skip_state()

    def test_benign_atr_messages_do_not_increment_counter(self):
        """1,000 benign messages containing 'ATR' or 'atr' must NOT increment skip count."""
        symbol = "NVDA"
        benign_templates = [
            "ATR calculation succeeded: ATR=1.52",
            "Checking ATR ratio for volatility sizing: 1.2x atr",
            "ATR period set to 14 bars",
            "Fallback atr computed from session range",
            "invalid atr value detected in warm-up dataframe, falling back to daily ATR",
            "failed to calculate atr due to insufficient bars (< 14)",
            "ATR trailing stop initialized at 1.5x ATR distance",
            "ATR resolution: intraday_df session_bars=35 history_bars=100",
            "ATR(14)=2.4500 noise_floor=0.1200",
            "ATR ratio 1.50 within normal regime bounds",
        ]

        for i in range(1000):
            template = benign_templates[i % len(benign_templates)]
            msg = f"[CYCLE #{i}] {template} (iteration {i})"
            record_hard_veto_skip(symbol, msg, bar_timestamp=f"bar_{i}")

        # Counter must remain strictly 0
        self.assertEqual(_session_skip_counts[symbol], 0)
        self.assertNotIn(symbol, _session_suppressed_until)
        self.assertNotIn(symbol, _session_permanently_suppressed)

        is_suppressed, reason = check_session_suppression(symbol)
        self.assertFalse(is_suppressed)
        self.assertEqual(reason, "")

    def test_legitimate_hard_veto_increments_and_triggers_cooldown(self):
        """Legitimate hard-veto reasons (NOISE_FLOOR, CRITICAL_STOP) DO increment counter."""
        symbol = "AMD"
        # 5 distinct bar vetoes should trigger 45-minute cooldown
        for i in range(5):
            record_hard_veto_skip(symbol, VetoReason.NOISE_FLOOR, bar_timestamp=f"bar_legit_{i}")

        # After 5 hard vetoes, cooldown must be active
        is_suppressed, reason = check_session_suppression(symbol)
        self.assertTrue(is_suppressed)
        self.assertIn("cooldown active", reason)
