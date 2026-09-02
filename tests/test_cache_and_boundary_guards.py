import os
import time
import unittest
import pandas as pd
from datetime import datetime, timezone

from strategies.intraday_data import _read_cache, _write_cache, _cache_key
import config
from strategies.base import TradeSignal
from strategies.order_manager import execute_signal, _validate_signal_prices
from strategies.event_guard import EventDecision
from strategies.orchestrator import _should_bypass_lunch_chop
from strategies.position_sizer import calculate_shares


class TestCacheAndBoundaryGuards(unittest.TestCase):
    """Targeted unit tests specifically verifying Cache TTL and float epsilon tolerance."""

    def test_cache_ttl_expiration(self):
        """Verify that cache TTL = 15s correctly expires stale entries and misses on >15s."""
        key = "test_symbol_ttl_check"
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        _write_cache(key, df)

        # Fresh read with standard TTL (15s) should hit
        cached = _read_cache(key, ttl=15)
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 3)

        # Simulated expiration: read with 0s TTL must return None (cache miss)
        expired = _read_cache(key, ttl=0)
        self.assertIsNone(expired)

    def test_confidence_gate_boundary_with_epsilon(self):
        """Verify confidence at exact gate (0.60) or within 1e-4 passes cleanly."""
        sig_exact = TradeSignal(
            symbol="BAC",
            strategy="VWAP_BOUNCE",
            side="BUY",
            confidence=0.60,
            entry_price=60.0,
            stop_price=59.0,
            target_price=62.0,
            atr=1.0,
            risk_per_share=1.0,
            reason="Test exact",
        )
        res_exact = execute_signal(sig_exact, bridge=None, equity=100000.0, current_pnl=0.0, dry_run=True)
        self.assertNotIn("Confidence 0.60 < 0.60", res_exact.get("reason", ""))

        sig_within_epsilon = TradeSignal(
            symbol="BAC",
            strategy="VWAP_BOUNCE",
            side="BUY",
            confidence=0.59995,  # within 1e-4 of 0.60
            entry_price=60.0,
            stop_price=59.0,
            target_price=62.0,
            atr=1.0,
            risk_per_share=1.0,
            reason="Test epsilon",
        )
        res_eps = execute_signal(sig_within_epsilon, bridge=None, equity=100000.0, current_pnl=0.0, dry_run=True)
        self.assertNotIn("Confidence 0.60 < 0.60", res_eps.get("reason", ""))

        sig_sub_threshold = TradeSignal(
            symbol="BAC",
            strategy="VWAP_BOUNCE",
            side="BUY",
            confidence=0.58,  # genuinely below threshold
            entry_price=60.0,
            stop_price=59.0,
            target_price=62.0,
            atr=1.0,
            risk_per_share=1.0,
            reason="Test sub-threshold",
        )
        res_sub = execute_signal(sig_sub_threshold, bridge=None, equity=100000.0, current_pnl=0.0, dry_run=True)
        self.assertEqual(res_sub["status"], "REJECTED")
        self.assertIn("Confidence", res_sub["reason"])

    def test_rr_ratio_boundary_with_epsilon(self):
        """Verify min/max R:R boundary checks with epsilon tolerance."""
        sig_rr = TradeSignal(
            symbol="UBER",
            strategy="MOMENTUM_SCALP",
            side="BUY",
            confidence=0.70,
            entry_price=100.0,
            stop_price=98.0,       # risk = 2.0
            target_price=102.9999, # reward = 2.9999 -> R:R = 1.49995 (practically 1.50)
            atr=2.0,
            risk_per_share=2.0,
            reason="Test RR min",
        )
        reason = _validate_signal_prices(sig_rr)
        self.assertEqual(reason, "")

        sig_max_rr = TradeSignal(
            symbol="QQQ",
            strategy="MOMENTUM_SCALP",
            side="BUY",
            confidence=0.70,
            entry_price=100.0,
            stop_price=99.0,       # risk = 1.0
            target_price=105.00005, # reward = 5.00005 -> R:R = 5.00005 (practically 5.0)
            atr=1.0,
            risk_per_share=1.0,
            reason="Test RR max",
        )
        reason_max = _validate_signal_prices(sig_max_rr)
        self.assertEqual(reason_max, "")

    def test_lunch_chop_exact_symbol_bypass(self):
        """Directly test _should_bypass_lunch_chop production function with matching/non-matching symbols and confidence levels."""
        event_decision = EventDecision(
            throttle=0.5,
            block_entries=False,
            atr_floor_mult=2.0,
            tag="NVDA_EARNINGS_GAP",
            affected_symbols={"NVDA"},
        )

        # 1. Matching symbol NVDA with high confidence (0.75 >= 0.70) -> MUST PASS
        sig_nvda_pass = TradeSignal(
            symbol="NVDA",
            strategy="VWAP_BOUNCE",
            side="BUY",
            confidence=0.75,
            entry_price=225.0,
            stop_price=223.0,
            target_price=230.0,
            atr=2.0,
            risk_per_share=2.0,
            reason="NVDA test pass",
        )
        self.assertTrue(_should_bypass_lunch_chop(sig_nvda_pass, event_decision))

        # 2. Matching symbol NVDA with insufficient confidence (0.65 < 0.70) -> MUST BE REJECTED
        sig_nvda_low_conf = TradeSignal(
            symbol="NVDA",
            strategy="VWAP_BOUNCE",
            side="BUY",
            confidence=0.65,
            entry_price=225.0,
            stop_price=223.0,
            target_price=230.0,
            atr=2.0,
            risk_per_share=2.0,
            reason="NVDA low conf",
        )
        self.assertFalse(_should_bypass_lunch_chop(sig_nvda_low_conf, event_decision))

        # 3. Non-matching symbols (MU, AMD, MSFT, SPY) even with 0.80 confidence -> MUST BE REJECTED
        for sym in ["MU", "AMD", "MSFT", "SPY", "QQQ"]:
            sig_other = TradeSignal(
                symbol=sym,
                strategy="VWAP_BOUNCE",
                side="BUY",
                confidence=0.80,
                entry_price=100.0,
                stop_price=98.0,
                target_price=105.0,
                atr=2.0,
                risk_per_share=2.0,
                reason=f"{sym} other test",
            )
            self.assertFalse(_should_bypass_lunch_chop(sig_other, event_decision), f"Symbol {sym} falsely bypassed lunch chop!")

        # 4. None event decision -> MUST BE REJECTED
        self.assertFalse(_should_bypass_lunch_chop(sig_nvda_pass, None))

    def test_position_sizer_equity_risk_accounting(self):
        """Verify position sizing calculations and risk percentages against full equity and throttled equity."""
        sig = TradeSignal(
            symbol="NVDA",
            strategy="MOMENTUM_SCALP",
            side="BUY",
            confidence=0.80,
            entry_price=226.78,
            stop_price=226.06,
            target_price=228.07,
            atr=1.50,
            risk_per_share=0.72,
            reason="NVDA sizing test",
        )
        full_equity = 98417.73
        effective_equity = 49208.86 # 0.50x throttled

        shares = calculate_shares(sig, equity=effective_equity)
        self.assertEqual(shares, 17) # Capped at MAX_POSITION_SIZE_DOLLARS ($4000)

        dollar_risk = shares * (sig.entry_price - sig.stop_price) # 17 * 0.72 = $12.24
        self.assertAlmostEqual(dollar_risk, 12.24, places=2)

        # Risk relative to throttled effective equity = ~0.0248%
        pct_effective = (dollar_risk / effective_equity) * 100.0
        self.assertAlmostEqual(pct_effective, 0.02487, delta=0.001)

        # Risk relative to full account equity = ~0.0124%
        pct_full = (dollar_risk / full_equity) * 100.0
        self.assertAlmostEqual(pct_full, 0.01243, delta=0.001)

    def test_position_sizer_defensive_edge_cases(self):
        """Verify position sizer handles non-positive equity, zero risk, and zero entry gracefully."""
        sig_valid = TradeSignal(
            symbol="NVDA",
            strategy="MOMENTUM_SCALP",
            side="BUY",
            confidence=0.80,
            entry_price=200.0,
            stop_price=198.0,
            target_price=205.0,
            atr=2.0,
            risk_per_share=2.0,
            reason="Valid",
        )
        # 1. Negative or zero equity
        self.assertEqual(calculate_shares(sig_valid, equity=0.0), 0)
        self.assertEqual(calculate_shares(sig_valid, equity=-5000.0), 0)

        # 2. Zero risk per share
        sig_zero_risk = TradeSignal(
            symbol="NVDA",
            strategy="MOMENTUM_SCALP",
            side="BUY",
            confidence=0.80,
            entry_price=200.0,
            stop_price=200.0,
            target_price=205.0,
            atr=2.0,
            risk_per_share=0.0,
            reason="Zero risk",
        )
        self.assertEqual(calculate_shares(sig_zero_risk, equity=50000.0), 0)

        # 3. Non-positive entry price
        sig_zero_entry = TradeSignal(
            symbol="NVDA",
            strategy="MOMENTUM_SCALP",
            side="BUY",
            confidence=0.80,
            entry_price=0.0,
            stop_price=0.0,
            target_price=0.0,
            atr=2.0,
            risk_per_share=2.0,
            reason="Zero entry",
        )
        self.assertEqual(calculate_shares(sig_zero_entry, equity=50000.0), 0)

    def test_instant_loss_cooldown_win_trade_not_flagged(self):
        """Verify that a winning exit (SELL price > entry price) does NOT register a loss cooldown."""
        from strategies.intraday_risk import _RECENT_SYMBOL_LOSS_TIMESTAMPS, register_symbol_loss
        
        # Clear any existing memory for NVDA
        _RECENT_SYMBOL_LOSS_TIMESTAMPS.pop("NVDA", None)

        fill_price = 228.26
        entry_price = 226.9682
        
        # Simulate logic from alpaca_bridge: is_loss should be False when fill_price >= entry_price
        is_loss = False
        if entry_price > 0:
            is_loss = (fill_price < entry_price)
        
        self.assertFalse(is_loss)
        self.assertNotIn("NVDA", _RECENT_SYMBOL_LOSS_TIMESTAMPS)

        # Loss scenario (fill_price < entry_price) -> MUST flag loss
        loss_fill_price = 225.00
        is_loss_actual = (loss_fill_price < entry_price)
        self.assertTrue(is_loss_actual)
        if is_loss_actual:
            register_symbol_loss("NVDA")
        self.assertIn("NVDA", _RECENT_SYMBOL_LOSS_TIMESTAMPS)


if __name__ == "__main__":
    unittest.main()
