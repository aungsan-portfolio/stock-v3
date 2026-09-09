"""
tests/test_geometry_hard_guards.py -- Unit tests for Option A noise-floor hard rejection,
maximum chase hard rejection, quote validity vetoes, and post-fill geometry revalidation.
"""
import time
import unittest
from unittest.mock import MagicMock

import config
from strategies.base import TradeSignal, StrategyName
from strategies.order_manager import (
    execute_signal,
    unfavorable_chase,
    validate_geometry,
    compute_effective_noise_floor,
    validate_quote_health,
    on_broker_fill_event,
    _pending_order_contexts,
    _pending_order_contexts_lock,
)


class TestGeometryHardGuards(unittest.TestCase):

    def setUp(self):
        with _pending_order_contexts_lock:
            _pending_order_contexts.clear()

    def tearDown(self):
        with _pending_order_contexts_lock:
            _pending_order_contexts.clear()

    def test_noise_floor_option_a_rejection(self):
        """When stop distance < effective noise floor, signal is deterministically rejected."""
        # Noise floor abs min is 0.08. With atr=0.05, floor is >= 0.08.
        sig = TradeSignal(
            symbol="QQQ",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.85,
            entry_price=400.00,
            stop_price=399.96,  # stop distance = 0.04 < 0.08 noise floor!
            target_price=400.50,
            atr=0.10,
            risk_per_share=0.04,
            reason="Tight stop signal",
        )

        floor = compute_effective_noise_floor(sig)
        self.assertGreaterEqual(floor, 0.08)

        res = execute_signal(
            signal=sig,
            bridge=None,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
        )
        self.assertEqual(res["status"], "REJECTED")
        self.assertIn("REJECTED_NOISE_FLOOR_BREACH", res["reason"])

    def test_maximum_chase_hard_rejection(self):
        """When executable limit moves unfavorably beyond MAX_ALLOWED_CHASE, order is rejected."""
        sig = TradeSignal(
            symbol="NVDA",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.85,
            entry_price=120.00,
            stop_price=118.00,
            target_price=125.00,
            atr=1.5,
            risk_per_share=2.0,
            reason="Chasing test",
        )

        # entry_limit_price = 120.25 -> unfavorable chase of $0.25 > max allowed $0.15
        res = execute_signal(
            signal=sig,
            bridge=None,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            entry_limit_price=120.25,
        )
        self.assertEqual(res["status"], "REJECTED")
        self.assertIn("REJECTED_MAX_CHASE_EXCEEDED", res["reason"])

    def test_quote_health_validation(self):
        """Validates real-time quote freshness, uncrossed spread, and positive prices."""
        now = time.time()

        # Valid quote
        valid_q = {"bid": 100.0, "ask": 100.05, "timestamp": now}
        ok, err, metrics = validate_quote_health(valid_q, now_ts=now, atr=0.50)
        self.assertTrue(ok)
        self.assertEqual(err, "")
        self.assertAlmostEqual(metrics["spread_dollars"], 0.05)
        self.assertAlmostEqual(metrics["spread_as_atr_fraction"], 0.10)

        # Crossed quote
        crossed_q = {"bid": 100.10, "ask": 100.00, "timestamp": now}
        ok, err, _ = validate_quote_health(crossed_q, now_ts=now)
        self.assertFalse(ok)
        self.assertIn("CROSSED_QUOTE", err)

        # Stale quote (> 5.0s)
        stale_q = {"bid": 100.0, "ask": 100.05, "timestamp": now - 10.0}
        ok, err, _ = validate_quote_health(stale_q, now_ts=now, max_age_seconds=5.0)
        self.assertFalse(ok)
        self.assertIn("STALE_QUOTE", err)

        # Non-positive quote
        zero_q = {"bid": 0.0, "ask": 100.05, "timestamp": now}
        ok, err, _ = validate_quote_health(zero_q, now_ts=now)
        self.assertFalse(ok)
        self.assertIn("NON_POSITIVE_QUOTE", err)

    def test_post_fill_geometry_rr_degraded_preserves_stop(self):
        """When fill price slips causing RR < min_rr, mark POST_FILL_RR_DEGRADED and keep stop."""
        sig = TradeSignal(
            symbol="TSLA",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.85,
            entry_price=200.00,
            stop_price=198.00,
            target_price=204.00,
            atr=2.0,
            risk_per_share=2.0,
            reason="Test signal",
        )

        with _pending_order_contexts_lock:
            _pending_order_contexts["p_tsla_fill"] = {
                "signal": sig,
                "limit_price": 200.00,
                "requested_qty": 10.0,
                "cumulative_filled_qty": 0.0,
                "average_fill_price": 0.0,
                "stop_id": "s_tsla",
                "tp_id": "t_tsla",
                "submitted_entry_price": 200.00,
                "submitted_stop_price": 198.00,
                "submitted_target_price": 204.00,
            }

        mock_bridge = MagicMock()
        # Slippage to 202.50: risk = 202.50 - 198 = 4.50, reward = 204.00 - 202.50 = 1.50 -> RR = 0.33 < 1.0
        on_broker_fill_event(
            parent_id="p_tsla_fill",
            fill_price=202.50,
            filled_qty=10.0,
            bridge=mock_bridge,
            cumulative_filled_qty=10.0,
            average_fill_price=202.50,
        )

        # Bridge close_position must NOT have been called because stop ordering is valid (198 < 202.50 < 204)
        mock_bridge.close_position.assert_not_called()

    def test_post_fill_geometry_inverted_triggers_emergency_flatten(self):
        """Catastrophic slippage past the stop price triggers emergency flatten."""
        sig = TradeSignal(
            symbol="TSLA",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.85,
            entry_price=200.00,
            stop_price=198.00,
            target_price=204.00,
            atr=2.0,
            risk_per_share=2.0,
            reason="Test signal",
        )

        with _pending_order_contexts_lock:
            _pending_order_contexts["p_tsla_breach"] = {
                "signal": sig,
                "limit_price": 200.00,
                "requested_qty": 10.0,
                "cumulative_filled_qty": 0.0,
                "average_fill_price": 0.0,
                "stop_id": "s_tsla",
                "tp_id": "t_tsla",
                "submitted_entry_price": 200.00,
                "submitted_stop_price": 198.00,
                "submitted_target_price": 204.00,
            }

        mock_bridge = MagicMock()
        # Catastrophic gap down fill at 197.00 (below 198.00 stop!)
        on_broker_fill_event(
            parent_id="p_tsla_breach",
            fill_price=197.00,
            filled_qty=10.0,
            bridge=mock_bridge,
            cumulative_filled_qty=10.0,
            average_fill_price=197.00,
        )

        mock_bridge.close_position.assert_called_once_with("TSLA")


if __name__ == "__main__":
    unittest.main()
