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
from strategies.trailing_stop import (
    DynamicTrailingStopManager,
    compute_canonical_post_fill_distances,
)



class TestGeometryHardGuards(unittest.TestCase):

    def setUp(self):
        with _pending_order_contexts_lock:
            _pending_order_contexts.clear()
        self._orig_entry_readiness = getattr(config, "DAYTRADE_REQUIRE_ENTRY_READINESS", True)
        config.DAYTRADE_REQUIRE_ENTRY_READINESS = False

    def tearDown(self):
        with _pending_order_contexts_lock:
            _pending_order_contexts.clear()
        config.DAYTRADE_REQUIRE_ENTRY_READINESS = self._orig_entry_readiness

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

    # ─────────────────────────────────────────────────────────────
    # Phase A Correctness Tests: Post-Fill Geometry Contract
    # ─────────────────────────────────────────────────────────────
    def test_spy_geometry_rebuild_does_not_jump_to_50_bps(self):
        """Historical Replay (2026-09-23): Rebuild must stay on canonical 5-bps basis ($769.90) and NOT jump to 50 bps ($766.49)."""
        manager = DynamicTrailingStopManager()
        # SPY fill: $770.34, initial proposed stop $770.03 (dist $0.31 < 5 bps floor $0.38517)
        # Canonical rebuild: min_req = $0.38517, min_rebuild = round(0.38517 * 1.15, 2) = $0.44
        # Rebuilt stop = 770.34 - 0.44 = 769.90
        resulting_stop = manager.validate_and_rebuild_geometry(
            symbol="SPY",
            signal_price=770.34,
            fill_price=770.34,
            initial_stop=770.03,
            target_price=776.12,
            side="BUY",
        )
        self.assertEqual(resulting_stop, 769.90)
        self.assertNotEqual(resulting_stop, 766.49, "Rebuild must NOT jump to 50 bps ($766.49)")
        stop_dist = round(770.34 - resulting_stop, 2)
        self.assertEqual(stop_dist, 0.44)

    def test_post_fill_distance_exactly_at_5_bps(self):
        """Exact 5-bps boundary: When distance == exactly 5 bps (fill=100.00, stop=99.95), valid intact without rebuild."""
        manager = DynamicTrailingStopManager()
        # $100.00 * 0.0005 = $0.05. Stop distance = 100.00 - 99.95 = $0.05.
        resulting_stop = manager.validate_and_rebuild_geometry(
            symbol="TEST",
            signal_price=100.00,
            fill_price=100.00,
            initial_stop=99.95,
            target_price=101.00,
            side="BUY",
        )
        self.assertEqual(resulting_stop, 99.95)

    def test_post_fill_distance_below_5_bps_rebuilds_at_canonical_floor(self):
        """Below 5-bps boundary: When distance < 5 bps (fill=100.00, stop=99.96, dist=0.04), rebuilds to canonical floor."""
        manager = DynamicTrailingStopManager()
        # min_required = $0.05, min_rebuild = round(0.05 * 1.15, 2) = $0.06
        # Rebuilt stop = 100.00 - 0.06 = 99.94
        resulting_stop = manager.validate_and_rebuild_geometry(
            symbol="TEST",
            signal_price=100.00,
            fill_price=100.00,
            initial_stop=99.96,
            target_price=101.00,
            side="BUY",
        )
        self.assertEqual(resulting_stop, 99.94)

    def test_post_fill_low_priced_symbol_absolute_floor(self):
        """Low-priced stocks: Absolute floor $0.05 takes precedence when 5 bps is smaller than $0.05."""
        manager = DynamicTrailingStopManager()
        # fill=10.00 (5 bps = $0.005 < $0.05 floor). min_required = $0.05, min_rebuild = $0.06.
        resulting_stop = manager.validate_and_rebuild_geometry(
            symbol="PENNY",
            signal_price=10.00,
            fill_price=10.00,
            initial_stop=9.98,  # dist = $0.02 < $0.05
            target_price=10.50,
            side="BUY",
        )
        self.assertEqual(resulting_stop, 9.94)

    def test_post_fill_normal_valid_stop_preserved(self):
        """Normal scalp stop: Stop distance >= 5 bps is preserved untouched."""
        manager = DynamicTrailingStopManager()
        resulting_stop = manager.validate_and_rebuild_geometry(
            symbol="QQQ",
            signal_price=400.00,
            fill_price=400.00,
            initial_stop=398.00,  # dist = $2.00 >> $0.20
            target_price=404.00,
            side="BUY",
        )
        self.assertEqual(resulting_stop, 398.00)

    def test_post_fill_repeated_rebuild_idempotent(self):
        """Anti-Widening Invariant: Re-evaluating a rebuilt stop is strictly idempotent with zero progressive widening."""
        manager = DynamicTrailingStopManager()
        # Step 1: Initial rebuild from invalid $770.03
        first_pass_stop = manager.validate_and_rebuild_geometry(
            symbol="SPY",
            signal_price=770.34,
            fill_price=770.34,
            initial_stop=770.03,
            target_price=776.12,
            side="BUY",
        )
        self.assertEqual(first_pass_stop, 769.90)

        # Step 2: Feed the rebuilt stop back into the validator
        second_pass_stop = manager.validate_and_rebuild_geometry(
            symbol="SPY",
            signal_price=770.34,
            fill_price=770.34,
            initial_stop=first_pass_stop,
            target_price=776.12,
            side="BUY",
        )
        self.assertEqual(second_pass_stop, 769.90, "Repeated evaluation must NOT widen stop further")

    def test_post_fill_long_short_symmetry(self):
        """Symmetry Invariant: Short position obeys identical 5-bps canonical floor and rebuild offset above fill."""
        manager = DynamicTrailingStopManager()
        # SHORT: fill=100.00, initial_stop=100.04 (dist=$0.04 < $0.05). min_rebuild = $0.06.
        # Rebuilt stop = 100.00 + 0.06 = 100.06.
        resulting_stop = manager.validate_and_rebuild_geometry(
            symbol="TEST",
            signal_price=100.00,
            fill_price=100.00,
            initial_stop=100.04,
            target_price=98.00,
            side="SELL",
        )
        self.assertEqual(resulting_stop, 100.06)

    def test_time_exit_interaction_contract(self):
        """Policy Isolation: Geometry calculation is a pure function and does not alter time-exit timeouts."""
        from strategies.trailing_stop import MAX_HOLD_MINUTES
        # Time-exit threshold remains untouched at 25.0 minutes
        self.assertEqual(MAX_HOLD_MINUTES, 25.0)
        req, reb = compute_canonical_post_fill_distances(770.34, decimals=2)
        self.assertAlmostEqual(req, 0.38517, places=4)
        self.assertEqual(reb, 0.44)


if __name__ == "__main__":
    unittest.main()

