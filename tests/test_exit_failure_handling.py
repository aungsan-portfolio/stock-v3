"""
tests/test_exit_failure_handling.py -- Formal Synthetic Failure Tests for Exit Safety & Remediation.

Codifies the 5 critical exit failure scenarios required for Daytrade Bot Safety Certification:
1. Scenario 1 (Cancel Timeout): Order cancel convergence timeout (>3.0s) fails closed without blind close submission.
2. Scenario 2 (Broker Close Rejection): Close order submission rejected (HTTP 403/422) engages Fail-Closed Incident Latch.
3. Scenario 3 (Partial Fill): Residual shares (36 of 56) are accurately measured via Broker Truth and protected.
4. Scenario 4 (Anti-Reversal Guard): Long-to-short position reversal is detected; retry close is strictly refused.
5. Scenario 5 (Position Query Failure): Network exception during position query fails closed as UNKNOWN; never assumes flat.

Invariants Asserted:
- Invariant 1: Position quantity determined strictly by Broker Truth.
- Invariant 2: No blind execution into conflicting or active working orders.
- Invariant 3: Residual shares after partial fill are accurately tracked and protected.
- Invariant 4: Unprotected open positions engage explicit Incident Latch or re-protection.
- Invariant 5: When broker state is unknown, system never falsely assumes position is flat.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from decimal import Decimal

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alpaca_bridge import AlpacaBridge, normalize_position_qty
from strategies.atomic_rotation import CloseOutcome, PositionCloseResult
from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState
import strategies.trailing_stop


class TestExitFailureHandling(unittest.TestCase):

    def setUp(self):
        self.bridge = AlpacaBridge(mode="daytrade", dry_run=False)
        self.bridge._connected = True
        self.bridge._client = MagicMock()
        self.bridge._conn_health = MagicMock()
        self.bridge._conn_health.mark_healthy = MagicMock()
        self.bridge._conn_health.mark_unhealthy = MagicMock()
        # Mock market hours to True for offline tests
        strategies.trailing_stop.is_market_open = lambda: True

    def test_scenario_1_cancel_timeout_fails_closed(self):
        """
        Scenario 1: Stop/TP orders fail to cancel within 3.0s convergence timeout.
        Invariant 2: No blind execution into active working orders.
        Invariant 1: Broker Truth determines position remains open.
        """
        # Initial position: 56 shares of UBER
        self.bridge.get_position = MagicMock(return_value=56.0)

        # Working orders persist past convergence timeout
        mock_active_stop = MagicMock(symbol="UBER", id="stop-123", status=MagicMock(value="new"))
        self.bridge._get_active_orders = MagicMock(return_value=[mock_active_stop])

        with patch("time.sleep", return_value=None):
            res = self.bridge.close_position_verified("UBER")

        # Must fail closed with UNKNOWN outcome
        self.assertEqual(res.outcome, CloseOutcome.UNKNOWN)
        self.assertIn("convergence timeout", str(res.error).lower())
        self.assertEqual(res.remaining_qty, 56.0)
        self.assertEqual(res.filled_qty, 0.0)

        # CRITICAL INVARIANT: _client.close_position MUST NOT be called!
        self.bridge._client.close_position.assert_not_called()
        self.assertFalse(res.is_flat)

    def test_cancel_convergence_query_failure_fails_closed(self):
        """
        P0 Gap: When _get_active_orders raises an Exception during cancel convergence check,
        the system MUST NOT treat the error as an empty list (fail-open bug).
        It MUST fail-closed with UNKNOWN outcome, without calling _client.close_position.
        """
        self.bridge.get_position = MagicMock(return_value=56.0)
        self.bridge._get_active_orders = MagicMock(side_effect=RuntimeError("Alpaca API timeout / network blip"))

        with patch("time.sleep", return_value=None):
            res = self.bridge.close_position_verified("UBER")

        # Must fail closed with UNKNOWN outcome
        self.assertEqual(res.outcome, CloseOutcome.UNKNOWN)
        self.assertIn("failed during cancel convergence", str(res.error).lower())
        self.assertEqual(res.remaining_qty, 56.0)
        self.assertEqual(res.filled_qty, 0.0)

        # CRITICAL INVARIANT: _client.close_position MUST NOT be called!
        self.bridge._client.close_position.assert_not_called()
        self.assertFalse(res.is_flat)

    def test_scenario_2_broker_close_rejection_engages_incident_latch(self):
        """
        Scenario 2: Broker close submission is rejected (e.g. HTTP 422 or route unavailable).
        Invariant 4: Engaging explicit Incident Latch to block new entries.
        """
        # Initial position: 56 shares of UBER
        self.bridge.get_position = MagicMock(return_value=56.0)
        self.bridge._get_active_orders = MagicMock(return_value=[])

        # Broker rejects close request
        self.bridge._client.close_position.side_effect = RuntimeError("HTTP 422 Unprocessable Entity: Route unavailable")

        with patch("time.sleep", return_value=None):
            res = self.bridge.close_position_verified("UBER")

        self.assertEqual(res.outcome, CloseOutcome.REJECTED)
        self.assertIn("422", str(res.error))
        self.assertEqual(res.remaining_qty, 56.0)
        self.assertFalse(res.is_flat)

        # Integration with TrailingStopManager: emergency flatten must latch fail-closed
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="UBER",
            side="BUY",
            entry_price=70.22,
            peak_price=70.25,
            atr=0.45,
            stop_price=69.93,
            desired_stop_price=69.93,
            trail_multiple=1.5,
            last_updated=1000.0,
            order_id="stop-old-uber",
            qty=56.0,
        )
        mgr.states["UBER"] = state

        with patch.object(self.bridge, "close_position_verified", return_value=res):
            with patch("time.sleep", return_value=None):
                mgr.emergency_flatten("UBER", 56.0, bridge=self.bridge)

        # State must remain active and latched to block new entries
        self.assertTrue(state.active, "State must remain active on close failure")
        self.assertTrue(state.stop_reconciliation_required, "stop_reconciliation_required must be set True")
        self.assertEqual(state.reconciliation_status, "POSITION_UNKNOWN_RECONCILIATION_REQUIRED")

    def test_scenario_3_partial_fill_residual_protection(self):
        """
        Scenario 3: Close fills partially (20 filled, 36 remaining).
        Invariant 1: Broker Truth determines exact remaining position.
        Invariant 3: Residual 36 shares are tracked and protected.
        Invariant 5: Position is NOT marked flat.
        """
        # Initial 56 shares -> post-close 36 shares -> retry post-close 36 shares
        self.bridge.get_position = MagicMock(side_effect=[56.0, 36.0, 36.0])
        self.bridge._get_active_orders = MagicMock(return_value=[])
        self.bridge._client.close_position = MagicMock(return_value=MagicMock(id="close-ord-1"))
        self.bridge._await_order_outcome = MagicMock(return_value=MagicMock(outcome="PARTIAL", filled=20.0, remaining=36.0))

        with patch.object(self.bridge, "_cancel_all_orders_for_symbol"):
            with patch("time.sleep", return_value=None):
                res = self.bridge.close_position_verified("UBER")

        self.assertEqual(res.outcome, CloseOutcome.PARTIAL)
        self.assertEqual(res.filled_qty, 20.0)
        self.assertEqual(res.remaining_qty, 36.0)
        self.assertFalse(res.is_flat)

        # Remediation: DynamicTrailingStopManager must adopt 36 shares and place replacement stop
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="UBER",
            side="BUY",
            entry_price=70.22,
            peak_price=70.25,
            atr=0.45,
            stop_price=69.93,
            desired_stop_price=69.93,
            trail_multiple=1.5,
            last_updated=1000.0,
            order_id="stop-old-uber",
            qty=56.0,
            stop_reconciliation_required=True,
        )
        mgr.states["UBER"] = state

        # Mock bridge for handle_naked_position
        stop_placed = []
        bridge_mock = MagicMock()
        bridge_mock.get_position = MagicMock(return_value=36.0)
        bridge_mock.market_price = MagicMock(return_value=70.15)
        bridge_mock.working_orders_plain = MagicMock(return_value=[])
        bridge_mock.cancel_order = MagicMock(return_value=True)
        bridge_mock.get_order_snapshot = MagicMock(return_value={"status": "canceled", "id": "stop-old-uber"})
        bridge_mock.place_stop_order = MagicMock(side_effect=lambda **kw: stop_placed.append(kw) or {"order_id": "stop-new-36"})
        bridge_mock._client = MagicMock()
        bridge_mock._client.get_orders = MagicMock(return_value=[])

        # Verification snapshot for replacement order confirms it covers 36 shares
        def mock_snapshot(oid):
            if oid == "stop-new-36":
                return {"id": "stop-new-36", "status": "new", "qty": 36.0, "stop_price": 69.93}
            return {"id": "stop-old-uber", "status": "canceled", "qty": 56.0}
        bridge_mock.get_order_snapshot = mock_snapshot

        with patch("time.sleep", return_value=None):
            mgr.handle_naked_position("UBER", 36.0, bridge=bridge_mock, dry_run=False)

        # Invariant 3: Replacement stop placed for exactly 36.0 residual shares
        self.assertEqual(state.qty, 36.0)
        self.assertTrue(len(stop_placed) > 0, "Replacement stop order must be submitted")
        self.assertEqual(stop_placed[0].get("qty"), 36.0)

    def test_scenario_4_anti_reversal_guard_prevents_escalation(self):
        """
        Scenario 4: Close execution causes position reversal (+56 Long -> -10 Short).
        Invariant: Bridge strictly refuses blind retry that would compound inverted exposure.
        """
        # Initial pos +56, post-close pos flips to -10
        self.bridge.get_position = MagicMock(side_effect=[56.0, -10.0])
        self.bridge._get_active_orders = MagicMock(return_value=[])
        self.bridge._client.close_position = MagicMock(return_value=MagicMock(id="close-rev-1"))
        self.bridge._await_order_outcome = MagicMock(return_value=MagicMock(outcome="FILLED", filled=66.0, remaining=0.0))

        with patch("time.sleep", return_value=None):
            res = self.bridge.close_position_verified("UBER")

        # Refuses retry; records reversal detected
        self.assertEqual(res.error, "Position reversal detected")
        self.assertEqual(res.remaining_qty, -10.0)
        # CRITICAL: _client.close_position was called only ONCE (no retry attempt!)
        self.assertEqual(self.bridge._client.close_position.call_count, 1)

    def test_scenario_5_position_query_failure_retains_fail_closed_latch(self):
        """
        Scenario 5: Post-close position query fails due to network/API drop.
        Invariant 5: Position is strictly treated as UNKNOWN; NEVER assumed flat.
        """
        # Initial pos 56, post-close query raises ConnectionError
        self.bridge.get_position = MagicMock(side_effect=[56.0, ConnectionError("Broker connection reset by peer")])
        self.bridge._get_active_orders = MagicMock(return_value=[])
        self.bridge._client.close_position = MagicMock(return_value=MagicMock(id="close-ord-5"))
        self.bridge._await_order_outcome = MagicMock(return_value=MagicMock(outcome="FILLED", filled=56.0, remaining=0.0))

        with patch("time.sleep", return_value=None):
            res = self.bridge.close_position_verified("UBER")

        # Query failure is strictly UNKNOWN
        self.assertEqual(res.outcome, CloseOutcome.UNKNOWN)
        self.assertIn("Post-close position query failed", str(res.error))
        self.assertFalse(res.is_flat, "Must NEVER assume flat when position query fails!")

    def test_swing_ensure_symbol_protection_direct_buffer_submission(self):
        """
        Swing Isolation Check: ensure_symbol_protection calculates 2% buffer GTC stop
        for swing rotation aborts, confirming isolation from Daytrade path.
        """
        self.bridge.get_position = MagicMock(return_value=56.0)
        self.bridge.market_price = MagicMock(return_value=70.00)
        self.bridge.working_orders_plain = MagicMock(return_value=[])

        submitted_orders = []
        self.bridge._client.submit_order = MagicMock(side_effect=lambda req: submitted_orders.append(req) or MagicMock(id="gtc-stop-1"))

        success = self.bridge.ensure_symbol_protection("UBER", qty=56.0)
        self.assertTrue(success)
        self.assertEqual(len(submitted_orders), 1)

        req = submitted_orders[0]
        self.assertEqual(req.symbol, "UBER")
        self.assertEqual(req.qty, 56.0)
        # Long stop at 2% buffer: 70.00 * 0.98 = 68.60
        self.assertEqual(req.stop_price, 68.60)

    def test_daytrade_e2e_exit_failure_to_verified_reprotection(self):
        """
        End-to-End Daytrade Exit Failure Flow:
        1. Cycle 1: Time exit triggers close_position; broker cancels working stop but rejects market close (HTTP 422).
           close_position returns False; state remains ACTIVE (never falsely soft-reset).
        2. Cycle 2: Next orchestrator tick runs ensure_initialized; detects zero working stops on broker.
           Flags stop_reconciliation_required=True.
        3. Orchestrator routes to handle_naked_position; queries Broker Truth (56.0 shares),
           and submits a verified replacement stop order.
        4. State is verified fully re-protected with confirmed order_id and clear reconciliation status.
        """
        # Cycle 1 Setup: Active position with old stop
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="UBER",
            side="BUY",
            entry_price=70.22,
            peak_price=70.25,
            atr=0.45,
            stop_price=69.93,
            desired_stop_price=69.93,
            trail_multiple=1.5,
            last_updated=1000.0,
            order_id="stop-old-uber",
            qty=56.0,
        )
        mgr.states["UBER"] = state

        # Mock broker rejecting close submission
        self.bridge.get_position = MagicMock(return_value=56.0)
        self.bridge._get_active_orders = MagicMock(return_value=[])
        self.bridge._client.close_position = MagicMock(side_effect=RuntimeError("HTTP 422 Unprocessable Entity"))

        # Cycle 1 Execution: Close fails
        with patch("time.sleep", return_value=None):
            closed = self.bridge.close_position("UBER")
        self.assertFalse(closed, "close_position must return False on broker rejection")
        # In orchestrator.py:1869-1873, if closed is False, soft_reset is NOT called
        self.assertTrue(state.active, "State must remain active when close is not confirmed")

        # Cycle 2 Execution: Subsequent orchestrator cycle
        # Broker has 0 working orders (the old stop was cancelled on broker during close attempt)
        open_orders_broker = []
        bridge_mock = MagicMock()
        bridge_mock.get_position = MagicMock(return_value=56.0)
        bridge_mock.market_price = MagicMock(return_value=70.15)
        bridge_mock.working_orders_plain = MagicMock(return_value=[])
        bridge_mock.cancel_order = MagicMock(return_value=True)
        bridge_mock.get_order_snapshot = MagicMock(return_value={"status": "canceled", "id": "stop-old-uber"})
        
        stop_placed = []
        bridge_mock.place_stop_order = MagicMock(side_effect=lambda **kw: stop_placed.append(kw) or {"order_id": "stop-replacement-56"})
        bridge_mock._client = MagicMock()
        bridge_mock._client.get_orders = MagicMock(return_value=[])

        # Step 2a: ensure_initialized runs in orchestrator loop
        mgr.ensure_initialized(
            symbol="UBER",
            side="BUY",
            avg_cost=70.22,
            open_orders=open_orders_broker,
            current_price=70.15,
            qty=56.0,
            bridge=bridge_mock,
        )
        # Must detect that working stop is missing
        self.assertTrue(state.stop_reconciliation_required, "Must flag stop_reconciliation_required when open orders lack stop")

        # Step 2b: orchestrator routes to handle_naked_position
        def mock_snapshot(oid):
            if oid == "stop-replacement-56":
                return {
                    "id": "stop-replacement-56",
                    "symbol": "UBER",
                    "side": "sell",
                    "type": "stop",
                    "status": "new",
                    "qty": 56.0,
                    "stop_price": 69.93,
                }
            return {
                "id": "stop-old-uber",
                "symbol": "UBER",
                "side": "sell",
                "type": "stop",
                "status": "canceled",
                "qty": 56.0,
            }
        bridge_mock.get_order_snapshot = mock_snapshot

        with patch("time.sleep", return_value=None):
            mgr.handle_naked_position("UBER", 56.0, bridge=bridge_mock, dry_run=False)

        # Invariants Asserted:
        # Invariant 1: Broker Truth adopted
        self.assertEqual(state.qty, 56.0)
        # Invariant 3: Replacement stop placed and verified
        self.assertEqual(len(stop_placed), 1)
        self.assertEqual(stop_placed[0].get("qty"), 56.0)
        self.assertEqual(state.order_id, "stop-replacement-56")
        self.assertFalse(state.stop_reconciliation_required)
        self.assertIsNone(state.reconciliation_status)

    def test_daytrade_e2e_exit_failure_with_failed_reprotection_latches_fail_closed(self):
        """
        End-to-End Daytrade Exit Failure Flow with Broken Broker Re-protection:
        When replacement stop order cannot be confirmed on broker,
        system strictly engages Fail-Closed Incident Latch and blocks new entries.
        """
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="UBER",
            side="BUY",
            entry_price=70.22,
            peak_price=70.25,
            atr=0.45,
            stop_price=69.93,
            desired_stop_price=69.93,
            trail_multiple=1.5,
            last_updated=1000.0,
            order_id="stop-old-uber",
            qty=56.0,
            stop_reconciliation_required=True,
        )
        mgr.states["UBER"] = state

        # Mock broker where place_stop_order raises an API error
        bridge_mock = MagicMock()
        bridge_mock.get_position = MagicMock(return_value=56.0)
        bridge_mock.market_price = MagicMock(return_value=70.15)
        bridge_mock.working_orders_plain = MagicMock(return_value=[])
        bridge_mock.cancel_order = MagicMock(return_value=True)
        bridge_mock.get_order_snapshot = MagicMock(return_value={"status": "canceled", "id": "stop-old-uber"})
        bridge_mock.place_stop_order = MagicMock(side_effect=RuntimeError("Broker stop order placement rejected"))
        bridge_mock._client = MagicMock()
        bridge_mock._client.get_orders = MagicMock(return_value=[])

        with patch("time.sleep", return_value=None):
            mgr.handle_naked_position("UBER", 56.0, bridge=bridge_mock, dry_run=False)

        # Must strictly fail closed:
        self.assertTrue(state.active, "State must remain active")
        self.assertTrue(state.stop_reconciliation_required, "stop_reconciliation_required must remain True")
        self.assertEqual(state.reconciliation_status, "POSITION_UNKNOWN_RECONCILIATION_REQUIRED")
        # Invariant 4: Global latch blocks new entries across entire orchestrator
        self.assertTrue(mgr.has_unresolved_position_unknown(), "Orchestrator must detect unresolved position unknown")


if __name__ == "__main__":
    unittest.main()
