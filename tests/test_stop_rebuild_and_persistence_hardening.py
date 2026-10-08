"""
tests/test_stop_rebuild_and_persistence_hardening.py

Unit tests verifying critical P0 execution safety and bar-freshness invariants:
1. Zero False Confirmation: When broker stop replacement fails, confirmed_broker_stop_price
   is None, stop_reconciliation_required is True, and the old stop order id is retained.
2. Verified Confirmation: When broker stop replacement succeeds, confirmed_broker_stop_price
   is updated, stop_reconciliation_required is False, and the new order id is recorded.
3. Fail-Closed Persistence: Disk write failures set persistence_healthy = False, flag all
   active positions for reconciliation, and raise TrailingStatePersistenceError if requested.
4. Safe Corrupted/Empty State Load: Empty or corrupted trailing state files do not crash the engine
   and explicitly mark persistence_healthy = False.
5. Bar-Close Freshness Semantics: 5m closed bars are evaluated against bar close time rather than
   bar start time, preventing false staleness rejections for freshly closed bars.
6. Post-Init Integrity: TrailingStopState.__post_init__ preserves confirmed_broker_stop_price = None
   whenever stop_reconciliation_required is True.
"""
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from strategies.base import TradeSignal, StrategyName
from strategies.execution_quote import validate_execution_quote
from strategies.trailing_stop import (
    DynamicTrailingStopManager,
    GeometryRebuildResult,
    TrailingStopState,
    TrailingStatePersistenceError,
)


class TestStopRebuildAndPersistenceHardening(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.state_file = os.path.join(self.temp_dir.name, "trailing_state.json")
        self.patcher = patch("strategies.trailing_stop.STATE_FILE", self.state_file)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.temp_dir.cleanup()

    def test_replace_fails_no_false_confirmation(self):
        """
        When fill causes geometry invalidation (e.g. fill=100.0, initial_stop=100.01 for LONG)
        and broker replace call fails, confirmed_broker_stop_price must remain None,
        stop_reconciliation_required must be True, and order_id must NOT be falsely marked replaced.
        """
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        # Mock broker replace call failing (e.g. 500 error or network disconnect)
        mock_bridge.replace_order_by_id.side_effect = RuntimeError("Broker connection dropped during replace")

        # 1. Test validate_and_rebuild_geometry direct call with return_details=True
        res = manager.validate_and_rebuild_geometry(
            symbol="AAPL",
            signal_price=100.00,
            fill_price=100.00,
            initial_stop=100.01,  # Inverted stop for BUY! Must be rebuilt.
            target_price=100.50,
            bridge=mock_bridge,
            stop_order_id="stop_order_orig_123",
            tp_order_id="tp_order_orig_456",
            side="BUY",
            return_details=True,
        )

        self.assertIsInstance(res, GeometryRebuildResult)
        self.assertTrue(res.rebuild_performed)
        self.assertFalse(res.rebuild_succeeded)
        self.assertIsNone(res.confirmed_broker_stop_price, "Must NOT claim confirmed stop on broker when replace failed")
        self.assertTrue(res.stop_reconciliation_required, "Must require reconciliation when replace failed")
        self.assertEqual(res.stop_order_id, "stop_order_orig_123", "Must retain original order ID on failure")
        self.assertLess(res.stop_price, 100.00, "Calculated safe stop should be below fill")

        # 2. Test initialize_position end-to-end with failed replace
        sig = TradeSignal(
            symbol="AAPL",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.85,
            entry_price=100.00,
            stop_price=100.01,  # Inverted
            target_price=100.50,
            atr=0.10,
            risk_per_share=0.04,
            signal_id="sig_test_001",
            reason="Test inverted stop",
        )

        state = manager.initialize_position(
            signal=sig,
            fill_price=100.00,
            bridge=mock_bridge,
            stop_order_id="stop_order_orig_123",
            tp_order_id="tp_order_orig_456",
            qty=100,
        )

        self.assertIsNone(state.confirmed_broker_stop_price)
        self.assertTrue(state.stop_reconciliation_required)
        self.assertTrue(state.remediation_in_progress)
        self.assertEqual(state.order_id, "stop_order_orig_123")

    def test_replace_succeeds_confirmed(self):
        """
        When broker stop replace succeeds, confirmed_broker_stop_price is updated to the
        new stop price, stop_reconciliation_required is False, and the new order id is stored.
        """
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_replace_res = MagicMock()
        mock_replace_res.id = "new_replaced_stop_789"
        mock_replace_res.stop_price = 199.89
        mock_replace_res.status = "new"
        mock_replace_res.symbol = "MSFT"
        mock_replace_res.side = "sell"
        mock_replace_res.type = "stop"
        mock_replace_res.qty = 50
        mock_bridge.replace_order_by_id.return_value = mock_replace_res

        sig = TradeSignal(
            symbol="MSFT",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.85,
            entry_price=200.00,
            stop_price=200.05,  # Inverted stop for BUY
            target_price=202.00,
            atr=0.20,
            risk_per_share=0.08,
            signal_id="sig_test_002",
            reason="Test rebuild success",
        )

        state = manager.initialize_position(
            signal=sig,
            fill_price=200.00,
            bridge=mock_bridge,
            stop_order_id="stop_orig_111",
            tp_order_id="tp_orig_222",
            qty=50,
        )

        self.assertIsNotNone(state.confirmed_broker_stop_price)
        self.assertEqual(state.confirmed_broker_stop_price, state.stop_price)
        self.assertFalse(state.stop_reconciliation_required)
        self.assertFalse(state.remediation_in_progress)
        self.assertEqual(state.order_id, "new_replaced_stop_789")

    def test_save_fails_no_false_persistent_state(self):
        """
        Atomic save failure must set persistence_healthy = False, mark all active positions
        as stop_reconciliation_required = True, and raise TrailingStatePersistenceError if requested.
        """
        manager = DynamicTrailingStopManager()
        sig = TradeSignal(
            symbol="NVDA",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.85,
            entry_price=100.00,
            stop_price=99.50,
            target_price=101.50,
            atr=0.20,
            risk_per_share=0.50,
            signal_id="sig_test_003",
            reason="Test persistence failure",
        )
        state = manager.initialize_position(
            signal=sig,
            fill_price=100.00,
            bridge=None,
            qty=10,
        )
        self.assertTrue(manager.persistence_healthy)
        self.assertFalse(state.stop_reconciliation_required)

        # Simulate atomic rename/replace failure (e.g. disk write/permission error)
        with patch("os.replace", side_effect=OSError("Disk write I/O error")):
            save_ok = manager._save_state(raise_on_error=False)
            self.assertFalse(save_ok)
            self.assertFalse(manager.persistence_healthy)
            self.assertIn("Disk write I/O error", manager.last_persistence_error)
            self.assertEqual(manager.consecutive_persistence_failures, 1)
            # All active positions must now require reconciliation
            self.assertTrue(manager.states["NVDA"].stop_reconciliation_required)

            # With raise_on_error=True, it must raise TrailingStatePersistenceError
            with self.assertRaises(TrailingStatePersistenceError):
                manager._save_state(raise_on_error=True)

    def test_restart_uncertain_trailing_state_reconciliation(self):
        """
        Restart with corrupted or empty state file must flag persistence_healthy = False
        and record the error without crashing.
        """
        # Create empty file
        with open(self.state_file, "w") as f:
            pass

        manager_empty = DynamicTrailingStopManager()
        self.assertFalse(manager_empty.persistence_healthy)
        self.assertEqual(manager_empty.last_persistence_error, "STATE_FILE_EMPTY")

        # Create corrupted JSON file
        with open(self.state_file, "w") as f:
            f.write("{invalid_json: true,")

        manager_corrupt = DynamicTrailingStopManager()
        self.assertFalse(manager_corrupt.persistence_healthy)
        self.assertIn("LOAD_FAILED", manager_corrupt.last_persistence_error)

    def test_bar_close_freshness_semantics(self):
        """
        Validates bar-close freshness vs bar-start freshness semantics:
        A 5m bar that completed 30 seconds ago has a start age of 330s.
        With bar_duration_seconds=300.0 and max_entry_bar_close_age_seconds=60.0:
        - Bar with start age = 330s (close age = 30s) must PASS.
        - Bar with start age = 380s (close age = 80s) must FAIL with BAR_STALE.
        - Bar with start age = 100s (close age = -200s, i.e. future/unclosed) must FAIL with BAR_TIMESTAMP_IN_FUTURE.
        """
        now = time.time()
        mock_bridge = MagicMock()
        mock_bridge.get_quote.return_value = {
            "symbol": "SPY",
            "bid": 500.00,
            "ask": 500.05,
            "last": 500.02,
            "timestamp": now - 1.0,
            "source": "alpaca",
            "is_realtime": True,
            "feed_identity": "iex",
        }

        # Case 1: Freshly closed 5m bar (30 seconds after close)
        bar_start_330s_ago = now - 330.0
        res_fresh = validate_execution_quote(
            bridge=mock_bridge,
            symbol="SPY",
            side="BUY",
            decision_price=500.00,
            decision_feed="iex",
            bar_timestamp=bar_start_330s_ago,
            max_bar_age_seconds=300.0,
            bar_duration_seconds=300.0,
            max_entry_bar_close_age_seconds=60.0,
        )
        self.assertTrue(res_fresh.valid, f"Freshly closed bar should pass: {res_fresh.reason}")

        # Case 2: Stale closed 5m bar (80 seconds after close > 60s limit)
        bar_start_380s_ago = now - 380.0
        res_stale = validate_execution_quote(
            bridge=mock_bridge,
            symbol="SPY",
            side="BUY",
            decision_price=500.00,
            decision_feed="iex",
            bar_timestamp=bar_start_380s_ago,
            max_bar_age_seconds=300.0,
            bar_duration_seconds=300.0,
            max_entry_bar_close_age_seconds=60.0,
        )
        self.assertFalse(res_stale.valid)
        self.assertIn("BAR_STALE", res_stale.reason)
        self.assertIn("close_age=80.0s > limit=60.0s", res_stale.reason)

        # Case 3: Future bar (timestamp in future)
        bar_in_future = now + 60.0
        res_future = validate_execution_quote(
            bridge=mock_bridge,
            symbol="SPY",
            side="BUY",
            decision_price=500.00,
            decision_feed="iex",
            bar_timestamp=bar_in_future,
            max_bar_age_seconds=300.0,
            bar_duration_seconds=300.0,
            max_entry_bar_close_age_seconds=60.0,
        )
        self.assertFalse(res_future.valid)
        self.assertIn("BAR_TIMESTAMP_IN_FUTURE", res_future.reason)

    def test_trailing_stop_state_post_init_preserves_unconfirmed(self):
        """
        Directly constructing TrailingStopState with stop_reconciliation_required=True
        and confirmed_broker_stop_price=None must NOT overwrite confirmed_broker_stop_price.
        """
        state = TrailingStopState(
            symbol="AMD",
            side="BUY",
            entry_price=150.00,
            peak_price=150.00,
            atr=1.50,
            stop_price=148.00,
            trail_multiple=1.5,
            last_updated=time.time(),
            confirmed_broker_stop_price=None,
            stop_reconciliation_required=True,
        )
        self.assertIsNone(state.confirmed_broker_stop_price)
        self.assertTrue(state.stop_reconciliation_required)

        # But when stop_reconciliation_required is False, confirmed defaults to stop_price
        state_confirmed = TrailingStopState(
            symbol="AMD",
            side="BUY",
            entry_price=150.00,
            peak_price=150.00,
            atr=1.50,
            stop_price=148.00,
            trail_multiple=1.5,
            last_updated=time.time(),
            confirmed_broker_stop_price=None,
            stop_reconciliation_required=False,
        )
        self.assertEqual(state_confirmed.confirmed_broker_stop_price, 148.00)

    def test_forming_candle_rejected_and_non_finite_inputs(self):
        """
        Closed-bar invariant tests:
        1. Candle started 30s ago on a 5m bar (forming) -> strictly REJECTED (BAR_FORMING).
        2. Candle closed 30s ago (start age = 330s) -> PASSED.
        3. Candle closed 61s ago (start age = 361s) -> strictly REJECTED (BAR_STALE).
        4. Non-finite / negative duration or grace limit -> REJECTED.
        """
        now = time.time()
        mock_bridge = MagicMock()
        mock_bridge.get_quote.return_value = {
            "symbol": "QQQ",
            "bid": 400.00,
            "ask": 400.05,
            "last": 400.02,
            "timestamp": now - 1.0,
            "source": "alpaca",
            "is_realtime": True,
            "feed_identity": "iex",
        }

        # 1. Forming candle: started 30s ago on a 5m candle
        res_forming = validate_execution_quote(
            bridge=mock_bridge,
            symbol="QQQ",
            side="BUY",
            decision_price=400.00,
            bar_timestamp=now - 30.0,
            bar_duration_seconds=300.0,
            max_entry_bar_close_age_seconds=60.0,
        )
        self.assertFalse(res_forming.valid)
        self.assertIn("BAR_FORMING", res_forming.reason)

        # 2. Closed 30s ago: started 330s ago on 5m candle
        res_closed_fresh = validate_execution_quote(
            bridge=mock_bridge,
            symbol="QQQ",
            side="BUY",
            decision_price=400.00,
            bar_timestamp=now - 330.0,
            bar_duration_seconds=300.0,
            max_entry_bar_close_age_seconds=60.0,
        )
        self.assertTrue(res_closed_fresh.valid)

        # 3. Closed 61s ago: started 361s ago on 5m candle
        res_closed_stale = validate_execution_quote(
            bridge=mock_bridge,
            symbol="QQQ",
            side="BUY",
            decision_price=400.00,
            bar_timestamp=now - 361.0,
            bar_duration_seconds=300.0,
            max_entry_bar_close_age_seconds=60.0,
        )
        self.assertFalse(res_closed_stale.valid)
        self.assertIn("BAR_STALE", res_closed_stale.reason)

        # 4. Non-finite / negative parameters
        res_nan_dur = validate_execution_quote(
            bridge=mock_bridge,
            symbol="QQQ",
            decision_price=400.00,
            bar_timestamp=now - 330.0,
            bar_duration_seconds=float("nan"),
        )
        self.assertFalse(res_nan_dur.valid)
        self.assertIn("Invalid quote-validation configuration", res_nan_dur.reason)

        res_neg_dur = validate_execution_quote(
            bridge=mock_bridge,
            symbol="QQQ",
            decision_price=400.00,
            bar_timestamp=now - 330.0,
            bar_duration_seconds=-10.0,
        )
        self.assertFalse(res_neg_dur.valid)
        self.assertIn("Invalid quote-validation configuration", res_neg_dur.reason)

    def test_stop_replace_none_response_no_false_confirmation(self):
        """replace_fn returning None must NOT result in confirmed_broker_stop_price."""
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = None  # None return without exception

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,  # Inverted
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_111",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)
        self.assertEqual(res.stop_order_id, "stop_111")

    def test_stop_replace_missing_id_no_false_confirmation(self):
        """replace_fn returning object without id must NOT result in confirmed_broker_stop_price."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace()  # No id or order_id

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_222",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_terminal_status_no_false_confirmation(self):
        """replace_fn returning rejected order must NOT result in confirmed_broker_stop_price."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_rej", status="rejected")

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_333",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_price_mismatch_no_false_confirmation(self):
        """replace_fn returning stop_price that doesn't match new stop must NOT confirm."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_wrong_price", status="new", stop_price=25.00)

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_444",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_snapshot_query_fails_no_false_confirmation(self):
        """When snapshot query fails during confirmation, stop must NOT be marked confirmed."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_snap_fail", status="new")
        mock_bridge.get_order_snapshot.side_effect = RuntimeError("Broker gateway timeout")

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_555",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_persistence_unhealthy_blocks_new_entry_gate(self):
        """When trailing stop manager persistence is unhealthy, execute_signal must veto new entries."""
        from strategies.order_manager import execute_signal
        from strategies.base import EntryReadiness
        from datetime import datetime, timezone
        import strategies.trailing_stop as ts_mod

        now = time.time()
        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.open_position_count.return_value = 0
        mock_bridge.list_positions.return_value = []
        mock_bridge.has_position.return_value = False
        mock_bridge.has_working_order.return_value = False
        mock_bridge.get_quote.return_value = {
            "symbol": "SPY",
            "bid": 500.00,
            "ask": 500.05,
            "last": 500.02,
            "timestamp": now - 1.0,
            "source": "alpaca",
            "is_realtime": True,
            "feed_identity": "iex",
        }

        signal = TradeSignal(
            symbol="SPY",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.85,
            entry_price=500.00,
            stop_price=498.00,
            target_price=504.00,
            atr=1.00,
            risk_per_share=2.00,
            reason="Persistence gate test",
            bar_timestamp=datetime.fromtimestamp(now - 330.0, tz=timezone.utc).isoformat(),
            decision_feed="iex",
        )
        readiness = EntryReadiness(
            entry_allowed=True,
            suppression_reason=None,
            cycle_id="cycle_persist",
            bar_ts=datetime.fromtimestamp(now - 330.0, tz=timezone.utc).isoformat(),
            created_at=now,
            expires_at=now + 300.0,
        )

        orig_healthy = ts_mod.trailing_stop_manager.persistence_healthy
        try:
            ts_mod.trailing_stop_manager.persistence_healthy = False
            ts_mod.trailing_stop_manager.last_persistence_error = "DISK_CORRUPTED"

            with patch("config.SPY_TREND_GUARD_ENABLED", False), \
                 patch("strategies.intraday_risk.get_recent_symbol_loss_time", return_value=None):
                res = execute_signal(
                    signal=signal,
                    bridge=mock_bridge,
                    equity=100000.0,
                    current_pnl=0.0,
                    dry_run=False,
                    entry_readiness=readiness,
                )
                self.assertEqual(res.get("status"), "VETOED")
                self.assertIn("Trailing Stop Persistence Unhealthy", res.get("reason", ""))
        finally:
            ts_mod.trailing_stop_manager.persistence_healthy = orig_healthy

    def test_strict_completed_5m_bar_boundaries(self):
        """
        Tests strict completed-bar enforcement across all boundary values:
        -2.001 -> reject (BAR_FORMING)
        -2.0   -> reject (BAR_FORMING)
        -1.0   -> reject (BAR_FORMING)
        -0.001 -> reject (BAR_FORMING)
        0.0    -> pass (exact close)
        60.0   -> pass (freshness boundary)
        60.001 -> reject (BAR_STALE)
        """
        now = 1700000000.0
        mock_bridge = MagicMock()
        mock_bridge.get_quote.return_value = {
            "symbol": "SPY",
            "bid": 500.00,
            "ask": 500.05,
            "last": 500.02,
            "timestamp": now - 0.5,
            "source": "alpaca",
            "is_realtime": True,
            "feed_identity": "iex",
        }

        duration = 300.0
        limit = 60.0

        boundary_cases = [
            (-2.001, False, "BAR_FORMING"),
            (-2.0, False, "BAR_FORMING"),
            (-1.0, False, "BAR_FORMING"),
            (-0.001, False, "BAR_FORMING"),
            (0.0, True, None),
            (60.0, True, None),
            (60.001, False, "BAR_STALE"),
        ]

        with patch("time.time", return_value=now):
            for close_age, expected_valid, expected_reason_substring in boundary_cases:
                bar_ts = now - close_age - duration
                res = validate_execution_quote(
                    bridge=mock_bridge,
                    symbol="SPY",
                    side="BUY",
                    decision_price=500.00,
                    bar_timestamp=bar_ts,
                    bar_duration_seconds=duration,
                    max_entry_bar_close_age_seconds=limit,
                )
                self.assertEqual(
                    res.valid,
                    expected_valid,
                    f"Failed for close_age={close_age}: expected valid={expected_valid}, got valid={res.valid} (reason: {res.reason})",
                )
                if expected_reason_substring:
                    self.assertIn(
                        expected_reason_substring,
                        res.reason,
                        f"Expected substring {expected_reason_substring!r} in {res.reason!r} for close_age={close_age}",
                    )

    def test_persistence_fail_closed_all_failure_modes(self):
        """
        Validates that ALL trailing stop persistence failure modes strictly veto new entries
        with ZERO broker submit calls:
        1. manager missing (get_trailing_manager returns None)
        2. persistence_healthy attribute missing
        3. persistence_healthy read exception
        4. persistence_healthy is False
        5. persistence_healthy is None (unknown state)
        """
        from strategies.order_manager import execute_signal
        from strategies.base import EntryReadiness
        from datetime import datetime, timezone
        import types

        now = time.time()
        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.open_position_count.return_value = 0
        mock_bridge.list_positions.return_value = []
        mock_bridge.has_position.return_value = False
        mock_bridge.has_working_order.return_value = False
        mock_bridge.get_quote.return_value = {
            "symbol": "SPY",
            "bid": 500.00,
            "ask": 500.05,
            "last": 500.02,
            "timestamp": now - 1.0,
            "source": "alpaca",
            "is_realtime": True,
            "feed_identity": "iex",
        }

        signal = TradeSignal(
            symbol="SPY",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.85,
            entry_price=500.00,
            stop_price=498.00,
            target_price=504.00,
            atr=1.00,
            risk_per_share=2.00,
            reason="Persistence fail-closed test",
            bar_timestamp=datetime.fromtimestamp(now - 330.0, tz=timezone.utc).isoformat(),
            decision_feed="iex",
        )
        readiness = EntryReadiness(
            entry_allowed=True,
            suppression_reason=None,
            cycle_id="cycle_persist_modes",
            bar_ts=datetime.fromtimestamp(now - 330.0, tz=timezone.utc).isoformat(),
            created_at=now,
            expires_at=now + 300.0,
        )

        test_cases = [
            ("manager_missing", lambda: None, "Manager instance missing"),
            ("attr_missing", lambda: types.SimpleNamespace(), "persistence_healthy attribute missing"),
            (
                "read_exception",
                lambda: type("BrokenManager", (), {"persistence_healthy": property(lambda self: (_ for _ in ()).throw(RuntimeError("disk read failure")))})(),
                "Trailing Stop Persistence Gate Exception",
            ),
            (
                "persistence_unhealthy",
                lambda: types.SimpleNamespace(persistence_healthy=False, last_persistence_error="DISK_FULL"),
                "Trailing Stop Persistence Unhealthy: DISK_FULL",
            ),
            (
                "persistence_unknown",
                lambda: types.SimpleNamespace(persistence_healthy=None, last_persistence_error="STATE_UNKNOWN"),
                "Trailing Stop Persistence Unhealthy",
            ),
        ]

        for case_name, mgr_factory, expected_reason_substr in test_cases:
            mgr_obj = mgr_factory()
            with patch("strategies.trailing_stop.get_trailing_manager", return_value=mgr_obj), \
                 patch("strategies.order_manager._place_bracket_order_once") as mock_place, \
                 patch("config.SPY_TREND_GUARD_ENABLED", False), \
                 patch("strategies.intraday_risk.get_recent_symbol_loss_time", return_value=None):

                res = execute_signal(
                    signal=signal,
                    bridge=mock_bridge,
                    equity=100000.0,
                    current_pnl=0.0,
                    dry_run=False,
                    entry_readiness=readiness,
                )

                self.assertEqual(res.get("status"), "VETOED", f"Case {case_name} should be VETOED")
                self.assertIn(expected_reason_substr, res.get("reason", ""), f"Case {case_name} reason mismatch")
                mock_place.assert_not_called()

    def test_stop_replace_enum_rejected_status(self):
        """SDK Enum status (OrderStatus.REJECTED) must be normalized and rejected without confirmation."""
        from enum import Enum
        class MockOrderStatus(Enum):
            REJECTED = "rejected"

        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(
            id="r_enum_rej",
            status=MockOrderStatus.REJECTED,
            symbol="INTC",
            side="sell",
            type="stop",
            stop_price=29.50,
        )

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_enum_1",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_snapshot_wrong_symbol(self):
        """Snapshot query returning order with wrong symbol must NOT confirm."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_snap_sym")
        mock_bridge.get_order_snapshot.return_value = {
            "id": "r_snap_sym",
            "symbol": "MSFT",
            "side": "sell",
            "type": "stop",
            "status": "new",
            "stop_price": 29.89,
        }

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_snap_sym",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_snapshot_wrong_side(self):
        """Snapshot query returning order with wrong side (buy instead of sell for BUY position) must NOT confirm."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_snap_side")
        mock_bridge.get_order_snapshot.return_value = {
            "id": "r_snap_side",
            "symbol": "INTC",
            "side": "buy",
            "type": "stop",
            "status": "new",
            "stop_price": 29.89,
        }

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_snap_side",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_snapshot_wrong_order_type(self):
        """Snapshot query returning non-stop order type (e.g. limit) must NOT confirm."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_snap_type")
        mock_bridge.get_order_snapshot.return_value = {
            "id": "r_snap_type",
            "symbol": "INTC",
            "side": "sell",
            "type": "limit",
            "status": "new",
            "stop_price": 29.89,
        }

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_snap_type",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_snapshot_terminal_status(self):
        """Snapshot query returning terminal/canceled order must NOT confirm."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_snap_term")
        mock_bridge.get_order_snapshot.return_value = {
            "id": "r_snap_term",
            "symbol": "INTC",
            "side": "sell",
            "type": "stop",
            "status": "canceled",
            "stop_price": 29.89,
        }

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_snap_term",
            side="BUY",
            return_details=True,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_snapshot_wrong_qty(self):
        """Snapshot query returning quantity mismatch must NOT confirm."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_snap_qty")
        mock_bridge.get_order_snapshot.return_value = {
            "id": "r_snap_qty",
            "symbol": "INTC",
            "side": "sell",
            "type": "stop",
            "status": "new",
            "qty": 50,
            "stop_price": 29.89,
        }

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_snap_qty",
            side="BUY",
            return_details=True,
            expected_qty=100,
        )
        self.assertIsNone(res.confirmed_broker_stop_price)
        self.assertTrue(res.stop_reconciliation_required)
        self.assertFalse(res.rebuild_succeeded)

    def test_stop_replace_snapshot_fully_verified_succeeds(self):
        """Snapshot query matching all required broker evidence fields confirms stop price."""
        import types
        manager = DynamicTrailingStopManager()
        mock_bridge = MagicMock()
        mock_bridge.replace_order_by_id.return_value = types.SimpleNamespace(id="r_snap_ok")
        mock_bridge.get_order_snapshot.return_value = {
            "id": "r_snap_ok",
            "symbol": "INTC",
            "side": "sell",
            "type": "stop",
            "status": "new",
            "qty": 100,
            "stop_price": 29.94,
        }

        res = manager.validate_and_rebuild_geometry(
            symbol="INTC",
            signal_price=30.00,
            fill_price=30.00,
            initial_stop=30.01,
            target_price=30.50,
            bridge=mock_bridge,
            stop_order_id="stop_snap_ok",
            side="BUY",
            return_details=True,
            expected_qty=100,
        )
        self.assertEqual(res.confirmed_broker_stop_price, 29.94)
        self.assertFalse(res.stop_reconciliation_required)
        self.assertTrue(res.rebuild_succeeded)
        self.assertEqual(res.stop_order_id, "r_snap_ok")


if __name__ == "__main__":
    unittest.main()


