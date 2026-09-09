"""
tests/test_lifecycle_halt_integration.py -- Unit tests for C.1 EntryHaltManager and Lifecycle Halt Integration.
"""
import threading
import unittest
from unittest.mock import MagicMock

from strategies.halt_manager import (
    EntryHaltManager,
    LATCHING_HALT_PROTECTION_FAILURE,
    LATCHING_HALT_DAILY_LOSS_BREACH,
    LATCHING_HALT_ORDER_TRACKING_FAILURE,
    TRANSIENT_HALT_DATA_DEGRADED,
    TRANSIENT_HALT_GLOBAL_OPENING_BUFFER,
    TRANSIENT_HALT_FILE_LOCK,
    halt_manager,
)
from strategies.base import TradeSignal, StrategyName
from strategies.entry_lifecycle import EntryOrderLifecycle
from strategies.regime_engine import MarketRegimeResult, MODE_RISK_ON, MODE_DATA_DEGRADED, RegimeStateTracker


import pathlib
import tempfile
import shutil
import config


class TestLifecycleHaltIntegration(unittest.TestCase):

    def setUp(self):
        halt_manager.reset_all()
        self._tmp_dir = tempfile.mkdtemp()
        self._orig_journal = config.DAYTRADE_TRADE_JOURNAL_FILE
        config.DAYTRADE_TRADE_JOURNAL_FILE = pathlib.Path(self._tmp_dir) / "test_journal.jsonl"

    def tearDown(self):
        halt_manager.reset_all()
        config.DAYTRADE_TRADE_JOURNAL_FILE = self._orig_journal
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    def test_entry_halt_manager_basic_transient_lifecycle(self):
        """Verify transient halts can be added and cleared normally."""
        mgr = EntryHaltManager()
        self.assertTrue(mgr.entries_allowed())

        mgr.add(TRANSIENT_HALT_DATA_DEGRADED)
        self.assertFalse(mgr.entries_allowed())
        self.assertIn(TRANSIENT_HALT_DATA_DEGRADED, mgr.active_halts())

        ok = mgr.clear_transient(TRANSIENT_HALT_DATA_DEGRADED)
        self.assertTrue(ok)
        self.assertTrue(mgr.entries_allowed())
        self.assertEqual(len(mgr.active_halts()), 0)

    def test_clear_transient_refuses_to_clear_latched_halts(self):
        """Negative guard: clear_transient must NEVER clear latched halts."""
        mgr = EntryHaltManager()
        mgr.add(LATCHING_HALT_PROTECTION_FAILURE)
        self.assertFalse(mgr.entries_allowed())

        # Attempt to clear via transient path -> must fail
        ok = mgr.clear_transient(LATCHING_HALT_PROTECTION_FAILURE)
        self.assertFalse(ok)
        self.assertFalse(mgr.entries_allowed())
        self.assertIn(LATCHING_HALT_PROTECTION_FAILURE, mgr.active_halts())

        # Also for daily loss breach
        mgr.add(LATCHING_HALT_DAILY_LOSS_BREACH)
        ok2 = mgr.clear_transient(LATCHING_HALT_DAILY_LOSS_BREACH)
        self.assertFalse(ok2)
        self.assertFalse(mgr.entries_allowed())

    def test_unknown_halt_reason_treated_as_latching_fail_closed(self):
        """Fail-closed guard: Any unknown custom halt string cannot be auto-cleared via clear_transient."""
        mgr = EntryHaltManager()
        mgr.add("CUSTOM_UNEXPECTED_HALT_REASON")
        self.assertFalse(mgr.entries_allowed())

        ok = mgr.clear_transient("CUSTOM_UNEXPECTED_HALT_REASON")
        self.assertFalse(ok)
        self.assertFalse(mgr.entries_allowed())

    def test_overlapping_halts_independent_clear_sequence(self):
        """Multiple overlapping halts require ALL halts to clear before entries are allowed."""
        mgr = EntryHaltManager()
        mgr.add(TRANSIENT_HALT_DATA_DEGRADED)
        mgr.add(TRANSIENT_HALT_GLOBAL_OPENING_BUFFER)
        mgr.add(TRANSIENT_HALT_FILE_LOCK)

        self.assertFalse(mgr.entries_allowed())
        self.assertEqual(len(mgr.active_halts()), 3)

        # Clear #1 -> still blocked
        mgr.clear_transient(TRANSIENT_HALT_DATA_DEGRADED)
        self.assertFalse(mgr.entries_allowed())
        self.assertEqual(len(mgr.active_halts()), 2)

        # Clear #2 -> still blocked
        mgr.clear_transient(TRANSIENT_HALT_GLOBAL_OPENING_BUFFER)
        self.assertFalse(mgr.entries_allowed())
        self.assertEqual(len(mgr.active_halts()), 1)

        # Clear #3 -> now allowed
        mgr.clear_transient(TRANSIENT_HALT_FILE_LOCK)
        self.assertTrue(mgr.entries_allowed())
        self.assertEqual(len(mgr.active_halts()), 0)

    def test_precedence_latched_halt_survives_regime_recovery(self):
        """CRITICAL PRECEDENCE: A latched protection failure must keep entries blocked even when regime recovers to RISK_ON."""
        mgr = halt_manager
        mgr.reset_all()

        # 1. Bot experiences transient DATA_DEGRADED
        mgr.add(TRANSIENT_HALT_DATA_DEGRADED)
        self.assertFalse(mgr.entries_allowed())

        # 2. Meanwhile, a background bracket fill experiences protection failure and latches
        mgr.add(LATCHING_HALT_PROTECTION_FAILURE)
        self.assertFalse(mgr.entries_allowed())

        # 3. Regime recovers: DATA_DEGRADED transient halt is cleared
        mgr.clear_transient(TRANSIENT_HALT_DATA_DEGRADED)

        # 4. Invariant: Entries MUST STILL BE BLOCKED because LATCHING_HALT_PROTECTION_FAILURE remains
        self.assertFalse(mgr.entries_allowed())
        self.assertIn(LATCHING_HALT_PROTECTION_FAILURE, mgr.active_halts())
        self.assertNotIn(TRANSIENT_HALT_DATA_DEGRADED, mgr.active_halts())

    def test_entry_lifecycle_protection_failure_registers_latched_halt(self):
        """Verify EntryOrderLifecycle._fail_protection records LATCHING_HALT_PROTECTION_FAILURE."""
        mgr = halt_manager
        mgr.reset_all()
        self.assertTrue(mgr.entries_allowed())

        trailing_mgr = MagicMock()
        lifecycle = EntryOrderLifecycle(trailing_mgr)

        sig = TradeSignal(
            symbol="NVDA",
            strategy=StrategyName.ORB,
            side="BUY",
            confidence=0.85,
            entry_price=100.0,
            stop_price=98.0,
            target_price=105.0,
            atr=1.5,
            risk_per_share=2.0,
            reason="Test Lifecycle Halt",
        )

        entry = lifecycle.register_submission(
            parent_order_id="p_test_100",
            signal=sig,
            requested_qty=10,
            stop_order_id="s_test_100",
            target_order_id="t_test_100",
        )

        bridge = MagicMock()
        # Mock bridge protection check failing (e.g. order not found on broker)
        bridge.get_order_snapshot = MagicMock(return_value=None)

        # Confirm fill triggers protection check which fails
        filled_ok = lifecycle.on_parent_fill(
            bridge=bridge,
            parent_order_id="p_test_100",
            cumulative_filled_qty=10,
            average_fill_price=100.0,
        )

        self.assertFalse(filled_ok)
        self.assertFalse(mgr.entries_allowed())
        self.assertIn(LATCHING_HALT_PROTECTION_FAILURE, mgr.active_halts())

    def test_multithreaded_concurrent_add_and_clear(self):
        """Verify EntryHaltManager RLock protects against concurrent thread races."""
        mgr = EntryHaltManager()
        iterations = 500

        def worker_transient():
            for _ in range(iterations):
                mgr.add(TRANSIENT_HALT_DATA_DEGRADED)
                mgr.clear_transient(TRANSIENT_HALT_DATA_DEGRADED)

        def worker_buffer():
            for _ in range(iterations):
                mgr.add(TRANSIENT_HALT_GLOBAL_OPENING_BUFFER)
                mgr.clear_transient(TRANSIENT_HALT_GLOBAL_OPENING_BUFFER)

        def worker_latched():
            for _ in range(50):
                mgr.add(LATCHING_HALT_PROTECTION_FAILURE)
                mgr.clear_transient(LATCHING_HALT_PROTECTION_FAILURE)  # Ignored

        t1 = threading.Thread(target=worker_transient)
        t2 = threading.Thread(target=worker_buffer)
        t3 = threading.Thread(target=worker_latched)

        t1.start()
        t2.start()
        t3.start()

        t1.join()
        t2.join()
        t3.join()

        # The latched halt must have survived all transient clears
        self.assertIn(LATCHING_HALT_PROTECTION_FAILURE, mgr.active_halts())
        self.assertFalse(mgr.entries_allowed())


if __name__ == "__main__":
    unittest.main()
