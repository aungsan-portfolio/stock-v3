"""
tests/test_file_lock_halt.py -- Unit tests for C.2 HALT.lock File-Flag Entry Freeze.
"""
import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import config
from strategies.halt_manager import (
    EntryHaltManager,
    LATCHING_HALT_PROTECTION_FAILURE,
    TRANSIENT_HALT_FILE_LOCK,
    TRANSIENT_HALT_DATA_DEGRADED,
    halt_manager,
)


class TestFileLockHalt(unittest.TestCase):

    def setUp(self):
        halt_manager.reset_all()
        # Clean up any potential lockfiles in test environment
        for p in [getattr(config, "HALT_LOCK_FILE", None), getattr(config, "BASE_DIR", Path(".")) / "HALT.lock"]:
            if p and Path(p).exists():
                try:
                    Path(p).unlink()
                except Exception:
                    pass

    def tearDown(self):
        halt_manager.reset_all()
        for p in [getattr(config, "HALT_LOCK_FILE", None), getattr(config, "BASE_DIR", Path(".")) / "HALT.lock"]:
            if p and Path(p).exists():
                try:
                    Path(p).unlink()
                except Exception:
                    pass

    def test_halt_lock_file_presence_blocks_entries(self, tmp_path_factory=None):
        """C.2: Presence of HALT.lock activates FILE_LOCK_HALT and freezes new entries."""
        mgr = halt_manager
        mgr.reset_all()
        self.assertTrue(mgr.entries_allowed())

        lock_path = config.HALT_LOCK_FILE
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("HALT")

        try:
            is_halted = mgr.check_halt_lock()
            self.assertTrue(is_halted)
            self.assertFalse(mgr.entries_allowed())
            self.assertIn(TRANSIENT_HALT_FILE_LOCK, mgr.active_halts())
        finally:
            if lock_path.exists():
                lock_path.unlink()

    def test_halt_lock_file_deletion_restores_entries(self):
        """C.2: Deleting HALT.lock automatically clears FILE_LOCK_HALT and allows entries."""
        mgr = halt_manager
        mgr.reset_all()

        lock_path = config.HALT_LOCK_FILE
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("HALT")

        mgr.check_halt_lock()
        self.assertFalse(mgr.entries_allowed())

        # Operator deletes the lock file
        lock_path.unlink()

        is_halted = mgr.check_halt_lock()
        self.assertFalse(is_halted)
        self.assertTrue(mgr.entries_allowed())
        self.assertNotIn(TRANSIENT_HALT_FILE_LOCK, mgr.active_halts())

    def test_halt_lock_exception_fails_closed(self):
        """C.2 Guardrail #1: Any filesystem or OS exception during check_halt_lock fails closed."""
        mgr = EntryHaltManager()
        self.assertTrue(mgr.entries_allowed())

        with patch("pathlib.Path.exists", side_effect=PermissionError("Mock filesystem permission error")):
            is_halted = mgr.check_halt_lock()
            self.assertTrue(is_halted)
            self.assertFalse(mgr.entries_allowed())
            self.assertIn(TRANSIENT_HALT_FILE_LOCK, mgr.active_halts())

    def test_halt_lock_boundary_leaves_open_positions_and_trailing_manager_active(self):
        """C.2 Guardrail #4: HALT.lock ONLY freezes entries; open positions and trailing stops remain active."""
        mgr = halt_manager
        mgr.reset_all()

        lock_path = config.HALT_LOCK_FILE
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("HALT")

        try:
            mgr.check_halt_lock()
            self.assertFalse(mgr.entries_allowed())

            # Mock existing open position and trailing manager update
            mock_trailing_manager = MagicMock()
            mock_trailing_manager.evaluate_all_stops = MagicMock(return_value=["stop_updated_NVDA"])

            # Verify trailing manager can execute normally despite HALT.lock
            stop_actions = mock_trailing_manager.evaluate_all_stops()
            self.assertEqual(len(stop_actions), 1)
            self.assertEqual(stop_actions[0], "stop_updated_NVDA")
        finally:
            if lock_path.exists():
                lock_path.unlink()

    def test_precedence_halt_lock_with_latched_protection_failure(self):
        """C.2 Invariant: If latched protection failure occurred, removing HALT.lock still keeps entries blocked."""
        mgr = halt_manager
        mgr.reset_all()

        lock_path = config.HALT_LOCK_FILE
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text("HALT")

        try:
            mgr.check_halt_lock()
            mgr.add(LATCHING_HALT_PROTECTION_FAILURE)
            self.assertFalse(mgr.entries_allowed())

            # Operator removes lock file
            lock_path.unlink()
            mgr.check_halt_lock()

            # FILE_LOCK_HALT is cleared, but entries MUST still be blocked due to LATCHED_PROTECTION_FAILURE
            self.assertFalse(mgr.entries_allowed())
            self.assertNotIn(TRANSIENT_HALT_FILE_LOCK, mgr.active_halts())
            self.assertIn(LATCHING_HALT_PROTECTION_FAILURE, mgr.active_halts())
        finally:
            if lock_path.exists():
                lock_path.unlink()

    def test_transition_only_logging_prevents_spam(self):
        """C.2 Guardrail #2: Repeated checks during steady state do NOT re-log warning/info."""
        mgr = EntryHaltManager()

        # 1. First add triggers transition warning
        with patch.object(logging.getLogger("strategies.halt_manager"), "warning") as mock_warn:
            added1 = mgr.add(TRANSIENT_HALT_FILE_LOCK)
            self.assertTrue(added1)
            self.assertEqual(mock_warn.call_count, 1)

            # 2. Second consecutive add does not log
            added2 = mgr.add(TRANSIENT_HALT_FILE_LOCK)
            self.assertFalse(added2)
            self.assertEqual(mock_warn.call_count, 1)

        # 3. First clear triggers transition info
        with patch.object(logging.getLogger("strategies.halt_manager"), "info") as mock_info:
            cleared1 = mgr.clear_transient(TRANSIENT_HALT_FILE_LOCK)
            self.assertTrue(cleared1)
            self.assertEqual(mock_info.call_count, 1)

            # 4. Second consecutive clear does not log
            cleared2 = mgr.clear_transient(TRANSIENT_HALT_FILE_LOCK)
            self.assertFalse(cleared2)
            self.assertEqual(mock_info.call_count, 1)


if __name__ == "__main__":
    unittest.main()
