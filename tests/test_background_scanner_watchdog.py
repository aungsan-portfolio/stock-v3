import time
import pytest
from unittest.mock import patch, MagicMock
from strategies.scanner.premarket_scanner import BackgroundScanner, ScanCandidate

def test_background_scanner_start_stop():
    """Verifies that BackgroundScanner starts and stops cleanly."""
    scanner = BackgroundScanner(interval_minutes=1, max_candidates=5)
    assert not scanner._running
    assert scanner._thread is None

    scanner.start()
    assert scanner._running
    assert scanner._thread is not None
    assert scanner._thread.is_alive()

    health = scanner.get_health_status()
    assert health["running"] is True
    assert health["thread_alive"] is True
    assert health["restart_count"] == 0

    scanner.stop()
    assert not scanner._running
    assert scanner._thread is None


def test_watchdog_auto_restarts_dead_thread():
    """Chaos test: kills the scanner thread and verifies ensure_running() revives it."""
    scanner = BackgroundScanner(interval_minutes=1, max_candidates=5)
    scanner.start()
    assert scanner._thread.is_alive()

    # Artificially kill the thread
    scanner._stop_event.set()
    time.sleep(0.1)
    scanner._thread = None

    health = scanner.get_health_status()
    assert health["thread_alive"] is False

    # Call watchdog
    scanner.ensure_running()

    # Verify auto-recovery
    assert scanner._thread is not None
    assert scanner._thread.is_alive()
    health_after = scanner.get_health_status()
    assert health_after["thread_alive"] is True
    assert health_after["restart_count"] >= 1

    scanner.stop()


def test_run_loop_survives_network_exceptions():
    """Verifies that unhandled network exceptions (e.g. 10054 ConnectionReset) do not crash the thread."""
    with patch("strategies.scanner.premarket_scanner.scan") as mock_scan, \
         patch("strategies.scanner.premarket_scanner.is_market_open", return_value=True):
        
        # Simulate network failure on scan
        mock_scan.side_effect = ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
        
        scanner = BackgroundScanner(interval_minutes=1, max_candidates=5)
        scanner.interval_seconds = 0.05  # Fast loop for testing
        scanner.start()

        # Let the loop execute and catch the exception
        time.sleep(0.3)

        # Thread must still be alive despite the exception
        assert scanner._thread is not None
        assert scanner._thread.is_alive()
        
        # Verify timestamp was updated to prevent runaway crash loops
        assert scanner.last_scan_completed > 0

        scanner.stop()
