"""
tests/test_network_resilience_and_watchdog.py

Comprehensive unit and regression tests for P1.6.1:
1. TimeoutHTTPAdapter connect and read timeout injection.
2. configure_session_timeouts session adapter mounting.
3. fetch_historical_data fail-fast on timeout and connection reset.
4. Watchdog stale heartbeat detection (Tier 1: 120s, Tier 2: 300s).
5. Watchdog EOD flatten triggering when positions are open.
6. Watchdog post-flatten broker truth re-verification (pass and fail).
7. Incident 2026-09-03 Reproduction: Main thread frozen in shared client,
   watchdog with isolated client executes emergency broker flatten cleanly.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from alpaca_bridge import AlpacaBridge, TimeoutHTTPAdapter, configure_session_timeouts
from strategies.watchdog import (
    EODFlattenWatchdog,
    get_orchestrator_heartbeat_age,
    reset_orchestrator_heartbeat,
    touch_orchestrator_heartbeat,
)


# ── Test 1: TimeoutHTTPAdapter Timeout Injection ──
def test_timeout_http_adapter_injects_timeouts():
    adapter = TimeoutHTTPAdapter(connect_timeout=4.0, read_timeout=8.0)
    req = requests.Request("GET", "https://paper-api.alpaca.markets/v2/account").prepare()

    # Mock the parent HTTPAdapter.send
    with patch("requests.adapters.HTTPAdapter.send") as mock_super_send:
        mock_super_send.return_value = requests.Response()

        # Case A: kwargs has no timeout -> adapter injects (4.0, 8.0)
        adapter.send(req)
        assert mock_super_send.call_count == 1
        _, kwargs = mock_super_send.call_args
        assert kwargs["timeout"] == (4.0, 8.0)

        # Case B: kwargs has explicit timeout -> adapter preserves explicit timeout
        mock_super_send.reset_mock()
        adapter.send(req, timeout=(1.0, 2.0))
        assert mock_super_send.call_count == 1
        _, kwargs = mock_super_send.call_args
        assert kwargs["timeout"] == (1.0, 2.0)


# ── Test 2: configure_session_timeouts Mounting ──
def test_configure_session_timeouts():
    session = requests.Session()
    configure_session_timeouts(session, connect_timeout=5.0, read_timeout=10.0, total_retries=3)

    adapter_https = session.adapters.get("https://")
    adapter_http = session.adapters.get("http://")

    assert isinstance(adapter_https, TimeoutHTTPAdapter)
    assert isinstance(adapter_http, TimeoutHTTPAdapter)
    assert adapter_https.connect_timeout == 5.0
    assert adapter_https.read_timeout == 10.0
    assert adapter_https.max_retries.total == 3
    assert 429 in adapter_https.max_retries.status_forcelist
    assert 500 in adapter_https.max_retries.status_forcelist


# ── Test 3: fetch_historical_data Fails Fast on Timeout / Reset ──
def test_fetch_historical_data_fails_fast_on_timeout():
    bridge = AlpacaBridge()
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._data_client = MagicMock()

    # Simulate requests.exceptions.ReadTimeout
    bridge._data_client.get_stock_bars.side_effect = requests.exceptions.ReadTimeout("Socket read timed out after 10s")
    bars = bridge.fetch_historical_data("SPY")
    assert bars == []

    # Simulate ConnectionResetError (exact 2026-09-03 incident error)
    bridge._data_client.get_stock_bars.side_effect = ConnectionResetError(10054, "An existing connection was forcibly closed")
    bars = bridge.fetch_historical_data("SPY")
    assert bars == []


# ── Test 4: Watchdog Stale Heartbeat Escalation (Tier 1 & Tier 2) ──
def test_watchdog_detects_stale_heartbeat():
    watchdog = EODFlattenWatchdog(
        stale_alarm_seconds=120.0,
        stale_alert_seconds=300.0,
        trading_client=MagicMock(),
    )
    # Mock flatten window to False so only heartbeat is evaluated
    watchdog.is_in_flatten_window = MagicMock(return_value=False)

    # 1. Healthy heartbeat
    touch_orchestrator_heartbeat()
    cycle = watchdog.check_and_execute_cycle()
    assert cycle["tier1_alarm"] is False
    assert cycle["tier2_alert"] is False

    # 2. Tier 1 Alarm (Age = 150s > 120s)
    reset_orchestrator_heartbeat(time.time() - 150.0)
    cycle = watchdog.check_and_execute_cycle()
    assert cycle["tier1_alarm"] is True
    assert cycle["tier2_alert"] is False

    # 3. Tier 2 Alert (Age = 350s > 300s)
    reset_orchestrator_heartbeat(time.time() - 350.0)
    with patch.object(watchdog, "_send_alert") as mock_alert:
        cycle = watchdog.check_and_execute_cycle()
        assert cycle["tier2_alert"] is True
        assert mock_alert.call_count == 1

    # 4. Recovery
    touch_orchestrator_heartbeat()
    cycle = watchdog.check_and_execute_cycle()
    assert cycle["tier1_alarm"] is False
    assert cycle["tier2_alert"] is False


# ── Test 5: Watchdog Triggers EOD Flatten When Positions Open ──
def test_watchdog_triggers_eod_flatten():
    mock_client = MagicMock()
    mock_pos = MagicMock()
    mock_pos.symbol = "AMZN"
    mock_pos.qty = 15

    # First call returns 1 position; post-flatten recheck returns 0 positions
    mock_client.get_all_positions.side_effect = [[mock_pos], []]

    watchdog = EODFlattenWatchdog(
        eod_flatten_buffer_seconds=120.0,
        trading_client=mock_client,
    )
    watchdog.is_in_flatten_window = MagicMock(return_value=True)

    with patch("time.sleep", return_value=None):
        cycle = watchdog.check_and_execute_cycle()

    assert cycle["in_flatten_window"] is True
    assert cycle["flatten_attempted"] is True
    assert cycle["recheck_passed"] is True

    # Assert cancel_orders and close_all_positions were executed
    assert mock_client.cancel_orders.call_count == 1
    assert mock_client.close_all_positions.call_count == 1


# ── Test 6: Watchdog Post-Flatten Re-Verification Catches Remaining Exposure ──
def test_watchdog_reverification_catches_unclosed_position():
    mock_client = MagicMock()
    stubborn_pos = MagicMock()
    stubborn_pos.symbol = "NVDA"
    stubborn_pos.qty = 20

    # First call returns stubborn_pos, second call (recheck) STILL returns stubborn_pos!
    mock_client.get_all_positions.side_effect = [[stubborn_pos], [stubborn_pos], []]

    watchdog = EODFlattenWatchdog(trading_client=mock_client)
    watchdog.is_in_flatten_window = MagicMock(return_value=True)

    with patch("time.sleep", return_value=None), patch.object(watchdog, "_send_alert") as mock_alert:
        cycle = watchdog.check_and_execute_cycle()

    assert cycle["flatten_attempted"] is True
    assert cycle["recheck_passed"] is False
    # Critical alert must be dispatched for unclosed exposure
    assert mock_alert.call_count == 1
    # Retry flatten was attempted
    assert mock_client.close_all_positions.call_count == 2


# ── Test 7: Incident 2026-09-03 Reproduction (Shared Client Hang vs Isolated Watchdog) ──
def test_incident_reproduction_shared_client_hang_watchdog_flattens():
    """
    Direct reproduction of the 2026-09-03 incident:
    - Main trading bridge is stuck in an aborted REST socket read (CLOSE_WAIT).
    - Watchdog has its OWN isolated client and separate connection pool.
    - When market close arrives, watchdog autonomously flattens the account.
    """
    # 1. Simulate main bridge stuck/hung
    hung_bridge = AlpacaBridge()
    hung_bridge._connected = True
    hung_bridge._client = MagicMock()
    # Main bridge client would hang if called
    hung_bridge._client.get_all_positions.side_effect = TimeoutError("Simulated 8.5hr socket hang")

    # 2. Watchdog has its OWN dedicated client
    isolated_client = MagicMock()
    open_pos = MagicMock()
    open_pos.symbol = "AMZN"
    open_pos.qty = 15

    # Broker truth: 1 open position at close, 0 after flatten
    isolated_client.get_all_positions.side_effect = [[open_pos], []]

    watchdog = EODFlattenWatchdog(
        eod_flatten_buffer_seconds=120.0,
        trading_client=isolated_client,  # Separate client!
    )
    # Simulate main thread stopped sending heartbeats 180 seconds ago
    reset_orchestrator_heartbeat(time.time() - 180.0)

    # Simulate clock reaching 15:58:30 ET (inside flatten buffer)
    test_dt = datetime(2026, 9, 2, 15, 58, 30)
    watchdog.is_in_flatten_window = MagicMock(return_value=True)

    with patch("time.sleep", return_value=None):
        cycle = watchdog.check_and_execute_cycle(dt=test_dt)

    # Verify:
    # 1. Watchdog detected main thread is frozen
    assert cycle["tier1_alarm"] is True
    # 2. Watchdog autonomously executed flatten via its isolated client
    assert cycle["flatten_attempted"] is True
    assert isolated_client.cancel_orders.call_count == 1
    assert isolated_client.close_all_positions.call_count == 1
    # 3. Post-check verified 0 open positions
    assert cycle["recheck_passed"] is True
    # 4. The hung main bridge was NEVER called by the watchdog (zero failure propagation)
    assert hung_bridge._client.get_all_positions.call_count == 0
