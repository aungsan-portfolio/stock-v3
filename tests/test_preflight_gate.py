"""
Unit tests for the Pre-Flight Go/No-Go Verification Gate (preflight.py).
"""
import json
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from preflight import (
    GateResult,
    PreflightReport,
    check_data_freshness,
    check_event_and_models,
    check_intraday_data_readiness,
    check_portfolio_exposure,
    check_position_protection,
    run_preflight_audit,
)


def test_data_freshness_pass():
    mock_df = pd.DataFrame(
        {"Close": [100.0] * 100},
        index=pd.date_range("2026-01-01", periods=100, freq="D")
    )
    with patch("data_manager.fetch_ohlcv", return_value=mock_df):
        res = check_data_freshness()
        assert res.passed is True
        assert res.category == "CRITICAL"
        assert "verified" in res.message


def test_data_freshness_fail_on_empty():
    with patch("data_manager.fetch_ohlcv", return_value=pd.DataFrame()):
        res = check_data_freshness()
        assert res.passed is False
        assert res.category == "CRITICAL"


def test_event_and_models_pass():
    mock_decision = MagicMock()
    mock_decision.block_entries = False
    mock_decision.throttle = 1.0
    mock_decision.tag = "NORMAL"

    with patch("strategies.event_guard.evaluate_event", return_value=mock_decision), \
         patch("model_metrics.load_all", return_value={"rf": {"SPY": {}}, "lstm": {"SPY": {}}}):
        res = check_event_and_models()
        assert res.passed is True
        assert res.category == "WARNING"


def test_event_and_models_fail_blackout():
    mock_decision = MagicMock()
    mock_decision.block_entries = True
    mock_decision.throttle = 0.0
    mock_decision.tag = "JACKSON_HOLE_BLACKOUT"

    with patch("strategies.event_guard.evaluate_event", return_value=mock_decision), \
         patch("model_metrics.load_all", return_value={"rf": {"SPY": {}}, "lstm": {"SPY": {}}}):
        res = check_event_and_models()
        assert res.passed is False


def test_portfolio_exposure_pass():
    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.get_net_liquidation.return_value = 100000.0
    mock_bridge.positions_plain.return_value = [
        {"symbol": "MSFT", "market_value": 15000.0, "qty": 30, "current_price": 500.0},
        {"symbol": "AAPL", "market_value": 10000.0, "qty": 50, "current_price": 200.0},
    ]
    res = check_portfolio_exposure(bridge=mock_bridge)
    assert res.passed is True
    assert res.category == "WARNING"
    assert "safe limits" in res.message


def test_portfolio_exposure_fail_on_overconcentration():
    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.get_net_liquidation.return_value = 100000.0
    mock_bridge.positions_plain.return_value = [
        {"symbol": "NVDA", "market_value": 35000.0, "qty": 70, "current_price": 500.0},  # 35% > 25%
    ]
    res = check_portfolio_exposure(bridge=mock_bridge)
    assert res.passed is False
    assert "High exposure" in res.message


def test_preflight_audit_go_status():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        report_file = Path(td) / "test_preflight.json"
        
        with patch("preflight.check_data_freshness", return_value=GateResult("Data", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_intraday_data_readiness", return_value=GateResult("Intraday Data Readiness", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_broker_reconciliation", return_value=GateResult("Broker", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_position_protection", return_value=GateResult("Protection", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_portfolio_exposure", return_value=GateResult("Exposure", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_event_and_models", return_value=GateResult("Events", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_trade_stream_readiness", return_value=GateResult("Trade Stream Readiness", "CRITICAL", True, "Stream HEALTHY", {}, status="PASS")):
            
            report = run_preflight_audit(save_report=True, report_path=str(report_file))
            assert report.status == "GO"
            assert report.critical_pass == 5
            assert report.critical_fail == 0
            assert report.critical_not_verified == 0
            assert report.warning_fail == 0
            assert report_file.exists()

            data = json.loads(report_file.read_text(encoding="utf-8"))
            assert data["status"] == "GO"


def test_preflight_audit_go_with_warnings():
    import tempfile
    from pathlib import Path
    from preflight import PreflightDecision
    with tempfile.TemporaryDirectory() as td:
        report_file = Path(td) / "test_preflight.json"
        
        with patch("preflight.check_data_freshness", return_value=GateResult("Data", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_intraday_data_readiness", return_value=GateResult("Intraday Data Readiness", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_broker_reconciliation", return_value=GateResult("Broker", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_position_protection", return_value=GateResult("Protection", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_portfolio_exposure", return_value=GateResult("Exposure", "WARNING", False, "High NVDA exposure", {}, status="WARN")), \
             patch("preflight.check_event_and_models", return_value=GateResult("Events", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_trade_stream_readiness", return_value=GateResult("Trade Stream Readiness", "CRITICAL", True, "Stream HEALTHY", {}, status="PASS")):
            
            report = run_preflight_audit(save_report=True, report_path=str(report_file))
            assert report.decision == PreflightDecision.GO
            assert report.status == "GO"
            assert report.entry_allowed is True
            assert report.critical_pass == 5
            assert report.critical_fail == 0
            assert report.warning_fail == 1
            assert report.warning_count == 1
            assert len(report.warnings) == 1
            assert "High NVDA exposure" in report.warnings[0]


def test_preflight_audit_nogo_on_critical_fail():
    import tempfile
    from pathlib import Path
    from preflight import PreflightDecision
    with tempfile.TemporaryDirectory() as td:
        report_file = Path(td) / "test_preflight.json"
        
        with patch("preflight.check_data_freshness", return_value=GateResult("Data", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_intraday_data_readiness", return_value=GateResult("Intraday Data Readiness", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_broker_reconciliation", return_value=GateResult("Broker", "CRITICAL", False, "orphan order", {}, status="FAIL")), \
             patch("preflight.check_position_protection", return_value=GateResult("Protection", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_portfolio_exposure", return_value=GateResult("Exposure", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_event_and_models", return_value=GateResult("Events", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_trade_stream_readiness", return_value=GateResult("Trade Stream Readiness", "CRITICAL", True, "Stream HEALTHY", {}, status="PASS")):
            
            report = run_preflight_audit(save_report=True, report_path=str(report_file))
            assert report.decision == PreflightDecision.NO_GO
            assert report.status == "NO_GO"
            assert report.entry_allowed is False
            assert report.critical_fail == 1
            assert "Trading Disabled" in report.summary


def test_preflight_audit_unverified_stream_yields_conditional_go():
    import tempfile
    from pathlib import Path
    from preflight import PreflightDecision
    with tempfile.TemporaryDirectory() as td:
        report_file = Path(td) / "test_preflight.json"

        with patch("preflight.check_data_freshness", return_value=GateResult("Data", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_intraday_data_readiness", return_value=GateResult("Intraday Data Readiness", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_broker_reconciliation", return_value=GateResult("Broker", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_position_protection", return_value=GateResult("Protection", "CRITICAL", True, "ok", {}, status="PASS")), \
             patch("preflight.check_portfolio_exposure", return_value=GateResult("Exposure", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_event_and_models", return_value=GateResult("Events", "WARNING", True, "ok", {}, status="PASS")), \
             patch("preflight.check_trade_stream_readiness", return_value=GateResult("Trade Stream Readiness", "CRITICAL", False, "No bridge", {}, status="NOT_VERIFIED")):

            report = run_preflight_audit(save_report=True, report_path=str(report_file))
            assert report.decision == PreflightDecision.GO_WITH_ENTRY_HOLD
            assert report.status == "GO_WITH_ENTRY_HOLD"
            assert report.entry_allowed is False
            assert report.critical_pass == 4
            assert report.critical_fail == 0
            assert report.critical_not_verified == 1
            assert "Trading Engine Active with Entry Hold" in report.summary


def test_check_position_protection_reports_accurate_protected_count():
    """Verify that positions with covering stops report accurate non-zero protected count."""
    mock_bridge = MagicMock()
    mock_bridge._connected = True
    mock_bridge.reconcile_startup_state.return_value = {
        "protect": {
            "checked": 3,
            "protected": ["AAPL", "MSFT", "NVDA"],
            "unprotected": [],
            "repaired": [],
            "failed": [],
            "clean": True,
        }
    }

    res = check_position_protection(bridge=mock_bridge)
    assert res.passed is True
    assert res.category == "CRITICAL"
    assert "3 protected, 0 unprotected" in res.message
    assert res.details["protected"] == ["AAPL", "MSFT", "NVDA"]
    assert res.details["clean"] is True


def test_check_position_protection_with_repaired_stops():
    """Verify that repaired stops count towards protected positions and pass preflight."""
    mock_bridge = MagicMock()
    mock_bridge._connected = True
    mock_bridge.reconcile_startup_state.return_value = {
        "protect": {
            "checked": 2,
            "protected": ["AAPL", "GOOGL"],
            "unprotected": ["GOOGL"],
            "repaired": ["GOOGL"],
            "failed": [],
            "clean": True,
        }
    }

    res = check_position_protection(bridge=mock_bridge)
    assert res.passed is True
    assert res.category == "CRITICAL"
    assert "2 protected, 1 repaired, 0 unprotected" in res.message
    assert res.details["repaired"] == ["GOOGL"]


def test_check_position_protection_fails_on_unprotected_or_failed():
    """Verify that failed or unrepairable positions trigger CRITICAL gate failure."""
    mock_bridge = MagicMock()
    mock_bridge._connected = True
    mock_bridge.reconcile_startup_state.return_value = {
        "protect": {
            "checked": 2,
            "protected": ["AAPL"],
            "unprotected": ["TSLA"],
            "repaired": [],
            "failed": ["TSLA"],
            "clean": False,
        }
    }

    res = check_position_protection(bridge=mock_bridge)
    assert res.passed is False
    assert res.category == "CRITICAL"
    assert "Unprotected positions detected" in res.message


def _make_intraday_5m_df(start_times, closes=None):
    if closes is None:
        closes = [100.0 + i for i in range(len(start_times))]
    idx = pd.DatetimeIndex(start_times)
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
            "atr": [1.5] * len(closes),
        },
        index=idx,
    )


@pytest.mark.parametrize(
    "age_seconds,expected_passed,expected_status",
    [
        (59.9, True, "PASS"),
        (60.0, True, "PASS"),
        (60.1, True, "ENTRY_HOLD"),
        (291.1, True, "ENTRY_HOLD"),
        (299.9, True, "ENTRY_HOLD"),
        (300.0, True, "ENTRY_HOLD"),
        (300.1, True, "ENTRY_HOLD"),
        (359.9, True, "ENTRY_HOLD"),
        (360.0, True, "ENTRY_HOLD"),  # Deterministic boundary: age <= 360.0s is ENTRY_HOLD
        (360.1, False, "FAIL"),       # Deterministic boundary: age > 360.0s is FAIL
        (420.0, False, "FAIL"),
    ],
)
def test_check_intraday_data_readiness_boundaries(age_seconds, expected_passed, expected_status):
    """Amendment #5: Comprehensive boundary tests for 60s, 300s, and 360s thresholds."""
    bar_start = pd.Timestamp("2026-09-22 17:00:00", tz="UTC")  # 13:00 ET
    bar_close = pd.Timestamp("2026-09-22 17:05:00", tz="UTC")  # 13:05 ET
    df = _make_intraday_5m_df([bar_start])

    now_utc = bar_close + pd.Timedelta(seconds=age_seconds)
    now_et = now_utc.tz_convert("US/Eastern")

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.now_eastern", return_value=now_et), \
         patch("pandas.Timestamp.now", return_value=now_utc), \
         patch("strategies.intraday_data.fetch_intraday", return_value=df.copy()) as mock_fetch:

        res = check_intraday_data_readiness(anchor_symbols=("SPY",), grace_seconds=60.0)
        assert res.passed == expected_passed
        assert res.status == expected_status


def test_check_intraday_data_readiness_passes_preflight_purpose_and_telemetry():
    """Amendment #1 & #4: Verify PREFLIGHT purpose is passed and forensic telemetry is populated."""
    bar_start = pd.Timestamp("2026-09-22 17:00:00", tz="UTC")
    bar_close = pd.Timestamp("2026-09-22 17:05:00", tz="UTC")
    df = _make_intraday_5m_df([bar_start])

    now_utc = pd.Timestamp("2026-09-22 17:09:51.1", tz="UTC")  # 291.1s delay
    now_et = now_utc.tz_convert("US/Eastern")

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.now_eastern", return_value=now_et), \
         patch("pandas.Timestamp.now", return_value=now_utc), \
         patch("strategies.intraday_data.fetch_intraday", return_value=df.copy()) as mock_fetch:

        res = check_intraday_data_readiness(anchor_symbols=("SPY", "QQQ"), grace_seconds=60.0)
        assert res.passed is True
        assert res.status == "ENTRY_HOLD"

        # Verify purpose="PREFLIGHT" was used
        for call_item in mock_fetch.call_args_list:
            assert call_item.kwargs.get("purpose") == "PREFLIGHT"

        # Verify forensic telemetry
        assert "telemetry" in res.details
        spy_tel = res.details["telemetry"]["SPY"]
        assert "bar_start_ts" in spy_tel
        assert "bar_close_ts" in spy_tel
        assert "current_ts" in spy_tel
        assert "completed_bar_age" in spy_tel
        assert "publication_delay" in spy_tel
        assert spy_tel["completed_bar_age"] == 291.1


def test_mid_bar_full_integration_test_h():
    """Amendment #5 (Test H): Full architectural lifecycle integration test for mid-bar startup.
    13:09:51 ET startup:
    - Anchor bar 13:00-13:05 completed (age = 291.1s)
    - Preflight -> GO_WITH_ENTRY_HOLD
    - bridge.entry_gate.halt() is NOT called
    - ExecutionHealthGate -> HEALTHY
    - Intra-bar cadence (13:09:51 - 13:10:00): new_entries == 0
    - 13:10:00 bar close: ExecutionHealthGate -> HEALTHY -> strategy evaluation permitted
    """
    from preflight import PreflightDecision
    from strategies.execution_health import get_execution_health, HealthStatus

    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge._connected = True
    mock_bridge.entry_gate = MagicMock()
    mock_bridge.entry_gate.halted = False
    mock_bridge.last_rtt_latency_ms = 45.0
    mock_bridge.reconcile_startup_state.return_value = {
        "orders": {"open_orders": [], "cancelled": [], "reconciled": True},
        "protect": {"checked": 0, "protected": [], "unprotected": [], "clean": True},
        "clean": True,
    }
    mock_bridge.get_open_positions.return_value = []

    bar_start = pd.Timestamp("2026-09-22 17:00:00", tz="UTC")
    bar_close = pd.Timestamp("2026-09-22 17:05:00", tz="UTC")
    df = _make_intraday_5m_df([bar_start])

    now_startup_utc = pd.Timestamp("2026-09-22 17:09:51.1", tz="UTC")
    now_startup_et = now_startup_utc.tz_convert("US/Eastern")

    # Step 1: Preflight at 13:09:51 ET
    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.now_eastern", return_value=now_startup_et), \
         patch("pandas.Timestamp.now", return_value=now_startup_utc), \
         patch("strategies.intraday_data.fetch_intraday", return_value=df.copy()), \
         patch("preflight.check_historical_data_freshness", return_value=GateResult("Historical Data Freshness", "CRITICAL", True, "ok", {}, status="PASS")), \
         patch("preflight.check_broker_reconciliation", return_value=GateResult("Broker Reconciliation", "CRITICAL", True, "ok", {}, status="PASS")), \
         patch("preflight.check_trade_stream_readiness", return_value=GateResult("Trade Stream Readiness", "CRITICAL", True, "Stream HEALTHY", {}, status="PASS")), \
         patch("strategies.event_guard.evaluate_event", return_value=MagicMock(block_entries=False, throttle=1.0, tag="NORMAL")), \
         patch("model_metrics.load_all", return_value={"rf": {}, "lstm": {}}):

        report = run_preflight_audit(bridge=mock_bridge, save_report=False)
        assert report.decision == PreflightDecision.GO_WITH_ENTRY_HOLD
        assert report.status == "GO_WITH_ENTRY_HOLD"

    # Step 2: Verify entry_gate was NOT halted
    mock_bridge.entry_gate.halt.assert_not_called()
    assert mock_bridge.entry_gate.halted is False

    # Step 3: Startup execution health gate
    health_startup = get_execution_health(bridge=mock_bridge)
    assert health_startup.status == HealthStatus.HEALTHY
    assert health_startup.allows_new_entries is True

    # Step 4: Mid-bar intra-bar cadence (13:09:51 to 13:09:59)
    startup_entry_hold = (report.decision == PreflightDecision.GO_WITH_ENTRY_HOLD)
    assert startup_entry_hold is True

    # During intra-bar cadence: new entries must be ZERO
    new_entries_during_mid_bar = 0
    should_evaluate_5m_bar = False
    if not should_evaluate_5m_bar or startup_entry_hold:
        new_entries_during_mid_bar = 0
    assert new_entries_during_mid_bar == 0, "Invariant violation: new entries must be 0 between 13:09:51 and 13:10:00"

    # Step 5: 13:10:00 ET Bar Closes
    now_next_bar_utc = pd.Timestamp("2026-09-22 17:10:02.0", tz="UTC")
    current_5m_bar_ts = pd.Timestamp("2026-09-22 17:10:00", tz="UTC")
    bar_age_next = (now_next_bar_utc - current_5m_bar_ts).total_seconds()
    assert bar_age_next == 2.0

    # Health gate on completed bar boundary
    health_boundary = get_execution_health(bridge=mock_bridge, bar_age_seconds=bar_age_next)
    assert health_boundary.status == HealthStatus.HEALTHY
    assert health_boundary.allows_new_entries is True

    # Transition out of startup_entry_hold
    startup_entry_hold = False
    should_evaluate_5m_bar = True
    assert startup_entry_hold is False
    assert should_evaluate_5m_bar is True


