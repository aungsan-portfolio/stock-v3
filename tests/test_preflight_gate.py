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
    check_portfolio_exposure,
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
