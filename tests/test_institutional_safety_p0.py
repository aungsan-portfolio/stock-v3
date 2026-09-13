"""
tests/test_institutional_safety_p0.py -- Comprehensive verification test suite for P0 institutional safety:
1. 3-State Preflight Decisions (GO, GO_WITH_ENTRY_HOLD, NO_GO)
2. Raw Precision Confidence Gating (strict raw comparison, 0.5999999 rejected, 0.6000000 accepted)
3. Order Intent Separation & Fail-Closed Execution Boundary (exits bypass entry hold; missing readiness fails closed)
4. Publication Lag, Timeout, and Late-Bar Rejection (_skipped_bar_ids)
5. Immutable EvaluationResult Dataclass (frozen, iterable, sequence protocol, no custom __bool__)
6. Near-Miss Telemetry & Aggregation
7. Config Range Validation (publication timeouts, poll intervals, coverage bounds)
"""

import time
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from dataclasses import FrozenInstanceError

import config
from preflight import (
    PreflightDecision,
    PreflightReport,
    GateResult,
    run_preflight_audit,
    check_intraday_data_readiness,
)
from strategies.base import TradeSignal, EntryReadiness
from strategies.constants import StrategyName, OrderIntent, EXIT_INTENTS
import strategies.orchestrator as orch
import strategies.order_manager as om


# ==============================================================================
# 1. 3-State Preflight Decision Tests
# ==============================================================================

def test_preflight_tri_state_go():
    """When all critical gates pass and no entry hold is required, decision is GO."""
    with patch("preflight.check_historical_data_freshness", return_value=GateResult("Data", "CRITICAL", True, "ok", {}, status="PASS")), \
         patch("preflight.check_intraday_data_readiness", return_value=GateResult("Intraday", "CRITICAL", True, "fresh", {}, status="PASS")), \
         patch("preflight.check_broker_reconciliation", return_value=GateResult("Recon", "CRITICAL", True, "clean", {}, status="PASS")), \
         patch("preflight.check_position_protection", return_value=GateResult("Protect", "CRITICAL", True, "100%", {}, status="PASS")), \
         patch("preflight.check_portfolio_exposure", return_value=GateResult("Exposure", "WARNING", True, "ok", {}, status="PASS")), \
         patch("preflight.check_event_and_models", return_value=GateResult("Event", "WARNING", True, "ok", {}, status="PASS")), \
         patch("preflight.check_trade_stream_readiness", return_value=GateResult("Stream", "CRITICAL", True, "healthy", {}, status="PASS")):

        report = run_preflight_audit(save_report=False)
        assert report.decision == PreflightDecision.GO
        assert report.status == "GO"
        assert report.entry_allowed is True
        assert report.hold_reason is None
        assert report.critical_fail == 0
        assert report.warning_count == 0


def test_preflight_tri_state_go_with_entry_hold_mid_bar():
    """When mid-bar startup occurs (age between 60s and 360s), decision is GO_WITH_ENTRY_HOLD."""
    with patch("preflight.check_historical_data_freshness", return_value=GateResult("Data", "CRITICAL", True, "ok", {}, status="PASS")), \
         patch("preflight.check_intraday_data_readiness", return_value=GateResult("Intraday", "CRITICAL", True, "Mid-bar startup: Anchor bars valid within current forming interval", {}, status="ENTRY_HOLD")), \
         patch("preflight.check_broker_reconciliation", return_value=GateResult("Recon", "CRITICAL", True, "clean", {}, status="PASS")), \
         patch("preflight.check_position_protection", return_value=GateResult("Protect", "CRITICAL", True, "100%", {}, status="PASS")), \
         patch("preflight.check_portfolio_exposure", return_value=GateResult("Exposure", "WARNING", True, "ok", {}, status="PASS")), \
         patch("preflight.check_event_and_models", return_value=GateResult("Event", "WARNING", True, "ok", {}, status="PASS")), \
         patch("preflight.check_trade_stream_readiness", return_value=GateResult("Stream", "CRITICAL", True, "healthy", {}, status="PASS")):

        report = run_preflight_audit(save_report=False)
        assert report.decision == PreflightDecision.GO_WITH_ENTRY_HOLD
        assert report.status == "GO_WITH_ENTRY_HOLD"
        assert report.entry_allowed is False
        assert report.hold_reason is not None
        assert "Mid-bar startup" in report.hold_reason
        assert report.critical_fail == 0


def test_preflight_tri_state_no_go_on_critical_fail():
    """When any critical gate fails, decision is NO_GO regardless of entry hold or warnings."""
    with patch("preflight.check_historical_data_freshness", return_value=GateResult("Data", "CRITICAL", False, "Missing data", {}, status="FAIL")), \
         patch("preflight.check_intraday_data_readiness", return_value=GateResult("Intraday", "CRITICAL", True, "Mid-bar", {}, status="ENTRY_HOLD")), \
         patch("preflight.check_broker_reconciliation", return_value=GateResult("Recon", "CRITICAL", True, "clean", {}, status="PASS")), \
         patch("preflight.check_position_protection", return_value=GateResult("Protect", "CRITICAL", True, "100%", {}, status="PASS")), \
         patch("preflight.check_portfolio_exposure", return_value=GateResult("Exposure", "WARNING", False, "Overweight", {}, status="WARN")), \
         patch("preflight.check_event_and_models", return_value=GateResult("Event", "WARNING", True, "ok", {}, status="PASS")), \
         patch("preflight.check_trade_stream_readiness", return_value=GateResult("Stream", "CRITICAL", True, "healthy", {}, status="PASS")):

        report = run_preflight_audit(save_report=False)
        assert report.decision == PreflightDecision.NO_GO
        assert report.status == "NO_GO"
        assert report.entry_allowed is False
        assert report.critical_fail == 1
        assert "critical gate(s) failed" in report.hold_reason
        assert report.warning_count == 1


# ==============================================================================
# 2. Raw Precision Confidence Gating Tests
# ==============================================================================

def test_raw_precision_confidence_gating():
    """Strict raw float comparison: 0.5999999 is rejected when threshold is 0.60, 0.6000000 is accepted."""
    orch._last_evaluated_5m_bar_str = None
    orch._skipped_bar_ids.clear()

    now_utc = datetime(2026, 9, 11, 18, 0, 10, tzinfo=timezone.utc)

    # Sub-threshold signal: 0.5999999 < 0.6000000
    sig_sub = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.5999999,
        entry_price=150.0,
        stop_price=148.0,
        target_price=155.0,
        atr=2.0,
        risk_per_share=2.0,
        reason="MOMENTUM_TEST",
    )

    # Exact threshold signal: 0.6000000 >= 0.6000000
    sig_exact = TradeSignal(
        symbol="MSFT",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.6000000,
        entry_price=300.0,
        stop_price=295.0,
        target_price=310.0,
        atr=5.0,
        risk_per_share=5.0,
        reason="MOMENTUM_TEST",
    )

    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.get_net_liquidation.return_value = 100000.0
    mock_bridge.get_open_positions.return_value = []
    mock_bridge.get_working_orders.return_value = []

    eval_result = orch.EvaluationResult(
        entry_signals=(sig_sub, sig_exact),
        entry_allowed=True,
        suppression_reason=None,
        coverage=1.0,
        required_coverage=0.95,
        received_symbols=("AAPL", "MSFT"),
        missing_symbols=(),
    )

    from zoneinfo import ZoneInfo
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=eval_result), \
         patch("strategies.orchestrator._min_confidence_for", return_value=0.6000000), \
         patch("strategies.orchestrator.execute_signal") as mock_exec, \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.session.now_eastern", return_value=now_et):

        mock_dt.now.return_value = now_utc
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        orch.evaluate_and_execute(["AAPL", "MSFT"], bridge=mock_bridge)

        # sig_sub (0.5999999) must have been rejected by the strict confidence check!
        # Only sig_exact (0.6000000) should have reached execute_signal.
        assert mock_exec.call_count == 1
        executed_sig = mock_exec.call_args[1]["signal"]
        assert executed_sig.symbol == "MSFT"
        assert executed_sig.confidence == 0.6000000


# ==============================================================================
# 3. Order Intent Separation & Fail-Closed Execution Boundary Tests
# ==============================================================================

def test_execution_boundary_fail_closed_missing_readiness():
    """Entry signal without EntryReadiness context token is rejected immediately."""
    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=120.0,
        stop_price=116.0,
        target_price=128.0,
        atr=2.0,
        risk_per_share=4.0,
        reason="BREAKOUT",
        intent=OrderIntent.ENTRY,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0

    res = om.execute_signal(
        signal=sig,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        entry_readiness=None,  # Missing readiness token
    )
    assert res["status"] == "REJECTED"
    assert "ENTRY_READINESS_CONTEXT_MISSING" in res["reason"]


def test_execution_boundary_suppressed_readiness():
    """Entry signal with entry_allowed=False is rejected with suppression reason."""
    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=120.0,
        stop_price=116.0,
        target_price=128.0,
        atr=2.0,
        risk_per_share=4.0,
        reason="BREAKOUT",
        intent=OrderIntent.ENTRY,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0

    readiness = EntryReadiness(
        entry_allowed=False,
        suppression_reason="LOW_COVERAGE: 80% < 95%",
        cycle_id="c123",
        bar_ts="2026-09-11T18:00:00+00:00",
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )

    res = om.execute_signal(
        signal=sig,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        entry_readiness=readiness,
    )
    assert res["status"] == "REJECTED"
    assert "ENTRY_READINESS_SUPPRESSED" in res["reason"]
    assert "LOW_COVERAGE" in res["reason"]


def test_execution_boundary_expired_readiness():
    """Entry signal with expired readiness token is rejected."""
    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=120.0,
        stop_price=116.0,
        target_price=128.0,
        atr=2.0,
        risk_per_share=4.0,
        reason="BREAKOUT",
        intent=OrderIntent.ENTRY,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0

    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="c123",
        bar_ts="2026-09-11T18:00:00+00:00",
        created_at=time.time() - 350.0,
        expires_at=time.time() - 50.0,  # Expired
    )

    res = om.execute_signal(
        signal=sig,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        entry_readiness=readiness,
    )
    assert res["status"] == "REJECTED"
    assert "ENTRY_READINESS_EXPIRED" in res["reason"]


def test_execution_boundary_exit_intent_bypasses_readiness():
    """Exit orders (STOP_PROTECTION, EOD_FLATTEN, etc.) completely bypass entry readiness gates."""
    for exit_intent in EXIT_INTENTS:
        sig = TradeSignal(
            symbol="AAPL",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="SELL",
            confidence=0.90,
            entry_price=150.0,
            stop_price=145.0,
            target_price=160.0,
            atr=2.0,
            risk_per_share=5.0,
            reason="PROTECTIVE_EXIT",
            intent=exit_intent,
        )

        mock_bridge = MagicMock()
        mock_bridge.get_net_liquidation.return_value = 100000.0

        # No entry readiness passed! Must NOT be rejected for ENTRY_READINESS_CONTEXT_MISSING
        res = om.execute_signal(
            signal=sig,
            bridge=mock_bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            entry_readiness=None,
        )
        assert "ENTRY_READINESS" not in res.get("reason", "")


def test_execution_boundary_bar_mismatch_readiness():
    """Entry signal with mismatched bar timestamp is rejected."""
    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=120.0,
        stop_price=116.0,
        target_price=128.0,
        atr=2.0,
        risk_per_share=4.0,
        reason="BREAKOUT",
        intent=OrderIntent.ENTRY,
        bar_timestamp="2026-09-11T17:55:00+00:00",
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0

    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="c123",
        bar_ts="2026-09-11T18:00:00+00:00",  # Mismatched bar ts
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )

    res = om.execute_signal(
        signal=sig,
        bridge=mock_bridge,
        equity=100000.0,
        current_pnl=0.0,
        entry_readiness=readiness,
    )
    assert res["status"] == "REJECTED"
    assert "ENTRY_READINESS_BAR_MISMATCH" in res["reason"]


def test_execution_boundary_readiness_disabled():
    """When DAYTRADE_REQUIRE_ENTRY_READINESS is disabled, missing readiness does not reject."""
    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=120.0,
        stop_price=116.0,
        target_price=128.0,
        atr=2.0,
        risk_per_share=4.0,
        reason="BREAKOUT",
        intent=OrderIntent.ENTRY,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0
    mock_bridge.open_position_count.return_value = 0
    mock_bridge.has_position.return_value = False
    mock_bridge.has_working_order.return_value = False
    mock_bridge.get_open_positions.return_value = []
    mock_bridge.get_working_orders.return_value = []

    with patch.object(config, "DAYTRADE_REQUIRE_ENTRY_READINESS", False), \
         patch("strategies.order_manager.pre_trade_check", return_value=None):
        res = om.execute_signal(
            signal=sig,
            bridge=mock_bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            entry_readiness=None,
        )
        assert "ENTRY_READINESS" not in res.get("reason", "")


# ==============================================================================
# 4. Publication Lag, Timeout, and Late-Bar Rejection Tests
# ==============================================================================

def test_bar_publication_timeout_adds_to_skipped_bars():
    """When elapsed time since bar close exceeds timeout (90s), bar is added to _skipped_bar_ids and suppressed."""
    orch._last_evaluated_5m_bar_str = "2026-09-11T17:50:00+00:00"
    orch._skipped_bar_ids.clear()

    # Completed bar closed at 18:00:00 UTC. Now is 18:01:35 UTC (age = 95s > 90s timeout).
    late_now = datetime(2026, 9, 11, 18, 1, 35, tzinfo=timezone.utc)
    completed_bar_str = "2026-09-11T17:55:00+00:00"

    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.get_net_liquidation.return_value = 100000.0
    mock_bridge.get_open_positions.return_value = []
    mock_bridge.get_working_orders.return_value = []

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.orchestrator.evaluate_symbols_parallel") as mock_eval, \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch.object(config, "DAYTRADE_BAR_PUBLICATION_TIMEOUT_S", 90.0):

        mock_dt.now.return_value = late_now
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge)

        # Evaluation should NOT run because bar timed out
        mock_eval.assert_not_called()
        # Bar should be recorded in _skipped_bar_ids
        assert completed_bar_str in orch._skipped_bar_ids


def test_late_arriving_bar_rejected_by_skipped_bar_ids():
    """If a bar previously marked skipped arrives later, it is explicitly rejected and suppressed."""
    orch._last_evaluated_5m_bar_str = "some_older_bar"
    orch._skipped_bar_ids = {"2026-09-11T17:55:00+00:00"}

    # Time inside current bar
    now_utc = datetime(2026, 9, 11, 18, 0, 15, tzinfo=timezone.utc)

    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.get_net_liquidation.return_value = 100000.0
    mock_bridge.get_open_positions.return_value = []
    mock_bridge.get_working_orders.return_value = []

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.orchestrator.evaluate_symbols_parallel") as mock_eval, \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.is_past_opening_buffer", return_value=True):

        mock_dt.now.return_value = now_utc
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge)

        mock_eval.assert_not_called()


# ==============================================================================
# 5. Immutable EvaluationResult Dataclass Tests
# ==============================================================================

def test_evaluation_result_immutability_and_sequence_protocol():
    """EvaluationResult is frozen, immutable, and implements sequence protocol without __bool__ override."""
    sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.8,
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
        atr=2.0,
        risk_per_share=5.0,
        reason="OK",
    )

    res = orch.EvaluationResult(
        entry_signals=(sig,),
        entry_allowed=True,
        suppression_reason=None,
        coverage=1.0,
        required_coverage=0.95,
        received_symbols=("AAPL",),
        missing_symbols=(),
    )

    # Sequence protocol
    assert len(res) == 1
    assert res[0] == sig
    assert list(res) == [sig]

    # Immutability
    with pytest.raises(FrozenInstanceError):
        res.entry_allowed = False


# ==============================================================================
# 6. Config Range Validation Tests
# ==============================================================================

def test_institutional_safety_config_validation():
    """Validates configuration bounds for publication timeout, poll interval, and coverage."""
    # Valid configuration should pass
    config.validate_institutional_safety_config()

    # Invalid publication timeout <= 0
    with patch.object(config, "DAYTRADE_BAR_PUBLICATION_TIMEOUT_S", 0.0):
        with pytest.raises(ValueError, match="DAYTRADE_BAR_PUBLICATION_TIMEOUT_S must be positive"):
            config.validate_institutional_safety_config()

    # Invalid poll interval >= timeout
    with patch.object(config, "DAYTRADE_BAR_PUBLICATION_POLL_INTERVAL_S", 95.0), \
         patch.object(config, "DAYTRADE_BAR_PUBLICATION_TIMEOUT_S", 90.0):
        with pytest.raises(ValueError, match="DAYTRADE_BAR_PUBLICATION_POLL_INTERVAL_S must be in"):
            config.validate_institutional_safety_config()

    # Invalid coverage <= 0 or > 1.0
    with patch.object(config, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 1.05):
        with pytest.raises(ValueError, match="DAYTRADE_BATCH_FETCH_MIN_COVERAGE must be in"):
            config.validate_institutional_safety_config()
