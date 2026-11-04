"""tests/test_pipeline_sequence_and_funnel_telemetry.py -- Targeted tests for Lane 5 (Pipeline sequence & Funnel Telemetry v2.1)."""

import logging
from datetime import datetime, timezone
import pytest
from unittest.mock import MagicMock, patch

from strategies.base import TradeSignal, StrategyName
from strategies.regime_engine import MarketRegimeResult, MODE_RISK_ON, TEXTURE_TRANSITIONING, TEXTURE_TRENDING
import config


def _make_signal(symbol: str, strategy: StrategyName, confidence: float = 0.85, side: str = "BUY") -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        strategy=strategy,
        side=side,
        confidence=confidence,
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        atr=1.5,
        risk_per_share=2.0,
        reason="Targeted test signal",
    )


def test_pre_sizer_texture_admission_filtering(caplog, test_isolation_root):
    """Signals disallowed by market texture must be rejected BEFORE portfolio correlation and sizer."""
    import strategies.orchestrator as orch
    from strategies.orchestrator import EvaluationResult

    # Reset orchestrator intra-bar and hysteresis state
    orch._last_evaluated_5m_bar_str = None
    orch._skipped_bar_ids.clear()
    orch._regime_state_tracker._active_texture = TEXTURE_TRANSITIONING
    orch._regime_state_tracker._candidate_texture = TEXTURE_TRANSITIONING
    orch._regime_state_tracker._candidate_texture_count = 0

    sig_mom = _make_signal("TSLA", StrategyName.MOMENTUM_SCALP, confidence=0.8)
    sig_mom.signal_id = "texture-rejection-regression"
    sig_mr = _make_signal("SPY", StrategyName.MEAN_REVERSION, confidence=0.75)

    bridge_mock = MagicMock()
    bridge_mock.get_positions.return_value = []
    bridge_mock.get_open_orders.return_value = []
    bridge_mock.get_account.return_value = MagicMock(equity=100000.0)
    bridge_mock.get_trailing_stop_manager.return_value.has_unresolved_position_unknown.return_value = False
    bridge_mock.get_trailing_stop_manager.return_value.states = {}

    # In TRANSITIONING texture, MOMENTUM_SCALP is strictly forbidden
    regime = MarketRegimeResult(
        mode=MODE_RISK_ON,
        score=0,
        allow_new_longs=True,
        requires_opening_buffer=False,
        position_scale=1.0,
        confidence_boost=0.0,
        market_texture=TEXTURE_TRANSITIONING,
        data={"market_texture": TEXTURE_TRANSITIONING},
    )

    eval_result = EvaluationResult(
        entry_signals=(sig_mom, sig_mr),
        entry_allowed=True,
        suppression_reason=None,
        coverage=1.0,
        required_coverage=0.95,
        received_symbols=("TSLA", "SPY"),
        missing_symbols=(),
    )

    sizer_mock = MagicMock()
    correlation_mock = MagicMock(side_effect=lambda sigs, pos, bridge=None: sigs)

    now_utc = datetime(2026, 10, 1, 14, 0, 2, tzinfo=timezone.utc)
    now_et = datetime(2026, 10, 1, 10, 0, 2)

    with caplog.at_level(logging.INFO):
        with patch("strategies.orchestrator.datetime") as mock_dt, \
             patch("strategies.regime_engine.evaluate_market_regime", return_value=regime), \
             patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=eval_result), \
             patch("strategies.orchestrator.apply_portfolio_correlation", correlation_mock), \
             patch("strategies.orchestrator.calculate_shares", sizer_mock), \
             patch("strategies.orchestrator.execute_signal", return_value={"status": "PLACED", "shares": 10}), \
             patch("strategies.session.is_market_open", return_value=True), \
             patch("strategies.session.is_past_opening_buffer", return_value=True), \
             patch("strategies.session.now_eastern", return_value=now_et), \
             patch.object(config, "MOMENTUM_SCALP_ENABLED", True), \
             patch.object(config, "STRATEGY_MEAN_REVERSION_ENABLED", True):

            mock_dt.now.return_value = now_utc
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            orch.evaluate_and_execute(
                watchlist=["TSLA", "SPY"],
                bridge=bridge_mock,
                live_paper=False,
            )

    # Correlation should only have received the admitted signals (sig_mr), not sig_mom
    assert correlation_mock.called
    admitted_to_correlation = correlation_mock.call_args[0][0]
    admitted_symbols = [s.symbol for s in admitted_to_correlation]
    assert "TSLA" not in admitted_symbols
    assert "SPY" in admitted_symbols

    # Verify texture admission reject log
    assert any("[TEXTURE ADMISSION REJECT] Skipped TSLA" in rec.message for rec in caplog.records)

    # The rejected candidate must also appear as a terminal structured decision.
    import json
    ledger_path = test_isolation_root / "data" / "trade_gate_decision.jsonl"
    entries = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    rejected = [r for r in entries if r["evaluation_id"] == "texture-rejection-regression"]
    assert len(rejected) == 1
    assert rejected[0]["symbol"] == "TSLA"
    assert rejected[0]["decision"] == "REJECT"
    assert rejected[0]["reason_code"] == "TEXTURE_NOT_ALLOWED"
    assert rejected[0]["terminal_decision"] is True


def test_funnel_telemetry_dual_denominators_logged(caplog):
    """Funnel Telemetry v2.1 must log raw candidates, texture admitted, placed, and dual ratios."""
    import strategies.orchestrator as orch
    from strategies.orchestrator import EvaluationResult

    # Reset orchestrator intra-bar and hysteresis state
    orch._last_evaluated_5m_bar_str = None
    orch._skipped_bar_ids.clear()
    orch._regime_state_tracker._active_texture = TEXTURE_TRENDING
    orch._regime_state_tracker._candidate_texture = TEXTURE_TRENDING
    orch._regime_state_tracker._candidate_texture_count = 0

    sig1 = _make_signal("NVDA", StrategyName.MOMENTUM_SCALP, confidence=0.85)
    sig2 = _make_signal("AAPL", StrategyName.MOMENTUM_SCALP, confidence=0.80)

    bridge_mock = MagicMock()
    bridge_mock.get_positions.return_value = []
    bridge_mock.get_open_orders.return_value = []
    bridge_mock.get_account.return_value = MagicMock(equity=100000.0)
    bridge_mock.get_trailing_stop_manager.return_value.has_unresolved_position_unknown.return_value = False
    bridge_mock.get_trailing_stop_manager.return_value.has_pending_exit.return_value = False
    bridge_mock.get_trailing_stop_manager.return_value.states = {}

    regime = MarketRegimeResult(
        mode=MODE_RISK_ON,
        score=0,
        allow_new_longs=True,
        requires_opening_buffer=False,
        position_scale=1.0,
        confidence_boost=0.0,
        market_texture=TEXTURE_TRENDING,
        data={"market_texture": TEXTURE_TRENDING},  # MOMENTUM_SCALP is permitted in TRENDING
    )

    eval_result = EvaluationResult(
        entry_signals=(sig1, sig2),
        entry_allowed=True,
        suppression_reason=None,
        coverage=1.0,
        required_coverage=0.95,
        received_symbols=("NVDA", "AAPL"),
        missing_symbols=(),
    )

    now_utc = datetime(2026, 10, 1, 14, 0, 2, tzinfo=timezone.utc)
    now_et = datetime(2026, 10, 1, 10, 0, 2)

    with caplog.at_level(logging.INFO):
        with patch("strategies.orchestrator.datetime") as mock_dt, \
             patch("strategies.regime_engine.evaluate_market_regime", return_value=regime), \
             patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=eval_result), \
             patch("strategies.orchestrator.apply_portfolio_correlation", side_effect=lambda sigs, p, bridge=None: sigs), \
             patch("strategies.orchestrator.execute_signal", return_value={"status": "PLACED", "shares": 15}), \
             patch("strategies.session.is_market_open", return_value=True), \
             patch("strategies.session.is_past_opening_buffer", return_value=True), \
             patch("strategies.session.now_eastern", return_value=now_et), \
             patch.object(config, "MOMENTUM_SCALP_ENABLED", True):

            mock_dt.now.return_value = now_utc
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

            orch.evaluate_and_execute(
                watchlist=["NVDA", "AAPL", "MSFT", "AMZN"],
                bridge=bridge_mock,
                live_paper=False,
            )

    funnel_logs = [rec.message for rec in caplog.records if "[FUNNEL TELEMETRY v2.1]" in rec.message]
    assert len(funnel_logs) == 1
    log_text = funnel_logs[0]

    # Verify both denominators and fields are present
    assert "raw_eval_passes=2" in log_text
    assert "universe_candidates=4" in log_text
    assert "screened_passes=2" in log_text
    assert "texture_admitted=2" in log_text
    assert "placed=2" in log_text
    assert "legacy: placed/screened=100.0%" in log_text
    assert "placed/universe=50.0%" in log_text
    assert "normalized: placed/admitted=100.0%" in log_text
    assert "correlated/admitted=100.0%" in log_text
