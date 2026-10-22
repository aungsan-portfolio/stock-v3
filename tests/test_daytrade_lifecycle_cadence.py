"""
tests/test_daytrade_lifecycle_cadence.py -- Isolated Verification for Daytrade Engine Lifecycle:
1. Mid-bar startup -> startup hold active (entries held until next bar close, evaluation skipped).
2. Completed 5m bar -> startup hold cleared -> market gate allows -> strategy evaluation reached.
3. Market gate closed (score <= -3) -> entry evaluation suppressed and zero entries placed.
"""
from datetime import datetime, timezone, timedelta
import time
from unittest.mock import MagicMock, patch
import pytest

from strategies.regime_engine import MarketRegimeResult
from strategies.base import TradeSignal, StrategyName
from strategies.halt_manager import halt_manager
from strategies.trailing_stop import manager as trailing_manager
import strategies.orchestrator as orch


@pytest.fixture(autouse=True)
def reset_orchestrator_state():
    """Reset global cadence tracking, halt states, and trailing stop states before each test."""
    orch._last_evaluated_5m_bar_str = None
    orch._skipped_bar_ids = set()
    orch._last_regime_bar_str = None
    halt_manager.reset_all()
    trailing_manager.states.clear()
    trailing_manager._consecutive_persistence_failures = 0
    trailing_manager._persistence_healthy = True
    yield
    orch._last_evaluated_5m_bar_str = None
    orch._skipped_bar_ids = set()
    orch._last_regime_bar_str = None
    halt_manager.reset_all()
    trailing_manager.states.clear()
    trailing_manager._consecutive_persistence_failures = 0
    trailing_manager._persistence_healthy = True


def _create_mock_bridge():
    bridge = MagicMock()
    bridge.is_connected = True
    bridge._connected = True
    bridge._daytrade_suspended = False
    bridge.get_net_liquidation.return_value = 100_000.0
    bridge.get_open_positions.return_value = []
    bridge.get_positions.return_value = []
    bridge.open_position_count.return_value = 0
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.get_position.return_value = 0
    bridge.place_bracket_order = MagicMock()
    bridge._client = MagicMock()
    bridge._client.get_account.return_value = MagicMock(equity="100000.00", last_equity="100000.00")
    bridge.entry_gate = MagicMock(halted=False, reasons=[])
    return bridge


def test_midbar_startup_holds_entries_and_skips_evaluation():
    """
    Scenario 1: Mid-bar startup
    When engine boots mid-bar (elapsed: 150s > 60s grace),
    startup cadence sync holds entries and suppresses strategy evaluation.
    """
    bridge = _create_mock_bridge()
    watchlist = ["AAPL", "NVDA", "MSFT"]

    # Current time: 14:02:30 UTC -> current 5m bar started at 14:00:00 (age = 150s)
    mock_now = datetime(2026, 10, 16, 14, 2, 30, tzinfo=timezone.utc)

    regime_res = MarketRegimeResult(
        mode="RISK_ON",
        score=1,
        allow_new_longs=True,
        allow_new_shorts=False,
        halt_new_entries=False,
        requires_opening_buffer=False,
        position_scale=1.0,
        confidence_boost=0.0,
        market_texture="TRENDING",
    )

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.regime_engine.evaluate_market_regime", return_value=regime_res), \
         patch("strategies.orchestrator.evaluate_symbols_parallel") as mock_eval, \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.validation_lane.get_production_risk_equity", return_value=(100_000.0, 0.0)):

        mock_dt.now.return_value = mock_now
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        orch.evaluate_and_execute(
            watchlist=watchlist,
            bridge=bridge,
            live_paper=True,
            allow_new_entries=True,
        )

        # Strategy evaluation must be skipped during mid-bar startup hold
        mock_eval.assert_not_called()
        bridge.place_bracket_order.assert_not_called()


def test_completed_5m_bar_clears_hold_and_executes_strategy_evaluation():
    """
    Scenario 2: Completed 5m bar arrives & Market Gate Allows
    When bar boundary arrives (elapsed: 15s <= 90s pub_timeout) and market regime allows longs,
    startup hold is cleared and strategy evaluation is reached.
    """
    bridge = _create_mock_bridge()
    watchlist = ["AAPL", "NVDA", "MSFT"]

    # Pre-set previous evaluated bar to simulate prior mid-bar sync
    orch._last_evaluated_5m_bar_str = "2026-10-16T13:55:00+00:00"

    # Current time: 14:05:15 UTC -> current 5m bar started at 14:05:00 (age = 15s <= 90s)
    # completed 5m bar is 14:00:00 -> brand new completed bar boundary!
    mock_now = datetime(2026, 10, 16, 14, 5, 15, tzinfo=timezone.utc)

    regime_res = MarketRegimeResult(
        mode="RISK_ON",
        score=1,
        allow_new_longs=True,
        allow_new_shorts=False,
        halt_new_entries=False,
        requires_opening_buffer=False,
        position_scale=1.0,
        confidence_boost=0.0,
        market_texture="TRENDING",
    )

    mock_signal = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.CANDLESTICK,
        side="BUY",
        confidence=0.85,
        entry_price=150.0,
        stop_price=145.0,
        target_price=160.0,
        atr=2.0,
        risk_per_share=5.0,
        reason="Test pattern",
    )

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.regime_engine.evaluate_market_regime", return_value=regime_res), \
         patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=[mock_signal]) as mock_eval, \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.validation_lane.get_production_risk_equity", return_value=(100_000.0, 0.0)), \
         patch("strategies.execution_health.get_execution_health") as mock_health:

        mock_dt.now.return_value = mock_now
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        mock_health.return_value = MagicMock(
            allows_new_entries=True,
            status=MagicMock(value="healthy"),
            summary="HEALTHY",
            bar_age_seconds=15.0,
            broker_latency_ms=10.0,
        )

        orch.evaluate_and_execute(
            watchlist=watchlist,
            bridge=bridge,
            live_paper=True,
            allow_new_entries=True,
        )

        # Strategy evaluation MUST be reached and executed!
        mock_eval.assert_called_once()


def test_market_gate_closed_suppresses_evaluation_and_blocks_entries():
    """
    Scenario 3: Market Gate Closed (e.g. score=-3, halt_new_entries=True)
    Even when completed 5m bar boundary is reached, if market regime vetoes entries,
    strategy evaluation is suppressed and zero orders are placed.
    """
    bridge = _create_mock_bridge()
    watchlist = ["AAPL", "NVDA", "MSFT"]  # No inverse ETFs

    orch._last_evaluated_5m_bar_str = "2026-10-16T13:55:00+00:00"

    # Current time: 14:05:15 UTC (boundary reached)
    mock_now = datetime(2026, 10, 16, 14, 5, 15, tzinfo=timezone.utc)

    # Market Gate Closed (like today's live session with QQQ and SOXX down)
    regime_res = MarketRegimeResult(
        mode="CAUTION",
        score=-3,
        allow_new_longs=False,
        allow_new_shorts=False,
        halt_new_entries=True,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.0,
        market_texture="VOLATILE",
        reasons=["QQQ tech index down -1.33%", "SOXX semiconductor ETF weak -1.97%"],
    )

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.regime_engine.evaluate_market_regime", return_value=regime_res), \
         patch("strategies.orchestrator.evaluate_symbols_parallel") as mock_eval, \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.validation_lane.get_production_risk_equity", return_value=(100_000.0, 0.0)):

        mock_dt.now.return_value = mock_now
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        orch.evaluate_and_execute(
            watchlist=watchlist,
            bridge=bridge,
            live_paper=True,
            allow_new_entries=True,
        )

        # Because market gate is closed and watchlist has no inverse ETFs,
        # evaluation must be suppressed and no orders placed
        mock_eval.assert_not_called()
        bridge.place_bracket_order.assert_not_called()


def test_paper_mode_valid_candidate_traces_through_to_submission():
    """
    Scenario 4: Valid candidate end-to-end trace to broker submission in PAPER mode.
    Proves that when market conditions and safety checks are green,
    a valid setup is NOT blocked by false global halts and cleanly reaches broker submission.
    """
    from strategies.trailing_stop import get_trailing_manager
    tsm = get_trailing_manager()
    tsm.persistence_healthy = True
    tsm.last_persistence_error = None

    bridge = _create_mock_bridge()
    bridge.place_bracket_order.return_value = {
        "status": "submitted",
        "parent_order_id": "p_valid_999",
        "stop_order_id": "stop_valid_999",
        "tp_order_id": "tp_valid_999",
        "order_ids": ["p_valid_999", "tp_valid_999", "stop_valid_999"],
    }
    mock_order = MagicMock(
        id="p_valid_999",
        status="accepted",
        filled_qty=0,
        filled_avg_price=None,
    )
    bridge.get_order_snapshot.return_value = mock_order
    orch._last_evaluated_5m_bar_str = "2026-10-16T13:55:00+00:00"
    mock_now = datetime(2026, 10, 16, 14, 5, 15, tzinfo=timezone.utc)
    mock_now_ts = mock_now.timestamp()

    bridge.get_quote = MagicMock(return_value={
        "symbol": "AAPL",
        "bid": 150.00,
        "ask": 150.02,
        "last": 150.01,
        "timestamp": mock_now_ts,
        "is_realtime": True,
    })
    bridge.get_realtime_quote = bridge.get_quote

    regime_res = MarketRegimeResult(
        mode="RISK_ON",
        score=1,
        allow_new_longs=True,
        allow_new_shorts=False,
        halt_new_entries=False,
        requires_opening_buffer=False,
        position_scale=1.0,
        confidence_boost=0.0,
        market_texture="TRENDING",
    )

    valid_signal = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.CANDLESTICK,
        side="BUY",
        confidence=0.85,
        entry_price=150.0,
        stop_price=145.0,
        target_price=160.0,
        atr=2.0,
        risk_per_share=5.0,
        reason="Bullish Engulfing setup",
        timestamp="2026-10-16T14:05:00Z",
        bar_timestamp="2026-10-16 14:00:00",
    )
    valid_signal.metadata = {
        "raw_confidence": 0.85,
        "bar_duration_seconds": 300.0,
        "created_at": mock_now_ts,
    }

    import config
    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.regime_engine.evaluate_market_regime", return_value=regime_res), \
         patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=[valid_signal]), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.validation_lane.get_production_risk_equity", return_value=(100_000.0, 0.0)), \
         patch("strategies.execution_health.get_execution_health") as mock_health, \
         patch("strategies.execution_quote.time.time", return_value=mock_now_ts), \
         patch("strategies.order_manager.time.time", return_value=mock_now_ts), \
         patch.object(config, "SPY_TREND_GUARD_ENABLED", False), \
         patch.object(config, "VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", False), \
         patch.object(config, "LUNCH_CHOP_FILTER_ENABLED", False), \
         patch("strategies.orchestrator.day_trades_in_last_5_days", return_value=0):

        mock_dt.now.return_value = mock_now
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        mock_health.return_value = MagicMock(
            allows_new_entries=True,
            status=MagicMock(value="healthy"),
            summary="HEALTHY",
            bar_age_seconds=15.0,
            broker_latency_ms=10.0,
        )

        orch.evaluate_and_execute(
            watchlist=["AAPL"],
            bridge=bridge,
            live_paper=True,
            allow_new_entries=True,
        )

        # Candidate passed all gates and reached broker bracket order submission!
        bridge.place_bracket_order.assert_called_once()
        call_kwargs = bridge.place_bracket_order.call_args[1]
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["side"] == "BUY"
        assert call_kwargs["entry_price"] == 150.02
        assert call_kwargs["stop_price"] == 145.02
        assert call_kwargs["target_price"] == 160.02
