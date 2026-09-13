"""
test_data_readiness_and_startup.py -- Verification test suite for:
1. Intraday data readiness gate (preflight warmup, anchor bar freshness, market-open grace)
2. Startup cadence synchronization (mid-bar boot synchronization vs fresh boot evaluation)
3. Low-coverage fail-closed protection (suppresses entries when batch coverage < 95%)
4. Confidence scoring breakdown and [NEAR-MISS] logging
5. Chase guard telemetry (quote_age_ms and chase_to_stop_ratio_pct)
"""

import math
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import config
from preflight import check_intraday_data_readiness, GateResult, PreflightReport
from strategies.base import TradeSignal, EntryReadiness
from strategies.constants import StrategyName
from strategies.scoring import RiskAdjustedSignalScorer, ConfidenceBreakdown
from strategies.execution_quote import BrokerQuote, QuoteValidationResult
from strategies.session import EASTERN_TZ
import strategies.orchestrator as orch
import strategies.order_manager as om


# =====================================================================
# 1. Intraday Data-Readiness Gate Tests
# =====================================================================

def test_preflight_intraday_warmup_period():
    """At 09:32 ET (within first 5 min of open), no completed 5m bar exists yet. Gate must pass with warmup notice."""
    mock_bridge = MagicMock()
    warmup_time_et = datetime(2026, 9, 11, 9, 32, 0, tzinfo=EASTERN_TZ)

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.now_eastern", return_value=warmup_time_et):

        res = check_intraday_data_readiness(mock_bridge)
        assert res.passed is True
        assert "warmup" in res.message.lower()
        assert res.details.get("warmup") is True


def test_preflight_intraday_fresh_anchor_bar():
    """At 10:05 ET (14:05 UTC), anchor symbol SPY has timely bar (delay 25s <= 60s). Gate passes."""
    mock_bridge = MagicMock()
    time_et = datetime(2026, 9, 11, 10, 5, 0, tzinfo=EASTERN_TZ)

    df_spy = pd.DataFrame({"close": [500.0] * 5})
    mock_completed_bar = MagicMock()
    mock_completed_bar.post_close_delay_seconds = 25.0

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.now_eastern", return_value=time_et), \
         patch("strategies.intraday_data.fetch_intraday", return_value=df_spy), \
         patch("strategies.intraday_data.select_last_completed_bar", return_value=mock_completed_bar):

        res = check_intraday_data_readiness(mock_bridge, anchor_symbols=("SPY",), grace_seconds=60.0)
        assert res.passed is True
        assert "fresh within 60.0s" in res.message


def test_preflight_intraday_entry_hold_mid_bar():
    """At 10:05 ET (14:05 UTC), anchor symbol SPY delay is 120s (60s < 120s <= 360s). Gate returns ENTRY_HOLD."""
    mock_bridge = MagicMock()
    time_et = datetime(2026, 9, 11, 10, 5, 0, tzinfo=EASTERN_TZ)

    df_spy = pd.DataFrame({"close": [500.0] * 5})
    mock_completed_bar = MagicMock()
    mock_completed_bar.post_close_delay_seconds = 120.0

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.now_eastern", return_value=time_et), \
         patch("strategies.intraday_data.fetch_intraday", return_value=df_spy), \
         patch("strategies.intraday_data.select_last_completed_bar", return_value=mock_completed_bar):

        res = check_intraday_data_readiness(mock_bridge, anchor_symbols=("SPY",), grace_seconds=60.0)
        assert res.passed is True
        assert res.status == "ENTRY_HOLD"
        assert "Mid-bar startup" in res.message
        assert "SPY:120s" in res.message


def test_preflight_intraday_stale_anchor_bar():
    """At 10:05 ET (14:05 UTC), anchor symbol SPY delay is 400s > 360s (interval + grace). Gate fails critical audit."""
    mock_bridge = MagicMock()
    time_et = datetime(2026, 9, 11, 10, 5, 0, tzinfo=EASTERN_TZ)

    df_spy = pd.DataFrame({"close": [500.0] * 5})
    mock_completed_bar = MagicMock()
    mock_completed_bar.post_close_delay_seconds = 400.0

    with patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.now_eastern", return_value=time_et), \
         patch("strategies.intraday_data.fetch_intraday", return_value=df_spy), \
         patch("strategies.intraday_data.select_last_completed_bar", return_value=mock_completed_bar):

        res = check_intraday_data_readiness(mock_bridge, anchor_symbols=("SPY",), grace_seconds=60.0)
        assert res.passed is False
        assert res.status == "FAIL"
        assert "Stale anchor bars" in res.message
        assert "SPY:400s > 360s" in res.message


# =====================================================================
# 2. Startup Cadence Synchronization Tests
# =====================================================================

def test_startup_cadence_sync_mid_bar_skips_evaluation():
    """Booting mid-bar (e.g. bar age 195s > grace 60s): orchestrator synchronizes and suppresses evaluation."""
    orch._last_evaluated_5m_bar_str = None
    orch._skipped_bar_ids.clear()

    # 17:58:15 UTC -> current open bar started at 17:55:00 UTC.
    # Completed bar closed at 17:55:00 UTC. Bar age = 195s (> 60s).
    boot_time = datetime(2026, 9, 11, 17, 58, 15, tzinfo=timezone.utc)

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.orchestrator.evaluate_symbols_parallel") as mock_eval, \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch.object(config, "DAYTRADE_STARTUP_CADENCE_GRACE_S", 60.0):
        mock_dt.now.return_value = boot_time
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.get_net_liquidation.return_value = 100000.0
        mock_bridge.get_open_positions.return_value = []
        mock_bridge.get_working_orders.return_value = []

        # Run cycle
        orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge)

        # Candidate evaluation should NOT have been run mid-bar
        mock_eval.assert_not_called()
        # _last_evaluated_5m_bar_str should now be set to the completed bar (17:50:00)
        assert orch._last_evaluated_5m_bar_str == "2026-09-11T17:50:00+00:00"


def test_startup_cadence_fresh_bar_runs_evaluation():
    """Booting freshly at bar boundary (e.g. bar age 20s <= grace 60s): orchestrator evaluates immediately."""
    orch._last_evaluated_5m_bar_str = None
    orch._skipped_bar_ids.clear()

    # 17:55:20 UTC -> completed bar closed at 17:55:00 UTC. Bar age = 20s (<= 60s).
    boot_time = datetime(2026, 9, 11, 17, 55, 20, tzinfo=timezone.utc)

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=[]) as mock_eval, \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch.object(config, "DAYTRADE_STARTUP_CADENCE_GRACE_S", 60.0):
        mock_dt.now.return_value = boot_time
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.get_net_liquidation.return_value = 100000.0
        mock_bridge.get_open_positions.return_value = []
        mock_bridge.get_working_orders.return_value = []

        orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge)

        # Candidate evaluation SHOULD have been called because candle was fresh
        mock_eval.assert_called_once()
        assert orch._last_evaluated_5m_bar_str == "2026-09-11T17:50:00+00:00"


# =====================================================================
# 3. Low-Coverage Entry Suppression Tests
# =====================================================================

def test_low_coverage_suppresses_entries():
    """If intraday batch coverage is 40% (< 95%), evaluate_symbols_parallel returns [] to avoid selection bias."""
    symbols = [f"SYM{i}" for i in range(10)]
    partial_data = {
        "SYM0": pd.DataFrame({"close": [10.0] * 5}),
        "SYM1": pd.DataFrame({"close": [20.0] * 5}),
        "SYM2": pd.DataFrame({"close": [30.0] * 5}),
        "SYM3": pd.DataFrame({"close": [40.0] * 5}),
    }
    telemetry = {"requested": 10, "received": 4, "coverage": 0.4, "duration_ms": 15.0}

    with patch.object(config, "DAYTRADE_BATCH_FETCH_ENABLED", True), \
         patch.object(config, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95), \
         patch("strategies.intraday_data.fetch_intraday_batch", return_value=(partial_data, telemetry)), \
         patch("strategies.orchestrator.recover_missing_intraday", return_value=({}, {})):

        signals = orch.evaluate_symbols_parallel(symbols, bridge=MagicMock())
        # Must fail-closed and return empty list
        assert len(signals) == 0
        assert signals.entry_allowed is False


def test_high_coverage_permits_evaluation():
    """If intraday batch coverage is 100% (>= 95%), evaluate_symbols_parallel evaluates all symbols."""
    symbols = ["AAPL", "MSFT"]
    data = {
        "AAPL": pd.DataFrame({"close": [150.0] * 5}),
        "MSFT": pd.DataFrame({"close": [300.0] * 5}),
    }
    telemetry = {"requested": 2, "received": 2, "coverage": 1.0, "duration_ms": 10.0}

    mock_sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        atr=0.5,
        risk_per_share=2.0,
        reason="MOMENTUM_BREAKOUT",
    )

    with patch.object(config, "DAYTRADE_BATCH_FETCH_ENABLED", True), \
         patch.object(config, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95), \
         patch("strategies.intraday_data.fetch_intraday_batch", return_value=(data, telemetry)), \
         patch("strategies.orchestrator._evaluate_on_df", return_value=[mock_sig]):

        signals = orch.evaluate_symbols_parallel(symbols, bridge=MagicMock())
        assert len(signals) >= 1
        assert signals[0].symbol == "AAPL"


# =====================================================================
# 4. Confidence Breakdown and Near-Miss Logging Tests
# =====================================================================

def test_confidence_breakdown_metadata_attachment():
    """Scorer attaches breakdown and adjustment factors into signal metadata."""
    sig = TradeSignal(
        symbol="TSLA",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.60,
        entry_price=200.0,
        stop_price=196.0,
        target_price=208.0,
        atr=2.0,
        risk_per_share=4.0,
        reason="VWAP_TOUCH",
    )

    scorer = RiskAdjustedSignalScorer()
    adjusted_conf = scorer.calculate_risk_adjusted_confidence(sig)

    assert sig.metadata is not None
    assert "base_confidence" in sig.metadata
    assert "vol_adjustment" in sig.metadata
    assert "rr_adjustment" in sig.metadata
    assert isinstance(sig.metadata["confidence_breakdown"], dict)
    assert "base_confidence" in sig.metadata["confidence_breakdown"]
    assert math.isclose(sig.metadata["base_confidence"], 0.60)


def test_near_miss_logging_triggered(caplog):
    """When confidence is just under min_conf (by <= 0.005), [NEAR-MISS] log is emitted."""
    orch._last_evaluated_5m_bar_str = "prev_bar"

    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.598,  # min_conf is 0.60 -> gap = 0.002 <= 0.005
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        atr=1.0,
        risk_per_share=2.0,
        reason="VWAP_TOUCH",
        metadata={
            "base_confidence": 0.60,
            "vol_adjustment": 0.9967,
            "rr_adjustment": 1.0,
            "regime_adjustment": 1.0,
            "volatility_pct": 0.008,
        }
    )

    now_utc = datetime(2026, 9, 11, 18, 0, 10, tzinfo=timezone.utc)

    from zoneinfo import ZoneInfo
    now_et = now_utc.astimezone(ZoneInfo("America/New_York"))

    with patch("strategies.orchestrator.datetime") as mock_dt, \
         patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=[sig]), \
         patch("strategies.session.is_market_open", return_value=True), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.session.now_eastern", return_value=now_et), \
         patch("strategies.orchestrator._min_confidence_for", return_value=0.60):
        mock_dt.now.return_value = now_utc
        mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)

        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.get_net_liquidation.return_value = 100000.0
        mock_bridge.get_open_positions.return_value = []
        mock_bridge.get_working_orders.return_value = []

        import logging
        with caplog.at_level(logging.INFO):
            orch.evaluate_and_execute(["NVDA"], bridge=mock_bridge)

        assert any("[NEAR-MISS]" in record.message for record in caplog.records)
        assert any("NVDA" in record.message for record in caplog.records)


# =====================================================================
# 5. Chase Guard Telemetry Tests
# =====================================================================

def test_chase_guard_telemetry_in_prepare_execution_prices():
    """Chase guard veto includes quote_age_ms, stop_dist, and chase_to_stop_ratio_pct."""
    sig = TradeSignal(
        symbol="AMD",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=100.0,
        stop_price=98.0,  # stop_dist = 2.0
        target_price=104.0,
        atr=1.0,
        risk_per_share=2.0,
        reason="MOMENTUM_BREAKOUT",
    )

    quote = BrokerQuote(
        symbol="AMD",
        bid=100.20,
        ask=100.30,  # exec_price = 100.30 -> unf_chase = 0.30 > max 0.15
        last=100.25,
        timestamp=time.time() - 0.5,  # 500ms age
    )
    quote_res = QuoteValidationResult(valid=True, reason="OK", quote=quote)

    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    mock_bridge.has_position.return_value = False
    mock_bridge.has_working_order.return_value = False
    mock_bridge.open_position_count.return_value = 0
    mock_bridge.get_open_positions.return_value = []
    mock_bridge.get_working_orders.return_value = []
    mock_bridge.get_net_liquidation.return_value = 100000.0

    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="test_cycle",
        bar_ts="2026-09-11T10:00:00+00:00",
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )

    with patch("strategies.order_manager.pre_trade_check", return_value=None), \
         patch("strategies.execution_quote.validate_execution_quote", return_value=quote_res), \
         patch.object(config, "SPY_TREND_GUARD_ENABLED", False), \
         patch.object(config, "MAX_ALLOWED_CHASE", 0.15):

        res = om.execute_signal(sig, bridge=mock_bridge, equity=100000.0, current_pnl=0.0, dry_run=False, entry_readiness=readiness)
        assert res["status"] == "VETOED"
        assert "Unfavorable chase $0.3000 exceeds safety guard $0.1500" in res["reason"]
        assert "chase/stop=15.0%" in res["reason"]
        assert "quote_age=" in res["reason"]


def test_chase_guard_telemetry_in_execute_signal():
    """execute_signal rejects orders where chase exceeds maximum allowed and includes chase/stop ratio."""
    sig = TradeSignal(
        symbol="INTC",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.70,
        entry_price=30.0,
        stop_price=29.0,  # stop_dist = 1.0
        target_price=35.0,  # R:R when chased to 30.25 is (35 - 30.25) / (30.25 - 29) = 3.8 > 1.5
        atr=0.5,
        risk_per_share=1.0,
        reason="MOMENTUM_BREAKOUT",
    )

    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="test_cycle",
        bar_ts="2026-09-11T10:00:00+00:00",
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )

    mock_bridge = MagicMock()
    mock_bridge.get_net_liquidation.return_value = 100000.0

    with patch.object(config, "MAX_ALLOWED_CHASE", 0.15):
        res = om.execute_signal(sig, bridge=mock_bridge, equity=100000.0, current_pnl=0.0, entry_limit_price=30.25, entry_readiness=readiness)
        assert res["status"] == "REJECTED"
        assert "REJECTED_MAX_CHASE_EXCEEDED" in res["reason"]
        assert "chase/stop=25.0%" in res["reason"]
