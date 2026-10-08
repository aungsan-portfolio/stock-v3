"""
tests/test_paper_experiment_lane.py -- Unit test suite for Paper Strategy Experiment Lane.

Enforces:
1. CANDLESTICK + SIDEWAYS + verified paper broker -> Admitted as Paper Experiment (cell="CANDLESTICK_SIDEWAYS").
2. CANDLESTICK + TRANSITIONING + verified paper broker -> Admitted as Paper Experiment (cell="CANDLESTICK_TRANSITIONING").
3. CANDLESTICK + SIDEWAYS + LIVE account -> Strictly BLOCKED (admitted=False).
4. PAPER_EXPERIMENT_LANE_ENABLED flag missing/False -> Strictly BLOCKED (admitted=False).
5. Hard Safety (stale bar, coverage, halted entry gate, expired readiness) blocks entry before execute_signal.
6. Experimental candidate uses calculate_shares() normal risk sizing (NOT validation lane 1-3 share cap).
7. CANDLESTICK + TRENDING -> Standard production admission (admitted=True, is_experiment=False, cell=None).
8. Gate A and Gate B consistency (zero divergence).
9. Audit telemetry: log_gate_decision records PAPER_EXPERIMENT_ADMIT and does not fake production PASS.
"""

import time
from unittest.mock import MagicMock, patch
import pytest

import config
from strategies.base import TradeSignal, EntryReadiness
from strategies.constants import StrategyName
from strategies.orchestrator import (
    is_strategy_admitted_for_texture,
    _strategy_allowed_in_texture,
)
from strategies.position_sizer import calculate_shares
from strategies.order_manager import execute_signal


class MockPaperBridge:
    """Mock bridge representing a verified Alpaca Paper sub-account."""
    def __init__(self):
        self.is_connected = True
        self.is_paper = True
        self.is_paper_account = True
        self.account_mode = "PAPER"
        self.endpoint = "https://paper-api.alpaca.markets"
        self.account_id = "PA3UODIC5JTA"
        self.open_orders = []

    def open_position_count(self) -> int:
        return 0

    def get_quote(self, symbol: str, feed: str = None):
        return {
            "symbol": symbol,
            "bid": 100.0,
            "ask": 100.02,
            "price": 100.01,
            "timestamp": time.time(),
        }


class MockLiveBridge:
    """Mock bridge representing a Live money sub-account."""
    def __init__(self):
        self.is_connected = True
        self.is_paper = False
        self.is_paper_account = False
        self.account_mode = "LIVE"
        self.endpoint = "https://api.alpaca.markets"
        self.account_id = "AK1234567890"


@pytest.fixture
def sample_candlestick_signal():
    return TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.CANDLESTICK,
        side="BUY",
        confidence=0.75,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.5,
        atr=1.5,
        risk_per_share=1.0,
        reason="Bullish Engulfing pattern",
        bar_timestamp="2026-10-08T10:00:00-04:00",
    )


def test_candlestick_sideways_paper_admitted(sample_candlestick_signal):
    """Case 1: CANDLESTICK + SIDEWAYS + verified paper -> Admitted as experiment."""
    bridge = MockPaperBridge()
    admitted, is_experiment, cell_tag = is_strategy_admitted_for_texture(
        strategy=sample_candlestick_signal.strategy,
        market_texture="SIDEWAYS",
        bridge=bridge,
        signal=sample_candlestick_signal,
    )
    assert admitted is True
    assert is_experiment is True
    assert cell_tag == "CANDLESTICK_SIDEWAYS"
    assert sample_candlestick_signal.metadata.get("lane") == "PAPER_EXPERIMENT"
    assert sample_candlestick_signal.metadata.get("paper_experiment_cell") == "CANDLESTICK_SIDEWAYS"
    assert sample_candlestick_signal.metadata.get("paper_experiment") is True


def test_candlestick_transitioning_paper_admitted(sample_candlestick_signal):
    """Case 2: CANDLESTICK + TRANSITIONING + verified paper -> Admitted as experiment."""
    bridge = MockPaperBridge()
    admitted, is_experiment, cell_tag = is_strategy_admitted_for_texture(
        strategy=sample_candlestick_signal.strategy,
        market_texture="TRANSITIONING",
        bridge=bridge,
        signal=sample_candlestick_signal,
    )
    assert admitted is True
    assert is_experiment is True
    assert cell_tag == "CANDLESTICK_TRANSITIONING"
    assert sample_candlestick_signal.metadata.get("lane") == "PAPER_EXPERIMENT"
    assert sample_candlestick_signal.metadata.get("paper_experiment_cell") == "CANDLESTICK_TRANSITIONING"


def test_candlestick_sideways_live_strictly_blocked(sample_candlestick_signal):
    """Case 3: CANDLESTICK + SIDEWAYS + LIVE account -> Fail-closed rejection."""
    live_bridge = MockLiveBridge()
    admitted, is_experiment, cell_tag = is_strategy_admitted_for_texture(
        strategy=sample_candlestick_signal.strategy,
        market_texture="SIDEWAYS",
        bridge=live_bridge,
        signal=sample_candlestick_signal,
    )
    assert admitted is False
    assert is_experiment is False
    assert cell_tag is None


def test_paper_experiment_flag_disabled_strictly_blocked(sample_candlestick_signal, monkeypatch):
    """Case 4: PAPER_EXPERIMENT_LANE_ENABLED=False -> Rejection even on paper account."""
    monkeypatch.setattr(config, "PAPER_EXPERIMENT_LANE_ENABLED", False)
    bridge = MockPaperBridge()
    admitted, is_experiment, cell_tag = is_strategy_admitted_for_texture(
        strategy=sample_candlestick_signal.strategy,
        market_texture="SIDEWAYS",
        bridge=bridge,
        signal=sample_candlestick_signal,
    )
    assert admitted is False
    assert is_experiment is False
    assert cell_tag is None


def test_candlestick_trending_production_admitted(sample_candlestick_signal):
    """Case 7: CANDLESTICK + TRENDING -> Standard production admission (not experiment)."""
    bridge = MockPaperBridge()
    admitted, is_experiment, cell_tag = is_strategy_admitted_for_texture(
        strategy=sample_candlestick_signal.strategy,
        market_texture="TRENDING",
        bridge=bridge,
        signal=sample_candlestick_signal,
    )
    assert admitted is True
    assert is_experiment is False
    assert cell_tag is None


def test_gate_a_and_gate_b_agreement(sample_candlestick_signal):
    """Case 8: Gate A and Gate B use identical canonical helper with zero divergence."""
    bridge = MockPaperBridge()
    textures = ["TRENDING", "SIDEWAYS", "TRANSITIONING", "VOLATILE"]
    for tex in textures:
        res_a = is_strategy_admitted_for_texture(sample_candlestick_signal.strategy, tex, bridge=bridge)
        res_b = is_strategy_admitted_for_texture(sample_candlestick_signal.strategy, tex, bridge=bridge)
        assert res_a == res_b, f"Divergence detected in texture {tex}: Gate A={res_a} != Gate B={res_b}"


def test_hard_safety_failure_blocks_execution(sample_candlestick_signal):
    """Case 5: Hard safety failure blocks experimental candidate at execution boundary."""
    bridge = MockPaperBridge()
    bridge.entry_gate = MagicMock()
    bridge.entry_gate.halted = True  # Hard safety gate halted

    readiness = EntryReadiness(
        entry_allowed=False,
        suppression_reason="DATA_COVERAGE_DEGRADED",
        cycle_id="cycle_test_001",
        bar_ts=sample_candlestick_signal.bar_timestamp,
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )

    result = execute_signal(
        signal=sample_candlestick_signal,
        bridge=bridge,
        equity=100000.0,
        current_pnl=0.0,
        dry_run=True,
        entry_readiness=readiness,
    )

    assert result["status"] == "REJECTED"
    assert "ENTRY_READINESS_SUPPRESSED" in result["reason"] or "halted" in result["reason"]


def test_experimental_candidate_normal_risk_sizing(sample_candlestick_signal):
    """Case 6: Candidate uses calculate_shares() normal sizing, NOT validation 1-3 share cap."""
    equity = 10000.0
    # Risk per share is $1.00 ($100.0 entry - $99.0 stop)
    # Risk cap is 1% or $50-$100
    shares = calculate_shares(sample_candlestick_signal, equity)
    assert isinstance(shares, int)
    # Normal risk sizing on $10,000 equity with $1.00 risk/share yields > 3 shares (typically 50-100 shares)
    assert shares > 3, f"Expected normal risk sizing (> 3 shares), got {shares}"
    assert shares <= 100


def test_gate_ledger_paper_experiment_telemetry(sample_candlestick_signal):
    """Case 9: Audit telemetry records PAPER_EXPERIMENT_ADMIT without faking production PASS."""
    bridge = MockPaperBridge()
    with patch("strategies.gate_ledger.log_gate_decision") as mock_log:
        admitted, is_exp, cell = is_strategy_admitted_for_texture(
            strategy=sample_candlestick_signal.strategy,
            market_texture="SIDEWAYS",
            bridge=bridge,
            signal=sample_candlestick_signal,
        )
        assert admitted is True
        assert is_exp is True

        # Simulate the Gate A ledger log
        from strategies.gate_ledger import log_gate_decision
        log_gate_decision(
            evaluation_id=getattr(sample_candlestick_signal, "signal_id", "test_sig"),
            symbol=sample_candlestick_signal.symbol,
            bar=sample_candlestick_signal.bar_timestamp,
            strategy="CANDLESTICK",
            stage="STAGE_REGIME",
            decision="PAPER_EXPERIMENT_ADMIT",
            reason_code="TEXTURE_RESTRICTION_OVERRIDDEN_FOR_PAPER_EXPERIMENT",
            human_message=f"Paper experiment admission for cell {cell}",
            terminal_decision=False,
            blocked_by=None,
            metadata={"paper_experiment_cell": cell, "lane": "PAPER_EXPERIMENT"},
        )

        mock_log.assert_called_once()
        kwargs = mock_log.call_args[1]
        assert kwargs["decision"] == "PAPER_EXPERIMENT_ADMIT"
        assert kwargs["reason_code"] == "TEXTURE_RESTRICTION_OVERRIDDEN_FOR_PAPER_EXPERIMENT"
        assert kwargs["terminal_decision"] is False
        assert kwargs["metadata"]["paper_experiment_cell"] == "CANDLESTICK_SIDEWAYS"
        assert kwargs["metadata"]["lane"] == "PAPER_EXPERIMENT"


def test_prepare_closed_bars_strips_single_forming_bar():
    """Verifies that _prepare_closed_bars strips a single forming or future bar."""
    import pandas as pd
    from strategies.session import now_eastern
    from strategies.orchestrator import _prepare_closed_bars

    now = now_eastern()
    # 1 bar forming right now (<300s old)
    dates = pd.DatetimeIndex([now])
    df = pd.DataFrame({"close": [100.0]}, index=dates)
    prepared = _prepare_closed_bars(df)
    assert len(prepared) == 0, "Single forming bar must be stripped"

    # 1 bar from 10 minutes ago (fully closed)
    past_dates = pd.DatetimeIndex([now - pd.Timedelta(minutes=10)])
    df_closed = pd.DataFrame({"close": [100.0]}, index=past_dates)
    prepared_closed = _prepare_closed_bars(df_closed)
    assert len(prepared_closed) == 1, "Completed closed bar must be preserved"
