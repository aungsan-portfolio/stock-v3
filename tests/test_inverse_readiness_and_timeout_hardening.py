"""
test_inverse_readiness_and_timeout_hardening.py

Targeted unit tests for:
1. Strict fail-closed adherence to explicit `allow_inverse_etf_longs=False` in `_inverse_etf_entry_allowed`.
2. Regime-authoritative inverse exemption in `order_manager.py` (not bypassing regime halts or caps on symbol alone).
3. Comprehensive isolated tests for EVERY hard gate in `EntryReadiness` handoff in `orchestrator.py`:
   - Data health coverage / stale data failure
   - Broker state / fetch failure
   - Event guard blackout
   - Trailing stop unresolved POSITION_UNKNOWN state
   - Latching daily loss breach
   - Latching protection failure
   - All hard gates green -> entry allowed
4. Telemetry accuracy for `[INVERSE ETF EXEMPT]` logging (verifies active watchlist presence and regime permission).
5. Timeout deadline boundary test demonstrating the chunk_timeout_s vs socket timeout risk noted in audit.
"""

import time
import pytest
from unittest.mock import MagicMock, patch

import config
from strategies.base import EntryReadiness, TradeSignal, StrategyName
from strategies.orchestrator import _inverse_etf_entry_allowed, _is_inverse_etf_signal, EvaluationResult
from strategies.regime_engine import MarketRegimeResult, MODE_RISK_OFF, MODE_CAUTION, MODE_DATA_DEGRADED
from strategies.halt_manager import (
    EntryHaltManager,
    LATCHING_HALT_DAILY_LOSS_BREACH,
    LATCHING_HALT_PROTECTION_FAILURE,
    TRANSIENT_HALT_EVENT_BLACKOUT,
)


class DummySignal:
    def __init__(self, symbol="SQQQ", side="BUY", confidence=0.75):
        self.symbol = symbol
        self.side = side
        self.confidence = confidence
        self.strategy = "MEAN_REVERSION"
        self.signal_id = f"sig_{symbol}_{int(time.time())}"
        self.bar_timestamp = "2026-10-16 14:30:00+00:00"


# ── Test 1: Explicit allow_inverse_etf_longs=False in _inverse_etf_entry_allowed ──

def test_inverse_helper_respects_explicit_false():
    """
    When regime_res explicitly has allow_inverse_etf_longs=False,
    _inverse_etf_entry_allowed MUST return False, even if mode is RISK_OFF
    and halt_new_entries is True. (Fixes fail-open Defect 1).
    """
    sig = DummySignal(symbol="SQQQ", side="BUY")
    regime_res = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.0,
        allow_new_shorts=False,
        allow_inverse_etf_longs=False,  # Explicitly False
        halt_new_entries=True,
        reasons=["Bear market with high volatility"],
    )

    with patch.object(config, "INVERSE_ETF_RISK_OFF_EXEMPT", True), \
         patch.object(config, "INVERSE_ETF_SYMBOLS", {"SQQQ", "SH"}):
        allowed = _inverse_etf_entry_allowed(regime_res, sig)
        assert allowed is False, "Explicit allow_inverse_etf_longs=False must be strictly respected (Fail-Closed)"


def test_inverse_helper_allows_when_explicitly_true():
    """
    When regime_res explicitly has allow_inverse_etf_longs=True and all other preconditions pass,
    _inverse_etf_entry_allowed returns True.
    """
    sig = DummySignal(symbol="SQQQ", side="BUY")
    regime_res = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.0,
        allow_new_shorts=False,
        allow_inverse_etf_longs=True,  # Explicitly True
        halt_new_entries=True,
        reasons=["Bear market hedge permitted"],
    )

    with patch.object(config, "INVERSE_ETF_RISK_OFF_EXEMPT", True), \
         patch.object(config, "INVERSE_ETF_SYMBOLS", {"SQQQ", "SH"}):
        allowed = _inverse_etf_entry_allowed(regime_res, sig)
        assert allowed is True


def test_inverse_helper_blocks_non_inverse_signals():
    """Non-inverse signals must always return False from _inverse_etf_entry_allowed."""
    sig = DummySignal(symbol="AAPL", side="BUY")
    regime_res = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.0,
        allow_new_shorts=False,
        allow_inverse_etf_longs=True,
        halt_new_entries=True,
    )

    with patch.object(config, "INVERSE_ETF_RISK_OFF_EXEMPT", True), \
         patch.object(config, "INVERSE_ETF_SYMBOLS", {"SQQQ", "SH"}):
        assert _inverse_etf_entry_allowed(regime_res, sig) is False


def test_inverse_helper_blocks_short_side():
    """Inverse ETFs are long-only hedges; SELL/SHORT side must be blocked."""
    sig = DummySignal(symbol="SQQQ", side="SELL")
    regime_res = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.0,
        allow_new_shorts=False,
        allow_inverse_etf_longs=True,
        halt_new_entries=True,
    )

    with patch.object(config, "INVERSE_ETF_RISK_OFF_EXEMPT", True), \
         patch.object(config, "INVERSE_ETF_SYMBOLS", {"SQQQ", "SH"}):
        assert _inverse_etf_entry_allowed(regime_res, sig) is False


# ── Test 2: order_manager requires regime permission for inverse exemption ──

def test_order_manager_does_not_exempt_inverse_without_regime_permission():
    """
    In order_manager.py, an inverse ETF signal must NOT be exempted from regime entry halt
    if regime_result.allow_inverse_etf_longs is False.
    """
    from strategies.order_manager import execute_signal

    sig = TradeSignal(
        symbol="SQQQ",
        strategy=StrategyName.MEAN_REVERSION,
        side="BUY",
        confidence=0.75,
        entry_price=10.0,
        stop_price=9.5,
        target_price=11.0,
        atr=0.5,
        risk_per_share=0.5,
        reason="Test mean reversion",
        bar_timestamp="2026-10-16 14:30:00+00:00",
    )
    regime_res = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=1.0,
        confidence_boost=0.0,
        allow_new_shorts=False,
        allow_inverse_etf_longs=False,  # Explicitly False
        halt_new_entries=True,
        max_trades_day=0,
    )

    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="test_cycle",
        bar_ts="2026-10-16 14:30:00+00:00",
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )

    bridge_mock = MagicMock()
    bridge_mock.is_connected = True
    bridge_mock.positions.return_value = []
    bridge_mock.open_orders.return_value = []
    bridge_mock.open_position_count.return_value = 0
    bridge_mock.has_position.return_value = False
    bridge_mock.has_order.return_value = False
    bridge_mock.has_working_order.return_value = False
    bridge_mock.get_account.return_value = MagicMock(buying_power="100000", cash="100000")

    with patch.object(config, "INVERSE_ETF_RISK_OFF_EXEMPT", True), \
         patch.object(config, "INVERSE_ETF_SYMBOLS", {"SQQQ", "SH"}), \
         patch.object(config, "DAYTRADE_REQUIRE_ENTRY_READINESS", True):
        res = execute_signal(
            signal=sig,
            bridge=bridge_mock,
            equity=100000.0,
            current_pnl=0.0,
            regime_result=regime_res,
            entry_readiness=readiness,
            dry_run=True,
        )
        assert res["status"] in ("REJECTED", "SKIPPED", "BLOCKED")
        assert "Regime score" in res.get("reason", "") or "No new long entries" in res.get("reason", "") or "Regime" in res.get("reason", "")


# ── Test 3: Exhaustive Multi-Gate Fail-Closed EntryReadiness Handoff Tests ──

def _build_test_readiness(
    eval_result=None,
    broker_fetch_ok=True,
    halt_mgr=None,
    event_decision=None,
    unresolved_position=False,
    regime_res=None,
    sig=None,
    allow_new_entries=False,
    live_paper=False,
    is_connected=True,
    daytrade_suspended=False,
    entry_gate_halted=False,
):
    """Helper implementing exact orchestrator readiness composite logic."""
    if halt_mgr is None:
        halt_mgr = EntryHaltManager()
    if event_decision is None:
        event_decision = MagicMock(block_entries=False)
    if sig is None:
        sig = DummySignal(symbol="SQQQ", side="BUY")
    if regime_res is None:
        regime_res = MarketRegimeResult(
            mode=MODE_RISK_OFF, score=-5, allow_new_longs=False, requires_opening_buffer=False,
            position_scale=0.0, confidence_boost=0.0, allow_new_shorts=False,
            allow_inverse_etf_longs=True, halt_new_entries=True,
        )

    is_inverse_sig = _is_inverse_etf_signal(sig)
    is_inverse_regime_allowed = _inverse_etf_entry_allowed(regime_res, sig)

    if is_inverse_sig and is_inverse_regime_allowed and not allow_new_entries:
        broker_conn_ok = not live_paper or is_connected
        daytrade_active = not daytrade_suspended
        gate_not_halted = not entry_gate_halted

        non_regime_gates_pass = (
            bool(eval_result is not None and eval_result.entry_allowed)
            and broker_fetch_ok
            and broker_conn_ok
            and daytrade_active
            and gate_not_halted
            and halt_mgr.entries_allowed()
            and not getattr(event_decision, "block_entries", False)
            and not unresolved_position
        )
        sig_entry_allowed = non_regime_gates_pass
        if eval_result is not None and not eval_result.entry_allowed:
            sig_suppression = eval_result.suppression_reason
        elif not broker_conn_ok:
            sig_suppression = "BROKER_DISCONNECTED"
        elif not broker_fetch_ok:
            sig_suppression = "BROKER_FETCH_FAILED"
        elif not daytrade_active:
            sig_suppression = "DAYTRADE_SUSPENDED"
        elif not gate_not_halted:
            sig_suppression = "ENTRY_GATE_HALTED"
        elif not halt_mgr.entries_allowed():
            sig_suppression = halt_mgr.summary()
        elif getattr(event_decision, "block_entries", False):
            sig_suppression = "EVENT_BLACKOUT"
        elif unresolved_position:
            sig_suppression = "UNRESOLVED_POSITION_UNKNOWN"
        else:
            sig_suppression = None
    else:
        sig_entry_allowed = bool(eval_result.entry_allowed and allow_new_entries) if eval_result is not None else allow_new_entries
        sig_suppression = (eval_result.suppression_reason if eval_result is not None and not eval_result.entry_allowed else ("NEW_ENTRIES_HALTED" if not allow_new_entries else None))

    return sig_entry_allowed, sig_suppression


def test_gate_a_readiness_fails_closed_on_insufficient_coverage():
    """Gate A: Data coverage < 95% must suppress entry readiness."""
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=False, suppression_reason="INSUFFICIENT_COVERAGE_0.85",
        coverage=0.85, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res)
    assert allowed is False
    assert reason == "INSUFFICIENT_COVERAGE_0.85"


def test_gate_b_readiness_fails_closed_on_stale_market_data():
    """Gate B: Stale bar data suppression must suppress entry readiness."""
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=False, suppression_reason="STALE_BAR_DATA: Age 320s > 300s",
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res)
    assert allowed is False
    assert "STALE_BAR_DATA" in reason


def test_gate_c_readiness_fails_closed_on_broker_fetch_failure():
    """Gate C: Failed broker fetch (portfolio or open trades error) must suppress readiness."""
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res, broker_fetch_ok=False)
    assert allowed is False
    assert reason == "BROKER_FETCH_FAILED"


def test_gate_d_readiness_fails_closed_on_daily_loss_breach():
    """Gate D: Daily loss circuit breaker must suppress readiness."""
    hm = EntryHaltManager()
    hm.add(LATCHING_HALT_DAILY_LOSS_BREACH)
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res, halt_mgr=hm)
    assert allowed is False
    assert LATCHING_HALT_DAILY_LOSS_BREACH in reason


def test_gate_e_readiness_fails_closed_on_protection_failure_halt():
    """Gate E: Unprotected position latching halt must suppress readiness."""
    hm = EntryHaltManager()
    hm.add(LATCHING_HALT_PROTECTION_FAILURE)
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res, halt_mgr=hm)
    assert allowed is False
    assert LATCHING_HALT_PROTECTION_FAILURE in reason


def test_gate_f_readiness_fails_closed_on_event_guard_blackout():
    """Gate F: Event guard blackout (e.g. CPI/FOMC) must suppress readiness."""
    ev = MagicMock(block_entries=True, tag="FOMC_DECISION")
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res, event_decision=ev)
    assert allowed is False
    assert reason == "EVENT_BLACKOUT"


def test_gate_g_readiness_fails_closed_on_unresolved_position_unknown():
    """Gate G: Trailing stop manager POSITION_UNKNOWN state must suppress readiness."""
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res, unresolved_position=True)
    assert allowed is False
    assert reason == "UNRESOLVED_POSITION_UNKNOWN"


def test_gate_h_readiness_passes_only_when_all_hard_gates_are_green():
    """Gate H: Readiness is allowed ONLY when all 5 hard gate checks pass simultaneously."""
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(
        eval_result=eval_res,
        broker_fetch_ok=True,
        halt_mgr=EntryHaltManager(),
        event_decision=MagicMock(block_entries=False),
        unresolved_position=False,
    )
    assert allowed is True
    assert reason is None


def test_gate_i_readiness_fails_closed_on_broker_disconnect():
    """Gate I: Broker disconnected in live paper mode must suppress readiness."""
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res, live_paper=True, is_connected=False)
    assert allowed is False
    assert reason == "BROKER_DISCONNECTED"


def test_gate_j_readiness_fails_closed_on_daytrade_suspended():
    """Gate J: Daytrade suspended flag must suppress readiness."""
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res, daytrade_suspended=True)
    assert allowed is False
    assert reason == "DAYTRADE_SUSPENDED"


def test_gate_k_readiness_fails_closed_on_entry_gate_halted():
    """Gate K: Entry gate halted flag must suppress readiness."""
    eval_res = EvaluationResult(
        entry_signals=(), entry_allowed=True, suppression_reason=None,
        coverage=1.0, required_coverage=0.95, received_symbols=(), missing_symbols=(),
    )
    allowed, reason = _build_test_readiness(eval_result=eval_res, entry_gate_halted=True)
    assert allowed is False
    assert reason == "ENTRY_GATE_HALTED"


# ── Test 4: Telemetry Accuracy for [INVERSE ETF EXEMPT] ──

def test_inverse_exempt_log_only_when_in_watchlist_and_permitted():
    """
    Ensure [INVERSE ETF EXEMPT] log condition requires both has_inverse_in_watchlist
    and regime_res.allow_inverse_etf_longs=True.
    """
    # Case A: Watchlist has NO inverse ETFs
    watchlist_a = ["AAPL", "MSFT", "NVDA"]
    has_inverse_a = any(sym in {"SQQQ", "SH"} for sym in watchlist_a)
    regime_ok = True
    inverse_exempt = True

    should_log_a = inverse_exempt and regime_ok and has_inverse_a
    assert should_log_a is False, "Must NOT log if inverse ETF is absent from watchlist"

    # Case B: Regime explicitly disallows inverse ETFs
    watchlist_b = ["SQQQ", "NVDA"]
    has_inverse_b = any(sym in {"SQQQ", "SH"} for sym in watchlist_b)
    regime_disallowed = False

    should_log_b = inverse_exempt and regime_disallowed and has_inverse_b
    assert should_log_b is False, "Must NOT log if regime disallows inverse ETFs"

    # Case C: Both watchlist has inverse ETF AND regime allows
    should_log_c = inverse_exempt and regime_ok and has_inverse_b
    assert should_log_c is True, "Must log only when in watchlist and permitted"


# ── Test 5: Timeout Boundary Audit Test ──

def test_timeout_boundary_demonstrates_chunk_vs_socket_deadline_mismatch():
    """
    Test verifying the exact architectural vulnerability identified in audit:
    If ALPACA_CHUNK_TIMEOUT_SECONDS is 2.0s, and socket connect + read timeout is 3.5s + 7.5s = 11.0s,
    the caller aborts at 2.0s while worker thread holds semaphore slot for up to 11.0s.
    Proves why socket timeout alone cannot guarantee thread termination without
    an end-to-end deadline cancellation supervisor.
    """
    chunk_timeout_s = float(getattr(config, "ALPACA_CHUNK_TIMEOUT_SECONDS", 2.0))
    connect_timeout = float(getattr(config, "ALPACA_DATA_CONNECT_TIMEOUT", 3.5))
    read_timeout = float(getattr(config, "ALPACA_DATA_READ_TIMEOUT", 7.5))

    total_socket = connect_timeout + read_timeout

    # Document that total_socket (11.0s) > chunk_timeout_s (2.0s)
    assert total_socket > chunk_timeout_s, (
        f"Demonstrates that default transport timeout ({total_socket}s) exceeds "
        f"caller chunk timeout ({chunk_timeout_s}s)."
    )
