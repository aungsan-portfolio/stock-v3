"""
tests/test_atomic_rotation.py — Unit Test Suite for ATOMIC-ROTATION-V1.

Covers:
  • 17 Scenario Tests
  • 12 Invariant Tests
Strictly enforces single source of truth verification, immutable evidence,
exact position delta equality, and fail-safe state machine transitions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import threading
import time
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alpaca_bridge import AlpacaBridge
from position_rotator import IncomingSignal, OpenPosition, RotationDecision
from predictor import Signal
from strategies.atomic_rotation import (
    ActivityFillRecord,
    AtomicRotationEngine,
    BrokerEvidenceBundle,
    BrokerStopState,
    OrderSnapshot,
    PositionSnapshot,
    ReplacementTruthStatus,
    RotationCompletionResult,
    RotationState,
    VALID_STATE_TRANSITIONS,
    verify_rotation_completion,
)


def _make_decision(
    exit_symbol: str = "AMZN",
    enter_symbol: str = "JPM",
    exit_qty: float = 8.0,
    exit_entry_price: float = 251.57,
    exit_current_price: float = 255.07,
    unrealized_r: float = 1.39,
    enter_conf: float = 0.62,
    enter_price: float = 215.50,
) -> RotationDecision:
    exit_pos = OpenPosition(
        symbol=exit_symbol,
        side="long",
        qty=exit_qty,
        avg_entry_price=exit_entry_price,
        current_price=exit_current_price,
        unrealized_pnl=(exit_current_price - exit_entry_price) * exit_qty,
        current_conf=0.43,
        unrealized_r=unrealized_r,
    )
    enter_sig = IncomingSignal(
        symbol=enter_symbol,
        action="BUY",
        confidence=enter_conf,
        price=enter_price,
    )
    return RotationDecision(
        should_rotate=True,
        exit_position=exit_pos,
        enter_signal=enter_sig,
        reason="Conviction upgrade",
    )


def _make_mock_bridge(
    exit_symbol: str = "AMZN",
    enter_symbol: str = "JPM",
    requested_qty: float = 10.0,
    filled_qty: float = 10.0,
    order_status: str = "FILLED",
    protection_state: BrokerStopState = BrokerStopState.ACTIVE,
    competing_orders: tuple = (),
    position_delta: float = 10.0,
    is_daily_loss_halted: bool = False,
    quote_valid: bool = True,
):
    bridge = MagicMock()
    bridge._is_daily_loss_halted.return_value = is_daily_loss_halted
    bridge.validate_quote.return_value = (quote_valid, 215.50, "OK" if quote_valid else "Spread too wide")
    bridge.working_order_symbols.return_value = set()

    # Pre snapshot
    bridge.get_position.return_value = None

    # Entry result
    entry_res = MagicMock()
    entry_res.requested_qty = requested_qty
    entry_res.filled = filled_qty
    entry_res.filled_qty = filled_qty
    entry_res.qty = requested_qty
    entry_res.order_id = "mock_ord_123"
    bridge.execute_signal_with_client_id.return_value = entry_res

    # Evidence bundle builder
    def _mock_gather_evidence(epoch, incumbent_symbol, replacement_symbol, client_order_id, pre_position_snapshot):
        entry_snap = OrderSnapshot(
            order_id="mock_ord_123",
            client_order_id=client_order_id,
            symbol=replacement_symbol,
            side="BUY",
            order_type="LIMIT",
            qty=requested_qty,
            filled_qty=filled_qty,
            status=order_status,
            filled_avg_price=215.50,
        )
        pre_snap = pre_position_snapshot
        post_snap = PositionSnapshot(
            symbol=replacement_symbol,
            qty=position_delta if pre_snap is None else pre_snap.qty + position_delta,
            avg_entry_price=215.50,
            current_price=215.50,
            unrealized_pl=0.0,
        )
        prot_snap = OrderSnapshot(
            order_id="mock_prot_456",
            client_order_id=f"stop_{replacement_symbol}",
            symbol=replacement_symbol,
            side="SELL",
            order_type="STOP",
            qty=requested_qty,
            filled_qty=0.0,
            status="NEW",
            stop_price=205.00,
        ) if protection_state == BrokerStopState.ACTIVE else None

        fills = (
            ActivityFillRecord(
                activity_id="fill_act_1",
                order_id="mock_ord_123",
                symbol=replacement_symbol,
                qty=filled_qty,
                price=215.50,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ),
        ) if filled_qty > 0 else ()

        return BrokerEvidenceBundle(
            reconciliation_epoch=epoch,
            captured_at=datetime.now(timezone.utc).isoformat(),
            incumbent_symbol=incumbent_symbol,
            replacement_symbol=replacement_symbol,
            close_order=OrderSnapshot(
                order_id="mock_close_789",
                client_order_id=f"close_{incumbent_symbol}",
                symbol=incumbent_symbol,
                side="SELL",
                order_type="MARKET",
                qty=8.0,
                filled_qty=8.0,
                status="FILLED",
            ),
            entry_order=entry_snap,
            incumbent_position=None,
            replacement_position=post_snap,
            replacement_position_pre=pre_snap,
            protection_order=prot_snap,
            protection_state=protection_state,
            competing_orders=competing_orders,
            activity_fills=fills,
            stale_threshold_seconds=60.0,
        )

    bridge.gather_evidence_bundle.side_effect = _mock_gather_evidence
    return bridge


def _make_signals(symbol="JPM", action="BUY", conf=0.62, price=215.50):
    sig = Signal(
        symbol=symbol,
        action=action,
        confidence=conf,
        price=price,
        rf_score=0.65,
        lstm_score=0.60,
        xgb_score=0.62,
        trans_score=0.61,
        tech_score=0.61,
        reason="ML Top Conviction",
    )
    return [sig]


# =============================================================================
#  SECTION 1: 17 SCENARIO TESTS
# =============================================================================

def test_scenario_1_clean_full_fill():
    """Scenario 1: Happy path — full fill, exact delta, stop ACTIVE."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is True
    assert summary.replacement_entered is True
    assert summary.broker_truth_confirmed is True
    assert summary.protection_truth_confirmed is True
    assert summary.state == RotationState.COMPLETED
    assert summary.completion_verdict.status == ReplacementTruthStatus.FULLY_CONFIRMED
    assert engine.state == RotationState.IDLE


def test_scenario_2_cycle_8_circuit_breaker_active():
    """Scenario 2: Cycle #8 regression — Daily loss halted halts pre-check, incumbent untouched."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(is_daily_loss_halted=True)
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.replacement_entered is False
    assert summary.state == RotationState.ABORTED
    bridge.close_position.assert_not_called()
    bridge._cancel_working_orders_for_symbol.assert_not_called()
    bridge.execute_signal_with_client_id.assert_not_called()
    assert engine.state == RotationState.IDLE


def test_scenario_3_partial_fill_transitions_to_degraded():
    """Scenario 3: Partial fill — requested 10, filled 4 -> RECONCILING -> DEGRADED."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(requested_qty=10.0, filled_qty=4.0, position_delta=4.0, order_status="PARTIALLY_FILLED")
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.DEGRADED
    assert "Partial fill" in summary.error_message


def test_scenario_4_position_delta_mismatch_fails():
    """Scenario 4: Order filled 10, but position delta is 5 -> POSITION_DELTA_MISMATCH -> DEGRADED."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(requested_qty=10.0, filled_qty=10.0, position_delta=5.0)
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.DEGRADED
    assert summary.completion_verdict.status == ReplacementTruthStatus.POSITION_DELTA_MISMATCH


def test_scenario_5_competing_order_detected():
    """Scenario 5: Competing active buy order for replacement -> pre-check vetoes."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    bridge.working_order_symbols.return_value = {"JPM"}
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.ABORTED
    assert "Competing working order" in summary.error_message
    bridge.close_position.assert_not_called()


def test_scenario_6_wide_quote_spread_veto():
    """Scenario 6: Quote spread check fails -> pre-check blocks rotation."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(quote_valid=False)
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.ABORTED
    assert "quote spread validation" in summary.error_message
    bridge.close_position.assert_not_called()


def test_scenario_7_ambiguous_broker_timeout():
    """Scenario 7: Replacement order remains in NEW / unconfirmed status -> TIMEOUT_AMBIGUOUS."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(requested_qty=10.0, filled_qty=0.0, position_delta=0.0, order_status="NEW")
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.DEGRADED
    assert summary.completion_verdict.status == ReplacementTruthStatus.TIMEOUT_AMBIGUOUS


def test_scenario_8_winner_preservation_intact():
    """Scenario 8: Incumbent with unrealized_r=1.5 is preserved when replacement entry fails pre-check."""
    engine = AtomicRotationEngine()
    decision = _make_decision(unrealized_r=1.5)
    bridge = _make_mock_bridge(is_daily_loss_halted=True)
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.ABORTED
    bridge.close_position.assert_not_called()


def test_scenario_9_stale_evidence_bundle():
    """Scenario 9: Evidence bundle older than stale threshold -> TIMEOUT_AMBIGUOUS."""
    stale_bundle = BrokerEvidenceBundle(
        reconciliation_epoch=1,
        captured_at="2020-01-01T00:00:00+00:00",
        incumbent_symbol="AMZN",
        replacement_symbol="JPM",
        close_order=None,
        entry_order=None,
        incumbent_position=None,
        replacement_position=None,
        replacement_position_pre=None,
        protection_order=None,
        protection_state=BrokerStopState.ACTIVE,
        stale_threshold_seconds=60.0,
    )
    res = verify_rotation_completion(stale_bundle, requested_qty=10.0, replacement_symbol="JPM", client_order_id="c1")
    assert res.status == ReplacementTruthStatus.TIMEOUT_AMBIGUOUS
    assert "stale" in res.error_message


def test_scenario_10_order_rejected_by_broker():
    """Scenario 10: Replacement order rejected by broker -> STAY_IN_CASH."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(requested_qty=10.0, filled_qty=0.0, position_delta=0.0, order_status="REJECTED")
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.STAY_IN_CASH
    assert engine.state == RotationState.IDLE


def test_scenario_11_missing_entry_order():
    """Scenario 11: Entry order snapshot is missing from evidence bundle -> ORDER_FAILED."""
    evidence = BrokerEvidenceBundle(
        reconciliation_epoch=1,
        captured_at=datetime.now(timezone.utc).isoformat(),
        incumbent_symbol="AMZN",
        replacement_symbol="JPM",
        close_order=None,
        entry_order=None,
        incumbent_position=None,
        replacement_position=None,
        replacement_position_pre=None,
        protection_order=None,
        protection_state=BrokerStopState.ACTIVE,
    )
    res = verify_rotation_completion(evidence, requested_qty=10.0, replacement_symbol="JPM", client_order_id="c1")
    assert res.status == ReplacementTruthStatus.ORDER_FAILED


def test_scenario_12_client_order_id_mismatch():
    """Scenario 12: Order client_order_id does not match expected -> ATTRIBUTION_UNRESOLVED."""
    entry_snap = OrderSnapshot(
        order_id="o1",
        client_order_id="other_client_id",
        symbol="JPM",
        side="BUY",
        order_type="LIMIT",
        qty=10.0,
        filled_qty=10.0,
        status="FILLED",
    )
    evidence = BrokerEvidenceBundle(
        reconciliation_epoch=1,
        captured_at=datetime.now(timezone.utc).isoformat(),
        incumbent_symbol="AMZN",
        replacement_symbol="JPM",
        close_order=None,
        entry_order=entry_snap,
        incumbent_position=None,
        replacement_position=None,
        replacement_position_pre=None,
        protection_order=None,
        protection_state=BrokerStopState.ACTIVE,
    )
    res = verify_rotation_completion(evidence, requested_qty=10.0, replacement_symbol="JPM", client_order_id="expected_id")
    assert res.status == ReplacementTruthStatus.ATTRIBUTION_UNRESOLVED


def test_scenario_13_protection_stop_not_active():
    """Scenario 13: Protection stop is NOT_ACTIVE -> DEGRADED."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(protection_state=BrokerStopState.NOT_ACTIVE)
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.DEGRADED
    assert summary.completion_verdict.status == ReplacementTruthStatus.DEGRADED


def test_scenario_14_protection_stop_quantity_mismatch():
    """Scenario 14: Protection stop order qty does not cover full position -> DEGRADED."""
    entry_snap = OrderSnapshot(
        order_id="o1",
        client_order_id="c1",
        symbol="JPM",
        side="BUY",
        order_type="LIMIT",
        qty=10.0,
        filled_qty=10.0,
        status="FILLED",
    )
    pos_snap = PositionSnapshot(symbol="JPM", qty=10.0, avg_entry_price=200.0, current_price=200.0, unrealized_pl=0.0)
    prot_snap = OrderSnapshot(
        order_id="stop1",
        client_order_id="c_stop",
        symbol="JPM",
        side="SELL",
        order_type="STOP",
        qty=6.0,  # Mismatch: 6 vs 10
        filled_qty=0.0,
        status="NEW",
        stop_price=190.0,
    )
    evidence = BrokerEvidenceBundle(
        reconciliation_epoch=1,
        captured_at=datetime.now(timezone.utc).isoformat(),
        incumbent_symbol="AMZN",
        replacement_symbol="JPM",
        close_order=None,
        entry_order=entry_snap,
        incumbent_position=None,
        replacement_position=pos_snap,
        replacement_position_pre=None,
        protection_order=prot_snap,
        protection_state=BrokerStopState.ACTIVE,
    )
    res = verify_rotation_completion(evidence, requested_qty=10.0, replacement_symbol="JPM", client_order_id="c1")
    assert res.status == ReplacementTruthStatus.DEGRADED
    assert "Protection stop coverage mismatch" in res.error_message


def test_scenario_15_activity_fill_attribution_mismatch():
    """Scenario 15: Activity fills exist but none match entry order_id -> ATTRIBUTION_UNRESOLVED."""
    entry_snap = OrderSnapshot(
        order_id="order_target",
        client_order_id="c1",
        symbol="JPM",
        side="BUY",
        order_type="LIMIT",
        qty=10.0,
        filled_qty=10.0,
        status="FILLED",
    )
    pos_snap = PositionSnapshot(symbol="JPM", qty=10.0, avg_entry_price=200.0, current_price=200.0, unrealized_pl=0.0)
    prot_snap = OrderSnapshot(
        order_id="stop1",
        client_order_id="c_stop",
        symbol="JPM",
        side="SELL",
        order_type="STOP",
        qty=10.0,
        filled_qty=0.0,
        status="NEW",
        stop_price=190.0,
    )
    fills = (
        ActivityFillRecord(
            activity_id="act_alien",
            order_id="order_alien_xyz",
            symbol="JPM",
            qty=10.0,
            price=200.0,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    evidence = BrokerEvidenceBundle(
        reconciliation_epoch=1,
        captured_at=datetime.now(timezone.utc).isoformat(),
        incumbent_symbol="AMZN",
        replacement_symbol="JPM",
        close_order=None,
        entry_order=entry_snap,
        incumbent_position=pos_snap,
        replacement_position=pos_snap,
        replacement_position_pre=None,
        protection_order=prot_snap,
        protection_state=BrokerStopState.ACTIVE,
        activity_fills=fills,
    )
    res = verify_rotation_completion(evidence, requested_qty=10.0, replacement_symbol="JPM", client_order_id="c1")
    assert res.status == ReplacementTruthStatus.ATTRIBUTION_UNRESOLVED


def test_scenario_16_incumbent_close_failure_triggers_reprotection():
    """Scenario 16: Incumbent close fails -> INCUMBENT_REPROTECTING -> ensure_protective_stops -> DEGRADED."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    bridge.close_position.side_effect = RuntimeError("Broker connection timeout on close")
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.DEGRADED
    bridge._ensure_protective_stops.assert_called_once()
    bridge.execute_signal_with_client_id.assert_not_called()


def test_scenario_17_incumbent_stop_cancel_failure():
    """Scenario 17: Incumbent stop cancel raises exception -> RECONCILING -> DEGRADED."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    bridge._cancel_working_orders_for_symbol.side_effect = RuntimeError("Failed to cancel stop leg")
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.DEGRADED
    bridge.close_position.assert_not_called()


# =============================================================================
#  SECTION 2: 12 INVARIANT TESTS
# =============================================================================

def test_invariant_1_no_false_success_event():
    """Invariant 1: Under any failure, rotation_completed is strictly False."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(order_status="REJECTED")
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)
    assert summary.rotation_completed is False
    assert summary.replacement_entered is False


def test_invariant_2_no_false_completed_state():
    """Invariant 2: RotationState.COMPLETED can only be reached if verify_rotation_completion returns FULLY_CONFIRMED."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(position_delta=2.0, requested_qty=10.0)  # Mismatch
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)
    assert summary.state != RotationState.COMPLETED
    assert summary.rotation_completed is False


def test_invariant_3_no_blind_retry():
    """Invariant 3: If order submission fails, no blind retry order is submitted."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    bridge.execute_signal_with_client_id.side_effect = RuntimeError("Broker rejected order")
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)
    assert bridge.execute_signal_with_client_id.call_count == 1
    assert summary.rotation_completed is False
    assert summary.state == RotationState.STAY_IN_CASH


def test_invariant_4_degraded_state_freezes_entries():
    """Invariant 4: DEGRADED state is a terminal state with no outbound transitions."""
    assert RotationState.DEGRADED in VALID_STATE_TRANSITIONS
    assert len(VALID_STATE_TRANSITIONS[RotationState.DEGRADED]) == 0


def test_invariant_5_strict_position_delta_equality():
    """
    Invariant 5: Position delta must STRICTLY EQUAL requested_qty and filled_qty.
    The >= operator is strictly prohibited: if delta > requested, must reject!
    """
    entry_snap = OrderSnapshot(
        order_id="o1", client_order_id="c1", symbol="JPM", side="BUY",
        order_type="LIMIT", qty=10.0, filled_qty=10.0, status="FILLED",
    )
    pos_snap = PositionSnapshot(symbol="JPM", qty=15.0, avg_entry_price=200.0, current_price=200.0, unrealized_pl=0.0)
    prot_snap = OrderSnapshot(
        order_id="stop1", client_order_id="c_stop", symbol="JPM", side="SELL",
        order_type="STOP", qty=10.0, filled_qty=0.0, status="NEW", stop_price=190.0,
    )
    evidence = BrokerEvidenceBundle(
        reconciliation_epoch=1,
        captured_at=datetime.now(timezone.utc).isoformat(),
        incumbent_symbol="AMZN",
        replacement_symbol="JPM",
        close_order=None,
        entry_order=entry_snap,
        incumbent_position=None,
        replacement_position=pos_snap,
        replacement_position_pre=None,  # Delta = 15.0 > 10.0
        protection_order=prot_snap,
        protection_state=BrokerStopState.ACTIVE,
    )
    res = verify_rotation_completion(evidence, requested_qty=10.0, replacement_symbol="JPM", client_order_id="c1")
    assert res.status == ReplacementTruthStatus.POSITION_DELTA_MISMATCH
    assert res.fully_confirmed is False


def test_invariant_6_immutable_results():
    """Invariant 6: Verification and snapshot results are immutable (frozen=True)."""
    res = RotationCompletionResult(
        status=ReplacementTruthStatus.FULLY_CONFIRMED,
        fully_confirmed=True,
        symbol="JPM",
        requested_qty=10.0,
        filled_qty=10.0,
        position_delta=10.0,
        protection_status=BrokerStopState.ACTIVE,
        reconciliation_epoch=1,
    )
    with pytest.raises(FrozenInstanceError):
        res.status = ReplacementTruthStatus.DEGRADED  # type: ignore

    snap = OrderSnapshot(
        order_id="1", client_order_id="c", symbol="JPM", side="BUY",
        order_type="LIMIT", qty=10.0, filled_qty=10.0, status="FILLED"
    )
    with pytest.raises(FrozenInstanceError):
        snap.status = "CANCELLED"  # type: ignore


def test_invariant_7_thread_safety_rlock():
    """Invariant 7: Concurrent rotations are strictly serialized through RLock."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    signals = _make_signals()

    execution_order = []

    def _worker(worker_id):
        s = engine.rotate(decision, bridge, signals)
        execution_order.append((worker_id, s.rotation_completed))

    t1 = threading.Thread(target=_worker, args=(1,))
    t2 = threading.Thread(target=_worker, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(execution_order) == 2
    assert all(comp is True for _, comp in execution_order)


def test_invariant_8_pre_rotation_circuit_breaker_check():
    """Invariant 8: Circuit breaker is checked prior to closing incumbent."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(is_daily_loss_halted=True)
    signals = _make_signals()

    feasible, reason = engine.check_feasibility(decision, bridge, signals)
    assert feasible is False
    assert "Daily loss circuit breaker is ACTIVE" in reason


def test_invariant_9_pre_rotation_quote_spread_check():
    """Invariant 9: Wide quote spread is checked prior to closing incumbent."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge(quote_valid=False)
    signals = _make_signals()

    feasible, reason = engine.check_feasibility(decision, bridge, signals)
    assert feasible is False
    assert "quote spread validation" in reason


def test_invariant_10_pre_rotation_competing_order_check():
    """Invariant 10: Competing working orders are checked prior to closing incumbent."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    bridge.working_order_symbols.return_value = {"JPM"}
    signals = _make_signals()

    feasible, reason = engine.check_feasibility(decision, bridge, signals)
    assert feasible is False
    assert "Competing working order already exists" in reason


def test_invariant_11_stay_in_cash_compensation():
    """Invariant 11: If replacement order submission fails, STAY_IN_CASH is chosen; REOPEN_INCUMBENT is not called."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    bridge.execute_signal_with_client_id.side_effect = RuntimeError("Broker submission timeout")
    signals = _make_signals()

    summary = engine.rotate(decision, bridge, signals)
    assert summary.state == RotationState.STAY_IN_CASH
    assert not hasattr(AlpacaBridge, "reopen_incumbent")
    assert not hasattr(AlpacaBridge, "_reopen_incumbent")
    assert all("reopen" not in str(call).lower() for call in bridge.mock_calls)


def test_invariant_12_clean_orphan_order_removal():
    """Invariant 12: Incumbent working stop order is cancelled before close_position."""
    engine = AtomicRotationEngine()
    decision = _make_decision()
    bridge = _make_mock_bridge()
    signals = _make_signals()

    call_order = []
    bridge._cancel_working_orders_for_symbol.side_effect = lambda sym: call_order.append("cancel")
    bridge.close_position.side_effect = lambda sym: call_order.append("close")

    summary = engine.rotate(decision, bridge, signals)
    assert summary.rotation_completed is True
    assert call_order == ["cancel", "close"]
