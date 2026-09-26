"""
tests/test_rotation_reread_reproduction.py — Verification Suite for Swing Bot Rotation Verifier Hardening (P1-A & P1-B).

Test Matrix:
  1. test_baseline_current_verifier_mismatch:
     Direct verifier & engine fail closed on position delta mismatch when order is filled.
  2. test_transient_position_observation_recovers:
     Transient delay sequence [0, 0, 15] recovers to COMPLETED via bounded reread.
  3. test_earlier_recovery_succeeds:
     Earlier recovery sequence [0, 15] recovers to COMPLETED on initial post-protect query or reread #1.
  4. test_persistent_position_mismatch_fails_closed:
     Negative control [0, 0, 0] strictly fails closed to DEGRADED upon exhausting bounded rereads.
  5. test_immediate_match_completes_without_extra_read_side_effects:
     Normal control [15] completes immediately with 0 extra reread attempts.
  6. test_reread_never_dispatches_duplicate_entry:
     Duplicate side-effect safety: exactly 1 BUY order is ever submitted.
  7. test_reread_does_not_duplicate_protection:
     Duplicate side-effect safety: stop loss is attached exactly once upon fill, never duplicated.
  8. test_no_fill_negative_does_not_trigger_reread:
     Scope constraint: non-FILLED orders (e.g. REJECTED or PARTIAL) do not enter the position reread loop.

Constraints & Governance:
  • ZERO POLICY DRIFT: Sizing, ML cutoffs, ATR multipliers, thresholds 100% frozen.
  • ZERO INFERENCE STANDARD: Verifier's single-observation failure behavior is empirically reproduced.
    The underlying cause of the transient position-observation mismatch remains unconfirmed.
  • ZERO TIME-SLEEP OVERHEAD: Deterministic synthetic sequence generators (no real-time sleep).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

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
    verify_rotation_completion,
)


class SequenceEvidenceGatherer:
    """
    Deterministic synthetic sequence generator for broker evidence bundles.
    Simulates transient or persistent position observation states without real-time sleeps.
    """
    def __init__(
        self,
        position_sequence: List[float],
        requested_qty: float = 15.0,
        filled_qty: float = 15.0,
        order_status: str = "FILLED",
        protection_state: BrokerStopState = BrokerStopState.ACTIVE,
        exit_symbol: str = "COST",
        enter_symbol: str = "AMZN",
        price: float = 248.11,
    ):
        self.position_sequence = list(position_sequence)
        self.requested_qty = requested_qty
        self.filled_qty = filled_qty
        self.order_status = order_status
        self.protection_state = protection_state
        self.exit_symbol = exit_symbol
        self.enter_symbol = enter_symbol
        self.price = price
        self.call_count = 0
        self.history = []

    def __call__(self, epoch, incumbent_symbol, replacement_symbol, client_order_id, pre_position_snapshot):
        self.call_count += 1
        if self.position_sequence:
            current_qty = self.position_sequence.pop(0)
        else:
            current_qty = self.history[-1] if self.history else 0.0
        self.history.append(current_qty)

        entry_snap = OrderSnapshot(
            order_id="ord_rep_amzn_123",
            client_order_id=client_order_id,
            symbol=replacement_symbol,
            side="BUY",
            order_type="LIMIT",
            qty=self.requested_qty,
            filled_qty=self.filled_qty,
            status=self.order_status,
            filled_avg_price=self.price,
        )
        pre_snap = pre_position_snapshot
        post_snap = PositionSnapshot(
            symbol=replacement_symbol,
            qty=current_qty if pre_snap is None else pre_snap.qty + current_qty,
            avg_entry_price=self.price,
            current_price=self.price,
            unrealized_pl=0.0,
        ) if current_qty > 0.0 else None

        prot_snap = OrderSnapshot(
            order_id="stop_rep_amzn_456",
            client_order_id=f"stop_{replacement_symbol}",
            symbol=replacement_symbol,
            side="SELL",
            order_type="STOP",
            qty=self.requested_qty,
            filled_qty=0.0,
            status="NEW",
            stop_price=235.00,
        ) if self.protection_state == BrokerStopState.ACTIVE else None

        fills = (
            ActivityFillRecord(
                activity_id="fill_amzn_1",
                order_id="ord_rep_amzn_123",
                symbol=replacement_symbol,
                qty=self.filled_qty,
                price=self.price,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            ),
        ) if self.filled_qty > 0 else ()

        return BrokerEvidenceBundle(
            reconciliation_epoch=epoch,
            captured_at=datetime.now(timezone.utc).isoformat(),
            incumbent_symbol=incumbent_symbol,
            replacement_symbol=replacement_symbol,
            close_order=OrderSnapshot(
                order_id="ord_close_cost_789",
                client_order_id=f"close_{incumbent_symbol}",
                symbol=incumbent_symbol,
                side="SELL",
                order_type="MARKET",
                qty=4.0,
                filled_qty=4.0,
                status="FILLED",
            ),
            entry_order=entry_snap,
            incumbent_position=None,
            replacement_position=post_snap,
            replacement_position_pre=pre_snap,
            protection_order=prot_snap,
            protection_state=self.protection_state,
            competing_orders=(),
            activity_fills=fills,
            stale_threshold_seconds=60.0,
        )


def _make_reproduction_decision(
    exit_symbol: str = "COST",
    enter_symbol: str = "AMZN",
    exit_qty: float = 4.0,
    exit_price: float = 948.50,
    enter_price: float = 248.11,
) -> RotationDecision:
    exit_pos = OpenPosition(
        symbol=exit_symbol,
        side="long",
        qty=exit_qty,
        avg_entry_price=exit_price,
        current_price=exit_price,
        unrealized_pnl=0.0,
        current_conf=0.45,
        unrealized_r=1.20,
    )
    enter_sig = IncomingSignal(
        symbol=enter_symbol,
        action="BUY",
        confidence=0.75,
        price=enter_price,
    )
    return RotationDecision(
        should_rotate=True,
        exit_position=exit_pos,
        enter_signal=enter_sig,
        reason="Conviction upgrade COST -> AMZN",
    )


def _make_reproduction_signals(symbol: str = "AMZN", price: float = 248.11) -> List[Signal]:
    sig = Signal(
        symbol=symbol,
        action="BUY",
        confidence=0.75,
        price=price,
        rf_score=0.76,
        lstm_score=0.74,
        xgb_score=0.75,
        trans_score=0.75,
        tech_score=0.75,
        reason="ML Conviction Upgrade",
    )
    return [sig]


def _make_reproduction_bridge(
    position_sequence: List[float],
    requested_qty: float = 15.0,
    filled_qty: float = 15.0,
    order_status: str = "FILLED",
    exit_symbol: str = "COST",
    enter_symbol: str = "AMZN",
    price: float = 248.11,
):
    bridge = MagicMock()
    bridge._is_daily_loss_halted.return_value = False
    bridge.validate_quote.return_value = (True, price, "OK")
    bridge.working_order_symbols.return_value = set()
    bridge._cancel_working_orders_for_symbol.return_value = None
    bridge.close_position.return_value = True

    # Pre-entry checks: exit symbol is flat (None -> 0.0), enter symbol is not held (None)
    bridge.get_position.return_value = None

    entry_res = MagicMock()
    entry_res.requested_qty = requested_qty
    entry_res.filled = filled_qty
    entry_res.filled_qty = filled_qty
    entry_res.qty = requested_qty
    entry_res.order_id = "ord_rep_amzn_123"
    bridge.execute_signal_with_client_id.return_value = entry_res

    bridge._ensure_protective_stops.return_value = None

    gatherer = SequenceEvidenceGatherer(
        position_sequence=position_sequence,
        requested_qty=requested_qty,
        filled_qty=filled_qty,
        order_status=order_status,
        protection_state=BrokerStopState.ACTIVE,
        exit_symbol=exit_symbol,
        enter_symbol=enter_symbol,
        price=price,
    )
    bridge.gather_evidence_bundle.side_effect = gatherer

    return bridge, gatherer


# =============================================================================
#  1. Baseline Characterization (Phase 0)
# =============================================================================

def test_baseline_current_verifier_mismatch():
    """
    Phase 0: Baseline Characterization
    Setup: Order FILLED = 15, Position snapshot = 0.
    Direct verifier and engine (with max_verification_rereads=0) must both confirm
    the single-observation mismatch contract: POSITION_DELTA_MISMATCH -> DEGRADED.
    """
    engine = AtomicRotationEngine(max_verification_rereads=0)
    decision = _make_reproduction_decision()
    bridge, gatherer = _make_reproduction_bridge(position_sequence=[0.0, 0.0])
    signals = _make_reproduction_signals()

    # 1. Direct unit verifier contract test
    initial_bundle = gatherer(
        epoch=1,
        incumbent_symbol="COST",
        replacement_symbol="AMZN",
        client_order_id="rot_AMZN_baseline",
        pre_position_snapshot=None,
    )
    direct_verdict = verify_rotation_completion(
        evidence=initial_bundle,
        requested_qty=15.0,
        replacement_symbol="AMZN",
        client_order_id="rot_AMZN_baseline",
    )
    assert direct_verdict.status == ReplacementTruthStatus.POSITION_DELTA_MISMATCH
    assert direct_verdict.fully_confirmed is False
    assert direct_verdict.position_delta == 0.0
    assert "Position delta mismatch" in (direct_verdict.error_message or "")

    # Reset gatherer state for engine run
    gatherer.position_sequence = [0.0, 0.0]
    gatherer.call_count = 0
    gatherer.history.clear()

    # 2. Engine execution contract test without rereads
    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.DEGRADED
    assert summary.completion_verdict is not None
    assert summary.completion_verdict.status == ReplacementTruthStatus.POSITION_DELTA_MISMATCH
    assert "Position delta mismatch" in (summary.error_message or "")
    assert gatherer.call_count == 2


# =============================================================================
#  2. Transient Position-Observation Delay Recovery (Phase A / P1-B)
# =============================================================================

def test_transient_position_observation_recovers():
    """
    Phase A / P1-B: Synthetic transient position-observation delay recovery.
    Sequence: [0, 0, 15] (Query #1=0, Query #2=0, Query #3=15).
    Bounded reread recovers to ROTATION_COMPLETED without side-effects.
    """
    engine = AtomicRotationEngine(max_verification_rereads=3)
    decision = _make_reproduction_decision()
    bridge, gatherer = _make_reproduction_bridge(position_sequence=[0.0, 0.0, 15.0])
    signals = _make_reproduction_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is True
    assert summary.state == RotationState.COMPLETED
    assert summary.completion_verdict is not None
    assert summary.completion_verdict.status == ReplacementTruthStatus.FULLY_CONFIRMED
    assert summary.broker_truth_confirmed is True
    assert summary.protection_truth_confirmed is True
    # Exactly 3 calls: Call 1 (pre-protect=0), Call 2 (post-protect=0), Reread 1 (Call 3=15)
    assert gatherer.call_count == 3


def test_earlier_recovery_succeeds():
    """
    Phase A / P1-B Variant: Earlier recovery sequence [0, 15].
    Query #1 = 0, Query #2 = 15 -> Position becomes available upon post-protect refresh.
    Completes immediately without entering subsequent reread attempts.
    """
    engine = AtomicRotationEngine(max_verification_rereads=3)
    decision = _make_reproduction_decision()
    bridge, gatherer = _make_reproduction_bridge(position_sequence=[0.0, 15.0])
    signals = _make_reproduction_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is True
    assert summary.state == RotationState.COMPLETED
    assert summary.completion_verdict.status == ReplacementTruthStatus.FULLY_CONFIRMED
    assert gatherer.call_count == 2


# =============================================================================
#  3. Genuine Failure Negative Control (Phase B / Fail-Closed Invariant)
# =============================================================================

def test_persistent_position_mismatch_fails_closed():
    """
    Phase B: Genuine failure negative control.
    Sequence: [0, 0, 0, 0, 0].
    Even after exhausting bounded rereads (3 rereads + 1 initial = 4 checks),
    persistent 0 delta MUST strictly fail closed to DEGRADED.
    Retry != lenient pass. Fail-closed invariant is absolute.
    """
    engine = AtomicRotationEngine(max_verification_rereads=3)
    decision = _make_reproduction_decision()
    bridge, gatherer = _make_reproduction_bridge(position_sequence=[0.0, 0.0, 0.0, 0.0, 0.0])
    signals = _make_reproduction_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.DEGRADED
    assert summary.completion_verdict is not None
    assert summary.completion_verdict.status == ReplacementTruthStatus.POSITION_DELTA_MISMATCH
    assert "Position delta mismatch" in (summary.error_message or "")
    # Call 1 (pre-protect) + Call 2 (post-protect) + 3 rereads = 5 calls total
    assert gatherer.call_count == 5


# =============================================================================
#  4. Immediate Match Normal Control (Phase C)
# =============================================================================

def test_immediate_match_completes_without_extra_read_side_effects():
    """
    Phase C: Normal control.
    Sequence: [15].
    When position is immediately observed, rotation completes immediately with
    zero extra reread overhead.
    """
    engine = AtomicRotationEngine(max_verification_rereads=3)
    decision = _make_reproduction_decision()
    bridge, gatherer = _make_reproduction_bridge(position_sequence=[15.0])
    signals = _make_reproduction_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is True
    assert summary.state == RotationState.COMPLETED
    assert summary.completion_verdict is not None
    assert summary.completion_verdict.status == ReplacementTruthStatus.FULLY_CONFIRMED
    assert summary.replacement_entered is True
    assert summary.broker_truth_confirmed is True
    assert summary.protection_truth_confirmed is True
    # Exactly 2 calls: 1 before stop attachment, 1 after stop attachment (0 rereads)
    assert gatherer.call_count == 2


# =============================================================================
#  5. Duplicate Side-Effect Safety Controls (Phase D)
# =============================================================================

def test_reread_never_dispatches_duplicate_entry():
    """
    Phase D: Duplicate side-effect safety under reread simulation.
    Regardless of transient position observation delay, execute_signal_with_client_id
    MUST be called EXACTLY ONCE. No duplicate BUY order is ever dispatched.
    """
    engine = AtomicRotationEngine(max_verification_rereads=3)
    decision = _make_reproduction_decision()
    bridge, gatherer = _make_reproduction_bridge(position_sequence=[0.0, 0.0, 15.0])
    signals = _make_reproduction_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is True
    assert bridge.execute_signal_with_client_id.call_count == 1


def test_reread_does_not_duplicate_protection():
    """
    Phase D: Protection attachment duplicate side-effect safety.
    Stop loss attachment (_ensure_protective_stops) MUST be called exactly once
    upon entry fill, never re-triggered across position reread attempts.
    """
    engine = AtomicRotationEngine(max_verification_rereads=3)
    decision = _make_reproduction_decision()
    bridge, gatherer = _make_reproduction_bridge(position_sequence=[0.0, 0.0, 15.0])
    signals = _make_reproduction_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is True
    assert bridge._ensure_protective_stops.call_count == 1


# =============================================================================
#  6. Non-FILLED Order Scope Constraint (Negative Control)
# =============================================================================

def test_no_fill_negative_does_not_trigger_reread():
    """
    Scope Constraint: Order Non-FILLED Negative Control.
    If replacement order is REJECTED by broker, the bounded reread loop MUST NOT be triggered.
    Rotation transitions to STAY_IN_CASH immediately.
    """
    engine = AtomicRotationEngine(max_verification_rereads=3)
    decision = _make_reproduction_decision()
    bridge, gatherer = _make_reproduction_bridge(
        position_sequence=[0.0, 0.0, 0.0],
        order_status="REJECTED",
        filled_qty=0.0,
    )
    signals = _make_reproduction_signals()

    summary = engine.rotate(decision, bridge, signals)

    assert summary.rotation_completed is False
    assert summary.state == RotationState.STAY_IN_CASH
    # Exactly 1 call (at entry fill check); never enters protection or position reread
    assert gatherer.call_count == 1
    assert bridge._ensure_protective_stops.call_count == 0
