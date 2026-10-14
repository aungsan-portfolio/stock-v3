"""
tests/test_swing_rotation_integration.py — Integration Test Suite for ATOMIC-ROTATION-V1.

Verifies end-to-end integration through the real production caller:
  • Real caller test: AlpacaBridge._evaluate_and_execute_rotation() invokes AtomicRotationEngine
  • Cycle #8 regression replay: Verbatim production inputs (AMZN +1.39%, JPM BUY 0.62, Daily Loss -$328.85 <= -$300 limit)
    proves AMZN is preserved, 0 orders placed, and NO false positive success log is emitted.
  • Partial fill replay: Order partially fills -> DEGRADED, no false success.
  • Delta mismatch replay: Position delta mismatch -> DEGRADED.
  • Timeout replay: Unconfirmed pending order -> DEGRADED, no blind retry.
"""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from alpaca_bridge import AlpacaBridge
from position_rotator import IncomingSignal, OpenPosition, PositionRotator, RotationDecision
from predictor import Signal
from strategies.atomic_rotation import (
    AtomicRotationEngine,
    BrokerEvidenceBundle,
    BrokerStopState,
    OrderSnapshot,
    PositionSnapshot,
    ReplacementTruthStatus,
    RotationState,
)


@dataclass
class MockRawPosition:
    symbol: str
    side: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float


def _build_test_bridge():
    """Create an AlpacaBridge in swing mode with mocked trading client and network calls."""
    bridge = AlpacaBridge(mode="swing", dry_run=False)
    bridge._client = MagicMock()
    bridge._connected = True
    bridge.dry_run = False
    bridge.rotator = PositionRotator()
    bridge.atomic_rotator = AtomicRotationEngine()
    bridge._start_of_day_equity = 100288.95
    bridge.get_account_balance = MagicMock(return_value=99974.66)
    bridge.validate_quote = MagicMock(return_value=(True, 215.50, "OK"))
    bridge.working_order_symbols = MagicMock(return_value=set())
    bridge.close_position = MagicMock()
    bridge._cancel_working_orders_for_symbol = MagicMock()
    bridge._ensure_protective_stops = MagicMock()
    bridge.get_current_stop = MagicMock(return_value=244.02)
    entry_res = MagicMock()
    entry_res.requested_qty = 8.0
    entry_res.filled = 8.0
    entry_res.filled_qty = 8.0
    entry_res.qty = 8.0
    entry_res.order_id = "jpm_entry_order_1"
    bridge.execute_signal_with_client_id = MagicMock(return_value=entry_res)
    bridge._client.get_all_positions = MagicMock(return_value=[
        MockRawPosition(
            symbol="AMZN",
            side="long",
            qty=8.0,
            avg_entry_price=251.57,
            current_price=255.07,
            unrealized_pl=28.00,
        ),
        MockRawPosition(
            symbol="MSFT",
            side="long",
            qty=12.0,
            avg_entry_price=410.00,
            current_price=412.00,
            unrealized_pl=24.00,
        ),
    ])
    return bridge


def _make_cycle_8_open_positions():
    """Verbatim open positions snapshot from September 22, 2026 Cycle #8."""
    return [
        OpenPosition(
            symbol="AMZN",
            side="long",
            qty=8.0,
            avg_entry_price=251.57,
            current_price=255.07,  # +1.39%
            unrealized_pnl=28.00,
            current_conf=0.43,
            unrealized_r=0.35,
        ),
        OpenPosition(
            symbol="MSFT",
            side="long",
            qty=12.0,
            avg_entry_price=410.00,
            current_price=412.00,
            unrealized_pnl=24.00,
            current_conf=0.56,
            unrealized_r=0.80,
        ),
    ]


def _make_cycle_8_signals():
    """Incoming signals for Cycle #8: JPM BUY 0.62, AMZN rescored HOLD 0.43, XOM HOLD 0.45."""
    return [
        Signal(
            symbol="JPM",
            action="BUY",
            confidence=0.62,
            price=215.50,
            rf_score=0.65,
            lstm_score=0.60,
            xgb_score=0.62,
            trans_score=0.61,
            tech_score=0.62,
            reason="ML Top Conviction",
        ),
        Signal(
            symbol="AMZN",
            action="HOLD",
            confidence=0.43,
            price=255.07,
            rf_score=0.43,
            lstm_score=0.43,
            xgb_score=0.43,
            trans_score=0.43,
            tech_score=0.43,
            reason="Rescored weak conviction",
        ),
        Signal(
            symbol="XOM",
            action="HOLD",
            confidence=0.45,
            price=110.00,
            rf_score=0.45,
            lstm_score=0.45,
            xgb_score=0.45,
            trans_score=0.45,
            tech_score=0.45,
            reason="Hold",
        ),
    ]


# =============================================================================
#  INTEGRATION TEST 1: Real Caller Production Path
# =============================================================================

def test_real_caller_production_path(caplog):
    """
    Test real production caller: AlpacaBridge._evaluate_and_execute_rotation().
    Ensures that when rotation is evaluated, it delegates execution to
    AtomicRotationEngine.rotate() and records completion only upon full confirmation.
    """
    bridge = _build_test_bridge()
    bridge.get_open_positions = MagicMock(return_value=_make_cycle_8_open_positions())
    bridge._is_daily_loss_halted = MagicMock(return_value=False)

    # Setup successful order entry and evidence bundle
    entry_res = MagicMock()
    entry_res.requested_qty = 8.0
    entry_res.filled = 8.0
    entry_res.filled_qty = 8.0
    entry_res.qty = 8.0
    entry_res.order_id = "jpm_entry_order_1"
    bridge.execute_signal_with_client_id = MagicMock(return_value=entry_res)

    def _gather_evidence(epoch, incumbent_symbol, replacement_symbol, client_order_id, pre_position_snapshot):
        return BrokerEvidenceBundle(
            reconciliation_epoch=epoch,
            captured_at=datetime.now(timezone.utc).isoformat(),
            incumbent_symbol=incumbent_symbol,
            replacement_symbol=replacement_symbol,
            close_order=OrderSnapshot(
                order_id="amzn_close_1", client_order_id="close_amzn", symbol="AMZN",
                side="SELL", order_type="MARKET", qty=8.0, filled_qty=8.0, status="FILLED",
            ),
            entry_order=OrderSnapshot(
                order_id="jpm_entry_order_1", client_order_id=client_order_id, symbol="JPM",
                side="BUY", order_type="LIMIT", qty=8.0, filled_qty=8.0, status="FILLED",
            ),
            incumbent_position=None,
            replacement_position=PositionSnapshot(
                symbol="JPM", qty=8.0, avg_entry_price=215.50, current_price=215.50, unrealized_pl=0.0
            ),
            replacement_position_pre=None,
            protection_order=OrderSnapshot(
                order_id="jpm_stop_1", client_order_id="stop_jpm", symbol="JPM",
                side="SELL", order_type="STOP", qty=8.0, filled_qty=0.0, status="NEW", stop_price=205.0,
            ),
            protection_state=BrokerStopState.ACTIVE,
            stale_threshold_seconds=999999.0,
        )

    bridge.gather_evidence_bundle = MagicMock(side_effect=_gather_evidence)

    with caplog.at_level(logging.INFO):
        bridge._evaluate_and_execute_rotation(_make_cycle_8_signals())

    # Assertions
    bridge.close_position.assert_called_once_with("AMZN")
    bridge.execute_signal_with_client_id.assert_called_once()
    assert bridge.rotator.state.rotations_this_cycle == 1
    assert len(bridge.rotator.state.rotation_history) == 1
    assert "Closed AMZN ✅" in caplog.text
    assert "Replaced AMZN with JPM ✅" in caplog.text


# =============================================================================
#  INTEGRATION TEST 2: Cycle #8 Regression Verbatim Replay
# =============================================================================

def test_cycle_8_regression_verbatim_replay(caplog):
    """
    Verbatim replay of Cycle #8:
      • Incumbent: AMZN (8 shs, +1.39% profit, conf 0.43)
      • Incoming: JPM (BUY conf 0.62)
      • Daily Loss: -$328.85 <= -$300 limit (_is_daily_loss_halted() == True)

    Expected Production Behavior:
      • Pre-check immediately vetoes rotation BEFORE touching AMZN.
      • AMZN close_position is NEVER called (0 shares liquidated).
      • AMZN working stop cancel is NEVER called.
      • JPM execute_signal_with_client_id is NEVER called (0 orders submitted).
      • "Entered JPM ✅" is DEFINITIVELY ABSENT from logs.
      • Rotator state records 0 rotations this cycle.
    """
    bridge = _build_test_bridge()
    bridge.get_open_positions = MagicMock(return_value=_make_cycle_8_open_positions())
    # Daily loss circuit breaker tripped: -$328.85 <= -$300
    bridge._is_daily_loss_halted = MagicMock(return_value=True)

    with caplog.at_level(logging.DEBUG):
        bridge._evaluate_and_execute_rotation(_make_cycle_8_signals())

    # Strict assertions
    bridge.close_position.assert_not_called()
    bridge._cancel_working_orders_for_symbol.assert_not_called()
    bridge.execute_signal_with_client_id.assert_not_called()
    assert bridge.rotator.state.rotations_this_cycle == 0
    assert len(bridge.rotator.state.rotation_history) == 0

    # Forensic log audit: NO false positive entry log
    assert "Entered JPM ✅" not in caplog.text
    assert "Replaced AMZN with JPM ✅" not in caplog.text
    assert "ROTATION PRE-CHECK VETO" in caplog.text
    assert "Daily loss circuit breaker is ACTIVE" in caplog.text


# =============================================================================
#  INTEGRATION TEST 3: Partial Fill Replay
# =============================================================================

def test_integration_partial_fill_replay(caplog):
    """
    Test partial fill scenario:
      • AMZN closed cleanly.
      • JPM order fills 4 of 8 shares.
      • Expected: Transitions RECONCILING -> DEGRADED, rotation_completed=False,
        NO success logged, rotation NOT recorded in rotator state.
    """
    bridge = _build_test_bridge()
    bridge.get_open_positions = MagicMock(return_value=_make_cycle_8_open_positions())
    bridge._is_daily_loss_halted = MagicMock(return_value=False)

    entry_res = MagicMock()
    entry_res.requested_qty = 8.0
    entry_res.filled = 4.0
    entry_res.filled_qty = 4.0
    entry_res.qty = 8.0
    bridge.execute_signal_with_client_id = MagicMock(return_value=entry_res)

    def _gather_evidence(epoch, incumbent_symbol, replacement_symbol, client_order_id, pre_position_snapshot):
        return BrokerEvidenceBundle(
            reconciliation_epoch=epoch,
            captured_at=datetime.now(timezone.utc).isoformat(),
            incumbent_symbol=incumbent_symbol,
            replacement_symbol=replacement_symbol,
            close_order=OrderSnapshot(
                order_id="amzn_close_1", client_order_id="close_amzn", symbol="AMZN",
                side="SELL", order_type="MARKET", qty=8.0, filled_qty=8.0, status="FILLED",
            ),
            entry_order=OrderSnapshot(
                order_id="jpm_entry_order_1", client_order_id=client_order_id, symbol="JPM",
                side="BUY", order_type="LIMIT", qty=8.0, filled_qty=4.0, status="PARTIALLY_FILLED",
            ),
            incumbent_position=None,
            replacement_position=PositionSnapshot(
                symbol="JPM", qty=4.0, avg_entry_price=215.50, current_price=215.50, unrealized_pl=0.0
            ),
            replacement_position_pre=None,
            protection_order=None,
            protection_state=BrokerStopState.NOT_ACTIVE,
            stale_threshold_seconds=999999.0,
        )

    bridge.gather_evidence_bundle = MagicMock(side_effect=_gather_evidence)

    with caplog.at_level(logging.INFO):
        bridge._evaluate_and_execute_rotation(_make_cycle_8_signals())

    assert bridge.rotator.state.rotations_this_cycle == 0
    assert "Replaced AMZN with JPM ✅" not in caplog.text
    assert "Entered JPM ✅" not in caplog.text
    assert "ROTATION NOT COMPLETED" in caplog.text


# =============================================================================
#  INTEGRATION TEST 4: Position Delta Mismatch Replay
# =============================================================================

def test_integration_delta_mismatch_replay(caplog):
    """
    Test position delta mismatch scenario:
      • AMZN closed, JPM order filled 8 shares.
      • But broker position delta reports only 4 shares.
      • Expected: POSITION_DELTA_MISMATCH -> DEGRADED, rotation_completed=False.
    """
    bridge = _build_test_bridge()
    bridge.get_open_positions = MagicMock(return_value=_make_cycle_8_open_positions())
    bridge._is_daily_loss_halted = MagicMock(return_value=False)

    entry_res = MagicMock()
    entry_res.requested_qty = 8.0
    entry_res.filled = 8.0
    entry_res.filled_qty = 8.0
    entry_res.qty = 8.0
    bridge.execute_signal_with_client_id = MagicMock(return_value=entry_res)

    def _gather_evidence(epoch, incumbent_symbol, replacement_symbol, client_order_id, pre_position_snapshot):
        return BrokerEvidenceBundle(
            reconciliation_epoch=epoch,
            captured_at=datetime.now(timezone.utc).isoformat(),
            incumbent_symbol=incumbent_symbol,
            replacement_symbol=replacement_symbol,
            close_order=OrderSnapshot(
                order_id="amzn_close_1", client_order_id="close_amzn", symbol="AMZN",
                side="SELL", order_type="MARKET", qty=8.0, filled_qty=8.0, status="FILLED",
            ),
            entry_order=OrderSnapshot(
                order_id="jpm_entry_order_1", client_order_id=client_order_id, symbol="JPM",
                side="BUY", order_type="LIMIT", qty=8.0, filled_qty=8.0, status="FILLED",
            ),
            incumbent_position=None,
            replacement_position=PositionSnapshot(
                symbol="JPM", qty=4.0, avg_entry_price=215.50, current_price=215.50, unrealized_pl=0.0
            ),
            replacement_position_pre=None,  # Delta = 4.0 != requested 8.0
            protection_order=OrderSnapshot(
                order_id="jpm_stop_1", client_order_id="stop_jpm", symbol="JPM",
                side="SELL", order_type="STOP", qty=8.0, filled_qty=0.0, status="NEW", stop_price=205.0,
            ),
            protection_state=BrokerStopState.ACTIVE,
            stale_threshold_seconds=999999.0,
        )

    bridge.gather_evidence_bundle = MagicMock(side_effect=_gather_evidence)

    with caplog.at_level(logging.INFO):
        bridge._evaluate_and_execute_rotation(_make_cycle_8_signals())

    assert bridge.rotator.state.rotations_this_cycle == 0
    assert "Replaced AMZN with JPM ✅" not in caplog.text
    assert "Entered JPM ✅" not in caplog.text
    assert "ROTATION VERIFICATION FAILED" in caplog.text


# =============================================================================
#  INTEGRATION TEST 5: Timeout / Ambiguous Order Replay (No Blind Retry)
# =============================================================================

def test_integration_timeout_replay(caplog):
    """
    Test unconfirmed / timeout order scenario:
      • JPM order submitted, but broker order hangs in NEW/pending status.
      • Expected: TIMEOUT_AMBIGUOUS -> DEGRADED, rotation_completed=False.
      • Exactly 1 order placed (no blind retry).
    """
    bridge = _build_test_bridge()
    bridge.get_open_positions = MagicMock(return_value=_make_cycle_8_open_positions())
    bridge._is_daily_loss_halted = MagicMock(return_value=False)

    entry_res = MagicMock()
    entry_res.requested_qty = 8.0
    entry_res.filled = 0.0
    entry_res.filled_qty = 0.0
    entry_res.qty = 8.0
    bridge.execute_signal_with_client_id = MagicMock(return_value=entry_res)

    def _gather_evidence(epoch, incumbent_symbol, replacement_symbol, client_order_id, pre_position_snapshot):
        return BrokerEvidenceBundle(
            reconciliation_epoch=epoch,
            captured_at=datetime.now(timezone.utc).isoformat(),
            incumbent_symbol=incumbent_symbol,
            replacement_symbol=replacement_symbol,
            close_order=OrderSnapshot(
                order_id="amzn_close_1", client_order_id="close_amzn", symbol="AMZN",
                side="SELL", order_type="MARKET", qty=8.0, filled_qty=8.0, status="FILLED",
            ),
            entry_order=OrderSnapshot(
                order_id="jpm_entry_order_1", client_order_id=client_order_id, symbol="JPM",
                side="BUY", order_type="LIMIT", qty=8.0, filled_qty=0.0, status="NEW",
            ),
            incumbent_position=None,
            replacement_position=None,
            replacement_position_pre=None,
            protection_order=None,
            protection_state=BrokerStopState.NOT_ACTIVE,
            stale_threshold_seconds=999999.0,
        )

    bridge.gather_evidence_bundle = MagicMock(side_effect=_gather_evidence)

    with caplog.at_level(logging.INFO):
        bridge._evaluate_and_execute_rotation(_make_cycle_8_signals())

    assert bridge.execute_signal_with_client_id.call_count == 1
    assert bridge.rotator.state.rotations_this_cycle == 0
    assert "Replaced AMZN with JPM ✅" not in caplog.text
    assert "Entered JPM ✅" not in caplog.text
    assert "ROTATION VERIFICATION FAILED" in caplog.text
