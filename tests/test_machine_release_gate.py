"""
tests/test_machine_release_gate.py
Machine-Enforced Release Gate verification (A-Prime Rule).
Asserts that broker API write call count remains strictly ZERO under:
1. Absent Principal Release
2. Wrong Account Lane
3. Active Safety Blocker (Circuit Breaker)
4. Non-Paper Account Mode
5. Validation Quantity > 1 Share
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from alpaca_bridge import AlpacaBridge, ReleaseGateViolation, AccountModeViolation, ValidationReleaseToken


@pytest.fixture
def gate_bridge():
    bridge = AlpacaBridge(mode="daytrade")
    bridge._connected = True
    bridge.dry_run = False
    bridge.account_id = "PA3UODIC5JTA"
    bridge._client = MagicMock()
    bridge._client._paper = True
    bridge._client._base_url = "https://paper-api.alpaca.markets"
    bridge.assert_account_mode_safety = MagicMock(return_value=True)
    bridge._require_connection = MagicMock(return_value=True)
    bridge._require_trade_stream = MagicMock(return_value=True)
    bridge._require_new_entries_allowed = MagicMock(return_value=True)
    bridge.entry_gate = MagicMock()
    bridge.entry_gate.halted = False
    bridge.principal_release_granted = False
    return bridge


def test_gate_denies_when_release_absent(gate_bridge):
    """When Principal Release is absent, order submission is denied with 0 broker calls."""
    gate_bridge.principal_release_granted = False

    with pytest.raises(ReleaseGateViolation, match="Principal Release Authorization is absent"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=1,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_test_001",
        )

    # CRITICAL INVARIANT: Broker API submit_order must NEVER be called
    assert gate_bridge._client.submit_order.call_count == 0


def test_gate_denies_when_wrong_lane(gate_bridge):
    """When account ID does not match authorized lane, order submission is denied with 0 broker calls."""
    gate_bridge.principal_release_granted = True
    gate_bridge.account_id = "PA_UNAUTHORIZED_LANE"

    with pytest.raises(ReleaseGateViolation, match="Wrong account lane|restricted to account"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=1,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_test_001",
        )

    assert gate_bridge._client.submit_order.call_count == 0


def test_gate_denies_when_safety_blocker_active(gate_bridge):
    """When entry gate / circuit breaker is halted, order submission is denied with 0 broker calls."""
    gate_bridge.principal_release_granted = True
    gate_bridge.entry_gate.halted = True
    gate_bridge.entry_gate.halt_reason = "MAX_DAILY_LOSS_BREACH"

    with pytest.raises(ReleaseGateViolation, match="Safety breaker is halted"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=1,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_test_001",
        )

    assert gate_bridge._client.submit_order.call_count == 0


def test_gate_denies_when_non_paper(gate_bridge):
    """When account mode is live / non-paper, order submission is denied with 0 broker calls."""
    gate_bridge.principal_release_granted = True
    gate_bridge.assert_account_mode_safety = MagicMock(side_effect=AccountModeViolation("Live trading forbidden"))

    with pytest.raises(AccountModeViolation, match="Live trading forbidden"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=1,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_test_001",
        )

    assert gate_bridge._client.submit_order.call_count == 0


def test_gate_denies_when_validation_qty_exceeds_1_share(gate_bridge):
    """When controlled paper validation attempts > 1 share, order submission is denied with 0 broker calls."""
    gate_bridge.principal_release_granted = True

    with pytest.raises(ReleaseGateViolation, match="maximum 1 share"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=5,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_test_001",
        )

    assert gate_bridge._client.submit_order.call_count == 0


def test_gate_admits_when_all_conditions_satisfied(gate_bridge):
    """When all 5 safety prerequisites are strictly met, order is admitted to broker API."""
    gate_bridge.principal_release_granted = True

    mock_submitted_order = MagicMock()
    mock_submitted_order.id = "submitted-order-999"
    gate_bridge._client.submit_order.return_value = mock_submitted_order
    gate_bridge._client.get_order_by_id.return_value = MagicMock(legs=[])

    res = gate_bridge.place_bracket_order(
        symbol="AAPL",
        side="BUY",
        qty=1,
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        client_order_id="val_test_001",
    )

    assert res["order_id"] == "submitted-order-999"
    assert gate_bridge._client.submit_order.call_count == 1


def test_gate_denies_swing_account_for_validation_release(gate_bridge):
    """Swing account PA3R71I7JFDZ must be strictly denied for validation release."""
    gate_bridge.principal_release_granted = True
    gate_bridge.mode = "swing"
    gate_bridge.account_id = "PA3R71I7JFDZ"

    with pytest.raises(ReleaseGateViolation, match="prohibited from validation release"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=1,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_test_001",
        )

    # Broker API submit_order must NEVER be called for validation on Swing account
    assert gate_bridge._client.submit_order.call_count == 0


def test_token_consumed_at_entry_admission_denies_second_entry(gate_bridge):
    """Single-use release token must be consumed at entry admission; second entry denied."""
    token = ValidationReleaseToken(
        release_id="token-single-use-001",
        account_id="PA3UODIC5JTA",
        paper_only=True,
        max_qty=1,
        consumed=False,
    )
    gate_bridge.validation_release_token = token

    mock_order_1 = MagicMock()
    mock_order_1.id = "submitted-order-111"
    gate_bridge._client.submit_order.return_value = mock_order_1
    gate_bridge._client.get_order_by_id.return_value = MagicMock(legs=[])

    # 1. First entry submission succeeds and consumes token
    res1 = gate_bridge.place_bracket_order(
        symbol="AAPL",
        side="BUY",
        qty=1,
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        client_order_id="val_first_entry",
    )
    assert res1["order_id"] == "submitted-order-111"
    assert token.consumed is True
    assert token.consumed_at is not None
    assert token.authorized_client_order_id == "val_first_entry"
    assert token.authorized_parent_order_id == "submitted-order-111"
    assert gate_bridge._client.submit_order.call_count == 1

    # 2. Second independent entry submission MUST be denied fail-closed
    with pytest.raises(ReleaseGateViolation, match="already consumed"):
        gate_bridge.place_bracket_order(
            symbol="MSFT",
            side="BUY",
            qty=1,
            entry_price=400.0,
            stop_price=395.0,
            target_price=410.0,
            client_order_id="val_second_entry",
        )

    # Call count MUST strictly remain 1 (no second order reached broker API)
    assert gate_bridge._client.submit_order.call_count == 1


def test_token_admits_descendant_protective_stop_after_entry_consumed(gate_bridge):
    """Descendant protective actions must remain permitted after entry has consumed token."""
    token = ValidationReleaseToken(
        release_id="token-consumed-001",
        account_id="PA3UODIC5JTA",
        paper_only=True,
        max_qty=1,
        consumed=True,
        consumed_at=datetime.now(timezone.utc),
        authorized_client_order_id="val_first_entry",
    )
    gate_bridge.validation_release_token = token

    mock_stop_order = MagicMock()
    mock_stop_order.id = "protective-stop-222"
    gate_bridge._client.submit_order.return_value = mock_stop_order

    # Descendant protective stop placement is admitted
    stop_id = gate_bridge.place_protective_stop(
        symbol="AAPL",
        side="SELL",
        qty=1,
        stop_price=148.0,
        client_order_id="val_descendant_stop",
    )

    assert stop_id == "protective-stop-222"
    assert gate_bridge._client.submit_order.call_count == 1


def test_token_denies_when_expired(gate_bridge):
    """Expired validation token must deny order submission with 0 broker calls."""
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=15)
    token = ValidationReleaseToken(
        release_id="token-expired-001",
        account_id="PA3UODIC5JTA",
        paper_only=True,
        max_qty=1,
        expires_at=expired_time,
        consumed=False,
    )
    gate_bridge.validation_release_token = token

    with pytest.raises(ReleaseGateViolation, match="expired"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=1,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_test_001",
        )

    assert gate_bridge._client.submit_order.call_count == 0


def test_token_consumed_when_entry_submission_fails_unknown_denies_retry_and_reconciles(gate_bridge):
    """Network failure on entry submission consumes token, forbids retry, and reconciles against broker truth."""
    token = ValidationReleaseToken(
        release_id="token-unknown-001",
        account_id="PA3UODIC5JTA",
        paper_only=True,
        max_qty=1,
        consumed=False,
    )
    gate_bridge.validation_release_token = token

    # Simulate network failure during broker submit
    gate_bridge._client.submit_order.side_effect = TimeoutError("Alpaca gateway timeout")

    with pytest.raises(TimeoutError, match="Alpaca gateway timeout"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=1,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_unknown_entry_001",
        )

    # Invariants: token is consumed, outcome is UNKNOWN
    assert token.consumed is True
    assert token.submission_outcome == "UNKNOWN"
    assert token.authorized_client_order_id == "val_unknown_entry_001"

    # Second entry submission attempt MUST be denied fail-closed (NO RETRY)
    with pytest.raises(ReleaseGateViolation, match="already consumed"):
        gate_bridge.place_bracket_order(
            symbol="AAPL",
            side="BUY",
            qty=1,
            entry_price=150.0,
            stop_price=148.0,
            target_price=154.0,
            client_order_id="val_retry_entry_002",
        )

    # Broker reconciliation: find the order that was placed remotely despite local timeout
    mock_remote_order = MagicMock()
    mock_remote_order.id = "remote-found-999"
    mock_remote_order.client_order_id = "val_unknown_entry_001"
    gate_bridge._client.get_orders.return_value = [mock_remote_order]

    reconciled_id = gate_bridge.reconcile_validation_token_outcome()
    assert reconciled_id == "remote-found-999"
    assert token.authorized_parent_order_id == "remote-found-999"
    assert token.submission_outcome == "RECONCILED"


def test_descendant_action_requires_matching_lifecycle_and_denies_unrelated_order(gate_bridge):
    """Descendant protective orders must match authorized lifecycle symbol and qty; unrelated orders denied."""
    token = ValidationReleaseToken(
        release_id="token-descendant-001",
        account_id="PA3UODIC5JTA",
        paper_only=True,
        max_qty=1,
        consumed=True,
        authorized_symbol="AAPL",
        authorized_qty=1,
        authorized_client_order_id="val_aapl_entry",
    )
    gate_bridge.validation_release_token = token

    # 1. Unrelated order on TSLA must be denied
    with pytest.raises(ReleaseGateViolation, match="Unrelated descendant action on symbol 'TSLA'"):
        gate_bridge.place_protective_stop(
            symbol="TSLA",
            side="SELL",
            qty=1,
            stop_price=240.0,
            client_order_id="val_tsla_unrelated",
        )
    assert gate_bridge._client.submit_order.call_count == 0

    # 2. Excessive quantity on AAPL must be denied
    with pytest.raises(ReleaseGateViolation, match="maximum 1 share|exceeds authorized lifecycle quantity"):
        gate_bridge.place_protective_stop(
            symbol="AAPL",
            side="SELL",
            qty=5,
            stop_price=148.0,
            client_order_id="val_aapl_excessive",
        )
    assert gate_bridge._client.submit_order.call_count == 0

    # 3. Matching descendant action on AAPL qty=1 is admitted
    mock_stop_order = MagicMock()
    mock_stop_order.id = "admitted-stop-888"
    gate_bridge._client.submit_order.return_value = mock_stop_order

    stop_id = gate_bridge.place_protective_stop(
        symbol="AAPL",
        side="SELL",
        qty=1,
        stop_price=148.0,
        client_order_id="val_aapl_matching_stop",
    )
    assert stop_id == "admitted-stop-888"
    assert gate_bridge._client.submit_order.call_count == 1


def test_normal_daytrade_paper_qty_not_blocked_by_validation_max_qty(gate_bridge):
    """Normal Daytrade Paper orders (qty=59) must NOT be blocked by validation 1-share limit."""
    gate_bridge.daytrade_paper_execution_enabled = True
    mock_order = MagicMock(id="normal-order-59")
    gate_bridge._client.submit_order.return_value = mock_order
    gate_bridge._client.get_order_by_id.return_value = MagicMock(legs=[])

    res = gate_bridge.place_bracket_order(
        symbol="NFLX",
        side="BUY",
        qty=59,
        entry_price=67.78,
        stop_price=67.47,
        target_price=68.26,
        client_order_id="dt_normal_001",
    )
    assert res["order_id"] == "normal-order-59"
    assert gate_bridge._client.submit_order.call_count == 1


def test_second_normal_daytrade_paper_order_not_blocked_by_consumed_validation_token(gate_bridge):
    """Normal Daytrade Paper allows multiple trades and is not blocked by consumed validation token."""
    gate_bridge.daytrade_paper_execution_enabled = True
    # Validation token previously consumed by a canary test
    token = ValidationReleaseToken(
        release_id="token-consumed-001",
        account_id="PA3UODIC5JTA",
        paper_only=True,
        max_qty=1,
        consumed=True,
        authorized_symbol="AAPL",
        authorized_qty=1,
    )
    gate_bridge.validation_release_token = token

    mock_order1 = MagicMock(id="normal-1")
    mock_order2 = MagicMock(id="normal-2")
    gate_bridge._client.submit_order.side_effect = [mock_order1, mock_order2]
    gate_bridge._client.get_order_by_id.return_value = MagicMock(legs=[])

    res1 = gate_bridge.place_bracket_order(
        symbol="NFLX",
        side="BUY",
        qty=59,
        entry_price=67.78,
        stop_price=67.47,
        target_price=68.26,
        client_order_id="dt_norm_1",
    )
    res2 = gate_bridge.place_bracket_order(
        symbol="MSFT",
        side="BUY",
        qty=20,
        entry_price=500.0,
        stop_price=495.0,
        target_price=510.0,
        client_order_id="dt_norm_2",
    )
    assert res1["order_id"] == "normal-1"
    assert res2["order_id"] == "normal-2"
    assert gate_bridge._client.submit_order.call_count == 2


def test_missing_paper_daytrade_authorization_blocks_entry(gate_bridge, monkeypatch):
    """Missing Daytrade Paper authorization strictly blocks normal entries."""
    monkeypatch.delenv("DAYTRADE_PAPER_EXECUTION_ENABLED", raising=False)
    gate_bridge.daytrade_paper_execution_enabled = False

    with pytest.raises(ReleaseGateViolation, match="Daytrade Paper execution authorization is absent"):
        gate_bridge.place_bracket_order(
            symbol="NFLX",
            side="BUY",
            qty=59,
            entry_price=67.78,
            stop_price=67.47,
            target_price=68.26,
            client_order_id="dt_normal_unauth",
        )
    assert gate_bridge._client.submit_order.call_count == 0


def test_normal_daytrade_paper_blocks_live_trading(gate_bridge):
    """Normal Daytrade Paper execution strictly blocks non-paper / live endpoints."""
    gate_bridge.daytrade_paper_execution_enabled = True
    gate_bridge.assert_account_mode_safety = MagicMock(side_effect=AccountModeViolation("Live trading forbidden"))

    with pytest.raises(AccountModeViolation, match="Live trading forbidden"):
        gate_bridge.place_bracket_order(
            symbol="NFLX",
            side="BUY",
            qty=59,
            entry_price=67.78,
            stop_price=67.47,
            target_price=68.26,
            client_order_id="dt_live_001",
        )
    assert gate_bridge._client.submit_order.call_count == 0


def test_normal_daytrade_paper_blocks_wrong_account(gate_bridge):
    """Normal Daytrade Paper execution blocks orders if account is not PA3UODIC5JTA."""
    gate_bridge.daytrade_paper_execution_enabled = True
    gate_bridge.account_id = "PA3R71I7JFDZ"  # Swing account lane

    with pytest.raises(ReleaseGateViolation, match="Wrong account lane"):
        gate_bridge.place_bracket_order(
            symbol="NFLX",
            side="BUY",
            qty=59,
            entry_price=67.78,
            stop_price=67.47,
            target_price=68.26,
            client_order_id="dt_wrong_lane",
        )
    assert gate_bridge._client.submit_order.call_count == 0


def test_protective_stop_can_be_created_without_entry_token(gate_bridge):
    """Protective stop orders do NOT depend on entry validation token."""
    gate_bridge.validation_release_token = None
    mock_stop = MagicMock(id="stop-999")
    gate_bridge._client.submit_order.return_value = mock_stop

    res = gate_bridge.place_protective_stop(
        symbol="NFLX",
        side="SELL",
        qty=59,
        stop_price=67.47,
        client_order_id="dt_stop_59",
    )
    assert res == "stop-999"
    assert gate_bridge._client.submit_order.call_count == 1


def test_protective_stop_cannot_increase_exposure(gate_bridge):
    """Protective stop quantity cannot exceed known open position exposure."""
    gate_bridge.validation_release_token = None
    mock_pos = MagicMock(qty=10.0)
    gate_bridge.positions = {"NFLX": mock_pos}

    with pytest.raises(ReleaseGateViolation, match="exceeds open position exposure"):
        gate_bridge.place_protective_stop(
            symbol="NFLX",
            side="SELL",
            qty=50,
            stop_price=67.47,
            client_order_id="dt_stop_overexposure",
        )
    assert gate_bridge._client.submit_order.call_count == 0


def test_expired_entry_token_cannot_block_verified_protective_action(gate_bridge):
    """Expired entry authorization cannot block risk-reducing protective stop."""
    expired_time = datetime.now(timezone.utc) - timedelta(minutes=30)
    token = ValidationReleaseToken(
        release_id="token-expired",
        account_id="PA3UODIC5JTA",
        paper_only=True,
        max_qty=1,
        expires_at=expired_time,
        consumed=True,
        authorized_symbol="AAPL",
        authorized_qty=1,
    )
    gate_bridge.validation_release_token = token
    mock_stop = MagicMock(id="stop-aapl")
    gate_bridge._client.submit_order.return_value = mock_stop

    res = gate_bridge.place_protective_stop(
        symbol="AAPL",
        side="SELL",
        qty=1,
        stop_price=145.0,
        client_order_id="val_stop_aapl",
    )
    assert res == "stop-aapl"
    assert gate_bridge._client.submit_order.call_count == 1



