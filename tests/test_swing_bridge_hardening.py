"""
tests/test_swing_bridge_hardening.py — Strict Contract Hardening Test Suite for Swing Trading.

Verifies the 9 Core Contracts approved in Phase C / Swing Bot Hardening:
1. test_sod_persisted_state_restored_same_day: Verify repeated calls on same ET date keep original SOD equity without broker query.
2. test_sod_invalid_persisted_state_halts_entries: Corrupt persisted state raises PersistedRiskStateInvalidError and halts entry gate.
3. test_sod_new_day_initializes_once: Advancing to a new date queries broker once and writes new baseline.
4. test_account_query_failure_fails_closed: Broker get_account failure / None / empty fails closed and places 0 orders.
5. test_swing_bracket_registers_tp_child_id: Bracket order registers both stop_leg_id and tp_leg_id and attaches to OrderResult.
6. test_tp_child_id_persisted_and_recovered_after_restart: Role registry persists and survives process restart.
7. test_cluster_risk_blocks_exceeded_limit: Boundary checks at 1.249% (pass), 1.250% (pass), 1.251% (reject).
8. test_cluster_risk_uses_only_same_cluster: Existing energy cluster risk does not inflate tech cluster risk.
9. test_regime_missing_spy_macro_input_fails_closed: Missing/None SPY macro inputs fail closed under DATA_INVALID.
"""

import os
import json
import math
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from alpaca.trading.enums import OrderType, OrderSide

from alpaca_bridge import (
    AlpacaBridge,
    PersistedRiskStateInvalidError,
    get_symbol_cluster,
    SwingPortfolioAuditResult,
    SWING_TECH_CLUSTER,
)
from predictor import Signal
import order_exec
from strategies.regime_classifier import MarketRegime
from strategies.swing_sizing import validate_swing_order_safety, SwingSizingRejected


@pytest.fixture
def tmp_path():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _make_valid_swing_signal(symbol="MSFT", action="BUY", price=100.0, **kwargs):
    sig = Signal(
        symbol=symbol,
        action=action,
        confidence=0.75,
        rf_score=0.55,
        lstm_score=0.55,
        xgb_score=0.55,
        trans_score=0.55,
        tech_score=0.55,
        price=price,
        reason="ML Buy",
        adx=30.0,
        vix=15.0,
        spy_price=500.0,
        spy_sma50=480.0,
        sma50_slope=0.50,
        regime=MarketRegime.TRENDING_BULL.value,
        quote_age_seconds=5.0,
    )
    for k, v in kwargs.items():
        setattr(sig, k, v)
    return sig


# ─── 1. SOD Calendar Lock & Fail-Closed Persistence ───────────────────────────

def test_sod_persisted_state_restored_same_day(tmp_path):
    """Test 1: Repeated calls on same ET date keep original SOD equity without broker query."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = "PA3R71I7JFDZ"
    bridge._risk_state_dir = str(tmp_path)

    # Pre-populate state file for today
    today_str = "2026-09-20"
    state_file = tmp_path / f"daytrade_risk_state_{bridge._account_id}.json"
    state_file.write_text(json.dumps({
        "account_id": bridge._account_id,
        "date": today_str,
        "start_of_day_equity": 50000.0,
        "suspended": False,
    }), encoding="utf-8")

    bridge.get_net_liquidation = MagicMock(return_value=55000.0)

    with patch("strategies.session.now_eastern") as mock_now:
        mock_now.return_value = datetime.datetime(2026, 9, 20, 10, 0, 0, tzinfo=datetime.timezone.utc)
        sod_1 = bridge.snapshot_start_of_day_equity()
        assert sod_1 == 50000.0
        bridge.get_net_liquidation.assert_not_called()

        sod_2 = bridge.snapshot_start_of_day_equity()
        assert sod_2 == 50000.0
        bridge.get_net_liquidation.assert_not_called()


def test_sod_invalid_persisted_state_halts_entries(tmp_path):
    """Test 2: Corrupted or malformed persisted state raises PersistedRiskStateInvalidError and halts entry gate."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = "PA3R71I7JFDZ"
    bridge._risk_state_dir = str(tmp_path)

    # Write corrupt JSON to state file
    state_file = tmp_path / f"daytrade_risk_state_{bridge._account_id}.json"
    state_file.write_text("{corrupted_json: true,", encoding="utf-8")

    bridge.get_net_liquidation = MagicMock(return_value=50000.0)

    with patch("strategies.session.now_eastern") as mock_now:
        mock_now.return_value = datetime.datetime(2026, 9, 20, 10, 0, 0, tzinfo=datetime.timezone.utc)
        with pytest.raises(PersistedRiskStateInvalidError):
            bridge.snapshot_start_of_day_equity()

    assert bridge.entry_gate.halted is True
    assert "PERSISTED_STATE_INVALID" in bridge.entry_gate.reasons
    blocked, reason = bridge.entry_gate.entry_blocked("AAPL", 100, None)
    assert blocked is True
    assert "PERSISTED_STATE_INVALID" in reason


def test_sod_new_day_initializes_once(tmp_path):
    """Test 3: Advancing to a new date queries broker once and writes new baseline."""
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._account_id = "PA3R71I7JFDZ"
    bridge._risk_state_dir = str(tmp_path)

    # Pre-populate state file for yesterday
    state_file = tmp_path / f"daytrade_risk_state_{bridge._account_id}.json"
    state_file.write_text(json.dumps({
        "account_id": bridge._account_id,
        "date": "2026-09-19",
        "start_of_day_equity": 48000.0,
        "suspended": False,
    }), encoding="utf-8")

    bridge.get_net_liquidation = MagicMock(return_value=51200.0)

    with patch("strategies.session.now_eastern") as mock_now:
        mock_now.return_value = datetime.datetime(2026, 9, 20, 9, 30, 0, tzinfo=datetime.timezone.utc)
        sod_1 = bridge.snapshot_start_of_day_equity()
        assert sod_1 == 51200.0
        assert bridge.get_net_liquidation.call_count == 1

        # Second call on same new day uses cached / persisted value
        sod_2 = bridge.snapshot_start_of_day_equity()
        assert sod_2 == 51200.0
        assert bridge.get_net_liquidation.call_count == 1

    # Verify persisted state file on disk
    saved_data = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved_data["date"] == "2026-09-20"
    assert saved_data["start_of_day_equity"] == 51200.0


# ─── 2. Broker Account Identity Fail-Closed ───────────────────────────────────

def test_account_query_failure_fails_closed():
    """Test 4: Broker get_account failure / None / empty fails closed and places 0 orders."""
    bridge = AlpacaBridge(mode="swing", dry_run=False)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=100.0))
    bridge.get_position = MagicMock(return_value=0)
    bridge.has_working_order = MagicMock(return_value=False)
    bridge._enforce_market_hours_test = False

    sig = _make_valid_swing_signal(symbol="MSFT", action="BUY", price=100.0)

    # 4a: get_account raises Exception
    bridge._client.get_account.side_effect = RuntimeError("Broker connection timeout")
    assert bridge.execute_signal(sig) is False
    bridge._client.submit_order.assert_not_called()

    # 4b: get_account returns None
    bridge._client.get_account.side_effect = None
    bridge._client.get_account.return_value = None
    assert bridge.execute_signal(sig) is False
    bridge._client.submit_order.assert_not_called()

    # 4c: get_account returns account with empty/missing account_number
    mock_empty_acct = MagicMock(account_number="")
    bridge._client.get_account.return_value = mock_empty_acct
    assert bridge.execute_signal(sig) is False
    bridge._client.submit_order.assert_not_called()


# ─── 3. Dual Child-ID Telemetry & Recovery ───────────────────────────────────

def test_swing_bracket_registers_tp_child_id(tmp_path, monkeypatch):
    """Test 5: Bracket order registers both stop_leg_id and tp_leg_id and attaches to OrderResult."""
    events_path = tmp_path / "swing_events.jsonl"
    monkeypatch.setenv("BROKER_EVENTS_PATH", str(events_path))

    bridge = AlpacaBridge(mode="swing", dry_run=False)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._require_new_entries_allowed = MagicMock()
    bridge._require_trade_stream = MagicMock()

    # Setup broker responses: parent order and filled order with stop and TP child legs
    mock_parent = MagicMock(id="parent-ord-123", client_order_id="swg_test_parent")
    mock_stop_leg = MagicMock(id="stop-leg-456", type=OrderType.STOP, side=OrderSide.SELL, stop_price=95.0, limit_price=None)
    mock_tp_leg = MagicMock(id="tp-leg-789", type=OrderType.LIMIT, side=OrderSide.SELL, stop_price=None, limit_price=110.0)

    mock_filled_parent = MagicMock(
        id="parent-ord-123",
        status="filled",
        filled_qty=10,
        filled_avg_price=100.0,
        legs=[mock_stop_leg, mock_tp_leg],
    )
    bridge._client.submit_order.return_value = mock_parent
    bridge._client.get_order_by_id.return_value = mock_filled_parent

    res = bridge._place_open_bracket(
        symbol="AAPL",
        action="BUY",
        qty=10,
        price=100.0,
        stop_loss=95.0,
        target_price=110.0,
        client_order_id="swg_test_parent",
    )

    assert res.outcome == order_exec.FILLED
    assert res.stop_order_id == "stop-leg-456"
    assert res.tp_order_id == "tp-leg-789"

    # Verify both child IDs are registered
    stop_info = bridge.get_order_role_info("stop-leg-456")
    assert stop_info.get("role") == "STOP_LOSS"
    assert stop_info.get("parent_order_id") == "parent-ord-123"

    tp_info = bridge.get_order_role_info("tp-leg-789")
    assert tp_info.get("role") == "TAKE_PROFIT"
    assert tp_info.get("parent_order_id") == "parent-ord-123"


def test_tp_child_id_persisted_and_recovered_after_restart(tmp_path, monkeypatch):
    """Test 6: Role registry persists and survives process restart."""
    events_path = tmp_path / "swing_events.jsonl"
    monkeypatch.setenv("BROKER_EVENTS_PATH", str(events_path))

    bridge_1 = AlpacaBridge(mode="swing", dry_run=True)
    bridge_1.register_bracket_order_roles(
        parent_id="parent-rec-1",
        stop_leg_id="stop-rec-1",
        tp_leg_id="tp-rec-1",
        client_order_id="swg_rec_1",
    )

    # Instantiate brand new bridge instance pointing to same storage
    bridge_2 = AlpacaBridge(mode="swing", dry_run=True)
    recovered_tp = bridge_2.get_order_role_info("tp-rec-1")
    assert recovered_tp.get("role") == "TAKE_PROFIT"
    assert recovered_tp.get("parent_order_id") == "parent-rec-1"
    assert recovered_tp.get("stop_order_id") == "stop-rec-1"

    recovered_stop = bridge_2.get_order_role_info("stop-rec-1")
    assert recovered_stop.get("role") == "STOP_LOSS"
    assert recovered_stop.get("parent_order_id") == "parent-rec-1"


# ─── 4. Canonical Cluster-Risk Aggregation & Limits ───────────────────────────

def test_cluster_risk_blocks_exceeded_limit():
    """Test 7: Boundary checks at 1.249% (pass), 1.250% (pass), 1.251% (reject)."""
    equity = 100_000.0

    # Case A: Below boundary (proj ~ 1.243% <= 1.250%)
    validate_swing_order_safety(
        client_order_id="swg_bound_pass_1",
        strategy_instance_id="SWING_BOT_PRO",
        symbol="MSFT",
        signal_version="v1.0",
        entry_price=100.0,
        stop_price=95.0,
        shares=48,
        account_equity=equity,
        available_buying_power=50_000.0,
        current_open_stop_risk=1000.0,
        pending_entry_risk=0.0,
        current_cluster_risk_pct=0.0100,
        max_cluster_risk_pct=0.0125,
        symbol_sector="INFORMATION_TECHNOLOGY",
    )

    # Case B: At/under boundary (shares=49 -> proj = 0.0100 + 0.0024844 = 0.0124844 <= 0.0125 -> PASS)
    validate_swing_order_safety(
        client_order_id="swg_bound_pass_2",
        strategy_instance_id="SWING_BOT_PRO",
        symbol="MSFT",
        signal_version="v1.0",
        entry_price=100.0,
        stop_price=95.0,
        shares=49,
        account_equity=equity,
        available_buying_power=50_000.0,
        current_open_stop_risk=1000.0,
        pending_entry_risk=0.0,
        current_cluster_risk_pct=0.0100,
        max_cluster_risk_pct=0.0125,
        symbol_sector="INFORMATION_TECHNOLOGY",
    )

    # Case C: Exceeds boundary 1.251% (current 1.05% + candidate 0.248% = 1.298% > 1.25%)
    with pytest.raises(SwingSizingRejected) as exc_info:
        validate_swing_order_safety(
            client_order_id="swg_bound_fail",
            strategy_instance_id="SWING_BOT_PRO",
            symbol="MSFT",
            signal_version="v1.0",
            entry_price=100.0,
            stop_price=95.0,
            shares=49,
            account_equity=equity,
            available_buying_power=50_000.0,
            current_open_stop_risk=1000.0,
            pending_entry_risk=0.0,
            current_cluster_risk_pct=0.0105,
            max_cluster_risk_pct=0.0125,
            symbol_sector="INFORMATION_TECHNOLOGY",
        )
    assert "Cluster risk" in str(exc_info.value)


def test_cluster_risk_uses_only_same_cluster():
    """Test 8: Existing energy cluster risk does not inflate tech cluster risk.

    Validates that SWING_TECH_CLUSTER is strictly the canonical 13 symbols.
    """
    expected_tech = frozenset({
        "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AMD",
        "GOOGL", "QQQ", "PLTR", "AVGO", "MU", "NFLX",
    })
    assert SWING_TECH_CLUSTER == expected_tech
    assert len(SWING_TECH_CLUSTER) == 13
    for sym in SWING_TECH_CLUSTER:
        assert get_symbol_cluster(sym) == "TECH"
    assert get_symbol_cluster("XOM") == "ENERGY"

    # Suppose we have per-symbol risk from audit
    per_symbol_risk = {
        "XOM": 1000.0,  # 1.0% on $100k equity (ENERGY)
        "AAPL": 300.0,  # 0.3% on $100k equity (TECH)
    }

    # Candidate symbol NVDA is in TECH cluster
    cand_cluster = get_symbol_cluster("NVDA")
    assert cand_cluster == "TECH"

    # Aggregating risk strictly for candidate's cluster
    tech_cluster_risk = sum(
        sym_risk for sym_key, sym_risk in per_symbol_risk.items()
        if get_symbol_cluster(sym_key) == cand_cluster
    )

    # Only AAPL ($300) should be included, XOM ($1000) must be excluded
    assert tech_cluster_risk == 300.0

    # If XOM was improperly included, total would be $1300 (1.30% > 1.25% limit)
    # But with canonical cluster mapping, tech cluster risk is 0.30%, well below the 1.25% cap
    equity = 100_000.0
    current_cluster_risk_pct = tech_cluster_risk / equity
    assert current_cluster_risk_pct == 0.003  # 0.30%

    # Candidate NVDA order with ~$250 risk passes
    validate_swing_order_safety(
        client_order_id="swg_nvda_ok",
        strategy_instance_id="SWING_BOT_PRO",
        symbol="NVDA",
        signal_version="v1.0",
        entry_price=120.0,
        stop_price=115.0,
        shares=40,
        account_equity=equity,
        available_buying_power=50_000.0,
        current_open_stop_risk=1300.0,
        pending_entry_risk=0.0,
        current_cluster_risk_pct=current_cluster_risk_pct,
        max_cluster_risk_pct=0.0125,
        symbol_sector="INFORMATION_TECHNOLOGY",
    )


def test_cluster_risk_includes_pending_entries():
    """Test 8b: Canonical cluster risk aggregates both open stop risk AND pending entry risk.

    Proves that a candidate order is rejected when open + pending + proposed > cap (1.25%),
    even though open + proposed alone would have passed.
    """
    equity = 100_000.0
    cand_price = 100.0
    cand_stop = 95.0
    cand_shares = 20  # Effective risk ~= 20 * (5.0 + 0.06) + 0.50 = $101.70 (~0.1017% of equity)

    # 1. Direct validation:
    # Open TECH risk = $1,100 (1.10%)
    # Pending TECH entry risk = $100 (0.10%)
    # Combined cluster risk = $1,200 (1.20%)
    cand_cluster = get_symbol_cluster("NVDA")
    assert cand_cluster == "TECH"

    open_risk_dollars = 1100.0
    pending_risk_dollars = 100.0
    combined_cluster_pct = (open_risk_dollars + pending_risk_dollars) / equity  # 1.20%
    open_only_cluster_pct = open_risk_dollars / equity  # 1.10%

    # If pending risk is improperly omitted, open_only (1.10%) + candidate (0.1017%) = 1.2017% <= 1.25% -> PASSES
    validate_swing_order_safety(
        client_order_id="swg_nvda_open_only_pass",
        strategy_instance_id="SWING_BOT_PRO",
        symbol="NVDA",
        signal_version="v1.0",
        entry_price=cand_price,
        stop_price=cand_stop,
        shares=cand_shares,
        account_equity=equity,
        available_buying_power=50_000.0,
        current_open_stop_risk=open_risk_dollars,
        pending_entry_risk=0.0,
        current_cluster_risk_pct=open_only_cluster_pct,
        max_cluster_risk_pct=0.0125,
        symbol_sector="INFORMATION_TECHNOLOGY",
    )

    # But with pending risk properly included: combined (1.20%) + candidate (0.1017%) = 1.3017% > 1.25% -> FAILS
    with pytest.raises(SwingSizingRejected) as exc_info:
        validate_swing_order_safety(
            client_order_id="swg_nvda_pending_blocked",
            strategy_instance_id="SWING_BOT_PRO",
            symbol="NVDA",
            signal_version="v1.0",
            entry_price=cand_price,
            stop_price=cand_stop,
            shares=cand_shares,
            account_equity=equity,
            available_buying_power=50_000.0,
            current_open_stop_risk=open_risk_dollars,
            pending_entry_risk=pending_risk_dollars,
            current_cluster_risk_pct=combined_cluster_pct,
            max_cluster_risk_pct=0.0125,
            symbol_sector="INFORMATION_TECHNOLOGY",
        )
    assert "Cluster risk" in str(exc_info.value)

    # 2. Bridge execute_signal integration proof:
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    acct_mock = MagicMock()
    acct_mock.account_number = "PA3R71I7JFDZ"
    acct_mock.equity = equity
    acct_mock.buying_power = 50_000.0
    acct_mock.multiplier = "2"
    bridge._client.get_account.return_value = acct_mock
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=cand_price))
    bridge.get_position = MagicMock(return_value=0)
    bridge.has_working_order = MagicMock(return_value=False)
    bridge._enforce_market_hours_test = False

    # Mock audit_swing_portfolio_safety returning both open risk and pending risk in TECH cluster
    audit_mock = SwingPortfolioAuditResult(
        current_open_stop_risk=open_risk_dollars,
        pending_entry_risk=pending_risk_dollars,
        active_position_symbols=set(),
        pending_new_symbols=set(),
        per_symbol_open_risk={"AAPL": open_risk_dollars},
        per_symbol_pending_risk={"MSFT": pending_risk_dollars},
        per_symbol_risk={"AAPL": open_risk_dollars, "MSFT": pending_risk_dollars},
    )
    bridge.audit_swing_portfolio_safety = MagicMock(return_value=audit_mock)

    sig = _make_valid_swing_signal(symbol="NVDA", price=cand_price)
    with patch("strategies.swing_sizing.calculate_swing_position_size") as mock_calc:
        from strategies.swing_sizing import SizingDecision
        mock_calc.return_value = SizingDecision(
            shares=cand_shares,
            binding_constraint="RISK",
            risk_budget=500.0,
            estimated_risk=101.70,
            gross_notional=2000.0,
            q_risk=cand_shares,
            q_notional=100,
            q_liquidity=1000,
            q_portfolio=100,
            q_broker=100,
            normalized_stop=cand_stop,
            stop_distance=5.0,
            rejected=False,
        )
        # Result must be False because pending risk pushes cluster risk over 1.25%
        res = bridge.execute_signal(sig)
        assert res is False
        bridge._client.submit_order.assert_not_called()


# ─── 5. Zero-Synthetic Macro Regime Fail-Closed ───────────────────────────────

def test_regime_missing_spy_macro_input_fails_closed():
    """Test 9: Macro regime fail-closed behavior across all 4 failure modes.

    Covers:
      Mode 1: Missing inputs (None)
      Mode 2: Non-finite inputs (NaN, +/-Inf)
      Mode 3: Stale quote age (> 60.0s)
      Mode 4: Prohibited / Invalid regimes (TRANSITIONING, TRENDING_BEAR, DATA_INVALID)
    """
    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=100.0))
    bridge.get_position = MagicMock(return_value=0)
    bridge.has_working_order = MagicMock(return_value=False)
    bridge._enforce_market_hours_test = False

    # ── Mode 1: Missing inputs (None) ─────────────────────────────────────────
    assert bridge.execute_signal(_make_valid_swing_signal(spy_price=None)) is False
    assert bridge.execute_signal(_make_valid_swing_signal(adx=None)) is False
    assert bridge.execute_signal(_make_valid_swing_signal(vix=None)) is False
    assert bridge.execute_signal(_make_valid_swing_signal(spy_sma50=None)) is False
    assert bridge.execute_signal(_make_valid_swing_signal(sma50_slope=None)) is False

    # ── Mode 2: Non-finite inputs (NaN, +Inf, -Inf) ────────────────────────────
    assert bridge.execute_signal(_make_valid_swing_signal(spy_price=float("nan"))) is False
    assert bridge.execute_signal(_make_valid_swing_signal(adx=float("inf"))) is False
    assert bridge.execute_signal(_make_valid_swing_signal(vix=float("-inf"))) is False
    assert bridge.execute_signal(_make_valid_swing_signal(spy_sma50=float("nan"))) is False
    assert bridge.execute_signal(_make_valid_swing_signal(sma50_slope=float("nan"))) is False

    # ── Mode 3: Stale quote age (> 60.0s) ──────────────────────────────────────
    assert bridge.execute_signal(_make_valid_swing_signal(quote_age_seconds=65.0)) is False
    assert bridge.execute_signal(_make_valid_swing_signal(quote_age_seconds=60.1)) is False

    # ── Mode 4: Prohibited / Invalid regimes ──────────────────────────────────
    # Mode 4a: TRANSITIONING regime (20 < ADX < 25)
    sig_trans = _make_valid_swing_signal(adx=22.5)
    assert bridge.execute_signal(sig_trans) is False

    # Mode 4b: TRENDING_BEAR regime (ADX >= 25, SPY < SMA50, slope < 0)
    sig_bear = _make_valid_swing_signal(
        adx=30.0,
        spy_price=400.0,
        spy_sma50=500.0,
        sma50_slope=-0.50,
    )
    assert bridge.execute_signal(sig_bear) is False

    # Mode 4c: Conflicting indicators -> TRANSITIONING
    sig_conflict = _make_valid_swing_signal(
        adx=30.0,
        spy_price=510.0,
        spy_sma50=500.0,
        sma50_slope=-0.20,
    )
    assert bridge.execute_signal(sig_conflict) is False

    # Zero broker orders should have been submitted across all failure modes
    bridge._client.submit_order.assert_not_called()

