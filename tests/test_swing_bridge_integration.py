import json
import math
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from alpaca_bridge import AlpacaBridge
from predictor import Signal
import order_exec
from strategies.swing_sizing import (
    calculate_swing_position_size,
    calculate_swing_stop_price,
    build_canonical_swing_order_id,
    PositionIntent,
    UnsupportedSwingDirection,
)
from strategies.regime_classifier import MarketRegime, classify_market_regime
from strategies.startup_security import (
    acquire_atomic_process_lock,
    release_atomic_process_lock,
    ProcessLockMetadata,
    StartupSafetyRejected,
)
from strategies.swing_risk_store import (
    PositionRiskBasis,
    save_risk_basis,
    load_risk_store,
    remove_risk_basis,
    get_unrealized_r,
)


# ─── 1. Five-Way Quantity Clipping Tests ─────────────────────────────────────

def test_five_way_clipping_risk_budget_binds():
    # Large equity, wide stop distance -> Risk budget binds
    res = calculate_swing_position_size(
        entry_price=100.0,
        stop_price=95.0,  #  distance + .06 slippage = .06 per share
        account_equity=100_000.0,  # Risk budget = min(500, 500) = 
        available_buying_power=50_000.0,
        adv_shares_20=1_000_000.0,
    )
    assert not res.rejected
    assert res.binding_constraint == 'RISK'
    assert res.q_risk == 97  # floor((495 - 0.50) / 5.06) = 97
    assert res.shares == 97


def test_five_way_clipping_symbol_notional_binds():
    # High risk tolerance, but symbol notional cap (10% = ,000) binds
    res = calculate_swing_position_size(
        entry_price=200.0,
        stop_price=199.50,  # Tiny stop (.50) -> q_risk would be huge
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        existing_symbol_exposure=7_000.0,  # Remaining symbol cap = ,000 -> 15 shares
        adv_shares_20=1_000_000.0,
    )
    assert not res.rejected
    assert res.binding_constraint == 'SYMBOL_NOTIONAL'
    assert res.q_notional == 15  # floor(3000 / 200) = 15
    assert res.shares == 15


def test_five_way_clipping_liquidity_binds():
    # Illiquid symbol with low ADV
    res = calculate_swing_position_size(
        entry_price=50.0,
        stop_price=49.0,
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        adv_shares_20=2_000.0,  # 1% participation = 20 shares
    )
    assert not res.rejected
    assert res.binding_constraint == 'LIQUIDITY'
    assert res.q_liquidity == 20
    assert res.shares == 20


def test_five_way_clipping_portfolio_exposure_binds():
    # Portfolio gross exposure (15% = ,000) almost full
    res = calculate_swing_position_size(
        entry_price=100.0,
        stop_price=98.0,
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        existing_portfolio_exposure=14_200.0,  # Remaining portfolio budget =  -> 8 shares
        adv_shares_20=1_000_000.0,
    )
    assert not res.rejected
    assert res.binding_constraint == 'PORTFOLIO_EXPOSURE'
    assert res.q_portfolio == 8  # floor(800 / 100) = 8
    assert res.shares == 8


def test_five_way_clipping_buying_power_binds():
    # Broker cash/buying power heavily constrained
    res = calculate_swing_position_size(
        entry_price=100.0,
        stop_price=98.0,
        account_equity=100_000.0,
        available_buying_power=600.0,  # 95% =  -> 5 shares
        adv_shares_20=1_000_000.0,
    )
    assert not res.rejected
    assert res.binding_constraint == 'BUYING_POWER'
    assert res.q_broker == 5  # floor(570 / 100) = 5
    assert res.shares == 5


def test_five_way_clipping_zero_shares_rejection():
    # Zero remaining portfolio budget -> rejection
    res = calculate_swing_position_size(
        entry_price=100.0,
        stop_price=98.0,
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        existing_portfolio_exposure=15_000.0,  # 100% full
    )
    assert res.rejected
    assert res.shares == 0
    assert res.binding_constraint == 'ZERO_SHARES'
    assert 'NO_WHOLE_SHARE' in res.rejection_reason


# ─── 2. ATR Stop Price Calculation & Clamping ────────────────────────────────

def test_atr_stop_price_bounds_and_tick_floor():
    # Entry 100, ATR 2.0, Bull (1.50x) -> raw distance 3.0 -> stop 97.0
    stop_px, dist = calculate_swing_stop_price(entry_price=100.0, atr_14=2.0, regime='TRENDING_BULL')
    assert stop_px == 97.0
    assert dist == 3.0

    # Test Min Clamp: distance < 1.50 * ATR (e.g. tight structural swing low at 99.0)
    stop_min, dist_min = calculate_swing_stop_price(
        entry_price=100.0, atr_14=2.0, swing_low=99.0, tick_buffer=0.01
    )
    assert dist_min == 3.0  # Clamped to 1.50 * 2.0 = 3.0
    assert stop_min == 97.0

    # Test Max Clamp: distance > 3.00 * ATR (e.g. wide structural swing low at 90.0)
    stop_max, dist_max = calculate_swing_stop_price(
        entry_price=100.0, atr_14=2.0, swing_low=90.0
    )
    assert dist_max == 6.0  # Clamped to 3.00 * 2.0 = 6.0
    assert stop_max == 94.0

    # Conservative floor check: never round stop up
    stop_odd, _ = calculate_swing_stop_price(entry_price=100.055, atr_14=2.001, tick_size=0.01)
    assert stop_odd <= (100.055 - (1.5 * 2.001))


# ─── 3. Dry-Run Mutation Barrier Tests ───────────────────────────────────────

def test_dry_run_blocks_all_broker_mutations():
    bridge = AlpacaBridge(mode='swing', dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._require_new_entries_allowed = MagicMock()
    bridge._require_trade_stream = MagicMock()

    # 1. place_bracket_order
    res_b = bridge.place_bracket_order('AAPL', 'BUY', 10, 150.0, 145.0, 160.0)
    assert res_b['status'] == 'dry_run_blocked'
    assert res_b['placed'] is False
    assert res_b['submitted'] is False
    bridge._client.submit_order.assert_not_called()

    # 2. _place_open_bracket
    res_open = bridge._place_open_bracket('AAPL', 'BUY', 10, 150.0, 0.60)
    assert res_open.outcome == order_exec.REJECTED
    assert res_open.status == 'DryRunBlocked'
    bridge._client.submit_order.assert_not_called()

    # 3. close_position
    assert bridge.close_position('AAPL') is True
    bridge._client.close_position.assert_not_called()

    # 4. _close_position
    bridge.get_position = MagicMock(return_value=10)
    res_close = bridge._close_position('AAPL', 'SELL', 10, 150.0, 'test')
    assert res_close.outcome == order_exec.REJECTED
    assert res_close.status == 'DryRunBlocked'
    bridge._client.submit_order.assert_not_called()

    # 5. cancel_order
    assert bridge.cancel_order('order-999') is True
    bridge._client.cancel_order_by_id.assert_not_called()

    # 6. place_stop_order
    res_stop = bridge.place_stop_order('AAPL', 'SELL', 10, 145.0)
    assert 'dry_stop' in res_stop['order_id']
    bridge._client.submit_order.assert_not_called()

    # 7. modify_stop_order & replace_stop
    assert 'dry_' in bridge.modify_stop_order('order-1', 146.0)
    bridge._client.replace_order_by_id.assert_not_called()


def _make_signal(symbol='MSFT', action='BUY', confidence=0.75, price=100.0, **kwargs):
    sig = Signal(
        symbol=symbol,
        action=action,
        confidence=confidence,
        rf_score=0.5,
        lstm_score=0.5,
        xgb_score=0.5,
        trans_score=0.5,
        tech_score=0.5,
        price=price,
        reason='test',
    )
    for k, v in kwargs.items():
        setattr(sig, k, v)
    return sig


# ─── 4. Regime Filter Veto Tests ─────────────────────────────────────────────

def test_regime_filter_veto_blocks_swing_entry():
    bridge = AlpacaBridge(mode='swing', dry_run=True)
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=100.0))
    bridge.get_position = MagicMock(return_value=0)
    bridge.has_working_order = MagicMock(return_value=False)
    bridge.get_net_liquidation = MagicMock(return_value=100_000.0)

    # Signal in TRANSITIONING regime
    sig_trans = _make_signal(
        symbol='MSFT', action='BUY', confidence=0.75, price=100.0,
        regime=MarketRegime.TRANSITIONING.value,
        adx=22.0, vix=17.0, spy_price=500.0, spy_sma50=500.0, sma50_slope=0.0,
    )
    assert bridge.execute_signal(sig_trans) is False

    # Signal in DATA_INVALID regime
    sig_invalid = _make_signal(
        symbol='MSFT', action='BUY', confidence=0.75, price=100.0,
        regime=MarketRegime.DATA_INVALID.value,
        adx=float('nan'), vix=17.0, spy_price=500.0, spy_sma50=500.0, sma50_slope=0.0,
    )
    assert bridge.execute_signal(sig_invalid) is False


# ─── 5. Long-Only Policy Tests ───────────────────────────────────────────────

def test_stage0_swing_blocks_short_entries_and_over_selling():
    bridge = AlpacaBridge(mode='swing', dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=100.0))
    bridge.get_position = MagicMock(return_value=0)

    # Short SELL with no existing position is blocked
    sig_sell = _make_signal(symbol='NVDA', action='SELL', confidence=0.80, price=100.0)
    assert bridge.execute_signal(sig_sell) is False

    # Close long with quantity exceeding long position is rejected
    bridge.get_position = MagicMock(return_value=5)
    res = bridge._close_position('NVDA', 'SELL', qty=10, price=100.0, note='over-close')
    assert res.outcome == order_exec.REJECTED
    assert res.status == 'InvalidCloseQuantity'


# ─── 6. Canonical Order ID Tests ─────────────────────────────────────────────

def test_canonical_swing_order_id():
    id_1 = build_canonical_swing_order_id(
        symbol='AAPL', side='BUY', session_date='2026-09-08',
        bar_timestamp_utc='2026-09-08T13:30:00Z', signal_kind='ML_SWING',
        account_number='PA3R71I7JFDZ',
    )
    id_2 = build_canonical_swing_order_id(
        symbol='AAPL', side='BUY', session_date='2026-09-08',
        bar_timestamp_utc='2026-09-08T13:30:00Z', signal_kind='ML_SWING',
        account_number='PA3R71I7JFDZ',
    )
    assert id_1.startswith('swg_')
    assert len(id_1) == 20  # 'swg_' (4) + 16 hex chars
    assert id_1 == id_2  # Deterministic


# ─── 7. Atomic Process Lock Tests ────────────────────────────────────────────

def test_atomic_process_lock_lifecycle(tmp_path):
    lock_file = tmp_path / 'swing_bot.lock'
    meta = ProcessLockMetadata(
        pid=os.getpid(),
        hostname='test-host',
        account='PA3R71I7JFDZ',
        mode='swing',
        branch='research/swing-atr-engine',
        release_sha='c1fa450',
        started_at_utc='2026-09-06T13:00:00Z',
    )

    # 1. Acquire lock
    fd = acquire_atomic_process_lock(lock_file, meta)
    assert fd >= 0
    assert lock_file.exists()

    # Verify content
    with open(lock_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    assert data['account'] == 'PA3R71I7JFDZ'
    assert data['pid'] == os.getpid()

    # 2. Duplicate acquire fails
    with pytest.raises(StartupSafetyRejected, match='STARTUP_LOCK_COLLISION'):
        acquire_atomic_process_lock(lock_file, meta)

    # 3. Release lock
    release_atomic_process_lock(lock_file, fd)
    assert not lock_file.exists()


# ─── 8. Risk Basis Store Tests ───────────────────────────────────────────────

def test_swing_risk_basis_store_and_unrealized_r(tmp_path):
    store_file = tmp_path / 'swing_risk.json'
    basis = PositionRiskBasis(
        symbol='AAPL',
        entry_price=150.0,
        initial_stop_price=145.0,
        initial_risk_per_share=5.0,
        initial_total_risk=50.0,
        shares=10,
        entry_timestamp_utc='2026-09-06T13:00:00Z',
    )

    save_risk_basis(basis, path=store_file)
    store = load_risk_store(path=store_file)
    assert 'AAPL' in store
    assert store['AAPL'].initial_risk_per_share == 5.0

    # At price 155 -> +5.0 profit / .0 risk = +1.0R
    r_val = get_unrealized_r('AAPL', current_price=155.0, avg_entry_price=150.0, path=store_file)
    assert pytest.approx(r_val, 0.01) == 1.0

    # At price 160 -> +10.0 profit / .0 risk = +2.0R
    r_val_2 = get_unrealized_r('AAPL', current_price=160.0, avg_entry_price=150.0, path=store_file)
    assert pytest.approx(r_val_2, 0.01) == 2.0

    remove_risk_basis('AAPL', path=store_file)
    assert 'AAPL' not in load_risk_store(path=store_file)
