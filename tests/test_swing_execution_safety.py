from unittest.mock import MagicMock, patch
import pytest
from alpaca_bridge import AlpacaBridge
from predictor import Signal
import order_exec


def test_execute_signal_soxx_gate_veto_blocks_tech_long():
    bridge = AlpacaBridge(mode="swing")
    bridge._connected = True
    bridge._client = MagicMock()
    
    # Mock quote decision
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=150.0))
    bridge.get_position = MagicMock(return_value=0)
    bridge.has_working_order = MagicMock(return_value=False)
    bridge.get_net_liquidation = MagicMock(return_value=100000.0)

    # Patch check_soxx_tech_long_gate to return FAIL
    with patch("strategies.intraday_risk.check_soxx_tech_long_gate", return_value=(False, "SOXX is -1.90% below threshold")):
        sig = Signal(
            symbol="AMD", action="BUY", confidence=0.56,
            rf_score=0.55, lstm_score=0.57, xgb_score=0.61, trans_score=0.61, tech_score=0.55,
            price=150.0, reason="ML Buy"
        )
        result = bridge.execute_signal(sig)
        
        assert result is False
        # Ensure bracket order submission was NOT attempted
        bridge._client.submit_order.assert_not_called()


def test_await_order_outcome_reconciles_and_cancels_unfilled_order():
    bridge = AlpacaBridge(mode="swing")
    bridge._client = MagicMock()

    mock_working_order = MagicMock()
    mock_working_order.status = "new"
    mock_working_order.filled_qty = 0
    mock_working_order.qty = 4
    mock_working_order.limit_price = 150.0

    mock_cancelled_order = MagicMock()
    mock_cancelled_order.status = "canceled"
    mock_cancelled_order.filled_qty = 0
    mock_cancelled_order.qty = 4

    # First returns working order (during timeout polling), then returns cancelled after cancel call
    bridge._client.get_order_by_id.side_effect = [
        mock_working_order, # initial poll
        mock_working_order, # post-timeout query
        mock_cancelled_order # cancel confirmation
    ]

    res = bridge._await_order_outcome(order_id="test-order-123", timeout=0.1)

    assert res.outcome == order_exec.CANCELLED
    assert res.filled == 0.0
    bridge._client.cancel_order_by_id.assert_called_once_with("test-order-123")


def test_await_order_outcome_handles_immediate_fill():
    bridge = AlpacaBridge(mode="swing")
    bridge._client = MagicMock()

    mock_filled_order = MagicMock()
    mock_filled_order.status = "filled"
    mock_filled_order.filled_qty = 4
    mock_filled_order.qty = 4
    mock_filled_order.filled_avg_price = 149.95

    bridge._client.get_order_by_id.return_value = mock_filled_order

    res = bridge._await_order_outcome(order_id="test-fill-123", timeout=1.0)

    assert res.outcome == order_exec.FILLED
    assert res.filled == 4.0
    assert res.avg_fill_price == 149.95
    bridge._client.cancel_order_by_id.assert_not_called()


def test_both_bracket_submission_paths_use_gtc_in_swing_mode():
    from alpaca.trading.enums import TimeInForce
    bridge = AlpacaBridge(mode="swing")
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._require_new_entries_allowed = MagicMock()
    bridge._require_trade_stream = MagicMock()

    # 1. Test helper
    assert bridge._entry_time_in_force() == TimeInForce.GTC

    # 2. Test _place_open_bracket
    mock_order = MagicMock(id="parent-123")
    bridge._client.submit_order.return_value = mock_order
    bridge._client.get_order_by_id.return_value = MagicMock(status="filled", filled_qty=5, filled_avg_price=100.0, legs=[])
    
    bridge._place_open_bracket("AAPL", "BUY", 5, 100.0, 0.60)
    args, kwargs = bridge._client.submit_order.call_args
    req = args[0] if args else kwargs.get("order_data")
    assert req.time_in_force == TimeInForce.GTC

    # 3. Test place_bracket_order
    bridge._client.submit_order.reset_mock()
    bridge._client.submit_order.return_value = mock_order
    bridge.place_bracket_order("AAPL", "BUY", 5, 100.0, 110.0, 95.0)
    args, kwargs = bridge._client.submit_order.call_args
    req = args[0] if args else kwargs.get("order_data")
    assert req.time_in_force == TimeInForce.GTC


def test_unknown_or_daytrade_mode_defaults_to_day_tif():
    from alpaca.trading.enums import TimeInForce
    b_day = AlpacaBridge(mode="daytrade")
    assert b_day._entry_time_in_force() == TimeInForce.DAY

    b_unknown = AlpacaBridge(mode="scalp")
    assert b_unknown._entry_time_in_force() == TimeInForce.DAY


def test_order_type_attribute_compatibility_and_normalization():
    from alpaca.trading.enums import OrderType
    bridge = AlpacaBridge(mode="swing")
    bridge._connected = True
    bridge._client = MagicMock()

    # Mock order where type is an OrderType enum and order_type attribute is absent
    mock_order = MagicMock(spec=["id", "symbol", "type", "stop_price"])
    mock_order.id = "order-stop-1"
    mock_order.symbol = "AAPL"
    mock_order.type = OrderType.STOP
    mock_order.stop_price = 145.50

    bridge._get_active_orders = MagicMock(return_value=[mock_order])

    stop_px = bridge.get_current_stop("AAPL")
    assert stop_px == 145.50


def test_reconcile_startup_state_refreshes_snapshot_on_repair():
    bridge = AlpacaBridge(mode="swing")
    bridge._connected = True
    bridge._client = MagicMock()
    bridge.entry_gate = MagicMock(halted=False)

    # Initial snapshot has unprotected long
    clean_snap = {
        "open_positions": [("AVGO", 5.0)],
        "n_working_orders": 1,
        "protected_longs": ["AVGO"],
        "unprotected_longs": [],
        "duplicate_refs": [],
        "orphan_exits": [],
        "clean": True,
    }
    dirty_snap = {
        "open_positions": [("AVGO", 5.0)],
        "n_working_orders": 0,
        "protected_longs": [],
        "unprotected_longs": ["AVGO"],
        "duplicate_refs": [],
        "orphan_exits": [],
        "clean": False,
    }

    bridge.positions_plain = MagicMock(return_value=[{"symbol": "AVGO", "qty": 5.0}])
    bridge.working_orders_plain = MagicMock(return_value=[])
    bridge.ensure_protective_stops = MagicMock(return_value={"checked": 1, "unprotected": ["AVGO"], "repaired": ["AVGO"], "failed": []})

    with patch("reconciliation.build_snapshot", side_effect=[dirty_snap, clean_snap]):
        with patch("order_audit.log_event"):
            with patch("alerts.emit"):
                res = bridge.reconcile_startup_state()

    assert res["clean"] is True
    assert res["snapshot"]["clean"] is True
    assert res["snapshot"]["unprotected_longs"] == []
    # Verify positions_plain and working_orders_plain were re-queried
    assert bridge.positions_plain.call_count >= 2
    assert bridge.working_orders_plain.call_count >= 2


def test_ratchet_tiers_and_no_downgrade_telemetry():
    from position_rotator import PositionRotator, _RATCHET_RULES, OpenPosition
    rotator = PositionRotator()

    # Verify 5 monotonic tiers exist
    thresholds = [r[0] for r in _RATCHET_RULES]
    offsets = [r[1] for r in _RATCHET_RULES]
    assert 0.12 in thresholds
    assert 0.20 in thresholds
    assert offsets == sorted(offsets)

    # Verify +12% matches 7% lock
    mock_pos = OpenPosition(
        symbol="TEST", side="long", qty=10, avg_entry_price=100.0,
        current_price=112.50, unrealized_pnl=125.0, current_conf=0.60
    )
    mock_bridge = MagicMock()
    mock_bridge.get_current_stop.return_value = 104.0  # resting stop at +4%

    results = rotator._ratchet_stops([mock_pos], bridge=mock_bridge, dry_run=True)
    assert len(results) == 1
    assert results[0].rule_applied == "lock-in-7pct"
    assert results[0].new_stop == 107.0

    # Verify that when resting stop is higher (e.g. 110.0), it skips without downgrade
    mock_bridge.get_current_stop.return_value = 110.0
    results = rotator._ratchet_stops([mock_pos], bridge=mock_bridge, dry_run=True)
    assert results[0].skipped is True
    assert "no stop downgrade permitted" in results[0].reason


def test_swing_loss_guard_percentage_and_optional_dollar_cap():
    from alpaca_bridge import EntryGateMock
    bridge = AlpacaBridge(mode="swing")
    gate = EntryGateMock(bridge=bridge)

    bridge._start_of_day_equity = 100000.0
    bridge.get_net_liquidation = MagicMock(return_value=98800.0) # -$1,200 loss (-1.2%)

    # With default 1.0% limit ($1,000 max loss), -$1,200 should halt
    blocked, reason = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
    assert blocked is True
    assert "Max daily loss limit reached" in reason

    # Test with custom dollar cap
    gate.halted = False
    with patch("config.SWING_MAX_DAILY_LOSS_PCT", 0.02): # 2% = $2,000
        with patch("config.SWING_MAX_DAILY_LOSS_DOLLAR_CAP", 500.0): # Dollar cap $500 takes precedence
            blocked, reason = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
            assert blocked is True
            assert "500.00" in reason


def test_protected_symbols_configuration_and_rotator():
    import config
    assert isinstance(config.SWING_PROTECTED_SYMBOLS, set)
    assert "SPY" in config.SWING_PROTECTED_SYMBOLS

    bridge = AlpacaBridge(mode="swing")
    bridge._connected = True
    bridge._client = MagicMock()
    
    mock_spy = MagicMock(symbol="SPY", qty=10, avg_entry_price=500.0, current_price=510.0, unrealized_pl=100.0, side="long")
    mock_weak = MagicMock(symbol="WEAK", qty=10, avg_entry_price=100.0, current_price=95.0, unrealized_pl=-50.0, side="long")
    bridge._client.get_all_positions.return_value = [mock_spy, mock_weak]

    sig = MagicMock(symbol="STRONG", action="BUY", confidence=0.75, price=100.0)
    
    with patch.object(bridge.rotator, "evaluate") as mock_eval:
        mock_eval.return_value = MagicMock(should_rotate=False)
        bridge._evaluate_and_execute_rotation([sig])
        
        args, _ = mock_eval.call_args
        open_pos = args[0]
        spy_pos = next(p for p in open_pos if p.symbol == "SPY")
        weak_pos = next(p for p in open_pos if p.symbol == "WEAK")
        
        assert spy_pos.is_protected is True
        assert weak_pos.is_protected is False


def test_market_open_scheduler_alignment_calculation():
    from datetime import datetime, timezone, timedelta
    now_dt = datetime.now(timezone.utc)
    next_open_dt = now_dt + timedelta(minutes=10) # 600s

    mock_clock = MagicMock()
    mock_clock.is_open = False
    mock_clock.timestamp = now_dt
    mock_clock.next_open = next_open_dt

    interval = 1800.0 # 30 min
    sec_to_open = (mock_clock.next_open - mock_clock.timestamp).total_seconds()
    assert 0 < sec_to_open < interval
    
    sleep_duration = max(1.0, min(interval, sec_to_open + 5.0))
    assert sleep_duration == 605.0 # Exactly next_open + 5s


def test_swing_loss_guard_three_point_boundary_simulation():
    """Three-point boundary test for Swing Daily Loss Guard:
    Strict inequality `daily_pnl < -abs(max_loss)` is used.
    At SOD equity $100,308.00 and 1.0% limit, threshold is exactly $1,003.08:
      - Loss $1,003.07 (-1003.07): entries allowed (not halted)
      - Loss $1,003.08 (-1003.08): entries allowed (boundary condition)
      - Loss $1,003.09 (-1003.09): entries blocked (strictly exceeds threshold)
    """
    from alpaca_bridge import EntryGateMock
    bridge = AlpacaBridge(mode="swing")
    gate = EntryGateMock(bridge=bridge)
    bridge._start_of_day_equity = 100308.0

    # 1. Below boundary ($1,003.07 loss) -> Allowed
    bridge.get_net_liquidation = MagicMock(return_value=100308.0 - 1003.07)
    gate.halted = False
    blocked, _ = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
    assert blocked is False

    # 2. Exactly at boundary ($1,003.08 loss) -> Allowed by strict inequality
    bridge.get_net_liquidation = MagicMock(return_value=100308.0 - 1003.08)
    gate.halted = False
    blocked, _ = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
    assert blocked is False

    # 3. Beyond boundary ($1,003.09 loss) -> Blocked
    bridge.get_net_liquidation = MagicMock(return_value=100308.0 - 1003.09)
    gate.halted = False
    blocked, reason = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
    assert blocked is True
    assert "Max daily loss limit reached" in reason


def test_startup_security_validation():
    from strategies.startup_security import (
        StartupSafetyRejected,
        credential_fingerprint,
        verify_startup_safety_configuration,
        verify_authenticated_account_identity,
    )

    verification_key = b"test-verification-key-12345"
    key = "TEST_API_KEY"
    secret = "TEST_API_SECRET"
    fp = credential_fingerprint(key, secret, verification_key)

    # 1. Valid paper configuration passes
    verify_startup_safety_configuration(
        strategy_type="SWING",
        account_id="PA3R71I7JFDZ",
        api_key=key,
        api_secret=secret,
        expected_fingerprint=fp,
        credential_verification_key=verification_key,
        deployment_env="PAPER",
    )

    # 2. Tri-part identity verification passes
    verify_authenticated_account_identity(
        configured_account_id="PA3R71I7JFDZ",
        authenticated_api_account_id="PA3R71I7JFDZ",
        permitted_account_id="PA3R71I7JFDZ",
    )

    # 3. LIVE environment is strictly rejected (without leaking sensitive data)
    with pytest.raises(StartupSafetyRejected, match="STARTUP_ENVIRONMENT_PROHIBITED"):
        verify_startup_safety_configuration(
            strategy_type="SWING",
            account_id="PA3R71I7JFDZ",
            api_key=key,
            api_secret=secret,
            expected_fingerprint=fp,
            credential_verification_key=verification_key,
            deployment_env="LIVE",
        )

    # 4. Strategy/Account mismatch rejected
    with pytest.raises(StartupSafetyRejected, match="STARTUP_ACCOUNT_POLICY_MISMATCH"):
        verify_startup_safety_configuration(
            strategy_type="SWING",
            account_id="PA3UODIC5JTA",  # Daytrade account passed to swing
            api_key=key,
            api_secret=secret,
            expected_fingerprint=fp,
            credential_verification_key=verification_key,
            deployment_env="PAPER",
        )

    # 5. Fingerprint mismatch rejected
    with pytest.raises(StartupSafetyRejected, match="STARTUP_CREDENTIAL_SET_MISMATCH"):
        verify_startup_safety_configuration(
            strategy_type="SWING",
            account_id="PA3R71I7JFDZ",
            api_key=key,
            api_secret="WRONG_SECRET",
            expected_fingerprint=fp,
            credential_verification_key=verification_key,
            deployment_env="PAPER",
        )


def test_swing_sizing_and_tick_normalization():
    from strategies.swing_sizing import (
        normalize_long_stop_price,
        calculate_swing_shares,
        SwingSizingRejected,
    )

    # Tick normalization rounds down conservatively using decimal precision
    assert normalize_long_stop_price(150.258, price_increment=0.01) == 150.25
    assert normalize_long_stop_price(150.250, price_increment=0.01) == 150.25
    assert normalize_long_stop_price(150.255, price_increment=0.05) == 150.25
    assert normalize_long_stop_price(150.28, price_increment=0.05) == 150.25

    # 100k equity, 0.5% risk = $500 hard risk budget.
    # Pre-sizing headroom uses 99% ($495.00) minus $0.50 fees = $494.50
    # Entry $100.00, stop $95.00 -> base risk $5.00 + $0.06 slippage = $5.06/share -> 97 shares
    shares = calculate_swing_shares(
        entry_price=100.0,
        stop_price=95.0,
        account_equity=100000.0,
        proposed_risk_pct=0.005,
        symbol_notional_cap=12000.0,
    )
    assert shares == 97

    # Test notional cap clipping: cap $6,000 at $100 entry = 60 shares
    shares_capped = calculate_swing_shares(
        entry_price=100.0,
        stop_price=95.0,
        account_equity=100000.0,
        proposed_risk_pct=0.005,
        symbol_notional_cap=6000.0,
    )
    assert shares_capped == 60

    # Stop >= entry raises SwingSizingRejected
    with pytest.raises(SwingSizingRejected, match="Entry 100.0 must exceed normalized stop 105.0"):
        calculate_swing_shares(
            entry_price=100.0,
            stop_price=105.0,
            account_equity=100000.0,
        )


def test_swing_exit_lots_whole_share_policy():
    from strategies.swing_sizing import calculate_swing_exit_lots

    # 1 share: partial target disabled, 1 share runner
    lot_1 = calculate_swing_exit_lots(1, target_weight=0.50)
    assert lot_1["allow_partial_target"] is False
    assert lot_1["actual_partial_quantity"] == 0
    assert lot_1["runner_quantity"] == 1

    # 5 shares: 2 shares partial target (40%), 3 shares runner (60%)
    lot_5 = calculate_swing_exit_lots(5, target_weight=0.50)
    assert lot_5["allow_partial_target"] is True
    assert lot_5["actual_partial_quantity"] == 2
    assert lot_5["runner_quantity"] == 3

    # 10 shares: 5 shares partial target, 5 shares runner
    lot_10 = calculate_swing_exit_lots(10, target_weight=0.50)
    assert lot_10["allow_partial_target"] is True
    assert lot_10["actual_partial_quantity"] == 5
    assert lot_10["runner_quantity"] == 5


def test_validate_swing_order_safety_guards():
    from strategies.swing_sizing import validate_swing_order_safety, SwingSizingRejected

    base_args = dict(
        client_order_id="SWING-ORD-001",
        strategy_instance_id="SWING-v1.1",
        symbol="AVGO",
        signal_version="v1.1.3",
        entry_price=350.0,
        stop_price=340.0,
        shares=10,
        account_equity=100000.0,
        available_buying_power=50000.0,
        current_open_stop_risk=500.0,
        pending_entry_risk=0.0,
        proposed_risk_per_trade_pct=0.005,
        max_portfolio_heat_pct=0.025,
        current_cluster_risk_pct=0.005,
        max_cluster_risk_pct=0.0125,
        adv_shares_20=2_000_000.0,
        max_participation_rate=0.01,
        symbol_notional_cap=12000.0,
        quote_age_seconds=5.0,
        max_quote_age_seconds=60.0,
        entry_slippage_per_share=0.02,
        stop_slippage_per_share=0.04,
        estimated_total_fees=0.50,
    )

    # 1. Valid order passes
    validate_swing_order_safety(**base_args)

    # 2. Composite Idempotency Duplicate rejected
    intent_key = ("SWING-v1.1", "AVGO", "BUY", "v1.1.3", "SWING-ORD-001")
    bad_idempotent = dict(base_args, active_entry_intents={intent_key})
    with pytest.raises(SwingSizingRejected, match="Duplicate entry intent rejected"):
        validate_swing_order_safety(**bad_idempotent)

    # 3. Duplicate working symbol rejected
    bad_dup = dict(base_args, existing_working_symbols={"AVGO"})
    with pytest.raises(SwingSizingRejected, match="Duplicate active working order rejected"):
        validate_swing_order_safety(**bad_dup)

    # 4. Stale quote rejected
    bad_stale = dict(base_args, quote_age_seconds=95.0)
    with pytest.raises(SwingSizingRejected, match="Stale quote"):
        validate_swing_order_safety(**bad_stale)

    # 5. Effective trade risk strictly exceeding hard budget rejected (zero tolerance)
    # entry $100, stop $90, 50 shares -> notional $5,000 <= $12,000
    # 50 shares * ($10 distance + $0.06 slippage) = $503 + $0.50 = $503.50 > $500 hard limit
    bad_risk = dict(base_args, entry_price=100.0, stop_price=90.0, shares=50)
    with pytest.raises(SwingSizingRejected, match="Effective trade risk .* strictly exceeds hard budget"):
        validate_swing_order_safety(**bad_risk)

    # 6. Excessive portfolio heat rejected
    bad_heat = dict(base_args, entry_price=100.0, stop_price=90.0, shares=10, current_open_stop_risk=2450.0) # $2450 + ~$101 > $2500 cap
    with pytest.raises(SwingSizingRejected, match="Projected portfolio heat"):
        validate_swing_order_safety(**bad_heat)


def test_regime_classifier_exhaustive_coverage():
    from strategies.regime_classifier import classify_market_regime, MarketRegime

    # 1. Precedence 1: DATA_INVALID when inputs missing or non-finite
    invalid_data = classify_market_regime(adx=float("nan"), vix=15.0, spy_price=500.0, spy_sma50=500.0, sma50_slope=0.5)
    assert invalid_data.regime == MarketRegime.DATA_INVALID
    assert invalid_data.allows_new_entries is False

    stale_data = classify_market_regime(adx=28.0, vix=15.0, spy_price=500.0, spy_sma50=500.0, sma50_slope=0.5, quote_age_seconds=90.0)
    assert stale_data.regime == MarketRegime.DATA_INVALID
    assert stale_data.allows_new_entries is False

    # 2. Precedence 2: Boundary Cross Override (>=3 regimes in last 5 bars)
    recent_shifts = [MarketRegime.SIDEWAYS_LOW_VOL, MarketRegime.TRENDING_BULL, MarketRegime.SIDEWAYS_NORMAL_VOL]
    shift_res = classify_market_regime(adx=28.0, vix=15.0, spy_price=510.0, spy_sma50=500.0, sma50_slope=0.5, recent_regimes=recent_shifts)
    assert shift_res.regime == MarketRegime.TRANSITIONING
    assert "Boundary cross override" in shift_res.reason

    # 3. Precedence 4: Confirmed Bull
    bull = classify_market_regime(adx=28.0, vix=15.0, spy_price=510.0, spy_sma50=500.0, sma50_slope=0.5)
    assert bull.regime == MarketRegime.TRENDING_BULL
    assert bull.allows_new_entries is True
    assert bull.is_sustained_bear is False

    # 4. Precedence 5: Confirmed Bear (<10 sessions)
    bear = classify_market_regime(adx=30.0, vix=22.0, spy_price=480.0, spy_sma50=500.0, sma50_slope=-0.5, consecutive_bear_sessions=3)
    assert bear.regime == MarketRegime.TRENDING_BEAR
    assert bear.allows_new_entries is False
    assert bear.consecutive_bear_sessions == 4

    # 5. Sustained Bear (>=10 sessions)
    sustained = classify_market_regime(adx=32.0, vix=28.0, spy_price=470.0, spy_sma50=500.0, sma50_slope=-0.8, consecutive_bear_sessions=9)
    assert sustained.regime == MarketRegime.TRENDING_BEAR
    assert sustained.is_sustained_bear is True
    assert sustained.consecutive_bear_sessions == 10

    # 6. Conflicting Trend -> Transitioning
    conflict = classify_market_regime(adx=28.0, vix=16.0, spy_price=510.0, spy_sma50=500.0, sma50_slope=-0.2)
    assert conflict.regime == MarketRegime.TRANSITIONING

    # 7. Sideways Regimes
    low_vol = classify_market_regime(adx=17.0, vix=14.0, spy_price=500.0, spy_sma50=500.0, sma50_slope=0.0)
    assert low_vol.regime == MarketRegime.SIDEWAYS_LOW_VOL

    norm_vol = classify_market_regime(adx=18.0, vix=21.0, spy_price=500.0, spy_sma50=500.0, sma50_slope=0.0)
    assert norm_vol.regime == MarketRegime.SIDEWAYS_NORMAL_VOL

    high_vol = classify_market_regime(adx=19.0, vix=29.0, spy_price=500.0, spy_sma50=500.0, sma50_slope=0.0)
    assert high_vol.regime == MarketRegime.SIDEWAYS_HIGH_VOL

    # 8. Intermediate ADX Zone (20 < ADX < 25) -> Transitioning
    mid_adx = classify_market_regime(adx=22.5, vix=17.0, spy_price=505.0, spy_sma50=500.0, sma50_slope=0.2)
    assert mid_adx.regime == MarketRegime.TRANSITIONING


def test_rotator_net_replacement_advantage_and_r_protection():
    from position_rotator import (
        calculate_net_replacement_advantage,
        PositionRotator,
        OpenPosition,
        IncomingSignal,
    )

    # 1. Test Hurdle Math:
    # EV_new = 0.50R, EV_current = 0.10R -> raw delta = 0.40R
    # replacement_cost = 0.05R -> delta after cost = 0.35R
    # z * SE = 1.645 * 0.05 = 0.08225R -> lower conf = 0.26775R >= 0.20R (PASS)
    adv, should_rotate = calculate_net_replacement_advantage(
        ev_new_r=0.50,
        ev_current_r=0.10,
        replacement_cost_r=0.05,
        se_ev_diff=0.05,
    )
    assert should_rotate is True
    assert adv > 0.20

    # 2. Marginal delta that fails hurdle:
    # EV_new = 0.30R, EV_current = 0.15R -> raw delta = 0.15R -> fails hurdle
    adv_fail, should_rotate_fail = calculate_net_replacement_advantage(
        ev_new_r=0.30,
        ev_current_r=0.15,
        replacement_cost_r=0.05,
        se_ev_diff=0.05,
    )
    assert should_rotate_fail is False

    # 3. Test R-based Profit Acceleration Protection in candidate filtering:
    rotator = PositionRotator()
    # Position with 1.2R floating profit should be skipped from candidate pool
    runner = OpenPosition(
        symbol="MSFT",
        side="long",
        qty=4,
        avg_entry_price=450.0,
        current_price=470.0,
        unrealized_pnl=80.0,
        unrealized_r=1.2, # Protected!
        current_conf=0.45, # Weak conf, but protected by acceleration
    )
    candidates = rotator._get_rotation_candidates([runner])
    assert len(candidates) == 0 # Skipped!



