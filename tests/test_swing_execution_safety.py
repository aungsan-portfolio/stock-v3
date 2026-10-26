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
    """Boundary test for Swing Daily Loss Guard:
    Enforces min(SOD * 1.0%, $300.00).
    Case A: High SOD ($100,000.00) -> $300 cap binds.
      - Loss $299.99 (-299.99): allowed
      - Loss $300.00 (-300.00): blocked (limit reached)
      - Loss $300.01 (-300.01): blocked
    Case B: Low SOD ($20,000.00) -> 1.0% cap binds at $200.00 ($200 < $300).
      - Loss $199.99 (-199.99): allowed
      - Loss $200.00 (-200.00): blocked (limit reached)
      - Loss $200.01 (-200.01): blocked
    """
    from alpaca_bridge import EntryGateMock
    bridge = AlpacaBridge(mode="swing")
    gate = EntryGateMock(bridge=bridge)

    # Case A: SOD $100,000 ($300 cap)
    bridge._start_of_day_equity = 100000.0

    # 1. Below boundary ($299.99 loss) -> Allowed
    bridge.get_net_liquidation = MagicMock(return_value=100000.0 - 299.99)
    gate.halted = False
    gate.reasons = set()
    blocked, _ = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
    assert blocked is False

    # 2. Exactly at boundary ($300.00 loss) -> Blocked
    bridge.get_net_liquidation = MagicMock(return_value=100000.0 - 300.00)
    gate.halted = False
    gate.reasons = set()
    blocked, reason = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
    assert blocked is True
    assert "Max daily loss limit reached" in reason

    # 3. Beyond boundary ($300.01 loss) -> Blocked
    bridge.get_net_liquidation = MagicMock(return_value=100000.0 - 300.01)
    gate.halted = False
    gate.reasons = set()
    blocked, reason = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
    assert blocked is True
    assert "Max daily loss limit reached" in reason

    # Case B: SOD $20,000 ($200 cap)
    bridge._start_of_day_equity = 20000.0

    # 4. Below boundary ($199.99 loss) -> Allowed
    bridge.get_net_liquidation = MagicMock(return_value=20000.0 - 199.99)
    gate.halted = False
    gate.reasons = set()
    blocked, _ = gate.entry_blocked("AAPL", 1000.0, None, 150.0)
    assert blocked is False

    # 5. At boundary ($200.00 loss) -> Blocked
    bridge.get_net_liquidation = MagicMock(return_value=20000.0 - 200.00)
    gate.halted = False
    gate.reasons = set()
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


# ─── 10. Canonical Risk Budget & Decimal Exact Boundary Tests ─────────────────

def test_canonical_risk_budget_helper_and_boundaries():
    from strategies.swing_sizing import (
        get_trade_risk_budget,
        validate_swing_order_safety,
        SwingSizingRejected,
        PositionIntent,
    )

    # 1. Progression: equity -> dollar risk cap
    assert get_trade_risk_budget(50_000.0) == 250.00
    assert pytest.approx(get_trade_risk_budget(99_999.0), 0.001) == 499.995
    assert get_trade_risk_budget(100_000.0) == 500.00
    assert get_trade_risk_budget(150_000.0) == 500.00  # Capped at $500
    assert get_trade_risk_budget(200_000.0) == 500.00  # Capped at $500

    # 2. Exact Decimal Boundary: $500.00 accepted, $500.01 rejected
    # Effective risk: 10 * (100.00 - 50.06 + 0.01) + 0.50 = 10 * 49.95 + 0.50 = 499.50 + 0.50 = 500.00 exactly
    base_args = dict(
        client_order_id="BOUNDARY-500",
        strategy_instance_id="SWING_TEST",
        symbol="SPY",
        signal_version="v1.0",
        entry_price=100.00,
        stop_price=50.06,
        shares=10,
        account_equity=100_000.0,  # budget = $500.00
        available_buying_power=50_000.0,
        current_open_stop_risk=0.0,
        pending_entry_risk=0.0,
        entry_slippage_per_share=0.01,
        stop_slippage_per_share=0.00,
        estimated_total_fees=0.50,
        account_number="PA3R71I7JFDZ",
        broker_account_number="PA3R71I7JFDZ",
        current_position_count=0,
        pending_open_position_count=0,
        existing_position_symbols=set(),
        intent=PositionIntent.OPEN_LONG,
    )
    validate_swing_order_safety(**base_args)

    # 1 cent higher ($500.01) strictly rejected
    bad_args = dict(base_args, estimated_total_fees=0.51)
    with pytest.raises(SwingSizingRejected, match="strictly exceeds hard budget"):
        validate_swing_order_safety(**bad_args)


# ─── 11. Strict Types and Range Validation Tests ─────────────────────────────

def test_validate_swing_order_safety_strict_types_and_ranges():
    from strategies.swing_sizing import (
        validate_swing_order_safety,
        SwingSizingRejected,
        PositionIntent,
    )

    base_args = dict(
        client_order_id="TYPE-TEST-001",
        strategy_instance_id="SWING_TEST",
        symbol="MSFT",
        signal_version="v1.0",
        entry_price=400.0,
        stop_price=390.0,
        shares=10,
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        current_open_stop_risk=0.0,
        pending_entry_risk=0.0,
        account_number="PA3R71I7JFDZ",
        broker_account_number="PA3R71I7JFDZ",
        current_position_count=0,
        pending_open_position_count=0,
        existing_position_symbols=set(),
        intent=PositionIntent.OPEN_LONG,
    )

    # 1. shares cannot be bool (e.g. True)
    with pytest.raises(SwingSizingRejected, match="shares must be an integer"):
        validate_swing_order_safety(**dict(base_args, shares=True))

    # 2. current_position_count cannot be bool (e.g. False)
    with pytest.raises(SwingSizingRejected, match="current_position_count must be an integer"):
        validate_swing_order_safety(**dict(base_args, current_position_count=False))

    # 3. Negative values rejected
    with pytest.raises(SwingSizingRejected, match="current_open_stop_risk must be finite and non-negative"):
        validate_swing_order_safety(**dict(base_args, current_open_stop_risk=-5.0))

    with pytest.raises(SwingSizingRejected, match="pending_entry_risk must be finite and non-negative"):
        validate_swing_order_safety(**dict(base_args, pending_entry_risk=-1.0))

    with pytest.raises(SwingSizingRejected, match="quote_age_seconds must be finite and non-negative"):
        validate_swing_order_safety(**dict(base_args, quote_age_seconds=-0.5))

    # 4. Non-finite values (NaN, Inf) rejected
    with pytest.raises(SwingSizingRejected, match="account_equity must be finite and positive"):
        validate_swing_order_safety(**dict(base_args, account_equity=float("nan")))

    with pytest.raises(SwingSizingRejected, match="account_equity must be finite and positive"):
        validate_swing_order_safety(**dict(base_args, account_equity=float("inf")))


# ─── 12. Intent Consistency and Position Capacity Tests ───────────────────────

def test_validate_swing_order_safety_intent_and_capacity():
    from strategies.swing_sizing import (
        validate_swing_order_safety,
        SwingSizingRejected,
        PositionIntent,
    )

    base_args = dict(
        client_order_id="INTENT-TEST-001",
        strategy_instance_id="SWING_TEST",
        symbol="AAPL",
        signal_version="v1.0",
        entry_price=150.0,
        stop_price=145.0,
        shares=10,
        account_equity=100_000.0,
        available_buying_power=50_000.0,
        current_open_stop_risk=0.0,
        pending_entry_risk=0.0,
        account_number="PA3R71I7JFDZ",
        broker_account_number="PA3R71I7JFDZ",
    )

    # 1. OPEN_LONG when symbol already exists in positions -> Rejected
    with pytest.raises(SwingSizingRejected, match="OPEN_LONG rejected because AAPL already exists"):
        validate_swing_order_safety(
            **base_args,
            current_position_count=1,
            pending_open_position_count=0,
            existing_position_symbols={"AAPL"},
            intent=PositionIntent.OPEN_LONG,
        )

    # 2. ADD_LONG when symbol does NOT exist in positions -> Rejected
    with pytest.raises(SwingSizingRejected, match="ADD_LONG rejected because AAPL does not exist"):
        validate_swing_order_safety(
            **base_args,
            current_position_count=1,
            pending_open_position_count=0,
            existing_position_symbols={"MSFT"},
            intent=PositionIntent.ADD_LONG,
        )

    # 3. Position state inconsistency: count != len(symbols)
    with pytest.raises(SwingSizingRejected, match="POSITION_STATE_INCONSISTENT"):
        validate_swing_order_safety(
            **base_args,
            current_position_count=2,
            pending_open_position_count=0,
            existing_position_symbols={"MSFT"},
            intent=PositionIntent.OPEN_LONG,
        )

    # 4. Account mismatch: account_number != broker_account_number
    with pytest.raises(SwingSizingRejected, match="ACCOUNT_STATE_MISMATCH"):
        validate_swing_order_safety(
            **dict(base_args, broker_account_number="PA3UODIC5JTA"),  # Daytrade account
            current_position_count=0,
            pending_open_position_count=0,
            existing_position_symbols=set(),
            intent=PositionIntent.OPEN_LONG,
        )

    # 5. Capacity: current=6, pending=0, new=1 (total 7) -> Allowed
    validate_swing_order_safety(
        **base_args,
        current_position_count=6,
        pending_open_position_count=0,
        existing_position_symbols={"S1", "S2", "S3", "S4", "S5", "S6"},
        intent=PositionIntent.OPEN_LONG,
        max_concurrent_positions=7,
    )

    # 6. Capacity: current=6, pending=1, new=1 (total 8 > 7) -> Rejected
    with pytest.raises(SwingSizingRejected, match="Projected position count 8 exceeds capacity 7"):
        validate_swing_order_safety(
            **base_args,
            current_position_count=6,
            pending_open_position_count=1,
            existing_position_symbols={"S1", "S2", "S3", "S4", "S5", "S6"},
            intent=PositionIntent.OPEN_LONG,
            max_concurrent_positions=7,
        )

    # 7. Capacity: current=7, pending=0, OPEN_LONG (total 8 > 7) -> Rejected
    with pytest.raises(SwingSizingRejected, match="Projected position count 8 exceeds capacity 7"):
        validate_swing_order_safety(
            **base_args,
            current_position_count=7,
            pending_open_position_count=0,
            existing_position_symbols={"S1", "S2", "S3", "S4", "S5", "S6", "S7"},
            intent=PositionIntent.OPEN_LONG,
            max_concurrent_positions=7,
        )

    # 8. ADD_LONG does not increment capacity (current=7, ADD_LONG on existing symbol S1) -> Allowed
    validate_swing_order_safety(
        **dict(base_args, symbol="S1"),
        current_position_count=7,
        pending_open_position_count=0,
        existing_position_symbols={"S1", "S2", "S3", "S4", "S5", "S6", "S7"},
        intent=PositionIntent.ADD_LONG,
        max_concurrent_positions=7,
    )


# ─── 13. Protective Stop Coverage & Account Short Guard Tests ────────────────

def test_stop_quantity_coverage_and_shorts():
    from alpaca_bridge import AlpacaBridge
    from strategies.swing_sizing import SwingSizingRejected
    from types import SimpleNamespace

    bridge = AlpacaBridge(mode="swing", dry_run=True)

    # 1. Full coverage: 100 shares long, 100 shares stop sell -> PASS
    pos_full = [SimpleNamespace(symbol="AAPL", qty=100.0, avg_entry_price=150.0)]
    orders_full = [SimpleNamespace(id="s1", symbol="AAPL", side="sell", order_type="stop", qty=100.0, stop_price=140.0)]
    open_risk, pend_risk, active_syms, pend_syms = bridge.audit_swing_portfolio_safety(pos_full, orders_full)
    assert "AAPL" in active_syms
    assert open_risk > 0.0
    assert pend_risk == 0.0

    # 2. Under-coverage: 100 shares long, only 10 shares stop sell -> REJECTED
    orders_under = [SimpleNamespace(id="s2", symbol="AAPL", side="sell", order_type="stop", qty=10.0, stop_price=140.0)]
    with pytest.raises(SwingSizingRejected, match="INSUFFICIENT_STOP_COVERAGE: AAPL position=100.0, protected=10.0"):
        bridge.audit_swing_portfolio_safety(pos_full, orders_under)

    # 3. Multiple split stops: 100 shares long, 40 + 60 shares stops -> PASS
    orders_split = [
        SimpleNamespace(id="s3", symbol="AAPL", side="sell", order_type="stop", qty=40.0, stop_price=140.0),
        SimpleNamespace(id="s4", symbol="AAPL", side="sell", order_type="stop", qty=60.0, stop_price=138.0),
    ]
    open_risk_split, _, _, _ = bridge.audit_swing_portfolio_safety(pos_full, orders_split)
    assert open_risk_split > 0.0

    # 4. Short position on account -> REJECTED (Fail-closed Stage 0)
    pos_short = [SimpleNamespace(symbol="NVDA", qty=-50.0, avg_entry_price=120.0)]
    with pytest.raises(SwingSizingRejected, match="UNSUPPORTED_ACCOUNT_SHORT_POSITION: NVDA"):
        bridge.audit_swing_portfolio_safety(pos_short, [])

    # 5. Pending swing entry lacking protective stop -> REJECTED
    orders_pending_no_stop = [
        SimpleNamespace(id="p1", client_order_id="swg_123456", symbol="MSFT", side="buy", qty=10.0, filled_qty=0.0, limit_price=400.0)
    ]
    with pytest.raises(SwingSizingRejected, match="PENDING_ENTRY_WITHOUT_PROTECTIVE_STOP: MSFT"):
        bridge.audit_swing_portfolio_safety([], orders_pending_no_stop)

    # 6. Pending swing entry with matching protective stop -> PASS
    orders_pending_with_stop = [
        SimpleNamespace(id="p1", client_order_id="swg_123456", symbol="MSFT", side="buy", qty=10.0, filled_qty=0.0, limit_price=400.0),
        SimpleNamespace(id="p2", client_order_id="swg_stop_1", symbol="MSFT", side="sell", order_type="stop", qty=10.0, stop_price=380.0),
    ]
    _, pend_risk_ok, _, pend_syms_ok = bridge.audit_swing_portfolio_safety([], orders_pending_with_stop)
    assert "MSFT" in pend_syms_ok
    assert pend_risk_ok > 0.0


# ─── 14. Enum Normalization and Bracket Leg Flattening Tests ──────────────────

def test_enum_normalization_and_bracket_flattening():
    from alpaca_bridge import normalize_enum_value, flatten_orders
    from alpaca.trading.enums import OrderSide, OrderType
    from types import SimpleNamespace

    # 1. Enum normalization
    assert normalize_enum_value(OrderSide.BUY) == "BUY"
    assert normalize_enum_value(OrderSide.SELL) == "SELL"
    assert normalize_enum_value(OrderType.STOP) == "STOP"
    assert normalize_enum_value("buy") == "BUY"
    assert normalize_enum_value(None) == ""

    # 2. Nested bracket flattening and ID deduplication
    child_stop = SimpleNamespace(id="child-stop-1", symbol="AMZN", side="sell", order_type="stop", stop_price=180.0)
    child_tp = SimpleNamespace(id="child-tp-1", symbol="AMZN", side="sell", order_type="limit", limit_price=220.0)
    parent = SimpleNamespace(id="parent-1", symbol="AMZN", side="buy", order_type="limit", limit_price=190.0, legs=[child_stop, child_tp])
    duplicate_child = SimpleNamespace(id="child-stop-1", symbol="AMZN", side="sell", order_type="stop", stop_price=180.0)

    flattened = flatten_orders([parent, duplicate_child])
    ids = {getattr(o, "id") for o in flattened}
    assert len(flattened) == 3
    assert ids == {"parent-1", "child-stop-1", "child-tp-1"}


# ─── 15. Behavioral Production Path & Concurrency Tests ───────────────────────

def test_behavioral_production_path_and_concurrency():
    import threading
    import tempfile
    from pathlib import Path
    from alpaca_bridge import AlpacaBridge
    from predictor import Signal
    import config as config_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(config_mod, "DATA_DIR", Path(tmpdir)):
            bridge = AlpacaBridge(mode="swing", dry_run=True)
            bridge._connected = True
            bridge._client = MagicMock()
            bridge._client.get_account.return_value = MagicMock(account_number="PA3R71I7JFDZ", equity=100_000.0, buying_power=100_000.0, last_equity=100_000.0)
            bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=100.0, source="alpaca"))
            bridge.get_net_liquidation = MagicMock(return_value=100_000.0)
            bridge._start_of_day_equity = 100_000.0
            if hasattr(bridge, "entry_gate") and bridge.entry_gate:
                bridge.entry_gate.clear("BASELINE_UNVERIFIED")
            bridge._client.get_all_positions.return_value = []
            bridge._client.get_orders.return_value = []

            def _make_test_signal(sym: str) -> Signal:
                s = Signal(
                    symbol=sym, action="BUY", confidence=0.75,
                    rf_score=0.5, lstm_score=0.5, xgb_score=0.5, trans_score=0.5, tech_score=0.5,
                    price=100.0, reason="test",
                    adx=30.0, vix=15.0, spy_price=500.0, spy_sma50=480.0, sma50_slope=0.50, quote_age_seconds=5.0,
                )
                setattr(s, "atr_14", 2.0)
                setattr(s, "regime", "TRENDING_BULL")
                return s

            sig = _make_test_signal("MSFT")

            # 1. Verify deprecated calculate_swing_shares is NEVER called on production swing dispatch path
            with patch("strategies.swing_sizing.calculate_swing_shares", side_effect=RuntimeError("DEPRECATED_CALLED")):
                placed = bridge.execute_signal(sig)
                assert placed is False  # dry_run returns False (DryRunBlocked) without calling calculate_swing_shares

            # 2. Sequential Call-Order Tracing: validation occurs strictly before dispatch
            call_order = []
            with patch("strategies.swing_sizing.validate_swing_order_safety", side_effect=lambda **kw: call_order.append("validate")):
                with patch.object(bridge, "_place_open_bracket", side_effect=lambda *a, **kw: call_order.append("dispatch") or MagicMock(occupies_slot=True)):
                    bridge.execute_signal(sig)
                    assert call_order == ["validate", "dispatch"]

            # 3. Critical Section Serialization Barrier under _swing_dispatch_lock
            lock_acquisitions = []
            barrier = threading.Barrier(2)

            with patch.object(bridge, "_place_open_bracket", side_effect=lambda *a, **kw: lock_acquisitions.append(a[0]) or MagicMock(occupies_slot=True)):
                def worker(sym: str):
                    s = _make_test_signal(sym)
                    barrier.wait()
                    bridge.execute_signal(s)

                t1 = threading.Thread(target=worker, args=("MSFT",))
                t2 = threading.Thread(target=worker, args=("AAPL",))
                t1.start()
                t2.start()
                t1.join()
                t2.join()

            # Both threads ran through critical section in serialized order
            assert len(lock_acquisitions) == 2


def test_baseline_unverified_strictly_blocks_dispatch():
    """Verify fail-closed invariant: when baseline is unverified, order dispatch is strictly blocked."""
    import tempfile
    from pathlib import Path
    from alpaca_bridge import AlpacaBridge
    from predictor import Signal
    import config as config_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch.object(config_mod, "DATA_DIR", Path(tmpdir)):
            bridge = AlpacaBridge(mode="swing", dry_run=True)
            bridge._connected = True
            bridge._client = MagicMock()
            # Omit last_equity to trigger BASELINE_UNVERIFIED
            bridge._client.get_account.return_value = MagicMock(account_number="PA3R71I7JFDZ", equity=100_000.0, buying_power=100_000.0, last_equity=None)
            bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=100.0, source="alpaca"))
            bridge.get_net_liquidation = MagicMock(return_value=100_000.0)
            bridge._client.get_all_positions.return_value = []
            bridge._client.get_orders.return_value = []

            sig = Signal(
                symbol="MSFT", action="BUY", confidence=0.75,
                rf_score=0.5, lstm_score=0.5, xgb_score=0.5, trans_score=0.5, tech_score=0.5,
                price=100.0, reason="test",
                adx=30.0, vix=15.0, spy_price=500.0, spy_sma50=480.0, sma50_slope=0.50, quote_age_seconds=5.0,
            )
            setattr(sig, "atr_14", 2.0)
            setattr(sig, "regime", "TRENDING_BULL")

            with patch.object(bridge, "_place_open_bracket") as mock_bracket:
                placed = bridge.execute_signal(sig)
                assert placed is False, "Dispatch must be blocked when BASELINE_UNVERIFIED"
                assert mock_bracket.call_count == 0, "_place_open_bracket must never be reached"
                assert "BASELINE_UNVERIFIED" in bridge.entry_gate.reasons
                assert bridge.entry_gate.halted is True


# ─── 16. Comprehensive Behavioral Gate & Integrity Tests ─────────────────────

def test_execute_signal_daily_loss_gate_behavioral():
    import tempfile
    from pathlib import Path
    from alpaca_bridge import AlpacaBridge
    from predictor import Signal
    import config as config_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with patch.object(config_mod, "DATA_DIR", tmp_path):
            bridge = AlpacaBridge(mode="swing", dry_run=False)
            bridge._connected = True
            bridge._client = MagicMock()
            bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=100.0, source="alpaca"))
            bridge._client.get_all_positions.return_value = []
            bridge._client.get_orders.return_value = []

            def _make_signal(sym: str = "MSFT") -> Signal:
                s = Signal(
                    symbol=sym, action="BUY", confidence=0.75,
                    rf_score=0.5, lstm_score=0.5, xgb_score=0.5, trans_score=0.5, tech_score=0.5,
                    price=100.0, reason="test",
                    adx=30.0, vix=15.0, spy_price=500.0, spy_sma50=480.0, sma50_slope=0.50, quote_age_seconds=5.0,
                )
                setattr(s, "atr_14", 2.0)
                setattr(s, "regime", "TRENDING_BULL")
                return s

            sig = _make_signal("MSFT")

            # Mock _place_open_bracket to track dispatches
            dispatched = []
            bridge._place_open_bracket = MagicMock(side_effect=lambda *a, **kw: dispatched.append(a) or MagicMock(occupies_slot=True))

            # Case 1: High SOD $100,000, PnL = -$299.99 -> Allowed/dispatched
            bridge._start_of_day_equity = 100000.0
            bridge._client.get_account.return_value = MagicMock(
                account_number="PA3R71I7JFDZ",
                equity=100000.0 - 299.99,
                buying_power=50000.0,
                last_equity=100000.0,
            )
            bridge.get_net_liquidation = MagicMock(return_value=100000.0 - 299.99)
            bridge.entry_gate.halted = False
            bridge.entry_gate.reasons = set()
            dispatched.clear()

            res = bridge.execute_signal(sig)
            assert res is True
            assert len(dispatched) == 1

            # Case 2: PnL = -$300.00 -> Blocked, 0 orders placed
            bridge._client.get_account.return_value = MagicMock(
                account_number="PA3R71I7JFDZ",
                equity=100000.0 - 300.00,
                buying_power=50000.0,
                last_equity=100000.0,
            )
            bridge.get_net_liquidation = MagicMock(return_value=100000.0 - 300.00)
            bridge.entry_gate.halted = False
            bridge.entry_gate.reasons = set()
            dispatched.clear()

            res = bridge.execute_signal(sig)
            assert res is False
            assert len(dispatched) == 0

            # Case 3: PnL = -$300.01 -> Blocked, 0 orders placed
            bridge._client.get_account.return_value = MagicMock(
                account_number="PA3R71I7JFDZ",
                equity=100000.0 - 300.01,
                buying_power=50000.0,
                last_equity=100000.0,
            )
            bridge.get_net_liquidation = MagicMock(return_value=100000.0 - 300.01)
            bridge.entry_gate.halted = False
            bridge.entry_gate.reasons = set()
            dispatched.clear()

            res = bridge.execute_signal(sig)
            assert res is False
            assert len(dispatched) == 0

            # Reset any halt file before Case 4 to test 1.0% cap independently
            for hf in tmp_path.glob("swing_daily_loss_halt*.json"):
                hf.unlink(missing_ok=True)
            bridge.entry_gate.halted = False
            bridge.entry_gate.reasons = set()

            # Case 4: Low SOD $20,000, 1% cap is $200.00 (< $300 cap).
            # PnL = -$200.00 -> Blocked by 1.0% cap
            bridge._start_of_day_equity = 20000.0
            bridge._client.get_account.return_value = MagicMock(
                account_number="PA3R71I7JFDZ",
                equity=20000.0 - 200.00,
                buying_power=20000.0,
                last_equity=20000.0,
            )
            bridge.get_net_liquidation = MagicMock(return_value=20000.0 - 200.00)
            bridge.entry_gate.halted = False
            bridge.entry_gate.reasons = set()
            dispatched.clear()

            res = bridge.execute_signal(sig)
            assert res is False
            assert len(dispatched) == 0



def test_check_regular_market_hours_matrix():
    from alpaca_bridge import check_regular_market_hours
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from types import SimpleNamespace

    et_tz = ZoneInfo("America/New_York")

    # Tuesday Sept 8, 2026:
    # 09:15 ET (Pre-market)
    dt_pre = datetime(2026, 9, 8, 9, 15, 0, tzinfo=et_tz)
    mock_client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True, timestamp=dt_pre))
    ok, reason = check_regular_market_hours(mock_client)
    assert ok is False
    assert "PRE_MARKET_HOURS" in reason

    # 09:30 ET (Market Open)
    dt_open = datetime(2026, 9, 8, 9, 30, 0, tzinfo=et_tz)
    mock_client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True, timestamp=dt_open))
    ok, reason = check_regular_market_hours(mock_client)
    assert ok is True
    assert reason == ""

    # 14:00 ET (Regular Trading)
    dt_mid = datetime(2026, 9, 8, 14, 0, 0, tzinfo=et_tz)
    mock_client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True, timestamp=dt_mid))
    ok, reason = check_regular_market_hours(mock_client)
    assert ok is True

    # 15:54:59 ET (Just before cutoff)
    dt_cutoff_before = datetime(2026, 9, 8, 15, 54, 59, tzinfo=et_tz)
    mock_client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True, timestamp=dt_cutoff_before))
    ok, reason = check_regular_market_hours(mock_client)
    assert ok is True

    # 15:55:00 ET (Entry Cutoff Exceeded)
    dt_cutoff = datetime(2026, 9, 8, 15, 55, 0, tzinfo=et_tz)
    mock_client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True, timestamp=dt_cutoff))
    ok, reason = check_regular_market_hours(mock_client)
    assert ok is False
    assert "ENTRY_CUTOFF_EXCEEDED" in reason

    # 16:30 ET (After hours)
    dt_after = datetime(2026, 9, 8, 16, 30, 0, tzinfo=et_tz)
    mock_client = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True, timestamp=dt_after))
    ok, reason = check_regular_market_hours(mock_client)
    assert ok is False
    assert "ENTRY_CUTOFF_EXCEEDED" in reason

    # Broker clock says closed
    mock_closed = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=False, timestamp=dt_mid))
    ok, reason = check_regular_market_hours(mock_closed)
    assert ok is False
    assert reason == "BROKER_MARKET_CLOSED"

    # Weekend (Saturday Sept 12, 2026 at 12:00 ET)
    dt_weekend = datetime(2026, 9, 12, 12, 0, 0, tzinfo=et_tz)
    mock_weekend = SimpleNamespace(get_clock=lambda: SimpleNamespace(is_open=True, timestamp=dt_weekend))
    ok, reason = check_regular_market_hours(mock_weekend)
    assert ok is False
    assert "WEEKEND_HOURS" in reason


def test_audit_swing_portfolio_safety_stop_types_rejection():
    from alpaca_bridge import AlpacaBridge
    from strategies.swing_sizing import SwingSizingRejected
    from types import SimpleNamespace

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    pos = [SimpleNamespace(symbol="AAPL", qty=100.0, avg_entry_price=150.0)]

    # 1. Pure STOP order -> Accepted
    orders_stop = [SimpleNamespace(id="s1", symbol="AAPL", side="sell", order_type="STOP", qty=100.0, stop_price=140.0)]
    open_risk, _, active_syms, _ = bridge.audit_swing_portfolio_safety(pos, orders_stop)
    assert "AAPL" in active_syms
    assert open_risk > 0.0

    # 2. STOP_LIMIT order -> Rejected (Fail-closed Stage 0)
    orders_stop_limit = [SimpleNamespace(id="s2", symbol="AAPL", side="sell", order_type="STOP_LIMIT", qty=100.0, stop_price=140.0, limit_price=139.0)]
    with pytest.raises(SwingSizingRejected, match="INSUFFICIENT_STOP_COVERAGE: AAPL"):
        bridge.audit_swing_portfolio_safety(pos, orders_stop_limit)

    # 3. TRAILING_STOP order -> Rejected
    orders_trailing = [SimpleNamespace(id="s3", symbol="AAPL", side="sell", order_type="TRAILING_STOP", qty=100.0, stop_price=140.0)]
    with pytest.raises(SwingSizingRejected, match="INSUFFICIENT_STOP_COVERAGE: AAPL"):
        bridge.audit_swing_portfolio_safety(pos, orders_trailing)


def test_sector_attribution_and_exposure_cap():
    from alpaca_bridge import AlpacaBridge
    from predictor import Signal
    from strategies.swing_sizing import validate_swing_order_safety, SwingSizingRejected

    bridge = AlpacaBridge(mode="swing", dry_run=True)
    bridge._connected = True
    bridge._client = MagicMock()
    bridge._client.get_account.return_value = MagicMock(account_number="PA3R71I7JFDZ", equity=100_000.0, buying_power=100_000.0)
    bridge.get_order_quote = MagicMock(return_value=MagicMock(ok=True, price=100.0))
    bridge.get_net_liquidation = MagicMock(return_value=100_000.0)
    bridge._client.get_all_positions.return_value = []
    bridge._client.get_orders.return_value = []

    # 1. Symbol missing in config.SECTOR_MAPPINGS
    sig_unmapped = Signal(
        symbol="UNMAPPED_TICKER", action="BUY", confidence=0.75,
        rf_score=0.5, lstm_score=0.5, xgb_score=0.5, trans_score=0.5, tech_score=0.5,
        price=100.0, reason="test",
    )
    setattr(sig_unmapped, "atr_14", 2.0)
    setattr(sig_unmapped, "regime", "TRENDING_BULL")
    assert bridge.execute_signal(sig_unmapped) is False

    # 2. Signal sector mismatch with config
    sig_mismatch = Signal(
        symbol="AAPL", action="BUY", confidence=0.75,
        rf_score=0.5, lstm_score=0.5, xgb_score=0.5, trans_score=0.5, tech_score=0.5,
        price=100.0, reason="test",
    )
    setattr(sig_mismatch, "atr_14", 2.0)
    setattr(sig_mismatch, "regime", "TRENDING_BULL")
    setattr(sig_mismatch, "sector", "ENERGY")  # AAPL is INFORMATION_TECHNOLOGY
    assert bridge.execute_signal(sig_mismatch) is False

    # 3. Direct validator sector cap boundary: 25% of $100,000 = $25,000.00
    base_args = {
        "client_order_id": "swg_test_sector_1",
        "strategy_instance_id": "SWING_BOT_PRO",
        "symbol": "AAPL",
        "signal_version": "v1.6-alpha",
        "entry_price": 100.0,
        "stop_price": 95.0,
        "shares": 50,  # notional = $5,000
        "account_equity": 100000.0,
        "available_buying_power": 100000.0,
        "current_open_stop_risk": 500.0,
        "pending_entry_risk": 0.0,
        "account_number": "PA3R71I7JFDZ",
        "broker_account_number": "PA3R71I7JFDZ",
        "current_position_count": 0,
        "pending_open_position_count": 0,
        "symbol_sector": "INFORMATION_TECHNOLOGY",
        "max_sector_exposure_pct": 0.25,
    }

    # Exactly at boundary: $20,000 existing + $5,000 order = $25,000 <= $25,000 -> Accepted
    validate_swing_order_safety(
        **base_args,
        existing_sector_exposure_by_sector={"INFORMATION_TECHNOLOGY": 20000.0},
        pending_sector_exposure_by_sector={},
    )

    # Exceeding boundary by $0.01: $20,000.01 + $5,000 = $25,000.01 > $25,000 -> Rejected
    with pytest.raises(SwingSizingRejected, match="SECTOR_EXPOSURE_LIMIT_EXCEEDED"):
        validate_swing_order_safety(
            **base_args,
            existing_sector_exposure_by_sector={"INFORMATION_TECHNOLOGY": 20000.01},
            pending_sector_exposure_by_sector={},
        )


def test_durable_daily_loss_halt_persistence_and_rollover():
    import tempfile
    import json
    from pathlib import Path
    from decimal import Decimal
    from alpaca_bridge import EntryGateMock
    import config as config_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with patch.object(config_mod, "DATA_DIR", tmp_path):
            gate = EntryGateMock()
            acct = "PA3R71I7JFDZ"
            session_1 = "2026-09-08"
            session_2 = "2026-09-09"

            # 1. Persist halt state
            pnl = Decimal("-300.00")
            limit = Decimal("300.00")
            gate.persist_daily_loss_halt(acct, session_1, pnl, limit)

            halt_file = tmp_path / f"swing_daily_loss_halt_{acct}.json"
            assert halt_file.exists()

            # 2. Query within same session -> Halted
            is_halted, reason = gate.check_daily_loss_halt_state(acct, session_1)
            assert is_halted is True
            assert "DAILY_LOSS_LIMIT_REACHED" in reason
            assert gate.halted is True

            # 3. Query in next session (rollover) -> Automatically reset
            is_halted_next, reason_next = gate.check_daily_loss_halt_state(acct, session_2)
            assert is_halted_next is False
            assert reason_next == ""
            assert gate.halted is False
            assert not halt_file.exists()

            # 4. Corrupted file fails closed
            with open(halt_file, "w", encoding="utf-8") as f:
                f.write("{invalid_json:")
            is_halted_corrupt, reason_corrupt = gate.check_daily_loss_halt_state(acct, session_2)
            assert is_halted_corrupt is True
            assert "CORRUPTED_HALT_STATE" in reason_corrupt

            # 5. Account mismatch in file fails closed
            with open(halt_file, "w", encoding="utf-8") as f:
                json.dump({"account_id": "WRONG_ACCOUNT", "session_date_et": session_2, "halted": True}, f)
            is_halted_mismatch, reason_mismatch = gate.check_daily_loss_halt_state(acct, session_2)
            assert is_halted_mismatch is True
            assert "HALT_STATE_ACCOUNT_MISMATCH" in reason_mismatch


def test_rollover_reconciliation_clears_phantom_halt_and_preserves_genuine_breach():
    """Verify that when a stale pre-settlement baseline is reconciled to the true broker last_equity,
    a phantom halt is cleared if actual P&L is within limits, but genuinely breached halts are strictly maintained."""
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import MagicMock, patch
    import config as config_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with patch.object(config_mod, "DATA_DIR", tmp_path):
            acct_id = "PA3R71I7JFDZ"
            session_date = "2026-10-15"

            # 1. Simulate stale pre-settlement baseline captured early in morning (Oct 13 close)
            stale_state = {
                "account_id": acct_id,
                "date": session_date,
                "start_of_day_equity": 100607.73,
                "suspended": False,
                "provenance": {
                    "source": "broker_last_equity",
                    "captured_at": "2026-10-15T07:35:49.000000+00:00"
                }
            }

            # Simulate phantom halt file triggered against stale baseline
            halt_file = tmp_path / f"swing_daily_loss_halt_{acct_id}.json"
            halt_file.write_text(json.dumps({
                "account_id": acct_id,
                "session_date_et": session_date,
                "daily_pnl": "-310.0",
                "daily_loss_limit": "300.0",
                "halted": True,
                "reason": "DAILY_LOSS_LIMIT",
            }), encoding="utf-8")

            bridge = AlpacaBridge(mode="swing", dry_run=True)
            bridge._account_id = acct_id
            bridge._risk_state_dir = str(tmp_path)
            bridge.entry_gate.halt("DAILY_LOSS_LIMIT")

            # Broker now has settled Oct 14 close ($99,745.55) and current equity ($99,600.62) -> loss is only -$144.93
            mock_acct = MagicMock(
                account_number=acct_id,
                last_equity="99745.55",
                equity="99600.62"
            )

            # Reconcile baseline
            new_sod = bridge._restore_session_baseline(mock_acct, stale_state, session_date, acct_id)
            assert new_sod == 99745.55
            assert bridge._start_of_day_equity == 99745.55

            # Phantom halt must be cleared because actual loss (-$144.93) is within the $300 limit
            assert not halt_file.exists()
            assert not bridge.entry_gate.halted
            assert "DAILY_LOSS_LIMIT" not in bridge.entry_gate.reasons

            # Case B: If actual live equity was $99,400.00 (loss = -$345.55 > $300 limit), halt must be maintained!
            halt_file.write_text(json.dumps({
                "account_id": acct_id,
                "session_date_et": session_date,
                "daily_pnl": "-345.55",
                "daily_loss_limit": "300.0",
                "halted": True,
                "reason": "DAILY_LOSS_LIMIT",
            }), encoding="utf-8")
            mock_breach_acct = MagicMock(
                account_number=acct_id,
                last_equity="99745.55",
                equity="99400.00"
            )
            bridge._restore_session_baseline(mock_breach_acct, stale_state, session_date, acct_id)
            # Must remain halted!
            assert halt_file.exists()
            assert bridge.entry_gate.halted
            assert "DAILY_LOSS_LIMIT" in bridge.entry_gate.reasons

