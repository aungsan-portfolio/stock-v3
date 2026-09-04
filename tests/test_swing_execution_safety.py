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

