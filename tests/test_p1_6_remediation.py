"""
test_p1_6_remediation.py — Targeted verification for Friday night surgical hotfixes:
1. Decoupling stale bar delay from 30m quarantine.
2. Unviable signals (failing confidence) do not claim symbols or cannibalize viable candidates.
"""
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

import config
from strategies.constants import StrategyName
from strategies.base import TradeSignal
from strategies.intraday_data import _validate_intraday_data, is_quarantined
from strategies.scanner.premarket_scanner import clear_quarantine
from strategies.regime_engine import MarketRegimeResult, MODE_RISK_ON


def test_stale_bar_skips_without_quarantine():
    """Verify that a bar exceeding grace period returns empty df but does NOT quarantine symbol."""
    clear_quarantine()
    ts_comp = pd.Timestamp("2026-09-04 15:20:00", tz="UTC")
    ts_form = pd.Timestamp("2026-09-04 15:25:00", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1000],
        "atr": [1.5, 1.5]
    }, index=[ts_comp, ts_form])

    # 70s post-close (grace_seconds is 45s)
    now_utc = pd.Timestamp("2026-09-04 15:26:10", tz="UTC")
    with patch("strategies.session.is_market_open", return_value=True), \
         patch("pandas.Timestamp.now", return_value=now_utc):
        result = _validate_intraday_data(df, symbol="JPM", interval="5m", validate_staleness=True, purpose="ENTRY_SIGNAL")
        assert result.empty, "Stale bar must return empty DataFrame for this cycle"
        assert not is_quarantined("JPM"), "Stale bar must NOT quarantine the symbol for 30 minutes"


def test_max_acceptable_spread_paper_tolerance():
    """Verify MAX_ACCEPTABLE_SPREAD_PCT is calibrated to at least 0.50% (0.75%)."""
    assert config.MAX_ACCEPTABLE_SPREAD_PCT >= 0.50
    assert config.MAX_ACCEPTABLE_SPREAD_PCT == 0.75


def test_unviable_candidate_confidence_gates():
    """Verify that in RISK_ON regime, unviable SHORT (0.594 < 0.78) fails gate while viable BUY (0.650 >= 0.60) passes."""
    from strategies.order_manager import _min_confidence_for

    regime_res = MarketRegimeResult(
        mode=MODE_RISK_ON,
        score=3,
        allow_new_longs=True,
        requires_opening_buffer=False,
        position_scale=1.0,
        confidence_boost=0.0,
    )

    sig_vwap = TradeSignal(
        strategy=StrategyName.VWAP_BOUNCE,
        symbol="AMD",
        side="SELL",
        entry_price=100.0,
        stop_price=101.0,
        target_price=98.0,
        confidence=0.594,
        atr=1.5,
        risk_per_share=1.0,
        reason="VWAP bounce short",
    )

    sig_mr = TradeSignal(
        strategy=StrategyName.MEAN_REVERSION,
        symbol="AMD",
        side="BUY",
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        confidence=0.650,
        atr=1.5,
        risk_per_share=1.0,
        reason="Mean reversion buy",
    )

    min_vwap = _min_confidence_for(sig_vwap, regime_result=regime_res)
    min_mr = _min_confidence_for(sig_mr, regime_result=regime_res)

    assert min_vwap == 0.78
    assert min_mr == 0.60
    assert sig_vwap.confidence < min_vwap, "VWAP SHORT must be recognized as unviable"
    assert sig_mr.confidence >= min_mr, "MR BUY must be recognized as viable"


def test_client_order_id_fallback_uniqueness():
    """Verify that when bar_timestamp is missing, fallback ID is unique across days and candle buckets."""
    from strategies.order_idempotency import build_client_order_id

    # Same day & bucket -> deterministic
    id1 = build_client_order_id("SPY", "BUY", bar_timestamp_utc="", session_date="20260904")
    id2 = build_client_order_id("SPY", "BUY", bar_timestamp_utc="", session_date="20260904")
    assert id1 == id2
    assert id1.startswith("se3-SPY-B-ord-20260904-")

    # Different session date -> unique
    id_diff_day = build_client_order_id("SPY", "BUY", bar_timestamp_utc="", session_date="20260905")
    assert id1 != id_diff_day

    # Different side -> unique
    id_sell = build_client_order_id("SPY", "SELL", bar_timestamp_utc="", session_date="20260904")
    assert id1 != id_sell


def test_422_broker_order_recovery_adoption():
    """Verify 422 client_order_id must be unique recovers existing order from broker without error."""
    from strategies.order_manager import execute_signal, idempotency_manager
    from unittest.mock import MagicMock

    sig = TradeSignal(
        strategy=StrategyName.VWAP_BOUNCE,
        symbol="BAC",
        side="BUY",
        entry_price=62.70,
        stop_price=62.61,
        target_price=62.94,
        confidence=0.70,
        atr=0.10,
        risk_per_share=0.09,
        reason="test_bac",
    )

    bridge = MagicMock()
    bridge.is_connected = True
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.get_price = MagicMock(return_value=62.69)
    quote_mock = MagicMock()
    quote_mock.last = 62.69
    bridge.get_order_quote = MagicMock(return_value=quote_mock)

    # First submit throws 422 duplicate client_order_id
    bridge.place_bracket_order = MagicMock(side_effect=Exception('{"code":42210000,"message":"client_order_id must be unique"}'))
    # Broker lookup recovers the existing order
    bridge.get_order_by_client_id = MagicMock(return_value={
        "id": "order-recov-123",
        "order_id": "order-recov-123",
        "status": "new",
        "qty": 63,
        "stop_order_id": "stop-recov-123",
        "tp_order_id": "tp-recov-123",
    })

    idempotency_manager.release_after_reconciliation("BAC", side="BUY", reason="test_init")

    with patch("strategies.order_manager.calculate_shares", return_value=63), \
         patch("strategies.execution_quote.validate_execution_quote", return_value=MagicMock(valid=True, quote=MagicMock(last=62.69))), \
         patch.object(config, "SPY_TREND_GUARD_ENABLED", False):
        res = execute_signal(signal=sig, bridge=bridge, equity=100000.0, current_pnl=0.0, dry_run=False)
        assert res["status"] == "PLACED"
        assert res["order"]["id"] == "order-recov-123"
        bridge.get_order_by_client_id.assert_called_once()
        idempotency_manager.release_after_reconciliation("BAC", side="BUY", reason="test_cleanup")


def test_422_broker_order_not_found_clean_release():
    """Verify 422 error where order is NOT found on broker releases idempotency cleanly."""
    from strategies.order_manager import execute_signal, idempotency_manager
    from unittest.mock import MagicMock

    sig = TradeSignal(
        strategy=StrategyName.VWAP_BOUNCE,
        symbol="JPM",
        side="BUY",
        entry_price=358.75,
        stop_price=358.24,
        target_price=360.74,
        confidence=0.70,
        atr=0.50,
        risk_per_share=0.51,
        reason="test_jpm",
    )

    bridge = MagicMock()
    bridge.is_connected = True
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.place_bracket_order = MagicMock(side_effect=Exception('{"code":42210000,"message":"client_order_id must be unique"}'))
    bridge.get_order_by_client_id = MagicMock(return_value=None)

    idempotency_manager.release("JPM", side="BUY", reason="test_init")

    with patch("strategies.order_manager.calculate_shares", return_value=11), \
         patch("strategies.execution_quote.validate_execution_quote", return_value=MagicMock(valid=True, quote=MagicMock(last=358.75))), \
         patch.object(config, "SPY_TREND_GUARD_ENABLED", False):
        res = execute_signal(signal=sig, bridge=bridge, equity=100000.0, current_pnl=0.0, dry_run=False)
        assert res["status"] == "REJECTED"
        # Verify reservation was released so symbol is not locked out
        is_reserved, _ = idempotency_manager.reserve("JPM", side="BUY")
        assert is_reserved is True
        idempotency_manager.release("JPM", side="BUY", reason="test_cleanup")


def test_vwap_stop_normalization_above_atr_floor():
    """Verify VWAP strategy normalizes ultra-tight stop distance to >= 0.75x ATR."""
    from strategies.vwap_strategy import VWAPBounceStrategy

    strat = VWAPBounceStrategy()
    dates = pd.date_range("2026-09-04 09:30", periods=40, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [769.50] * 40,
        "high": [770.50] * 40,
        "low": [769.53] * 40,
        "close": [769.55] * 40,
        "volume": [10000] * 40,
        "vwap": [769.50] * 40,
        "vwap_lower": [769.00] * 40,
        "vwap_upper": [770.80] * 40,
        "rsi": [45.0] * 40,
        "atr": [0.12] * 40,
        "volume_ma": [5000] * 40,
    }, index=dates)

    sig = strat.evaluate("SPY", df)
    if sig:
        stop_dist = abs(sig.entry_price - sig.stop_price)
        min_expected_floor = round(0.12 * 0.75, 4)
        assert stop_dist >= (min_expected_floor - 1e-4), f"Stop dist ${stop_dist} must be >= ATR floor ${min_expected_floor}"
        assert sig.risk_per_share >= (min_expected_floor - 1e-4)


def test_vwap_target_rr_clamping_and_rejection():
    """Verify VWAP strategy clamps target to <= MAX_RR and returns None if reward < MIN_RR."""
    from strategies.vwap_strategy import VWAPBounceStrategy

    strat = VWAPBounceStrategy()
    dates = pd.date_range("2026-09-04 09:30", periods=40, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0] * 40,
        "high": [101.0] * 40,
        "low": [99.5] * 40,
        "close": [100.0] * 40,
        "volume": [10000] * 40,
        "vwap": [100.0] * 40,
        "vwap_lower": [99.0] * 40,
        "vwap_upper": [100.01] * 40,
        "rsi": [45.0] * 40,
        "atr": [1.0] * 40,
        "volume_ma": [5000] * 40,
    }, index=dates)

    sig = strat.evaluate("TEST", df)
    if sig:
        rr = abs(sig.target_price - sig.entry_price) / abs(sig.entry_price - sig.stop_price)
        assert rr <= float(config.MAX_TARGET_RR_RATIO)
        assert rr >= float(config.MIN_RR_RATIO)


def test_client_id_same_bucket_different_intent_distinct_ids():
    """Verify distinct signal intents in the same 5-minute bucket produce distinct client order IDs."""
    from strategies.order_idempotency import build_client_order_id

    # Intent 1 vs Intent 2 for same symbol, side, strategy in same 5m bucket (bar_ts omitted)
    id_1 = build_client_order_id(
        symbol="AAPL",
        side="BUY",
        bar_timestamp_utc="",
        signal_hash="hash_alpha_111",
        intent_nonce="intent-uuid-1",
        strategy="VWAP_BOUNCE",
    )
    id_2 = build_client_order_id(
        symbol="AAPL",
        side="BUY",
        bar_timestamp_utc="",
        signal_hash="hash_beta_222",
        intent_nonce="intent-uuid-2",
        strategy="VWAP_BOUNCE",
    )
    assert id_1 != id_2, f"Distinct intents in same bucket must not collide: {id_1} vs {id_2}"


def test_client_id_same_intent_retry_identical_id():
    """Verify transport retry attempt increment retains exact client order ID while strategy re-entry alters it."""
    from strategies.order_idempotency import build_client_order_id

    id_first = build_client_order_id(
        symbol="AAPL",
        side="BUY",
        bar_timestamp_utc="",
        signal_hash="hash_same_123",
        intent_nonce="intent-uuid-same",
        strategy="VWAP_BOUNCE",
        strategy_entry_attempt=0,
        submission_retry_attempt=0,
    )
    id_retry_1 = build_client_order_id(
        symbol="AAPL",
        side="BUY",
        bar_timestamp_utc="",
        signal_hash="hash_same_123",
        intent_nonce="intent-uuid-same",
        strategy="VWAP_BOUNCE",
        strategy_entry_attempt=0,
        submission_retry_attempt=1,  # Transport submission retry attempt 1
    )
    id_retry_2 = build_client_order_id(
        symbol="AAPL",
        side="BUY",
        bar_timestamp_utc="",
        signal_hash="hash_same_123",
        intent_nonce="intent-uuid-same",
        strategy="VWAP_BOUNCE",
        strategy_entry_attempt=0,
        submission_retry_attempt=2,  # Transport submission retry attempt 2
    )
    assert id_first == id_retry_1, "Transport retry attempt 1 must retain identical client order ID"
    assert id_first == id_retry_2, "Transport retry attempt 2 must retain identical client order ID"

    # Deliberate strategy re-entry or scale-in must produce distinct ID
    id_reentry = build_client_order_id(
        symbol="AAPL",
        side="BUY",
        bar_timestamp_utc="",
        signal_hash="hash_same_123",
        intent_nonce="intent-uuid-same",
        strategy="VWAP_BOUNCE",
        strategy_entry_attempt=1,
        submission_retry_attempt=0,
    )
    assert id_first != id_reentry, "Strategy re-entry attempt must produce a distinct client order ID"


def test_422_order_visible_on_second_lookup_is_adopted():
    """Verify 422 recovery with eventual consistency: order visible on 2nd attempt is successfully adopted."""
    from strategies.order_manager import execute_signal, idempotency_manager
    from unittest.mock import MagicMock

    sig = TradeSignal(
        strategy=StrategyName.VWAP_BOUNCE,
        symbol="BAC",
        side="BUY",
        entry_price=62.70,
        stop_price=62.61,
        target_price=62.94,
        confidence=0.70,
        atr=0.10,
        risk_per_share=0.09,
        reason="test_bac_retry",
    )

    bridge = MagicMock()
    bridge.is_connected = True
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.get_price = MagicMock(return_value=62.69)
    quote_mock = MagicMock()
    quote_mock.last = 62.69
    bridge.get_order_quote = MagicMock(return_value=quote_mock)

    bridge.place_bracket_order = MagicMock(side_effect=Exception('{"code":42210000,"message":"client_order_id must be unique"}'))
    
    # First lookup returns None (broker lag), second lookup returns recovered order
    lookup_results = [
        None,
        {
            "id": "order-recov-attempt-2",
            "order_id": "order-recov-attempt-2",
            "status": "new",
            "qty": 63,
            "stop_order_id": "stop-attempt-2",
            "tp_order_id": "tp-attempt-2",
        },
    ]
    bridge.get_order_by_client_id = MagicMock(side_effect=lookup_results)

    idempotency_manager.release_after_reconciliation("BAC", side="BUY", reason="test_init")

    with patch("strategies.order_manager.calculate_shares", return_value=63), \
         patch("strategies.execution_quote.validate_execution_quote", return_value=MagicMock(valid=True, quote=MagicMock(last=62.69))), \
         patch.object(config, "SPY_TREND_GUARD_ENABLED", False):
        res = execute_signal(signal=sig, bridge=bridge, equity=100000.0, current_pnl=0.0, dry_run=False)
        assert res["status"] == "PLACED"
        assert res["order"]["id"] == "order-recov-attempt-2"
        assert bridge.get_order_by_client_id.call_count == 2
        idempotency_manager.release_after_reconciliation("BAC", side="BUY", reason="test_cleanup")


def test_vwap_spread_floor_can_exceed_atr_floor():
    """Verify VWAP geometry normalization enforces spread floor (1.5x spread) when it exceeds 0.75x ATR."""
    from strategies.vwap_strategy import VWAPBounceStrategy

    strat = VWAPBounceStrategy()
    dates = pd.date_range("2026-09-04 09:30", periods=40, freq="5min", tz="UTC")
    # Small ATR = 0.10 -> 0.75 * ATR = 0.075
    # Wide Spread = 0.12 -> 1.5 * Spread = 0.18 > 0.075
    df = pd.DataFrame({
        "open": [100.0] * 40,
        "high": [100.5] * 40,
        "low": [99.95] * 40,  # structural low very close to price
        "close": [100.0] * 40,
        "volume": [10000] * 40,
        "vwap": [100.0] * 40,
        "vwap_lower": [99.95] * 40,
        "vwap_upper": [102.00] * 40,
        "rsi": [45.0] * 40,
        "atr": [0.10] * 40,
        "volume_ma": [5000] * 40,
    }, index=dates)

    sig = strat.evaluate("TEST", df, spread_dollars=0.12)
    if sig:
        stop_dist = abs(sig.entry_price - sig.stop_price)
        expected_spread_floor = round(0.12 * 1.5, 4)
        assert stop_dist >= (expected_spread_floor - 1e-4), f"Stop dist ${stop_dist} must reflect spread floor ${expected_spread_floor}"


def test_invalid_quote_never_reaches_geometry_normalization():
    """Verify invalid/vetoed execution quote is vetoed immediately and never modifies order geometry."""
    from strategies.order_manager import execute_signal
    from unittest.mock import MagicMock

    sig = TradeSignal(
        strategy=StrategyName.VWAP_BOUNCE,
        symbol="MSFT",
        side="BUY",
        entry_price=400.0,
        stop_price=398.0,
        target_price=404.0,
        confidence=0.75,
        atr=2.0,
        risk_per_share=2.0,
        reason="test_quote_veto",
    )

    bridge = MagicMock()
    bridge.is_connected = True
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.get_price = MagicMock(return_value=400.0)
    bridge.place_bracket_order = MagicMock()

    with patch("strategies.order_manager.calculate_shares", return_value=10), \
         patch("strategies.execution_quote.validate_execution_quote", return_value=MagicMock(valid=False, reason="Spread 1.25% exceeds ceiling 0.75%")), \
         patch.object(config, "SPY_TREND_GUARD_ENABLED", False):
        res = execute_signal(signal=sig, bridge=bridge, equity=100000.0, current_pnl=0.0, dry_run=False)
        assert res["status"] == "VETOED"
        assert "Execution Quote Gate" in res["reason"]
        bridge.place_bracket_order.assert_not_called()


def test_long_reconciliation_never_loosens_broker_stop():
    """Verify long position reconciliation adheres to stop monotonicity: effective_stop = max(broker, desired)."""
    import time
    from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState

    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="BAC",
        side="BUY",
        entry_price=62.68,
        peak_price=62.68,
        atr=0.10,
        stop_price=62.65,
        trail_multiple=1.5,
        last_updated=time.time(),
        desired_stop_price=62.65,  # Tighter local stop
        qty=63.0,
        strategy="RECONCILED",
        active=True,
    )
    mgr.states["BAC"] = state

    # Broker has looser stop ($62.60 < $62.65)
    broker_orders = [{
        "id": "stop-order-1",
        "symbol": "BAC",
        "side": "sell",
        "type": "stop",
        "stop_price": 62.60,
        "qty": 63,
        "status": "new",
    }]

    mgr._reconcile_broker_order(state, broker_orders)
    # Long stop must NOT loosen down to $62.60
    assert state.desired_stop_price == 62.65, f"Desired stop should remain at tightest value 62.65, got {state.desired_stop_price}"


def test_short_reconciliation_never_loosens_broker_stop():
    """Verify short position reconciliation adheres to stop monotonicity: effective_stop = min(broker, desired)."""
    import time
    from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState

    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="AAPL",
        side="SELL",
        entry_price=220.0,
        peak_price=220.0,
        atr=1.0,
        stop_price=220.50,
        trail_multiple=1.5,
        last_updated=time.time(),
        desired_stop_price=220.50,  # Tighter local stop (lower for short)
        qty=50.0,
        strategy="RECONCILED",
        active=True,
    )
    mgr.states["AAPL"] = state

    # Broker has looser stop ($221.00 > $220.50)
    broker_orders = [{
        "id": "stop-order-short-1",
        "symbol": "AAPL",
        "side": "buy",
        "type": "stop",
        "stop_price": 221.00,
        "qty": 50,
        "status": "new",
    }]

    mgr._reconcile_broker_order(state, broker_orders)
    # Short stop must NOT loosen up to $221.00
    assert state.desired_stop_price == 220.50, f"Desired stop should remain at tightest value 220.50, got {state.desired_stop_price}"


def test_broker_stop_must_match_position_side_and_quantity():
    """Verify _is_valid_stop_order rejects orders with mismatched side or insufficient quantity."""
    from strategies.trailing_stop import DynamicTrailingStopManager

    mgr = DynamicTrailingStopManager()

    # Case 1: Wrong side (BUY order when SELL is needed to protect long position)
    wrong_side_order = {
        "id": "ord-1",
        "symbol": "BAC",
        "side": "buy",
        "type": "stop",
        "stop_price": 62.60,
        "qty": 100,
        "status": "new",
    }
    assert mgr._is_valid_stop_order(wrong_side_order, target_side="sell", required_qty=100) is False

    # Case 2: Insufficient quantity (50 shares to protect 100 shares position)
    under_qty_order = {
        "id": "ord-2",
        "symbol": "BAC",
        "side": "sell",
        "type": "stop",
        "stop_price": 62.60,
        "qty": 50,
        "status": "new",
    }
    assert mgr._is_valid_stop_order(under_qty_order, target_side="sell", required_qty=100) is False

    # Case 3: Valid matching order
    valid_order = {
        "id": "ord-3",
        "symbol": "BAC",
        "side": "sell",
        "type": "stop",
        "stop_price": 62.60,
        "qty": 100,
        "status": "new",
    }
    assert mgr._is_valid_stop_order(valid_order, target_side="sell", required_qty=100) is True

    # Case 4: Mismatched symbol (AAPL stop order when protecting BAC position)
    wrong_symbol_order = {
        "id": "ord-4",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "stop_price": 62.60,
        "qty": 100,
        "status": "new",
    }
    assert mgr._is_valid_stop_order(wrong_symbol_order, target_side="sell", required_qty=100, expected_symbol="BAC") is False
    assert mgr._is_valid_stop_order(valid_order, target_side="sell", required_qty=100, expected_symbol="BAC") is True


def test_labor_day_2026_market_holiday_handling():
    """Verify Labor Day (2026-09-07) is detected as closed and Tuesday (2026-09-08) is an active market day."""
    from datetime import date, datetime
    from unittest.mock import patch, MagicMock
    from strategies.session import is_market_day, _nyse_holidays, EASTERN_TZ
    import main

    # 1. Labor Day calendar detection
    labor_day_2026 = date(2026, 9, 7)
    tuesday_2026 = date(2026, 9, 8)

    assert is_market_day(labor_day_2026) is False, "Labor Day 2026 must be recognized as market closed"
    assert is_market_day(tuesday_2026) is True, "Tuesday post-Labor Day must be an active trading day"
    assert labor_day_2026 in _nyse_holidays(2026), "Labor Day 2026 must be present in NYSE holidays list"

    # 2. Verify cmd_daytrade_bot cleanly exits with code 0 on holiday without starting scanner
    fake_labor_day_now = datetime(2026, 9, 7, 9, 0, 0, tzinfo=EASTERN_TZ)
    with patch("strategies.session.now_eastern", return_value=fake_labor_day_now), \
         patch("alpaca_bridge.AlpacaBridge") as MockBridge, \
         patch("preflight.run_preflight_audit") as mock_preflight, \
         patch("strategies.scanner.premarket_scanner.scan") as mock_scan:

        mock_bridge_inst = MagicMock()
        mock_bridge_inst.connect.return_value = True
        mock_bridge_inst.reconcile_startup_state.return_value = {"halt_new_entries": False}
        MockBridge.return_value = mock_bridge_inst

        mock_pf_report = MagicMock()
        mock_pf_report.status = "GO"
        mock_preflight.return_value = mock_pf_report

        ret_code = main.cmd_daytrade_bot(args=MagicMock())
        assert ret_code == 0, f"cmd_daytrade_bot must return 0 on holiday, got {ret_code}"
        mock_scan.assert_not_called()


def test_vwap_geometry_numerical_safety_invariants():
    """Release Gate Test: Verify all 5 numerical safety invariants in VWAP strategy:
    1. Structure-first target validation (rejection of insufficient structural reward without force-extension).
    2. Pre-rounding vs post-rounding risk consistency (risk_per_share matches actual rounded stop distance).
    3. Direction-aware tick rounding (long stop floors down, short stop ceils up, never breaches noise floor).
    4. Negative structural risk rejection (no masking of inverted structure).
    5. Positivity and ordering invariant: 0 < stop < price < target (BUY) and 0 < target < price < stop (SELL).
    """
    from strategies.vwap_strategy import VWAPBounceStrategy

    strat = VWAPBounceStrategy()
    dates = pd.date_range("2026-09-04 09:30", periods=40, freq="5min", tz="UTC")

    # --- CASE 1: Valid BUY Signal (All invariants hold) ---
    df_buy = pd.DataFrame({
        "open": [100.0] * 40,
        "high": [100.5] * 40,
        "low": [99.70] * 40,  # pullback_low = 99.70
        "close": [100.0] * 40,
        "volume": [10000] * 40,
        "vwap": [100.0] * 40,
        "vwap_lower": [99.60] * 40,
        "vwap_upper": [101.50] * 40,
        "rsi": [45.0] * 40,
        "atr": [0.50] * 40,  # Noise floor = 0.75 * 0.50 = 0.375
        "volume_ma": [5000] * 40,
    }, index=dates)

    sig_buy = strat.evaluate("TEST_BUY", df_buy)
    assert sig_buy is not None, "Valid BUY setup must generate a signal"
    assert sig_buy.side == "BUY"

    # Invariant 5: Positivity and ordering
    assert 0 < sig_buy.stop_price < sig_buy.entry_price < sig_buy.target_price, (
        f"Long ordering breached: stop={sig_buy.stop_price}, entry={sig_buy.entry_price}, target={sig_buy.target_price}"
    )

    # Invariant 2: Pre-rounding vs post-rounding consistency
    actual_buy_risk = round(sig_buy.entry_price - sig_buy.stop_price, 2)
    assert sig_buy.risk_per_share == pytest.approx(actual_buy_risk, abs=1e-4), (
        f"risk_per_share ({sig_buy.risk_per_share}) must match actual stop distance ({actual_buy_risk})"
    )

    # Invariant 3: Direction-aware floor rounding preserves noise floor
    expected_atr_floor = 0.50 * 0.75  # 0.375
    assert sig_buy.risk_per_share >= (expected_atr_floor - 1e-4), (
        f"Risk per share ({sig_buy.risk_per_share}) fell below noise floor ({expected_atr_floor})"
    )

    # Invariant 1: Structural R:R between min_rr and max_rr
    actual_buy_rr = round(sig_buy.target_price - sig_buy.entry_price, 2) / sig_buy.risk_per_share
    assert actual_buy_rr >= (float(config.MIN_RR_RATIO) - 1e-4)
    assert actual_buy_rr <= float(config.MAX_TARGET_RR_RATIO)

    # --- CASE 2: Valid SELL Signal (All invariants hold) ---
    df_sell = pd.DataFrame({
        "open": [100.0] * 40,
        "high": [100.30] * 40,  # rally_high = 100.30
        "low": [99.50] * 40,
        "close": [100.0] * 40,
        "volume": [10000] * 40,
        "vwap": [100.0] * 40,
        "vwap_lower": [98.50] * 40,
        "vwap_upper": [100.40] * 40,
        "rsi": [75.0] * 40,  # High RSI for sell rejection
        "atr": [0.50] * 40,
        "volume_ma": [5000] * 40,
    }, index=dates)

    with patch.object(config, "ALLOW_SHORT", True):
        sig_sell = strat.evaluate("TEST_SELL", df_sell)
        assert sig_sell is not None, "Valid SELL setup must generate a signal"
        assert sig_sell.side == "SELL"

        # Invariant 5: Positivity and ordering
        assert 0 < sig_sell.target_price < sig_sell.entry_price < sig_sell.stop_price, (
            f"Short ordering breached: target={sig_sell.target_price}, entry={sig_sell.entry_price}, stop={sig_sell.stop_price}"
        )

        # Invariant 2: Pre-rounding vs post-rounding consistency
        actual_sell_risk = round(sig_sell.stop_price - sig_sell.entry_price, 2)
        assert sig_sell.risk_per_share == pytest.approx(actual_sell_risk, abs=1e-4), (
            f"risk_per_share ({sig_sell.risk_per_share}) must match actual stop distance ({actual_sell_risk})"
        )

        # Invariant 3: Direction-aware ceil rounding preserves noise floor
        assert sig_sell.risk_per_share >= (expected_atr_floor - 1e-4)

        # Invariant 1: Structural R:R between min_rr and max_rr
        actual_sell_rr = round(sig_sell.entry_price - sig_sell.target_price, 2) / sig_sell.risk_per_share
        assert actual_sell_rr >= (float(config.MIN_RR_RATIO) - 1e-4)
        assert actual_sell_rr <= float(config.MAX_TARGET_RR_RATIO)

    # --- CASE 3: Inverted/Negative Structural Risk Rejection (Defect 4) ---
    df_inverted = df_buy.copy()
    # Pullback low above price: structural_stop > price -> raw_risk <= 0
    df_inverted["low"] = 100.50
    sig_inv = strat.evaluate("TEST_INV", df_inverted)
    assert sig_inv is None, "Inverted structural stop (raw_risk <= 0) must be rejected immediately"

    # --- CASE 4: Insufficient Structural Reward Rejection (Defect 1: Structure-First) ---
    df_poor_reward = df_buy.copy()
    # vwap_upper is only 2 cents above price, while risk is >= 0.375 -> structural RR ~ 0.05 << min_rr 1.0
    df_poor_reward["vwap_upper"] = 100.02
    sig_poor = strat.evaluate("TEST_POOR_RR", df_poor_reward)
    assert sig_poor is None, (
        "Setup with insufficient structural reward (RR < min_rr) must be rejected under Structure-First policy"
    )


def test_persistence_failure_blocks_submission(tmp_path):
    """Invariant I-1: Failure to persist reservation to disk must fail-closed, roll back memory, and mark store unhealthy."""
    from unittest.mock import patch
    from strategies.order_idempotency import EntryIdempotencyStore

    state_file = tmp_path / "test_idempotency.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    # Mock _save_state to fail
    with patch.object(store, "_save_state", side_effect=IOError("Simulated disk write failure")):
        reserved, reason = store.reserve("NVDA", side="BUY")
        assert reserved is False
        assert reason == "PERSISTENCE_FAILED"
        assert "NVDA_BUY" not in store.reservations
        assert store.is_healthy is False
        assert store.health_reason == "PERSISTENCE_FAILURE"

    # Subsequent reserve attempts must also fail-closed while store is unhealthy
    reserved_subseq, reason_subseq = store.reserve("AAPL", side="BUY")
    assert reserved_subseq is False
    assert reason_subseq == "IDEMPOTENCY_STORE_UNHEALTHY"


def test_corrupt_state_fails_closed(tmp_path):
    """Invariant I-6: Corrupt persisted state file must mark store unhealthy and reject all submissions."""
    from strategies.order_idempotency import EntryIdempotencyStore

    state_file = tmp_path / "corrupt_idempotency.json"
    state_file.write_text("NOT_VALID_JSON{{{corrupt", encoding="utf-8")

    store = EntryIdempotencyStore(storage_path=state_file)
    assert store.is_healthy is False
    assert store.health_reason == "LOAD_FAILURE"

    reserved, reason = store.reserve("NVDA", side="BUY")
    assert reserved is False
    assert reason == "IDEMPOTENCY_STORE_UNHEALTHY"


def test_acknowledged_reservation_cannot_be_manually_released(tmp_path):
    """Invariant I-5: An ACKNOWLEDGED reservation with active broker order cannot be released via ordinary release()."""
    from strategies.order_idempotency import EntryIdempotencyStore, STATUS_ACKNOWLEDGED

    state_file = tmp_path / "ack_idempotency.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    # Reserve and acknowledge
    reserved, _ = store.reserve("NVDA", side="BUY")
    assert reserved is True
    bound = store.bind_parent("NVDA_BUY", "order-parent-999")
    assert bound is True

    before = store.reservations["NVDA_BUY"]
    assert before.status == STATUS_ACKNOWLEDGED
    assert before.parent_order_id == "order-parent-999"

    # Attempt ordinary release
    released = store.release("NVDA", side="BUY", reason="test_accidental_release")
    assert released is False
    assert "NVDA_BUY" in store.reservations
    assert store.reservations["NVDA_BUY"] is before
    assert store.reservations["NVDA_BUY"].parent_order_id == "order-parent-999"
    assert store.reservations["NVDA_BUY"].status == STATUS_ACKNOWLEDGED


def test_stale_unacknowledged_transitions_to_reconciliation_required(tmp_path):
    """Invariant I-2: Unacknowledged reservation exceeding TTL transitions to RECONCILIATION_REQUIRED and remains blocked."""
    import time
    from unittest.mock import patch
    from strategies.order_idempotency import EntryIdempotencyStore, STATUS_RECONCILIATION_REQUIRED

    state_file = tmp_path / "stale_idempotency.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    reserved, _ = store.reserve("NVDA", side="BUY", ttl_seconds=30.0)
    assert reserved is True

    # Advance time beyond 30s TTL
    future_time = time.time() + 35.0
    with patch("time.time", return_value=future_time):
        reserved_second, reason_second = store.reserve("NVDA", side="BUY", ttl_seconds=30.0)
        assert reserved_second is False
        assert reason_second == "RECONCILIATION_REQUIRED"
        assert store.reservations["NVDA_BUY"].status == STATUS_RECONCILIATION_REQUIRED

        # Subsequent attempts must continue to be blocked
        reserved_again, reason_again = store.reserve("NVDA", side="BUY")
        assert reserved_again is False
        assert reason_again == "RECONCILIATION_REQUIRED"


def test_stale_transition_save_failure_remains_blocked(tmp_path):
    """Invariant I-1/I-2: Persistence failure during stale transition must roll back mutation and remain blocked."""
    import time
    from unittest.mock import patch
    from strategies.order_idempotency import EntryIdempotencyStore

    state_file = tmp_path / "stale_save_fail_idempotency.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    reserved, _ = store.reserve("NVDA", side="BUY", ttl_seconds=30.0)
    assert reserved is True

    future_time = time.time() + 35.0
    with patch("time.time", return_value=future_time), \
         patch.object(store, "_save_state", side_effect=IOError("Disk error on stale transition")):
        reserved_stale, reason_stale = store.reserve("NVDA", side="BUY", ttl_seconds=30.0)
        assert reserved_stale is False
        assert reason_stale == "PERSISTENCE_FAILED"
        assert store.is_healthy is False
        assert store.health_reason == "PERSISTENCE_FAILURE"

    # Must remain blocked
    reserved_next, _ = store.reserve("NVDA", side="BUY")
    assert reserved_next is False


def test_full_strategy_name_participates_in_digest():
    """Invariant I-4: Strategies with same 4-letter prefix must produce distinct client_order_ids via full strategy name."""
    from strategies.order_idempotency import build_client_order_id

    id_bounce = build_client_order_id(
        symbol="SPY",
        side="BUY",
        session_date="20260905",
        strategy="VWAP_BOUNCE",
        bar_timestamp_utc="2026-09-05T14:30:00Z",
    )
    id_breakout = build_client_order_id(
        symbol="SPY",
        side="BUY",
        session_date="20260905",
        strategy="VWAP_BREAKOUT",
        bar_timestamp_utc="2026-09-05T14:30:00Z",
    )
    # Both start with se3-SPY-B-vwap-20260905-
    assert id_bounce.startswith("se3-SPY-B-vwap-20260905-")
    assert id_breakout.startswith("se3-SPY-B-vwap-20260905-")
    # But because full strategy participated in digest, the digests must NOT collide
    assert id_bounce != id_breakout, f"VWAP_BOUNCE and VWAP_BREAKOUT must not collide: {id_bounce} == {id_breakout}"


def test_invalid_side_is_rejected():
    """Exact Identity Invariant: Side must strictly be 'BUY' or 'SELL', any invalid side raises ValueError."""
    import pytest
    from strategies.order_idempotency import build_client_order_id

    with pytest.raises(ValueError, match="Invalid order side"):
        build_client_order_id("AAPL", "HOLD")

    with pytest.raises(ValueError, match="Side must be a non-empty string"):
        build_client_order_id("AAPL", "")

    with pytest.raises(ValueError, match="Invalid order side"):
        build_client_order_id("AAPL", "INVALID")


def test_empty_symbol_is_rejected():
    """Exact Identity Invariant: Symbol must be non-empty string with valid characters."""
    import pytest
    from strategies.order_idempotency import build_client_order_id

    with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
        build_client_order_id("", "BUY")

    with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
        build_client_order_id("   ", "BUY")

    with pytest.raises(ValueError, match="Invalid characters in symbol"):
        build_client_order_id("AAPL/USD!@#", "BUY")


def test_negative_entry_attempt_is_rejected():
    """Exact Identity Invariant: Attempt counters must be non-negative integers."""
    import pytest
    from strategies.order_idempotency import build_client_order_id

    with pytest.raises(ValueError, match="entry_attempt must be >= 0"):
        build_client_order_id("AAPL", "BUY", attempt=-1)

    with pytest.raises(ValueError, match="entry_attempt must be >= 0"):
        build_client_order_id("AAPL", "BUY", strategy_entry_attempt=-2)

    with pytest.raises(ValueError, match="submission_retry_attempt must be >= 0"):
        build_client_order_id("AAPL", "BUY", submission_retry_attempt=-1)


def test_client_order_id_respects_broker_constraints():
    """Broker Constraint Invariant: Generated client_order_id must be <= 128 chars and contain only [A-Za-z0-9_-]."""
    import re
    from strategies.order_idempotency import build_client_order_id

    # Normal symbol
    cid = build_client_order_id("NVDA", "BUY", strategy="MEAN_REVERSION")
    assert len(cid) <= 128
    assert re.match(r"^[A-Za-z0-9_-]+$", cid)

    # Symbol with dot (e.g. BRK.B)
    cid_dot = build_client_order_id("BRK.B", "BUY", strategy="VWAP_BOUNCE")
    assert len(cid_dot) <= 128
    assert re.match(r"^[A-Za-z0-9_-]+$", cid_dot)
    assert "BRK_B" in cid_dot


def test_transport_retry_preserves_client_order_id():
    """Invariant I-3: Transport retry attempts strictly preserve the exact same client_order_id."""
    from strategies.order_idempotency import build_client_order_id

    base_id = build_client_order_id(
        symbol="MSFT",
        side="BUY",
        session_date="20260905",
        strategy="VWAP_BOUNCE",
        bar_timestamp_utc="2026-09-05T14:30:00Z",
        submission_retry_attempt=0,
    )
    retry_1 = build_client_order_id(
        symbol="MSFT",
        side="BUY",
        session_date="20260905",
        strategy="VWAP_BOUNCE",
        bar_timestamp_utc="2026-09-05T14:30:00Z",
        submission_retry_attempt=1,
    )
    retry_2 = build_client_order_id(
        symbol="MSFT",
        side="BUY",
        session_date="20260905",
        strategy="VWAP_BOUNCE",
        bar_timestamp_utc="2026-09-05T14:30:00Z",
        submission_retry_attempt=2,
    )
    assert base_id == retry_1 == retry_2


def test_strategy_reentry_changes_client_order_id():
    """Invariant I-4: Deliberate strategy re-entry or scale-in increments entry_attempt and changes client_order_id."""
    from strategies.order_idempotency import build_client_order_id

    entry_0 = build_client_order_id(
        symbol="MSFT",
        side="BUY",
        session_date="20260905",
        strategy="VWAP_BOUNCE",
        bar_timestamp_utc="2026-09-05T14:30:00Z",
        strategy_entry_attempt=0,
    )
    entry_1 = build_client_order_id(
        symbol="MSFT",
        side="BUY",
        session_date="20260905",
        strategy="VWAP_BOUNCE",
        bar_timestamp_utc="2026-09-05T14:30:00Z",
        strategy_entry_attempt=1,
    )
    assert entry_0 != entry_1


def test_intent_nonce_changes_client_order_id():
    """Exact Identity Invariant: Distinct intent nonces produce distinct client_order_id."""
    from strategies.order_idempotency import build_client_order_id

    cid_a = build_client_order_id("TSLA", "BUY", intent_nonce="intent-alpha-1")
    cid_b = build_client_order_id("TSLA", "BUY", intent_nonce="intent-beta-2")
    assert cid_a != cid_b


def test_bind_parent_exact_key_matching(tmp_path):
    """Commit 3B Invariant: bind_parent must match exact reservation key and never mutate other side or cross-symbol."""
    from strategies.order_idempotency import EntryIdempotencyStore, STATUS_ACKNOWLEDGED, STATUS_PENDING

    state_file = tmp_path / "exact_bind_idempotency.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    # Reserve both BUY and SELL for AAPL
    res_buy, _ = store.reserve("AAPL", side="BUY")
    res_sell, _ = store.reserve("AAPL", side="SELL")
    assert res_buy is True and res_sell is True

    # Bind parent to AAPL_BUY only
    bound = store.bind_parent("AAPL_BUY", "order-parent-buy-123")
    assert bound is True

    # BUY reservation must be ACKNOWLEDGED with parent order
    assert store.reservations["AAPL_BUY"].status == STATUS_ACKNOWLEDGED
    assert store.reservations["AAPL_BUY"].parent_order_id == "order-parent-buy-123"

    # SELL reservation must NOT be altered
    assert store.reservations["AAPL_SELL"].status == STATUS_PENDING
    assert store.reservations["AAPL_SELL"].parent_order_id is None


def test_bind_parent_idempotent_same_parent(tmp_path):
    """Commit 3B Invariant: Repeated bind_parent calls with identical parent_order_id are idempotent and succeed."""
    from strategies.order_idempotency import EntryIdempotencyStore, STATUS_ACKNOWLEDGED

    state_file = tmp_path / "idempotent_bind.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    store.reserve("NVDA", side="BUY")
    bound_1 = store.bind_parent("NVDA_BUY", "order-parent-nvda-1")
    assert bound_1 is True

    # Repeated identical bind
    bound_2 = store.bind_parent("NVDA_BUY", "order-parent-nvda-1")
    assert bound_2 is True
    assert store.reservations["NVDA_BUY"].parent_order_id == "order-parent-nvda-1"
    assert store.reservations["NVDA_BUY"].status == STATUS_ACKNOWLEDGED


def test_bind_parent_conflict_rejection(tmp_path):
    """Commit 3B Invariant: Binding conflict to a different parent_order_id must be rejected without mutating state."""
    from strategies.order_idempotency import EntryIdempotencyStore, STATUS_ACKNOWLEDGED

    state_file = tmp_path / "conflict_bind.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    store.reserve("NVDA", side="BUY")
    bound_1 = store.bind_parent("NVDA_BUY", "order-parent-nvda-1")
    assert bound_1 is True

    # Attempt to bind conflicting parent ID
    conflict_bound = store.bind_parent("NVDA_BUY", "order-parent-CONFLICT-999")
    assert conflict_bound is False
    # Original parent binding preserved
    assert store.reservations["NVDA_BUY"].parent_order_id == "order-parent-nvda-1"
    assert store.reservations["NVDA_BUY"].status == STATUS_ACKNOWLEDGED


def test_bind_parent_persistence_failure_rollback(tmp_path):
    """Commit 3B Invariant: Persistence failure during bind_parent rolls back mutation and marks store unhealthy."""
    from unittest.mock import patch
    from strategies.order_idempotency import EntryIdempotencyStore, STATUS_PENDING

    state_file = tmp_path / "fail_bind.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    store.reserve("NVDA", side="BUY")
    assert store.reservations["NVDA_BUY"].status == STATUS_PENDING
    assert store.reservations["NVDA_BUY"].parent_order_id is None

    # Force persistence failure
    with patch.object(store, "_save_state", side_effect=IOError("Disk write failed during bind_parent")):
        bound = store.bind_parent("NVDA_BUY", "order-parent-fail")
        assert bound is False
        # In-memory rollback verification
        assert store.reservations["NVDA_BUY"].status == STATUS_PENDING
        assert store.reservations["NVDA_BUY"].parent_order_id is None
        assert store.is_healthy is False
        assert store.health_reason == "PERSISTENCE_FAILURE"


def test_attach_parent_order_raises_on_failure(tmp_path):
    """Commit 3B Invariant: attach_parent_order must raise RuntimeError on failure so callers trigger reconciliation."""
    import pytest
    from unittest.mock import patch
    from strategies.order_idempotency import EntryIdempotencyStore

    state_file = tmp_path / "attach_fail.json"
    store = EntryIdempotencyStore(storage_path=state_file)

    store.reserve("NVDA", side="BUY")
    with patch.object(store, "_save_state", side_effect=IOError("Disk write failed")):
        with pytest.raises(RuntimeError, match="Failed to attach parent order"):
            store.attach_parent_order("NVDA", "BUY", "order-fail-parent")
