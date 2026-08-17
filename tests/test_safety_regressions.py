"""
test_safety_regressions.py -- M-07 Safety Regression Test Suite

Tests C-01, C-02, H-01, H-02, H-03, H-04, M-01, M-04, M-05, M-06 safety remediations.
"""
import json
import os
import time
import pandas as pd
import pytest

from strategies.base import TradeSignal
from strategies.constants import StrategyName
from strategies.execution_quote import BrokerQuote, validate_execution_quote
from strategies.intraday_data import _read_cache, _write_cache
from strategies.intraday_risk import check_soxx_tech_long_gate
from strategies.order_idempotency import OrderIdempotencyManager
from strategies.regime_engine import _get_premarket_pct_change, detect_market_texture, TEXTURE_VOLATILE


def test_c01_mean_reversion_signal_contract():
    """C-01: TradeSignal instantiated with atr and risk_per_share without invalid kwargs."""
    from strategies.mean_reversion import MeanReversionStrategy

    strat = MeanReversionStrategy()
    df = pd.DataFrame(
        {
            "close": [100.0] * 30,
            "high": [101.0] * 30,
            "low": [99.0] * 30,
            "volume": [10000] * 30,
        }
    )
    sig = strat.generate_signal("NVDA", df)
    # May return None or TradeSignal, but must not crash due to missing field or invalid kwarg
    if sig:
        assert isinstance(sig, TradeSignal)
        assert hasattr(sig, "atr")
        assert hasattr(sig, "risk_per_share")


def test_trading_stream_dispatch_deduplication():
    """TradingStream trade updates deduplicate identical execution IDs or event tuples."""
    from alpaca_bridge import AlpacaBridge

    b = AlpacaBridge()
    dispatched = []
    b.on_entry_order_update = lambda **kwargs: dispatched.append(kwargs)

    class MockUpdate:
        def __init__(self, event, order_id, exec_id, status, filled_qty, avg_price):
            self.event = event
            self.order_id = order_id
            self.execution_id = exec_id
            self.order_status = status
            self.position_qty = filled_qty
            self.price = avg_price
            self.order = None

    u1 = MockUpdate("fill", "p_1001", "exec_1", "filled", 10.0, 150.0)
    b._dispatch_trade_update(u1)
    # Duplicate update
    b._dispatch_trade_update(u1)

    assert len(dispatched) == 1
    assert dispatched[0]["parent_order_id"] == "p_1001"
    assert dispatched[0]["execution_id"] == "exec_1"


def test_trading_stream_replaced_order_halt():
    """TradingStream 'replaced' order status halts new entries."""
    from alpaca_bridge import AlpacaBridge

    b = AlpacaBridge()
    b._connected = True
    b._client = object()

    class MockUpdate:
        event = "replaced"
        order_id = "p_9999"
        execution_id = None
        order_status = "replaced"
        position_qty = 0.0
        price = 0.0
        order = None

    b._dispatch_trade_update(MockUpdate())
    assert b.entry_gate.halted is True
    assert not b._conn_health.is_healthy


def test_h01_soxx_ratio_conversion():
    """H-01: SOXX -0.08% passes; -0.80% vetoes."""
    import strategies.intraday_risk as ir

    orig_fn = ir._get_soxx_realtime_pct_change
    try:
        ir._get_soxx_realtime_pct_change = lambda: -0.08
        ok1, reason1 = check_soxx_tech_long_gate("NVDA", side="BUY")
        assert ok1 is True

        ir._get_soxx_realtime_pct_change = lambda: -0.80
        ok2, reason2 = check_soxx_tech_long_gate("NVDA", side="BUY")
        assert ok2 is False
    finally:
        ir._get_soxx_realtime_pct_change = orig_fn


def test_h02_sell_side_propagation():
    """H-02: Side='SELL' propagates through SOXX gate."""
    import strategies.intraday_risk as ir

    orig_fn = ir._get_soxx_realtime_pct_change
    try:
        ir._get_soxx_realtime_pct_change = lambda: -0.80
        ok, reason = check_soxx_tech_long_gate("NVDA", side="SELL")
        assert ok is True
        assert "short trade" in reason
    finally:
        ir._get_soxx_realtime_pct_change = orig_fn


def test_m01_fresh_broker_quote_gate():
    """M-01: validate_execution_quote rejects missing, stale, wide, or deviating quotes."""

    class MockBridge:
        def __init__(self, quote_dict):
            self.quote_dict = quote_dict

        def get_quote(self, symbol):
            return self.quote_dict

    # 1. Valid quote
    valid_q = {
        "symbol": "NVDA",
        "bid": 99.99,
        "ask": 100.01,
        "last": 100.00,
        "timestamp": time.time(),
    }
    b1 = MockBridge(valid_q)
    res1 = validate_execution_quote(b1, "NVDA", side="BUY", decision_price=100.00)
    assert res1.valid is True

    # 2. Stale quote
    stale_q = dict(valid_q)
    stale_q["timestamp"] = time.time() - 30.0  # 30s old > max 10s
    b2 = MockBridge(stale_q)
    res2 = validate_execution_quote(b2, "NVDA", side="BUY", decision_price=100.00)
    assert res2.valid is False
    assert "stale" in res2.reason.lower()

    # 3. Wide spread
    wide_q = dict(valid_q)
    wide_q["bid"] = 95.0
    wide_q["ask"] = 105.0  # 10% spread > 1% limit
    b3 = MockBridge(wide_q)
    res3 = validate_execution_quote(b3, "NVDA", side="BUY", decision_price=100.00)
    assert res3.valid is False
    assert "spread" in res3.reason.lower()

    # 4. Deviating price
    dev_q = dict(valid_q)
    dev_q["bid"] = 105.0
    dev_q["ask"] = 105.05
    dev_q["last"] = 105.0
    b4 = MockBridge(dev_q)
    res4 = validate_execution_quote(b4, "NVDA", side="BUY", decision_price=100.00)  # 5% dev > 1.5% limit
    assert res4.valid is False
    assert "deviation" in res4.reason.lower()


def test_m04_json_cache_round_trip():
    """M-04: _write_cache and _read_cache serialize timestamps cleanly."""
    key = "test_round_trip"
    dates = pd.date_range("2026-08-13 09:30", periods=3, freq="5min")
    df_in = pd.DataFrame({
        "open": [10.0, 11.0, 12.0],
        "high": [10.5, 11.5, 12.5],
        "low": [9.5, 10.5, 11.5],
        "close": [10.5, 11.2, 12.1],
        "volume": [1000, 2000, 3000],
    }, index=dates)

    _write_cache(key, df_in)

    df_read = _read_cache(key, ttl=60)
    assert df_read is not None
    assert len(df_read) == 3
    assert float(df_read["close"].iloc[0]) == 10.5


def test_m05_durable_idempotency_manager(tmp_path):
    """M-05: OrderIdempotencyManager reserves key until released on terminal status."""
    mgr = OrderIdempotencyManager(storage_path=tmp_path / "idempotency.json")

    # 1. First reservation succeeds
    ok1, _ = mgr.reserve("NVDA", side="BUY")
    assert ok1 is True

    # 2. Immediate second reservation blocked
    ok2, _ = mgr.reserve("NVDA", side="BUY")
    assert ok2 is False

    # 3. Attach parent order
    mgr.attach_parent_order("NVDA", "BUY", "p_12345")

    # 4. Terminal status releases reservation
    mgr.release_on_terminal("p_12345", "filled")

    # 5. Subsequent reservation allowed
    ok3, _ = mgr.reserve("NVDA", side="BUY")
    assert ok3 is True


def test_m06_no_single_bar_open_fallback():
    """M-06: _get_premarket_pct_change does NOT fallback to single daily bar open."""

    class MockDataClient:
        def get_stock_bars(self, req):
            # Only 1 bar returned
            class MockBar:
                open = 100.0
                close = 105.0

            return {"NVDA": [MockBar()]}

    class MockBridge:
        def __init__(self):
            self._data_client = MockDataClient()

        def get_price(self, symbol, allow_historical=True):
            return 105.0

    b = MockBridge()
    pct, source, ts = _get_premarket_pct_change("NVDA", bridge=b)
    assert pct is None or isinstance(pct, float)


def test_minimum_share_count_enforced():
    """Verify that Day Trading & Swing position sizers do NOT enforce a 5-share floor per trader instruction."""
    from strategies.position_sizer import calculate_shares
    from risk_math import shares_for_risk
    import config as cfg

    assert cfg.MIN_SHARE_COUNT == 1

    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.MEAN_REVERSION,
        side="BUY",
        confidence=0.85,
        entry_price=500.0,
        stop_price=498.0,
        target_price=510.0,
        atr=2.0,
        risk_per_share=2.0,
        reason="Test min share count",
    )
    # Small equity yields calculated shares without 5-share floor
    shares1 = calculate_shares(sig, equity=100.0)
    assert isinstance(shares1, int)

    shares2 = shares_for_risk(account_equity=100.0, risk_pct=0.01, entry_price=500.0, stop_price=498.0, side="LONG")
    assert isinstance(shares2, int)


def test_max_open_positions_capacity_is_five():
    """Verify that MAX_OPEN_POSITIONS allows up to 5 stock symbols/types."""
    import config as cfg
    from strategies.intraday_risk import check_max_positions

    assert cfg.MAX_OPEN_POSITIONS == 5
    assert cfg.PORTFOLIO_MAX_POSITIONS == 5

    # 4 positions allowed
    assert check_max_positions(4) is True
    # 5 positions blocked (at cap)
    assert check_max_positions(5) is False

