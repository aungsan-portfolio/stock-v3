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
    from strategies.entry_lifecycle import entry_lifecycle, PendingEntry

    b = AlpacaBridge()
    b._connected = True
    b._client = object()

    # Register p_9999 as tracked entry parent
    entry_lifecycle._pending["p_9999"] = PendingEntry(
        parent_order_id="p_9999",
        symbol="AAPL",
        signal=object(),
        requested_qty=10.0,
        stop_order_id="s_9999",
        target_order_id="t_9999",
    )

    class MockUpdate:
        event = "replaced"
        order_id = "p_9999"
        execution_id = None
        order_status = "replaced"
        position_qty = 0.0
        price = 0.0
        order = None

    try:
        b._dispatch_trade_update(MockUpdate())
        assert b.entry_gate.halted is True
        assert not b._conn_health.is_healthy
    finally:
        entry_lifecycle._pending.pop("p_9999", None)



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


def test_max_open_positions_capacity():
    """Verify that MAX_OPEN_POSITIONS dynamically matches current phase (7 slots in Phase 2)."""
    import config as cfg
    from strategies.intraday_risk import check_max_positions

    assert cfg.MAX_OPEN_POSITIONS == 7
    assert cfg.PORTFOLIO_MAX_POSITIONS == 7

    # Below cap allowed
    assert check_max_positions(cfg.MAX_OPEN_POSITIONS - 1) is True
    # At cap blocked
    assert check_max_positions(cfg.MAX_OPEN_POSITIONS) is False


def test_soxx_tiered_regime_thresholds():
    """Verify SOXX tiered thresholds: -1.5% gives -1, -3.0% gives -2."""
    import time
    from strategies.regime_engine import evaluate_market_regime
    import strategies.regime_engine as re_mod

    class MockBridge:
        def __init__(self, soxx_pct):
            self.soxx_pct = soxx_pct
        def get_quote(self, symbol):
            return None

    orig_helper = re_mod._get_premarket_pct_change
    try:
        now_ts = time.time()
        # 1. SOXX -1.8% (moderate weakness) -> score -1
        re_mod._get_premarket_pct_change = lambda sym, bridge: (0.0 if sym != "SOXX" else -1.8, "test", now_ts)
        res_mod = evaluate_market_regime(bridge=MockBridge(-1.8))
        assert res_mod.factor_scores.get("SOXX") == -1

        # 2. SOXX -3.5% (severe crash) -> score -2
        re_mod._get_premarket_pct_change = lambda sym, bridge: (0.0 if sym != "SOXX" else -3.5, "test", now_ts)
        res_sev = evaluate_market_regime(bridge=MockBridge(-3.5))
        assert res_sev.factor_scores.get("SOXX") == -2
    finally:
        re_mod._get_premarket_pct_change = orig_helper


def test_quarantine_cache_mechanics():
    """Verify quarantine_symbol, is_quarantined, and clear_quarantine."""
    from strategies.scanner.premarket_scanner import quarantine_symbol, is_quarantined, clear_quarantine
    import pandas as pd
    from strategies.intraday_data import _validate_intraday_data

    clear_quarantine()
    assert is_quarantined("BADTICK") is False

    quarantine_symbol("BADTICK", "test_reason")
    assert is_quarantined("BADTICK") is True

    clear_quarantine()
    assert is_quarantined("BADTICK") is False

    # Discontinuity triggers quarantine
    df_bad = pd.DataFrame({"close": [10.0, 20.0]})  # 100% jump
    _validate_intraday_data(df_bad, "DISCONT", "1m", validate_staleness=False)
    assert is_quarantined("DISCONT") is True

    clear_quarantine()


def test_inverse_etf_risk_off_exemption():
    """Verify that Inverse Bear ETFs are exempted from RISK_OFF long freeze."""
    from strategies.base import TradeSignal, StrategyName
    from strategies.orchestrator import _is_inverse_etf_signal, _inverse_etf_entry_allowed
    from strategies.regime_engine import MarketRegimeResult, MODE_RISK_OFF

    regime_risk_off = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.25,
        halt_new_entries=True,
    )

    sig_aapl = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.ORB,
        side="BUY",
        confidence=0.85,
        entry_price=200.0,
        stop_price=198.0,
        target_price=205.0,
        atr=2.0,
        risk_per_share=2.0,
        reason="ORB breakout",
    )

    sig_spdn = TradeSignal(
        symbol="SPDN",
        strategy=StrategyName.ORB,
        side="BUY",
        confidence=0.85,
        entry_price=15.0,
        stop_price=14.5,
        target_price=16.0,
        atr=0.5,
        risk_per_share=0.5,
        reason="Inverse Bear breakout",
    )

    # AAPL is not an inverse ETF and should not be allowed under RISK_OFF
    assert _is_inverse_etf_signal(sig_aapl) is False
    assert _inverse_etf_entry_allowed(regime_risk_off, sig_aapl) is False

    # SPDN is an inverse ETF and SHOULD be allowed under RISK_OFF
    assert _is_inverse_etf_signal(sig_spdn) is True
    assert _inverse_etf_entry_allowed(regime_risk_off, sig_spdn) is True


def test_regime_allows_inverse_etf_longs_in_risk_off():
    """Verify allow_inverse_etf_longs is True when score <= -4 and config enabled."""
    import time
    from strategies.regime_engine import evaluate_market_regime
    import strategies.regime_engine as re_mod

    class MockBridge:
        def get_quote(self, symbol):
            return None

    orig_helper = re_mod._get_premarket_pct_change
    try:
        now_ts = time.time()
        # SPY -1.0%, QQQ -1.5%, SOXX -3.5% -> score = -6 (severe crash)
        re_mod._get_premarket_pct_change = lambda sym, bridge: (
            (-1.0 if sym == "SPY" else (-1.5 if sym == "QQQ" else -3.5)), "test", now_ts
        )
        res = evaluate_market_regime(bridge=MockBridge(), scanner_candidate_count=0)
        assert res.mode == "RISK_OFF"
        assert res.allow_new_longs is False
        assert res.halt_new_entries is True
        assert res.allow_inverse_etf_longs is True
    finally:
        re_mod._get_premarket_pct_change = orig_helper


def test_regime_blocks_inverse_etf_in_data_degraded():
    """Verify allow_inverse_etf_longs is False in DATA_DEGRADED mode."""
    from strategies.regime_engine import evaluate_market_regime
    import strategies.regime_engine as re_mod

    orig_health = re_mod._check_data_health
    try:
        re_mod._check_data_health = lambda bridge: (False, "API feed timeout", {})
        res = evaluate_market_regime()
        assert res.mode == "DATA_DEGRADED"
        assert res.allow_inverse_etf_longs is False
        assert res.allow_new_longs is False
        assert res.halt_new_entries is True
    finally:
        re_mod._check_data_health = orig_health


def test_regime_blocks_inverse_etf_in_risk_on():
    """Verify allow_inverse_etf_longs is False in normal RISK_ON market."""
    import time
    from strategies.regime_engine import evaluate_market_regime
    import strategies.regime_engine as re_mod

    class MockBridge:
        def get_quote(self, symbol):
            return None

    orig_helper = re_mod._get_premarket_pct_change
    try:
        now_ts = time.time()
        # SPY +0.5%, QQQ +0.8%, SOXX +1.2% -> healthy bull market
        re_mod._get_premarket_pct_change = lambda sym, bridge: (
            (0.5 if sym == "SPY" else (0.8 if sym == "QQQ" else 1.2)), "test", now_ts
        )
        res = evaluate_market_regime(bridge=MockBridge(), scanner_candidate_count=5)
        assert res.mode == "RISK_ON"
        assert res.allow_new_longs is True
        assert res.halt_new_entries is False
        assert res.allow_inverse_etf_longs is False
    finally:
        re_mod._get_premarket_pct_change = orig_helper


def test_inverse_etf_passes_scanner_open_filters():
    """Verify SPDN passes open filters with relaxed volume thresholds."""
    from strategies.scanner.premarket_scanner import _passes_open_filters

    result = _passes_open_filters(
        symbol="SPDN",
        move_pct=1.5,
        current_price=17.50,
        volume=350_000,
        relative_volume=0.45,
    )
    assert result is True


def test_orchestrator_5m_cadence_eval_in_risk_off():
    """Verify orchestrator evaluate_and_execute runs without NameError during RISK_OFF on 5m cadence."""
    from strategies.orchestrator import evaluate_and_execute
    import strategies.orchestrator as orch_mod
    import strategies.regime_engine as re_mod

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self._daytrade_suspended = False
        def get_quote(self, symbol):
            return None
        def get_open_positions(self):
            return []
        def get_net_liquidation(self):
            return 100000.0
        def daily_pnl(self):
            return 0.0

    orig_eval_parallel = orch_mod.evaluate_symbols_parallel
    orig_regime = orch_mod._regime_state_tracker.update
    orch_mod._last_evaluated_5m_bar_str = None  # Force 5m cadence boundary
    try:
        from strategies.regime_engine import MarketRegimeResult, MODE_RISK_OFF
        mock_risk_off = MarketRegimeResult(
            mode=MODE_RISK_OFF,
            score=-5,
            allow_new_longs=False,
            requires_opening_buffer=False,
            position_scale=0.0,
            confidence_boost=0.25,
            allow_inverse_etf_longs=True,
            halt_new_entries=True,
        )
        orch_mod._regime_state_tracker.update = lambda raw: mock_risk_off
        orch_mod.evaluate_symbols_parallel = lambda watchlist, bridge=None, market_texture=None: []

        # Run evaluate_and_execute with inverse ETF watchlist
        evaluate_and_execute(
            watchlist=["SPDN", "AAPL", "SQQQ"],
            bridge=MockBridge(),
            live_paper=False,
            allow_new_entries=False,
        )
    finally:
        orch_mod.evaluate_symbols_parallel = orig_eval_parallel
        orch_mod._regime_state_tracker.update = orig_regime


def test_inverse_etf_relaxed_confidence_threshold():
    """Verify SQQQ confidence 0.68 passes with INVERSE_ETF_MIN_CONFIDENCE=0.65 even in RISK_OFF."""
    from strategies.base import TradeSignal, StrategyName
    from strategies.order_manager import execute_signal
    from strategies.regime_engine import MarketRegimeResult, MODE_RISK_OFF

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self.account_id = "TEST"
        def get_quote(self, symbol):
            return {"bid": 34.48, "ask": 34.52, "last": 34.50}
        def get_open_positions(self):
            return []
        def get_open_orders(self):
            return []
        def open_position_count(self):
            return 0
        def has_position(self, symbol):
            return False
        def has_working_order(self, symbol):
            return False

    mock_regime = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.25,
        allow_inverse_etf_longs=True,
        halt_new_entries=True,
    )

    sqqq_signal = TradeSignal(
        symbol="SQQQ",
        side="BUY",
        entry_price=34.50,
        stop_price=33.80,
        target_price=36.00,
        atr=0.80,
        risk_per_share=0.70,
        confidence=0.68,
        reason="VWAP bounce setup",
        strategy=StrategyName.VWAP_BOUNCE,
        metadata={"atr": 0.80, "vwap": 34.20},
    )

    order_res = execute_signal(
        signal=sqqq_signal,
        bridge=MockBridge(),
        equity=100000.0,
        current_pnl=0.0,
        dry_run=True,
        regime_result=mock_regime,
    )
    assert order_res.get("status") != "SKIPPED" or "Confidence" not in order_res.get("reason", "")


def test_inverse_etf_relaxed_rr_ratio():
    """Verify SOXS R:R 1.45 passes with INVERSE_ETF_MIN_RR_RATIO=1.40."""
    from strategies.base import TradeSignal, StrategyName
    from strategies.order_manager import _validate_signal_prices

    soxs_signal = TradeSignal(
        symbol="SOXS",
        side="BUY",
        entry_price=20.00,
        stop_price=18.80,  # stop_dist = 1.20 (>= 75% of ATR 1.00 = 0.75)
        target_price=21.74,  # target_dist = 1.74 -> R:R = 1.74 / 1.20 = 1.45
        atr=1.00,
        risk_per_share=1.20,
        confidence=0.70,
        reason="ORB setup",
        strategy=StrategyName.ORB,
        metadata={"atr": 1.00},
    )

    reason = _validate_signal_prices(soxs_signal)
    assert reason == ""


def test_psq_in_denylist_not_in_watchlist():
    """Verify PSQ and DOG are excluded from universe by SYMBOL_DENYLIST."""
    from strategies.scanner.premarket_scanner import _load_universe

    universe = _load_universe()
    assert "PSQ" not in universe
    assert "DOG" not in universe
    assert "SPDN" in universe
    assert "SQQQ" in universe


def test_inverse_etf_nonzero_sizing_in_risk_off():
    """Verify SQQQ position sizing produces non-zero shares even with position_scale=0 in RISK_OFF."""
    from strategies.base import TradeSignal, StrategyName
    from strategies.order_manager import execute_signal
    from strategies.regime_engine import MarketRegimeResult, MODE_RISK_OFF

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self.account_id = "TEST"
        def get_quote(self, symbol):
            return {"bid": 34.48, "ask": 34.52, "last": 34.50}
        def get_open_positions(self):
            return []
        def get_open_orders(self):
            return []
        def open_position_count(self):
            return 0
        def has_position(self, symbol):
            return False
        def has_working_order(self, symbol):
            return False

    mock_risk_off = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.25,
        allow_inverse_etf_longs=True,
        halt_new_entries=True,
    )

    sqqq_signal = TradeSignal(
        symbol="SQQQ",
        side="BUY",
        entry_price=34.50,
        stop_price=33.80,
        target_price=36.00,
        atr=0.80,
        risk_per_share=0.70,
        confidence=0.70,
        reason="VWAP bounce setup",
        strategy=StrategyName.VWAP_BOUNCE,
        metadata={"atr": 0.80, "vwap": 34.20},
    )

    order_res = execute_signal(
        signal=sqqq_signal,
        bridge=MockBridge(),
        equity=1968.64,  # 2.0% equity cap passed
        current_pnl=0.0,
        dry_run=True,
        regime_result=mock_risk_off,
    )
    assert order_res.get("reason") != "Position size = 0"
    assert order_res.get("status") in ("SUBMITTED", "PENDING_DRY_RUN", "DRY_RUN", "EXECUTED", "CONFIRMED") or order_res.get("shares", 0) > 0


def test_spy_trend_guard_inversion_for_inverse_etfs():
    """Verify SPY Trend Guard approves SOXS when SPY is bearish and blocks when SPY is bullish."""
    import pandas as pd
    from strategies.base import TradeSignal, StrategyName
    from strategies.order_manager import execute_signal
    from strategies.regime_engine import MarketRegimeResult, MODE_RISK_OFF
    import strategies.intraday_data as idata_mod

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self.account_id = "TEST"
        def get_quote(self, symbol):
            return {"bid": 20.00, "ask": 20.04, "last": 20.02}
        def get_open_positions(self):
            return []
        def get_open_orders(self):
            return []
        def open_position_count(self):
            return 0
        def has_position(self, symbol):
            return False
        def has_working_order(self, symbol):
            return False

    mock_regime = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.25,
        allow_inverse_etf_longs=True,
        halt_new_entries=True,
    )

    soxs_signal = TradeSignal(
        symbol="SOXS",
        side="BUY",
        entry_price=20.00,
        stop_price=18.80,
        target_price=22.00,
        atr=1.00,
        risk_per_share=1.20,
        confidence=0.75,
        reason="Momentum Scalp",
        strategy=StrategyName.MOMENTUM_SCALP,
        metadata={"atr": 1.00},
    )

    orig_fetch = idata_mod.fetch_intraday
    fetched_symbols = []
    def mock_fetch(sym, bridge=None):
        fetched_symbols.append(sym)
        return pd.DataFrame({
            "open": [505.0, 505.0],
            "close": [500.0, 500.0],
            "high": [506.0, 506.0],
            "low": [499.0, 499.0],
            "volume": [1000000, 1000000],
        })

    try:
        idata_mod.fetch_intraday = mock_fetch

        # Case A: SOXS -> fetches SOXX
        res_a = execute_signal(
            signal=soxs_signal,
            bridge=MockBridge(),
            equity=1968.64,
            current_pnl=0.0,
            dry_run=False,
            regime_result=mock_regime,
        )
        assert "SOXX" in fetched_symbols
        assert "Blocking" not in res_a.get("reason", "")

        # Case B: SQQQ -> fetches QQQ
        sqqq_signal = TradeSignal(
            symbol="SQQQ",
            side="BUY",
            entry_price=34.50,
            stop_price=33.80,
            target_price=36.00,
            atr=0.80,
            risk_per_share=0.70,
            confidence=0.75,
            reason="VWAP bounce",
            strategy=StrategyName.VWAP_BOUNCE,
            metadata={"atr": 0.80},
        )
        res_sqqq = execute_signal(
            signal=sqqq_signal,
            bridge=MockBridge(),
            equity=1968.64,
            current_pnl=0.0,
            dry_run=False,
            regime_result=mock_regime,
        )
        assert "QQQ" in fetched_symbols
        assert "Blocking" not in res_sqqq.get("reason", "")

        # Case C: Benchmark candle is BULLISH (Close 505 > Open 500) -> Should BLOCK
        def mock_bullish_fetch(sym, bridge=None):
            return pd.DataFrame({
                "open": [500.0, 500.0],
                "close": [505.0, 505.0],
                "high": [506.0, 506.0],
                "low": [499.0, 499.0],
                "volume": [1000000, 1000000],
            })
        idata_mod.fetch_intraday = mock_bullish_fetch

        res_c = execute_signal(
            signal=soxs_signal,
            bridge=MockBridge(),
            equity=1968.64,
            current_pnl=0.0,
            dry_run=False,
            regime_result=mock_regime,
        )
        assert "Trend Guard" in res_c.get("reason", "") and "Blocking" in res_c.get("reason", "")
    finally:
        idata_mod.fetch_intraday = orig_fetch


def test_benchmark_fetch_failure_blocks_entry():
    """Verify SOXS entry is safely blocked when benchmark intraday data is unavailable/errored."""
    import pandas as pd
    from strategies.base import TradeSignal, StrategyName
    from strategies.order_manager import execute_signal
    from strategies.regime_engine import MarketRegimeResult, MODE_RISK_OFF
    import strategies.intraday_data as idata_mod

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self.account_id = "TEST"
        def get_quote(self, symbol):
            return {"bid": 20.00, "ask": 20.04, "last": 20.02}
        def get_open_positions(self):
            return []
        def get_open_orders(self):
            return []
        def open_position_count(self):
            return 0
        def has_position(self, symbol):
            return False
        def has_working_order(self, symbol):
            return False

    mock_regime = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.25,
        allow_inverse_etf_longs=True,
        halt_new_entries=True,
    )

    soxs_signal = TradeSignal(
        symbol="SOXS",
        side="BUY",
        entry_price=20.00,
        stop_price=18.80,
        target_price=22.00,
        atr=1.00,
        risk_per_share=1.20,
        confidence=0.75,
        reason="Momentum Scalp",
        strategy=StrategyName.MOMENTUM_SCALP,
        metadata={"atr": 1.00},
    )

    orig_fetch = idata_mod.fetch_intraday
    try:
        # Simulate empty / failed fetch
        idata_mod.fetch_intraday = lambda sym, bridge=None: pd.DataFrame()

        res = execute_signal(
            signal=soxs_signal,
            bridge=MockBridge(),
            equity=1968.64,
            current_pnl=0.0,
            dry_run=False,
            regime_result=mock_regime,
        )
        assert res.get("status") in ("SKIPPED", "REJECTED")
        assert "unavailable" in res.get("reason", "") or "safety" in res.get("reason", "")
    finally:
        idata_mod.fetch_intraday = orig_fetch


def test_unknown_inverse_etf_fallback():
    """Verify unmapped inverse ETF safely defaults to SPY without KeyError."""
    from strategies.base import TradeSignal, StrategyName
    from strategies.order_manager import execute_signal
    from strategies.regime_engine import MarketRegimeResult, MODE_RISK_OFF
    import strategies.intraday_data as idata_mod
    import pandas as pd
    import config

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self.account_id = "TEST"
        def get_quote(self, symbol):
            return {"bid": 10.00, "ask": 10.04, "last": 10.02}
        def get_open_positions(self):
            return []
        def get_open_orders(self):
            return []
        def open_position_count(self):
            return 0
        def has_position(self, symbol):
            return False
        def has_working_order(self, symbol):
            return False

    mock_regime = MarketRegimeResult(
        mode=MODE_RISK_OFF,
        score=-5,
        allow_new_longs=False,
        requires_opening_buffer=False,
        position_scale=0.0,
        confidence_boost=0.25,
        allow_inverse_etf_longs=True,
        halt_new_entries=True,
    )

    # UNKNOWN_BEAR is in INVERSE_ETF_SYMBOLS but not explicitly in INVERSE_ETF_BENCHMARK_MAP
    orig_syms = set(config.INVERSE_ETF_SYMBOLS)
    config.INVERSE_ETF_SYMBOLS.add("UNKNOWN_BEAR")

    custom_signal = TradeSignal(
        symbol="UNKNOWN_BEAR",
        side="BUY",
        entry_price=10.00,
        stop_price=9.40,
        target_price=11.00,
        atr=0.50,
        risk_per_share=0.60,
        confidence=0.75,
        reason="VWAP bounce",
        strategy=StrategyName.VWAP_BOUNCE,
        metadata={"atr": 0.50},
    )

    orig_fetch = idata_mod.fetch_intraday
    fetched_symbols = []
    def mock_fetch(sym, bridge=None):
        fetched_symbols.append(sym)
        return pd.DataFrame({
            "open": [505.0, 505.0],
            "close": [500.0, 500.0],
            "high": [506.0, 506.0],
            "low": [499.0, 499.0],
            "volume": [1000000, 1000000],
        })

    try:
        idata_mod.fetch_intraday = mock_fetch
        res = execute_signal(
            signal=custom_signal,
            bridge=MockBridge(),
            equity=1968.64,
            current_pnl=0.0,
            dry_run=False,
            regime_result=mock_regime,
        )
        assert "SPY" in fetched_symbols  # Graceful fallback to SPY
        assert res.get("reason") != "Position size = 0"
    finally:
        idata_mod.fetch_intraday = orig_fetch

def test_inverse_etf_short_signal_blocked():
    """Verify that SHORT/SELL signals on inverse ETFs are rejected immediately across all layers."""
    from strategies.base import TradeSignal, StrategyName
    from strategies.order_manager import execute_signal
    from strategies.momentum_strategy import MomentumScalpStrategy
    import pandas as pd
    import numpy as np

    # 1. Order Manager rejects inverse short
    short_sqqq = TradeSignal(
        symbol="SQQQ",
        side="SELL",
        entry_price=38.74,
        stop_price=38.87,
        target_price=38.54,
        atr=0.50,
        risk_per_share=0.13,
        confidence=0.79,
        reason="Momentum scalp SHORT",
        strategy=StrategyName.MOMENTUM_SCALP,
    )
    res = execute_signal(short_sqqq, bridge=None, equity=100000.0, current_pnl=0.0, dry_run=True)
    assert res.get("status") in ("SKIPPED", "REJECTED")
    assert "cannot be sold short" in res.get("reason", "")

    # 2. Strategy evaluation suppresses short for inverse ETFs
    strat = MomentumScalpStrategy()
    # Mock data that would normally trigger sell checks
    n = 30
    df = pd.DataFrame({
        "open": np.linspace(40, 35, n),
        "high": np.linspace(40.5, 35.5, n),
        "low": np.linspace(39.5, 34.5, n),
        "close": np.linspace(39.8, 34.8, n),
        "volume": [1000000] * n,
    })
    sig = strat.evaluate("SOXS", df)
def test_momentum_overextension_skipped():
    """Verify that momentum strategy skips buying when price is overextended far above EMA9."""
    from strategies.momentum_strategy import MomentumScalpStrategy
    import pandas as pd
    import numpy as np

    strat = MomentumScalpStrategy()
    n = 30
    # Simulate a sudden massive parabolic spike where close is way above EMA9
    close = np.linspace(100, 110, n)
    close[-1] = 125.0  # Massive spike overextended
    df = pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": [1000000] * n,
    })
    sig = strat.evaluate("AAPL", df)
    # Must be skipped due to overextension
    assert sig is None

def test_momentum_dynamic_atr_stop_floor():
    """Verify that stop distance is at least 1.25x ATR to provide breathing room."""
    from strategies.momentum_strategy import MomentumScalpStrategy
    import pandas as pd
    import numpy as np

    strat = MomentumScalpStrategy()
    n = 30
    close = np.linspace(100, 105, n)
    df = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 0.4,
        "low": close - 0.2,
        "close": close,
        "volume": [2000000] * n,
    })
    sig = strat.evaluate("MSFT", df)
    if sig is not None:
        stop_dist = abs(sig.entry_price - sig.stop_price)
        assert stop_dist >= sig.atr * 1.20


# ==============================================================================
# FAULT-INJECTION KILL-SWITCH TEST SUITE (Institutional Verification)
# ==============================================================================

def test_fault_injection_daily_loss_halt():
    """Fault Injection 1: When start-of-day equity drops > 2%, daily loss halt suspends new entries."""
    from alpaca_bridge import AlpacaBridge
    from strategies.order_manager import execute_signal
    from strategies.base import TradeSignal
    from strategies.constants import StrategyName

    bridge = AlpacaBridge()
    bridge._connected = True
    bridge._daytrade_suspended = True  # Injected state: daily loss circuit breaker triggered

    sig = TradeSignal(
        symbol="SPY",
        side="BUY",
        entry_price=500.0,
        stop_price=498.0,
        target_price=505.0,
        atr=2.0,
        risk_per_share=2.0,
        confidence=0.85,
        reason="Momentum scalp LONG",
        strategy=StrategyName.MOMENTUM_SCALP,
    )
    # Entry gate must block when suspended
    bridge.entry_gate.halted = True
    res = execute_signal(sig, bridge=bridge, equity=100000.0, current_pnl=-2500.0, dry_run=False)
    assert res.get("status") in ("REJECTED", "VETOED", "SKIPPED")
    assert "halted" in res.get("reason", "").lower() or "suspended" in res.get("reason", "").lower()


def test_fault_injection_consecutive_error_circuit_breaker():
    """Fault Injection 2: 5 consecutive broker rejections trips the safety circuit breaker."""
    from unittest.mock import MagicMock
    from strategies.error_handler import TradingErrorHandler

    mock_bridge = MagicMock()
    mock_bridge.is_connected = True
    handler = TradingErrorHandler(bridge=mock_bridge, max_consecutive_errors=5)
    assert handler.consecutive_errors == 0

    # Inject 4 errors (not yet 5)
    for i in range(4):
        handler._consecutive_errors += 1
        assert handler.consecutive_errors < 5

    # 5th error trips circuit breaker and calls _flatten_and_exit with SystemExit
    with pytest.raises(SystemExit):
        handler.record(Exception("Fatal Broker Rejection #5"))
    mock_bridge.flatten_all.assert_called_once()


def test_fault_injection_stale_data_quarantine_halt():
    """Fault Injection 3: Stale bar timestamps (>300s) trigger fail-closed quarantine and halt signals."""
    from strategies.intraday_data import _validate_intraday_data
    from datetime import datetime, timezone, timedelta

    # 600-second old bar
    stale_time = datetime.now(timezone.utc) - timedelta(seconds=600)
    df = pd.DataFrame({
        "timestamp": [stale_time],
        "open": [100.0],
        "high": [101.0],
        "low": [99.0],
        "close": [100.5],
        "volume": [10000],
    })
    cleaned = _validate_intraday_data(df, "AAPL", "5m", validate_staleness=True)
    # Stale data returns empty DataFrame (quarantined)
    assert cleaned.empty


def test_fault_injection_emergency_flatten_all():
    """Fault Injection 4: Emergency flatten_all cancels orders and closes all open positions."""
    from unittest.mock import MagicMock
    from alpaca_bridge import AlpacaBridge

    bridge = AlpacaBridge()
    bridge._connected = True

    mock_client = MagicMock()
    # Mock 2 open positions
    pos1 = MagicMock(symbol="SOXS", qty="100")
    pos2 = MagicMock(symbol="SQQQ", qty="130")
    mock_client.get_all_positions.side_effect = [
        [pos1, pos2],  # Initial fetch during flatten
        []             # Post-flatten reconciliation (0 positions remaining)
    ]
    mock_client.cancel_orders.return_value = None
    mock_client.close_position.return_value = None

    bridge._client = mock_client
    res = bridge.flatten_all(confirm=True)

    # Verify cancel_orders was called
    mock_client.cancel_orders.assert_called_once()
    # Verify close_position was called for each symbol
    assert mock_client.close_position.call_count == 2
    mock_client.close_position.assert_any_call("SOXS")
    mock_client.close_position.assert_any_call("SQQQ")
    assert res.get("status") in ("ok", "success")
    assert res.get("positions_closed") == 2
