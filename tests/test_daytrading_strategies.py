import pytest
import pandas as pd
import numpy as np

import config
from strategies.indicators import add_vwap, add_opening_range, add_all_indicators
from strategies.orb_strategy import ORBStrategy
from strategies.vwap_strategy import VWAPBounceStrategy
from strategies.backtester import _realized_pnl

# ----------------------------------------------------------------------
# Pytest Fixture: Config Isolator / Backup & Restore
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def isolate_config():
    """Backup config attributes before each test, and restore them after."""
    attrs_to_backup = [
        "STRATEGY_ORB_ENABLED",
        "STRATEGY_VWAP_BOUNCE_ENABLED",
        "ORB_WINDOW_MINUTES",
        "ORB_MIN_VOLUME_RATIO",
        "ORB_MIN_RANGE_ATR_RATIO",
        "DEFAULT_TARGET_RR_RATIO",
        "VOLUME_MA_PERIOD",
        "ATR_PERIOD",
        "RSI_PERIOD",
        "VWAP_BOUNCE_TOLERANCE_PCT",
        "VWAP_RECLAIM_BARS",
        "VWAP_RSI_OVERBOUGHT",
        "VWAP_RSI_OVERSOLD"
    ]
    
    backup = {}
    for attr in attrs_to_backup:
        if hasattr(config, attr):
            backup[attr] = getattr(config, attr)
            
    yield
    
    for attr, val in backup.items():
        setattr(config, attr, val)


@pytest.fixture(autouse=True)
def mock_market_open(monkeypatch):
    import strategies.session
    import strategies.trailing_stop
    monkeypatch.setattr(strategies.session, "is_market_open", lambda: True)
    monkeypatch.setattr(strategies.trailing_stop, "is_market_open", lambda: True)


# ----------------------------------------------------------------------
# 1. VWAP Calculation Math Test
# ----------------------------------------------------------------------
def test_vwap_calculation_math():
    times = pd.date_range("2026-07-20 09:30:00", periods=3, freq="1min", tz="US/Eastern")
    mock_data = {
        "open":   [100.0, 101.0, 102.0],
        "high":   [101.0, 102.0, 103.0],
        "low":    [99.0,  100.0, 101.0],
        "close":  [100.0, 101.0, 102.0],
        "volume": [100.0, 200.0, 300.0]
    }
    df = pd.DataFrame(mock_data, index=times)

    df_vwap = add_vwap(df)
    assert np.isclose(df_vwap["vwap"].iloc[0], 100.0)
    assert np.isclose(df_vwap["vwap"].iloc[1], 30200.0 / 300.0)
    assert np.isclose(df_vwap["vwap"].iloc[2], 60800.0 / 600.0)


# ----------------------------------------------------------------------
# 2. Opening Range (ORB) High/Low Detection Test
# ----------------------------------------------------------------------
def test_opening_range_high_low_detection():
    times_orb = pd.date_range("2026-07-20 09:30:00", periods=20, freq="1min", tz="US/Eastern")
    mock_data_orb = {
        "open":  [100.0] * 20,
        "high":  [100.0] * 20,
        "low":   [100.0] * 20,
        "close": [100.0] * 20,
        "volume": [100.0] * 20
    }
    df_orb = pd.DataFrame(mock_data_orb, index=times_orb)
    df_orb.loc["2026-07-20 09:32:00", "high"] = 105.0
    df_orb.loc["2026-07-20 09:34:00", "low"] = 97.0
    df_orb.loc["2026-07-20 09:45:00", "high"] = 110.0

    config.ORB_WINDOW_MINUTES = 5
    df_orb_res = add_opening_range(df_orb, orb_minutes=5)

    assert np.isnan(df_orb_res["orb_high"].iloc[0])
    assert np.isnan(df_orb_res["orb_low"].iloc[0])
    assert df_orb_res["orb_high"].iloc[-1] == 105.0


# ----------------------------------------------------------------------
# 3a. ORB Strategy Signal Trigger Test (Happy Path)
# ----------------------------------------------------------------------
def test_orb_strategy_signal_trigger_happy_path():
    config.STRATEGY_ORB_ENABLED = True
    config.ORB_MIN_VOLUME_RATIO = 1.0
    config.ORB_MIN_RANGE_ATR_RATIO = 0.1
    config.DEFAULT_TARGET_RR_RATIO = 2.0
    config.ORB_WINDOW_MINUTES = 5

    config.VOLUME_MA_PERIOD = 2
    config.ATR_PERIOD = 2
    config.RSI_PERIOD = 14

    times_strat = pd.date_range("2026-07-20 09:30:00", periods=20, freq="1min", tz="US/Eastern")
    mock_data_strat = {
        "open":   [100.0] * 20,
        "high":   [101.0] * 20,
        "low":    [99.0] * 20,
        "close":  [100.0] * 20,
        "volume": [100.0] * 20
    }
    df_strat = pd.DataFrame(mock_data_strat, index=times_strat)

    df_strat.iloc[6, df_strat.columns.get_loc("close")] = 97.0
    df_strat.iloc[6, df_strat.columns.get_loc("low")] = 97.0
    df_strat.iloc[12, df_strat.columns.get_loc("close")] = 96.0
    df_strat.iloc[12, df_strat.columns.get_loc("low")] = 96.0

    df_strat.iloc[-1, df_strat.columns.get_loc("close")] = 101.5
    df_strat.iloc[-1, df_strat.columns.get_loc("high")] = 101.5
    df_strat.iloc[-1, df_strat.columns.get_loc("volume")] = 150.0

    orb_strat = ORBStrategy()
    signal = orb_strat.evaluate("AAPL", df_strat)
    
    assert signal is not None
    assert signal.side == "BUY"
    assert signal.entry_price == 101.5
    assert signal.stop_price == 99.0
    assert signal.target_price == 106.5


# ----------------------------------------------------------------------
# 3b. ORB Strategy RSI Overbought Guard Test (Negative Test)
# ----------------------------------------------------------------------
def test_orb_strategy_rsi_overbought_guard():
    config.STRATEGY_ORB_ENABLED = True
    config.ORB_MIN_VOLUME_RATIO = 1.0
    config.ORB_MIN_RANGE_ATR_RATIO = 0.1
    config.DEFAULT_TARGET_RR_RATIO = 2.0
    config.ORB_WINDOW_MINUTES = 5

    config.VOLUME_MA_PERIOD = 2
    config.ATR_PERIOD = 2
    config.RSI_PERIOD = 14

    times_strat = pd.date_range("2026-07-20 09:30:00", periods=20, freq="1min", tz="US/Eastern")
    mock_data_strat = {
        "open":   [100.0] * 20,
        "high":   [101.0] * 20,
        "low":    [99.0] * 20,
        "close":  [100.0] * 20,
        "volume": [100.0] * 20
    }
    df_strat = pd.DataFrame(mock_data_strat, index=times_strat)

    df_strat.iloc[-1, df_strat.columns.get_loc("close")] = 105.0
    df_strat.iloc[-1, df_strat.columns.get_loc("high")] = 105.0

    orb_strat = ORBStrategy()
    signal = orb_strat.evaluate("AAPL", df_strat)
    assert signal is None


# ----------------------------------------------------------------------
# 4. VWAP Bounce Strategy Signal Trigger Test (Happy Path)
# ----------------------------------------------------------------------
def test_vwap_bounce_strategy_signal_trigger_happy_path():
    config.STRATEGY_VWAP_BOUNCE_ENABLED = True
    config.VWAP_BOUNCE_TOLERANCE_PCT = 0.5
    config.VWAP_RECLAIM_BARS = 3
    config.VWAP_RSI_OVERBOUGHT = 95.0
    config.VWAP_RSI_OVERSOLD = 35.0

    config.VOLUME_MA_PERIOD = 2
    config.ATR_PERIOD = 2
    config.RSI_PERIOD = 14

    times_vwap = pd.date_range("2026-07-20 09:30:00", periods=30, freq="1min", tz="US/Eastern")
    mock_data_vwap = {
        "open":   [100.0] * 30,
        "high":   [100.5] * 30,
        "low":    [99.5] * 30,
        "close":  [100.1] * 30,
        "volume": [100.0] * 30
    }
    df_vwap_test = pd.DataFrame(mock_data_vwap, index=times_vwap)

    df_vwap_test.iloc[-4, df_vwap_test.columns.get_loc("low")] = 100.02
    df_vwap_test.iloc[-4, df_vwap_test.columns.get_loc("close")] = 100.05

    df_vwap_test.iloc[-3, df_vwap_test.columns.get_loc("close")] = 100.2
    df_vwap_test.iloc[-2, df_vwap_test.columns.get_loc("close")] = 100.3
    df_vwap_test.iloc[-1, df_vwap_test.columns.get_loc("close")] = 100.4
    df_vwap_test.iloc[-1, df_vwap_test.columns.get_loc("volume")] = 200.0

    vwap_strat = VWAPBounceStrategy()
    signal_vwap = vwap_strat.evaluate("AAPL", df_vwap_test)

    assert signal_vwap is not None
    assert signal_vwap.side == "BUY"
    assert signal_vwap.entry_price == 100.4


# ----------------------------------------------------------------------
# 5. Backtest Calculator PnL math consistency Test
# ----------------------------------------------------------------------
def test_backtest_calculator_pnl_consistency():
    slippage_pct = 0.05 / 100.0
    commission = 1.00
    shares = 100

    entry_price_raw = 101.5
    exit_price_raw = 106.5

    executed_entry = entry_price_raw * (1 + slippage_pct)
    executed_exit = exit_price_raw * (1 - slippage_pct)

    expected_net_pnl = (executed_exit - executed_entry) * shares - commission
    actual_net_pnl = _realized_pnl("BUY", executed_entry, executed_exit, shares, commission)

    assert np.isclose(actual_net_pnl, expected_net_pnl)
    assert np.isclose(actual_net_pnl, 488.60)


# ----------------------------------------------------------------------
# 6. Confidence Score Formula Breakdown Test
# ----------------------------------------------------------------------
def test_confidence_score_formula_breakdown():
    config.STRATEGY_ORB_ENABLED = True
    config.ORB_MIN_VOLUME_RATIO = 1.0
    config.ORB_MIN_RANGE_ATR_RATIO = 0.1
    config.DEFAULT_TARGET_RR_RATIO = 2.0
    config.ORB_WINDOW_MINUTES = 5

    config.VOLUME_MA_PERIOD = 2
    config.ATR_PERIOD = 2
    config.RSI_PERIOD = 14

    times_strat = pd.date_range("2026-07-20 09:30:00", periods=20, freq="1min", tz="US/Eastern")
    mock_data_strat = {
        "open":   [100.0] * 20,
        "high":   [101.0] * 20,
        "low":    [99.0] * 20,
        "close":  [100.0] * 20,
        "volume": [100.0] * 20
    }
    df_strat = pd.DataFrame(mock_data_strat, index=times_strat)

    df_strat.iloc[6, df_strat.columns.get_loc("close")] = 97.0
    df_strat.iloc[6, df_strat.columns.get_loc("low")] = 97.0
    df_strat.iloc[12, df_strat.columns.get_loc("close")] = 96.0
    df_strat.iloc[12, df_strat.columns.get_loc("low")] = 96.0

    df_strat.iloc[-1, df_strat.columns.get_loc("close")] = 101.5
    df_strat.iloc[-1, df_strat.columns.get_loc("high")] = 101.5
    df_strat.iloc[-1, df_strat.columns.get_loc("volume")] = 150.0

    orb_strat = ORBStrategy()
    signal = orb_strat.evaluate("AAPL", df_strat)
    
    expected_confidence = 0.5 + (1.2 * 0.1) + (2.0 / 2.2855224609375) * 0.1
    assert np.isclose(signal.confidence, expected_confidence)


# ----------------------------------------------------------------------
# 7. Edge Case Test: Gap-up Days ORB High/Low boundaries isolation
# ----------------------------------------------------------------------
def test_orb_gap_up_boundary_isolation():
    times_day1 = pd.date_range("2026-07-20 15:58:00", periods=3, freq="1min", tz="US/Eastern")
    times_day2 = pd.date_range("2026-07-21 09:30:00", periods=10, freq="1min", tz="US/Eastern")
    
    mock_day1 = {
        "open": [100.0]*3, "high": [100.1]*3, "low": [99.9]*3, "close": [100.0]*3, "volume": [100.0]*3
    }
    mock_day2 = {
        "open": [120.0]*10, "high": [121.0]*10, "low": [119.0]*10, "close": [120.0]*10, "volume": [100.0]*10
    }
    
    df1 = pd.DataFrame(mock_day1, index=times_day1)
    df2 = pd.DataFrame(mock_day2, index=times_day2)
    
    df = pd.concat([df1, df2])
    
    config.ORB_WINDOW_MINUTES = 5
    df_res = add_opening_range(df, orb_minutes=5)
    
    day2_mask = df_res.index.date == pd.Timestamp("2026-07-21").date()
    df_day2 = df_res.loc[day2_mask]
    
    assert np.isnan(df_day2["orb_high"].iloc[0])
    assert np.isnan(df_day2["orb_low"].iloc[0])
    assert df_day2["orb_high"].iloc[-1] == 121.0
    assert df_day2["orb_low"].iloc[-1] == 119.0


# ----------------------------------------------------------------------
# 8. Coverage Expansion: Target indicators.py Coverage Gaps to reach 100%
# ----------------------------------------------------------------------
def test_indicators_coverage_gaps(monkeypatch):
    # 8.1 Missing close or volume columns in add_vwap (lines 17-18)
    df_missing = pd.DataFrame({"open": [100.0], "high": [101.0], "low": [99.0]})
    df_res_missing = add_vwap(df_missing)
    assert df_res_missing["vwap"].isna().all()

    # 8.2 Non-datetime index in add_vwap dates Series fallback (line 26)
    df_string_idx = pd.DataFrame({
        "open": [100.0]*3, "high": [101.0]*3, "low": [99.0]*3, "close": [100.0]*3, "volume": [100.0]*3
    }, index=["a", "b", "c"])
    df_res_string_idx = add_vwap(df_string_idx)
    assert "vwap" in df_res_string_idx.columns

    # 8.3 Non-datetime index in add_opening_range (line 125)
    df_no_datetime = pd.DataFrame({
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [100.0]
    }, index=["row1"])
    df_res_no_datetime = add_opening_range(df_no_datetime)
    assert "orb_high" in df_res_no_datetime.columns
    assert df_res_no_datetime["orb_high"].isna().all()

    # 8.4 Naive index tz conversion inside add_opening_range (line 130)
    times_naive = pd.date_range("2026-07-20 09:30:00", periods=3, freq="1min")
    df_naive = pd.DataFrame({
        "open": [100.0]*3, "high": [101.0]*3, "low": [99.0]*3, "close": [100.0]*3, "volume": [100.0]*3
    }, index=times_naive)
    df_naive_orb = add_opening_range(df_naive, orb_minutes=5)
    assert "orb_high" in df_naive_orb.columns

    # 8.5 Empty day df fallback via monkeypatch (line 144)
    def mock_unique(self):
        return [pd.Timestamp("2099-01-01").date()]
    monkeypatch.setattr(pd.Series, "unique", mock_unique)
    df_res_empty_day = add_opening_range(df_naive, orb_minutes=5)
    assert df_res_empty_day["orb_high"].isna().all()
    monkeypatch.undo()

    # 8.6 Premarket-only day mask (line 152)
    times_premarket = pd.date_range("2026-07-20 08:30:00", periods=5, freq="1min", tz="US/Eastern")
    df_premarket = pd.DataFrame({
        "open": [100.0]*5, "high": [101.0]*5, "low": [99.0]*5, "close": [100.0]*5, "volume": [100.0]*5
    }, index=times_premarket)
    df_res_premarket = add_opening_range(df_premarket, orb_minutes=5)
    assert df_res_premarket["orb_high"].isna().all()

    # 8.7 Columns already computed early return in add_all_indicators (line 170)
    df_existing = pd.DataFrame({
        "ema_9": [100.0], "vwap": [100.0], "close": [100.0]
    })
    df_res_existing = add_all_indicators(df_existing)
    assert "rsi" not in df_res_existing.columns

# 9. Backtest Engine End-to-End Simulation Test
# ----------------------------------------------------------------------
def test_run_backtest_happy_path(monkeypatch):
    """Verify that run_backtest steps through bars, opens trades, and registers exits correctly."""
    # 9.1 Set up mock data of 40 rows
    times = pd.date_range("2026-07-20 09:30:00", periods=40, freq="1min", tz="US/Eastern")
    mock_data = {
        "open":   [100.0] * 40,
        "high":   [101.0] * 40,
        "low":    [99.0] * 40,
        "close":  [100.0] * 40,
        "volume": [100.0] * 40
    }
    df = pd.DataFrame(mock_data, index=times)

    # 9.2 Add some down bars to cool down RSI early on
    df.iloc[6, df.columns.get_loc("close")] = 97.0
    df.iloc[6, df.columns.get_loc("low")] = 97.0
    df.iloc[12, df.columns.get_loc("close")] = 96.0
    df.iloc[12, df.columns.get_loc("low")] = 96.0

    # 9.3 Setup breakout row at row 32 (well after opening range of 5 minutes)
    df.iloc[32, df.columns.get_loc("close")] = 101.5
    df.iloc[32, df.columns.get_loc("high")] = 101.5
    df.iloc[32, df.columns.get_loc("volume")] = 150.0

    # 9.4 Setup exit touch target on next bar at row 33
    # We raise the low to 100.0 so that the stop loss (99.0) is not triggered!
    df.iloc[33, df.columns.get_loc("high")] = 108.0
    df.iloc[33, df.columns.get_loc("low")] = 100.0
    df.iloc[33, df.columns.get_loc("close")] = 107.0

    # 9.5 Mock yfinance fetch inside backtester module namespace to return this df
    def mock_fetch(symbol, lookback_days=5):
        return df

    import strategies.backtester
    monkeypatch.setattr(strategies.backtester, "fetch_intraday_yfinance", mock_fetch)

    # 9.6 Setup configuration overrides
    config.STRATEGY_ORB_ENABLED = True
    config.ORB_MIN_VOLUME_RATIO = 1.0
    config.ORB_MIN_RANGE_ATR_RATIO = 0.1
    config.DEFAULT_TARGET_RR_RATIO = 2.0
    config.ORB_WINDOW_MINUTES = 5

    config.VOLUME_MA_PERIOD = 2
    config.ATR_PERIOD = 2
    config.RSI_PERIOD = 14

    config.SIZING_METHOD = "fixed"
    config.FIXED_SHARE_COUNT = 100
    config.BACKTEST_COMMISSION_PER_TRADE = 1.00
    config.BACKTEST_SLIPPAGE_FRAC = 0.0005
    config.BACKTEST_SLIPPAGE_PCT = 0.0005

    # 9.7 Run the backtester
    result = strategies.backtester.run_backtest("AAPL", "ORB", lookback_days=5)

    # 9.8 Assertions on results dictionary
    assert result["total_trades"] == 1
    assert result["wins"] == 1
    assert result["losses"] == 0
    assert result["win_rate"] == 100.0
    # Net P&L should match realized trade profit of $488.60 (shares = 100, RR = 2.0)
    assert np.isclose(result["net_pnl"], 488.60)


# ----------------------------------------------------------------------
# 10. Same-Symbol Deduplication - Highest Confidence Test
# ----------------------------------------------------------------------
def test_same_symbol_dedup_highest_confidence(monkeypatch):
    from unittest import mock
    from strategies.base import TradeSignal
    from strategies.orchestrator import evaluate_and_execute
    import strategies.orchestrator
    
    # 1. Create multiple signals for the same symbol
    sig_orb = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        risk_per_share=2.0,
        confidence=0.88,
        strategy="ORB",
        atr=1.0,
        reason="ORB trigger"
    )
    sig_mom = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        risk_per_share=2.0,
        confidence=0.81,
        strategy="Momentum",
        atr=1.0,
        reason="Momentum trigger"
    )
    sig_gap = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=150.0,
        stop_price=148.0,
        target_price=154.0,
        risk_per_share=2.0,
        confidence=0.63,
        strategy="Gap",
        atr=1.0,
        reason="Gap trigger"
    )
    
    # Mock evaluate_symbols_parallel to return all 3 signals
    monkeypatch.setattr(strategies.orchestrator, "evaluate_symbols_parallel", lambda w, bridge: [sig_mom, sig_orb, sig_gap])
    
    # Mock execute_signal to capture calls
    executed_signals = []
    def mock_execute(signal, bridge, equity, current_pnl, day_trades_last_5_days, dry_run):
        executed_signals.append(signal)
        return {"status": "DRY_RUN", "reason": "Mocked execution"}
    
    monkeypatch.setattr(strategies.orchestrator, "execute_signal", mock_execute)
    monkeypatch.setattr(strategies.orchestrator, "apply_portfolio_correlation", lambda sigs, open_pos, bridge: sigs)
    monkeypatch.setattr(strategies.orchestrator, "today_pnl", lambda: 0.0)
    monkeypatch.setattr(strategies.orchestrator, "day_trades_in_last_5_days", lambda: 0)
    
    # Mock bridge
    bridge_mock = mock.MagicMock()
    bridge_mock.is_connected = False
    
    evaluate_and_execute(["AAPL"], bridge_mock, live_paper=False)
    
    # Verify that only the highest confidence signal (sig_orb, 0.88) was executed
    assert len(executed_signals) == 1
    assert executed_signals[0].strategy == "ORB"
    assert executed_signals[0].confidence == 0.88


# ----------------------------------------------------------------------
# 11. Orchestrator active position / pending order skips
# ----------------------------------------------------------------------
def test_orchestrator_skips_active_and_pending_symbols(monkeypatch):
    from unittest import mock
    from types import SimpleNamespace
    from strategies.base import TradeSignal
    from strategies.orchestrator import evaluate_and_execute
    import strategies.orchestrator
    
    sig_aapl = TradeSignal(symbol="AAPL", side="BUY", entry_price=100.0, stop_price=90.0, target_price=120.0, risk_per_share=10.0, confidence=0.85, strategy="ORB", atr=1.0, reason="")
    sig_msft = TradeSignal(symbol="MSFT", side="BUY", entry_price=200.0, stop_price=190.0, target_price=220.0, risk_per_share=10.0, confidence=0.80, strategy="ORB", atr=1.0, reason="")
    sig_goog = TradeSignal(symbol="GOOG", side="BUY", entry_price=300.0, stop_price=290.0, target_price=320.0, risk_per_share=10.0, confidence=0.75, strategy="ORB", atr=1.0, reason="")
    
    monkeypatch.setattr(strategies.orchestrator, "evaluate_symbols_parallel", lambda w, bridge: [sig_aapl, sig_msft, sig_goog])
    
    executed_signals = []
    def mock_execute(signal, bridge, equity, current_pnl, day_trades_last_5_days, dry_run):
        executed_signals.append(signal)
        return {"status": "DRY_RUN", "reason": "Mocked execution"}
    
    monkeypatch.setattr(strategies.orchestrator, "execute_signal", mock_execute)
    monkeypatch.setattr(strategies.orchestrator, "apply_portfolio_correlation", lambda sigs, open_pos, bridge: sigs)
    monkeypatch.setattr(strategies.orchestrator, "today_pnl", lambda: 0.0)
    monkeypatch.setattr(strategies.orchestrator, "day_trades_in_last_5_days", lambda: 0)
    
    # Mock bridge
    bridge_mock = mock.MagicMock()
    bridge_mock.is_connected = True
    bridge_mock.market_price.return_value = 100.0
    
    # AAPL has position
    class FakePosition:
        def __init__(self, symbol, position):
            self.contract = SimpleNamespace(symbol=symbol)
            self.position = position
            self.avgCost = 100.0
            self.marketValue = 100.0
            self.unrealizedPnL = 0.0
    
    bridge_mock.ib.positions.return_value = [FakePosition("AAPL", 10)]
    
    # MSFT has open order
    class FakeOrder:
        def __init__(self, symbol):
            self.symbol = symbol
    
    bridge_mock.ib.openTrades.return_value = [FakeOrder("MSFT")]
    
    strategies.orchestrator._last_evaluated_5m_bar_str = None
    evaluate_and_execute(["AAPL", "MSFT", "GOOG"], bridge_mock, live_paper=True)
    
    # Verify:
    # AAPL (active position) and MSFT (pending order) must be skipped.
    # GOOG (clean symbol) must be executed successfully.
    executed_symbols = {s.symbol for s in executed_signals}
    assert "GOOG" in executed_symbols
    assert "AAPL" not in executed_symbols
    assert "MSFT" not in executed_symbols
    assert len(executed_symbols) == 1


# ----------------------------------------------------------------------
# 12. Orchestrator Fail-Closed on Broker Outage Test
# ----------------------------------------------------------------------
def test_orchestrator_fail_closed_on_broker_outage(monkeypatch):
    from unittest import mock
    from strategies.base import TradeSignal
    from strategies.orchestrator import evaluate_and_execute
    import strategies.orchestrator
    
    sig_goog = TradeSignal(symbol="GOOG", side="BUY", entry_price=300.0, stop_price=290.0, target_price=320.0, risk_per_share=10.0, confidence=0.75, strategy="ORB", atr=1.0, reason="")
    monkeypatch.setattr(strategies.orchestrator, "evaluate_symbols_parallel", lambda w, bridge: [sig_goog])
    
    executed_signals = []
    def mock_execute(signal, bridge, equity, current_pnl, day_trades_last_5_days, dry_run):
        executed_signals.append(signal)
        return {"status": "DRY_RUN", "reason": "Mocked execution"}
    
    monkeypatch.setattr(strategies.orchestrator, "execute_signal", mock_execute)
    monkeypatch.setattr(strategies.orchestrator, "apply_portfolio_correlation", lambda sigs, open_pos, bridge: sigs)
    monkeypatch.setattr(strategies.orchestrator, "today_pnl", lambda: 0.0)
    monkeypatch.setattr(strategies.orchestrator, "day_trades_in_last_5_days", lambda: 0)
    
    # 1. Test Broker Outage on positions in Live Paper (fail-closed)
    bridge_mock_1 = mock.MagicMock()
    bridge_mock_1.is_connected = True
    bridge_mock_1.ib.positions.side_effect = RuntimeError("Broker connection lost")
    
    strategies.orchestrator._last_evaluated_5m_bar_str = None
    evaluate_and_execute(["GOOG"], bridge_mock_1, live_paper=True)
    assert len(executed_signals) == 0  # Should be skipped/blocked
    
    # 2. Test Broker Outage on openTrades in Live Paper (fail-closed)
    bridge_mock_2 = mock.MagicMock()
    bridge_mock_2.is_connected = True
    bridge_mock_2.ib.positions.return_value = []
    bridge_mock_2.ib.openTrades.side_effect = RuntimeError("Broker API error")
    
    strategies.orchestrator._last_evaluated_5m_bar_str = None
    evaluate_and_execute(["GOOG"], bridge_mock_2, live_paper=True)
    assert len(executed_signals) == 0  # Should be skipped/blocked
    
    # 3. Test Broker Outage in Dry Run (should NOT block, since it's not live)
    strategies.orchestrator._last_evaluated_5m_bar_str = None
    evaluate_and_execute(["GOOG"], bridge_mock_2, live_paper=False)
    assert len(executed_signals) == 1
    assert executed_signals[0].symbol == "GOOG"


# ----------------------------------------------------------------------
# 13. Strategy Priority Sorting Test
# ----------------------------------------------------------------------
def test_strategy_priority_sorting(monkeypatch):
    from unittest import mock
    from strategies.base import TradeSignal
    from strategies.orchestrator import evaluate_and_execute
    import strategies.orchestrator
    
    # sig_gap has lower priority (4) but higher confidence (0.95)
    sig_gap = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=100.0,
        stop_price=90.0,
        target_price=120.0,
        risk_per_share=10.0,
        confidence=0.95,
        strategy="GAP_AND_GO",
        atr=1.0,
        reason=""
    )
    # sig_orb has higher priority (1) but lower confidence (0.80)
    sig_orb = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=100.0,
        stop_price=90.0,
        target_price=120.0,
        risk_per_share=10.0,
        confidence=0.80,
        strategy="ORB",
        atr=1.0,
        reason=""
    )
    
    monkeypatch.setattr(strategies.orchestrator, "evaluate_symbols_parallel", lambda w, bridge: [sig_gap, sig_orb])
    
    executed_signals = []
    def mock_execute(signal, bridge, equity, current_pnl, day_trades_last_5_days, dry_run):
        executed_signals.append(signal)
        return {"status": "DRY_RUN", "reason": "Mocked execution"}
        
    monkeypatch.setattr(strategies.orchestrator, "execute_signal", mock_execute)
    monkeypatch.setattr(strategies.orchestrator, "apply_portfolio_correlation", lambda sigs, open_pos, bridge: sigs)
    monkeypatch.setattr(strategies.orchestrator, "today_pnl", lambda: 0.0)
    monkeypatch.setattr(strategies.orchestrator, "day_trades_in_last_5_days", lambda: 0)
    
    bridge_mock = mock.MagicMock()
    bridge_mock.is_connected = False
    
    strategies.orchestrator._last_evaluated_5m_bar_str = None
    evaluate_and_execute(["AAPL"], bridge_mock, live_paper=False)
    
    # Should execute sig_orb first and discard sig_gap due to same-symbol deduplication
    assert len(executed_signals) == 1
    assert executed_signals[0].strategy == "ORB"
    assert executed_signals[0].confidence == 0.80


# ----------------------------------------------------------------------
# 14. Strategy-Specific Time Cutoff Test
# ----------------------------------------------------------------------
def test_strategy_specific_time_cutoff(monkeypatch):
    from unittest import mock
    from datetime import datetime, time
    from strategies.base import TradeSignal
    from strategies.orchestrator import evaluate_and_execute
    import strategies.orchestrator
    
    # GAP_AND_GO has cutoff at 10:30
    sig_gap = TradeSignal(symbol="MSFT", side="BUY", entry_price=100.0, stop_price=90.0, target_price=120.0, risk_per_share=10.0, confidence=0.85, strategy="GAP_AND_GO", atr=1.0, reason="")
    # ORB has no cutoff
    sig_orb = TradeSignal(symbol="AAPL", side="BUY", entry_price=100.0, stop_price=90.0, target_price=120.0, risk_per_share=10.0, confidence=0.85, strategy="ORB", atr=1.0, reason="")
    
    monkeypatch.setattr(strategies.orchestrator, "evaluate_symbols_parallel", lambda w, bridge: [sig_gap, sig_orb])
    
    executed_signals = []
    def mock_execute(signal, bridge, equity, current_pnl, day_trades_last_5_days, dry_run):
        executed_signals.append(signal)
        return {"status": "DRY_RUN", "reason": "Mocked execution"}
        
    monkeypatch.setattr(strategies.orchestrator, "execute_signal", mock_execute)
    monkeypatch.setattr(strategies.orchestrator, "apply_portfolio_correlation", lambda sigs, open_pos, bridge: sigs)
    monkeypatch.setattr(strategies.orchestrator, "today_pnl", lambda: 0.0)
    monkeypatch.setattr(strategies.orchestrator, "day_trades_in_last_5_days", lambda: 0)
    
    # Mock current time to be 11:00 AM (past the GAP_AND_GO cutoff of 10:30 AM)
    class MockDatetime:
        @classmethod
        def now(cls, tz=None):
            from zoneinfo import ZoneInfo
            return datetime(2026, 7, 20, 11, 0, 0, tzinfo=ZoneInfo("US/Eastern"))
            
    # Mock session's now_eastern to return our custom time
    import strategies.session
    monkeypatch.setattr(strategies.session, "now_eastern", MockDatetime.now)
    
    bridge_mock = mock.MagicMock()
    bridge_mock.is_connected = False
    
    strategies.orchestrator._last_evaluated_5m_bar_str = None
    evaluate_and_execute(["AAPL", "MSFT"], bridge_mock, live_paper=False)
    
    # GAP_AND_GO (MSFT) should be skipped/filtered out.
    # ORB (AAPL) should execute successfully because it has no cutoff time.
    executed_symbols = {s.symbol for s in executed_signals}
    assert "AAPL" in executed_symbols
    assert "MSFT" not in executed_symbols
    assert len(executed_symbols) == 1


# ----------------------------------------------------------------------
# 15. Consecutive Outage Alert Test
# ----------------------------------------------------------------------
def test_consecutive_outage_alert(monkeypatch):
    from unittest import mock
    from strategies.orchestrator import evaluate_and_execute
    import strategies.orchestrator
    import config
    
    # Reset failures count
    strategies.orchestrator._consecutive_broker_failures = 9  # Almost at limit 10
    config.CONSECUTIVE_OUTAGE_LIMIT = 10
    
    # Mock webhook alert to verify it is called exactly once
    alerts_sent = []
    def mock_send_alert(msg):
        alerts_sent.append(msg)
        
    import strategies.webhook
    monkeypatch.setattr(strategies.webhook, "send_discord_alert", mock_send_alert)
    monkeypatch.setattr(strategies.orchestrator, "evaluate_symbols_parallel", lambda w, bridge: [])
    
    # Outage 1 (hits limit 10)
    bridge_mock_1 = mock.MagicMock()
    bridge_mock_1.is_connected = True
    bridge_mock_1.ib.positions.side_effect = RuntimeError("Outage")
    
    evaluate_and_execute([], bridge_mock_1, live_paper=True)
    assert len(alerts_sent) == 1
    assert "10 consecutive broker communication failures" in alerts_sent[0]
    assert strategies.orchestrator._consecutive_broker_failures == 10
    
    # Outage 2 (at 11, should NOT alert again since it's only sent once when it equals the limit)
    evaluate_and_execute([], bridge_mock_1, live_paper=True)
    assert len(alerts_sent) == 1  # Still 1
    assert strategies.orchestrator._consecutive_broker_failures == 11
    
    # Success resets to 0
    bridge_mock_success = mock.MagicMock()
    bridge_mock_success.is_connected = True
    bridge_mock_success.ib.positions.return_value = []
    bridge_mock_success.ib.openTrades.return_value = []
    
    evaluate_and_execute([], bridge_mock_success, live_paper=True)
    assert strategies.orchestrator._consecutive_broker_failures == 0


# ----------------------------------------------------------------------
# 16. Production Shape MockOrder Symbol Filter Test
# ----------------------------------------------------------------------
def test_production_shape_mockorder_filter(monkeypatch):
    from unittest import mock
    from strategies.base import TradeSignal
    from strategies.orchestrator import evaluate_and_execute
    import strategies.orchestrator
    
    # sig_goog has strategy ORB
    sig_goog = TradeSignal(symbol="GOOG", side="BUY", entry_price=300.0, stop_price=290.0, target_price=320.0, risk_per_share=10.0, confidence=0.75, strategy="ORB", atr=1.0, reason="")
    monkeypatch.setattr(strategies.orchestrator, "evaluate_symbols_parallel", lambda w, bridge: [sig_goog])
    
    executed_signals = []
    def mock_execute(signal, bridge, equity, current_pnl, day_trades_last_5_days, dry_run):
        executed_signals.append(signal)
        return {"status": "DRY_RUN", "reason": "Mocked execution"}
        
    monkeypatch.setattr(strategies.orchestrator, "execute_signal", mock_execute)
    monkeypatch.setattr(strategies.orchestrator, "apply_portfolio_correlation", lambda sigs, open_pos, bridge: sigs)
    monkeypatch.setattr(strategies.orchestrator, "today_pnl", lambda: 0.0)
    monkeypatch.setattr(strategies.orchestrator, "day_trades_in_last_5_days", lambda: 0)
    
    # MockOrder object that does NOT have symbol attribute but has contract.symbol
    class TestMockOrder:
        def __init__(self, symbol):
            class Contract:
                def __init__(self, s):
                    self.symbol = s
            self.contract = Contract(symbol)
            self.id = "mock-id-123"
            
    bridge_mock = mock.MagicMock()
    bridge_mock.is_connected = True
    bridge_mock.ib.positions.return_value = []
    bridge_mock.ib.openTrades.return_value = [TestMockOrder("GOOG")]
    
    # GOOG should be skipped because there is a working order for it.
    evaluate_and_execute(["GOOG"], bridge_mock, live_paper=True)
    assert len(executed_signals) == 0

    # Test completely invalid order shape (no symbol and no contract)
    class BadMockOrder:
        pass
    
    bridge_mock_bad = mock.MagicMock()
    bridge_mock_bad.is_connected = True
    bridge_mock_bad.ib.positions.return_value = []
    bridge_mock_bad.ib.openTrades.return_value = [BadMockOrder()]
    
    executed_signals_bad = []
    def mock_execute_bad(signal, bridge, equity, current_pnl, day_trades_last_5_days, dry_run):
        executed_signals_bad.append(signal)
        return {"status": "DRY_RUN", "reason": "Mocked execution"}
    monkeypatch.setattr(strategies.orchestrator, "execute_signal", mock_execute_bad)
    
    # Evaluating signals should trigger fail-closed (allow_new_entries=False), blocking GOOG
    evaluate_and_execute(["GOOG"], bridge_mock_bad, live_paper=True)
    assert len(executed_signals_bad) == 0


# ----------------------------------------------------------------------
# 17. Naked Position with Existing Stop & Cancel-then-Flatten Test
# ----------------------------------------------------------------------
def test_naked_position_with_existing_stop_and_flatten(monkeypatch):
    from unittest import mock
    from strategies.trailing_stop import manager, TrailingStopState
    
    # Setup trailing stop manager states
    import time
    manager.reset("AAPL")
    state = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=150.0,
        peak_price=150.0,
        atr=2.0,
        stop_price=145.0,
        trail_multiple=2.5,
        last_updated=time.time(),
        active=True,
        order_id=None
    )
    manager.states["AAPL"] = state
    
    class TestMockOrder:
        def __init__(self, symbol, order_id):
            class Contract:
                def __init__(self, s):
                    self.symbol = s
            self.contract = Contract(symbol)
            self.id = order_id
            self.side = "sell"
            self.stop_price = 145.0
            
    class TestMockPosition:
        def __init__(self, symbol, qty):
            class Contract:
                def __init__(self, s):
                    self.symbol = s
            self.contract = Contract(symbol)
            self.position = qty
            
    bridge_mock = mock.MagicMock()
    bridge_mock.ib.positions.return_value = [TestMockPosition("AAPL", 100.0)]
    # Mock openTrades to return an order with the correct stop price, allowing reconciliation
    bridge_mock.ib.openTrades.return_value = [TestMockOrder("AAPL", "broker-stop-order-999")]
    
    # 1. Test existing stop order adoption (reconciliation)
    ret_state = manager.ensure_initialized(
        symbol="AAPL",
        side="BUY",
        avg_cost=150.0,
        open_orders=[TestMockOrder("AAPL", "broker-stop-order-999")],
        current_price=149.0
    )
    assert ret_state.order_id == "broker-stop-order-999"
    
    # Reset order_id to trigger naked position flow
    state.order_id = None
    
    # Mock market price to be below stop price, forcing fallback to flatten
    bridge_mock.market_price.return_value = 140.0 # Stale/invalid stop (145.0 > 140.0)
    
    # Mock close_position and cancel_order to record calls
    cancelled_orders = []
    def mock_cancel(order_id):
        cancelled_orders.append(order_id)
        return True
    bridge_mock.cancel_order = mock_cancel
    
    flattened_symbols = []
    def mock_close(symbol):
        flattened_symbols.append(symbol)
        return True
    bridge_mock.close_position = mock_close
    
    # Mock discord alert to not make HTTP calls
    import strategies.trailing_stop
    monkeypatch.setattr(strategies.trailing_stop, "send_discord_alert", lambda msg: None)
    
    # 2. Test emergency flatten calls cancel_order first then close_position
    manager.handle_naked_position("AAPL", 100, bridge_mock, dry_run=False)
    
    assert "broker-stop-order-999" in cancelled_orders
    assert "AAPL" in flattened_symbols
    assert state.active is False


def test_bracket_leg_held_status_not_naked(monkeypatch):
    import time
    from unittest import mock
    from strategies.trailing_stop import manager, TrailingStopState
    
    manager.reset("AAPL")
    state = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=150.0,
        peak_price=150.0,
        atr=2.0,
        stop_price=145.0,
        trail_multiple=2.5,
        last_updated=time.time(),
        active=True,
        order_id=None # starts with None to force reconciliation
    )
    manager.states["AAPL"] = state
    
    class TestMockOrderHeld:
        def __init__(self, symbol, order_id, status="held"):
            class Contract:
                def __init__(self, s):
                    self.symbol = s
            self.contract = Contract(symbol)
            self.id = order_id
            self.side = "sell"
            self.stop_price = 145.0
            self.status = status # 'held' or 'suspended'
            
    # Mock open_orders returned to contain a held stop order
    mock_held_order = TestMockOrderHeld("AAPL", "broker-stop-order-held", status="held")
    
    # ensure_initialized should find and adopt the held order, setting state.order_id to 'broker-stop-order-held'
    ret_state = manager.ensure_initialized(
        symbol="AAPL",
        side="BUY",
        avg_cost=150.0,
        open_orders=[mock_held_order],
        current_price=149.0
    )
    
    assert ret_state.order_id == "broker-stop-order-held"
    assert ret_state.active is True


def test_regression_c1_pre_trade_check_open_symbols():
    """Regression test C1: verify pre_trade_check accepts open_symbols without TypeError."""
    from strategies.intraday_risk import pre_trade_check
    result = pre_trade_check(
        equity=100000.0,
        current_pnl=50.0,
        trades_today=1,
        open_positions=1,
        day_trades_last_5_days=0,
        risk_dollars=100.0,
        symbol="AAPL",
        max_risk_pct=1.0,
        open_symbols=["MSFT"],
    )
    # Should pass without TypeError
    assert isinstance(result, str) or result is None or result == ""


def test_regression_c6_orb_new_york_nan_masking():
    """Regression test C6: verify ORB high/low are masked as NaN prior to 09:45 AM America/New_York."""
    import pandas as pd
    import numpy as np
    from strategies.indicators import add_opening_range
    import config

    # Generate 5-min bars starting at 09:30 AM America/New_York (13:30 UTC)
    idx = pd.date_range("2026-07-24 13:30:00", periods=6, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5],
        "volume": [1000, 1100, 1200, 1300, 1400, 1500]
    }, index=idx)

    res = add_opening_range(df, orb_minutes=15)
    
    # 09:30 (13:30 UTC), 09:35 (13:35 UTC), 09:40 (13:40 UTC) bars must be NaN
    assert np.isnan(res["orb_high"].iloc[0])
    assert np.isnan(res["orb_high"].iloc[1])
    assert np.isnan(res["orb_high"].iloc[2])
    
    # Post-09:45 AM NY bar (13:45 UTC and later) should have valid numeric ORB high/low
    assert not np.isnan(res["orb_high"].iloc[3])
    assert res["orb_high"].iloc[3] == 104.0  # Max high of bars up to 13:45


def test_regression_c2_canonical_slot_cap_single_source():
    """Regression test C2: verify MAX_OPEN_POSITIONS = 5 is single canonical source across config."""
    import config
    assert config.MAX_OPEN_POSITIONS == 5
    assert getattr(config, "PORTFOLIO_MAX_POSITIONS", None) == 5









