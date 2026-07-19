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

    assert df_orb_res["orb_high"].iloc[0] == 105.0
    assert df_orb_res["orb_low"].iloc[0] == 97.0
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
    
    assert df_day2["orb_high"].iloc[0] == 121.0
    assert df_day2["orb_low"].iloc[0] == 119.0


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
    config.BACKTEST_SLIPPAGE_PCT = 0.05

    # 9.7 Run the backtester
    result = strategies.backtester.run_backtest("AAPL", "ORB", lookback_days=5)

    # 9.8 Assertions on results dictionary
    assert result["total_trades"] == 1
    assert result["wins"] == 1
    assert result["losses"] == 0
    assert result["win_rate"] == 100.0
    # Net P&L should match realized trade profit of $488.60 (shares = 100, RR = 2.0)
    assert np.isclose(result["net_pnl"], 488.60)








