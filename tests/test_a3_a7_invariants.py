import pytest
import pandas as pd
from unittest.mock import MagicMock
from strategies.regime_engine import compute_vwap_breadth
from strategies.order_manager import _cap_submission_shares
import strategies.regime_engine as re_mod

def _make_df(above=True):
    now = pd.Timestamp.now(tz="UTC")
    times = [now - pd.Timedelta(minutes=i) for i in range(10, 0, -1)]
    closes = [100.0 + (i if above else -i) for i in range(10)]
    return pd.DataFrame({"close": closes, "vwap": [100.0] * 10}, index=times)

def test_a3_breadth_holds_last_valid_when_valid_count_drops():
    re_mod._last_valid_breadth_cache = None

    # First cycle: 6 symbols (>= 5 and 100% coverage) -> 4 above (66.67%)
    syms = [f"SYM{i}" for i in range(6)]
    sym_data = {f"SYM{i}": _make_df(above=(i < 4)) for i in range(6)}

    pct, mode, det = compute_vwap_breadth(syms, symbol_data=sym_data)
    assert mode == "WATCHLIST_VWAP"
    assert pytest.approx(pct, 0.01) == 66.67

    # Second cycle: transient drop to 0 valid symbols (< 5)
    # Must hold the last valid breadth (66.67%) instead of jumping to index fallback!
    pct2, mode2, det2 = compute_vwap_breadth(syms, symbol_data={})
    assert pytest.approx(pct2, 0.01) == 66.67
    assert "hold-last" in det2.get("fallback_reason", "").lower()

def test_a5_sizer_cap_reduction():
    # 10 shares requested, but risk allows only 5
    capped = _cap_submission_shares(
        shares=10,
        checked_risk_dollars=50.0,
        checked_notional=10000.0,
        risk_per_share=10.0,
        entry_price=100.0,
    )
    assert capped == 5

def test_a7_index_outlier_quote_suppressed():
    import time
    re_mod._last_index_prices = {"QQQ": (721.22, time.time() - 10.0)}

    bridge = MagicMock()
    # Mock bad quote with spike to 734.15 (+1.79%)
    bad_quote = MagicMock()
    bad_quote.last = 734.15
    bad_quote.timestamp = time.time()
    bridge.get_quote.return_value = bad_quote

    from strategies.session import get_previous_market_day, now_eastern
    prev_day = get_previous_market_day(now_eastern().date())

    bar_mock = MagicMock()
    bar_mock.close = 720.0
    bar_mock.timestamp = pd.Timestamp(f"{prev_day} 20:00:00", tz="UTC")
    bridge.fetch_historical_data.return_value = [bar_mock]

    pct, src, ts = re_mod._get_premarket_pct_change("QQQ", bridge=bridge)
    # Suppressed to 721.22 -> pct = (721.22 - 720.0)/720.0 * 100 = 0.17%
    assert pct is not None
    assert src == "alpaca_quote_vs_prev_close"
    assert pytest.approx(pct, 0.01) == 0.17
