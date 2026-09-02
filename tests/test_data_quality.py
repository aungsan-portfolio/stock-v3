"""
tests/test_data_quality.py -- Verify data quality enhancements:
1. Frozen-feed and stale-feed auto-quarantine (30m TTL).
2. Universe dollar-volume & liquidity gate (with crypto/inverse ETF exemptions).
"""
import pytest
from strategies.intraday_data import (
    validate_feed,
    is_quarantined,
    clear_quarantine,
    QUARANTINE_TTL,
)
from strategies.scanner.premarket_scanner import (
    passes_liquidity_gate,
    build_universe,
)


class FrozenBars:
    frozen = True


class LiveBars:
    frozen = False
    age_seconds = 10.0


class StaleBars:
    frozen = False
    age_seconds = 350.0


@pytest.fixture(autouse=True)
def _reset_quarantine():
    clear_quarantine()
    yield
    clear_quarantine()


# --- 3a: Frozen feed caged once, no repeat warns ---
def test_spdn_frozen_quarantined_once(monkeypatch):
    monkeypatch.setattr("strategies.intraday_data._is_frozen_feed", lambda b: getattr(b, "frozen", False))
    monkeypatch.setattr("strategies.intraday_data._bar_age", lambda b, now=None: getattr(b, "age_seconds", 0.0))

    t0 = 1_000_000.0
    # 21:14 first detect -> caged
    assert validate_feed("SPDN", FrozenBars(), now=t0) == (False, "FROZEN_FEED")
    # 21:15 / 21:20 / 21:25 -> silent QUARANTINED, not re-evaluated
    for dt in (60, 360, 660):
        assert validate_feed("SPDN", FrozenBars(), now=t0 + dt) == (False, "QUARANTINED")


def test_quarantine_auto_release_after_ttl(monkeypatch):
    monkeypatch.setattr("strategies.intraday_data._is_frozen_feed", lambda b: getattr(b, "frozen", False))
    monkeypatch.setattr("strategies.intraday_data._bar_age", lambda b, now=None: getattr(b, "age_seconds", 0.0))

    t0 = 2_000_000.0
    validate_feed("SPDN", FrozenBars(), now=t0)
    assert is_quarantined("SPDN", now=t0 + 100) is True
    assert is_quarantined("SPDN", now=t0 + QUARANTINE_TTL + 1) is False   # released


def test_validate_feed_stale_quote_quarantines(monkeypatch):
    monkeypatch.setattr("strategies.intraday_data._is_frozen_feed", lambda b: False)
    monkeypatch.setattr("strategies.intraday_data._bar_age", lambda b, now=None: getattr(b, "age_seconds", 0.0))

    t0 = 3_000_000.0
    assert validate_feed("VELL", StaleBars(), now=t0) == (False, "STALE")
    # Subsequent calls should return QUARANTINED
    assert validate_feed("VELL", StaleBars(), now=t0 + 30) == (False, "QUARANTINED")


# --- 3b: Dead tickers dropped, watchlist exempt ---
class Bar:
    def __init__(self, c, v):
        self.close = c
        self.volume = v


@pytest.mark.parametrize("sym", ["VELL", "TCAN", "COHH", "INFH"])
def test_dead_tickers_dropped(sym):
    ok, reason = passes_liquidity_gate(sym, Bar(1.20, 10_000))
    assert ok is False
    assert "PRICE<" in reason or "SHVOL<" in reason or "$VOL<" in reason


@pytest.mark.parametrize("sym", ["SH", "SQQQ", "SOXS", "IBIT", "ETHA", "BITO", "MARA", "SPDN"])
def test_watchlist_never_dropped(sym):
    # even with a garbage bar, exempt symbols pass
    ok, reason = passes_liquidity_gate(sym, Bar(0.01, 1))
    assert ok is True
    assert reason == "EXEMPT"


def test_build_universe_filter():
    candidates = ["VELL", "TCAN", "NVDA", "IBIT", "SQQQ"]
    prev_bars = {
        "VELL": Bar(0.50, 20_000),      # drops
        "TCAN": Bar(1.10, 5_000),       # drops
        "NVDA": Bar(125.0, 50_000_000), # keeps
        "IBIT": Bar(35.0, 100_000),     # exempt -> keeps
        "SQQQ": Bar(8.0, 50_000),       # exempt -> keeps
    }
    clean = build_universe(candidates, prev_bars)
    assert "NVDA" in clean
    assert "IBIT" in clean
    assert "SQQQ" in clean
    assert "VELL" not in clean
    assert "TCAN" not in clean
