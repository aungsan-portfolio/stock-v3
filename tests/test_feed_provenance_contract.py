"""
tests/test_feed_provenance_contract.py -- Comprehensive verification suite for Blocker 1:
Canonical Market-Data Feed Provenance Contract.

Verifies the single auditable chain:
[Historical Bar] -> [Decision Price] -> [Real-Time Quote] -> [Feed Concordance] -> [Freshness Delta] -> [Divergence Guard] -> [Execution Eligibility]

Enforces the mandated 11-scenario acceptance matrix plus architectural invariants:
 1. SIP bar + SIP quote -> Concordant & Eligible
 2. IEX bar + IEX quote -> Concordant & Eligible
 3. SIP bar + IEX quote -> CROSS_FEED_PROVENANCE_DISLOCATION
 4. IEX bar + SIP quote -> CROSS_FEED_PROVENANCE_DISLOCATION
 5. Quote stale (> 10s) -> QUOTE_STALE veto
 6. Bar stale (> 300s) -> BAR_STALE veto
 7. Feed identity missing -> FEED_IDENTITY_MISSING veto
 8. Quote timestamp missing (<= 0) -> QUOTE_TIMESTAMP_MISSING veto
 9. Divergence > 1.5% + same feed -> PRICE_DRIFT_DEVIATION_VETO
10. Divergence > 1.5% + different feed -> PROVENANCE_FEED_DISCREPANCY_VETO
11. All valid -> Execution eligible
12. Negative latency check (quote older than bar by > 2s) -> NEGATIVE_LATENCY_ERROR
13. TradeSignal metadata & deterministic hash invariant
14. alpaca_bridge.get_quote explicit feed parameterization
15. intraday_data cache roundtrip preserves df.attrs metadata
16. order_manager.execute_signal fail-closed integration
"""
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from strategies.base import TradeSignal, StrategyName, OrderIntent
from strategies.execution_quote import (
    validate_execution_quote,
    BrokerQuote,
    QuoteValidationResult,
    _parse_timestamp_to_epoch,
)


class MockBridge:
    """Mock broker bridge providing configurable quote payloads."""
    def __init__(self, quote_payload: Optional[Dict[str, Any]] = None):
        self._quote = quote_payload
        self.is_connected = True

    def set_quote(self, quote_payload: Optional[Dict[str, Any]]):
        self._quote = quote_payload

    def get_quote(self, symbol: str, feed: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self._quote is None:
            return None
        q = dict(self._quote)
        if "symbol" not in q:
            q["symbol"] = symbol.upper().strip()
        return q

    def open_position_count(self) -> int:
        return 0

    def has_position(self, symbol: str) -> bool:
        return False

    def has_working_order(self, symbol: str) -> bool:
        return False

    def get_positions(self) -> list:
        return []


# ── Scenario 1: SIP bar + SIP quote ──────────────────────────────────────────
def test_scenario_01_sip_concordant_eligible():
    """Scenario 1: SIP decision bar + SIP quote -> Concordant & Eligible."""
    now = time.time()
    bridge = MockBridge({
        "symbol": "JPM",
        "bid": 338.00,
        "ask": 338.05,
        "last": 338.02,
        "timestamp": now - 0.5,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="JPM",
        side="BUY",
        decision_price=338.00,
        decision_feed="sip",
        bar_timestamp=now - 60.0,
        require_feed_concordance=True,
    )
    assert res.valid is True
    assert res.accepted is True
    assert res.reason == "Quote valid"
    assert res.quote.feed_identity == "sip"


# ── Scenario 2: IEX bar + IEX quote ──────────────────────────────────────────
def test_scenario_02_iex_concordant_eligible():
    """Scenario 2: IEX decision bar + IEX quote -> Concordant & Eligible."""
    now = time.time()
    bridge = MockBridge({
        "symbol": "BAC",
        "bid": 56.25,
        "ask": 56.27,
        "last": 56.26,
        "timestamp": now - 0.2,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "iex",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="BAC",
        side="BUY",
        decision_price=56.26,
        decision_feed="iex",
        bar_timestamp=now - 30.0,
        require_feed_concordance=True,
    )
    assert res.valid is True
    assert res.accepted is True
    assert res.reason == "Quote valid"
    assert res.quote.feed_identity == "iex"


# ── Scenario 3: SIP bar + IEX quote ──────────────────────────────────────────
def test_scenario_03_sip_bar_iex_quote_dislocation():
    """Scenario 3: SIP decision bar + IEX quote -> CROSS_FEED_PROVENANCE_DISLOCATION."""
    now = time.time()
    bridge = MockBridge({
        "symbol": "JPM",
        "bid": 338.00,
        "ask": 338.06,
        "last": 338.03,
        "timestamp": now - 0.5,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "iex",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="JPM",
        side="BUY",
        decision_price=338.00,
        decision_feed="sip",
        bar_timestamp=now - 60.0,
        require_feed_concordance=True,
    )
    assert res.valid is False
    assert "CROSS_FEED_PROVENANCE_DISLOCATION" in res.reason
    assert "decision_feed=sip vs quote_feed=iex" in res.reason


# ── Scenario 4: IEX bar + SIP quote ──────────────────────────────────────────
def test_scenario_04_iex_bar_sip_quote_dislocation():
    """Scenario 4: IEX decision bar + SIP quote -> CROSS_FEED_PROVENANCE_DISLOCATION."""
    now = time.time()
    bridge = MockBridge({
        "symbol": "AAPL",
        "bid": 220.00,
        "ask": 220.05,
        "last": 220.02,
        "timestamp": now - 0.5,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="AAPL",
        side="BUY",
        decision_price=220.00,
        decision_feed="iex",
        bar_timestamp=now - 60.0,
        require_feed_concordance=True,
    )
    assert res.valid is False
    assert "CROSS_FEED_PROVENANCE_DISLOCATION" in res.reason
    assert "decision_feed=iex vs quote_feed=sip" in res.reason


# ── Scenario 5: Quote stale (> 10s) ──────────────────────────────────────────
def test_scenario_05_quote_stale_veto():
    """Scenario 5: Quote older than max_age_seconds (10s) -> QUOTE_STALE veto."""
    now = time.time()
    bridge = MockBridge({
        "symbol": "JPM",
        "bid": 338.00,
        "ask": 338.05,
        "last": 338.02,
        "timestamp": now - 15.0,  # 15s stale
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="JPM",
        side="BUY",
        decision_price=338.00,
        decision_feed="sip",
        bar_timestamp=now - 60.0,
        max_age_seconds=10.0,
        require_feed_concordance=True,
    )
    assert res.valid is False
    assert "QUOTE_STALE" in res.reason
    assert "limit=10.0s" in res.reason


# ── Scenario 6: Bar stale (> 300s) ───────────────────────────────────────────
def test_scenario_06_bar_stale_veto():
    """Scenario 6: Decision bar older than max_bar_age_seconds (300s) -> BAR_STALE veto."""
    now = time.time()
    bridge = MockBridge({
        "symbol": "JPM",
        "bid": 338.00,
        "ask": 338.05,
        "last": 338.02,
        "timestamp": now - 1.0,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    # 350s old bar
    stale_bar_ts = now - 350.0
    res = validate_execution_quote(
        bridge=bridge,
        symbol="JPM",
        side="BUY",
        decision_price=338.00,
        decision_feed="sip",
        bar_timestamp=stale_bar_ts,
        max_bar_age_seconds=300.0,
        require_feed_concordance=True,
    )
    assert res.valid is False
    assert "BAR_STALE" in res.reason
    assert "limit=300.0s" in res.reason


# ── Scenario 7: Feed identity missing ────────────────────────────────────────
def test_scenario_07_feed_identity_missing():
    """Scenario 7: Feed identity missing from either quote or decision -> FEED_IDENTITY_MISSING veto."""
    now = time.time()
    # Case 7A: Quote has no feed_identity
    bridge_no_feed = MockBridge({
        "symbol": "JPM",
        "bid": 338.00,
        "ask": 338.05,
        "last": 338.02,
        "timestamp": now - 0.5,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": None,
    })
    res_a = validate_execution_quote(
        bridge=bridge_no_feed,
        symbol="JPM",
        side="BUY",
        decision_price=338.00,
        decision_feed="sip",
        bar_timestamp=now - 60.0,
        require_feed_concordance=True,
    )
    assert res_a.valid is False
    assert "FEED_IDENTITY_MISSING" in res_a.reason

    # Case 7B: Signal has no decision_feed
    bridge_with_feed = MockBridge({
        "symbol": "JPM",
        "bid": 338.00,
        "ask": 338.05,
        "last": 338.02,
        "timestamp": now - 0.5,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    res_b = validate_execution_quote(
        bridge=bridge_with_feed,
        symbol="JPM",
        side="BUY",
        decision_price=338.00,
        decision_feed=None,
        bar_timestamp=now - 60.0,
        require_feed_concordance=True,
    )
    assert res_b.valid is False
    assert "FEED_IDENTITY_MISSING" in res_b.reason


# ── Scenario 8: Quote timestamp missing (<= 0) ───────────────────────────────
def test_scenario_08_quote_timestamp_missing():
    """Scenario 8: Quote timestamp missing, null, or <= 0 -> QUOTE_TIMESTAMP_MISSING veto."""
    bridge_zero_ts = MockBridge({
        "symbol": "JPM",
        "bid": 338.00,
        "ask": 338.05,
        "last": 338.02,
        "timestamp": 0.0,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    res = validate_execution_quote(
        bridge=bridge_zero_ts,
        symbol="JPM",
        side="BUY",
        decision_price=338.00,
        decision_feed="sip",
        require_feed_concordance=True,
    )
    assert res.valid is False
    assert "QUOTE_TIMESTAMP_MISSING" in res.reason


# ── Scenario 9: Divergence > 1.5% + same feed ────────────────────────────────
def test_scenario_09_price_drift_deviation_veto():
    """Scenario 9: Divergence > 1.5% on same feed -> PRICE_DRIFT_DEVIATION_VETO."""
    now = time.time()
    # Decision: 100.00, Quote mid: 102.00 (2.0% divergence > 1.5% max)
    bridge = MockBridge({
        "symbol": "AAPL",
        "bid": 101.95,
        "ask": 102.05,
        "last": 102.00,
        "timestamp": now - 0.5,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="AAPL",
        side="BUY",
        decision_price=100.00,
        decision_feed="sip",
        bar_timestamp=now - 60.0,
        max_deviation_pct=1.5,
        require_feed_concordance=True,
    )
    assert res.valid is False
    assert "PRICE_DRIFT_DEVIATION_VETO" in res.reason
    assert "PROVENANCE_FEED_DISCREPANCY_VETO" not in res.reason


# ── Scenario 10: Divergence > 1.5% + different feed ──────────────────────────
def test_scenario_10_provenance_feed_discrepancy_veto():
    """Scenario 10: Divergence > 1.5% across discordant feeds -> PROVENANCE_FEED_DISCREPANCY_VETO (JPM Anomaly pattern)."""
    now = time.time()
    # Decision: 337.93 (SIP historical bar), Quote: 346.06 (IEX quote), 2.41% dislocation
    bridge = MockBridge({
        "symbol": "JPM",
        "bid": 346.00,
        "ask": 346.12,
        "last": 346.06,
        "timestamp": now - 0.5,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "iex",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="JPM",
        side="BUY",
        decision_price=337.93,
        decision_feed="sip",
        bar_timestamp=now - 60.0,
        max_deviation_pct=1.5,
        require_feed_concordance=True,
    )
    assert res.valid is False
    assert "PROVENANCE_FEED_DISCREPANCY_VETO" in res.reason
    assert "quote mid $346.06 (iex) vs decision $337.93 (sip)" in res.reason


# ── Scenario 11: All valid -> Execution eligible ─────────────────────────────
def test_scenario_11_all_valid_execution_eligible():
    """Scenario 11: All 6 upstream links valid & concordant -> Execution eligible."""
    now = time.time()
    bridge = MockBridge({
        "symbol": "MSFT",
        "bid": 420.10,
        "ask": 420.20,
        "last": 420.15,
        "timestamp": now - 0.3,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="MSFT",
        side="BUY",
        decision_price=420.12,
        decision_feed="sip",
        bar_timestamp=now - 120.0,
        max_spread_pct=1.0,
        max_deviation_pct=1.5,
        max_age_seconds=10.0,
        max_bar_age_seconds=300.0,
        require_feed_concordance=True,
    )
    assert res.valid is True
    assert res.accepted is True
    assert res.reason == "Quote valid"
    assert res.quote is not None
    assert res.quote.executable_price("BUY") == 420.20
    assert res.quote.executable_price("SELL") == 420.10


# ── Additional Invariant Tests ────────────────────────────────────────────────
def test_negative_latency_guard():
    """Verify quote older than decision bar by > 2.0s triggers NEGATIVE_LATENCY_ERROR."""
    now = time.time()
    bar_ts = now - 10.0
    quote_ts = bar_ts - 5.0  # Quote timestamp 5s before bar!
    bridge = MockBridge({
        "symbol": "JPM",
        "bid": 338.00,
        "ask": 338.05,
        "last": 338.02,
        "timestamp": quote_ts,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "sip",
    })
    res = validate_execution_quote(
        bridge=bridge,
        symbol="JPM",
        side="BUY",
        decision_price=338.00,
        decision_feed="sip",
        bar_timestamp=bar_ts,
        max_age_seconds=60.0,
        require_feed_concordance=True,
    )
    assert res.valid is False
    assert "NEGATIVE_LATENCY_ERROR" in res.reason


def test_trade_signal_decision_feed_and_hash():
    """Verify TradeSignal accepts decision_feed and produces deterministic 64-char SHA256."""
    sig1 = TradeSignal(
        symbol="JPM",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=338.00,
        stop_price=336.00,
        target_price=342.00,
        atr=2.00,
        risk_per_share=2.00,
        reason="Test signal",
        bar_timestamp="2026-09-24T15:30:00Z",
        decision_feed="sip",
    )
    assert sig1.decision_feed == "sip"
    assert len(sig1.signal_hash) == 64

    # Backwards compatibility: signal_hash remains identical for same price geometry
    sig2 = TradeSignal(
        symbol="JPM",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=338.00,
        stop_price=336.00,
        target_price=342.00,
        atr=2.00,
        risk_per_share=2.00,
        reason="Another reason",
        bar_timestamp="2026-09-24T15:30:00Z",
        decision_feed="iex",
    )
    assert sig1.signal_hash == sig2.signal_hash


def test_alpaca_bridge_get_quote_feed_contract():
    """Verify alpaca_bridge.get_quote requests specified feed and returns feed_identity."""
    from alpaca_bridge import AlpacaBridge
    bridge = AlpacaBridge()
    mock_data_client = MagicMock()
    mock_quote = MagicMock()
    mock_quote.bid_price = 150.00
    mock_quote.ask_price = 150.05
    mock_quote.timestamp = datetime.now(timezone.utc)
    mock_data_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}
    bridge._data_client = mock_data_client

    # Test explicit SIP feed request
    res = bridge.get_quote("AAPL", feed="sip")
    assert res is not None
    assert res["feed_identity"] == "sip"
    assert res["is_realtime"] is True
    assert res["bid"] == 150.00
    assert res["ask"] == 150.05

    # Verify StockLatestQuoteRequest received DataFeed.SIP
    call_args = mock_data_client.get_stock_latest_quote.call_args[0][0]
    from alpaca.data.enums import DataFeed
    assert call_args.feed == DataFeed.SIP


def test_intraday_data_cache_preserves_attrs():
    """Verify _write_cache and _read_cache preserve df.attrs metadata across disk storage."""
    from strategies.intraday_data import _write_cache, _read_cache
    test_key = "test_feed_prov_cache_roundtrip"
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.date_range("2026-09-24 10:00", periods=3, freq="5min"),
    )
    df.attrs["feed_identity"] = "sip"
    df.attrs["requested_feed"] = "sip"
    df.attrs["subscription_tier"] = "algo"
    df.attrs["bar_end_utc"] = "2026-09-24T10:10:00"

    _write_cache(test_key, df)
    cached_df = _read_cache(test_key, ttl=60)
    assert cached_df is not None
    assert cached_df.attrs.get("feed_identity") == "sip"
    assert cached_df.attrs.get("requested_feed") == "sip"
    assert cached_df.attrs.get("subscription_tier") == "algo"
    assert cached_df.attrs.get("bar_end_utc") == "2026-09-24T10:10:00"


def test_order_manager_fail_closed_on_feed_dislocation():
    """Verify order_manager.execute_signal vetos execution when feed concordance fails."""
    from strategies.order_manager import execute_signal
    from strategies.base import EntryReadiness

    now = time.time()
    bridge = MockBridge({
        "symbol": "BAC",
        "bid": 56.25,
        "ask": 56.27,
        "last": 56.26,
        "timestamp": now - 0.2,
        "source": "alpaca",
        "is_realtime": True,
        "feed_identity": "iex",
    })
    # Signal generated from SIP feed
    signal = TradeSignal(
        symbol="BAC",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=56.26,
        stop_price=56.10,
        target_price=56.80,
        atr=0.20,
        risk_per_share=0.16,
        reason="Momentum test",
        bar_timestamp=datetime.fromtimestamp(now - 60.0, tz=timezone.utc).isoformat(),
        decision_feed="sip",
    )
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="cycle_001",
        bar_ts=datetime.fromtimestamp(now - 60.0, tz=timezone.utc).isoformat(),
        created_at=now,
        expires_at=now + 300.0,
    )

    with patch("config.REQUIRE_FEED_CONCORDANCE", True), \
         patch("config.SPY_TREND_GUARD_ENABLED", False), \
         patch("strategies.intraday_risk.get_recent_symbol_loss_time", return_value=None):
        res = execute_signal(
            signal=signal,
            bridge=bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=False,
            entry_readiness=readiness,
        )
        assert res.get("status") == "VETOED", f"Expected VETOED, got status={res.get('status')}, reason={res.get('reason')}"
        assert "CROSS_FEED_PROVENANCE_DISLOCATION" in res["reason"]
