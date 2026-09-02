"""
tests/test_p0_critical.py -- Verification unit tests for Phase 1 (P0 Critical & High Fixes).
"""

import pytest
import pandas as pd
import numpy as np

from strategies.base import TradeSignal
from strategies.constants import StrategyName
from strategies.mean_reversion import MeanReversionStrategy
from strategies.regime_engine import (
    detect_market_texture,
    evaluate_market_regime,
    TEXTURE_TRENDING,
    TEXTURE_SIDEWAYS,
    TEXTURE_VOLATILE,
    TEXTURE_TRANSITIONING,
)
from strategies.intraday_risk import check_soxx_tech_long_gate, pre_trade_check
from strategies.order_manager import _pending_order_contexts, register_pending_order, on_broker_fill_event
from strategies.trailing_stop import manager as trailing_manager


@pytest.fixture(autouse=True)
def isolate_journal(tmp_path, monkeypatch):
    import config
    fake_journal = tmp_path / "test_trade_journal.jsonl"
    monkeypatch.setattr(config, "DAYTRADE_TRADE_JOURNAL_FILE", fake_journal)
    # Clear any active idempotency locks
    from strategies.order_idempotency import idempotency_manager
    idempotency_manager.reservations.clear()


def test_mean_reversion_builds_valid_trade_signal():
    """C-01: MeanReversionStrategy must create a valid TradeSignal without TypeError."""
    np.random.seed(42)
    n = 30
    dates = pd.date_range("2026-08-13 09:30", periods=n, freq="5min")
    prices = np.full(n, 100.0)
    prices[-1] = 95.0
    highs = prices + 0.5
    lows = prices - 0.5
    highs[-1] = 95.5
    lows[-1] = 94.5
    volumes = np.full(n, 10000.0)
    volumes[-1] = 12000.0
    
    df = pd.DataFrame({
        "open": prices,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
        "vwap": np.full(n, 100.0)
    }, index=dates)

    strat = MeanReversionStrategy()
    sig = strat.generate_signal("PLTR", df, market_texture="SIDEWAYS")
    assert sig is not None
    assert sig.is_valid is True
    assert sig.strategy == StrategyName.MEAN_REVERSION
    assert sig.atr > 0
    assert sig.risk_per_share > 0


def test_order_ack_does_not_create_trailing_state():
    """C-02: Pending order registration alone does NOT create trailing stop position state until fill event."""
    parent_id = "test_parent_order_123"
    sig = TradeSignal(
        symbol="NVDA",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.70,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        atr=0.5,
        risk_per_share=1.0,
        reason="Test order ACK",
    )


    # Clean up trailing and idempotency state if present
    trailing_manager.reset("NVDA")
    trailing_manager.states.pop("NVDA", None)
    assert "NVDA" not in trailing_manager.states
    from strategies.order_idempotency import idempotency_manager
    idempotency_manager.reservations.pop("NVDA_BUY", None)

    # 1. Register pending order (Placement ACK)
    register_pending_order(parent_id=parent_id, signal=sig, limit_price=100.0, qty=10)

    assert parent_id in _pending_order_contexts
    # Trailing state MUST remain empty until broker fill event!
    assert "NVDA" not in trailing_manager.states

    # 2. Emulate broker execution fill event
    on_broker_fill_event(parent_id=parent_id, fill_price=100.05, filled_qty=10)
    assert parent_id not in _pending_order_contexts
    assert "NVDA" in trailing_manager.states
    assert trailing_manager.states["NVDA"].entry_price == 100.05


def test_large_negative_spy_is_volatile():
    """H-04: SPY = -2.0% fixture produces VOLATILE texture, not TRENDING."""
    texture = detect_market_texture(adx=30.0, bb_width=0.03, atr_ratio=1.0, price_change_5m=-0.025)
    assert texture == TEXTURE_VOLATILE


def test_soxx_percent_unit_is_consistent():
    """H-01: -0.08% passes -0.75% threshold; -0.80% vetoes it."""
    # Mock _get_soxx_realtime_pct_change to test unit conversion
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


def test_soxx_gate_receives_actual_side():
    """H-02: Short candidates must not be vetoed by the long-only SOXX gate."""
    import strategies.intraday_risk as ir
    orig_fn = ir._get_soxx_realtime_pct_change
    try:
        ir._get_soxx_realtime_pct_change = lambda: -0.80  # Bearish SOXX -0.80%
        ok_short, reason_short = check_soxx_tech_long_gate("NVDA", side="SELL")
        assert ok_short is True
        assert "short trade" in reason_short
    finally:
        ir._get_soxx_realtime_pct_change = orig_fn


def test_entry_lifecycle_confirmed_fill():
    """Verify EntryOrderLifecycle pending registration, protection checking, and confirmed fill handling."""
    from strategies.entry_lifecycle import EntryOrderLifecycle
    from strategies.trailing_stop import manager as tm

    lifecycle = EntryOrderLifecycle(tm)
    sig = TradeSignal(
        symbol="AMD",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.70,
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        atr=0.5,
        risk_per_share=1.0,
        reason="Test lifecycle order",
    )

    tm.reset("AMD")
    tm.states.pop("AMD", None)
    assert "AMD" not in tm.states

    # Register submission
    entry = lifecycle.register_submission(
        parent_order_id="p_999",
        signal=sig,
        requested_qty=10.0,
        stop_order_id="stop_999",
        target_order_id="target_999",
    )
    assert entry.parent_order_id == "p_999"
    assert "AMD" not in tm.states

    # Mock bridge with get_order_snapshot
    class MockBridge:
        def get_order_snapshot(self, order_id):
            return {"status": "accepted", "qty": 10.0}

    mock_b = MockBridge()
    ok = lifecycle.on_parent_fill(
        bridge=mock_b,
        parent_order_id="p_999",
        cumulative_filled_qty=10.0,
        average_fill_price=100.05,
    )
    assert ok is True
    assert "AMD" in tm.states
    assert tm.states["AMD"].entry_price == 100.05


def test_config_p0_defaults_are_secure():
    """Verify P0 configuration settings match fail-safe requirements."""
    import config
    assert config.REQUIRE_REALTIME_DATA_FOR_ORDERS is True
    assert getattr(config, "PAPER_MODE_AUTO_CONFIRM_FILL", False) is False


def test_execute_signal_p0_auto_fill_disabled_and_idempotency_released():
    """Verify execute_signal does not auto-confirm fill and releases idempotency on dry_run."""
    import config
    from strategies.order_manager import execute_signal
    from strategies.order_idempotency import idempotency_manager

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self.ib = type("MockIB", (), {"positions": lambda: []})()
        def open_position_count(self): return 0
        def get_positions(self): return []
        def has_position(self, sym): return False
        def has_working_order(self, sym): return False

    sig = TradeSignal(
        symbol="MSFT",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.80,
        entry_price=400.0,
        stop_price=398.0,
        target_price=404.0,
        atr=1.0,
        risk_per_share=2.0,
        reason="Test dry run idempotency release",
    )

    b = MockBridge()
    res = execute_signal(signal=sig, bridge=b, equity=10000.0, current_pnl=0.0, dry_run=True)
    assert res.get("status") == "DRY_RUN"
    # Verify idempotency reservation for MSFT_BUY was released
    key = "MSFT_BUY"
    assert key not in idempotency_manager.reservations or idempotency_manager.reservations[key].status == "TERMINAL"


def test_execute_signal_quote_veto_releases_idempotency():
    """Verify execute_signal quote gate veto releases idempotency reservation."""
    import config
    from strategies.order_manager import execute_signal
    from strategies.order_idempotency import idempotency_manager

    class MockBridge:
        def __init__(self):
            self.is_connected = True
            self.ib = type("MockIB", (), {"positions": lambda: []})()
        def open_position_count(self): return 0
        def get_positions(self): return []
        def has_position(self, sym): return False
        def has_working_order(self, sym): return False

    sig = TradeSignal(
        symbol="TESTSYM",
        strategy=StrategyName.MEAN_REVERSION,
        side="BUY",
        confidence=0.85,
        entry_price=220.0,
        stop_price=218.0,
        target_price=224.0,
        atr=1.0,
        risk_per_share=2.0,
        reason="Test quote veto release",
    )

    # Force real-time quote validation failure by providing stale quote
    b = MockBridge()
    import strategies.execution_quote as eq
    orig_validate = eq.validate_execution_quote
    try:
        eq.validate_execution_quote = lambda **kwargs: eq.ExecutionQuoteResult(valid=False, reason="Mock Stale Quote")
        from unittest.mock import patch
        with patch.object(config, "SPY_TREND_GUARD_ENABLED", False):
            res = execute_signal(signal=sig, bridge=b, equity=100000.0, current_pnl=0.0, dry_run=False)
        assert res.get("status") == "VETOED"
        # Idempotency reservation should be released
        assert "TESTSYM_BUY" not in idempotency_manager.reservations
    finally:
        eq.validate_execution_quote = orig_validate


def test_h1_quote_gate_threshold_config_binding():
    """H-1: Verify execution quote gate thresholds are bound to declared config attributes."""
    import config
    assert hasattr(config, "MAX_ACCEPTABLE_SPREAD_PCT")
    assert hasattr(config, "MAX_DECISION_DEVIATION_PCT")
    assert hasattr(config, "MAX_QUOTE_AGE_SECONDS")
    assert config.MAX_ACCEPTABLE_SPREAD_PCT in (0.20, 1.0)
    assert config.MAX_DECISION_DEVIATION_PCT == 1.5
    assert config.MAX_QUOTE_AGE_SECONDS == 10.0


def test_h2_ambiguous_submit_preserves_idempotency():
    """H-2: Missing parent ID after submission must return AMBIGUOUS_SUBMIT and keep idempotency lock."""
    import config
    from strategies.order_manager import execute_signal
    from strategies.order_idempotency import idempotency_manager

    class MockBridgeNoParentID:
        def __init__(self):
            self.is_connected = True
            self.ib = type("MockIB", (), {"positions": lambda: []})()
        def open_position_count(self): return 0
        def get_positions(self): return []
        def has_position(self, sym): return False
        def has_working_order(self, sym): return False
        def place_bracket_order(self, symbol, side, qty, entry_price, stop_price, target_price):
            return {"status": "submitted", "order_ids": []}  # Missing parent ID

    sig = TradeSignal(
        symbol="AMZN",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.80,
        entry_price=180.0,
        stop_price=178.0,
        target_price=184.0,
        atr=1.0,
        risk_per_share=2.0,
        reason="Test ambiguous submit",
    )

    b = MockBridgeNoParentID()
    import strategies.execution_quote as eq
    orig_validate = eq.validate_execution_quote
    try:
        eq.validate_execution_quote = lambda **kwargs: eq.ExecutionQuoteResult(valid=True, reason="Mock Valid Quote")
        res = execute_signal(signal=sig, bridge=b, equity=10000.0, current_pnl=0.0, dry_run=False)
        assert res.get("status") == "AMBIGUOUS_SUBMIT"
        # Idempotency reservation MUST remain locked for reconciliation (not released)
        assert "AMZN_BUY" in idempotency_manager.reservations
        assert idempotency_manager.reservations["AMZN_BUY"].status == "PENDING"
    finally:
        eq.validate_execution_quote = orig_validate
        idempotency_manager.release("AMZN", "BUY", "cleanup_test")


def test_h3_lifecycle_registration_failure_is_fail_closed():
    """H-3: Entry lifecycle registration failure must return AMBIGUOUS_LIFECYCLE and not PLACED."""
    import config
    from strategies.order_manager import execute_signal, register_pending_order
    from strategies.order_idempotency import idempotency_manager

    class MockBridgeOK:
        def __init__(self):
            self.is_connected = True
            self.ib = type("MockIB", (), {"positions": lambda: []})()
        def open_position_count(self): return 0
        def get_positions(self): return []
        def has_position(self, sym): return False
        def has_working_order(self, sym): return False
        def place_bracket_order(self, symbol, side, qty, entry_price, stop_price, target_price):
            return {"status": "submitted", "order_ids": ["p_101", "tp_101", "stop_101"]}

    sig = TradeSignal(
        symbol="GOOGL",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.80,
        entry_price=170.0,
        stop_price=168.0,
        target_price=174.0,
        atr=1.0,
        risk_per_share=2.0,
        reason="Test lifecycle failure",
    )

    b = MockBridgeOK()
    import strategies.execution_quote as eq
    import strategies.order_manager as om
    orig_validate = eq.validate_execution_quote
    orig_reg = om.register_pending_order
    try:
        eq.validate_execution_quote = lambda **kwargs: eq.ExecutionQuoteResult(valid=True, reason="Mock Valid Quote")
        def raise_reg_err(*args, **kwargs):
            raise RuntimeError("Mock lifecycle store error")
        om.register_pending_order = raise_reg_err

        res = execute_signal(signal=sig, bridge=b, equity=10000.0, current_pnl=0.0, dry_run=False)
        assert res.get("status") == "AMBIGUOUS_LIFECYCLE"
        assert "Lifecycle registration failed" in res.get("reason", "")
    finally:
        eq.validate_execution_quote = orig_validate
        om.register_pending_order = orig_reg
        idempotency_manager.release("GOOGL", "BUY", "cleanup_test")


def test_entry_gate_halt_blocks_order_placement_defense_in_depth():
    """Verify entry gate halt blocks orders across bridge, order_manager, and main loop."""
    from strategies.order_manager import execute_signal
    from alpaca_bridge import AlpacaBridge

    class MockEntryGate:
        def __init__(self):
            self.halted = True
        def halt(self):
            self.halted = True

    class MockHaltedBridge:
        def __init__(self):
            self.is_connected = True
            self.entry_gate = MockEntryGate()
            self._connected = True
            self._trade_stream_ready = True
        def open_position_count(self): return 0
        def get_positions(self): return []
        def has_position(self, sym): return False
        def has_working_order(self, sym): return False
        def place_bracket_order(self, symbol, side, qty, entry_price, stop_price, target_price):
            if self.entry_gate.halted:
                raise RuntimeError("New entries blocked: startup reconciliation entry gate is halted")
            return {"status": "submitted"}

    sig = TradeSignal(
        symbol="META",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.85,
        entry_price=500.0,
        stop_price=495.0,
        target_price=510.0,
        atr=2.0,
        risk_per_share=5.0,
        reason="Test halted entry gate",
    )

    b = MockHaltedBridge()
    res = execute_signal(signal=sig, bridge=b, equity=10000.0, current_pnl=0.0, dry_run=False)
    assert res.get("status") == "REJECTED"
    assert "Entry gate is halted" in res.get("reason", "")


def test_execute_signal_cache_key_defined():
    """Verify execute_signal defines cache_key and does not raise NameError."""
    from strategies.order_manager import execute_signal

    class MockPlacementBridge:
        def __init__(self):
            self.is_connected = True
            self.ib = type("MockIB", (), {"positions": lambda: []})()
        def open_position_count(self): return 0
        def get_positions(self): return []
        def has_position(self, sym): return False
        def has_working_order(self, sym): return False
        def place_bracket_order(self, symbol, side, qty, entry_price, stop_price, target_price):
            return {"status": "submitted", "order_ids": ["p_555", "tp_555", "stop_555"]}

    sig = TradeSignal(
        symbol="SMCI",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.85,
        entry_price=39.50,
        stop_price=39.00,
        target_price=40.50,
        atr=0.50,
        risk_per_share=0.50,
        reason="Test cache_key defined",
    )

    b = MockPlacementBridge()
    import config
    import strategies.execution_quote as eq
    orig_validate = eq.validate_execution_quote
    orig_allow_short = getattr(config, "ALLOW_SHORT", False)
    orig_spy_guard = getattr(config, "SPY_TREND_GUARD_ENABLED", True)
    try:
        config.ALLOW_SHORT = True
        config.SPY_TREND_GUARD_ENABLED = False
        eq.validate_execution_quote = lambda **kwargs: eq.ExecutionQuoteResult(valid=True, reason="Mock Valid Quote")
        res = execute_signal(signal=sig, bridge=b, equity=10000.0, current_pnl=0.0, dry_run=False)
        print("DEBUG RESULT:", res)
        assert res.get("status") == "PLACED", f"Expected PLACED, got {res.get('status')}: {res.get('reason')}"
        assert res.get("reason") != "Order failed: name 'cache_key' is not defined"
    finally:
        config.ALLOW_SHORT = orig_allow_short
        config.SPY_TREND_GUARD_ENABLED = orig_spy_guard
        eq.validate_execution_quote = orig_validate
        from strategies.order_idempotency import idempotency_manager
        idempotency_manager.release("SMCI", "BUY", "cleanup_test")


def test_build_client_order_id_determinism_and_length():
    """Verify build_client_order_id is deterministic, handles attempt counter, and adheres to Alpaca length limit."""
    from strategies.order_idempotency import build_client_order_id

    id1 = build_client_order_id("NVDA", "BUY", "2026-08-14T13:30:00Z", "hash123456", attempt=0)
    id2 = build_client_order_id("NVDA", "BUY", "2026-08-14T13:30:00Z", "hash123456", attempt=0)
    assert id1 == id2
    assert id1.startswith("se3-NVDA-")
    assert len(id1) <= 128

    # Attempt disambiguation for scale-in / retry
    id_att1 = build_client_order_id("NVDA", "BUY", "2026-08-14T13:30:00Z", "hash123456", attempt=1)
    assert id1 != id_att1

    # Bar timestamp disambiguation
    id_bar2 = build_client_order_id("NVDA", "BUY", "2026-08-14T13:35:00Z", "hash123456", attempt=0)
    assert id1 != id_bar2


def test_trade_signal_deterministic_hash_float_formatting():
    """Verify TradeSignal.signal_hash is deterministic across float representations."""
    sig1 = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.MEAN_REVERSION,
        side="BUY",
        confidence=0.85,
        entry_price=220.000000001,
        stop_price=218.0,
        target_price=224.0,
        atr=1.0,
        risk_per_share=2.0,
        reason="Test hash",
        bar_timestamp="2026-08-14T13:30:00Z",
    )
    sig2 = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.MEAN_REVERSION,
        side="BUY",
        confidence=0.85,
        entry_price=220.0,
        stop_price=218.000000002,
        target_price=224.0,
        atr=1.0,
        risk_per_share=2.0,
        reason="Test hash",
        bar_timestamp="2026-08-14T13:30:00Z",
    )
    assert sig1.signal_hash == sig2.signal_hash
    assert len(sig1.signal_hash) == 64


def test_is_regular_hours_timezone_aware_enforced():
    """Verify data_integrity.is_regular_hours rejects naive datetimes fail-closed."""
    import datetime as dt
    from zoneinfo import ZoneInfo
    import data_integrity as di
    import pytest

    naive_dt = dt.datetime(2026, 6, 16, 10, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        di.is_regular_hours(naive_dt)

    with pytest.raises(ValueError, match="timezone-aware"):
        di.regular_hours_reason(naive_dt)

    aware_dt = dt.datetime(2026, 6, 16, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert di.is_regular_hours(aware_dt) is True
    assert di.regular_hours_reason(aware_dt) == "open"


def test_settings_mirror_is_complete():
    """Verify Settings dataclass dynamically mirrors 100% of module-level config constants."""
    import re
    import config

    with open(config.__file__, encoding="utf-8") as f:
        src = f.read()
    mod_constants = {m.group(1) for m in re.finditer(r'^([A-Z_][A-Z0-9_]*)\s*=', src, re.M)}
    s = config.get_settings()
    missing = sorted(k for k in mod_constants if not hasattr(s, k))
    assert not missing, f"Settings mirror missing constants: {missing}"








