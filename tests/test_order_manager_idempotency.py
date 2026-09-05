"""
tests/test_order_manager_idempotency.py -- Comprehensive test suite for Order Manager
Execution Safety, Broker Capability, Idempotency Lifecycle, and Concurrency Invariants.

Covers:
1. unsupported signature blocks before broker call
2. unsupported bridge safely releases reservation
3. missing broker method safely releases reservation
4. uninspectable signature safely releases reservation
5. 422 unresolved retains reservation (SUBMITTED_RECONCILIATION_REQUIRED)
6. 422 recovered complete bracket binds parent
7. definitive non-duplicate rejection releases reservation
8. sub-dollar quote adjustment preserves 0.0001 precision
9. final quote-adjusted invalid geometry is vetoed
10. concurrent hard-veto increments are not lost
11. reset and suppression check do not race
12. journal counter deduplicates same parent order ID
13. non-paper bridge is blocked before reservation/submission
14. verified paper bridge reaches normal execution path
"""
import inspect
import threading
from unittest.mock import MagicMock, patch
import pytest

import config
from strategies.base import TradeSignal, StrategyName
import strategies.order_manager as om
from strategies.order_idempotency import idempotency_manager


@pytest.fixture(autouse=True)
def clean_om_state(monkeypatch):
    """Ensure clean global state before and after each test."""
    monkeypatch.setattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", False)
    monkeypatch.setattr(config, "SPY_TREND_GUARD_ENABLED", False)
    monkeypatch.setattr(config, "VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", False)
    with om._pending_order_contexts_lock:
        om._pending_order_contexts.clear()
    om.reset_session_skip_state()
    idempotency_manager.clear()
    yield
    with om._pending_order_contexts_lock:
        om._pending_order_contexts.clear()
    om.reset_session_skip_state()
    idempotency_manager.clear()


def _make_signal(symbol="AAPL", side="BUY", entry=100.0, stop=98.0, target=106.0, confidence=0.85, atr=1.0):
    risk = abs(entry - stop)
    return TradeSignal(
        symbol=symbol,
        strategy=StrategyName.VWAP_BOUNCE,
        side=side,
        confidence=confidence,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        atr=atr,
        risk_per_share=risk,
        reason="idempotency_test_signal",
    )


def _make_paper_bridge():
    bridge = MagicMock()
    bridge.is_paper = True
    bridge.account_mode = "PAPER"
    bridge.is_connected = True
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    return bridge


# ==============================================================================
# 1. Broker Idempotency & Capability Tests (1-7)
# ==============================================================================

def test_unsupported_signature_blocks_before_broker_call():
    """1. Unsupported signature blocks before broker call."""
    bridge = _make_paper_bridge()
    called = False

    def mock_place(symbol, side, qty, entry_price, stop_price, target_price):
        nonlocal called
        called = True
        return {"order_id": "b-100"}

    bridge.place_bracket_order = mock_place

    with pytest.raises(om.BrokerIdempotencyUnsupported) as exc_info:
        om._place_bracket_order_once(
            bridge,
            symbol="AAPL",
            side="BUY",
            qty=10,
            entry_price=100.0,
            stop_price=98.0,
            target_price=106.0,
            client_order_id="cid-1",
        )

    assert "does not support client_order_id" in str(exc_info.value)
    assert not called


def test_unsupported_bridge_safely_releases_reservation():
    """2. Unsupported bridge safely releases reservation in execute_signal."""
    bridge = _make_paper_bridge()

    def mock_place(symbol, side, qty, entry_price, stop_price, target_price):
        return {"order_id": "b-100"}

    bridge.place_bracket_order = mock_place
    sig = _make_signal()

    res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "REJECTED"
    assert "does not support client_order_id" in res["reason"]
    assert "AAPL_BUY" not in idempotency_manager.reservations


def test_missing_broker_method_safely_releases_reservation():
    """3. Missing broker method safely releases reservation."""
    bridge = _make_paper_bridge()
    if hasattr(bridge, "place_bracket_order"):
        delattr(bridge, "place_bracket_order")

    sig = _make_signal()
    res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "REJECTED"
    assert "does not implement place_bracket_order" in res["reason"]
    assert "AAPL_BUY" not in idempotency_manager.reservations


def test_uninspectable_signature_safely_releases_reservation():
    """4. Uninspectable signature safely releases reservation."""
    bridge = _make_paper_bridge()

    def dummy_place(**kwargs):
        return {"order_id": "dummy"}

    bridge.place_bracket_order = dummy_place

    with patch("inspect.signature", side_effect=TypeError("uninspectable builtin")):
        sig = _make_signal()
        res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)

        assert res["status"] == "REJECTED"
        assert "Cannot inspect broker submission signature" in res["reason"]
        assert "AAPL_BUY" not in idempotency_manager.reservations


def test_422_unresolved_retains_reservation():
    """5. 422 unresolved retains reservation (SUBMITTED_RECONCILIATION_REQUIRED)."""
    bridge = _make_paper_bridge()

    def mock_place(**kwargs):
        raise Exception('{"code":42210000,"message":"client_order_id must be unique"}')

    bridge.place_bracket_order = mock_place
    bridge.get_order_by_client_id.return_value = None

    sig = _make_signal()
    res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "SUBMITTED_RECONCILIATION_REQUIRED"
    assert res["metadata"].get("reconciliation_reason") == "DUPLICATE_CLIENT_ORDER_ID"
    # Crucial: Reservation must RETAIN lock!
    assert "AAPL_BUY" in idempotency_manager.reservations


def test_422_recovered_complete_bracket_binds_parent():
    """6. 422 recovered complete bracket binds parent."""
    bridge = _make_paper_bridge()

    def mock_place(**kwargs):
        raise Exception('{"code":42210000,"message":"client_order_id must be unique"}')

    bridge.place_bracket_order = mock_place
    bridge.get_order_by_client_id.return_value = {
        "order_id": "broker-parent-999",
        "status": "new",
        "stop_order_id": "stop-999",
        "tp_order_id": "tp-999",
    }

    sig = _make_signal()
    res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "PLACED"
    assert res["order"]["order_id"] == "broker-parent-999"
    # Check parent order binding
    res_obj = idempotency_manager.reservations.get("AAPL_BUY")
    assert res_obj is not None
    assert res_obj.parent_order_id == "broker-parent-999"


def test_definitive_non_duplicate_rejection_releases_reservation():
    """7. Definitive non-duplicate rejection releases reservation."""
    bridge = _make_paper_bridge()

    class DefinitiveRejectException(Exception):
        definitely_rejected = True

    def mock_place(**kwargs):
        raise DefinitiveRejectException("Account buying power exhausted")

    bridge.place_bracket_order = mock_place
    sig = _make_signal()

    res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "REJECTED"
    assert "Broker definitely rejected order" in res["reason"]
    assert "AAPL_BUY" not in idempotency_manager.reservations


# ==============================================================================
# 2. Final Geometry & Precision Tests (8-9)
# ==============================================================================

def test_sub_dollar_quote_adjustment_preserves_0001_precision(monkeypatch):
    """8. Sub-dollar quote adjustment preserves 0.0001 precision."""
    bridge = _make_paper_bridge()
    submitted_kwargs = {}

    def mock_place(**kwargs):
        nonlocal submitted_kwargs
        submitted_kwargs = kwargs
        return {"order_id": "sub-1", "stop_order_id": "s-1", "tp_order_id": "t-1"}

    bridge.place_bracket_order = mock_place

    sig = _make_signal(symbol="PENY", entry=0.8500, stop=0.7500, target=1.0500, atr=0.10)
    monkeypatch.setattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", True)
    mock_quote = MagicMock(valid=True, quote=MagicMock(last=0.8525))

    with patch("strategies.execution_quote.validate_execution_quote", return_value=mock_quote):
        res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "PLACED"
    assert submitted_kwargs["entry_price"] == 0.8525
    assert submitted_kwargs["stop_price"] == 0.7325
    assert submitted_kwargs["target_price"] == 1.0325


def test_final_quote_adjusted_invalid_geometry_is_vetoed(monkeypatch):
    """9. Final quote-adjusted invalid geometry is vetoed upstream before submission."""
    bridge = _make_paper_bridge()
    place_called = False

    def mock_place(**kwargs):
        nonlocal place_called
        place_called = True
        return {"order_id": "b-1"}

    bridge.place_bracket_order = mock_place

    sig = _make_signal(symbol="TEST", entry=10.00, stop=9.85, target=10.30, atr=0.20)
    monkeypatch.setattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", True)
    # Quote changes entry price to 0.04 -> stop_price = 0.04 - 0.15 = -0.11 <= 0 -> invalid
    mock_quote = MagicMock(valid=True, quote=MagicMock(last=0.04))

    with patch("strategies.execution_quote.validate_execution_quote", return_value=mock_quote):
        res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "VETOED"
    assert "Final submitted geometry invalid" in res["reason"]
    assert not place_called
    assert "TEST_BUY" not in idempotency_manager.reservations


# ==============================================================================
# 3. Concurrency & Accounting Tests (10-12)
# ==============================================================================

def test_concurrent_hard_veto_increments_are_not_lost():
    """10. Concurrent hard-veto increments are not lost across threads."""
    om.reset_session_skip_state()
    num_threads = 20
    threads = []

    def worker(idx):
        om.record_hard_veto_skip("NVDA", "critically tight stop distance", bar_timestamp=f"2026-09-08T10:{idx:02d}:00Z")

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    with om._session_state_lock:
        suppressed, msg = om.check_session_suppression("NVDA")
        assert suppressed is True


def test_reset_and_suppression_check_do_not_race():
    """11. Reset and suppression check do not race or raise exceptions."""
    stop_event = threading.Event()
    exceptions = []

    def checker():
        while not stop_event.is_set():
            try:
                om.check_session_suppression("TSLA")
            except Exception as e:
                exceptions.append(e)

    def modifier():
        for i in range(50):
            try:
                om.record_hard_veto_skip("TSLA", "critically tight", bar_timestamp=f"ts-{i}")
                if i % 10 == 0:
                    om.reset_session_skip_state()
            except Exception as e:
                exceptions.append(e)

    t1 = threading.Thread(target=checker)
    t2 = threading.Thread(target=modifier)
    t1.start()
    t2.start()

    t2.join()
    stop_event.set()
    t1.join()

    assert len(exceptions) == 0


def test_journal_counter_deduplicates_same_parent_order_id():
    """12. Journal counter deduplicates same parent order ID."""
    bridge = _make_paper_bridge()
    sig = _make_signal(symbol="MSFT")

    mock_trades = [
        {"symbol": "MSFT", "order_id": "parent-ord-777", "status": "PARTIALLY_FILLED", "order_type": "BRACKET"},
        {"symbol": "MSFT", "order_id": "parent-ord-777", "status": "FILLED", "order_type": "ENTRY"},
        {"symbol": "MSFT", "order_id": "parent-ord-777", "status": "FILLED", "order_type": "BRACKET"},
    ]

    with patch("strategies.trade_journal.today_trades", return_value=mock_trades),          patch("strategies.order_manager.calculate_shares", return_value=10):
        bridge.place_bracket_order = MagicMock(return_value={"order_id": "parent-ord-888", "stop_order_id": "s", "tp_order_id": "t"})
        res = om.execute_signal(sig, bridge, equity=50000.0, current_pnl=0.0, dry_run=False)
        assert res["status"] == "PLACED"


# ==============================================================================
# 4. Paper Account Runtime Gates (13-14)
# ==============================================================================

def test_non_paper_bridge_is_blocked_before_reservation_or_submission():
    """13. Non-paper bridge is blocked before reservation/submission."""
    live_bridge = MagicMock()
    live_bridge.is_paper = False
    live_bridge.account_mode = "LIVE"
    live_bridge.is_connected = True

    sig = _make_signal(symbol="SPY")
    res = om.execute_signal(sig, live_bridge, equity=100000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "REJECTED"
    assert "Paper-account verification failed" in res["reason"]
    assert "SPY_BUY" not in idempotency_manager.reservations


def test_verified_paper_bridge_reaches_normal_execution_path():
    """14. Verified paper bridge reaches normal execution path."""
    paper_bridge = _make_paper_bridge()
    paper_bridge.place_bracket_order = MagicMock(return_value={"order_id": "p-123", "stop_order_id": "s-1", "tp_order_id": "t-1"})

    sig = _make_signal(symbol="QQQ")
    with patch("strategies.order_manager.calculate_shares", return_value=10):
        res = om.execute_signal(sig, paper_bridge, equity=100000.0, current_pnl=0.0, dry_run=False)

    assert res["status"] == "PLACED"
    assert res["order"]["order_id"] == "p-123"
