"""tests/test_quote_age_refetch.py -- Targeted tests for Lane 2 (Quote-Age Concordance Fix)."""

import time
import pytest
from unittest.mock import MagicMock, patch

from strategies.base import TradeSignal, EntryReadiness
from strategies.constants import StrategyName
from strategies.order_manager import execute_signal, validate_quote_health
from strategies.execution_quote import validate_execution_quote, BrokerQuote

BAR_TS = "2026-10-01 15:30:00"


def _make_dummy_signal(symbol="AMD", side="BUY", entry=150.0, stop=148.0, target=154.0, age=0.0):
    sig = TradeSignal(
        symbol=symbol,
        strategy=StrategyName.MOMENTUM_SCALP,
        side=side,
        confidence=0.85,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        atr=1.5,
        risk_per_share=abs(entry - stop),
        reason="Test setup",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age)),
        bar_timestamp=BAR_TS,
    )
    sig.metadata = {
        "raw_confidence": 0.85,
        "created_at": time.time() - age,
    }
    return sig


def _make_readiness():
    return EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="cycle_test",
        bar_ts=BAR_TS,
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )


def test_quote_refetch_overrides_stale_batch_quote():
    """If signal metadata has a 12s-old quote, bridge.get_quote provides fresh quote so it's not vetoed."""
    sig = _make_dummy_signal(age=12.0)
    sig.metadata["quote"] = {
        "bid": 149.95,
        "ask": 150.05,
        "timestamp": time.time() - 12.0,
        "is_realtime": True,
    }

    fresh_quote = {
        "bid": 150.00,
        "ask": 150.02,
        "last": 150.01,
        "timestamp": time.time(),
        "is_realtime": True,
        "feed_identity": "iex",
    }
    bridge = MagicMock()
    bridge.get_quote.return_value = fresh_quote
    bridge.is_connected = True
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []

    res = execute_signal(
        signal=sig,
        bridge=bridge,
        equity=10000.0,
        current_pnl=0.0,
        dry_run=True,
        entry_readiness=_make_readiness(),
    )

    bridge.get_quote.assert_called_once_with("AMD")
    assert res["status"] == "DRY_RUN"
    assert "STALE_QUOTE" not in res.get("reason", "")
    assert "quote_metrics" in res["metadata"]
    assert res["metadata"]["quote_metrics"]["age_seconds"] < 2.0


def test_quote_refetch_fail_closed_in_live_mode():
    """In live paper mode, if bridge.get_quote returns None, fail closed with QUOTE_REFETCH_FAILED."""
    sig = _make_dummy_signal()
    bridge = MagicMock()
    bridge.get_quote.return_value = None
    bridge.is_connected = True
    bridge._connected = True

    with patch("strategies.order_manager._assert_paper_broker", return_value=""):
        res = execute_signal(
            signal=sig,
            bridge=bridge,
            equity=10000.0,
            current_pnl=0.0,
            dry_run=False,
            entry_readiness=_make_readiness(),
        )

    assert res["status"] == "REJECTED"
    assert "QUOTE_REFETCH_FAILED" in res["reason"]


def test_quote_refetch_exception_fail_closed_in_live_mode():
    """In live paper mode, if bridge.get_quote raises exception, fail closed with QUOTE_REFETCH_FAILED."""
    sig = _make_dummy_signal()
    bridge = MagicMock()
    bridge.get_quote.side_effect = RuntimeError("Network timeout to quote venue")
    bridge.is_connected = True
    bridge._connected = True

    with patch("strategies.order_manager._assert_paper_broker", return_value=""):
        res = execute_signal(
            signal=sig,
            bridge=bridge,
            equity=10000.0,
            current_pnl=0.0,
            dry_run=False,
            entry_readiness=_make_readiness(),
        )

    assert res["status"] == "REJECTED"
    assert "QUOTE_REFETCH_FAILED" in res["reason"]
    assert "Network timeout" in res["reason"]


def test_diagnostic_timestamps_recorded_in_metadata():
    """Verify that quote_fetch_time, signal_generation_time, veto_evaluation_time are logged and recorded."""
    sig = _make_dummy_signal(age=2.5)
    sig.metadata["quote"] = {
        "bid": 149.98,
        "ask": 150.02,
        "timestamp": time.time(),
        "is_realtime": True,
    }
    bridge = MagicMock()
    bridge.get_quote.return_value = sig.metadata["quote"]
    bridge.is_connected = True
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []

    res = execute_signal(
        signal=sig,
        bridge=bridge,
        equity=10000.0,
        current_pnl=0.0,
        dry_run=True,
        entry_readiness=_make_readiness(),
    )

    meta = res["metadata"]
    assert "quote_fetch_time" in meta
    assert "signal_generation_time" in meta
    assert "veto_evaluation_time" in meta
    assert "execution_latency_ms" in meta
    assert meta["quote_fetch_time"] > 0
    assert meta["signal_generation_time"] > 0
    assert meta["veto_evaluation_time"] >= meta["quote_fetch_time"]
    assert meta["execution_latency_ms"] >= 0


def test_validate_execution_quote_concordance_with_passed_quote():
    """validate_execution_quote must use passed quote and avoid redundant bridge.get_quote call."""
    bridge = MagicMock()
    passed_quote = {
        "bid": 150.00,
        "ask": 150.05,
        "last": 150.02,
        "timestamp": time.time(),
        "source": "broker",
        "is_realtime": True,
    }

    res = validate_execution_quote(
        bridge=bridge,
        symbol="AMD",
        decision_price=150.02,
        quote=passed_quote,
    )

    assert res.valid is True
    assert res.quote.bid == 150.00
    assert res.quote.ask == 150.05
    # bridge.get_quote was NOT called because quote was provided
    bridge.get_quote.assert_not_called()
