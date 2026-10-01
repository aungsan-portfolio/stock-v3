"""tests/test_repricing_geometry_alignment.py -- Targeted tests for Lane 3 (D2-Geometry Repricing Realignment)."""

import time
import pytest
from unittest.mock import MagicMock

from strategies.base import TradeSignal, EntryReadiness
from strategies.constants import StrategyName
from strategies.order_manager import _executable_signal, execute_signal
from strategies.position_sizer import calculate_shares


def _make_signal(symbol="AMD", side="BUY", entry=100.0, stop=99.0, target=102.0, strategy=StrategyName.MOMENTUM_SCALP):
    return TradeSignal(
        symbol=symbol,
        strategy=strategy,
        side=side,
        confidence=0.85,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        atr=1.0,
        risk_per_share=abs(entry - stop),
        reason="Test geometry setup",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        bar_timestamp="2026-10-01 15:30:00",
    )


def test_buy_relative_geometry_recalculation():
    """BUY signal with live limit repricing must shift stop and target to preserve R:R and stop distance."""
    sig = _make_signal(side="BUY", entry=100.0, stop=99.0, target=102.0)
    exec_sig = _executable_signal(sig, entry_limit_price=100.50)

    assert exec_sig.entry_price == 100.50
    assert exec_sig.stop_price == 99.50
    assert exec_sig.target_price == 102.50
    assert exec_sig.risk_per_share == 1.00

    # Verify R:R preservation
    reward = exec_sig.target_price - exec_sig.entry_price
    risk = exec_sig.entry_price - exec_sig.stop_price
    assert pytest.approx(reward / risk) == 2.0

    # Verify metadata preservation
    meta = exec_sig.metadata
    assert meta["orig_entry"] == 100.0
    assert meta["orig_stop"] == 99.0
    assert meta["orig_target"] == 102.0
    assert meta["recomputed_entry"] == 100.50
    assert meta["recomputed_stop"] == 99.50
    assert meta["recomputed_target"] == 102.50
    assert meta["recomputed_stop_dist"] == 1.00


def test_sell_relative_geometry_recalculation():
    """SELL signal with live limit repricing must shift stop and target to preserve R:R and stop distance."""
    sig = _make_signal(side="SELL", entry=100.0, stop=101.0, target=98.0)
    exec_sig = _executable_signal(sig, entry_limit_price=99.50)

    assert exec_sig.entry_price == 99.50
    assert exec_sig.stop_price == 100.50
    assert exec_sig.target_price == 97.50
    assert exec_sig.risk_per_share == 1.00

    # Verify R:R preservation
    reward = exec_sig.entry_price - exec_sig.target_price
    risk = exec_sig.stop_price - exec_sig.entry_price
    assert pytest.approx(reward / risk) == 2.0

    # Verify metadata preservation
    meta = exec_sig.metadata
    assert meta["orig_entry"] == 100.0
    assert meta["orig_stop"] == 101.0
    assert meta["orig_target"] == 98.0
    assert meta["recomputed_entry"] == 99.50
    assert meta["recomputed_stop"] == 100.50
    assert meta["recomputed_target"] == 97.50
    assert meta["recomputed_stop_dist"] == 1.00


def test_anchored_strategy_retains_fixed_anchors():
    """MEAN_REVERSION signals retain anchored VWAP stop and target."""
    sig = _make_signal(
        side="BUY", entry=100.0, stop=99.0, target=102.0,
        strategy=StrategyName.MEAN_REVERSION,
    )
    sig.metadata["target_anchor"] = "VWAP"
    exec_sig = _executable_signal(sig, entry_limit_price=100.20)

    assert exec_sig.entry_price == 100.20
    assert exec_sig.stop_price == 99.00
    assert exec_sig.target_price == 102.00
    assert pytest.approx(exec_sig.risk_per_share, abs=1e-4) == 1.20


def test_dollar_risk_invariant_under_recomputed_stop_sizing():
    """calculate_shares with recomputed geometry maintains dollar risk cap <= $50."""
    sig = _make_signal(entry=100.0, stop=99.0, target=102.0)
    exec_sig = _executable_signal(sig, entry_limit_price=100.50)

    equity = 10000.0
    shares = calculate_shares(exec_sig, equity)
    assert shares > 0

    dollar_risk = shares * exec_sig.risk_per_share
    assert dollar_risk <= 50.0 + 1e-6


def test_execute_signal_with_recomputed_geometry_no_rr_collapse():
    """Full execute_signal pipeline does not reject due to R:R collapse when entry reprices."""
    sig = _make_signal(entry=100.0, stop=99.0, target=102.0)
    bridge = MagicMock()
    bridge.is_connected = True
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.open_position_count.return_value = 0
    bridge.get_positions.return_value = []
    bridge.get_quote.return_value = {
        "bid": 100.45,
        "ask": 100.50,
        "last": 100.48,
        "timestamp": time.time(),
        "is_realtime": True,
    }

    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="cycle_geo",
        bar_ts=sig.bar_timestamp,
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )

    res = execute_signal(
        signal=sig,
        bridge=bridge,
        equity=10000.0,
        current_pnl=0.0,
        dry_run=True,
        entry_limit_price=100.10,
        entry_readiness=readiness,
    )

    assert res["status"] == "DRY_RUN"
    assert "Risk:Reward" not in res.get("reason", "")
