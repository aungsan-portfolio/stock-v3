"""
tests/test_golden_master_invariants.py - Deterministic Golden-Master Verification.

Verifies that the Gate 1 Phase A observability telemetry (order_audit Schema 2.0)
is strictly passive and side-effect free:
  - Generates deterministic test signals across diverse market conditions
  - Runs execution pipeline (execute_signal, noise floor checks, risk checks, bracket generation)
  - Canonicalizes normalized decision outputs
  - Verifies SHA-256 invariant hash stability
"""
import hashlib
import json
import time
from unittest.mock import MagicMock, patch
import pytest

from strategies.base import TradeSignal, EntryReadiness
from strategies.constants import StrategyName
import strategies.order_manager as om


def _create_mock_bridge():
    bridge = MagicMock()
    bridge.get_net_liquidation.return_value = 100000.0
    bridge.open_position_count.return_value = 0
    bridge.has_position.return_value = False
    bridge.has_working_order.return_value = False
    bridge.get_open_positions.return_value = []
    bridge.get_working_orders.return_value = []
    bridge.paper = True
    return bridge


def run_deterministic_scenarios():
    bridge = _create_mock_bridge()
    cycle_bar_ts = "2026-09-15T14:15:00+00:00"
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="golden_master_cycle",
        bar_ts=cycle_bar_ts,
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )

    test_signals = [
        TradeSignal(
            symbol="AAPL",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.720,
            entry_price=220.50,
            stop_price=219.00,
            target_price=223.50,
            atr=1.80,
            risk_per_share=1.50,
            reason="VWAP bounce",
            bar_timestamp="2026-09-15 14:15:00+00:00",
        ),
        TradeSignal(
            symbol="SPY",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.740,
            entry_price=560.20,
            stop_price=558.50,
            target_price=563.50,
            atr=2.50,
            risk_per_share=1.70,
            reason="VWAP bounce",
            bar_timestamp="2026-09-15 14:15:00+00:00",
        ),
        TradeSignal(
            symbol="QQQ",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.710,
            entry_price=480.10,
            stop_price=478.00,
            target_price=484.00,
            atr=3.20,
            risk_per_share=2.10,
            reason="VWAP bounce",
            bar_timestamp="2026-09-15 14:15:00+00:00",
        ),
        TradeSignal(
            symbol="MSFT",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.750,
            entry_price=430.00,
            stop_price=427.00,
            target_price=436.00,
            atr=2.50,
            risk_per_share=3.00,
            reason="Stale bar test",
            bar_timestamp="2026-09-15 14:05:00+00:00",
        ),
        TradeSignal(
            symbol="AMZN",
            strategy=StrategyName.VWAP_BOUNCE,
            side="BUY",
            confidence=0.750,
            entry_price=185.00,
            stop_price=183.00,
            target_price=189.00,
            atr=1.50,
            risk_per_share=2.00,
            reason="Missing ts test",
            bar_timestamp=None,
        ),
    ]

    results = []
    with patch("strategies.order_manager.pre_trade_check", return_value=None), \
         patch("strategies.order_manager._min_confidence_for", return_value=0.6000000), \
         patch("config.VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", False):

        for sig in test_signals:
            res = om.execute_signal(
                signal=sig,
                bridge=bridge,
                equity=100000.0,
                current_pnl=0.0,
                dry_run=True,
                entry_readiness=readiness,
            )
            norm_res = {
                "symbol": sig.symbol,
                "strategy": sig.strategy.value if hasattr(sig.strategy, "value") else str(sig.strategy),
                "status": res.get("status"),
                "reason": res.get("reason", "")[:60],
                "shares": res.get("shares"),
                "entry_price": res.get("entry_price"),
                "stop_price": res.get("stop_price"),
                "target_price": res.get("target_price"),
            }
            results.append(norm_res)

    canonical_json = json.dumps(results, sort_keys=True)
    sha256_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return sha256_hash, results


def test_golden_master_behavior_hash_invariance():
    """Verify that normalized execution behavior hash is deterministic and invariant."""
    hash1, results1 = run_deterministic_scenarios()
    hash2, results2 = run_deterministic_scenarios()

    for idx, r in enumerate(results1):
        print(f"Signal {idx} ({r['symbol']}): status={r['status']}, reason={r['reason']}")

    assert hash1 == hash2, "Behavior hash must be 100% deterministic across executions"
    assert len(results1) == 5
    assert results1[0]["status"] == "DRY_RUN"
    assert results1[1]["status"] == "REJECTED"
    assert "REJECTED_NOISE_FLOOR_BREACH" in results1[1]["reason"]
    assert results1[2]["status"] == "REJECTED"
    assert "REJECTED_NOISE_FLOOR_BREACH" in results1[2]["reason"]
    assert results1[3]["status"] == "REJECTED"
    assert "ENTRY_READINESS_BAR_MISMATCH" in results1[3]["reason"]
    assert results1[4]["status"] == "REJECTED"
    assert "ENTRY_READINESS_BAR_INVALID" in results1[4]["reason"]
    assert hash1 == "8a35d1e08ef09e17532ef2fe1debdcb23a732a73137af86f80129ca1cd126030"


def test_golden_master_production_gate_fail_closed_without_regime():
    """Verify that when VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED=True, missing regime context fails closed."""
    bridge = _create_mock_bridge()
    cycle_bar_ts = "2026-09-15T14:15:00+00:00"
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="golden_master_cycle",
        bar_ts=cycle_bar_ts,
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )
    sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.720,
        entry_price=220.50,
        stop_price=219.00,
        target_price=223.50,
        atr=1.80,
        risk_per_share=1.50,
        reason="VWAP bounce",
        bar_timestamp="2026-09-15 14:15:00+00:00",
    )

    with patch("strategies.order_manager.pre_trade_check", return_value=None), \
         patch("strategies.order_manager._min_confidence_for", return_value=0.60), \
         patch("config.VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", True):

        # Execute with regime_result=None -> must fail-closed
        res = om.execute_signal(
            signal=sig,
            bridge=bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            entry_readiness=readiness,
            regime_result=None,
        )

        assert res["status"] == "REJECTED"
        assert "Regime/SOXX Gate VETO: Missing regime context for VWAP_BOUNCE (fail-closed)" in res["reason"]


class _MockRegime:
    def __init__(self, mode="RISK_ON", allow_new_shorts=False, data=None, confidence_boost=0.0):
        self.mode = mode
        self.allow_new_shorts = allow_new_shorts
        self.data = data or {}
        self.confidence_boost = confidence_boost


def test_golden_master_production_gate_fail_closed_on_invalid_or_breached_soxx():
    """Verify fail-closed rejection when regime is RISK_OFF and SOXX is invalid or breached."""
    bridge = _create_mock_bridge()
    cycle_bar_ts = "2026-09-15T14:15:00+00:00"
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="golden_master_cycle",
        bar_ts=cycle_bar_ts,
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )
    sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.720,
        entry_price=220.50,
        stop_price=219.00,
        target_price=223.50,
        atr=1.80,
        risk_per_share=1.50,
        reason="VWAP bounce",
        bar_timestamp="2026-09-15 14:15:00+00:00",
    )

    with patch("strategies.order_manager.pre_trade_check", return_value=None), \
         patch("strategies.order_manager._min_confidence_for", return_value=0.60), \
         patch("config.VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", True):

        # Case A: Regime is RISK_OFF but SOXX data is missing
        regime_missing_soxx = _MockRegime(mode="RISK_OFF", data={})
        res_a = om.execute_signal(
            signal=sig,
            bridge=bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            entry_readiness=readiness,
            regime_result=regime_missing_soxx,
        )
        assert res_a["status"] == "REJECTED"
        assert "SOXX benchmark data unavailable or invalid" in res_a["reason"]

        # Case B: Regime is RISK_OFF and SOXX is breached (-2.50% <= -1.50%)
        regime_breached_soxx = _MockRegime(
            mode="RISK_OFF",
            data={"SOXX": {"pct": -2.50, "status": "VALID"}},
        )
        res_b = om.execute_signal(
            signal=sig,
            bridge=bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            entry_readiness=readiness,
            regime_result=regime_breached_soxx,
        )
        assert res_b["status"] == "REJECTED"
        assert "SOXX -2.50% (<= -1.50%)" in res_b["reason"]


def test_golden_master_production_gate_allows_valid_signal():
    """Verify that with VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED=True and valid regime input, signal proceeds."""
    bridge = _create_mock_bridge()
    cycle_bar_ts = "2026-09-15T14:15:00+00:00"
    readiness = EntryReadiness(
        entry_allowed=True,
        suppression_reason=None,
        cycle_id="golden_master_cycle",
        bar_ts=cycle_bar_ts,
        created_at=time.time(),
        expires_at=time.time() + 300.0,
    )
    sig = TradeSignal(
        symbol="AAPL",
        strategy=StrategyName.VWAP_BOUNCE,
        side="BUY",
        confidence=0.720,
        entry_price=220.50,
        stop_price=219.00,
        target_price=223.50,
        atr=1.80,
        risk_per_share=1.50,
        reason="VWAP bounce",
        bar_timestamp="2026-09-15 14:15:00+00:00",
    )

    # Valid RISK_ON regime with positive SOXX benchmark
    valid_regime = _MockRegime(
        mode="RISK_ON",
        allow_new_shorts=False,
        data={"SOXX": {"pct": 0.85, "status": "VALID"}},
    )

    with patch("strategies.order_manager.pre_trade_check", return_value=None), \
         patch("strategies.order_manager._min_confidence_for", return_value=0.60), \
         patch("config.VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", True):

        res = om.execute_signal(
            signal=sig,
            bridge=bridge,
            equity=100000.0,
            current_pnl=0.0,
            dry_run=True,
            entry_readiness=readiness,
            regime_result=valid_regime,
        )
        assert res["status"] == "DRY_RUN", f"Expected DRY_RUN but got: {res}"
        assert res["shares"] > 0
        assert res["symbol"] == "AAPL"
