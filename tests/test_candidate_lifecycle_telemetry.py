import pytest
from unittest.mock import Mock
from strategies.base import TradeSignal
from strategies.constants import StrategyName, OrderIntent
from strategies.gate_ledger import log_gate_decision
import order_audit


def test_trade_signal_deterministic_setup_id():
    sig1 = TradeSignal(
        symbol="UBER",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.72,
        entry_price=70.22,
        stop_price=69.80,
        target_price=71.00,
        atr=0.50,
        risk_per_share=0.42,
        reason="Breakout test",
        bar_timestamp="2026-10-19T20:30:00+00:00",
    )
    sig2 = TradeSignal(
        symbol="UBER",
        strategy=StrategyName.MOMENTUM_SCALP,
        side="BUY",
        confidence=0.75,
        entry_price=70.25,
        stop_price=69.80,
        target_price=71.10,
        atr=0.50,
        risk_per_share=0.45,
        reason="Repeated evaluation pass on same bar",
        bar_timestamp="2026-10-19T20:30:00+00:00",
    )

    # Both passes on same bar/setup share identical setup_id
    assert sig1.setup_id is not None
    assert sig1.setup_id == sig2.setup_id
    assert "UBER" in sig1.setup_id
    assert "MOMENTUM_SCALP" in sig1.setup_id
    assert "BUY" in sig1.setup_id

    # But their signal_id (individual evaluation pass) differs
    assert sig1.signal_id != sig2.signal_id


def test_order_audit_records_setup_id(tmp_path):
    tracker = order_audit.OrderLifecycleTracker()
    client_id = "test_cl_order_123"
    setup_id = "stp_UBER_MOMENTUM_SCALP_BUY_20261019T203000"

    ord_obj = tracker.record_submission(
        client_order_id=client_id,
        symbol="UBER",
        side="BUY",
        limit_price=70.22,
        requested_qty=50,
        signal_id="sig_pass_999",
        setup_id=setup_id,
    )

    assert ord_obj.setup_id == setup_id
    assert ord_obj.signal_id == "sig_pass_999"


def test_gate_ledger_records_setup_id(tmp_path):
    ledger_file = tmp_path / "trade_gate_decision.jsonl"
    setup_id = "stp_AMD_MOMENTUM_SCALP_BUY_20261019T203500"

    rec = log_gate_decision(
        evaluation_id="sig_amd_eval_1",
        symbol="AMD",
        bar="2026-10-19T20:35:00+00:00",
        stage="STAGE_MAX_CHASE",
        decision="REJECT",
        reason_code="CHASE_EXCEEDED",
        human_message="Quote drifted $2.27 > $0.15 chase limit",
        strategy="MOMENTUM_SCALP",
        terminal_decision=True,
        setup_id=setup_id,
        filepath=str(ledger_file),
    )

    assert rec["setup_id"] == setup_id
    assert rec["evaluation_id"] == "sig_amd_eval_1"
    assert rec["decision"] == "REJECT"
    assert rec["symbol"] == "AMD"


def test_unique_setups_vs_repeated_evaluation_passes():
    # 5 evaluation passes on 2 unique setups
    signals = [
        TradeSignal(
            symbol="AMD",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.65,
            entry_price=150.0,
            stop_price=149.0,
            target_price=152.0,
            atr=1.0,
            risk_per_share=1.0,
            reason="Pass 1",
            bar_timestamp="2026-10-19T20:30:00+00:00",
        ),
        TradeSignal(
            symbol="AMD",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.66,
            entry_price=150.1,
            stop_price=149.0,
            target_price=152.0,
            atr=1.0,
            risk_per_share=1.1,
            reason="Pass 2 (same setup)",
            bar_timestamp="2026-10-19T20:30:00+00:00",
        ),
        TradeSignal(
            symbol="UBER",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.70,
            entry_price=70.0,
            stop_price=69.5,
            target_price=71.0,
            atr=0.5,
            risk_per_share=0.5,
            reason="Pass 3 (different setup)",
            bar_timestamp="2026-10-19T20:30:00+00:00",
        ),
        TradeSignal(
            symbol="UBER",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.71,
            entry_price=70.1,
            stop_price=69.5,
            target_price=71.0,
            atr=0.5,
            risk_per_share=0.6,
            reason="Pass 4 (same UBER setup)",
            bar_timestamp="2026-10-19T20:30:00+00:00",
        ),
        TradeSignal(
            symbol="UBER",
            strategy=StrategyName.MOMENTUM_SCALP,
            side="BUY",
            confidence=0.72,
            entry_price=70.2,
            stop_price=69.5,
            target_price=71.0,
            atr=0.5,
            risk_per_share=0.7,
            reason="Pass 5 (same UBER setup)",
            bar_timestamp="2026-10-19T20:30:00+00:00",
        ),
    ]

    raw_eval_passes = len(signals)
    unique_setups = len({s.setup_id for s in signals})

    assert raw_eval_passes == 5
    assert unique_setups == 2
