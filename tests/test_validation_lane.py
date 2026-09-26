"""
tests/test_validation_lane.py -- Unit test suite for Phase C: Two-Lane Active Paper Validation Engine.

Enforces:
1. Hard Safety is 100% Non-Bypassable:
   - Data Degraded regime fail-closed
   - SOXX tech Long fail-closed
   - Quote staleness (> 10s) veto
   - Cross-feed provenance dislocation veto
   - Price drift deviation (> 1.5%) veto
   - Negative latency veto
   - Invalid geometry & noise floor veto
   - Direct hard safety rejection codes can never enter validation
2. Permissible Soft-Policy Capture:
   - Confidence near-miss / below floor
   - Midday Lunch Chop Zone
3. Fixed Tiny Sizing: 1-3 shares (default: 1 share)
4. Isolated Client Order IDs: val_{symbol}_{strategy}_{hash}
5. Isolated Validation Ledger: data/paper_validation_trades.jsonl
6. Schema completeness: ValidationTradeRecord dataclass
"""
import os
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any
from unittest.mock import MagicMock
import pytest

from strategies.base import TradeSignal, StrategyName, OrderIntent
from strategies.validation_lane import (
    ValidationTradeRecord,
    evaluate_and_route_validation_candidate,
    is_hard_safety_failure,
    log_validation_record,
    HARD_SAFETY_VETO_CODES,
    SOFT_POLICY_REJECTION_CODES,
)


class MockValidationBridge:
    """Mock broker bridge providing configurable quote payloads and tracking orders."""
    def __init__(self, quote_payload: Optional[Dict[str, Any]] = None):
        self._quote = quote_payload
        self.placed_orders = []
        self.is_connected = True

    def set_quote(self, quote_payload: Optional[Dict[str, Any]]):
        self._quote = quote_payload

    def get_quote(self, symbol: str, feed: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if self._quote is None:
            return None
        q = dict(self._quote)
        if "symbol" not in q:
            q["symbol"] = symbol.upper().strip()
        if feed is not None and "feed_identity" not in q:
            q["feed_identity"] = feed
        return q

    def place_order(self, symbol: str, side: str, shares: int, limit_price: float,
                    stop_price: float, target_price: float, client_order_id: str) -> Any:
        order_obj = MagicMock()
        order_obj.id = f"ord_{client_order_id}"
        order_obj.client_order_id = client_order_id
        order_obj.symbol = symbol
        order_obj.side = side
        order_obj.qty = shares
        order_obj.status = "new"
        self.placed_orders.append({
            "symbol": symbol,
            "side": side,
            "shares": shares,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "client_order_id": client_order_id,
        })
        return order_obj


def _make_signal(
    symbol: str = "AAPL",
    strategy: StrategyName = StrategyName.ORB,
    side: str = "BUY",
    entry_price: float = 150.00,
    stop_price: float = 149.00,
    target_price: float = 152.00,
    atr: float = 1.20,
    confidence: float = 0.58,
    decision_feed: str = "sip",
    bar_timestamp: Optional[str] = None,
    reason: str = "test signal",
) -> TradeSignal:
    now_utc = datetime.now(timezone.utc)
    bar_ts = bar_timestamp or (now_utc - timedelta(seconds=60)).isoformat()
    risk = abs(entry_price - stop_price)
    return TradeSignal(
        symbol=symbol,
        strategy=strategy,
        side=side,
        confidence=confidence,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        atr=atr,
        risk_per_share=risk,
        reason=reason,
        decision_feed=decision_feed,
        bar_timestamp=bar_ts,
    )


def _make_quote(
    symbol: str = "AAPL",
    bid: float = 149.98,
    ask: float = 150.02,
    timestamp: Optional[float] = None,
    feed: str = "sip",
) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "last": (bid + ask) / 2.0,
        "timestamp": timestamp if timestamp is not None else time.time(),
        "feed_identity": feed,
        "source": "broker",
        "is_realtime": True,
    }


# =========================================================================
# Unit Tests
# =========================================================================

def test_hard_safety_veto_codes_rejected_immediately(tmp_path):
    """Direct hard safety rejection codes can never enter validation."""
    ledger_path = str(tmp_path / "val_trades.jsonl")
    bridge = MockValidationBridge(_make_quote())
    sig = _make_signal()

    for veto_code in ["CROSS_FEED_PROVENANCE_DISLOCATION", "QUOTE_STALE", "DATA_DEGRADED", "SOXX_TECH_LONG_VETO"]:
        res = evaluate_and_route_validation_candidate(
            signal=sig,
            bridge=bridge,
            policy_stage="STAGE_HARD_SAFETY",
            policy_reason_code=veto_code,
            policy_reason_msg=f"Violation of {veto_code}",
            filepath=ledger_path,
        )
        assert res["status"] == "VETOED"
        assert "Hard safety failure" in res["reason"]
        assert len(bridge.placed_orders) == 0
        assert not os.path.exists(ledger_path)


def test_regime_data_degraded_veto(tmp_path):
    """If regime is DATA_DEGRADED, candidate is vetoed fail-closed."""
    ledger_path = str(tmp_path / "val_trades.jsonl")
    bridge = MockValidationBridge(_make_quote())
    sig = _make_signal()

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.58 < 0.60",
        regime_mode="DATA_DEGRADED",
        filepath=ledger_path,
    )
    assert res["status"] == "VETOED"
    assert "DATA_DEGRADED" in res["reason"]
    assert len(bridge.placed_orders) == 0


def test_soxx_vwap_bounce_long_veto(tmp_path):
    """VWAP_BOUNCE Long during CAUTION/RISK_OFF with missing/invalid SOXX is vetoed."""
    ledger_path = str(tmp_path / "val_trades.jsonl")
    bridge = MockValidationBridge(_make_quote())
    sig = _make_signal(strategy=StrategyName.VWAP_BOUNCE, side="BUY")

    # Missing SOXX context
    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.58 < 0.60",
        regime_mode="CAUTION",
        regime_result=None,
        filepath=ledger_path,
    )
    assert res["status"] == "VETOED"
    assert "SOXX_TECH_LONG_VETO" in res["reason"]

    # SOXX below -1.50% threshold
    mock_reg = MagicMock()
    mock_reg.data = {"SOXX": {"pct": -1.85, "status": "VALID"}}
    res2 = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.58 < 0.60",
        regime_mode="CAUTION",
        regime_result=mock_reg,
        filepath=ledger_path,
    )
    assert res2["status"] == "VETOED"
    assert "SOXX -1.85%" in res2["reason"]
    assert len(bridge.placed_orders) == 0


def test_quote_staleness_hard_safety_veto(tmp_path):
    """Quote older than 10 seconds is strictly vetoed."""
    ledger_path = str(tmp_path / "val_trades.jsonl")
    stale_time = time.time() - 15.0  # 15s old
    bridge = MockValidationBridge(_make_quote(timestamp=stale_time))
    sig = _make_signal()

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.58 < 0.60",
        filepath=ledger_path,
    )
    assert res["status"] == "VETOED"
    assert "QUOTE_STALE" in res["reason"]
    assert len(bridge.placed_orders) == 0


def test_cross_feed_provenance_dislocation_veto(tmp_path):
    """Decision feed SIP + Quote feed IEX is strictly vetoed."""
    ledger_path = str(tmp_path / "val_trades.jsonl")
    bridge = MockValidationBridge(_make_quote(feed="iex"))
    sig = _make_signal(decision_feed="sip")

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_LUNCH_CHOP",
        policy_reason_code="LUNCH_CHOP_ZONE",
        policy_reason_msg="Lunch score 45 < 60",
        filepath=ledger_path,
    )
    assert res["status"] == "VETOED"
    assert "CROSS_FEED_PROVENANCE_DISLOCATION" in res["reason"]
    assert len(bridge.placed_orders) == 0


def test_excessive_price_deviation_veto(tmp_path):
    """Real-time quote deviated > 1.5% from decision price is vetoed."""
    ledger_path = str(tmp_path / "val_trades.jsonl")
    # Entry price 150.00, quote ask 153.00 (> 2.0% divergence)
    bridge = MockValidationBridge(_make_quote(bid=152.95, ask=153.05))
    sig = _make_signal(entry_price=150.00)

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.58 < 0.60",
        filepath=ledger_path,
    )
    assert res["status"] == "VETOED"
    assert "DEVIATION" in res["reason"] or "DRIFT" in res["reason"]
    assert len(bridge.placed_orders) == 0


def test_negative_latency_veto(tmp_path):
    """Quote timestamp older than bar timestamp by > 2.0s is vetoed."""
    ledger_path = str(tmp_path / "val_trades.jsonl")
    now = time.time()
    quote_ts = now - 6.0
    bar_str = datetime.fromtimestamp(now - 1.0, timezone.utc).isoformat()
    bridge = MockValidationBridge(_make_quote(timestamp=quote_ts, feed="sip"))
    sig = _make_signal(bar_timestamp=bar_str, decision_feed="sip")

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.58 < 0.60",
        filepath=ledger_path,
    )
    assert res["status"] == "VETOED"
    assert "NEGATIVE_LATENCY" in res["reason"]
    assert len(bridge.placed_orders) == 0


def test_invalid_geometry_noise_floor_veto(tmp_path):
    """Stop distance smaller than noise floor ($0.08) is adjusted or vetoed."""
    ledger_path = str(tmp_path / "val_trades.jsonl")
    bridge = MockValidationBridge(_make_quote())
    # Inverted geometry: BUY with stop above entry
    sig = _make_signal(entry_price=150.00, stop_price=151.00, target_price=149.00)

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.58 < 0.60",
        filepath=ledger_path,
    )
    assert res["status"] == "VETOED"
    assert "Geometry" in res["reason"] or "INVALID" in res["reason"]
    assert len(bridge.placed_orders) == 0


def test_validation_candidate_placed_paper_order(tmp_path):
    """
    Valid soft-policy rejection with passing hard safety places 1 share paper order
    and writes complete schema to isolated ledger.
    """
    ledger_path = str(tmp_path / "paper_validation_trades.jsonl")
    bridge = MockValidationBridge(_make_quote(bid=149.99, ask=150.01, feed="sip"))
    sig = _make_signal(
        symbol="AAPL",
        entry_price=150.00,
        stop_price=149.00,
        target_price=152.00,
        confidence=0.5850,
        decision_feed="sip",
    )

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.5850 < 0.6000",
        regime_mode="NEUTRAL",
        market_texture="NORMAL",
        fixed_shares=1,
        dry_run=False,
        filepath=ledger_path,
    )

    assert res["status"] == "PLACED"
    assert len(bridge.placed_orders) == 1
    placed = bridge.placed_orders[0]
    assert placed["symbol"] == "AAPL"
    assert placed["shares"] == 1
    assert placed["client_order_id"].startswith("val_aapl_")
    assert placed["limit_price"] == 150.01

    # Verify ledger record
    assert os.path.exists(ledger_path)
    with open(ledger_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    record_dict = json.loads(lines[0])

    assert record_dict["symbol"] == "AAPL"
    assert record_dict["validation_only"] is True
    assert record_dict["policy_rejected"] is True
    assert record_dict["hard_safety_pass"] is True
    assert record_dict["confidence"] == 0.5850
    assert record_dict["terminal_gate"] == "STAGE_CONFIDENCE"
    assert record_dict["decision_feed"] == "sip"
    assert record_dict["quote_feed"] == "sip"
    assert record_dict["stop_price"] == 149.01
    assert record_dict["target_price"] == 152.01
    assert record_dict["status"] == "PLACED" or record_dict["status"] == "SUBMITTED"


def test_validation_candidate_dry_run(tmp_path):
    """Dry run mode validates candidate and writes to ledger without calling bridge.place_order."""
    ledger_path = str(tmp_path / "paper_validation_trades.jsonl")
    bridge = MockValidationBridge(_make_quote(feed="sip"))
    sig = _make_signal(decision_feed="sip")

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_LUNCH_CHOP",
        policy_reason_code="LUNCH_CHOP_ZONE",
        policy_reason_msg="Lunch score 50 < 60",
        dry_run=True,
        filepath=ledger_path,
    )

    assert res["status"] == "DRY_RUN"
    assert len(bridge.placed_orders) == 0
    assert os.path.exists(ledger_path)
    with open(ledger_path, "r", encoding="utf-8") as f:
        record_dict = json.loads(f.readline())
    assert record_dict["status"] == "DRY_RUN"
    assert record_dict["validation_only"] is True


def test_isolated_ledger_does_not_pollute_production(tmp_path):
    """Validation ledger writes strictly to validation file, preserving production journal."""
    val_ledger = str(tmp_path / "paper_validation_trades.jsonl")
    prod_ledger = str(tmp_path / "trade_journal.json")

    bridge = MockValidationBridge(_make_quote(feed="sip"))
    sig = _make_signal(decision_feed="sip")

    evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.58 < 0.60",
        dry_run=True,
        filepath=val_ledger,
    )

    assert os.path.exists(val_ledger)
    assert not os.path.exists(prod_ledger)


def test_validation_candidate_with_place_bracket_order_bridge(tmp_path):
    """Verifies that bridge implementing place_bracket_order (like AlpacaBridge) works seamlessly."""
    ledger_path = str(tmp_path / "paper_val.jsonl")

    class MockAlpacaBridgeLike:
        def __init__(self):
            self.calls = []

        def get_quote(self, symbol, feed=None):
            return _make_quote(symbol=symbol, feed="sip")

        def place_bracket_order(self, symbol: str, side: str, qty: int, entry_price: float,
                                stop_price: float, target_price: float, client_order_id: str):
            self.calls.append({
                "symbol": symbol, "side": side, "qty": qty,
                "entry_price": entry_price, "stop_price": stop_price,
                "target_price": target_price, "client_order_id": client_order_id,
            })
            return {
                "order_id": f"alpaca_{client_order_id}",
                "stop_order_id": "stop_123",
                "tp_order_id": "tp_123",
                "placed": True,
                "submitted": True,
            }

    bridge = MockAlpacaBridgeLike()
    sig = _make_signal(symbol="MSFT", decision_feed="sip", confidence=0.585)

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.585 < 0.600",
        dry_run=False,
        filepath=ledger_path,
    )

    assert res["status"] == "PLACED"
    assert len(bridge.calls) == 1
    call = bridge.calls[0]
    assert call["symbol"] == "MSFT"
    assert call["qty"] == 1
    assert call["client_order_id"].startswith("val_msft_")
    assert res["record"].entry_order_id == f"alpaca_{call['client_order_id']}"


def test_validation_lane_strictly_blocks_real_money_account(tmp_path):
    """Validation lane MUST hard-fail closed if connected to real-money broker (is_paper=False)."""
    ledger_path = str(tmp_path / "paper_val.jsonl")

    bridge = MockValidationBridge(_make_quote(feed="sip"))
    bridge.is_paper = False  # Real money simulation!
    sig = _make_signal(symbol="AAPL", decision_feed="sip", confidence=0.585)

    res = evaluate_and_route_validation_candidate(
        signal=sig,
        bridge=bridge,
        policy_stage="STAGE_CONFIDENCE",
        policy_reason_code="CONFIDENCE_BELOW_FLOOR",
        policy_reason_msg="Confidence 0.585 < 0.600",
        dry_run=False,
        filepath=ledger_path,
    )

    assert res["status"] == "VETOED"
    assert "real-money" in res["reason"]
    assert len(bridge.placed_orders) == 0
    assert not os.path.exists(ledger_path)


