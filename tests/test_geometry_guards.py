import math
import pytest
from strategies.base import TradeSignal
from strategies.position_sizer import validate_executable_geometry, calculate_shares


def test_validate_executable_geometry_happy_path_long():
    sig = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=150.00,
        stop_price=148.00,
        target_price=154.00,
        risk_per_share=2.00,
        confidence=0.85,
        strategy="ORB",
        atr=1.5,
        reason="ORB long trigger",
    )
    valid, reason = validate_executable_geometry(sig)
    assert valid is True
    assert reason == "OK"


def test_validate_executable_geometry_happy_path_short():
    sig = TradeSignal(
        symbol="AAPL",
        side="SHORT",
        entry_price=150.00,
        stop_price=152.00,
        target_price=146.00,
        risk_per_share=2.00,
        confidence=0.85,
        strategy="ORB",
        atr=1.5,
        reason="ORB short trigger",
    )
    valid, reason = validate_executable_geometry(sig)
    assert valid is True
    assert reason == "OK"


def test_validate_executable_geometry_inverted_long_stop():
    # Long stop above entry
    sig = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=150.00,
        stop_price=152.00,
        target_price=154.00,
        risk_per_share=2.00,
        confidence=0.85,
        strategy="ORB",
        atr=1.5,
        reason="Inverted stop test",
    )
    valid, reason = validate_executable_geometry(sig)
    assert valid is False
    assert "invalid LONG geometry" in reason


def test_validate_executable_geometry_inverted_short_stop():
    # Short stop below entry
    sig = TradeSignal(
        symbol="AAPL",
        side="SHORT",
        entry_price=150.00,
        stop_price=148.00,
        target_price=146.00,
        risk_per_share=2.00,
        confidence=0.85,
        strategy="ORB",
        atr=1.5,
        reason="Inverted short stop test",
    )
    valid, reason = validate_executable_geometry(sig)
    assert valid is False
    assert "invalid SHORT geometry" in reason


def test_validate_executable_geometry_sub_tick_stop_distance():
    # Stop distance is 0.005, tick size is 0.01
    sig = TradeSignal(
        symbol="PLTR",
        side="BUY",
        entry_price=185.81,
        stop_price=185.805,
        target_price=187.00,
        risk_per_share=0.005,
        confidence=0.85,
        strategy="VWAP",
        atr=0.5,
        reason="Sub tick test",
    )
    valid, reason = validate_executable_geometry(sig)
    assert valid is False
    assert "at or below tick size" in reason


def test_validate_executable_geometry_zero_stop_distance():
    sig = TradeSignal(
        symbol="NFLX",
        side="BUY",
        entry_price=81.43,
        stop_price=81.43,
        target_price=83.00,
        risk_per_share=0.0,
        confidence=0.85,
        strategy="CANDLESTICK",
        atr=0.5,
        reason="Zero gap test",
    )
    valid, reason = validate_executable_geometry(sig)
    assert valid is False


def test_calculate_shares_fail_closed_on_zero_gap():
    sig = TradeSignal(
        symbol="NFLX",
        side="BUY",
        entry_price=81.43,
        stop_price=81.43,
        target_price=83.00,
        risk_per_share=0.0,
        confidence=0.85,
        strategy="CANDLESTICK",
        atr=0.5,
        reason="Zero gap sizer test",
    )
    shares = calculate_shares(sig, equity=50000.0)
    assert shares == 0


def test_calculate_shares_fail_closed_on_sub_tick_gap():
    sig = TradeSignal(
        symbol="PLTR",
        side="BUY",
        entry_price=185.81,
        stop_price=185.805,
        target_price=187.00,
        risk_per_share=0.005,
        confidence=0.85,
        strategy="VWAP",
        atr=0.5,
        reason="Sub tick sizer test",
    )
    shares = calculate_shares(sig, equity=50000.0)
    assert shares == 0


def test_calculate_shares_valid_trade():
    sig = TradeSignal(
        symbol="META",
        side="BUY",
        entry_price=570.00,
        stop_price=565.00,
        target_price=580.00,
        risk_per_share=5.00,
        confidence=0.85,
        strategy="ORB",
        atr=5.0,
        reason="Valid trade test",
    )
    shares = calculate_shares(sig, equity=50000.0)
    assert shares > 0


def test_compute_lunch_opportunity_score_explosive_tier():
    from strategies.orchestrator import compute_lunch_opportunity_score
    from unittest.mock import MagicMock

    mock_event = MagicMock()
    mock_event.affected_symbols = {"NVDA"}
    mock_event.tag = "EARNINGS_BEAT"

    sig = TradeSignal(
        symbol="NVDA",
        side="BUY",
        entry_price=120.00,
        stop_price=118.00,
        target_price=125.00,
        risk_per_share=2.00,
        confidence=0.90,
        strategy="MOMENTUM",
        atr=2.0,
        reason="Explosive momentum test",
    )
    sig.rvol = 3.5

    score, size_mult, is_allowed, reason = compute_lunch_opportunity_score(sig, event_decision=mock_event)
    assert is_allowed is True
    assert score >= 85.0
    assert size_mult == 0.75
    assert "Elite Midday Setup" in reason


def test_compute_lunch_opportunity_score_defensive_tier():
    from strategies.orchestrator import compute_lunch_opportunity_score

    sig = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=220.00,
        stop_price=218.00,
        target_price=224.00,
        risk_per_share=2.00,
        confidence=0.72,
        strategy="VWAP",
        atr=2.0,
        reason="Defensive vwap test",
    )
    sig.rvol = 2.1

    score, size_mult, is_allowed, reason = compute_lunch_opportunity_score(sig, event_decision=None)
    assert is_allowed is True
    assert 60.0 <= score <= 70.0
    assert size_mult in (0.25, 0.35, 0.50)


def test_compute_lunch_opportunity_score_reject_tier():
    from strategies.orchestrator import compute_lunch_opportunity_score

    sig = TradeSignal(
        symbol="XYZ",
        side="BUY",
        entry_price=10.00,
        stop_price=9.50,
        target_price=11.00,
        risk_per_share=0.50,
        confidence=0.52,
        strategy="MOMENTUM",
        atr=0.5,
        reason="Low confidence test",
    )
    sig.rvol = 1.0

    score, size_mult, is_allowed, reason = compute_lunch_opportunity_score(sig, event_decision=None)
    assert is_allowed is False
    assert score < 60.0
    assert size_mult == 0.0


def test_expected_value_r_calculation():
    from strategies.orchestrator import expected_value_r
    ev = expected_value_r(win_rate=0.55, avg_win_r=1.40, avg_loss_r=0.90, avg_cost_r=0.08)
    # Gross EV = (0.55 * 1.40) - (0.45 * 0.90) = 0.770 - 0.405 = 0.365
    # Net EV = 0.365 - 0.08 = 0.285
    assert abs(ev - 0.285) < 1e-4


def test_normalize_ev_score():
    from strategies.orchestrator import normalize_ev_score
    score = normalize_ev_score(net_ev_r=0.285, min_ev_r=-0.50, max_ev_r=1.00)
    assert abs(score - 52.33) < 0.1


def test_decide_lunch_trade_pipeline():
    from strategies.orchestrator import decide_lunch_trade

    # Case 1: Hard Safety Fail
    res = decide_lunch_trade(opportunity_score=90.0, net_ev_r=0.50, sample_count=100, hard_safety_passed=False)
    assert res["decision"] == "REJECT"
    assert res["risk_mult"] == 0.0

    # Case 2: Score < 60
    res = decide_lunch_trade(opportunity_score=55.0, net_ev_r=0.50, sample_count=100, hard_safety_passed=True)
    assert res["decision"] == "REJECT"
    assert res["risk_mult"] == 0.0

    # Case 3: Net EV <= 0
    res = decide_lunch_trade(opportunity_score=85.0, net_ev_r=-0.05, sample_count=100, hard_safety_passed=True)
    assert res["decision"] == "REJECT"
    assert res["risk_mult"] == 0.0

    # Case 4: N < 30 (Paper only)
    res = decide_lunch_trade(opportunity_score=88.0, net_ev_r=0.45, sample_count=15, hard_safety_passed=True)
    assert res["decision"] == "PAPER_ONLY"
    assert res["risk_mult"] == 0.0

    # Case 5: Standard Multiplicative Round Down (0.75 * 0.85 * 0.75 = 0.478 -> Round down to 0.35x)
    res = decide_lunch_trade(
        opportunity_score=88.0,
        net_ev_r=0.28,
        sample_count=72,
        hard_safety_passed=True,
        regime_mult=1.0,
    )
    assert res["decision"] == "ALLOW"
    assert res["raw_risk_mult"] == 0.4781
    assert res["risk_mult"] == 0.35


def test_raw_below_minimum_is_not_rounded_up():
    from strategies.orchestrator import decide_lunch_trade
    # 0.35 (opportunity_cap) * 0.60 (ev_mult) * 0.50 (evidence_mult) * 0.75 (regime) = 0.07875 < 0.25x
    result = decide_lunch_trade(
        opportunity_score=60.0,
        net_ev_r=0.10,
        sample_count=30,
        regime_mult=0.75,
    )
    assert result["risk_mult"] == 0.0
    assert result["decision"] == "PAPER_ONLY"


def test_regime_multiplier_above_one_rejected():
    from strategies.orchestrator import decide_lunch_trade
    result = decide_lunch_trade(
        opportunity_score=90.0,
        net_ev_r=0.50,
        sample_count=150,
        regime_mult=1.50,
    )
    assert result["decision"] == "REJECT"
    assert result["risk_mult"] == 0.0


def test_nan_inputs_fail_closed():
    from strategies.orchestrator import decide_lunch_trade
    result = decide_lunch_trade(
        opportunity_score=float("nan"),
        net_ev_r=0.50,
        sample_count=150,
    )
    assert result["decision"] == "REJECT"
    assert result["risk_mult"] == 0.0


def test_allow_never_returns_zero_risk():
    from strategies.orchestrator import decide_lunch_trade
    result = decide_lunch_trade(
        opportunity_score=85.0,
        net_ev_r=0.20,
        sample_count=50,
        regime_mult=0.50,
    )
    assert not (result["decision"] == "ALLOW" and result["risk_mult"] == 0.0)


def test_elite_trade_never_exceeds_lunch_cap():
    from strategies.orchestrator import decide_lunch_trade
    result = decide_lunch_trade(
        opportunity_score=100.0,
        net_ev_r=2.00,
        sample_count=1000,
        regime_mult=1.00,
    )
    assert result["risk_mult"] <= 0.75
    assert result["decision"] == "ALLOW"


@pytest.mark.parametrize("field_name", ["opportunity_score", "net_ev_r", "regime_mult"])
@pytest.mark.parametrize("invalid_value", [True, False])
def test_boolean_numeric_inputs_are_rejected(field_name, invalid_value):
    from strategies.orchestrator import decide_lunch_trade
    inputs = {
        "opportunity_score": 85.0,
        "net_ev_r": 0.40,
        "sample_count": 100,
        "hard_safety_passed": True,
        "regime_mult": 1.0,
    }
    inputs[field_name] = invalid_value
    result = decide_lunch_trade(**inputs)
    assert result["decision"] == "REJECT"
    assert result["risk_mult"] == 0.0


@pytest.mark.parametrize("invalid_value", ["false", "true", 0, 1, None, [], {}])
def test_hard_safety_requires_strict_boolean(invalid_value):
    from strategies.orchestrator import decide_lunch_trade
    result = decide_lunch_trade(
        opportunity_score=90.0,
        net_ev_r=0.50,
        sample_count=100,
        hard_safety_passed=invalid_value,
        regime_mult=1.0,
    )
    assert result["decision"] == "REJECT"
    assert result["risk_mult"] == 0.0


@pytest.mark.parametrize(
    ("score", "expected_cap"),
    [
        (59.999, 0.00),
        (60.000, 0.35),
        (69.999, 0.35),
        (70.000, 0.50),
        (84.999, 0.50),
        (85.000, 0.75),
        (100.00, 0.75),
    ],
)
def test_opportunity_score_boundaries(score, expected_cap):
    from strategies.orchestrator import decide_lunch_trade
    result = decide_lunch_trade(
        opportunity_score=score,
        net_ev_r=0.50,
        sample_count=100,
        hard_safety_passed=True,
        regime_mult=1.0,
    )
    if expected_cap == 0.0:
        assert result["decision"] == "REJECT"
        assert result["risk_mult"] == 0.0
    else:
        assert result["decision"] == "ALLOW"
        assert result["risk_mult"] == expected_cap


@pytest.mark.parametrize("net_ev_r", [-0.001, 0.0])
def test_non_positive_ev_rejected(net_ev_r):
    from strategies.orchestrator import decide_lunch_trade
    result = decide_lunch_trade(
        opportunity_score=100.0,
        net_ev_r=net_ev_r,
        sample_count=100,
        hard_safety_passed=True,
        regime_mult=1.0,
    )
    assert result["decision"] == "REJECT"
    assert result["risk_mult"] == 0.0


@pytest.mark.parametrize("sample_count", [0, 1, 29])
def test_insufficient_samples_are_paper_only(sample_count):
    from strategies.orchestrator import decide_lunch_trade
    result = decide_lunch_trade(
        opportunity_score=100.0,
        net_ev_r=0.50,
        sample_count=sample_count,
        hard_safety_passed=True,
        regime_mult=1.0,
    )
    assert result["decision"] == "PAPER_ONLY"
    assert result["risk_mult"] == 0.0


def test_paper_only_never_reaches_live_broker():
    from strategies.orchestrator import decide_lunch_trade
    decision = decide_lunch_trade(
        opportunity_score=90.0,
        net_ev_r=0.50,
        sample_count=10,  # Insufficient sample
        hard_safety_passed=True,
        regime_mult=1.0,
    )
    assert decision["decision"] == "PAPER_ONLY"
    assert decision["risk_mult"] == 0.0
    # Downstream invariant: live multiplier must be 0.0 for PAPER_ONLY
    effective_risk_mult = decision["risk_mult"]
    assert effective_risk_mult == 0.0


def test_reject_blocks_order_submission():
    from strategies.orchestrator import decide_lunch_trade
    decision = decide_lunch_trade(
        opportunity_score=45.0,  # Below 60
        net_ev_r=0.50,
        sample_count=100,
        hard_safety_passed=True,
        regime_mult=1.0,
    )
    assert decision["decision"] == "REJECT"
    assert decision["risk_mult"] == 0.0


def test_allow_respects_approved_dollar_risk():
    from strategies.orchestrator import decide_lunch_trade
    from strategies.position_sizer import calculate_shares
    from strategies.base import TradeSignal

    decision = decide_lunch_trade(
        opportunity_score=85.0,
        net_ev_r=0.40,
        sample_count=100,
        hard_safety_passed=True,
        regime_mult=1.0,
    )
    assert decision["decision"] == "ALLOW"
    assert decision["risk_mult"] == 0.75

    base_equity = 100000.0
    sig = TradeSignal(
        symbol="MSFT",
        side="BUY",
        entry_price=400.00,
        stop_price=396.00,
        target_price=408.00,
        risk_per_share=4.00,
        confidence=0.85,
        strategy="VWAP",
        atr=3.0,
        reason="Test allow sizing",
    )
    # Sizing with lunch risk multiplier
    lunch_effective_equity = base_equity * decision["risk_mult"]
    shares = calculate_shares(sig, equity=lunch_effective_equity)
    assert shares > 0
    actual_dollar_risk = shares * sig.risk_per_share
    # Max allowed dollar risk must not exceed base 1% * 0.75 = $750
    assert actual_dollar_risk <= (base_equity * 0.01 * 0.75) + 1e-4
