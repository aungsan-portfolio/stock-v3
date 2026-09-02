"""
test_target_reachability_guard.py - Unit tests for Target Reachability Sanity Guard.

Verifies:
1. Valid intraday signals with multi-bar targets pass even when target_dist >> 1-minute ATR.
2. Runaway / glitchy signals with R:R > 5.0 are rejected.
3. Signals exceeding 3.5x Daily ATR are rejected when daily_atr is present.
4. Signals within 3.5x Daily ATR pass when daily_atr is present.
5. Missing / None / zero daily_atr is gracefully skipped without errors.
"""
import pytest
from strategies.base import TradeSignal
from strategies.constants import StrategyName
from strategies.order_manager import _validate_signal_prices


def _make_signal(
    symbol="WMT",
    side="BUY",
    entry_price=100.0,
    stop_price=99.0,
    target_price=101.5,
    atr_1m=0.03,
    metadata=None,
) -> TradeSignal:
    risk = abs(entry_price - stop_price)
    return TradeSignal(
        symbol=symbol,
        strategy=StrategyName.MOMENTUM_SCALP,
        side=side,
        confidence=0.85,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        atr=atr_1m,
        risk_per_share=risk,
        reason="Test setup",
        metadata=metadata if metadata is not None else {},
    )


def test_valid_intraday_signal_passes_despite_small_1m_atr():
    """Valid 1.5R target should pass even if target distance is 20x 1-minute ATR."""
    # entry=100, stop=99, risk=1.0, target=101.5 (target_dist=1.50)
    # 1m ATR = 0.05 -> target_dist / 1m_atr = 30.0x
    sig = _make_signal(
        entry_price=100.0,
        stop_price=99.0,
        target_price=101.5,
        atr_1m=0.05,
    )
    result = _validate_signal_prices(sig)
    assert result == "", f"Expected valid signal to pass, got: {result}"


def test_wmt_reconstructed_geometry_passes():
    """Exact WMT signal geometry from Aug 22 (risk=0.28, target=0.42, 1m_atr=0.027) should pass."""
    # SHORT signal: entry=102.91, stop=103.19 (risk=0.28), target=102.49 (target_dist=0.42, R:R=1.5)
    sig = _make_signal(
        symbol="WMT",
        side="SELL",
        entry_price=102.91,
        stop_price=103.19,
        target_price=102.49,
        atr_1m=0.027,
    )
    result = _validate_signal_prices(sig)
    assert result == "", f"Expected WMT geometry to pass, got: {result}"


def test_runaway_glitch_target_fails_max_rr_guard():
    """Runaway glitch signal with excessive R:R ratio (> 5.0) should be vetoed."""
    # entry=100, stop=99 (risk=1.0), target=120 (target_dist=20.0, R:R=20.0)
    sig = _make_signal(
        entry_price=100.0,
        stop_price=99.0,
        target_price=120.0,
        atr_1m=0.5,
    )
    result = _validate_signal_prices(sig)
    assert "Unrealistic target R:R ratio" in result
    assert "20.00" in result


def test_daily_atr_reachability_guard_rejects_excessive_target():
    """Target exceeding 3.5x Daily ATR should be rejected when daily_atr is present."""
    # entry=100, stop=98 (risk=2.0), target=108.0 (target_dist=8.0, R:R=4.0 <= 5.0)
    # Daily ATR = 2.0 -> target_dist / daily_atr = 4.0x (> 3.5x limit)
    sig = _make_signal(
        entry_price=100.0,
        stop_price=98.0,
        target_price=108.0,
        atr_1m=0.1,
        metadata={"daily_atr": 2.0},
    )
    result = _validate_signal_prices(sig)
    assert "Unrealistic target distance $8.00 (4.0x Daily ATR > 3.5x limit)" in result


def test_daily_atr_reachability_guard_passes_realistic_target():
    """Target within 3.5x Daily ATR should pass when daily_atr is present."""
    # entry=100, stop=98 (risk=2.0), target=104.0 (target_dist=4.0, R:R=2.0)
    # Daily ATR = 2.0 -> target_dist / daily_atr = 2.0x (<= 3.5x limit)
    sig = _make_signal(
        entry_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        atr_1m=0.1,
        metadata={"daily_atr": 2.0},
    )
    result = _validate_signal_prices(sig)
    assert result == ""


def test_daily_atr_absent_skips_gracefully():
    """When daily_atr is absent or None or <= 0, the daily ATR guard should gracefully skip."""
    # Cases where daily_atr is missing, None, 0, or negative
    for meta in [{}, {"other": 123}, {"daily_atr": None}, {"daily_atr": 0.0}, {"daily_atr": -1.0}]:
        sig = _make_signal(
            entry_price=100.0,
            stop_price=99.0,
            target_price=102.0,
            atr_1m=0.02,
            metadata=meta,
        )
        result = _validate_signal_prices(sig)
        assert result == "", f"Expected metadata {meta} to skip gracefully, got: {result}"
