"""
tests/test_execution_geometry_sync.py -- Unit tests for shared execution geometry and strategy/order-manager synchronization.
"""
import pytest
import math
from decimal import Decimal
from dataclasses import FrozenInstanceError

from strategies.execution_geometry import (
    StopGeometry,
    floor_to_tick,
    ceil_to_tick,
    floor_to_tick_float,
    ceil_to_tick_float,
    calculate_minimum_stop_distance,
    calculate_directional_stop_and_validate,
)
from strategies.base import TradeSignal
from strategies.order_manager import compute_effective_noise_floor


class TestExecutionGeometrySync:
    """Test suite verifying StopGeometry contract, Decimal quantization, and cross-module sync."""

    def test_stop_geometry_dataclass_immutable(self):
        geom = StopGeometry(stop_price=100.0, minimum_distance=0.5, final_distance=0.5)
        assert geom.stop_price == 100.0
        assert geom.minimum_distance == 0.5
        assert geom.final_distance == 0.5
        with pytest.raises(FrozenInstanceError):
            geom.stop_price = 101.0

    def test_decimal_floor_to_tick_exact(self):
        # Round down away from entry for long stops
        assert floor_to_tick(Decimal("172.239"), Decimal("0.01")) == Decimal("172.23")
        assert floor_to_tick(Decimal("172.231"), Decimal("0.01")) == Decimal("172.23")
        assert floor_to_tick(Decimal("172.230"), Decimal("0.01")) == Decimal("172.23")
        assert floor_to_tick(Decimal("0.54329"), Decimal("0.0001")) == Decimal("0.5432")

    def test_decimal_ceil_to_tick_exact(self):
        # Round up away from entry for short stops
        assert ceil_to_tick(Decimal("172.231"), Decimal("0.01")) == Decimal("172.24")
        assert ceil_to_tick(Decimal("172.239"), Decimal("0.01")) == Decimal("172.24")
        assert ceil_to_tick(Decimal("172.230"), Decimal("0.01")) == Decimal("172.23")
        assert ceil_to_tick(Decimal("0.54321"), Decimal("0.0001")) == Decimal("0.5433")

    def test_absolute_eight_cent_base_floor_applies(self):
        # Even if ATR and spread are tiny, Order Manager absolute minimum applies
        sig = TradeSignal(
            symbol="PENNY",
            strategy="VWAP_BOUNCE",
            side="BUY",
            confidence=0.5,
            entry_price=5.0,
            stop_price=4.95,
            target_price=5.10,
            risk_per_share=0.05,
            reason="penny test",
            atr=0.01,
        )
        eff_floor = compute_effective_noise_floor(sig, df=None, quote=None)
        assert eff_floor >= 0.08

    def test_atr_floor_applies(self):
        floor = calculate_minimum_stop_distance(atr=1.0, spread_floor=0.0, tick_size=0.01, use_buffer=False)
        # 1.0 * 0.75 = 0.75
        assert floor == pytest.approx(0.75)

    def test_spread_floor_applies(self):
        floor = calculate_minimum_stop_distance(atr=0.1, spread_floor=0.50, tick_size=0.01, use_buffer=False)
        assert floor == pytest.approx(0.50)

    def test_tick_floor_applies(self):
        # 2 * tick_size applies when ATR and spread are near zero
        floor = calculate_minimum_stop_distance(atr=0.01, spread_floor=0.0, tick_size=0.05, use_buffer=False)
        # max(0.01 * 0.75 = 0.0075, 0.0, 0.05 * 2 = 0.10)
        assert floor == pytest.approx(0.10)

    def test_strategy_floor_equals_base_floor_multiplied_by_1_05(self):
        base = calculate_minimum_stop_distance(atr=0.886, spread_floor=0.0, tick_size=0.01, use_buffer=False)
        strategy = calculate_minimum_stop_distance(atr=0.886, spread_floor=0.0, tick_size=0.01, use_buffer=True)
        assert base == pytest.approx(0.6645)
        assert strategy == pytest.approx(0.697725)
        assert strategy == pytest.approx(base * 1.05)

    def test_raw_risk_below_buffered_floor_is_clamped(self):
        # raw_risk = 0.30 is clamped up to buffered floor (0.697725)
        geom = calculate_directional_stop_and_validate(
            side="long",
            price=200.0,
            raw_risk=0.30,
            atr=0.886,
            spread_floor=0.0,
            tick_size=0.01,
            precision=2,
        )
        assert geom.final_distance >= 0.697725
        assert geom.final_distance == pytest.approx(0.70)

    def test_raw_risk_above_buffered_floor_is_preserved(self):
        # raw_risk = 1.20 exceeds buffered floor, preserved
        geom = calculate_directional_stop_and_validate(
            side="long",
            price=200.0,
            raw_risk=1.20,
            atr=0.886,
            spread_floor=0.0,
            tick_size=0.01,
            precision=2,
        )
        assert geom.final_distance == pytest.approx(1.20)
        assert geom.stop_price == pytest.approx(198.80)

    def test_long_stop_floors_away_from_entry(self):
        geom = calculate_directional_stop_and_validate(
            side="long",
            price=100.00,
            raw_risk=0.555,
            atr=0.50,
            tick_size=0.01,
        )
        # 100 - 0.555 = 99.445 -> floor to tick = 99.44
        assert geom.stop_price == 99.44
        assert geom.final_distance == pytest.approx(0.56)

    def test_short_stop_ceils_away_from_entry(self):
        geom = calculate_directional_stop_and_validate(
            side="short",
            price=100.00,
            raw_risk=0.555,
            atr=0.50,
            tick_size=0.01,
        )
        # 100 + 0.555 = 100.555 -> ceil to tick = 100.56
        assert geom.stop_price == 100.56
        assert geom.final_distance == pytest.approx(0.56)

    def test_post_quantization_distance_remains_valid(self):
        geom = calculate_directional_stop_and_validate(
            side="long",
            price=50.0,
            raw_risk=0.50,
            atr=0.50,
            tick_size=0.01,
        )
        assert geom.final_distance >= geom.minimum_distance

    def test_invalid_side_fails_closed(self):
        with pytest.raises(ValueError, match="Unsupported position direction"):
            calculate_directional_stop_and_validate(side="invalid", price=100.0, raw_risk=0.5, atr=0.5)

    def test_sell_action_is_not_silently_treated_as_short(self):
        # 'sell' must fail closed because it could be closing a long
        with pytest.raises(ValueError, match="Unsupported position direction: 'sell'"):
            calculate_directional_stop_and_validate(side="sell", price=100.0, raw_risk=0.5, atr=0.5)
        with pytest.raises(ValueError, match="Unsupported position direction: 'buy'"):
            calculate_directional_stop_and_validate(side="buy", price=100.0, raw_risk=0.5, atr=0.5)

    def test_zero_and_negative_tick_size_rejected(self):
        with pytest.raises(ValueError, match="tick_size must be.*> 0"):
            calculate_directional_stop_and_validate(side="long", price=100.0, raw_risk=0.5, atr=0.5, tick_size=0.0)
        with pytest.raises(ValueError, match="tick_size must be.*> 0"):
            calculate_directional_stop_and_validate(side="long", price=100.0, raw_risk=0.5, atr=0.5, tick_size=-0.01)

    def test_negative_and_non_finite_inputs_rejected(self):
        with pytest.raises(ValueError, match="price must be finite and > 0"):
            calculate_directional_stop_and_validate(side="long", price=-10.0, raw_risk=0.5, atr=0.5)
        with pytest.raises(ValueError, match="raw_risk must be finite and > 0"):
            calculate_directional_stop_and_validate(side="long", price=10.0, raw_risk=float("inf"), atr=0.5)
        with pytest.raises(ValueError, match="atr must be finite and > 0"):
            calculate_directional_stop_and_validate(side="long", price=10.0, raw_risk=0.5, atr=float("nan"))
        with pytest.raises(ValueError, match="spread_floor must be finite and >= 0"):
            calculate_directional_stop_and_validate(side="long", price=10.0, raw_risk=0.5, atr=0.5, spread_floor=-0.05)

    def test_tick_size_0_05_remains_compliant(self):
        # Test non-standard tick size (e.g. 0.05)
        geom = calculate_directional_stop_and_validate(
            side="long",
            price=100.00,
            raw_risk=0.72,
            atr=0.80,
            tick_size=0.05,
            precision=2,
        )
        # 100.00 - 0.72 = 99.28 -> floor to 0.05 = 99.25
        assert geom.stop_price == pytest.approx(99.25)
        assert geom.final_distance == pytest.approx(0.75)
        assert geom.final_distance >= geom.minimum_distance

    def test_nvda_like_geometry_clears_noise_floor_veto(self):
        """
        Critical regression test for NVDA-like parameters.
        Verified exact math:
        ATR=0.886 -> base floor = 0.6645 -> buffered min = 0.697725
        Entry=218.00 -> unquantized stop = 217.302275 -> floor to tick = 217.30
        Actual distance = 0.70 >= 0.697725
        """
        geometry = calculate_directional_stop_and_validate(
            side="long",
            price=218.00,
            raw_risk=0.50,
            atr=0.886,
            spread_floor=0.00,
            tick_size=0.01,
            precision=2,
        )

        assert geometry.minimum_distance == pytest.approx(0.697725)
        assert geometry.stop_price == pytest.approx(217.30)
        assert geometry.final_distance == pytest.approx(0.70)
        assert geometry.final_distance >= geometry.minimum_distance

        base_floor = calculate_minimum_stop_distance(
            atr=0.886,
            spread_floor=0.00,
            tick_size=0.01,
            use_buffer=False,
        )

        assert base_floor == pytest.approx(0.6645)
        assert geometry.final_distance >= base_floor
