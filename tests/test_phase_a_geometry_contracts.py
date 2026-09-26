"""
tests/test_phase_a_geometry_contracts.py -- Specialized Test 2: Geometry Invariant Fuzz & Contract Test.

Validates that:
1. Long stop is strictly < (price - tick_size), short stop is strictly > (price + tick_size).
2. Fuzzing 10,000 randomized execution geometries preserves ordering invariants:
   BUY: 0 < stop < price < target
   SELL: 0 < target < price < stop
3. Pathological values (NaN, Inf, <= 0 tick_size, inverted stops) are cleanly rejected
   without unhandled crashes or inverted bracket order generation.
"""
import math
import random
import unittest
import numpy as np
import pandas as pd

import config
from strategies.base import TradeSignal, StrategyName
from strategies.execution_geometry import (
    calculate_directional_stop_and_validate,
    floor_to_tick_float,
    ceil_to_tick_float,
    calculate_minimum_stop_distance,
    StopGeometry,
)
from strategies.order_manager import validate_geometry
from strategies.vwap_strategy import VWAPBounceStrategy


def _build_valid_base_df(n_bars: int = 35, base_price: float = 100.0) -> pd.DataFrame:
    """Builds a DataFrame that triggers the VWAP touch & hold precondition."""
    times = pd.date_range("2026-07-20 09:30:00", periods=n_bars, freq="1min", tz="US/Eastern")
    df = pd.DataFrame({
        "open": [base_price] * n_bars,
        "high": [base_price + 0.50] * n_bars,
        "low": [base_price - 0.50] * n_bars,
        "close": [base_price + 0.10] * n_bars,
        "volume": [100.0] * n_bars,
        "atr": [0.50] * n_bars,
        "vwap": [base_price] * n_bars,
        "vwap_lower": [base_price - 1.00] * n_bars,
        "vwap_upper": [base_price + 2.00] * n_bars,
        "rsi": [45.0] * n_bars,
        "volume_ma": [100.0] * n_bars,
    }, index=times)

    reclaim_bars = getattr(config, "VWAP_RECLAIM_BARS", 3)
    df.iloc[-reclaim_bars - 1, df.columns.get_loc("low")] = base_price  # touches VWAP
    for i in range(-reclaim_bars, 0):
        df.iloc[i, df.columns.get_loc("close")] = base_price + 0.20 + 0.05 * (i + reclaim_bars)
    df.iloc[-1, df.columns.get_loc("close")] = base_price + 0.40
    return df


class TestPhaseAGeometryContracts(unittest.TestCase):

    def setUp(self):
        self._orig_min_rr = getattr(config, "MIN_RR_RATIO", 1.0)
        config.MIN_RR_RATIO = 1.0

    def tearDown(self):
        config.MIN_RR_RATIO = self._orig_min_rr

    def test_fuzz_10000_directional_stop_invariants(self):
        """
        Fuzzes 10,000 randomized tuples across BUY and SELL directions.
        Asserts directional invariants:
          BUY:  stop <= price - tick_size AND price - stop >= minimum_distance
          SELL: stop >= price + tick_size AND stop - price >= minimum_distance
        """
        rng = random.Random(42)

        for i in range(10000):
            side = rng.choice(["long", "short"])
            price = round(rng.uniform(2.0, 500.0), 2)
            raw_risk = round(rng.uniform(0.05, 10.0), 2)
            atr = round(rng.uniform(0.05, 5.0), 2)
            spread_floor = round(rng.uniform(0.0, 0.20), 4)
            tick_size = rng.choice([0.01, 0.05, 0.10])
            precision = 2

            # Ensure price > raw_risk + 1.0 so long stop > 0
            if price <= raw_risk + 1.0:
                price = raw_risk + 5.0

            stop_geom = calculate_directional_stop_and_validate(
                side=side,
                price=price,
                raw_risk=raw_risk,
                atr=atr,
                spread_floor=spread_floor,
                tick_size=tick_size,
                precision=precision,
            )

            stop = stop_geom.stop_price
            min_dist = stop_geom.minimum_distance
            final_dist = stop_geom.final_distance

            # Invariant 1 & 2: Structural and noise floor bounds
            if side == "long":
                self.assertLess(stop, price, f"Iter {i}: BUY stop {stop} must be < price {price}")
                self.assertLessEqual(
                    stop,
                    price - tick_size + 1e-6,
                    f"Iter {i}: BUY stop {stop} must be <= price - tick_size ({price - tick_size})",
                )
                self.assertGreaterEqual(
                    final_dist,
                    min_dist - 1e-6,
                    f"Iter {i}: Final distance {final_dist} must be >= min distance {min_dist}",
                )
                # Verify order manager validate_geometry accepts this stop with a 2R target
                target = price + (final_dist * 2.0)
                valid, err, _, _, _ = validate_geometry("BUY", price, stop, target)
                self.assertTrue(valid, f"Iter {i}: validate_geometry failed for BUY: {err}")
            else:
                self.assertGreater(stop, price, f"Iter {i}: SELL stop {stop} must be > price {price}")
                self.assertGreaterEqual(
                    stop,
                    price + tick_size - 1e-6,
                    f"Iter {i}: SELL stop {stop} must be >= price + tick_size ({price + tick_size})",
                )
                self.assertGreaterEqual(
                    final_dist,
                    min_dist - 1e-6,
                    f"Iter {i}: Final distance {final_dist} must be >= min distance {min_dist}",
                )
                target = max(0.01, price - (final_dist * 2.0))
                if target < price:
                    valid, err, _, _, _ = validate_geometry("SELL", price, stop, target)
                    self.assertTrue(valid, f"Iter {i}: validate_geometry failed for SELL: {err}")

    def test_pathological_inputs_rejected_cleanly(self):
        """
        Tests that NaN, Inf, zero/negative prices, tick_size <= 0, and inverted ordering
        raise ValueError or return valid=False without unhandled crashes.
        """
        pathological_floats = [float("nan"), float("inf"), float("-inf"), 0.0, -1.5]

        # 1. calculate_directional_stop_and_validate with pathological values
        for bad_val in pathological_floats:
            with self.assertRaises(Exception):
                calculate_directional_stop_and_validate(
                    side="long",
                    price=bad_val if not math.isnan(bad_val) else 100.0,
                    raw_risk=bad_val,
                    atr=0.5,
                    tick_size=0.01,
                )

            with self.assertRaises(Exception):
                calculate_directional_stop_and_validate(
                    side="long",
                    price=bad_val,
                    raw_risk=0.5,
                    atr=0.5,
                    tick_size=0.01,
                )

        # 2. Non-positive tick size
        for bad_tick in [0.0, -0.01, float("nan")]:
            with self.assertRaises(Exception):
                calculate_directional_stop_and_validate(
                    side="long",
                    price=100.0,
                    raw_risk=1.0,
                    atr=0.5,
                    tick_size=bad_tick,
                )

        # 3. Inverted orders in validate_geometry
        # BUY: stop >= entry
        valid, err, _, _, _ = validate_geometry("BUY", entry=100.0, stop=100.0, target=105.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

        valid, err, _, _, _ = validate_geometry("BUY", entry=100.0, stop=102.0, target=105.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

        # BUY: target <= entry
        valid, err, _, _, _ = validate_geometry("BUY", entry=100.0, stop=98.0, target=100.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

        # SELL: stop <= entry
        valid, err, _, _, _ = validate_geometry("SELL", entry=100.0, stop=100.0, target=95.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

        valid, err, _, _, _ = validate_geometry("SELL", entry=100.0, stop=98.0, target=95.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

        # SELL: target >= entry
        valid, err, _, _, _ = validate_geometry("SELL", entry=100.0, stop=102.0, target=100.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

    def test_vwap_structural_stop_inversion_rejected(self):
        """
        Verifies that when pullback_low >= price (or >= price - tick_size),
        VWAPBounceStrategy rejects the setup cleanly (returns None) without producing
        negative raw_risk or an inverted bracket order.
        """
        strat = VWAPBounceStrategy()
        df = _build_valid_base_df(35, 100.0)
        price = float(df.iloc[-1]["close"])  # 100.40

        # Simulate pathological scenario: pullback_low was recorded at 100.45 (ABOVE current price)
        # and vwap_lower is also above price - tick_size
        reclaim_bars = getattr(config, "VWAP_RECLAIM_BARS", 3)
        df.iloc[-reclaim_bars - 2 : -1, df.columns.get_loc("low")] = price + 0.05
        df["vwap_lower"] = price + 0.10

        # Should reject cleanly with None (valid_supports is empty)
        sig = strat.evaluate("AAPL", df, spread_dollars=0.01)
        self.assertIsNone(sig, "Strategy must reject when all supports are >= price - tick_size")


if __name__ == "__main__":
    unittest.main()
