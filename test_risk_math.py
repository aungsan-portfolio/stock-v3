import math
import unittest

import risk_math as rm


class RiskMathTests(unittest.TestCase):
    def test_long_risk_reward(self):
        self.assertAlmostEqual(rm.risk_per_share(30, 29.25), 0.75)
        self.assertAlmostEqual(rm.reward_per_share(30, 32.25), 2.25)
        self.assertAlmostEqual(rm.risk_reward_ratio(30, 29.25, 32.25), 3.0)

    def test_required_target(self):
        self.assertAlmostEqual(rm.required_target(30, 29.25, required_r=3), 32.25)

    def test_shares_for_risk(self):
        self.assertEqual(rm.account_risk_dollars(100_000, 0.001), 100)
        self.assertEqual(rm.shares_for_risk(100_000, 0.001, 100, 98), 50)

    def test_max_trade_value_cap(self):
        self.assertEqual(
            rm.shares_for_risk(100_000, 0.01, 100, 98, max_trade_value=500),
            5,
        )

    def test_max_position_pct_cap(self):
        self.assertEqual(
            rm.shares_for_risk(100_000, 0.01, 100, 98, max_position_pct=0.005),
            5,
        )

    def test_r_multiple(self):
        self.assertAlmostEqual(rm.planned_risk_dollars(10, 100, 98), 20)
        self.assertAlmostEqual(rm.realized_pnl(100, 106, 10), 60)
        self.assertAlmostEqual(rm.r_multiple(100, 98, 106, shares=10), 3.0)

    def test_expectancy_win_rate_profit_factor(self):
        r_list = [3, -1, 2, -1, 0.5]
        self.assertAlmostEqual(rm.expectancy_r(r_list), 0.7)
        self.assertAlmostEqual(rm.win_rate(r_list), 0.6)
        self.assertAlmostEqual(rm.profit_factor_from_r(r_list), 5.5 / 2)
        self.assertEqual(rm.profit_factor_from_r([1, 2]), float("inf"))
        self.assertEqual(rm.profit_factor_from_r([-1, -2]), 0.0)
        self.assertEqual(rm.expectancy_r([math.inf, math.nan]), 0.0)

    def test_atr_stop_price_long_and_short(self):
        self.assertAlmostEqual(rm.atr_stop_price(100, 2, 1.5), 97)
        self.assertAlmostEqual(rm.atr_stop_price(100, 2, 1.5, side="SHORT"), 103)

    def test_invalid_inputs_raise(self):
        invalid_calls = [
            lambda: rm.risk_per_share(0, 29),
            lambda: rm.risk_per_share(30, 31),
            lambda: rm.reward_per_share(30, 29),
            lambda: rm.account_risk_dollars(100_000, 0),
            lambda: rm.account_risk_dollars(100_000, 1.1),
            lambda: rm.shares_for_risk(100_000, 0.01, 100, 98, max_position_pct=2),
            lambda: rm.r_multiple(100, 98, 106, shares=0),
            lambda: rm.atr_stop_price(100, 0),
            lambda: rm.risk_per_share(30, 29, side="SIDEWAYS"),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_risk_math_additional_coverage(self):
        # 1. TypeError/ValueError inside _require_positive
        with self.assertRaises(ValueError):
            rm.risk_per_share("invalid", 29)
        with self.assertRaises(ValueError):
            rm.risk_per_share(math.inf, 29)
        with self.assertRaises(ValueError):
            rm.risk_per_share(math.nan, 29)

        # 2. TypeError/ValueError inside _require_non_negative_int
        with self.assertRaises(ValueError):
            rm.planned_risk_dollars("invalid", 100, 98)
        with self.assertRaises(ValueError):
            rm.planned_risk_dollars(-5, 100, 98)

        # 3. Stop price check negative/invalid in risk_per_share
        with self.assertRaises(ValueError):
            rm.risk_per_share(30, "invalid")

        # 4. Reward per share negative/invalid
        with self.assertRaises(ValueError):
            rm.reward_per_share(30, "invalid")

        # 5. shares_for_risk invalid max_position_pct
        with self.assertRaises(ValueError):
            rm.shares_for_risk(100_000, 0.01, 100, 98, max_position_pct="invalid")

        # 6. trailing_stop_price invalid parameter raises
        # Call with both atr=None and trailing_pct=None
        with self.assertRaises(ValueError):
            rm.trailing_stop_price(100, atr=None, trailing_pct=None)
        
        # LONG mode with trailing_pct > 1
        with self.assertRaises(ValueError):
            rm.trailing_stop_price(100, trailing_pct=1.5, side="LONG")

        # SHORT mode with trailing_pct without lowest_price and atr
        with self.assertRaises(ValueError):
            rm.trailing_stop_price(100, trailing_pct=0.05, side="SHORT", lowest_price=None, atr=None)

        # SHORT mode with trailing_pct > 1
        with self.assertRaises(ValueError):
            rm.trailing_stop_price(100, trailing_pct=1.5, side="SHORT", lowest_price=90)

        # 7. Happy paths of trailing_stop_price
        # LONG ATR
        self.assertAlmostEqual(rm.trailing_stop_price(100, atr=2, atr_multiple=1.5, side="LONG"), 97.0)
        self.assertAlmostEqual(rm.trailing_stop_price(100, atr=2, atr_multiple=1.5, side="LONG", highest_price=105), 102.0)
        self.assertAlmostEqual(rm.trailing_stop_price(100, atr=2, atr_multiple=1.5, side="LONG", highest_price_since_entry=105), 102.0)
        # LONG percent trailing
        self.assertAlmostEqual(rm.trailing_stop_price(100, trailing_pct=0.05, side="LONG"), 95.0)

        # SHORT ATR
        self.assertAlmostEqual(rm.trailing_stop_price(100, atr=2, atr_multiple=1.5, side="SHORT"), 103.0)
        self.assertAlmostEqual(rm.trailing_stop_price(100, atr=2, atr_multiple=1.5, side="SHORT", lowest_price=95), 98.0)
        self.assertAlmostEqual(rm.trailing_stop_price(100, atr=2, atr_multiple=1.5, side="SHORT", lowest_price_since_entry=95), 98.0)
        # SHORT percent trailing
        self.assertAlmostEqual(rm.trailing_stop_price(100, trailing_pct=0.05, side="SHORT", lowest_price=90), 94.5)

        # 8. R multiple exception on zero planned risk
        from unittest import mock
        with mock.patch("risk_math.planned_risk_dollars", return_value=0.0):
            with self.assertRaises(ValueError):
                rm.r_multiple(100, 98, 106, shares=10)

        # 9. _finite_values handling non-numeric / infinite values in iterable
        self.assertAlmostEqual(rm.win_rate([3, -1, "invalid", math.inf]), 0.5)


if __name__ == "__main__":
    unittest.main()

