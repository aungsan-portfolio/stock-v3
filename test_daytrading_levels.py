import unittest

import pandas as pd

import daytrading_levels as dl


class DayTradingLevelTests(unittest.TestCase):
    def test_gap_pct(self):
        self.assertAlmostEqual(dl.gap_pct(102, 100), 0.02)

    def test_daily_relative_volume(self):
        self.assertAlmostEqual(dl.daily_relative_volume(1_500_000, 1_000_000), 1.5)

    def test_pivot_points(self):
        levels = dl.pivot_points(110, 100, 105)
        self.assertAlmostEqual(levels["pp"], 105)
        self.assertAlmostEqual(levels["r1"], 110)
        self.assertAlmostEqual(levels["s1"], 100)
        self.assertAlmostEqual(levels["r2"], 115)
        self.assertAlmostEqual(levels["s2"], 95)
        self.assertAlmostEqual(levels["r3"], 120)
        self.assertAlmostEqual(levels["s3"], 90)

    def test_nearest_level_up_down(self):
        levels = {"s1": 95, "pp": 100, "r1": 105, "r2": 115}
        self.assertEqual(dl.nearest_level(101, levels, direction="UP"), 105)
        self.assertEqual(dl.nearest_level(101, levels, direction="DOWN"), 100)
        self.assertIsNone(dl.nearest_level(120, levels, direction="UP"))

    def test_level_zone(self):
        self.assertEqual(dl.level_zone(100, 100, buffer_pct=0.01, min_buffer=0.05), (99, 101))
        self.assertEqual(dl.level_zone(10, 10, buffer_pct=0.001, min_buffer=0.05), (9.95, 10.05))

    def test_atr_dollars_from_ohlc(self):
        df = pd.DataFrame(
            {
                "High": [11, 12, 13, 14, 15, 16],
                "Low": [9, 10, 11, 12, 13, 14],
                "Close": [10, 11, 12, 13, 14, 15],
            }
        )
        self.assertAlmostEqual(dl.atr_dollars_from_ohlc(df, period=3), 2.0)
        self.assertIsNone(dl.atr_dollars_from_ohlc(df.head(3), period=3))
        self.assertIsNone(dl.atr_dollars_from_ohlc(pd.DataFrame({"Close": [1, 2, 3]})))

    def test_invalid_inputs_raise(self):
        invalid_calls = [
            lambda: dl.gap_pct(100, 0),
            lambda: dl.daily_relative_volume(100, 0),
            lambda: dl.daily_relative_volume(-1, 100),
            lambda: dl.pivot_points(90, 100, 95),
            lambda: dl.nearest_level(100, [90], direction="SIDEWAYS"),
            lambda: dl.level_zone(100, 100, buffer_pct=0),
            lambda: dl.atr_dollars_from_ohlc(pd.DataFrame(), period=0),
        ]
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
