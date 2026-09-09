"""
tests/test_direction_aware_geometry.py -- Unit tests for direction-aware chase guard,
side-specific executable quote selection, and side-neutral geometry validation.
"""
import unittest

from strategies.order_manager import unfavorable_chase, validate_geometry


class TestDirectionAwareGeometry(unittest.TestCase):

    def test_unfavorable_chase_buy(self):
        # Unfavorable movement (paying more)
        self.assertAlmostEqual(unfavorable_chase("BUY", 100.0, 101.5), 1.5)
        # Favorable improvement (paying less) -> must be 0.0
        self.assertAlmostEqual(unfavorable_chase("BUY", 100.0, 99.0), 0.0)
        # Identical price
        self.assertAlmostEqual(unfavorable_chase("BUY", 100.0, 100.0), 0.0)

    def test_unfavorable_chase_sell(self):
        # Unfavorable movement (selling lower)
        self.assertAlmostEqual(unfavorable_chase("SELL", 100.0, 98.5), 1.5)
        # Favorable improvement (selling higher) -> must be 0.0
        self.assertAlmostEqual(unfavorable_chase("SELL", 100.0, 102.0), 0.0)
        # Identical price
        self.assertAlmostEqual(unfavorable_chase("SELL", 100.0, 100.0), 0.0)

    def test_validate_geometry_buy_valid(self):
        valid, err, risk, reward, rr = validate_geometry("BUY", entry=100.0, stop=98.0, target=105.0)
        self.assertTrue(valid)
        self.assertEqual(err, "")
        self.assertAlmostEqual(risk, 2.0)
        self.assertAlmostEqual(reward, 5.0)
        self.assertAlmostEqual(rr, 2.5)

    def test_validate_geometry_buy_inverted_ordering(self):
        # Stop above entry
        valid, err, _, _, _ = validate_geometry("BUY", entry=100.0, stop=101.0, target=105.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

        # Target below entry
        valid, err, _, _, _ = validate_geometry("BUY", entry=100.0, stop=98.0, target=95.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

    def test_validate_geometry_sell_valid(self):
        valid, err, risk, reward, rr = validate_geometry("SELL", entry=100.0, stop=102.0, target=95.0)
        self.assertTrue(valid)
        self.assertEqual(err, "")
        self.assertAlmostEqual(risk, 2.0)
        self.assertAlmostEqual(reward, 5.0)
        self.assertAlmostEqual(rr, 2.5)

    def test_validate_geometry_sell_inverted_ordering(self):
        # Stop below entry for a short
        valid, err, _, _, _ = validate_geometry("SELL", entry=100.0, stop=98.0, target=95.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

        # Target above entry for a short
        valid, err, _, _, _ = validate_geometry("SELL", entry=100.0, stop=102.0, target=105.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")

    def test_validate_geometry_non_positive_risk_or_reward(self):
        # Stop equals entry
        valid, err, _, _, _ = validate_geometry("BUY", entry=100.0, stop=100.0, target=105.0)
        self.assertFalse(valid)
        self.assertEqual(err, "invalid_price_ordering")


if __name__ == "__main__":
    unittest.main()
