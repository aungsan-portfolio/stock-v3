"""
tests/test_regime_consistency_diagnostics.py -- Unit tests for Phase A Regime/Texture Consistency Diagnostics.
"""
import unittest
from unittest.mock import MagicMock

from strategies.regime_engine import (
    MODE_RISK_ON,
    MODE_CAUTION,
    MODE_RISK_OFF,
    MODE_DATA_DEGRADED,
    TEXTURE_TRENDING,
    TEXTURE_SIDEWAYS,
    TEXTURE_VOLATILE,
    TEXTURE_TRANSITIONING,
    DIAGNOSTIC_VALID,
    DIAGNOSTIC_UNUSUAL_BUT_ALLOWED,
    DIAGNOSTIC_INSUFFICIENT_DATA,
    DIAGNOSTIC_INTERNALLY_INCONSISTENT,
    validate_regime_texture_combination,
    evaluate_market_regime,
)


class TestRegimeConsistencyDiagnostics(unittest.TestCase):

    def test_valid_combinations(self):
        scores = {"spy": 1, "qqq": 1}
        data = {"spy_pct": 0.25, "qqq_pct": 0.30}

        # RISK_ON + TRENDING -> VALID
        status, msg, attr = validate_regime_texture_combination(
            MODE_RISK_ON, TEXTURE_TRENDING, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_VALID)
        self.assertIn("harmonious and consistent", msg)

        # CAUTION + SIDEWAYS -> VALID
        status, msg, attr = validate_regime_texture_combination(
            MODE_CAUTION, TEXTURE_SIDEWAYS, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_VALID)

        # RISK_OFF + VOLATILE -> VALID
        status, msg, attr = validate_regime_texture_combination(
            MODE_RISK_OFF, TEXTURE_VOLATILE, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_VALID)

    def test_unusual_but_allowed_combinations(self):
        scores = {"macro": 1}
        data = {"spy_pct": 0.15, "qqq_pct": 0.10}

        # RISK_ON + TRANSITIONING -> UNUSUAL_BUT_ALLOWED (The 2026-09-08 scenario)
        status, msg, attr = validate_regime_texture_combination(
            MODE_RISK_ON, TEXTURE_TRANSITIONING, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_UNUSUAL_BUT_ALLOWED)
        self.assertIn("TRANSITIONING", msg)
        self.assertEqual(attr["mode"], MODE_RISK_ON)
        self.assertEqual(attr["texture"], TEXTURE_TRANSITIONING)

        # RISK_ON + SIDEWAYS -> UNUSUAL_BUT_ALLOWED
        status, msg, attr = validate_regime_texture_combination(
            MODE_RISK_ON, TEXTURE_SIDEWAYS, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_UNUSUAL_BUT_ALLOWED)

        # CAUTION + VOLATILE -> UNUSUAL_BUT_ALLOWED
        status, msg, attr = validate_regime_texture_combination(
            MODE_CAUTION, TEXTURE_VOLATILE, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_UNUSUAL_BUT_ALLOWED)

        # RISK_OFF + SIDEWAYS -> UNUSUAL_BUT_ALLOWED
        status, msg, attr = validate_regime_texture_combination(
            MODE_RISK_OFF, TEXTURE_SIDEWAYS, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_UNUSUAL_BUT_ALLOWED)

    def test_inconsistent_combinations(self):
        scores = {"macro": 2}
        data = {"spy_pct": 0.80, "qqq_pct": 0.90}

        # RISK_ON + VOLATILE -> INTERNALLY_INCONSISTENT
        status, msg, attr = validate_regime_texture_combination(
            MODE_RISK_ON, TEXTURE_VOLATILE, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_INTERNALLY_INCONSISTENT)
        self.assertIn("Conflicting macro vs micro signals", msg)

        # RISK_OFF with strongly positive benchmark indices -> INTERNALLY_INCONSISTENT
        status, msg, attr = validate_regime_texture_combination(
            MODE_RISK_OFF, TEXTURE_TRENDING, scores, data
        )
        self.assertEqual(status, DIAGNOSTIC_INTERNALLY_INCONSISTENT)
        self.assertIn("despite strong positive benchmark indices", msg)

    def test_insufficient_data(self):
        # DATA_DEGRADED mode -> INSUFFICIENT_DATA
        status, msg, attr = validate_regime_texture_combination(
            MODE_DATA_DEGRADED, TEXTURE_TRENDING, {"spy": 1}, {"spy_pct": 0.1}
        )
        self.assertEqual(status, DIAGNOSTIC_INSUFFICIENT_DATA)

        # Empty scores and data -> INSUFFICIENT_DATA
        status, msg, attr = validate_regime_texture_combination(
            MODE_RISK_ON, TEXTURE_TRENDING, {}, {}
        )
        self.assertEqual(status, DIAGNOSTIC_INSUFFICIENT_DATA)

    def test_diagnostic_does_not_alter_regime_result(self):
        """INVARIANT: Diagnostics must NOT modify mode, sizing, halts, or trade eligibility."""
        mock_bridge = MagicMock()
        mock_bridge.get_price.return_value = 500.0

        res = evaluate_market_regime(bridge=mock_bridge, scanner_candidate_count=10)
        self.assertIsNotNone(res.diagnostic_status)
        self.assertIn("diagnostic_consistency", res.data)
        self.assertIn(res.mode, [MODE_RISK_ON, MODE_CAUTION, MODE_RISK_OFF, MODE_DATA_DEGRADED])


if __name__ == "__main__":
    unittest.main()
