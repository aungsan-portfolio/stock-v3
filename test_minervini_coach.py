"""
test_minervini_coach.py — offline tests for the M1 read-only Minervini coach view.

Pure stdlib `unittest`, fully offline (synthetic OHLCV injected via the `ohlcv`
arg — no network, no IBKR, no trained models). Run with either:

    python -m unittest test_minervini_coach -v
    python test_minervini_coach.py

M1 is explanation-only: the coach view never places, sizes, or blocks an order.
These tests assert it is a complete no-op unless BOTH MINERVINI_OVERLAY_ENABLED
and MINERVINI_COACH_ENABLED are True, and that it never raises into the caller.
"""
import io
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd

import config
import trade_coach


# ── Synthetic OHLCV builders (mirror test_minervini.py) ──────────────────────
def _frame(closes, highs=None, lows=None, vols=None) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    highs = closes * 1.01 if highs is None else np.asarray(highs, dtype=float)
    lows = closes * 0.99 if lows is None else np.asarray(lows, dtype=float)
    vols = np.full(n, 1_000_000.0) if vols is None else np.asarray(vols, dtype=float)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(
        {"Open": closes, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=idx,
    )


def uptrend(n: int = 300, g: float = 0.004) -> pd.DataFrame:
    return _frame(100.0 * np.power(1.0 + g, np.arange(n)))


def downtrend(n: int = 300, g: float = 0.003) -> pd.DataFrame:
    return _frame(150.0 * np.power(1.0 - g, np.arange(n)))


def fake_signal(symbol="AAPL", action="BUY", confidence=0.80, price=150.0):
    return types.SimpleNamespace(
        symbol=symbol, action=action, confidence=confidence,
        rf_score=0.8, lstm_score=0.7, tech_score=0.6, price=price, reason="",
    )


def _enable_coach():
    """Context-managerless helper: patch both switches True (caller stops them)."""
    p1 = mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", True)
    p2 = mock.patch.object(config, "MINERVINI_COACH_ENABLED", True)
    p1.start(); p2.start()
    return (p1, p2)


# ── Gating: off by default, requires BOTH switches ───────────────────────────
class TestGating(unittest.TestCase):
    def test_disabled_by_default_returns_none(self):
        self.assertIsNone(trade_coach.build_minervini_view(fake_signal(), ohlcv=uptrend()))

    def test_overlay_on_but_coach_off_returns_none(self):
        with mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", True), \
             mock.patch.object(config, "MINERVINI_COACH_ENABLED", False):
            self.assertIsNone(trade_coach.build_minervini_view(fake_signal(), ohlcv=uptrend()))

    def test_coach_on_but_overlay_off_returns_none(self):
        with mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", False), \
             mock.patch.object(config, "MINERVINI_COACH_ENABLED", True):
            self.assertIsNone(trade_coach.build_minervini_view(fake_signal(), ohlcv=uptrend()))


# ── Enabled: real M0 evaluation on injected synthetic data ───────────────────
class TestEnabledView(unittest.TestCase):
    def setUp(self):
        self._patches = _enable_coach()
        self.addCleanup(mock.patch.stopall)

    def test_clean_uptrend_is_stage2_with_advisory_stop(self):
        view = trade_coach.build_minervini_view(fake_signal(), ohlcv=uptrend())
        self.assertTrue(view["available"])
        self.assertTrue(view["stage2_ok"])
        self.assertIn("Passes the Stage-2 trend template", view["reason_text"])
        self.assertIsNotNone(view["advisory_stop_price"])
        self.assertLess(view["advisory_stop_price"], view["pivot_low"])

    def test_downtrend_is_not_stage2_and_lists_reasons(self):
        view = trade_coach.build_minervini_view(fake_signal(), ohlcv=downtrend())
        self.assertTrue(view["available"])
        self.assertFalse(view["stage2_ok"])
        self.assertIn("ma_order_fail", view["reasons"])
        self.assertIn("Not a Stage-2 setup", view["reason_text"])

    def test_fetch_failure_degrades_to_unavailable_not_raise(self):
        # No ohlcv passed => lazy fetch path; force it to raise.
        with mock.patch("data_manager.fetch_ohlcv", side_effect=RuntimeError("no net")):
            view = trade_coach.build_minervini_view(fake_signal())
        self.assertFalse(view["available"])
        self.assertIn("Could not evaluate", view["reason_text"])


# ── Integration into lesson / preview dicts ──────────────────────────────────
class TestLessonPreviewIntegration(unittest.TestCase):
    def test_lesson_minervini_none_when_disabled(self):
        lesson = trade_coach.build_trade_lesson(fake_signal())
        self.assertIn("minervini", lesson)
        self.assertIsNone(lesson["minervini"])

    def test_preview_minervini_none_when_disabled(self):
        preview = trade_coach.build_trade_preview(fake_signal(), cash=100_000.0)
        self.assertTrue(preview["tradeable"])
        self.assertIsNone(preview["minervini"])
        # Sizing/stop math must be unchanged by the (disabled) overlay.
        self.assertGreater(preview["quantity"], 0)
        self.assertIsNotNone(preview["stop_price"])

    def test_lesson_carries_view_when_enabled(self):
        self.addCleanup(mock.patch.stopall)
        _enable_coach()
        with mock.patch("data_manager.fetch_ohlcv", return_value=uptrend()):
            lesson = trade_coach.build_trade_lesson(fake_signal())
        self.assertIsNotNone(lesson["minervini"])
        self.assertTrue(lesson["minervini"]["available"])
        self.assertTrue(lesson["minervini"]["stage2_ok"])

    def test_preview_carries_view_when_enabled_without_changing_sizing(self):
        self.addCleanup(mock.patch.stopall)
        _enable_coach()
        base = trade_coach.build_trade_preview(fake_signal(), cash=100_000.0)  # disabled-equivalent qty
        with mock.patch("data_manager.fetch_ohlcv", return_value=uptrend()):
            preview = trade_coach.build_trade_preview(fake_signal(), cash=100_000.0)
        self.assertIsNotNone(preview["minervini"])
        # The overlay does not resize the position in M1.
        self.assertEqual(preview["quantity"], base["quantity"])
        self.assertEqual(preview["stop_price"], base["stop_price"])


# ── Printers must never raise ────────────────────────────────────────────────
class TestPrintersSmoke(unittest.TestCase):
    def test_print_minervini_view_none_is_noop(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            trade_coach.print_minervini_view(None)
        self.assertEqual(buf.getvalue(), "")

    def test_print_lesson_and_preview_with_view(self):
        self.addCleanup(mock.patch.stopall)
        _enable_coach()
        with mock.patch("data_manager.fetch_ohlcv", return_value=uptrend()):
            lesson = trade_coach.build_trade_lesson(fake_signal())
            preview = trade_coach.build_trade_preview(fake_signal(), cash=100_000.0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            trade_coach.print_trade_lesson(lesson)
            trade_coach.print_trade_preview(preview)
        out = buf.getvalue()
        self.assertIn("Minervini / SEPA setup", out)
        self.assertIn("Advisory 1R stop", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
