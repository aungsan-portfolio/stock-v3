"""
test_minervini_gate.py — offline tests for the M2 Minervini Stage-2 entry gate.

This covers ONLY the bridge-side hard gate that the M2 milestone added to
ibkr_bridge.py:

    IBKRBridge._minervini_stage2_blocks(symbol, signal) -> bool
    IBKRBridge._new_entries_blocked(...) -> (blocked, reason)   # reason "stage2_filter"

The gate is a NEW-BUY-only, default-OFF, fail-open overlay. It must:
  * BLOCK a NEW BUY only when BOTH switches are on AND the action is exactly
    "BUY" AND the M0 verdict has stage2_ok == False (reason "stage2_filter"),
  * NEVER block a SELL / close / flatten (the close path never even reaches the
    gate, and the guard is `action == "BUY"`, not `action != "HOLD"`),
  * be a complete no-op when either switch is off,
  * FAIL OPEN — a fetch_ohlcv / evaluate_entry exception must never block a BUY.

Fully offline + deterministic. The M0 evaluator (minervini.evaluate_entry) and
the data layer (data_manager.fetch_ohlcv) are mocked so these tests isolate the
GATE's decision logic, not the (separately tested) evaluator. One class drives
the REAL evaluator on synthetic frames to prove the gate reads stage2_ok
correctly end to end.

Run with either:
    python -m unittest test_minervini_gate -v
    python test_minervini_gate.py
"""
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import config
import data_manager
import minervini
import risk_engine
import risk_state

# Reuse the fully offline fake-ib_insync harness (FakeIB + bridge factory).
from test_ibkr_fill_flow import (
    _BridgeBase, make_bridge, fill_market_only, placed_orders,
)


# A unique sentinel so a test can assert the gate passed the fetched frame
# straight into the evaluator without caring about its contents.
_SENTINEL_DF = object()


def _signal(action="BUY", symbol="AAPL", confidence=0.7, price=150.0):
    """Minimal Signal stand-in; the gate reads only .action (and .price upstream)."""
    return types.SimpleNamespace(
        symbol=symbol, action=action, confidence=confidence, price=price,
    )


def _verdict(stage2_ok):
    """Minimal MinerviniVerdict stand-in — the gate reads only .stage2_ok."""
    return types.SimpleNamespace(stage2_ok=stage2_ok)


# ── Synthetic OHLCV builders (mirror test_minervini / test_minervini_coach) ───
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


# ── 1. _minervini_stage2_blocks: the boolean helper in isolation ─────────────
class TestStage2HelperGate(_BridgeBase):
    """Direct tests on the helper, with the M0 evaluator mocked."""

    def _enable(self, overlay=True, stage2=True):
        mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", overlay).start()
        mock.patch.object(config, "MINERVINI_STAGE2_BLOCK_ENABLED", stage2).start()

    def _mock_eval(self, stage2_ok):
        """Patch the data + evaluator layer; return (fetch_mock, eval_mock)."""
        fetch = mock.patch.object(
            data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        ev = mock.patch.object(
            minervini, "evaluate_entry", return_value=_verdict(stage2_ok)).start()
        return fetch, ev

    # Req 1 — blocks a NEW BUY when every condition holds.
    def test_blocks_buy_when_all_conditions_hold(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        fetch, ev = self._mock_eval(stage2_ok=False)
        self.assertTrue(br._minervini_stage2_blocks("AAPL", _signal("BUY")))
        # It actually consulted the evaluator with the fetched frame.
        fetch.assert_called_once()
        ev.assert_called_once_with(_SENTINEL_DF)

    # Complement — a passing Stage-2 verdict must NOT block (no over-blocking).
    def test_passing_verdict_does_not_block(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        self._mock_eval(stage2_ok=True)
        self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("BUY")))

    # Req 3 — overlay master switch off => complete no-op, never touches data.
    def test_overlay_disabled_is_noop(self):
        br = make_bridge()
        self._enable(overlay=False, stage2=True)
        fetch, ev = self._mock_eval(stage2_ok=False)  # would block IF consulted
        self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("BUY")))
        fetch.assert_not_called()
        ev.assert_not_called()

    # Req 4 — stage2 sub-switch off => no-op even with overlay on.
    def test_stage2_subswitch_disabled_is_noop(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=False)
        fetch, ev = self._mock_eval(stage2_ok=False)  # would block IF consulted
        self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("BUY")))
        fetch.assert_not_called()
        ev.assert_not_called()

    # Req 2 / Req 6 — a SELL is never filtered and never reaches the data layer.
    def test_sell_action_never_blocks(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        fetch, ev = self._mock_eval(stage2_ok=False)  # failing verdict on purpose
        self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("SELL")))
        fetch.assert_not_called()
        ev.assert_not_called()

    # Req 6 — the guard is `action == "BUY"` (normalized), NOT `action != "HOLD"`.
    # Every non-BUY action (incl. SELL/HOLD/blank/None) must fail open even with a
    # failing verdict and both switches on; only BUY/"buy" may block.
    def test_guard_uses_exact_buy_equality_not_not_hold(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        self._mock_eval(stage2_ok=False)
        for action in ("SELL", "sell", "HOLD", "hold", "", "FOO", None):
            self.assertFalse(
                br._minervini_stage2_blocks("AAPL", _signal(action)),
                f"non-BUY action {action!r} must never trigger stage2_filter",
            )
        # ... while a BUY (case-insensitively) does block under the same verdict.
        self.assertTrue(br._minervini_stage2_blocks("AAPL", _signal("BUY")))
        self.assertTrue(br._minervini_stage2_blocks("AAPL", _signal(" buy ")))

    # Req 2 fail-open robustness — a missing signal must never block.
    def test_missing_signal_never_blocks(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        self._mock_eval(stage2_ok=False)
        self.assertFalse(br._minervini_stage2_blocks("AAPL", None))

    # Req 5a — a fetch_ohlcv failure fails OPEN (does not block the BUY).
    def test_fetch_failure_fails_open(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        mock.patch.object(
            data_manager, "fetch_ohlcv", side_effect=RuntimeError("no data")).start()
        ev = mock.patch.object(minervini, "evaluate_entry").start()
        self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("BUY")))
        ev.assert_not_called()  # never reached the evaluator

    # Req 5b — an evaluate_entry failure fails OPEN (does not block the BUY).
    def test_evaluate_failure_fails_open(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        mock.patch.object(
            data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(
            minervini, "evaluate_entry", side_effect=ValueError("bad frame")).start()
        self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("BUY")))

    # Fail-open contract — a malformed verdict OBJECT with no stage2_ok attribute
    # must not crash or block (defaults to "ok" -> do not block).
    def test_malformed_verdict_missing_stage2_ok_fails_open(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        mock.patch.object(
            data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(
            minervini, "evaluate_entry", return_value=object()).start()  # no stage2_ok
        self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("BUY")))

    # Fail-open contract — an explicit stage2_ok=None must not block.
    def test_verdict_stage2_ok_none_fails_open(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        mock.patch.object(
            data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(
            minervini, "evaluate_entry", return_value=_verdict(None)).start()
        self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("BUY")))


# ── 2. _new_entries_blocked: the reason string wiring ────────────────────────
class TestNewEntriesBlockedReason(_BridgeBase):
    """The Stage-2 result is surfaced as reason 'stage2_filter', and only for BUY.

    The earlier gates (halt flag, connection health, market hours, price agreement,
    daily loss, drawdown, exposure) are neutralized so each test isolates the
    Stage-2 decision. order_price is passed as None so the price-agreement check is
    skipped.
    """

    def _enable(self, overlay=True, stage2=True):
        mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", overlay).start()
        mock.patch.object(config, "MINERVINI_STAGE2_BLOCK_ENABLED", stage2).start()

    def _permit_earlier_gates(self):
        # Connection health defaults healthy on a fresh bridge; neutralize the rest.
        mock.patch.object(config, "MARKET_HOURS_GATE_ENABLED", False).start()
        mock.patch.object(risk_state, "daily_loss_blocked", return_value=False).start()
        mock.patch.object(risk_engine, "drawdown_halt_breached", return_value=False).start()
        mock.patch.object(risk_engine, "symbol_exposure_exceeded", return_value=False).start()

    def _mock_eval(self, stage2_ok):
        mock.patch.object(data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        mock.patch.object(minervini, "evaluate_entry",
                          return_value=_verdict(stage2_ok)).start()

    # Req 1 — a failing Stage-2 verdict surfaces exactly ("stage2_filter").
    def test_block_reason_is_stage2_filter(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        self._permit_earlier_gates()
        self._mock_eval(stage2_ok=False)
        blocked, reason = br._new_entries_blocked("AAPL", 1000.0, _signal("BUY"), None)
        self.assertTrue(blocked)
        self.assertEqual(reason, "stage2_filter")

    # A passing verdict leaves the gate open.
    def test_passing_verdict_not_blocked(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        self._permit_earlier_gates()
        self._mock_eval(stage2_ok=True)
        blocked, reason = br._new_entries_blocked("AAPL", 1000.0, _signal("BUY"), None)
        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    # Req 6 (integration layer) — a SELL entry that DOES pass through the gate
    # (e.g. a short-open) must never be refused with "stage2_filter", proving the
    # guard keys on action == "BUY", not action != "HOLD".
    def test_sell_entry_is_not_filtered_by_stage2(self):
        br = make_bridge()
        self._enable(overlay=True, stage2=True)
        self._permit_earlier_gates()
        self._mock_eval(stage2_ok=False)  # failing verdict on purpose
        blocked, reason = br._new_entries_blocked("AAPL", 1000.0, _signal("SELL"), None)
        self.assertFalse(blocked)
        self.assertNotEqual(reason, "stage2_filter")


# ── 3. SELL close regression, end to end through execute_signal ──────────────
class TestSellCloseRegression(_BridgeBase):
    """The required regression: a SELL that closes a long must still execute even
    with the Stage-2 overlay enabled and FAILING. The close path never routes
    through the entry gate, so the overlay must never even be consulted."""

    def test_long_close_executes_with_stage2_failing(self):
        br = make_bridge(on_wait=fill_market_only)
        risk_state.snapshot_start_of_day_equity(100000.0)
        br.ib.account_values = [("NetLiquidation", "USD", 100000.0)]
        br.get_price = lambda *a, **k: 100.0                       # type: ignore
        br.get_cash = lambda: 100000.0                            # type: ignore
        br.has_working_order = lambda symbol, action=None: False  # type: ignore
        br.get_position = lambda symbol: 100.0                    # long -> SELL closes it
        mock.patch.object(risk_state, "can_open_more", return_value=True).start()
        # Market-hours gate is irrelevant to closes; disabled defensively anyway.
        mock.patch.object(config, "MARKET_HOURS_GATE_ENABLED", False).start()

        # Overlay ON and failing -> would block a BUY, must NOT block this close.
        mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", True).start()
        mock.patch.object(config, "MINERVINI_STAGE2_BLOCK_ENABLED", True).start()
        fetch = mock.patch.object(
            data_manager, "fetch_ohlcv", return_value=_SENTINEL_DF).start()
        ev = mock.patch.object(
            minervini, "evaluate_entry", return_value=_verdict(False)).start()

        ok = br.execute_signal(
            types.SimpleNamespace(symbol="AAPL", action="SELL", confidence=0.7))

        self.assertTrue(ok, "a long close must execute even under a failing Stage-2 overlay")
        self.assertTrue(
            any(str(getattr(o, "orderType", "")) == "MKT" for o in placed_orders(br.ib)),
            "expected a market close order for the SELL",
        )
        # The close path never consults the entry overlay.
        fetch.assert_not_called()
        ev.assert_not_called()


# ── 4. Real M0 evaluator integration (no mock of evaluate_entry) ─────────────
class TestRealVerdictIntegration(_BridgeBase):
    """Patch ONLY the data layer; drive the REAL minervini.evaluate_entry on
    synthetic frames to prove the gate reads stage2_ok end to end (default args,
    no benchmark — matching how the gate calls the evaluator)."""

    def setUp(self):
        super().setUp()
        mock.patch.object(config, "MINERVINI_OVERLAY_ENABLED", True).start()
        mock.patch.object(config, "MINERVINI_STAGE2_BLOCK_ENABLED", True).start()

    def test_real_downtrend_blocks_buy(self):
        br = make_bridge()
        with mock.patch.object(data_manager, "fetch_ohlcv", return_value=downtrend()):
            self.assertTrue(br._minervini_stage2_blocks("AAPL", _signal("BUY")))

    def test_real_uptrend_does_not_block_buy(self):
        br = make_bridge()
        with mock.patch.object(data_manager, "fetch_ohlcv", return_value=uptrend()):
            self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("BUY")))

    def test_real_downtrend_does_not_block_sell(self):
        br = make_bridge()
        with mock.patch.object(data_manager, "fetch_ohlcv", return_value=downtrend()):
            self.assertFalse(br._minervini_stage2_blocks("AAPL", _signal("SELL")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
