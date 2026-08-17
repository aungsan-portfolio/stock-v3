"""
test_model_gate.py - Phase 1 tests: truthful validation + signal safety.

Pure stdlib `unittest`, fully offline and deterministic (no IBKR, no network,
no real training). Exercises the Phase-1 building blocks from
reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md:

  * model_metrics.evaluate_gate  - the fail-closed performance/staleness gate
  * model_metrics.save_rf_metrics / save_lstm_metrics - per-symbol persistence
    that MERGES RF and LSTM entries and never raises during training
  * predictor.Predictor.predict_all - a pre-gate BUY is forced to HOLD when the
    gate blocks the symbol (and passes through when the gate is satisfied)
  * backtest.check_signal_parity - backtest/live signal-construction parity
  * main._collect_readiness_gates - the Phase-1 P1 scorecard gates PASS while the
    bot still reports NOT READY overall
  * main.cmd_signal_parity - the read-only `signal-parity` command

Every test patches config.MODEL_METRICS_FILE to a temp file so the real
models/model_metrics.json is never read or written.

Run with:
    python -m unittest test_model_gate -v
    python test_model_gate.py
"""
import contextlib
import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config
import model_metrics
from errors import ModelNotAvailableError
from test_logic import make_ohlcv  # deterministic synthetic OHLCV (offline)


# A fixed "today" so staleness math never depends on the wall clock.
NOW = dt.date(2026, 6, 16)


class _GateBase(unittest.TestCase):
    """Patch config thresholds + a temp metrics file so each test is hermetic."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model_metrics.json"
        # Pin every config knob the gate reads so tests don't drift if defaults
        # are later tuned.
        for name, value in {
            "MODEL_METRICS_FILE": self.path,
            "MODEL_GATE_ENABLED": True,
            "MODEL_MIN_RF_ACCURACY": 0.50,
            "MODEL_MIN_RF_F1": 0.0,
            "MODEL_MAX_AGE_DAYS": 30,
        }.items():
            mock.patch.object(config, name, value).start()
        self.addCleanup(mock.patch.stopall)

    # -- helpers ----------------------------------------------------------
    def write_doc(self, doc: dict):
        """Write a raw metrics document straight to the temp file."""
        self.path.write_text(json.dumps(doc), encoding="utf-8")

    def save_rf(self, symbol, acc, f1=0.6, trained_at="2026-06-16"):
        model_metrics.save_rf_metrics(
            {symbol: {"test_acc": acc, "test_f1": f1, "oob_score": acc, "n_samples": 500}},
            trained_at=trained_at,
        )

    def save_lstm(self, symbol, val_loss=0.1, trained_at="2026-06-16"):
        model_metrics.save_lstm_metrics(
            {symbol: {"best_val_loss": val_loss, "n_train_seq": 400}},
            trained_at=trained_at,
        )


# ── 1. evaluate_gate decision table ──────────────────────────────────
class TestEvaluateGate(_GateBase):
    def test_disabled_gate_allows_everything(self):
        mock.patch.object(config, "MODEL_GATE_ENABLED", False).start()
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertTrue(res.ok)
        self.assertEqual(res.status, "disabled")

    def test_no_metrics_is_missing_and_blocks(self):
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "missing")

    def test_missing_train_date_is_missing(self):
        # Metrics exist but carry no usable train date -> fail-closed missing.
        self.write_doc({"rf": {"AAPL": {"accuracy": 0.8, "trained_at": None}}})
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "missing")

    def test_fresh_good_rf_passes(self):
        self.save_rf("AAPL", acc=0.70, trained_at="2026-06-16")
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertTrue(res.ok, res.reason)
        self.assertEqual(res.status, "ok")

    def test_stale_model_blocks(self):
        self.save_rf("AAPL", acc=0.90, trained_at="2026-01-01")  # ~166d old
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "stale")
        self.assertIn("stale", res.reason)

    def test_accuracy_below_floor_blocks(self):
        self.save_rf("AAPL", acc=0.40, trained_at="2026-06-16")
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "below_threshold")

    def test_accuracy_exactly_at_floor_passes(self):
        # Check is strict `< floor`, so 0.50 == floor is allowed.
        self.save_rf("AAPL", acc=0.50, trained_at="2026-06-16")
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertTrue(res.ok, res.reason)

    def test_nan_accuracy_is_missing(self):
        self.save_rf("AAPL", acc=float("nan"), trained_at="2026-06-16")
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "missing")

    def test_f1_floor_enforced_only_when_enabled(self):
        mock.patch.object(config, "MODEL_MIN_RF_F1", 0.50).start()
        # Good accuracy but weak F1 -> blocked when the F1 floor is on.
        self.save_rf("AAPL", acc=0.80, f1=0.20, trained_at="2026-06-16")
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "below_threshold")
        # Strong F1 clears it.
        self.save_rf("MSFT", acc=0.80, f1=0.75, trained_at="2026-06-16")
        self.assertTrue(model_metrics.evaluate_gate("MSFT", now=NOW).ok)

    def test_lstm_only_fresh_passes(self):
        # No RF entry: only the staleness check applies (no RF accuracy floor).
        self.save_lstm("AAPL", trained_at="2026-06-16")
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertTrue(res.ok, res.reason)

    def test_lstm_only_stale_blocks(self):
        self.save_lstm("AAPL", trained_at="2026-01-01")
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertFalse(res.ok)
        self.assertEqual(res.status, "stale")

    def test_freshest_date_wins_for_staleness(self):
        # RF is old, but a fresh LSTM retrain keeps the symbol tradeable.
        self.save_rf("AAPL", acc=0.80, trained_at="2026-01-01")
        self.save_lstm("AAPL", trained_at="2026-06-16")
        res = model_metrics.evaluate_gate("AAPL", now=NOW)
        self.assertTrue(res.ok, res.reason)

    def test_symbol_is_case_insensitive(self):
        self.save_rf("aapl", acc=0.70, trained_at="2026-06-16")
        self.assertTrue(model_metrics.evaluate_gate("AAPL", now=NOW).ok)
        self.assertTrue(model_metrics.evaluate_gate("aApL", now=NOW).ok)


# ── 2. persistence: save/merge/load ──────────────────────────────────
class TestPersistence(_GateBase):
    def test_rf_and_lstm_entries_coexist(self):
        self.save_rf("AAPL", acc=0.7)
        self.save_lstm("AAPL")
        data = model_metrics.load_all()
        self.assertIn("AAPL", data["rf"])
        self.assertIn("AAPL", data["lstm"])
        self.assertAlmostEqual(data["rf"]["AAPL"]["accuracy"], 0.7)

    def test_save_uppercases_symbol(self):
        self.save_rf("aapl", acc=0.7)
        self.assertIn("AAPL", model_metrics.load_all()["rf"])

    def test_metrics_for_returns_none_for_unknown(self):
        m = model_metrics.metrics_for("NOPE")
        self.assertIsNone(m["rf"])
        self.assertIsNone(m["lstm"])

    def test_save_never_raises_on_bad_input(self):
        # Persistence must never break a training run.
        model_metrics.save_rf_metrics(None)
        model_metrics.save_rf_metrics({"X": None})  # type: ignore[arg-type]
        model_metrics.save_lstm_metrics({})

    def test_load_all_empty_when_file_missing(self):
        self.assertFalse(self.path.exists())
        self.assertEqual(model_metrics.load_all(), {})

    def test_load_all_empty_on_corrupt_file(self):
        self.path.write_text("{ this is not json", encoding="utf-8")
        self.assertEqual(model_metrics.load_all(), {})  # fail-closed


# ── 3. predict_all integration: the gate forces HOLD ─────────────────
class TestPredictGateIntegration(_GateBase):
    """A genuine pre-gate BUY (RF=0.9, LSTM missing -> ensemble 0.67) must be
    overridden to HOLD unless the symbol's metrics satisfy the gate."""

    def _run_predict(self):
        from predictor import Predictor
        with mock.patch("predictor.fetch_ohlcv", return_value=make_ohlcv()):
            p = Predictor()
            mock.patch.object(p.rf, "predict", return_value=0.9).start()
            mock.patch.object(
                p.lstm, "predict", side_effect=ModelNotAvailableError("no lstm")
            ).start()
            return p.predict_all(symbols=["TEST"])

    def test_buy_blocked_when_no_metrics(self):
        sigs = self._run_predict()
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].action, "HOLD")
        self.assertIn("BLOCKED[missing]", sigs[0].reason)

    def test_buy_passes_when_metrics_fresh_and_good(self):
        # Trained today with strong accuracy -> the gate lets the BUY through.
        self.save_rf("TEST", acc=0.80, trained_at=NOW.isoformat())
        with mock.patch.object(model_metrics, "_dt") as mdt:
            mdt.date.today.return_value = NOW
            mdt.date.fromisoformat = dt.date.fromisoformat
            sigs = self._run_predict()
        self.assertEqual(sigs[0].action, "BUY")
        self.assertNotIn("BLOCKED", sigs[0].reason)

    def test_buy_passes_when_gate_disabled(self):
        mock.patch.object(config, "MODEL_GATE_ENABLED", False).start()
        sigs = self._run_predict()
        self.assertEqual(sigs[0].action, "BUY")
        self.assertNotIn("BLOCKED", sigs[0].reason)

    def tearDown(self):
        mock.patch.stopall()
        super().tearDown()


# ── 4. capability flags (Phase-1 implemented => True) ────────────────
class TestCapabilityFlags(unittest.TestCase):
    def test_performance_and_staleness_flags_true(self):
        self.assertTrue(model_metrics.SUPPORTS_MODEL_PERFORMANCE_GATE)
        self.assertTrue(model_metrics.SUPPORTS_MODEL_STALENESS_GATE)


# ── 5. backtest/live signal-construction parity ──────────────────────
class TestSignalParity(unittest.TestCase):
    def test_parity_holds_and_helpers_are_shared(self):
        import backtest
        p = backtest.check_signal_parity()
        self.assertTrue(p["parity_ok"])
        self.assertTrue(p["shared_weighted_blend"])
        self.assertTrue(p["shared_action_fn"])
        self.assertTrue(p["shared_position_rule"])
        self.assertTrue(p["numeric_blend_match"])
        # thresholds/weights are surfaced for the scorecard + report.
        self.assertIsInstance(p["buy_threshold"], float)
        self.assertIsInstance(p["sell_threshold"], float)


# ── 6. Phase-1 scorecard gates ───────────────────────────────────────
class TestScorecardP1Gates(unittest.TestCase):
    def test_p1_gates_pass_but_overall_not_ready(self):
        import main
        gates = main._collect_readiness_gates(None)
        p1_auto = [g for g in gates if g["phase"] == "P1" and g["status"] in ("PASS", "FAIL")]
        self.assertTrue(p1_auto)
        # Every automated P1 gate is implemented + tested -> all PASS.
        self.assertTrue(all(g["status"] == "PASS" for g in p1_auto),
                        [g["title"] for g in p1_auto if g["status"] != "PASS"])
        # The 3 implemented Phase-1 gates are present by title.
        titles = " ".join(g["title"] for g in p1_auto)
        self.assertIn("performance gate", titles)
        self.assertIn("staleness gate", titles)
        self.assertIn("parity", titles)
        # Later phases still fail -> the bot must report NOT READY overall.
        self.assertTrue(any(g["status"] == "FAIL" for g in gates),
                        "later-phase gates must still fail in Phase 1")


# ── 7. signal-parity command (read-only) ─────────────────────────────
class TestSignalParityCommand(_GateBase):
    def test_cmd_signal_parity_exit_zero_and_prints(self):
        import main
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = main.cmd_signal_parity(mock.Mock())
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("Signal Parity", out)
        self.assertIn("Model Performance / Staleness Gate", out)
        # With an empty temp metrics file the warning branch must show.
        self.assertIn("No persisted metrics", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
