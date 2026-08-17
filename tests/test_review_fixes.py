import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import joblib
import numpy as np
import pandas as pd

import ai_engine
import config
import data_manager
import lstm_engine
import main


class _FakeTorchModel:
    def __init__(self, probability: float):
        self.logit = math.log(probability / (1.0 - probability))

    def eval(self):
        return self

    def __call__(self, _seq):
        return lstm_engine.torch.tensor(self.logit)


class ReviewFixesTests(unittest.TestCase):
    def test_build_features_uses_injected_settings(self):
        df = pd.DataFrame(
            {
                "Open": np.linspace(100, 140, 320),
                "High": np.linspace(101, 141, 320),
                "Low": np.linspace(99, 139, 320),
                "Close": np.linspace(100, 140, 320),
                "Volume": np.full(320, 1_000_000.0),
            },
            index=pd.bdate_range("2020-01-01", periods=320),
        )

        cfg = config.Settings()
        cfg.USE_CANDLESTICK_FEATURES = True
        cfg.USE_MICRO_FEATURES = True

        with mock.patch.object(config, "USE_CANDLESTICK_FEATURES", False), mock.patch.object(
            config, "USE_MICRO_FEATURES", False
        ):
            base_cols = data_manager.get_feature_columns()
            injected_cols = data_manager.get_feature_columns(cfg)
            injected_feat = data_manager.build_features(df, cfg=cfg)

        self.assertNotIn("doji", base_cols)
        self.assertIn("doji", injected_cols)
        self.assertIn("close_loc", injected_cols)
        self.assertIn("doji", injected_feat.columns)
        self.assertIn("close_loc", injected_feat.columns)

    def test_rf_load_skips_incompatible_feature_widths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = config.Settings()
            cfg.MODELS_DIR = tmpdir
            cfg.ML_MODELS_FILE = tmpdir / "rf_models.joblib"
            cfg.USE_CANDLESTICK_FEATURES = True

            joblib.dump(
                {
                    "OLD": SimpleNamespace(n_features_in_=14),
                    "NEW": SimpleNamespace(n_features_in_=len(data_manager.get_feature_columns(cfg))),
                },
                cfg.ML_MODELS_FILE,
            )

            engine = ai_engine.StockRFEngine(settings=cfg)
            self.assertTrue(engine.load())
            self.assertEqual(set(engine.models.keys()), {"NEW"})

    def test_lstm_load_uses_cfg_scaler_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            cfg = config.Settings()
            cfg.LSTM_CKPT_FILE = tmpdir / "custom_lstm.pt"
            cfg.LSTM_CKPT_FILE.write_bytes(b"not-used")

            global_scalers = tmpdir / "global.scalers.npz"
            global_scalers.write_bytes(b"unused")

            engine = lstm_engine.StockLSTMEngine(settings=cfg)
            with mock.patch.object(lstm_engine, "SCALERS_FILE", global_scalers):
                self.assertFalse(engine.load())
                self.assertEqual(engine.scalers_file, cfg.LSTM_CKPT_FILE.with_suffix(".scalers.npz"))

    @unittest.skipIf(lstm_engine.torch is None, "torch not installed")
    def test_meta_labeling_suppresses_score_instead_of_reinflating(self):
        cfg = config.Settings()
        cfg.LSTM_META_ENSEMBLE = True
        cfg.LSTM_WINDOW = 4
        engine = lstm_engine.StockLSTMEngine(settings=cfg)
        feature_cols = engine.feature_cols
        engine.models = {"SPY": _FakeTorchModel(0.9)}
        engine.meta_models = {"SPY": _FakeTorchModel(0.3)}
        engine.scalers = {
            "SPY": (
                np.zeros(len(feature_cols), dtype=np.float32),
                np.ones(len(feature_cols), dtype=np.float32),
            )
        }

        feat = pd.DataFrame(
            np.ones((cfg.LSTM_WINDOW + 2, len(feature_cols)), dtype=np.float32),
            columns=feature_cols,
        )
        with mock.patch.object(lstm_engine, "build_features", return_value=feat):
            score = engine.predict("SPY", df=pd.DataFrame())

        self.assertAlmostEqual(score, 0.27, places=6)

    def test_paper_coach_passes_coach_true(self):
        candidate = SimpleNamespace(symbol="AAPL", action="BUY", confidence=0.9)
        bridge = mock.MagicMock()
        bridge.get_cash.return_value = 100_000.0
        bridge.ib.positions.return_value = []
        bridge.working_order_symbols.return_value = set()
        bridge.has_working_order.return_value = False
        bridge.execute_signal.return_value = True

        args = SimpleNamespace(confirm=True, chart_checked=True)
        with mock.patch.object(main, "_print_coach_intro"), \
             mock.patch.object(main, "Predictor") as predictor_cls, \
             mock.patch.object(main, "_connect_bridge", return_value=(bridge, 0)), \
             mock.patch.object(main, "select_coach_candidates", return_value=[candidate]), \
             mock.patch.object(main, "build_trade_lesson", return_value={}), \
             mock.patch.object(main, "build_trade_preview", return_value={"tradeable": True}), \
             mock.patch.object(main, "print_trade_lesson"), \
             mock.patch.object(main, "print_trade_preview"), \
             mock.patch.object(main, "write_trade_note"), \
             mock.patch.object(main, "_run_paper_startup_checks", return_value={}), \
             mock.patch.object(main, "_print_startup_reconciliation"):
            predictor_cls.return_value.predict_all.return_value = [candidate]
            rc = main.cmd_paper_coach(args)

        self.assertEqual(rc, 0)
        bridge.execute_signal.assert_called_once_with(candidate, coach=True)
        bridge.disconnect.assert_called_once()

    def test_daily_coach_execute_passes_coach_true(self):
        candidate = SimpleNamespace(symbol="AAPL", action="BUY", confidence=0.9)
        bridge = mock.MagicMock()
        bridge.get_cash.return_value = 100_000.0
        bridge.ib.positions.return_value = []
        bridge.working_order_symbols.return_value = set()
        bridge.execute_signal.return_value = True

        fake_risk_state = SimpleNamespace(can_open_more=lambda: True)
        with mock.patch.dict("sys.modules", {"risk_state": fake_risk_state}), \
             mock.patch.object(main, "assert_paper_trading_only"), \
             mock.patch.object(main, "_connect_bridge", return_value=(bridge, 0)), \
             mock.patch.object(main, "_run_paper_startup_checks", return_value={}), \
             mock.patch.object(main, "_print_startup_reconciliation"), \
             mock.patch.object(main, "build_trade_preview", return_value={"tradeable": True}), \
             mock.patch.object(main, "assess_chart_status", return_value=(main.CHART_OK, "ok")), \
             mock.patch.object(main, "evaluate_daily_candidate", return_value={"accepted": True, "symbol": "AAPL"}), \
             mock.patch.object(main, "print_daily_candidate"), \
             mock.patch.object(main, "build_trade_lesson", return_value={}), \
             mock.patch.object(main, "write_trade_note"):
            rc = main._daily_coach_execute([candidate], 1)

        self.assertEqual(rc, 0)
        bridge.execute_signal.assert_called_once_with(candidate, coach=True)
        bridge.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
