"""
shadow_logger.py — Paired Inference Logger for Option C Shadow Models.

Logs paired incumbent vs. shadow model inferences to data/shadow_predictions.jsonl
following the Option C schema (reports/OPTION_C_SHADOW_PROMOTION_SPEC_20260927.md).

Safety Invariants:
1. Append-only logging.
2. Anti-lookahead: realized_outcome fields remain null at prediction time.
3. Fail-safe: Any shadow inference failure is caught and logged at debug level,
   ensuring ZERO interruption or side-effects on live incumbent execution.
"""

import os
import json
import logging
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

import config
from ai_engine import StockRFEngine
from lstm_engine import StockLSTMEngine
from errors import ModelNotAvailableError

logger = logging.getLogger(__name__)

SHADOW_LOG_FILE = config.DATA_DIR / "shadow_predictions.jsonl"
SHADOW_DIR = config.MODELS_DIR / "shadow_v20260927"


class ShadowInferenceLogger:
    def __init__(self, shadow_dir: Optional[Path] = None):
        self.shadow_dir = shadow_dir or SHADOW_DIR
        self.enabled = False
        self.shadow_rf = None
        self.shadow_lstm = None
        self._init_shadow_engines()

    def _init_shadow_engines(self) -> None:
        rf_path = self.shadow_dir / "rf_models.joblib"
        lstm_path = self.shadow_dir / "lstm_checkpoint.pt"

        if not rf_path.exists() or not lstm_path.exists():
            logger.debug("[SHADOW] Shadow model files not found at %s. Shadow logging inactive.", self.shadow_dir)
            self.enabled = False
            return

        try:
            # Construct shadow settings namespace pointing to shadow directory
            rf_cfg = types.SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if not k.startswith("_")})
            rf_cfg.ML_MODELS_FILE = rf_path

            lstm_cfg = types.SimpleNamespace(**{k: getattr(config, k) for k in dir(config) if not k.startswith("_")})
            lstm_cfg.LSTM_CKPT_FILE = lstm_path

            self.shadow_rf = StockRFEngine(settings=rf_cfg)
            self.shadow_lstm = StockLSTMEngine(settings=lstm_cfg)

            rf_ok = self.shadow_rf.load()
            lstm_ok = self.shadow_lstm.load()

            if rf_ok and lstm_ok:
                self.enabled = True
                logger.info("[SHADOW] Shadow models v20260927 loaded successfully (n=%d). Paired logging active.", len(self.shadow_rf.models))
            else:
                logger.warning("[SHADOW] Failed to load one or more shadow models (rf_ok=%s, lstm_ok=%s).", rf_ok, lstm_ok)
                self.enabled = False
        except Exception as e:
            logger.warning("[SHADOW] Exception initializing shadow engines: %s", e)
            self.enabled = False

    def log_prediction_pair(
        self,
        symbol: str,
        bar_date: str,
        bar_close: float,
        incumbent_signal: Any,
        df: pd.DataFrame,
    ) -> None:
        if not self.enabled:
            return

        try:
            # 1. Shadow RF Inference
            shadow_rf_score = None
            try:
                shadow_rf_score = float(self.shadow_rf.predict(symbol, df))
            except Exception as e_rf:
                logger.debug("[SHADOW] Shadow RF predict failed for %s: %s", symbol, e_rf)

            # 2. Shadow LSTM Inference
            shadow_lstm_score = None
            try:
                shadow_lstm_score = float(self.shadow_lstm.predict(symbol, df))
            except Exception as e_lstm:
                logger.debug("[SHADOW] Shadow LSTM predict failed for %s: %s", symbol, e_lstm)

            # 3. Shadow Ensemble Blending (50/50 ML blend matching default config weights)
            shadow_conf = None
            shadow_action = "HOLD"
            available_scores = [s for s in (shadow_rf_score, shadow_lstm_score) if s is not None]
            if available_scores:
                shadow_conf = float(np.mean(available_scores))
                if shadow_conf >= config.BUY_THRESHOLD:
                    shadow_action = "BUY"
                elif shadow_conf <= config.SELL_THRESHOLD:
                    shadow_action = "SELL"
                else:
                    shadow_action = "HOLD"

            now_iso = datetime.now(timezone.utc).isoformat()
            pred_id = f"pred_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{symbol}_{int(datetime.now(timezone.utc).timestamp())}"

            record = {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "record_type": "SHADOW_INFERENCE_PAIR",
                "prediction_id": pred_id,
                "timestamp": now_iso,
                "symbol": symbol,
                "bar_date": str(bar_date),
                "bar_close": round(float(bar_close), 4),
                "incumbent": {
                    "vintage": "2026-08-29",
                    "rf_prob": getattr(incumbent_signal, "rf_score", None),
                    "lstm_prob": getattr(incumbent_signal, "lstm_score", None),
                    "ensemble_conf": getattr(incumbent_signal, "confidence", None),
                    "action": getattr(incumbent_signal, "action", "HOLD"),
                },
                "shadow_v20260927": {
                    "vintage": "2026-09-27",
                    "rf_prob": round(shadow_rf_score, 4) if shadow_rf_score is not None else None,
                    "lstm_prob": round(shadow_lstm_score, 4) if shadow_lstm_score is not None else None,
                    "ensemble_conf": round(shadow_conf, 4) if shadow_conf is not None else None,
                    "action": shadow_action,
                },
                "realized_outcome": {
                    "evaluated": False,
                    "eval_timestamp": None,
                    "horizon_bars": int(config.ML_HORIZON),
                    "future_bar_date": None,
                    "future_close": None,
                    "realized_return": None,
                    "binary_label": None,
                },
            }

            # Ensure parent data directory exists
            SHADOW_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SHADOW_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            logger.debug("[SHADOW] Logged inference pair for %s: inc_conf=%s shd_conf=%s", symbol, record["incumbent"]["ensemble_conf"], record["shadow_v20260927"]["ensemble_conf"])
        except Exception as e:
            logger.debug("[SHADOW] Error logging shadow prediction for %s: %s", symbol, e)


# Global singleton logger
_LOGGER_INSTANCE: Optional[ShadowInferenceLogger] = None


def get_shadow_logger() -> ShadowInferenceLogger:
    global _LOGGER_INSTANCE
    if _LOGGER_INSTANCE is None:
        _LOGGER_INSTANCE = ShadowInferenceLogger()
    return _LOGGER_INSTANCE
