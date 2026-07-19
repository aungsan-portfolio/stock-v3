"""
alternative_models.py — Alternative ML models for Ensemble Diversification.
Provides:
1. StockXGBEngine: Gradient boosted decision trees via XGBoost (or scikit-learn fallback).
2. StockTransformerEngine: PyTorch Transformer Encoder for sequence-rich patterns.
"""
import os
import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

import config as config_module
import eval_metrics
import model_metrics
from data_manager import fetch_ohlcv, build_features, make_labels, get_feature_columns
from errors import ModelNotAvailableError

logger = logging.getLogger(__name__)
FEATURE_COLS = get_feature_columns()

# PyTorch import check for Transformer
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    class _MockModule:
        Module = object
    nn = _MockModule
    DEVICE = "cpu"


# ── 1. XGBoost / Gradient Boosting Engine ──────────────────────────────

class StockXGBEngine:
    """
    Gradient Boosting model for cross-sectional prediction.
    Falls back gracefully to scikit-learn GradientBoostingClassifier if xgboost package is missing.
    """

    def __init__(self, settings=None):
        self.cfg = settings or config_module.get_settings()
        self.feature_cols = get_feature_columns(self.cfg)
        self.models: Dict[str, any] = {}
        self._lock = threading.RLock()
        self.models_file = self.cfg.MODELS_DIR / "xgb_models.joblib"

    def _make_model(self):
        try:
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                eval_metric="logloss",
                random_state=self.cfg.RANDOM_STATE,
                n_jobs=-1
            )
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            logger.info("XGBoost not installed. Falling back to scikit-learn GradientBoostingClassifier.")
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                random_state=self.cfg.RANDOM_STATE
            )

    def train(self, symbols: Optional[list] = None) -> dict:
        symbols = symbols or self.cfg.WATCHLIST
        new_models = {}
        results = {}

        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol)
                feat = build_features(df, cfg=self.cfg)
                labels = make_labels(feat)

                valid_idx = labels.dropna().index
                X = feat.loc[valid_idx, self.feature_cols].values
                y = labels.loc[valid_idx].values.astype(int)

                if len(X) < 100 or len(np.unique(y)) < 2:
                    continue

                clf = self._make_model()
                clf.fit(X, y)

                new_models[symbol] = clf
                results[symbol] = {"trained": True}
            except Exception as e:
                logger.exception("Failed to train Gradient Boosting for %s: %s", symbol, e)

        if new_models:
            with self._lock:
                self.models.update(new_models)
            self.cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.models, self.models_file)
            logger.info("Gradient Boosting models saved to %s", self.models_file)

        return results

    def load(self) -> bool:
        if not self.models_file.exists():
            return False
        try:
            loaded = joblib.load(self.models_file)
            expected = len(self.feature_cols)
            self.models = {
                symbol: model
                for symbol, model in loaded.items()
                if getattr(model, "n_features_in_", None) == expected
            }
            return bool(self.models)
        except Exception:
            return False

    def predict(self, symbol: str, df: Optional[pd.DataFrame] = None) -> float:
        if not self.models and not self.load():
            raise ModelNotAvailableError(f"No Gradient Boosting model for {symbol}")
        if df is None:
            df = fetch_ohlcv(symbol)
        feat = build_features(df, cfg=self.cfg)
        x = feat[self.feature_cols].iloc[-1:].values

        with self._lock:
            clf = self.models.get(symbol)
        if clf is None:
            raise ModelNotAvailableError(f"No Gradient Boosting model for {symbol}")

        proba = clf.predict_proba(x)[0]
        return float(proba[1])


# ── 2. PyTorch Transformer Encoder Sequence Model ──────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) if torch else None
        if pe.shape[1] % 2 == 0:
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0::2] = torch.sin(position * div_term[:-1] if div_term is not None else 0.0)
            pe[:, 1::2] = torch.cos(position * div_term if div_term is not None else 0.0)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


BaseModule = nn.Module if nn is not None else object

class TransformerTSModel(BaseModule):
    """Self-Attention Transformer Encoder for sequence predictions."""
    def __init__(self, input_size: int, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class StockTransformerEngine:
    """
    Transformer Encoder Sequence Engine for predicting stock directions.
    Falls back gracefully if PyTorch is not available.
    """

    def __init__(self, settings=None):
        self.cfg = settings or config_module.get_settings()
        self.feature_cols = get_feature_columns(self.cfg)
        self.models: Dict[str, TransformerTSModel] = {}
        self.scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.ckpt_file = self.cfg.MODELS_DIR / "transformer_checkpoint.pt"
        self.scalers_file = self.ckpt_file.with_suffix(".scalers.npz")

    @staticmethod
    def _normalize(arr, mean=None, std=None):
        arr = np.asarray(arr, dtype=np.float32)
        if mean is None:
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std, mean, std

    def _split_and_scale(self, X_raw: np.ndarray, y_raw: np.ndarray):
        split = int(len(X_raw) * (1 - self.cfg.ML_TEST_RATIO))
        from lstm_engine import _make_sequences
        # Reuse sequence generator from lstm_engine (same sequence structures)
        X_train_norm, mean, std = self._normalize(X_raw[:split])
        X_full_norm, _, _ = self._normalize(X_raw, mean, std)

        X_tr, y_tr = _make_sequences(X_full_norm[:split], y_raw[:split], self.cfg.LSTM_WINDOW)
        X_val, y_val = _make_sequences(X_full_norm, y_raw, self.cfg.LSTM_WINDOW, min_end=split)
        return X_tr, y_tr, X_val, y_val, mean, std

    def train(self, symbols: Optional[list] = None) -> dict:
        if not torch:
            logger.warning("PyTorch not installed. Transformer engine cannot train.")
            return {}

        symbols = symbols or self.cfg.WATCHLIST
        new_models = {}
        new_scalers = {}
        results = {}

        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol)
                feat = build_features(df, cfg=self.cfg)
                labels = make_labels(feat)

                valid_idx = labels.dropna().index
                X_raw = feat.loc[valid_idx, self.feature_cols].values.astype(np.float32)
                y_raw = labels.loc[valid_idx].values.astype(np.float32)

                if len(X_raw) < self.cfg.LSTM_WINDOW + 60:
                    continue

                X_tr, y_tr, X_val, y_val, mean, std = self._split_and_scale(X_raw, y_raw)
                from lstm_engine import _subsample_train_sequences
                X_tr, y_tr = _subsample_train_sequences(X_tr, y_tr, self.cfg.ML_HORIZON)

                tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=32, shuffle=False)
                model = TransformerTSModel(len(self.feature_cols)).to(DEVICE)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                best_val = float("inf")
                best_state = None

                for epoch in range(15):  # Keep epochs low for speed
                    model.train()
                    for xb, yb in tr_dl:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        optimizer.zero_grad()
                        loss = criterion(model(xb), yb)
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(model(torch.from_numpy(X_val).to(DEVICE)), torch.from_numpy(y_val).to(DEVICE)).item()

                    if val_loss < best_val:
                        best_val = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if best_state:
                    model.load_state_dict(best_state)
                    new_models[symbol] = model.cpu()
                    new_scalers[symbol] = (mean, std)
                    results[symbol] = {"trained": True}

            except Exception as e:
                logger.exception("Failed to train Transformer for %s: %s", symbol, e)

        if new_models:
            self.models.update(new_models)
            self.scalers.update(new_scalers)
            self.cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)

            from lstm_engine import _atomic_torch_save, _atomic_npz_save
            _atomic_torch_save(
                {
                    "symbols": list(self.models.keys()),
                    "state_dicts": {s: m.state_dict() for s, m in self.models.items()},
                    "input_size": len(self.feature_cols),
                },
                self.ckpt_file
            )
            npz_payload = {}
            for sym, (mean, std) in self.scalers.items():
                npz_payload[f"{sym}__mean"] = mean
                npz_payload[f"{sym}__std"] = std
            _atomic_npz_save(self.scalers_file, **npz_payload)

        return results

    def load(self) -> bool:
        if not torch or not self.ckpt_file.exists() or not self.scalers_file.exists():
            return False
        try:
            ckpt = torch.load(self.ckpt_file, map_location="cpu", weights_only=True)
            n_features = int(ckpt["input_size"])
            if n_features != len(self.feature_cols):
                logger.warning(
                    "Transformer checkpoint feature width mismatch: checkpoint=%d current=%d",
                    n_features,
                    len(self.feature_cols),
                )
                return False
            state_dicts = ckpt["state_dicts"]

            for sym, state in state_dicts.items():
                m = TransformerTSModel(n_features)
                m.load_state_dict(state)
                m.eval()
                self.models[sym] = m

            npz = np.load(self.scalers_file)
            for sym in self.models.keys():
                mean_key, std_key = f"{sym}__mean", f"{sym}__std"
                if mean_key in npz.files and std_key in npz.files:
                    self.scalers[sym] = (npz[mean_key], npz[std_key])
            return True
        except Exception:
            return False

    def predict(self, symbol: str, df: Optional[pd.DataFrame] = None) -> float:
        if not torch or (not self.models and not self.load()):
            raise ModelNotAvailableError(f"No Transformer model for {symbol}")
        if df is None:
            df = fetch_ohlcv(symbol)

        feat = build_features(df, cfg=self.cfg)
        X_raw = feat[self.feature_cols].values.astype(np.float32)

        model = self.models.get(symbol)
        scaler = self.scalers.get(symbol)
        if not model or not scaler:
            raise ModelNotAvailableError(f"No Transformer model/scaler for {symbol}")

        mean, std = scaler
        X_norm, _, _ = self._normalize(X_raw, mean, std)
        if len(X_norm) < self.cfg.LSTM_WINDOW:
            raise ModelNotAvailableError(f"Not enough rows for Transformer window for {symbol}")

        seq = torch.from_numpy(X_norm[-self.cfg.LSTM_WINDOW:].astype(np.float32)).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            logit = model(seq).item()
        return float(torch.sigmoid(torch.tensor(logit)).item())


import math
