"""
lstm_engine.py — PyTorch LSTM for stock direction prediction.

Production hardening over the priority-1 patch:
- Modern AMP API: torch.amp.GradScaler / torch.amp.autocast (the cuda.amp
  variants are deprecated since PyTorch 2.x).
- True walk-forward validation: validation sequences only use rows whose label
  index is >= split. Previously val_start = split - WINDOW + 1 caused a few
  early "validation" sequences to end inside the training range.
- Atomic checkpoint write (tmp + os.replace).
- Scalers stored in a separate .npz file so the model checkpoint can be
  loaded with weights_only=True (safer than pickled objects).
- Idempotent training: if a symbol fails, others continue, and the final swap
  is all-or-nothing (we keep the previous on-disk checkpoint if no symbol
  trained successfully).
"""
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
import eval_metrics
import model_metrics
from data_manager import (
    fetch_ohlcv,
    build_features,
    make_labels,
    get_feature_columns,
)
from errors import ModelNotAvailableError

logger = logging.getLogger(__name__)
FEATURE_COLS = get_feature_columns()


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device()
logger.info("LSTM device: %s", DEVICE)

SCALERS_FILE = config.LSTM_CKPT_FILE.with_suffix(".scalers.npz")


class LSTMModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            dropout=config.LSTM_DROPOUT if config.LSTM_LAYERS > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(config.LSTM_DROPOUT),
            nn.Linear(config.LSTM_HIDDEN, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _subsample_train_sequences(
    X_tr: np.ndarray, y_tr: np.ndarray, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Stride training sequences by ``horizon`` to decorrelate overlapping
    forward-horizon labels. TRAINING rows only — validation sequences are never
    touched, so early stopping still sees every out-of-sample bar.
    """
    horizon = max(1, int(horizon))
    if len(X_tr) == 0:
        return X_tr, y_tr
    keep = np.arange(0, len(X_tr), horizon, dtype=int)
    return X_tr[keep], y_tr[keep]


def _make_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    window: int,
    min_end: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window sequences. Each sample X[k] = features[end-window+1:end+1]
    is paired with the label at `end`. Sequences whose end index is below
    `min_end` are skipped (used to enforce a strict train/val boundary).
    """
    if window <= 0:
        raise ValueError("window must be positive")

    X, y = [], []
    start_end = max(window - 1, min_end)
    for end in range(start_end, len(features)):
        label = labels[end]
        if np.isnan(label):
            continue
        X.append(features[end - window + 1:end + 1])
        y.append(label)

    if not X:
        return (
            np.zeros((0, window, features.shape[1] if features.ndim == 2 else 0), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
    )


def _atomic_torch_save(state: dict, path) -> None:
    path = str(path)
    tmp = f"{path}.tmp"
    torch.save(state, tmp)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(tmp, path)


def _atomic_npz_save(path, **arrays) -> None:
    """Write .npz atomically without numpy auto-appending a second .npz suffix."""
    path = str(path)
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    os.replace(tmp, path)


class StockLSTMEngine:
    def __init__(self) -> None:
        self.models: Dict[str, LSTMModel] = {}
        self.scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _normalize(arr, mean=None, std=None):
        arr = np.asarray(arr, dtype=np.float32)
        if mean is None:
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            std = np.where(std < 1e-8, 1.0, std)
        return (arr - mean) / std, mean, std

    def _split_and_scale(self, X_raw: np.ndarray, y_raw: np.ndarray):
        split = int(len(X_raw) * (1 - config.ML_TEST_RATIO))
        min_train = config.LSTM_WINDOW + 10
        min_val = 5

        if split < min_train or len(X_raw) - split < min_val:
            raise ValueError(
                f"Not enough rows for split: train={split}, val={len(X_raw) - split}"
            )

        # Train scaler from train rows only — no leakage.
        X_train_norm, mean, std = self._normalize(X_raw[:split])
        # Apply same mean/std to entire series so val sequences see proper context.
        X_full_norm, _, _ = self._normalize(X_raw, mean, std)

        # Train: sequences whose end < split. Val: sequences whose end >= split.
        X_tr, y_tr = _make_sequences(X_full_norm[:split], y_raw[:split], config.LSTM_WINDOW)
        X_val, y_val = _make_sequences(X_full_norm, y_raw, config.LSTM_WINDOW, min_end=split)

        if len(X_tr) == 0 or len(X_val) == 0:
            raise ValueError(f"Empty sequences: train={len(X_tr)}, val={len(X_val)}")
        if len(np.unique(y_tr)) < 2:
            raise ValueError("LSTM train labels contain only one class")

        return X_tr, y_tr, X_val, y_val, mean, std

    def train(self, symbols: Optional[list] = None, verbose: bool = True) -> dict:
        symbols = symbols or config.WATCHLIST
        results: dict = {}
        new_models: Dict[str, LSTMModel] = {}
        new_scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        use_cuda_amp = DEVICE.type == "cuda"

        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol)
                feat = build_features(df)
                labels = make_labels(feat)

                valid_idx = labels.dropna().index
                X_raw = feat.loc[valid_idx, FEATURE_COLS].values.astype(np.float32)
                y_raw = labels.loc[valid_idx].values.astype(np.float32)

                if len(X_raw) < config.LSTM_WINDOW + 60:
                    logger.warning("Not enough data for LSTM %s (n=%d)", symbol, len(X_raw))
                    continue

                X_tr, y_tr, X_val, y_val, mean, std = self._split_and_scale(X_raw, y_raw)

                # Decorrelate overlapping horizon labels on TRAIN sequences only.
                # Validation sequences are left intact for honest early stopping.
                n_train_seq_full = int(len(X_tr))
                X_tr, y_tr = _subsample_train_sequences(X_tr, y_tr, config.ML_HORIZON)
                if len(np.unique(y_tr)) < 2:
                    logger.warning(
                        "%s LSTM train collapsed to one class after horizon "
                        "subsampling — using full train sequences", symbol,
                    )
                    X_tr, y_tr, X_val, y_val, mean, std = self._split_and_scale(X_raw, y_raw)

                neg = float((y_tr == 0).sum())
                pos = float((y_tr == 1).sum())
                pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=DEVICE)

                tr_dl = DataLoader(
                    TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                    batch_size=config.LSTM_BATCH,
                    shuffle=False,
                )

                model = LSTMModel(len(FEATURE_COLS)).to(DEVICE)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_LR)

                # Modern AMP API. cuda.amp.* is deprecated.
                scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

                X_val_t = torch.from_numpy(X_val).to(DEVICE)
                y_val_t = torch.from_numpy(y_val).to(DEVICE)

                best_val_loss = float("inf")
                patience_ctr = 0
                best_state = None

                for epoch in range(config.LSTM_EPOCHS):
                    model.train()
                    for xb, yb in tr_dl:
                        xb = xb.to(DEVICE, non_blocking=True)
                        yb = yb.to(DEVICE, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)

                        with torch.amp.autocast(device_type=DEVICE.type, enabled=use_cuda_amp):
                            loss = criterion(model(xb), yb)

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(model(X_val_t), y_val_t).item()

                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                        patience_ctr = 0
                    else:
                        patience_ctr += 1
                        if patience_ctr >= config.LSTM_PATIENCE:
                            if verbose:
                                logger.info(
                                    "%s early stop @ epoch %d val_loss=%.4f",
                                    symbol, epoch + 1, best_val_loss,
                                )
                            break

                if best_state is None:
                    raise RuntimeError("LSTM training produced no best_state")

                model.load_state_dict(best_state)

                # Edge-detection metrics on the validation set — AUC + precision at
                # the BUY threshold match how the bot trades far better than BCE
                # val_loss alone. Computed while the model/val tensors are still on
                # DEVICE; guarded for the single-class case (returns None).
                model.eval()
                with torch.no_grad():
                    val_scores = torch.sigmoid(model(X_val_t)).detach().cpu().numpy()
                val_metrics = eval_metrics.pooled_classification_metrics(
                    y_val_t.detach().cpu().numpy(), val_scores
                )

                model.cpu().eval()

                new_models[symbol] = model
                new_scalers[symbol] = (mean.astype(np.float32), std.astype(np.float32))
                results[symbol] = {
                    "best_val_loss": round(float(best_val_loss), 4),
                    "best_val_auc": eval_metrics.round4(val_metrics["auc"]),
                    "val_precision_at_buy": eval_metrics.round4(val_metrics["precision_at_buy"]),
                    "n_val_at_buy": int(val_metrics["n_at_buy"]),
                    "n_train_seq": int(len(X_tr)),
                    "n_train_seq_full": int(n_train_seq_full),
                    "n_val_seq": int(len(X_val)),
                    "train_subsample_horizon": int(max(1, int(config.ML_HORIZON))),
                }
                if verbose:
                    v_auc = results[symbol]["best_val_auc"]
                    logger.info(
                        "%s LSTM val_loss=%.4f val_auc=%s train_seq=%d/%d (stride=%d) val_seq=%d",
                        symbol, best_val_loss,
                        f"{v_auc:.3f}" if v_auc is not None else "n/a",
                        len(X_tr), n_train_seq_full,
                        max(1, int(config.ML_HORIZON)), len(X_val),
                    )

            except Exception:
                logger.exception("LSTM training failed for %s", symbol)

        if not new_models:
            logger.error("LSTM training produced no models — keeping existing checkpoint")
            return results

        self.models = new_models
        self.scalers = new_scalers

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(
            {
                "symbols": list(self.models.keys()),
                "state_dicts": {s: m.state_dict() for s, m in self.models.items()},
                "input_size": len(FEATURE_COLS),
            },
            config.LSTM_CKPT_FILE,
        )
        # Save scalers separately so the model file can use weights_only=True at load.
        npz_payload: dict = {}
        for sym, (mean, std) in self.scalers.items():
            npz_payload[f"{sym}__mean"] = mean
            npz_payload[f"{sym}__std"] = std
        _atomic_npz_save(SCALERS_FILE, **npz_payload)

        logger.info("LSTM checkpoint saved → %s (n=%d)", config.LSTM_CKPT_FILE, len(new_models))
        # Phase 1: persist per-symbol metrics + train date for the staleness gate.
        # Wrapped internally so it never breaks training.
        model_metrics.save_lstm_metrics({s: results[s] for s in new_models if s in results})
        return results

    def load(self) -> bool:
        if not config.LSTM_CKPT_FILE.exists() or not SCALERS_FILE.exists():
            return False
        try:
            try:
                ckpt = torch.load(
                    config.LSTM_CKPT_FILE,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                # Older torch without weights_only kwarg.
                ckpt = torch.load(config.LSTM_CKPT_FILE, map_location="cpu")

            n_features = int(ckpt.get("input_size", len(FEATURE_COLS)))
            state_dicts = ckpt["state_dicts"]

            loaded_models: Dict[str, LSTMModel] = {}
            for symbol, state in state_dicts.items():
                m = LSTMModel(n_features)
                m.load_state_dict(state)
                m.eval()
                loaded_models[symbol] = m

            npz = np.load(SCALERS_FILE)
            scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for sym in loaded_models.keys():
                mean_key = f"{sym}__mean"
                std_key = f"{sym}__std"
                if mean_key in npz.files and std_key in npz.files:
                    scalers[sym] = (npz[mean_key], npz[std_key])

            self.models = loaded_models
            self.scalers = scalers
            return True
        except Exception:
            logger.exception("Failed to load LSTM checkpoint")
            return False

    def predict(self, symbol: str, df=None) -> float:
        if not self.models and not self.load():
            raise RuntimeError("LSTM not trained. Run train() first.")
        if df is None:
            df = fetch_ohlcv(symbol)

        feat = build_features(df)
        X_raw = feat[FEATURE_COLS].values.astype(np.float32)

        model = self.models.get(symbol)
        scaler = self.scalers.get(symbol)
        if model is None or scaler is None:
            raise ModelNotAvailableError(f"No LSTM model/scaler for {symbol}")
        mean, std = scaler

        X_norm, _, _ = self._normalize(X_raw, mean, std)
        if len(X_norm) < config.LSTM_WINDOW:
            raise ModelNotAvailableError(
                f"Not enough rows to build an LSTM window for {symbol} "
                f"({len(X_norm)} < {config.LSTM_WINDOW})"
            )

        seq = torch.from_numpy(
            X_norm[-config.LSTM_WINDOW:].astype(np.float32)
        ).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            logit = model(seq).item()
        return float(torch.sigmoid(torch.tensor(logit)).item())
