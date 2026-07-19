"""
lstm_engine.py — PyTorch LSTM for stock direction prediction (v2 refactor).

Changes vs v1:
- Attention-augmented LSTM: learned query/key/value over the temporal
  sequence so the model learns WHERE to look in the window, not just the
  final state.
- Bidirectional option (config.LSTM_BIDIRECTIONAL): 2 layers become 4.
- Gradient clipping (LSTM_GRAD_CLIP_NORM) to prevent exploding gradients.
- Cyclical LR scheduler (LSTM_CYCLE_LR_STEP / LSTM_CYCLE_LR_BASE) for
  better convergence and saddle-point escape.
- Micro-structure features (config.USE_MICRO_FEATURES) wired into the
  feature set.
- Meta-label mode: secondary model predicts "will the primary be right?"
  (config.LSTM_META_ENSEMBLE). The ensemble confidence is the product of
  flip-prediction and primary confidence.
"""
import logging
import os
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    nn = None
    DEVICE = "cpu"

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


def _select_device():
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device()
logger.info("LSTM device: %s", DEVICE)

SCALERS_FILE = config.LSTM_CKPT_FILE.with_suffix(".scalers.npz")


# ── Attention-Augmented LSTM ──────────────────────────────────────────

# If torch is not available, provide a stub that raises ModelNotAvailableError when used
if torch is not None and nn is not None:
    class AttentionLSTM(nn.Module):
        """LSTM with per-step attention over the hidden sequence.

        For each time-step the hidden state is attended with a learned query
        projected from the final hidden state — this gives the network a
        learnable "look-back" over earlier parts of the sequence instead of
        collapsing everything into the last hidden vector.
        """

        def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                     dropout: float, bidirectional: bool):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
            lstm_out = hidden_size * (2 if bidirectional else 1)
            self.W_q = nn.Linear(lstm_out, lstm_out)
            self.W_k = nn.Linear(lstm_out, lstm_out)
            self.W_v = nn.Linear(lstm_out, lstm_out)
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(lstm_out, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)  # (B, T, H)
            # Self-attention: Q=W_q(final_state), K=W_k(all_states)
            final = out[:, -1:, :]  # (B, 1, H)
            Q = self.W_q(final)  # (B, 1, H)
            K = self.W_k(out)  # (B, T, H)
            V = self.W_v(out)  # (B, T, H)
            # Scaled dot-product attention
            scale = math.sqrt(K.size(-1))
            attn_weights = torch.softmax((Q @ K.transpose(-2, -1)) / scale, dim=-1)
            context = attn_weights @ V  # (B, 1, H)
            attended = context.squeeze(1)  # (B, H)
            return self.head(attended).squeeze(-1)
else:
    class AttentionLSTM:
        """Stub class when torch is not available."""
        def __init__(self, *args, **kwargs):
            raise ModelNotAvailableError("PyTorch not installed - LSTM models unavailable")


# ── Sequence building ──────────────────────────────────────────────────


def _subsample_train_sequences(
    X_tr: np.ndarray, y_tr: np.ndarray, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Stride training sequences by ``horizon`` to decorrelate overlapping
    forward-horizon labels. TRAINING rows only.
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


# ── Atomic save helpers ────────────────────────────────────────────────


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


# ── Engine ─────────────────────────────────────────────────────────────


class StockLSTMEngine:
    def __init__(self, settings=None) -> None:
        self.cfg = settings or config.get_settings()
        self.feature_cols = get_feature_columns(self.cfg)
        self.scalers_file = self.cfg.LSTM_CKPT_FILE.with_suffix(".scalers.npz")
        self.models: Dict[str, AttentionLSTM] = {}
        self.meta_models: Dict[str, AttentionLSTM] = {}
        self.scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._loaded: bool = False

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
        min_train = self.cfg.LSTM_WINDOW + 10
        min_val = 5

        if split < min_train or len(X_raw) - split < min_val:
            raise ValueError(
                f"Not enough rows for split: train={split}, val={len(X_raw) - split}"
            )

        X_train_norm, mean, std = self._normalize(X_raw[:split])
        X_full_norm, _, _ = self._normalize(X_raw, mean, std)

        X_tr, y_tr = _make_sequences(X_full_norm[:split], y_raw[:split], self.cfg.LSTM_WINDOW)
        X_val, y_val = _make_sequences(X_full_norm, y_raw, self.cfg.LSTM_WINDOW, min_end=split)

        if len(X_tr) == 0 or len(X_val) == 0:
            raise ValueError(f"Empty sequences: train={len(X_tr)}, val={len(X_val)}")
        if len(np.unique(y_tr)) < 2:
            raise ValueError("LSTM train labels contain only one class")

        return X_tr, y_tr, X_val, y_val, mean, std

    def _make_model(self) -> AttentionLSTM:
        return AttentionLSTM(
            input_size=len(self.feature_cols),
            hidden_size=self.cfg.LSTM_HIDDEN,
            num_layers=self.cfg.LSTM_LAYERS,
            dropout=self.cfg.LSTM_DROPOUT,
            bidirectional=bool(getattr(self.cfg, "LSTM_BIDIRECTIONAL", False)),
        )

    def _train_one(self, symbol: str, X_tr, y_tr, X_val, y_val) -> Tuple[AttentionLSTM, dict]:
        """Train a single LSTM model for one symbol. Returns (model, metrics_dict)."""

        neg = float((y_tr == 0).sum())
        pos = float((y_tr == 1).sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float, device=DEVICE)

        tr_dl = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=self.cfg.LSTM_BATCH,
            shuffle=False,
        )

        model = self._make_model().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg.LSTM_LR)

        use_cuda_amp = DEVICE.type == "cuda"
        grad_scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_amp)

        # Cyclical LR scheduler
        cycle_step = int(getattr(self.cfg, "LSTM_CYCLE_LR_STEP", 0))
        if cycle_step > 0:
            base_lr = float(getattr(self.cfg, "LSTM_CYCLE_LR_BASE", 1e-5))
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=base_lr, max_lr=self.cfg.LSTM_LR,
                step_size_up=cycle_step, step_size_down=cycle_step,
                mode="triangular2", cycle_momentum=False,
            )

        X_val_t = torch.from_numpy(X_val).to(DEVICE)
        y_val_t = torch.from_numpy(y_val).to(DEVICE)

        best_val_loss = float("inf")
        patience_ctr = 0
        best_state = None

        grad_clip = float(getattr(self.cfg, "LSTM_GRAD_CLIP_NORM", 0.0) or 0.0)

        for epoch in range(self.cfg.LSTM_EPOCHS):
            model.train()
            for xb, yb in tr_dl:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type=DEVICE.type, enabled=use_cuda_amp):
                    loss = criterion(model(xb), yb)

                grad_scaler.scale(loss).backward()
                if grad_clip > 0:
                    # Unscale gradients first, then clip
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_val_t), y_val_t).item()

            if cycle_step > 0:
                scheduler.step()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.cfg.LSTM_PATIENCE:
                    logger.info(
                        "%s early stop @ epoch %d val_loss=%.4f",
                        symbol, epoch + 1, best_val_loss,
                    )
                    break

        if best_state is None:
            raise RuntimeError("LSTM training produced no best_state")

        model.load_state_dict(best_state)

        # Validation metrics
        model.eval()
        with torch.no_grad():
            val_scores = torch.sigmoid(model(X_val_t)).detach().cpu().numpy()
        val_metrics = eval_metrics.pooled_classification_metrics(
            y_val_t.detach().cpu().numpy(), val_scores
        )

        model.cpu().eval()

        results = {
            "best_val_loss": round(float(best_val_loss), 4),
            "best_val_auc": eval_metrics.round4(val_metrics["auc"]),
            "val_precision_at_buy": eval_metrics.round4(val_metrics["precision_at_buy"]),
            "n_val_at_buy": int(val_metrics["n_at_buy"]),
            "n_train_seq": int(len(X_tr)),
            "n_val_seq": int(len(X_val)),
            "bidirectional": bool(getattr(self.cfg, "LSTM_BIDIRECTIONAL", False)),
            "attention": True,
            "grad_clip": grad_clip,
            "cycle_lr_step": cycle_step,
            "n_features": len(self.feature_cols),
        }
        return model, results

    def _train_meta(self, symbol: str, X_val, y_val, primary_scores: np.ndarray,
                    metrics: dict) -> Optional[AttentionLSTM]:
        """Train a meta-labeling model: given the primary model's confidence,
        can we predict whether the primary will be correct?

        The meta-model uses the primary's sigmoid output as an additional
        feature alongside the raw feature sequence. Returns None if meta
        is disabled or the data is insufficient for a second stage.
        """
        if not bool(getattr(self.cfg, "LSTM_META_ENSEMBLE", False)):
            return None
        try:
            # Meta target: 1.0 where primary's prediction matches label
            primary_pred = (primary_scores > 0.5).astype(np.float32)
            y_meta = (primary_pred == y_val).astype(np.float32)

            if len(np.unique(y_meta)) < 2:
                logger.info("%s meta-label collapsed to one class — skipping", symbol)
                return None

            # Augment sequences with the primary score as a time-invariant feature
            # (we broadcast the final sigmoid output across the sequence). This is
            # reductive but keeps the meta model architecture identical.
            extra = np.full((X_val.shape[0], X_val.shape[1], 1),
                            primary_scores[:len(X_val)].mean(), dtype=np.float32)
            X_meta = np.concatenate([X_val, extra], axis=-1).astype(np.float32)

            # Train meta-model on val set — same architecture, no labels beyond val
            neg_m = float((y_meta == 0).sum())
            pos_m = float((y_meta == 1).sum())
            pwl = torch.tensor([neg_m / max(pos_m, 1.0)], dtype=torch.float, device=DEVICE)

            meta_dl = DataLoader(
                TensorDataset(torch.from_numpy(X_meta), torch.from_numpy(y_meta)),
                batch_size=self.cfg.LSTM_BATCH, shuffle=False,
            )

            meta_model = AttentionLSTM(
                input_size=X_meta.shape[-1],
                hidden_size=self.cfg.LSTM_HIDDEN,
                num_layers=1,
                dropout=self.cfg.LSTM_DROPOUT,
                bidirectional=False,
            ).to(DEVICE)

            meta_optim = torch.optim.Adam(meta_model.parameters(), lr=self.cfg.LSTM_LR / 2)
            meta_criterion = nn.BCEWithLogitsLoss(pos_weight=pwl)
            meta_best = None
            meta_loss = float("inf")
            meta_ctr = 0
            X_mt = torch.from_numpy(X_meta).to(DEVICE)
            y_mt = torch.from_numpy(y_meta).to(DEVICE)

            for epoch in range(8):
                meta_model.train()
                for xb, yb in meta_dl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    meta_optim.zero_grad()
                    loss_m = meta_criterion(meta_model(xb), yb)
                    loss_m.backward()
                    if grad_clip := float(getattr(self.cfg, "LSTM_GRAD_CLIP_NORM", 0.0)):
                        torch.nn.utils.clip_grad_norm_(meta_model.parameters(), grad_clip)
                    meta_optim.step()

                meta_model.eval()
                with torch.no_grad():
                    v_loss = meta_criterion(meta_model(X_mt), y_mt).item()
                if v_loss < meta_loss - 1e-6:
                    meta_loss = v_loss
                    meta_best = {k: v.detach().cpu().clone()
                                 for k, v in meta_model.state_dict().items()}
                    meta_ctr = 0
                else:
                    meta_ctr += 1
                    if meta_ctr >= 3:
                        break

            if meta_best is not None:
                meta_model.load_state_dict(meta_best)
                meta_model.cpu().eval()
                metrics["meta_trained"] = True
                metrics["meta_val_loss"] = round(float(meta_loss), 4)
                logger.info("%s meta-label model trained (val_loss=%.4f)", symbol, meta_loss)
                return meta_model

            metrics["meta_trained"] = False
            return None
        except Exception:
            logger.debug("Meta-label training failed for %s", symbol, exc_info=True)
            metrics["meta_trained"] = False
            return None

    def train(self, symbols: Optional[list] = None, verbose: bool = True) -> dict:
        symbols = symbols or self.cfg.WATCHLIST
        results: dict = {}
        new_models: Dict[str, AttentionLSTM] = {}
        new_meta: Dict[str, AttentionLSTM] = {}
        new_scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        use_cuda_amp = DEVICE.type == "cuda"

        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol)
                feat = build_features(df, cfg=self.cfg)
                labels = make_labels(feat)

                valid_idx = labels.dropna().index
                X_raw = feat.loc[valid_idx, self.feature_cols].values.astype(np.float32)
                y_raw = labels.loc[valid_idx].values.astype(np.float32)

                if len(X_raw) < self.cfg.LSTM_WINDOW + 60:
                    logger.warning("Not enough data for LSTM %s (n=%d)", symbol, len(X_raw))
                    continue

                X_tr, y_tr, X_val, y_val, mean, std = self._split_and_scale(X_raw, y_raw)

                n_train_seq_full = int(len(X_tr))
                X_tr, y_tr = _subsample_train_sequences(X_tr, y_tr, self.cfg.ML_HORIZON)
                if len(np.unique(y_tr)) < 2:
                    logger.warning(
                        "%s LSTM train collapsed to one class after horizon "
                        "subsampling — using full train sequences", symbol,
                    )
                    X_tr, y_tr, X_val, y_val, mean, std = self._split_and_scale(X_raw, y_raw)

                model, sym_results = self._train_one(symbol, X_tr, y_tr, X_val, y_val)

                # Meta-labeling: get primary model's sigmoid scores on validation set
                with torch.no_grad():
                    primary_val_logits = model(torch.from_numpy(X_val).to(DEVICE))
                    primary_val_scores = torch.sigmoid(primary_val_logits).detach().cpu().numpy()
                meta_model = self._train_meta(symbol, X_val, y_val,
                                              primary_val_scores,
                                              sym_results)
                if meta_model is not None:
                    new_meta[symbol] = meta_model

                new_models[symbol] = model
                new_scalers[symbol] = (mean.astype(np.float32), std.astype(np.float32))
                results[symbol] = sym_results

                if verbose:
                    v_auc = sym_results["best_val_auc"]
                    logger.info(
                        "%s LSTM val_loss=%.4f val_auc=%s meta=%s",
                        symbol, sym_results["best_val_loss"],
                        f"{v_auc:.3f}" if v_auc is not None else "n/a",
                        sym_results.get("meta_trained", False),
                    )

            except Exception:
                logger.exception("LSTM training failed for %s", symbol)

        if not new_models:
            logger.error("LSTM training produced no models — keeping existing checkpoint")
            return results

        self.models = new_models
        self.meta_models = new_meta
        self.scalers = new_scalers

        self.cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_torch_save(
            {
                "symbols": list(self.models.keys()),
                "state_dicts": {s: m.state_dict() for s, m in self.models.items()},
                "input_size": len(self.feature_cols),
                "meta_symbols": list(self.meta_models.keys()),
                "meta_state_dicts": {s: m.state_dict() for s, m in self.meta_models.items()},
            },
            self.cfg.LSTM_CKPT_FILE,
        )
        npz_payload: dict = {}
        for sym, (mean, std) in self.scalers.items():
            npz_payload[f"{sym}__mean"] = mean
            npz_payload[f"{sym}__std"] = std
        _atomic_npz_save(self.scalers_file, **npz_payload)

        logger.info("LSTM checkpoint saved → %s (n=%d, meta=%d)",
                     self.cfg.LSTM_CKPT_FILE, len(new_models), len(new_meta))
        model_metrics.save_lstm_metrics({s: results[s] for s in new_models if s in results})
        return results

    def load(self) -> bool:
        if not self.cfg.LSTM_CKPT_FILE.exists() or not self.scalers_file.exists():
            return False
        try:
            try:
                ckpt = torch.load(
                    self.cfg.LSTM_CKPT_FILE,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                ckpt = torch.load(self.cfg.LSTM_CKPT_FILE, map_location="cpu")

            n_features = int(ckpt.get("input_size", len(self.feature_cols)))
            if n_features != len(self.feature_cols):
                logger.warning(
                    "LSTM checkpoint feature width mismatch: checkpoint=%d current=%d",
                    n_features,
                    len(self.feature_cols),
                )
                return False
            state_dicts = ckpt["state_dicts"]
            meta_sds = ckpt.get("meta_state_dicts", {})

            loaded_models: Dict[str, AttentionLSTM] = {}
            for symbol, state in state_dicts.items():
                m = AttentionLSTM(
                    input_size=n_features,
                    hidden_size=self.cfg.LSTM_HIDDEN,
                    num_layers=self.cfg.LSTM_LAYERS,
                    dropout=self.cfg.LSTM_DROPOUT,
                    bidirectional=bool(getattr(self.cfg, "LSTM_BIDIRECTIONAL", False)),
                )
                m.load_state_dict(state, strict=False)
                m.eval()
                loaded_models[symbol] = m

            loaded_meta: Dict[str, AttentionLSTM] = {}
            for symbol, state in meta_sds.items():
                # Meta model may have one extra input feature (primary score)
                mb = AttentionLSTM(
                    input_size=n_features + 1,
                    hidden_size=self.cfg.LSTM_HIDDEN,
                    num_layers=1,
                    dropout=self.cfg.LSTM_DROPOUT,
                    bidirectional=False,
                )
                mb.load_state_dict(state, strict=False)
                mb.eval()
                loaded_meta[symbol] = mb

            npz = np.load(self.scalers_file)
            scalers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
            for sym in loaded_models.keys():
                mean_key = f"{sym}__mean"
                std_key = f"{sym}__std"
                if mean_key in npz.files and std_key in npz.files:
                    scalers[sym] = (npz[mean_key], npz[std_key])

            self.models = loaded_models
            self.meta_models = loaded_meta
            self.scalers = scalers
            if not self._loaded:
                logger.info("LSTM checkpoint loaded (n=%d, meta=%d)", len(loaded_models), len(loaded_meta))
                self._loaded = True
            return True
        except Exception:
            logger.exception("Failed to load LSTM checkpoint")
            return False

    def predict(self, symbol: str, df=None) -> float:
        """Primary direction score in [0, 1].

        With meta-labeling enabled, the ensemble confidence is adjusted:
          confidence = primary * meta_p(primary_correct)
        so low meta confidence suppresses a primary prediction instead of
        increasing its score.
        """
        if not self.models and not self.load():
            raise RuntimeError("LSTM not trained. Run train() first.")
        if df is None:
            df = fetch_ohlcv(symbol)

        feat = build_features(df, cfg=self.cfg)
        X_raw = feat[self.feature_cols].values.astype(np.float32)

        model = self.models.get(symbol)
        scaler = self.scalers.get(symbol)
        if model is None or scaler is None:
            raise ModelNotAvailableError(f"No LSTM model/scaler for {symbol}")
        mean, std = scaler

        X_norm, _, _ = self._normalize(X_raw, mean, std)
        if len(X_norm) < self.cfg.LSTM_WINDOW:
            raise ModelNotAvailableError(
                f"Not enough rows to build an LSTM window for {symbol} "
                f"({len(X_norm)} < {self.cfg.LSTM_WINDOW})"
            )

        seq = torch.from_numpy(
            X_norm[-self.cfg.LSTM_WINDOW:].astype(np.float32)
        ).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            primary_logit = model(seq).item()
        primary = float(torch.sigmoid(torch.tensor(primary_logit)).item())

        # Meta-label adjustment
        meta_model = self.meta_models.get(symbol)
        if meta_model is not None and bool(getattr(self.cfg, "LSTM_META_ENSEMBLE", False)):
            try:
                # Augment sequence with the primary score as a constant extra channel
                extra = np.full((1, self.cfg.LSTM_WINDOW, 1), primary, dtype=np.float32)
                seq_meta = torch.from_numpy(
                    np.concatenate([
                        seq.numpy() if not seq.is_cuda else seq.cpu().numpy(),
                        extra,
                    ], axis=-1).astype(np.float32)
                )
                meta_model.eval()
                with torch.no_grad():
                    meta_logit = meta_model(seq_meta).item()
                meta_conf = float(torch.sigmoid(torch.tensor(meta_logit)).item())
                # Ensemble: primary * P(primary_correct), re-squash to [0,1]
                adjusted = primary * meta_conf
                return min(adjusted, 1.0)
            except Exception:
                logger.debug("Meta-label inference failed for %s, using primary", symbol)
                return primary

        return primary
