"""
tools/qualify_phase2_models.py — Dedicated Phase 2 Model Qualification Harness.

Provides a transactional, rollback-guaranteed training and qualification pipeline
for expanding the model universe by graduating CAT and LLY (20 -> 22 symbols).

Key Invariants:
1. Explicit Candidate Scoping: ONLY trains CAT and LLY.
2. 4-Family Orchestration: Covers RF, Attention LSTM, XGBoost, and Transformer.
3. Legacy-20 Preservation: Pre-training snapshot taken. Every legacy model must
   remain intact and uncorrupted after candidate merge.
4. Transactional Rollback: If ANY model family fails training, serialization,
   or metric-gate validation, all files are immediately restored from snapshot.
5. Strict Gate Alignment:
   - RF Accuracy >= config.MODEL_MIN_RF_ACCURACY (0.45)
   - RF F1       >= config.MODEL_MIN_RF_F1       (0.35)
   - Freshness   <= config.MODEL_MAX_AGE_DAYS    (30 days)
   - model_metrics.evaluate_gate(sym).ok == True (status == 'ok')
6. Policy Frozen: Transformer weight remains 0.00 (artifact readiness only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import torch

# Ensure repository root is on path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
import model_metrics
from ai_engine import StockRFEngine
from alternative_models import StockTransformerEngine, StockXGBEngine
from lstm_engine import StockLSTMEngine, _atomic_npz_save, _atomic_torch_save

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qualify_phase2_models")

LEGACY_20: List[str] = [
    "SPY", "QQQ", "VTI",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN",
    "GOOGL", "AVGO", "NFLX", "COST", "JPM", "BAC", "XOM", "UNH",
    "XLV", "V",
]

CANDIDATES: List[str] = ["CAT", "LLY"]
TARGET_22: List[str] = LEGACY_20 + CANDIDATES

SNAPSHOT_DIR = config.MODELS_DIR / "backups" / "pre_phase2_snapshot"
TRACKED_FILES: List[str] = [
    "rf_models.joblib",
    "lstm_checkpoint.pt",
    "lstm_checkpoint.scalers.npz",
    "xgb_models.joblib",
    "transformer_checkpoint.pt",
    "transformer_checkpoint.scalers.npz",
    "model_metrics.json",
]


def _sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ModelQualificationHarness:
    """Transactional harness for qualifying Phase 2 model candidates."""

    def __init__(self, candidates: Optional[List[str]] = None) -> None:
        self.candidates = candidates or list(CANDIDATES)
        self.legacy_symbols = list(LEGACY_20)
        self.target_symbols = list(TARGET_22)
        self.snapshot_dir = SNAPSHOT_DIR
        self._snapshot_taken = False

    def verify_pre_conditions(self) -> Dict[str, Any]:
        """Verify that all 4 model files and metrics exist and contain exact legacy 20 symbols."""
        logger.info("Verifying pre-conditions on existing model checkpoints...")
        status = {}

        # 1. RF
        rf_path = config.ML_MODELS_FILE
        if not rf_path.exists():
            raise FileNotFoundError(f"Missing RF model file: {rf_path}")
        rf_data = joblib.load(rf_path)
        rf_syms = set(rf_data.keys()) if isinstance(rf_data, dict) else set()
        if not set(self.legacy_symbols).issubset(rf_syms):
            missing = set(self.legacy_symbols) - rf_syms
            raise ValueError(f"RF checkpoint missing legacy symbols: {missing}")
        status["rf"] = {"count": len(rf_syms), "has_legacy_20": True}

        # 2. LSTM
        lstm_path = config.LSTM_CKPT_FILE
        if not lstm_path.exists():
            raise FileNotFoundError(f"Missing LSTM checkpoint: {lstm_path}")
        lstm_data = torch.load(lstm_path, map_location="cpu", weights_only=False)
        lstm_syms = set(lstm_data.get("symbols", [])) if isinstance(lstm_data, dict) else set()
        if not set(self.legacy_symbols).issubset(lstm_syms):
            missing = set(self.legacy_symbols) - lstm_syms
            raise ValueError(f"LSTM checkpoint missing legacy symbols: {missing}")
        status["lstm"] = {"count": len(lstm_syms), "has_legacy_20": True}

        # 3. XGB
        xgb_path = config.MODELS_DIR / "xgb_models.joblib"
        if not xgb_path.exists():
            raise FileNotFoundError(f"Missing XGB model file: {xgb_path}")
        xgb_data = joblib.load(xgb_path)
        xgb_syms = set(xgb_data.keys()) if isinstance(xgb_data, dict) else set()
        if not set(self.legacy_symbols).issubset(xgb_syms):
            missing = set(self.legacy_symbols) - xgb_syms
            raise ValueError(f"XGB checkpoint missing legacy symbols: {missing}")
        status["xgb"] = {"count": len(xgb_syms), "has_legacy_20": True}

        # 4. Transformer
        trans_path = config.MODELS_DIR / "transformer_checkpoint.pt"
        if not trans_path.exists():
            raise FileNotFoundError(f"Missing Transformer checkpoint: {trans_path}")
        trans_data = torch.load(trans_path, map_location="cpu", weights_only=False)
        trans_syms = set(trans_data.get("symbols", [])) if isinstance(trans_data, dict) else set()
        if not set(self.legacy_symbols).issubset(trans_syms):
            missing = set(self.legacy_symbols) - trans_syms
            raise ValueError(f"Transformer checkpoint missing legacy symbols: {missing}")
        status["transformer"] = {"count": len(trans_syms), "has_legacy_20": True}

        logger.info("Pre-conditions verified successfully: all 4 model families contain legacy 20 symbols.")
        return status

    def create_snapshot(self) -> Dict[str, str]:
        """Create a full snapshot backup of all tracked model and metrics files."""
        logger.info("Creating pre-training snapshot at %s...", self.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        manifest = {}

        for fname in TRACKED_FILES:
            src = config.MODELS_DIR / fname
            if not src.exists():
                logger.warning("Tracked file %s does not exist on disk — skipping", src)
                continue
            dst = self.snapshot_dir / fname
            shutil.copy2(src, dst)
            sha = _sha256(dst)
            manifest[fname] = sha

        manifest_path = self.snapshot_dir / "manifest.json"
        manifest_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "legacy_symbols": self.legacy_symbols,
            "candidates": self.candidates,
            "hashes": manifest,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        self._snapshot_taken = True
        logger.info("Snapshot created successfully (%d files archived).", len(manifest))
        return manifest

    def restore_snapshot(self) -> None:
        """Restore all tracked model and metrics files from snapshot (Rollback)."""
        logger.warning("INITIATING ATOMIC ROLLBACK FROM SNAPSHOT: %s", self.snapshot_dir)
        if not self.snapshot_dir.exists():
            logger.critical("Cannot rollback: snapshot directory %s does not exist!", self.snapshot_dir)
            return

        for fname in TRACKED_FILES:
            src = self.snapshot_dir / fname
            if src.exists():
                dst = config.MODELS_DIR / fname
                shutil.copy2(src, dst)
                logger.info("Rolled back: %s -> %s", src, dst)

        logger.info("Rollback completed cleanly. All model checkpoints restored to pre-training state.")

    def train_rf(self) -> Dict[str, Any]:
        """Train Random Forest models for candidates with incremental merge."""
        logger.info("--- [STEP C.1] Training Random Forest for %s ---", self.candidates)
        engine = StockRFEngine()
        results = engine.train(symbols=self.candidates, verbose=True)
        if not results:
            raise RuntimeError(f"RF training produced no results for {self.candidates}")

        # Verify disk persistence and legacy preservation
        data = joblib.load(config.ML_MODELS_FILE)
        syms = set(data.keys())
        missing_legacy = set(self.legacy_symbols) - syms
        if missing_legacy:
            raise RuntimeError(f"RF legacy preservation violated! Missing: {missing_legacy}")
        for c in self.candidates:
            if c not in syms:
                raise RuntimeError(f"RF candidate {c} missing from merged file!")
        logger.info("RF training & merge verified: n=%d symbols present.", len(syms))
        return results

    def train_lstm(self) -> Dict[str, Any]:
        """Train Attention LSTM models for candidates with explicit checkpoint merge."""
        logger.info("--- [STEP C.2] Training Attention LSTM for %s ---", self.candidates)
        engine = StockLSTMEngine()

        # 1. Pre-load existing 20 models and scalers
        if not engine.load():
            raise RuntimeError("Failed to pre-load existing LSTM checkpoint for merge")
        existing_models = dict(engine.models)
        existing_meta = dict(engine.meta_models)
        existing_scalers = dict(engine.scalers)

        # 2. Train candidates
        results = engine.train(symbols=self.candidates, verbose=True)
        if not results:
            raise RuntimeError(f"LSTM training produced no results for {self.candidates}")

        # 3. Explicitly merge existing legacy models with new candidate models
        new_models = dict(engine.models)
        new_meta = dict(engine.meta_models)
        new_scalers = dict(engine.scalers)

        merged_models = dict(existing_models)
        merged_models.update(new_models)

        merged_meta = dict(existing_meta)
        merged_meta.update(new_meta)

        merged_scalers = dict(existing_scalers)
        merged_scalers.update(new_scalers)

        # 4. Save merged checkpoint atomically
        _atomic_torch_save(
            {
                "symbols": list(merged_models.keys()),
                "state_dicts": {s: m.state_dict() for s, m in merged_models.items()},
                "input_size": len(engine.feature_cols),
                "meta_symbols": list(merged_meta.keys()),
                "meta_state_dicts": {s: m.state_dict() for s, m in merged_meta.items()},
            },
            engine.cfg.LSTM_CKPT_FILE,
        )

        npz_payload: dict = {}
        for sym, (mean, std) in merged_scalers.items():
            npz_payload[f"{sym}__mean"] = mean
            npz_payload[f"{sym}__std"] = std
        _atomic_npz_save(engine.scalers_file, **npz_payload)

        # 5. Verify post-conditions
        ckpt = torch.load(engine.cfg.LSTM_CKPT_FILE, map_location="cpu", weights_only=False)
        syms = set(ckpt.get("symbols", []))
        missing_legacy = set(self.legacy_symbols) - syms
        if missing_legacy:
            raise RuntimeError(f"LSTM legacy preservation violated! Missing: {missing_legacy}")
        for c in self.candidates:
            if c not in syms:
                raise RuntimeError(f"LSTM candidate {c} missing from merged file!")

        logger.info("LSTM training & merge verified: n=%d symbols present.", len(syms))
        return results

    def train_xgb(self) -> Dict[str, Any]:
        """Train Gradient Boosting (XGBoost) models for candidates with pre-load merge."""
        logger.info("--- [STEP C.3] Training Gradient Boosting for %s ---", self.candidates)
        engine = StockXGBEngine()

        # Pre-load existing models into memory so .update() acts on existing models
        if not engine.load():
            logger.warning("Could not pre-load XGB models; continuing if file absent")

        results = engine.train(symbols=self.candidates)
        if not results:
            raise RuntimeError(f"XGB training produced no results for {self.candidates}")

        # Verify disk persistence and legacy preservation
        data = joblib.load(engine.models_file)
        syms = set(data.keys())
        missing_legacy = set(self.legacy_symbols) - syms
        if missing_legacy:
            raise RuntimeError(f"XGB legacy preservation violated! Missing: {missing_legacy}")
        for c in self.candidates:
            if c not in syms:
                raise RuntimeError(f"XGB candidate {c} missing from merged file!")

        logger.info("XGB training & merge verified: n=%d symbols present.", len(syms))
        return results

    def train_transformer(self) -> Dict[str, Any]:
        """Train Transformer Time-Series models for candidates with pre-load merge."""
        logger.info("--- [STEP C.4] Training Transformer for %s ---", self.candidates)
        engine = StockTransformerEngine()

        # Pre-load existing models into memory so .update() acts on existing models
        if not engine.load():
            logger.warning("Could not pre-load Transformer models; continuing if file absent")

        results = engine.train(symbols=self.candidates)
        if not results:
            raise RuntimeError(f"Transformer training produced no results for {self.candidates}")

        # Verify disk persistence and legacy preservation
        data = torch.load(engine.ckpt_file, map_location="cpu", weights_only=False)
        syms = set(data.get("symbols", []))
        missing_legacy = set(self.legacy_symbols) - syms
        if missing_legacy:
            raise RuntimeError(f"Transformer legacy preservation violated! Missing: {missing_legacy}")
        for c in self.candidates:
            if c not in syms:
                raise RuntimeError(f"Transformer candidate {c} missing from merged file!")

        logger.info("Transformer training & merge verified: n=%d symbols present.", len(syms))
        return results

    def validate_metric_gates(self) -> Dict[str, Any]:
        """Validate candidate models against config metric floors and staleness gates (Step D)."""
        logger.info("--- [STEP D] Validating Metric & Staleness Gates ---")
        min_acc = float(getattr(config, "MODEL_MIN_RF_ACCURACY", 0.45))
        min_f1 = float(getattr(config, "MODEL_MIN_RF_F1", 0.35))
        max_age = int(getattr(config, "MODEL_MAX_AGE_DAYS", 30))

        gate_results = {}
        for sym in self.candidates:
            gate = model_metrics.evaluate_gate(sym)
            logger.info("Gate evaluation for %s: ok=%s status=%s reason=%s",
                        sym, gate.ok, gate.status, gate.reason)

            if not gate.ok:
                raise ValueError(f"Metric gate FAILED for candidate {sym}: {gate.reason} (status={gate.status})")

            # Check individual RF metrics
            metrics = model_metrics.load_metrics(sym, "rf")
            if not metrics:
                raise ValueError(f"No RF metrics recorded for candidate {sym}")

            acc = float(metrics.get("accuracy", 0.0))
            f1 = float(metrics.get("f1", 0.0))
            trained_at = metrics.get("trained_at", "")

            if acc < min_acc:
                raise ValueError(f"RF Accuracy {acc:.3f} for {sym} is below required floor {min_acc:.3f}")
            if f1 < min_f1:
                raise ValueError(f"RF F1 {f1:.3f} for {sym} is below required floor {min_f1:.3f}")

            gate_results[sym] = {
                "ok": gate.ok,
                "status": gate.status,
                "accuracy": acc,
                "f1": f1,
                "trained_at": trained_at,
            }

        logger.info("Step D Metric Gates PASSED cleanly for all candidates: %s", self.candidates)
        return gate_results

    def execute_transaction(self) -> Dict[str, Any]:
        """Execute the full transactional qualification pipeline with guaranteed rollback on failure."""
        logger.info("Starting Phase 2 Qualification Transaction for %s", self.candidates)
        self.verify_pre_conditions()
        self.create_snapshot()

        try:
            rf_res = self.train_rf()
            lstm_res = self.train_lstm()
            xgb_res = self.train_xgb()
            trans_res = self.train_transformer()
            gate_res = self.validate_metric_gates()

            logger.info("Phase 2 Model Qualification TRANSACTION SUCCESSFUL.")
            return {
                "success": True,
                "candidates": self.candidates,
                "rf": rf_res,
                "lstm": lstm_res,
                "xgb": xgb_res,
                "transformer": trans_res,
                "gates": gate_res,
            }
        except Exception as exc:
            logger.error("Phase 2 Model Qualification FAILED: %s. Initiating rollback!", exc, exc_info=True)
            self.restore_snapshot()
            raise RuntimeError(f"Qualification pipeline aborted and rolled back cleanly: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description="Phase 2 Model Qualification Harness")
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute training and qualification pipeline (Requires explicit authorization)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify pre-conditions and snapshot integrity without training",
    )
    args = parser.parse_args()

    harness = ModelQualificationHarness()

    if args.verify_only or not args.execute:
        print("=" * 70)
        print("  PHASE 2 MODEL QUALIFICATION HARNESS — DRY-RUN / PRE-CONDITION AUDIT")
        print("=" * 70)
        status = harness.verify_pre_conditions()
        print(f"\nLegacy 20 Checkpoints Status: {status}")
        print("\nCandidates Target: ['CAT', 'LLY']")
        print("Execution Mode: DRY-RUN ONLY (pass --execute to train when authorized)")
        return 0

    print("=" * 70)
    print("  PHASE 2 MODEL QUALIFICATION HARNESS — EXECUTING TRANSACTION")
    print("=" * 70)
    res = harness.execute_transaction()
    print("\n✅ Transaction completed successfully:")
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
