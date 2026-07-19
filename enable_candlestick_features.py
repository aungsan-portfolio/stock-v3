"""
enable_candlestick_features.py -- Candlestick feature activation + retrain.

This script:
  1. Backs up the current 14-feature models (models/ -> models_backup_before_candlestick/)
  2. Flips USE_CANDLESTICK_FEATURES = True in config.py
  3. Retrains RF + LSTM on the expanded 30-feature vector (base 14 + 16 candlestick)
  4. Prints a before/after comparison report

Usage:
  python enable_candlestick_features.py            # backup + enable + retrain
  python enable_candlestick_features.py --dry-run   # show what would happen, change nothing

After running, the model files in models/ will expect 30 features. To revert:
  1. Copy models_backup_before_candlestick/* back into models/
  2. Set USE_CANDLESTICK_FEATURES = False in config.py
"""
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Resolve paths BEFORE touching config so we can manipulate the flag file.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "config.py"
MODELS_DIR = BASE_DIR / "models"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = BASE_DIR / f"models_backup_before_candlestick_{TIMESTAMP}"

# Model files to back up
MODEL_FILES = [
    "rf_models.joblib",
    "lstm_checkpoint.pt",
    "lstm_checkpoint.scalers.npz",
    "model_metrics.json",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _backup_models(dry_run: bool) -> bool:
    """Copy current model files into a timestamped backup folder."""
    files_to_copy = [MODELS_DIR / f for f in MODEL_FILES if (MODELS_DIR / f).exists()]
    if not files_to_copy:
        print("  [SKIP] No model files found to back up.")
        return True

    if dry_run:
        print(f"  [DRY-RUN] Would back up {len(files_to_copy)} files -> {BACKUP_DIR.name}/")
        for f in files_to_copy:
            print(f"    {f.name}")
        return True

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for f in files_to_copy:
        dest = BACKUP_DIR / f.name
        shutil.copy2(f, dest)
        print(f"  Backed up {f.name} -> {BACKUP_DIR.name}/{f.name}")
    return True


def _read_config() -> str:
    return CONFIG_FILE.read_text(encoding="utf-8")


def _enable_flag_in_config(dry_run: bool) -> bool:
    """Set USE_CANDLESTICK_FEATURES = True in config.py."""
    text = _read_config()

    # Match the line: USE_CANDLESTICK_FEATURES = False  (with optional comment)
    pattern = r"^(USE_CANDLESTICK_FEATURES\s*=\s*)False(\s*#.*)?$"
    match = re.search(pattern, text, re.MULTILINE)

    if not match:
        # Check if already True
        if re.search(r"^USE_CANDLESTICK_FEATURES\s*=\s*True", text, re.MULTILINE):
            print("  [OK] USE_CANDLESTICK_FEATURES is already True.")
            return True
        print("  [ERROR] Could not find USE_CANDLESTICK_FEATURES line in config.py!")
        return False

    if dry_run:
        print("  [DRY-RUN] Would set USE_CANDLESTICK_FEATURES = True in config.py")
        return True

    # Replace False -> True, preserving the comment
    comment = match.group(2) or ""
    new_line = f"{match.group(1)}True {comment}".rstrip()
    new_text = text[:match.start()] + new_line + text[match.end():]

    CONFIG_FILE.write_text(new_text, encoding="utf-8")
    print("  [OK] USE_CANDLESTICK_FEATURES = True  (config.py updated)")
    return True


def _load_existing_metrics() -> dict:
    """Load the current model_metrics.json for before/after comparison."""
    metrics_file = MODELS_DIR / "model_metrics.json"
    if not metrics_file.exists():
        return {}
    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _retrain(dry_run: bool) -> dict:
    """Retrain RF + LSTM with the new 30-feature vector. Returns results dict."""
    if dry_run:
        print("  [DRY-RUN] Would retrain RF + LSTM with 30-feature vector.")
        return {}

    # Import AFTER config.py has been modified so the flag is live.
    # Force reimport of config + data_manager + engines to pick up the change.
    # Remove cached modules so Python re-reads them.
    for mod_name in list(sys.modules.keys()):
        if mod_name in (
            "config", "data_manager", "ai_engine", "lstm_engine",
            "eval_metrics", "model_metrics", "predictor", "errors",
        ):
            del sys.modules[mod_name]

    import config as cfg
    from data_manager import get_feature_columns
    from ai_engine import StockRFEngine
    from lstm_engine import StockLSTMEngine

    feature_cols = get_feature_columns()
    print(f"\n  Feature vector width: {len(feature_cols)} columns")
    print(f"  Candlestick flag    : {cfg.USE_CANDLESTICK_FEATURES}")
    if len(feature_cols) != 30:
        print(f"  [WARNING] Expected 30 features but got {len(feature_cols)}.")
        print(f"            (Regime={cfg.USE_REGIME_FEATURES}, "
              f"MarketRel={cfg.USE_MARKET_RELATIVE_FEATURES}, "
              f"Candle={cfg.USE_CANDLESTICK_FEATURES})")

    results = {}

    # -- RF --
    print("\n  === Training Random Forest models ===")
    rf_engine = StockRFEngine()
    rf_results = rf_engine.train(verbose=True)
    results["rf"] = rf_results
    if rf_results:
        print(f"\n  RF trained {len(rf_results)} symbol(s):")
        for sym, m in rf_results.items():
            auc = m.get("auc")
            auc_str = f"{auc:.4f}" if auc is not None else "n/a"
            print(f"    {sym:<6}  oob={m['oob_score']:.4f}  "
                  f"acc={m['test_acc']:.4f}  f1={m['test_f1']:.4f}  "
                  f"auc={auc_str}  n={m['n_samples']}")
    else:
        print("  [WARNING] No RF models were trained!")

    # -- LSTM --
    print("\n  === Training LSTM models ===")
    lstm_engine = StockLSTMEngine()
    lstm_results = lstm_engine.train(verbose=True)
    results["lstm"] = lstm_results
    if lstm_results:
        print(f"\n  LSTM trained {len(lstm_results)} symbol(s):")
        for sym, m in lstm_results.items():
            v_auc = m.get("best_val_auc")
            auc_str = f"{v_auc:.4f}" if v_auc is not None else "n/a"
            print(f"    {sym:<6}  val_loss={m['best_val_loss']:.4f}  "
                  f"val_auc={auc_str}  "
                  f"train_seq={m.get('n_train_seq', 0)}  "
                  f"val_seq={m.get('n_val_seq', 0)}")
    else:
        print("  [WARNING] No LSTM models were trained!")

    return results


def _print_comparison(before_metrics: dict, after_results: dict):
    """Print a before/after comparison table."""
    rf_after = after_results.get("rf", {})
    if not rf_after:
        return

    print("\n" + "=" * 70)
    print("  BEFORE / AFTER COMPARISON  (RF metrics)")
    print("=" * 70)
    print(f"  {'Symbol':<8} {'Metric':<12} {'Before (14-feat)':>18} {'After (30-feat)':>18}")
    print("  " + "-" * 60)

    rf_before = before_metrics.get("rf", {})
    for sym in sorted(rf_after.keys()):
        bm = rf_before.get(sym, {})
        am = rf_after[sym]
        for key in ("oob_score", "test_acc", "test_f1", "auc"):
            bv = bm.get(key)
            av = am.get(key)
            bv_str = f"{bv:.4f}" if bv is not None else "n/a"
            av_str = f"{av:.4f}" if av is not None else "n/a"
            delta = ""
            if bv is not None and av is not None:
                d = av - bv
                sign = "+" if d >= 0 else ""
                delta = f" ({sign}{d:.4f})"
            print(f"  {sym:<8} {key:<12} {bv_str:>18} {av_str:>18}{delta}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Enable candlestick features and retrain models (30-feature vector)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without making any changes.",
    )
    args = parser.parse_args()
    dry_run = args.dry_run

    print()
    print("=" * 70)
    print("  Candlestick Feature Activation" + (" (DRY RUN)" if dry_run else ""))
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)

    # Step 1: Snapshot current metrics for comparison
    print("\n[1/4] Loading current model metrics for comparison...")
    before_metrics = _load_existing_metrics()
    if before_metrics:
        print(f"  Found metrics for {len(before_metrics.get('rf', {}))} RF symbol(s)")
    else:
        print("  No existing metrics found (fresh install or first train).")

    # Step 2: Backup
    print("\n[2/4] Backing up current models...")
    if not _backup_models(dry_run):
        print("  [ABORT] Backup failed.")
        return 1

    # Step 3: Enable flag
    print("\n[3/4] Enabling USE_CANDLESTICK_FEATURES in config.py...")
    if not _enable_flag_in_config(dry_run):
        print("  [ABORT] Config update failed.")
        return 1

    # Step 4: Retrain
    print("\n[4/4] Retraining models with candlestick features...")
    after_results = _retrain(dry_run)

    # Comparison
    if not dry_run and after_results:
        _print_comparison(before_metrics, after_results)

    # Summary
    print("\n" + "=" * 70)
    if dry_run:
        print("  DRY RUN COMPLETE -- no changes were made.")
        print("  Run without --dry-run to activate candlestick features.")
    else:
        print("  DONE! Candlestick features are now ACTIVE.")
        print(f"  Feature vector: 14 -> 30 columns (16 candlestick patterns added)")
        print(f"  Model backup  : {BACKUP_DIR.name}/")
        print()
        print("  To revert:")
        print(f"    1. Copy {BACKUP_DIR.name}/* back to models/")
        print("    2. Set USE_CANDLESTICK_FEATURES = False in config.py")
    print("=" * 70)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
