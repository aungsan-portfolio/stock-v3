"""
model_doctor.py - Model coverage & freshness diagnostics + a merge-preserving
model refresh for the exact candidates in reports/hot_candidates.csv.

Why this exists
---------------
The full-market scanner rotates to a different slice of symbols each run, but the
RF/LSTM models on disk were trained on an *earlier* scan. So the report's
candidates usually have no model, and predictor.py forces them to HOLD with
"ML models missing, forced HOLD" - daily-coach then finds no BUY candidates even
though the scan worked.

This module adds two read/maintenance commands. NEITHER connects to IBKR, places
orders, enables live trading, or changes any trading / prediction / risk logic:

  * model-doctor  : inspect RF/LSTM coverage + file freshness for report symbols,
                    print a summary and write reports/model_doctor_report.md.
  * model-refresh : train/update models for the exact report symbols, MERGING the
                    newly trained models into the existing model files (never
                    replacing the whole file with only today's symbols) and
                    creating timestamped backups first.

The model-file overwrite behaviour of StockRFEngine.train / StockLSTMEngine.train
(they save only the freshly trained batch) is intentionally NOT changed. The
refresh command merges around it: back up -> train -> merge new into existing ->
atomically re-save the merged store.
"""
import csv
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import joblib

import config
from ai_engine import StockRFEngine, _atomic_dump
from lstm_engine import (
    StockLSTMEngine,
    _atomic_torch_save,
    _atomic_npz_save,
    SCALERS_FILE,
    FEATURE_COLS,
)

logger = logging.getLogger(__name__)

# Model files older than this many days are flagged "stale". This is a reporting
# hint only - it never blocks trading and is not a safety setting.
STALE_MODEL_DAYS = 7

MODEL_DOCTOR_REPORT_FILE = config.REPORTS_DIR / "model_doctor_report.md"
MODEL_BACKUP_DIR = config.MODELS_DIR / "backups"


# -- Report candidates -------------------------------------------------------
def read_report_symbols(top_n: Optional[int] = None) -> Optional[List[str]]:
    """Read symbols (first column) from reports/hot_candidates.csv, in order.

    Returns None when the report file does not exist, [] when it is empty.
    Duplicates are removed while preserving first-seen order; when ``top_n`` is
    given only the first ``top_n`` symbols are returned (the report is sorted by
    hot_score descending, so this keeps the strongest candidates).
    """
    path = config.HOT_CANDIDATES_FILE
    if not path.exists():
        return None
    symbols: List[str] = []
    seen: Set[str] = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = (row.get("symbol") or "").strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                symbols.append(sym)
    if top_n is not None and top_n >= 0:
        symbols = symbols[:top_n]
    return symbols


# -- Model availability ------------------------------------------------------
def _file_mtime(path) -> Optional[float]:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def rf_available_symbols() -> Tuple[Set[str], Optional[float]]:
    """Symbols present in the RF model store, plus the file's mtime (or None)."""
    path = config.ML_MODELS_FILE
    if not path.exists():
        return set(), None
    try:
        models = joblib.load(path) or {}
        return set(models.keys()), _file_mtime(path)
    except Exception:
        logger.exception("Could not read RF model store %s", path)
        return set(), _file_mtime(path)


def lstm_available_symbols() -> Tuple[Set[str], Optional[float]]:
    """Symbols present in the LSTM checkpoint, plus the file's mtime (or None)."""
    path = config.LSTM_CKPT_FILE
    if not path.exists():
        return set(), None
    try:
        import torch

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        state_dicts = ckpt.get("state_dicts", {}) or {}
        return set(state_dicts.keys()), _file_mtime(path)
    except Exception:
        logger.exception("Could not read LSTM checkpoint %s", path)
        return set(), _file_mtime(path)


def _freshness(mtime: Optional[float]) -> Tuple[str, str]:
    """Return (timestamp_text, status_text) for a model-file mtime.

    Per-symbol training timestamps are not persisted, so freshness is reported at
    the model-file level. status is one of "fresh", "stale (>Nd)", or "unknown".
    """
    if mtime is None:
        return "unknown", "unknown"
    ts = datetime.fromtimestamp(mtime)
    age_days = (datetime.now() - ts).total_seconds() / 86400.0
    status = f"stale (>{STALE_MODEL_DAYS}d)" if age_days > STALE_MODEL_DAYS else "fresh"
    return f"{ts:%Y-%m-%d %H:%M} ({age_days:.1f}d old)", status


def inspect(symbols: List[str]) -> dict:
    """Coverage of ``symbols`` against the RF and LSTM stores."""
    rf_syms, rf_mtime = rf_available_symbols()
    lstm_syms, lstm_mtime = lstm_available_symbols()

    per_symbol = []
    for s in symbols:
        has_rf = s in rf_syms
        has_lstm = s in lstm_syms
        per_symbol.append({"symbol": s, "rf": has_rf, "lstm": has_lstm})

    rf_count = sum(1 for r in per_symbol if r["rf"])
    lstm_count = sum(1 for r in per_symbol if r["lstm"])
    both_count = sum(1 for r in per_symbol if r["rf"] and r["lstm"])
    # "missing" = neither model present -> predictor would force HOLD.
    missing = [r["symbol"] for r in per_symbol if not r["rf"] and not r["lstm"]]

    return {
        "total": len(symbols),
        "rf_count": rf_count,
        "lstm_count": lstm_count,
        "both_count": both_count,
        "missing": missing,
        "per_symbol": per_symbol,
        "rf_freshness": _freshness(rf_mtime),
        "lstm_freshness": _freshness(lstm_mtime),
    }


# -- model-doctor ------------------------------------------------------------
def _write_doctor_report(info: dict) -> "config.Path":
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rf_ts, rf_status = info["rf_freshness"]
    lstm_ts, lstm_status = info["lstm_freshness"]
    lines: List[str] = []
    lines.append("# Model Doctor Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.now():%Y-%m-%d %H:%M:%S}_")
    lines.append("")
    lines.append("Coverage of the candidates in `reports/hot_candidates.csv` against")
    lines.append("the trained RF/LSTM model stores. No IBKR connection was made and no")
    lines.append("orders were placed.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total candidates : {info['total']}")
    lines.append(f"- RF models available : {info['rf_count']}")
    lines.append(f"- LSTM models available : {info['lstm_count']}")
    lines.append(f"- Both models available : {info['both_count']}")
    lines.append(f"- Missing (neither model -> forced HOLD) : {len(info['missing'])}")
    lines.append("")
    lines.append("## Model file freshness")
    lines.append("")
    lines.append(f"- RF store   : {rf_ts}  ->  {rf_status}")
    lines.append(f"- LSTM store : {lstm_ts}  ->  {lstm_status}")
    lines.append("")
    lines.append("_Per-symbol training timestamps are not stored; freshness is")
    lines.append("reported at the model-file level._")
    lines.append("")
    if info["missing"]:
        lines.append("## Missing models")
        lines.append("")
        lines.append("These candidates have neither an RF nor an LSTM model, so the")
        lines.append("predictor forces them to HOLD:")
        lines.append("")
        lines.append("  " + ", ".join(info["missing"]))
        lines.append("")
        lines.append("Run `python -X utf8 main.py model-refresh --from-report --top-n 30`")
        lines.append("to train/update models for these exact candidates.")
        lines.append("")
    lines.append("## Per-candidate coverage")
    lines.append("")
    lines.append("| Symbol | RF | LSTM |")
    lines.append("|--------|----|------|")
    for r in info["per_symbol"]:
        lines.append(
            f"| {r['symbol']} | {'yes' if r['rf'] else 'no'} | "
            f"{'yes' if r['lstm'] else 'no'} |"
        )
    lines.append("")
    MODEL_DOCTOR_REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    return MODEL_DOCTOR_REPORT_FILE


def run_doctor(top_n: Optional[int] = None) -> int:
    """Inspect model coverage/freshness for the report candidates. No IBKR/orders."""
    print("\n=== Model Doctor (read-only; no IBKR, no orders) ===")
    symbols = read_report_symbols(top_n=top_n)
    if symbols is None:
        print(f"\nNo hot candidates file at {config.HOT_CANDIDATES_FILE}.")
        print("Run `python -X utf8 main.py scan-hot --full-market` (or daily-coach) first.")
        return 0
    if not symbols:
        print("\nHot candidates file is empty - re-run a scan first.")
        return 0

    info = inspect(symbols)
    rf_ts, rf_status = info["rf_freshness"]
    lstm_ts, lstm_status = info["lstm_freshness"]

    print(f"\nReport: {config.HOT_CANDIDATES_FILE}")
    print(f"  Total candidates       : {info['total']}")
    print(f"  RF models available     : {info['rf_count']}")
    print(f"  LSTM models available   : {info['lstm_count']}")
    print(f"  Both models available   : {info['both_count']}")
    print(f"  Missing (forced HOLD)   : {len(info['missing'])}")
    print("\nModel file freshness (file-level; per-symbol timestamps not stored):")
    print(f"  RF store   : {rf_ts}  ->  {rf_status}")
    print(f"  LSTM store : {lstm_ts}  ->  {lstm_status}")

    if info["missing"]:
        shown = info["missing"][:20]
        more = len(info["missing"]) - len(shown)
        tail = f", +{more} more" if more > 0 else ""
        print("\nMissing models for:")
        print("  " + ", ".join(shown) + tail)
        print(
            "\nCoverage is low. Run "
            "`python -X utf8 main.py model-refresh --from-report --top-n 30`"
        )
        print("to train/update models for these exact candidates.")
    else:
        print("\nAll candidates have at least one model. Coverage looks good.")

    report = _write_doctor_report(info)
    print(f"\nReport written to {report}")
    print("No IBKR connection was made and no orders were placed.")
    return 0


# -- model-refresh (merge-preserving) ----------------------------------------
def _backup_existing_models() -> List[str]:
    """Copy existing model files into models/backups/ with a timestamp suffix.

    Returns the list of backup paths created. Runs BEFORE any save so the prior
    model files are always recoverable.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    made: List[str] = []
    for src in (config.ML_MODELS_FILE, config.LSTM_CKPT_FILE, SCALERS_FILE):
        if src.exists():
            dst = MODEL_BACKUP_DIR / f"{src.name}.{stamp}.bak"
            shutil.copy2(src, dst)
            made.append(str(dst))
    return made


def _refresh_rf(targets: List[str]) -> Tuple[int, List[str], int]:
    """Train RF for ``targets`` and merge into the existing RF store.

    Returns (existing_count, newly_trained_symbols, final_count). The existing
    file may be transiently overwritten by engine.train (it saves only the new
    batch); we then re-save the merged dict so no previously trained symbol is
    lost. A backup was already taken by the caller.
    """
    existing: Dict = {}
    if config.ML_MODELS_FILE.exists():
        try:
            existing = joblib.load(config.ML_MODELS_FILE) or {}
        except Exception:
            logger.exception("Could not load existing RF store; treating as empty")
            existing = {}
    existing_count = len(existing)

    engine = StockRFEngine()
    engine.train(symbols=targets, verbose=False)
    newly = dict(engine.models)  # symbols trained this run (empty if all failed)

    merged = {**existing, **newly}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_dump(merged, config.ML_MODELS_FILE)
    return existing_count, sorted(newly.keys()), len(merged)


def _load_existing_lstm() -> Tuple[Dict, Dict, int]:
    """Load existing LSTM state_dicts, scalers, and input_size from disk."""
    state_dicts: Dict = {}
    input_size = len(FEATURE_COLS)
    if config.LSTM_CKPT_FILE.exists():
        try:
            import torch

            try:
                ckpt = torch.load(
                    config.LSTM_CKPT_FILE, map_location="cpu", weights_only=True
                )
            except TypeError:
                ckpt = torch.load(config.LSTM_CKPT_FILE, map_location="cpu")
            state_dicts = dict(ckpt.get("state_dicts", {}) or {})
            input_size = int(ckpt.get("input_size", input_size))
        except Exception:
            logger.exception("Could not load existing LSTM checkpoint; treating as empty")

    scalers: Dict = {}
    if SCALERS_FILE.exists():
        try:
            import numpy as np

            npz = np.load(SCALERS_FILE)
            syms = {k.rsplit("__", 1)[0] for k in npz.files if k.endswith("__mean")}
            for sym in syms:
                mean_key, std_key = f"{sym}__mean", f"{sym}__std"
                if mean_key in npz.files and std_key in npz.files:
                    scalers[sym] = (npz[mean_key], npz[std_key])
        except Exception:
            logger.exception("Could not load existing LSTM scalers; treating as empty")

    return state_dicts, scalers, input_size


def _refresh_lstm(targets: List[str]) -> Tuple[int, List[str], int]:
    """Train LSTM for ``targets`` and merge into the existing checkpoint.

    Returns (existing_count, newly_trained_symbols, final_count).
    """
    existing_states, existing_scalers, input_size = _load_existing_lstm()
    existing_count = len(existing_states)

    engine = StockLSTMEngine()
    engine.train(symbols=targets, verbose=False)
    new_states = {s: m.state_dict() for s, m in engine.models.items()}
    new_scalers = dict(engine.scalers)

    merged_states = {**existing_states, **new_states}
    merged_scalers = {**existing_scalers, **new_scalers}

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {
            "symbols": list(merged_states.keys()),
            "state_dicts": merged_states,
            "input_size": input_size,
        },
        config.LSTM_CKPT_FILE,
    )
    npz_payload: Dict = {}
    for sym, (mean, std) in merged_scalers.items():
        npz_payload[f"{sym}__mean"] = mean
        npz_payload[f"{sym}__std"] = std
    _atomic_npz_save(SCALERS_FILE, **npz_payload)
    return existing_count, sorted(new_states.keys()), len(merged_states)


def run_refresh(top_n: int = 30) -> int:
    """Train/update RF+LSTM for the exact report candidates, preserving existing
    models via merge + backup. No IBKR connection, no orders, no live trading.
    """
    print("\n=== Model Refresh (no IBKR, no orders, no live trading) ===")
    targets = read_report_symbols(top_n=top_n)
    if targets is None:
        print(f"\nNo hot candidates file at {config.HOT_CANDIDATES_FILE}.")
        print("Run `python -X utf8 main.py scan-hot --full-market` (or daily-coach) first.")
        return 0
    if not targets:
        print("\nHot candidates file is empty - re-run a scan first.")
        return 0

    print(f"Symbols requested ({len(targets)}): {', '.join(targets)}")

    backups = _backup_existing_models()
    if backups:
        print("\nBacked up existing model files before saving:")
        for b in backups:
            print(f"  {b}")
    else:
        print("\nNo existing model files to back up (fresh install).")

    print("\nTraining RF models (this may take a while)...")
    rf_existing, rf_new, rf_final = _refresh_rf(targets)
    print("Training LSTM models (this may take a while)...")
    lstm_existing, lstm_new, lstm_final = _refresh_lstm(targets)

    rf_failed = [s for s in targets if s not in set(rf_new)]
    lstm_failed = [s for s in targets if s not in set(lstm_new)]

    def _fmt(items: List[str]) -> str:
        return ", ".join(items) if items else "none"

    print("\n-- RF -------------------------------------------------")
    print(f"  Existing model count : {rf_existing}")
    print(f"  Symbols requested    : {len(targets)}")
    print(f"  Newly trained        : {len(rf_new)}  ({_fmt(rf_new)})")
    print(f"  Failed / skipped     : {len(rf_failed)}  ({_fmt(rf_failed)})")
    print(f"  Final model count    : {rf_final}")
    print("\n-- LSTM -----------------------------------------------")
    print(f"  Existing model count : {lstm_existing}")
    print(f"  Symbols requested    : {len(targets)}")
    print(f"  Newly trained        : {len(lstm_new)}  ({_fmt(lstm_new)})")
    print(f"  Failed / skipped     : {len(lstm_failed)}  ({_fmt(lstm_failed)})")
    print(f"  Final model count    : {lstm_final}")
    print(
        "\nExisting models were preserved (merged). No IBKR connection was made "
        "and no orders were placed."
    )
    return 0
