"""
model_metrics.py - Per-symbol model metrics persistence + the signal-safety gate.

Phase 1 of the live-trading readiness plan (reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md):
"truthful validation and signal safety".

What it does:
  * Training (ai_engine / lstm_engine) writes per-symbol metrics + a train date to
    models/model_metrics.json (merged, so RF and LSTM entries coexist).
  * predictor.predict_all reads them through `evaluate_gate(symbol)` BEFORE issuing
    any BUY/SELL. A symbol whose metrics are MISSING, STALE, or BELOW the accuracy
    floor is forced to HOLD with a human-readable reason.

Design:
  * Fail-closed: no trusted metric => no trade (returns ok=False).
  * Pure stdlib + config only (no IBKR, no network, no heavy ML imports) so the
    gate is fast and offline-testable.
  * Thresholds live in config (MODEL_MIN_RF_ACCURACY, MODEL_MIN_RF_F1,
    MODEL_MAX_AGE_DAYS) and the gate can be master-disabled with MODEL_GATE_ENABLED.

The capability flags below are read by the `live-readiness` scorecard. They are
True because this gate is implemented AND covered by test_model_gate.py - per the
plan, a SUPPORTS_* flag is only ever True when its logic genuinely exists.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import config

logger = logging.getLogger(__name__)

# -- Live-readiness capability flags (Phase 1) --------------------------------
SUPPORTS_MODEL_PERFORMANCE_GATE = True   # accuracy/F1 floor enforced in predict path
SUPPORTS_MODEL_STALENESS_GATE = True     # max-age enforced in predict path


@dataclass(frozen=True)
class GateResult:
    """Outcome of the signal-safety gate for one symbol."""
    ok: bool
    reason: str
    status: str   # "ok" | "missing" | "stale" | "below_threshold" | "disabled"


def _path() -> Path:
    """Resolve the metrics file from config at call time (so tests can patch it)."""
    return Path(config.MODEL_METRICS_FILE)


def load_all() -> dict:
    """Load the whole metrics document: {"rf": {SYM: {...}}, "lstm": {SYM: {...}}}.

    Returns an empty dict on any error / missing file (fail-closed: the gate then
    treats every symbol as 'missing').
    """
    try:
        path = _path()
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Could not read model metrics; treating as missing", exc_info=True)
        return {}


def _save_all(data: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _today_iso() -> str:
    return _dt.date.today().isoformat()


def save_rf_metrics(results: dict, trained_at: str | None = None) -> None:
    """Persist RF training metrics, merging with any existing LSTM entries.

    `results` is StockRFEngine.train()'s return value: {symbol: {test_acc, test_f1,
    oob_score, n_samples, ...}}. Never raises (persistence must not break training).
    """
    try:
        trained_at = trained_at or _today_iso()
        data = load_all()
        rf = data.setdefault("rf", {})
        for sym, m in (results or {}).items():
            rf[str(sym).upper()] = {
                "accuracy": _to_float(m.get("test_acc")),
                "f1": _to_float(m.get("test_f1")),
                "oob": _to_float(m.get("oob_score")),
                "n_samples": int(m.get("n_samples", 0) or 0),
                "trained_at": trained_at,
            }
        _save_all(data)
        logger.info("Persisted RF metrics for %d symbol(s) -> %s", len(results or {}), _path())
    except Exception:
        logger.warning("Could not persist RF model metrics", exc_info=True)


def save_lstm_metrics(results: dict, trained_at: str | None = None) -> None:
    """Persist LSTM training metrics, merging with any existing RF entries.

    `results` is StockLSTMEngine.train()'s return value: {symbol: {best_val_loss,
    n_train_seq, ...}}. Never raises.
    """
    try:
        trained_at = trained_at or _today_iso()
        data = load_all()
        lstm = data.setdefault("lstm", {})
        for sym, m in (results or {}).items():
            lstm[str(sym).upper()] = {
                "best_val_loss": _to_float(m.get("best_val_loss")),
                "n_train_seq": int(m.get("n_train_seq", 0) or 0),
                "trained_at": trained_at,
            }
        _save_all(data)
        logger.info("Persisted LSTM metrics for %d symbol(s) -> %s", len(results or {}), _path())
    except Exception:
        logger.warning("Could not persist LSTM model metrics", exc_info=True)


def metrics_for(symbol: str) -> dict:
    """Return {"rf": {...}|None, "lstm": {...}|None} for a symbol (read-only)."""
    sym = str(symbol).upper()
    data = load_all()
    return {
        "rf": (data.get("rf", {}) or {}).get(sym),
        "lstm": (data.get("lstm", {}) or {}).get(sym),
    }


def evaluate_gate(symbol: str, now: "_dt.date | None" = None) -> GateResult:
    """Decide whether `symbol` is safe to trade based on persisted model metrics.

    Order of checks (fail-closed):
      1. gate disabled         -> ok (explicitly opted out via config)
      2. no metrics at all     -> HOLD (status "missing")
      3. no usable train date  -> HOLD (status "missing")
      4. stale (age > max)     -> HOLD (status "stale")
      5. RF accuracy < floor   -> HOLD (status "below_threshold")
      6. RF F1 < floor (opt)   -> HOLD (status "below_threshold")
      7. otherwise             -> ok
    """
    if not bool(getattr(config, "MODEL_GATE_ENABLED", True)):
        return GateResult(True, "model gate disabled (MODEL_GATE_ENABLED=False)", "disabled")

    sym = str(symbol).upper()
    m = metrics_for(sym)
    rf, lstm = m["rf"], m["lstm"]

    if not rf and not lstm:
        return GateResult(False, f"no persisted model metrics for {sym} (run `train`)", "missing")

    # -- staleness on the freshest available train date --
    dates = [d for d in (_parse_date((rf or {}).get("trained_at")),
                         _parse_date((lstm or {}).get("trained_at"))) if d is not None]
    if not dates:
        return GateResult(False, f"{sym} model metrics have no train date", "missing")
    now = now or _dt.date.today()
    age_days = (now - max(dates)).days
    max_age = int(getattr(config, "MODEL_MAX_AGE_DAYS", 30))
    if age_days > max_age:
        return GateResult(
            False, f"{sym} model stale: trained {age_days}d ago > max {max_age}d", "stale"
        )

    # -- performance floor (RF accuracy / optional F1) --
    if rf is not None:
        floor = float(getattr(config, "MODEL_MIN_RF_ACCURACY", 0.50))
        acc = rf.get("accuracy")
        if acc is None or (isinstance(acc, float) and math.isnan(acc)):
            return GateResult(False, f"{sym} RF accuracy missing", "missing")
        if float(acc) < floor:
            return GateResult(
                False, f"{sym} RF accuracy {float(acc):.3f} < floor {floor:.3f}", "below_threshold"
            )
        f1_floor = float(getattr(config, "MODEL_MIN_RF_F1", 0.0))
        if f1_floor > 0:
            f1 = rf.get("f1")
            if f1 is None or (isinstance(f1, float) and math.isnan(f1)) or float(f1) < f1_floor:
                return GateResult(
                    False, f"{sym} RF F1 {f1} < floor {f1_floor:.3f}", "below_threshold"
                )

    return GateResult(True, "ok", "ok")


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _parse_date(value) -> "_dt.date | None":
    if not value:
        return None
    try:
        return _dt.date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None
