"""
config.py — Stock Prediction Engine Configuration
Single source of truth — no magic numbers anywhere else.

Dual-mode:
  Module-level aliases (CAPITAL_NAMES) → backward-compatible global state.
  Settings dataclass → testable dependency injection via get_settings() / set_settings().
"""
from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# ── Paths ────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
_TEST_ROOT_ENV  = os.environ.get("STOCK_ENGINE_TEST_ROOT")
if _TEST_ROOT_ENV:
    _TEST_ROOT_PATH = Path(_TEST_ROOT_ENV).resolve()
    DATA_DIR        = _TEST_ROOT_PATH / "data"
    REPORTS_DIR     = _TEST_ROOT_PATH / "reports"
    LOG_DIR         = _TEST_ROOT_PATH / "logs"
else:
    DATA_DIR        = BASE_DIR / "data"
    REPORTS_DIR     = BASE_DIR / "reports"
    LOG_DIR         = BASE_DIR / "logs"

# Auto-load .env into os.environ if present (skipped during automated pytest runs)
if (BASE_DIR / ".env").exists() and not os.environ.get("PYTEST_CURRENT_TEST"):
    try:
        with open(BASE_DIR / ".env", "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip("'").strip('"'))
    except Exception:
        pass
MODELS_DIR      = BASE_DIR / "models"

ML_MODELS_FILE  = MODELS_DIR / "rf_models.joblib"
LSTM_CKPT_FILE  = MODELS_DIR / "lstm_checkpoint.pt"

# ── Kill Switch File Flags ────────────────────────
HALT_LOCK_FILE    = DATA_DIR / "HALT.lock"
FLATTEN_LOCK_FILE = DATA_DIR / "FLATTEN.lock"


def validate_test_isolation_root(root: Any) -> Path:
    """
    Validates that a candidate test isolation root does NOT overlap with
    production directories in either direction (ancestor, descendant, or equal).
    Raises ValueError on any violation BEFORE creating directories or modifying environment.
    """
    if root is None:
        raise ValueError("INVALID_TEST_ROOT: Root path cannot be None")

    if isinstance(root, (bytes, bytearray)):
        root_str = os.fsdecode(root)
    elif hasattr(root, "__fspath__"):
        fsp = root.__fspath__()
        root_str = os.fsdecode(fsp) if isinstance(fsp, (bytes, bytearray)) else str(fsp)
    else:
        root_str = str(root)

    resolved_root = Path(root_str).resolve()
    base_resolved = Path(BASE_DIR).resolve()

    # Reject if candidate equals or is ancestor of BASE_DIR
    if resolved_root == base_resolved or resolved_root in base_resolved.parents:
        raise ValueError(
            f"INVALID_TEST_ROOT: Candidate test root '{resolved_root}' equals or contains workspace base directory '{base_resolved}'"
        )

    # Check each protected production directory
    for pdir in [(base_resolved / "data").resolve(), (base_resolved / "logs").resolve(), (base_resolved / "reports").resolve()]:
        if resolved_root == pdir:
            raise ValueError(
                f"INVALID_TEST_ROOT: Candidate test root '{resolved_root}' equals protected production directory '{pdir}'"
            )
        if pdir in resolved_root.parents:
            raise ValueError(
                f"INVALID_TEST_ROOT: Candidate test root '{resolved_root}' is nested inside protected production directory '{pdir}'"
            )
        if resolved_root in pdir.parents:
            raise ValueError(
                f"INVALID_TEST_ROOT: Candidate test root '{resolved_root}' contains protected production directory '{pdir}'"
            )

    return resolved_root


def init_isolated_test_env(root: Path | str) -> Path:
    """Redirects all mutable runtime state, logs, and reports to an isolated test root."""
    global DATA_DIR, REPORTS_DIR, LOG_DIR, HALT_LOCK_FILE, FLATTEN_LOCK_FILE
    global FULL_MARKET_ROTATION_STATE_FILE, SYMBOL_UNIVERSE_FILE, HOT_CANDIDATES_FILE
    global SELECTED_SCAN_SYMBOLS_FILE, LOG_FILE, INCIDENT_DIR, BROKER_EVENTS_PATH
    global COACH_REPORT_FILE, SCAN_RESULTS_FILE, VALIDATION_LANE_LEDGER_PATH
    global DAYTRADE_LOG_FILE, DAYTRADE_TRADE_JOURNAL_FILE, DAYTRADE_ML_MODEL_PATH
    global SHADOW_BASE_DIR

    global EXPECTANCY_BACKTEST_TRADES, EXPECTANCY_REPORT_JSON, EXPECTANCY_REPORT_MD
    global FORWARD_TEST_JOURNAL, FORWARD_TEST_REPORT_JSON, FORWARD_TEST_REPORT_MD

    root_path = validate_test_isolation_root(root)
    os.environ["STOCK_ENGINE_TEST_ROOT"] = str(root_path)
    DATA_DIR = root_path / "data"
    REPORTS_DIR = root_path / "reports"
    LOG_DIR = root_path / "logs"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "incidents").mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    HALT_LOCK_FILE = DATA_DIR / "HALT.lock"
    FLATTEN_LOCK_FILE = DATA_DIR / "FLATTEN.lock"
    FULL_MARKET_ROTATION_STATE_FILE = DATA_DIR / "scan_rotation_state.json"
    SYMBOL_UNIVERSE_FILE = DATA_DIR / "symbol_universe.csv"
    HOT_CANDIDATES_FILE = REPORTS_DIR / "hot_candidates.csv"
    SELECTED_SCAN_SYMBOLS_FILE = REPORTS_DIR / "selected_scan_symbols.csv"
    LOG_FILE = LOG_DIR / "stock_engine.log"
    INCIDENT_DIR = LOG_DIR / "incidents"
    BROKER_EVENTS_PATH = LOG_DIR / "broker_events.jsonl"
    COACH_REPORT_FILE = REPORTS_DIR / "trade_coach_report.md"
    SCAN_RESULTS_FILE = REPORTS_DIR / "scan_results.csv"
    SHADOW_BASE_DIR = str(DATA_DIR / "shadow")
    VALIDATION_LANE_LEDGER_PATH = DATA_DIR / "paper_validation_trades.jsonl"
    DAYTRADE_LOG_FILE = LOG_DIR / "daytrading.log"
    DAYTRADE_TRADE_JOURNAL_FILE = REPORTS_DIR / "trade_journal.jsonl"
    DAYTRADE_ML_MODEL_PATH = DATA_DIR / "models" / "signal_predictor.pkl"
    EXPECTANCY_BACKTEST_TRADES = REPORTS_DIR / "backtest_trades.csv"
    EXPECTANCY_REPORT_JSON = REPORTS_DIR / "expectancy_metrics.json"
    EXPECTANCY_REPORT_MD = REPORTS_DIR / "expectancy_report.md"
    FORWARD_TEST_JOURNAL = REPORTS_DIR / "paper_trades.jsonl"
    FORWARD_TEST_REPORT_JSON = REPORTS_DIR / "forward_test_metrics.json"
    FORWARD_TEST_REPORT_MD = REPORTS_DIR / "forward_test_report.md"
    return root_path


# ── Production Write-Protection Tripwire ─────────────────────────────────────
class TestIsolationViolationError(BaseException):
    """Raised when test code attempts unauthorized write/mutation on protected production directories."""
    __test__ = False


SESSION_VIOLATIONS: List[dict] = []
_GUARD_INSTALLED: bool = False

_orig_builtins_open = None
_orig_io_open = None
_orig_os_open = None
_orig_os_replace = None
_orig_os_rename = None
_orig_os_remove = None
_orig_os_unlink = None
_orig_os_mkdir = None
_orig_os_makedirs = None
_orig_os_rmdir = None
_orig_path_open = None
_orig_path_write_text = None
_orig_path_write_bytes = None
_orig_path_unlink = None
_orig_path_mkdir = None
_orig_path_rmdir = None
_orig_path_rename = None
_orig_path_replace = None

_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | getattr(os, "O_APPEND", 0) | getattr(os, "O_TRUNC", 0)


def _normalize_target_path(target: Any) -> Optional[Path]:
    """Coerce str, bytes, Path, or os.PathLike to resolved Path. Returns None for int (file descriptors)."""
    if target is None or isinstance(target, int):
        return None
    try:
        if isinstance(target, (bytes, bytearray)):
            target = os.fsdecode(target)
        elif hasattr(target, "__fspath__"):
            fsp = target.__fspath__()
            target = os.fsdecode(fsp) if isinstance(fsp, (bytes, bytearray)) else str(fsp)
        else:
            target = str(target)
        return Path(target).resolve()
    except Exception:
        return None


def _is_protected_target(target: Any) -> bool:
    """
    Returns True if target resolves inside any protected production directory
    (data/, logs/, reports/). Coerces bytes, Path, str, PathLike.
    Fail-closed on any unresolvable target matching protected directory markers.
    """
    if target is None or isinstance(target, int):
        return False

    base_resolved = Path(BASE_DIR).resolve()
    protected_dirs = [
        (base_resolved / "data").resolve(),
        (base_resolved / "logs").resolve(),
        (base_resolved / "reports").resolve(),
    ]

    resolved = _normalize_target_path(target)
    if resolved is None:
        try:
            raw = os.fsdecode(target) if isinstance(target, (bytes, bytearray)) else str(target)
            raw_norm = raw.replace("\\", "/").lower()
            for seg in ("/data/", "/logs/", "/reports/", "data/", "logs/", "reports/"):
                if seg in raw_norm:
                    return True
        except Exception:
            return True
        return True  # Fail closed for unresolvable paths

    # Check active test root exemption (only if candidate root does NOT overlap production trees)
    test_root = os.environ.get("STOCK_ENGINE_TEST_ROOT")
    if test_root:
        try:
            resolved_test_root = _normalize_target_path(test_root)
            if resolved_test_root:
                overlaps = (resolved_test_root == base_resolved or resolved_test_root in base_resolved.parents)
                if not overlaps:
                    for pdir in protected_dirs:
                        if resolved_test_root == pdir or pdir in resolved_test_root.parents or resolved_test_root in pdir.parents:
                            overlaps = True
                            break
                if not overlaps:
                    if resolved == resolved_test_root or resolved_test_root in resolved.parents:
                        return False
        except Exception:
            pass

    for pdir in protected_dirs:
        if resolved == pdir or pdir in resolved.parents:
            return True
    return False


def _record_and_raise_violation(op: str, target: Any, details: str = ""):
    record = {
        "operation": op,
        "target": str(target),
        "details": details,
        "pid": os.getpid(),
    }
    SESSION_VIOLATIONS.append(record)
    msg = f"TEST ISOLATION VIOLATION [{op}]: Attempted unauthorized write/mutation on protected production path '{target}'. {details}".strip()
    raise TestIsolationViolationError(msg)


def _guarded_builtins_open(file, mode="r", *args, **kwargs):
    mode_str = str(mode)
    if any(c in mode_str for c in ("w", "a", "+", "x")):
        if _is_protected_target(file):
            _record_and_raise_violation("builtins.open", file, f"mode='{mode}'")
    return _orig_builtins_open(file, mode, *args, **kwargs)


def _guarded_io_open(file, mode="r", *args, **kwargs):
    mode_str = str(mode)
    if any(c in mode_str for c in ("w", "a", "+", "x")):
        if _is_protected_target(file):
            _record_and_raise_violation("io.open", file, f"mode='{mode}'")
    return _orig_io_open(file, mode, *args, **kwargs)


def _guarded_path_open(self, mode="r", *args, **kwargs):
    mode_str = str(mode)
    if any(c in mode_str for c in ("w", "a", "+", "x")):
        if _is_protected_target(self):
            _record_and_raise_violation("Path.open", self, f"mode='{mode}'")
    return _orig_path_open(self, mode, *args, **kwargs)


def _guarded_path_write_text(self, data, *args, **kwargs):
    if _is_protected_target(self):
        _record_and_raise_violation("Path.write_text", self)
    return _orig_path_write_text(self, data, *args, **kwargs)


def _guarded_path_write_bytes(self, data, *args, **kwargs):
    if _is_protected_target(self):
        _record_and_raise_violation("Path.write_bytes", self)
    return _orig_path_write_bytes(self, data, *args, **kwargs)


def _guarded_path_unlink(self, *args, **kwargs):
    if _is_protected_target(self):
        _record_and_raise_violation("Path.unlink", self)
    try:
        return _orig_path_unlink(self, *args, **kwargs)
    except PermissionError:
        if "pytest" in str(self).lower():
            return
        raise


def _guarded_path_mkdir(self, *args, **kwargs):
    if _is_protected_target(self):
        _record_and_raise_violation("Path.mkdir", self)
    return _orig_path_mkdir(self, *args, **kwargs)


def _guarded_path_rmdir(self, *args, **kwargs):
    if _is_protected_target(self):
        _record_and_raise_violation("Path.rmdir", self)
    return _orig_path_rmdir(self, *args, **kwargs)


def _guarded_path_rename(self, target, *args, **kwargs):
    if _is_protected_target(self):
        _record_and_raise_violation("Path.rename[src]", self, f"target='{target}'")
    if _is_protected_target(target):
        _record_and_raise_violation("Path.rename[dst]", target, f"src='{self}'")
    return _orig_path_rename(self, target, *args, **kwargs)


def _guarded_path_replace(self, target, *args, **kwargs):
    if _is_protected_target(self):
        _record_and_raise_violation("Path.replace[src]", self, f"target='{target}'")
    if _is_protected_target(target):
        _record_and_raise_violation("Path.replace[dst]", target, f"src='{self}'")
    return _orig_path_replace(self, target, *args, **kwargs)


def _guarded_os_open(path, flags, *args, **kwargs):
    if (flags & _WRITE_FLAGS) != 0:
        if _is_protected_target(path):
            _record_and_raise_violation("os.open", path, f"flags={flags}")
    return _orig_os_open(path, flags, *args, **kwargs)


def _guarded_os_replace(src, dst, *args, **kwargs):
    if _is_protected_target(src):
        _record_and_raise_violation("os.replace[src]", src, f"dst='{dst}'")
    if _is_protected_target(dst):
        _record_and_raise_violation("os.replace[dst]", dst, f"src='{src}'")
    return _orig_os_replace(src, dst, *args, **kwargs)


def _guarded_os_rename(src, dst, *args, **kwargs):
    if _is_protected_target(src):
        _record_and_raise_violation("os.rename[src]", src, f"dst='{dst}'")
    if _is_protected_target(dst):
        _record_and_raise_violation("os.rename[dst]", dst, f"src='{src}'")
    return _orig_os_rename(src, dst, *args, **kwargs)


def _guarded_os_remove(path, *args, **kwargs):
    if _is_protected_target(path):
        _record_and_raise_violation("os.remove", path)
    try:
        return _orig_os_remove(path, *args, **kwargs)
    except PermissionError:
        if "pytest" in str(path).lower():
            return
        raise


def _guarded_os_unlink(path, *args, **kwargs):
    if _is_protected_target(path):
        _record_and_raise_violation("os.unlink", path)
    try:
        return _orig_os_unlink(path, *args, **kwargs)
    except PermissionError:
        if "pytest" in str(path).lower():
            return
        raise


def _guarded_os_mkdir(path, *args, **kwargs):
    if _is_protected_target(path):
        _record_and_raise_violation("os.mkdir", path)
    return _orig_os_mkdir(path, *args, **kwargs)


def _guarded_os_makedirs(name, *args, **kwargs):
    if _is_protected_target(name):
        _record_and_raise_violation("os.makedirs", name)
    return _orig_os_makedirs(name, *args, **kwargs)


def _guarded_os_rmdir(path, *args, **kwargs):
    if _is_protected_target(path):
        _record_and_raise_violation("os.rmdir", path)
    return _orig_os_rmdir(path, *args, **kwargs)


def install_production_write_guard() -> None:
    """Installs fail-closed monkeypatches blocking any write/mutation to production data/, logs/, reports/."""
    global _GUARD_INSTALLED, _orig_builtins_open, _orig_io_open, _orig_os_open
    global _orig_os_replace, _orig_os_rename, _orig_os_remove, _orig_os_unlink
    global _orig_os_mkdir, _orig_os_makedirs, _orig_os_rmdir, _orig_path_open
    global _orig_path_write_text, _orig_path_write_bytes, _orig_path_unlink
    global _orig_path_mkdir, _orig_path_rmdir, _orig_path_rename, _orig_path_replace

    if _GUARD_INSTALLED:
        return
    import builtins
    import io

    _orig_builtins_open = builtins.open
    _orig_io_open = io.open
    _orig_os_open = os.open
    _orig_os_replace = os.replace
    _orig_os_rename = os.rename
    _orig_os_remove = os.remove
    _orig_os_unlink = getattr(os, "unlink", os.remove)
    _orig_os_mkdir = os.mkdir
    _orig_os_makedirs = os.makedirs
    _orig_os_rmdir = os.rmdir
    _orig_path_open = Path.open
    _orig_path_write_text = Path.write_text
    _orig_path_write_bytes = Path.write_bytes
    _orig_path_unlink = Path.unlink
    _orig_path_mkdir = Path.mkdir
    _orig_path_rmdir = Path.rmdir
    _orig_path_rename = Path.rename
    _orig_path_replace = Path.replace

    builtins.open = _guarded_builtins_open
    io.open = _guarded_io_open
    try:
        import _io
        _io.open = _guarded_io_open
    except Exception:
        pass

    Path.open = _guarded_path_open
    Path.write_text = _guarded_path_write_text
    Path.write_bytes = _guarded_path_write_bytes
    Path.unlink = _guarded_path_unlink
    Path.mkdir = _guarded_path_mkdir
    Path.rmdir = _guarded_path_rmdir
    Path.rename = _guarded_path_rename
    Path.replace = _guarded_path_replace

    os.open = _guarded_os_open
    os.replace = _guarded_os_replace
    os.rename = _guarded_os_rename
    os.remove = _guarded_os_remove
    os.unlink = _guarded_os_unlink
    os.mkdir = _guarded_os_mkdir
    os.makedirs = _guarded_os_makedirs
    os.rmdir = _guarded_os_rmdir

    _GUARD_INSTALLED = True


def uninstall_production_write_guard() -> None:
    """Removes production write-protection monkeypatches (used for cleanup or testing)."""
    global _GUARD_INSTALLED
    if not _GUARD_INSTALLED:
        return
    import builtins
    import io
    builtins.open = _orig_builtins_open
    io.open = _orig_io_open
    try:
        import _io
        _io.open = _orig_io_open
    except Exception:
        pass

    Path.open = _orig_path_open
    Path.write_text = _orig_path_write_text
    Path.write_bytes = _orig_path_write_bytes
    Path.unlink = _orig_path_unlink
    Path.mkdir = _orig_path_mkdir
    Path.rmdir = _orig_path_rmdir
    Path.rename = _orig_path_rename
    Path.replace = _orig_path_replace

    os.open = _orig_os_open
    os.replace = _orig_os_replace
    os.rename = _orig_os_rename
    os.remove = _orig_os_remove
    os.unlink = _orig_os_unlink
    os.mkdir = _orig_os_mkdir
    os.makedirs = _orig_os_makedirs
    os.rmdir = _orig_os_rmdir
    _GUARD_INSTALLED = False


# ── Stock Universe ───────────────────────────────
WATCHLIST = [
    "SPY", "QQQ", "VTI",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN",
    "GOOGL", "AVGO", "NFLX", "COST", "JPM", "BAC", "XOM", "UNH",
    # ── Step 3 Promoted (2026-08-05 Backtest Audit) ──────────────────────
    "XLV", "V",
]

# ── Shadow Monitor — signals logged, NO orders placed ────────────────────────
SHADOW_MONITOR = [
    "CAT", "LLY", "CVX",
]

# ── Hard-to-Borrow Short Blacklist ───────────────
HTB_SHORT_BLACKLIST = {
    "SPCX",   # Hard-to-borrow / illiquid
    "SPCH",   # Hard-to-borrow day order restriction
    "SNXX",   # Micro-cap illiquid
}

FULL_MARKET_SCAN_ENABLED        = True
FULL_MARKET_CACHE_HOURS         = 24
FULL_MARKET_MAX_SYMBOLS_TO_CHECK = 500
FULL_MARKET_SELECTION_MODE = "hybrid"
MAX_ACCEPTABLE_SLIPPAGE_PCT = 2.0
TIER1_SYMBOLS_SOURCE = "cached_json"
FULL_MARKET_CORE_SYMBOLS = [
    "SPY", "QQQ", "VTI",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "META", "AMZN",
    "GOOGL", "AVGO", "NFLX", "COST", "JPM", "BAC", "XOM", "UNH",
    "XLV", "V", "PLTR",
]
FULL_MARKET_RANDOM_SEED = 42
FULL_MARKET_ROTATION_STATE_FILE = DATA_DIR / "scan_rotation_state.json"
HOT_SCAN_TOP_N                  = 30
HOT_SCAN_MIN_PRICE              = 5.0
HOT_SCAN_MAX_PRICE              = 1000.0
HOT_SCAN_MIN_AVG_VOLUME         = 1_000_000
HOT_SCAN_EXCLUDE_ETFS           = False
HOT_SCAN_MAX_ATR_PCT            = 0.12
HOT_SCAN_CHUNK_SIZE             = 100
HOT_SCAN_SLEEP_SECONDS          = 0.2
NASDAQ_LISTED_URL  = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL   = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
SYMBOL_UNIVERSE_FILE = DATA_DIR / "symbol_universe.csv"
HOT_CANDIDATES_FILE  = REPORTS_DIR / "hot_candidates.csv"
SELECTED_SCAN_SYMBOLS_FILE = REPORTS_DIR / "selected_scan_symbols.csv"

# ── IPO / New Listing Watch (WATCH-ONLY) ─────────
IPO_WATCH_SYMBOLS = ["SPCX"]

# ── Data Settings ────────────────────────────────
PRICE_PERIOD        = "5y"
PRICE_INTERVAL      = "1d"
MIN_HISTORY_DAYS    = 240
CACHE_TTL_SECONDS   = 300

# ── Technical Indicators ─────────────────────────
SMA_SHORT           = 20
SMA_LONG            = 50
EMA_PERIOD          = 21
RSI_PERIOD          = 14
RSI_SMOOTHING       = "wilder"  # "wilder" (TradingView default) or "simple" (legacy SMA)
MACD_FAST           = 12
MACD_SLOW           = 26
MACD_SIGNAL         = 9
BOLLINGER_PERIOD    = 20
BOLLINGER_STD       = 2
ATR_PERIOD          = 14
VOLUME_MA_PERIOD    = 20

# ── Feature Flags ────────────────────────────────
USE_REGIME_FEATURES          = True
USE_MARKET_RELATIVE_FEATURES = True
MARKET_BENCHMARK_SYMBOL      = "SPY"
REGIME_VOL_SHORT             = 20
REGIME_VOL_LONG              = 60
REGIME_MOM_SHORT             = 63
REGIME_MOM_LONG              = 126
REGIME_RANK_WINDOW           = 126
REGIME_HIGH_WINDOW           = 252
REL_RET_SHORT                = 20
REL_RET_LONG                 = 63
RS_SLOPE_WINDOW              = 20
MKT_TREND_SMA                = 50
USE_CANDLESTICK_FEATURES            = True
CANDLESTICK_DOJI_THRESHOLD          = 0.10
CANDLESTICK_LONGBODY_THRESHOLD      = 0.60
CANDLESTICK_LONGSHADOW_MULTIPLIER   = 2.0
CANDLESTICK_SHORT_BODY_MULTIPLIER   = 0.5
USE_MICRO_FEATURES                  = False

# ── ML Settings ─────────────────────────────────
ML_WINDOW           = 20
ML_HORIZON          = 5
ML_TEST_RATIO       = 0.15
RANDOM_STATE        = 42
RF_N_ESTIMATORS     = 100
RF_MAX_DEPTH        = 6

# ── LSTM Settings ────────────────────────────────
LSTM_HIDDEN            = 64
LSTM_LAYERS            = 2
LSTM_DROPOUT           = 0.3
LSTM_EPOCHS            = 50
LSTM_BATCH             = 32
LSTM_LR                = 1e-3
LSTM_PATIENCE          = 7
LSTM_WINDOW            = 30
LSTM_BIDIRECTIONAL     = False
LSTM_GRAD_CLIP_NORM    = 1.0
LSTM_CYCLE_LR_STEP     = 0
LSTM_CYCLE_LR_BASE     = 1e-5
LSTM_HYPEROPT_TRIALS   = 0
LSTM_META_ENSEMBLE     = False

# ── Signal Thresholds ────────────────────────────
BUY_THRESHOLD       = 0.55
SELL_THRESHOLD      = 0.40
MIN_HOLD_BARS       = ML_HORIZON

# ── Ensemble Weights ─────────────────────────────
WEIGHT_RF           = 0.50
WEIGHT_LSTM         = 0.10
WEIGHT_XGB          = 0.10
WEIGHT_TRANSFORMER  = 0.00
WEIGHT_TECHNICAL    = 0.30
MIN_ML_MODELS_FOR_SIGNAL = 1

# ── Model Gate ───────────────────────────────────
MODEL_METRICS_FILE   = MODELS_DIR / "model_metrics.json"
MODEL_GATE_ENABLED   = True
MODEL_MIN_RF_ACCURACY   = 0.45
MODEL_MIN_RF_F1       = 0.35
MODEL_MAX_AGE_DAYS    = 30
MODEL_MIN_RF_AUC      = 0.0
MODEL_MIN_LSTM_AUC    = 0.40
MODEL_HOLDOUT_RATIO   = 0.15

# ── Permutation test ─────────────────────────────
PERMUTATION_N_SHUFFLES     = 50
PERMUTATION_SAMPLE_SYMBOLS = 20
PERMUTATION_FDR_Q          = 0.05

# ── Dynamic Market Texture Settings ────────────────
MARKET_REGIME_SETTINGS = {
    "TRENDING": {
        "noise_floor_ratio": 0.75,
        "confidence_min": 0.60,
        "tp_atr_mult": 2.0,
        "halt_new_entries": False,
        "allowed_strategies": ["ORB", "VWAP_BOUNCE", "GAP_AND_GO", "MOMENTUM_SCALP", "CANDLESTICK"]
    },

    "SIDEWAYS": {
        "noise_floor_ratio": 0.50,
        "confidence_min": 0.55,
        "tp_atr_mult": 0.9,
        "halt_new_entries": False,
        "allowed_strategies": ["MEAN_REVERSION", "VWAP_BOUNCE"]
    },
    "VOLATILE": {
        "noise_floor_ratio": 0.85,
        "confidence_min": 0.75,
        "tp_atr_mult": 2.5,
        "halt_new_entries": True
    },
    "TRANSITIONING": {
        "noise_floor_ratio": 0.75,
        "confidence_min": 0.60,
        "tp_atr_mult": 1.5,
        "halt_new_entries": False,
        "allowed_strategies": ["MEAN_REVERSION", "VWAP_BOUNCE", "ORB"]
    }
}

CAUTION_CONFIDENCE_BOOST    = 0.0  # Calibrated to 0.0 so CAUTION mode threshold stays at 0.60


# ── Risk Management ──────────────────────────────
MAX_POSITION_PCT    = 0.02
MAX_TRADE_VALUE     = 2000.0
MAX_OPEN_POSITIONS  = 7       # Capacity for up to 7 concurrent positions (Phase 2 Expansion)
PORTFOLIO_MAX_POSITIONS = MAX_OPEN_POSITIONS
MAX_CORRELATED_TECH_POSITIONS = 2  # Tightened from 4 to 2 to prevent over-concentrated tech beta
STOP_LOSS_PCT       = 0.008
TAKE_PROFIT_PCT     = 0.015

MIN_PROFIT_MARGIN   = 0.003
ESTIMATED_COMMISSION_PER_TRADE = 1.00
MIN_COMMISSION_PER_TRADE       = 1.00
COMMISSION_PER_SHARE           = 0.005
BACKTEST_COMMISSION_PER_TRADE  = 1.00
ATR_STOP_MULTIPLE              = 2.5  # DEPRECATED: VWAP_BOUNCE uses structural stop

# ── Labeling ─────────────────────────────────────
LABEL_MODE          = "binary"
LABEL_TP_PCT        = 0.015
LABEL_STOP_PCT      = 0.004
HARD_STOP_LOSS_PCT  = 0.03
USE_TRAILING_EXIT   = True
TRAILING_STOP_PCT   = 0.01

# ── Production Safety ────────────────────────────
ALLOW_SHORT          = True   # Enabled short selling for bearish market regimes
SHORT_HARD_STOP_PCT  = 0.01   # Hard Stop-Loss (+1.0% above entry) for short positions

# ── Inverse Bear ETF Gating & Hedging ────────────
INVERSE_ETF_SYMBOLS: set = {"SPDN", "SH", "SQQQ", "SOXS", "SDOW", "SPXU", "TZA", "TECS"}
INVERSE_ETF_RISK_OFF_EXEMPT: bool = True  # Allow buying inverse ETFs as hedges during RISK_OFF/VOLATILE
INVERSE_ETF_MAX_SIZE_PCT: float = 5.0     # Max position size (% of equity) for 1x unleveraged inverse ETF hedge
LEVERAGED_INVERSE_ETF_SYMBOLS: set = {"SQQQ", "SOXS", "SDOW", "SPXU", "TZA", "TECS"}
LEVERAGED_INVERSE_ETF_MAX_SIZE_PCT: float = 2.0  # Conservative 2.0% equity cap for 3x leveraged inverse ETFs
INVERSE_ETF_SCANNER_EXEMPT: bool = True   # Relaxed volume/rvol filters in scanner for inverse ETFs
INVERSE_ETF_MIN_AVG_DAILY_VOLUME: int = 200_000  # 200k daily volume floor for inverse ETFs
INVERSE_ETF_MIN_RVOL: float = 0.30        # 0.30 rvol floor for inverse ETFs
INVERSE_ETF_MIN_CONFIDENCE: float = 0.65  # Relaxed confidence floor for inverse ETF hedges
INVERSE_ETF_MIN_RR_RATIO: float = 1.40    # Relaxed R:R threshold for inverse ETFs (permits 1.45-1.49 setups)
INVERSE_ETF_BENCHMARK_MAP: dict = {
    "SQQQ": "QQQ",
    "PSQ": "QQQ",
    "SOXS": "SOXX",
    "TECS": "XLK",
    "SDOW": "DIA",
    "DOG": "DIA",
    "TZA": "IWM",
    "SPDN": "SPY",
    "SH": "SPY",
    "SPXU": "SPY",
}
MIN_TRADE_CASH       = 100.0
LIMIT_ORDER_OFFSET_PCT = 0.001
MAX_DAILY_TRADES     = 100

# ── Paper TradingStream Safety & Recovery ────────
TRADE_STREAM_START_TIMEOUT_SECONDS        = 10.0
TRADE_STREAM_STOP_TIMEOUT_SECONDS         = 8.0
TRADE_STREAM_RECONNECT_MAX_ATTEMPTS        = 8
TRADE_STREAM_RECONNECT_BASE_DELAY_SECONDS = float(os.environ.get("TRADE_STREAM_RECONNECT_BASE_DELAY_SECONDS", 2.0))
TRADE_STREAM_RECONNECT_MAX_DELAY_SECONDS  = float(os.environ.get("TRADE_STREAM_RECONNECT_MAX_DELAY_SECONDS", 30.0))
TRADE_STREAM_RECONNECT_JITTER_RATIO       = float(os.environ.get("TRADE_STREAM_RECONNECT_JITTER_RATIO", 0.20))

# ── Alpaca Network & HTTP Resilience ────────────
ALPACA_CONNECT_TIMEOUT_SECONDS            = float(os.environ.get("ALPACA_CONNECT_TIMEOUT", 15.0))
ALPACA_READ_TIMEOUT_SECONDS               = float(os.environ.get("ALPACA_READ_TIMEOUT", 30.0))
ALPACA_TOTAL_RETRIES                      = int(os.environ.get("ALPACA_TOTAL_RETRIES", 3))
ALPACA_RETRY_BACKOFF_FACTOR               = float(os.environ.get("ALPACA_RETRY_BACKOFF_FACTOR", 0.5))
ALPACA_CONNECT_MAX_ATTEMPTS               = int(os.environ.get("ALPACA_CONNECT_MAX_ATTEMPTS", 3))

# Swing Trading daily loss & protection configuration (Option C: Hard Dollar Cap mandatory)
SWING_MAX_DAILY_LOSS_PCT = float(os.environ.get("SWING_MAX_DAILY_LOSS_PCT", "0.01"))
SWING_MAX_DAILY_LOSS_DOLLAR_CAP = float(os.environ.get("SWING_MAX_DAILY_LOSS_DOLLAR_CAP", "300.00"))
SWING_PROTECTED_SYMBOLS = {
    s.strip().upper()
    for s in os.environ.get("SWING_PROTECTED_SYMBOLS", "SPY,QQQ").split(",")
    if s.strip()
}

if not 0 < SWING_MAX_DAILY_LOSS_PCT <= 0.10:
    raise ValueError("SWING_MAX_DAILY_LOSS_PCT must be between 0 and 0.10")
if SWING_MAX_DAILY_LOSS_DOLLAR_CAP <= 0:
    raise ValueError("SWING_MAX_DAILY_LOSS_DOLLAR_CAP must be greater than 0")

MAX_DAILY_LOSS_SWING   = float(os.environ.get("MAX_DAILY_LOSS_SWING", "150.0"))
MAX_DAILY_LOSS_DAYTRADE = 300.0  # Expanded limit for tonight per trader instruction ($300.00)
MAX_DAILY_LOSS_DOLLARS  = MAX_DAILY_LOSS_DAYTRADE
MAX_DAILY_LOSS_USD     = MAX_DAILY_LOSS_SWING
ALLOW_HISTORICAL_PRICE_FOR_ORDERS = False
COACH_ALLOW_HISTORICAL_PRICE = True

# ── Order Execution ──────────────────────────────
MIN_SHARE_COUNT            = 1      # Set to 1 per trader instruction (no 5-share minimum limit)
MIN_ORDER_SHARES           = 1
ORDER_FILL_TIMEOUT_SECONDS = 15.0
ORDER_POLL_SECONDS         = 0.25
CLOSE_MAX_ATTEMPTS         = 3
PROTECTIVE_TIF             = "GTC"
ACCOUNT_MAX_DRAWDOWN_PCT   = 0.10
MAX_SYMBOL_EXPOSURE_PCT    = 0.10


# ── Data Integrity & Health ───────────────────────────
REQUIRE_REALTIME_DATA_FOR_ORDERS = True
PAPER_MODE_AUTO_CONFIRM_FILL     = False  # C-02: Only real Alpaca TradingStream fills initialize trailing/lifecycle state
MAX_QUOTE_SPREAD_PCT             = 0.02
DECISION_PRICE_MAX_DEVIATION_PCT = 0.05
MAX_ACCEPTABLE_SPREAD_PCT        = 0.75   # 0.75% max spread (authoritative single source of truth)
MAX_DECISION_DEVIATION_PCT       = 1.5    # 1.5% max price deviation
MAX_QUOTE_AGE_SECONDS            = 10.0   # 10 seconds max quote staleness
MARKET_HOURS_GATE_ENABLED        = True

# ── P1.5.2 / P1.5.2.1 Bar Staleness & Reconciliation Settings ───
INTRADAY_BAR_SECONDS             = 300.0  # 5-minute default bar interval
ENTRY_BAR_CLOSE_GRACE_SECONDS    = float(os.environ.get("ENTRY_BAR_CLOSE_GRACE_SECONDS", "90.0"))   # 90s grace period post completed bar close (matches DAYTRADE_BAR_PUBLICATION_TIMEOUT_S)
BAR_CLOSE_GRACE_SECONDS          = ENTRY_BAR_CLOSE_GRACE_SECONDS   # Backward-compatible alias for entry grace
POSITION_ATR_MAX_AGE_SECONDS     = 360.0  # 360s max age for active-position ATR updates during mid-bar polling
PREFLIGHT_MAX_RECOVERABLE_DELAY_SECONDS = float(os.environ.get("PREFLIGHT_MAX_RECOVERABLE_DELAY_SECONDS", "360.0"))  # 360s (300s bar + 60s grace) max recoverable delay for preflight startup audit
BAR_FUTURE_TOLERANCE_SECONDS     = 5.0    # 5s clock skew tolerance for completed bars
QUOTE_FUTURE_TOLERANCE_SECONDS   = float(os.environ.get("QUOTE_FUTURE_TOLERANCE_SECONDS", "2.0"))  # 2s clock skew tolerance for real-time quotes
INTRADAY_DATA_SOURCE_TIMEZONE    = "US/Eastern" # Default fallback timezone for naive bar feeds
TRAILING_STOP_RECONCILE_TOLERANCE = None  # None defaults dynamically to tick_size / 2.0

# ── Sprint #1: Multi-Symbol Batch Fetch Settings ──
DAYTRADE_BATCH_FETCH_ENABLED        = os.environ.get("DAYTRADE_BATCH_FETCH_ENABLED", "true").lower() in ("true", "1")
DAYTRADE_BATCH_FETCH_MAX_CHUNK      = int(os.environ.get("DAYTRADE_BATCH_FETCH_MAX_CHUNK", "25"))
DAYTRADE_BATCH_FETCH_TIMEOUT_S      = float(os.environ.get("DAYTRADE_BATCH_FETCH_TIMEOUT_S", "6.0"))
DAYTRADE_BATCH_FETCH_MIN_COVERAGE   = float(os.environ.get("DAYTRADE_BATCH_FETCH_MIN_COVERAGE", "0.95"))
DAYTRADE_STARTUP_CADENCE_GRACE_S    = float(os.environ.get("DAYTRADE_STARTUP_CADENCE_GRACE_S", "60.0"))
MAX_ALLOWED_CHASE                   = float(os.environ.get("MAX_ALLOWED_CHASE", "0.15"))
DAYTRADE_BAR_PUBLICATION_TIMEOUT_S        = float(os.environ.get("DAYTRADE_BAR_PUBLICATION_TIMEOUT_S", "90.0"))
DAYTRADE_BAR_PUBLICATION_POLL_INTERVAL_S  = float(os.environ.get("DAYTRADE_BAR_PUBLICATION_POLL_INTERVAL_S", "5.0"))
DAYTRADE_REQUIRE_ENTRY_READINESS          = os.environ.get("DAYTRADE_REQUIRE_ENTRY_READINESS", "true").lower() in ("true", "1")


# ── Latency Remediation & Resilient Recovery Settings ──
DAYTRADE_RECOVERY_MAX_WORKERS          = int(os.environ.get("DAYTRADE_RECOVERY_MAX_WORKERS", "4"))
DAYTRADE_SYMBOL_FETCH_TIMEOUT_SECONDS   = float(os.environ.get("DAYTRADE_SYMBOL_FETCH_TIMEOUT_SECONDS", "2.5"))
DAYTRADE_RECOVERY_MAX_CANDIDATES       = int(os.environ.get("DAYTRADE_RECOVERY_MAX_CANDIDATES", "31"))
DAYTRADE_RECOVERY_TOTAL_TIMEOUT_SECONDS = float(os.environ.get("DAYTRADE_RECOVERY_TOTAL_TIMEOUT_SECONDS", "8.0"))
ALPACA_DATA_CONNECT_TIMEOUT            = float(os.environ.get("ALPACA_DATA_CONNECT_TIMEOUT", "3.5"))
ALPACA_DATA_READ_TIMEOUT               = float(os.environ.get("ALPACA_DATA_READ_TIMEOUT", "7.5"))



DATA_HEALTH_SETTINGS = {
    "critical_symbols": ["SPY", "QQQ", "SOXX"],
    "stale_threshold_s": {
        "SPY": 60.0,
        "QQQ": 60.0,
        "SOXX": 60.0,
        "_default": 180.0,
    },
    "recovery_bars": 3,
}

# ── Backtest ─────────────────────────────────────
BACKTEST_TRANSACTION_COST_PCT = 0.0005
BACKTEST_SLIPPAGE_FRAC        = 0.0005  # Fractional decimal (0.0005 = 0.05% or 5 bps). DO NOT divide by 100.
BACKTEST_SLIPPAGE_PCT         = BACKTEST_SLIPPAGE_FRAC  # DEPRECATED alias: value is a FRACTION (e.g. 0.0005 = 0.05%), DO NOT divide by 100
BACKTEST_INCLUDE_LSTM         = False
BACKTEST_LSTM_EPOCHS          = 8
BACKTEST_LSTM_BATCH           = 64
BACKTEST_PURGE_GAP            = ML_HORIZON  # Purge gap between training cutoff and test start to eliminate forward label leakage

# ── IBKR (Deprecated for compatibility) ──────────
IBKR_HOST           = "127.0.0.1"
IBKR_PORT           = 7497
PAPER_IBKR_PORT     = 7497
REQUIRE_PAPER_PORT  = True
CLIENT_ID_BOT       = 1
CLIENT_ID_CHECK     = 11
CLIENT_ID_CANCEL    = 12
CLIENT_ID_FLATTEN   = 13
IBKR_CLIENT_ID      = CLIENT_ID_BOT
PAPER_CAPITAL       = 100_000.0
IBKR_MARKET_DATA_TYPE = 3

# ── Alpaca ───────────────────────────────────────
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_FEED      = "iex"

# Chunked batch fetch & data feed settings
ALPACA_CHUNK_SIZE: int = int(os.environ.get("ALPACA_CHUNK_SIZE", 5))
ALPACA_CHUNK_DELAY_MS: int = int(os.environ.get("ALPACA_CHUNK_DELAY_MS", 200))
ALPACA_CHUNK_TIMEOUT_SECONDS: float = float(os.environ.get("ALPACA_CHUNK_TIMEOUT_SECONDS", 2.0))
ALPACA_SUBSCRIPTION_TIER: str = os.environ.get("ALPACA_SUBSCRIPTION_TIER", "free").lower().strip()
REQUIRE_FEED_CONCORDANCE: bool = os.environ.get("REQUIRE_FEED_CONCORDANCE", "false").lower().strip() in ("true", "1", "yes")
MAX_BAR_STALENESS_SECONDS: float = float(os.environ.get("MAX_BAR_STALENESS_SECONDS", 300.0))
ENTRY_BAR_DURATION_SECONDS: float = float(os.environ.get("ENTRY_BAR_DURATION_SECONDS", 300.0))
ENTRY_BAR_CLOSE_GRACE_SECONDS: float = float(os.environ.get("ENTRY_BAR_CLOSE_GRACE_SECONDS", 90.0))

# WebSocket keepalive & connection management
ALPACA_WS_PING_INTERVAL_SECONDS: int = int(os.environ.get("ALPACA_WS_PING_INTERVAL_SECONDS", 20))
ALPACA_WS_PING_TIMEOUT_SECONDS: int = int(os.environ.get("ALPACA_WS_PING_TIMEOUT_SECONDS", 60))
ALPACA_WS_MAX_QUEUE: int = int(os.environ.get("ALPACA_WS_MAX_QUEUE", 1024))

# Zero-budget feed & fail-closed risk safety flags
ALLOW_VIX_PROXY_FALLBACK: bool = os.environ.get("ALLOW_VIX_PROXY_FALLBACK", "false").lower().strip() in ("true", "1", "yes")
SOXX_GATE_FAIL_CLOSED: bool = os.environ.get("SOXX_GATE_FAIL_CLOSED", "true").lower().strip() in ("true", "1", "yes")


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _require_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive float, got {value!r}")
    return float(value)


def validate_alpaca_runtime_config() -> None:
    """
    Validates Alpaca daytrade runtime configuration.
    Must only be called explicitly during bot startup, NEVER at module import time.
    """
    _require_positive_int("ALPACA_CHUNK_SIZE", ALPACA_CHUNK_SIZE)
    _require_non_negative_int("ALPACA_CHUNK_DELAY_MS", ALPACA_CHUNK_DELAY_MS)
    _require_positive_float("ALPACA_CHUNK_TIMEOUT_SECONDS", ALPACA_CHUNK_TIMEOUT_SECONDS)
    tier = str(ALPACA_SUBSCRIPTION_TIER).lower().strip()
    if tier not in ("free", "unlimited"):
        raise ValueError(f"ALPACA_SUBSCRIPTION_TIER must be 'free' or 'unlimited', got {ALPACA_SUBSCRIPTION_TIER!r}")
    _require_positive_int("ALPACA_WS_PING_INTERVAL_SECONDS", ALPACA_WS_PING_INTERVAL_SECONDS)
    _require_positive_int("ALPACA_WS_PING_TIMEOUT_SECONDS", ALPACA_WS_PING_TIMEOUT_SECONDS)
    _require_positive_int("ALPACA_WS_MAX_QUEUE", ALPACA_WS_MAX_QUEUE)
    validate_institutional_safety_config()


def validate_institutional_safety_config() -> None:
    """Validates parameters for cadence, publication, coverage, and chase protection."""
    import sys
    cfg = sys.modules.get("config", sys.modules[__name__])
    pub_timeout = float(getattr(cfg, "DAYTRADE_BAR_PUBLICATION_TIMEOUT_S", 90.0))
    poll_interval = float(getattr(cfg, "DAYTRADE_BAR_PUBLICATION_POLL_INTERVAL_S", 5.0))
    coverage = float(getattr(cfg, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95))
    max_chase = float(getattr(cfg, "MAX_ALLOWED_CHASE", 0.15))

    if pub_timeout <= 0:
        raise ValueError(f"DAYTRADE_BAR_PUBLICATION_TIMEOUT_S must be positive, got {pub_timeout}")
    if not (0 < poll_interval <= 30):
        raise ValueError(f"DAYTRADE_BAR_PUBLICATION_POLL_INTERVAL_S must be in (0, 30], got {poll_interval}")
    if poll_interval >= pub_timeout:
        raise ValueError(f"DAYTRADE_BAR_PUBLICATION_POLL_INTERVAL_S ({poll_interval}) must be less than timeout ({pub_timeout})")
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"DAYTRADE_BATCH_FETCH_MIN_COVERAGE must be in (0.0, 1.0], got {coverage}")
    if max_chase <= 0:
        raise ValueError(f"MAX_ALLOWED_CHASE must be positive, got {max_chase}")



# ── Logging ──────────────────────────────────────
LOG_FILE            = LOG_DIR / "stock_engine.log"
LOG_MAX_BYTES       = 10 * 1024 * 1024
LOG_BACKUP_COUNT    = 5

# ── Observability & Incident Telemetry ────────────
INCIDENT_DIR            = LOG_DIR / "incidents"
FLIGHT_RECORDER_CAPACITY = 500
LATCH_HEARTBEAT_SECONDS = 300
if _TEST_ROOT_ENV:
    BROKER_EVENTS_PATH = LOG_DIR / "broker_events.jsonl"
else:
    BROKER_EVENTS_PATH = Path(os.environ.get("BROKER_EVENTS_PATH", str(LOG_DIR / "broker_events.jsonl")))


# ── Coach ────────────────────────────────────────
COACH_MIN_CONFIDENCE_FOR_CANDIDATE = 0.60
COACH_REQUIRE_CHART_CHECK          = True
COACH_REQUIRE_USER_CONFIRM         = True
COACH_MAX_NEW_TRADES_PER_RUN       = 2
COACH_REPORT_FILE                  = REPORTS_DIR / "trade_coach_report.md"
COACH_SKIP_IF_POSITION_OR_WORKING  = True
COACH_MAX_PAPER_TRADES_PER_RUN     = 2
COACH_DEFAULT_PREVIEW_CANDIDATES   = 1
COACH_LIVE_TRADING_ENABLED         = False

# ── Day-trade Practice ───────────────────────────
import datetime as _dt

PREMARKET_START = _dt.time(4, 0)
MARKET_OPEN = _dt.time(9, 30)
MARKET_CLOSE = _dt.time(16, 0)
POSTMARKET_END = _dt.time(20, 0)
TIMEZONE = "US/Eastern"

# Autonomous EOD Flatten Buffer (seconds before 16:00 ET close, default 120s -> 15:58:00 ET)
EOD_FLATTEN_BUFFER_SECONDS: float = float(os.getenv("EOD_FLATTEN_BUFFER_SECONDS", "120.0"))

ORB_WINDOW_MINUTES = 15
ORB_EXTENDED_MINUTES = 30

ENFORCE_GLOBAL_OPENING_BUFFER = True
GLOBAL_OPENING_BUFFER_END_ET = "09:45"

# Manual Denylist for scanner / trading exclusion
SYMBOL_DENYLIST: list = [
    "TTD",
    # Ultra-high volatility Leveraged Bull ETFs Denylist (high slippage & whipsaw risk)
    "SOXL", "TSLL", "TQQQ", "SPXL",
    "LABU", "LABD", "UVXY", "SVXY", "BOIL", "KOLD", "NUGT", "DUST",
    # Untradeable low ATR / stale quote inverse ETFs
    "PSQ", "DOG", "QID",
    # Micro-cap / Low-price exclusion denylist
    "DRAM", "SKHY", "PATH", "CDE", "NU", "SOFI", "PLTD", "BTG", "SNXX", "DOCS", "SPCH",
    # Stale quote / illiquid / extreme discontinuity symbols (Aug 2026 Audit)
    "EDSA", "MRAL", "EDHL", "CRDX", "HVII", "TNON", "XXRP", "GXRP", "XRPT", "UMAL",
    "BTDL", "WETO", "ETHD", "ZSTK", "STKH", "PCLA", "DBGI"
]

INTRADAY_INTERVAL = "1m"
INTRADAY_LOOKBACK_DAYS = 5
DAILY_LOOKBACK_DAYS = 60
PREMARKET_INTERVAL = "1m"
INTRADAY_CACHE_TTL_SECONDS = 15  # 15s TTL (reduced from 300s) to guarantee fresh 5m bars on every cycle
STALE_QUOTE_BUFFER_SECONDS = 30

SCAN_MIN_PRICE = 5.0  # $5.0+ price floor (lowered from $10 for expanded paper candidate discovery)
SCAN_MAX_PRICE = 1000.0
SCAN_MIN_PREMARKET_VOLUME = 10_000  # 10k premarket volume floor (lowered from 50k to capture early premarket movers)
SCAN_MIN_AVG_DAILY_VOLUME = 1_000_000  # 1.0M+ volume floor for Large-Cap / Mid-Cap focus
LARGE_CAP_PIVOT_SYMBOLS = ["PLTR", "NVDA", "AMD", "TSLA", "MSFT", "AMZN", "META", "AAPL", "GOOGL", "IBIT", "NKE", "COIN", "MARA", "SPDN", "SH", "SQQQ", "SOXS", "SDOW", "SPXU", "TZA", "TECS"]
SCAN_PREMARKET_MAX_SPREAD_PCT = 0.008  # 0.8% max spread in premarket
SCAN_REGULAR_MAX_SPREAD_PCT   = 0.008  # 0.8% max spread in regular session
SCAN_MIN_RELATIVE_VOLUME = 0.80  # 0.80 minimum relative volume for high-quality catalyst filter
SCAN_MIN_GAP_PCT = 1.0  # 1.0% (lowered from 1.4%)
SCAN_MAX_GAP_PCT = 15.0
SCAN_MIN_FLOAT_SHARES = 10_000_000
SCAN_MAX_CANDIDATES = 5
SCAN_CHUNK_SIZE = 50
SCAN_SLEEP_SECONDS = 1.0
SCANNER_REFRESH_MINUTES = 10

# ── Noise Floor — ATR-Based Veto Calibration ────────
NOISE_FLOOR_ATR_PERIOD       = 10    # ATR window (10 for responsiveness)
NOISE_FLOOR_ATR_MULTIPLIER   = 0.75  # Canonical shared geometry ATR floor multiplier
NOISE_FLOOR_ABS_MIN          = 0.08  # minimum floor $0.08
NOISE_FLOOR_HARD_VETO_RATIO  = 0.75  # Legacy Option B only; Option A enforces the full floor
MAX_RISK_PER_TRADE_DOLLARS   = 50.0  # Max per-trade dollar risk cap (Calibrated to $50 for Phase 2 7-slot allocation)
MAX_TARGET_RR_RATIO           = 5.0   # Target reachability ceiling (5.0x risk per share)
MAX_TARGET_DAILY_ATR_MULTIPLE = 3.5   # Target ceiling multiple vs Daily ATR (if daily_atr present)

# ── Short Counter-Trend Veto Rules in RISK_ON ──────
APPLY_SHORT_VETO_TO_ALL_SYMBOLS = True
SHORT_VETO_RSI_THRESHOLD        = 40.0

# ── Earnings Calendar Blackout (Oct 21, 2026: META, TSLA after-close) ────
EARNINGS_CALENDAR = {
    "META": "2026-10-21",
    "TSLA": "2026-10-21",
}
EARNINGS_BLACKOUT_DAYS = 1

# ── Gap Quality Filter Thresholds ───────────────────
GAP_MIN_RVOL          = 1.5   # Require Relative Volume >= 1.5x (Institutional Conviction)
GAP_MIN_ATR_MULTIPLE  = 0.5   # Require Gap Dollar Magnitude >= 0.5x ATR(14)
GAP_ATR_PERIOD        = 14

MAX_SYMBOL_CONSECUTIVE_LOSSES = 2
MAX_ENTRIES_PER_SYMBOL_PER_SESSION = 2

# Blacklist illiquid / stale quote symbols to prevent noisy intraday data warnings
SYMBOL_STALE_QUOTE_BLACKLIST = [
    # Original batch
    "PFAI", "OSCX", "SOUX", "CISS",
    "PAVS", "XCH", "UNX", "FXHO",
    "BIVI", "FIGG", "CSEX", "YXT", "ATNI",
    # 2026-08-07 batch — scanner stale 35+ min
    "SMTG", "OLOX", "GWAV",
    # IPO / watch-only (stale quotes expected)
    "SPCX",
    # 2026-08-21 batch — persistent IEX thin-tape stale (>300s repeatedly)
    "HOOX", "ELOX", "TXXS", "EUDA", "COIG", "COIA",
    "DFNS", "BAFN", "RFAI", "HUTG", "GEMG",
    "CIFG", "MFI", "AEHG", "SMST", "UXRP",
    "ROBN", "SBIT", "CONI", "SDOT",
    "CRMX", "KEEX", "PUR", "JUNS", "SUIL",
    "APLX", "USAX", "USGG", "MRNX", "BABX",
    # NOTE: TECS, SPXU, SDOW are intentionally omitted from this blacklist
    # because they belong to INVERSE_ETF_SYMBOLS used for RISK_OFF portfolio hedging.
]

DYNAMIC_SCREENER_ENABLED = True
DYNAMIC_SCREENER_TARGET_SYMBOLS = 150
DYNAMIC_SCREENER_CACHE_TTL = 120
SCAN_SCORE_GAP_WEIGHT = 0.4
SCAN_SCORE_RVOL_WEIGHT = 0.3
SCAN_SCORE_VOL_WEIGHT = 0.3
SCAN_SCORE_RVOL_CAP = 10.0
SCAN_SCORE_VOL_NORMALIZER = 100_000
SCAN_OPEN_MIN_MOVE_PCT = 0.10
SCAN_OPEN_MIN_VOLUME = 50_000
SCAN_OPEN_MIN_RELATIVE_VOLUME = 0.80

SCAN_RESULTS_FILE = REPORTS_DIR / "scan_results.csv"

VWAP_BOUNCE_TOLERANCE_PCT = 0.3
VWAP_RECLAIM_BARS = 3

EMA_9 = 9
EMA_20 = 20
EMA_50 = 50
# Day-trade technical indicator settings
VWAP_STD_BANDS = 2

STRATEGY_ORB_ENABLED = False
STRATEGY_VWAP_BOUNCE_ENABLED = False  # Disabled per Governance Audit 2026-09-26 (Loss Containment: combined PF 0.05, N=65)
STRATEGY_GAP_AND_GO_ENABLED = True
STRATEGY_MOMENTUM_SCALP_ENABLED = False  # Disabled per Governance Audit 2026-09-26 (Loss Containment: combined PF 0.28, N=32)
STRATEGY_CANDLESTICK_ENABLED = True

# Canonical aliases
VWAP_BOUNCE_ENABLED: bool = STRATEGY_VWAP_BOUNCE_ENABLED
MOMENTUM_SCALP_ENABLED: bool = STRATEGY_MOMENTUM_SCALP_ENABLED


@dataclass
class StrategyConfig:
    confidence_min: float = 0.65
    max_risk_pct: float = 1.0
    rr_ratio: float = 2.0
    priority: int = 10
    entry_cutoff_time: Optional[str] = None

    def __post_init__(self):
        from datetime import datetime
        self.cutoff_time_obj = None
        if self.entry_cutoff_time:
            try:
                self.cutoff_time_obj = datetime.strptime(self.entry_cutoff_time, "%H:%M").time()
            except Exception:
                pass

from strategies.constants import StrategyName
STRATEGY_SETTINGS = {
    StrategyName.ORB: StrategyConfig(confidence_min=0.55, max_risk_pct=1.0, priority=1),
    StrategyName.MEAN_REVERSION: StrategyConfig(confidence_min=0.50, max_risk_pct=1.0, priority=2),
    StrategyName.VWAP_BOUNCE: StrategyConfig(confidence_min=0.50, max_risk_pct=1.0, priority=3),
    StrategyName.MOMENTUM_SCALP: StrategyConfig(confidence_min=0.58, max_risk_pct=1.0, priority=4),
    StrategyName.GAP_AND_GO: StrategyConfig(confidence_min=0.55, max_risk_pct=1.0, priority=5, entry_cutoff_time="10:30"),
    StrategyName.CANDLESTICK: StrategyConfig(confidence_min=0.40, max_risk_pct=0.5, priority=6),
}

STRATEGY_MEAN_REVERSION_ENABLED = True  # Native high-win-rate strategy for SIDEWAYS regimes

# =============================================================================
# P1.6 SHADOW ALPHA RESEARCH & ORB CONFIGURATION
# =============================================================================
P1_6_ORB_SHADOW_ENABLED = True
P1_6_SHADOW_BROKER_EXECUTION_ENABLED = False  # Hardcoded invariant: Never True

# Existing production ORB remains strictly disabled
STRATEGY_ORB_ENABLED = False

ORB_RANGE_MINUTES = 15
ORB_EARLIEST_ENTRY_TIME_ET = "10:00"
ORB_LATEST_ENTRY_TIME_ET = "11:30"

ORB_SHADOW_ALLOW_LONG = True
ORB_SHADOW_ALLOW_SHORT = False  # Phase 1 is strictly Long-only

ORB_MIN_RELATIVE_VOLUME = 2.0
ORB_MIN_BREAKOUT_VOLUME_RATIO = 1.5
ORB_MIN_PREMARKET_GAP_PCT = 2.0
ORB_MIN_PREMARKET_VOLUME_RATIO = 2.0

ORB_TARGET_R_MULTIPLE = 2.0
ORB_MAX_ATR_EXTENSION = 1.25
ORB_MIN_STOP_ATR_MULTIPLE = 0.50

SHADOW_QUOTE_MAX_AGE_SECONDS = 5.0
SHADOW_SLIPPAGE_BPS = 5.0
SHADOW_STRATEGY_VERSION = "ORB_CONFIRMED_V1_FIXED_2R"

SHADOW_BASE_DIR = str(DATA_DIR / "shadow")
SHADOW_ENGINE_ROLE = "daytrade"

# ── D2 Shadow Experiment Flag (Strictly Default 1.2, Zero Production Change) ──
EXPERIMENTAL_MEAN_REVERSION_ATR_DEV = float(os.getenv("EXPERIMENTAL_MEAN_REVERSION_ATR_DEV", "1.2"))
EXPERIMENTAL_MR_SHADOW_ENABLED: bool = os.getenv("EXPERIMENTAL_MR_SHADOW_ENABLED", "false").lower() in ("true", "1")

# Direct constant fail-closed assertion (No getattr(config, ...) to prevent NameError)
if P1_6_ORB_SHADOW_ENABLED and P1_6_SHADOW_BROKER_EXECUTION_ENABLED:
    raise RuntimeError("Invalid configuration: P1.6 shadow mode cannot enable broker execution")


ORB_MIN_VOLUME_RATIO = 1.5
ORB_MIN_RANGE_ATR_RATIO = 0.5
VWAP_RSI_OVERSOLD = 35
VWAP_RSI_OVERBOUGHT = 65
RSI_MAX_BUY_ENTRY = 65.0
RSI_MIN_SHORT_ENTRY = 30.0
MIN_RR_RATIO = 1.5
BETA_HIGH_THRESHOLD = 1.5
EARNINGS_BLOCK_DAYS = 1
ATR_STOP_MULTIPLIER = 1.0
ATR_STOP_HIGH_BETA = 1.5
ATR_STOP_EARNINGS = 2.0
SOXX_GATE_ENABLED = True
SKIP_SUPPRESSOR_ENABLED = True
HARD_VETO_SKIP_LIMIT = 5
SESSION_SUPPRESSION_COOLDOWN_SECONDS = float(os.getenv("SESSION_SUPPRESSION_COOLDOWN_SECONDS", "2700.0"))  # 45m cooldown
MAX_DAILY_COOLDOWN_CYCLES = int(os.getenv("MAX_DAILY_COOLDOWN_CYCLES", "3"))  # Max 3 cooldowns before full-day lock
VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED = os.getenv("VWAP_BOUNCE_REGIME_SOXX_GATE_ENABLED", "True").lower() in ("true", "1", "yes")
VWAP_BOUNCE_SOXX_VETO_THRESHOLD = float(os.getenv("VWAP_BOUNCE_SOXX_VETO_THRESHOLD", "-1.50"))
VWAP_BOUNCE_BLOCKED_REGIMES = ("CAUTION", "RISK_OFF", "DATA_DEGRADED")
NEWS_HYGIENE_ENABLED = True
TIME_EXIT_CHECK_ENABLED = True
BREAKEVEN_BAND_PCT = 0.20

# ── Midday Lunch Chop Filter (11:30 AM - 1:30 PM ET / 10:30 PM - 12:30 AM ICT) ──
LUNCH_CHOP_FILTER_ENABLED = True  # Blocks low-volume false breakout entries during NY midday
LUNCH_CHOP_START = "11:30"
LUNCH_CHOP_END   = "13:30"
GAP_CONTINUATION_MIN_HOLD_BARS = 3
VOLATILITY_PENALTY_MULTIPLIER = 4.0

# ── Phase C: Two-Lane Active Paper Validation Engine ──────
VALIDATION_LANE_ENABLED = os.getenv("VALIDATION_LANE_ENABLED", "True").lower() in ("true", "1", "yes")
VALIDATION_LANE_SHARES = int(os.getenv("VALIDATION_LANE_SHARES", "1"))
VALIDATION_LANE_LEDGER_PATH = DATA_DIR / "paper_validation_trades.jsonl"

# ── Paper Strategy Experiment Lane (Fail-Closed Paper Only) ──────
PAPER_EXPERIMENT_LANE_ENABLED: bool = os.getenv("PAPER_EXPERIMENT_LANE_ENABLED", "False").lower() in ("true", "1", "yes")
PAPER_EXPERIMENT_CELLS: tuple = ("CANDLESTICK_SIDEWAYS", "CANDLESTICK_TRANSITIONING")

# ── Breakeven Stop Lock ──────────────────────────────
BREAKEVEN_LOCK_ENABLED          = True
BREAKEVEN_LOCK_THRESHOLD_ATR    = 1.0   # Lock at +1.0x ATR unrealized profit
BREAKEVEN_SLIPPAGE_CUSHION      = True  # Add $0.01 cushion above entry for BUY

# ── SPY/QQQ Intraday Trend Guard ─────────────────────
SPY_TREND_GUARD_ENABLED         = True
SPY_TREND_GUARD_SYMBOL          = "SPY"

MAX_RISK_PER_TRADE_PCT = 0.3

# Day Trading daily loss percentage cap:
# Percentage points: 0.9 means 0.9% of equity. (DO NOT use decimal fraction 0.009 here)
# Compare with SWING_MAX_DAILY_LOSS_PCT above which is a fractional ratio (0.01 means 1.0%).
MAX_DAILY_LOSS_PCT = float(os.environ.get("MAX_DAILY_LOSS_PCT", "0.9"))
if not 0.0 < MAX_DAILY_LOSS_PCT <= 10.0:
    raise ValueError(
        "MAX_DAILY_LOSS_PCT must be greater than 0 "
        "and no greater than 10.0 percentage points; "
        "1.0 means 1.0% of equity"
    )
MAX_DAILY_LOSS_DOLLARS  = MAX_DAILY_LOSS_DAYTRADE
MAX_TRADES_PER_DAY = 100
MAX_CONSECUTIVE_LOSSES = 10
CONSECUTIVE_OUTAGE_LIMIT = 3
MAX_POSITION_SIZE_DOLLARS = 4000.0  # $4,000 per slot (Phase 2 7-slot allocation)
MAX_POSITION_PCT_OF_EQUITY = 0.20

CORRELATION_USE_ABS = True
CORRELATION_TIMEFRAME = "5m"
CORRELATION_LOOKBACK_BARS = 30
CORRELATION_REJECT_THRESHOLD = 0.80
CORRELATION_WARN_THRESHOLD = 0.60
CORRELATION_MILD_PENALTY = 0.90
CORRELATION_STRONG_PENALTY = 0.75
CORRELATION_HARD_PENALTY = 0.50
CORRELATION_CACHE_TTL_SECONDS = 300
CORRELATION_CACHE_MAX_ENTRIES = 5000
CORRELATION_CONSERVATIVE_REJECT = False
MIN_CORRELATION_SAMPLES = 10

TRAILING_STOP_ATR_MULTIPLE = 1.5
TRAILING_STOP_FALLBACK_PCT = 0.02
TRAILING_STOP_MIN_UPDATE_DELTA = 0.25
TRAILING_STOP_UPDATE_COOLDOWN_SECONDS = 30
TRAILING_STOP_ENABLE_SHORT = True
TRAILING_STOP_USE_ATR = True
TRAILING_STOP_MIN_PROFIT_PCT = 0.01
TRAILING_STOP_MIN_TRAIL_PCT = 0.005
TRAILING_ATR_MAX_REFRESH_CHANGE_PCT: float = 0.25
TRAILING_ATR_REFRESH_ALPHA: float = 0.25
TRAILING_HIGH_VOL_ATR_PCT: float = 0.008
TRAILING_STOP_MIN_COOLDOWN_SECONDS: float = 15.0
TRAILING_FAST_UPDATE_SYMBOLS: frozenset = frozenset({"NVDA", "TSLA", "AVGO", "AMD", "PLTR"})
TIME_EXIT_MIN_MFE_PCT: float = 0.30
TIME_EXIT_MAX_GIVEBACK_PCT: float = 25.0
NAKED_LIMIT_PROTECTION = "replace"

DEFAULT_STOP_ATR_MULTIPLE = 1.5
DEFAULT_TARGET_RR_RATIO = 2.0

FLATTEN_BEFORE_CLOSE_MINUTES = 5
FLATTEN_WARNING_MINUTES = 15
FLATTEN_ON_DAILY_LOSS = True
REENTRY_COOLDOWN_MINUTES = 45

PDT_ENABLED = True
PDT_MIN_EQUITY = 25_000.0
PDT_MAX_DAY_TRADES_5_DAYS = 3

SIZING_METHOD = "risk_based"
FIXED_SHARE_COUNT = 100
FIXED_DOLLAR_AMOUNT = 1000.0
ATR_SIZING_MULTIPLIER = 1.5
MIN_ATR_THRESHOLD = 0.01

DEFAULT_ORDER_TYPE = "LMT"
LIMIT_OFFSET_CENTS = 5
IOC_ENABLED = False
BRACKET_ORDER_ENABLED = True

SCAN_ENABLE_SENTIMENT = True
SCAN_SENTIMENT_MAX_HEADLINES = 5
SCAN_SCORE_SENTIMENT_WEIGHT = 0.2
SCAN_SENTIMENT_CACHE_TTL = 3600

DAYTRADE_LOG_FILE = LOG_DIR / "daytrading.log"
DAYTRADE_TRADE_JOURNAL_FILE = REPORTS_DIR / "trade_journal.jsonl"
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

EXPECTED_DAYTRADE_WATCHLIST_COUNT: int = 31
DAYTRADE_WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "AMD", "GOOGL", "QQQ", "SPY",
    "PLTR", "AVGO", "JPM", "BAC", "UBER", "MU", "NFLX",
    "LLY", "UNH", "ABBV", "COST", "WMT", "HD", "GS", "MS", "CAT", "GE",
    "XOM", "CVX", "CRM", "ORCL",
]
USE_DYNAMIC_SCANNER = False  # False = Strictly trade pristine 31 institutional DAYTRADE_WATCHLIST with zero slippage; True = Dynamic full-market scanner

DAYTRADE_ML_ENABLED = False
DAYTRADE_ML_CONFIDENCE_WEIGHT = 0.3
DAYTRADE_ML_MODEL_PATH = DATA_DIR / "models" / "signal_predictor.pkl"
DAYTRADE_ML_MIN_TRAINING_SAMPLES = 10
DAYTRADE_ML_AUTO_RETRAIN = True
DAYTRADE_ML_RETRAIN_INTERVAL = 100
DAYTRADE_ML_N_ESTIMATORS = 100
DAYTRADE_ML_MAX_TRAINING_FILES = 90
DAYTRADE_ML_MAX_TRAINING_ROWS = 100000
DAYTRADE_ML_MAX_JOURNAL_ROWS = 100000
DAYTRADE_ML_MAX_JOURNAL_BYTES = 16 * 1024 * 1024
DAYTRADE_ML_MODEL_CACHE_MAX_ENTRIES = 4
DAYTRADE_ML_TARGET_MATCH_HOURS = 24
DAYTRADE_PARALLEL_EVALUATION_MAX_WORKERS = 3
DAYTRADE_PARALLEL_EVALUATION_MIN_SYMBOLS = 4
DAYTRADE_PARALLEL_EVALUATION_ENABLED = True
DAYTRADE_PARALLEL_EVALUATION_FALLBACK_TO_SERIAL = True
DAYTRADE_ML_MAX_DEPTH = 5
DAYTRADE_ML_TEST_SIZE = 0.2

DAYTRADE_PAPER_RISK_PCT            = 0.001
DAYTRADE_MIN_RR                    = MIN_RR_RATIO  # Explicit canonical alias (1.50)
DAYTRADE_ATR_STOP_MULTIPLE         = 1.25
DAYTRADE_MAX_PAPER_TRADES_PER_RUN  = 1
MOMENTUM_OVEREXTENSION_ATR_MULT    = 1.25
MAX_ACCEPTABLE_SPREAD_PCT          = 0.75  # Max bid-ask spread (0.75%) allowed for liquid execution (paper trading calibrated)

# ── Scheduler ────────────────────────────────────
SCHEDULER_ENABLED            = True
SCHEDULER_DRY_RUN_DEFAULT    = True
SCHEDULER_REQUIRE_RTH        = True

# ── Reconnect ────────────────────────────────────
IBKR_RECONNECT_ENABLED            = True
IBKR_RECONNECT_MAX_ATTEMPTS       = 3
IBKR_RECONNECT_BASE_DELAY_SECONDS = 2.0
IBKR_RECONNECT_MAX_DELAY_SECONDS  = 30.0
IBKR_REQUEST_TIMEOUT_SECONDS      = 30.0

# ── Alerts ───────────────────────────────────────
ALERTS_ENABLED        = False
ALERTS_LOG_ONLY       = True
ALERT_MIN_SEVERITY    = "warning"

# ── Account ──────────────────────────────────────
ASSERT_ACCOUNT_TYPE       = True
EXPECTED_PAPER_ACCOUNT_ID = None
LIVE_ACCOUNT_ID           = None

# ── Minervini ────────────────────────────────────
MINERVINI_OVERLAY_ENABLED        = False
MINERVINI_STAGE2_BLOCK_ENABLED   = False
MINERVINI_SIZING_ENABLED         = False
MINERVINI_COACH_ENABLED          = False
MINERVINI_LIVE_STOP_ENABLED      = False
MINERVINI_NEAR_52W_HIGH_PCT      = 0.25
MINERVINI_OFF_52W_LOW_PCT        = 0.30
MINERVINI_RS_RANK_MIN            = 70.0
MINERVINI_MA_200_RISING_LOOKBACK = 20
MINERVINI_VCP_MIN_CONTRACTIONS   = 2
MINERVINI_VCP_MAX_BASE_DEPTH_PCT = 0.35
MINERVINI_VCP_PIVOT_LOOKBACK     = 15
MINERVINI_POCKET_VOL_LOOKBACK    = 10
MINERVINI_RISK_PER_TRADE_USD     = 25.0
MINERVINI_STOP_BUFFER_PCT        = 0.005
MINERVINI_MAX_STOP_DISTANCE_PCT  = 0.10
MINERVINI_COACH_LANGUAGE         = "en"

# ── Expectancy Report ────────────────────────────
EXPECTANCY_BACKTEST_TRADES   = REPORTS_DIR / "backtest_trades.csv"
EXPECTANCY_REPORT_JSON       = REPORTS_DIR / "expectancy_metrics.json"
EXPECTANCY_REPORT_MD         = REPORTS_DIR / "expectancy_report.md"
EXPECTANCY_ENABLE_PROXY_RISK = False

# ── Forward Test Tracking ────────────────────────
FORWARD_TEST_JOURNAL     = REPORTS_DIR / "paper_trades.jsonl"
FORWARD_TEST_REPORT_JSON = REPORTS_DIR / "forward_test_metrics.json"
FORWARD_TEST_REPORT_MD   = REPORTS_DIR / "forward_test_report.md"
FORWARD_TEST_SESSION     = None

# ── Swing Trading Hard Risk Ceilings & Sector Policies ──────────────
SWING_ACCOUNT_ID: str                  = "PA3R71I7JFDZ"
SECTOR_EXPOSURE_CAP_PCT: float         = 0.25
MAX_CLUSTER_RISK_PCT: float            = 0.0125

# Authoritative sector attribution with source and versioning
SECTOR_MAPPINGS: dict = {
    "SPY": {"sector": "INDEX_ETF", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "QQQ": {"sector": "INDEX_ETF", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "VTI": {"sector": "INDEX_ETF", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "AAPL": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "MSFT": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "NVDA": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "AMD": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "AVGO": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "TSLA": {"sector": "CONSUMER_DISCRETIONARY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "AMZN": {"sector": "CONSUMER_DISCRETIONARY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "META": {"sector": "COMMUNICATION_SERVICES", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "GOOGL": {"sector": "COMMUNICATION_SERVICES", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "NFLX": {"sector": "COMMUNICATION_SERVICES", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "COST": {"sector": "CONSUMER_STAPLES", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "JPM": {"sector": "FINANCIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "BAC": {"sector": "FINANCIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "V": {"sector": "FINANCIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "XOM": {"sector": "ENERGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "UNH": {"sector": "HEALTHCARE", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "XLV": {"sector": "HEALTHCARE", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "CAT": {"sector": "INDUSTRIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "LLY": {"sector": "HEALTHCARE", "source": "STATIC_CONFIG", "mapping_version": "2026-09-08"},
    "CVX": {"sector": "ENERGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-20"},
    "PLTR": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "UBER": {"sector": "INDUSTRIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "MU": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "ABBV": {"sector": "HEALTHCARE", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "WMT": {"sector": "CONSUMER_STAPLES", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "HD": {"sector": "CONSUMER_DISCRETIONARY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "GS": {"sector": "FINANCIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "MS": {"sector": "FINANCIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "GE": {"sector": "INDUSTRIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "BA": {"sector": "INDUSTRIALS", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "CRM": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
    "ORCL": {"sector": "INFORMATION_TECHNOLOGY", "source": "STATIC_CONFIG", "mapping_version": "2026-09-28"},
}

# ── Observation-Only Sector Benchmarks (Strictly Non-Tradable) ────────────────
SECTOR_BENCHMARKS: tuple[str, ...] = (
    "SPY", "XLV", "XLP", "XLE", "XLK", "XLF", "XLU", "SOXX"
)
# Pure observation ETFs not present in the active tradable universe (preserves SPY & XLV eligibility)
NON_TRADABLE_BENCHMARKS: tuple[str, ...] = tuple(
    s for s in SECTOR_BENCHMARKS if s not in WATCHLIST
)


# ═══════════════════════════════════════════════════════════════════════
# Dataclass-based Settings (dependency-injection friendly)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Settings:
    """Immutable-equivalent settings namespace for dependency injection.

    Usage:
        # copy from module-level defaults
        cfg = Settings.from_defaults()
        # override
        cfg.BUY_THRESHOLD = 0.70
        # inject
        predictor = Predictor(settings=cfg)
    """
    # Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = DATA_DIR
    MODELS_DIR: Path = MODELS_DIR
    REPORTS_DIR: Path = REPORTS_DIR
    LOG_DIR: Path = LOG_DIR
    ML_MODELS_FILE: Path = ML_MODELS_FILE
    LSTM_CKPT_FILE: Path = LSTM_CKPT_FILE

    # Universe
    WATCHLIST: List[str] = field(default_factory=lambda: list(WATCHLIST))
    SHADOW_MONITOR: List[str] = field(default_factory=lambda: list(SHADOW_MONITOR))
    FULL_MARKET_SCAN_ENABLED: bool = FULL_MARKET_SCAN_ENABLED
    FULL_MARKET_CACHE_HOURS: int = FULL_MARKET_CACHE_HOURS
    FULL_MARKET_MAX_SYMBOLS_TO_CHECK: int = FULL_MARKET_MAX_SYMBOLS_TO_CHECK
    FULL_MARKET_SELECTION_MODE: str = FULL_MARKET_SELECTION_MODE
    FULL_MARKET_CORE_SYMBOLS: List[str] = field(default_factory=lambda: list(FULL_MARKET_CORE_SYMBOLS))
    FULL_MARKET_RANDOM_SEED: int = FULL_MARKET_RANDOM_SEED
    FULL_MARKET_ROTATION_STATE_FILE: Path = FULL_MARKET_ROTATION_STATE_FILE
    HOT_SCAN_TOP_N: int = HOT_SCAN_TOP_N
    HOT_SCAN_MIN_PRICE: float = HOT_SCAN_MIN_PRICE
    HOT_SCAN_MAX_PRICE: float = HOT_SCAN_MAX_PRICE
    HOT_SCAN_MIN_AVG_VOLUME: int = HOT_SCAN_MIN_AVG_VOLUME
    HOT_SCAN_EXCLUDE_ETFS: bool = HOT_SCAN_EXCLUDE_ETFS
    HOT_SCAN_MAX_ATR_PCT: float = HOT_SCAN_MAX_ATR_PCT
    HOT_SCAN_CHUNK_SIZE: int = HOT_SCAN_CHUNK_SIZE
    HOT_SCAN_SLEEP_SECONDS: float = HOT_SCAN_SLEEP_SECONDS
    NASDAQ_LISTED_URL: str = NASDAQ_LISTED_URL
    OTHER_LISTED_URL: str = OTHER_LISTED_URL
    SYMBOL_UNIVERSE_FILE: Path = SYMBOL_UNIVERSE_FILE
    HOT_CANDIDATES_FILE: Path = HOT_CANDIDATES_FILE
    SELECTED_SCAN_SYMBOLS_FILE: Path = SELECTED_SCAN_SYMBOLS_FILE
    IPO_WATCH_SYMBOLS: List[str] = field(default_factory=lambda: list(IPO_WATCH_SYMBOLS))

    # Data
    PRICE_PERIOD: str = PRICE_PERIOD
    PRICE_INTERVAL: str = PRICE_INTERVAL
    MIN_HISTORY_DAYS: int = MIN_HISTORY_DAYS
    CACHE_TTL_SECONDS: int = CACHE_TTL_SECONDS

    # Technical
    SMA_SHORT: int = SMA_SHORT
    SMA_LONG: int = SMA_LONG
    EMA_PERIOD: int = EMA_PERIOD
    RSI_PERIOD: int = RSI_PERIOD
    RSI_SMOOTHING: str = RSI_SMOOTHING
    MACD_FAST: int = MACD_FAST
    MACD_SLOW: int = MACD_SLOW
    MACD_SIGNAL: int = MACD_SIGNAL
    BOLLINGER_PERIOD: int = BOLLINGER_PERIOD
    BOLLINGER_STD: int = BOLLINGER_STD
    ATR_PERIOD: int = ATR_PERIOD
    VOLUME_MA_PERIOD: int = VOLUME_MA_PERIOD

    # Features
    USE_REGIME_FEATURES: bool = USE_REGIME_FEATURES
    USE_MARKET_RELATIVE_FEATURES: bool = USE_MARKET_RELATIVE_FEATURES
    MARKET_BENCHMARK_SYMBOL: str = MARKET_BENCHMARK_SYMBOL
    REGIME_VOL_SHORT: int = REGIME_VOL_SHORT
    REGIME_VOL_LONG: int = REGIME_VOL_LONG
    REGIME_MOM_SHORT: int = REGIME_MOM_SHORT
    REGIME_MOM_LONG: int = REGIME_MOM_LONG
    REGIME_RANK_WINDOW: int = REGIME_RANK_WINDOW
    REGIME_HIGH_WINDOW: int = REGIME_HIGH_WINDOW
    REL_RET_SHORT: int = REL_RET_SHORT
    REL_RET_LONG: int = REL_RET_LONG
    RS_SLOPE_WINDOW: int = RS_SLOPE_WINDOW
    MKT_TREND_SMA: int = MKT_TREND_SMA
    USE_CANDLESTICK_FEATURES: bool = USE_CANDLESTICK_FEATURES
    CANDLESTICK_DOJI_THRESHOLD: float = CANDLESTICK_DOJI_THRESHOLD
    CANDLESTICK_LONGBODY_THRESHOLD: float = CANDLESTICK_LONGBODY_THRESHOLD
    CANDLESTICK_LONGSHADOW_MULTIPLIER: float = CANDLESTICK_LONGSHADOW_MULTIPLIER
    CANDLESTICK_SHORT_BODY_MULTIPLIER: float = CANDLESTICK_SHORT_BODY_MULTIPLIER
    USE_MICRO_FEATURES: bool = USE_MICRO_FEATURES

    # ML
    ML_WINDOW: int = ML_WINDOW
    ML_HORIZON: int = ML_HORIZON
    ML_TEST_RATIO: float = ML_TEST_RATIO
    RANDOM_STATE: int = RANDOM_STATE
    RF_N_ESTIMATORS: int = RF_N_ESTIMATORS
    RF_MAX_DEPTH: int = RF_MAX_DEPTH

    # LSTM
    LSTM_HIDDEN: int = LSTM_HIDDEN
    LSTM_LAYERS: int = LSTM_LAYERS
    LSTM_DROPOUT: float = LSTM_DROPOUT
    LSTM_EPOCHS: int = LSTM_EPOCHS
    LSTM_BATCH: int = LSTM_BATCH
    LSTM_LR: float = LSTM_LR
    LSTM_PATIENCE: int = LSTM_PATIENCE
    LSTM_WINDOW: int = LSTM_WINDOW
    LSTM_BIDIRECTIONAL: bool = LSTM_BIDIRECTIONAL
    LSTM_GRAD_CLIP_NORM: float = LSTM_GRAD_CLIP_NORM
    LSTM_CYCLE_LR_STEP: int = LSTM_CYCLE_LR_STEP
    LSTM_CYCLE_LR_BASE: float = LSTM_CYCLE_LR_BASE
    LSTM_HYPEROPT_TRIALS: int = LSTM_HYPEROPT_TRIALS
    LSTM_META_ENSEMBLE: bool = LSTM_META_ENSEMBLE

    # Signal
    BUY_THRESHOLD: float = BUY_THRESHOLD
    SELL_THRESHOLD: float = SELL_THRESHOLD
    MIN_HOLD_BARS: int = MIN_HOLD_BARS
    WEIGHT_RF: float = WEIGHT_RF
    WEIGHT_LSTM: float = WEIGHT_LSTM
    WEIGHT_XGB: float = WEIGHT_XGB
    WEIGHT_TRANSFORMER: float = WEIGHT_TRANSFORMER
    WEIGHT_TECHNICAL: float = WEIGHT_TECHNICAL
    MIN_ML_MODELS_FOR_SIGNAL: int = MIN_ML_MODELS_FOR_SIGNAL

    # Model gate
    MODEL_METRICS_FILE: Path = MODEL_METRICS_FILE
    MODEL_GATE_ENABLED: bool = MODEL_GATE_ENABLED
    MODEL_MIN_RF_ACCURACY: float = MODEL_MIN_RF_ACCURACY
    MODEL_MIN_RF_F1: float = MODEL_MIN_RF_F1
    MODEL_MAX_AGE_DAYS: int = MODEL_MAX_AGE_DAYS
    MODEL_MIN_RF_AUC: float = MODEL_MIN_RF_AUC
    MODEL_MIN_LSTM_AUC: float = MODEL_MIN_LSTM_AUC
    MODEL_HOLDOUT_RATIO: float = MODEL_HOLDOUT_RATIO

    # Permutation
    PERMUTATION_N_SHUFFLES: int = PERMUTATION_N_SHUFFLES
    PERMUTATION_SAMPLE_SYMBOLS: int = PERMUTATION_SAMPLE_SYMBOLS
    PERMUTATION_FDR_Q: float = PERMUTATION_FDR_Q

    # Risk
    MAX_POSITION_PCT: float = MAX_POSITION_PCT
    MAX_TRADE_VALUE: float = MAX_TRADE_VALUE
    MAX_OPEN_POSITIONS: int = MAX_OPEN_POSITIONS
    STOP_LOSS_PCT: float = STOP_LOSS_PCT
    TAKE_PROFIT_PCT: float = TAKE_PROFIT_PCT
    MIN_PROFIT_MARGIN: float = MIN_PROFIT_MARGIN
    ESTIMATED_COMMISSION_PER_TRADE: float = ESTIMATED_COMMISSION_PER_TRADE
    MIN_COMMISSION_PER_TRADE: float = MIN_COMMISSION_PER_TRADE
    COMMISSION_PER_SHARE: float = COMMISSION_PER_SHARE

    # Labeling
    LABEL_MODE: str = LABEL_MODE
    LABEL_TP_PCT: float = LABEL_TP_PCT
    LABEL_STOP_PCT: float = LABEL_STOP_PCT
    HARD_STOP_LOSS_PCT: float = HARD_STOP_LOSS_PCT
    USE_TRAILING_EXIT: bool = USE_TRAILING_EXIT
    TRAILING_STOP_PCT: float = TRAILING_STOP_PCT

    # Safety
    ALLOW_SHORT: bool = ALLOW_SHORT
    MIN_TRADE_CASH: float = MIN_TRADE_CASH
    LIMIT_ORDER_OFFSET_PCT: float = LIMIT_ORDER_OFFSET_PCT
    MAX_DAILY_TRADES: int = MAX_DAILY_TRADES
    MAX_DAILY_LOSS_USD: float = MAX_DAILY_LOSS_USD
    ALLOW_HISTORICAL_PRICE_FOR_ORDERS: bool = ALLOW_HISTORICAL_PRICE_FOR_ORDERS
    COACH_ALLOW_HISTORICAL_PRICE: bool = COACH_ALLOW_HISTORICAL_PRICE

    # Order exec
    ORDER_FILL_TIMEOUT_SECONDS: float = ORDER_FILL_TIMEOUT_SECONDS
    ORDER_POLL_SECONDS: float = ORDER_POLL_SECONDS
    CLOSE_MAX_ATTEMPTS: int = CLOSE_MAX_ATTEMPTS
    PROTECTIVE_TIF: str = PROTECTIVE_TIF
    ACCOUNT_MAX_DRAWDOWN_PCT: float = ACCOUNT_MAX_DRAWDOWN_PCT
    MAX_SYMBOL_EXPOSURE_PCT: float = MAX_SYMBOL_EXPOSURE_PCT

    # Data integrity
    REQUIRE_REALTIME_DATA_FOR_ORDERS: bool = REQUIRE_REALTIME_DATA_FOR_ORDERS
    PAPER_MODE_AUTO_CONFIRM_FILL: bool = PAPER_MODE_AUTO_CONFIRM_FILL
    MAX_QUOTE_SPREAD_PCT: float = MAX_QUOTE_SPREAD_PCT
    DECISION_PRICE_MAX_DEVIATION_PCT: float = DECISION_PRICE_MAX_DEVIATION_PCT
    MAX_ACCEPTABLE_SPREAD_PCT: float = MAX_ACCEPTABLE_SPREAD_PCT
    MAX_DECISION_DEVIATION_PCT: float = MAX_DECISION_DEVIATION_PCT
    MAX_QUOTE_AGE_SECONDS: float = MAX_QUOTE_AGE_SECONDS
    MARKET_HOURS_GATE_ENABLED: bool = MARKET_HOURS_GATE_ENABLED

    # Backtest
    BACKTEST_TRANSACTION_COST_PCT: float = BACKTEST_TRANSACTION_COST_PCT
    BACKTEST_SLIPPAGE_PCT: float = BACKTEST_SLIPPAGE_PCT
    BACKTEST_INCLUDE_LSTM: bool = BACKTEST_INCLUDE_LSTM
    BACKTEST_LSTM_EPOCHS: int = BACKTEST_LSTM_EPOCHS
    BACKTEST_LSTM_BATCH: int = BACKTEST_LSTM_BATCH
    BACKTEST_PURGE_GAP: int = BACKTEST_PURGE_GAP

    # IBKR
    IBKR_HOST: str = IBKR_HOST
    IBKR_PORT: int = IBKR_PORT
    PAPER_IBKR_PORT: int = PAPER_IBKR_PORT
    REQUIRE_PAPER_PORT: bool = REQUIRE_PAPER_PORT
    CLIENT_ID_BOT: int = CLIENT_ID_BOT
    CLIENT_ID_CHECK: int = CLIENT_ID_CHECK
    CLIENT_ID_CANCEL: int = CLIENT_ID_CANCEL
    CLIENT_ID_FLATTEN: int = CLIENT_ID_FLATTEN
    IBKR_CLIENT_ID: int = IBKR_CLIENT_ID
    PAPER_CAPITAL: float = PAPER_CAPITAL
    IBKR_MARKET_DATA_TYPE: int = IBKR_MARKET_DATA_TYPE

    # Alpaca
    ALPACA_PAPER_BASE_URL: str = ALPACA_PAPER_BASE_URL
    ALPACA_DATA_FEED: str = ALPACA_DATA_FEED
    ALPACA_CHUNK_SIZE: int = ALPACA_CHUNK_SIZE
    ALPACA_CHUNK_DELAY_MS: int = ALPACA_CHUNK_DELAY_MS
    ALPACA_CHUNK_TIMEOUT_SECONDS: float = ALPACA_CHUNK_TIMEOUT_SECONDS
    ALPACA_SUBSCRIPTION_TIER: str = ALPACA_SUBSCRIPTION_TIER
    ALPACA_WS_PING_INTERVAL_SECONDS: int = ALPACA_WS_PING_INTERVAL_SECONDS
    ALPACA_WS_PING_TIMEOUT_SECONDS: int = ALPACA_WS_PING_TIMEOUT_SECONDS
    ALPACA_WS_MAX_QUEUE: int = ALPACA_WS_MAX_QUEUE
    TRADE_STREAM_START_TIMEOUT_SECONDS: float = TRADE_STREAM_START_TIMEOUT_SECONDS
    TRADE_STREAM_STOP_TIMEOUT_SECONDS: float = TRADE_STREAM_STOP_TIMEOUT_SECONDS
    TRADE_STREAM_RECONNECT_MAX_ATTEMPTS: int = TRADE_STREAM_RECONNECT_MAX_ATTEMPTS
    TRADE_STREAM_RECONNECT_BASE_DELAY_SECONDS: float = TRADE_STREAM_RECONNECT_BASE_DELAY_SECONDS
    TRADE_STREAM_RECONNECT_MAX_DELAY_SECONDS: float = TRADE_STREAM_RECONNECT_MAX_DELAY_SECONDS
    TRADE_STREAM_RECONNECT_JITTER_RATIO: float = TRADE_STREAM_RECONNECT_JITTER_RATIO
    ALPACA_CONNECT_TIMEOUT_SECONDS: float = ALPACA_CONNECT_TIMEOUT_SECONDS
    ALPACA_READ_TIMEOUT_SECONDS: float = ALPACA_READ_TIMEOUT_SECONDS
    ALPACA_TOTAL_RETRIES: int = ALPACA_TOTAL_RETRIES
    ALPACA_RETRY_BACKOFF_FACTOR: float = ALPACA_RETRY_BACKOFF_FACTOR
    ALPACA_CONNECT_MAX_ATTEMPTS: int = ALPACA_CONNECT_MAX_ATTEMPTS


    # Logging
    LOG_FILE: Path = LOG_FILE
    LOG_MAX_BYTES: int = LOG_MAX_BYTES
    LOG_BACKUP_COUNT: int = LOG_BACKUP_COUNT

    # Coach
    COACH_MIN_CONFIDENCE_FOR_CANDIDATE: float = COACH_MIN_CONFIDENCE_FOR_CANDIDATE
    COACH_REQUIRE_CHART_CHECK: bool = COACH_REQUIRE_CHART_CHECK
    COACH_REQUIRE_USER_CONFIRM: bool = COACH_REQUIRE_USER_CONFIRM
    COACH_MAX_NEW_TRADES_PER_RUN: int = COACH_MAX_NEW_TRADES_PER_RUN
    COACH_REPORT_FILE: Path = COACH_REPORT_FILE
    COACH_SKIP_IF_POSITION_OR_WORKING: bool = COACH_SKIP_IF_POSITION_OR_WORKING
    COACH_MAX_PAPER_TRADES_PER_RUN: int = COACH_MAX_PAPER_TRADES_PER_RUN
    COACH_DEFAULT_PREVIEW_CANDIDATES: int = COACH_DEFAULT_PREVIEW_CANDIDATES
    COACH_LIVE_TRADING_ENABLED: bool = COACH_LIVE_TRADING_ENABLED

    # Daytrade
    DAYTRADE_PAPER_RISK_PCT: float = DAYTRADE_PAPER_RISK_PCT
    DAYTRADE_MIN_RR: float = DAYTRADE_MIN_RR
    DAYTRADE_ATR_STOP_MULTIPLE: float = DAYTRADE_ATR_STOP_MULTIPLE
    DAYTRADE_MAX_PAPER_TRADES_PER_RUN: int = DAYTRADE_MAX_PAPER_TRADES_PER_RUN

    # Scheduler
    SCHEDULER_ENABLED: bool = SCHEDULER_ENABLED
    SCHEDULER_DRY_RUN_DEFAULT: bool = SCHEDULER_DRY_RUN_DEFAULT
    SCHEDULER_REQUIRE_RTH: bool = SCHEDULER_REQUIRE_RTH

    # Reconnect
    IBKR_RECONNECT_ENABLED: bool = IBKR_RECONNECT_ENABLED
    IBKR_RECONNECT_MAX_ATTEMPTS: int = IBKR_RECONNECT_MAX_ATTEMPTS
    IBKR_RECONNECT_BASE_DELAY_SECONDS: float = IBKR_RECONNECT_BASE_DELAY_SECONDS
    IBKR_RECONNECT_MAX_DELAY_SECONDS: float = IBKR_RECONNECT_MAX_DELAY_SECONDS
    IBKR_REQUEST_TIMEOUT_SECONDS: float = IBKR_REQUEST_TIMEOUT_SECONDS

    # Alerts
    ALERTS_ENABLED: bool = ALERTS_ENABLED
    ALERTS_LOG_ONLY: bool = ALERTS_LOG_ONLY
    ALERT_MIN_SEVERITY: str = ALERT_MIN_SEVERITY

    # Account
    ASSERT_ACCOUNT_TYPE: bool = ASSERT_ACCOUNT_TYPE
    EXPECTED_PAPER_ACCOUNT_ID: Optional[str] = EXPECTED_PAPER_ACCOUNT_ID
    LIVE_ACCOUNT_ID: Optional[str] = LIVE_ACCOUNT_ID

    # Minervini
    MINERVINI_OVERLAY_ENABLED: bool = MINERVINI_OVERLAY_ENABLED
    MINERVINI_STAGE2_BLOCK_ENABLED: bool = MINERVINI_STAGE2_BLOCK_ENABLED
    MINERVINI_SIZING_ENABLED: bool = MINERVINI_SIZING_ENABLED
    MINERVINI_COACH_ENABLED: bool = MINERVINI_COACH_ENABLED
    MINERVINI_LIVE_STOP_ENABLED: bool = MINERVINI_LIVE_STOP_ENABLED
    MINERVINI_NEAR_52W_HIGH_PCT: float = MINERVINI_NEAR_52W_HIGH_PCT
    MINERVINI_OFF_52W_LOW_PCT: float = MINERVINI_OFF_52W_LOW_PCT
    MINERVINI_RS_RANK_MIN: float = MINERVINI_RS_RANK_MIN
    MINERVINI_MA_200_RISING_LOOKBACK: int = MINERVINI_MA_200_RISING_LOOKBACK
    MINERVINI_VCP_MIN_CONTRACTIONS: int = MINERVINI_VCP_MIN_CONTRACTIONS
    MINERVINI_VCP_MAX_BASE_DEPTH_PCT: float = MINERVINI_VCP_MAX_BASE_DEPTH_PCT
    MINERVINI_VCP_PIVOT_LOOKBACK: int = MINERVINI_VCP_PIVOT_LOOKBACK
    MINERVINI_POCKET_VOL_LOOKBACK: int = MINERVINI_POCKET_VOL_LOOKBACK
    MINERVINI_RISK_PER_TRADE_USD: float = MINERVINI_RISK_PER_TRADE_USD
    MINERVINI_STOP_BUFFER_PCT: float = MINERVINI_STOP_BUFFER_PCT
    MINERVINI_MAX_STOP_DISTANCE_PCT: float = MINERVINI_MAX_STOP_DISTANCE_PCT
    MINERVINI_COACH_LANGUAGE: str = MINERVINI_COACH_LANGUAGE

    # Expectancy
    EXPECTANCY_BACKTEST_TRADES: Path = EXPECTANCY_BACKTEST_TRADES
    EXPECTANCY_REPORT_JSON: Path = EXPECTANCY_REPORT_JSON
    EXPECTANCY_REPORT_MD: Path = EXPECTANCY_REPORT_MD
    EXPECTANCY_ENABLE_PROXY_RISK: bool = EXPECTANCY_ENABLE_PROXY_RISK

    # Forward test
    # Daytrade V1 Integrated Settings
    PREMARKET_START: _dt.time = PREMARKET_START
    MARKET_OPEN: _dt.time = MARKET_OPEN
    MARKET_CLOSE: _dt.time = MARKET_CLOSE
    POSTMARKET_END: _dt.time = POSTMARKET_END
    TIMEZONE: str = TIMEZONE
    ORB_WINDOW_MINUTES: int = ORB_WINDOW_MINUTES
    ORB_EXTENDED_MINUTES: int = ORB_EXTENDED_MINUTES
    ENFORCE_GLOBAL_OPENING_BUFFER: bool = ENFORCE_GLOBAL_OPENING_BUFFER
    GLOBAL_OPENING_BUFFER_END_ET: str = GLOBAL_OPENING_BUFFER_END_ET
    SYMBOL_DENYLIST: List[str] = field(default_factory=lambda: list(SYMBOL_DENYLIST))
    INTRADAY_INTERVAL: str = INTRADAY_INTERVAL
    INTRADAY_LOOKBACK_DAYS: int = INTRADAY_LOOKBACK_DAYS
    DAILY_LOOKBACK_DAYS: int = DAILY_LOOKBACK_DAYS
    PREMARKET_INTERVAL: str = PREMARKET_INTERVAL
    INTRADAY_CACHE_TTL_SECONDS: int = INTRADAY_CACHE_TTL_SECONDS
    STALE_QUOTE_BUFFER_SECONDS: int = STALE_QUOTE_BUFFER_SECONDS
    SCAN_MIN_PRICE: float = SCAN_MIN_PRICE
    SCAN_MAX_PRICE: float = SCAN_MAX_PRICE
    SCAN_MIN_PREMARKET_VOLUME: int = SCAN_MIN_PREMARKET_VOLUME
    SCAN_MIN_RELATIVE_VOLUME: float = SCAN_MIN_RELATIVE_VOLUME
    SCAN_MIN_GAP_PCT: float = SCAN_MIN_GAP_PCT
    SCAN_MAX_GAP_PCT: float = SCAN_MAX_GAP_PCT
    SCAN_MIN_FLOAT_SHARES: int = SCAN_MIN_FLOAT_SHARES
    SCAN_MAX_CANDIDATES: int = SCAN_MAX_CANDIDATES
    SCAN_CHUNK_SIZE: int = SCAN_CHUNK_SIZE
    SCAN_SLEEP_SECONDS: float = SCAN_SLEEP_SECONDS
    SCANNER_REFRESH_MINUTES: int = SCANNER_REFRESH_MINUTES
    DYNAMIC_SCREENER_ENABLED: bool = DYNAMIC_SCREENER_ENABLED
    DYNAMIC_SCREENER_TARGET_SYMBOLS: int = DYNAMIC_SCREENER_TARGET_SYMBOLS
    DYNAMIC_SCREENER_CACHE_TTL: int = DYNAMIC_SCREENER_CACHE_TTL
    SCAN_SCORE_GAP_WEIGHT: float = SCAN_SCORE_GAP_WEIGHT
    SCAN_SCORE_RVOL_WEIGHT: float = SCAN_SCORE_RVOL_WEIGHT
    SCAN_SCORE_VOL_WEIGHT: float = SCAN_SCORE_VOL_WEIGHT
    SCAN_SCORE_RVOL_CAP: float = SCAN_SCORE_RVOL_CAP
    SCAN_SCORE_VOL_NORMALIZER: int = SCAN_SCORE_VOL_NORMALIZER
    SCAN_OPEN_MIN_MOVE_PCT: float = SCAN_OPEN_MIN_MOVE_PCT
    SCAN_OPEN_MIN_VOLUME: int = SCAN_OPEN_MIN_VOLUME
    SCAN_OPEN_MIN_RELATIVE_VOLUME: float = SCAN_OPEN_MIN_RELATIVE_VOLUME
    SCAN_RESULTS_FILE: Path = SCAN_RESULTS_FILE
    VWAP_BOUNCE_TOLERANCE_PCT: float = VWAP_BOUNCE_TOLERANCE_PCT
    VWAP_RECLAIM_BARS: int = VWAP_RECLAIM_BARS
    VWAP_STD_BANDS: int = VWAP_STD_BANDS
    EMA_9: int = EMA_9
    EMA_20: int = EMA_20
    EMA_50: int = EMA_50
    STRATEGY_ORB_ENABLED: bool = STRATEGY_ORB_ENABLED
    STRATEGY_VWAP_BOUNCE_ENABLED: bool = STRATEGY_VWAP_BOUNCE_ENABLED
    STRATEGY_GAP_AND_GO_ENABLED: bool = STRATEGY_GAP_AND_GO_ENABLED
    STRATEGY_MOMENTUM_SCALP_ENABLED: bool = STRATEGY_MOMENTUM_SCALP_ENABLED
    STRATEGY_CANDLESTICK_ENABLED: bool = STRATEGY_CANDLESTICK_ENABLED
    VWAP_BOUNCE_ENABLED: bool = VWAP_BOUNCE_ENABLED
    MOMENTUM_SCALP_ENABLED: bool = MOMENTUM_SCALP_ENABLED
    STRATEGY_SETTINGS: dict = field(default_factory=lambda: dict(STRATEGY_SETTINGS))
    ORB_MIN_VOLUME_RATIO: float = ORB_MIN_VOLUME_RATIO
    ORB_MIN_RANGE_ATR_RATIO: float = ORB_MIN_RANGE_ATR_RATIO
    VWAP_RSI_OVERSOLD: float = VWAP_RSI_OVERSOLD
    VWAP_RSI_OVERBOUGHT: float = VWAP_RSI_OVERBOUGHT
    RSI_MAX_BUY_ENTRY: float = RSI_MAX_BUY_ENTRY
    BETA_HIGH_THRESHOLD: float = BETA_HIGH_THRESHOLD
    EARNINGS_BLOCK_DAYS: int = EARNINGS_BLOCK_DAYS
    ATR_STOP_MULTIPLIER: float = ATR_STOP_MULTIPLIER
    ATR_STOP_HIGH_BETA: float = ATR_STOP_HIGH_BETA
    ATR_STOP_EARNINGS: float = ATR_STOP_EARNINGS
    SOXX_GATE_ENABLED: bool = SOXX_GATE_ENABLED
    SKIP_SUPPRESSOR_ENABLED: bool = SKIP_SUPPRESSOR_ENABLED
    NEWS_HYGIENE_ENABLED: bool = NEWS_HYGIENE_ENABLED
    TIME_EXIT_CHECK_ENABLED: bool = TIME_EXIT_CHECK_ENABLED
    BREAKEVEN_BAND_PCT: float = BREAKEVEN_BAND_PCT
    GAP_CONTINUATION_MIN_HOLD_BARS: int = GAP_CONTINUATION_MIN_HOLD_BARS
    VOLATILITY_PENALTY_MULTIPLIER: float = VOLATILITY_PENALTY_MULTIPLIER
    MAX_RISK_PER_TRADE_PCT: float = MAX_RISK_PER_TRADE_PCT
    MAX_DAILY_LOSS_PCT: float = MAX_DAILY_LOSS_PCT
    MAX_DAILY_LOSS_DOLLARS: float = MAX_DAILY_LOSS_DOLLARS
    MAX_TRADES_PER_DAY: int = MAX_TRADES_PER_DAY
    MAX_CONSECUTIVE_LOSSES: int = MAX_CONSECUTIVE_LOSSES
    CONSECUTIVE_OUTAGE_LIMIT: int = CONSECUTIVE_OUTAGE_LIMIT
    MAX_POSITION_SIZE_DOLLARS: float = MAX_POSITION_SIZE_DOLLARS
    MAX_POSITION_PCT_OF_EQUITY: float = MAX_POSITION_PCT_OF_EQUITY
    CORRELATION_USE_ABS: bool = CORRELATION_USE_ABS
    CORRELATION_TIMEFRAME: str = CORRELATION_TIMEFRAME
    CORRELATION_LOOKBACK_BARS: int = CORRELATION_LOOKBACK_BARS
    CORRELATION_REJECT_THRESHOLD: float = CORRELATION_REJECT_THRESHOLD
    CORRELATION_WARN_THRESHOLD: float = CORRELATION_WARN_THRESHOLD
    CORRELATION_MILD_PENALTY: float = CORRELATION_MILD_PENALTY
    CORRELATION_STRONG_PENALTY: float = CORRELATION_STRONG_PENALTY
    CORRELATION_HARD_PENALTY: float = CORRELATION_HARD_PENALTY
    CORRELATION_CACHE_TTL_SECONDS: int = CORRELATION_CACHE_TTL_SECONDS
    CORRELATION_CACHE_MAX_ENTRIES: int = CORRELATION_CACHE_MAX_ENTRIES
    CORRELATION_CONSERVATIVE_REJECT: bool = CORRELATION_CONSERVATIVE_REJECT
    MIN_CORRELATION_SAMPLES: int = MIN_CORRELATION_SAMPLES
    TRAILING_STOP_ATR_MULTIPLE: float = TRAILING_STOP_ATR_MULTIPLE
    TRAILING_STOP_FALLBACK_PCT: float = TRAILING_STOP_FALLBACK_PCT
    TRAILING_STOP_MIN_UPDATE_DELTA: float = TRAILING_STOP_MIN_UPDATE_DELTA
    TRAILING_STOP_UPDATE_COOLDOWN_SECONDS: int = TRAILING_STOP_UPDATE_COOLDOWN_SECONDS
    TRAILING_ATR_MAX_REFRESH_CHANGE_PCT: float = TRAILING_ATR_MAX_REFRESH_CHANGE_PCT
    TRAILING_ATR_REFRESH_ALPHA: float = TRAILING_ATR_REFRESH_ALPHA
    TRAILING_HIGH_VOL_ATR_PCT: float = TRAILING_HIGH_VOL_ATR_PCT
    TRAILING_STOP_MIN_COOLDOWN_SECONDS: float = TRAILING_STOP_MIN_COOLDOWN_SECONDS
    TRAILING_FAST_UPDATE_SYMBOLS: frozenset = TRAILING_FAST_UPDATE_SYMBOLS
    TIME_EXIT_MIN_MFE_PCT: float = TIME_EXIT_MIN_MFE_PCT
    TIME_EXIT_MAX_GIVEBACK_PCT: float = TIME_EXIT_MAX_GIVEBACK_PCT

    # Breakeven Lock
    BREAKEVEN_LOCK_ENABLED:         bool  = BREAKEVEN_LOCK_ENABLED
    BREAKEVEN_LOCK_THRESHOLD_ATR:   float = BREAKEVEN_LOCK_THRESHOLD_ATR
    BREAKEVEN_SLIPPAGE_CUSHION:     bool  = BREAKEVEN_SLIPPAGE_CUSHION

    # SPY Trend Guard
    SPY_TREND_GUARD_ENABLED:        bool  = SPY_TREND_GUARD_ENABLED
    SPY_TREND_GUARD_SYMBOL:         str   = SPY_TREND_GUARD_SYMBOL
    TRAILING_STOP_ENABLE_SHORT: bool = TRAILING_STOP_ENABLE_SHORT
    TRAILING_STOP_USE_ATR: bool = TRAILING_STOP_USE_ATR
    NAKED_LIMIT_PROTECTION: str = NAKED_LIMIT_PROTECTION
    DEFAULT_STOP_ATR_MULTIPLE: float = DEFAULT_STOP_ATR_MULTIPLE
    DEFAULT_TARGET_RR_RATIO: float = DEFAULT_TARGET_RR_RATIO
    FLATTEN_BEFORE_CLOSE_MINUTES: int = FLATTEN_BEFORE_CLOSE_MINUTES
    FLATTEN_WARNING_MINUTES: int = FLATTEN_WARNING_MINUTES
    FLATTEN_ON_DAILY_LOSS: bool = FLATTEN_ON_DAILY_LOSS
    REENTRY_COOLDOWN_MINUTES: int = REENTRY_COOLDOWN_MINUTES
    PDT_ENABLED: bool = PDT_ENABLED
    PDT_MIN_EQUITY: float = PDT_MIN_EQUITY
    PDT_MAX_DAY_TRADES_5_DAYS: int = PDT_MAX_DAY_TRADES_5_DAYS
    SIZING_METHOD: str = SIZING_METHOD
    FIXED_SHARE_COUNT: int = FIXED_SHARE_COUNT
    FIXED_DOLLAR_AMOUNT: float = FIXED_DOLLAR_AMOUNT
    ATR_SIZING_MULTIPLIER: float = ATR_SIZING_MULTIPLIER
    MIN_ATR_THRESHOLD: float = MIN_ATR_THRESHOLD
    DEFAULT_ORDER_TYPE: str = DEFAULT_ORDER_TYPE
    LIMIT_OFFSET_CENTS: int = LIMIT_OFFSET_CENTS
    IOC_ENABLED: bool = IOC_ENABLED
    BRACKET_ORDER_ENABLED: bool = BRACKET_ORDER_ENABLED
    SCAN_ENABLE_SENTIMENT: bool = SCAN_ENABLE_SENTIMENT
    SCAN_SENTIMENT_MAX_HEADLINES: int = SCAN_SENTIMENT_MAX_HEADLINES
    SCAN_SCORE_SENTIMENT_WEIGHT: float = SCAN_SCORE_SENTIMENT_WEIGHT
    SCAN_SENTIMENT_CACHE_TTL: int = SCAN_SENTIMENT_CACHE_TTL
    DAYTRADE_LOG_FILE: Path = DAYTRADE_LOG_FILE
    DAYTRADE_TRADE_JOURNAL_FILE: Path = DAYTRADE_TRADE_JOURNAL_FILE
    DISCORD_WEBHOOK_URL: str = DISCORD_WEBHOOK_URL
    DAYTRADE_WATCHLIST: List[str] = field(default_factory=lambda: DAYTRADE_WATCHLIST)
    DAYTRADE_ML_ENABLED: bool = DAYTRADE_ML_ENABLED
    DAYTRADE_ML_CONFIDENCE_WEIGHT: float = DAYTRADE_ML_CONFIDENCE_WEIGHT
    DAYTRADE_ML_MODEL_PATH: Path = DAYTRADE_ML_MODEL_PATH
    DAYTRADE_ML_MIN_TRAINING_SAMPLES: int = DAYTRADE_ML_MIN_TRAINING_SAMPLES
    DAYTRADE_ML_AUTO_RETRAIN: bool = DAYTRADE_ML_AUTO_RETRAIN
    DAYTRADE_ML_RETRAIN_INTERVAL: int = DAYTRADE_ML_RETRAIN_INTERVAL
    DAYTRADE_ML_N_ESTIMATORS: int = DAYTRADE_ML_N_ESTIMATORS
    DAYTRADE_ML_MAX_TRAINING_FILES: int = DAYTRADE_ML_MAX_TRAINING_FILES
    DAYTRADE_ML_MAX_TRAINING_ROWS: int = DAYTRADE_ML_MAX_TRAINING_ROWS
    DAYTRADE_ML_MAX_JOURNAL_ROWS: int = DAYTRADE_ML_MAX_JOURNAL_ROWS
    DAYTRADE_ML_MAX_JOURNAL_BYTES: int = DAYTRADE_ML_MAX_JOURNAL_BYTES
    DAYTRADE_ML_MODEL_CACHE_MAX_ENTRIES: int = DAYTRADE_ML_MODEL_CACHE_MAX_ENTRIES
    DAYTRADE_ML_TARGET_MATCH_HOURS: int = DAYTRADE_ML_TARGET_MATCH_HOURS
    DAYTRADE_PARALLEL_EVALUATION_MAX_WORKERS: int = DAYTRADE_PARALLEL_EVALUATION_MAX_WORKERS
    DAYTRADE_PARALLEL_EVALUATION_MIN_SYMBOLS: int = DAYTRADE_PARALLEL_EVALUATION_MIN_SYMBOLS
    DAYTRADE_PARALLEL_EVALUATION_ENABLED: bool = DAYTRADE_PARALLEL_EVALUATION_ENABLED
    DAYTRADE_PARALLEL_EVALUATION_FALLBACK_TO_SERIAL: bool = DAYTRADE_PARALLEL_EVALUATION_FALLBACK_TO_SERIAL
    DAYTRADE_ML_MAX_DEPTH: int = DAYTRADE_ML_MAX_DEPTH
    DAYTRADE_ML_TEST_SIZE: float = DAYTRADE_ML_TEST_SIZE

    FORWARD_TEST_JOURNAL: Path = FORWARD_TEST_JOURNAL
    FORWARD_TEST_REPORT_JSON: Path = FORWARD_TEST_REPORT_JSON
    FORWARD_TEST_REPORT_MD: Path = FORWARD_TEST_REPORT_MD
    FORWARD_TEST_SESSION: Optional[str] = FORWARD_TEST_SESSION

    def __post_init__(self):
        # Dynamically mirror all uppercase module-level constants to guarantee zero sync drift
        import sys
        mod = sys.modules.get(__name__)
        if mod:
            for k, v in mod.__dict__.items():
                if k.isupper() and not k.startswith("_") and not hasattr(self, k):
                    setattr(self, k, v)
        if "ALPACA_CONNECT_TIMEOUT" in os.environ:
            try:
                self.ALPACA_CONNECT_TIMEOUT_SECONDS = float(os.environ["ALPACA_CONNECT_TIMEOUT"])
            except ValueError:
                pass
        if "ALPACA_READ_TIMEOUT" in os.environ:
            try:
                self.ALPACA_READ_TIMEOUT_SECONDS = float(os.environ["ALPACA_READ_TIMEOUT"])
            except ValueError:
                pass

    def __getattr__(self, name: str):
        import sys
        mod = sys.modules.get(__name__)
        if mod and hasattr(mod, name):
            return getattr(mod, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# Default instance — mutate only in tests via set_config().
_default_settings = Settings()

def get_settings() -> Settings:
    return _default_settings

def set_settings(s: Settings) -> None:
    global _default_settings
    _default_settings = s


def get_effective_daytrade_config() -> dict:
    """Returns a standardized dictionary of all authoritative daytrade runtime parameters."""
    import sys
    mod = sys.modules.get(__name__)
    def _g(key: str, default: Any) -> Any:
        return getattr(mod, key, default) if mod else globals().get(key, default)

    return {
        "MIN_RR_RATIO": float(_g("MIN_RR_RATIO", 1.50)),
        "DAYTRADE_MIN_RR": float(_g("DAYTRADE_MIN_RR", 1.50)),
        "MAX_ACCEPTABLE_SPREAD_PCT": float(_g("MAX_ACCEPTABLE_SPREAD_PCT", 0.75)),
        "MAX_ALLOWED_CHASE": float(_g("MAX_ALLOWED_CHASE", 0.15)),
        "DAYTRADE_BATCH_FETCH_MIN_COVERAGE": float(_g("DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95)),
        "DAYTRADE_BAR_PUBLICATION_TIMEOUT_S": float(_g("DAYTRADE_BAR_PUBLICATION_TIMEOUT_S", 90.0)),
        "LUNCH_CHOP_FILTER_ENABLED": bool(_g("LUNCH_CHOP_FILTER_ENABLED", True)),
        "REENTRY_COOLDOWN_MINUTES": int(_g("REENTRY_COOLDOWN_MINUTES", 45)),
        "HARD_VETO_SKIP_LIMIT": int(_g("HARD_VETO_SKIP_LIMIT", 5)),
        "MAX_RISK_PER_TRADE_DOLLARS": float(_g("MAX_RISK_PER_TRADE_DOLLARS", 50.0)),
        "SOXX_GATE_ENABLED": bool(_g("SOXX_GATE_ENABLED", True)),
        "MAX_CORRELATED_POSITIONS": int(_g("MAX_CORRELATED_POSITIONS", 2)),
        "VALIDATION_LANE_ENABLED": bool(_g("VALIDATION_LANE_ENABLED", True)),
        "VALIDATION_LANE_SHARES": int(_g("VALIDATION_LANE_SHARES", 1)),
    }


# Auto-initialize and guard child processes or test executions inheriting STOCK_ENGINE_TEST_ROOT
if _TEST_ROOT_ENV:
    init_isolated_test_env(_TEST_ROOT_ENV)
    install_production_write_guard()


# Regime Breadth Stale Delay Grace (A4)
REGIME_BREADTH_MAX_DELAY_SECONDS = 300.0

# ── Strategy Feature Flags (Group C) - Default=ON ─────────────
EARNINGS_BLACKOUT = True
DAYTRADE_EARNINGS_BLACKOUT = False
SWING_EARNINGS_BLACKOUT = True
EARNINGS_BLACKOUT_DAYS = 2
EARNINGS_EXIT_MODE = "exit"  # "exit" | "half" | "hold"
EARNINGS_CALENDAR = {
    "META": "2026-10-29",
    "AAPL": "2026-10-29",
    "AMZN": "2026-10-29",
    "GOOGL": "2026-10-28",
    "MSFT": "2026-10-28",
    "INTC": "2026-10-29",
    "V": "2026-10-28",
    "TSLA": "2026-10-21",
}

MACRO_EVENT_BLACKOUT = True
MACRO_EVENT_WINDOW_MINUTES = 15

MR_STOP_ATR_MULT = 1.25
MR_BREAKEVEN_PROFIT_ATR = 1.5
MR_BREAKEVEN_LOCK_ATR = 0.1

MR_RS_FILTER = True
MR_RS_FILTER_THRESHOLD = -1.0

SECTOR_TREND_OVERRIDE = True
SOXX_TREND_THRESHOLD = 1.5
SOXX_BREADTH_THRESHOLD = 60.0

SIGNAL_HYSTERESIS = True
SIGNAL_HYSTERESIS_CYCLES = 2

CLUSTER_NOTIONAL_CAP = 0.15

def log_strategy_feature_flags():
    """Log authoritative state of all Phase C Strategy Feature Flags at startup."""
    import logging
    _log = logging.getLogger("strategy_flags")
    _log.info("[STRATEGY FEATURE FLAGS]")
    _log.info("  C1. EARNINGS_BLACKOUT: Day=%s, Swing=%s (days=%d, exit_mode=%s)", DAYTRADE_EARNINGS_BLACKOUT, SWING_EARNINGS_BLACKOUT, EARNINGS_BLACKOUT_DAYS, EARNINGS_EXIT_MODE)
    _log.info("  C2. MACRO_EVENT_BLACKOUT: %s (window=±%dm)", MACRO_EVENT_BLACKOUT, MACRO_EVENT_WINDOW_MINUTES)
    _log.info("  C3. MR_STOP_ATR_MULT: %.2f (breakeven profit=%.1f ATR, lock=%.1f ATR)", MR_STOP_ATR_MULT, MR_BREAKEVEN_PROFIT_ATR, MR_BREAKEVEN_LOCK_ATR)
    _log.info("  C4. MR_RS_FILTER: %s (threshold=%.1f%%)", MR_RS_FILTER, MR_RS_FILTER_THRESHOLD)
    _log.info("  C5. SECTOR_TREND_OVERRIDE: %s (SOXX threshold=%.1f%%, breadth=%.1f)", SECTOR_TREND_OVERRIDE, SOXX_TREND_THRESHOLD, SOXX_BREADTH_THRESHOLD)
    _log.info("  C6. SIGNAL_HYSTERESIS: %s (min_cycles=%d)", SIGNAL_HYSTERESIS, SIGNAL_HYSTERESIS_CYCLES)
    _log.info("  C7. CLUSTER_NOTIONAL_CAP: %.2f (<= %.1f%% equity per sector)", CLUSTER_NOTIONAL_CAP, CLUSTER_NOTIONAL_CAP * 100)

