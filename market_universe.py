"""
market_universe.py — Broad US symbol universe discovery (READ-ONLY).

This module *only* discovers candidate tickers. It downloads the official
Nasdaq Trader symbol directory files, filters out instruments that are not
plain common stocks or normal ETFs, normalizes symbols for yfinance, and
caches the result to ``data/symbol_universe.csv``.

It never fetches prices, never predicts, and never places orders. Producing a
symbol here is *not* a buy signal — prediction + risk rules downstream still
decide everything.

Data sources (pipe-delimited text):
  - https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt
  - https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt

Public API:
  load_symbol_universe(force_refresh=False) -> pd.DataFrame
  get_universe(limit=None, exclude_etfs=None, force_refresh=False) -> pd.DataFrame
  get_universe_symbols(limit=None, exclude_etfs=None, force_refresh=False) -> list[str]
  select_symbols(max_symbols=None, mode=None, ...) -> (pd.DataFrame, dict)

DataFrame columns: symbol, name, source_exchange, instrument_type
"""
from __future__ import annotations

import io
import json
import logging
import random
import re
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config

logger = logging.getLogger(__name__)

# A polite User-Agent; nasdaqtrader.com may reject the bare urllib default.
_HTTP_HEADERS = {"User-Agent": "stock-engine-pro/1.0 (+paper-trading-research)"}
_DOWNLOAD_TIMEOUT = 30  # seconds

# Map the single-letter exchange code used in otherlisted.txt to a label.
_EXCHANGE_CODE = {
    "A": "NYSE_AMERICAN",
    "N": "NYSE",
    "P": "NYSE_ARCA",
    "Z": "CBOE_BZX",
    "V": "IEX",
}

# Security-name keywords that mark an instrument we do NOT want. Word-boundary
# regexes avoid false positives ("United" must not match "unit", "Granite" must
# not match "right"). ADRs ("American Depositary Shares") are intentionally kept
# — they are common-equity-like; only preferred/percent-bearing names are cut.
_EXCLUDE_NAME_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwarrant",
        r"\bright(s)?\b",
        r"\bunit(s)?\b",
        r"\bpreferred\b",
        r"\bnote(s)?\b",
        r"\bbond(s)?\b",
        r"\bdebenture",
        r"\bconvertible\b",
        r"\bsubordinated\b",
        r"\bwhen[\s-]issued\b",
        r"%",                       # e.g. "5.00% Senior Notes"
    )
]

# Share-class suffixes we keep (converted to yfinance "-" form). Anything else
# after a "." (e.g. ".WS" warrant, ".U" unit, ".PR" preferred, ".RT" right) is
# treated as a non-common instrument and dropped.
_CLASS_SUFFIXES = {"A", "B", "C", "D"}

_UNIVERSE_COLUMNS = ["symbol", "name", "source_exchange", "instrument_type"]


# ── Download ────────────────────────────────────────────────────────────────
def _http_get(url: str) -> str:
    req = urllib.request.Request(url, headers=_HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_pipe_file(text: str) -> pd.DataFrame:
    """Parse a Nasdaq Trader pipe-delimited file, dropping its footer row."""
    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str, keep_default_na=False)
    if df.empty:
        return df
    # The last data line is always "File Creation Time: ...".
    first_col = df.columns[0]
    df = df[~df[first_col].str.startswith("File Creation Time", na=False)]
    return df


def _parse_nasdaq_listed(text: str) -> pd.DataFrame:
    """nasdaqlisted.txt → normalized rows (all Nasdaq-listed)."""
    raw = _parse_pipe_file(text)
    if raw.empty:
        return pd.DataFrame(columns=_UNIVERSE_COLUMNS + ["test_issue", "is_etf"])
    out = pd.DataFrame()
    out["symbol"] = raw.get("Symbol", "").astype(str)
    out["name"] = raw.get("Security Name", "").astype(str)
    out["source_exchange"] = "NASDAQ"
    out["test_issue"] = raw.get("Test Issue", "N").astype(str)
    out["is_etf"] = raw.get("ETF", "N").astype(str)
    return out


def _parse_other_listed(text: str) -> pd.DataFrame:
    """otherlisted.txt → normalized rows (NYSE / NYSE American / ARCA / etc.)."""
    raw = _parse_pipe_file(text)
    if raw.empty:
        return pd.DataFrame(columns=_UNIVERSE_COLUMNS + ["test_issue", "is_etf"])
    out = pd.DataFrame()
    # Prefer the ACT Symbol; fall back to NASDAQ Symbol column if present.
    out["symbol"] = raw.get("ACT Symbol", raw.get("NASDAQ Symbol", "")).astype(str)
    out["name"] = raw.get("Security Name", "").astype(str)
    out["source_exchange"] = (
        raw.get("Exchange", "").astype(str).map(lambda c: _EXCHANGE_CODE.get(c, c or "OTHER"))
    )
    out["test_issue"] = raw.get("Test Issue", "N").astype(str)
    out["is_etf"] = raw.get("ETF", "N").astype(str)
    return out


def download_symbol_directories() -> pd.DataFrame:
    """Download + combine both directory files. Raises on network failure."""
    nasdaq = _parse_nasdaq_listed(_http_get(config.NASDAQ_LISTED_URL))
    other = _parse_other_listed(_http_get(config.OTHER_LISTED_URL))
    combined = pd.concat([nasdaq, other], ignore_index=True)
    logger.info(
        "Downloaded symbol directories: nasdaq=%d other=%d total=%d",
        len(nasdaq), len(other), len(combined),
    )
    return combined


# ── Filtering / normalization ───────────────────────────────────────────────
def _normalize_symbol(symbol: str) -> Optional[str]:
    """Return a yfinance-compatible ticker, or None if it should be dropped.

    - Reject anything with characters outside [A-Z0-9.-].
    - "BRK.A"/"BRK.B" share classes → "BRK-A"/"BRK-B".
    - Any other "." suffix (warrant/unit/right/preferred markers) → drop.
    """
    if symbol is None:
        return None
    sym = str(symbol).strip().upper()
    if not sym:
        return None

    # Reject preferred/warrant/unit markers and any unexpected punctuation early.
    if any(ch in sym for ch in "$+^=~*#!/ "):
        return None
    if not re.fullmatch(r"[A-Z0-9.\-]+", sym):
        return None

    if "." in sym:
        base, _, suffix = sym.partition(".")
        if suffix not in _CLASS_SUFFIXES:
            return None
        sym = f"{base}-{suffix}"

    base_len = len(sym.split("-")[0])
    if base_len == 0 or base_len > 5:
        return None
    return sym


def _name_is_unwanted(name: str) -> bool:
    text = str(name or "")
    return any(p.search(text) for p in _EXCLUDE_NAME_PATTERNS)


def build_universe(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter raw directory rows down to clean common stocks + normal ETFs."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=_UNIVERSE_COLUMNS)

    rows = []
    seen = set()
    for r in raw.itertuples(index=False):
        test_issue = str(getattr(r, "test_issue", "N")).strip().upper()
        if test_issue == "Y":
            continue  # never include test issues

        name = getattr(r, "name", "")
        if _name_is_unwanted(name):
            continue

        sym = _normalize_symbol(getattr(r, "symbol", ""))
        if not sym or sym in seen:
            continue
        seen.add(sym)

        is_etf = str(getattr(r, "is_etf", "N")).strip().upper() == "Y"
        rows.append({
            "symbol": sym,
            "name": str(name).strip(),
            "source_exchange": str(getattr(r, "source_exchange", "")).strip(),
            "instrument_type": "ETF" if is_etf else "STOCK",
        })

    df = pd.DataFrame(rows, columns=_UNIVERSE_COLUMNS)
    df.sort_values("symbol", inplace=True, ignore_index=True)
    logger.info("Built clean universe: %d tradable-looking tickers", len(df))
    return df


# ── Cache + public API ──────────────────────────────────────────────────────
def _cache_path():
    return config.SYMBOL_UNIVERSE_FILE


def _cache_age_hours() -> Optional[float]:
    path = _cache_path()
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 3600.0


def _read_cache() -> Optional[pd.DataFrame]:
    path = _cache_path()
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        missing = [c for c in _UNIVERSE_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("Cached universe missing columns %s — ignoring cache", missing)
            return None
        return df[_UNIVERSE_COLUMNS]
    except Exception:
        logger.warning("Could not read cached universe %s", path, exc_info=True)
        return None


def _write_cache(df: pd.DataFrame) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Cached symbol universe → %s (n=%d)", path, len(df))


def load_symbol_universe(
    force_refresh: bool = False,
    cache_hours: Optional[float] = None,
) -> pd.DataFrame:
    """Return the clean symbol universe DataFrame.

    Uses the local cache when it is fresh. Otherwise downloads, filters, and
    re-caches. If the download fails, gracefully falls back to whatever cache
    exists (even if stale) so a transient network failure never breaks a scan.
    """
    cache_hours = (
        cache_hours
        if cache_hours is not None
        else float(getattr(config, "FULL_MARKET_CACHE_HOURS", 24))
    )

    age = _cache_age_hours()
    if not force_refresh and age is not None and age < cache_hours:
        cached = _read_cache()
        if cached is not None and not cached.empty:
            logger.info("Using cached universe (age=%.1fh, n=%d)", age, len(cached))
            return cached

    try:
        raw = download_symbol_directories()
        universe = build_universe(raw)
        if universe.empty:
            raise ValueError("download produced an empty universe")
        _write_cache(universe)
        return universe
    except Exception:
        logger.warning("Universe download failed — falling back to cache", exc_info=True)
        cached = _read_cache()
        if cached is not None and not cached.empty:
            logger.info("Fallback to cached universe (n=%d)", len(cached))
            return cached
        logger.error("No cached universe available and download failed")
        return pd.DataFrame(columns=_UNIVERSE_COLUMNS)


def get_universe(
    limit: Optional[int] = None,
    exclude_etfs: Optional[bool] = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Universe DataFrame with optional ETF exclusion and a hard symbol cap."""
    df = load_symbol_universe(force_refresh=force_refresh)
    if df.empty:
        return df

    if exclude_etfs is None:
        exclude_etfs = bool(getattr(config, "HOT_SCAN_EXCLUDE_ETFS", False))
    if exclude_etfs:
        df = df[df["instrument_type"] != "ETF"]

    if limit is not None and limit > 0:
        df = df.head(int(limit))
    return df.reset_index(drop=True)


def get_universe_symbols(
    limit: Optional[int] = None,
    exclude_etfs: Optional[bool] = None,
    force_refresh: bool = False,
) -> List[str]:
    """Clean list of tradable-looking tickers (yfinance-normalized)."""
    return get_universe(
        limit=limit, exclude_etfs=exclude_etfs, force_refresh=force_refresh
    )["symbol"].tolist()


# ── Symbol selection (which slice of the universe to actually scan) ───────────
# The universe is stored alphabetically, so naively taking the first N symbols
# biases every scan toward A/AB tickers. These helpers pick a broad, rotating,
# or seeded-random slice instead. This is *selection only* — it reads no prices,
# predicts nothing, and places no orders.
SELECTION_MODES = ("alphabetical", "random", "rotation", "hybrid")

SELECTED_COLUMNS = _UNIVERSE_COLUMNS + ["source", "selection_reason"]


def _read_rotation_state() -> dict:
    path = config.FULL_MARKET_ROTATION_STATE_FILE
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.warning("Could not read rotation state %s — starting fresh", path, exc_info=True)
        return {}


def _write_rotation_state(state: dict) -> None:
    path = config.FULL_MARKET_ROTATION_STATE_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        logger.warning("Could not write rotation state %s", path, exc_info=True)


def _next_rotation_offset(key: str, length: int, step: int, advance: bool) -> int:
    """Return the current rotation offset for ``key`` and optionally advance it.

    Offsets are stored per-key (one for plain "rotation", one for the hybrid
    fill) so the two modes never fight over a single cursor.
    """
    if length <= 0:
        return 0
    state = _read_rotation_state()
    try:
        offset = int(state.get(key, 0)) % length
    except (TypeError, ValueError):
        offset = 0
    if advance:
        state[key] = (offset + max(0, int(step))) % length
        _write_rotation_state(state)
    return offset


def _slice_wrap(items: List[str], offset: int, n: int) -> List[str]:
    """Take ``n`` items starting at ``offset``, wrapping around the end."""
    if not items or n <= 0:
        return []
    size = len(items)
    n = min(n, size)
    offset %= size
    end = offset + n
    if end <= size:
        return items[offset:end]
    return items[offset:] + items[: end - size]


def select_symbols(
    max_symbols: Optional[int] = None,
    mode: Optional[str] = None,
    exclude_etfs: Optional[bool] = None,
    force_refresh: bool = False,
    seed: Optional[int] = None,
    advance_rotation: bool = True,
    write_report: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """Select which symbols to scan out of the broad universe.

    Returns ``(selected_df, info)`` where ``selected_df`` has the universe
    columns plus ``source`` and ``selection_reason``, and ``info`` summarizes the
    selection (universe size, mode, counts) for printing. Writes
    ``reports/selected_scan_symbols.csv`` when ``write_report`` is True.

    Pure discovery: it reads the cached/​downloaded directory only — no prices,
    no predictions, no orders.
    """
    mode = str(mode or getattr(config, "FULL_MARKET_SELECTION_MODE", "hybrid")).lower().strip()
    if mode not in SELECTION_MODES:
        logger.warning("Unknown selection mode %r — falling back to 'hybrid'", mode)
        mode = "hybrid"

    cap = (
        int(max_symbols)
        if max_symbols is not None
        else int(getattr(config, "FULL_MARKET_MAX_SYMBOLS_TO_CHECK", 500))
    )
    cap = max(1, cap)
    seed = int(seed if seed is not None else getattr(config, "FULL_MARKET_RANDOM_SEED", 42))

    df = get_universe(limit=None, exclude_etfs=exclude_etfs, force_refresh=force_refresh)
    universe_size = len(df)
    all_symbols = df["symbol"].tolist()

    # (symbol, source, reason) in final scan order.
    picks: List[Tuple[str, str, str]] = []

    if mode == "alphabetical":
        for s in all_symbols[:cap]:
            picks.append((s, "alphabetical", "first-N alphabetical (debug mode)"))

    elif mode == "random":
        pool = list(all_symbols)
        random.Random(seed).shuffle(pool)
        for s in pool[:cap]:
            picks.append((s, "random", f"random sample (seed={seed})"))

    elif mode == "rotation":
        offset = _next_rotation_offset("rotation", universe_size, cap, advance_rotation)
        for s in _slice_wrap(all_symbols, offset, cap):
            picks.append((s, "rotation", f"rotation slice @offset {offset}"))

    else:  # hybrid (default)
        universe_set = set(all_symbols)
        seen: set = set()
        core: List[str] = []
        for raw in getattr(config, "FULL_MARKET_CORE_SYMBOLS", []):
            sym = str(raw).upper().strip()
            if sym and sym in universe_set and sym not in seen:
                seen.add(sym)
                core.append(sym)
        core = core[:cap]
        for s in core:
            picks.append((s, "core", "core symbol (always included)"))

        fill_count = cap - len(core)
        if fill_count > 0:
            # Shuffle the non-core remainder (deterministically) so the fill is
            # never alphabetical, then rotate through it across runs.
            rest = [s for s in all_symbols if s not in seen]
            random.Random(seed).shuffle(rest)
            offset = _next_rotation_offset("hybrid", len(rest), fill_count, advance_rotation)
            for s in _slice_wrap(rest, offset, fill_count):
                picks.append((s, "fill", f"shuffled rotation fill @offset {offset} (seed={seed})"))

    # Build the selected DataFrame (carry universe metadata), deduplicated.
    meta_by_symbol: Dict[str, tuple] = {
        r.symbol: (r.name, r.source_exchange, r.instrument_type)
        for r in df.itertuples(index=False)
    }
    rows = []
    emitted: set = set()
    for sym, source, reason in picks:
        if sym in emitted:
            continue
        emitted.add(sym)
        name, exch, itype = meta_by_symbol.get(sym, ("", "", "STOCK"))
        rows.append({
            "symbol": sym, "name": name, "source_exchange": exch,
            "instrument_type": itype, "source": source, "selection_reason": reason,
        })
    selected_df = pd.DataFrame(rows, columns=SELECTED_COLUMNS)

    info = {
        "mode": mode,
        "universe_size": universe_size,
        "selected": len(selected_df),
        "core_included": int((selected_df["source"] == "core").sum()) if not selected_df.empty else 0,
        "first_20": selected_df["symbol"].head(20).tolist(),
    }

    if write_report:
        _write_selection_report(selected_df)

    logger.info(
        "Selected %d/%d symbols (mode=%s, core=%d)",
        info["selected"], universe_size, mode, info["core_included"],
    )
    return selected_df, info


def _write_selection_report(df: pd.DataFrame) -> None:
    path = config.SELECTED_SCAN_SYMBOLS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Selected-symbols report → %s (n=%d)", path, len(df))


if __name__ == "__main__":
    from logging_setup import setup_logging

    setup_logging()
    uni = load_symbol_universe()
    print(f"Universe size: {len(uni)}")
    print(uni.head(20).to_string(index=False))
