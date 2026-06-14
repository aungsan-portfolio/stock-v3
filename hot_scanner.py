"""
hot_scanner.py — Rank "hot" stock candidates by momentum + volume (READ-ONLY).

This module turns a symbol universe into a ranked shortlist of candidates. It is
purely a discovery/ranking step:

  * It fetches OHLCV via the existing ``data_manager.fetch_ohlcv``.
  * A bad/empty ticker is logged and skipped — one failure never aborts a scan.
  * "Hot" means momentum + volume + trend, *not* a buy signal and *not* a
    profit guarantee. Prediction (Predictor) and the IBKR risk rules still decide
    BUY/HOLD/SELL and whether any paper order is ever placed.

Public API:
    scan_hot_stocks(universe=None, top_n=None, full_market=False,
                    max_symbols=None, write_report=True) -> list[str]

Report (when write_report=True): reports/hot_candidates.csv
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from data_manager import fetch_ohlcv

logger = logging.getLogger(__name__)

REPORT_COLUMNS = [
    "symbol", "name", "source_exchange", "instrument_type",
    "price", "avg_volume_20d", "ret_1d", "ret_5d", "ret_20d",
    "vol_ratio", "atr_pct", "above_sma20", "above_sma50",
    "hot_score", "reason",
]

# Stats from the most recent scan_hot_stocks() call. Read-only convenience for
# callers (e.g. daily-coach) that want to report how much market was scanned
# without changing the function's list[str] return type. Updated in place each
# scan; never relied upon for any trading decision.
LAST_SCAN_STATS: Dict[str, int] = {
    "universe_size": 0,
    "scanned": 0,
    "passed_filters": 0,
    "kept": 0,
    "failed": 0,
}


# ── Metric computation ──────────────────────────────────────────────────────
def _compute_metrics(df: pd.DataFrame) -> Optional[dict]:
    """Compute momentum/volume/trend metrics from an OHLCV frame.

    Returns None if the data is missing, empty, or too short to be meaningful.
    """
    if df is None or df.empty or len(df) < 50:
        return None

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    price = float(close.iloc[-1])
    if not np.isfinite(price) or price <= 0:
        return None

    avg_volume_20d = float(volume.rolling(20).mean().iloc[-1])
    last_volume = float(volume.iloc[-1])
    vol_ratio = (
        last_volume / avg_volume_20d
        if np.isfinite(avg_volume_20d) and avg_volume_20d > 0
        else 0.0
    )

    def _pct(n: int) -> float:
        if len(close) <= n:
            return 0.0
        prev = float(close.iloc[-1 - n])
        if prev <= 0:
            return 0.0
        return price / prev - 1.0

    ret_1d = _pct(1)
    ret_5d = _pct(5)
    ret_20d = _pct(20)

    # ATR(14) as a fraction of price.
    prev_close = close.shift()
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = float(true_range.rolling(config.ATR_PERIOD).mean().iloc[-1])
    atr_pct = atr / price if np.isfinite(atr) and price > 0 else 0.0

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    above_sma20 = bool(np.isfinite(sma20) and price > sma20)
    above_sma50 = bool(np.isfinite(sma50) and price > sma50)

    if not np.isfinite(avg_volume_20d):
        avg_volume_20d = 0.0

    return {
        "price": round(price, 2),
        "avg_volume_20d": int(avg_volume_20d),
        "ret_1d": round(ret_1d, 4),
        "ret_5d": round(ret_5d, 4),
        "ret_20d": round(ret_20d, 4),
        "vol_ratio": round(float(vol_ratio), 3),
        "atr_pct": round(float(atr_pct), 4),
        "above_sma20": above_sma20,
        "above_sma50": above_sma50,
    }


def _score_and_filter(m: dict) -> Optional[dict]:
    """Apply hard rejection filters, then score the survivors.

    Returns the metrics dict augmented with ``hot_score`` + ``reason``, or None
    if the symbol is rejected (out of price band, illiquid, too volatile, …).
    """
    price = m["price"]
    if price < float(config.HOT_SCAN_MIN_PRICE):
        return None
    if price > float(config.HOT_SCAN_MAX_PRICE):
        return None
    if m["avg_volume_20d"] < float(config.HOT_SCAN_MIN_AVG_VOLUME):
        return None
    if m["atr_pct"] > float(config.HOT_SCAN_MAX_ATR_PCT):
        return None

    # Continuous, interpretable score. Reward momentum + participation + trend;
    # penalize excess volatility. Weights favor shorter-horizon momentum.
    score = 0.0
    score += 100.0 * m["ret_1d"] * 1.0
    score += 100.0 * m["ret_5d"] * 0.6
    score += 100.0 * m["ret_20d"] * 0.3

    if m["vol_ratio"] > 1.2:
        score += min((m["vol_ratio"] - 1.0) * 10.0, 20.0)

    if m["above_sma20"]:
        score += 5.0
    if m["above_sma50"]:
        score += 5.0

    # Penalize volatility above a calm ~4% ATR baseline.
    score -= max(0.0, m["atr_pct"] - 0.04) * 100.0

    reason = (
        f"r1={m['ret_1d']*100:+.1f}% r5={m['ret_5d']*100:+.1f}% "
        f"r20={m['ret_20d']*100:+.1f}% volx={m['vol_ratio']:.2f} "
        f"atr={m['atr_pct']*100:.1f}% "
        f"{'>SMA20 ' if m['above_sma20'] else ''}"
        f"{'>SMA50' if m['above_sma50'] else ''}"
    ).strip()

    out = dict(m)
    out["hot_score"] = round(float(score), 2)
    out["reason"] = reason
    return out


# ── Universe resolution ─────────────────────────────────────────────────────
def _resolve_universe(
    universe: Optional[List[str]],
    full_market: bool,
    max_symbols: Optional[int],
    selection_mode: Optional[str] = None,
) -> "tuple[List[str], Dict[str, dict], Optional[dict]]":
    """Return (symbols, meta, selection_info).

    ``meta`` maps symbol -> name/exchange/type. ``selection_info`` summarizes how
    a full-market slice was chosen (None when not a full-market scan).
    """
    cap = (
        int(max_symbols)
        if max_symbols is not None
        else int(getattr(config, "FULL_MARKET_MAX_SYMBOLS_TO_CHECK", 500))
    )

    if universe:
        symbols = [str(s).upper().strip() for s in universe if str(s).strip()]
        return symbols[:cap], {}, None

    if full_market:
        if not bool(getattr(config, "FULL_MARKET_SCAN_ENABLED", True)):
            logger.warning("FULL_MARKET_SCAN_ENABLED is False — using WATCHLIST instead")
            return list(config.WATCHLIST), {}, None
        import market_universe
        selected_df, info = market_universe.select_symbols(
            max_symbols=cap, mode=selection_mode, write_report=True
        )
        meta = {
            row.symbol: {
                "name": row.name,
                "source_exchange": row.source_exchange,
                "instrument_type": row.instrument_type,
            }
            for row in selected_df.itertuples(index=False)
        }
        return selected_df["symbol"].tolist(), meta, info

    # Default: small, safe static universe.
    return list(config.WATCHLIST), {}, None


def _print_selection(info: dict) -> None:
    """Announce which slice of the broad market this run will scan."""
    first20 = ", ".join(info.get("first_20", [])) or "(none)"
    print("📡 Full-market symbol selection:")
    print(f"   universe size      : {info.get('universe_size', 0)}")
    print(f"   selection mode     : {info.get('mode', '?')}")
    print(f"   selected symbols   : {info.get('selected', 0)}")
    print(f"   core symbols incl. : {info.get('core_included', 0)}")
    print(f"   first 20 selected  : {first20}")


# ── Public scan ─────────────────────────────────────────────────────────────
def scan_hot_stocks(
    universe: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    full_market: bool = False,
    max_symbols: Optional[int] = None,
    write_report: bool = True,
    selection_mode: Optional[str] = None,
) -> List[str]:
    """Scan a universe and return the top hot candidate symbols.

    This performs NO trading and places NO orders. It only reads price data and
    writes a CSV report of ranked candidates.
    """
    top_n = int(top_n) if top_n is not None else int(getattr(config, "HOT_SCAN_TOP_N", 30))
    symbols, meta, selection_info = _resolve_universe(
        universe, full_market, max_symbols, selection_mode
    )

    if selection_info is not None:
        _print_selection(selection_info)

    if not symbols:
        print("⚠️  No symbols to scan (empty universe).")
        LAST_SCAN_STATS.update(
            {"universe_size": 0, "scanned": 0, "passed_filters": 0, "kept": 0, "failed": 0}
        )
        if write_report:
            _write_report([])
        return []

    chunk_size = max(1, int(getattr(config, "HOT_SCAN_CHUNK_SIZE", 50)))
    sleep_seconds = float(getattr(config, "HOT_SCAN_SLEEP_SECONDS", 1.0))
    total = len(symbols)
    mode = "FULL-MARKET" if full_market else "watchlist"
    print(f"🔎 Scanning {total} symbols ({mode}) for hot candidates…")

    candidates: List[dict] = []
    scanned = 0
    failed = 0

    for start in range(0, total, chunk_size):
        chunk = symbols[start:start + chunk_size]
        for symbol in chunk:
            scanned += 1
            try:
                df = fetch_ohlcv(symbol)
                metrics = _compute_metrics(df)
                if metrics is None:
                    continue
                scored = _score_and_filter(metrics)
                if scored is None:
                    continue
                info = meta.get(symbol, {})
                scored.update({
                    "symbol": symbol,
                    "name": info.get("name", ""),
                    "source_exchange": info.get("source_exchange", ""),
                    "instrument_type": info.get("instrument_type", "STOCK"),
                })
                candidates.append(scored)
            except Exception:
                failed += 1
                logger.debug("Scan skipped %s", symbol, exc_info=True)

        done = min(start + chunk_size, total)
        print(f"   …{done}/{total} scanned | {len(candidates)} candidates | {failed} skipped")
        # Throttle between chunks (not after the final chunk).
        if sleep_seconds > 0 and done < total:
            time.sleep(sleep_seconds)

    candidates.sort(key=lambda c: c["hot_score"], reverse=True)
    top = candidates[:top_n]

    LAST_SCAN_STATS.update({
        "universe_size": total,
        "scanned": scanned,
        "passed_filters": len(candidates),
        "kept": len(top),
        "failed": failed,
    })

    print(
        f"✅ Scan complete: {scanned} scanned, {len(candidates)} passed filters, "
        f"keeping top {len(top)}."
    )

    if write_report:
        _write_report(top)

    return [c["symbol"] for c in top]


def _write_report(rows: List[dict]) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    df.to_csv(config.HOT_CANDIDATES_FILE, index=False)
    logger.info("Hot candidates report → %s (n=%d)", config.HOT_CANDIDATES_FILE, len(df))
    print(f"💾 Report → {config.HOT_CANDIDATES_FILE}")


if __name__ == "__main__":
    from logging_setup import setup_logging

    setup_logging()
    hot = scan_hot_stocks(full_market=False, top_n=10)
    print("\nTop hot candidates:", hot)
