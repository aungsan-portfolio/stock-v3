"""
strategies/sector_rotation_monitor.py — Observe-Only Sector Relative Strength Monitor.

Strict Scope & Safety Invariants:
1. Purely observational telemetry for Sector Benchmark ETFs (XLV, XLP, XLE, XLK, XLF, XLU, SOXX vs SPY).
2. Never issues BUY/SELL signals, never places orders, never alters portfolio holdings, never rotates capital.
3. Aligns completed daily bars with timestamp, freshness, and completeness validation; fails closed on missing/stale data.
4. Non-tradable guarantee: tradable attribute is permanently False.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from data_manager import fetch_ohlcv
from strategies.session import now_eastern

logger = logging.getLogger(__name__)

SECTOR_BENCHMARK_NAMES: Dict[str, str] = {
    "XLV": "Healthcare",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLK": "Technology",
    "XLF": "Financials",
    "XLU": "Utilities",
    "SOXX": "Semiconductors",
}

DEFAULT_TELEMETRY_LOG = Path(getattr(config, "REPORTS_DIR", "reports")) / "sector_rotation_telemetry.jsonl"


@dataclass(frozen=True)
class SectorObservationTelemetry:
    """Immutable observe-only telemetry record for a single sector benchmark ETF."""
    timestamp: str
    symbol: str
    sector_name: str
    benchmark: str
    latest_bar_date: str
    ret_1d_etf: Optional[float]
    ret_1d_spy: Optional[float]
    rs_1d: Optional[float]
    ret_5d_etf: Optional[float]
    ret_5d_spy: Optional[float]
    rs_5d: Optional[float]
    ret_20d_etf: Optional[float]
    ret_20d_spy: Optional[float]
    rs_20d: Optional[float]
    valid: bool
    status: str   # "VALID" | "VETO_STALE" | "VETO_INSUFFICIENT_HISTORY" | "VETO_DATA_INVALID" | "VETO_TIMESTAMP_MISALIGNMENT"
    reason: str
    provenance: str
    tradable: bool = False  # GUARANTEED NON-TRADABLE

    def to_dict(self) -> dict:
        return asdict(self)


class SectorRotationMonitor:
    """
    Isolated, observe-only Sector Relative Strength Monitor.

    Evaluates 1-day, 5-day, and 20-day relative strength of sector benchmark ETFs
    against SPY using strictly completed daily bars. Emits structured telemetry
    without any execution or order-mutation capability.
    """

    def __init__(self, telemetry_path: Optional[Path] = None) -> None:
        self.telemetry_path = telemetry_path or DEFAULT_TELEMETRY_LOG
        self.benchmarks = tuple(
            s.upper().strip() for s in getattr(config, "SECTOR_BENCHMARKS", (
                "SPY", "XLV", "XLP", "XLE", "XLK", "XLF", "XLU", "SOXX"
            ))
        )

    def compute_sector_relative_strength(
        self,
        symbol: str,
        symbol_df: Optional[pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame],
        benchmark_symbol: str = "SPY",
        now_date: Optional[date] = None,
        max_staleness_days: int = 5,
    ) -> SectorObservationTelemetry:
        """
        Compute aligned 1d, 5d, and 20d sector relative returns vs benchmark.
        Strictly fails closed on missing data, misalignment, non-finite values, or staleness.
        """
        sym = str(symbol).upper().strip()
        bm = str(benchmark_symbol).upper().strip()
        sec_name = SECTOR_BENCHMARK_NAMES.get(sym, "Unknown Sector")
        now_ts = now_eastern().isoformat()
        eval_date = now_date or now_eastern().date()

        # ── 1b. Exclude Partial Daily Candle During Live Trading (B3) ─────────
        # During live market hours (< 16:00 ET), today's daily bar is incomplete/partial.
        # Stripping today's partial bar aligns both ETF and Benchmark strictly to the
        # latest completed daily bar, eliminating persistent daily VETO_TIMESTAMP_MISALIGNMENT.
        if now_date is None and symbol_df is not None and benchmark_df is not None:
            try:
                now_et_b3 = now_eastern()
                if now_et_b3.hour < 16:
                    today_date = now_et_b3.date()
                    if isinstance(symbol_df.index, pd.DatetimeIndex):
                        symbol_df = symbol_df[symbol_df.index.date < today_date]
                    if isinstance(benchmark_df.index, pd.DatetimeIndex):
                        benchmark_df = benchmark_df[benchmark_df.index.date < today_date]
            except Exception as b3_err:
                logger.debug("[SECTOR MONITOR B3] Partial bar alignment error: %s", b3_err)

        # ── 1. Null / Empty Validation ───────────────────────────────────────
        if symbol_df is None or symbol_df.empty:
            return self._fail_closed(
                sym, sec_name, bm, now_ts, "UNKNOWN",
                "VETO_DATA_INVALID", f"{sym} DataFrame is None or empty",
            )
        if benchmark_df is None or benchmark_df.empty:
            return self._fail_closed(
                sym, sec_name, bm, now_ts, "UNKNOWN",
                "VETO_DATA_INVALID", f"Benchmark {bm} DataFrame is None or empty",
            )

        # ── 2. DatetimeIndex Validation ──────────────────────────────────────
        if not isinstance(symbol_df.index, pd.DatetimeIndex) or not isinstance(benchmark_df.index, pd.DatetimeIndex):
            return self._fail_closed(
                sym, sec_name, bm, now_ts, "UNKNOWN",
                "VETO_DATA_INVALID", "DataFrames must have DatetimeIndex",
            )

        # ── 3. Align Completed Daily Bars (Inner Join) ───────────────────────
        s_idx = symbol_df.index.normalize()
        b_idx = benchmark_df.index.normalize()
        common_dates = s_idx.intersection(b_idx)

        if len(common_dates) < 21:
            return self._fail_closed(
                sym, sec_name, bm, now_ts, "UNKNOWN",
                "VETO_INSUFFICIENT_HISTORY",
                f"Insufficient aligned completed daily bars: {len(common_dates)} < 21 required for 20d RS",
            )

        # Extract aligned close series
        # Note: mapping back to normalized index to ensure 1-to-1 alignment
        s_series = symbol_df["Close"].copy()
        s_series.index = s_idx
        s_aligned = s_series.loc[common_dates].sort_index()
        s_aligned = s_aligned[~s_aligned.index.duplicated(keep="last")]

        b_series = benchmark_df["Close"].copy()
        b_series.index = b_idx
        b_aligned = b_series.loc[common_dates].sort_index()
        b_aligned = b_aligned[~b_aligned.index.duplicated(keep="last")]

        aligned_count = min(len(s_aligned), len(b_aligned))
        if aligned_count < 21:
            return self._fail_closed(
                sym, sec_name, bm, now_ts, "UNKNOWN",
                "VETO_INSUFFICIENT_HISTORY",
                f"Deduplicated aligned bars {aligned_count} < 21 required",
            )

        # ── 4. Monotonicity & Latest Bar Alignment ───────────────────────────
        if not s_aligned.index.is_monotonic_increasing or not b_aligned.index.is_monotonic_increasing:
            return self._fail_closed(
                sym, sec_name, bm, now_ts, "UNKNOWN",
                "VETO_DATA_INVALID", "Aligned timestamps are not monotonic increasing",
            )

        raw_s_latest = symbol_df.index[-1].date()
        raw_b_latest = benchmark_df.index[-1].date()
        if raw_s_latest != raw_b_latest:
            return self._fail_closed(
                sym, sec_name, bm, now_ts, str(raw_s_latest),
                "VETO_TIMESTAMP_MISALIGNMENT",
                f"Latest completed daily bar mismatch: {sym}={raw_s_latest} vs {bm}={raw_b_latest}",
            )

        latest_s_date = s_aligned.index[-1].date()

        # ── 5. Staleness Validation ──────────────────────────────────────────
        age_days = (eval_date - latest_s_date).days
        if age_days > max_staleness_days:
            return self._fail_closed(
                sym, sec_name, bm, now_ts, str(latest_s_date),
                "VETO_STALE",
                f"Stale daily bars: latest bar {latest_s_date} is {age_days}d old (> max {max_staleness_days}d)",
            )

        # ── 6. Numerical Validity Check ──────────────────────────────────────
        s_tail21 = s_aligned.iloc[-21:].astype(float)
        b_tail21 = b_aligned.iloc[-21:].astype(float)

        if not np.isfinite(s_tail21).all() or not np.isfinite(b_tail21).all():
            return self._fail_closed(
                sym, sec_name, bm, now_ts, str(latest_s_date),
                "VETO_DATA_INVALID", "Non-finite prices detected in trailing 21-bar window",
            )

        if (s_tail21 <= 0).any() or (b_tail21 <= 0).any():
            return self._fail_closed(
                sym, sec_name, bm, now_ts, str(latest_s_date),
                "VETO_DATA_INVALID", "Non-positive prices detected in trailing 21-bar window",
            )

        # ── 7. Aligned Returns & Relative Strength Computation ───────────────
        p_s_now = float(s_tail21.iloc[-1])
        p_b_now = float(b_tail21.iloc[-1])

        # 1-day horizon
        p_s_1d = float(s_tail21.iloc[-2])
        p_b_1d = float(b_tail21.iloc[-2])
        ret_1d_etf = (p_s_now - p_s_1d) / p_s_1d
        ret_1d_spy = (p_b_now - p_b_1d) / p_b_1d
        rs_1d = ret_1d_etf - ret_1d_spy

        # 5-day horizon
        p_s_5d = float(s_tail21.iloc[-6])
        p_b_5d = float(b_tail21.iloc[-6])
        ret_5d_etf = (p_s_now - p_s_5d) / p_s_5d
        ret_5d_spy = (p_b_now - p_b_5d) / p_b_5d
        rs_5d = ret_5d_etf - ret_5d_spy

        # 20-day horizon
        p_s_20d = float(s_tail21.iloc[-21])
        p_b_20d = float(b_tail21.iloc[-21])
        ret_20d_etf = (p_s_now - p_s_20d) / p_s_20d
        ret_20d_spy = (p_b_now - p_b_20d) / p_b_20d
        rs_20d = ret_20d_etf - ret_20d_spy

        provenance = getattr(symbol_df, "attrs", {}).get("volume_source", "data_manager.fetch_ohlcv")

        telemetry = SectorObservationTelemetry(
            timestamp=now_ts,
            symbol=sym,
            sector_name=sec_name,
            benchmark=bm,
            latest_bar_date=str(latest_s_date),
            ret_1d_etf=round(ret_1d_etf, 6),
            ret_1d_spy=round(ret_1d_spy, 6),
            rs_1d=round(rs_1d, 6),
            ret_5d_etf=round(ret_5d_etf, 6),
            ret_5d_spy=round(ret_5d_spy, 6),
            rs_5d=round(rs_5d, 6),
            ret_20d_etf=round(ret_20d_etf, 6),
            ret_20d_spy=round(ret_20d_spy, 6),
            rs_20d=round(rs_20d, 6),
            valid=True,
            status="VALID",
            reason="Aligned completed daily bar calculation successful",
            provenance=str(provenance),
            tradable=False,
        )

        logger.info(
            "[SECTOR ROTATION TELEMETRY] %s (%s vs %s) | RS_1d=%+.2f%% | RS_5d=%+.2f%% | RS_20d=%+.2f%% | valid=True",
            sym, sec_name, bm, rs_1d * 100.0, rs_5d * 100.0, rs_20d * 100.0
        )
        return telemetry

    def scan_all_benchmarks(self, force_refresh: bool = False) -> Dict[str, SectorObservationTelemetry]:
        """
        Scan all configured observation benchmarks against SPY.
        Appends telemetry records to reports/sector_rotation_telemetry.jsonl.
        Strictly observe-only; zero mutation.
        """
        results: Dict[str, SectorObservationTelemetry] = {}
        try:
            spy_df = fetch_ohlcv("SPY", force_refresh=force_refresh)
        except Exception as exc:
            logger.error("[SECTOR ROTATION MONITOR] Failed to fetch benchmark SPY: %s", exc)
            return results

        benchmarks_to_scan = [s for s in self.benchmarks if s != "SPY"]
        for sym in benchmarks_to_scan:
            try:
                sym_df = fetch_ohlcv(sym, force_refresh=force_refresh)
                telem = self.compute_sector_relative_strength(
                    symbol=sym,
                    symbol_df=sym_df,
                    benchmark_df=spy_df,
                    benchmark_symbol="SPY",
                )
                results[sym] = telem
                self._append_telemetry(telem)
            except Exception as exc:
                logger.warning("[SECTOR ROTATION MONITOR] Failed scanning %s: %s", sym, exc)
                veto_telem = self._fail_closed(
                    sym, SECTOR_BENCHMARK_NAMES.get(sym, "Unknown Sector"),
                    "SPY", now_eastern().isoformat(), "UNKNOWN",
                    "VETO_DATA_INVALID", f"Fetch or evaluation exception: {exc}",
                )
                results[sym] = veto_telem
                self._append_telemetry(veto_telem)

        return results

    def _append_telemetry(self, telemetry: SectorObservationTelemetry) -> None:
        """Persist telemetry record to jsonl log file."""
        try:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.telemetry_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(telemetry.to_dict()) + "\n")
        except Exception as exc:
            logger.warning("[SECTOR ROTATION MONITOR] Could not write telemetry log: %s", exc)

    def _fail_closed(
        self,
        symbol: str,
        sector_name: str,
        benchmark: str,
        timestamp: str,
        latest_bar_date: str,
        status: str,
        reason: str,
    ) -> SectorObservationTelemetry:
        """Helper to construct fail-closed telemetry record."""
        logger.warning("[SECTOR ROTATION VETO] %s (%s): status=%s reason=%s", symbol, sector_name, status, reason)
        return SectorObservationTelemetry(
            timestamp=timestamp,
            symbol=symbol,
            sector_name=sector_name,
            benchmark=benchmark,
            latest_bar_date=latest_bar_date,
            ret_1d_etf=None,
            ret_1d_spy=None,
            rs_1d=None,
            ret_5d_etf=None,
            ret_5d_spy=None,
            rs_5d=None,
            ret_20d_etf=None,
            ret_20d_spy=None,
            rs_20d=None,
            valid=False,
            status=status,
            reason=reason,
            provenance="failed_closed",
            tradable=False,
        )
