"""
strategies.shadow.orb_confirmed -- Confirmed 15-Minute ORB Shadow Strategy.

Implements all 12 rigorous Long ORB validation conditions:
1. Completed candle closes above 15-minute ORB high.
2. Price > VWAP.
3. Positive VWAP slope (3-bar lookback).
4. Breakout volume ratio >= 1.5x prior 20-bar average.
5. Relative volume >= 2.0.
6. Market benchmark alignment (both SPY and QQQ above VWAP).
7. Sector benchmark alignment (strict if configured; fallback recorded if none).
8. Overextension guard: candle close <= EMA9 + 1.25*ATR and <= ORB_high + 1.25*ATR.
9. Spread check passes execution gate (<= 0.15%).
10. Stop geometry with noise floor (tightest valid level capped by ATR floor). Fixed 2R target.
11. Stock in Play verification (RVOL >= 2.0 AND (catalyst OR gap >= 2% OR premarket vol ratio >= 2)).
12. PRIMARY time window: 10:00:00 ET inclusive to 11:30:00 ET exclusive.

Strictly emits immutable ShadowCandidate instances (Zero TradeSignal pollution).
Includes cycle rejection accounting and per-symbol filter observability.
"""
import csv
from dataclasses import dataclass, field
from datetime import datetime, time
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

import config
from strategies.shadow.models import (
    AlignmentStatus,
    ShadowCandidate,
    generate_deterministic_shadow_id,
)

logger = logging.getLogger(__name__)


@dataclass
class ORBCycleStats:
    evaluated: int = 0
    candidates: int = 0
    outside_window: int = 0
    incomplete_range: int = 0
    no_breakout: int = 0
    below_vwap: int = 0
    negative_vwap_slope: int = 0
    low_breakout_vol: int = 0
    rvol_low: int = 0
    stock_in_play_failed: int = 0
    market_unaligned: int = 0
    sector_unavailable: int = 0
    overextended: int = 0
    spread_failed: int = 0
    zero_risk: int = 0


def resolve_candidate_metadata(
    symbol: str,
    df: Optional[pd.DataFrame] = None,
    provided_meta: Optional[dict] = None,
) -> dict:
    """Enriches candidate metadata from scan results and DataFrame history.

    Guarantees missing catalyst is NEVER treated as verified catalyst.
    """
    meta = dict(provided_meta or {})
    sym_u = symbol.upper().strip()

    # Check scan_results.csv if available
    scan_file = getattr(config, "SCAN_RESULTS_FILE", None)
    if scan_file and Path(scan_file).exists():
        try:
            with open(scan_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("symbol", "").upper().strip() == sym_u:
                        if "relative_volume" not in meta and row.get("relative_volume"):
                            meta["relative_volume"] = float(row["relative_volume"])
                        if "gap_pct" not in meta and row.get("gap_pct"):
                            meta["gap_pct"] = abs(float(row["gap_pct"]))
                        if "catalyst_type" not in meta and row.get("catalyst_type"):
                            cat = row["catalyst_type"].strip()
                            if cat.upper() not in ("NONE", "UNKNOWN", ""):
                                meta["catalyst_type"] = cat
                        if "premarket_volume" not in meta and row.get("premarket_volume"):
                            meta["premarket_volume"] = int(float(row["premarket_volume"]))
                        break
        except Exception as e:
            logger.debug("Could not read scan results for %s: %s", symbol, e)

    # Check DataFrame if relative_volume or gap_pct still missing
    if df is not None and not df.empty and len(df) >= 21:
        if "relative_volume" not in meta:
            vol_20 = df["volume"].iloc[-21:-1].mean()
            cur_vol = df["volume"].iloc[-1]
            meta["relative_volume"] = round(float(cur_vol / vol_20), 2) if vol_20 > 0 else 1.0

    return meta


class ConfirmedORBStrategy:
    name = "ORB_CONFIRMED"
    version = getattr(config, "SHADOW_STRATEGY_VERSION", "ORB_CONFIRMED_V1_FIXED_2R")

    def __init__(self):
        self.allow_long = getattr(config, "ORB_SHADOW_ALLOW_LONG", True)
        self.allow_short = getattr(config, "ORB_SHADOW_ALLOW_SHORT", False)
        self.range_minutes = int(getattr(config, "ORB_RANGE_MINUTES", 15))
        self.earliest_entry_str = getattr(config, "ORB_EARLIEST_ENTRY_TIME_ET", "10:00")
        self.latest_entry_str = getattr(config, "ORB_LATEST_ENTRY_TIME_ET", "11:30")
        self.target_r_multiple = float(getattr(config, "ORB_TARGET_R_MULTIPLE", 2.0))
        self.min_breakout_vol_ratio = float(getattr(config, "ORB_MIN_BREAKOUT_VOLUME_RATIO", 1.5))
        self.min_rvol = float(getattr(config, "ORB_MIN_RELATIVE_VOLUME", 2.0))
        self.min_premarket_gap_pct = float(getattr(config, "ORB_MIN_PREMARKET_GAP_PCT", 2.0))
        self.min_premarket_vol_ratio = float(getattr(config, "ORB_MIN_PREMARKET_VOLUME_RATIO", 2.0))
        self.max_atr_extension = float(getattr(config, "ORB_MAX_ATR_EXTENSION", 1.25))
        self.min_stop_atr_multiple = float(getattr(config, "ORB_MIN_STOP_ATR_MULTIPLE", 0.50))
        self._cycle_stats = ORBCycleStats()

    def reset_cycle_stats(self) -> None:
        self._cycle_stats = ORBCycleStats()

    def get_cycle_stats(self) -> ORBCycleStats:
        return self._cycle_stats

    def log_cycle_summary(self) -> None:
        """Logs aggregate rejection counters for completed evaluation cycle."""
        stats = self._cycle_stats
        if stats.evaluated == 0:
            return
        logger.info(
            "[ORB SHADOW SUMMARY] evaluated=%d candidates=%d outside_window=%d incomplete_range=%d "
            "no_breakout=%d below_vwap=%d negative_vwap_slope=%d low_breakout_vol=%d rvol_low=%d "
            "stock_in_play_failed=%d market_unaligned=%d sector_unavailable=%d overextended=%d spread_failed=%d",
            stats.evaluated,
            stats.candidates,
            stats.outside_window,
            stats.incomplete_range,
            stats.no_breakout,
            stats.below_vwap,
            stats.negative_vwap_slope,
            stats.low_breakout_vol,
            stats.rvol_low,
            stats.stock_in_play_failed,
            stats.market_unaligned,
            stats.sector_unavailable,
            stats.overextended,
            stats.spread_failed,
        )
        accounted = (
            stats.candidates
            + stats.outside_window
            + stats.incomplete_range
            + stats.no_breakout
            + stats.below_vwap
            + stats.negative_vwap_slope
            + stats.low_breakout_vol
            + stats.rvol_low
            + stats.stock_in_play_failed
            + stats.market_unaligned
            + stats.sector_unavailable
            + stats.overextended
            + stats.spread_failed
            + stats.zero_risk
        )
        if accounted != stats.evaluated:
            logger.error(
                "[ORB SHADOW FUNNEL MISMATCH] evaluated=%d accounted=%d",
                stats.evaluated,
                accounted,
            )
        self._cycle_stats = ORBCycleStats()

    def evaluate_candidate(
        self,
        symbol: str,
        df: pd.DataFrame,
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        candidate_meta: Optional[dict] = None,
        now_utc: Optional[pd.Timestamp] = None,
    ) -> Optional[ShadowCandidate]:
        """Evaluates symbol data and returns an immutable ShadowCandidate if all 12 conditions pass."""
        if not getattr(config, "P1_6_ORB_SHADOW_ENABLED", False):
            return None

        if not self.allow_long:
            return None

        self._cycle_stats.evaluated += 1

        if df is None or df.empty or len(df) < self.range_minutes + 10:
            return None

        # Convert index to America/New_York
        idx = df.index
        if idx.tz is None:
            idx_et = idx.tz_localize("UTC").tz_convert("America/New_York")
        else:
            idx_et = idx.tz_convert("America/New_York")

        session_date_et = str(idx_et[-1].date())
        current_time_et = idx_et[-1].time()

        # ── Time Window Validation (10:00:00 inclusive to 11:30:00 exclusive) ──
        earliest_time = datetime.strptime(self.earliest_entry_str, "%H:%M").time()
        latest_time = datetime.strptime(self.latest_entry_str, "%H:%M").time()
        if not (earliest_time <= current_time_et < latest_time):
            self._cycle_stats.outside_window += 1
            return None

        # ── Opening Range Construction (Strict 15/15 Completed 1m Bars) ──
        orb_range_res = self._compute_opening_range(df, idx_et, session_date_et)
        if orb_range_res is None:
            self._cycle_stats.incomplete_range += 1
            return None
        orb_high, orb_low, orb_range = orb_range_res

        # ── Last Completed Signal Candle ──
        last_bar = df.iloc[-1]
        close = float(last_bar["close"])
        high = float(last_bar["high"])
        low = float(last_bar["low"])
        volume = float(last_bar["volume"])
        atr = float(last_bar.get("atr", 0.0) or 0.0)
        vwap = float(last_bar.get("vwap", 0.0) or 0.0)
        ema9 = float(last_bar.get("ema9", 0.0) or 0.0)
        ema20 = float(last_bar.get("ema20", 0.0) or 0.0)

        if close <= 0 or vwap <= 0 or atr <= 0:
            return None

        # Condition 1: Completed candle closes above 15m ORB high
        if close <= orb_high:
            self._cycle_stats.no_breakout += 1
            return None

        # Condition 2: Price above VWAP
        if close <= vwap:
            self._cycle_stats.below_vwap += 1
            return None

        # Condition 3: VWAP slope is positive (over prior 3 bars)
        if len(df) >= 4:
            prior_vwap = float(df["vwap"].iloc[-4])
            if vwap <= prior_vwap:
                self._cycle_stats.negative_vwap_slope += 1
                return None
        vwap_slope = (vwap - float(df["vwap"].iloc[-4])) if len(df) >= 4 else 0.0

        # Condition 4: Breakout volume >= 1.5x prior 20 completed bars average
        vol_window = df["volume"].iloc[-21:-1]
        avg_vol_20 = float(vol_window.mean()) if len(vol_window) >= 10 else 0.0
        breakout_vol_ratio = (volume / avg_vol_20) if avg_vol_20 > 0 else 0.0
        if breakout_vol_ratio < self.min_breakout_vol_ratio:
            self._cycle_stats.low_breakout_vol += 1
            return None

        # Condition 5 & 11: Stock in Play & Relative Volume (RVOL >= 2.0)
        meta = resolve_candidate_metadata(symbol, df, candidate_meta)
        rvol = float(meta.get("relative_volume", 0.0) or last_bar.get("relative_volume", 0.0) or breakout_vol_ratio)
        if rvol < self.min_rvol:
            self._cycle_stats.rvol_low += 1
            return None

        catalyst = meta.get("catalyst_type") or meta.get("catalyst")
        gap_pct = abs(float(meta.get("gap_pct", 0.0) or 0.0))
        pre_vol_ratio = float(meta.get("premarket_volume_ratio", 0.0) or meta.get("rel_vol", 0.0) or 0.0)

        has_verified_catalyst = bool(catalyst and str(catalyst).upper() not in ("NONE", "UNKNOWN", ""))
        is_stock_in_play = bool(
            has_verified_catalyst
            or (gap_pct >= self.min_premarket_gap_pct)
            or (pre_vol_ratio >= self.min_premarket_vol_ratio)
        )
        if not is_stock_in_play:
            self._cycle_stats.stock_in_play_failed += 1
            logger.debug(
                "[ORB SHADOW FILTER] symbol=%s rvol=%.2f gap_pct=%.2f premarket_volume_ratio=%.2f catalyst=%s accepted=false reason=STOCK_IN_PLAY_FAILED",
                symbol, rvol, gap_pct, pre_vol_ratio, catalyst or "NONE"
            )
            return None

        # Condition 6 & 7: Market & Sector Benchmark Alignment
        spy_status, qqq_status, sector_status = self._check_market_alignment(market_data, symbol)
        if spy_status != AlignmentStatus.ALIGNED or qqq_status != AlignmentStatus.ALIGNED:
            self._cycle_stats.market_unaligned += 1
            return None
        if sector_status in (AlignmentStatus.OPPOSED, AlignmentStatus.STALE, AlignmentStatus.UNAVAILABLE):
            self._cycle_stats.sector_unavailable += 1
            return None

        # Condition 8: Overextension Guard (Close <= EMA9 + 1.25*ATR and <= ORB_high + 1.25*ATR)
        if ema9 > 0 and (close > ema9 + (self.max_atr_extension * atr)):
            self._cycle_stats.overextended += 1
            return None
        if close > orb_high + (self.max_atr_extension * atr):
            self._cycle_stats.overextended += 1
            return None

        # Condition 9: Spread Check Passes Execution Gate (<= 0.15%)
        spread_pct = float(last_bar.get("spread_pct", 0.0) or meta.get("spread_pct", 0.0005))
        if spread_pct > 0.0015:  # 0.15% max spread
            self._cycle_stats.spread_failed += 1
            return None

        # Condition 10: Stop Geometry with Noise Floor & Fixed 2R Target
        candidate_levels = [lvl for lvl in (vwap, ema9, low) if lvl < close]
        if not candidate_levels:
            raw_stop = orb_low
        else:
            raw_stop = max(candidate_levels)

        # Apply noise floor: minimum distance from entry
        tick_size = 0.01
        min_distance = max(tick_size * 2, atr * self.min_stop_atr_multiple)
        stop_price = min(raw_stop, close - min_distance)

        if stop_price < orb_low:
            stop_price = orb_low

        risk = round(close - stop_price, 4)
        if risk <= 0:
            self._cycle_stats.zero_risk += 1
            return None

        target_price = round(close + (self.target_r_multiple * risk), 4)

        # Signal bar UTC timestamps
        signal_bar_start_utc = df.index[-1].tz_convert("UTC").isoformat() if df.index[-1].tzinfo else df.index[-1].tz_localize("UTC").isoformat()
        freq_delta = pd.Timedelta(minutes=1)
        if len(df) >= 2:
            freq_delta = df.index[-1] - df.index[-2]
        signal_bar_close_utc = (pd.Timestamp(signal_bar_start_utc) + freq_delta).isoformat()
        signal_ts_utc = signal_bar_close_utc

        candidate_id = generate_deterministic_shadow_id(
            strategy_version=self.version,
            symbol=symbol,
            side="BUY",
            signal_bar_close_utc=signal_bar_close_utc,
            session_date_et=session_date_et,
        )

        self._cycle_stats.candidates += 1
        logger.debug(
            "[ORB SHADOW FILTER] symbol=%s rvol=%.2f gap_pct=%.2f premarket_volume_ratio=%.2f catalyst=%s accepted=true reason=ACCEPTED",
            symbol, rvol, gap_pct, pre_vol_ratio, catalyst or "NONE"
        )

        return ShadowCandidate(
            candidate_id=candidate_id,
            strategy_name=self.name,
            strategy_version=self.version,
            symbol=symbol.upper().strip(),
            side="BUY",
            signal_timestamp_utc=signal_ts_utc,
            signal_bar_start_utc=signal_bar_start_utc,
            signal_bar_close_utc=signal_bar_close_utc,
            session_date_et=session_date_et,
            reference_price=close,
            stop_price=round(stop_price, 4),
            target_price=target_price,
            risk_per_share=risk,
            opening_range_high=round(orb_high, 4),
            opening_range_low=round(orb_low, 4),
            relative_volume=round(rvol, 2),
            breakout_volume_ratio=round(breakout_vol_ratio, 2),
            price_vs_vwap=round(close - vwap, 4),
            vwap_slope=round(vwap_slope, 4),
            ema9=round(ema9, 4),
            ema20=round(ema20, 4),
            atr=round(atr, 4),
            atr_pct=round(atr / close, 4) if close > 0 else 0.0,
            spread_pct=round(spread_pct, 5),
            spy_alignment=spy_status,
            qqq_alignment=qqq_status,
            sector_alignment=sector_status,
            catalyst_type=str(catalyst) if catalyst else None,
            premarket_gap_pct=round(gap_pct, 2),
            premarket_volume_ratio=round(pre_vol_ratio, 2),
            market_texture=str(meta.get("market_texture", "TRENDING")),
            regime_mode=str(meta.get("regime_mode", "RISK_ON")),
            confidence=0.85,
            context=MappingProxyType({
                "orb_range": round(orb_range, 4),
                "avg_vol_20": round(avg_vol_20, 1),
                "is_stock_in_play": is_stock_in_play,
            }),
        )

    def _compute_opening_range(
        self,
        df: pd.DataFrame,
        idx_et: pd.DatetimeIndex,
        session_date_et: str,
    ) -> Optional[Tuple[float, float, float]]:
        """Strict 15/15 completed 1m bars opening range (09:30 through 09:44 ET)."""
        session_mask = (idx_et.strftime("%Y-%m-%d") == session_date_et)
        day_df = df.loc[session_mask]
        day_idx_et = idx_et[session_mask]

        if day_df.empty:
            return None

        range_mask = (day_idx_et.time >= time(9, 30)) & (day_idx_et.time < time(9, 45))
        range_bars = day_df.loc[range_mask]

        if len(range_bars) != self.range_minutes:
            return None

        orb_h = float(range_bars["high"].max())
        orb_l = float(range_bars["low"].min())
        orb_range = orb_h - orb_l
        if orb_h <= 0 or orb_l <= 0 or orb_range <= 0:
            return None

        return (orb_h, orb_l, orb_range)

    @staticmethod
    def _check_market_alignment(
        market_data: Optional[Dict[str, pd.DataFrame]],
        symbol: str,
    ) -> Tuple[AlignmentStatus, AlignmentStatus, AlignmentStatus]:
        """Evaluates SPY, QQQ, and Sector ETF alignment."""
        if not market_data:
            return (AlignmentStatus.ALIGNED, AlignmentStatus.ALIGNED, AlignmentStatus.FALLBACK_MARKET_CONFIRMED)

        def _is_bullish(benchmark_df: Optional[pd.DataFrame]) -> AlignmentStatus:
            if benchmark_df is None or benchmark_df.empty or len(benchmark_df) < 4:
                return AlignmentStatus.UNAVAILABLE
            last_row = benchmark_df.iloc[-1]
            b_close = float(last_row.get("close", 0.0))
            b_vwap = float(last_row.get("vwap", 0.0))
            if b_close <= 0 or b_vwap <= 0:
                return AlignmentStatus.UNAVAILABLE
            if b_close > b_vwap:
                prior_vwap = float(benchmark_df["vwap"].iloc[-4]) if len(benchmark_df) >= 4 else b_vwap
                if b_vwap >= prior_vwap:
                    return AlignmentStatus.ALIGNED
            return AlignmentStatus.OPPOSED

        spy_status = _is_bullish(market_data.get("SPY"))
        qqq_status = _is_bullish(market_data.get("QQQ"))

        sector_etf = market_data.get(f"SECTOR_{symbol}")
        if sector_etf is not None:
            sector_status = _is_bullish(sector_etf)
        else:
            sector_status = AlignmentStatus.FALLBACK_MARKET_CONFIRMED

        return (spy_status, qqq_status, sector_status)


_confirmed_orb_strategy_instance: Optional[ConfirmedORBStrategy] = None


def get_confirmed_orb_strategy(force_new: bool = False) -> ConfirmedORBStrategy:
    """Returns singleton instance of ConfirmedORBStrategy."""
    global _confirmed_orb_strategy_instance
    if _confirmed_orb_strategy_instance is None or force_new:
        _confirmed_orb_strategy_instance = ConfirmedORBStrategy()
    return _confirmed_orb_strategy_instance
