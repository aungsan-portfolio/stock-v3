"""
analyze_market_data.py — Export OHLCV + feature analysis to CSV.

This is an offline research/helper script. It does not connect to IBKR and does
not place orders. It fetches market data through data_manager.fetch_ohlcv(),
adds technical features, and writes CSV files under reports/market_analysis/.

Examples:
    python analyze_market_data.py
    python analyze_market_data.py --symbols NVDA QQQ AAPL --period 60d --interval 15m
    python analyze_market_data.py --symbols VTI --period 5y --interval 1d

Notes:
    - For intraday "what time does it tend to rise/fall?" analysis, use 5m/15m/30m.
    - For longer trend review, use 1d.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config
from data_manager import build_features, fetch_ohlcv
from logging_setup import setup_logging
from predictor import technical_score_from_feature_row

setup_logging()
logger = logging.getLogger(__name__)


def _pct(series: pd.Series) -> pd.Series:
    return series.astype(float) * 100.0


def _safe_win_rate(series: pd.Series) -> float:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return float((clean > 0).mean())


def _add_analysis_columns(features: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = features.copy()
    out.insert(0, "symbol", symbol.upper().strip())

    out["timestamp"] = pd.to_datetime(out.index)
    out["date"] = out["timestamp"].dt.date.astype(str)
    out["weekday"] = out["timestamp"].dt.day_name()
    out["hour"] = out["timestamp"].dt.hour
    out["minute"] = out["timestamp"].dt.minute
    out["time_bucket"] = out["timestamp"].dt.strftime("%H:%M")

    out["bar_return_pct"] = _pct(out["Close"].pct_change())
    out["candle_return_pct"] = _pct(out["Close"] / out["Open"] - 1.0)
    out["range_pct"] = _pct((out["High"] - out["Low"]) / out["Close"])
    out["gap_pct"] = _pct(out["Open"] / out["Close"].shift(1) - 1.0)

    # Forward returns are for research only. Do not use these columns as live
    # trading inputs because they contain future information.
    out["next_1bar_return_pct"] = _pct(out["Close"].shift(-1) / out["Close"] - 1.0)
    out["next_3bar_return_pct"] = _pct(out["Close"].shift(-3) / out["Close"] - 1.0)
    out["next_6bar_return_pct"] = _pct(out["Close"].shift(-6) / out["Close"] - 1.0)
    out["next_12bar_return_pct"] = _pct(out["Close"].shift(-12) / out["Close"] - 1.0)

    out["bar_direction"] = np.select(
        [out["bar_return_pct"] > 0, out["bar_return_pct"] < 0],
        ["UP", "DOWN"],
        default="FLAT",
    )
    out["next_1bar_direction"] = np.select(
        [out["next_1bar_return_pct"] > 0, out["next_1bar_return_pct"] < 0],
        ["UP", "DOWN"],
        default="FLAT",
    )

    out["technical_score"] = out.apply(technical_score_from_feature_row, axis=1)
    out["momentum_tag"] = np.select(
        [out["technical_score"] >= 0.65, out["technical_score"] <= 0.35],
        ["BULLISH", "BEARISH"],
        default="NEUTRAL",
    )

    # Make the CSV easier to read by placing common columns first.
    first_cols = [
        "symbol", "timestamp", "date", "weekday", "hour", "minute", "time_bucket",
        "Open", "High", "Low", "Close", "Volume",
        "bar_return_pct", "candle_return_pct", "range_pct", "gap_pct",
        "next_1bar_return_pct", "next_3bar_return_pct", "next_6bar_return_pct", "next_12bar_return_pct",
        "bar_direction", "next_1bar_direction", "technical_score", "momentum_tag",
    ]
    remaining_cols = [c for c in out.columns if c not in first_cols]
    return out[first_cols + remaining_cols]


def _summarize_by_time(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["symbol", "time_bucket"], dropna=False)
    summary = grouped.agg(
        bars=("Close", "size"),
        avg_bar_return_pct=("bar_return_pct", "mean"),
        avg_next_1bar_return_pct=("next_1bar_return_pct", "mean"),
        median_next_1bar_return_pct=("next_1bar_return_pct", "median"),
        avg_next_3bar_return_pct=("next_3bar_return_pct", "mean"),
        avg_range_pct=("range_pct", "mean"),
        avg_volume_ratio=("vol_ratio", "mean"),
        avg_technical_score=("technical_score", "mean"),
    ).reset_index()

    win_rates = grouped["next_1bar_return_pct"].apply(_safe_win_rate).reset_index(name="next_1bar_win_rate")
    summary = summary.merge(win_rates, on=["symbol", "time_bucket"], how="left")
    summary.sort_values(["symbol", "avg_next_1bar_return_pct"], ascending=[True, False], inplace=True)
    return summary


def _summarize_by_weekday(rows: pd.DataFrame) -> pd.DataFrame:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    grouped = rows.groupby(["symbol", "weekday"], dropna=False)
    summary = grouped.agg(
        bars=("Close", "size"),
        avg_bar_return_pct=("bar_return_pct", "mean"),
        avg_next_1bar_return_pct=("next_1bar_return_pct", "mean"),
        median_next_1bar_return_pct=("next_1bar_return_pct", "median"),
        avg_next_3bar_return_pct=("next_3bar_return_pct", "mean"),
        avg_range_pct=("range_pct", "mean"),
        avg_volume_ratio=("vol_ratio", "mean"),
        avg_technical_score=("technical_score", "mean"),
    ).reset_index()
    win_rates = grouped["next_1bar_return_pct"].apply(_safe_win_rate).reset_index(name="next_1bar_win_rate")
    summary = summary.merge(win_rates, on=["symbol", "weekday"], how="left")
    summary["weekday"] = pd.Categorical(summary["weekday"], categories=weekday_order, ordered=True)
    summary.sort_values(["symbol", "weekday"], inplace=True)
    summary["weekday"] = summary["weekday"].astype(str)
    return summary


def export_market_analysis(
    symbols: Iterable[str],
    period: str,
    interval: str,
    out_dir: Path,
    force_refresh: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    errors: list[dict] = []

    for raw_symbol in symbols:
        symbol = raw_symbol.upper().strip()
        if not symbol:
            continue
        try:
            logger.info("Analyzing %s period=%s interval=%s", symbol, period, interval)
            df = fetch_ohlcv(symbol, period=period, interval=interval, force_refresh=force_refresh)
            features = build_features(df)
            if features.empty:
                raise ValueError(f"No feature rows after indicator warmup for {symbol}")
            rows = _add_analysis_columns(features, symbol)
            frames.append(rows)

            per_symbol_path = out_dir / f"{symbol}_{period}_{interval}_analysis.csv"
            rows.to_csv(per_symbol_path, index=False)
            logger.info("Wrote %s", per_symbol_path)
        except Exception as exc:  # keep analyzing remaining symbols
            logger.exception("Failed to analyze %s", symbol)
            errors.append({"symbol": symbol, "error": str(exc)})

    if not frames:
        if errors:
            pd.DataFrame(errors).to_csv(out_dir / "market_analysis_errors.csv", index=False)
        raise RuntimeError("No symbols were analyzed successfully")

    all_rows = pd.concat(frames, ignore_index=True)
    rows_path = out_dir / "market_analysis_rows.csv"
    time_summary_path = out_dir / "market_analysis_by_time.csv"
    weekday_summary_path = out_dir / "market_analysis_by_weekday.csv"

    all_rows.to_csv(rows_path, index=False)
    _summarize_by_time(all_rows).to_csv(time_summary_path, index=False)
    _summarize_by_weekday(all_rows).to_csv(weekday_summary_path, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(out_dir / "market_analysis_errors.csv", index=False)

    return {
        "rows": str(rows_path),
        "by_time": str(time_summary_path),
        "by_weekday": str(weekday_summary_path),
        "symbols_ok": sorted(all_rows["symbol"].unique().tolist()),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export market data analysis CSV files")
    parser.add_argument("--symbols", nargs="*", default=config.WATCHLIST, help="Symbols to analyze")
    parser.add_argument("--period", default="60d", help="yfinance period, e.g. 60d, 1y, 5y")
    parser.add_argument("--interval", default="15m", help="yfinance interval, e.g. 5m, 15m, 1h, 1d")
    parser.add_argument(
        "--out-dir",
        default=str(config.REPORTS_DIR / "market_analysis"),
        help="Directory where CSV files will be written",
    )
    parser.add_argument("--use-cache", action="store_true", help="Use in-memory cache when available")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = export_market_analysis(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
        out_dir=Path(args.out_dir),
        force_refresh=not args.use_cache,
    )

    print("\n✅ Market analysis CSV export complete")
    print(f"Rows CSV       : {result['rows']}")
    print(f"By time CSV    : {result['by_time']}")
    print(f"By weekday CSV : {result['by_weekday']}")
    print(f"Symbols OK     : {', '.join(result['symbols_ok'])}")
    if result["errors"]:
        print("\n⚠️ Some symbols failed. See market_analysis_errors.csv")
        for item in result["errors"]:
            print(f"  {item['symbol']}: {item['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
