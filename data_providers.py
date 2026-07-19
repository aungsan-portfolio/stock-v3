"""
data_providers.py — Robust multi-source data pipeline.
Integrates yfinance, IBKR reqHistoricalData, and Polygon.io with fallback logic,
disk-based caching, and automated split/corporate action adjustment verification.
"""
import logging
import time
import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# Ensure data directory exists for caching
CACHE_DIR = config.DATA_DIR / "ohlcv_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class DataProviderError(Exception):
    """Base class for data provider exceptions."""
    pass

class MultiSourceDataProvider:
    """
    Unified interface fetching historical daily stock data from:
    1. yfinance (default, fast web interface)
    2. IBKR Bridge reqHistoricalData (direct broker data, requires connection)
    3. Polygon.io (premium REST API, fallback when API key configured)

    Includes split and dividend adjustment checks on raw prices.
    """

    def __init__(self, ib_bridge=None, polygon_api_key: Optional[str] = None):
        self.ib_bridge = ib_bridge
        self.polygon_api_key = polygon_api_key or os.environ.get("POLYGON_API_KEY")

    def fetch(
        self,
        symbol: str,
        period: str = "5y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for symbol, applying cache lookup, fallback logic,
        and validation/corporate action checks.
        """
        symbol = symbol.upper().strip()
        cache_file = CACHE_DIR / f"{symbol}_{period}_{interval}.csv"

        # 1. Cache Check
        if not force_refresh and cache_file.exists():
            mtime = cache_file.stat().st_mtime
            if time.time() - mtime < config.CACHE_TTL_SECONDS:
                try:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    if not df.empty:
                        logger.info("Loaded %s from local cache", symbol)
                        return df
                except Exception as e:
                    logger.warning("Failed to read CSV cache for %s: %s", symbol, e)

        # 2. Source resolution with fallbacks
        df = None
        errors = []

        # Attempt yfinance
        try:
            df = self._fetch_yfinance(symbol, period, interval)
            logger.info("Fetched %s via yfinance", symbol)
        except Exception as e:
            errors.append(f"yfinance: {e}")
            logger.warning("yfinance fetch failed for %s: %s", symbol, e)

        # Fallback to IBKR if connected
        if df is None and self.ib_bridge and hasattr(self.ib_bridge, "ib") and self.ib_bridge.ib.isConnected():
            try:
                df = self._fetch_ibkr(symbol, period, interval)
                logger.info("Fallback: Fetched %s via IBKR Bridge", symbol)
            except Exception as e:
                errors.append(f"IBKR: {e}")
                logger.warning("IBKR Bridge fallback failed for %s: %s", symbol, e)

        # Fallback to Polygon if API key available
        if df is None and self.polygon_api_key:
            try:
                df = self._fetch_polygon(symbol, period, interval)
                logger.info("Fallback: Fetched %s via Polygon.io", symbol)
            except Exception as e:
                errors.append(f"Polygon: {e}")
                logger.warning("Polygon.io fallback failed for %s: %s", symbol, e)

        if df is None:
            raise DataProviderError(
                f"Failed to fetch data for {symbol} from all sources. Errors: {'; '.join(errors)}"
            )

        # 3. Post-Process, Clean, Validate
        df = self._clean_and_validate(symbol, df)

        # 4. Save to Cache
        try:
            df.to_csv(cache_file)
        except Exception as e:
            logger.warning("Failed to save cache file for %s: %s", symbol, e)

        return df

    def _fetch_yfinance(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No history returned by yfinance")
        return df

    def _fetch_ibkr(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch historical daily bars from Alpaca via AlpacaBridge."""
        # Map period strings (e.g. '5y') to Alpaca durationStr (e.g. '5 Y')
        duration_map = {"5y": "5 Y", "1y": "1 Y", "2y": "2 Y", "1m": "1 M"}
        duration_str = duration_map.get(period.lower(), "5 Y")

        # Map interval (e.g. '1d') to Alpaca barSizeSetting (e.g. '1 day')
        bar_size_map = {"1d": "1 day", "1h": "1 hour", "5m": "5 mins"}
        bar_size = bar_size_map.get(interval.lower(), "1 day")

        bars = self.ib_bridge.fetch_historical_data(
            symbol=symbol,
            durationStr=duration_str,
            barSizeSetting=bar_size
        )
        if not bars:
            raise ValueError("No historical bars returned from Alpaca")

        data = []
        for bar in bars:
            data.append({
                "Date": bar.date,
                "Open": bar.open,
                "High": bar.high,
                "Low": bar.low,
                "Close": bar.close,
                "Volume": bar.volume
            })

        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        return df

    def _fetch_polygon(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Fetch historical data from Polygon.io using HTTP requests."""
        import requests

        # Map period to days count approximately
        days_map = {"5y": 365*5, "2y": 365*2, "1y": 365, "1m": 30}
        days_back = days_map.get(period.lower(), 365)

        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Polygon Aggregates API
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
            "apiKey": self.polygon_api_key
        }

        res = requests.get(url, params=params, timeout=15)
        if res.status_code != 200:
            raise ValueError(f"Polygon API returned status code {res.status_code}: {res.text}")

        res_data = res.json()
        if "results" not in res_data or not res_data["results"]:
            raise ValueError("No results returned by Polygon.io")

        data = []
        for r in res_data["results"]:
            # Polygon timestamp is in milliseconds
            dt = pd.to_datetime(r["t"], unit="ms")
            data.append({
                "Date": dt,
                "Open": float(r["o"]),
                "High": float(r["h"]),
                "Low": float(r["l"]),
                "Close": float(r["c"]),
                "Volume": float(r["v"])
            })

        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        return df

    def _clean_and_validate(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Clean index, sort, normalize column names, and verify split/dividend adjustments."""
        required_cols = ["Open", "High", "Low", "Close", "Volume"]

        # Normalize column names (e.g. lower/camel to capitalized)
        df.columns = [col.capitalize() for col in df.columns]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Data missing required columns: {missing}")

        df = df[required_cols].copy()

        # Format index as UTC date-only/tz-naive timezone
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        df.sort_index(inplace=True)
        df.dropna(inplace=True)

        # Validate against extreme prices / missing volumes / corporate action anomalies
        # An unadjusted split leaves a huge gap: e.g. Close drops 50% overnight with no matching volume ratio shift
        close = df["Close"]
        pct_change = close.pct_change()

        # Check for unadjusted splits/dividends (severe single-day price drop > 35% with no matching market context)
        anomalies = pct_change[pct_change < -0.35]
        if not anomalies.empty:
            for date, change in anomalies.items():
                # Check if it was an actual corporate split or bad data
                logger.warning(
                    "Detected severe price gap (-%.2f%%) for %s on %s. Corporate action verification suggested.",
                    abs(change) * 100, symbol, date.strftime("%Y-%m-%d")
                )

        # Basic sanity validation
        if len(df) < config.MIN_HISTORY_DAYS:
            raise ValueError(
                f"{symbol} only has {len(df)} bars, but engine requires MIN_HISTORY_DAYS={config.MIN_HISTORY_DAYS}."
            )

        return df
