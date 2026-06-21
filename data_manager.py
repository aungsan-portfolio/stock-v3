"""
data_manager.py — OHLCV data fetch + caching + feature engineering.

Priority-1 production fixes:
- Cache key includes (symbol, period, interval), not symbol only.
- Cached DataFrames are copied on read/write to avoid accidental mutation.
- make_labels() preserves unknown future rows as NaN instead of converting them to 0.
"""
import logging
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# key = (symbol, period, interval), value = (timestamp, dataframe)
_cache: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}


def fetch_ohlcv(
    symbol: str,
    period: str = None,
    interval: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance with a small in-memory TTL cache."""
    symbol = symbol.upper().strip()
    period = period or config.PRICE_PERIOD
    interval = interval or config.PRICE_INTERVAL
    cache_key = (symbol, period, interval)

    cached = _cache.get(cache_key)
    if not force_refresh and cached:
        ts, cached_df = cached
        if time.time() - ts < config.CACHE_TTL_SECONDS:
            return cached_df.copy()

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol} data missing required columns: {missing}")

    df = df[required_cols].copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)

    if len(df) < config.MIN_HISTORY_DAYS:
        raise ValueError(
            f"{symbol} only {len(df)} days — need {config.MIN_HISTORY_DAYS}"
        )

    _cache[cache_key] = (time.time(), df.copy())
    logger.info("Fetched %d rows for %s", len(df), symbol)
    return df.copy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    close, high, low, volume = f["Close"], f["High"], f["Low"], f["Volume"]

    f["sma_short"] = close.rolling(config.SMA_SHORT).mean()
    f["sma_long"] = close.rolling(config.SMA_LONG).mean()
    f["ema"] = close.ewm(span=config.EMA_PERIOD, adjust=False).mean()
    # Close-normalized so the feature is comparable across price levels/symbols
    # instead of carrying raw dollar magnitude.
    f["sma_cross"] = (f["sma_short"] - f["sma_long"]) / close
    f["dist_ema"] = (close - f["ema"]) / close

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(config.RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(config.RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = close.ewm(span=config.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=config.MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=config.MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd - macd_sig
    # Close-normalized MACD family so magnitudes are price-level independent.
    f["macd"] = macd / close
    f["macd_sig"] = macd_sig / close
    f["macd_hist"] = macd_hist / close

    bb_mid = close.rolling(config.BOLLINGER_PERIOD).mean()
    bb_std = close.rolling(config.BOLLINGER_PERIOD).std()
    f["bb_upper"] = bb_mid + config.BOLLINGER_STD * bb_std
    f["bb_lower"] = bb_mid - config.BOLLINGER_STD * bb_std
    f["bb_pct"] = (close - f["bb_lower"]) / (
        f["bb_upper"] - f["bb_lower"]
    ).replace(0, np.nan)

    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    f["atr"] = tr.rolling(config.ATR_PERIOD).mean()
    f["atr_pct"] = f["atr"] / close

    f["vol_ma"] = volume.rolling(config.VOLUME_MA_PERIOD).mean()
    f["vol_ratio"] = volume / f["vol_ma"].replace(0, np.nan)

    f["ret_1d"] = close.pct_change(1)
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)

    f["dist_sma20"] = (close - f["sma_short"]) / close
    f["dist_sma50"] = (close - f["sma_long"]) / close

    f.dropna(inplace=True)
    return f


def make_labels(
    df: pd.DataFrame,
    horizon: int = None,
    min_profit_margin: float = None,
    mode: str = None,
    tp_pct: float = None,
    stop_pct: float = None,
) -> pd.Series:
    """
    Create binary direction labels for the next ``horizon`` bars.

    Two modes (selected by ``mode`` or, when None, ``config.LABEL_MODE``):

    * ``"binary"`` (default, unchanged): label 1.0 when the endpoint forward
      return clears ``min_profit_margin`` (default ``config.MIN_PROFIT_MARGIN``).
      Path-blind — it only looks at Close[t+horizon].
        1.0 if Close[t+horizon] / Close[t] - 1 > min_profit_margin
        0.0 otherwise (move too small, flat, or negative)
        NaN for rows where the future close is unknown

    * ``"triple_barrier"``: path-aware (see ``_triple_barrier_labels``). 1.0 if a
      +tp_pct take-profit barrier is touched BEFORE a -stop_pct stop barrier
      within the horizon, else 0.0; NaN when the window is incomplete.

    Both modes return strictly {0.0, 1.0, NaN}, so every downstream class check,
    ``pos_weight``, ``class_weight="balanced"`` and ``predict_proba`` path is
    unaffected by the choice.

    Important: do NOT cast `(future_return > margin)` directly to int over the
    whole series, because NaN > margin becomes False and silently creates fake
    0 labels at the end.
    """
    horizon = horizon or config.ML_HORIZON
    if horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if "Close" not in df.columns:
        raise ValueError("make_labels() requires a 'Close' column")

    mode = (mode or getattr(config, "LABEL_MODE", "binary")).lower()

    if mode == "binary":
        if min_profit_margin is None:
            min_profit_margin = config.MIN_PROFIT_MARGIN
        future_return = df["Close"].shift(-horizon) / df["Close"] - 1.0
        labels = pd.Series(index=df.index, dtype="float64")
        valid = future_return.notna()
        labels.loc[valid] = (future_return.loc[valid] > min_profit_margin).astype(float)
        return labels

    if mode == "triple_barrier":
        return _triple_barrier_labels(df, horizon, tp_pct=tp_pct, stop_pct=stop_pct)

    raise ValueError(
        f"unknown LABEL_MODE: {mode!r} (expected 'binary' or 'triple_barrier')"
    )


def _triple_barrier_labels(
    df: pd.DataFrame, horizon: int, tp_pct: float = None, stop_pct: float = None
) -> pd.Series:
    """Path-aware binary labels using intrabar High/Low.

    For each row t, scan ahead bars t+1 … t+horizon with entry = Close[t]:
        1.0  if High >= entry*(1+tp_pct) on a bar STRICTLY BEFORE any bar where
             Low <= entry*(1-stop_pct)  (take-profit reached first)
        0.0  otherwise — stop reached first, both barriers on the SAME bar
             (pessimistic tie-break = stop), or neither touched within horizon
        NaN  when fewer than ``horizon`` future bars exist (incomplete window —
             same valid-row set as the binary label, no fake 0s at the tail)

    Requires High and Low columns (present on build_features() output, which
    preserves OHLCV). Defaults: tp_pct=config.LABEL_TP_PCT, stop_pct=
    config.LABEL_STOP_PCT.

    Caveat: this simulates a FIXED TP/SL bracket. It is a faithful proxy only for
    the fixed-bracket exit; with config.USE_TRAILING_EXIT=True (default) the live
    exit is a trailing stop with no fixed TP, so this label approximates — does
    not exactly reproduce — live P&L.
    """
    for col in ("Close", "High", "Low"):
        if col not in df.columns:
            raise ValueError(
                f"triple_barrier labels require a {col!r} column "
                "(call make_labels on build_features() output, which keeps OHLCV)"
            )
    if tp_pct is None:
        tp_pct = float(getattr(config, "LABEL_TP_PCT", config.TAKE_PROFIT_PCT))
    if stop_pct is None:
        stop_pct = float(getattr(config, "LABEL_STOP_PCT", config.STOP_LOSS_PCT))

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    n = len(close)
    labels = np.full(n, np.nan, dtype="float64")

    for t in range(n - horizon):
        entry = close[t]
        if entry <= 0:
            continue
        upper = entry * (1.0 + tp_pct)
        lower = entry * (1.0 - stop_pct)
        outcome = 0.0  # neither barrier touched within horizon -> 0.0
        for k in range(t + 1, t + horizon + 1):
            hit_stop = low[k] <= lower
            hit_tp = high[k] >= upper
            if hit_tp and not hit_stop:
                outcome = 1.0
                break
            if hit_stop:  # stop alone, or same-bar tie -> pessimistic stop
                outcome = 0.0
                break
        labels[t] = outcome

    return pd.Series(labels, index=df.index, dtype="float64")


def get_feature_columns() -> list:
    return [
        "sma_cross",
        "dist_ema",
        "rsi",
        "macd",
        "macd_sig",
        "macd_hist",
        "bb_pct",
        "atr_pct",
        "vol_ratio",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "dist_sma20",
        "dist_sma50",
    ]
