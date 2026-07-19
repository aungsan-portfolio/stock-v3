"""
data_manager.py — OHLCV data fetch + caching + feature engineering.

Priority-1 production fixes:
- Cache key includes (symbol, period, interval), not symbol only.
- Cached DataFrames are copied on read/write to avoid accidental mutation.
- make_labels() preserves unknown future rows as NaN instead of converting them to 0.
"""
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# key = (symbol, period, interval), value = (timestamp, dataframe)
_cache: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}

# ── Item 3: gated regime / relative-strength feature columns ─────────
# These are appended to the model feature set ONLY when the matching config
# flag is on (see get_feature_columns / build_features). Default-off keeps the
# frozen 14-feature list and the existing on-disk models intact.
_REGIME_FEATURE_COLS = [
    "realized_vol",   # short-window realized volatility (std of daily returns)
    "vol_regime",     # short-vol / long-vol ratio (vol expansion vs contraction)
    "mom_short",      # ~3-month time-series momentum (price % change)
    "mom_long",       # ~6-month time-series momentum
    "ts_rank",        # time-series percentile rank of close in a trailing window (0, 1]
    "dist_high",      # distance below the trailing high, <= 0 (drawdown/strength regime)
]
_MARKET_REL_FEATURE_COLS = [
    "rel_ret_short",  # symbol minus benchmark return over the short window
    "rel_ret_long",   # symbol minus benchmark return over the long window
    "rs_slope",       # momentum of the relative-strength (close / benchmark) line
    "mkt_trend",      # benchmark trend regime (benchmark vs its SMA), broadcast to the row
]
# ── Item 6B: Advanced market micro-structure features (LSTM edge recovery) ──
# Computed entirely from OHLCV — no order-book data required. Enabled via
# config.USE_MICRO_FEATURES. All columns are stationary and comparable across
# symbols: returns, log-ratios, normalized spreads.
_MICRO_FEATURE_COLS = [
    "close_loc",          # log close / VWAP proxy: close / (hlc3/typical) — deviation from mean print
    "volume_imbalance",   # directional volume pressure: (up_vol - down_vol) / total_vol
    "intraday_volatility",# high-low / close as a % — daily range efficiency
    "gap_pct",            # overnight gap: (Open - prev Close) / prev Close
    "rolling_efficiency", # 10-bar efficiency ratio: |close - close[-10]| / sum(|ret_i|)
    "atr_slope",          # 5-bar change in ATR — volatility expansion/contraction
    "volume_shock",       # volume ratio z-score: (vol_ratio - 1) / vol_ratio_rolling_std
    "close_rank_20",      # where close ranks in the trailing 20 bars, (0,1]
    "high_low_spread_pct",# HL / close — normalized daily range
]
_CANDLESTICK_FEATURE_COLS = [
    "doji",               # open approx = close (indecision)
    "dragonfly_doji",     # doji with long lower shadow (bullish reversal)
    "gravestone_doji",    # doji with long upper shadow (bearish reversal)
    "white_marubozu",     # strong bullish with no shadows (continuation)
    "black_marubozu",     # strong bearish with no shadows (continuation)
    "hammer",             # long lower shadow, small body in downtrend (reversal)
    "hanging_man",        # long lower shadow, small body in uptrend (reversal)
    "shooting_star",      # long upper shadow, small body in uptrend (reversal)
    "bullish_engulfing",  # white body engulfs prior black body (reversal)
    "bearish_engulfing",  # black body engulfs prior white body (reversal)
    "bullish_harami",     # white body inside prior black body (weakening)
    "bearish_harami",     # black body inside prior white body (weakening)
    "piercing_line",      # white closes above prior black midpoint (reversal)
    "dark_cloud_cover",   # black closes below prior white midpoint (reversal)
    "morning_star",       # 3-bar: black, doji gap down, white gap up (major reversal)
    "evening_star",       # 3-bar: white, doji gap up, black gap down (major reversal)
]


# One-shot warning latch so an unwired market_df does not spam the logs.
_MARKET_DF_WARNED = False


# Singleton MultiSourceDataProvider shared across the module. Callers that pass
# an explicit IBKRBridge to data_manager at init time can set this once at startup;
# otherwise it defaults to yfinance-only, matching the old behavior exactly.
_PROVIDER = None

def set_data_provider(provider=None):
    """Replace the global data provider (e.g. with a MultiSourceDataProvider wired
    to your IBKRBridge). Passing None resets to the default yfinance-only provider.
    Call once at startup (e.g. in main.py) so the module import order does not
    matter."""
    global _PROVIDER
    _PROVIDER = provider

def _get_or_create_provider():
    if _PROVIDER is not None:
        return _PROVIDER
    from data_providers import MultiSourceDataProvider
    return MultiSourceDataProvider()

def fetch_ohlcv(
    symbol: str,
    period: str = None,
    interval: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch OHLCV data via the configured MultiSourceDataProvider (default:
    yfinance) with an in-memory TTL cache overlay for frequently-accessed
    symbols during a single run.

    When a data_provider is set via ``set_data_provider(provider)`` the
    provider's multi-source fallback and disk cache are used; otherwise the
    legacy yfinance-only path applies (unchanged behaviour).
    """
    symbol = symbol.upper().strip()
    period = period or config.PRICE_PERIOD
    interval = interval or config.PRICE_INTERVAL
    cache_key = (symbol, period, interval)

    cached = _cache.get(cache_key)
    if not force_refresh and cached:
        ts, cached_df = cached
        if time.time() - ts < config.CACHE_TTL_SECONDS:
            return cached_df.copy()

    # If a provider was explicitly set, delegate to its multi-source fetch
    # (which includes its own disk cache, fallback, and corporate-action checks).
    if _PROVIDER is not None:
        df = _get_or_create_provider().fetch(
            symbol, period=period, interval=interval, force_refresh=force_refresh
        )
        _cache[cache_key] = (time.time(), df.copy())
        return df.copy()

    # Default: legacy yfinance-only path (byte-identical to original behaviour).
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


def build_features(df: pd.DataFrame, market_df: Optional[pd.DataFrame] = None, cfg=None) -> pd.DataFrame:
    """Engineer the model feature matrix from a single symbol's OHLCV.

    ``market_df`` is an optional benchmark (e.g. SPY) OHLCV frame used only by the
    relative-strength features, and only when ``config.USE_MARKET_RELATIVE_FEATURES``
    is on. It is backward compatible: every existing caller passes ``df`` alone and
    is unaffected. When both Item-3 flags are off (the default) the output is
    byte-identical to the original 14-feature frame.
    """
    cfg = cfg or config
    f = df.copy()
    close, high, low, volume = f["Close"], f["High"], f["Low"], f["Volume"]

    f["sma_short"] = close.rolling(cfg.SMA_SHORT).mean()
    f["sma_long"] = close.rolling(cfg.SMA_LONG).mean()
    f["ema"] = close.ewm(span=cfg.EMA_PERIOD, adjust=False).mean()
    # Close-normalized so the feature is comparable across price levels/symbols
    # instead of carrying raw dollar magnitude.
    f["sma_cross"] = (f["sma_short"] - f["sma_long"]) / close
    f["dist_ema"] = (close - f["ema"]) / close

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(cfg.RSI_PERIOD).mean()
    loss = (-delta.clip(upper=0)).rolling(cfg.RSI_PERIOD).mean()
    rs = gain / loss.replace(0, np.nan)
    f["rsi"] = 100 - (100 / (1 + rs))

    ema_fast = close.ewm(span=cfg.MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=cfg.MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_sig = macd.ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd - macd_sig
    # Close-normalized MACD family so magnitudes are price-level independent.
    f["macd"] = macd / close
    f["macd_sig"] = macd_sig / close
    f["macd_hist"] = macd_hist / close

    bb_mid = close.rolling(cfg.BOLLINGER_PERIOD).mean()
    bb_std = close.rolling(cfg.BOLLINGER_PERIOD).std()
    f["bb_upper"] = bb_mid + cfg.BOLLINGER_STD * bb_std
    f["bb_lower"] = bb_mid - cfg.BOLLINGER_STD * bb_std
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
    f["atr"] = tr.rolling(cfg.ATR_PERIOD).mean()
    f["atr_pct"] = f["atr"] / close

    f["vol_ma"] = volume.rolling(cfg.VOLUME_MA_PERIOD).mean()
    f["vol_ratio"] = volume / f["vol_ma"].replace(0, np.nan)

    f["ret_1d"] = close.pct_change(1)
    f["ret_5d"] = close.pct_change(5)
    f["ret_20d"] = close.pct_change(20)

    f["dist_sma20"] = (close - f["sma_short"]) / close
    f["dist_sma50"] = (close - f["sma_long"]) / close

    # ── Item 3 (default OFF): regime / relative-strength feature set ──
    # Only computed when the matching flag is on, so the default path adds no
    # columns and the final dropna() drops the same warmup rows as before.
    if getattr(cfg, "USE_REGIME_FEATURES", False):
        _add_regime_features(f, close, cfg)
    if getattr(cfg, "USE_MARKET_RELATIVE_FEATURES", False):
        _add_market_relative_features(f, close, market_df, cfg)
    if getattr(cfg, "USE_CANDLESTICK_FEATURES", False):
        _add_candlestick_features(f, close, high, low, f["Open"], cfg)
    if getattr(cfg, "USE_MICRO_FEATURES", False):
        _add_micro_features(f, close, high, low, f["Open"], f["Volume"], cfg)

    f.dropna(inplace=True)
    return f


def _add_regime_features(f: pd.DataFrame, close: pd.Series, cfg=None) -> None:
    """Single-symbol volatility / momentum / trend-regime features (Item 3).

    All values are stationary and price-level independent (returns, ratios,
    normalized distances) so they are comparable across symbols. Mutates ``f``
    in place; computed only when ``config.USE_REGIME_FEATURES`` is on.
    """
    cfg = cfg or config
    ret1 = close.pct_change(1)
    vol_short = ret1.rolling(cfg.REGIME_VOL_SHORT).std()
    vol_long = ret1.rolling(cfg.REGIME_VOL_LONG).std()
    f["realized_vol"] = vol_short
    # Vol regime: >1 expanding vol, <1 contracting. Guard divide-by-zero.
    f["vol_regime"] = vol_short / vol_long.replace(0, np.nan)
    f["mom_short"] = close.pct_change(cfg.REGIME_MOM_SHORT)
    f["mom_long"] = close.pct_change(cfg.REGIME_MOM_LONG)
    # Time-series percentile rank of the latest close within a trailing window,
    # in (0, 1]. A per-symbol, offline "where does today sit vs its own recent
    # history" strength proxy — NOT a cross-sectional rank across symbols.
    win = int(cfg.REGIME_RANK_WINDOW)
    f["ts_rank"] = close.rolling(win).apply(
        lambda a: float((a <= a[-1]).mean()), raw=True
    )
    # Distance below the trailing high (<= 0): 0 at a fresh high, negative in a
    # drawdown. Captures the strength/regime of the symbol's own trend.
    roll_high = close.rolling(cfg.REGIME_HIGH_WINDOW).max()
    f["dist_high"] = close / roll_high.replace(0, np.nan) - 1.0


def _add_market_relative_features(
    f: pd.DataFrame, close: pd.Series, market_df: Optional[pd.DataFrame], cfg=None
) -> None:
    """SPY/benchmark relative-strength features (Item 3).

    Requires a benchmark close series aligned to the symbol's dates. When
    ``market_df`` is None the columns are emitted as neutral 0.0 (with a one-shot
    warning) so no FEATURE_COLS consumer crashes before the engines are wired to
    pass ``market_df``. Mutates ``f`` in place; computed only when
    ``config.USE_MARKET_RELATIVE_FEATURES`` is on.
    """
    cfg = cfg or config
    if market_df is None or "Close" not in getattr(market_df, "columns", []):
        global _MARKET_DF_WARNED
        if not _MARKET_DF_WARNED:
            logger.warning(
                "USE_MARKET_RELATIVE_FEATURES is on but no market_df was passed to "
                "build_features() — emitting NEUTRAL (0.0) relative-strength features. "
                "Wire market_df (data_manager.fetch_market_benchmark) through the "
                "engines before trusting these features in training."
            )
            _MARKET_DF_WARNED = True
        for col in _MARKET_REL_FEATURE_COLS:
            f[col] = 0.0
        return

    # Align the benchmark to the symbol's trading dates; forward-fill any gaps.
    mkt_close = market_df["Close"].reindex(f.index).ffill()
    s_short = int(cfg.REL_RET_SHORT)
    s_long = int(cfg.REL_RET_LONG)
    f["rel_ret_short"] = close.pct_change(s_short) - mkt_close.pct_change(s_short)
    f["rel_ret_long"] = close.pct_change(s_long) - mkt_close.pct_change(s_long)
    # Relative-strength line = symbol / benchmark; its slope is RS momentum.
    rs_line = close / mkt_close.replace(0, np.nan)
    f["rs_slope"] = rs_line.pct_change(cfg.RS_SLOPE_WINDOW)
    # Broadcast market-regime feature: benchmark vs its own trend SMA.
    mkt_sma = mkt_close.rolling(cfg.MKT_TREND_SMA).mean()
    f["mkt_trend"] = mkt_close / mkt_sma.replace(0, np.nan) - 1.0


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


def fetch_market_benchmark(
    period: str = None, interval: str = None, force_refresh: bool = False
) -> pd.DataFrame:
    """Fetch the benchmark (config.MARKET_BENCHMARK_SYMBOL, e.g. SPY) OHLCV for the
    relative-strength features. Thin wrapper over fetch_ohlcv so engines can pass
    the result as ``build_features(df, market_df=...)``.

    This performs a network fetch and is intended to be called by the engines
    ONLY when ``config.USE_MARKET_RELATIVE_FEATURES`` is on. build_features()
    itself never calls it — feature engineering stays pure/offline/testable.
    """
    return fetch_ohlcv(
        config.MARKET_BENCHMARK_SYMBOL,
        period=period,
        interval=interval,
        force_refresh=force_refresh,
    )



def _add_candlestick_features(f: pd.DataFrame, close: pd.Series, high: pd.Series,
                               low: pd.Series, open_: pd.Series, cfg=None) -> None:
    """Candlestick pattern boolean flags from Steve Nison's Japanese Candlestick
    Charting Techniques. Each pattern returns {0, 1}. Mutates f in place;
    computed only when config.USE_CANDLESTICK_FEATURES is on.
    """
    cfg = cfg or config
    body = (close - open_).abs()
    total_range = high - low
    avg_body = body.rolling(20).mean().replace(0, np.nan)
    body_pct = body / total_range.replace(0, np.nan)
    upper_shadow = high - close.where(close >= open_, open_)
    lower_shadow = open_.where(close >= open_, close) - low

    doji_th = float(getattr(cfg, "CANDLESTICK_DOJI_THRESHOLD", 0.10))
    long_shadow_mult = float(getattr(cfg, "CANDLESTICK_LONGSHADOW_MULTIPLIER", 2.0))
    long_body_th = float(getattr(cfg, "CANDLESTICK_LONGBODY_THRESHOLD", 0.60))
    short_body_mult = float(getattr(cfg, "CANDLESTICK_SHORT_BODY_MULTIPLIER", 0.5))

    # -- Single-candle patterns --
    is_doji = body_pct < doji_th
    dragonfly = is_doji & (lower_shadow > long_shadow_mult * body) & (upper_shadow < body)
    gravestone = is_doji & (upper_shadow > long_shadow_mult * body) & (lower_shadow < body)
    f["doji"] = is_doji.astype(float)
    f["dragonfly_doji"] = dragonfly.astype(float)
    f["gravestone_doji"] = gravestone.astype(float)

    white = close >= open_
    black = close < open_
    strong_body = body / total_range.replace(0, np.nan) > long_body_th
    no_upper = upper_shadow < body * 0.1
    no_lower = lower_shadow < body * 0.1
    f["white_marubozu"] = (white & strong_body & no_upper & no_lower).astype(float)
    f["black_marubozu"] = (black & strong_body & no_upper & no_lower).astype(float)

    small_body = body < avg_body * short_body_mult
    long_lower = lower_shadow > long_shadow_mult * body
    long_upper = upper_shadow > long_shadow_mult * body

    # Hammer/Hanging Man: differentiated by short-term trend context
    ret_5d_col = close.pct_change(5)
    downtrend = ret_5d_col < 0
    uptrend = ret_5d_col >= 0
    hammer_shape = white & small_body & long_lower & (upper_shadow < body)
    f["hammer"] = (hammer_shape & downtrend).astype(float)
    f["hanging_man"] = (hammer_shape & uptrend).astype(float)

    # Shooting Star
    shooting_shape = black & small_body & long_upper & (lower_shadow < body)
    f["shooting_star"] = (shooting_shape & uptrend).astype(float)

    # -- Two-candle patterns --
    prev_open = open_.shift(1)
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_white = prev_close >= prev_open
    prev_black = prev_close < prev_open

    f["bullish_engulfing"] = (
        prev_black & white &
        (open_ < prev_close) & (close > prev_open)
    ).astype(float)
    f["bearish_engulfing"] = (
        prev_white & black &
        (open_ > prev_close) & (close < prev_open)
    ).astype(float)
    f["bullish_harami"] = (
        prev_black & white &
        (open_ > prev_close) & (close < prev_open)
    ).astype(float)
    f["bearish_harami"] = (
        prev_white & black &
        (open_ < prev_close) & (close > prev_open)
    ).astype(float)

    prev_mid = (prev_open + prev_close) / 2.0
    f["piercing_line"] = (
        prev_black & white &
        (open_ < prev_low) & (close > prev_mid)
    ).astype(float)
    f["dark_cloud_cover"] = (
        prev_white & black &
        (open_ > prev_high) & (close < prev_mid)
    ).astype(float)

    # -- Three-candle patterns --
    prev2_open = open_.shift(2)
    prev2_close = close.shift(2)
    prev2_high = high.shift(2)
    prev2_low = low.shift(2)
    prev2_white = prev2_close >= prev2_open
    prev2_black = prev2_close < prev2_open
    prev2_mid = (prev2_open + prev2_close) / 2.0

    prev1_body = (prev_close - prev_open).abs()
    prev1_small = prev1_body < avg_body.shift(1) * short_body_mult

    f["morning_star"] = (
        prev2_black & prev1_small &
        (prev_high < prev2_low) &  # gap down to star
        white & (low > prev_high) &  # gap up from star
        (close > prev2_mid)
    ).astype(float)
    f["evening_star"] = (
        prev2_white & prev1_small &
        (prev_low > prev2_high) &  # gap up to star
        black & (high < prev_low) &  # gap down from star
        (close < prev2_mid)
    ).astype(float)
def _add_micro_features(f: pd.DataFrame, close: pd.Series, high: pd.Series, low: pd.Series,
                         open_: pd.Series, volume: pd.Series, cfg=None) -> None:
    """Micro-structure / market quality features computed from daily OHLCV.
    No order-book data required. All values stationary & price-level independent.
    Gated by config.USE_MICRO_FEATURES (default OFF).

    Features added:
      * close_loc — log(close / hlc3): mean-reversion proxy
      * volume_imbalance — (up_day_vol - down_day_vol) / total_rolling_vol
      * intraday_volatility — log range / close: daily efficiency
      * gap_pct — overnight gap
      * rolling_efficiency — 10-day efficiency ratio
      * atr_slope — ATR rate of change
      * volume_shock — z-score of volume ratio
      * close_rank_20 — percentile rank of close in trailing 20 bars
      * high_low_spread_pct — HL normalized by close
    """
    cfg = cfg or config
    hlc3 = (high + low + close) / 3.0
    f["close_loc"] = np.log(close / hlc3.replace(0, np.nan))

    # Volume imbalance: directional volume pressure
    up_day = close >= open_
    down_day = close < open_
    up_vol = volume.where(up_day, 0.0)
    down_vol = volume.where(down_day, 0.0)
    vol_total = volume.rolling(20).sum().replace(0, np.nan)
    f["volume_imbalance"] = (up_vol.rolling(20).sum() - down_vol.rolling(20).sum()) / vol_total

    f["intraday_volatility"] = (high - low) / close

    # Overnight gap
    f["gap_pct"] = open_ / close.shift(1) - 1.0

    # Rolling efficiency ratio: net move / total path
    ret = close.pct_change(1).abs()
    eff_num = (close - close.shift(10)).abs()
    eff_den = ret.rolling(10).sum().replace(0, np.nan)
    f["rolling_efficiency"] = eff_num / eff_den

    # ATR slope (5-bar change in ATR%)
    atr = (high - low + (high - close.shift(1)).abs() + (low - close.shift(1)).abs()).rolling(14).mean()
    atr_pct = atr / close
    f["atr_slope"] = atr_pct.pct_change(5)

    # Volume shock z-score: how unusual is today's volume ratio
    vr = volume / volume.rolling(20).mean().replace(0, np.nan)
    vr_std = vr.rolling(40).std().replace(0, np.nan)
    f["volume_shock"] = (vr - vr.mean()) / vr_std

    # Close rank in trailing 20 bars
    f["close_rank_20"] = close.rolling(20).apply(
        lambda a: float((a <= a[-1]).mean()), raw=True
    )

    # Normalized daily range
    f["high_low_spread_pct"] = (high - low) / close


def get_feature_columns(cfg=None) -> list:
    """Model feature columns. The frozen base-14 set is returned unless an Item-3
    or Item-6B flag is on, in which case the matching gated columns are APPENDED
    (order preserved) — which changes the feature-vector width and requires retraining.
    """
    cfg = cfg or config
    cols = [
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
    if getattr(cfg, "USE_REGIME_FEATURES", False):
        cols = cols + _REGIME_FEATURE_COLS
    if getattr(cfg, "USE_MARKET_RELATIVE_FEATURES", False):
        cols = cols + _MARKET_REL_FEATURE_COLS
    if getattr(cfg, "USE_CANDLESTICK_FEATURES", False):
        cols = cols + _CANDLESTICK_FEATURE_COLS
    if getattr(cfg, "USE_MICRO_FEATURES", False):
        cols = cols + _MICRO_FEATURE_COLS
    return cols
