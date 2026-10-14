"""
predictor.py — Ensemble: RF + LSTM + Technical → BUY / HOLD / SELL.

Production hardening:
- If RF and LSTM are both unavailable, signals are forced to HOLD. Technical-only
  trading is intentionally disabled by default because it can place orders without
  any trained ML model.
- Score blending renormalizes only the available ML weights plus the technical
  weight when enough ML models are present.
- Technical scoring is exposed as a row-level helper so backtest.py can use the
  same heuristic instead of duplicating divergent logic.
"""
import logging
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import model_metrics
from data_manager import fetch_ohlcv, build_features
from ai_engine import StockRFEngine
from lstm_engine import StockLSTMEngine
from alternative_models import StockXGBEngine, StockTransformerEngine
from errors import ModelNotAvailableError

logger = logging.getLogger(__name__)

# ─── Confidence Threshold Configuration ───────────────────────────────────────
MIN_SWING_CONFIDENCE = 0.45   # Below this → force HOLD regardless of signal
MIN_SELL_CONFIDENCE  = 0.45   # Separate floor for SELL signals (can tune independently)
MIN_BUY_CONFIDENCE   = 0.55   # Stricter floor for BUY/LONG entries


def apply_confidence_filter(action: str, confidence: float, symbol: str) -> tuple[str, float]:
    """
    Noise-filter layer applied AFTER ensemble aggregation.

    Rules:
        BUY  signal  with conf < MIN_BUY_CONFIDENCE   → demoted to HOLD
        SELL signal  with conf < MIN_SELL_CONFIDENCE   → demoted to HOLD
        Any  signal  with conf < MIN_SWING_CONFIDENCE  → demoted to HOLD
    """
    original = action

    # Global floor (catches everything below baseline)
    if confidence < MIN_SWING_CONFIDENCE:
        action = "HOLD"
    elif action == "BUY" and confidence < MIN_BUY_CONFIDENCE:
        action = "HOLD"
    elif action == "SELL" and confidence < MIN_SELL_CONFIDENCE:
        action = "HOLD"

    if action != original:
        logger.info(
            "[CONF FILTER] %s %s→HOLD conf=%.2f (threshold=%.2f)",
            symbol, original, confidence,
            MIN_BUY_CONFIDENCE if original == "BUY" else MIN_SELL_CONFIDENCE
        )

    return action, confidence


@dataclass
class Signal:
    symbol: str
    action: str
    confidence: float
    rf_score: float
    lstm_score: float
    xgb_score: float
    trans_score: float
    tech_score: float
    price: float
    reason: str
    shadow_mode: bool = False
    adx: Optional[float] = None
    vix: Optional[float] = None
    spy_price: Optional[float] = None
    spy_sma50: Optional[float] = None
    sma50_slope: Optional[float] = None
    regime: Optional[str] = None
    timestamp: Optional[str] = None
    quote_age_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Computes 14-period Wilder's ADX from daily OHLCV."""
    if df is None or len(df) < period * 2:
        return None
    try:
        high = df["High"]
        low = df["Low"]
        close_prev = df["Close"].shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean()
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / tr_smooth
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / tr_smooth

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
        return float(adx) if math.isfinite(adx) else None
    except Exception as exc:
        logger.debug("Error computing ADX: %s", exc)
        return None


def get_macro_benchmark_indicators() -> Dict[str, Any]:
    """
    Fetches SPY and VIX to compute the 5 canonical macro inputs:
    adx, vix, spy_price, spy_sma50, sma50_slope, and classifies market regime.
    """
    from strategies.regime_classifier import classify_market_regime, MarketRegime
    from strategies.session import now_eastern

    res = {
        "adx": None,
        "vix": None,
        "spy_price": None,
        "spy_sma50": None,
        "sma50_slope": None,
        "regime": MarketRegime.DATA_INVALID.value,
        "quote_age_seconds": 0.0,
        "timestamp": now_eastern().isoformat(),
    }

    try:
        spy_df = fetch_ohlcv("SPY")
        if spy_df is not None and len(spy_df) >= 50:
            spy_px = float(spy_df["Close"].iloc[-1])
            sma50_series = spy_df["Close"].rolling(50).mean()
            spy_sma50 = float(sma50_series.iloc[-1])
            if len(sma50_series.dropna()) >= 2:
                sma50_slope = float(sma50_series.iloc[-1] - sma50_series.iloc[-2])
            else:
                sma50_slope = 0.0
            adx_val = compute_adx(spy_df, 14)

            res["spy_price"] = round(spy_px, 2)
            res["spy_sma50"] = round(spy_sma50, 2)
            res["sma50_slope"] = round(sma50_slope, 4)
            res["adx"] = round(adx_val, 2) if adx_val is not None else None
    except Exception as exc:
        logger.warning("[MACRO BENCHMARK] Failed fetching or computing SPY indicators: %s", exc)

    try:
        vix_df = None
        for vix_sym in ("^VIX", "VIX"):
            try:
                vix_df = fetch_ohlcv(vix_sym)
                if vix_df is not None and not vix_df.empty:
                    break
            except Exception:
                continue

        if vix_df is not None and not vix_df.empty:
            res["vix"] = round(float(vix_df["Close"].iloc[-1]), 2)
        elif spy_df is not None and len(spy_df) >= 20:
            # Zero-budget empirical fallback proxy: SPY 20-day annualized realized volatility
            spy_daily_returns = spy_df["Close"].pct_change().dropna()
            if len(spy_daily_returns) >= 20:
                realized_vol_20d = float(spy_daily_returns.tail(20).std() * (252 ** 0.5) * 100.0)
                if math.isfinite(realized_vol_20d) and realized_vol_20d > 0:
                    res["vix"] = round(realized_vol_20d, 2)
                    logger.info("[MACRO BENCHMARK] VIX feed unavailable; using SPY 20d realized vol proxy: %.2f", res["vix"])
    except Exception as exc:
        logger.warning("[MACRO BENCHMARK] Failed fetching VIX: %s", exc)

    reg_class = classify_market_regime(
        adx=res["adx"],
        vix=res["vix"],
        spy_price=res["spy_price"],
        spy_sma50=res["spy_sma50"],
        sma50_slope=res["sma50_slope"],
        quote_age_seconds=res["quote_age_seconds"],
    )
    res["regime"] = reg_class.regime.value
    res["allows_new_entries"] = reg_class.allows_new_entries
    res["regime_reason"] = reg_class.reason
    return res


def technical_score_from_feature_row(row: pd.Series) -> float:
    """Deterministic technical score in [0, 1] from one feature row."""
    score = 0.5

    rsi = float(row.get("rsi", 50.0))
    if rsi < 30:
        score += 0.15
    elif rsi < 45:
        score += 0.07
    elif rsi > 70:
        score -= 0.15
    elif rsi > 55:
        score -= 0.07

    macd_hist = float(row.get("macd_hist", 0.0))
    score += 0.10 if macd_hist > 0 else -0.10

    bb_pct = float(row.get("bb_pct", 0.5))
    if bb_pct < 0.2:
        score += 0.10
    elif bb_pct > 0.8:
        score -= 0.10

    sma_cross = float(row.get("sma_cross", 0.0))
    score += 0.05 if sma_cross > 0 else -0.05

    return float(np.clip(score, 0.0, 1.0))


def _technical_score(df: pd.DataFrame, cfg=None) -> float:
    feat = build_features(df, cfg=cfg)
    if feat.empty:
        return 0.5
    return technical_score_from_feature_row(feat.iloc[-1])


def _safe_score(fn, *args, **kwargs) -> Tuple[float, bool]:
    """Returns (score, ok). ok=False means the score should be treated as missing."""
    try:
        score = float(fn(*args, **kwargs))
        if not np.isfinite(score):
            raise ValueError(f"non-finite score: {score}")
        return float(np.clip(score, 0.0, 1.0)), True
    except ModelNotAvailableError as exc:
        logger.debug("Sub-model unavailable: %s", exc)
        return 0.5, False
    except Exception:
        logger.exception("Sub-model prediction failed")
        return 0.5, False


def ml_model_count(rf_ok: bool, lstm_ok: bool, xgb_ok: bool = False, trans_ok: bool = False, cfg=None) -> int:
    """Count ML models that can actually influence the ensemble confidence."""
    cfg = cfg or config.get_settings()
    rf_counts = bool(rf_ok) and cfg.WEIGHT_RF > 0
    lstm_counts = bool(lstm_ok) and cfg.WEIGHT_LSTM > 0
    xgb_counts = bool(xgb_ok) and getattr(cfg, "WEIGHT_XGB", 0) > 0
    trans_counts = bool(trans_ok) and getattr(cfg, "WEIGHT_TRANSFORMER", 0) > 0
    return int(rf_counts) + int(lstm_counts) + int(xgb_counts) + int(trans_counts)


def enough_ml_models(rf_ok: bool, lstm_ok: bool, xgb_ok: bool = False, trans_ok: bool = False, cfg=None) -> bool:
    cfg = cfg or config.get_settings()
    return ml_model_count(rf_ok, lstm_ok, xgb_ok, trans_ok, cfg) >= cfg.MIN_ML_MODELS_FOR_SIGNAL


def weighted_blend(
    rf_score: float,
    rf_ok: bool,
    lstm_score: float,
    lstm_ok: bool,
    tech_score: float,
    xgb_score: float = 0.5,
    xgb_ok: bool = False,
    trans_score: float = 0.5,
    trans_ok: bool = False,
    cfg=None,
) -> float:
    """Blend available scores while respecting missing model flags."""
    cfg = cfg or config.get_settings()
    pairs = [
        (rf_score, cfg.WEIGHT_RF if rf_ok else 0.0),
        (lstm_score, cfg.WEIGHT_LSTM if lstm_ok else 0.0),
        (xgb_score, getattr(cfg, "WEIGHT_XGB", 0.20) if xgb_ok else 0.0),
        (trans_score, getattr(cfg, "WEIGHT_TRANSFORMER", 0.15) if trans_ok else 0.0),
        (tech_score, cfg.WEIGHT_TECHNICAL),
    ]
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.5
    return float(np.clip(sum(s * w for s, w in pairs) / total_w, 0.0, 1.0))


def action_from_confidence(confidence: float, cfg=None) -> str:
    cfg = cfg or config.get_settings()
    if confidence >= cfg.BUY_THRESHOLD:
        return "BUY"
    if confidence <= cfg.SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def apply_position_rule_with_hold(
    position: int,
    signal: str,
    allow_short: bool,
    bars_held: int,
    min_hold: int,
    entry_price: Optional[float] = None,
    current_price: Optional[float] = None,
    hard_stop_pct: Optional[float] = None,
    low_price: Optional[float] = None,
) -> Tuple[int, bool, str, int]:
    """Advance a unit position by one bar according to a broker-like rule."""
    signal = signal.upper()
    min_hold = max(1, int(min_hold))

    # ── Hard-stop backstop (long-only, bypasses min_hold and the signal) ──
    if (
        position > 0
        and hard_stop_pct is not None
        and entry_price is not None
        and entry_price > 0
    ):
        eval_price = low_price if low_price is not None else current_price
        if eval_price is not None:
            current_return = float(eval_price) / float(entry_price) - 1.0
            if current_return < -abs(float(hard_stop_pct)):
                return 0, True, f"hard-stop ({current_return:.4f})", 0

    if signal == "BUY":
        if position > 0:
            return position, False, "hold-long (already long)", bars_held + 1
        if position < 0:
            if bars_held < min_hold:
                return position, False, f"hold-short (min_hold {bars_held + 1}/{min_hold})", bars_held + 1
            return 0, True, "cover-short", 0
        return 1, True, "open-long", 1

    if signal == "SELL":
        if position > 0:
            if bars_held < min_hold:
                return position, False, f"hold-long (min_hold {bars_held + 1}/{min_hold})", bars_held + 1
            return 0, True, "close-long", 0
        if position < 0:
            return position, False, "hold-short (already short)", bars_held + 1
        if allow_short:
            return -1, True, "open-short", 1
        return 0, False, "skip-sell (flat, long-only)", 0

    # HOLD
    if position != 0:
        return position, False, "hold", bars_held + 1
    return 0, False, "flat", 0


class Predictor:
    def __init__(self, settings=None) -> None:
        self.cfg = settings or config.get_settings()
        self.rf = StockRFEngine(settings=self.cfg)
        self.lstm = StockLSTMEngine(settings=self.cfg)
        self.xgb = StockXGBEngine(settings=self.cfg)
        self.trans = StockTransformerEngine(settings=self.cfg)

        # Best-effort load up-front so each predict_all call doesn't pay the cost.
        self.rf.load()
        self.lstm.load()
        self.xgb.load()
        self.trans.load()

    def predict_all(self, symbols: Optional[List[str]] = None) -> List[Signal]:
        symbols = symbols or self.cfg.WATCHLIST
        signals: List[Signal] = []

        # Market Close Buffer Guard: Avoid running swing rebalances in final 15 mins (stale/unstable data)
        try:
            import zoneinfo
            now_et = datetime.now(zoneinfo.ZoneInfo("US/Eastern"))
            if now_et.hour == 15 and now_et.minute >= 45:
                logger.info("[MARKET CLOSE BUFFER] Within 15 mins of market close (> 15:45 EDT). Skipping swing rebalancing cycle.")
                return []
        except Exception:
            pass

        failed_count = 0
        macro_info = get_macro_benchmark_indicators()
        logger.info(
            "[MACRO REGIME] SPY=%s SMA50=%s Slope=%s ADX=%s VIX=%s → Regime: %s (allows_entries=%s)",
            macro_info.get("spy_price"),
            macro_info.get("spy_sma50"),
            macro_info.get("sma50_slope"),
            macro_info.get("adx"),
            macro_info.get("vix"),
            macro_info.get("regime"),
            macro_info.get("allows_new_entries", False),
        )

        for symbol in symbols:
            try:
                df = fetch_ohlcv(symbol)

                rf_score, rf_ok = _safe_score(self.rf.predict, symbol, df)
                lstm_score, lstm_ok = _safe_score(self.lstm.predict, symbol, df)
                xgb_score, xgb_ok = _safe_score(self.xgb.predict, symbol, df)
                trans_score, trans_ok = _safe_score(self.trans.predict, symbol, df)
                tech_score = _technical_score(df, self.cfg)

                if not enough_ml_models(rf_ok, lstm_ok, xgb_ok, trans_ok, self.cfg):
                    confidence = 0.5
                    action = "HOLD"
                    reason = (
                        f"RF={rf_score:.2f}{'' if rf_ok else '?'} "
                        f"LSTM={lstm_score:.2f}{'' if lstm_ok else '?'} "
                        f"XGB={xgb_score:.2f}{'' if xgb_ok else '?'} "
                        f"TRANS={trans_score:.2f}{'' if trans_ok else '?'} "
                        f"Tech={tech_score:.2f} → ML models missing, forced HOLD"
                    )
                else:
                    confidence = weighted_blend(
                        rf_score, rf_ok, lstm_score, lstm_ok, tech_score,
                        xgb_score, xgb_ok, trans_score, trans_ok, self.cfg
                    )
                    action = action_from_confidence(confidence, self.cfg)
                    action, confidence = apply_confidence_filter(action, confidence, symbol)
                    reason = (
                        f"RF={rf_score:.2f}{'' if rf_ok else '?'} "
                        f"LSTM={lstm_score:.2f}{'' if lstm_ok else '?'} "
                        f"XGB={xgb_score:.2f}{'' if xgb_ok else '?'} "
                        f"TRANS={trans_score:.2f}{'' if trans_ok else '?'} "
                        f"Tech={tech_score:.2f} → ensemble={confidence:.2f}"
                    )

                # Phase 1 signal-safety gate: never act on a model whose persisted
                # metrics are missing, stale, or below the performance floor. A
                # blocked symbol is forced to HOLD with the reason shown to the user.
                if action != "HOLD":
                    gate = model_metrics.evaluate_gate(symbol)
                    if not gate.ok:
                        action = "HOLD"
                        reason = f"{reason} → BLOCKED[{gate.status}]: {gate.reason} → HOLD"

                # Phase 2 macro regime gate: fail closed if macro regime restricts entries
                if action == "BUY" and not macro_info.get("allows_new_entries", False):
                    action = "HOLD"
                    reason = f"{reason} → BLOCKED[{macro_info.get('regime', 'UNKNOWN')}]: {macro_info.get('regime_reason', 'macro regime restricts entries')} → HOLD"

                price = float(df["Close"].iloc[-1])
                is_shadow = symbol in getattr(self.cfg, "SHADOW_MONITOR", [])
                if is_shadow:
                    reason = f"[SHADOW MONITOR] {reason} (monitor-only, no order)"
                signals.append(Signal(
                    symbol=symbol,
                    action=action,
                    confidence=round(float(confidence), 4),
                    rf_score=round(rf_score, 4),
                    lstm_score=round(lstm_score, 4),
                    xgb_score=round(xgb_score, 4),
                    trans_score=round(trans_score, 4),
                    tech_score=round(tech_score, 4),
                    price=round(price, 2),
                    reason=reason,
                    shadow_mode=is_shadow,
                    adx=macro_info.get("adx"),
                    vix=macro_info.get("vix"),
                    spy_price=macro_info.get("spy_price"),
                    spy_sma50=macro_info.get("spy_sma50"),
                    sma50_slope=macro_info.get("sma50_slope"),
                    regime=macro_info.get("regime"),
                    timestamp=macro_info.get("timestamp"),
                    quote_age_seconds=macro_info.get("quote_age_seconds", 0.0),
                ))

                # Option C: Log paired inference (incumbent vs shadow) to data/shadow_predictions.jsonl
                try:
                    from shadow_logger import get_shadow_logger
                    bar_dt = str(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1])
                    get_shadow_logger().log_prediction_pair(symbol, bar_dt, price, signals[-1], df)
                except Exception as exc_sh:
                    logger.debug("[SHADOW] Logger invocation failed for %s: %s", symbol, exc_sh)

                if is_shadow:
                    logger.info(
                        "👁️ %-6s [SHADOW] %s conf=%.2f price=%.2f | %s",
                        symbol, action, confidence, price, reason,
                    )
                else:
                    logger.info(
                        "%-6s %s conf=%.2f price=%.2f | %s",
                        symbol, action, confidence, price, reason,
                    )
            except Exception:
                failed_count += 1
                logger.exception("Prediction failed for %s", symbol)

        if failed_count > 5:
            logger.warning(
                "[PARTIAL SCAN GUARD] Data fetch failed for %d/%d symbols. Aborting rebalance cycle to prevent partial universe bias.",
                failed_count, len(symbols)
            )
            return []

        signals.sort(key=lambda s: s.confidence, reverse=True)
        xgb_fallback_count = sum(1 for s in signals if "XGB=0.50?" in s.reason)
        if xgb_fallback_count > 0:
            logger.info("[ML PREDICTOR] %d/%d symbols used XGBoost 0.50 fallback", xgb_fallback_count, len(signals))
        return signals
