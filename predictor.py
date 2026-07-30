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
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

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

    def to_dict(self) -> dict:
        return asdict(self)


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

                price = float(df["Close"].iloc[-1])
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
                ))
                logger.info(
                    "%-6s %s conf=%.2f price=%.2f | %s",
                    symbol, action, confidence, price, reason,
                )
            except Exception:
                logger.exception("Prediction failed for %s", symbol)

        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
