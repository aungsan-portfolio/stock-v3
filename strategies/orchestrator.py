from datetime import datetime, timezone
from pathlib import Path
import os
import time
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple
import sys

import config
from dataclasses import dataclass
from strategies.base import TradeSignal, EntryReadiness
from strategies.orb_strategy import ORBStrategy
from strategies.vwap_strategy import VWAPBounceStrategy
from strategies.gap_strategy import GapAndGoStrategy
from strategies.momentum_strategy import MomentumScalpStrategy
from strategies.candlestick_strategy import CandlestickPatternStrategy
from strategies.constants import StrategyName
from strategies.intraday_data import fetch_intraday
from strategies.order_manager import execute_signal, _min_confidence_for
from strategies.trade_journal import today_trade_count, today_pnl, day_trades_in_last_5_days
from strategies.performance import profile_latency
from strategies.position_sizer import calculate_shares, validate_executable_geometry

logger = logging.getLogger(__name__)

_consecutive_broker_failures = 0
_last_evaluated_5m_bar_str: Optional[str] = None
_skipped_bar_ids: set[str] = set()
_engine_boot_ts: float = time.time()



@dataclass(frozen=True)
class EvaluationResult:
    """Immutable result from symbol evaluation, containing signals and data-readiness context."""
    entry_signals: tuple[TradeSignal, ...]
    entry_allowed: bool
    suppression_reason: Optional[str]
    coverage: float
    required_coverage: float
    received_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]

    def __iter__(self):
        return iter(self.entry_signals)

    def __len__(self):
        return len(self.entry_signals)

    def __getitem__(self, idx):
        return self.entry_signals[idx]


from strategies.regime_engine import RegimeStateTracker

_regime_state_tracker = RegimeStateTracker()

_MR_SHADOW_RUNNER = None

def _get_mr_shadow_runner():
    """Hoisted singleton runner for D2 Mean Reversion shadow experiment."""
    global _MR_SHADOW_RUNNER
    if _MR_SHADOW_RUNNER is None:
        from strategies.shadow.mean_reversion_shadow import (
            MeanReversionShadowRunner,
            SHADOW_DEVIATION_MULT,
        )
        _MR_SHADOW_RUNNER = MeanReversionShadowRunner(deviation_mult=SHADOW_DEVIATION_MULT)
    return _MR_SHADOW_RUNNER

_ALL_STRATEGIES = [
    ORBStrategy(),
    VWAPBounceStrategy(),
    GapAndGoStrategy(),
    MomentumScalpStrategy(),
    CandlestickPatternStrategy(),
]

import re


def _extract_signal_rsi(sig: TradeSignal) -> float:
    """Extracts numeric RSI value from signal attributes, metadata, or reason string."""
    if hasattr(sig, "rsi") and getattr(sig, "rsi", None) is not None:
        try:
            return float(getattr(sig, "rsi"))
        except Exception:
            pass
    if isinstance(getattr(sig, "metadata", None), dict) and "rsi" in sig.metadata:
        try:
            return float(sig.metadata["rsi"])
        except Exception:
            pass
    reason_str = str(getattr(sig, "reason", ""))
    match = re.search(r"RSI\s*([0-9]+(?:\.[0-9]+)?)", reason_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            pass
    return 50.0


def _is_inverse_etf_signal(sig: TradeSignal) -> bool:
    """Returns True if signal is for an inverse/bear ETF."""
    inverse_etfs = getattr(config, "INVERSE_ETF_SYMBOLS", set())
    return sig.symbol.upper().strip() in inverse_etfs


def _inverse_etf_entry_allowed(regime_res, sig: TradeSignal) -> bool:
    """
    Determines if an inverse ETF long entry is allowed under current regime.
    Inverse ETFs are 'long' buys that profit from market decline,
    so they are VALID in RISK_OFF mode.
    """
    if not getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False):
        return False
    if not _is_inverse_etf_signal(sig):
        return False
    if regime_res is None:
        return False
    if sig.side.upper() not in ("BUY", "LONG"):
        return False

    if hasattr(regime_res, "allow_inverse_etf_longs") and regime_res.allow_inverse_etf_longs:
        return True

    from strategies.regime_engine import MODE_RISK_OFF, MODE_CAUTION
    if getattr(regime_res, "mode", "") in (MODE_RISK_OFF, MODE_CAUTION):
        return bool(getattr(regime_res, "halt_new_entries", False))
    return False


def _normalize_strategy_key(strategy: Any) -> str:
    if strategy is None:
        return ""
    if hasattr(strategy, "value"):
        return str(strategy.value).upper().strip()
    if hasattr(strategy, "name"):
        return str(strategy.name).upper().strip()
    return str(strategy).upper().strip()


def _strategy_allowed_in_texture(strategy: Any, allowed_strats: Optional[Any]) -> bool:
    if allowed_strats is None:
        return True
    strat_key = _normalize_strategy_key(strategy)
    for allowed in allowed_strats:
        if strategy == allowed or strat_key == _normalize_strategy_key(allowed):
            return True
    return False


def is_strategy_admitted_for_texture(
    strategy: Any,
    market_texture: str,
    bridge: Optional[Any] = None,
    signal: Optional[TradeSignal] = None,
) -> Tuple[bool, bool, Optional[str]]:
    """
    Evaluates whether a strategy is admitted under the current market texture.
    Supports fail-closed experimental paper routing for specific research cells
    (e.g., CANDLESTICK in SIDEWAYS or TRANSITIONING).

    Returns:
        (is_admitted: bool, is_paper_experiment: bool, cell_tag: Optional[str])
    """
    market_tex = str(market_texture or "TRENDING").upper().strip()
    tex_cfg = getattr(config, "MARKET_REGIME_SETTINGS", {}).get(market_tex, {})
    raw_allowed = tex_cfg.get("allowed_strategies", None)

    is_inverse_exempt = False
    if signal is not None:
        is_inverse_exempt = _is_inverse_etf_signal(signal) and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)

    if is_inverse_exempt or _strategy_allowed_in_texture(strategy, raw_allowed):
        return True, False, None

    # Check Paper Strategy Experiment Lane (Strict fail-closed)
    strat_key = _normalize_strategy_key(strategy)
    cell_tag = f"{strat_key}_{market_tex}"
    allowed_cells = set(getattr(config, "PAPER_EXPERIMENT_CELLS", ("CANDLESTICK_SIDEWAYS", "CANDLESTICK_TRANSITIONING")))

    if cell_tag not in allowed_cells:
        return False, False, None

    is_experiment_flag_on = getattr(config, "PAPER_EXPERIMENT_LANE_ENABLED", False) is True
    if not is_experiment_flag_on:
        return False, False, None

    # Explicit verified paper account guard
    is_paper_broker = False
    if bridge is not None:
        is_paper_attr = getattr(bridge, "is_paper", None)
        is_paper_account_attr = getattr(bridge, "is_paper_account", None)
        endpoint = str(getattr(bridge, "endpoint", "") or getattr(bridge, "_endpoint", "")).lower()

        if is_paper_attr is True or is_paper_account_attr is True or "paper-api" in endpoint:
            acct_mode = str(getattr(bridge, "account_mode", "")).upper()
            if acct_mode != "LIVE":
                is_paper_broker = True

    if not is_paper_broker:
        return False, False, None

    # Admitted as Paper Experiment: tag signal metadata if provided
    if signal is not None:
        if getattr(signal, "metadata", None) is None:
            signal.metadata = {}
        signal.metadata["lane"] = "PAPER_EXPERIMENT"
        signal.metadata["paper_experiment"] = True
        signal.metadata["paper_experiment_cell"] = cell_tag

    return True, True, cell_tag


SIGNAL_FEATURE_COLUMNS = [
    "timestamp",
    "timestamp_et",
    "symbol",
    "trigger_price",
    "entry_price",
    "stop_price",
    "target_price",
    "side",
    "confidence",
    "raw_confidence",
    "correlation_penalty",
    "max_correlation",
    "risk_per_share",
    "rr_ratio",
    "regime_score",
    "regime_mode",
    "regime_texture",
    "market_texture",
    "spy_pct",
    "qqq_pct",
    "soxx_pct",
    "sizing_pct",
    "halt_entries",
    "would_have_entered",
    "signal_type",
    "strategy",
    "signal_id",
]


def export_candidate_signals(
    signals: list,
    regime: Optional[object] = None,
    market_texture: str = "TRENDING",
    export_dir: str = "data/exports",
) -> None:
    """
    Write candidate signals to daily CSV for ML training (daytrade-train).
    Appends to today's file; creates it with headers if new.
    """
    if not signals:
        return

    try:
        from pathlib import Path
        import csv
        import pytz
        from datetime import timezone
        _ET = pytz.timezone("America/New_York")

        Path(export_dir).mkdir(parents=True, exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        today_str = datetime.now(_ET).strftime("%Y%m%d")
        filepath = os.path.join(export_dir, f"signals_{today_str}.csv")

        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0

        ts_utc = now_utc.isoformat()
        ts_et = datetime.now(_ET).strftime("%Y-%m-%d %H:%M:%S")

        reg_score = getattr(regime, "score", 0)
        reg_mode = getattr(regime, "mode", "UNKNOWN")
        reg_texture = getattr(regime, "market_texture", market_texture)
        sizing_pct = f"{getattr(regime, 'position_scale', 1.0):.2f}"
        halt_entries = getattr(regime, "halt_new_entries", False)

        reg_data = getattr(regime, "data", {}) or {}
        spy_pct = reg_data.get("SPY", {}).get("pct", "")
        qqq_pct = reg_data.get("QQQ", {}).get("pct", "")
        soxx_pct = reg_data.get("SOXX", {}).get("pct", "")

        rows = []
        for sig in signals:
            strat_val = getattr(sig, "strategy", "UNKNOWN")
            strat_str = strat_val.value if hasattr(strat_val, "value") else str(strat_val)
            meta = getattr(sig, "metadata", {}) or {}

            entry_px = float(getattr(sig, "entry_price", 0.0) or 0.0)
            stop_px = float(getattr(sig, "stop_price", 0.0) or 0.0)
            target_px = float(getattr(sig, "target_price", 0.0) or 0.0)
            dist = abs(entry_px - stop_px)
            reward = abs(target_px - entry_px)
            rr = getattr(sig, "reward_risk_ratio", None)
            if rr is None or rr <= 0:
                rr = round(reward / dist, 2) if dist > 1e-6 else 2.0

            rows.append({
                "timestamp": ts_utc,
                "timestamp_et": ts_et,
                "symbol": getattr(sig, "symbol", "").upper().strip(),
                "trigger_price": entry_px,
                "entry_price": entry_px,
                "stop_price": stop_px,
                "target_price": target_px,
                "side": getattr(sig, "side", "BUY").upper(),
                "confidence": float(getattr(sig, "confidence", 0.70)),
                "raw_confidence": float(meta.get("raw_confidence", getattr(sig, "confidence", 0.70))),
                "correlation_penalty": float(meta.get("correlation_penalty", 1.0)),
                "max_correlation": float(meta.get("max_correlation", 0.0)),
                "risk_per_share": float(getattr(sig, "risk_per_share", dist)),
                "rr_ratio": float(rr),
                "regime_score": reg_score,
                "regime_mode": reg_mode,
                "regime_texture": reg_texture,
                "market_texture": market_texture,
                "spy_pct": spy_pct,
                "qqq_pct": qqq_pct,
                "soxx_pct": soxx_pct,
                "sizing_pct": sizing_pct,
                "halt_entries": halt_entries,
                "would_have_entered": True,
                "signal_type": strat_str,
                "strategy": strat_str,
                "signal_id": getattr(sig, "signal_id", ""),
            })

        with open(filepath, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SIGNAL_FEATURE_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        logger.debug("[SIGNAL EXPORT] Exported %d candidate signals to %s", len(rows), filepath)
    except Exception as exp_err:
        logger.debug("[SIGNAL EXPORT] Failed exporting signals: %s", exp_err)


def _prepare_closed_bars(df):
    """Ensure only fully closed bars are evaluated.
    If the last bar in df is still actively forming (<5m old or in future), remove it.
    If the last bar is already a completed closed bar, preserve it.
    """
    if df is not None and not df.empty and len(df) >= 1:
        try:
            from strategies.session import now_eastern
            now = now_eastern()
            last_ts = df.index[-1]
            if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is not None:
                last_ts_et = last_ts.astimezone(now.tzinfo)
            else:
                last_ts_et = last_ts
            # A 5m bar must have a full 300 seconds before it is closed.
            if hasattr(last_ts_et, "hour") and (now - last_ts_et).total_seconds() < 300:
                return df.iloc[:-1]
        except Exception:
            return df.iloc[:-1]
    return df


def expected_value_r(
    win_rate: float,
    avg_win_r: float,
    avg_loss_r: float,
    avg_cost_r: float = 0.08,
) -> float:
    """
    Calculates Friction-Adjusted Net Expected Value (Net EV) in R-multiples:
    Net EV (R) = (Pwin * AvgWinR) - (Ploss * AvgLossR) - AvgCostR
    """
    loss_rate = 1.0 - win_rate
    return (win_rate * avg_win_r) - (loss_rate * abs(avg_loss_r)) - avg_cost_r


def normalize_ev_score(
    net_ev_r: float,
    min_ev_r: float = -0.50,
    max_ev_r: float = 1.00,
) -> float:
    """Normalizes Net EV (R) into a 0 - 100 continuous score."""
    if max_ev_r <= min_ev_r:
        raise ValueError("max_ev_r must be greater than min_ev_r")
    raw_score = 100.0 * (net_ev_r - min_ev_r) / (max_ev_r - min_ev_r)
    return max(0.0, min(100.0, raw_score))


from numbers import Real


def is_finite_real(value: object) -> bool:
    """Validates that value is a real finite number and strictly not a boolean."""
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float, Real))
        and math.isfinite(float(value))
    )


def decide_lunch_trade(
    opportunity_score: float,
    net_ev_r: float,
    sample_count: int,
    hard_safety_passed: bool = True,
    regime_mult: float = 1.0,
) -> Dict[str, Any]:
    """
    4-Component Multiplicative Risk Pipeline with Round-Down to Approved Tiers:
    Final Risk Multiplier = Opportunity Tier Cap * EV Multiplier * Evidence Multiplier * Regime Multiplier
    """
    # ── Strict Numeric Input Validation (Fail-Closed) ──
    if not is_finite_real(opportunity_score):
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": "Opportunity score must be a finite real number"}
    if not 0.0 <= float(opportunity_score) <= 100.0:
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": f"Opportunity score outside 0-100: {opportunity_score}"}

    if not is_finite_real(net_ev_r):
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": "Net EV must be a finite real number"}

    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": "Sample count must be an integer"}
    if sample_count < 0:
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": "Sample count cannot be negative"}

    if not is_finite_real(regime_mult):
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": "Regime multiplier must be a finite real number"}
    if not 0.0 <= float(regime_mult) <= 1.0:
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": f"Regime multiplier outside approved range ({regime_mult:.3f}, expected 0.0 to 1.0)"}

    # ── Strict Boolean Check for Hard Safety ──
    if not isinstance(hard_safety_passed, bool):
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": "Hard safety status must be boolean"}
    if not hard_safety_passed:
        return {"decision": "REJECT", "risk_mult": 0.0, "raw_risk_mult": 0.0, "reason": "Hard safety gate failed"}

    # ── Sequential Gate Hierarchy with Precise Audit Explanations ──
    if opportunity_score < 60.0:
        return {
            "decision": "REJECT",
            "risk_mult": 0.0,
            "raw_risk_mult": 0.0,
            "reason": f"Lunch opportunity score below minimum: {opportunity_score:.1f} < 60.0",
        }
    if net_ev_r <= 0.0:
        return {
            "decision": "REJECT",
            "risk_mult": 0.0,
            "raw_risk_mult": 0.0,
            "reason": f"Historical net expected value is not positive: {net_ev_r:+.3f}R <= 0",
        }
    if sample_count < 30:
        return {
            "decision": "PAPER_ONLY",
            "risk_mult": 0.0,
            "raw_risk_mult": 0.0,
            "reason": f"Insufficient historical evidence: N={sample_count} < 30",
        }

    # 1. Opportunity Tier Cap
    if opportunity_score >= 85.0:
        opportunity_cap = 0.75
    elif opportunity_score >= 70.0:
        opportunity_cap = 0.50
    else:
        opportunity_cap = 0.35

    # 2. EV Multiplier
    if net_ev_r >= 0.40:
        ev_mult = 1.00
    elif net_ev_r >= 0.20:
        ev_mult = 0.85
    else:
        ev_mult = 0.60

    # 3. Evidence Multiplier
    if sample_count >= 100:
        evidence_mult = 1.00
    elif sample_count >= 50:
        evidence_mult = 0.75
    else:
        evidence_mult = 0.50

    raw_risk_mult = opportunity_cap * ev_mult * evidence_mult * regime_mult
    approved_tiers = (0.0, 0.25, 0.35, 0.50, 0.75)
    final_risk_mult = max(tier for tier in approved_tiers if tier <= raw_risk_mult)

    if final_risk_mult == 0.0:
        return {
            "decision": "PAPER_ONLY",
            "risk_mult": 0.0,
            "raw_risk_mult": round(raw_risk_mult, 4),
            "reason": f"Calculated risk multiplier is below the minimum live tier ({raw_risk_mult:.4f}x < 0.25x)",
        }

    tier_label = "Elite Midday Setup" if opportunity_score >= 85.0 else ("Strong Midday Trend" if opportunity_score >= 70.0 else "Defensive Midday Participation")
    return {
        "decision": "ALLOW",
        "risk_mult": final_risk_mult,
        "raw_risk_mult": round(raw_risk_mult, 4),
        "reason": (
            f"{tier_label}: Score={opportunity_score:.1f}, EV={net_ev_r:+.3f}R, "
            f"N={sample_count}, Regime={regime_mult:.2f}x, Raw={raw_risk_mult:.4f}x, Approved={final_risk_mult:.2f}x"
        ),
    }


def compute_lunch_opportunity_score(
    sig: TradeSignal,
    event_decision: Optional[object] = None,
    rvol: float = 1.0,
    spread_pct: float = 0.02,
    sample_count: int = 100
) -> Tuple[float, float, bool, str]:
    """
    Computes 5-Factor Lunch Alpha Allocation Score & Net EV Gate.
    Returns: (score, size_multiplier, is_allowed, reason)
    """
    if sig is None:
        return 0.0, 0.0, False, "Null signal"

    # Factor 1: Signal Confidence (0 - 25 pts)
    conf = float(getattr(sig, "confidence", 0.5) or 0.5)
    if conf >= 0.85:
        conf_pts = 25.0
    elif conf >= 0.70:
        conf_pts = 18.0
    elif conf >= 0.60:
        conf_pts = 10.0
    else:
        conf_pts = 4.0

    # Factor 2: RVOL & Liquidity (0 - 25 pts)
    sig_rvol = float(getattr(sig, "rvol", rvol) or rvol)
    if sig_rvol >= 3.0:
        rvol_pts = 25.0
    elif sig_rvol >= 2.0:
        rvol_pts = 18.0
    elif sig_rvol >= 1.5:
        rvol_pts = 10.0
    else:
        rvol_pts = 0.0

    # Factor 3: Market Context & Trend Alignment (0 - 20 pts)
    trend_pts = 15.0  # Default baseline for verified setup
    if hasattr(sig, "metadata") and isinstance(sig.metadata, dict):
        if sig.metadata.get("above_vwap") and sig.side.upper() in ("BUY", "LONG"):
            trend_pts = 20.0
        elif sig.metadata.get("below_vwap") and sig.side.upper() in ("SELL", "SHORT"):
            trend_pts = 20.0

    # Factor 4: Catalyst Strength (0 - 15 pts)
    catalyst_pts = 0.0
    cat_tag = "None"
    if event_decision is not None:
        affected = getattr(event_decision, "affected_symbols", None) or set()
        if sig.symbol.upper().strip() in affected:
            catalyst_pts = 15.0
            cat_tag = getattr(event_decision, "tag", "CATALYST") or "CATALYST"

    # Factor 5: Execution Quality & Spread (0 - 15 pts)
    spread_pts = 15.0 if spread_pct <= 0.05 else (8.0 if spread_pct <= 0.10 else 0.0)

    total_score = conf_pts + rvol_pts + trend_pts + catalyst_pts + spread_pts

    # Estimate EV parameters based on score bucket
    if total_score >= 85.0:
        p_win, avg_w, avg_l = 0.62, 1.60, 0.85
    elif total_score >= 70.0:
        p_win, avg_w, avg_l = 0.55, 1.30, 0.90
    elif total_score >= 60.0:
        p_win, avg_w, avg_l = 0.55, 1.30, 0.85
    else:
        p_win, avg_w, avg_l = 0.39, 0.90, 1.05

    net_ev = expected_value_r(p_win, avg_w, avg_l, avg_cost_r=0.08)
    res = decide_lunch_trade(
        opportunity_score=total_score,
        net_ev_r=net_ev,
        sample_count=sample_count,
        hard_safety_passed=True,
    )
    is_allowed = res["decision"] == "ALLOW"
    size_mult = res["risk_mult"]
    reason = res["reason"]

    return total_score, size_mult, is_allowed, reason


def _should_bypass_lunch_chop(sig: TradeSignal, event_decision: Optional[object]) -> bool:
    """Determine if a signal qualifies for Lunch Chop exception via Composite Score & Net EV."""
    score, size_mult, is_allowed, _ = compute_lunch_opportunity_score(sig, event_decision)
    return is_allowed


def _evaluate_mean_reversion(symbol, df, market_texture, scorer, bridge=None):
    """Route MR with real execution context without relaxing any acceptance gate."""
    if market_texture != "SIDEWAYS" or not getattr(config, "STRATEGY_MEAN_REVERSION_ENABLED", True):
        return None
    from strategies.mean_reversion import MeanReversionStrategy
    context = dict(getattr(df, "attrs", {}) or {})
    # Daily ATR here must already be a dollar distance. Broker's daily ATR percent
    # API is not substituted: its reference price and fallback provenance differ.
    quote = context.get("quote")
    get_quote = getattr(bridge, "get_quote", None) if bridge is not None else None
    if callable(get_quote):
        try:
            fresh_quote = get_quote(symbol)
            if fresh_quote is not None:
                quote = fresh_quote
        except Exception as exc:
            logger.debug("[MEAN REVERSION] %s: upstream quote unavailable: %s", symbol, exc)
    signal = MeanReversionStrategy().generate_signal(
        symbol, df, market_texture=market_texture, quote=quote, metadata=context,
    )
    if signal is None or not signal.is_valid:
        return None
    signal.metadata["raw_confidence"] = signal.confidence
    signal.confidence = scorer.calculate_risk_adjusted_confidence(signal)
    if not signal.bar_timestamp and hasattr(df, "index") and len(df.index) > 0:
        last_idx = df.index[-1]
        signal.bar_timestamp = last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx)
    feed_id = context.get("feed_identity")
    if not signal.decision_feed and feed_id:
        signal.decision_feed = str(feed_id).lower().strip()
    return signal


def select_viable_cycle_candidates(signals: List[TradeSignal]) -> List[TradeSignal]:
    """Candidate Selection & Deduplication Gate (Canonical Production Implementation):
    Evaluates candidate signals against RSI filters, inverse-ETF short restrictions,
    and executable geometry. Only when a candidate passes all checks does it claim
    the symbol (seen_symbols.add(sym)) and get retained in deduped_signals.
    """
    from strategies.position_sizer import validate_executable_geometry
    seen_symbols = set()
    deduped_signals = []

    for sig in signals:
        sym = sig.symbol.upper().strip()

        if sym in seen_symbols:
            logger.info("[SIGNAL DISPOSITION] Skipped %s: duplicate signal in this cycle (superseded by previous candidate)", sym)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_PENDING_ORDER",
                    decision="REJECT",
                    reason_code=VetoReason.DUPLICATE_ORDER,
                    human_message="Duplicate signal in cycle superseded by higher priority candidate",
                    terminal_decision=True,
                    blocked_by="STAGE_PENDING_ORDER",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        rsi_max = getattr(config, "RSI_MAX_BUY_ENTRY", 70.0)
        rsi_min = getattr(config, "RSI_MIN_SHORT_ENTRY", 30.0)
        sig_rsi = _extract_signal_rsi(sig)
        if sig.side.upper() in ("BUY", "LONG") and sig_rsi >= rsi_max:
            logger.info("Skipped %s: RSI %.1f >= %.1f (Overbought BUY entry filter)", sig.symbol, sig_rsi, rsi_max)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_RSI_FILTER",
                    decision="REJECT",
                    reason_code=VetoReason.RSI_OVERBOUGHT,
                    human_message=f"RSI {sig_rsi:.1f} >= {rsi_max:.1f} (Overbought)",
                    terminal_decision=True,
                    blocked_by="STAGE_RSI_FILTER",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue
        if sig.side.upper() in ("SELL", "SHORT") and sig_rsi <= rsi_min:
            logger.info("Skipped %s: RSI %.1f <= %.1f (Oversold SHORT entry filter)", sig.symbol, sig_rsi, rsi_min)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_RSI_FILTER",
                    decision="REJECT",
                    reason_code=VetoReason.RSI_OVERSOLD,
                    human_message=f"RSI {sig_rsi:.1f} <= {rsi_min:.1f} (Oversold)",
                    terminal_decision=True,
                    blocked_by="STAGE_RSI_FILTER",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue
        if _is_inverse_etf_signal(sig) and sig.side.upper() in ("SELL", "SHORT"):
            logger.info("Skipped %s: Inverse ETFs cannot be sold short (Long-only hedge)", sig.symbol)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_GEOMETRY",
                    decision="REJECT",
                    reason_code=VetoReason.INVALID_GEOMETRY,
                    human_message="Inverse ETFs cannot be sold short",
                    terminal_decision=True,
                    blocked_by="STAGE_GEOMETRY",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        geom_ok, geom_msg = validate_executable_geometry(sig)
        if not geom_ok:
            logger.info("[PRE-EXECUTION GEOMETRY VETO] %s: %s", sig.symbol, geom_msg)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_GEOMETRY",
                    decision="REJECT",
                    reason_code=VetoReason.INVALID_GEOMETRY,
                    human_message=geom_msg,
                    terminal_decision=True,
                    blocked_by="STAGE_GEOMETRY",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        seen_symbols.add(sym)
        deduped_signals.append(sig)

    return deduped_signals


def evaluate_symbol(symbol: str, bridge: object = None, market_texture: str = "TRENDING") -> List[TradeSignal]:
    """Evaluate one symbol through all strategies."""
    from strategies.intraday_data import fetch_intraday
    df = fetch_intraday(symbol, bridge=bridge)
    if df is None or df.empty:
        logger.warning("No data for %s", symbol)
        return []
    return _evaluate_on_df(symbol, df, market_texture=market_texture, bridge=bridge)


def _evaluate_on_df(symbol: str, df, market_texture: str = "TRENDING", bridge: object = None) -> List[TradeSignal]:
    """Internal sync implementation for one symbol given its DataFrame."""
    if df is None or df.empty:
        return []

    df = _prepare_closed_bars(df)
    if df is None or df.empty:
        return []


    signals = []
    failures = []

    # ── P1.6 Confirmed ORB Shadow Pipeline Integration ──
    if getattr(config, "P1_6_ORB_SHADOW_ENABLED", False):
        try:
            from strategies.shadow.lifecycle import get_shadow_lifecycle_manager
            from strategies.shadow.orb_confirmed import get_confirmed_orb_strategy
            shadow_mgr = get_shadow_lifecycle_manager()
            if shadow_mgr.is_enabled:
                # 1. Update existing active shadow trades on latest completed bar
                shadow_mgr.update_active_trades(symbol, df.iloc[-1])
                # 2. Evaluate new ORB shadow candidate
                orb_strat = get_confirmed_orb_strategy()
                shadow_cand = orb_strat.evaluate_candidate(
                    symbol=symbol,
                    df=df,
                    candidate_meta={"market_texture": market_texture},
                )
                if shadow_cand is not None:
                    shadow_mgr.register_candidate(shadow_cand)
                    logger.info(
                        "[ORB SHADOW CANDIDATE] Discovered %s %s @ %.2f (Candidate %s)",
                        shadow_cand.side, shadow_cand.symbol, shadow_cand.reference_price, shadow_cand.candidate_id[:8]
                    )
        except Exception as shadow_err:
            logger.warning("[ORB SHADOW ERROR] Shadow evaluation error on %s: %s", symbol, shadow_err)

    # ── P1.7 Mean Reversion Shadow Experiment Hook (D2 Shadow) ──
    if getattr(config, "EXPERIMENTAL_MR_SHADOW_ENABLED", False):
        try:
            from strategies.shadow.mean_reversion_shadow import log_mr_shadow_candidate
            mr_runner = _get_mr_shadow_runner()
            mr_result = mr_runner.evaluate_bar(symbol, df)
            if mr_result is not None:
                log_mr_shadow_candidate(mr_result, market_texture=market_texture)
        except Exception as mr_err:
            logger.debug("[MR SHADOW] Evaluation error on %s: %s", symbol, mr_err)

    from strategies.scoring import RiskAdjustedSignalScorer
    scorer = RiskAdjustedSignalScorer()

    import copy

    for strategy in _ALL_STRATEGIES:
        try:
            raw_signal = strategy.evaluate(symbol, df)
            if raw_signal is not None:
                if raw_signal.is_valid:
                    signal = copy.copy(raw_signal)
                    signal.metadata = copy.copy(raw_signal.metadata)
                    signal.metadata['raw_confidence'] = signal.confidence
                    signal.confidence = scorer.calculate_risk_adjusted_confidence(signal)
                    if not signal.bar_timestamp and hasattr(df, "index") and len(df.index) > 0:
                        last_idx = df.index[-1]
                        signal.bar_timestamp = last_idx.isoformat() if hasattr(last_idx, "isoformat") else str(last_idx)
                    if not getattr(signal, "decision_feed", None):
                        feed_id = getattr(df, "attrs", {}).get("feed_identity")
                        if feed_id:
                            signal.decision_feed = str(feed_id).lower().strip()
                    signals.append(signal)
                else:
                    failures.append(f"{strategy.name}: INVALID (reward/risk or geometry)")
        except Exception as e:
            logger.exception("Strategy %s crashed on %s", strategy.name, symbol)
            failures.append(f"{strategy.name}: CRASH ({e})")

    # Evaluate MeanReversionStrategy when in SIDEWAYS market texture
    if market_texture == "SIDEWAYS":
        try:
            mr_signal = _evaluate_mean_reversion(symbol, df, market_texture, scorer, bridge=bridge)
            if mr_signal is not None:
                signals.append(mr_signal)
        except Exception as mr_err:
            logger.exception("MeanReversionStrategy failed for %s: %s", symbol, mr_err)


    if signals:
        # ML prediction layer
        try:
            ml_enabled = getattr(config, "DAYTRADE_ML_ENABLED", False)
            if ml_enabled:
                from strategies.models.predictor import SignalPredictor
                predictor = SignalPredictor()
                if predictor.ready:
                    for sig in signals:
                        ml_score = predictor.predict(sig, {})
                        if ml_score is not None:
                            orig = sig.confidence
                            weight = getattr(config, "DAYTRADE_ML_CONFIDENCE_WEIGHT", 0.3)
                            sig.confidence = (1 - weight) * orig + weight * ml_score
                            sig.metadata['ml_score'] = ml_score
                            sig.metadata['ml_raw_confidence'] = orig
        except Exception as e:
            logger.debug(f"ML prediction skipped: {e}")

        return signals
    if failures:
        logger.info("Evaluated %s: No signals. (%s)", symbol, " | ".join(failures))
    return []


from dataclasses import dataclass


@dataclass(frozen=True)
class RecoveryPlan:
    held_symbols: Tuple[str, ...]
    candidate_symbols: Tuple[str, ...]
    skipped_symbols: Tuple[str, ...]


def build_recovery_plan(
    missing_symbols: List[str],
    *,
    held_symbols: Any = (),
    ranked_candidates: Any = (),
    candidate_limit: int = 5,
) -> RecoveryPlan:
    missing = list(dict.fromkeys(missing_symbols))
    missing_set = set(missing)
    held = tuple(
        symbol for symbol in (held_symbols or ()) if symbol in missing_set
    )
    held_set = set(held)
    candidates = tuple(
        symbol for symbol in (ranked_candidates or ())
        if symbol in missing_set and symbol not in held_set
    )[:candidate_limit]
    selected = held_set | set(candidates)
    skipped = tuple(
        symbol for symbol in missing if symbol not in selected
    )
    return RecoveryPlan(
        held_symbols=held,
        candidate_symbols=candidates,
        skipped_symbols=skipped,
    )


def recover_unresolved_concurrently(
    symbols: List[str],
    *,
    deadline: float,
    bridge = None,
    max_workers: Optional[int] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
    from strategies.intraday_data import fetch_intraday, _is_usable_intraday_data

    recovered: Dict[str, pd.DataFrame] = {}
    failures: Dict[str, str] = {}
    if not symbols:
        return recovered, failures

    workers_limit = max_workers or getattr(config, "DAYTRADE_RECOVERY_MAX_WORKERS", 4)
    eff_workers = max(1, min(workers_limit, len(symbols)))
    pool = ThreadPoolExecutor(
        max_workers=eff_workers,
        thread_name_prefix="intraday-recovery",
    )

    def _seconds_remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    future_to_symbol = {
        pool.submit(
            fetch_intraday,
            sym,
            interval="5m",
            bridge=bridge,
            purpose="PARTIAL_RECOVERY",
        ): sym
        for sym in symbols
    }
    pending = set(future_to_symbol)
    try:
        while pending:
            remaining = _seconds_remaining()
            if remaining <= 0:
                break
            done, pending = wait(
                pending,
                timeout=min(0.25, remaining),
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                sym = future_to_symbol[future]
                try:
                    df = future.result()
                    if _is_usable_intraday_data(df, interval="5m"):
                        recovered[sym] = df
                    else:
                        failures[sym] = "empty_or_stale"
                except Exception as exc:
                    failures[sym] = type(exc).__name__

        for future in pending:
            sym = future_to_symbol[future]
            cancelled = future.cancel()
            failures[sym] = (
                "total_deadline_cancelled" if cancelled else "total_deadline_running"
            )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    return recovered, failures


def recover_missing_intraday(
    plan: RecoveryPlan,
    *,
    bridge = None,
    total_timeout_seconds: Optional[float] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    from strategies.intraday_data import fetch_batch_intraday_alpaca

    timeout_budget = total_timeout_seconds if total_timeout_seconds is not None else float(getattr(config, "DAYTRADE_RECOVERY_TOTAL_TIMEOUT_SECONDS", 8.0))
    deadline = time.monotonic() + timeout_budget

    def _seconds_remaining() -> float:
        return max(0.0, deadline - time.monotonic())

    recovered: Dict[str, pd.DataFrame] = {}
    failures: Dict[str, str] = {}

    priority_symbols = list(plan.held_symbols) + list(plan.candidate_symbols)
    if not priority_symbols:
        for symbol in plan.skipped_symbols:
            failures[symbol] = "non_priority_fail_closed"
        return recovered, failures

    # 1. Prefer micro-batch request first
    try:
        batch_recovered = fetch_batch_intraday_alpaca(
            priority_symbols,
            interval="5m",
            purpose="PARTIAL_RECOVERY",
            total_timeout_seconds=_seconds_remaining(),
        )
        if batch_recovered:
            recovered.update(batch_recovered)
    except Exception as exc:
        logger.warning(
            "Batch partial recovery failed: symbols=%s error=%r",
            priority_symbols, exc,
        )

    unresolved = [sym for sym in priority_symbols if sym not in recovered]
    if unresolved and _seconds_remaining() > 0:
        single_recovered, single_failures = recover_unresolved_concurrently(
            unresolved,
            deadline=deadline,
            bridge=bridge,
        )
        recovered.update(single_recovered)
        failures.update(single_failures)

    for symbol in plan.skipped_symbols:
        failures[symbol] = "non_priority_fail_closed"

    for symbol in plan.candidate_symbols:
        if symbol not in recovered:
            failures.setdefault(symbol, "candidate_data_unavailable_fail_closed")

    for symbol in plan.held_symbols:
        if symbol not in recovered:
            failures.setdefault(symbol, "held_symbol_data_unavailable")
            logger.critical(
                "🚨 [HELD POSITION DATA FAILURE] Stale or missing data for active position: %s. Relying on broker protective orders.",
                symbol,
            )

    return recovered, failures


def evaluate_symbols_parallel(symbols: List[str], bridge: object = None, market_texture: str = "TRENDING") -> EvaluationResult:
    """Evaluate many symbols in parallel when the batch is large enough."""
    min_coverage = getattr(config, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95)
    if not symbols:
        return EvaluationResult(
            entry_signals=(),
            entry_allowed=True,
            suppression_reason=None,
            coverage=1.0,
            required_coverage=min_coverage,
            received_symbols=(),
            missing_symbols=(),
        )

    from strategies.intraday_data import fetch_intraday, fetch_intraday_batch

    batch_enabled = getattr(config, "DAYTRADE_BATCH_FETCH_ENABLED", False)
    parallel_enabled = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_ENABLED", True)
    min_batch = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_MIN_SYMBOLS", 4)
    max_workers = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_MAX_WORKERS", 3)
    min_coverage = getattr(config, "DAYTRADE_BATCH_FETCH_MIN_COVERAGE", 0.95)

    symbol_data = {}

    # 1. Primary Path: Multi-Symbol Batch Fetch (if enabled via feature flag)
    if batch_enabled and symbols:
        try:
            batch_results, telemetry = fetch_intraday_batch(symbols, interval="5m", bridge=bridge, purpose="ENTRY_SIGNAL")
            logger.info(
                "[BATCH FETCH] symbols=%d received=%d coverage=%.1f%% duration=%.1fms",
                telemetry["requested"],
                telemetry["received"],
                telemetry["coverage"] * 100.0,
                telemetry["duration_ms"]
            )
            symbol_data = batch_results

            missing_symbols = [s for s in symbols if s not in symbol_data or symbol_data[s].empty]

            try:
                from strategies.watchdog import touch_orchestrator_heartbeat
                touch_orchestrator_heartbeat(
                    stage="primary_data_retrieval_completed",
                    metadata={
                        "requested_symbols": len(symbols),
                        "missing_symbols": len(missing_symbols),
                        "elapsed_ms": telemetry.get("duration_ms", 0.0),
                    },
                )
            except Exception:
                pass

            # Priority-Partitioned Recovery: micro-batch first, bounded concurrent second
            if missing_symbols:
                held_set = set()
                try:
                    if bridge and hasattr(bridge, "get_open_positions"):
                        held_set = {p.symbol for p in bridge.get_open_positions()}
                    elif bridge and hasattr(bridge, "ib") and hasattr(bridge.ib, "positions"):
                        held_set = {p.contract.symbol for p in bridge.ib.positions() if p.position != 0}
                except Exception:
                    held_set = set()

                candidate_limit = int(getattr(config, "DAYTRADE_RECOVERY_MAX_CANDIDATES", 5))
                recovery_plan = build_recovery_plan(
                    missing_symbols,
                    held_symbols=held_set,
                    ranked_candidates=symbols,
                    candidate_limit=candidate_limit,
                )
                logger.warning(
                    "[PARTIAL RECOVERY PLAN] missing=%d held=%s candidates=%s skipped=%s",
                    len(missing_symbols),
                    recovery_plan.held_symbols,
                    recovery_plan.candidate_symbols,
                    recovery_plan.skipped_symbols,
                )

                t_rec_start = time.perf_counter()
                recovered_dfs, recovery_failures = recover_missing_intraday(
                    recovery_plan,
                    bridge=bridge,
                )
                rec_elapsed_ms = (time.perf_counter() - t_rec_start) * 1000.0
                symbol_data.update(recovered_dfs)

                try:
                    from strategies.watchdog import touch_orchestrator_heartbeat
                    touch_orchestrator_heartbeat(
                        stage="partial_recovery_completed",
                        metadata={
                            "recovered_symbols": len(recovered_dfs),
                            "failed_symbols": len(recovery_failures),
                            "elapsed_ms": rec_elapsed_ms,
                        },
                    )
                except Exception:
                    pass
        except Exception as batch_exc:
            logger.warning(
                "[CIRCUIT BREAKER] Batch fetch raised exception: %s. Degrading gracefully to full sequential fetch.",
                batch_exc
            )
            symbol_data = {}


    # 2. Fail-Safe Fallback: Proven ThreadPoolExecutor / Sequential Path
    if not symbol_data:
        if parallel_enabled and len(symbols) >= min_batch:
            fetched_data = {}
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(fetch_intraday, sym, bridge=bridge): sym for sym in symbols
                }
                for fut in as_completed(futures):
                    sym = futures[fut]
                    try:
                        df = fut.result()
                        if df is not None and not df.empty:
                            fetched_data[sym] = df
                    except Exception:
                        logger.exception("Data fetch failed for %s", sym)
            symbol_data = {sym: fetched_data[sym] for sym in symbols if sym in fetched_data}
        else:
            for sym in symbols:
                try:
                    df = fetch_intraday(sym, bridge=bridge)
                    if df is not None and not df.empty:
                        symbol_data[sym] = df
                except Exception as e:
                    logger.exception("Data fetch failed for %s", sym)

    if not symbol_data:
        return EvaluationResult(
            entry_signals=(),
            entry_allowed=False,
            suppression_reason="NO_DATA_FETCHED",
            coverage=0.0,
            required_coverage=min_coverage,
            received_symbols=(),
            missing_symbols=tuple(symbols),
        )

    # Fail-Closed Universe Selection Protection:
    # If coverage < DAYTRADE_BATCH_FETCH_MIN_COVERAGE (0.95), partial data would introduce severe
    # universe selection bias (e.g. evaluating only 5 arbitrary symbols instead of 23).
    # Suppress new entry signal generation completely while keeping position risk management active.
    received_symbols = tuple(s for s in symbols if s in symbol_data and not symbol_data[s].empty)
    missing_symbols = tuple(s for s in symbols if s not in symbol_data or symbol_data[s].empty)
    coverage = len(received_symbols) / len(symbols) if symbols else 1.0
    if coverage < min_coverage:
        reason = (
            f"LOW_COVERAGE: Intraday batch coverage {coverage*100:.1f}% ({len(received_symbols)}/{len(symbols)}) "
            f"below gate threshold {min_coverage*100:.1f}%"
        )
        logger.warning(
            "[LOW COVERAGE SUPPRESSION] %s. "
            "Suppressing entry evaluation to avoid universe selection bias. Positions/risk management remains active.",
            reason,
        )
        try:
            from strategies.gate_ledger import log_gate_decision
            from strategies.veto_reasons import VetoReason
            log_gate_decision(
                evaluation_id=f"cov_{int(time.time())}",
                symbol="UNIVERSE",
                bar="",
                stage="STAGE_COVERAGE",
                decision="REJECT",
                reason_code=VetoReason.LOW_COVERAGE,
                human_message=reason,
                terminal_decision=True,
                blocked_by="STAGE_COVERAGE",
                metadata={"coverage": coverage, "required": min_coverage, "missing": missing_symbols},
            )
        except Exception:
            pass
        return EvaluationResult(
            entry_signals=(),
            entry_allowed=False,
            suppression_reason=reason,
            coverage=coverage,
            required_coverage=min_coverage,
            received_symbols=received_symbols,
            missing_symbols=missing_symbols,
        )

    out = []
    if not parallel_enabled or len(symbol_data) < min_batch:
        for sym, df in symbol_data.items():
            out.extend(_evaluate_on_df(sym, df, market_texture=market_texture, bridge=bridge))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_evaluate_on_df, sym, df, market_texture, bridge): sym for sym, df in symbol_data.items()}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    out.extend(fut.result())
                except Exception as e:
                    logger.exception("Parallel evaluation failed for %s", sym)
                    fallback_serial = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_FALLBACK_TO_SERIAL", True)
                    if fallback_serial:
                        out.extend(_evaluate_on_df(sym, symbol_data[sym], market_texture=market_texture, bridge=bridge))
                    else:
                        raise e

    return EvaluationResult(
        entry_signals=tuple(out),
        entry_allowed=True,
        suppression_reason=None,
        coverage=coverage,
        required_coverage=min_coverage,
        received_symbols=received_symbols,
        missing_symbols=missing_symbols,
    )



def apply_portfolio_correlation(signals: List[TradeSignal], open_positions: List, bridge: object = None) -> List[TradeSignal]:
    """Applies portfolio correlation checks to signals."""
    # ── Patch A: Correlation Empty-Input Guard ──
    if not signals:
        return []

    from strategies.correlation import CorrelationAnalyzer, PortfolioPosition
    from strategies.position_sizer import calculate_shares

    analyzer = CorrelationAnalyzer()

    is_connected = getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)
    if is_connected:
        try:
            equity = bridge.get_net_liquidation()
        except Exception:
            equity = config.PDT_MIN_EQUITY
    else:
        equity = config.PDT_MIN_EQUITY

    position_details = []
    for position in open_positions:
        if isinstance(position, str):
            position_details.append(PortfolioPosition(position, "BUY", equity))
            continue
        qty = float(getattr(position, "position", 0.0) or 0.0)
        if qty == 0:
            continue
        symbol = getattr(getattr(position, "contract", None), "symbol", None)
        if not symbol:
            continue
        avg_cost = abs(float(getattr(position, "avgCost", 0.0) or 0.0))
        exposure = abs(qty) * avg_cost
        position_details.append(
            PortfolioPosition(
                symbol=symbol,
                side="BUY" if qty > 0 else "SELL",
                exposure=exposure,
            )
        )

    returns = analyzer.get_returns_map(
        [*(signal.symbol for signal in signals), *(p.symbol for p in position_details)],
        bridge=bridge,
    ) if position_details else {}

    from strategies.position_sizer import validate_executable_geometry
    for s in signals:
        if s.confidence <= 0:
            continue

        geom_valid, geom_reason = validate_executable_geometry(s)
        if not geom_valid:
            logger.info("[PRE-SIZER REJECT] %s: %s", s.symbol, geom_reason)
            s.confidence = 0
            continue

        candidate_exposure = calculate_shares(s, equity) * s.entry_price
        allowed, penalty, max_corr, overlap_symbols = analyzer.check_portfolio_impact(
            s.symbol,
            s.side,
            position_details,
            equity,
            bridge=bridge,
            precomputed_returns=returns,
            candidate_exposure=candidate_exposure,
        )

        s.confidence *= penalty
        s.metadata["correlation_penalty"] = penalty
        s.metadata["max_correlation"] = max_corr
        s.metadata["portfolio_overlap_symbols"] = overlap_symbols

        if not allowed:
            logger.info("Signal %s %s blocked due to high correlation.", s.side, s.symbol)
            s.confidence = 0

    return [s for s in signals if s.confidence > 0.0]


def _classify_execution_rejection(reason_str: str) -> tuple:
    """Map execution rejection reason strings to authoritative typed stages and VetoReason codes."""
    from strategies.veto_reasons import VetoReason
    raw = str(reason_str).strip().lower()
    r = raw.replace("-", "_")

    # 1. Counter-Trend Short Veto
    if "counter_trend" in r or "countertrend" in r or "counter trend" in raw:
        return "STAGE_COUNTERTREND_GUARD", VetoReason.COUNTER_TREND_SHORT_VETO

    # 2. Sector / Tech Cluster Cap
    if "cluster cap" in raw or "cluster_cap" in r or "tech cluster" in raw or "tech_cluster" in r or "sector concentration" in raw or "sector_concentration" in r:
        return "STAGE_SECTOR_CLUSTER_CAP", VetoReason.TECH_CLUSTER_CAP

    # 3. Real-Time Quote Health (freshness, inverted quotes, stale quotes)
    if "quote health" in raw or "stale quote" in raw or "quote_health" in r or "stale_quote" in r or ("quote" in raw and "stale" in raw):
        return "STAGE_QUOTE_HEALTH", VetoReason.STALE_QUOTE
    if "inverted quote" in raw or "inverted_quote" in r or "spread" in raw:
        return "STAGE_QUOTE_HEALTH", VetoReason.EXCESSIVE_SPREAD

    # 4. Other known typed execution/policy reasons
    if "suppression" in raw or "cooldown" in raw:
        return "STAGE_COOLDOWN", VetoReason.SESSION_COOLDOWN
    if "chase" in raw:
        return "STAGE_MAX_CHASE", VetoReason.MAX_CHASE
    if "trend guard" in raw or "trend_guard" in r or "spy 5m" in raw or "spy_5m" in r or "spy benchmark" in raw or ("blocking" in raw and "spy" in raw):
        return "STAGE_SPY_GUARD", VetoReason.SPY_BEARISH_5M
    if "soxx" in raw:
        return "STAGE_SOXX_GUARD", VetoReason.SOXX_TECH_LONG_VETO
    if "noise floor" in raw or "noise_floor" in r or "critically tight" in raw or "critically_tight" in r:
        return "STAGE_NOISE_FLOOR", VetoReason.NOISE_FLOOR
    if "position size is 0" in raw or "0 shares" in raw or "shares=0" in raw or "zero position" in raw or "zero_position" in r:
        return "STAGE_POSITION_SIZE", VetoReason.ZERO_POSITION_SIZE
    if "entry readiness" in raw or "entry_readiness" in r:
        return "STAGE_ENTRY_READINESS", VetoReason.ENTRY_READINESS_FAILED
    if "price ordering" in raw or "price_ordering" in r or "geometry" in raw:
        return "STAGE_GEOMETRY", VetoReason.INVALID_GEOMETRY

    # 5. Fallback strictly for true unhandled broker rejection errors
    return "STAGE_ORDER_EXECUTION", "ORDER_REJECTED"


def evaluate_and_execute(
    watchlist: List[str],
    bridge: object,
    live_paper: bool = False,
    error_handler: Optional['TradingErrorHandler'] = None,
    prime_broker_state: bool = True,
    allow_new_entries: bool = True,
):
    """Evaluates strategies for symbols in the watchlist and executes trades."""
    if not prime_broker_state:
        logger.debug("prime_broker_state=False is deprecated and ignored")

    global _effective_config_dumped
    if "_effective_config_dumped" not in globals() or not _effective_config_dumped:
        try:
            from strategies.gate_ledger import dump_effective_config_startup
            dump_effective_config_startup(logger)
            _effective_config_dumped = True
        except Exception as d_err:
            logger.debug("Config dump error: %s", d_err)

    logger.info(f"[ORCHESTRATOR POLL] Polling cycle started for {len(watchlist)} watchlist symbols | allow_new_entries={allow_new_entries}")
    print(f"[ORCHESTRATOR POLL] Polling cycle started for {len(watchlist)} watchlist symbols | allow_new_entries={allow_new_entries}")

    poll_id = f"poll_{int(time.time())}"
    try:
        from strategies.watchdog import touch_orchestrator_heartbeat
        touch_orchestrator_heartbeat(
            stage="poll_start",
            poll_id=poll_id,
            metadata={"watchlist_count": len(watchlist), "allow_new_entries": allow_new_entries},
        )
    except Exception:
        pass


    global _consecutive_broker_failures
    broker_fetch_ok = True

    broker_state_available = False
    broker_pnl = 0.0
    cached_open_positions = []
    open_orders = []
    equity = config.PDT_MIN_EQUITY
    from strategies.halt_manager import halt_manager
    # Sub-phase C.2: Poll HALT.lock file flag (<1ms OS check, fail-closed)
    halt_manager.check_halt_lock()

    if not allow_new_entries:
        logger.warning("[ENTRY GUARD] New entries explicitly disabled for this evaluation cycle.")
    if not halt_manager.entries_allowed() or getattr(bridge, "_daytrade_suspended", False) is True:
        logger.warning("[ENTRY GUARD] Active entry halts present: %s", halt_manager.summary())
        allow_new_entries = False
        try:
            from observability import get_latch_state_machine
            get_latch_state_machine().heartbeat_if_due(
                interval_seconds=float(getattr(config, "LATCH_HEARTBEAT_SECONDS", 300)),
                active_halts=list(halt_manager.active_halts()),
            )
        except Exception:
            pass

    is_connected = getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)
    if live_paper and not is_connected:
        logger.warning("Broker is not connected in live paper mode — skipping NEW entries this cycle")
        allow_new_entries = False
        broker_fetch_ok = False

    equity: float = float(getattr(config, "PDT_MIN_EQUITY", 25000.0))
    if bridge and is_connected:
        try:
            if hasattr(bridge, "ib") and hasattr(bridge.ib, "positions"):
                cached_open_positions = bridge.ib.positions()
            elif hasattr(bridge, "get_open_positions"):
                cached_open_positions = bridge.get_open_positions()
            else:
                cached_open_positions = []
            try:
                from strategies.validation_lane import get_production_risk_equity
                equity, broker_pnl = get_production_risk_equity(bridge, include_validation=False)
            except Exception as eq_err:
                logger.debug("get_production_risk_equity failed: %s", eq_err)
                try:
                    equity = float(bridge.get_net_liquidation())
                except Exception:
                    equity = float(getattr(config, "PDT_MIN_EQUITY", 25000.0))
                broker_pnl = today_pnl()

            if hasattr(bridge, "sync_today_trades_to_journal"):
                try:
                    bridge.sync_today_trades_to_journal()
                    from strategies.trade_journal import auto_verify_first_trades
                    auto_verify_first_trades()
                except Exception as sync_err:
                    logger.warning("sync_today_trades_to_journal error: %s", sync_err)
            broker_state_available = True
        except Exception as e:
            logger.error(f"Failed to fetch Alpaca portfolio state: {e}")
            if live_paper:
                logger.warning("Could not fetch portfolio state in live paper mode — skipping NEW entries this cycle")
                allow_new_entries = False
                broker_fetch_ok = False

    current_pnl = broker_pnl if live_paper else today_pnl()
    gate = getattr(bridge, "entry_gate", None)
    if (gate is not None and getattr(gate, "halted", False) is True) or getattr(bridge, "_daytrade_suspended", False) is True:
        logger.warning("Entry gate is HALTED or daytrade is suspended — skipping NEW entries this cycle")
        allow_new_entries = False

    if (
        live_paper
        and broker_state_available
        and getattr(config, "FLATTEN_ON_DAILY_LOSS", False)
        and current_pnl is not None
    ):
        try:
            dollar_limit = float(getattr(config, "MAX_DAILY_LOSS_DOLLARS", 300.0))
        except (ValueError, TypeError):
            dollar_limit = 300.0

        try:
            if isinstance(equity, (int, float)) or (hasattr(equity, "__float__") and not hasattr(equity, "mock_calls")):
                pct_val = getattr(config, "MAX_DAILY_LOSS_PCT", 3.0)
                pct_limit = float(equity) * (float(pct_val) / 100.0)
            else:
                pct_limit = dollar_limit
        except (ValueError, TypeError):
            pct_limit = dollar_limit

        governing_limit = min(dollar_limit, pct_limit)
        anchor_eq = getattr(bridge, "_start_of_day_equity", equity)

        from strategies.validation_lane import get_validation_daily_pnl
        val_daily_pnl = get_validation_daily_pnl()

        logger.info(
            "[EVALUATION PROVENANCE] live_equity=%.2f start_of_day_anchor=%.2f formula=(live_equity - start_of_day_anchor) "
            "calendar_day_pnl=%+.2f governing_limit=-%.2f budget_remaining=%.2f | "
            "[VALIDATION ISOLATION] val_daily_pnl=%+.2f production_pnl=%+.2f production_equity=%.2f",
            float(equity), float(anchor_eq),
            float(current_pnl), float(governing_limit), float(governing_limit + current_pnl),
            float(val_daily_pnl), float(current_pnl), float(equity)
        )

        try:
            if isinstance(current_pnl, (int, float)) or (hasattr(current_pnl, "__float__") and not hasattr(current_pnl, "mock_calls")):
                is_breached = float(current_pnl) <= -governing_limit
            else:
                is_breached = False
        except (ValueError, TypeError):
            is_breached = False

        if is_breached and getattr(bridge, "_daytrade_suspended", False) is not True:
            from strategies.session import now_eastern
            from strategies.halt_manager import halt_manager, LATCHING_HALT_DAILY_LOSS_BREACH
            et_date = str(now_eastern().date())
            bridge._daytrade_suspended = True
            halt_manager.add(LATCHING_HALT_DAILY_LOSS_BREACH)
            bridge._save_daytrade_risk_state(et_date, getattr(bridge, "_start_of_day_equity", equity), True)

            msg = f"🚨 **[CIRCUIT BREAKER] Daily loss limit breached!** 🚨\n**PnL**: ${current_pnl:.2f}\n**Governing Limit**: -${governing_limit:.2f}\nTriggering emergency flatten and suspending all new daytrade entries!"
            logger.warning(msg)
            try:
                from strategies.webhook import send_discord_alert
                send_discord_alert(msg)
            except Exception as w_exc:
                logger.error("Failed to send daily loss breach discord alert: %s", w_exc)

            try:
                bridge.flatten_all()
            except Exception as exc:
                logger.error("Flatten on daily loss failed: %s", exc)

        if getattr(bridge, "_daytrade_suspended", False) is True or not halt_manager.entries_allowed():
            allow_new_entries = False

    regime_res = None
    # Phase 4.5: Pre-Market Health Gate & Market Regime Check
    try:
        from strategies.halt_manager import (
            halt_manager,
            TRANSIENT_HALT_DATA_DEGRADED,
            TRANSIENT_HALT_GLOBAL_OPENING_BUFFER,
        )
        from strategies.regime_engine import evaluate_market_regime, MODE_RISK_OFF, MODE_CAUTION, MODE_DATA_DEGRADED
        from strategies.session import is_past_opening_buffer

        scanner_count = getattr(bridge, "_last_scanner_candidate_count", None) if bridge else None
        if not isinstance(scanner_count, (int, float)):
            scanner_count = None
        if scanner_count is None or scanner_count == 0:
            scanner_count = len(watchlist) if watchlist else 0
        if not live_paper and scanner_count > 0 and scanner_count < 2:
            scanner_count = 5  # Unit test mock default
        raw_regime_res = evaluate_market_regime(bridge=bridge, scanner_candidate_count=scanner_count, symbols=watchlist)
        regime_res = _regime_state_tracker.update(raw_regime_res)

        # Synchronize transient DATA_DEGRADED halt
        if regime_res.mode == MODE_DATA_DEGRADED:
            halt_manager.add(TRANSIENT_HALT_DATA_DEGRADED)
        else:
            halt_manager.clear_transient(TRANSIENT_HALT_DATA_DEGRADED)

        # Synchronize transient opening buffer
        enforce_buffer = getattr(config, "ENFORCE_GLOBAL_OPENING_BUFFER", True)
        if enforce_buffer and not is_past_opening_buffer(15):
            halt_manager.add(TRANSIENT_HALT_GLOBAL_OPENING_BUFFER)
        else:
            halt_manager.clear_transient(TRANSIENT_HALT_GLOBAL_OPENING_BUFFER)

        # Fail-closed gating: check halt_manager and regime boolean invariants
        if not halt_manager.entries_allowed() or regime_res.halt_new_entries or not regime_res.allow_new_longs:
            allow_new_entries = False
            if not halt_manager.entries_allowed():
                logger.warning("[ENTRY GATE HALT] %s", halt_manager.summary())
            elif regime_res.mode == MODE_DATA_DEGRADED:
                logger.critical("[DATA DEGRADED HALT] %s", regime_res.summary())
            else:
                mode_str = getattr(regime_res, "mode", "UNKNOWN")
                logger.warning("[MARKET ENTRY HALT] New long day trades disabled (mode=%s score=%d). Summary: %s", mode_str, regime_res.score, regime_res.summary())
                inverse_exempt = getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
                if inverse_exempt and regime_res is not None:
                    logger.info("[INVERSE ETF EXEMPT] Entry halt active; eligible inverse ETFs remain enabled (mode=%s score=%d)", mode_str, regime_res.score)
        elif regime_res.mode == MODE_CAUTION:
            logger.info("[MARKET REGIME CAUTION] Fallback watchlist active after 09:45 EST buffer (50%% sizing). Summary: %s", regime_res.summary())
    except Exception as r_exc:
        logger.critical("[REGIME FAILURE] Evaluation error -- halting all new entries (Fail-Closed). Error: %s", r_exc, exc_info=True)
        from strategies.halt_manager import halt_manager, TRANSIENT_HALT_DATA_DEGRADED
        from strategies.regime_engine import MarketRegimeResult, MODE_DATA_DEGRADED
        halt_manager.add(TRANSIENT_HALT_DATA_DEGRADED)
        regime_res = MarketRegimeResult(
            mode=MODE_DATA_DEGRADED,
            score=-99,
            allow_new_longs=False,
            allow_new_shorts=False,
            halt_new_entries=True,
            requires_opening_buffer=True,
            position_scale=0.0,
            confidence_boost=0.0,
            reasons=[f"Fallback regime on evaluation exception: {r_exc}"]
        )
        allow_new_entries = False

    # Event Guard Layer (Macro & High-Impact Scheduled Events)
    event_decision = None
    try:
        from strategies.event_guard import evaluate_event
        from strategies.session import now_eastern
        event_decision = evaluate_event(now_eastern())
    except Exception as ev_exc:
        logger.error("[EVENT GUARD] Evaluation error, failing CLOSED: %s", ev_exc)
        from strategies.event_guard import EventDecision
        event_decision = EventDecision(throttle=0.0, block_entries=True, atr_floor_mult=2.0, tag="EVAL_EXCEPTION_BLACKOUT")

    halt_manager.apply_event_blackout(event_decision.block_entries, tag=event_decision.tag)

    if event_decision.block_entries:
        allow_new_entries = False
        logger.warning("[EVENT GUARD BLACKOUT] %s: New entries hard blocked this cycle", event_decision.tag)
    elif event_decision.tag:
        logger.info(
            "[EVENT GUARD] %s active: sizing throttle=%.2fx, atr_floor=%.2fx",
            event_decision.tag, event_decision.throttle, event_decision.atr_floor_mult
        )


    try:
        from strategies.trailing_stop import manager
        if broker_state_available:
            active_symbols = {
                p.contract.symbol for p in cached_open_positions if p.position != 0
            }
            for symbol in list(manager.states):
                if symbol not in active_symbols:
                    manager.reset(symbol)

            # Fetch open orders once
            try:
                open_orders = bridge.ib.openTrades()
            except Exception as exc:
                logger.error(f"Failed to fetch open trades: {exc}")
                open_orders = []
                if live_paper:
                    logger.warning("Could not fetch open orders in live paper mode — skipping NEW entries this cycle")
                    allow_new_entries = False
                    broker_fetch_ok = False

            for p in cached_open_positions:
                if p.position != 0:
                    sym = p.contract.symbol
                    price = bridge.market_price(sym)
                    if price:
                        side = "BUY" if p.position > 0 else "SELL"
                        state = manager.ensure_initialized(
                            symbol=sym,
                            side=side,
                            avg_cost=p.avgCost,
                            open_orders=open_orders,
                            current_price=price,
                            qty=p.position,
                            bridge=bridge,
                        )
                        if state and state.active and getattr(state, "entry_time", 0) > 0:
                            from strategies.trailing_stop import check_time_based_exit
                            time_exit, time_reason = check_time_based_exit(
                                symbol=sym,
                                side=side,
                                entry_price=state.entry_price,
                                current_price=price,
                                entry_timestamp=state.entry_time,
                                peak_price=getattr(state, "peak_price", None),
                            )
                            if time_exit:
                                logger.warning(f"[TIME EXIT TRIGGERED] symbol={sym} side={side} reason={time_reason}")
                                if live_paper:
                                    closed = bridge.close_position(sym)
                                    if closed:
                                        logger.info(f"[TIME EXIT EXECUTED] symbol={sym} broker close confirmed. Resetting trailing stop.")
                                        manager.soft_reset(sym)
                                    else:
                                        logger.critical(f"[TIME EXIT FAILED] {sym} remains active because broker close was not confirmed")
                                else:
                                    logger.info(f"[TIME EXIT EXECUTED DRY RUN] symbol={sym}. Resetting trailing stop.")
                                    manager.soft_reset(sym)
                                continue

                        if state and state.active and state.order_id is None:
                            manager.handle_naked_position(
                                symbol=sym,
                                position_qty=p.position,
                                bridge=bridge,
                                dry_run=not live_paper
                            )
                        else:
                            sym_atr = None
                            try:
                                from strategies.intraday_data import get_position_atr_if_fresh, get_intraday_features
                                sym_atr = get_position_atr_if_fresh(sym, interval="5m", max_age_seconds=float(getattr(config, "POSITION_ATR_MAX_AGE_SECONDS", 360.0)))
                                if sym_atr is None:
                                    df_feat = get_intraday_features(sym, interval="5m")
                                    if df_feat is not None and not df_feat.empty and "atr" in df_feat.columns:
                                        sym_atr = float(df_feat["atr"].iloc[-1])
                            except Exception:
                                pass
                            manager.update_stop(sym, price, bridge=bridge, dry_run=not live_paper, current_atr=sym_atr)
    except Exception as e:
        logger.error(f"Failed to update trailing stops: {e}")

    try:
        from strategies.watchdog import touch_orchestrator_heartbeat
        touch_orchestrator_heartbeat(
            stage="risk_and_stops_completed",
            metadata={
                "open_positions": len(cached_open_positions),
                "broker_pnl": broker_pnl,
                "allow_new_entries": allow_new_entries,
            },
        )
    except Exception:
        pass


    if live_paper:
        if not broker_fetch_ok:
            _consecutive_broker_failures += 1
            limit = getattr(config, "CONSECUTIVE_OUTAGE_LIMIT", 3)
            if _consecutive_broker_failures == limit:
                msg = f"🚨 [OUTAGE ALERT] Daytrading bot has encountered {limit} consecutive broker communication failures/outages. New entries are blocked."
                logger.critical(msg)
                try:
                    from strategies.webhook import send_discord_alert
                    send_discord_alert(msg)
                except Exception as exc:
                    logger.warning("Failed to send outage alert: %s", exc)
        else:
            _consecutive_broker_failures = 0

    logger.debug("[HOOK] before fetch & evaluate_symbols_parallel")

    # Phase 5: 5-Minute Bar Boundary Cadence Gate
    now_utc = datetime.now(timezone.utc)
    # Determine current open 5m bar boundary timestamp
    current_5m_bar_ts = now_utc.replace(second=0, microsecond=0)
    current_5m_bar_ts = current_5m_bar_ts.replace(minute=(current_5m_bar_ts.minute // 5) * 5)

    # Calculate last completed 5m bar boundary (bar_start -> bar_end)
    from datetime import timedelta
    completed_5m_bar_ts = current_5m_bar_ts - timedelta(minutes=5)
    completed_5m_bar_str = completed_5m_bar_ts.isoformat()

    global _last_evaluated_5m_bar_str, _skipped_bar_ids
    if "_last_evaluated_5m_bar_str" not in globals():
        _last_evaluated_5m_bar_str = None
    if "_skipped_bar_ids" not in globals():
        _skipped_bar_ids = set()

    inverse_exempt = getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
    has_inverse_in_watchlist = any(str(sym).upper().strip() in getattr(config, "INVERSE_ETF_SYMBOLS", set()) for sym in watchlist)
    can_evaluate_signals = allow_new_entries or (inverse_exempt and has_inverse_in_watchlist)

    should_evaluate_5m_bar = False
    grace_seconds = float(getattr(config, "DAYTRADE_STARTUP_CADENCE_GRACE_S", 60.0))
    pub_timeout = float(getattr(config, "DAYTRADE_BAR_PUBLICATION_TIMEOUT_S", 90.0))
    bar_age = (now_utc - current_5m_bar_ts).total_seconds()

    if can_evaluate_signals:
        if completed_5m_bar_str in _skipped_bar_ids:
            logger.warning(
                f"[LATE BAR SUPPRESSION] Bar {completed_5m_bar_str} was previously marked skipped/timed-out. "
                f"Rejecting late-arriving stale evaluation."
            )
            should_evaluate_5m_bar = False
        elif _last_evaluated_5m_bar_str is None and bar_age > grace_seconds:
            # Mid-bar startup/buffer sync: synchronize forward to avoid evaluating stale bars
            _last_evaluated_5m_bar_str = completed_5m_bar_str
            if bar_age > pub_timeout:
                _skipped_bar_ids.add(completed_5m_bar_str)
            next_boundary_ts = current_5m_bar_ts + timedelta(minutes=5)
            boot_elapsed = time.time() - getattr(sys.modules[__name__], "_engine_boot_ts", time.time())
            from strategies.session import is_past_opening_buffer
            if boot_elapsed > 180.0 and is_past_opening_buffer(15):
                sync_context = "Opening buffer cleared mid-bar"
            else:
                sync_context = "Engine booted mid-bar"
            logger.info(
                f"[STARTUP CADENCE SYNC] {sync_context} (elapsed: {bar_age:.1f}s > grace: {grace_seconds:.1f}s "
                f"since {current_5m_bar_ts.strftime('%H:%M:%S')} UTC close). "
                f"Synchronizing to next 5m bar boundary at {next_boundary_ts.strftime('%H:%M:%S')} UTC. "
                f"Skipping stale bar evaluation."
            )
        elif _last_evaluated_5m_bar_str != completed_5m_bar_str:
            if bar_age > pub_timeout:
                _skipped_bar_ids.add(completed_5m_bar_str)
                _last_evaluated_5m_bar_str = completed_5m_bar_str
                logger.warning(
                    f"[BAR PUBLICATION TIMEOUT] Bar {completed_5m_bar_str} exceeded publication timeout "
                    f"(elapsed: {bar_age:.1f}s > {pub_timeout:.1f}s). Marking skipped and suppressing late evaluation."
                )
                should_evaluate_5m_bar = False
            else:
                should_evaluate_5m_bar = True
                _last_evaluated_5m_bar_str = completed_5m_bar_str
                logger.info(
                    f"[5M BAR CADENCE] Completed 5m bar boundary reached "
                    f"[feature_bar_start: {completed_5m_bar_ts.strftime('%H:%M:%S')} UTC, "
                    f"feature_bar_end: {current_5m_bar_ts.strftime('%H:%M:%S')} UTC, "
                    f"is_closed=True, decision_ts: {now_utc.strftime('%H:%M:%S')} UTC]. "
                    f"Running strategy signal evaluation..."
                )
        else:
            logger.debug(
                f"[5M BAR CADENCE] Intra-bar poll tick ({now_utc.strftime('%H:%M:%S')} UTC). "
                f"Skipping strategy evaluation (last completed bar: {completed_5m_bar_str})."
            )
    else:
        logger.info(
            f"[POLLING CYCLE] Intra-bar health check ({len(watchlist)} symbols); "
            f"new entries disabled or waiting for 5m bar completion."
        )

    market_tex = getattr(regime_res, "market_texture", "TRENDING") if regime_res else "TRENDING"
    should_run_eval = bool(can_evaluate_signals and should_evaluate_5m_bar)

    # ── Phase A Execution Health Gate & Synchronous Reaper Sweep ──
    try:
        from strategies.order_manager import reap_expired_entry_orders
        reap_expired_entry_orders(
            bridge=bridge,
            current_closed_bar_start=completed_5m_bar_ts,
            worker_id="orchestrator_poll_sweep",
        )
    except Exception as sweep_err:
        logger.warning("[ORCHESTRATOR SWEEP] Pre-eval sweep error: %s", sweep_err)

    try:
        from strategies.execution_health import get_execution_health
        from strategies.intraday_data import get_intraday_feed_telemetry
        feed_tel = get_intraday_feed_telemetry()
        health_gate = get_execution_health(
            bridge=bridge,
            bar_age_seconds=bar_age,
            feed_telemetry=feed_tel,
        )
        if not health_gate.allows_new_entries:
            if not should_evaluate_5m_bar and health_gate.status.value == "stale":
                logger.debug(
                    "[EXECUTION HEALTH GATE] Intra-bar cadence (%s): %s (bar_age=%.1fs). Waiting for next 5m bar boundary.",
                    health_gate.status.value,
                    health_gate.summary,
                    health_gate.bar_age_seconds or 0.0,
                )
            else:
                logger.warning(
                    "[EXECUTION HEALTH GATE] %s (status=%s, bar_age=%.1fs, latency=%.1fms, stale_skips=%d). "
                    "New entries blocked for this cycle.",
                    health_gate.summary,
                    health_gate.status.value,
                    health_gate.bar_age_seconds or 0.0,
                    health_gate.broker_latency_ms or 0.0,
                    feed_tel.get("stale_bar_skip_count", 0),
                )
            allow_new_entries = False
            can_evaluate_signals = False
            # If degraded or stale, allow 5m bar evaluation to probe market feed and unlatch telemetry
            # once fresh bars arrive, while strictly disabling order entry (fail-closed execution).
            if should_evaluate_5m_bar and str(health_gate.status.value).lower() in ("degraded", "stale"):
                should_run_eval = True
            else:
                should_run_eval = False
        else:
            logger.debug("[EXECUTION HEALTH GATE] %s", health_gate.summary)
    except Exception as h_err:
        logger.error("[EXECUTION HEALTH GATE] Evaluation failed, failing closed: %s", h_err)
        allow_new_entries = False
        can_evaluate_signals = False
        should_run_eval = False

    import uuid
    cycle_id = uuid.uuid4().hex[:12]
    eval_result: Optional[EvaluationResult] = None

    if should_run_eval:
        raw_eval = evaluate_symbols_parallel(watchlist, bridge=bridge, market_texture=market_tex)
        if isinstance(raw_eval, EvaluationResult):
            eval_result = raw_eval
            all_signals = list(raw_eval.entry_signals)
        else:
            all_signals = list(raw_eval)
            eval_result = EvaluationResult(
                entry_signals=tuple(all_signals),
                entry_allowed=True,
                suppression_reason=None,
                coverage=1.0,
                required_coverage=0.95,
                received_symbols=tuple(watchlist),
                missing_symbols=(),
            )
        # Filter: if normal entries halted, keep inverse ETF signals only
        if not allow_new_entries and can_evaluate_signals:
            inverse_set = set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
            before_count = len(all_signals)
            all_signals = [s for s in all_signals if s.symbol.upper().strip() in inverse_set]
            if before_count != len(all_signals):
                logger.info(
                    "[INVERSE ETF FILTER] Kept %d/%d signals (inverse ETFs only, normal longs suppressed)",
                    len(all_signals), before_count
                )
    else:
        all_signals = []
        eval_result = EvaluationResult(
            entry_signals=(),
            entry_allowed=False,
            suppression_reason="CYCLE_EVALUATION_SKIPPED",
            coverage=0.0,
            required_coverage=0.95,
            received_symbols=(),
            missing_symbols=(),
        )
    logger.debug("[HOOK] after fetch & evaluate_symbols_parallel")

    # ── P1.6 ORB Shadow Cycle Rejection Summary Hook ──
    if getattr(config, "P1_6_ORB_SHADOW_ENABLED", False):
        try:
            from strategies.shadow.orb_confirmed import get_confirmed_orb_strategy
            get_confirmed_orb_strategy().log_cycle_summary()
        except Exception:
            pass

    # ── P1.7 MR Shadow Cycle Summary Hook ──
    if getattr(config, "EXPERIMENTAL_MR_SHADOW_ENABLED", False):
        try:
            _get_mr_shadow_runner().log_cycle_summary()
        except Exception:
            pass

    total_universe_candidates = len(watchlist) if watchlist else 0
    raw_screened_setups = len(all_signals)

    # ── Lane 5: Pre-Correlation / Pre-Sizing Texture Admission Filter ──
    texture_admitted_signals = []
    for sig in all_signals:
        admitted, is_experiment, cell_tag = is_strategy_admitted_for_texture(
            strategy=sig.strategy,
            market_texture=market_tex,
            bridge=bridge,
            signal=sig,
        )
        if admitted:
            if is_experiment:
                logger.info(
                    "[PAPER EXPERIMENT ADMISSION] %s (%s) admitted for paper research cell '%s' in texture '%s'",
                    sig.symbol, sig.strategy, cell_tag, market_tex
                )
                try:
                    from strategies.gate_ledger import log_gate_decision
                    log_gate_decision(
                        evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sig.symbol}_{int(time.time())}",
                        symbol=sig.symbol,
                        bar=getattr(sig, "bar_timestamp", "") or "",
                        strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                        stage="STAGE_REGIME",
                        decision="PAPER_EXPERIMENT_ADMIT",
                        reason_code="TEXTURE_RESTRICTION_OVERRIDDEN_FOR_PAPER_EXPERIMENT",
                        human_message=f"Paper experiment admission for cell {cell_tag}",
                        terminal_decision=False,
                        blocked_by=None,
                        raw_conf=getattr(sig, "confidence", 0.0),
                        final_conf=getattr(sig, "confidence", 0.0),
                        metadata={"paper_experiment_cell": cell_tag, "lane": "PAPER_EXPERIMENT"},
                    )
                except Exception:
                    pass
            texture_admitted_signals.append(sig)
        else:
            logger.info(
                "[TEXTURE ADMISSION REJECT] Skipped %s: strategy %s not allowed in market texture '%s'",
                sig.symbol, sig.strategy, market_tex
            )
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sig.symbol}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_REGIME",
                    decision="REJECT",
                    reason_code=VetoReason.REGIME_TEXTURE_FILTER,
                    human_message=f"Strategy {sig.strategy} not allowed in market texture '{market_tex}'",
                    terminal_decision=True,
                    blocked_by="STAGE_REGIME",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass

    texture_admitted_count = len(texture_admitted_signals)
    all_signals = texture_admitted_signals

    all_signals = apply_portfolio_correlation(all_signals, cached_open_positions, bridge=bridge)
    correlated_count = len(all_signals)

    # ── Signal Export Hook for ML Training ──
    if all_signals:
        try:
            export_candidate_signals(
                signals=all_signals,
                regime=regime_res,
                market_texture=market_tex,
                export_dir=str(Path(config.DATA_DIR) / "exports"),
            )
        except Exception as exp_err:
            logger.debug("Candidate signal export hook failed: %s", exp_err)

    def get_signal_sorting_key(sig):
        strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(sig.strategy)
        priority = getattr(strat_cfg, "priority", 10) if strat_cfg else 10
        if market_tex == "SIDEWAYS" and getattr(sig, "strategy", None) == StrategyName.MEAN_REVERSION:
            priority = 1
        return (priority, -sig.confidence)

    all_signals.sort(key=get_signal_sorting_key)

    # Prevent same-symbol multi-order entry within the same cycle by keeping only the highest priority/confidence signal,
    # skip any symbols with existing active positions or pending working orders at the broker,
    # and skip any signals past their strategy-specific entry cutoff time.
    deduped_signals = []
    seen_symbols = set()
    near_misses = []

    active_symbols = set()
    active_position_sides = {}
    pending_symbols = set()
    validation_routed_symbols = {}
    if broker_state_available:
        for p in cached_open_positions:
            qty = getattr(p, "position", 0.0) or 0.0
            if qty != 0:
                p_sym = getattr(getattr(p, "contract", None), "symbol", None)
                if p_sym:
                    clean_sym = str(p_sym).upper().strip()
                    active_symbols.add(clean_sym)
                    active_position_sides[clean_sym] = "BUY" if float(qty) > 0 else "SELL"
        pending_symbols = set()
        for o in open_orders:
            o_sym = getattr(o, "symbol", None)
            if not o_sym:
                contract = getattr(o, "contract", None)
                if contract:
                    o_sym = getattr(contract, "symbol", "")
            if o_sym:
                pending_symbols.add(str(o_sym).upper().strip())
            else:
                logger.error("Failed to extract symbol from order object: %s. Disabling new entries this cycle (fail-closed).", o)
                if live_paper:
                    allow_new_entries = False

    def _can_route_validation_candidate(sig, routed_map, active_sides):
        sym_u = sig.symbol.upper().strip()
        if sym_u in routed_map:
            return False, f"Duplicate validation candidate in cycle (already routed {routed_map[sym_u]})"
        if sym_u in active_sides:
            curr_side = active_sides[sym_u]
            sig_side = "BUY" if sig.side.upper().strip() in ("BUY", "LONG") else "SELL"
            if sig_side != curr_side:
                return False, f"Opposing validation candidate {sig_side} conflicts with active {curr_side} position"
        return True, ""

    for sig in all_signals:
        sym = sig.symbol.upper().strip()

        # Check strategy global enablement gate (Loss containment)
        strat_key = getattr(sig.strategy, "name", str(sig.strategy)).upper().replace("STRATEGYNAME.", "").strip()
        is_strat_enabled = True
        if strat_key in ("MOMENTUM_SCALP", "STRATEGY_MOMENTUM_SCALP"):
            is_strat_enabled = getattr(config, "MOMENTUM_SCALP_ENABLED", getattr(config, "STRATEGY_MOMENTUM_SCALP_ENABLED", True))
        elif strat_key in ("VWAP_BOUNCE", "STRATEGY_VWAP_BOUNCE"):
            is_strat_enabled = getattr(config, "VWAP_BOUNCE_ENABLED", getattr(config, "STRATEGY_VWAP_BOUNCE_ENABLED", True))
        elif strat_key in ("ORB", "STRATEGY_ORB"):
            is_strat_enabled = getattr(config, "STRATEGY_ORB_ENABLED", False)
        elif strat_key in ("GAP_AND_GO", "STRATEGY_GAP_AND_GO"):
            is_strat_enabled = getattr(config, "STRATEGY_GAP_AND_GO_ENABLED", True)
        elif strat_key in ("CANDLESTICK", "STRATEGY_CANDLESTICK"):
            is_strat_enabled = getattr(config, "STRATEGY_CANDLESTICK_ENABLED", True)
        elif strat_key in ("MEAN_REVERSION", "STRATEGY_MEAN_REVERSION"):
            is_strat_enabled = getattr(config, "STRATEGY_MEAN_REVERSION_ENABLED", True)

        if not is_strat_enabled:
            logger.info("Skipped %s: strategy %s is disabled by configuration (governance loss containment)", sig.symbol, sig.strategy)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_REGIME",
                    decision="REJECT",
                    reason_code=VetoReason.STRATEGY_DISABLED,
                    human_message=f"Strategy {sig.strategy} is disabled by configuration (governance loss containment)",
                    terminal_decision=True,
                    blocked_by="STAGE_REGIME",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        # Regime-aware strategy filtering
        admitted, is_experiment, cell_tag = is_strategy_admitted_for_texture(
            strategy=sig.strategy,
            market_texture=market_tex,
            bridge=bridge,
            signal=sig,
        )
        if not admitted:
            logger.info("Skipped %s: strategy %s not allowed in market texture '%s'", sig.symbol, sig.strategy, market_tex)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_REGIME",
                    decision="REJECT",
                    reason_code=VetoReason.TEXTURE_NOT_ALLOWED,
                    human_message=f"Strategy {sig.strategy} not allowed in market texture '{market_tex}'",
                    terminal_decision=True,
                    blocked_by="STAGE_REGIME",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        # Check strategy-specific time cutoff
        strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(sig.strategy)
        cutoff_time_obj = getattr(strat_cfg, "cutoff_time_obj", None) if strat_cfg else None
        if cutoff_time_obj:
            from strategies.session import now_eastern
            current_et = now_eastern().time()
            if current_et >= cutoff_time_obj:
                logger.info(
                    "Skipped %s: strategy %s is past entry cutoff time (%s)",
                    sig.symbol,
                    sig.strategy,
                    strat_cfg.entry_cutoff_time
                )
                try:
                    from strategies.gate_ledger import log_gate_decision
                    from strategies.veto_reasons import VetoReason
                    log_gate_decision(
                        evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                        symbol=sig.symbol,
                        bar=getattr(sig, "bar_timestamp", "") or "",
                        strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                        stage="STAGE_REGIME",
                        decision="REJECT",
                        reason_code=VetoReason.ENTRY_CUTOFF_TIME,
                        human_message=f"Past entry cutoff time {strat_cfg.entry_cutoff_time}",
                        terminal_decision=True,
                        blocked_by="STAGE_REGIME",
                        raw_conf=getattr(sig, "confidence", 0.0),
                        final_conf=getattr(sig, "confidence", 0.0),
                    )
                except Exception:
                    pass
                continue

        # ── Lunch Chop Filter (11:30 AM - 1:30 PM ET) ──
        if getattr(config, "LUNCH_CHOP_FILTER_ENABLED", False):
            from strategies.session import now_eastern
            from datetime import time as dt_time
            current_et = now_eastern().time()
            lunch_start = dt_time(11, 30)
            lunch_end = dt_time(13, 30)
            if lunch_start <= current_et < lunch_end:
                l_score, l_mult, l_allowed, l_reason = compute_lunch_opportunity_score(sig, event_decision)
                if l_allowed:
                    sig.lunch_size_mult = l_mult
                    logger.info(
                        "[LUNCH OPPORTUNITY ALLOWED] %s: Score=%.1f/100 (Sizing Multiplier: %.2fx) | %s",
                        sig.symbol, l_score, l_mult, l_reason
                    )
                else:
                    logger.info(
                        "Skipped %s: Entry blocked during Midday Lunch Chop Zone (Score: %.1f/100 < 60) | %s",
                        sig.symbol, l_score, l_reason
                    )
                    try:
                        from strategies.gate_ledger import log_gate_decision
                        from strategies.veto_reasons import VetoReason
                        log_gate_decision(
                            evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                            symbol=sig.symbol,
                            bar=getattr(sig, "bar_timestamp", "") or "",
                            strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                            stage="STAGE_LUNCH_CHOP",
                            decision="REJECT",
                            reason_code=VetoReason.LUNCH_CHOP_ZONE,
                            human_message=f"Lunch score {l_score:.1f} < 60 | {l_reason}",
                            terminal_decision=True,
                            blocked_by="STAGE_LUNCH_CHOP",
                            raw_conf=getattr(sig, "confidence", 0.0),
                            final_conf=getattr(sig, "confidence", 0.0),
                            metadata={"lunch_score": l_score, "lunch_reason": l_reason},
                        )
                    except Exception:
                        pass

                    if getattr(config, "VALIDATION_LANE_ENABLED", False):
                        can_val, val_arb_reason = _can_route_validation_candidate(sig, validation_routed_symbols, active_position_sides)
                        if not can_val:
                            logger.info("[VALIDATION ARBITRATION] Skipped %s: %s", sig.symbol, val_arb_reason)
                        else:
                            try:
                                from strategies.validation_lane import evaluate_and_route_validation_candidate
                                val_res = evaluate_and_route_validation_candidate(
                                    signal=sig,
                                    bridge=bridge,
                                    policy_stage="STAGE_LUNCH_CHOP",
                                    policy_reason_code="LUNCH_CHOP_ZONE",
                                    policy_reason_msg=f"Lunch score {l_score:.1f} < 60 | {l_reason}",
                                    regime_mode=getattr(regime_res, "mode", "NEUTRAL") if regime_res else "NEUTRAL",
                                    market_texture=market_tex if 'market_tex' in locals() else "UNKNOWN",
                                    fixed_shares=getattr(config, "VALIDATION_LANE_SHARES", 1),
                                    dry_run=not live_paper,
                                    regime_result=regime_res,
                                )
                                if val_res and val_res.get("status") in ("PLACED", "DRY_RUN"):
                                    validation_routed_symbols[sym] = sig.side.upper().strip()
                            except Exception as val_err:
                                logger.warning("[VALIDATION ROUTE ERROR] Failed to route %s: %s", sig.symbol, val_err)
                    continue


        if sym in active_symbols:
            logger.info("Skipped %s: active position already exists", sig.symbol)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_ACTIVE_POSITION",
                    decision="REJECT",
                    reason_code=VetoReason.DUPLICATE_ORDER,
                    human_message="Active position already exists for symbol",
                    terminal_decision=True,
                    blocked_by="STAGE_ACTIVE_POSITION",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue
        if sym in pending_symbols:
            logger.info("Skipped %s: working order already pending", sig.symbol)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_PENDING_ORDER",
                    decision="REJECT",
                    reason_code=VetoReason.DUPLICATE_ORDER,
                    human_message="Working order already pending for symbol",
                    terminal_decision=True,
                    blocked_by="STAGE_PENDING_ORDER",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        # Viability Gate: check strategy/regime confidence floor before deduplication
        # so unviable signals cannot claim the symbol and suppress viable candidates.
        is_inverse = sym in set(getattr(config, "INVERSE_ETF_SYMBOLS", set()))
        if is_inverse and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False):
            min_conf = float(getattr(config, "INVERSE_ETF_MIN_CONFIDENCE", 0.65))
        else:
            min_conf = _min_confidence_for(sig, regime_result=regime_res)
            if regime_res and getattr(regime_res, "confidence_boost", 0.0) > 0:
                min_conf += float(regime_res.confidence_boost)

        # Exact raw floating-point comparison without epsilon offset (P0-4)
        if sig.confidence < min_conf:
            reg_mode = getattr(regime_res, "mode", "NEUTRAL") if regime_res else "NEUTRAL"
            side_norm = sig.side.upper().strip()
            is_counter = (reg_mode == "RISK_ON" and side_norm in ("SELL", "SHORT")) or (reg_mode == "RISK_OFF" and side_norm in ("BUY", "LONG"))
            alignment = "COUNTER_TREND" if is_counter else "TREND_ALIGNED"
            strat_name = getattr(sig.strategy, "name", str(sig.strategy))
            meta = getattr(sig, "metadata", {}) or {}
            base_conf = meta.get("base_confidence", getattr(sig, "confidence", 0.0))
            vol_adj = meta.get("vol_adjustment", 1.0)
            rr_adj = meta.get("rr_adjustment", 1.0)
            reg_adj = meta.get("regime_adjustment", 1.0)
            vol_pct = meta.get("volatility_pct", 0.0)

            logger.info(
                "[SIGNAL DISPOSITION] Skipped %s: Confidence %.4f < %.4f (%s | side=%s | regime=%s | alignment=%s | base=%.3f vol_adj=%.3f rr_adj=%.3f reg_adj=%.3f vol_pct=%.2f%%)",
                sig.symbol, sig.confidence, min_conf, strat_name, side_norm, reg_mode, alignment,
                base_conf, vol_adj, rr_adj, reg_adj, (vol_pct * 100.0) if vol_pct else 0.0
            )

            diff = min_conf - sig.confidence
            if 0 < diff <= 0.005:
                near_misses.append({
                    "symbol": sig.symbol,
                    "confidence": sig.confidence,
                    "required": min_conf,
                    "diff": diff,
                })
                logger.info(
                    "[NEAR-MISS] %s: Missed gate by %.4f (confidence=%.4f, required=%.4f) [base=%.3f, vol_adj=%.3f, rr_adj=%.3f, reg_adj=%.3f]",
                    sig.symbol, diff, sig.confidence, min_conf, base_conf, vol_adj, rr_adj, reg_adj
                )

            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=strat_name,
                    stage="STAGE_CONFIDENCE",
                    decision="REJECT",
                    reason_code=VetoReason.CONFIDENCE_BELOW_FLOOR,
                    human_message=f"Confidence {sig.confidence:.4f} < {min_conf:.4f}",
                    terminal_decision=True,
                    blocked_by="STAGE_CONFIDENCE",
                    raw_conf=base_conf,
                    final_conf=sig.confidence,
                    required_conf=min_conf,
                    metadata={
                        "vol_adj": vol_adj,
                        "rr_adj": rr_adj,
                        "reg_adj": reg_adj,
                        "vol_pct": vol_pct,
                        "near_miss": (0 < diff <= 0.005),
                    },
                )
            except Exception:
                pass

            if getattr(config, "VALIDATION_LANE_ENABLED", False):
                can_val, val_arb_reason = _can_route_validation_candidate(sig, validation_routed_symbols, active_position_sides)
                if not can_val:
                    logger.info("[VALIDATION ARBITRATION] Skipped %s: %s", sig.symbol, val_arb_reason)
                else:
                    try:
                        from strategies.validation_lane import evaluate_and_route_validation_candidate
                        val_res = evaluate_and_route_validation_candidate(
                            signal=sig,
                            bridge=bridge,
                            policy_stage="STAGE_CONFIDENCE",
                            policy_reason_code="CONFIDENCE_BELOW_FLOOR",
                            policy_reason_msg=f"Confidence {sig.confidence:.4f} < {min_conf:.4f}",
                            regime_mode=reg_mode,
                            market_texture=market_tex if 'market_tex' in locals() else "UNKNOWN",
                            fixed_shares=getattr(config, "VALIDATION_LANE_SHARES", 1),
                            dry_run=not live_paper,
                            regime_result=regime_res,
                        )
                        if val_res and val_res.get("status") in ("PLACED", "DRY_RUN"):
                            validation_routed_symbols[sym] = sig.side.upper().strip()
                    except Exception as val_err:
                        logger.warning("[VALIDATION ROUTE ERROR] Failed to route %s: %s", sig.symbol, val_err)
            continue

        if sym in seen_symbols:
            logger.info("[SIGNAL DISPOSITION] Skipped %s: duplicate signal in this cycle (superseded by previous candidate)", sym)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_PENDING_ORDER",
                    decision="REJECT",
                    reason_code=VetoReason.DUPLICATE_ORDER,
                    human_message="Duplicate signal in cycle superseded by higher priority candidate",
                    terminal_decision=True,
                    blocked_by="STAGE_PENDING_ORDER",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        rsi_max = getattr(config, "RSI_MAX_BUY_ENTRY", 70.0)
        rsi_min = getattr(config, "RSI_MIN_SHORT_ENTRY", 30.0)
        sig_rsi = _extract_signal_rsi(sig)
        if sig.side.upper() in ("BUY", "LONG") and sig_rsi >= rsi_max:
            logger.info("Skipped %s: RSI %.1f >= %.1f (Overbought BUY entry filter)", sig.symbol, sig_rsi, rsi_max)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_RSI_FILTER",
                    decision="REJECT",
                    reason_code=VetoReason.RSI_OVERBOUGHT,
                    human_message=f"RSI {sig_rsi:.1f} >= {rsi_max:.1f} (Overbought)",
                    terminal_decision=True,
                    blocked_by="STAGE_RSI_FILTER",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue
        if sig.side.upper() in ("SELL", "SHORT") and sig_rsi <= rsi_min:
            logger.info("Skipped %s: RSI %.1f <= %.1f (Oversold SHORT entry filter)", sig.symbol, sig_rsi, rsi_min)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_RSI_FILTER",
                    decision="REJECT",
                    reason_code=VetoReason.RSI_OVERSOLD,
                    human_message=f"RSI {sig_rsi:.1f} <= {rsi_min:.1f} (Oversold)",
                    terminal_decision=True,
                    blocked_by="STAGE_RSI_FILTER",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue
        if _is_inverse_etf_signal(sig) and sig.side.upper() in ("SELL", "SHORT"):
            logger.info("Skipped %s: Inverse ETFs cannot be sold short (Long-only hedge)", sig.symbol)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_GEOMETRY",
                    decision="REJECT",
                    reason_code=VetoReason.INVALID_GEOMETRY,
                    human_message="Inverse ETFs cannot be sold short",
                    terminal_decision=True,
                    blocked_by="STAGE_GEOMETRY",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        geom_ok, geom_msg = validate_executable_geometry(sig)
        if not geom_ok:
            logger.info("[PRE-EXECUTION GEOMETRY VETO] %s: %s", sig.symbol, geom_msg)
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_GEOMETRY",
                    decision="REJECT",
                    reason_code=VetoReason.INVALID_GEOMETRY,
                    human_message=geom_msg,
                    terminal_decision=True,
                    blocked_by="STAGE_GEOMETRY",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        seen_symbols.add(sym)
        deduped_signals.append(sig)
    all_signals = deduped_signals

    if near_misses:
        mean_near_miss_gap = sum(m["diff"] for m in near_misses) / len(near_misses)
        logger.info(
            "[NEAR-MISS CYCLE SUMMARY] count=%d mean_gap=%.4f symbols=%s",
            len(near_misses),
            mean_near_miss_gap,
            [m["symbol"] for m in near_misses],
        )

    current_pnl = broker_pnl if live_paper else today_pnl()

    from strategies.error_handler import TradingErrorHandler
    if error_handler is None:
        error_handler = TradingErrorHandler(bridge=bridge)

    execution_results = []
    logger.debug("[HOOK] before execute")
    for sig in all_signals:
        entry_allowed_for_sig = allow_new_entries or _inverse_etf_entry_allowed(regime_res, sig)
        if not entry_allowed_for_sig:
            logger.info(
                "Skipped execution of signal %s: new entries disabled this cycle [allow_new_entries=%s, inverse_etf_allowed=%s, is_inverse=%s]",
                sig.symbol,
                allow_new_entries,
                _inverse_etf_entry_allowed(regime_res, sig),
                _is_inverse_etf_signal(sig),
            )
            try:
                from strategies.gate_ledger import log_gate_decision
                from strategies.veto_reasons import VetoReason
                log_gate_decision(
                    evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                    symbol=sig.symbol,
                    bar=getattr(sig, "bar_timestamp", "") or "",
                    strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                    stage="STAGE_REGIME",
                    decision="REJECT",
                    reason_code=VetoReason.REGIME_RISK_OFF,
                    human_message="New entries disabled this cycle by Regime / Halt Manager",
                    terminal_decision=True,
                    blocked_by="STAGE_REGIME",
                    raw_conf=getattr(sig, "confidence", 0.0),
                    final_conf=getattr(sig, "confidence", 0.0),
                )
            except Exception:
                pass
            continue

        effective_equity = equity
        if hasattr(sig, "lunch_size_mult") and getattr(sig, "lunch_size_mult", 1.0) < 1.0:
            l_mult = float(sig.lunch_size_mult)
            effective_equity = min(effective_equity, effective_equity * l_mult)
            logger.info(
                "[LUNCH SIZING] Applying lunch risk sizing %.2fx: effective equity $%.2f for %s",
                l_mult, effective_equity, sig.symbol
            )

        if event_decision and event_decision.throttle < 1.0:
            effective_equity = min(effective_equity, effective_equity * event_decision.throttle)
            logger.info(
                "[EVENT GUARD] Applying event throttle %.2fx: effective equity $%.2f for %s",
                event_decision.throttle, effective_equity, sig.symbol
            )

        if _is_inverse_etf_signal(sig) and not allow_new_entries:
            sym_upper = sig.symbol.upper().strip()
            leveraged_etfs = getattr(config, "LEVERAGED_INVERSE_ETF_SYMBOLS", set())
            if sym_upper in leveraged_etfs:
                max_pct = getattr(config, "LEVERAGED_INVERSE_ETF_MAX_SIZE_PCT", 2.0)
                etf_type = "3x Leveraged"
            else:
                max_pct = getattr(config, "INVERSE_ETF_MAX_SIZE_PCT", 5.0)
                etf_type = "1x Unleveraged"
            effective_equity = min(equity, equity * (max_pct / 100.0))
            logger.info(
                "[INVERSE ETF] Applying %s reduced sizing: %.1f%% of equity = $%.2f for %s",
                etf_type, max_pct, effective_equity, sig.symbol
            )

        # Build cycle-bound EntryReadiness context for execution boundary validation
        readiness = EntryReadiness(
            entry_allowed=bool(eval_result.entry_allowed and allow_new_entries) if eval_result else allow_new_entries,
            suppression_reason=(eval_result.suppression_reason if eval_result and not eval_result.entry_allowed else ("NEW_ENTRIES_HALTED" if not allow_new_entries else None)),
            cycle_id=cycle_id,
            bar_ts=completed_5m_bar_str,
            created_at=time.time(),
            expires_at=time.time() + 300.0,
        )

        try:
            result = execute_signal(
                signal=sig,
                bridge=bridge,
                equity=effective_equity,
                current_pnl=current_pnl,
                day_trades_last_5_days=day_trades_in_last_5_days(bridge=bridge),
                dry_run=not live_paper,
                regime_result=regime_res,
                entry_readiness=readiness,
            )
            execution_results.append(result)
            if result["status"] in {"PLACED", "DRY_RUN"}:
                error_handler.reset()
                try:
                    from strategies.gate_ledger import log_gate_decision
                    log_gate_decision(
                        evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                        symbol=sig.symbol,
                        bar=getattr(sig, "bar_timestamp", "") or "",
                        strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                        stage="STAGE_ORDER_SUBMIT",
                        decision="PASS",
                        reason_code="ORDER_SUBMITTED",
                        human_message=f"Order submitted status={result.get('status')}",
                        terminal_decision=True,
                        blocked_by=None,
                        raw_conf=getattr(sig, "confidence", 0.0),
                        metadata={
                            "order_status": result.get("status"),
                            "shares": result.get("shares"),
                            "parent_order_id": result.get("parent_order_id"),
                            "client_order_id": result.get("client_order_id"),
                            "stop_order_id": result.get("stop_order_id"),
                            "tp_order_id": result.get("tp_order_id"),
                        },
                    )
                except Exception:
                    pass
            else:
                try:
                    from strategies.gate_ledger import log_gate_decision
                    c_stage, c_reason = _classify_execution_rejection(result.get("reason", ""))
                    log_gate_decision(
                        evaluation_id=getattr(sig, "signal_id", "") or f"sig_{sym}_{int(time.time())}",
                        symbol=sig.symbol,
                        bar=getattr(sig, "bar_timestamp", "") or "",
                        strategy=getattr(sig.strategy, "name", str(sig.strategy)),
                        stage=c_stage,
                        decision="REJECT",
                        reason_code=c_reason,
                        human_message=result.get("reason", "Order execution rejected"),
                        terminal_decision=True,
                        blocked_by=c_stage,
                        raw_conf=getattr(sig, "confidence", 0.0),
                        final_conf=getattr(sig, "confidence", 0.0),
                        metadata={"shares": result.get("shares", 0), "order_result": result.get("reason")},
                    )
                except Exception:
                    pass
                if result["reason"].startswith("Order failed:"):
                    error_handler.record(RuntimeError(result["reason"]))
        except Exception as e:
            logger.exception(f"Error executing signal: {e}")
            print(f"Error executing signal: {e}")
            if error_handler:
                error_handler.record(e)

    # ── Lane 5: Funnel Telemetry v2.1 (Dual Denominators) ──
    placed_count = sum(1 for r in execution_results if r.get("status") in {"PLACED", "DRY_RUN"})
    legacy_placed_pct_screened = (placed_count / raw_screened_setups * 100.0) if raw_screened_setups > 0 else 0.0
    legacy_placed_pct_universe = (placed_count / total_universe_candidates * 100.0) if total_universe_candidates > 0 else 0.0
    norm_placed_pct_admitted = (placed_count / texture_admitted_count * 100.0) if texture_admitted_count > 0 else 0.0
    norm_corr_pct_admitted = (correlated_count / texture_admitted_count * 100.0) if texture_admitted_count > 0 else 0.0

    logger.info(
        "[FUNNEL TELEMETRY v2.1] raw_candidates_total=%d, universe_candidates=%d, screened_setups=%d, "
        "texture_admitted=%d, correlated=%d, placed=%d | "
        "legacy: placed/screened=%.1f%%, placed/universe=%.1f%% | "
        "normalized: placed/admitted=%.1f%%, correlated/admitted=%.1f%%",
        raw_screened_setups,
        total_universe_candidates,
        raw_screened_setups,
        texture_admitted_count,
        correlated_count,
        placed_count,
        legacy_placed_pct_screened,
        legacy_placed_pct_universe,
        norm_placed_pct_admitted,
        norm_corr_pct_admitted,
    )
    try:
        from strategies.order_manager import reap_expired_entry_orders
        reap_expired_entry_orders(
            bridge=bridge,
            current_closed_bar_start=completed_5m_bar_ts if 'completed_5m_bar_ts' in locals() else None,
            worker_id="orchestrator_post_eval_sweep",
        )
    except Exception as sweep_err:
        logger.warning("[ORCHESTRATOR SWEEP] Post-eval sweep error: %s", sweep_err)

    try:
        from strategies.watchdog import touch_orchestrator_heartbeat
        touch_orchestrator_heartbeat(
            stage="cycle_finished",
            metadata={
                "signals_executed": len(all_signals) if 'all_signals' in locals() else 0,
                "watchlist_count": len(watchlist),
            },
        )
    except Exception:
        pass
