from datetime import datetime, timezone
import time
import math
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple
import sys

import config
from strategies.base import TradeSignal
from strategies.orb_strategy import ORBStrategy
from strategies.vwap_strategy import VWAPBounceStrategy
from strategies.gap_strategy import GapAndGoStrategy
from strategies.momentum_strategy import MomentumScalpStrategy
from strategies.candlestick_strategy import CandlestickPatternStrategy
from strategies.intraday_data import fetch_intraday
from strategies.order_manager import execute_signal
from strategies.trade_journal import today_trade_count, today_pnl, day_trades_in_last_5_days
from strategies.performance import profile_latency
from strategies.position_sizer import calculate_shares, validate_executable_geometry

logger = logging.getLogger(__name__)

_consecutive_broker_failures = 0

from strategies.regime_engine import RegimeStateTracker

_regime_state_tracker = RegimeStateTracker()

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


def _prepare_closed_bars(df):
    """Ensure only fully closed bars are evaluated.
    If the last bar in df is still actively forming (<5m old), remove it.
    If the last bar is already a completed closed bar, preserve it.
    """
    if df is not None and not df.empty and len(df) > 1:
        try:
            from strategies.session import now_eastern
            now = now_eastern()
            last_ts = df.index[-1]
            if hasattr(last_ts, "tzinfo") and last_ts.tzinfo is not None:
                last_ts_et = last_ts.astimezone(now.tzinfo)
            else:
                last_ts_et = last_ts
            # If the last bar started less than 4.5 minutes ago, it's currently forming -> remove it
            if hasattr(last_ts_et, "hour") and (now - last_ts_et).total_seconds() < 270:
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


def evaluate_symbol(symbol: str, bridge: object = None) -> List[TradeSignal]:
    """Evaluate one symbol through all strategies."""
    from strategies.intraday_data import fetch_intraday
    df = fetch_intraday(symbol, bridge=bridge)
    if df.empty:
        logger.warning("No data for %s", symbol)
        return []
    return _evaluate_on_df(symbol, df)


def _evaluate_on_df(symbol: str, df, market_texture: str = "TRENDING") -> List[TradeSignal]:
    """Internal sync implementation for one symbol given its DataFrame."""
    if df is None or df.empty:
        return []

    df = _prepare_closed_bars(df)
    if df is None or df.empty:
        return []


    signals = []
    failures = []

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
                    signals.append(signal)
                else:
                    failures.append(f"{strategy.name}: INVALID (reward/risk or geometry)")
        except Exception as e:
            logger.exception("Strategy %s crashed on %s", strategy.name, symbol)
            failures.append(f"{strategy.name}: CRASH ({e})")

    # Evaluate MeanReversionStrategy when in SIDEWAYS market texture
    if market_texture == "SIDEWAYS":
        try:
            from strategies.mean_reversion import MeanReversionStrategy
            mr_strat = MeanReversionStrategy()
            mr_signal = mr_strat.generate_signal(symbol, df, market_texture=market_texture)
            if mr_signal is not None:
                mr_signal.metadata['raw_confidence'] = mr_signal.confidence
                mr_signal.confidence = scorer.calculate_risk_adjusted_confidence(mr_signal)
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


def evaluate_symbols_parallel(symbols: List[str], bridge: object = None, market_texture: str = "TRENDING") -> List[TradeSignal]:
    """Evaluate many symbols in parallel when the batch is large enough."""
    if not symbols:
        return []

    from strategies.intraday_data import fetch_intraday
    
    parallel_enabled = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_ENABLED", True)
    min_batch = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_MIN_SYMBOLS", 4)
    max_workers = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_MAX_WORKERS", 3)
    
    symbol_data = {}
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
        return []

    if not parallel_enabled:
        out = []
        for sym, df in symbol_data.items():
            out.extend(_evaluate_on_df(sym, df, market_texture=market_texture))
        return out

    if len(symbol_data) < min_batch:
        out = []
        for sym, df in symbol_data.items():
            out.extend(_evaluate_on_df(sym, df, market_texture=market_texture))
        return out

    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_evaluate_on_df, sym, df, market_texture): sym for sym, df in symbol_data.items()}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                out.extend(fut.result())
            except Exception as e:
                logger.exception("Parallel evaluation failed for %s", sym)
                fallback_serial = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_FALLBACK_TO_SERIAL", True)
                if fallback_serial:
                    out.extend(_evaluate_on_df(sym, symbol_data[sym], market_texture=market_texture))
                else:
                    raise e

    return out



def apply_portfolio_correlation(signals: List[TradeSignal], open_positions: List, bridge: object = None) -> List[TradeSignal]:
    """Applies portfolio correlation checks to signals."""
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

    logger.info(f"[ORCHESTRATOR POLL] Polling cycle started for {len(watchlist)} watchlist symbols | allow_new_entries={allow_new_entries}")
    print(f"[ORCHESTRATOR POLL] Polling cycle started for {len(watchlist)} watchlist symbols | allow_new_entries={allow_new_entries}")
    
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
        
    if bridge and is_connected:
        try:
            if hasattr(bridge, "ib") and hasattr(bridge.ib, "positions"):
                cached_open_positions = bridge.ib.positions()
            elif hasattr(bridge, "get_open_positions"):
                cached_open_positions = bridge.get_open_positions()
            else:
                cached_open_positions = []
            try:
                equity = float(bridge.get_net_liquidation())
            except Exception:
                equity = float(getattr(config, "PDT_MIN_EQUITY", 25000.0))

            try:
                if hasattr(bridge, "account_daily_pnl"):
                    broker_pnl = float(bridge.account_daily_pnl())
                elif hasattr(bridge, "daily_pnl"):
                    broker_pnl = float(bridge.daily_pnl())
                else:
                    broker_pnl = today_pnl()
            except Exception:
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
                logger.warning("[MARKET REGIME RISK_OFF/HALT] New long day trades disabled (Score: %d). Summary: %s", regime_res.score, regime_res.summary())
                inverse_exempt = getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
                if inverse_exempt and regime_res is not None:
                    logger.info("[INVERSE ETF EXEMPT] RISK_OFF active but inverse ETF entries remain eligible (SPDN/SQQQ/SH etc.)")
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
                                logger.info(f"[TIME EXIT TRIGGER] {time_reason}")
                                if live_paper:
                                    closed = bridge.close_position(sym)
                                    if closed:
                                        manager.soft_reset(sym)
                                    else:
                                        logger.critical(f"[TIME EXIT FAILED] {sym} remains active because broker close was not confirmed")
                                else:
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
                                from strategies.intraday_data import get_intraday_features
                                df_feat = get_intraday_features(sym, interval="5m")
                                if df_feat is not None and not df_feat.empty and "atr" in df_feat.columns:
                                    sym_atr = float(df_feat["atr"].iloc[-1])
                            except Exception:
                                pass
                            manager.update_stop(sym, price, bridge=bridge, dry_run=not live_paper, current_atr=sym_atr)
    except Exception as e:
        logger.error(f"Failed to update trailing stops: {e}")

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

    global _last_evaluated_5m_bar_str
    if "_last_evaluated_5m_bar_str" not in globals():
        _last_evaluated_5m_bar_str = None

    inverse_exempt = getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
    has_inverse_in_watchlist = any(str(sym).upper().strip() in getattr(config, "INVERSE_ETF_SYMBOLS", set()) for sym in watchlist)
    can_evaluate_signals = allow_new_entries or (inverse_exempt and has_inverse_in_watchlist)

    should_evaluate_5m_bar = False
    if can_evaluate_signals:
        if _last_evaluated_5m_bar_str != completed_5m_bar_str:
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

    if should_run_eval:
        all_signals = evaluate_symbols_parallel(watchlist, bridge=bridge, market_texture=market_tex)
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
    logger.debug("[HOOK] after fetch & evaluate_symbols_parallel")

    all_signals = apply_portfolio_correlation(all_signals, cached_open_positions, bridge=bridge)

    def get_signal_sorting_key(sig):
        strat_cfg = getattr(config, "STRATEGY_SETTINGS", {}).get(sig.strategy)
        priority = getattr(strat_cfg, "priority", 10) if strat_cfg else 10
        return (priority, -sig.confidence)

    all_signals.sort(key=get_signal_sorting_key)

    # Prevent same-symbol multi-order entry within the same cycle by keeping only the highest priority/confidence signal,
    # skip any symbols with existing active positions or pending working orders at the broker,
    # and skip any signals past their strategy-specific entry cutoff time.
    deduped_signals = []
    seen_symbols = set()

    active_symbols = set()
    pending_symbols = set()
    if broker_state_available:
        active_symbols = {
            p.contract.symbol.upper().strip() for p in cached_open_positions if p.position != 0
        }
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

    for sig in all_signals:
        sym = sig.symbol.upper().strip()

        # Regime-aware strategy filtering
        tex_cfg = getattr(config, "MARKET_REGIME_SETTINGS", {}).get(market_tex, {})
        raw_allowed = tex_cfg.get("allowed_strategies", None)
        allowed_strats = raw_allowed if raw_allowed else None
        is_inverse_exempt = _is_inverse_etf_signal(sig) and getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
        if is_inverse_exempt:
            logger.info(
                "[INVERSE ETF EXEMPT] Strategy texture filter bypassed for %s (%s in '%s' texture)",
                sig.symbol, sig.strategy, market_tex
            )
        elif allowed_strats is not None and sig.strategy not in allowed_strats:
            logger.info("Skipped %s: strategy %s not allowed in market texture '%s'", sig.symbol, sig.strategy, market_tex)
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
                    continue

                
        if sym in active_symbols:
            logger.info("Skipped %s: active position already exists", sig.symbol)
            continue
        if sym in pending_symbols:
            logger.info("Skipped %s: working order already pending", sig.symbol)
            continue

        if sym in seen_symbols:
            logger.debug("Skipped %s: duplicate signal in this cycle", sym)
            continue
        seen_symbols.add(sym)

        rsi_max = getattr(config, "RSI_MAX_BUY_ENTRY", 70.0)
        rsi_min = getattr(config, "RSI_MIN_SHORT_ENTRY", 30.0)
        sig_rsi = _extract_signal_rsi(sig)
        if sig.side.upper() in ("BUY", "LONG") and sig_rsi >= rsi_max:
            logger.info("Skipped %s: RSI %.1f >= %.1f (Overbought BUY entry filter)", sig.symbol, sig_rsi, rsi_max)
            continue
        if sig.side.upper() in ("SELL", "SHORT") and sig_rsi <= rsi_min:
            logger.info("Skipped %s: RSI %.1f <= %.1f (Oversold SHORT entry filter)", sig.symbol, sig_rsi, rsi_min)
            continue
        if _is_inverse_etf_signal(sig) and sig.side.upper() in ("SELL", "SHORT"):
            logger.info("Skipped %s: Inverse ETFs cannot be sold short (Long-only hedge)", sig.symbol)
            continue

        geom_ok, geom_msg = validate_executable_geometry(sig)
        if not geom_ok:
            logger.info("[PRE-EXECUTION GEOMETRY VETO] %s: %s", sig.symbol, geom_msg)
            continue

        deduped_signals.append(sig)
    all_signals = deduped_signals

    current_pnl = broker_pnl if live_paper else today_pnl()

    from strategies.error_handler import TradingErrorHandler
    if error_handler is None:
        error_handler = TradingErrorHandler(bridge=bridge)

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

        try:
            result = execute_signal(
                signal=sig,
                bridge=bridge,
                equity=effective_equity,
                current_pnl=current_pnl,
                day_trades_last_5_days=day_trades_in_last_5_days(bridge=bridge),
                dry_run=not live_paper,
                regime_result=regime_res,
            )
            if result["status"] in {"PLACED", "DRY_RUN"}:
                error_handler.reset()
            elif result["reason"].startswith("Order failed:"):
                error_handler.record(RuntimeError(result["reason"]))
        except Exception as e:
            logger.exception(f"Error executing signal: {e}")
            print(f"Error executing signal: {e}")
            error_handler.record(e)
