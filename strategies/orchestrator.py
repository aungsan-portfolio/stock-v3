import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
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

logger = logging.getLogger(__name__)

_ALL_STRATEGIES = [
    ORBStrategy(),
    VWAPBounceStrategy(),
    GapAndGoStrategy(),
    MomentumScalpStrategy(),
    CandlestickPatternStrategy(),
]

def evaluate_symbol(symbol: str, bridge: object = None) -> List[TradeSignal]:
    """Evaluate one symbol through all strategies."""
    from strategies.intraday_data import fetch_intraday
    df = fetch_intraday(symbol, bridge=bridge)
    if df.empty:
        logger.warning("No data for %s", symbol)
        return []
    return _evaluate_on_df(symbol, df)


def _evaluate_on_df(symbol: str, df) -> List[TradeSignal]:
    """Internal sync implementation for one symbol given its DataFrame."""
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
        logger.debug("Evaluated %s: No signals. (%s)", symbol, " | ".join(failures))
    return []


def evaluate_symbols_parallel(symbols: List[str], bridge: object = None) -> List[TradeSignal]:
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
            out.extend(_evaluate_on_df(sym, df))
        return out

    if len(symbol_data) < min_batch:
        out = []
        for sym, df in symbol_data.items():
            out.extend(_evaluate_on_df(sym, df))
        return out

    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_evaluate_on_df, sym, df): sym for sym, df in symbol_data.items()}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                out.extend(fut.result())
            except Exception as e:
                logger.exception("Parallel evaluation failed for %s", sym)
                fallback_serial = getattr(config, "DAYTRADE_PARALLEL_EVALUATION_FALLBACK_TO_SERIAL", True)
                if fallback_serial:
                    out.extend(_evaluate_on_df(sym, symbol_data[sym]))
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

    for s in signals:
        if s.confidence <= 0:
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
):
    """Evaluates strategies for symbols in the watchlist and executes trades."""
    if not prime_broker_state:
        logger.debug("prime_broker_state=False is deprecated and ignored")

    logger.info(f"Evaluating signals for {len(watchlist)} symbols...")
    print(f"Evaluating signals for {len(watchlist)} symbols...")
    
    cached_open_positions = []
    equity = config.PDT_MIN_EQUITY
    broker_pnl = None
    broker_state_available = False
    
    is_connected = getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)
    if bridge and is_connected:
        try:
            cached_open_positions = bridge.ib.positions()
            equity = bridge.get_net_liquidation()
            if hasattr(bridge, "account_daily_pnl"):
                broker_pnl = bridge.account_daily_pnl()
            elif hasattr(bridge, "daily_pnl"):
                broker_pnl = bridge.daily_pnl()
            else:
                broker_pnl = today_pnl()
            broker_state_available = True
        except Exception as e:
            logger.error(f"Failed to fetch Alpaca portfolio state: {e}")

    current_pnl = broker_pnl if live_paper else today_pnl()
    if (
        live_paper
        and broker_state_available
        and getattr(config, "FLATTEN_ON_DAILY_LOSS", False)
        and current_pnl is not None
    ):
        from strategies.intraday_risk import check_daily_loss
        if not check_daily_loss(current_pnl, equity):
            logger.warning("Daily loss limit reached at %.2f; flattening all positions", current_pnl)
            try:
                bridge.flatten_all()
            except Exception as exc:
                logger.error("Flatten on daily loss failed: %s", exc)
            return

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
            open_orders = bridge.ib.openTrades()
            
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
                            current_price=price
                        )
                        if state and state.active and state.order_id is None:
                            manager.handle_naked_position(
                                symbol=sym,
                                position_qty=p.position,
                                bridge=bridge,
                                dry_run=not live_paper
                            )
                        else:
                            manager.update_stop(sym, price, bridge=bridge, dry_run=not live_paper)
    except Exception as e:
        logger.error(f"Failed to update trailing stops: {e}")

    logger.debug("[HOOK] before fetch & evaluate_symbols_parallel")

    all_signals = evaluate_symbols_parallel(watchlist, bridge=bridge)
    logger.debug("[HOOK] after fetch & evaluate_symbols_parallel")

    all_signals = apply_portfolio_correlation(all_signals, cached_open_positions, bridge=bridge)

    all_signals.sort(key=lambda s: s.confidence, reverse=True)

    current_pnl = broker_pnl if live_paper else today_pnl()

    from strategies.error_handler import TradingErrorHandler
    if error_handler is None:
        error_handler = TradingErrorHandler(bridge=bridge)

    logger.debug("[HOOK] before execute")
    for sig in all_signals:
        try:
            result = execute_signal(
                signal=sig,
                bridge=bridge,
                equity=equity,
                current_pnl=current_pnl,
                day_trades_last_5_days=day_trades_in_last_5_days(),
                dry_run=not live_paper,
            )
            if result["status"] in {"PLACED", "DRY_RUN"}:
                error_handler.reset()
            elif result["reason"].startswith("Order failed:"):
                error_handler.record(RuntimeError(result["reason"]))
        except Exception as e:
            logger.exception(f"Error executing signal: {e}")
            print(f"Error executing signal: {e}")
            error_handler.record(e)
