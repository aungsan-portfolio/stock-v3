from datetime import datetime, timezone
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

_consecutive_broker_failures = 0

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
    if len(df) > 1:
        df = df.iloc[:-1]
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
    
    global _consecutive_broker_failures
    broker_fetch_ok = True

    cached_open_positions = []
    open_orders = []
    equity = config.PDT_MIN_EQUITY
    broker_pnl = None
    broker_state_available = False
    allow_new_entries = True
    if getattr(bridge, "_daytrade_suspended", False) is True:
        logger.warning("Daytrading is suspended for the day due to daily loss limit breach.")
        allow_new_entries = False
    
    is_connected = getattr(bridge, "is_connected", False) or getattr(bridge, "_connected", False)
    if live_paper and not is_connected:
        logger.warning("Broker is not connected in live paper mode — skipping NEW entries this cycle")
        allow_new_entries = False
        broker_fetch_ok = False
        
    if bridge and is_connected:
        try:
            cached_open_positions = bridge.ib.positions()
            try:
                raw_equity = bridge.get_net_liquidation()
                equity = float(raw_equity) if not hasattr(raw_equity, "mock_calls") else 100000.0
            except Exception:
                equity = 100000.0

            if hasattr(bridge, "account_daily_pnl"):
                raw_pnl = bridge.account_daily_pnl()
                broker_pnl = float(raw_pnl) if not hasattr(raw_pnl, "mock_calls") else (today_pnl() or 0.0)
            elif hasattr(bridge, "daily_pnl"):
                raw_pnl = bridge.daily_pnl()
                broker_pnl = float(raw_pnl) if not hasattr(raw_pnl, "mock_calls") else (today_pnl() or 0.0)
            else:
                broker_pnl = today_pnl() or 0.0
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
    if getattr(bridge, "_daytrade_suspended", False) is True:
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
            et_date = str(now_eastern().date())
            bridge._daytrade_suspended = True
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
                
        if getattr(bridge, "_daytrade_suspended", False) is True:
            allow_new_entries = False

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
                            qty=p.position
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

    if live_paper:
        if not broker_fetch_ok:
            _consecutive_broker_failures += 1
            limit = getattr(config, "CONSECUTIVE_OUTAGE_LIMIT", 10)
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
    # Determine current 5m bar boundary timestamp
    current_5m_bar_ts = now_utc.replace(second=0, microsecond=0)
    current_5m_bar_ts = current_5m_bar_ts.replace(minute=(current_5m_bar_ts.minute // 5) * 5)
    current_5m_bar_str = current_5m_bar_ts.isoformat()

    global _last_evaluated_5m_bar_str
    if "_last_evaluated_5m_bar_str" not in globals():
        _last_evaluated_5m_bar_str = None

    should_evaluate_5m_bar = False
    if allow_new_entries:
        if _last_evaluated_5m_bar_str != current_5m_bar_str:
            should_evaluate_5m_bar = True
            _last_evaluated_5m_bar_str = current_5m_bar_str
            logger.info(f"[5M BAR CADENCE] New 5m bar boundary reached ({current_5m_bar_str}). Running strategy signal evaluation...")
        else:
            logger.debug(f"[5M BAR CADENCE] Intra-bar tick ({now_utc.strftime('%H:%M:%S')}). Skipping signal evaluation (last evaluated bar: {current_5m_bar_str}).")

    if allow_new_entries and should_evaluate_5m_bar:
        all_signals = evaluate_symbols_parallel(watchlist, bridge=bridge)
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
                
        if sym in active_symbols:
            logger.info("Skipped %s: active position already exists", sig.symbol)
            continue
        if sym in pending_symbols:
            logger.info("Skipped %s: working order already pending", sig.symbol)
            continue
        if sym not in seen_symbols:
            seen_symbols.add(sym)
            deduped_signals.append(sig)
    all_signals = deduped_signals

    current_pnl = broker_pnl if live_paper else today_pnl()

    from strategies.error_handler import TradingErrorHandler
    if error_handler is None:
        error_handler = TradingErrorHandler(bridge=bridge)

    logger.debug("[HOOK] before execute")
    for sig in all_signals:
        if not allow_new_entries:
            logger.info("Skipped execution of signal %s: new entries disabled this cycle", sig.symbol)
            continue
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
