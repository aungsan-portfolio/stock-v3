import logging
import os
import math
import numpy as np
import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt

import config
import risk_math
from strategies.intraday_data import fetch_intraday_yfinance
from strategies.base import TradeSignal
from strategies.orb_strategy import ORBStrategy
from strategies.vwap_strategy import VWAPBounceStrategy
from strategies.gap_strategy import GapAndGoStrategy
from strategies.momentum_strategy import MomentumScalpStrategy
from strategies.candlestick_strategy import CandlestickPatternStrategy
from strategies.trade_journal import count_day_trades_in_records

logger = logging.getLogger(__name__)


def _intrabar_exit_price(side, bar_open, bar_high, bar_low, stop, target):
    """Return a conservative executable exit price for an OHLC bar."""
    if side == "BUY":
        if bar_low <= stop:
            return min(stop, bar_open)
        if bar_high >= target:
            return target
    else:
        if bar_high >= stop:
            return max(stop, bar_open)
        if bar_low <= target:
            return target
    return None


def _realized_pnl(side, entry_price, exit_price, shares, commission):
    """Return dollar PnL after applying a single per-trade commission."""
    if side == "BUY":
        per_share = exit_price - entry_price
    else:
        per_share = entry_price - exit_price
    return per_share * shares - commission


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Return dollar drawdown from the running peak, including the zero origin."""
    return equity - equity.cummax()


def _minimum_confidence_for(strategy) -> float:
    strategy_config = getattr(config, "STRATEGY_SETTINGS", {}).get(strategy.name)
    if strategy_config:
        return float(getattr(strategy_config, "confidence_min", 0.65))
    return 0.65


def run_backtest(symbol: str, strategy_name: str, lookback_days: int = 5, override_min_confidence: float = None, plot: bool = False) -> dict:
    logger.info(f"Running backtest for {symbol} using {strategy_name} over {lookback_days} days")
    
    # 1. Load Data
    df = fetch_intraday_yfinance(symbol, lookback_days=lookback_days)
    if df.empty:
        logger.error(f"No data fetched for {symbol}")
        return {
            "total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "wins": 0, "losses": 0,
            "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0
        }
        
    # ONE-TIME pre-compute indicators
    from strategies.indicators import add_all_indicators
    df = add_all_indicators(df)

    strategies = {
        "ORB": ORBStrategy(),
        "VWAP_BOUNCE": VWAPBounceStrategy(),
        "GAP_AND_GO": GapAndGoStrategy(),
        "MOMENTUM_SCALP": MomentumScalpStrategy(),
        "CANDLESTICK": CandlestickPatternStrategy()
    }
    
    strategy = strategies.get(strategy_name.upper())
    if not strategy:
        logger.error(f"Unknown strategy: {strategy_name}")
        return {
            "total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "wins": 0, "losses": 0,
            "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0
        }
        
    if override_min_confidence is not None:
        min_confidence = override_min_confidence
    else:
        min_confidence = _minimum_confidence_for(strategy)
        
    trades: List[dict] = []
    active_trade = None
    
    start_equity = 100_000.0
    current_equity = start_equity
    equity_curve = [{"time": df.index[0], "equity": start_equity}]
    
    slippage_pct = getattr(config, 'BACKTEST_SLIPPAGE_FRAC', getattr(config, 'BACKTEST_SLIPPAGE_PCT', 0.0005))
    commission = getattr(config, 'BACKTEST_COMMISSION_PER_TRADE', 1.00)
    
    last_date = None
    day_opening_equity = start_equity
    mock_executions = []
    trade_id_counter = 0
    
    # Step through data row by row
    for i in range(30, len(df)):
        current_df = df.iloc[:i]
        last_row = current_df.iloc[-1]
        current_open = last_row['open']
        current_high = last_row['high']
        current_low = last_row['low']
        current_time = current_df.index[-1]
        current_date = current_time.date()
        
        if last_date is None or current_date != last_date:
            day_opening_equity = current_equity
            last_date = current_date
            
        # 1. Check active trade exits using t-1 stop
        if active_trade:
            exit_price = _intrabar_exit_price(
                active_trade['side'],
                current_open,
                current_high,
                current_low,
                active_trade['stop'],
                active_trade['target'],
            )
            if exit_price is not None:
                active_trade['exit'] = exit_price
                if active_trade['side'] == 'BUY':
                    executed_exit = active_trade['exit'] * (1 - slippage_pct)
                else:
                    executed_exit = active_trade['exit'] * (1 + slippage_pct)
                active_trade['pnl'] = _realized_pnl(
                    active_trade['side'],
                    active_trade['entry'],
                    executed_exit,
                    active_trade['shares'],
                    commission,
                )
                active_trade['realized_pnl'] = active_trade['pnl']
                active_trade['exit_time'] = current_time
                trades.append(active_trade)
                current_equity += active_trade['pnl']
                
                # Append exit mock execution
                mock_executions.append({
                    "timestamp": current_time.isoformat(),
                    "event_type": "FILL",
                    "type": "FILL",
                    "execution_id": f"backtest-exit-{active_trade['id']}",
                    "qty": float(active_trade['shares']),
                    "side": "SELL" if active_trade['side'] == "BUY" else "BUY",
                    "symbol": symbol
                })
                
                active_trade = None
                equity_curve.append({'time': current_time, 'equity': current_equity})
                continue
                
            # If not exited, update Trailing Stop at end of bar
            bar_atr = last_row.get('atr')
            if bar_atr is not None and not np.isnan(bar_atr) and bar_atr > 0:
                use_atr = bar_atr
            else:
                fallback_pct = getattr(config, 'TRAILING_STOP_FALLBACK_PCT', 2.0) / 100.0
                use_atr = active_trade['entry'] * fallback_pct / 1.5
                
            math_side = "LONG" if active_trade['side'] == "BUY" else "SHORT"
            if math_side == "LONG":
                active_trade['peak'] = max(active_trade.get('peak', active_trade['entry']), current_high)
                stop_val = risk_math.trailing_stop_price(
                    current_price=current_high,
                    atr=use_atr,
                    atr_multiple=getattr(config, 'TRAILING_STOP_ATR_MULTIPLE', 1.5),
                    side=math_side,
                    highest_price=active_trade['peak']
                )
                active_trade['stop'] = max(active_trade['stop'], stop_val)
            else:
                active_trade['peak'] = min(active_trade.get('peak', active_trade['entry']), current_low)
                stop_val = risk_math.trailing_stop_price(
                    current_price=current_low,
                    atr=use_atr,
                    atr_multiple=getattr(config, 'TRAILING_STOP_ATR_MULTIPLE', 1.5),
                    side=math_side,
                    lowest_price=active_trade['peak']
                )
                active_trade['stop'] = min(active_trade['stop'], stop_val)
            
        # 2. No active trade, check risk limits & entries
        if not active_trade:
            # Check Daily Loss halt parity (Dollar & Percentage limits)
            max_daily_loss_dollars = getattr(config, "MAX_DAILY_LOSS_DOLLARS", 300.0)
            max_daily_loss_pct = getattr(config, "MAX_DAILY_LOSS_PCT", 3.0) / 100.0
            
            daily_loss_hit = False
            if max_daily_loss_dollars > 0 and (current_equity - day_opening_equity) <= -max_daily_loss_dollars:
                daily_loss_hit = True
            elif max_daily_loss_pct > 0 and day_opening_equity > 0:
                loss_pct = abs(current_equity - day_opening_equity) / day_opening_equity
                if (current_equity - day_opening_equity) < 0 and loss_pct >= max_daily_loss_pct:
                    daily_loss_hit = True
                    
            if daily_loss_hit:
                continue
                
            # Check Consecutive Losses rule (Chronologically sorted)
            sorted_trades = sorted(trades, key=lambda x: x['exit_time'])
            consecutive_losses = 0
            for t_rec in reversed(sorted_trades):
                if t_rec['exit_time'].date() != current_date:
                    break
                if t_rec.get('realized_pnl', 0.0) < 0:
                    consecutive_losses += 1
                else:
                    break
            if consecutive_losses >= getattr(config, "MAX_CONSECUTIVE_LOSSES", 3):
                continue
                
            # Check PDT rule
            day_trades_count = count_day_trades_in_records(mock_executions, current_date)
            if getattr(config, "PDT_ENABLED", True) and current_equity < getattr(config, "PDT_MIN_EQUITY", 25000.0):
                if day_trades_count >= getattr(config, "PDT_MAX_DAY_TRADES_5_DAYS", 3):
                    continue
                    
            # Check Strategy Entry Signal
            signal = strategy.evaluate(symbol, current_df)
            if signal and signal.confidence >= min_confidence:
                # Apply slippage to entry price
                entry_price = signal.entry_price
                entry_price = entry_price * (1 + slippage_pct) if signal.side == 'BUY' else entry_price * (1 - slippage_pct)
                
                # Apply Sizing Parity
                sizing_method = getattr(config, "SIZING_METHOD", "risk_based")
                if sizing_method == "risk_based":
                    risk_pct = getattr(config, "MAX_RISK_PER_TRADE_PCT", 1.0) / 100.0
                    max_trade_val = getattr(config, "MAX_TRADE_VALUE", None)
                    max_pos_pct = getattr(config, "MAX_POSITION_PCT", None)
                    math_side = "LONG" if signal.side == "BUY" else "SHORT"
                    shares = risk_math.shares_for_risk(
                        account_equity=current_equity,
                        risk_pct=risk_pct,
                        entry_price=entry_price,
                        stop_price=signal.stop_price,
                        side=math_side,
                        max_trade_value=max_trade_val,
                        max_position_pct=max_pos_pct
                    )
                elif sizing_method == "fixed_dollars":
                    dollar_amount = getattr(config, "FIXED_DOLLAR_AMOUNT", 1000.0)
                    shares = math.floor(dollar_amount / entry_price)
                else:
                    shares = getattr(config, "BACKTEST_SHARE_COUNT", getattr(config, "FIXED_SHARE_COUNT", 100))
                    
                if shares <= 0:
                    continue
                    
                trade_id_counter += 1
                active_trade = {
                    'id': trade_id_counter,
                    'symbol': symbol,
                    'side': signal.side,
                    'shares': shares,
                    'entry': entry_price,
                    'stop': signal.stop_price,
                    'target': signal.target_price,
                    'entry_time': current_time,
                    'reason': signal.reason
                }
                
                # Append entry mock execution
                mock_executions.append({
                    "timestamp": current_time.isoformat(),
                    "event_type": "FILL",
                    "type": "FILL",
                    "execution_id": f"backtest-entry-{trade_id_counter}",
                    "qty": float(shares),
                    "side": "BUY" if signal.side == "BUY" else "SELL",
                    "symbol": symbol
                })
                
    # Force close any open trade at the end
    if active_trade:
        final_price = df.iloc[-1]['close']
        active_trade['exit'] = final_price
        if active_trade['side'] == 'BUY':
            executed_exit = final_price * (1 - slippage_pct)
        else:
            executed_exit = final_price * (1 + slippage_pct)
        active_trade['pnl'] = _realized_pnl(
            active_trade['side'],
            active_trade['entry'],
            executed_exit,
            active_trade['shares'],
            commission,
        )
        active_trade['realized_pnl'] = active_trade['pnl']
        active_trade['exit_time'] = df.index[-1]
        trades.append(active_trade)
        current_equity += active_trade['pnl']
        equity_curve.append({'time': df.index[-1], 'equity': current_equity})
        
    total_pnl = 0
    wins = 0
    win_pnls = []
    loss_pnls = []
    
    for t in trades:
        total_pnl += t['pnl']
        if t['pnl'] > 0:
            wins += 1
            win_pnls.append(t['pnl'])
        else:
            loss_pnls.append(t['pnl'])
        
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    losses = len(trades) - wins
    
    avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
    avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0
    
    sum_wins = sum(win_pnls)
    sum_losses = sum(loss_pnls)
    if sum_wins > 0 and sum_losses == 0:
        profit_factor = float('inf')
    elif sum_losses == 0:
        profit_factor = 0.0
    else:
        profit_factor = float(sum_wins / abs(sum_losses))
    
    # --- Visualization ---
    if plot and equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        eq_df.set_index('time', inplace=True)
        
        eq_df['peak'] = eq_df['equity'].cummax()
        eq_df['drawdown'] = _drawdown_series(eq_df['equity'])
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(eq_df.index, eq_df['equity'], label='Net PnL (Costs Included)', color='blue')
        plt.title(f"{symbol} - {strategy_name} Equity Curve")
        plt.ylabel("Cumulative PnL ($)")
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.fill_between(eq_df.index, eq_df['drawdown'], 0, color='red', alpha=0.3)
        plt.plot(eq_df.index, eq_df['drawdown'], color='red')
        plt.title("Drawdown")
        plt.ylabel("Drawdown ($)")
        plt.grid(True)
        
        plt.tight_layout()
        chart_filename = f"backtest_{symbol}_{strategy_name}.png"
        plt.savefig(chart_filename)
        plt.close()
        logger.info(f"Visualization saved to {chart_filename}")
        
    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_pnl": total_pnl,
        "ending_equity": current_equity,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }


def run_portfolio_backtest(
    symbols: List[str],
    strategy_name: str,
    lookback_days: int = 5,
    override_min_confidence: float = None,
    plot: bool = False,
    live_parity: bool = True,
    data_dict: dict = None,
) -> dict:
    logger.info(f"Running PORTFOLIO backtest for {len(symbols)} symbols using {strategy_name} (live_parity={live_parity})")
    
    strategies_map = {
        "ORB": ORBStrategy(),
        "VWAP_BOUNCE": VWAPBounceStrategy(),
        "GAP_AND_GO": GapAndGoStrategy(),
        "MOMENTUM_SCALP": MomentumScalpStrategy(),
        "CANDLESTICK": CandlestickPatternStrategy()
    }

    is_live_mix = strategy_name.upper() in {"LIVE_MIX", "ALL_STRATEGIES", "MIX"}
    strategy = None if is_live_mix else strategies_map.get(strategy_name.upper())
    
    if not is_live_mix and not strategy:
        logger.error(f"Unknown strategy: {strategy_name}")
        return {
            "total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "wins": 0, "losses": 0,
            "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0
        }
        
    if override_min_confidence is not None:
        min_confidence = override_min_confidence
    elif strategy:
        min_confidence = _minimum_confidence_for(strategy)
    else:
        min_confidence = 0.65

    # Load data if not pre-cached
    if data_dict is None:
        from strategies.indicators import add_all_indicators
        data_dict = {}
        for sym in symbols:
            df = fetch_intraday_yfinance(sym, lookback_days=lookback_days)
            if not df.empty:
                df = add_all_indicators(df)
                data_dict[sym] = df
            
    if not data_dict:
        logger.error("No data available for any symbol.")
        return {
            "total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "wins": 0, "losses": 0,
            "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0
        }
        
    all_times = set()
    for df in data_dict.values():
        all_times.update(df.index.tolist())
    all_times = sorted(list(all_times))
    
    start_equity = 100_000.0
    current_equity = start_equity
    
    max_positions = getattr(config, 'PORTFOLIO_MAX_POSITIONS', 3)
    slippage_pct = getattr(config, 'BACKTEST_SLIPPAGE_FRAC', getattr(config, 'BACKTEST_SLIPPAGE_PCT', 0.0005))
    commission = getattr(config, 'BACKTEST_COMMISSION_PER_TRADE', 1.00)
    alloc_pct = getattr(config, "PORTFOLIO_ALLOCATION_PER_TRADE", 0.20)
    
    active_positions = {}
    trades = []
    equity_curve = []
    
    last_date = None
    day_opening_equity = start_equity
    mock_executions = []
    trade_id_counter = 0

    # Live Parity state tracking
    last_symbol_loss_time = {}
    symbol_consecutive_losses = {}
    
    for t in all_times:
        current_date = t.date()
        if last_date is None or current_date != last_date:
            day_opening_equity = current_equity
            last_date = current_date
            symbol_consecutive_losses = {} # Reset per-symbol loss counts on new day

        # 1. Check Exits using t-1 stop
        exited_symbols = []
        for sym, trade in active_positions.items():
            df = data_dict[sym]
            if t not in df.index:
                continue
            
            row = df.loc[t]
            current_open = row['open']
            current_high = row['high']
            current_low = row['low']

            exit_price = _intrabar_exit_price(
                trade['side'],
                current_open,
                current_high,
                current_low,
                trade['stop'],
                trade['target'],
            )
            if exit_price is not None:
                trade['exit'] = exit_price
                trade['exit_time'] = t
                shares = trade.get('shares', 100)
                if trade['side'] == 'BUY':
                    executed_exit = trade['exit'] * (1 - slippage_pct)
                else:
                    executed_exit = trade['exit'] * (1 + slippage_pct)
                realized_pnl = _realized_pnl(
                    trade["side"], trade["entry"], executed_exit, shares, commission
                )
                trade['pnl'] = realized_pnl
                current_equity += realized_pnl
                trade['realized_pnl'] = realized_pnl
                trades.append(trade)
                exited_symbols.append(sym)

                # Parity state updates
                if live_parity:
                    if realized_pnl < 0:
                        last_symbol_loss_time[sym] = t
                        symbol_consecutive_losses[sym] = symbol_consecutive_losses.get(sym, 0) + 1
                    else:
                        symbol_consecutive_losses[sym] = 0
                        last_symbol_loss_time.pop(sym, None)
                
                # Append exit mock execution
                mock_executions.append({
                    "timestamp": t.isoformat(),
                    "event_type": "FILL",
                    "type": "FILL",
                    "execution_id": f"backtest-exit-{trade['id']}",
                    "qty": float(shares),
                    "side": "SELL" if trade['side'] == "BUY" else "BUY",
                    "symbol": sym
                })
                continue
                
            # Update Trailing Stop at end of bar using risk_math
            bar_atr = row.get('atr')
            if bar_atr is not None and not np.isnan(bar_atr) and bar_atr > 0:
                use_atr = bar_atr
            else:
                fallback_pct = getattr(config, 'TRAILING_STOP_FALLBACK_PCT', 2.0) / 100.0
                use_atr = trade['entry'] * fallback_pct / 1.5
                
            math_side = "LONG" if trade['side'] == "BUY" else "SHORT"
            if math_side == "LONG":
                trade['peak'] = max(trade.get('peak', trade['entry']), current_high)
                stop_val = risk_math.trailing_stop_price(
                    current_price=current_high,
                    atr=use_atr,
                    atr_multiple=getattr(config, 'TRAILING_STOP_ATR_MULTIPLE', 1.5),
                    side=math_side,
                    highest_price=trade['peak']
                )
                trade['stop'] = max(trade['stop'], stop_val)
            else:
                trade['peak'] = min(trade.get('peak', trade['entry']), current_low)
                stop_val = risk_math.trailing_stop_price(
                    current_price=current_low,
                    atr=use_atr,
                    atr_multiple=getattr(config, 'TRAILING_STOP_ATR_MULTIPLE', 1.5),
                    side=math_side,
                    lowest_price=trade['peak']
                )
                trade['stop'] = min(trade['stop'], stop_val)
                
        for sym in exited_symbols:
            del active_positions[sym]
            
        # 2. Check Entries
        if len(active_positions) < max_positions:
            # Check Daily Loss Halt Parity
            max_daily_loss_dollars = getattr(config, "MAX_DAILY_LOSS_DOLLARS", 300.0)
            max_daily_loss_pct = getattr(config, "MAX_DAILY_LOSS_PCT", 3.0) / 100.0
            
            daily_loss_hit = False
            if max_daily_loss_dollars > 0 and (current_equity - day_opening_equity) <= -max_daily_loss_dollars:
                daily_loss_hit = True
            elif max_daily_loss_pct > 0 and day_opening_equity > 0:
                loss_pct = abs(current_equity - day_opening_equity) / day_opening_equity
                if (current_equity - day_opening_equity) < 0 and loss_pct >= max_daily_loss_pct:
                    daily_loss_hit = True
                    
            if not daily_loss_hit:
                # Check portfolio-level max trades per day
                today_trades_count = len([tr for tr in trades if tr['exit_time'].date() == current_date])
                if live_parity and today_trades_count >= getattr(config, "MAX_TRADES_PER_DAY", 15):
                    continue

                for sym, df in data_dict.items():
                    if len(active_positions) >= max_positions:
                        break
                    if sym in active_positions: continue
                    if t not in df.index: continue
                    
                    current_df = df.loc[:t]
                    if len(current_df) < 30: continue

                    if live_parity:
                        # Check correlated tech cluster exposure cap (Option A Parity)
                        from strategies.intraday_risk import check_correlated_cluster_exposure
                        if check_correlated_cluster_exposure(list(active_positions.keys()), sym):
                            continue

                        # Check 5-min symbol cooldown on loss
                        if last_loss := last_symbol_loss_time.get(sym):
                            if (t - last_loss).total_seconds() / 60.0 < getattr(config, "REENTRY_COOLDOWN_MINUTES", 5):
                                continue
                        # Check 2-loss blacklist per symbol
                        if symbol_consecutive_losses.get(sym, 0) >= getattr(config, "MAX_SYMBOL_CONSECUTIVE_LOSSES", 2):
                            continue

                    # Evaluate signal(s)
                    best_signal = None
                    if is_live_mix:
                        candidates_signals = []
                        for s_inst in strategies_map.values():
                            sig = s_inst.evaluate(sym, current_df)
                            if sig and sig.confidence >= min_confidence:
                                candidates_signals.append(sig)
                        if candidates_signals:
                            candidates_signals.sort(key=lambda s: s.confidence, reverse=True)
                            best_signal = candidates_signals[0]
                    else:
                        sig = strategy.evaluate(sym, current_df)
                        if sig and sig.confidence >= min_confidence:
                            best_signal = sig

                    if best_signal:
                        # Phase 3: Pre-submission Veto Parity (< max(0.05, entry * 0.002))
                        stop_dist = abs(best_signal.entry_price - best_signal.stop_price)
                        min_required_dist = max(0.05, best_signal.entry_price * 0.002)
                        if stop_dist < min_required_dist:
                            best_signal = None

                    if best_signal:
                        entry_price = best_signal.entry_price
                        entry_price = entry_price * (1 + slippage_pct) if best_signal.side == 'BUY' else entry_price * (1 - slippage_pct)
                        
                        # Apply Sizing Parity
                        sizing_method = getattr(config, "SIZING_METHOD", "risk_based")
                        if sizing_method == "risk_based":
                            risk_pct = getattr(config, "MAX_RISK_PER_TRADE_PCT", 1.0) / 100.0
                            max_trade_val = getattr(config, "MAX_TRADE_VALUE", None)
                            max_pos_pct = getattr(config, "MAX_POSITION_PCT", None)
                            math_side = "LONG" if best_signal.side == "BUY" else "SHORT"
                            shares = risk_math.shares_for_risk(
                                account_equity=current_equity,
                                risk_pct=risk_pct,
                                entry_price=entry_price,
                                stop_price=best_signal.stop_price,
                                side=math_side,
                                max_trade_value=max_trade_val,
                                max_position_pct=max_pos_pct
                            )
                        elif sizing_method == "fixed_dollars":
                            dollar_amount = getattr(config, "FIXED_DOLLAR_AMOUNT", 1000.0)
                            shares = math.floor(dollar_amount / entry_price)
                        else:
                            alloc_dollars = current_equity * alloc_pct
                            shares = int(alloc_dollars / entry_price)
                            
                        if shares <= 0:
                            continue
                            
                        trade_id_counter += 1
                        active_positions[sym] = {
                            'id': trade_id_counter,
                            'symbol': sym,
                            'side': best_signal.side,
                            'shares': shares,
                            'entry': entry_price,
                            'stop': best_signal.stop_price,
                            'target': best_signal.target_price,
                            'peak': entry_price,
                            'entry_time': t,
                            'strategy': getattr(best_signal, "strategy", strategy_name),
                            'reason': best_signal.reason
                        }
                        
                        # Append entry mock execution
                        mock_executions.append({
                            "timestamp": t.isoformat(),
                            "event_type": "FILL",
                            "type": "FILL",
                            "execution_id": f"backtest-entry-{trade_id_counter}",
                            "qty": float(shares),
                            "side": "BUY" if best_signal.side == "BUY" else "SELL",
                            "symbol": sym
                        })
                        
                        if len(active_positions) >= max_positions:
                            break # portfolio full
                        
        # Record equity
        mtm_equity = current_equity
        for sym, trade in active_positions.items():
            if t in data_dict[sym].index:
                curr_price = data_dict[sym].loc[t]['close']
                shares = trade.get('shares', 100)
                if trade['side'] == 'BUY':
                    unrealized = (curr_price - trade['entry']) * shares
                else:
                    unrealized = (trade['entry'] - curr_price) * shares
                mtm_equity += unrealized
                
        equity_curve.append({'time': t, 'equity': mtm_equity})
        
    # Force close open positions at end
    final_time = all_times[-1]
    for sym, trade in active_positions.items():
        final_price = data_dict[sym].iloc[-1]['close']
        trade['exit'] = final_price
        shares = trade.get('shares', 100)
        if trade['side'] == 'BUY':
            executed_exit = final_price * (1 - slippage_pct)
        else:
            executed_exit = final_price * (1 + slippage_pct)
        realized_pnl = _realized_pnl(
            trade["side"], trade["entry"], executed_exit, shares, commission
        )
        trade['pnl'] = realized_pnl
        current_equity += realized_pnl
        trade['realized_pnl'] = realized_pnl
        trade['exit_time'] = final_time
        trades.append(trade)
    
    total_return = (current_equity - start_equity) / start_equity * 100
    wins = len([t for t in trades if t['realized_pnl'] > 0])
    win_rate = (wins / len(trades) * 100) if trades else 0.0
    losses = len(trades) - wins
    
    win_pnls = [t['realized_pnl'] for t in trades if t['realized_pnl'] > 0]
    loss_pnls = [t['realized_pnl'] for t in trades if t['realized_pnl'] < 0]
    
    avg_win = float(np.mean(win_pnls)) if win_pnls else 0.0
    avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0.0
    
    sum_wins = sum(win_pnls)
    sum_losses = sum(loss_pnls)
    if sum_wins > 0 and sum_losses == 0:
        profit_factor = float('inf')
    elif sum_losses == 0:
        profit_factor = 0.0
    else:
        profit_factor = float(sum_wins / abs(sum_losses))
        
    # Visualization
    if plot and equity_curve:
        eq_df = pd.DataFrame(equity_curve)
        eq_df.set_index('time', inplace=True)
        
        eq_df['peak'] = eq_df['equity'].cummax()
        eq_df['drawdown'] = eq_df['peak'] - eq_df['equity']
        max_dd = eq_df['drawdown'].max()
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, (1, 2))
        plt.plot(eq_df.index, eq_df['equity'], label='Portfolio Equity', color='blue')
        plt.title(f"Portfolio Equity Curve ({strategy_name})")
        plt.ylabel("Equity ($)")
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.fill_between(eq_df.index, eq_df['drawdown'], 0, color='red', alpha=0.3)
        plt.title("Portfolio Drawdown ($)")
        plt.ylabel("Drawdown ($)")
        plt.grid(True)
        
        plt.subplot(2, 2, 4)
        pnls = [t['realized_pnl'] for t in trades]
        plt.hist(pnls, bins=20, color='green', alpha=0.7)
        plt.axvline(0, color='black', linestyle='--')
        plt.title("Trade PnL Distribution")
        plt.xlabel("Realized PnL ($)")
        
        plt.tight_layout()
        chart_filename = f"backtest_PORTFOLIO_{strategy_name}.png"
        plt.savefig(chart_filename)
        plt.close()
        logger.info(f"Portfolio visualization saved to {chart_filename}")
        
    return {
        "total_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "net_pnl": (current_equity - start_equity),
        "ending_equity": current_equity,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "trades": trades,
    }
