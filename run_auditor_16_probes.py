#!/usr/bin/env python3
"""
Auditor's 16 boundary probe suite for testing execution safety,
durability, and provenance invariants.
"""
import ast, json, math, os, time, logging, threading, types, sys, tempfile
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from typing import *
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
G = dict(globals(), config=NS(NAKED_LIMIT_PROTECTION='warn'), logger=logging.getLogger('offline'), TradeSignal=object, TrailingStopState=object, _journal_lock=threading.RLock(), is_market_open=lambda: False, send_discord_alert=lambda *a: None)
G['logger'].addHandler(logging.NullHandler())
G['logger'].propagate = False

def extract(rel, name, cls=None):
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    nodes = tree.body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    n = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    n.decorator_list = []
    exec(compile(ast.Module(body=[n], type_ignores=[]), str(ROOT / rel), 'exec'), G)
    return G[name]

for n in ['normalize_order_type', '_normalize_sdk_field', '_extract_order_field', '_is_finite_positive', '_is_finite_non_negative', 'verify_broker_stop_order_evidence', '_extract_order_info', '_normalize_strategy_name']:
    extract('strategies/trailing_stop.py', n)

G.update(
    _STOP_ORDER_TYPES={'stop', 'stop_limit', 'trailing_stop'},
    _WORKING_STOP_STATUSES={'new', 'accepted', 'pending_new', 'accepted_for_bidding', 'held', 'calculated', 'partially_filled', 'open'},
    _TERMINAL_STOP_STATUSES={'rejected', 'canceled', 'cancelled', 'expired', 'failed', 'done_for_day', 'stopped', 'filled'}
)

for n in ['_query_broker_position_qty', 'ensure_initialized', 'handle_naked_position', 'reconcile_pending_exits', '_is_valid_stop_order', '_reconcile_broker_order']:
    extract('strategies/trailing_stop.py', n, 'DynamicTrailingStopManager')

extract('strategies/trailing_stop.py', 'is_stop_order_type')
snap = extract('alpaca_bridge.py', 'get_order_snapshot', 'AlpacaBridge')
extract('strategies/trade_journal.py', '_normalize_exit_idempotency_key')
log = extract('strategies/trade_journal.py', 'log_trade')
extract('strategies/regime_engine.py', '_extract_prev_session_close')
extract('strategies/regime_engine.py', '_get_premarket_pct_change')

mod = types.ModuleType('strategies.session')
mod.get_previous_market_day = lambda *a, **kw: date(2026, 10, 8)
mod.now_eastern = lambda: NS(date=lambda: date(2026, 10, 9))
sys.modules['strategies.session'] = mod
jm = types.ModuleType('strategies.trade_journal')
jm.log_trade = log
sys.modules['strategies.trade_journal'] = jm
req = types.ModuleType('alpaca.trading.requests')
req.GetOrdersRequest = lambda **kw: NS(**kw)
en = types.ModuleType('alpaca.trading.enums')
en.QueryOrderStatus = NS(CLOSED='closed')
for n in ['alpaca', 'alpaca.trading']:
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules['alpaca.trading.requests'] = req
sys.modules['alpaca.trading.enums'] = en

out = []
def rec(n, x): out.append({'case': n, 'result': x})

base = {'id': 'stop-1', 'symbol': 'AAPL', 'side': 'sell', 'type': 'stop', 'status': 'new', 'qty': '12', 'filled_qty': '0', 'stop_price': '331.89'}
def adopt_state():
    return NS(symbol='AAPL', side='BUY', active=True, order_id=None, qty=12., entry_qty=12., entry_price=331.88, stop_price=331.89, desired_stop_price=331.89, strategy='MEAN_REVERSION', reconciliation_status='POSITION_UNKNOWN_RECONCILIATION_REQUIRED', stop_reconciliation_required=True)
def adopt_manager(s):
    m = NS(_lock=threading.RLock(), states={'AAPL': s}, pending_exits={}, _save_state=lambda: True, _extract_all_orders_including_legs=lambda x: x, _record_pending_exit=lambda s, qty_closed=None: None)
    m._is_valid_stop_order = types.MethodType(G['_is_valid_stop_order'], m)
    return m

def item():
    return dict(episode_id='AAPL:sig-1', symbol='AAPL', side='BUY', qty=12., entry_price=331.88, stop_price=331.89, target_price=334.99, tp_order_id='tp-1', stop_order_id='stop-1', last_stop_order_id='stop-1', is_validation=False, strategy='MEAN_REVERSION', signal_id='sig-1')
def manager(save=lambda: True):
    m = NS(_lock=threading.RLock(), states={}, pending_exits={'AAPL:sig-1': item()}, _save_state=save)
    m._query_broker_position_qty = types.MethodType(G['_query_broker_position_qty'], m)
    return m
def filled(**kw):
    return dict(id='tp-1', status='filled', symbol='AAPL', side='sell', qty=12., filled_qty=12., filled_avg_price=335.10, **kw)
def bridge(snapshot=None, orders=None, pos=0.0):
    return NS(get_position=lambda s: pos, get_order_snapshot=lambda oid: snapshot if oid == 'tp-1' else None, _client=NS(get_orders=lambda req: orders) if orders is not None else None)

partial = {**base, 'status': 'partially_filled', 'filled_qty': '4'}

# 1-2
for q in (12., 8.):
    s = adopt_state(); m = adopt_manager(s)
    r = G['_is_valid_stop_order'](m, partial, 'sell', q, 'AAPL')
    rec('partial_remaining_8_required_' + str(q), {'accepted': r, 'remaining_capacity': 8., 'required_qty': q})

# 3
s = adopt_state(); m = adopt_manager(s); G['_reconcile_broker_order'](m, s, [partial])
rec('partial_first_adoption_coverage', {'active': s.active, 'local_qty': s.qty, 'stop_remaining': 8., 'unknown_status': s.reconciliation_status, 'stop_reconciliation_required': s.stop_reconciliation_required})

# 4
s = adopt_state(); s.order_id = 'stop-1'; m = adopt_manager(s)
r = G['_reconcile_broker_order'](m, s, [{**base, 'status': 'cancelled'}])
rec('tracked_cancelled_stop_not_position_flat', {'returned': r, 'active': s.active, 'order_id': s.order_id, 'pending_exits': len(m.pending_exits)})

# 5
s = adopt_state(); s.order_id = 'unverified-old-id'; m = adopt_manager(s); m._query_broker_position_qty = lambda b, sym: (True, 12.)
G['handle_naked_position'](m, 'AAPL', 12., NS(), dry_run=True)
rec('unknown_cleared_by_nonempty_unverified_id', {'status': s.reconciliation_status, 'order_id': s.order_id, 'stop_reconciliation_required': s.stop_reconciliation_required})

with tempfile.TemporaryDirectory() as t:
    t = Path(t)
    def reconcile_case(name, b):
        p = t / (name + '.jsonl'); G['config'].DAYTRADE_TRADE_JOURNAL_FILE = p; m = manager(); r = G['reconcile_pending_exits'](m, b)
        rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        rec(name, {'return': r, 'pending': len(m.pending_exits), 'records': [{k: e.get(k) for k in ('qty', 'exit_price', 'pnl', 'exit_reason')} for e in rows]})

    for name, snapshot in [('snapshot_missing_filled_qty_uses_order_qty', {k: v for k, v in filled().items() if k != 'filled_qty'}), ('snapshot_infinite_filled_qty', dict(filled(), filled_qty=float('inf')))]:
        reconcile_case(name, bridge(snapshot))

    for name, kw in [('substring_other_episode_sig10', {'id': 'other-exit', 'client_order_id': 'exit_AAPL:sig-10'}), ('fallback_nan_filled_qty', {'id': 'tp-1', 'client_order_id': '', 'filled_qty': float('nan')}), ('fallback_partial_status', {'id': 'tp-1', 'client_order_id': '', 'status': 'partially_filled', 'qty': 14.}), ('fallback_missing_filled_qty_uses_qty', {'id': 'tp-1', 'client_order_id': ''})]:
        vals = dict(id='tp-1', client_order_id='', symbol='AAPL', side='sell', status='filled', order_type='limit', qty=12., filled_qty=12., filled_avg_price=340.)
        vals.update(kw)
        if name == 'fallback_missing_filled_qty_uses_qty': vals.pop('filled_qty')
        reconcile_case(name, bridge(orders=[NS(**vals)]))

    orders = [NS(id='stop-1', client_order_id='', symbol='AAPL', side='sell', status='filled', order_type='stop', qty=4., filled_qty=4., filled_avg_price=330.), NS(id='tp-1', client_order_id='', symbol='AAPL', side='sell', status='filled', order_type='limit', qty=8., filled_qty=8., filled_avg_price=335.)]
    reconcile_case('split_exits_4_plus_8_cover_12', bridge(orders=orders))

    p = t / 'retry.jsonl'; G['config'].DAYTRADE_TRADE_JOURNAL_FILE = p; m = manager()
    with patch('os.fsync', side_effect=OSError('disk sync still unavailable')) as f:
        r1 = G['reconcile_pending_exits'](m, bridge(filled())); count1 = f.call_count
        r2 = G['reconcile_pending_exits'](m, bridge(filled())); count2 = f.call_count
    rec('fsync_failure_retry_dedup_evicts_without_sync', {'first': r1, 'second': r2, 'first_fsync_calls': count1, 'additional_fsync_calls_on_retry': count2 - count1, 'pending_after_retry': len(m.pending_exits)})

    p = t / 'flat.jsonl'; G['config'].DAYTRADE_TRADE_JOURNAL_FILE = p; m = manager(); b = bridge(filled()); calls = []; b.get_position = lambda s: calls.append(s) or 12.
    r = G['reconcile_pending_exits'](m, b); rec('pending_exit_no_current_position_check', {'result': r, 'broker_position_query_calls': len(calls), 'pending': len(m.pending_exits)})

b = NS(get_quote=lambda s: None, get_price=lambda s, **kw: 110., fetch_historical_data=lambda s: [{'timestamp': '2026-10-08T20:00:00Z', 'close': 105.}])
rec('fallback_price_epoch_reused_from_baseline', {'price_observation_timestamp': None, 'result': G['_get_premarket_pct_change']('SPY', b)})

yf = types.ModuleType('yfinance'); yf.Ticker = lambda s: NS(fast_info=NS(last_price=110., previous_close=105.))
with patch.dict('sys.modules', {'yfinance': yf}), patch('time.time', return_value=1791554400.):
    rec('yfinance_missing_provider_timestamp_labelled_fresh', G['_get_premarket_pct_change']('SPY', None))

if __name__ == '__main__':
    print(json.dumps(out, indent=2, allow_nan=True))
