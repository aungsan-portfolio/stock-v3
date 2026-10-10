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

from calendar import monthrange
G['monthrange'] = monthrange
for n in ['_observed', '_nth_weekday', '_last_weekday', '_easter_sunday', '_nyse_holidays', 'is_market_day', 'get_previous_market_day']:
    extract('strategies/session.py', n)

mod = types.ModuleType('strategies.session')
mod.get_previous_market_day = G['get_previous_market_day']
mod.now_eastern = lambda: NS(date=lambda: date(2026, 10, 9))
sys.modules.setdefault('strategies', types.ModuleType('strategies'))
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
def rec(name, data):
    out.append({'case': name, 'result': data})

def write(path, key, reason='TAKE_PROFIT'):
    return log(symbol='AAPL', side='SELL', strategy='MEAN_REVERSION', qty=12, entry_price=331.88, stop_price=331.89, target_price=334.99, exit_price=335.10, pnl=38.64, exit_reason=reason, event_type='TRADE_CLOSED', journal_file=path, idempotency_key=key)

def item():
    return dict(episode_id='AAPL:sig-1', symbol='AAPL', side='BUY', qty=12., entry_price=331.88, stop_price=331.89, target_price=334.99, tp_order_id='tp-1', stop_order_id='stop-1', last_stop_order_id='stop-1', is_validation=False, strategy='MEAN_REVERSION', signal_id='sig-1')

def manager(save=lambda: True):
    return NS(_lock=threading.RLock(), states={}, pending_exits={'AAPL:sig-1': item()}, _save_state=save)

def filled(**kw):
    return dict(id='tp-1', status='filled', symbol='AAPL', side='sell', qty=12., filled_qty=12., filled_avg_price=335.10, **kw)

def bridge(snapshot=None, orders=None):
    return NS(get_order_snapshot=lambda oid: snapshot if oid == 'tp-1' else None, _client=NS(get_orders=lambda req: orders) if orders is not None else None)

base = {'id': 'stop-1', 'symbol': 'AAPL', 'side': 'sell', 'type': 'stop', 'status': 'new', 'qty': '12', 'filled_qty': '0', 'stop_price': '331.89'}

# 1-3. Snapshots
for label, raw in [('snapshot_complete', base), ('snapshot_missing_id', {k: v for k, v in base.items() if k != 'id'}), ('snapshot_partial_missing_filled_qty', {**{k: v for k, v in base.items() if k != 'filled_qty'}, 'status': 'partially_filled'})]:
    d = snap(NS(_client=NS(get_order_by_id=lambda oid, raw=raw: raw)), 'stop-1')
    v = G['verify_broker_stop_order_evidence'](d, 'stop-1', 'AAPL', 'BUY', 331.89, expected_qty=12., require_qty=True)
    rec(label, v)

# 4-7. Broker position queries
for label, q in [('position_timeout', RuntimeError('timeout')), ('position_none', None), ('position_nan', float('nan')), ('position_short', -12.)]:
    b = NS(get_position=lambda symbol, q=q: (_ for _ in ()).throw(q) if isinstance(q, Exception) else q)
    rec(label, G['_query_broker_position_qty'](None, b, 'AAPL'))

# Adopt state & manager
def adopt_state():
    return NS(symbol='AAPL', side='BUY', active=True, order_id=None, qty=12., entry_price=331.88, stop_price=331.89, desired_stop_price=331.89, strategy='MEAN_REVERSION', reconciliation_status='POSITION_UNKNOWN_RECONCILIATION_REQUIRED', stop_reconciliation_required=True)

def adopt_manager(s):
    m = NS(_lock=threading.RLock(), states={'AAPL': s}, pending_exits={}, _save_state=lambda: True, _extract_all_orders_including_legs=lambda x: x)
    m._is_valid_stop_order = types.MethodType(G['_is_valid_stop_order'], m)
    return m

# 8-9. Ensure initialized
for name, val in [('ensure_timeout_retains_active', RuntimeError('timeout')), ('ensure_short_retains_active', -12.)]:
    s = adopt_state()
    s.order_id = 'stop-1'
    if isinstance(val, float):
        s.side = 'SELL'
    m = adopt_manager(s)
    m._query_broker_position_qty = types.MethodType(G['_query_broker_position_qty'], m)
    m._reconcile_broker_order = types.MethodType(G['_reconcile_broker_order'], m)
    b = NS(get_order_snapshot=lambda oid: {**base, 'status': 'canceled'}, get_position=lambda sym, val=val: (_ for _ in ()).throw(val) if isinstance(val, Exception) else val)
    G['ensure_initialized'](m, 'AAPL', s.side, 331.88, [], 332., qty=val if isinstance(val, float) else 12., bridge=b)
    rec(name, dict(active=s.active, order_id=s.order_id, qty=s.qty, status=s.reconciliation_status))

# 10. Adoption missing quantity
s = adopt_state()
m = adopt_manager(s)
bad = {k: v for k, v in base.items() if k != 'qty'}
r = G['_reconcile_broker_order'](m, s, [bad])
rec('legacy_adoption_missing_quantity', dict(adopted=r, order_id=s.order_id, status=s.reconciliation_status, stop_reconciliation_required=s.stop_reconciliation_required))

# 11. Partial stop reconciles
s = adopt_state()
m = adopt_manager(s)
partial = {**base, 'status': 'partially_filled', 'filled_qty': '4'}
r1 = G['_reconcile_broker_order'](m, s, [partial])
first = dict(adopted=r1, active=s.active, qty=s.qty, order_id=s.order_id)
r2 = G['_reconcile_broker_order'](m, s, [partial])
rec('legacy_partial_stop_two_reconciles', dict(first=first, second=dict(adopted=r2, active=s.active, order_id=s.order_id, pending_remaining=len(m.pending_exits))))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # 12. Fsync failure
    p = tmp / 'fsync.jsonl'
    with patch('os.fsync', side_effect=OSError('injected fsync failure')):
        try:
            write(p, 'exit_AAPL:fsync')
            result = 'returned_success'
        except Exception as e:
            result = type(e).__name__
    rec('journal_fsync_failure', {'return': result, 'file_exists': p.exists()})

    # 13. Prefix collision
    p = tmp / 'prefix.jsonl'
    write(p, 'exit_AAPL:sig-10')
    write(p, 'exit_AAPL:sig-1')
    rec('distinct_episode_prefix_collision', {'attempted_episodes': 2, 'records': len(p.read_text().splitlines())})

    # 14. Truncated tail
    p = tmp / 'tail.jsonl'
    p.write_text('{"event_type":"TRADE_CLOSED","idempotency_key":"exit_AAPL:sig-1"')
    write(p, 'exit_AAPL:sig-1')
    lines = p.read_text().splitlines()
    parsed = []
    for line in lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    rec('truncated_tail_append', {'lines': len(lines), 'parseable_records_after_new_write': len(parsed)})

    # 15-20. Reconcile tests
    for label, b, save in [
        ('reconcile_fsync_failure', bridge(filled()), lambda: True),
        ('snapshot_wrong_identity_side_symbol', bridge({'id': 'other', 'status': 'filled', 'symbol': 'MSFT', 'side': 'buy', 'qty': 1., 'filled_qty': 1., 'filled_avg_price': 335.10}), lambda: True),
        ('snapshot_one_share_closes_twelve', bridge({'id': 'tp-1', 'status': 'filled', 'symbol': 'AAPL', 'side': 'sell', 'qty': 1., 'filled_qty': 1., 'filled_avg_price': 335.10}), lambda: True),
        ('unrelated_dt_prefix_closed_exit', bridge(orders=[NS(id='old-exit', client_order_id='dt_previous_episode', symbol='AAPL', side='sell', status='filled', order_type='limit', filled_avg_price=340., qty=12., filled_qty=12.)]), lambda: True),
        ('queue_save_returns_false', bridge(filled()), lambda: False),
        ('missing_actual_fill_stays_pending', bridge({'id': 'tp-1', 'status': 'filled', 'filled_avg_price': None, 'limit_price': 334.99}), lambda: True),
    ]:
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tmp / (label + '.jsonl')
        m = manager(save)
        if label == 'reconcile_fsync_failure':
            with patch('os.fsync', side_effect=OSError('injected fsync failure')):
                r = G['reconcile_pending_exits'](m, b)
        else:
            r = G['reconcile_pending_exits'](m, b)
        path = G['config'].DAYTRADE_TRADE_JOURNAL_FILE
        entries = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
        rec(label, {'return': r, 'pending_remaining': len(m.pending_exits), 'journal_records': [{k: e[k] for k in ['qty', 'exit_price', 'pnl', 'exit_reason']} for e in entries]})

    # 21. Unknown cleared without protection
    s = NS(symbol='AAPL', side='BUY', active=True, order_id=None, qty=12., reconciliation_status='POSITION_UNKNOWN_RECONCILIATION_REQUIRED', stop_reconciliation_required=True)
    m = NS(_lock=threading.RLock(), states={'AAPL': s}, pending_exits={}, _save_state=lambda: True, _query_broker_position_qty=lambda b, sym: (True, 12.))
    G['handle_naked_position'](m, 'AAPL', 12., NS(), dry_run=True)
    rec('unknown_cleared_without_protection', dict(status=s.reconciliation_status, active=s.active, order_id=s.order_id, stop_reconciliation_required=s.stop_reconciliation_required))

# 22-24. Baseline tests
for label, bars in [
    ('baseline_only_today', [{'timestamp': '2026-10-09T14:00:00Z', 'close': 108.}]),
    ('baseline_missing_previous_session', [{'timestamp': '2026-10-07T20:00:00Z', 'close': 100.}]),
    ('baseline_previous_correct_timestamp', [{'timestamp': '2026-10-08T20:00:00Z', 'close': 105.}, {'timestamp': '2026-10-09T14:00:00Z', 'close': 108.}]),
]:
    rec(label, G['_extract_prev_session_close'](bars, today_date=date(2026, 10, 9)))

# 25. Historical fallback
b = NS(get_quote=lambda s: None, get_price=lambda s, **kw: 105., fetch_historical_data=lambda s: [{'timestamp': '2026-10-08T20:00:00Z', 'close': 105.}])
with patch('time.time', return_value=1791554400.):
    rec('historical_price_relabelled_fresh', G['_get_premarket_pct_change']('SPY', b))

print(json.dumps(out, indent=2, allow_nan=True))
