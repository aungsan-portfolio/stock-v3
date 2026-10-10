"""Additional offline unchanged-source observations; external boundaries stubbed, not E2E tests."""
import runpy, contextlib, io, pathlib, json, tempfile, types, threading
from types import SimpleNamespace as NS

with contextlib.redirect_stdout(io.StringIO()):
    d = runpy.run_path(str(pathlib.Path(__file__).with_name('run_auditor_16_probes.py')))

G = d['G']
out = []

def rec(n, x):
    out.append({'case': n, 'result': x})

def state(**kw):
    s = d['adopt_state']()
    s.__dict__.update(
        signal_id='sig-1',
        tp_order_id='tp-1',
        last_stop_order_id=None,
        entry_time=1791554400.,
        target_price=334.99,
        is_validation=False,
        remediation_in_progress=False,
    )
    s.__dict__.update(kw)
    return s

def manager(s):
    m = d['adopt_manager'](s)
    m._query_broker_position_qty = types.MethodType(G['_query_broker_position_qty'], m)
    m._reconcile_broker_order = types.MethodType(G['_reconcile_broker_order'], m)
    return m

# Extract actual _record_pending_exit
tree = G['ast'].parse((G['ROOT'] / 'strategies/trailing_stop.py').read_text(encoding="utf-8"))
cls = next(n for n in tree.body if isinstance(n, G['ast'].ClassDef) and n.name == 'DynamicTrailingStopManager')
node = next(n for n in cls.body if isinstance(n, G['ast'].FunctionDef) and n.name == '_record_pending_exit')
exec(compile(G['ast'].Module(body=[node], type_ignores=[]), '_record_pending_exit', 'exec'), G)

# 1. new_unknown_from_previously_healthy & signed_short_with_stale_long_side
for label, pos, kw in [
    ('new_unknown_from_previously_healthy', None, dict(reconciliation_status=None, stop_reconciliation_required=False)),
    ('signed_short_with_stale_long_side', -12., {}),
]:
    s = state(**kw)
    m = manager(s)
    b = NS(get_position=lambda sym, pos=pos: pos)
    r = G['_reconcile_broker_order'](m, s, [d['base']], bridge=b)
    rec(label, dict(
        adopted=r,
        order_id=s.order_id,
        local_side=s.side,
        qty=s.qty,
        status=s.reconciliation_status,
        stop_reconciliation_required=s.stop_reconciliation_required,
    ))

# 2. filled_stop_with_residual_position_5
s = state(order_id='stop-1')
m = manager(s)
m._record_pending_exit = types.MethodType(G['_record_pending_exit'], m)
r = G['_reconcile_broker_order'](m, s, [dict(d['base'], status='filled', filled_qty='12')], bridge=NS(get_position=lambda sym: 5.))
rec('filled_stop_with_residual_position_5', dict(
    returned=r,
    active=s.active,
    order_id=s.order_id,
    broker_qty=5.,
    pending=list(m.pending_exits.values()),
))

# 3. tracked_direct_lookup_wrong_identity_quantity
s = state(order_id='stop-1', reconciliation_status=None, stop_reconciliation_required=False)
m = manager(s)
calls = []
b = NS(
    get_order_snapshot=lambda oid: dict(d['base'], id='other-id', symbol='MSFT', side='buy', qty='1', stop_price='999'),
    get_position=lambda sym: calls.append(sym) or 12.,
)
G['ensure_initialized'](m, 'AAPL', 'BUY', 331.88, [], 332., qty=12., bridge=b)
rec('tracked_direct_lookup_wrong_identity_quantity', dict(
    active=s.active,
    order_id=s.order_id,
    stop_price=s.stop_price,
    unknown=s.reconciliation_status,
    reconciliation_required=s.stop_reconciliation_required,
    position_queries=len(calls),
))

# 4. tracked_present_one_share_stop_for_12
s = state(order_id='stop-1', reconciliation_status=None, stop_reconciliation_required=False)
m = manager(s)
G['ensure_initialized'](m, 'AAPL', 'BUY', 331.88, [dict(d['base'], qty='1')], 332., qty=12., bridge=NS(get_position=lambda sym: 12.))
rec('tracked_present_one_share_stop_for_12', dict(
    active=s.active,
    order_id=s.order_id,
    qty=s.qty,
    stop_capacity=1.,
    unknown=s.reconciliation_status,
    reconciliation_required=s.stop_reconciliation_required,
))

# 5. partial_episode_12_reduced_to_8_then_full_exit_12
s = state()
m = manager(s)
m._record_pending_exit = types.MethodType(G['_record_pending_exit'], m)
G['_reconcile_broker_order'](m, s, [dict(d['base'], status='partially_filled', filled_qty='4')], bridge=NS(get_position=lambda sym: 8.))
m._record_pending_exit(s)
with tempfile.TemporaryDirectory() as t:
    t = pathlib.Path(t)
    G['config'].DAYTRADE_TRADE_JOURNAL_FILE = t / 'split.jsonl'
    orders = [
        NS(id='stop-1', client_order_id='', symbol='AAPL', side='sell', status='canceled', order_type='stop', filled_qty=4., filled_avg_price=330.),
        NS(id='tp-1', client_order_id='', symbol='AAPL', side='sell', status='filled', order_type='limit', filled_qty=8., filled_avg_price=335.),
    ]
    pending_qty = next(iter(m.pending_exits.values()))['qty']
    r = G['reconcile_pending_exits'](m, NS(get_position=lambda sym: 0., _client=NS(get_orders=lambda req: orders)))
    rec('partial_episode_12_reduced_to_8_then_full_exit_12', dict(
        local_qty=s.qty,
        queued_qty=pending_qty,
        actual_episode_exit_qty=12.,
        result=r,
        pending=len(m.pending_exits),
    ))

    def reconcile_case(name, snapshot, orders, qty=12.):
        p = t / (name + '.jsonl')
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = p
        m_case = d['manager']()
        m_case.pending_exits['AAPL:sig-1']['qty'] = qty
        b_case = NS(
            get_position=lambda sym: 0.,
            get_order_snapshot=lambda oid: snapshot if oid == 'tp-1' else None,
            _client=NS(get_orders=lambda req: orders),
        )
        r_case = G['reconcile_pending_exits'](m_case, b_case)
        rows = [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
        rec(name, dict(
            result=r_case,
            pending=len(m_case.pending_exits),
            records=[{k: x.get(k) for k in ('qty', 'exit_price', 'pnl', 'exit_reason')} for x in rows],
        ))

    snap = dict(id='tp-1', symbol='AAPL', side='sell', type='limit', status='filled', filled_qty=12., filled_avg_price=335.1)
    co = NS(id='tp-1', symbol='AAPL', side='sell', order_type='limit', status='filled', filled_qty=16., filled_avg_price=340., client_order_id='')
    reconcile_case('conflicting_same_id_snapshot_12_closed_16', snap, [co])

    co = NS(id='tp-1', symbol='AAPL', side='sell', order_type='limit', status='filled', filled_qty=12., filled_avg_price=340., client_order_id='')
    reconcile_case('conflicting_same_id_price_335_vs_340', snap, [co])

    reconcile_case('fractional_episode_12_5_journal_int_truncation', dict(snap, filled_qty=12.5), [], 12.5)

    # 6. Cancelled stop with broker-known nonzero now correctly stays active, control
    s = state(order_id='stop-1')
    m = manager(s)
    r = G['_reconcile_broker_order'](m, s, [dict(d['base'], status='canceled')], bridge=NS(get_position=lambda sym: 12.))
    rec('cancelled_stop_open_position_control', dict(
        returned=r,
        active=s.active,
        order_id=s.order_id,
        status=s.reconciliation_status,
    ))

# 7. Held-quantity recovery path must not classify an order type as sufficient protection
node = next(n for n in tree.body if isinstance(n, G['ast'].FunctionDef) and n.name == '_classify_held_order')
exec(compile(G['ast'].Module(body=[node], type_ignores=[]), '_classify_held_order', 'exec'), G)
G['config'].NAKED_LIMIT_PROTECTION = 'replace'
G['is_market_open'] = lambda: True
G['MAX_REMEDIATION_STRIKES'] = 3
import sys
sys.modules['alpaca.trading.enums'].QueryOrderStatus.ALL = 'all'
sys.modules['alpaca.trading.requests'].GetOrderByIdRequest = lambda **kw: NS(**kw)

s = state()
m = manager(s)
bad = NS(id='bad-stop', symbol='AAPL', side='buy', order_type='stop', status='new', stop_price=331.89, qty=None, filled_qty=None, legs=[])
b = NS(
    get_position=lambda sym: 12.,
    working_orders_plain=lambda: [],
    market_price=lambda sym: 332.,
    _client=NS(get_orders=lambda req: [], get_order_by_id=lambda *a: bad),
    place_stop_order=lambda **kw: (_ for _ in ()).throw(RuntimeError('insufficient qty; related_orders": ["bad-stop"]')),
)
G['handle_naked_position'](m, 'AAPL', 12., b, dry_run=False)
rec('held_qty_related_order_wrong_side_missing_qty', dict(
    order_id=s.order_id,
    active=s.active,
    unknown=s.reconciliation_status,
    reconciliation_required=s.stop_reconciliation_required,
    candidate_side='buy',
    required_side='sell',
    candidate_qty=None,
))

if __name__ == '__main__':
    print(json.dumps(out, indent=2))
