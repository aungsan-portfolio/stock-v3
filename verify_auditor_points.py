#!/usr/bin/env python3
"""
Direct Boundary Verification Probes for Auditor's 4 Technical Hardening Points:
1. Broker-flat probe & actual position query (12 -> open retained, 0 -> reconciled, timeout/None/NaN -> unknown retained)
2. fsync retry durability & duplicate prevention (0 duplicate records on persistent fsync failure)
3. Split-exit aggregation, quantity conservation & duplicate orders (snapshot+closed dedup, cancelled partial fill, excess quantity fail-closed)
4. Stop adoption uses fresh signed broker position (12 pos vs 8 stop rejected, 8 pos vs 8 stop accepted, unknown retained if position unknown)
"""
import ast, json, math, os, sys, tempfile, threading, types
from datetime import datetime, date, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent

G = dict(
    globals(),
    config=NS(NAKED_LIMIT_PROTECTION='warn'),
    logger=types.SimpleNamespace(
        warning=lambda *a, **k: None,
        info=lambda *a, **k: None,
        debug=lambda *a, **k: None,
        error=lambda *a, **k: None,
        critical=lambda *a, **k: None,
    ),
    TradeSignal=object,
    TrailingStopState=object,
    _journal_lock=threading.RLock(),
    is_market_open=lambda: False,
    send_discord_alert=lambda *a: None,
)

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

extract('strategies/trailing_stop.py', 'is_stop_order_type')

G.update(
    _STOP_ORDER_TYPES={'stop', 'stop_limit', 'trailing_stop'},
    _WORKING_STOP_STATUSES={'new', 'accepted', 'pending_new', 'accepted_for_bidding', 'held', 'calculated', 'partially_filled', 'open'},
    _TERMINAL_STOP_STATUSES={'rejected', 'canceled', 'cancelled', 'expired', 'failed', 'done_for_day', 'stopped', 'filled'}
)

for n in ['_query_broker_position_qty', 'ensure_initialized', 'handle_naked_position', 'reconcile_pending_exits', '_is_valid_stop_order', '_reconcile_broker_order']:
    extract('strategies/trailing_stop.py', n, 'DynamicTrailingStopManager')

extract('strategies/trade_journal.py', '_normalize_exit_idempotency_key')
log = extract('strategies/trade_journal.py', 'log_trade')

jm = types.ModuleType('strategies.trade_journal')
jm.log_trade = log
sys.modules['strategies.trade_journal'] = jm

req = types.ModuleType('alpaca.trading.requests')
req.GetOrdersRequest = lambda **kw: NS(**kw)
en = types.ModuleType('alpaca.trading.enums')
en.QueryOrderStatus = NS(CLOSED='closed', ALL='all')
for n in ['alpaca', 'alpaca.trading']:
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules['alpaca.trading.requests'] = req
sys.modules['alpaca.trading.enums'] = en

def make_item():
    return dict(
        episode_id='AAPL:sig-1',
        symbol='AAPL',
        side='BUY',
        qty=12.0,
        entry_price=331.88,
        stop_price=331.89,
        target_price=334.99,
        tp_order_id='tp-1',
        stop_order_id='stop-1',
        last_stop_order_id='stop-1',
        is_validation=False,
        strategy='MEAN_REVERSION',
        signal_id='sig-1',
    )

def make_manager(save=lambda: True):
    m = NS(_lock=threading.RLock(), states={}, pending_exits={'AAPL:sig-1': make_item()}, _save_state=save)
    m._query_broker_position_qty = types.MethodType(G['_query_broker_position_qty'], m)
    return m

def make_bridge(pos_val=0.0, snapshot=None, orders=None, raise_exc=None):
    def get_pos(sym):
        if raise_exc:
            raise raise_exc
        return pos_val
    return NS(
        get_position=get_pos,
        get_order_snapshot=lambda oid: snapshot if oid in ('tp-1', 'stop-1') else None,
        _client=NS(get_orders=lambda req: orders) if orders is not None else None,
    )

def run_tests():
    evidence = {}

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)

        # -------------------------------------------------------------
        # POINT 1: Broker-flat probe & actual position query (3 cases)
        # -------------------------------------------------------------
        # 1A: Broker qty = 12 -> POSITION_STILL_OPEN_ON_BROKER
        m1a = make_manager()
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tdp / "p1a.jsonl"
        b1a = make_bridge(pos_val=12.0, snapshot=dict(id='tp-1', status='filled', symbol='AAPL', side='sell', qty=12., filled_qty=12., filled_avg_price=335.10))
        r1a = G['reconcile_pending_exits'](m1a, b1a)
        evidence["point_1a_broker_qty_12_retained"] = {
            "result": r1a,
            "pending_count": len(m1a.pending_exits),
            "status": "PASS" if r1a.get("AAPL") == "POSITION_STILL_OPEN_ON_BROKER" and len(m1a.pending_exits) == 1 else "FAIL",
        }

        # 1B: Broker qty = 0 -> Reconcile when valid fill evidence exists
        m1b = make_manager()
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tdp / "p1b.jsonl"
        b1b = make_bridge(pos_val=0.0, snapshot=dict(id='tp-1', type='limit', status='filled', symbol='AAPL', side='sell', qty=12., filled_qty=12., filled_avg_price=335.10))
        r1b = G['reconcile_pending_exits'](m1b, b1b)
        evidence["point_1b_broker_qty_0_reconciled"] = {
            "result": r1b,
            "pending_count": len(m1b.pending_exits),
            "status": "PASS" if r1b.get("AAPL") == "RECONCILED_TAKE_PROFIT" and len(m1b.pending_exits) == 0 else "FAIL",
        }

        # 1C: Broker position timeout / None / NaN -> POSITION_UNKNOWN_RETAINED
        m1c1 = make_manager()
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tdp / "p1c1.jsonl"
        b1c1 = make_bridge(raise_exc=RuntimeError("connection timeout"))
        r1c1 = G['reconcile_pending_exits'](m1c1, b1c1)

        m1c2 = make_manager()
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tdp / "p1c2.jsonl"
        b1c2 = make_bridge(pos_val=None)
        r1c2 = G['reconcile_pending_exits'](m1c2, b1c2)

        m1c3 = make_manager()
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tdp / "p1c3.jsonl"
        b1c3 = make_bridge(pos_val=float("nan"))
        r1c3 = G['reconcile_pending_exits'](m1c3, b1c3)

        evidence["point_1c_broker_unknown_retained"] = {
            "timeout_result": r1c1,
            "none_result": r1c2,
            "nan_result": r1c3,
            "status": "PASS" if (
                r1c1.get("AAPL") == "POSITION_UNKNOWN_RETAINED"
                and r1c2.get("AAPL") == "POSITION_UNKNOWN_RETAINED"
                and r1c3.get("AAPL") == "POSITION_UNKNOWN_RETAINED"
            ) else "FAIL",
        }

        # -------------------------------------------------------------
        # POINT 2: fsync retry durability & duplicate prevention
        # -------------------------------------------------------------
        journal_p2 = tdp / "p2_retry.jsonl"
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = journal_p2
        m2 = make_manager()
        b2 = make_bridge(pos_val=0.0, snapshot=dict(id='tp-1', type='limit', status='filled', symbol='AAPL', side='sell', qty=12., filled_qty=12., filled_avg_price=335.10))

        # First attempt fails fsync
        with patch('os.fsync', side_effect=OSError("disk I/O error")):
            r2_1 = G['reconcile_pending_exits'](m2, b2)

        # Retry also fails fsync
        with patch('os.fsync', side_effect=OSError("disk I/O error")):
            r2_2 = G['reconcile_pending_exits'](m2, b2)

        lines_after_failures = journal_p2.read_text(encoding="utf-8").splitlines() if journal_p2.exists() else []

        # Third attempt: fsync recovers! Single record durably acknowledged
        r2_3 = G['reconcile_pending_exits'](m2, b2)
        lines_after_recovery = journal_p2.read_text(encoding="utf-8").splitlines() if journal_p2.exists() else []

        evidence["point_2_fsync_durability_and_dedup"] = {
            "first_failure_result": r2_1,
            "second_failure_result": r2_2,
            "lines_after_2_failures": len(lines_after_failures),
            "recovery_result": r2_3,
            "lines_after_recovery": len(lines_after_recovery),
            "pending_after_recovery": len(m2.pending_exits),
            "status": "PASS" if (
                len(lines_after_recovery) == 1
                and r2_3.get("AAPL") == "RECONCILED_TAKE_PROFIT"
                and len(m2.pending_exits) == 0
            ) else "FAIL",
        }

        # -------------------------------------------------------------
        # POINT 3: Split-exit aggregation, quantity conservation & duplicate orders
        # -------------------------------------------------------------
        # 3A: Cancelled stop with partial fill 4 + TP filled 8 = 12 total
        m3a = make_manager()
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tdp / "p3a.jsonl"
        orders_3a = [
            NS(id='stop-1', client_order_id='AAPL:sig-1', symbol='AAPL', side='sell', status='canceled', order_type='stop', qty=12., filled_qty=4., filled_avg_price=330.0),
            NS(id='tp-1', client_order_id='AAPL:sig-1', symbol='AAPL', side='sell', status='filled', order_type='limit', qty=8., filled_qty=8., filled_avg_price=335.0),
        ]
        b3a = make_bridge(pos_val=0.0, orders=orders_3a)
        r3a = G['reconcile_pending_exits'](m3a, b3a)
        evidence["point_3a_cancelled_partial_plus_tp_reconciled"] = {
            "result": r3a,
            "pending_count": len(m3a.pending_exits),
            "status": "PASS" if r3a.get("AAPL") == "RECONCILED_SPLIT_EXIT" and len(m3a.pending_exits) == 0 else "FAIL",
        }

        # 3B: Deduplication across snapshot and closed orders (both report tp-1 with 12 shares)
        m3b = make_manager()
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tdp / "p3b.jsonl"
        snap_3b = dict(id='tp-1', type='limit', status='filled', symbol='AAPL', side='sell', qty=12., filled_qty=12., filled_avg_price=335.0)
        orders_3b = [
            NS(id='tp-1', client_order_id='AAPL:sig-1', symbol='AAPL', side='sell', status='filled', order_type='limit', qty=12., filled_qty=12., filled_avg_price=335.0),
            NS(id='tp-1', client_order_id='AAPL:sig-1', symbol='AAPL', side='sell', status='filled', order_type='limit', qty=12., filled_qty=12., filled_avg_price=335.0), # duplicate in closed
        ]
        b3b = make_bridge(pos_val=0.0, snapshot=snap_3b, orders=orders_3b)
        r3b = G['reconcile_pending_exits'](m3b, b3b)
        evidence["point_3b_order_deduplication_exact_12"] = {
            "result": r3b,
            "pending_count": len(m3b.pending_exits),
            "status": "PASS" if r3b.get("AAPL") == "RECONCILED_TAKE_PROFIT" and len(m3b.pending_exits) == 0 else "FAIL",
        }

        # 3C: Excess quantity discrepancy (> 12 shares) triggers EXCESS_EXIT_QTY_DISCREPANCY fail-closed
        m3c = make_manager()
        G['config'].DAYTRADE_TRADE_JOURNAL_FILE = tdp / "p3c.jsonl"
        orders_3c = [
            NS(id='stop-1', client_order_id='AAPL:sig-1', symbol='AAPL', side='sell', status='filled', order_type='stop', qty=8., filled_qty=8., filled_avg_price=330.0),
            NS(id='tp-1', client_order_id='AAPL:sig-1', symbol='AAPL', side='sell', status='filled', order_type='limit', qty=8., filled_qty=8., filled_avg_price=335.0),
        ]
        b3c = make_bridge(pos_val=0.0, orders=orders_3c)
        r3c = G['reconcile_pending_exits'](m3c, b3c)
        evidence["point_3c_excess_qty_discrepancy_fail_closed"] = {
            "result": r3c,
            "pending_count": len(m3c.pending_exits),
            "status": "PASS" if r3c.get("AAPL") == "EXCESS_EXIT_QTY_DISCREPANCY" and len(m3c.pending_exits) == 1 else "FAIL",
        }

        # -------------------------------------------------------------
        # POINT 4: Stop adoption uses fresh signed broker position
        # -------------------------------------------------------------
        # 4A: Actual position 12, candidate stop capacity 8 -> REJECT
        s4a = NS(symbol='AAPL', side='BUY', active=True, order_id=None, qty=12., entry_price=331.88, stop_price=331.89, desired_stop_price=331.89, strategy='MEAN_REVERSION', reconciliation_status='POSITION_UNKNOWN_RECONCILIATION_REQUIRED', stop_reconciliation_required=True)
        m4a = NS(_lock=threading.RLock(), states={'AAPL': s4a}, pending_exits={}, _save_state=lambda: True, _extract_all_orders_including_legs=lambda x: x)
        m4a._is_valid_stop_order = types.MethodType(G['_is_valid_stop_order'], m4a)
        m4a._query_broker_position_qty = types.MethodType(G['_query_broker_position_qty'], m4a)
        candidate_stop_8 = {'id': 'stop-8', 'symbol': 'AAPL', 'side': 'sell', 'type': 'stop', 'status': 'new', 'qty': '8', 'filled_qty': '0', 'stop_price': '331.89'}
        b4a = make_bridge(pos_val=12.0)
        adop_4a = G['_reconcile_broker_order'](m4a, s4a, [candidate_stop_8], bridge=b4a)
        evidence["point_4a_actual_pos_12_stop_8_rejected"] = {
            "adopted": adop_4a,
            "order_id": s4a.order_id,
            "status": "PASS" if (not adop_4a and s4a.order_id is None) else "FAIL",
        }

        # 4B: Actual position 8, candidate stop capacity 8 -> ACCEPT
        s4b = NS(symbol='AAPL', side='BUY', active=True, order_id=None, qty=12., entry_price=331.88, stop_price=331.89, desired_stop_price=331.89, strategy='MEAN_REVERSION', reconciliation_status='POSITION_UNKNOWN_RECONCILIATION_REQUIRED', stop_reconciliation_required=True)
        m4b = NS(_lock=threading.RLock(), states={'AAPL': s4b}, pending_exits={}, _save_state=lambda: True, _extract_all_orders_including_legs=lambda x: x)
        m4b._is_valid_stop_order = types.MethodType(G['_is_valid_stop_order'], m4b)
        m4b._query_broker_position_qty = types.MethodType(G['_query_broker_position_qty'], m4b)
        b4b = make_bridge(pos_val=8.0)
        adop_4b = G['_reconcile_broker_order'](m4b, s4b, [candidate_stop_8], bridge=b4b)
        evidence["point_4b_actual_pos_8_stop_8_accepted"] = {
            "adopted": adop_4b,
            "order_id": s4b.order_id,
            "state_qty": s4b.qty,
            "unknown_cleared": s4b.reconciliation_status is None,
            "status": "PASS" if (adop_4b and s4b.order_id == "stop-8" and s4b.qty == 8.0 and s4b.reconciliation_status is None) else "FAIL",
        }

        # 4C: Stop found but broker position UNKNOWN -> Do NOT clear POSITION_UNKNOWN_RECONCILIATION_REQUIRED
        s4c = NS(symbol='AAPL', side='BUY', active=True, order_id=None, qty=8., entry_price=331.88, stop_price=331.89, desired_stop_price=331.89, strategy='MEAN_REVERSION', reconciliation_status='POSITION_UNKNOWN_RECONCILIATION_REQUIRED', stop_reconciliation_required=True)
        m4c = NS(_lock=threading.RLock(), states={'AAPL': s4c}, pending_exits={}, _save_state=lambda: True, _extract_all_orders_including_legs=lambda x: x)
        m4c._is_valid_stop_order = types.MethodType(G['_is_valid_stop_order'], m4c)
        m4c._query_broker_position_qty = types.MethodType(G['_query_broker_position_qty'], m4c)
        b4c = make_bridge(raise_exc=RuntimeError("broker timeout"))
        adop_4c = G['_reconcile_broker_order'](m4c, s4c, [candidate_stop_8], bridge=b4c)
        evidence["point_4c_position_unknown_retains_unknown_status"] = {
            "adopted": adop_4c,
            "order_id": s4c.order_id,
            "unknown_status": s4c.reconciliation_status,
            "status": "PASS" if (adop_4c and s4c.order_id == "stop-8" and s4c.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED") else "FAIL",
        }

    return evidence

if __name__ == '__main__':
    res = run_tests()
    print(json.dumps(res, indent=2))
    all_pass = all(v.get("status") == "PASS" for v in res.values())
    sys.exit(0 if all_pass else 1)
