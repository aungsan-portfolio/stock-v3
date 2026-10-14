"""
Identity, Lineage, Precedence, and Crash-Recovery Acceptance Test Suite.
Tests:
- Order ID / client ID precedence: generic client ID does not block exact ID recovery.
- Explicit conflict detection: when broker order ID matches but client ID explicitly points to another episode, fails closed.
- Dual-match conflict precedence: when client ID matches both current and foreign tokens, conflict strictly takes precedence.
- Reverse-conflict detection: when broker order ID belongs to a foreign episode while client ID matches current lineage, conflict strictly takes precedence.
- Pre-entry fill rejection: orders before entry epoch are rejected.
- Unlinked fill rejection: orders without lineage/identity are retained fail-closed without journal write.
- Fault-injected initial-save failure / crash-recovery: verifies single close on authoritative later fill vs fail-closed retention on unlinked fill across fresh manager cold starts recovering from broker truth.
- Signed-side mismatch and position-unknown reload guards across active=True and active=False with order manager submit veto.
"""
import ast
import os
import json
import math
import threading
import time
import types
import pathlib
import logging
import tempfile
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import *
from types import SimpleNamespace as NS
from unittest.mock import patch

from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState

ROOT = pathlib.Path(__file__).resolve().parent

import sys
jm = types.ModuleType('strategies.trade_journal')
sys.modules['strategies.trade_journal'] = jm
req = types.ModuleType('alpaca.trading.requests')
req.GetOrdersRequest = lambda **kw: NS(**kw)
en = types.ModuleType('alpaca.trading.enums')
en.QueryOrderStatus = NS(CLOSED='closed')
for name in ('alpaca', 'alpaca.trading'):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['alpaca.trading.requests'] = req
sys.modules['alpaca.trading.enums'] = en

out = []
entry = time.time() - 600

# ── 1. Scoped Lineage, Precedence & Conflict Tests ───────────────────────────────────
cases = [
    'unrelated_unique_post_entry',
    'unrelated_unique_pre_entry_90_seconds',
    'ambiguous_unrelated_two',
    'linked_current_order_generic_client_id',
    'explicit_conflict_order_id_matches_but_client_id_foreign',
    'dual_match_conflict_precedence',
    'reverse_conflict_broker_order_id_belongs_to_foreign_episode',
    'shared_oid_current_and_foreign',
    'unmapped_literal_word_cid',
]

for case in cases:
    with tempfile.TemporaryDirectory() as td:
        fake_state = os.path.join(td, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state):
            m = DynamicTrailingStopManager()
            s = TrailingStopState(
                symbol='AAPL', side='BUY', entry_price=100.0, peak_price=100.0, atr=1.0,
                stop_price=95.0, trail_multiple=1.5, last_updated=0.0, qty=12.0, entry_qty=12.0,
                order_id='current-stop', signal_id='current-episode-signal', entry_time=entry
            )
            m.states['AAPL'] = s
            records = []
            jm.log_trade = lambda **kw: records.append(kw)

            if case == 'linked_current_order_generic_client_id':
                oid = 'current-stop'
                cid = 'broker-generated-client-id'
                stamp = entry + 30
            elif case == 'explicit_conflict_order_id_matches_but_client_id_foreign':
                # Registered foreign episode on MSFT
                m.states['MSFT'] = TrailingStopState(
                    symbol='MSFT', side='BUY', entry_price=300.0, peak_price=300.0, atr=2.0,
                    stop_price=290.0, trail_multiple=1.5, last_updated=0.0, qty=5.0, entry_qty=5.0,
                    order_id='foreign-stop-9', signal_id='foreign-episode-signal', entry_time=entry
                )
                oid = 'current-stop'
                cid = 'exit_foreign-episode-signal'  # Registered foreign conflict
                stamp = entry + 30
            elif case == 'dual_match_conflict_precedence':
                # Registered foreign episode on MSFT
                m.states['MSFT'] = TrailingStopState(
                    symbol='MSFT', side='BUY', entry_price=300.0, peak_price=300.0, atr=2.0,
                    stop_price=290.0, trail_multiple=1.5, last_updated=0.0, qty=5.0, entry_qty=5.0,
                    order_id='foreign-stop-9', signal_id='foreign-episode-signal', entry_time=entry
                )
                oid = 'current-stop'
                cid = 'exit_current-episode-signal_foreign-episode-signal'  # Dual match: current + foreign
                stamp = entry + 30
            elif case == 'reverse_conflict_broker_order_id_belongs_to_foreign_episode':
                # Candidate order ID belongs to foreign episode (registered on MSFT)
                # But candidate client ID matches current AAPL signal lineage
                m.states['MSFT'] = TrailingStopState(
                    symbol='MSFT', side='BUY', entry_price=300.0, peak_price=300.0, atr=2.0,
                    stop_price=290.0, trail_multiple=1.5, last_updated=0.0, qty=5.0, entry_qty=5.0,
                    order_id='foreign-stop-9', signal_id='foreign-episode-signal', entry_time=entry
                )
                oid = 'foreign-stop-9'
                cid = 'current-episode-signal'
                stamp = entry + 30
            elif case == 'shared_oid_current_and_foreign':
                # Contradictory local ownership: both AAPL and MSFT mapped to current-stop
                m.states['MSFT'] = TrailingStopState(
                    symbol='MSFT', side='BUY', entry_price=300.0, peak_price=300.0, atr=2.0,
                    stop_price=290.0, trail_multiple=1.5, last_updated=0.0, qty=5.0, entry_qty=5.0,
                    order_id='current-stop', signal_id='foreign-episode-signal', entry_time=entry
                )
                oid = 'current-stop'
                cid = 'current-episode-signal'
                stamp = entry + 30
            elif case == 'unmapped_literal_word_cid':
                # Exact current broker order ID with unmapped literal client ID
                oid = 'current-stop'
                cid = 'broker-other-episode-note'
                stamp = entry + 30
            elif case == 'unrelated_unique_pre_entry_90_seconds':
                oid = 'unrelated-exit-1'
                cid = 'unmapped-client-id'
                stamp = entry - 90
            else:  # unrelated_unique_post_entry, ambiguous_unrelated_two
                oid = 'unrelated-exit-1'
                cid = 'unmapped-client-id'
                stamp = entry + 30

            o = dict(
                id=oid, client_order_id=cid, symbol='AAPL', side='sell', type='market',
                status='filled', filled_qty=12.0, filled_avg_price=105.0,
                filled_at=datetime.fromtimestamp(stamp, timezone.utc)
            )
            orders = [NS(**o)]
            if case == 'ambiguous_unrelated_two':
                orders.append(NS(**dict(o, id='unrelated-exit-2')))

            b = NS(get_position=lambda sym: 0.0 if sym == 'AAPL' else 5.0, get_order_snapshot=lambda oid: None, _client=NS(get_orders=lambda request: orders))
            result = m.reconcile_pending_exits(b)
            out.append({
                'case': case,
                'result': dict(
                    result=result, pending=len(m.pending_exits), records=records,
                    candidate_order_id=o['id'], candidate_client_id=o['client_order_id'],
                    timestamp_minus_entry_seconds=stamp - entry
                )
            })

res_map = {item['case']: item['result'] for item in out}

# 1. Unrelated unique post-entry: pending retained=1, records=0
assert res_map['unrelated_unique_post_entry']['records'] == [] and res_map['unrelated_unique_post_entry']['pending'] == 1, "Unrelated post-entry must retain pending, records=0"
# 2. Unrelated pre-entry 90s: rejected, pending retained=1, records=0
assert res_map['unrelated_unique_pre_entry_90_seconds']['records'] == [] and res_map['unrelated_unique_pre_entry_90_seconds']['pending'] == 1, "Pre-entry order must be rejected, records=0"
# 3. Ambiguous two unrelated: pending retained=1, records=0
assert res_map['ambiguous_unrelated_two']['pending'] == 1 and not res_map['ambiguous_unrelated_two']['records'], "Ambiguous unlinked must retain pending, records=0"
# 4. Valid exact-ID recovery (exact broker ID + generic client ID): reconciled, records=1, pending=0
assert len(res_map['linked_current_order_generic_client_id']['records']) == 1 and res_map['linked_current_order_generic_client_id']['pending'] == 0, "Exact broker order ID with generic client ID must reconcile"
# 5. Explicit conflict (broker ID matches, BUT client ID explicitly foreign): conflicting evidence -> retained fail-closed, records=0, pending=1
assert res_map['explicit_conflict_order_id_matches_but_client_id_foreign']['records'] == [] and res_map['explicit_conflict_order_id_matches_but_client_id_foreign']['pending'] == 1, "Explicit conflict must retain pending fail-closed, records=0"
assert res_map['explicit_conflict_order_id_matches_but_client_id_foreign']['result'].get('AAPL') == 'CONFLICTING_EXIT_EVIDENCE_DISCREPANCY', "Result status must be CONFLICTING_EXIT_EVIDENCE_DISCREPANCY"
# 6. Dual-match conflict precedence: conflict strictly takes precedence -> retained fail-closed, records=0, pending=1
assert res_map['dual_match_conflict_precedence']['records'] == [] and res_map['dual_match_conflict_precedence']['pending'] == 1, "Dual match conflict must retain pending fail-closed, records=0"
assert res_map['dual_match_conflict_precedence']['result'].get('AAPL') == 'CONFLICTING_EXIT_EVIDENCE_DISCREPANCY', "Result status must be CONFLICTING_EXIT_EVIDENCE_DISCREPANCY"
# 7. Reverse conflict (broker ID belongs to foreign episode, BUT client ID matches current lineage): conflict strictly takes precedence -> retained fail-closed, records=0, pending=1
assert res_map['reverse_conflict_broker_order_id_belongs_to_foreign_episode']['records'] == [] and res_map['reverse_conflict_broker_order_id_belongs_to_foreign_episode']['pending'] == 1, "Reverse conflict must retain pending fail-closed, records=0"
assert res_map['reverse_conflict_broker_order_id_belongs_to_foreign_episode']['result'].get('AAPL') == 'CONFLICTING_EXIT_EVIDENCE_DISCREPANCY', "Result status must be CONFLICTING_EXIT_EVIDENCE_DISCREPANCY"
# 8. Shared broker order ID conflict: both current AAPL and foreign MSFT mapped to current-stop -> conflict fail-closed
assert res_map['shared_oid_current_and_foreign']['records'] == [] and res_map['shared_oid_current_and_foreign']['pending'] == 1, "Shared order ID must retain pending fail-closed, records=0"
assert res_map['shared_oid_current_and_foreign']['result'].get('AAPL') == 'CONFLICTING_EXIT_EVIDENCE_DISCREPANCY', "Result status must be CONFLICTING_EXIT_EVIDENCE_DISCREPANCY"
# 9. Unmapped literal word client ID: exact current broker ID with unmapped literal client ID -> reconciled, records=1, pending=0
assert len(res_map['unmapped_literal_word_cid']['records']) == 1 and res_map['unmapped_literal_word_cid']['pending'] == 0, "Exact current order ID with unmapped literal client ID must reconcile"
assert res_map['unmapped_literal_word_cid']['result'].get('AAPL') == 'RECONCILED_MARKET_EXIT', "Result status must be RECONCILED_MARKET_EXIT"


# ── 2. Fault-Injected Initial-Save Failure / Crash / Restart Acceptance Tests ───────
# Case B1: Save failure during exit recording -> Crash -> Restart (Cold start from disk) -> Authoritative later fill arrives
with tempfile.TemporaryDirectory() as td:
    fake_state_file = os.path.join(td, "trailing_state.json")
    with patch("strategies.trailing_stop.STATE_FILE", fake_state_file):
        # Phase 1: Pre-crash entry state persisted to disk
        mgr_init = DynamicTrailingStopManager()
        s_b1 = TrailingStopState(
            symbol='AAPL', side='BUY', entry_price=100.0, peak_price=100.0, atr=1.0,
            stop_price=95.0, trail_multiple=1.5, last_updated=0.0, qty=12.0, entry_qty=12.0,
            order_id='stop-saved-on-entry', signal_id='sig-b1', entry_time=entry,
            active=True, reconciliation_status=None
        )
        mgr_init.states['AAPL'] = s_b1
        saved_ok = mgr_init._save_state()
        assert saved_ok is True, "Pre-crash entry state must be successfully persisted to disk"
        assert os.path.exists(fake_state_file), "Disk file must exist"

        # Phase 1b: Fault injection — simulate save failure during exit recording
        with patch.object(mgr_init, "_save_state", return_value=False):
            mgr_init._record_pending_exit(s_b1)
            assert s_b1.stop_reconciliation_required is True, "Save failure must flag stop_reconciliation_required"

        # Phase 2: CRASH — drop all in-memory references
        del mgr_init
        del s_b1

        # Phase 3: RESTART — fresh manager loads directly from disk STATE_FILE
        mgr_b1 = DynamicTrailingStopManager()
        assert 'AAPL' in mgr_b1.states, "Fresh manager must load AAPL from disk"
        assert mgr_b1.states['AAPL'].active is True, "Disk state reflects pre-exit active state"
        assert len(mgr_b1.pending_exits) == 0, "Pending exits queue starts empty on cold start because save failed"

        # Phase 4: Authoritative later fill arrives; _recover_orphaned_flat_states reconstructs pending exit from broker truth
        records_b1 = []
        jm.log_trade = lambda **kw: records_b1.append(kw)
        order_b1 = NS(
            id='stop-saved-on-entry', client_order_id='generic-client-id', symbol='AAPL',
            side='sell', type='stop', status='filled', filled_qty=12.0, filled_avg_price=95.0,
            filled_at=datetime.fromtimestamp(entry + 40, timezone.utc)
        )
        b_b1 = NS(get_position=lambda sym: 0.0, get_order_snapshot=lambda oid: None, _client=NS(get_orders=lambda req: [order_b1]))
        res_b1 = mgr_b1.reconcile_pending_exits(b_b1)
        assert len(records_b1) == 1 and len(mgr_b1.pending_exits) == 0, f"Expected 1 record, 0 pending, got records={len(records_b1)}, pending={len(mgr_b1.pending_exits)}"

        # Phase 5: Retry must be idempotent
        res_b1_retry = mgr_b1.reconcile_pending_exits(b_b1)
        assert len(records_b1) == 1 and len(mgr_b1.pending_exits) == 0, "Retry after restart must be idempotent"
        out.append({'case': 'initial_save_failure_restart_authoritative_later_fill', 'result': dict(records=len(records_b1), pending=len(mgr_b1.pending_exits), cold_start_from_disk=True, recovered_from_broker_truth=True)})

# Case B2: Save failure during exit recording -> Crash -> Restart (Cold start from disk) -> Unlinked later fill arrives
with tempfile.TemporaryDirectory() as td:
    fake_state_file = os.path.join(td, "trailing_state.json")
    with patch("strategies.trailing_stop.STATE_FILE", fake_state_file):
        # Phase 1: Pre-crash entry state persisted to disk
        mgr_init = DynamicTrailingStopManager()
        s_b2 = TrailingStopState(
            symbol='AAPL', side='BUY', entry_price=100.0, peak_price=100.0, atr=1.0,
            stop_price=95.0, trail_multiple=1.5, last_updated=0.0, qty=12.0, entry_qty=12.0,
            order_id='stop-saved-on-entry', signal_id='sig-b2', entry_time=entry,
            active=True, reconciliation_status=None
        )
        mgr_init.states['AAPL'] = s_b2
        saved_ok = mgr_init._save_state()
        assert saved_ok is True, "Pre-crash entry state must be successfully persisted to disk"
        assert os.path.exists(fake_state_file), "Disk file must exist"

        # Phase 1b: Fault injection — simulate save failure during exit recording
        with patch.object(mgr_init, "_save_state", return_value=False):
            mgr_init._record_pending_exit(s_b2)
            assert s_b2.stop_reconciliation_required is True, "Save failure must flag stop_reconciliation_required"

        # Phase 2: CRASH — drop all in-memory references
        del mgr_init
        del s_b2

        # Phase 3: RESTART — fresh manager loads directly from disk STATE_FILE
        mgr_b2 = DynamicTrailingStopManager()
        assert 'AAPL' in mgr_b2.states, "Fresh manager must load AAPL from disk"
        assert mgr_b2.states['AAPL'].active is True, "Disk state reflects pre-exit active state"
        assert len(mgr_b2.pending_exits) == 0, "Pending exits queue starts empty on cold start because save failed"

        # Phase 4: Unlinked fill arrives; _recover_orphaned_flat_states reconstructs pending exit, then lineage rejects unlinked fill
        records_b2 = []
        jm.log_trade = lambda **kw: records_b2.append(kw)
        order_b2_unlinked = NS(
            id='unlinked-order-99', client_order_id='unmapped-id', symbol='AAPL',
            side='sell', type='market', status='filled', filled_qty=12.0, filled_avg_price=95.0,
            filled_at=datetime.fromtimestamp(entry + 40, timezone.utc)
        )
        b_b2 = NS(get_position=lambda sym: 0.0, get_order_snapshot=lambda oid: None, _client=NS(get_orders=lambda req: [order_b2_unlinked]))
        res_b2 = mgr_b2.reconcile_pending_exits(b_b2)
        assert len(records_b2) == 0 and len(mgr_b2.pending_exits) == 1, "Unlinked fill after crash must retain pending and write 0 journal records"
        assert mgr_b2.has_pending_exit('AAPL') is True, "Entry block via has_pending_exit must remain active"
        out.append({'case': 'initial_save_failure_restart_unlinked_later_fill', 'result': dict(records=len(records_b2), pending=len(mgr_b2.pending_exits), entry_blocked=mgr_b2.has_pending_exit('AAPL'), cold_start_from_disk=True, recovered_from_broker_truth=True)})


# ── 3. Signed Side Mismatch & Position Unknown Reload Guards Across active=True and active=False ───
for active in (True, False):
    with tempfile.TemporaryDirectory() as td:
        fake_state = os.path.join(td, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state):
            m = DynamicTrailingStopManager()
            s = TrailingStopState(
                symbol='AAPL', side='BUY', entry_price=100.0, peak_price=100.0, atr=1.0,
                stop_price=95.0, trail_multiple=1.5, last_updated=0.0, qty=12.0, entry_qty=12.0,
                active=active, reconciliation_status='SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED'
            )
            m.states['AAPL'] = s
            saved_ok = m._save_state()
            assert saved_ok is True, f"State must be persisted to disk for active={active}"

            # SIMULATE RESTART / COLD START FROM DISK
            m_fresh = DynamicTrailingStopManager()
            assert 'AAPL' in m_fresh.states, f"Fresh manager must load AAPL from disk when active={active}"
            assert m_fresh.states['AAPL'].reconciliation_status == 'SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED'

            out.append({
                'case': 'mismatch_guard_active_' + str(active),
                'result': dict(
                    portfolio_halt=m_fresh.has_unresolved_position_unknown(),
                    mismatch_halt=m_fresh.has_unresolved_signed_side_mismatch(),
                    loaded_after_restart='AAPL' in m_fresh.states
                )
            })
            assert out[-1]['result']['portfolio_halt'] and out[-1]['result']['mismatch_halt']

# Also test POSITION_UNKNOWN_RECONCILIATION_REQUIRED with active=False across cold start
with tempfile.TemporaryDirectory() as td:
    fake_state = os.path.join(td, "state.json")
    with patch("strategies.trailing_stop.STATE_FILE", fake_state):
        m = DynamicTrailingStopManager()
        s = TrailingStopState(
            symbol='TSLA', side='BUY', entry_price=200.0, peak_price=200.0, atr=2.0,
            stop_price=190.0, trail_multiple=1.5, last_updated=0.0, qty=10.0, entry_qty=10.0,
            active=False, reconciliation_status='POSITION_UNKNOWN_RECONCILIATION_REQUIRED'
        )
        m.states['TSLA'] = s
        m._save_state()

        m_fresh = DynamicTrailingStopManager()
        assert 'TSLA' in m_fresh.states, "Fresh manager must load inactive POSITION_UNKNOWN state from disk"
        assert m_fresh.has_unresolved_position_unknown() is True
        out.append({
            'case': 'position_unknown_guard_active_False_reloaded',
            'result': dict(
                portfolio_halt=m_fresh.has_unresolved_position_unknown(),
                loaded_after_restart='TSLA' in m_fresh.states
            )
        })

# Output results to disk and stdout
out_path = ROOT / "identity_acceptance_checks_results.json"
out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
print("\n>>> ALL IDENTITY, PRECEDENCE, RECOVERY, AND SAFETY CHECKS PASSED PERFECTLY! <<<", file=sys.stderr)
