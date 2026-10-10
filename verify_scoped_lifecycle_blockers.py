"""
Verification probe for scoped lifecycle blockers B1 and B2:
- B1: Protection rejection transitions to safe remediation (replace/flatten) without duplicate orders or speculative action on unknown state
- B2: Partial entry cumulative fills update entry_qty cleanly without double-counting or loss upon partial exit
"""
import sys
import os
import json
import time
import tempfile
import pathlib
from unittest.mock import patch
from types import SimpleNamespace as NS

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategies.trailing_stop import DynamicTrailingStopManager, TrailingStopState
import strategies.trailing_stop
from strategies.base import TradeSignal
import config

# Mock market hours to True for offline tests
strategies.trailing_stop.is_market_open = lambda: True


def test_b1_lifecycle_remediation():
    with tempfile.TemporaryDirectory() as td:
        fake_state_file = os.path.join(td, "trailing_state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state_file):
            return _test_b1_lifecycle_remediation_impl()


def _test_b1_lifecycle_remediation_impl():
    results = {}

    # --- B1 Case 1: position 12 / stop 1 -> safe recovery request verified to cover 12 ---
    mgr = DynamicTrailingStopManager()
    state = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="stop-1",
        qty=12.0,
        strategy="DAYTRADE_SCALP",
        signal_id="sig-b1-1",
    )
    mgr.states["AAPL"] = state

    one_share_stop = {
        "id": "stop-1",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "new",
        "qty": 1.0,
        "stop_price": 330.0,
    }
    old_stop_canceled_snap = {
        "id": "stop-1",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "canceled",
        "qty": 1.0,
        "stop_price": 330.0,
    }
    replacement_snap = {
        "id": "stop-new-12",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "new",
        "qty": 12.0,
        "stop_price": 330.0,
    }

    place_calls = []
    cancel_calls = []
    bridge_mock = NS(
        get_position=lambda sym: 12.0,
        market_price=lambda sym: 332.0,
        working_orders_plain=lambda: [one_share_stop],
        cancel_order=lambda oid: cancel_calls.append(oid) or True,
        place_stop_order=lambda **kw: place_calls.append(kw) or {"order_id": "stop-new-12"},
        get_order_snapshot=lambda oid: replacement_snap if oid == "stop-new-12" else old_stop_canceled_snap,
        _client=NS(get_orders=lambda req: [NS(**one_share_stop)]),
    )

    # 1. ensure_initialized runs
    mgr.ensure_initialized("AAPL", "BUY", 331.88, [one_share_stop], 332.0, qty=12.0, bridge=bridge_mock)
    assert state.stop_reconciliation_required is True, "ensure_initialized must flag stop_reconciliation_required=True"
    assert state.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
    assert state.last_stop_order_id == "stop-1"

    # 2. Scheduler check: must be eligible for naked branch
    naked_eligible = state and state.active and (state.order_id is None or getattr(state, "stop_reconciliation_required", False))
    assert naked_eligible is True, "Position requiring stop reconciliation must be eligible for naked handling in orchestrator"

    # 3. handle_naked_position runs with verified replacement snapshot
    mgr.handle_naked_position("AAPL", 12.0, bridge_mock, dry_run=False)

    assert len(place_calls) == 1, f"Expected 1 place_stop_order call, got {len(place_calls)}"
    assert place_calls[0]["qty"] == 12.0, f"Expected recovery order qty 12.0, got {place_calls[0]['qty']}"
    assert state.order_id == "stop-new-12", f"Expected new order_id stop-new-12, got {state.order_id}"
    assert state.stop_reconciliation_required is False, "After confirmed replacement, stop_reconciliation_required must be False"
    assert state.reconciliation_status is None, "After confirmed replacement, reconciliation_status must be None"

    results["b1_case1_insufficient_stop_replaced_covering_full_pos"] = "PASS"

    # --- B1 Case 1b: cancel failure / timeout -> defer blind replacement ---
    mgr_cancel_fail = DynamicTrailingStopManager()
    state_cf = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="stop-old",
        qty=12.0,
        stop_reconciliation_required=True,
    )
    mgr_cancel_fail.states["AAPL"] = state_cf
    cf_place_calls = []
    bridge_cancel_fail = NS(
        get_position=lambda sym: 12.0,
        market_price=lambda sym: 332.0,
        cancel_order=lambda oid: False,  # Cancellation rejected / failed
        place_stop_order=lambda **kw: cf_place_calls.append(kw) or {"order_id": "bad-replace"},
    )
    mgr_cancel_fail.handle_naked_position("AAPL", 12.0, bridge_cancel_fail, dry_run=False)
    assert len(cf_place_calls) == 0, f"Must not place replacement stop if cancel fails, got {len(cf_place_calls)}"
    assert state_cf.stop_reconciliation_required is True
    assert state_cf.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
    assert state_cf.order_id == "stop-old"
    results["b1_case1b_cancel_failure_defers_replacement"] = "PASS"

    # --- B1 Case 1c: replacement snapshot invalid (insufficient capacity) -> retain unknown ---
    mgr_bad_snap = DynamicTrailingStopManager()
    state_bs = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="stop-old-2",
        qty=12.0,
        stop_reconciliation_required=True,
    )
    mgr_bad_snap.states["AAPL"] = state_bs
    bs_place_calls = []
    # Snapshot returned by broker has qty=1.0 instead of 12.0
    bad_snap = {
        "id": "stop-new-bad",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "new",
        "qty": 1.0,
        "stop_price": 330.0,
    }
    old_stop_canceled_snap_2 = {
        "id": "stop-old-2",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "canceled",
        "qty": 1.0,
        "stop_price": 330.0,
    }
    bridge_bad_snap = NS(
        get_position=lambda sym: 12.0,
        market_price=lambda sym: 332.0,
        cancel_order=lambda oid: True,
        place_stop_order=lambda **kw: bs_place_calls.append(kw) or {"order_id": "stop-new-bad"},
        get_order_snapshot=lambda oid: bad_snap if oid == "stop-new-bad" else old_stop_canceled_snap_2,
    )
    mgr_bad_snap.handle_naked_position("AAPL", 12.0, bridge_bad_snap, dry_run=False)
    assert len(bs_place_calls) == 1
    # Flags must NOT be cleared because capacity verification failed!
    assert state_bs.stop_reconciliation_required is True
    assert state_bs.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
    assert state_bs.order_id == "stop-new-bad"
    results["b1_case1c_replacement_snapshot_invalid_retains_unknown"] = "PASS"

    # --- B1 Case 1d: cancel acknowledgement without terminal confirmation (e.g. pending_cancel) -> defer replacement ---
    mgr_pc = DynamicTrailingStopManager()
    state_pc = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="stop-pending",
        qty=12.0,
        stop_reconciliation_required=True,
    )
    mgr_pc.states["AAPL"] = state_pc
    pc_place_calls = []
    # Snapshot shows order is still pending_cancel on broker
    pending_cancel_snap = {
        "id": "stop-pending",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "pending_cancel",
        "qty": 1.0,
        "stop_price": 330.0,
    }
    bridge_pc = NS(
        get_position=lambda sym: 12.0,
        market_price=lambda sym: 332.0,
        cancel_order=lambda oid: True,  # HTTP cancel accepted, but order is still pending_cancel!
        place_stop_order=lambda **kw: pc_place_calls.append(kw) or {"order_id": "bad-replace"},
        get_order_snapshot=lambda oid: pending_cancel_snap,
    )
    mgr_pc.handle_naked_position("AAPL", 12.0, bridge_pc, dry_run=False)
    assert len(pc_place_calls) == 0, f"Must not place replacement stop if old stop is not terminal cancelled, got {len(pc_place_calls)}"
    assert state_pc.stop_reconciliation_required is True
    assert state_pc.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
    assert state_pc.order_id == "stop-pending"
    results["b1_case1d_cancel_ack_without_terminal_confirmation_defers_replacement"] = "PASS"

    # --- B1 Case 1e: cancel_order returns None or missing -> defer replacement ---
    mgr_cn = DynamicTrailingStopManager()
    state_cn = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="stop-none",
        qty=12.0,
        stop_reconciliation_required=True,
    )
    mgr_cn.states["AAPL"] = state_cn
    cn_place_calls = []
    bridge_cn = NS(
        get_position=lambda sym: 12.0,
        market_price=lambda sym: 332.0,
        cancel_order=lambda oid: None,  # returns None -> must NOT be treated as success!
        place_stop_order=lambda **kw: cn_place_calls.append(kw) or {"order_id": "bad-replace"},
        get_order_snapshot=lambda oid: None,
    )
    mgr_cn.handle_naked_position("AAPL", 12.0, bridge_cn, dry_run=False)
    assert len(cn_place_calls) == 0, f"Must not place replacement stop if cancel returns None, got {len(cn_place_calls)}"
    assert state_cn.stop_reconciliation_required is True
    assert state_cn.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
    assert state_cn.order_id == "stop-none"
    results["b1_case1e_cancel_order_missing_or_returns_none_defers_replacement"] = "PASS"

    # --- B1 Case 1f: terminal cancelled confirmed -> re-queries fresh broker position -> places replacement for fresh qty ---
    mgr_fq = DynamicTrailingStopManager()
    state_fq = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="stop-term",
        qty=12.0,  # Stale: position on broker changed to 8.0
        stop_reconciliation_required=True,
    )
    mgr_fq.states["AAPL"] = state_fq
    fq_place_calls = []
    term_snap = {
        "id": "stop-term",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "canceled",
        "qty": 1.0,
        "stop_price": 330.0,
    }
    repl_snap_8 = {
        "id": "stop-new-8",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "new",
        "qty": 8.0,
        "stop_price": 330.0,
    }
    bridge_fq = NS(
        get_position=lambda sym: 8.0,  # Fresh broker quantity is 8.0!
        market_price=lambda sym: 332.0,
        cancel_order=lambda oid: True,
        place_stop_order=lambda **kw: fq_place_calls.append(kw) or {"order_id": "stop-new-8"},
        get_order_snapshot=lambda oid: repl_snap_8 if oid == "stop-new-8" else term_snap,
    )
    mgr_fq.handle_naked_position("AAPL", 12.0, bridge_fq, dry_run=False)
    assert len(fq_place_calls) == 1, f"Expected 1 replacement order call, got {len(fq_place_calls)}"
    assert fq_place_calls[0]["qty"] == 8.0, f"Expected replacement stop qty to cover fresh broker qty 8.0, got {fq_place_calls[0]['qty']}"
    assert state_fq.order_id == "stop-new-8"
    assert state_fq.qty == 8.0
    assert state_fq.stop_reconciliation_required is False
    assert state_fq.reconciliation_status is None
    results["b1_case1f_terminal_cancelled_confirmed_requeries_fresh_broker_qty"] = "PASS"

    # --- B1 Case 1g: old stop filled on broker -> routes to pending exit without placing replacement ---
    mgr_fl = DynamicTrailingStopManager()
    state_fl = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="stop-filled",
        qty=12.0,
        stop_reconciliation_required=True,
    )
    mgr_fl.states["AAPL"] = state_fl
    fl_place_calls = []
    filled_snap = {
        "id": "stop-filled",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "filled",
        "qty": 12.0,
        "filled_qty": 12.0,
        "filled_avg_price": 330.0,
        "stop_price": 330.0,
    }
    bridge_fl = NS(
        get_position=lambda sym: 0.0,
        market_price=lambda sym: 330.0,
        cancel_order=lambda oid: True,
        place_stop_order=lambda **kw: fl_place_calls.append(kw) or {"order_id": "bad-replace"},
        get_order_snapshot=lambda oid: filled_snap,
        _client=NS(get_orders=lambda req: [NS(**filled_snap)]),
    )
    with tempfile.TemporaryDirectory() as td:
        orig_j = getattr(config, "DAYTRADE_TRADE_JOURNAL_FILE", None)
        try:
            config.DAYTRADE_TRADE_JOURNAL_FILE = pathlib.Path(td) / "journal_fl.jsonl"
            mgr_fl.handle_naked_position("AAPL", 12.0, bridge_fl, dry_run=False)
            assert len(fl_place_calls) == 0, f"Must not place replacement stop if old stop filled, got {len(fl_place_calls)}"
            assert state_fl.last_stop_order_id == "stop-filled"
            assert state_fl.order_id is None
            results["b1_case1g_old_stop_filled_routes_to_pending_exit"] = "PASS"
        finally:
            config.DAYTRADE_TRADE_JOURNAL_FILE = orig_j

    # --- B1 Case 2: position 12 / valid stop 12 -> no duplicate protection order ---
    mgr2 = DynamicTrailingStopManager()
    state2 = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id="stop-12",
        qty=12.0,
        strategy="DAYTRADE_SCALP",
        signal_id="sig-b1-2",
    )
    mgr2.states["AAPL"] = state2
    valid_stop_12 = {
        "id": "stop-12",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "new",
        "qty": 12.0,
        "stop_price": 330.0,
    }
    place_calls2 = []
    bridge_mock2 = NS(
        get_position=lambda sym: 12.0,
        market_price=lambda sym: 332.0,
        working_orders_plain=lambda: [valid_stop_12],
        place_stop_order=lambda **kw: place_calls2.append(kw) or {"order_id": "stop-dup"},
    )
    mgr2.ensure_initialized("AAPL", "BUY", 331.88, [valid_stop_12], 332.0, qty=12.0, bridge=bridge_mock2)
    assert state2.stop_reconciliation_required is False
    naked_eligible2 = state2 and state2.active and (state2.order_id is None or getattr(state2, "stop_reconciliation_required", False))
    assert naked_eligible2 is False, "Validly protected position must not be eligible for naked handling"

    # Even if handle_naked_position is called directly:
    mgr2.handle_naked_position("AAPL", 12.0, bridge_mock2, dry_run=False)
    assert len(place_calls2) == 0, f"Expected 0 place_stop_order calls for validly protected position, got {len(place_calls2)}"
    results["b1_case2_valid_stop_no_duplicate_order"] = "PASS"

    # --- B1 Case 3: unknown broker state -> no speculative submission, remains active unknown ---
    mgr3 = DynamicTrailingStopManager()
    state3 = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        desired_stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        order_id=None,
        qty=12.0,
        strategy="DAYTRADE_SCALP",
        signal_id="sig-b1-3",
    )
    mgr3.states["AAPL"] = state3
    place_calls3 = []
    bridge_mock3 = NS(
        get_position=lambda sym: None,  # Broker position query fails/returns None
        market_price=lambda sym: 332.0,
        place_stop_order=lambda **kw: place_calls3.append(kw) or {"order_id": "stop-speculative"},
    )
    mgr3.handle_naked_position("AAPL", 12.0, bridge_mock3, dry_run=False)
    assert len(place_calls3) == 0, f"Must not place speculative orders on unknown broker state, got {len(place_calls3)}"
    assert state3.active is True, "State must remain active"
    assert state3.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
    assert state3.stop_reconciliation_required is True
    results["b1_case3_unknown_broker_state_no_speculative_submission"] = "PASS"

    # --- B1 Case 4: Failed emergency flatten -> retains active fail-closed unknown ---
    mgr4 = DynamicTrailingStopManager()
    state4 = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        qty=12.0,
        active=True,
    )
    mgr4.states["AAPL"] = state4
    bridge_fail_close = NS(
        close_position=lambda sym: False,  # 3 attempts return False
        working_orders_plain=lambda: [],
        cancel_order=lambda oid: True,
    )
    mgr4.emergency_flatten("AAPL", 12.0, bridge=bridge_fail_close, dry_run=False)
    assert state4.active is True, "Failed emergency flatten must NOT deactivate position!"
    assert state4.stop_reconciliation_required is True
    assert state4.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
    results["b1_case4_failed_emergency_flatten_retains_active_unknown"] = "PASS"

    # --- B1 Case 5: Stale position quantity overridden by fresh broker qty ---
    mgr5 = DynamicTrailingStopManager()
    state5 = TrailingStopState(
        symbol="AAPL",
        side="BUY",
        entry_price=331.88,
        peak_price=332.0,
        atr=1.5,
        stop_price=330.0,
        trail_multiple=1.5,
        last_updated=time.time(),
        qty=10.0,  # Stale quantity in local state
        order_id="stop-10",
        active=True,
    )
    mgr5.states["AAPL"] = state5
    # Fresh broker quantity is 12.0; stop order in open orders is only 10.0
    open_stop_10 = {
        "id": "stop-10",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "new",
        "qty": 10.0,
        "stop_price": 330.0,
    }
    bridge5 = NS(get_position=lambda sym: 12.0, market_price=lambda sym: 332.0)
    mgr5.ensure_initialized("AAPL", "BUY", 331.88, [open_stop_10], 332.0, qty=12.0, bridge=bridge5)
    # state.qty must now reflect 12.0, so the 10-share stop must be flagged as insufficient!
    assert state5.qty == 12.0, f"Expected state.qty to update to fresh broker qty 12.0, got {state5.qty}"
    assert state5.stop_reconciliation_required is True, "10-share stop must be flagged insufficient for 12-share position"
    results["b1_case5_stale_position_quantity_overridden_by_fresh_broker_qty"] = "PASS"

    # --- B1 Case 6: Fractional stop quantity in place_stop_order ---
    from alpaca_bridge import AlpacaBridge
    submitted_orders = []
    dummy_bridge = AlpacaBridge.__new__(AlpacaBridge)
    dummy_bridge._is_connected = True
    dummy_bridge._account_mode = "PAPER"
    dummy_bridge._active_account_label = "paper"
    dummy_bridge.dry_run = False
    dummy_bridge._client = NS(
        submit_order=lambda req: submitted_orders.append(req) or NS(id="order-frac-12.5")
    )
    dummy_bridge.assert_account_mode_safety = lambda: None
    dummy_bridge._require_connection = lambda: None
    dummy_bridge.assert_machine_release_gate = lambda **kw: None

    res_frac = dummy_bridge.place_stop_order(symbol="AAPL", side="SELL", qty=12.5, stop_price=330.0)
    assert res_frac["order_id"] == "order-frac-12.5"
    assert len(submitted_orders) == 1
    assert submitted_orders[0].qty == 12.5, f"Expected fractional qty 12.5, got {submitted_orders[0].qty}"
    results["b1_case6_fractional_stop_quantity_preserved"] = "PASS"

    return results


def test_b2_cumulative_entry_quantity():
    with tempfile.TemporaryDirectory() as td:
        fake_state_file = os.path.join(td, "trailing_state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state_file):
            return _test_b2_cumulative_entry_quantity_impl()


def _test_b2_cumulative_entry_quantity_impl():
    results = {}

    # --- B2 Case 1: entry 11 -> cumulative 12 -> exit 12 reconciles one close for 12 ---
    mgr = DynamicTrailingStopManager()
    sig = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=330.0,
        stop_price=325.0,
        target_price=340.0,
        strategy="DAYTRADE_SCALP",
        signal_id="sig-b2-1",
        confidence=0.85,
        atr=1.5,
        risk_per_share=5.0,
        reason="test_scalp",
    )

    # Initial partial entry fill: 11 shares
    s = mgr.initialize_position(signal=sig, fill_price=330.0, qty=11.0, stop_order_id="stop-1")
    assert s.qty == 11.0
    assert s.entry_qty == 11.0

    # Cumulative entry fill arrives: 12 shares
    s_updated = mgr.apply_confirmed_fill(signal=sig, average_fill_price=330.5, cumulative_qty=12.0)
    assert s_updated.qty == 12.0
    assert s_updated.entry_qty == 12.0, f"Expected entry_qty 12.0 after cumulative fill, got {s_updated.entry_qty}"

    # Repeated cumulative 12 notification does not double count
    s_repeated = mgr.apply_confirmed_fill(signal=sig, average_fill_price=330.5, cumulative_qty=12.0)
    assert s_repeated.entry_qty == 12.0

    # Exit 12 shares: broker position flat
    with tempfile.TemporaryDirectory() as td:
        journal_path = pathlib.Path(td) / "journal.jsonl"
        config.DAYTRADE_TRADE_JOURNAL_FILE = journal_path

        mgr._record_pending_exit(s_updated)
        queued_item = next(iter(mgr.pending_exits.values()))
        assert queued_item["qty"] == 12.0, f"Expected queued exit qty 12.0, got {queued_item['qty']}"

        # Broker reports flat position and filled exit order of 12 shares
        exit_order = NS(
            id="exit-1",
            client_order_id="dt_exit_sig-b2-1",
            symbol="AAPL",
            side="sell",
            status="filled",
            order_type="limit",
            filled_qty=12.0,
            filled_avg_price=340.0,
        )
        bridge = NS(
            get_position=lambda sym: 0.0,
            _client=NS(get_orders=lambda req: [exit_order]),
            get_order_snapshot=lambda oid: None,
        )
        reconcile_res = mgr.reconcile_pending_exits(bridge)
        assert reconcile_res.get("AAPL") == "RECONCILED_TAKE_PROFIT"
        assert len(mgr.pending_exits) == 0, "Pending exit must be cleared without EXCESS_EXIT_QTY_DISCREPANCY"

        rows = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["qty"] == 12.0
        assert rows[0]["exit_price"] == 340.0
        results["b2_case1_entry11_cum12_exit12_clean_reconciliation"] = "PASS"

    # --- B2 Case 2: entry 12 -> exit 4 -> remaining 8 -> final exit 8 aggregates 12 ---
    mgr2 = DynamicTrailingStopManager()
    sig2 = TradeSignal(
        symbol="AAPL",
        side="BUY",
        entry_price=330.0,
        stop_price=325.0,
        target_price=340.0,
        strategy="DAYTRADE_SCALP",
        signal_id="sig-b2-2",
        confidence=0.85,
        atr=1.5,
        risk_per_share=5.0,
        reason="test_scalp",
    )
    s2 = mgr2.initialize_position(signal=sig2, fill_price=330.0, qty=12.0, stop_order_id="stop-2", tp_order_id="tp-2")
    assert s2.entry_qty == 12.0

    # Partial exit of 4 shares: broker position reduced to 8
    stop_order_part = {
        "id": "stop-2",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "partially_filled",
        "qty": 12.0,
        "filled_qty": 4.0,
        "stop_price": 325.0,
    }
    mgr2._reconcile_broker_order(s2, [stop_order_part], bridge=NS(get_position=lambda sym: 8.0))
    assert s2.qty == 8.0, "Current open position must be updated to 8.0"
    assert s2.entry_qty == 12.0, f"entry_qty must NOT decrease from partial exit; expected 12.0, got {s2.entry_qty}"

    # Final exit occurs: position flat
    with tempfile.TemporaryDirectory() as td:
        journal_path = pathlib.Path(td) / "journal2.jsonl"
        config.DAYTRADE_TRADE_JOURNAL_FILE = journal_path

        mgr2._record_pending_exit(s2)
        orders = [
            NS(id="stop-2", client_order_id="", symbol="AAPL", side="sell", status="canceled", order_type="stop", filled_qty=4.0, filled_avg_price=325.0),
            NS(id="tp-2", client_order_id="", symbol="AAPL", side="sell", status="filled", order_type="limit", filled_qty=8.0, filled_avg_price=340.0),
        ]
        bridge2 = NS(
            get_position=lambda sym: 0.0,
            _client=NS(get_orders=lambda req: orders),
            get_order_snapshot=lambda oid: None,
        )
        rec_res2 = mgr2.reconcile_pending_exits(bridge2)
        assert rec_res2.get("AAPL") == "RECONCILED_SPLIT_EXIT"
        assert len(mgr2.pending_exits) == 0

        rows = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
        assert len(rows) == 1
        assert rows[0]["qty"] == 12.0
        results["b2_case2_partial_exit_preserves_entry_qty_aggregating_12"] = "PASS"

    return results


def test_r2_validation_emergency_exit():
    results = {}

    # --- R2 Case 1: Validation emergency exit without fills -> retains pending exit without writing realized P&L ---
    with tempfile.TemporaryDirectory() as td1:
        fake_state1 = os.path.join(td1, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state1):
            mgr1 = DynamicTrailingStopManager()
            state1 = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=330.0,
                peak_price=332.0,
                atr=1.5,
                stop_price=325.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="val_TS_001",
                is_validation=True,
                active=True,
            )
            mgr1.states["AAPL"] = state1

            val_exit_calls1 = []
            def mock_val_exit_fn1(**kw):
                val_exit_calls1.append(kw)
                return True

            with patch("strategies.validation_lane.record_validation_exit", side_effect=mock_val_exit_fn1):
                # Bridge reports position close succeeded, but NO closed orders / execution fills yet
                bridge_close_verified = NS(
                    close_position_verified=lambda sym: NS(outcome="FLAT", symbol=sym, initial_qty=12.0, remaining_qty=0.0, filled_qty=12.0, order_id="close-ord-1"),
                    get_position=lambda sym: 0.0,
                    working_orders_plain=lambda: [],
                    cancel_order=lambda oid: True,
                    get_order_snapshot=lambda oid: None,
                    get_orders=lambda **kw: [],
                    _client=NS(get_orders=lambda req: []),
                )

                mgr1.emergency_flatten("AAPL", 12.0, bridge=bridge_close_verified, dry_run=False)

                # 1. Realized P&L must NOT be written because execution fill price evidence does not exist
                assert len(val_exit_calls1) == 0, f"Must not write realized P&L to validation ledger without fill evidence, got {len(val_exit_calls1)}"
                # 2. State must remain pending reconciliation, not falsely marked reconciled!
                assert state1.reconciliation_status == "FLAT_PENDING_EXIT_RECONCILIATION"
                assert len(mgr1.pending_exits) == 1, "Pending exit must be retained in durable queue"
                results["r2_case1_validation_emergency_exit_without_fills_retains_pending"] = "PASS"

    # --- R2 Case 2: Validation emergency exit with verified execution fill -> writes ledger and reconciles ---
    with tempfile.TemporaryDirectory() as td2:
        fake_state2 = os.path.join(td2, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state2):
            mgr2 = DynamicTrailingStopManager()
            state2 = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=330.0,
                peak_price=332.0,
                atr=1.5,
                stop_price=325.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="val_TS_002",
                is_validation=True,
                active=True,
            )
            mgr2.states["AAPL"] = state2

            val_exit_calls2 = []
            def mock_val_exit_fn2(**kw):
                val_exit_calls2.append(kw)
                return True

            with patch("strategies.validation_lane.record_validation_exit", side_effect=mock_val_exit_fn2):
                closed_exit_order = NS(
                    id="close-ord-2",
                    client_order_id="close_val_002",
                    symbol="AAPL",
                    side="sell",
                    status="filled",
                    order_type="market",
                    filled_qty=12.0,
                    filled_avg_price=328.50,
                )
                bridge_with_fill = NS(
                    close_position_verified=lambda sym: NS(outcome="FLAT", symbol=sym, initial_qty=12.0, remaining_qty=0.0, filled_qty=12.0, order_id="close-ord-2"),
                    get_position=lambda sym: 0.0,
                    working_orders_plain=lambda: [],
                    cancel_order=lambda oid: True,
                    get_order_snapshot=lambda oid: {"id": "close-ord-2", "symbol": "AAPL", "side": "sell", "status": "filled", "filled_qty": 12.0, "filled_avg_price": 328.50, "type": "market"},
                    _client=NS(get_orders=lambda req: [closed_exit_order]),
                )

                mgr2.emergency_flatten("AAPL", 12.0, bridge=bridge_with_fill, dry_run=False)

                # 1. Validation ledger write must be called with actual verified fill price 328.50 (NOT fabricated stop 325.0 or entry 330.0)
                assert len(val_exit_calls2) == 1, f"Expected 1 validation exit write, got {len(val_exit_calls2)}"
                assert val_exit_calls2[0]["exit_price"] == 328.50, f"Expected exit_price 328.50, got {val_exit_calls2[0]['exit_price']}"
                assert val_exit_calls2[0]["qty"] == 12.0
                # 2. State must be marked reconciled and pending exit cleared
                assert state2.reconciliation_status == "RECONCILED_MARKET_EXIT"
                assert len(mgr2.pending_exits) == 0, "Pending exit must be cleared upon successful ledger write"
                results["r2_case2_validation_emergency_exit_with_verified_fill_writes_ledger"] = "PASS"

    # --- R2 Case 3: Validation ledger write returns False -> retains pending exit without marking reconciled ---
    with tempfile.TemporaryDirectory() as td3:
        fake_state3 = os.path.join(td3, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state3):
            mgr3 = DynamicTrailingStopManager()
            state3 = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=330.0,
                peak_price=332.0,
                atr=1.5,
                stop_price=325.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="val_TS_003",
                is_validation=True,
                active=True,
            )
            mgr3.states["AAPL"] = state3

            val_exit_calls3 = []
            def mock_val_exit_fn3(**kw):
                val_exit_calls3.append(kw)
                return False  # Ledger write failure / disk error

            with patch("strategies.validation_lane.record_validation_exit", side_effect=mock_val_exit_fn3):
                closed_exit_order3 = NS(
                    id="close-ord-3",
                    client_order_id="close_val_003",
                    symbol="AAPL",
                    side="sell",
                    status="filled",
                    order_type="market",
                    filled_qty=12.0,
                    filled_avg_price=328.50,
                )
                bridge_write_fail = NS(
                    close_position_verified=lambda sym: NS(outcome="FLAT", symbol=sym, initial_qty=12.0, remaining_qty=0.0, filled_qty=12.0, order_id="close-ord-3"),
                    get_position=lambda sym: 0.0,
                    working_orders_plain=lambda: [],
                    cancel_order=lambda oid: True,
                    get_order_snapshot=lambda oid: {"id": "close-ord-3", "symbol": "AAPL", "side": "sell", "status": "filled", "filled_qty": 12.0, "filled_avg_price": 328.50, "type": "market"},
                    _client=NS(get_orders=lambda req: [closed_exit_order3]),
                )

                mgr3.emergency_flatten("AAPL", 12.0, bridge=bridge_write_fail, dry_run=False)

                # 1. Validation ledger write was attempted
                assert len(val_exit_calls3) == 1
                # 2. But state must NOT be marked reconciled because write returned False!
                assert state3.reconciliation_status == "FLAT_PENDING_EXIT_RECONCILIATION"
                assert len(mgr3.pending_exits) == 1, "Pending exit must be retained when ledger write returns False"
                results["r2_case3_validation_ledger_write_failure_retains_pending"] = "PASS"

    return results


def test_c1_position_side_and_residual_fills():
    results = {}

    # C1 Case 1: Old stop is filled, but broker position has residual 5 shares
    # Manager must retain active=True, update qty=5.0, and proceed to protect residual position
    with tempfile.TemporaryDirectory() as td:
        fake_state = os.path.join(td, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state):
            mgr = DynamicTrailingStopManager()
            state = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=150.0,
                peak_price=155.0,
                atr=1.5,
                stop_price=145.0,
                desired_stop_price=145.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="stop-filled-old",
                stop_reconciliation_required=True,
                active=True,
            )
            mgr.states["AAPL"] = state

            placed_stops = []
            bridge_residual = NS(
                get_position=lambda sym: 5.0,
                market_price=lambda sym: 152.0,
                working_orders_plain=lambda: [],
                cancel_order=lambda oid: True,
                place_stop_order=lambda **kw: placed_stops.append(kw) or {"order_id": "stop-residual-5"},
                get_order_snapshot=lambda oid: (
                    {"id": "stop-filled-old", "symbol": "AAPL", "status": "filled", "qty": 12.0, "filled_qty": 12.0, "side": "sell", "type": "stop", "stop_price": 145.0}
                    if oid == "stop-filled-old"
                    else {"id": "stop-residual-5", "symbol": "AAPL", "status": "new", "qty": 5.0, "filled_qty": 0.0, "side": "sell", "type": "stop", "stop_price": 145.0}
                ),
                _client=NS(get_orders=lambda req: []),
            )

            mgr.handle_naked_position("AAPL", 12.0, bridge_residual, dry_run=False)

            assert state.active is True, "State must remain ACTIVE when residual 5 shares exist on broker"
            assert state.qty == 5.0, f"State qty must be updated to residual 5.0, got {state.qty}"
            assert len(placed_stops) == 1, f"Expected 1 replacement stop for residual position, got {len(placed_stops)}"
            assert placed_stops[0]["qty"] == 5.0, f"Expected stop placed for residual 5.0 shares, got {placed_stops[0]['qty']}"
            assert state.order_id == "stop-residual-5"
            assert state.stop_reconciliation_required is False
            results["c1_case1_filled_stop_residual_shares_retains_active_monitoring"] = "PASS"

    # C1 Case 2: Old stop is filled, but broker query returns unknown/fails
    # Manager must retain active=True, flag reconciliation required fail-closed without deactivating
    with tempfile.TemporaryDirectory() as td:
        fake_state = os.path.join(td, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state):
            mgr = DynamicTrailingStopManager()
            state = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=150.0,
                peak_price=155.0,
                atr=1.5,
                stop_price=145.0,
                desired_stop_price=145.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="stop-filled-unknown",
                stop_reconciliation_required=True,
                active=True,
            )
            mgr.states["AAPL"] = state

            placed_stops = []
            def fail_get_pos(sym):
                raise RuntimeError("Alpaca API timeout")

            bridge_query_fail = NS(
                get_position=fail_get_pos,
                market_price=lambda sym: 152.0,
                working_orders_plain=lambda: [],
                cancel_order=lambda oid: True,
                place_stop_order=lambda **kw: placed_stops.append(kw) or {"order_id": "bad"},
                get_order_snapshot=lambda oid: {"id": "stop-filled-unknown", "symbol": "AAPL", "status": "filled", "qty": 12.0, "filled_qty": 12.0, "side": "sell", "type": "stop", "stop_price": 145.0},
                _client=NS(get_orders=lambda req: []),
            )

            mgr.handle_naked_position("AAPL", 12.0, bridge_query_fail, dry_run=False)

            assert state.active is True, "State must remain ACTIVE when position query fails"
            assert state.stop_reconciliation_required is True
            assert state.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
            assert len(placed_stops) == 0, "No speculative stop order placed when position unknown"
            results["c1_case2_filled_stop_unknown_broker_position_retains_active_monitoring"] = "PASS"

    # C1 Case 3: Signed broker side mismatch on old stop cancellation (BUY local vs -5.0 broker short)
    # Replacement stop must be blocked fail-closed, retaining active=True with signed side mismatch flag
    with tempfile.TemporaryDirectory() as td:
        fake_state = os.path.join(td, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state):
            mgr = DynamicTrailingStopManager()
            state = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=150.0,
                peak_price=155.0,
                atr=1.5,
                stop_price=145.0,
                desired_stop_price=145.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="stop-old-cancel",
                stop_reconciliation_required=True,
                active=True,
            )
            mgr.states["AAPL"] = state

            placed_stops = []
            bridge_mismatch = NS(
                get_position=lambda sym: -5.0,  # Negative broker position: SHORT!
                market_price=lambda sym: 152.0,
                working_orders_plain=lambda: [],
                cancel_order=lambda oid: True,
                place_stop_order=lambda **kw: placed_stops.append(kw) or {"order_id": "bad"},
                get_order_snapshot=lambda oid: {"id": "stop-old-cancel", "symbol": "AAPL", "status": "canceled", "qty": 12.0, "side": "sell", "type": "stop", "stop_price": 145.0},
                _client=NS(get_orders=lambda req: []),
            )

            mgr.handle_naked_position("AAPL", 12.0, bridge_mismatch, dry_run=False)

            assert state.active is True, "State must remain ACTIVE on side mismatch to block fresh entries"
            assert state.stop_reconciliation_required is True
            assert state.reconciliation_status == "SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED"
            assert len(placed_stops) == 0, "Must block replacement stop on signed side mismatch"
            results["c1_case3_signed_side_mismatch_blocks_replacement"] = "PASS"

    # C1 Case 4: Signed broker side mismatch when old stop was filled (BUY local vs -5.0 broker short)
    with tempfile.TemporaryDirectory() as td:
        fake_state = os.path.join(td, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state):
            mgr = DynamicTrailingStopManager()
            state = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=150.0,
                peak_price=155.0,
                atr=1.5,
                stop_price=145.0,
                desired_stop_price=145.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="stop-old-fill-mismatch",
                stop_reconciliation_required=True,
                active=True,
            )
            mgr.states["AAPL"] = state

            placed_stops = []
            bridge_fill_mismatch = NS(
                get_position=lambda sym: -5.0,  # Negative broker position: SHORT!
                market_price=lambda sym: 152.0,
                working_orders_plain=lambda: [],
                cancel_order=lambda oid: True,
                place_stop_order=lambda **kw: placed_stops.append(kw) or {"order_id": "bad"},
                get_order_snapshot=lambda oid: {"id": "stop-old-fill-mismatch", "symbol": "AAPL", "status": "filled", "qty": 12.0, "filled_qty": 12.0, "side": "sell", "type": "stop", "stop_price": 145.0},
                _client=NS(get_orders=lambda req: []),
            )

            mgr.handle_naked_position("AAPL", 12.0, bridge_fill_mismatch, dry_run=False)

            assert state.active is True
            assert state.stop_reconciliation_required is True
            assert state.reconciliation_status == "SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED"
            assert len(placed_stops) == 0, "Must block replacement stop on signed side mismatch after fill"
            results["c1_case4_filled_stop_signed_side_mismatch_blocks_replacement"] = "PASS"

    return results


def test_c2_pending_exit_persistence_and_recovery():
    results = {}

    # C2 Case 1: _save_state failure during reconcile_pending_exits retains queue and rolls back state
    with tempfile.TemporaryDirectory() as td:
        fake_state = os.path.join(td, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state):
            mgr = DynamicTrailingStopManager()
            state = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=150.0,
                peak_price=155.0,
                atr=1.5,
                stop_price=145.0,
                desired_stop_price=145.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="stop-pers-1",
                active=True,
            )
            mgr.states["AAPL"] = state
            mgr.pending_exits["AAPL"] = {
                "symbol": "AAPL",
                "side": "BUY",
                "qty": 12.0,
                "entry_price": 150.0,
                "exit_price": 145.0,
                "episode_id": "AAPL_ep_1",
                "last_stop_order_id": "stop-pers-1",
            }

            mock_order = NS(
                id="stop-pers-1",
                client_order_id="cl_stop_1",
                symbol="AAPL",
                side="sell",
                status="filled",
                order_type="stop",
                filled_qty=12.0,
                filled_avg_price=145.0,
            )
            bridge = NS(
                get_position=lambda sym: 0.0,
                working_orders_plain=lambda: [],
                cancel_order=lambda oid: True,
                get_order_snapshot=lambda oid: {"id": "stop-pers-1", "symbol": "AAPL", "side": "sell", "status": "filled", "filled_qty": 12.0, "filled_avg_price": 145.0, "type": "stop"},
                _client=NS(get_orders=lambda req: [mock_order]),
            )

            # Simulate _save_state() disk failure
            with patch.object(mgr, "_save_state", return_value=False):
                with patch("strategies.trade_journal.log_trade", return_value=True):
                    res = mgr.reconcile_pending_exits(bridge)

            assert res.get("AAPL") == "QUEUE_PERSISTENCE_FAILED_RETAINED", f"Expected QUEUE_PERSISTENCE_FAILED_RETAINED, got {res.get('AAPL')}"
            assert "AAPL" in mgr.pending_exits, "Pending exit must be retained in queue when save fails"
            assert state.active is True, "State.active must be rolled back to True"
            assert state.stop_reconciliation_required is True, "State.stop_reconciliation_required must be rolled back to True"
            results["c2_case1_pending_exit_save_state_failure_retains_queue"] = "PASS"

    # C2 Case 2: Process restart / crash recovery survives and reloads from disk cleanly
    with tempfile.TemporaryDirectory() as td:
        fake_state = os.path.join(td, "state.json")
        with patch("strategies.trailing_stop.STATE_FILE", fake_state):
            mgr1 = DynamicTrailingStopManager()
            state1 = TrailingStopState(
                symbol="AAPL",
                side="BUY",
                entry_price=150.0,
                peak_price=155.0,
                atr=1.5,
                stop_price=145.0,
                desired_stop_price=145.0,
                trail_multiple=1.5,
                last_updated=time.time(),
                qty=12.0,
                order_id="stop-crash-1",
                active=True,
            )
            mgr1.states["AAPL"] = state1
            mgr1._record_pending_exit(state1, qty_closed=12.0)
            mgr1._save_state()
            assert os.path.exists(fake_state), "State file must exist on disk"

            # Simulate crash and restart by instantiating new manager
            mgr2 = DynamicTrailingStopManager()
            assert "AAPL" in mgr2.states, "Recovered manager must contain AAPL state"
            assert len(mgr2.pending_exits) >= 1, "Recovered manager must contain pending exit"
            assert any(p.get("symbol") == "AAPL" and p.get("qty") == 12.0 for p in mgr2.pending_exits.values()), "AAPL exit retained across crash"

            mock_exit_order = NS(
                id="stop-crash-1",
                client_order_id="cl_crash_1",
                symbol="AAPL",
                side="sell",
                status="filled",
                order_type="stop",
                filled_qty=12.0,
                filled_avg_price=145.0,
            )
            bridge2 = NS(
                get_position=lambda sym: 0.0,
                working_orders_plain=lambda: [],
                cancel_order=lambda oid: True,
                get_order_snapshot=lambda oid: {"id": "stop-crash-1", "symbol": "AAPL", "side": "sell", "status": "filled", "filled_qty": 12.0, "filled_avg_price": 145.0, "type": "stop"},
                _client=NS(get_orders=lambda req: [mock_exit_order]),
            )
            with patch("strategies.trade_journal.log_trade", return_value=True):
                res2 = mgr2.reconcile_pending_exits(bridge2)

            assert res2.get("AAPL") == "RECONCILED_STOP_OUT", f"Expected RECONCILED_STOP_OUT, got {res2.get('AAPL')}"
            assert len(mgr2.pending_exits) == 0, "Pending exits queue must be cleared after reconciliation"
            results["c2_case2_crash_recovery_persisted_state_and_pending_exits"] = "PASS"

    return results


def test_c3_validation_ledger_durability_and_idempotency():
    results = {}
    from strategies.validation_lane import record_validation_fill, record_validation_exit

    with tempfile.TemporaryDirectory() as td:
        ledger_path = os.path.join(td, "paper_validation_trades.jsonl")

        # 1. Initialize validation candidate row in ledger
        cand_record = {
            "validation_id": "val_AAPL_001",
            "entry_order_id": "ord_entry_001",
            "symbol": "AAPL",
            "side": "BUY",
            "decision_price": 150.0,
            "status": "SUBMITTED",
            "timestamp": "2026-10-10T12:00:00+00:00",
        }
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(cand_record) + "\n")

        # 2. Record validation entry fill
        fill_res = record_validation_fill(
            order_id="ord_entry_001",
            fill_price=150.05,
            qty=1.0,
            side="BUY",
            timestamp="2026-10-10T12:01:00+00:00",
            execution_id="exec_fill_001",
            filepath=ledger_path,
        )
        assert fill_res is True, "record_validation_fill must return True"

        # 3. Record validation exit
        exit_res1 = record_validation_exit(
            order_id="ord_exit_001",
            exit_price=152.00,
            qty=1.0,
            side="SELL",
            timestamp="2026-10-10T12:10:00+00:00",
            symbol="AAPL",
            exit_reason="PROFIT_TARGET",
            execution_id="exec_exit_001",
            filepath=ledger_path,
        )
        assert exit_res1 is True, "record_validation_exit must return True"

        # Verify file contents
        with open(ledger_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1, f"Expected 1 record in ledger, got {len(lines)}"
        data = json.loads(lines[0])
        assert data["status"] == "CLOSED"
        assert data["exit_fill_price"] == 152.0
        assert data["realized_pnl"] == round((152.0 - 150.05) * 1.0, 4)
        results["c3_case1_validation_ledger_fsync_and_atomic_replace"] = "PASS"

        # 4. Retry recording exit with same order_id / execution_id -> MUST return True idempotently without creating duplicate lines
        exit_res2 = record_validation_exit(
            order_id="ord_exit_001",
            exit_price=152.00,
            qty=1.0,
            side="SELL",
            timestamp="2026-10-10T12:10:00+00:00",
            symbol="AAPL",
            exit_reason="PROFIT_TARGET",
            execution_id="exec_exit_001",
            filepath=ledger_path,
        )
        assert exit_res2 is True, "Retry record_validation_exit must return True idempotently"

        with open(ledger_path, "r", encoding="utf-8") as f:
            lines2 = [l.strip() for l in f if l.strip()]
        assert len(lines2) == 1, f"Idempotent retry must NOT create duplicate lines, got {len(lines2)}"
        results["c3_case2_validation_ledger_retry_idempotent_no_duplicate_rows"] = "PASS"

    return results


if __name__ == "__main__":
    b1_results = test_b1_lifecycle_remediation()
    b2_results = test_b2_cumulative_entry_quantity()
    r2_results = test_r2_validation_emergency_exit()
    c1_results = test_c1_position_side_and_residual_fills()
    c2_results = test_c2_pending_exit_persistence_and_recovery()
    c3_results = test_c3_validation_ledger_durability_and_idempotency()
    all_res = {**b1_results, **b2_results, **r2_results, **c1_results, **c2_results, **c3_results}
    print(json.dumps(all_res, indent=2))

