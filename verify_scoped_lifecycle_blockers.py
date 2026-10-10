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

    # Open orders has 1-share stop (capacity insufficient)
    one_share_stop = {
        "id": "stop-1",
        "symbol": "AAPL",
        "side": "sell",
        "type": "stop",
        "status": "new",
        "qty": 1.0,
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

    # 3. handle_naked_position runs
    mgr.handle_naked_position("AAPL", 12.0, bridge_mock, dry_run=False)

    assert len(place_calls) == 1, f"Expected 1 place_stop_order call, got {len(place_calls)}"
    assert place_calls[0]["qty"] == 12.0, f"Expected recovery order qty 12.0, got {place_calls[0]['qty']}"
    assert state.order_id == "stop-new-12", f"Expected new order_id stop-new-12, got {state.order_id}"
    assert state.stop_reconciliation_required is False, "After successful replacement, stop_reconciliation_required must be False"
    assert state.reconciliation_status is None, "After successful replacement, reconciliation_status must be None"

    results["b1_case1_insufficient_stop_replaced_covering_full_pos"] = "PASS"

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

    return results


def test_b2_cumulative_entry_quantity():
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


if __name__ == "__main__":
    b1_results = test_b1_lifecycle_remediation()
    b2_results = test_b2_cumulative_entry_quantity()
    all_res = {**b1_results, **b2_results}
    print(json.dumps(all_res, indent=2))
