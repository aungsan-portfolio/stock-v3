"""One symbol's management failure must not strand the rest of the portfolio."""

from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import config
from strategies import orchestrator as orch
from strategies.event_guard import EventDecision
from strategies.regime_engine import MarketRegimeResult, MODE_RISK_ON


@pytest.fixture
def management_cycle(monkeypatch):
    """A healthy, due-for-evaluation paper cycle with two existing positions."""
    states = {
        sym: SimpleNamespace(active=True, entry_time=1, entry_price=100.0,
                             peak_price=100.0, order_id=f"stop-{sym}",
                             stop_reconciliation_required=False)
        for sym in ("SPY", "HD")
    }
    manager = MagicMock()
    manager.states = states
    manager.has_unresolved_position_unknown.return_value = False
    manager.ensure_initialized.side_effect = lambda **kw: states[kw["symbol"]]
    bridge = MagicMock()
    bridge.is_connected = True
    bridge.entry_gate = None
    bridge._daytrade_suspended = False
    bridge.ib.positions.return_value = [
        SimpleNamespace(contract=SimpleNamespace(symbol=sym), position=qty, avgCost=100.0)
        for sym, qty in (("SPY", -1.0), ("HD", 1.0))
    ]
    bridge.ib.openTrades.return_value = []
    bridge.market_price.return_value = 100.0
    bridge.get_trailing_stop_manager.return_value = manager
    bridge.close_position_verified.return_value = SimpleNamespace(is_flat=True)
    regime = MarketRegimeResult(
        mode=MODE_RISK_ON, score=0, allow_new_longs=True,
        requires_opening_buffer=False, position_scale=1.0, confidence_boost=0.0,
        market_texture="TRENDING", data={"market_texture": "TRENDING"},
    )
    now_utc = datetime(2026, 11, 4, 15, 0, 2, tzinfo=timezone.utc)
    monkeypatch.setattr(orch, "_last_evaluated_5m_bar_str", None, raising=False)
    monkeypatch.setattr(orch, "_skipped_bar_ids", set(), raising=False)
    monkeypatch.setattr(config, "FLATTEN_ON_DAILY_LOSS", False)
    monkeypatch.setattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
    monkeypatch.setattr(config, "EXPERIMENTAL_MR_SHADOW_ENABLED", False)
    health = SimpleNamespace(allows_new_entries=True, summary="healthy")
    with ExitStack() as stack:
        clock = stack.enter_context(patch.object(orch, "datetime"))
        clock.now.return_value = now_utc
        clock.side_effect = lambda *a, **kw: datetime(*a, **kw)
        stack.enter_context(patch("strategies.validation_lane.get_production_risk_equity", return_value=(100000.0, 0.0)))
        stack.enter_context(patch.object(orch, "today_pnl", return_value=0.0))
        stack.enter_context(patch("strategies.regime_engine.evaluate_market_regime", return_value=regime))
        stack.enter_context(patch.object(orch._regime_state_tracker, "update", return_value=regime))
        stack.enter_context(patch("strategies.event_guard.evaluate_event", return_value=EventDecision(1.0, False, 1.0, None)))
        stack.enter_context(patch("strategies.session.is_past_opening_buffer", return_value=True))
        stack.enter_context(patch("strategies.execution_health.get_execution_health", return_value=health))
        stack.enter_context(patch("strategies.order_manager.reap_expired_entry_orders"))
        stack.enter_context(patch("strategies.intraday_data.get_position_atr_if_fresh", return_value=1.0))
        time_exit = stack.enter_context(patch("strategies.trailing_stop.check_time_based_exit", return_value=(False, "")))
        evaluate = stack.enter_context(patch.object(orch, "evaluate_symbols_parallel", return_value=[]))
        execute = stack.enter_context(patch.object(orch, "execute_signal"))
        heartbeat = stack.enter_context(patch("strategies.watchdog.touch_orchestrator_heartbeat"))
        yield SimpleNamespace(bridge=bridge, manager=manager, states=states,
                              time_exit=time_exit, evaluate=evaluate, execute=execute,
                              heartbeat=heartbeat)


def _run(cycle):
    orch.evaluate_and_execute(["SPY", "HD", "NEW"], cycle.bridge, live_paper=True)


def test_healthy_management_still_updates_both_positions_and_evaluates_entries(management_cycle):
    cycle = management_cycle
    _run(cycle)
    assert [c.args[0] for c in cycle.manager.update_stop.call_args_list] == ["SPY", "HD"]
    cycle.evaluate.assert_called_once()


def test_stop_update_exception_keeps_later_stop_update_and_blocks_entries(management_cycle, caplog):
    cycle = management_cycle
    cycle.manager.update_stop.side_effect = [NameError("undefined stop parameter"), None]
    _run(cycle)
    assert [c.args[0] for c in cycle.manager.update_stop.call_args_list] == ["SPY", "HD"]
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()
    assert cycle.states["SPY"].order_id == "stop-SPY"
    assert cycle.states["SPY"].active is True
    cycle.manager.handle_naked_position.assert_not_called()
    cycle.manager.soft_reset.assert_not_called()
    errors = [r for r in caplog.records if "[POSITION MANAGEMENT ERROR]" in r.message]
    assert len(errors) == 1 and "SPY" in errors[0].message and errors[0].exc_info


def test_stop_update_exception_keeps_later_time_exit_without_retry(management_cycle):
    cycle = management_cycle
    cycle.manager.update_stop.side_effect = NameError("undefined stop parameter")
    cycle.time_exit.side_effect = [(False, ""), (True, "existing time exit rule")]
    _run(cycle)
    assert [c.kwargs["symbol"] for c in cycle.time_exit.call_args_list] == ["SPY", "HD"]
    cycle.bridge.close_position_verified.assert_called_once_with("HD")
    cycle.manager.soft_reset.assert_called_once_with("HD")
    cycle.manager.update_stop.assert_called_once()
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()


@pytest.mark.parametrize("failure", ["quote", "initialization", "naked_repair"])
def test_other_symbol_failures_do_not_skip_later_position(management_cycle, failure):
    cycle = management_cycle
    if failure == "quote":
        cycle.bridge.market_price.side_effect = [ConnectionError("quote unavailable"), 100.0]
    elif failure == "initialization":
        cycle.manager.ensure_initialized.side_effect = [RuntimeError("state unavailable"), cycle.states["HD"]]
    else:
        cycle.states["SPY"].order_id = None
        cycle.manager.handle_naked_position.side_effect = RuntimeError("broker mutation outcome unknown")
    _run(cycle)
    cycle.manager.update_stop.assert_called_once_with("HD", 100.0, bridge=cycle.bridge, dry_run=False, current_atr=1.0)
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()
    if failure == "naked_repair":
        cycle.manager.handle_naked_position.assert_called_once()


def test_management_error_cannot_be_cleared_by_health_recovery_or_inverse_exemption(management_cycle, monkeypatch):
    cycle = management_cycle
    cycle.manager.update_stop.side_effect = [RuntimeError("unknown stop update outcome"), None]
    gate = SimpleNamespace(halted=True, reasons={"OFF_MARKET_STANDBY"})
    cycle.bridge.entry_gate = gate
    cycle.bridge._clear_recovery_halts.side_effect = lambda: setattr(gate, "halted", False)
    monkeypatch.setattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", True)
    monkeypatch.setattr(config, "INVERSE_ETF_SYMBOLS", {"SPY"})
    _run(cycle)
    cycle.bridge._clear_recovery_halts.assert_called_once()
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()


def test_outer_reconciliation_failure_blocks_entries_without_retry(management_cycle):
    cycle = management_cycle
    cycle.manager.reconcile_pending_exits.side_effect = [None, RuntimeError("pending exit reconciliation failed")]
    _run(cycle)
    assert cycle.manager.reconcile_pending_exits.call_count == 2
    cycle.manager.update_stop.assert_not_called()
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()


def test_management_failure_is_cycle_local(management_cycle):
    cycle = management_cycle
    cycle.manager.update_stop.side_effect = [RuntimeError("temporary update failure"), None, None, None]
    _run(cycle)
    cycle.evaluate.assert_not_called()
    _run(cycle)
    cycle.evaluate.assert_called_once()
    assert cycle.manager.update_stop.call_count == 4


@pytest.mark.parametrize("status", ["UNKNOWN", "PENDING_CANCELLATION", "ABSENT"])
def test_unconfirmed_time_exit_blocks_entries_and_keeps_later_management(management_cycle, status):
    cycle = management_cycle
    cycle.time_exit.side_effect = [(True, "existing exit rule"), (False, "")]
    cycle.bridge.close_position_verified.return_value = SimpleNamespace(
        is_flat=False, protection_status=status, outcome="UNKNOWN"
    )
    _run(cycle)
    cycle.bridge.close_position_verified.assert_called_once_with("SPY")
    cycle.manager.update_stop.assert_called_once_with("HD", 100.0, bridge=cycle.bridge, dry_run=False, current_atr=1.0)
    cycle.manager.soft_reset.assert_not_called()
    assert cycle.states["SPY"].active is True
    assert cycle.states["SPY"].stop_reconciliation_required is True
    assert cycle.manager.handle_naked_position.call_count == (1 if status == "ABSENT" else 0)
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()


@pytest.mark.parametrize("uncertainty", ["modify_pending", "position_unknown"])
def test_nonexception_management_uncertainty_blocks_entries(management_cycle, uncertainty):
    cycle = management_cycle

    def update(symbol, *args, **kwargs):
        if symbol == "SPY":
            if uncertainty == "modify_pending":
                cycle.states[symbol].stop_reconciliation_required = True
            else:
                cycle.states[symbol].reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
        return cycle.states[symbol]

    cycle.manager.update_stop.side_effect = update
    _run(cycle)
    assert [c.args[0] for c in cycle.manager.update_stop.call_args_list] == ["SPY", "HD"]
    cycle.manager.handle_naked_position.assert_not_called()
    cycle.manager.soft_reset.assert_not_called()
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()


def test_unresolved_naked_repair_blocks_entries_without_retry(management_cycle):
    cycle = management_cycle
    cycle.states["SPY"].order_id = None
    _run(cycle)
    cycle.manager.handle_naked_position.assert_called_once()
    cycle.manager.update_stop.assert_called_once_with("HD", 100.0, bridge=cycle.bridge, dry_run=False, current_atr=1.0)
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()


def test_existing_position_unknown_gate_survives_health_recovery(management_cycle):
    cycle = management_cycle
    cycle.manager.has_unresolved_position_unknown.return_value = True
    gate = SimpleNamespace(halted=True, reasons={"OFF_MARKET_STANDBY"})
    cycle.bridge.entry_gate = gate
    cycle.bridge._clear_recovery_halts.side_effect = lambda: setattr(gate, "halted", False)
    _run(cycle)
    cycle.bridge._clear_recovery_halts.assert_called_once()
    assert cycle.manager.update_stop.call_count == 2
    cycle.evaluate.assert_not_called()
    cycle.execute.assert_not_called()
