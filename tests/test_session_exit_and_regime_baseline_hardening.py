"""
test_session_exit_and_regime_baseline_hardening.py

Regression test suite for:
1. Calendar-aware session previous-close baseline extraction in regime_engine
2. AlpacaBridge.get_order_snapshot completeness and verify_broker_stop_order_evidence integration
3. TrailingStopManager.handle_naked_position exit reconciliation (TAKE_PROFIT vs STOP_OUT vs FLAT_PENDING_EXIT_RECONCILIATION)
"""

import math
import time
from datetime import datetime, date, timezone
from unittest.mock import MagicMock, patch

import pytest
import pytz

from strategies.regime_engine import _extract_prev_session_close, _get_premarket_pct_change
from strategies.trailing_stop import (
    DynamicTrailingStopManager,
    TrailingStopState,
    verify_broker_stop_order_evidence,
)
from alpaca_bridge import AlpacaBridge


# ---------------------------------------------------------------------------
# Suite 1: Calendar-Aware Regime Baseline Extraction
# ---------------------------------------------------------------------------

class TestCalendarAwareRegimeBaseline:
    def test_extract_prev_session_close_premarket_vs_intraday(self):
        """Verify baseline does not shift between pre-market (today absent) and intraday (today present)."""
        today_date = date(2026, 10, 9)

        # Bar 1: Oct 7 close = 100.0
        # Bar 2: Oct 8 close = 105.0 (Yesterday)
        bars_premarket = [
            MagicMock(close=100.0, timestamp=datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc)),
            MagicMock(close=105.0, timestamp=datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc)),
        ]

        # In pre-market, today's bar is not yet present
        prev_close_pm, _, dt_str_pm = _extract_prev_session_close(bars_premarket, today_date=today_date)
        assert prev_close_pm == pytest.approx(105.0)
        assert dt_str_pm == "2026-10-08"

        # Intraday: Today's partial daily bar (Oct 9) is now present with current close = 108.0
        bars_intraday = [
            MagicMock(close=100.0, timestamp=datetime(2026, 10, 7, 20, 0, tzinfo=timezone.utc)),
            MagicMock(close=105.0, timestamp=datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc)),
            MagicMock(close=108.0, timestamp=datetime(2026, 10, 9, 14, 0, tzinfo=timezone.utc)),
        ]

        # Intraday lookup must STILL select Oct 8 close (105.0), NOT Oct 7 (100.0) or Oct 9 (108.0)
        prev_close_intra, _, dt_str_intra = _extract_prev_session_close(bars_intraday, today_date=today_date)
        assert prev_close_intra == pytest.approx(105.0)
        assert dt_str_intra == "2026-10-08"

    def test_extract_prev_session_close_weekend_and_dicts(self):
        """Verify dict bars and weekend session baseline correctly selects Friday."""
        # Today is Monday Oct 12, 2026
        today_date = date(2026, 10, 12)

        bars = [
            {"close": 210.0, "timestamp": "2026-10-08T20:00:00Z"},
            {"close": 215.5, "timestamp": "2026-10-09T20:00:00Z"},  # Friday
            {"close": 218.0, "timestamp": "2026-10-12T13:30:00Z"},  # Monday morning
        ]

        prev_close, _, dt_str = _extract_prev_session_close(bars, today_date=today_date)
        assert prev_close == pytest.approx(215.5)
        assert dt_str == "2026-10-09"

    def test_extract_prev_session_close_unparsed_fallback(self):
        """Verify unparsed timestamps safely fail closed with (None, None, None)."""
        bars = [
            MagicMock(close=50.0, timestamp=None, date=None),
            MagicMock(close=55.0, timestamp=None, date=None),
        ]
        prev_close, _, dt_str = _extract_prev_session_close(bars, today_date=date(2026, 10, 9))
        assert prev_close is None
        assert dt_str is None

    def test_extract_prev_session_close_only_today_bar(self):
        """Verify that if only today's partial bar exists, fail closed with None."""
        bars = [
            {"close": 150.0, "timestamp": "2026-10-09T14:30:00Z"},
        ]
        prev_close, _, dt_str = _extract_prev_session_close(bars, today_date=date(2026, 10, 9))
        assert prev_close is None
        assert dt_str is None

    def test_extract_prev_session_close_stale_gap_over_5_days(self):
        """Verify that bars older than 5 calendar days are rejected as stale."""
        bars = [
            {"close": 150.0, "timestamp": "2026-10-01T20:00:00Z"},  # 8 days ago
        ]
        prev_close, _, dt_str = _extract_prev_session_close(bars, today_date=date(2026, 10, 9))
        assert prev_close is None
        assert dt_str is None




# ---------------------------------------------------------------------------
# Suite 2: AlpacaBridge get_order_snapshot & verify_broker_stop_order_evidence
# ---------------------------------------------------------------------------

class TestOrderSnapshotVerification:
    def test_get_order_snapshot_returns_all_canonical_fields(self):
        """Verify get_order_snapshot extracts symbol, side, order_type, stop_price, etc."""
        bridge = object.__new__(AlpacaBridge)
        mock_client = MagicMock()

        mock_order = MagicMock()
        mock_order.id = "stop-ord-12345"
        mock_order.client_order_id = "dt_aapl_stop"
        mock_order.status = "new"
        mock_order.symbol = "AAPL"
        mock_order.side = "sell"
        mock_order.order_type = "stop"
        mock_order.qty = "12"
        mock_order.filled_qty = "0"
        mock_order.stop_price = "331.89"
        mock_order.limit_price = None
        mock_order.filled_avg_price = None
        mock_order.replaced_by = None

        mock_client.get_order_by_id.return_value = mock_order
        bridge._client = mock_client

        snap = bridge.get_order_snapshot("stop-ord-12345")
        assert snap is not None
        assert snap["id"] == "stop-ord-12345"
        assert snap["symbol"] == "AAPL"
        assert snap["side"] == "sell"
        assert snap["order_type"] == "stop"
        assert snap["type"] == "stop"
        assert snap["qty"] == 12.0
        assert snap["stop_price"] == pytest.approx(331.89)

        # Direct integration check with verify_broker_stop_order_evidence
        valid, reason = verify_broker_stop_order_evidence(
            order_data=snap,
            expected_order_id="stop-ord-12345",
            expected_symbol="AAPL",
            position_side="BUY",
            expected_stop_price=331.89,
            expected_qty=12.0,
            fill_price=331.84,
        )
        assert valid is True
        assert reason == "EVIDENCE_VERIFIED"

    def test_get_order_snapshot_missing_fields_fail_evidence(self):
        """Verify missing fields (None) correctly trigger evidence rejection fail-closed."""
        # Snapshot missing filled_qty
        snap_missing_filled_qty = {
            "id": "ord-1",
            "symbol": "AAPL",
            "side": "sell",
            "order_type": "stop",
            "qty": 10.0,
            "filled_qty": None,
            "stop_price": 150.0,
            "status": "partially_filled",
        }
        valid, reason = verify_broker_stop_order_evidence(
            snap_missing_filled_qty,
            expected_order_id="ord-1",
            expected_symbol="AAPL",
            position_side="BUY",
            expected_stop_price=150.0,
            expected_qty=10.0,
            fill_price=151.0,
        )
        assert valid is False
        assert "EVIDENCE_FILLED_QTY_MISSING" in reason

        # Snapshot missing id
        snap_missing_id = {
            "id": None,
            "symbol": "AAPL",
            "side": "sell",
            "order_type": "stop",
            "qty": 10.0,
            "filled_qty": 0.0,
            "stop_price": 150.0,
            "status": "new",
        }
        valid, reason = verify_broker_stop_order_evidence(
            snap_missing_id,
            expected_order_id="ord-1",
            expected_symbol="AAPL",
            position_side="BUY",
            expected_stop_price=150.0,
            expected_qty=10.0,
            fill_price=151.0,
        )
        assert valid is False
        assert "EVIDENCE_ID_MISSING" in reason




# ---------------------------------------------------------------------------
# Suite 3: handle_naked_position Exit Reconciliation
# ---------------------------------------------------------------------------

class TestHandleNakedPositionExitReconciliation:
    def test_handle_naked_position_reconciles_take_profit_exit(self, tmp_path):
        """When position is flat and TP order snapshot is filled, log TAKE_PROFIT (not STOP_OUT)."""
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="AAPL",
            side="BUY",
            entry_price=331.84,
            peak_price=335.10,
            atr=1.50,
            stop_price=331.89,
            trail_multiple=1.5,
            last_updated=time.time(),
            order_id=None,
            last_stop_order_id="stop-leg-1",
            tp_order_id="tp-leg-1",
            qty=12.0,
            target_price=335.10,
            active=True,
            strategy="MEAN_REVERSION",
        )
        mgr.states["AAPL"] = state

        mock_bridge = MagicMock()
        mock_bridge.get_position.return_value = 0.0

        def mock_snap(oid):
            if oid == "tp-leg-1":
                return {
                    "id": "tp-leg-1",
                    "status": "filled",
                    "symbol": "AAPL",
                    "side": "sell",
                    "order_type": "limit",
                    "filled_avg_price": 335.10,
                    "qty": 12.0,
                }
            elif oid == "stop-leg-1":
                return {
                    "id": "stop-leg-1",
                    "status": "canceled",
                    "symbol": "AAPL",
                    "side": "sell",
                    "order_type": "stop",
                }
            return None

        mock_bridge.get_order_snapshot.side_effect = mock_snap

        with patch("strategies.trade_journal.log_trade") as mock_log:
            with patch.object(mgr, "_save_state", return_value=True):
                mgr.handle_naked_position("AAPL", 12.0, bridge=mock_bridge)

        assert mock_log.called
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs["symbol"] == "AAPL"
        assert call_kwargs["exit_reason"] == "TAKE_PROFIT"
        assert call_kwargs["exit_price"] == pytest.approx(335.10)
        assert call_kwargs["pnl"] == pytest.approx((335.10 - 331.84) * 12.0)
        assert not state.active
        assert state.reconciliation_status == "RECONCILED_TAKE_PROFIT"

    def test_handle_naked_position_reconciles_stop_out_exit(self, tmp_path):
        """When position is flat and Stop order snapshot is filled, log STOP_OUT with actual fill price."""
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="MSFT",
            side="BUY",
            entry_price=420.00,
            peak_price=421.00,
            atr=2.00,
            stop_price=418.50,
            trail_multiple=1.5,
            last_updated=time.time(),
            order_id=None,
            last_stop_order_id="stop-leg-2",
            tp_order_id="tp-leg-2",
            qty=10.0,
            target_price=425.00,
            active=True,
            strategy="VWAP_BOUNCE",
        )
        mgr.states["MSFT"] = state

        mock_bridge = MagicMock()
        mock_bridge.get_position.return_value = 0.0

        def mock_snap(oid):
            if oid == "stop-leg-2":
                return {
                    "id": "stop-leg-2",
                    "status": "filled",
                    "symbol": "MSFT",
                    "side": "sell",
                    "order_type": "stop",
                    "filled_avg_price": 418.45,
                    "qty": 10.0,
                }
            elif oid == "tp-leg-2":
                return {
                    "id": "tp-leg-2",
                    "status": "canceled",
                    "symbol": "MSFT",
                    "side": "sell",
                    "order_type": "limit",
                }
            return None

        mock_bridge.get_order_snapshot.side_effect = mock_snap

        with patch("strategies.trade_journal.log_trade") as mock_log:
            with patch.object(mgr, "_save_state", return_value=True):
                mgr.handle_naked_position("MSFT", 10.0, bridge=mock_bridge)

        assert mock_log.called
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs["symbol"] == "MSFT"
        assert call_kwargs["exit_reason"] == "STOP_OUT"
        assert call_kwargs["exit_price"] == pytest.approx(418.45)
        assert not state.active
        assert state.reconciliation_status == "RECONCILED_STOP_OUT"

    def test_handle_naked_position_defers_when_fill_not_yet_confirmed(self, tmp_path):
        """When position is flat but fill details are pending, defer close and do NOT log phantom stop-out."""
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="NVDA",
            side="BUY",
            entry_price=120.00,
            peak_price=122.00,
            atr=1.00,
            stop_price=119.00,
            trail_multiple=1.5,
            last_updated=time.time(),
            order_id=None,
            last_stop_order_id="stop-leg-3",
            tp_order_id="tp-leg-3",
            qty=5.0,
            target_price=125.00,
            active=True,
            strategy="MOMENTUM_SCALP",
        )
        mgr.states["NVDA"] = state

        mock_bridge = MagicMock()
        mock_bridge.get_position.return_value = 0.0
        # Snapshots report pending/canceled or None (in-flight state)
        mock_bridge.get_order_snapshot.return_value = None
        mock_bridge._client = None

        with patch("strategies.trade_journal.log_trade") as mock_log:
            with patch.object(mgr, "_save_state", return_value=True):
                mgr.handle_naked_position("NVDA", 5.0, bridge=mock_bridge)

        # MUST NOT log phantom trade
        assert not mock_log.called
        assert not state.active
        assert state.reconciliation_status == "FLAT_PENDING_EXIT_RECONCILIATION"
        assert "NVDA" in mgr.pending_exits

    def test_position_query_failure_retains_active_state(self):
        """When broker position query raises exception, retain active state fail-closed."""
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="AMD",
            side="BUY",
            entry_price=160.00,
            peak_price=162.00,
            atr=2.00,
            stop_price=158.00,
            trail_multiple=1.5,
            last_updated=time.time(),
            order_id=None,
            qty=20.0,
            active=True,
        )
        mgr.states["AMD"] = state

        mock_bridge = MagicMock()
        mock_bridge.get_position.side_effect = ConnectionError("Broker API timeout")

        with patch.object(mgr, "_save_state", return_value=True):
            mgr.handle_naked_position("AMD", 20.0, bridge=mock_bridge)

        # Must retain active state and flag reconciliation required
        assert state.active is True
        assert state.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
        assert state.stop_reconciliation_required is True

    def test_short_position_not_treated_as_flat_on_stop_cancellation(self):
        """When short position (-10 qty) stop is cancelled, ensure_initialized must NOT treat it as flat."""
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="SOXX",
            side="SELL",
            entry_price=220.00,
            peak_price=220.00,
            atr=2.00,
            stop_price=223.00,
            trail_multiple=1.5,
            last_updated=time.time(),
            order_id="stop-short-1",
            qty=10.0,
            active=True,
        )
        mgr.states["SOXX"] = state

        mock_bridge = MagicMock()
        # Broker reports short position (-10.0 shares)
        mock_bridge.get_position.return_value = -10.0
        # Stop order snapshot reports canceled
        mock_bridge.get_order_snapshot.return_value = {
            "id": "stop-short-1",
            "status": "canceled",
            "symbol": "SOXX",
            "side": "buy",
            "order_type": "stop",
        }

        with patch.object(mgr, "_save_state", return_value=True):
            res_state = mgr.ensure_initialized(
                symbol="SOXX",
                side="SELL",
                avg_cost=220.00,
                open_orders=[],
                current_price=221.00,
                qty=10.0,
                bridge=mock_bridge,
            )

        assert res_state is not None
        assert res_state.active is True
        assert res_state.qty == 10.0
        assert res_state.order_id is None
        assert res_state.last_stop_order_id == "stop-short-1"

    def test_two_step_pending_exit_reconciliation_flow(self):
        """Step 1: flat pos with pending fill enters pending_exits. Step 2: fill confirmed commits journal and cleans up."""
        mgr = DynamicTrailingStopManager()
        state = TrailingStopState(
            symbol="META",
            side="BUY",
            entry_price=500.00,
            peak_price=510.00,
            atr=4.00,
            stop_price=495.00,
            trail_multiple=1.5,
            last_updated=time.time(),
            order_id=None,
            last_stop_order_id="stop-meta-1",
            qty=5.0,
            active=True,
            strategy="DAYTRADE_SCALP",
        )
        mgr.states["META"] = state

        mock_bridge = MagicMock()
        mock_bridge.get_position.return_value = 0.0
        # Step 1: Snapshot unavailable / None
        mock_bridge.get_order_snapshot.return_value = None
        mock_bridge._client = None

        with patch("strategies.trade_journal.log_trade") as mock_log:
            with patch.object(mgr, "_save_state", return_value=True):
                mgr.handle_naked_position("META", 5.0, bridge=mock_bridge)

        assert not mock_log.called
        assert "META" in mgr.pending_exits
        assert not state.active
        assert state.reconciliation_status == "FLAT_PENDING_EXIT_RECONCILIATION"

        # Step 2: Snapshot now returns filled stop
        mock_bridge.get_order_snapshot.return_value = {
            "id": "stop-meta-1",
            "status": "filled",
            "symbol": "META",
            "side": "sell",
            "order_type": "stop",
            "filled_avg_price": 494.80,
            "qty": 5.0,
        }

        with patch("strategies.trade_journal.log_trade") as mock_log:
            with patch.object(mgr, "_save_state", return_value=True):
                mgr.reconcile_pending_exits(bridge=mock_bridge)

        assert mock_log.called
        call_kwargs = mock_log.call_args[1]
        assert call_kwargs["symbol"] == "META"
        assert call_kwargs["exit_reason"] == "STOP_OUT"
        assert call_kwargs["exit_price"] == pytest.approx(494.80)
        assert "META" not in mgr.pending_exits
        assert state.reconciliation_status == "RECONCILED_STOP_OUT"

    def test_journal_rejects_nan_and_inf_exit_prices(self):
        """trade_journal.log_trade must fail closed on NaN or Inf exit prices."""
        from strategies.trade_journal import log_trade
        with pytest.raises(ValueError, match="finite"):
            log_trade(
                symbol="AAPL",
                side="SELL",
                strategy="DAYTRADE_SCALP",
                qty=10,
                entry_price=150.0,
                stop_price=145.0,
                target_price=160.0,
                exit_price=float("nan"),
                exit_reason="STOP_OUT",
                pnl=-50.0,
            )

        with pytest.raises(ValueError, match="finite"):
            log_trade(
                symbol="AAPL",
                side="SELL",
                strategy="DAYTRADE_SCALP",
                qty=10,
                entry_price=150.0,
                stop_price=145.0,
                target_price=160.0,
                exit_price=float("inf"),
                exit_reason="STOP_OUT",
                pnl=-50.0,
            )

