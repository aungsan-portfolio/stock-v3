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
        """Verify unparsed timestamps safely fallback to index [-2]."""
        bars = [
            MagicMock(close=50.0, timestamp=None, date=None),
            MagicMock(close=55.0, timestamp=None, date=None),
        ]
        prev_close, _, dt_str = _extract_prev_session_close(bars, today_date=date(2026, 10, 9))
        assert prev_close == pytest.approx(50.0)
        assert dt_str == "unparsed_index_fallback"


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
