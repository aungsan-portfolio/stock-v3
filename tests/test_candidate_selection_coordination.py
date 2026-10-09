"""
tests/test_candidate_selection_coordination.py

Tests candidate selection, deduplication, and strategy coordination:
1. Fix 1 Verification (Production Function): Direct testing of canonical
   `strategies.orchestrator.select_viable_cycle_candidates` across all 7 probe scenarios.
2. Final-Submit Boundary & Non-Fallback Integration: Validates that when a selected
   candidate passes pre-flight but fails the final-submit boundary gate, it is vetoed
   without broker submission, releases reservation, and does NOT trigger an automatic
   retry loop over rejected candidates.
3. Setup-Scoped Suppression Cooldown: Validates that 5 noise-floor vetoes on Strategy A
   trigger a setup-scoped cooldown (symbol, strategy, side) while leaving Strategy B
   on the same symbol completely unsuppressed.
"""

import unittest
from unittest.mock import MagicMock, patch
import time
import types

from strategies.base import TradeSignal
from strategies.constants import StrategyName
from strategies.orchestrator import select_viable_cycle_candidates
import strategies.order_manager as om
import config


def _make_candidate(
    symbol: str = "AAPL",
    strategy: StrategyName = StrategyName.CANDLESTICK,
    side: str = "BUY",
    confidence: float = 0.80,
    entry_price: float = 150.0,
    stop_price: float = 148.0,
    target_price: float = 154.0,
    atr: float = 1.0,
    risk_per_share: float = 2.0,
    rsi: float = 50.0,
    signal_id: str = "sig_test",
) -> TradeSignal:
    sig = TradeSignal(
        symbol=symbol,
        strategy=strategy,
        side=side,
        confidence=confidence,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        atr=atr,
        risk_per_share=risk_per_share,
        reason=f"Candidate test RSI {rsi:.1f}",
        signal_id=signal_id,
        bar_timestamp="2026-10-09T14:00:00Z",
        decision_feed="iex",
    )
    sig.rsi = rsi
    sig.metadata = {"rsi": rsi}
    return sig


class TestCandidateSelectionCoordination(unittest.TestCase):

    def setUp(self):
        om.reset_session_skip_state()
        from strategies.order_idempotency import idempotency_manager
        idempotency_manager.clear()
        with om._pending_order_contexts_lock:
            om._pending_order_contexts.clear()

    def tearDown(self):
        om.reset_session_skip_state()
        from strategies.order_idempotency import idempotency_manager
        idempotency_manager.clear()
        with om._pending_order_contexts_lock:
            om._pending_order_contexts.clear()

    # ── Canonical Orchestrator Candidate Selection Tests ──────────────────

    def test_leader_rsi_overbought_follower_selected(self):
        """Case 1: Leader BUY fails RSI (80 >= 70), follower BUY passes (50) -> Follower selected."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", rsi=80.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="B")

        selected = select_viable_cycle_candidates([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "B")

    def test_leader_geometry_fails_follower_valid(self):
        """Case 2: Leader geometry invalid (stop >= entry for BUY), follower valid -> Follower selected."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=151.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=148.0, signal_id="B")

        selected = select_viable_cycle_candidates([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "B")

    def test_both_valid_one_winner(self):
        """Case 3: Both valid for same symbol -> Leader A selected, follower B rejected as duplicate."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", rsi=52.0, signal_id="B")

        selected = select_viable_cycle_candidates([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "A")

    def test_different_symbols_both_valid(self):
        """Case 4: Different symbols both valid -> Both A and B retained."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="A")
        sig_b = _make_candidate(symbol="MSFT", side="BUY", rsi=52.0, signal_id="B")

        selected = select_viable_cycle_candidates([sig_a, sig_b])
        self.assertEqual(len(selected), 2)
        self.assertEqual({s.signal_id for s in selected}, {"A", "B"})

    def test_all_geometry_invalid(self):
        """Case 5: Both candidates fail geometry -> Neither retained."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=151.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=152.0, signal_id="B")

        selected = select_viable_cycle_candidates([sig_a, sig_b])
        self.assertEqual(len(selected), 0)

    def test_same_rsi_opposite_sides_leader_invalid(self):
        """Case 6: Same RSI=20. Leader SELL fails oversold short (<=30), Follower BUY passes -> Follower B selected."""
        sig_a = _make_candidate(symbol="AAPL", side="SELL", entry_price=150.0, stop_price=152.0, target_price=146.0, rsi=20.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=148.0, target_price=154.0, rsi=20.0, signal_id="B")

        selected = select_viable_cycle_candidates([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "B")

    def test_short_rsi_fails_follower_valid(self):
        """Case 7: Leader SELL fails oversold (RSI=20 <= 30), Follower SELL passes (RSI=50) -> Follower B selected."""
        sig_a = _make_candidate(symbol="AAPL", side="SELL", entry_price=150.0, stop_price=152.0, target_price=146.0, rsi=20.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="SELL", entry_price=150.0, stop_price=152.0, target_price=146.0, rsi=50.0, signal_id="B")

        selected = select_viable_cycle_candidates([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "B")

    # ── Final-Submit Boundary & Non-Fallback Integration ───────────────────

    def test_final_boundary_veto_and_no_fallback_loop(self):
        """
        Tests the true final submit boundary:
        - Call 1 (Pre-flight Phase 0): Manager is healthy (persistence_healthy=True).
        - Call 2 (Final submit boundary Phase C): Manager degrades (persistence_healthy=False).
        - execute_signal vetoes at the final submit boundary without broker call.
        - Pre-submit reservation is cleanly released.
        - The execution terminates fail-closed; no automatic loop retry over superseded candidates.
        """
        from strategies.order_manager import execute_signal, idempotency_manager
        from strategies.base import EntryReadiness

        call_count = [0]

        def get_manager_two_phase():
            call_count[0] += 1
            if call_count[0] == 1:
                # Pre-flight Phase 0 passes
                return types.SimpleNamespace(persistence_healthy=True, last_persistence_error=None)
            else:
                # Final-submit Phase C degrades
                return types.SimpleNamespace(persistence_healthy=False, last_persistence_error="FINAL_SUBMIT_DEGRADED")

        from datetime import datetime, timezone

        now = time.time()
        completed_bar_ts = datetime.fromtimestamp(now - 330.0, tz=timezone.utc).isoformat()

        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.open_position_count.return_value = 0
        mock_bridge.list_positions.return_value = []
        mock_bridge.has_position.return_value = False
        mock_bridge.has_working_order.return_value = False
        mock_bridge.get_quote.return_value = {
            "symbol": "AAPL",
            "bid": 150.00,
            "ask": 150.05,
            "last": 150.02,
            "timestamp": now - 1.0,
            "source": "alpaca",
            "is_realtime": True,
            "feed_identity": "iex",
        }

        sig_b = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="B")
        sig_b.bar_timestamp = completed_bar_ts
        readiness = EntryReadiness(
            entry_allowed=True,
            suppression_reason=None,
            cycle_id="cycle_two_phase_test",
            bar_ts=completed_bar_ts,
            created_at=now,
            expires_at=now + 300.0,
        )

        with patch("strategies.trailing_stop.get_trailing_manager", side_effect=get_manager_two_phase), \
             patch("strategies.order_manager._place_bracket_order_once") as mock_place, \
             patch("config.SPY_TREND_GUARD_ENABLED", False), \
             patch("strategies.intraday_risk.get_recent_symbol_loss_time", return_value=None):

            res = execute_signal(
                signal=sig_b,
                bridge=mock_bridge,
                equity=100000.0,
                current_pnl=0.0,
                dry_run=False,
                entry_readiness=readiness,
                regime_result=types.SimpleNamespace(mode="NORMAL", position_scale=1.0),
            )

            # Verification of final-submit boundary trigger
            self.assertEqual(res["status"], "VETOED")
            self.assertIn("Final Submit", res["reason"])
            self.assertIn("FINAL_SUBMIT_DEGRADED", res["reason"])

            # Broker submission must NEVER be reached
            mock_place.assert_not_called()
            mock_bridge.submit_order.assert_not_called()

            # Pre-submit idempotency reservation must be cleanly released
            self.assertNotIn(("AAPL", "BUY"), idempotency_manager.reservations)

    # ── Setup-Scoped Cooldown vs Cross-Strategy Suppression ────────────────

    def test_setup_scoped_cooldown_does_not_block_other_strategies(self):
        """
        Tests the Two-Tier Suppression Architecture:
        - Strategy A (CANDLESTICK, BUY) triggers 5 noise-floor skips on AAPL.
        - Strategy A (CANDLESTICK, BUY) is placed in 45m cooldown.
        - Strategy B (MEAN_REVERSION, BUY) on AAPL is NOT suppressed.
        - Strategy A (CANDLESTICK, SELL) on AAPL is NOT suppressed.
        - Whole-symbol hazard correctly suppresses ALL strategies on AAPL.
        """
        sym = "AAPL"
        reason = "Noise floor critically tight: stop 0.05 < 0.10"

        # Record 5 setup skips for CANDLESTICK BUY on AAPL
        for i in range(5):
            om.record_hard_veto_skip(
                sym,
                reason,
                bar_timestamp=f"bar_{i}",
                strategy=StrategyName.CANDLESTICK,
                side="BUY",
            )

        # 1. Strategy A (CANDLESTICK, BUY) must be suppressed in cooldown
        suppressed_a, msg_a = om.check_session_suppression(
            sym, strategy=StrategyName.CANDLESTICK, side="BUY"
        )
        self.assertTrue(suppressed_a)
        self.assertIn("[SESSION SUPPRESSED] AAPL [CANDLESTICK:BUY]: cooldown active", msg_a)

        # 2. Strategy B (MEAN_REVERSION, BUY) on AAPL must NOT be suppressed
        suppressed_b, msg_b = om.check_session_suppression(
            sym, strategy=StrategyName.MEAN_REVERSION, side="BUY"
        )
        self.assertFalse(suppressed_b, f"Strategy B should be unsuppressed, got: {msg_b}")
        self.assertEqual(msg_b, "")

        # 3. Strategy A (CANDLESTICK, SELL) on AAPL must NOT be suppressed
        suppressed_a_sell, msg_a_sell = om.check_session_suppression(
            sym, strategy=StrategyName.CANDLESTICK, side="SELL"
        )
        self.assertFalse(suppressed_a_sell, f"Opposite side should be unsuppressed, got: {msg_a_sell}")

        # 4. Global / whole-symbol hazard must suppress all strategies on the symbol
        om._session_permanently_suppressed.add(sym)
        suppressed_b_global, msg_b_global = om.check_session_suppression(
            sym, strategy=StrategyName.MEAN_REVERSION, side="BUY"
        )
        self.assertTrue(suppressed_b_global)
        self.assertIn("permanently suppressed this session", msg_b_global)

    # ── Candidate-Local Veto Fallback vs Shared Safety Hazard Invariants ───

    def test_candidate_local_veto_permits_independent_candidate_execution(self):
        """
        Acceptance Criteria (Item 2 - Local Veto Alternative Execution):
        - Candidate A (AAPL) fails a candidate-local pre-submit gate (e.g. quote spread too wide).
        - Candidate A is vetoed fail-closed before broker submission; reservation released.
        - Cycle proceeds to evaluate Candidate B (MSFT).
        - Candidate B passes all candidate-specific checks and shared safety gates.
        - Candidate B is successfully submitted ('PLACED'); broker submits MSFT.
        """
        from strategies.order_manager import execute_signal, idempotency_manager
        from strategies.base import EntryReadiness
        from datetime import datetime, timezone
        import time

        now = time.time()
        completed_bar_ts = datetime.fromtimestamp(now - 330.0, tz=timezone.utc).isoformat()

        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.is_paper = True
        mock_bridge.account_mode = "PAPER"
        mock_bridge.open_position_count.return_value = 0
        mock_bridge.has_position.return_value = False
        mock_bridge.has_working_order.return_value = False
        mock_bridge.get_positions.return_value = []

        def get_quote_for_sym(sym):
            if sym == "AAPL":
                # Candidate A: wide spread 150.00 vs 155.00 (> 1.0% spread -> candidate-local veto)
                return {
                    "symbol": "AAPL",
                    "bid": 150.00,
                    "ask": 155.00,
                    "last": 150.02,
                    "timestamp": now - 1.0,
                    "source": "alpaca",
                    "is_realtime": True,
                    "feed_identity": "iex",
                }
            else:
                # Candidate B: normal valid quote
                return {
                    "symbol": "MSFT",
                    "bid": 300.00,
                    "ask": 300.05,
                    "last": 300.02,
                    "timestamp": now - 1.0,
                    "source": "alpaca",
                    "is_realtime": True,
                    "feed_identity": "iex",
                }

        mock_bridge.get_quote.side_effect = get_quote_for_sym

        sig_a = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=148.0, target_price=154.0, risk_per_share=2.0, rsi=50.0, signal_id="A")
        sig_a.bar_timestamp = completed_bar_ts
        sig_b = _make_candidate(symbol="MSFT", side="BUY", entry_price=300.0, stop_price=296.0, target_price=308.0, risk_per_share=4.0, rsi=50.0, signal_id="B")
        sig_b.bar_timestamp = completed_bar_ts

        readiness = EntryReadiness(
            entry_allowed=True,
            suppression_reason=None,
            cycle_id="cycle_local_veto_test",
            bar_ts=completed_bar_ts,
            created_at=now,
            expires_at=now + 300.0,
        )

        healthy_manager = types.SimpleNamespace(persistence_healthy=True, last_persistence_error=None)

        with patch("strategies.trailing_stop.get_trailing_manager", return_value=healthy_manager), \
             patch("strategies.order_manager._place_bracket_order_once", return_value={"status": "PLACED", "order_id": "order_msft_1", "stop_order_id": "stop_msft_1", "tp_order_id": "tp_msft_1"}) as mock_place, \
             patch("config.SPY_TREND_GUARD_ENABLED", False), \
             patch("strategies.intraday_risk.get_recent_symbol_loss_time", return_value=None):

            # 1. Candidate A encounters candidate-local spread veto
            res_a = execute_signal(
                signal=sig_a,
                bridge=mock_bridge,
                equity=100000.0,
                current_pnl=0.0,
                dry_run=False,
                entry_readiness=readiness,
                regime_result=types.SimpleNamespace(mode="NORMAL", position_scale=1.0),
            )
            self.assertIn(res_a["status"], ("VETOED", "REJECTED"))
            self.assertIn("spread", res_a["reason"].lower(), f"Expected spread veto reason, got: {res_a['reason']}")
            self.assertNotIn(("AAPL", "BUY"), idempotency_manager.reservations)

            # 2. Cycle proceeds to evaluate Candidate B (valid alternative)
            res_b = execute_signal(
                signal=sig_b,
                bridge=mock_bridge,
                equity=100000.0,
                current_pnl=0.0,
                dry_run=False,
                entry_readiness=readiness,
                regime_result=types.SimpleNamespace(mode="NORMAL", position_scale=1.0),
            )
            self.assertEqual(res_b["status"], "PLACED", f"Candidate B rejected unexpectedly: {res_b.get('reason')}")
            mock_place.assert_called_once()
            self.assertEqual(mock_place.call_args[1]["symbol"], "MSFT")

    def test_shared_safety_hazard_terminates_and_blocks_subsequent_candidates(self):
        """
        Acceptance Criteria (Item 2 - Shared Hazard Non-Bypass):
        - When a shared engine-wide hazard occurs (persistence manager degraded),
        - Candidate A is vetoed fail-closed before broker submission.
        - Candidate B (subsequent candidate in the cycle) is ALSO blocked by the shared hazard.
        - Broker receives ZERO orders for either Candidate A or Candidate B.
        """
        from strategies.order_manager import execute_signal, idempotency_manager
        from strategies.base import EntryReadiness
        from datetime import datetime, timezone
        import time

        now = time.time()
        completed_bar_ts = datetime.fromtimestamp(now - 330.0, tz=timezone.utc).isoformat()

        mock_bridge = MagicMock()
        mock_bridge.is_connected = True
        mock_bridge.is_paper = True
        mock_bridge.account_mode = "PAPER"
        mock_bridge.open_position_count.return_value = 0
        mock_bridge.has_position.return_value = False
        mock_bridge.has_working_order.return_value = False
        mock_bridge.get_positions.return_value = []
        mock_bridge.get_quote.return_value = {
            "symbol": "TEST",
            "bid": 150.00,
            "ask": 150.05,
            "last": 150.02,
            "timestamp": now - 1.0,
            "source": "alpaca",
            "is_realtime": True,
            "feed_identity": "iex",
        }

        sig_a = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="A")
        sig_a.bar_timestamp = completed_bar_ts
        sig_b = _make_candidate(symbol="MSFT", side="BUY", rsi=50.0, signal_id="B")
        sig_b.bar_timestamp = completed_bar_ts

        readiness = EntryReadiness(
            entry_allowed=True,
            suppression_reason=None,
            cycle_id="cycle_shared_hazard_test",
            bar_ts=completed_bar_ts,
            created_at=now,
            expires_at=now + 300.0,
        )

        degraded_manager = types.SimpleNamespace(persistence_healthy=False, last_persistence_error="SHARED_PERSISTENCE_DOWN")

        with patch("strategies.trailing_stop.get_trailing_manager", return_value=degraded_manager), \
             patch("strategies.order_manager._place_bracket_order_once") as mock_place, \
             patch("config.SPY_TREND_GUARD_ENABLED", False), \
             patch("strategies.intraday_risk.get_recent_symbol_loss_time", return_value=None):

            # 1. Candidate A hits shared safety hazard
            res_a = execute_signal(
                signal=sig_a,
                bridge=mock_bridge,
                equity=100000.0,
                current_pnl=0.0,
                dry_run=False,
                entry_readiness=readiness,
                regime_result=types.SimpleNamespace(mode="NORMAL", position_scale=1.0),
            )
            self.assertIn(res_a["status"], ("VETOED", "REJECTED"))
            self.assertIn("SHARED_PERSISTENCE_DOWN", res_a["reason"])

            # 2. Candidate B cannot bypass the shared safety hazard
            res_b = execute_signal(
                signal=sig_b,
                bridge=mock_bridge,
                equity=100000.0,
                current_pnl=0.0,
                dry_run=False,
                entry_readiness=readiness,
                regime_result=types.SimpleNamespace(mode="NORMAL", position_scale=1.0),
            )
            self.assertIn(res_b["status"], ("VETOED", "REJECTED"))
            self.assertIn("SHARED_PERSISTENCE_DOWN", res_b["reason"])

            # ZERO broker orders placed across both candidates
            mock_place.assert_not_called()
            mock_bridge.submit_order.assert_not_called()
            self.assertNotIn(("AAPL", "BUY"), idempotency_manager.reservations)
            self.assertNotIn(("MSFT", "BUY"), idempotency_manager.reservations)


if __name__ == "__main__":
    unittest.main()
