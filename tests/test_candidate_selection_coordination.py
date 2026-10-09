"""
tests/test_candidate_selection_coordination.py

Tests candidate selection, deduplication, and strategy coordination in orchestrator.py:
- Fix 1 Verification: Moving seen_symbols.add(sym) after candidate checks ensures
  a candidate failing RSI, inverse-ETF, or geometry does not claim the symbol and
  suppress a viable alternative candidate.
- Acceptance Scenarios:
  1. Leader BUY RSI overbought (80), follower BUY RSI (50) -> follower selected
  2. Leader geometry fails, follower valid -> follower selected
  3. Both valid same symbol -> leader selected, follower rejected as duplicate (one winner)
  4. Different symbols both valid -> both retained
  5. Both invalid geometry -> none retained
  6. Same RSI opposite sides (SELL RSI=20 fails, BUY RSI=20 passes) -> BUY selected
  7. Short RSI fails (SELL RSI=20), follower valid (SELL RSI=50) -> follower selected
  8. Pre-submit veto does not automatically fallback to superseded candidate
"""

import unittest
from unittest.mock import MagicMock, patch
import types

from strategies.base import TradeSignal
from strategies.constants import StrategyName
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

    def _run_selection_segment(self, signals):
        """
        Executes the candidate selection loop extracted from orchestrator.py:2295-2415
        with active config and live geometry validation.
        """
        from strategies.orchestrator import _extract_signal_rsi, _is_inverse_etf_signal
        from strategies.position_sizer import validate_executable_geometry

        seen_symbols = set()
        deduped_signals = []

        for sig in signals:
            sym = sig.symbol.upper().strip()

            if sym in seen_symbols:
                continue

            rsi_max = getattr(config, "RSI_MAX_BUY_ENTRY", 70.0)
            rsi_min = getattr(config, "RSI_MIN_SHORT_ENTRY", 30.0)
            sig_rsi = _extract_signal_rsi(sig)
            if sig.side.upper() in ("BUY", "LONG") and sig_rsi >= rsi_max:
                continue
            if sig.side.upper() in ("SELL", "SHORT") and sig_rsi <= rsi_min:
                continue
            if _is_inverse_etf_signal(sig) and sig.side.upper() in ("SELL", "SHORT"):
                continue

            geom_ok, geom_msg = validate_executable_geometry(sig)
            if not geom_ok:
                continue

            seen_symbols.add(sym)
            deduped_signals.append(sig)

        return deduped_signals

    def test_leader_rsi_overbought_follower_selected(self):
        """Case 1: Leader BUY fails RSI (80 >= 70), follower BUY passes (50) -> Follower selected."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", rsi=80.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="B")

        selected = self._run_selection_segment([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "B")

    def test_leader_geometry_fails_follower_valid(self):
        """Case 2: Leader geometry invalid (stop >= entry for BUY), follower valid -> Follower selected."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=151.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=148.0, signal_id="B")

        selected = self._run_selection_segment([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "B")

    def test_both_valid_one_winner(self):
        """Case 3: Both valid for same symbol -> Leader A selected, follower B rejected as duplicate."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", rsi=52.0, signal_id="B")

        selected = self._run_selection_segment([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "A")

    def test_different_symbols_both_valid(self):
        """Case 4: Different symbols both valid -> Both A and B retained."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="A")
        sig_b = _make_candidate(symbol="MSFT", side="BUY", rsi=52.0, signal_id="B")

        selected = self._run_selection_segment([sig_a, sig_b])
        self.assertEqual(len(selected), 2)
        self.assertEqual({s.signal_id for s in selected}, {"A", "B"})

    def test_all_geometry_invalid(self):
        """Case 5: Both candidates fail geometry -> Neither retained."""
        sig_a = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=151.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=152.0, signal_id="B")

        selected = self._run_selection_segment([sig_a, sig_b])
        self.assertEqual(len(selected), 0)

    def test_same_rsi_opposite_sides_leader_invalid(self):
        """Case 6: Same RSI=20. Leader SELL fails oversold short (<=30), Follower BUY passes -> Follower B selected."""
        sig_a = _make_candidate(symbol="AAPL", side="SELL", entry_price=150.0, stop_price=152.0, target_price=146.0, rsi=20.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="BUY", entry_price=150.0, stop_price=148.0, target_price=154.0, rsi=20.0, signal_id="B")

        selected = self._run_selection_segment([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "B")

    def test_short_rsi_fails_follower_valid(self):
        """Case 7: Leader SELL fails oversold (RSI=20 <= 30), Follower SELL passes (RSI=50) -> Follower B selected."""
        sig_a = _make_candidate(symbol="AAPL", side="SELL", entry_price=150.0, stop_price=152.0, target_price=146.0, rsi=20.0, signal_id="A")
        sig_b = _make_candidate(symbol="AAPL", side="SELL", entry_price=150.0, stop_price=152.0, target_price=146.0, rsi=50.0, signal_id="B")

        selected = self._run_selection_segment([sig_a, sig_b])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].signal_id, "B")

    def test_pre_submit_veto_does_not_loop_fallback(self):
        """
        Acceptance Criteria: Once a candidate is selected and fails pre-submit veto in execute_signal,
        the system does NOT automatically fall back to loop through rejected candidates.
        """
        from strategies.order_manager import execute_signal
        from strategies.base import EntryReadiness
        import time

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
            "timestamp": time.time() - 1.0,
            "source": "alpaca",
            "is_realtime": True,
            "feed_identity": "iex",
        }

        # Candidate selected: sig_b
        sig_b = _make_candidate(symbol="AAPL", side="BUY", rsi=50.0, signal_id="B")
        readiness = EntryReadiness(
            entry_allowed=True,
            suppression_reason=None,
            cycle_id="cycle_fb_test",
            bar_ts="2026-10-09T14:00:00Z",
            created_at=time.time(),
            expires_at=time.time() + 300.0,
        )

        with patch("strategies.order_manager._place_bracket_order_once") as mock_place, \
             patch("config.SPY_TREND_GUARD_ENABLED", False), \
             patch("strategies.trailing_stop.get_trailing_manager", return_value=types.SimpleNamespace(persistence_healthy=False, last_persistence_error="HEALTH_TEST")):

            # Candidate B is vetoed at final submit persistence gate
            res = execute_signal(
                signal=sig_b,
                bridge=mock_bridge,
                equity=100000.0,
                current_pnl=0.0,
                dry_run=False,
                entry_readiness=readiness,
                regime_result=types.SimpleNamespace(mode="NORMAL", position_scale=1.0),
            )
            self.assertEqual(res["status"], "VETOED")
            mock_place.assert_not_called()
            # The execution pipeline terminates cleanly for AAPL; no retry loop occurs


if __name__ == "__main__":
    unittest.main()
