import sys
import os
sys.path.insert(0, os.path.abspath("."))
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

def test_5m_bar_cadence_gate():
    import strategies.orchestrator as orch

    # Reset module-level cadence state
    orch._last_evaluated_5m_bar_str = None

    mock_bridge = MagicMock()
    mock_bridge._connected = True
    mock_bridge.is_connected = True
    mock_bridge.ib.positions.return_value = []
    mock_bridge.get_net_liquidation.return_value = 100000.0

    eval_count = 0
    def mock_eval_parallel(watchlist, bridge=None):
        nonlocal eval_count
        eval_count += 1
        return []

    with patch("strategies.orchestrator.evaluate_symbols_parallel", side_effect=mock_eval_parallel):
        # Tick 1: 20:30:00 (5m boundary)
        t1 = datetime(2026, 7, 24, 20, 30, 0, tzinfo=timezone.utc)
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)
            assert eval_count == 1, f"Expected 1 evaluation on 20:30:00, got {eval_count}"

        # Tick 2: 20:30:30 (Intra-bar)
        t2 = datetime(2026, 7, 24, 20, 30, 30, tzinfo=timezone.utc)
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t2
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)
            assert eval_count == 1, f"Expected signal evaluation skipped on intra-bar tick 20:30:30, got count={eval_count}"

        # Tick 3: 20:31:00 (Intra-bar)
        t3 = datetime(2026, 7, 24, 20, 31, 0, tzinfo=timezone.utc)
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t3
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)
            assert eval_count == 1, f"Expected signal evaluation skipped on intra-bar tick 20:31:00, got count={eval_count}"

        # Tick 4: 20:35:00 (Next 5m boundary)
        t4 = datetime(2026, 7, 24, 20, 35, 0, tzinfo=timezone.utc)
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t4
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)
            assert eval_count == 2, f"Expected 2nd evaluation on next 5m boundary 20:35:00, got count={eval_count}"

    print(f"[PASS Phase 5] 5m Bar Cadence Gate Verified: 4 ticks tested, signal eval ran ONLY on 2 bar boundaries (20:30 and 20:35), skipped intra-bar ticks.")

if __name__ == "__main__":
    test_5m_bar_cadence_gate()
