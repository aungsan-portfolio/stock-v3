"""
tests/test_hysteresis_invariants.py

Formal test suite verifying production RegimeStateTracker hysteresis invariants
and orchestrator caller wiring.

Invariants asserted:
1. Direct production import verification (inspect.getfile(RegimeStateTracker))
2. Monotonic bar acceptance & duplicate intra-bar poll rejection (20x intra-bar polls)
3. Out-of-order bar rejection (10:00 -> 10:05 -> 10:00 preserves watermark)
4. Gap streak reset on missing bar interval (> 360s)
5. Immediate intra-bar emergency risk halt (0 delay, fires on intra-bar ticks)
6. Degraded mode recovery watermark preservation
7. Dwell window transition after N=3 distinct completed bars
8. Liveness during entry halts (dwell continues observing completed bars while entries blocked)
"""

import inspect
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import pytest

from strategies.regime_engine import (
    RegimeStateTracker,
    MarketRegimeResult,
    MODE_RISK_ON,
    MODE_CAUTION,
    MODE_RISK_OFF,
    MODE_DATA_DEGRADED,
    TEXTURE_TRENDING,
    TEXTURE_VOLATILE,
    TEXTURE_TRANSITIONING,
)


def _make_result(
    mode=MODE_RISK_ON,
    texture=TEXTURE_TRENDING,
    score=2,
    halt=False,
    allow_longs=True,
    reasons=None,
) -> MarketRegimeResult:
    return MarketRegimeResult(
        mode=mode,
        score=score,
        allow_new_longs=allow_longs,
        requires_opening_buffer=False,
        position_scale=1.0 if mode == MODE_RISK_ON else (0.5 if mode == MODE_CAUTION else 0.0),
        confidence_boost=0.0,
        allow_inverse_etf_longs=False,
        reasons=reasons or ["Healthy market"],
        factor_scores={},
        data={},
        evaluated_at=datetime.now(timezone.utc),
        scanner_candidate_count=10,
        max_trades_day=5,
        halt_new_entries=halt,
        market_texture=texture,
    )


def test_production_import_provenance():
    """Verify that RegimeStateTracker is imported directly from strategies/regime_engine.py."""
    source_file = inspect.getfile(RegimeStateTracker).replace("\\", "/")
    assert source_file.endswith("strategies/regime_engine.py"), (
        f"Expected RegimeStateTracker to originate from strategies/regime_engine.py, got: {source_file}"
    )


def test_monotonic_acceptance_and_duplicate_poll_rejection():
    """
    Verify that 20 intra-bar polls for the same completed bar boundary
    only advance candidate_texture_count once.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3)
    bar_ts = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)

    # Initial bootstrap bar: TRENDING
    res1 = tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=bar_ts, is_completed_bar_boundary=True)
    assert res1.market_texture == TEXTURE_TRENDING
    assert tracker.active_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 1
    assert tracker.last_processed_bar_dt == bar_ts

    # Next completed bar: VOLATILE
    bar_ts_2 = datetime(2026, 10, 12, 10, 5, 0, tzinfo=timezone.utc)
    res2 = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=bar_ts_2, is_completed_bar_boundary=True)
    assert res2.market_texture == TEXTURE_TRENDING  # Dwell active: still TRENDING
    assert tracker.candidate_texture == TEXTURE_VOLATILE
    assert tracker.candidate_texture_count == 1
    assert tracker.last_processed_bar_dt == bar_ts_2

    # Now poll 20 times within this same bar (intra-bar ticks, is_completed_bar_boundary=False or True)
    for _ in range(20):
        res_intra = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=bar_ts_2, is_completed_bar_boundary=False)
        assert res_intra.market_texture == TEXTURE_TRENDING
        assert tracker.candidate_texture_count == 1  # MUST NOT ADVANCE

    # Even if caller erroneously passed is_completed_bar_boundary=True with same bar timestamp, watermark rejects duplicate
    res_dup = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=bar_ts_2, is_completed_bar_boundary=True)
    assert res_dup.market_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 1


def test_out_of_order_bar_rejection():
    """
    Verify out-of-order sequence (10:00 -> 10:05 -> 10:00) rejects older bar
    and does not move watermark backwards.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3)
    t_10_00 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
    t_10_05 = datetime(2026, 10, 12, 10, 5, 0, tzinfo=timezone.utc)

    tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=t_10_00, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t_10_00

    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_10_05, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t_10_05
    assert tracker.candidate_texture_count == 1

    # Stale/out-of-order bar arriving late: 10:00 <= 10:05 -> REJECT
    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_10_00, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t_10_05  # Watermark unchanged!
    assert tracker.candidate_texture_count == 1      # Streak not advanced!


def test_gap_resets_consecutive_streak():
    """
    Verify interval gap (delta > 360s, e.g. 15 minutes) resets candidate streak to 1.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3)
    t0 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 12, 10, 5, 0, tzinfo=timezone.utc)
    t2_gap = datetime(2026, 10, 12, 10, 20, 0, tzinfo=timezone.utc)  # 15m gap > 360s

    tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=t0, is_completed_bar_boundary=True)
    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t1, is_completed_bar_boundary=True)
    assert tracker.candidate_texture_count == 1

    # Third observation arrives after 15m gap: resets streak to 1 instead of reaching 2
    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t2_gap, is_completed_bar_boundary=True)
    assert tracker.candidate_texture_count == 1
    assert tracker.active_texture == TEXTURE_TRENDING  # Transition blocked


def test_immediate_intra_bar_emergency_risk_halt():
    """
    Verify emergency RISK_OFF and halt_new_entries fire immediately on intra-bar ticks
    with zero delay and without waiting for dwell window or bar boundaries.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3)
    t0 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
    tracker.update(_make_result(mode=MODE_RISK_ON, texture=TEXTURE_TRENDING), completed_bar_ts=t0, is_completed_bar_boundary=True)

    # Intra-bar tick 30 seconds later: Sudden RISK_OFF mode
    t_intra = datetime(2026, 10, 12, 10, 0, 30, tzinfo=timezone.utc)
    risk_off_res = _make_result(
        mode=MODE_RISK_OFF,
        texture=TEXTURE_VOLATILE,
        score=-3,
        halt=True,
        allow_longs=False,
        reasons=["Sudden market flash crash"],
    )
    res = tracker.update(risk_off_res, completed_bar_ts=t0, is_completed_bar_boundary=False)

    # Emergency halt MUST fire immediately
    assert res.mode == MODE_RISK_OFF
    assert res.halt_new_entries is True
    assert res.allow_new_longs is False


def test_degraded_recovery_preserves_watermark():
    """
    Verify degraded mode preserves watermark so that recovery rejects duplicate or older bars.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3, recovery_bars=3)
    t0 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
    tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=t0, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t0

    # Degraded event occurs
    deg_res = _make_result(mode=MODE_DATA_DEGRADED, texture=TEXTURE_VOLATILE, halt=True, allow_longs=False)
    tracker.update(deg_res)
    assert tracker.last_processed_bar_dt == t0  # Watermark retained

    # 3 consecutive healthy recovery polls
    healthy = _make_result(mode=MODE_RISK_ON, texture=TEXTURE_TRENDING)
    tracker.update(healthy)
    tracker.update(healthy)
    res_rec = tracker.update(healthy)
    assert res_rec.mode == MODE_RISK_ON

    # Save candidate count before feeding stale bar
    count_before = tracker.candidate_texture_count

    # Now feed the old bar t0 again -> must be rejected by watermark (t0 <= last_processed_bar_dt)
    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t0, is_completed_bar_boundary=True)
    assert tracker.candidate_texture_count == count_before  # Not incremented!
    assert tracker.last_processed_bar_dt == t0

    # Newer bar t1 > t0 is accepted
    t1 = t0 + timedelta(minutes=5)
    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t1, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t1


def test_dwell_window_transition_confirmation():
    """
    Verify texture flips from TRENDING to VOLATILE only after 3 consecutive distinct completed bars.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3)
    t0 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 12, 10, 5, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 12, 10, 10, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 10, 12, 10, 15, 0, tzinfo=timezone.utc)

    # Bootstrap bar: TRENDING
    tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=t0, is_completed_bar_boundary=True)
    assert tracker.active_texture == TEXTURE_TRENDING

    # Bar 1 VOLATILE (count=1)
    res1 = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t1, is_completed_bar_boundary=True)
    assert res1.market_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 1

    # Bar 2 VOLATILE (count=2)
    res2 = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t2, is_completed_bar_boundary=True)
    assert res2.market_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 2

    # Bar 3 VOLATILE (count=3 -> Confirmed transition!)
    res3 = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t3, is_completed_bar_boundary=True)
    assert res3.market_texture == TEXTURE_VOLATILE
    assert tracker.active_texture == TEXTURE_VOLATILE
    assert tracker.candidate_texture_count == 0


def test_liveness_during_active_entry_halt():
    """
    Verify that when entry authorization is blocked (e.g. entry halt),
    texture dwell liveness continues observing completed bars.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3)
    t0 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 12, 10, 5, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 10, 12, 10, 10, 0, tzinfo=timezone.utc)
    t3 = datetime(2026, 10, 12, 10, 15, 0, tzinfo=timezone.utc)

    tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=t0, is_completed_bar_boundary=True)

    # Entry halt is active on all subsequent bars (halt_new_entries=True)
    r1 = _make_result(texture=TEXTURE_VOLATILE, halt=True, allow_longs=False)
    r2 = _make_result(texture=TEXTURE_VOLATILE, halt=True, allow_longs=False)
    r3 = _make_result(texture=TEXTURE_VOLATILE, halt=True, allow_longs=False)

    tracker.update(r1, completed_bar_ts=t1, is_completed_bar_boundary=True)
    tracker.update(r2, completed_bar_ts=t2, is_completed_bar_boundary=True)
    res3 = tracker.update(r3, completed_bar_ts=t3, is_completed_bar_boundary=True)

    # Texture transition STILL completes (liveness preserved)
    assert res3.market_texture == TEXTURE_VOLATILE
    assert tracker.active_texture == TEXTURE_VOLATILE
    # But entries remain halted
    assert res3.halt_new_entries is True
    assert res3.allow_new_longs is False


def test_orchestrator_caller_cadence_wiring():
    """
    Verify orchestrator evaluate_and_execute correctly wires completed_bar_ts
    and is_completed_bar_boundary to _regime_state_tracker.update().
    """
    import strategies.orchestrator as orch

    orch._regime_state_tracker.reset()
    orch._last_regime_bar_str = None
    orch._last_evaluated_5m_bar_str = None

    mock_bridge = MagicMock()
    mock_bridge._connected = True
    mock_bridge.is_connected = True
    mock_bridge.ib.positions.return_value = []
    mock_bridge.get_net_liquidation.return_value = 100000.0

    mock_regime = _make_result(mode=MODE_RISK_ON, texture=TEXTURE_TRENDING)

    with patch("strategies.regime_engine.evaluate_market_regime", return_value=mock_regime), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=[]):

        # Poll 1: 10:00:15 UTC (boundary for completed 09:55:00 bar)
        t1 = datetime(2026, 10, 12, 10, 0, 15, tzinfo=timezone.utc)
        expected_completed_1 = datetime(2026, 10, 12, 9, 55, 0, tzinfo=timezone.utc)
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)

        assert orch._regime_state_tracker.last_processed_bar_dt == expected_completed_1
        assert orch._regime_state_tracker.active_texture == TEXTURE_TRENDING

        # Poll 2: 10:00:45 UTC (intra-bar tick 30s later)
        t2 = datetime(2026, 10, 12, 10, 0, 45, tzinfo=timezone.utc)
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t2
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)

        # Intra-bar poll must NOT advance or corrupt watermark
        assert orch._regime_state_tracker.last_processed_bar_dt == expected_completed_1

        # Poll 3: 10:05:10 UTC (next completed bar boundary for 10:00:00 bar)
        t3 = datetime(2026, 10, 12, 10, 5, 10, tzinfo=timezone.utc)
        expected_completed_2 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t3
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)

        assert orch._regime_state_tracker.last_processed_bar_dt == expected_completed_2


def test_missing_timestamp_negative_does_not_advance_dwell():
    """
    Negative test for auditor probe:
    Verify that in strict mode, calling tracker.update(result) repeatedly without
    completed_bar_ts does NOT advance candidate count and does NOT change active texture.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3, allow_unanchored_advance=False)
    t0 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)

    # Initial valid completed bar bootstraps to TRENDING
    tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=t0, is_completed_bar_boundary=True)
    assert tracker.active_texture == TEXTURE_TRENDING
    assert tracker.last_processed_bar_dt == t0
    initial_count = tracker.candidate_texture_count

    # Now call 5 times with candidate texture VOLATILE, but NO timestamp passed
    for _ in range(5):
        res = tracker.update(_make_result(texture=TEXTURE_VOLATILE))  # completed_bar_ts=None
        # Active texture MUST remain TRENDING, count MUST NOT advance!
        assert tracker.active_texture == TEXTURE_TRENDING
        assert tracker.candidate_texture_count == initial_count
        assert res.market_texture == TEXTURE_TRENDING


def test_sub_interval_bars_rejected():
    """
    Negative test for auditor probe:
    Verify that 1-second timestamps (sub-interval noise < 150s for 300s bars)
    are rejected and do NOT advance dwell streak or watermark.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3, bar_interval_seconds=300.0)
    t0 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)

    tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=t0, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t0

    # 1-second step (sub-interval noise)
    t_1s = datetime(2026, 10, 12, 10, 0, 1, tzinfo=timezone.utc)
    res_1s = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_1s, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t0  # Watermark retained!
    assert tracker.candidate_texture_count == 1  # Streak NOT advanced!
    assert res_1s.market_texture == TEXTURE_TRENDING

    # Another 1-second step
    t_2s = datetime(2026, 10, 12, 10, 0, 2, tzinfo=timezone.utc)
    res_2s = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_2s, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t0  # Watermark retained!
    assert tracker.candidate_texture_count == 1  # Streak NOT advanced!
    assert res_2s.market_texture == TEXTURE_TRENDING

    # Legitimate 5-minute bar step (300s) -> Accepted!
    t_5m = datetime(2026, 10, 12, 10, 5, 0, tzinfo=timezone.utc)
    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_5m, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t_5m
    assert tracker.candidate_texture_count == 1


def test_orchestrator_retries_boundary_within_publication_window_if_first_poll_degraded():
    """
    Verify that if the first poll at a 5m boundary encounters DATA_DEGRADED,
    the boundary watermark is NOT prematurely consumed, allowing subsequent polls
    within the publication timeout window (e.g. 90s) to successfully retry and accept
    the observation once market data recovers.
    """
    import strategies.orchestrator as orch

    orch._regime_state_tracker.reset()
    orch._regime_state_tracker._recovery_polls = 1
    orch._last_regime_bar_str = None
    orch._last_evaluated_5m_bar_str = None

    mock_bridge = MagicMock()
    mock_bridge._connected = True
    mock_bridge.is_connected = True
    mock_bridge.ib.positions.return_value = []
    mock_bridge.get_net_liquidation.return_value = 100000.0

    degraded_regime = _make_result(mode=MODE_DATA_DEGRADED, texture=TEXTURE_VOLATILE, halt=True, allow_longs=False)
    healthy_regime = _make_result(mode=MODE_RISK_ON, texture=TEXTURE_TRENDING)

    # Boundary: completed bar is 09:55:00
    expected_completed = datetime(2026, 10, 12, 9, 55, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 12, 10, 0, 15, tzinfo=timezone.utc)  # elapsed: 15s < 90s
    t2 = datetime(2026, 10, 12, 10, 0, 45, tzinfo=timezone.utc)  # elapsed: 45s < 90s

    # Poll 1 at 10:00:15: Market data temporarily degraded
    with patch("strategies.regime_engine.evaluate_market_regime", return_value=degraded_regime), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=[]):
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)

    # Boundary was NOT consumed because of DEGRADED mode
    assert orch._last_regime_bar_str is None

    # Poll 2 at 10:00:45: Market data recovered within publication window (45s <= 90s)
    with patch("strategies.regime_engine.evaluate_market_regime", return_value=healthy_regime), \
         patch("strategies.session.is_past_opening_buffer", return_value=True), \
         patch("strategies.orchestrator.evaluate_symbols_parallel", return_value=[]):
        with patch("strategies.orchestrator.datetime") as mock_dt:
            mock_dt.now.return_value = t2
            orch.evaluate_and_execute(["AAPL"], bridge=mock_bridge, live_paper=False)

    # Boundary successfully retried and consumed!
    assert orch._last_regime_bar_str == expected_completed.isoformat()
    assert orch._regime_state_tracker.last_processed_bar_dt == expected_completed


def test_canonical_5m_grid_alignment_rejects_off_grid_bars():
    """
    Verify canonical 5-minute grid alignment contract:
    - 2.5-minute bars (e.g. 10:02:30) and non-5m-modulo timestamps (10:02:00, 10:03:00)
      are strictly rejected and cannot spoof dwell progression.
    - Only canonical grid-aligned 5m completed bars (10:00:00, 10:05:00, 10:10:00)
      can advance the dwell candidate counter.
    """
    tracker = RegimeStateTracker(texture_dwell_bars=3, bar_interval_seconds=300.0)

    # 1. Off-grid bootstrap bar (10:02:30) -> REJECTED
    t_off_bootstrap = datetime(2026, 10, 12, 10, 2, 30, tzinfo=timezone.utc)
    res0 = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_off_bootstrap, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt is None
    assert tracker.active_texture is None
    assert tracker.candidate_texture_count == 0

    # 2. Canonical bootstrap bar (10:00:00) -> ACCEPTED
    t0 = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
    tracker.update(_make_result(texture=TEXTURE_TRENDING), completed_bar_ts=t0, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t0
    assert tracker.active_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 1

    # 3. Off-grid 2.5m bar (10:02:30) -> REJECTED (watermark unchanged)
    t_2_5m = datetime(2026, 10, 12, 10, 2, 30, tzinfo=timezone.utc)
    res_2_5m = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_2_5m, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t0
    assert tracker.active_texture == TEXTURE_TRENDING
    assert tracker.candidate_texture_count == 1  # Did NOT advance!

    # 4. Off-grid non-multiple minute bar (10:03:00) -> REJECTED (second=0, but minute%5 != 0)
    t_3m = datetime(2026, 10, 12, 10, 3, 0, tzinfo=timezone.utc)
    res_3m = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_3m, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t0
    assert tracker.candidate_texture_count == 1  # Did NOT advance!

    # 5. Canonical next 5m bar (10:05:00) -> ACCEPTED (delta 300s >= 240s)
    t1 = datetime(2026, 10, 12, 10, 5, 0, tzinfo=timezone.utc)
    res1 = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t1, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t1
    assert tracker.active_texture == TEXTURE_TRENDING  # Dwell still active
    assert tracker.candidate_texture_count == 1

    # 6. Another off-grid attempt (10:07:30) -> REJECTED
    t_spoof = datetime(2026, 10, 12, 10, 7, 30, tzinfo=timezone.utc)
    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t_spoof, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t1
    assert tracker.candidate_texture_count == 1

    # 7. Canonical bar 2 (10:10:00) -> ACCEPTED
    t2 = datetime(2026, 10, 12, 10, 10, 0, tzinfo=timezone.utc)
    tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t2, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t2
    assert tracker.candidate_texture_count == 2

    # 8. Canonical bar 3 (10:15:00) -> ACCEPTED -> Confirms transition!
    t3 = datetime(2026, 10, 12, 10, 15, 0, tzinfo=timezone.utc)
    res3 = tracker.update(_make_result(texture=TEXTURE_VOLATILE), completed_bar_ts=t3, is_completed_bar_boundary=True)
    assert tracker.last_processed_bar_dt == t3
    assert tracker.active_texture == TEXTURE_VOLATILE
    assert tracker.candidate_texture_count == 0



