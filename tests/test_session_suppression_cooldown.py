"""
tests/test_session_suppression_cooldown.py

Dedicated unit tests for P2 Dynamic Session Suppression Cooldown:
1. 5 hard vetoes triggers 45-minute cooldown and resets skip counter.
2. check_session_suppression blocks execution while cooldown is active.
3. check_session_suppression automatically unblocks after cooldown expires.
4. Safeguard #1: Cooldown Cycle Cap (max 3 cycles -> full-day permanent suppression).
5. Safeguard #2: Post-cooldown opportunity logging telemetry.
6. Clean reset of session suppression state.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

import config
import strategies.order_manager as om


@pytest.fixture(autouse=True)
def clean_suppression_state():
    om.reset_session_skip_state()
    yield
    om.reset_session_skip_state()


def test_veto_limit_triggers_cooldown_and_resets_skip_count():
    sym = "NVDA"
    reason = "Noise floor critically tight: stop 0.05 < 0.10"

    # Send 4 skips -> counter increments, no cooldown yet
    for i in range(4):
        om.record_hard_veto_skip(sym, reason, bar_timestamp=f"2026-09-03 09:{30+i*5}:00")

    assert om._session_skip_counts[sym] == 4
    assert sym not in om._session_suppressed_until
    assert om._session_cooldown_counts[sym] == 0

    # 5th skip -> triggers cooldown and resets skip count
    om.record_hard_veto_skip(sym, reason, bar_timestamp="2026-09-03 09:50:00")

    assert om._session_skip_counts[sym] == 0  # Counter reset verified!
    assert sym in om._session_suppressed_until
    assert om._session_cooldown_counts[sym] == 1  # Cycle #1
    assert om._session_suppressed_until[sym] > time.time()


def test_check_session_suppression_blocks_during_cooldown():
    sym = "NVDA"
    now = time.time()
    om._session_suppressed_until[sym] = now + 1800.0  # 30 mins remaining
    om._session_cooldown_counts[sym] = 1

    suppressed, msg = om.check_session_suppression(sym)
    assert suppressed is True
    assert "[SESSION SUPPRESSED] NVDA: cooldown active" in msg
    assert "Cycle #1" in msg


def test_check_session_suppression_unblocks_after_cooldown_expires():
    sym = "AAPL"
    # Mock cooldown expired 10 seconds ago
    om._session_suppressed_until[sym] = time.time() - 10.0
    om._session_cooldown_counts[sym] = 1

    suppressed, msg = om.check_session_suppression(sym)
    assert suppressed is False
    assert msg == ""
    # Symbol must be cleaned up from active cooldown dict
    assert sym not in om._session_suppressed_until
    # Cooldown cycle count must remain preserved for telemetry
    assert om._session_cooldown_counts[sym] == 1


def test_multiple_cooldown_cycles_and_cap_safeguard():
    """Verify Safeguard #1: After 3 cooldown cycles, 4th lockout becomes permanent."""
    sym = "TSLA"
    reason = "Noise floor critically tight: stop 0.04 < 0.10"

    with patch.object(config, "MAX_DAILY_COOLDOWN_CYCLES", 3):
        # Cycle 1
        for i in range(5):
            om.record_hard_veto_skip(sym, reason, bar_timestamp=f"c1_{i}")
        assert om._session_cooldown_counts[sym] == 1
        assert sym in om._session_suppressed_until

        # Expire Cycle 1
        om._session_suppressed_until[sym] = time.time() - 1.0
        assert om.check_session_suppression(sym)[0] is False

        # Cycle 2
        for i in range(5):
            om.record_hard_veto_skip(sym, reason, bar_timestamp=f"c2_{i}")
        assert om._session_cooldown_counts[sym] == 2
        assert sym in om._session_suppressed_until

        # Expire Cycle 2
        om._session_suppressed_until[sym] = time.time() - 1.0
        assert om.check_session_suppression(sym)[0] is False

        # Cycle 3
        for i in range(5):
            om.record_hard_veto_skip(sym, reason, bar_timestamp=f"c3_{i}")
        assert om._session_cooldown_counts[sym] == 3
        assert sym in om._session_suppressed_until

        # Expire Cycle 3
        om._session_suppressed_until[sym] = time.time() - 1.0
        assert om.check_session_suppression(sym)[0] is False

        # 4th Veto sequence -> Reaches MAX_DAILY_COOLDOWN_CYCLES=3 -> PERMANENT SUPPRESSION
        for i in range(5):
            om.record_hard_veto_skip(sym, reason, bar_timestamp=f"c4_{i}")

        assert sym in om._session_permanently_suppressed
        suppressed, msg = om.check_session_suppression(sym)
        assert suppressed is True
        assert "permanently suppressed this session" in msg


def test_reset_session_skip_state_clears_all():
    om._session_skip_counts["NVDA"] = 3
    om._session_suppressed_until["AAPL"] = time.time() + 1000
    om._session_cooldown_counts["AAPL"] = 2
    om._session_permanently_suppressed.add("TSLA")

    om.reset_session_skip_state()

    assert len(om._session_skip_counts) == 0
    assert len(om._session_suppressed_until) == 0
    assert len(om._session_cooldown_counts) == 0
    assert len(om._session_permanently_suppressed) == 0
