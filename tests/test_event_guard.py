"""
tests/test_event_guard.py -- Comprehensive unit tests for EventGuard and HaltManager integration.
"""
from datetime import datetime
import pytz
import pytest

from strategies.event_guard import evaluate_event, EventDecision
from strategies.halt_manager import (
    EntryHaltManager,
    TRANSIENT_HALT_EVENT_BLACKOUT,
)

ET = pytz.timezone("America/New_York")


class TestEventGuard:
    def test_normal_day_no_event(self):
        """Aug 25, 2026 (Tuesday) should be normal trading (no throttle, no halt)."""
        dt = ET.localize(datetime(2026, 8, 25, 10, 0, 0))
        decision = evaluate_event(dt)
        assert decision.throttle == 1.0
        assert decision.block_entries is False
        assert decision.atr_floor_mult == 1.0
        assert decision.tag is None

    def test_nvda_earnings_gap_day(self):
        """Aug 27, 2026 (Thursday) should trigger defensive 0.5x sizing and 2.0x ATR floor."""
        dt = ET.localize(datetime(2026, 8, 27, 9, 45, 0))
        decision = evaluate_event(dt)
        assert decision.throttle == 0.5
        assert decision.block_entries is False
        assert decision.atr_floor_mult == 2.0
        assert decision.tag == "NVDA_EARNINGS_GAP"

    def test_jackson_hole_warsh_keynote_blackout(self):
        """Aug 28, 2026 at 10:00 ET (Friday) falls inside 09:30-11:15 blackout window."""
        dt = ET.localize(datetime(2026, 8, 28, 10, 0, 0))
        decision = evaluate_event(dt)
        assert decision.throttle == 0.0
        assert decision.block_entries is True
        assert decision.atr_floor_mult == 2.0
        assert decision.tag == "JACKSON_HOLE_WARSH_BLACKOUT"

    def test_jackson_hole_after_blackout_defensive(self):
        """Aug 28, 2026 at 12:00 ET (Friday afternoon) outside blackout window."""
        dt = ET.localize(datetime(2026, 8, 28, 12, 0, 0))
        decision = evaluate_event(dt)
        assert decision.throttle == 0.5
        assert decision.block_entries is False
        assert decision.atr_floor_mult == 2.0
        assert decision.tag == "JACKSON_HOLE_WARSH"

    def test_timezone_naive_auto_localize(self):
        """Naive datetime should be localized without error."""
        dt_naive = datetime(2026, 8, 28, 10, 30, 0)
        decision = evaluate_event(dt_naive)
        assert decision.block_entries is True
        assert decision.tag == "JACKSON_HOLE_WARSH_BLACKOUT"

    def test_halt_manager_integration(self):
        """Event blackout should activate and clear transient halt cleanly."""
        hm = EntryHaltManager()
        assert hm.entries_allowed() is True

        # Activate blackout
        hm.apply_event_blackout(True, tag="JACKSON_HOLE_WARSH_BLACKOUT")
        assert hm.entries_allowed() is False
        assert TRANSIENT_HALT_EVENT_BLACKOUT in hm.active_halts()

        # Clear blackout
        hm.apply_event_blackout(False)
        assert hm.entries_allowed() is True
        assert TRANSIENT_HALT_EVENT_BLACKOUT not in hm.active_halts()
