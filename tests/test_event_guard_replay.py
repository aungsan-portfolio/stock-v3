"""
tests/test_event_guard_replay.py -- Replay backtest: verify event_guard decisions against the Aug 24 2026
session and the upcoming NVDA / Jackson Hole event windows.
"""
import pytest
from datetime import datetime
import pytz

from strategies.event_guard import evaluate_event

ET = pytz.timezone("America/New_York")


def et(y, m, d, hh, mm):
    return ET.localize(datetime(y, m, d, hh, mm))


# ---------------------------------------------------------------
# GROUP A — Aug 24 baseline (the actual logged session)
# Event guard must be INERT: no throttle, no block, no ATR change.
# All protection that day came from regime_engine, NOT event_guard.
# ---------------------------------------------------------------
class TestAug24Baseline:
    @pytest.mark.parametrize("hh,mm", [
        (9, 30),   # 20:30 ICT market open
        (9, 38),   # RISK_OFF plunge (SOXX -3.04%)
        (9, 55),   # API drop / reconnect window
        (11, 14),  # SPDN frozen-feed loop
        (15, 55),  # near close
    ])
    def test_no_event_interference_aug24(self, hh, mm):
        d = evaluate_event(et(2026, 8, 24, hh, mm))
        assert d.tag is None
        assert d.block_entries is False
        assert d.throttle == 1.0
        assert d.atr_floor_mult == 1.0


# ---------------------------------------------------------------
# GROUP B — Thu Aug 27 NVDA gap day: DEFENSIVE all session
# ---------------------------------------------------------------
class TestNvdaGapDay:
    @pytest.mark.parametrize("hh,mm", [(9, 30), (11, 0), (14, 30), (15, 59)])
    def test_defensive_all_day(self, hh, mm):
        d = evaluate_event(et(2026, 8, 27, hh, mm))
        assert d.tag == "NVDA_EARNINGS_GAP"
        assert d.block_entries is False        # throttle, not block
        assert d.throttle == 0.5
        assert d.atr_floor_mult == 2.0


# ---------------------------------------------------------------
# GROUP C — Fri Aug 28 Jackson Hole: blackout inside window,
# defensive outside it.
# ---------------------------------------------------------------
class TestJacksonHole:
    @pytest.mark.parametrize("hh,mm", [(9, 30), (10, 0), (11, 15)])
    def test_hard_block_in_window(self, hh, mm):
        d = evaluate_event(et(2026, 8, 28, hh, mm))
        assert d.tag == "JACKSON_HOLE_WARSH_BLACKOUT"
        assert d.block_entries is True
        assert d.throttle == 0.0
        assert d.atr_floor_mult == 2.0

    @pytest.mark.parametrize("hh,mm", [(9, 29), (11, 16), (14, 0)])
    def test_defensive_outside_window(self, hh, mm):
        d = evaluate_event(et(2026, 8, 28, hh, mm))
        assert d.tag == "JACKSON_HOLE_WARSH"
        assert d.block_entries is False
        assert d.throttle == 0.5

    def test_window_boundaries_exact(self):
        # 09:30:00 and 11:15:00 are INCLUSIVE edges
        assert evaluate_event(et(2026, 8, 28, 9, 30)).block_entries is True
        assert evaluate_event(et(2026, 8, 28, 11, 15)).block_entries is True
        # one minute either side -> defensive, not blocked
        assert evaluate_event(et(2026, 8, 28, 9, 29)).block_entries is False
        assert evaluate_event(et(2026, 8, 28, 11, 16)).block_entries is False
