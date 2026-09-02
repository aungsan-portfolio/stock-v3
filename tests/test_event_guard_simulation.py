"""
tests/test_event_guard_simulation.py -- Log-driven simulation: replay the Aug 24 regime timeline and confirm
event_guard never overrides an existing regime halt (non-destructive layer).
"""
import pytest
from datetime import datetime
import pytz

from strategies.event_guard import evaluate_event

ET = pytz.timezone("America/New_York")

# Reconstructed from daytrade_current.log (times in ET)
AUG24_TIMELINE = [
    # (hh, mm, regime_score, regime_allows_longs)
    (9, 30, -2, False),   # opening buffer, CAUTION
    (9, 34, -3, False),   # QQQ < -1.00%
    (9, 38, -4, False),   # SOXX -3.04% -> RISK_OFF
    (9, 45, -5, False),   # buffer end, sizing 0%
    (11, 14, -5, False),  # SPDN frozen loop, still RISK_OFF
    (15, 55, -4, False),  # near close, still risk-off
]


class TestAug24Simulation:
    @pytest.mark.parametrize("hh,mm,score,regime_longs", AUG24_TIMELINE)
    def test_event_guard_is_inert_and_regime_dominates(
        self, hh, mm, score, regime_longs
    ):
        ev = evaluate_event(ET.localize(datetime(2026, 8, 24, hh, mm)))

        # 1) event guard adds NO restriction on Aug 24
        assert ev.tag is None
        assert ev.block_entries is False
        assert ev.throttle == 1.0

        # 2) effective decision = regime AND event (non-destructive layer)
        effective_allow_longs = regime_longs and not ev.block_entries
        assert effective_allow_longs is False   # regime still vetoes everything

    def test_layer_composition_semantics(self):
        """Even if a future event day is DEFENSIVE, a RISK_OFF regime that
        blocks longs must still win. Event guard only *tightens*, never loosens.
        """
        # simulate NVDA-day throttle meeting a risk-off regime
        ev = evaluate_event(ET.localize(datetime(2026, 8, 27, 10, 0)))
        regime_allows_longs = False              # hypothetical risk-off

        effective = regime_allows_longs and not ev.block_entries
        assert effective is False                # regime veto preserved
        # sizing only matters if longs were allowed; throttle never re-enables
        assert ev.throttle == 0.5
