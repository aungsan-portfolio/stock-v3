import pytest
from unittest.mock import MagicMock
from position_rotator import PositionRotator, _RATCHET_RULES

def make_rotator(current_stop: float = 440.0) -> PositionRotator:
    bridge = MagicMock()
    bridge.get_current_stop.return_value = current_stop
    bridge.replace_stop.return_value = 'mock_order_id'
    rotator = PositionRotator(bridge=bridge)
    return rotator

MSFT_ENTRY = 445.84

def msft_pos(pnl_pct: float) -> dict:
    current = MSFT_ENTRY * (1.0 + pnl_pct)
    return {
        'symbol': 'MSFT',
        'avg_entry_price': str(MSFT_ENTRY),
        'current_price': str(round(current, 2)),
    }

class TestRatchetRules:
    def test_below_threshold_skips(self):
        r = make_rotator()
        results = r._ratchet_stops([msft_pos(0.03)], dry_run=True)
        assert results[0].skipped
        assert results[0].rule_applied == 'none'

    def test_breakeven_plus_at_5pct(self):
        r = make_rotator(current_stop=440.0)
        results = r._ratchet_stops([msft_pos(0.05)], dry_run=True)
        assert results[0].rule_applied == 'breakeven+'
        assert results[0].new_stop == round(MSFT_ENTRY * 1.005, 2)
        assert not results[0].skipped

    def test_lock_in_4pct_at_8pct(self):
        r = make_rotator(current_stop=440.0)
        results = r._ratchet_stops([msft_pos(0.0825)], dry_run=True)
        assert results[0].rule_applied == 'lock-in-4pct'
        expected = round(MSFT_ENTRY * 1.04, 2)
        assert results[0].new_stop == expected

    def test_lock_in_10pct_at_15pct(self):
        r = make_rotator(current_stop=463.67)
        results = r._ratchet_stops([msft_pos(0.15)], dry_run=True)
        assert results[0].rule_applied == 'lock-in-10pct'

class TestRatchetGuards:
    def test_one_direction_only(self):
        r = make_rotator(current_stop=470.0)
        results = r._ratchet_stops([msft_pos(0.0825)], dry_run=False)
        assert results[0].skipped
        assert 'already protected' in results[0].reason
        r.bridge.replace_stop.assert_not_called()

    def test_bridge_failure_does_not_block(self):
        r = make_rotator(current_stop=440.0)
        r.bridge.replace_stop.side_effect = ConnectionError('timeout')
        results = r._ratchet_stops([msft_pos(0.0825)], dry_run=False)
        assert results[0].skipped
        assert 'bridge_error' in results[0].reason

    def test_no_resting_stop_still_executes(self):
        r = make_rotator()
        r.bridge.get_current_stop.return_value = None
        results = r._ratchet_stops([msft_pos(0.0825)], dry_run=False)
        assert not results[0].skipped
        r.bridge.replace_stop.assert_called_once()

    def test_dry_run_no_order_sent(self):
        r = make_rotator(current_stop=440.0)
        r._ratchet_stops([msft_pos(0.0825)], dry_run=True)
        r.bridge.replace_stop.assert_not_called()
