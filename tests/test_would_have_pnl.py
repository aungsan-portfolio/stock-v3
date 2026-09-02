"""
tests/test_would_have_pnl.py -- Test counterfactual P&L audit for suppressed/vetoed candidates.
"""
import pytest
from strategies.audit.would_have_pnl import (
    SuppressedSignal,
    simulate_exit,
    run_audit,
    ExitReason,
)


class Bar:
    def __init__(self, h, l, c):
        self.high = h
        self.low = l
        self.close = c


# IBIT long: passed GAP QUALITY but regime-suppressed. On a -3.5% SOXX /
# risk-off day, simulate a fade -> stop hit (veto SAVED capital).
def test_ibit_long_fades_veto_saved():
    sig = SuppressedSignal(
        "IBIT", "REGIME_SUPPRESS", entry_price=62.00,
        atr=1.00, entry_idx=0
    )
    bars = [Bar(62.2, 61.9, 62.1), Bar(62.1, 60.4, 60.5)]  # dips to stop 60.5
    r = simulate_exit(sig, bars)
    assert r.exit_reason == ExitReason.STOP
    assert r.pnl_pct < 0
    assert r.verdict == "VETO_SAVED"


# SH inverse long on a risk-off day COULD have run -> target hit (veto MISSED).
def test_sh_inverse_runs_veto_missed():
    sig = SuppressedSignal(
        "SH", "TIGHT_STOP_VS_ATR", entry_price=40.00,
        atr=0.50, entry_idx=0
    )
    bars = [Bar(40.6, 39.9, 40.5), Bar(41.2, 40.4, 41.1)]  # hits target 41.0
    r = simulate_exit(sig, bars)
    assert r.exit_reason == ExitReason.TARGET
    assert r.verdict == "VETO_MISSED"


# No stop, no target -> EOD flatten path.
def test_eod_flatten_when_neither_hit():
    sig = SuppressedSignal(
        "VALE", "REGIME_LONG_BLOCK", entry_price=10.00,
        atr=0.30, entry_idx=0
    )
    bars = [Bar(10.1, 9.95, 10.02), Bar(10.05, 9.92, 9.99)]
    r = simulate_exit(sig, bars)
    assert r.exit_reason == ExitReason.EOD


def test_aggregate_net_verdict():
    signals = [
        SuppressedSignal("IBIT", "REGIME", 62.00, 1.00, 0),
        SuppressedSignal("SH",   "TIGHT",  40.00, 0.50, 0),
    ]
    bars = {
        "IBIT": [Bar(62.2, 60.4, 60.5)],      # stop -> saved
        "SH":   [Bar(41.2, 39.9, 41.1)],      # target -> missed
    }
    summary = run_audit(signals, bars)
    assert summary["n_saved"] == 1
    assert summary["n_missed"] == 1
    assert summary["net_verdict"] in ("REGIME_CORRECT", "TOO_CONSERVATIVE")
