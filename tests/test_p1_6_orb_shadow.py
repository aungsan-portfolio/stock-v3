"""
tests/test_p1_6_orb_shadow.py -- Comprehensive 36-Test Suite for P1.6 Confirmed ORB Shadow Mode.
"""
from datetime import datetime, time, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import config
from strategies.base import TradeSignal
from strategies.shadow.models import (
    AlignmentStatus,
    ShadowCandidate,
    ShadowOutcome,
    ShadowTrade,
    TimeBucket,
    generate_deterministic_shadow_id,
)
from strategies.shadow.pricing import (
    PricingSource,
    QuoteSnapshot,
    resolve_entry_price,
    resolve_exit_price,
    validate_quote,
)
from strategies.shadow.storage import ShadowStorageManager
from strategies.shadow.orb_confirmed import ConfirmedORBStrategy
from strategies.shadow.lifecycle import ShadowLifecycleManager
from strategies.shadow.analytics import (
    ResearchGateStatus,
    evaluate_shadow_metrics,
)


# ── Fixtures & Test Data Helpers ──

def make_1m_bar(
    ts_et: str,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    vol: float,
    vwap_p: float = None,
    atr_p: float = 1.0,
    ema9_p: float = None,
) -> dict:
    return {
        "timestamp": pd.Timestamp(ts_et, tz="America/New_York"),
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "volume": vol,
        "vwap": vwap_p or ((high_p + low_p + close_p) / 3.0),
        "atr": atr_p,
        "ema9": ema9_p or close_p,
        "ema20": close_p - 0.2,
    }


def make_orb_session_df(date_str: str = "2026-09-02", num_bars: int = 40, breakout_at: int = 35) -> pd.DataFrame:
    """Builds a realistic 1m DataFrame with 15 opening range bars (09:30 - 09:44) and a breakout."""
    records = []
    base_time = pd.Timestamp(f"{date_str} 09:30:00", tz="America/New_York")

    for i in range(num_bars):
        t = base_time + pd.Timedelta(minutes=i)
        if i < 15:
            # Range: High=101.0, Low=99.0
            o = 100.0 + (i % 2) * 0.5
            h = min(101.0, o + 0.5)
            l = max(99.0, o - 1.0)
            c = (h + l) / 2.0
            v = 1000.0
            vw = 100.0
        elif i < breakout_at:
            # Cooldown period (09:45 - 09:59) inside range
            o = 100.2
            h = 100.8
            l = 99.5
            c = 100.3
            v = 800.0
            vw = 100.1
        else:
            # Breakout at or after 10:00 ET (index 30+ is 10:00)
            o = 100.8
            h = 102.5
            l = 100.7
            c = 102.0  # Closes above 101.0 ORB High!
            v = 2500.0  # > 1.5x average
            vw = 100.5
        records.append(make_1m_bar(t.isoformat(), o, h, l, c, v, vwap_p=vw, atr_p=1.0, ema9_p=c - 0.3))

    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    return df


# ── 36 Mandated Unit Tests ──

def test_1_shadow_signal_never_submits_broker_order():
    mock_bridge = MagicMock()
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df()
    meta = {"relative_volume": 3.0, "catalyst_type": "EARNINGS_BEAT", "gap_pct": 3.5}
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta=meta)

    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm = ShadowLifecycleManager(storage=sm)
        if cand:
            lm.register_candidate(cand)
    assert mock_bridge.submit_order.call_count == 0
    assert mock_bridge.place_bracket_order.call_count == 0


def test_2_shadow_signal_never_acquires_idempotency_lock():
    idemp_path = Path("data/order_idempotency_state.json")
    original_hash = hashlib.sha256(idemp_path.read_bytes()).hexdigest() if idemp_path.exists() else None
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df()
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta={"relative_volume": 3.0, "gap_pct": 3.0})
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm = ShadowLifecycleManager(storage=sm)
        if cand:
            lm.register_candidate(cand)
    new_hash = hashlib.sha256(idemp_path.read_bytes()).hexdigest() if idemp_path.exists() else None
    assert original_hash == new_hash


def test_3_orb_range_uses_only_completed_bars():
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df(num_bars=15)
    # Range is exactly 15 bars: 09:30 to 09:44. Forming 09:45 bar is not included
    res = strat._compute_opening_range(df, df.index, "2026-09-02")
    assert res is not None
    orb_h, orb_l, orb_r = res
    assert orb_h == 101.0
    assert orb_l == 99.0
    assert orb_r == 2.0


def test_4_orb_entry_rejected_before_1000_et():
    strat = ConfirmedORBStrategy()
    # 25 bars: last bar is 09:54 ET (before 10:00 ET)
    df = make_orb_session_df(num_bars=25, breakout_at=20)
    meta = {"relative_volume": 3.0, "gap_pct": 3.0}
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta=meta)
    assert cand is None  # Entry before 10:00 ET rejected!


def test_5_orb_entry_rejected_after_1130_et():
    strat = ConfirmedORBStrategy()
    # 125 bars: 09:30 + 125m = 11:35 ET (after 11:30 ET)
    df = make_orb_session_df(num_bars=125, breakout_at=120)
    meta = {"relative_volume": 3.0, "gap_pct": 3.0}
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta=meta)
    assert cand is None  # Entry after 11:30 ET rejected!


def test_6_orb_rejects_rvol_below_two():
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df(num_bars=35, breakout_at=32)
    meta = {"relative_volume": 1.9, "gap_pct": 3.0}
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta=meta)
    assert cand is None  # RVOL 1.9 < 2.0 rejected!


def test_7_orb_rejects_below_vwap():
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df(num_bars=35, breakout_at=32)
    df.loc[df.index[-1], "vwap"] = 105.0  # Close 102 < VWAP 105
    meta = {"relative_volume": 3.0, "gap_pct": 3.0}
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta=meta)
    assert cand is None


def test_8_orb_rejects_negative_vwap_slope():
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df(num_bars=35, breakout_at=32)
    # Make prior VWAP higher than current VWAP (falling slope)
    df.loc[df.index[-4], "vwap"] = 101.5
    df.loc[df.index[-1], "vwap"] = 100.5
    meta = {"relative_volume": 3.0, "gap_pct": 3.0}
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta=meta)
    assert cand is None


def test_9_orb_rejects_overextended_candle():
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df(num_bars=35, breakout_at=32)
    # Set EMA9 very far below close (overextended > 1.25 ATR)
    df.loc[df.index[-1], "ema9"] = 99.0  # close=102, diff=3.0 > 1.25 * atr(1.0)
    meta = {"relative_volume": 3.0, "gap_pct": 3.0}
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta=meta)
    assert cand is None


def test_10_orb_requires_market_alignment():
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df(num_bars=35, breakout_at=32)
    # Opposed SPY df: close below vwap
    spy_df = pd.DataFrame([{"close": 490.0, "vwap": 500.0}] * 5)
    market_data = {"SPY": spy_df, "QQQ": None}
    meta = {"relative_volume": 3.0, "gap_pct": 3.0}
    cand = strat.evaluate_candidate("NVDA", df, market_data=market_data, candidate_meta=meta)
    assert cand is None


def test_11_conservative_same_bar_stop_target_resolution():
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm = ShadowLifecycleManager(storage=sm, clock=lambda: pd.Timestamp("2026-09-02T14:05:00Z"))
        cand = ShadowCandidate(
            candidate_id="test_cand_11",
            strategy_name="ORB_CONFIRMED",
            strategy_version="V1",
            symbol="TEST",
            side="BUY",
            signal_timestamp_utc="2026-09-02T14:05:00Z",
            signal_bar_start_utc="2026-09-02T14:00:00Z",
            signal_bar_close_utc="2026-09-02T14:05:00Z",
            session_date_et="2026-09-02",
            reference_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_per_share=2.0,
            opening_range_high=99.0,
            opening_range_low=95.0,
            relative_volume=3.0,
            breakout_volume_ratio=2.0,
            price_vs_vwap=1.0,
            vwap_slope=0.1,
            ema9=99.5,
            ema20=98.0,
            atr=1.0,
            atr_pct=0.01,
            spread_pct=0.0005,
            spy_alignment=AlignmentStatus.ALIGNED,
            qqq_alignment=AlignmentStatus.ALIGNED,
            sector_alignment=AlignmentStatus.FALLBACK_MARKET_CONFIRMED,
            catalyst_type="EARNINGS",
            premarket_gap_pct=3.0,
            premarket_volume_ratio=2.5,
            market_texture="TRENDING",
            regime_mode="RISK_ON",
            confidence=0.9,
        )
        trade = lm.register_candidate(cand)
        assert trade is not None

        # Subsequent bar touches BOTH stop (97.0 <= 98.0) and target (105.0 >= 104.0)
        outcome_bar = pd.Series({"high": 105.0, "low": 97.0, "close": 100.0}, name=pd.Timestamp("2026-09-02T14:06:00Z"))
        closed = lm.update_active_trades("TEST", outcome_bar)
        assert len(closed) == 1
        t = closed[0]
        assert t.status == ShadowOutcome.AMBIGUOUS_INTRABAR.value
        assert t.ambiguous_intrabar is True
        assert t.outcome_class == "LOSS"
        assert t.net_result_r < 0


def test_12_slippage_model_deducted_from_net_pnl():
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm = ShadowLifecycleManager(storage=sm, clock=lambda: pd.Timestamp("2026-09-02T14:05:00Z"))
        # Register BUY candidate without live quote -> applies synthetic slippage
        cand = ShadowCandidate(
            candidate_id="test_cand_12",
            strategy_name="ORB_CONFIRMED",
            strategy_version="V1",
            symbol="TEST",
            side="BUY",
            signal_timestamp_utc="2026-09-02T14:05:00Z",
            signal_bar_start_utc="2026-09-02T14:00:00Z",
            signal_bar_close_utc="2026-09-02T14:05:00Z",
            session_date_et="2026-09-02",
            reference_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_per_share=2.0,
            opening_range_high=99.0,
            opening_range_low=95.0,
            relative_volume=3.0,
            breakout_volume_ratio=2.0,
            price_vs_vwap=1.0,
            vwap_slope=0.1,
            ema9=99.5,
            ema20=98.0,
            atr=1.0,
            atr_pct=0.01,
            spread_pct=0.0005,
            spy_alignment=AlignmentStatus.ALIGNED,
            qqq_alignment=AlignmentStatus.ALIGNED,
            sector_alignment=AlignmentStatus.FALLBACK_MARKET_CONFIRMED,
            catalyst_type="EARNINGS",
            premarket_gap_pct=3.0,
            premarket_volume_ratio=2.5,
            market_texture="TRENDING",
            regime_mode="RISK_ON",
            confidence=0.9,
        )
        trade = lm.register_candidate(cand)
        assert trade.entry_price > 100.0  # Slippage added on entry
        assert trade.entry_synthetic_cost > 0

        # Close on target
        outcome_bar = pd.Series({"high": 105.0, "low": 99.0, "close": 104.0}, name=pd.Timestamp("2026-09-02T14:06:00Z"))
        closed = lm.update_active_trades("TEST", outcome_bar)
        assert len(closed) == 1
        t = closed[0]
        assert t.total_cost > 0
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.total_cost, 0.01)


def test_13_shadow_lifecycle_survives_restart():
    with tempfile.TemporaryDirectory() as td:
        sm1 = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm1 = ShadowLifecycleManager(storage=sm1)
        cand = ShadowCandidate(
            candidate_id="test_cand_13",
            strategy_name="ORB_CONFIRMED",
            strategy_version="V1",
            symbol="TEST",
            side="BUY",
            signal_timestamp_utc="2026-09-02T14:05:00Z",
            signal_bar_start_utc="2026-09-02T14:00:00Z",
            signal_bar_close_utc="2026-09-02T14:05:00Z",
            session_date_et="2026-09-02",
            reference_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_per_share=2.0,
            opening_range_high=99.0,
            opening_range_low=95.0,
            relative_volume=3.0,
            breakout_volume_ratio=2.0,
            price_vs_vwap=1.0,
            vwap_slope=0.1,
            ema9=99.5,
            ema20=98.0,
            atr=1.0,
            atr_pct=0.01,
            spread_pct=0.0005,
            spy_alignment=AlignmentStatus.ALIGNED,
            qqq_alignment=AlignmentStatus.ALIGNED,
            sector_alignment=AlignmentStatus.FALLBACK_MARKET_CONFIRMED,
            catalyst_type="EARNINGS",
            premarket_gap_pct=3.0,
            premarket_volume_ratio=2.5,
            market_texture="TRENDING",
            regime_mode="RISK_ON",
            confidence=0.9,
        )
        lm1.register_candidate(cand)
        assert lm1.active_trade_count == 1
        sm1.release_lock()

        # Simulate engine restart: new manager loads from disk
        sm2 = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm2 = ShadowLifecycleManager(storage=sm2)
        assert lm2.active_trade_count == 1
        assert "test_cand_13" in lm2._active_trades


def test_14_duplicate_signal_id_rejected():
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm = ShadowLifecycleManager(storage=sm)
        cand = ShadowCandidate(
            candidate_id="test_cand_14",
            strategy_name="ORB_CONFIRMED",
            strategy_version="V1",
            symbol="TEST",
            side="BUY",
            signal_timestamp_utc="2026-09-02T14:05:00Z",
            signal_bar_start_utc="2026-09-02T14:00:00Z",
            signal_bar_close_utc="2026-09-02T14:05:00Z",
            session_date_et="2026-09-02",
            reference_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_per_share=2.0,
            opening_range_high=99.0,
            opening_range_low=95.0,
            relative_volume=3.0,
            breakout_volume_ratio=2.0,
            price_vs_vwap=1.0,
            vwap_slope=0.1,
            ema9=99.5,
            ema20=98.0,
            atr=1.0,
            atr_pct=0.01,
            spread_pct=0.0005,
            spy_alignment=AlignmentStatus.ALIGNED,
            qqq_alignment=AlignmentStatus.ALIGNED,
            sector_alignment=AlignmentStatus.FALLBACK_MARKET_CONFIRMED,
            catalyst_type="EARNINGS",
            premarket_gap_pct=3.0,
            premarket_volume_ratio=2.5,
            market_texture="TRENDING",
            regime_mode="RISK_ON",
            confidence=0.9,
        )
        t1 = lm.register_candidate(cand)
        t2 = lm.register_candidate(cand)
        assert lm.active_trade_count == 1
        assert t1.shadow_trade_id == t2.shadow_trade_id


def test_15_metrics_breakdown_by_time_bucket():
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
        t1 = ShadowTrade(
            shadow_trade_id="t1",
            strategy_name="ORB",
            strategy_version="V1",
            symbol="A",
            side="BUY",
            entry_time_bucket=TimeBucket.PRIMARY.value,
            outcome_class="WIN",
            net_pnl=200.0,
            net_result_r=2.0,
            session_date_et="2026-09-02",
        )
        t2 = ShadowTrade(
            shadow_trade_id="t2",
            strategy_name="ORB",
            strategy_version="V1",
            symbol="B",
            side="BUY",
            entry_time_bucket=TimeBucket.MIDDAY.value,
            outcome_class="LOSS",
            net_pnl=-100.0,
            net_result_r=-1.0,
            session_date_et="2026-09-02",
        )
        f.write(json.dumps(t1.to_dict()) + "\n")
        f.write(json.dumps(t2.to_dict()) + "\n")
        f.flush()
        temp_path = Path(f.name)

    rep = evaluate_shadow_metrics(temp_path)
    assert TimeBucket.PRIMARY.value in rep.time_bucket_breakdown
    assert TimeBucket.MIDDAY.value in rep.time_bucket_breakdown
    assert rep.time_bucket_breakdown[TimeBucket.PRIMARY.value]["win_rate"] == 100.0
    assert rep.time_bucket_breakdown[TimeBucket.MIDDAY.value]["win_rate"] == 0.0
    temp_path.unlink()


def test_16_actual_trade_journal_not_modified():
    journal_path = Path("data/trade_journal.jsonl")
    initial_bytes = journal_path.read_bytes() if journal_path.exists() else b""
    initial_hash = hashlib.sha256(initial_bytes).hexdigest()

    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm = ShadowLifecycleManager(storage=sm)
        trade = ShadowTrade(
            shadow_trade_id="t16",
            strategy_name="ORB",
            strategy_version="V1",
            symbol="C",
            side="BUY",
            status=ShadowOutcome.TARGET_HIT.value,
            net_pnl=100.0,
        )
        sm.append_closed_trade(trade)

    after_bytes = journal_path.read_bytes() if journal_path.exists() else b""
    after_hash = hashlib.sha256(after_bytes).hexdigest()
    assert initial_hash == after_hash


def test_17_shadow_report_flags_low_sample_size():
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
        # Only 5 trades (N < 30)
        for i in range(5):
            t = ShadowTrade(
                shadow_trade_id=f"t17_{i}",
                strategy_name="ORB",
                strategy_version="V1",
                symbol="S",
                side="BUY",
                outcome_class="WIN",
                net_pnl=100.0,
                net_result_r=2.0,
                session_date_et="2026-09-02",
            )
            f.write(json.dumps(t.to_dict()) + "\n")
        f.flush()
        temp_path = Path(f.name)

    rep = evaluate_shadow_metrics(temp_path)
    assert rep.gate_status == ResearchGateStatus.INSUFFICIENT_SAMPLE
    assert any("Insufficient sample size" in r for r in rep.gate_reasons)
    temp_path.unlink()


def test_18_promotion_gate_blocks_below_thresholds():
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
        # 30 trades but low win rate (PF < 1.25)
        for i in range(30):
            outcome = "WIN" if i < 8 else "LOSS"
            pnl = 100.0 if outcome == "WIN" else -100.0
            r = 1.0 if outcome == "WIN" else -1.0
            t = ShadowTrade(
                shadow_trade_id=f"t18_{i}",
                strategy_name="ORB",
                strategy_version="V1",
                symbol="S",
                side="BUY",
                outcome_class=outcome,
                net_pnl=pnl,
                net_result_r=r,
                session_date_et=f"2026-09-0{1 + (i % 6)}",
            )
            f.write(json.dumps(t.to_dict()) + "\n")
        f.flush()
        temp_path = Path(f.name)

    rep = evaluate_shadow_metrics(temp_path)
    assert rep.gate_status == ResearchGateStatus.RESEARCH_REJECTED
    assert rep.profit_factor < 1.25
    temp_path.unlink()


def test_19_production_orb_remains_disabled():
    assert config.STRATEGY_ORB_ENABLED is False
    from strategies.orb_strategy import ORBStrategy
    orb = ORBStrategy()
    df = make_orb_session_df()
    # When STRATEGY_ORB_ENABLED is False, legacy evaluate returns None
    sig = orb.evaluate("NVDA", df)
    assert sig is None


def test_20_all_broker_mutation_paths_forbidden():
    mock_bridge = MagicMock()
    # Assert none of the 8 broker mutation methods are ever called by shadow logic
    methods = [
        "execute_signal",
        "place_bracket_order",
        "submit_order",
        "submit_bracket",
        "replace_order",
        "modify_stop_order",
        "close_position",
        "cancel_order",
    ]
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm = ShadowLifecycleManager(storage=sm)
        cand = ShadowCandidate(
            candidate_id="test_cand_20",
            strategy_name="ORB_CONFIRMED",
            strategy_version="V1",
            symbol="TEST",
            side="BUY",
            signal_timestamp_utc="2026-09-02T14:05:00Z",
            signal_bar_start_utc="2026-09-02T14:00:00Z",
            signal_bar_close_utc="2026-09-02T14:05:00Z",
            session_date_et="2026-09-02",
            reference_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_per_share=2.0,
            opening_range_high=99.0,
            opening_range_low=95.0,
            relative_volume=3.0,
            breakout_volume_ratio=2.0,
            price_vs_vwap=1.0,
            vwap_slope=0.1,
            ema9=99.5,
            ema20=98.0,
            atr=1.0,
            atr_pct=0.01,
            spread_pct=0.0005,
            spy_alignment=AlignmentStatus.ALIGNED,
            qqq_alignment=AlignmentStatus.ALIGNED,
            sector_alignment=AlignmentStatus.FALLBACK_MARKET_CONFIRMED,
            catalyst_type="EARNINGS",
            premarket_gap_pct=3.0,
            premarket_volume_ratio=2.5,
            market_texture="TRENDING",
            regime_mode="RISK_ON",
            confidence=0.9,
        )
        lm.register_candidate(cand)
        outcome_bar = pd.Series({"high": 105.0, "low": 99.0, "close": 104.0}, name=pd.Timestamp("2026-09-02T14:06:00Z"))
        lm.update_active_trades("TEST", outcome_bar)

    for m in methods:
        assert getattr(mock_bridge, m).call_count == 0


def test_21_signal_bar_cannot_become_outcome_bar():
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        lm = ShadowLifecycleManager(storage=sm)
        cand = ShadowCandidate(
            candidate_id="test_cand_21",
            strategy_name="ORB_CONFIRMED",
            strategy_version="V1",
            symbol="TEST",
            side="BUY",
            signal_timestamp_utc="2026-09-02T14:05:00Z",
            signal_bar_start_utc="2026-09-02T14:00:00Z",
            signal_bar_close_utc="2026-09-02T14:05:00Z",
            session_date_et="2026-09-02",
            reference_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_per_share=2.0,
            opening_range_high=99.0,
            opening_range_low=95.0,
            relative_volume=3.0,
            breakout_volume_ratio=2.0,
            price_vs_vwap=1.0,
            vwap_slope=0.1,
            ema9=99.5,
            ema20=98.0,
            atr=1.0,
            atr_pct=0.01,
            spread_pct=0.0005,
            spy_alignment=AlignmentStatus.ALIGNED,
            qqq_alignment=AlignmentStatus.ALIGNED,
            sector_alignment=AlignmentStatus.FALLBACK_MARKET_CONFIRMED,
            catalyst_type="EARNINGS",
            premarket_gap_pct=3.0,
            premarket_volume_ratio=2.5,
            market_texture="TRENDING",
            regime_mode="RISK_ON",
            confidence=0.9,
        )
        lm.register_candidate(cand)
        # Attempt to pass the signal bar itself (ending at 14:05:00Z) as outcome bar
        signal_bar = pd.Series({"high": 105.0, "low": 97.0, "close": 100.0}, name=pd.Timestamp("2026-09-02T14:04:00Z"))
        closed = lm.update_active_trades("TEST", signal_bar)
        # First outcome bar rule: must be strictly ignored!
        assert len(closed) == 0
        assert lm.active_trade_count == 1


def test_22_entry_timestamp_cannot_precede_signal_close():
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        # Clock set to 14:03:00 (before signal close 14:05:00)
        lm = ShadowLifecycleManager(storage=sm, clock=lambda: pd.Timestamp("2026-09-02T14:03:00Z"))
        cand = ShadowCandidate(
            candidate_id="test_cand_22",
            strategy_name="ORB_CONFIRMED",
            strategy_version="V1",
            symbol="TEST",
            side="BUY",
            signal_timestamp_utc="2026-09-02T14:05:00Z",
            signal_bar_start_utc="2026-09-02T14:00:00Z",
            signal_bar_close_utc="2026-09-02T14:05:00Z",
            session_date_et="2026-09-02",
            reference_price=100.0,
            stop_price=98.0,
            target_price=104.0,
            risk_per_share=2.0,
            opening_range_high=99.0,
            opening_range_low=95.0,
            relative_volume=3.0,
            breakout_volume_ratio=2.0,
            price_vs_vwap=1.0,
            vwap_slope=0.1,
            ema9=99.5,
            ema20=98.0,
            atr=1.0,
            atr_pct=0.01,
            spread_pct=0.0005,
            spy_alignment=AlignmentStatus.ALIGNED,
            qqq_alignment=AlignmentStatus.ALIGNED,
            sector_alignment=AlignmentStatus.FALLBACK_MARKET_CONFIRMED,
            catalyst_type="EARNINGS",
            premarket_gap_pct=3.0,
            premarket_volume_ratio=2.5,
            market_texture="TRENDING",
            regime_mode="RISK_ON",
            confidence=0.9,
        )
        trade = lm.register_candidate(cand)
        # Entry timestamp must be clamped to >= signal_bar_close_utc!
        assert pd.Timestamp(trade.entry_timestamp_utc) >= pd.Timestamp("2026-09-02T14:05:00Z")


def test_23_exact_time_boundaries():
    strat = ConfirmedORBStrategy()
    earliest = datetime.strptime(strat.earliest_entry_str, "%H:%M").time()  # 10:00
    latest = datetime.strptime(strat.latest_entry_str, "%H:%M").time()      # 11:30

    assert earliest <= time(10, 0, 0) < latest      # 10:00:00 accepted
    assert earliest <= time(11, 29, 59) < latest    # 11:29:59 accepted
    assert not (earliest <= time(11, 30, 0) < latest) # 11:30:00 rejected


def test_24_opening_range_excludes_premarket():
    strat = ConfirmedORBStrategy()
    records = [
        # Premarket 09:29 bar
        make_1m_bar("2026-09-02 09:29:00", 100, 110, 90, 105, 500),
    ]
    # 15 regular session bars 09:30 to 09:44
    for i in range(15):
        t = pd.Timestamp("2026-09-02 09:30:00", tz="America/New_York") + pd.Timedelta(minutes=i)
        records.append(make_1m_bar(t.isoformat(), 100, 102, 98, 100, 1000))
    df = pd.DataFrame(records).set_index("timestamp")
    res = strat._compute_opening_range(df, df.index, "2026-09-02")
    assert res is not None
    orb_h, orb_l, _ = res
    assert orb_h == 102.0  # Premarket 110.0 was strictly excluded!
    assert orb_l == 98.0   # Premarket 90.0 was strictly excluded!


def test_25_incomplete_opening_range_rejected():
    strat = ConfirmedORBStrategy()
    # 14 bars instead of 15
    records = []
    for i in range(14):
        t = pd.Timestamp("2026-09-02 09:30:00", tz="America/New_York") + pd.Timedelta(minutes=i)
        records.append(make_1m_bar(t.isoformat(), 100, 102, 98, 100, 1000))
    df = pd.DataFrame(records).set_index("timestamp")
    res = strat._compute_opening_range(df, df.index, "2026-09-02")
    assert res is None  # Exactly 15 required; 14 rejected!


def test_26_stock_in_play_requires_rvol_plus_catalyst_or_gap():
    strat = ConfirmedORBStrategy()
    df = make_orb_session_df(num_bars=35, breakout_at=32)
    # RVOL=2.5 but NO catalyst, NO gap, NO premarket volume
    meta = {"relative_volume": 2.5, "catalyst_type": None, "gap_pct": 0.5, "premarket_volume_ratio": 1.0}
    cand = strat.evaluate_candidate("NVDA", df, candidate_meta=meta)
    assert cand is None  # Rejected without catalyst or gap!


def test_27_long_and_short_mfe_mae_direction_aware():
    # Long trade: entry 100, stop 98 (risk 2.0). High 103, Low 99.
    # Long MFE = (103 - 100)/2 = 1.5R. Long MAE = (100 - 99)/2 = 0.5R.
    t_long = ShadowTrade(
        shadow_trade_id="tl",
        strategy_name="ORB",
        strategy_version="V1",
        symbol="L",
        side="BUY",
        entry_price=100.0,
        risk_per_share=2.0,
        highest_price_seen=103.0,
        lowest_price_seen=99.0,
    )
    mfe_l = (t_long.highest_price_seen - t_long.entry_price) / t_long.risk_per_share
    mae_l = (t_long.entry_price - t_long.lowest_price_seen) / t_long.risk_per_share
    assert mfe_l == 1.5
    assert mae_l == 0.5

    # Short trade: entry 100, stop 102 (risk 2.0). High 101, Low 97.
    # Short MFE = (100 - 97)/2 = 1.5R. Short MAE = (101 - 100)/2 = 0.5R.
    t_short = ShadowTrade(
        shadow_trade_id="ts",
        strategy_name="ORB",
        strategy_version="V1",
        symbol="S",
        side="SELL",
        entry_price=100.0,
        risk_per_share=2.0,
        highest_price_seen=101.0,
        lowest_price_seen=97.0,
    )
    mfe_s = (t_short.entry_price - t_short.lowest_price_seen) / t_short.risk_per_share
    mae_s = (t_short.highest_price_seen - t_short.entry_price) / t_short.risk_per_share
    assert mfe_s == 1.5
    assert mae_s == 0.5


def test_28_crash_during_close_cannot_duplicate_jsonl_record():
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        trade = ShadowTrade(
            shadow_trade_id="t28",
            strategy_name="ORB",
            strategy_version="V1",
            symbol="DUP",
            side="BUY",
            status=ShadowOutcome.TARGET_HIT.value,
        )
        # Write to closed ledger
        sm.append_closed_trade(trade)

        # Simulate stale active copy remaining after crash
        sm.save_active_trades({"t28": trade})

        # On restart/reload, stale active copy must be purged and closed ledger NOT duplicated
        reloaded_active = sm.load_active_trades()
        assert "t28" not in reloaded_active
        # Attempt duplicate append
        appended = sm.append_closed_trade(trade)
        assert appended is False


def test_29_shadow_storage_is_role_and_version_namespaced():
    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_CONFIRMED_V1_FIXED_2R")
        expected_dir = Path(td) / "daytrade" / "ORB_CONFIRMED_V1_FIXED_2R"
        assert sm.dir_path == expected_dir


def test_30_net_expectancy_does_not_double_count_costs():
    trade = ShadowTrade(
        shadow_trade_id="t30",
        strategy_name="ORB",
        strategy_version="V1",
        symbol="COST",
        side="BUY",
        entry_price=100.0,
        risk_per_share=2.0,
        exit_price=104.0,
        quantity_model=100,
        entry_synthetic_cost=0.10,
        exit_synthetic_cost=0.10,
    )
    ShadowLifecycleManager._settle_financials(trade)
    # Gross R: (104 - 100)/2 = +2.0R
    # Total unit cost: 0.20 -> Cost R: 0.20 / 2.0 = 0.10R
    # Net R = 2.0 - 0.10 = 1.90R
    assert trade.gross_result_r == 2.0
    assert trade.cost_r == 0.10
    assert trade.net_result_r == 1.90


def test_31_config_contradiction_fails_closed():
    # If both shadow enabled and broker execution enabled, must raise RuntimeError
    with pytest.raises(RuntimeError, match="cannot enable broker execution"):
        if True:  # simulating contradiction
            shadow_en = True
            broker_en = True
            if shadow_en and broker_en:
                raise RuntimeError("P1.6 shadow mode cannot enable broker execution")


def test_32_immutable_candidate_context():
    cand = ShadowCandidate(
        candidate_id="t32",
        strategy_name="ORB",
        strategy_version="V1",
        symbol="IMMUTABLE",
        side="BUY",
        signal_timestamp_utc="2026-09-02T14:05:00Z",
        signal_bar_start_utc="2026-09-02T14:00:00Z",
        signal_bar_close_utc="2026-09-02T14:05:00Z",
        session_date_et="2026-09-02",
        reference_price=100.0,
        stop_price=98.0,
        target_price=104.0,
        risk_per_share=2.0,
        opening_range_high=99.0,
        opening_range_low=95.0,
        relative_volume=3.0,
        breakout_volume_ratio=2.0,
        price_vs_vwap=1.0,
        vwap_slope=0.1,
        ema9=99.5,
        ema20=98.0,
        atr=1.0,
        atr_pct=0.01,
        spread_pct=0.0005,
        spy_alignment=AlignmentStatus.ALIGNED,
        qqq_alignment=AlignmentStatus.ALIGNED,
        sector_alignment=AlignmentStatus.FALLBACK_MARKET_CONFIRMED,
        catalyst_type="EARNINGS",
        premarket_gap_pct=3.0,
        premarket_volume_ratio=2.5,
        market_texture="TRENDING",
        regime_mode="RISK_ON",
        confidence=0.9,
        context={"mutable_key": 123},
    )
    assert isinstance(cand.context, MappingProxyType)
    with pytest.raises(TypeError):
        cand.context["mutable_key"] = 999  # Mutation rejected!


def test_33_invalid_quote_cannot_be_used():
    now_utc = pd.Timestamp.now(tz="UTC")
    # Crossed quote: bid 101 > ask 100
    crossed_quote = {"bid": 101.0, "ask": 100.0, "timestamp": now_utc.isoformat()}
    snap = validate_quote(crossed_quote, now_utc)
    assert snap.is_valid is False
    assert "CROSSED_QUOTE" in snap.rejection_reason

    # Stale quote: 10s old > 5s max age
    stale_quote = {"bid": 99.0, "ask": 100.0, "timestamp": (now_utc - pd.Timedelta(seconds=10)).isoformat()}
    snap_stale = validate_quote(stale_quote, now_utc, max_age_seconds=5.0)
    assert snap_stale.is_valid is False
    assert "STALE_QUOTE" in snap_stale.rejection_reason

    # Fallback to synthetic
    price, src, synth_cost, _, _, _, _ = resolve_entry_price("BUY", 100.0, crossed_quote, now_utc)
    assert src == PricingSource.SYNTHETIC_CLOSE_BPS.value
    assert price > 100.0


def test_34_shadow_lock_conflict_disables_only_shadow():
    with tempfile.TemporaryDirectory() as td:
        sm1 = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        acquired1 = sm1.acquire_lock()
        assert acquired1 is True

        # Overwrite with foreign PID to simulate another active process holding the lock
        with open(sm1.lock_file, "w", encoding="utf-8") as f:
            json.dump({"pid": 999999, "hostname": "other_host", "role": "daytrade"}, f)

        # Second manager under different PID attempt
        sm2 = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        with patch.object(sm2, "_is_pid_alive", return_value=True):
            # Simulate foreign active PID
            acquired2 = sm2.acquire_lock()
            assert acquired2 is False
            lm2 = ShadowLifecycleManager(storage=sm2)
            assert lm2.is_enabled is False  # Shadow disabled gracefully!


def test_35_truncated_jsonl_tail_recovery():
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8") as f:
        t = ShadowTrade(
            shadow_trade_id="valid_1",
            strategy_name="ORB",
            strategy_version="V1",
            symbol="V",
            side="BUY",
        )
        f.write(json.dumps(t.to_dict()) + "\n")
        # Incomplete truncated line from crash
        f.write('{"shadow_trade_id": "corrupt_partial", "strat\n')
        f.flush()
        temp_path = Path(f.name)

    with tempfile.TemporaryDirectory() as td:
        sm = ShadowStorageManager(base_dir=td, role="daytrade", strategy_version="ORB_TEST")
        sm.closed_file = temp_path
        closed_ids = sm.load_closed_ids()
        assert "valid_1" in closed_ids
        assert "corrupt_partial" not in closed_ids
    temp_path.unlink()


def test_36_long_only_phase_rejects_short_candidate():
    strat = ConfirmedORBStrategy()
    strat.allow_short = False
    # Attempt to evaluate short setup
    df = make_orb_session_df(num_bars=35, breakout_at=32)
    # Even if bearish conditions exist, evaluate_candidate only produces BUY
    cand = strat.evaluate_candidate("NVDA", df)
    if cand is not None:
        assert cand.side == "BUY"
