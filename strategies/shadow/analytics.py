"""
strategies.shadow.analytics -- Statistical Expectancy & Quantitative Gate Analysis.

Evaluates shadow trade logs from data/shadow/{role}/{strategy_version}/closed.jsonl.
Computes:
1. Sample size N and unique ET session dates.
2. Win / Loss / Breakeven classification.
3. Profit Factor with denominator zero handling (unstable).
4. Friction-adjusted net expectancy in R-multiples.
5. Outlier dependence check (top winner <= 35% of gross profit).
6. Synthetic pricing share tracking.
7. Breakdowns by time bucket, market texture, and pricing source.
8. Research promotion gate status (RESEARCH_PROMISING vs INSUFFICIENT_SAMPLE vs RESEARCH_REJECTED).
"""
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from strategies.shadow.models import ShadowTrade

logger = logging.getLogger(__name__)


class ResearchGateStatus(str, Enum):
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    RESEARCH_PROMISING = "RESEARCH_PROMISING"
    RESEARCH_REJECTED = "RESEARCH_REJECTED"
    NEEDS_MORE_DATA = "NEEDS_MORE_DATA"


@dataclass
class ShadowAnalyticsReport:
    strategy_version: str
    total_trades: int
    unique_sessions: int
    win_count: int
    loss_count: int
    breakeven_count: int
    win_rate_pct: float
    profit_factor: float
    pf_is_unstable: bool
    gross_profit: float
    gross_loss: float
    net_pnl: float
    average_win_r: float
    average_loss_r: float
    win_loss_ratio: float
    net_expectancy_r: float
    max_consecutive_losses: int
    top_winner_pct: float
    synthetic_pricing_pct: float
    gate_status: ResearchGateStatus
    gate_reasons: List[str] = field(default_factory=list)
    time_bucket_breakdown: Dict[str, dict] = field(default_factory=dict)
    texture_breakdown: Dict[str, dict] = field(default_factory=dict)
    pricing_source_breakdown: Dict[str, dict] = field(default_factory=dict)


def evaluate_shadow_metrics(
    closed_jsonl_path: Optional[Path] = None,
    min_sample_size: int = 30,
    min_sessions: int = 5,
    min_profit_factor: float = 1.25,
    min_win_loss_ratio: float = 1.30,
    max_outlier_winner_pct: float = 35.0,
) -> ShadowAnalyticsReport:
    """Parses closed shadow trades and computes research-grade audit statistics."""
    default_path = Path("data/shadow/daytrade/ORB_CONFIRMED_V1_FIXED_2R/closed.jsonl")
    path = Path(closed_jsonl_path or default_path)

    trades: List[ShadowTrade] = []
    if path.exists():
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    data = json.loads(line_s)
                    trades.append(ShadowTrade.from_dict(data))
                except Exception as e:
                    logger.debug("Analytics parse error on line: %s", e)

    strat_ver = trades[0].strategy_version if trades else "ORB_CONFIRMED_V1_FIXED_2R"
    total_n = len(trades)

    if total_n == 0:
        return ShadowAnalyticsReport(
            strategy_version=strat_ver,
            total_trades=0,
            unique_sessions=0,
            win_count=0,
            loss_count=0,
            breakeven_count=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            pf_is_unstable=False,
            gross_profit=0.0,
            gross_loss=0.0,
            net_pnl=0.0,
            average_win_r=0.0,
            average_loss_r=0.0,
            win_loss_ratio=0.0,
            net_expectancy_r=0.0,
            max_consecutive_losses=0,
            top_winner_pct=0.0,
            synthetic_pricing_pct=0.0,
            gate_status=ResearchGateStatus.INSUFFICIENT_SAMPLE,
            gate_reasons=[f"Zero closed trades found in {path}"],
        )

    # Unique sessions by ET trading date
    unique_dates = set(t.session_date_et for t in trades if t.session_date_et)
    session_count = len(unique_dates)

    wins = [t for t in trades if t.outcome_class == "WIN"]
    losses = [t for t in trades if t.outcome_class == "LOSS"]
    bes = [t for t in trades if t.outcome_class == "BREAKEVEN"]

    win_count = len(wins)
    loss_count = len(losses)
    be_count = len(bes)
    win_rate = round((win_count / total_n) * 100.0, 2)

    gross_profit = sum(t.net_pnl for t in wins if t.net_pnl > 0)
    gross_loss = abs(sum(t.net_pnl for t in losses if t.net_pnl < 0))
    net_pnl = round(sum(t.net_pnl for t in trades), 2)

    # Profit factor calculation
    if gross_loss == 0:
        pf = float("inf") if gross_profit > 0 else 0.0
        pf_unstable = True
    else:
        pf = round(gross_profit / gross_loss, 2)
        pf_unstable = False

    # Excursion R averages
    win_r_vals = [t.net_result_r for t in wins]
    loss_r_vals = [abs(t.net_result_r) for t in losses]
    avg_win_r = round(float(np.mean(win_r_vals)), 2) if win_r_vals else 0.0
    avg_loss_r = round(float(np.mean(loss_r_vals)), 2) if loss_r_vals else 0.0
    win_loss_ratio = round(avg_win_r / avg_loss_r, 2) if avg_loss_r > 0 else 0.0

    # Net expectancy in R: average net result R across all trades
    all_net_r = [t.net_result_r for t in trades]
    net_expectancy = round(float(np.mean(all_net_r)), 4) if all_net_r else 0.0

    # Max consecutive losses
    max_cons_losses = 0
    cur_cons_losses = 0
    for t in trades:
        if t.outcome_class == "LOSS":
            cur_cons_losses += 1
            max_cons_losses = max(max_cons_losses, cur_cons_losses)
        else:
            cur_cons_losses = 0

    # Top winner outlier check
    max_win_pnl = max([t.net_pnl for t in wins] or [0.0])
    top_winner_pct = round((max_win_pnl / gross_profit) * 100.0, 2) if gross_profit > 0 else 0.0

    # Synthetic pricing share
    synthetic_count = sum(1 for t in trades if t.entry_pricing_source == "SYNTHETIC_CLOSE_BPS")
    synth_pct = round((synthetic_count / total_n) * 100.0, 2)

    # ── Promotion Gate Evaluation ──
    reasons = []
    if total_n < min_sample_size:
        reasons.append(f"Insufficient sample size: {total_n} < {min_sample_size}")
    if session_count < min_sessions:
        reasons.append(f"Insufficient session breadth: {session_count} < {min_sessions} days")
    if pf_unstable:
        reasons.append("Profit factor unstable (0 losses recorded)")
    elif pf < min_profit_factor:
        reasons.append(f"Profit factor below threshold: {pf:.2f} < {min_profit_factor:.2f}")
    if net_expectancy <= 0:
        reasons.append(f"Net expectancy non-positive: {net_expectancy:.4f}R <= 0.0R")
    if win_loss_ratio < min_win_loss_ratio and loss_count > 0:
        reasons.append(f"Win/Loss ratio below threshold: {win_loss_ratio:.2f} < {min_win_loss_ratio:.2f}")
    if top_winner_pct > max_outlier_winner_pct and win_count > 1:
        reasons.append(f"Outlier dependence: Top winner represents {top_winner_pct:.1f}% > {max_outlier_winner_pct:.1f}% of profit")

    if total_n < min_sample_size or session_count < min_sessions or pf_unstable:
        gate_status = ResearchGateStatus.INSUFFICIENT_SAMPLE
    elif reasons:
        gate_status = ResearchGateStatus.RESEARCH_REJECTED
    else:
        gate_status = ResearchGateStatus.RESEARCH_PROMISING

    # ── Breakdowns ──
    def _calc_subset(subset: List[ShadowTrade]) -> dict:
        if not subset:
            return {"count": 0, "win_rate": 0.0, "net_pnl": 0.0, "expectancy_r": 0.0}
        sub_wins = sum(1 for x in subset if x.outcome_class == "WIN")
        sub_pnl = sum(x.net_pnl for x in subset)
        sub_r = np.mean([x.net_result_r for x in subset])
        return {
            "count": len(subset),
            "win_rate": round((sub_wins / len(subset)) * 100.0, 1),
            "net_pnl": round(sub_pnl, 2),
            "expectancy_r": round(float(sub_r), 4),
        }

    # Time bucket breakdown
    time_buckets = {}
    for bucket in set(t.entry_time_bucket for t in trades):
        time_buckets[bucket] = _calc_subset([t for t in trades if t.entry_time_bucket == bucket])

    # Texture breakdown
    textures = {}
    for tex in set(t.market_texture for t in trades):
        textures[tex] = _calc_subset([t for t in trades if t.market_texture == tex])

    # Pricing source breakdown
    pricing_sources = {}
    for src in set(t.entry_pricing_source for t in trades):
        pricing_sources[src] = _calc_subset([t for t in trades if t.entry_pricing_source == src])

    return ShadowAnalyticsReport(
        strategy_version=strat_ver,
        total_trades=total_n,
        unique_sessions=session_count,
        win_count=win_count,
        loss_count=loss_count,
        breakeven_count=be_count,
        win_rate_pct=win_rate,
        profit_factor=pf,
        pf_is_unstable=pf_unstable,
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        net_pnl=net_pnl,
        average_win_r=avg_win_r,
        average_loss_r=avg_loss_r,
        win_loss_ratio=win_loss_ratio,
        net_expectancy_r=net_expectancy,
        max_consecutive_losses=max_cons_losses,
        top_winner_pct=top_winner_pct,
        synthetic_pricing_pct=synth_pct,
        gate_status=gate_status,
        gate_reasons=reasons,
        time_bucket_breakdown=time_buckets,
        texture_breakdown=textures,
        pricing_source_breakdown=pricing_sources,
    )
