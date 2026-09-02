"""
strategies/audit/would_have_pnl.py -- Counterfactual P&L audit for suppressed/vetoed candidates.

For each suppressed signal, simulate the bot's OWN exit logic
(ATR stop, target, EOD flatten) against actual intraday bars to answer:
did the veto SAVE capital or MISS profit?
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional


class Side(Enum):
    LONG = 1
    SHORT = -1          # inverse ETFs held long = bearish exposure, still Side.LONG


class ExitReason(Enum):
    TARGET = "TARGET_HIT"
    STOP = "ATR_STOP_HIT"
    EOD = "EOD_FLATTEN"


@dataclass
class SuppressedSignal:
    symbol: str
    reason: str                 # veto reason from log
    entry_price: float
    atr: float                  # ATR at signal time
    entry_idx: int = 0          # bar index at hypothetical entry
    stop_mult: float = 1.5      # bot's ATR stop multiple
    target_mult: float = 2.0    # bot's ATR target multiple (R:R ~1.33)


@dataclass
class AuditResult:
    symbol: str
    reason: str
    entry: float
    exit: float
    exit_reason: ExitReason
    pnl_pct: float
    verdict: str                # "VETO_SAVED" | "VETO_MISSED" | "NEUTRAL"


def simulate_exit(sig: SuppressedSignal, bars: List[Any]) -> AuditResult:
    """
    Walk bars forward from entry using the bot's own stop/target/EOD rules.
    `bars` = list of objects with .high .low .close, from entry_idx onward.
    Long-only convention (inverse ETFs are held long for bearish exposure).
    """
    stop_price   = sig.entry_price - sig.stop_mult * sig.atr
    target_price = sig.entry_price + sig.target_mult * sig.atr

    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None

    eval_bars = bars[sig.entry_idx:] if bars and len(bars) > sig.entry_idx else bars
    if not eval_bars:
        # Fallback if no bars
        return AuditResult(
            symbol=sig.symbol,
            reason=sig.reason,
            entry=sig.entry_price,
            exit=sig.entry_price,
            exit_reason=ExitReason.EOD,
            pnl_pct=0.0,
            verdict="NEUTRAL",
        )

    for bar in eval_bars:
        # Stop checked first (conservative: assume worst intrabar fill)
        bar_low = float(getattr(bar, "low", getattr(bar, "close", sig.entry_price)))
        bar_high = float(getattr(bar, "high", getattr(bar, "close", sig.entry_price)))

        if bar_low <= stop_price:
            exit_price, exit_reason = stop_price, ExitReason.STOP
            break
        if bar_high >= target_price:
            exit_price, exit_reason = target_price, ExitReason.TARGET
            break

    if exit_price is None:  # never hit -> EOD flatten
        last_close = float(getattr(eval_bars[-1], "close", sig.entry_price))
        exit_price, exit_reason = last_close, ExitReason.EOD

    pnl_pct = (exit_price - sig.entry_price) / sig.entry_price * 100.0 if sig.entry_price > 0 else 0.0

    if pnl_pct > 0.10:
        verdict = "VETO_MISSED"                     # would have made money
    elif pnl_pct < -0.10:
        verdict = "VETO_SAVED"                      # would have lost money
    else:
        verdict = "NEUTRAL"

    return AuditResult(
        symbol=sig.symbol,
        reason=sig.reason,
        entry=sig.entry_price,
        exit=round(exit_price, 4),
        exit_reason=exit_reason,
        pnl_pct=round(pnl_pct, 3),
        verdict=verdict,
    )


def run_audit(signals: List[SuppressedSignal], bars_lookup: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Audit all suppressed signals; return per-signal + aggregate summary."""
    results = [simulate_exit(s, bars_lookup.get(s.symbol, [])) for s in signals]

    saved  = sum(r.pnl_pct for r in results if r.verdict == "VETO_SAVED")
    missed = sum(r.pnl_pct for r in results if r.verdict == "VETO_MISSED")

    return {
        "results": results,
        "n_saved":  sum(1 for r in results if r.verdict == "VETO_SAVED"),
        "n_missed": sum(1 for r in results if r.verdict == "VETO_MISSED"),
        "n_neutral":sum(1 for r in results if r.verdict == "NEUTRAL"),
        "total_saved_pct":  round(saved, 3),        # negative = losses avoided
        "total_missed_pct": round(missed, 3),       # positive = profit forgone
        "net_verdict": "REGIME_CORRECT" if abs(saved) >= missed else "TOO_CONSERVATIVE",
    }
