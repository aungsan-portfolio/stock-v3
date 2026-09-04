# ─────────────────────────────────────────────────────────────────────────────
#  position_rotator.py
#  Dynamic Position Rotation Engine
#
#  တာဝန်:
#    • Current open positions များကို Re-score ပြုလုပ်သည်
#    • New high-conviction signals နှင့် နှိုင်းယှဉ်သည်
#    • Rotation လုပ်သင့်/မလုပ်သင့် ဆုံးဖြတ်သည်
#    • Alpaca Bridge ကို Exit + Entry order pair ပို့သည်
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from strategies.rotation_config import (
    ROTATION_MIN_INCOMING_CONF,
    ROTATION_WEAK_POSITION_CONF,
    ROTATION_MIN_CONF_DELTA,
    ROTATION_PROTECT_PROFIT_PCT,
    ROTATION_HARD_STOP_LOSS_PCT,
    ROTATION_COOLDOWN_CYCLES,
    MAX_ROTATIONS_PER_CYCLE,
    SCORE_CONF_WEIGHT,
    SCORE_PNL_WEIGHT,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    """Alpaca Bridge မှ ရယူသော လက်ရှိ Open Position တစ်ခု၏ Snapshot"""
    symbol:          str
    side:            str            # "long" | "short"
    qty:             float
    avg_entry_price: float
    current_price:   float
    unrealized_pnl:  float
    current_conf:    float = 0.0    # Re-scored confidence (predictor မှ)
    is_protected:    bool  = False  # alpaca_bridge ၏ protected list မှ

    @property
    def pnl_pct(self) -> float:
        """Unrealised P&L as a percentage of entry cost"""
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price

    @property
    def rotation_score(self) -> float:
        """
        Composite score — ဒီ score နိမ့်လေ Rotation Candidate ဖြစ်ဖို့
        အလားအလာ ပိုများလေ ဖြစ်သည်။

        pnl_score logic:
          • P&L ကောင်းလေ → score ကောင်းလေ (ဖယ်မထုတ်ချင်)
          • P&L ဆိုးလေ   → score ဆိုးလေ   (ဖယ်ထုတ်ချင်)
        """
        # P&L ကို 0-1 range သို့ Normalise (clamp -10% … +10%)
        pnl_clamped  = max(-0.10, min(0.10, self.pnl_pct))
        pnl_score    = (pnl_clamped + 0.10) / 0.20   # 0.0 → 1.0

        return (SCORE_CONF_WEIGHT * self.current_conf +
                SCORE_PNL_WEIGHT  * pnl_score)


@dataclass
class IncomingSignal:
    """Predictor မှ ထွက်ရှိလာသော Signal တစ်ခု"""
    symbol:     str
    action:     str       # "BUY" | "SELL" | "HOLD"
    confidence: float
    price:      float

    @property
    def rotation_score(self) -> float:
        """Incoming signal ၏ attractiveness score (confidence only)"""
        return self.confidence


@dataclass
class RotationDecision:
    """Rotation Engine ၏ ဆုံးဖြတ်ချက် Output"""
    should_rotate:   bool
    exit_position:   Optional[OpenPosition]  = None
    enter_signal:    Optional[IncomingSignal] = None
    reason:          str                      = ""
    blocked_reason:  str                      = ""


@dataclass
class RotationState:
    """
    Cooldown နှင့် Rotation History ကို Session တစ်ခုလုံး ထိမ်းသိမ်းသော State
    (alpaca_bridge instance တွင် တစ်ခုတည်း instantiate လုပ်ပါ)
    """
    cooldown_remaining:    int             = 0
    rotations_this_cycle:  int             = 0
    rotation_history:      list[dict]      = field(default_factory=list)

    def reset_cycle(self):
        """Scan Cycle အသစ်တိုင်း ဒါကို ခေါ်ပါ"""
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        self.rotations_this_cycle = 0

    def record_rotation(self, decision: RotationDecision):
        self.cooldown_remaining   = ROTATION_COOLDOWN_CYCLES
        self.rotations_this_cycle += 1
        self.rotation_history.append({
            "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "exited":       decision.exit_position.symbol  if decision.exit_position  else None,
            "entered":      decision.enter_signal.symbol   if decision.enter_signal   else None,
            "exit_conf":    decision.exit_position.current_conf if decision.exit_position else None,
            "enter_conf":   decision.enter_signal.confidence   if decision.enter_signal  else None,
            "reason":       decision.reason,
        })


# ── Ratchet configuration ──────────────────────────────────────────────────
# (pnl_threshold, stop_offset_from_entry, label)
_RATCHET_RULES: list[tuple[float, float, str]] = [
    (0.05, 0.005, "breakeven+"),     # +5%  → entry + 0.5%
    (0.08, 0.040, "lock-in-4pct"),   # +8%  → entry + 4.0%
    (0.12, 0.070, "lock-in-7pct"),   # +12% → entry + 7.0%
    (0.15, 0.100, "lock-in-10pct"),  # +15% → entry + 10.0%
    (0.20, 0.150, "lock-in-15pct"),  # +20% → entry + 15.0%
]


@dataclass
class RatchetResult:
    symbol:       str
    old_stop:     float
    new_stop:     float
    rule_applied: str
    pnl_pct:      float
    skipped:      bool
    reason:       str


# ─────────────────────────────────────────────────────────────────────────────
#  Core Rotation Engine
# ─────────────────────────────────────────────────────────────────────────────

class PositionRotator:
    """
    Usage (alpaca_bridge.py ထဲမှာ):

        rotator = PositionRotator(bridge=bridge)  # session start တွင် တစ်ကြိမ်

        # Scan cycle အစတွင်
        rotator.state.reset_cycle()

        # Signal + Position data ရပြီးနောက်
        decision = rotator.evaluate(open_positions, incoming_signals)

        if decision.should_rotate:
            bridge.close_position(decision.exit_position.symbol)
            bridge.place_order(decision.enter_signal)
            rotator.state.record_rotation(decision)
    """

    def __init__(self, bridge=None):
        self.state = RotationState()
        self.bridge = bridge
        logger.info("PositionRotator initialised.")

    # ── Ratchet Stop Layer (Risk Protection) ──────────────────────────────────

    def _ratchet_stops(
        self,
        positions: list[dict | OpenPosition | object],
        dry_run: bool = False,
        bridge=None,
    ) -> list[RatchetResult]:
        """
        Profit-Lock Ratchet Layer — Risk/Safety မှ ဆင်းသည်၊ Strategy မဟုတ်။

        Guarantees:
          • Stop ကို NEVER ချနိုင် (one-direction only)
          • Bridge failure → log + skip, trade block မဖြစ်
          • dry_run=True → order မထုတ်, log သာ
        """
        active_bridge = bridge or self.bridge
        results: list[RatchetResult] = []

        for pos in positions:
            if isinstance(pos, dict):
                symbol = str(pos.get("symbol", "")).upper().strip()
                entry = float(pos.get("avg_entry_price", 0.0) or 0.0)
                current = float(pos.get("current_price", 0.0) or 0.0)
            else:
                symbol = str(getattr(pos, "symbol", "")).upper().strip()
                entry = float(getattr(pos, "avg_entry_price", 0.0) or 0.0)
                current = float(getattr(pos, "current_price", 0.0) or 0.0)

            if entry <= 0:
                continue

            pnl_pct = (current - entry) / entry

            # ── Rule ရှာသည် (highest threshold ဦးစွာ) ──────────────────
            matched_offset: Optional[float] = None
            matched_label: str = "none"

            for threshold, offset, label in sorted(
                _RATCHET_RULES, key=lambda r: r[0], reverse=True
            ):
                if pnl_pct >= (threshold - 1e-5):
                    matched_offset = offset
                    matched_label = label
                    break

            if matched_offset is None:
                results.append(RatchetResult(
                    symbol=symbol, old_stop=0.0, new_stop=0.0,
                    rule_applied="none", pnl_pct=pnl_pct,
                    skipped=True, reason=f"pnl {pnl_pct:.2%} below min threshold",
                ))
                continue

            new_stop = round(entry * (1.0 + matched_offset), 2)
            current_stop = None
            if active_bridge is not None and hasattr(active_bridge, "get_current_stop"):
                try:
                    current_stop = active_bridge.get_current_stop(symbol)
                except Exception as exc:
                    logger.debug("[RATCHET] get_current_stop error for %s: %s", symbol, exc)

            logger.info(
                "[RATCHET EVAL] %-6s | rule=%-14s | entry=%7.2f | current_px=%7.2f (pnl=%+.2f%%) | resting_stop=%s | target_stop=%7.2f",
                symbol, matched_label, entry, current, pnl_pct * 100,
                f"{current_stop:.2f}" if current_stop is not None else "None", new_stop,
            )

            # ── One-direction guard ──────────────────────────────────────
            if current_stop is not None and new_stop <= current_stop:
                reason_str = (
                    f"already protected at {current_stop:.2f}, above/equal target {new_stop:.2f} "
                    f"for current PnL {pnl_pct * 100:+.2f}%; no stop downgrade permitted"
                )
                logger.info("[RATCHET SKIP] %-6s | %s", symbol, reason_str)
                results.append(RatchetResult(
                    symbol=symbol, old_stop=current_stop, new_stop=new_stop,
                    rule_applied=matched_label, pnl_pct=pnl_pct,
                    skipped=True,
                    reason=reason_str,
                ))
                continue

            # ── Dry-run path ─────────────────────────────────────────────
            if dry_run:
                logger.info(
                    "[RATCHET DRY-RUN] %-6s | %-16s | %7.2f → %7.2f | pnl=%+.2f%%",
                    symbol, matched_label,
                    current_stop or 0.0, new_stop, pnl_pct * 100,
                )
                results.append(RatchetResult(
                    symbol=symbol, old_stop=current_stop or 0.0, new_stop=new_stop,
                    rule_applied=matched_label, pnl_pct=pnl_pct,
                    skipped=False, reason="dry_run",
                ))
                continue

            # ── Live path ────────────────────────────────────────────────
            try:
                if active_bridge is not None and hasattr(active_bridge, "replace_stop"):
                    active_bridge.replace_stop(symbol=symbol, new_stop_price=new_stop)
                logger.info(
                    "[RATCHET] %-6s | %-16s | %7.2f → %7.2f | pnl=%+.2f%%",
                    symbol, matched_label,
                    current_stop or 0.0, new_stop, pnl_pct * 100,
                )
                results.append(RatchetResult(
                    symbol=symbol, old_stop=current_stop or 0.0, new_stop=new_stop,
                    rule_applied=matched_label, pnl_pct=pnl_pct,
                    skipped=False, reason="ok",
                ))
            except Exception as exc:
                logger.error("[RATCHET FAILED] %s | %s", symbol, exc)
                results.append(RatchetResult(
                    symbol=symbol, old_stop=current_stop or 0.0, new_stop=new_stop,
                    rule_applied=matched_label, pnl_pct=pnl_pct,
                    skipped=True, reason=f"bridge_error: {exc}",
                ))

        return results

    def _log_ratchet_summary(self, results: list[RatchetResult]) -> None:
        """Cycle တိုင်း telemetry snapshot — shadow log နှင့် compatible"""
        acted = [r for r in results if not r.skipped]
        skipped = [r for r in results if r.skipped]
        logger.info(
            "[RATCHET SUMMARY] acted=%d skipped=%d | %s",
            len(acted), len(skipped),
            {r.symbol: r.rule_applied for r in acted} or "none",
        )

    def run_cycle(self, dry_run: bool = False) -> None:
        """Main rotation cycle — ratchet ကို FIRST run ရသည်"""
        if not self.bridge or not hasattr(self.bridge, "get_positions"):
            return
        positions = self.bridge.get_positions()

        # ── Step 1: Risk layer (ratchet) ────────────────────────────────
        ratchet_results = self._ratchet_stops(positions, dry_run=dry_run)
        self._log_ratchet_summary(ratchet_results)

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        open_positions:   list[OpenPosition],
        incoming_signals: list[IncomingSignal],
    ) -> RotationDecision:
        """
        Main entry point.
        Returns a RotationDecision — caller decides whether to execute it.
        """

        # ── Guard: Cooldown ──────────────────────────────────────────────────
        if self.state.cooldown_remaining > 0:
            msg = (f"Rotation cooldown active "
                   f"({self.state.cooldown_remaining} cycles remaining)")
            logger.info("[ROTATION BLOCKED] %s", msg)
            return RotationDecision(should_rotate=False, blocked_reason=msg)

        # ── Guard: Max rotations this cycle ──────────────────────────────────
        if self.state.rotations_this_cycle >= MAX_ROTATIONS_PER_CYCLE:
            msg = f"Max rotations/cycle ({MAX_ROTATIONS_PER_CYCLE}) reached"
            logger.info("[ROTATION BLOCKED] %s", msg)
            return RotationDecision(should_rotate=False, blocked_reason=msg)

        # ── Step 1: Find rotation candidates (weak existing positions) ────────
        candidates = self._get_rotation_candidates(open_positions)
        if not candidates:
            logger.info("[ROTATION] No weak positions found — nothing to rotate.")
            return RotationDecision(
                should_rotate=False,
                blocked_reason="No positions below weak-conf threshold"
            )

        # ── Step 2: Find high-conviction incoming signals ─────────────────────
        strong_signals = self._get_strong_incoming_signals(
            incoming_signals, open_positions
        )
        if not strong_signals:
            logger.info("[ROTATION] No strong incoming signals — no rotation.")
            return RotationDecision(
                should_rotate=False,
                blocked_reason="No incoming signals above min-conf threshold"
            )

        # ── Step 3: Best pair selection ───────────────────────────────────────
        weakest_position = candidates[0]           # lowest rotation_score first
        strongest_signal = strong_signals[0]       # highest confidence first

        conf_delta = (strongest_signal.confidence -
                      weakest_position.current_conf)

        if conf_delta < ROTATION_MIN_CONF_DELTA:
            msg = (f"Conf delta too small: "
                   f"{strongest_signal.symbol}({strongest_signal.confidence:.2f}) "
                   f"vs {weakest_position.symbol}({weakest_position.current_conf:.2f}) "
                   f"Δ={conf_delta:.2f} < {ROTATION_MIN_CONF_DELTA}")
            logger.info("[ROTATION BLOCKED] %s", msg)
            return RotationDecision(should_rotate=False, blocked_reason=msg)

        # ── Step 4: Build & return decision ───────────────────────────────────
        reason = (
            f"ROTATE: exit {weakest_position.symbol} "
            f"(conf={weakest_position.current_conf:.2f}, "
            f"pnl={weakest_position.pnl_pct*100:.2f}%, "
            f"score={weakest_position.rotation_score:.3f}) "
            f"→ enter {strongest_signal.symbol} "
            f"(conf={strongest_signal.confidence:.2f}, "
            f"Δ={conf_delta:.2f})"
        )
        logger.info("[ROTATION APPROVED] %s", reason)

        return RotationDecision(
            should_rotate=True,
            exit_position=weakest_position,
            enter_signal=strongest_signal,
            reason=reason,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _get_rotation_candidates(
        self,
        positions: list[OpenPosition]
    ) -> list[OpenPosition]:
        """
        ဖယ်ထုတ်နိုင်သော Position များကို စစ်ထုတ်၍ Weakest ကို အရင်ဆုံး ထားပါ။
        Protected Winners နှင့် Hard-Stop candidates များကို ဖယ်ထားသည်။
        """
        candidates = []

        for pos in positions:

            # ── Skip: Protected by alpaca_bridge ────────────────────────────
            if pos.is_protected:
                logger.debug(
                    "[ROTATION SKIP] %s — marked protected by bridge", pos.symbol
                )
                continue

            # ── Skip: Winner Protection ──────────────────────────────────────
            if pos.pnl_pct >= ROTATION_PROTECT_PROFIT_PCT:
                logger.info(
                    "[ROTATION SKIP] %s — protected winner "
                    "(pnl=%.2f%% ≥ %.0f%%)",
                    pos.symbol, pos.pnl_pct * 100,
                    ROTATION_PROTECT_PROFIT_PCT * 100
                )
                continue

            # ── Skip: Hard Stop territory (separate risk handler owns this) ──
            if pos.pnl_pct <= ROTATION_HARD_STOP_LOSS_PCT:
                logger.warning(
                    "[ROTATION SKIP] %s — hard-stop territory "
                    "(pnl=%.2f%% ≤ %.0f%%). "
                    "Risk manager should handle this.",
                    pos.symbol, pos.pnl_pct * 100,
                    ROTATION_HARD_STOP_LOSS_PCT * 100
                )
                continue

            # ── Candidate: Confidence below weak threshold ────────────────────
            if pos.current_conf < ROTATION_WEAK_POSITION_CONF:
                candidates.append(pos)
                logger.debug(
                    "[ROTATION CANDIDATE] %s  conf=%.2f  pnl=%.2f%%  "
                    "score=%.3f",
                    pos.symbol, pos.current_conf,
                    pos.pnl_pct * 100, pos.rotation_score
                )

        # Sort: weakest rotation_score first (most likely to rotate out)
        candidates.sort(key=lambda p: p.rotation_score)
        return candidates

    def _get_strong_incoming_signals(
        self,
        signals:    list[IncomingSignal],
        positions:  list[OpenPosition],
    ) -> list[IncomingSignal]:
        """
        High-conviction signals ကို စစ်ထုတ်သည်။
        ရှိပြီးသား Open Positions ဖြစ်နေသောကြောင့် HOLD / မလိုအပ်ဘဲ
        ဝင်မည့် Symbols များကို ဖယ်ထားသည်။
        """
        open_symbols = {p.symbol for p in positions}
        strong = []

        for sig in signals:
            # Already holding this symbol
            if sig.symbol in open_symbols:
                continue

            # Only BUY or actionable SELL (for short positions)
            if sig.action not in ("BUY", "SELL"):
                continue

            # Confidence floor
            if sig.confidence < ROTATION_MIN_INCOMING_CONF:
                continue

            strong.append(sig)

        # Sort: highest confidence first
        strong.sort(key=lambda s: s.confidence, reverse=True)
        return strong

    # ── Reporting ─────────────────────────────────────────────────────────────

    def rotation_summary(self) -> str:
        """Log-friendly summary of rotation history"""
        if not self.state.rotation_history:
            return "No rotations recorded this session."

        lines = ["─── Rotation History ───────────────────────────"]
        for i, rec in enumerate(self.state.rotation_history, 1):
            lines.append(
                f"  #{i:02d} [{rec['timestamp']}] "
                f"EXIT {rec['exited']} (conf={rec['exit_conf']:.2f}) "
                f"→ ENTER {rec['entered']} (conf={rec['enter_conf']:.2f}) "
                f"| {rec['reason'][:60]}"
            )
        lines.append("────────────────────────────────────────────────")
        return "\n".join(lines)
