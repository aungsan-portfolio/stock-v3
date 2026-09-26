#!/usr/bin/env python3
"""
tools/apply_post_session_patches.py

Applies the production patches to:
  1. strategies/regime_engine.py (Hysteresis & Time-Aware Buffer)
  2. strategies/trailing_stop.py (Strategy Attribution & Emergency Journal)
  3. strategies/orchestrator.py (Signal Export Hook)
  4. tests/test_regime_engine.py (Hysteresis Unit Tests)

To be executed immediately AFTER market close (after 3:00 AM ICT / 4:00 PM ET).
Makes automated .bak backups of each modified file.

Usage:
    python tools/apply_post_session_patches.py
    python tools/apply_post_session_patches.py --dry-run
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def backup_file(path: Path) -> Path:
    bak_path = path.with_suffix(path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copy2(path, bak_path)
        print(f"[BACKUP] Created {bak_path}")
    return bak_path


def patch_regime_engine(dry_run: bool = False) -> bool:
    target = BASE_DIR / "strategies" / "regime_engine.py"
    if not target.exists():
        print(f"[ERROR] Target not found: {target}")
        return False

    content = target.read_text(encoding="utf-8")

    # Check if already patched
    if "_apply_spy_hysteresis" in content:
        print("[SKIP] strategies/regime_engine.py already contains hysteresis patch.")
        return True

    # 1. Add hysteresis constants & functions after get_soxx_pct_change()
    anchor_1 = 'def get_soxx_pct_change() -> Optional[float]:\n    """Return the latest cached percentage change for SOXX semiconductor ETF."""\n    with _last_soxx_lock:\n        return _last_soxx_pct'
    
    hysteresis_code = '''
# ── Benchmark Score Hysteresis Deadbands ─────────────────────────────────────
_SPY_ESCALATE_THRESHOLD = -0.75   # step to -2 when SPY < this
_SPY_DEESC_FROM_2_TO_1  = -0.70   # recover above this to drop back to -1
_SPY_DEESC_FROM_1_TO_0  = -0.35   # recover above this to drop back to  0

_QQQ_ESCALATE_THRESHOLD = -1.00   # step to -2 when QQQ < this
_QQQ_DEESC_FROM_2_TO_1  = -0.90   # recover above this to drop back to -1
_QQQ_DEESC_FROM_1_TO_0  = -0.45   # recover above this to drop back to  0

_factor_hysteresis_lock = threading.Lock()
_last_spy_factor: Optional[int] = None
_last_qqq_factor: Optional[int] = None


def reset_factor_hysteresis() -> None:
    """Reset hysteresis state. Call at session start and from tests."""
    global _last_spy_factor, _last_qqq_factor
    with _factor_hysteresis_lock:
        _last_spy_factor = None
        _last_qqq_factor = None


def _apply_spy_hysteresis(spy_pct: Optional[float]) -> int:
    """
    Return hysteresis-smoothed SPY factor score.
    Escalates immediately; de-escalates only after recovery clears deadband.
    """
    if spy_pct is None:
        return 0

    global _last_spy_factor
    with _factor_hysteresis_lock:
        prev = _last_spy_factor

        if spy_pct < _SPY_ESCALATE_THRESHOLD:        # < -0.75 -> -2
            new_factor = -2
        elif spy_pct < -0.40:                        # < -0.40 -> at most -1
            if prev is not None and prev <= -2:
                new_factor = -2 if spy_pct < _SPY_DEESC_FROM_2_TO_1 else -1
            else:
                new_factor = -1
        else:                                        # >= -0.40 -> 0 or -1
            if prev is not None and prev <= -2:
                new_factor = -2 if spy_pct < _SPY_DEESC_FROM_2_TO_1 else (
                             -1 if spy_pct < _SPY_DEESC_FROM_1_TO_0 else 0)
            elif prev is not None and prev == -1:
                new_factor = -1 if spy_pct < _SPY_DEESC_FROM_1_TO_0 else 0
            else:
                new_factor = 0

        _last_spy_factor = new_factor
        return new_factor


def _apply_qqq_hysteresis(qqq_pct: Optional[float]) -> int:
    """
    Return hysteresis-smoothed QQQ factor score.
    Same escalate-fast / de-escalate-slow principle as SPY.
    """
    if qqq_pct is None:
        return 0

    global _last_qqq_factor
    with _factor_hysteresis_lock:
        prev = _last_qqq_factor

        if qqq_pct < _QQQ_ESCALATE_THRESHOLD:        # < -1.00 -> -2
            new_factor = -2
        elif qqq_pct < -0.50:                        # < -0.50 -> at most -1
            if prev is not None and prev <= -2:
                new_factor = -2 if qqq_pct < _QQQ_DEESC_FROM_2_TO_1 else -1
            else:
                new_factor = -1
        else:                                        # >= -0.50 -> 0 or -1
            if prev is not None and prev <= -2:
                new_factor = -2 if qqq_pct < _QQQ_DEESC_FROM_2_TO_1 else (
                             -1 if qqq_pct < _QQQ_DEESC_FROM_1_TO_0 else 0)
            elif prev is not None and prev == -1:
                new_factor = -1 if qqq_pct < _QQQ_DEESC_FROM_1_TO_0 else 0
            else:
                new_factor = 0

        _last_qqq_factor = new_factor
        return new_factor
'''

    if anchor_1 not in content:
        print("[ERROR] Could not find anchor_1 in strategies/regime_engine.py")
        return False

    content = content.replace(anchor_1, anchor_1 + "\n" + hysteresis_code)

    # 2. Add _buffer_display() and update summary() in MarketRegimeResult
    anchor_summary = '    def summary(self) -> str:\n        if self.reasons:'
    replacement_summary = '''    def _buffer_display(self) -> str:
        """Human-readable opening buffer status for log/summary output."""
        if not self.requires_opening_buffer:
            return "False"
        try:
            from strategies.session import is_past_opening_buffer
            return "ELAPSED(past_0945)" if is_past_opening_buffer(15) else "ACTIVE(wait_0945)"
        except Exception:
            return str(self.requires_opening_buffer)

    def summary(self) -> str:
        if self.reasons:'''

    if anchor_summary not in content:
        print("[ERROR] Could not find anchor_summary in strategies/regime_engine.py")
        return False
    content = content.replace(anchor_summary, replacement_summary)

    # Format base vs effective sizing/trades in summary
    anchor_sum_line = '            f"max_trades={self.max_trades_day} open_buffer_0945={self.requires_opening_buffer} "\n            f"sizing={self.position_scale*100:.0f}% boost=+{self.confidence_boost:.2f} reasons=[{reasons_str}]"'
    replacement_sum_line = '''            f"max_trades={0 if (self.halt_new_entries or not self.allow_new_longs) else self.max_trades_day} "
            f"open_buffer_0945={self._buffer_display()} "
            f"sizing={0 if (self.halt_new_entries or not self.allow_new_longs) else int(self.position_scale*100)}% "
            f"boost=+{self.confidence_boost:.2f} reasons=[{reasons_str}]"'''

    if anchor_sum_line in content:
        content = content.replace(anchor_sum_line, replacement_sum_line)
    else:
        content = content.replace(
            'open_buffer_0945={self.requires_opening_buffer}',
            'open_buffer_0945={self._buffer_display()}'
        )

    # 3. Update factor scoring in evaluate_market_regime
    anchor_factors = '''    # 1. Benchmark Index Changes (SPY / QQQ) — guaranteed available and fresh by DataHealth gate
    spy_pct = health_data["SPY"]["pct"]
    spy_source = health_data["SPY"]["source"]
    if spy_pct < -0.75:
        factor_scores["SPY"] = -2
        reasons.append(f"SPY benchmark down {spy_pct:.2f}% (< -0.75%) [{spy_source}]")
    elif spy_pct < -0.40:
        factor_scores["SPY"] = -1
        reasons.append(f"SPY benchmark weak {spy_pct:.2f}% [{spy_source}]")
    else:
        factor_scores["SPY"] = 0

    qqq_pct = health_data["QQQ"]["pct"]
    qqq_source = health_data["QQQ"]["source"]
    if qqq_pct < -1.00:
        factor_scores["QQQ"] = -2
        reasons.append(f"QQQ tech index down {qqq_pct:.2f}% (< -1.00%) [{qqq_source}]")
    elif qqq_pct < -0.50:
        factor_scores["QQQ"] = -1
        reasons.append(f"QQQ tech index weak {qqq_pct:.2f}% [{qqq_source}]")
    else:
        factor_scores["QQQ"] = 0'''

    replacement_factors = '''    # 1. Benchmark Index Changes (SPY / QQQ) — with hysteresis deadbands to prevent flapping
    spy_pct = health_data["SPY"]["pct"]
    spy_source = health_data["SPY"]["source"]
    spy_factor = _apply_spy_hysteresis(spy_pct)
    factor_scores["SPY"] = spy_factor
    if spy_factor == -2:
        reasons.append(f"SPY benchmark down {spy_pct:.2f}% (< -0.75% / hyst) [{spy_source}]")
    elif spy_factor == -1:
        reasons.append(f"SPY benchmark weak {spy_pct:.2f}% [{spy_source}]")

    qqq_pct = health_data["QQQ"]["pct"]
    qqq_source = health_data["QQQ"]["source"]
    qqq_factor = _apply_qqq_hysteresis(qqq_pct)
    factor_scores["QQQ"] = qqq_factor
    if qqq_factor == -2:
        reasons.append(f"QQQ tech index down {qqq_pct:.2f}% (< -1.00% / hyst) [{qqq_source}]")
    elif qqq_factor == -1:
        reasons.append(f"QQQ tech index weak {qqq_pct:.2f}% [{qqq_source}]")'''

    if anchor_factors not in content:
        print("[ERROR] Could not find anchor_factors in strategies/regime_engine.py")
        return False
    content = content.replace(anchor_factors, replacement_factors)

    if not dry_run:
        backup_file(target)
        target.write_text(content, encoding="utf-8")
        print(f"[PATCHED] strategies/regime_engine.py")
    else:
        print(f"[DRY-RUN] strategies/regime_engine.py would be patched cleanly.")

    return True


def patch_trailing_stop(dry_run: bool = False) -> bool:
    target = BASE_DIR / "strategies" / "trailing_stop.py"
    if not target.exists():
        print(f"[ERROR] Target not found: {target}")
        return False

    content = target.read_text(encoding="utf-8")

    if "_normalize_strategy_name" in content:
        print("[SKIP] strategies/trailing_stop.py already contains strategy attribution patch.")
        return True

    # 1. Add _normalize_strategy_name helper
    anchor_helper = 'def _classify_held_order(order) -> str:'
    helper_code = '''def _normalize_strategy_name(raw) -> str:
    """Safely extract string strategy name from enum, string, or object."""
    if hasattr(raw, "value"):
        return str(raw.value)
    if isinstance(raw, str) and raw.strip() and raw.strip() != "UNKNOWN":
        return raw.strip()
    return "UNKNOWN"


'''
    if anchor_helper not in content:
        print("[ERROR] Could not find anchor_helper in strategies/trailing_stop.py")
        return False
    content = content.replace(anchor_helper, helper_code + anchor_helper)

    # 2. Update initialize_position to normalize strategy
    anchor_init = '                strategy=getattr(signal, "strategy", "UNKNOWN"),'
    replacement_init = '''                strategy=_normalize_strategy_name(getattr(signal, "strategy", None)) if _normalize_strategy_name(getattr(signal, "strategy", None)) != "UNKNOWN" else ("DAYTRADE_SCALP" if getattr(signal, "signal_id", None) else "UNKNOWN"),'''
    if anchor_init not in content:
        print("[ERROR] Could not find anchor_init in strategies/trailing_stop.py")
        return False
    content = content.replace(anchor_init, replacement_init)

    # 3. Update line 1171 and 1535 fallback to DAYTRADE_SCALP
    content = content.replace(
        'strategy=getattr(state, "strategy", "UNKNOWN"),',
        'strategy=_normalize_strategy_name(getattr(state, "strategy", None)) if _normalize_strategy_name(getattr(state, "strategy", None)) != "UNKNOWN" else ("DAYTRADE_SCALP" if getattr(state, "signal_id", None) else "UNKNOWN"),'
    )
    content = content.replace(
        'strategy=getattr(current_state, "strategy", "UNKNOWN"),',
        'strategy=_normalize_strategy_name(getattr(current_state, "strategy", None)) if _normalize_strategy_name(getattr(current_state, "strategy", None)) != "UNKNOWN" else ("DAYTRADE_SCALP" if getattr(current_state, "signal_id", None) else "UNKNOWN"),'
    )

    # 4. In emergency_flatten(), add trade journal exit record
    anchor_flatten = '''        with self._lock:
            current_state = self.states.get(symbol)
            if current_state:
                current_state.active = False
                current_state.remediation_in_progress = False
            self._save_state()'''

    replacement_flatten = '''        with self._lock:
            current_state = self.states.get(symbol)
            if current_state:
                current_state.active = False
                current_state.remediation_in_progress = False
                if close_ok:
                    try:
                        from strategies.trade_journal import log_trade
                        raw_strat = getattr(current_state, "strategy", None)
                        strat_str = _normalize_strategy_name(raw_strat)
                        if strat_str == "UNKNOWN" and getattr(current_state, "signal_id", None):
                            strat_str = "DAYTRADE_SCALP"

                        exit_price = getattr(current_state, "stop_price", 0.0) or 0.0
                        if bridge and hasattr(bridge, "market_price"):
                            try:
                                exit_price = bridge.market_price(symbol) or exit_price
                            except Exception:
                                pass
                        if exit_price <= 0.0:
                            exit_price = getattr(current_state, "entry_price", 0.0)

                        entry_price = getattr(current_state, "entry_price", exit_price)
                        side = getattr(current_state, "side", "BUY")
                        qty_val = abs(float(position_qty or getattr(current_state, "qty", 1.0)))
                        pnl_val = (exit_price - entry_price) * qty_val if side == "BUY" else (entry_price - exit_price) * qty_val

                        log_trade(
                            symbol=symbol,
                            side="SELL" if side == "BUY" else "BUY",
                            strategy=strat_str,
                            qty=int(qty_val),
                            entry_price=entry_price,
                            stop_price=getattr(current_state, "stop_price", exit_price),
                            target_price=getattr(current_state, "target_price", None) or exit_price,
                            exit_price=exit_price,
                            exit_reason="EMERGENCY_FLATTEN",
                            pnl=pnl_val,
                            notes="Emergency flatten forced exit",
                            event_type="TRADE_CLOSED",
                            signal_id=getattr(current_state, "signal_id", None),
                        )
                        logger.info(
                            f"[EMERGENCY FLATTEN JOURNAL] {symbol} | exit_px={exit_price} | strategy={strat_str} | pnl={pnl_val:.2f}"
                        )
                    except Exception as e_log:
                        logger.error("Failed to log emergency flatten close to journal: %s", e_log)
            self._save_state()'''

    if anchor_flatten not in content:
        print("[ERROR] Could not find anchor_flatten in strategies/trailing_stop.py")
        return False
    content = content.replace(anchor_flatten, replacement_flatten)

    if not dry_run:
        backup_file(target)
        target.write_text(content, encoding="utf-8")
        print(f"[PATCHED] strategies/trailing_stop.py")
    else:
        print(f"[DRY-RUN] strategies/trailing_stop.py would be patched cleanly.")

    return True


def patch_orchestrator(dry_run: bool = False) -> bool:
    target = BASE_DIR / "strategies" / "orchestrator.py"
    if not target.exists():
        print(f"[ERROR] Target not found: {target}")
        return False

    content = target.read_text(encoding="utf-8")

    if "export_candidate_signals" in content:
        print("[SKIP] strategies/orchestrator.py already contains signal exporter patch.")
        return True

    # 1. Add export_candidate_signals definition
    anchor_helper = 'def _prepare_closed_bars(df):'
    exporter_code = '''SIGNAL_FEATURE_COLUMNS = [
    "timestamp",
    "timestamp_et",
    "symbol",
    "trigger_price",
    "entry_price",
    "stop_price",
    "target_price",
    "side",
    "confidence",
    "raw_confidence",
    "correlation_penalty",
    "max_correlation",
    "risk_per_share",
    "rr_ratio",
    "regime_score",
    "regime_mode",
    "regime_texture",
    "market_texture",
    "spy_pct",
    "qqq_pct",
    "soxx_pct",
    "sizing_pct",
    "halt_entries",
    "would_have_entered",
    "signal_type",
    "strategy",
    "signal_id",
]


def export_candidate_signals(
    signals: list,
    regime: Optional[object] = None,
    market_texture: str = "TRENDING",
    export_dir: str = "data/exports",
) -> None:
    """
    Write candidate signals to daily CSV for ML training (daytrade-train).
    Appends to today's file; creates it with headers if new.
    """
    if not signals:
        return

    try:
        from pathlib import Path
        import csv
        import pytz
        from datetime import timezone
        _ET = pytz.timezone("America/New_York")

        Path(export_dir).mkdir(parents=True, exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        today_str = datetime.now(_ET).strftime("%Y%m%d")
        filepath = os.path.join(export_dir, f"signals_{today_str}.csv")

        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0

        ts_utc = now_utc.isoformat()
        ts_et = datetime.now(_ET).strftime("%Y-%m-%d %H:%M:%S")

        reg_score = getattr(regime, "score", 0)
        reg_mode = getattr(regime, "mode", "UNKNOWN")
        reg_texture = getattr(regime, "market_texture", market_texture)
        sizing_pct = f"{getattr(regime, 'position_scale', 1.0):.2f}"
        halt_entries = getattr(regime, "halt_new_entries", False)

        reg_data = getattr(regime, "data", {}) or {}
        spy_pct = reg_data.get("SPY", {}).get("pct", "")
        qqq_pct = reg_data.get("QQQ", {}).get("pct", "")
        soxx_pct = reg_data.get("SOXX", {}).get("pct", "")

        rows = []
        for sig in signals:
            strat_val = getattr(sig, "strategy", "UNKNOWN")
            strat_str = strat_val.value if hasattr(strat_val, "value") else str(strat_val)
            meta = getattr(sig, "metadata", {}) or {}

            entry_px = float(getattr(sig, "entry_price", 0.0) or 0.0)
            stop_px = float(getattr(sig, "stop_price", 0.0) or 0.0)
            target_px = float(getattr(sig, "target_price", 0.0) or 0.0)
            dist = abs(entry_px - stop_px)
            reward = abs(target_px - entry_px)
            rr = getattr(sig, "reward_risk_ratio", None)
            if rr is None or rr <= 0:
                rr = round(reward / dist, 2) if dist > 1e-6 else 2.0

            rows.append({
                "timestamp": ts_utc,
                "timestamp_et": ts_et,
                "symbol": getattr(sig, "symbol", "").upper().strip(),
                "trigger_price": entry_px,
                "entry_price": entry_px,
                "stop_price": stop_px,
                "target_price": target_px,
                "side": getattr(sig, "side", "BUY").upper(),
                "confidence": float(getattr(sig, "confidence", 0.70)),
                "raw_confidence": float(meta.get("raw_confidence", getattr(sig, "confidence", 0.70))),
                "correlation_penalty": float(meta.get("correlation_penalty", 1.0)),
                "max_correlation": float(meta.get("max_correlation", 0.0)),
                "risk_per_share": float(getattr(sig, "risk_per_share", dist)),
                "rr_ratio": float(rr),
                "regime_score": reg_score,
                "regime_mode": reg_mode,
                "regime_texture": reg_texture,
                "market_texture": market_texture,
                "spy_pct": spy_pct,
                "qqq_pct": qqq_pct,
                "soxx_pct": soxx_pct,
                "sizing_pct": sizing_pct,
                "halt_entries": halt_entries,
                "would_have_entered": True,
                "signal_type": strat_str,
                "strategy": strat_str,
                "signal_id": getattr(sig, "signal_id", ""),
            })

        with open(filepath, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=SIGNAL_FEATURE_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        logger.debug("[SIGNAL EXPORT] Exported %d candidate signals to %s", len(rows), filepath)
    except Exception as exp_err:
        logger.debug("[SIGNAL EXPORT] Failed exporting signals: %s", exp_err)


'''
    if anchor_helper not in content:
        print("[ERROR] Could not find anchor_helper in strategies/orchestrator.py")
        return False
    content = content.replace(anchor_helper, exporter_code + anchor_helper)

    # 2. Hook into evaluate_and_execute
    anchor_hook = '    all_signals = apply_portfolio_correlation(all_signals, cached_open_positions, bridge=bridge)'
    hook_code = '''    all_signals = apply_portfolio_correlation(all_signals, cached_open_positions, bridge=bridge)

    # ── Signal Export Hook for ML Training ──
    if all_signals:
        try:
            export_candidate_signals(
                signals=all_signals,
                regime=regime_res,
                market_texture=market_tex,
                export_dir=str(Path(config.DATA_DIR) / "exports"),
            )
        except Exception as exp_err:
            logger.debug("Candidate signal export hook failed: %s", exp_err)'''

    if anchor_hook not in content:
        print("[ERROR] Could not find anchor_hook in strategies/orchestrator.py")
        return False
    content = content.replace(anchor_hook, hook_code)

    # 3. Fix semantic mode mismatch in orchestrator logging (e.g. CAUTION mode showing RISK_OFF/HALT)
    anchor_log = '''            else:
                logger.warning("[MARKET REGIME RISK_OFF/HALT] New long day trades disabled (Score: %d). Summary: %s", regime_res.score, regime_res.summary())
                inverse_exempt = getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
                if inverse_exempt and regime_res is not None:
                    logger.info("[INVERSE ETF EXEMPT] RISK_OFF active but inverse ETF entries remain eligible (SPDN/SQQQ/SH etc.)")'''

    replacement_log = '''            else:
                mode_str = getattr(regime_res, "mode", "UNKNOWN")
                logger.warning("[MARKET ENTRY HALT] New long day trades disabled (mode=%s score=%d). Summary: %s", mode_str, regime_res.score, regime_res.summary())
                inverse_exempt = getattr(config, "INVERSE_ETF_RISK_OFF_EXEMPT", False)
                if inverse_exempt and regime_res is not None:
                    logger.info("[INVERSE ETF EXEMPT] Entry halt active; eligible inverse ETFs remain enabled (mode=%s score=%d)", mode_str, regime_res.score)'''

    if anchor_log in content:
        content = content.replace(anchor_log, replacement_log)

    if not dry_run:
        backup_file(target)
        target.write_text(content, encoding="utf-8")
        print(f"[PATCHED] strategies/orchestrator.py")
    else:
        print(f"[DRY-RUN] strategies/orchestrator.py would be patched cleanly.")

    return True


def patch_test_regime_engine(dry_run: bool = False) -> bool:
    target = BASE_DIR / "tests" / "test_regime_engine.py"
    if not target.exists():
        print(f"[ERROR] Target not found: {target}")
        return False

    content = target.read_text(encoding="utf-8")

    if "test_benchmark_score_hysteresis" in content:
        print("[SKIP] tests/test_regime_engine.py already contains hysteresis unit test.")
        return True

    test_method = '''
    def test_benchmark_score_hysteresis(self):
        """
        Verify SPY/QQQ factor scores do not flap at threshold boundaries.
        Once escalated to -2, should remain -2 until recovery clears deadband.
        """
        from strategies.regime_engine import (
            reset_factor_hysteresis,
            _apply_spy_hysteresis,
            _apply_qqq_hysteresis,
        )
        reset_factor_hysteresis()

        # Drop below -0.75 -> factor -2
        self.assertEqual(_apply_spy_hysteresis(-0.76), -2)

        # Recover to -0.74% (in deadband -0.75 to -0.70) -> must STAY at -2 (no flapping)
        self.assertEqual(_apply_spy_hysteresis(-0.74), -2)

        # Recovery above -0.70 -> steps down to -1
        self.assertEqual(_apply_spy_hysteresis(-0.68), -1)

        # Drop back to -0.74% -> stays -1 until dropping below -0.75%
        self.assertEqual(_apply_spy_hysteresis(-0.74), -1)

        # Recovery above -0.35 -> steps down to 0
        self.assertEqual(_apply_spy_hysteresis(-0.30), 0)

        # QQQ Hysteresis Tests
        reset_factor_hysteresis()
        self.assertEqual(_apply_qqq_hysteresis(-1.02), -2)
        # Recover to -0.99% -> must STAY at -2
        self.assertEqual(_apply_qqq_hysteresis(-0.99), -2)
        # Recovery above -0.90 -> steps down to -1
        self.assertEqual(_apply_qqq_hysteresis(-0.88), -1)
        # Recovery above -0.45 -> steps down to 0
        self.assertEqual(_apply_qqq_hysteresis(-0.40), 0)

        # Combined oscillation simulation
        reset_factor_hysteresis()
        factors = [_apply_qqq_hysteresis(v) for v in [-1.02, -0.99, -1.04, -0.99, -1.01, -0.99]]
        self.assertTrue(all(f == -2 for f in factors), f"QQQ should remain at -2; got {factors}")
'''

    # Append before the last line of TestRegimeEngine class
    content = content.rstrip() + "\n" + test_method

    if not dry_run:
        backup_file(target)
        target.write_text(content, encoding="utf-8")
        print(f"[PATCHED] tests/test_regime_engine.py")
    else:
        print(f"[DRY-RUN] tests/test_regime_engine.py would be patched cleanly.")

    return True


def run_tests() -> None:
    print("\n[RUNNING TEST SUITE] Running pytest tests/test_regime_engine.py --no-cov...")
    cmd = [sys.executable, "-m", "pytest", "tests/test_regime_engine.py", "--no-cov"]
    res = subprocess.run(cmd, cwd=BASE_DIR)
    if res.returncode == 0:
        print("[SUCCESS] All regime engine tests PASSED!")
    else:
        print(f"[WARNING] Test suite exited with code {res.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply post-session Day Trading engine patches.")
    parser.add_argument("--dry-run", action="store_true", help="Preview patch targets without modifying files")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest verification after patching")
    args = parser.parse_args()

    print("=" * 60)
    print("  DAY TRADING ENGINE — POST-SESSION PATCH APPLIER")
    print("=" * 60)

    ok1 = patch_regime_engine(dry_run=args.dry_run)
    ok2 = patch_trailing_stop(dry_run=args.dry_run)
    ok3 = patch_orchestrator(dry_run=args.dry_run)
    ok4 = patch_test_regime_engine(dry_run=args.dry_run)

    if all([ok1, ok2, ok3, ok4]):
        print("\n[ALL PATCHES APPLIED SUCCESSFULLY]")
        if args.run_tests and not args.dry_run:
            run_tests()
    else:
        print("\n[WARNING] Some patches failed to apply. Check messages above.")


if __name__ == "__main__":
    main()
