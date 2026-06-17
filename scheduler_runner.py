"""
scheduler_runner.py - Phase 5B-1: supervised scheduler / market-hours runner (Path A).

See reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md (task 5.1). The Path A bot is
ONE-SHOT: every command does connect -> execute -> disconnect and exits; there is
no long-running loop. This module is the THIN orchestration layer that decides
whether a *scheduled* invocation (e.g. from Windows Task Scheduler) should run
right now, and if so dispatches the EXISTING one-shot paper command. It is the
H17 answer to "run_dryrun.bat just dry-runs and crashes outside hours".

Design rules (every one is enforced and unit-tested in test_scheduler_runner.py):
  * NO daemon / NO loop. `run_scheduled()` evaluates the window exactly ONCE and
    dispatches at most once, then returns. Recurrence is the OS scheduler's job.
  * NO live trading, EVER. It ships in plan/dry-run mode; even `--execute` only
    forwards to the *paper* order path, and only when SCHEDULER_DRY_RUN_DEFAULT is
    deliberately flipped False. The bot stays paper-port-locked throughout.
  * NO duplicated trading logic. The dispatcher calls `main.cmd_paper`, so every
    existing gate still runs inside that command: model gate, data-integrity gate,
    market-hours gate, daily-loss kill-switch, startup reconciliation, paper-port
    lock. This layer only ADDS a pre-gate ("should a scheduled run start?").
  * NO new capability flag. Phase 5B does not advertise a live-readiness
    capability; this module intentionally exposes no SUPPORTS_* attribute.

Like data_integrity.py / reconciliation.py the decision half is PURE and offline:
`evaluate(now_et, ...)` takes a US/Eastern datetime and config and returns a
`ScheduleDecision`. It never reads the wall clock itself (the caller resolves ET
via `now_eastern()`), so it is fully deterministic in tests.
"""
from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import config
import data_integrity
import order_audit

logger = logging.getLogger(__name__)

# ── Decision reason codes ────────────────────────────────────────────────────
# A blocked decision carries one of these (or, for an RTH block, the
# data_integrity reason: "weekend" / "holiday" / "before_open" / "after_close").
REASON_ALLOWED = "allowed"
REASON_DISABLED = "scheduler_disabled"
REASON_PAPER_SAFETY = "paper_safety"
REASON_CLOCK_ERROR = "clock_error"
REASON_OUTSIDE_RTH = "outside_rth"


@dataclass
class ScheduleDecision:
    """Verdict on whether a scheduled run may start now. Describes only -- it
    never connects or places an order. ``allowed`` is True only when the bot is
    paper-safe, the scheduler is enabled, the clock resolved, and (if required)
    the market is in regular hours."""
    allowed: bool
    reason: str
    detail: str
    now_iso: str = ""
    rth_reason: str = ""
    paper_safe: bool = True
    enabled: bool = True
    require_rth: bool = True
    violations: List[str] = field(default_factory=list)


# ── Config resolution (read at call time so tests can patch config) ──────────
def scheduler_enabled() -> bool:
    return bool(getattr(config, "SCHEDULER_ENABLED", True))


def scheduler_require_rth() -> bool:
    return bool(getattr(config, "SCHEDULER_REQUIRE_RTH", True))


def scheduler_dry_run_default() -> bool:
    return bool(getattr(config, "SCHEDULER_DRY_RUN_DEFAULT", True))


# ── Paper-only safety (defense in depth) ─────────────────────────────────────
def paper_safety_violations() -> List[str]:
    """Return a list of reasons the current config is NOT unambiguously paper-only
    (empty list == safe). Mirrors trade_coach.assert_paper_trading_only so the
    scheduler refuses BEFORE doing anything if a live-trading config slipped in.
    This is a pre-gate; IBKRBridge.connect() still enforces the paper-port lock."""
    v: List[str] = []
    if bool(getattr(config, "COACH_LIVE_TRADING_ENABLED", False)):
        v.append("COACH_LIVE_TRADING_ENABLED is True")
    if not bool(getattr(config, "REQUIRE_PAPER_PORT", True)):
        v.append("REQUIRE_PAPER_PORT is False")
    paper_port = int(getattr(config, "PAPER_IBKR_PORT", 7497))
    if int(getattr(config, "IBKR_PORT", paper_port)) != paper_port:
        v.append(f"IBKR_PORT {getattr(config, 'IBKR_PORT', None)} != paper port {paper_port}")
    if bool(getattr(config, "ALLOW_SHORT", False)):
        v.append("ALLOW_SHORT is True")
    if bool(getattr(config, "ALLOW_HISTORICAL_PRICE_FOR_ORDERS", False)):
        v.append("ALLOW_HISTORICAL_PRICE_FOR_ORDERS is True")
    return v


def now_eastern() -> Optional[_dt.datetime]:
    """Current time in US/Eastern, or None when the timezone cannot be resolved.

    Isolated here (and never called by the pure ``evaluate``) so tests stay
    deterministic. A None return makes ``evaluate`` fail CLOSED (block), so a
    clock/zoneinfo failure can never let a scheduled run slip through at an
    unknown time -- the same fail-closed posture as ibkr_bridge._market_hours_ok.
    """
    try:
        from zoneinfo import ZoneInfo
        return _dt.datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        logger.error("Scheduler: could not resolve US/Eastern time -> failing closed",
                     exc_info=True)
        return None


# ── Pure decision ────────────────────────────────────────────────────────────
def evaluate(now_et: Optional[_dt.datetime], *,
             enabled: Optional[bool] = None,
             require_rth: Optional[bool] = None) -> ScheduleDecision:
    """Decide whether a scheduled run may start. PURE given ``now_et``.

    Order of checks (most-protective first, all fail-closed):
      1. paper-only safety  -- a live-trading config blocks unconditionally.
      2. scheduler enabled  -- config.SCHEDULER_ENABLED.
      3. clock resolved     -- ``now_et`` is None => block (fail-closed).
      4. regular hours      -- when required, must be inside RTH on a trading day.
    """
    if enabled is None:
        enabled = scheduler_enabled()
    if require_rth is None:
        require_rth = scheduler_require_rth()

    violations = paper_safety_violations()
    paper_safe = not violations
    now_iso = now_et.isoformat(timespec="seconds") if now_et is not None else ""
    rth_reason = ""
    if now_et is not None:
        try:
            rth_reason = data_integrity.regular_hours_reason(now_et)
        except Exception:
            rth_reason = ""

    def decision(allowed, reason, detail):
        return ScheduleDecision(
            allowed=allowed, reason=reason, detail=detail, now_iso=now_iso,
            rth_reason=rth_reason, paper_safe=paper_safe, enabled=bool(enabled),
            require_rth=bool(require_rth), violations=violations,
        )

    # 1) paper-only safety -- highest priority, never bypassed.
    if not paper_safe:
        return decision(False, REASON_PAPER_SAFETY,
                        "paper-only safety violated: " + "; ".join(violations))
    # 2) master switch.
    if not enabled:
        return decision(False, REASON_DISABLED,
                        "scheduler disabled (config.SCHEDULER_ENABLED=False)")
    # 3) clock -- fail closed when ET cannot be resolved.
    if now_et is None:
        return decision(False, REASON_CLOCK_ERROR,
                        "could not resolve US/Eastern time -> blocked (fail-closed)")
    # 4) regular-hours gate.
    if require_rth and not data_integrity.is_regular_hours(now_et):
        return decision(False, rth_reason or REASON_OUTSIDE_RTH,
                        f"outside US regular trading hours ({rth_reason or 'closed'})")

    where = rth_reason or "rth_not_required"
    return decision(True, REASON_ALLOWED, f"inside scheduled run window ({where})")


def summary_line(d: ScheduleDecision) -> str:
    """One-line human-readable summary of a decision for console / logs."""
    return (
        "scheduler: allowed={allowed} reason={reason} now_et={now} "
        "rth={rth} enabled={en} require_rth={rrth} paper_safe={ps}"
    ).format(
        allowed=d.allowed, reason=d.reason, now=d.now_iso or "n/a",
        rth=d.rth_reason or "n/a", en=d.enabled, rrth=d.require_rth, ps=d.paper_safe,
    )


# ── Dispatch ─────────────────────────────────────────────────────────────────
def _default_dispatch(dry_run: bool) -> int:
    """Run the existing one-shot ``paper`` command (preview when dry_run=True).

    Lazy-imports main to avoid a circular import (main imports this module). This
    is the ONLY trading entry point the scheduler uses -- it adds no trading
    logic of its own. cmd_paper carries every existing gate; with dry_run=True it
    predicts + prints and never connects to IBKR, so it places nothing.
    """
    import argparse as _argparse
    import main
    ns = _argparse.Namespace(dry_run=bool(dry_run))
    return int(main.cmd_paper(ns))


def _audit_decision(decision: ScheduleDecision, *, execute: bool,
                    effective_dry_run: bool) -> None:
    """Record the scheduler decision to the order audit log (req 10). Never raises
    (order_audit.log_event is best-effort)."""
    order_audit.log_event(
        order_audit.STAGE_SCHEDULE,
        decision="allowed" if decision.allowed else "blocked",
        reason=decision.reason,
        detail=decision.detail,
        now_et=decision.now_iso,
        rth_reason=decision.rth_reason,
        require_rth=decision.require_rth,
        enabled=decision.enabled,
        paper_safe=decision.paper_safe,
        violations=decision.violations,
        execute=bool(execute),
        dry_run=bool(effective_dry_run),
    )


def run_scheduled(*, now_et: Optional[_dt.datetime] = None,
                  execute: bool = False,
                  dry_run: Optional[bool] = None,
                  enabled: Optional[bool] = None,
                  require_rth: Optional[bool] = None,
                  dispatch: Optional[Callable[[bool], int]] = None,
                  resolve_now: bool = True) -> dict:
    """Decide whether a scheduled run is allowed and, if so, dispatch ONCE.

    Single-shot by contract: evaluates the window once and calls ``dispatch`` at
    most once. It NEVER loops, NEVER daemonizes, NEVER places a live order, and
    NEVER bypasses a downstream gate.

    Effective dry-run resolution (most-conservative wins):
      * explicit ``dry_run=`` (tests / callers) is honoured verbatim;
      * else, without ``execute`` the run is ALWAYS a dry-run/plan (no orders);
      * else (``execute`` set) it follows config.SCHEDULER_DRY_RUN_DEFAULT -- which
        ships True, so even ``--execute`` stays a preview until an operator
        deliberately flips that config False.

    Returns a dict: {"ran": bool, "exit_code": int, "dry_run": bool,
    "decision": ScheduleDecision}. ``exit_code`` is 0 for a cleanly blocked run
    (market closed / disabled is a normal no-op, not a failure) and otherwise the
    underlying command's exit code.
    """
    if now_et is None and resolve_now:
        now_et = now_eastern()

    decision = evaluate(now_et, enabled=enabled, require_rth=require_rth)

    dry_run_default = scheduler_dry_run_default()
    if dry_run is not None:
        effective_dry_run = bool(dry_run)
    elif execute:
        effective_dry_run = bool(dry_run_default)
    else:
        effective_dry_run = True

    _audit_decision(decision, execute=execute, effective_dry_run=effective_dry_run)

    if not decision.allowed:
        logger.info("Scheduler: run BLOCKED (%s) - %s", decision.reason, decision.detail)
        return {"ran": False, "exit_code": 0,
                "dry_run": effective_dry_run, "decision": decision}

    # Allowed: belt-and-suspenders paper-only re-assert through the canonical
    # guard before any dispatch (evaluate already checked; this is defense in
    # depth and keeps the scheduler honest if config changed between the calls).
    try:
        import trade_coach
        trade_coach.assert_paper_trading_only()
    except Exception as exc:
        logger.error("Scheduler: paper-only assertion failed at dispatch -> abort: %s", exc)
        order_audit.log_event(order_audit.STAGE_SCHEDULE,
                              decision="abort_paper_safety", error=str(exc))
        return {"ran": False, "exit_code": 0,
                "dry_run": effective_dry_run, "decision": decision}

    runner = dispatch or _default_dispatch
    logger.info("Scheduler: run ALLOWED -> dispatch (dry_run=%s, execute=%s)",
                effective_dry_run, execute)
    rc = int(runner(effective_dry_run))
    order_audit.log_event(
        order_audit.STAGE_SCHEDULE, decision="dispatched",
        dry_run=bool(effective_dry_run), execute=bool(execute),
        exit_code=rc, now_et=decision.now_iso,
    )
    return {"ran": True, "exit_code": rc,
            "dry_run": effective_dry_run, "decision": decision}
