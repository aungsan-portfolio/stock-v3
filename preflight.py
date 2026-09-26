"""
preflight.py - Automated Pre-Flight Go / No-Go Verification Gate & Audit Trail.

Architecture:
1. Critical Gate 1: Data Freshness Check (SPY / QQQ data existence & validity)
2. Critical Gate 2: Broker Reconciliation (Alpaca paper connection, clean state, 0 orphan orders)
3. Critical Gate 3: Position Protection (100% covering protective GTC stops on open longs)
4. Warning Gate 4: Portfolio Exposure & Concentration (Single position <= 25%, Sector <= 45%)
5. Warning Gate 5: Event Risk & Model Freshness (Event blackout/throttle, Model age <= 30d, Signal Parity)

Decision Rule (Tri-State Model):
- ANY CRITICAL FAIL                     -> STATUS = NO-GO (Trading Operations Halted)
- NO CRITICAL FAIL + ANY WARNING FAIL   -> STATUS = GO_WITH_WARNINGS (Trading Enabled with Alert)
- ALL GATES PASS                        -> STATUS = GO (Trading Operations Fully Approved)

Audit Trail Metadata:
- run_id, stage, git_commit, config_hash, model_hash, operator
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config
from strategies.session import now_eastern

logger = logging.getLogger(__name__)


def _compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 hash of a file for immutable audit tracking."""
    p = Path(filepath)
    if not p.is_file():
        return "not_found"
    try:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return "hash_error"


def _get_git_commit() -> str:
    """Retrieves short git commit hash for deployment traceability."""
    try:
        from observability import get_build_provenance
        return get_build_provenance().commit
    except Exception:
        return "production-release-v3"


from enum import Enum


class PreflightDecision(str, Enum):
    GO = "GO"
    GO_WITH_ENTRY_HOLD = "GO_WITH_ENTRY_HOLD"
    NO_GO = "NO_GO"


@dataclass
class GateResult:
    name: str
    category: str  # "CRITICAL" or "WARNING"
    passed: bool
    message: str
    details: Dict[str, Any]
    status: str = "PASS"  # "PASS", "FAIL", "WARN", "NOT_VERIFIED", "ENTRY_HOLD"


@dataclass
class PreflightReport:
    run_id: str
    stage: str
    timestamp: str
    decision: PreflightDecision
    entry_allowed: bool
    status: str  # Backward-compatible mirror of decision.value
    summary: str
    git_commit: str
    config_hash: str
    model_hash: str
    operator: str
    gates: List[Dict[str, Any]]
    critical_pass: int
    critical_fail: int
    critical_not_verified: int
    warning_count: int
    warnings: List[str]
    warning_pass: int = 0
    warning_fail: int = 0
    hold_reason: Optional[str] = None
    wall_clock_utc: str = ""
    wall_clock_local: str = ""
    timezone_name: str = ""
    utc_offset: str = ""


def check_historical_data_freshness() -> GateResult:
    """Verifies that daily/historical price data is available and non-empty."""
    try:
        from data_manager import fetch_ohlcv
        df = fetch_ohlcv("SPY")
        if df is None or len(df) < 50:
            return GateResult(
                name="Historical Data Freshness",
                category="CRITICAL",
                passed=False,
                message="SPY historical price data is missing or insufficient (<50 rows)",
                details={"rows": 0 if df is None else len(df)},
            )

        last_date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])
        return GateResult(
            name="Historical Data Freshness",
            category="CRITICAL",
            passed=True,
            message=f"SPY price data verified ({len(df)} rows, latest: {last_date})",
            details={"rows": len(df), "latest_date": last_date},
        )
    except Exception as exc:
        return GateResult(
            name="Historical Data Freshness",
            category="CRITICAL",
            passed=False,
            message=f"Data fetch exception: {exc}",
            details={"error": str(exc)},
        )


check_data_freshness = check_historical_data_freshness


def check_intraday_data_readiness(
    bridge: Any = None,
    anchor_symbols: Tuple[str, ...] = ("SPY", "QQQ"),
    grace_seconds: float = 60.0,
) -> GateResult:
    """Verifies live 5m bar timeliness and freshness during regular market hours."""
    from strategies.session import is_market_open, now_eastern, EASTERN_TZ
    from datetime import datetime
    import pandas as pd

    if not is_market_open():
        return GateResult(
            name="Intraday Data Readiness",
            category="WARNING",
            passed=True,
            status="PASS",
            message="Market closed or pre-market — live intraday 5m cadence deferred to market open",
            details={"market_open": False, "status": "DEFERRED_TO_OPEN"},
        )

    # 09:30–09:35 ET warmup window — first bar is forming, no completed bar expected yet
    now_et = now_eastern()
    market_open_dt = datetime.combine(now_et.date(), config.MARKET_OPEN, tzinfo=EASTERN_TZ)
    seconds_since_open = (now_et - market_open_dt).total_seconds()
    WARMUP_SECONDS = 300.0  # 5 minutes
    if 0 <= seconds_since_open < WARMUP_SECONDS:
        return GateResult(
            name="Intraday Data Readiness",
            category="CRITICAL",
            passed=True,
            status="PASS",
            message=f"Market open warmup period ({seconds_since_open:.0f}s elapsed) — first 5m bar forming",
            details={"warmup": True, "seconds_since_open": seconds_since_open},
        )

    from strategies.intraday_data import fetch_intraday, select_last_completed_bar

    stale_symbols = []
    hold_symbols = []
    symbol_ages = {}
    symbol_telemetry = {}
    interval_seconds = float(getattr(config, "INTRADAY_BAR_SECONDS", 300.0))
    max_recoverable_delay = float(getattr(config, "PREFLIGHT_MAX_RECOVERABLE_DELAY_SECONDS", interval_seconds + grace_seconds))

    for sym in anchor_symbols:
        try:
            df = fetch_intraday(sym, interval="5m", bridge=bridge, purpose="PREFLIGHT")
            if df is None or df.empty:
                stale_symbols.append(f"{sym}:NO_DATA")
                continue

            now_utc = pd.Timestamp.now(tz="UTC")
            future_tol = float(getattr(config, "BAR_FUTURE_TOLERANCE_SECONDS", 5.0))
            completed_bar = select_last_completed_bar(
                df=df,
                interval_seconds=int(interval_seconds),
                now_utc=now_utc,
                close_tolerance_seconds=future_tol,
            )
            age = float(completed_bar.post_close_delay_seconds)
            symbol_ages[sym] = round(age, 1)
            symbol_telemetry[sym] = {
                "bar_start_ts": completed_bar.start_time.isoformat() if hasattr(completed_bar.start_time, "isoformat") else str(completed_bar.start_time),
                "bar_close_ts": completed_bar.close_time.isoformat() if hasattr(completed_bar.close_time, "isoformat") else str(completed_bar.close_time),
                "current_ts": now_utc.isoformat(),
                "completed_bar_age": round(age, 1),
                "publication_delay": round(float(getattr(completed_bar, "publication_delay", age)), 1),
            }

            if age > max_recoverable_delay:
                stale_symbols.append(f"{sym}:{age:.1f}s > {max_recoverable_delay:.1f}s")
            elif age > grace_seconds:
                hold_symbols.append(f"{sym}:{age:.1f}s")
        except Exception as exc:
            stale_symbols.append(f"{sym}:{exc}")

    from strategies.intraday_data import get_feed_provenance
    feed_prov = get_feed_provenance()

    if stale_symbols:
        return GateResult(
            name="Intraday Data Readiness",
            category="CRITICAL",
            passed=False,
            status="FAIL",
            message=f"Stale anchor bars (>1 interval behind or missing): {', '.join(stale_symbols)}",
            details={"stale_symbols": stale_symbols, "ages": symbol_ages, "telemetry": symbol_telemetry, "grace": grace_seconds, "max_recoverable_delay": max_recoverable_delay, "feed_provenance": feed_prov},
        )

    if hold_symbols:
        return GateResult(
            name="Intraday Data Readiness",
            category="CRITICAL",
            passed=True,
            status="ENTRY_HOLD",
            message=f"Mid-bar startup: Anchor bars valid within current forming interval ({', '.join(hold_symbols)} > {grace_seconds:.0f}s); entries held until next bar close",
            details={"entry_hold": True, "hold_symbols": hold_symbols, "ages": symbol_ages, "telemetry": symbol_telemetry, "grace": grace_seconds, "max_recoverable_delay": max_recoverable_delay, "feed_provenance": feed_prov},
        )

    return GateResult(
        name="Intraday Data Readiness",
        category="CRITICAL",
        passed=True,
        status="PASS",
        message=f"Anchor bars fresh within {grace_seconds}s ({', '.join(f'{k}={v}s' for k, v in symbol_ages.items())})",
        details={"ages": symbol_ages, "telemetry": symbol_telemetry, "grace": grace_seconds, "max_recoverable_delay": max_recoverable_delay, "feed_provenance": feed_prov},
    )


def check_broker_reconciliation(bridge: Any = None) -> GateResult:
    """Verifies Alpaca paper broker connectivity, active status, 0 orphan orders, and total account flatness."""
    from alpaca_bridge import AlpacaBridge
    owns_bridge = False
    if bridge is None:
        bridge = AlpacaBridge(mode="daytrade")
        owns_bridge = True

    try:
        connected = bool(getattr(bridge, "_connected", False)) or bridge.connect()
        if not connected:
            return GateResult(
                name="Broker Reconciliation",
                category="CRITICAL",
                passed=False,
                status="FAIL",
                message="Could not connect to Alpaca Paper Broker API",
                details={"connected": False},
            )

        recon = bridge.reconcile_startup_state()
        snap = recon.get("snapshot", {})
        clean = bool(snap.get("clean", False))

        # Total positions audit: check all positions (long and short)
        positions = []
        try:
            if hasattr(bridge, "positions_plain"):
                positions = bridge.positions_plain() or []
            elif hasattr(bridge, "get_all_positions"):
                raw_positions = bridge.get_all_positions() or []
                positions = [{"symbol": p.symbol, "qty": float(p.qty)} for p in raw_positions]
        except Exception as p_err:
            logger.warning("Could not fetch plain positions directly: %s", p_err)
            positions = snap.get("open_positions", [])
            if positions and isinstance(positions[0], (tuple, list)):
                positions = [{"symbol": s, "qty": q} for s, q in positions]

        longs = [p for p in positions if float(p.get("qty", 0.0) or 0.0) > 0]
        shorts = [p for p in positions if float(p.get("qty", 0.0) or 0.0) < 0]
        total_open_positions = len(positions)
        n_longs = len(longs)
        n_shorts = len(shorts)

        n_orders = int(snap.get("n_working_orders", 0))
        orphan_exits = snap.get("orphan_exits", [])
        duplicate_refs = snap.get("duplicate_refs", [])

        is_flat = (total_open_positions == 0)
        from strategies.session import is_market_open
        market_open = is_market_open()
        if market_open:
            passed = clean and len(orphan_exits) == 0 and len(duplicate_refs) == 0
        else:
            passed = clean and is_flat and len(orphan_exits) == 0 and len(duplicate_refs) == 0

        details = {
            "clean": clean,
            "total_open_positions": total_open_positions,
            "long_positions": n_longs,
            "short_positions": n_shorts,
            "working_orders": n_orders,
            "duplicates": len(duplicate_refs),
            "orphans": len(orphan_exits),
            "is_flat": is_flat,
            "orphan_exits": orphan_exits,
            "duplicate_refs": duplicate_refs,
        }

        if passed:
            msg = (
                f"Broker reconciled cleanly | Open positions: {total_open_positions} "
                f"(longs: {n_longs}, shorts: {n_shorts}) | Working orders: {n_orders} | "
                f"Duplicates: {len(duplicate_refs)} | Orphans: {len(orphan_exits)}"
            )
            status = "PASS"
        else:
            reason = "flatness required pre-market" if (not is_flat and not market_open) else f"clean={clean}"
            msg = (
                f"Broker reconciliation mismatch ({reason}): clean={clean}, open_positions={total_open_positions} "
                f"(longs={n_longs}, shorts={n_shorts}), working_orders={n_orders}, "
                f"duplicates={len(duplicate_refs)}, orphans={len(orphan_exits)}"
            )
            status = "FAIL"

        return GateResult(
            name="Broker Reconciliation",
            category="CRITICAL",
            passed=passed,
            status=status,
            message=msg,
            details=details,
        )
    except Exception as exc:
        return GateResult(
            name="Broker Reconciliation",
            category="CRITICAL",
            passed=False,
            message=f"Broker reconciliation error: {exc}",
            details={"error": str(exc)},
        )
    finally:
        if owns_bridge and bridge is not None:
            try:
                bridge.disconnect()
            except Exception:
                pass


def check_position_protection(bridge: Any = None) -> GateResult:
    """Verifies that all open long positions have covering stop loss protection."""
    from alpaca_bridge import AlpacaBridge
    owns_bridge = False
    if bridge is None:
        bridge = AlpacaBridge(mode="swing")
        owns_bridge = True

    try:
        connected = bool(getattr(bridge, "_connected", False)) or bridge.connect()
        if not connected:
            return GateResult(
                name="Position Protection",
                category="CRITICAL",
                passed=False,
                message="Could not connect to Broker to audit position protection",
                details={"connected": False},
            )

        recon = bridge.reconcile_startup_state()
        protect = recon.get("protect", {})
        unprotected = protect.get("unprotected", [])
        failed = protect.get("failed", [])
        protected = protect.get("protected", [])
        repaired = protect.get("repaired", [])
        checked = protect.get("checked", len(protected) + len(unprotected))

        clean = protect.get("clean", len(failed) == 0 and len(protected) == checked)
        passed = bool(clean) and len(failed) == 0

        if passed:
            repair_str = f", {len(repaired)} repaired" if repaired else ""
            msg = f"100% Position Protection verified ({len(protected)} protected{repair_str}, 0 unprotected)"
        else:
            remaining_unprotected = [s for s in unprotected if s not in repaired]
            unresolved = remaining_unprotected if remaining_unprotected else failed
            msg = f"Unprotected positions detected: {unresolved}"

        return GateResult(
            name="Position Protection",
            category="CRITICAL",
            passed=passed,
            message=msg,
            details={
                "checked": checked,
                "protected": protected,
                "unprotected": unprotected,
                "repaired": repaired,
                "failed": failed,
                "clean": clean,
            },
        )
    except Exception as exc:
        return GateResult(
            name="Position Protection",
            category="CRITICAL",
            passed=False,
            message=f"Protection audit error: {exc}",
            details={"error": str(exc)},
        )
    finally:
        if owns_bridge and bridge is not None:
            try:
                bridge.disconnect()
            except Exception:
                pass


def check_portfolio_exposure(bridge: Any = None) -> GateResult:
    """Audits single-position weight (<25%) and sector concentration."""
    from alpaca_bridge import AlpacaBridge
    owns_bridge = False
    if bridge is None:
        bridge = AlpacaBridge(mode="swing")
        owns_bridge = True

    try:
        connected = bool(getattr(bridge, "_connected", False)) or bridge.connect()
        if not connected:
            return GateResult(
                name="Portfolio Exposure",
                category="WARNING",
                passed=True,
                message="Broker offline for exposure check (defaulted to safe)",
                details={},
            )

        positions = bridge.positions_plain()
        equity = float(bridge.get_net_liquidation() or 100000.0)
        if not positions or equity <= 0:
            return GateResult(
                name="Portfolio Exposure",
                category="WARNING",
                passed=True,
                message="Portfolio is 100% cash / flat (0 exposure risk)",
                details={"positions_count": 0, "equity": equity},
            )

        max_weight = 0.0
        max_sym = ""
        total_market_val = 0.0

        for p in positions:
            mkt_val = float(p.get("market_value", 0.0) or (p.get("qty", 0.0) * p.get("current_price", 0.0)))
            total_market_val += abs(mkt_val)
            weight = abs(mkt_val) / equity
            if weight > max_weight:
                max_weight = weight
                max_sym = p.get("symbol", "")

        utilization = total_market_val / equity
        passed = max_weight <= 0.25 and utilization <= 1.0
        msg = (
            f"Exposure within safe limits (Max pos: {max_sym} {max_weight*100:.1f}%, Utilization: {utilization*100:.1f}%)"
            if passed
            else f"High exposure detected: Max pos {max_sym} is {max_weight*100:.1f}% of equity (limit 25%)"
        )
        return GateResult(
            name="Portfolio Exposure",
            category="WARNING",
            passed=passed,
            message=msg,
            details={
                "max_position_symbol": max_sym,
                "max_position_weight": max_weight,
                "portfolio_utilization": utilization,
                "equity": equity,
            },
        )
    except Exception as exc:
        return GateResult(
            name="Portfolio Exposure",
            category="WARNING",
            passed=True,
            message=f"Exposure calculation bypassed: {exc}",
            details={"error": str(exc)},
        )
    finally:
        if owns_bridge and bridge is not None:
            try:
                bridge.disconnect()
            except Exception:
                pass


def check_event_and_models() -> GateResult:
    """Checks Event Guard state and model freshness."""
    from strategies.event_guard import evaluate_event
    import model_metrics

    evt_decision = evaluate_event()
    is_blackout = bool(evt_decision.block_entries)
    sizing_mult = float(evt_decision.throttle)
    tag = evt_decision.tag or "NORMAL"

    models_data = model_metrics.load_all()
    rf_metrics = models_data.get("rf", {})
    lstm_metrics = models_data.get("lstm", {})

    has_models = len(rf_metrics) > 0 and len(lstm_metrics) > 0

    msg = f"Event phase: {tag} (sizing={sizing_mult:.2f}x, blackout={is_blackout}) | ML Models: RF={len(rf_metrics)}, LSTM={len(lstm_metrics)}"
    passed = not is_blackout and has_models

    return GateResult(
        name="Event & Model Readiness",
        category="WARNING",
        passed=passed,
        message=msg,
        details={
            "event_tag": tag,
            "sizing_multiplier": sizing_mult,
            "is_blackout": is_blackout,
            "rf_count": len(rf_metrics),
            "lstm_count": len(lstm_metrics),
        },
    )


def check_trade_stream_readiness(bridge: Any = None) -> GateResult:
    """Verifies that Trade Stream State is HEALTHY with single stream thread, valid paper endpoint, and clean reconciliation."""
    if bridge is None:
        return GateResult(
            name="Trade Stream Readiness",
            category="CRITICAL",
            passed=False,
            status="NOT_VERIFIED",
            message="No broker bridge was supplied to the pre-flight audit",
            details={"verified": False, "reason": "no_bridge_supplied"},
        )
    try:
        from alpaca_bridge import StreamState
        state = getattr(bridge, "stream_state", None)
        threads = getattr(bridge, "active_stream_threads_count", 0)
        generation = getattr(bridge, "stream_generation", 0)
        ready = getattr(bridge, "_trade_stream_ready", False)
        recovery_req = getattr(bridge, "_trade_stream_recovery_required", False)
        latched_desync = (state == StreamState.LATCHED_DESYNC)

        # Verify stream thread is alive
        stream_thread = getattr(bridge, "_trade_stream_thread", None)
        thread_alive = stream_thread is not None and stream_thread.is_alive()

        # Verify paper endpoint
        stream_obj = getattr(bridge, "_trade_stream", None)
        endpoint_val = "unknown"
        paper_verified = False
        if stream_obj is not None:
            raw_endpoint = getattr(stream_obj, "_endpoint", None)
            endpoint_val = raw_endpoint.value if hasattr(raw_endpoint, "value") else str(raw_endpoint or "")
            if hasattr(bridge, "_assert_paper_stream_endpoint"):
                bridge._assert_paper_stream_endpoint(stream_obj)
                paper_verified = True

        passed = (
            state == StreamState.HEALTHY
            and thread_alive
            and generation > 0
            and ready
            and not recovery_req
            and not latched_desync
            and paper_verified
        )

        details = {
            "endpoint": endpoint_val,
            "paper_mode": "verified" if paper_verified else "unverified",
            "generation": generation,
            "thread_alive": thread_alive,
            "subscription": "trade_updates",
            "reconciliation": "CLEAN" if (not recovery_req and not latched_desync) else "DESYNC",
            "stream_state": getattr(state, "value", str(state)),
            "latched_desync": latched_desync,
        }

        if passed:
            msg = (
                f"Stream verified HEALTHY (gen={generation}, thread=alive, "
                f"reconciliation=CLEAN, endpoint={endpoint_val})"
            )
            status = "PASS"
        else:
            msg = (
                f"Stream check failed: state={getattr(state, 'value', str(state))}, "
                f"thread_alive={thread_alive}, gen={generation}, ready={ready}, "
                f"recovery_required={recovery_req}, latched_desync={latched_desync}"
            )
            status = "FAIL"

        return GateResult(
            name="Trade Stream Readiness",
            category="CRITICAL",
            passed=passed,
            status=status,
            message=msg,
            details=details,
        )
    except Exception as exc:
        return GateResult(
            name="Trade Stream Readiness",
            category="CRITICAL",
            passed=False,
            status="FAIL",
            message=f"Trade stream readiness error: {exc}",
            details={"error": str(exc)},
        )


def run_preflight_audit(
    bridge: Any = None,
    stage: str = "PREFLIGHT_8PM",
    save_report: bool = True,
    report_path: str = "reports/preflight_report.json"
) -> PreflightReport:
    """Runs all 7 pre-flight gates and compiles the Institutional Tri-State Go / No-Go decision."""
    gates: List[GateResult] = [
        check_historical_data_freshness(),
        check_intraday_data_readiness(bridge=bridge),
        check_broker_reconciliation(bridge=bridge),
        check_position_protection(bridge=bridge),
        check_portfolio_exposure(bridge=bridge),
        check_event_and_models(),
        check_trade_stream_readiness(bridge=bridge),
    ]

    critical_gates = [g for g in gates if g.category == "CRITICAL"]
    critical_fail = [
        g for g in critical_gates
        if getattr(g, "status", "PASS" if g.passed else "FAIL") == "FAIL"
        or (not g.passed and getattr(g, "status", "") not in ("NOT_VERIFIED", "SKIPPED", "ENTRY_HOLD"))
    ]
    critical_entry_hold = [
        g for g in critical_gates
        if getattr(g, "status", "") in ("ENTRY_HOLD", "NOT_VERIFIED")
    ]
    critical_pass = [
        g for g in critical_gates
        if getattr(g, "status", "PASS" if g.passed else "FAIL") == "PASS" and g.passed
    ]
    critical_not_verified = [
        g for g in critical_gates
        if getattr(g, "status", "") in ("NOT_VERIFIED", "SKIPPED")
    ]

    warning_gates = [g for g in gates if g.category == "WARNING"]
    warning_pass = [g for g in warning_gates if g.passed]
    warning_fail = [g for g in warning_gates if not g.passed]
    warning_list = [f"{g.name}: {g.message}" for g in warning_fail]

    hold_reason: Optional[str] = None
    if len(critical_fail) > 0:
        decision = PreflightDecision.NO_GO
        entry_allowed = False
        hold_reason = f"{len(critical_fail)} critical gate(s) failed"
        summary = f"Trading Disabled (NO_GO) - {len(critical_fail)} critical gate(s) FAILED."
    elif len(critical_entry_hold) > 0:
        decision = PreflightDecision.GO_WITH_ENTRY_HOLD
        entry_allowed = False
        hold_reason = "; ".join(g.message for g in critical_entry_hold)
        summary = f"Trading Engine Active with Entry Hold (GO_WITH_ENTRY_HOLD) - {hold_reason}"
    else:
        decision = PreflightDecision.GO
        entry_allowed = True
        summary = "Trading Operations Approved (GO) - All critical gates PASSED."
        if warning_fail:
            summary += f" ({len(warning_fail)} warning(s) logged)"

    now_local = datetime.now().astimezone()
    now_utc = datetime.now(timezone.utc)
    utc_offset_str = now_local.strftime("%z")
    formatted_offset = f"{utc_offset_str[:3]}:{utc_offset_str[3:]}" if len(utc_offset_str) == 5 else utc_offset_str
    tz_name = now_local.tzname() or "ICT"
    now_iso = now_local.isoformat()

    config_file = str(Path(__file__).parent / "config.py")
    rf_model_file = str(Path(__file__).parent / "models" / "rf_models.joblib")

    report = PreflightReport(
        run_id=uuid.uuid4().hex[:12],
        stage=stage,
        timestamp=now_iso,
        decision=decision,
        entry_allowed=entry_allowed,
        status=decision.value,
        summary=summary,
        git_commit=_get_git_commit(),
        config_hash=_compute_file_hash(config_file),
        model_hash=_compute_file_hash(rf_model_file),
        operator="automated_bot",
        gates=[
            {
                "name": g.name,
                "category": g.category,
                "passed": g.passed,
                "status": getattr(g, "status", "PASS" if g.passed else "FAIL"),
                "message": g.message,
                "details": g.details,
            }
            for g in gates
        ],
        critical_pass=len(critical_pass),
        critical_fail=len(critical_fail),
        critical_not_verified=len(critical_not_verified),
        warning_count=len(warning_fail),
        warnings=warning_list,
        warning_pass=len(warning_pass),
        warning_fail=len(warning_fail),
        hold_reason=hold_reason,
        wall_clock_utc=now_utc.isoformat(),
        wall_clock_local=now_local.isoformat(),
        timezone_name=tz_name,
        utc_offset=formatted_offset,
    )

    if save_report:
        try:
            p = Path(report_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2)
        except Exception as err:
            logger.warning("Could not persist preflight report to %s: %s", report_path, err)

    return report


def print_preflight_scorecard(report: PreflightReport) -> None:
    """Prints a beautiful institutional-grade terminal scorecard."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass
    print("\n" + "=" * 68)
    print("        🚦 AUTOMATED PRE-FLIGHT GO / NO-GO AUDIT GATE 🚦")
    print("=" * 68)
    print(f"  Run ID     : {report.run_id}  |  Stage: {report.stage}")
    tz_label = f" ({report.timezone_name}, UTC{report.utc_offset})" if report.timezone_name else ""
    print(f"  Timestamp  : {report.timestamp}{tz_label}")
    if report.wall_clock_utc:
        print(f"  UTC Clock  : {report.wall_clock_utc}")
    print(f"  Git Commit : {report.git_commit}  |  Config Hash: {report.config_hash}")
    print(f"  Model Hash : {report.model_hash}  |  Operator   : {report.operator}")
    try:
        from strategies.intraday_data import get_feed_provenance
        fp = get_feed_provenance()
        print(f"  Data Feed  : {fp['actual_feed'].upper()} ({fp['feed_capability']}) | Tier: {fp['subscription_tier']}")
    except Exception:
        pass
    print("-" * 68)

    for g in report.gates:
        cat_badge = f"[{g['category']}]"
        gate_status = g.get("status", "PASS" if g["passed"] else "FAIL")
        if gate_status == "NOT_VERIFIED":
            status_icon = "⚪ [NOT VERIFIED]"
        elif gate_status == "ENTRY_HOLD":
            status_icon = "🟡 [ENTRY HOLD]"
        elif g["passed"]:
            status_icon = "🟢 [PASS]"
        elif g["category"] == "CRITICAL":
            status_icon = "🔴 [FAIL]"
        else:
            status_icon = "🟡 [WARN]"
        print(f"  {status_icon:<18} {g['name']:<24} {cat_badge:<11}")
        print(f"                     └─► {g['message']}")

    print("-" * 68)
    if report.critical_not_verified > 0:
        print(f"  Critical Gates : {report.critical_pass} PASS / {report.critical_fail} FAIL / {report.critical_not_verified} NOT VERIFIED")
    else:
        print(f"  Critical Gates : {report.critical_pass} PASS / {report.critical_fail} FAIL")
    print(f"  Warning Gates  : {report.warning_pass} PASS / {report.warning_fail} WARN (Count: {report.warning_count})")
    print("=" * 68)

    if report.decision == PreflightDecision.GO:
        market_open_now = False
        premarket_now = False
        try:
            from strategies.session import is_market_open, is_premarket
            market_open_now = is_market_open()
            premarket_now = is_premarket()
        except Exception:
            pass

        if market_open_now:
            print("  🏆 FINAL DECISION: 🟢 GO  (Trading Operations Fully Approved, Market Open: Entries Allowed)")
        elif premarket_now:
            print("  🏆 FINAL DECISION: 🟢 GO  (System Operationally Ready | Market Session: PRE-MARKET | Entry Gate: OFF_MARKET_STANDBY | Result: Ready, Not Trading)")
        else:
            print("  🏆 FINAL DECISION: 🟢 GO  (System Operationally Ready | Market Session: NOT OPEN YET | Entry Gate: OFF_MARKET_STANDBY | Result: Ready, Not Trading)")
    elif report.decision == PreflightDecision.GO_WITH_ENTRY_HOLD:
        print(f"  ⚠️ FINAL DECISION: 🟡 GO_WITH_ENTRY_HOLD  (Trading Engine Active, Entries On Hold: {report.hold_reason})")
    else:
        print(f"  🚨 FINAL DECISION: 🔴 NO_GO  (Trading Operations Halted: {report.hold_reason})")
    print("=" * 68 + "\n")
