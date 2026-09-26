"""
gate_ledger.py -- Append-only Structured Gate Attribution Ledger.
Logs every candidate evaluation decision, gate transition, and terminal rejection
to data/trade_gate_decision.jsonl for complete mathematical traceability.
"""
import os
import json
import threading
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from strategies.veto_reasons import VetoReason

logger = logging.getLogger(__name__)

_ledger_lock = threading.Lock()
DEFAULT_LEDGER_PATH = os.path.join("data", "trade_gate_decision.jsonl")

# Canonical single source of truth for the engine's sequential gating hierarchy
CANONICAL_PIPELINE_STAGES = (
    ("STAGE_COVERAGE", "1. Coverage Guard (95%)"),
    ("STAGE_REGIME", "2. Market Regime & Halts"),
    ("STAGE_LUNCH_CHOP", "3. Midday Lunch Chop Zone"),
    ("STAGE_ACTIVE_POSITION", "4. Active Position Filter"),
    ("STAGE_PENDING_ORDER", "5. Pending Order Filter"),
    ("STAGE_CONFIDENCE", "6. Confidence Floor"),
    ("STAGE_RSI_FILTER", "7. RSI Overbought/Oversold"),
    ("STAGE_GEOMETRY", "8. Pre-Execution Geometry"),
    ("STAGE_ENTRY_READINESS", "9. Entry Data Readiness"),
    ("STAGE_COOLDOWN", "10. Session Skip Cooldown"),
    ("STAGE_QUOTE_HEALTH", "11. Real-Time Quote Health"),
    ("STAGE_NOISE_FLOOR", "12. Noise Floor Hard Veto"),
    ("STAGE_MAX_CHASE", "13. Directional Chase Guard"),
    ("STAGE_SPY_GUARD", "14. SPY 5m Benchmark Guard"),
    ("STAGE_COUNTERTREND_GUARD", "15. Counter-Trend Short Guard"),
    ("STAGE_SOXX_GUARD", "16. SOXX Tech/Semi Gate"),
    ("STAGE_SECTOR_CLUSTER_CAP", "17. Sector Concentration Cluster Cap"),
    ("STAGE_POSITION_SIZE", "18. Position Sizing & Capital"),
    ("STAGE_ORDER_EXECUTION", "19. Order Execution Gate"),
    ("STAGE_ORDER_SUBMIT", "20. Order Submitted"),
    ("STAGE_FILLED", "21. Broker Filled"),
)


def log_gate_decision(
    evaluation_id: str,
    symbol: str,
    bar: str,
    stage: str,
    decision: str,
    reason_code: Optional[Any] = None,
    human_message: str = "",
    strategy: str = "",
    terminal_decision: bool = False,
    blocked_by: Optional[str] = None,
    raw_conf: Optional[float] = None,
    final_conf: Optional[float] = None,
    required_conf: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    filepath: str = DEFAULT_LEDGER_PATH,
) -> dict:
    """Record a structured gate decision event into the append-only JSONL ledger."""
    code_str = reason_code.value if isinstance(reason_code, VetoReason) else str(reason_code or "")

    record = {
        "evaluation_id": str(evaluation_id),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": str(symbol).upper().strip(),
        "bar": str(bar or ""),
        "strategy": str(strategy or ""),
        "stage": str(stage),
        "decision": str(decision).upper(),
        "reason_code": code_str,
        "human_message": str(human_message),
        "terminal_decision": bool(terminal_decision),
        "blocked_by": str(blocked_by or "") if blocked_by else None,
        "raw_conf": round(float(raw_conf), 4) if raw_conf is not None else None,
        "final_conf": round(float(final_conf), 4) if final_conf is not None else None,
        "required_conf": round(float(required_conf), 4) if required_conf is not None else None,
        "metadata": metadata or {},
    }

    try:
        dir_name = os.path.dirname(filepath)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        line = json.dumps(record, default=str) + "\n"
        with _ledger_lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception as e:
        logger.warning("[GATE LEDGER ERROR] Failed to write gate decision: %s", e)

    return record


def summarize_gate_decisions(date_str: Optional[str] = None, filepath: str = DEFAULT_LEDGER_PATH) -> dict:
    """Summarizes first-order decisive blockers from the ledger."""
    summary = {
        "total_evaluations": 0,
        "terminal_rejections": 0,
        "terminal_passes": 0,
        "blocker_counts": {},
        "reason_code_counts": {},
    }

    if not os.path.exists(filepath):
        return summary

    with _ledger_lock:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ts = record.get("timestamp", "")
                    if date_str and not ts.startswith(date_str):
                        continue

                    summary["total_evaluations"] += 1

                    if record.get("terminal_decision"):
                        dec = record.get("decision")
                        if dec in ("REJECT", "HALT", "VETO"):
                            summary["terminal_rejections"] += 1
                            stage = record.get("blocked_by") or record.get("stage") or "UNKNOWN"
                            summary["blocker_counts"][stage] = summary["blocker_counts"].get(stage, 0) + 1

                            code = record.get("reason_code")
                            if code:
                                summary["reason_code_counts"][code] = summary["reason_code_counts"].get(code, 0) + 1
                        elif dec in ("PASS", "ACCEPT", "SUBMIT"):
                            summary["terminal_passes"] += 1
        except Exception as e:
            logger.warning("[GATE LEDGER ERROR] Failed to summarize decisions: %s", e)

    return summary


def dump_effective_config_startup(logger_inst=None):
    """Prints and logs the complete block of effective daytrade settings at engine boot."""
    import config
    eff = config.get_effective_daytrade_config() if hasattr(config, "get_effective_daytrade_config") else {}
    log_func = logger_inst.info if logger_inst else logger.info

    header = "=" * 65
    log_func(header)
    log_func(" AUTHORITATIVE EFFECTIVE DAYTRADE RUNTIME CONFIGURATION")
    log_func(header)
    for k, v in eff.items():
        log_func(f"  [CONFIG] {k:<38} = {v}")
    log_func(header)
    return eff
