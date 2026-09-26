"""
tools/cross_engine_exposure_telemetry.py — Cross-Engine Aggregate Exposure Telemetry Auditor.

Implements Boundary 3 of the Ratified Cross-Engine Exposure Governance Policy (Option A):
  "Opposing exposure for the same underlying across engines must be observable in diagnostics
   and session reporting.
   Required telemetry should identify, at minimum:
   symbol, engine, account, side, quantity, timestamp.
   The existence of opposing exposure does not itself authorize or block an order."

Governance Invariants:
  • TELEMETRY ONLY: This tool is strictly read-only. Never modifies, cancels, or places orders.
  • BROKER TRUTH: Queries the live Alpaca REST API for both sub-accounts independently.
  • ZERO POLICY DRIFT: Does not enforce automated veto or modify strategy sizing.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alpaca_bridge import AlpacaBridge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("cross_engine_telemetry")


def audit_cross_engine_exposure() -> Dict[str, Any]:
    """
    Polls active positions across Daytrade and Swing Alpaca sub-accounts,
    detects opposing exposures, and logs structured telemetry.
    """
    logger.info("=" * 70)
    logger.info("  CROSS-ENGINE AGGREGATE EXPOSURE TELEMETRY AUDIT")
    logger.info("=" * 70)

    now_ts = datetime.now(timezone.utc).isoformat()

    # 1. Connect Daytrade bridge (read-only positions)
    daytrade_positions = []
    daytrade_account_id = "UNKNOWN"
    try:
        dt_bridge = AlpacaBridge(mode="daytrade")
        if dt_bridge.connect(start_stream=False):
            daytrade_account_id = getattr(dt_bridge, "_account_id", "PA3UODIC5JTA")
            daytrade_positions = dt_bridge.get_positions()
    except Exception as exc:
        logger.warning("Failed querying Daytrade account positions: %s", exc)

    # 2. Connect Swing bridge (read-only positions)
    swing_positions = []
    swing_account_id = "UNKNOWN"
    try:
        sw_bridge = AlpacaBridge(mode="swing")
        if sw_bridge.connect(start_stream=False):
            swing_account_id = getattr(sw_bridge, "_account_id", "PA3R71I7JFDZ")
            swing_positions = sw_bridge.get_positions()
    except Exception as exc:
        logger.warning("Failed querying Swing account positions: %s", exc)

    logger.info("Daytrade Account (%s) Open Positions: %d", daytrade_account_id, len(daytrade_positions))
    logger.info("Swing Account    (%s) Open Positions: %d", swing_account_id, len(swing_positions))

    # 3. Build telemetry records
    telemetry_records: List[Dict[str, Any]] = []

    for p in daytrade_positions:
        sym = str(getattr(p, "symbol", "")).upper().strip()
        qty = float(getattr(p, "position", getattr(p, "qty", 0.0)))
        side = "LONG" if qty > 0 else "SHORT"
        telemetry_records.append({
            "symbol": sym,
            "engine": "daytrade",
            "account": daytrade_account_id,
            "side": side,
            "quantity": abs(qty),
            "timestamp": now_ts,
        })

    for p in swing_positions:
        sym = str(getattr(p, "symbol", "")).upper().strip()
        qty = float(getattr(p, "position", getattr(p, "qty", 0.0)))
        side = "LONG" if qty > 0 else "SHORT"
        telemetry_records.append({
            "symbol": sym,
            "engine": "swing",
            "account": swing_account_id,
            "side": side,
            "quantity": abs(qty),
            "timestamp": now_ts,
        })

    # 4. Detect opposing exposures (same symbol, opposing sides across engines)
    opposing_exposures = []
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for rec in telemetry_records:
        by_symbol.setdefault(rec["symbol"], []).append(rec)

    for sym, recs in by_symbol.items():
        if len(recs) >= 2:
            sides = {r["side"] for r in recs}
            if len(sides) > 1:
                opposing_exposures.append({
                    "symbol": sym,
                    "exposures": recs,
                    "status": "OPPOSING_CROSS_ENGINE_TACTICAL_EXPOSURE",
                    "governance_classification": "PERMITTED_SEGREGATED_HEDGE",
                })

    # 5. Output structured telemetry
    logger.info("-" * 70)
    logger.info("Total Positions Tracked: %d across 2 engines", len(telemetry_records))
    for r in telemetry_records:
        logger.info("  [%s] %s | Engine: %-8s | Account: %s | Qty: %6.1f %s",
                    r["timestamp"], r["symbol"], r["engine"], r["account"], r["quantity"], r["side"])

    if opposing_exposures:
        logger.warning("=" * 70)
        logger.warning("  OPPOSING CROSS-ENGINE EXPOSURE DETECTED (TELEMETRY ONLY)")
        logger.warning("=" * 70)
        for opp in opposing_exposures:
            logger.warning("  • Symbol: %s | Status: %s | Governance: %s",
                           opp["symbol"], opp["status"], opp["governance_classification"])
            for e in opp["exposures"]:
                logger.warning("      Engine: %-8s | Account: %s | Side: %s | Qty: %.1f",
                               e["engine"], e["account"], e["side"], e["quantity"])
    else:
        logger.info("Opposing cross-engine exposures: NONE currently active.")

    report = {
        "timestamp": now_ts,
        "daytrade_account": daytrade_account_id,
        "swing_account": swing_account_id,
        "total_tracked_positions": len(telemetry_records),
        "positions": telemetry_records,
        "opposing_exposures": opposing_exposures,
    }

    # Persist telemetry to disk for diagnostic audit trail
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    out_file = log_dir / "cross_engine_exposure_telemetry.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Telemetry report persisted to: %s", out_file)

    return report


if __name__ == "__main__":
    audit_cross_engine_exposure()
