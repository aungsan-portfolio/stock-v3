"""
tools/repair_swing_risk_store.py

Audit, backup, and reconstruct `data/swing_position_risk.json` using authoritative
broker state (via AlpacaBridge) or the paper trading journal.
Removes mock/test data contamination (e.g. entry_price=100.0 for MSFT/AAPL, SYM1, SYM2).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategies.swing_risk_store import (
    DEFAULT_RISK_STORE_PATH,
    PositionRiskBasis,
    load_risk_store,
    save_risk_basis,
    reconstruct_risk_basis_from_broker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("repair_risk_store")


def backup_risk_store(store_path: Path) -> Path:
    """Create a timestamped backup of the current risk store."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = store_path.with_name(f"{store_path.name}.bak.{ts}")
    if store_path.exists():
        shutil.copy2(store_path, backup_path)
        logger.info("[BACKUP] Created backup at %s", backup_path)
    return backup_path


def audit_current_store(store_path: Path) -> dict:
    """Scan existing store entries and flag obvious mock/polluted entries."""
    store = load_risk_store(store_path)
    flagged = []
    clean = []

    for sym, basis in store.items():
        is_mock = False
        reasons = []

        if "SYM" in sym:
            is_mock = True
            reasons.append("Mock symbol format (SYM*)")
        if basis.entry_price == 100.0 and basis.initial_stop_price == 97.0:
            is_mock = True
            reasons.append("Default test fixture values (entry=100.0, stop=97.0)")
        if basis.shares in (99, 100) and basis.entry_price == 100.0:
            is_mock = True
            reasons.append(f"Test fixture sizing ({basis.shares} shares @ $100)")

        item = {
            "symbol": sym,
            "entry_price": basis.entry_price,
            "stop_price": basis.initial_stop_price,
            "risk_per_share": basis.initial_risk_per_share,
            "shares": basis.shares,
            "basis_source": basis.basis_source,
            "timestamp": basis.entry_timestamp_utc,
            "is_mock": is_mock,
            "reasons": reasons,
        }

        if is_mock:
            flagged.append(item)
        else:
            clean.append(item)

    return {"flagged": flagged, "clean": clean, "total": len(store)}


def repair_store_from_broker(store_path: Path, dry_run: bool = True) -> bool:
    """
    Connect to Alpaca Paper account, inspect live open positions and stops,
    and atomically rebuild swing_position_risk.json.
    """
    logger.info("Connecting to AlpacaBridge (mode='swing') to inspect live broker truth...")
    try:
        from alpaca_bridge import AlpacaBridge
        bridge = AlpacaBridge(mode="swing", dry_run=False)
        if not getattr(bridge, "_connected", False):
            bridge.connect()
        broker_positions = bridge._client.get_all_positions()
    except Exception as exc:
        logger.error("Failed to connect to Alpaca or fetch positions: %s", exc)
        return False

    existing_store = load_risk_store(store_path)
    reconstructed = {}
    now_iso = datetime.now(timezone.utc).isoformat()

    logger.info("Found %d open positions at broker.", len(broker_positions))
    for raw in broker_positions:
        sym = raw.symbol.upper().strip()
        qty = int(float(raw.qty))
        avg_entry = float(raw.avg_entry_price)
        curr_price = float(raw.current_price)

        # If existing record already has valid non-mock entry, preserve its true initial risk
        existing_basis = existing_store.get(sym)
        if (
            existing_basis
            and existing_basis.entry_price > 105.0  # Not the $100 mock
            and existing_basis.initial_risk_per_share > 0.50
            and abs(existing_basis.entry_price - avg_entry) < 2.0  # Matches broker entry
        ):
            reconstructed[sym] = existing_basis
            u_r = (curr_price - existing_basis.entry_price) / existing_basis.initial_risk_per_share
            logger.info(
                "[PRESERVE EXISTING CLEAN] %s: entry=$%.2f initial_risk=$%.2f -> unrealized_r=%.2fR",
                sym, existing_basis.entry_price, existing_basis.initial_risk_per_share, u_r
            )
            continue

        # Query broker for active resting stop order
        curr_stop = bridge.get_current_stop(sym)
        # If stop has already ratcheted above or equal to entry (in-profit lock)
        if curr_stop is not None and curr_stop >= avg_entry:
            initial_stop_proxy = round(avg_entry * 0.975, 2)  # ~2.5% initial risk proxy
            logger.info(
                "[%s] Resting stop ($%.2f) has ratcheted into profit (> entry $%.2f). Using initial risk proxy stop: $%.2f",
                sym, curr_stop, avg_entry, initial_stop_proxy
            )
            stop_for_basis = initial_stop_proxy
        elif curr_stop is not None and curr_stop > 0.0:
            stop_for_basis = curr_stop
        else:
            stop_for_basis = round(avg_entry * 0.975, 2)
            logger.warning(
                "[%s] No active resting stop below entry; using 2.5%% proxy stop: $%.2f",
                sym, stop_for_basis
            )

        basis = reconstruct_risk_basis_from_broker(
            symbol=sym,
            avg_entry_price=avg_entry,
            current_stop_price=stop_for_basis,
            shares=qty,
            entry_timestamp_utc=now_iso,
        )
        reconstructed[sym] = basis
        unrealized_r = (curr_price - avg_entry) / max(0.01, (avg_entry - stop_for_basis))
        logger.info(
            "[RECONSTRUCT] %s: shares=%d entry=$%.2f stop=$%.2f curr=$%.2f -> unrealized_r=%.2fR",
            sym, qty, avg_entry, stop_for_basis, curr_price, unrealized_r
        )

    if dry_run:
        logger.info("[DRY-RUN] Would write %d clean records to %s. No files modified.", len(reconstructed), store_path)
        return True

    backup_risk_store(store_path)

    # Save reconstructed records
    store_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = store_path.with_suffix(".tmp")
    from dataclasses import asdict
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump({sym: asdict(b) for sym, b in reconstructed.items()}, f, indent=2)
    temp_path.replace(store_path)

    logger.info("[SUCCESS] Atomically updated %s with %d authoritative positions.", store_path, len(reconstructed))
    return True


def main():
    parser = argparse.ArgumentParser(description="Audit and repair swing_position_risk.json")
    parser.add_argument("--path", type=str, default=str(DEFAULT_RISK_STORE_PATH), help="Path to risk store")
    parser.add_argument("--apply", action="store_true", help="Apply repairs (default is dry-run audit only)")
    args = parser.parse_args()

    store_path = Path(args.path)
    logger.info("=== STEP 1: AUDITING EXISTING STORE [%s] ===", store_path)
    audit = audit_current_store(store_path)

    logger.info("Total entries: %d", audit["total"])
    logger.info("Clean entries: %d", len(audit["clean"]))
    for item in audit["clean"]:
        logger.info("  [CLEAN] %s: entry=$%.2f shares=%d stop=$%.2f (%s)",
                    item["symbol"], item["entry_price"], item["shares"], item["stop_price"], item["basis_source"])

    logger.warning("Flagged mock/polluted entries: %d", len(audit["flagged"]))
    for item in audit["flagged"]:
        logger.warning("  [POLLUTED] %s: entry=$%.2f shares=%d stop=$%.2f | Reasons: %s",
                       item["symbol"], item["entry_price"], item["shares"], item["stop_price"], ", ".join(item["reasons"]))

    if not args.apply:
        logger.info("=== AUDIT COMPLETE. Run with `--apply` to pull live broker positions and fix ===")
        return

    logger.info("=== STEP 2: RECONSTRUCTING FROM BROKER SOURCE OF TRUTH ===")
    repair_store_from_broker(store_path, dry_run=False)


if __name__ == "__main__":
    main()
