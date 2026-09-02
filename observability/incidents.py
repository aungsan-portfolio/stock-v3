"""
observability/incidents.py -- Incident bundle generation and snapshot packaging.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import platform
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from observability.flight_recorder import get_flight_recorder

logger = logging.getLogger(__name__)

BUNDLE_VERSION = 1
OBSERVABILITY_VERSION = 1


class IncidentBuilder:
    """
    Constructs a 5-file Incident Bundle for root-cause forensic analysis.
    Implements a strict NEVER-RAISE contract: failures during bundle generation
    are logged but never interrupt or crash the calling protection/trading path.
    """

    def __init__(
        self,
        incident_base_dir: Optional[str] = None,
        broker_events_path: Optional[str] = None,
    ) -> None:
        self.incident_base_dir = Path(incident_base_dir or getattr(config, "INCIDENT_DIR", "logs/incidents"))
        self.broker_events_path = Path(broker_events_path or getattr(config, "BROKER_EVENTS_PATH", "logs/broker_events.jsonl"))

    def create_incident_dir(self, run_id: str, ts: Optional[datetime] = None) -> Path:
        ts = ts or datetime.now(timezone.utc).astimezone()
        ts_str = ts.strftime("%Y%m%d_%H%M%S")
        clean_run_id = (run_id or "run")[:8]
        dir_name = f"{clean_run_id}_{ts_str}"
        incident_dir = self.incident_base_dir / dir_name
        incident_dir.mkdir(parents=True, exist_ok=True)
        return incident_dir

    def get_git_commit(self) -> str:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception:
            pass
        return getattr(config, "GIT_COMMIT", "unknown")

    def get_config_hash(self) -> str:
        try:
            import hashlib
            config_file = Path(getattr(config, "__file__", "config.py"))
            if config_file.exists():
                return hashlib.sha256(config_file.read_bytes()).hexdigest()[:16]
        except Exception:
            pass
        return "unknown"

    def get_model_hash(self) -> str:
        try:
            import hashlib
            model_path = Path(getattr(config, "SAVED_MODELS_DIR", "saved_models"))
            if model_path.exists():
                all_bytes = b"".join(f.read_bytes() for f in sorted(model_path.glob("*.pkl"))[:5])
                if all_bytes:
                    return hashlib.sha256(all_bytes).hexdigest()[:16]
        except Exception:
            pass
        return "unknown"

    def dump_account_state(self, bridge: Optional[object], target_file: Path) -> None:
        """Capture broker account state with short timeout and fallback."""
        account_data: Dict[str, Any] = {
            "captured_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "source": "broker" if bridge is not None else "unavailable",
            "equity": None,
            "cash": None,
            "buying_power": None,
            "positions": [],
            "working_orders": [],
        }

        if bridge is not None:
            try:
                # 1. Equity / Balance
                if hasattr(bridge, "get_account"):
                    acc = bridge.get_account()
                    if acc:
                        account_data["equity"] = float(getattr(acc, "equity", 0.0) or 0.0)
                        account_data["cash"] = float(getattr(acc, "cash", 0.0) or 0.0)
                        account_data["buying_power"] = float(getattr(acc, "buying_power", 0.0) or 0.0)
                elif hasattr(bridge, "get_account_balance"):
                    account_data["equity"] = bridge.get_account_balance()

                # 2. Positions
                if hasattr(bridge, "get_positions"):
                    raw_pos = bridge.get_positions() or []
                    for p in raw_pos:
                        account_data["positions"].append({
                            "symbol": getattr(p, "symbol", str(p)),
                            "qty": float(getattr(p, "qty", 0.0) or 0.0),
                            "avg_entry_price": float(getattr(p, "avg_entry_price", 0.0) or 0.0),
                            "market_value": float(getattr(p, "market_value", 0.0) or 0.0),
                            "unrealized_pl": float(getattr(p, "unrealized_pl", 0.0) or 0.0),
                        })

                # 3. Working Orders
                if hasattr(bridge, "get_working_orders"):
                    raw_orders = bridge.get_working_orders() or []
                    for o in raw_orders:
                        account_data["working_orders"].append({
                            "id": str(getattr(o, "id", "")),
                            "client_order_id": str(getattr(o, "client_order_id", "")),
                            "symbol": getattr(o, "symbol", ""),
                            "side": str(getattr(o, "side", "")),
                            "qty": float(getattr(o, "qty", 0.0) or 0.0),
                            "type": str(getattr(o, "type", "")),
                            "status": str(getattr(o, "status", "")),
                        })
            except Exception as broker_exc:
                logger.warning("[INCIDENT BUILDER] Account state query had partial failure: %s", broker_exc)
                account_data["error"] = str(broker_exc)

        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(account_data, f, indent=2)

    def dump_manifest(self, run_id: str, target_file: Path) -> None:
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "observability_version": OBSERVABILITY_VERSION,
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "git_commit": self.get_git_commit(),
            "config_hash": self.get_config_hash(),
            "model_hash": self.get_model_hash(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "argv": sys.argv,
        }
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def slice_broker_events(self, target_file: Path, window_minutes: int = 10) -> None:
        """Slice broker events log within +/- window_minutes."""
        sliced_events: List[str] = []
        if self.broker_events_path.exists():
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
            try:
                with open(self.broker_events_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        try:
                            parsed = json.loads(line_str)
                            ts_str = parsed.get("ts") or parsed.get("timestamp")
                            if ts_str:
                                dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                if dt.tzinfo is None:
                                    dt = dt.replace(tzinfo=timezone.utc)
                                if dt >= cutoff:
                                    sliced_events.append(line_str)
                            else:
                                sliced_events.append(line_str)
                        except Exception:
                            sliced_events.append(line_str)
            except Exception as slice_exc:
                logger.warning("[INCIDENT BUILDER] Failed reading broker events: %s", slice_exc)

        with open(target_file, "w", encoding="utf-8") as f:
            for ev_line in sliced_events[-200:]:  # Keep recent 200 events within window
                f.write(ev_line + "\n")

    def dump_bundle(
        self,
        *,
        latch_event: Dict[str, Any],
        bridge: Optional[object] = None,
    ) -> Optional[Path]:
        """
        Dumps the full 5-file Incident Bundle.
        NEVER-RAISE CONTRACT: Guaranteed never to raise exceptions.
        """
        try:
            run_id = latch_event.get("run_id", "unknown_run")
            incident_dir = self.create_incident_dir(run_id)
            latch_event["incident_dir"] = str(incident_dir)

            # 1. latch_event.json
            with open(incident_dir / "latch_event.json", "w", encoding="utf-8") as f:
                json.dump(latch_event, f, indent=2)

            # 2. snapshot.jsonl (Flight Recorder Ring Buffer Dump)
            flight_rec = get_flight_recorder()
            flight_rec.dump(str(incident_dir / "snapshot.jsonl"))

            # 3. account_state.json
            self.dump_account_state(bridge, incident_dir / "account_state.json")

            # 4. broker_events.jsonl
            self.slice_broker_events(incident_dir / "broker_events.jsonl")

            # 5. manifest.json
            self.dump_manifest(run_id, incident_dir / "manifest.json")

            logger.info("[INCIDENT BUILDER] Successfully generated Incident Bundle at: %s", incident_dir)
            return incident_dir
        except Exception as exc:
            # Never-Raise Contract: log and gracefully return None
            logger.exception("[INCIDENT BUILDER] Failed to complete incident bundle dump: %s", exc)
            return None
