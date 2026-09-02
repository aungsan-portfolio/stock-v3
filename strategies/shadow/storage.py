"""
strategies.shadow.storage -- Crash-Safe, Namespaced Persistence for Shadow Alpha.

Enforces:
1. Exact directory namespacing: data/shadow/{role}/{strategy_version}/
2. Process lock: Contention disables shadow mode without halting production engine.
3. Atomic active state saves (.tmp + replace).
4. Idempotent closure recovery: Closed IDs tracked; duplicate records blocked.
5. Corrupted tail recovery: Partial trailing JSONL lines safely quarantined.
"""
import json
import logging
import os
from pathlib import Path
import socket
import time
from typing import Dict, List, Optional, Set
import pandas as pd

from strategies.shadow.models import ShadowTrade

logger = logging.getLogger(__name__)


class ShadowStorageManager:
    def __init__(
        self,
        base_dir: str = "data/shadow",
        role: str = "daytrade",
        strategy_version: str = "ORB_CONFIRMED_V1_FIXED_2R",
    ):
        self.role = str(role or "daytrade").lower()
        self.strategy_version = str(strategy_version)
        self.dir_path = Path(base_dir) / self.role / self.strategy_version
        self.dir_path.mkdir(parents=True, exist_ok=True)

        self.active_file = self.dir_path / "active.json"
        self.closed_file = self.dir_path / "closed.jsonl"
        self.lock_file = self.dir_path / "shadow.lock"

        self._lock_acquired = False
        self._closed_ids: Set[str] = set()

    @property
    def is_locked(self) -> bool:
        return self._lock_acquired

    def acquire_lock(self) -> bool:
        """Attempts to acquire exclusive single-writer process lock.

        If lock is held by another active process, fails cleanly and disables
        shadow pipeline without impacting production engine availability.
        """
        now_utc = pd.Timestamp.now(tz="UTC").isoformat()
        lock_info = {
            "pid": os.getpid(),
            "process_start_utc": now_utc,
            "hostname": socket.gethostname(),
            "role": self.role,
            "strategy_version": self.strategy_version,
        }

        if self.lock_file.exists():
            try:
                with open(self.lock_file, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing_pid = existing.get("pid")
                # Check if process is still alive on Windows/Unix
                if existing_pid and self._is_pid_alive(existing_pid) and existing_pid != os.getpid():
                    logger.warning(
                        "[SHADOW DISABLED: LOCK UNAVAILABLE] Process lock held by PID %d (%s). Shadow mode disabled.",
                        existing_pid,
                        existing.get("hostname", "unknown"),
                    )
                    self._lock_acquired = False
                    return False
            except Exception as e:
                logger.debug("Failed reading existing shadow lock (%s); will attempt overwrite", e)

        # Write our lock
        try:
            tmp_lock = self.dir_path / f"shadow.lock.{os.getpid()}.tmp"
            with open(tmp_lock, "w", encoding="utf-8") as f:
                json.dump(lock_info, f, indent=2)
            os.replace(tmp_lock, self.lock_file)
            self._lock_acquired = True
            logger.info("[SHADOW STORAGE] Acquired exclusive shadow lock for %s/%s", self.role, self.strategy_version)
            return True
        except Exception as exc:
            logger.warning("[SHADOW DISABLED: LOCK UNAVAILABLE] Could not write shadow lock: %s", exc)
            self._lock_acquired = False
            return False

    def release_lock(self) -> None:
        if self._lock_acquired and self.lock_file.exists():
            try:
                self.lock_file.unlink()
                self._lock_acquired = False
                logger.info("[SHADOW STORAGE] Released shadow lock")
            except Exception as e:
                logger.debug("Error releasing shadow lock: %s", e)

    @staticmethod
    def _is_pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            # os.kill(pid, 0) works on Unix and Windows in modern Python
            os.kill(pid, 0)
            return True
        except (OSError, PermissionError):
            return False

    def load_closed_ids(self) -> Set[str]:
        """Loads closed shadow trade IDs with corrupt tail recovery."""
        self._closed_ids.clear()
        if not self.closed_file.exists():
            return self._closed_ids

        corrupt_lines = []
        with open(self.closed_file, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            line_s = line.strip()
            if not line_s:
                continue
            try:
                rec = json.loads(line_s)
                tid = rec.get("shadow_trade_id")
                if tid:
                    self._closed_ids.add(tid)
            except Exception as parse_err:
                # Capture corrupt line (e.g. final truncated line from hard crash)
                corrupt_lines.append((idx + 1, line_s, str(parse_err)))

        if corrupt_lines:
            logger.warning(
                "[SHADOW STORAGE CORRUPTION RECOVERY] Found %d unparseable line(s) in %s. Prior valid records loaded (%d IDs).",
                len(corrupt_lines),
                self.closed_file.name,
                len(self._closed_ids),
            )
        return self._closed_ids

    def load_active_trades(self) -> Dict[str, ShadowTrade]:
        """Loads active shadow trades and reconciles against closed ledger."""
        self.load_closed_ids()
        active_map: Dict[str, ShadowTrade] = {}

        if not self.active_file.exists():
            return active_map

        try:
            with open(self.active_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                return active_map

            reconciled_out = 0
            for tid, trade_dict in data.items():
                if tid in self._closed_ids:
                    # Stale active copy from crash between closed append and active delete!
                    reconciled_out += 1
                    continue
                try:
                    trade = ShadowTrade.from_dict(trade_dict)
                    active_map[trade.shadow_trade_id] = trade
                except Exception as ex:
                    logger.warning("[SHADOW STORAGE] Malformed active trade record %s skipped: %s", tid, ex)

            if reconciled_out > 0:
                logger.info(
                    "[SHADOW STORAGE RECONCILE] Discarded %d stale active shadow trade(s) already recorded in closed ledger",
                    reconciled_out,
                )
                self.save_active_trades(active_map)
        except Exception as e:
            logger.warning("[SHADOW STORAGE] Error loading active trades from %s: %s", self.active_file, e)

        return active_map

    def save_active_trades(self, active_trades: Dict[str, ShadowTrade]) -> None:
        """Atomic write of active shadow trades via temporary file."""
        tmp_file = self.dir_path / f"active.{os.getpid()}.tmp"
        data = {tid: trade.to_dict() for tid, trade in active_trades.items()}
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.active_file)
        except Exception as e:
            logger.error("[SHADOW STORAGE] Failed atomic save of active trades: %s", e)
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except Exception:
                    pass

    def append_closed_trade(self, trade: ShadowTrade) -> bool:
        """Appends closed shadow trade to closed.jsonl idempotently with fsync."""
        if trade.shadow_trade_id in self._closed_ids:
            logger.warning(
                "[SHADOW STORAGE IDEMPOTENCY] Trade %s already in closed ledger. Skipping duplicate append.",
                trade.shadow_trade_id,
            )
            return False

        line = json.dumps(trade.to_dict()) + "\n"
        try:
            with open(self.closed_file, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            self._closed_ids.add(trade.shadow_trade_id)
            return True
        except Exception as e:
            logger.error("[SHADOW STORAGE] Failed appending closed trade %s: %s", trade.shadow_trade_id, e)
            return False
