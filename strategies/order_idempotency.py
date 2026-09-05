"""
order_idempotency.py -- M-05 Durable Order Idempotency Module

Prevents duplicate order submissions until broker terminal state (filled, canceled, rejected, expired).
Persists reservation state to disk so protection survives bot restarts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from config import DATA_DIR

from datetime import datetime, timezone

logger = logging.getLogger(__name__)

IDEMPOTENCY_FILE = DATA_DIR / "order_idempotency_state.json"

STATUS_PENDING = "PENDING"
STATUS_ACKNOWLEDGED = "ACKNOWLEDGED"
STATUS_RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"

TERMINAL_STATUSES = {"filled", "canceled", "cancelled", "rejected", "expired", "done_for_day"}


def build_client_order_id(
    symbol: str,
    side: str,
    bar_timestamp_utc: str = "",
    signal_hash: str = "",
    attempt: int = 0,
    strategy: str = "ord",
    session_date: str = "",
    intent_nonce: str = "",
    strategy_entry_attempt: Optional[int] = None,
    submission_retry_attempt: int = 0,
) -> str:
    """
    Broker-level deterministic idempotency key. (Alpaca client_order_id <= 128 chars).
    
    Combines symbol, side, strategy, session date, bar timestamp, signal hash, intent nonce,
    and strategy-level re-entry attempt. Transport retries (submission_retry_attempt) are explicitly
    excluded from the digest so repeated transport submissions maintain identical client_order_id.

    Outputs: se3-{SYMBOL}-{SIDE[:1]}-{STRAT[:4]}-{SESSION_DATE}-{DIGEST[:16]}
    """
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError(f"Symbol must be a non-empty string, got {symbol!r}")
    sym = symbol.upper().strip()
    if not re.match(r"^[A-Za-z0-9_.-]+$", sym):
        raise ValueError(f"Invalid characters in symbol: {symbol!r}")
    clean_sym = sym.replace(".", "_")

    if not isinstance(side, str) or not side.strip():
        raise ValueError(f"Side must be a non-empty string, got {side!r}")
    s_side = side.upper().strip()
    if s_side not in {"BUY", "SELL"}:
        raise ValueError(f"Invalid order side: {side!r}. Must be 'BUY' or 'SELL'")
    side_code = s_side[:1]

    # Normalize strategy: full canonical name participates in hash digest to prevent collision
    # across strategies sharing the same 4-letter prefix (e.g. VWAP_BOUNCE vs VWAP_BREAKOUT)
    strategy_str = str(strategy or "ord").replace("StrategyName.", "").strip().lower()
    strategy_full = re.sub(r"[^a-z0-9_]", "", strategy_str) or "ord"
    strat_code = strategy_full.replace("_", "")[:4] or "ord"

    if strategy_entry_attempt is not None:
        try:
            entry_attempt = int(strategy_entry_attempt)
        except (ValueError, TypeError):
            raise ValueError(f"strategy_entry_attempt must be an integer >= 0, got {strategy_entry_attempt!r}")
    else:
        try:
            entry_attempt = int(attempt if attempt is not None else 0)
        except (ValueError, TypeError):
            raise ValueError(f"attempt must be an integer >= 0, got {attempt!r}")

    if entry_attempt < 0:
        raise ValueError(f"entry_attempt must be >= 0, got {entry_attempt}")

    try:
        retry_attempt = int(submission_retry_attempt if submission_retry_attempt is not None else 0)
    except (ValueError, TypeError):
        raise ValueError(f"submission_retry_attempt must be an integer >= 0, got {submission_retry_attempt!r}")

    if retry_attempt < 0:
        raise ValueError(f"submission_retry_attempt must be >= 0, got {retry_attempt}")

    now_utc = datetime.now(timezone.utc)
    s_date = str(session_date or "").strip()
    bar_ts = str(bar_timestamp_utc or "").strip()

    if not s_date:
        if bar_ts and len(bar_ts) >= 10 and bar_ts[:4].isdigit():
            s_date = bar_ts[:10].replace("-", "")
        else:
            s_date = now_utc.strftime("%Y%m%d")

    if not bar_ts:
        # Fallback to 5-minute bucket of current time so it's deterministic within candle
        bucket_min = (now_utc.minute // 5) * 5
        bar_ts = now_utc.replace(minute=bucket_min, second=0, microsecond=0).strftime("%Y%m%d_%H%M")

    sig_hash = str(signal_hash or "").strip()
    nonce = str(intent_nonce or "").strip()

    # Parameter semantics:
    # - strategy_full participates in raw digest to guarantee unique keys across distinct strategies
    # - strategy_entry_attempt represents strategy-level entry sequence/re-entry/scale-in and alters client_order_id
    # - submission_retry_attempt represents transport/HTTP retry and is NEVER included in the digest
    raw = f"{clean_sym}|{s_side}|{strategy_full}|{s_date}|{bar_ts}|{sig_hash}|{nonce}|{entry_attempt}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    client_order_id = f"se3-{clean_sym}-{side_code}-{strat_code}-{s_date}-{digest}"

    if len(client_order_id) > 128:
        raise ValueError(f"client_order_id exceeds Alpaca limit (128 chars): {len(client_order_id)} chars ({client_order_id})")
    if not re.match(r"^[A-Za-z0-9_-]+$", client_order_id):
        raise ValueError(f"client_order_id contains invalid broker characters: {client_order_id}")

    return client_order_id


@dataclass
class OrderReservation:
    key: str
    symbol: str
    side: str
    parent_order_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    status: str = STATUS_PENDING


class EntryIdempotencyStore:
    """Thread-safe persistent order idempotency reservation store enforcing fail-closed invariants."""

    def __init__(self, storage_path: Path = IDEMPOTENCY_FILE):
        self.storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self.reservations: Dict[str, OrderReservation] = {}
        self.is_healthy: bool = True
        self.health_reason: str = "OK"
        self._load_state()

    def _load_state(self) -> None:
        with self._lock:
            if not self.storage_path.exists():
                return
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for key, item in data.items():
                        self.reservations[key] = OrderReservation(**item)
                logger.info(f"[IDEMPOTENCY] Loaded {len(self.reservations)} active order reservations")
            except Exception as e:
                self.is_healthy = False
                self.health_reason = "LOAD_FAILURE"
                logger.exception(f"[IDEMPOTENCY] Failed to load reservation state from {self.storage_path}: {e}")

    def _save_state(self) -> None:
        with self._lock:
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.storage_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump({k: asdict(v) for k, v in self.reservations.items()}, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self.storage_path)
            except Exception:
                logger.exception(
                    "[IDEMPOTENCY] Failed to persist reservation state path=%s",
                    self.storage_path,
                )
                raise

    def reserve(
        self,
        reservation_key_or_symbol: str,
        signal_id: Optional[str] = None,
        symbol: Optional[str] = None,
        side: Optional[str] = None,
        ttl_seconds: float = 30.0,
    ) -> Tuple[bool, str]:
        """
        Attempt to reserve submission slot.
        Accepts reserve("NVDA", side="BUY") or reserve("NVDA_BUY", signal_id="sig-1", symbol="NVDA", side="BUY").
        Returns (reserved: bool, reason: str).
        """
        now = time.time()
        if symbol and side:
            sym = symbol.upper().strip()
            s_side = side.upper().strip()
            key = f"{sym}_{s_side}"
        elif "_" in reservation_key_or_symbol and not symbol and not side:
            key = reservation_key_or_symbol.upper().strip()
            parts = key.split("_", 1)
            sym = parts[0]
            s_side = parts[1]
        else:
            sym = reservation_key_or_symbol.upper().strip()
            s_side = (side or "BUY").upper().strip()
            key = f"{sym}_{s_side}"

        with self._lock:
            if not self.is_healthy:
                return False, "IDEMPOTENCY_STORE_UNHEALTHY"

            existing = self.reservations.get(key)

            if existing and existing.status == STATUS_ACKNOWLEDGED:
                return False, "ACTIVE_ACKNOWLEDGED_RESERVATION"

            if existing and existing.status == STATUS_RECONCILIATION_REQUIRED:
                return False, "RECONCILIATION_REQUIRED"

            if existing and existing.status == STATUS_PENDING:
                age = now - existing.created_at

                if age <= ttl_seconds:
                    return False, "ACTIVE_PENDING_RESERVATION"

                old_status = existing.status
                existing.status = STATUS_RECONCILIATION_REQUIRED

                try:
                    self._save_state()
                except Exception:
                    existing.status = old_status
                    self.is_healthy = False
                    self.health_reason = "PERSISTENCE_FAILURE"
                    return False, "PERSISTENCE_FAILED"

                return False, "RECONCILIATION_REQUIRED"

            new_reservation = OrderReservation(
                key=key,
                symbol=sym,
                side=s_side,
                created_at=now,
                status=STATUS_PENDING,
            )

            previous = self.reservations.get(key)
            self.reservations[key] = new_reservation

            try:
                self._save_state()
            except Exception:
                if previous is None:
                    self.reservations.pop(key, None)
                else:
                    self.reservations[key] = previous

                self.is_healthy = False
                self.health_reason = "PERSISTENCE_FAILURE"
                return False, "PERSISTENCE_FAILED"

            return True, "RESERVED"

    def bind_parent(self, reservation_key_or_symbol: str, parent_order_id: str) -> bool:
        """Bind parent order ID once broker acknowledges placement."""
        key = reservation_key_or_symbol.upper().strip()
        with self._lock:
            target_key = None
            for k, res in self.reservations.items():
                if k == key or res.symbol == key or res.key == key:
                    target_key = k
                    break
            if not target_key and key in self.reservations:
                target_key = key

            if target_key and target_key in self.reservations:
                res = self.reservations[target_key]
                old_parent = res.parent_order_id
                old_status = res.status
                res.parent_order_id = str(parent_order_id)
                res.status = STATUS_ACKNOWLEDGED
                try:
                    self._save_state()
                    logger.info(f"[IDEMPOTENCY] Bound parent order {parent_order_id} to reservation {target_key}")
                    return True
                except Exception:
                    res.parent_order_id = old_parent
                    res.status = old_status
                    self.is_healthy = False
                    self.health_reason = "PERSISTENCE_FAILURE"
                    return False
        return False

    def attach_parent_order(self, symbol: str, side: str, parent_order_id: str) -> None:
        """Alias for bind_parent for order_manager compatibility."""
        sym = symbol.upper().strip()
        s_side = side.upper().strip()
        self.bind_parent(f"{sym}_{s_side}", parent_order_id)

    def complete(self, parent_order_id: str, terminal_status: str) -> bool:
        """Complete and release reservation when parent order reaches broker terminal state."""
        st = str(terminal_status).lower()
        if st not in TERMINAL_STATUSES:
            return False

        with self._lock:
            keys_to_release = [
                k for k, v in self.reservations.items()
                if v.parent_order_id == str(parent_order_id)
            ]
            if not keys_to_release:
                return False

            backup = {k: self.reservations[k] for k in keys_to_release}
            for k in keys_to_release:
                del self.reservations[k]
                logger.info(f"[IDEMPOTENCY] Completed reservation {k} for terminal order {parent_order_id} ({terminal_status})")
            try:
                self._save_state()
                return True
            except Exception:
                for k, v in backup.items():
                    self.reservations[k] = v
                self.is_healthy = False
                self.health_reason = "PERSISTENCE_FAILURE"
                return False

    def release(self, symbol: str, side: str = "BUY", reason: str = "") -> bool:
        """Release reservation if order placement was aborted before broker parent ID attachment."""
        sym = symbol.upper().strip()
        s_side = side.upper().strip()
        key = f"{sym}_{s_side}"
        with self._lock:
            res = self.reservations.get(key)
            if res is None:
                return False

            if res.status != STATUS_PENDING or res.parent_order_id is not None:
                logger.error(
                    "[IDEMPOTENCY] Unsafe release refused "
                    "key=%s status=%s parent_order_id=%s reason=%s",
                    key,
                    res.status,
                    res.parent_order_id,
                    reason,
                )
                return False

            del self.reservations[key]
            try:
                self._save_state()
            except Exception:
                self.reservations[key] = res
                self.is_healthy = False
                self.health_reason = "PERSISTENCE_FAILURE"
                return False

            logger.info(f"[IDEMPOTENCY] Released reservation for {key} (reason: {reason})")
            return True

    def release_after_reconciliation(self, symbol: str, side: str = "BUY", reason: str = "") -> bool:
        """Audited administrative release after broker reconciliation confirms order is absent/terminal."""
        sym = symbol.upper().strip()
        s_side = side.upper().strip()
        key = f"{sym}_{s_side}"
        with self._lock:
            if key in self.reservations:
                del self.reservations[key]
                try:
                    self._save_state()
                    logger.info(f"[IDEMPOTENCY] Reconciled and released reservation for {key} (reason: {reason})")
                    return True
                except Exception:
                    self.is_healthy = False
                    self.health_reason = "PERSISTENCE_FAILURE"
                    return False
        return False

    def release_on_terminal(self, parent_order_id: str, status: str) -> None:
        """Alias for complete for entry_lifecycle compatibility."""
        self.complete(parent_order_id, status)

    def clear(self) -> None:
        """Reset in-memory and persisted reservation store (for tests and administrative maintenance)."""
        with self._lock:
            self.reservations.clear()
            self.is_healthy = True
            self.health_reason = "OK"
            try:
                self._save_state()
            except Exception:
                self.is_healthy = False
                self.health_reason = "PERSISTENCE_FAILURE"


OrderIdempotencyManager = EntryIdempotencyStore
idempotency_manager = EntryIdempotencyStore()

