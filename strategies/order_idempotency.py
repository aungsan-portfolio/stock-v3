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
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from config import DATA_DIR

from datetime import datetime, timezone

logger = logging.getLogger(__name__)

IDEMPOTENCY_FILE = DATA_DIR / "order_idempotency_state.json"
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
    sym = symbol.upper().strip()
    s_side = side.upper().strip()
    side_code = s_side[:1] if s_side else "B"
    strat_code = str(strategy or "ord").replace("StrategyName.", "").replace("_", "").lower()[:4]

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
    # - strategy_entry_attempt (preferred) or legacy `attempt` represents strategy-level entry sequence/re-entry/scale-in.
    # - submission_retry_attempt represents transport/HTTP-level retry and is NEVER included in the digest.
    if strategy_entry_attempt is not None:
        entry_attempt = int(strategy_entry_attempt)
    else:
        # Legacy fallback: 'attempt' is treated strictly as strategy-level entry attempt (default 0)
        entry_attempt = int(attempt or 0)

    # submission_retry_attempt is intentionally omitted from raw so transport retries do not alter order identity
    raw = f"{sym}|{s_side}|{strat_code}|{s_date}|{bar_ts}|{sig_hash}|{nonce}|{entry_attempt}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"se3-{sym}-{side_code}-{strat_code}-{s_date}-{digest}"


@dataclass
class OrderReservation:
    key: str
    symbol: str
    side: str
    parent_order_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    status: str = "PENDING"  # PENDING, ACKNOWLEDGED, TERMINAL


class EntryIdempotencyStore:
    """Thread-safe persistent order idempotency reservation store."""

    def __init__(self, storage_path: Path = IDEMPOTENCY_FILE):
        self.storage_path = Path(storage_path)
        self._lock = threading.RLock()
        self.reservations: Dict[str, OrderReservation] = {}
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
                logger.error(f"[IDEMPOTENCY] Failed to load reservation state: {e}")

    def _save_state(self) -> None:
        with self._lock:
            try:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = self.storage_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump({k: asdict(v) for k, v in self.reservations.items()}, f, indent=2)
                os.replace(temp_path, self.storage_path)
            except Exception as e:
                logger.error(f"[IDEMPOTENCY] Failed to save reservation state: {e}")

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
            if key in self.reservations:
                res = self.reservations[key]
                if res.status != "TERMINAL":
                    if not res.parent_order_id and (now - res.created_at) > ttl_seconds:
                        logger.warning(f"[IDEMPOTENCY] Expired stale un-acknowledged reservation for {key}")
                        del self.reservations[key]
                        self._save_state()
                    else:
                        msg = f"[IDEMPOTENCY] Active reservation blocks {key} (status={res.status}, order={res.parent_order_id})"
                        logger.info(msg)
                        return False, msg

            self.reservations[key] = OrderReservation(
                key=key,
                symbol=sym,
                side=s_side,
                created_at=now,
                status="PENDING",
            )
            self._save_state()
            return True, "Reserved"

    def bind_parent(self, reservation_key_or_symbol: str, parent_order_id: str) -> bool:
        """Bind parent order ID once broker acknowledges placement."""
        key = reservation_key_or_symbol.upper().strip()
        with self._lock:
            for k, res in self.reservations.items():
                if k == key or res.symbol == key or res.key == key:
                    res.parent_order_id = str(parent_order_id)
                    res.status = "ACKNOWLEDGED"
                    self._save_state()
                    logger.info(f"[IDEMPOTENCY] Bound parent order {parent_order_id} to reservation {k}")
                    return True
            # Fallback if key not formatted with underscore
            if key in self.reservations:
                self.reservations[key].parent_order_id = str(parent_order_id)
                self.reservations[key].status = "ACKNOWLEDGED"
                self._save_state()
                return True
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
            for k in keys_to_release:
                del self.reservations[k]
                logger.info(f"[IDEMPOTENCY] Completed reservation {k} for terminal order {parent_order_id} ({terminal_status})")
            if keys_to_release:
                self._save_state()
                return True
        return False

    def release(self, symbol: str, side: str = "BUY", reason: str = "") -> bool:
        """Release reservation if order placement was aborted before broker parent ID attachment."""
        sym = symbol.upper().strip()
        s_side = side.upper().strip()
        key = f"{sym}_{s_side}"
        with self._lock:
            if key in self.reservations:
                del self.reservations[key]
                self._save_state()
                logger.info(f"[IDEMPOTENCY] Released reservation for {key} (reason: {reason})")
                return True
        return False

    def release_on_terminal(self, parent_order_id: str, status: str) -> None:
        """Alias for complete for entry_lifecycle compatibility."""
        self.complete(parent_order_id, status)


OrderIdempotencyManager = EntryIdempotencyStore
idempotency_manager = EntryIdempotencyStore()

