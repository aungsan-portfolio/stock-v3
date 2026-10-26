"""
validation_lane.py -- Phase C Active Paper Validation Engine.

Enforces the Two-Lane Architecture:
1. Production Lane (Frozen): High-confidence, strict policy, full Kelly/risk sizing.
2. Validation Lane (Active): Diagnostic execution on soft-policy rejected candidates
   that strictly pass all hard safety gates.

Core Invariants:
- Hard Safety is 100% Non-Bypassable:
  * Zero execution on missing/stale SOXX
  * Zero execution on cross-feed dislocation / feed mismatch
  * Zero execution on stale quote (> 10s)
  * Zero execution on negative latency
  * Zero execution on price divergence (> 1.5%)
  * Zero execution on stream recovery lock / latched desync
  * Zero execution on invalid geometry / noise floor violation
- Fixed Tiny Sizing: 1-3 shares (default: 1 share)
- Separate Isolated Ledger: data/paper_validation_trades.jsonl
- Isolated Client Order IDs: val_{symbol}_{strategy}_{hash}
"""
import os
import json
import time
import math
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Tuple

import config
from strategies.base import TradeSignal, StrategyName, OrderIntent
from strategies.execution_quote import validate_execution_quote, BrokerQuote
from pathlib import Path
from strategies.position_sizer import validate_executable_geometry

logger = logging.getLogger(__name__)

_validation_lock = threading.Lock()

def get_validation_ledger_path() -> str:
    """Dynamically resolve paper validation trades ledger path against config."""
    override = globals().get("_ledger_path_override")
    if override is not None:
        return str(override)
    try:
        import config
        return str(getattr(config, "VALIDATION_LANE_LEDGER_PATH", Path(config.DATA_DIR) / "paper_validation_trades.jsonl"))
    except Exception:
        return os.path.join("data", "paper_validation_trades.jsonl")

DEFAULT_VALIDATION_LEDGER_PATH = get_validation_ledger_path()

# Non-bypassable hard safety veto codes
HARD_SAFETY_VETO_CODES = {
    "CROSS_FEED_PROVENANCE_DISLOCATION",
    "PROVENANCE_FEED_DISCREPANCY_VETO",
    "FEED_IDENTITY_MISSING",
    "QUOTE_STALE",
    "QUOTE_TIMESTAMP_MISSING",
    "BAR_STALE",
    "BAR_TIMESTAMP_MALFORMED",
    "BAR_TIMESTAMP_IN_FUTURE",
    "NEGATIVE_LATENCY_ERROR",
    "SOXX_TECH_LONG_VETO",
    "DATA_DEGRADED",
    "STREAM_RECOVERY_REQUIRED",
    "LATCHED_DESYNC",
    "RECONCILE_MISMATCH",
    "INVALID_GEOMETRY",
    "NOISE_FLOOR_VIOLATION",
}

# Permissible soft policy rejection codes that may qualify for Validation Lane
SOFT_POLICY_REJECTION_CODES = {
    "CONFIDENCE_BELOW_FLOOR",
    "LUNCH_CHOP_ZONE",
    "REGIME_RISK_OFF",
    "ENTRY_CUTOFF_TIME",
    "TEXTURE_RESTRICTION",
}


@dataclass
class ValidationTradeRecord:
    validation_id: str
    timestamp: str
    symbol: str
    strategy: str
    side: str
    confidence: float
    regime: str
    market_texture: str
    terminal_gate: str
    rejection_reason: str
    decision_price: float
    decision_feed: str
    bar_timestamp: str
    quote_mid: float
    quote_feed: str
    quote_timestamp: float
    entry_order_id: Optional[str] = None
    entry_fill_price: Optional[float] = None
    exit_order_id: Optional[str] = None
    exit_fill_price: Optional[float] = None
    stop_price: float = 0.0
    target_price: float = 0.0
    slippage_bps: Optional[float] = None
    realized_pnl: Optional[float] = None
    hard_safety_pass: bool = False
    policy_rejected: bool = True
    validation_only: bool = True
    quarantined_from_production_budget: bool = True
    status: str = "PENDING"  # PENDING, SUBMITTED, FILLED, CLOSED, VETOED
    qty: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def log_validation_record(
    record: ValidationTradeRecord,
    filepath: Optional[str] = None,
) -> None:
    """Append a validation trade record to paper validation trades ledger."""
    filepath = filepath or get_validation_ledger_path()
    dir_name = os.path.dirname(filepath)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    line = json.dumps(record.to_dict(), default=str) + "\n"
    with _validation_lock:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        if os.name != "nt":
            try:
                dfd = os.open(dir_name or ".", os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except Exception:
                pass


def build_fill_composite_key(
    order_id: str,
    side: str,
    fill_price: float,
    qty: float,
    timestamp: Optional[str] = None,
    execution_id: Optional[str] = None,
) -> str:
    """Build a deterministic composite deduplication key safe against partial fills."""
    if execution_id and str(execution_id).strip():
        return str(execution_id).strip()
    ts = str(timestamp or "").strip()
    return f"VAL_FILL:{order_id}:{side.upper().strip()}:{float(fill_price):.4f}:{float(qty):.4f}:{ts}"


def load_seen_validation_order_ids(filepath: str = DEFAULT_VALIDATION_LEDGER_PATH) -> set:
    """Build a set of all seen validation entry and exit order IDs on startup."""
    seen = set()
    if not os.path.exists(filepath):
        return seen
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if not line_s:
                    continue
                try:
                    d = json.loads(line_s)
                    for k in ("entry_order_id", "exit_order_id", "order_id"):
                        val = d.get(k)
                        if val:
                            seen.add(str(val).strip())
                except Exception:
                    continue
    except Exception as e:
        logger.error("[VALIDATION LEDGER ERROR] Failed to load seen order IDs: %s", e)
    return seen


def dedupe_on_read(filepath: str = DEFAULT_VALIDATION_LEDGER_PATH) -> list:
    """
    Returns a consolidated in-memory projection of validation trade records.
    Deduplicates repeated entries, aggregates partial fills, and preserves historical records
    without rewriting or mutating the underlying ledger file on disk.
    """
    if not os.path.exists(filepath):
        return []

    with _validation_lock:
        raw_records = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            raw_records.append(json.loads(line_s))
                        except Exception:
                            continue
        except Exception as e:
            logger.error("[VALIDATION LEDGER ERROR] dedupe_on_read failed on %s: %s", filepath, e)
            return []

        trades_by_id = {}
        ordered_keys = []

        for r in raw_records:
            v_id = str(r.get("validation_id") or r.get("entry_order_id") or "").strip()
            if not v_id:
                v_id = f"synth_{r.get('symbol')}_{r.get('timestamp')}"

            if v_id not in trades_by_id:
                trades_by_id[v_id] = dict(r)
                ordered_keys.append(v_id)
            else:
                existing = trades_by_id[v_id]
                for field in (
                    "entry_order_id", "entry_fill_price", "exit_order_id", "exit_fill_price",
                    "realized_pnl", "pnl", "slippage_bps", "exit_reason", "status", "event_type",
                    "entry_price", "exit_price"
                ):
                    if r.get(field) is not None:
                        existing[field] = r[field]

        return [trades_by_id[k] for k in ordered_keys]


def load_validation_fill_states(
    account_id: Optional[str] = None,
    trading_date: Optional[Any] = None,
    filepath: str = DEFAULT_VALIDATION_LEDGER_PATH,
) -> dict:
    """
    Extracts OrderFillState objects for all validation orders from paper_validation_trades.jsonl.
    Enables AlpacaBridge reconciliation to recognize already-logged validation orders
    and avoid replaying fills on subsequent polls.
    """
    from decimal import Decimal
    from strategies.trade_journal import OrderFillState
    import pytz

    states = {}
    if not os.path.exists(filepath):
        return states

    target_date_str = str(trading_date) if trading_date is not None else None
    target_acc_str = str(account_id) if account_id is not None else "PA3UODIC5JTA"

    with _validation_lock:
        records = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            records.append(json.loads(line_s))
                        except Exception:
                            continue
        except Exception as e:
            logger.error("[VALIDATION LEDGER ERROR] Failed to load validation fill states from %s: %s", filepath, e)
            return states

        for rec in records:
            sym = str(rec.get("symbol") or "").upper().strip()
            side = str(rec.get("side") or "BUY").upper().strip()
            ts_str = rec.get("timestamp") or ""
            rec_date_str = None
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    et_tz = pytz.timezone("US/Eastern")
                    rec_date_str = str(dt.astimezone(et_tz).date())
                except Exception:
                    rec_date_str = None

            if target_date_str and rec_date_str and rec_date_str != target_date_str:
                continue

            scoped_acc = target_acc_str
            date_key = rec_date_str or target_date_str or ""

            # 1. Entry fill
            entry_oid = str(rec.get("entry_order_id") or "").strip()
            entry_px = rec.get("entry_fill_price") or rec.get("entry_price")
            entry_status = str(rec.get("status") or "").upper().strip()
            if entry_oid and entry_px is not None and entry_status in ("FILLED", "CLOSED"):
                e_key = (scoped_acc, date_key, entry_oid, sym, side)
                if e_key not in states:
                    states[e_key] = OrderFillState(broker_order_id=entry_oid)
                e_qty = Decimal(str(rec.get("qty", 1)))
                e_px = Decimal(str(entry_px))
                exec_id = str(rec.get("entry_execution_id") or f"VAL:ENTRY:{entry_oid}")
                states[e_key].add_fill(e_qty, e_px, exec_id)

            # 2. Exit fill
            exit_oid = str(rec.get("exit_order_id") or "").strip()
            exit_px = rec.get("exit_fill_price") or rec.get("exit_price")
            if exit_oid and exit_px is not None and entry_status == "CLOSED":
                exit_side = "SELL" if side in ("BUY", "LONG") else "BUY"
                x_key = (scoped_acc, date_key, exit_oid, sym, exit_side)
                if x_key not in states:
                    states[x_key] = OrderFillState(broker_order_id=exit_oid)
                x_qty = Decimal(str(rec.get("qty", 1)))
                x_px = Decimal(str(exit_px))
                exec_id = str(rec.get("exit_execution_id") or f"VAL:EXIT:{exit_oid}")
                states[x_key].add_fill(x_qty, x_px, exec_id)

    return states


def record_validation_fill(
    order_id: str,
    fill_price: float,
    qty: float,
    side: str,
    timestamp: Optional[str] = None,
    slippage: Optional[float] = None,
    client_order_id: Optional[str] = None,
    symbol: Optional[str] = None,
    execution_id: Optional[str] = None,
    filepath: str = DEFAULT_VALIDATION_LEDGER_PATH,
) -> bool:
    """Record a broker entry fill on a validation trade record in paper_validation_trades.jsonl."""
    if not os.path.exists(filepath):
        return False

    with _validation_lock:
        records = []
        updated = False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        records.append(json.loads(line_s))
                    except Exception:
                        continue
        except Exception as read_err:
            logger.error("[VALIDATION LEDGER ERROR] Failed to read %s: %s", filepath, read_err)
            return False

        target_oid = str(order_id or "").strip()
        target_cid = str(client_order_id or "").strip()
        target_sym = str(symbol or "").upper().strip()
        comp_key = build_fill_composite_key(target_oid, side, fill_price, qty, timestamp, execution_id)

        for rec in reversed(records):
            r_oid = str(rec.get("entry_order_id") or "").strip()
            r_cid = str(rec.get("validation_id") or "").strip()
            r_sym = str(rec.get("symbol") or "").upper().strip()
            r_status = str(rec.get("status") or "").upper().strip()
            r_exec = str(rec.get("entry_execution_id") or "").strip()

            is_match = False
            if target_oid and (r_oid == target_oid or target_oid in r_oid):
                is_match = True
            elif target_cid and (r_cid == target_cid or target_cid in r_cid):
                is_match = True
            elif target_sym and r_sym == target_sym and r_status == "SUBMITTED":
                is_match = True

            # If already filled with exact same execution ID or composite key, idempotent no-op
            if is_match and (r_exec == comp_key or (execution_id and r_exec == execution_id)):
                return True

            if is_match and r_status != "CLOSED":
                # Handle partial fill aggregation if already partially filled
                prev_qty = float(rec.get("qty", 0.0) or 0.0) if r_status == "FILLED" else 0.0
                prev_px = float(rec.get("entry_fill_price", 0.0) or 0.0) if r_status == "FILLED" else 0.0
                new_delta_qty = float(qty)
                total_qty = prev_qty + new_delta_qty
                if total_qty > 0 and prev_qty > 0:
                    weighted_px = (prev_qty * prev_px + new_delta_qty * float(fill_price)) / total_qty
                else:
                    weighted_px = float(fill_price)

                rec["entry_fill_price"] = float(round(weighted_px, 4))
                rec["entry_price"] = float(round(weighted_px, 4))
                rec["qty"] = total_qty
                rec["event_type"] = "TRADE"
                rec["entry_execution_id"] = comp_key
                if slippage is not None:
                    rec["slippage_bps"] = round(float(slippage) * 10000.0, 2)
                if target_oid and not rec.get("entry_order_id"):
                    rec["entry_order_id"] = target_oid
                rec["status"] = "FILLED"
                updated = True
                logger.info(
                    "[VALIDATION FILL RECORDED] %s %s @ $%.2f (order_id=%s, qty=%.2f, slippage=%.4f)",
                    rec.get("symbol"), rec.get("side"), fill_price, order_id, total_qty, slippage or 0.0
                )
                break

        if updated:
            temp_path = f"{filepath}.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, filepath)
                if os.name != "nt":
                    try:
                        dir_name = os.path.dirname(os.path.abspath(filepath))
                        dfd = os.open(dir_name or ".", os.O_RDONLY)
                        try:
                            os.fsync(dfd)
                        finally:
                            os.close(dfd)
                    except Exception:
                        pass
                return True
            except Exception as write_err:
                logger.error("[VALIDATION LEDGER ERROR] Failed to update %s: %s", filepath, write_err)
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                return False

    return False


def record_validation_exit(
    order_id: str,
    exit_price: float,
    qty: float,
    side: str,
    timestamp: Optional[str] = None,
    client_order_id: Optional[str] = None,
    symbol: Optional[str] = None,
    exit_reason: str = "STOP_OUT",
    execution_id: Optional[str] = None,
    filepath: str = DEFAULT_VALIDATION_LEDGER_PATH,
    parent_order_id: Optional[str] = None,
) -> bool:
    """Record an exit fill on an open validation trade record in paper_validation_trades.jsonl."""
    if not os.path.exists(filepath):
        return False

    with _validation_lock:
        records = []
        updated = False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if not line_s:
                        continue
                    try:
                        records.append(json.loads(line_s))
                    except Exception:
                        continue
        except Exception as read_err:
            logger.error("[VALIDATION LEDGER ERROR] Failed to read %s: %s", filepath, read_err)
            return False

        target_oid = str(order_id or "").strip()
        target_sym = str(symbol or "").upper().strip()
        target_cid = str(client_order_id or "").strip()
        target_poid = str(parent_order_id or "").strip()
        exit_side_norm = str(side or "").upper().strip()
        try:
            fill_qty = float(qty)
            exit_px = float(exit_price)
        except (ValueError, TypeError):
            logger.error("[VALIDATION EXIT INVALID NUMBERS] qty=%s exit_price=%s", qty, exit_price)
            return False

        if not math.isfinite(fill_qty) or fill_qty <= 0 or not math.isfinite(exit_px) or exit_px <= 0:
            logger.error("[VALIDATION EXIT NON-FINITE] fill_qty=%s exit_px=%s must be finite and > 0", fill_qty, exit_px)
            return False

        comp_key = build_fill_composite_key(target_oid, exit_side_norm, exit_price, qty, timestamp, execution_id)

        # 1. Partial-Fill Safe Deduplication Guard:
        # Check if this exact execution (by execution_id or deterministic composite key)
        # has already been recorded in any record in the ledger.
        for rec in reversed(records):
            r_sym = str(rec.get("symbol") or "").upper().strip()
            if target_sym and r_sym != target_sym:
                continue
            r_exit_exec = str(rec.get("exit_execution_id") or "").strip()
            r_exit_exec_ids = [str(x).strip() for x in rec.get("exit_execution_ids", []) if x]

            if (execution_id and (r_exit_exec == execution_id or execution_id in r_exit_exec_ids)) or \
               (comp_key and (r_exit_exec == comp_key or comp_key in r_exit_exec_ids)):
                # This specific exit fill execution was already recorded
                return True

        # 2. Candidate Selection with Strict Directional and Filled Guards:
        # - Must match symbol
        # - Must be FILLED with verified entry fill price > 0 (prohibit closing un-filled SUBMITTED records)
        # - Must match opposite side (SELL closes BUY/LONG; BUY closes SELL/SHORT)
        # - Prohibit closing records that are already fully CLOSED
        eligible_candidates = []
        for rec in reversed(records):
            r_sym = str(rec.get("symbol") or "").upper().strip()
            r_status = str(rec.get("status") or "").upper().strip()
            r_side = str(rec.get("side") or "BUY").upper().strip()
            r_epx = rec.get("entry_fill_price")

            if target_sym and r_sym != target_sym:
                continue
            if r_status != "FILLED" or r_epx is None or float(r_epx) <= 0:
                continue

            # Direction check
            if exit_side_norm == "SELL" and r_side not in ("BUY", "LONG"):
                continue
            if exit_side_norm == "BUY" and r_side not in ("SELL", "SHORT"):
                continue

            eligible_candidates.append(rec)

        if not eligible_candidates:
            logger.warning(
                "[VALIDATION EXIT UNMATCHED] No open filled candidate found for %s (side=%s, oid=%s)",
                target_sym, exit_side_norm, target_oid
            )
            return False

        # 3. Mandatory Lineage Matching:
        # Rule: Lineage correlation MUST be proven.
        # - If target_poid is provided: must EXACTLY match candidate entry_order_id.
        #   If target_poid does not match, it is an explicit parent mismatch (weaker token matching PROHIBITED).
        # - If target_poid is not provided, but target_cid is provided: must EXACTLY match candidate client_order_id or validation_id.
        # - If neither target_poid nor target_cid is provided: lineage is absent.
        #   Even if only one candidate exists, it CANNOT be matched. Fail closed and retain pending.
        matched_rec = None

        if target_poid:
            for cand in eligible_candidates:
                c_eoid = str(cand.get("entry_order_id") or "").strip()
                if c_eoid == target_poid:
                    matched_rec = cand
                    break
            if matched_rec is None:
                logger.warning(
                    "[VALIDATION EXIT PARENT MISMATCH] Parent order ID '%s' did not match any open entry "
                    "for %s across %d candidate(s). Rejecting exit fail-closed.",
                    target_poid, target_sym, len(eligible_candidates)
                )
                return False

        elif target_cid:
            for cand in eligible_candidates:
                c_cid = str(cand.get("client_order_id") or "").strip()
                c_vid = str(cand.get("validation_id") or "").strip()
                if (c_cid and target_cid == c_cid) or (c_vid and target_cid == c_vid):
                    matched_rec = cand
                    break
            if matched_rec is None:
                logger.warning(
                    "[VALIDATION EXIT CLIENT ID MISMATCH] Client order ID '%s' did not match any open entry "
                    "for %s across %d candidate(s). Rejecting exit fail-closed.",
                    target_cid, target_sym, len(eligible_candidates)
                )
                return False

        else:
            logger.warning(
                "[VALIDATION EXIT NO LINEAGE] Exit order %s for %s has no parent_order_id or client_order_id. "
                "Lineage correlation cannot be proven across %d candidate(s); retaining pending fail-closed.",
                target_oid, target_sym, len(eligible_candidates)
            )
            return False

        # 4. Entry Quantity Validation & Overfill Rejection:
        rec = matched_rec
        entry_qty_val = rec.get("qty")
        if entry_qty_val is None:
            logger.error("[VALIDATION EXIT UNKNOWN ENTRY QTY] Record %s has no qty field. Failing closed.", rec.get("validation_id"))
            return False
        try:
            entry_qty = float(entry_qty_val)
        except (ValueError, TypeError):
            logger.error("[VALIDATION EXIT INVALID ENTRY QTY] Record %s has invalid qty %s. Failing closed.", rec.get("validation_id"), entry_qty_val)
            return False

        if not math.isfinite(entry_qty) or entry_qty <= 0:
            logger.error("[VALIDATION EXIT NON-POSITIVE ENTRY QTY] entry_qty=%s must be > 0. Failing closed.", entry_qty)
            return False

        already_closed = float(rec.get("closed_qty", 0.0) or 0.0)
        remaining_qty = entry_qty - already_closed

        if fill_qty > remaining_qty + 1e-4:
            logger.error(
                "[VALIDATION OVERFILL REJECTED] %s: fill_qty=%.4f > remaining_qty=%.4f "
                "(entry_qty=%.4f, already_closed=%.4f). Rejecting exit fill and retaining evidence.",
                target_sym, fill_qty, remaining_qty, entry_qty, already_closed
            )
            return False

        delta_closed = fill_qty
        new_closed_qty = round(already_closed + delta_closed, 4)
        rec["closed_qty"] = new_closed_qty

        # 5. Quantity-Weighted Average Exit Price & Cumulative Realized PnL:
        prior_exit_notional = float(rec.get("exit_notional", 0.0) or (already_closed * float(rec.get("exit_fill_price", 0.0) or 0.0)))
        delta_notional = delta_closed * exit_px
        new_exit_notional = round(prior_exit_notional + delta_notional, 4)
        rec["exit_notional"] = new_exit_notional

        entry_px = float(rec.get("entry_fill_price") or 0.0)
        pos_side = str(rec.get("side") or "BUY").upper().strip()
        if pos_side in ("BUY", "LONG"):
            delta_pnl = (exit_px - entry_px) * delta_closed
        else:
            delta_pnl = (entry_px - exit_px) * delta_closed

        prior_realized_pnl = float(rec.get("realized_pnl", 0.0) or rec.get("pnl", 0.0) or 0.0)
        cumulative_realized_pnl = round(prior_realized_pnl + delta_pnl, 4)
        rec["realized_pnl"] = cumulative_realized_pnl
        rec["pnl"] = cumulative_realized_pnl

        weighted_exit_px = round(new_exit_notional / new_closed_qty, 4) if new_closed_qty > 0 else exit_px
        rec["exit_fill_price"] = weighted_exit_px
        rec["exit_price"] = weighted_exit_px

        # Record exit execution ID for deduplication
        exit_execs = rec.get("exit_execution_ids", [])
        if not isinstance(exit_execs, list):
            exit_execs = [str(exit_execs)] if exit_execs else []
        if comp_key not in exit_execs:
            exit_execs.append(comp_key)
        if execution_id and execution_id not in exit_execs:
            exit_execs.append(execution_id)
        rec["exit_execution_ids"] = exit_execs
        rec["exit_execution_id"] = comp_key
        rec["exit_order_id"] = target_oid

        # Mark CLOSED only when cumulative closed quantity satisfies entry quantity
        if new_closed_qty >= (entry_qty - 1e-6):
            rec["status"] = "CLOSED"
            rec["event_type"] = "TRADE_CLOSED"
            rec["exit_reason"] = str(exit_reason)
            logger.info(
                "[VALIDATION EXIT RECORDED] %s %s fully exited (total_qty=%.4f, weighted_exit_px=$%.4f) | Realized P&L: $%.4f (reason=%s, exit_oid=%s)",
                rec.get("symbol"), pos_side, entry_qty, weighted_exit_px, cumulative_realized_pnl, exit_reason, order_id
            )
        else:
            rec["event_type"] = "PARTIAL_EXIT"
            logger.info(
                "[VALIDATION PARTIAL EXIT] %s closed %.4f/%.4f shares @ $%.2f (weighted_avg=$%.4f, partial_pnl=$%.4f, exit_oid=%s)",
                rec.get("symbol"), new_closed_qty, entry_qty, exit_px, weighted_exit_px, delta_pnl, order_id
            )
        updated = True

        if updated:
            temp_path = f"{filepath}.tmp.{os.getpid()}.{threading.get_ident()}"
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, filepath)
                if os.name != "nt":
                    try:
                        dir_name = os.path.dirname(os.path.abspath(filepath))
                        dfd = os.open(dir_name or ".", os.O_RDONLY)
                        try:
                            os.fsync(dfd)
                        finally:
                            os.close(dfd)
                    except Exception:
                        pass
                return True
            except Exception as write_err:
                logger.error("[VALIDATION LEDGER ERROR] Failed to update %s: %s", filepath, write_err)
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
                return False

    return False


def is_hard_safety_failure(reason_code: str, reason_str: str = "") -> bool:
    """Check if a rejection represents an un-bypassable hard safety fault."""
    code_upper = str(reason_code or "").upper().strip()
    reason_upper = str(reason_str or "").upper().strip()
    if code_upper in HARD_SAFETY_VETO_CODES:
        return True
    return any(k in reason_upper for k in HARD_SAFETY_VETO_CODES)


def evaluate_and_route_validation_candidate(
    signal: TradeSignal,
    bridge: Any,
    policy_stage: str,
    policy_reason_code: str,
    policy_reason_msg: str,
    regime_mode: str = "NORMAL",
    market_texture: str = "UNKNOWN",
    fixed_shares: int = 1,
    dry_run: bool = False,
    filepath: str = DEFAULT_VALIDATION_LEDGER_PATH,
    regime_result: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate a soft-policy rejected candidate for the Active Paper Validation Lane.
    
    Invariant: Hard Safety is 100% Non-Bypassable.
    If hard safety fails, returns VETOED and zero orders are placed.
    If hard safety passes, places a tiny (1-3 share) paper order and logs to validation ledger.
    """
    result = {
        "status": "VETOED",
        "validation_id": f"val_{signal.symbol}_{int(time.time()*1000)}",
        "reason": "",
        "order": None,
        "record": None,
    }

    # 0. HARD NON-BYPASSABLE PAPER-ONLY & FEATURE INVARIANTS
    # Validation lane is strictly prohibited from running against real-money accounts.
    if not getattr(config, "VALIDATION_LANE_ENABLED", False):
        result["reason"] = "HARD_SAFETY_VETO: Validation Lane is disabled in configuration"
        logger.warning("[VALIDATION VETO] %s: %s", signal.symbol, result["reason"])
        return result

    if hasattr(bridge, "is_paper") and not getattr(bridge, "is_paper", False):
        result["reason"] = "HARD_SAFETY_VETO: Validation Lane is strictly prohibited on real-money accounts (is_paper=False)"
        logger.critical("[VALIDATION HARD VETO] %s: %s", signal.symbol, result["reason"])
        return result

    if hasattr(bridge, "assert_account_mode_safety"):
        try:
            bridge.assert_account_mode_safety()
        except Exception as acct_err:
            result["reason"] = f"HARD_SAFETY_VETO: Account mode safety verification failed: {acct_err}"
            logger.critical("[VALIDATION HARD VETO] %s: %s", signal.symbol, result["reason"])
            return result

    # 1. Reject if not a valid soft policy rejection or if regime is DATA_DEGRADED
    if is_hard_safety_failure(policy_reason_code, policy_reason_msg):
        result["reason"] = f"Hard safety failure cannot enter validation: {policy_reason_code}"
        logger.warning("[VALIDATION VETO] %s: %s", signal.symbol, result["reason"])
        return result

    if str(regime_mode).upper().strip() == "DATA_DEGRADED":
        result["reason"] = "Hard safety failure: regime is DATA_DEGRADED (fail-closed)"
        logger.warning("[VALIDATION VETO] %s: %s", signal.symbol, result["reason"])
        return result

    # 1b. Defense-in-depth: SOXX Benchmark Gate for VWAP_BOUNCE Longs
    strat_str = getattr(signal.strategy, "value", str(signal.strategy)).upper()
    if strat_str == "VWAP_BOUNCE" and signal.side.upper().strip() in ("BUY", "LONG"):
        blocked_regimes = getattr(config, "VWAP_BOUNCE_BLOCKED_REGIMES", ("CAUTION", "RISK_OFF", "DATA_DEGRADED"))
        if regime_mode in blocked_regimes:
            soxx_data = getattr(regime_result, "data", {}).get("SOXX", {}) if regime_result and hasattr(regime_result, "data") else {}
            if not isinstance(soxx_data, dict):
                soxx_data = {}
            soxx_pct = soxx_data.get("pct")
            soxx_status = soxx_data.get("status")
            if soxx_status is None:
                soxx_status = "VALID" if soxx_pct is not None else "MISSING"
            threshold = getattr(config, "VWAP_BOUNCE_SOXX_VETO_THRESHOLD", -1.50)
            if soxx_pct is None or soxx_status != "VALID":
                result["reason"] = f"SOXX_TECH_LONG_VETO: SOXX benchmark data unavailable or invalid ({soxx_status})"
                logger.warning("[VALIDATION SOXX VETO] %s: %s", signal.symbol, result["reason"])
                return result
            elif soxx_pct <= threshold:
                result["reason"] = f"SOXX_TECH_LONG_VETO: SOXX {soxx_pct:.2f}% <= {threshold:.2f}%"
                logger.warning("[VALIDATION SOXX VETO] %s: %s", signal.symbol, result["reason"])
                return result

    # 1c. Defense-in-depth: Broker Position Side Arbitration (Patch C)
    if bridge is not None:
        try:
            get_pos_fn = getattr(bridge, "get_position", None)
            if callable(get_pos_fn):
                pos = get_pos_fn(signal.symbol)
                if pos is not None:
                    pos_qty = float(getattr(pos, "qty", 0.0) or (pos.get("qty", 0.0) if isinstance(pos, dict) else 0.0))
                    side_norm = signal.side.upper().strip()
                    if pos_qty > 0 and side_norm in ("SELL", "SHORT"):
                        result["reason"] = f"OPPOSING_POSITION_VETO: Cannot submit validation SELL bracket on existing LONG {signal.symbol} ({pos_qty} shs)"
                        logger.warning("[VALIDATION ARBITRATION VETO] %s", result["reason"])
                        return result
                    elif pos_qty < 0 and side_norm in ("BUY", "LONG"):
                        result["reason"] = f"OPPOSING_POSITION_VETO: Cannot submit validation BUY bracket on existing SHORT {signal.symbol} ({pos_qty} shs)"
                        logger.warning("[VALIDATION ARBITRATION VETO] %s", result["reason"])
                        return result
        except Exception as pos_chk_err:
            logger.debug("Validation position check failed: %s", pos_chk_err)

    # 2. Strict Hard Safety Verification Gate
    now = time.time()
    decision_feed = getattr(signal, "decision_feed", None) or "iex"
    bar_ts = getattr(signal, "bar_timestamp", None)
    strat_obj = getattr(signal, "strategy", None)
    strat_key = getattr(strat_obj, "name", getattr(strat_obj, "value", str(strat_obj or ""))).upper()
    is_candle_strategy = strat_key in {
        "ORB", "VWAP_BOUNCE", "GAP_AND_GO", "MOMENTUM_SCALP", "CANDLESTICK", "MEAN_REVERSION"
    }
    order_intent_raw = getattr(signal, "order_intent", "ENTRY")
    order_intent_str = getattr(order_intent_raw, "value", str(order_intent_raw)).upper()
    is_candle_entry = is_candle_strategy and (order_intent_str == "ENTRY")

    if is_candle_entry:
        if bar_ts is None or str(bar_ts).strip() == "":
            result["reason"] = "Hard Safety Quote Gate: BAR_TIMESTAMP_MISSING (candle-based entry signal missing bar_timestamp)"
            logger.warning("[VALIDATION HARD SAFETY VETO] %s: Candle entry signal missing bar_timestamp.", signal.symbol)
            return result

    if bar_ts is not None:
        bar_dur_raw = getattr(signal, "bar_duration_seconds", getattr(config, "ENTRY_BAR_DURATION_SECONDS", 300.0))
        if bar_dur_raw is None:
            result["reason"] = "Hard Safety Quote Gate: BAR_DURATION_INVALID (bar_duration_seconds missing)"
            logger.warning("[VALIDATION HARD SAFETY VETO] %s: bar_duration_seconds missing", signal.symbol)
            return result
        try:
            bar_dur_f = float(bar_dur_raw)
            if not math.isfinite(bar_dur_f) or bar_dur_f <= 0.0:
                result["reason"] = f"Hard Safety Quote Gate: BAR_DURATION_INVALID (bar_duration_seconds={bar_dur_raw} must be > 0)"
                logger.warning("[VALIDATION HARD SAFETY VETO] %s: bar_duration_seconds=%s must be > 0", signal.symbol, bar_dur_raw)
                return result
        except (TypeError, ValueError):
            result["reason"] = f"Hard Safety Quote Gate: BAR_DURATION_INVALID (could not parse {bar_dur_raw})"
            logger.warning("[VALIDATION HARD SAFETY VETO] %s: could not parse bar_duration_seconds=%s", signal.symbol, bar_dur_raw)
            return result
    else:
        bar_dur_f = None

    try:
        quote_res = validate_execution_quote(
            bridge=bridge,
            symbol=signal.symbol,
            side=signal.side,
            decision_price=signal.entry_price,
            max_spread_pct=getattr(config, "MAX_ACCEPTABLE_SPREAD_PCT", 1.0),
            max_deviation_pct=getattr(config, "MAX_DECISION_DEVIATION_PCT", 1.5),
            max_age_seconds=getattr(config, "MAX_QUOTE_AGE_SECONDS", 10.0),
            decision_feed=decision_feed,
            bar_timestamp=bar_ts,
            max_bar_age_seconds=getattr(config, "MAX_BAR_STALENESS_SECONDS", 300.0),
            require_feed_concordance=True,  # STRICT CONCORDANCE MANDATORY
            bar_duration_seconds=bar_dur_f,
            max_entry_bar_close_age_seconds=float(getattr(config, "ENTRY_BAR_CLOSE_GRACE_SECONDS", 60.0)),
        )
    except Exception as exc:
        result["reason"] = f"Execution quote safety check crashed: {exc}"
        logger.exception("[VALIDATION SAFETY CRASH] %s", signal.symbol)
        return result

    if not quote_res.valid:
        result["reason"] = f"Hard Safety Quote Gate: {quote_res.reason}"
        logger.warning("[VALIDATION HARD SAFETY VETO] %s: %s", signal.symbol, quote_res.reason)
        return result

    # 3. Strict Geometry and Noise Floor Check
    raw_geo_ok, raw_geo_err = validate_executable_geometry(signal)
    if not raw_geo_ok:
        result["reason"] = f"Hard Safety Geometry Gate: {raw_geo_err}"
        logger.warning("[VALIDATION HARD SAFETY GEOMETRY VETO] %s: %s", signal.symbol, raw_geo_err)
        return result

    quote = quote_res.quote
    exec_price = quote.executable_price(signal.side) if quote else signal.entry_price
    tick_size = 0.01 if exec_price >= 1.0 else 0.0001
    precision = 2 if tick_size == 0.01 else 4
    limit_price = round(exec_price, precision)

    side_norm = signal.side.upper().strip()
    if side_norm in ("BUY", "LONG"):
        stop_distance = max(signal.entry_price - signal.stop_price, getattr(config, "NOISE_FLOOR_ABS_MIN", 0.08))
        target_distance = max(signal.target_price - signal.entry_price, stop_distance * 1.50)
        stop_price = round(limit_price - stop_distance, precision)
        target_price = round(limit_price + target_distance, precision)
    else:
        stop_distance = max(signal.stop_price - signal.entry_price, getattr(config, "NOISE_FLOOR_ABS_MIN", 0.08))
        target_distance = max(signal.entry_price - signal.target_price, stop_distance * 1.50)
        stop_price = round(limit_price + stop_distance, precision)
        target_price = round(limit_price - target_distance, precision)

    val_sig = TradeSignal(
        symbol=signal.symbol,
        strategy=signal.strategy,
        side=signal.side,
        confidence=signal.confidence,
        entry_price=limit_price,
        stop_price=stop_price,
        target_price=target_price,
        atr=getattr(signal, "atr", 1.0),
        risk_per_share=abs(limit_price - stop_price),
        reason=f"Validation Lane: {getattr(signal, 'reason', '')}",
        decision_feed=getattr(signal, "decision_feed", None),
        bar_timestamp=getattr(signal, "bar_timestamp", None),
    )
    geo_ok, geo_err = validate_executable_geometry(
        val_sig,
        atr_hard_veto_line=getattr(config, "NOISE_FLOOR_ABS_MIN", 0.08),
    )
    if not geo_ok:
        result["reason"] = f"Hard Safety Geometry Gate: {geo_err}"
        logger.warning("[VALIDATION HARD SAFETY GEOMETRY VETO] %s: %s", signal.symbol, geo_err)
        return result

    # 4. Construct Validation Trade Record
    shares = max(1, min(int(fixed_shares), 3))
    val_id = f"val_{signal.symbol}_{signal.strategy}_{int(now)}"
    quote_mid = (quote.bid + quote.ask) / 2.0 if quote else exec_price
    record = ValidationTradeRecord(
        validation_id=val_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        symbol=signal.symbol.upper().strip(),
        strategy=getattr(signal.strategy, "value", str(signal.strategy)),
        side=signal.side.upper().strip(),
        confidence=float(signal.confidence),
        regime=str(regime_mode),
        market_texture=str(market_texture),
        terminal_gate=str(policy_stage),
        rejection_reason=str(policy_reason_msg),
        decision_price=float(signal.entry_price),
        decision_feed=str(decision_feed),
        bar_timestamp=str(bar_ts or ""),
        quote_mid=round(quote_mid, 4),
        quote_feed=str(quote.feed_identity if quote else ""),
        quote_timestamp=float(quote.timestamp if quote else now),
        stop_price=stop_price,
        target_price=target_price,
        hard_safety_pass=True,
        policy_rejected=True,
        validation_only=True,
        status="SUBMITTED" if not dry_run else "DRY_RUN",
        qty=float(shares),
    )

    # 5. Order Placement (1 Share Fixed Tiny Sizing)
    client_order_id = f"val_{signal.symbol.lower()}_{int(now % 100000)}"

    order_payload = {
        "symbol": signal.symbol,
        "shares": shares,
        "side": signal.side,
        "limit_price": limit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "client_order_id": client_order_id,
        "validation_only": True,
    }

    if not dry_run:
        # Pre-check: Verify validation release authorization locally before attempting broker submission
        can_release = True
        if hasattr(bridge, "is_validation_release_authorized"):
            can_release = bool(bridge.is_validation_release_authorized())
        elif hasattr(bridge, "assert_machine_release_gate"):
            token = getattr(bridge, "validation_release_token", None)
            release_env = os.environ.get("PRINCIPAL_RELEASE_GRANTED", "false").strip().lower() in ("1", "true", "yes")
            release_attr = getattr(bridge, "principal_release_granted", False)
            can_release = bool((token is not None and not token.consumed) or release_env or release_attr)

        if not can_release:
            logger.info(
                "[VALIDATION LANE SKIP] [SHADOW RECORDING ONLY] Candidate %s was already rejected by production gates. Validation-only broker execution skipped (principal release absent). Telemetry quarantined to shadow log.",
                signal.symbol,
            )
            result["status"] = "VALIDATION_RELEASE_BLOCKED"
            result["reason"] = "Validation release authorization unavailable (pre-check skip)"
            record.status = "PRE_CHECK_SKIPPED"
            log_validation_record(record, filepath=filepath)
            return result

        place_fn = getattr(bridge, "place_bracket_order", None) or getattr(bridge, "place_order", None)
        if place_fn is not None:
            try:
                import inspect
                sig_params = {}
                try:
                    sig_params = inspect.signature(place_fn).parameters
                except Exception:
                    pass

                kwargs = {
                    "symbol": signal.symbol,
                    "side": signal.side,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "client_order_id": client_order_id,
                }
                if "qty" in sig_params:
                    kwargs["qty"] = shares
                else:
                    kwargs["shares"] = shares

                if "entry_price" in sig_params:
                    kwargs["entry_price"] = limit_price
                else:
                    kwargs["limit_price"] = limit_price

                placed = place_fn(**kwargs)
                if isinstance(placed, dict):
                    record.entry_order_id = placed.get("order_id") or client_order_id
                else:
                    record.entry_order_id = getattr(placed, "id", None) or client_order_id

                try:
                    from strategies.order_registry import register_order_expected_price
                    if record.entry_order_id:
                        register_order_expected_price(
                            order_id=record.entry_order_id,
                            expected_price=limit_price,
                            side=signal.side,
                            order_type="ENTRY",
                            submit_ts=now,
                            symbol=signal.symbol,
                            strategy=getattr(signal.strategy, "value", str(signal.strategy)),
                            is_validation=True,
                            client_order_id=client_order_id,
                        )
                except Exception as reg_err:
                    logger.debug("Failed to register validation order in order_registry: %s", reg_err)

                order_payload["order_response"] = str(placed)
                result["status"] = "PLACED"
            except Exception as place_err:
                logger.error("[VALIDATION ORDER ERROR] Failed to place order for %s: %s", signal.symbol, place_err)
                record.status = "ORDER_ERROR"
                result["status"] = "ERROR"
                result["reason"] = f"Broker order error: {place_err}"
                log_validation_record(record, filepath=filepath)
                return result
        else:
            result["status"] = "DRY_RUN"
    else:
        result["status"] = "DRY_RUN"

    # 6. Append to Isolated Validation Ledger
    log_validation_record(record, filepath=filepath)
    result["order"] = order_payload
    result["record"] = record
    result["reason"] = f"Candidate validated (Hard Safety PASS, Policy REJECT: {policy_stage})"
    logger.info(
        "[VALIDATION LANE %s] %s %s %d shs @ $%.2f (Stop $%.2f, TP $%.2f) [Conf: %.3f, Policy: %s]",
        result["status"], signal.side, signal.symbol, shares, limit_price, stop_price, target_price,
        signal.confidence, policy_stage
    )
    return result


def get_validation_daily_pnl(
    session_date: Optional[str] = None,
    filepath: str = DEFAULT_VALIDATION_LEDGER_PATH,
) -> float:
    """Compute realized P&L from closed validation trades for a given session date (YYYY-MM-DD).
    If session_date is None, uses today's Eastern date.
    """
    if session_date is None:
        try:
            from strategies.session import now_eastern
            session_date = str(now_eastern().date())
        except Exception:
            session_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    records = dedupe_on_read(filepath)
    val_pnl = 0.0
    for rec in records:
        ts = rec.get("timestamp", "")
        rec_date = ts[:10] if ts else ""
        if rec_date == session_date:
            pnl_val = rec.get("realized_pnl")
            if pnl_val is None:
                pnl_val = rec.get("pnl")
            if pnl_val is not None:
                try:
                    val_pnl += float(pnl_val)
                except (ValueError, TypeError):
                    pass
    return round(val_pnl, 4)


def get_production_risk_equity(bridge, include_validation: bool = False) -> Tuple[float, float]:
    """Returns (governing_equity, current_production_pnl) strictly isolated from validation lane P&L.

    If include_validation is False:
        - governing_equity excludes validation loss/gain (adjusts broker net_liq by -validation_daily_pnl)
        - current_production_pnl = raw_daily_pnl - validation_daily_pnl
    """
    import config
    raw_equity = 0.0
    if bridge is not None:
        try:
            if hasattr(bridge, "get_net_liquidation"):
                raw_equity = float(bridge.get_net_liquidation())
            else:
                raw_equity = float(getattr(bridge, "equity", getattr(config, "PDT_MIN_EQUITY", 25000.0)))
        except Exception:
            raw_equity = float(getattr(config, "PDT_MIN_EQUITY", 25000.0))
    else:
        raw_equity = float(getattr(config, "PDT_MIN_EQUITY", 25000.0))

    raw_pnl = 0.0
    if bridge is not None:
        try:
            if hasattr(bridge, "account_daily_pnl"):
                import inspect
                sig = inspect.signature(bridge.account_daily_pnl)
                if "include_validation" in sig.parameters:
                    raw_pnl = float(bridge.account_daily_pnl(include_validation=True))
                else:
                    raw_pnl = float(bridge.account_daily_pnl())
            elif hasattr(bridge, "daily_pnl"):
                raw_pnl = float(bridge.daily_pnl())
            else:
                from strategies.trade_journal import today_pnl
                raw_pnl = today_pnl()
        except Exception:
            from strategies.trade_journal import today_pnl
            raw_pnl = today_pnl()
    else:
        from strategies.trade_journal import today_pnl
        raw_pnl = today_pnl()

    if include_validation:
        return raw_equity, raw_pnl

    val_pnl = get_validation_daily_pnl()
    governing_equity = round(raw_equity - val_pnl, 2)
    prod_pnl = round(raw_pnl - val_pnl, 2)
    return governing_equity, prod_pnl

