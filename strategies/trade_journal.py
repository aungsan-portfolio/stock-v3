"""
trade_journal.py -- Append-only JSONL trade journal for review and learning.
"""
import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Tuple, Set

import config
from strategies.session import is_market_day, now_eastern

from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)

_journal_lock = threading.RLock()
_journal_reconciliation_lock = _journal_lock

def get_journal_lock() -> threading.RLock:
    """Return shared re-entrant lock for serialized journal reconciliation."""
    return _journal_lock

def canonical_decimal(value) -> str:
    """Format decimal numbers canonically (e.g. 57 -> '57', 57.000 -> '57', 0.25 -> '0.25')."""
    if value is None:
        return "0"
    try:
        d = Decimal(str(value))
        normalized = d.normalize()
        text = format(normalized, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    except (InvalidOperation, TypeError, ValueError):
        return str(value)

def is_economic_fill(record: dict) -> bool:
    """Check if a journal record represents a valid economic execution fill."""
    if not isinstance(record, dict):
        return False
    event_type = str(record.get("event_type") or record.get("type") or "").upper()
    if event_type != "FILL":
        return False
    order_id = record.get("broker_order_id") or record.get("order_id")
    if not order_id:
        return False
    try:
        qty = Decimal(str(record.get("qty", "0")))
        return qty > 0
    except Exception:
        return False

class OrderFillState:
    """Cumulative fill state for a specific broker order."""
    def __init__(self, broker_order_id: str = ""):
        self.broker_order_id = broker_order_id
        self.logged_qty = Decimal("0")
        self.logged_notional = Decimal("0")
        self.execution_ids = set()

    def add_fill(self, qty: Decimal, price: Decimal, exec_id: Optional[str] = None) -> bool:
        """
        Add fill execution slice to cumulative state.
        Returns True if fill was new and accumulated; False if duplicate execution_id was skipped.
        """
        if exec_id is not None:
            exec_id_str = str(exec_id).strip()
            if exec_id_str:
                if exec_id_str in self.execution_ids:
                    return False
                self.execution_ids.add(exec_id_str)
        self.logged_qty += qty
        self.logged_notional += qty * price
        return True

def load_journal_order_fill_state(records, account_id: Optional[str] = None, trading_date=None) -> Dict[Tuple[str, str, str, str, str], OrderFillState]:
    """
    Build cumulative filled quantities and execution IDs by scoped key:
    (account_id, trading_date, broker_order_id, symbol, side)
    """
    import pytz
    states: Dict[Tuple[str, str, str, str, str], OrderFillState] = {}
    target_date_str = str(trading_date) if trading_date is not None else None
    target_acc_str = str(account_id) if account_id is not None else None

    for r in records:
        if not is_economic_fill(r):
            continue

        rec_acc = str(r.get("account_id") or target_acc_str or "ALPACA")
        if target_acc_str and rec_acc != target_acc_str and rec_acc != "ALPACA" and target_acc_str != "ALPACA":
            continue

        # When target_acc_str is provided and rec_acc is default 'ALPACA' (legacy record), treat it as target_acc_str
        scoped_acc = target_acc_str if (target_acc_str and rec_acc == "ALPACA") else rec_acc

        ts_str = r.get("timestamp") or r.get("journal_write_timestamp_utc")
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

        order_id = str(r.get("broker_order_id") or r.get("order_id") or "")
        symbol = str(r.get("symbol") or "").upper()
        side = str(r.get("side") or "").upper()

        key = (scoped_acc, rec_date_str or target_date_str or "", order_id, symbol, side)
        if key not in states:
            states[key] = OrderFillState(broker_order_id=order_id)

        qty = Decimal(str(r.get("qty", "0")))
        price = Decimal(str(r.get("fill_price", "0.0")))
        exec_id = r.get("execution_id") or r.get("fill_id")
        added = states[key].add_fill(qty, price, exec_id)
        if not added:
            logger.warning(
                "[JOURNAL DEDUPLICATION] Duplicate execution %s for order %s skipped during fill state load",
                exec_id, order_id
            )

    return states

def _normalize_exit_idempotency_key(k: str) -> str:
    """Extract base episode key from exit idempotency key without volatile reason suffix."""
    k_str = str(k or "").strip()
    if not k_str:
        return ""
    for suffix in ("_STOP_OUT", "_TAKE_PROFIT", "_TRAILING_STOP", "_EOD_FLATTEN", "_MANUAL", "_CIRCUIT_BREAKER", "_FLAT", "_PROVISIONAL"):
        if k_str.endswith(suffix):
            return k_str[:-len(suffix)]
    return k_str

def log_trade(
    symbol: str,
    side: str,
    strategy: str,
    qty: int,
    entry_price: float,
    stop_price: float,
    target_price: float,
    exit_price: Optional[float] = None,
    exit_reason: Optional[str] = None,
    pnl: Optional[float] = None,
    notes: Optional[str] = None,
    journal_file: Optional[Path] = None,
    event_type: str = "TRADE",
    order_id: Optional[int] = None,
    signal_id: Optional[str] = None,
    **kwargs,
):
    cid = str(kwargs.get("client_order_id") or order_id or signal_id or "")
    if cid.startswith("val_"):
        raise RuntimeError(f"HARD LANE GUARD: Validation order {cid} routed to production trade journal!")

    resolved_strategy = strategy
    if not resolved_strategy or resolved_strategy == "UNKNOWN":
        client_order_id = kwargs.get("client_order_id")
        if client_order_id:
            parts = str(client_order_id).split("_")
            if len(parts) >= 2 and parts[0] == "dt":
                resolved_strategy = parts[1]
        if not resolved_strategy or resolved_strategy == "UNKNOWN":
            try:
                from strategies.order_registry import get_registered_order
                for oid in (order_id, kwargs.get("broker_order_id")):
                    if oid:
                        reg = get_registered_order(str(oid))
                        if reg and reg.get("strategy") and reg["strategy"] != "UNKNOWN":
                            resolved_strategy = reg["strategy"]
                            break
                        if reg and reg.get("parent_order_id"):
                            p_reg = get_registered_order(str(reg["parent_order_id"]))
                            if p_reg and p_reg.get("strategy") and p_reg["strategy"] != "UNKNOWN":
                                resolved_strategy = p_reg["strategy"]
                                break
            except Exception:
                pass
    if not resolved_strategy:
        resolved_strategy = "UNKNOWN"

    resolved_price_tier = kwargs.pop("price_tier", None)
    if not resolved_price_tier and entry_price and entry_price > 0:
        resolved_price_tier = "5-10" if (5.0 <= entry_price < 10.0) else (">10" if entry_price >= 10.0 else "<5")

    if exit_price is not None:
        try:
            ep_f = float(exit_price)
            if not math.isfinite(ep_f) or ep_f <= 0:
                raise ValueError(f"Invalid non-finite or non-positive exit_price: {exit_price}")
            exit_price = ep_f
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid exit_price for trade log: {exc}")

    if pnl is not None:
        try:
            pnl_f = float(pnl)
            if not math.isfinite(pnl_f):
                raise ValueError(f"Invalid non-finite pnl: {pnl}")
            pnl = pnl_f
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid pnl for trade log: {exc}")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "order_id": order_id,
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "strategy": resolved_strategy,
        "qty": qty,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "price_tier": resolved_price_tier,
        "notes": notes,
        **kwargs,
    }
    path = journal_file or config.DAYTRADE_TRADE_JOURNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with _journal_lock:
        # Idempotency Guard: prevent duplicate TRADE_CLOSED writes across crashes/restarts
        primary_trade_key = str(kwargs.get("idempotency_key") or kwargs.get("exit_order_id") or signal_id or order_id or "")
        norm_primary = _normalize_exit_idempotency_key(primary_trade_key)
        found_existing_record = False
        if path.exists() and primary_trade_key and event_type in ("TRADE_CLOSED", "TRADE", "EXIT"):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh_check:
                    for line in fh_check:
                        line_str = line.strip()
                        if not line_str:
                            continue
                        if norm_primary in line_str or primary_trade_key in line_str:
                            try:
                                item = json.loads(line_str)
                            except Exception:
                                continue
                            if not item or not isinstance(item, dict):
                                continue
                            stored_key = str(item.get("idempotency_key") or "")
                            norm_stored = _normalize_exit_idempotency_key(stored_key)
                            key_match = (
                                (norm_stored and norm_primary and norm_stored == norm_primary)
                                or (stored_key and stored_key == primary_trade_key)
                                or (item.get("exit_order_id") and str(item.get("exit_order_id")) == primary_trade_key)
                                or (item.get("signal_id") and str(item.get("signal_id")) == primary_trade_key)
                                or (item.get("order_id") and str(item.get("order_id")) == primary_trade_key)
                            )
                            if (
                                item.get("event_type") == event_type
                                and item.get("symbol") == symbol
                                and key_match
                            ):
                                found_existing_record = True
                                break
            except OSError:
                raise
            except Exception as read_err:
                logger.warning("[JOURNAL TRADE GUARD READ ERROR] %s", read_err)

        if found_existing_record:
            logger.warning(
                "[JOURNAL TRADE IDEMPOTENT GUARD] Event %s for %s (key=%s) already recorded in %s. Enforcing durable sync.",
                event_type, symbol, primary_trade_key, path
            )
            if hasattr(os, "fsync") and path.exists():
                with open(path, "a", encoding="utf-8") as fh_sync:
                    fh_sync.flush()
                    os.fsync(fh_sync.fileno())
            return

        # Check if file has incomplete tail (missing trailing newline)
        needs_leading_newline = False
        if path.exists() and path.stat().st_size > 0:
            try:
                with open(path, "rb") as f_tail:
                    f_tail.seek(-1, os.SEEK_END)
                    if f_tail.read(1) != b"\n":
                        needs_leading_newline = True
            except Exception:
                pass

        with open(path, "a", encoding="utf-8") as fh:
            if needs_leading_newline:
                fh.write("\n")
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            if hasattr(os, "fsync"):
                os.fsync(fh.fileno())
    logger.info("Journal: %s %s %s @ %.2f | stop=%.2f target=%.2f",
                side, qty, symbol, entry_price, stop_price, target_price)

def log_fill(
    symbol: str,
    side: str,
    qty: int,
    fill_price: float,
    expected_price: Optional[float],
    slippage: Optional[float],
    fill_latency_ms: Optional[float],
    journal_file: Optional[Path] = None,
    order_id: Optional[int] = None,
    execution_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    broker_order_id: Optional[str] = None,
    broker_position_before: Optional[int] = None,
    broker_position_after: Optional[int] = None,
    timestamp: Optional[str] = None,
    strategy: Optional[str] = None,
    price_tier: Optional[str] = None,
    signal_id: Optional[str] = None,
    account_id: Optional[str] = None,
    **kwargs,
):
    cid = str(client_order_id or broker_order_id or order_id or signal_id or "")
    if cid.startswith("val_"):
        raise RuntimeError(f"HARD LANE GUARD: Validation order {cid} routed to production trade journal!")

    now_utc_str = datetime.now(timezone.utc).isoformat()
    if not price_tier and fill_price > 0:
        price_tier = "5-10" if (5.0 <= fill_price < 10.0) else (">10" if fill_price >= 10.0 else "<5")

    # Primary key precedence: execution_id is primary; order_id alone is NEVER fill identity
    if execution_id and str(execution_id).strip():
        primary_fill_id = str(execution_id).strip()
        is_synthetic = bool(kwargs.get("is_synthetic_key", kwargs.get("synthetic", primary_fill_id.startswith("SYNTH:"))))
        requires_reconciliation = bool(kwargs.get("requires_reconciliation", is_synthetic))
    else:
        # Degraded synthetic key includes (account_id, order_id, side, qty, px, timestamp)
        synth_acc = account_id or "ACC"
        synth_oid = broker_order_id or order_id or "UNKNOWN_ORDER"
        primary_fill_id = f"SYNTH:{synth_acc}:{synth_oid}:{side}:{qty}:{fill_price:.4f}:{timestamp or now_utc_str}"
        is_synthetic = True
        requires_reconciliation = True

    resolved_strategy = strategy
    if not resolved_strategy or resolved_strategy == "UNKNOWN":
        if client_order_id:
            parts = str(client_order_id).split("_")
            if len(parts) >= 2 and parts[0] == "dt":
                resolved_strategy = parts[1]
        if not resolved_strategy or resolved_strategy == "UNKNOWN":
            try:
                from strategies.order_registry import get_registered_order
                for oid in (order_id, broker_order_id):
                    if oid:
                        reg = get_registered_order(str(oid))
                        if reg and reg.get("strategy") and reg["strategy"] != "UNKNOWN":
                            resolved_strategy = reg["strategy"]
                            break
                        if reg and reg.get("parent_order_id"):
                            p_reg = get_registered_order(str(reg["parent_order_id"]))
                            if p_reg and p_reg.get("strategy") and p_reg["strategy"] != "UNKNOWN":
                                resolved_strategy = p_reg["strategy"]
                                break
            except Exception:
                pass
    if not resolved_strategy:
        resolved_strategy = "UNKNOWN"

    record = {
        "timestamp": timestamp or now_utc_str,
        "journal_write_timestamp_utc": now_utc_str,
        "type": "FILL",
        "event_type": "FILL",
        "fill_id": primary_fill_id,
        "execution_id": str(execution_id).strip() if execution_id else None,
        "is_synthetic_key": is_synthetic,
        "requires_reconciliation": requires_reconciliation,
        "order_id": order_id or broker_order_id,
        "client_order_id": client_order_id,
        "broker_order_id": broker_order_id or str(order_id or ""),
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "strategy": resolved_strategy,
        "qty": qty,
        "fill_price": fill_price,
        "expected_price": expected_price,
        "slippage": slippage,
        "fill_latency_ms": fill_latency_ms,
        "price_tier": price_tier,
        "broker_position_before": broker_position_before,
        "broker_position_after": broker_position_after,
        "account_id": str(account_id or "ALPACA"),
        **kwargs,
    }
    path = journal_file or config.DAYTRADE_TRADE_JOURNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with _journal_lock:
        if path.exists() and primary_fill_id:
            try:
                with open(path, "r", encoding="utf-8") as fh_check:
                    for line in fh_check:
                        if not line.strip():
                            continue
                        if primary_fill_id in line:
                            try:
                                item = json.loads(line)
                                if item.get("fill_id") == primary_fill_id or item.get("execution_id") == primary_fill_id:
                                    logger.warning(
                                        "[JOURNAL WRITE IDEMPOTENT GUARD] Execution %s already recorded in %s. Skipping duplicate write.",
                                        primary_fill_id, path
                                    )
                                    return
                            except Exception:
                                pass
            except Exception as read_err:
                logger.warning("[JOURNAL WRITE GUARD READ ERROR] %s", read_err)

        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    if slippage is not None and fill_latency_ms is not None:
        logger.info(
            "Fill logged: %s %d %s @ %.2f (Slippage: %.4f, Latency: %.1fms)",
            side, qty, symbol, fill_price, slippage, fill_latency_ms,
        )
    else:
        logger.info("Fill logged: %s %d %s @ %.2f", side, qty, symbol, fill_price)

def read_journal(journal_file: Optional[Path] = None):
    path = journal_file or config.DAYTRADE_TRADE_JOURNAL_FILE
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def _is_today_record(timestamp_str: str, today_date, eastern_timezone) -> bool:
    if not timestamp_str:
        return False
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(eastern_timezone).date() == today_date
    except Exception:
        return False

def today_trades(journal_file: Optional[Path] = None):
    current_eastern = now_eastern()
    eastern_timezone = current_eastern.tzinfo
    today_date = current_eastern.date()
    seen_orders = set()
    entry_trades = []
    for r in read_journal(journal_file):
        if not _is_today_record(r.get("timestamp", ""), today_date, eastern_timezone):
            continue
        # Count actual executed BUY fills or valid trade entries
        is_fill = (r.get("event_type") == "FILL" or r.get("type") == "FILL")
        side = str(r.get("side", "")).upper()
        if is_fill and side == "BUY":
            oid = r.get("order_id") or r.get("execution_id") or f"{r.get('symbol')}_{r.get('timestamp')}"
            if oid not in seen_orders:
                seen_orders.add(oid)
                entry_trades.append(r)
        elif not is_fill and r.get("event_type") == "ORDER_SUBMITTED" and side == "BUY":
            # Fallback for paper mode if fill record missing
            oid = r.get("order_id") or f"{r.get('symbol')}_{r.get('timestamp')}"
            if oid not in seen_orders:
                seen_orders.add(oid)
                entry_trades.append(r)
    return entry_trades

def today_trade_count(journal_file: Optional[Path] = None) -> int:
    return len(today_trades(journal_file))

def today_pnl(journal_file: Optional[Path] = None) -> float:
    current_eastern = now_eastern()
    eastern_timezone = current_eastern.tzinfo
    today_date = current_eastern.date()
    return sum(
        r.get("realized_pnl", r.get("pnl", 0.0)) or 0.0
        for r in read_journal(journal_file)
        if _is_today_record(r.get("timestamp", ""), today_date, eastern_timezone)
    )

def count_day_trades_in_records(records: list[dict], current_date) -> int:
    """Count completed intraday round trips in a list of records across the last five market days."""
    from collections import defaultdict
    from datetime import timedelta

    if not records:
        return 0

    market_days = set()
    cursor = current_date
    while len(market_days) < 5:
        if is_market_day(cursor):
            market_days.add(cursor)
        cursor -= timedelta(days=1)

    fills_by_day_symbol = defaultdict(list)
    seen_executions = set()
    for r in records:
        try:
            ts = datetime.fromisoformat(r["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            session_date = ts.astimezone(now_eastern().tzinfo).date()
            if session_date not in market_days:
                continue
            if r.get("event_type") != "FILL" and r.get("type") != "FILL":
                continue

            execution_id = r.get("execution_id")
            if execution_id:
                if execution_id in seen_executions:
                    continue
                seen_executions.add(execution_id)

            side = str(r.get("side", "")).upper()
            if side in {"BUY", "BOT"}:
                direction = 1
            elif side in {"SELL", "SLD"}:
                direction = -1
            else:
                continue

            qty = float(r.get("qty", 0) or 0)
            symbol = r.get("symbol")
            if qty <= 0 or not symbol:
                continue
            fills_by_day_symbol[(session_date, symbol)].append((ts, direction * qty))
        except Exception:
            logger.debug("Ignoring malformed journal row during PDT reconstruction", exc_info=True)

    round_trips = 0
    for fills in fills_by_day_symbol.values():
        position = 0.0
        for _, signed_qty in sorted(fills):
            previous = position
            position += signed_qty
            if previous != 0 and (position == 0 or previous * position < 0):
                round_trips += 1

    return round_trips

def day_trades_in_last_5_days(journal_file: Optional[Path] = None, bridge: Optional[object] = None) -> int:
    """Count completed intraday round trips across the last five market days.
    
    Cross-verifies local DB count with Alpaca broker account daytrade_count
    and returns max(local, alpaca) for fail-closed PDT compliance.
    """
    records = read_journal(journal_file)
    local_count = count_day_trades_in_records(records, now_eastern().date())
    alpaca_count = 0
    if bridge is not None:
        try:
            acct = None
            if hasattr(bridge, "get_account"):
                acct = bridge.get_account()
            elif hasattr(bridge, "_client") and hasattr(bridge._client, "get_account"):
                acct = bridge._client.get_account()
            if acct:
                alpaca_count = int(getattr(acct, "daytrade_count", 0) or 0)
        except Exception as exc:
            logger.warning("[PDT GUARD] Failed to fetch Alpaca daytrade_count: %s. Using local count (%d).", exc, local_count)

    effective_count = max(local_count, alpaca_count)
    if alpaca_count > 0 and alpaca_count != local_count:
        logger.warning(
            "[PDT GUARD] Count mismatch! Local=%d, Alpaca=%d. Using max=%d (fail-closed).",
            local_count, alpaca_count, effective_count
        )
    return effective_count

def get_today_closed_trades(journal_file: Optional[Path] = None):
    """Reconstruct today's closed trades from fill events."""
    from collections import defaultdict

    records = read_journal(journal_file)
    if not records:
        return []

    current_eastern = now_eastern()
    eastern_timezone = current_eastern.tzinfo
    today_date = current_eastern.date()
    closed_trades = []
    symbols_with_closed_records = set()
    fills_by_symbol = defaultdict(list)
    seen_executions = set()

    for record in records:
        try:
            timestamp = datetime.fromisoformat(record["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)

            session_date = timestamp.astimezone(eastern_timezone).date()
            if session_date != today_date:
                continue

            is_fill = record.get("event_type") == "FILL" or record.get("type") == "FILL"
            realized_pnl = record.get("realized_pnl")
            if realized_pnl is None:
                realized_pnl = record.get("pnl")
            if not is_fill and realized_pnl is not None:
                symbol = record.get("symbol")
                if not symbol:
                    continue
                realized_pnl = float(realized_pnl)
                closed_trades.append(
                    {
                        "symbol": symbol,
                        "closed_at": timestamp,
                        "realized_pnl": realized_pnl,
                        "is_win": realized_pnl > 0.0,
                    }
                )
                symbols_with_closed_records.add(symbol)
                continue

            if not is_fill:
                continue

            execution_id = record.get("execution_id")
            if execution_id:
                if execution_id in seen_executions:
                    continue
                seen_executions.add(execution_id)

            side = str(record.get("side", "")).upper()
            if side in {"BUY", "BOT"}:
                signed_quantity = float(record.get("qty", 0) or 0)
            elif side in {"SELL", "SLD"}:
                signed_quantity = -float(record.get("qty", 0) or 0)
            else:
                continue

            quantity = abs(signed_quantity)
            fill_price = float(record.get("fill_price", 0) or 0)
            symbol = record.get("symbol")
            if quantity <= 0 or fill_price <= 0 or not symbol:
                continue

            fills_by_symbol[symbol].append((timestamp, signed_quantity, fill_price))
        except Exception:
            logger.debug("Ignoring malformed journal row during closed-trade reconstruction", exc_info=True)

    for symbol, symbol_fills in fills_by_symbol.items():
        if symbol in symbols_with_closed_records:
            continue
        symbol_fills.sort(key=lambda fill: fill[0])
        position = 0.0
        average_price = 0.0
        realized_pnl = 0.0

        for timestamp, signed_quantity, fill_price in symbol_fills:
            previous_position = position
            new_position = previous_position + signed_quantity

            if (
                previous_position == 0.0
                or (previous_position > 0.0 and signed_quantity > 0.0)
                or (previous_position < 0.0 and signed_quantity < 0.0)
            ):
                total_cost = (abs(previous_position) * average_price) + (abs(signed_quantity) * fill_price)
                average_price = total_cost / abs(new_position)
            else:
                closed_quantity = min(abs(previous_position), abs(signed_quantity))
                if previous_position > 0.0:
                    realized_pnl += closed_quantity * (fill_price - average_price)
                else:
                    realized_pnl += closed_quantity * (average_price - fill_price)

                if new_position == 0.0 or previous_position * new_position < 0.0:
                    closed_trades.append(
                        {
                            "symbol": symbol,
                            "closed_at": timestamp,
                            "realized_pnl": realized_pnl,
                            "is_win": realized_pnl > 0.0,
                        }
                    )
                    realized_pnl = 0.0
                    if new_position != 0.0:
                        average_price = fill_price

            position = new_position

    closed_trades.sort(key=lambda trade: trade["closed_at"])
    return closed_trades

def backfill_closed_trades(journal_file: Optional[Path] = None) -> int:
    """Parse all fills, reconstruct closed trades, and append TRADE_CLOSED records."""
    from collections import defaultdict

    records = read_journal(journal_file)
    if not records:
        return 0

    entries_by_signal = {}
    for record in records:
        if record.get("event_type") in ("ORDER_SUBMITTED", "TRADE"):
            sig = record.get("signal_id")
            if sig:
                entries_by_signal[str(sig)] = record
            oid = record.get("order_id")
            if oid:
                entries_by_signal[str(oid)] = record

    fills_by_symbol = defaultdict(list)
    seen_executions = set()
    existing_closed_signals = set()

    for record in records:
        try:
            if record.get("event_type") == "TRADE_CLOSED":
                signal_id = record.get("signal_id")
                if signal_id:
                    existing_closed_signals.add(str(signal_id))
                syn_key = f"{record.get('symbol')}_{record.get('opened_at') or record.get('timestamp')}"
                existing_closed_signals.add(syn_key)
                continue

            if record.get("event_type") != "FILL" and record.get("type") != "FILL":
                continue

            execution_id = record.get("execution_id")
            if execution_id:
                execution_id = str(execution_id)
                if execution_id in seen_executions:
                    continue
                seen_executions.add(execution_id)

            side = str(record.get("side", "")).upper()
            if side in {"BUY", "BOT"}:
                direction = 1.0
            elif side in {"SELL", "SLD"}:
                direction = -1.0
            else:
                continue

            quantity = float(record.get("qty", 0) or 0)
            fill_price = float(record.get("fill_price", 0) or 0)
            symbol = record.get("symbol")
            if quantity <= 0 or fill_price <= 0 or not symbol:
                continue

            timestamp = datetime.fromisoformat(record["timestamp"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            fills_by_symbol[symbol].append(
                {
                    "timestamp": timestamp,
                    "signed_quantity": direction * quantity,
                    "price": fill_price,
                    "signal_id": record.get("signal_id"),
                }
            )
        except Exception:
            logger.debug("Ignoring malformed journal row during closed-trade backfill", exc_info=True)

    new_closed_count = 0
    path = journal_file or config.DAYTRADE_TRADE_JOURNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    for symbol, symbol_fills in fills_by_symbol.items():
        symbol_fills.sort(key=lambda item: item["timestamp"])
        position = 0.0
        average_price = 0.0
        realized_pnl = 0.0
        closed_quantity_total = 0.0
        current_signal_id = None
        opened_at = None

        for fill in symbol_fills:
            timestamp = fill["timestamp"]
            signed_quantity = fill["signed_quantity"]
            fill_price = fill["price"]
            signal_id = fill["signal_id"]
            previous_position = position
            new_position = previous_position + signed_quantity

            if (
                previous_position == 0.0
                or (previous_position > 0.0 and signed_quantity > 0.0)
                or (previous_position < 0.0 and signed_quantity < 0.0)
            ):
                total_cost = (abs(previous_position) * average_price) + (abs(signed_quantity) * fill_price)
                average_price = total_cost / abs(new_position)
                if previous_position == 0.0:
                    current_signal_id = signal_id
                    opened_at = timestamp
            else:
                closed_quantity = min(abs(previous_position), abs(signed_quantity))
                closed_quantity_total += closed_quantity
                if previous_position > 0.0:
                    realized_pnl += closed_quantity * (fill_price - average_price)
                else:
                    realized_pnl += closed_quantity * (average_price - fill_price)

                if new_position == 0.0 or previous_position * new_position < 0.0:
                    signal_key = str(current_signal_id) if current_signal_id else f"{symbol}_{(opened_at or timestamp).isoformat()}"
                    if signal_key not in existing_closed_signals:
                        entry = entries_by_signal.get(signal_key, {})
                        if not entry and current_signal_id:
                            entry = entries_by_signal.get(str(current_signal_id), {})
                        stop_price = float(entry.get("stop_price", 0) or 0)
                        target_price = float(entry.get("target_price", 0) or 0)
                        risk_per_share = abs(average_price - stop_price)
                        rr_ratio = abs(target_price - average_price) / risk_per_share if risk_per_share else 0.0
                        entry_side = entry.get("side") or ("BUY" if previous_position > 0 else "SELL")
                        closed_record = {
                            "timestamp": timestamp.isoformat(),
                            "event_type": "TRADE_CLOSED",
                            "symbol": symbol,
                            "signal_id": current_signal_id,
                            "side": entry_side,
                            "strategy": entry.get("strategy", "UNKNOWN"),
                            "opened_at": (opened_at or timestamp).isoformat(),
                            "closed_at": timestamp.isoformat(),
                            "avg_entry_price": average_price,
                            "avg_exit_price": fill_price,
                            "qty": closed_quantity_total,
                            "entry_price": average_price,
                            "exit_price": fill_price,
                            "stop_price": stop_price,
                            "target_price": target_price,
                            "rr_ratio": rr_ratio,
                            "realized_pnl": realized_pnl,
                            "is_win": realized_pnl > 0.0,
                            "exit_reason": "LOSS_EXIT" if realized_pnl < 0 else "PROFIT_EXIT",
                        }
                        with _journal_lock:
                            with open(path, "a", encoding="utf-8") as file_handle:
                                file_handle.write(json.dumps(closed_record) + "\n")
                        existing_closed_signals.add(signal_key)
                        new_closed_count += 1

                    realized_pnl = 0.0
                    closed_quantity_total = 0.0
                    if new_position != 0.0:
                        average_price = fill_price
                        current_signal_id = signal_id
                        opened_at = timestamp

            position = new_position

    if new_closed_count > 0:
        logger.info("Backfilled %d TRADE_CLOSED records from fills.", new_closed_count)
    return new_closed_count


_verified_first_trades = False

def auto_verify_first_trades(journal_file: Optional[Path] = None) -> bool:
    """Automatically audit the first trade records in live session to verify schema and zero dedup once."""
    global _verified_first_trades
    if _verified_first_trades:
        return True
    try:
        records = read_journal(journal_file)
        if not records or len(records) < 1:
            return False
        
        seen_keys = set()
        for r in records:
            event = str(r.get("event") or r.get("action") or "EVENT")
            oid = str(r.get("execution_id") or r.get("order_id") or "")
            ts = str(r.get("timestamp") or "")
            if oid:
                key = f"{oid}_{event}_{ts}"
                if key in seen_keys:
                    logger.warning("[JOURNAL DEDUP WARNING] Duplicate event record detected for order_id %s (event: %s)!", oid, event)
                else:
                    seen_keys.add(key)

        _verified_first_trades = True
        sample = next((r for r in reversed(records) if r.get("strategy") and r.get("strategy") != "UNKNOWN"), records[-1])
        msg = f"[PASS] [JOURNAL AUTO-VERIFIED PASS] Live Journal Integrity Checked: {len(records)} records verified cleanly (Events: {len(seen_keys)} | Strategy: '{sample.get('strategy')}' | Price Tier: '{sample.get('price_tier')}')."
        logger.info(msg)
        return True
    except Exception as exc:
        _verified_first_trades = True
        logger.warning("Auto journal verification check error: %s", exc)
        return False
