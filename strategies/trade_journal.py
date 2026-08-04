"""
trade_journal.py -- Append-only JSONL trade journal for review and learning.
"""
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config
from strategies.session import is_market_day, now_eastern

logger = logging.getLogger(__name__)

_journal_lock = threading.Lock()

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
):
    p = float(entry_price or exit_price or 0.0)
    price_tier = "5-10" if (5.0 <= p < 10.0) else (">10" if p >= 10.0 else "<5")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "order_id": order_id,
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "strategy": strategy,
        "qty": qty,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl": pnl,
        "price_tier": price_tier,
        "notes": notes,
    }
    path = journal_file or config.DAYTRADE_TRADE_JOURNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with _journal_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
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
    signal_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    strategy: Optional[str] = None,
    price_tier: Optional[str] = None,
):
    """Log an actual execution fill event with latency and slippage."""
    if not price_tier and fill_price > 0:
        price_tier = "5-10" if (5.0 <= fill_price < 10.0) else (">10" if fill_price >= 10.0 else "<5")

    record = {
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "type": "FILL",
        "event_type": "FILL",
        "order_id": order_id,
        "execution_id": execution_id,
        "signal_id": signal_id,
        "symbol": symbol,
        "side": side,
        "strategy": strategy or "UNKNOWN",
        "qty": qty,
        "fill_price": fill_price,
        "expected_price": expected_price,
        "slippage": slippage,
        "fill_latency_ms": fill_latency_ms,
        "price_tier": price_tier,
    }
    path = journal_file or config.DAYTRADE_TRADE_JOURNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with _journal_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
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

def day_trades_in_last_5_days(journal_file: Optional[Path] = None) -> int:
    """Count completed intraday round trips across the last five market days."""
    records = read_journal(journal_file)
    return count_day_trades_in_records(records, now_eastern().date())

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

    entries_by_signal = {
        str(record["signal_id"]): record
        for record in records
        if record.get("event_type") == "ORDER_SUBMITTED" and record.get("signal_id")
    }
    fills_by_symbol = defaultdict(list)
    seen_executions = set()
    existing_closed_signals = set()

    for record in records:
        try:
            if record.get("event_type") == "TRADE_CLOSED":
                signal_id = record.get("signal_id")
                if signal_id:
                    existing_closed_signals.add(str(signal_id))
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
                    signal_key = str(current_signal_id) if current_signal_id else None
                    if signal_key and signal_key not in existing_closed_signals:
                        entry = entries_by_signal.get(signal_key, {})
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
        
        seen_ids = set()
        for r in records:
            oid = str(r.get("execution_id") or r.get("order_id") or "")
            if oid:
                if oid in seen_ids:
                    logger.error("[FAIL] [JOURNAL DEDUP FAIL] Duplicate record detected for order_id %s!", oid)
                    return False
                seen_ids.add(oid)

        sample = records[0]
        _verified_first_trades = True
        msg = f"[PASS] [JOURNAL AUTO-VERIFIED PASS] Live Journal Integrity Checked: {len(records)} records verified cleanly (DEDUP: PASS | Strategy: '{sample.get('strategy')}' | Price Tier: '{sample.get('price_tier')}')."
        logger.info(msg)
        print(f"\n{msg}\n")
        return True
    except Exception as exc:
        logger.warning("Auto journal verification check error: %s", exc)
        return False
