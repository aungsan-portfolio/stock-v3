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
from strategies.position_sizer import validate_executable_geometry

logger = logging.getLogger(__name__)

_validation_lock = threading.Lock()
DEFAULT_VALIDATION_LEDGER_PATH = os.path.join("data", "paper_validation_trades.jsonl")

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
    status: str = "PENDING"  # PENDING, SUBMITTED, FILLED, CLOSED, VETOED

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def log_validation_record(
    record: ValidationTradeRecord,
    filepath: str = DEFAULT_VALIDATION_LEDGER_PATH,
) -> None:
    """Append a validation trade record to data/paper_validation_trades.jsonl."""
    dir_name = os.path.dirname(filepath)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    line = json.dumps(record.to_dict(), default=str) + "\n"
    with _validation_lock:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line)


def record_validation_fill(
    order_id: str,
    fill_price: float,
    qty: float,
    side: str,
    timestamp: Optional[str] = None,
    slippage: Optional[float] = None,
    client_order_id: Optional[str] = None,
    symbol: Optional[str] = None,
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

        for rec in reversed(records):
            r_oid = str(rec.get("entry_order_id") or "").strip()
            r_cid = str(rec.get("validation_id") or "").strip()
            r_sym = str(rec.get("symbol") or "").upper().strip()
            r_status = str(rec.get("status") or "").upper().strip()

            is_match = False
            if target_oid and (r_oid == target_oid or target_oid in r_oid):
                is_match = True
            elif target_cid and (r_cid == target_cid or target_cid in r_cid):
                is_match = True
            elif target_sym and r_sym == target_sym and r_status == "SUBMITTED":
                is_match = True

            if is_match and r_status != "CLOSED":
                rec["entry_fill_price"] = float(fill_price)
                rec["entry_price"] = float(fill_price)
                rec["event_type"] = "TRADE"
                if slippage is not None:
                    rec["slippage_bps"] = round(float(slippage) * 10000.0, 2)
                if target_oid and not rec.get("entry_order_id"):
                    rec["entry_order_id"] = target_oid
                rec["status"] = "FILLED"
                updated = True
                logger.info(
                    "[VALIDATION FILL RECORDED] %s %s @ $%.2f (order_id=%s, slippage=%.4f)",
                    rec.get("symbol"), rec.get("side"), fill_price, order_id, slippage or 0.0
                )
                break

        if updated:
            try:
                temp_path = filepath + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, default=str) + "\n")
                os.replace(temp_path, filepath)
                return True
            except Exception as write_err:
                logger.error("[VALIDATION LEDGER ERROR] Failed to update %s: %s", filepath, write_err)
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
    filepath: str = DEFAULT_VALIDATION_LEDGER_PATH,
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

        for rec in reversed(records):
            r_sym = str(rec.get("symbol") or "").upper().strip()
            r_status = str(rec.get("status") or "").upper().strip()

            if target_sym and r_sym == target_sym and r_status in ("FILLED", "SUBMITTED"):
                rec["exit_order_id"] = target_oid
                rec["exit_fill_price"] = float(exit_price)
                rec["exit_price"] = float(exit_price)
                rec["status"] = "CLOSED"
                rec["event_type"] = "TRADE_CLOSED"
                rec["exit_reason"] = str(exit_reason)

                entry_px = float(rec.get("entry_fill_price") or rec.get("decision_price") or 0.0)
                pos_side = str(rec.get("side") or "BUY").upper().strip()
                if entry_px > 0:
                    if pos_side in ("BUY", "LONG"):
                        pnl = (float(exit_price) - entry_px) * float(qty)
                    else:
                        pnl = (entry_px - float(exit_price)) * float(qty)
                    rec["realized_pnl"] = round(pnl, 4)
                    rec["pnl"] = round(pnl, 4)

                updated = True
                logger.info(
                    "[VALIDATION EXIT RECORDED] %s exited @ $%.2f | P&L: $%.4f (reason=%s, exit_oid=%s)",
                    rec.get("symbol"), exit_price, rec.get("realized_pnl", 0.0), exit_reason, order_id
                )
                break

        if updated:
            try:
                temp_path = filepath + ".tmp"
                with open(temp_path, "w", encoding="utf-8") as f:
                    for rec in records:
                        f.write(json.dumps(rec, default=str) + "\n")
                os.replace(temp_path, filepath)
                return True
            except Exception as write_err:
                logger.error("[VALIDATION LEDGER ERROR] Failed to update %s: %s", filepath, write_err)
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
    )

    # 5. Order Placement (1 Share Fixed Tiny Sizing)
    shares = max(1, min(int(fixed_shares), 3))
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
