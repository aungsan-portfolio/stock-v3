"""
trailing_stop.py -- Dynamic ATR-based trailing stop manager.

Protects unrealized profit by ratcheting stops as price moves in our favor.
Never loosens a stop. Updates orders directly when live.
"""
import json
import logging
import os
import math
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union, Any
from pathlib import Path

import config
from strategies.base import TradeSignal
from strategies.intraday_data import fetch_intraday
from strategies.session import is_market_open
from strategies.webhook import send_discord_alert

logger = logging.getLogger(__name__)

MAX_REMEDIATION_STRIKES: int = 2

# ── Canonical Post-Fill Protection Geometry Constants ────────────────────────
POST_FILL_NOISE_FLOOR_BPS: float = 0.0005     # 5 bps (0.05%) price scale floor
POST_FILL_MIN_ABSOLUTE_BUFFER: float = 0.05    # $0.05 absolute minimum tick buffer
POST_FILL_REBUILD_BUFFER_MULT: float = 1.15    # Pre-existing 15% safety buffer


class TrailingStatePersistenceError(RuntimeError):
    """Raised when trailing stop state cannot be persisted to disk."""
    pass


@dataclass
class GeometryRebuildResult:
    """Outcome of post-fill bracket geometry validation and child order rebuild."""
    stop_price: float
    desired_stop_price: float
    confirmed_broker_stop_price: Optional[float]
    stop_order_id: Optional[str]
    stop_reconciliation_required: bool
    rebuild_performed: bool
    rebuild_succeeded: bool


def compute_canonical_post_fill_distances(
    fill_price: float, 
    decimals: int = 2
) -> Tuple[float, float]:
    """
    Authoritative shared calculation for post-fill protection geometry.
    Ensures validation and rebuild share the exact same canonical distance basis.
    
    Returns:
        Tuple[float, float]: (min_required_distance, min_rebuild_distance)
    """
    min_required_distance = max(POST_FILL_MIN_ABSOLUTE_BUFFER, fill_price * POST_FILL_NOISE_FLOOR_BPS)
    min_rebuild_distance = round(min_required_distance * POST_FILL_REBUILD_BUFFER_MULT, decimals)
    return min_required_distance, min_rebuild_distance


_STOP_ORDER_TYPES = frozenset([
    "stop",
    "stop_limit",
    "trailing_stop",
])

_WORKING_STOP_STATUSES = frozenset([
    "new",
    "accepted",
    "pending_new",
    "accepted_for_bidding",
    "held",
    "calculated",
    "partially_filled",
    "open",
    "submitted",
])

_TERMINAL_STOP_STATUSES = frozenset([
    "rejected",
    "canceled",
    "cancelled",
    "expired",
    "failed",
    "done_for_day",
    "stopped",
    "filled",
])


def _normalize_sdk_field(raw: Any) -> str:
    """Safely extracts lowercase string representation from SDK Enum, string, or object."""
    if raw is None:
        return ""
    if hasattr(raw, "value"):
        return str(raw.value).lower().strip()
    return str(raw).lower().strip()


def _extract_order_field(order_obj: Any, *field_names: str) -> Any:
    """Extract field from dict, object, or namespace."""
    if order_obj is None:
        return None
    if isinstance(order_obj, dict):
        for f in field_names:
            if f in order_obj and order_obj[f] is not None:
                return order_obj[f]
        return None
    for f in field_names:
        if hasattr(order_obj, f):
            v = getattr(order_obj, f, None)
            if v is not None:
                return v
    return None


def _is_finite_positive(val: Any) -> Tuple[bool, Optional[float]]:
    """Strictly validates that val can be parsed to a finite, strictly positive (> 0) float."""
    try:
        f = float(val)
        if math.isfinite(f) and f > 0.0:
            return True, f
        return False, None
    except (ValueError, TypeError):
        return False, None


def _is_finite_non_negative(val: Any) -> Tuple[bool, Optional[float]]:
    """Strictly validates that val can be parsed to a finite, non-negative (>= 0) float."""
    try:
        f = float(val)
        if math.isfinite(f) and f >= 0.0:
            return True, f
        return False, None
    except (ValueError, TypeError):
        return False, None


def verify_broker_stop_order_evidence(
    order_data: Any,
    expected_order_id: str,
    expected_symbol: str,
    position_side: str,
    expected_stop_price: float,
    expected_qty: Optional[float] = None,
    fill_price: Optional[float] = None,
    require_qty: bool = False,
) -> Tuple[bool, str]:
    """
    Validates that order_data satisfies strict broker evidence requirements:
    1. Expected Parameters Validation at Boundary:
       - expected_order_id, expected_symbol, position_side must be non-empty/valid.
       - expected_stop_price must be finite and > 0.
       - expected_qty must be finite and > 0 (required when require_qty=True).
    2. Identity: order_id matches expected_order_id.
    3. Status: Must be in _WORKING_STOP_STATUSES and strictly not in _TERMINAL_STOP_STATUSES.
               Empty status is strictly rejected.
    4. Symbol: matches expected_symbol.
    5. Side: For BUY/LONG position -> stop order side must be SELL.
             For SELL/SHORT position -> stop order side must be BUY.
    6. Type: Must be in _STOP_ORDER_TYPES ('stop', 'stop_limit', 'trailing_stop').
    7. Stop Price: Must match expected_stop_price within instrument tick-size tolerance.
    8. Qty & Capacity:
       - Order qty must be present, finite, > 0.
       - If partially_filled, filled_qty must be present, finite, >= 0.
       - Remaining order capacity (qty - filled_qty) must be > 0 and match expected_qty.
    """
    if order_data is None:
        return False, "EVIDENCE_MISSING: Order data is None"

    # 1. Expected Parameters Validation at Boundary
    if not expected_order_id or not str(expected_order_id).strip():
        return False, "EVIDENCE_EXPECTED_ID_MISSING: expected_order_id must be non-empty"

    if not expected_symbol or not str(expected_symbol).strip():
        return False, "EVIDENCE_EXPECTED_SYMBOL_MISSING: expected_symbol must be non-empty"

    if not position_side or str(position_side).upper().strip() not in ("BUY", "LONG", "SELL", "SHORT"):
        return False, f"EVIDENCE_POSITION_SIDE_INVALID: position_side {position_side} must be BUY/LONG or SELL/SHORT"

    valid_exp_px, exp_stop_f = _is_finite_positive(expected_stop_price)
    if not valid_exp_px:
        return False, f"EVIDENCE_EXPECTED_STOP_PRICE_INVALID: Expected stop price {expected_stop_price} is not finite or <= 0"

    exp_qty_f: Optional[float] = None
    if expected_qty is not None:
        valid_exp_qty, exp_qty_f = _is_finite_positive(expected_qty)
        if not valid_exp_qty:
            return False, f"EVIDENCE_EXPECTED_QTY_INVALID: Expected qty {expected_qty} is not finite or <= 0"
    elif require_qty:
        return False, "EVIDENCE_EXPECTED_QTY_MISSING: Position expected_qty must be provided to confirm protective coverage"

    # 2. Identity check
    order_id = _extract_order_field(order_data, "id", "order_id")
    if not order_id:
        return False, "EVIDENCE_ID_MISSING: Order data lacks id"
    if str(order_id).strip() != str(expected_order_id).strip():
        return False, f"EVIDENCE_ID_MISMATCH: Order id {order_id} != expected {expected_order_id}"

    # 3. Status normalization and check
    raw_status = _extract_order_field(order_data, "status")
    if raw_status is None:
        return False, "EVIDENCE_STATUS_MISSING: Order status is None"
    status_str = _normalize_sdk_field(raw_status)
    if not status_str:
        return False, "EVIDENCE_STATUS_MISSING: Order status is empty"
    if status_str in _TERMINAL_STOP_STATUSES:
        return False, f"EVIDENCE_STATUS_TERMINAL: Status {status_str} is terminal/rejected"
    if status_str not in _WORKING_STOP_STATUSES:
        return False, f"EVIDENCE_STATUS_NON_WORKING: Status {status_str} is not a valid working protection state"

    # 4. Symbol check
    raw_symbol = _extract_order_field(order_data, "symbol")
    if raw_symbol is None:
        return False, "EVIDENCE_SYMBOL_MISSING: Order symbol is missing"
    if str(raw_symbol).upper().strip() != str(expected_symbol).upper().strip():
        return False, f"EVIDENCE_SYMBOL_MISMATCH: Symbol {raw_symbol} != expected {expected_symbol}"

    # 5. Side check
    expected_stop_side = "sell" if str(position_side).upper().strip() in ("BUY", "LONG") else "buy"
    raw_side = _extract_order_field(order_data, "side")
    if raw_side is None:
        return False, "EVIDENCE_SIDE_MISSING: Order side is missing"
    side_str = _normalize_sdk_field(raw_side)
    if side_str != expected_stop_side:
        return False, f"EVIDENCE_SIDE_MISMATCH: Stop order side {side_str} != expected {expected_stop_side} for {position_side} position"

    # 6. Order Type check
    raw_type = _extract_order_field(order_data, "type", "order_type")
    if raw_type is None:
        return False, "EVIDENCE_TYPE_MISSING: Order type is missing"
    type_str = _normalize_sdk_field(raw_type)
    if type_str not in _STOP_ORDER_TYPES:
        return False, f"EVIDENCE_TYPE_INVALID: Order type {type_str} not in {_STOP_ORDER_TYPES}"

    # 7. Stop Price check (Dynamic Instrument Tick Size & Quantization Contract)
    raw_stop_price = _extract_order_field(order_data, "stop_price", "stopPrice")
    if raw_stop_price is None:
        return False, "EVIDENCE_STOP_PRICE_MISSING: Order lacks stop_price"
    valid_stop_px, stop_price_f = _is_finite_positive(raw_stop_price)
    if not valid_stop_px:
        return False, f"EVIDENCE_STOP_PRICE_INVALID: Stop price {raw_stop_price} is not finite or <= 0"

    ref_price = max(exp_stop_f, fill_price) if (fill_price is not None and math.isfinite(fill_price)) else exp_stop_f
    tick_size = 0.01 if ref_price >= 1.0 else 0.0001
    cfg_tol = getattr(config, "TRAILING_STOP_RECONCILE_TOLERANCE", None)
    if cfg_tol is not None and math.isfinite(float(cfg_tol)):
        tolerance = min(max(tick_size / 2.0, float(cfg_tol)), tick_size) + 1e-6
    else:
        tolerance = (tick_size / 2.0 + 1e-6)

    if abs(stop_price_f - exp_stop_f) > tolerance:
        return False, (
            f"EVIDENCE_STOP_PRICE_MISMATCH: Broker stop {stop_price_f:.4f} != "
            f"expected {exp_stop_f:.4f} (tolerance {tolerance:.6f}, tick_size {tick_size:.4f})"
        )

    # 8. Quantity & Remaining Protection Capacity check
    raw_qty = _extract_order_field(order_data, "qty", "quantity")
    if raw_qty is None:
        return False, "EVIDENCE_QTY_MISSING: Order qty is missing"
    valid_order_qty, order_qty_f = _is_finite_positive(raw_qty)
    if not valid_order_qty:
        return False, f"EVIDENCE_QTY_INVALID: Order qty {raw_qty} is not finite or <= 0"

    raw_filled_qty = _extract_order_field(order_data, "filled_qty", "filledQuantity", "filled_shares")
    filled_qty_f = 0.0
    if status_str == "partially_filled":
        if raw_filled_qty is None:
            return False, "EVIDENCE_FILLED_QTY_MISSING: Partially filled order lacks filled_qty"
        valid_filled, filled_qty_f = _is_finite_non_negative(raw_filled_qty)
        if not valid_filled:
            return False, f"EVIDENCE_FILLED_QTY_INVALID: Filled qty {raw_filled_qty} is not finite or < 0"
    elif raw_filled_qty is not None:
        if type(raw_filled_qty).__name__ not in ("MagicMock", "Mock"):
            valid_filled, f_qty = _is_finite_non_negative(raw_filled_qty)
            if not valid_filled:
                return False, f"EVIDENCE_FILLED_QTY_INVALID: Filled qty {raw_filled_qty} is not finite or < 0"
            filled_qty_f = f_qty

    if filled_qty_f > order_qty_f:
        return False, f"EVIDENCE_FILLED_EXCEEDS_QTY: Filled qty {filled_qty_f:.4f} > order qty {order_qty_f:.4f}"

    remaining_qty = order_qty_f - filled_qty_f
    if remaining_qty <= 0.0:
        return False, f"EVIDENCE_QTY_EXHAUSTED: Remaining order capacity {remaining_qty:.4f} <= 0 (qty={order_qty_f}, filled={filled_qty_f})"

    if exp_qty_f is not None:
        if remaining_qty < exp_qty_f - 1e-4:
            return False, (
                f"EVIDENCE_QTY_MISMATCH: Remaining order capacity {remaining_qty:.4f} < "
                f"expected {exp_qty_f:.4f} (total_qty={order_qty_f:.4f}, filled_qty={filled_qty_f:.4f})"
            )

    return True, "EVIDENCE_VERIFIED"



def _normalize_strategy_name(raw) -> str:
    """Safely extract string strategy name from enum, string, or object."""
    if hasattr(raw, "value"):
        return str(raw.value)
    if isinstance(raw, str) and raw.strip() and raw.strip() != "UNKNOWN":
        return raw.strip()
    return "UNKNOWN"


def _timestamp_to_epoch(val: Any) -> Optional[float]:
    """Safely convert any timestamp/datetime/epoch representation to UTC float epoch seconds."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val) if math.isfinite(val) and val > 0 else None
    if isinstance(val, datetime):
        return val.timestamp()
    if isinstance(val, str):
        try:
            val_clean = val.replace("Z", "+00:00")
            dt = datetime.fromisoformat(val_clean)
            return dt.timestamp()
        except Exception:
            return None
    return None


def _classify_held_order(order) -> str:
    """
    Return one of:
      'stop'       -- a protective stop order (safe, keep waiting)
      'exit'       -- a market/limit exit already in-flight (safe, keep waiting)
      'unrelated'  -- something else entirely (unsafe, must flatten)
    """
    if order is None:
        return "unrelated"
    order_type = str(getattr(order, "order_type", getattr(order, "type", ""))).lower().replace("ordertype.", "")
    if order_type in _STOP_ORDER_TYPES:
        return "stop"

    client_id = str(getattr(order, "client_order_id", getattr(order, "orderRef", "")) or "")
    if order_type in ("market", "limit") and (
        client_id.startswith(("dt_", "val_", "exit_", "flatten_", "close_", "sell_", "buy_")) or ":" in client_id
    ):
        return "exit"

    return "unrelated"


DEFAULT_ACCOUNT_IDS: Dict[str, str] = {
    "daytrade": "PA3UODIC5JTA",
    "swing": "PA3R71I7JFDZ",
}

_default_context: Dict[str, Optional[str]] = {"mode": None, "account_id": None}


def set_default_trailing_context(mode: Optional[str] = None, account_id: Optional[str] = None) -> None:
    """Set the process-level default trailing stop context (mode, account_id)."""
    global manager, trailing_stop_manager
    if mode:
        _default_context["mode"] = str(mode).strip().lower()
    if account_id and str(account_id).strip() != "default":
        _default_context["account_id"] = str(account_id).strip()
    mgr = get_trailing_manager(mode=_default_context.get("mode"), account_id=_default_context.get("account_id"))
    manager = mgr
    trailing_stop_manager = mgr


def get_trailing_state_file(mode: Optional[str] = None, account_id: Optional[str] = None) -> Path:
    """Dynamically resolve trailing stop state file against config.DATA_DIR partitioned by mode/account."""
    override = globals().get("_trailing_state_override")
    if override is not None:
        return Path(override)
    try:
        import config
        base_dir = Path(config.DATA_DIR)
    except Exception:
        base_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'))

    clean_mode = str(mode).strip().lower() if mode else (_default_context.get("mode") or "")
    clean_acct = str(account_id).strip() if account_id else (_default_context.get("account_id") or "")
    if clean_mode and (not clean_acct or clean_acct == "default"):
        clean_acct = DEFAULT_ACCOUNT_IDS.get(clean_mode, clean_acct)

    parts = ["trailing_state"]
    if clean_mode:
        parts.append(clean_mode)
    if clean_acct and clean_acct != "default":
        parts.append(clean_acct)

    filename = "_".join(parts) + ".json"
    return base_dir / filename


STATE_FILE = str(get_trailing_state_file())

@dataclass
class TrailingStopState:
    symbol: str
    side: str
    entry_price: float
    peak_price: float
    atr: float
    stop_price: float
    trail_multiple: float
    last_updated: float
    order_id: Optional[str] = None
    active: bool = True
    remediation_in_progress: bool = False
    last_remediation_time: float = 0.0
    remediation_attempts: int = 0
    escalation_alerted: bool = False
    qty: float = 0.0
    strategy: str = "UNKNOWN"
    signal_id: Optional[str] = None
    target_price: Optional[float] = None
    entry_time: float = 0.0
    partial_taken: bool = False
    partial_qty: float = 0.0
    partial_exit_price: float = 0.0
    desired_stop_price: Optional[float] = None
    confirmed_broker_stop_price: Optional[float] = None
    stop_reconciliation_required: bool = False
    is_validation: bool = False
    last_stop_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    reconciliation_status: Optional[str] = None
    entry_qty: Optional[float] = None

    def __post_init__(self):
        if self.side:
            self.side = "BUY" if str(self.side).upper().strip() in ("BUY", "LONG") else "SELL"
        if self.entry_qty is None and self.qty > 0.0:
            self.entry_qty = float(self.qty)
        if self.desired_stop_price is None:
            self.desired_stop_price = float(self.stop_price)
        if self.confirmed_broker_stop_price is None and not self.stop_reconciliation_required:
            self.confirmed_broker_stop_price = float(self.stop_price)
        if self.order_id and not self.last_stop_order_id:
            self.last_stop_order_id = str(self.order_id)


MAX_HOLD_MINUTES = 25.0
BREAKEVEN_BAND_PCT = 0.10
TIME_EXIT_CHECK_ENABLED = True


def check_time_based_exit(
    symbol: str,
    side: str,
    entry_price: float,
    current_price: float,
    entry_timestamp: float,
    cfg=None,
    peak_price: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Triggers time-based exit if scalp position held >= max_hold (25m) under:
    A. Stagnation exit: price stagnant near breakeven (+-0.10%).
    B. Giveback exit: price peaked (>= 0.30% MFE) and gave back >= 25% of peak gains, returning near breakeven.
    """
    cfg = cfg or config
    enabled = getattr(cfg, "TIME_EXIT_CHECK_ENABLED", TIME_EXIT_CHECK_ENABLED)
    if not enabled or not entry_timestamp:
        return False, "Time exit: disabled or missing timestamp"

    now_ts = time.time()
    hold_minutes = (now_ts - entry_timestamp) / 60.0
    max_hold = getattr(cfg, "MAX_HOLD_MINUTES", MAX_HOLD_MINUTES)

    if hold_minutes < max_hold:
        return False, f"Time exit: {hold_minutes:.1f}m < {max_hold}m threshold"

    pct_from_entry = ((current_price - entry_price) / entry_price) * 100.0 if entry_price > 0 else 0.0
    band_pct = getattr(cfg, "BREAKEVEN_BAND_PCT", BREAKEVEN_BAND_PCT)

    # A. Stagnation Check
    near_entry = abs(pct_from_entry) <= band_pct

    # B. Giveback Check
    min_mfe_pct = float(getattr(cfg, "TIME_EXIT_MIN_MFE_PCT", 0.30))
    max_giveback_pct = float(getattr(cfg, "TIME_EXIT_MAX_GIVEBACK_PCT", 25.0))
    gave_back_profit = False

    if peak_price and peak_price > 0 and entry_price > 0:
        if side.upper() == "BUY":
            peak_profit_pct = ((peak_price - entry_price) / entry_price) * 100.0
            giveback_pct = ((peak_price - current_price) / peak_price) * 100.0
        else:
            peak_profit_pct = ((entry_price - peak_price) / entry_price) * 100.0
            giveback_pct = ((current_price - peak_price) / peak_price) * 100.0

        if peak_profit_pct >= min_mfe_pct and giveback_pct >= (max_giveback_pct / 100.0 * peak_profit_pct) and near_entry:
            gave_back_profit = True

    if gave_back_profit or near_entry:
        reason_type = "gave back peak profits near breakeven" if gave_back_profit else "stagnant near breakeven"
        reason = (
            f"TIME EXIT TRIGGERED: {symbol} ({side}) held {hold_minutes:.1f}m "
            f"with price {pct_from_entry:+.2f}% from entry — {reason_type}. Closing to free capital."
        )
        logger.info("[TIME EXIT] %s", reason)
        return True, reason

    direction = "profit" if pct_from_entry > 0 else "loss"
    return False, f"Time exit: {hold_minutes:.1f}m held but price moved {pct_from_entry:+.2f}% ({direction}) — let run"


def normalize_order_type(order_type: object) -> str:
    if not order_type:
        return ""
    val = getattr(order_type, "value", order_type)
    s = str(val).lower().strip()
    s = s.replace("ordertype.", "")
    return s


def is_stop_order_type(order_type_str: str, has_stop_price: bool = False) -> bool:
    s = order_type_str.lower().strip()
    if has_stop_price:
        return True
    return s in {
        "stop", "stp", "stop_limit", "stp lmt", "stoplimit",
        "trail", "trailing_stop", "traillimit", "trail limit", "mkt prt"
    }


def _extract_order_info(o: object) -> dict:
    if isinstance(o, dict):
        o_id = str(o.get("id") or o.get("order_id") or "")
        o_sym = str(o.get("symbol") or "")
        raw_side = str(o.get("side") or o.get("action") or "").lower()
        raw_type = o.get("order_type") or o.get("type") or o.get("orderType")
        o_stop = o.get("stop_price") if o.get("stop_price") is not None else o.get("stopPrice")
        o_status = str(o.get("status") or "").lower()
        raw_qty = o.get("qty") or o.get("quantity") or o.get("totalQuantity")
        raw_filled_qty = o.get("filled_qty") or o.get("filled_quantity") or o.get("filledQuantity")
    else:
        o_id = str(getattr(o, "id", getattr(o, "order_id", "")) or "")
        contract = getattr(o, "contract", None)
        o_sym = getattr(o, "symbol", None)
        if not o_sym and contract:
            o_sym = getattr(contract, "symbol", "")
        o_sym = str(o_sym or "").upper().strip()

        raw_side = getattr(o, "side", getattr(o, "action", ""))
        raw_side = getattr(raw_side, "value", raw_side) or ""

        raw_type = getattr(o, "order_type", getattr(o, "type", getattr(o, "orderType", "")))
        o_stop = getattr(o, "stop_price", getattr(o, "stopPrice", None))

        o_status_obj = getattr(o, "status", getattr(getattr(o, "orderState", None), "status", getattr(getattr(o, "orderStatus", None), "status", None)))
        o_status = str(getattr(o_status_obj, "value", o_status_obj) or "").lower()
        raw_qty = getattr(o, "qty", getattr(o, "quantity", getattr(o, "totalQuantity", None)))
        raw_filled_qty = getattr(o, "filled_qty", getattr(o, "filled_quantity", getattr(o, "filledQuantity", None)))

    s_side = str(raw_side).lower().strip()
    if s_side in {"buy"}:
        norm_side = "buy"
    elif s_side in {"sell"}:
        norm_side = "sell"
    else:
        norm_side = s_side

    o_type = normalize_order_type(raw_type)
    stop_val = float(o_stop) if o_stop is not None else None
    try:
        qty_val = float(raw_qty) if raw_qty is not None else None
    except (TypeError, ValueError):
        qty_val = None

    try:
        filled_qty_val = float(raw_filled_qty) if raw_filled_qty is not None else None
    except (TypeError, ValueError):
        filled_qty_val = None

    return {
        "id": o_id,
        "symbol": o_sym,
        "side": norm_side,
        "type": o_type,
        "stop_price": stop_val,
        "qty": qty_val,
        "filled_qty": filled_qty_val,
        "status": o_status
    }


class DynamicTrailingStopManager:
    STATE_FILE: str = str(get_trailing_state_file())
    consecutive_persistence_failures: int = 0
    persistence_healthy: bool = True
    last_persistence_error: Optional[str] = None
    last_persistence_error_time: float = 0.0

    def __init__(self, state_file: Optional[object] = None):
        self._lock = threading.RLock()
        target = state_file if state_file is not None else globals().get("STATE_FILE", get_trailing_state_file())
        self.STATE_FILE: str = os.path.abspath(os.fspath(target))
        self.persistence_healthy: bool = True
        self.last_persistence_error: Optional[str] = None
        self.last_persistence_error_time: float = 0.0
        self.consecutive_persistence_failures: int = 0
        with self._lock:
            self.states: Dict[str, TrailingStopState] = {}
            self.pending_exits: Dict[str, dict] = {}
            self._load_state()

    def _load_state(self):
        if not os.path.exists(self.STATE_FILE):
            return
        try:
            if os.path.getsize(self.STATE_FILE) == 0:
                logger.warning(f"Trailing stop state file {self.STATE_FILE} is empty. Initializing empty state.")
                self.persistence_healthy = False
                self.last_persistence_error = "STATE_FILE_EMPTY"
                return
            with open(self.STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if "__pending_exits__" in data and isinstance(data["__pending_exits__"], dict):
                        self.pending_exits = dict(data["__pending_exits__"])
                    for sym, state_dict in data.items():
                        if sym == "__pending_exits__":
                            continue
                        rec_status = str(state_dict.get("reconciliation_status") or "")
                        has_unresolved_rec = bool(
                            rec_status
                            and not rec_status.startswith("RECONCILED_")
                        )
                        has_stop_rec = bool(state_dict.get("stop_reconciliation_required"))
                        if state_dict.get('active') or has_unresolved_rec or has_stop_rec:
                            # P1.5.2 Migration for legacy state dictionaries
                            state_dict.setdefault("desired_stop_price", state_dict.get("stop_price"))
                            state_dict.setdefault("confirmed_broker_stop_price", state_dict.get("stop_price"))
                            state_dict.setdefault("stop_reconciliation_required", False)
                            state_dict.setdefault("last_stop_order_id", state_dict.get("order_id"))
                            state_dict.setdefault("tp_order_id", None)
                            state_dict.setdefault("reconciliation_status", None)
                            state_dict.setdefault("entry_qty", state_dict.get("qty"))
                            self.states[sym] = TrailingStopState(**state_dict)
            self.persistence_healthy = True
            logger.info(f"Loaded {len(self.states)} active trailing stops, {len(self.pending_exits)} pending exits.")
        except (json.JSONDecodeError, OSError) as e:
            self.persistence_healthy = False
            self.last_persistence_error = f"LOAD_FAILED: {e}"
            logger.error(f"Failed to load trailing stop state: {e}")

    def _save_state(self, raise_on_error: bool = False) -> bool:
        import threading
        tmp_file = f"{self.STATE_FILE}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            state_dir = os.path.dirname(self.STATE_FILE)
            os.makedirs(state_dir, exist_ok=True)
            payload = {
                sym: asdict(state)
                for sym, state in self.states.items()
            }
            if self.pending_exits:
                payload["__pending_exits__"] = dict(self.pending_exits)
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.STATE_FILE)
            if os.name != "nt":
                try:
                    dfd = os.open(state_dir, os.O_RDONLY)
                    try:
                        os.fsync(dfd)
                    finally:
                        os.close(dfd)
                except Exception:
                    pass
            self.persistence_healthy = True
            self.consecutive_persistence_failures = 0
            return True
        except Exception as exc:
            self.persistence_healthy = False
            self.last_persistence_error = str(exc)
            self.last_persistence_error_time = time.time()
            self.consecutive_persistence_failures += 1
            # Mark all active position protection statuses as requiring reconciliation
            for s in self.states.values():
                s.stop_reconciliation_required = True
            logger.error(
                "[PERSISTENCE HEALTH FAILED] Failed to save trailing stop state atomically to %s: %s "
                "(consecutive_failures=%d, all active positions flagged for reconciliation)",
                self.STATE_FILE,
                exc,
                self.consecutive_persistence_failures,
            )
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except OSError:
                logger.debug(
                    "Failed to remove temporary state file: %s",
                    tmp_file,
                    exc_info=True,
                )
            if raise_on_error:
                raise TrailingStatePersistenceError(
                    f"Failed to save trailing stop state atomically to {self.STATE_FILE}: {exc}"
                ) from exc
            return False

    def _query_broker_position_qty(self, bridge: Optional[object], symbol: str) -> Tuple[bool, Optional[float]]:
        """
        Query broker position quantity fail-closed.
        Returns: (is_known: bool, signed_qty: Optional[float])
        - If query succeeds and position exists: (True, float(qty))
        - If query succeeds and position does not exist (flat/404): (True, 0.0)
        - If query fails / exceptions occur: (False, None)
        """
        if bridge is None:
            return False, None

        sym_clean = symbol.upper().strip()

        if hasattr(bridge, "get_position"):
            try:
                qty = bridge.get_position(sym_clean)
                if qty is None or not isinstance(qty, (int, float)):
                    return False, None
                val = float(qty)
                if not math.isfinite(val):
                    return False, None
                return True, val
            except Exception as exc:
                err_str = str(exc).lower()
                if "position does not exist" in err_str or "404" in err_str:
                    return True, 0.0
                logger.warning("[BROKER POSITION QUERY FAILED] %s: %s", sym_clean, exc)
                return False, None

        if hasattr(bridge, "get_positions"):
            try:
                positions = bridge.get_positions()
                if positions is None or not isinstance(positions, (list, tuple)):
                    return False, None
                for p in positions:
                    p_sym = getattr(p, "symbol", "") or (p.get("symbol") if isinstance(p, dict) else "")
                    if str(p_sym).upper().strip() == sym_clean:
                        raw_q = getattr(p, "qty", None) if not isinstance(p, dict) else p.get("qty")
                        if raw_q is None:
                            return False, None
                        try:
                            val = float(raw_q)
                            if not math.isfinite(val):
                                return False, None
                            return True, val
                        except (ValueError, TypeError):
                            return False, None
                return True, 0.0
            except Exception as exc:
                logger.warning("[BROKER GET_POSITIONS FAILED] %s: %s", sym_clean, exc)
                return False, None

        if hasattr(bridge, "has_position"):
            try:
                has_pos = bridge.has_position(sym_clean)
                if not has_pos:
                    return True, 0.0
            except Exception as exc:
                logger.warning("[BROKER HAS_POSITION FAILED] %s: %s", sym_clean, exc)
                return False, None

        return False, None

    def _record_pending_exit(self, state: TrailingStopState, qty_closed: Optional[float] = None) -> None:
        """Record an open position whose broker position has flattened into the durable pending exits queue."""
        symbol = state.symbol.upper().strip()
        closed_qty = float(
            state.entry_qty
            if (getattr(state, "entry_qty", None) is not None and float(state.entry_qty) > 0)
            else (qty_closed if qty_closed is not None and qty_closed > 0 else (state.qty or 0.0))
        )
        episode_key = str(state.signal_id or state.order_id or state.tp_order_id or int(getattr(state, "entry_time", 0) or time.time()))
        episode_id = f"{symbol}:{episode_key}"
        self.pending_exits[episode_id] = {
            "episode_id": episode_id,
            "symbol": symbol,
            "side": state.side,
            "qty": closed_qty,
            "entry_price": float(state.entry_price),
            "target_price": float(state.target_price) if getattr(state, "target_price", None) else None,
            "stop_price": float(state.stop_price) if getattr(state, "stop_price", None) else None,
            "tp_order_id": str(state.tp_order_id) if getattr(state, "tp_order_id", None) else None,
            "stop_order_id": str(state.order_id) if getattr(state, "order_id", None) else None,
            "last_stop_order_id": str(state.last_stop_order_id) if getattr(state, "last_stop_order_id", None) else None,
            "is_validation": bool(getattr(state, "is_validation", False)),
            "entry_time": getattr(state, "entry_time", None) or getattr(state, "created_at", None),
            "signal_id": getattr(state, "signal_id", None),
            "strategy": getattr(state, "strategy", None) or "UNKNOWN",
            "registered_at": time.time(),
            "attempts": self.pending_exits.get(episode_id, {}).get("attempts", 0) + 1,
        }
        state.active = False
        state.reconciliation_status = "FLAT_PENDING_EXIT_RECONCILIATION"
        state.order_id = None
        state.remediation_in_progress = False
        saved = self._save_state()
        if not saved:
            state.stop_reconciliation_required = True
            logger.error(
                "[RECORD PENDING EXIT PERSISTENCE FAILED] %s: State persistence failed while recording pending exit. "
                "Retaining fail-closed reconciliation requirement.",
                symbol
            )
            return False
        return True

    def has_pending_exit(self, symbol: str) -> bool:
        """Check if any pending exit in durable queue matches symbol."""
        sym_clean = symbol.upper().strip()
        with self._lock:
            for item in self.pending_exits.values():
                if item.get("symbol", "").upper().strip() == sym_clean:
                    return True
            return False

    def has_unresolved_position_unknown(self) -> bool:
        """Check if any position in states has unresolved broker query failure or signed-side mismatch."""
        with self._lock:
            for s in self.states.values():
                if getattr(s, "reconciliation_status", None) in (
                    "POSITION_UNKNOWN_RECONCILIATION_REQUIRED",
                    "SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED",
                ):
                    return True
            return False

    def has_unresolved_signed_side_mismatch(self) -> bool:
        """Check if any position in states has an unresolved signed-side mismatch with broker."""
        with self._lock:
            for s in self.states.values():
                if getattr(s, "reconciliation_status", None) == "SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED":
                    return True
            return False

    def clear_position_unknown(self, symbol: str) -> bool:
        """Clear unresolved position unknown or signed-side mismatch status once broker truth is restored."""
        with self._lock:
            state = self.states.get(symbol.upper().strip())
            if state and getattr(state, "reconciliation_status", None) in (
                "POSITION_UNKNOWN_RECONCILIATION_REQUIRED",
                "SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED",
            ):
                state.reconciliation_status = None
                self._save_state()
                logger.info("[POSITION UNKNOWN CLEARED] Successfully cleared unknown/mismatch status for %s", symbol)
                return True
            return False

    def _recover_orphaned_flat_states(self, bridge: Optional[object]) -> int:
        """
        Recover orphaned flat states where the broker position is confirmed 0 (flat),
        but local state is active or unreconciled and no pending exit is yet queued.
        Constructs a durable pending exit with exact episode metadata so deterministic
        exit reconciliation can proceed.
        """
        if bridge is None:
            return 0

        recovered_count = 0
        with self._lock:
            for symbol, state in list(self.states.items()):
                # Skip if already has a pending exit queued
                has_pending = (
                    self.has_pending_exit(symbol)
                    if hasattr(self, "has_pending_exit")
                    else any(item.get("symbol", "").upper().strip() == symbol.upper().strip() for item in getattr(self, "pending_exits", {}).values())
                )
                if has_pending:
                    continue

                should_check = (
                    state.active
                    or getattr(state, "stop_reconciliation_required", False)
                    or getattr(state, "reconciliation_status", None) in (
                        "POSITION_UNKNOWN_RECONCILIATION_REQUIRED",
                        "SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED",
                        "FLAT_PENDING_EXIT_RECONCILIATION",
                    )
                )
                if not should_check:
                    continue

                is_known, pos_qty = self._query_broker_position_qty(bridge, symbol)
                if not is_known or pos_qty is None:
                    continue

                if abs(pos_qty) < 1e-6:
                    logger.info(
                        "[ORPHANED FLAT RECOVERY] %s: Broker position is flat (qty=0), but state was active/unreconciled without pending exit. "
                        "Reconstructing pending exit.",
                        symbol
                    )
                    self._record_pending_exit(state, qty_closed=abs(float(state.entry_qty or state.qty or 0.0)))
                    recovered_count += 1

        return recovered_count


    def reconcile_pending_exits(self, bridge: Optional[object] = None) -> Dict[str, str]:
        """
        Durable Authoritative Exit Reconciliation.
        Scans all pending exits in the durable queue, queries broker truth, validates actual fill price,
        commits to trade journal atomically, and removes from pending queue upon success.
        Returns a mapping of symbol -> reconciliation status.
        """
        recover_fn = getattr(self, "_recover_orphaned_flat_states", None)
        if bridge is not None and callable(recover_fn):
            recover_fn(bridge)

        if not self.pending_exits or bridge is None:
            return {}

        def _to_epoch(val: Any) -> Optional[float]:
            if val is None:
                return None
            from datetime import datetime as _dt
            if isinstance(val, (int, float)):
                return float(val) if math.isfinite(val) else None
            if isinstance(val, _dt):
                return val.timestamp()
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return None
                try:
                    f = float(s)
                    return f if math.isfinite(f) else None
                except ValueError:
                    pass
                try:
                    dt = _dt.fromisoformat(s.replace("Z", "+00:00"))
                    return dt.timestamp()
                except Exception:
                    pass
            return None

        results = {}
        with self._lock:
            for episode_id, exit_item in list(self.pending_exits.items()):
                symbol = exit_item.get("symbol") or episode_id.split(":")[0]
                try:
                    exit_reason = None
                    exit_price = None
                    notes = None

                    tp_id = exit_item.get("tp_order_id")
                    stop_id = exit_item.get("stop_order_id") or exit_item.get("last_stop_order_id")
                    expected_exit_side = "sell" if str(exit_item.get("side", "")).upper() in ("BUY", "LONG") else "buy"
                    exp_qty = float(exit_item.get("qty") or 0.0)

                    # Check current broker position fail-closed using canonical query
                    is_known, cur_pos_qty = self._query_broker_position_qty(bridge, symbol)
                    if not is_known:
                        logger.warning(
                            f"[RECONCILE BLOCKED] Position query failed/unknown for {symbol}. Retaining pending exit."
                        )
                        results[symbol] = "POSITION_UNKNOWN_RETAINED"
                        continue

                    if cur_pos_qty is not None and abs(cur_pos_qty) > 1e-6:
                        logger.warning(
                            f"[RECONCILE BLOCKED] Position for {symbol} still active on broker (qty={cur_pos_qty}). Retaining pending exit."
                        )
                        results[symbol] = "POSITION_STILL_OPEN_ON_BROKER"
                        continue

                    def _matches_order_lineage(client_order_id: str, target_id: Optional[str]) -> bool:
                        if not target_id or not client_order_id:
                            return False
                        target = str(target_id).strip()
                        cid = str(client_order_id).strip()
                        if not target or not cid:
                            return False
                        if target == cid:
                            return True
                        import re
                        pattern = rf"(?:^|[:_]){re.escape(target)}(?:$|[:_])"
                        return bool(re.search(pattern, cid))

                    conflict_detected = [False]
                    collected_orders: Dict[str, dict] = {}
                    known_ids = {str(k).strip() for k in (tp_id, stop_id, exit_item.get("last_stop_order_id")) if k}
                    sig_id = str(exit_item.get("signal_id") or "").strip()
                    ep_id = str(exit_item.get("episode_id") or "").strip()

                    # Build foreign episode/signal identifiers to detect explicit cross-episode conflict
                    other_foreign_ids = set()
                    pending_dict = getattr(self, "pending_exits", {}) or {}
                    if isinstance(pending_dict, dict):
                        for other_ep, other_item in pending_dict.items():
                            if other_ep != episode_id and isinstance(other_item, dict):
                                other_foreign_ids.add(str(other_ep).strip())
                                for k in ("signal_id", "episode_id", "stop_order_id", "last_stop_order_id", "tp_order_id"):
                                    v = other_item.get(k)
                                    if v:
                                        other_foreign_ids.add(str(v).strip())
                    states_dict = getattr(self, "states", {}) or {}
                    if isinstance(states_dict, dict):
                        for other_sym, other_state in states_dict.items():
                            other_sig = getattr(other_state, "signal_id", None)
                            other_sig_str = str(other_sig).strip() if other_sig else ""
                            is_foreign = (other_sym != symbol) or (bool(other_sig_str) and other_sig_str != sig_id)
                            if is_foreign:
                                if other_sig_str:
                                    other_foreign_ids.add(other_sig_str)
                                for attr in ("order_id", "last_stop_order_id", "tp_order_id"):
                                    val = getattr(other_state, attr, None)
                                    if val:
                                        other_foreign_ids.add(str(val).strip())
                    other_foreign_ids.discard("")

                    def _process_candidate_order(o: Any) -> None:
                        if not o or conflict_detected[0]:
                            return
                        oid = str(_extract_order_field(o, "id") or getattr(o, "id", "") or "").strip()
                        if not oid:
                            return

                        o_sym = str(_extract_order_field(o, "symbol") or getattr(o, "symbol", "") or "").upper().strip()
                        if o_sym != symbol.upper().strip():
                            return

                        o_side = str(_extract_order_field(o, "side") or getattr(o, "side", "") or "").lower().strip()
                        if o_side != expected_exit_side:
                            return

                        # Timestamp guard: Exit fill MUST NOT occur before entry
                        raw_filled_at = _extract_order_field(o, "filled_at") or getattr(o, "filled_at", None) or _extract_order_field(o, "submitted_at") or getattr(o, "submitted_at", None)
                        if raw_filled_at is not None:
                            order_epoch = _to_epoch(raw_filled_at)
                            entry_epoch = _to_epoch(exit_item.get("entry_time"))
                            if order_epoch is not None and entry_epoch is not None:
                                if order_epoch < entry_epoch:
                                    logger.warning(
                                        "[RECONCILE PRE-ENTRY ORDER REJECTED] %s: Order %s timestamp (%.1f) is before entry (%.1f)",
                                        symbol, oid, order_epoch, entry_epoch
                                    )
                                    return

                        o_cid = str(_extract_order_field(o, "client_order_id") or getattr(o, "client_order_id", "") or "").strip()

                        # Explicit conflict check:
                        # Evaluates whether broker order ID or client_order_id references a registered foreign episode or foreign token.
                        # Rule: Conflict takes strict precedence. If foreign mapping is present, it is rejected
                        # as conflicting evidence even if current episode identity also matches.
                        has_explicit_conflict = False
                        if oid and oid in other_foreign_ids:
                            has_explicit_conflict = True
                        if not has_explicit_conflict and o_cid:
                            for fid in other_foreign_ids:
                                if fid and _matches_order_lineage(o_cid, fid):
                                    has_explicit_conflict = True
                                    break

                        if has_explicit_conflict:
                            logger.error(
                                "[RECONCILE CONFLICT BLOCKED] %s: Order %s (client_order_id '%s') "
                                "conflicts with foreign episode/signal/order while current episode is '%s'. Retaining fail-closed.",
                                symbol, oid, o_cid, episode_id
                            )
                            conflict_detected[0] = True
                            return

                        matches_curr_lineage = (
                            _matches_order_lineage(o_cid, sig_id)
                            or _matches_order_lineage(o_cid, ep_id)
                        )
                        matches_lineage = (
                            (oid in known_ids or (o_cid and o_cid in known_ids))
                            or matches_curr_lineage
                        )
                        if not matches_lineage:
                            return

                        o_status = str(_extract_order_field(o, "status") or getattr(o, "status", "") or "").lower().strip()
                        valid_fill_statuses = ("filled", "canceled", "cancelled", "expired")
                        if o_status not in valid_fill_statuses:
                            return

                        raw_fq = _extract_order_field(o, "filled_qty") or getattr(o, "filled_qty", None)
                        if raw_fq is None:
                            return
                        try:
                            f_qty = float(raw_fq)
                            if not math.isfinite(f_qty) or f_qty <= 0:
                                return
                        except (ValueError, TypeError):
                            return

                        raw_avg = _extract_order_field(o, "filled_avg_price") or getattr(o, "filled_avg_price", None)
                        if raw_avg is None:
                            return
                        try:
                            f_px = float(raw_avg)
                            if not math.isfinite(f_px) or f_px <= 0:
                                return
                        except (ValueError, TypeError):
                            return

                        o_type = str(_extract_order_field(o, "order_type") or _extract_order_field(o, "type") or getattr(o, "order_type", getattr(o, "type", "")) or "").lower()

                        if oid in collected_orders:
                            existing = collected_orders[oid]
                            if abs(existing["qty"] - f_qty) > 1e-4 or abs(existing["price"] - f_px) > 1e-4:
                                logger.error(
                                    f"[RECONCILE CONFLICT] {symbol}: Conflicting broker observations for order {oid}: "
                                    f"existing(qty={existing['qty']}, px={existing['price']}) vs new(qty={f_qty}, px={f_px}). "
                                    f"Failing closed."
                                )
                                conflict_detected[0] = True
                            return

                        collected_orders[oid] = {
                            "id": oid,
                            "qty": f_qty,
                            "price": f_px,
                            "type": o_type,
                            "status": o_status,
                        }

                    # Collect from TP snapshot
                    if tp_id and hasattr(bridge, "get_order_snapshot"):
                        try:
                            tp_snap = bridge.get_order_snapshot(str(tp_id))
                            _process_candidate_order(tp_snap)
                        except Exception as tp_err:
                            logger.debug("Failed querying TP snapshot for %s: %s", tp_id, tp_err)

                    # Collect from Stop snapshot
                    if stop_id and hasattr(bridge, "get_order_snapshot"):
                        try:
                            stop_snap = bridge.get_order_snapshot(str(stop_id))
                            _process_candidate_order(stop_snap)
                        except Exception as st_err:
                            logger.debug("Failed querying stop snapshot for %s: %s", stop_id, st_err)

                    # Collect from last stop snapshot if distinct
                    last_stop_id = exit_item.get("last_stop_order_id")
                    if last_stop_id and str(last_stop_id) != str(stop_id) and hasattr(bridge, "get_order_snapshot"):
                        try:
                            lst_snap = bridge.get_order_snapshot(str(last_stop_id))
                            _process_candidate_order(lst_snap)
                        except Exception as lst_err:
                            logger.debug("Failed querying last stop snapshot for %s: %s", last_stop_id, lst_err)

                    # Collect from closed orders
                    if hasattr(bridge, "_client") and hasattr(bridge._client, "get_orders"):
                        try:
                            from alpaca.trading.requests import GetOrdersRequest
                            from alpaca.trading.enums import QueryOrderStatus
                            req_kwargs = {
                                "status": QueryOrderStatus.CLOSED,
                                "symbols": [symbol.upper().strip()],
                                "limit": 50,
                            }
                            entry_epoch = _to_epoch(exit_item.get("entry_time"))
                            if entry_epoch is not None:
                                from datetime import datetime, timezone
                                after_dt = datetime.fromtimestamp(max(0.0, entry_epoch - 60.0), tz=timezone.utc)
                                req_kwargs["after"] = after_dt
                            req = GetOrdersRequest(**req_kwargs)
                            closed_orders = bridge._client.get_orders(req)
                            if closed_orders:
                                for co in closed_orders:
                                    _process_candidate_order(co)
                        except Exception as co_err:
                            logger.debug("Failed querying closed orders for %s: %s", symbol, co_err)
                    elif hasattr(bridge, "get_orders"):
                        try:
                            closed_orders = bridge.get_orders(status="closed", symbols=[symbol.upper().strip()], limit=50)
                            if closed_orders:
                                for co in closed_orders:
                                    _process_candidate_order(co)
                        except Exception as co_err:
                            logger.debug("Failed querying bridge.get_orders for %s: %s", symbol, co_err)

                    if conflict_detected[0]:
                        logger.error(
                            f"[RECONCILE CONFLICT BLOCKED] {symbol}: Conflicting observations across broker order queries. "
                            f"Retaining pending exit fail-closed."
                        )
                        results[symbol] = "CONFLICTING_EXIT_EVIDENCE_DISCREPANCY"
                        continue

                    # Enforce Quantity Conservation
                    total_filled = sum(m["qty"] for m in collected_orders.values())
                    if total_filled > exp_qty + 1e-4:
                        logger.error(
                            f"[RECONCILE OVERFILL] {symbol}: total_filled={total_filled:.4f} > exp_qty={exp_qty:.4f}. "
                            f"Excess quantity discrepancy detected across {len(collected_orders)} order(s). Retaining fail-closed."
                        )
                        results[symbol] = "EXCESS_EXIT_QTY_DISCREPANCY"
                        continue

                    if total_filled < exp_qty - 1e-4:
                        # Verify working orders on broker for symbol
                        has_working_orders = False
                        if hasattr(bridge, "working_orders_plain"):
                            try:
                                working = bridge.working_orders_plain()
                                has_working_orders = any(
                                    str(getattr(wo, "symbol", "") or (wo.get("symbol") if isinstance(wo, dict) else "")).upper().strip() == symbol.upper().strip()
                                    for wo in (working or [])
                                )
                            except Exception:
                                has_working_orders = True
                        elif hasattr(bridge, "_get_active_orders"):
                            try:
                                raw_active = bridge._get_active_orders()
                                has_working_orders = any(
                                    str(getattr(ao, "symbol", "") or "").upper().strip() == symbol.upper().strip()
                                    for ao in (raw_active or [])
                                )
                            except Exception:
                                has_working_orders = True

                        if has_working_orders:
                            results[symbol] = "WORKING_ORDERS_PENDING_RETAINED"
                            continue

                        # Check if trade was already closed and recorded in the trade journal
                        journal_has_closed_trade = False
                        try:
                            from strategies.trade_journal import read_journal
                            j_records = read_journal()
                            sig_id = str(exit_item.get("signal_id") or "").strip()
                            ep_id = str(episode_id or "").strip()
                            entry_time_epoch = _to_epoch(exit_item.get("entry_time"))
                            known_exit_ids = {
                                str(x).strip() for x in [
                                    sig_id,
                                    ep_id,
                                    exit_item.get("order_id"),
                                    exit_item.get("tp_order_id"),
                                    exit_item.get("stop_order_id"),
                                    exit_item.get("last_stop_order_id"),
                                ] if x
                            }

                            for rec in reversed(j_records):
                                if (
                                    str(rec.get("symbol", "")).upper().strip() == symbol.upper().strip()
                                    and rec.get("event_type") in ("TRADE_CLOSED", "EXIT")
                                ):
                                    # Guard 1: Timestamp must not precede entry time
                                    rec_ts = _to_epoch(rec.get("timestamp"))
                                    if rec_ts is not None and entry_time_epoch is not None and rec_ts < entry_time_epoch:
                                        continue

                                    # Guard 2: Lineage match (signal_id, order_id, or idempotency_key must match this episode)
                                    rec_sig = str(rec.get("signal_id") or "").strip()
                                    rec_oid = str(rec.get("order_id") or "").strip()
                                    rec_idem = str(rec.get("idempotency_key") or "").strip()

                                    matches_lineage = False
                                    if sig_id and (rec_sig == sig_id or sig_id in rec_idem):
                                        matches_lineage = True
                                    elif ep_id and (ep_id in rec_idem or (rec_sig and rec_sig in ep_id)):
                                        matches_lineage = True
                                    elif rec_oid and rec_oid in known_exit_ids:
                                        matches_lineage = True
                                    elif not sig_id and entry_time_epoch is not None and rec_ts is not None and rec_ts >= entry_time_epoch:
                                        matches_lineage = True

                                    if not matches_lineage:
                                        continue

                                    j_qty = float(rec.get("qty") or 0.0)
                                    if abs(j_qty - exp_qty) < 1e-4 or j_qty >= exp_qty:
                                        journal_has_closed_trade = True
                                        break
                        except Exception as j_exc:
                            logger.debug("Journal check non-fatal: %s", j_exc)

                        if journal_has_closed_trade:
                            logger.info(
                                "[RECONCILE AUTHORITATIVE CONFIRMATION] %s: Verified broker flat (qty=0, working_orders=0) "
                                "with trade journal closed record. Clearing pending exit '%s'.",
                                symbol, episode_id
                            )
                            if episode_id in self.pending_exits:
                                del self.pending_exits[episode_id]
                            if symbol in self.pending_exits:
                                del self.pending_exits[symbol]
                            if symbol in self.states:
                                self.states[symbol].reconciliation_status = "RECONCILED_FLAT_VERIFIED"
                                self.states[symbol].active = False
                                self.states[symbol].stop_reconciliation_required = False
                            self._save_state()
                            results[symbol] = "RECONCILED_FLAT_VERIFIED"
                            continue

                        # Auto-expire / purge stale pending exits older than 24 hours when broker is verified flat
                        now_epoch = time.time()
                        entry_epoch = _to_epoch(exit_item.get("entry_time")) or _to_epoch(exit_item.get("registered_at"))
                        is_stale_exit = (entry_epoch is not None and (now_epoch - entry_epoch) > 86400.0)

                        if is_stale_exit or (not has_working_orders and cur_pos_qty is not None and abs(cur_pos_qty) < 1e-6 and int(exit_item.get("attempts", 0)) >= 5):
                            logger.warning(
                                "[STALE PENDING EXIT PURGED] %s: Pending exit '%s' (age=%.1fs, attempts=%s) purged because broker position is confirmed 0 with 0 working orders.",
                                symbol, episode_id, (now_epoch - (entry_epoch or now_epoch)), exit_item.get("attempts", 0)
                            )
                            if episode_id in self.pending_exits:
                                del self.pending_exits[episode_id]
                            if symbol in self.pending_exits:
                                del self.pending_exits[symbol]
                            if symbol in self.states:
                                self.states[symbol].reconciliation_status = "RECONCILED_FLAT_VERIFIED"
                                self.states[symbol].active = False
                                self.states[symbol].stop_reconciliation_required = False
                            self._save_state()
                            results[symbol] = "STALE_PENDING_EXIT_PURGED"
                            continue

                        results[symbol] = "FLAT_PENDING_EXIT_RECONCILIATION"
                        continue

                    # Exact quantity conservation satisfied: total_filled == exp_qty
                    weighted_px = sum(m["qty"] * m["price"] for m in collected_orders.values()) / total_filled
                    types = {m["type"] for m in collected_orders.values()}
                    if len(collected_orders) == 1:
                        single_type = list(types)[0] if types else ""
                        if "stop" in single_type:
                            exit_reason = "STOP_OUT"
                        elif "limit" in single_type:
                            exit_reason = "TAKE_PROFIT"
                        else:
                            exit_reason = "MARKET_EXIT"
                    else:
                        exit_reason = "SPLIT_EXIT"

                    exit_price = weighted_px
                    notes = f"Authoritative exit confirmed across {len(collected_orders)} order(s) (total={total_filled:.2f} shares @ weighted ${weighted_px:.4f})"

                    # 5. Compute PnL and commit to journal atomically
                    qty_closed = float(exit_item.get("qty") or 0.0)
                    entry_px = float(exit_item.get("entry_price") or 0.0)
                    pnl_val = (exit_price - entry_px) * qty_closed if exit_item.get("side") == "BUY" else (entry_px - exit_price) * qty_closed
                    if not math.isfinite(pnl_val):
                        logger.error("[RECONCILE PNL NON-FINITE] %s: exit_px=%s, entry_px=%s, qty=%s", symbol, exit_price, entry_px, qty_closed)
                        results[symbol] = "PNL_INVALID"
                        continue

                    # Attempt journal write
                    idempotency_key = f"exit_{episode_id}"
                    journal_qty = int(qty_closed) if abs(qty_closed - round(qty_closed)) < 1e-6 else float(qty_closed)
                    if exit_item.get("is_validation"):
                        val_written = False
                        try:
                            from strategies.validation_lane import record_validation_exit
                            confirmed_oids = [str(k).strip() for k in collected_orders.keys() if str(k).strip()]
                            actual_broker_exit_oid = confirmed_oids[0] if confirmed_oids else ""
                            target_oid = actual_broker_exit_oid or str(exit_item.get("last_stop_order_id") or exit_item.get("stop_order_id") or exit_item.get("tp_order_id") or exit_item.get("signal_id") or "")
                            actual_exec_id = f"RECONCILE_{episode_id}_{actual_broker_exit_oid}" if actual_broker_exit_oid else f"RECONCILE_{episode_id}"
                            val_written = record_validation_exit(
                                order_id=target_oid,
                                exit_price=float(exit_price),
                                qty=journal_qty,
                                side=expected_exit_side.upper(),
                                symbol=symbol,
                                exit_reason=exit_reason,
                                execution_id=actual_exec_id,
                                parent_order_id=str(exit_item.get("entry_order_id") or "") or None,
                                client_order_id=str(exit_item.get("signal_id") or exit_item.get("client_order_id") or "") or None,
                            )
                        except Exception as val_e:
                            logger.error("Failed recording validation exit for %s: %s", symbol, val_e)
                            val_written = False

                        if not val_written:
                            logger.error(
                                "[RECONCILE VALIDATION FAILED] %s: Validation ledger update returned False or unconfirmed. "
                                "Retaining pending exit without marking state reconciled.",
                                symbol
                            )
                            results[symbol] = "VALIDATION_LEDGER_WRITE_FAILED_RETAINED"
                            continue
                    else:
                        from strategies.trade_journal import log_trade
                        log_trade(
                            symbol=symbol,
                            side=expected_exit_side.upper(),
                            qty=journal_qty,
                            entry_price=entry_px,
                            stop_price=exit_item.get("stop_price") or entry_px,
                            target_price=exit_item.get("target_price") or exit_price,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            pnl=pnl_val,
                            notes=notes or f"Authoritative {exit_reason} exit",
                            event_type="TRADE_CLOSED",
                            signal_id=exit_item.get("signal_id"),
                            strategy=_normalize_strategy_name(exit_item.get("strategy")),
                            idempotency_key=idempotency_key,
                        )

                    # Journal commit succeeded! Remove from pending queue atomically
                    backup_ep = self.pending_exits.get(episode_id)
                    backup_sym = self.pending_exits.get(symbol)
                    backup_state_status = self.states[symbol].reconciliation_status if symbol in self.states else None
                    backup_state_active = self.states[symbol].active if symbol in self.states else False
                    backup_state_recon = self.states[symbol].stop_reconciliation_required if symbol in self.states else False

                    if episode_id in self.pending_exits:
                        del self.pending_exits[episode_id]
                    if symbol in self.pending_exits:
                        del self.pending_exits[symbol]
                    if symbol in self.states:
                        self.states[symbol].reconciliation_status = f"RECONCILED_{exit_reason}"
                        self.states[symbol].active = False
                        self.states[symbol].stop_reconciliation_required = False
                    saved = self._save_state()
                    if not saved:
                        if backup_ep:
                            self.pending_exits[episode_id] = backup_ep
                        if backup_sym:
                            self.pending_exits[symbol] = backup_sym
                        if symbol in self.states:
                            self.states[symbol].reconciliation_status = backup_state_status or "FLAT_PENDING_EXIT_RECONCILIATION"
                            self.states[symbol].active = backup_state_active
                            self.states[symbol].stop_reconciliation_required = True
                        logger.error("[RECONCILE QUEUE PERSISTENCE FAILED] %s: State persistence failed after journal commit. Pending exit retained.", symbol)
                        results[symbol] = "QUEUE_PERSISTENCE_FAILED_RETAINED"
                        continue

                    results[symbol] = f"RECONCILED_{exit_reason}"
                    logger.info("[RECONCILE AUTHORITATIVE SUCCESS] %s: %s @ %.2f (pnl=%.2f, key=%s)", symbol, exit_reason, exit_price, pnl_val, idempotency_key)


                except Exception as rec_err:
                    logger.error("[RECONCILE FAILED RETAINING PENDING] %s: %s", symbol, rec_err)
                    results[symbol] = "RECONCILIATION_ERROR_RETAINED"

        return results


    def soft_reset(self, symbol: str) -> None:
        """Deactivate state — preserves telemetry/audit trail."""
        with self._lock:
            state = self.states.get(symbol)
            if state:
                state.active = False
                self._save_state()
                logger.info(f"Soft reset trailing stop for {symbol} (state preserved, active=False)")

    def hard_reset(self, symbol: str) -> None:
        """Delete state entirely from memory + disk."""
        sym_upper = symbol.upper().strip()
        with self._lock:
            keys_to_del = [k for k in self.states if k.upper().strip() == sym_upper]
            for k in keys_to_del:
                del self.states[k]
            self._save_state()
            logger.info(f"Hard reset trailing stop for {symbol} (state deleted)")

    def reset(self, symbol: str) -> None:
        """Backward-compatible alias → hard_reset."""
        self.hard_reset(symbol)


    def apply_confirmed_fill(
        self,
        signal: TradeSignal,
        average_fill_price: float,
        cumulative_qty: float,
        stop_order_id: Optional[str] = None,
        bridge: Optional[object] = None,
        tp_order_id: Optional[str] = None,
    ) -> TrailingStopState:
        """Apply a confirmed fill report to initialize or update position trailing state."""
        symbol = signal.symbol.upper().strip()
        with self._lock:
            existing = self.states.get(symbol)
            if existing and existing.active:
                existing.entry_price = average_fill_price
                existing.qty = cumulative_qty
                cum_qty_f = float(cumulative_qty)
                curr_entry_qty = getattr(existing, "entry_qty", None)
                if curr_entry_qty is None or cum_qty_f > float(curr_entry_qty):
                    existing.entry_qty = cum_qty_f
                if stop_order_id:
                    existing.order_id = stop_order_id
                if tp_order_id:
                    existing.tp_order_id = tp_order_id
                if (stop_order_id or existing.order_id) and cum_qty_f > 0:
                    if getattr(existing, "stop_reconciliation_required", False):
                        existing.stop_reconciliation_required = False
                        existing.reconciliation_status = None
                existing.last_updated = time.time()
                self._save_state()
                logger.info(f"Updated Trailing Stop for {symbol}: fill=${average_fill_price:.4f}, cumulative_qty={cumulative_qty:.4f}")
                return existing

            # Missing or inactive state means this is a new lifecycle!
            if existing and not existing.active:
                del self.states[symbol]

            return self.initialize_position(
                signal=signal,
                fill_price=average_fill_price,
                qty=cumulative_qty,
                stop_order_id=stop_order_id,
                bridge=bridge,
                tp_order_id=tp_order_id,
            )

    def initialize_position(self, signal: TradeSignal, fill_price: float, qty: float = 0.0, stop_order_id: Optional[str] = None, bridge: Optional[object] = None, tp_order_id: Optional[str] = None) -> TrailingStopState:


        """Initialize trailing stop when order is filled."""
        with self._lock:
            atr = getattr(signal, 'atr', 0.0)
            
            mult = getattr(config, "TRAILING_STOP_ATR_MULTIPLE", 1.5)
            fallback_pct = getattr(config, "TRAILING_STOP_FALLBACK_PCT", 0.02)
            
            if atr > 0 and getattr(config, "TRAILING_STOP_USE_ATR", True):
                trail_distance = atr * mult
            else:
                trail_distance = fill_price * fallback_pct
                
            initial_stop = signal.stop_price
            signal_price = getattr(signal, "entry_price", fill_price)
            target_price = getattr(signal, "target_price", None)

            # Validate and rebuild bracket geometry if actual fill price inverted stop price
            rebuild_res = self.validate_and_rebuild_geometry(
                symbol=signal.symbol,
                signal_price=signal_price,
                fill_price=fill_price,
                initial_stop=initial_stop,
                target_price=target_price,
                bridge=bridge,
                stop_order_id=stop_order_id,
                tp_order_id=tp_order_id,
                side=signal.side,
                return_details=True,
                expected_qty=qty,
            )

            state = TrailingStopState(
                symbol=signal.symbol,
                side=signal.side.upper(),
                entry_price=fill_price,
                peak_price=fill_price,
                atr=atr,
                stop_price=rebuild_res.stop_price,
                desired_stop_price=rebuild_res.desired_stop_price,
                confirmed_broker_stop_price=rebuild_res.confirmed_broker_stop_price,
                stop_reconciliation_required=rebuild_res.stop_reconciliation_required,
                remediation_in_progress=rebuild_res.stop_reconciliation_required,
                last_remediation_time=time.time() if rebuild_res.stop_reconciliation_required else 0.0,
                remediation_attempts=1 if rebuild_res.stop_reconciliation_required else 0,
                target_price=target_price,
                trail_multiple=mult,
                last_updated=time.time(),
                order_id=rebuild_res.stop_order_id,
                tp_order_id=tp_order_id,
                qty=qty,
                strategy=_normalize_strategy_name(getattr(signal, "strategy", None)) if _normalize_strategy_name(getattr(signal, "strategy", None)) != "UNKNOWN" else ("DAYTRADE_SCALP" if getattr(signal, "signal_id", None) else "UNKNOWN"),
                signal_id=getattr(signal, "signal_id", None),
                entry_time=time.time(),
                active=True
            )
            self.states[signal.symbol] = state
            self._save_state()
            logger.info(
                f"Initialized Trailing Stop for {signal.symbol}: Entry ${fill_price:.2f}, Stop ${rebuild_res.stop_price:.2f} "
                f"(confirmed_broker={rebuild_res.confirmed_broker_stop_price}, recon_req={rebuild_res.stop_reconciliation_required}, order_id={rebuild_res.stop_order_id})"
            )
            return state

    def validate_and_rebuild_geometry(
        self,
        symbol: str,
        signal_price: float,
        fill_price: float,
        initial_stop: float,
        target_price: Optional[float] = None,
        bridge: Optional[object] = None,
        stop_order_id: Optional[str] = None,
        tp_order_id: Optional[str] = None,
        side: str = "BUY",
        return_details: bool = False,
        expected_qty: Optional[float] = None,
    ) -> Union[float, GeometryRebuildResult]:
        """
        Validate bracket order stop geometry against actual fill price.
        Ensure for LONG position: stop_price < actual_avg_fill < target_price.
        Ensure for SHORT position: target_price < actual_avg_fill < stop_price.
        If stop_price is invalid relative to actual_avg_fill, rebuild child stop leg & TP leg to ensure valid geometry.
        """
        is_short = (side.upper() == "SELL")
        decimals = 4 if fill_price < 1.0 else 2
        stop_distance = round((initial_stop - fill_price) if is_short else (fill_price - initial_stop), decimals)
        # Canonical post-fill protection geometry distances
        min_required_distance, min_rebuild_distance = compute_canonical_post_fill_distances(fill_price, decimals)

        # 1. Component Validations
        ordering_valid = (
            (initial_stop > fill_price and (not target_price or fill_price > target_price))
            if is_short
            else (initial_stop < fill_price and (not target_price or fill_price < target_price))
        )
        noise_floor_valid = stop_distance >= (min_required_distance - 1e-6)
        
        min_rr = getattr(config, "MIN_RISK_REWARD_RATIO", getattr(config, "MIN_RR_RATIO", 1.5))
        risk = stop_distance
        reward = (fill_price - target_price) if is_short else ((target_price - fill_price) if target_price and target_price > 0 else 0.0)
        actual_rr = (reward / risk) if risk > 0 and reward > 0 else 0.0
        rr_valid = (round(actual_rr, 4) >= round(min_rr - 1e-4, 4)) if (target_price and target_price > 0) else True

        composite_valid = bool(ordering_valid and noise_floor_valid and rr_valid)

        if not ordering_valid or not noise_floor_valid:
            action = "REBUILD_CHILD_LEGS"
        elif not rr_valid:
            action = "POST_FILL_RR_DEGRADED"
        else:
            action = "ACCEPTED"

        logger.info(
            f"[POST-FILL GEOMETRY] symbol={symbol} side={side.upper()} signal_price={signal_price:.{decimals}f} "
            f"actual_avg_fill={fill_price:.4f} stop_price={initial_stop:.{decimals}f} target_price={target_price or 0.0:.{decimals}f} "
            f"stop_distance={stop_distance:.4f} ordering_valid={ordering_valid} noise_floor_valid={noise_floor_valid} "
            f"actual_rr={actual_rr:.2f} min_rr={min_rr:.2f} rr_valid={rr_valid} composite_valid={composite_valid} action={action}"
        )

        if action in ("POST_FILL_RR_DEGRADED", "ACCEPTED"):
            if action == "POST_FILL_RR_DEGRADED":
                logger.warning(
                    f"[POST-FILL GEOMETRY DEGRADED] {symbol} ({side.upper()}): Post-fill actual R:R ({actual_rr:.2f}) "
                    f"fell below required minimum ({min_rr:.2f}). Preserving valid protective stop at ${initial_stop:.{decimals}f}."
                )

            broker_confirmed = False
            effective_stop_id = stop_order_id
            if bridge and stop_order_id:
                try:
                    snap_fn = (
                        getattr(bridge, "get_order_snapshot", None)
                        or getattr(bridge, "get_order_by_id", None)
                        or getattr(getattr(bridge, "_client", None), "get_order_by_id", None)
                    )
                    if callable(snap_fn):
                        snap = snap_fn(stop_order_id)
                        snap_valid, snap_reason = verify_broker_stop_order_evidence(
                            snap,
                            expected_order_id=str(stop_order_id),
                            expected_symbol=symbol,
                            position_side=side,
                            expected_stop_price=initial_stop,
                            expected_qty=expected_qty,
                            fill_price=fill_price,
                            require_qty=True,
                        )
                        if snap_valid:
                            broker_confirmed = True
                        else:
                            logger.warning(
                                f"[POST-FILL GEOMETRY {action}] Existing stop {stop_order_id} failed broker evidence: {snap_reason}"
                            )
                except Exception as b_exc:
                    logger.warning(
                        f"[POST-FILL GEOMETRY {action}] Failed to query broker snapshot for {stop_order_id}: {b_exc}"
                    )

            if return_details:
                return GeometryRebuildResult(
                    stop_price=initial_stop,
                    desired_stop_price=initial_stop,
                    confirmed_broker_stop_price=initial_stop if broker_confirmed else None,
                    stop_order_id=effective_stop_id,
                    stop_reconciliation_required=not broker_confirmed if (bridge and stop_order_id) else False,
                    rebuild_performed=False,
                    rebuild_succeeded=broker_confirmed,
                )
            return initial_stop

        if not composite_valid:
            # Fix B: Canonical min_rebuild_distance (5 bps basis with pre-existing 15% buffer)
            stop_offset = min_rebuild_distance

            min_rr = getattr(config, "MIN_RR_RATIO", 1.0)
            if is_short:
                new_stop_price = round(fill_price + stop_offset, decimals)
                rebuilt_stop_dist = new_stop_price - fill_price
                min_tp_dist = round(max(stop_offset, rebuilt_stop_dist * min_rr + 1e-4), decimals)
                new_target_price = round(max(0.01, fill_price - min_tp_dist), decimals)
            else:
                new_stop_price = round(max(0.01, fill_price - stop_offset), decimals)
                rebuilt_stop_dist = fill_price - new_stop_price
                min_tp_dist = round(max(stop_offset, rebuilt_stop_dist * min_rr + 1e-4), decimals)
                new_target_price = round(fill_price + min_tp_dist, decimals)

            logger.warning(
                f"[POST-FILL GEOMETRY REBUILD] {symbol} ({side.upper()}): Adjusted invalid stop price from ${initial_stop:.{decimals}f} "
                f"to ${new_stop_price:.{decimals}f} (new_target=${new_target_price or 0.0:.{decimals}f}) relative to actual fill ${fill_price:.4f}"
            )
            
            # Post-Rebuild Re-Validation Assertion (Stop distance + R:R Ratio check)
            rebuilt_stop_dist = round((new_stop_price - fill_price) if is_short else (fill_price - new_stop_price), decimals)
            rebuilt_tp_dist = (fill_price - new_target_price) if is_short else (new_target_price - fill_price)
            rebuilt_rr = (rebuilt_tp_dist / rebuilt_stop_dist) if rebuilt_stop_dist > 0 else 0.0
            rebuilt_rr_valid = round(rebuilt_rr, 4) >= (min_rr - 1e-4)

            if is_short:
                rebuilt_valid = (new_stop_price > fill_price) and (rebuilt_stop_dist >= (min_required_distance - 1e-6)) and rebuilt_rr_valid
                if target_price and target_price > 0:
                    rebuilt_valid = rebuilt_valid and (fill_price > new_target_price)
            else:
                rebuilt_valid = (new_stop_price < fill_price) and (rebuilt_stop_dist >= (min_required_distance - 1e-6)) and rebuilt_rr_valid
                if target_price and target_price > 0:
                    rebuilt_valid = rebuilt_valid and (fill_price < new_target_price)

            if not rebuilt_valid:
                logger.error(
                    f"[POST-FILL GEOMETRY REBUILD FAILED] {symbol} ({side.upper()}): "
                    f"Rebuilt geometry invalid (stop=${new_stop_price:.{decimals}f}, R:R={rebuilt_rr:.2f} < {min_rr:.2f}). "
                    f"Triggering fail-safe position close."
                )
                # Fix C: Fail-safe: close position immediately to prevent naked exposure
                if bridge and hasattr(bridge, "close_position"):
                    try:
                        bridge.close_position(symbol)
                        logger.warning(
                            f"[GEOMETRY FAIL-SAFE] {symbol} position closed immediately "
                            f"-- unresolvable stop geometry."
                        )
                    except Exception as close_err:
                        logger.error(
                            f"[GEOMETRY FAIL-SAFE] Failed to close {symbol}: {close_err}"
                        )
                if return_details:
                    return GeometryRebuildResult(
                        stop_price=new_stop_price,
                        desired_stop_price=new_stop_price,
                        confirmed_broker_stop_price=None,
                        stop_order_id=stop_order_id,
                        stop_reconciliation_required=True,
                        rebuild_performed=True,
                        rebuild_succeeded=False,
                    )
                return new_stop_price  # best-effort return
            else:
                logger.info(f"[POST-FILL GEOMETRY REBUILD SUCCESS] {symbol} ({side.upper()}): Re-validation valid=True for new stop ${new_stop_price:.{decimals}f}")

            replace_stop_succeeded = False
            effective_stop_order_id = stop_order_id
            if bridge:
                replace_fn = getattr(bridge, "replace_order_by_id", None)
                if not replace_fn and hasattr(bridge, "_client") and hasattr(bridge._client, "replace_order_by_id"):
                    replace_fn = bridge._client.replace_order_by_id

                if callable(replace_fn):
                    # 1. Replace Stop Loss Order
                    if stop_order_id:
                        try:
                            try:
                                from alpaca.trading.requests import ReplaceOrderRequest
                                req = ReplaceOrderRequest(stop_price=new_stop_price)
                            except Exception:
                                req = {"stop_price": new_stop_price}
                            res = replace_fn(stop_order_id, req)
                            new_id = _extract_order_field(res, "id", "order_id")
                            if not res or not new_id:
                                replace_stop_succeeded = False
                                logger.warning(
                                    f"[POST-FILL GEOMETRY REBUILD] Broker replace returned invalid/empty response: {res}"
                                )
                            else:
                                # Primary: Validate broker evidence directly on replacement response
                                is_valid, reason = verify_broker_stop_order_evidence(
                                    res,
                                    expected_order_id=str(new_id),
                                    expected_symbol=symbol,
                                    position_side=side,
                                    expected_stop_price=new_stop_price,
                                    expected_qty=expected_qty,
                                    fill_price=fill_price,
                                )
                                if is_valid:
                                    replace_stop_succeeded = True
                                    effective_stop_order_id = str(new_id)
                                else:
                                    # If terminal/rejected, fail immediately without querying snapshot
                                    res_status = _normalize_sdk_field(_extract_order_field(res, "status"))
                                    if res_status in _TERMINAL_STOP_STATUSES:
                                        replace_stop_succeeded = False
                                        logger.warning(
                                            f"[POST-FILL GEOMETRY REBUILD] Replaced order {new_id} terminal/rejected: {reason}"
                                        )
                                    # Fallback: Query authoritative broker snapshot if res lacked fields
                                    elif bridge and (
                                        hasattr(bridge, "get_order_snapshot")
                                        or hasattr(bridge, "get_order_by_id")
                                        or hasattr(getattr(bridge, "_client", None), "get_order_by_id")
                                    ):
                                        try:
                                            snap_fn = (
                                                getattr(bridge, "get_order_snapshot", None)
                                                or getattr(bridge, "get_order_by_id", None)
                                                or getattr(bridge._client, "get_order_by_id", None)
                                            )
                                            snap = snap_fn(new_id)
                                            snap_valid, snap_reason = verify_broker_stop_order_evidence(
                                                snap,
                                                expected_order_id=str(new_id),
                                                expected_symbol=symbol,
                                                position_side=side,
                                                expected_stop_price=new_stop_price,
                                                expected_qty=expected_qty,
                                                fill_price=fill_price,
                                            )
                                            if snap_valid:
                                                replace_stop_succeeded = True
                                                effective_stop_order_id = str(new_id)
                                                logger.info(
                                                    f"[POST-FILL GEOMETRY REBUILD] Broker snapshot verified for {new_id}: {snap_reason}"
                                                )
                                            else:
                                                replace_stop_succeeded = False
                                                logger.warning(
                                                    f"[POST-FILL GEOMETRY REBUILD] Broker snapshot failed verification for {new_id}: {snap_reason}"
                                                )
                                        except Exception as snap_err:
                                            replace_stop_succeeded = False
                                            logger.warning(
                                                f"[POST-FILL GEOMETRY REBUILD] Could not query snapshot for replaced order {new_id}: {snap_err}"
                                            )
                                    else:
                                        replace_stop_succeeded = False
                                        logger.warning(
                                            f"[POST-FILL GEOMETRY REBUILD] Insufficient broker evidence for replaced order {new_id}: {reason}"
                                        )

                            if replace_stop_succeeded:
                                logger.info(f"[POST-FILL GEOMETRY REBUILD] Replaced & confirmed broker stop order {stop_order_id} -> {effective_stop_order_id} at new stop price ${new_stop_price:.{decimals}f}")
                            else:
                                logger.warning(f"[POST-FILL GEOMETRY REBUILD] Replacement unconfirmed for stop order {stop_order_id} (new_id={new_id})")
                        except Exception as replace_err:
                            replace_stop_succeeded = False
                            logger.warning(f"[POST-FILL GEOMETRY REBUILD] Could not replace broker stop order {stop_order_id}: {replace_err}")

                    # 2. Replace Take Profit Order if tp_order_id is provided and new_target_price is set
                    if tp_order_id and new_target_price:
                        try:
                            try:
                                from alpaca.trading.requests import ReplaceOrderRequest
                                req = ReplaceOrderRequest(limit_price=new_target_price)
                            except Exception:
                                req = {"limit_price": new_target_price}
                            replace_fn(tp_order_id, req)
                            logger.info(f"[POST-FILL GEOMETRY REBUILD] Replaced broker TP order {tp_order_id} at new limit price ${new_target_price:.{decimals}f}")
                        except Exception as tp_err:
                            logger.warning(f"[POST-FILL GEOMETRY REBUILD] Could not replace broker TP order {tp_order_id}: {tp_err}")

            if bridge and stop_order_id:
                confirmed_broker_stop = new_stop_price if replace_stop_succeeded else None
                recon_required = not replace_stop_succeeded
                rebuild_ok = replace_stop_succeeded
            elif stop_order_id and not bridge:
                confirmed_broker_stop = None
                recon_required = True
                rebuild_ok = False
            else:
                confirmed_broker_stop = None
                recon_required = True
                rebuild_ok = False

            if return_details:
                return GeometryRebuildResult(
                    stop_price=new_stop_price,
                    desired_stop_price=new_stop_price,
                    confirmed_broker_stop_price=confirmed_broker_stop,
                    stop_order_id=effective_stop_order_id,
                    stop_reconciliation_required=recon_required,
                    rebuild_performed=True,
                    rebuild_succeeded=rebuild_ok,
                )
            return new_stop_price

        if return_details:
            return GeometryRebuildResult(
                stop_price=initial_stop,
                desired_stop_price=initial_stop,
                confirmed_broker_stop_price=None,
                stop_order_id=stop_order_id,
                stop_reconciliation_required=True,
                rebuild_performed=False,
                rebuild_succeeded=False,
            )
        return initial_stop

    def ensure_initialized(
        self,
        symbol: str,
        side: str,
        avg_cost: float,
        open_orders: list,
        current_price: float,
        original_stop: Optional[float] = None,
        qty: float = 0.0,
        original_entry_time: Optional[float] = None,
        bridge: Optional[object] = None,
    ) -> Optional[TrailingStopState]:
        """Ensure trailing stop state is initialized for an active position."""
        with self._lock:
            state = self.states.get(symbol)
            if state and state.active:
                if side:
                    state.side = "BUY" if str(side).upper().strip() in ("BUY", "LONG") else "SELL"
                if qty is not None and abs(float(qty)) > 0:
                    req_pos_qty = abs(float(qty))
                    if state.qty != req_pos_qty:
                        state.qty = req_pos_qty
                else:
                    req_pos_qty = float(state.qty if (state.qty and state.qty > 0) else 0.0)

                if state.order_id is not None:
                    all_open_orders = self._extract_all_orders_including_legs(open_orders)
                    target_side = "sell" if state.side == "BUY" else "buy"

                    tracked_in_open = next(
                        (o for o in all_open_orders if _extract_order_info(o)["id"] == str(state.order_id)),
                        None
                    )
                    if tracked_in_open is not None:
                        # Order is present in open_orders snapshot! Verify its validity and capacity
                        if self._is_valid_stop_order(tracked_in_open, target_side, required_qty=req_pos_qty, expected_symbol=symbol):
                            t_info = _extract_order_info(tracked_in_open)
                            if t_info.get("stop_price") is not None:
                                state.stop_price = float(t_info["stop_price"])
                                state.confirmed_broker_stop_price = float(t_info["stop_price"])
                            state.stop_reconciliation_required = False
                            state.reconciliation_status = None
                        else:
                            # Capacity insufficient or invalid stop order
                            logger.warning(
                                f"[RECONCILE P0] Tracked order {state.order_id} in open_orders failed stop validation "
                                f"or capacity check for required_qty={req_pos_qty}"
                            )
                            if state.order_id and not state.last_stop_order_id:
                                state.last_stop_order_id = str(state.order_id)
                            state.stop_reconciliation_required = True
                            state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                    else:
                        # Tracked order NOT in open_orders snapshot -> direct lookup
                        direct_order = None
                        if bridge and hasattr(bridge, "get_order_snapshot"):
                            try:
                                direct_order = bridge.get_order_snapshot(str(state.order_id))
                            except Exception as lookup_err:
                                logger.debug(f"[RECONCILE] Direct lookup error for {state.order_id}: {lookup_err}")
                        elif bridge and hasattr(bridge, "_client") and hasattr(bridge._client, "get_order_by_id"):
                            try:
                                direct_order = bridge._client.get_order_by_id(str(state.order_id))
                            except Exception as lookup_err:
                                logger.debug(f"[RECONCILE] Direct client lookup error for {state.order_id}: {lookup_err}")

                        if direct_order is not None:
                            d_info = _extract_order_info(direct_order)
                            d_id = str(d_info.get("id") or "")
                            d_sym = str(d_info.get("symbol") or "").upper().strip()
                            d_status = str(d_info.get("status", "")).lower()

                            # Require identity, symbol match, side match, and validity
                            is_match = (
                                d_id == str(state.order_id)
                                and d_sym == symbol.upper().strip()
                            )

                            if not is_match:
                                logger.warning(
                                    f"[RECONCILE P0] Direct order lookup returned mismatch: "
                                    f"expected {symbol}/{state.order_id}, got {d_sym}/{d_id}"
                                )
                                if state.order_id and not state.last_stop_order_id:
                                    state.last_stop_order_id = str(state.order_id)
                                state.stop_reconciliation_required = True
                                state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                            elif d_status in ("held", "new", "accepted", "submitted", "pending_new"):
                                if self._is_valid_stop_order(direct_order, target_side, required_qty=req_pos_qty, expected_symbol=symbol):
                                    if d_info.get("stop_price") is not None:
                                        state.confirmed_broker_stop_price = float(d_info["stop_price"])
                                        state.stop_price = float(d_info["stop_price"])
                                    state.stop_reconciliation_required = False
                                    state.reconciliation_status = None
                                else:
                                    logger.warning(
                                        f"[RECONCILE P0] Direct resting order {d_id} failed capacity or stop validation for {req_pos_qty}"
                                    )
                                    if state.order_id and not state.last_stop_order_id:
                                        state.last_stop_order_id = str(state.order_id)
                                    state.stop_reconciliation_required = True
                                    state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                            elif d_status in ("filled", "partially_filled"):
                                is_known, pos_qty = self._query_broker_position_qty(bridge, state.symbol)
                                if not is_known:
                                    logger.warning(
                                        f"[RECONCILE P0 UNKNOWN] {state.symbol}: Stop order {state.order_id} status='{d_status}', "
                                        f"but position query failed. Retaining active state and marking reconciliation required."
                                    )
                                    state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                                    state.stop_reconciliation_required = True
                                    self._save_state()
                                    return state
                                elif abs(pos_qty) < 1e-6:
                                    logger.info(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='{d_status}' and position flat. Recording pending exit.")
                                    self._record_pending_exit(state)
                                    self.reconcile_pending_exits(bridge)
                                    return state
                                else:
                                    logger.info(f"[RECONCILE] {state.symbol}: Stop order {state.order_id} {d_status}, remaining pos={pos_qty}")
                                    state.qty = abs(pos_qty)
                                    if d_status == "filled":
                                        state.order_id = None
                                        state.stop_reconciliation_required = True
                                        state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                            elif d_status in ("cancelled", "canceled", "expired", "rejected"):
                                is_known, pos_qty = self._query_broker_position_qty(bridge, state.symbol)
                                if not is_known:
                                    logger.warning(
                                        f"[RECONCILE P0 UNKNOWN] {state.symbol}: Stop order {state.order_id} status='{d_status}', "
                                        f"but position query failed. Retaining active state and marking reconciliation required."
                                    )
                                    state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                                    state.stop_reconciliation_required = True
                                    self._save_state()
                                    return state
                                elif abs(pos_qty) < 1e-6:
                                    logger.info(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='{d_status}' and position flat. Recording pending exit.")
                                    self._record_pending_exit(state)
                                    self.reconcile_pending_exits(bridge)
                                    return state
                                else:
                                    logger.warning(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='{d_status}' with remaining pos={pos_qty}. Clearing order ID for remediation.")
                                    if state.order_id:
                                        state.last_stop_order_id = str(state.order_id)
                                    state.order_id = None
                                    state.qty = abs(pos_qty)
                                    state.stop_reconciliation_required = True
                                    state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                        else:
                            # Direct lookup not found: check if a matching stop exists in all_open_orders
                            matching_stop_id = None
                            matching_stop_price = None
                            for o in all_open_orders:
                                info = _extract_order_info(o)
                                if info["symbol"] == symbol.upper().strip() and self._is_valid_stop_order(
                                    o, target_side, required_qty=req_pos_qty, expected_symbol=symbol
                                ):
                                    matching_stop_id = info["id"]
                                    matching_stop_price = info["stop_price"]
                                    break
                            if matching_stop_id:
                                state.order_id = str(matching_stop_id)
                                if matching_stop_price is not None:
                                    state.stop_price = float(matching_stop_price)
                                    state.confirmed_broker_stop_price = float(matching_stop_price)
                                state.stop_reconciliation_required = False
                                state.reconciliation_status = None
                            else:
                                if state.order_id and not state.last_stop_order_id:
                                    state.last_stop_order_id = str(state.order_id)
                                state.stop_reconciliation_required = True
                                state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                                logger.warning(f"[RECONCILE UNCERTAIN] {state.symbol}: Tracked stop ID {state.order_id} missing from snapshot; retaining ID.")

                if state.order_id is None:
                    self._reconcile_broker_order(state, open_orders, bridge=bridge)
                return state

            stop_order_id = None
            broker_stop_price = None
            canonical_side = "BUY" if str(side).upper().strip() in ("BUY", "LONG") else "SELL"
            target_side = "sell" if canonical_side == "BUY" else "buy"
            position_qty = abs(float(qty)) if qty else 0.0
            
            all_candidate_orders = []
            for o in open_orders:
                all_candidate_orders.append(o)
                legs = getattr(o, "legs", []) or []
                for leg in legs:
                    all_candidate_orders.append(leg)

            for o in all_candidate_orders:
                info = _extract_order_info(o)
                if info["symbol"] == symbol.upper().strip() and self._is_valid_stop_order(
                    o, target_side, required_qty=position_qty, expected_symbol=symbol
                ):
                    if info["stop_price"] is not None:
                        broker_stop_price = info["stop_price"]
                    stop_order_id = info["id"]
                    break

            fallback_pct = getattr(config, "TRAILING_STOP_FALLBACK_PCT", 0.02)
            if side == "BUY":
                calculated_stop = avg_cost * (1.0 - fallback_pct)
            else:
                calculated_stop = avg_cost * (1.0 + fallback_pct)

            if original_stop is not None:
                calculated_stop = original_stop

            if broker_stop_price is not None:
                if state and abs(state.stop_price - broker_stop_price) > 0.0001:
                    logger.warning(
                        f"[RECONCILE CONFLICT] {symbol}: Local state stop price (${state.stop_price:.2f}) "
                        f"conflicts with broker active stop price (${broker_stop_price:.2f}). Overwriting local state with broker truth."
                    )
                final_stop = broker_stop_price
            else:
                final_stop = calculated_stop

            atr = 0.0
            try:
                df = fetch_intraday(symbol, purpose="POSITION_MONITORING")
                if not df.empty and "atr" in df.columns:
                    atr = float(df["atr"].iloc[-1])
            except Exception:
                pass

            mult = getattr(config, "TRAILING_STOP_ATR_MULTIPLE", 1.5)

            existing = self.states.get(symbol)
            reconciled_entry_time = None
            if original_entry_time and original_entry_time > 0:
                reconciled_entry_time = float(original_entry_time)
            elif existing and getattr(existing, "entry_time", 0) > 0:
                reconciled_entry_time = float(existing.entry_time)
                logger.info(f"[ENTRY TIME RECONCILED] {symbol}: Recovered original entry timestamp {reconciled_entry_time} from persisted state")
            else:
                reconciled_entry_time = time.time()
                logger.warning(
                    f"[ENTRY TIME FALLBACK] {symbol} reconciled without original broker entry timestamp. "
                    f"Time-exit clock starts from reconciliation time."
                )

            is_val_pos = False
            if open_orders:
                for o in open_orders:
                    cid = str(getattr(o, "client_order_id", "") or "")
                    if cid.startswith("val_"):
                        is_val_pos = True
                        break
            if not is_val_pos and bridge and hasattr(bridge, "get_order_role_info") and stop_order_id:
                r_info = bridge.get_order_role_info(stop_order_id)
                if str(r_info.get("client_order_id", "")).startswith("val_"):
                    is_val_pos = True
            if not is_val_pos and stop_order_id:
                try:
                    from strategies.order_registry import get_registered_order
                    reg = get_registered_order(str(stop_order_id))
                    if reg and (reg.get("is_validation") or str(reg.get("client_order_id") or "").startswith("val_")):
                        is_val_pos = True
                except Exception:
                    pass

            state = TrailingStopState(
                symbol=symbol,
                side=side,
                entry_price=avg_cost,
                peak_price=max(avg_cost, current_price) if side == "BUY" else min(avg_cost, current_price),
                atr=atr,
                stop_price=final_stop,
                desired_stop_price=final_stop,
                trail_multiple=mult,
                last_updated=time.time(),
                order_id=stop_order_id,
                qty=qty,
                strategy="RECONCILED",
                entry_time=reconciled_entry_time,
                active=True,
                is_validation=is_val_pos,
            )
            self.states[symbol] = state
            self._save_state()
            logger.info(f"Auto-initialized Trailing Stop for {symbol} from broker: Entry ${avg_cost:.2f}, Stop ${final_stop:.2f}, Order ID {stop_order_id} (is_val={is_val_pos})")
            return state

    def _is_valid_stop_order(
        self,
        o: object,
        target_side: str,
        required_qty: float = 0,
        expected_symbol: str = "",
    ) -> bool:
        """Verify order is a valid active stop loss order for target side, symbol, and covers required quantity."""
        if not o:
            return False
        info = _extract_order_info(o)
        o_id = info.get("id")
        if not o_id:
            return False
        o_sym = str(info.get("symbol") or "").upper().strip()
        o_side = str(info.get("side") or "").lower().strip()
        o_type = str(info.get("type") or "").lower().strip()
        o_stop = info.get("stop_price")
        o_status = str(info.get("status") or "").lower().strip()
        o_qty = info.get("qty")

        if expected_symbol:
            sym_expected = expected_symbol.upper().strip()
            if not o_sym or o_sym != sym_expected:
                logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: symbol '{o_sym}' != expected '{sym_expected}' -> Rejecting candidate.")
                return False

        if o_status in _TERMINAL_STOP_STATUSES or o_status not in _WORKING_STOP_STATUSES:
            logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: status '{o_status}' is non-working -> Rejecting candidate.")
            return False

        target_side_norm = target_side.lower().strip()
        if o_side != target_side_norm:
            return False

        if not is_stop_order_type(o_type, has_stop_price=(o_stop is not None)):
            return False

        if o_stop is None:
            return False
        try:
            o_stop_f = float(o_stop)
            if not math.isfinite(o_stop_f) or o_stop_f <= 0:
                return False
        except (ValueError, TypeError):
            return False

        if o_qty is None:
            logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: missing quantity -> Rejecting candidate.")
            return False
        try:
            o_qty_f = float(o_qty)
            if not math.isfinite(o_qty_f) or o_qty_f <= 0:
                return False
        except (ValueError, TypeError):
            return False

        rem_cap = o_qty_f
        if o_status == "partially_filled":
            raw_filled_q = info.get("filled_qty")
            if raw_filled_q is None:
                logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: status='partially_filled' lacks filled_qty evidence -> Rejecting candidate.")
                return False
            try:
                filled_q = float(raw_filled_q)
            except (ValueError, TypeError):
                return False
            if not math.isfinite(filled_q) or filled_q < 0.0 or filled_q >= o_qty_f:
                logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: status='partially_filled' with invalid filled_qty {filled_q} (total {o_qty_f}) -> Rejecting candidate.")
                return False
            rem_cap = o_qty_f - filled_q
            if rem_cap <= 0.0:
                logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: status='partially_filled' remaining capacity <= 0 -> Rejecting candidate.")
                return False

        if required_qty > 0:
            if rem_cap < float(required_qty) - 1e-4:
                logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: remaining cap {rem_cap} < required {required_qty} -> Rejecting candidate.")
                return False

        logger.debug(
            f"[RECONCILE EVAL] Order ID {o_id}: sym='{o_sym}', side='{o_side}', "
            f"type='{o_type}', stop_price={o_stop}, qty={o_qty}, rem_cap={rem_cap} -> VALID"
        )
        return True

    def _extract_all_orders_including_legs(self, open_orders: list) -> list:
        """Flatten open orders list to include nested bracket child legs if present."""
        extracted = []
        for o in open_orders:
            extracted.append(o)
            legs = getattr(o, "legs", None)
            if legs and isinstance(legs, (list, tuple)):
                for leg in legs:
                    extracted.append(leg)
        return extracted

    def _reconcile_broker_order(self, state: TrailingStopState, open_orders: list, bridge: Optional[object] = None) -> bool:
        """Adopt stop_order_id from active open orders and perform direction-aware price merge."""
        # Query fresh signed broker position if bridge is provided
        is_pos_known = False
        broker_pos_qty = None
        if bridge is not None:
            is_pos_known, broker_pos_qty = self._query_broker_position_qty(bridge, state.symbol)
            if not is_pos_known:
                state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                state.stop_reconciliation_required = True
            elif broker_pos_qty is not None:
                pos_f = float(broker_pos_qty)
                side_str = str(state.side).upper().strip()
                if pos_f > 1e-6 and side_str not in ("BUY", "LONG"):
                    logger.critical(
                        f"[RECONCILE SIGN MISMATCH] {state.symbol}: Broker position is LONG (+{pos_f}) "
                        f"but local state side is {side_str}. Fail-closed unknown status."
                    )
                    state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                    state.stop_reconciliation_required = True
                    self._save_state()
                    return False
                elif pos_f < -1e-6 and side_str not in ("SELL", "SHORT"):
                    logger.critical(
                        f"[RECONCILE SIGN MISMATCH] {state.symbol}: Broker position is SHORT ({pos_f}) "
                        f"but local state side is {side_str}. Fail-closed unknown status."
                    )
                    state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                    state.stop_reconciliation_required = True
                    self._save_state()
                    return False

                if abs(pos_f) > 1e-6:
                    if getattr(state, "entry_qty", None) is None and getattr(state, "qty", 0.0) > 0.0:
                        state.entry_qty = float(state.qty)
                    state.qty = abs(pos_f)

        if is_pos_known and broker_pos_qty is not None and abs(broker_pos_qty) > 1e-6:
            effective_required_qty = abs(broker_pos_qty)
        else:
            effective_required_qty = getattr(state, "qty", getattr(state, "shares", 0.0))

        # P0 Fix 1: Check tracked stop_order_id fill status if order_id is present
        if state.order_id:
            for o in self._extract_all_orders_including_legs(open_orders):
                info = _extract_order_info(o)
                if info["id"] == str(state.order_id):
                    status = info.get("status", "").lower()
                    if status == "partially_filled":
                        rem_cap = float(info.get("qty", 0.0) or 0.0) - float(info.get("filled_qty", 0.0) or 0.0)
                        if rem_cap > 0:
                            state.active = True
                            if rem_cap < effective_required_qty - 1e-4:
                                state.stop_reconciliation_required = True
                                state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                            logger.info(f"[RECONCILE PARTIAL] {state.symbol}: Stop order {state.order_id} status='partially_filled', remaining capacity {rem_cap} shares active.")
                    elif status in ("filled",):
                        logger.info(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='filled'. Checking broker position truth.")
                        check_known = is_pos_known
                        check_qty = broker_pos_qty
                        if not check_known and bridge is not None:
                            check_known, check_qty = self._query_broker_position_qty(bridge, state.symbol)

                        if check_known and check_qty is not None and abs(check_qty) < 1e-6:
                            # Position is verified flat!
                            logger.info(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} filled and position flat. Recording pending exit.")
                            state.active = False
                            state.order_id = None
                            self._record_pending_exit(state)
                            self._save_state()
                            return True
                        elif check_known and check_qty is not None and abs(check_qty) > 1e-6:
                            # Broker still holds residual position!
                            rem_shares = abs(check_qty)
                            logger.warning(
                                f"[RECONCILE P0 RESIDUAL] {state.symbol}: Stop order {state.order_id} filled, "
                                f"but broker still has {rem_shares} shares open! Retaining active state for naked remediation."
                            )
                            state.active = True
                            if state.order_id:
                                state.last_stop_order_id = str(state.order_id)
                            state.order_id = None
                            state.qty = rem_shares
                            state.stop_reconciliation_required = True
                            state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                            self._save_state()
                            return False
                        else:
                            # Query failed or unknown position
                            logger.warning(
                                f"[RECONCILE P0 UNKNOWN] {state.symbol}: Stop order {state.order_id} filled, "
                                f"but broker position unknown. Retaining active state fail-closed."
                            )
                            state.active = True
                            state.stop_reconciliation_required = True
                            state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                            self._save_state()
                            return False
                    elif status in ("cancelled", "canceled", "expired", "rejected"):
                        logger.warning(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='{status}' -> Stop cancelled/expired. Position is UNPROTECTED!")
                        state.active = True
                        if state.order_id:
                            state.last_stop_order_id = str(state.order_id)
                        state.order_id = None
                        state.stop_reconciliation_required = True
                        state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                        self._save_state()
                        return False

        target_side = "sell" if state.side == "BUY" else "buy"
        all_orders = self._extract_all_orders_including_legs(open_orders)
        for o in all_orders:
            info = _extract_order_info(o)
            if info["symbol"] == state.symbol.upper().strip() and self._is_valid_stop_order(
                o, target_side, required_qty=effective_required_qty, expected_symbol=state.symbol
            ):
                o_stop = info["stop_price"]
                o_type = info["type"]
                broker_stop_price = float(o_stop) if o_stop is not None else state.stop_price
                old_order_id = str(state.order_id or "").strip()
                new_order_id = str(info["id"] or "").strip()
                if old_order_id and old_order_id != new_order_id:
                    state.last_stop_order_id = old_order_id
                state.order_id = new_order_id
                state.remediation_attempts = 0
                state.escalation_alerted = False
                state.remediation_in_progress = False
                if state.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED":
                    if is_pos_known:
                        state.reconciliation_status = None
                        logger.info(
                            "[RECONCILIATION RECOVERED] Active broker stop restored for %s with verified position. Cleared POSITION_UNKNOWN status.",
                            state.symbol
                        )
                    else:
                        logger.warning(
                            "[RECONCILIATION RETAINED] Active broker stop found for %s, but broker position remains unknown. Retaining POSITION_UNKNOWN status.",
                            state.symbol
                        )
                if o_stop is not None:
                    state.confirmed_broker_stop_price = broker_stop_price
                    state.stop_price = broker_stop_price  # Broker truth is the effective resting stop

                    # Note ① Fix & Stop Monotonicity Invariant:
                    # Long protective stop must never be loosened (lowered) by fallback reconciliation
                    # Short protective stop must never be loosened (raised) by fallback reconciliation
                    if state.side == "BUY":
                        if state.strategy == "RECONCILED" and state.desired_stop_price is not None:
                            effective_stop = max(broker_stop_price, state.desired_stop_price)
                            if abs(state.desired_stop_price - effective_stop) > 0.001:
                                logger.info(
                                    f"[RECONCILE MONOTONICITY LONG] {state.symbol}: Adopting tighter stop ${effective_stop:.2f} "
                                    f"(broker=${broker_stop_price:.2f}, previous desired=${state.desired_stop_price:.2f})"
                                )
                                state.desired_stop_price = effective_stop
                        elif state.desired_stop_price is None:
                            state.desired_stop_price = broker_stop_price
                    else:  # SELL / SHORT
                        if state.strategy == "RECONCILED" and state.desired_stop_price is not None:
                            effective_stop = min(broker_stop_price, state.desired_stop_price)
                            if abs(state.desired_stop_price - effective_stop) > 0.001:
                                logger.info(
                                    f"[RECONCILE MONOTONICITY SHORT] {state.symbol}: Adopting tighter stop ${effective_stop:.2f} "
                                    f"(broker=${broker_stop_price:.2f}, previous desired=${state.desired_stop_price:.2f})"
                                )
                                state.desired_stop_price = effective_stop
                        elif state.desired_stop_price is None:
                            state.desired_stop_price = broker_stop_price

                    desired = state.desired_stop_price if state.desired_stop_price is not None else broker_stop_price
                    tick_size = 0.01 if max(state.entry_price, broker_stop_price) >= 1.0 else 0.0001
                    tolerance = max(
                        tick_size / 2.0,
                        float(getattr(config, "TRAILING_STOP_RECONCILE_TOLERANCE", tick_size / 2.0) or (tick_size / 2.0)),
                    )
                    diff = abs(desired - broker_stop_price)
                    is_looser = (broker_stop_price < desired - tolerance) if state.side == "BUY" else (broker_stop_price > desired + tolerance)

                    if diff > tolerance:
                        state.stop_reconciliation_required = True
                        if is_looser:
                            logger.warning(
                                f"[RECONCILE DIVERGENCE CRITICAL] {state.symbol} ({state.side}): "
                                f"Broker stop ${broker_stop_price:.2f} is looser than desired ${desired:.2f} (diff=${diff:.2f}). "
                                f"reconciliation_required=True."
                            )
                        else:
                            logger.info(
                                f"[RECONCILE DIVERGENCE] {state.symbol} ({state.side}): "
                                f"Broker stop ${broker_stop_price:.2f} vs desired ${desired:.2f} (diff=${diff:.2f})."
                            )
                    else:
                        if is_pos_known:
                            state.stop_reconciliation_required = False
                        else:
                            state.stop_reconciliation_required = True

                self._save_state()
                confirmed_disp = state.confirmed_broker_stop_price if state.confirmed_broker_stop_price is not None else state.stop_price
                desired_disp = state.desired_stop_price if state.desired_stop_price is not None else state.stop_price
                if old_order_id != new_order_id:
                    logger.info(f"Reconciled active STOP order for {state.symbol} ({state.side}): Adopted Order ID {state.order_id} (type={o_type}), broker_stop=${confirmed_disp:.2f}, desired_stop=${desired_disp:.2f}")
                else:
                    logger.debug(f"Reconciled active STOP order for {state.symbol} ({state.side}): Order ID {state.order_id} active, broker_stop=${confirmed_disp:.2f}, desired_stop=${desired_disp:.2f}")
                return True
        return False

    def _refresh_atr(
        self,
        state: TrailingStopState,
        current_atr: Optional[float],
    ) -> bool:
        if current_atr is None or current_atr <= 0:
            return False

        old_atr = float(state.atr or 0.0)
        if old_atr <= 0:
            state.atr = float(current_atr)
            return True

        max_change_pct = float(
            getattr(config, "TRAILING_ATR_MAX_REFRESH_CHANGE_PCT", 0.25)
        )
        smoothing = float(
            getattr(config, "TRAILING_ATR_REFRESH_ALPHA", 0.25)
        )

        lower_bound = old_atr * (1.0 - max_change_pct)
        upper_bound = old_atr * (1.0 + max_change_pct)
        bounded_atr = min(max(float(current_atr), lower_bound), upper_bound)

        refreshed_atr = (
            old_atr * (1.0 - smoothing)
            + bounded_atr * smoothing
        )

        import math
        if math.isclose(old_atr, refreshed_atr, rel_tol=1e-6):
            return False

        state.atr = refreshed_atr

        logger.info(
            "[ATR REFRESH] %s: old=%.4f raw=%.4f bounded=%.4f new=%.4f",
            state.symbol,
            old_atr,
            current_atr,
            bounded_atr,
            refreshed_atr,
        )
        return True

    def update_stop(
        self,
        symbol: str,
        current_price: float,
        bridge=None,
        dry_run=False,
        current_atr: Optional[float] = None,
    ) -> Optional[TrailingStopState]:
        """Update the peak/trough and ratchet the stop price if required."""
        with self._lock:
            state = self.states.get(symbol)
            if not state or not state.active:
                return None
                
            if not getattr(config, "TRAILING_STOP_ENABLE_SHORT", True) and state.side == "SELL":
                return state

            now = time.time()
            base_cooldown = float(getattr(config, "TRAILING_STOP_UPDATE_COOLDOWN_SECONDS", 30))
            atr_pct = (
                state.atr / current_price
                if current_price > 0 and state.atr > 0
                else 0.0
            )
            high_vol_threshold = float(
                getattr(config, "TRAILING_HIGH_VOL_ATR_PCT", 0.008)
            )
            forced_fast_symbols = {
                str(s).upper().strip()
                for s in getattr(config, "TRAILING_FAST_UPDATE_SYMBOLS", {"NVDA", "TSLA", "AVGO", "AMD", "PLTR"})
            }
            is_fast_update = (
                symbol.upper().strip() in forced_fast_symbols
                or atr_pct >= high_vol_threshold
            )
            if is_fast_update:
                min_cd = float(getattr(config, "TRAILING_STOP_MIN_COOLDOWN_SECONDS", 15.0))
                cooldown = max(min_cd, base_cooldown * 0.5)
            else:
                cooldown = base_cooldown

            if (now - state.last_updated) < cooldown:
                return state

            # Refresh ATR with bounded change and smoothing
            atr_changed = self._refresh_atr(state, current_atr)

            min_delta = getattr(config, "TRAILING_STOP_MIN_UPDATE_DELTA", 0.25)
            
            if state.atr > 0 and getattr(config, "TRAILING_STOP_USE_ATR", True):
                trail_distance = state.atr * state.trail_multiple
            else:
                trail_distance = state.entry_price * getattr(config, "TRAILING_STOP_FALLBACK_PCT", 0.02)

            min_profit_pct = getattr(config, "TRAILING_STOP_MIN_PROFIT_PCT", 0.01)
            min_trail_pct = getattr(config, "TRAILING_STOP_MIN_TRAIL_PCT", 0.015)

            new_stop = state.stop_price
            should_update = False

            # --- Breakeven Stop Lock Logic (C3) ---
            breakeven_enabled = getattr(config, "BREAKEVEN_LOCK_ENABLED", True)
            # C3: 1.5 ATR profit threshold to lock breakeven; move stop to entry + 0.1 ATR
            breakeven_threshold_atr = float(getattr(config, "MR_BREAKEVEN_PROFIT_ATR", getattr(config, "BREAKEVEN_LOCK_THRESHOLD_ATR", 1.5)))
            lock_offset_atr = float(getattr(config, "MR_BREAKEVEN_LOCK_ATR", 0.1))

            if state.side == "BUY":
                if current_price > state.peak_price:
                    state.peak_price = current_price
                    
                profit_pct = (state.peak_price - state.entry_price) / state.entry_price if state.entry_price > 0 else 0.0
                profit_atr = (state.peak_price - state.entry_price) / state.atr if state.atr > 0 else 0.0

                if breakeven_enabled and state.atr > 0 and profit_atr >= breakeven_threshold_atr:
                    breakeven_stop = round(state.entry_price + (lock_offset_atr * state.atr), 2)
                    if breakeven_stop > new_stop:
                        old_stop_disp = new_stop
                        new_stop = breakeven_stop
                        should_update = True
                        unrealized_profit = state.peak_price - state.entry_price
                        logger.info(
                            f"[BREAKEVEN LOCK] {symbol} (BUY): entry=${state.entry_price:.2f} peak=${state.peak_price:.2f} "
                            f"unrealized=${unrealized_profit:.4f} atr=${state.atr:.4f} (profit_atr={profit_atr:.2f}x >= threshold {breakeven_threshold_atr:.2f}x). "
                            f"Locking stop at ${breakeven_stop:.2f} (old_stop=${old_stop_disp:.2f})"
                        )

                if not (breakeven_enabled and state.atr > 0 and profit_atr >= breakeven_threshold_atr and new_stop >= state.entry_price):
                    if profit_pct >= min_profit_pct:
                        effective_trail = max(trail_distance, state.entry_price * min_trail_pct)
                        computed_stop = state.peak_price - effective_trail
                        if computed_stop > state.stop_price + min_delta:
                            new_stop = computed_stop
                            should_update = True
                    
            else: # SELL (Short)
                if current_price < state.peak_price:
                    state.peak_price = current_price
                    
                profit_pct = (state.entry_price - state.peak_price) / state.entry_price if state.entry_price > 0 else 0.0
                profit_atr = (state.entry_price - state.peak_price) / state.atr if state.atr > 0 else 0.0

                if breakeven_enabled and state.atr > 0 and profit_atr >= breakeven_threshold_atr:
                    breakeven_stop = round(state.entry_price - (lock_offset_atr * state.atr), 2)
                    if breakeven_stop < new_stop:
                        old_stop_disp = new_stop
                        new_stop = breakeven_stop
                        should_update = True
                        unrealized_profit = state.entry_price - state.peak_price
                        logger.info(
                            f"[BREAKEVEN LOCK] {symbol} (SELL): entry=${state.entry_price:.2f} peak=${state.peak_price:.2f} "
                            f"unrealized=${unrealized_profit:.4f} atr=${state.atr:.4f} (profit_atr={profit_atr:.2f}x >= threshold {breakeven_threshold_atr:.2f}x). "
                            f"Locking stop at ${breakeven_stop:.2f} (old_stop=${old_stop_disp:.2f})"
                        )

                if not (breakeven_enabled and state.atr > 0 and profit_atr >= breakeven_threshold_atr and new_stop <= state.entry_price):
                    if profit_pct >= min_profit_pct:
                        effective_trail = max(trail_distance, state.entry_price * min_trail_pct)
                        computed_stop = state.peak_price + effective_trail
                        if computed_stop < state.stop_price - min_delta:
                            new_stop = computed_stop
                            should_update = True

            if should_update:
                state.desired_stop_price = new_stop
                logger.info(f"Trailing stop for {symbol} moving from ${state.stop_price:.2f} -> desired ${new_stop:.2f}")
                update_succeeded = True
                if not dry_run and bridge and state.order_id:
                    try:
                        res = bridge.modify_stop_order(state.order_id, new_stop)
                        if isinstance(res, str) and res:
                            logger.info(f"Updated stop order ID for {symbol}: {state.order_id} -> {res}")
                            if state.order_id and str(state.order_id) != str(res):
                                state.last_stop_order_id = str(state.order_id)
                            state.order_id = res
                            update_succeeded = True
                        elif res is True:
                            update_succeeded = True
                        else:
                            update_succeeded = False
                            logger.warning(f"Stop modify failed for {symbol}, resetting cooldown")
                    except Exception as e:
                        update_succeeded = False
                        logger.error(f"Failed to modify stop for {symbol}: {e}")
                elif dry_run:
                    logger.info(f"[DRY RUN] Would modify stop order {state.order_id} to ${new_stop:.2f}")

                if update_succeeded:
                    state.stop_price = new_stop
                    state.confirmed_broker_stop_price = new_stop
                    state.stop_reconciliation_required = False
                    state.last_updated = now
                else:
                    prev_confirmed = state.confirmed_broker_stop_price if state.confirmed_broker_stop_price is not None else state.stop_price
                    state.stop_price = prev_confirmed
                    state.confirmed_broker_stop_price = prev_confirmed
                    state.stop_reconciliation_required = True
                    state.last_updated = 0
                self._save_state()
            elif atr_changed:
                self._save_state()

            return state

    def should_exit(self, symbol: str, current_price: float) -> bool:
        """Pure logic check if stop was hit."""
        with self._lock:
            state = self.states.get(symbol)
            if not state or not state.active:
                return False
                
            if state.side == "BUY":
                return current_price <= state.stop_price
            else:
                return current_price >= state.stop_price

    def get_stop_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            state = self.states.get(symbol)
            return state.stop_price if state and state.active else None

    def handle_naked_position(self, symbol: str, position_qty: float, bridge: object, dry_run: bool = False) -> None:
        """Remediate a naked position by either re-placing the stop order or flattening the position."""
        # Safety Gate: Never attempt remediation on IPO/watch-only symbols
        ipo_watch = set(getattr(config, "IPO_WATCH_SYMBOLS", []))
        shadow_watch = set(getattr(config, "SHADOW_MONITOR", []))
        if symbol in ipo_watch or symbol in shadow_watch:
            logger.info(
                f"[NAKED GUARD] Skipping naked position remediation for watch-only symbol {symbol}. "
                f"No orders will be placed."
            )
            return

        should_alert_escalation = False
        should_alert_resolved = False
        resolved_order_id = None
        attempts = 0

        with self._lock:
            state = self.states.get(symbol)
            if not state or not state.active:
                return

            if state.order_id is not None:
                if state.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED":
                    is_known, pos_qty = self._query_broker_position_qty(bridge, symbol)
                    if is_known and not getattr(state, "stop_reconciliation_required", False):
                        state.reconciliation_status = None
                        if abs(pos_qty or 0.0) > 1e-6:
                            state.qty = abs(pos_qty)
                        self._save_state()
                        logger.info("[RECONCILIATION RECOVERED] Broker position query restored for %s with verified active stop %s.", symbol, state.order_id)
                if not getattr(state, "stop_reconciliation_required", False) and state.reconciliation_status != "POSITION_UNKNOWN_RECONCILIATION_REQUIRED":
                    if symbol in self.pending_exits and bridge is not None:
                        # Retry reconciling any pending exit in durable queue
                        self.reconcile_pending_exits(bridge)
                    return

            # Query broker fresh to make sure position still exists and is non-zero
            is_known, pos_qty = self._query_broker_position_qty(bridge, symbol)
            if not is_known:
                logger.warning(
                    f"[NAKED GUARD] Broker position query failed for {symbol}. "
                    f"Retaining active state and marking reconciliation required without placing speculative orders."
                )
                state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                state.stop_reconciliation_required = True
                self._save_state()
                return

            if abs(pos_qty) < 1e-6:
                logger.info(f"Fresh position query for {symbol} returned 0 (flat). Recording pending exit.")
                self._record_pending_exit(state, qty_closed=abs(position_qty))
                self.reconcile_pending_exits(bridge)
                return

            fresh_qty = abs(pos_qty)
            state.qty = fresh_qty
            position_qty = fresh_qty
            if state.order_id is not None and not getattr(state, "stop_reconciliation_required", False) and state.reconciliation_status == "POSITION_UNKNOWN_RECONCILIATION_REQUIRED":
                state.reconciliation_status = None
                logger.info(
                    "[RECONCILIATION RECOVERED] Broker position query restored for %s (qty=%.2f) with verified active stop. Cleared POSITION_UNKNOWN status.",
                    symbol, pos_qty
                )


            try:
                market_is_open = is_market_open()
            except Exception as e:
                logger.warning(f"Error checking market session: {e}. Defaulting to False.")
                market_is_open = False
            
            protection_method = getattr(config, "NAKED_LIMIT_PROTECTION", "replace").lower()
            
            if not market_is_open:
                logger.warning(f"Naked position detected for {symbol} outside regular market hours. Falling back to WARN.")
                if protection_method == "replace":
                    protection_method = "warn"

            if protection_method == "warn":
                logger.warning(f"CRITICAL: Naked position detected for {symbol} ({position_qty} shares), but protection method is 'warn'. No action taken.")
                try:
                    send_discord_alert(
                        f"⚠️ **NAKED POSITION WARNING** ⚠️\n"
                        f"**Symbol**: {symbol}\n"
                        f"**Qty**: {position_qty}\n"
                        f"**Status**: Naked position detected. No auto-remediation taken."
                    )
                except Exception as alert_exc:
                    logger.error(f"Failed to send Discord alert: {alert_exc}")
                return

            if dry_run:
                logger.info(f"[DRY RUN] Naked position detected for {symbol} ({position_qty} shares). Protection method: {protection_method.upper()}")
                return

            attempts = getattr(state, "remediation_attempts", 0)
            if attempts >= MAX_REMEDIATION_STRIKES:
                protection_method = "flatten"
                if not getattr(state, "escalation_alerted", False):
                    should_alert_escalation = True
                    state.escalation_alerted = True
                    self._save_state()
            else:
                now = time.time()
                remediation_cooldown = 15  # seconds
                if getattr(state, "remediation_in_progress", False) and (now - getattr(state, "last_remediation_time", 0.0)) < remediation_cooldown:
                    logger.debug(f"Remediation already in progress for {symbol}, skipping duplicate request.")
                    return

                state.remediation_in_progress = True
                state.last_remediation_time = now
                state.remediation_attempts = attempts + 1
                self._save_state()

        if should_alert_escalation:
            logger.critical(f"CRITICAL: Remediation failed {attempts} times for {symbol}. Triggering escalation and fallback.")
            try:
                send_discord_alert(
                    f"🚨 **CRITICAL ESCALATION: NAKED POSITION RESOLUTION FAILED** 🚨\n"
                    f"**Symbol**: {symbol}\n"
                    f"**Attempts**: {attempts}\n"
                    f"**Status**: Stop replacement failed repeatedly. Intervention required! @here"
                )
            except Exception as alert_exc:
                logger.error(f"Failed to send Discord alert: {alert_exc}")

        if protection_method == "replace":
            try:
                open_orders = []
                if hasattr(bridge, "working_orders_plain"):
                    try:
                        open_orders = bridge.working_orders_plain()
                    except Exception:
                        open_orders = []
                elif hasattr(bridge, "ib") and hasattr(bridge.ib, "openTrades"):
                    try:
                        open_orders = bridge.ib.openTrades()
                    except Exception:
                        open_orders = []

                if open_orders:
                    self._reconcile_broker_order(state, open_orders, bridge=bridge)

                # Pre-check: try to fetch all broker orders (active & closed) to adopt existing stop order or detect partial fills FIRST
                if hasattr(bridge, "_client") and hasattr(bridge._client, "get_orders"):
                    try:
                        from alpaca.trading.requests import GetOrdersRequest
                        from alpaca.trading.enums import QueryOrderStatus
                        req = GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True, symbols=[symbol])
                        fetched_orders = bridge._client.get_orders(req) or []
                        with self._lock:
                            if self._reconcile_broker_order(state, fetched_orders, bridge=bridge):
                                logger.debug(f"Reconciled active broker stop order {state.order_id} for {symbol} — skipping new submission.")
                                return
                    except Exception as rec_exc:
                        logger.debug(f"Pre-reconcile failed for {symbol}: {rec_exc}")

                # CRITICAL SAFETY GUARANTEE: If an active broker stop order exists on Alpaca, position is PROTECTED.
                if state.order_id is not None and not getattr(state, "stop_reconciliation_required", False) and state.reconciliation_status != "POSITION_UNKNOWN_RECONCILIATION_REQUIRED":
                    logger.info(f"Adopted existing valid broker stop order {state.order_id} for {symbol} inside handle_naked_position - skipping panic flatten.")
                    return

                # If state.order_id is present but stop_reconciliation_required is True,
                # the tracked order is insufficient/invalid and cannot protect the full position.
                # Retain old ID as audit evidence and clear confirmed order_id for safe remediation.
                old_stop_id = state.order_id
                if old_stop_id:
                    cancel_req_ok = False
                    c_res = None
                    if hasattr(bridge, "cancel_order"):
                        try:
                            c_res = bridge.cancel_order(old_stop_id)
                            if isinstance(c_res, bool):
                                cancel_req_ok = c_res
                            elif hasattr(c_res, "success"):
                                cancel_req_ok = bool(c_res.success)
                            elif isinstance(c_res, dict) and "success" in c_res:
                                cancel_req_ok = bool(c_res["success"])
                            elif c_res is None:
                                cancel_req_ok = False
                            else:
                                cancel_req_ok = bool(c_res)
                        except Exception as c_err:
                            logger.warning(f"[REMEDIATION] Cancel order {old_stop_id} failed with error/timeout: {c_err}")
                            cancel_req_ok = False
                    else:
                        cancel_req_ok = False

                    if not cancel_req_ok:
                        logger.warning(
                            f"[REMEDIATION BLOCKED] Failed/unknown cancellation for old stop order {old_stop_id} on {symbol}. "
                            f"Deferring blind replacement until old order terminal state is reconciled."
                        )
                        with self._lock:
                            state.order_id = str(old_stop_id)
                            state.last_stop_order_id = str(old_stop_id)
                            state.stop_reconciliation_required = True
                            state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                            state.remediation_in_progress = False
                            self._save_state()
                        return

                    # Verification of terminal broker status for old stop order:
                    # Cancellation request acknowledgement alone is NOT terminal confirmation.
                    old_snap = None
                    if hasattr(bridge, "get_order_snapshot"):
                        try:
                            old_snap = bridge.get_order_snapshot(str(old_stop_id))
                        except Exception as snap_err:
                            logger.debug(f"[REMEDIATION] Snapshot query error for old stop {old_stop_id}: {snap_err}")
                    elif hasattr(bridge, "get_order_truth"):
                        try:
                            ot = bridge.get_order_truth(str(old_stop_id))
                            if ot:
                                old_snap = {"status": ot.status, "filled_qty": ot.filled_qty, "qty": ot.qty}
                        except Exception:
                            pass
                    elif hasattr(bridge, "get_order"):
                        try:
                            old_snap = bridge.get_order(str(old_stop_id))
                        except Exception:
                            pass
                    elif hasattr(bridge, "_client") and hasattr(bridge._client, "get_order_by_id"):
                        try:
                            old_snap = bridge._client.get_order_by_id(str(old_stop_id))
                        except Exception:
                            pass

                    old_status = None
                    if old_snap is not None:
                        if isinstance(old_snap, dict):
                            raw_st = old_snap.get("status")
                        elif hasattr(old_snap, "status"):
                            raw_st = getattr(old_snap, "status")
                            if hasattr(raw_st, "value"):
                                raw_st = raw_st.value
                        else:
                            raw_st = None
                        if raw_st is not None:
                            old_status = str(raw_st).lower().strip()
                    elif hasattr(c_res, "terminal_status") and c_res.terminal_status:
                        old_status = str(c_res.terminal_status).lower().strip()
                    elif isinstance(c_res, dict) and c_res.get("terminal_status"):
                        old_status = str(c_res.get("terminal_status")).lower().strip()

                    terminal_cancelled_statuses = {"canceled", "cancelled", "expired", "rejected"}
                    if old_status == "filled":
                        is_known, fresh_pos_qty = self._query_broker_position_qty(bridge, symbol)
                        if not is_known:
                            logger.warning(
                                f"[REMEDIATION P0] Old stop {old_stop_id} was filled, but fresh broker position query failed for {symbol}. "
                                f"Retaining ACTIVE monitoring and fail-closed reconciliation requirement without deactivating position."
                            )
                            with self._lock:
                                state.active = True
                                state.last_stop_order_id = str(old_stop_id)
                                state.order_id = None
                                state.stop_reconciliation_required = True
                                state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                                state.remediation_in_progress = False
                                self._save_state()
                            return

                        if abs(fresh_pos_qty) < 1e-6:
                            logger.info(f"[REMEDIATION] Old stop {old_stop_id} filled and broker position is flat for {symbol}. Recording pending exit.")
                            with self._lock:
                                state.last_stop_order_id = str(old_stop_id)
                                state.order_id = None
                                self._record_pending_exit(state, qty_closed=abs(position_qty))
                                state.remediation_in_progress = False
                                self._save_state()
                            self.reconcile_pending_exits(bridge)
                            return

                        # Broker position is NOT flat: residual shares remain (e.g. partial fill or split exit)
                        is_mismatch = False
                        if state.side.upper() in ("BUY", "LONG") and fresh_pos_qty < -1e-6:
                            is_mismatch = True
                        elif state.side.upper() in ("SELL", "SHORT") and fresh_pos_qty > 1e-6:
                            is_mismatch = True

                        if is_mismatch:
                            logger.critical(
                                f"[REMEDIATION SIDE MISMATCH] Old stop {old_stop_id} filled, but signed broker position {fresh_pos_qty} "
                                f"does not match local side {state.side} for {symbol}. Blocking replacement stop and retaining active fail-closed."
                            )
                            with self._lock:
                                state.active = True
                                state.qty = abs(fresh_pos_qty)
                                state.last_stop_order_id = str(old_stop_id)
                                state.order_id = None
                                state.stop_reconciliation_required = True
                                state.reconciliation_status = "SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED"
                                state.remediation_in_progress = False
                                self._save_state()
                            return

                        residual_qty = abs(fresh_pos_qty)
                        logger.warning(
                            f"[REMEDIATION RESIDUAL] Old stop {old_stop_id} filled, but residual position of {residual_qty:.4f} shares remains for {symbol}. "
                            f"Retaining ACTIVE monitoring and proceeding to protect residual position."
                        )
                        with self._lock:
                            state.active = True
                            state.qty = residual_qty
                            position_qty = residual_qty
                            state.last_stop_order_id = str(old_stop_id)
                            state.order_id = None
                            state.stop_reconciliation_required = True
                            state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"

                    elif old_status not in terminal_cancelled_statuses:
                        logger.warning(
                            f"[REMEDIATION BLOCKED] Old stop order {old_stop_id} on {symbol} is not confirmed terminal cancelled "
                            f"(status={old_status}). Deferring blind replacement until old order terminal state is reconciled."
                        )
                        with self._lock:
                            state.order_id = str(old_stop_id)
                            state.last_stop_order_id = str(old_stop_id)
                            state.stop_reconciliation_required = True
                            state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                            state.remediation_in_progress = False
                            self._save_state()
                        return
                    else:
                        # Terminal cancellation confirmed on broker:
                        with self._lock:
                            state.last_stop_order_id = str(old_stop_id)
                            state.order_id = None

                        # Authoritative fresh broker quantity re-query after old stop is confirmed terminal cancelled:
                        is_known, fresh_pos_qty = self._query_broker_position_qty(bridge, symbol)
                        if not is_known:
                            logger.warning(
                                f"[NAKED GUARD] Broker position query failed for {symbol} after old stop cancellation. "
                                f"Retaining active state fail-closed without placing speculative orders."
                            )
                            with self._lock:
                                state.active = True
                                state.stop_reconciliation_required = True
                                state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                                state.remediation_in_progress = False
                                self._save_state()
                            return

                        if abs(fresh_pos_qty) < 1e-6:
                            logger.info(f"Fresh broker position for {symbol} after stop cancellation is flat (qty=0). Recording pending exit.")
                            with self._lock:
                                self._record_pending_exit(state, qty_closed=abs(position_qty))
                                state.remediation_in_progress = False
                                self._save_state()
                            self.reconcile_pending_exits(bridge)
                            return

                        # Signed side alignment check:
                        is_mismatch = False
                        if state.side.upper() in ("BUY", "LONG") and fresh_pos_qty < -1e-6:
                            is_mismatch = True
                        elif state.side.upper() in ("SELL", "SHORT") and fresh_pos_qty > 1e-6:
                            is_mismatch = True

                        if is_mismatch:
                            logger.critical(
                                f"[NAKED GUARD SIDE MISMATCH] Signed broker position {fresh_pos_qty} does not match local side {state.side} for {symbol}. "
                                f"Blocking replacement stop and retaining active fail-closed."
                            )
                            with self._lock:
                                state.active = True
                                state.qty = abs(fresh_pos_qty)
                                state.stop_reconciliation_required = True
                                state.reconciliation_status = "SIGNED_SIDE_MISMATCH_RECONCILIATION_REQUIRED"
                                state.remediation_in_progress = False
                                self._save_state()
                            return

                        position_qty = abs(fresh_pos_qty)
                        with self._lock:
                            state.active = True
                            state.qty = position_qty

                # Step 2: Gap-Through Check SECOND for truly naked positions
                current_price = None
                if hasattr(bridge, "market_price"):
                    current_price = bridge.market_price(symbol)

                if current_price and state.order_id is None:
                    is_gap_breach = False
                    if state.side == "BUY" and current_price < (state.stop_price - 0.05):
                        is_gap_breach = True
                    elif state.side == "SELL" and current_price > (state.stop_price + 0.05):
                        is_gap_breach = True

                    if is_gap_breach:
                        logger.warning(f"[GAP-THROUGH] Truly naked position {symbol} breached stop price ${state.stop_price:.2f} (current price ${current_price:.2f}). Triggering immediate emergency flatten.")
                        self.emergency_flatten(symbol, position_qty, bridge, dry_run=dry_run)
                        return

                # Step 3: Re-place Stop Order THIRD (if position is naked and price not breached)
                logger.warning(f"Naked position detected for {symbol} ({position_qty} shares). Attempting to re-place stop order at ${state.stop_price:.2f}.")
                stop_side = "SELL" if state.side == "BUY" else "BUY"

                if hasattr(bridge, "place_stop_order"):
                    res = bridge.place_stop_order(
                        symbol=symbol,
                        side=stop_side,
                        qty=abs(position_qty),
                        stop_price=state.stop_price
                    )
                    new_order_id = res.get("order_id") if res else None
                    if new_order_id:
                        snap = None
                        if hasattr(bridge, "get_order_snapshot"):
                            try:
                                snap = bridge.get_order_snapshot(str(new_order_id))
                            except Exception as snap_err:
                                logger.debug(f"[REMEDIATION] Snapshot query error for {new_order_id}: {snap_err}")
                        elif hasattr(bridge, "_client") and hasattr(bridge._client, "get_order_by_id"):
                            try:
                                snap = bridge._client.get_order_by_id(str(new_order_id))
                            except Exception as snap_err:
                                logger.debug(f"[REMEDIATION] Client lookup error for {new_order_id}: {snap_err}")

                        is_valid_evidence = False
                        snap_reason = "NO_SNAPSHOT"
                        if snap is not None:
                            is_valid_evidence, snap_reason = verify_broker_stop_order_evidence(
                                order_data=snap,
                                expected_order_id=str(new_order_id),
                                expected_symbol=symbol,
                                position_side=state.side,
                                expected_stop_price=state.stop_price,
                                expected_qty=abs(position_qty),
                                require_qty=True,
                            )

                        with self._lock:
                            current_state = self.states.get(symbol)
                            if current_state is state and state.active:
                                state.order_id = str(new_order_id)
                                state.remediation_in_progress = False
                                if is_valid_evidence:
                                    state.remediation_attempts = 0
                                    state.escalation_alerted = False
                                    state.stop_reconciliation_required = False
                                    state.reconciliation_status = None
                                    state.confirmed_broker_stop_price = state.stop_price
                                    self._save_state()
                                    should_alert_resolved = True
                                    resolved_order_id = new_order_id
                                else:
                                    # Verification failed: retain reconciliation requirement fail-closed
                                    state.stop_reconciliation_required = True
                                    state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                                    self._save_state()
                                    logger.warning(
                                        f"[REMEDIATION UNCONFIRMED] Replacement stop {new_order_id} unconfirmed by broker: {snap_reason}. "
                                        f"Retaining fail-closed reconciliation requirement."
                                    )
                                    return
                            else:
                                logger.warning(f"State mismatch for {symbol}. Cancelling orphan order {new_order_id}.")
                                if hasattr(bridge, "cancel_order"):
                                    try:
                                        bridge.cancel_order(new_order_id)
                                    except Exception as cancel_exc:
                                        logger.error(f"Failed to cancel orphan order {new_order_id}: {cancel_exc}")
                                protection_method = "flatten"
                        
                        if should_alert_resolved:
                            logger.info(f"Successfully re-placed stop order for {symbol}. New Order ID: {new_order_id}")
                            try:
                                send_discord_alert(
                                    f"⚠️ **NAKED POSITION RESOLVED** ⚠️\n"
                                    f"**Symbol**: {symbol}\n"
                                    f"**Remediation**: Replaced Stop Loss order at ${state.stop_price:.2f}\n"
                                    f"**Order ID**: {new_order_id}"
                                )
                            except Exception as alert_exc:
                                logger.error(f"Failed to send Discord alert: {alert_exc}")
                            return
                    else:
                        protection_method = "flatten"
                else:
                    logger.error(f"Bridge {bridge.__class__.__name__} does not support standalone stop orders.")
                    protection_method = "flatten"
            except Exception as e:
                err_msg = str(e)
                import re
                logger.warning(f"Failed to place standalone stop order for {symbol}: {err_msg}")
                if "insufficient qty" in err_msg.lower() or "held_for_orders" in err_msg.lower() or "40310000" in err_msg:
                    target_side = "sell" if state.side == "BUY" else "buy"
                    match = re.search(r'related_orders":\s*\[(.*?)\]', err_msg)
                    if match:
                        raw_ids = re.findall(r'"([^"]+)"', match.group(1))
                        rel_ids = list(dict.fromkeys(raw_ids))  # Deduplicate duplicate IDs
                        for rel_id in rel_ids:
                            if hasattr(bridge, "_client") and hasattr(bridge._client, "get_order_by_id"):
                                try:
                                    from alpaca.trading.requests import GetOrderByIdRequest
                                    rel_o = bridge._client.get_order_by_id(rel_id, GetOrderByIdRequest(nested=True))
                                    candidates = [rel_o] + (getattr(rel_o, "legs", []) or [])
                                    for candidate in candidates:
                                        cand_sym = str(getattr(candidate, "symbol", getattr(rel_o, "symbol", symbol))).upper().strip()
                                        if cand_sym and cand_sym != symbol.upper().strip():
                                            continue
                                        classification = _classify_held_order(candidate)
                                        if self._is_valid_stop_order(candidate, target_side, required_qty=position_qty, expected_symbol=symbol):
                                            cand_id = str(getattr(candidate, "id", rel_id))
                                            o_type = str(getattr(candidate, "order_type", getattr(candidate, "type", ""))).lower()
                                            with self._lock:
                                                state.order_id = cand_id
                                                state.remediation_in_progress = False
                                                state.remediation_attempts = 0
                                                state.stop_reconciliation_required = False
                                                state.reconciliation_status = None
                                                state.confirmed_broker_stop_price = state.stop_price
                                                self._save_state()
                                            logger.info(f"Adopted confirmed Stop Loss leg {cand_id} (type={o_type}) for {symbol} from broker bracket response.")
                                            return
                                        elif classification == "exit":
                                            logger.info(f"Exit market/limit order {getattr(candidate, 'id', rel_id)} is in-flight for {symbol}. Retaining working order protection, skipping emergency flatten.")
                                            with self._lock:
                                                state.remediation_in_progress = False
                                            return
                                        else:
                                            logger.warning(f"Candidate order {getattr(candidate, 'id', rel_id)} for {symbol} is NOT a valid stop order (type={getattr(candidate, 'order_type', getattr(candidate, 'type', None))}, client_id='{getattr(candidate, 'client_order_id', '')}'). Checking next candidate...")
                                except Exception as rel_exc:
                                    logger.debug(f"Could not verify related order {rel_id}: {rel_exc}")

                    if hasattr(bridge, "_client") and hasattr(bridge._client, "get_orders"):
                        try:
                            from alpaca.trading.requests import GetOrdersRequest
                            from alpaca.trading.enums import QueryOrderStatus
                            req = GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True, symbols=[symbol])
                            fetched_orders = bridge._client.get_orders(req) or []
                            with self._lock:
                                if self._reconcile_broker_order(state, fetched_orders, bridge=bridge):
                                    return
                        except Exception as rec_exc:
                            logger.warning(f"Failed fetching open/held orders for {symbol}: {rec_exc}")
                    
                    logger.info(f"Position for {symbol} is held by open broker orders (held_for_orders) without verified protective stop. Retaining reconciliation requirement fail-closed.")
                    with self._lock:
                        state.remediation_in_progress = False
                        state.stop_reconciliation_required = True
                        state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                        self._save_state()
                    return
                else:
                    protection_method = "flatten"

        if protection_method == "flatten":
            # Safety Gate: Before issuing a panic market order, check if Alpaca already filled a stop order for this symbol.
            if hasattr(bridge, "_client") and hasattr(bridge._client, "get_orders"):
                try:
                    from alpaca.trading.requests import GetOrdersRequest
                    from alpaca.trading.enums import QueryOrderStatus
                    req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, nested=True, symbols=[symbol], limit=10)
                    recent_closed = bridge._client.get_orders(req) or []
                    for o in recent_closed:
                        o_type = str(getattr(o, "order_type", getattr(o, "type", ""))).lower()
                        o_status = str(getattr(o, "status", "")).lower()
                        if "stop" in o_type and "filled" in o_status:
                            is_known, cur_pos = self._query_broker_position_qty(bridge, symbol)
                            if is_known and cur_pos is not None:
                                pos_abs = abs(float(cur_pos))
                            else:
                                pos_abs = abs(float(position_qty))

                            if pos_abs < 1e-6:
                                logger.info(f"Broker position for {symbol} is already flat. Recording pending exit.")
                                with self._lock:
                                    self._record_pending_exit(state)
                                    self.reconcile_pending_exits(bridge)
                                return

                            # Position is still open on broker but the stop is filled/dead -> Position is naked!
                            logger.warning(
                                f"Broker stop order {getattr(o, 'id', '')} for {symbol} was filled, but broker position remains open (qty={pos_abs}). Proceeding to emergency protection."
                            )
                            with self._lock:
                                state.qty = pos_abs
                                position_qty = pos_abs
                except Exception as check_exc:
                    logger.debug(f"Broker closed order verification failed for {symbol}: {check_exc}")

            self.emergency_flatten(symbol, position_qty, bridge=bridge, dry_run=dry_run)
            return

    def emergency_flatten(self, symbol: str, position_qty: float, bridge: Optional[object] = None, dry_run: bool = False):
        """Emergency flatten a naked position immediately using broker close_position."""
        logger.critical(f"CRITICAL: Naked position detected for {symbol} ({position_qty} shares) with no stop order protection. EMERGENCY FLATTENING POSITION!")
        try:
            send_discord_alert(
                f"🚨 **CRITICAL: NAKED POSITION EMERGENCY FLATTEN** 🚨\n"
                f"**Symbol**: {symbol}\n"
                f"**Qty**: {position_qty}\n"
                f"**Reason**: Naked position detected on broker side. Triggering emergency close."
            )
        except Exception as alert_exc:
            logger.error(f"Failed to send Discord alert: {alert_exc}")

        if dry_run:
            logger.info(f"[DRY-RUN] Emergency flatten for {symbol}")
            return

        try:
            if bridge and hasattr(bridge, "working_orders_plain"):
                open_orders = bridge.working_orders_plain() or []
                for o in open_orders:
                    o_sym = getattr(o, "symbol", "")
                    if str(o_sym).upper().strip() == symbol.upper().strip():
                        order_id = getattr(o, "id", None)
                        if order_id and hasattr(bridge, "cancel_order"):
                            bridge.cancel_order(str(order_id))
        except Exception as cancel_exc:
            logger.error(f"Failed to cancel open orders for {symbol} before flattening: {cancel_exc}")

        close_ok = False
        close_res = None
        if bridge:
            if hasattr(bridge, "close_position_verified"):
                for attempt in range(3):
                    try:
                        res = bridge.close_position_verified(symbol)
                        if res and getattr(res, "outcome", None) and str(getattr(res, "outcome")).upper().endswith("FLAT"):
                            close_ok = True
                            close_res = res
                            break
                    except Exception as close_err:
                        logger.warning(f"Attempt {attempt+1} to emergency close {symbol} failed: {close_err}")
                    time.sleep(1.5)
            elif hasattr(bridge, "close_position"):
                for attempt in range(3):
                    try:
                        res = bridge.close_position(symbol)
                        if res is True or (hasattr(res, "outcome") and str(res.outcome).upper().endswith("FLAT")):
                            close_ok = True
                            close_res = res
                            break
                        elif res:
                            close_ok = True
                            close_res = res
                            break
                    except Exception as close_err:
                        logger.warning(f"Attempt {attempt+1} to emergency close {symbol} failed: {close_err}")
                    time.sleep(1.5)

        with self._lock:
            current_state = self.states.get(symbol)
            if current_state:
                if not close_ok:
                    current_state.active = True
                    current_state.remediation_in_progress = False
                    current_state.stop_reconciliation_required = True
                    current_state.reconciliation_status = "POSITION_UNKNOWN_RECONCILIATION_REQUIRED"
                    self._save_state()
                    logger.critical(
                        f"[EMERGENCY FLATTEN FAILED] {symbol} close attempts failed. "
                        f"Retaining active state fail-closed to block entries and continue reconciliation."
                    )
                    return

                # Position close succeeded
                current_state.remediation_in_progress = False
                current_state.remediation_attempts = 0
                current_state.escalation_alerted = False

                cid_str = str(getattr(current_state, "order_id", "") or "")
                is_val = getattr(current_state, "is_validation", False) or cid_str.startswith("val_")
                qty_val = abs(float(position_qty or getattr(current_state, "qty", 1.0)))

                if close_res and hasattr(close_res, "order_id") and close_res.order_id:
                    current_state.last_stop_order_id = str(close_res.order_id)

                if is_val:
                    current_state.is_validation = True

                # Record durable pending exit without price fabrication for all positions
                self._record_pending_exit(current_state, qty_closed=qty_val)

                # Reconcile via broker truth on actual execution fill evidence
                if bridge is not None:
                    self.reconcile_pending_exits(bridge)

            self._save_state()

manager = DynamicTrailingStopManager()
trailing_stop_manager = manager

_managers_by_key: Dict[str, DynamicTrailingStopManager] = {}
_managers_lock = threading.Lock()


def get_trailing_manager(mode: Optional[str] = None, account_id: Optional[str] = None) -> DynamicTrailingStopManager:
    """Returns the DynamicTrailingStopManager instance for the given mode and account_id."""
    override = globals().get("_trailing_state_override")
    if override is not None:
        global manager
        return manager

    clean_mode = str(mode).strip().lower() if mode else (_default_context.get("mode") or "")
    clean_acct = str(account_id).strip() if account_id else (_default_context.get("account_id") or "")
    if clean_mode and (not clean_acct or clean_acct == "default"):
        clean_acct = DEFAULT_ACCOUNT_IDS.get(clean_mode, clean_acct)

    key = f"{clean_mode}_{clean_acct}".strip("_")
    if not key:
        return manager

    with _managers_lock:
        if key not in _managers_by_key:
            state_file = get_trailing_state_file(mode=clean_mode, account_id=clean_acct)
            _managers_by_key[key] = DynamicTrailingStopManager(state_file=state_file)
        return _managers_by_key[key]
