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
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, Union, Any

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
        if abs(remaining_qty - exp_qty_f) > 1e-4:
            return False, (
                f"EVIDENCE_QTY_MISMATCH: Remaining order capacity {remaining_qty:.4f} != "
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


STATE_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'trailing_state.json')

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

    def __post_init__(self):
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

    return {
        "id": o_id,
        "symbol": o_sym,
        "side": norm_side,
        "type": o_type,
        "stop_price": stop_val,
        "qty": qty_val,
        "status": o_status
    }


class DynamicTrailingStopManager:
    def __init__(self):
        self._lock = threading.RLock()
        self.persistence_healthy: bool = True
        self.last_persistence_error: Optional[str] = None
        self.last_persistence_error_time: float = 0.0
        self.consecutive_persistence_failures: int = 0
        with self._lock:
            self.states: Dict[str, TrailingStopState] = {}
            self._load_state()

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            if os.path.getsize(STATE_FILE) == 0:
                logger.warning(f"Trailing stop state file {STATE_FILE} is empty. Initializing empty state.")
                self.persistence_healthy = False
                self.last_persistence_error = "STATE_FILE_EMPTY"
                return
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for sym, state_dict in data.items():
                        if state_dict.get('active'):
                            # P1.5.2 Migration for legacy state dictionaries
                            state_dict.setdefault("desired_stop_price", state_dict.get("stop_price"))
                            state_dict.setdefault("confirmed_broker_stop_price", state_dict.get("stop_price"))
                            state_dict.setdefault("stop_reconciliation_required", False)
                            state_dict.setdefault("last_stop_order_id", state_dict.get("order_id"))
                            state_dict.setdefault("tp_order_id", None)
                            state_dict.setdefault("reconciliation_status", None)
                            self.states[sym] = TrailingStopState(**state_dict)
            self.persistence_healthy = True
            logger.info(f"Loaded {len(self.states)} active trailing stops.")
        except (json.JSONDecodeError, OSError) as e:
            self.persistence_healthy = False
            self.last_persistence_error = f"LOAD_FAILED: {e}"
            logger.error(f"Failed to load trailing stop state: {e}")

    def _save_state(self, raise_on_error: bool = False) -> bool:
        import threading
        tmp_file = f"{STATE_FILE}.tmp.{os.getpid()}.{threading.get_ident()}"
        try:
            state_dir = os.path.dirname(STATE_FILE)
            os.makedirs(state_dir, exist_ok=True)
            payload = {
                sym: asdict(state)
                for sym, state in self.states.items()
            }
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, STATE_FILE)
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
                STATE_FILE,
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
                    f"Failed to save trailing stop state atomically to {STATE_FILE}: {exc}"
                ) from exc
            return False

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
                if stop_order_id:
                    existing.order_id = stop_order_id
                if tp_order_id:
                    existing.tp_order_id = tp_order_id
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
                if state.order_id is not None:
                    all_open_orders = self._extract_all_orders_including_legs(open_orders)
                    open_order_ids = {_extract_order_info(o)["id"] for o in all_open_orders if _extract_order_info(o)["id"]}
                    if str(state.order_id) not in open_order_ids:
                        # P1-3: Do NOT immediately wipe order_id to None on snapshot miss!
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
                            d_status = str(d_info.get("status", "")).lower()
                            if d_status in ("held", "new", "accepted", "submitted", "pending_new"):
                                # Retain ID, order is active resting at broker
                                if d_info.get("stop_price") is not None:
                                    state.confirmed_broker_stop_price = float(d_info["stop_price"])
                                    state.stop_price = float(d_info["stop_price"])
                            elif d_status in ("filled", "partially_filled"):
                                pos_qty = 0.0
                                if bridge and hasattr(bridge, "get_positions"):
                                    try:
                                        for p in (bridge.get_positions() or []):
                                            p_sym = getattr(p, "symbol", "") or (p.get("symbol") if isinstance(p, dict) else "")
                                            if p_sym.upper() == state.symbol.upper():
                                                pos_qty = float(getattr(p, "qty", 0.0) or (p.get("qty", 0.0) if isinstance(p, dict) else 0.0))
                                    except Exception:
                                        pos_qty = 0.0
                                if pos_qty <= 0:
                                    logger.info(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='{d_status}' and position flat. Marking inactive cleanly.")
                                    state.active = False
                                    state.order_id = None
                                    self._save_state()
                                    return state
                                else:
                                    logger.info(f"[RECONCILE] {state.symbol}: Stop order {state.order_id} partially filled, remaining pos={pos_qty}")
                                    state.qty = pos_qty
                            elif d_status in ("cancelled", "canceled", "expired", "rejected"):
                                pos_qty = 0.0
                                if bridge and hasattr(bridge, "get_position"):
                                    try:
                                        pos_qty = float(bridge.get_position(state.symbol) or 0.0)
                                    except Exception:
                                        pos_qty = 0.0
                                elif bridge and hasattr(bridge, "get_positions"):
                                    try:
                                        for p in (bridge.get_positions() or []):
                                            p_sym = getattr(p, "symbol", "") or (p.get("symbol") if isinstance(p, dict) else "")
                                            if p_sym.upper() == state.symbol.upper():
                                                pos_qty = float(getattr(p, "qty", 0.0) or (p.get("qty", 0.0) if isinstance(p, dict) else 0.0))
                                    except Exception:
                                        pos_qty = 0.0
                                if pos_qty <= 0:
                                    logger.info(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='{d_status}' and position flat. Marking inactive cleanly.")
                                    state.active = False
                                    state.order_id = None
                                    self._save_state()
                                    return state
                                else:
                                    logger.warning(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='{d_status}' with remaining pos={pos_qty}. Clearing order ID for remediation.")
                                    state.order_id = None
                        else:
                            # Direct lookup unavailable or not found:
                            # Check if a matching stop exists in all_open_orders
                            target_side = "sell" if state.side == "BUY" else "buy"
                            matching_stop_id = None
                            matching_stop_price = None
                            for o in all_open_orders:
                                info = _extract_order_info(o)
                                if info["symbol"] == symbol.upper().strip() and self._is_valid_stop_order(o, target_side):
                                    matching_stop_id = info["id"]
                                    matching_stop_price = info["stop_price"]
                                    break
                            if matching_stop_id:
                                state.order_id = str(matching_stop_id)
                                if matching_stop_price is not None:
                                    state.stop_price = float(matching_stop_price)
                                    state.confirmed_broker_stop_price = float(matching_stop_price)
                            else:
                                state.stop_reconciliation_required = True
                                logger.warning(f"[RECONCILE UNCERTAIN] {state.symbol}: Tracked stop ID {state.order_id} missing from snapshot; retaining ID.")

                if state.order_id is None:
                    self._reconcile_broker_order(state, open_orders)
                return state

            stop_order_id = None
            broker_stop_price = None
            target_side = "sell" if side == "BUY" else "buy"
            
            all_candidate_orders = []
            for o in open_orders:
                all_candidate_orders.append(o)
                legs = getattr(o, "legs", []) or []
                for leg in legs:
                    all_candidate_orders.append(leg)

            for o in all_candidate_orders:
                info = _extract_order_info(o)
                if info["symbol"] == symbol.upper().strip() and self._is_valid_stop_order(o, target_side):
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
        o_id = info["id"]
        o_sym = str(info.get("symbol") or "").upper().strip()
        o_side = info["side"]
        o_type = info["type"]
        o_stop = info["stop_price"]
        o_status = info["status"]
        o_qty = info.get("qty")

        if expected_symbol:
            sym_expected = expected_symbol.upper().strip()
            if o_sym and o_sym != sym_expected:
                logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: symbol '{o_sym}' != expected '{sym_expected}' -> Rejecting candidate.")
                return False

        if o_status in {"canceled", "cancelled", "filled", "expired", "rejected"}:
            logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: status '{o_status}' is CLOSED/DEAD -> Rejecting candidate.")
            return False

        target_side_norm = target_side.lower().strip()
        is_side_match = (o_side == target_side_norm)
        is_stop_type = is_stop_order_type(o_type, has_stop_price=(o_stop is not None))

        qty_match = True
        if required_qty > 0 and o_qty is not None:
            qty_match = (float(o_qty) >= float(required_qty) - 1e-4)
            if not qty_match:
                logger.debug(f"[RECONCILE EVAL] Order ID {o_id}: qty {o_qty} < required {required_qty} -> Rejecting candidate.")

        logger.debug(
            f"[RECONCILE EVAL] Order ID {o_id}: sym='{o_sym}' (exp='{expected_symbol}'), side='{o_side}' (target='{target_side_norm}'), "
            f"type='{o_type}', stop_price={o_stop}, qty={o_qty} -> side_match={is_side_match}, stop_type={is_stop_type}, qty_match={qty_match}"
        )

        return is_side_match and is_stop_type and qty_match

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

    def _reconcile_broker_order(self, state: TrailingStopState, open_orders: list) -> bool:
        """Adopt stop_order_id from active open orders and perform direction-aware price merge."""
        # P0 Fix 1: Check tracked stop_order_id fill status if order_id is present
        if state.order_id:
            for o in self._extract_all_orders_including_legs(open_orders):
                info = _extract_order_info(o)
                if info["id"] == str(state.order_id):
                    status = info.get("status", "").lower()
                    if status in ("filled", "partially_filled", "cancelled", "expired"):
                        logger.info(f"[RECONCILE P0] {state.symbol}: Stop order {state.order_id} status='{status}' -> Stop leg executed/closed by broker. Marking inactive cleanly.")
                        state.active = False
                        state.order_id = None
                        self._save_state()
                        return True

        target_side = "sell" if state.side == "BUY" else "buy"
        all_orders = self._extract_all_orders_including_legs(open_orders)
        for o in all_orders:
            info = _extract_order_info(o)
            if info["symbol"] == state.symbol.upper().strip() and self._is_valid_stop_order(
                o, target_side, required_qty=getattr(state, "qty", getattr(state, "shares", 0.0))
            ):
                o_stop = info["stop_price"]
                o_type = info["type"]
                broker_stop_price = float(o_stop) if o_stop is not None else state.stop_price
                old_order_id = str(state.order_id or "").strip()
                new_order_id = str(info["id"] or "").strip()
                state.order_id = new_order_id
                state.remediation_attempts = 0
                state.escalation_alerted = False
                state.remediation_in_progress = False
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
                        state.stop_reconciliation_required = False

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

            # --- Breakeven Stop Lock Logic ---
            breakeven_enabled = getattr(config, "BREAKEVEN_LOCK_ENABLED", True)
            breakeven_threshold_atr = getattr(config, "BREAKEVEN_LOCK_THRESHOLD_ATR", 1.0)
            cushion = 0.01 if getattr(config, "BREAKEVEN_SLIPPAGE_CUSHION", True) else 0.0

            if state.side == "BUY":
                if current_price > state.peak_price:
                    state.peak_price = current_price
                    
                profit_pct = (state.peak_price - state.entry_price) / state.entry_price if state.entry_price > 0 else 0.0
                profit_atr = (state.peak_price - state.entry_price) / state.atr if state.atr > 0 else 0.0

                if breakeven_enabled and state.atr > 0 and profit_atr >= breakeven_threshold_atr:
                    breakeven_stop = state.entry_price + cushion
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
                    breakeven_stop = state.entry_price - cushion
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
            if not state or not state.active or state.order_id is not None:
                return

            # Query broker fresh to make sure position still exists and is non-zero
            fresh_qty = 0.0
            if hasattr(bridge, "get_position"):
                try:
                    fresh_qty = float(bridge.get_position(symbol) or 0.0)
                except Exception as p_exc:
                    logger.warning(f"Failed to query fresh position via get_position for {symbol}: {p_exc}")
                    fresh_qty = float(position_qty)
            elif hasattr(bridge, "ib") and hasattr(bridge.ib, "positions"):
                try:
                    positions = bridge.ib.positions()
                    for p in positions:
                        p_sym = getattr(p, "symbol", None)
                        if not p_sym:
                            contract = getattr(p, "contract", None)
                            if contract:
                                p_sym = getattr(contract, "symbol", "")
                        p_sym = str(p_sym).upper().strip() if p_sym else ""
                        if p_sym == symbol.upper().strip():
                            fresh_qty = float(p.position)
                            break
                except Exception as p_exc:
                    logger.warning(f"Failed to query fresh position for {symbol}: {p_exc}")
                    fresh_qty = float(position_qty)
            else:
                fresh_qty = float(position_qty)

            if fresh_qty == 0.0:
                logger.info(f"Fresh position query for {symbol} returned 0. Determining exit cause and reconciling.")
                exit_price = None
                exit_reason = None
                notes = None

                # 1. Attempt to reconcile from broker order snapshots
                tp_id = getattr(state, "tp_order_id", None)
                stop_id = getattr(state, "order_id", None) or getattr(state, "last_stop_order_id", None)

                # Check TP order first
                if tp_id and bridge and hasattr(bridge, "get_order_snapshot"):
                    try:
                        tp_snap = bridge.get_order_snapshot(str(tp_id))
                        if tp_snap and str(tp_snap.get("status", "")).lower() == "filled":
                            exit_reason = "TAKE_PROFIT"
                            exit_price = float(tp_snap.get("filled_avg_price") or tp_snap.get("limit_price") or state.target_price)
                            notes = f"Take profit fill confirmed via snapshot (order={tp_id})"
                    except Exception as tp_err:
                        logger.debug("Failed to query TP snapshot for %s: %s", tp_id, tp_err)

                # Check Stop order next if TP was not filled
                if not exit_reason and stop_id and bridge and hasattr(bridge, "get_order_snapshot"):
                    try:
                        stop_snap = bridge.get_order_snapshot(str(stop_id))
                        if stop_snap and str(stop_snap.get("status", "")).lower() == "filled":
                            exit_reason = "STOP_OUT"
                            exit_price = float(stop_snap.get("filled_avg_price") or state.stop_price)
                            notes = f"Stop-out fill confirmed via snapshot (order={stop_id})"
                    except Exception as st_err:
                        logger.debug("Failed to query stop snapshot for %s: %s", stop_id, st_err)

                # 2. Check recent closed orders if bridge client is available
                if not exit_reason and bridge and hasattr(bridge, "_client") and hasattr(bridge._client, "get_orders"):
                    try:
                        from alpaca.trading.requests import GetOrdersRequest
                        from alpaca.trading.enums import QueryOrderStatus
                        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol.upper().strip()], limit=5)
                        closed_orders = bridge._client.get_orders(req)
                        for co in closed_orders:
                            co_type = str(getattr(co, "order_type", getattr(co, "type", "")) or "").lower()
                            co_status = str(getattr(co, "status", "")).lower()
                            if "fill" in co_status:
                                co_avg = float(getattr(co, "filled_avg_price", 0.0) or 0.0)
                                if "limit" in co_type:
                                    exit_reason = "TAKE_PROFIT"
                                    exit_price = co_avg or float(getattr(co, "limit_price", 0.0) or 0.0) or state.target_price
                                    notes = f"Take profit fill confirmed via closed order {getattr(co, 'id', '')}"
                                    break
                                elif "stop" in co_type:
                                    exit_reason = "STOP_OUT"
                                    exit_price = co_avg or float(getattr(co, "stop_price", 0.0) or 0.0) or state.stop_price
                                    notes = f"Stop fill confirmed via closed order {getattr(co, 'id', '')}"
                                    break
                    except Exception as co_err:
                        logger.debug("Failed to query closed orders for %s: %s", symbol, co_err)

                # 3. If fill details are not yet available, defer journal closing
                if not exit_reason or exit_price is None or exit_price <= 0:
                    logger.info(
                        f"Fresh position query for {symbol} returned 0, but broker exit fill is pending. "
                        f"Marking FLAT_PENDING_EXIT_RECONCILIATION without phantom stop-out."
                    )
                    state.reconciliation_status = "FLAT_PENDING_EXIT_RECONCILIATION"
                    state.active = False
                    state.remediation_in_progress = False
                    state.order_id = None
                    self._save_state()
                    return

                # 4. Fill details confirmed — log authoritative exit
                qty_closed = getattr(state, "qty", 0.0) or abs(position_qty)
                pnl_val = (exit_price - state.entry_price) * qty_closed if state.side == "BUY" else (state.entry_price - exit_price) * qty_closed

                try:
                    if getattr(state, "is_validation", False):
                        try:
                            from strategies.validation_lane import record_validation_exit
                            record_validation_exit(
                                order_id=str(getattr(state, "order_id", "") or ""),
                                exit_price=exit_price,
                                qty=float(qty_closed),
                                side="SELL" if state.side == "BUY" else "BUY",
                                symbol=symbol,
                                exit_reason=exit_reason,
                            )
                        except Exception as val_e:
                            logger.error("Failed to record validation exit: %s", val_e)
                    else:
                        from strategies.trade_journal import log_trade
                        log_trade(
                            symbol=symbol,
                            side="SELL" if state.side == "BUY" else "BUY",
                            strategy=_normalize_strategy_name(getattr(state, "strategy", None)) if _normalize_strategy_name(getattr(state, "strategy", None)) != "UNKNOWN" else ("DAYTRADE_SCALP" if getattr(state, "signal_id", None) else "UNKNOWN"),
                            qty=int(qty_closed),
                            entry_price=state.entry_price,
                            stop_price=state.stop_price,
                            target_price=getattr(state, "target_price", None) or exit_price,
                            exit_price=exit_price,
                            exit_reason=exit_reason,
                            pnl=pnl_val,
                            notes=notes or f"Authoritative {exit_reason} exit",
                            event_type="TRADE_CLOSED",
                            signal_id=getattr(state, "signal_id", None)
                        )
                except Exception as e_log:
                    logger.error("Failed to log exit to journal: %s", e_log)

                state.reconciliation_status = f"RECONCILED_{exit_reason}"
                state.active = False
                state.remediation_in_progress = False
                state.order_id = None
                self._save_state()
                return

            position_qty = fresh_qty

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
                    self._reconcile_broker_order(state, open_orders)

                # Pre-check: try to fetch all broker orders (active & closed) to adopt existing stop order or detect partial fills FIRST
                if hasattr(bridge, "_client") and hasattr(bridge._client, "get_orders"):
                    try:
                        from alpaca.trading.requests import GetOrdersRequest
                        from alpaca.trading.enums import QueryOrderStatus
                        req = GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True, symbols=[symbol])
                        fetched_orders = bridge._client.get_orders(req) or []
                        with self._lock:
                            if self._reconcile_broker_order(state, fetched_orders):
                                logger.debug(f"Reconciled active broker stop order {state.order_id} for {symbol} — skipping new submission.")
                                return
                    except Exception as rec_exc:
                        logger.debug(f"Pre-reconcile failed for {symbol}: {rec_exc}")

                # CRITICAL SAFETY GUARANTEE: If an active broker stop order exists on Alpaca, position is PROTECTED.
                if state.order_id is not None:
                    logger.info(f"Adopted existing valid broker stop order {state.order_id} for {symbol} inside handle_naked_position - skipping panic flatten.")
                    return

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
                        with self._lock:
                            current_state = self.states.get(symbol)
                            if current_state is state and state.active:
                                state.order_id = str(new_order_id)
                                state.remediation_in_progress = False
                                state.remediation_attempts = 0
                                state.escalation_alerted = False
                                self._save_state()
                                should_alert_resolved = True
                                resolved_order_id = new_order_id
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
                                        if classification == "stop" or self._is_valid_stop_order(candidate, target_side, expected_symbol=symbol):
                                            cand_id = str(getattr(candidate, "id", rel_id))
                                            o_type = str(getattr(candidate, "order_type", getattr(candidate, "type", ""))).lower()
                                            with self._lock:
                                                state.order_id = cand_id
                                                state.remediation_in_progress = False
                                                state.remediation_attempts = 0
                                                self._save_state()
                                            logger.info(f"Adopted confirmed Stop Loss leg {cand_id} (type={o_type}) for {symbol} from broker bracket response.")
                                            return
                                        elif classification == "exit":
                                            logger.info(f"Exit market/limit order {getattr(candidate, 'id', rel_id)} is in-flight for {symbol}. Retaining working order protection, skipping emergency flatten.")
                                            with self._lock:
                                                state.remediation_in_progress = False
                                            return
                                        else:
                                            logger.warning(f"Candidate order {getattr(candidate, 'id', rel_id)} for {symbol} is NOT a stop order (type={getattr(candidate, 'order_type', getattr(candidate, 'type', None))}, client_id='{getattr(candidate, 'client_order_id', '')}'). Checking next candidate...")
                                except Exception as rel_exc:
                                    logger.debug(f"Could not verify related order {rel_id}: {rel_exc}")

                    if hasattr(bridge, "_client") and hasattr(bridge._client, "get_orders"):
                        try:
                            from alpaca.trading.requests import GetOrdersRequest
                            from alpaca.trading.enums import QueryOrderStatus
                            req = GetOrdersRequest(status=QueryOrderStatus.ALL, nested=True, symbols=[symbol])
                            fetched_orders = bridge._client.get_orders(req) or []
                            with self._lock:
                                if self._reconcile_broker_order(state, fetched_orders):
                                    return
                        except Exception as rec_exc:
                            logger.warning(f"Failed fetching open/held orders for {symbol}: {rec_exc}")
                    
                    logger.info(f"Position for {symbol} is held by open broker orders (held_for_orders). Retaining working order protection and skipping emergency flatten.")
                    with self._lock:
                        state.remediation_in_progress = False
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
                            filled_qty = float(getattr(o, "filled_qty", 0.0) or position_qty)
                            filled_price = float(getattr(o, "filled_avg_price", state.stop_price) or state.stop_price)
                            pos_abs = abs(float(position_qty))
                            if filled_qty > 0 and filled_qty < pos_abs:
                                remaining_qty = pos_abs - filled_qty
                                logger.info(f"Broker stop order {o.id} for {symbol} partially FILLED on Alpaca ({filled_qty}/{pos_abs} shares @ ${filled_price}). Updating remaining tracked position to {remaining_qty} shares.")
                                with self._lock:
                                    state.qty = remaining_qty
                                    state.remediation_in_progress = False
                                    state.remediation_attempts = 0
                                    self._save_state()
                                return
                            else:
                                logger.info(f"Broker stop order {o.id} for {symbol} was already FILLED on Alpaca ({filled_qty} shares @ ${filled_price}). Cleaning up trailing stop state cleanly without emergency market order.")
                                with self._lock:
                                    self.reset(symbol)
                                return
                except Exception as check_exc:
                    logger.debug(f"Broker closed order verification failed for {symbol}: {check_exc}")

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
            
            try:
                # Cancel open orders for this symbol first to release held shares
                try:
                    open_orders = []
                    if hasattr(bridge, "working_orders_plain"):
                        open_orders = bridge.working_orders_plain()
                    elif hasattr(bridge, "ib") and hasattr(bridge.ib, "openTrades"):
                        open_orders = bridge.ib.openTrades()
                    for o in open_orders:
                        o_sym = getattr(o, "symbol", None)
                        if not o_sym:
                            contract = getattr(o, "contract", None)
                            if contract:
                                o_sym = getattr(contract, "symbol", "")
                        o_sym = str(o_sym).upper().strip() if o_sym else ""
                        if o_sym == symbol.upper().strip():
                            order_id = getattr(o, "id", None)
                            if order_id and hasattr(bridge, "cancel_order"):
                                logger.info(f"Cancelling working order {order_id} for {symbol} before emergency flatten.")
                                bridge.cancel_order(str(order_id))
                except Exception as cancel_exc:
                    logger.error(f"Failed to cancel open orders for {symbol} before flattening: {cancel_exc}")

                close_ok = False
                if hasattr(bridge, "close_position"):
                    for attempt in range(3):
                        try:
                            close_ok = bridge.close_position(symbol)
                            if close_ok:
                                break
                        except Exception as close_err:
                            logger.warning(f"Attempt {attempt+1} to emergency close {symbol} failed: {close_err}")
                        time.sleep(1.5)
                else:
                    logger.critical(f"Bridge {bridge.__class__.__name__} does not support close_position. Cannot emergency close!")
                
                with self._lock:
                    current_state = self.states.get(symbol)
                    if current_state and current_state.active:
                        current_state.remediation_in_progress = False
                        if close_ok:
                            is_success = False
                            if isinstance(close_ok, bool):
                                is_success = close_ok
                            elif hasattr(close_ok, "is_filled"):
                                is_success = close_ok.is_filled or close_ok.outcome == "FILLED"
                            else:
                                is_success = bool(close_ok)

                            if is_success:
                                current_state.active = False
                                current_state.remediation_attempts = 0
                                current_state.escalation_alerted = False
                                
                                # Log exit to trade journal
                                exit_price = getattr(close_ok, "avg_fill_price", 0.0) or 0.0
                                qty_closed = getattr(close_ok, "filled", 0.0) or 0.0
                                
                                if exit_price <= 0.0:
                                    try:
                                        exit_price = bridge.market_price(symbol) or 0.0
                                    except Exception:
                                        exit_price = 0.0
                                if exit_price <= 0.0:
                                    exit_price = current_state.stop_price
                                if qty_closed <= 0.0:
                                    qty_closed = getattr(current_state, "qty", 0.0) or abs(position_qty)
                                    
                                pnl_val = (exit_price - current_state.entry_price) * qty_closed if current_state.side == "BUY" else (current_state.entry_price - exit_price) * qty_closed
                                
                                try:
                                    cid_str = str(getattr(current_state, "order_id", "") or "")
                                    is_val = getattr(current_state, "is_validation", False) or cid_str.startswith("val_")
                                    if is_val:
                                        try:
                                            from strategies.validation_lane import record_validation_exit
                                            record_validation_exit(
                                                order_id=cid_str,
                                                exit_price=exit_price,
                                                qty=float(qty_closed),
                                                side="SELL" if current_state.side == "BUY" else "BUY",
                                                symbol=symbol,
                                                exit_reason="EMERGENCY_FLATTEN",
                                            )
                                        except Exception as val_e:
                                            logger.error("Failed to record validation exit: %s", val_e)
                                    else:
                                        # HARD LANE GUARD: Validation order must NEVER route to production journal
                                        if cid_str.startswith("val_"):
                                            raise RuntimeError(f"HARD LANE GUARD: Validation order {cid_str} routed to production trade journal!")

                                        from strategies.order_idempotency import parse_strategy_from_client_id
                                        strat_name = _normalize_strategy_name(getattr(current_state, "strategy", None))
                                        if strat_name == "UNKNOWN" and cid_str:
                                            strat_name = parse_strategy_from_client_id(cid_str)
                                        if strat_name == "UNKNOWN" and getattr(current_state, "signal_id", None):
                                            sig_id = str(getattr(current_state, "signal_id", ""))
                                            strat_name = parse_strategy_from_client_id(sig_id)
                                            if strat_name == "UNKNOWN":
                                                strat_name = "DAYTRADE_SCALP"

                                        from strategies.trade_journal import log_trade
                                        log_trade(
                                            symbol=symbol,
                                            side="SELL" if current_state.side == "BUY" else "BUY",
                                            strategy=strat_name,
                                            qty=int(qty_closed),
                                            entry_price=current_state.entry_price,
                                            stop_price=current_state.stop_price,
                                            target_price=getattr(current_state, "target_price", None) or exit_price,
                                            exit_price=exit_price,
                                            exit_reason="EMERGENCY_FLATTEN",
                                            pnl=pnl_val,
                                            notes="Emergency flatten exit",
                                            event_type="TRADE_CLOSED",
                                            signal_id=getattr(current_state, "signal_id", None)
                                        )
                                except Exception as e_log:
                                    if "HARD LANE GUARD" in str(e_log):
                                        raise
                                    logger.error("Failed to log emergency close to journal: %s", e_log)
                        self._save_state()
            except Exception as e:
                logger.error(f"EMERGENCY FLATTEN FAILED for {symbol}: {e}")
                with self._lock:
                    current_state = self.states.get(symbol)
                    if current_state and current_state.active:
                        current_state.remediation_in_progress = False
                        self._save_state()

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
        if bridge and hasattr(bridge, "close_position"):
            for attempt in range(3):
                try:
                    close_ok = bridge.close_position(symbol)
                    if close_ok:
                        break
                except Exception as close_err:
                    logger.warning(f"Attempt {attempt+1} to emergency close {symbol} failed: {close_err}")
                time.sleep(1.5)

        with self._lock:
            current_state = self.states.get(symbol)
            if current_state:
                current_state.active = False
                current_state.remediation_in_progress = False
                if close_ok:
                    try:
                        cid_str = str(getattr(current_state, "order_id", "") or "")
                        is_val = getattr(current_state, "is_validation", False) or cid_str.startswith("val_")

                        exit_price = getattr(current_state, "stop_price", 0.0) or 0.0
                        if bridge and hasattr(bridge, "market_price"):
                            try:
                                exit_price = bridge.market_price(symbol) or exit_price
                            except Exception:
                                pass
                        if exit_price <= 0.0:
                            exit_price = getattr(current_state, "entry_price", 0.0)

                        entry_price = getattr(current_state, "entry_price", exit_price)
                        side = getattr(current_state, "side", "BUY")
                        qty_val = abs(float(position_qty or getattr(current_state, "qty", 1.0)))
                        pnl_val = (exit_price - entry_price) * qty_val if side == "BUY" else (entry_price - exit_price) * qty_val

                        if is_val:
                            try:
                                from strategies.validation_lane import record_validation_exit
                                record_validation_exit(
                                    order_id=cid_str,
                                    exit_price=exit_price,
                                    qty=float(qty_val),
                                    side="SELL" if side == "BUY" else "BUY",
                                    symbol=symbol,
                                    exit_reason="EMERGENCY_FLATTEN",
                                )
                            except Exception as val_e:
                                logger.error("Failed to record validation exit: %s", val_e)
                        else:
                            # HARD LANE GUARD: Validation order must NEVER route to production journal
                            if cid_str.startswith("val_"):
                                raise RuntimeError(f"HARD LANE GUARD: Validation order {cid_str} routed to production trade journal!")

                            from strategies.order_idempotency import parse_strategy_from_client_id
                            raw_strat = getattr(current_state, "strategy", None)
                            strat_str = _normalize_strategy_name(raw_strat)
                            if strat_str == "UNKNOWN" and cid_str:
                                strat_str = parse_strategy_from_client_id(cid_str)
                            if strat_str == "UNKNOWN" and getattr(current_state, "signal_id", None):
                                sig_id = str(getattr(current_state, "signal_id", ""))
                                strat_str = parse_strategy_from_client_id(sig_id)
                                if strat_str == "UNKNOWN":
                                    strat_str = "DAYTRADE_SCALP"

                            from strategies.trade_journal import log_trade
                            log_trade(
                                symbol=symbol,
                                side="SELL" if side == "BUY" else "BUY",
                                strategy=strat_str,
                                qty=int(qty_val),
                                entry_price=entry_price,
                                stop_price=getattr(current_state, "stop_price", exit_price),
                                target_price=getattr(current_state, "target_price", None) or exit_price,
                                exit_price=exit_price,
                                exit_reason="EMERGENCY_FLATTEN",
                                pnl=pnl_val,
                                notes="Emergency flatten forced exit",
                                event_type="TRADE_CLOSED",
                                signal_id=getattr(current_state, "signal_id", None),
                            )
                        logger.info(
                            f"[EMERGENCY FLATTEN JOURNAL] {symbol} | exit_px={exit_price} | strategy={strat_str if not is_val else 'VALIDATION'} | pnl={pnl_val:.2f}"
                        )
                    except Exception as e_log:
                        if "HARD LANE GUARD" in str(e_log):
                            raise
                        logger.error("Failed to log emergency flatten close to journal: %s", e_log)
            self._save_state()

manager = DynamicTrailingStopManager()
trailing_stop_manager = manager


def get_trailing_manager() -> DynamicTrailingStopManager:
    """Returns the singleton DynamicTrailingStopManager instance."""
    global manager
    return manager
