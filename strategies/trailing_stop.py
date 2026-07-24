"""
trailing_stop.py -- Dynamic ATR-based trailing stop manager.

Protects unrealized profit by ratcheting stops as price moves in our favor.
Never loosens a stop. Updates orders directly when live.
"""
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import config
from strategies.base import TradeSignal
from strategies.intraday_data import fetch_intraday
from strategies.session import is_market_open
from strategies.webhook import send_discord_alert

logger = logging.getLogger(__name__)

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

class DynamicTrailingStopManager:
    def __init__(self):
        self._lock = threading.Lock()
        with self._lock:
            self.states: Dict[str, TrailingStopState] = {}
            self._load_state()

    def _load_state(self):
        if not os.path.exists(STATE_FILE):
            return
        try:
            with open(STATE_FILE, "r") as f:
                data = json.load(f)
                for sym, state_dict in data.items():
                    if state_dict.get('active'):
                        self.states[sym] = TrailingStopState(**state_dict)
            logger.info(f"Loaded {len(self.states)} active trailing stops.")
        except Exception as e:
            logger.error(f"Failed to load trailing stop state: {e}")

    def _save_state(self):
        try:
            os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump({sym: asdict(state) for sym, state in self.states.items()}, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save trailing stop state: {e}")

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
            initial_stop = self.validate_and_rebuild_geometry(
                symbol=signal.symbol,
                signal_price=signal_price,
                fill_price=fill_price,
                initial_stop=initial_stop,
                target_price=target_price,
                bridge=bridge,
                stop_order_id=stop_order_id,
                tp_order_id=tp_order_id,
            )

            state = TrailingStopState(
                symbol=signal.symbol,
                side=signal.side.upper(),
                entry_price=fill_price,
                peak_price=fill_price,
                atr=atr,
                stop_price=initial_stop,
                trail_multiple=mult,
                last_updated=time.time(),
                order_id=stop_order_id,
                qty=qty,
                strategy=getattr(signal, "strategy", "UNKNOWN"),
                signal_id=getattr(signal, "signal_id", None),
                active=True
            )
            self.states[signal.symbol] = state
            self._save_state()
            logger.info(f"Initialized Trailing Stop for {signal.symbol}: Entry ${fill_price:.2f}, Stop ${initial_stop:.2f}")
            return state

    def validate_and_rebuild_geometry(self, symbol: str, signal_price: float, fill_price: float, initial_stop: float, target_price: Optional[float] = None, bridge: Optional[object] = None, stop_order_id: Optional[str] = None, tp_order_id: Optional[str] = None) -> float:
        """
        Validate bracket order stop geometry against actual fill price.
        Ensure for LONG position: stop_price < actual_avg_fill < target_price.
        If stop_price >= actual_avg_fill, rebuild child stop leg & TP leg to ensure valid geometry.
        """
        decimals = 4 if fill_price < 1.0 else 2
        stop_distance = fill_price - initial_stop
        min_required_distance = max(0.05, fill_price * 0.002)  # At least $0.05 or 0.2%
        
        is_valid = (initial_stop < fill_price) and (stop_distance >= min_required_distance)
        if target_price and target_price > 0:
            is_valid = is_valid and (fill_price < target_price)

        action = "ACCEPTED" if is_valid else "REBUILD_CHILD_LEGS"

        logger.info(
            f"[POST-FILL GEOMETRY] symbol={symbol} signal_price={signal_price:.{decimals}f} actual_avg_fill={fill_price:.4f} "
            f"stop_price={initial_stop:.{decimals}f} target_price={target_price or 0.0:.{decimals}f} stop_distance={stop_distance:.4f} "
            f"valid={is_valid} action={action}"
        )

        if not is_valid:
            if initial_stop >= fill_price:
                # Inverted Stop Case
                stop_offset = fill_price * 0.005 # 0.5% offset
            else:
                # Too-Tight Stop Case: enforce min_required_distance
                stop_offset = min_required_distance
            
            new_stop_price = round(max(0.01, fill_price - stop_offset), decimals)
            
            # Preserve Take-Profit R:R offset relative to actual fill if target_price is provided
            new_target_price = None
            if target_price and target_price > signal_price:
                tp_offset = target_price - signal_price
                new_target_price = round(fill_price + tp_offset, decimals)

            logger.warning(
                f"[POST-FILL GEOMETRY REBUILD] {symbol}: Adjusted invalid stop price from ${initial_stop:.{decimals}f} "
                f"to ${new_stop_price:.{decimals}f} (new_target=${new_target_price or 0.0:.{decimals}f}) relative to actual fill ${fill_price:.4f}"
            )
            
            # Post-Rebuild Re-Validation Assertion
            rebuilt_stop_dist = fill_price - new_stop_price
            rebuilt_valid = (new_stop_price < fill_price) and (rebuilt_stop_dist >= min_required_distance)
            if target_price and target_price > 0:
                rebuilt_valid = rebuilt_valid and (fill_price < (new_target_price or target_price))

            if not rebuilt_valid:
                logger.error(
                    f"[POST-FILL GEOMETRY REBUILD FAILED] {symbol}: Rebuilt stop price ${new_stop_price:.{decimals}f} "
                    f"is still invalid relative to fill ${fill_price:.4f} (min_dist=${min_required_distance:.4f})"
                )
            else:
                logger.info(f"[POST-FILL GEOMETRY REBUILD SUCCESS] {symbol}: Re-validation valid=True for new stop ${new_stop_price:.{decimals}f}")

            if bridge:
                replace_fn = getattr(bridge, "replace_order_by_id", None)
                if not replace_fn and hasattr(bridge, "_client") and hasattr(bridge._client, "replace_order_by_id"):
                    replace_fn = bridge._client.replace_order_by_id

                if callable(replace_fn):
                    # 1. Replace Stop Loss Order
                    if stop_order_id:
                        try:
                            from alpaca.trading.requests import ReplaceOrderRequest
                            req = ReplaceOrderRequest(stop_price=new_stop_price)
                            res = replace_fn(stop_order_id, req)
                            new_id = getattr(res, "id", stop_order_id)
                            logger.info(f"[POST-FILL GEOMETRY REBUILD] Replaced broker stop order {stop_order_id} -> {new_id} at new stop price ${new_stop_price:.{decimals}f}")
                        except Exception as replace_err:
                            logger.warning(f"[POST-FILL GEOMETRY REBUILD] Could not replace broker stop order {stop_order_id}: {replace_err}")

                    # 2. Replace Take Profit Order if tp_order_id is provided and new_target_price is set
                    if tp_order_id and new_target_price:
                        try:
                            from alpaca.trading.requests import ReplaceOrderRequest
                            req = ReplaceOrderRequest(limit_price=new_target_price)
                            replace_fn(tp_order_id, req)
                            logger.info(f"[POST-FILL GEOMETRY REBUILD] Replaced broker TP order {tp_order_id} at new limit price ${new_target_price:.{decimals}f}")
                        except Exception as tp_err:
                            logger.warning(f"[POST-FILL GEOMETRY REBUILD] Could not replace broker TP order {tp_order_id}: {tp_err}")

            return new_stop_price

        return initial_stop

    def ensure_initialized(self, symbol: str, side: str, avg_cost: float, open_orders: list, current_price: float, original_stop: Optional[float] = None, qty: float = 0.0) -> Optional[TrailingStopState]:
        """Ensure trailing stop state is initialized for an active position."""
        with self._lock:
            state = self.states.get(symbol)
            if state and state.active:
                if state.order_id is not None:
                    all_open_orders = self._extract_all_orders_including_legs(open_orders)
                    open_order_ids = {str(getattr(o, "id", "")) for o in all_open_orders}
                    if str(state.order_id) not in open_order_ids:
                        logger.warning(
                            f"Stop order {state.order_id} for {symbol} is no longer open in active legs. "
                            f"Reconciling broker open orders before marking naked..."
                        )
                        state.order_id = None

                if state.order_id is None:
                    self._reconcile_broker_order(state, open_orders)
                return state

            stop_order_id = None
            broker_stop_price = None
            target_side = "sell" if side == "BUY" else "buy"
            
            for o in open_orders:
                o_sym = getattr(o, "symbol", None)
                if not o_sym:
                    contract = getattr(o, "contract", None)
                    if contract:
                        o_sym = getattr(contract, "symbol", "")
                o_sym = str(o_sym).upper().strip() if o_sym else ""
                o_side = getattr(o, "side", "")
                o_side_str = getattr(o_side, "value", o_side)
                o_side_str = str(o_side_str).lower() if o_side_str else ""
                
                if o_sym == symbol.upper().strip() and o_side_str == target_side:
                    o_stop = getattr(o, "stop_price", None)
                    if o_stop is not None:
                        broker_stop_price = float(o_stop)
                        stop_order_id = str(o.id)
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
                df = fetch_intraday(symbol)
                if not df.empty and "atr" in df.columns:
                    atr = float(df["atr"].iloc[-1])
            except Exception:
                pass

            mult = getattr(config, "TRAILING_STOP_ATR_MULTIPLE", 1.5)

            state = TrailingStopState(
                symbol=symbol,
                side=side,
                entry_price=avg_cost,
                peak_price=max(avg_cost, current_price) if side == "BUY" else min(avg_cost, current_price),
                atr=atr,
                stop_price=final_stop,
                trail_multiple=mult,
                last_updated=time.time(),
                order_id=stop_order_id,
                qty=qty,
                strategy="RECONCILED",
                active=True
            )
            self.states[symbol] = state
            self._save_state()
            logger.info(f"Auto-initialized Trailing Stop for {symbol} from broker: Entry ${avg_cost:.2f}, Stop ${final_stop:.2f}, Order ID {stop_order_id}")
            return state

    def _is_valid_stop_order(self, o: object, target_side: str) -> bool:
        """Verify that an order object or dict is a valid stop loss order on the target side."""
        if not o:
            return False
        if isinstance(o, dict):
            o_side = o.get("side", "")
    def _is_valid_stop_order(self, o: object, target_side: str) -> bool:
        """Verify order is a valid active stop loss order for target side."""
        if isinstance(o, dict):
            o_id = o.get("id", "")
            o_side = o.get("side", "")
            o_type = str(o.get("order_type") or o.get("type") or "").lower()
            o_stop = o.get("stop_price")
            o_status = o.get("status")
        else:
            o_id = getattr(o, "id", "")
            o_side = getattr(o, "side", "")
            o_type = str(getattr(o, "order_type", getattr(o, "type", ""))).lower()
            o_stop = getattr(o, "stop_price", None)
            o_status = getattr(o, "status", None)

        if o_status is not None:
            o_status_str = str(getattr(o_status, "value", o_status)).lower()
            if o_status_str in {"canceled", "cancelled", "filled", "expired", "rejected"}:
                logger.info(f"[RECONCILE EVAL] Order ID {o_id}: status '{o_status_str}' is CLOSED/DEAD -> Rejecting candidate.")
                return False

        o_side_str = getattr(o_side, "value", o_side)
        o_side_str = str(o_side_str).lower() if o_side_str else ""
        
        is_side_match = (o_side_str == target_side.lower())
        is_stop_type = (o_stop is not None or "stop" in o_type)

        logger.info(
            f"[RECONCILE EVAL] Order ID {o_id}: side='{o_side_str}' (target='{target_side.lower()}'), "
            f"type='{o_type}', stop_price={o_stop} -> side_match={is_side_match}, stop_type={is_stop_type}"
        )

        return is_side_match and is_stop_type

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
        target_side = "sell" if state.side == "BUY" else "buy"
        all_orders = self._extract_all_orders_including_legs(open_orders)
        for o in all_orders:
            o_sym = getattr(o, "symbol", None)
            if not o_sym:
                contract = getattr(o, "contract", None)
                if contract:
                    o_sym = getattr(contract, "symbol", "")
            o_sym = str(o_sym).upper().strip() if o_sym else ""
            
            if o_sym == state.symbol.upper().strip() and self._is_valid_stop_order(o, target_side):
                o_stop = getattr(o, "stop_price", None)
                o_type = str(getattr(o, "order_type", getattr(o, "type", ""))).lower()
                broker_stop_price = float(o_stop) if o_stop is not None else state.stop_price
                state.order_id = str(getattr(o, "id", ""))
                state.remediation_attempts = 0
                state.escalation_alerted = False
                state.remediation_in_progress = False
                if o_stop is not None:
                    if state.side == "BUY":
                        state.stop_price = max(state.stop_price, broker_stop_price)
                    else:
                        state.stop_price = min(state.stop_price, broker_stop_price)
                self._save_state()
                logger.info(f"Reconciled active STOP order for {state.symbol}: Adopted Order ID {state.order_id} (type={o_type}), stop_price ${state.stop_price:.2f}")
                return True
        return False

    def update_stop(self, symbol: str, current_price: float, bridge=None, dry_run=False) -> Optional[TrailingStopState]:
        """Update the peak/trough and ratched the stop price if required."""
        with self._lock:
            state = self.states.get(symbol)
            if not state or not state.active:
                return None
                
            if not getattr(config, "TRAILING_STOP_ENABLE_SHORT", True) and state.side == "SELL":
                return state

            now = time.time()
            cooldown = getattr(config, "TRAILING_STOP_UPDATE_COOLDOWN_SECONDS", 30)
            if (now - state.last_updated) < cooldown:
                return state

            min_delta = getattr(config, "TRAILING_STOP_MIN_UPDATE_DELTA", 0.25)
            
            if state.atr > 0 and getattr(config, "TRAILING_STOP_USE_ATR", True):
                trail_distance = state.atr * state.trail_multiple
            else:
                trail_distance = state.entry_price * getattr(config, "TRAILING_STOP_FALLBACK_PCT", 0.02)

            min_profit_pct = getattr(config, "TRAILING_STOP_MIN_PROFIT_PCT", 0.01)
            min_trail_pct = getattr(config, "TRAILING_STOP_MIN_TRAIL_PCT", 0.015)

            new_stop = state.stop_price
            should_update = False

            if state.side == "BUY":
                if current_price > state.peak_price:
                    state.peak_price = current_price
                    
                profit_pct = (state.peak_price - state.entry_price) / state.entry_price if state.entry_price > 0 else 0.0
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
                if profit_pct >= min_profit_pct:
                    effective_trail = max(trail_distance, state.entry_price * min_trail_pct)
                    computed_stop = state.peak_price + effective_trail
                    if computed_stop < state.stop_price - min_delta:
                        new_stop = computed_stop
                        should_update = True

            if should_update:
                logger.info(f"Trailing stop for {symbol} moving from ${state.stop_price:.2f} -> ${new_stop:.2f}")
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
                    state.last_updated = now
                else:
                    state.last_updated = 0
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
            if hasattr(bridge, "ib") and hasattr(bridge.ib, "positions"):
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
                logger.info(f"Fresh position query for {symbol} returned 0. Marking trailing stop inactive (stopped out).")
                # Log stopped out exit to trade journal
                exit_price = state.stop_price
                qty_closed = getattr(state, "qty", 0.0) or abs(position_qty)
                pnl_val = (exit_price - state.entry_price) * qty_closed if state.side == "BUY" else (state.entry_price - exit_price) * qty_closed
                
                try:
                    from strategies.trade_journal import log_trade
                    log_trade(
                        symbol=symbol,
                        side="SELL" if state.side == "BUY" else "BUY",
                        strategy=getattr(state, "strategy", "UNKNOWN"),
                        qty=int(qty_closed),
                        entry_price=state.entry_price,
                        stop_price=state.stop_price,
                        target_price=exit_price,
                        exit_price=exit_price,
                        exit_reason="STOP_OUT",
                        pnl=pnl_val,
                        notes="Stopped out exit",
                        event_type="TRADE_CLOSED",
                        signal_id=getattr(state, "signal_id", None)
                    )
                except Exception as e_log:
                    logger.error("Failed to log stopped out exit to journal: %s", e_log)
                
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
            if attempts >= 3:
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
                current_price = None
                if hasattr(bridge, "market_price"):
                    current_price = bridge.market_price(symbol)
                
                if current_price:
                    is_invalid = False
                    if state.side == "BUY" and state.stop_price >= current_price:
                        is_invalid = True
                    elif state.side == "SELL" and state.stop_price <= current_price:
                        is_invalid = True
                    
                    if is_invalid:
                        logger.warning(f"Tracked stop price ${state.stop_price:.2f} is stale/invalid for {symbol} (current price ${current_price:.2f}). Falling back to flatten.")
                        protection_method = "flatten"

                if protection_method == "replace":
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
                                        if self._is_valid_stop_order(candidate, target_side):
                                            cand_id = str(getattr(candidate, "id", rel_id))
                                            o_type = str(getattr(candidate, "order_type", getattr(candidate, "type", ""))).lower()
                                            with self._lock:
                                                state.order_id = cand_id
                                                state.remediation_in_progress = False
                                                state.remediation_attempts = 0
                                                self._save_state()
                                            logger.info(f"Adopted confirmed Stop Loss leg {cand_id} (type={o_type}) for {symbol} from broker bracket response.")
                                            return
                                        else:
                                            logger.warning(f"Candidate order {getattr(candidate, 'id', rel_id)} for {symbol} is NOT a stop order (type={getattr(candidate, 'order_type', getattr(candidate, 'type', None))}). Checking next candidate...")
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
                    
                    logger.warning(f"Position for {symbol} is held by broker order, but no valid STOP leg was confirmed. Proceeding to emergency flatten.")
                    protection_method = "flatten"
                else:
                    protection_method = "flatten"

        if protection_method == "flatten":
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
                                    from strategies.trade_journal import log_trade
                                    log_trade(
                                        symbol=symbol,
                                        side="SELL" if current_state.side == "BUY" else "BUY",
                                        strategy=getattr(current_state, "strategy", "UNKNOWN"),
                                        qty=int(qty_closed),
                                        entry_price=current_state.entry_price,
                                        stop_price=current_state.stop_price,
                                        target_price=exit_price,
                                        exit_price=exit_price,
                                        exit_reason="EMERGENCY_FLATTEN",
                                        pnl=pnl_val,
                                        notes="Emergency flatten exit",
                                        event_type="TRADE_CLOSED",
                                        signal_id=getattr(current_state, "signal_id", None)
                                    )
                                except Exception as e_log:
                                    logger.error("Failed to log emergency close to journal: %s", e_log)
                        self._save_state()
            except Exception as e:
                logger.error(f"EMERGENCY FLATTEN FAILED for {symbol}: {e}")
                with self._lock:
                    current_state = self.states.get(symbol)
                    if current_state and current_state.active:
                        current_state.remediation_in_progress = False
                        self._save_state()

    def reset(self, symbol: str) -> None:
        with self._lock:
            if symbol in self.states:
                self.states[symbol].active = False
                self._save_state()

manager = DynamicTrailingStopManager()
