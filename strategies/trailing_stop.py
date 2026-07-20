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

    def initialize_position(self, signal: TradeSignal, fill_price: float, stop_order_id: Optional[str] = None) -> TrailingStopState:
        """Initialize trailing stop when order is filled."""
        with self._lock:
            atr = getattr(signal, 'atr', 0.0)
            
            mult = getattr(config, "TRAILING_STOP_ATR_MULTIPLE", 1.5)
            fallback_pct = getattr(config, "TRAILING_STOP_FALLBACK_PCT", 0.02)
            
            if atr > 0 and getattr(config, "TRAILING_STOP_USE_ATR", True):
                trail_distance = atr * mult
            else:
                trail_distance = fill_price * fallback_pct
                
            if signal.side.upper() == "BUY":
                initial_stop = fill_price - trail_distance
            else:
                initial_stop = fill_price + trail_distance
                
            # Ensure we don't start looser than the signal's original stop
            if signal.side.upper() == "BUY":
                initial_stop = max(initial_stop, signal.stop_price)
            else:
                initial_stop = min(initial_stop, signal.stop_price)

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
                active=True
            )
            self.states[signal.symbol] = state
            self._save_state()
            logger.info(f"Initialized Trailing Stop for {signal.symbol}: Entry ${fill_price:.2f}, Stop ${initial_stop:.2f}")
            return state

    def ensure_initialized(self, symbol: str, side: str, avg_cost: float, open_orders: list, current_price: float, original_stop: Optional[float] = None) -> Optional[TrailingStopState]:
        """Ensure trailing stop state is initialized for an active position."""
        with self._lock:
            state = self.states.get(symbol)
            if state and state.active:
                if state.order_id is not None:
                    open_order_ids = {str(getattr(o, "id", "")) for o in open_orders}
                    if str(state.order_id) not in open_order_ids:
                        logger.warning(
                            f"Stop order {state.order_id} for {symbol} is no longer open. "
                            f"Marking position as naked."
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

            final_stop = calculated_stop
            if broker_stop_price is not None:
                if side == "BUY":
                    final_stop = max(calculated_stop, broker_stop_price)
                else:
                    final_stop = min(calculated_stop, broker_stop_price)

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
                active=True
            )
            self.states[symbol] = state
            self._save_state()
            logger.info(f"Auto-initialized Trailing Stop for {symbol} from broker: Entry ${avg_cost:.2f}, Stop ${final_stop:.2f}, Order ID {stop_order_id}")
            return state

    def _reconcile_broker_order(self, state: TrailingStopState, open_orders: list):
        """Adopt stop_order_id from active open orders and perform direction-aware price merge."""
        target_side = "sell" if state.side == "BUY" else "buy"
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
            
            if o_sym == state.symbol.upper().strip() and o_side_str == target_side:
                o_stop = getattr(o, "stop_price", None)
                if o_stop is not None:
                    broker_stop_price = float(o_stop)
                    state.order_id = str(o.id)
                    state.remediation_attempts = 0
                    state.escalation_alerted = False
                    if state.side == "BUY":
                        state.stop_price = max(state.stop_price, broker_stop_price)
                    else:
                        state.stop_price = min(state.stop_price, broker_stop_price)
                    self._save_state()
                    logger.info(f"Reconciled active stop order for {state.symbol}: Adopted Order ID {state.order_id}, stop_price ${state.stop_price:.2f}")
                    break

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

            new_stop = state.stop_price
            should_update = False

            if state.side == "BUY":
                if current_price > state.peak_price:
                    state.peak_price = current_price
                    
                computed_stop = state.peak_price - trail_distance
                if computed_stop > state.stop_price + min_delta:
                    new_stop = computed_stop
                    should_update = True
                    
            else: # SELL (Short)
                if current_price < state.peak_price:
                    state.peak_price = current_price
                    
                computed_stop = state.peak_price + trail_distance
                if computed_stop < state.stop_price - min_delta:
                    new_stop = computed_stop
                    should_update = True

            if should_update:
                logger.info(f"Trailing stop for {symbol} moving from ${state.stop_price:.2f} -> ${new_stop:.2f}")
                update_succeeded = True
                if not dry_run and bridge and state.order_id:
                    try:
                        update_succeeded = bridge.modify_stop_order(state.order_id, new_stop)
                        if not update_succeeded:
                            logger.warning(f"Stop modify failed, resetting cooldown for {symbol}")
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
                logger.error(f"Failed to place standalone stop order for {symbol}: {e}")
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
                    close_ok = bridge.close_position(symbol)
                else:
                    logger.critical(f"Bridge {bridge.__class__.__name__} does not support close_position. Cannot emergency close!")
                
                with self._lock:
                    current_state = self.states.get(symbol)
                    if current_state and current_state.active:
                        current_state.remediation_in_progress = False
                        if close_ok:
                            current_state.active = False
                            current_state.remediation_attempts = 0
                            current_state.escalation_alerted = False
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
