"""
entry_gate_service.py — NEW entry safety gates for IBKRBridge.
Extracted from ibkr_bridge.py to reduce coupling.
"""
import logging
import math

import config as config_module
import data_integrity
import order_audit
import risk_engine
import risk_state

logger = logging.getLogger(__name__)


class EntryGateService:
    """Read-only gates that check whether a NEW entry is allowed.
    Moves together so ibkr_bridge's execute_signal shrinks."""

    def __init__(self, cfg=None, net_liq_fn=None):
        self._cfg = cfg or config_module.get_settings()
        # net_liq_fn: callable returning current NetLiquidation
        self._net_liq_fn = net_liq_fn or (lambda: 0.0)
        self._halt = False

    def _c(self, name, default=None):
        return getattr(self._cfg, name, default)

    @property
    def halted(self) -> bool:
        return self._halt

    def halt(self) -> None:
        self._halt = True

    # ── Market-hours ────────────────────────────────
    def market_hours_ok(self) -> bool:
        if not bool(self._c("MARKET_HOURS_GATE_ENABLED", True)):
            return True
        try:
            from zoneinfo import ZoneInfo
            now_et = __import__("datetime").datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            logger.error("Could not resolve US/Eastern time for market-hours gate -> blocking")
            return False
        return data_integrity.is_regular_hours(now_et)

    # ── Combined gate (called for every NEW entry) ──
    def entry_blocked(self, symbol: str, intended_value: float = 0.0,
                      signal=None, order_price: float = None):
        """Returns (blocked: bool, reason: str). Fails closed on any check."""
        if self._halt:
            return True, "halt_flag"

        equity = self._net_liq_fn()
        if not self.market_hours_ok():
            return True, "market_closed"
        if signal is not None and order_price is not None:
            max_dev = float(self._c("DECISION_PRICE_MAX_DEVIATION_PCT", 0.0))
            if not data_integrity.decision_price_ok(
                    getattr(signal, "price", None), order_price, max_dev):
                return True, "price_mismatch"
        if risk_state.daily_loss_blocked(equity):
            return True, "daily_loss"
        start = risk_state.start_of_day_equity()
        if risk_engine.drawdown_halt_breached(start, equity):
            return True, "drawdown_halt"
        if risk_engine.symbol_exposure_exceeded(intended_value, 0.0, equity):
            return True, "symbol_exposure"
        if self._minervini_stage2_blocks(symbol, signal):
            return True, "stage2_filter"
        return False, ""

    # ── Minervini M2 — fail-open ────────────────────
    def _minervini_stage2_blocks(self, symbol: str, signal=None) -> bool:
        if not bool(self._c("MINERVINI_OVERLAY_ENABLED", False)):
            return False
        if not bool(self._c("MINERVINI_STAGE2_BLOCK_ENABLED", False)):
            return False
        action = str(getattr(signal, "action", "")).upper().strip()
        if action != "BUY":
            return False
        try:
            import minervini
            from data_manager import fetch_ohlcv
            df = fetch_ohlcv(symbol)
            verdict = minervini.evaluate_entry(df)
            stage2_ok = getattr(verdict, "stage2_ok", True)
            if stage2_ok is None:
                return False
            return not bool(stage2_ok)
        except Exception:
            logger.debug("Minervini Stage-2 filter unavailable for %s -> fail open", symbol, exc_info=True)
            return False

    # ── Minervini M3 1R risk sizing ────────────
    def risk_sized_qty(self, symbol: str, signal=None,
                       entry_price: float = 0.0,
                       notional_qty: int = 0) -> int:
        if not isinstance(notional_qty, int):
            try:
                notional_qty = int(notional_qty)
            except (TypeError, ValueError):
                return notional_qty
        if notional_qty <= 0:
            return notional_qty
        if not bool(self._c("MINERVINI_OVERLAY_ENABLED", False)):
            return notional_qty
        if not bool(self._c("MINERVINI_SIZING_ENABLED", False)):
            return notional_qty
        action = str(getattr(signal, "action", "")).upper().strip()
        if action != "BUY":
            return notional_qty
        try:
            entry = float(entry_price)
            if not math.isfinite(entry) or entry <= 0:
                return notional_qty
            import minervini
            from data_manager import fetch_ohlcv
            df = fetch_ohlcv(symbol)
            verdict = minervini.evaluate_entry(df)
            stop = minervini.minervini_stop_price(getattr(verdict, "pivot_low", None))
            if stop is None:
                return notional_qty
            stop = float(stop)
            if not math.isfinite(stop) or stop <= 0:
                return notional_qty
            if stop >= entry:
                return notional_qty
            risk_per_share = entry - stop
            if not math.isfinite(risk_per_share) or risk_per_share <= 0:
                return notional_qty
            max_dist = float(self._c("MINERVINI_MAX_STOP_DISTANCE_PCT", 0.10))
            if max_dist > 0 and risk_per_share > entry * max_dist:
                return notional_qty
            risk_budget = float(self._c("MINERVINI_RISK_PER_TRADE_USD", 0.0))
            if not math.isfinite(risk_budget) or risk_budget <= 0:
                return notional_qty
            risk_qty = int(risk_budget / risk_per_share)
            return max(0, min(notional_qty, risk_qty))
        except Exception:
            logger.debug("Minervini 1R sizing unavailable for %s -> fail open", symbol, exc_info=True)
            return notional_qty
