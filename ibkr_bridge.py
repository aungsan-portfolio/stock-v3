"""
ibkr_bridge.py — IBKR Paper Trading bridge via ib_insync.
TWS/Gateway must be running with API enabled on port 7497 (paper).

Production hardening:
- Paper-only lock by default; refuses non-paper ports unless config opts out.
- Long-only by default. SELL closes existing longs; never opens accidental shorts.
- Duplicate-entry guard: pending/working orders count as occupied symbols.
- Opening trades use either a fixed TP/SL bracket or a parent limit + trailing stop.
- Snapshot price fetch waits for the snapshot to populate (event-driven, not
  fixed sleep). Historical close fallback is blocked for order placement by default.
- qualifyContracts() result captured (the contract is mutated in place but we
  also return the qualified contract from a single source of truth).
"""
import datetime as _dt
import logging
import math
from typing import List, Optional

from ib_insync import IB, Stock, LimitOrder, MarketOrder, Contract, Order

import account_guard
import alerts
import config
import data_integrity
import live_invariants
import order_audit
import order_exec
import reconciliation
import reconnect_watchdog
import risk_engine
import risk_state
import shutdown_guard
from predictor import Signal

logger = logging.getLogger(__name__)

# "Working" = live at the broker but not yet a position; used by the duplicate-
# entry / open-order guards. Fill vs. acceptance classification now lives in
# order_exec (single source of truth) so "accepted" can never be mistaken for a
# fill (H4).
WORKING_ORDER_STATUSES = {"PendingSubmit", "ApiPending", "PreSubmitted", "Submitted"}

# ── Live-readiness capability flags ───────────────────────────────────────────
# These advertise which live-trading safety capabilities are ACTUALLY IMPLEMENTED
# in this bridge. They start False and each is flipped to True ONLY by the phase
# that builds the backing logic (see reports/LIVE_TRADING_IMPLEMENTATION_PLAN_MM.md).
# The `live-readiness` command reads them to produce an honest go-live scorecard.
# Flipping one True without its implementation is a deliberate footgun — do not.
SUPPORTS_FILL_VERIFICATION       = True    # Phase 2 (H4, H5): wait for real fill, not "accepted"
SUPPORTS_PARTIAL_FILL_HANDLING   = True    # Phase 2 (H6): size children from actual filled qty
SUPPORTS_PROTECTIVE_CHILD_VERIFY = True    # Phase 2 (H7): confirm stop child is live or flatten
SUPPORTS_SERVER_SIDE_GTC_STOP    = True    # Phase 3 (C2, H19): resting GTC/OCA hard stop per entry
SUPPORTS_DAILY_LOSS_KILLSWITCH   = True    # Phase 3 (H1): loss_breached() wired into the order gate
SUPPORTS_REALTIME_DATA_GUARD     = True    # Phase 4 (H11-H13, H16): sane real-time/delayed quote, reject stale-close/crossed/wide
SUPPORTS_MARKET_HOURS_GATE       = True     # Phase 4 (H15): refuse NEW entries outside RTH/holidays
SUPPORTS_STARTUP_RECONCILIATION  = True    # Phase 5A (H18): broker = source of truth on startup
# Phase 6.1 BUILDS the account-type assertion (account_guard.assert_account, wired
# into connect() below as SAFETY GATE #2 and covered fail-closed by
# test_account_guard.py): a live "U..." account on the paper port now fails closed.
# This flag advertises ONLY that capability -- the guard is implemented and tested --
# so it is honestly True. It does NOT advertise that the bot is ready for live
# trading: the live conversion (6.2 live-port wiring + 6.3 live_safety_preflight +
# 6.4 config flip) is NOT built, and the SEPARATE scorecard gates for
# IBKR_MARKET_DATA_TYPE==1 and an explicit LIVE_ACCOUNT_ID still FAIL, so
# live-readiness stays NOT READY. PAPER ONLY.
SUPPORTS_ACCOUNT_TYPE_ASSERTION  = True    # Phase 6.1: assert paper(DU)/live(U) account, not just port


def _order_status_name(trade) -> str:
    return str(getattr(getattr(trade, "orderStatus", None), "status", "UNKNOWN"))


def _is_order_working(status: str) -> bool:
    return str(status) in WORKING_ORDER_STATUSES


class IBKRBridge:
    def __init__(self) -> None:
        self.ib = IB()
        self._contract_cache: dict = {}
        # Phase 3: sticky kill-switch for NEW entries. Set when the startup
        # protection scan cannot protect an open long; closes/flatten/repair stay
        # allowed. Daily-loss + drawdown are checked dynamically, not via this flag.
        self._halt_new_entries = False
        # Phase 5B-2: connection-health for the one-shot run. Defaults healthy
        # ("presumed usable until we learn otherwise") so a bridge that has not
        # connected behaves exactly as before; a mid-run disconnect flips it
        # closed so _new_entries_blocked refuses NEW entries (fail-closed). The
        # watchdog never places orders.
        self._conn_health = reconnect_watchdog.ConnectionHealth()
        self._intentional_disconnect = False
        self._disconnect_handler_registered = False

    # ── Connection ───────────────────────────────────────────────
    def connect(self) -> bool:
        # SAFETY GATE #1 (UNCHANGED, highest priority): paper-port lock. Enforced
        # BEFORE any reconnect/backoff so a non-paper port is refused outright and
        # the Phase-5B-2 watchdog can never retry it (or otherwise bypass it).
        if bool(getattr(config, "REQUIRE_PAPER_PORT", True)):
            paper_port = int(getattr(config, "PAPER_IBKR_PORT", 7497))
            if int(config.IBKR_PORT) != paper_port:
                logger.error(
                    "Refusing to connect: IBKR_PORT=%s is not paper port %s",
                    config.IBKR_PORT,
                    paper_port,
                )
                self._conn_health.mark_unhealthy(reconnect_watchdog.REASON_PAPER_PORT)
                return False

        # Phase 5B-2: bound the socket/request so a dead TWS cannot hang a
        # one-shot run (flatten_vti.py / check_positions.py pattern).
        timeout = reconnect_watchdog.request_timeout()
        try:
            if timeout and timeout > 0:
                self.ib.RequestTimeout = timeout
        except Exception:
            logger.debug("Could not set ib.RequestTimeout", exc_info=True)

        # Register the disconnect watchdog once (safe: it only marks health +
        # audits, never places orders). Guarded so the offline fake-IB tests are
        # unaffected.
        self._register_disconnect_watchdog()
        self._intentional_disconnect = False

        timeout_kw = {"timeout": timeout} if timeout and timeout > 0 else {}

        def _connect_once() -> bool:
            try:
                self.ib.connect(
                    host=config.IBKR_HOST,
                    port=config.IBKR_PORT,
                    clientId=getattr(config, "CLIENT_ID_BOT", config.IBKR_CLIENT_ID),
                    **timeout_kw,
                )
                # Fall back to delayed (15-min) data when the account lacks a
                # real-time market-data subscription. 3 = DELAYED, 4 = DELAYED_FROZEN.
                try:
                    self.ib.reqMarketDataType(config.IBKR_MARKET_DATA_TYPE)
                except Exception:
                    logger.warning("Could not set market data type", exc_info=True)
                logger.info("Connected to IBKR | account=%s", self.ib.managedAccounts())
                return True
            except Exception:
                logger.exception("Failed to connect to IBKR TWS")
                return False

        # Phase 5B-2: bounded exponential-backoff retry of the INITIAL connect.
        # Uses ib.sleep so the (cooperative) event loop keeps pumping between
        # tries; bounded by IBKR_RECONNECT_MAX_ATTEMPTS — never a forever loop,
        # never a daemon, never an order.
        result = reconnect_watchdog.attempt_connect(_connect_once, sleep=self.ib.sleep)

        if result.connected:
            # SAFETY GATE #2 (Phase 6.1): account-type assertion. The paper-port
            # lock above guarded the PORT; this guards the ACCOUNT actually logged
            # into that port -- a live ("U...") account on the paper port FAILS
            # CLOSED here. Read-only: it only reads managedAccounts() and places
            # NO order. Run BEFORE marking the link healthy so a wrong account can
            # never present as a usable connection.
            if not self._assert_account_type():
                self._conn_health.mark_unhealthy(account_guard.HEALTH_REASON)
                # Do not stay connected to an unverified (possibly live) account.
                self.disconnect()
                logger.error("Refusing connection: account-type assertion failed "
                             "(fail closed).")
                return False
            self._conn_health.mark_healthy(reconnect_watchdog.REASON_CONNECTED)
            if result.attempts > 1:
                logger.info("Connected to IBKR after %s attempt(s)", result.attempts)
                reconnect_watchdog.audit_connection(
                    reconnect_watchdog.REASON_CONNECTED, healthy=True,
                    attempts=result.attempts, retried=True,
                )
            return True

        # Gave up after the bounded retries (or a single try with reconnect off).
        self._conn_health.mark_unhealthy(reconnect_watchdog.REASON_GAVE_UP)
        reconnect_watchdog.audit_connection(
            reconnect_watchdog.REASON_GAVE_UP, healthy=False,
            attempts=result.attempts, enabled=result.enabled,
        )
        # Phase 5B-4: alert the operator that the (bounded) reconnect gave up.
        # emit() is disabled/log-only by default and never places an order.
        alerts.emit(alerts.EVENT_RECONNECT_FAILURE,
                    message="IBKR connect gave up after bounded retries",
                    attempts=result.attempts, enabled=result.enabled)
        logger.error("IBKR connect failed after %s attempt(s) (reconnect enabled=%s)",
                     result.attempts, result.enabled)
        return False

    def disconnect(self) -> None:
        # Flag the intentional close so the disconnect watchdog treats it as a
        # clean shutdown (it must NOT mark the connection unhealthy / audit a
        # mid-run drop for our own teardown).
        self._intentional_disconnect = True
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IBKR")

    # ── Phase 6.1: account-type assertion ────────────────────────
    def _assert_account_type(self) -> bool:
        """Assert the connected broker account matches the expected ENVIRONMENT.

        Read-only safety guard (Phase 6.1). The paper-port lock guards the PORT;
        this guards the ACCOUNT logged into that port. Reads ib.managedAccounts()
        and delegates the decision to the PURE account_guard.assert_account, which
        FAILS CLOSED on an empty / malformed / ambiguous account list or the wrong
        environment prefix:

          * paper mode (default, and the ONLY mode used this phase): require a
            single "DU..." paper account, or an explicitly-configured
            EXPECTED_PAPER_ACCOUNT_ID when TWS exposes several;
          * live mode (INERT -- account_guard.live_mode_enabled() is False for the
            shipped paper config): require a "U..." account matching the explicit
            LIVE_ACCOUNT_ID.

        Returns True only on a verified account. Places NO order and NEVER enables
        live trading. Both the pass and the fail are audited (order_audit stage
        ``account_assertion``) and logged; a fail is logged at ERROR level (req 11).
        Never raises (a managedAccounts() error is treated as empty -> fail closed).
        """
        # Documented escape hatch; ships True so the default behaviour is the
        # fail-closed assertion. Leaving it on is the paper-safe default.
        if not bool(getattr(config, "ASSERT_ACCOUNT_TYPE", True)):
            logger.warning("ASSERT_ACCOUNT_TYPE is disabled -- skipping account-type "
                           "assertion (NOT recommended).")
            return True

        live_mode = account_guard.live_mode_enabled(config)
        if live_mode:
            expected = getattr(config, "LIVE_ACCOUNT_ID", None)
        else:
            expected = getattr(config, "EXPECTED_PAPER_ACCOUNT_ID", None)

        try:
            managed = self.ib.managedAccounts()
        except Exception:
            logger.error("Could not read managedAccounts() for the account-type "
                         "assertion -> failing closed", exc_info=True)
            managed = []

        result = account_guard.assert_account(
            managed, live_mode=live_mode, expected_account=expected)

        order_audit.log_event(account_guard.AUDIT_STAGE,
                              **account_guard.audit_fields(result))
        if result.ok:
            logger.info("Account-type assertion passed | %s",
                        account_guard.summary_line(result))
            return True

        logger.error("Account-type assertion FAILED -> refusing connection (fail "
                     "closed) | %s", account_guard.summary_line(result))
        return False

    # ── Phase 5B-3: graceful shutdown ────────────────────────────
    def prepare_graceful_shutdown(self) -> dict:
        """Build the READ-ONLY graceful-shutdown plan (Phase 5B-3).

        Reads broker truth (positions + working orders), delegates to the PURE
        shutdown_guard.build_shutdown_plan, and audits the plan. Places NOTHING
        and cancels NOTHING: the plan's only verbs are PRESERVE every resting
        protective stop and DETECT any unprotected long. Safe to call any time;
        never raises.
        """
        try:
            positions = self._open_positions_plain()
            working = self._working_orders_plain()
            plan = shutdown_guard.build_shutdown_plan(positions, working)
        except Exception:
            logger.warning("Could not build graceful-shutdown plan", exc_info=True)
            plan = shutdown_guard.build_shutdown_plan([], [])
        order_audit.log_event(
            order_audit.STAGE_SHUTDOWN, phase="plan",
            **shutdown_guard.audit_fields(plan),
        )
        logger.info("Graceful shutdown | %s", shutdown_guard.summary_line(plan))
        return plan

    def graceful_shutdown(self, repair: bool = True) -> dict:
        """Tear the one-shot connection down SAFELY (Phase 5B-3).

        Steps, in order:
          1. Build the read-only plan (preserve all resting protective stops;
             detect unprotected longs). Opens NO new entry, cancels NO order.
          2. If a long is unprotected AND ``repair`` is on AND the link is still
             healthy/connected, repair ONLY through the existing, already-tested
             ensure_protective_stops() -- it places a GTC stop priced off the
             broker avgCost or HALTS new entries; it never opens a position and
             never guesses avgCost (a missing basis -> halt, not a guess). A
             resting GTC protective stop is NEVER cancelled.
          3. Mark the disconnect intentional (so the Phase-5B-2 watchdog reads it
             as a clean close, not an unexpected mid-run drop) and disconnect.
          4. Audit the final shutdown state.

        Never raises (it is meant for a ``finally:`` teardown). Returns a report
        dict.
        """
        report = {
            "plan": None, "repair_attempted": False,
            "repaired": [], "failed": [],
            "halt_new_entries": bool(self._halt_new_entries),
            "disconnected": False,
        }
        try:
            plan = self.prepare_graceful_shutdown()
            report["plan"] = plan

            # Repair unprotected longs ONLY via the existing Phase-3 path, and
            # only while the link is usable (fail-closed: never place orders on a
            # dropped/unhealthy connection). ensure_protective_stops never opens a
            # position and never guesses avgCost.
            if (repair and shutdown_guard.needs_protection_repair(plan)
                    and self._connection_healthy() and self.ib.isConnected()):
                report["repair_attempted"] = True
                protect = self.ensure_protective_stops()
                report["repaired"] = protect.get("repaired", [])
                report["failed"] = protect.get("failed", [])
                # Re-plan so the audited final state reflects the repair.
                plan = self.prepare_graceful_shutdown()
                report["plan"] = plan
        except Exception:
            logger.warning("Graceful-shutdown pre-disconnect step failed", exc_info=True)

        # Always tear the connection down cleanly, even if the steps above raised.
        try:
            self.disconnect()
            report["disconnected"] = True
        except Exception:
            logger.warning("Graceful-shutdown disconnect failed", exc_info=True)

        report["halt_new_entries"] = bool(self._halt_new_entries)
        final_plan = report["plan"] or shutdown_guard.build_shutdown_plan([], [])
        # Phase 5B-4: if a long is STILL unprotected at shutdown (repair off or
        # could not repair), alert the operator. This runs AFTER the protective
        # steps + disconnect and emit() never cancels/places an order, so it cannot
        # strip a resting stop or block the teardown.
        unprotected = list(final_plan.get("unprotected_longs", []) or [])
        if unprotected:
            alerts.emit(alerts.EVENT_SHUTDOWN_UNPROTECTED_LONG,
                        message="unprotected long(s) remain at graceful shutdown",
                        symbols=unprotected,
                        repair_attempted=report["repair_attempted"],
                        failed=report["failed"])
        order_audit.log_event(
            order_audit.STAGE_SHUTDOWN, phase="complete",
            repair_attempted=report["repair_attempted"],
            repaired=report["repaired"], failed=report["failed"],
            halt_new_entries=report["halt_new_entries"],
            disconnected=report["disconnected"],
            **shutdown_guard.audit_fields(final_plan),
        )
        return report

    def _register_disconnect_watchdog(self) -> None:
        """Attach the Phase-5B-2 disconnect handler to ib.disconnectedEvent once.

        The handler ONLY marks the connection unhealthy + audits; it NEVER places,
        cancels, or modifies an order. Registration is fully guarded so the
        offline fake-IB tests (whose stub IB has no disconnectedEvent) are
        unaffected.
        """
        if self._disconnect_handler_registered:
            return
        event = getattr(self.ib, "disconnectedEvent", None)
        if event is None:
            return
        try:
            handler = reconnect_watchdog.make_disconnect_handler(
                self._conn_health,
                is_intentional=lambda: self._intentional_disconnect,
            )
            event += handler
            self._disconnect_handler_registered = True
        except Exception:
            logger.debug("Could not register disconnect watchdog", exc_info=True)

    def _connection_healthy(self) -> bool:
        """True when the IBKR link is healthy for NEW entries. False after a
        mid-run disconnect (fail-closed). Never raises."""
        health = getattr(self, "_conn_health", None)
        return bool(health is None or health.is_healthy)

    # ── Account state ────────────────────────────────────────────
    def get_cash(self) -> float:
        for av in self.ib.accountValues():
            if av.tag == "AvailableFunds" and av.currency == "USD":
                try:
                    return float(av.value)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def get_position(self, symbol: str) -> float:
        symbol = symbol.upper().strip()
        for pos in self.ib.positions():
            if pos.contract.symbol.upper() == symbol:
                try:
                    return float(pos.position)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def get_net_liquidation(self) -> float:
        """Current account equity (NetLiquidation, USD) -- the source of truth for
        the daily-loss kill-switch and drawdown halt (Phase 3 H1/H3)."""
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                try:
                    return float(av.value)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def snapshot_start_of_day_equity(self) -> float:
        """Snapshot today's start-of-day equity ONCE (restart-safe; H1). Call at
        connect time. Returns the persisted baseline (0.0 if equity unavailable)."""
        equity = self.get_net_liquidation()
        stored = risk_state.snapshot_start_of_day_equity(equity)
        order_audit.log_event(
            order_audit.STAGE_SNAPSHOT, phase="start_of_day_equity",
            equity=equity, baseline=stored,
        )
        return stored

    @staticmethod
    def _position_basis(position) -> float:
        """Per-share cost basis for an existing position (broker avgCost). Used to
        price startup-repair protection from the REAL entry, not a guess. Returns
        0.0 when avgCost is missing/invalid (caller must halt -- never guess)."""
        try:
            cost = float(getattr(position, "avgCost", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return cost if cost > 0 else 0.0

    def _market_hours_ok(self) -> bool:
        """True when a NEW entry is allowed by the US regular-hours gate (H15).

        Reads the wall clock in US/Eastern and defers the calendar decision to
        the pure data_integrity helper. Disabled (MARKET_HOURS_GATE_ENABLED
        False) -> always True. Fails CLOSED (blocks) if the timezone cannot be
        resolved, so a clock failure never lets an entry slip through at an
        unknown time. Only ever consulted for NEW opens.
        """
        if not bool(getattr(config, "MARKET_HOURS_GATE_ENABLED", True)):
            return True
        try:
            from zoneinfo import ZoneInfo
            now_et = _dt.datetime.now(ZoneInfo("America/New_York"))
        except Exception:
            logger.error("Could not resolve US/Eastern time for market-hours gate "
                         "-> blocking new entries", exc_info=True)
            return False
        return data_integrity.is_regular_hours(now_et)

    def _new_entries_blocked(self, symbol: str, intended_value: float = 0.0,
                             signal=None, order_price: Optional[float] = None):
        """Entry gate for NEW opens ONLY (never closes/flatten/repair).

        Returns (blocked: bool, reason: str). Checks, in order: the sticky halt
        flag (startup protection failure), the Phase-5B-2 connection-health gate
        (mid-run disconnect -> fail closed), the Phase-4 US market-hours gate
        (H15), the Phase-4 decision-vs-order price agreement (H14), the
        daily-loss kill-switch (H1), the account drawdown halt (H3), and the
        per-symbol exposure cap (H3). Each is conservative: a missing baseline /
        missing decision price never spuriously blocks.
        """
        if self._halt_new_entries:
            return True, "halt_flag"
        # Phase 5B-2: an unexpected mid-run disconnect marks the link unhealthy;
        # refuse NEW entries until reconnected (closes/flatten/repair are never
        # routed through this gate). Default health is healthy, so this never
        # blocks a normally-connected run.
        if not self._connection_healthy():
            return True, "connection_unhealthy"
        if not self._market_hours_ok():
            return True, "market_closed"
        if signal is not None and order_price is not None:
            if not data_integrity.decision_price_ok(
                    getattr(signal, "price", None), order_price,
                    getattr(config, "DECISION_PRICE_MAX_DEVIATION_PCT", 0.0)):
                return True, "price_mismatch"
        equity = self.get_net_liquidation()
        if risk_state.daily_loss_blocked(equity):
            return True, "daily_loss"
        start = risk_state.start_of_day_equity()
        if risk_engine.drawdown_halt_breached(start, equity):
            return True, "drawdown_halt"
        if risk_engine.symbol_exposure_exceeded(intended_value, 0.0, equity):
            return True, "symbol_exposure"
        return False, ""

    def working_order_symbols(self) -> set:
        """Symbols with working orders, across all visible client ids."""
        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(1)
        except Exception:
            logger.warning("Could not refresh open orders", exc_info=True)

        symbols = set()
        for trade in self.ib.openTrades():
            status = _order_status_name(trade)
            if _is_order_working(status):
                symbol = str(getattr(trade.contract, "symbol", "")).upper().strip()
                if symbol:
                    symbols.add(symbol)
        return symbols

    def has_working_order(self, symbol: str, action: Optional[str] = None) -> bool:
        """Return True when IBKR already has a working order for this symbol.

        This prevents duplicate entries when a previous limit/bracket/trailing
        order is still PendingSubmit/PreSubmitted/Submitted but has not yet
        become a position. reqAllOpenOrders is used so orders from other
        clientIds are visible too.
        """
        symbol = symbol.upper().strip()
        action = action.upper().strip() if action else None

        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(1)
        except Exception:
            logger.warning("Could not refresh open orders", exc_info=True)

        for trade in self.ib.openTrades():
            contract_symbol = str(getattr(trade.contract, "symbol", "")).upper().strip()
            order_action = str(getattr(trade.order, "action", "")).upper().strip()
            status = _order_status_name(trade)

            if contract_symbol != symbol:
                continue
            if not _is_order_working(status):
                continue
            if action is not None and order_action != action:
                continue
            return True

        return False

    # ── Contracts & pricing ──────────────────────────────────────
    def _contract(self, symbol: str) -> Contract:
        symbol = symbol.upper().strip()
        cached = self._contract_cache.get(symbol)
        if cached is not None:
            return cached
        contract = Stock(symbol, "SMART", "USD")
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"Could not qualify contract for {symbol}")
        self._contract_cache[symbol] = qualified[0]
        return qualified[0]

    def get_order_quote(self, symbol: str, timeout: float = 5.0) -> data_integrity.QuoteDecision:
        """Fetch a snapshot quote and validate it for ORDER use (Phase 4,
        H11-H13/H16). Returns a data_integrity.QuoteDecision: ``ok`` + ``price``
        when a sane real-time (or, in paper, delayed) last/midpoint is available,
        else ``ok=False`` with a reason (crossed/wide-spread/stale-close/
        delayed-blocked/no-data). NEVER prices an order from the plain prior
        daily ``close``. Read-only: places no order.
        """
        contract = self._contract(symbol)
        ticker = self.ib.reqMktData(contract, "", True, False)
        require_rt = bool(getattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", False))
        max_spread = float(getattr(config, "MAX_QUOTE_SPREAD_PCT", 0.02))
        decision = data_integrity.QuoteDecision(
            False, 0.0, data_integrity.SOURCE_NONE, data_integrity.REASON_NO_DATA)
        try:
            # Wait for the snapshot to populate; re-evaluate each poll and stop
            # as soon as a usable price appears (or the budget is exhausted).
            for _ in range(max(1, int(timeout * 4))):
                self.ib.sleep(0.25)
                decision = data_integrity.evaluate_order_quote(
                    last=getattr(ticker, "last", None),
                    bid=getattr(ticker, "bid", None),
                    ask=getattr(ticker, "ask", None),
                    close=getattr(ticker, "close", None),
                    delayed_last=getattr(ticker, "delayedLast", None),
                    delayed_close=getattr(ticker, "delayedClose", None),
                    delayed_bid=getattr(ticker, "delayedBid", None),
                    delayed_ask=getattr(ticker, "delayedAsk", None),
                    require_realtime=require_rt,
                    max_spread_pct=max_spread,
                )
                if decision.ok:
                    break
        finally:
            try:
                self.ib.cancelMktData(contract)
            except Exception:
                pass

        order_audit.log_event(
            order_audit.STAGE_QUOTE, symbol=symbol, ok=decision.ok,
            price=decision.price, source=decision.source, reason=decision.reason,
            require_realtime=require_rt,
        )
        if not decision.ok:
            logger.warning("Unusable order quote for %s -> %s", symbol, decision.reason)
        return decision

    def get_price(self, symbol: str, timeout: float = 5.0, allow_historical: bool = True) -> float:
        """Validated order price (Phase 4). Returns the sane real-time/delayed
        last or midpoint from get_order_quote, or 0.0 when no usable quote
        exists. The historical (prior daily close) fallback is used ONLY when the
        caller opts in (``allow_historical=True``) AND real-time is not required;
        order callers pass ALLOW_HISTORICAL_PRICE_FOR_ORDERS (False) so a stale
        close is never an order price."""
        decision = self.get_order_quote(symbol, timeout=timeout)
        if decision.ok and decision.price > 0:
            return decision.price

        require_rt = bool(getattr(config, "REQUIRE_REALTIME_DATA_FOR_ORDERS", False))
        if (not allow_historical) or require_rt:
            return 0.0

        # Fallback: last daily close from historical data. Defense-in-depth keeps
        # this OFF for order placement (callers pass allow_historical=False); it
        # only serves non-order callers that explicitly opt in.
        hist_price = self._historical_close(self._contract(symbol))
        if hist_price is not None:
            return hist_price
        return 0.0

    def _historical_close(self, contract) -> Optional[float]:
        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="5 D",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )
        except Exception:
            logger.warning("Historical data request failed for %s", contract.symbol, exc_info=True)
            return None
        for bar in reversed(bars or []):
            try:
                close = float(bar.close)
                if close > 0 and not math.isnan(close):
                    return close
            except (TypeError, ValueError):
                continue
        return None

    # ── Quantity / pricing helpers ───────────────────────────────
    def _calc_quantity(self, price: float, cash: float) -> int:
        if price <= 0:
            return 0
        max_value = cash * config.MAX_POSITION_PCT
        cap = getattr(config, "MAX_TRADE_VALUE", None)
        if cap is not None:
            max_value = min(max_value, float(cap))
        return max(int(max_value / price), 0)

    def _limit_price(self, action: str, price: float) -> float:
        offset = float(config.LIMIT_ORDER_OFFSET_PCT)
        if action == "BUY":
            return round(price * (1 + offset), 2)
        return round(price * (1 - offset), 2)

    def _initial_stop_price(self, action: str, price: float) -> float:
        if action == "BUY":
            return round(price * (1 - config.STOP_LOSS_PCT), 2)
        return round(price * (1 + config.STOP_LOSS_PCT), 2)

    def _take_profit_price(self, action: str, price: float) -> float:
        if action == "BUY":
            return round(price * (1 + config.TAKE_PROFIT_PCT), 2)
        return round(price * (1 - config.TAKE_PROFIT_PCT), 2)

    # ── Fill-driven order primitives (Phase 2: H4-H9) ────────────────────────
    @staticmethod
    def _today() -> str:
        return _dt.date.today().isoformat()

    def _order_wait(self, poll: float) -> None:
        """Yield to the event loop so fills/acks can arrive. Prefers the
        event-driven waitOnUpdate (returns early on any update) and falls back to
        sleep. Never raises into the order path."""
        try:
            self.ib.waitOnUpdate(timeout=poll)
        except Exception:
            try:
                self.ib.sleep(poll)
            except Exception:
                pass

    def _await_order_outcome(self, trade, timeout: Optional[float] = None,
                             poll: Optional[float] = None) -> order_exec.OrderResult:
        """Wait for `trade` to reach a terminal state, then classify the REAL
        outcome (H4/H5). Replaces the old fixed `ib.sleep(1)` + "accepted" check.

        Bounded by `ORDER_FILL_TIMEOUT_SECONDS`; a trade still working when the
        budget is exhausted is returned WORKING/TIMEOUT (or PARTIALLY_FILLED if
        some shares filled). Iteration-bounded so it is deterministic offline.
        """
        if timeout is None:
            timeout = float(getattr(config, "ORDER_FILL_TIMEOUT_SECONDS", 8.0))
        if poll is None:
            poll = float(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        steps = max(1, int(math.ceil(timeout / poll))) if poll > 0 else 1
        for _ in range(steps):
            if order_exec.is_terminal(trade):
                break
            self._order_wait(poll)
        if order_exec.is_terminal(trade):
            return order_exec.classify_trade(trade)
        return order_exec.outcome_on_timeout(trade)

    def _working_orders_plain(self) -> list:
        """Refresh open orders from the broker and adapt to plain dicts (with
        parent_id/qty/status) for the pure order_exec verifiers."""
        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        except Exception:
            logger.warning("Could not refresh open orders", exc_info=True)
        return live_invariants.adapt_working_orders(self.ib.openTrades())

    def _open_positions_plain(self) -> list:
        """Adapt broker positions() to the plain {"symbol","qty"} shape used by
        the pure reconciliation / live-invariant checkers. Broker = source of
        truth; read-only."""
        try:
            return live_invariants.adapt_positions(self.ib.positions())
        except Exception:
            logger.warning("Could not read positions for reconciliation", exc_info=True)
            return []

    def _cancel_symbol_working_orders(self, symbol: str) -> int:
        """Cancel ALL working orders for `symbol` (any side / type). Used before
        an emergency flatten so no resting exit child can fire AFTER we go flat
        and open an accidental short. Returns the number cancelled."""
        symbol = symbol.upper().strip()
        n = 0
        for t in self.ib.openTrades():
            if str(getattr(t.contract, "symbol", "")).upper().strip() != symbol:
                continue
            try:
                self.ib.cancelOrder(t.order)
                n += 1
            except Exception:
                logger.warning("Failed to cancel working order for %s", symbol, exc_info=True)
        if n:
            self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        return n

    def _market_close_symbol(self, symbol: str) -> dict:
        """Emergency-flatten one symbol (H7 fallback when exit protection cannot
        be made safe). Cancels EVERY working order for the symbol first -- so no
        resting child can execute after we go flat (which would open an accidental
        short) -- then market-closes the whole position."""
        symbol = symbol.upper().strip()
        self._cancel_symbol_working_orders(symbol)
        position = self.get_position(symbol)
        qty = int(abs(position))
        if qty <= 0:
            return {"symbol": symbol, "qty": 0, "outcome": "flat"}
        action = "SELL" if position > 0 else "BUY"
        try:
            contract = self._contract(symbol)
            trade = self.ib.placeOrder(contract, MarketOrder(action, qty))
            res = self._await_order_outcome(trade)
        except Exception:
            logger.error("Emergency close failed for %s", symbol, exc_info=True)
            return {"symbol": symbol, "qty": qty, "outcome": "error"}
        order_audit.log_event(
            order_audit.STAGE_FLATTEN, phase="unprotected_close", symbol=symbol,
            action=action, qty=qty, outcome=res.outcome, filled=res.filled,
        )
        logger.warning("Emergency close %s %s x%d -> %s (filled=%.0f)",
                       action, symbol, qty, res.outcome, res.filled)
        return {"symbol": symbol, "qty": qty, "action": action,
                "outcome": res.outcome, "filled": res.filled}

    def _cancel_oversized_exit_children(self, symbol: str, exit_action: str, filled: float) -> int:
        """Cancel EVERY working exit (``exit_action``) order for `symbol` whose
        quantity exceeds the actual filled qty -- protective stop, trailing stop,
        AND take-profit LIMIT alike (H6). After this no resting child can sell
        more than we hold, so a partial fill cannot leave an accidental-short
        trap. Returns the number cancelled."""
        symbol = symbol.upper().strip()
        exit_action = exit_action.upper().strip()
        cancelled = 0
        for t in self.ib.openTrades():
            if str(getattr(t.contract, "symbol", "")).upper().strip() != symbol:
                continue
            if str(getattr(t.order, "action", "")).upper().strip() != exit_action:
                continue
            cqty = float(getattr(t.order, "totalQuantity", 0) or 0)
            if order_exec.child_needs_resize(cqty, filled):  # cqty > filled
                try:
                    self.ib.cancelOrder(t.order)
                    cancelled += 1
                except Exception:
                    logger.warning("Failed to cancel oversized exit child for %s", symbol, exc_info=True)
        if cancelled:
            self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        return cancelled

    def _place_protection(self, symbol: str, filled: float, entry_basis: float,
                          exit_action: str = "SELL") -> bool:
        """Place SERVER-SIDE GTC protective exits for a long, sized to the ACTUAL
        filled qty and sharing ONE OCA group (C2 + H19):

          * an independent catastrophe hard stop at HARD_STOP_LOSS_PCT off the
            actual entry basis (avgFillPrice), and
          * the primary protective exit -- a trailing stop (USE_TRAILING_EXIT) or
            a fixed initial stop plus a take-profit limit.

        Every exit is qty == filled, tif=GTC, and shares the same ocaGroup
        (ocaType=1) so the first to fill auto-cancels the rest (no leftover
        resting SELL -> no accidental short at the broker). Returns True only when
        a GTC protective stop covering the fill is confirmed working AND no exit
        child exceeds the filled qty. A non-positive basis returns False (cannot
        price protection -- caller must flatten/halt; never guess).
        """
        symbol = symbol.upper().strip()
        target_qty = order_exec.protective_exit_qty(filled)
        if target_qty <= 0:
            return False
        basis = float(entry_basis or 0.0)
        if basis <= 0:
            logger.error("No valid entry basis (avgFillPrice/avgCost) for %s -> cannot price GTC protection", symbol)
            return False

        tif = str(getattr(config, "PROTECTIVE_TIF", "GTC"))
        oca = order_exec.oca_group_id(self._today(), symbol)
        hard_px = order_exec.hard_stop_price(basis, getattr(config, "HARD_STOP_LOSS_PCT", 0.03), "BUY")

        specs = []  # (kind, Order); entry side is BUY (long), so exits are SELL
        if hard_px > 0:
            specs.append(("hard_stop", Order(action=exit_action, orderType="STP",
                                             totalQuantity=target_qty, auxPrice=hard_px)))
        if bool(getattr(config, "USE_TRAILING_EXIT", False)):
            trail_pct = round(float(config.TRAILING_STOP_PCT) * 100.0, 4)
            specs.append(("trailing_stop", Order(
                action=exit_action, orderType="TRAIL", totalQuantity=target_qty,
                trailingPercent=trail_pct, trailStopPrice=self._initial_stop_price("BUY", basis))))
        else:
            specs.append(("stop", Order(action=exit_action, orderType="STP", totalQuantity=target_qty,
                                       auxPrice=self._initial_stop_price("BUY", basis))))
            specs.append(("take_profit", Order(action=exit_action, orderType="LMT", totalQuantity=target_qty,
                                              lmtPrice=self._take_profit_price("BUY", basis))))

        try:
            contract = self._contract(symbol)
            for kind, order in specs:
                order.tif = tif
                order.ocaGroup = oca
                order.ocaType = 1
                order.transmit = True
                order.orderRef = order_exec.deterministic_order_ref(
                    self._today(), symbol, f"{exit_action}_{kind.upper()}")
                self.ib.placeOrder(contract, order)
            self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
        except Exception:
            logger.error("Failed to place GTC protection for %s", symbol, exc_info=True)
            return False

        working = self._working_orders_plain()
        gtc_ok = order_exec.has_gtc_protective_stop(symbol, target_qty, working)
        oversized = order_exec.oversized_exit_children(symbol, exit_action, target_qty, working)
        ok = bool(gtc_ok) and not oversized
        order_audit.log_event(
            order_audit.STAGE_PROTECT, symbol=symbol, action=exit_action, qty=target_qty,
            oca_group=oca, tif=tif, hard_stop=hard_px, gtc_ok=gtc_ok,
            n_oversized=len(oversized), ok=ok,
        )
        return ok

    def _verify_or_protect(self, symbol: str, parent_order, result: order_exec.OrderResult,
                           entry_basis: float, exit_action: str = "SELL") -> order_exec.OrderResult:
        """After a long parent OPEN fills (full or partial), guarantee ONE of two
        safe end states (C2/H6/H7/H19):

          (a) a live GTC protective stop covers the ACTUAL filled qty, all exits
              share an OCA group, and NO exit child exceeds the filled qty, or
          (b) the position is flat (emergency-flattened; result marked aborted).

        Protection is placed AFTER the fill is confirmed, sized to the actual
        filled qty and priced off the actual avgFillPrice (entry_basis). Mutates
        and returns `result`.
        """
        filled = order_exec.protective_exit_qty(result.filled)

        # 1) On a partial fill, stop chasing the remainder so the held qty is final.
        if result.outcome == order_exec.PARTIALLY_FILLED:
            try:
                self.ib.cancelOrder(parent_order)
                self.ib.sleep(getattr(config, "ORDER_POLL_SECONDS", 0.25))
            except Exception:
                logger.warning("Could not cancel partial parent for %s", symbol, exc_info=True)

        # 2) Defensively clear any pre-existing oversized exit child for the symbol.
        self._cancel_oversized_exit_children(symbol, exit_action, filled)

        # 3) Place server-side GTC/OCA protection sized to the actual filled qty.
        placed_ok = self._place_protection(symbol, filled, entry_basis, exit_action)

        # 4) Safety gate: a covering GTC stop exists AND no exit child oversells.
        working = self._working_orders_plain()
        gtc_ok = order_exec.has_gtc_protective_stop(symbol, filled, working)
        oversized = order_exec.oversized_exit_children(symbol, exit_action, filled, working)
        if (not placed_ok) or (not gtc_ok) or oversized:
            reason = ("placement_failed" if not placed_ok
                      else "no_gtc_protective_stop" if not gtc_ok
                      else "oversized_exit_child")
            order_audit.log_event(
                order_audit.STAGE_STOP_CONFIRMED, symbol=symbol, ok=False,
                reason=reason, n_oversized=len(oversized), action="emergency_flatten",
            )
            logger.error("Unsafe GTC protection for filled %s (%s) -> emergency flatten", symbol, reason)
            self._market_close_symbol(symbol)
            # Phase 5B-4: alert AFTER the flatten has been issued, so the alert can
            # never delay or block the protective action. emit() places no order.
            alerts.emit(alerts.EVENT_PROTECTIVE_CHILD_FAILURE, symbol=symbol,
                        message="unsafe GTC protection -> emergency flatten",
                        reason=reason, n_oversized=len(oversized),
                        action="emergency_flatten")
            result.protective_ok = False
            result.aborted = True
            return result

        result.protective_ok = True
        order_audit.log_event(
            order_audit.STAGE_STOP_CONFIRMED, symbol=symbol, ok=True, qty=filled, tif="GTC",
        )
        return result

    def ensure_protective_stops(self) -> dict:
        """Startup protection invariant (Phase 3, plan 3.6): every open long must
        have a live GTC protective stop. Scan positions; for any unprotected long,
        attempt repair by placing GTC/OCA protection priced off the broker
        avgCost. If repair fails -- or avgCost is missing/invalid (cannot price;
        never guess) -- set the sticky new-entries halt flag and alert. Never
        continues trading silently with an unprotected long.

        Returns a report dict. Places only repair orders, never opens.
        """
        report = {"checked": 0, "unprotected": [], "repaired": [], "failed": []}
        try:
            longs = [p for p in self.ib.positions() if float(p.position) > 0]
        except Exception:
            logger.warning("Could not read positions for startup protection scan", exc_info=True)
            return report

        working = self._working_orders_plain()
        for p in longs:
            symbol = str(p.contract.symbol).upper().strip()
            qty = order_exec.protective_exit_qty(float(p.position))
            report["checked"] += 1
            if order_exec.has_gtc_protective_stop(symbol, qty, working):
                continue
            report["unprotected"].append(symbol)

            basis = self._position_basis(p)
            if basis <= 0:
                self._halt_new_entries = True
                report["failed"].append(symbol)
                order_audit.log_event(order_audit.STAGE_HALT, phase="startup_unprotected_no_basis",
                                      symbol=symbol, qty=qty)
                logger.error("Startup: unprotected long %s with no valid avgCost -> HALT new entries", symbol)
                continue

            ok = self._place_protection(symbol, qty, basis, "SELL")
            working = self._working_orders_plain()
            if ok and order_exec.has_gtc_protective_stop(symbol, qty, working):
                report["repaired"].append(symbol)
                order_audit.log_event(order_audit.STAGE_PROTECT, phase="startup_repair",
                                      symbol=symbol, qty=qty, basis=basis)
                logger.warning("Startup: repaired GTC protection for unprotected long %s x%d", symbol, qty)
            else:
                self._halt_new_entries = True
                report["failed"].append(symbol)
                order_audit.log_event(order_audit.STAGE_HALT, phase="startup_repair_failed",
                                      symbol=symbol, qty=qty)
                logger.error("Startup: could not protect long %s -> HALT new entries", symbol)
        return report

    def reconcile_startup_state(self) -> dict:
        """Startup reconciliation (Phase 5A, H18): BROKER = source of truth.

        Reads the broker's open positions and working orders, builds a pure
        broker-truth snapshot (reconciliation.build_snapshot) describing protected
        vs. unprotected longs, duplicate orderRefs, and orphan exit orders, and
        writes the result to the audit log. Local files are NEVER treated as truth
        over broker state.

        Repair is delegated to the existing, already-tested Phase-3 path
        (ensure_protective_stops): an unprotected long gets a GTC stop priced off
        the broker avgCost, or new entries are halted. This method places NO new
        entries and cancels nothing itself; it is safe to call on every startup.

        Returns a report dict:
          {"snapshot": <pure snapshot>, "protect": <ensure_protective_stops report>,
           "halt_new_entries": bool, "clean": bool}
        """
        positions = self._open_positions_plain()
        working = self._working_orders_plain()
        snapshot = reconciliation.build_snapshot(positions, working)

        order_audit.log_event(
            order_audit.STAGE_RECONCILE, phase="startup",
            source="broker", **reconciliation.audit_fields(snapshot),
        )
        logger.info("Startup reconciliation | %s", reconciliation.summary_line(snapshot))

        # Phase 5B-4: surface the broker-truth discrepancies to the operator. These
        # emit()s are disabled/log-only by default, never raise, and never place or
        # cancel an order -- the actual repair below still goes ONLY through the
        # existing Phase-3 ensure_protective_stops path.
        if snapshot["unprotected_longs"]:
            alerts.emit(alerts.EVENT_RECONCILE_UNPROTECTED_LONG,
                        message="startup: long(s) with no resting GTC protective stop",
                        symbols=list(snapshot["unprotected_longs"]))

        # Repair unprotected longs ONLY through the existing protective-repair
        # path. It scans broker positions itself (source of truth) and either
        # places a GTC stop or halts NEW entries -- it never opens a position.
        protect = {"checked": 0, "unprotected": [], "repaired": [], "failed": []}
        if snapshot["unprotected_longs"]:
            protect = self.ensure_protective_stops()

        # Duplicate refs / orphan exits are surfaced + audited for operator action
        # (and Phase 5B alerting). Phase 5A does NOT auto-cancel them: cancelling a
        # resting order is a mutation we keep out of the minimal reconcile step.
        if snapshot["duplicate_refs"]:
            logger.warning("Startup reconciliation: duplicate orderRefs at broker: %s",
                           snapshot["duplicate_refs"])
            alerts.emit(alerts.EVENT_DUPLICATE_ORDER_REF,
                        message="startup: duplicate orderRef(s) at broker",
                        refs=list(snapshot["duplicate_refs"]))
        if snapshot["orphan_exits"]:
            orphan_syms = sorted({str(o.get("symbol", "")).upper() for o in snapshot["orphan_exits"]})
            logger.warning("Startup reconciliation: orphan exit order(s) with no covering long: %s",
                           orphan_syms)
            alerts.emit(alerts.EVENT_ORPHAN_EXIT_ORDER,
                        message="startup: resting SELL order(s) with no covering long",
                        symbols=orphan_syms)

        return {
            "snapshot": snapshot,
            "protect": protect,
            "halt_new_entries": bool(self._halt_new_entries),
            "clean": bool(snapshot["clean"]),
        }

    # Back-compat alias: the plan refers to this capability as startup_reconcile().
    startup_reconcile = reconcile_startup_state

    def _finalize_open(self, symbol: str, action: str, parent_trade, result: order_exec.OrderResult,
                       intended_qty: int) -> order_exec.OrderResult:
        """Audit the verified open outcome and, for a real long fill, place and
        verify SERVER-SIDE GTC/OCA protection priced off the ACTUAL avgFillPrice
        (C2/H7/H19) -- never a stale signal price. Returns the (possibly aborted)
        result."""
        if result.outcome == order_exec.FILLED:
            order_audit.log_event(
                order_audit.STAGE_FILLED, symbol=symbol, action=action, filled=result.filled,
                avg_fill_price=result.avg_fill_price, intended_qty=intended_qty,
            )
        elif result.outcome == order_exec.PARTIALLY_FILLED:
            order_audit.log_event(
                order_audit.STAGE_PARTIAL, symbol=symbol, action=action, filled=result.filled,
                remaining=result.remaining, avg_fill_price=result.avg_fill_price, intended_qty=intended_qty,
            )
            # Phase 5B-4: alert on a partial fill (log-only/disabled by default;
            # never places an order). Protection sizing below is unaffected.
            alerts.emit(alerts.EVENT_PARTIAL_FILL, symbol=symbol, action=action,
                        filled=result.filled, remaining=result.remaining,
                        intended_qty=intended_qty)
        else:
            order_audit.log_event(
                order_audit.STAGE_REJECTED if result.outcome == order_exec.REJECTED else order_audit.STAGE_ACK,
                symbol=symbol, action=action, outcome=result.outcome, status=result.status,
                intended_qty=intended_qty,
            )
            if result.outcome == order_exec.REJECTED:
                alerts.emit(alerts.EVENT_ORDER_REJECTED, symbol=symbol, action=action,
                            status=result.status, intended_qty=intended_qty)
            logger.info("Open %s %s x%d -> %s (no fill)", action, symbol, intended_qty, result.outcome)
            return result

        # A real (full/partial) long fill must be backed by live GTC protection,
        # priced off the ACTUAL avg fill -- not the stale signal price.
        if action == "BUY":
            result = self._verify_or_protect(symbol, parent_trade.order, result,
                                             result.avg_fill_price, "SELL")
        logger.info(
            "Open %s %s -> %s filled=%.0f avg=%.2f protective_ok=%s aborted=%s",
            action, symbol, result.outcome, result.filled, result.avg_fill_price,
            result.protective_ok, result.aborted,
        )
        return result

    # ── Order placement ──────────────────────────────────────────
    def _close_position(self, symbol: str, action: str, qty: int, price: float,
                        note: str) -> order_exec.OrderResult:
        """Robustly close `qty` shares (H8): marketable-limit first, then escalate
        to a market order, confirming each attempt by fill until remaining == 0 or
        attempts are exhausted. Replaces the old single bare-limit + one-check
        close. Never silently treats an unconfirmed close as flat."""
        symbol = symbol.upper().strip()
        if qty <= 0:
            logger.warning("Invalid close qty=%s for %s %s", qty, action, symbol)
            return order_exec.OrderResult(outcome=order_exec.REJECTED, status="bad_qty")

        contract = self._contract(symbol)
        order_ref = order_exec.deterministic_order_ref(self._today(), symbol, action)
        max_attempts = max(1, int(getattr(config, "CLOSE_MAX_ATTEMPTS", 3)))
        remaining = int(qty)
        total_filled = 0.0
        last = order_exec.OrderResult(outcome=order_exec.WORKING)

        for attempt in range(1, max_attempts + 1):
            if attempt == 1 and price > 0:
                limit_price = self._limit_price(action, price)
                order = LimitOrder(action, remaining, limit_price)
                kind, px = "close_limit", limit_price
            else:
                order = MarketOrder(action, remaining)  # escalate to certainty of exit
                kind, px = "close_market", 0.0
            order.orderRef = order_ref
            order_audit.log_event(
                order_audit.STAGE_SUBMIT, kind=kind, note=note, symbol=symbol,
                action=action, qty=remaining, attempt=attempt, limit_price=px,
            )
            trade = self.ib.placeOrder(contract, order)
            res = self._await_order_outcome(trade)
            total_filled += res.filled
            last = res

            decision = order_exec.close_followup(res, attempt, max_attempts)
            stage = (order_audit.STAGE_FILLED if res.outcome == order_exec.FILLED
                     else order_audit.STAGE_PARTIAL if res.outcome == order_exec.PARTIALLY_FILLED
                     else order_audit.STAGE_REJECTED)
            order_audit.log_event(
                stage, kind=kind, note=note, symbol=symbol, action=action, attempt=attempt,
                outcome=res.outcome, filled=res.filled, remaining=res.remaining, decision=decision,
            )
            logger.info("%s | %s %s attempt=%d -> %s filled=%.0f remaining=%.0f (%s)",
                        note, action, symbol, attempt, res.outcome, res.filled, res.remaining, decision)

            if decision == "done":
                break
            if decision == "giveup":
                logger.error("Close %s %s NOT flat after %d attempts (remaining=%.0f) -> ALERT",
                             action, symbol, attempt, res.remaining)
                break
            remaining = max(int(round(res.remaining)), 0)
            if remaining <= 0:
                break

        last.filled = total_filled
        return last

    def _place_open_bracket(
        self, symbol: str, action: str, qty: int, price: float, confidence: float,
    ) -> order_exec.OrderResult:
        """Open a position with a plain marketable-limit entry, CONFIRM the real
        fill (H4/H5), then -- for a long -- place and verify SERVER-SIDE GTC/OCA
        protection sized to the actual filled qty and priced off the actual
        avgFillPrice (C2/H7/H19).

        Phase 3: protection is NOT attached at submission. It is placed only AFTER
        the fill is confirmed, so a partial fill can never leave an oversized
        resting exit and the hard stop is always based on the real entry price.
        """
        contract = self._contract(symbol)
        limit_price = self._limit_price(action, price)
        order = LimitOrder(action, qty, limit_price)
        order.orderRef = order_exec.deterministic_order_ref(self._today(), symbol, action)
        order.transmit = True

        order_audit.log_event(
            order_audit.STAGE_SUBMIT, kind="open", symbol=symbol, action=action,
            qty=qty, limit_price=limit_price, confidence=confidence, order_ref=order.orderRef,
        )
        trade = self.ib.placeOrder(contract, order)
        result = self._await_order_outcome(trade)
        return self._finalize_open(symbol, action, trade, result, qty)

    # ── Public API ───────────────────────────────────────────────
    def _duplicate_ref_working(self, symbol: str, action: str) -> bool:
        """Idempotency guard (H9): refuse to place an OPEN whose deterministic
        orderRef (date:symbol:action) is already working at the broker -- e.g. a
        duplicate after a restart / mid-flight. Complements has_working_order with
        intent-level matching and lays the groundwork for Phase 5 reconciliation.
        """
        ref = order_exec.deterministic_order_ref(self._today(), symbol, action)
        try:
            working = self._working_orders_plain()
        except Exception:
            return False
        return order_exec.has_order_ref(ref, working)

    def _record_if_filled(self, symbol: str, action: str, result: order_exec.OrderResult) -> None:
        """Record a daily trade ONLY for a real, held fill (H4/H6) -- never on a
        bare acknowledgement, rejection, timeout, or emergency-aborted entry."""
        if not order_exec.should_record_trade(result):
            return
        trades = risk_state.record_trade()
        order_audit.log_event(
            order_audit.STAGE_TRADE_RECORDED, symbol=symbol, action=action,
            qty=order_exec.protective_exit_qty(result.filled),
            avg_fill_price=result.avg_fill_price, daily_trade_count=trades, on="fill",
        )
        logger.info("Recorded daily trade #%d for %s (filled=%.0f)", trades, symbol, result.filled)

    def execute_signal(self, signal: Signal) -> bool:
        symbol = signal.symbol.upper().strip()
        action = signal.action.upper().strip()
        allow_short = bool(config.ALLOW_SHORT)
        min_cash = float(config.MIN_TRADE_CASH)

        if action == "HOLD":
            return False
        if action not in {"BUY", "SELL"}:
            logger.warning("Unsupported signal action for %s: %s", symbol, signal.action)
            return False

        if self.has_working_order(symbol):
            logger.info("Working order already exists for %s — skipping %s", symbol, action)
            return False

        position = self.get_position(symbol)
        price = self.get_price(
            symbol,
            allow_historical=bool(getattr(config, "ALLOW_HISTORICAL_PRICE_FOR_ORDERS", False)),
        )
        if price <= 0:
            logger.warning("Could not get valid order price for %s", symbol)
            return False

        if action == "BUY":
            if position > 0:
                logger.info("Already long %s position=%.2f — skipping BUY", symbol, position)
                return False
            if position < 0:
                qty = int(abs(position))
                # Cover an existing short: a robust, fill-confirmed close (H8).
                return self._close_position(symbol, "BUY", qty, price, "Close short").has_fill

            if not risk_state.can_open_more():
                logger.warning("Max daily trades (%d) reached — skipping BUY %s", config.MAX_DAILY_TRADES, symbol)
                return False

            cash = self.get_cash()
            if cash < min_cash:
                logger.warning("Insufficient cash for BUY %s: $%.2f", symbol, cash)
                return False
            qty = self._calc_quantity(price, cash)
            if qty == 0:
                logger.warning("Qty=0 for BUY %s price=%.2f cash=%.2f", symbol, price, cash)
                return False
            blocked, reason = self._new_entries_blocked(symbol, qty * price, signal, price)
            if blocked:
                order_audit.log_event(order_audit.STAGE_HALT, phase="entry_gate",
                                      symbol=symbol, action="BUY", reason=reason)
                logger.warning("New entry blocked for %s BUY (%s)", symbol, reason)
                return False
            if self._duplicate_ref_working(symbol, "BUY"):
                logger.info("Deterministic orderRef already working for %s BUY — skipping (idempotency)", symbol)
                return False
            result = self._place_open_bracket(symbol, "BUY", qty, price, signal.confidence)
            self._record_if_filled(symbol, "BUY", result)
            return result.occupies_slot

        # action == "SELL"
        if position > 0:
            qty = int(position)
            # Close an existing long: a robust, fill-confirmed close (H8).
            return self._close_position(symbol, "SELL", qty, price, "Close long").has_fill
        if position < 0:
            logger.info("Already short %s position=%.2f — skipping SELL", symbol, position)
            return False
        if not allow_short:
            logger.info("No long in %s — skipping SELL (ALLOW_SHORT=False)", symbol)
            return False

        if not risk_state.can_open_more():
            logger.warning("Max daily trades (%d) reached — skipping short SELL %s", config.MAX_DAILY_TRADES, symbol)
            return False

        cash = self.get_cash()
        if cash < min_cash:
            logger.warning("Insufficient cash for short SELL %s: $%.2f", symbol, cash)
            return False
        qty = self._calc_quantity(price, cash)
        if qty == 0:
            logger.warning("Qty=0 for short SELL %s price=%.2f cash=%.2f", symbol, price, cash)
            return False
        blocked, reason = self._new_entries_blocked(symbol, qty * price, signal, price)
        if blocked:
            order_audit.log_event(order_audit.STAGE_HALT, phase="entry_gate",
                                  symbol=symbol, action="SELL", reason=reason)
            logger.warning("New entry blocked for %s short SELL (%s)", symbol, reason)
            return False
        if self._duplicate_ref_working(symbol, "SELL"):
            logger.info("Deterministic orderRef already working for %s SELL — skipping (idempotency)", symbol)
            return False
        result = self._place_open_bracket(symbol, "SELL", qty, price, signal.confidence)
        self._record_if_filled(symbol, "SELL", result)
        return result.occupies_slot

    def execute_all(self, signals: List[Signal]) -> dict:
        open_symbols = {
            p.contract.symbol.upper()
            for p in self.ib.positions()
            if float(p.position) != 0.0
        }
        working_symbols = self.working_order_symbols()
        occupied_symbols = open_symbols | working_symbols
        planned_new_symbols: set = set()
        placed, skipped = 0, 0
        allow_short = bool(config.ALLOW_SHORT)

        for signal in signals:
            symbol = signal.symbol.upper().strip()
            action = signal.action.upper().strip()

            if action == "HOLD":
                skipped += 1
                continue

            if symbol in working_symbols:
                logger.info("Working order already exists for %s — skipping", symbol)
                skipped += 1
                continue

            current_position = self.get_position(symbol)
            opens_new = (
                (action == "BUY" and current_position == 0)
                or (action == "SELL" and current_position == 0 and allow_short)
            )

            if opens_new:
                planned_total = len(occupied_symbols | planned_new_symbols)
                if planned_total >= config.MAX_OPEN_POSITIONS:
                    logger.warning(
                        "Max positions (%d) reached — skipping %s",
                        config.MAX_OPEN_POSITIONS, symbol,
                    )
                    skipped += 1
                    continue

            if self.execute_signal(signal):
                placed += 1
                if opens_new:
                    planned_new_symbols.add(symbol)
            else:
                skipped += 1

        return {"placed": placed, "skipped": skipped, "total": len(signals)}

    # ── Emergency flatten (kill-switch) ──────────────────────────────────────
    def flatten_all(self, confirm: bool = False) -> dict:
        """Emergency kill-switch: cancel every working order and market-close
        every open position. PAPER-port-locked via connect() like all order
        paths.

        confirm=False (default) is a DRY-RUN: it reports exactly what it would
        cancel/flatten and places nothing. confirm=True actually cancels orders
        and submits MarketOrders to flatten. Returns a structured plan/result
        dict so callers (the `panic-flatten` command, later graceful shutdown)
        can print and verify it.

        Uses market orders on purpose: in an emergency, certainty of exit beats
        price. Handles longs (SELL to close) and any shorts (BUY to cover).
        """
        try:
            self.ib.reqAllOpenOrders()
            self.ib.sleep(1)
        except Exception:
            logger.warning("Could not refresh open orders before flatten", exc_info=True)

        open_trades = list(self.ib.openTrades())
        positions = [p for p in self.ib.positions() if float(p.position) != 0.0]

        plan = {
            "confirm": bool(confirm),
            "orders_to_cancel": [
                {
                    "symbol": str(getattr(t.contract, "symbol", "")),
                    "action": str(getattr(t.order, "action", "")),
                    "order_type": str(getattr(t.order, "orderType", "")),
                    "qty": float(getattr(t.order, "totalQuantity", 0) or 0),
                    "order_id": int(getattr(t.order, "orderId", 0) or 0),
                    "status": _order_status_name(t),
                }
                for t in open_trades
                if _is_order_working(_order_status_name(t))
            ],
            "positions_to_flatten": [
                {
                    "symbol": str(p.contract.symbol),
                    "qty": float(p.position),
                    "action": "SELL" if float(p.position) > 0 else "BUY",
                }
                for p in positions
            ],
            "cancelled": 0,
            "flattened": 0,
            "flatten_results": [],
        }

        order_audit.log_event(
            order_audit.STAGE_FLATTEN, phase="plan", confirm=bool(confirm),
            n_orders=len(plan["orders_to_cancel"]), n_positions=len(plan["positions_to_flatten"]),
        )

        if not confirm:
            return plan  # dry-run: nothing placed

        # 1) Cancel all working orders (protective children included; they are
        #    redundant once we market-close the underlying position).
        for t in open_trades:
            if not _is_order_working(_order_status_name(t)):
                continue
            try:
                self.ib.cancelOrder(t.order)
                plan["cancelled"] += 1
            except Exception:
                logger.warning("Failed to cancel order id=%s", getattr(t.order, "orderId", "?"), exc_info=True)
        self.ib.sleep(1)

        # 2) Market-close every open position.
        for p in positions:
            symbol = str(p.contract.symbol).upper().strip()
            qty = int(abs(float(p.position)))
            if qty <= 0:
                continue
            action = "SELL" if float(p.position) > 0 else "BUY"
            try:
                contract = self._contract(symbol)
                trade = self.ib.placeOrder(contract, MarketOrder(action, qty))
                # Confirm the emergency close by fill, not a fixed sleep (H8).
                res = self._await_order_outcome(trade)
                result = {
                    "symbol": symbol, "action": action, "qty": qty, "status": res.status,
                    "outcome": res.outcome, "filled": res.filled, "avg_fill": res.avg_fill_price,
                }
                plan["flatten_results"].append(result)
                plan["flattened"] += 1
                order_audit.log_event(order_audit.STAGE_FLATTEN, phase="executed", **result)
                logger.info("Flatten %s %s x%d -> %s filled=%.0f", action, symbol, qty, res.outcome, res.filled)
            except Exception:
                logger.warning("Failed to flatten %s", symbol, exc_info=True)
        self.ib.sleep(1)
        return plan
