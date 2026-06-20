"""paper_ledger.py - append-only journal of REAL paper fills + round-trip
reconstruction for the forward paper-test report.

PAPER-ONLY, ADDITIVE, NEVER-RAISE. This module has two halves:

  1. WRITER (used from inside the already-paper-locked `paper` run via
     ibkr_bridge): ``record_entry`` / ``record_stop`` / ``record_exit`` append one
     JSON object per line to ``reports/paper_trades.jsonl``. Like
     ``order_audit.log_event``, every write is wrapped so a logging failure can
     NEVER propagate into the order path - it cannot change trading behavior.
     Every event carries a durable ``event_id`` (a content hash) for stable
     reference.

  2. RECONSTRUCTOR (used only by the offline, read-only ``forward-test-report``):
     ``load_journal`` + ``reconstruct_paper_trades`` pair journal events into
     LONG round-trips (FIFO per symbol), with DOLLAR realized PnL and a stop-based
     R. It reuses ``expectancy.ClosedTrade`` / reason codes / ``r_multiple``
     (imported lazily) so the forward report and the backtest expectancy report
     stay definitionally reconciled. Open positions and exits-without-an-entry are
     EXCLUDED with a reason - never silently dropped.

Risk model: a forward paper trade is scored against the REAL resting broker
protective stop placed at entry (GTC/trailing). The resulting R is labeled
``risk_source="broker_protective_stop"`` - this is BROKER PROTECTIVE-STOP RISK,
NOT a Minervini setup-based R. When no valid protective stop is on record (or it
sits at/above entry) the trade is excluded from the R aggregates with a reason.

Event sourcing (one authoritative source per event type, so nothing double
counts):
  * ENTRY  - live hook only (carries the Signal reason + model scores).
  * STOP   - live hook only (carries the real broker protective-stop price).
  * EXIT   - broker-executions sweep only (broker-truth price + execId + orderRef;
             deduped by execId, or a deterministic fallback hash when an execId is
             absent). This is what captures a SERVER-SIDE stop-out that fills
             between one-shot runs and otherwise touches no code path.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import config
import order_exec  # pure/offline (no ib_insync); shared fill-recording semantics

logger = logging.getLogger(__name__)

_EPS = 1e-9

# ── Event types (journal `event` field) ──────────────────────────────────────
EVENT_ENTRY = "entry"
EVENT_STOP = "stop"
EVENT_EXIT = "exit"

# ── Exit kinds (how a position left) ─────────────────────────────────────────
EXIT_PROTECTIVE_STOP = "protective_stop"  # server-side GTC/trailing stop hit
EXIT_TAKE_PROFIT = "take_profit"          # protective take-profit limit hit
EXIT_SIGNAL_CLOSE = "signal_close"        # bot-issued SELL close (signal flip)
EXIT_UNKNOWN = "unknown"                  # could not classify from orderRef

# ── Risk source (matches expectancy.RISK_SOURCE_* string convention) ─────────
# A forward paper trade is scored against the REAL resting broker protective stop
# placed at entry. The risk_source string is deliberately explicit so reports/JSON
# can never be confused with a Minervini setup-based R.
RISK_SOURCE_PROTECTIVE_STOP = "broker_protective_stop"
RISK_SOURCE_UNAVAILABLE = "unavailable"


# ── Small, defensive coercion helpers (never raise) ──────────────────────────
def _num(x) -> Optional[float]:
    """Parse a finite float, else None. Never raises."""
    try:
        if x is None or x == "":
            return None
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sym(x) -> str:
    return str(x or "").strip().upper()


def _safe(value):
    """Coerce a value into something json.dumps can handle without raising."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


def _now() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _journal_path() -> Path:
    """Resolve the journal path dynamically so tests can patch config."""
    return Path(getattr(config, "FORWARD_TEST_JOURNAL",
                        config.REPORTS_DIR / "paper_trades.jsonl"))


def _default_session():
    return getattr(config, "FORWARD_TEST_SESSION", None)


def _event_id(record: dict) -> str:
    """Durable content-hash id for a journal line (stable + recomputable)."""
    try:
        return hashlib.sha1(
            json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


# ── Writer (append-only, never raises into the caller) ───────────────────────
def _append(record: dict, ts: Optional[str] = None) -> None:
    """Append one journal event. NEVER raises into the caller (order-path safe).

    A durable ``event_id`` (content hash, including ts) is stamped on every line.
    """
    try:
        out = {"ts": str(ts) if ts else _now()}
        for key, value in record.items():
            out[key] = _safe(value)
        out["event_id"] = _event_id(out)
        path = _journal_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(out, sort_keys=True) + "\n")
    except Exception:
        # Journal writes must never break the order path. Best-effort only.
        try:
            logger.debug("paper_ledger._append failed for event=%s",
                         record.get("event"), exc_info=True)
        except Exception:
            pass


def record_entry(signal, result, *, session=None, order_ref=None,
                 model_gate_status=None) -> None:
    """Append an ENTRY event for a real, held long fill.

    Gated by the SAME rule as the daily-trade counter
    (``order_exec.should_record_trade``): only FILLED / PARTIALLY_FILLED with
    ``filled > 0`` and ``not aborted``. ``signal`` is a predictor.Signal (model
    reason + RF/LSTM/Tech scores); ``result`` is an order_exec.OrderResult (ACTUAL
    filled qty + avgFillPrice). Both are read defensively so a plain namespace
    works in tests.
    """
    try:
        if not order_exec.should_record_trade(result):
            return
    except Exception:
        # Conservative fallback that still enforces FILLED/PARTIALLY_FILLED:
        # never record on a missing/other outcome, an aborted attempt, or no fill.
        if getattr(result, "outcome", None) not in (
                order_exec.FILLED, order_exec.PARTIALLY_FILLED):
            return
        if getattr(result, "aborted", False):
            return
        if (_num(getattr(result, "filled", None)) or 0.0) <= _EPS:
            return
    _append({
        "event": EVENT_ENTRY,
        "session": session if session is not None else _default_session(),
        "symbol": _sym(getattr(signal, "symbol", "")),
        "qty": _num(getattr(result, "filled", None)),
        "entry_avg_price": _num(getattr(result, "avg_fill_price", None)),
        "order_ref": order_ref,
        "signal_action": getattr(signal, "action", None),
        "confidence": _num(getattr(signal, "confidence", None)),
        "rf_score": _num(getattr(signal, "rf_score", None)),
        "lstm_score": _num(getattr(signal, "lstm_score", None)),
        "tech_score": _num(getattr(signal, "tech_score", None)),
        "reason": getattr(signal, "reason", None),
        "model_gate_status": model_gate_status,
    })


def record_stop(symbol, qty, stop_price, *, hard_stop=None, entry_basis=None,
                session=None, order_ref=None) -> None:
    """Append a STOP event recording the PRIMARY broker protective stop (the R
    denominator) and, for reference, the catastrophe ``hard_stop``."""
    _append({
        "event": EVENT_STOP,
        "session": session if session is not None else _default_session(),
        "symbol": _sym(symbol),
        "qty": _num(qty),
        "stop_price": _num(stop_price),
        "hard_stop": _num(hard_stop),
        "entry_basis": _num(entry_basis),
        "order_ref": order_ref,
    })


def exit_dedupe_key(*, symbol, side, qty, price, ts, order_ref) -> str:
    """Deterministic fallback identity for an exit when the broker gives no execId.
    Hashed from symbol/side/qty/price/time/order_ref."""
    raw = "|".join([
        _sym(symbol),
        str(side or "").upper().strip(),
        f"{_num(qty) or 0.0:.6f}",
        f"{_num(price) or 0.0:.6f}",
        str(ts or ""),
        str(order_ref or ""),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def record_exit(symbol, *, exit_kind, exit_avg_price=None, qty=None, result=None,
                side="SELL", session=None, order_ref=None, exec_id=None, ts=None,
                decision=None) -> None:
    """Append an EXIT event. Pulls qty/price from an OrderResult when given, else
    uses the explicit ``qty`` / ``exit_avg_price`` (the broker-executions path).
    ``ts`` preserves the real execution time when known (a stop-out that filled in
    the past), so the reconstructed equity curve is correctly ordered.

    Identity: ``exec_id`` when the broker supplies one, else a deterministic
    ``dedupe_id`` (see ``exit_dedupe_key``). Both are recorded; the sweep and the
    reconstructor dedupe on ``dedupe_id``.
    """
    if result is not None:
        if qty is None:
            qty = getattr(result, "filled", None)
        if exit_avg_price is None:
            exit_avg_price = getattr(result, "avg_fill_price", None)
    ts_final = str(ts) if ts else _now()
    dedupe_id = (str(exec_id) if exec_id is not None
                 else exit_dedupe_key(symbol=symbol, side=side, qty=qty,
                                      price=exit_avg_price, ts=ts_final,
                                      order_ref=order_ref))
    _append({
        "event": EVENT_EXIT,
        "session": session if session is not None else _default_session(),
        "symbol": _sym(symbol),
        "side": str(side or "").upper().strip() or "SELL",
        "qty": _num(qty),
        "exit_avg_price": _num(exit_avg_price),
        "exit_kind": exit_kind,
        "order_ref": order_ref,
        "exec_id": (str(exec_id) if exec_id is not None else None),
        "dedupe_id": dedupe_id,
        "decision": decision,
    }, ts=ts_final)


# ── Pure classification (testable; used by the broker-executions sweep) ───────
def classify_exit_kind(order_ref) -> str:
    """Classify a SELL execution's exit kind from its deterministic orderRef.

    Protective children use refs like ``DATE:SYM:SELL_HARD_STOP`` /
    ``SELL_TRAILING_STOP`` / ``SELL_STOP`` / ``SELL_TAKE_PROFIT`` (see
    order_exec.deterministic_order_ref + ibkr_bridge._place_protection); a
    bot-issued close uses ``DATE:SYM:SELL``.
    """
    ref = str(order_ref or "").upper()
    if "TAKE_PROFIT" in ref:
        return EXIT_TAKE_PROFIT
    if "TRAIL" in ref or "STOP" in ref:
        return EXIT_PROTECTIVE_STOP
    if ref.endswith(":SELL") or ref.endswith(":SLD"):
        return EXIT_SIGNAL_CLOSE
    return EXIT_UNKNOWN


# ── Reader (read-only; never raises) ─────────────────────────────────────────
def load_journal(path=None) -> List[dict]:
    """Read journal events in file order. Returns [] if missing/empty. Never raises."""
    try:
        p = Path(path) if path is not None else _journal_path()
        if not p.exists():
            return []
        events = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                events.append(json.loads(ln))
            except Exception:
                continue
        return events
    except Exception:
        return []


def recorded_exit_keys(path=None) -> set:
    """Set of exit dedupe ids already journaled (so the sweep skips re-recording)."""
    keys = set()
    for e in load_journal(path):
        if e.get("event") != EVENT_EXIT:
            continue
        k = e.get("dedupe_id") or e.get("exec_id")
        if k:
            keys.add(str(k))
    return keys


def open_long_qty(path=None) -> dict:
    """Net-open long qty per symbol from the journal (entries minus journaled
    exits, keeping only positives).

    The read-only broker-executions sweep uses this to journal an exit ONLY for a
    symbol that still has a tracked open long; a SELL execution for an untracked
    symbol (no journaled entry) is ignored rather than recorded as a phantom
    round-trip. Never raises.
    """
    net: dict = {}
    for e in load_journal(path):
        et = e.get("event")
        sym = _sym(e.get("symbol"))
        q = _num(e.get("qty")) or 0.0
        if et == EVENT_ENTRY:
            net[sym] = net.get(sym, 0.0) + q
        elif et == EVENT_EXIT:
            net[sym] = net.get(sym, 0.0) - q
    return {s: q for s, q in net.items() if q > _EPS}


# ── Reconstructed round-trip ─────────────────────────────────────────────────
@dataclass
class PaperTrade:
    """One reconstructed paper round-trip (or an excluded candidate). DOLLAR
    realized PnL; a broker protective-stop-based R. ``to_closed_trade`` maps to
    ``expectancy.ClosedTrade`` so the SAME ``expectancy.compute_metrics`` powers
    the headline (win rate / expectancy_R / profit factor) for both reports."""
    symbol: str
    side: str = "long"
    entry_date: Optional[str] = None
    exit_date: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    qty: Optional[float] = None
    realized_pnl: Optional[float] = None
    realized_pnl_pct: Optional[float] = None
    initial_risk: Optional[float] = None
    r_multiple: Optional[float] = None
    risk_source: str = RISK_SOURCE_UNAVAILABLE
    exit_kind: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    rf_score: Optional[float] = None
    lstm_score: Optional[float] = None
    tech_score: Optional[float] = None
    model_gate_status: Optional[str] = None
    session: Optional[str] = None
    pnl_included: bool = False
    r_included: bool = False
    exclusion_reason: str = ""
    r_exclusion_reason: str = ""

    def to_closed_trade(self):
        """Map to expectancy.ClosedTrade (fold_start carries the session label)."""
        import expectancy as _exp
        return _exp.ClosedTrade(
            symbol=self.symbol,
            fold_start=str(self.session or "live"),
            side=self.side,
            entry_date=self.entry_date,
            exit_date=self.exit_date,
            entry_price=self.entry_price,
            exit_price=self.exit_price,
            realized_pnl=self.realized_pnl,
            initial_risk=self.initial_risk,
            r_multiple=self.r_multiple,
            risk_source=self.risk_source,
            pnl_included=self.pnl_included,
            r_included=self.r_included,
            exclusion_reason=self.exclusion_reason,
            r_exclusion_reason=self.r_exclusion_reason,
        )

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k in ("realized_pnl", "realized_pnl_pct", "initial_risk", "r_multiple",
                  "entry_price", "exit_price", "qty"):
            if d.get(k) is not None:
                d[k] = round(float(d[k]), 6)
        return d


def _type_rank(event_type) -> int:
    return {EVENT_ENTRY: 0, EVENT_STOP: 1, EVENT_EXIT: 2}.get(event_type, 3)


def _lot_from_entry(e: dict) -> dict:
    return {
        "qty": _num(e.get("qty")) or 0.0,
        "entry_price": _num(e.get("entry_avg_price")),
        "entry_ts": str(e.get("ts", "")) or None,
        "order_ref": e.get("order_ref"),
        "stop_price": None,
        "hard_stop": None,
        "reason": e.get("reason"),
        "confidence": _num(e.get("confidence")),
        "rf_score": _num(e.get("rf_score")),
        "lstm_score": _num(e.get("lstm_score")),
        "tech_score": _num(e.get("tech_score")),
        "model_gate_status": e.get("model_gate_status"),
        "session": e.get("session"),
    }


def _stop_from_event(e: dict) -> dict:
    return {
        "stop_price": _num(e.get("stop_price")),
        "hard_stop": _num(e.get("hard_stop")),
        "order_ref": e.get("order_ref"),
    }


def _attach_stop(lot: dict, stop: dict) -> None:
    lot["stop_price"] = stop.get("stop_price")
    lot["hard_stop"] = stop.get("hard_stop")


def _pop_matching_stop(pending_stops: "deque[dict]", order_ref) -> Optional[dict]:
    """Pop a pre-entry STOP that belongs to this entry, falling back to FIFO.

    Real hook order is STOP then ENTRY because protection is placed inside
    _place_open_bracket before execute_signal records the entry. The deterministic
    BUY order_ref ties that pre-entry stop back to the entry; if either side lacks
    a ref, FIFO is the safest per-symbol fallback.
    """
    if not pending_stops:
        return None
    ref = str(order_ref or "").strip()
    if ref:
        for i, stop in enumerate(pending_stops):
            stop_ref = str(stop.get("order_ref") or "").strip()
            if stop_ref == ref:
                del pending_stops[i]
                return stop
        for i, stop in enumerate(pending_stops):
            if not str(stop.get("order_ref") or "").strip():
                del pending_stops[i]
                return stop
        return None
    for i, stop in enumerate(pending_stops):
        if not str(stop.get("order_ref") or "").strip():
            del pending_stops[i]
            return stop
    return None


def _attach_stop_to_open_lot(open_lots: "deque[dict]", stop: dict) -> bool:
    """Attach a STOP to an open lot, preferring the durable order_ref match."""
    ref = str(stop.get("order_ref") or "").strip()
    if ref:
        for lot in reversed(open_lots):
            lot_ref = str(lot.get("order_ref") or "").strip()
            if lot.get("stop_price") is None and lot_ref == ref:
                _attach_stop(lot, stop)
                return True
        return False
    for lot in reversed(open_lots):
        if (lot.get("stop_price") is None and
                not str(lot.get("order_ref") or "").strip()):
            _attach_stop(lot, stop)
            return True
    return False


def _orphan_exit(symbol, exit_ts, exit_price, qty, exit_kind, session, _exp) -> PaperTrade:
    """An exit (or surplus portion) with no covering open lot. Never paired
    backwards with a later entry; excluded with a reason."""
    return PaperTrade(
        symbol=symbol, exit_date=exit_ts, exit_price=exit_price, qty=qty,
        exit_kind=exit_kind, session=session,
        exclusion_reason=_exp.REASON_ORPHAN_EXIT,
        r_exclusion_reason=_exp.R_REASON_NO_PNL)


def _close_trade(symbol: str, lot: dict, qty: float, exit_price, exit_ts,
                 exit_kind, _exp) -> PaperTrade:
    """Build one closed round-trip from a lot (or portion) and an exit."""
    base = dict(
        symbol=symbol, side="long", entry_date=lot.get("entry_ts"),
        exit_date=exit_ts, entry_price=lot.get("entry_price"), exit_price=exit_price,
        qty=qty, exit_kind=exit_kind, reason=lot.get("reason"),
        confidence=lot.get("confidence"), rf_score=lot.get("rf_score"),
        lstm_score=lot.get("lstm_score"), tech_score=lot.get("tech_score"),
        model_gate_status=lot.get("model_gate_status"), session=lot.get("session"),
    )
    entry_price = lot.get("entry_price")
    if entry_price is None or entry_price <= 0:
        return PaperTrade(**base, exclusion_reason=_exp.REASON_MISSING_ENTRY_PRICE,
                          r_exclusion_reason=_exp.R_REASON_NO_PNL)
    if exit_price is None or exit_price <= 0:
        return PaperTrade(**base, exclusion_reason=_exp.REASON_MISSING_EXIT_PRICE,
                          r_exclusion_reason=_exp.R_REASON_NO_PNL)

    realized = qty * (exit_price - entry_price)
    pct = (exit_price / entry_price) - 1.0

    # R from the REAL broker protective stop (fall back to the catastrophe hard
    # stop only if the primary is missing). A stop at/above entry yields no valid
    # risk -> excluded from R (never silently inflated). This is broker
    # protective-stop risk, NOT a Minervini setup R.
    stop = lot.get("stop_price")
    if stop is None or stop <= 0:
        stop = lot.get("hard_stop")
    initial_risk = None
    r = None
    risk_source = RISK_SOURCE_UNAVAILABLE
    r_included = False
    r_reason = _exp.R_REASON_UNAVAILABLE_RISK
    if stop is not None and 0 < stop < entry_price:
        initial_risk = qty * (entry_price - stop)
        r = _exp.r_multiple(realized, initial_risk)
        if r is not None:
            risk_source = RISK_SOURCE_PROTECTIVE_STOP
            r_included = True
            r_reason = ""
        else:
            r_reason = _exp.R_REASON_INVALID_RISK

    return PaperTrade(**base, realized_pnl=realized, realized_pnl_pct=pct,
                      initial_risk=initial_risk, r_multiple=r, risk_source=risk_source,
                      pnl_included=True, r_included=r_included, r_exclusion_reason=r_reason)


def reconstruct_paper_trades(events: List[dict]) -> List[PaperTrade]:
    """Pair journal events into LONG round-trips (FIFO per symbol).

    A still-open lot at the end is EXCLUDED as ``open_at_period_end`` (reported
    separately as an open position); an exit with no covering open lot (or a
    surplus beyond it) is EXCLUDED as ``orphan_exit``. Exit events are deduped by
    ``dedupe_id`` (execId or fallback hash). Reuses expectancy's reason codes +
    ClosedTrade so the metrics layer is shared with the backtest report.
    """
    import expectancy as _exp

    by_symbol: "OrderedDict[str, List[dict]]" = OrderedDict()
    for e in events:
        by_symbol.setdefault(_sym(e.get("symbol")), []).append(e)

    trades: List[PaperTrade] = []
    seen_exit: set = set()

    for symbol, evs in by_symbol.items():
        evs = sorted(evs, key=lambda e: (str(e.get("ts", "")),
                                         _type_rank(e.get("event"))))
        open_lots: "deque[dict]" = deque()
        pending_stops: "deque[dict]" = deque()

        for e in evs:
            etype = e.get("event")
            if etype == EVENT_ENTRY:
                lot = _lot_from_entry(e)
                stop = _pop_matching_stop(pending_stops, lot.get("order_ref"))
                if stop is not None:
                    _attach_stop(lot, stop)
                open_lots.append(lot)
            elif etype == EVENT_STOP:
                stop = _stop_from_event(e)
                # Attach by durable order_ref when available; legacy ref-less
                # events fall back to most-recent/FIFO within the same symbol.
                if not _attach_stop_to_open_lot(open_lots, stop):
                    pending_stops.append(stop)
            elif etype == EVENT_EXIT:
                key = e.get("dedupe_id") or e.get("exec_id")
                if key is not None:
                    if str(key) in seen_exit:
                        continue
                    seen_exit.add(str(key))
                exit_price = _num(e.get("exit_avg_price"))
                exit_ts = str(e.get("ts", "")) or None
                exit_kind = e.get("exit_kind") or classify_exit_kind(e.get("order_ref"))
                session = e.get("session")
                remaining = _num(e.get("qty")) or 0.0
                if remaining <= _EPS:
                    continue
                if not open_lots:
                    trades.append(_orphan_exit(symbol, exit_ts, exit_price,
                                               remaining, exit_kind, session, _exp))
                    continue
                while remaining > _EPS and open_lots:
                    lot = open_lots[0]
                    take = min(remaining, lot["qty"])
                    trades.append(_close_trade(symbol, lot, take, exit_price,
                                               exit_ts, exit_kind, _exp))
                    lot["qty"] -= take
                    remaining -= take
                    if lot["qty"] <= _EPS:
                        open_lots.popleft()
                if remaining > _EPS:
                    # Sold more than we ever opened (carried-in / untracked entry).
                    trades.append(_orphan_exit(symbol, exit_ts, exit_price,
                                               remaining, exit_kind, session, _exp))

        # Lots still open at the end never closed in-window -> open positions.
        for lot in open_lots:
            trades.append(PaperTrade(
                symbol=symbol, side="long", entry_date=lot.get("entry_ts"),
                entry_price=lot.get("entry_price"), qty=lot.get("qty"),
                reason=lot.get("reason"), confidence=lot.get("confidence"),
                rf_score=lot.get("rf_score"), lstm_score=lot.get("lstm_score"),
                tech_score=lot.get("tech_score"),
                model_gate_status=lot.get("model_gate_status"),
                session=lot.get("session"),
                exclusion_reason=_exp.REASON_OPEN_AT_PERIOD_END,
                r_exclusion_reason=_exp.R_REASON_NO_PNL))

    return trades
