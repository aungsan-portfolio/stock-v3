"""trade_coach.py — Guided Paper Trading Coach (READ-MOSTLY, beginner-friendly).

This module turns ensemble signals into a trade lesson + a safe paper-trade
preview. It NEVER places orders on its own. Order placement is gated by:

    1. `BUY` signal from the existing Predictor ensemble
    2. confidence >= config.COACH_MIN_CONFIDENCE_FOR_CANDIDATE
    3. NO existing position in the symbol
    4. NO working order in the symbol
    5. Explicit chart-check acknowledgement (caller passes chart_checked=True)
    6. Explicit user confirm (caller passes confirm=True)

Existing IBKRBridge risk controls (MAX_TRADE_VALUE, MAX_POSITION_PCT,
MAX_OPEN_POSITIONS, MAX_DAILY_TRADES, ALLOW_SHORT=False, paper-port lock)
are NOT bypassed. The bridge is still the only thing that touches the order
ticket; the coach just decides *whether* to call it.

Public API:
    build_trade_lesson(signal, position_info=None, open_order_info=None) -> dict
    build_trade_preview(signal, cash, current_positions, open_orders) -> dict
    print_trade_lesson(lesson)
    print_trade_preview(preview)
    write_trade_note(lesson_or_preview, action_taken, order_result=None)
    select_coach_candidates(signals, max_n=1) -> list[Signal]
    assert_paper_trading_only()                       # hard live-trading block
    assess_chart_status(signal) -> (status, detail)   # TOO_EXTENDED / OK / ...
    evaluate_daily_candidate(...) -> dict             # accept/skip + reason
    print_daily_candidate(evaluation)

The `daily-coach` flow may place MORE than one paper order in a single run
(up to config.COACH_MAX_PAPER_TRADES_PER_RUN) so a beginner can practice and
learn faster — but it is still PAPER-ONLY and every existing risk control is
preserved. Live trading is hard-blocked (see assert_paper_trading_only).
"""
from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, List, Optional, Tuple

import config
import daytrading_levels
import risk_math

logger = logging.getLogger(__name__)


# ─── Hard live-trading block ─────────────────────────────────────────────────
# This bot is paper-trading only. There is no live-trading mode and no live
# account port. Any attempt to enable live trading must fail loudly here.
_LIVE_TRADING_REFUSAL = "Live trading is disabled. This bot is paper-trading only."


class LiveTradingDisabledError(RuntimeError):
    """Raised when any config/command attempts to enable live trading."""


def assert_paper_trading_only() -> None:
    """Refuse to proceed unless the configuration is unambiguously paper-only.

    Checks, in order:
      * config.COACH_LIVE_TRADING_ENABLED must be falsy (no live mode exists).
      * config.REQUIRE_PAPER_PORT must stay True (paper-port lock intact).
      * With the paper-port lock on, IBKR_PORT must equal PAPER_IBKR_PORT —
        i.e. no live account port may be configured.
      * Short selling must stay disabled (ALLOW_SHORT=False).
      * Historical-close pricing for orders must stay disabled.

    Raises LiveTradingDisabledError with a fixed message on any violation. This
    is a belt-and-suspenders guard *in front of* IBKRBridge — the bridge still
    enforces the paper-port lock itself; this just refuses earlier and clearer.
    """
    if bool(getattr(config, "COACH_LIVE_TRADING_ENABLED", False)):
        raise LiveTradingDisabledError(_LIVE_TRADING_REFUSAL)

    if not bool(getattr(config, "REQUIRE_PAPER_PORT", True)):
        # Refusing here means we never silently drop the paper-port lock.
        raise LiveTradingDisabledError(_LIVE_TRADING_REFUSAL)

    paper_port = int(getattr(config, "PAPER_IBKR_PORT", 7497))
    if int(getattr(config, "IBKR_PORT", paper_port)) != paper_port:
        # A non-paper port while the lock is on is treated as a live-trading attempt.
        raise LiveTradingDisabledError(_LIVE_TRADING_REFUSAL)

    if bool(getattr(config, "ALLOW_SHORT", False)):
        raise LiveTradingDisabledError(_LIVE_TRADING_REFUSAL)

    if bool(getattr(config, "ALLOW_HISTORICAL_PRICE_FOR_ORDERS", False)):
        raise LiveTradingDisabledError(_LIVE_TRADING_REFUSAL)


# ─── Ticker education map (short, beginner-friendly) ─────────────────────────
# Keep it short. Beginners learn one ticker at a time.
_TICKER_GUIDE = {
    "SPY":  "S&P 500 ETF — broad US large-cap market proxy.",
    "QQQ":  "Nasdaq-100 ETF — large-cap tech-heavy Nasdaq exposure.",
    "VTI":  "Total US stock market ETF — broadest US equity diversification.",
    "AAPL": "Apple Inc. — consumer electronics, services, silicon.",
    "MSFT": "Microsoft Corp. — software, cloud (Azure), enterprise, AI.",
    "GOOGL":"Alphabet Inc. — Google search, YouTube, cloud, Android.",
    "AMZN": "Amazon.com Inc. — e-commerce, AWS cloud, ads, devices.",
    "NVDA": "NVIDIA Corp. — GPUs, AI accelerators, data-center chips.",
    "TSLA": "Tesla Inc. — EVs, energy, autonomy, FSD software.",
    "AMD":  "Advanced Micro Devices — CPUs, GPUs, data-center accelerators.",
    "META": "Meta Platforms — Facebook, Instagram, WhatsApp, AI/Reality Labs.",
    "AVGO": "Broadcom Inc. — semiconductors, infrastructure software.",
    "NFLX": "Netflix Inc. — streaming video, ad-tier, content.",
    "COST": "Costco Wholesale — membership warehouse retail.",
    "JPM":  "JPMorgan Chase — largest US bank, diversified financials.",
    "BAC":  "Bank of America — consumer & commercial banking.",
    "XOM":  "Exxon Mobil — integrated oil & gas major.",
    "UNH":  "UnitedHealth Group — health insurance, Optum services.",
}


def _ticker_guide(symbol: str) -> str:
    return _TICKER_GUIDE.get(
        symbol.upper().strip(),
        f"{symbol.upper()} — equity ticker. Look up sector, market cap, and recent "
        f"news on a finance site before considering any trade.",
    )


def _format_pct(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _format_money(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _daytrade_context_from_ohlcv(ohlcv) -> Dict[str, Any]:
    """Build optional daily OHLCV-derived context without fetching data."""
    context = {
        "previous_day_pivot_levels": None,
        "gap_pct": None,
        "atr_dollars": None,
    }
    if ohlcv is None:
        return context

    try:
        context["atr_dollars"] = daytrading_levels.atr_dollars_from_ohlc(
            ohlcv, period=int(getattr(config, "ATR_PERIOD", 14))
        )
    except Exception:
        logger.debug("ATR preview unavailable", exc_info=True)

    try:
        if len(ohlcv) >= 2:
            prev = ohlcv.iloc[-2]
            today = ohlcv.iloc[-1]
            context["previous_day_pivot_levels"] = daytrading_levels.pivot_points(
                prev["High"], prev["Low"], prev["Close"]
            )
            context["gap_pct"] = daytrading_levels.gap_pct(today["Open"], prev["Close"])
    except Exception:
        logger.debug("Daily pivot/gap preview unavailable", exc_info=True)

    return context


def _daytrade_formula_fields(signal, cash: float, ohlcv=None) -> Dict[str, Any]:
    """Read-only day-trading formula preview fields.

    This does not feed quantity, stop, or target into IBKRBridge. It exists only
    to print practice math and to block weak/invalid paper practice candidates.
    """
    symbol = str(getattr(signal, "symbol", "")).upper().strip()
    action = str(getattr(signal, "action", "")).upper().strip()
    context = _daytrade_context_from_ohlcv(ohlcv)
    fields: Dict[str, Any] = {
        "planned_entry": None,
        "suggested_stop": None,
        "suggested_target_2r": None,
        "suggested_target_3r": None,
        "risk_per_share": None,
        "reward_to_2r": None,
        "rr_2r": None,
        "suggested_shares_by_risk": 0,
        "planned_risk_dollars": None,
        "daytrade_formula_tradeable": False,
        "daytrade_formula_reason": "daytrade formula preview unavailable",
        "daytrade_formula_note": (
            "Formula preview only; existing IBKRBridge sizing still applies."
        ),
        **context,
    }

    if action != "BUY":
        fields["daytrade_formula_reason"] = f"action is {action}, not BUY"
        return fields

    try:
        entry = float(getattr(signal, "price"))
        if entry <= 0:
            raise ValueError("entry price must be positive")

        atr = fields.get("atr_dollars")
        if atr is not None and float(atr) > 0:
            stop = risk_math.atr_stop_price(
                entry,
                atr,
                atr_multiple=float(getattr(config, "DAYTRADE_ATR_STOP_MULTIPLE", 1.5)),
            )
        else:
            stop = entry * (1 - float(getattr(config, "STOP_LOSS_PCT", 0.004)))

        target_2r = risk_math.required_target(entry, stop, 2.0)
        target_3r = risk_math.required_target(entry, stop, 3.0)
        rps = risk_math.risk_per_share(entry, stop)
        reward_2r = risk_math.reward_per_share(entry, target_2r)
        rr_2r = risk_math.risk_reward_ratio(entry, stop, target_2r)
        shares = risk_math.shares_for_risk(
            cash,
            float(getattr(config, "DAYTRADE_PAPER_RISK_PCT", 0.001)),
            entry,
            stop,
            max_trade_value=getattr(config, "MAX_TRADE_VALUE", None),
            max_position_pct=getattr(config, "MAX_POSITION_PCT", None),
        )
        planned_risk = risk_math.planned_risk_dollars(shares, entry, stop)

        fields.update(
            {
                "planned_entry": entry,
                "suggested_stop": stop,
                "suggested_target_2r": target_2r,
                "suggested_target_3r": target_3r,
                "risk_per_share": rps,
                "reward_to_2r": reward_2r,
                "rr_2r": rr_2r,
                "suggested_shares_by_risk": shares,
                "planned_risk_dollars": planned_risk,
                "daytrade_formula_tradeable": shares > 0,
                "daytrade_formula_reason": None if shares > 0 else "suggested shares by risk is 0",
            }
        )
    except Exception as exc:
        logger.debug("Daytrade formula preview unavailable for %s", symbol, exc_info=True)
        fields["daytrade_formula_reason"] = f"invalid risk formula inputs: {exc}"

    return fields


def daytrade_refusal_reasons(
    signal,
    preview: Dict[str, Any],
    positions: Optional[Dict[str, Any]] = None,
    working: Optional[set] = None,
) -> List[str]:
    """Strict paper-practice refusal gates shared by paper/daily coach."""
    positions = positions or {}
    working = working or set()

    symbol = str(getattr(signal, "symbol", "")).upper().strip()
    action = str(getattr(signal, "action", "")).upper().strip()
    confidence = float(getattr(signal, "confidence", 0.0))
    buy_threshold = float(getattr(config, "BUY_THRESHOLD", 0.65))
    min_rr = float(getattr(config, "DAYTRADE_MIN_RR", 2.0))

    reasons: List[str] = []
    if action != "BUY":
        reasons.append(f"action is {action}, not BUY")
    if confidence < buy_threshold:
        reasons.append(f"confidence {confidence:.2f} < BUY_THRESHOLD {buy_threshold:.2f}")
    if symbol in positions:
        reasons.append("already holding this symbol")
    if symbol in working:
        reasons.append("a working order already exists for this symbol")
    if not preview.get("tradeable", False):
        reasons.append(preview.get("skip_reason") or "preview not tradeable")

    try:
        rr_2r = float(preview.get("rr_2r"))
    except (TypeError, ValueError):
        rr_2r = 0.0
    if rr_2r + 1e-9 < min_rr:
        reasons.append(f"R:R {rr_2r:.2f} < DAYTRADE_MIN_RR {min_rr:.2f}")

    try:
        risk_per_share = float(preview.get("risk_per_share"))
    except (TypeError, ValueError):
        risk_per_share = 0.0
    if risk_per_share <= 0:
        reasons.append("risk/share is invalid")

    try:
        suggested_shares = int(preview.get("suggested_shares_by_risk", 0))
    except (TypeError, ValueError):
        suggested_shares = 0
    if suggested_shares <= 0:
        reasons.append("suggested shares by risk is 0")

    if preview.get("daytrade_formula_reason"):
        reason = str(preview.get("daytrade_formula_reason"))
        if reason not in reasons:
            reasons.append(reason)

    return reasons


# ─── Minervini / SEPA overlay view (READ-ONLY, default-OFF) ──────────────────
# A purely educational, additive layer that explains a symbol's Minervini/SEPA
# setup ("why blocked / why this setup") next to the existing lesson/preview.
# It is gated behind BOTH config switches and is a complete no-op unless both are
# True. It NEVER places, sizes, or blocks an order in this milestone (M1): the
# advisory 1R stop here is for explanation only and is clearly marked as such.
def minervini_coach_enabled() -> bool:
    """True only when BOTH the master overlay switch and the coach switch are on."""
    return bool(getattr(config, "MINERVINI_OVERLAY_ENABLED", False)) and bool(
        getattr(config, "MINERVINI_COACH_ENABLED", False)
    )


def build_minervini_view(
    signal,
    ohlcv=None,
    benchmark_df=None,
) -> Optional[Dict[str, Any]]:
    """Build a read-only Minervini/SEPA explanation dict for one signal.

    Returns None when the overlay/coach switches are off (so callers add nothing
    to their output) — this is the default. When enabled, it evaluates the symbol
    via the M0 ``minervini`` module on the last CLOSED bar (no lookahead) and
    returns a beginner-readable dict. NEVER raises into the caller: any data /
    compute failure degrades to an "unavailable" view, mirroring
    ``assess_chart_status``. This milestone is explanation-only — nothing here
    blocks a BUY or changes position size.

    Args:
        signal:        a predictor.Signal-like object (needs .symbol).
        ohlcv:         optional pre-fetched OHLCV DataFrame; fetched lazily if None.
        benchmark_df:  optional benchmark (e.g. SPY) OHLCV for relative strength.

    Returns:
        None, or a dict with keys: available, symbol, stage2_ok, reasons,
        reason_text, vcp_like, pocket_pivot, rs_rank, pivot_low,
        advisory_stop_price, setup_summary, note.
    """
    if not minervini_coach_enabled():
        return None

    symbol = str(getattr(signal, "symbol", "")).upper().strip()
    note = (
        "Educational overlay only. The Stage-2 trend template, VCP-like "
        "approximation, and pocket pivot are extra context; they do NOT place, "
        "size, or block any order in this view."
    )

    try:
        import minervini

        df = ohlcv
        if df is None:
            from data_manager import fetch_ohlcv
            df = fetch_ohlcv(symbol)
        verdict = minervini.evaluate_entry(df, benchmark_df=benchmark_df)
        advisory_stop = minervini.minervini_stop_price(verdict.pivot_low)
    except Exception:
        logger.debug("Minervini coach view unavailable for %s", symbol, exc_info=True)
        return {
            "available": False,
            "symbol": symbol,
            "reason_text": "Could not evaluate the Minervini/SEPA setup for this symbol.",
            "note": note,
        }

    reason_text = (
        "Passes the Stage-2 trend template."
        if verdict.stage2_ok
        else "Not a Stage-2 setup: " + (", ".join(verdict.reasons) or "trend template not met")
    )

    setup_bits: List[str] = []
    setup_bits.append("Stage-2 uptrend: " + ("YES" if verdict.stage2_ok else "no"))
    setup_bits.append("VCP-like (approximation): " + ("YES" if verdict.vcp_like else "no"))
    setup_bits.append("Pocket pivot: " + ("YES" if verdict.pocket_pivot else "no"))
    if verdict.rs_rank is not None:
        setup_bits.append(f"RS rank ~{verdict.rs_rank:.0f}/100 (SPY-relative)")

    return {
        "available": True,
        "symbol": symbol,
        "stage2_ok": bool(verdict.stage2_ok),
        "reasons": list(verdict.reasons),
        "reason_text": reason_text,
        "vcp_like": bool(verdict.vcp_like),
        "pocket_pivot": bool(verdict.pocket_pivot),
        "rs_rank": verdict.rs_rank,
        "pivot_low": verdict.pivot_low,
        "advisory_stop_price": advisory_stop,
        "setup_summary": " | ".join(setup_bits),
        "note": note,
    }


def print_minervini_view(view: Optional[Dict[str, Any]]) -> None:
    """Print the Minervini/SEPA coach view (ASCII, Windows-safe). No-op on None."""
    if not view:
        return
    print("\n── Minervini / SEPA setup (educational, no order) ────────────")
    print(f"  Symbol              : {view.get('symbol', '')}")
    if not view.get("available", False):
        print(f"  Setup               : unavailable ({view.get('reason_text', '')})")
        print("───────────────────────────────────────────────────────────────")
        return
    print(f"  Stage-2 uptrend     : {'YES' if view.get('stage2_ok') else 'NO'}")
    print(f"  Setup summary       : {view.get('setup_summary', '')}")
    print(f"  Why / Why not       : {view.get('reason_text', '')}")
    if view.get("advisory_stop_price") is not None:
        print(
            f"  Advisory 1R stop    : {_format_money(view.get('advisory_stop_price'))} "
            f"(below VCP-like pivot {_format_money(view.get('pivot_low'))}; explanation only)"
        )
    print(f"  Note                : {view.get('note', '')}")
    print("───────────────────────────────────────────────────────────────")


# ─── Lesson builder (educational summary, no order sizing) ──────────────────
def build_trade_lesson(
    signal,
    position_info: Optional[Dict[str, Any]] = None,
    open_order_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a beginner-friendly lesson dict for one signal.

    Args:
        signal:            A predictor.Signal object.
        position_info:     {symbol -> qty (float)} of current paper positions.
        open_order_info:   set of symbols with working orders.

    Returns:
        dict with keys: symbol, ticker_meaning, action, action_meaning,
        confidence, confidence_meaning, model_breakdown, price,
        is_candidate, candidate_reason, position_state, working_order,
        why_or_why_not, next_action, chart_check_required.
    """
    symbol = signal.symbol.upper().strip()
    position_info = position_info or {}
    open_order_info = open_order_info or set()

    action = signal.action.upper().strip()
    confidence = float(signal.confidence)
    rf = float(signal.rf_score)
    lstm = float(signal.lstm_score)
    tech = float(signal.tech_score)
    price = float(signal.price)

    min_conf = float(getattr(config, "COACH_MIN_CONFIDENCE_FOR_CANDIDATE", 0.65))
    in_position = float(position_info.get(symbol, 0.0)) > 0
    has_working = symbol in open_order_info

    is_candidate = (
        action == "BUY"
        and confidence >= min_conf
        and not in_position
        and not has_working
    )

    if confidence >= min_conf + 0.10:
        conf_label = "STRONG"
    elif confidence >= min_conf:
        conf_label = "BORDERLINE"
    else:
        conf_label = "WEAK"

    if action == "BUY":
        if is_candidate:
            why = (
                f"Ensemble crossed the BUY threshold ({min_conf:.2f}); this is a "
                f"trade CANDIDATE, not an automatic buy."
            )
        elif in_position:
            why = "Already long this symbol — adding more is blocked by the duplicate-position guard."
        elif has_working:
            why = "A working order is already pending for this symbol — coach skips duplicates."
        elif confidence < min_conf:
            why = f"Model confidence {confidence:.2f} is below the {min_conf:.2f} candidate floor."
        else:
            why = "Buy signal present but not enough confidence to be a candidate."
    elif action == "SELL":
        if in_position:
            why = "SELL closes an existing long. Coach flow stays long-only; this is informational."
        else:
            why = "SELL signal but no long position to close (long-only by default)."
    else:  # HOLD
        if rf < 0.45 and lstm < 0.45:
            why = "Model weak, chart weak — model and technical both below mid-range."
        else:
            why = "Confidence low — model and technical do not agree strongly enough."

    return {
        "symbol": symbol,
        "ticker_meaning": _ticker_guide(symbol),
        "action": action,
        "action_meaning": {
            "BUY":  "BUY = a candidate entry. NEVER an automatic buy. You must check the chart and confirm.",
            "SELL": "SELL = close an existing long position (long-only by default).",
            "HOLD": "HOLD = no action. The model does not see an edge right now.",
        }[action],
        "confidence": confidence,
        "confidence_meaning": (
            f"Confidence is the blended ensemble score in [0, 1]. "
            f"{confidence:.2f} is {conf_label} vs the {min_conf:.2f} BUY floor. "
            f"Higher = more sub-models agree on a near-term UP move."
        ),
        "model_breakdown": {
            "rf": rf,
            "lstm": lstm,
            "tech": tech,
            "rf_meaning": "Random Forest — supervised classifier on engineered features.",
            "lstm_meaning": "LSTM — sequence model that reads the last 30 bars of features.",
            "tech_meaning": "Technical — deterministic RSI / MACD / Bollinger / trend score.",
        },
        "price": price,
        "is_candidate": bool(is_candidate),
        "candidate_reason": why,
        "position_state": "in_position" if in_position else "flat",
        "working_order": bool(has_working),
        "why_or_why_not": why,
        "next_action": (
            "Check the chart manually for support/resistance, recent news, and the "
            "proposed stop. Then run `paper-coach` to see a preview, and only after "
            "that run `paper-coach --confirm --chart-checked` if you still want to trade."
        ),
        "chart_check_required": bool(getattr(config, "COACH_REQUIRE_CHART_CHECK", True)),
        # Read-only Minervini/SEPA setup explanation (None unless the overlay +
        # coach switches are both on; never affects trading).
        "minervini": build_minervini_view(signal),
    }


# ─── Preview builder (paper-trade math, still no order) ──────────────────────
def build_trade_preview(
    signal,
    cash: float,
    current_positions: Optional[Dict[str, Any]] = None,
    open_orders: Optional[List[Any]] = None,
    ohlcv=None,
) -> Dict[str, Any]:
    """Build a paper-trade preview for one signal. Never places an order.

    The math here mirrors IBKRBridge._calc_quantity and _initial_stop_price
    so the preview matches what the bridge would actually do, but no order
    is sent.
    """
    current_positions = current_positions or {}
    open_orders = open_orders or []

    symbol = signal.symbol.upper().strip()
    action = signal.action.upper().strip()
    confidence = float(signal.confidence)
    price = float(signal.price)

    min_conf = float(getattr(config, "COACH_MIN_CONFIDENCE_FOR_CANDIDATE", 0.65))
    in_position = float(current_positions.get(symbol, 0.0)) > 0

    # Refuse to preview a SELL when flat (long-only) and a HOLD when not actionable.
    if action == "HOLD":
        return {
            "symbol": symbol, "action": "HOLD", "tradeable": False,
            "skip_reason": "HOLD is not a trade. Nothing to preview.",
            "quantity": 0, "estimated_cost": 0.0, "price": price, "confidence": confidence,
        }

    if action == "SELL" and not in_position:
        return {
            "symbol": symbol, "action": "SELL", "tradeable": False,
            "skip_reason": "No long position to close. Coach is long-only by default.",
            "quantity": 0, "estimated_cost": 0.0, "price": price, "confidence": confidence,
        }

    if in_position:
        return {
            "symbol": symbol, "action": action, "tradeable": False,
            "skip_reason": f"Already long {current_positions.get(symbol, 0)} shares — coach skips duplicate entries.",
            "quantity": 0, "estimated_cost": 0.0, "price": price, "confidence": confidence,
        }

    if confidence < min_conf:
        return {
            "symbol": symbol, "action": action, "tradeable": False,
            "skip_reason": (
                f"Confidence {confidence:.2f} below candidate floor {min_conf:.2f}."
            ),
            "quantity": 0, "estimated_cost": 0.0, "price": price, "confidence": confidence,
        }

    # Mirror IBKRBridge._calc_quantity exactly (no actual order is placed).
    max_value = float(cash) * float(config.MAX_POSITION_PCT)
    cap = getattr(config, "MAX_TRADE_VALUE", None)
    if cap is not None:
        max_value = min(max_value, float(cap))
    qty = max(int(max_value / price), 0) if price > 0 else 0

    # Commission breakeven check (mirrors ibkr_bridge._breakeven_pct).
    _com_per_side = max(
        float(getattr(config, "MIN_COMMISSION_PER_TRADE", 1.00)),
        qty * float(getattr(config, "COMMISSION_PER_SHARE", 0.005)),
    ) if qty > 0 and price > 0 else 0.0
    _com_round_trip = 2.0 * _com_per_side
    breakeven_pct = round(_com_round_trip / (qty * price), 4) if qty > 0 and price > 0 else 0.0

    # Mirror IBKRBridge._initial_stop_price for BUY.
    stop_price = round(price * (1 - float(config.STOP_LOSS_PCT)), 2) if action == "BUY" else None
    est_cost = round(qty * price, 2)
    max_loss = round(qty * (price - stop_price), 2) if (action == "BUY" and stop_price and qty > 0) else None
    avg_cost = round(price, 2)  # first entry — average cost == entry price

    using_trailing = bool(getattr(config, "USE_TRAILING_EXIT", True))
    trail_pct = float(getattr(config, "TRAILING_STOP_PCT", 0.0))
    stop_explanation = (
        f"Initial protective stop at {_format_pct(config.STOP_LOSS_PCT)} below entry "
        f"({_format_money(stop_price)}). "
        + (
            f"After fill, a trailing stop of {_format_pct(trail_pct)} locks in profit as "
            f"price moves in your favor. Worst-case loss if the stop fires immediately: "
            f"{_format_money(max_loss)}."
            if using_trailing
            else f"A fixed take-profit at {_format_pct(config.TAKE_PROFIT_PCT)} is also set. "
                 f"Worst-case loss if the stop fires immediately: {_format_money(max_loss)}."
        )
    )
    # Append commission note to stop_explanation.
    if qty > 0 and breakeven_pct > 0:
        _breakeven_note = (
            f" Estimated round-trip commission: {_format_money(_com_round_trip)} "
            f"({_format_pct(breakeven_pct)}). "
            f"Price must move at least {_format_pct(breakeven_pct)} to break even."
        )
        stop_explanation += _breakeven_note

    formula_fields = _daytrade_formula_fields(signal, cash, ohlcv=ohlcv)

    return {
        "symbol": symbol,
        "action": action,
        "tradeable": True,
        "skip_reason": None,
        "quantity": qty,
        "estimated_cost": est_cost,
        "avg_cost_if_adding": avg_cost,
        "price": price,
        "stop_price": stop_price,
        "max_loss_if_stop_triggers": max_loss,
        "stop_explanation": stop_explanation,
        "trailing_exit": using_trailing,
        "trailing_pct": trail_pct,
        "take_profit_pct": float(getattr(config, "TAKE_PROFIT_PCT", 0.0)),
        "stop_loss_pct": float(getattr(config, "STOP_LOSS_PCT", 0.0)),
        "breakeven_pct": breakeven_pct,
        "est_commission_roundtrip": round(_com_round_trip, 2) if qty > 0 else 0.0,
        "position_cap_pct": float(getattr(config, "MAX_POSITION_PCT", 0.0)),
        "trade_cap_usd": float(getattr(config, "MAX_TRADE_VALUE", 0.0)) if getattr(config, "MAX_TRADE_VALUE", None) is not None else None,
        "confidence": confidence,
        "min_confidence_required": min_conf,
        **formula_fields,
        # Read-only Minervini/SEPA setup explanation for this entry candidate
        # (None unless the overlay + coach switches are both on; never sizes or
        # blocks the order).
        "minervini": build_minervini_view(signal),
    }


# ─── Candidate selection (top-N BUY signals above the floor) ──────────────────
def select_coach_candidates(signals, max_n: Optional[int] = None) -> List:
    """Return up to max_n BUY signals at or above the confidence floor.

    Falls back to the highest-confidence BUY if nothing crosses the floor,
    so the coach still has something to teach about.
    """
    max_n = int(max_n) if max_n is not None else int(getattr(config, "COACH_MAX_NEW_TRADES_PER_RUN", 1))
    min_conf = float(getattr(config, "COACH_MIN_CONFIDENCE_FOR_CANDIDATE", 0.65))

    buy_signals = [s for s in signals if s.action == "BUY"]
    buy_signals.sort(key=lambda s: float(s.confidence), reverse=True)

    eligible = [s for s in buy_signals if float(s.confidence) >= min_conf]
    if eligible:
        return eligible[:max_n]

    if buy_signals:
        return buy_signals[:max_n]
    return []


# ─── Chart-status gate (blocks extended / below-trend / thin / model-less) ───
# A candidate must clear a basic chart sanity check before any paper execution.
# These are intentionally simple, deterministic, and beginner-readable. They are
# NOT a replacement for the user's own manual chart check (--chart-checked); they
# are an automatic floor that rejects obviously poor entries.
CHART_OK                 = "OK"
CHART_TOO_EXTENDED       = "TOO_EXTENDED"
CHART_BELOW_TREND        = "BELOW_TREND"
CHART_LOW_VOLUME         = "LOW_VOLUME"
CHART_MODEL_MISSING      = "MODEL_MISSING"
CHART_DATA_UNAVAILABLE   = "DATA_UNAVAILABLE"

# Statuses that BLOCK paper execution in daily-coach.
CHART_BLOCKING_STATUSES = {
    CHART_TOO_EXTENDED,
    CHART_BELOW_TREND,
    CHART_LOW_VOLUME,
    CHART_MODEL_MISSING,
    CHART_DATA_UNAVAILABLE,
}

# Tunable chart-gate thresholds (kept local; they describe a *chart sanity*
# floor, not a trading strategy).
_CHART_RSI_OVERBOUGHT    = 75.0    # RSI above this = too extended    # RSI above this = too extended
_CHART_BB_OVERBOUGHT     = 0.95    # within top 5% of the Bollinger band = extended
_CHART_MAX_DIST_SMA20    = 0.12    # >12% above the 20-day SMA = stretched
_CHART_MIN_VOL_RATIO     = 0.7     # last volume must be at least 70% of 20d avg     # last volume must be at least 70% of 20d avg


def assess_chart_status(signal) -> Tuple[str, str]:
    """Classify a signal's chart into a status + human-readable detail.

    Returns one of:
        OK                — chart looks acceptable for a paper entry.
        TOO_EXTENDED      — overbought / stretched far above trend.
        BELOW_TREND       — price below both the 20- and 50-day SMAs.
        LOW_VOLUME        — last bar's volume is thin vs the 20-day average.
        MODEL_MISSING     — ensemble had no ML model (forced HOLD).
        DATA_UNAVAILABLE  — could not fetch/parse data for the symbol.

    Only OK clears the gate; every other status is blocking. The data fetch is
    cached by data_manager, so this is cheap to call right after prediction.
    """
    # The predictor forces HOLD with this phrase when no ML model is available.
    reason = str(getattr(signal, "reason", "") or "")
    if "ML models missing" in reason:
        return CHART_MODEL_MISSING, "No trained ML model for this symbol — ensemble forced HOLD."

    symbol = str(getattr(signal, "symbol", "")).upper().strip()
    try:
        from data_manager import fetch_ohlcv, build_features
        feat = build_features(fetch_ohlcv(symbol))
        if feat.empty:
            return CHART_DATA_UNAVAILABLE, "No usable feature rows for this symbol."
        row = feat.iloc[-1]
    except Exception:
        logger.debug("Chart status data fetch failed for %s", symbol, exc_info=True)
        return CHART_DATA_UNAVAILABLE, "Could not fetch price data for the chart check."

    rsi = float(row.get("rsi", 50.0))
    bb_pct = float(row.get("bb_pct", 0.5))
    dist_sma20 = float(row.get("dist_sma20", 0.0))
    dist_sma50 = float(row.get("dist_sma50", 0.0))
    vol_ratio = float(row.get("vol_ratio", 1.0))

    # Below trend: price under both moving averages — not a healthy long entry.
    if dist_sma20 < 0 and dist_sma50 < 0:
        return (
            CHART_BELOW_TREND,
            f"Price is below both SMA20 ({dist_sma20*100:.1f}%) and SMA50 "
            f"({dist_sma50*100:.1f}%) — not in an uptrend.",
        )

    # Too extended: overbought RSI, pinned to the upper band, or stretched above SMA20.
    if rsi >= _CHART_RSI_OVERBOUGHT:
        return CHART_TOO_EXTENDED, f"RSI {rsi:.0f} is overbought (≥ {_CHART_RSI_OVERBOUGHT:.0f})."
    if bb_pct >= _CHART_BB_OVERBOUGHT:
        return CHART_TOO_EXTENDED, f"Price is pinned to the upper Bollinger band (bb_pct {bb_pct:.2f})."
    if dist_sma20 >= _CHART_MAX_DIST_SMA20:
        return (
            CHART_TOO_EXTENDED,
            f"Price is {dist_sma20*100:.1f}% above SMA20 (≥ {_CHART_MAX_DIST_SMA20*100:.0f}%) — stretched.",
        )

    # Low volume: thin participation vs the 20-day average.
    if vol_ratio < _CHART_MIN_VOL_RATIO:
        return (
            CHART_LOW_VOLUME,
            f"Last volume is {vol_ratio:.2f}× the 20-day average (< {_CHART_MIN_VOL_RATIO:.2f}) — thin.",
        )

    return (
        CHART_OK,
        f"RSI {rsi:.0f}, bb_pct {bb_pct:.2f}, {dist_sma20*100:+.1f}% vs SMA20, "
        f"vol {vol_ratio:.2f}× avg — acceptable for a paper entry.",
    )


# ─── Daily-coach per-candidate evaluation (static gates only) ────────────────
def evaluate_daily_candidate(
    signal,
    preview: Dict[str, Any],
    chart_status: str,
    chart_detail: str,
    positions: Optional[Dict[str, Any]] = None,
    working: Optional[set] = None,
) -> Dict[str, Any]:
    """Decide whether ONE candidate clears the per-symbol gates for paper trading.

    This evaluates only the *static* gates that depend on the candidate itself:
        * action must be BUY
        * confidence >= config.BUY_THRESHOLD
        * no existing position in the symbol
        * no working order in the symbol
        * chart status must not be blocking
        * preview must be tradeable (qty > 0, etc.)

    The *dynamic* run caps (MAX_OPEN_POSITIONS, MAX_DAILY_TRADES, --max-trades)
    are enforced by the caller as it iterates, because they depend on how many
    orders have already been placed this run.

    Returns a display-ready dict; ``accepted`` is True only when every static
    gate passes. ``skip_reason`` is None when accepted, else a short string.
    """
    positions = positions or {}
    working = working or set()

    symbol = str(getattr(signal, "symbol", "")).upper().strip()
    action = str(getattr(signal, "action", "")).upper().strip()
    confidence = float(getattr(signal, "confidence", 0.0))
    buy_threshold = float(getattr(config, "BUY_THRESHOLD", 0.65))

    why_selected = (
        f"Ranked among the top BUY candidates from the full-market scan; "
        f"ensemble confidence {confidence:.2f}."
    )

    reasons: List[str] = []
    if action != "BUY":
        reasons.append(f"action is {action}, not BUY")
    if confidence < buy_threshold:
        reasons.append(f"confidence {confidence:.2f} < BUY_THRESHOLD {buy_threshold:.2f}")
    if symbol in positions:
        reasons.append("already holding this symbol")
    if symbol in working:
        reasons.append("a working order already exists for this symbol")
    if chart_status in CHART_BLOCKING_STATUSES:
        reasons.append(f"chart status {chart_status}: {chart_detail}")
    reasons.extend(daytrade_refusal_reasons(signal, preview, positions=positions, working=working))

    deduped_reasons: List[str] = []
    for reason in reasons:
        if reason and reason not in deduped_reasons:
            deduped_reasons.append(reason)

    accepted = not deduped_reasons
    return {
        "symbol": symbol,
        "action": action,
        "confidence": confidence,
        "chart_status": chart_status,
        "chart_detail": chart_detail,
        "why_selected": why_selected,
        "accepted": accepted,
        "skip_reason": None if accepted else "; ".join(deduped_reasons),
        "quantity": preview.get("quantity", 0),
        "estimated_cost": preview.get("estimated_cost", 0.0),
        "stop_price": preview.get("stop_price"),
        "stop_explanation": preview.get("stop_explanation", ""),
        "trailing_exit": preview.get("trailing_exit", False),
        "trailing_pct": preview.get("trailing_pct", 0.0),
        "max_loss_if_stop_triggers": preview.get("max_loss_if_stop_triggers"),
        "price": preview.get("price", float(getattr(signal, "price", 0.0))),
        "planned_entry": preview.get("planned_entry"),
        "suggested_stop": preview.get("suggested_stop"),
        "suggested_target_2r": preview.get("suggested_target_2r"),
        "suggested_target_3r": preview.get("suggested_target_3r"),
        "risk_per_share": preview.get("risk_per_share"),
        "reward_to_2r": preview.get("reward_to_2r"),
        "rr_2r": preview.get("rr_2r"),
        "suggested_shares_by_risk": preview.get("suggested_shares_by_risk", 0),
        "planned_risk_dollars": preview.get("planned_risk_dollars"),
        "previous_day_pivot_levels": preview.get("previous_day_pivot_levels"),
        "gap_pct": preview.get("gap_pct"),
        "atr_dollars": preview.get("atr_dollars"),
        "daytrade_formula_note": preview.get("daytrade_formula_note"),
    }


# ─── Daily-coach beginner summary (Windows-safe ASCII only) ──────────────────
# These helpers build a short, plain-text summary for the `daily-coach` command.
# They are display/report only — no trading logic, no order placement, no config
# changes. All output is ASCII so it renders cleanly on a Windows console and in
# the markdown report (no emoji / box-drawing characters that turn into mojibake).
def _daily_friendly_reason(signal, min_conf: float) -> str:
    """One short, beginner-readable reason string for a single analyzed signal."""
    reason = str(getattr(signal, "reason", "") or "")
    if "ML models missing" in reason:
        return "Model missing for this symbol"
    action = str(getattr(signal, "action", "")).upper().strip()
    conf = float(getattr(signal, "confidence", 0.0))
    if action == "BUY" and conf >= float(min_conf):
        return f"BUY candidate (confidence {conf:.2f} >= {float(min_conf):.2f})"
    if conf < 0.50:
        return "Weak model score"
    if conf < float(min_conf):
        return f"Model exists but confidence below {float(min_conf):.2f}"
    return "Confidence not strong enough for a BUY"


def build_daily_coach_summary(
    scanned,
    analyzed_signals,
    eligible_buys,
    min_conf: float,
    top_n: int = 5,
    chart_risk_failed: int = 0,
) -> List[str]:
    """Build the beginner-friendly daily-coach summary as a list of ASCII lines.

    This is pure formatting: it summarizes what the scan/prediction already
    produced. It never changes trading logic, never connects to IBKR, and never
    places an order.
    """
    analyzed = list(analyzed_signals or [])
    buys = list(eligible_buys or [])
    n_analyzed = len(analyzed)
    n_buy = len(buys)

    model_missing = sum(
        1 for s in analyzed
        if "ML models missing" in str(getattr(s, "reason", "") or "")
    )
    chart_risk = max(int(chart_risk_failed), 0)
    # Everything analyzed that is neither model-missing nor an eligible BUY is
    # bucketed as "confidence below threshold".
    conf_below = max(n_analyzed - model_missing - n_buy, 0)

    lines: List[str] = []
    lines.append("Daily Trading Coach")
    lines.append(f"Scanned market: {int(scanned)} symbols")
    lines.append(f"Candidates analyzed: {n_analyzed}")
    lines.append(f"BUY candidates: {n_buy}")
    decision = "No trade today" if n_buy == 0 else f"{n_buy} BUY candidate(s) to review"
    lines.append(f"Decision: {decision}")
    lines.append("")
    lines.append("Top watchlist:")
    lines.append("Symbol | Action | Confidence | Reason")
    if analyzed:
        for s in analyzed[: max(0, int(top_n))]:
            sym = str(getattr(s, "symbol", "")).upper().strip()
            act = str(getattr(s, "action", "")).upper().strip()
            conf = float(getattr(s, "confidence", 0.0))
            lines.append(f"{sym} | {act} | {conf:.2f} | {_daily_friendly_reason(s, min_conf)}")
    else:
        lines.append("(no symbols were analyzed)")
    lines.append("")
    lines.append("Skipped reason summary:")
    lines.append(f"- Model missing: {model_missing} symbols")
    lines.append(f"- Confidence below threshold: {conf_below} symbols")
    lines.append(f"- Chart/risk filter failed: {chart_risk} symbols")
    lines.append("")
    lines.append("Lesson:")
    lines.append("No trade is also a valid trading decision.")
    lines.append("Do not force a trade when confidence is low.")
    return lines


def build_ipo_watch_section(price_lookup=None, history_lookup=None) -> List[str]:
    """Build the watch-only "IPO / New Listing Watch" section.

    Pure formatting + read-only data lookups. This NEVER trains, NEVER trades,
    NEVER places an order, and NEVER promotes a symbol to an official BUY
    candidate. Symbols come from config.IPO_WATCH_SYMBOLS and are always labelled
    WATCH_ONLY_NEW_LISTING.

    Args:
        price_lookup:   optional callable(symbol) -> float|None for last price.
        history_lookup: optional callable(symbol) -> int|None for number of
                        available history days (used to flag insufficient model
                        history vs config.MIN_HISTORY_DAYS).

    The section is ALWAYS rendered (even with no data) so the user consistently
    sees that these symbols are watch-only.
    """
    symbols = list(getattr(config, "IPO_WATCH_SYMBOLS", []) or [])
    min_history = int(getattr(config, "MIN_HISTORY_DAYS", 252))

    lines: List[str] = []
    lines.append("IPO / New Listing Watch")
    lines.append("These symbols are WATCH_ONLY_NEW_LISTING: shown for visibility")
    lines.append("only. They are never auto-trained, auto-traded, or ordered, and")
    lines.append("only become a BUY candidate if they pass the normal model/risk rules.")
    if not symbols:
        lines.append("(no IPO / new listing symbols configured)")
        return lines

    for sym in symbols:
        sym = str(sym).upper().strip()
        if not sym:
            continue

        price = None
        if callable(price_lookup):
            try:
                price = price_lookup(sym)
            except Exception:  # read-only display must never break the coach
                price = None

        history_days = None
        if callable(history_lookup):
            try:
                history_days = history_lookup(sym)
            except Exception:
                history_days = None

        price_str = _format_money(price) if price is not None else "n/a (no data)"
        lines.append("")
        lines.append(f"{sym} | WATCH_ONLY_NEW_LISTING | Price: {price_str}")

        if history_days is not None and history_days < min_history:
            lines.append(
                f"{sym} is a new listing. Model history is insufficient, "
                "so this is watch-only."
            )

    return lines


def print_daily_coach_summary(lines: List[str]) -> None:
    """Print the daily-coach summary lines (ASCII, Windows-safe)."""
    print()
    for ln in lines:
        print(ln)


def write_daily_coach_summary(lines: List[str], path=None) -> str:
    """Write the same clean summary to reports/daily_coach_report.md (overwrite)."""
    path = path or (config.REPORTS_DIR / "daily_coach_report.md")
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    body = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Daily Trading Coach Report\n\n")
        f.write(f"_Generated {ts}_\n\n")
        f.write("```\n")
        f.write(body)
        f.write("\n```\n")
    return str(path)


def print_daily_candidate(ev: Dict[str, Any], index: Optional[int] = None) -> None:
    """Print one daily-coach candidate with every required field (ASCII only)."""
    header = "-- Daily Coach Candidate"
    if index is not None:
        header += f" #{index}"
    print(f"\n{header} " + "-" * 34)
    print(f"  Symbol              : {ev['symbol']}")
    print(f"  Bot action          : {ev['action']}")
    print(f"  Confidence          : {ev['confidence']:.2f}")
    print(f"  Chart status        : {ev['chart_status']}  ({ev['chart_detail']})")
    print(f"  Why selected        : {ev['why_selected']}")
    verdict = "ACCEPTED for paper trade" if ev["accepted"] else f"SKIPPED - {ev['skip_reason']}"
    print(f"  Accepted / skipped  : {verdict}")
    print(f"  Estimated quantity  : {ev['quantity']} shares")
    print(f"  Estimated cost      : {_format_money(ev['estimated_cost'])}")
    if ev["trailing_exit"]:
        print(
            f"  Stop / trailing stop: initial {_format_money(ev['stop_price'])}, "
            f"then {_format_pct(ev['trailing_pct'])} trailing stop"
        )
    else:
        print(f"  Stop / trailing stop: {_format_money(ev['stop_price'])} (fixed protective stop)")
    print(f"  Est. possible loss  : {_format_money(ev['max_loss_if_stop_triggers'])}")
    print(f"  Formula entry       : {_format_money(ev.get('planned_entry'))}")
    print(f"  Formula stop        : {_format_money(ev.get('suggested_stop'))}")
    print(f"  Formula 2R / 3R     : {_format_money(ev.get('suggested_target_2r'))} / {_format_money(ev.get('suggested_target_3r'))}")
    print(f"  Formula risk/share  : {_format_money(ev.get('risk_per_share'))}")
    rr_2r = ev.get("rr_2r")
    rr_text = "n/a" if rr_2r is None else f"{float(rr_2r):.2f}"
    print(f"  Formula R:R to 2R   : {rr_text}")
    print(f"  Formula risk shares : {ev.get('suggested_shares_by_risk', 0)} shares")
    print(f"  Formula planned risk: {_format_money(ev.get('planned_risk_dollars'))}")
    if ev.get("atr_dollars") is not None:
        print(f"  Daily ATR context   : {_format_money(ev.get('atr_dollars'))}")
    if ev.get("gap_pct") is not None:
        print(f"  Daily gap context   : {_format_pct(ev.get('gap_pct'))}")
    pivots = ev.get("previous_day_pivot_levels")
    if pivots:
        print(
            "  Previous pivots     : "
            f"PP {_format_money(pivots.get('pp'))}, "
            f"R1 {_format_money(pivots.get('r1'))}, "
            f"S1 {_format_money(pivots.get('s1'))}"
        )
    if ev.get("daytrade_formula_note"):
        print(f"  Formula note        : {ev['daytrade_formula_note']}")
    print("-" * 63)


# ─── Printing helpers ────────────────────────────────────────────────────────
def print_trade_lesson(lesson: Dict[str, Any]) -> None:
    print("\n── Trade Coach Lesson ─────────────────────────────────────────")
    print(f"  Ticker         : {lesson['symbol']}")
    print(f"  What it is     : {lesson['ticker_meaning']}")
    print(f"  Action         : {lesson['action']}")
    print(f"  Action meaning : {lesson['action_meaning']}")
    print(f"  Confidence     : {lesson['confidence']:.2f}")
    print(f"  Confidence note: {lesson['confidence_meaning']}")
    mb = lesson["model_breakdown"]
    print(f"  RF score       : {mb['rf']:.2f}  (Random Forest — engineered features)")
    print(f"  LSTM score     : {mb['lstm']:.2f}  (sequence model on last 30 bars)")
    print(f"  Tech score     : {mb['tech']:.2f}  (RSI / MACD / Bollinger / trend)")
    print(f"  Price (last)   : {_format_money(lesson['price'])}")
    print(f"  Position state : {lesson['position_state']}")
    print(f"  Working order  : {'yes' if lesson['working_order'] else 'no'}")
    print(f"  Trade candidate: {'YES' if lesson['is_candidate'] else 'NO'}")
    print(f"  Why / Why not  : {lesson['candidate_reason']}")
    print(f"  Next action    : {lesson['next_action']}")
    if lesson["chart_check_required"]:
        print("  ⚠ Chart check is REQUIRED before any execution.")
    _be_pct = lesson.get("breakeven_pct", 0.0)
    if _be_pct > 0:
        _be_str = _format_pct(_be_pct)
        print(f"  Commission cost  : {_format_money(lesson.get('est_commission_roundtrip', 0.0))} round-trip")
        print(f"  Breakeven        : price must move {_be_str} to cover commission")
        if lesson.get("trailing_pct", 0) and _be_pct > lesson.get("trailing_pct", 0):
            _gap = _format_pct(lesson["trailing_pct"])
            print(f"  ⚠ Breakeven ({_be_str}) exceeds trailing stop ({_gap}) — commission may not be covered!")
    print("───────────────────────────────────────────────────────────────")
    print_minervini_view(lesson.get("minervini"))


def print_trade_preview(preview: Dict[str, Any]) -> None:
    print("\n── Trade Coach Preview (NO order placed) ─────────────────────")
    print(f"  Symbol              : {preview['symbol']}")
    print(f"  Action              : {preview['action']}")
    if not preview.get("tradeable", True):
        print(f"  Skip reason         : {preview.get('skip_reason', 'not tradeable')}")
        print("───────────────────────────────────────────────────────────────")
        return
    print(f"  Quantity (proposed) : {preview['quantity']} shares")
    print(f"  Estimated cost      : {_format_money(preview['estimated_cost'])}")
    print(f"  Avg cost if adding  : {_format_money(preview['avg_cost_if_adding'])}")
    print(f"  Stop price          : {_format_money(preview['stop_price'])}")
    print(f"  Max loss if stop    : {_format_money(preview['max_loss_if_stop_triggers'])}")
    print(f"  Stop explanation    : {preview['stop_explanation']}")
    print(f"  Cash % cap          : {_format_pct(preview['position_cap_pct'])}")
    if preview.get("trade_cap_usd") is not None:
        print(f"  Trade $ cap         : {_format_money(preview['trade_cap_usd'])}")
    print("\n  Paper day-trading formula preview:")
    print(f"    Planned entry      : {_format_money(preview.get('planned_entry'))}")
    print(f"    Suggested stop     : {_format_money(preview.get('suggested_stop'))}")
    print(f"    Target 2R / 3R     : {_format_money(preview.get('suggested_target_2r'))} / {_format_money(preview.get('suggested_target_3r'))}")
    print(f"    Risk/share         : {_format_money(preview.get('risk_per_share'))}")
    print(f"    Reward to 2R       : {_format_money(preview.get('reward_to_2r'))}")
    rr_2r = preview.get("rr_2r")
    rr_text = "n/a" if rr_2r is None else f"{float(rr_2r):.2f}"
    print(f"    R:R to 2R          : {rr_text}")
    print(f"    Shares by 0.1% risk: {preview.get('suggested_shares_by_risk', 0)}")
    print(f"    Planned risk       : {_format_money(preview.get('planned_risk_dollars'))}")
    if preview.get("atr_dollars") is not None:
        print(f"    Daily ATR          : {_format_money(preview.get('atr_dollars'))}")
    if preview.get("gap_pct") is not None:
        print(f"    Daily gap          : {_format_pct(preview.get('gap_pct'))}")
    pivots = preview.get("previous_day_pivot_levels")
    if pivots:
        print(
            "    Previous pivots    : "
            f"PP {_format_money(pivots.get('pp'))}, "
            f"R1 {_format_money(pivots.get('r1'))}, "
            f"S1 {_format_money(pivots.get('s1'))}"
        )
    print(f"    Note               : {preview.get('daytrade_formula_note')}")
    print("───────────────────────────────────────────────────────────────")
    print_minervini_view(preview.get("minervini"))


# ─── Markdown report writer ──────────────────────────────────────────────────
def write_trade_note(
    lesson: Dict[str, Any],
    action_taken: str,
    preview: Optional[Dict[str, Any]] = None,
    order_result: Optional[Dict[str, Any]] = None,
    path=None,
) -> str:
    """Append a markdown note to the coach report.

    action_taken: one of {"preview_only", "paper_order_placed", "refused"}.
    Returns the path written to.
    """
    path = path or getattr(config, "COACH_REPORT_FILE", None) or (
        config.REPORTS_DIR / "trade_coach_report.md"
    )
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append(f"\n## {ts} — Trade Coach Note")
    lines.append("")
    lines.append(f"- **Symbol**         : {lesson['symbol']}")
    lines.append(f"- **Ticker meaning** : {lesson['ticker_meaning']}")
    lines.append(f"- **Action**         : {lesson['action']}")
    lines.append(f"- **Confidence**     : {lesson['confidence']:.2f}  "
                 f"(floor {float(getattr(config, 'COACH_MIN_CONFIDENCE_FOR_CANDIDATE', 0.65)):.2f})")
    mb = lesson["model_breakdown"]
    lines.append(f"- **RF / LSTM / Tech**: {mb['rf']:.2f} / {mb['lstm']:.2f} / {mb['tech']:.2f}")
    lines.append(f"- **Price (last)**   : {_format_money(lesson['price'])}")
    lines.append(f"- **Position state** : {lesson['position_state']}")
    lines.append(f"- **Working order**  : {'yes' if lesson['working_order'] else 'no'}")
    lines.append(f"- **Trade candidate**: {'YES' if lesson['is_candidate'] else 'NO'}")
    lines.append(f"- **Why / Why not**  : {lesson['candidate_reason']}")
    lines.append("")
    if preview is not None and preview.get("tradeable"):
        lines.append("### Proposed trade (preview math)")
        lines.append(f"- **Quantity**         : {preview['quantity']} shares")
        lines.append(f"- **Estimated cost**   : {_format_money(preview['estimated_cost'])}")
        lines.append(f"- **Avg cost (entry)** : {_format_money(preview['avg_cost_if_adding'])}")
        lines.append(f"- **Stop price**       : {_format_money(preview['stop_price'])}")
        lines.append(f"- **Max loss @ stop**  : {_format_money(preview['max_loss_if_stop_triggers'])}")
        lines.append(f"- **Stop explanation** : {preview['stop_explanation']}")
        lines.append("")
        lines.append("### Paper day-trading formula preview")
        lines.append(f"- **Planned entry**       : {_format_money(preview.get('planned_entry'))}")
        lines.append(f"- **Suggested stop**      : {_format_money(preview.get('suggested_stop'))}")
        lines.append(f"- **Suggested target 2R** : {_format_money(preview.get('suggested_target_2r'))}")
        lines.append(f"- **Suggested target 3R** : {_format_money(preview.get('suggested_target_3r'))}")
        lines.append(f"- **Risk/share**          : {_format_money(preview.get('risk_per_share'))}")
        lines.append(f"- **Reward to 2R**        : {_format_money(preview.get('reward_to_2r'))}")
        rr_2r = preview.get("rr_2r")
        lines.append(f"- **R:R to 2R**           : {'n/a' if rr_2r is None else f'{float(rr_2r):.2f}'}")
        lines.append(f"- **Shares by 0.1% risk** : {preview.get('suggested_shares_by_risk', 0)}")
        lines.append(f"- **Planned risk dollars**: {_format_money(preview.get('planned_risk_dollars'))}")
        lines.append(f"- **ATR dollars**         : {_format_money(preview.get('atr_dollars'))}")
        lines.append(f"- **Gap pct**             : {_format_pct(preview.get('gap_pct'))}")
        lines.append(f"- **Formula note**        : {preview.get('daytrade_formula_note')}")
        lines.append("")
    lines.append("### Action taken")
    if action_taken == "paper_order_placed" and order_result is not None:
        lines.append(f"- **Action** : PAPER order placed through existing IBKRBridge.")
        lines.append(f"- **Result** : {order_result}")
    elif action_taken == "refused":
        lines.append(f"- **Action** : Refused. Reason: {order_result.get('reason') if isinstance(order_result, dict) else order_result}")
    else:
        lines.append("- **Action** : Preview only — no order placed.")
    lines.append("")
    lines.append("### Chart check reminder")
    lines.append("- ⚠ Always check the chart (support / resistance / news) before any execution.")
    lines.append("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Trade coach note written → %s", path)
    return str(path)
