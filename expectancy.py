"""expectancy.py - M4-core: offline, read-only expectancy / R-multiple report.

PURE + READ-ONLY. This module never connects to IBKR, never places orders,
never imports the order path (ibkr_bridge / order_exec / order_audit /
risk_state / account_guard), and never alters trading behavior. Its ONLY side
effects are writing report artifacts (JSON + Markdown) under reports/.

Source (M4-core, "A"): the walk-forward backtest ledger
``reports/backtest_trades.csv`` - a PER-BAR simulation ledger in RETURN units
(``net_pnl`` is a position-signed fractional return; there is no share quantity,
no dollar notional, and no Minervini 1R stop). Closed trades are reconstructed
as LONG round-trips (flat -> long -> flat) using the SAME entry/exit pairing +
equity-ratio return convention as ``main.cmd_backtest_summary``, grouped by
``(symbol, fold_start)`` so the numbers reconcile with the existing summary.

Because the backtest ledger carries NO true Minervini initial risk, R-multiple
and ``expectancy_R`` are NOT computed by default: every trade is tagged
``risk_source="unavailable"`` and excluded from the R metrics (with a reason).
PnL / win-rate / profit-factor are still reported. An OPTIONAL proxy-risk mode
(default OFF) computes a clearly-labeled hard-stop *proxy* R - never presented
as true Minervini R.

DEFERRED (NOT in M4-core): live/paper audit-journal reconstruction, dollar-
denominated risk, partial-fill handling, and short round-trips. These are noted
explicitly in the report rather than silently approximated.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional

import config
import coach_i18n  # M5A: DISPLAY-ONLY i18n (config + stdlib only; no order path)

logger = logging.getLogger(__name__)

# ── Risk sources ─────────────────────────────────────────────────────────────
RISK_SOURCE_UNAVAILABLE = "unavailable"   # no real initial risk -> excluded from R
RISK_SOURCE_HARD_STOP = "hard_stop"       # PROXY only (never true Minervini R)

# ── Exclusion reason codes (PnL-level) - never drop a trade silently ─────────
REASON_OPEN_AT_PERIOD_END = "open_at_period_end"   # entry with no matching exit
REASON_ORPHAN_EXIT = "orphan_exit"                 # exit with no matching entry
REASON_SHORT_UNSUPPORTED = "short_unsupported"     # short open (M4-core is long-only)
REASON_UNMODELED_TRANSITION = "unmodeled_transition"  # any other executed move
REASON_MISSING_ENTRY_PRICE = "missing_entry_price"
REASON_MISSING_EXIT_PRICE = "missing_exit_price"

# ── Exclusion reason codes (R-level) ─────────────────────────────────────────
R_REASON_UNAVAILABLE_RISK = "unavailable_risk"  # no initial risk persisted (default)
R_REASON_INVALID_RISK = "invalid_risk"          # proxy requested but risk <= 0 / non-finite
R_REASON_NO_PNL = "no_pnl"                       # trade itself is excluded from PnL


# ── Pure helpers ─────────────────────────────────────────────────────────────
def _f(x, default=None):
    """Parse a finite float, else ``default``. Never raises."""
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _int(x, default=None):
    """Parse an int (via float), else ``default``. Never raises."""
    try:
        if x is None or x == "":
            return default
        return int(float(x))
    except (TypeError, ValueError):
        return default


def _is_true(x) -> bool:
    return str(x).strip().lower() == "true"


def _round(x, n: int = 8):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def r_multiple(realized_pnl, initial_risk) -> Optional[float]:
    """Pure R-multiple = realized_pnl / initial_risk.

    Returns ``None`` when risk is missing, zero, negative, or non-finite, or when
    realized_pnl is missing - so invalid risk can never silently inflate R.
    """
    if realized_pnl is None or initial_risk is None:
        return None
    try:
        risk = float(initial_risk)
        pnl = float(realized_pnl)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(risk) or risk <= 0 or not math.isfinite(pnl):
        return None
    val = pnl / risk
    return val if math.isfinite(val) else None


# ── Closed-trade record ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class ClosedTrade:
    """One reconstructed round-trip (or an excluded candidate).

    ``realized_pnl`` and ``initial_risk`` are in RETURN units for the backtest
    source (per-unit-notional). ``pnl_included`` gates the PnL aggregates;
    ``r_included`` gates the R aggregates.
    """
    symbol: str
    fold_start: str
    side: str
    entry_date: Optional[str] = None
    exit_date: Optional[str] = None
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None
    initial_risk: Optional[float] = None
    r_multiple: Optional[float] = None
    risk_source: str = RISK_SOURCE_UNAVAILABLE
    pnl_included: bool = False
    r_included: bool = False
    exclusion_reason: str = ""
    r_exclusion_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "fold_start": self.fold_start,
            "side": self.side,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "realized_pnl": _round(self.realized_pnl) if self.realized_pnl is not None else None,
            "initial_risk": self.initial_risk,
            "r_multiple": _round(self.r_multiple, 6) if self.r_multiple is not None else None,
            "risk_source": self.risk_source,
            "pnl_included": self.pnl_included,
            "r_included": self.r_included,
            "exclusion_reason": self.exclusion_reason,
            "r_exclusion_reason": self.r_exclusion_reason,
        }


# ── Step 1: load the ledger (read-only) ──────────────────────────────────────
def load_backtest_rows(path) -> List[dict]:
    """Read ``backtest_trades.csv`` rows. Returns [] if missing/empty. Never raises."""
    try:
        p = Path(path)
        if not p.exists():
            return []
        text = p.read_text(encoding="utf-8")
        if not text.strip():
            return []
        return list(csv.DictReader(io.StringIO(text)))
    except Exception:  # pragma: no cover - defensive read-only
        logger.warning("expectancy.load_backtest_rows failed for %s", path, exc_info=True)
        return []


# ── Step 2: reconstruct closed trades (long round-trips) ─────────────────────
def _build_long_trade(symbol: str, fold: str, entry: dict, exit_: dict) -> ClosedTrade:
    """Pair one long entry with one long exit; realized PnL = equity-ratio return.

    Mirrors ``main.cmd_backtest_summary``: ret = exit_equity / entry_equity - 1.
    """
    entry_eq = _f(entry.get("equity"))
    exit_eq = _f(exit_.get("equity"))
    entry_price = _f(entry.get("close_price")) or _f(entry.get("entry_price"))
    exit_price = _f(exit_.get("close_price")) or _f(exit_.get("exit_price"))
    entry_date = (str(entry.get("date", "")).strip() or None)
    exit_date = (str(exit_.get("date", "")).strip() or None)

    if entry_eq is None or entry_eq == 0:
        return ClosedTrade(symbol, fold, "long", entry_date, exit_date,
                           entry_price, exit_price,
                           exclusion_reason=REASON_MISSING_ENTRY_PRICE,
                           r_exclusion_reason=R_REASON_NO_PNL)
    if exit_eq is None:
        return ClosedTrade(symbol, fold, "long", entry_date, exit_date,
                           entry_price, exit_price,
                           exclusion_reason=REASON_MISSING_EXIT_PRICE,
                           r_exclusion_reason=R_REASON_NO_PNL)

    realized = (exit_eq / entry_eq) - 1.0

    initial_stop = _f(entry.get("initial_stop_price"))
    initial_risk = None
    risk_source = RISK_SOURCE_UNAVAILABLE
    r_inc = False
    r_reason = R_REASON_UNAVAILABLE_RISK
    r_mult = None

    if initial_stop is not None and entry_price is not None and entry_price > 0 and initial_stop < entry_price:
        initial_risk = (entry_price - initial_stop) / entry_price
        risk_source = "initial_stop"
        r_mult = r_multiple(realized, initial_risk)
        if r_mult is not None:
            r_inc = True
            r_reason = ""

    return ClosedTrade(symbol, fold, "long", entry_date, exit_date,
                       entry_price, exit_price,
                       realized_pnl=realized, initial_risk=initial_risk,
                       r_multiple=r_mult,
                       pnl_included=True,
                       risk_source=risk_source, r_included=r_inc,
                       r_exclusion_reason=r_reason)


def _excluded_trade(symbol: str, fold: str, row: dict, side: str, reason: str) -> ClosedTrade:
    return ClosedTrade(symbol, fold, side,
                       entry_date=(str(row.get("date", "")).strip() or None),
                       entry_price=_f(row.get("close_price")),
                       exclusion_reason=reason,
                       r_exclusion_reason=R_REASON_NO_PNL)


def reconstruct_closed_trades(rows: List[dict]) -> List[ClosedTrade]:
    trades: List[ClosedTrade] = []
    groups: Dict[Tuple[str, str], List[dict]] = {}

    for r in rows:
        sym = str(r.get("symbol", "")).upper().strip()
        fold = str(r.get("fold_start", ""))
        key = (sym, fold)
        groups.setdefault(key, []).append(r)

    for (symbol, fold), group_rows in groups.items():
        pending_entry: Optional[dict] = None
        for r in group_rows:
            if not _is_true(r.get("order_executed")):
                continue
            old_pos = _int(r.get("old_position"), 0)
            new_pos = _int(r.get("new_position"), 0)

            # Open a long position (flat -> long)
            if old_pos == 0 and new_pos > 0:
                if pending_entry is not None:
                    trades.append(_excluded_trade(symbol, fold, pending_entry, "long",
                                                  REASON_OPEN_AT_PERIOD_END))
                pending_entry = r
                continue

            # Close a long position (long -> flat)
            if old_pos > 0 and new_pos == 0:
                if pending_entry is None:
                    trades.append(_excluded_trade(symbol, fold, r, "long",
                                                  REASON_ORPHAN_EXIT))
                else:
                    trades.append(_build_long_trade(symbol, fold, pending_entry, r))
                    pending_entry = None
                continue

            # Open a short position (flat -> short)
            if old_pos == 0 and new_pos < 0:
                trades.append(_excluded_trade(symbol, fold, r, "short",
                                              REASON_SHORT_UNSUPPORTED))
                continue

            # Any other transition while holding or scaling
            trades.append(_excluded_trade(symbol, fold, r, "unknown",
                                          REASON_UNMODELED_TRANSITION))

        if pending_entry is not None:
            trades.append(_excluded_trade(symbol, fold, pending_entry, "long",
                                          REASON_OPEN_AT_PERIOD_END))

    return trades


# ── Step 3: attach the risk model ────────────────────────────────────────────
RISK_SOURCE_ATR = "atr"
def apply_risk_model(trades: List[ClosedTrade], *, enable_proxy_risk: bool = False,
                     proxy_pct: Optional[float] = None,
                     proxy_mode: str = "hard_stop") -> List[ClosedTrade]:
    """Assign ``risk_source`` / R-multiple to each PnL-included trade.

    Default: if true initial risk exists (from initial_stop_price), use it.
    If unavailable and proxy on: a fixed hard-stop or ATR fraction is used as initial risk.
    """
    out: List[ClosedTrade] = []
    risk_src = RISK_SOURCE_ATR if (enable_proxy_risk and proxy_mode == "atr") else (
        RISK_SOURCE_HARD_STOP if enable_proxy_risk else RISK_SOURCE_UNAVAILABLE
    )
    for t in trades:
        if not t.pnl_included:
            out.append(t)
            continue

        # Preserve true initial risk if reconstructed during trade building
        if t.r_included and t.initial_risk is not None:
            out.append(t)
            continue

        if enable_proxy_risk:
            rv = r_multiple(t.realized_pnl, proxy_pct)
            if rv is not None:
                out.append(replace(t, risk_source=risk_src,
                                   initial_risk=float(proxy_pct), r_multiple=rv,
                                   r_included=True, r_exclusion_reason=""))
            else:
                out.append(replace(t, risk_source=RISK_SOURCE_UNAVAILABLE,
                                   initial_risk=None, r_multiple=None, r_included=False,
                                   r_exclusion_reason=R_REASON_INVALID_RISK))
        else:
            out.append(replace(t, risk_source=RISK_SOURCE_UNAVAILABLE,
                               initial_risk=None, r_multiple=None, r_included=False,
                               r_exclusion_reason=R_REASON_UNAVAILABLE_RISK))
    return out


# ── Step 4: aggregate metrics ────────────────────────────────────────────────
def compute_metrics(trades: List[ClosedTrade]) -> dict:
    """Aggregate PnL + R metrics with explicit included/excluded accounting."""
    pnl_trades = [t for t in trades if t.pnl_included]
    n_inc = len(pnl_trades)
    n_exc = len(trades) - n_inc

    realized = [t.realized_pnl for t in pnl_trades]
    total = sum(realized) if realized else 0.0
    wins = [r for r in realized if r > 0]
    losses = [r for r in realized if r < 0]
    scratch = [r for r in realized if r == 0]

    win_rate = (len(wins) / n_inc) if n_inc else None
    gross_win = sum(wins)
    gross_loss = sum(-r for r in losses)  # positive magnitude
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
        pf_display = f"{profit_factor:.2f}"
    elif gross_win > 0:
        profit_factor = None
        pf_display = "inf (no losing trades)"
    else:
        profit_factor = None
        pf_display = "n/a"
    avg_win = (gross_win / len(wins)) if wins else None
    avg_loss = (sum(losses) / len(losses)) if losses else None  # negative magnitude

    # R metrics over R-included trades only.
    r_trades = [t for t in pnl_trades if t.r_included and t.r_multiple is not None]
    r_vals = [t.r_multiple for t in r_trades]
    expectancy_R = (sum(r_vals) / len(r_vals)) if r_vals else None
    r_wins = [r for r in r_vals if r > 0]
    r_losses = [r for r in r_vals if r < 0]
    avg_win_R = (sum(r_wins) / len(r_wins)) if r_wins else None
    avg_loss_R = (sum(r_losses) / len(r_losses)) if r_losses else None

    risk_source_counts = Counter(t.risk_source for t in pnl_trades)
    excluded_by_reason = Counter(t.exclusion_reason for t in trades if not t.pnl_included)
    r_excluded_by_reason = Counter(t.r_exclusion_reason for t in pnl_trades if not t.r_included)

    return {
        "n_trades_included": n_inc,
        "n_trades_excluded": n_exc,
        "total_realized_pnl": _round(total),
        "wins": len(wins),
        "losses": len(losses),
        "scratch": len(scratch),
        "win_rate": _round(win_rate, 6) if win_rate is not None else None,
        "gross_win": _round(gross_win),
        "gross_loss": _round(gross_loss),
        "profit_factor": _round(profit_factor, 6) if profit_factor is not None else None,
        "profit_factor_display": pf_display,
        "avg_win": _round(avg_win) if avg_win is not None else None,
        "avg_loss": _round(avg_loss) if avg_loss is not None else None,
        "excluded_by_reason": dict(excluded_by_reason),
        "r": {
            "expectancy_R": _round(expectancy_R, 6) if expectancy_R is not None else None,
            "n_r_included": len(r_vals),
            "n_r_excluded": n_inc - len(r_vals),
            "avg_win_R": _round(avg_win_R, 6) if avg_win_R is not None else None,
            "avg_loss_R": _round(avg_loss_R, 6) if avg_loss_R is not None else None,
            "risk_source_counts": dict(risk_source_counts),
            "r_excluded_by_reason": dict(r_excluded_by_reason),
        },
    }


def _build_notes(enable_proxy_risk: bool) -> List[str]:
    notes = [
        "Read-only reporting module (M4-core). No IBKR connection, no orders, no "
        "changes to trading behavior.",
        "Source: reports/backtest_trades.csv - a per-bar walk-forward simulation "
        "ledger in RETURN units (no share quantity, no dollars, no Minervini stop).",
        "Closed trades are reconstructed as LONG round-trips (flat->long->flat) "
        "using the same entry/exit pairing + equity-ratio return as "
        "main.cmd_backtest_summary, grouped by (symbol, fold_start).",
        "Short opens are excluded with reason `short_unsupported` (safer minimal; "
        "the backtest default is long-only).",
        "Partial fills are not applicable to this source (positions are whole "
        "units); reserved for a future dollar-denominated source.",
        "Live/paper audit-journal reconstruction and true dollar-denominated "
        "Minervini R are intentionally deferred (NOT in M4-core).",
    ]
    if enable_proxy_risk:
        notes.append("PROXY-RISK MODE ON: R uses a fixed hard-stop fraction as a "
                     "PROXY - this is clearly NOT true Minervini 1R.")
    else:
        notes.append("Default risk mode: initial risk unavailable -> R-multiple "
                     "and expectancy_R are NOT computed (every trade excluded from "
                     "R with reason `unavailable_risk`).")
    return notes


def generate_report(trades_csv_path, *, enable_proxy_risk: bool = False,
                    proxy_pct: Optional[float] = None,
                    proxy_mode: str = "hard_stop") -> dict:
    """Build the full report dict from the backtest ledger. Read-only (reads CSV)."""
    rows = load_backtest_rows(trades_csv_path)
    trades = reconstruct_closed_trades(rows)
    if enable_proxy_risk and proxy_pct is None:
        if proxy_mode == "atr":
            atr_mult = float(getattr(config, "DEFAULT_STOP_ATR_MULTIPLE", 1.5))
            proxy_pct = atr_mult * 0.015  # 1.5x 1.5% ATR proxy (2.25%)
        else:
            proxy_pct = getattr(config, "HARD_STOP_LOSS_PCT", 0.03)

    risk_mode_name = f"proxy_{proxy_mode}" if enable_proxy_risk else "none"
    trades = apply_risk_model(trades, enable_proxy_risk=enable_proxy_risk, proxy_pct=proxy_pct, proxy_mode=proxy_mode)
    metrics = compute_metrics(trades)
    return {
        "source": "backtest_ledger",
        "source_file": str(trades_csv_path),
        "n_rows_read": len(rows),
        "pnl_units": "return_fraction",
        "risk_mode": risk_mode_name,
        "proxy_pct": (proxy_pct if enable_proxy_risk else None),
        "metrics": metrics,
        "trades": [t.as_dict() for t in trades],
        "notes": _build_notes(enable_proxy_risk),
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _r(x) -> str:
    return "n/a" if x is None else f"{x:.4f}R"


def report_to_markdown(report: dict, lang: Optional[str] = None) -> str:
    """Render a human-readable Markdown report. Pure string builder.

    ``lang`` (M5A) selects the DISPLAY language for headings / notes /
    disclaimers ONLY - default English (and ``None`` -> config default, then
    English). Numbers, file paths, reason codes, risk_source counts, and the
    JSON schema are NEVER localized. Safety / disclaimer text is kept BILINGUAL
    (English + Burmese) so the English warning is never lost.
    """
    m = report["metrics"]
    r = m["r"]
    lines = [
        "# " + coach_i18n.t("md_title", lang),
        "",
        "_" + coach_i18n.t("md_readonly_note", lang) + "_",
        "",
        f"- **Source:** `{report['source_file']}`",
        f"- **Rows read:** {report['n_rows_read']}",
        f"- **PnL units:** {report['pnl_units']} (per-unit-notional returns; the "
        "backtest ledger has no share quantity or dollars)",
        "- **Risk mode:** " + report["risk_mode"]
        + (f" (proxy_pct={report['proxy_pct']})" if report["risk_mode"] != "none" else ""),
        "",
        "## " + coach_i18n.t("md_h_realized", lang),
        "",
        f"- **Closed trades included:** {m['n_trades_included']}",
        f"- **Excluded trades:** {m['n_trades_excluded']}",
        f"- **Total realized PnL:** {m['total_realized_pnl']}",
        f"- **Wins / Losses / Scratch:** {m['wins']} / {m['losses']} / {m['scratch']}",
        f"- **Win rate:** {_pct(m['win_rate'])}",
        f"- **Gross win / Gross loss:** {m['gross_win']} / {m['gross_loss']}",
        f"- **Profit factor:** {m['profit_factor_display']}",
        f"- **Average win / Average loss:** {m['avg_win']} / {m['avg_loss']}",
        "",
        "## " + coach_i18n.t("md_h_rmultiple", lang),
        "",
    ]
    if report["risk_mode"] == "none":
        lines += [
            "> " + coach_i18n.t("md_disclaimer_none", lang),
            "",
        ]
    else:
        lines += [
            "> " + coach_i18n.t("md_disclaimer_proxy", lang, proxy_pct=report["proxy_pct"]),
            "",
        ]
    lines += [
        f"- **Expectancy (R):** {_r(r['expectancy_R'])}",
        f"- **R trades included / excluded:** {r['n_r_included']} / {r['n_r_excluded']}",
        f"- **Avg win R / Avg loss R:** {_r(r['avg_win_R'])} / {_r(r['avg_loss_R'])}",
        f"- **Risk-source counts:** {r['risk_source_counts']}",
        f"- **R excluded by reason:** {r['r_excluded_by_reason']}",
        "",
        "## " + coach_i18n.t("md_h_excluded", lang),
        "",
    ]
    if m["excluded_by_reason"]:
        for reason, n in sorted(m["excluded_by_reason"].items()):
            lines.append(f"- `{reason}`: {n}")
    else:
        lines.append("- " + coach_i18n.t("md_none", lang))
    lines += ["", "## " + coach_i18n.t("md_h_notes", lang), ""]
    lines += [f"- {n}" for n in report["notes"]]
    lines += [""]
    return "\n".join(lines)


def write_report(report: dict, *, json_path, md_path, lang: Optional[str] = None):
    """Write the JSON + Markdown artifacts. The ONLY side effect of this module.

    ``lang`` (M5A) localizes the Markdown DISPLAY only. The JSON artifact is
    NEVER localized - identical bytes regardless of language - so the machine
    schema stays stable.
    """
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report_to_markdown(report, lang=lang), encoding="utf-8")
    return json_path, md_path
