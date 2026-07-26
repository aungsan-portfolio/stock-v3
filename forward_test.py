"""forward_test.py - offline, read-only forward PAPER-TEST report.

PURE + READ-ONLY. Like expectancy.py, this module never connects to IBKR, never
places orders, never imports the order path (ibkr_bridge / account_guard /
ib_insync), and never alters trading behavior. Its ONLY side effects are writing
report artifacts (JSON + Markdown) under reports/.

Source: ``reports/paper_trades.jsonl`` - the append-only journal of REAL paper
fills written during ``python main.py paper`` runs (see paper_ledger.py). Closed
round-trips are reconstructed by ``paper_ledger.reconstruct_paper_trades`` (FIFO
per symbol, DOLLAR PnL, broker-protective-stop R). The headline metrics (win
rate / expectancy_R / profit factor / avg win-loss) are produced by the SAME
``expectancy.compute_metrics`` used for the backtest report, so the two reports
are definitionally reconciled. This module adds the forward-test-specific
aggregates the backtest cannot give: dollar + percent max drawdown, Sharpe,
per-day realized PnL vs the daily-loss cap, expectancy bucketed by exit kind and
confidence band, and a summary of still-open positions.

R-multiple here is BROKER PROTECTIVE-STOP risk (entry vs. the real resting stop),
NOT a Minervini setup-based R - see paper_ledger.RISK_SOURCE_PROTECTIVE_STOP.
"""
from __future__ import annotations

import json
import logging
import math
import statistics
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import config
import expectancy
import paper_ledger

logger = logging.getLogger(__name__)

# Trading days per year, matching backtest._sharpe's annualization factor.
_TRADING_DAYS = 252


def _round(x, n: int = 6):
    try:
        return round(float(x), n)
    except (TypeError, ValueError):
        return x


def _day_of(ts) -> str:
    """Calendar day (YYYY-MM-DD) from an ISO timestamp string."""
    return str(ts or "")[:10]


def _in_window(trade, since) -> bool:
    """Is a reconstructed trade inside the report window? A trade's effective date
    is its exit (closed trades + orphan exits) or, lacking one, its entry (still-
    open positions). Applied uniformly so included AND excluded counts honor the
    same ``since`` window."""
    if not since:
        return True
    eff = trade.exit_date or trade.entry_date or ""
    return str(eff) >= str(since)


# ── Aggregates the backtest report cannot produce ────────────────────────────
def _drawdown(realized_seq: List[float], base: float):
    """Max drawdown over the realized-equity curve (base + cumulative PnL).

    Returns ``(max_dd_usd, max_dd_pct)`` as NON-POSITIVE magnitudes (0.0 when the
    curve never draws down). The dollar figure is independent of ``base`` (it
    cancels); the percent figure is measured against the running peak equity.
    """
    peak = float(base)
    eq = float(base)
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    for pnl in realized_seq:
        eq += float(pnl)
        if eq > peak:
            peak = eq
        dd_usd = eq - peak  # <= 0
        if dd_usd < max_dd_usd:
            max_dd_usd = dd_usd
        if peak > 0:
            frac = dd_usd / peak
            if frac < max_dd_pct:
                max_dd_pct = frac
    return max_dd_usd, max_dd_pct


def _sharpe(daily_returns: List[float]) -> float:
    """Annualized Sharpe of a per-day return series (rf=0), matching the
    backtest's mean/std*sqrt(252) convention. Stdlib-only; 0.0 when undefined."""
    rs = [float(r) for r in daily_returns
          if isinstance(r, (int, float)) and math.isfinite(r)]
    if len(rs) < 2:
        return 0.0
    try:
        std = statistics.stdev(rs)  # sample std (ddof=1)
    except statistics.StatisticsError:
        return 0.0
    if std <= 0 or not math.isfinite(std):
        return 0.0
    return float((statistics.mean(rs) / std) * math.sqrt(_TRADING_DAYS))


def _daily_breakdown(closed: List[paper_ledger.PaperTrade], cap: float) -> List[dict]:
    """Per-day realized PnL + a read-only flag for whether the daily-loss cap
    WOULD have tripped (the live gate enforces the cap; this only reports it)."""
    by_day: "OrderedDict[str, dict]" = OrderedDict()
    for t in closed:
        day = _day_of(t.exit_date)
        rec = by_day.setdefault(day, {"date": day, "realized_pnl": 0.0, "n_trades": 0})
        rec["realized_pnl"] += float(t.realized_pnl or 0.0)
        rec["n_trades"] += 1
    out = []
    for day in sorted(by_day):
        rec = by_day[day]
        pnl = rec["realized_pnl"]
        out.append({
            "date": day,
            "realized_pnl": _round(pnl, 2),
            "n_trades": rec["n_trades"],
            "daily_loss_cap_usd": float(cap),
            "daily_loss_breach": bool(cap > 0 and (-pnl) >= float(cap)),
        })
    return out


def _bucket_metrics(trades: List[paper_ledger.PaperTrade]) -> dict:
    """Reuse the shared metric engine on a subset (so every bucket reconciles
    with the headline). Returns a trimmed view of expectancy.compute_metrics."""
    m = expectancy.compute_metrics([t.to_closed_trade() for t in trades])
    return {
        "n_trades": m["n_trades_included"],
        "total_realized_pnl": m["total_realized_pnl"],
        "win_rate": m["win_rate"],
        "profit_factor_display": m["profit_factor_display"],
        "expectancy_R": m["r"]["expectancy_R"],
        "n_r_included": m["r"]["n_r_included"],
    }


_CONF_EDGES = [0.0, 0.6, 0.7, 0.8, 0.9, 1.0001]


def _confidence_band(conf) -> str:
    if conf is None:
        return "unknown"
    try:
        c = float(conf)
    except (TypeError, ValueError):
        return "unknown"
    for lo, hi in zip(_CONF_EDGES, _CONF_EDGES[1:]):
        if lo <= c < hi:
            top = min(hi, 1.0)
            return f"{lo:.2f}-{top:.2f}"
    return "unknown"


def _group_buckets(closed: List[paper_ledger.PaperTrade], key_fn) -> dict:
    groups: "OrderedDict[str, list]" = OrderedDict()
    for t in closed:
        groups.setdefault(key_fn(t), []).append(t)
    return {k: _bucket_metrics(v) for k, v in groups.items()}


def _open_positions(trades: List[paper_ledger.PaperTrade]) -> List[dict]:
    out = []
    for t in trades:
        if t.exclusion_reason == expectancy.REASON_OPEN_AT_PERIOD_END:
            out.append({
                "symbol": t.symbol,
                "qty": _round(t.qty, 6),
                "entry_price": _round(t.entry_price, 6),
                "entry_date": t.entry_date,
                "session": t.session,
            })
    return out


def _build_notes() -> List[str]:
    return [
        "Read-only forward paper-test report. No IBKR connection, no orders, no "
        "changes to trading behavior.",
        "Source: reports/paper_trades.jsonl - the append-only journal of REAL "
        "paper fills (entry/stop from the live hook, exit from the read-only "
        "broker-executions sweep).",
        "PnL is in US DOLLARS (qty * (exit - entry)). Win rate / expectancy_R / "
        "profit factor come from the SAME expectancy.compute_metrics as the "
        "backtest report, so the two reconcile.",
        "R-multiple uses BROKER PROTECTIVE-STOP risk (entry vs. the real resting "
        "stop). This is NOT a Minervini setup-based R; trades with no valid stop "
        "on record are excluded from the R aggregates with a reason.",
        "Max drawdown $ is curve-independent; the % is measured against the "
        "running peak of (start_equity_basis + cumulative realized PnL).",
        "Daily-loss is REPORTED only (does the per-day realized PnL breach "
        "MAX_DAILY_LOSS_USD?). The live daily-loss kill-switch is enforced "
        "elsewhere in the order path and is unchanged by this report.",
        "Open positions are listed separately and excluded from realized metrics "
        "(reason open_at_period_end); exits with no covering entry are orphan_exit.",
    ]


# ── Orchestration ────────────────────────────────────────────────────────────
def generate_report(journal_path=None, *, session: Optional[str] = None,
                    since: Optional[str] = None,
                    start_equity: Optional[float] = None) -> dict:
    """Build the full forward-test report dict from the paper journal. Read-only.

    ``session`` filters events by their stamped session label (clean entry/exit
    pairing). ``since`` (YYYY-MM-DD or ISO) keeps only CLOSED trades whose exit is
    on/after it for the windowed realized metrics; open positions are always
    listed. ``start_equity`` is the notional baseline for the drawdown/Sharpe
    curve (default config.PAPER_CAPITAL).
    """
    path = Path(journal_path) if journal_path is not None else None
    events = paper_ledger.load_journal(path)
    n_events = len(events)
    if session is not None:
        events = [e for e in events if (e.get("session") or None) == session]

    # Reconstruct over the FULL (session-filtered) history so FIFO pairing is
    # correct, then apply the `since` window uniformly to every reconstructed
    # trade so included (closed) AND excluded (open/orphan) counts are consistent.
    all_trades = paper_ledger.reconstruct_paper_trades(events)
    trades = [t for t in all_trades if _in_window(t, since)]

    closed = [t for t in trades if t.pnl_included]

    # Headline metrics through the SHARED engine (included = windowed closed,
    # excluded = windowed open positions + orphan exits), so the forward and
    # backtest reports use identical win-rate / expectancy / profit-factor defs.
    metric_trades = [t.to_closed_trade() for t in trades]
    metrics = expectancy.compute_metrics(metric_trades)

    base = float(start_equity if start_equity is not None
                 else getattr(config, "PAPER_CAPITAL", 100_000.0))
    cap = float(getattr(config, "MAX_DAILY_LOSS_SWING", getattr(config, "MAX_DAILY_LOSS_USD", 0.0)) or 0.0)

    ordered = sorted(closed, key=lambda t: (t.exit_date or ""))
    realized_seq = [float(t.realized_pnl or 0.0) for t in ordered]
    max_dd_usd, max_dd_pct = _drawdown(realized_seq, base)
    daily = _daily_breakdown(closed, cap)
    daily_returns = [float(d["realized_pnl"]) / base for d in daily] if base > 0 else []

    return {
        "source": "paper_journal",
        "source_file": str(path if path is not None else paper_ledger._journal_path()),
        "n_events_read": n_events,
        "session_filter": session,
        "since_filter": since,
        "pnl_units": "usd",
        "start_equity_basis": base,
        "metrics": metrics,
        "drawdown": {
            "max_drawdown_usd": _round(max_dd_usd, 2),
            "max_drawdown_pct": _round(max_dd_pct, 6),
        },
        "sharpe": _round(_sharpe(daily_returns), 6),
        "daily": daily,
        "by_exit_kind": _group_buckets(closed, lambda t: t.exit_kind or "unknown"),
        "by_confidence": _group_buckets(closed, lambda t: _confidence_band(t.confidence)),
        "open_positions": _open_positions(trades),
        "trades": [t.as_dict() for t in trades],
        "notes": _build_notes(),
    }


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _r(x) -> str:
    return "n/a" if x is None else f"{x:.4f}R"


def report_to_markdown(report: dict) -> str:
    """Render a human-readable Markdown report. Pure string builder (English)."""
    m = report["metrics"]
    r = m["r"]
    dd = report["drawdown"]
    lines = [
        "# Forward Paper-Test Report",
        "",
        "_Read-only. No IBKR connection, no orders, no changes to trading behavior._",
        "",
        f"- **Source:** `{report['source_file']}`",
        f"- **Events read:** {report['n_events_read']}",
        f"- **Session filter:** {report['session_filter'] or 'all'}",
        f"- **Since filter:** {report['since_filter'] or 'all'}",
        f"- **PnL units:** {report['pnl_units']} (US dollars)",
        f"- **Start-equity basis:** {report['start_equity_basis']}",
        "",
        "## Realized performance",
        "",
        f"- **Closed trades included:** {m['n_trades_included']}",
        f"- **Excluded (open/orphan):** {m['n_trades_excluded']}",
        f"- **Total realized PnL ($):** {m['total_realized_pnl']}",
        f"- **Wins / Losses / Scratch:** {m['wins']} / {m['losses']} / {m['scratch']}",
        f"- **Win rate:** {_pct(m['win_rate'])}",
        f"- **Gross win / Gross loss ($):** {m['gross_win']} / {m['gross_loss']}",
        f"- **Profit factor:** {m['profit_factor_display']}",
        f"- **Average win / Average loss ($):** {m['avg_win']} / {m['avg_loss']}",
        f"- **Max drawdown:** {dd['max_drawdown_usd']} ({_pct(dd['max_drawdown_pct'])})",
        f"- **Sharpe (daily, annualized):** {report['sharpe']}",
        "",
        "## R-multiple (broker protective-stop risk)",
        "",
        "> R uses entry vs. the REAL resting broker protective stop. This is NOT a "
        "Minervini setup-based R; trades with no valid stop are excluded from R.",
        "",
        f"- **Expectancy (R):** {_r(r['expectancy_R'])}",
        f"- **R trades included / excluded:** {r['n_r_included']} / {r['n_r_excluded']}",
        f"- **Avg win R / Avg loss R:** {_r(r['avg_win_R'])} / {_r(r['avg_loss_R'])}",
        f"- **Risk-source counts:** {r['risk_source_counts']}",
        f"- **R excluded by reason:** {r['r_excluded_by_reason']}",
        "",
        "## Per-day realized PnL vs daily-loss cap",
        "",
    ]
    if report["daily"]:
        for d in report["daily"]:
            flag = "  ⚠️ BREACH" if d["daily_loss_breach"] else ""
            lines.append(f"- `{d['date']}`: {d['realized_pnl']} "
                         f"({d['n_trades']} trades){flag}")
    else:
        lines.append("- (none)")
    lines += ["", "## Expectancy by exit kind", ""]
    if report["by_exit_kind"]:
        for kind, b in report["by_exit_kind"].items():
            lines.append(f"- `{kind}`: {b['n_trades']} trades, PnL {b['total_realized_pnl']}, "
                         f"win {_pct(b['win_rate'])}, exp {_r(b['expectancy_R'])}")
    else:
        lines.append("- (none)")
    lines += ["", "## Expectancy by confidence band", ""]
    if report["by_confidence"]:
        for band, b in report["by_confidence"].items():
            lines.append(f"- `{band}`: {b['n_trades']} trades, PnL {b['total_realized_pnl']}, "
                         f"win {_pct(b['win_rate'])}, exp {_r(b['expectancy_R'])}")
    else:
        lines.append("- (none)")
    lines += ["", "## Open positions (excluded from realized metrics)", ""]
    if report["open_positions"]:
        for p in report["open_positions"]:
            lines.append(f"- {p['symbol']}: qty {p['qty']} @ {p['entry_price']} "
                         f"(entered {p['entry_date']})")
    else:
        lines.append("- (none)")
    lines += ["", "## Excluded by reason", ""]
    if m["excluded_by_reason"]:
        for reason, n in sorted(m["excluded_by_reason"].items()):
            lines.append(f"- `{reason}`: {n}")
    else:
        lines.append("- (none)")
    lines += ["", "## Notes", ""]
    lines += [f"- {n}" for n in report["notes"]]
    lines += [""]
    return "\n".join(lines)


def write_report(report: dict, *, json_path, md_path):
    """Write the JSON + Markdown artifacts. The ONLY side effect of this module."""
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(report_to_markdown(report), encoding="utf-8")
    return json_path, md_path
