"""
stock_prompts.py — Wall-Street-style research prompt pack runner.

Reads templates in `prompts/*.md`, fills `{{placeholders}}` with live numbers
sourced from this repo (data_manager / market_universe / hot_scanner /
config.WATCHLIST), and writes a finished Markdown brief to
`reports/prompts/`.

Design rules:
- This module never places orders. It only produces research briefs.
- All network / data calls are best-effort. On any failure, the slot is
  left as `TBD` and execution continues — bad data must never break
  the run.
- Imports are wrapped in try/except so this module can also run in a
  lean environment (e.g. just the templates and placeholder fills).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = ROOT / "prompts"
REPORTS_DIR = ROOT / "reports" / "prompts"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TBD = "TBD"


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

@dataclass
class Template:
    key: str            # e.g. "01"
    file: str           # basename inside prompts/
    title: str
    persona: str
    description: str
    needs_ticker: bool = True
    needs_sector: bool = False
    needs_amount: bool = False
    extra_vars: List[str] = field(default_factory=list)


TEMPLATES: List[Template] = [
    Template(
        key="01",
        file="01_goldman_sachs_screener.md",
        title="Goldman Sachs Stock Screener",
        persona="Goldman Sachs senior equity analyst",
        description="Top-10 stock screen with P/E, D/E, moat, 12m targets.",
        extra_vars=["risk_tolerance", "investment_amount", "time_horizon",
                    "preferred_sectors", "excluded_sectors", "account_type",
                    "investment_profile"],
    ),
    Template(
        key="02",
        file="02_morgan_stanley_dcf.md",
        title="Morgan Stanley DCF Deep Dive",
        persona="Morgan Stanley VP investment banker",
        description="5y DCF, WACC, sensitivity table, valuation verdict.",
    ),
    Template(
        key="03",
        file="03_blackstone_risk_assessment.md",
        title="Blackstone Risk Assessment",
        persona="Blackstone MD, private equity risk",
        description="VaR, drawdown, stress test, position sizing.",
        extra_vars=["position_size", "time_horizon"],
    ),
    Template(
        key="04",
        file="04_jpmorgan_earnings.md",
        title="JPMorgan Earnings Breakdown",
        persona="JPMorgan senior equity research",
        description="Pre-earnings brief, bull/bear case, options-implied move.",
        extra_vars=["earnings_date", "days_until_earnings"],
    ),
    Template(
        key="05",
        file="05_peter_lynch_growth.md",
        title="Peter Lynch Growth Framework",
        persona="Peter Lynch (Magellan Fund)",
        description="Two-minute drill, category, PEG, Lynch verdict.",
    ),
    Template(
        key="06",
        file="06_citadel_technical.md",
        title="Citadel Technical Analysis",
        persona="Citadel senior quant trader",
        description="Multi-TF trend, S/R, indicators, trade plan.",
        extra_vars=["current_position"],
    ),
    Template(
        key="07",
        file="07_harvard_dividend.md",
        title="Harvard Endowment Dividend Strategy",
        persona="Harvard endowment CIO",
        description="Dividend picks, payout safety, DRIP compounding.",
        needs_ticker=False,
        needs_amount=True,
        extra_vars=["monthly_income_goal", "account_type", "tax_bracket",
                    "risk_tolerance"],
    ),
    Template(
        key="08",
        file="08_bain_competitive.md",
        title="Bain Competitive Advantage Analysis",
        persona="Bain & Company senior partner",
        description="Competitive landscape, moats, SWOT, single best pick.",
        needs_ticker=False,
        needs_sector=True,
        extra_vars=["reference_companies"],
    ),
    Template(
        key="09",
        file="09_renaissance_patterns.md",
        title="Renaissance Technologies Pattern Finder",
        persona="Renaissance Technologies quant researcher",
        description="Seasonality, day-of-week, insider, options, edge summary.",
        extra_vars=["time_period"],
    ),
    Template(
        key="10",
        file="10_mckinsey_macro.md",
        title="McKinsey Macro Impact Assessment",
        persona="McKinsey Global Institute senior partner",
        description="Rates, CPI, GDP, USD, Fed path, portfolio action plan.",
        needs_ticker=False,
        extra_vars=["current_holdings", "biggest_economic_concern",
                    "time_horizon"],
    ),
]


def find_template(key: str) -> Template:
    key = key.zfill(2)
    for t in TEMPLATES:
        if t.key == key:
            return t
    raise KeyError(f"Unknown template key '{key}'. Run `prompts list`.")


# ---------------------------------------------------------------------------
# Placeholder extraction
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")


def extract_placeholders(text: str) -> List[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(text)))


# ---------------------------------------------------------------------------
# Live data helpers
# ---------------------------------------------------------------------------

def _safe_float(x: Any, digits: int = 4) -> str:
    if x is None:
        return TBD
    try:
        f = float(x)
    except (TypeError, ValueError):
        return TBD
    if f != f:  # NaN
        return TBD
    return f"{f:.{digits}f}"


def _fmt_money(x: Any) -> str:
    if x is None:
        return TBD
    try:
        v = float(x)
    except (TypeError, ValueError):
        return TBD
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v/1e3:.2f}K"
    return f"${v:,.2f}"


def _try_data_manager(symbol: str):
    """Best-effort fetch of OHLCV + features. Returns (raw_df, feat_df) or (None, None)."""
    try:
        import data_manager  # local module
        raw = data_manager.fetch_ohlcv(symbol)
        if raw is None or len(raw) < 60:
            return raw, None
        feat = data_manager.build_features(raw)
        return raw, feat
    except Exception as e:  # noqa: BLE001
        logger.debug("data_manager fetch failed for %s: %s", symbol, e)
        return None, None


def ticker_context(symbol: str) -> Dict[str, str]:
    """Build a context dict of auto-filled placeholders for a single ticker."""
    ctx: Dict[str, str] = {
        "ticker": (symbol or TBD).upper(),
        "company_name": TBD,
        "ticker_company": f"{symbol.upper() if symbol else TBD} ({TBD})",
        "reference_price": TBD,
        "ttm_revenue": TBD,
        "ttm_ebitda": TBD,
        "ttm_fcf": TBD,
        "shares_diluted_m": TBD,
        "net_debt": TBD,
        "revenue_cagr_5y": TBD,
        "beta": TBD,
        "vol_60d": TBD,
        "vol_1y": TBD,
        "max_drawdown_1y": TBD,
        "short_interest_pct": TBD,
        "debt_to_equity": TBD,
        "interest_coverage": TBD,
        "eps_history": TBD,
        "revenue_history": TBD,
        "consensus_revenue": TBD,
        "consensus_eps": TBD,
        "implied_move_pct": TBD,
        "post_earnings_moves": TBD,
        "eps_cagr_5y": TBD,
        "peg_ratio": TBD,
        "insider_ownership_pct": TBD,
        "institutional_ownership_pct": TBD,
        "cfo_ttm": TBD,
        "sma50": TBD,
        "sma100": TBD,
        "sma200": TBD,
        "rsi_14": TBD,
        "macd_value": TBD,
        "macd_signal": TBD,
        "bollinger_pct_b": TBD,
        "avg_volume_20d": TBD,
        "atr_pct": TBD,
        "fifty_two_week_range": TBD,
        "bar_count": TBD,
        "date_range": TBD,
        "avg_daily_return": TBD,
        "daily_return_stdev": TBD,
        "up_down_day_ratio": TBD,
        "days_since_earnings": TBD,
    }

    raw, feat = _try_data_manager(symbol)
    if feat is None or feat.empty:
        return ctx

    close = feat["Close"] if "Close" in feat.columns else None
    last = feat.iloc[-1]

    if close is not None:
        ctx["reference_price"] = _safe_float(last["Close"], 2)
        # Moving averages (compute SMA100/200 from raw close)
        try:
            raw_close = (raw["Close"] if raw is not None else feat["Close"])
            ctx["sma50"] = _safe_float(raw_close.rolling(50).mean().iloc[-1], 2)
            ctx["sma100"] = _safe_float(raw_close.rolling(100).mean().iloc[-1], 2)
            ctx["sma200"] = _safe_float(raw_close.rolling(200).mean().iloc[-1], 2)
        except Exception:
            pass
        # 52w range
        try:
            hi = raw_close.tail(252).max()
            lo = raw_close.tail(252).min()
            ctx["fifty_two_week_range"] = (
                f"{_safe_float(lo, 2)} – {_safe_float(hi, 2)}"
            )
        except Exception:
            pass
        # Returns
        try:
            ret_5y = (raw_close.iloc[-1] / raw_close.iloc[0]) - 1.0
            years = max((raw_close.index[-1] - raw_close.index[0]).days / 365.25, 0.1)
            cagr = (ret_5y + 1.0) ** (1.0 / years) - 1.0
            ctx["revenue_cagr_5y"] = f"{cagr*100:.2f}%"
        except Exception:
            pass
        # Volatility
        try:
            daily_ret = raw_close.pct_change().dropna()
            vol_60 = daily_ret.tail(60).std() * (252 ** 0.5)
            vol_1y = daily_ret.tail(252).std() * (252 ** 0.5)
            ctx["vol_60d"] = f"{vol_60*100:.2f}%"
            ctx["vol_1y"] = f"{vol_1y*100:.2f}%"
            ctx["avg_daily_return"] = f"{daily_ret.mean()*100:.4f}%"
            ctx["daily_return_stdev"] = f"{daily_ret.std()*100:.4f}%"
            up = (daily_ret > 0).sum()
            dn = (daily_ret < 0).sum()
            ctx["up_down_day_ratio"] = f"{up}:{dn}"
            cum = (1 + daily_ret).cumprod()
            peak = cum.cummax()
            dd = (cum / peak - 1.0).min()
            ctx["max_drawdown_1y"] = f"{dd*100:.2f}%"
        except Exception:
            pass
        # ATR pct
        if "atr_pct" in feat.columns:
            ctx["atr_pct"] = _safe_float(last.get("atr_pct"), 4)
        # Volume
        if "vol_ratio" in feat.columns:
            pass
        if "Volume" in raw.columns:
            ctx["avg_volume_20d"] = f"{int(raw['Volume'].tail(20).mean()):,}"

    if "rsi" in feat.columns:
        ctx["rsi_14"] = _safe_float(last.get("rsi"), 2)
    if "macd" in feat.columns:
        ctx["macd_value"] = _safe_float(last.get("macd"), 6)
    if "macd_sig" in feat.columns:
        ctx["macd_signal"] = _safe_float(last.get("macd_sig"), 6)
    if "bb_pct" in feat.columns:
        ctx["bollinger_pct_b"] = _safe_float(last.get("bb_pct"), 4)

    ctx["bar_count"] = str(len(feat))
    if feat.index.size:
        ctx["date_range"] = (
            f"{feat.index[0].date()} ? {feat.index[-1].date()}"
        )

    return ctx


def sector_context(sector_name: str) -> Dict[str, str]:
    """Best-effort sector aggregates. Falls back to TBDs."""
    return {
        "sector_name": sector_name or TBD,
        "universe_source": "config.WATCHLIST (static)",
        "candidate_count": TBD,
        "median_revenue": TBD,
        "median_op_margin": TBD,
        "median_rd_intensity": TBD,
        "market_share_trend": TBD,
    }


def dividend_context() -> Dict[str, str]:
    return {
        "universe_source": "config.WATCHLIST (static)",
        "candidate_count": TBD,
        "avg_candidate_yield": TBD,
        "avg_payout_ratio": TBD,
        "avg_div_growth_5y": TBD,
        "top_sector_exposures": TBD,
    }


def macro_context() -> Dict[str, str]:
    """Static fallback. We do not pull live macro data in v1."""
    return {
        "fed_funds_rate": TBD,
        "ust_10y": TBD,
        "ust_2y": TBD,
        "cpi_yoy": TBD,
        "core_cpi_yoy": TBD,
        "unemployment_rate": TBD,
        "dxy": TBD,
        "gdp_forecast": TBD,
        "vix": TBD,
        "sector_tilt": TBD,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_template(template: Template, vars_in: Dict[str, str]) -> str:
    path = PROMPTS_DIR / template.file
    if not path.exists():
        raise FileNotFoundError(f"Missing template file: {path}")
    text = path.read_text(encoding="utf-8")

    # Start with the explicit variables the user passed
    ctx: Dict[str, str] = {k: ("" if v is None else str(v)) for k, v in vars_in.items()}

    # Always-fill common fields
    ctx.setdefault("generated_at", _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    ctx.setdefault("mode", "paper-trading research brief")

    # Ticker-specific auto-fill
    if template.needs_ticker and ctx.get("ticker"):
        ctx.update(ticker_context(ctx["ticker"]))
        # Ticker/company composite for inline use
        tk = ctx.get("ticker", TBD)
        co = ctx.get("company_name", TBD)
        ctx.setdefault("ticker_company", f"{tk} ({co})")
    if template.needs_sector:
        ctx.setdefault("sector_name", TBD)
        ctx.update(sector_context(ctx.get("sector_name", "")))
    if template.needs_amount:
        ctx.update(dividend_context())
    if template.key == "10":
        ctx.update(macro_context())

    # Fill any placeholders the caller did not supply with TBD
    for ph in extract_placeholders(text):
        ctx.setdefault(ph, TBD)

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        return ctx.get(key, TBD)

    return _PLACEHOLDER_RE.sub(_sub, text)


def write_render(template: Template, vars_in: Dict[str, str],
                 out_path: Optional[Path] = None) -> Path:
    rendered = render_template(template, vars_in)

    if out_path is None:
        slug_parts = [
            template.key,
            (vars_in.get("ticker") or vars_in.get("sector_name") or "brief"),
        ]
        slug = "_".join(p for p in slug_parts if p).lower().replace(" ", "_")
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = REPORTS_DIR / f"{slug}_{stamp}.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="prompts",
        description="Stock Engine Pro v3 — research prompt pack",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all available templates")

    p_show = sub.add_parser("show", help="Show template with placeholders visible")
    p_show.add_argument("--template", required=True)
    p_show.add_argument("--ticker", default=None)
    p_show.add_argument("--sector", default=None)

    p_render = sub.add_parser("render", help="Fill placeholders and write report")
    p_render.add_argument("--template", required=True)
    p_render.add_argument("--ticker", default=None)
    p_render.add_argument("--sector", default=None)
    p_render.add_argument("--position", default=None,
                          help="Current position (Citadel template)")
    p_render.add_argument("--amount", default=None, type=float,
                          help="Investment amount (USD)")
    p_render.add_argument("--monthly-goal", default=None, type=float,
                          help="Monthly income goal (USD)")
    p_render.add_argument("--account-type", default=None)
    p_render.add_argument("--tax-bracket", default=None)
    p_render.add_argument("--time-horizon", default=None)
    p_render.add_argument("--risk-tolerance", default=None)
    p_render.add_argument("--preferred-sectors", default=None)
    p_render.add_argument("--excluded-sectors", default=None)
    p_render.add_argument("--earnings-date", default=None)
    p_render.add_argument("--current-holdings", default=None)
    p_render.add_argument("--biggest-economic-concern", default=None)
    p_render.add_argument("--out", default=None,
                          help="Output path. Default: reports/prompts/<slug>_<ts>.md")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.cmd == "list":
        print(f"{'#':<4}{'Persona':<48}Title")
        print("-" * 110)
        for t in TEMPLATES:
            print(f"{t.key:<4}{t.persona[:47]:<48}{t.title}")
        print()
        print("Files live in: prompts/")
        print("Rendered briefs go to: reports/prompts/")
        return 0

    template = find_template(args.template)

    if args.cmd == "show":
        text = (PROMPTS_DIR / template.file).read_text(encoding="utf-8")
        print(text)
        return 0

    if args.cmd == "render":
        vars_in: Dict[str, str] = {}
        if args.ticker:
            vars_in["ticker"] = args.ticker.upper()
        if args.sector:
            vars_in["sector_name"] = args.sector
        if args.position:
            vars_in["current_position"] = args.position
        if args.amount is not None:
            vars_in["investment_amount"] = f"{args.amount:,.2f}"
        if args.monthly_goal is not None:
            vars_in["monthly_income_goal"] = f"{args.monthly_goal:,.2f}"
        if args.account_type:
            vars_in["account_type"] = args.account_type
        if args.tax_bracket:
            vars_in["tax_bracket"] = args.tax_bracket
        if args.time_horizon:
            vars_in["time_horizon"] = args.time_horizon
        if args.risk_tolerance:
            vars_in["risk_tolerance"] = args.risk_tolerance
        if args.preferred_sectors:
            vars_in["preferred_sectors"] = args.preferred_sectors
        if args.excluded_sectors:
            vars_in["excluded_sectors"] = args.excluded_sectors
        if args.earnings_date:
            vars_in["earnings_date"] = args.earnings_date
        if args.current_holdings:
            vars_in["current_holdings"] = args.current_holdings
        if args.biggest_economic_concern:
            vars_in["biggest_economic_concern"] = args.biggest_economic_concern

        # Friendly investment_profile composite for the Goldman template
        if template.key == "01":
            bits = []
            if vars_in.get("risk_tolerance"):
                bits.append(f"risk={vars_in['risk_tolerance']}")
            if vars_in.get("investment_amount"):
                bits.append(f"amount=${vars_in['investment_amount']}")
            if vars_in.get("time_horizon"):
                bits.append(f"horizon={vars_in['time_horizon']}")
            if vars_in.get("preferred_sectors"):
                bits.append(f"sectors={vars_in['preferred_sectors']}")
            if vars_in.get("excluded_sectors"):
                bits.append(f"exclude={vars_in['excluded_sectors']}")
            if vars_in.get("account_type"):
                bits.append(f"account={vars_in['account_type']}")
            if bits:
                vars_in["investment_profile"] = " | ".join(bits)

        out_path = Path(args.out) if args.out else None
        written = write_render(template, vars_in, out_path)
        print(f"OK: wrote {written}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
