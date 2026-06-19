"""coach_i18n.py - M5A: minimal, DISPLAY-ONLY Burmese (``my``) localization.

This module exists only to translate human-readable *display* strings (console
labels, Markdown headings, safety/disclaimer notes) for the read-only coach and
expectancy reports. It is deliberately tiny and dependency-free:

  * stdlib + ``config`` only - never imports the order path, the broker layer,
    minervini decision logic, sizing, or gates.
  * Language is a DISPLAY concern ONLY. It must NEVER enter
    decision/gate/sizing/order code. Numbers, JSON keys, file paths, reason
    codes, and BUY/SELL/HOLD action tokens are never localized.
  * English (``"en"``) is always the fallback - for an unknown language, an
    unknown key, or a missing translation.
  * Critical safety / disclaimer strings are kept BILINGUAL (English + Burmese)
    rather than Burmese-only, so the English warning is never lost.

Public API:
  * ``resolve_language(cli_value=None)`` -> ``"en"`` | ``"my"``
  * ``t(key, lang=None, **kwargs)`` -> localized, format-safe string
"""
from __future__ import annotations

import logging

import config

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ("en", "my")


# ── Language resolution ──────────────────────────────────────────────────────
def _normalize(value):
    """Return a supported language code, or ``None`` if ``value`` is unusable.

    Tolerant of casing / surrounding whitespace; never raises.
    """
    if value is None:
        return None
    try:
        v = str(value).strip().lower()
    except Exception:  # pragma: no cover - defensive
        return None
    return v if v in SUPPORTED_LANGUAGES else None


def resolve_language(cli_value=None) -> str:
    """Resolve the effective display language.

    Precedence (display-only, never affects trading):
      1. ``cli_value`` if it is a supported language (CLI wins over config).
      2. ``config.MINERVINI_COACH_LANGUAGE`` if supported.
      3. English (``"en"``) as the final fallback.

    An unknown / unsupported language at any level falls through to the next
    step, so an unknown value ultimately yields English.
    """
    cli = _normalize(cli_value)
    if cli is not None:
        return cli
    cfg = _normalize(getattr(config, "MINERVINI_COACH_LANGUAGE", DEFAULT_LANGUAGE))
    if cfg is not None:
        return cfg
    return DEFAULT_LANGUAGE


# ── Format-safe string lookup ────────────────────────────────────────────────
class _SafeDict(dict):
    """A mapping that renders missing format keys as a literal ``{key}`` instead
    of raising ``KeyError`` - so ``t()`` never explodes on a missing kwarg."""

    def __missing__(self, key):  # noqa: D401 - simple passthrough
        return "{" + str(key) + "}"


def _safe_format(template: str, kwargs: dict) -> str:
    """Best-effort ``str.format_map`` that never raises.

    Missing keys are left as ``{key}`` placeholders; a malformed template falls
    back to the raw template text.
    """
    try:
        return template.format_map(_SafeDict(kwargs or {}))
    except Exception:  # pragma: no cover - defensive (malformed template)
        return template


def t(key, lang=None, **kwargs) -> str:
    """Translate ``key`` into ``lang`` (display only). Never raises.

    Resolution / fallback:
      * ``lang`` is resolved via :func:`resolve_language` (so ``None`` consults
        config, and an unknown language falls back to English).
      * Unknown ``key`` -> the key itself is returned (format-safe).
      * Known key but no translation for the resolved language -> English.

    ``**kwargs`` are substituted into the template with format-safe semantics; a
    missing substitution never raises.
    """
    resolved = resolve_language(lang)
    table = _STRINGS.get(key)
    if not isinstance(table, dict):
        # Unknown key: echo it (still run it through safe-format for consistency).
        return _safe_format(str(key), kwargs)
    template = table.get(resolved)
    if template is None:
        template = table.get(DEFAULT_LANGUAGE)
    if template is None:  # pragma: no cover - every entry defines English
        return _safe_format(str(key), kwargs)
    return _safe_format(template, kwargs)


# ── Translation table ────────────────────────────────────────────────────────
# Every entry MUST define an "en" template (the guaranteed fallback). "my" is
# Burmese. Code tokens (`risk_source=unavailable`, `--proxy-risk`, `--confirm`,
# `--chart-checked`, proxy_pct, BUY/SELL/HOLD) and numeric placeholders are kept
# literal in BOTH languages. Safety / disclaimer strings embed the full English
# text inside the "my" value (bilingual) so the warning is never lost.
_STRINGS = {
    # ── Expectancy report: Markdown headings / notes (display only) ──────────
    "md_title": {
        "en": "Expectancy / R-Multiple Report",
        "my": "မျှော်မှန်းတန်ဖိုး / R-Multiple အစီရင်ခံစာ",
    },
    "md_readonly_note": {
        "en": ("Read-only M4 report. No backtest was run, no models trained, no "
               "IBKR connection, no orders placed. Source is the offline backtest "
               "ledger."),
        "my": ("Read-only M4 report. No backtest was run, no models trained, no "
               "IBKR connection, no orders placed. Source is the offline backtest "
               "ledger. — ဖတ်ရှုရုံသာ M4 အစီရင်ခံစာ။ backtest မလုပ်ဆောင်ထားပါ၊ "
               "မော်ဒယ်များ မလေ့ကျင့်ထားပါ၊ IBKR ချိတ်ဆက်မှု မရှိပါ၊ အော်ဒါများ "
               "မတင်ထားပါ။ အရင်းအမြစ်မှာ offline backtest ledger ဖြစ်သည်။"),
    },
    "md_h_realized": {
        "en": "Realized performance (PnL-based)",
        "my": "အကောင်အထည်ဖော်ပြီး စွမ်းဆောင်ရည် (PnL အခြေခံ)",
    },
    "md_h_rmultiple": {
        "en": "R-multiple / expectancy",
        "my": "R-multiple / မျှော်မှန်းတန်ဖိုး",
    },
    "md_h_excluded": {
        "en": "Excluded trades by reason",
        "my": "ဖယ်ထုတ်ထားသော အရောင်းအဝယ်များ (အကြောင်းအရင်းအလိုက်)",
    },
    "md_h_notes": {
        "en": "Notes",
        "my": "မှတ်ချက်များ",
    },
    "md_none": {
        "en": "none",
        "my": "မရှိပါ",
    },
    "md_disclaimer_none": {
        "en": ("**R-multiple and expectancy are NOT computed.** The backtest ledger "
               "carries no true Minervini 1R initial risk, so every trade is tagged "
               "`risk_source=unavailable` and excluded from R metrics. Pass "
               "`--proxy-risk` for a clearly-labeled hard-stop *proxy* (never true "
               "Minervini R)."),
        "my": ("**R-multiple and expectancy are NOT computed.** The backtest ledger "
               "carries no true Minervini 1R initial risk, so every trade is tagged "
               "`risk_source=unavailable` and excluded from R metrics. Pass "
               "`--proxy-risk` for a clearly-labeled hard-stop *proxy* (never true "
               "Minervini R). — **R-multiple နှင့် expectancy ကို တွက်ချက်ထားခြင်း "
               "မရှိပါ။** backtest ledger တွင် စစ်မှန်သော Minervini 1R အစ risk "
               "မပါသဖြင့် အရောင်းအဝယ်တိုင်းကို `risk_source=unavailable` အဖြစ် "
               "သတ်မှတ်ကာ R metrics မှ ဖယ်ထုတ်ထားသည်။ label တပ်ထားသော hard-stop "
               "*proxy* (စစ်မှန်သော Minervini R မဟုတ်) အတွက် `--proxy-risk` ကို "
               "သုံးပါ။"),
    },
    "md_disclaimer_proxy": {
        "en": ("**PROXY R - not true Minervini R.** initial_risk is a fixed hard-stop "
               "fraction (proxy_pct={proxy_pct}), NOT a real VCP setup stop. Treat "
               "these R values as a rough proxy only."),
        "my": ("**PROXY R - not true Minervini R.** initial_risk is a fixed hard-stop "
               "fraction (proxy_pct={proxy_pct}), NOT a real VCP setup stop. Treat "
               "these R values as a rough proxy only. — **PROXY R - စစ်မှန်သော "
               "Minervini R မဟုတ်ပါ။** initial_risk သည် သတ်မှတ် hard-stop အချိုး "
               "(proxy_pct={proxy_pct}) ဖြစ်ပြီး စစ်မှန်သော VCP setup stop မဟုတ်ပါ။ "
               "ဤ R တန်ဖိုးများကို ခန့်မှန်း proxy အဖြစ်သာ သဘောထားပါ။"),
    },
    # ── Expectancy report: console labels (display only) ─────────────────────
    "cli_exp_header": {
        "en": "Expectancy / R-Multiple Report (read-only)",
        "my": "မျှော်မှန်းတန်ဖိုး / R-Multiple အစီရင်ခံစာ (ဖတ်ရှုရုံသာ)",
    },
    "cli_exp_source": {
        "en": "Source: {path}",
        "my": "အရင်းအမြစ်: {path}",
    },
    "cli_exp_rows": {
        "en": "Rows read: {n}",
        "my": "ဖတ်ထားသော အတန်း: {n}",
    },
    "cli_exp_risk_mode": {
        "en": "Risk mode: {mode}{suffix}",
        "my": "Risk mode: {mode}{suffix}",
    },
    "cli_exp_trades": {
        "en": "Closed trades included / excluded: {inc} / {exc}",
        "my": "ပိတ်ထားသော အရောင်းအဝယ် ပါဝင် / ဖယ်ထုတ်: {inc} / {exc}",
    },
    "cli_exp_total_pnl": {
        "en": "Total realized PnL (return units): {v}",
        "my": "စုစုပေါင်း အကောင်အထည်ဖော် PnL (return ယူနစ်): {v}",
    },
    "cli_exp_wls": {
        "en": "Wins / Losses / Scratch: {w} / {l} / {s}",
        "my": "နိုင် / ရှုံး / သရေ: {w} / {l} / {s}",
    },
    "cli_exp_win_rate": {
        "en": "Win rate: {v}",
        "my": "နိုင်နှုန်း: {v}",
    },
    "cli_exp_profit_factor": {
        "en": "Profit factor: {v}",
        "my": "အမြတ် factor: {v}",
    },
    "cli_exp_expectancy_na": {
        "en": "Expectancy (R): n/a (no valid initial risk; use --proxy-risk for a labeled proxy)",
        "my": ("Expectancy (R): n/a (စစ်မှန်သော initial risk မရှိပါ; label တပ် proxy "
               "အတွက် --proxy-risk ကို သုံးပါ)"),
    },
    "cli_exp_expectancy_val": {
        "en": "Expectancy (R): {v} over {n} trades",
        "my": "Expectancy (R): {v} ({n} အရောင်းအဝယ်အပေါ်)",
    },
    "cli_exp_excluded": {
        "en": "Excluded by reason: {pairs}",
        "my": "အကြောင်းအရင်းအလိုက် ဖယ်ထုတ်: {pairs}",
    },
    "cli_exp_reports_written": {
        "en": "Reports written -> {jp}",
        "my": "အစီရင်ခံစာများ ရေးသားပြီး -> {jp}",
    },
    # ── Shared safety line (BILINGUAL in my) ─────────────────────────────────
    "no_ibkr_no_orders": {
        "en": "No IBKR connection was made and no orders were placed.",
        "my": ("No IBKR connection was made and no orders were placed. / IBKR "
               "ချိတ်ဆက်မှု မပြုလုပ်ခဲ့ပါ၊ အော်ဒါ မည်သည့်အရာမျှ မတင်ခဲ့ပါ။"),
    },
    # ── Coach intro (display only; safety line BILINGUAL) ────────────────────
    "coach_intro_title": {
        "en": "Guided Paper Trading Coach",
        "my": "လမ်းညွှန် Paper Trading နည်းပြ",
    },
    "coach_intro_learn": {
        "en": "A learning tool: it explains signals and previews PAPER trades.",
        "my": "သင်ယူရေး ကိရိယာ: signal များကို ရှင်းပြပြီး PAPER trade များကို ကြိုတင်ပြသသည်။",
    },
    "coach_intro_safety": {
        "en": "It never auto-buys. A paper order needs BOTH --confirm AND --chart-checked.",
        "my": ("It never auto-buys. A paper order needs BOTH --confirm AND "
               "--chart-checked. / အလိုအလျောက် ဘယ်တော့မှ မဝယ်ပါ။ paper order "
               "တစ်ခုအတွက် --confirm နှင့် --chart-checked နှစ်ခုစလုံး လိုအပ်သည်။"),
    },
}
