"""test_coach_i18n.py - offline tests for the M5A DISPLAY-ONLY localization layer.

Pure unittest: no network, no IBKR, no order path, no orders. All file IO uses
temp dirs. Proves that language is a display concern only - it never changes the
JSON schema, numeric metrics, or file paths, and English is always the fallback.

    python -m unittest test_coach_i18n
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import config
import coach_i18n
import expectancy


def _has_burmese(s: str) -> bool:
    """True if ``s`` contains any character in the Myanmar Unicode block."""
    return any(0x1000 <= ord(ch) <= 0x109F for ch in s)


class _ConfigLangGuard(unittest.TestCase):
    """Base class: save/restore config.MINERVINI_COACH_LANGUAGE around each test."""

    def setUp(self):
        self._orig_lang = config.MINERVINI_COACH_LANGUAGE
        config.MINERVINI_COACH_LANGUAGE = "en"

    def tearDown(self):
        config.MINERVINI_COACH_LANGUAGE = self._orig_lang


class TestResolveLanguage(_ConfigLangGuard):
    def test_default_is_english(self):
        config.MINERVINI_COACH_LANGUAGE = "en"
        self.assertEqual(coach_i18n.resolve_language(None), "en")

    def test_config_used_when_no_cli(self):
        config.MINERVINI_COACH_LANGUAGE = "my"
        self.assertEqual(coach_i18n.resolve_language(None), "my")

    def test_cli_overrides_config(self):
        config.MINERVINI_COACH_LANGUAGE = "my"
        # Explicit CLI English must win over a Burmese config default.
        self.assertEqual(coach_i18n.resolve_language("en"), "en")

    def test_cli_my_over_config_en(self):
        config.MINERVINI_COACH_LANGUAGE = "en"
        self.assertEqual(coach_i18n.resolve_language("my"), "my")

    def test_unknown_cli_falls_back_to_english(self):
        config.MINERVINI_COACH_LANGUAGE = "en"
        self.assertEqual(coach_i18n.resolve_language("klingon"), "en")

    def test_unknown_config_falls_back_to_english(self):
        config.MINERVINI_COACH_LANGUAGE = "zz"
        self.assertEqual(coach_i18n.resolve_language(None), "en")

    def test_normalization_case_and_whitespace(self):
        self.assertEqual(coach_i18n.resolve_language("  MY "), "my")
        self.assertEqual(coach_i18n.resolve_language("EN"), "en")

    def test_supported_languages_constant(self):
        self.assertEqual(set(coach_i18n.SUPPORTED_LANGUAGES), {"en", "my"})
        self.assertEqual(coach_i18n.DEFAULT_LANGUAGE, "en")


class TestTranslate(_ConfigLangGuard):
    def test_known_key_english(self):
        self.assertEqual(coach_i18n.t("md_h_notes", "en"), "Notes")

    def test_known_key_burmese_differs_and_is_burmese(self):
        s = coach_i18n.t("md_h_notes", "my")
        self.assertNotEqual(s, "Notes")
        self.assertTrue(_has_burmese(s))

    def test_unknown_language_falls_back_to_english(self):
        self.assertEqual(coach_i18n.t("md_h_notes", "xx"), "Notes")

    def test_unknown_key_returns_key(self):
        self.assertEqual(coach_i18n.t("no_such_key_123", "en"), "no_such_key_123")

    def test_none_key_does_not_raise(self):
        self.assertIsInstance(coach_i18n.t(None, "en"), str)

    def test_missing_format_kwarg_does_not_raise(self):
        # Template references {proxy_pct} but we deliberately omit it.
        out = coach_i18n.t("md_disclaimer_proxy", "en")
        self.assertIn("PROXY R", out)  # returned safely, no exception

    def test_format_kwarg_is_applied(self):
        out = coach_i18n.t("md_disclaimer_proxy", "en", proxy_pct=0.03)
        self.assertIn("0.03", out)

    def test_lang_none_consults_config(self):
        config.MINERVINI_COACH_LANGUAGE = "my"
        self.assertTrue(_has_burmese(coach_i18n.t("md_h_notes", None)))
        config.MINERVINI_COACH_LANGUAGE = "en"
        self.assertEqual(coach_i18n.t("md_h_notes", None), "Notes")

    def test_safety_disclaimer_is_bilingual_in_my(self):
        # Critical safety strings must keep the English text in Burmese mode.
        out = coach_i18n.t("md_disclaimer_none", "my")
        self.assertIn("NOT computed", out)          # English preserved
        self.assertTrue(_has_burmese(out))          # Burmese added
        self.assertIn("`risk_source=unavailable`", out)  # code token unchanged

    def test_coach_safety_keeps_cli_tokens_and_english(self):
        out = coach_i18n.t("coach_intro_safety", "my")
        self.assertIn("--confirm", out)
        self.assertIn("--chart-checked", out)
        self.assertIn("never auto-buys", out)       # English safety preserved
        self.assertTrue(_has_burmese(out))

    def test_no_ibkr_line_bilingual_in_my_english_in_en(self):
        en = coach_i18n.t("no_ibkr_no_orders", "en")
        self.assertEqual(en, "No IBKR connection was made and no orders were placed.")
        my = coach_i18n.t("no_ibkr_no_orders", "my")
        self.assertIn("No IBKR connection was made and no orders were placed.", my)
        self.assertTrue(_has_burmese(my))


# ── Expectancy localization: display-only, schema/metrics untouched ──────────
def _write_round_trip_csv(tmp: str) -> Path:
    """One clean flat -> long -> flat round-trip (a single winning trade)."""
    text = (
        "symbol,date,fold_start,order_executed,old_position,new_position,close_price,equity\n"
        "SPY,2022-01-03,252,True,0,1,100.0,1.0\n"
        "SPY,2022-01-10,252,True,1,0,106.0,1.06\n"
    )
    p = Path(tmp) / "backtest_trades.csv"
    p.write_text(text, encoding="utf-8")
    return p


_EXPECTED_JSON_KEYS = {
    "source", "source_file", "n_rows_read", "pnl_units", "risk_mode",
    "proxy_pct", "metrics", "trades", "notes",
}


class TestExpectancyLocalization(_ConfigLangGuard):
    def test_default_english_markdown_keeps_current_substrings(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = expectancy.generate_report(_write_round_trip_csv(tmp))
            md = expectancy.report_to_markdown(report)  # default language
            self.assertIn("# Expectancy / R-Multiple Report", md)
            self.assertIn("## Realized performance (PnL-based)", md)
            self.assertIn("## Notes", md)
            self.assertIn("NOT computed", md)  # default-off R disclaimer present

    def test_rendering_does_not_mutate_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = expectancy.generate_report(_write_round_trip_csv(tmp))
            before = json.dumps(report, indent=2, sort_keys=True)
            expectancy.report_to_markdown(report, lang="my")
            expectancy.report_to_markdown(report, lang="en")
            after = json.dumps(report, indent=2, sort_keys=True)
            self.assertEqual(before, after)

    def test_json_keys_unchanged_across_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_round_trip_csv(tmp)
            report = expectancy.generate_report(csv_path)
            jp_en, mp_en = Path(tmp) / "en.json", Path(tmp) / "en.md"
            jp_my, mp_my = Path(tmp) / "my.json", Path(tmp) / "my.md"
            expectancy.write_report(report, json_path=jp_en, md_path=mp_en, lang="en")
            expectancy.write_report(report, json_path=jp_my, md_path=mp_my, lang="my")
            j_en = json.loads(jp_en.read_text(encoding="utf-8"))
            j_my = json.loads(jp_my.read_text(encoding="utf-8"))
            # JSON is identical regardless of display language (schema is stable).
            self.assertEqual(j_en, j_my)
            self.assertEqual(set(j_en.keys()), _EXPECTED_JSON_KEYS)
            # The JSON bytes are byte-for-byte identical too.
            self.assertEqual(jp_en.read_bytes(), jp_my.read_bytes())

    def test_language_does_not_affect_numeric_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = expectancy.generate_report(_write_round_trip_csv(tmp))
            md_en = expectancy.report_to_markdown(report, lang="en")
            md_my = expectancy.report_to_markdown(report, lang="my")
            # The single round-trip realizes +0.06; that number is unchanged.
            self.assertAlmostEqual(report["metrics"]["total_realized_pnl"], 0.06)
            self.assertIn("0.06", md_en)
            self.assertIn("0.06", md_my)
            # Win/loss counts identical and language-independent.
            self.assertEqual(report["metrics"]["wins"], 1)
            self.assertIn("1 / 0 / 0", md_en)
            self.assertIn("1 / 0 / 0", md_my)

    def test_file_paths_not_localized(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_round_trip_csv(tmp)
            report = expectancy.generate_report(csv_path)
            md_my = expectancy.report_to_markdown(report, lang="my")
            # The source path is rendered verbatim even in Burmese mode.
            self.assertIn(str(csv_path), md_my)

    def test_burmese_markdown_roundtrips_as_utf8_and_stays_bilingual(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_round_trip_csv(tmp)
            report = expectancy.generate_report(csv_path)
            jp = Path(tmp) / "report_my.json"
            mp = Path(tmp) / "report_my.md"
            expectancy.write_report(report, json_path=jp, md_path=mp, lang="my")
            md = mp.read_text(encoding="utf-8")  # must decode cleanly as UTF-8
            self.assertTrue(_has_burmese(md))            # Burmese present
            self.assertIn("NOT computed", md)            # English safety preserved
            self.assertIn("Minervini R", md)             # English safety preserved
            # The sibling JSON remains valid + English-schema.
            loaded = json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(loaded["source"], "backtest_ledger")

    def test_proxy_disclaimer_bilingual_with_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_round_trip_csv(tmp)
            report = expectancy.generate_report(csv_path, enable_proxy_risk=True, proxy_pct=0.03)
            md_my = expectancy.report_to_markdown(report, lang="my")
            self.assertIn("PROXY R", md_my)              # English preserved
            self.assertIn("not true Minervini R", md_my)  # English preserved
            self.assertIn("0.03", md_my)                 # numeric proxy_pct unchanged
            self.assertTrue(_has_burmese(md_my))


class TestI18nModuleIsDisplayOnly(unittest.TestCase):
    """Guardrail: the i18n layer must never reach the order path / decision code."""

    def test_module_only_imports_config_and_stdlib(self):
        import inspect
        src = inspect.getsource(coach_i18n)
        for forbidden in ("ibkr_bridge", "order_exec", "order_audit", "risk_state",
                          "account_guard", "ib_insync", "reconciliation", "minervini",
                          "trade_coach"):
            self.assertNotIn(f"import {forbidden}", src)
            self.assertNotIn(f"from {forbidden}", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
