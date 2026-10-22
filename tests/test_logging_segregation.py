"""
tests/test_logging_segregation.py — Regression Test Suite for Log Isolation and Security.

Verifies 5 mandatory logging invariants:
1. Daytrade logs route strictly to daytrade_engine.log with [DAYTRADE:<ACCT>] tag.
2. Swing logs route strictly to swing_engine.log with [SWING:<ACCT>] tag.
3. Idempotent setup_logging() calls do not generate duplicate handlers or repeated emissions.
4. account_tag strictly scrubs API keys, secrets, and malicious injection characters.
5. Default log file routing automatically isolates Daytrade and Swing targets.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import pytest

import config
from logging_setup import (
    ProcessAccountFilter,
    _sanitize_account_tag,
    set_log_context,
    setup_logging,
)


@pytest.fixture(autouse=True)
def restore_logging():
    """Ensure logging handlers and context are cleanly reset after each test."""
    orig_proc = ProcessAccountFilter.process_tag
    orig_acct = ProcessAccountFilter.account_tag
    yield
    ProcessAccountFilter.process_tag = orig_proc
    ProcessAccountFilter.account_tag = orig_acct
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass


def test_daytrade_logging_isolation(tmp_path: Path):
    """
    Invariant 1: Daytrade logs are isolated in daytrade_engine.log with correct [DAYTRADE:<ACCT>] tag.
    """
    dt_log = tmp_path / "daytrade_engine.log"
    sw_log = tmp_path / "swing_engine.log"

    setup_logging(process_tag="DAYTRADE", account_tag="PA3UODIC5JTA", log_file=dt_log)
    logger = logging.getLogger("test_daytrade_logger")
    logger.info("Executing Daytrade signal evaluation cycle 1")

    # Flush handlers
    for h in logging.getLogger().handlers:
        h.flush()

    assert dt_log.exists()
    dt_content = dt_log.read_text(encoding="utf-8")
    assert "[DAYTRADE:PA3UODIC5JTA]" in dt_content
    assert "Executing Daytrade signal evaluation cycle 1" in dt_content
    assert not sw_log.exists()


def test_swing_logging_isolation(tmp_path: Path):
    """
    Invariant 2: Swing logs are isolated in swing_engine.log with correct [SWING:<ACCT>] tag.
    """
    dt_log = tmp_path / "daytrade_engine.log"
    sw_log = tmp_path / "swing_engine.log"

    setup_logging(process_tag="SWING", account_tag="PA3R71I7JFDZ", log_file=sw_log)
    logger = logging.getLogger("test_swing_logger")
    logger.info("Executing Swing multi-day position scan")

    # Flush handlers
    for h in logging.getLogger().handlers:
        h.flush()

    assert sw_log.exists()
    sw_content = sw_log.read_text(encoding="utf-8")
    assert "[SWING:PA3R71I7JFDZ]" in sw_content
    assert "Executing Swing multi-day position scan" in sw_content
    assert not dt_log.exists()


def test_logging_setup_idempotency_no_duplicate_handlers(tmp_path: Path):
    """
    Invariant 3: Repeated setup_logging() calls replace handlers cleanly without duplicates.
    """
    test_log = tmp_path / "idempotent.log"

    # Call setup_logging 3 times in a row
    setup_logging(process_tag="SYSTEM", account_tag="MAIN", log_file=test_log)
    setup_logging(process_tag="SYSTEM", account_tag="MAIN", log_file=test_log)
    setup_logging(process_tag="SYSTEM", account_tag="MAIN", log_file=test_log)

    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, RotatingFileHandler)]
    console_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler)]

    assert len(file_handlers) == 1
    assert len(console_handlers) == 1

    logger = logging.getLogger("test_idempotent_logger")
    logger.info("Idempotent log message verification")

    for h in root.handlers:
        h.flush()

    content = test_log.read_text(encoding="utf-8")
    occurrences = content.count("Idempotent log message verification")
    assert occurrences == 1, f"Expected exactly 1 occurrence, found {occurrences}"


def test_account_tag_scrubbing_and_sanitization():
    """
    Invariant 4: Sensitive API keys, secrets, and illegal injection chars are scrubbed from account_tag.
    """
    # 1. Legitimate Alpaca sub-account numbers are preserved
    assert _sanitize_account_tag("PA3UODIC5JTA") == "PA3UODIC5JTA"
    assert _sanitize_account_tag("PA3R71I7JFDZ") == "PA3R71I7JFDZ"
    assert _sanitize_account_tag("DAYTRADE_ACCT") == "DAYTRADE_ACCT"

    # 2. Alpaca API key (20 chars, PK...) is masked
    api_key = "PKIM1234567890123456"
    sanitized_key = _sanitize_account_tag(api_key)
    assert sanitized_key != api_key
    assert "..." in sanitized_key
    assert api_key not in sanitized_key

    # 3. Secret keys / token strings (> 16 chars or containing SECRET) are masked
    secret = "sk_live_9876543210abcdef123456"
    sanitized_secret = _sanitize_account_tag(secret)
    assert sanitized_secret != secret
    assert secret not in sanitized_secret

    token_secret = "MY_SECRET_TOKEN"
    sanitized_token = _sanitize_account_tag(token_secret)
    assert "MASKED" in sanitized_token

    # 4. None / empty defaults to "MAIN"
    assert _sanitize_account_tag(None) == "MAIN"
    assert _sanitize_account_tag("") == "MAIN"

    # 5. Injection characters are stripped
    assert _sanitize_account_tag("ACCT;DROP TABLE;--") == "ACCTDROPTABLE--"

    # 6. Verify set_log_context applies sanitization
    set_log_context(account_tag="PKIM9999888877776666")
    assert ProcessAccountFilter.account_tag != "PKIM9999888877776666"
    assert "..." in ProcessAccountFilter.account_tag


def test_default_log_file_routing(monkeypatch, tmp_path: Path):
    """
    Invariant 5: Default file routing selects daytrade_engine.log vs swing_engine.log automatically.
    """
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)

    # 1. Daytrade default
    setup_logging(process_tag="DAYTRADE")
    assert any(h.baseFilename.endswith("daytrade_engine.log") for h in logging.getLogger().handlers if hasattr(h, "baseFilename"))

    # 2. Swing default
    setup_logging(process_tag="SWING")
    assert any(h.baseFilename.endswith("swing_engine.log") for h in logging.getLogger().handlers if hasattr(h, "baseFilename"))
