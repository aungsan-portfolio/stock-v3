from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import config


class ProcessAccountFilter(logging.Filter):
    """Injects process_tag and account_tag into every LogRecord."""
    process_tag: str = "SYSTEM"
    account_tag: str = "MAIN"

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "process_tag"):
            record.process_tag = ProcessAccountFilter.process_tag
        if not hasattr(record, "account_tag"):
            record.account_tag = ProcessAccountFilter.account_tag
        return True


def _sanitize_account_tag(tag: Optional[str]) -> str:
    """Sanitize account tag to prevent API keys or secrets from leaking into logs."""
    if not tag:
        return "MAIN"
    tag_str = str(tag).strip()
    upper_tag = tag_str.upper()
    # 1. Filter to only alphanumeric + hyphen/underscore
    cleaned = "".join(c for c in upper_tag if c.isalnum() or c in ("-", "_"))
    if not cleaned:
        return "UNKNOWN"
    # 2. Block obvious key patterns
    if cleaned.startswith("PK") or cleaned.startswith("SK") or "SECRET" in cleaned or "KEY" in cleaned:
        return f"{cleaned[:4]}...MASKED"
    # 3. Check length: Alpaca account numbers are 12 chars (e.g. PA3UODIC5JTA). Long strings are masked.
    if len(cleaned) > 16:
        return f"{cleaned[:4]}...{cleaned[-4:]}"
    return cleaned


def set_log_context(
    process_tag: Optional[str] = None,
    account_tag: Optional[str] = None,
) -> None:
    """Dynamically update process_tag and/or account_tag across all future logs."""
    if process_tag:
        clean_proc = str(process_tag).strip().upper()
        clean_proc = "".join(c for c in clean_proc if c.isalnum() or c in ("-", "_")) or "SYSTEM"
        ProcessAccountFilter.process_tag = clean_proc
        os.environ["STOCK_ENGINE_PROCESS_TAG"] = clean_proc
    if account_tag:
        clean_acct = _sanitize_account_tag(account_tag)
        ProcessAccountFilter.account_tag = clean_acct
        os.environ["STOCK_ENGINE_ACCOUNT_TAG"] = clean_acct


class SafeRotatingFileHandler(RotatingFileHandler):
    """Windows-safe RotatingFileHandler that catches PermissionError during rollover
    and suppresses Python 3.14 reentrant signal logging errors cleanly."""
    def doRollover(self):
        try:
            super().doRollover()
        except (PermissionError, Exception):
            pass

    def emit(self, record):
        try:
            super().emit(record)
        except RuntimeError as e:
            if "reentrant" in str(e):
                pass
            else:
                raise
        except Exception:
            self.handleError(record)


def setup_logging(
    process_tag: Optional[str] = None,
    account_tag: Optional[str] = None,
    log_file: Optional[Path | str] = None,
) -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize log context
    initial_proc = process_tag or os.environ.get("STOCK_ENGINE_PROCESS_TAG", "SYSTEM")
    initial_acct = account_tag or os.environ.get("STOCK_ENGINE_ACCOUNT_TAG", "MAIN")
    set_log_context(process_tag=initial_proc, account_tag=initial_acct)

    # Determine dedicated log file
    target_log_file: Path
    if log_file:
        target_log_file = Path(log_file)
    else:
        p_upper = initial_proc.upper()
        if p_upper == "DAYTRADE":
            target_log_file = config.LOG_DIR / "daytrade_engine.log"
        elif p_upper == "SWING":
            target_log_file = config.LOG_DIR / "swing_engine.log"
        else:
            target_log_file = config.LOG_FILE

    target_log_file.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(process_tag)s:%(account_tag)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    # Idempotent: drop any existing handlers from previous setup_logging() calls.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    ctx_filter = ProcessAccountFilter()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    console.addFilter(ctx_filter)

    file_handler = SafeRotatingFileHandler(
        target_log_file,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    file_handler.addFilter(ctx_filter)

    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy third-party libs.
    for noisy in ("yfinance", "urllib3", "peewee", "ib_insync.wrapper", "ib_insync.client"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
