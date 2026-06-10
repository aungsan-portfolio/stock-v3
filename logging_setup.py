"""
logging_setup.py — Rotating file + console logger.

Production fixes:
- Idempotent: clears existing handlers so repeated calls don't duplicate output.
- Quiets noisy third-party loggers (yfinance, urllib3, ib_insync) at INFO+.
"""
import logging
from logging.handlers import RotatingFileHandler

import config


def setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
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

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    root.setLevel(logging.DEBUG)
    root.addHandler(console)
    root.addHandler(file_handler)

    # Quiet noisy third-party libs.
    for noisy in ("yfinance", "urllib3", "peewee", "ib_insync.wrapper", "ib_insync.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
