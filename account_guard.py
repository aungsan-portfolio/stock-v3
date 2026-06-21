"""
account_guard.py - Phase 6.1: account-type assertion (paper ``DU`` / live ``U``).

PURE and offline like data_integrity.py / reconciliation.py / shutdown_guard.py /
reconnect_watchdog.py: NO ib_insync, NO network, NO clock, NO order placement.
The bridge reads ``ib.managedAccounts()`` and hands the plain list to
``assert_account``; this module decides whether the connected account matches the
expected ENVIRONMENT and FAILS CLOSED on anything empty, malformed, ambiguous, or
wrong-prefix.

Why this guard exists (the gap the paper-port lock does NOT cover)
------------------------------------------------------------------
The paper-port lock (``IBKRBridge.connect`` step 1) guards the *port* (7497). It
does NOT guard *which account* is logged into TWS/Gateway on that port. If a live
(``U...``) account is logged into Gateway on 7497, every existing guard passes and
a real-money order could be placed. This module closes that gap by asserting the
account id's environment prefix:

    paper -> the account id must start with ``DU`` (IBKR paper accounts)
    live  -> the account id must start with ``U``  (IBKR live accounts) AND match
             the explicitly-configured ``LIVE_ACCOUNT_ID``

Fail-closed rules (all unit-tested in test_account_guard.py)
------------------------------------------------------------
  * empty ``managedAccounts()``          -> fail (REASON_EMPTY)
  * any malformed account id             -> fail (REASON_MALFORMED)
  * >1 account and no explicit expected  -> fail (REASON_AMBIGUOUS); we NEVER pick
                                            one silently, so a live account in the
                                            list can never be auto-selected
  * explicit expected not present        -> fail (REASON_EXPECTED_MISSING)
  * wrong environment prefix             -> fail (REASON_WRONG_PREFIX); e.g. a
                                            ``U...`` live account on the paper port

LIVE MODE IS INERT IN PHASE 6.1
-------------------------------
``live_mode_enabled`` returns False for the shipped paper config, so the bridge
ALWAYS asserts a paper (``DU``) account. The live branch is fully implemented and
tested so Phase 6.2+ can wire it, but nothing in this phase passes
``live_mode=True`` from the bridge, sets ``LIVE_ACCOUNT_ID``, or otherwise enables
live trading. This module adds NO live-readiness capability flag and NEVER writes
config.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config

logger = logging.getLogger(__name__)

# ── Environment prefixes ─────────────────────────────────────────────────────
PAPER_PREFIX = "DU"   # IBKR paper accounts: DU + alphanumeric id
LIVE_PREFIX = "U"     # IBKR live accounts:  U  + digits

MODE_PAPER = "paper"
MODE_LIVE = "live"

# Audit stage for order_audit.log_event (order_audit tolerates free-form stages,
# so this lives here and does NOT require touching order_audit.py).
AUDIT_STAGE = "account_assertion"
# ConnectionHealth reason recorded by the bridge when the assertion fails closed.
HEALTH_REASON = "account_assertion_failed"

# ── Machine reason codes (recorded on the assertion + the audit log) ─────────
REASON_OK = "ok"
REASON_EMPTY = "empty_managed_accounts"
REASON_MALFORMED = "malformed_account_id"
REASON_AMBIGUOUS = "ambiguous_multiple_accounts"
REASON_EXPECTED_MISSING = "expected_account_not_found"
REASON_WRONG_PREFIX = "wrong_account_prefix"

# A well-formed IBKR account id: 1-4 leading uppercase letters then >=3 digits
# (e.g. DU1234567, U1234567). Anything else is treated as malformed -> fail closed.
_ACCOUNT_RE = re.compile(r"^[A-Z]{1,4}[0-9]{3,}$")
# Environment-prefix tests require the prefix to be IMMEDIATELY followed by an
# account-id character. Paper accounts may include a letter after DU (for example
# DUQ710212), while live accounts remain U + digit so "UABC123" cannot masquerade
# as a live account.
_PAPER_PREFIX_RE = re.compile(r"^DU[A-Z0-9]")
_LIVE_PREFIX_RE = re.compile(r"^U[0-9]")


@dataclass(frozen=True)
class AccountAssertion:
    """The verdict on the broker's managed accounts for a connect.

    ``ok`` is True only when a single account has been unambiguously identified
    AND it carries the expected environment prefix. On a reject ``account`` is
    None, ``reason`` is one of the REASON_* codes, and ``detail`` explains why so
    the caller can audit, log, and FAIL CLOSED.
    """
    ok: bool
    mode: str                          # MODE_PAPER | MODE_LIVE
    account: Optional[str]             # the verified account id (None on reject)
    reason: str                        # REASON_* code
    detail: str                        # human-readable explanation
    accounts: Tuple[str, ...] = field(default_factory=tuple)  # normalized list seen


# ── Normalization / validation (pure) ────────────────────────────────────────
def normalize_account(value) -> str:
    """Coerce one account id to a canonical UPPER, stripped string (``""`` if
    unusable). Never raises."""
    if value is None:
        return ""
    try:
        return str(value).strip().upper()
    except Exception:
        return ""


def normalize_accounts(managed_accounts) -> List[str]:
    """Normalize ``ib.managedAccounts()`` to a list of non-empty UPPER ids.

    Accepts the usual ``list[str]`` and also the comma-joined string form some
    ib_insync versions return ("DU1,DU2"). Blank/None tokens (e.g. a trailing
    comma artifact) are dropped; every RETAINED token is still validated and
    prefix-checked downstream, so dropping blanks can never let a wrong account
    slip through. A non-iterable input yields ``[]`` (-> fail closed as empty).
    """
    if managed_accounts is None:
        return []
    if isinstance(managed_accounts, str):
        raw = managed_accounts.split(",")
    else:
        try:
            raw = list(managed_accounts)
        except TypeError:
            return []
    out: List[str] = []
    for item in raw:
        acct = normalize_account(item)
        if acct:
            out.append(acct)
    return out


def is_well_formed(account) -> bool:
    """True when ``account`` looks like a real IBKR account id (letters+digits)."""
    return bool(_ACCOUNT_RE.match(normalize_account(account)))


def has_environment_prefix(account, *, live: bool) -> bool:
    """True when ``account`` carries the expected environment prefix.

    paper -> ``^DU[A-Z0-9]``   live -> ``^U[0-9]``. Paper accounts may include
    a letter after ``DU``; live accounts still require ``U`` followed by a digit
    so stray ids like "UABC123" never pass live mode."""
    acct = normalize_account(account)
    if not acct:
        return False
    rx = _LIVE_PREFIX_RE if live else _PAPER_PREFIX_RE
    return bool(rx.match(acct))


# ── The assertion (pure; the heart of the guard) ─────────────────────────────
def assert_account(managed_accounts, *, live_mode: bool = False,
                   expected_account=None) -> AccountAssertion:
    """Decide whether the broker's managed accounts are safe to trade on.

    PURE: no IB, no I/O, no order placement. The bridge maps the result onto its
    ConnectionHealth + audit log and FAILS CLOSED (refuses the connection) on any
    ``ok=False`` verdict. ``live_mode`` is INERT in Phase 6.1 (the bridge always
    passes False); the live branch is implemented for Phase 6.2+ and is unit
    tested here.
    """
    mode = MODE_LIVE if live_mode else MODE_PAPER
    prefix = LIVE_PREFIX if live_mode else PAPER_PREFIX
    accounts = normalize_accounts(managed_accounts)
    expected = normalize_account(expected_account) or None

    def fail(reason: str, detail: str, account: Optional[str] = None) -> AccountAssertion:
        return AccountAssertion(False, mode, account, reason, detail, tuple(accounts))

    # (5) Empty managed accounts -> fail closed.
    if not accounts:
        return fail(REASON_EMPTY,
                    "broker returned no managed accounts; refusing to proceed")

    # (6) Any malformed id -> fail closed (never trust a garbled account list).
    malformed = [a for a in accounts if not is_well_formed(a)]
    if malformed:
        return fail(REASON_MALFORMED,
                    f"malformed managed account id(s): {malformed}")

    # Live mode ALWAYS requires an explicit expected account (LIVE_ACCOUNT_ID):
    # we never auto-select a live account, even when only one is present.
    if live_mode and expected is None:
        return fail(REASON_EXPECTED_MISSING,
                    "live mode requires an explicit expected account "
                    "(LIVE_ACCOUNT_ID) but none is configured")

    # (4) Multiple accounts -> require an explicit expected account; NEVER pick
    # one silently (so a live account in the list can never be auto-selected).
    if len(accounts) > 1:
        if expected is None:
            return fail(REASON_AMBIGUOUS,
                        f"{len(accounts)} managed accounts {list(accounts)} but no "
                        "explicit expected account configured")
        if expected not in accounts:
            return fail(REASON_EXPECTED_MISSING,
                        f"expected account {expected} not among managed accounts "
                        f"{list(accounts)}")
        selected = expected
    else:
        selected = accounts[0]
        # A configured expected account must still match the single account.
        if expected is not None and expected != selected:
            return fail(REASON_EXPECTED_MISSING,
                        f"expected account {expected} not among managed accounts "
                        f"{list(accounts)}")

    # (2/3) Environment-prefix check -> fail closed on the wrong environment
    # (e.g. a live "U..." account connected on the paper port).
    if not has_environment_prefix(selected, live=live_mode):
        return fail(REASON_WRONG_PREFIX,
                    f"account {selected} is not a {mode} account "
                    f"(expected '{prefix}' prefix)", account=selected)

    return AccountAssertion(True, mode, selected, REASON_OK,
                            f"{mode} account {selected} verified ('{prefix}' prefix)",
                            tuple(accounts))


# ── Mode resolution (paper by default; live is INERT this phase) ─────────────
def live_mode_enabled(cfg=config) -> bool:
    """True ONLY when the configuration UNAMBIGUOUSLY opts into live trading.

    The shipped paper config returns False, so the bridge always asserts a paper
    (``DU``) account. This function NEVER itself enables live trading - it only
    REPORTS whether live has been deliberately configured (a future phase must
    turn on the live flag, drop the paper-port lock, AND set ``LIVE_ACCOUNT_ID``).
    Any one of those missing keeps us in paper mode (fail-safe default)."""
    if not bool(getattr(cfg, "COACH_LIVE_TRADING_ENABLED", False)):
        return False
    if bool(getattr(cfg, "REQUIRE_PAPER_PORT", True)):
        return False
    if not normalize_account(getattr(cfg, "LIVE_ACCOUNT_ID", None)):
        return False
    return True


# ── Audit / log helpers ──────────────────────────────────────────────────────
def audit_fields(assertion: AccountAssertion) -> dict:
    """JSON-safe fields for order_audit.log_event (stage = AUDIT_STAGE)."""
    return {
        "ok": bool(assertion.ok),
        "mode": assertion.mode,
        "account": assertion.account,
        "reason": assertion.reason,
        "accounts": list(assertion.accounts),
        "detail": assertion.detail,
    }


def summary_line(assertion: AccountAssertion) -> str:
    """One-line human summary for the standard logger."""
    status = "OK" if assertion.ok else "FAIL"
    return (f"{status} mode={assertion.mode} account={assertion.account} "
            f"reason={assertion.reason} accounts={list(assertion.accounts)}")
