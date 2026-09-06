"""
strategies/startup_security.py — Production Startup Security & Configuration Integrity

Enforces fail-closed configuration invariants and cryptographic credential fingerprinting
prior to trading engine initialization.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Dict, Optional


class StartupSafetyRejected(RuntimeError):
    """Raised when runtime configuration fails startup safety validation."""
    pass


def require_startup(condition: bool, message: str) -> None:
    """Fail-closed assertion helper for engine startup."""
    if not condition:
        raise StartupSafetyRejected(message)


def credential_fingerprint(
    api_key: str,
    api_secret: str,
    credential_verification_key: bytes,
) -> str:
    """
    Generate a secure HMAC-SHA256 fingerprint over complete credentials.
    Never exposes raw keys or reversible fragments.
    """
    require_startup(bool(api_key), "API key is missing or empty")
    require_startup(bool(api_secret), "API secret is missing or empty")
    require_startup(bool(credential_verification_key), "Credential verification key is missing or empty")

    credential_payload = api_key.encode("utf-8") + b"\x00" + api_secret.encode("utf-8")
    return hmac.new(credential_verification_key, credential_payload, hashlib.sha256).hexdigest()


def verify_authenticated_account_identity(
    configured_account_id: str,
    authenticated_api_account_id: str,
    permitted_account_id: str,
) -> None:
    """
    Tri-part verification ensuring:
      Configured Account ID == Authenticated API Account ID == Strategy-Permitted Account ID.
    Does not expose sensitive IDs in exception messages.
    """
    require_startup(bool(configured_account_id), "Configured account ID missing")
    require_startup(bool(authenticated_api_account_id), "Authenticated account ID missing")
    require_startup(bool(permitted_account_id), "Permitted account ID missing")

    c_norm = configured_account_id.strip()
    a_norm = authenticated_api_account_id.strip()
    p_norm = permitted_account_id.strip()

    require_startup(
        hmac.compare_digest(c_norm, p_norm),
        "STARTUP_ACCOUNT_POLICY_MISMATCH: Configured account does not match policy",
    )
    require_startup(
        hmac.compare_digest(a_norm, p_norm),
        "STARTUP_ACCOUNT_AUTHENTICATION_MISMATCH: Authenticated broker account does not match policy",
    )


def verify_startup_safety_configuration(
    strategy_type: str,
    account_id: str,
    api_key: str,
    api_secret: str,
    expected_fingerprint: str,
    credential_verification_key: bytes,
    deployment_env: str = "PAPER",
    custom_permitted_accounts: Optional[Dict[str, str]] = None,
) -> None:
    """
    Fail-closed runtime configuration and cryptographic credential verification.

    Guarantees:
      1. Strategy type is known and mapped to a designated account.
      2. Deployment environment is strictly PAPER (LIVE is prohibited in research/paper phase).
      3. Account ID matches permitted allocation for strategy without cross-pollution.
      4. Credential payload HMAC matches the expected cryptographic fingerprint.
      5. Error messages do NOT leak account IDs or credential material into logs.
    """
    permitted_accounts = custom_permitted_accounts or {
        "SWING": "PA3R71I7JFDZ",
        "DAYTRADE": "PA3UODIC5JTA",
    }

    norm_strategy = str(strategy_type).upper().strip()
    require_startup(
        norm_strategy in permitted_accounts,
        "STARTUP_STRATEGY_UNAUTHORIZED: Strategy type unauthorized or unknown",
    )

    # In current release phase, LIVE environment is strictly prohibited
    norm_env = str(deployment_env).upper().strip()
    require_startup(
        norm_env == "PAPER",
        "STARTUP_ENVIRONMENT_PROHIBITED: LIVE environment is prohibited by safety specification v1.1.3",
    )

    expected_account = permitted_accounts[norm_strategy]
    norm_account = str(account_id).strip()
    require_startup(
        hmac.compare_digest(norm_account, expected_account),
        "STARTUP_ACCOUNT_POLICY_MISMATCH: Strategy and account pairing mismatch",
    )

    actual_fingerprint = credential_fingerprint(
        api_key=api_key,
        api_secret=api_secret,
        credential_verification_key=credential_verification_key,
    )

    require_startup(
        hmac.compare_digest(actual_fingerprint, expected_fingerprint),
        "STARTUP_CREDENTIAL_SET_MISMATCH: Credential fingerprint mismatch! Refusing engine startup.",
    )


import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class ProcessLockMetadata:
    pid: int
    hostname: str
    account: str
    mode: str
    branch: str
    release_sha: str
    started_at_utc: str


def is_pid_active(pid: int) -> bool:
    """Check if process with given pid is actively running."""
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        pass
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def acquire_atomic_process_lock(
    lock_path: Path,
    metadata: ProcessLockMetadata,
) -> int:
    """
    Acquire atomic single-instance process lock via exclusive file creation.
    Returns file descriptor on success; raises StartupSafetyRejected on duplicate active instance.
    """
    lock_file = Path(lock_path)
    lock_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            payload = json.dumps(asdict(metadata), indent=2).encode("utf-8")
            os.write(fd, payload)
            if hasattr(os, "fsync"):
                os.fsync(fd)
        except Exception:
            os.close(fd)
            if lock_file.exists():
                lock_file.unlink()
            raise
        return fd
    except FileExistsError:
        # Check if existing lock is stale
        existing_meta = None
        try:
            with open(lock_file, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
        except Exception:
            pass

        if existing_meta and isinstance(existing_meta, dict):
            existing_pid = int(existing_meta.get("pid", 0))
            if existing_pid > 0 and not is_pid_active(existing_pid):
                try:
                    lock_file.unlink()
                    return acquire_atomic_process_lock(lock_path, metadata)
                except Exception as clean_exc:
                    raise StartupSafetyRejected(
                        f"STARTUP_LOCK_STALE_CLEANUP_FAILED: Could not remove stale lock for PID {existing_pid}: {clean_exc}"
                    )

        raise StartupSafetyRejected(
            f"STARTUP_LOCK_COLLISION: Another engine instance is running with lock at {lock_file}"
        )


def release_atomic_process_lock(lock_path: Path, fd: Optional[int] = None) -> None:
    """Safely close lock fd and remove lock file."""
    if fd is not None and fd >= 0:
        try:
            os.close(fd)
        except Exception:
            pass
    lock_file = Path(lock_path)
    if lock_file.exists():
        try:
            lock_file.unlink()
        except Exception:
            pass

