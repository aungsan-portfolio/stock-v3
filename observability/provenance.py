"""
observability/provenance.py — Single Source of Truth for Build Provenance & Runtime Identification.
"""
import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BuildProvenance:
    commit: str
    branch: str
    working_tree_clean: bool
    source_root: str
    python_executable: str
    stream_state_machine: str = "v1"
    persistent_bridge: bool = True
    broker_order_check: str = "TRI_STATE"

    def marker_string(self) -> str:
        return (
            f"[P1 BUILD] git_commit={self.commit} "
            f"branch={self.branch} "
            f"stream_state_machine={self.stream_state_machine} "
            f"persistent_bridge={self.persistent_bridge} "
            f"broker_order_check={self.broker_order_check}"
        )


_CACHED_PROVENANCE: Optional[BuildProvenance] = None


def get_build_provenance() -> BuildProvenance:
    global _CACHED_PROVENANCE
    if _CACHED_PROVENANCE is not None:
        return _CACHED_PROVENANCE

    root_dir = Path(__file__).resolve().parent.parent
    commit = "ceb5b12"
    branch = "feature/p1-resilience-reconnect-hardening"
    clean = True

    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=str(root_dir),
        )
        if res.returncode == 0 and res.stdout.strip():
            commit = res.stdout.strip()
    except Exception:
        pass

    try:
        res_b = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=str(root_dir),
        )
        if res_b.returncode == 0 and res_b.stdout.strip():
            branch = res_b.stdout.strip()
    except Exception:
        pass

    try:
        res_s = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=str(root_dir),
        )
        if res_s.returncode == 0:
            clean = len(res_s.stdout.strip()) == 0
    except Exception:
        pass

    _CACHED_PROVENANCE = BuildProvenance(
        commit=commit,
        branch=branch,
        working_tree_clean=clean,
        source_root=str(root_dir),
        python_executable=sys.executable,
    )
    return _CACHED_PROVENANCE
