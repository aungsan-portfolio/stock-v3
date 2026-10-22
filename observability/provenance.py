"""
observability/provenance.py — Single Source of Truth for Build Provenance & Runtime Identification.
"""
import hashlib
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
    dirty_count: int = 0
    alpaca_bridge_sha256: str = ""
    order_manager_sha256: str = ""
    module_hashes: tuple = ()

    def marker_string(self) -> str:
        clean_tag = "clean" if self.working_tree_clean else f"DIRTY({self.dirty_count}_changed_paths)"
        parts = [
            f"[P1 BUILD] git_commit={self.commit}",
            f"branch={self.branch}",
            f"working_tree={clean_tag}",
            f"stream_state_machine={self.stream_state_machine}",
            f"persistent_bridge={self.persistent_bridge}",
            f"broker_order_check={self.broker_order_check}",
        ]
        if self.alpaca_bridge_sha256:
            parts.append(f"alpaca_bridge_sha256={self.alpaca_bridge_sha256[:12]}")
        return " ".join(parts)

    def source_hash_report(self) -> str:
        lines = [
            f"[SOURCE HASH AUDIT] alpaca_bridge={self.alpaca_bridge_sha256} order_manager={self.order_manager_sha256}"
        ]
        for rel_path, full_path, hsh in self.module_hashes:
            lines.append(f"  [MODULE] {rel_path} -> sha256={hsh} (path={full_path})")
        return "\n".join(lines)


def _sha256_file(path: Path) -> str:
    try:
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        pass
    return "unknown"


_CACHED_PROVENANCE: Optional[BuildProvenance] = None


def get_build_provenance() -> BuildProvenance:
    global _CACHED_PROVENANCE
    if _CACHED_PROVENANCE is not None:
        return _CACHED_PROVENANCE

    root_dir = Path(__file__).resolve().parent.parent
    commit = "ceb5b12"
    branch = "feature/p1-resilience-reconnect-hardening"
    clean = True
    dirty_count = 0

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
            lines = [l for l in res_s.stdout.strip().splitlines() if l.strip()]
            clean = len(lines) == 0
            dirty_count = len(lines)
    except Exception:
        pass

    bridge_hash = _sha256_file(root_dir / "alpaca_bridge.py")
    om_hash = _sha256_file(root_dir / "strategies" / "order_manager.py")

    tracked_runtime_files = [
        "alpaca_bridge.py",
        "strategies/order_manager.py",
        "strategies/orchestrator.py",
        "strategies/execution_quote.py",
        "strategies/intraday_data.py",
        "strategies/execution_health.py",
        "strategies/mean_reversion.py",
        "strategies/regime_engine.py",
        "strategies/shadow_stream_manager.py",
        "logging_setup.py",
        "main.py",
        "observability/provenance.py",
    ]
    module_hashes = []
    for rel in tracked_runtime_files:
        p = root_dir / rel
        hsh = _sha256_file(p)
        module_hashes.append((rel, str(p), hsh))

    _CACHED_PROVENANCE = BuildProvenance(
        commit=commit,
        branch=branch,
        working_tree_clean=clean,
        source_root=str(root_dir),
        python_executable=sys.executable,
        dirty_count=dirty_count,
        alpaca_bridge_sha256=bridge_hash,
        order_manager_sha256=om_hash,
        module_hashes=tuple(module_hashes),
    )
    return _CACHED_PROVENANCE
