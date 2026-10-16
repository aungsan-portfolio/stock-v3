"""
tools/generate_directory_manifest.py — Generate cryptographic file manifest for data/, logs/, reports/.

Scans all production state, audit, and log directories (excluding ephemeral data/cache)
and computes exact file sizes and SHA-256 hashes to guarantee test isolation invariance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception as e:
        return f"ERROR: {e}"


def generate_manifest(root: Path, output_file: Path) -> Dict[str, Any]:
    target_dirs = ["data", "logs", "reports"]
    exclude_subdirs = {
        (root / "data" / "cache").resolve(),
    }

    manifest: Dict[str, Any] = {
        "root": str(root.resolve()),
        "target_directories": target_dirs,
        "excluded": ["data/cache"],
        "files": {},
    }

    for dir_name in target_dirs:
        target_path = root / dir_name
        if not target_path.exists():
            continue

        for p in sorted(target_path.rglob("*")):
            if not p.is_file():
                continue

            # Check if file is in an excluded subdirectory
            resolved_p = p.resolve()
            if any(ex in resolved_p.parents or resolved_p == ex for ex in exclude_subdirs):
                continue

            rel_path = p.relative_to(root).as_posix()
            stat = p.stat()
            sha256 = compute_sha256(p)

            manifest["files"][rel_path] = {
                "size_bytes": stat.st_size,
                "sha256": sha256,
                "mtime": stat.st_mtime,
            }

    manifest["total_files"] = len(manifest["files"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {output_file} ({manifest['total_files']} files tracked).")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate directory manifest for data/, logs/, reports/.")
    parser.add_argument("--output", "-o", required=True, help="Output JSON path")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    generate_manifest(repo_root, Path(args.output).resolve())
