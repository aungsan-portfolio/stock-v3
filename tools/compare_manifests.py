"""
tools/compare_manifests.py — Compare two cryptographic manifests to verify 0 additions, deletions, or mutations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Any, List


def compare_manifests(before_path: Path, after_path: Path, report_path: Path | None = None) -> bool:
    with open(before_path, "r", encoding="utf-8") as f:
        before = json.load(f)
    with open(after_path, "r", encoding="utf-8") as f:
        after = json.load(f)

    before_files: Dict[str, Any] = before.get("files", {})
    after_files: Dict[str, Any] = after.get("files", {})

    integrity_errors: List[str] = []
    for m_name, m_files in [("before", before_files), ("after", after_files)]:
        for p, info in m_files.items():
            if not isinstance(info, dict):
                integrity_errors.append(f"{m_name}: '{p}' invalid entry structure")
                continue
            sha = info.get("sha256")
            size = info.get("size_bytes")
            if not isinstance(sha, str) or len(sha) != 64 or not all(c in "0123456789abcdefABCDEF" for c in sha):
                integrity_errors.append(f"{m_name}: '{p}' invalid sha256 digest: {repr(sha)}")
            if not isinstance(size, int) or size < 0:
                integrity_errors.append(f"{m_name}: '{p}' invalid size_bytes: {repr(size)}")

    added = [p for p in after_files if p not in before_files]
    removed = [p for p in before_files if p not in after_files]
    modified = []

    for p, b_info in before_files.items():
        if p in after_files:
            a_info = after_files[p]
            if not isinstance(b_info, dict) or not isinstance(a_info, dict):
                continue
            if b_info.get("size_bytes") != a_info.get("size_bytes") or b_info.get("sha256") != a_info.get("sha256"):
                modified.append({
                    "path": p,
                    "before_size": b_info.get("size_bytes"),
                    "after_size": a_info.get("size_bytes"),
                    "before_sha256": str(b_info.get("sha256")),
                    "after_sha256": str(a_info.get("sha256")),
                })

    report_lines: List[str] = [
        "============================================================",
        "TEST RUNTIME AUDIT ISOLATION -- MANIFEST VERIFICATION REPORT",
        "============================================================",
        f"Before Manifest : {before_path} ({len(before_files)} files)",
        f"After Manifest  : {after_path} ({len(after_files)} files)",
        "Excluded Paths  : data/cache",
        "------------------------------------------------------------",
        f"Integrity Errors: {len(integrity_errors)}",
        f"Files Added     : {len(added)}",
        f"Files Removed   : {len(removed)}",
        f"Files Modified  : {len(modified)}",
        "------------------------------------------------------------",
    ]

    if integrity_errors:
        report_lines.append("\n[MANIFEST INTEGRITY ERRORS]")
        for err in integrity_errors:
            report_lines.append(f"  ! {err}")

    if added:
        report_lines.append("\n[UNAUTHORIZED ADDITIONS]")
        for a in added:
            report_lines.append(f"  + {a}")

    if removed:
        report_lines.append("\n[UNAUTHORIZED REMOVALS]")
        for r in removed:
            report_lines.append(f"  - {r}")

    if modified:
        report_lines.append("\n[UNAUTHORIZED MODIFICATIONS]")
        for m in modified:
            report_lines.append(
                f"  * {m['path']}: size {m['before_size']} -> {m['after_size']}, "
                f"sha256 {m['before_sha256'][:16]}... -> {m['after_sha256'][:16]}..."
            )

    success = (len(integrity_errors) == 0 and len(added) == 0 and len(removed) == 0 and len(modified) == 0)

    if success:
        report_lines.append("\nRESULT: PASS -- 100% Zero Mutations Across All Tracked Production Files.")
    else:
        report_lines.append("\nRESULT: FAIL -- Isolation Leak or Manifest Integrity Error Detected!")

    report_text = "\n".join(report_lines)
    print(report_text)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"\nReport saved to {report_path}")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two file manifests.")
    parser.add_argument("--before", "-b", required=True, help="Before manifest JSON")
    parser.add_argument("--after", "-a", required=True, help="After manifest JSON")
    parser.add_argument("--report", "-r", default=None, help="Output report TXT path")
    args = parser.parse_args()

    success = compare_manifests(
        Path(args.before).resolve(),
        Path(args.after).resolve(),
        Path(args.report).resolve() if args.report else None,
    )
    sys.exit(0 if success else 1)
