"""
scripts/check_source_encoding.py -- Validate that all repository Python files are valid UTF-8.
"""
import sys
from pathlib import Path

def main():
    errors = []
    excluded = {
        ".git",
        ".venv",
        ".testdeps",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "archive_cleanup_2026_08_17",
    }

    for path in Path(".").rglob("*.py"):
        if any(part in excluded for part in path.parts):
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            errors.append((str(path), str(exc)))

    print(f"UTF-8 encoding validation: {len(errors)} error(s) found.")
    for path, exc in errors:
        print(f"  FAILED: {path} -> {exc}")

    if errors:
        sys.exit(1)
    print("SUCCESS: 100% of repository Python files are valid UTF-8.")
    sys.exit(0)

if __name__ == "__main__":
    main()
