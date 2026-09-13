"""
scripts/check_python_compilation.py -- Explicitly compile all project Python files.
"""
import os
import py_compile
import sys

def main():
    errors = []
    excluded = [
        ".git",
        ".venv",
        ".testdeps",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "archive_cleanup_2026_08_17",
    ]

    for root, dirs, files in os.walk("."):
        if any(p in root for p in excluded):
            continue
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    py_compile.compile(p, doraise=True)
                except Exception as exc:
                    errors.append((p, str(exc)))

    print(f"Compilation validation: {len(errors)} error(s) found.")
    for p, exc in errors:
        print(f"  FAILED: {p} -> {exc}")

    if errors:
        sys.exit(1)
    print("SUCCESS: 100% of project Python files compile cleanly.")
    sys.exit(0)

if __name__ == "__main__":
    main()
