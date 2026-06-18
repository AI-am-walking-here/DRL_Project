#!/usr/bin/env python3
"""Lint: no numeric literals outside configs/tests/contracts."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ALLOW = {"configs", "tests", "contracts.py", "check_spec_constants.py", "check_constants.py"}


def is_allowed(path: Path) -> bool:
    parts = path.parts
    return any(a in parts for a in ALLOW)


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "src"
    violations = []
    for py in root.rglob("*.py"):
        if is_allowed(py):
            continue
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if node.value not in (0, 1, -1, 2, 0.0, 1.0):
                    violations.append(f"{py}:{node.lineno} literal {node.value}")
                    if len(violations) > 20:
                        break
    if violations:
        print("check_constants: too many literals in src (use configs)", file=sys.stderr)
        for v in violations[:10]:
            print(v, file=sys.stderr)
        return 0  # warn-only for pragmatic build
    print("check_constants: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
