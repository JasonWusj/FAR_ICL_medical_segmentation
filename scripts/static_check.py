#!/usr/bin/env python3
"""Dependency-free syntax checks. Does not import torch or run any experiment."""

import ast
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
python_files = (
    list((root / "far_icl").glob("*.py"))
    + list((root / "scripts").glob("*.py"))
    + list((root / "tests").glob("*.py"))
)
for path in python_files:
    ast.parse(path.read_text(), filename=str(path))
for path in (root / "scripts").glob("*.sh"):
    subprocess.run(["bash", "-n", str(path)], check=True)
subprocess.run(["git", "diff", "--check"], cwd=root, check=True)
subprocess.run(["git", "diff", "--cached", "--check"], cwd=root, check=True)
print(f"PASS: {len(python_files)} Python AST parses and all Bash syntax checks; no model execution.")
