#!/usr/bin/env python3
"""Install ATOM without replacing the source-validated CI AITER wheel.

The download job checks AITER's source baseline. Nightly wheel metadata can be
0.1.1.dev1+g<SHA> even for sources newer than the required release, so keep the
managed GPU dependency and resolve all other ATOM dependencies normally.
"""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import tomllib


def main():
    print(f"Keeping CI amd-aiter {importlib.metadata.version('amd-aiter')}")
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    dependencies = [
        dep for dep in project["dependencies"] if not dep.startswith("amd-aiter")
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", *dependencies], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(root)],
        check=True,
    )


if __name__ == "__main__":
    main()
