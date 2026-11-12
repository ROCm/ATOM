#!/usr/bin/env python3
"""Install ATOM's CI dependencies without installing or replacing AITER.

GPU CI manages AITER separately because nightly/source builds can report a
version below the release dependency floor. CPU CI omits the GPU dependency.
Python 3.10 callers install the tomli backport before invoking this script.
"""

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def main(extras=()):
    try:
        aiter_version = importlib.metadata.version("amd-aiter")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        print(f"Keeping CI amd-aiter {aiter_version}")
    root = Path(__file__).resolve().parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    dependencies = list(project["dependencies"])
    for extra in extras:
        dependencies.extend(project["optional-dependencies"][extra])
    dependencies = [dep for dep in dependencies if not dep.startswith("amd-aiter")]
    subprocess.run([sys.executable, "-m", "pip", "install", *dependencies], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(root)],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", nargs="*", default=())
    main(parser.parse_args().extras)
