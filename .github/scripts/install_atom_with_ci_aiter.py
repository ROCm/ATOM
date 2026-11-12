#!/usr/bin/env python3
"""Install ATOM without replacing the source-validated CI AITER wheel.

The download job checks AITER's source baseline. Nightly wheel metadata can be
0.1.1.dev1+g<SHA> even for sources newer than the required release, so keep the
managed GPU dependency and resolve all other ATOM dependencies normally. Source
builders use --aiter-source to check their checkout before installing ATOM.
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

MINIMUM_AITER_COMMIT = "f1965cd26021c43436ae2bd614846a5977c2636e"  # v0.1.28


def validate_aiter_source(source):
    """Require the output APIs even when a source build has nightly metadata."""
    git = ["git", "-C", str(source)]
    shallow = subprocess.check_output(
        [*git, "rev-parse", "--is-shallow-repository"], text=True
    ).strip()
    if shallow == "true":
        subprocess.run(
            [*git, "fetch", "--unshallow", "--filter=blob:none", "origin"], check=True
        )
    subprocess.run(
        [*git, "merge-base", "--is-ancestor", MINIMUM_AITER_COMMIT, "HEAD"],
        check=True,
    )
    head = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
    print(f"Verified AITER source {head} includes {MINIMUM_AITER_COMMIT}")


def main(aiter_source=None):
    print(f"Keeping CI amd-aiter {importlib.metadata.version('amd-aiter')}")
    if aiter_source is not None:
        validate_aiter_source(aiter_source)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter-source", type=Path)
    main(parser.parse_args().aiter_source)
