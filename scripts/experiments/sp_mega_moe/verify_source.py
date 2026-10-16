#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Require the pinned clean AITER checkout and matching model metadata."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import subprocess
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    expected = json.loads((here / "experiment-source.json").read_text())
    spec = importlib.util.find_spec("aiter")
    if spec is None or spec.origin is None:
        raise RuntimeError("AITER is not installed in this Python environment")
    aiter = Path(spec.origin).resolve().parents[1]
    errors = []
    atom = here.parents[2]
    atom_tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD:atom"], cwd=atom, text=True
    ).strip()
    if atom_tree != expected["atom_tree_sha"] or subprocess.check_output(
        ["git", "diff", "HEAD", "--", "atom"], cwd=atom
    ):
        errors.append("ATOM implementation differs from the validated integration")
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=aiter, text=True
    ).strip()
    if sha != expected["aiter_commit"]:
        errors.append(f"AITER HEAD: {sha}")
    aiter_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=aiter,
        text=True,
    )
    if aiter_status:
        errors.append(
            "AITER must be a clean checkout without local patches or untracked files"
        )
    for name, sha256 in expected["model_metadata_sha256"].items():
        path = args.model / name
        if not path.is_file() or digest(path) != sha256:
            errors.append(f"Missing or different file: {path}")
    for name, size in expected["model_safetensors_sizes"].items():
        path = args.model / name
        if not path.is_file() or path.stat().st_size != size:
            errors.append(f"Missing or different shard size: {path}")
    packages = {}
    for name in expected["packages"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    print(
        json.dumps(
            {
                "aiter": str(aiter),
                "aiter_commit": sha,
                "atom_tree": atom_tree,
                "aiter_status": aiter_status,
                "model": str(args.model.resolve()),
                "packages": packages,
                "package_differences": {
                    k: {"expected": v, "actual": packages[k]}
                    for k, v in expected["packages"].items()
                    if packages[k] != v
                },
                "errors": errors,
                "note": expected["model_identity_note"],
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
