# SPDX-License-Identifier: MIT
"""Verify that Gluon scheduling and tuning match the official v0.1.26 source."""

import argparse
import ast
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def fingerprint(source, symbols):
    if not symbols:
        return hashlib.sha256(source).hexdigest()
    tree = ast.parse(source)
    # Scale loads need the logical tile index; it does not change the loop.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            node.keywords = [kw for kw in node.keywords if kw.arg != "tile_index"]
    selected = {node.name: node for node in tree.body if hasattr(node, "name")}
    return {
        name: hashlib.sha256(
            ast.dump(selected[name], include_attributes=False).encode()
        ).hexdigest()
        for name in symbols
    }


def verify(read_source, manifest):
    for path, entry in manifest["files"].items():
        actual = fingerprint(read_source(path), entry.get("symbols"))
        if actual != entry["sha256"]:
            raise ValueError(f"Official v0.1.26 implementation changed: {path}")
    print(
        "[gluon-upstream] official v0.1.26 split, tuning, attention math and "
        f"reduce verified ({manifest['upstream_commit']})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--source", type=Path)
    group.add_argument("--ref", help="Full ROCm/aiter commit SHA")
    args = parser.parse_args()
    manifest = json.loads(
        Path(__file__).with_name("m3_gluon_v0126_source.json").read_text()
    )
    if args.source:

        def read_source(path):
            return (args.source / path).read_bytes()

    else:
        if not re.fullmatch(r"[0-9a-f]{40}", args.ref):
            parser.error("--ref must be a full commit SHA")

        def read_source(path):
            url = f"https://raw.githubusercontent.com/ROCm/aiter/{args.ref}/{path}"
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()

    verify(read_source, manifest)


if __name__ == "__main__":
    main()
