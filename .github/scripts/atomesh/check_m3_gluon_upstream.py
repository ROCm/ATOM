# SPDX-License-Identifier: MIT
"""Verify that Gluon scheduling and tuning match the official v0.1.26 source."""

import argparse
import ast
import hashlib
import json
import re
import urllib.request
from pathlib import Path


def canonical_ast(node):
    """Use a stable form across Python versions, including 3.12+ type_params."""
    if isinstance(node, ast.AST):
        return {
            "node": type(node).__name__,
            **{
                key: canonical_ast(value)
                for key, value in ast.iter_fields(node)
                if value is not None and value != []
            },
        }
    if isinstance(node, list):
        return [canonical_ast(value) for value in node]
    return node


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
            json.dumps(canonical_ast(selected[name]), sort_keys=True).encode()
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
