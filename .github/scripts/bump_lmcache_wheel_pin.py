#!/usr/bin/env python3
"""Point the Dockerfiles' LMCache wheel pin at a new release.

The pin is LMCACHE_WHEEL_RELEASE and LMCACHE_WHEEL_SHA256; the Dockerfiles
derive the wheel name, its URL and the expected lmcache.__version__ from the
release tag, so nothing else changes. Every file must carry each arg exactly
once, otherwise the layout has drifted from what this script knows and it
refuses to guess.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_FILES = ["docker/Dockerfile", "docker/atom_release.dockerfile"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release", required=True, help="Release tag, e.g. lmcache-v...-rocm-torch210."
    )
    parser.add_argument("--sha256", required=True)
    parser.add_argument("files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    # The Dockerfiles parse the tag with the same pattern.
    if not re.fullmatch(r"lmcache-v[^-]+-g[0-9a-f]{8}-rocm-torch210", args.release):
        parser.error(f"not an LMCache wheel release tag: {args.release}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        parser.error(f"not a sha256 digest: {args.sha256}")

    values = {
        "LMCACHE_WHEEL_RELEASE": args.release,
        "LMCACHE_WHEEL_SHA256": args.sha256,
    }
    for file in args.files:
        path = Path(file)
        text = path.read_text()
        for key, value in values.items():
            matches = list(re.finditer(rf"^ARG {key}=.*$", text, re.MULTILINE))
            if len(matches) != 1:
                print(f"{path}: expected one 'ARG {key}=', found {len(matches)}")
                return 1
            start, end = matches[0].span()
            text = f"{text[:start]}ARG {key}={value}{text[end:]}"
        path.write_text(text)
        print(f"{path}: pinned {args.release}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
