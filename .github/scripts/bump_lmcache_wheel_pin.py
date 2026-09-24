#!/usr/bin/env python3
"""Point the Dockerfiles' LMCache wheel pin at a new wheel.

The pin is the three LMCACHE_WHEEL_* build args; the Dockerfiles derive the
expected lmcache.__version__ from LMCACHE_WHEEL_NAME, so nothing else changes.
Every file must carry each arg exactly once, otherwise the layout has drifted
from what this script knows and it refuses to guess.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_FILES = ["docker/Dockerfile", "docker/atom_release.dockerfile"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Wheel file name.")
    parser.add_argument("--url", required=True, help="Download URL of the wheel.")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    if not re.fullmatch(r"lmcache-[^-]+-cp\d+-cp\d+-[\w.]+\.whl", args.name):
        parser.error(f"not an LMCache wheel file name: {args.name}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.sha256):
        parser.error(f"not a sha256 digest: {args.sha256}")

    values = {
        "LMCACHE_WHEEL_NAME": args.name,
        "LMCACHE_WHEEL_URL": args.url,
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
        print(f"{path}: pinned {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
