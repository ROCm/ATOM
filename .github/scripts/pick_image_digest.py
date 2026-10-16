#!/usr/bin/env python3
"""Pick the container image digest a perf-check run measured on.

Every cell records provenance, including one that died before it had an image
to record: those steps are `if: always()`, and a cell that failed in `Start
container` never reached the pin, so its `image_digest` is the empty string.
Taking the first file found made the whole run report `unknown` whenever such a
cell happened to sort first -- run 37922341753 did exactly that with six cells
measured and four dead.

Prints the digest, `MIXED: a, b` when cells disagree, or nothing when no cell
recorded one. Disagreement is reported rather than resolved: two digests mean
the halves were not paired on one image after all, and that premise is what the
whole comparison rests on.
"""

from __future__ import annotations

import json
import sys


def pick(paths: list[str]) -> str:
    seen: list[str] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                digest = (json.load(fh).get("image_digest") or "").strip()
        except (OSError, ValueError):
            continue  # a cell that died mid-write says nothing, not "unknown"
        if digest and digest not in seen:
            seen.append(digest)
    if len(seen) == 1:
        return seen[0]
    if len(seen) > 1:
        return "MIXED: " + ", ".join(seen)
    return ""


if __name__ == "__main__":
    out = pick(sys.argv[1:])
    if out:
        print(out)
