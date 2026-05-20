#!/usr/bin/env python3
"""Backfill historical `AMD Radeon Graphics` labels in the dashboard data.js.

`benchmark-action/github-action-benchmark@v1` (the action that powers
https://rocm.github.io/ATOM/benchmark-dashboard/) is append-only: every nightly
run pushes a new `{commit, date, benches}` entry to `benchmark-dashboard/data.js`
on the `gh-pages` branch and never rewrites history.

That meant the entries written before PR #692 — and the entries written by
accuracy-validation workflows that use `linux-atom-mi35x-*` runners which a
follow-up patch fixed — are permanently stuck with the mis-detected
`GPU: AMD Radeon Graphics` label, even though every ATOM CI machine is
physically an MI355X.

This script does a one-time, in-place rewrite of `data.js` to relabel those
entries. It is intentionally a literal text substitution (not a JSON
round-trip) so the action's exact formatting is preserved and the diff stays
inspectable.

Assumptions (verified against the live data.js on 2026-05-11):

* The dashboard only ever stores two GPU labels: `AMD Radeon Graphics` (6357
  occurrences, all mis-detections) and `AMD Instinct MI355X` (4321 correct).
* No MI300X / MI250X / MI210 entries have ever been written — the ATOM project
  has only run on MI355X-class hardware since the dashboard was created.

If either assumption stops holding (e.g. someone adds an MI300X runner), this
script needs to be tightened to gate the substitution on the runner hint.

Usage:
    # Inspect what would change (default — no writes):
    python tools/backfill_dashboard_gpu_name.py path/to/data.js

    # Apply the rewrite in place:
    python tools/backfill_dashboard_gpu_name.py path/to/data.js --write

    # Full end-to-end against the live gh-pages branch:
    git fetch origin gh-pages
    git worktree add /tmp/atom-gh-pages gh-pages
    python tools/backfill_dashboard_gpu_name.py /tmp/atom-gh-pages/benchmark-dashboard/data.js --write
    cd /tmp/atom-gh-pages
    git add benchmark-dashboard/data.js
    git commit -m "ci(dashboard): backfill MI355X label on historical Radeon entries"
    git push origin gh-pages
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WRONG = "GPU: AMD Radeon Graphics"
RIGHT = "GPU: AMD Instinct MI355X"

# Defensive: only touch occurrences that appear inside a benchmark `extra`
# string. Every `extra` value the action writes starts with `Run: https://...`,
# so anchoring on `"extra": "Run:` makes accidental matches in commit messages
# or other free-form fields essentially impossible.
EXTRA_LINE_RE = re.compile(r'("extra":\s*"Run: [^"]*?)' + re.escape(WRONG))


def rewrite(text: str) -> tuple[str, int]:
    """Return (new_text, replacement_count)."""
    new_text, count = EXTRA_LINE_RE.subn(r"\1" + RIGHT, text)
    return new_text, count


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("data_js", type=Path, help="Path to benchmark-dashboard/data.js")
    p.add_argument(
        "--write",
        action="store_true",
        help="Rewrite the file in place (default is dry-run).",
    )
    args = p.parse_args()

    if not args.data_js.is_file():
        print(f"error: {args.data_js} not found", file=sys.stderr)
        return 2

    original = args.data_js.read_text(encoding="utf-8")
    pre_wrong = original.count(WRONG)
    pre_right = original.count(RIGHT)

    new_text, n = rewrite(original)
    post_wrong = new_text.count(WRONG)
    post_right = new_text.count(RIGHT)

    print(f"file:               {args.data_js}")
    print(f"size:               {len(original):,} bytes")
    print(f"before:             {pre_wrong:>6} Radeon, {pre_right:>6} MI355X")
    print(f"after:              {post_wrong:>6} Radeon, {post_right:>6} MI355X")
    print(f"replacements made:  {n:,}")

    if post_wrong != 0:
        print(
            f"warning: {post_wrong} Radeon mention(s) remain outside of "
            f'`"extra": "Run: ..."` strings; left untouched.',
            file=sys.stderr,
        )

    if not args.write:
        print("\nDry-run only. Re-run with --write to apply.")
        return 0

    if n == 0:
        print("\nNothing to write.")
        return 0

    args.data_js.write_text(new_text, encoding="utf-8")
    print(f"\nWrote {args.data_js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
