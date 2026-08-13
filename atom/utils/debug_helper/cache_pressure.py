# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Read a server log's cache lines and say which pool, if either, was the wall.

    python -m atom.utils.debug_helper.cache_pressure server.log

A hit rate below 100% has three causes that look identical in the rate alone,
and the fix differs for each:

  * the prompt had no reusable prefix     -- nothing to do
  * the prefix was there and was evicted  -- the pool that evicted is too small
  * the prefix was there and was declined -- a checkpoint was missing

`[Cache Stats]` separates the third from the first two; `[Pool Pressure]` and
`[Checkpoint Fates]` separate the first two from each other. This reads all
three and reports the last snapshot plus what moved over the run.

The eviction counters are cumulative and monotone, so the final line carries
the whole run. Occupancy is instantaneous, and a run's *last* reading is taken
as it drains -- `peak_used` is tracked across lines instead, because the
question is whether a pool was ever full, not whether it is full now.
"""

from __future__ import annotations

import re
import sys

# Tolerant of the logger's own prefix (timestamp, level) ahead of the tag.
_PRESSURE = re.compile(
    r"\[Pool Pressure\] paged: (?P<used>\d+)/(?P<total>\d+) used, "
    r"(?P<reusable>\d+) reusable-free, (?P<vacant>\d+) vacant, "
    r"(?P<indexed>\d+) indexed \| "
    r"evicted: (?P<evicted>\d+), retired: (?P<retired>\d+) \| "
    r"state: (?P<gused>\d+)/(?P<gtotal>\d+) used, "
    r"(?P<gheld>\d+) checkpointed, (?P<gvacant>\d+) vacant"
)
_FATES = re.compile(
    r"\[Checkpoint Fates\] kept: (?P<kept>\d+), dropped: (?P<dropped>\d+), "
    r"evicted: (?P<cevicted>\d+), orphaned: (?P<orphaned>\d+)"
)
_STATS = re.compile(
    r"\[Cache Stats\s*\] Reqs: (?P<reqs>\d+), "
    r"Cached/Total: (?P<cached>\d+)/(?P<full>\d+)"
    r"(?:.*?Lost-to-checkpoint: (?P<lost_ckpt>[\d.]+)%)?"
)
# The interval variant repeats the tag with a different label; excluded above by
# requiring the tag to close right after the padding, so only cumulative counts
# are read and the last one carries the run.


def _pct(num: int, den: int) -> str:
    return f"{num / den:.2%}" if den else "n/a"


def analyse(lines) -> dict:
    """Last reading of each line type, plus peak occupancy across the run."""
    out: dict = {"peak_paged_used": 0, "peak_state_used": 0, "samples": 0}
    for line in lines:
        if m := _PRESSURE.search(line):
            g = {k: int(v) for k, v in m.groupdict().items()}
            out.update(g)
            out["samples"] += 1
            out["peak_paged_used"] = max(out["peak_paged_used"], g["used"])
            out["peak_state_used"] = max(out["peak_state_used"], g["gused"])
        elif (m := _FATES.search(line)) or (m := _STATS.search(line)):
            # Optional groups come back None when absent, and the loss field is
            # a decimal percentage -- neither survives a blanket int().
            for k, v in m.groupdict().items():
                if v is not None:
                    out[k] = float(v) if "." in v else int(v)
    return out


def verdict(a: dict) -> list[str]:
    """The reading, in the order a reader needs it to decide what to change."""
    if not a.get("samples"):
        return [
            "No [Pool Pressure] lines found.",
            "  They are emitted by CacheStats only when prefix caching is on,",
            "  once per `log_interval` (default 100) completed prefills.",
        ]

    evicted, retired = a.get("evicted", 0), a.get("retired", 0)
    orphaned, cevicted = a.get("orphaned", 0), a.get("cevicted", 0)
    # `cevicted` counts checkpoints displaced by later checkpoints, which any
    # run longer than the pool does by construction -- it tracks `kept` minus
    # the pool size, not shortage. `cdropped` is the one that means a keep was
    # refused for lack of room, so that is what decides the state verdict.
    cdropped = a.get("dropped", 0)
    paged_headroom = a["total"] - a["peak_paged_used"]
    state_headroom = a["gtotal"] - a["peak_state_used"]

    paged_pct = _pct(a["peak_paged_used"], a["total"])
    state_pct = _pct(a["peak_state_used"], a["gtotal"])
    out = [
        f"samples: {a['samples']}",
        (
            f"paged: peak {a['peak_paged_used']}/{a['total']} used "
            f"({paged_pct}), {paged_headroom} never used"
        ),
        (
            f"state: peak {a['peak_state_used']}/{a['gtotal']} used "
            f"({state_pct}), {state_headroom} never used"
        ),
        (
            f"evictions: paged {evicted} (+{retired} to boundary moves), "
            f"checkpoints {cdropped} refused for room "
            f"({cevicted} displaced, {orphaned} orphaned)"
        ),
    ]
    if "cached" in a:
        out.append(f"hit rate: {_pct(a['cached'], a['full'])} over {a['reqs']} reqs")

    out.append("")
    if evicted == 0 and cdropped == 0:
        out += [
            "VERDICT: neither pool ever ran out.",
            "  Every miss above is reuse that was absent or declined, not lost.",
            "  Growing either pool cannot raise the hit rate on this workload.",
        ]
    elif evicted and not cdropped:
        out += [
            "VERDICT: the paged pool is the wall.",
            (
                f"  It spent {evicted} cached blocks, while the state pool never "
                f"refused a keep (peak {state_pct} of its groups live)."
            ),
            "  Bytes reserved for state that state never used are the ones to move.",
        ]
    elif cdropped and not evicted:
        out += [
            "VERDICT: the state pool is the wall.",
            f"  {cdropped} checkpoints could not be kept for lack of a free group,",
            f"  while the paged pool peaked at {paged_pct}.",
        ]
    else:
        out += [
            "VERDICT: both pools ran out -- the budget is short, not the split.",
            f"  paged evicted {evicted}, checkpoint keeps refused {cdropped}.",
        ]

    # Whatever the pools did, this bounds the fix: the share of tokens that were
    # present in KV but unusable because the checkpoint to resume from was gone.
    if "lost_ckpt" in a:
        out += [
            "",
            f"CEILING: {a['lost_ckpt']}% of tokens were lost to a missing checkpoint.",
            "  No amount of state-pool growth can buy back more than that.",
        ]

    if orphaned:
        out += [
            "",
            f"NOTE: {orphaned} checkpoints were orphaned -- the prefix they were",
            "  filed under left the KV index first, so nothing could resume off",
            "  them. That is paged pressure showing up in a state counter; it",
            "  argues for a bigger paged pool, not a bigger state pool.",
        ]
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    with open(argv[1], errors="replace") as fh:
        analysis = analyse(fh)
    print("\n".join(verdict(analysis)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
