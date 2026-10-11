#!/usr/bin/env python3
"""Compute the pre-registered statistic from the raw rounds. Written first.

Reads lines of the form

    pair=<n> arm=<on|off> round=<r> req_s=<x> p50=<x> p90=<x> p99=<x>
             tier_pct=<x> errors=<n> done=<n> reqs=<n> graph=<0|1>

and applies exactly what
`recipes/atom_vllm/DeepSeek-V4.1-LMCache-perf-preregistration.md` says:
discard round 1 of each boot as the warm round, require the validity
conditions, take the per-pair ratio of medians, report the median of the pair
ratios, and refuse to call a difference real when it is inside the larger
arm's own range.

Nothing here chooses anything. If it did, the choosing would have happened
after the data arrived, which is the thing the pre-registration is for.
"""

import collections
import statistics
import sys

MIN_TIER_PCT = 50.0


def median(xs):
    return statistics.median(xs) if xs else float("nan")


rows = []
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("pair="):
        continue
    d = dict(kv.split("=", 1) for kv in line.split())
    rows.append(d)

by = collections.defaultdict(list)
for d in rows:
    by[(int(d["pair"]), d["arm"])].append(d)

print("=== rounds, with the warm round marked and validity applied ===")
kept = collections.defaultdict(list)
void_pairs = set()
for key in sorted(by):
    pair, arm = key
    rounds = sorted(by[key], key=lambda d: int(d["round"]))
    for i, d in enumerate(rounds):
        req_s = float(d["req_s"])
        why = []
        if i == 0:
            why.append("warm round, discarded by design")
        if d.get("graph") != "1":
            why.append("NOT graph mode")
        if int(d["errors"]) != 0:
            why.append(f"errors={d['errors']}")
        if int(d["done"]) != int(d["reqs"]):
            why.append(f"done {d['done']}/{d['reqs']}")
        if arm == "on" and float(d["tier_pct"]) < MIN_TIER_PCT:
            why.append(f"tier supply {d['tier_pct']}% < {MIN_TIER_PCT}%")
        tag = "; ".join(why) if why else "kept"
        print(
            f"  pair{pair} {arm:3s} round{d['round']} req_s={req_s:.4f} "
            f"tier={d['tier_pct']}% p50={d['p50']} -> {tag}"
        )
        if why and i == 0 and len(why) == 1:
            continue  # warm round only: discarded, not a failure
        if why:
            void_pairs.add(pair)
            continue
        kept[key].append(req_s)

print("\n=== per-arm spread, measured here and not inherited ===")
for key in sorted(kept):
    xs = kept[key]
    if not xs:
        continue
    print(
        f"  pair{key[0]} {key[1]:3s} n={len(xs)} median={median(xs):.4f} "
        f"min={min(xs):.4f} max={max(xs):.4f} range={100 * (max(xs) - min(xs)) / median(xs):.1f}%"
    )

print("\n=== pair ratios ===")
ratios = []
pairs = sorted({k[0] for k in kept})
for pair in pairs:
    on = kept.get((pair, "on"), [])
    off = kept.get((pair, "off"), [])
    if pair in void_pairs or not on or not off:
        print(f"  pair{pair}: VOID")
        continue
    r = median(on) / median(off)
    ratios.append(r)
    print(f"  pair{pair}: {median(on):.4f} / {median(off):.4f} = {r:.4f}")

print("\n=== verdict ===")
if len(ratios) < 3:
    print(f"NOT MEASURABLE: {len(ratios)} valid pairs of the 3 the design asks for.")
    sys.exit(1)
signs = {r > 1 for r in ratios}
if len(signs) > 1:
    print("NOT MEASURABLE: the pair ratios disagree in sign:",
          ", ".join(f"{r:.4f}" for r in ratios))
    print("The remedy the design names is more pairs, not appending a fourth")
    print("to a set of three that disagreed.")
    sys.exit(1)
allon = [x for k, v in kept.items() if k[1] == "on" for x in v]
alloff = [x for k, v in kept.items() if k[1] == "off" for x in v]
widest = max(
    (max(allon) - min(allon)) / median(allon),
    (max(alloff) - min(alloff)) / median(alloff),
)
effect = abs(median(ratios) - 1)
print(f"pair ratios: {', '.join(f'{r:.4f}' for r in ratios)}")
print(f"HEADLINE median pair ratio = {median(ratios):.4f}")
print(f"widest within-arm range = {100 * widest:.1f}%   effect = {100 * effect:.1f}%")
if effect < widest:
    print("NOT MEASURABLE at this sample size: the effect is inside the arm's")
    print("own range. Reported as that, not as a small effect.")
    sys.exit(1)
print("MEASURABLE: the effect exceeds the within-arm range.")
