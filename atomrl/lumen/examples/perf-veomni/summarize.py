#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""One compact table for a whole matrix: steady step time, within-mode spread,
gain against the control, peak memory and loss deviation, for every mode at
once, with the steady figure at full precision.

    summarize.py --control pvq-A0 --reps r1 r2 r3 -- \
                 pvq-A0 pvq-B1 pvq-B2 ...
    summarize.py --control pvq-A0 --chain -- pvq-A0 pvq-B1 ...   # each vs the row above

The estimator is the MEDIAN, not the mean, and that is measured rather than
stylistic. Per-step time on this node is reproducible to about +-0.4%, but an
occasional step stalls for seconds -- 1.19 s steps with one 4.2 s step in the
same run. A single such stall moves an 8-step mean by ~10%, which is larger
than anything being measured, while the median ignores it. The stall column
counts steps above 1.5x the run's own median so the interference stays visible
instead of being quietly smoothed away.

Step 1 is always dropped: it carries kernel autotune, and JIT on a FlyDSL run.
A gain smaller than the control's own spread is reported as within spread, not
as a speedup -- that call is the entire point of repeating runs.
"""

import argparse
import glob
import itertools
import logging
import math
import os
import statistics as st

from google.protobuf.message import DecodeError

logger = logging.getLogger(__name__)

OUT_DIR = os.environ.get("OUT_DIR", "/work/outputs")
LOG_DIR = os.environ.get("LOG_DIR", "/work/logs")


def read_rows(run):
    root = os.path.join(OUT_DIR, f"wandb-{run}")
    paths = sorted(
        glob.glob(os.path.join(root, "wandb", "offline-run-*", "run-*.wandb"))
    )
    if not paths:
        return None
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    ds = DataStore()
    ds.open_for_scan(paths[-1])
    rows = {}
    while True:
        data = ds.scan_data()
        if data is None:
            break
        pb = wandb_internal_pb2.Record()
        try:
            pb.ParseFromString(data)
        except DecodeError as exc:
            logger.debug("Skipping malformed wandb record: %s", exc)
            continue
        if pb.WhichOneof("record_type") != "history":
            continue
        item = {}
        for h in pb.history.item:
            key = h.nested_key[0] if h.nested_key else h.key
            try:
                item[key] = float(h.value_json)
            except (TypeError, ValueError):
                pass
        if item.get("_step") is not None:
            rows[int(item["_step"])] = item
    return rows


def gaps(rows):
    steps = sorted(rows)
    ts = [rows[s].get("_timestamp") for s in steps]
    if any(t is None for t in ts) or len(ts) < 3:
        return []
    return [b - a for a, b in itertools.pairwise(ts)][1:]


def meta_field(run, field):
    path = os.path.join(LOG_DIR, f"{run}.meta.txt")
    if not os.path.exists(path):
        return "?"
    with open(path) as fh:
        for line in fh:
            if line.startswith(field) and ":" in line:
                return line.split(":", 1)[1].strip()
    return "?"


def ranksum(a, b):
    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r_a = sum(ranks[i] for i, (_, g) in enumerate(combined) if g == 0)
    n1, n2 = len(a), len(b)
    u = r_a - n1 * (n1 + 1) / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - n1 * n2 / 2) / sigma if sigma else 0.0
    return math.erfc(abs(z) / math.sqrt(2))


def collect(mode, reps):
    """Per-run medians, the pooled per-step samples, memory, loss, step 1."""
    medians, pooled, mems, losses, firsts, stalls = [], [], [], {}, [], 0
    for rep in reps:
        run = f"{mode}-{rep}"
        rows = read_rows(run)
        if not rows:
            continue
        g = gaps(rows)
        if not g:
            continue
        med = st.median(g)
        medians.append(med)
        stalls += sum(1 for v in g if v > 1.5 * med)
        pooled += g
        steps = sorted(rows)
        mems.append(max(rows[s].get("max_memory_allocated(GB)", 0) for s in steps))
        firsts.append(rows[steps[0]].get("_runtime") or 0.0)
        losses[run] = {s: rows[s].get("training/total_loss") for s in steps}
    return medians, pooled, mems, losses, firsts, stalls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--reps", nargs="+", default=["r1", "r2", "r3"])
    ap.add_argument(
        "--chain",
        action="store_true",
        help="read the modes as a ladder: compare each with the one listed before it, not with the control",
    )
    ap.add_argument("modes", nargs="+")
    args = ap.parse_args()

    data = {m: collect(m, args.reps) for m in args.modes}
    ctrl = data.get(args.control)
    if not ctrl or not ctrl[0]:
        raise SystemExit(f"no data for control {args.control}")
    ctrl_med = st.median(ctrl[0])
    ctrl_spread = (
        (max(ctrl[0]) - min(ctrl[0])) / ctrl_med * 100 if len(ctrl[0]) > 1 else 0.0
    )

    # Loss floor: how much the control's own repeats differ from each other.
    ctrl_losses = ctrl[3]
    names = sorted(ctrl_losses)
    floor_max = 0.0
    if len(names) > 1:
        ref = ctrl_losses[names[0]]
        devs = [
            abs(v - ref[s]) / abs(ref[s])
            for n in names[1:]
            for s, v in ctrl_losses[n].items()
            if ref.get(s) and v is not None
        ]
        floor_max = max(devs) * 100 if devs else 0.0

    print(f"control = {args.control}   reps = {' '.join(args.reps)}")
    print(
        f"control spread (per-run medians) {ctrl_spread:.1f}%   loss floor across its repeats: max {floor_max:.4f}%\n"
    )
    versus = "vs prev" if args.chain else "vs ctrl"
    hdr = (
        f"{'mode':17}{'n':>2}{'median':>9}{'spread':>8}{versus:>9}{'p':>9}{'x ctrl':>8}"
        f"{'peak GB':>9}{'step1':>7}{'lossdev':>9}{'stall':>6}  exit"
    )
    print(hdr)
    print("-" * len(hdr))
    prev = None
    for m in args.modes:
        medians, pooled, mems, losses, firsts, stalls = data[m]
        if not medians:
            print(f"{m:17}{'-':>2}  (no data)")
            continue
        med = st.median(medians)
        spread = (max(medians) - min(medians)) / med * 100 if len(medians) > 1 else 0.0
        ref_data = prev if args.chain else ctrl
        first = ref_data is None or (not args.chain and m == args.control)
        ref_med = st.median(ref_data[0]) if ref_data is not None else med
        rel = (med - ref_med) / ref_med * 100
        p = f"{'':>9}" if first else f"{ranksum(ref_data[1], pooled):>9.1e}"
        prev = data[m]
        ref = ctrl_losses.get(names[0], {}) if names else {}
        devs = [
            abs(v - ref[s]) / abs(ref[s])
            for run in sorted(losses)
            for s, v in losses[run].items()
            if ref.get(s) and v is not None
        ]
        ld = max(devs) * 100 if devs else 0.0
        codes = {
            meta_field(f"{m}-{rep}", "EXIT_CODE")
            for rep in args.reps
            if meta_field(f"{m}-{rep}", "EXIT_CODE") != "?"
        }
        print(
            f"{m:17}{len(medians):>2}{med:>8.3f}s{spread:>7.1f}%{rel:>+8.2f}%{p}{ctrl_med / med:>7.2f}x"
            f"{max(mems):>9.2f}{st.mean(firsts):>6.1f}s{ld:>8.3f}%{stalls:>6}  {','.join(sorted(codes))}"
        )
    if args.chain:
        print(
            "\nEach row is the increment of its own change: vs prev and p are against the row above;"
        )
        print(
            "x ctrl is the cumulative speed-up over the control. An increment smaller than the"
        )
        print("spread of either row is not a result.")
    else:
        print(
            f"\nA gain smaller than the control's {ctrl_spread:.1f}% spread is not a result."
        )
    print("p is Mann-Whitney over pooled per-step samples.")
    print(
        "stall = steps above 1.5x their own run's median; these are host interference,"
    )
    print(
        "not the configuration, which is why the median and not the mean is reported."
    )
    print("lossdev is max |relative| vs the control's first repeat; the control's own")
    print(
        f"floor is {floor_max:.4f}% max -- read deviations against that, not against zero."
    )


if __name__ == "__main__":
    main()
