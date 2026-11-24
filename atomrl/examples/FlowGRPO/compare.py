# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Compare FlowGRPO runs with each other and with the official verl-omni reference.

Usage: python3 compare.py $WORK/runs/<run> [$WORK/runs/<run> ...]
Reads <run>/metrics.jsonl (verl file logger). Timing medians skip step 1 and the validation / checkpoint steps.
"""

import json
import statistics as st
import sys
from pathlib import Path

VAL = "val-core/flow_grpo/ocr/reward/mean@1"
REWARD = "critic/rewards/mean"
PHASES = ["gen", "reward", "old_log_prob", "update_actor", "update_weights", "step"]
# verl-omni's public reference run https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/8p8y9olb (4x H200, FSDP engine).
OFFICIAL = {
    "val": {
        30: 0.689,
        60: 0.851,
        90: 0.932,
        120: 0.964,
        150: 0.973,
        180: 0.975,
        210: 0.954,
        240: 0.950,
        270: 0.924,
    },
    "throughput": 0.517,
}


def load(run):
    d = {}
    for line in (Path(run) / "metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        d.setdefault(r["step"], {}).update(r["data"])
    return d


runs = {Path(p).name: load(p) for p in sys.argv[1:]}
names = list(runs)
w = max(12, *(len(n) for n in names))

val_steps = sorted(
    {s for d in runs.values() for s in d if VAL in d[s]} | set(OFFICIAL["val"])
)
print("## validation reward (OCR, 1018 test prompts)")
print("step".ljust(6) + "official".rjust(10) + "".join(n.rjust(w + 2) for n in names))
for s in val_steps:
    off = OFFICIAL["val"].get(s)
    print(
        f"{s:<6d}"
        + (f"{off:10.3f}" if off else " " * 10)
        + "".join(
            (
                f"{runs[n][s][VAL]:{w + 2}.3f}"
                if VAL in runs[n].get(s, {})
                else "-".rjust(w + 2)
            )
            for n in names
        )
    )

print("\n## train reward, mean per 30-step window")
last = max(s for d in runs.values() for s in d if REWARD in d[s])
print("steps".ljust(10) + "".join(n.rjust(w + 2) for n in names))
for a in range(1, last + 1, 30):
    row = []
    for n in names:
        v = [
            runs[n][s][REWARD] for s in range(a, a + 30) if REWARD in runs[n].get(s, {})
        ]
        row.append(f"{st.mean(v):{w + 2}.3f}" if v else "-".rjust(w + 2))
    print(f"{a}-{a + 29}".ljust(10) + "".join(row))

print("\n## time per phase, median (s); throughput (images/GPU/s)")
print(
    "run".ljust(w)
    + "".join(p.rjust(15) for p in PHASES)
    + "   throughput  (official {:.3f})".format(OFFICIAL["throughput"])
)
for n in names:
    d = runs[n]
    steps = [
        s
        for s in d
        if s > 1
        and "timing_s/step" in d[s]
        and "timing_s/testing" not in d[s]
        and "timing_s/save_checkpoint" not in d[s]
    ]
    if not steps:
        continue
    print(
        n.ljust(w)
        + "".join(
            f"{st.median(d[s].get(f'timing_s/{p}', 0) for s in steps):15.1f}"
            for p in PHASES
        )
        + f"   {st.median(d[s]['perf/throughput'] for s in steps):.3f}  ({len(steps)} steps)"
    )
