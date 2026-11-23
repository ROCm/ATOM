# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batches of the first draft design's stale-poll diagnostic
(``steptail_markov_check.py`` DESIGN=v1), a process a configuration, one
thing changed at a time against the 9x4 baseline: the memory the ranks'
regions are in, where the poll load's wait and the clock read sit, the pause,
how many waves and CTAs poll. The batch stops at the first configuration that
does not end cleanly.

    python tools/steptail_stale_matrix.py
    env: BATCH (memory | poll | shape | isa), LAUNCHES (1000: launches eager
    and as many replayed), BOUND (2000000: the poll bound, 10 ns ticks)
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = os.environ.get("BATCH", "memory")
LAUNCHES = os.environ.get("LAUNCHES", "1000")
BOUND = os.environ.get("BOUND", "2000000")
CHECK, ISA = "steptail_markov_check.py", "steptail_markov_isa.py"
LOOP = {"SHAPES": "9x4", "BEFORE": "2", "AFTER": "22"}
BATCHES = {
    "memory": [
        ("9x4 baseline (torch memory)", CHECK, {"SHAPES": "9x4"}),
        (
            "9x4, regions in uncached memory",
            CHECK,
            {"SHAPES": "9x4", "REGIONS": "uncached"},
        ),
        (
            "9x4, regions in fine-grained memory",
            CHECK,
            {"SHAPES": "9x4", "REGIONS": "fine"},
        ),
    ],
    "poll": [
        (
            "9x4, each poll load waited for at once",
            CHECK,
            {"SHAPES": "9x4", "SETTLE": "1"},
        ),
        (
            "9x4, the clock read before the load",
            CHECK,
            {"SHAPES": "9x4", "CLOCKFIRST": "1"},
        ),
        ("9x4, no clock in the loop", CHECK, {"SHAPES": "9x4", "POLLROUNDS": "20000"}),
        ("9x4, pause of 16 units", CHECK, {"SHAPES": "9x4", "SLEEP": "16"}),
    ],
    "shape": [
        ("8x4 + 8 threads of a third wave", CHECK, {"SHAPES": "8x4", "EXTRA": "8"}),
        ("8x4 + a whole third wave", CHECK, {"SHAPES": "8x4", "EXTRA": "32"}),
        ("9x4, 32 CTAs poll", CHECK, {"SHAPES": "9x4", "POLLERS": "32"}),
        ("9x4, 2 CTAs poll", CHECK, {"SHAPES": "9x4", "POLLERS": "2"}),
        ("12x4", CHECK, {"SHAPES": "12x4"}),
    ],
    "isa": [
        ("9x4 loop, each poll load waited for at once", ISA, {**LOOP, "SETTLE": "1"}),
        ("9x4 loop, the clock read before the load", ISA, {**LOOP, "CLOCKFIRST": "1"}),
        ("9x4 loop, no clock in it", ISA, {**LOOP, "POLLROUNDS": "20000"}),
        ("9x4 loop, pause of 16 units", ISA, {**LOOP, "SLEEP": "16"}),
    ],
}
KNOBS = [
    "SHAPES",
    "REGIONS",
    "VOLATILE",
    "SLEEP",
    "POLLROUNDS",
    "EXTRA",
    "POLLERS",
    "SETTLE",
    "CLOCKFIRST",
    "BEFORE",
    "AFTER",
]


def main():
    if BATCH not in BATCHES:
        raise SystemExit(f"BATCH {BATCH!r}: one of {sorted(BATCHES)}")
    for name, tool, changes in BATCHES[BATCH]:
        env = {k: v for k, v in os.environ.items() if k not in KNOBS}
        env.update(DESIGN="v1", RANKS="6", DIAG=LAUNCHES, TICKS=BOUND, **changes)
        started = time.perf_counter()
        try:
            run = subprocess.run(
                [sys.executable, os.path.join(HERE, tool)],
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
            out, code = run.stdout + run.stderr, run.returncode
        except subprocess.TimeoutExpired as late:
            out, code = f"{late.stdout or ''}{late.stderr or ''}", "timeout"
        lines = out.splitlines()
        clean = code == 0 and (tool == ISA or "PASSED 0" in lines)
        print(
            f"=== {name}: {changes}, exit {code}, {time.perf_counter() - started:.0f} s",
            flush=True,
        )
        if tool == CHECK and clean:
            shown = [
                ln
                for ln in lines
                if ln.lstrip().startswith(("diag", "the CTAs", "a key"))
            ]
            shown = shown[:6] + shown[-1:] if len(shown) > 7 else shown
        else:
            shown = lines if clean else lines[-40:]
        for ln in shown:
            print(ln[:1500], flush=True)
        if not clean:
            print(f"BATCH {BATCH} STOPPED at {name!r}", flush=True)
            raise SystemExit(1)
    print(f"BATCH {BATCH} DONE", flush=True)


if __name__ == "__main__":
    main()
