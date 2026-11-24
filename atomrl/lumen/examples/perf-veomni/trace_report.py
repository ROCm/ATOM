#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""One-shot report for a VeOmni torch-profiler trace of a host-bound step.

Consolidates every analysis that was needed to explain a 4.6 s Qwen-Image step
on 8 GPUs. Reads a chrome trace and answers, in order:

  0. which steps are in the trace, and how long they are
  1. is the step compute-bound at all (GPU busy / wall)
  2. where the GPU time goes, by category and by kernel
  3. which host ops own the wall clock
  4. are the collectives overlapped with compute
  5. is the idle time many small gaps or a few big stalls
  6. per-phase GPU busy, and per-candidate host/GPU/launch cost (needs --stack
     trace, i.e. one captured with --train.profile.with_stack true)

Usage:
    trace_report.py <trace.json.gz>            # sections 0-5
    trace_report.py <trace.json.gz> --stack    # adds 6

Read absolute ms from a trace captured WITHOUT with_stack; stack sampling
inflates the step (measured: 4584 ms -> 5240 ms) and therefore manufactures
idle. Use the with_stack trace for attribution and shares only.
"""

import argparse
import gzip
import json
import re
from bisect import bisect_left
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
COMM_RX = re.compile(r"nccl|rccl", re.IGNORECASE)

# Python frames that correspond to training phases. Keys are display order.
PHASES = [
    ("condition model (VAE + text encode)", r"get_condition"),
    ("DiT forward", r"predict_noise"),
    ("backward", r"run_backward"),
    ("clip_grad_norm", r"clip_grad"),
    ("optimizer.step", r"_fused_adamw_|optimizer.*step"),
    ("gc.collect + empty_cache", r"built-in function collect>|_cuda_emptyCache"),
    ("Tensor.item (host drains the queue)", r"built-in method item of Tensor"),
    ("dataloader", r"_BaseDataLoaderIter|fetch|collate"),
]

# Op-replacement candidates, ranked later by the host time they occupy.
CANDIDATES = [
    (
        "nn.Linear / F.linear",
        r"built-in function linear$|nn/modules/linear\.py.*forward",
    ),
    (
        "RMSNorm (diffusers normalization.forward)",
        r"diffusers/models/normalization\.py\(541\)",
    ),
    ("AdaLayerNorm _modulate", r"transformer_qwenimage\.py\(628\): _modulate"),
    ("RoPE apply_rotary_emb_qwen", r"apply_rotary_emb|view_as_complex|view_as_real"),
    (
        "LayerNorm (affine-free -- unsafe to swap, see handoff)",
        r"built-in method layer_norm",
    ),
    ("scaled_dot_product_attention", r"scaled_dot_product_attention"),
    (
        "gradient-checkpoint hooks (incl. recompute)",
        r"utils/checkpoint\.py\((1077|1139|1149)\)",
    ),
    ("FSDP2 collectives (host side)", r"_fsdp_collectives\.py"),
    ("DTensor spec bookkeeping", r"_dtensor_spec\.py"),
]


def load(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def merge(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s > out[-1][1]:
            out.append([s, e])
        else:
            out[-1][1] = max(out[-1][1], e)
    return out


def total(merged):
    return sum(e - s for s, e in merged)


def intersect(a, b):
    i = j = 0
    acc = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            acc += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return acc


def shorten(name, width=66):
    name = re.sub(r"void\s+", "", name)
    name = re.sub(r"<.*?>", "<..>", name)
    return name if len(name) <= width else name[: width - 3] + "..."


def outermost_spans(by_thread, pattern):
    """Spans of the outermost frames matching pattern, per thread.

    Taking outermost only is what keeps a recursive or repeated frame inside
    one phase from being counted twice.
    """
    rx = re.compile(pattern)
    spans = []
    for evs in by_thread.values():
        until = -1.0
        for e in evs:
            if not rx.search(e["name"]) or e["ts"] < until:
                continue
            spans.append((e["ts"], e["ts"] + e["dur"]))
            until = e["ts"] + e["dur"]
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument(
        "--stack",
        action="store_true",
        help="trace has python_function events; add section 6",
    )
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    events = load(args.trace)["traceEvents"]

    # ---- 0. steps ---------------------------------------------------------
    # Each step appears three times: one host-side user_annotation spanning the
    # step, plus gpu_user_annotation copies projected onto the streams. Only the
    # host one delimits the step. Taking the last event by timestamp across all
    # three picks a ~140 ms GPU copy and silently halves the window -- that bug
    # produced a 25.8% busy figure that happened to look plausible.
    steps = [
        e
        for e in events
        if e.get("name", "").startswith("ProfilerStep#")
        and e.get("cat") == "user_annotation"
    ]
    steps.sort(key=lambda e: e["ts"])
    if not steps:
        raise SystemExit("no ProfilerStep# user_annotation events -- wrong trace?")
    n = len(steps)
    s0 = steps[0]["ts"]
    e0 = steps[-1]["ts"] + steps[-1]["dur"]
    wall = e0 - s0
    print(
        f"=== 0. window: {n} step(s), {wall / 1e3:.1f} ms total, {wall / 1e3 / n:.1f} ms/step ==="
    )
    for e in steps:
        print(f"  {e['name']}  {e['dur'] / 1e3:.1f} ms")

    gpu = [
        e
        for e in events
        if e.get("cat") in GPU_CATS
        and e.get("ph") == "X"
        and s0 <= e["ts"] <= e0
        and e["dur"] > 0
    ]
    if not gpu:
        raise SystemExit("no GPU events in the window")
    busy = merge([(e["ts"], e["ts"] + e["dur"]) for e in gpu])
    sum_all = sum(e["dur"] for e in gpu)

    # ---- 1. compute-bound? ------------------------------------------------
    print("\n=== 1. is the step compute-bound? ===")
    print(f"  wall                    {wall / 1e3 / n:9.1f} ms/step")
    print(
        f"  GPU busy (union)        {total(busy) / 1e3 / n:9.1f} ms/step  {total(busy) / wall * 100:5.1f}% of wall"
    )
    print(
        f"  GPU idle                {(wall - total(busy)) / 1e3 / n:9.1f} ms/step  {(wall - total(busy)) / wall * 100:5.1f}% of wall"
    )
    print(
        f"  sum over streams        {sum_all / 1e3 / n:9.1f} ms/step  (overlap, so > busy)"
    )

    # ---- 2. GPU time -----------------------------------------------------
    def classify(nm):
        low = nm.lower()
        if COMM_RX.search(low):
            return "communication (RCCL)"
        if nm.startswith("Cijk_") or "gemm" in low or "hipblas" in low:
            return "GEMM"
        if "attn" in low or "bwd_kernel" in low or "softmax" in low or "flash" in low:
            return "attention"
        if "conv" in low or "miopen" in low or "implicit" in low:
            return "convolution"
        if "memcpy" in low:
            return "memcpy"
        if "adam" in low or "multi_tensor" in low:
            return "optimizer"
        if "norm" in low:
            return "normalisation"
        if "reduce" in low:
            return "reduction"
        if (
            "elementwise" in low
            or "vectorized" in low
            or "copy" in low
            or "cat" in low
            or "fill" in low
        ):
            return "elementwise/copy"
        return "other"

    cat_t, cat_c, nm_t, nm_c = (
        defaultdict(float),
        defaultdict(int),
        defaultdict(float),
        defaultdict(int),
    )
    for e in gpu:
        c = classify(e["name"])
        cat_t[c] += e["dur"]
        cat_c[c] += 1
        nm_t[e["name"]] += e["dur"]
        nm_c[e["name"]] += 1
    print("\n=== 2. GPU time by category (sum over streams) ===")
    print(
        f"  {'category':22} {'ms/step':>9} {'% GPU':>7} {'% wall':>7} {'calls/step':>11}"
    )
    for c, t in sorted(cat_t.items(), key=lambda kv: -kv[1]):
        print(
            f"  {c:22} {t / 1e3 / n:9.2f} {t / sum_all * 100:6.1f}% {t / wall * 100:6.1f}% {cat_c[c] / n:11.1f}"
        )
    print(f"\n  top {args.top} kernels")
    print(f"  {'ms/step':>9} {'% GPU':>7} {'calls/step':>11} {'us/call':>9}  kernel")
    for nm, t in sorted(nm_t.items(), key=lambda kv: -kv[1])[: args.top]:
        print(
            f"  {t / 1e3 / n:9.2f} {t / sum_all * 100:6.1f}% {nm_c[nm] / n:11.1f} {t / nm_c[nm]:9.1f}  {shorten(nm)}"
        )

    # ---- 3. host ops ------------------------------------------------------
    cpu = [
        e
        for e in events
        if e.get("cat") == "cpu_op"
        and e.get("ph") == "X"
        and s0 <= e["ts"] <= e0
        and e["dur"] > 0
    ]
    if cpu:
        per_thread = defaultdict(list)
        for e in cpu:
            per_thread[e.get("tid")].append(e)
        self_t, calls = defaultdict(float), defaultdict(int)
        for evs in per_thread.values():
            evs.sort(key=lambda e: (e["ts"], -e["dur"]))
            stack = []
            for e in evs:
                s, en = e["ts"], e["ts"] + e["dur"]
                while stack and stack[-1][1] <= s:
                    stack.pop()
                if stack:
                    self_t[stack[-1][2]] -= min(en, stack[-1][1]) - s
                self_t[e["name"]] += e["dur"]
                calls[e["name"]] += 1
                stack.append((s, en, e["name"]))
        print(f"\n=== 3. top {args.top} host ops by self time (all threads) ===")
        print(f"  {'ms/step':>9} {'% wall':>7} {'calls/step':>11}  op")
        for nm, t in sorted(self_t.items(), key=lambda kv: -kv[1])[: args.top]:
            print(
                f"  {t / 1e3 / n:9.2f} {t / wall * 100:6.1f}% {calls[nm] / n:11.1f}  {shorten(nm)}"
            )

        print(
            "\n  per-thread busy (cpu_op union) -- the backward runs on its own thread,"
        )
        print("  so a single-thread view reports the main thread as mostly 'idle'")
        for tid, evs in sorted(
            per_thread.items(),
            key=lambda kv: -total(
                merge([(e["ts"], e["ts"] + e["dur"]) for e in kv[1]])
            ),
        )[:4]:
            b = total(merge([(e["ts"], e["ts"] + e["dur"]) for e in evs]))
            print(
                f"    tid {tid}: {b / 1e3 / n:8.1f} ms/step  {b / wall * 100:5.1f}% of wall"
            )

    # ---- 4. overlap -------------------------------------------------------
    comm = merge(
        [(e["ts"], e["ts"] + e["dur"]) for e in gpu if COMM_RX.search(e["name"])]
    )
    comp = merge(
        [(e["ts"], e["ts"] + e["dur"]) for e in gpu if not COMM_RX.search(e["name"])]
    )
    ov = intersect(comm, comp)
    print("\n=== 4. are the collectives overlapped with compute? ===")
    print(f"  communication busy      {total(comm) / 1e3 / n:9.1f} ms/step")
    print(f"  compute busy            {total(comp) / 1e3 / n:9.1f} ms/step")
    print(f"  overlapped              {ov / 1e3 / n:9.1f} ms/step")
    if total(comm):
        print(
            f"  => {ov / total(comm) * 100:.1f}% of communication hidden; {(total(comm) - ov) / 1e3 / n:.1f} ms/step exposed"
        )
    print("  NOTE a collective kernel's duration includes waiting for peers, so this")
    print(
        "  number grows when the host slows down. It is partly rank skew, not transfer."
    )

    # ---- 5. idle shape ----------------------------------------------------
    gaps = [
        busy[i][0] - busy[i - 1][1]
        for i in range(1, len(busy))
        if busy[i][0] > busy[i - 1][1]
    ]
    idle = wall - total(busy)
    print("\n=== 5. is the idle many small gaps or a few big stalls? ===")
    print(f"  idle {idle / 1e3 / n:.1f} ms/step in {len(gaps) / n:.0f} gaps/step")
    print(f"    {'gap size':>16} {'count/step':>11} {'ms/step':>9} {'% of idle':>10}")
    for lo, hi in [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, 10**9)]:
        sel = [g for g in gaps if lo <= g < hi]
        if not sel:
            continue
        label = f"{lo}-{hi} us" if hi < 10**9 else f">{lo} us"
        print(
            f"    {label:>16} {len(sel) / n:11.0f} {sum(sel) / 1e3 / n:9.1f} {sum(sel) / (idle or 1) * 100:9.1f}%"
        )
    if gaps:
        print(
            f"  largest gaps (ms): {', '.join(f'{g / 1e3:.0f}' for g in sorted(gaps, reverse=True)[:8])}"
        )

    # ---- 6. phases and candidates ----------------------------------------
    if not args.stack:
        print(
            "\n(sections 6 skipped; pass --stack with a with_stack trace for phase attribution)"
        )
        return
    py = [
        e
        for e in events
        if e.get("cat") == "python_function"
        and e.get("ph") == "X"
        and s0 <= e["ts"] <= e0
    ]
    if not py:
        raise SystemExit(
            "--stack given but no python_function events: capture with --train.profile.with_stack true"
        )
    by_thread = defaultdict(list)
    for e in py:
        by_thread[e.get("tid")].append(e)
    for evs in by_thread.values():
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))

    print("\n=== 6a. GPU busy inside each phase ===")
    print(f"  {'phase ms':>9} {'GPU busy':>9} {'busy %':>7}  phase")
    for label, pattern in PHASES:
        m = merge(outermost_spans(by_thread, pattern))
        if not m:
            continue
        b = intersect(m, busy)
        print(
            f"  {total(m) / 1e3 / n:9.1f} {b / 1e3 / n:9.1f} {b / (total(m) or 1) * 100:6.1f}%  {label}"
        )

    kern_by_corr = defaultdict(float)
    for e in events:
        if e.get("cat") in GPU_CATS and e.get("ph") == "X":
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                kern_by_corr[c] += e["dur"]
    launches = sorted(
        (e["ts"], (e.get("args") or {}).get("correlation"))
        for e in events
        if e.get("cat") == "cuda_runtime" and e.get("ph") == "X" and s0 <= e["ts"] <= e0
    )
    launch_ts = [t for t, _ in launches]

    print("\n=== 6b. op-replacement candidates ===")
    print("  host ms is the wall time the frame occupies -- in a host-bound step that,")
    print("  not the GPU ms, is what fusing it removes.")
    print(
        f"  {'host ms':>9} {'% step':>7} {'GPU ms':>8} {'launches':>9} {'calls':>7} {'l/call':>7}  candidate"
    )
    rows = []
    for label, pattern in CANDIDATES:
        spans = outermost_spans(by_thread, pattern)
        host = sum(b - a for a, b in spans)
        g = 0.0
        nl = 0
        for a, b in spans:
            i = bisect_left(launch_ts, a)
            while i < len(launches) and launches[i][0] <= b:
                nl += 1
                g += kern_by_corr.get(launches[i][1], 0.0)
                i += 1
        rows.append((label, host, g, nl, len(spans)))
    for label, host, g, nl, c in sorted(rows, key=lambda r: -r[1]):
        print(
            f"  {host / 1e3 / n:9.1f} {host / wall * 100:6.1f}% {g / 1e3 / n:8.1f} {nl / n:9.0f} {c / n:7.0f} "
            f"{(nl / c if c else 0):7.1f}  {label}"
        )


if __name__ == "__main__":
    main()
