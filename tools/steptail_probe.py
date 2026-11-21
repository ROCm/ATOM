# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The production step-tail kernels on one GPU in the serving shapes (TP8 shard
sizes): kernel names and durations under the torch profiler, and CUDA-graph
replay time an op where it captures. Also the sha256 of the AITER sources the
step tail mirrors.

    python tools/steptail_probe.py
    env: ROWS (8,16,40,64), ITERS (40), REPS (16)
"""

import collections
import hashlib
import os
import statistics
import time

import torch

ROWS = [int(r) for r in os.environ.get("ROWS", "8,16,40,64").split(",")]
ITERS = int(os.environ.get("ITERS", "40"))
REPS = int(os.environ.get("REPS", "16"))
TP, HIDDEN, VOCAB, RANK = 8, 7168, 163840, 256
VR = VOCAB // TP
dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(11)
KEEP = []


def sources():
    import aiter

    root = os.path.dirname(aiter.__file__)
    for rel in (
        "ops/flydsl/kernels/gemm_a16w16_kernel_gfx1250.py",
        "ops/flydsl/gemm_a16w16_gfx1250.py",
        "ops/flydsl/gemm_a16w16_policy.py",
        "ops/flydsl/kernels/topk/topk_per_row_argmax.py",
        "ops/topk_select.py",
        "tuned_gemm.py",
        "configs/bf16_tuned_gemm.csv",
    ):
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"src {rel}: missing")
            continue
        with open(path, "rb") as f:
            data = f.read()
        print(f"src {rel}: {hashlib.sha256(data).hexdigest()[:16]} {len(data)} B")


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def kernels_of(label, fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            fn()
        torch.cuda.synchronize()
    by = collections.defaultdict(list)
    order = []
    for e in prof.events():
        if "cuda" not in str(e.device_type).lower():
            continue
        name = e.name[:72]
        if name not in by:
            order.append(name)
        by[name].append(e.time_range.elapsed_us())
    total = 0.0
    for name in order:
        med = statistics.median(by[name])
        per = len(by[name]) / ITERS
        total += med * per
        print(f"  {label}: {med:7.1f} us x{per:4.1f} {name}")
    print(f"  {label}: kernel sum {total:.1f} us")


def graph_time(label, fn):
    try:
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(REPS):
                fn()
        KEEP.append(graph)
        times = []
        for _ in range(ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graph.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / REPS * 1e6)
        times.sort()
        print(
            f"  {label}: graph {statistics.median(times):.1f} us/op "
            f"(p10 {times[len(times) // 10]:.1f})"
        )
    except Exception as err:  # noqa: BLE001 - a probe: report and go on
        print(f"  {label}: graph capture failed: {type(err).__name__}: {err}")


def main():
    from aiter import topk_select
    from aiter.tuned_gemm import tgemm

    from atom.model_ops.dspark_markov_argmax import dspark_markov_argmax
    from atom.model_ops.rejection_sampler import rejection_sample

    sources()
    print(
        f"lower bound head shard: {VR * HIDDEN * 2 / 17.5e12 * 1e6:.1f} us "
        f"({VR * HIDDEN * 2 / 2**20:.0f} MB)"
    )
    print(f"lower bound W2 shard a position: {VR * RANK * 2 / 17.5e12 * 1e6:.2f} us")
    head = bf16(VR, HIDDEN, scale=0.02)
    w1, w2 = bf16(VOCAB, RANK, scale=0.05), bf16(VOCAB, RANK, scale=0.05)
    KEEP.extend([head, w1, w2])
    for rows in ROWS:
        width(
            rows,
            head,
            w1,
            w2,
            tgemm,
            topk_select,
            rejection_sample,
            dspark_markov_argmax,
        )


def width(
    rows, head, w1, w2, tgemm, topk_select, rejection_sample, dspark_markov_argmax
):
    """The production step-tail kernels of a ``rows``-row step."""
    qlen = 8 if rows % 8 == 0 else 5
    bs, k = rows // qlen, qlen - 1
    x = bf16(rows, HIDDEN)
    logits = bf16(rows, VOCAB, scale=3.0)
    ids = torch.randint(0, VOCAB, (rows + 1,), device=dev, dtype=torch.int32)
    t_idx = torch.tensor(
        [r * qlen + p for r in range(bs) for p in range(k)], device=dev
    )
    b_idx = torch.tensor([r * qlen + k for r in range(bs)], device=dev)
    cu = torch.arange(1, bs + 1, device=dev, dtype=torch.int32) * k
    KEEP.extend([x, logits, ids, t_idx, b_idx, cu])
    print(f"rows {rows} (bs {bs}, {k} drafts)")

    def lm_head():
        return tgemm.mm(x, head, None)

    def select():
        bonus = torch.index_select(logits, 0, b_idx)
        target = torch.index_select(logits, 0, t_idx)
        return bonus, target

    def sample():
        bonus, target = select()
        bonus_ids = topk_select(bonus, 1, tie="low")[1].view(-1)
        return rejection_sample(ids, t_idx, k, cu, None, target, bonus_ids, b_idx)

    kernels_of(f"head s{rows}", lm_head)
    graph_time(f"head s{rows}", lm_head)
    kernels_of(f"sample s{rows}", sample)
    graph_time(f"sample s{rows}", sample)

    # the draft block: bs requests x k positions over the whole vocab
    base = bf16(bs, k, VOCAB, scale=3.0)
    block = torch.zeros(bs, k + 1, dtype=torch.int32, device=dev)
    block[:, 0] = torch.randint(0, VOCAB, (bs,), device=dev)
    xd = bf16(bs * k, HIDDEN)
    KEEP.extend([base, block, xd])

    def draft_head():
        return tgemm.mm(xd, head, None)

    def markov():
        for p in range(k):
            dspark_markov_argmax(base[:, p], block[:, p], w1, w2, block[:, p + 1])

    kernels_of(f"draft head s{bs * k}", draft_head)
    graph_time(f"draft head s{bs * k}", draft_head)
    kernels_of(f"markov bs{bs} T{k}", markov)
    graph_time(f"markov bs{bs} T{k}", markov)


if __name__ == "__main__":
    main()
