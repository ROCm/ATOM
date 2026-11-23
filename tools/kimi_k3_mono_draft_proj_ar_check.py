# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 DSpark draft projection all-reduce (``draft_proj_ar_gfx1250``:
o_proj, or the MLP's down_proj, with its activation quant and its TP
all-reduce in one launch) against the production sequence, fed a rank's TP
shard: ``per_1x128_e8m0_quant`` (``dynamic_per_group_scaled_quant``), the
FlyDSL mxfp8_128 GEMM the dispatch picks at the width, and the all-reduce's
rank-order fp32 sum of the ranks' BF16 partials (aiter's custom all-reduce's
arithmetic; at one rank the partial itself).

One process: one rank on one GPU. Under torchrun (``--nproc_per_node=N``):
N ranks, their banks mapped over torch.symm_mem (MORI) as
``kimi_k3_mono_tail_check.py`` maps them, a rank barrier before every launch.
Before any launch every rank checks it holds N non-null, distinct peer
pointers and the ranks vote on it (exit 3 otherwise). The banks lie between
two 64 KB guard bands of a byte pattern, which every rank checks after every
eager launch and every graph replay (exit 4 if one was written).

Per width and projection (CHECK): resources; ITERS eager launches and
REPLAYS replays of a captured launch on fresh inputs, each bit for bit; the
polls' give-up mark. Then (BENCH) CUDA-graph time a launch against the
production quant + GEMM (no all-reduce: one rank), inputs and weights
rotating over COPIES sets, the graphs' replays alternating.

    python tools/kimi_k3_mono_draft_proj_ar_check.py
    env: ROWS (1,7,14,21,28,35,42,49,56), KINDS (o_proj,down), SHARD_TP (8:
    the TP the shard's K is cut for), CHECK, BENCH, ITERS (2), REPLAYS (20),
    GN (32), REPS (9), COPIES (16), BOUND (4194304; 0: unbounded polls),
    PROFILE (1: a profile build's per-CTA stamps, summarized)

Every kernel, buffer and graph stays referenced in KEEP to the end.
"""

import json
import os
import statistics

import torch
import torch.distributed as dist
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import draft_proj_ar_gfx1250 as dp
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

KEEP = []
ROWS = [int(r) for r in os.environ.get("ROWS", "1,7,14,21,28,35,42,49,56").split(",")]
KINDS = os.environ.get("KINDS", "o_proj,down").split(",")
SHARD_TP = int(os.environ.get("SHARD_TP", "8"))
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
ITERS = int(os.environ.get("ITERS", "2"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
GN = int(os.environ.get("GN", "32"))
REPS = int(os.environ.get("REPS", "9"))
COPIES = int(os.environ.get("COPIES", "16"))
BOUND = int(os.environ.get("BOUND", str(1 << 22))) or None
PROFILE = os.environ.get("PROFILE", "0") == "1"
WIDTH = {"o_proj": 8192 // SHARD_TP, "down": 14336 // SHARD_TP}

MULTI = "LOCAL_RANK" in os.environ
if MULTI:
    dist.init_process_group("gloo")
    RANK, WORLD = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
else:
    RANK, WORLD = 0, 1
dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(11 + RANK)


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def fp8_linear(rows, k, scale=0.02):
    q, s = quant_weight_online(
        bf16(rows, k, scale=scale),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


SHARDS = {
    kind: [fp8_linear(HIDDEN, WIDTH[kind]) for _ in range(COPIES if BENCH else 1)]
    for kind in KINDS
}
KEEP.append(SHARDS)

# The banks, between two guard bands: GUARD bytes of PATTERN on either side
# of the bytes the kernels may touch, checked on every rank after every
# launch (a store outside a peer's bank does not fault on this platform).
GUARD = 65536
PATTERN = 0xA5
nbytes = lay.peer_bytes(WORLD, max(ROWS))
if MULTI:
    import mori.allocator  # noqa: F401
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.set_backend("MORI")
    bank = symm_mem.empty(GUARD + nbytes + GUARD, dtype=torch.uint8, device=dev)
    bank.zero_()
    bank[:GUARD] = PATTERN
    bank[GUARD + nbytes :] = PATTERN
    torch.cuda.synchronize()
    ptrs = list(symm_mem.rendezvous(bank, dist.group.WORLD).buffer_ptrs)
    ptrs[RANK] = bank.data_ptr()
    # every rank holds WORLD non-null, distinct peer pointers, its own among
    # them, or no rank launches anything
    mapped = torch.tensor(
        [
            int(
                len(ptrs) == WORLD
                and all(int(ptr) != 0 for ptr in ptrs)
                and len({int(ptr) for ptr in ptrs}) == WORLD
            )
        ]
    )
    dist.all_reduce(mapped, op=dist.ReduceOp.MIN)
    if not mapped.item():
        print(
            json.dumps({"rank": RANK, "abort": "peer pointers", "count": len(ptrs),
                        "null": sum(int(ptr) == 0 for ptr in ptrs),
                        "distinct": len({int(ptr) for ptr in ptrs})}),
            flush=True,
        )  # fmt: skip
        raise SystemExit(3)
else:
    bank = torch.zeros(GUARD + nbytes + GUARD, dtype=torch.uint8, device=dev)
    bank[:GUARD] = PATTERN
    bank[GUARD + nbytes :] = PATTERN
    ptrs = [bank.data_ptr()]
# the kernels' bank base: past the front guard
peers = torch.tensor([int(ptr) + GUARD for ptr in ptrs], dtype=torch.int64, device=dev)
epochs = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
flags = torch.zeros(4, dtype=torch.int32, device=dev)
KEEP += [bank, peers, epochs, flags]
GUARD_CHECKS = [0]


def guards_intact():
    """This rank's guard bands still hold the pattern (after a launch: no
    rank stored outside this rank's banks)."""
    torch.cuda.synchronize()
    GUARD_CHECKS[0] += 1
    front = bool((bank[:GUARD] == PATTERN).all())
    back = bool((bank[GUARD + nbytes :] == PATTERN).all())
    if not (front and back):
        print(
            json.dumps({"rank": RANK, "abort": "guard band overwritten",
                        "front_intact": front, "back_intact": back,
                        "checks": GUARD_CHECKS[0]}),
            flush=True,
        )  # fmt: skip
        raise SystemExit(4)


def barrier():
    torch.cuda.synchronize()
    if MULTI:
        dist.barrier()


def inputs(rows, k):
    """A projection's BF16 activation: a group with a large amax, an
    all-zero group (the amax floor), a few denormals."""
    x = bf16(rows, k, scale=0.5)
    x[0, :64] *= 40.0
    x[rows - 1, 128:256] = 0.0
    # BF16 denormals of both signs: the HIP quant flushes f32 denormals
    tiny = torch.tensor([3, 64, -32768 + 5, 127, -32768 + 96, 1, 0, -32768],
                        dtype=torch.int16, device=dev)  # fmt: skip
    x.view(torch.int16)[rows // 2, 320:328] = tiny
    return x


class Mono:
    def __init__(self, rows, k):
        self.kernel = dp.build_draft_proj_ar(rows, WORLD, k, poll_bound=BOUND)
        self.out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        KEEP.append(self)

    def __call__(self, x, shard):
        p = torch.Tensor.data_ptr
        values = {
            "x": p(x), "w": p(shard[0]), "w_scale": p(shard[1]), "out": p(self.out),
            "peers": p(peers), "epochs": p(epochs), "rank": RANK,
        }  # fmt: skip
        if BOUND is not None:
            values["flags"] = p(flags)
        self.kernel.launcher(
            *self.kernel.abi.pack(values), stream=torch.cuda.current_stream()
        )


def production(x, shard):
    """The rank's partial as LinearBase.forward computes it."""
    x8, xs = per_1x128_e8m0_quant(x, dtypes.fp8, True)
    return gemm_a8w8_blockscale_bpreshuffle(x8, shard[0], xs, shard[1], torch.bfloat16)


def reduced(partial):
    """The all-reduce of the ranks' BF16 partials: rank order, fp32, one
    rounding to BF16."""
    local = partial.cpu()
    if not MULTI:
        return local
    every = [torch.empty_like(local) for _ in range(WORLD)]
    dist.all_gather(every, local)
    acc = every[0].float()
    for r in range(1, WORLD):
        acc = acc + every[r].float()
    return acc.to(torch.bfloat16)


def bits(t):
    return t.contiguous().view(torch.int16)


def resources(kernel):
    (launcher,) = compile_kernels([(kernel.launcher, kernel.abi)])
    res = launcher_resources(launcher)
    hazards = [
        h
        for r in res
        for h, bad in (
            ("scratch", r["scratch"]),
            ("vgpr_spill", r["vgpr_spill"]),
            ("sgpr_spill", r["sgpr_spill"]),
        )
        if bad
    ]
    return res, hazards


def graph_of(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            barrier()
            fn()
            guards_intact()
    torch.cuda.current_stream().wait_stream(side)
    barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        fn()
    KEEP.append((graph, side, fn))
    return graph


def mismatch(got, want):
    bad = bits(got) != bits(want)
    d = (got.double() - want.double()).abs()
    return {
        "fraction": bad.double().mean().item(),
        "max_abs": d.max().item(),
        "max_rel": (d.max() / want.double().abs().max()).item(),
    }


def check(rows, kind):
    k = WIDTH[kind]
    mono = Mono(rows, k)
    res, hazards = resources(mono.kernel)
    shard = SHARDS[kind][0]
    report = {
        "rank": RANK, "world": WORLD, "rows": rows, "kind": kind, "k": k,
        "kernel": mono.kernel.name, "resources": res, "hazards": hazards,
    }  # fmt: skip
    flags.zero_()
    eager, last = [], None
    for _ in range(ITERS):
        x = inputs(rows, k)
        barrier()
        mono(x, shard)
        guards_intact()
        partial = production(x, shard)
        torch.cuda.synchronize()
        want, got = reduced(partial), mono.out.cpu()
        eager.append(torch.equal(bits(got), bits(want)))
        if not eager[-1] and "eager_mismatch" not in report:
            report["eager_mismatch"] = mismatch(got, want)
        report["moving"] = last is None or not torch.equal(got, last)
        last = got
    gx = inputs(rows, k)
    graph = graph_of(lambda: mono(gx, shard))
    replays = []
    for _ in range(REPLAYS):
        x = inputs(rows, k)
        gx.copy_(x)
        barrier()
        graph.replay()
        guards_intact()
        want, got = reduced(production(x, shard)), mono.out.cpu()
        replays.append(torch.equal(bits(got), bits(want)))
        if not replays[-1] and "replay_mismatch" not in report:
            report["replay_mismatch"] = mismatch(got, want)
    barrier()
    report["eager_exact"] = eager
    report["replays_exact"] = sum(replays)
    report["replays"] = len(replays)
    report["gave_up"] = int(flags[0])
    report["epochs_agree"] = bool((epochs == epochs[0]).all())
    report["guard_checks"] = GUARD_CHECKS[0]
    report["ok"] = (
        not hazards and all(eager) and all(replays) and report["gave_up"] == 0
    )
    print(json.dumps(report), flush=True)
    return mono


def timed(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    if MULTI:
        dist.barrier()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def bench(rows, kind, mono):
    k = WIDTH[kind]
    xs = [inputs(rows, k) for _ in range(COPIES)]
    KEEP.append(xs)

    def run_mono():
        for i in range(GN):
            mono(xs[i % COPIES], SHARDS[kind][i % COPIES])

    def run_prod():
        for i in range(GN):
            KEEP.append(production(xs[i % COPIES], SHARDS[kind][i % COPIES]))

    graphs = {"mono": graph_of(run_mono), "production_quant_gemm": graph_of(run_prod)}
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, graph in graphs.items():
            times[name].append(timed(graph))
            guards_intact()
    barrier()
    if RANK == 0:
        out = {"rows": rows, "kind": kind, "world": WORLD, "gn": GN}
        for name, t in times.items():
            out[name] = {
                "median": round(statistics.median(t), 2),
                "min": round(min(t), 2),
                "max": round(max(t), 2),
            }
        print(json.dumps({"bench": out}), flush=True)


def profile(rows, kind):
    """A profile build's stamps over GN launches in a graph on rotating
    inputs: per phase, us from the CTA's own start (the XCDs' clocks are
    offset), the CTAs' median and the slowest CTA, medians over launches."""
    k = WIDTH[kind]
    kernel = dp.build_draft_proj_ar(rows, WORLD, k, poll_bound=BOUND, profile=True)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    stamps = torch.zeros(lay.CTAS * dp.STAMPS, dtype=torch.int32, device=dev)
    snaps = torch.zeros(GN, lay.CTAS * dp.STAMPS, dtype=torch.int32, device=dev)
    xs = [inputs(rows, k) for _ in range(COPIES)]
    KEEP.extend([kernel, out, stamps, snaps, xs])
    p = torch.Tensor.data_ptr

    def run():
        for i in range(GN):
            shard = SHARDS[kind][i % len(SHARDS[kind])]
            kernel.launcher(
                *kernel.abi.pack({
                    "x": p(xs[i % COPIES]), "w": p(shard[0]), "w_scale": p(shard[1]),
                    "out": p(out), "peers": p(peers), "epochs": p(epochs),
                    "rank": RANK, "flags": p(flags), "stamps": p(stamps),
                }),
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
            snaps[i].copy_(stamps)

    graph = graph_of(run)
    for _ in range(3):
        barrier()
        graph.replay()
    barrier()
    guards_intact()
    st = snaps.cpu().long().view(GN, lay.CTAS, dp.STAMPS)
    delta = ((st - st[:, :, :1]) & 0xFFFFFFFF).float() / 100.0
    names = ["start", "landed", "quantized", "staged", "pushed", "summed", "end"]
    report = {"rows": rows, "kind": kind, "rank": RANK, "world": WORLD}
    for i, name in enumerate(names[1:], 1):
        per_launch = delta[GN // 2 :, :, i]
        report[name] = {
            "cta_median": round(per_launch.median(dim=1).values.median().item(), 2),
            "cta_max": round(per_launch.max(dim=1).values.median().item(), 2),
        }
    print(json.dumps({"profile": report}), flush=True)


for rows in ROWS:
    for kind in KINDS:
        mono = check(rows, kind) if CHECK else Mono(rows, WIDTH[kind])
        if BENCH:
            bench(rows, kind, mono)
        if PROFILE:
            profile(rows, kind)
barrier()
