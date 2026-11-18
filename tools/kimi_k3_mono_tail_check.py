# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 K2 tail at TP N against the three launches it replaces
(gathering all-reduce, AITER rmsnorm2d_fwd, replicated finalize), through the
runner's own ``_replicated_tail`` on the runner's own K2 build: front, lite
finish and fused MoE feed every path, ITERS back-to-back launches with the
step input and the layer weights rotating, each launch's outputs copied out
in stream order. Per rank: the final rows, the reduced shared rows and the
normed routed rows bit for bit, the MoE outputs every path saw identical,
the outputs moving launch to launch, the final rows equal across ranks.
Widths past ``tail_max_rows`` check the runner builds no tail there.

Then CUDA-graph time (a rank barrier before each replay): GN launches of the
tail alone and of the three launches on rotating inputs, and GN K2 layers
(front, lite, fused MoE, then the tail or the three launches) with the
weights rotating, the variants' replays alternating.

    torchrun --nproc_per_node=4 tools/kimi_k3_mono_tail_check.py
    env: ROWS (10,20,30,40,42), ITERS (10), CHECK, BENCH, PROFILE (per-CTA
    stamps of profile builds), SPLIT (0,1: the tail builds compared), BOUND
    (scratch builds: polls give up past it), GN (20), GREPS (7), REPS (4),
    AITER_FLYDSL_USE_HERD

Every kernel, buffer and graph stays referenced in KEEP to the end: a graph
replays only kernels and buffers still alive.
"""

import dataclasses
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from aiter import dtypes, rmsnorm2d_fwd
from aiter.ops.shuffle import shuffle_weight

import atom.models.kimi_k3.mono.runner_gfx1250 as rm
from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT, TOPK
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tk
from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

KEEP = []
ROWS = [int(r) for r in os.environ.get("ROWS", "10,20,30,40,42").split(",")]
ITERS = int(os.environ.get("ITERS", "10"))
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
PROFILE = os.environ.get("PROFILE", "0") == "1"
SPLITS = [s == "1" for s in os.environ.get("SPLIT", "0,1").split(",") if s]
BOUND = int(os.environ["BOUND"]) if os.environ.get("BOUND") else None
GN = int(os.environ.get("GN", "20"))
GREPS = int(os.environ.get("GREPS", "7"))
REPS = int(os.environ.get("REPS", "4"))
COPIES, SETS = 3, 4
SH, INTER, NB, EPS = 768, 384, 4, 1e-5

dist.init_process_group("gloo")
RANK, WORLD = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(11 + RANK)
# replicated across ranks, as the layer's norm and up-projection are
rep = torch.Generator(device=dev).manual_seed(5)


def bf16(*shape, scale=1.0, g=None):
    return (torch.randn(shape, generator=g or gen, device=dev) * scale).to(
        torch.bfloat16
    )


def fp8_preshuffled(rows, cols):
    w = bf16(rows, cols, scale=0.02)
    q, s = quantize_weight_to_fp8_128x128_blockscale(
        w, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
    )
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def u8(*shape, lo=0, hi=256):
    return torch.randint(lo, hi, shape, generator=gen, device=dev, dtype=torch.uint8)


def norm_of(ln_w):
    """The routed norm as the model's RMSNorm runs it."""
    return lambda x: rmsnorm2d_fwd(x, ln_w, EPS, use_model_sensitive_rmsnorm=0)


layers = []
for _ in range(COPIES):
    sgu, sgu_s = fp8_preshuffled(2 * SH, HIDDEN)
    sd, sd_s = fp8_preshuffled(HIDDEN, SH)
    w13s = u8(EXPERTS, 2 * INTER, LATENT // 32, lo=118, hi=126)
    w2s = u8(EXPERTS, LATENT, INTER // 32, lo=118, hi=126)
    ln_w = bf16(LATENT, g=rep) + 1.0
    layers.append(
        SimpleNamespace(
            w_gate=bf16(EXPERTS, HIDDEN, scale=0.02),
            w_ld=bf16(LATENT, HIDDEN, scale=0.02),
            w_sgu=sgu,
            w_sgu_scale=sgu_s,
            w_sd=sd,
            w_sd_scale=sd_s,
            bias=(torch.randn(EXPERTS, generator=gen, device=dev) * 0.1).float(),
            w13=u8(EXPERTS, 2 * INTER, LATENT // 2),
            w2=u8(EXPERTS, LATENT, INTER // 2),
            w13s=w13s,
            w2s=w2s,
            w_up_full=bf16(HIDDEN, LATENT, scale=0.02, g=rep),
            ln_w=ln_w,
            ln_eps=EPS,
            moe=SimpleNamespace(
                experts=SimpleNamespace(w13_weight_scale=w13s, w2_weight_scale=w2s),
                routed_expert_norm=norm_of(ln_w),
            ),
        )
    )
KEEP.append(layers)

import mori.allocator  # noqa: F401
import torch.distributed._symmetric_memory as symm_mem

symm_mem.set_backend("MORI")
bank = symm_mem.empty(
    ar.allreduce_peer_bytes(WORLD, max(ROWS)) + 4096, dtype=torch.uint8, device=dev
)
bank.zero_()
torch.cuda.synchronize()
ptrs = list(symm_mem.rendezvous(bank, dist.group.WORLD).buffer_ptrs)
ptrs[RANK] = bank.data_ptr()
peers = torch.tensor(ptrs, dtype=torch.int64, device=dev)
epochs = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
KEEP += [bank, peers, epochs]

runner = object.__new__(rm.K3Gfx1250MonoDecodeRunner)
runner.device, runner.tp, runner.rank, runner.blocks = dev, WORLD, RANK, 256
runner.k2_peers = SimpleNamespace(peers=peers, epochs=epochs)
runner.dims = SimpleNamespace(sh_inter=SH, inter=INTER)
runner.k2_tail_flags = torch.zeros(tk.tail_flag_words(), dtype=torch.int32, device=dev)
KEEP.append(runner)


def barrier():
    torch.cuda.synchronize()
    dist.barrier()


def gathered_ints(value):
    out = [torch.zeros(1, dtype=torch.int64) for _ in range(WORLD)]
    dist.all_gather(out, torch.tensor([value], dtype=torch.int64))
    return [int(v) for v in out]


def bits(t):
    return t.contiguous().view(torch.int16)


def build(rows):
    """The runner's K2 build at ``rows`` (its own tail decision), and a
    local of it per path: the three launches, and the tail per SPLIT."""
    local = runner._build_routed_k2(rows, NB, False, True, True, (True, True), EPS)
    paths = {"sep": dataclasses.replace(local, tail=None)}
    if local.tail is not None:
        for split in SPLITS:
            tail = tk.build_k2_tail(
                WORLD, rows, EPS, identity_routes=True, split=split, poll_bound=BOUND
            )
            paths["tail_sw" if split else "tail"] = dataclasses.replace(
                local, tail=tail
            )
    KEEP.extend([local, paths])
    return local, paths


def front(local, xs, i, s):
    w = layers[i % COPIES]
    b = local.buffers
    local.front.launcher(
        xs[i % len(xs)].data_ptr(),
        w.w_gate.data_ptr(),
        w.w_ld.data_ptr(),
        w.w_sgu.data_ptr(),
        w.w_sgu_scale.data_ptr(),
        b["router_logits"].data_ptr(),
        b["latent_fp4"].data_ptr(),
        b["latent_scales"].data_ptr(),
        b["shared_hidden"].data_ptr(),
        stream=s,
    )


def k2(local, xs, i, s):
    """One K2 layer: front, lite finish, fused MoE, then the path's tail."""
    w = layers[i % COPIES]
    front(local, xs, i, s)
    runner._finish(w, local, local.buffers["router_logits"], s)
    gathered = runner._fused_moe(w, local, w.moe, s)
    runner._replicated_tail(w, local, w.moe, gathered, s)
    return gathered


def check(rows, local, paths):
    b = local.buffers
    entry = {
        "rank": RANK,
        "world": WORLD,
        "rows": rows,
        "herd": local.herd,
        "fused_moe": local.fused_moe is not None,
        "allreduce": None if local.allreduce is None else local.allreduce.name,
        "tail_built": local.tail is not None,
        "tail_max_rows": tk.tail_max_rows(),
        "paths": {k: getattr(v.tail, "name", None) for k, v in paths.items()},
    }
    xs = [bf16(rows, HIDDEN) for _ in range(ITERS)]
    KEEP.append(xs)
    # every kernel built and launched once first: a rank still compiling
    # holds its peers' polls for seconds
    for path in paths.values():
        k2(path, xs, 0, torch.cuda.current_stream())
    results = {}
    for name, path in paths.items():
        runner.k2_tail_flags.zero_()
        barrier()
        s = torch.cuda.current_stream()
        taken = []
        for i in range(ITERS):
            gathered = k2(path, xs, i, s)
            shot = {
                "final": b["final"].clone(),
                "reduce": b["reduce"].clone(),
                "grouped": gathered[0].clone(),
                "weights": gathered[2].clone(),
            }
            taken.append(shot)
        torch.cuda.synchronize()
        results[name] = taken
        entry[f"{name}_gave_up"] = int(runner.k2_tail_flags[tk.GAVE_UP_WORD])
    KEEP.append(results)
    split = rows * LATENT
    ref = results["sep"]
    for name in paths:
        if name == "sep":
            continue
        got = results[name]
        bad = {"final": 0, "shared": 0, "normed": 0, "inputs": 0}
        for i in range(ITERS):
            a, t = ref[i], got[i]
            normed = norm_of(layers[i % COPIES].ln_w)(
                a["reduce"][:split].view(rows, LATENT)
            )
            bad["final"] += not torch.equal(bits(a["final"]), bits(t["final"]))
            bad["shared"] += not torch.equal(
                bits(a["reduce"][split:]), bits(t["reduce"][split:])
            )
            bad["normed"] += not torch.equal(
                bits(normed), bits(t["reduce"][:split].view(rows, LATENT))
            )
            bad["inputs"] += not (
                torch.equal(bits(a["grouped"]), bits(t["grouped"]))
                and torch.equal(a["weights"], t["weights"])
            )
        entry[f"{name}_mismatched_launches"] = bad
        entry[f"{name}_bitwise"] = not any(bad.values())
        entry[f"{name}_nan"] = bool(torch.isnan(got[-1]["final"].float()).any())
    entry["moving"] = all(
        not torch.equal(ref[i]["final"], ref[i - 1]["final"]) for i in range(1, ITERS)
    )
    # every rank holds the same reduced rows and final rows
    for name in paths:
        h = int(
            sum(
                bits(r["final"]).long().sum() * (i + 1) + bits(r["reduce"]).long().sum()
                for i, r in enumerate(results[name])
            )
        )
        entry[f"{name}_ranks_agree"] = len(set(gathered_ints(h))) == 1
    print(json.dumps(entry), flush=True)


def graph_capture(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for k in range(min(GN, 4)):
            fn(side, k)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for k in range(GN):
            fn(torch.cuda.current_stream(), k)
    KEEP.append((graph, side, fn))
    barrier()
    graph.replay()
    barrier()
    return graph


def graph_us(graph):
    samples = []
    for _ in range(GREPS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        dist.barrier()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / GN)
    return sorted(samples)[len(samples) // 2]


def bench(rows, local, paths):
    """Each path's graphs, then REPS rounds of their replays in turn."""
    b = local.buffers
    width = rows * (LATENT + HIDDEN)
    sets = [
        (
            (torch.randn(width, generator=gen, device=dev) * 0.5).to(torch.bfloat16),
            bf16(rows * TOPK, LATENT, scale=0.5),
            torch.rand(rows, TOPK, generator=gen, device=dev).float(),
        )
        for _ in range(SETS)
    ]
    finals = [torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)]
    xs = [bf16(rows, HIDDEN) for _ in range(COPIES)]
    KEEP.extend([sets, finals, xs])
    identity = b["fused_rows"]

    def alone(path):
        """The path's tail alone on input set ``k``."""

        def go(s, k):
            data, grouped, weights = sets[k % SETS]
            w = layers[k % COPIES]
            view = dataclasses.replace(
                path,
                buffers={
                    **b,
                    "reduce": data,
                    "routed": data[: rows * LATENT].view(rows, LATENT),
                    "final": finals[0],
                },
            )
            KEEP.append(view)
            runner._replicated_tail(w, view, w.moe, (grouped, identity, weights), s)

        return go

    def layer(path):
        return lambda s, k: k2(path, xs, k, s)

    graphs = {}
    for name, path in paths.items():
        graphs[f"{name}_alone"] = graph_capture(alone(path))
    for name, path in paths.items():
        graphs[f"{name}_k2"] = graph_capture(layer(path))
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, graph in graphs.items():
            times[name].append(graph_us(graph))
    barrier()
    if RANK == 0:
        out = {"rows": rows, "world": WORLD, "herd": local.herd, "gn": GN}
        for name, v in times.items():
            v = sorted(v)
            out[name] = {
                "median": round(v[len(v) // 2], 2),
                "all": [round(x, 2) for x in v],
            }
        print(json.dumps({"bench": out}), flush=True)


def profile(rows, local):
    """Each SPLIT's profile build, GN launches in a graph on rotating inputs,
    each launch's stamps copied out: per launch (this rank's clock, us from
    its first CTA's start) the last routed poll seen, the last flag
    published, every CTA seeing every flag (median and last), the end."""
    b = local.buffers
    width = rows * (LATENT + HIDDEN)
    sets = [
        (
            (torch.randn(width, generator=gen, device=dev) * 0.5).to(torch.bfloat16),
            bf16(rows * TOPK, LATENT, scale=0.5),
            torch.rand(rows, TOPK, generator=gen, device=dev).float(),
        )
        for _ in range(SETS)
    ]
    final = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    KEEP.extend([sets, final])
    out = {"rows": rows, "rank": RANK}
    for split in SPLITS:
        kern = tk.build_k2_tail(
            WORLD,
            rows,
            EPS,
            identity_routes=True,
            split=split,
            poll_bound=BOUND,
            profile=True,
        )
        flags = torch.zeros(tk.tail_flag_words(True), dtype=torch.int32, device=dev)
        snaps = torch.zeros(GN, tk.MAX_CTAS * tk.STAMPS, dtype=torch.int32, device=dev)
        KEEP.extend([kern, flags, snaps])

        def go(s, k, kern=kern, flags=flags, snaps=snaps):
            data, grouped, weights = sets[k % SETS]
            w = layers[k % COPIES]
            kern.launcher(
                data.data_ptr(),
                grouped.data_ptr(),
                b["fused_rows"].data_ptr(),
                weights.data_ptr(),
                w.ln_w.data_ptr(),
                w.w_up_full.data_ptr(),
                final.data_ptr(),
                flags.data_ptr(),
                peers.data_ptr(),
                epochs.data_ptr(),
                RANK,
                stream=s,
            )
            snaps[k % GN].copy_(flags[tk.MAX_CTAS :])

        graph = graph_capture(go)
        graph.replay()
        barrier()
        st = snaps.cpu().long().view(GN, tk.MAX_CTAS, tk.STAMPS)
        ar_ctas = tk.tail_ar_ctas(rows)
        grid = tk.tail_grid(rows)
        parts = {}

        def add(key, ticks, parts=parts):
            parts.setdefault(key, []).append(float(ticks) / 100)

        for k in range(GN // 2, GN):
            s = st[k, :grid]
            s = s - s[:, 0].min()
            add("last_routed_polled", s[: 2 * rows, 6].max())
            add("last_flag", s[:ar_ctas, 1].max())
            add("median_flag", s[:ar_ctas, 1].float().median())
            add("seen_median", s[:, 2].float().median())
            add("seen_last", s[:, 2].max())
            add("end_last", s[:, 4].max())
        out["tail_sw" if split else "tail"] = {
            key: round(sorted(v)[len(v) // 2], 2) for key, v in parts.items()
        }
    print(json.dumps({"profile": out}), flush=True)


for rows in ROWS:
    local, paths = build(rows)
    if CHECK:
        check(rows, local, paths)
    if BENCH:
        bench(rows, local, paths)
    if PROFILE and local.tail is not None:
        profile(rows, local)
barrier()
