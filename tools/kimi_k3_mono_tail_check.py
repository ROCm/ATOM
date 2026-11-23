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
    VARIANTS, PVARIANTS, SPY, K2, LEAN (see below),
    AITER_FLYDSL_USE_HERD; OPROJ=1 checks and times the o_proj all-reduce
    instead (OPROJ_PROJ: o_proj's input width over the ranks, 12288)

Every kernel, buffer and graph stays referenced in KEEP to the end: a graph
replays only kernels and buffers still alive.
"""

import ast
import dataclasses
import json
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from aiter import dtypes, rmsnorm2d_fwd
from aiter.ops.shuffle import shuffle_weight

import atom.models.kimi_k3.mono.runner_gfx1250 as rm
from atom.models.kimi_k3.mono.config import EXPERTS, HIDDEN, LATENT, TOPK
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import k2_tail_gfx1250 as tk
from atom.quantization.quark.utils import quantize_weight_to_fp8_128x128_blockscale

KEEP = []
_T0 = time.time()


def note(what):
    """Progress, a line a stage (rank 0): where a run's minutes went."""
    if os.environ.get("RANK", "0") == "0":
        print(json.dumps({"t": round(time.time() - _T0, 1), "at": what}), flush=True)


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
OPROJ = os.environ.get("OPROJ", "0") == "1"
OPROJ_PROJ = int(os.environ.get("OPROJ_PROJ", "12288"))
# VARIANTS: extra tail builds, name:kwarg=value,..+name:.. (build_k2_tail's
# keywords), each checked and timed as "tail_<name>"; PVARIANTS: the names
# also profiled; SPY=1: profile builds spy; K2=0: no whole-K2 graphs
VARIANTS = {}
for _spec in os.environ.get("VARIANTS", "").split("+"):
    if _spec:
        _name, _, _body = _spec.partition(":")
        VARIANTS[_name] = {
            k: ast.literal_eval(v)
            for k, _, v in (item.partition("=") for item in _body.split(",") if item)
        }
PVARIANTS = [v for v in os.environ.get("PVARIANTS", "").split(",") if v]
SPY = os.environ.get("SPY", "0") == "1"
# LEAN=1: no K2 build (minutes of compiles): the tail's builds alone, see lean()
LEAN = os.environ.get("LEAN", "0") == "1"
# LEAN_KIND: tail (lean), oproj (lean_oproj; ONESHOT=0 its two-hop) or gather
# (lean_gather: the gathering all-reduce alone)
LEAN_KIND = os.environ.get("LEAN_KIND", "tail")
ONESHOT = os.environ.get("ONESHOT", "1") == "1"
K2 = os.environ.get("K2", "1") == "1"
COPIES, SETS = 3, 4
SH, INTER, NB, EPS = 768, 384, 4, 1e-5

dist.init_process_group("gloo")
RANK, WORLD = dist.get_rank(), dist.get_world_size()
# EMU_TP=N (one rank, LEAN): the tails built for N ranks on one GPU, every
# peer's pointer into this rank's own bank so that its push to peer p lands
# in the slot source p's line would take: the N-rank kernel's code with
# local stores, each line summed N times over. The bank sits between guards.
EMU_TP = int(os.environ.get("EMU_TP", "0"))
if EMU_TP:
    assert WORLD == 1 and os.environ.get("LEAN") == "1", "EMU_TP: one rank, LEAN"
    WORLD = EMU_TP
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

GUARD = 1 << 20
BANK_BYTES = (
    2
    * max(
        max(
            ar.allreduce_peer_bytes(WORLD, r),
            oar.oproj_peer_bytes(WORLD, r, True),
            oar.oproj_peer_bytes(WORLD, r, False),
        )
        for r in ROWS
    )
    + 4096
)
if EMU_TP:
    bank_all = torch.full(
        (GUARD + BANK_BYTES + GUARD,), 0xA5, dtype=torch.uint8, device=dev
    )
    bank = bank_all[GUARD : GUARD + BANK_BYTES]
    bank.zero_()
    torch.cuda.synchronize()
    ptrs = [bank.data_ptr()] * WORLD
    KEEP.append(bank_all)
else:
    import mori.allocator  # noqa: F401
    import torch.distributed._symmetric_memory as symm_mem

    symm_mem.set_backend("MORI")
    bank = symm_mem.empty(BANK_BYTES, dtype=torch.uint8, device=dev)
    bank.zero_()
    torch.cuda.synchronize()
    ptrs = list(symm_mem.rendezvous(bank, dist.group.WORLD).buffer_ptrs)
    ptrs[RANK] = bank.data_ptr()
peers = torch.tensor(ptrs, dtype=torch.int64, device=dev)


def peers_for(rows, step=None):
    """The peers' bank pointers; emulated, peer p's is this bank moved by p
    sources' lines (``step`` bytes a source; by default the seven-element
    bank's at ``rows``)."""
    if not EMU_TP:
        return peers
    if step is None:
        step = ar.allreduce7_lines(rows) * ar.LINE_WORDS * 4
    assert 2 * WORLD * step <= BANK_BYTES
    table = torch.tensor(
        [bank.data_ptr() + p * step for p in range(WORLD)],
        dtype=torch.int64,
        device=dev,
    )
    KEEP.append(table)
    return table


def guard_ok():
    """An emulated bank's guards untouched (None: a real one)."""
    if not EMU_TP:
        return None
    return bool(
        (bank_all[:GUARD] == 0xA5).all()
        and (bank_all[GUARD + BANK_BYTES :] == 0xA5).all()
    )


# past the counters, a tool build's marks (ar.POLLS_WORD, ar.GAVE_UP_WORD)
epochs = torch.zeros(ar.GAVE_UP_WORD + 16, dtype=torch.int32, device=dev)
KEEP += [bank, peers, epochs]

runner = object.__new__(rm.K3Gfx1250MonoDecodeRunner)
runner.device, runner.tp, runner.rank, runner.blocks = dev, WORLD, RANK, 256
runner.k2_peers = SimpleNamespace(peers=peers, epochs=epochs)
runner.dims = SimpleNamespace(sh_inter=SH, inter=INTER)
runner.k2_tail_flags = torch.zeros(
    tk.tail_flag_words(bounded=BOUND is not None), dtype=torch.int32, device=dev
)
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
            if split and tk.tail_rounds(rows) > 1:
                continue
            tail = tk.build_k2_tail(
                WORLD,
                rows,
                EPS,
                identity_routes=local.fused_moe is not None,
                split=split,
                poll_bound=BOUND,
            )
            paths["tail_sw" if split else "tail"] = dataclasses.replace(
                local, tail=tail
            )
        for name, kw in VARIANTS.items():
            if tk.tail_rounds(rows) > 1:
                continue
            tail = tk.build_k2_tail(
                WORLD,
                rows,
                EPS,
                identity_routes=local.fused_moe is not None,
                poll_bound=BOUND,
                **kw,
            )
            paths[f"tail_{name}"] = dataclasses.replace(local, tail=tail)
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


def k2(local, xs, i, s, pre=None):
    """One K2 layer: front, lite finish, the fused MoE (or past its widths
    AITER's grouped MoE on the HERD routes), then the path's tail. ``pre``:
    gets the all-reduce's input rows (``reduce`` before it)."""
    w = layers[i % COPIES]
    front(local, xs, i, s)
    runner._finish(w, local, local.buffers["router_logits"], s)
    if local.fused_moe is not None:
        gathered = runner._fused_moe(w, local, w.moe, s)
    else:
        gathered = runner._grouped_moe(w, local, w.moe)
    if pre is not None:
        pre.append(local.buffers["reduce"].clone())
    runner._replicated_tail(w, local, w.moe, gathered, s)
    return gathered


def gather_reference(rows, pre, gathered):
    """The gathering all-reduce's output from this launch's inputs: each
    rank's routed lines (its TOPK grouped rows weighted, in route order, in
    fp32, to BF16) and shared rows, summed in rank order in fp32 from rank
    0's, to BF16. Routed lines both with and without a fused multiply-add."""
    grouped, route_rows, weights = gathered
    picks = grouped.reshape(-1, LATENT)[route_rows.long().view(-1)]
    picks = picks.view(rows, TOPK, LATENT).float()
    wt = weights.float().view(rows, TOPK, 1)
    plain = torch.zeros(rows, LATENT, device=dev)
    fused = torch.zeros(rows, LATENT, device=dev, dtype=torch.float64)
    for k in range(TOPK):
        plain = plain + wt[:, k] * picks[:, k]
        # one rounding a step: the fused multiply-add's
        fused = (fused + wt[:, k].double() * picks[:, k].double()).float().double()
    out = {}
    for name, routed in (("plain", plain), ("fma", fused.float())):
        local = torch.cat(
            [routed.to(torch.bfloat16).view(-1), pre[rows * LATENT :]]
        ).cpu()
        every = [torch.empty_like(local) for _ in range(WORLD)]
        dist.all_gather(every, local)
        acc = every[0].float()
        for r in range(1, WORLD):
            acc = acc + every[r].float()
        out[name] = acc.to(torch.bfloat16)
    return out


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
            pre = [] if name == "sep" else None
            gathered = k2(path, xs, i, s, pre)
            shot = {
                "final": b["final"].clone(),
                "reduce": b["reduce"].clone(),
                "grouped": gathered[0].clone(),
                "rows": gathered[1].clone(),
                "weights": gathered[2].clone(),
            }
            if pre:
                shot["pre"] = pre[0]
            taken.append(shot)
        torch.cuda.synchronize()
        results[name] = taken
        entry[f"{name}_gave_up"] = (
            0 if BOUND is None else int(runner.k2_tail_flags[tk.GAVE_UP_WORD])
        )
    KEEP.append(results)
    split = rows * LATENT
    ref = results["sep"]
    # the gathering all-reduce itself, against the rank-order fp32 sum
    if local.allreduce is not None and "gather" in local.allreduce.name:
        bad = {"routed_plain": 0, "routed_fma": 0, "shared": 0}
        for shot in ref:
            want = gather_reference(
                rows, shot["pre"], (shot["grouped"], shot["rows"], shot["weights"])
            )
            got = bits(shot["reduce"].cpu())
            for kind in ("plain", "fma"):
                bad[f"routed_{kind}"] += not torch.equal(
                    got[:split], bits(want[kind][:split])
                )
            bad["shared"] += not torch.equal(got[split:], bits(want["plain"][split:]))
        entry["gather_ar_mismatched_launches"] = bad
        entry["gather_ar_bitwise"] = bad["shared"] == 0 and (
            bad["routed_plain"] == 0 or bad["routed_fma"] == 0
        )
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

            # the grouped MoE's rows may land in another order: compare
            # each route's row
            def routed_rows(x):
                g = x["grouped"].reshape(-1, LATENT)
                return bits(g[x["rows"].long().view(-1)])

            bad["inputs"] += not (
                torch.equal(routed_rows(a), routed_rows(t))
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
    xs = [bf16(rows, HIDDEN) for _ in range(COPIES)]
    # the MoE's own outputs' shapes, dtypes and routes (the grouped MoE's
    # rows are not the fused MoE's)
    real = [t.clone() for t in k2(paths["sep"], xs, 0, torch.cuda.current_stream())]
    torch.cuda.synchronize()
    sets = [
        (
            (torch.randn(width, generator=gen, device=dev) * 0.5).to(torch.bfloat16),
            (torch.randn(real[0].shape, generator=gen, device=dev) * 0.5).to(
                real[0].dtype
            ),
            torch.rand(real[2].shape, generator=gen, device=dev).to(real[2].dtype),
        )
        for _ in range(SETS)
    ]
    finals = [torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)]
    identity = real[1]
    KEEP.extend([sets, finals, xs, real])

    def gather_ar(s, k):
        data, grouped, weights = sets[k % SETS]
        local.allreduce.launcher(
            data.data_ptr(),
            grouped.data_ptr(),
            identity.data_ptr(),
            weights.data_ptr(),
            peers.data_ptr(),
            epochs.data_ptr(),
            RANK,
            stream=s,
        )

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
    if local.allreduce is not None and "gather" in local.allreduce.name:
        graphs["gather_ar_alone"] = graph_capture(gather_ar)
    for name, path in paths.items():
        graphs[f"{name}_alone"] = graph_capture(alone(path))
    for name, path in paths.items():
        if K2:
            graphs[f"{name}_k2"] = graph_capture(layer(path))
    times = {name: [] for name in graphs}
    for rep in range(REPS):
        order = list(graphs.items())
        for name, graph in order if rep % 2 == 0 else reversed(order):
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


POINTS = [
    "start",
    "issued",
    "gathered",
    "pushed",
    "polled",
    "row",
    "rstd",
    "stored",
    "drained",
    "flag",
    "seen",
    "gemv",
    "end",
    "rt0",
    "spy",
    "spy_all",
    "rt1",
    "more",
    "x0",
    "x1",
]
SKIP_POINTS = (tk.P_RT0, tk.P_RT1, tk.P_SPARE0, tk.P_SPARE1)


def _med(values):
    values = sorted(values)
    return round(values[len(values) // 2], 2)


def timeline(st, rows, grid, ar_ctas, helpers=0):
    """``st`` [launches, grid, STAMPS] raw stamps (cycle counters' low words,
    the 100 MHz clock at P_RT0 / P_RT1) -> per class of CTA each point's
    [median, max] over the CTAs, us from the launch's first CTA start,
    medians over the launches. A CTA's points count from its own cycle
    counter; the CTAs are aligned on every CTA seeing the flags at once."""
    full = 1 << 32

    def delta(a, b):
        d = (a - b) % full
        return torch.where(d >= full // 2, d - full, d)

    ticks = delta(st[:, :, tk.P_RT1], st[:, :, tk.P_RT0]).double()
    span = delta(st[:, :, tk.P_END], st[:, :, tk.P_START]).double()
    per_tick = float((span[ticks > 0] / ticks[ticks > 0]).median())
    rel = delta(st, st[:, :, tk.P_SEEN : tk.P_SEEN + 1]).double() / (100 * per_tick)
    t = (rel - rel[:, :, tk.P_START].min(1).values[:, None, None]).float()
    extra = 2 * rows * helpers
    classes = {
        "routed": list(range(2 * rows)),
        "shared": list(range(2 * rows, ar_ctas)),
        "helper": list(range(ar_ctas, ar_ctas + extra)),
        "idle": list(range(ar_ctas + extra, grid)),
    }
    out = {"mhz": round(per_tick * 100, 1)}
    for name, ids in classes.items():
        if not ids:
            continue
        part = {}
        for i, point in enumerate(POINTS):
            if i in SKIP_POINTS or not bool((st[:, ids, i] != 0).all()):
                continue
            x = t[:, ids, i]
            part[point] = [
                _med(x.median(1).values.tolist()),
                _med(x.max(1).values.tolist()),
            ]
        out[name] = part
    flag = t[:, :ar_ctas, tk.P_FLAG]
    last = flag.argmax(1)
    out["last_flag"] = _med(flag.max(1).values.tolist())
    out["last_flag_routed_share"] = round(float((last < 2 * rows).float().mean()), 2)
    out["last_flag_minus_median_flag"] = _med(
        (flag.max(1).values - flag.median(1).values).tolist()
    )
    shared = classes["shared"]
    if shared:
        out["last_shared_flag"] = _med(t[:, shared, tk.P_FLAG].max(1).values.tolist())
    crit = t[torch.arange(st.shape[0]), last]
    raw = st[torch.arange(st.shape[0]), last]
    out["last_cta"] = {
        point: _med(crit[:, i].tolist())
        for i, point in enumerate(POINTS)
        if i not in SKIP_POINTS and bool((raw[:, i] != 0).all())
    }
    seen = t[:, :, tk.P_SEEN]
    out["seen"] = _med(seen.median(1).values.tolist())
    # thread 0's poll loads until its lines carried the tag: [median, max]
    out["polls"] = {
        name: [
            _med(st[:, ids, tk.P_SPARE0].float().median(1).values.tolist()),
            _med(st[:, ids, tk.P_SPARE0].float().max(1).values.tolist()),
        ]
        for name, ids in classes.items()
        if ids and name in ("routed", "shared")
    }
    idle = classes["idle"]
    if idle and bool((st[:, idle, tk.P_SPY] != 0).all()):
        target = [b % (2 * rows) for b in idle]
        out["spy_first_thread_before_owner"] = _med(
            (t[:, target, tk.P_POLLED] - t[:, idle, tk.P_SPY]).flatten().tolist()
        )
        out["spy_all_threads_before_owner_row"] = _med(
            (t[:, target, tk.P_ROW] - t[:, idle, tk.P_SPY_ALL]).flatten().tolist()
        )
    return out


def lean(rows):
    """No K2 build: the tail and each of VARIANTS' builds alone on random
    inputs (identity routes), every rank launching them in the same order.
    Each variant's final rows and reduced / normed rows against the tail's
    bit for bit: ITERS eager launches, then a graph of GN launches of each
    replayed 20 times. Then their graphs' times, the arms in turn and back
    (BENCH), and the profiles (PROFILE)."""
    pr = peers_for(rows)
    width = rows * (LATENT + HIDDEN)
    identity = torch.arange(rows * TOPK, dtype=torch.int32, device=dev).view(rows, TOPK)
    sets = [
        (
            (torch.randn(width, generator=gen, device=dev) * 0.5).to(torch.bfloat16),
            bf16(rows * TOPK, LATENT, scale=0.5),
            torch.rand(rows, TOPK, generator=gen, device=dev).float(),
        )
        for _ in range(SETS)
    ]
    kerns = {
        name: tk.build_k2_tail(
            WORLD, rows, EPS, identity_routes=True, poll_bound=BOUND, **kw
        )
        for name, kw in [("tail", {})]
        + [(f"tail_{n}", kw) for n, kw in VARIANTS.items()]
    }
    # a build that pushes nothing (bounded) is launched once, apart
    never = {n: k for n, k in kerns.items() if "_np" in k.name}
    builds = dict(kerns)
    kerns = {n: k for n, k in kerns.items() if n not in never}
    flags = torch.zeros(
        tk.tail_flag_words(bounded=BOUND is not None), dtype=torch.int32, device=dev
    )
    work = torch.empty(width, dtype=torch.bfloat16, device=dev)
    final = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    KEEP.extend([identity, sets, kerns, flags, work, final])
    note(f"lean {rows}: built")

    def launch(name, k, s, data):
        _, grouped, weights = sets[k % SETS]
        w = layers[k % COPIES]
        builds[name].launcher(
            data.data_ptr(),
            grouped.data_ptr(),
            identity.data_ptr(),
            weights.data_ptr(),
            w.ln_w.data_ptr(),
            w.w_up_full.data_ptr(),
            final.data_ptr(),
            flags.data_ptr(),
            pr.data_ptr(),
            epochs.data_ptr(),
            RANK,
            stream=s,
        )

    entry = {
        "rank": RANK,
        "rows": rows,
        "names": {k: v.name for k, v in kerns.items()},
    }
    if os.environ.get("DIAG") == "1":
        # a bounded build's launches one at a time: its give-up mark, and how
        # many of each source rank's lines in this rank's bank carry the
        # launch's tag (read on the host)
        lines = ar.allreduce7_lines(rows)
        s = torch.cuda.current_stream()
        for i in range(ITERS):
            barrier()
            start = time.time()
            work.copy_(sets[i % SETS][0])
            launch("tail", i, s, work)
            torch.cuda.synchronize()
            took = time.time() - start
            epoch = int(epochs[0])
            words = bank[: 2 * WORLD * lines * 16].view(torch.int32)
            tags = (words.view(2, WORLD, lines, 4)[epoch & 1, :, :, 3] >> 16) & 0xFFFF
            diag = {
                "rank": RANK,
                "launch": i,
                "s": round(took, 3),
                "epoch": epoch,
                "gave_up": hex(int(flags[tk.GAVE_UP_WORD])) if BOUND else None,
                "landed_of": lines,
                "landed": (tags == (epoch & 0xFFFF)).sum(1).tolist(),
                "flags_set": int((flags[: tk.tail_ar_ctas(rows)] == epoch).sum()),
            }
            print(json.dumps({"diag": diag}), flush=True)
        return
    if CHECK:
        s = torch.cuda.current_stream()
        for name in kerns:
            work.copy_(sets[0][0])
            launch(name, 0, s, work)
        barrier()
        note(f"lean {rows}: every build launched once")
        # a bounded build's give-ups while a peer still compiled do not count
        flags.zero_()
        barrier()
        bad = {name: 0 for name in kerns if name != "tail"}
        moving, last = 0, None
        for i in range(ITERS):
            got = {}
            for name in kerns:
                work.copy_(sets[i % SETS][0])
                launch(name, i, s, work)
                got[name] = (final.clone(), work.clone())
            torch.cuda.synchronize()
            for name in bad:
                bad[name] += not all(
                    torch.equal(bits(a), bits(b))
                    for a, b in zip(got["tail"], got[name])
                )
            moving += last is not None and not torch.equal(got["tail"][0], last)
            last = got["tail"][0]
        entry["eager_mismatched_launches"] = bad
        entry["moving"] = moving == ITERS - 1
        snaps = {
            name: (
                torch.empty(GN, rows, HIDDEN, dtype=torch.bfloat16, device=dev),
                torch.empty(GN, width, dtype=torch.bfloat16, device=dev),
            )
            for name in kerns
        }
        KEEP.append(snaps)

        def every(s, k):
            for name in kerns:
                work.copy_(sets[k % SETS][0])
                launch(name, k, s, work)
                snaps[name][0][k].copy_(final)
                snaps[name][1][k].copy_(work)

        graph = graph_capture(every)
        gbad = {name: 0 for name in bad}
        for _ in range(20):
            barrier()
            graph.replay()
            torch.cuda.synchronize()
            for name in gbad:
                gbad[name] += not all(
                    torch.equal(bits(a), bits(b))
                    for a, b in zip(snaps["tail"], snaps[name])
                )
        entry["graph_mismatched_replays"] = gbad
        entry["bitwise"] = not any(bad.values()) and not any(gbad.values())
        h = int(
            bits(snaps["tail"][0]).long().sum() + bits(snaps["tail"][1]).long().sum()
        )
        entry["ranks_agree"] = EMU_TP > 0 or len(set(gathered_ints(h))) == 1
        entry["guard_ok"] = guard_ok()
        entry["nan"] = bool(torch.isnan(snaps["tail"][0].float()).any())
        if BOUND is not None:
            entry["gave_up"] = int(flags[tk.GAVE_UP_WORD])
            entry["max_polls"] = int(flags[tk.GAVE_UP_WORD + 1 :].max())
        for name in never:
            flags.zero_()
            work.copy_(sets[0][0])
            barrier()
            start = time.time()
            launch(name, 0, s, work)
            torch.cuda.synchronize()
            entry.setdefault("never", {})[name] = {
                "s": round(time.time() - start, 3),
                "gave_up": hex(int(flags[tk.GAVE_UP_WORD])),
                "max_polls": int(flags[tk.GAVE_UP_WORD + 1 :].max()),
            }
    print(json.dumps({"lean": entry}), flush=True)
    if BENCH:
        graphs = {}
        for name in kerns:
            graphs[name] = graph_capture(
                lambda s, k, name=name: launch(name, k, s, sets[k % SETS][0])
            )
            note(f"lean {rows}: {name} captured")
        times = {name: [] for name in graphs}
        for rep in range(REPS):
            order = list(graphs.items())
            for name, graph in order if rep % 2 == 0 else reversed(order):
                times[name].append(graph_us(graph))
        barrier()
        if RANK == 0:
            out = {"rows": rows, "world": WORLD, "gn": GN, "guard_ok": guard_ok()}
            for name, v in times.items():
                v = sorted(v)
                out[name] = {
                    "median": round(v[len(v) // 2], 2),
                    "all": [round(x, 2) for x in v],
                }
            print(json.dumps({"lean_bench": out}), flush=True)
    if PROFILE:
        profile(rows, identity)


def lean_run(
    kind, rows, kerns, launch, fresh, outputs, pre=None, graph=True, stale=None
):
    """``kerns`` (name -> kernel, the first the reference) alone on one GPU:
    each of the others' ``outputs()`` against the reference's bit for bit --
    ITERS eager launches (``fresh(i)`` restores an in-place input, ``pre(i)``
    places what the host emulates), then with ``graph`` a graph of GN
    launches of each replayed 20 times -- a bounded build's marks, a build
    that pushes nothing launched once, and the graphs' times, the arms in
    turn and back."""
    never = {n: k for n, k in kerns.items() if "_np" in k.name}
    kerns = {n: k for n, k in kerns.items() if n not in never}
    ref = next(iter(kerns))
    s = torch.cuda.current_stream()
    entry = {
        "kind": kind,
        "rows": rows,
        "tp": WORLD,
        "names": {k: v.name for k, v in {**kerns, **never}.items()},
    }
    marks = epochs[ar.POLLS_WORD :]

    def go(name, i):
        if pre is not None:
            pre(i)
        fresh(i)
        launch(name, i, s)

    if CHECK:
        for name in kerns:
            go(name, 0)
        barrier()
        marks.zero_()
        bad = {name: 0 for name in kerns if name != ref}
        polls = {name: 0 for name in kerns}
        moving, last = 0, None
        for i in range(ITERS):
            got = {}
            for name in kerns:
                go(name, i)
                got[name] = [t.clone() for t in outputs()]
                if BOUND is not None:
                    torch.cuda.synchronize()
                    most = int(marks[: ar.MAX_CTAS].max())
                    polls[name] = max(polls[name], most)
                    marks[: ar.MAX_CTAS].zero_()
            torch.cuda.synchronize()
            for name in bad:
                bad[name] += not all(
                    torch.equal(bits(a), bits(b)) for a, b in zip(got[ref], got[name])
                )
            moving += last is not None and not torch.equal(got[ref][0], last)
            last = got[ref][0]
        entry["eager_mismatched_launches"] = bad
        entry["moving"] = moving == ITERS - 1
        gbad = {name: 0 for name in bad}
        if graph:
            snaps = {
                name: [
                    torch.empty(GN, *t.shape, dtype=t.dtype, device=dev)
                    for t in outputs()
                ]
                for name in kerns
            }
            KEEP.append(snaps)

            def every(s_, k):
                for name in kerns:
                    fresh(k)
                    launch(name, k, s_)
                    for snap, t in zip(snaps[name], outputs()):
                        snap[k].copy_(t)

            g = graph_capture(every)
            for _ in range(20):
                barrier()
                g.replay()
                torch.cuda.synchronize()
                for name in gbad:
                    gbad[name] += not all(
                        torch.equal(bits(a), bits(b))
                        for a, b in zip(snaps[ref], snaps[name])
                    )
            entry["graph_mismatched_replays"] = gbad
            entry["nan"] = bool(torch.isnan(snaps[ref][0].float()).any())
        entry["bitwise"] = not any(bad.values()) and not any(gbad.values())
        entry["guard_ok"] = guard_ok()
        if BOUND is not None:
            entry["gave_up"] = hex(int(epochs[ar.GAVE_UP_WORD]))
            entry["max_polls"] = polls
        for name in never:
            marks.zero_()
            barrier()
            start = time.time()
            go(name, 0)
            torch.cuda.synchronize()
            entry.setdefault("never", {})[name] = {
                "s": round(time.time() - start, 3),
                "gave_up": hex(int(epochs[ar.GAVE_UP_WORD])),
                "max_polls": int(marks[: ar.MAX_CTAS].max()),
                "guard_ok": guard_ok(),
            }
        if stale is not None and BOUND is not None:
            # ``stale(i)``: the host's lines of a launch carry an old epoch
            # (lines that never arrive): every build gives up
            for name in kerns:
                marks.zero_()
                barrier()
                start = time.time()
                stale(0)
                fresh(0)
                launch(name, 0, s)
                torch.cuda.synchronize()
                entry.setdefault("stale", {})[name] = {
                    "s": round(time.time() - start, 3),
                    "gave_up": hex(int(epochs[ar.GAVE_UP_WORD])),
                    "guard_ok": guard_ok(),
                }
    print(json.dumps({"lean": entry}), flush=True)
    if BENCH and graph:
        graphs = {}
        for name in kerns:
            graphs[name] = graph_capture(lambda s_, k, name=name: launch(name, k, s_))
        times = {name: [] for name in graphs}
        for rep in range(REPS):
            order = list(graphs.items())
            for name, g in order if rep % 2 == 0 else reversed(order):
                times[name].append(graph_us(g))
        barrier()
        out = {
            "kind": kind,
            "rows": rows,
            "world": WORLD,
            "gn": GN,
            "guard_ok": guard_ok(),
        }
        for name, v in times.items():
            v = sorted(v)
            out[name] = {
                "median": round(v[len(v) // 2], 2),
                "all": [round(x, 2) for x in v],
            }
        print(json.dumps({"lean_bench": out}), flush=True)


def lean_oproj(rows):
    """The o_proj all-reduce alone (ONESHOT=0: two-hop) and each of VARIANTS'
    builds of it. Emulated ranks: one-shot, peer p's pointer is this bank
    moved by p sources' lines; two-hop, the other ranks' pushes go to a dump
    region of the guarded buffer and the host places, before each eager
    launch, the other ranks' partials of this rank's columns and their sums
    of their own (no graphs, no times)."""
    proj = OPROJ_PROJ // WORLD
    plan = runner._plan(rows, 1)
    kerns = {
        name: oar.build_oproj_allreduce(
            plan, WORLD, proj, oneshot=ONESHOT, poll_bound=BOUND, **kw
        )
        for name, kw in [("oproj", {})]
        + [(f"oproj_{n}", kw) for n, kw in VARIANTS.items()]
    }
    w, ws = fp8_preshuffled(HIDDEN, proj)
    cores = [bf16(rows, proj) for _ in range(SETS)]
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    count, own, half = oar.triples(rows), oar.own_pairs(WORLD), HIDDEN // 2
    pre = None
    stale = None
    if not EMU_TP:
        pr = peers
    elif ONESHOT:
        pr = peers_for(rows, count * half * oar.LINE_WORDS * 4)
    else:
        words = oar.bank_words(WORLD, rows)
        assert 2 * words * 4 <= BANK_BYTES // 2
        pr = torch.tensor(
            [bank.data_ptr()] + [bank.data_ptr() + BANK_BYTES // 2] * (WORLD - 1),
            dtype=torch.int64,
            device=dev,
        )
        foreign = [
            (
                bf16(WORLD - 1, count, own, 3, 2).view(torch.int32).squeeze(-1),
                bf16(count, half - own, 3, 2).view(torch.int32).squeeze(-1),
            )
            for _ in range(SETS)
        ]
        KEEP.append(foreign)

        def pre(i):
            torch.cuda.synchronize()
            epoch = int(epochs[0]) + 1
            v = bank[: 2 * words * 4].view(torch.int32).view(2, words)[epoch & 1]
            partials = v[: 4 * WORLD * count * own].view(WORLD, count, own, 4)
            partials[1:, :, :, :3] = foreign[i % SETS][0]
            partials[1:, :, :, 3] = epoch
            sums = v[4 * WORLD * count * own :].view(count, half, 4)
            sums[:, own:, :3] = foreign[i % SETS][1]
            sums[:, own:, 3] = epoch

        def stale(i):
            # the other ranks' partials as a launch two before left them
            pre(i)
            epoch = int(epochs[0]) + 1
            v = bank[: 2 * words * 4].view(torch.int32).view(2, words)[epoch & 1]
            partials = v[: 4 * WORLD * count * own].view(WORLD, count, own, 4)
            partials[1:, :, :, 3] = epoch - 2
            torch.cuda.synchronize()

    KEEP.extend([kerns, w, ws, cores, out, pr])

    def launch(name, k, s):
        kerns[name].launcher(
            cores[k % SETS].data_ptr(),
            w.data_ptr(),
            ws.data_ptr(),
            out.data_ptr(),
            pr.data_ptr(),
            epochs.data_ptr(),
            RANK,
            stream=s,
        )

    lean_run(
        "oproj" if ONESHOT else "oproj_twohop",
        rows,
        kerns,
        launch,
        lambda i: None,
        lambda: [out],
        pre,
        graph=ONESHOT or not EMU_TP,
        stale=stale,
    )


def lean_gather(rows):
    """The gathering all-reduce alone (seven-element lines to 42 rows, six
    past) and each of VARIANTS' builds of it; emulated ranks as the tail's."""
    width = rows * (LATENT + HIDDEN)
    seven = ar.allreduce7_lines(rows) <= ar.MAX_CTAS * ar.THREADS
    lines = ar.allreduce7_lines(rows) if seven else ar.allreduce_lines(rows)
    pr = peers_for(rows, lines * ar.LINE_WORDS * 4)
    routes = torch.arange(rows * TOPK, dtype=torch.int32, device=dev).view(rows, TOPK)
    sets = [
        (
            (torch.randn(width, generator=gen, device=dev) * 0.5).to(torch.bfloat16),
            bf16(rows * TOPK, LATENT, scale=0.5),
            torch.rand(rows, TOPK, generator=gen, device=dev).float(),
        )
        for _ in range(SETS)
    ]
    kerns = {
        name: ar.build_gather_allreduce(WORLD, rows, poll_bound=BOUND, **kw)
        for name, kw in [("gather", {})]
        + [(f"gather_{n}", kw) for n, kw in VARIANTS.items()]
    }
    work = torch.empty(width, dtype=torch.bfloat16, device=dev)
    KEEP.extend([kerns, sets, routes, work, pr])

    def launch(name, k, s):
        _, grouped, weights = sets[k % SETS]
        kerns[name].launcher(
            work.data_ptr(),
            grouped.data_ptr(),
            routes.data_ptr(),
            weights.data_ptr(),
            pr.data_ptr(),
            epochs.data_ptr(),
            RANK,
            stream=s,
        )

    lean_run(
        "gather7" if seven else "gather6",
        rows,
        kerns,
        launch,
        lambda i: work.copy_(sets[i % SETS][0]),
        lambda: [work],
    )


def profile(rows, identity):
    """The tail's profile build and each of PVARIANTS', GN launches in a
    graph on rotating inputs, each launch's stamps copied out: ``timeline``
    of the graph's second half, and the graph's time a launch."""
    pr = peers_for(rows)
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
    words = tk.MAX_CTAS * tk.STAMPS
    for name, kw in [("tail", {})] + [(n, VARIANTS[n]) for n in PVARIANTS]:
        kern = tk.build_k2_tail(
            WORLD,
            rows,
            EPS,
            identity_routes=True,
            poll_bound=BOUND,
            profile=True,
            spy=SPY,
            **kw,
        )
        flags = torch.zeros(
            tk.tail_flag_words(True, bounded=BOUND is not None),
            dtype=torch.int32,
            device=dev,
        )
        snaps = torch.zeros(GN, words, dtype=torch.int32, device=dev)
        KEEP.extend([kern, flags, snaps])

        def go(s, k, kern=kern, flags=flags, snaps=snaps):
            data, grouped, weights = sets[k % SETS]
            w = layers[k % COPIES]
            kern.launcher(
                data.data_ptr(),
                grouped.data_ptr(),
                identity.data_ptr(),
                weights.data_ptr(),
                w.ln_w.data_ptr(),
                w.w_up_full.data_ptr(),
                final.data_ptr(),
                flags.data_ptr(),
                pr.data_ptr(),
                epochs.data_ptr(),
                RANK,
                stream=s,
            )
            snaps[k % GN].copy_(flags[tk.MAX_CTAS : tk.MAX_CTAS + words])

        graph = graph_capture(go)
        note(f"profile {rows}: {name} captured")
        us = [graph_us(graph) for _ in range(3)]
        barrier()
        graph.replay()
        barrier()
        grid = tk.tail_grid(rows)
        st = snaps.cpu().long().view(GN, tk.MAX_CTAS, tk.STAMPS)[GN // 2 :, :grid]
        entry = timeline(st, rows, grid, tk.tail_ar_ctas(rows), kw.get("helpers", 0))
        entry["name"] = kern.name
        entry["graph_us"] = _med(us)
        out[name] = entry
    print(json.dumps({"profile": out}), flush=True)


def oproj(rows):
    """The o_proj all-reduce, one hop against two (the owner's rank-order
    sum), bit for bit on ITERS random inputs, and on sparse small-integer
    inputs against the exact rank-order sum; then both in graphs."""
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    proj = OPROJ_PROJ // WORLD
    kerns = {
        hop: oar.build_oproj_allreduce(runner._plan(rows, 1), WORLD, proj, oneshot=hop)
        for hop in (True, False)
    }
    w, ws = fp8_preshuffled(HIDDEN, proj)
    # small integers, exact through the FP8 E8M0 quantization
    wi = torch.randint(-2, 3, (HIDDEN, proj), generator=gen, device=dev).to(
        torch.bfloat16
    )
    qi, si = quantize_weight_to_fp8_128x128_blockscale(
        wi, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
    )
    wiq = shuffle_weight(qi, layout=(16, 16)).contiguous()
    cores = [bf16(rows, proj) for _ in range(ITERS)]
    sparse = torch.zeros(rows, proj, device=dev)
    cols = torch.randint(0, proj, (rows, 4), generator=gen, device=dev)
    vals = torch.randint(-2, 3, (rows, 4), generator=gen, device=dev).float()
    sparse.scatter_(1, cols, vals)
    sparse = sparse.to(torch.bfloat16)
    outs = {
        hop: torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        for hop in kerns
    }
    KEEP.extend([kerns, w, ws, wi, wiq, si, cores, sparse, outs])

    def launch(hop, core, weight, scale, s):
        kerns[hop].launcher(
            core.data_ptr(),
            weight.data_ptr(),
            scale.data_ptr(),
            outs[hop].data_ptr(),
            opeers.data_ptr(),
            oepochs.data_ptr(),
            RANK,
            stream=s,
        )

    entry = {"rank": RANK, "rows": rows, "names": [k.name for k in kerns.values()]}
    s = torch.cuda.current_stream()
    for hop in kerns:
        launch(hop, cores[0], w, ws, s)
    barrier()
    bad, moving, h = 0, 0, 0
    last = None
    for i in range(ITERS):
        taken = {}
        for hop in kerns:
            launch(hop, cores[i], w, ws, s)
            taken[hop] = outs[hop].clone()
        torch.cuda.synchronize()
        bad += not torch.equal(bits(taken[True]), bits(taken[False]))
        moving += last is not None and not torch.equal(taken[True], last)
        last = taken[True]
        h += int(bits(taken[True]).long().sum()) * (i + 1)
    entry["oneshot_vs_twohop_mismatched"] = bad
    entry["moving"] = moving == ITERS - 1
    entry["ranks_agree"] = len(set(gathered_ints(h))) == 1
    partial = (sparse.double() @ wi.double().t()).float().cpu()
    every = [torch.empty_like(partial) for _ in range(WORLD)]
    dist.all_gather(every, partial)
    acc = every[0]
    for r in range(1, WORLD):
        acc = acc + every[r]
    want = acc.to(torch.bfloat16)
    exact = {}
    for hop in kerns:
        launch(hop, sparse, wiq, si, s)
        torch.cuda.synchronize()
        exact[hop] = torch.equal(bits(outs[hop].cpu()), bits(want))
    entry["exact_oneshot"], entry["exact_twohop"] = exact[True], exact[False]
    entry["oproj_bitwise"] = bad == 0 and all(exact.values())
    print(json.dumps({"oproj": entry}), flush=True)
    if BENCH:
        graphs = {
            ("oproj_oneshot" if hop else "oproj_twohop"): graph_capture(
                lambda s, k, hop=hop: launch(hop, cores[k % ITERS], w, ws, s)
            )
            for hop in kerns
        }
        times = {name: [] for name in graphs}
        for _ in range(REPS):
            for name, graph in graphs.items():
                times[name].append(graph_us(graph))
        barrier()
        if RANK == 0:
            out = {"rows": rows}
            for name, v in times.items():
                v = sorted(v)
                out[name] = {"median": round(v[len(v) // 2], 2), "all": v}
            print(json.dumps({"oproj_bench": out}), flush=True)


if OPROJ:
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as _oar

    obank = symm_mem.empty(
        max(_oar.oproj_peer_bytes(WORLD, r, True) for r in ROWS)
        + max(_oar.oproj_peer_bytes(WORLD, r, False) for r in ROWS)
        + 4096,
        dtype=torch.uint8,
        device=dev,
    )
    obank.zero_()
    torch.cuda.synchronize()
    optrs = list(symm_mem.rendezvous(obank, dist.group.WORLD).buffer_ptrs)
    optrs[RANK] = obank.data_ptr()
    opeers = torch.tensor(optrs, dtype=torch.int64, device=dev)
    oepochs = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
    KEEP += [obank, opeers, oepochs]
    for rows in ROWS:
        oproj(rows)
    barrier()
    raise SystemExit(0)

if LEAN:
    for rows in ROWS:
        {"tail": lean, "oproj": lean_oproj, "gather": lean_gather}[LEAN_KIND](rows)
    barrier()
    raise SystemExit(0)

for rows in ROWS:
    local, paths = build(rows)
    if local.fused_moe is None and local.grouped_moe is None:
        # AITER's fused_moe and the plain all-reduce: no gathering, no tail
        allreduce = None if local.allreduce is None else local.allreduce.name
        skip = {"rank": RANK, "rows": rows, "herd": local.herd, "allreduce": allreduce}
        print(json.dumps({"skipped": skip}), flush=True)
        continue
    if CHECK:
        check(rows, local, paths)
    if BENCH:
        bench(rows, local, paths)
    if PROFILE and local.tail is not None:
        profile(rows, local.buffers["fused_rows"])
barrier()
