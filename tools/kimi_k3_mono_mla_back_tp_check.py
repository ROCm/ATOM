# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 MLA back half (``mla_back_gfx1250``) at TP N = DCP N against
the four launches it replaces, every rank running both on the same inputs:
the gluon ``batched_gemm_a16w8`` W_UV on the DCP group's heads, aiter's DCP
merge (``CustomAllreduce.custom_dcp_merge``: ``dcp_merge_ll_gfx1250``),
AITER's ``fused_sigmoid_mul`` and ``k2_oproj_ar_gfx1250`` (one-shot).

Per width (CHECK): resources, ITERS eager launches then REPLAYS CUDA-graph
replays on new inputs -- the attention output bit for bit on every rank and
equal across ranks, the gated core bit for bit (W_UV, the merge and the gate
without o_proj). LSEs carry -inf / NaN / +inf rows (a shard holding no KV of
a row), one row empty on every rank.

Each eager launch also runs this tree's own merge, ``dcp_ops.cp_lse_a2a``
(the Triton pack and combine; its all-to-all, a byte copy, carried by the
host), on the same W_UV rows and reports whether it is aiter's merge bit for
bit (``triton_merge``): where it is not, the back half may only bind where
production's merge is aiter's kernel.

Then (BENCH) CUDA-graph time, GN launches a replay with the weights and
inputs rotating, a rank barrier before each replay, the arms' replays
alternating REPS times: ``back``, ``production`` (the four launches) and
each production launch alone.

Stages (STAGE), each only once the one before has passed on the node:

* ``merge`` (the default): production only, no mono kernel and no bank of
  this tool's -- W_UV, aiter's merge and the gate through the serving setup.
  Reports whether this tree's Triton merge and the one-GPU tool's host
  emulation of the merge are aiter's kernel bit for bit.
* ``full`` (needs PEER_PUSH=1 as well): the back half's real pushes.

Setup, as serving does it and in its order: AITER_CUSTOM_AR_USE_SYMM_MEM=1
is required (aiter's custom all-reduce then maps its peers through
torch.symm_mem with the MORI backend; unset, it takes the VMM or hipIpc
transport, which serving does not run); the custom all-reduce is built
first (it probes the backend and votes); only ``full`` then maps one bank of
its own (the back half's and the o_proj all-reduce's regions between two
guard bands) with the backend aiter chose, every rank checking it mapped
every peer before any kernel is built. Guard bands are checked after every
launch.

    torchrun --nproc_per_node=4 tools/kimi_k3_mono_mla_back_tp_check.py
    env: ROWS (1,7,8,16,24,32,40,48,64), CHECK (1), BENCH (1), ITERS (2),
    REPLAYS (20), GN (16), REPS (7), COPIES (8), HEADS (96: the group's),
    TRITON (1: also compare this tree's a2a merge with aiter's),
    PROFILE (0; 1: a profile build's per-CTA stamps, each rank's medians),
    STAGE (merge), PEER_PUSH (0),
    BOUND (4194304: polls give up past it; 0 waits forever)

Every kernel, buffer and graph stays referenced in KEEP to the end: a graph
replays only kernels and buffers still alive.
"""

import json
import os
import statistics
from types import SimpleNamespace

import torch
import torch.distributed as dist
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as fp8_bmm,
)
from aiter.ops.triton.fusions.fused_sigmoid_mul import fused_sigmoid_mul

from atom.model_ops.utils import dynamic_per_batched_tensor_quant
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar
from atom.models.kimi_k3.mono.kernels.stages import mla_back_gfx1250 as mb
from atom.models.kimi_k3.mono.kernels.stages import mla_back_layout as lay
from atom.mono.plan.architecture import GFX1250
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

KEEP = []
ROWS = [int(r) for r in os.environ.get("ROWS", "1,7,8,16,24,32,40,48,64").split(",")]
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
ITERS = int(os.environ.get("ITERS", "2"))
REPLAYS = int(os.environ.get("REPLAYS", "20"))
GN = int(os.environ.get("GN", "16"))
REPS = int(os.environ.get("REPS", "7"))
COPIES = int(os.environ.get("COPIES", "8"))
HEADS = int(os.environ.get("HEADS", "96"))
BOUND = int(os.environ.get("BOUND", str(1 << 22))) or None
TRITON = os.environ.get("TRITON", "1") == "1"
PROFILE = os.environ.get("PROFILE", "0") == "1"
STAGE = os.environ.get("STAGE", "merge")
PEER_PUSH = os.environ.get("PEER_PUSH", "0") == "1"
# guard bytes either side of this tool's bank
GUARD = 1 << 16
GUARD_BYTE = 0x5A
PROFILE_LAUNCHES = 16

dist.init_process_group("gloo")
RANK, WORLD = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
dev = torch.device("cuda", torch.cuda.current_device())
HL = HEADS // WORLD
PROJ = HL * lay.V_DIM
gen = torch.Generator(device=dev).manual_seed(101 + RANK)


def barrier():
    torch.cuda.synchronize()
    dist.barrier()


def gathered_ints(value):
    out = [torch.zeros(1, dtype=torch.int64) for _ in range(WORLD)]
    dist.all_gather(out, torch.tensor([value], dtype=torch.int64))
    return [int(v) for v in out]


def bf16(*shape, scale=1.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale).to(torch.bfloat16)


def bits(t):
    return t.contiguous().view(torch.int16)


class Layer:
    """One layer's weights on this rank: W_UV for the group's heads (the
    same on every rank in serving; any here) and this rank's o_proj shard."""

    def __init__(self):
        self.w_uv, s_uv = dynamic_per_batched_tensor_quant(
            bf16(HEADS, lay.V_DIM, lay.KV_LORA, scale=0.05), dtype=dtypes.fp8
        )
        self.s_uv = s_uv.reshape(1).contiguous()
        q, s = quant_weight_online(
            bf16(HIDDEN, PROJ, scale=0.02),
            QuantType.per_1x128,
            dtypes.fp8,
            scale_dtype=dtypes.fp8_e8m0,
        )
        self.w_o = shuffle_weight(q, layout=(16, 16)).contiguous()
        self.s_o = s.contiguous()


LAYERS = [Layer() for _ in range(COPIES if BENCH else 1)]
KEEP.append(LAYERS)


def inputs(rows):
    """This rank's partials over its KV shard, and the layer's gate rows."""
    o = bf16(rows, HEADS, lay.KV_LORA)
    lse = torch.randn(rows, HEADS, generator=gen, device=dev) * 2.0 + 4.0
    drop = torch.rand(rows, HEADS, generator=gen, device=dev)
    lse[drop < 0.08] = float("-inf")
    lse[(drop >= 0.08) & (drop < 0.10)] = float("nan")
    lse[(drop >= 0.10) & (drop < 0.12)] = float("inf")
    if rows > 1:
        lse[1, ::5] = float("-inf")  # heads no shard holds a row of
    gate = bf16(rows, PROJ, scale=2.0)
    return {"o": o, "lse": lse.float().contiguous(), "gate": gate}


def plan_of(rows):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=GFX1250.max_workgroup_lds_bytes,
        tokens=rows,
        nblocks=1,
    )


def agreed(ok) -> bool:
    """Whether ``ok`` holds on every rank (a host collective)."""
    vote = torch.tensor([int(bool(ok))])
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    return bool(vote.item())


def stop(why):
    """Every rank leaves, before anything is mapped or launched."""
    if RANK == 0:
        print(json.dumps({"refused": why}), flush=True)
    dist.barrier()
    raise SystemExit(0)


if STAGE not in ("merge", "full"):
    stop(f"STAGE {STAGE!r}: merge or full")
if os.getenv("AITER_CUSTOM_AR_USE_SYMM_MEM", "0") not in ("1", "true", "True"):
    stop(
        "AITER_CUSTOM_AR_USE_SYMM_MEM=1 is required: serving's custom "
        "all-reduce maps its peers through torch.symm_mem (MORI); unset, it "
        "takes the VMM or hipIpc transport, which serving does not run"
    )
if STAGE == "full" and not PEER_PUSH:
    stop("STAGE=full pushes into peers' memory: set PEER_PUSH=1 once approved")

from aiter.dist.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)

# serving's order: aiter's custom all-reduce first (it probes the MORI
# backend, votes and maps its own buffers), then the mono banks
CA = CustomAllreduce(dist.group.WORLD, dev)
KEEP.append(CA)
if not agreed(not CA.disabled and bool(getattr(CA, "_use_symm_mem", False))):
    stop("aiter's custom all-reduce did not come up on the symm_mem transport")

BACK_PEERS = OPROJ_PEERS = BANK = None
if STAGE == "full":
    import torch.distributed._symmetric_memory as symm_mem

    if not agreed(symm_mem.get_backend(dev) == "MORI"):
        stop("torch.symm_mem is not on the MORI backend aiter chose")
    back_bytes = max(lay.BackPlan(r, WORLD, WORLD, HL).peer_bytes for r in ROWS)
    oproj_bytes = max(oar.oproj_peer_bytes(WORLD, r, True) for r in ROWS)
    BANK = symm_mem.empty(
        GUARD + back_bytes + oproj_bytes + GUARD, dtype=torch.uint8, device=dev
    )
    BANK.fill_(GUARD_BYTE)
    BANK[GUARD:-GUARD].zero_()
    # the fills are stream-ordered, the rendezvous runs on the host
    torch.cuda.synchronize()
    ptrs = list(symm_mem.rendezvous(BANK, dist.group.WORLD).buffer_ptrs)
    if not agreed(len(ptrs) == WORLD and all(ptrs)):
        stop("a rank could not map every peer's bank")
    table = torch.tensor(ptrs, dtype=torch.int64, device=dev)
    BACK_PEERS = table + GUARD
    OPROJ_PEERS = table + GUARD + back_bytes
    KEEP += [BANK, table, BACK_PEERS, OPROJ_PEERS]
BACK_EPOCHS = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
OPROJ_EPOCHS = torch.zeros(ar.MAX_CTAS, dtype=torch.int32, device=dev)
KEEP += [BACK_EPOCHS, OPROJ_EPOCHS]


def guards_ok() -> bool:
    """Whether nothing wrote this rank's guard bands."""
    return BANK is None or bool(
        (BANK[:GUARD] == GUARD_BYTE).all() and (BANK[-GUARD:] == GUARD_BYTE).all()
    )


def merge_reference(parts, lses):
    """``dcp_merge_ll_gfx1250``'s combine of ``parts`` [N, rows, HL, 128]
    BF16 by ``lses`` [N, rows, HL] f32 on the host
    (``kimi_k3_mono_mla_back_check.merge_reference``): the device's exp / log
    through torch, the multiply-adds fused through f64."""
    n = parts.size(0)
    neg_inf = torch.tensor(float("-inf"), device=dev)
    ls = torch.where(torch.isnan(lses) | (lses == float("inf")), neg_inf, lses)
    top = ls.max(0).values
    top = torch.where(top == float("-inf"), torch.zeros_like(top), top)
    total = torch.exp(ls[0] - top)
    for src in range(1, n):
        total = total + torch.exp(ls[src] - top)
    g = torch.log(total) + top
    acc = torch.zeros(parts.shape[1:], dtype=torch.float32, device=dev)
    for src in range(n):
        f = torch.exp(ls[src] - g)
        f = torch.where(torch.isnan(f), torch.zeros_like(f), f)[..., None]
        fused = (f.double() * parts[src].double() + acc.double()).float()
        acc = torch.where(f != 0, fused, acc)
    return acc.to(torch.bfloat16)


def gathered(t):
    """``t`` of every rank, by rank, through the host."""
    mine = (
        t.contiguous()
        .view(torch.int32 if t.dtype == torch.float32 else torch.int16)
        .cpu()
    )
    parts = [torch.empty_like(mine) for _ in range(WORLD)]
    dist.all_gather(parts, mine)
    return torch.stack(parts).to(dev).view(t.dtype).view(WORLD, *t.shape)


def differ(got, ref):
    """How ``got`` misses ``ref`` (BF16), for a report."""
    same = bits(got) == bits(ref)
    d = (got.double() - ref.double()).abs()
    finite = torch.isfinite(d)
    scale = (ref.double().abs()[finite].max().item() or 1.0) if finite.any() else 1.0
    worst = float(d[finite].max()) if finite.any() else 0.0
    return {
        "mismatched": int((~same).sum()),
        "elements": same.numel(),
        "max_abs": worst,
        "max_rel": worst / scale,
        "nonfinite_differ": int(
            (torch.isfinite(got.float()) != torch.isfinite(ref.float())).sum()
        ),
    }


def _host_all_to_all(recv, send, group=None):
    """``all_to_all_single`` of the a2a merge's ``[rank, ...]`` buffers
    through the host: rank n's slice for this rank into ``recv[n]``."""
    mine = send.contiguous().view(torch.int16).cpu()
    parts = [torch.empty_like(mine) for _ in range(WORLD)]
    dist.all_gather(parts, mine)
    for n in range(WORLD):
        recv.view(torch.int16)[n].copy_(parts[n][RANK].to(recv.device))


def triton_merge(y, lse):
    """This tree's a2a merge of ``y`` (``dcp_ops.cp_lse_a2a``: Triton pack,
    all-to-all, Triton combine), the ranks in step."""
    from atom.model_ops import dcp_ops

    group = SimpleNamespace(world_size=WORLD, device_group=None)
    real = torch.distributed.all_to_all_single
    torch.distributed.all_to_all_single = _host_all_to_all
    try:
        return dcp_ops.cp_lse_a2a(y, lse, group)
    finally:
        torch.distributed.all_to_all_single = real


class Production:
    """The four launches, in the serving order."""

    def __init__(self, rows, oproj=True):
        self.rows = rows
        self.oproj = (
            oar.build_oproj_allreduce(plan_of(rows), WORLD, PROJ, oneshot=True)
            if oproj
            else None
        )
        KEEP.append(self)

    def w_uv(self, inp, layer):
        x = inp["o"].reshape(-1, HEADS, lay.KV_LORA).transpose(0, 1)
        out = torch.empty(self.rows, HEADS, lay.V_DIM, dtype=torch.bfloat16, device=dev)
        fp8_bmm(x, layer.w_uv, layer.s_uv, group_size=128, YQ=out, transpose_bm=True)
        return out

    def merge(self, y, inp):
        return CA.custom_dcp_merge(y, inp["lse"])

    def gate(self, merged, inp):
        attn = merged.reshape(self.rows, PROJ)
        out = torch.empty_like(attn)
        return fused_sigmoid_mul(attn, inp["gate"], out=out)

    def o_proj(self, core, layer, stream=None):
        out = core.new_empty(self.rows, HIDDEN)
        self.oproj.launcher(
            core.data_ptr(),
            layer.w_o.data_ptr(),
            layer.s_o.data_ptr(),
            out.data_ptr(),
            OPROJ_PEERS.data_ptr(),
            OPROJ_EPOCHS.data_ptr(),
            RANK,
            stream=stream or torch.cuda.current_stream(),
        )
        KEEP.append(core)
        return out

    def __call__(self, inp, layer, stream=None, taps=None):
        y = self.w_uv(inp, layer)
        merged = self.merge(y, inp)
        if taps is not None:
            taps.update(y=y, merged=merged)
        core = self.gate(merged, inp)
        return core, self.o_proj(core, layer, stream)


class Back:
    def __init__(self, rows):
        self.rows = rows
        self.kernel = mb.build_mla_back(plan_of(rows), WORLD, WORLD, HL, bound=BOUND)
        p = self.kernel.plan
        self.out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.core = torch.empty(rows, PROJ, dtype=torch.bfloat16, device=dev)
        self.flags = torch.zeros(p.flag_words, dtype=torch.int32, device=dev)
        KEEP.append(self)

    def __call__(self, inp, layer, out=None, stream=None):
        p = torch.Tensor.data_ptr
        out = self.out if out is None else out
        args = mb.MLA_BACK_ABI.pack({
            "o": p(inp["o"]), "lse": p(inp["lse"]), "w_uv": p(layer.w_uv),
            "s_uv": p(layer.s_uv), "gate": p(inp["gate"]), "w_o": p(layer.w_o),
            "w_o_scale": p(layer.s_o), "out": p(out), "core": p(self.core),
            "flags": p(self.flags), "peers": p(BACK_PEERS),
            "merge_peers": p(BACK_PEERS), "epochs": p(BACK_EPOCHS),
            "rank": RANK, "dcp_rank": RANK,
        })  # fmt: skip
        self.kernel.launcher(*args, stream=stream or torch.cuda.current_stream())
        return out


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


def one(back, prod, inp, layer):
    """Both paths on ``inp`` (the ranks in step), this rank's verdict."""
    barrier()
    back(inp, layer)
    barrier()
    taps = {}
    core, out = prod(inp, layer, taps=taps)
    barrier()
    got_out, got_core = back.out, back.core
    exact = {
        "out": torch.equal(bits(got_out), bits(out)),
        "core": torch.equal(bits(got_core), bits(core)),
        "guards": guards_ok(),
    }
    if TRITON:
        other = triton_merge(taps["y"], inp["lse"])
        barrier()
        ref = taps["merged"]
        same = bits(other) == bits(ref)
        exact["triton_merge"] = bool(same.all())
        if not exact["triton_merge"]:
            d = (other.double() - ref.double()).abs()
            finite = torch.isfinite(d)
            scale = ref.double().abs()[finite].max().item() or 1.0
            exact["triton_merge_mismatched"] = int((~same).sum())
            exact["triton_merge_elements"] = same.numel()
            exact["triton_merge_max_abs"] = float(d[finite].max())
            exact["triton_merge_max_rel"] = float(d[finite].max()) / scale
            exact["triton_merge_nonfinite_differ"] = int(
                (torch.isfinite(other.float()) != torch.isfinite(ref.float())).sum()
            )
    if not exact["core"]:
        d = (got_core.float() - core.float()).abs()
        exact["core_mismatched"] = int((bits(got_core) != bits(core)).sum())
        finite = d[torch.isfinite(d)]
        exact["core_max_abs"] = float(finite.max()) if finite.numel() else 0.0
    if not exact["out"]:
        exact["out_mismatched"] = int((bits(got_out) != bits(out)).sum())
    return exact, int(bits(got_out).long().sum())


def check(rows):
    back, prod = Back(rows), Production(rows)
    res, hazards = resources(back.kernel)
    if RANK == 0:
        print(json.dumps({"rows": rows, "kernel": back.kernel.name, "resources": res,
                          "hazards": hazards}), flush=True)  # fmt: skip
    layer = LAYERS[0]
    report = {"rank": RANK, "rows": rows, "eager": [], "ok": not hazards}
    last = None
    for _ in range(ITERS):
        inp = inputs(rows)
        exact, total = one(back, prod, inp, layer)
        exact["ranks_agree"] = len(set(gathered_ints(total))) == 1
        exact["moving"] = last is None or not torch.equal(back.out, last)
        last = back.out.clone()
        report["eager"].append(exact)
        report["ok"] = (
            report["ok"]
            and exact["out"]
            and exact["core"]
            and exact["ranks_agree"]
            and exact["guards"]
        )
    inp = inputs(rows)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    barrier()
    with torch.cuda.graph(graph, stream=side):
        back(inp, layer, stream=side)
    torch.cuda.current_stream().wait_stream(side)
    KEEP.append((graph, side, inp))
    bad = 0
    for _ in range(REPLAYS):
        new = inputs(rows)
        for key in inp:
            inp[key].copy_(new[key])
        barrier()
        graph.replay()
        barrier()
        core, out = prod(inp, layer)
        barrier()
        same = torch.equal(bits(back.out), bits(out)) and torch.equal(
            bits(back.core), bits(core)
        )
        agree = len(set(gathered_ints(int(bits(back.out).long().sum())))) == 1
        bad += not (same and agree and guards_ok())
    report["replays"] = REPLAYS
    report["replays_mismatched"] = bad
    report["ok"] = report["ok"] and bad == 0
    print(json.dumps(report), flush=True)
    return back, prod


def graph_capture(fn):
    """``fn(stream, k)`` for k in range(GN) on a side stream, as one graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    barrier()
    with torch.cuda.stream(side):
        for k in range(2):
            fn(side, k)
    torch.cuda.current_stream().wait_stream(side)
    barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for k in range(GN):
            fn(side, k)
    torch.cuda.current_stream().wait_stream(side)
    KEEP.append((graph, side, fn))
    barrier()
    graph.replay()
    barrier()
    return graph


def graph_us(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    barrier()
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def bench(rows, back, prod):
    inps = [inputs(rows) for _ in range(COPIES)]
    outs = [
        torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev) for _ in range(4)
    ]
    for inp in inps:
        # the single launches' inputs: a W_UV output, a merged row, a core
        inp["y"] = bf16(rows, HEADS, lay.V_DIM)
        inp["merged"] = bf16(rows, HL, lay.V_DIM)
        inp["core"] = bf16(rows, PROJ)
    KEEP.extend([inps, outs])

    def pick(k):
        return inps[k % COPIES], LAYERS[k % COPIES]

    def run_back(s, k):
        inp, layer = pick(k)
        back(inp, layer, outs[k % 4], stream=s)

    def run_prod(s, k):
        inp, layer = pick(k)
        KEEP.append(prod(inp, layer, s))

    def run_w_uv(s, k):
        inp, layer = pick(k)
        KEEP.append(prod.w_uv(inp, layer))

    def run_merge(s, k):
        inp, _ = pick(k)
        KEEP.append(prod.merge(inp["y"], inp))

    def run_gate(s, k):
        inp, _ = pick(k)
        KEEP.append(prod.gate(inp["merged"], inp))

    def run_oproj(s, k):
        inp, layer = pick(k)
        KEEP.append(prod.o_proj(inp["core"], layer, s))

    graphs = {
        "back": graph_capture(run_back),
        "production": graph_capture(run_prod),
        "w_uv": graph_capture(run_w_uv),
        "dcp_merge": graph_capture(run_merge),
        "gate": graph_capture(run_gate),
        "oproj_ar": graph_capture(run_oproj),
    }
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, graph in graphs.items():
            times[name].append(graph_us(graph))
    barrier()
    if RANK == 0:
        med = {name: round(statistics.median(t), 2) for name, t in times.items()}
        spread = {name: round(max(t) - min(t), 2) for name, t in times.items()}
        print(json.dumps({"rows": rows, "tp": WORLD, "us_per_layer": med,
                          "spread": spread}), flush=True)  # fmt: skip


def profile(rows):
    """A profile build's stamps over PROFILE_LAUNCHES launches in a graph on
    rotating inputs, the last of three replays: each CTA's phases from its
    own start (a GPU's two dies do not share a clock), their median over the
    CTAs and launches in us, and the launch's span (the last CTA's end)."""
    kernel = mb.build_mla_back(
        plan_of(rows), WORLD, WORLD, HL, bound=BOUND, profile=True
    )
    plan = kernel.plan
    core = torch.empty(rows, PROJ, dtype=torch.bfloat16, device=dev)
    flags = torch.zeros(plan.flag_words, dtype=torch.int32, device=dev)
    out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=dev)
    stamps = torch.zeros(
        PROFILE_LAUNCHES, lay.GRID * mb.STAMPS, dtype=torch.int32, device=dev
    )
    inps = [inputs(rows) for _ in range(COPIES)]
    KEEP.extend([kernel, core, flags, out, stamps, inps])
    p = torch.Tensor.data_ptr

    def go(s, k):
        inp, layer = inps[k % COPIES], LAYERS[k % len(LAYERS)]
        args = mb.MLA_BACK_PROFILE_ABI.pack({
            "o": p(inp["o"]), "lse": p(inp["lse"]), "w_uv": p(layer.w_uv),
            "s_uv": p(layer.s_uv), "gate": p(inp["gate"]), "w_o": p(layer.w_o),
            "w_o_scale": p(layer.s_o), "out": p(out), "core": p(core),
            "flags": p(flags), "peers": p(BACK_PEERS),
            "merge_peers": p(BACK_PEERS), "epochs": p(BACK_EPOCHS),
            "rank": RANK, "dcp_rank": RANK,
            "stamps": p(stamps[k % PROFILE_LAUNCHES]),
        })  # fmt: skip
        kernel.launcher(*args, stream=s)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    barrier()
    with torch.cuda.stream(side):
        go(side, 0)
    torch.cuda.current_stream().wait_stream(side)
    barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        for k in range(PROFILE_LAUNCHES):
            go(side, k)
    torch.cuda.current_stream().wait_stream(side)
    KEEP.append((graph, side))
    for _ in range(3):
        barrier()
        graph.replay()
    barrier()
    s = stamps.cpu().long().view(PROFILE_LAUNCHES, lay.GRID, mb.STAMPS)
    rel = (s - s[:, :, :1]) & 0xFFFFFFFF  # 10 ns ticks from the CTA's start
    names = ["uv_pushed", "lines_seen", "core_stored", "core_seen", "staged",
             "summed", "end"]  # fmt: skip
    report = {"rank": RANK, "rows": rows, "kernel": kernel.name}
    oproj = slice(0, lay.OPROJ_CTAS)
    for i, name in enumerate(names, start=1):
        ctas = oproj if i >= mb.STAMP_CORE_SEEN else slice(None)
        v = rel[:, ctas, i].float()
        v = v[s[:, ctas, i] != 0]
        if v.numel():
            report[name] = {
                "p50": round(float(v.median()) / 100.0, 2),
                "p90": round(float(v.quantile(0.9)) / 100.0, 2),
                "max": round(float(v.max()) / 100.0, 2),
            }
    print(json.dumps({"profile": report}), flush=True)


def merge_stage(rows):
    """Production only: W_UV, aiter's merge and the gate on every rank, and
    against aiter's merged rows this tree's Triton merge and the host
    emulation of the merge the one-GPU tool checks the back half with."""
    prod = Production(rows, oproj=False)
    layer = LAYERS[0]
    report = {"stage": "merge", "rank": RANK, "rows": rows, "launches": []}
    for _ in range(ITERS):
        inp = inputs(rows)
        barrier()
        y = prod.w_uv(inp, layer)
        merged = prod.merge(y, inp)
        barrier()
        core = prod.gate(merged, inp)
        torch.cuda.synchronize()
        entry = {"core_finite": float(torch.isfinite(core.float()).float().mean())}
        mine = slice(RANK * HL, (RANK + 1) * HL)
        parts = gathered(y)[:, :, mine]
        lses = gathered(inp["lse"])[:, :, mine]
        emulated = merge_reference(parts.contiguous(), lses.contiguous())
        entry["emulation_is_aiters"] = torch.equal(bits(emulated), bits(merged))
        if not entry["emulation_is_aiters"]:
            entry["emulation"] = differ(emulated, merged)
        if TRITON:
            other = triton_merge(y, inp["lse"])
            barrier()
            entry["triton_is_aiters"] = torch.equal(bits(other), bits(merged))
            if not entry["triton_is_aiters"]:
                entry["triton"] = differ(other, merged)
        report["launches"].append(entry)
    print(json.dumps(report), flush=True)


for rows in ROWS:
    if STAGE == "merge":
        merge_stage(rows)
        continue
    if PROFILE:
        profile(rows)
        continue
    if CHECK:
        back, prod = check(rows)
    else:
        back, prod = Back(rows), Production(rows)
    if BENCH:
        bench(rows, back, prod)
barrier()
