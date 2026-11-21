# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 DSpark draft gate_up kernel (``draft_gate_up_gfx1250``:
gate_up_proj and SiLU * mul in one launch) on one GPU against what
``K3DSparkMLP.forward`` runs, fed a rank's TP shard and the post-attention
norm's own output (``add_rmsnorm_quant``: E4M3 row, transposed per-1x128
E8M0 codes): ``gemm_a8w8_blockscale_bpreshuffle`` (the FlyDSL mxfp8_128 GEMM
at the width's tuned split-K, and its reduce) and ``silu_and_mul``.

Per width (CHECK): resources; the activation and the GEMM's BF16 rows (a
tap) bit for bit over eager launches at several input scales (the last ones
drive the gate past the SiLU's exp range fixups) and REPLAYS replays of a
captured launch on fresh inputs. Then (BENCH) CUDA-graph time a launch
against production's GEMM + reduce + act, inputs and weights rotating over
COPIES sets, the graphs' replays alternating.

    python tools/kimi_k3_mono_draft_gate_up_check.py
    env: ROWS (1,7,14,21,28,35,42,48: the widths production splits K in
    4 or 8), SHARD_TP (8), CHECK, BENCH,
    SCALES (1,1,12,60), REPLAYS (20), GN (32), REPS (9), COPIES (16),
    SPLIT_K (production's)
"""

import json
import os
import statistics

import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle, silu_and_mul
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.layernorm import _aiter_rms_quant
from atom.models.kimi_k3.mono import draft_dispatch
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import draft_gate_up_gfx1250 as gk
from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

KEEP = []
ROWS = [int(r) for r in os.environ.get("ROWS", "1,7,14,21,28,35,42,48").split(",")]
SHARD_TP = int(os.environ.get("SHARD_TP", "8"))
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
SCALES = [float(s) for s in os.environ.get("SCALES", "1,1,12,60").split(",")]
REPLAYS = int(os.environ.get("REPLAYS", "20"))
GN = int(os.environ.get("GN", "32"))
REPS = int(os.environ.get("REPS", "9"))
COPIES = int(os.environ.get("COPIES", "16"))
SPLIT_K = os.environ.get("SPLIT_K")
INTER = 14336 // SHARD_TP
EPS = 1e-6
PER_1X128 = QuantType.per_1x128.value

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(3)


def bf16(*shape, scale=1.0, shift=0.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale + shift).to(
        torch.bfloat16
    )


def fp8_linear(rows, k, scale=0.02):
    q, s = quant_weight_online(
        bf16(rows, k, scale=scale),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


SHARDS = [fp8_linear(2 * INTER, HIDDEN) for _ in range(COPIES if BENCH else 1)]
NORM = bf16(HIDDEN, scale=0.1, shift=1.0)
PARTIALS = torch.empty(lay.gate_up_partial_bytes(INTER), dtype=torch.uint8, device=dev)
KEEP += [SHARDS, NORM, PARTIALS]


def inputs(rows, scale=1.0):
    """The post-attention norm's output for random rows: (E4M3, codes)."""
    x8, xs, _ = _aiter_rms_quant(
        bf16(rows, HIDDEN, scale=0.5), (NORM.float() * scale).to(torch.bfloat16),
        EPS, PER_1X128, True, bf16(rows, HIDDEN), dtypes.fp8, True,
    )  # fmt: skip
    return x8.contiguous(), xs


def production(x8, xs, shard):
    gu = gemm_a8w8_blockscale_bpreshuffle(x8, shard[0], xs, shard[1], torch.bfloat16)
    act = torch.empty(gu.size(0), INTER, dtype=gu.dtype, device=dev)
    silu_and_mul(act, gu)
    return gu, act


def split_k(rows):
    if SPLIT_K:
        return int(SPLIT_K)
    return draft_dispatch.mxfp8_split_k(rows, 2 * INTER, HIDDEN, "gate_up_proj")


class Mono:
    def __init__(self, rows):
        self.kernel = gk.build_draft_gate_up(rows, INTER, split_k(rows))
        self.act = torch.empty(rows, INTER, dtype=torch.bfloat16, device=dev)
        self.gu = torch.empty(rows, 2 * INTER, dtype=torch.bfloat16, device=dev)
        KEEP.append(self)

    def __call__(self, x8, xs, shard, tap=True, act=None):
        p = torch.Tensor.data_ptr
        args = self.kernel.abi.pack({
            "x": p(x8), "x_scale": p(xs), "w": p(shard[0]), "w_scale": p(shard[1]),
            "act": p(self.act if act is None else act), "partials": p(PARTIALS),
            "gu": p(self.gu) if tap else 0,
        })  # fmt: skip
        self.kernel.launcher(*args, stream=torch.cuda.current_stream())


def bits(t):
    return t.contiguous().view(torch.int16)


def mismatch(got, want):
    bad = bits(got) != bits(want)
    d = (got.double() - want.double()).abs()
    return {
        "fraction": bad.double().mean().item(),
        "max_abs": d[torch.isfinite(d)].max().item() if bad.any() else 0.0,
        "first": [list(map(int, i)) for i in bad.nonzero()[:4].tolist()],
    }


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
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        fn()
    KEEP.append((graph, side, fn))
    return graph


def check(rows):
    mono = Mono(rows)
    res, hazards = resources(mono.kernel)
    shard = SHARDS[0]
    report = {
        "rows": rows, "inter": INTER, "split_k": mono.kernel.split_k,
        "kernel": mono.kernel.name, "resources": res, "hazards": hazards,
    }  # fmt: skip
    eager = []
    for scale in SCALES:
        x8, xs = inputs(rows, scale)
        mono(x8, xs, shard)
        gu, act = production(x8, xs, shard)
        torch.cuda.synchronize()
        exact = {
            "gu": torch.equal(bits(mono.gu), bits(gu)),
            "act": torch.equal(bits(mono.act), bits(act)),
        }
        eager.append(exact)
        gate = gu[:, :INTER].float()
        report.setdefault("gate_range", []).append(
            [round(gate.min().item(), 1), round(gate.max().item(), 1)]
        )
        for name, got, want in (("gu", mono.gu, gu), ("act", mono.act, act)):
            if not exact[name] and f"{name}_mismatch" not in report:
                report[f"{name}_mismatch"] = {"scale": scale, **mismatch(got, want)}
    gx8, gxs = inputs(rows)
    graph = graph_of(lambda: mono(gx8, gxs, shard, tap=False))
    replays = []
    for i in range(REPLAYS):
        x8, xs = inputs(rows, SCALES[i % len(SCALES)])
        gx8.copy_(x8)
        gxs.copy_(xs)
        torch.cuda.synchronize()
        graph.replay()
        _, act = production(x8, xs, shard)
        torch.cuda.synchronize()
        replays.append(torch.equal(bits(mono.act), bits(act)))
    report["eager_exact"] = eager
    report["replays_exact"] = sum(replays)
    report["replays"] = len(replays)
    report["ok"] = not hazards and all(all(e.values()) for e in eager) and all(replays)
    print(json.dumps(report), flush=True)
    return mono


def timed(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def bench(rows, mono):
    sets = [inputs(rows) for _ in range(COPIES)]
    acts = [torch.empty(rows, INTER, dtype=torch.bfloat16, device=dev)]
    KEEP.extend([sets, acts])

    def run_mono():
        for i in range(GN):
            mono(*sets[i % COPIES], SHARDS[i % COPIES], tap=False, act=acts[0])

    def run_prod():
        for i in range(GN):
            KEEP.append(production(*sets[i % COPIES], SHARDS[i % COPIES]))

    graphs = {"mono": graph_of(run_mono), "production_gemm_act": graph_of(run_prod)}
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, graph in graphs.items():
            times[name].append(timed(graph))
    out = {"rows": rows, "split_k": mono.kernel.split_k, "gn": GN}
    for name, t in times.items():
        out[name] = {
            "median": round(statistics.median(t), 2),
            "min": round(min(t), 2),
            "max": round(max(t), 2),
        }
    print(json.dumps({"bench": out}), flush=True)


for rows in ROWS:
    mono = check(rows) if CHECK else Mono(rows)
    if BENCH:
        bench(rows, mono)
