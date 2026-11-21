# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What a Kimi-K3 DSpark draft layer's dense MLP and o_proj launch in
production on one gfx1250 GPU, fed a rank's TP shards (TP 8 by default): the
post-attention ``add_rmsnorm_quant`` (residual add, RMSNorm, per-1x128 E8M0
FP8), gate_up_proj's ``gemm_a8w8_blockscale_bpreshuffle`` (and its split-K
reduce), ``silu_and_mul``, down_proj's activation quant
(``per_1x128_e8m0_quant``) and GEMM; o_proj's activation quant and GEMM. The
TP all-reduce behind down_proj and o_proj is not run here (one GPU).

Per width: the kernels each stage launches and their time under the profiler
(names as the runtime reports them), and CUDA-graph time a launch of every
stage alone and of the whole MLP / o_proj sequence, the stages' inputs and
the weights rotating over COPIES sets, the graphs' replays alternating.

With MONO (default), the same sequences on the draft kernels beside them,
as ``draft_mlp`` runs them at one rank: production's norm, then
``draft_gate_up_gfx1250`` (where production splits K in 4 or 8; else
production's GEMM and activation) and ``draft_proj_ar_gfx1250`` (down_proj;
o_proj) -- the projection's all-reduce at one rank included, which
production's side does not run.

    python tools/kimi_k3_mono_draft_mlp_probe.py
    env: ROWS (7,14,21,28,56), TP (8), COPIES (32), GN (32), REPS (9),
    PROF (1), MONO (1)
"""

import collections
import itertools
import json
import os
import statistics
import tempfile

import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle, silu_and_mul
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.layernorm import _aiter_rms_quant
from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.quantization.quark.utils import quant_weight_online

ROWS = [int(r) for r in os.environ.get("ROWS", "7,14,21,28,56").split(",")]
TP = int(os.environ.get("TP", "8"))
COPIES = int(os.environ.get("COPIES", "32"))
GN = int(os.environ.get("GN", "32"))
REPS = int(os.environ.get("REPS", "9"))
PROF = os.environ.get("PROF", "1") == "1"
MONO = os.environ.get("MONO", "1") == "1"
BOUND = 1 << 22
HIDDEN = 7168
INTER = 14336 // TP
PROJ = 8192 // TP
EPS = 1e-6
KEEP = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(7)


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


class Layer:
    def __init__(self):
        self.norm = bf16(HIDDEN, scale=0.1, shift=1.0)
        self.w_gu, self.s_gu = fp8_linear(2 * INTER, HIDDEN)
        self.w_d, self.s_d = fp8_linear(HIDDEN, INTER)
        self.w_o, self.s_o = fp8_linear(HIDDEN, PROJ)


LAYERS = [Layer() for _ in range(COPIES)]
PER_1X128 = QuantType.per_1x128.value


def norm(x, res, layer):
    return _aiter_rms_quant(x, layer.norm, EPS, PER_1X128, True, res, dtypes.fp8, True)


def gemm(x, xs, w, s):
    return gemm_a8w8_blockscale_bpreshuffle(x, w, xs, s, torch.bfloat16)


def act(gu):
    out = torch.empty(gu.size(0), gu.size(1) // 2, dtype=gu.dtype, device=gu.device)
    silu_and_mul(out, gu)
    return out


def quant(x):
    return per_1x128_e8m0_quant(x, dtypes.fp8, True)


def stage_inputs(rows):
    """Every stage's input of every copy: the upstream stages' real outputs."""
    sets = []
    for layer in LAYERS:
        s = {"x": bf16(rows, HIDDEN, scale=0.5), "res": bf16(rows, HIDDEN)}
        s["x8"], s["xs"], s["res_out"] = norm(s["x"], s["res"], layer)
        s["gu"] = gemm(s["x8"], s["xs"], layer.w_gu, layer.s_gu)
        s["act"] = act(s["gu"])
        s["a8"], s["as"] = quant(s["act"])
        s["down"] = gemm(s["a8"], s["as"], layer.w_d, layer.s_d)
        s["core"] = bf16(rows, PROJ, scale=0.5)
        s["c8"], s["cs"] = quant(s["core"])
        s["o"] = gemm(s["c8"], s["cs"], layer.w_o, layer.s_o)
        sets.append(s)
    torch.cuda.synchronize()
    return sets


def mlp(s, layer):
    x8, xs, res = norm(s["x"], s["res"], layer)
    a8, a_s = quant(act(gemm(x8, xs, layer.w_gu, layer.s_gu)))
    return gemm(a8, a_s, layer.w_d, layer.s_d), res


def oproj(s, layer):
    c8, cs = quant(s["core"])
    return gemm(c8, cs, layer.w_o, layer.s_o)


if MONO:
    from atom.models.kimi_k3.mono import draft_dispatch
    from atom.models.kimi_k3.mono.kernels.stages import draft_gate_up_gfx1250 as gk
    from atom.models.kimi_k3.mono.kernels.stages import draft_mlp_layout as lay
    from atom.models.kimi_k3.mono.kernels.stages import draft_proj_ar_gfx1250 as dp
    from atom.mono.runtime.consensus import MonoUnsupported

    BANK = torch.zeros(lay.peer_bytes(1, max(ROWS)), dtype=torch.uint8, device=dev)
    PEERS = torch.tensor([BANK.data_ptr()], dtype=torch.int64, device=dev)
    EPOCHS = torch.zeros(lay.EPOCH_COUNTERS, dtype=torch.int32, device=dev)
    FLAGS = torch.zeros(4, dtype=torch.int32, device=dev)
    PARTIALS = torch.empty(
        lay.gate_up_partial_bytes(INTER), dtype=torch.uint8, device=dev
    )
    KEEP += [BANK, PEERS, EPOCHS, FLAGS, PARTIALS]


def mono_kernels(rows):
    """The width's draft kernels at one rank; gate_up None where production
    sums K in one chain."""
    try:
        split = draft_dispatch.mxfp8_split_k(rows, 2 * INTER, HIDDEN, "gate_up_proj")
        gate_up = gk.build_draft_gate_up(rows, INTER, split)
    except MonoUnsupported:
        gate_up = None
    kernels = {
        "gate_up": gate_up,
        "down": dp.build_draft_proj_ar(rows, 1, INTER, poll_bound=BOUND),
        "o_proj": dp.build_draft_proj_ar(rows, 1, PROJ, poll_bound=BOUND),
    }
    KEEP.append(kernels)
    return kernels


def mono_proj(kernel, x, w, s):
    out = torch.empty(x.size(0), HIDDEN, dtype=torch.bfloat16, device=dev)
    p = torch.Tensor.data_ptr
    kernel.launcher(
        *kernel.abi.pack({
            "x": p(x), "w": p(w), "w_scale": p(s), "out": p(out), "peers": p(PEERS),
            "epochs": p(EPOCHS), "rank": 0, "flags": p(FLAGS),
        }),
        stream=torch.cuda.current_stream(),
    )  # fmt: skip
    return out


def mono_mlp(kernels):
    def go(s, layer):
        x8, xs, res = norm(s["x"], s["res"], layer)
        if kernels["gate_up"] is None:
            a = act(gemm(x8, xs, layer.w_gu, layer.s_gu))
        else:
            a = torch.empty(x8.size(0), INTER, dtype=torch.bfloat16, device=dev)
            p = torch.Tensor.data_ptr
            kernels["gate_up"].launcher(
                *kernels["gate_up"].abi.pack({
                    "x": p(x8), "x_scale": p(xs), "w": p(layer.w_gu),
                    "w_scale": p(layer.s_gu), "act": p(a), "partials": p(PARTIALS),
                    "gu": 0,
                }),
                stream=torch.cuda.current_stream(),
            )  # fmt: skip
        return mono_proj(kernels["down"], a, layer.w_d, layer.s_d), res

    return go


STAGES = {
    "norm": lambda s, w: norm(s["x"], s["res"], w),
    "gate_up": lambda s, w: gemm(s["x8"], s["xs"], w.w_gu, w.s_gu),
    "act": lambda s, w: act(s["gu"]),
    "quant": lambda s, w: quant(s["act"]),
    "down": lambda s, w: gemm(s["a8"], s["as"], w.w_d, w.s_d),
    "mlp": mlp,
    "o_quant": lambda s, w: quant(s["core"]),
    "o_proj": lambda s, w: gemm(s["c8"], s["cs"], w.w_o, w.s_o),
    "o_quant_proj": oproj,
}


def graph_of(fn):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    KEEP.append((g, fn))
    return g


def timed(g):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def profiled(g):
    """The graph's kernels under the profiler, three replays: per kernel
    name, in launch order, launches a sequence and its p50 time."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
    path = tempfile.mktemp(suffix=".json")
    prof.export_chrome_trace(path)
    with open(path) as f:
        events = json.load(f)["traceEvents"]
    os.unlink(path)
    events = sorted(
        (e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"),
        key=lambda e: e["ts"],
    )
    times = collections.OrderedDict()
    for e in events:
        times.setdefault(e["name"], []).append(e["dur"])
    gaps = [b["ts"] - a["ts"] - a["dur"] for a, b in itertools.pairwise(events)]
    return {
        "kernels": [
            {
                "name": name[:200],
                "per_sequence": round(len(v) / (3 * GN), 2),
                "p50_us": round(statistics.median(v), 2),
            }
            for name, v in times.items()
        ],
        "gap_p50_us": round(statistics.median(gaps), 2) if gaps else None,
    }


def run(rows):
    sets = stage_inputs(rows)
    KEEP.append(sets)

    def body(fn):
        def go():
            for i in range(GN):
                KEEP.append(fn(sets[i % COPIES], LAYERS[i % COPIES]))

        return go

    stages = dict(STAGES)
    if MONO:
        kernels = mono_kernels(rows)
        stages["mlp_mono"] = mono_mlp(kernels)
        stages["o_proj_mono"] = lambda s, w: mono_proj(
            kernels["o_proj"], s["core"], w.w_o, w.s_o
        )
    graphs = {name: graph_of(body(fn)) for name, fn in stages.items()}
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    out = {
        "rows": rows,
        "tp": TP,
        "mono_gate_up": MONO and kernels["gate_up"] is not None,
        "weight_mb": {
            "gate_up": round(2 * INTER * HIDDEN / 1e6, 2),
            "down": round(HIDDEN * INTER / 1e6, 2),
            "o_proj": round(HIDDEN * PROJ / 1e6, 2),
        },
        "us_per_launch": {
            name: {
                "median": round(statistics.median(t), 2),
                "min": round(min(t), 2),
                "max": round(max(t), 2),
            }
            for name, t in times.items()
        },
    }
    print(json.dumps(out), flush=True)
    if PROF:
        for name in ("mlp", "o_quant_proj"):
            print(
                json.dumps({"rows": rows, "profile": name, **profiled(graphs[name])}),
                flush=True,
            )


for rows in ROWS:
    run(rows)
