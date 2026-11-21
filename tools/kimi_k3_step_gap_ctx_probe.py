# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What the Kimi-K3 DSpark draft's context path launches in production on one
gfx1250 GPU, per context width (the target rows of a step): the cat of the five
aux taps, ``context_proj``'s BF16 GEMM ([7168, 35840], replicated: no TP
shard), ``context_norm``, then a layer's E8M0 activation quant,
``fused_qkv_a_proj``'s FP8 GEMM (its split-K reduce) and the fused
norm + rope + cache store, five layers.

Per width: the kernels the sequence launches under the profiler, and the
CUDA-graph time a launch of every stage alone, of the whole production sequence
and of the same with the activation quant run once (the five layers quantize
the same tensor), GN sequences a graph, the graphs' replays alternating.

    python tools/kimi_k3_step_gap_ctx_probe.py
    env: ROWS (8,16,24,32,40,48,56,64), GN (8), REPS (7), SETS (4), PROF (1)
"""

import collections
import itertools
import json
import os
import statistics
import tempfile

import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle, rmsnorm2d_fwd
from aiter.ops.shuffle import shuffle_weight
from aiter.tuned_gemm import tgemm

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.model_ops.triton_fused_mla_ctx_kv import fused_mla_ctx_norm_rope_cache
from atom.quantization.quark.utils import quant_weight_online

ROWS = [int(r) for r in os.environ.get("ROWS", "8,16,24,32,40,48,56,64").split(",")]
GN = int(os.environ.get("GN", "8"))
REPS = int(os.environ.get("REPS", "7"))
SETS = int(os.environ.get("SETS", "4"))
PROF = os.environ.get("PROF", "1") == "1"
HIDDEN, TAPS, LAYERS = 7168, 5, 5
Q_LORA, KV_LORA, ROPE = 1536, 512, 64
QKV = Q_LORA + KV_LORA + ROPE
SLOTS = 8192
EPS = 1e-5
KEEP = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(11)


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
    pad = -rows % 128
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def rope():
    from aiter.rotary_embedding import get_rope

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    r = get_rope(
        64, rotary_dim=64, max_position=1048576, base=50000.0, is_neox_style=False,
        rope_scaling={
            "rope_type": "deepseek_yarn", "factor": 32.0, "original_max_position_embeddings": 32768,
            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0,
        },
    ).to(dev)  # fmt: skip
    torch.set_default_dtype(default)
    return r


ROTARY = rope()
ONE = torch.ones(1, dtype=torch.float32, device=dev)
CTX_W = bf16(HIDDEN, TAPS * HIDDEN, scale=0.01)
CTX_NORM = bf16(HIDDEN, scale=0.1, shift=1.0)


class Layer:
    def __init__(self):
        self.w_a, self.s_a = fp8_linear(QKV, HIDDEN)
        self.kv_norm = bf16(KV_LORA, scale=0.1, shift=1.0)
        self.cache = torch.zeros(SLOTS, 1, KV_LORA + ROPE, dtype=dtypes.fp8, device=dev)


WEIGHTS = [[Layer() for _ in range(LAYERS)] for _ in range(SETS)]


def cat(s):
    return torch.cat(s["taps"], dim=-1)


def proj(x):
    return tgemm.mm(x, CTX_W, None, otype=dtypes.bf16)


def norm(x):
    return rmsnorm2d_fwd(x, CTX_NORM, EPS)


def quant(x):
    return per_1x128_e8m0_quant(x, dtypes.fp8, True)


def gemm(x8, xs, layer):
    return gemm_a8w8_blockscale_bpreshuffle(
        x8, layer.w_a, xs, layer.s_a, torch.bfloat16
    )


def store(y, s, layer):
    fused_mla_ctx_norm_rope_cache(
        y[..., Q_LORA:QKV], layer.kv_norm, s["positions"], ROTARY.cos_cache, ROTARY.sin_cache, s["slots"],
        layer.cache, ONE, EPS, KV_LORA, ROPE, ROTARY.is_neox_style, True,
    )  # fmt: skip


def production(s, layers):
    h = norm(proj(cat(s)))
    for layer in layers:
        x8, xs = quant(h)
        store(gemm(x8, xs, layer), s, layer)
    return h


def quant_once(s, layers):
    h = norm(proj(cat(s)))
    x8, xs = quant(h)
    for layer in layers:
        store(gemm(x8, xs, layer), s, layer)
    return h


def kv_all(s, layers):
    for layer in layers:
        x8, xs = quant(s["h"])
        store(gemm(x8, xs, layer), s, layer)


STAGES = {
    "cat": lambda s, w: cat(s),
    "proj": lambda s, w: proj(s["cat"]),
    "norm": lambda s, w: norm(s["proj"]),
    "quant": lambda s, w: quant(s["h"]),
    "gemm": lambda s, w: gemm(s["x8"], s["xs"], w[0]),
    "store": lambda s, w: store(s["y"], s, w[0]),
    "kv_all": kv_all,
    "production": production,
    "quant_once": quant_once,
}


def inputs(rows):
    sets = []
    for layers in WEIGHTS:
        s = {"taps": [bf16(rows, HIDDEN, scale=0.5) for _ in range(TAPS)]}
        s["positions"] = torch.randint(
            0, 200000, (rows,), generator=gen, device=dev
        ).long()
        s["slots"] = torch.randperm(SLOTS, generator=gen, device=dev)[:rows].long()
        s["cat"] = cat(s)
        s["proj"] = proj(s["cat"])
        s["h"] = norm(s["proj"])
        s["x8"], s["xs"] = quant(s["h"])
        s["y"] = gemm(s["x8"], s["xs"], layers[0])
        sets.append(s)
    torch.cuda.synchronize()
    return sets


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
                "name": n[:110],
                "per_sequence": round(len(v) / (3 * GN), 2),
                "p50_us": round(statistics.median(v), 2),
            }
            for n, v in times.items()
        ],
        "gap_p50_us": round(statistics.median(gaps), 2) if gaps else None,
    }


def run(rows):
    sets = inputs(rows)
    KEEP.append(sets)

    def body(fn):
        def go():
            for i in range(GN):
                KEEP.append(fn(sets[i % SETS], WEIGHTS[i % SETS]))

        return go

    graphs = {name: graph_of(body(fn)) for name, fn in STAGES.items()}
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    print(
        json.dumps({
            "rows": rows,
            "us_per_sequence": {
                n: {"median": round(statistics.median(t), 2), "min": round(min(t), 2), "max": round(max(t), 2)}
                for n, t in times.items()
            },
        }),
        flush=True,
    )  # fmt: skip
    if PROF:
        print(
            json.dumps(
                {
                    "rows": rows,
                    "profile": "production",
                    **profiled(graphs["production"]),
                }
            ),
            flush=True,
        )


for rows in ROWS:
    run(rows)
