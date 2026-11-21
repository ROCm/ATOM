# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Kimi-K3 DSpark draft's context KV projection (``ctx_kv_kernels``) on one
gfx1250 GPU against production, bit for bit, per context width:

production, a layer: ``per_1x128_e8m0_quant`` of the context tensor,
``gemm_a8w8_blockscale_bpreshuffle`` with the layer's ``fused_qkv_a_proj``
(2112 rows; the FlyDSL mxfp8_128 GEMM at the width's split-K and its reduce),
``fused_mla_ctx_norm_rope_cache`` of the kv_a | rope columns into the layer's
cache; here: one quant, one launch for every layer's kv_a | rope rows, then the
same cache write a layer ("five") or ``ctx_kv_store``'s one launch for every
layer's write ("one").

Checked per width: the launch's rows against the GEMM's columns 1536..2111,
every layer; the five KV caches whole, with either store; that the five
production quants of one tensor are one result; guards around the launch's
output; an eager run and REPLAYS CUDA-graph replays (outputs and caches reset
before each); no scratch, no spill. With BENCH=1: CUDA-graph time a context KV
sequence (five layers), production's against the launch's with five stores
("mono") and with one ("mono_one"), GN sequences a graph over SETS weight
sets, the graphs' replays alternating.

    python tools/kimi_k3_step_gap_ctx_check.py
    env: ROWS (8,16,24,32,40,48,56,64,5,10,15,20,25,30,35), REPLAYS (20),
    BENCH (0), GN (8), REPS (9), SETS (4), KSPLIT (production's)
"""

import json
import os
import statistics

import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.model_ops.triton_fused_mla_ctx_kv import fused_mla_ctx_norm_rope_cache
from atom.models.kimi_k3.mono import draft_dispatch
from atom.models.kimi_k3.mono.runner_gfx1250 import _resident_hazards
from atom.models.kimi_k3.step_gap import ctx_kv_kernels as K
from atom.models.kimi_k3.step_gap import ctx_kv_layout as lay
from atom.models.kimi_k3.step_gap import ctx_kv_store as S
from atom.mono.runtime.compile import _code_object, compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

ROWS = [
    int(r)
    for r in os.environ.get("ROWS", "8,16,24,32,40,48,56,64,5,10,15,20,25,30,35").split(
        ","
    )
]
REPLAYS = int(os.environ.get("REPLAYS", "20"))
BENCH = os.environ.get("BENCH", "0") == "1"
GN = int(os.environ.get("GN", "8"))
REPS = int(os.environ.get("REPS", "9"))
SETS = int(os.environ.get("SETS", "4")) if BENCH else 1
KSPLIT = os.environ.get("KSPLIT")
HIDDEN, LAYERS = 7168, 5
Q_LORA, KV_LORA, ROPE = 1536, 512, 64
QKV = Q_LORA + KV_LORA + ROPE
SLOTS = 4096
EPS = 1e-5
GUARD = 4096  # BF16 elements either side of the launch's output
MARK = 0x5A5A
KEEP = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(17)


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


class Layer:
    def __init__(self):
        self.w_a, self.s_a = fp8_linear(QKV, HIDDEN)
        self.kv_norm = bf16(KV_LORA, scale=0.1, shift=1.0)
        self.cache = torch.zeros(SLOTS, 1, KV_LORA + ROPE, dtype=dtypes.fp8, device=dev)
        self.cache_ref = torch.zeros_like(self.cache)


WEIGHTS = [[Layer() for _ in range(LAYERS)] for _ in range(SETS)]
TABS = [
    torch.tensor(
        [p for layer in layers for p in (layer.w_a.data_ptr(), layer.s_a.data_ptr())],
        dtype=torch.int64,
        device=dev,
    )
    for layers in WEIGHTS
]
STORE_TABS = [
    torch.tensor(
        [
            p
            for layer in layers
            for p in (layer.kv_norm.data_ptr(), layer.cache.data_ptr())
        ],
        dtype=torch.int64,
        device=dev,
    )
    for layers in WEIGHTS
]
PARTIALS = torch.empty(lay.partial_bytes(LAYERS), dtype=torch.uint8, device=dev)
assert all(
    bool((t != 0).all()) for t in TABS + STORE_TABS
), "a null pointer in a layer table"
KEEP += [WEIGHTS, TABS, STORE_TABS, PARTIALS]


def bits(t):
    return t.contiguous().view(torch.uint8)


def quant(h):
    return per_1x128_e8m0_quant(h, dtypes.fp8, True)


def gemm(x8, xs, layer):
    return gemm_a8w8_blockscale_bpreshuffle(
        x8, layer.w_a, xs, layer.s_a, torch.bfloat16
    )


def store(kv_lora, s, layer, cache):
    fused_mla_ctx_norm_rope_cache(
        kv_lora, layer.kv_norm, s["positions"], ROTARY.cos_cache, ROTARY.sin_cache, s["slots"], cache, ONE, EPS,
        KV_LORA, ROPE, ROTARY.is_neox_style, True,
    )  # fmt: skip


def production(s, layers, cache="cache_ref"):
    """A layer's (x8, x_scale, GEMM output), as ``write_context_kv`` runs it."""
    out = []
    for layer in layers:
        x8, xs = quant(s["h"])
        y = gemm(x8, xs, layer)
        store(y[..., :QKV][..., Q_LORA:], s, layer, getattr(layer, cache))
        out.append((x8, xs, y))
    return out


def production_ksplit(rows):
    return int(KSPLIT) if KSPLIT else draft_dispatch.qkv_a_split_k(rows)


class Mono:
    def __init__(self, rows):
        self.rows = rows
        self.ksplit = production_ksplit(rows)
        self.kernel = K.build_ctx_kv_proj(rows, self.ksplit, LAYERS)
        (self.launch,) = compile_kernels([(self.kernel.launcher, self.kernel.abi)])
        self.hazards = _resident_hazards(self.launch)
        self.resources = launcher_resources(self.launch)
        self.elf_bytes = [
            len(_code_object(a._ir_text)) for a in self.launch._mem_cache.values()
        ]
        words = GUARD + lay.out_bytes(LAYERS, rows) // 2 + GUARD
        self.buf = torch.full((words,), MARK, dtype=torch.int16, device=dev)
        self.out = (
            self.buf[GUARD : words - GUARD]
            .view(torch.bfloat16)
            .view(LAYERS, rows, lay.ENTRY)
        )
        KEEP.append(self)

    def project(self, x8, xs, tab):
        ptrs = {
            "tab": tab.data_ptr(), "x": x8.data_ptr(), "x_scale": xs.data_ptr(), "out": self.out.data_ptr(),
            "partials": PARTIALS.data_ptr(),
        }  # fmt: skip
        assert all(
            ptrs.values()
        ), ptrs  # (host values only: this runs inside captures too)
        assert x8.is_contiguous() and tuple(x8.shape) == (self.rows, HIDDEN), x8.shape
        assert xs.numel() == self.rows * (HIDDEN // 128), xs.shape
        self.kernel.launcher(
            *self.kernel.abi.pack(ptrs), stream=torch.cuda.current_stream()
        )

    def store_one(self, s, layers, store_tab):
        """Every layer's cache write in one launch (``ctx_kv_store``)."""
        S.store_layers(
            self.out, store_tab, s["positions"], ROTARY.cos_cache, ROTARY.sin_cache, s["slots"], layers[0].cache, ONE,
            EPS, KV_LORA, ROPE, ROTARY.is_neox_style, True,
        )  # fmt: skip

    def sequence(self, s, layers, tab, store_tab=None):
        x8, xs = quant(s["h"])
        x8 = x8.contiguous()
        self.project(x8, xs, tab)
        if store_tab is not None:
            self.store_one(s, layers, store_tab)
            return x8, xs
        for i, layer in enumerate(layers):
            store(self.out[i], s, layer, layer.cache)
        return x8, xs

    def guards_intact(self):
        return bool(
            (self.buf[:GUARD] == MARK).all() and (self.buf[-GUARD:] == MARK).all()
        )


def inputs(rows, special):
    h = bf16(rows, HIDDEN, scale=0.5)
    if special:
        h[0, :64] *= 40.0  # a group with a large amax
        h[rows - 1, 128:256] = 0.0  # a group of zeros
        h[rows // 2, 300:310] = 1e-30  # values far below their group's amax
    slots = torch.randperm(SLOTS, generator=gen, device=dev)[:rows].long()
    if special and rows > 2:
        slots[1::3] = -1  # rows another DCP rank owns: the store skips them
    positions = torch.randint(0, 200000, (rows,), generator=gen, device=dev).long()
    return {"h": h, "slots": slots, "positions": positions}


def reset(mono, layers):
    mono.out.view(torch.int16).fill_(0x7FC1)
    for layer in layers:
        layer.cache.zero_()


def compare(mono, ref, layers):
    """Mismatched BF16 values of the launch's rows, and mismatched cache bytes."""
    rows_bad = sum(
        int(
            (
                mono.out[i].view(torch.int16)
                != y[..., Q_LORA:QKV].contiguous().view(torch.int16)
            ).sum()
        )
        for i, (_, _, y) in enumerate(ref)
    )
    cache_bad = sum(
        int((bits(layer.cache) != bits(layer.cache_ref)).sum()) for layer in layers
    )
    return rows_bad, cache_bad


def check(rows):
    layers, tab = WEIGHTS[0], TABS[0]
    mono = Mono(rows)
    res = {
        "rows": rows, "ksplit": mono.ksplit, "kernel": mono.kernel.name, "grid": mono.kernel.grid[0],
        "hazards": list(mono.hazards), "resources": mono.resources, "elf_bytes": mono.elf_bytes,
    }  # fmt: skip
    for special in (False, True):
        s = inputs(rows, special)
        for layer in layers:
            layer.cache_ref.zero_()
        ref = production(s, layers)
        torch.cuda.synchronize()
        quants_equal = all(
            torch.equal(bits(x8), bits(ref[0][0]))
            and torch.equal(bits(xs), bits(ref[0][1]))
            for x8, xs, _ in ref
        )
        written = sum(
            int((bits(layer.cache_ref) != 0).any(dim=-1).sum()) for layer in layers
        )
        for variant, store_tab in (("five", None), ("one", STORE_TABS[0])):
            reset(mono, layers)
            mono.sequence(s, layers, tab, store_tab)
            torch.cuda.synchronize()
            eager = compare(mono, ref, layers)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                KEEP.append(mono.sequence(s, layers, tab, store_tab))
            KEEP.append(graph)
            replays_bad = 0
            for _ in range(REPLAYS):
                reset(mono, layers)
                graph.replay()
                torch.cuda.synchronize()
                replays_bad += compare(mono, ref, layers) != (0, 0)
            res[("special" if special else "plain") + "_" + variant] = {
                "production_quants_identical": quants_equal,
                "eager_row_mismatches": eager[0], "eager_cache_byte_mismatches": eager[1],
                "replays_bad": replays_bad, "replays": REPLAYS, "guards_intact": mono.guards_intact(),
                "cache_slots_written": written,
            }  # fmt: skip
    res["ok"] = not mono.hazards and all(
        v["production_quants_identical"] and v["eager_row_mismatches"] == 0 and v["eager_cache_byte_mismatches"] == 0
        and v["replays_bad"] == 0 and v["guards_intact"] and v["cache_slots_written"] > 0
        for v in (res["plain_five"], res["plain_one"], res["special_five"], res["special_one"])
    )  # fmt: skip
    print(json.dumps(res), flush=True)
    return mono, res["ok"]


def graph_of(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(side)
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


def bench(rows, mono):
    sets = [inputs(rows, False) for _ in range(SETS)]
    pre = [quant(s["h"]) for s in sets]
    pre = [(x8.contiguous(), xs) for x8, xs in pre]
    KEEP.append((sets, pre))

    def body(fn):
        def go():
            for i in range(GN):
                KEEP.append(fn(i % SETS))

        return go

    def stores_only(i):
        for j, layer in enumerate(WEIGHTS[i]):
            store(mono.out[j], sets[i], layer, layer.cache)

    arms = {
        "production": lambda i: production(sets[i], WEIGHTS[i], "cache"),
        "mono": lambda i: mono.sequence(sets[i], WEIGHTS[i], TABS[i]),
        "mono_one": lambda i: mono.sequence(
            sets[i], WEIGHTS[i], TABS[i], STORE_TABS[i]
        ),
        "mono_launch": lambda i: mono.project(pre[i][0], pre[i][1], TABS[i]),
        "stores": stores_only,
        "store_one": lambda i: mono.store_one(sets[i], WEIGHTS[i], STORE_TABS[i]),
    }
    graphs = {name: graph_of(body(fn)) for name, fn in arms.items()}
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    print(
        json.dumps({
            "rows": rows, "ksplit": mono.ksplit,
            "us_per_sequence": {
                n: {"median": round(statistics.median(t), 2), "min": round(min(t), 2), "max": round(max(t), 2)}
                for n, t in times.items()
            },
        }),
        flush=True,
    )  # fmt: skip


ok = True
for rows in ROWS:
    mono, good = check(rows)
    ok = ok and good
    if BENCH:
        bench(rows, mono)
print(json.dumps({"all_ok": ok, "widths": len(ROWS)}), flush=True)
