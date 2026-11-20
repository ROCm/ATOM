# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-CTA phase timeline of the gfx1250 MLA front (``tdm_timeline``): a CUDA
graph of GN launches over COPIES weight sets (cold, as between two MLA layers
in serving), the same graph built with the timeline on, then per CTA role
(q, kv, rope, gate GEMV CTAs and the norm CTAs) the median over launches of
each point's median and max over the role's CTAs, in us from the CTA's
entry: ``landS`` slab S's copies landed on the wave that waits for them
(only the waves sharing wave 0's SIMD counter: 0 and 4), ``slabS`` past its
barrier, ``looped`` the K loop done, ``exchanged`` the cluster exchange's
barrier passed, ``project_done`` the sums in LDS, ``stores_issued`` /
``stores_done`` the outputs stored / acknowledged, ``norm_polled`` a norm
wave's row seen whole, ``norm_reduced`` its rstd, ``norm_stored`` its stores
issued; ``dur`` a CTA's entry to end.

    python tools/kimi_k3_mono_mla_front_phase.py
    env: ROWS (8,24), GN (32), COPIES (32), BUILD (build_mla_front keywords,
    k=v,..: norm_ctas, gate_delay)
"""

import json
import os
import statistics

import torch
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_weight

from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import mla_front_gfx1250 as mf
from atom.models.kimi_k3.mono.kernels.stages import mla_front_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import tdm_timeline as tl
from atom.quantization.quark.utils import quant_weight_online

ROWS = [int(r) for r in os.environ.get("ROWS", "8,24").split(",")]
GN = int(os.environ.get("GN", "32"))
COPIES = int(os.environ.get("COPIES", "32"))
GATE = 1536
BUILD = {
    k: int(v)
    for k, v in (kv.split("=") for kv in os.environ.get("BUILD", "").split(",") if kv)
}
dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(3)
KEEP = []


def bf16(*shape, scale=1.0, shift=0.0):
    values = torch.randn(shape, generator=gen, device=dev) * scale + shift
    return values.to(torch.bfloat16)


def fp8_weight(rows):
    q, s = quant_weight_online(
        bf16(rows, HIDDEN, scale=0.02),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    pad = -rows % 128
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


LAYERS = []
for _ in range(COPIES):
    w_a, s_a = fp8_weight(lay.QKV_A_ROWS)
    w_g, s_g = fp8_weight(GATE)
    norms = bf16(lay.Q_LORA, scale=0.1, shift=1), bf16(lay.KV_LORA, scale=0.1, shift=1)
    LAYERS.append((w_a, s_a, w_g, s_g, *norms))
QKV_C = torch.full((lay.qkv_words(),), -1, dtype=torch.int32, device=dev)
POINTS = {
    **{f"land{s}": tl.SLAB0 + s * tl.SLAB_WORDS + tl.WEIGHT for s in (0, 4)},
    **{f"slab{s}": tl.SLAB0 + s * tl.SLAB_WORDS + tl.B for s in (0, 6)},
    "looped": tl.SPLIT_LOOPED,
    "exchanged": tl.SPLIT_EXCHANGED,
    "project_done": tl.PROJECT_DONE,
    "stores_issued": tl.STORES_ISSUED,
    "stores_done": tl.STORES_DONE,
    "norm_polled": tl.NORM_POLLED,
    "norm_reduced": tl.NORM_REDUCED,
    "norm_stored": tl.NORM_STORED,
}


def roles(norm_ctas):
    out = {}
    for cluster in range(lay.gemv_ctas(GATE) // 4):
        kind, _ = lay.cluster_rows(cluster, GATE)
        out.setdefault(kind, []).extend(range(4 * cluster, 4 * cluster + 4))
    first, count = lay.norm_placement(GATE, norm_ctas)
    if norm_ctas:
        out["norm"] = list(range(first, first + count))
    return out


def run(rows):
    buf = torch.zeros(tl.buffer_words(GN), dtype=torch.int32, device=dev)
    KEEP.append(buf)
    tl.enable(buf.data_ptr(), GN)
    timed_kernel = mf.build_mla_front(rows, GATE, **BUILD)
    tl.disable()
    plain = mf.build_mla_front(rows, GATE, **BUILD)

    def e(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device=dev)

    out = {
        "q_fp8": e(rows, lay.Q_LORA, dtype=dtypes.fp8),
        "q_scale": e(rows, lay.Q_GROUPS, dtype=dtypes.fp8_e8m0),
        "kv": e(rows, lay.KV_LORA),
        "k_rope": e(rows, lay.ROPE),
        "gate": e(rows, GATE),
        "partials": e(lay.partial_bytes(rows, GATE), dtype=torch.uint8),
    }
    xs = [bf16(rows, HIDDEN) for _ in range(COPIES)]
    KEEP.extend([out, xs])
    p = torch.Tensor.data_ptr

    def launch(kernel, i):
        w_a, s_a, w_g, s_g, q_norm, kv_norm = LAYERS[i % COPIES]
        args = {
            "x": p(xs[i % COPIES]), "w_a": p(w_a), "s_a": p(s_a), "w_g": p(w_g),
            "s_g": p(s_g), "q_norm": p(q_norm), "kv_norm": p(kv_norm),
            "qkv_c": p(QKV_C), "q": 0, "qkv_seen": 0,
        }  # fmt: skip
        args.update({name: p(t) for name, t in out.items()})
        kernel.launcher(
            *mf.MLA_FRONT_ABI.pack(args), stream=torch.cuda.current_stream()
        )

    us = {}
    for name, kernel in (("plain", plain), ("timeline", timed_kernel)):
        for i in range(2):
            launch(kernel, i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(GN):
                launch(kernel, i)
        KEEP.append(g)
        times = []
        for _ in range(7):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            g.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / GN)
        us[name] = round(statistics.median(times), 2)
    words = buf[tl.CTAS :].view(GN, tl.CTAS, tl.WORDS).long().cpu()
    m32 = 1 << 32
    per = {}
    norm_ctas = timed_kernel.grid[0] - lay.gemv_ctas(GATE)
    for launch_words in words[2:]:
        cyc0 = launch_words[:, tl.CYCLES_ENTRY]
        rt = (launch_words[:, tl.END] - launch_words[:, tl.ENTRY]) % m32
        cyc = (launch_words[:, tl.CYCLES_END] - cyc0) % m32
        live = (launch_words[:, tl.ENTRY] != 0) & (rt > 0)
        mhz = float((cyc[live].float() / (rt[live].float() / 100.0)).median())
        for role, ctas in roles(norm_ctas).items():
            idx = torch.tensor(ctas)
            idx = idx[live[idx]]
            if not len(idx):
                continue
            values = {"dur": rt[idx].float() / 100.0}
            for name, point in POINTS.items():
                stamped = idx[launch_words[idx, point] != 0]
                if len(stamped):
                    values[name] = (
                        (launch_words[stamped, point] - cyc0[stamped]) % m32
                    ).float() / mhz
            for name, v in values.items():
                per.setdefault(role, {}).setdefault(name, []).append(
                    (float(v.median()), float(v.max()))
                )
    phases = {
        role: {
            name: (
                round(statistics.median(a for a, _ in v), 2),
                round(statistics.median(b for _, b in v), 2),
            )
            for name, v in points.items()
        }
        for role, points in per.items()
    }
    print(
        json.dumps({"rows": rows, "build": BUILD, "us": us, "phases_us": phases}),
        flush=True,
    )


for rows in ROWS:
    run(rows)
