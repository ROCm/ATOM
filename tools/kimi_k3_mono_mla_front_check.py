# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 MLA front (``mla_front_gfx1250``) on one GPU against the
production sequence it replaces in ``KimiFullAttention._gated_core``, on the
same inputs: fused_qkv_a_proj's and g_proj's per_1x128 E8M0 FP8 weights laid
out as ATOM's online quant leaves them (quantized, padded to 128 rows,
16x16-preshuffled), each Linear's activation quant and AITER's preshuffled
block-scale GEMM, AITER ``fused_qk_rmsnorm`` and q_b_proj's own
``per_1x128_e8m0_quant``.

Per width (CHECK): the launch's resources (VGPRs, SGPRs, spills, scratch,
LDS); the front's unnormalized q_a / kv_a rows, rope rows and gate against an
f64 reference of the BF16 input times the dequantized weights and against
production's GEMM outputs (max |diff|, max |diff| / max |ref|, BF16 rows equal
to the reference rounded); the norm and quant bit for bit against
production's kernels run on the front's own unnormalized rows (read back
through the ``qkv_seen`` tap); the hand-off words restored; a launch without
the taps writing the same outputs; and the normalized outputs against
production's end to end. ITERS launches, a fresh input each.

Then (BENCH) CUDA-graph time: GN launches a replay of the front and of the
production sequence, over COPIES weight sets and inputs (so neither reads a
cache the other warmed), REPS replays of each alternating; the median per
launch.

    python tools/kimi_k3_mono_mla_front_check.py
    env: ROWS (1,2,4,8,16,24,32,40,48,56,64), CHECK (1), BENCH (1),
    ITERS (4), GN (32), REPS (7), COPIES (32: ~870 MB of weights, past the
    MALL, as between two MLA layers in serving), GATE_ROWS (1536)
"""

import json
import os
import statistics

import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import mla_front_gfx1250 as mf
from atom.models.kimi_k3.mono.kernels.stages import mla_front_layout as lay
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

ROWS = [
    int(r) for r in os.environ.get("ROWS", "1,2,4,8,16,24,32,40,48,56,64").split(",")
]
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
ITERS = int(os.environ.get("ITERS", "4"))
GN = int(os.environ.get("GN", "32"))
REPS = int(os.environ.get("REPS", "7"))
COPIES = int(os.environ.get("COPIES", "32"))
GATE = int(os.environ.get("GATE_ROWS", "1536"))
EPS = 1e-6
KEEP = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(7)


def bf16(*shape, scale=1.0, shift=0.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale + shift).to(
        torch.bfloat16
    )


def fp8_weight(rows, dequantized=True):
    """ATOM's online per_1x128 E8M0 FP8 linear: (preshuffled weight padded to
    128 rows, its scales, the dequantized [rows, HIDDEN] f64 weight or None)."""
    w = bf16(rows, HIDDEN, scale=0.02)
    q, s = quant_weight_online(
        w, QuantType.per_1x128, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
    )
    deq = None
    if dequantized:
        scale = s.float().repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)
        deq = (q.float() * scale).double()
    pad = -rows % 128
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous(), deq


class Layer:
    def __init__(self, dequantized=True):
        self.w_a, self.s_a, self.d_a = fp8_weight(lay.QKV_A_ROWS, dequantized)
        self.w_g, self.s_g, self.d_g = fp8_weight(GATE, dequantized)
        self.q_norm = bf16(lay.Q_LORA, scale=0.1, shift=1.0)
        self.kv_norm = bf16(lay.KV_LORA, scale=0.1, shift=1.0)


LAYERS = [Layer(i == 0) for i in range(COPIES if BENCH else 1)]
KEEP.append(LAYERS)
# one hand-off buffer for every width, as the runner keeps it
QKV_C = torch.full((lay.qkv_words(),), -1, dtype=torch.int32, device=dev)


class Front:
    """A width's kernel and its buffers."""

    def __init__(self, rows):
        self.rows = rows
        self.kernel = mf.build_mla_front(rows, GATE, EPS, EPS)

        def e(*shape, dtype=torch.bfloat16):
            return torch.empty(shape, dtype=dtype, device=dev)

        self.seen = e(rows, lay.QKV_C)
        self.q = e(rows, lay.Q_LORA)
        self.q_fp8 = e(rows, lay.Q_LORA, dtype=dtypes.fp8)
        self.q_scale = e(rows, lay.Q_GROUPS, dtype=dtypes.fp8_e8m0)
        self.kv = e(rows, lay.KV_LORA)
        self.k_rope = e(rows, lay.ROPE)
        self.gate = e(rows, GATE)
        self.partials = torch.empty(
            lay.partial_bytes(rows, GATE), dtype=torch.uint8, device=dev
        )
        KEEP.append(self)

    def __call__(self, x, layer, q=True, seen=True):
        p = torch.Tensor.data_ptr
        args = mf.MLA_FRONT_ABI.pack({
            "x": p(x), "w_a": p(layer.w_a), "s_a": p(layer.s_a),
            "w_g": p(layer.w_g), "s_g": p(layer.s_g),
            "q_norm": p(layer.q_norm), "kv_norm": p(layer.kv_norm),
            "qkv_c": p(QKV_C), "q": p(self.q) if q else 0,
            "q_fp8": p(self.q_fp8), "q_scale": p(self.q_scale),
            "kv": p(self.kv), "k_rope": p(self.k_rope), "gate": p(self.gate),
            "partials": p(self.partials), "qkv_seen": p(self.seen) if seen else 0,
        })  # fmt: skip
        self.kernel.launcher(*args, stream=torch.cuda.current_stream())


def production(x, layer):
    """``_gated_core``'s sequence: fused_qkv_a_proj (its activation quant, the
    preshuffled GEMM, the padded rows dropped), the fused q_a / kv_a norm,
    q_b_proj's activation quant, g_proj."""
    xq, xs = per_1x128_e8m0_quant(x, dtypes.fp8, True)
    y = gemm_a8w8_blockscale_bpreshuffle(xq, layer.w_a, xs, layer.s_a, torch.bfloat16)
    q_c, kv_c, k_rope = torch.split(
        y[..., : lay.QKV_A_ROWS], [lay.Q_LORA, lay.KV_LORA, lay.ROPE], dim=-1
    )
    q, kv = norm(q_c, kv_c, layer)
    q_fp8, q_scale = per_1x128_e8m0_quant(q, dtypes.fp8, True)
    gq, gs = per_1x128_e8m0_quant(x, dtypes.fp8, True)
    gate = gemm_a8w8_blockscale_bpreshuffle(
        gq, layer.w_g, gs, layer.s_g, torch.bfloat16
    )
    return {
        "q_c": q_c, "kv_c": kv_c, "k_rope": k_rope, "q": q, "kv": kv,
        "q_fp8": q_fp8, "q_scale": q_scale, "gate": gate,
    }  # fmt: skip


def norm(q_c, kv_c, layer):
    """``_fuse_rmsnorm_quant``'s unquantized branch: AITER fused_qk_rmsnorm."""
    q = torch.empty_like(q_c)
    kv = torch.empty_like(kv_c)
    fused_qk_rmsnorm(
        q_out_quantized=q,
        q_out_scale=None,
        q=q_c,
        q_weight=layer.q_norm,
        q_epsilon=EPS,
        k_out=kv,
        k=kv_c,
        k_weight=layer.kv_norm,
        k_epsilon=EPS,
        quant_type=QuantType.No,
    )
    return q, kv


def bits(t):
    return t.contiguous().view(torch.uint8)


def diff(got, ref):
    got, ref = got.double(), ref.double()
    d = (got - ref).abs()
    scale = ref.abs().max().item() or 1.0
    return {
        "max_abs": d.max().item(),
        "max_rel": d.max().item() / scale,
        "bf16_equal": (got == ref.to(torch.bfloat16).double()).double().mean().item(),
    }


def resources(front):
    (launcher,) = compile_kernels([(front.kernel.launcher, mf.MLA_FRONT_ABI)])
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


def check(rows):
    front = Front(rows)
    res, hazards = resources(front)
    print(json.dumps({"rows": rows, "kernel": front.kernel.name, "resources": res,
                      "hazards": hazards}), flush=True)  # fmt: skip
    layer = LAYERS[0]
    report = {"rows": rows}
    ok = not hazards
    for it in range(ITERS):
        x = bf16(rows, HIDDEN)
        front(x, layer)
        torch.cuda.synchronize()
        mine = {
            "q_c": front.seen[:, : lay.Q_LORA],
            "kv_c": front.seen[:, lay.Q_LORA :],
            "k_rope": front.k_rope,
            "gate": front.gate,
        }
        # every launch leaves the hand-off words as it found them
        exact_restored = bool((QKV_C == -1).all())
        ref_a = x.double() @ layer.d_a.T
        ref = {
            "q_c": ref_a[:, : lay.Q_LORA],
            "kv_c": ref_a[:, lay.Q_LORA : lay.QKV_C],
            "k_rope": ref_a[:, lay.QKV_C :],
            "gate": x.double() @ layer.d_g.T,
        }
        prod = production(x, layer)
        # the norm and the quant bit for bit on the front's own rows
        q_n, kv_n = norm(mine["q_c"], mine["kv_c"], layer)
        q8, qs = per_1x128_e8m0_quant(front.q, dtypes.fp8, True)
        exact = {
            "kv": torch.equal(bits(front.kv), bits(kv_n)),
            "q": torch.equal(bits(front.q), bits(q_n)),
            "q_fp8": torch.equal(bits(front.q_fp8), bits(q8)),
            "q_scale": torch.equal(bits(front.q_scale), bits(qs)),
            "restored": exact_restored,
        }
        # a launch without the taps writes the same outputs
        before = [t.clone() for t in (front.q_fp8, front.q_scale, front.kv, front.gate)]
        front(x, layer, q=False, seen=False)
        torch.cuda.synchronize()
        exact["without_taps"] = all(
            torch.equal(bits(a), bits(b))
            for a, b in zip(before, (front.q_fp8, front.q_scale, front.kv, front.gate))
        )
        ok = ok and all(exact.values())
        if it == 0:
            report["vs_f64"] = {k: diff(mine[k], ref[k]) for k in mine}
            report["vs_production"] = {
                k: diff(mine[k], prod[k]) for k in ("q_c", "kv_c", "k_rope", "gate")
            }
            report["production_vs_f64"] = {
                k: diff(prod[k], ref[k]) for k in ("q_c", "kv_c", "k_rope", "gate")
            }
            report["normed_vs_production"] = {
                "q": diff(front.q, prod["q"]),
                "kv": diff(front.kv, prod["kv"]),
                "q_fp8_dequant": diff(
                    dequant_q(front.q_fp8, front.q_scale), dequant_q(
                        prod["q_fp8"], prod["q_scale"])),
            }  # fmt: skip
        report.setdefault("exact", []).append(exact)
    report["ok"] = ok
    print(json.dumps(report), flush=True)
    return front


def dequant_q(q8, scale):
    """[rows, 1536] FP8 times its transposed [12, rows] E8M0 codes."""
    rows = q8.size(0)
    codes = scale.contiguous().view(torch.uint8).view(lay.Q_GROUPS, rows).T
    factor = torch.exp2(codes.double() - 127.0).repeat_interleave(lay.QUANT_GROUP, 1)
    return q8.double() * factor


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
    KEEP.append(g)
    return g


def timed(g):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / GN


def bench(rows, front):
    xs = [bf16(rows, HIDDEN) for _ in range(COPIES)]
    KEEP.append(xs)

    def run_front():
        for i in range(GN):
            front(xs[i % COPIES], LAYERS[i % COPIES], q=False, seen=False)

    def run_prod():
        for i in range(GN):
            KEEP.append(production(xs[i % COPIES], LAYERS[i % COPIES]))

    graphs = {"front": graph_of(run_front), "production": graph_of(run_prod)}
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    med = {name: statistics.median(t) for name, t in times.items()}
    print(json.dumps({"rows": rows, "us_per_launch": med, "reps": times,
                      "speedup": med["production"] / med["front"]}), flush=True)  # fmt: skip


for rows in ROWS:
    front = check(rows) if CHECK else Front(rows)
    if BENCH:
        bench(rows, front)
