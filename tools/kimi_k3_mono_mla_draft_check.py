# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 DSpark draft MLA kernels on one GPU against what
``K3DSparkMLAAttention.forward`` runs at TP8 / DCP8 with QREP, fed the
tensors that path sees.

Front (``mla_draft_front_gfx1250``): the input norm's FP8 row and transposed
E8M0 codes, fused_qkv_a_proj's online per_1x128 E8M0 FP8 weight (preshuffled,
padded to 2176 rows); production: ``gemm_a8w8_blockscale_bpreshuffle`` (the
FlyDSL mxfp8_128 GEMM at the width's tuned split-K, and its reduce) and
``fused_qk_rmsnorm_group_quant`` (q quantized with fp32 per-1x128 scales,
transposed). Per width (CHECK): resources; the GEMM's BF16 rows, q's FP8 and
scales, kv and the rope rows against production's bits, the GEMM against an
f64 reference. Then (BENCH) CUDA-graph time over COPIES weight sets.

    python tools/kimi_k3_mono_mla_draft_check.py
    env: ROWS (7,14,21,28), CHECK (1), BENCH (1), ITERS (2), GN (32),
    REPS (7), COPIES (32), KSPLIT (production's)
"""

import json
import os
import statistics

import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.models.deepseek_v2 import _fused_rms_fp8_quant
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_gfx1250 as mf
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as flay
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

ROWS = [int(r) for r in os.environ.get("ROWS", "7,14,21,28").split(",")]
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
ITERS = int(os.environ.get("ITERS", "2"))
GN = int(os.environ.get("GN", "32"))
REPS = int(os.environ.get("REPS", "7"))
COPIES = int(os.environ.get("COPIES", "32"))
KSPLIT = os.environ.get("KSPLIT")
EPS = 1e-5
KEEP = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(5)


def bf16(*shape, scale=1.0, shift=0.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale + shift).to(
        torch.bfloat16
    )


def production_ksplit(rows):
    from aiter.ops.gemm_op_a8w8 import (
        _flydsl_mxfp8_fallback_kernel,
        get_mxscale_bpreshuffle_config,
    )

    if KSPLIT:
        return int(KSPLIT)
    cfg = get_mxscale_bpreshuffle_config(rows, flay.QKV_A_PADDED, HIDDEN, "128x128")
    if cfg is not None:
        return flay.qkv_a_split_k(cfg["kernelName"])
    return _flydsl_mxfp8_fallback_kernel(rows, flay.QKV_A_PADDED, HIDDEN).split_k


def fp8_linear(rows, k, scale=0.02):
    w = bf16(rows, k, scale=scale)
    q, s = quant_weight_online(
        w, QuantType.per_1x128, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
    )
    sc = s.float().repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)
    deq = (q.float() * sc).double()
    pad = -rows % 128
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous(), deq


class Layer:
    def __init__(self, dequantized=True):
        self.w_a, self.s_a, deq = fp8_linear(flay.QKV_A_ROWS, HIDDEN)
        self.d_a = deq if dequantized else None
        self.q_norm = bf16(flay.Q_LORA, scale=0.1, shift=1.0)
        self.kv_norm = bf16(flay.KV_LORA, scale=0.1, shift=1.0)


LAYERS = [Layer(i == 0) for i in range(COPIES if BENCH else 1)]
KEEP.append(LAYERS)


def inputs(rows):
    """The input norm's FP8 row and its transposed E8M0 codes."""
    x = bf16(rows, HIDDEN, scale=0.5)
    x[0, :64] *= 40.0  # a group with a large amax
    xq, xs = per_1x128_e8m0_quant(x, dtypes.fp8, True)
    return {"x": xq.contiguous(), "x_scale": xs}


QKV_C = torch.full(
    (flay.qkv_words(),), flay.UNWRITTEN - (1 << 32), dtype=torch.int32, device=dev
)
PARTIALS = torch.empty(flay.partial_bytes(), dtype=torch.uint8, device=dev)


class Front:
    def __init__(self, rows, ksplit):
        self.rows = rows
        self.kernel = mf.build_mla_draft_front(rows, ksplit, EPS, EPS)
        e = torch.empty
        self.q_fp8 = e(rows, flay.Q_LORA, dtype=dtypes.fp8, device=dev)
        self.q_scale = e(rows, flay.Q_GROUPS, dtype=torch.float32, device=dev)
        self.kv = e(rows, flay.KV_LORA, dtype=torch.bfloat16, device=dev)
        self.k_rope = e(rows, flay.ROPE, dtype=torch.bfloat16, device=dev)
        self.seen = e(rows, flay.QKV_C, dtype=torch.bfloat16, device=dev)
        KEEP.append(self)

    def __call__(self, inp, layer, tap=True):
        p = torch.Tensor.data_ptr
        args = mf.MLA_DRAFT_FRONT_ABI.pack({
            "x": p(inp["x"]), "x_scale": p(inp["x_scale"]), "w_a": p(layer.w_a),
            "s_a": p(layer.s_a), "q_norm": p(layer.q_norm),
            "kv_norm": p(layer.kv_norm), "qkv_c": p(QKV_C),
            "q_fp8": p(self.q_fp8), "q_scale": p(self.q_scale), "kv": p(self.kv),
            "k_rope": p(self.k_rope), "partials": p(PARTIALS),
            "qkv_seen": p(self.seen) if tap else 0,
        })  # fmt: skip
        self.kernel.launcher(*args, stream=torch.cuda.current_stream())


def production(inp, layer):
    y = gemm_a8w8_blockscale_bpreshuffle(
        inp["x"], layer.w_a, inp["x_scale"], layer.s_a, torch.bfloat16
    )
    q_c, kv_c, k_pe = torch.split(
        y[..., : flay.QKV_A_ROWS], [flay.Q_LORA, flay.KV_LORA, flay.ROPE], dim=-1
    )
    q8, qs, _, kv, _ = _fused_rms_fp8_quant(
        q_c, layer.q_norm, EPS, kv_c, layer.kv_norm, EPS, None,
        dtype_quant=dtypes.fp8, group_size=128,
        quant_type=QuantType.per_1x128.value, output_unquantized_inp1=False,
        transpose_scale=True,
    )  # fmt: skip
    return {"y": y, "q_fp8": q8, "q_scale": qs, "kv": kv, "k_pe": k_pe}


def bits(t):
    return t.contiguous().view(torch.uint8)


def diff(got, ref):
    got, ref = got.double(), ref.double()
    d = (got - ref).abs()
    scale = ref.abs().max().item() or 1.0
    return {"max_abs": d.max().item(), "max_rel": d.max().item() / scale}


def resources(launcher, abi):
    (launcher,) = compile_kernels([(launcher, abi)])
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


def raw(t):
    """A tensor's storage bytes in memory order (the transposed scales)."""
    return (
        torch.empty(0, dtype=t.dtype, device=t.device)
        .set_(t.untyped_storage(), t.storage_offset(), (t.numel(),))
        .view(torch.uint8)
    )


def check(rows, ksplit):
    front = Front(rows, ksplit)
    res, hazards = resources(front.kernel.launcher, front.kernel.abi)
    print(json.dumps({"rows": rows, "ksplit": ksplit, "kernel": front.kernel.name,
                      "resources": res, "hazards": hazards}), flush=True)  # fmt: skip
    layer = LAYERS[0]
    report = {"rows": rows, "ksplit": ksplit}
    ok = not hazards
    for it in range(ITERS):
        inp = inputs(rows)
        front(inp, layer)
        prod = production(inp, layer)
        torch.cuda.synchronize()
        exact = {
            "qkv_a": torch.equal(bits(front.seen), bits(prod["y"][:, : flay.QKV_C])),
            "k_rope": torch.equal(bits(front.k_rope), bits(prod["k_pe"])),
            "kv": torch.equal(bits(front.kv), bits(prod["kv"])),
            "q_fp8": torch.equal(bits(front.q_fp8), bits(prod["q_fp8"])),
            "q_scale": torch.equal(raw(front.q_scale), raw(prod["q_scale"])),
            "qkv_c_restored": bool((QKV_C == flay.UNWRITTEN - (1 << 32)).all()),
        }
        ok = ok and all(exact.values())
        if it == 0:
            xd = inp["x"].double() * torch.exp2(
                raw(inp["x_scale"]).view(flay.GROUPS, rows).T.double() - 127.0
            ).repeat_interleave(128, 1)
            ref = xd @ layer.d_a.T
            report["qkv_a_vs_f64"] = diff(prod["y"][:, : flay.QKV_A_ROWS], ref)
            if not exact["qkv_a"]:
                report["qkv_a_vs_production"] = diff(
                    front.seen, prod["y"][:, : flay.QKV_C]
                )
            if not exact["q_fp8"]:
                report["q_fp8_equal_bytes"] = (
                    (bits(front.q_fp8) == bits(prod["q_fp8"])).double().mean().item()
                )
            if not exact["kv"]:
                report["kv_vs_production"] = diff(front.kv, prod["kv"])
            if not exact["q_scale"]:
                mine = raw(front.q_scale).view(torch.float32).view(flay.Q_GROUPS, rows)
                ref = raw(prod["q_scale"]).view(torch.float32).view(flay.Q_GROUPS, rows)
                bad = mine != ref
                report["q_scale_mismatch"] = {
                    "fraction": bad.double().mean().item(),
                    "rows_with_any": bad.any(0).sum().item(),
                    "per_row": bad.sum(0).tolist(),
                    "max_rel": ((mine - ref).abs() / ref.abs()).max().item(),
                }
        report.setdefault("exact", []).append(exact)
    report["ok"] = ok
    print(json.dumps(report), flush=True)
    return front


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
    inps = [inputs(rows) for _ in range(COPIES)]
    KEEP.append(inps)

    def run_front():
        for i in range(GN):
            front(inps[i % COPIES], LAYERS[i % COPIES], tap=False)

    def run_prod():
        for i in range(GN):
            KEEP.append(production(inps[i % COPIES], LAYERS[i % COPIES]))

    graphs = {"draft_front": graph_of(run_front), "production": graph_of(run_prod)}
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    med = {name: round(statistics.median(t), 2) for name, t in times.items()}
    print(json.dumps({"rows": rows, "us_per_launch": med}), flush=True)


for rows in ROWS:
    ksplit = production_ksplit(rows)
    front = check(rows, ksplit) if CHECK else Front(rows, ksplit)
    if BENCH:
        bench(rows, front)
