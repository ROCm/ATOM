# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 MLA absorb (``mla_absorb_gfx1250``) on one GPU against the
decode branch of ``MLAAttention.forward_impl`` that Kimi-K3 serves at TP8 with
DCP8 and QREP, fed the tensors that branch sees: q_b_proj replicated over the
DCP group (96 heads x 192 rows of the online per_1x128 E8M0 FP8, preshuffled),
``W_K_qrep`` (96 x 512 x 128, ``dynamic_per_batched_tensor_quant``'s FP8 and
scale), the FP8 q and its transposed E8M0 codes the front writes, FP8 q_out
(``dtype_q``) and an FP8 KV cache ``[slots, 1, 576]`` with slots of -1 (the
tokens another DCP rank owns), the NoPE rotary embedding's bf16 cos / sin,
``k_scale`` = ``q_scale`` = 1. Production: q_b_proj's GEMM
(``gemm_a8w8_blockscale_bpreshuffle``), the absorb's
``batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant``
(gluon ``batched_gemm_a16w8`` on gfx1250) and
``fused_qk_rope_concat_and_cache_mla`` (``is_nope_first``,
``compute_all_q_rope``).

Per width (CHECK): resources; q against an f64 reference and production's;
production's W_UK on the absorb's own q (``ql_nope`` bit for bit) and its
rope + cache kernel on the absorb's own ql_nope and q (q_out and the whole KV
cache bit for bit); the end-to-end q_out against production's. HEADS /
OUT_HEADS / COMPUTE_ALL=0 take the other branch (this rank's 12 heads, q_out
padded to 16 zero heads, a token without a slot zero).

Then (BENCH) CUDA-graph time, GN launches a replay over COPIES weight sets,
REPS alternating replays: ``absorb`` against production's three kernels,
and with FRONT=1 the whole pre-attention part -- the front and the absorb --
against production's sequence from the normed row (both input quants, both
GEMMs and reduces, fused_qk_rmsnorm, q_b's quant, q_b, W_UK, rope + cache).

    python tools/kimi_k3_mono_mla_absorb_check.py
    env: ROWS (1,8,16,24,32,40,48,56,64), CHECK (1), BENCH (1), FRONT (1),
    ITERS (2), GN (32), REPS (7), COPIES (32), HEADS (96), OUT_HEADS,
    COMPUTE_ALL (1)
"""

import json
import os
import statistics

import aiter
import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as fp8_bmm,
)

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.model_ops.utils import dynamic_per_batched_tensor_quant
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_gfx1250 as ma
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import mla_front_gfx1250 as mf
from atom.models.kimi_k3.mono.kernels.stages import mla_front_layout as flay
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

ROWS = [int(r) for r in os.environ.get("ROWS", "1,8,16,24,32,40,48,56,64").split(",")]
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
FRONT = os.environ.get("FRONT", "1") == "1"
ITERS = int(os.environ.get("ITERS", "2"))
GN = int(os.environ.get("GN", "32"))
REPS = int(os.environ.get("REPS", "7"))
COPIES = int(os.environ.get("COPIES", "32"))
HEADS = int(os.environ.get("HEADS", "96"))
OUT_HEADS = int(os.environ.get("OUT_HEADS", str(HEADS)))
COMPUTE_ALL = os.environ.get("COMPUTE_ALL", "1") == "1"
GATE = 1536
SLOTS = 8192
MAX_POS = 4096
EPS = 1e-6
KEEP = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(11)
torch.set_default_dtype(torch.bfloat16)
from atom.model_ops.rotary_embedding import NoPositionalRotaryEmbedding

ROTARY = NoPositionalRotaryEmbedding(lay.ROPE, lay.ROPE, MAX_POS, 10000.0).to(dev)
torch.set_default_dtype(torch.float32)
ONE = torch.ones(1, dtype=torch.float32, device=dev)


def bf16(*shape, scale=1.0, shift=0.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale + shift).to(
        torch.bfloat16
    )


def fp8_linear(rows, k, scale=0.02, dequantized=True):
    """ATOM's online per_1x128 E8M0 FP8 linear: (preshuffled weight padded to
    128 rows, scales, the f64 dequantized weight or None)."""
    w = bf16(rows, k, scale=scale)
    q, s = quant_weight_online(
        w, QuantType.per_1x128, dtypes.fp8, scale_dtype=dtypes.fp8_e8m0
    )
    deq = None
    if dequantized:
        sc = s.float().repeat_interleave(128, 0)[:rows].repeat_interleave(128, 1)
        deq = (q.float() * sc).double()
    pad = -rows % 128
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous(), deq


class Layer:
    def __init__(self, dequantized=True):
        self.w_qb, self.s_qb, self.d_qb = fp8_linear(
            HEADS * lay.HEAD, lay.Q_LORA, 0.03, dequantized
        )
        w_uk = bf16(HEADS, lay.KV_LORA, lay.NOPE, scale=0.05)
        self.w_uk, self.s_uk = dynamic_per_batched_tensor_quant(w_uk, dtype=dtypes.fp8)
        self.s_uk = self.s_uk.reshape(1).contiguous()
        self.d_uk = self.w_uk.double() * self.s_uk.double()
        if FRONT:
            self.w_a, self.s_a, _ = fp8_linear(flay.QKV_A_ROWS, HIDDEN, 0.02, False)
            self.w_g, self.s_g, _ = fp8_linear(GATE, HIDDEN, 0.02, False)
            self.q_norm = bf16(flay.Q_LORA, scale=0.1, shift=1.0)
            self.kv_norm = bf16(flay.KV_LORA, scale=0.1, shift=1.0)


LAYERS = [Layer(i == 0) for i in range(COPIES if BENCH else 1)]
KEEP.append(LAYERS)


def inputs(rows):
    """The front's outputs for ``rows`` tokens and a step's slot mapping."""
    q = bf16(rows, lay.Q_LORA)
    q_fp8, q_scale = per_1x128_e8m0_quant(q, dtypes.fp8, True)
    kv = bf16(rows, lay.KV_LORA)
    k_rope = bf16(rows, lay.ROPE, scale=200.0)  # past +-448: the unclamped cast
    kv[0, :8] = 1000.0  # past +-448: the saturating cast
    k_rope[0, 1] = -0.0
    slots = torch.randperm(SLOTS, generator=gen, device=dev)[:rows].long()
    slots[1::3] = -1  # tokens another DCP rank owns
    positions = torch.randint(0, MAX_POS, (rows,), generator=gen, device=dev).long()
    return {
        "q_fp8": q_fp8.contiguous(),
        # the transposed codes as the GEMM reads them (not .contiguous())
        "q_scale": q_scale,
        "kv": kv,
        "k_rope": k_rope,
        "slots": slots,
        "positions": positions,
    }


class Absorb:
    def __init__(self, rows):
        self.rows = rows
        self.kernel = ma.build_mla_absorb(rows, HEADS, OUT_HEADS, COMPUTE_ALL)
        self.q = torch.empty(rows, HEADS, lay.HEAD, dtype=torch.bfloat16, device=dev)
        self.ql_nope = torch.empty(
            rows, HEADS, lay.KV_LORA, dtype=torch.bfloat16, device=dev
        )
        self.q_out = torch.empty(
            rows, OUT_HEADS, lay.ENTRY, dtype=dtypes.fp8, device=dev
        )
        KEEP.append(self)

    def __call__(self, inp, layer, cache, tap=True, q_out=None):
        p = torch.Tensor.data_ptr
        q_out = self.q_out if q_out is None else q_out
        args = ma.MLA_ABSORB_ABI.pack({
            "q_fp8": p(inp["q_fp8"]), "q_scale": p(inp["q_scale"]),
            "w_qb": p(layer.w_qb), "s_qb": p(layer.s_qb), "w_uk": p(layer.w_uk),
            "s_uk": p(layer.s_uk), "kv": p(inp["kv"]), "k_rope": p(inp["k_rope"]),
            "slots": p(inp["slots"]), "kv_cache": p(cache),
            "kv_stride": cache.stride(0), "q_out": p(q_out), "q": p(self.q),
            "ql_nope": p(self.ql_nope) if tap else 0,
        })  # fmt: skip
        self.kernel.launcher(*args, stream=torch.cuda.current_stream())
        return q_out


def production_q(inp, layer):
    return gemm_a8w8_blockscale_bpreshuffle(
        inp["q_fp8"], layer.w_qb, inp["q_scale"], layer.s_qb, torch.bfloat16
    ).view(-1, HEADS, lay.HEAD)


def production_uk(q, layer):
    q_nope = q[..., : lay.NOPE].transpose(0, 1)
    return fp8_bmm(q_nope, layer.w_uk, layer.s_uk, group_size=128, transpose_bm=True)


def production_rope_cache(ql_nope, q, inp, cache):
    rows = q.size(0)
    if OUT_HEADS != HEADS:
        q_out = torch.zeros(rows, OUT_HEADS, lay.ENTRY, dtype=dtypes.fp8, device=dev)
        q_write = q_out[:, :HEADS]
    else:
        q_out = torch.empty(rows, HEADS, lay.ENTRY, dtype=dtypes.fp8, device=dev)
        q_write = q_out
    aiter.fused_qk_rope_concat_and_cache_mla(
        ql_nope,
        q[..., lay.NOPE :],
        inp["kv"],
        inp["k_rope"],
        cache.view(cache.shape[0], -1, lay.ENTRY),
        q_write,
        inp["slots"],
        ONE,
        ONE,
        inp["positions"],
        ROTARY.cos_cache,
        ROTARY.sin_cache,
        is_neox=ROTARY.is_neox_style,
        is_nope_first=True,
        compute_all_q_rope=COMPUTE_ALL,
    )
    return q_out


def production(inp, layer, cache):
    q = production_q(inp, layer)
    return production_rope_cache(production_uk(q, layer), q, inp, cache)


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


def fp8_equal(a, b):
    return (bits(a) == bits(b)).double().mean().item()


def new_cache():
    return torch.randint(
        0, 255, (SLOTS, 1, lay.ENTRY), generator=gen, device=dev, dtype=torch.uint8
    ).view(dtypes.fp8)


def resources(kernel_launcher, abi):
    (launcher,) = compile_kernels([(kernel_launcher, abi)])
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


def dequant_q(q8, scale):
    rows = q8.size(0)
    codes = scale.contiguous().view(torch.uint8).view(flay.Q_GROUPS, rows).T
    factor = torch.exp2(codes.double() - 127.0).repeat_interleave(128, 1)
    return q8.double() * factor


def check(rows):
    absorb = Absorb(rows)
    res, hazards = resources(absorb.kernel.launcher, ma.MLA_ABSORB_ABI)
    print(json.dumps({"rows": rows, "kernel": absorb.kernel.name, "resources": res,
                      "hazards": hazards}), flush=True)  # fmt: skip
    layer = LAYERS[0]
    report = {"rows": rows}
    ok = not hazards
    for it in range(ITERS):
        inp = inputs(rows)
        cache = new_cache()
        cache_ref = cache.clone()
        absorb(inp, layer, cache)
        torch.cuda.synchronize()
        q_ref = (dequant_q(inp["q_fp8"], inp["q_scale"]) @ layer.d_qb.T).view(
            rows, HEADS, lay.HEAD
        )
        q_prod = production_q(inp, layer)
        # production's W_UK, rope and cache on the absorb's own q / ql_nope
        uk_mine = production_uk(absorb.q, layer)
        q_out_mine = production_rope_cache(absorb.ql_nope, absorb.q, inp, cache_ref)
        torch.cuda.synchronize()
        exact = {
            "ql_nope": torch.equal(bits(absorb.ql_nope), bits(uk_mine)),
            "q_out": torch.equal(bits(absorb.q_out), bits(q_out_mine)),
            "kv_cache": torch.equal(bits(cache), bits(cache_ref)),
        }
        # without the tap: the same q_out and cache
        cache2 = cache_ref.clone()
        before = absorb.q_out.clone()
        absorb(inp, layer, cache2, tap=False)
        torch.cuda.synchronize()
        exact["without_tap"] = torch.equal(bits(before), bits(absorb.q_out)) and (
            torch.equal(bits(cache2), bits(cache_ref))
        )
        ok = ok and all(exact.values())
        if it == 0:
            nope_ref = torch.einsum(
                "thk,hnk->thn", absorb.q[..., : lay.NOPE].double(), layer.d_uk
            )
            prod_end = production(inp, layer, new_cache())
            torch.cuda.synchronize()
            if not exact["ql_nope"]:
                report["ql_nope_vs_production_on_own_q"] = diff(absorb.ql_nope, uk_mine)
            if not exact["q_out"]:
                report["q_out_equal_bytes"] = fp8_equal(absorb.q_out, q_out_mine)
            report["q_vs_f64"] = diff(absorb.q, q_ref)
            report["q_vs_production"] = diff(absorb.q, q_prod)
            report["production_q_vs_f64"] = diff(q_prod, q_ref)
            report["ql_nope_vs_f64_on_own_q"] = diff(absorb.ql_nope, nope_ref)
            report["q_out_vs_production_end_to_end"] = {
                "equal_bytes": fp8_equal(absorb.q_out, prod_end),
                **diff(absorb.q_out.float(), prod_end.float()),
            }
        report.setdefault("exact", []).append(exact)
    report["ok"] = ok
    print(json.dumps(report), flush=True)
    return absorb


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


QKV_C = torch.full((flay.qkv_words(),), -1, dtype=torch.int32, device=dev)


class Front:
    """Stage 1 at a width, writing the absorb's inputs in place."""

    def __init__(self, rows, inp):
        self.kernel = mf.build_mla_front(rows, GATE, EPS, EPS)
        e = lambda *shape, dtype=torch.bfloat16: torch.empty(
            shape, dtype=dtype, device=dev
        )
        self.inp = inp
        self.gate = e(rows, GATE)
        self.partials = e(flay.partial_bytes(rows, GATE), dtype=torch.uint8)
        KEEP.append(self)

    def __call__(self, x, layer):
        p = torch.Tensor.data_ptr
        args = mf.MLA_FRONT_ABI.pack({
            "x": p(x), "w_a": p(layer.w_a), "s_a": p(layer.s_a),
            "w_g": p(layer.w_g), "s_g": p(layer.s_g),
            "q_norm": p(layer.q_norm), "kv_norm": p(layer.kv_norm),
            "qkv_c": p(QKV_C), "q": 0, "q_fp8": p(self.inp["q_fp8"]),
            "q_scale": p(self.inp["q_scale"]), "kv": p(self.inp["kv"]),
            "k_rope": p(self.inp["k_rope"]), "gate": p(self.gate),
            "partials": p(self.partials), "qkv_seen": 0,
        })  # fmt: skip
        self.kernel.launcher(*args, stream=torch.cuda.current_stream())


def production_front(x, layer):
    """``_gated_core``'s projections, as the front check times them."""
    xq, xs = per_1x128_e8m0_quant(x, dtypes.fp8, True)
    y = gemm_a8w8_blockscale_bpreshuffle(xq, layer.w_a, xs, layer.s_a, torch.bfloat16)
    q_c, kv_c, k_rope = torch.split(
        y[..., : flay.QKV_A_ROWS], [flay.Q_LORA, flay.KV_LORA, flay.ROPE], dim=-1
    )
    q = torch.empty_like(q_c)
    kv = torch.empty_like(kv_c)
    fused_qk_rmsnorm(
        q_out_quantized=q, q_out_scale=None, q=q_c, q_weight=layer.q_norm,
        q_epsilon=EPS, k_out=kv, k=kv_c, k_weight=layer.kv_norm, k_epsilon=EPS,
        quant_type=QuantType.No,
    )  # fmt: skip
    q_fp8, q_scale = per_1x128_e8m0_quant(q, dtypes.fp8, True)
    gq, gs = per_1x128_e8m0_quant(x, dtypes.fp8, True)
    gate = gemm_a8w8_blockscale_bpreshuffle(
        gq, layer.w_g, gs, layer.s_g, torch.bfloat16
    )
    return q_fp8, q_scale, kv, k_rope, gate


def bench(rows, absorb):
    inps = [inputs(rows) for _ in range(COPIES)]
    caches = [new_cache() for _ in range(4)]
    KEEP.extend([inps, caches])

    def run_absorb():
        for i in range(GN):
            absorb(inps[i % COPIES], LAYERS[i % COPIES], caches[i % 4], tap=False)

    def run_prod():
        for i in range(GN):
            KEEP.append(production(inps[i % COPIES], LAYERS[i % COPIES], caches[i % 4]))

    graphs = {"absorb": graph_of(run_absorb), "production": graph_of(run_prod)}
    if FRONT:
        xs = [bf16(rows, HIDDEN) for _ in range(COPIES)]
        fronts = [Front(rows, inp) for inp in inps]
        KEEP.append(xs)

        def run_both():
            for i in range(GN):
                fronts[i % COPIES](xs[i % COPIES], LAYERS[i % COPIES])
                absorb(inps[i % COPIES], LAYERS[i % COPIES], caches[i % 4], tap=False)

        def run_prod_all():
            for i in range(GN):
                layer = LAYERS[i % COPIES]
                q_fp8, q_scale, kv, k_rope, gate = production_front(
                    xs[i % COPIES], layer
                )
                inp = {"q_fp8": q_fp8, "q_scale": q_scale, "kv": kv,
                       "k_rope": k_rope, "slots": inps[i % COPIES]["slots"],
                       "positions": inps[i % COPIES]["positions"]}  # fmt: skip
                KEEP.append((production(inp, layer, caches[i % 4]), gate))

        graphs["front+absorb"] = graph_of(run_both)
        graphs["production_all"] = graph_of(run_prod_all)
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    med = {name: round(statistics.median(t), 2) for name, t in times.items()}
    print(json.dumps({"rows": rows, "us_per_launch": med}), flush=True)


for rows in ROWS:
    absorb = check(rows) if CHECK else Absorb(rows)
    if BENCH:
        bench(rows, absorb)
