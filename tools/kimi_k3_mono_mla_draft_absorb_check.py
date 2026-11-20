# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The gfx1250 DSpark draft MLA absorb (``mla_absorb_draft_gfx1250``) on one
GPU against the decode branch of ``MLAAttention.forward_impl`` the draft's
layers run at TP8 / DCP8 with QREP, fed the tensors that branch sees:
q_b_proj replicated over the DCP group (HEADS x 192 rows of the online
per_1x128 E8M0 FP8, preshuffled), the FP8 q with its transposed fp32 scales
(``fused_qk_rmsnorm_group_quant``'s), ``W_K_qrep`` (FP8, one scale), the
draft's YaRN rotary embedding (GPT-J pairs, BF16 cos / sin), an FP8 KV cache
``[slots, 1, 576]`` with slots of -1, ``k_scale`` = ``q_scale`` = 1.
Production: q_b_proj's GEMM (``gemm_a8w8_blockscale_bpreshuffle``: the E8M0
weight scales widened to fp32, the gluon blockscale GEMM at the width's
split-K and its reduce), the W_UK batched GEMM and
``fused_qk_rope_concat_and_cache_mla`` (GPT-J, ``is_nope_first``,
``compute_all_q_rope``), q_out padded with zero heads to OUT_HEADS as
``_forward_decode``'s pad does.

Per width (CHECK): resources; q, ql_nope, q_out and the whole KV cache
against production's bits, q against an f64 reference. Then (BENCH)
CUDA-graph time over COPIES weight sets, and with FRONT=1 the whole
pre-attention part (``mla_draft_front`` + the absorb) against production's
sequence from the input norm's FP8 row.

    python tools/kimi_k3_mono_mla_draft_absorb_check.py
    env: ROWS (7,14,21,28), HEADS (64), OUT_HEADS, COMPUTE_ALL (1), CHECK (1),
    BENCH (1), FRONT (1), ITERS (2), GN (32), REPS (7), COPIES (32)
"""

import json
import os
import statistics

import aiter
import torch
from aiter import QuantType, dtypes, gemm_a8w8_blockscale_bpreshuffle
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as fp8_bmm,
)

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.model_ops.utils import dynamic_per_batched_tensor_quant
from atom.models.deepseek_v2 import _fused_rms_fp8_quant
from atom.models.kimi_k3.mono import draft_dispatch
from atom.models.kimi_k3.mono.config import HIDDEN
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_draft_gfx1250 as ma
from atom.models.kimi_k3.mono.kernels.stages import mla_absorb_layout as lay
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_gfx1250 as mf
from atom.models.kimi_k3.mono.kernels.stages import mla_draft_front_layout as flay
from atom.mono.runtime.compile import compile_kernels, launcher_resources
from atom.quantization.quark.utils import quant_weight_online

ROWS = [int(r) for r in os.environ.get("ROWS", "7,14,21,28").split(",")]
HEADS = int(os.environ.get("HEADS", "64"))
OUT_HEADS = int(os.environ.get("OUT_HEADS", str(HEADS)))
COMPUTE_ALL = os.environ.get("COMPUTE_ALL", "1") == "1"
CHECK = os.environ.get("CHECK", "1") == "1"
BENCH = os.environ.get("BENCH", "1") == "1"
FRONT = os.environ.get("FRONT", "1") == "1"
ITERS = int(os.environ.get("ITERS", "2"))
GN = int(os.environ.get("GN", "32"))
REPS = int(os.environ.get("REPS", "7"))
COPIES = int(os.environ.get("COPIES", "32"))
SLOTS = 8192
EPS = 1e-5
KEEP = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(13)


def rope():
    from aiter.rotary_embedding import get_rope

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    r = get_rope(
        64,
        rotary_dim=64,
        max_position=1048576,
        base=50000.0,
        is_neox_style=False,
        rope_scaling={
            "rope_type": "deepseek_yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 32768,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    ).to(dev)
    torch.set_default_dtype(default)
    return r


ROTARY = rope()
MAX_POS = ROTARY.cos_cache.size(0)
ONE = torch.ones(1, dtype=torch.float32, device=dev)


def bf16(*shape, scale=1.0, shift=0.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale + shift).to(
        torch.bfloat16
    )


def fp8_linear(rows, k, scale=0.02, dequantized=True):
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
        if FRONT:
            self.w_a, self.s_a, _ = fp8_linear(flay.QKV_A_ROWS, HIDDEN, 0.02, False)
            self.q_norm = bf16(flay.Q_LORA, scale=0.1, shift=1.0)
            self.kv_norm = bf16(flay.KV_LORA, scale=0.1, shift=1.0)


LAYERS = [Layer(i == 0) for i in range(COPIES if BENCH else 1)]
KEEP.append(LAYERS)


def inputs(rows):
    """The front's outputs (production's group quant on a random q) and a
    step's slot mapping and positions."""
    q_c = bf16(rows, lay.Q_LORA, scale=2.0)
    kv_c = bf16(rows, lay.KV_LORA)
    q_w = bf16(lay.Q_LORA, scale=0.1, shift=1.0)
    kv_w = bf16(lay.KV_LORA, scale=0.1, shift=1.0)
    q8, qs, _, kv, _ = _fused_rms_fp8_quant(
        q_c, q_w, EPS, kv_c, kv_w, EPS, None, dtype_quant=dtypes.fp8, group_size=128,
        quant_type=QuantType.per_1x128.value, output_unquantized_inp1=False,
        transpose_scale=True,
    )  # fmt: skip
    k_rope = bf16(rows, lay.ROPE, scale=200.0)  # past +-448: the unclamped cast
    kv[0, :8] = 1000.0  # past +-448: the saturating cast
    slots = torch.randperm(SLOTS, generator=gen, device=dev)[:rows].long()
    slots[1::3] = -1  # tokens another DCP rank owns
    positions = torch.randint(0, 200000, (rows,), generator=gen, device=dev).long()
    positions[0] = MAX_POS + 5  # clamped into the cache
    if rows > 2:
        positions[2] = -3
    return {"q_fp8": q8, "q_scale": qs, "kv": kv, "k_rope": k_rope, "slots": slots,
            "positions": positions}  # fmt: skip


class Absorb:
    def __init__(self, rows):
        self.rows = rows
        split = draft_dispatch.q_b_split_groups(rows, HEADS * lay.HEAD)
        self.kernel = ma.build_mla_absorb_draft(
            rows, HEADS, OUT_HEADS, COMPUTE_ALL, split, MAX_POS
        )
        e = torch.empty
        self.q = e(rows, HEADS, lay.HEAD, dtype=torch.bfloat16, device=dev)
        self.ql_nope = e(rows, HEADS, lay.KV_LORA, dtype=torch.bfloat16, device=dev)
        self.q_out = e(rows, OUT_HEADS, lay.ENTRY, dtype=dtypes.fp8, device=dev)
        KEEP.append(self)

    def __call__(self, inp, layer, cache, tap=True):
        p = torch.Tensor.data_ptr
        args = ma.MLA_ABSORB_DRAFT_ABI.pack({
            "q_fp8": p(inp["q_fp8"]), "q_scale": p(inp["q_scale"]),
            "w_qb": p(layer.w_qb), "s_qb": p(layer.s_qb), "w_uk": p(layer.w_uk),
            "s_uk": p(layer.s_uk), "kv": p(inp["kv"]), "k_rope": p(inp["k_rope"]),
            "k_rope_stride": inp["k_rope"].stride(0), "slots": p(inp["slots"]), "positions": p(inp["positions"]),
            "cos": p(ROTARY.cos_cache), "sin": p(ROTARY.sin_cache),
            "kv_cache": p(cache), "kv_stride": cache.stride(0),
            "q_out": p(self.q_out), "q": p(self.q),
            "ql_nope": p(self.ql_nope) if tap else 0,
        })  # fmt: skip
        self.kernel.launcher(*args, stream=torch.cuda.current_stream())


def production_q(inp, layer):
    return gemm_a8w8_blockscale_bpreshuffle(
        inp["q_fp8"], layer.w_qb, inp["q_scale"], layer.s_qb, torch.bfloat16
    ).view(-1, HEADS, lay.HEAD)


def production_uk(q, layer):
    q_nope = q[..., : lay.NOPE].transpose(0, 1)
    return fp8_bmm(q_nope, layer.w_uk, layer.s_uk, group_size=128, transpose_bm=True)


def production_rope_cache(ql_nope, q, inp, cache):
    """DCP (COMPUTE_ALL): every token's q_out, padded to OUT_HEADS after the
    write as ``_pad_decode_query_heads`` does; else the zeroed padded buffer
    of ``_fused_q_head_pad``, a token without a slot left zero."""
    rows = q.size(0)
    if COMPUTE_ALL:
        q_out = torch.empty(rows, HEADS, lay.ENTRY, dtype=dtypes.fp8, device=dev)
        q_write = q_out
    else:
        q_out = torch.zeros(rows, OUT_HEADS, lay.ENTRY, dtype=dtypes.fp8, device=dev)
        q_write = q_out[:, :HEADS]
    aiter.fused_qk_rope_concat_and_cache_mla(
        ql_nope, q[..., lay.NOPE :], inp["kv"], inp["k_rope"],
        cache.view(cache.shape[0], -1, lay.ENTRY), q_write, inp["slots"], ONE, ONE,
        inp["positions"], ROTARY.cos_cache, ROTARY.sin_cache,
        is_neox=ROTARY.is_neox_style, is_nope_first=True,
        compute_all_q_rope=COMPUTE_ALL,
    )  # fmt: skip
    if COMPUTE_ALL and OUT_HEADS != HEADS:
        q_out = torch.nn.functional.pad(q_out, (0, 0, 0, OUT_HEADS - HEADS))
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
    return {"max_abs": d.max().item(), "max_rel": d.max().item() / scale}


def new_cache():
    return torch.randint(
        0, 255, (SLOTS, 1, lay.ENTRY), generator=gen, device=dev, dtype=torch.uint8
    ).view(dtypes.fp8)


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
    return torch.empty(0, dtype=t.dtype, device=t.device).set_(
        t.untyped_storage(), t.storage_offset(), (t.numel(),)
    )


def check(rows):
    absorb = Absorb(rows)
    res, hazards = resources(absorb.kernel.launcher, absorb.kernel.abi)
    print(json.dumps({"rows": rows, "kernel": absorb.kernel.name, "resources": res,
                      "hazards": hazards}), flush=True)  # fmt: skip
    layer = LAYERS[0]
    report = {"rows": rows, "q_b_split_groups": absorb.kernel.qb_split_groups}
    ok = not hazards
    for it in range(ITERS):
        inp = inputs(rows)
        cache = new_cache()
        cache_ref = cache.clone()
        absorb(inp, layer, cache)
        q_prod = production_q(inp, layer)
        uk_mine = production_uk(absorb.q, layer)
        q_out_mine = production_rope_cache(absorb.ql_nope, absorb.q, inp, cache_ref)
        torch.cuda.synchronize()
        exact = {
            "q": torch.equal(bits(absorb.q), bits(q_prod)),
            "ql_nope": torch.equal(bits(absorb.ql_nope), bits(uk_mine)),
            "q_out": torch.equal(bits(absorb.q_out), bits(q_out_mine)),
            "kv_cache": torch.equal(bits(cache), bits(cache_ref)),
        }
        ok = ok and all(exact.values())
        if it == 0:
            scales = raw(inp["q_scale"]).view(lay.GROUPS, rows).T.double()
            xd = inp["q_fp8"].double() * scales.repeat_interleave(128, 1)
            q_ref = (xd @ layer.d_qb.T).view(rows, HEADS, lay.HEAD)
            prod_end = production(inp, layer, new_cache())
            torch.cuda.synchronize()
            report["q_vs_f64"] = diff(absorb.q, q_ref)
            report["production_q_vs_f64"] = diff(q_prod, q_ref)
            if not exact["q"]:
                report["q_vs_production"] = diff(absorb.q, q_prod)
                report["q_equal"] = (
                    (bits(absorb.q) == bits(q_prod)).double().mean().item()
                )
            if not exact["q_out"]:
                report["q_out_equal_bytes"] = (
                    (bits(absorb.q_out) == bits(q_out_mine)).double().mean().item()
                )
            report["q_out_end_to_end_equal"] = torch.equal(
                bits(absorb.q_out), bits(prod_end)
            )
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


QKV_C = torch.full(
    (flay.qkv_words(),), flay.UNWRITTEN - (1 << 32), dtype=torch.int32, device=dev
)
PARTIALS = torch.empty(flay.partial_bytes(), dtype=torch.uint8, device=dev)


class Front:
    def __init__(self, rows, inp):
        self.kernel = mf.build_mla_draft_front(
            rows, draft_dispatch.qkv_a_split_k(rows), EPS, EPS
        )
        self.inp = inp
        KEEP.append(self)

    def __call__(self, x, layer):
        p = torch.Tensor.data_ptr
        args = mf.MLA_DRAFT_FRONT_ABI.pack({
            "x": p(x["x"]), "x_scale": p(x["x_scale"]), "w_a": p(layer.w_a),
            "s_a": p(layer.s_a), "q_norm": p(layer.q_norm),
            "kv_norm": p(layer.kv_norm), "qkv_c": p(QKV_C),
            "q_fp8": p(self.inp["q_fp8"]), "q_scale": p(self.inp["q_scale"]),
            "kv": p(self.inp["kv"]), "k_rope": p(self.inp["k_rope"]),
            "partials": p(PARTIALS), "qkv_seen": 0,
        })  # fmt: skip
        self.kernel.launcher(*args, stream=torch.cuda.current_stream())


def production_front(x, layer):
    y = gemm_a8w8_blockscale_bpreshuffle(
        x["x"], layer.w_a, x["x_scale"], layer.s_a, torch.bfloat16
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
    return q8, qs, kv, k_pe


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
        xs = []
        for _ in range(COPIES):
            x = bf16(rows, HIDDEN, scale=0.5)
            xq, xsc = per_1x128_e8m0_quant(x, dtypes.fp8, True)
            xs.append({"x": xq.contiguous(), "x_scale": xsc})
        # as draft_mla runs it: the front where production splits
        # fused_qkv_a_proj's K in four, production's projections and norms
        # (the same bits) feeding the absorb elsewhere
        use_front = draft_dispatch.qkv_a_split_k(rows) == 4
        fronts = [Front(rows, inp) for inp in inps] if use_front else None
        KEEP.append(xs)

        def run_both():
            for i in range(GN):
                layer = LAYERS[i % COPIES]
                if use_front:
                    fronts[i % COPIES](xs[i % COPIES], layer)
                    inp = inps[i % COPIES]
                else:
                    q8, qs, kv, k_pe = production_front(xs[i % COPIES], layer)
                    inp = {"q_fp8": q8, "q_scale": qs, "kv": kv, "k_rope": k_pe,
                           "slots": inps[i % COPIES]["slots"],
                           "positions": inps[i % COPIES]["positions"]}  # fmt: skip
                    KEEP.append(inp)
                absorb(inp, layer, caches[i % 4], tap=False)

        def run_prod_all():
            for i in range(GN):
                layer = LAYERS[i % COPIES]
                q8, qs, kv, k_pe = production_front(xs[i % COPIES], layer)
                inp = {"q_fp8": q8, "q_scale": qs, "kv": kv, "k_rope": k_pe,
                       "slots": inps[i % COPIES]["slots"],
                       "positions": inps[i % COPIES]["positions"]}  # fmt: skip
                KEEP.append(production(inp, layer, caches[i % 4]))

        graphs["front+absorb"] = graph_of(run_both)
        graphs["production_all"] = graph_of(run_prod_all)
    times = {name: [] for name in graphs}
    for _ in range(REPS):
        for name, g in graphs.items():
            times[name].append(timed(g))
    med = {name: round(statistics.median(t), 2) for name, t in times.items()}
    print(json.dumps({"rows": rows, "heads": HEADS, "us_per_launch": med}), flush=True)


for rows in ROWS:
    absorb = check(rows) if CHECK else Absorb(rows)
    if BENCH:
        bench(rows, absorb)
