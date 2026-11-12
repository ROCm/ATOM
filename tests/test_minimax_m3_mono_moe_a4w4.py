# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Torch reference of the mono routed MoE under a4w4 (MXFP4 activations x MXFP4 weights), CPU only.

``fp4_quant_kernel_rule`` is the integer-op form of the quantization the mono kernels will run
(E8M0 RoundUp, ``code_ceil(amax * f32(1/6))``, the forward scale, RNE e2m1: aiter's
``_e8m0_roundup`` + ``emit_f32_to_e2m1``); it is held bit for bit to aiter's
``per_1x32_f4_quant`` (production's CPU reference of the same convention). ``ref_moe``
is the whole routed MoE for one rank at the M3.1 TP4 shapes for a16w4 / a8w4 / a4w4 activations.
"""

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes
from aiter.ops.quant import per_1x32_f4_quant
from aiter.utility import fp4_utils

HIDDEN, INTER, TOP_K, SWIGLU_LIMIT = 6144, 768, 4, 7.0
INV6 = torch.tensor(
    1.0 / 6.0, dtype=torch.float32
)  # 0x3E2AAAAB, the constant aiter's HIP / FlyDSL multiply by


def _code_ceil(v):
    """mono's ``code_ceil`` (atom/mono/device/mx.py): biased exponent of the smallest power of
    two >= v, clamped to [1, 254]."""
    bits = v.view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return e.clamp(1, 254)


def _f32_to_e2m1(q):
    """aiter ``emit_f32_to_e2m1`` (quant_utils.py:187) in torch int ops: RNE, saturating at 6."""
    qx = q.view(torch.int32)
    sign = qx & -(2**31)
    a = (qx & 0x7FFFFFFF).to(torch.int64)
    denorm_bias = 0x4A800000
    denorm = (
        a.to(torch.int32).view(torch.float32)
        + torch.tensor(denorm_bias, dtype=torch.int32).view(torch.float32)
    ).view(torch.int32) - denorm_bias
    mant_odd = (a >> 22) & 1
    normal = ((a + 0xC11FFFFF + mant_odd) & 0xFFFFFFFF) >> 22
    e = torch.where((a < 0x40C00000) & (a >= 0x3F800000), normal, torch.full_like(a, 7))
    e = torch.where(a < 0x3F800000, denorm.to(torch.int64), e)
    return ((sign >> 28) & 8).to(torch.int64) | e


def fp4_quant_kernel_rule(x):
    """x [rows, K] float32 -> (nibbles [rows, K] int64, e8m0 codes [rows, K/32] int32)."""
    r = x.float().view(x.shape[0], -1, 32)
    amax = r.abs().amax(-1)
    code = _code_ceil(amax * INV6)
    scale = (code << 23).view(torch.float32)  # the forward scale, 2^(code - 127)
    q = r / scale[..., None]  # exact: a power of two
    return _f32_to_e2m1(q.reshape(r.shape[0], -1).contiguous()), code


def _grid(nib):
    t = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    return t.to(nib.device)[nib]


def dq_fp4(x):
    nib, code = fp4_quant_kernel_rule(x)
    sc = (code << 23).view(torch.float32)
    return (_grid(nib).view(x.shape[0], -1, 32) * sc[..., None]).view(x.shape)


def dq_fp8(x):
    """MXFP8 as mono does today: ceil_pow2(amax / 448), RNE e4m3 saturating."""
    r = x.float().view(x.shape[0], -1, 32)
    code = _code_ceil(r.abs().amax(-1) * torch.tensor(1.0 / 448.0, dtype=torch.float32))
    sc = (code << 23).view(torch.float32)[..., None]
    return ((r / sc).clamp(-448, 448).to(torch.float8_e4m3fn).float() * sc).view(
        x.shape
    )


def _bf16(x):
    return x.to(torch.bfloat16).float()


def swiglu(g, u, limit=SWIGLU_LIMIT):
    g = torch.clamp(g, max=limit)
    u = torch.clamp(u, -limit, limit)
    return g * torch.sigmoid(1.702 * g) * (u + 1.0)


def ref_moe(x, w13, w2, route, rw, act="a4w4"):
    """x [T, HIDDEN] (bf16 values), w13[e] [2 INTER, HIDDEN] / w2[e] [HIDDEN, INTER] dequantized MXFP4
    (gate rows then up rows), route [T, TOP_K], rw [T, TOP_K] -> [T, HIDDEN] f32 (f32 accumulation).
    """
    q = {"a16w4": lambda v: v, "a8w4": dq_fp8, "a4w4": dq_fp4}[act]
    xq = q(_bf16(x))
    out = torch.zeros(x.shape[0], HIDDEN)
    for k in range(x.shape[0]):
        for s in range(TOP_K):
            e = int(route[k, s])
            gu = w13[e] @ xq[k]
            mid = q(_bf16(swiglu(gu[:INTER], gu[INTER:]))[None])[0]
            out[k] += rw[k, s] * (w2[e] @ mid)
    return out


def test_quant_matches_aiter_bit_for_bit():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 6144, generator=g) * torch.exp2(
        torch.randint(-6, 6, (256, 192), generator=g).float()
    ).repeat_interleave(32, 1)
    x = x.to(torch.bfloat16).float()
    nib, code = fp4_quant_kernel_rule(x)
    q, s = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2)
    ref_codes = s.view(torch.uint8).view(256, -1).to(torch.int32)
    ref_nib = (
        torch.stack([(q.view(torch.uint8) & 0xF), (q.view(torch.uint8) >> 4)], -1)
        .view(256, -1)
        .to(torch.int64)
    )
    assert torch.equal(code, ref_codes)
    assert torch.equal(nib, ref_nib)


def test_scale_boundary_multiply_equals_divide():
    """amax = 6 * 2^k: the HIP / FlyDSL kernels multiply by f32(1/6) (1 + 3e-8), aiter's CPU ref
    divides; the f32 product rounds back to exactly 2^k, so both give scale 2^k (no extra round-up).
    """
    for k in range(-4, 6):
        x = torch.zeros(1, 32)
        x[0, 0] = 6.0 * 2.0**k
        _, code = fp4_quant_kernel_rule(x)
        _, s = per_1x32_f4_quant(x, quant_dtype=dtypes.fp4x2)
        assert int(code[0, 0]) == int(s.view(torch.uint8)[0, 0]) == 127 + k


def test_zero_block_quantizes_to_zero():
    nib, _ = fp4_quant_kernel_rule(torch.zeros(2, 64))
    assert not (_grid(nib) != 0).any()


def test_fp4_error_is_about_five_times_fp8():
    """The numerics budget of the switch (PLAN.md 4.1): one routed-MoE call vs a16w4."""
    g = torch.Generator().manual_seed(1)
    E = 4
    deq = lambda w: _dq_weight(w)
    w13 = [deq(torch.randn(2 * INTER, HIDDEN, generator=g) * 0.03) for _ in range(E)]
    w2 = [deq(torch.randn(HIDDEN, INTER, generator=g) * 0.03) for _ in range(E)]
    x = torch.randn(2, HIDDEN, generator=g)
    route = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    rw = torch.full((2, TOP_K), 0.25)
    o16 = ref_moe(x, w13, w2, route, rw, "a16w4")
    e8 = ((ref_moe(x, w13, w2, route, rw, "a8w4") - o16).norm() / o16.norm()).item()
    e4 = ((ref_moe(x, w13, w2, route, rw, "a4w4") - o16).norm() / o16.norm()).item()
    assert 0.03 < e8 < 0.07 and 0.15 < e4 < 0.30, (e8, e4)


def _dq_weight(w):
    q, s = per_1x32_f4_quant(w, quant_dtype=dtypes.fp4x2)
    f = fp4_utils.mxfp4_to_f32(q.view(torch.uint8)).view(w.shape[0], -1, 32)
    sc = fp4_utils.e8m0_to_f32(s.view(dtypes.fp8_e8m0)).view(w.shape[0], -1, 1)
    return (f * sc).view(w.shape)
