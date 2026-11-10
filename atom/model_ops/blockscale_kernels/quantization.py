# SPDX-License-Identifier: MIT
import torch
import triton
import triton.language as tl
from aiter import dtypes

# The FP8 an activation is quantized to, and the three forms every writer and
# reader of one of those rows needs. gfx942 is FNUZ and every other target is
# OCP E4M3; they differ in exponent bias, so a row written under one encoding
# and read under the other is garbage. Taking all three from one declaration is
# what keeps a clamp, a cast and a bitcast from disagreeing. The two a kernel
# reads are `tl.constexpr`: Triton refuses a plain global inside `@triton.jit`.
FP8_DTYPE = dtypes.fp8
FP8_MAX = tl.constexpr(torch.finfo(FP8_DTYPE).max)
FP8_TL_DTYPE = tl.constexpr(
    tl.float8e4b8 if FP8_DTYPE is torch.float8_e4m3fnuz else tl.float8e4nv
)
# The amax floor of every FP8 activation quant: aiter's MoE input quant's, so
# one model's quant sites all agree
AMAX_FLOOR = tl.constexpr(1e-10)


@triton.jit
def ceil_pow2_code(value):
    bits = value.to(tl.uint32, bitcast=True)
    return (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)


@triton.jit
def pow2_reciprocal(code):
    """1 / 2**(code - 127) for an E8M0 code. Exact, so a multiply by it rounds
    as the division by the scale does; a code past the finite range (an Inf
    amax) gives 0, which multiplies as the division by Inf does."""
    return (tl.maximum(254 - code.to(tl.int32), 0) << 23).to(tl.float32, bitcast=True)


@triton.jit
def fp8_groups(x):
    """``x`` [groups, 32] f32 as E4M3 with an E8M0 scale a group: (values,
    biased exponent codes, scales)."""
    amax = tl.maximum(tl.max(tl.abs(x), 1), AMAX_FLOOR)
    code = ceil_pow2_code(amax * (1.0 / FP8_MAX))
    scale = (code << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(
        tl.maximum(x * pow2_reciprocal(code)[:, None], -FP8_MAX), FP8_MAX
    ).to(FP8_TL_DTYPE)
    return q, code, scale


@triton.jit
def fp4_groups(x, E4M3_SCALE: tl.constexpr):
    """``x`` [groups, GROUP] f32 as E2M1 with a scale a group, E4M3 or E8M0:
    (magnitude codes, sign bits, scales, E8M0 exponents -- zero under E4M3)."""
    amax = tl.max(tl.abs(x), 1)
    if E4M3_SCALE:
        scale = (
            (tl.maximum(amax, 6.0 * 2.0**-9) * (1.0 / 6.0))
            .to(tl.float8e4nv)
            .to(tl.float32)
        )
        exponent = tl.zeros_like(amax).to(tl.uint32)
        magnitude = tl.abs(x / scale[:, None])
    else:
        exponent = ceil_pow2_code(tl.maximum(amax, 6.0 * 2.0**-126) * (1.0 / 6.0))
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        magnitude = tl.abs(x * pow2_reciprocal(exponent)[:, None])
    # E2M1 nearest-even rounding, including the asymmetric midpoint comparisons.
    code = (magnitude > 0.25).to(tl.uint8)
    code = tl.where(magnitude >= 0.75, 2, code)
    code = tl.where(magnitude > 1.25, 3, code)
    code = tl.where(magnitude >= 1.75, 4, code)
    code = tl.where(magnitude > 2.5, 5, code)
    code = tl.where(magnitude >= 3.5, 6, code)
    code = tl.where(magnitude > 5.0, 7, code)
    sign = (x.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8)
    return code, sign, scale, exponent


@triton.jit
def fp4_pack(code, sign):
    """`fp4_groups`' codes and signs [groups, GROUP] as bytes [groups,
    GROUP // 2], each element's E2M1 nibble low first."""
    pairs = tl.reshape(code | (sign << 3), (code.shape[0], code.shape[1] // 2, 2))
    lo, hi = tl.split(pairs)
    return lo | (hi << 4)


@triton.jit
def quantize_fp8_kernel(X, Y, S, SIZE, DEQUANT: tl.constexpr):
    # SIZE is the token count. A `tl.constexpr` here builds one kernel per
    # prefill chunk length and per batch size, and it only bounds the tiles.
    groups = tl.program_id(0) * 32 + tl.arange(0, 32)
    offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + offsets, offsets < SIZE, other=0).to(tl.float32)
    q, code, scale = fp8_groups(x)
    if DEQUANT:
        tl.store(Y + offsets, q.to(tl.float32) * scale[:, None], offsets < SIZE)
    else:
        tl.store(Y + offsets, q, offsets < SIZE)
        tl.store(S + groups, code.to(tl.uint8), groups < SIZE // 32)


@triton.jit
def quantize_fp4_kernel(
    X,
    Y,
    S,
    SIZE,  # runtime: see `quantize_fp8_kernel`
    GROUP: tl.constexpr,
    E4M3_SCALE: tl.constexpr,
    DEQUANT: tl.constexpr,
):
    groups = tl.program_id(0) * 32 + tl.arange(0, 32)
    offsets = groups[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(X + offsets, offsets < SIZE, other=0).to(tl.float32)
    code, sign, scale, exponent = fp4_groups(x, E4M3_SCALE)
    if DEQUANT:
        value = tl.where(
            code < 4, code * 0.5, tl.where(code < 6, code - 2.0, (code - 4.0) * 2.0)
        )
        value = tl.where(sign != 0, -value, value)
        tl.store(Y + offsets, value * scale[:, None], offsets < SIZE)
    else:
        packed_offsets = (
            groups[:, None] * (GROUP // 2) + tl.arange(0, GROUP // 2)[None, :]
        )
        tl.store(Y + packed_offsets, fp4_pack(code, sign), packed_offsets < SIZE // 2)
        if E4M3_SCALE:
            tl.store(S + groups, scale, groups < SIZE // GROUP)
        else:
            tl.store(S + groups, exponent.to(tl.uint8), groups < SIZE // GROUP)
