# SPDX-License-Identifier: MIT
import triton
import triton.language as tl


@triton.jit
def _ceil_pow2_code(value):
    bits = value.to(tl.uint32, bitcast=True)
    return (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.uint32)


@triton.jit
def quantize_fp8_kernel(X, Y, S, SIZE, DEQUANT: tl.constexpr):
    # SIZE is the token count. A `tl.constexpr` here builds one kernel per
    # prefill chunk length and per batch size, and it only bounds the tiles.
    groups = tl.program_id(0) * 32 + tl.arange(0, 32)
    offsets = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + offsets, offsets < SIZE, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1e-4)
    code = _ceil_pow2_code(amax * (1.0 / 448.0))
    scale = (code << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
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
    amax = tl.max(tl.abs(x), 1)
    if E4M3_SCALE:
        scale = (
            (tl.maximum(amax, 6.0 * 2.0**-9) * (1.0 / 6.0))
            .to(tl.float8e4nv)
            .to(tl.float32)
        )
    else:
        exponent = _ceil_pow2_code(tl.maximum(amax, 6.0 * 2.0**-126) * (1.0 / 6.0))
        scale = (exponent << 23).to(tl.float32, bitcast=True)
    magnitude = tl.abs(x / scale[:, None])
    # E2M1 nearest-even rounding, including the asymmetric midpoint comparisons.
    code = (magnitude > 0.25).to(tl.uint8)
    code = tl.where(magnitude >= 0.75, 2, code)
    code = tl.where(magnitude > 1.25, 3, code)
    code = tl.where(magnitude >= 1.75, 4, code)
    code = tl.where(magnitude > 2.5, 5, code)
    code = tl.where(magnitude >= 3.5, 6, code)
    code = tl.where(magnitude > 5.0, 7, code)
    sign = (x.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8)
    if DEQUANT:
        value = tl.where(
            code < 4, code * 0.5, tl.where(code < 6, code - 2.0, (code - 4.0) * 2.0)
        )
        value = tl.where(sign != 0, -value, value)
        tl.store(Y + offsets, value * scale[:, None], offsets < SIZE)
    else:
        code = (code | (sign << 3)).reshape(32, GROUP // 2, 2)
        packed = tl.sum(code.to(tl.int32) << (tl.arange(0, 2)[None, None, :] * 4), 2)
        packed_offsets = (
            groups[:, None] * (GROUP // 2) + tl.arange(0, GROUP // 2)[None, :]
        )
        tl.store(Y + packed_offsets, packed.to(tl.uint8), packed_offsets < SIZE // 2)
        if E4M3_SCALE:
            tl.store(S + groups, scale, groups < SIZE // GROUP)
        else:
            tl.store(S + groups, exponent.to(tl.uint8), groups < SIZE // GROUP)
