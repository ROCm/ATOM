# SPDX-License-Identifier: MIT
"""Group32 FP8 dot products with compact FP8 or packed FP4 weight storage.

Compute group32 dot products with BF16 MFMA, then retain scaled block sums in
FP64 until the output conversion. Quantized projections can cancel across blocks
with very different scales; losing a small term before A8 QAT amplifies the
error through the model. Inputs and weights remain in their native storage;
only register accumulators and the bounded split-K scratch use FP64.
"""

import triton
import triton.language as tl


@triton.jit
def blockscale_gemm_kernel(
    A,
    B,
    AS,
    BS,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    B_GROUP_N: tl.constexpr,
    FP4_WEIGHT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    PART_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    tl.static_assert(BK == 32)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    acc = tl.zeros((BM, BN), tl.float64)
    for base in range(split * PART_K, tl.minimum((split + 1) * PART_K, K), BK):
        offsets = base + tl.arange(0, BK)
        a = tl.load(
            A + row[:, None] * K + offsets[None, :],
            (row[:, None] < M) & (offsets[None, :] < K),
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        if FP4_WEIGHT:
            packed = tl.load(
                B + col[None, :] * (K // 2) + offsets[:, None] // 2,
                (col[None, :] < N) & (offsets[:, None] < K),
                other=0,
            )
            code = (packed >> ((offsets[:, None] % 2) * 4)) & 15
            magnitude = code & 7
            value = tl.where(
                magnitude < 4,
                magnitude * 0.5,
                tl.where(magnitude < 6, magnitude - 2.0, (magnitude - 4.0) * 2.0),
            )
            b = tl.where(code >= 8, -value, value).to(tl.float8e4nv)
        else:
            b = tl.load(
                B + col[None, :] * K + offsets[:, None],
                (col[None, :] < N) & (offsets[:, None] < K),
                other=0,
            ).to(tl.float8e4nv, bitcast=True)
        a_code = tl.load(AS + row * (K // 32) + base // 32, row < M, other=127).to(
            tl.uint32
        )
        b_code = tl.load(
            BS + (col // B_GROUP_N) * (K // 32) + base // 32, col < N, other=127
        ).to(tl.uint32)
        # E8M0 exponent zero is 2**-127, not IEEE FP32 zero. Form the
        # product directly as a normal FP64 power of two, including code zero.
        exponent = a_code[:, None] + b_code[None, :] + (1023 - 2 * 127)
        scale = (exponent.to(tl.uint64) << 52).to(tl.float64, bitcast=True)
        scale = tl.where(
            (a_code[:, None] == 255) | (b_code[None, :] == 255), float("nan"), scale
        )
        partial = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
        acc += partial.to(tl.float64) * scale
    out = C + split * M * N + row[:, None] * N + col[None, :]
    tl.store(out, acc, (row[:, None] < M) & (col[None, :] < N))
