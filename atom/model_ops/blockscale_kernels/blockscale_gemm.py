# SPDX-License-Identifier: MIT
"""Group32 W4A8 GEMM against packed FP4 weights.

E4M3 activations and E2M1 weights use E8M0 row/group scales. The kernel
unpacks one group at a time through BF16 MFMA and accumulates in FP32.
"""

import triton
import triton.language as tl


@triton.jit
def _tile_scale(a_code, b_code):
    """2**(a + b - 254) as FP32, from two E8M0 codes.

    E8M0 code zero is 2**-127, not IEEE zero, so the codes are summed into one
    exponent field rather than converted separately. The sum saturates instead
    of wrapping the bit pattern; code 255 is NaN.
    """
    exponent = tl.minimum(tl.maximum(a_code + b_code - 127, 0), 255)
    scale = (exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
    return tl.where((a_code == 255) | (b_code == 255), float("nan"), scale)


@triton.jit(do_not_specialize=["M"])
def blockscale_gemm_fp4_kernel(
    A,
    B,
    AS,
    BS,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    PART_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """E4M3 activations against packed E2M1 weights, one scale group per tile.

    The nibbles unpack to BF16 for a plain MFMA and the tile sum is scaled
    afterwards, which is what fixes BK at a single group. Weight scales are
    per row here, so there is no group width to divide by.
    """
    tl.static_assert(BK == 32)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    groups: tl.constexpr = K // 32
    ks = tl.arange(0, BK)
    rows = row[:, None] < M
    cols = col < N
    # base steps by 32, so which nibble a K index selects never changes.
    shift = (ks[:, None] % 2) * 4

    acc = tl.zeros((BM, BN), tl.float32)
    start = split * PART_K
    for base in range(start, tl.minimum(start + PART_K, K), BK):
        offs = base + ks
        live = offs < K
        a = tl.load(
            A + row[:, None] * K + offs[None, :], rows & live[None, :], other=0.0
        )
        packed = tl.load(
            B + col[None, :] * (K // 2) + offs[:, None] // 2,
            cols[None, :] & live[:, None],
            other=0,
        )
        code = (packed >> shift) & 15
        magnitude = code & 7
        value = tl.where(
            magnitude < 4,
            magnitude * 0.5,
            tl.where(magnitude < 6, magnitude - 2.0, (magnitude - 4.0) * 2.0),
        )
        b = tl.where(code >= 8, -value, value)
        a_code = tl.load(AS + row * groups + base // 32, row < M, other=127).to(
            tl.int32
        )
        b_code = tl.load(BS + col * groups + base // 32, cols, other=127).to(tl.int32)
        scale = _tile_scale(a_code[:, None], b_code[None, :])
        acc += tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16)) * scale
    tl.store(
        C + split * M * N + row[:, None] * N + col[None, :], acc, rows & cols[None, :]
    )
