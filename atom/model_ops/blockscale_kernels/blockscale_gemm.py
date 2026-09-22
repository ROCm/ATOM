# SPDX-License-Identifier: MIT
"""Group32 GEMMs against native FP8 or packed FP4 weights.

Both take E4M3 activations with E8M0 group scales, accumulate in FP32, and
index the compact weight scale grid. FP8 uses microscaling MFMA; its small-M
variant packs K panels into matrix rows/columns to avoid a split-K reduction.
FP4 unpacks and scales one group at a time through a BF16 MFMA.
"""

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def blockscale_gemm_fp8_kernel(
    A,
    B,
    AS,
    BS,
    C,
    # Runtime, not constexpr: M is the token count, and specializing on it
    # recompiles the kernel for every prefill chunk length.
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    B_GROUP_N: tl.constexpr,
    PART_K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    N_FIRST: tl.constexpr = False,
):
    """E4M3 x E4M3 with E8M0 group scales, on the CDNA4 microscaling MFMA.

    A tile spans BK/32 scale groups. Below BK 64 Triton lowers dot_scaled to a
    BF16 emulation instead, which is several times slower.
    """
    tl.static_assert(BK >= 64 and BK % 32 == 0)
    # The first grid dimension advances fastest. N-first traversal reuses A;
    # M-first traversal reuses B. The wrapper swaps the launch dimensions.
    row = tl.program_id(1 if N_FIRST else 0) * BM + tl.arange(0, BM)
    col = tl.program_id(0 if N_FIRST else 1) * BN + tl.arange(0, BN)
    split = tl.program_id(2)
    groups: tl.constexpr = K // 32
    ks = tl.arange(0, BK)
    gs = tl.arange(0, BK // 32)
    rows = row[:, None] < M
    cols = col < N
    # One scale row per B_GROUP_N output columns, so the column index is
    # divided rather than the grid expanded.
    bs_row = (col[:, None] // B_GROUP_N) * groups

    acc = tl.zeros((BM, BN), tl.float32)
    start = split * PART_K
    for base in range(start, tl.minimum(start + PART_K, K), BK):
        offs = base + ks
        span = base // 32 + gs
        live = offs < K
        held = span < groups
        a = tl.load(
            A + row[:, None] * K + offs[None, :], rows & live[None, :], other=0.0
        )
        b = tl.load(
            B + col[:, None] * K + offs[None, :],
            cols[:, None] & live[None, :],
            other=0.0,
        )
        a_code = tl.load(
            AS + row[:, None] * groups + span[None, :], rows & held[None, :], other=127
        )
        b_code = tl.load(
            BS + bs_row + span[None, :], cols[:, None] & held[None, :], other=127
        )
        # acc= leaves the sum in the matrix core's registers: one rounding per
        # tile instead of two, and no separate vector add.
        acc = tl.dot_scaled(a, a_code, "e4m3", b.T, b_code, "e4m3", acc=acc)
    tl.store(
        C + split * M * N + row[:, None] * N + col[None, :], acc, rows & cols[None, :]
    )


@triton.jit(do_not_specialize=["M"])
def blockscale_gemm_fp8_packed_kernel(
    A,
    B,
    AS,
    BS,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PACK: tl.constexpr,
):
    """Small-M group32 GEMM with K panels packed into MFMA rows/columns.

    Each packed row/column retains its own E8M0 scales. Only matching panel
    pairs contribute to the output; cross-panel products are discarded. One
    CTA owns the full K reduction, so no partial buffer or second launch is
    needed. BM is an unpacked token tile, independent of runtime M.
    """
    tl.static_assert(PACK == 1 or PACK == 2 or PACK == 4)
    tl.static_assert(BM * PACK >= 16)
    tl.static_assert(BK >= 128 and BK % 32 == 0)
    rows = tl.program_id(0) * BM + tl.arange(0, BM * PACK) // PACK
    a_panel = tl.arange(0, BM * PACK) % PACK
    cols = tl.program_id(1) * BN + tl.arange(0, BN * PACK) // PACK
    b_panel = tl.arange(0, BN * PACK) % PACK
    ks = tl.arange(0, BK)
    gs = tl.arange(0, BK // 32)
    groups: tl.constexpr = K // 32
    acc = tl.zeros((BM * PACK, BN * PACK), tl.float32)
    for base in range(0, K, BK * PACK):
        ak = base + a_panel[:, None] * BK + ks[None, :]
        bk = base + b_panel[:, None] * BK + ks[None, :]
        a = tl.load(
            A + rows[:, None] * K + ak,
            (rows[:, None] < M) & (ak < K),
            other=0.0,
        )
        b = tl.load(
            B + cols[:, None] * K + bk,
            (cols[:, None] < N) & (bk < K),
            other=0.0,
        )
        ag = base // 32 + a_panel[:, None] * (BK // 32) + gs[None, :]
        bg = base // 32 + b_panel[:, None] * (BK // 32) + gs[None, :]
        a_code = tl.load(
            AS + rows[:, None] * groups + ag,
            (rows[:, None] < M) & (ag < groups),
            other=127,
        )
        b_code = tl.load(
            BS + (cols[:, None] // 32) * groups + bg,
            (cols[:, None] < N) & (bg < groups),
            other=127,
        )
        acc = tl.dot_scaled(a, a_code, "e4m3", b.T, b_code, "e4m3", acc=acc)
    panels = acc.reshape(BM, PACK, BN, PACK).trans(0, 2, 1, 3)
    pair = tl.arange(0, PACK)
    diagonal = tl.where(
        pair[None, None, :, None] == pair[None, None, None, :], panels, 0.0
    )
    output = tl.sum(tl.sum(diagonal, 3), 2)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    tl.store(
        C + row[:, None] * N + col[None, :],
        output,
        (row[:, None] < M) & (col[None, :] < N),
    )


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
