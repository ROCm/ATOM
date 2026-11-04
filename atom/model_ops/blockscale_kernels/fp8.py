# SPDX-License-Identifier: MIT
"""Pre-migration native FP8 kernels for AITER builds without group32 support.

Keep the original packed-K and split-K launch policy as a compatibility path.
The preferred AITER backend owns subsequent tuning and configuration dispatch.
"""

import os

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)


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
        # Triton 3.7's async LDS load can drop nonzero `other` for masked
        # scales. Load from valid addresses, then select the neutral scale
        # explicitly, preserving both K/N tails and the two-stage pipeline.
        safe_col = tl.minimum(cols[:, None], N - 1)
        safe_bg = tl.minimum(bg, groups - 1)
        b_code = tl.load(BS + (safe_col // 32) * groups + safe_bg)
        b_code = tl.where((cols[:, None] < N) & (bg < groups), b_code, 127)
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


def _select_fp8_tile(m: int, n: int, k: int, cu_num: int):
    """(BM, BN, BK, num_warps, num_stages, N_FIRST) for the FP8 kernel.

    Swept through ``native_quant_linear`` so each candidate ran with its real
    split count, and with the inputs rotated -- a sweep on pinned tensors reads
    a third of the decode time and picks differently. The M bands are also the
    JIT variant count per projection; N and K are constexpr there, so branching
    on them is free.

    Small M is weight-bandwidth bound on a grid too small to fill the CUs, so
    BM drops to the MFMA minimum and BK widens. The tile squares up on the grid
    it would leave rather than on N: a narrow projection at m=1024 still wants
    the small tile, because a 128-wide one leaves 80 workgroups of 256.
    """
    # Short reductions benefit from reusing A across adjacent N tiles. Keep
    # the grid large enough to fill the CUs before moving to a 128-wide tile.
    if 32 < m <= 2048 and k <= 2048 and n >= 4096 and get_gfx() == "gfx950":
        if m <= 64:
            if 8192 <= n < 32768:
                return (
                    (32, 32, 256, 2, 2, True)
                    if n < 16384
                    else (64, 64, 256, 4, 2, True)
                )
        elif m <= 128 or -(-m // 128) * -(-n // 128) < cu_num:
            if -(-m // 64) * -(-n // 64) >= cu_num:
                # Wide weights at small M benefit more from reusing B.
                return 64, 64, 256, 4, 2, m > 128 or n < 32768
        else:
            return 128, 128, 256, 4, 2 if m * n <= 2**22 else 1, True
    if m <= 16:
        # Deep, wide weights already fill the GPU without split-K; a shorter
        # K tile reduces the live operand footprint.
        return 16, 32, 256 if n * k >= 2**27 else 512, 2, 2, False
    if m <= 64:
        # A wide, deep weight has enough work per output tile to pay for a
        # square one this early; kv_a and wq_a at m=32 do not.
        return (
            (64, 64, 256, 4, 2, False) if n * k >= 2**25 else (32, 32, 512, 2, 2, False)
        )
    if m <= 256:
        # A narrow projection cannot make enough 64-wide tiles to fill the CUs
        # here -- kv_a at m=128 leaves 16 workgroups -- and split-K only lifts
        # that to 160. Halving the tile is what fills it. Picked on the sum
        # over N in {512, 768, 1024} x m in {128, 256}, not on kv_a alone: the
        # 16x16x1024 tile that wins kv_a at m=128 is 30% slower at N=768.
        return (32, 32, 512, 2, 2, False) if n <= 1024 else (64, 64, 256, 4, 2, False)
    if n <= 1024 or -(-m // 128) * -(-n // 128) < cu_num // 2:
        return 64, 64, 256, 4, 2, False
    return 128, 128, 256, 4, 1, False


def _select_fp8_packed_tile(m: int, n: int, k: int):
    """(BM, BN, BK, PACK) where a full-K CTA beats split-K on gfx950.

    Narrow N still needs split-K to occupy the GPU. Large M or deep weights
    need the original kernel's tiling; only short reductions pay for packing
    beyond 16 tokens. These bands keep M a runtime argument in both kernels.
    """
    if m > 32 or n < 2048 or k > 6144 or n * k >= 2**27:
        return None
    if m > 16:
        return (32, 32, 256, 1) if k <= 2048 and n <= 6144 else None
    if k <= 1024:
        return 16, 32, 256, 1
    if k <= 2048:
        if n >= 32768 and m > 4:
            return None
        pack = 4 if m <= 4 else 2
        bn = 32 if m <= 4 and (n >= 8192 or k == 2048) else 16
        return max(16 // pack, triton.next_power_of_2(m)), bn, 256, pack
    if k > 4096 and m > 4:
        return None
    pack = 4 if m <= 4 else 2
    return max(16 // pack, triton.next_power_of_2(m)), 16, 512, pack


def _gemm_fp8_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    weight_group_rows: int,
    split_k: int | None,
    cu_num: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=dtype)


@torch_compile_guard(mutates_args=[], gen_fake=_gemm_fp8_fake)
def gemm_fp8_local(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    weight_group_rows: int,
    split_k: int | None,
    cu_num: int,
) -> torch.Tensor:
    """Run the original FP8 path on validated contiguous native operands."""
    n, k = weight.shape
    m = x.numel() // k
    output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=dtype)
    if m == 0:
        return output
    packed_tile = (
        _select_fp8_packed_tile(m, n, k)
        if weight_group_rows == 32 and split_k is None and get_gfx() == "gfx950"
        else None
    )
    if packed_tile is not None:
        bm, bn, bk, pack = packed_tile
        blockscale_gemm_fp8_packed_kernel[(-(-m // bm), -(-n // bn))](
            x,
            weight,
            x_scale.view(torch.uint8),
            weight_scale.view(torch.uint8),
            output,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            pack,
            num_warps=2,
            num_stages=2,
            matrix_instr_nonkdim=16,
        )
        return output

    bm, bn, bk, warps, stages, n_first = _select_fp8_tile(m, n, k, cu_num)
    if split_k is None:
        blocks = -(-m // bm) * -(-n // bn)
        split_k = (
            1
            if blocks >= cu_num
            else max(1, min(16, k // bk, -(-(2 * cu_num) // blocks)))
        )
    slice_k = -(-k // split_k)
    part_k = -(-slice_k // bk) * bk
    splits = -(-k // part_k)
    partial = (
        output
        if splits == 1
        else torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
    )
    grid_m, grid_n = -(-m // bm), -(-n // bn)
    grid = (grid_n, grid_m, splits) if n_first else (grid_m, grid_n, splits)
    blockscale_gemm_fp8_kernel[grid](
        x,
        weight,
        x_scale.view(torch.uint8),
        weight_scale.view(torch.uint8),
        partial,
        m,
        n,
        k,
        weight_group_rows,
        part_k,
        bm,
        bn,
        bk,
        N_FIRST=n_first,
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16 if n_first and bm == 32 else 0,
    )
    if splits > 1:
        _gemm_splitk_reduce_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](
            partial,
            output,
            None,
            m,
            n,
            m * n,
            n,
            1,
            n,
            1,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=32,
            ACTUAL_KSPLIT=splits,
            MAX_KSPLIT=triton.next_power_of_2(splits),
            ADD_BIAS=False,
            activation=None,
            use_activation=False,
            KERNEL_NAME="native_blockscale_reduce",
        )
    return output


@triton.jit
def _e8m0_to_f32(code):
    """E8M0 scale codes (uint8) -> float32 powers of two."""
    return (code.to(tl.uint32) << 23).to(tl.float32, bitcast=True)


@triton.jit
def _group32_dequant_act_kernel(x_ptr, s_ptr, y_ptr, K: tl.constexpr, G: tl.constexpr):
    """One group of 32 FP8 activations -> BF16. Decode launches this instead of
    the eager cast/mul chain, which is several kernels for a few thousand elements."""
    row = tl.program_id(0)
    grp = tl.program_id(1)
    offs = grp * 32 + tl.arange(0, 32)
    vals = tl.load(x_ptr + row * K + offs).to(tl.float32)
    code = tl.load(s_ptr + row * G + grp)
    tl.store(y_ptr + row * K + offs, (vals * _e8m0_to_f32(code)).to(tl.bfloat16))


def _dequant_group32_act(x: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
    """FP8 group32 activations to BF16. ``x``'s last dim is K, contiguous."""
    k = x.shape[-1]
    m = x.numel() // k
    out = torch.empty(x.shape, device=x.device, dtype=torch.bfloat16)
    if m == 0:
        return out
    groups = k // 32
    flat_x = x.reshape(m, k)
    # Triton has no pointer type for float8_e8m0fnu; the codes are raw bytes.
    flat_s = x_scale.view(torch.uint8).reshape(m, groups)
    flat_y = out.view(m, k)
    _group32_dequant_act_kernel[(m, groups)](
        flat_x, flat_s, flat_y, K=k, G=groups, num_warps=1
    )
    return out


@triton.jit(do_not_specialize=["M"])
def blockscale_gemm_fp8_bf16_dequant_kernel(
    A,
    B,
    AS,
    BS,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    B_GROUP_N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """gfx942 group32 GEMM: one K=32 group per step, BF16 ``tl.dot`` (K=16 MFMA).

    Do not use ``tl.dot_scaled`` on e4m3: Triton lowers that to the missing
    ``v_mfma_f32_16x16x32_bf16`` on CDNA3. Stepping by 32 keeps each scale a
    single broadcast, avoiding a broken multi-group expand.
    """
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    groups: tl.constexpr = K // 32
    ks = tl.arange(0, 32)
    rows = row[:, None] < M
    cols = col < N
    bs_row = col // B_GROUP_N

    acc = tl.zeros((BM, BN), tl.float32)
    for g in range(0, groups):
        offs = g * 32 + ks
        a = tl.load(
            A + row[:, None] * K + offs[None, :], rows & (offs[None, :] < K), other=0.0
        ).to(tl.float32)
        b = tl.load(
            B + col[:, None] * K + offs[None, :],
            cols[:, None] & (offs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        a_code = tl.load(AS + row * groups + g, row < M, other=127)
        b_code = tl.load(BS + bs_row * groups + g, col < N, other=127)
        a_bf16 = (a * _e8m0_to_f32(a_code)[:, None]).to(tl.bfloat16)
        b_bf16 = (b * _e8m0_to_f32(b_code)[:, None]).to(tl.bfloat16)
        acc = tl.dot(a_bf16, tl.trans(b_bf16), acc=acc)
    tl.store(C + row[:, None] * N + col[None, :], acc, rows & cols[None, :])


@triton.jit(do_not_specialize=["M"])
def blockscale_batched_gemm_fp8_bf16_dequant_kernel(
    A,
    B,
    AS,
    BS,
    C,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    B_GROUP_N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    stride_ag,
    stride_bg,
    stride_asg,
    stride_bsg,
    stride_cg,
):
    """Batched gfx942 group32 GEMM: one CTA grid covers every LoRA group."""
    batch = tl.program_id(0)
    row = tl.program_id(1) * BM + tl.arange(0, BM)
    col = tl.program_id(2) * BN + tl.arange(0, BN)
    groups: tl.constexpr = K // 32
    ks = tl.arange(0, 32)
    rows = row[:, None] < M
    cols = col < N
    bs_row = col // B_GROUP_N
    A_b = A + batch * stride_ag
    B_b = B + batch * stride_bg
    AS_b = AS + batch * stride_asg
    BS_b = BS + batch * stride_bsg
    C_b = C + batch * stride_cg

    acc = tl.zeros((BM, BN), tl.float32)
    for g in range(0, groups):
        offs = g * 32 + ks
        a = tl.load(
            A_b + row[:, None] * K + offs[None, :],
            rows & (offs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        b = tl.load(
            B_b + col[:, None] * K + offs[None, :],
            cols[:, None] & (offs[None, :] < K),
            other=0.0,
        ).to(tl.float32)
        a_code = tl.load(AS_b + row * groups + g, row < M, other=127)
        b_code = tl.load(BS_b + bs_row * groups + g, col < N, other=127)
        a_bf16 = (a * _e8m0_to_f32(a_code)[:, None]).to(tl.bfloat16)
        b_bf16 = (b * _e8m0_to_f32(b_code)[:, None]).to(tl.bfloat16)
        acc = tl.dot(a_bf16, tl.trans(b_bf16), acc=acc)
    tl.store(C_b + row[:, None] * N + col[None, :], acc, rows & cols[None, :])


def _select_fp8_bf16_tile(m: int, n: int, k: int, cu_num: int):
    """Tiles for the BF16-dequant path (K stepped by 32 inside the kernel)."""
    if m <= 16:
        return 16, 32, 2, 2
    if m <= 64:
        return 32, 32, 2, 2
    if m <= 256 or -(-m // 64) * -(-n // 64) < cu_num:
        return 64, 64, 4, 2
    return 64, 64, 4, 2


def _gemm_fp8_bf16_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    weight_group_rows: int,
    cu_num: int,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=dtype)


def _group32_host_bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    weight_group_rows: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reference / tiny-M path: e8m0 dequant on device then hipBLAS."""
    xs = (x_scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    ws = (weight_scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    xf = x.float().reshape(*x.shape[:-1], -1, 32) * xs[..., None]
    xf = xf.reshape(x.shape).to(torch.bfloat16)
    n, k = weight.shape
    wf = weight.float().reshape(n, -1, 32)
    if weight_group_rows == 32:
        ws = ws.repeat_interleave(32, dim=0)
    elif weight_group_rows != 1:
        raise ValueError(f"Unsupported weight_group_rows={weight_group_rows}")
    wf = (wf * ws[:, :, None]).reshape(n, k).to(torch.bfloat16)
    return torch.nn.functional.linear(xf, wf).to(dtype)


@torch_compile_guard(mutates_args=[], gen_fake=_gemm_fp8_bf16_fake)
def gemm_fp8_bf16_dequant(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    weight_group_rows: int,
    cu_num: int,
) -> torch.Tensor:
    """In-kernel e8m0 dequant + BF16 GEMM for arches without MXFP8 MFMA."""
    n, k = weight.shape
    m = x.numel() // k
    output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=dtype)
    if m == 0:
        return output
    # hipBLAS wins the decode-width regime; Triton pays for the weight traffic
    # once M is large enough that the MFMA path amortizes.
    # Prefer a weight-side BF16 cache when present (filled once on gfx942 load).
    cached = getattr(weight, "_atom_group32_bf16_cache", None)
    if cached is not None:
        # Decode is a handful of rows: the cast/mul/cast chain is more kernels
        # than math. One dequant kernel plus hipBLAS. Prefill keeps the bulk
        # torch cast, which is faster once M is wide.
        if m <= 32:
            xf = _dequant_group32_act(x, x_scale)
        else:
            xs = (x_scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
            xf = x.float().reshape(*x.shape[:-1], -1, 32) * xs[..., None]
            xf = xf.reshape(x.shape).to(torch.bfloat16)
        return torch.nn.functional.linear(xf, cached).to(dtype)

    # Triton BF16-dequant GEMM is wired above for follow-up tuning; the host
    # path is the correctness baseline on gfx942 until that kernel matches
    # hipBLAS across the prefill M range.
    use_triton = os.environ.get("ATOM_GFX942_GROUP32_TRITON", "0") == "1"
    if use_triton and m > 16:
        bm, bn, warps, stages = _select_fp8_bf16_tile(m, n, k, cu_num)
        blockscale_gemm_fp8_bf16_dequant_kernel[(-(-m // bm), -(-n // bn))](
            x,
            weight,
            x_scale.view(torch.uint8),
            weight_scale.view(torch.uint8),
            output,
            m,
            n,
            k,
            weight_group_rows,
            bm,
            bn,
            num_warps=warps,
            num_stages=stages,
            matrix_instr_nonkdim=16,
        )
        return output
    return _group32_host_bf16_gemm(
        x,
        weight,
        x_scale,
        weight_scale,
        weight_group_rows=weight_group_rows,
        dtype=dtype,
    )


def gemm_fp8_bf16_dequant_batched(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    dtype: torch.dtype,
    weight_group_rows: int = 32,
    cu_num: int | None = None,
) -> torch.Tensor:
    """Batched [G,M,K] x [G,N,K] group32 GEMM for wo_a on gfx942."""
    if x.ndim != 3 or weight.ndim != 3:
        raise ValueError("Batched GEMM expects [G, M, K] activations and weights")
    g, m, k = x.shape
    gn, n, wk = weight.shape
    if g != gn or k != wk:
        raise ValueError("Batch/K mismatch for batched group32 GEMM")
    if cu_num is None:
        cu_num = torch.cuda.get_device_properties(x.device).multi_processor_count
    out = torch.empty((g, m, n), device=x.device, dtype=dtype)
    if m == 0 or g == 0:
        return out
    bm, bn, warps, stages = _select_fp8_bf16_tile(m, n, k, cu_num)
    blockscale_batched_gemm_fp8_bf16_dequant_kernel[
        (g, -(-m // bm), -(-n // bn))
    ](
        x,
        weight,
        x_scale.view(torch.uint8),
        weight_scale.view(torch.uint8),
        out,
        m,
        n,
        k,
        weight_group_rows,
        bm,
        bn,
        x.stride(0),
        weight.stride(0),
        x_scale.stride(0),
        weight_scale.stride(0),
        out.stride(0),
        num_warps=warps,
        num_stages=stages,
        matrix_instr_nonkdim=16,
    )
    return out
