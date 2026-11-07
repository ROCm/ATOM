# SPDX-License-Identifier: MIT
"""Quantize the compressed main latent to FP4 and scatter it into the paged
main pool, in one pass.

The packed pool keeps a row as E2M1 data (`dim // 2` bytes) followed by its
group-16 E4M3 scales (`dim // 16` bytes), which `fused_compress_attn` cannot
write itself. Before this, the rotated echo went through `quantize_fp4` into a
temporary, `pack_rows` concatenated the two halves into another, and
`PagedAttentionCache._scatter_rows` resolved each plan row's address in a chain
of torch ops (casts, divides, clamps, a gather, the offsets) before one
scatter: over a dozen launches per FULL layer per step, nearly all over a few
hundred indices. Here each program reads its plan row, resolves its page and
row the way `index_write` does and stores both halves, bit for bit
`quantize_fp4(**MAIN_FP4)`.

Data and scales are taken as two strided `[pages, per_page, width]` views, so
the same kernel writes a pool that keeps them in one row (this one) or in two
separate planes.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_main_fp4_kernel(
    rotated,
    data,
    scales,
    plan,
    block_tables,
    per_page,
    table_stride,
    data_page_stride,
    data_row_stride,
    scale_page_stride,
    scale_row_stride,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUP: tl.constexpr,
):
    """One program per plan row; a sentinel (`plan[:, 1] < 0`) returns.

    The arithmetic is `quantize_fp4_kernel` with `E4M3_SCALE` verbatim -- the
    floor, the scale rounded through E4M3, the division and the asymmetric
    E2M1 midpoints -- so the stored bytes do not depend on which path wrote
    them.
    """
    token = tl.program_id(0)
    batch = tl.load(plan + token * 4 + 1)
    if batch < 0:
        return
    row = tl.load(plan + token * 4 + 2) // RATIO
    page = tl.load(block_tables + batch * table_stride + row // per_page).to(tl.int64)
    row = (row % per_page).to(tl.int64)

    GROUPS: tl.constexpr = HEAD_DIM // GROUP
    g = tl.arange(0, GROUPS)
    offsets = g[:, None] * GROUP + tl.arange(0, GROUP)[None, :]
    x = tl.load(rotated + token.to(tl.int64) * HEAD_DIM + offsets).to(tl.float32)
    amax = tl.max(tl.abs(x), 1)
    scale = (
        (tl.maximum(amax, 6.0 * 2.0**-9) * (1.0 / 6.0)).to(tl.float8e4nv).to(tl.float32)
    )
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
    code = (code | (sign << 3)).reshape(GROUPS, GROUP // 2, 2)
    packed = tl.sum(code.to(tl.int32) << (tl.arange(0, 2)[None, None, :] * 4), 2)

    packed_offsets = g[:, None] * (GROUP // 2) + tl.arange(0, GROUP // 2)[None, :]
    tl.store(
        data + page * data_page_stride + row * data_row_stride + packed_offsets,
        packed.to(tl.uint8),
    )
    tl.store(scales + page * scale_page_stride + row * scale_row_stride + g, scale)


def write_main_rows_fp4(rotated, planes, plan, block_tables, *, ratio):
    """`quantize_fp4(rotated, **MAIN_FP4)` then `_scatter_rows`, in one launch.

    `planes` is `(data, scales)`: `[pages, per_page, dim // 2]` and
    `[pages, per_page, dim // 16]` byte views, unit stride along the last axis;
    their page and row strides are free. `plan` and `block_tables` are the
    step's, as `_scatter_rows` reads them.
    """
    count = plan.shape[0]
    if not count:
        return
    data, scales = planes
    head_dim = rotated.shape[-1]
    group = 16
    if head_dim % group or rotated.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError("main FP4 rows need a float row divisible by 16")
    if (
        data.shape[-1] != head_dim // 2
        or scales.shape[-1] != head_dim // group
        or data.shape[:2] != scales.shape[:2]
        or data.stride(-1) != 1
        or scales.stride(-1) != 1
    ):
        raise ValueError("main FP4 planes must be [pages, per_page, width] byte views")
    _write_main_fp4_kernel[(count,)](
        rotated.contiguous(),
        data.view(torch.uint8),
        scales.view(torch.float8_e4m3fn),
        plan,
        block_tables,
        data.shape[1],
        block_tables.stride(0),
        data.stride(0),
        data.stride(1),
        scales.stride(0),
        scales.stride(1),
        RATIO=ratio,
        HEAD_DIM=head_dim,
        GROUP=group,
        enable_fp_fusion=False,
    )
