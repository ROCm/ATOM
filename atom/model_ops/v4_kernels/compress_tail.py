# SPDX-License-Identifier: MIT
"""One launch for the tail of a quantized-pool compress: the compressor ring
update, the quant of the rotated latent, and its scatter into the PAGE rows
the plan names, in the pool's native data and scale planes.

Replaces `update_compressor_states` + `quantize_fp4` (NVFP4: E2M1 values,
group16 E4M3 scales) or `quantize_fp8` (MXFP8: group32 E8M0) + the torch
address arithmetic and the per-plane `swa_scatter_rows` of
`PagedAttentionCache._scatter_rows`. The parts touch disjoint memory (the
ring, the two PAGE planes), so one grid of `plan rows + write rows` programs
does all of it, and the bytes written are the ones the separate kernels
write: the quant is the standalone kernel's arithmetic on the same BF16 echo,
and each plane's address is `_scatter_rows`' formula with Python's floor
division and the scatter's bounds check.

Graph-safe: no host reads, no allocations; the grid is the two plan slices'
fixed lengths.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import (
    AMAX_FLOOR,
    FP8_MAX,
    FP8_TL_DTYPE,
    ceil_pow2_code,
)
from atom.model_ops.v4_kernels.pool_index import row_offset


@triton.jit
def _floor_divmod(a, b: tl.constexpr):
    # Python/torch floor semantics; Triton's integer `//` and `%` truncate.
    q = a // b
    r = a - q * b
    neg = (r != 0) & (r < 0)
    q = tl.where(neg, q - 1, q)
    r = tl.where(neg, r + b, r)
    return q, r


@triton.jit
def _fp4_row(x):
    """`quantize_fp4_kernel` (group16, E4M3 scale), term for term, on one
    row's `[D // 16, 16]` groups: `([D // 16, 8] packed bytes, [D // 16]
    scale bytes)`."""
    G: tl.constexpr = x.shape[0]
    amax = tl.max(tl.abs(x), 1)
    step = (tl.maximum(amax, 6.0 * 2.0**-9) * (1.0 / 6.0)).to(tl.float8e4nv)
    magnitude = tl.abs(x / step.to(tl.float32)[:, None])
    code = (magnitude > 0.25).to(tl.uint8)
    code = tl.where(magnitude >= 0.75, 2, code)
    code = tl.where(magnitude > 1.25, 3, code)
    code = tl.where(magnitude >= 1.75, 4, code)
    code = tl.where(magnitude > 2.5, 5, code)
    code = tl.where(magnitude >= 3.5, 6, code)
    code = tl.where(magnitude > 5.0, 7, code)
    sign = (x.to(tl.uint32, bitcast=True) >> 31).to(tl.uint8)
    code = (code | (sign << 3)).reshape(G, 8, 2)
    packed = tl.sum(code.to(tl.int32) << (tl.arange(0, 2)[None, None, :] * 4), 2)
    # The standalone kernel stores the FP32 scale through an E4M3 pointer:
    # the same second cast, which is what a non-finite group's code comes from.
    code = step.to(tl.float32).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    return packed.to(tl.uint8), code


@triton.jit
def _fp8_row(x):
    """`quantize_fp8_kernel` (group32, E8M0 code), term for term:
    `([D // 32, 32] value bytes, [D // 32] codes)`."""
    amax = tl.maximum(tl.max(tl.abs(x), 1), AMAX_FLOOR)
    code = ceil_pow2_code(amax * (1.0 / FP8_MAX))
    step = (code << 23).to(tl.float32, bitcast=True)
    q = tl.minimum(tl.maximum(x / step[:, None], -FP8_MAX), FP8_MAX).to(FP8_TL_DTYPE)
    return q.to(tl.uint8, bitcast=True), code.to(tl.uint8)


@triton.jit
def _compress_tail_kernel(
    rotated_ptr,
    rotated_stride,
    plan_ptr,  # [C, 4] int32 (ragged, batch, position, window)
    block_tables_ptr,
    bt_stride0,
    bt_stride1,
    data_ptr,  # uint8 data plane, [pages, rows, D / VPB]
    data_page_stride,
    data_row_stride,
    data_last,  # highest row start the data plane holds
    scale_ptr,  # uint8 scale plane, [pages, rows, D / GROUP]
    scale_page_stride,
    scale_row_stride,
    scale_last,
    num_compress,  # C: programs [0, C) scatter, [C, C + W) update the ring
    kv_ptr,
    kv_row_stride,
    score_ptr,
    score_row_stride,
    ape_ptr,
    write_plan_ptr,  # [W, 4] int32
    slots_ptr,
    kv_state_ptr,
    kv_state_slot_stride,
    kv_state_pos_stride,
    score_state_ptr,
    score_state_slot_stride,
    score_state_pos_stride,
    RATIO: tl.constexpr,
    PER_PAGE: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    FP4: tl.constexpr,
    UPDATE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_compress:
        base = plan_ptr + pid * 4
        batch = tl.load(base + 1)
        position = tl.load(base + 2)
        # `swa_scatter_rows` skips a row with no request; the address is
        # `_scatter_rows`' own, clamps included, plane by plane.
        if batch < 0:
            return
        row, _ = _floor_divmod(position, RATIO)
        col, in_page = _floor_divmod(row, PER_PAGE)
        col = tl.maximum(col, 0)
        page = tl.load(
            block_tables_ptr
            + batch.to(tl.int64) * bt_stride0
            + col.to(tl.int64) * bt_stride1
        ).to(tl.int64)
        in_page = in_page.to(tl.int64)
        data_dst = page * data_page_stride + in_page * data_row_stride
        scale_dst = page * scale_page_stride + in_page * scale_row_stride
        # Each plane keeps its own scatter's bounds rule.
        data_ok = (data_dst >= 0) & (data_dst <= data_last)
        scale_ok = (scale_dst >= 0) & (scale_dst <= scale_last)
        if data_ok | scale_ok:
            GROUP: tl.constexpr = 16 if FP4 else 32
            WIDTH: tl.constexpr = 8 if FP4 else 32  # bytes a group's values take
            G: tl.constexpr = D // GROUP
            groups = tl.arange(0, G)
            x = tl.load(
                rotated_ptr
                + pid.to(tl.int64) * rotated_stride
                + groups[:, None] * GROUP
                + tl.arange(0, GROUP)[None, :]
            ).to(tl.float32)
            if FP4:
                values, scales = _fp4_row(x)
            else:
                values, scales = _fp8_row(x)
            tl.store(
                data_ptr
                + data_dst
                + groups[:, None] * WIDTH
                + tl.arange(0, WIDTH)[None, :],
                values,
                mask=data_ok,
            )
            tl.store(scale_ptr + scale_dst + groups, scales, mask=scale_ok)
    elif UPDATE:
        # `_update_compressor_states_kernel`, term for term
        wbase = write_plan_ptr + (pid - num_compress) * 4
        ragged_id = tl.load(wbase + 0)
        batch_id = tl.load(wbase + 1)
        wpos = tl.load(wbase + 2)
        if wpos < 0:
            return
        slot = tl.load(slots_ptr + batch_id)
        ring = wpos % STATE_SIZE
        ring_idx_ape = wpos % RATIO
        d = tl.arange(0, BLOCK_D)
        m = d < D
        kv_v = tl.load(kv_ptr + ragged_id * kv_row_stride + d, mask=m).to(tl.float32)
        sc_v = tl.load(score_ptr + ragged_id * score_row_stride + d, mask=m).to(
            tl.float32
        )
        ape_v = tl.load(ape_ptr + ring_idx_ape * D + d, mask=m).to(tl.float32)
        kv_slot_base = row_offset(slot, kv_state_slot_stride)
        score_slot_base = row_offset(slot, score_state_slot_stride)
        tl.store(
            kv_state_ptr + kv_slot_base + ring * kv_state_pos_stride + d, kv_v, mask=m
        )
        tl.store(
            score_state_ptr + score_slot_base + ring * score_state_pos_stride + d,
            sc_v + ape_v,
            mask=m,
        )


def compress_tail(
    planes,
    block_tables,
    compress_plan,
    rotated,
    ratio,
    *,
    fp4,
    kv=None,
    score=None,
    ape=None,
    kv_state=None,
    score_state=None,
    write_plan=None,
    state_slot_mapping=None,
):
    """`planes = (data, scale)`, each `[pages, rows, width]` uint8 <- the
    NVFP4 (`fp4`) or MXFP8 quant of `rotated` `[C, D]` at the plan's rows;
    with `kv` given, also the compressor ring update
    `update_compressor_states(kv, score, ape, kv_state, score_state,
    write_plan=..., state_slot_mapping=..., ratio=ratio, overlap=False)` would
    do, in the same launch. `rotated=None` (no plan rows) leaves only the
    ring update."""
    data, scale = planes
    group, values_per_byte = (16, 2) if fp4 else (32, 1)
    if rotated is None:
        if kv is None:
            return
        rotated, num_compress = kv, 0
        head_dim = kv.shape[1]
        compress_plan = write_plan
    else:
        num_compress, head_dim = rotated.shape
        if compress_plan.shape != (num_compress, 4):
            raise ValueError("The echo and the compress plan disagree")
        if compress_plan.dtype != torch.int32 or not compress_plan.is_contiguous():
            raise ValueError("The compress plan must be contiguous int32")
        if rotated.stride(-1) != 1:
            raise ValueError("The echo's rows must be dense")
    if head_dim % 32 or data.shape[-1] != head_dim // values_per_byte:
        raise ValueError("The data plane's width is not the head's")
    if scale.shape[-1] != head_dim // group:
        raise ValueError("The scale plane's width is not the head's")
    for plane in planes:
        if plane.dtype != torch.uint8 or plane.dim() != 3 or plane.stride(-1) != 1:
            raise ValueError("Planes must be [pages, rows, width] uint8")
    if data.shape[:2] != scale.shape[:2]:
        raise ValueError("The two planes must hold the same rows")
    per_page = data.shape[1]

    def last(plane):
        return (plane.shape[0] - 1) * plane.stride(0) + (per_page - 1) * plane.stride(1)

    update = kv is not None
    if update:
        state_size = kv_state.shape[1]
        if state_size < ratio or kv_state.shape[2] != head_dim:
            raise ValueError("The compressor ring is shorter than the ratio")
        if kv.shape != score.shape or kv.shape[1] != head_dim:
            raise ValueError("kv and score must share a shape")
        if ape.shape != (ratio, head_dim) or not ape.is_contiguous():
            raise ValueError("ape must be a dense [ratio, head_dim]")
        if write_plan.dim() != 2 or write_plan.shape[1] != 4:
            raise ValueError("The write plan must be [rows, 4]")
        if write_plan.dtype != torch.int32 or state_slot_mapping.dtype != torch.int32:
            raise ValueError("Plans and slots are int32")
        if kv.stride(-1) != 1 or score.stride(-1) != 1:
            raise ValueError("kv and score columns must be dense")
        if kv_state.stride(-1) != 1 or score_state.stride(-1) != 1:
            raise ValueError("Ring columns must be dense")
        num_write = write_plan.shape[0]
    else:
        kv = score = ape = write_plan = state_slot_mapping = rotated
        kv_state = score_state = rotated[None]
        state_size, num_write = ratio, 0
    grid = num_compress + num_write
    if grid == 0:
        return
    _compress_tail_kernel[(grid,)](
        rotated,
        rotated.stride(0),
        compress_plan,
        block_tables,
        block_tables.stride(0),
        block_tables.stride(1),
        data,
        data.stride(0),
        data.stride(1),
        last(data),
        scale,
        scale.stride(0),
        scale.stride(1),
        last(scale),
        num_compress,
        kv,
        kv.stride(0),
        score,
        score.stride(0),
        ape,
        write_plan,
        state_slot_mapping,
        kv_state,
        kv_state.stride(0),
        kv_state.stride(1),
        score_state,
        score_state.stride(0),
        score_state.stride(1),
        RATIO=ratio,
        PER_PAGE=per_page,
        STATE_SIZE=state_size,
        D=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        FP4=fp4,
        UPDATE=update,
        num_warps=4,
        enable_fp_fusion=False,
    )
