# SPDX-License-Identifier: MIT
"""The UVA gather: this rank's hash heads, read from host memory by the device.

The page-locking, and the host gather this path replaces, stay in `..tables`:
that module is what the prefetch runtime imports, and it has to keep importing
on a machine with no Triton.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _uva_lookup_kernel(
    weight,
    scales,
    ids,
    out,
    num_rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """Gather this rank's hash heads from a host table over UVA, dequantize, store.

    `weight`/`scales` address page-locked HOST memory holding only this rank's
    shard, so a row is addressed by `index - vocab_start`. A head this rank does
    not own writes zeros, which is what makes the all-gather that follows a plain
    concatenation. The ue8m0 scale byte IS an fp32 exponent field, so its decode
    is a shift.

    The grid is persistent and sized to the device rather than to the batch: the
    table dwarfs any TLB, so the win is in reusing warmed translations, not in
    one program per row.
    """
    cols = tl.arange(0, DIM)
    for base in tl.range(
        tl.program_id(0) * BLOCK_R, num_rows, tl.num_programs(0) * BLOCK_R
    ):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        ).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(
                scales
                + local[:, None] * (DIM // QUANT_BLOCK)
                + (cols // QUANT_BLOCK)[None, :],
                mask=owned[:, None],
                other=0,
            )
            values = values * (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(out + rows[:, None] * DIM + cols[None, :], values, mask=valid[:, None])


def uva_gather_into(
    table,
    ids: torch.Tensor,
    out: torch.Tensor,
    *,
    head_start: int,
    local_heads: int,
    total_heads: int,
) -> None:
    """Fill `out` ([tokens, local_heads, head_dim], device) from `ids`.

    `ids` is the FULL `[tokens, total_heads]` index matrix -- every rank sees
    every index and skips the ones outside its shard, so no index exchange is
    needed. Heads this rank does not own are left as zeros for the caller's
    all-gather. Requires `table.enable_uva()`.
    """
    tokens = ids.shape[0]
    num_rows = tokens * local_heads
    if num_rows == 0:
        return
    if out.shape != (tokens, local_heads, table.head_dim):
        raise ValueError(
            f"engram UVA output is {tuple(out.shape)}, expected "
            f"{(tokens, local_heads, table.head_dim)}"
        )
    weight, scales, row_start, row_end = table.registered_shard()
    num_sms = torch.cuda.get_device_properties(out.device).multi_processor_count
    block = 16
    grid = min(-(-num_rows // block), num_sms)
    _uva_lookup_kernel[(grid,)](
        weight,
        scales,
        ids,
        out,
        num_rows,
        row_start,
        row_end,
        ids.stride(0),
        HEAD_START=head_start,
        LOCAL_HEADS=local_heads,
        TOTAL_HEADS=total_heads,
        DIM=table.head_dim,
        QUANT_BLOCK=table.block_size or 1,
        BLOCK_R=block,
        HAS_SCALE=table.quantized,
    )
