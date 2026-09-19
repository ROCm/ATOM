# SPDX-License-Identifier: MIT
"""Quantize index keys and scatter them into the paged FP8 plane, in one pass.

The plane's block is `[16 rows of data][16 fp32 scales]`, preshuffled for the
MFMA 16x16 tile -- the layout `fp8_indexer_block_fields` declares, aiter's
`indexer_k_quant_and_cache` writes and `pa_mqa_logits` reads. This writes it
from the compression plan and the request's PAGE table directly, the way
DeepSeek-V4's `fused_compress` does, instead of resolving each row's address
in a chain of tensor ops and handing the kernel a flat slot list.

What that buys is the chain: seven launches over a few thousand indices become
one, and a plan's sentinel rows leave on a branch rather than relying on
`-1 * per_page + (per_page - 1)` landing back on `-1`.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.pool_layout.v4_pool_fields import (
    MQA_LOGITS_PRESHUFFLE_ROWS,
)

# The rows one block id names. A plain int for the launch arithmetic, handed
# to the kernel as a constexpr because that is the only way it may read one.
TILE = MQA_LOGITS_PRESHUFFLE_ROWS


@triton.jit
def _write_index_kernel(
    keys,
    data,
    scales,
    plan,
    block_tables,
    per_page,
    table_stride,
    block_bytes,
    RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    UE8M0: tl.constexpr,
    TILE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """One program per plan row: quantize its key and store it at its address.

    `plan` column 1 is the row's request and column 2 its absolute position,
    one 16-byte row so the second is free. A sentinel carries -1 in column 1
    and returns here, the same guard V4's compressor uses and the reason this
    needs no arithmetic that keeps a negative negative.
    """
    token = tl.program_id(0)
    batch = tl.load(plan + token * 4 + 1)
    if batch < 0:
        return

    # Never negative past that guard, so a plain floor divide stands; `RATIO`
    # is constexpr, so ratio 1 compiles it away.
    row = tl.load(plan + token * 4 + 2) // RATIO
    page = tl.load(block_tables + batch * table_stride + row // per_page)
    flat = page.to(tl.int64) * per_page + row % per_page
    tile = flat // TILE
    in_tile = flat % TILE

    d = tl.arange(0, HEAD_DIM)
    key = tl.load(keys + token.to(tl.int64) * HEAD_DIM + d).to(tl.float32)
    # One scale per row, so the row is the quantization block and `amax` is a
    # scalar; the floor keeps a row of zeros from dividing by nothing.
    scale = tl.maximum(tl.max(tl.abs(key)), 1e-4) / FP8_MAX
    if UE8M0:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    # The reciprocal, not a division: aiter's writer and V4's `fused_compress`
    # both scale by `1/scale`, and the two disagree in the last bit.
    stored = tl.clamp(key * (1.0 / scale), -FP8_MAX, FP8_MAX).to(data.dtype.element_ty)

    # Disjoint halves of one buffer, handed in twice because each store wants
    # its own element type: data first, then the scales past all of it.
    base = tile * block_bytes
    tl.store(
        data + base + (d // TILE) * (TILE * TILE) + in_tile * TILE + d % TILE, stored
    )
    tl.store(scales + (base + TILE * HEAD_DIM) // 4 + in_tile, scale)


def write_index_rows(keys, plane, plan, block_tables, per_page, *, ratio, scale_fmt):
    """Write `keys` into `plane` at the addresses `plan` and `block_tables` give.

    `plane` is the owner's whole index region as bytes; a plan row's
    compressed-row index is `position // ratio`, which the kernel derives from
    the row it already loaded. `scale_fmt` is the plane's declared spelling,
    read here rather than at the call site so one place knows what it means.
    """
    count = plan.shape[0]
    if not count:
        return
    # `view`, not `reshape`: a plane that could not be seen flat would come
    # back as a copy, and the kernel would fill the copy.
    flat = plane.view(-1)
    fp8 = torch.float8_e4m3fn
    _write_index_kernel[(count,)](
        keys,
        flat.view(fp8),
        flat.view(torch.float32),
        plan,
        block_tables,
        per_page,
        block_tables.stride(0),
        # The plane's own last axis is one row's share of a block, so the block
        # is read off the tensor rather than rebuilt from the field list.
        TILE * plane.shape[-1],
        RATIO=ratio,
        HEAD_DIM=keys.shape[-1],
        UE8M0=scale_fmt == "ue8m0",
        TILE=TILE,
        FP8_MAX=torch.finfo(fp8).max,
    )
