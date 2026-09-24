# SPDX-License-Identifier: MIT
"""Quantize index keys and scatter them into the paged FP8 plane, in one pass.

The plane's block is `[TILE rows of data][TILE fp32 scales]`, the data
preshuffled in groups of `SHUFFLE` rows by `MFMA_N` columns -- the layout
`fp8_indexer_block_fields` declares, aiter's `indexer_k_quant_and_cache` writes
and `pa_mqa_logits` reads. This writes it from the compression plan and the
request's PAGE table directly, the way DeepSeek-V4's `fused_compress` does,
instead of resolving each row's address in a chain of tensor ops and handing
the kernel a flat slot list.

What that buys is the chain: seven launches over a few thousand indices become
one, and a plan's sentinel rows leave on a branch rather than relying on
`-1 * per_page + (per_page - 1)` landing back on `-1`.
"""

import torch
import triton
import triton.language as tl

MFMA_N = 16
"""Columns one shuffled group interleaves: the MFMA B operand's width.

Not the rows a block holds, nor the rows a group holds. All three were 16 while
16 was the only page `pa_mqa_logits` took, and the addressing below spelled
them with one name."""


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
    SHUFFLE: tl.constexpr,
    MFMA_N: tl.constexpr,
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
    # its own element type: data first, then the scales past all of it. The
    # scales are not shuffled -- the kernel reads them at `row % TILE`.
    base = tile * block_bytes
    tl.store(
        data
        + base
        + (d // MFMA_N) * (SHUFFLE * MFMA_N)
        + (in_tile % SHUFFLE) * MFMA_N
        + (in_tile // SHUFFLE) * SHUFFLE * HEAD_DIM
        + d % MFMA_N,
        stored,
    )
    tl.store(scales + (base + TILE * HEAD_DIM) // 4 + in_tile, scale)


def write_index_rows(
    keys, plane, plan, block_tables, per_page, *, ratio, rows_per_block, scale_fmt
):
    """Write `keys` into `plane` at the addresses `plan` and `block_tables` give.

    `plane` is the owner's whole index region as bytes; a plan row's
    compressed-row index is `position // ratio`, which the kernel derives from
    the row it already loaded. `scale_fmt` is the plane's declared spelling,
    read here rather than at the call site so one place knows what it means.

    `rows_per_block` comes from the geometry, not a constant here: it is the
    same number the scorer passes as `KVBlockSize`, and a second copy would be
    a second place for them to disagree.
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
        rows_per_block * plane.shape[-1],
        RATIO=ratio,
        HEAD_DIM=keys.shape[-1],
        UE8M0=scale_fmt == "ue8m0",
        TILE=rows_per_block,
        # A block shorter than the MFMA tile shuffles in groups of its own
        # length; the kernel assembles one tile from several blocks.
        SHUFFLE=min(rows_per_block, MFMA_N),
        MFMA_N=MFMA_N,
        FP8_MAX=torch.finfo(fp8).max,
    )
