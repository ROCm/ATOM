# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused DSpark Markov block-sampling step: low-rank bias GEMV + argmax.

DSpark drafts a T-token block in one backbone pass, then samples it
left-to-right with a first-order Markov head whose ``V x V`` transition matrix
is factorized as ``W1 @ W2^T`` (``W1, W2 in R^{V x r}``)::

    logits_k = base_logits_k + W1[x_{k-1}] @ W2^T ;   x_k = argmax(logits_k)

Only the argmax leaves this step, so nothing needs the ``[B, V]`` biased
logits. The unfused spelling materializes them three times over -- an fp32
copy of the whole ``W2`` table, the ``[B, V]`` bias, and the ``[B, V]`` sum --
and then reads the sum back for the reduction. This op keeps ``W2`` bf16, adds
the base logits in the GEMM epilogue and reduces to ids in registers, so per
block position the only large read is ``W2`` itself, exactly once.

Numerics. The reference does the GEMV as an fp32 matmul over an fp32 copy of
bf16 data. Here ``tl.dot`` takes the bf16 operands directly into MFMA with an
fp32 accumulator: a bf16 value is exactly representable in fp32 and the
product of two bf16 values needs 16 mantissa bits, so every product is exact
in fp32 either way and the two paths sum the SAME r terms. What they do not
share is summation order (MFMA's K-blocking vs whatever tiling hipBLASLt
picks), and fp32 addition is not associative -- so this is equal to the
reference *up to accumulation order*, not bit-identical. The fp32
accumulator is what the existing "this bias lands inside the softmax that
decides acceptance" guarantee asks for, and that is preserved; what no argument
settles is whether an argmax anywhere flips on a pair of logits separated by
less than the last-ulp disagreement. That one is answered by measurement rather
than by proof, which is why ``ATOM_DSPARK_FUSED_MARKOV_SAMPLE`` remains a
switch after being turned on by default (see it for the acceptance-rate
numbers).

Tie-breaking matches ``torch.argmax``: lowest index wins. Within a V tile the
lowest index attaining the tile max is taken, and the cross-tile reduce takes
the lowest index among tiles attaining the global max -- tiles being ordered by
vocab id, that is the global lowest index.

NaN. ``torch.argmax`` orders NaN above every number, so a row carrying one
returns that lane rather than the largest finite one, and an all-NaN row
returns 0. The reduce here is written around ``==``, which no NaN lane
satisfies, so both stages carry NaN candidates on the side and prefer them --
without that, an all-NaN row leaves the ``vocab_size`` tie-break sentinel
standing and returns it as the id, which the caller then gathers a ``[V, r]``
embedding row with. A NaN reaching this op means the draft logits are already
garbage and the block will be rejected by the target either way; what matters
is that the id stays inside the table.

CUDA-graph safety. Both launches have host-int grids derived from static
shapes, no host sync and no ``.item()``; the ``prev_ids`` dependence is a
data-dependent *address* inside the W1 gather the caller already does, never a
data-dependent shape. See ``docs/environment_variables.md``.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

# Untuned, and chosen for headroom rather than peak: the stage-1 accumulator is
# [BLOCK_ROW, BLOCK_V] fp32 (64x128 -> 16 VGPR/lane at 8 warps) and one pipeline
# stage stages [BLOCK_ROW, BLOCK_K] + [BLOCK_K, BLOCK_V] bf16 = 24KB of the 64KB
# LDS. Left as-is once these shapes measured 145us/step of savings at B=1 and
# 235us at B=64 on Kimi-K3; a sweep is the obvious next thing if that matters.
_BLOCK_V = 128
_BLOCK_K = 64
# MFMA needs M >= 16, so a batch smaller than that runs padded-and-masked
# rather than falling off the kernel. Capped at 64 because the accumulator is
# [BLOCK_ROW, BLOCK_V] fp32 and lives in registers: batches past the cap take
# more row tiles rather than a wider accumulator, which would spill (native
# ATOM defaults max_num_seqs to 512, so this is reachable, not theoretical).
_MIN_BLOCK_ROW = 16
_MAX_BLOCK_ROW = 64
# Triton forms the offset expressions below in int32 before sign-extending
# into the pointer, so a tensor whose largest element offset does not fit is
# handed to torch instead of silently wrapping.
_MAX_INT32 = 2**31 - 1


@triton.jit
def _dspark_markov_argmax_stage1(
    base_ptr,
    embed_ptr,
    w2_ptr,
    part_val_ptr,
    part_idx_ptr,
    num_rows,
    vocab_size,
    rank,
    stride_base_row,
    stride_embed_row,
    stride_w2_vocab,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (V tile, row tile): bias GEMV, add base logits, reduce to (max, id)."""
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size

    acc = tl.zeros((BLOCK_ROW, BLOCK_V), dtype=tl.float32)
    for k in range(tl.cdiv(rank, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < rank
        # W1[x_{k-1}] rows, [BLOCK_ROW, BLOCK_K]. Tiny and re-read by every
        # tile program, so it lives in L2 after the first wave.
        embed = tl.load(
            embed_ptr + offs_row[:, None] * stride_embed_row + offs_k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # W2^T tile, [BLOCK_K, BLOCK_V]. W2 is stored [V, r] so the rank axis
        # is the contiguous one: each vocab column is one 2*BLOCK_K-byte run.
        w2 = tl.load(
            w2_ptr + offs_v[None, :] * stride_w2_vocab + offs_k[:, None],
            mask=v_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        # Zeros from the rank mask contribute exact 0.0 terms, so padding the
        # K tile does not perturb the sum.
        acc = tl.dot(embed, w2, acc)

    base = tl.load(
        base_ptr + offs_row[:, None] * stride_base_row + offs_v[None, :],
        mask=row_mask[:, None] & v_mask[None, :],
        other=0.0,
    )
    vals = tl.where(v_mask[None, :], base.to(tl.float32) + acc, float("-inf"))
    # NaN lanes are taken out of the max/equality reduce and carried separately.
    # `x == max` is false for every lane of an all-NaN row, so the plain reduce
    # would leave `cand` at the `vocab_size` tie-break sentinel and return it as
    # the id -- one past the table the caller gathers with. torch.argmax instead
    # orders NaN above every number, lowest index winning ties, so that is what
    # the split below reproduces.
    nan_mask = (vals != vals) & v_mask[None, :]  # noqa: PLR0124 - Triton NaN check
    # `vals` reduces directly rather than through a NaN-scrubbed copy. Holding
    # a second [BLOCK_ROW, BLOCK_V] fp32 tile beside the register-resident
    # accumulator costs occupancy: the scrubbed copy measured +14% on the
    # BLOCK_ROW=32 specialisation (36.3 -> 41.5us at V=163840, against a 0.3us
    # spread; the 16 and 64 tiles moved less than their noise). It buys nothing,
    # because either tl.max NaN convention lands somewhere the row-level
    # overwrite below covers -- propagating leaves `tile_max` NaN so no lane
    # matches and `cand` stays at the sentinel, ignoring leaves the largest
    # number and its id -- and a row with a NaN takes `nan_idx`/NaN regardless.
    # A row without one never differed to begin with.
    tile_max = tl.max(vals, axis=1)
    # `& v_mask` also covers the all--inf row: without it a padded lane, which
    # is -inf too, could win the id.
    cand = tl.where(
        (vals == tile_max[:, None]) & v_mask[None, :], offs_v[None, :], vocab_size
    )
    tile_idx = tl.min(cand, axis=1)
    # Lowest NaN id in this tile, or `vocab_size` when the tile has none.
    nan_idx = tl.min(tl.where(nan_mask, offs_v[None, :], vocab_size), axis=1)
    row_has_nan = nan_idx < vocab_size
    # A NaN partial value is the marker stage 2 reads to order this tile above
    # every finite one, mirroring the ordering applied within the tile here.
    tile_max = tl.where(row_has_nan, float("nan"), tile_max)
    tile_idx = tl.where(row_has_nan, nan_idx, tile_idx)

    tl.store(part_val_ptr + tile * num_rows + offs_row, tile_max, mask=row_mask)
    tl.store(part_idx_ptr + tile * num_rows + offs_row, tile_idx, mask=row_mask)


@triton.jit
def _dspark_markov_argmax_stage2(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    num_rows,
    num_tiles,
    vocab_size,
    BLOCK_TILE: tl.constexpr,
):
    """One row: pick the winning V tile, lowest id among tiles that tie."""
    row = tl.program_id(0)
    offs_tile = tl.arange(0, BLOCK_TILE)
    tile_mask = offs_tile < num_tiles
    vals = tl.load(
        part_val_ptr + offs_tile * num_rows + row,
        mask=tile_mask,
        other=float("-inf"),
    )
    idxs = tl.load(
        part_idx_ptr + offs_tile * num_rows + row, mask=tile_mask, other=vocab_size
    )
    # Same NaN ordering as stage 1, now across tiles: a tile that saw a NaN
    # carries a NaN partial value and wins over every finite tile, and tiles
    # being ordered by vocab id the lowest such tile holds the lowest NaN id.
    nan_mask = (vals != vals) & tile_mask  # noqa: PLR0124 - Triton NaN check
    # Reduced over `vals` rather than a scrubbed copy for the reason stage 1
    # gives: `nan_res` below overwrites every row this could differ on.
    best = tl.max(vals, axis=0)
    cand = tl.where((vals == best) & tile_mask, idxs, vocab_size)
    res = tl.min(cand, axis=0)
    nan_res = tl.min(tl.where(nan_mask, idxs, vocab_size), axis=0)
    res = tl.where(nan_res < vocab_size, nan_res, res)
    # Belt and braces: the caller feeds this id straight into a [V, r] embedding
    # gather, where an out-of-range id is an illegal access on a device queue
    # rather than an exception. Nothing above can reach the sentinel any more,
    # so this only bounds a future reduce that regresses the property.
    tl.store(out_ptr + row, tl.minimum(res, vocab_size - 1).to(tl.int64))


def _dspark_markov_argmax_fake(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        base_logits.shape[0], dtype=torch.int64, device=base_logits.device
    )


def _torch_dspark_markov_argmax(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2: torch.Tensor,
) -> torch.Tensor:
    """The reference this op replaces, kept as the out-of-range fallback."""
    bias = torch.matmul(markov_embed.float(), markov_w2.float().t())
    return (base_logits + bias).argmax(dim=-1)


def _fused_is_supported(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2: torch.Tensor,
) -> bool:
    num_rows, vocab_size = base_logits.shape
    rank = markov_w2.shape[1]
    if not base_logits.is_cuda:
        return False
    # tl.dot wants one dtype for both operands, and bf16/fp16 are the only ones
    # whose products are exactly representable in the fp32 accumulator -- which
    # is the whole numerics argument above.
    if markov_embed.dtype is not markov_w2.dtype:
        return False
    if markov_embed.dtype not in (torch.bfloat16, torch.float16):
        return False
    # The kernels index the vocab axis with stride 1 and the rank axis with
    # stride 1; anything else would need a transposed load path.
    if base_logits.stride(1) != 1 or markov_embed.stride(1) != 1:
        return False
    if markov_w2.stride(1) != 1:
        return False
    largest_offset = max(
        (num_rows - 1) * base_logits.stride(0) + vocab_size,
        (vocab_size - 1) * markov_w2.stride(0) + rank,
    )
    return largest_offset <= _MAX_INT32


@torch_compile_guard(gen_fake=_dspark_markov_argmax_fake)
def dspark_markov_argmax(
    base_logits: torch.Tensor,
    markov_embed: torch.Tensor,
    markov_w2: torch.Tensor,
) -> torch.Tensor:
    """``argmax_v(base_logits[b, v] + markov_embed[b] . markov_w2[v])``.

    Args:
        base_logits:  [B, V]     this block position's base logits.
        markov_embed: [B, r]     ``W1[x_{k-1}]``, the previous token's row.
        markov_w2:    [V, r]     the shared projection table.

    Returns:
        [B] int64 ids, matching ``(base_logits + bias).argmax(-1)`` including
        its lowest-index tie-break.
    """
    if base_logits.dim() != 2 or markov_embed.dim() != 2 or markov_w2.dim() != 2:
        raise ValueError("dspark_markov_argmax expects 2-D base_logits/embed/w2")
    num_rows, vocab_size = base_logits.shape
    rank = markov_w2.shape[1]
    if markov_embed.shape != (num_rows, rank):
        raise ValueError(
            f"markov_embed {tuple(markov_embed.shape)} does not match "
            f"[{num_rows}, {rank}] implied by base_logits and markov_w2"
        )
    if markov_w2.shape[0] != vocab_size:
        raise ValueError(
            f"markov_w2 covers {markov_w2.shape[0]} ids but base_logits has "
            f"{vocab_size} columns"
        )
    if num_rows == 0:
        return torch.empty(0, dtype=torch.int64, device=base_logits.device)
    if not _fused_is_supported(base_logits, markov_embed, markov_w2):
        return _torch_dspark_markov_argmax(base_logits, markov_embed, markov_w2)

    num_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    block_row = min(
        _MAX_BLOCK_ROW, max(_MIN_BLOCK_ROW, triton.next_power_of_2(num_rows))
    )
    # Fresh per call rather than a persistent buffer: under CUDA-graph capture
    # these come from the graph's private pool and are freed back into it each
    # block position, so the pool holds one pair of them and replay reuses the
    # recorded addresses. A module-level cache would instead hand graph-private
    # memory to whatever ran next.
    part_val = torch.empty(
        (num_tiles, num_rows), dtype=torch.float32, device=base_logits.device
    )
    part_idx = torch.empty(
        (num_tiles, num_rows), dtype=torch.int32, device=base_logits.device
    )
    out = torch.empty(num_rows, dtype=torch.int64, device=base_logits.device)

    _dspark_markov_argmax_stage1[(num_tiles, triton.cdiv(num_rows, block_row))](
        base_logits,
        markov_embed,
        markov_w2,
        part_val,
        part_idx,
        num_rows,
        vocab_size,
        rank,
        stride_base_row=base_logits.stride(0),
        stride_embed_row=markov_embed.stride(0),
        stride_w2_vocab=markov_w2.stride(0),
        BLOCK_ROW=block_row,
        BLOCK_V=_BLOCK_V,
        BLOCK_K=_BLOCK_K,
        num_warps=8 if block_row >= 32 else 4,
        num_stages=2,
    )
    _dspark_markov_argmax_stage2[(num_rows,)](
        part_val,
        part_idx,
        out,
        num_rows,
        num_tiles,
        vocab_size,
        BLOCK_TILE=triton.next_power_of_2(num_tiles),
        num_warps=4,
        num_stages=1,
    )
    return out
