# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused DSpark Markov block-sampling step: low-rank bias GEMV + argmax.

DSpark drafts a T-token block in one backbone pass, then samples it
left-to-right with a first-order Markov head whose ``V x V`` transition matrix
is factorized as ``W1 @ W2^T`` (``W1, W2 in R^{V x r}``)::

    logits_k = base_logits_k + W1[x_{k-1}] @ W2^T ;   x_k = argmax(logits_k)

Only the argmax and the ``[B, r]`` row the bias came from leave this step, so
nothing needs the ``[B, V]`` biased logits. The unfused spelling materializes
them three times over -- an fp32 copy of the whole ``W2`` table, the ``[B, V]``
bias, and the ``[B, V]`` sum -- and then reads the sum back for the reduction.
This op keeps ``W2`` bf16, adds the base logits in the GEMM epilogue and
reduces to ids in registers, so per block position the only large read is
``W2`` itself, exactly once.

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

A row that is entirely NaN attains no maximum, and stage 2 answers 0 for it
rather than the sentinel it carries. Reachable: the warmup forward drafts
before ``allocate_kv_cache``, reading an fp8 KV pool where ``0x7F`` is NaN.

CUDA-graph safety. Both launches have host-int grids derived from static
shapes, no host sync and no ``.item()``; the ``prev_ids`` dependence is a
data-dependent *address* inside stage 1's W1 gather, never a data-dependent
shape. See ``docs/environment_variables.md``.
"""

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.sampling_scratch import SamplingScratch, take
from atom.model_ops.softmax_tiles import (
    argmax_lowest_id,
    race_key,
    softmax_combine_kernel,
)

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
# The GEMV addresses base, W1 and W2 in int64 (`_markov_logits_tile`), but the
# id and output offsets are still formed in int32 before sign-extending into
# the pointer, so a tensor whose largest element offset does not fit is
# handed to torch instead of silently wrapping.
_MAX_INT32 = 2**31 - 1
# An id block is one of these two; both ends of this op index with it.
_ID_DTYPES = (torch.int32, torch.int64)


@triton.jit
def _markov_logits_tile(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    embed_out_ptr,
    tile,
    offs_row,
    offs_v,
    row_mask,
    v_mask,
    rank,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    vocab_offset,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """``base + W1[prev] @ W2^T`` on one (row tile, V tile), fp32, -inf on
    padded vocab lanes; tile 0 also publishes ``W1[prev]`` to ``embed_out``.

    The GEMV both stage-1 kernels run; each keeps only its own tail. ``base``
    holds the tile's columns from 0, W2 is read from vocab id ``vocab_offset``
    on (a TP rank's shard of the LM head; 0 for the whole vocab). The vocab
    and row offsets are int64: `[max_bs, T, V]` logits and a whole W2 pass
    2**31 elements at serving sizes.
    """
    # Strided: the caller's ids are one column of its `[B, T + 1]` block, so
    # assuming unit stride here reads another position's tokens.
    prev = tl.load(prev_ids_ptr + offs_row * stride_prev, mask=row_mask, other=0)

    acc = tl.zeros((BLOCK_ROW, BLOCK_V), dtype=tl.float32)
    for k in range(tl.cdiv(rank, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < rank
        # W1[x_{k-1}] rows, [BLOCK_ROW, BLOCK_K], gathered here rather than
        # handed in: every tile program reads these same values anyway, so
        # the gather rides along in L2 and costs no launch of its own.
        embed = tl.load(
            w1_ptr + prev[:, None].to(tl.int64) * stride_w1_vocab + offs_k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # One tile publishes the row for the confidence head; the rest hold
        # the same values and would only write them again.
        if tile == 0:
            tl.store(
                embed_out_ptr + offs_row[:, None] * stride_embed_row + offs_k[None, :],
                embed,
                mask=row_mask[:, None] & k_mask[None, :],
            )
        # W2^T tile, [BLOCK_K, BLOCK_V]. W2 is stored [V, r] so the rank axis
        # is the contiguous one: each vocab column is one 2*BLOCK_K-byte run.
        w2 = tl.load(
            w2_ptr
            + (vocab_offset + offs_v[None, :]).to(tl.int64) * stride_w2_vocab
            + offs_k[:, None],
            mask=v_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        # Zeros from the rank mask contribute exact 0.0 terms, so padding the
        # K tile does not perturb the sum.
        acc = tl.dot(embed, w2, acc)

    base = tl.load(
        base_ptr + offs_row[:, None].to(tl.int64) * stride_base_row + offs_v[None, :],
        mask=row_mask[:, None] & v_mask[None, :],
        other=0.0,
    )
    return tl.where(v_mask[None, :], base.to(tl.float32) + acc, float("-inf"))


@triton.jit
def _dspark_markov_argmax_stage1(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    embed_out_ptr,
    part_val_ptr,
    part_idx_ptr,
    num_rows,
    vocab_size,
    rank,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    vocab_offset,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (V tile, row tile): bias GEMV, add base logits, reduce to (max, id).

    ``base_ptr`` holds ``vocab_size`` columns starting at vocab id
    ``vocab_offset`` (a TP rank's shard of the LM head; 0 for the whole
    vocab), so W2 is read from that row on and the ids published are global.
    """
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    vals = _markov_logits_tile(
        base_ptr,
        prev_ids_ptr,
        w1_ptr,
        w2_ptr,
        embed_out_ptr,
        tile,
        offs_row,
        offs_v,
        row_mask,
        v_mask,
        rank,
        stride_base_row,
        stride_prev,
        stride_w1_vocab,
        stride_embed_row,
        stride_w2_vocab,
        vocab_offset,
        BLOCK_ROW,
        BLOCK_V,
        BLOCK_K,
    )
    # `v_mask` also covers the all--inf row: without it a padded lane, which
    # is -inf too, could win the id.
    tile_max, tile_idx = argmax_lowest_id(
        vals, offs_v[None, :], v_mask[None, :], vocab_size
    )
    # Global id; a padded lane's sentinel stays `vocab_size`, which stage 2
    # recognizes before adding the offset back in.
    tile_idx = tl.where(tile_idx < vocab_size, tile_idx + vocab_offset, vocab_size)

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
    stride_out,
    packed_ptr,
    BLOCK_TILE: tl.constexpr,
    PACKED: tl.constexpr,
):
    """One row: pick the winning V tile, lowest id among tiles that tie.

    ``PACKED`` writes ``(max, id)`` as two fp32 to ``packed_ptr`` instead of
    the id to ``out_ptr``: the TP form, whose ranks then compare their bests.
    """
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
    best = tl.max(vals, axis=0)
    # Ids are global, so the sentinel is anything past every tile's id.
    cand = tl.where((vals == best) & tile_mask, idxs, 2147483647)
    winner = tl.min(cand, axis=0)
    if PACKED:
        # An all-NaN row keeps the sentinel; publish id 0 with the NaN, which
        # `torch.argmax` across ranks then treats as the max, as it would the
        # whole row's.
        winner = tl.where(winner == 2147483647, 0, winner)
        # ids < 2**24 are exact in fp32 (V4.1's vocab is 129280).
        tl.store(packed_ptr + row * 2, best)
        tl.store(packed_ptr + row * 2 + 1, winner.to(tl.float32))
        return
    # NaN never equals itself, so no lane passes its `== max` test and every
    # candidate keeps the out-of-range sentinel. Answer 0, as `torch.argmax`
    # does, so a caller gathering `W1[x]` cannot read past the table.
    winner = tl.where(winner == 2147483647, 0, winner)
    # Strided and in the destination's own dtype: the caller's `[B, T + 1]`
    # block is int32, and the copy this replaces was narrowing as well as
    # moving -- doing one and not the other would just relocate the cast.
    tl.store(out_ptr + row * stride_out, winner.to(out_ptr.dtype.element_ty))


@triton.jit
def _argmax_gather_kernel(
    pairs_ptr,
    out_ptr,
    num_rows,
    stride_out,
    NUM_J: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """``out[i] = ids[argmax_j vals[j, i], i]`` over ``pairs`` ``[J, N, 2]``
    fp32 ``(val, id)``: ``torch.argmax`` over ``j``, the gather of the ids at
    the winner and the copy into a strided, integer ``out`` in one launch.

    The same pick as ``torch.argmax``: the lowest ``j`` among equal values,
    and the first NaN if any (``torch.argmax`` takes NaN as the max).
    """
    rows = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    mask = rows < num_rows
    best = tl.load(pairs_ptr + rows * 2, mask=mask, other=0.0)
    best_id = tl.load(pairs_ptr + rows * 2 + 1, mask=mask, other=0.0)
    best_nan = best != best  # noqa: PLR0124 -- NaN test in Triton
    for j in tl.static_range(1, NUM_J):
        base = pairs_ptr + (j * num_rows + rows) * 2
        val = tl.load(base, mask=mask, other=0.0)
        idx = tl.load(base + 1, mask=mask, other=0.0)
        val_nan = val != val  # noqa: PLR0124
        # strict `>`: an equal value keeps the lower j
        take = (~best_nan) & (val_nan | (val > best))
        best = tl.where(take, val, best)
        best_id = tl.where(take, idx, best_id)
        best_nan = best_nan | val_nan
    tl.store(
        out_ptr + rows * stride_out,
        best_id.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def argmax_gather(pairs: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """``out[i] = pairs[argmax_j pairs[j, i, 0], i, 1]`` for fp32 ``pairs``
    ``[J, N, 2]`` of ``(value, id)``, written into integer ``out`` ``[N]``
    (strided is fine); see ``_argmax_gather_kernel``."""
    num_j, num_rows = pairs.shape[0], pairs.shape[1]
    if pairs.shape != (num_j, num_rows, 2) or pairs.dtype != torch.float32:
        raise ValueError(
            f"pairs {tuple(pairs.shape)} {pairs.dtype} is not [J, N, 2] fp32"
        )
    if out.shape != (num_rows,):
        raise ValueError(f"out {tuple(out.shape)} is not [{num_rows}]")
    if num_rows:
        pairs = pairs if pairs.is_contiguous() else pairs.contiguous()
        block_row = min(1024, triton.next_power_of_2(num_rows))
        _argmax_gather_kernel[(triton.cdiv(num_rows, block_row),)](
            pairs,
            out,
            num_rows,
            out.stride(0),
            NUM_J=num_j,
            BLOCK_ROW=block_row,
            num_warps=1 if block_row <= 64 else 4,
        )
    return out


def _dspark_markov_argmax_fake(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        base_logits.shape[0],
        markov_w1.shape[1],
        dtype=markov_w1.dtype,
        device=base_logits.device,
    )


def _torch_dspark_markov_argmax(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """The reference this op replaces, kept as the out-of-range fallback."""
    markov_embed = markov_w1[prev_ids]
    bias = torch.matmul(markov_embed.float(), markov_w2.float().t())
    out.copy_((base_logits + bias).argmax(dim=-1))
    return markov_embed


def _fused_is_supported(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    out: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
) -> bool:
    num_rows, vocab_size = base_logits.shape
    rank = markov_w2.shape[1]
    if not base_logits.is_cuda:
        return False
    # tl.dot wants one dtype for both operands, and bf16/fp16 are the only ones
    # whose products are exactly representable in the fp32 accumulator -- which
    # is the whole numerics argument above.
    if markov_w1.dtype is not markov_w2.dtype:
        return False
    if markov_w1.dtype not in (torch.bfloat16, torch.float16):
        return False
    # The kernels index the vocab axis with stride 1 and the rank axis with
    # stride 1; anything else would need a transposed load path.
    if base_logits.stride(1) != 1 or markov_w1.stride(1) != 1:
        return False
    if markov_w2.stride(1) != 1:
        return False
    largest_offset = max(
        (num_rows - 1) * base_logits.stride(0) + vocab_size,
        (num_rows - 1) * prev_ids.stride(0) + 1,
        (num_rows - 1) * out.stride(0) + 1,
        (vocab_size - 1) * markov_w1.stride(0) + rank,
        (vocab_size - 1) * markov_w2.stride(0) + rank,
    )
    return largest_offset <= _MAX_INT32


@torch_compile_guard(gen_fake=_dspark_markov_argmax_fake, mutates_args=["out"])
def dspark_markov_argmax(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out[b] = argmax_v(base_logits[b, v] + W1[prev_ids[b]] . markov_w2[v])``.

    Matches ``(base_logits + bias).argmax(-1)`` including its lowest-index
    tie-break.

    Args:
        base_logits: [B, V]  this block position's base logits.
        prev_ids:    [B]     ids of the previously sampled token x_{k-1}.
        markov_w1:   [V, r]  the table the bias row is gathered from.
        markov_w2:   [V, r]  the shared projection table.
        out:         [B]     int32 or int64, where the ids land. Required and
                             never handed back: `torch.compile` is free to
                             functionalize this call, and what it returns then
                             is a copy, so the destination is the one place an
                             id can be read from. A strided view is fine, which
                             is how the caller's own block becomes the
                             destination rather than a copy's source.

    Returns:
        ``[B, r]`` ``W1[prev_ids]``. The gather belongs to this op because
        every tile program reads those same rows to form the bias: handing
        them in costs a launch that reads what the kernel is about to read
        anyway.
    """
    if base_logits.dim() != 2 or markov_w1.dim() != 2 or markov_w2.dim() != 2:
        raise ValueError("dspark_markov_argmax expects 2-D base_logits/w1/w2")
    num_rows, vocab_size = base_logits.shape
    rank = markov_w2.shape[1]
    if prev_ids.shape != (num_rows,) or prev_ids.dtype not in _ID_DTYPES:
        raise ValueError(
            f"prev_ids {tuple(prev_ids.shape)} {prev_ids.dtype} is not the "
            f"[{num_rows}] int32 or int64 an id block is made of"
        )
    if markov_w2.shape[0] != vocab_size:
        raise ValueError(
            f"markov_w2 covers {markov_w2.shape[0]} ids but base_logits has "
            f"{vocab_size} columns"
        )
    if markov_w1.shape != (vocab_size, rank):
        raise ValueError(
            f"markov_w1 {tuple(markov_w1.shape)} does not match markov_w2's "
            f"{tuple(markov_w2.shape)}"
        )
    if out.shape != (num_rows,) or out.dtype not in _ID_DTYPES:
        raise ValueError(
            f"out {tuple(out.shape)} {out.dtype} is not the [{num_rows}] int32 "
            "or int64 an id block is made of"
        )
    if num_rows == 0 or not _fused_is_supported(
        base_logits, prev_ids, out, markov_w1, markov_w2
    ):
        return _torch_dspark_markov_argmax(
            base_logits, prev_ids, markov_w1, markov_w2, out
        )
    return _launch(base_logits, prev_ids, markov_w1, markov_w2, out, 0, None)


def _launch(
    base_logits,
    prev_ids,
    markov_w1,
    markov_w2,
    out,
    vocab_offset,
    packed,
    embed_out=None,
):
    """Both stages over ``base_logits``' columns, which are vocab ids
    ``vocab_offset ..``; ids to ``out``, or ``(max, id)`` to ``packed``.
    ``W1[prev_ids]`` goes to ``embed_out`` (rows may be strided) when given."""
    num_rows, vocab_size = base_logits.shape
    rank = markov_w2.shape[1]
    markov_embed = (
        torch.empty(num_rows, rank, dtype=markov_w1.dtype, device=base_logits.device)
        if embed_out is None
        else embed_out
    )
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
    _dspark_markov_argmax_stage1[(num_tiles, triton.cdiv(num_rows, block_row))](
        base_logits,
        prev_ids,
        markov_w1,
        markov_w2,
        markov_embed,
        part_val,
        part_idx,
        num_rows,
        vocab_size,
        rank,
        stride_base_row=base_logits.stride(0),
        stride_prev=prev_ids.stride(0),
        stride_w1_vocab=markov_w1.stride(0),
        stride_embed_row=markov_embed.stride(0),
        stride_w2_vocab=markov_w2.stride(0),
        vocab_offset=vocab_offset,
        BLOCK_ROW=block_row,
        BLOCK_V=_BLOCK_V,
        BLOCK_K=_BLOCK_K,
        num_warps=8 if block_row >= 32 else 4,
        num_stages=2,
    )
    _dspark_markov_argmax_stage2[(num_rows,)](
        part_val,
        part_idx,
        out if packed is None else part_idx,
        num_rows,
        num_tiles,
        vocab_size,
        out.stride(0) if packed is None else 0,
        part_val if packed is None else packed,
        BLOCK_TILE=triton.next_power_of_2(num_tiles),
        PACKED=packed is not None,
        num_warps=4,
        num_stages=1,
    )
    return markov_embed


def dspark_markov_argmax_tp(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
    vocab_start: int,
    tp_group,
    use_custom: bool = True,
    embed_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``dspark_markov_argmax`` over a TP-sharded vocab.

    ``embed_out``, ``[B, r]`` with unit stride along ``r``, receives the
    gathered ``W1[prev_ids]`` rows instead of a fresh tensor: a caller that
    stacks the steps hands in its block's column and skips the stack.

    ``base_logits`` is this rank's ``[B, V / tp]`` slice of the LM head, ids
    ``vocab_start ..``; the Markov tables are whole. Each rank reduces its
    slice to ``(max, id)`` and only those ``[B, 2]`` cross ranks, rather than
    the ``[B, V]`` logits. Bit-identical to the whole-vocab op: each column's
    bias is the same ``r``-term dot product in the same K order whichever tile
    holds it, the base logit is the same bf16 value, and the cross-rank pick
    takes the lowest rank among equal maxima -- the lowest id, as
    ``torch.argmax`` does.
    """
    num_rows = base_logits.shape[0]
    if num_rows == 0:
        return markov_w1.new_empty(0, markov_w1.shape[1])
    packed = torch.empty((num_rows, 2), dtype=torch.float32, device=base_logits.device)
    markov_embed = _launch(
        base_logits,
        prev_ids,
        markov_w1,
        markov_w2,
        out,
        vocab_start,
        packed,
        embed_out,
    )
    gathered = tp_group.all_gather(packed, dim=0, use_custom=use_custom)
    # each rank's (max, id) -> the id of the largest max
    argmax_gather(gathered.view(tp_group.world_size, num_rows, 2), out)
    return markov_embed


# ---------------------------------------------------------------------------
# Sampled variant: the same bias GEMV, but drawing x_k from the top of
# softmax(logits_k / T) and publishing that q.
# ---------------------------------------------------------------------------
#
# This deliberately gives up part of the property the op above exists for: q
# has to leave the kernel for the probability-ratio accept rule to have
# anything to test. It leaves as its top K entries only -- ids and
# probabilities, renormalized over the stored set -- and the draft samples from
# exactly those, so q is 0 for every id not stored and the ratio rule stays
# exact against it. What the truncation costs is q's tail mass, i.e. a little
# acceptance, never correctness. See `envs.ATOM_SPEC_LOSSLESS_SAMPLING` and
# `ATOM_SPEC_LOSSLESS_Q_TOPK` for the trade, and
# `atom/model_ops/rejection_sampler.py` for the consumer.
#
# The store (`sparse_q_store.SparseQStore`) holds, per draft row
# R = q_row_base[b] + position, the stored ids in ascending order (as
# compaction emits them), their q and the slot of the token the draft emitted.
# A row that drafts a deterministic token (greedy, a non-finite or otherwise
# degenerate row) stores the delta on it: n = 1, ids[0] = x, probs[0] = 1.
# A negative base writes nothing -- pad rows, a dummy forward, an align-only
# propose, and rows whose q nobody reads (greedy requests, requests still
# mid-prompt) -- and such a row drafts its logits' argmax, as a greedy row does.
#
# Which K. A top-K by exact value would need a sort or a radix select over V
# per row; instead every id gets a level, its gap below the row max in 1/8
# nats (0 for the max itself), and the cut is made by level: everything below
# the boundary level, then the boundary level's lowest ids until exactly
# n = min(K, the request's top-k, ids within 19 nats of the max) are taken.
# Ids more than 19 nats down (q < 6e-9 of the max) are never stored. The
# set is not the exact top-K -- within one level the lowest ids win rather than
# the largest values -- which, like the level-granular top-p, only moves
# acceptance: the draft samples from what is stored, whatever that is.
#
# Six launches per block position:
#   1. GEMV + base + 1/T into the dense ``s`` scratch, with per-tile max and
#      raw-argmax partials (the greedy op's stage 1, plus the scaling);
#   2. `softmax_combine_kernel`: the row max (and the raw argmax). No 1/Z:
#      levels need only the max, and finalize normalizes over the stored
#      entries itself, so a full-vocab sum would be read by nobody;
#   3. histogram: per (row, V tile), how many ids sit at each level;
#   4. cut: per row, the boundary level and each tile's output slot range;
#   5. compact: per (row, V tile), write the chosen ids and exp(s - max);
#   6. finalize: per row, normalize, run the exponential race over the stored
#      entries, emit x and its slot.
# Levels come from one helper (`_q_level`) in 3 and 5, so the counts the cut
# is made on and the lanes compaction selects cannot disagree. Every reduction
# is a fixed-order tree, with no atomics: TP ranks run this redundantly and
# must store, and draw, bit for bit the same.
#
# Arithmetic of the draw: dividing a categorical by i.i.d. Exp(1) and taking
# the argmax is a multinomial draw (the exponential race / Gumbel-max trick),
# so this and `torch.multinomial` over the stored probabilities are one
# sampler. A greedy row is one whose temperature was at or below `SAMPLER_EPS`
# before the caller clamped it; `greedy_mask` carries that.
#
# The noise is drawn in-kernel from Philox rather than handed in, at the same
# counter per (position, batch row, vocab id) as a full-vocab race would use.
#
# CUDA-graph safety. The seed, row bases and request filters are device
# tensors whose addresses are baked at capture and whose values are refreshed
# outside the recording; the position and noise offset are host ints fixed by
# the block width. No host RNG, no `.item()`, no data-dependent shape -- n
# bounds loops inside the kernels, never a grid.
#
# TP determinism. Every rank must draw the same uniforms or the drafts fork.
# Philox keyed on an SPMD step counter gives that. The key's high word is 0
# here; the accept rule's draws use other high words (`rejection_sampler`), so
# the two can never share a stream whatever their counters do.
#
# Addressing is int64 throughout: `[max_bs, T, V]` logits and a
# `[2 * max_bs * T, K]` store at K = V both pass 2**31 elements at serving
# sizes.

# Stages 3 and 5 tile V identically (the cut hands each tile a slot range).
# 256 bins at 4096 / 4 warps measured fastest for the histogram on gfx950.
_Q_BLOCK_V = 4096
_Q_BINS = 256
# Level = 1 + floor(gap * 8) for a gap in (0, 19] nats: at most 153, so
# everything fits below the last bin, which collects the rest (and NaN/inf).
_Q_LEVELS_PER_NAT = 8.0
_Q_MAX_GAP = 19.0
# Stored entries one finalize iteration covers.
_Q_FINAL_BLOCK = 1024


@triton.jit
def _dspark_markov_sample_stage1(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    temperatures_ptr,
    q_row_base_ptr,
    embed_out_ptr,
    s_ptr,
    part_max_ptr,
    part_amax_val_ptr,
    part_amax_idx_ptr,
    num_rows,
    vocab_size,
    rank,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    stride_s_row,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (V tile, row tile): bias GEMV, add base, scale by 1/T, write ``s``
    into the row's scratch row and publish this tile's max of ``s``, plus the
    unscaled logits' (max, lowest id) a greedy row drafts from.
    """
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    logits = _markov_logits_tile(
        base_ptr,
        prev_ids_ptr,
        w1_ptr,
        w2_ptr,
        embed_out_ptr,
        tile,
        offs_row,
        offs_v,
        row_mask,
        v_mask,
        rank,
        stride_base_row,
        stride_prev,
        stride_w1_vocab,
        stride_embed_row,
        stride_w2_vocab,
        0,
        BLOCK_ROW,
        BLOCK_V,
        BLOCK_K,
    )
    # The greedy op's own (max, lowest id), taken before the division: a
    # near-zero temperature can round distinct logits onto one value.
    raw_max, raw_idx = argmax_lowest_id(
        logits, offs_v[None, :], v_mask[None, :], vocab_size
    )
    tl.store(part_amax_val_ptr + tile * num_rows + offs_row, raw_max, mask=row_mask)
    tl.store(part_amax_idx_ptr + tile * num_rows + offs_row, raw_idx, mask=row_mask)
    # Pre-clamped by the caller to SAMPLER_EPS, so no row divides by zero. A
    # greedy row's scores are only scratch: finalize publishes the exact delta
    # on its argmax instead.
    temperature = tl.load(temperatures_ptr + offs_row, mask=row_mask, other=1.0)
    scaled = logits / temperature[:, None]
    # Only a row with a destination is read back; the rest draft the argmax.
    q_base = tl.load(q_row_base_ptr + offs_row, mask=row_mask, other=-1)
    tl.store(
        s_ptr + offs_row[:, None].to(tl.int64) * stride_s_row + offs_v[None, :],
        scaled,
        mask=(row_mask & (q_base >= 0))[:, None] & v_mask[None, :],
    )

    # Masked lanes are already -inf.
    tile_max = tl.max(scaled, axis=1)
    tl.store(part_max_ptr + tile * num_rows + offs_row, tile_max, mask=row_mask)


@triton.jit
def _q_level(
    s,
    row_max,
    valid,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
    MAX_GAP: tl.constexpr,
):
    """Each score's level: 0 at the row max, ``1 + floor(gap * LEVELS_PER_NAT)``
    for a gap in (0, MAX_GAP] nats, and the last bin for everything else --
    farther down, masked, NaN, or any score of a row whose max is not finite
    (every gap is then NaN or inf). The one definition the histogram and the
    compaction share; they must agree lane for lane.
    """
    gap = row_max - s
    # Truncation is floor here: the gap is >= 0 wherever the level is kept.
    level = 1 + (gap * LEVELS_PER_NAT).to(tl.int32)
    level = tl.where(gap == 0.0, 0, level)
    keep = valid & (gap >= 0.0) & (gap <= MAX_GAP)
    return tl.where(keep, level, NUM_BINS - 1)


@triton.jit
def _dspark_q_histogram(
    s_ptr,
    row_max_ptr,
    q_row_base_ptr,
    greedy_mask_ptr,
    hist_ptr,
    vocab_size,
    stride_s_row,
    num_tiles,
    BLOCK_V: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
    MAX_GAP: tl.constexpr,
):
    """One (row, V tile): how many of the tile's ids sit at each level."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    # Rows that store nothing stop here, and in the cut and compaction too.
    if (tl.load(q_row_base_ptr + row) < 0) | (tl.load(greedy_mask_ptr + row) != 0):
        return
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < vocab_size
    s = tl.load(
        s_ptr + row.to(tl.int64) * stride_s_row + offs_v,
        mask=v_mask,
        other=float("-inf"),
    )
    level = _q_level(
        s, tl.load(row_max_ptr + row), v_mask, NUM_BINS, LEVELS_PER_NAT, MAX_GAP
    )
    counts = tl.histogram(level, NUM_BINS)
    tl.store(
        hist_ptr + (row * num_tiles + tile) * NUM_BINS + tl.arange(0, NUM_BINS),
        counts,
    )


@triton.jit
def _dspark_q_cut(
    hist_ptr,
    q_row_base_ptr,
    greedy_mask_ptr,
    top_k_ptr,
    top_p_ptr,
    tile_off_ptr,
    tile_quota_ptr,
    row_cut_ptr,
    row_n_ptr,
    num_tiles,
    store_width,
    BLOCK_TILE: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
):
    """One row: how many to store, the boundary level, and each tile's slots.

    n = min(store width, the request's top-k, ids within reach of the max),
    then the request's top-p at level granularity. The boundary level is the
    lowest one whose cumulative count reaches n; its ids are taken lowest id
    first, so each tile's share of them (its quota) is what the tiles before
    it leave over, and each tile's first output slot is the running total of
    the tiles before it. Exactly n ids are then stored.
    """
    row = tl.program_id(0)
    if (tl.load(q_row_base_ptr + row) < 0) | (tl.load(greedy_mask_ptr + row) != 0):
        return
    offs_tile = tl.arange(0, BLOCK_TILE)
    tile_mask = offs_tile < num_tiles
    bins = tl.arange(0, NUM_BINS)
    table = tl.load(
        hist_ptr + (row * num_tiles + offs_tile[:, None]) * NUM_BINS + bins[None, :],
        mask=tile_mask[:, None],
        other=0,
    )
    # The last bin holds what can never be stored.
    counts = tl.where(bins < NUM_BINS - 1, tl.sum(table, axis=0), 0)
    cum = tl.cumsum(counts, axis=0)
    top_k = tl.load(top_k_ptr + row)
    limit = tl.where(top_k > 0, tl.minimum(top_k, store_width), store_width)
    n = tl.minimum(limit, tl.sum(counts, axis=0))
    # Top-p over the unfiltered mass, then the smaller of the two cuts: the
    # rule `Sampler`'s aiter kernels apply (an intersection, not top-p over
    # the top-k survivors). Each level's mass is approximated by its count at
    # the level's mid gap; every level the threshold has not been reached
    # before is kept.
    mid_gap = tl.where(bins == 0, 0.0, (bins.to(tl.float32) - 0.5) / LEVELS_PER_NAT)
    mass = counts.to(tl.float32) * tl.exp(-mid_gap)
    mass_before = tl.cumsum(mass, axis=0) - mass
    top_p = tl.load(top_p_ptr + row)
    nucleus = tl.sum(
        tl.where(mass_before < top_p * tl.sum(mass, axis=0), counts, 0), axis=0
    )
    n = tl.where(top_p < 1.0, tl.minimum(n, nucleus), n)

    cut = tl.sum((cum < n).to(tl.int32), axis=0)
    quota = n - tl.sum(tl.where(bins < cut, counts, 0), axis=0)
    tile_below = tl.sum(tl.where(bins[None, :] < cut, table, 0), axis=1)
    # The boundary level's column, read back rather than reduced out of the
    # table: one program per row runs this, so every full-table pass shows.
    tile_edge = tl.load(
        hist_ptr + (row * num_tiles + offs_tile) * NUM_BINS + cut,
        mask=tile_mask,
        other=0,
    )
    edge_before = tl.cumsum(tile_edge, axis=0) - tile_edge
    tile_quota = tl.minimum(tile_edge, tl.maximum(quota - edge_before, 0))
    tile_n = tile_below + tile_quota
    tile_off = tl.cumsum(tile_n, axis=0) - tile_n
    tl.store(tile_off_ptr + row * num_tiles + offs_tile, tile_off, mask=tile_mask)
    tl.store(tile_quota_ptr + row * num_tiles + offs_tile, tile_quota, mask=tile_mask)
    tl.store(row_cut_ptr + row, cut)
    tl.store(row_n_ptr + row, n)


@triton.jit
def _dspark_q_compact(
    s_ptr,
    row_max_ptr,
    q_row_base_ptr,
    greedy_mask_ptr,
    tile_off_ptr,
    tile_quota_ptr,
    row_cut_ptr,
    q_ids_ptr,
    q_probs_ptr,
    vocab_size,
    stride_s_row,
    num_tiles,
    store_width,
    position,
    BLOCK_V: tl.constexpr,
    NUM_BINS: tl.constexpr,
    LEVELS_PER_NAT: tl.constexpr,
    MAX_GAP: tl.constexpr,
):
    """One (row, V tile): store the tile's chosen ids and their unnormalized
    weights ``exp(s - max)`` at its slot range, in id order."""
    row = tl.program_id(0)
    tile = tl.program_id(1)
    q_base = tl.load(q_row_base_ptr + row)
    if (q_base < 0) | (tl.load(greedy_mask_ptr + row) != 0):
        return
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < vocab_size
    s = tl.load(
        s_ptr + row.to(tl.int64) * stride_s_row + offs_v,
        mask=v_mask,
        other=float("-inf"),
    )
    row_max = tl.load(row_max_ptr + row)
    level = _q_level(s, row_max, v_mask, NUM_BINS, LEVELS_PER_NAT, MAX_GAP)
    cut = tl.load(row_cut_ptr + row)
    quota = tl.load(tile_quota_ptr + row * num_tiles + tile)
    edge = level == cut
    take = (level < cut) | (edge & (tl.cumsum(edge.to(tl.int32), axis=0) <= quota))
    slot = tl.load(tile_off_ptr + row * num_tiles + tile) + (
        tl.cumsum(take.to(tl.int32), axis=0) - 1
    )
    dest = (q_base + position).to(tl.int64) * store_width + slot
    tl.store(q_ids_ptr + dest, offs_v, mask=take)
    tl.store(q_probs_ptr + dest, tl.exp(s - row_max), mask=take)


@triton.jit
def _dspark_q_finalize(
    q_ids_ptr,
    q_probs_ptr,
    q_meta_ptr,
    q_row_base_ptr,
    greedy_mask_ptr,
    row_argmax_ptr,
    row_n_ptr,
    seed_ptr,
    out_ptr,
    vocab_size,
    store_width,
    position,
    noise_offset,
    stride_out,
    BLOCK: tl.constexpr,
):
    """One row: normalize the stored weights into q, draw x from them, record
    x's slot.

    ``probs.div_(noise).argmax()`` over the stored entries, lowest id on a tie.
    A greedy row, or one with nothing stored or no positive race key, drafts
    its logits' lowest-id raw argmax and, if it has a destination, stores the
    exact delta on it: the id is not a draw from any softmax, and the accept
    rule must see the distribution it actually came from.
    """
    row = tl.program_id(0)
    q_base = tl.load(q_row_base_ptr + row)
    has_q = q_base >= 0
    sampled = has_q & (tl.load(greedy_mask_ptr + row) == 0)
    n = tl.where(sampled, tl.load(row_n_ptr + row), 0)
    base = (tl.where(has_q, q_base, 0) + position).to(tl.int64) * store_width
    offs = tl.arange(0, BLOCK)

    # One fixed-order sum, so every rank normalizes by the same number.
    total = 0.0
    for start in range(0, n, BLOCK):
        mask = start + offs < n
        total += tl.sum(
            tl.load(q_probs_ptr + base + start + offs, mask=mask, other=0.0), axis=0
        )
    inv_total = 1.0 / total

    # Philox, keyed on a device scalar every rank fills with the same SPMD step
    # counter, counter = the vocab id's flat index: a given (step, position,
    # row, id) draws one uniform, the same one on every rank.
    seed = tl.load(seed_ptr)
    noise_base = noise_offset + row.to(tl.int64) * vocab_size
    best = 0.0
    best_id = 2147483647
    best_slot = 0
    for start in range(0, n, BLOCK):
        slots = start + offs
        mask = slots < n
        probs = tl.load(q_probs_ptr + base + slots, mask=mask, other=0.0) * inv_total
        tl.store(q_probs_ptr + base + slots, probs, mask=mask)
        ids = tl.load(q_ids_ptr + base + slots, mask=mask, other=0)
        # The verify's race and tie rule (`softmax_tiles`). Stored ids are in
        # slot order, not id order, so chunks merge on the id below.
        key = race_key(probs, seed, noise_base, ids, mask)
        chunk_best, chunk_id = argmax_lowest_id(key, ids, mask, 2147483647)
        chunk_slot = tl.min(
            tl.where(mask & (ids == chunk_id), slots, 2147483647), axis=0
        )
        better = (chunk_best > best) | ((chunk_best == best) & (chunk_id < best_id))
        best = tl.where(better, chunk_best, best)
        best_id = tl.where(better, chunk_id, best_id)
        best_slot = tl.where(better, chunk_slot, best_slot)

    drawn = best > 0.0
    # An all-NaN row's tiles reduce to -inf with stage 1's out-of-range id
    # (tl.max skips NaN, so no lane equals it), and the combine keeps that id.
    # Answer 0, in range, as the greedy op does for such a row.
    argmax = tl.load(row_argmax_ptr + row)
    x = tl.where(drawn, best_id, tl.where(argmax < vocab_size, argmax, 0))
    tl.store(q_ids_ptr + base, x, mask=has_q & ~drawn)
    tl.store(q_probs_ptr + base, 1.0, mask=has_q & ~drawn)
    tl.store(q_meta_ptr + (q_base + position).to(tl.int64) * 2, n, mask=has_q & drawn)
    tl.store(q_meta_ptr + (q_base + position).to(tl.int64) * 2, 1, mask=has_q & ~drawn)
    tl.store(
        q_meta_ptr + (q_base + position).to(tl.int64) * 2 + 1,
        tl.where(drawn, best_slot, 0),
        mask=has_q,
    )
    tl.store(out_ptr + row * stride_out, x.to(out_ptr.dtype.element_ty))


def _dspark_markov_sample_fake(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
    q_ids: torch.Tensor,
    q_probs: torch.Tensor,
    q_meta: torch.Tensor,
    s_scratch: torch.Tensor,
    q_row_base: torch.Tensor,
    position: int,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    noise_offset: int,
    scratch_f32: torch.Tensor | None = None,
    scratch_i32: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty(
        base_logits.shape[0],
        markov_w1.shape[1],
        dtype=markov_w1.dtype,
        device=base_logits.device,
    )


def dspark_markov_sample_unsupported(
    markov_w1: torch.Tensor, markov_w2: torch.Tensor
) -> str | None:
    """Why the fused sampled op cannot run these weights, or None if it can.

    Checked once, when lossless sampling is armed, so an unsupported head
    keeps the greedy pair rather than failing on the first sampled step.
    """
    if not markov_w1.is_cuda:
        return "the markov head is not on a GPU"
    # tl.dot wants one dtype for both operands, and bf16/fp16 are the only ones
    # whose products are exactly representable in the fp32 accumulator.
    if markov_w1.dtype is not markov_w2.dtype or markov_w1.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return f"markov weights are {markov_w1.dtype}/{markov_w2.dtype}"
    if markov_w1.stride(1) != 1 or markov_w2.stride(1) != 1:
        return "markov weights are not contiguous along the rank axis"
    return None


def _sample_scratch_shapes(num_rows: int, vocab_size: int):
    """fp32 and int32 shapes one `dspark_markov_sample` call needs, in order."""
    gemv_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    q_tiles = triton.cdiv(vocab_size, _Q_BLOCK_V)
    f32 = [
        (gemv_tiles, num_rows),  # part_max
        (gemv_tiles, num_rows),  # part_amax_val
        (num_rows,),  # row_max
    ]
    i32 = [
        (gemv_tiles, num_rows),  # part_amax_idx
        (num_rows,),  # row_argmax
        (num_rows, q_tiles, _Q_BINS),  # hist
        (num_rows, q_tiles),  # tile_off
        (num_rows, q_tiles),  # tile_quota
        (num_rows,),  # row_cut
        (num_rows,),  # row_n
    ]
    return f32, i32


def dspark_markov_sample_scratch(max_rows: int, vocab_size: int, device):
    """Scratch for any `dspark_markov_sample` call of up to ``max_rows`` rows,
    allocated once. One call runs at a time, so a block's positions share it.
    The dense ``s`` scratch is not part of it: it is the caller's, sized
    ``[max_rows, V]``."""
    f32, i32 = _sample_scratch_shapes(max_rows, vocab_size)
    return SamplingScratch.for_shapes(f32, i32, device)


def _check_rows(name, t, rows, dtype):
    if t.shape != (rows,) or t.dtype is not dtype or t.stride(0) != 1:
        raise ValueError(
            f"{name} {tuple(t.shape)} {t.dtype} is not a contiguous [{rows}] {dtype}"
        )


@torch_compile_guard(
    gen_fake=_dspark_markov_sample_fake,
    mutates_args=["out", "q_ids", "q_probs", "q_meta", "s_scratch"],
)
def dspark_markov_sample(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
    q_ids: torch.Tensor,
    q_probs: torch.Tensor,
    q_meta: torch.Tensor,
    s_scratch: torch.Tensor,
    q_row_base: torch.Tensor,
    position: int,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    noise_offset: int,
    scratch_f32: torch.Tensor | None = None,
    scratch_i32: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw ``x_k`` from the top of ``softmax((base_logits + W1[prev_ids] @
    W2^T) / T)`` and publish exactly the q it was drawn from.

    Args:
        base_logits:  [B, V]  this block position's base logits.
        prev_ids:     [B]     ids of the previously sampled token x_{k-1}.
        markov_w1:    [V, r]  the table the bias row is gathered from.
        markov_w2:    [V, r]  the shared projection table.
        out:          [B]     int32/int64 destination for the ids. A strided
                              view is fine -- that is how the caller's own
                              `[B, T + 1]` block becomes the destination.
        q_ids, q_probs, q_meta:
                              a `SparseQStore`'s three tensors; row
                              ``q_row_base[b] + position`` receives row b's q.
                              Their width K bounds how many ids a row stores.
        s_scratch:    [>=B, V] fp32, the scaled scores between launches.
        q_row_base:   [B]     int32; negative writes nothing.
        position:             host int, this block position.
        temperatures: [B]     fp32, pre-clamped away from 0 by the caller.
        greedy_mask:  [B]     int32, nonzero where the draw is an argmax.
        top_k:        [B]     int32, the request's top-k; 0 is off.
        top_p:        [B]     fp32, the request's top-p; 1.0 is off.
        seed:         [1]     int32 device scalar. Must hold the same value on
                              every TP rank; see the module notes.
        noise_offset:         host int, the Philox counter base for this block
                              position. Static (position * max_B * V), so it
                              does not make the launch data-dependent.
        scratch_f32, scratch_i32: flat buffers the call's partials are carved
                              from (`dspark_markov_sample_scratch`); None
                              allocates per call.

    Returns:
        ``[B, r]`` ``W1[prev_ids]``, for the same reason the greedy op returns it.

    Raises rather than falling back: an eager fallback would draw from the
    global RNG, which TP ranks do not share, and the drafts would fork.
    """
    if base_logits.dim() != 2 or markov_w1.dim() != 2 or markov_w2.dim() != 2:
        raise ValueError("dspark_markov_sample expects 2-D base_logits/w1/w2")
    num_rows, vocab_size = base_logits.shape
    rank = markov_w2.shape[1]
    if prev_ids.shape != (num_rows,) or prev_ids.dtype not in _ID_DTYPES:
        raise ValueError(
            f"prev_ids {tuple(prev_ids.shape)} {prev_ids.dtype} is not the "
            f"[{num_rows}] int32 or int64 an id block is made of"
        )
    if out.shape != (num_rows,) or out.dtype not in _ID_DTYPES:
        raise ValueError(
            f"out {tuple(out.shape)} {out.dtype} is not the [{num_rows}] int32 "
            "or int64 an id block is made of"
        )
    store_rows, store_width = q_ids.shape if q_ids.dim() == 2 else (0, 0)
    if (
        store_width < 1
        or q_ids.dtype is not torch.int32
        or q_probs.shape != q_ids.shape
        or q_probs.dtype is not torch.float32
        or q_meta.shape != (store_rows, 2)
        or q_meta.dtype is not torch.int32
        or not (q_ids.is_contiguous() and q_probs.is_contiguous())
        or not q_meta.is_contiguous()
    ):
        raise ValueError(
            f"q_ids {tuple(q_ids.shape)} {q_ids.dtype}, q_probs "
            f"{tuple(q_probs.shape)} {q_probs.dtype}, q_meta "
            f"{tuple(q_meta.shape)} {q_meta.dtype} are not a contiguous "
            "[N, K] int32 / [N, K] fp32 / [N, 2] int32 SparseQStore"
        )
    if (
        s_scratch.dim() != 2
        or s_scratch.shape[0] < num_rows
        or s_scratch.shape[1] != vocab_size
        or s_scratch.dtype is not torch.float32
        or not s_scratch.is_contiguous()
    ):
        raise ValueError(
            f"s_scratch {tuple(s_scratch.shape)} {s_scratch.dtype} is not a "
            f"contiguous [>= {num_rows}, {vocab_size}] fp32"
        )
    _check_rows("q_row_base", q_row_base, num_rows, torch.int32)
    _check_rows("top_k", top_k, num_rows, torch.int32)
    _check_rows("top_p", top_p, num_rows, torch.float32)
    if base_logits.stride(1) != 1:
        raise ValueError("base_logits rows must be contiguous over the vocab axis")
    why = dspark_markov_sample_unsupported(markov_w1, markov_w2)
    if why is not None:
        raise ValueError(f"dspark_markov_sample cannot run: {why}")
    device = base_logits.device
    markov_embed = torch.empty(num_rows, rank, dtype=markov_w1.dtype, device=device)
    if num_rows == 0:
        return markov_embed
    block_row = min(
        _MAX_BLOCK_ROW, max(_MIN_BLOCK_ROW, triton.next_power_of_2(num_rows))
    )
    row_tiles = triton.cdiv(num_rows, block_row)

    # Stage 1's GEMV keeps the greedy op's narrow V tile (its accumulator is the
    # MFMA one); the q stages tile wide, because they only stream.
    gemv_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    q_tiles = triton.cdiv(vocab_size, _Q_BLOCK_V)
    f32_shapes, i32_shapes = _sample_scratch_shapes(num_rows, vocab_size)
    (
        (part_max, part_amax_val, row_max),
        (part_amax_idx, row_argmax, hist, tile_off, tile_quota, row_cut, row_n),
    ) = take(f32_shapes, i32_shapes, device, scratch_f32, scratch_i32)

    _dspark_markov_sample_stage1[(gemv_tiles, row_tiles)](
        base_logits,
        prev_ids,
        markov_w1,
        markov_w2,
        temperatures,
        q_row_base,
        markov_embed,
        s_scratch,
        part_max,
        part_amax_val,
        part_amax_idx,
        num_rows,
        vocab_size,
        rank,
        stride_base_row=base_logits.stride(0),
        stride_prev=prev_ids.stride(0),
        stride_w1_vocab=markov_w1.stride(0),
        stride_embed_row=markov_embed.stride(0),
        stride_w2_vocab=markov_w2.stride(0),
        stride_s_row=s_scratch.stride(0),
        BLOCK_ROW=block_row,
        BLOCK_V=_BLOCK_V,
        BLOCK_K=_BLOCK_K,
        num_warps=8 if block_row >= 32 else 4,
        num_stages=2,
    )
    softmax_combine_kernel[(num_rows,)](
        part_max,
        None,
        row_max,
        None,
        part_amax_val,
        part_amax_idx,
        row_argmax,
        num_rows,
        gemv_tiles,
        BLOCK_TILE=triton.next_power_of_2(gemv_tiles),
        WITH_ARGMAX=True,
        WITH_INV_Z=False,
        num_warps=4,
        num_stages=1,
    )
    levels = {
        "NUM_BINS": _Q_BINS,
        "LEVELS_PER_NAT": _Q_LEVELS_PER_NAT,
        "MAX_GAP": _Q_MAX_GAP,
    }
    _dspark_q_histogram[(num_rows, q_tiles)](
        s_scratch,
        row_max,
        q_row_base,
        greedy_mask,
        hist,
        vocab_size,
        s_scratch.stride(0),
        q_tiles,
        BLOCK_V=_Q_BLOCK_V,
        **levels,
        num_warps=4,
    )
    _dspark_q_cut[(num_rows,)](
        hist,
        q_row_base,
        greedy_mask,
        top_k,
        top_p,
        tile_off,
        tile_quota,
        row_cut,
        row_n,
        q_tiles,
        store_width,
        BLOCK_TILE=triton.next_power_of_2(q_tiles),
        NUM_BINS=_Q_BINS,
        LEVELS_PER_NAT=_Q_LEVELS_PER_NAT,
        num_warps=8,
    )
    _dspark_q_compact[(num_rows, q_tiles)](
        s_scratch,
        row_max,
        q_row_base,
        greedy_mask,
        tile_off,
        tile_quota,
        row_cut,
        q_ids,
        q_probs,
        vocab_size,
        s_scratch.stride(0),
        q_tiles,
        store_width,
        position,
        BLOCK_V=_Q_BLOCK_V,
        **levels,
        num_warps=4,
    )
    _dspark_q_finalize[(num_rows,)](
        q_ids,
        q_probs,
        q_meta,
        q_row_base,
        greedy_mask,
        row_argmax,
        row_n,
        seed,
        out,
        vocab_size,
        store_width,
        position,
        noise_offset,
        out.stride(0),
        BLOCK=_Q_FINAL_BLOCK,
        num_warps=4,
    )
    return markov_embed
