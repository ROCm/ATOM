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

from math import prod

import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.sampling_scratch import SamplingScratch, take
from atom.model_ops.softmax_tiles import softmax_combine_kernel, tile_max_sumexp

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
# An id block is one of these two; both ends of this op index with it.
_ID_DTYPES = (torch.int32, torch.int64)


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
            w1_ptr + prev[:, None] * stride_w1_vocab + offs_k[None, :],
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
            + (vocab_offset + offs_v[None, :]) * stride_w2_vocab
            + offs_k[:, None],
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
    tile_max = tl.max(vals, axis=1)
    # `& v_mask` also covers the all--inf row: without it a padded lane, which
    # is -inf too, could win the id.
    cand = tl.where(
        (vals == tile_max[:, None]) & v_mask[None, :], offs_v[None, :], vocab_size
    )
    tile_idx = tl.min(cand, axis=1)
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
# Sampled variant: the same bias GEMV, but drawing x_k ~ softmax(logits_k / T)
# and publishing that q.
# ---------------------------------------------------------------------------
#
# This deliberately gives up the property the op above exists for. q has to
# leave the kernel for the probability-ratio accept rule to have anything to
# test, and q is `[B, V]` -- a full-vocab fp32 row per request and block
# position, the one thing the module docstring says this is written not to
# materialize. See `envs.ATOM_SPEC_LOSSLESS_SAMPLING` for the trade, and
# `atom/model_ops/rejection_sampler.py` for the consumer.
#
# Arithmetic: `probs = softmax(step_logits.float() / temperatures[:, None])`,
# then `probs.div_(noise).argmax(-1)` with `noise = where(greedy_mask, 1.0,
# exp_noise)` and `exp_noise ~ Exp(1)`. Dividing a categorical by i.i.d. Exp(1)
# and taking the argmax is a multinomial draw (the exponential race / Gumbel-max
# trick), so this and `torch.multinomial` are one sampler. A greedy row is one
# whose temperature was at or below `SAMPLER_EPS` before the caller clamped it;
# `greedy_mask` carries that and the row falls out as a plain argmax.
#
# Where q goes. Each row writes its distribution straight into the persistent
# q store, at row `q_row_base[row] + position`: the block that row's request
# owns. A negative base writes nothing -- pad rows, a dummy forward, an
# align-only propose, and rows whose q nobody reads (greedy requests, requests
# still mid-prompt) -- and such a row drafts its logits' argmax, as a greedy
# row does.
#
# The noise is drawn in-kernel from Philox rather than handed in as a `[B, V]`
# tensor, and the softmax is online (per-tile max and sumexp combined in a
# second launch, `softmax_tiles`), so V is read once per block position.
#
# CUDA-graph safety. The seed and the row bases are device tensors whose
# addresses are baked at capture and whose values are refreshed outside the
# recording; the position and noise offset are host ints fixed by the block
# width. No host RNG, no `.item()`, no data-dependent shape.
#
# TP determinism. Every rank must draw the same uniforms or the drafts fork.
# Philox keyed on an SPMD step counter gives that. The key's high word is 0
# here; the accept rule's draws use other high words (`rejection_sampler`), so
# the two can never share a stream whatever their counters do.
#
# Addressing is int64 throughout: `[max_bs, T, V]` logits and a
# `[2 * max_bs * T, V]` store both pass 2**31 elements at serving sizes.

# Bandwidth-bound over V rather than MFMA-bound, so a wider V tile than the
# GEMV's.
_SAMPLE_BLOCK_V = 1024


@triton.jit
def _dspark_markov_sample_stage1(
    base_ptr,
    prev_ids_ptr,
    w1_ptr,
    w2_ptr,
    temperatures_ptr,
    q_row_base_ptr,
    embed_out_ptr,
    q_ptr,
    part_max_ptr,
    part_sum_ptr,
    part_amax_val_ptr,
    part_amax_idx_ptr,
    num_rows,
    vocab_size,
    rank,
    position,
    stride_base_row,
    stride_prev,
    stride_w1_vocab,
    stride_embed_row,
    stride_w2_vocab,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (V tile, row tile): bias GEMV, add base, scale by 1/T, write ``s``
    into the row's q row and publish this tile's (max, sum exp(s - max)), plus
    the unscaled logits' (max, lowest id) a greedy row drafts from.

    ``s`` lands in q itself, which stage 3 then normalizes in place: the
    alternative is a second `[B, V]` scratch for no gain.
    """
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    prev = tl.load(prev_ids_ptr + offs_row * stride_prev, mask=row_mask, other=0)

    acc = tl.zeros((BLOCK_ROW, BLOCK_V), dtype=tl.float32)
    for k in range(tl.cdiv(rank, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < rank
        embed = tl.load(
            w1_ptr + prev[:, None].to(tl.int64) * stride_w1_vocab + offs_k[None, :],
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        if tile == 0:
            tl.store(
                embed_out_ptr + offs_row[:, None] * stride_embed_row + offs_k[None, :],
                embed,
                mask=row_mask[:, None] & k_mask[None, :],
            )
        w2 = tl.load(
            w2_ptr + offs_v[None, :].to(tl.int64) * stride_w2_vocab + offs_k[:, None],
            mask=v_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        acc = tl.dot(embed, w2, acc)

    base = tl.load(
        base_ptr + offs_row[:, None].to(tl.int64) * stride_base_row + offs_v[None, :],
        mask=row_mask[:, None] & v_mask[None, :],
        other=0.0,
    )
    # Pre-clamped by the caller to SAMPLER_EPS, so no row divides by zero. A
    # greedy row's scores are only scratch: stage 3 publishes the exact delta
    # on its argmax instead.
    logits = tl.where(v_mask[None, :], base.to(tl.float32) + acc, float("-inf"))
    # The greedy op's own (max, lowest id), taken before the division: a
    # near-zero temperature can round distinct logits onto one value.
    raw_max = tl.max(logits, axis=1)
    raw_idx = tl.min(
        tl.where(
            (logits == raw_max[:, None]) & v_mask[None, :], offs_v[None, :], vocab_size
        ),
        axis=1,
    )
    tl.store(part_amax_val_ptr + tile * num_rows + offs_row, raw_max, mask=row_mask)
    tl.store(part_amax_idx_ptr + tile * num_rows + offs_row, raw_idx, mask=row_mask)
    temperature = tl.load(temperatures_ptr + offs_row, mask=row_mask, other=1.0)
    scaled = logits / temperature[:, None]
    q_base = tl.load(q_row_base_ptr + offs_row, mask=row_mask, other=-1)
    q_row = (q_base + position).to(tl.int64) * vocab_size
    tl.store(
        q_ptr + q_row[:, None] + offs_v[None, :],
        scaled,
        mask=(row_mask & (q_base >= 0))[:, None] & v_mask[None, :],
    )

    tile_max, tile_sum = tile_max_sumexp(scaled, v_mask[None, :])
    tl.store(part_max_ptr + tile * num_rows + offs_row, tile_max, mask=row_mask)
    tl.store(part_sum_ptr + tile * num_rows + offs_row, tile_sum, mask=row_mask)


@triton.jit
def _dspark_markov_sample_stage3(
    q_ptr,
    q_row_base_ptr,
    row_max_ptr,
    row_inv_z_ptr,
    row_argmax_ptr,
    greedy_mask_ptr,
    seed_ptr,
    part_key_ptr,
    part_idx_ptr,
    num_rows,
    vocab_size,
    position,
    noise_offset,
    BLOCK_V: tl.constexpr,
):
    """One (row, V tile): normalize ``s`` into q in place, then run the
    exponential race over this tile.

    ``probs.div_(noise).argmax(-1)``. A greedy row publishes the exact delta on
    its logits' lowest-id argmax instead of ``softmax(s)``, which on a tie is
    1/n per tied id: the id is the argmax, not a draw from that softmax, and
    the accept rule must see the distribution the id actually came from.

    One row per program, deliberately: this stage only streams, so its live tile
    is ``[1, BLOCK_V]`` of fp32 rather than stage 1's MFMA accumulator. Tiling
    it ``[BLOCK_ROW, BLOCK_V]`` instead puts enough VGPR/lane in flight to
    spill, which costs far more than the extra programs save.
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < vocab_size

    q_base = tl.load(q_row_base_ptr + row)
    has_q = q_base >= 0
    offs = (tl.where(has_q, q_base, 0) + position).to(tl.int64) * vocab_size + offs_v
    scaled = tl.load(q_ptr + offs, mask=v_mask & has_q, other=float("-inf"))
    row_max = tl.load(row_max_ptr + row)
    inv_z = tl.load(row_inv_z_ptr + row)
    greedy = tl.load(greedy_mask_ptr + row) != 0
    # A row with no destination publishes nothing and drafts its argmax, like
    # a greedy one: its drafts are meant to be dropped, but if one ever were
    # verified it would be a sensible block, not a run of id 0.
    delta = greedy | (q_base < 0)
    if delta:
        argmax = tl.load(row_argmax_ptr + row)
        probs = tl.where(v_mask & (offs_v == argmax), 1.0, 0.0)
    else:
        probs = tl.where(v_mask & has_q, tl.exp(scaled - row_max) * inv_z, 0.0)
    tl.store(q_ptr + offs, probs, mask=v_mask & has_q)

    # Philox, keyed on a device scalar every rank fills with the same SPMD step
    # counter. The flat element index is the counter, so a given (step,
    # position, row, id) draws one uniform, the same one on every rank.
    seed = tl.load(seed_ptr)
    u = tl.rand(seed, noise_offset + row.to(tl.int64) * vocab_size + offs_v)
    # `-log(u)` rather than `-log(1 - u)`: tl.rand is [0, 1), so E = -log(u) is
    # in (0, inf] and the key probs / E is always finite (u == 0 gives key 0,
    # which simply loses the race).
    exp_noise = -tl.log(u)
    denom = tl.where(delta, 1.0, exp_noise)
    key = tl.where(v_mask, probs / denom, -1.0)

    tile_best = tl.max(key, axis=0)
    # Lowest index among equal keys, within the tile and then across them, so
    # the greedy rows reproduce `torch.argmax` exactly.
    cand = tl.where((key == tile_best) & v_mask, offs_v, vocab_size)
    tl.store(part_key_ptr + tile * num_rows + row, tile_best)
    tl.store(part_idx_ptr + tile * num_rows + row, tl.min(cand, axis=0))


def _dspark_markov_sample_fake(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
    q_store: torch.Tensor,
    q_row_base: torch.Tensor,
    position: int,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
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


def dspark_markov_sample_reference(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    exp_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The eager spelling of the op, for tests: ``(ids [B], q [B, V])``.

    Takes its ``[B, V]`` Exp(1) noise as an argument rather than drawing it,
    so a test can feed it the same draws the kernel would make, or none.
    """
    markov_embed = markov_w1[prev_ids]
    bias = torch.matmul(markov_embed.float(), markov_w2.float().t())
    q = torch.softmax(
        (base_logits.float() + bias) / temperatures[:, None].float(), dim=-1
    )
    noise = torch.where(greedy_mask.bool()[:, None], 1.0, exp_noise)
    return (q / noise).argmax(dim=-1), q


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
    sample_tiles = triton.cdiv(vocab_size, _SAMPLE_BLOCK_V)
    f32 = [
        (gemv_tiles, num_rows),  # part_max
        (gemv_tiles, num_rows),  # part_sum
        (gemv_tiles, num_rows),  # part_amax_val
        (num_rows,),  # row_max
        (num_rows,),  # row_inv_z
        (sample_tiles, num_rows),  # part_key
    ]
    i32 = [
        (gemv_tiles, num_rows),  # part_amax_idx
        (num_rows,),  # row_argmax
        (sample_tiles, num_rows),  # part_idx
    ]
    return f32, i32


def dspark_markov_sample_scratch(max_rows: int, vocab_size: int, device):
    """Scratch for any `dspark_markov_sample` call of up to ``max_rows`` rows,
    allocated once. One call runs at a time, so a block's positions share it."""
    f32, i32 = _sample_scratch_shapes(max_rows, vocab_size)
    return SamplingScratch(sum(prod(x) for x in f32), sum(prod(x) for x in i32), device)


@torch_compile_guard(
    gen_fake=_dspark_markov_sample_fake, mutates_args=["out", "q_store"]
)
def dspark_markov_sample(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
    q_store: torch.Tensor,
    q_row_base: torch.Tensor,
    position: int,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
    seed: torch.Tensor,
    noise_offset: int,
    scratch_f32: torch.Tensor | None = None,
    scratch_i32: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw ``x_k ~ softmax((base_logits + W1[prev_ids] @ W2^T) / T)`` and
    publish that distribution.

    Args:
        base_logits:  [B, V]  this block position's base logits.
        prev_ids:     [B]     ids of the previously sampled token x_{k-1}.
        markov_w1:    [V, r]  the table the bias row is gathered from.
        markov_w2:    [V, r]  the shared projection table.
        out:          [B]     int32/int64 destination for the ids. A strided
                              view is fine -- that is how the caller's own
                              `[B, T + 1]` block becomes the destination.
        q_store:      [N, V]  fp32 persistent q; row ``q_row_base[b] +
                              position`` receives row b's distribution. Used as
                              scratch for the unnormalized scores first.
        q_row_base:   [B]     int32; negative writes nothing.
        position:             host int, this block position.
        temperatures: [B]     fp32, pre-clamped away from 0 by the caller.
        greedy_mask:  [B]     int32, nonzero where the draw is an argmax.
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
    if (
        q_store.dim() != 2
        or q_store.shape[1] != vocab_size
        or q_store.dtype is not torch.float32
        or not q_store.is_contiguous()
    ):
        raise ValueError(
            f"q_store {tuple(q_store.shape)} {q_store.dtype} is not a contiguous "
            f"[N, {vocab_size}] fp32 store the accept kernel reads q from"
        )
    if (
        q_row_base.shape != (num_rows,)
        or q_row_base.dtype is not torch.int32
        or q_row_base.stride(0) != 1
    ):
        raise ValueError(
            f"q_row_base {tuple(q_row_base.shape)} {q_row_base.dtype} is not a "
            f"contiguous [{num_rows}] int32"
        )
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
    # MFMA one); stage 3's is wide, because it only streams.
    gemv_tiles = triton.cdiv(vocab_size, _BLOCK_V)
    f32_shapes, i32_shapes = _sample_scratch_shapes(num_rows, vocab_size)
    (
        (part_max, part_sum, part_amax_val, row_max, row_inv_z, part_key),
        (part_amax_idx, row_argmax, part_idx),
    ) = take(f32_shapes, i32_shapes, device, scratch_f32, scratch_i32)

    _dspark_markov_sample_stage1[(gemv_tiles, row_tiles)](
        base_logits,
        prev_ids,
        markov_w1,
        markov_w2,
        temperatures,
        q_row_base,
        markov_embed,
        q_store,
        part_max,
        part_sum,
        part_amax_val,
        part_amax_idx,
        num_rows,
        vocab_size,
        rank,
        position,
        stride_base_row=base_logits.stride(0),
        stride_prev=prev_ids.stride(0),
        stride_w1_vocab=markov_w1.stride(0),
        stride_embed_row=markov_embed.stride(0),
        stride_w2_vocab=markov_w2.stride(0),
        BLOCK_ROW=block_row,
        BLOCK_V=_BLOCK_V,
        BLOCK_K=_BLOCK_K,
        num_warps=8 if block_row >= 32 else 4,
        num_stages=2,
    )
    softmax_combine_kernel[(num_rows,)](
        part_max,
        part_sum,
        row_max,
        row_inv_z,
        part_amax_val,
        part_amax_idx,
        row_argmax,
        num_rows,
        gemv_tiles,
        BLOCK_TILE=triton.next_power_of_2(gemv_tiles),
        WITH_ARGMAX=True,
        num_warps=4,
        num_stages=1,
    )

    sample_tiles = triton.cdiv(vocab_size, _SAMPLE_BLOCK_V)
    _dspark_markov_sample_stage3[(num_rows, sample_tiles)](
        q_store,
        q_row_base,
        row_max,
        row_inv_z,
        row_argmax,
        greedy_mask,
        seed,
        part_key,
        part_idx,
        num_rows,
        vocab_size,
        position,
        noise_offset,
        BLOCK_V=_SAMPLE_BLOCK_V,
        num_warps=4,
        num_stages=2,
    )
    # Stage 4 is the greedy op's stage 2: "pick the winning V tile, lowest id
    # among tiles that tie" is the same reduction over a different key. Driven
    # in its non-packed form, where `part_key` stands in for the unused
    # `packed_ptr` that branch never dereferences.
    _dspark_markov_argmax_stage2[(num_rows,)](
        part_key,
        part_idx,
        out,
        num_rows,
        sample_tiles,
        vocab_size,
        out.stride(0),
        part_key,
        BLOCK_TILE=triton.next_power_of_2(sample_tiles),
        PACKED=False,
        num_warps=4,
        num_stages=1,
    )
    return markov_embed
