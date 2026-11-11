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

The sampled lossless draft shares the stage-1 GEMV and its launch
(`dspark_markov_stage1`); it is `dspark_markov_sample.dspark_markov_sample`.

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

from atom.model_ops.dspark_markov_stage1 import (
    BLOCK_K,
    BLOCK_V,
    markov_launch_geometry,
    markov_logits_tile,
    publish_tile_argmax,
)
from atom.model_ops.tile_argmax import argmax_of_tiles

# The GEMV addresses base, W1 and W2 in int64 (`markov_logits_tile`), but the
# id and output offsets are still formed in int32 before sign-extending into
# the pointer, so a tensor whose largest element offset does not fit is
# handed to torch instead of silently wrapping.
_MAX_INT32 = 2**31 - 1
# An id block is one of these two; both ends of this op index with it.
_ID_DTYPES = (torch.int32, torch.int64)


@triton.jit
def _markov_stage1(
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
    """One (V tile, row tile): bias GEMV, add base logits, publish the tile's
    (max, lowest id) of the logits.

    ``base_ptr`` holds ``vocab_size`` columns starting at vocab id
    ``vocab_offset`` (a TP rank's shard of the LM head; 0 for the whole
    vocab), so W2 is read from that row on and the ids published are global.
    """
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    vals = markov_logits_tile(
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
        True,
    )
    publish_tile_argmax(
        vals,
        tile,
        offs_row,
        offs_v,
        row_mask,
        v_mask,
        num_rows,
        vocab_offset,
        part_val_ptr,
        part_idx_ptr,
    )


@triton.jit
def _dspark_markov_argmax_stage2(
    part_val_ptr,
    part_idx_ptr,
    out_ptr,
    num_rows,
    num_tiles,
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
    # NaN never equals itself, so an all-NaN row answers id 0: what
    # `torch.argmax` does, so a caller gathering `W1[x]` cannot read past the
    # table. Packed, the NaN goes out with it, which `torch.argmax` across
    # ranks then treats as the max, as it would the whole row's.
    best, winner = argmax_of_tiles(
        part_val_ptr, part_idx_ptr, row, num_rows, num_tiles, BLOCK_TILE
    )
    if PACKED:
        # ids < 2**24 are exact in fp32 (V4.1's vocab is 129280).
        tl.store(packed_ptr + row * 2, best)
        tl.store(packed_ptr + row * 2 + 1, winner.to(tl.float32))
        return
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


def markov_weights_unsupported(
    markov_w1: torch.Tensor, markov_w2: torch.Tensor
) -> str | None:
    """Why the fused Markov kernels cannot run these weights, or None. The one
    check both ops, and lossless sampling's arming, ask."""
    if not markov_w1.is_cuda:
        return "the markov head is not on a GPU"
    # tl.dot wants one dtype for both operands, and bf16/fp16 are the only ones
    # whose products are exactly representable in the fp32 accumulator -- which
    # is the whole numerics argument above.
    if markov_w1.dtype is not markov_w2.dtype or markov_w1.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return f"markov weights are {markov_w1.dtype}/{markov_w2.dtype}"
    # The kernels index the rank axis with stride 1; anything else would need
    # a transposed load path.
    if markov_w1.stride(1) != 1 or markov_w2.stride(1) != 1:
        return "markov weights are not contiguous along the rank axis"
    return None


def check_markov_inputs(
    op: str,
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Raise ``ValueError`` unless the inputs are a ``[B, V]`` logits block,
    ``[B]`` previous and output ids, and ``[V, r]`` W1 / W2 tables over the
    same V. Both ops index W2 by the logits' column, so a table of another
    vocab would read the wrong rows or past the end."""
    if base_logits.dim() != 2 or markov_w1.dim() != 2 or markov_w2.dim() != 2:
        raise ValueError(f"{op} expects 2-D base_logits/w1/w2")
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


def _fused_is_supported(
    base_logits: torch.Tensor,
    prev_ids: torch.Tensor,
    out: torch.Tensor,
    markov_w1: torch.Tensor,
    markov_w2: torch.Tensor,
) -> bool:
    num_rows = base_logits.shape[0]
    if not base_logits.is_cuda or base_logits.stride(1) != 1:
        return False
    if markov_weights_unsupported(markov_w1, markov_w2) is not None:
        return False
    # base, W1 and W2 are addressed in int64; only these two are not.
    largest_offset = max(
        (num_rows - 1) * prev_ids.stride(0) + 1,
        (num_rows - 1) * out.stride(0) + 1,
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
    check_markov_inputs(
        "dspark_markov_argmax", base_logits, prev_ids, markov_w1, markov_w2, out
    )
    if base_logits.shape[0] == 0 or not _fused_is_supported(
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
    geometry = markov_launch_geometry(num_rows, vocab_size)
    num_tiles = geometry.v_tiles
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
    _markov_stage1[(num_tiles, geometry.row_tiles)](
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
        BLOCK_ROW=geometry.block_row,
        BLOCK_V=BLOCK_V,
        BLOCK_K=BLOCK_K,
        num_warps=geometry.num_warps,
        num_stages=2,
    )
    _dspark_markov_argmax_stage2[(num_rows,)](
        part_val,
        part_idx,
        out if packed is None else part_idx,
        num_rows,
        num_tiles,
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
