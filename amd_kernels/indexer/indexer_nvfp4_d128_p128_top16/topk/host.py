# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache + launch wrappers of the FlyDSL index top-k (decode and prefill rows)."""

import functools
import math

import torch

from amd_kernels.warm_up import launcher

from . import kernel as _kernel
from . import prefill_kernel as _prefill_kernel
from ..host_stream import current_stream as _current_stream

MAX_SLOTS = 128  # AITER's TOPK_MAX_SLOTS: 8192 blocks


@functools.lru_cache(maxsize=len(_prefill_kernel.PREFILL_STRIPS))
@launcher
def _get_prefill_indices_kernel(strips: int):
    return _prefill_kernel.compile_topk_prefill(strips)


def index_topk_prefill_indices(
    score: torch.Tensor,
    topk_idx: torch.Tensor,
    *,
    max_seq_len: int,
    block_size: int,
    num_valid_pages: torch.Tensor,
    stream=None,
) -> None:
    """The top-16 block ids of ragged prefill rows, without the attend's page table; ``stream``: the launch stream
    (default the current one)."""
    heads, rows, width = score.shape
    assert score.dtype == torch.float32 and score.stride(2) == 1
    assert topk_idx.dtype == torch.int32 and topk_idx.stride(2) == 1
    assert topk_idx.shape == (heads, rows, _kernel.TOPK)
    assert block_size == _kernel.BLK
    assert num_valid_pages.dtype == torch.int32
    assert num_valid_pages.is_contiguous() and num_valid_pages.numel() >= rows
    slots = _pow2_ceil((max_seq_len + block_size * 64 - 1) // (block_size * 64))
    assert slots <= MAX_SLOTS and width >= slots * 64
    if rows == 0:
        return
    tiles = _prefill_kernel.PREFILL_STRIPS
    strips = next((s for s in tiles if max_seq_len <= s * block_size * 64), tiles[-1])
    _run_compiled(
        _get_prefill_indices_kernel(strips),
        score.data_ptr(),
        score.stride(0),
        score.stride(1),
        slots * 64,
        num_valid_pages.data_ptr(),
        1,
        topk_idx.data_ptr(),
        topk_idx.stride(0),
        topk_idx.stride(1),
        rows,
        heads,
        stream if stream is not None else _current_stream(),
    )


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def _run_compiled(exe, *args):
    """The first call compiles and runs (``flyc.compile``); later calls dispatch the cached CompiledFunction."""
    import flydsl.compiler as flyc

    # a handle compiled by a compile-only warm-up (amd_kernels.warm_up) is kept: compiling it again would count as a
    # compile after warm-up
    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(heads_per_group: int, strips: int):
    return _kernel.compile_topk_ragged(heads_per_group, strips)


def _window_strips(max_seq_len: int, block_size: int) -> int:
    """The window in 64-block strips from max_seq_len, not from the padded score width (longer rows continue through
    later windows)."""
    tokens_per_strip = block_size * 64
    strips = (max_seq_len + tokens_per_strip - 1) // tokens_per_strip
    return min(max(1, strips), _kernel.MAX_STRIPS)


def _validate_topk_outputs(score, topk_idx, sparse_bt, sparse_ctx, num_kv_heads):
    """Validate score/output layouts and return the score dimensions."""
    num_heads, total_q, width = score.shape
    assert score.dtype == torch.float32 and score.stride(2) == 1
    assert topk_idx.dtype == torch.int32 and topk_idx.stride(2) == 1
    assert topk_idx.shape == (num_heads, total_q, _kernel.TOPK)
    assert num_kv_heads % num_heads == 0
    rows = total_q * num_kv_heads
    assert sparse_bt.dtype == torch.int32 and sparse_ctx.dtype == torch.int32
    assert sparse_bt.shape == (rows, _kernel.TOPK * _kernel.PPB)
    assert sparse_bt.stride(1) == 1
    assert sparse_ctx.numel() == rows and sparse_ctx.stride(0) == 1
    return num_heads, total_q, width


def _span_bytes(t: torch.Tensor) -> int:
    n = t.numel()
    if n > 0 and t.is_contiguous():
        return n * 4  # the same span, without a Python pass over shape and strides
    return (sum((s - 1) * st for s, st in zip(t.shape, t.stride())) + 1) * 4


# The kernel addresses every tensor through a buffer resource (i32 byte sizes and offsets), so one launch covers less
# than 2 GiB of each; larger score tensors (e.g. 16384 rows x 1M tokens) launch per head and run of rows on views.
SPAN_LIMIT = 1 << 31


def _launch(exe, score, topk, bt, seq, sbt, sctx, slots, nkv, heads, h0, meta,
            stream=None, sizes=None):
    """One launch over ``heads`` x ``score.shape[1]`` rows.

    Args:
        score, topk: views starting at index head ``h0``.
        sbt, sctx: views starting at the launch's first row, all KV heads (the kernel adds ``h0`` to the KV head of
            its rows and page ids).
        meta: (row_req_id, kv_lens, num_valid_pages) of the rows.
        sizes: the five tensors' spans if the caller has them (an eager call is host-bound at small M).
    """
    total_q = score.shape[1]
    if sizes is None:
        sizes = [_span_bytes(t) for t in (score, bt, topk, sbt, sctx)]
    assert max(sizes) < SPAN_LIMIT, sizes
    rid, kvl, nvp = meta
    _run_compiled(
        exe,
        score.data_ptr(),
        seq.data_ptr(),
        bt.data_ptr(),
        sizes[0],
        sizes[1],
        score.stride(0),
        score.stride(1),
        bt.stride(0),
        max(1, seq.numel()),
        slots * 64,
        topk.data_ptr(),
        sbt.data_ptr(),
        sctx.data_ptr(),
        sizes[2],
        sizes[3],
        sizes[4],
        topk.stride(0),
        topk.stride(1),
        sbt.stride(0),
        nkv,
        rid.data_ptr(),
        kvl.data_ptr(),
        nvp.data_ptr(),
        total_q * 4,
        h0,
        total_q,
        heads,
        stream if stream is not None else _current_stream(),
    )


def _dispatch(exe, score, topk, bt, seq, sbt, sctx, slots, nkv, meta, stream=None):
    """All rows in one launch when every tensor spans < SPAN_LIMIT, else per head and per run of rows: views of
    score / topk at (head, rows), of the page table and context rows at the run's first token (all KV heads; the
    launch's head offset places the rows and the page ids), and of the ragged row arrays."""
    heads, total_q, _ = score.shape
    sizes = [_span_bytes(t) for t in (score, bt, topk, sbt, sctx)]
    if max(sizes) < SPAN_LIMIT:
        _launch(exe, score, topk, bt, seq, sbt, sctx, slots, nkv, heads, 0, meta, stream, sizes)
        return
    per_row = 4 * max(score.stride(1), topk.stride(1), nkv * sbt.stride(0))
    run = max(1, (SPAN_LIMIT // 2) // per_row)
    for h in range(heads):
        for r0 in range(0, total_q, run):
            r1 = min(total_q, r0 + run)
            # page-table and context rows of these tokens, every KV head: the kernel picks head h's by h0
            lo, hi = r0 * nkv, r1 * nkv
            _launch(
                exe,
                score[h : h + 1, r0:r1],
                topk[h : h + 1, r0:r1],
                bt,
                seq,
                sbt[lo:hi],
                sctx[lo:hi],
                slots,
                nkv,
                1,
                h,
                tuple(t[r0:r1] for t in meta),
                stream,
            )


def supports_index_topk(
    topk_blocks: int, block_size: int, pages_per_block: int
) -> bool:
    """gfx950 and the shape AITER's top-k is built for (16 blocks of 128 tokens, 8 pages each)."""
    from ..gfx950 import on_gfx950

    return (
        on_gfx950()
        and topk_blocks == _kernel.TOPK
        and block_size == _kernel.BLK
        and pages_per_block == _kernel.PPB
    )


def index_topk_prefill(
    score: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    sparse_bt: torch.Tensor,
    sparse_ctx: torch.Tensor,
    *,
    max_seq_len: int,
    block_size: int,
    num_valid_pages: torch.Tensor,
    row_req_id: torch.Tensor,
    kv_lens: torch.Tensor,
    num_kv_heads: int,
    pages_per_block: int,
    window_strips: int | None = None,
    stream=None,
) -> None:
    """``aiter.ops.msa_attention.pa_sparse_block_topk`` for ragged prefill rows (row n: request ``row_req_id[n]``,
    ``kv_lens[n]`` causal tokens over ``num_valid_pages[n]`` blocks): the same ``topk_idx``, ``sparse_bt`` and
    ``sparse_ctx`` bit for bit.

    Args:
        window_strips: the window (64-block strips) instead of max_seq_len's; any window gives the same output, a
            fixed set bounds the compiled variants.
        stream: the launch stream (default the current one).
    """
    num_heads, total_q, width = _validate_topk_outputs(
        score, topk_idx, sparse_bt, sparse_ctx, num_kv_heads
    )
    assert block_size == _kernel.BLK and pages_per_block == _kernel.PPB
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
    for t in (num_valid_pages, row_req_id, kv_lens):
        assert t.dtype == torch.int32 and t.is_contiguous() and t.numel() >= total_q
    slots = _pow2_ceil(math.ceil(math.ceil(max_seq_len / block_size) / 64))
    assert slots <= MAX_SLOTS and width >= slots * 64
    if total_q == 0:
        return
    _dispatch(
        _get_kernel(
            max(num_kv_heads // num_heads, 1),
            window_strips or _window_strips(max_seq_len, block_size),
        ),
        score,
        topk_idx,
        block_table,
        seq_lens,
        sparse_bt,
        sparse_ctx,
        slots,
        num_kv_heads,
        # whole tensors: a launch reads rows 0..total_q - 1 by pointer, a per-run launch slices its rows
        (row_req_id, kv_lens, num_valid_pages),
        stream,
    )
