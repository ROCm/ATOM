# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP4 scoring for a DSA sparse indexer: the predicate, the ABI, the schedule.

FP4 is a change of dtype, not a second cache: `indexer_qk_rope_quant_and_cache`
writes packed E2M1 keys plus their e8m0 scales in place of the FP8 row, and
`flydsl_pa_mqa_logits_fp4[_prefill]` score them straight out of the paged cache.
Every number here is one those kernels fix, so it drifts with them.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger("atom")

# Persistent-grid schedule params for the `pa_mqa_logits_fp4*` kernels, which
# decode and prefill both score through. A metadata builder precomputes each
# path's cta_info with these and the scorer passes the matching block_k, so
# layout and grid agree. DeepSeek-V4 drives the same two kernels and carries its
# own copy of this pair in `v4_kernels`; the two must be changed together until
# something merges them, which is not this module's to do.
#
# Both floors are CTA-count targets, not kernel defaults: every consumer takes
# `max(floor, rows)`, so a floor only adds split-K to grids too small to fill
# the GPU and is an identity for the wide ones. Splits are numerically inert --
# each CTA gets a disjoint KV-column range, no cross-CTA partial sums.
#
# Neither can be fitted to the work. A CUDAGraph bakes the grid at capture,
# while the work is `rows * ceil(ctx / block_k)` and the context term is only
# known per replay; of what capture does know, the batch size measures flat, so
# there is nothing to derive one from. Both are therefore the smallest
# worst-case regret over the served context range rather than any shape's
# optimum -- re-tune against a real workload mix. At prefill rows=1024 and
# W~32768, 4096 measured 206.2us a call against 512's 224.6us.
FP4_MQA_PARALLEL_UNIT_NUM = 4096
# The varctx scorer takes a lower floor than the prefill one. Its rows are
# sequences rather than query tokens, so `max(floor, rows)` never lifts off the
# floor and every CTA past the work is pure setup: 8.75us a layer at 4096
# against 2.99 at 512, flat in context. Going lower still gives back more at
# 131k+ than it wins at 4k. This is not "the decode floor" -- DeepSeek-V4's
# decode scores through the prefill kernel and keeps 4096.
FP4_MQA_VARCTX_PARALLEL_UNIT_NUM = 512
FP4_MQA_BLOCK_K = 256

# The fused writer's FP4 group width, against 128 on the FP8 path.
FP4_QUANT_BLOCK_SIZE = 32
# `pa_mqa_logits_fp4*` pack four N-tiles' e8m0 bytes into one dword and read
# them with N_PHYS == 1, which holds only where a paged block covers exactly
# NTPW(4) x MFMA_N(16) indexer rows.
FP4_KV_BLOCK_SIZE = 64
_MFMA_M = 16
_K_TILE = 128


def _gfx() -> str:
    """The chip this is running on, imported on use rather than at module scope:
    a non-GPU install carries `aiter` without `aiter.jit`, and the rest of this
    module is ABI constants such an install still has to be able to read."""
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx()


def sparse_indexer_fp4_enabled(
    index_cache_dtype: str | None, config: Any, *, warn: bool = False
) -> bool:
    """Does this layer's DSA indexer score in FP4? The one place that decides.

    Structural, never per-model: the kernels need an indexer whose head dim is a
    single 128-wide K tile and whose head count tiles by MFMA_M, on a chip that
    has them. That also reaches the right verdict for an MTP draft, whose
    `model_type` `SpeculativeConfig._MTP_TYPE_MAP` has rewritten but whose
    indexer geometry -- and cache, shared with the target -- is unchanged.

    The metadata builder and `Indexer.__init__` must agree: the builder picks the
    cache-pool width, but warmup traces the indexer's graph piece before
    `allocate_kv_cache` runs, so a layer that guessed wrong bakes in the other
    branch. Pass `warn=True` from the builder only; the Indexer runs per layer.
    """
    if index_cache_dtype != "fp4":
        return False
    head_dim = getattr(config, "index_head_dim", None)
    n_heads = getattr(config, "index_n_heads", None)
    if not getattr(config, "index_topk", None):
        why = "the layer has no sparse indexer"
    elif head_dim != _K_TILE:
        why = f"index_head_dim is {head_dim}, not {_K_TILE}"
    elif not n_heads or n_heads % _MFMA_M:
        why = f"index_n_heads is {n_heads}, not a multiple of {_MFMA_M}"
    elif (getattr(config, "index_kpool", 1) or 1) != 1:
        # A pooled indexer scores through `sparse_attn_indexer_kpool`, which has
        # no FP4 arm, and keeps `block_size // index_kpool` rows per block --
        # so it would also hand the scorer a block size the cache never used.
        why = f"index_kpool is {config.index_kpool}, and pooling has no FP4 scorer"
    elif (gfx := _gfx()) != "gfx950":
        why = f"{gfx} does not have the FP4 mqa-logits kernels"
    else:
        return True
    if warn:
        logger.warning("FP4 sparse indexer unavailable (%s); scoring in FP8.", why)
    return False


def assert_fp4_indexer_supported(
    *, fused_writer: bool, prefill_context_parallel: bool, prefill_ubatching: bool
) -> None:
    """Reject the FP4 requests this build cannot serve.

    Both are knobs the user set, not geometry we can read off the config, so
    they raise instead of falling back: quietly ignoring `--index-cache-dtype
    fp4` would hide which of the two cost the memory the flag was asked for.
    """
    if not fused_writer:
        raise ValueError(
            "The FP4 sparse indexer requires the fused QK/RoPE/cache kernel, "
            "the only writer of the packed E2M1 planes. Either "
            "ATOM_DISABLE_DS_INDEXER_QK_ROPE_CACHE_FUSION is set, which unset "
            "restores it, or this indexer has no fused kernel to begin with: "
            "that needs head_dim == quant_block_size and rope_dim == "
            "head_dim // 2, so a NoPE indexer has no FP4 route at all. Pass "
            "--index-cache-dtype fp8."
        )
    if prefill_context_parallel:
        raise ValueError(
            "The FP4 sparse indexer does not support PCP, whose candidate "
            "exchange is the one reader of the fp32 `weights` the FP4 writer "
            "does not produce. Pass --index-cache-dtype fp8."
        )
    if prefill_ubatching:
        raise ValueError(
            "The FP4 sparse indexer does not support prefill micro-batching. "
            "`split_attn_metadata` rebuilds the metadata from its declared "
            "fields, and the FP4 schedule rides on undeclared ones whose row "
            "ids a ubatch rebases. Decode micro-batching (--enable-tbo-decode) "
            "is supported. Pass --index-cache-dtype fp8."
        )


def fp4_decode_parallel_units(max_bs: int, next_n: int) -> int:
    """CTAs the varctx schedule is built for -- the grid a CUDAGraph captures.

    `compute_varctx_schedule` needs a multiple of `next_n` leaving at least one
    slot per sequence, so the floor rounds up to both. Not monotonic in `next_n`
    -- at 512 units `f(3)` is 513 against `f(4)`'s 512. What lets one buffer
    serve every width is the weaker `f(next_n) >= f(1)`, which holds because
    both arms of the max scale with `next_n`: the buffer is sized at
    `max_seqlen_qo` and the draft, the only caller asking for a different width,
    asks for 1.
    """
    return next_n * max(-(-FP4_MQA_VARCTX_PARALLEL_UNIT_NUM // next_n), max_bs)


def fp4_q_scale_shape(tokens: int, heads: int, head_dim: int) -> tuple:
    """`q_scale_out`'s shape: `[T, k_tiles, 4, 16, round_up(H // 16, 4)]`.

    The trailing pad is the dword a lane loads its four M-tile scale bytes with,
    so it is four even where the head count needs two.
    """
    m_tiles = heads // _MFMA_M
    return (tokens, head_dim // _K_TILE, 4, _MFMA_M, -(-m_tiles // 4) * 4)


def fp4_index_scale_rows(rows: torch.Tensor, block_size: int) -> torch.Tensor:
    """Where logical block rows `rows` sit on the e8m0 plane's row axis.

    `indexer_qk_rope_quant_and_cache` stores that axis as an `_MFMA_M`-wide
    transpose of the packed plane's, which is flat. Everything else addresses
    both planes by the same logical row, so any reader that moves the two
    together has to bend exactly here or it mixes exponents across a block --
    silently, since every index stays in bounds.

    `block_size` is the caller's own page size, taken rather than assumed: the
    lane count is a property of the plane the kernel wrote, and a caller paging
    the cache differently would otherwise get a wrong mapping that is still
    in-bounds. `fp4_index_block_shapes` is what holds the two equal.
    """
    if block_size != FP4_KV_BLOCK_SIZE:
        raise ValueError(
            f"the FP4 e8m0 row swizzle describes {FP4_KV_BLOCK_SIZE}-row blocks, "
            f"got {block_size}"
        )
    lanes = FP4_KV_BLOCK_SIZE // _MFMA_M
    return (rows % _MFMA_M) * lanes + rows // _MFMA_M


def fp4_index_block_shapes(rows: int, head_dim: int) -> tuple[tuple, tuple]:
    """One block's packed-E2M1 and e8m0 shapes, for one indexer layer."""
    if rows != FP4_KV_BLOCK_SIZE:
        raise ValueError(
            f"The FP4 sparse indexer requires --block-size {FP4_KV_BLOCK_SIZE}, "
            f"got {rows}"
        )
    k_tiles = head_dim // _K_TILE
    return (k_tiles, 4, rows, 16), (k_tiles, 4, rows)


def fp4_prefill_schedule(
    row_to_batch: torch.Tensor,
    local_ends: torch.Tensor,
    block_k: int,
    parallel_floor: int,
    max_seq_len: int,
    local_starts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Schedule one ragged-prefill forward, and the `local_starts` it scores from.

    `local_ends` is each row's causal upper bound in the column space the paged
    FP4 scorer emits -- DeepSeek-V4's `visible_end`, MLA's `cu_seqlen_ke -
    cu_seqlen_ks`. Rows start at column 0 there, so `local_starts` defaults to
    zeros; pass it when the columns are one concatenated plane instead, as they
    are under DCP, where the scorer reads a gathered copy of every sequence.

    `parallel_floor` is raised to the row count: prefill has one row per query
    token and every (row, chunk-split) needs a slot. `max_seq_len` has to be the
    width the scorer allocates its logits at, which the schedule bakes in.
    """
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
        compute_prefill_schedule,
    )

    if local_starts is None:
        local_starts = torch.zeros_like(local_ends)
    _, cta_info, n_ctas = compute_prefill_schedule(
        row_to_batch.to(torch.int32),
        local_starts,
        local_ends,
        block_k,
        max(parallel_floor, local_ends.shape[0]),
        max_seq_len,
    )
    return cta_info, n_ctas, local_starts


def fp4_decode_schedule(
    context_lens: torch.Tensor,
    block_k: int,
    parallel_units: int,
    max_seq_len: int,
    next_n: int,
    cta_info_out: torch.Tensor,
) -> None:
    """Refresh a decode step's schedule in place, at a CUDAGraph-stable address.

    `cta_info_out` has to be the buffer the captured kernel was handed and
    `parallel_units` its row count: the grid is baked at capture, so only the
    contents may change between replays.
    """
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        compute_varctx_schedule,
    )

    compute_varctx_schedule(
        context_lens,
        block_k,
        parallel_units,
        max_seq_len,
        next_n=next_n,
        cta_info_out=cta_info_out,
    )
