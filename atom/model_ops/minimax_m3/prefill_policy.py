"""Pure host policies for the MiniMax-M3 prefill indexer."""

import functools
from typing import NamedTuple

# Block-axis chunks provide grid parallelism, so Q128 can retain page reuse.
SCORE_BLOCK_SIZE_Q = 128
SCORE_CHUNK_CTAS_PER_CU = 8
PREFILL_TOPK_MIN_BLOCK_SIZE_K = 128
# gfx950: rows of <= 64 blocks sort one next_pow2(blocks) tile, up to 1.2x faster
GFX950_PREFILL_TOPK_MIN_BLOCK_SIZE_K = 16
PREFILL_TOPK_MAX_BLOCK_SIZE_K = 1024
# Prefill launches one selector program per query token. One warp won all 28
# measured (row width x warp count) cells; more only added cross-wave work.
PREFILL_TOPK_NUM_WARPS = 1
_SCORE_NUM_STAGES = 3
# Six workgroups per CU hid the score loop latency best across the MI455 sweep.
_GFX1250_CTAS_PER_CU = 6
# The four-block store specialization needs more independent CTAs as its block
# loop grows. Two CTA waves per rounded context-block wave, clamped to the
# measured resident range, scales that latency from the workload and CU count.
_GFX1250_GROUPED_MIN_CTAS_PER_CU = 4
_GFX1250_GROUPED_MAX_CTAS_PER_CU = 12
# Q256 reuses each page across more rows; two workgroups per CU keep that larger
# tile resident without turning context length into a shape table.
_GFX1250_WIDE_CTAS_PER_CU = 2
# Multi-plane rows already supply occupancy. A short block chunk with two waves
# improves the long-query cases without the register cost of grouped stores.
_GFX1250_MULTI_PLANE_CHUNK = 28


class PrefillScoreConfig(NamedTuple):
    """The complete scorer launch selected from launch-time bounds."""

    block_size_q: int
    chunk_blocks: int
    num_warps: int
    num_stages: int
    store_group: int
    prefetch_pages: bool
    post_reduce_scale: bool


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _floor_power_of_two(value: int) -> int:
    return 1 << (max(1, value).bit_length() - 1)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def parallel_prefill_selector_wins(
    arch: str,
    width: int,
    query_rows: int,
    compute_units: int,
    emit: bool,
    topk: int,
    num_idx_heads: int,
    decode_max_q: int,
    pages_per_block: int,
) -> bool:
    """Whether parallel 1024-block local top-k passes repay their merge."""
    num_tiles = _cdiv(width, PREFILL_TOPK_MAX_BLOCK_SIZE_K)
    partials_per_tile = _next_power_of_two(topk)
    # sparse_bt is int32 and the partial keys are int64 views of the same row.
    workspace_keys = topk * pages_per_block // 2
    return (
        arch == "gfx1250"
        and decode_max_q == 0
        and emit
        and topk == 16
        and num_idx_heads == 1
        # The second tile has at least one block per CU.
        and width - PREFILL_TOPK_MAX_BLOCK_SIZE_K >= compute_units
        # Pay for the merge only when the extra tiles add two CTAs per CU.
        and query_rows * (num_tiles - 1) >= 2 * compute_units
        # Every local list must fit in the aliased emitted-table workspace.
        and num_tiles * partials_per_tile <= workspace_keys
    )


def _score_chunk_blocks(
    max_block: int, q_tiles: int, planes: int, compute_units: int
) -> int:
    want = compute_units * SCORE_CHUNK_CTAS_PER_CU
    chunks = max(1, want // max(1, q_tiles * planes))
    chunks = min(_floor_power_of_two(chunks), max(1, max_block))
    return _cdiv(max_block, chunks)


def _default_prefill_score_config(
    arch: str,
    is_fp8: bool,
    max_query_len: int,
    max_block: int,
    batch: int,
    heads: int,
    compute_units: int,
) -> PrefillScoreConfig:
    q_tiles = _cdiv(max_query_len, SCORE_BLOCK_SIZE_Q)
    chunk_blocks = _score_chunk_blocks(max_block, q_tiles, batch * heads, compute_units)
    if arch == "gfx1250":
        num_warps, num_stages = (4, 1) if is_fp8 else (8, _SCORE_NUM_STAGES)
    else:
        num_warps, num_stages = 4, _SCORE_NUM_STAGES
    return PrefillScoreConfig(
        block_size_q=SCORE_BLOCK_SIZE_Q,
        chunk_blocks=chunk_blocks,
        num_warps=num_warps,
        num_stages=num_stages,
        store_group=1,
        prefetch_pages=False,
        # Same scores (a positive scale commutes with the block max), fewer multiplies.
        post_reduce_scale=arch == "gfx950",
    )


@functools.lru_cache(maxsize=128)
def prefill_score_config(
    arch: str,
    is_fp8: bool,
    max_query_len: int,
    max_block: int,
    batch: int,
    heads: int,
    compute_units: int,
) -> PrefillScoreConfig:
    """Choose the complete scorer launch without live device-side lengths."""
    fallback = _default_prefill_score_config(
        arch,
        is_fp8,
        max_query_len,
        max_block,
        batch,
        heads,
        compute_units,
    )
    if arch != "gfx1250" or not is_fp8:
        return fallback

    planes = batch * heads
    q_tiles = _cdiv(max_query_len, SCORE_BLOCK_SIZE_Q)
    grouped_q_tiles = _cdiv(max_query_len, 64)
    grouped_plane_tiles = grouped_q_tiles * planes
    block_waves = max(1, (max_block + compute_units // 2) // compute_units)
    grouped_ctas_per_cu = min(
        _GFX1250_GROUPED_MAX_CTAS_PER_CU,
        max(_GFX1250_GROUPED_MIN_CTAS_PER_CU, 2 * block_waves),
    )
    grouped_chunks = min(
        max_block,
        max(
            1,
            _cdiv(
                compute_units * grouped_ctas_per_cu,
                max(1, grouped_plane_tiles),
            ),
        ),
    )
    grouped_chunk_blocks = _cdiv(max_block, grouped_chunks)
    grouped_store_too_deep = (
        planes == 1
        and q_tiles >= _cdiv(compute_units, 8)
        and grouped_chunk_blocks > _cdiv(compute_units, _GFX1250_CTAS_PER_CU)
    )
    if (
        max_block > compute_units
        # Too little work cannot amortize the Q64 specialization.
        and grouped_plane_tiles > _cdiv(compute_units, 16)
        # More than two tile waves per CU makes rereading K at Q64 dominate.
        and grouped_plane_tiles <= 2 * compute_units
        # Bound each plane as well as their aggregate: extra heads/requests add
        # independent query work but cannot justify an arbitrarily narrow tile.
        and grouped_q_tiles <= _cdiv(compute_units, 2)
        # With one already-wide Q plane, a deep Q64 chunk rereads too much K.
        # Multi-plane launches have more independent query work and are governed
        # by the aggregate tile bound above instead.
        and not grouped_store_too_deep
    ):
        return PrefillScoreConfig(
            block_size_q=64,
            chunk_blocks=grouped_chunk_blocks,
            num_warps=2,
            num_stages=3,
            store_group=4,
            prefetch_pages=True,
            post_reduce_scale=True,
        )

    # Unlike index heads, separate batches do not share score pages.
    if batch != 1:
        return fallback

    planes = heads
    # The optimized paths repay their setup only with a long block row and more
    # than one Q128 tile per 16 CUs across independent heads.
    if max_block <= compute_units or q_tiles * planes <= _cdiv(compute_units, 16):
        return fallback

    if planes > 1:
        plane_tiles = q_tiles * planes
        if plane_tiles < _cdiv(compute_units, 2) or (
            max_block < 2 * compute_units and plane_tiles < compute_units
        ):
            return fallback
        # At both long-Q and long-context, fixed 28-block chunks over-split the grid.
        if max_block > 4 * compute_units and plane_tiles > 4 * compute_units:
            return fallback
        return PrefillScoreConfig(
            block_size_q=SCORE_BLOCK_SIZE_Q,
            chunk_blocks=_GFX1250_MULTI_PLANE_CHUNK,
            num_warps=2,
            num_stages=1,
            store_group=1,
            prefetch_pages=False,
            post_reduce_scale=True,
        )

    narrow_q = q_tiles < _cdiv(compute_units, 8)
    if narrow_q:
        fallback_chunk = fallback.chunk_blocks
        fallback_programs = q_tiles * _cdiv(max_block, fallback_chunk)
        if fallback_programs >= compute_units * (_GFX1250_CTAS_PER_CU - 1):
            return fallback

    if (
        not narrow_q
        and max_block < 2 * compute_units
        and q_tiles < _cdiv(compute_units, _GFX1250_CTAS_PER_CU)
    ):
        return fallback
    # A short context already exposes enough Q128 tiles at very large Q.
    if max_block < 4 * compute_units and q_tiles >= 2 * compute_units:
        return fallback

    tile_q = 256
    q_tiles = _cdiv(max_query_len, tile_q)
    chunks = min(
        max_block,
        max(1, compute_units * _GFX1250_WIDE_CTAS_PER_CU // q_tiles),
    )
    return PrefillScoreConfig(
        block_size_q=tile_q,
        chunk_blocks=_cdiv(max_block, chunks),
        num_warps=4,
        num_stages=1,
        store_group=1,
        prefetch_pages=False,
        post_reduce_scale=True,
    )
