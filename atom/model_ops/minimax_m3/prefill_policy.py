"""Pure host policies for the MiniMax-M3 prefill indexer."""

import functools
from typing import NamedTuple

# Block-axis chunks provide grid parallelism, so Q128 can retain page reuse.
SCORE_BLOCK_SIZE_Q = 128
SCORE_CHUNK_CTAS_PER_CU = 8
PREFILL_TOPK_MIN_BLOCK_SIZE_K = 128
PREFILL_TOPK_MAX_BLOCK_SIZE_K = 1024
PREFILL_TOPK_NUM_WARPS = 1
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
# Multi-plane rows already supply occupancy. A short page chunk with two waves
# improves the long-query cases without the register cost of grouped stores.
_GFX1250_MULTI_PLANE_CHUNK = 28


class PrefillScoreConfig(NamedTuple):
    """One scorer specialization selected entirely from launch-time bounds."""

    block_size_q: int
    chunk_blocks: int | None
    num_warps: int | None
    num_stages: int | None
    store_group: int
    prefetch_pages: bool
    post_reduce_scale: bool


DEFAULT_PREFILL_SCORE_CONFIG = PrefillScoreConfig(
    block_size_q=SCORE_BLOCK_SIZE_Q,
    chunk_blocks=None,
    num_warps=None,
    num_stages=None,
    store_group=1,
    prefetch_pages=False,
    post_reduce_scale=False,
)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _floor_power_of_two(value: int) -> int:
    return 1 << (max(1, value).bit_length() - 1)


def _next_power_of_two(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


@functools.lru_cache(maxsize=128)
def parallel_prefill_selector_wins(
    arch: str,
    width: int,
    rows: int,
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
        and rows * (num_tiles - 1) >= 2 * compute_units
        # Every local list must fit in the aliased emitted-table workspace.
        and num_tiles * partials_per_tile <= workspace_keys
    )


def score_chunk_blocks(
    max_block: int, q_tiles: int, planes: int, compute_units: int
) -> int:
    want = compute_units * SCORE_CHUNK_CTAS_PER_CU
    chunks = max(1, want // max(1, q_tiles * planes))
    chunks = min(_floor_power_of_two(chunks), max(1, max_block))
    return _cdiv(max_block, chunks)


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
    """Choose a scorer launch without consulting live device-side lengths.

    ``None`` warp, stage, or chunk values mean that the generic launch policy
    owns that field. Keeping that distinction in a named value makes the
    policy/caller contract explicit and keeps CUDA Graph bounds stable.
    """
    if arch != "gfx1250" or not is_fp8:
        return DEFAULT_PREFILL_SCORE_CONFIG

    planes = batch * heads
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
    q128_tiles = _cdiv(max_query_len, SCORE_BLOCK_SIZE_Q)
    grouped_store_too_deep = (
        planes == 1
        and q128_tiles >= _cdiv(compute_units, 8)
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
        return DEFAULT_PREFILL_SCORE_CONFIG

    planes = heads
    q_tiles = _cdiv(max_query_len, SCORE_BLOCK_SIZE_Q)
    # The optimized paths repay their setup only with a long block row and more
    # than one Q128 tile per 16 CUs across independent heads.
    if max_block <= compute_units or q_tiles * planes <= _cdiv(compute_units, 16):
        return DEFAULT_PREFILL_SCORE_CONFIG

    if planes > 1:
        plane_tiles = q_tiles * planes
        if plane_tiles < _cdiv(compute_units, 2) or (
            max_block < 2 * compute_units and plane_tiles < compute_units
        ):
            return DEFAULT_PREFILL_SCORE_CONFIG
        # At both long-Q and long-context, fixed 28-page chunks over-split the grid.
        if max_block > 4 * compute_units and plane_tiles > 4 * compute_units:
            return DEFAULT_PREFILL_SCORE_CONFIG
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
        fallback_chunk = score_chunk_blocks(max_block, q_tiles, 1, compute_units)
        fallback_programs = q_tiles * _cdiv(max_block, fallback_chunk)
        if fallback_programs >= compute_units * (_GFX1250_CTAS_PER_CU - 1):
            return DEFAULT_PREFILL_SCORE_CONFIG

    if (
        not narrow_q
        and max_block < 2 * compute_units
        and q_tiles < _cdiv(compute_units, _GFX1250_CTAS_PER_CU)
    ):
        return DEFAULT_PREFILL_SCORE_CONFIG
    # A short context already exposes enough Q128 tiles at very large Q.
    if max_block < 4 * compute_units and q_tiles >= 2 * compute_units:
        return DEFAULT_PREFILL_SCORE_CONFIG

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
