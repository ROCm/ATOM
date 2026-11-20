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


def parallel_prefill_selector_wins(
    arch: str,
    width: int,
    rows: int,
    compute_units: int,
    emit: bool,
    topk: int,
    num_idx_heads: int,
    decode_max_q: int,
) -> bool:
    """Whether parallel 1024-block local top-k passes repay their merge."""
    num_tiles = _cdiv(width, PREFILL_TOPK_MAX_BLOCK_SIZE_K)
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
        # Four local lists fit in sparse_bt's 128-int32 output row.
        and width <= 4 * PREFILL_TOPK_MAX_BLOCK_SIZE_K
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
    fp4: bool = False,
) -> PrefillScoreConfig:
    """Choose a scorer launch without consulting live device-side lengths.

    ``None`` warp, stage, or chunk values mean that the generic launch policy
    owns that field. Keeping that distinction in a named value makes the
    policy/caller contract explicit and keeps CUDA Graph bounds stable.

    ``fp4``: the FP4 index cache (``index_fp4``). It takes the FP8 buckets --
    the same page traffic pattern at 72 B a row instead of 128 -- with every
    specialization at one stage: its rows are dequantized in registers
    (``nvfp4_load_rows``), which deeper pipelining only multiplies (gfx1250
    offline build at one stage: Q64 grouped 368 VGPRs, Q256 340, the Q128
    multi-plane tile 338 at four warps -- 694 at its FP8 two -- none spilled).
    """
    if arch != "gfx1250" or not (is_fp8 or fp4):
        return DEFAULT_PREFILL_SCORE_CONFIG
    if fp4 and _fp4_batched_wide(max_query_len, max_block, batch):
        return FP4_BATCHED_PREFILL_SCORE_CONFIG
    config = _gfx1250_prefill_score_config(
        max_query_len, max_block, batch, heads, compute_units
    )
    if fp4 and config is not DEFAULT_PREFILL_SCORE_CONFIG:
        config = config._replace(num_stages=FP4_PREFILL_SCORE_NUM_STAGES)
        if config.block_size_q == SCORE_BLOCK_SIZE_Q and config.num_warps == 2:
            # The multi-plane Q128 tile: at two warps the FP4 build needs 694
            # VGPRs and spills SGPRs, at four 338.
            config = config._replace(num_warps=4)
        if (
            config.block_size_q == FP4_WIDE_SCORE_BLOCK_SIZE_Q
            and max_block >= FP4_WIDE_STORE_GROUP_MIN_BLOCKS
        ):
            # The wide tile's stores, not its WMMAs, are what it waits on: one
            # 4-byte store a (row, block), 256 rows apart. With no store at all
            # the same loop runs 1.55x faster (MI455X, 5979 query tokens against
            # 514K: 328us -> 212us a layer). Writing four blocks a row at once
            # recovers part of that -- 328 -> 264us for the scorer, 414 -> 371us
            # with the top-k, same scores -- on a long row; at 1024 blocks it
            # loses 5%, so the short contexts keep one store a block.
            config = config._replace(store_group=FP4_WIDE_STORE_GROUP)
    return config


# Stages of every tuned FP4 prefill specialization (see `prefill_score_config`).
FP4_PREFILL_SCORE_NUM_STAGES = 1
# FP4, several requests in one prefill forward (AgentX at concurrency 64 batches
# 2-7 of them, contexts 50K-720K): the FP8 buckets hand most of these the
# default launch -- Q128, a chunk of ~1000 blocks, a store a block, the scale on
# every product -- and the rest the Q64 grouped one at a chunk of 200-560. The
# wide tile with grouped stores at a fixed 64-block chunk beats both. MI455X,
# score + top-k a layer, us (the bucket's launch -> this one), same selection:
#
#     requests  query tokens  longest context   bucket          us
#     7         32667         519K              default         2048 -> 944
#     3         30506         574K              default         2103 -> 1008
#     4         32732         218K              default         1475 -> 636
#     5         15256         723K              Q64 grouped     1006 -> 530
#     2         8602          373K              Q64 grouped      418 -> 344
#     4         13478         371K              Q64 grouped      432 -> 460
#
# The chunk is what moves it: at the default's ~1000 blocks every tile gives
# 600-700 TFLOP/s, at 64 the wide tile gives 1250-1350. Applied from the
# smallest shape measured -- a 1024-block bound and 16 wide query tiles over
# the batch; smaller batches keep their bucket.
FP4_BATCHED_PREFILL_SCORE_CONFIG = PrefillScoreConfig(
    block_size_q=256,
    chunk_blocks=64,
    num_warps=8,
    num_stages=1,
    store_group=4,
    prefetch_pages=False,
    post_reduce_scale=True,
)
FP4_BATCHED_MIN_BLOCKS = 1024
FP4_BATCHED_MIN_QUERY_TILES = 16


def _fp4_batched_wide(max_query_len: int, max_block: int, batch: int) -> bool:
    return (
        batch > 1
        and max_block >= FP4_BATCHED_MIN_BLOCKS
        and batch * _cdiv(max_query_len, FP4_BATCHED_PREFILL_SCORE_CONFIG.block_size_q)
        >= FP4_BATCHED_MIN_QUERY_TILES
    )


# The FP4 wide (Q256) tile's grouped store: blocks a store, and the launch
# bound from which it pays (see `prefill_score_config`).
FP4_WIDE_SCORE_BLOCK_SIZE_Q = 256
FP4_WIDE_STORE_GROUP = 4
FP4_WIDE_STORE_GROUP_MIN_BLOCKS = 2048


def _gfx1250_prefill_score_config(
    max_query_len: int,
    max_block: int,
    batch: int,
    heads: int,
    compute_units: int,
) -> PrefillScoreConfig:
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
        # The 4-warp build (505 VGPRs, SGPRs spilled to VGPR lanes) hangs the GPU at long prefill.
        num_warps=8,
        num_stages=1,
        store_group=1,
        prefetch_pages=False,
        post_reduce_scale=True,
    )
