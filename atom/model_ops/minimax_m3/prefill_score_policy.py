"""Host launch policy for the MiniMax-M3 prefill index scorer."""

import functools

# Block-axis chunks provide grid parallelism, so Q128 can retain page reuse.
SCORE_BLOCK_SIZE_Q = 128
SCORE_CHUNK_CTAS_PER_CU = 8
# Six workgroups per CU hid the score loop latency best across the MI455 sweep.
_GFX1250_CTAS_PER_CU = 6
# Q256 reuses each page across more rows; two workgroups per CU keep that larger
# tile resident without turning context length into a shape table.
_GFX1250_WIDE_CTAS_PER_CU = 2
# Q64 reads every K page twice as often as Q128. Keep it to contexts where the
# extra query tiles still fit the measured four-block-store working envelope.
_GFX1250_Q64_BLOCKS_PER_CU = 4
# Multi-plane rows already supply occupancy. A short page chunk with two waves
# improves the long-query cases without the register cost of grouped stores.
_GFX1250_MULTI_PLANE_CHUNK = 28


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _floor_power_of_two(value: int) -> int:
    return 1 << (max(1, value).bit_length() - 1)


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
) -> tuple[int, int | None, int | None, int | None, int, bool, bool]:
    fallback = (SCORE_BLOCK_SIZE_Q, None, None, None, 1, False, False)
    if arch != "gfx1250" or not is_fp8:
        return fallback
    # Unlike index heads, separate batches do not share score pages.
    if batch != 1:
        return fallback

    planes = heads
    q_tiles = _cdiv(max_query_len, SCORE_BLOCK_SIZE_Q)
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
        # At both long-Q and long-context, fixed 28-page chunks over-split the grid.
        if max_block > 4 * compute_units and plane_tiles > 4 * compute_units:
            return fallback
        return (
            SCORE_BLOCK_SIZE_Q,
            _GFX1250_MULTI_PLANE_CHUNK,
            2,
            1,
            1,
            False,
            True,
        )

    # A lone plane below one Q128 tile per eight CUs needs Q64 for occupancy.
    narrow_q = q_tiles < _cdiv(compute_units, 8)
    # Otherwise doubling the page reads costs more than the extra Q64 grid earns.
    if narrow_q and max_block <= compute_units * _GFX1250_Q64_BLOCKS_PER_CU:
        tile_q = 64
        q_tiles_64 = _cdiv(max_query_len, tile_q)
        if max_query_len < 3 * max_block or q_tiles_64 > _cdiv(
            compute_units, _GFX1250_CTAS_PER_CU
        ):
            return fallback
        chunks = min(
            max_block,
            max(1, _cdiv(compute_units * _GFX1250_CTAS_PER_CU, q_tiles_64)),
        )
        chunk_blocks = _cdiv(max_block, chunks)
        # Multiples of 16 pages are a repeatable memory-scheduling cliff on gfx1250.
        if chunk_blocks % 16 == 0:
            chunk_blocks += 1
        return tile_q, chunk_blocks, 2, 3, 4, True, True

    if narrow_q:
        fallback_chunk = score_chunk_blocks(max_block, q_tiles, 1, compute_units)
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
    return tile_q, _cdiv(max_block, chunks), 4, 1, 1, False, True
