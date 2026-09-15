# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Contract for the chunk-major staging grid's tile table.

The grid used to be rectangular -- one row per (chunk, segment) job and a
column count taken from the widest segment -- so a 512-byte MXFP8 scale was
launched with the tile count of a megabyte-sized cache and masked off all but
the first tile. The table replaces that with one entry per tile that has bytes
to move, which means the grid is now only as correct as the table: an off-by-
one no longer lands in a masked region where it cannot be seen, it moves the
wrong bytes. So pin both halves -- the table covers every tile each job needs,
and it is sized by the work rather than by the largest segment.
"""

from __future__ import annotations

import math

import torch

# One Triton-stubbing module loader for the whole dense staging suite; see
# test_dense_staging_plan_key for why the wrapper needs one at all.
from test_dense_staging_plan_key import _staging_module


def _table(counts, segment_block_bytes):
    module = _staging_module()
    job, pos = module._tile_table(
        list(counts), list(segment_block_bytes), torch.device("cpu")
    )
    return job.tolist(), pos.tolist()


def _expected(counts, segment_block_bytes, block_bytes):
    """Every (job, tile) pair the kernel has to be handed, in job order."""
    pairs = []
    for chunk_id, nblocks in enumerate(counts):
        for seg_id, seg_bytes in enumerate(segment_block_bytes):
            job = chunk_id * len(segment_block_bytes) + seg_id
            tiles = math.ceil(nblocks * seg_bytes / block_bytes)
            pairs.extend((job, tile) for tile in range(tiles))
    return pairs


def test_table_covers_every_tile_of_every_job():
    # Segment sizes that straddle the tile: under it, exactly on it, one byte
    # over, and three orders of magnitude above.
    sizes = [7, 512, 1024, 1025, 16384, 980992]
    counts = [3, 1, 8]
    block_bytes = _staging_module()._BLOCK_BYTES

    job, pos = _table(counts, sizes)

    assert list(zip(job, pos)) == _expected(counts, sizes, block_bytes)


def test_a_chunk_with_no_blocks_contributes_no_tiles():
    sizes = [512, 16384]
    job, _ = _table([2, 0, 1], sizes)

    # Jobs 2 and 3 are the empty chunk's two segments.
    assert 2 not in job and 3 not in job
    assert set(job) == {0, 1, 4, 5}


def test_grid_is_sized_by_total_bytes_not_by_the_widest_segment():
    counts = [8] * 8
    block_bytes = _staging_module()._BLOCK_BYTES
    small = [16384] * 240

    # Same total bytes, redistributed so that one segment is much larger than
    # the rest. The rectangular grid charged every segment the widest one's
    # tile count, so this redistribution multiplied the program count by ~60;
    # the table has to leave it unchanged.
    lopsided = [512] * 239 + [240 * 16384 - 239 * 512]
    assert sum(lopsided) == sum(small)

    job_small, _ = _table(counts, small)
    job_lopsided, _ = _table(counts, lopsided)

    total_bytes = sum(counts) * sum(small)
    assert len(job_small) == math.ceil(total_bytes / block_bytes)
    # Not exactly equal: a segment that does not fill its last tile rounds up,
    # and the two spellings have different numbers of such segments.
    assert len(job_lopsided) <= len(job_small) + len(lopsided) * len(counts)
    assert len(job_lopsided) < 2 * len(job_small)


def test_table_is_int32_on_the_requested_device():
    job, pos = _staging_module()._tile_table([4], [1024, 2048], torch.device("cpu"))

    assert job.dtype is torch.int32 and pos.dtype is torch.int32
    assert job.device.type == "cpu" and pos.device.type == "cpu"
