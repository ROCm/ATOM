# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Contract for the chunk-major staging grid's tile table.

The grid used to be rectangular -- one row per (chunk, segment) job and a
column count taken from the widest segment -- so a 512-byte MXFP8 scale was
launched with the tile count of a megabyte-sized cache and masked off all but
the first tile. The table replaces that with one entry per tile that has bytes
to move, which means the grid is now only as correct as the table: an off-by-
one no longer lands in a masked region where it cannot be seen, it moves the
wrong bytes. So pin all three parts -- the table covers every tile each job
needs, it is sized by the work rather than by the largest segment, and a
transfer's groups each get their own slice of the one concatenated upload.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import torch

_MODULE_NAME = "atom.kv_transfer.offload.dense.triton_kv_staging"
_REAL_TRITON = importlib.util.find_spec("triton") is not None
_CPU_MODULE = None


def _staging_module():
    """The real wrapper source, loaded with an import-only Triton stub if needed.

    ``from __future__ import annotations`` in the wrapper keeps every
    ``tl.constexpr`` annotation a string, so nothing in the stub is consulted
    past ``@triton.jit`` accepting a function.
    """
    global _CPU_MODULE
    if _REAL_TRITON:
        return importlib.import_module(_MODULE_NAME)
    if _CPU_MODULE is not None:
        return _CPU_MODULE

    fake_triton = ModuleType("triton")
    fake_language = ModuleType("triton.language")
    fake_triton.__path__ = []
    fake_triton.language = fake_language
    fake_triton.jit = lambda function: function
    isolated_name = "_dense_triton_kv_staging_cpu_contract"
    source = (
        Path(__file__).parents[1]
        / "atom/kv_transfer/offload/dense/triton_kv_staging.py"
    )
    spec = importlib.util.spec_from_file_location(isolated_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    missing = object()
    original_triton = sys.modules.get("triton", missing)
    original_language = sys.modules.get("triton.language", missing)
    sys.modules["triton"] = fake_triton
    sys.modules["triton.language"] = fake_language
    sys.modules[isolated_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if original_triton is missing:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = original_triton
        if original_language is missing:
            sys.modules.pop("triton.language", None)
        else:
            sys.modules["triton.language"] = original_language
    _CPU_MODULE = module
    return module


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


def test_each_group_slices_its_own_table_out_of_the_shared_upload():
    # prepare_chunk_major_groups uploads one table for the whole transfer and
    # hands each group two views into it. Groups here differ in shape so a
    # slice taken from the wrong group would move another group's bytes.
    module = _staging_module()
    sizes = [512, 16384, 980992]
    group_counts = [[3, 1], [8], [0, 0], [2, 5, 1]]

    table, spans = module._group_tile_tables(group_counts, sizes)

    assert table.dtype is torch.int32
    assert len(spans) == len(group_counts)
    covered = 0
    for counts, (job_span, pos_span, num_tiles) in zip(group_counts, spans):
        want_job, want_pos = _table(counts, sizes)
        assert num_tiles == len(want_job)
        assert table[job_span].tolist() == want_job
        assert table[pos_span].tolist() == want_pos
        # Contiguous and non-overlapping: job then pos, group after group.
        assert job_span.start == covered
        assert pos_span.start == job_span.stop
        covered = pos_span.stop
    assert covered == int(table.numel())


def test_a_transfer_with_no_groups_still_yields_an_empty_table():
    table, spans = _staging_module()._group_tile_tables([], [1024])

    assert spans == []
    assert table.dtype is torch.int32 and int(table.numel()) == 0
