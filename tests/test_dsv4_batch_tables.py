"""Deterministic tests for DSV4 logical-to-physical batch tables."""

import numpy as np

from atom.kv_cache.dsv4.batch_tables import build_dsv4_batch_tables
from atom.model_engine.unified_kv_arena import UnifiedKvArena


def _group(name, owners, *, num_chunks=4):
    return {
        "name": name,
        "num_chunks": num_chunks,
        "bytes_per_chunk": 128,
        "chunk_rows": 128,
        "owners": owners,
    }


def _arena():
    return UnifiedKvArena(
        block_size=128,
        group_specs=[
            _group("c4", {"swa": 128, "compress": 32}),
            _group("c128", {"swa": 128, "compress": 1}),
            _group("dense", {"swa": 128}),
        ],
    )


def test_arena_off_is_empty_with_logical_source_passthrough():
    sources = np.asarray([7, -1], dtype=np.int32)
    tables = build_dsv4_batch_tables(
        arena=None,
        block_tables=[[0, 1]],
        swa_block_tables=[[3, -1]],
        v4_csa_boundary_source_ids=sources,
    )
    assert tables.is_empty
    assert tables.v4_csa_boundary_source_main is sources
    assert tables.v4_csa_boundary_source_idx is sources


def test_builds_every_group_physical_table():
    arena = _arena()
    arena.alloc_compressed(0)
    arena.alloc_swa(10)
    tables = build_dsv4_batch_tables(
        arena=arena,
        block_tables=[[0, -1]],
        swa_block_tables=[[10, -1]],
        v4_csa_boundary_source_ids=[10],
    )
    assert set(tables.arena_block_tables) == {"c4", "c128", "dense"}
    assert set(tables.arena_swa_block_tables) == {"c4", "c128", "dense"}
    assert tables.arena_block_tables["c4"][0][0] == arena.compress_page("c4", 0)
    assert tables.arena_swa_block_tables["dense"][0][0] == arena.swa_page("dense", 10)


def test_csa_logical_minus_one_and_unbacked_stay_minus_one():
    arena = _arena()
    arena.alloc_swa(10)
    physical = arena.swa_page("c4", 10)
    tables = build_dsv4_batch_tables(
        arena=arena,
        block_tables=[[0, 1, 2]],
        swa_block_tables=[[10, -1, 11]],
        v4_csa_boundary_source_ids=[10, -1, 11],
    )
    assert tables.csa_main_page_tables == [[physical, -1, -1]]
    assert tables.csa_main_page_tables is tables.csa_idx_page_tables
    np.testing.assert_array_equal(
        tables.v4_csa_boundary_source_main,
        np.asarray([physical, -1, -1], dtype=np.int32),
    )
    assert tables.v4_csa_boundary_source_main is tables.v4_csa_boundary_source_idx


def test_main_and_indexer_share_the_same_c4_page():
    arena = _arena()
    arena.alloc_swa(2)
    tables = build_dsv4_batch_tables(
        arena=arena,
        block_tables=[[0]],
        swa_block_tables=[[2]],
        v4_csa_boundary_source_ids=[2],
    )
    expected = arena.swa_page("c4", 2)
    assert tables.csa_main_page_tables[0][0] == expected
    assert tables.csa_idx_page_tables[0][0] == expected
