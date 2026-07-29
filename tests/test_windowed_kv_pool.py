"""Tests for layout-neutral windowed KV lifecycle."""

from atom.kv_cache.pools.windowed_kv_pool import WindowedKvPool
from atom.model_engine.sequence import Sequence


def _sequence(tokens):
    seq = Sequence(tokens, block_size=4)
    seq.swa_block_table = [-1] * seq.num_blocks
    return seq


def test_disabled_pool_is_identity():
    pool = WindowedKvPool(0, 0, 4, 16, 0)
    seq = _sequence([1, 2, 3, 4])
    assert pool.has_free(100)
    assert pool.bounded_hit(seq, 3, []) == 3
    pool.ensure_for_tokens(seq, 0, 4)
    assert seq.swa_block_table == [-1]


def test_materialize_and_release_conserve_ids():
    pool = WindowedKvPool(4, 4, 4, 16, 0)
    seq = _sequence(list(range(8)))
    pool.ensure_for_tokens(seq, 0, 8)
    assert seq.swa_block_table[0] == -1
    assert seq.swa_block_table[1] >= 0
    assert len(pool.used_block_ids) == 1
    pool.release(seq)
    assert seq.swa_block_table == []
    assert pool._free_list.ids_conserved()


def test_hash_publish_allows_window_bounded_reuse():
    pool = WindowedKvPool(4, 4, 4, 16, 0)
    seq = _sequence([1, 2, 3, 4])
    pool.ensure_for_tokens(seq, 0, 4)
    block_hash = 123
    pool.publish_hash(seq, 0, block_hash, seq.block(0))
    pool.release(seq)

    hit = _sequence([1, 2, 3, 4])
    assert pool.bounded_hit(hit, 1, [block_hash]) == 1
    # Cached-prefix allocation appends into an initially empty table.
    hit.swa_block_table.clear()
    pool.claim_cached(hit, block_hash, hit.block(0))
    assert hit.swa_block_table[0] >= 0
