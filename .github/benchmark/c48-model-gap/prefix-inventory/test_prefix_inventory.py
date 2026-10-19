"""Validate later-key evidence without altering the real pool or matcher."""

import io
import json
from types import SimpleNamespace

import pytest
from pd_c48_cache_lifecycle import CacheLifecycleProbe
from pd_c48_physical_lifecycle import PhysicalLifecycle
from pd_c48_prefix_inventory import PrefixInventory
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id


def state(pool):
    return (
        [b.block_id for b in pool.free_block_queue.get_all_free_blocks()],
        [(b.ref_cnt, b.block_hash) for b in pool.blocks],
    )


def fixture():
    pool = BlockPool(num_gpu_blocks=12, enable_caching=True, hash_block_size=128)
    hashes = [BlockHash(i.to_bytes(32, "big")) for i in range(8)]

    def match(request, unused):
        for i, h in enumerate(request.block_hashes):
            if pool.get_cached_block(h, [3]) is None:
                return i * 128, False
        return len(request.block_hashes) * 128, False

    offload = SimpleNamespace(cpu_block_pool=pool, get_num_new_matched_tokens=match)
    logical_stream = io.StringIO()
    logical = CacheLifecycleProbe(
        offload, logical_stream, min_prompt=0, watch_published_group_ids=set()
    ).install()
    physical = PhysicalLifecycle(pool).install()
    physical.attach_to_map_probe(logical)
    blocks = pool.get_new_blocks(8)
    for i, block in enumerate(blocks):
        pool._insert_block_hash(make_block_hash_with_group_id(hashes[i], 3), block, 128)
    pool.free_blocks(blocks)
    pool.evict_blocks({blocks[2].block_id})
    request = SimpleNamespace(
        request_id="target",
        num_prompt_tokens=544100,
        arrival_time=0,
        skip_reading_prefix_cache=False,
        block_hashes=hashes,
    )
    return pool, logical, physical, logical_stream, request


def test_inventory_exposes_later_resident_keys_and_preserves_matcher_and_queue():
    pool, logical, physical, logical_stream, request = fixture()
    before = state(pool)
    sequence = physical.coverage()["last_sequence"]
    inventory_stream = io.StringIO()
    inventory = PrefixInventory(
        logical, physical, inventory_stream, group_ids=(0, 1, 2, 3)
    ).install()
    assert logical.offload.get_num_new_matched_tokens(request, 0) == (256, False)
    assert state(pool) == before
    assert physical.coverage()["last_sequence"] == sequence
    actual = json.loads(logical_stream.getvalue().splitlines()[-1])
    assert actual["groups"]["3"]["queries"] == 3
    assert actual["groups"]["3"]["hits"] == 2
    assert actual["groups"]["3"]["misses"] == 1
    snapshot = json.loads(inventory_stream.getvalue())
    fa = snapshot["groups"]["3"]
    assert fa["examined_hashes"] == 8
    assert fa["resident_hashes"] == 7
    hole = fa["entries"][2]
    assert hole["hash_index"] == 2 and hole["block_id"] is None
    assert hole["history"]["last_remove"]["reason"] == "explicit_eviction"
    assert all(x["block_id"] is not None for x in fa["entries"][3:])
    assert snapshot["groups"]["0"]["entries"] == []
    assert snapshot["history_complete_from_installation"]
    assert not inventory.coverage()["errors"]
    # Repeated retries do not serialize the same request again.
    assert logical.offload.get_num_new_matched_tokens(request, 0) == (256, False)
    assert inventory.captured == 1


@pytest.mark.parametrize("mode", ["out_of_range", "cap", "hash_cap", "io_failure"])
def test_bounded_or_failed_capture_preserves_the_serving_lookup(mode):
    pool, logical, physical, _, request = fixture()
    stream = io.StringIO()
    kwargs = {}
    if mode == "out_of_range":
        request.num_prompt_tokens = 100
    elif mode == "cap":
        kwargs["max_requests"] = 0
    elif mode == "hash_cap":
        kwargs["max_hashes"] = 1
    else:
        stream.close()
    observer = PrefixInventory(
        logical, physical, stream, group_ids=(3,), **kwargs
    ).install()
    before = state(pool)
    assert logical.offload.get_num_new_matched_tokens(request, 0) == (256, False)
    assert state(pool) == before
    assert observer.captured == 0
    if mode == "io_failure":
        assert observer.errors == ["ValueError"]
    if mode == "cap":
        assert observer.cap_rejections == 1
    if mode == "hash_cap":
        assert observer.hash_rejections == 1
