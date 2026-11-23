# SPDX-License-Identifier: MIT
"""Raw checkpoint images cannot share the attention padding block."""

from array import array
from types import SimpleNamespace

import pytest
from test_state_checkpoint import (
    BLOCK,
    PAGED_COPY_RUNTIME,
    ckpt_config,
    make_block_manager,
    make_scheduler,
    stateful_seq,
)

from atom.model_engine.block_pool import BlockPool


def test_reserved_block_is_never_reissued_after_recycling_or_resize():
    pool = BlockPool(4, max_blocks=6, reserve_null_block=True)
    assert pool.num_blocks == 4
    assert pool.num_usable_blocks == pool.num_free == 3
    for _ in range(2):
        ids = [pool.pop() for _ in range(pool.num_free)]
        assert ids == [1, 2, 3]
        for block_id in ids:
            pool.allocate(block_id)
        for block_id in ids:
            pool.free(block_id)
        pool.clear_index()
        raw = pool.reserve_units(pool.num_usable_blocks, "checkpoint")
        assert raw == [1, 2, 3]
        pool.release_units(raw, "checkpoint")
    while pool.retire_top() is not None:
        pass
    assert pool.num_blocks == 1
    assert pool.num_usable_blocks == pool.num_free == 0
    assert pool.extend(5) == 5
    assert pool.reserve_units(5, "grown") == [1, 2, 3, 4, 5]


def test_unreserved_pool_keeps_its_full_capacity():
    pool = BlockPool(4)
    assert pool.num_usable_blocks == 4
    assert pool.reserve_units(4, "plain") == [0, 1, 2, 3]


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_transfer_image_cannot_take_null_block_without_prefix_caching(prefix_caching):
    bm = make_block_manager(
        ckpt_config(enable_prefix_caching=prefix_caching),
        state_runtime=PAGED_COPY_RUNTIME,
    )
    coordinator = bm.paged_state_checkpoints
    assert coordinator.enabled is prefix_caching
    for generation in range(2):
        units = coordinator.reserve_transfer_units(generation)
        assert units and 0 not in units
        coordinator.release_transfer_units(generation)
        bm.kv.clear_index()


def test_pool_sized_prompt_is_refused_and_frontend_reports_usable_limit():
    scheduler = make_scheduler(
        ckpt_config(num_kvcache_blocks=8), state_runtime=PAGED_COPY_RUNTIME
    )
    bm = scheduler.block_manager
    assert bm.max_pool_tokens == 7 * BLOCK
    reason = scheduler._unschedulable_reason(stateful_seq(range(8 * BLOCK)))
    assert reason is not None
    assert "total pool blocks=7" in reason
    assert scheduler._unschedulable_reason(stateful_seq(range(7 * BLOCK))) is None


def test_occupancy_and_watermark_use_the_same_usable_capacity():
    scheduler = make_scheduler(
        ckpt_config(num_kvcache_blocks=8), state_runtime=PAGED_COPY_RUNTIME
    )
    bm = scheduler.block_manager
    raw = bm.kv.reserve_units(3, "image")
    occupancy = bm.pool_occupancy()
    assert occupancy["total"] == occupancy["used"] + occupancy["free"] == 7
    assert bm.kv.eviction_stats()["blocks_total"] == 7
    assert scheduler._kv_usage() == pytest.approx(3 / 7)
    bm.kv.release_units(raw, "image")


def test_exported_metrics_keep_physical_id_bounds_and_usable_totals():
    from atom.model_engine.engine_utility import EngineUtilityHandler

    scheduler = make_scheduler(
        ckpt_config(num_kvcache_blocks=8), state_runtime=PAGED_COPY_RUNTIME
    )
    pool = scheduler.block_manager.kv
    pool.allocate(7)
    scheduler.deferred_free_blocks["held"] = SimpleNamespace(
        block_table=array("i", [7])
    )
    handler = object.__new__(EngineUtilityHandler)
    handler.scheduler = scheduler

    metrics = handler.collect_metrics()

    assert metrics["kv_blocks_total"] == 7
    assert metrics["kv_blocks_used"] + metrics["kv_blocks_free"] == 7
    assert metrics["kv_blocks_deferred_free"] == 1
