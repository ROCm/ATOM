# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_ZERO_EVICTED_BLOCKS: block 0 is the null page, released units are scrubbed.

Paged-attention kernels send masked / out-of-context pages to physical block 0
and read the unwritten tail of a sequence's last page. Once a released state
checkpoint's units come back as KV blocks, those lanes see bf16 GDN state bytes
(fp8 NaN patterns). The pool keeps block 0 out of circulation and queues every
released raw unit for a zero that rides the next built batch, ahead of anything
that can write the id again.
"""

import array

import numpy as np
import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointSpec,
    PageUnitCheckpointStore,
)


def toks(*ids):
    return array.array("i", ids)


def scrubbing_pool(n, **kw):
    return BlockPool(n, null_block=True, scrub_released=True, **kw)


class FreeOrderPool(BlockPool):
    """Asserts, at every `free`, that a raw unit was queued for its zero first."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.frees_checked = 0
        self._releasing = False

    def release_units(self, unit_ids, owner):
        self._releasing = True
        try:
            super().release_units(unit_ids, owner)
        finally:
            self._releasing = False

    def free(self, block_id):
        if self._releasing:
            assert (
                block_id in self._unscrubbed_set
            ), f"raw unit {block_id} became poppable before its zero was queued"
            self.frees_checked += 1
        super().free(block_id)


# ------------------------------- null page -------------------------------- #
class TestNullBlock:
    def test_pop_never_returns_block_zero(self):
        pool = scrubbing_pool(64)
        got = [pool.pop() for _ in range(63)]
        assert 0 not in got
        assert sorted(got) == list(range(1, 64))
        assert not pool.has_free(1)
        with pytest.raises(AssertionError):
            pool.pop()

    def test_block_zero_never_returned_after_full_pool_lru_eviction(self):
        evicted = []
        pool = scrubbing_pool(16, on_evict=evicted.append)
        # Fill the whole pool with published, freed (cached) blocks ...
        for i in range(15):
            b = pool.pop()
            pool.allocate(b)
            pool.publish(b, 1000 + i, toks(i))
            pool.free(b)
        assert pool.num_reusable_free == 15
        # ... then allocate through it twice: every allocation evicts.
        seen = []
        for _ in range(2):
            ids = []
            for _ in range(15):
                b = pool.pop()
                pool.allocate(b)
                ids.append(b)
            seen += ids
            for b in ids:
                pool.free(b)
        assert len(evicted) == 15
        assert 0 not in seen

    def test_reserve_units_and_clear_index_skip_block_zero(self):
        pool = scrubbing_pool(400)
        units = pool.reserve_units(380, owner="a")
        assert 0 not in units
        pool.release_units(units, owner="a")
        pool.clear_index()
        units = pool.reserve_units(399, owner="b")
        assert 0 not in units and len(units) == 399
        assert pool.reserve_units(1, owner="c") is None

    def test_allocate_refuses_block_zero_and_retire_stops_above_it(self):
        pool = scrubbing_pool(3)
        with pytest.raises(AssertionError):
            pool.allocate(0)
        assert pool.retire_top().retired == 2
        assert pool.retire_top().retired == 1
        assert pool.retire_top() is None

    def test_off_by_default_in_the_class(self):
        pool = BlockPool(4)
        assert pool.pop() == 0


# --------------------------- scrub on release ----------------------------- #
class TestScrubOnRelease:
    def test_release_queues_exactly_the_released_ids_in_order(self):
        pool = scrubbing_pool(1000)
        units = pool.reserve_units(380, owner="ck")
        assert pool.take_released_units() == ()
        pool.release_units(units, owner="ck")
        assert pool.take_released_units() == tuple(units)
        assert pool.take_released_units() == ()
        assert pool.units_scrubbed == 380

    def test_released_id_is_queued_before_it_can_be_popped(self):
        pool = FreeOrderPool(1000, null_block=True, scrub_released=True)
        units = pool.reserve_units(380, owner="ck")
        pool.release_units(units, owner="ck")
        assert pool.frees_checked == 380
        # Re-handed straight away (lowest id first): every one is still in the
        # queue the batch being built will drain, so its zero precedes the
        # new owner's writes.
        again = [pool.pop() for _ in range(380)]
        assert sorted(again) == sorted(units)
        assert set(again) <= set(pool.take_released_units())

    def test_plain_kv_free_is_not_scrubbed(self):
        pool = scrubbing_pool(10)
        b = pool.pop()
        pool.allocate(b)
        pool.free(b)
        assert pool.take_released_units() == ()

    def test_switch_off_queues_nothing(self):
        pool = BlockPool(100, null_block=False, scrub_released=False)
        units = pool.reserve_units(10, owner="ck")
        pool.release_units(units, owner="ck")
        assert pool.take_released_units() == ()

    def test_release_reserve_release_in_one_pass_is_queued_once(self):
        pool = scrubbing_pool(20)
        units = pool.reserve_units(5, owner="a")
        pool.release_units(units, owner="a")
        again = pool.reserve_units(5, owner="b")
        assert sorted(again) == sorted(units)
        pool.release_units(again, owner="b")
        taken = pool.take_released_units()
        assert sorted(taken) == sorted(units) and len(taken) == 5

    def test_cancel_scrub_for_out_of_band_writers(self):
        pool = scrubbing_pool(20)
        units = pool.reserve_units(5, owner="a")
        pool.release_units(units, owner="a")
        transfer = pool.reserve_units(3, owner="t")
        pool.cancel_scrub(transfer)
        assert set(pool.take_released_units()) == set(units) - set(transfer)


# ------------------ through the checkpoint store eviction ------------------ #
def test_kv_eviction_orphans_a_checkpoint_and_its_units_are_queued_first():
    """The production chain: allocate -> _unindex -> on_evict -> unindex ->
    release_units -> free, with the released units re-handed in the same pass."""
    holder = {}
    pool = FreeOrderPool(
        64,
        null_block=True,
        scrub_released=True,
        on_evict=lambda h: holder["store"].unindex(h),
    )
    spec = PagedStateCheckpointSpec(
        page_unit_bytes=10, slot_bytes=95, layout_id="l", image_bytes=95
    )
    store = holder["store"] = PageUnitCheckpointStore(pool, spec)
    assert store.units_per_checkpoint == 10
    # One KV block published under h, a checkpoint stored at h.
    kv = pool.pop()
    pool.allocate(kv)
    pool.publish(kv, 777, toks(1, 2, 3, 4))
    pool.free(kv)
    assert store.begin_store(777, src_slot=0) is not None
    store.complete_inflight()
    ck_units = set(next(iter(store.records.values())).unit_ids)
    assert 0 not in ck_units
    pool.take_released_units()
    # Fill every vacant block, then one more allocation must evict `kv`.
    while pool.num_free - pool.num_reusable_free:
        pool.allocate(pool.pop())
    b = pool.pop()
    assert b == kv
    pool.allocate(b)  # evicts 777 -> orphans the checkpoint -> releases units
    assert pool.frees_checked == 10
    assert not store.records
    rehanded = [pool.pop() for _ in range(10)]
    assert set(rehanded) == ck_units
    assert set(pool.take_released_units()) == ck_units


# ------------------------ BlockManager carries them ------------------------ #
def test_block_manager_puts_released_units_in_the_batch(monkeypatch):
    monkeypatch.setenv("ATOM_ZERO_EVICTED_BLOCKS", "1")
    bm = BlockManager(MockConfig(num_kvcache_blocks=50))
    assert bm.kv.first_block == 1
    units = bm.kv.reserve_units(7, owner="ck")
    bm.kv.release_units(units, owner="ck")
    ops = bm.take_state_maintenance_ops()
    assert ops.zero_units == tuple(units)
    assert not ops.empty
    assert bm.take_state_maintenance_ops().zero_units == ()


def test_block_manager_switch_off_restores_old_pool(monkeypatch):
    monkeypatch.setenv("ATOM_ZERO_EVICTED_BLOCKS", "0")
    bm = BlockManager(MockConfig(num_kvcache_blocks=50))
    assert bm.kv.first_block == 0
    units = bm.kv.reserve_units(7, owner="ck")
    assert 0 in units
    bm.kv.release_units(units, owner="ck")
    assert bm.take_state_maintenance_ops().zero_units == ()


# --------------------------- the zero kernel (GPU) -------------------------- #
def test_unit_zeroer_zeroes_every_region_of_exactly_those_units():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from atom.model_ops.attentions.pool_layout.mha_page_unit_geometry import (
        mha_page_unit_regions,
    )
    from atom.model_ops.attentions.pool_layout.paged_state_copy import UnitZeroer

    class _Pool:
        def __init__(self, tensors):
            self._t = tensors

        def region_tensors(self):
            return list(self._t.items())

    n = 64

    def region(field):
        if field in ("k", "v"):
            return torch.full((n, 4096), 7, dtype=torch.uint8, device="cuda")
        return torch.full((n, 16), 7, dtype=torch.float32, device="cuda")

    tensors = {
        f"{field}.layer_{layer}": region(field)
        for field in ("k", "v", "k_scale", "v_scale")
        for layer in range(3)
    }
    regions = mha_page_unit_regions([_Pool(tensors)])
    ids = [5, 1, 63, 17]
    UnitZeroer(regions, n, "cuda").launch(ids)
    torch.cuda.synchronize()
    mask = np.zeros(n, bool)
    mask[ids] = True
    for t in tensors.values():
        rows = t.reshape(n, -1)
        assert bool((rows[mask] == 0).all())
        assert bool((rows[~mask] == 7).all())
