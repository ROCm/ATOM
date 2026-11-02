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
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig

from atom.model_engine import state_runtime as state_runtime_mod
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointSpec,
    PageUnitCheckpointStore,
)
from atom.model_engine.state_runtime import (
    GDN_PAGED_LAYOUT_PREFIX,
    StateMaintenanceOps,
    StateRuntime,
    StateTransfer,
)
from atom.utils import envs


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
        with pytest.raises(ValueError, match="null page"):
            pool.allocate(0)
        assert pool.retire_top().retired == 2
        assert pool.retire_top().retired == 1
        assert pool.retire_top() is None

    def test_off_by_default_in_the_class(self):
        pool = BlockPool(4)
        assert pool.pop() == 0

    def test_one_block_pool_has_nothing_to_hand_out(self):
        with pytest.raises(ValueError, match="1-block pool"):
            scrubbing_pool(1)
        assert BlockPool(1).pop() == 0  # switch off: unchanged

    def test_growing_to_only_the_null_page_is_refused(self):
        pool = BlockPool(0, max_blocks=4, null_block=True)
        with pytest.raises(ValueError):
            pool.extend(1)
        assert pool.num_blocks == 0
        assert pool.extend(2) == 2 and pool.pop() == 1

    def test_pool_grown_from_empty_keeps_block_zero_out(self):
        pool = BlockPool(0, max_blocks=4, null_block=True)
        assert pool.extend(4) == 4
        assert sorted(pool.pop() for _ in range(3)) == [1, 2, 3]
        assert not pool.has_free(1)


class TestRawUnitsLeaveOnlyThroughRelease:
    """`release_units` is the only way a raw unit gets back to the free list."""

    def test_free_refuses_a_raw_unit(self):
        pool = scrubbing_pool(8)
        (unit,) = pool.reserve_units(1, owner="ck")
        with pytest.raises(AssertionError, match="use release_units"):
            pool.free(unit)

    def test_clear_index_keeps_raw_units_owned(self):
        pool = scrubbing_pool(8)
        units = pool.reserve_units(3, owner="ck")
        pool.clear_index()
        assert all(pool.is_used(u) for u in units)
        assert not set(units) & set(pool._vacant)
        assert pool.take_released_units() == ()
        pool.release_units(units, owner="ck")
        assert pool.take_released_units() == tuple(units)

    def test_retire_top_refuses_a_raw_top_unit(self):
        pool = scrubbing_pool(4)
        units = pool.reserve_units(3, owner="ck")
        assert pool.num_blocks - 1 in units
        assert pool.retire_top() is None
        assert pool.num_blocks == 4
        assert pool.take_released_units() == ()


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

    def test_cancel_scrub_leaves_drained_zeros_alone(self):
        # A drained zero already rides a built batch; cancelling only edits
        # the queue (see `cancel_scrub` for why that batch runs first).
        pool = scrubbing_pool(20)
        units = pool.reserve_units(5, owner="a")
        pool.release_units(units, owner="a")
        drained = pool.take_released_units()
        transfer = pool.reserve_units(5, owner="t")
        pool.cancel_scrub(transfer)
        assert drained == tuple(units)
        assert pool.take_released_units() == ()


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
def _paged_runtime(layout_id, zeroes=True):
    spec = PagedStateCheckpointSpec(
        page_unit_bytes=10, slot_bytes=95, layout_id=layout_id, image_bytes=95
    )
    return StateRuntime(
        StateTransfer.copy(layout_id), spec, zeroes_released_units=zeroes
    )


def test_block_manager_puts_released_units_in_the_batch(monkeypatch):
    monkeypatch.setenv("ATOM_ZERO_EVICTED_BLOCKS", "1")
    bm = BlockManager(
        MockConfig(num_kvcache_blocks=50),
        state_runtime=_paged_runtime(GDN_PAGED_LAYOUT_PREFIX + "v1:test"),
    )
    assert bm.kv.first_block == 1
    units = bm.kv.reserve_units(7, owner="ck")
    bm.kv.release_units(units, owner="ck")
    ops = bm.take_state_maintenance_ops()
    assert ops.zero_units == tuple(units)
    assert not ops.empty
    assert bm.take_state_maintenance_ops().zero_units == ()


@pytest.mark.parametrize(
    "layout_id", [None, "dsv4-paged-state-v3:test", GDN_PAGED_LAYOUT_PREFIX + "v1"]
)
def test_block_manager_queues_nothing_without_a_zero_kernel(monkeypatch, layout_id):
    # No PAGE checkpoints, or a runtime whose builder has no zero: nothing is
    # queued (a drained id nobody zeroes would only look handled), but block 0
    # is still the null page. Keyed on the capability, not the layout name:
    # even a GDN-named layout queues nothing without it.
    monkeypatch.setenv("ATOM_ZERO_EVICTED_BLOCKS", "1")
    kw = (
        {}
        if layout_id is None
        else {"state_runtime": _paged_runtime(layout_id, zeroes=False)}
    )
    bm = BlockManager(MockConfig(num_kvcache_blocks=50), **kw)
    assert bm.kv.first_block == 1 and not bm.kv.scrub_released
    units = bm.kv.reserve_units(7, owner="ck")
    bm.kv.release_units(units, owner="ck")
    assert bm.take_state_maintenance_ops().zero_units == ()


def test_a_renamed_layout_keeps_scrubbing(monkeypatch):
    monkeypatch.setenv("ATOM_ZERO_EVICTED_BLOCKS", "1")
    runtime = StateRuntime.from_wire(_paged_runtime("renamed-layout-v9").to_wire())
    assert runtime.zeroes_released_units
    bm = BlockManager(MockConfig(num_kvcache_blocks=50), state_runtime=runtime)
    assert bm.kv.scrub_released


def test_zero_capability_needs_a_checkpoint_spec():
    with pytest.raises(ValueError):
        StateRuntime(zeroes_released_units=True)


def test_builder_capability_is_the_zero_override():
    from atom.model_ops.attentions.backends import AttentionMetadataBuilder

    class _NoZero(AttentionMetadataBuilder):
        pass

    class _Zero(_NoZero):
        def zero_page_units(self, unit_ids):
            pass

    for cls in (_NoZero, _Zero):
        cls.__abstractmethods__ = frozenset()
    assert not _NoZero.__new__(_NoZero).zeroes_page_units()
    assert _Zero.__new__(_Zero).zeroes_page_units()


@pytest.mark.parametrize(
    "raw, on",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        ("0", False),
        ("false", False),
        ("off", False),
        ("", False),
    ],
)
def test_switch_parses_like_other_bool_flags(monkeypatch, raw, on):
    monkeypatch.setenv("ATOM_ZERO_EVICTED_BLOCKS", raw)
    assert envs.ATOM_ZERO_EVICTED_BLOCKS is on


def test_switch_defaults_on(monkeypatch):
    monkeypatch.delenv("ATOM_ZERO_EVICTED_BLOCKS", raising=False)
    assert envs.ATOM_ZERO_EVICTED_BLOCKS is True


# ----------------- engine: zero before a KV connector dispatch -------------- #
class _Runner:
    def __init__(self):
        self.calls = []

    def call_func(self, name, *args, **kw):
        self.calls.append((name, args))


def _engine(pool):
    from atom.model_engine.engine_core import EngineCore

    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(block_manager=SimpleNamespace(kv=pool))
    engine.runner_mgr = _Runner()
    return engine


def test_engine_zeroes_batch_and_queued_units_before_a_load():
    pool = scrubbing_pool(40)
    units = pool.reserve_units(4, owner="a")
    pool.release_units(units, owner="a")
    batch = SimpleNamespace(
        state_maintenance_ops=StateMaintenanceOps(zero_units=(30, 31))
    )
    engine = _engine(pool)
    engine._zero_released_units_before_loads(batch)
    assert engine.runner_mgr.calls == [("zero_page_units", ((30, 31, *units),))]
    # build() must not zero the loaded bytes again.
    assert batch.state_maintenance_ops.zero_units == ()
    assert pool.take_released_units() == ()


def test_engine_zeroes_queued_units_for_a_batch_without_ops():
    # A decode pass that scheduled no sequence carries no maintenance ops but
    # can still dispatch loads; an idle tick has no batch at all.
    pool = scrubbing_pool(40)
    units = pool.reserve_units(3, owner="a")
    pool.release_units(units, owner="a")
    engine = _engine(pool)
    engine._zero_released_units_before_loads(
        SimpleNamespace(state_maintenance_ops=None)
    )
    assert engine.runner_mgr.calls == [("zero_page_units", (tuple(units),))]
    engine._zero_released_units_before_loads()
    assert len(engine.runner_mgr.calls) == 1


@pytest.mark.parametrize("switch, usable", [("1", 9), ("0", 10)])
def test_capacity_checks_count_only_usable_blocks(
    monkeypatch, seq_factory, switch, usable
):
    # 10 blocks of 4 tokens. With block 0 the null page only 9 can be handed
    # out, so a 40-token prompt never fits; the frontend ceiling and the
    # scheduler's drop must both say so, or it waits at the head of the queue
    # forever and blocks everything behind it.
    from atom.model_engine.scheduler import Scheduler

    monkeypatch.setenv("ATOM_ZERO_EVICTED_BLOCKS", switch)
    sched = Scheduler(MockConfig(num_kvcache_blocks=10))
    bm = sched.block_manager
    assert bm.kv.num_usable == usable
    assert bm.max_pool_tokens == usable * 4
    fits = seq_factory(list(range(usable * 4)))
    too_long = seq_factory(list(range(usable * 4 + 1)))
    assert sched._unschedulable_reason(fits) is None
    assert "usable pool blocks=" in sched._unschedulable_reason(too_long)
    # A full pool reads as fully used.
    for _ in range(usable):
        bm.kv.allocate(bm.kv.pop())
    assert sched._kv_usage() == 1.0


# ------------------- loads wait for zeros queued before them ------------------ #
class _FakeEvent:
    """A zero's event: `done` is what `query` reports; records waits."""

    def __init__(self, done=False):
        self.done = done
        self.synchronized = False

    def query(self):
        return self.done

    def synchronize(self):
        self.synchronized = True
        self.done = True


@pytest.fixture
def no_cuda_events(monkeypatch):
    """No load-side event may be created: the fence is only the zero's own."""
    from atom.kv_transfer.offload import _offload_common

    def _refuse(*a, **kw):
        raise AssertionError("a load must not record its own event")

    monkeypatch.setattr(_offload_common.torch.cuda, "Event", _refuse)
    monkeypatch.setattr(state_runtime_mod, "_pending_unit_zero", None)


def _dense_load_meta(n_loads):
    from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta

    metadata = LMCacheOffloadMetadata()
    metadata.lookup_requests_in_step = []
    for i in range(n_loads):
        metadata.add_request(
            LMCacheReqMeta(
                req_id=f"r{i}",
                token_ids=list(range(8)),
                block_ids=[1, 2],
                load_spec=SimpleNamespace(hbm_cached_tokens=4, lmcache_cached_tokens=8),
            )
        )
    return metadata


def _dense_conn(submitted):
    from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector

    conn = DenseOffloadConnector.__new__(DenseOffloadConnector)
    conn._do_load, conn._do_save = True, False
    conn._engine = SimpleNamespace(lookup_unpin=lambda _id: None)
    conn._load_executor = SimpleNamespace(submit=lambda *a, **kw: submitted.append(kw))
    conn._track_job = lambda req_id, future: None
    return conn


@pytest.mark.parametrize("zero", [None, "done", "pending"])
def test_dense_loads_wait_only_on_a_pending_zero(no_cuda_events, zero):
    # Dense LMCache loads write from the load thread on their own streams. They
    # carry the last unit zero's event only while the GPU has not finished it;
    # with no zero, or a finished one, there is nothing to wait on.
    event = None if zero is None else _FakeEvent(done=zero == "done")
    state_runtime_mod.note_unit_zero(event)
    submitted = []
    _dense_conn(submitted).start_load_kv(_dense_load_meta(2))
    want = event if zero == "pending" else None
    assert [kw["ready_event"] for kw in submitted] == [want, want]


def test_a_finished_zero_is_forgotten(no_cuda_events):
    from atom.kv_transfer.offload._offload_common import pending_zero_fence

    state_runtime_mod.note_unit_zero(_FakeEvent(done=True))
    assert pending_zero_fence() is None
    assert state_runtime_mod._pending_unit_zero is None


def test_dense_load_waits_on_the_fence_before_its_first_write():
    from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector

    class _Stop(Exception):
        pass

    class _Fence:
        def synchronize(self):
            raise _Stop

    retrieved = []
    conn = DenseOffloadConnector.__new__(DenseOffloadConnector)
    conn.chunk_size = 4
    conn._engine = SimpleNamespace(retrieve=lambda *a, **kw: retrieved.append(a))
    req = _dense_load_meta(1).requests[0]
    with pytest.raises(_Stop):
        conn._do_load_req(req, ready_event=_Fence())
    assert retrieved == []


def test_state_load_writes_only_after_the_fence():
    from atom.kv_transfer.offload.hybrid.kimi_k3.state_tier import StateOffloadTier

    fence = _FakeEvent()
    seen = []

    class _Codec:
        def get(self, h, slot):
            seen.append(fence.synchronized)
            return True

    tier = StateOffloadTier(_Codec())
    try:
        tier.submit_load("fenced", 7, 3, ready_event=fence)
        tier.submit_load("unfenced", 8, 4)
        tier.drain()
        assert seen == [True, True]
        assert tier.get_finished() == ({"fenced", "unfenced"}, set())
    finally:
        tier.shutdown()


@pytest.mark.parametrize("pending", [True, False])
def test_k3_state_loads_carry_only_a_pending_zero(no_cuda_events, pending):
    from atom.kv_transfer.offload.hybrid.kimi_k3.connector import (
        KimiK3OffloadConnector,
    )

    event = _FakeEvent(done=not pending)
    state_runtime_mod.note_unit_zero(event)
    calls = []
    conn = KimiK3OffloadConnector.__new__(KimiK3OffloadConnector)
    conn._do_load = True
    conn._state_tier = SimpleNamespace(
        submit_load=lambda *a, **kw: calls.append(kw["ready_event"])
    )
    conn._start_state_loads(SimpleNamespace(state_loads=[("r0", 7, 3), ("r1", 8, 4)]))
    want = event if pending else None
    assert calls == [want, want]


# --------------------- warm-up units of builders without a zero ---------------- #
def test_zero_units_once_zeroes_exactly_those_units(monkeypatch):
    from atom.model_ops.attentions.pool_layout import paged_state_copy

    launched = []

    class _Zeroer:
        def __init__(self, regions, num_blocks, device):
            launched.append((regions, num_blocks, device))

        def launch(self, ids):
            launched.append(list(ids))

    monkeypatch.setattr(paged_state_copy, "UnitZeroer", _Zeroer)
    paged_state_copy.zero_units_once("regions", range(3), "dev")
    paged_state_copy.zero_units_once("regions", [], "dev")
    assert launched == [("regions", 3, "dev"), [0, 1, 2]]


def test_dsv4_warmup_zeroes_the_units_it_copied_into(monkeypatch):
    dsv4 = pytest.importorskip("atom.model_ops.attentions.deepseek_v4_attn")
    _assert_warmup_zeroes(monkeypatch, dsv4, dsv4.DeepseekV4AttentionMetadataBuilder)


def test_kimi_warmup_zeroes_the_units_it_copied_into(monkeypatch):
    gdn = pytest.importorskip("atom.model_ops.attentions.gdn_attn")
    kimi = pytest.importorskip("atom.model_ops.attentions.kimi_mla_gdn_attn")
    _assert_warmup_zeroes(monkeypatch, gdn, kimi.KimiTritonMLAGDNMetadataBuilder)


def _assert_warmup_zeroes(monkeypatch, module, cls):
    # The warm-up copies slot 0 into units 0..n-1, block 0 among them. A
    # builder with no `zero_page_units` must zero them through its regions.
    events = []
    monkeypatch.setattr(
        module, "launch_copy_descriptor", lambda *a: events.append("copy")
    )
    monkeypatch.setattr(
        module,
        "zero_units_once",
        lambda regions, ids, device: events.append(("zero", list(ids), regions)),
    )
    spec = SimpleNamespace(units_per_checkpoint=3)
    builder = cls.__new__(cls)
    assert not builder.zeroes_page_units()
    builder.model_runner = SimpleNamespace(
        state_runtime=SimpleNamespace(checkpoint_spec=spec)
    )
    staging = SimpleNamespace(rows=lambda i: np.zeros((4, 8)), upload=lambda *a: None)
    plan = SimpleNamespace(num_spans=2, write_descriptor=lambda *a, **kw: None)
    builder._checkpoint_copy_plan = lambda: plan
    builder._checkpoint_staging = lambda: staging
    builder._checkpoint_slot_bases = lambda: np.zeros((1, 2))
    builder._page_unit_bases = lambda ids: np.zeros((1, 2))
    builder._page_unit_regions = lambda: "regions"
    builder._checkpoint_descriptor_device = lambda: "dev"
    builder.warmup_per_req_cache()
    assert events == ["copy", ("zero", [0, 1, 2], "regions")]


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
    # More launches than the pinned ring has buffers, so a buffer is reused.
    zeroer = UnitZeroer(regions, n, "cuda")
    batches = [[5, 1], [63], [17, 40, 41], [2], [33], [9, 10]]
    assert len(batches) > UnitZeroer.RING
    for batch in batches:
        zeroer.launch(batch)
    torch.cuda.synchronize()
    ids = [i for batch in batches for i in batch]
    mask = np.zeros(n, bool)
    mask[ids] = True
    for t in tensors.values():
        rows = t.reshape(n, -1)
        assert bool((rows[mask] == 0).all())
        assert bool((rows[~mask] == 7).all())
