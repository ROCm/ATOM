# SPDX-License-Identifier: MIT
"""The V4.1 slot allocator and the recurrent tier must agree on slot zero."""

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector
from atom.plugin.vllm.kv_transfer.kda_state import KdaLoad, KdaStateTier
from atom.plugin.vllm.kv_transfer.v41_state import V41BoundaryPlanner, V41StateViews
from atom.plugin.vllm.state_slot_allocator import StateSlotAllocator


@pytest.fixture
def worker(monkeypatch):
    import atom.plugin.vllm.kv_transfer.v41_state as state
    from atom.kv_transfer.offload.hybrid.kimi_k3 import staging, state_object

    codec = Mock()
    codec.get.return_value = True
    monkeypatch.setattr(staging, "StagedTransfer", Mock())
    monkeypatch.setattr(state_object, "StateByteCodec", Mock(return_value=codec))
    monkeypatch.setattr(
        state,
        "V41StateViews",
        Mock(
            return_value=SimpleNamespace(
                entry_bytes=32, num_slots=2, stage_depth=2, layout_id="test-v41"
            )
        ),
    )
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._is_deepseek_v41 = True
    c._v41_cache = object()
    c._v41_slots = StateSlotAllocator(2)
    c._offload_backend = "inproc"
    c._config = SimpleNamespace(kv_transfer_config={})
    c._worker = SimpleNamespace(
        _engine=SimpleNamespace(
            gpu_connector=SimpleNamespace(
                device="cpu", release_gpu_staging_after_transfer=False
            ),
            storage_manager=object(),
        ),
        _lmcache_metadata=SimpleNamespace(model_name="v41", world_size=4, worker_id=0),
    )
    c._kda_expect = set()
    c._init_v41_state_tier()
    yield c, codec
    c._kda_tier.close()


def _dispatch(c, load):
    c._dispatch_kda_loads(SimpleNamespace(kda_loads=[load]))
    c._kda_tier.wait_for_requests([load.req_id])
    return c._kda_tier.take_load_results()[load.req_id]


def test_first_and_second_reserved_slots_both_reach_codec(worker):
    c, codec = worker
    for slot in range(2):
        rid = f"r{slot}"
        result = _dispatch(c, KdaLoad(rid, 123, (), (11, 12)))
        assert c._v41_slots.slot_for(rid) == slot
        assert result.ok
        assert result.error_block_ids == ()
        codec.get.assert_called_with(123, (slot,))
    assert codec.get.call_count == 2


def test_refused_admission_neither_reserves_nor_restores_slot_zero(worker):
    c, codec = worker
    planner = V41BoundaryPlanner(
        hash_block_size=32,
        boundary_grid=256,
        chunk_size=256,
        state_interval=4096,
        max_num_batched_tokens=8192,
        world_size=4,
    )
    planner.refuse_admission(SimpleNamespace(request_id="refused"), (11, 12))
    (load,) = planner.take_loads()
    result = _dispatch(c, load)
    assert not result.ok
    assert result.error_block_ids == (11, 12)
    assert c._v41_slots.slot_for("refused") is None
    codec.get.assert_not_called()
    assert _dispatch(c, KdaLoad("valid", 123, (), (11, 12))).ok
    assert c._v41_slots.slot_for("valid") == 0


@pytest.mark.parametrize("destination", [(), (0,)])
def test_full_slot_pool_fails_without_touching_another_request(worker, destination):
    c, codec = worker
    assert c._v41_slots.reserve("owner0") == 0
    assert c._v41_slots.reserve("owner1") == 1
    result = _dispatch(c, KdaLoad("no-room", 123, destination, (11, 12)))
    assert not result.ok
    assert result.error_block_ids == (11, 12)
    codec.get.assert_not_called()


@pytest.mark.parametrize("outcome", [False, RuntimeError("storage failed")])
def test_slot_zero_still_reports_codec_failure(worker, outcome):
    c, codec = worker
    if isinstance(outcome, Exception):
        codec.get.side_effect = outcome
    else:
        codec.get.return_value = outcome
    result = _dispatch(c, KdaLoad("r0", 123, (), (11, 12)))
    codec.get.assert_called_once_with(123, (0,))
    assert not result.ok
    assert result.error_block_ids == (11, 12)


@pytest.mark.parametrize("blocks, ok", [((0,), False), ((1, 0), False), ((1, 2), True)])
def test_kda_null_block_is_still_rejected(blocks, ok):
    codec = Mock()
    codec.get.return_value = True
    tier = KdaStateTier(codec)
    try:
        tier.submit_load(KdaLoad("k3", 123, blocks, (11,)))
        tier.wait_for_requests(["k3"])
        assert tier.take_load_results()["k3"].ok is ok
        assert codec.get.call_count == int(ok)
    finally:
        tier.close()


def test_aborted_restore_is_fenced_before_its_slot_is_released(worker, monkeypatch):
    c, codec = worker
    entered, allow_write, written = Event(), Event(), Event()

    def load_bytes(_prefix_hash, _slots):
        entered.set()
        assert allow_write.wait(3), "test did not release the blocked STATE write"
        written.set()
        return True

    codec.get.side_effect = load_bytes
    c._dispatch_kda_loads(SimpleNamespace(kda_loads=[KdaLoad("aborted", 123, (), ())]))
    assert entered.wait(3)
    wait = c._kda_tier.wait_for_requests
    release = c._v41_slots.release_unclaimed

    def fence(req_ids):
        assert list(req_ids) == ["aborted"]
        assert c._v41_slots.slot_for("aborted") == 0
        allow_write.set()
        wait(req_ids)

    def release_after_write(req_id):
        assert written.is_set(), "slot released before the STATE write finished"
        return release(req_id)

    monkeypatch.setattr(c._kda_tier, "wait_for_requests", fence)
    monkeypatch.setattr(c._v41_slots, "release_unclaimed", release_after_write)
    try:
        c._release_v41_slots({"aborted"})
        assert c._v41_slots.slot_for("aborted") is None
        assert c._v41_slots.reserve("next") == 0
    finally:
        allow_write.set()
        wait(["aborted"])


def test_v41_views_really_address_raw_slot_zero():
    import torch

    views = V41StateViews.__new__(V41StateViews)
    views.num_slots = 2
    views._cache = SimpleNamespace(state_bytes=torch.zeros((2, 32), dtype=torch.uint8))
    (dest,) = views.state_entry_views((0,))
    dest.fill_(7)
    assert views._cache.state_bytes[0].tolist() == [7] * 32
    assert views._cache.state_bytes[1].tolist() == [0] * 32


def _ride_connector(resolves: bool):
    """A ride-mode connector whose planner accepts or refuses every boundary."""
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._is_deepseek_v41 = True
    c._offload_backend = "mp"
    c._state_transport = "mp"
    c._v41_checkpoint_group_ids = (1,)
    c._mamba_groups = ()
    c._attn_group_id = 0
    c._requests = {}
    c._abandoned_loads = set()
    c._promised_loads = {}
    c._kda_planner = None
    c._scheduler = SimpleNamespace(update_state_after_alloc=lambda seq: None)
    seq = SimpleNamespace(num_cached_tokens=512, set_block_table=lambda ids: None)
    c._seqs = SimpleNamespace(get_or_create=lambda request: seq)
    c._v41_planner = SimpleNamespace(
        resolve_load=lambda *a, **k: resolves,
        refuse_admission=Mock(),
    )
    return c


def _blocks():
    return SimpleNamespace(get_block_ids=lambda: ([3, 4], [7, 8]))


@pytest.mark.parametrize("external", [0, 256])
def test_a_refused_ride_withdraws_nothing_itself(external):
    """`_abandoned_loads` is reported as `finished_recving`, once per id.

    A refused ride used to withdraw its promise here as well. The scheduler
    already declines the load when `take_ride_state` has no destination, and
    `_check_promised_loads` withdraws the promise it leaves behind -- so the
    request was released twice: once into RUNNING, and the second report hit
    vLLM's `assert RequestStatus.is_finished(req.status)` in
    `_update_from_kv_xfer_finished`. Measured as a bare AssertionError on the
    second gsm8k pass of a DSpark run, with and without external tokens.
    """
    c = _ride_connector(resolves=False)
    c.update_state_after_alloc(SimpleNamespace(request_id="r0"), _blocks(), external)
    c._v41_planner.refuse_admission.assert_called_once()
    assert c._abandoned_loads == set()


def test_a_resolved_ride_withdraws_nothing():
    c = _ride_connector(resolves=True)
    c.update_state_after_alloc(SimpleNamespace(request_id="r0"), _blocks(), 256)
    c._v41_planner.refuse_admission.assert_not_called()
    assert c._abandoned_loads == set()


def _ride_planner():
    from atom.plugin.vllm.kv_transfer.page_ride import PageRideMixin

    class _P(PageRideMixin):
        hash_block_size = 256
        ride_boundary_grid = 256
        ride_group_ids = (1,)

        def __init__(self):
            self._counters = {}
            self._init_page_ride()

    p = _P()
    block = SimpleNamespace(block_id=42)
    p._pool = SimpleNamespace(
        get_cached_block=lambda h, gids: [block],
        touch=lambda blocks: None,
        blocks={42: block},
    )
    return p


def test_a_load_rides_only_on_its_recorded_destination():
    """A load with nothing recorded has no correct destination.

    The committed source blocks a save would carry belong to whoever stored
    that prefix; loading into them is a write into somebody else's state, and
    it also put a load in flight for a request whose admission was refused.
    """
    p = _ride_planner()
    request = SimpleNamespace(block_hashes=tuple(range(1, 9)))
    assert p.take_ride_state(request, "r0", 512, operation=None) is None
    assert p._counters["ride_missing_load_unrecorded"] == 1
    p._record_ride_boundary("r0", 512, (7,))
    got = p.take_ride_state(request, "r0", 512, operation=None)
    assert tuple(got.block_ids) == (7,)


def test_a_save_still_rides_on_the_committed_source():
    p = _ride_planner()
    request = SimpleNamespace(block_hashes=tuple(range(1, 9)))
    got = p.take_ride_state(request, "r0", 512, operation="save-op")
    assert tuple(got.block_ids) == (42,)
    assert p._counters["ride_pinned"] == 1
