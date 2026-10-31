# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 LMCache offload on the vLLM plugin: the MP ride and its guards.

V4.1's state checkpoints ride the PAGE object as a native-state image: a
transfer names the checkpoint group blocks as the image's units. These pin the
decisions that keep a restored prefix and its state at one boundary, and the
scheduler/connector fixes that a two-pass DSpark run depends on.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from atom.plugin.vllm.kv_transfer.connector import AtomLMCacheOffloadConnector
from atom.plugin.vllm.kv_transfer.page_ride import PageRideMixin
from atom.plugin.vllm.kv_transfer.v41_state import V41BoundaryPlanner

# -- planner -----------------------------------------------------------------


def _planner(hash_unit=256, grid=256, *, keep_tail=False, can_serve=False):
    return V41BoundaryPlanner(
        hash_block_size=hash_unit,
        boundary_grid=grid,
        chunk_size=256,
        state_checkpoints_can_serve=can_serve,
        checkpoint_group_ids=(1,),
        prefill_tail=2 if keep_tail else 1,
    )


def _cap(planner, hit, *, num_tokens=4096, armed_for="r"):
    request = SimpleNamespace(request_id=armed_for, num_tokens=num_tokens)
    planner.begin_lookup(request)
    try:
        return planner.cap_hit(SimpleNamespace(id="r"), hit)
    finally:
        planner.end_lookup()


def test_a_hit_is_capped_to_the_page_grid_not_the_hash_unit():
    """With DSpark the hash unit is 32 and the PAGE edge 256."""
    p = _planner(32, 256)
    assert _cap(p, 1312) == 1280
    assert _cap(p, 200) == 0


def test_a_cap_outside_its_lookup_declines():
    """Serving an untested hit restores PAGE with no state behind it."""
    p = _planner()
    assert p.cap_hit(SimpleNamespace(id="r"), 1024) == 0
    assert _cap(p, 1024, armed_for="someone-else") == 0


@pytest.mark.parametrize("keep_tail, expected", [(False, 1024), (True, 768)])
def test_a_speculative_ride_never_resumes_one_token_short(keep_tail, expected):
    """vLLM pads a one-token resume into a verify row; V4.1 refuses the step."""
    assert _cap(_planner(keep_tail=keep_tail), 1024, num_tokens=1025) == expected


def test_a_local_hit_stands_only_where_a_checkpoint_can_restore_it():
    request = SimpleNamespace(request_id="r", num_tokens=4096)
    assert _planner(can_serve=False).cap_local_hit(request, 1280) == 0
    assert _planner(can_serve=True).cap_local_hit(request, 1280) == 1280


@pytest.mark.parametrize("num_tokens, expected", [(1025, 0), (1026, 1024)])
def test_a_speculative_local_hit_keeps_a_prefill_tail(num_tokens, expected):
    request = SimpleNamespace(request_id="r", num_tokens=num_tokens)
    planner = _planner(keep_tail=True, can_serve=True)
    assert planner.cap_local_hit(request, 1024) == expected


@pytest.mark.parametrize(
    "computed, group_blocks, expected",
    [
        (512, ([3, 4], [7, 8]), (8,)),
        (500, ([3, 4], [7, 8]), None),  # not a PAGE edge
        (768, ([3, 4], [7, 8]), None),  # past the allocated rows
        (512, ([3, 4], [7, 0]), None),  # the null block is no destination
    ],
)
def test_a_load_names_its_checkpoint_blocks_or_nothing(
    computed, group_blocks, expected
):
    p = _planner()
    ok = p.resolve_load(
        SimpleNamespace(request_id="r"), computed, group_blocks=group_blocks
    )
    assert ok is (expected is not None)
    assert p._ride_boundaries.get("r", {}).get(computed) == expected


# -- the ride itself ---------------------------------------------------------


def _ride():
    class _P(PageRideMixin):
        hash_block_size = 256
        ride_boundary_grid = 256
        ride_group_ids = (1,)

        def __init__(self):
            self._init_page_ride()

    p = _P()
    block = SimpleNamespace(block_id=42)
    p._pool = SimpleNamespace(
        get_cached_block=lambda h, gids: [block],
        touch=lambda blocks: None,
        blocks={42: block},
    )
    return p, SimpleNamespace(block_hashes=tuple(range(1, 9)))


def test_a_load_rides_only_on_its_recorded_destination():
    """The committed source of a save is somebody else's state."""
    p, request = _ride()
    assert p.take_ride_units(request, "r0", 512, operation=None) is None
    p._record_ride_boundary("r0", 512, (7,))
    assert p.take_ride_units(request, "r0", 512, operation=None) == (7,)


def test_a_save_rides_on_the_committed_source_and_pins_it():
    """Never on a recorded load destination: a failed load leaves it stale."""
    p, request = _ride()
    p._record_ride_boundary("r0", 512, (7,))
    assert p.take_ride_units(request, "r0", 512, operation="op") == (42,)
    assert p.take_ride_units(request, "r0", 500, operation="op2") is None
    assert p._ride_pins == {"op": (42,)}


def test_a_finished_requests_save_still_finds_its_request():
    """The scheduler's SeqView outlives the connector's request map."""
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._requests = {}
    request = SimpleNamespace(request_id="r0")
    assert c._v41_request_of(SimpleNamespace(id="r0", _request=request)) is request
    c._requests = {"r1": request}
    assert c._v41_request_of(SimpleNamespace(id="r1")) is request
    assert c._v41_request_of(SimpleNamespace(id="r2")) is None


def test_an_unridable_finished_save_retires_and_frees_its_lease():
    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase

    s = ChunkedOffloadSchedulerBase.__new__(ChunkedOffloadSchedulerBase)
    seq, other = SimpleNamespace(id="r0"), object()
    s._save_tracker = {"r0": [seq, 256]}
    s._save_lease_blocks = {id(seq): {10, 11, 12}}
    s._save_lease_at = {id(seq): 0.0}
    s._save_lease_owner = {id(seq): seq}
    # An earlier operation of this request still reads block 10.
    s._save_operation_owner = {"op1": seq, "op2": other}
    s._save_operation_blocks = {"op1": {0: 10}, "op2": {0: 11}}
    s._pending_source_safe_releases = []
    s.total_source_safe_released_blocks = 0
    s._retire_finished_save("r0", seq)
    assert "r0" not in s._save_tracker
    assert s._pending_source_safe_releases == [frozenset({11, 12})]
    assert s._save_lease_blocks == {id(seq): {10}}


# -- the MP scheduler ----------------------------------------------------------


class _Pool:
    """vLLM's block pool as the ride uses it: checkpoint blocks by hash."""

    def __init__(self, cached):
        self.cached = dict(cached)
        self.blocks = {
            block_id: SimpleNamespace(block_id=block_id, ref_cnt=1)
            for block_ids in self.cached.values()
            for block_id in block_ids
        }
        self.freed = []

    def get_cached_block(self, block_hash, group_ids):
        block_ids = self.cached.get(block_hash)
        return None if block_ids is None else [self.blocks[b] for b in block_ids]

    def touch(self, blocks):
        pass

    def free_blocks(self, blocks):
        self.freed.append(tuple(block.block_id for block in blocks))


# vLLM's hash of each 256-token block of a 1024-token prompt.
_HASHES = (b"h256", b"h512", b"h768", b"h1024")


def _v41_scheduler(cached):
    """The V4.1 MP scheduler over a pool holding `cached` checkpoints.

    Two checkpoint groups (1, 2), so an image is two units.
    """
    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
    from atom.plugin.vllm.kv_transfer.v41_mp_scheduler import V41StateMPScheduler

    s = V41StateMPScheduler.__new__(V41StateMPScheduler)
    ChunkedOffloadSchedulerBase.__init__(
        s,
        SimpleNamespace(
            kv_cache_block_size=256, kv_transfer_config={"kv_role": "kv_both"}
        ),
        chunk_size=256,
        lookup_client=SimpleNamespace(
            prepare_retrieve=lambda *_a: None, clear_lookup_status=lambda _sid: None
        ),
    )
    s._mp_adapter = SimpleNamespace(end_session=lambda _session: None)
    s._scheduler_deadline_s = 3600.0
    s._transfer_seen_at = {}
    s._finished_frontiers = {}
    s._min_load_tokens = 0
    planner = V41BoundaryPlanner(
        hash_block_size=256, chunk_size=256, checkpoint_group_ids=(1, 2)
    )
    pool = _Pool(cached)
    planner.bind_gpu_block_pool(pool)
    s.bind_planner(planner, lambda seq: seq._request)
    return s, planner, pool


def _prompt(n, *, computed=None):
    request = SimpleNamespace(
        request_id="r0", block_hashes=_HASHES[: n // 256], num_tokens=n
    )
    return SimpleNamespace(
        id="r0",
        num_prompt_tokens=n,
        num_cached_tokens=n if computed is None else computed,
        token_ids=list(range(n)),
        block_table=list(range(10, 10 + n // 256)),
        _request=request,
    )


def _store_landed(s, operation):
    from atom.kv_transfer.disaggregation.types import ConnectorCompletion
    from atom.kv_transfer.offload.chunked_scheduler import DENSE_PAGE_STORE_CHANNEL

    s.connector_completion(
        ConnectorCompletion(DENSE_PAGE_STORE_CHANNEL, operation, True)
    )


def test_a_save_carries_its_checkpoint_units_and_its_pin_ends_with_it():
    """The checkpoint rides as a native-state image, its units pinned while
    the save is live; the build after its terminal report gives them back."""
    from atom.kv_transfer.offload.metadata import NativeStateTransfer

    s, planner, pool = _v41_scheduler({_HASHES[1]: (41, 42)})
    seq = _prompt(512)
    s.update_state_after_alloc(seq)
    [save] = s.build_connector_meta().requests
    assert save.token_ids == list(range(512))
    assert save.native_state == NativeStateTransfer(
        (41, 42), 512, int.from_bytes(b"h512", "big")
    )
    assert planner._ride_pins == {save.save_operation: (41, 42)}

    s.build_connector_meta()
    assert planner._ride_pins and pool.freed == []  # still in flight
    _store_landed(s, save.save_operation)
    assert planner._ride_pins  # a report alone gives nothing back
    s.build_connector_meta()
    assert planner._ride_pins == {} and pool.freed == [(41, 42)]


def test_a_running_save_waits_for_the_checkpoint_at_its_end():
    s, planner, _ = _v41_scheduler({_HASHES[0]: (41, 42)})
    s.update_state_after_alloc(_prompt(512))
    assert s.build_connector_meta().requests == []
    assert s._save_tracker["r0"][1] == 0
    assert planner._ride_pins == {}


@pytest.mark.parametrize(
    "cached, leased",
    [({_HASHES[1]: (41, 42)}, {10, 11}), ({}, {10, 11, 12, 13})],
)
def test_a_finished_save_keeps_every_boundary_whose_checkpoint_exists(cached, leased):
    """1024 tokens, a checkpoint at 512 only: the save ends at 512 rather than
    dropping [0, 1024), and teardown leases only the blocks it reads. With no
    checkpoint at all the save is retired and its lease given back."""
    s, _, _ = _v41_scheduler(cached)
    seq = _prompt(1024)
    s.update_state_after_alloc(seq)
    protected = s.protected_block_ids(seq)
    assert protected == leased
    s.activate_block_leases(seq, protected)
    requests = s.build_connector_meta().requests
    if not cached:
        assert requests == []
        assert "r0" not in s._save_tracker
        assert s.take_source_safe_releases() == [frozenset(leased)]
        return
    [save] = requests
    assert save.token_ids == list(range(512))
    assert save.native_state.unit_ids == (41, 42)
    assert save.native_state.boundary_tokens == 512
    _store_landed(s, save.save_operation)
    assert s.take_source_safe_releases() == [frozenset(leased)]
    assert "r0" not in s._save_tracker


@pytest.mark.parametrize("recorded", [True, False])
def test_a_load_carries_its_recorded_destination_or_is_declined(recorded):
    """Asked before vLLM allocates there is no destination yet, so the park
    question stands; at dispatch a load without one is declined, not sent."""
    from atom.kv_transfer.offload.metadata import LoadSpec, NativeStateTransfer

    s, planner, _ = _v41_scheduler({})
    seq = _prompt(1024, computed=0)
    s._load_specs["r0"] = LoadSpec(0, 512)
    s._lookup_results["r0"] = (seq, 512)
    assert s.should_park_for_load_after_alloc(seq)
    s.update_state_after_alloc(seq)
    if recorded:
        planner._record_ride_boundary("r0", 512, (41, 42))
    loads = [r for r in s.build_connector_meta().requests if r.load_spec]
    if not recorded:
        assert loads == [] and s.declined_loads == {"r0"}
        return
    [load] = loads
    assert load.token_ids == list(range(512))
    assert load.native_state == NativeStateTransfer(
        (41, 42), 512, int.from_bytes(b"h512", "big")
    )
    assert load.native_state.destination_slot is None
    assert planner._ride_pins == {}  # vLLM holds a parked request's blocks


def test_the_v41_layout_publishes_the_native_checkpoint_contract(monkeypatch):
    """The fields native V4.1 publishes, so the worker takes the native layout."""
    import torch

    from atom.kv_transfer.disaggregation.page_region import page_region
    from atom.kv_transfer.disaggregation.types import KVTransferTensors
    from atom.plugin.vllm import deepseek_v41_state_checkpoint as checkpoint

    copier = SimpleNamespace(execute_paged_state_copies=lambda *_a, **_k: None)
    spec = SimpleNamespace(image_bytes=40, units_per_checkpoint=2)
    worker = SimpleNamespace(copier=copier)
    monkeypatch.setattr(
        checkpoint, "current_state_checkpoint_worker", lambda *_a: worker
    )
    monkeypatch.setattr(
        checkpoint, "v41_state_checkpoint_sizing", lambda _c: SimpleNamespace(spec=spec)
    )
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._vllm_config = object()
    planes = [torch.zeros(4, 16, dtype=torch.uint8) for _ in range(2)]
    transfer = KVTransferTensors(
        pages=[page_region(p, semantic_role=f"p{i}") for i, p in enumerate(planes)]
    )
    c._publish_v41_state_checkpoints(transfer, 4)
    assert transfer.paged_state_checkpoint_spec is spec
    assert transfer.execute_paged_state_copies == copier.execute_paged_state_copies
    assert transfer.paged_state_region_count == 2
    assert transfer.native_state_tp_replication_factor == 4

    worker.copier = None
    with pytest.raises(ValueError, match="no checkpoint copier"):
        c._publish_v41_state_checkpoints(transfer, 4)


# -- connector ---------------------------------------------------------------


def _options_stub(
    *,
    v41=True,
    groups=(1,),
    mamba=(),
    policy="recompute",
    prefix_caching=True,
    **options,
):
    stub = SimpleNamespace(
        _config=SimpleNamespace(kv_transfer_config=dict(options)),
        _vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(kv_load_failure_policy=policy),
            cache_config=SimpleNamespace(enable_prefix_caching=prefix_caching),
        ),
        _is_deepseek_v41=v41,
        _v41_checkpoint_group_ids=tuple(groups),
        _mamba_groups=list(mamba),
    )
    stub._offload_backend = AtomLMCacheOffloadConnector._resolve_offload_backend(stub)
    return stub


@pytest.mark.parametrize(
    "options, groups, match",
    [
        ({}, (1,), "MP server only"),
        ({"atom.offload.backend": "mp"}, (), "state checkpoints live"),
        (
            {"atom.offload.backend": "mp", "lmcache.mp.state_transport": "mp"},
            (1,),
            "no longer",
        ),
    ],
)
def test_v41_offload_refusals(options, groups, match):
    with pytest.raises(ValueError, match=match):
        stub = _options_stub(groups=groups, **options)
        AtomLMCacheOffloadConnector._refuse_v41_without_ride(stub)


def test_v41_offload_refuses_the_fail_policy():
    """A declined ride is reported as a failed load; `fail` ends the request."""
    stub = _options_stub(policy="fail", **{"atom.offload.backend": "mp"})
    with pytest.raises(ValueError, match="kv_load_failure_policy='recompute'"):
        AtomLMCacheOffloadConnector._refuse_v41_without_ride(stub)


def test_v41_offload_refuses_a_run_without_prefix_caching():
    """The checkpoint groups exist with the switch alone; nothing fills them."""
    stub = _options_stub(prefix_caching=False, **{"atom.offload.backend": "mp"})
    with pytest.raises(ValueError, match="state checkpoints live"):
        AtomLMCacheOffloadConnector._refuse_v41_without_ride(stub)


@pytest.mark.parametrize("policy, warned", [("fail", True), ("recompute", False)])
def test_other_models_are_told_what_the_fail_policy_does(monkeypatch, policy, warned):
    """V4.1 refuses `fail`; every other model hears it once, from the scheduler."""
    from atom.plugin.vllm.kv_transfer import connector as connector_module

    log = Mock()
    monkeypatch.setattr(connector_module, "logger", log)
    stub = _options_stub(v41=False, policy=policy)
    AtomLMCacheOffloadConnector._warn_unless_recompute(stub)
    assert log.warning.called == warned


def test_v41_rides_its_checkpoint_groups_on_mp():
    stub = _options_stub(**{"atom.offload.backend": "mp"})
    AtomLMCacheOffloadConnector._refuse_v41_without_ride(stub)
    assert AtomLMCacheOffloadConnector._state_rides_mp.fget(stub)


def _alloc_connector():
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._attn_group_id = 0
    c._config = SimpleNamespace(kv_cache_block_size=256)
    c._requests = {}
    c._abandoned_loads = set()
    c._promised_loads, c._detach_after_alloc = {}, {}
    c._kda_planner = None
    c._scheduler = SimpleNamespace(update_state_after_alloc=lambda seq: None)
    seq = SimpleNamespace(num_cached_tokens=0, set_block_table=lambda ids: None)
    c._seqs = SimpleNamespace(get_or_create=lambda request: seq)
    c._v41_planner = SimpleNamespace(admit_load=Mock())
    return c


def test_an_unresolved_ride_does_not_release_the_request_itself():
    """`_settle_promised_loads` withdraws it once; a second report asserts in vLLM."""
    c = _alloc_connector()
    blocks = SimpleNamespace(get_block_ids=lambda: ([3, 4], [7, 8]))
    c.update_state_after_alloc(SimpleNamespace(request_id="r0"), blocks, 512)
    assert c._v41_planner.admit_load.call_args.kwargs["group_blocks"] == (
        [3, 4],
        [7, 8],
    )
    assert c._abandoned_loads == set()


def _promising_connector(parked, declined=()):
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._promised_loads = {"7": [0, 32, 64, parked]}
    c._config = SimpleNamespace(kv_cache_block_size=16)
    c._seqs = {"7": SimpleNamespace(block_table=list(range(100, 116)))}
    c._scheduler = SimpleNamespace(declined_loads=set(declined))
    c._finished_withdrawals = []
    return c


@pytest.mark.parametrize("parked", [True, False])
def test_an_undispatched_promise_is_withdrawn_only_if_vllm_parked_it(parked):
    """An unparked request reported as received trips vLLM's assertion."""
    c = _promising_connector(parked)
    out = []
    for _ in range(c._PROMISE_GRACE_STEPS + 2):
        out.extend(c._settle_promised_loads(SimpleNamespace(requests=[])))
    assert out == ([("7", [102, 103, 104, 105])] if parked else [])
    assert c._promised_loads == {}


def test_a_parked_promise_is_withdrawn_when_its_request_finishes():
    """vLLM frees an aborted parked request's blocks on a `finished_recving`."""
    c = _promising_connector(True)
    c._requests, c._kda_planner, c._detach_after_alloc = {}, None, {}
    c._scheduler = SimpleNamespace(
        declined_loads=set(),
        request_finished=Mock(),
        should_defer_free=Mock(return_value=False),
    )
    c._seqs = SimpleNamespace(
        get=lambda req_id: SimpleNamespace(block_table=list(range(100, 116))),
        drop=lambda req_id: None,
    )
    c._request_finished(SimpleNamespace(request_id="7"))
    out = c._settle_promised_loads(SimpleNamespace(requests=[]))
    assert out == [("7", [102, 103, 104, 105])] and c._promised_loads == {}


def test_a_load_declined_at_dispatch_is_withdrawn_that_step():
    """No later step dispatches it, so the parked request is released now."""
    c = _promising_connector(True, declined={"7"})
    out = c._settle_promised_loads(SimpleNamespace(requests=[]))
    assert out == [("7", [102, 103, 104, 105])] and c._promised_loads == {}


def _lookup_connector(*, pending, detachable):
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    seq = SimpleNamespace(
        id="r",
        num_cached_tokens=512,
        set_num_cached_tokens=Mock(),
        set_block_table=Mock(),
    )
    c._seqs = SimpleNamespace(get_or_create=lambda request: seq)
    c._requests, c._promised_loads, c._detach_after_alloc = {}, {}, {}
    c._kda_planner = c._v41_planner = None
    c._scheduler = SimpleNamespace(
        get_num_new_matched_tokens=Mock(return_value=(512, True)),
        should_park_for_load_after_alloc=Mock(return_value=True),
        prefetch_lookups=Mock(),
        lookup_pending=Mock(return_value=pending),
        can_detach_lookup=Mock(return_value=detachable),
        detach_lookup=Mock(return_value=True),
    )
    return c


@pytest.mark.parametrize(
    "pending, detachable, expected",
    [
        (True, False, (None, False)),
        (False, False, (512, True)),
        (True, True, (0, False)),
    ],
)
def test_a_tier_lookup_never_blocks_schedule(pending, detachable, expected):
    """In flight: ask again next step. Ruled out by the local hit: don't wait."""
    c = _lookup_connector(pending=pending, detachable=detachable)
    assert c.get_num_new_matched_tokens(SimpleNamespace(request_id="r"), 0) == expected


def test_a_detached_lookup_is_sent_once_vllm_has_allocated():
    """A request vLLM cannot allocate for is asked again; no lookup each time."""
    c = _lookup_connector(pending=False, detachable=True)
    request = SimpleNamespace(request_id="r")
    for _ in range(3):
        assert c.get_num_new_matched_tokens(request, 512) == (0, False)
    c._scheduler.detach_lookup.assert_not_called()
    c._scheduler.update_state_after_alloc = Mock()
    c._attn_group_id = 0
    c.update_state_after_alloc(request, ([1, 2],), 0)
    c._scheduler.detach_lookup.assert_called_once()
    assert c._detach_after_alloc == {}


def test_an_admission_that_failed_allocation_does_not_keep_its_detach():
    """Asked again, the request's old frontier must not reach the save floor."""
    c = _lookup_connector(pending=True, detachable=False)
    c._detach_after_alloc["r"] = 512
    assert c.get_num_new_matched_tokens(SimpleNamespace(request_id="r"), 0) == (
        None,
        False,
    )
    assert c._detach_after_alloc == {}


def test_the_lookup_depth_is_one_budget_per_scheduling_pass(monkeypatch):
    """vLLM asks once per waiting request; the depth bounds the whole pass."""
    monkeypatch.setenv("OFFLOAD_ASYNC_LOOKUP_DEPTH", "2")
    sent = []
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._scheduler = SimpleNamespace(
        prefetch_lookups=lambda seqs: sent.extend(s.id for s in seqs),
        lookup_pending=lambda seq: seq.id in sent,
    )
    seqs = [SimpleNamespace(id=str(i)) for i in range(4)]
    # All four are passed over this pass; only two were sent.
    assert all(c._lookup_in_flight(seq) for seq in seqs)
    assert sent == ["0", "1"]
    # One in flight is not sent again and spends nothing.
    c._lookups_sent_this_step = 0
    assert c._lookup_in_flight(seqs[0]) and sent == ["0", "1"]
    assert c._lookup_in_flight(seqs[2]) and sent == ["0", "1", "2"]


def test_a_multi_group_failure_travels_by_request_and_stays_received():
    """0.31 refuses block ids on more than one group and needs the receive quorum."""
    c = object.__new__(AtomLMCacheOffloadConnector)
    c._worker = SimpleNamespace(
        take_load_error_blocks=lambda: {5}, take_load_error_reqs=lambda: {"r0"}
    )
    c._kv_cache_config = SimpleNamespace(kv_cache_groups=(object(), object()))
    c._kda_error_blocks, c._kda_error_reqs = set(), set()
    c._abandoned_error_blocks, c._abandoned_loads = set(), set()
    c._pending_failed_recving = set()
    c.get_finished = lambda ids: ((), {"r0", "r1"})
    assert c.get_block_ids_with_load_errors() == set()
    res = c.get_transfer_results(set())
    assert res.failed_recving == {"r0"}
    assert res.finished_recving == {"r0", "r1"}


# -- vLLM scheduler patches --------------------------------------------------


@pytest.mark.parametrize(
    "prompt, start, budget, expected",
    [
        (65536, 0, 4090, 3840),  # mid-prompt: back to a checkpoint
        (5000, 4096, 904, 768),  # last chunk: stop at the prompt-end anchor
        (65536, 0, 200, 200),  # never cut to nothing
    ],
)
def test_prefill_chunks_end_where_a_checkpoint_can_sit(prompt, start, budget, expected):
    """The checkpoint split alone keeps chunk ends on the grid V4.1 rides.

    The offload connector forces the checkpoint interval to its chunk, 256
    here, so every rung is a PAGE edge.
    """
    from atom.plugin.vllm.paged_state_checkpoint_scheduler import (
        StateCheckpointPlacement,
    )

    placement = StateCheckpointPlacement(interval=256, demand=False, block=256)
    anchor = placement.anchor(prompt, prompt, tail=2)
    cut = placement.cut(start, start + budget, prompt, 0, anchor, tail=2)
    assert (cut - start if cut else budget) == expected


def test_the_prefill_tail_has_one_owner(monkeypatch):
    """The hit caps, the chunk cut and the lookup extent read one number."""
    from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
        register_v41_state_checkpoint_adapter,
    )

    adapter = register_v41_state_checkpoint_adapter()
    spec = SimpleNamespace(speculative_config=object())
    assert adapter.prefill_tail(spec) == 2
    assert adapter.prefill_tail(SimpleNamespace(speculative_config=None)) == 1
    planner = _planner(keep_tail=True)
    assert planner.prefill_tail == adapter.prefill_tail(spec)


def test_a_landed_load_in_kv_holding_waiting_clears_the_drafts():
    """vLLM drains `kv_holding_waiting` first; checking `waiting` alone missed it."""
    pytest.importorskip("vllm")
    from vllm.v1.request import RequestStatus

    from atom.plugin.vllm.v41_speculative import _kv_holding_request_will_prefill

    parked = SimpleNamespace(
        request_id="r1", status=RequestStatus.WAITING_FOR_REMOTE_KVS
    )
    sched = SimpleNamespace(
        kv_holding_waiting=[parked], finished_recving_kv_req_ids={"r1"}
    )
    assert _kv_holding_request_will_prefill(sched)
    sched.finished_recving_kv_req_ids = set()
    assert not _kv_holding_request_will_prefill(sched)


# -- state-checkpoint adapter ------------------------------------------------


def test_the_checkpoint_spec_matches_the_pool_geometry():
    """`StateCopies` refuses a spec that disagrees with the allocated cache."""
    geo_mod = pytest.importorskip(
        "atom.model_ops.attentions.pool_layout.v41_pool_geometry"
    )
    from atom.plugin.vllm import deepseek_v41_state_checkpoint as adapter

    geo = geo_mod.V41PoolGeometry(
        layers=40,
        owners=((2, 2), (8, 2), (14, 2), (20, 2)),
        block_size=256,
        window_size=128,
        head_dim=512,
        index_dim=128,
        history_size=3,
        kv_dtype="bf16",
        speculative_tokens=5,
        layer_ratios=tuple([0, 0] + [2] * 18 + [1] * 20),
        index_topk=512,
    )
    sizing = adapter._with_geometry(
        adapter.V41StateCheckpointSizing(
            num_slots=64,
            max_num_seqs=64,
            page_unit_bytes=int(geo.paged_bytes),
            page_bytes=-(-int(geo.paged_bytes) // 256) * 256,
            slot_bytes=int(geo.state_bytes),
            image_bytes=int(geo.checkpoint_image_bytes),
            layout_id=str(geo.layout_id),
        ),
        geo,
    )
    assert sizing.spec.page_unit_bytes == int(geo.paged_bytes)
    assert sizing.spec.slot_bytes == int(geo.state_bytes)
    assert 0 < sizing.image_bytes < sizing.slot_bytes
    assert sizing.k == -(-sizing.image_bytes // sizing.page_unit_bytes)


@pytest.mark.parametrize(
    "overrides, refused",
    [
        ({}, None),
        ({"mamba_cache_mode": "none"}, "mamba_cache_mode"),
        ({"prefix_match_unit": 32}, "prefix_match_unit"),
        ({"enable_mamba_shared_prefix_checkpoint": True}, "shared_prefix_checkpoint"),
    ],
)
def test_unsupported_vllm_cache_settings_are_refused(overrides, refused):
    from atom.plugin.vllm import deepseek_v41_state_checkpoint as adapter

    cache = {
        "mamba_cache_mode": "align",
        "prefix_match_unit": None,
        "prefix_cache_retention_interval": 0,
        "enable_mamba_shared_prefix_checkpoint": False,
        **overrides,
    }
    check = (
        adapter.register_v41_state_checkpoint_adapter().check_state_checkpoints_supported
    )
    config = SimpleNamespace(cache_config=SimpleNamespace(**cache))
    if refused is None:
        check(config)
    else:
        with pytest.raises(ValueError, match=refused):
            check(config)


# -- checkpoint restores and keeps -------------------------------------------


def _from_source(relpath, name, scope, cls=None):
    """One function from the repository's source, without importing its module.

    Both modules below pull in the GPU stack at import; CPU CI cannot.
    """
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / relpath
    body = ast.parse(path.read_text()).body
    if cls is not None:
        body = next(
            n for n in body if isinstance(n, ast.ClassDef) and n.name == cls
        ).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(  # noqa: S102 - execute only AST nodes from this repository's source.
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope
    )
    return scope[name]


def test_a_mid_prompt_request_entering_another_requests_slot_is_restored():
    """A previous tenant that stopped on the same boundary is not this state."""
    import torch

    scope = {"torch": torch, "_V41_SLOT_OWNERS": {}}
    invalidate = _from_source(
        "atom/plugin/vllm/deepseek_v41_bridge.py",
        "_v41_invalidate_reowned_slots",
        scope,
    )
    cursor = torch.tensor([[256, 0], [256, 0], [128, 0]])
    # Without a restore nothing resumes mid-prompt, so nothing is invalidated.
    invalidate(SimpleNamespace(cursor=cursor), [0], ["z"], [256])
    assert cursor[0, 0].item() == 256 and scope["_V41_SLOT_OWNERS"] == {}
    cache = SimpleNamespace(cursor=cursor, checkpoint_restore=object())
    invalidate(cache, [0, 1, 2], ["a", "b", "c"], [256, 0, 128])
    # New owners mid-prompt lose their cursor; one starting at 0 is fresh.
    assert cache.cursor[:, 0].tolist() == [-1, 256, -1]
    cache.cursor[:, 0] = torch.tensor([256, 256, 128])
    invalidate(cache, [0, 1, 2], ["a", "b", "c"], [256, 300, 128])
    assert cache.cursor[:, 0].tolist() == [256, 256, 128]  # same owners
    invalidate(cache, [0], ["d"], [256])
    assert cache.cursor[0, 0].item() == -1


def test_a_host_resident_group_is_refused_not_skipped():
    """Skipped, its layers would have no cache tensor and nothing saying so."""
    from atom.plugin.vllm.spec_decode_patch import _v41_overlay_split

    group = SimpleNamespace(host_resident=True, layer_names=["model.layers.0.attn"])
    with pytest.raises(NotImplementedError, match="host-resident"):
        _v41_overlay_split([group])


def test_a_returning_request_id_does_not_inherit_its_old_slot():
    """Ids come back (client reuse, no randomization); the record must not."""
    import torch

    scope = {"torch": torch, "_V41_SLOT_OWNERS": {}}
    bridge = "atom/plugin/vllm/deepseek_v41_bridge.py"
    invalidate = _from_source(bridge, "_v41_invalidate_reowned_slots", scope)
    forget = _from_source(bridge, "forget_v41_slot_owners", scope)
    cache = SimpleNamespace(
        cursor=torch.tensor([[256, 0], [256, 0]]), checkpoint_restore=object()
    )
    invalidate(cache, [0, 1], ["a", "b"], [256, 256])
    cache.cursor[:, 0] = 256
    forget({"a"})
    assert scope["_V41_SLOT_OWNERS"] == {1: "b"}
    # A new "a" mid-prompt in slot 0 is restored, not read as the old tenant.
    invalidate(cache, [0, 1], ["a", "b"], [256, 256])
    assert cache.cursor[:, 0].tolist() == [-1, 256]


def test_decode_keeps_are_stored_after_prepare_attn_returns(monkeypatch):
    """The V2 runner commits from `sample_tokens`, after `prepare_attn` has
    returned and the batch getters with it. The keep is still stored, at the
    committed end, from the rows its build recorded."""
    import numpy as np

    from atom.plugin.vllm import paged_state_checkpoint as psc
    from atom.plugin.vllm import req_id_passthrough_patch as passthrough

    worker = SimpleNamespace(
        group_ids=[0], copier=None, decode_candidates={}, decode_rows=None
    )

    class _ModelState:
        def prepare_attn(self, input_batch, cudagraph_mode, block_tables):
            psc.record_decode_candidates(
                worker,
                passthrough.get_current_req_ids(),
                [3, 4],
                input_batch.num_computed_tokens_cpu,
                np.array([6, 6]),
                256,
                "V4.1",
            )

    passthrough._wrap_with_req_id_snapshot(
        _ModelState, "prepare_attn", batch_from_arg=True
    )
    monkeypatch.setattr(psc, "_decode_plan", {"a": 0, "b": 0})
    monkeypatch.setattr(
        psc,
        "state_checkpoint_unit_ids",
        lambda _groups, _rows, cols, _name: [(col,) for col in cols],
    )
    stored = []
    monkeypatch.setattr(
        psc,
        "execute_state_checkpoint_stores",
        lambda _copier, slots, units, descriptor_slot=1: stored.append(
            (slots, units, descriptor_slot)
        ),
    )
    batch = SimpleNamespace(
        req_ids=["a", "b"], num_computed_tokens_cpu=np.array([250, 1020])
    )
    _ModelState().prepare_attn(batch, None, None)
    assert passthrough.get_current_req_ids() is None
    # Six rows each (one token plus five drafts); "a" keeps 6, "b" keeps 4,
    # so "a" commits to 256 and "b" to 1024, block columns 0 and 3.
    psc.store_committed_decode_keeps(worker, np.array([0, 2]))
    # On its own descriptor buffer, apart from the step's boundary stores.
    assert stored == [([3, 4], [(0,), (3,)], 2)]
    assert worker.decode_candidates == {} and worker.decode_rows is None


def test_checkpoints_refuse_async_speculative_scheduling(monkeypatch):
    import sys
    import types

    stub = types.ModuleType("atom.plugin.vllm.deepseek_v41_state_checkpoint")
    stub.v41_state_checkpoints_requested = lambda: True
    stub.register_deepseek_v41_state_checkpoint_layers = Mock()
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    patches = types.ModuleType("atom.plugin.vllm.scheduler")
    patches.v41_scheduler_patches_missing = list
    monkeypatch.setitem(sys.modules, patches.__name__, patches)
    register = _from_source(
        "atom/plugin/vllm/model_wrapper.py", "_maybe_register_v41_state_checkpoints", {}
    )
    spec = SimpleNamespace(speculative_config=object())
    with pytest.raises(ValueError, match="no-async-scheduling"):
        register(
            SimpleNamespace(
                **vars(spec), scheduler_config=SimpleNamespace(async_scheduling=True)
            )
        )
    register(
        SimpleNamespace(
            **vars(spec), scheduler_config=SimpleNamespace(async_scheduling=False)
        )
    )
    stub.register_deepseek_v41_state_checkpoint_layers.assert_called_once()

    # A vLLM whose Scheduler lacks a patched method is refused, not run uncapped.
    patches.v41_scheduler_patches_missing = lambda: ["_mamba_block_aligned_split"]
    with pytest.raises(ValueError, match="_mamba_block_aligned_split"):
        register(
            SimpleNamespace(
                **vars(spec), scheduler_config=SimpleNamespace(async_scheduling=False)
            )
        )


def test_the_hit_cap_forwards_vllms_newer_arguments(monkeypatch):
    """vLLM 0.33 calls `_get_local_prefix_cache_hit(request, max_cache_hit_length)`."""
    import sys
    import types

    from atom.plugin.vllm import scheduler as patches

    seen = []

    class Scheduler:
        def _get_local_prefix_cache_hit(self, request, max_cache_hit_length=None):
            seen.append(max_cache_hit_length)
            return [], 0, None, False

    module = types.ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    patches.apply_vllm_v41_local_hit_cap_patch()
    assert Scheduler()._get_local_prefix_cache_hit(object(), 1024) == (
        [],
        0,
        None,
        False,
    )
    assert seen == [1024]
    assert patches.v41_scheduler_patches_missing() == ["_mamba_block_aligned_split"]


def test_the_pool_hook_leaves_v41_without_checkpoints_to_the_reserve_patch(
    monkeypatch,
):
    """Both withholding the STATE tail would charge the pool for it twice."""
    from atom.plugin.vllm import paged_state_checkpoint as psc
    from atom.plugin.vllm.deepseek_v41_state_checkpoint import (
        register_v41_state_checkpoint_adapter,
    )

    adapter = register_v41_state_checkpoint_adapter()
    assert adapter.sizes_pool_without_checkpoints is False
    monkeypatch.setattr(psc, "adapter_for_config", lambda _config: adapter)
    monkeypatch.setattr(psc, "state_checkpoints_on", lambda _adapter, _config: False)
    configs = [object()]
    out = psc._kv_cache_configs_hook(
        lambda *_args: configs, object(), {adapter.proxy_layer_name: object()}, [0]
    )
    assert out is configs


def test_a_stalled_mp_save_keeps_its_rank_reports():
    """Dropping them would free nothing on MP and make the quorum unreachable."""
    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._deferred_frees, c._deferred_free_at = {"r0"}, {"r0": 0.0}
    c._next_save_reconcile_at = 0.0
    c._save_reports = {"r0": {0, 1}}
    c._scheduler = SimpleNamespace(abandons_saves=False, abandon_save=Mock())
    c._reconcile_stale_saves()
    c._scheduler.abandon_save.assert_not_called()
    assert c._save_reports == {"r0": {0, 1}}


def test_an_idle_streaming_session_does_not_drop_the_drafts():
    from vllm.v1.request import RequestStatus

    from atom.plugin.vllm.v41_speculative import _kv_holding_request_will_prefill

    status = getattr(RequestStatus, "WAITING_FOR_STREAMING_REQ", None)
    if status is None:
        pytest.skip("this vLLM has no streaming-input sessions")
    idle = SimpleNamespace(status=status, request_id="s")
    scheduler = SimpleNamespace(kv_holding_waiting=[idle])
    assert not _kv_holding_request_will_prefill(scheduler)
    resumed = SimpleNamespace(status=RequestStatus.WAITING, request_id="w")
    scheduler.kv_holding_waiting.append(resumed)
    assert _kv_holding_request_will_prefill(scheduler)


def test_the_v41_adapter_matches_only_with_the_switch_on(monkeypatch):
    """Off, no checkpoint group exists for its hooks to size or place."""
    from atom.plugin.vllm import deepseek_v41_bridge
    from atom.plugin.vllm.deepseek_v41_state_checkpoint import _V41_ADAPTER

    monkeypatch.setattr(
        deepseek_v41_bridge, "is_deepseek_v41_vllm_config", lambda _config: True
    )
    monkeypatch.setenv("ATOM_V41_STATE_CHECKPOINTS", "0")
    assert not _V41_ADAPTER.matches(object())
    monkeypatch.setenv("ATOM_V41_STATE_CHECKPOINTS", "1")
    assert _V41_ADAPTER.matches(object())


# -- review round 9 ----------------------------------------------------------


def test_a_withdrawn_promise_names_its_blocks_in_virtual_units():
    """Under DCP one block id covers `kv_cache_block_size * DCP` tokens."""
    c = _promising_connector(True, declined={"7"})
    c._scheduler.virtual_block_size = 32
    assert c._settle_promised_loads(SimpleNamespace(requests=[])) == [("7", [101, 102])]


@pytest.mark.parametrize(
    "promised, external, cancelled",
    [(True, 0, True), (True, 512, False), (False, 0, False)],
)
def test_an_admission_without_the_promised_tokens_cancels_the_load(
    promised, external, cancelled
):
    """vLLM can admit RUNNING with 0 external tokens (skip_reading_prefix_cache,
    a local sub-block tail); the queued load would then report finished_recving
    for a request vLLM never parked."""
    c = _alloc_connector()
    c._scheduler = SimpleNamespace(
        update_state_after_alloc=Mock(), cancel_pending_load=Mock()
    )
    if promised:
        c._promised_loads["r0"] = [0, 0, 512, False]
    blocks = SimpleNamespace(get_block_ids=lambda: ([3, 4], [7, 8]))
    c.update_state_after_alloc(SimpleNamespace(request_id="r0"), blocks, external)
    assert c._scheduler.cancel_pending_load.called is cancelled
    assert ("r0" in c._promised_loads) is (promised and not cancelled)


def test_a_failed_resolve_drops_an_earlier_admissions_destination():
    """The old destination names blocks vLLM freed when the load failed."""
    p = _planner()
    request = SimpleNamespace(request_id="r")
    assert p.resolve_load(request, 512, group_blocks=([3, 4], [7, 8]))
    assert not p.resolve_load(request, 512, group_blocks=([3, 4], [7, 0]))
    assert p.take_ride_units(request, "r", 512, operation=None) is None


@pytest.mark.parametrize("ref_cnt, frees", [(1, True), (2, False)])
def test_only_a_pin_that_holds_a_block_alone_frees_it(ref_cnt, frees):
    """Retiring a pin drops one share; a block its request still holds stays."""
    p, request = _ride()
    p._pool.blocks[42].ref_cnt = ref_cnt
    p.take_ride_units(request, "r0", 512, operation="op")
    assert p.has_ride_pins(max_age_s=10.0)
    assert p.has_ride_pins(max_age_s=10.0, sole_holder=True) is frees


@pytest.mark.parametrize("age_s, pending", [(0.0, True), (60.0, False)])
def test_only_a_recent_deferred_free_holds_off_preemption(age_s, pending):
    """A deferral stuck behind a stalled MP save frees nothing until its
    deadline; counting it would stall every preemption until then."""
    import time

    c = AtomLMCacheOffloadConnector.__new__(AtomLMCacheOffloadConnector)
    c._releases_in_flight = set()
    c._deferred_frees = {"r"}
    c._deferred_free_at = {"r": time.monotonic() - age_s}
    c._kda_planner = c._v41_planner = c._scheduler = None
    assert c.has_pending_block_frees() is pending


@pytest.mark.parametrize("tail, expected", [(1, 4096), (2, 0)])
def test_a_cut_never_leaves_less_than_the_tail(tail, expected):
    """A rung at 4096 of a 4097-token prompt leaves a one-token prefill step
    under speculation; the anchor (3840) is the last usable boundary."""
    from atom.plugin.vllm.paged_state_checkpoint_scheduler import (
        StateCheckpointPlacement,
    )

    placement = StateCheckpointPlacement(interval=256, demand=False, block=256)
    assert placement.cut(3840, 4097, 4097, 0, 3840, tail=tail) == expected


def test_a_one_token_tail_hit_is_asked_again_not_dropped(monkeypatch):
    """A resubmitted N * PAGE + 1 prompt hits one token short under DSpark;
    vLLM is asked again for a hit leaving two instead of recomputing from 0."""
    import sys
    import types

    from atom.plugin.vllm import scheduler as patches

    asked = []

    class Coordinator:
        def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
            asked.append(max_cache_hit_length)
            return ["blocks to 3840"], 3840, 0

    class Manager:
        coordinator = Coordinator()

        def create_kv_cache_blocks(self, computed):
            return ("kv", computed)

    class Scheduler:
        kv_cache_manager = Manager()
        _atom_state_checkpoints = SimpleNamespace(prefill_tail=2)
        connector = SimpleNamespace(
            _v41_planner=SimpleNamespace(cap_local_hit=lambda request, hit: hit)
        )

        def _get_local_prefix_cache_hit(self, request):
            return ["blocks to 4096"], 4096, 0, False

    module = types.ModuleType("vllm.v1.core.sched.scheduler")
    module.Scheduler = Scheduler
    monkeypatch.setitem(sys.modules, module.__name__, module)
    patches.apply_vllm_v41_local_hit_cap_patch()
    request = SimpleNamespace(num_tokens=4097, block_hashes=(1, 2, 3))
    assert Scheduler()._get_local_prefix_cache_hit(request) == (
        ("kv", ["blocks to 3840"]),
        3840,
        0,
        False,
    )
    assert asked == [4095]
