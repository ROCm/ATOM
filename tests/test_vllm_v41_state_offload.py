# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1's STATE boundary policy.

Only the decisions whose failure is silent or fatal are pinned here; the byte
movement is `KdaStateTier`/`StateByteCodec`, which this leg reuses unchanged.

* `cap_hit` is the whole correctness argument. A hit restored past the last
  stored boundary hands the forward a cursor the scheduler does not claim, and
  `PagedAttentionCache._report_stale_state` turns that into a dead engine --
  so "nothing claimed" must answer 0, never the uncapped hit.
* `collect_frontier_stores` must offer a boundary exactly when the state for
  it is still in the slot, which is only at the step whose frontier *is* that
  boundary. Everything else is a boundary that no longer exists.
* The cursor guard is what stands between a scheduling change and a wrongly
  keyed image.

No GPU, no vLLM: the views object is faked down to the three methods the
planner calls.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.v41_state import V41BoundaryPlanner

CHUNK = 256
HASH_BLOCK = 256
INTERVAL = 1024


class FakeViews:
    """The three things the planner asks of `V41StateViews`."""

    def __init__(self, depth=4, committed=None):
        self.entry_bytes = 5_276_672
        self._free = list(range(depth))
        self.released = []
        # slot -> the position its cursor reads. Defaults to "exactly where
        # the scheduler says", so a test that cares about the guard sets it.
        self._committed = committed if committed is not None else {}
        self.default_committed = None
        # How many device reads the worker leg made; one per batch, not one
        # per candidate -- the read is a blocking D2H on the compute stream.
        self.cursor_reads = 0

    def acquire_stage(self):
        return self._free.pop() if self._free else None

    def release_stage(self, stage):
        self._free.append(int(stage))
        self.released.append(int(stage))

    def committed_positions(self, slots):
        self.cursor_reads += 1
        return [self._committed.get(s, self.default_committed) for s in slots]


def make_planner(**overrides):
    fields = dict(
        hash_block_size=HASH_BLOCK,
        chunk_size=CHUNK,
        state_interval=INTERVAL,
        max_num_batched_tokens=INTERVAL * 2,
        world_size=1,
    )
    fields.update(overrides)
    return V41BoundaryPlanner(**fields)


def block_hashes(n_tokens):
    """One opaque hash per hash block, the way vLLM hands them over."""
    return [bytes([i % 251]) * 8 for i in range(n_tokens // HASH_BLOCK)]


def make_request(req_id="r0", tokens=INTERVAL * 8):
    # Salt the hashes with the id: two requests over the same prefix share a
    # key by design (that is what dedupe is), so a test about anything else
    # has to give them different prefixes or it measures the dedupe instead.
    salt = req_id.encode()[:1] or b"\x00"
    return SimpleNamespace(
        request_id=req_id,
        block_hashes=[salt + h for h in block_hashes(tokens)],
    )


def sweep(planner, views, frontier, *, req_id="r0", slot=0, request=None):
    request = request or make_request(req_id)
    views.default_committed = frontier
    return planner.collect_frontier_stores({req_id: frontier}, {req_id: request})


# ---- configuration contracts -------------------------------------------


@pytest.mark.parametrize("interval", [CHUNK * 3 + 1, 100])
def test_an_interval_that_does_not_divide_the_chunk_is_refused(interval):
    # A boundary inside a chunk is a boundary the PAGE prefix cannot end on.
    with pytest.raises(ValueError, match="chunk size"):
        make_planner(state_interval=interval)


def test_an_interval_without_a_block_hash_is_refused():
    with pytest.raises(ValueError, match="hash block size"):
        make_planner(hash_block_size=384, chunk_size=128, state_interval=256)


# ---- cap_hit ------------------------------------------------------------


def test_an_unarmed_cap_passes_the_hit_through():
    # The hook is armed per lookup; outside one there is no request to judge.
    assert make_planner().cap_hit(SimpleNamespace(id="r0"), 4096) == 4096


def test_a_hit_with_no_stored_boundary_is_declined_outright():
    """0, never the hit: a longer answer is a dead engine, not a slow one."""
    planner = make_planner()
    request = make_request()
    planner.begin_lookup(request)
    assert planner.cap_hit(SimpleNamespace(id="r0"), INTERVAL * 3 + 7) == 0
    assert planner.stats()["cap_declined"] == 1


def test_a_hit_is_capped_to_the_claimed_boundary():
    planner = make_planner()
    request = make_request()
    hashes = request.block_hashes
    claimed = INTERVAL * 2
    planner._index.note_stored(planner.boundary_hash(hashes, claimed))
    planner.begin_lookup(request)
    # The hit runs past the claimed boundary and is pulled back to it.
    assert planner.cap_hit(SimpleNamespace(id="r0"), INTERVAL * 3 + 99) == claimed
    assert planner.stats()["cap_kept"] == 1


def test_the_cap_descends_in_whole_intervals():
    """Only interval boundaries are stored, so only those may be returned."""
    planner = make_planner()
    request = make_request()
    planner._index.note_stored(planner.boundary_hash(request.block_hashes, INTERVAL))
    planner.begin_lookup(request)
    capped = planner.cap_hit(SimpleNamespace(id="r0"), INTERVAL * 4 + 1)
    assert capped == INTERVAL
    assert capped % INTERVAL == 0


def test_a_cap_armed_for_another_request_declines():
    planner = make_planner()
    planner.begin_lookup(make_request("r0"))
    assert planner.cap_hit(SimpleNamespace(id="r1"), 8192) == 0


def test_a_hit_below_the_first_boundary_is_declined():
    planner = make_planner()
    planner.begin_lookup(make_request())
    assert planner.cap_hit(SimpleNamespace(id="r0"), INTERVAL - 1) == 0


# ---- collect_frontier_stores -------------------------------------------


def test_a_frontier_on_a_boundary_is_offered_once():
    planner, views = make_planner(), FakeViews()
    assert len(sweep(planner, views, INTERVAL)) == 1
    # Same frontier again (two sweeps in one step, or a step that computed
    # nothing): the state has not moved, and neither has the offer.
    assert sweep(planner, views, INTERVAL) == []
    assert planner.stats()["sweep_stores"] == 1


def test_a_frontier_between_boundaries_is_not_offered():
    planner, views = make_planner(), FakeViews()
    assert sweep(planner, views, INTERVAL + 1) == []
    assert planner.stats()["sweep_offered"] == 0


def test_stepping_over_a_boundary_is_counted_not_deferred():
    """The ring has already moved; there is no hole to come back to."""
    planner, views = make_planner(), FakeViews()
    sweep(planner, views, 0)                          # fresh: starts at zero
    sweep(planner, views, INTERVAL)
    assert sweep(planner, views, INTERVAL * 3) != []  # lands on one
    assert planner.stats()["boundary_passed"] == 1  # stepped over 2*INTERVAL
    assert planner.stats()["restored_start"] == 0


def test_a_restored_requests_first_frontier_is_not_counted_as_lost():
    """Its lower boundaries are in the tier, which is why it started there.

    They were once counted as passed, which made a working restore read like
    a budget problem and would send the next person tuning the wrong knob.
    """
    planner, views = make_planner(), FakeViews()
    sweep(planner, views, INTERVAL * 3 + 7)   # arrives mid-prompt, restored
    assert planner.stats()["boundary_passed"] == 0
    assert planner.stats()["restored_start"] == 1


def test_a_budget_that_straddles_boundaries_counts_every_one_it_passes():
    """From the second sighting on, a skipped boundary really is lost."""
    planner, views = make_planner(), FakeViews()
    sweep(planner, views, 0)                   # fresh
    sweep(planner, views, INTERVAL)            # lands, establishes a cursor
    sweep(planner, views, INTERVAL * 4 + 7)    # skips 2..4 x INTERVAL
    assert planner.stats()["boundary_passed"] == 3
    assert planner.stats()["restored_start"] == 0


def test_an_already_stored_boundary_is_not_stored_again():
    planner, views = make_planner(), FakeViews()
    planner._index.note_stored(
        planner.boundary_hash(make_request().block_hashes, INTERVAL)
    )
    assert sweep(planner, views, INTERVAL) == []
    assert planner.stats()["sweep_known"] == 1


def test_a_boundary_past_the_hashed_prefix_is_skipped():
    planner, views = make_planner(), FakeViews()
    short = make_request(tokens=INTERVAL)  # hashes cover one interval only
    assert sweep(planner, views, INTERVAL * 4, request=short) == []
    assert planner.stats()["sweep_no_hash"] == 1


def test_a_refused_snapshot_closes_the_quorum_as_a_failure():
    """A store the worker refused will never report, so it rides back failed.

    Anything else leaves the boundary pending for the life of the process --
    and, worse, leaves it unclaimed-but-not-disowned.
    """
    planner, views = make_planner(world_size=1), FakeViews()
    store, = sweep(planner, views, INTERVAL)
    planner.absorb_reports({}, {store.op_id: 1})
    assert store.op_id not in planner._pending_stores
    assert not planner._index.could_serve(store.prefix_hash)


def test_two_requests_over_one_prefix_store_it_once():
    planner, views = make_planner(), FakeViews()
    shared = make_request("a")
    views.default_committed = INTERVAL
    first = planner.collect_frontier_stores({"a": INTERVAL}, {"a": shared})
    second = planner.collect_frontier_stores(
        {"b": INTERVAL},
        {"b": SimpleNamespace(request_id="b", block_hashes=shared.block_hashes)},
    )
    assert len(first) == 1 and second == []
    assert planner.stats()["sweep_known"] == 1


def test_a_resumed_request_is_offered_at_a_lower_frontier_again():
    """Preemption restarts the request at 0 under the same id.

    Its new life climbs back through frontiers below the old cursor, which the
    "offer each frontier once" rule would otherwise suppress for good.
    """
    planner, views = make_planner(), FakeViews()
    sweep(planner, views, INTERVAL * 4)
    assert sweep(planner, views, INTERVAL) == []  # still the old life
    planner.forget_request("r0")
    assert sweep(planner, views, INTERVAL) != []


def test_a_skipped_request_is_not_offered():
    planner, views = make_planner(), FakeViews()
    request = make_request()
    views.default_committed = INTERVAL
    assert (
        planner.collect_frontier_stores(
            {"r0": INTERVAL}, {"r0": request}, skip_req_ids={"r0"}
        )
        == []
    )


# ---- reports ------------------------------------------------------------


def test_a_boundary_is_claimed_only_once_every_rank_stored_it():
    planner = make_planner(world_size=4)
    views = FakeViews()
    store, = sweep(planner, views, INTERVAL)
    planner.absorb_reports({store.op_id: 3}, {})
    assert not planner._index.could_serve(store.prefix_hash)
    planner.absorb_reports({store.op_id: 1}, {})
    assert planner._index.could_serve(store.prefix_hash)


def test_one_rank_failing_disowns_the_whole_boundary():
    """A state stored on three of four ranks is one `cap_hit` would accept."""
    planner = make_planner(world_size=4)
    views = FakeViews()
    store, = sweep(planner, views, INTERVAL)
    planner.absorb_reports({store.op_id: 3}, {store.op_id: 1})
    assert not planner._index.could_serve(store.prefix_hash)


def test_quorum_counts_failures_so_a_boundary_is_never_stranded():
    """A rank that could not write never sends a second report."""
    planner = make_planner(world_size=2)
    views = FakeViews(depth=2)
    store, = sweep(planner, views, INTERVAL)
    planner.absorb_reports({}, {store.op_id: 2})
    assert store.op_id not in planner._pending_stores


# ---- load ---------------------------------------------------------------


def test_a_load_is_queued_only_for_a_claimed_boundary():
    planner = make_planner()
    request = make_request()
    assert planner.resolve_load(request, INTERVAL * 2) is False
    planner._index.note_stored(planner.boundary_hash(request.block_hashes, INTERVAL * 2))
    assert planner.resolve_load(request, INTERVAL * 2) is True
    assert [load.req_id for load in planner.take_loads()] == ["r0"]


def test_a_non_boundary_frontier_has_no_state_leg():
    planner = make_planner()
    assert planner.resolve_load(make_request(), INTERVAL + 1) is False


# ---- worker leg ---------------------------------------------------------


class FakeSlots:
    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def slot_for(self, key):
        return self._mapping.get(key)


class RecordingTier:
    def __init__(self):
        self.stores = []

    def submit_store(self, store, event):
        self.stores.append(store)


def worker_leg(views, slots):
    from atom.plugin.vllm.kv_transfer.v41_state import V41StateWorkerLeg

    return V41StateWorkerLeg(views, RecordingTier(), FakeSlots(slots))


def decided(planner, views, frontier, req_id="r0"):
    return sweep(planner, views, frontier, req_id=req_id)


def test_a_request_with_no_slot_is_refused_by_the_worker():
    """A decided boundary whose request never reached a batch row."""
    planner, views = make_planner(), FakeViews()
    store, = decided(planner, views, INTERVAL)
    leg = worker_leg(views, {})
    assert leg.snapshot_and_submit([store]) == {"worker_no_slot": [store.op_id]}


def test_a_cursor_that_disagrees_with_the_schedule_is_refused():
    """The hard contract, checked against the bytes rather than the schedule.

    Keying an image to a position it is not at is the one failure nothing
    downstream catches: `cap_hit` would accept the boundary and the forward it
    was restored into would then refuse the request.
    """
    planner, views = make_planner(), FakeViews(committed={0: INTERVAL - 256})
    store, = decided(planner, views, INTERVAL)
    leg = worker_leg(views, {"r0": 0})
    assert leg.snapshot_and_submit([store]) == {"cursor_mismatch": [store.op_id]}


def test_an_exhausted_staging_ring_refuses_rather_than_stalls():
    planner, views = make_planner(), FakeViews(depth=0)
    views.default_committed = INTERVAL
    store, = decided(planner, views, INTERVAL)
    leg = worker_leg(views, {"r0": 0})
    assert leg.snapshot_and_submit([store]) == {"stage_full": [store.op_id]}


def test_a_refused_store_holds_no_staging_slab():
    planner, views = make_planner(), FakeViews(depth=1)
    store, = decided(planner, views, INTERVAL)
    leg = worker_leg(views, {})
    leg.snapshot_and_submit([store])
    # The slab was never taken, so releasing the op is a no-op and the ring is
    # still whole for the next boundary.
    leg.release_reported([store.op_id])
    assert views.acquire_stage() is not None


# ---- local prefix-cache hits -------------------------------------------


def test_a_local_hit_is_refused_even_at_a_claimed_boundary():
    """Shortening a local hit is not enough; nothing restores the state.

    `resolve_load` runs only for tokens the connector supplied, so a locally
    served prefix arrives with its pages in HBM and a slot nobody wrote --
    measured as `needs state at 8192, found 0`, the cap having picked 8192.
    PAGE reuse and STATE restore are one operation here, and only the
    connector performs both.
    """
    planner = make_planner()
    request = make_request()
    claimed = INTERVAL * 2
    planner._index.note_stored(planner.boundary_hash(request.block_hashes, claimed))
    assert planner.cap_local_hit(request, INTERVAL * 3 + 50) == 0
    assert planner.stats()["local_cap_declined"] == 1


def test_a_local_hit_with_no_stored_boundary_is_declined():
    """0, not the hit: an uncapped local hit is EngineDeadError, not latency."""
    planner = make_planner()
    assert planner.cap_local_hit(make_request(), INTERVAL * 3) == 0
    assert planner.stats()["local_cap_declined"] == 1


def test_local_and_external_caps_are_counted_apart():
    """They answer different questions and must not be summed by accident.

    The external one measures the tier. The local one measures how much reuse
    prefix caching is being denied on this model -- which is all of it.
    """
    planner = make_planner()
    request = make_request()
    planner._index.note_stored(planner.boundary_hash(request.block_hashes, INTERVAL))
    planner.cap_local_hit(request, INTERVAL * 2)
    planner.begin_lookup(request)
    planner.cap_hit(SimpleNamespace(id="r0"), INTERVAL * 2)
    stats = planner.stats()
    assert stats["local_cap_declined"] == 1 and stats["cap_kept"] == 1


def test_a_local_hit_needs_no_armed_lookup():
    """Unlike the connector path there is no lookup in progress to read."""
    planner = make_planner()
    assert planner._lookup_ctx is None
    assert planner.cap_local_hit(make_request(), INTERVAL) == 0


def test_the_worker_reads_every_cursor_in_one_device_round_trip():
    """The read is a blocking D2H on the compute stream inside start_load_kv.

    One sync per candidate store puts that stall on the critical path of every
    step that stores anything -- the path this leg exists to stay off. Each
    store is then refused on its cursor, which keeps the assertion about the
    read rather than about what follows it.
    """
    planner = make_planner()
    views = FakeViews(depth=4, committed={0: 1, 1: 2, 2: 3})
    decided = [
        sweep(planner, views, INTERVAL, req_id=r)[0] for r in ("a", "b", "c")
    ]
    leg = worker_leg(views, {"a": 0, "b": 1, "c": 2})
    views.cursor_reads = 0
    refused = leg.snapshot_and_submit(decided)
    assert views.cursor_reads == 1
    assert len(refused["cursor_mismatch"]) == 3
