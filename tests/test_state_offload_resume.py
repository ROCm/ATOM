# SPDX-License-Identifier: MIT
"""What a *spilled* checkpoint's hit is allowed to become at admission.

`StateGroupPool._resumable_from` may answer True for a hash that lives only in
`StateOffloadIndex.hashes` -- bytes that are in LMCache and not in HBM. That
makes `can_allocate` return a boundary whose state group does not exist yet, so
`BlockManager._attach_state_group` has to decide what a resumer gets. Getting
that wrong is silent wrong output: a recycled group plus `num_cached_tokens > 0`
makes `has_initial_state` True over another request's leftovers.

Two worlds are exercised here, and which one a test wants is the whole point:

  loads unwired (the branch today, `STATE_OFFLOAD_LOADS_WIRED` False) -- an
      offload-only hash is not resumable, so it must not shorten a hit that a
      still-resident HBM checkpoint further left could have served.
  loads wired (`loads_wired`) -- an offload-only hash is a candidate, and the
      `_attach_state_group` guard is what makes a miss behind it safe rather
      than corrupt. LMCache's own LRU can drop bytes at any time, so that miss
      never stops being possible.

Kept out of `test_state_checkpoint.py` because every case here needs the tier
switched on, which that file's fixtures deliberately never do.
"""

from conftest import MockConfig

from atom.model_engine import state_pool
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence

BLOCK = 4
MIN_FORK = 8


def tier_config(**overrides):
    """`ckpt_config`'s shape plus a connector that hosts the state tier."""
    defaults = {
        "kv_cache_block_size": BLOCK,
        "num_kvcache_blocks": 200,
        "enable_prefix_caching": True,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 256,
        "max_model_len": 256,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "stop_token_ids": [],
        "scheduler_delay_factor": 0.0,
        "speculative_config": None,
        "pool_entries": {"state": 4},
        "state_transfer_kind": "fork",
        "state_fork_tokens": MIN_FORK,
        "state_checkpoint_interval_tokens": BLOCK,
        "kv_transfer_config": {
            "kv_connector": "lmcache_offload",
            "kv_role": "offload",
        },
    }
    defaults.update(overrides)
    return MockConfig(**defaults)


def stateful_seq(token_ids):
    return Sequence(token_ids, BLOCK, has_per_req_cache=True)


def loads_wired(monkeypatch) -> None:
    """Put the pool in the world where the load direction exists.

    An offload-only hash votes in `_resumable_from` only there. Tests of the
    `_attach_state_group` guard need this world: with loads unwired the pool
    declines a spilled boundary up front and the guard is never reached.
    """
    monkeypatch.setattr(state_pool, "STATE_OFFLOAD_LOADS_WIRED", True)


def resident_publisher(bm: BlockManager, tokens: list[int]) -> tuple[int, int]:
    """Run a prompt to its checkpoint boundary and leave that checkpoint in HBM.

    Returns `(boundary hash, boundary in tokens)`.
    """
    seq = stateful_seq(tokens)
    hit = bm.can_allocate(seq)
    assert hit >= 0
    bm.allocate(seq, hit)
    boundary = bm.checkpoint_limit(seq)
    assert boundary > 0
    bm.hash_blocks(seq, boundary - seq.num_cached_tokens)
    last = boundary // bm.hash_block_size - 1
    h = bm.kv.block(seq.block_table[last]).hash
    bm.release_state_pins()
    bm.release_state_pins()
    assert bm.state.lookup(h) >= 0, "the publisher kept no checkpoint"
    return h, boundary


def spilled_publisher(bm: BlockManager, tokens: list[int]) -> tuple[int, int]:
    """Run a prompt to its checkpoint boundary, then spill that checkpoint.

    Returns `(boundary hash, boundary in tokens)`. The hash is afterwards in
    the tier's index and nowhere in HBM -- exactly the state the branch reaches
    once a real spill is confirmed by
    `Scheduler._update_from_kv_xfer_finished`. The boundary comes back too so a
    caller can pin the resumer's hit against a figure it did not measure from
    the thing under test.
    """
    h, boundary = resident_publisher(bm, tokens)
    group = bm.state.lookup(h)
    assert group >= 0, "the publisher kept no checkpoint"
    # What `pop()` does to a checkpoint it spends, minus the pool pressure:
    # stage it, then let the engine confirm the bytes landed.
    bm.state._spill(group)
    bm.state.take_spill_copies()
    for pending_hash, slot in bm.state_offload.take_pending():
        bm.state_offload.confirm_spill(pending_hash)
        bm.state_offload.release_staging(slot)
    bm.state.invalidate(group)
    assert bm.state.lookup(h) == -1  # gone from HBM
    assert h in bm.state_offload.hashes  # present in the tier
    return h, boundary


def test_the_tier_is_installed_when_the_connector_hosts_it(monkeypatch):
    """Guard for every test below: without this the tier is None and each of
    them would pass by testing the plain no-tier path."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    bm = BlockManager(tier_config())
    assert bm.state_offload is not None
    assert bm.state.offload is bm.state_offload


def test_a_spilled_hash_is_never_resumed_from_an_unfilled_group(monkeypatch):
    """The Critical, in the form that survives the load path.

    `_resumable_from` accepts a spilled hash, so the hit is non-zero, but
    `self.state.lookup(h)` is HBM-only and misses. Before the guard the miss
    fell through to `self.state.pop()` -- another request's state, with
    `num_cached_tokens > 0` marking it as this request's history.

    The invariant is a disjunction and always was: a positive boundary is legal
    exactly when something is going to fill the group. Either a load was
    requested, or the boundary is disowned. Never a boundary over a group
    nobody filled. The `not` case has its own test
    (`test_a_hash_the_tier_never_had_is_still_disowned`), which is what
    LMCache's LRU produces at any moment.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    spilled_publisher(bm, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    hit = bm.can_allocate(resumer)
    assert hit > 0, "the spilled hash must still produce a hit to be a hazard"
    bm.allocate(resumer, hit)

    assert resumer.per_req_cache_group >= 0
    if resumer.num_cached_tokens:
        assert resumer.state_load_hash != -1, "a boundary over an unfilled group"


def test_the_blocks_of_a_disowned_boundary_are_still_reused(monkeypatch):
    """Declining the *state* resume must not throw away the KV hit's blocks.

    `allocate` claimed them before `_attach_state_group` ran, and dropping them
    would leak a reference. The forward simply recomputes over them.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    _, boundary = spilled_publisher(bm, list(range(40)))

    free_before = bm.kv.num_free
    resumer = stateful_seq(list(range(40)))
    hit = bm.can_allocate(resumer)
    total = bm._dcp_num_blocks(len(resumer))
    # Pin the hit independently. The free-pool assertion below is written in
    # terms of `hit`, so it would hold for whatever `can_allocate` returned --
    # including 0, where "the hit's blocks were reused" is vacuously true
    # because there were none. The publisher only hashed as far as its
    # checkpoint boundary, so that boundary in blocks is the whole hit.
    assert hit == boundary // bm.hash_block_size
    assert 0 < hit < total
    bm.allocate(resumer, hit)
    assert len(resumer.block_table) == total
    # Only the blocks past the hit came off the free pool; the hit's own were
    # claimed out of the index.
    assert bm.kv.num_free == free_before - (total - hit)


def test_an_hbm_checkpoint_is_still_resumed_with_the_tier_on(monkeypatch):
    """The guard must cost the ordinary path nothing: a hash that IS in HBM
    resumes exactly as before, boundary and fork source intact."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    bm = BlockManager(tier_config())
    publisher = stateful_seq(list(range(40)))
    hit = bm.can_allocate(publisher)
    bm.allocate(publisher, hit)
    boundary = bm.checkpoint_limit(publisher)
    bm.hash_blocks(publisher, boundary - publisher.num_cached_tokens)
    h = bm.kv.block(publisher.block_table[boundary // bm.hash_block_size - 1]).hash
    bm.release_state_pins()
    bm.release_state_pins()
    src = bm.state.lookup(h)
    assert src >= 0

    resumer = stateful_seq(list(range(40)))
    bm.allocate(resumer, bm.can_allocate(resumer))
    assert resumer.num_cached_tokens == boundary
    assert resumer.state_fork_src == src
    assert resumer.per_req_cache_group != src


def _two_rungs(bm: BlockManager) -> tuple[int, int]:
    """Publish two checkpoint rungs over a shared prefix, spilling the longer.

    24 and 40 tokens at `MIN_FORK` 8 put rungs at 16 and 32. Returns
    `(short boundary, short boundary's HBM group)`; the rung at 32 is left in
    the tier and nowhere in HBM.
    """
    short_h, short_boundary = resident_publisher(bm, list(range(24)))
    spilled_publisher(bm, list(range(40)))
    short_group = bm.state.lookup(short_h)
    assert short_group >= 0, "the shorter rung must survive in HBM"
    return short_boundary, short_group


def test_a_spilled_rung_does_not_shadow_a_resident_one(monkeypatch):
    """The regression Task 2 introduced, and what the gate exists to stop.

    `resumable_hit` scans right to left and returns the FIRST boundary
    `_resumable_from` accepts. While the tier is write-only the rightmost rung
    is spilled and unreachable, so accepting it ends the scan on a boundary
    `_attach_state_group` then disowns -- and the shorter rung still sitting in
    HBM, which the walk-back would have reached, is never tried. Not wrong
    output (the disown keeps it correct) but a real resume thrown away, and it
    grows with spill volume.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    bm = BlockManager(tier_config(pool_entries={"state": 8}, max_num_seqs=8))
    short_boundary, short_group = _two_rungs(bm)

    resumer = stateful_seq(list(range(40)))
    bm.allocate(resumer, bm.can_allocate(resumer))
    assert resumer.num_cached_tokens == short_boundary
    assert resumer.state_fork_src == short_group
    assert resumer.per_req_cache_group != short_group


def test_the_gate_is_the_only_thing_holding_the_spilled_rung_back(monkeypatch):
    """Same two rungs, loads wired: the rightmost boundary wins, and is kept.

    The control for the test above -- it proves the shorter rung is chosen
    there because the spilled one is *unreachable*, not because the scan
    stopped preferring the right. The pair is what makes the shadowing property
    testable at all, so both halves must survive any change to the gate.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config(pool_entries={"state": 8}, max_num_seqs=8))
    short_boundary, _ = _two_rungs(bm)

    resumer = stateful_seq(list(range(40)))
    hit = bm.can_allocate(resumer)
    assert hit * bm.hash_block_size > short_boundary, "the scan stopped too early"
    bm.allocate(resumer, hit)
    assert resumer.num_cached_tokens == hit * bm.hash_block_size
    assert resumer.state_load_hash != -1, "the longer rung was not fetched"


# ── Admission against a hash the tier can fetch back ───────────────────────


def test_a_spilled_hash_becomes_a_load_and_keeps_its_boundary(monkeypatch):
    """The point of the whole tier. A checkpoint the pool had to spend is
    fetched back into the group the resumer will read, and the boundary the
    scan found survives -- which is the prefill that is not recomputed."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    h, boundary = spilled_publisher(bm, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    bm.allocate(resumer, bm.can_allocate(resumer))

    assert resumer.num_cached_tokens == boundary, "the boundary was disowned"
    assert resumer.state_load_hash == h
    assert resumer.per_req_cache_group >= 0
    # The target is a real pool group, never a staging entry: the bytes have to
    # land where the resuming forward reads them.
    assert resumer.per_req_cache_group < bm.state.num_groups
    assert bm.take_state_loads() == [(resumer.id, h, resumer.per_req_cache_group)]


def test_a_load_target_is_not_forked_from(monkeypatch):
    """The loaded group *is* the incoming state. A fork source would send the
    forward to read some other group instead of the bytes just written."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    spilled_publisher(bm, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    bm.allocate(resumer, bm.can_allocate(resumer))
    assert resumer.state_fork_src == -1


def test_take_state_loads_drains(monkeypatch):
    """Every drain site issues the loads it took; a second reader would submit
    the same transfer twice into a group the first one is already filling."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    spilled_publisher(bm, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    bm.allocate(resumer, bm.can_allocate(resumer))
    assert len(bm.take_state_loads()) == 1
    assert bm.take_state_loads() == []


def test_a_hash_the_tier_never_had_is_still_disowned(monkeypatch):
    """The guard of §6 stays. `_gated_hit` accepts a boundary the tier
    advertises; LMCache's LRU can drop those bytes at any time, and so can a
    stale index entry. The miss must land on the disown, not on a load nobody
    can serve -- a load offered for an unknown hash would park the request
    against a report that never comes.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    h, _ = spilled_publisher(bm, list(range(40)))
    bm.state_offload.forget(h)  # what the LRU does, without the LRU

    resumer = stateful_seq(list(range(40)))
    bm.allocate(resumer, bm.can_allocate(resumer))
    assert resumer.num_cached_tokens == 0
    assert resumer.state_load_hash == -1
    assert bm.take_state_loads() == []


def test_an_aborted_resumer_gives_its_load_back(monkeypatch):
    """A request deallocated with a load outstanding must leave nothing behind:
    the index would otherwise hold a pending entry per aborted request, and its
    counters would drift from what actually happened."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    h, _ = spilled_publisher(bm, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    bm.allocate(resumer, bm.can_allocate(resumer))
    bm.take_state_loads()
    bm.deallocate(resumer)

    assert bm.state_offload.pending_loads == {}
    assert resumer.state_load_hash == -1
    # Abandoning is not failing: the bytes are still there for the next request.
    assert h in bm.state_offload.hashes
    assert bm.state_offload.loads_failed == 0


def test_an_aborted_resumers_group_is_held_until_the_bytes_land(monkeypatch):
    """The transfer does not stop because the request went away.

    A worker thread is still writing that group on its own stream. Handing it
    straight back would let the next admission be given a buffer someone else
    is filling -- another request's state arriving after the fact, under a
    `has_initial_state` that is already true. Held until the report.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    spilled_publisher(bm, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    bm.allocate(resumer, bm.can_allocate(resumer))
    group = resumer.per_req_cache_group
    bm.take_state_loads()
    bm.deallocate(resumer)
    assert not bm.state.is_free(group), "handed out while a transfer writes it"

    bm.settle_state_load(resumer.id, ok=True)
    assert bm.state.is_free(group)
    # Settling an abandoned load moves neither counter: it neither arrived
    # anywhere useful nor proved anything about the bytes.
    assert bm.state_offload.loads_completed == 0
    assert bm.state_offload.loads_failed == 0


def test_a_report_for_a_plain_kv_load_settles_nothing(monkeypatch):
    """State loads share `finished_loading` with KV loads, so every id lands
    here. One the index never issued must move nothing."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    bm.settle_state_load(4242, ok=False)
    assert bm.state_offload.loads_failed == 0


# ── What the disowned request tells the user it got ────────────────────────


def test_a_disowned_request_reports_no_prefix_cache_hit(monkeypatch):
    """`prefix_cache_hit_tokens` must be the hit kept, not the hit offered.

    It is what the OpenAI response reports as
    `prompt_tokens_details.cached_tokens`. A disown means the engine threw the
    boundary away and recomputes it, so claiming a hit there is a false number
    -- and one that disagrees with `CacheStats`, which reads the post-disown
    `num_cached_tokens`. Driven through `Scheduler.schedule` rather than
    `BlockManager.allocate` because the scheduler is where the field is set.

    The tier is made to decline the fetch, which is the general form of every
    reason `_attach_state_group` can fail to produce the state behind an
    accepted boundary. Forgetting the hash instead would not reach the guard at
    all -- the scan would decline the rung first and there would be no boundary
    to disown.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    sched = Scheduler(tier_config(pool_entries={"state": 8}, max_num_seqs=8))
    spilled_publisher(sched.block_manager, list(range(40)))
    monkeypatch.setattr(
        sched.block_manager.state_offload, "request_load", lambda *_: False
    )

    resumer = stateful_seq(list(range(40)))
    sched.add(resumer)
    _, scheduled = sched.schedule()

    # Both assertions below check for 0, which is also the untouched default:
    # a resumer the scheduler never admitted would satisfy them without the
    # disown path running at all. Pin that it was actually scheduled first.
    assert resumer.id in scheduled, "the resumer was never scheduled"
    assert resumer.num_cached_tokens == 0, "the boundary was not disowned"
    assert resumer.prefix_cache_hit_tokens == 0


def test_a_resumed_request_still_reports_its_prefix_cache_hit(monkeypatch):
    """The other half: on the ordinary path the reported hit is unchanged.

    Guards against fixing the disown by simply reporting nothing. The value is
    `seq.num_cached_tokens`, which is `num_cached_blocks * hash_block_size` --
    the same units `CacheStats` uses.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    sched = Scheduler(tier_config(pool_entries={"state": 8}, max_num_seqs=8))
    _, boundary = resident_publisher(sched.block_manager, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    sched.add(resumer)
    sched.schedule()

    assert resumer.num_cached_tokens == boundary, "the boundary was not resumed"
    assert resumer.prefix_cache_hit_tokens == boundary


def test_turning_the_tier_on_warns_that_it_is_write_only(monkeypatch, caplog):
    """An operator must be told the spill buys nothing yet.

    The README says so, but nothing reaches someone who set the env var and is
    watching counters. Delete this test when the load path lands -- together
    with the warning it guards.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    with caplog.at_level("WARNING"):
        bm = BlockManager(tier_config())
    assert bm.state_offload is not None
    assert any(
        "not wired" in r.message and "OFFLOAD_STATE is on" in r.message
        for r in caplog.records
    ), caplog.text


def test_the_write_only_warning_is_silent_when_the_tier_is_off(caplog):
    """The default path stays quiet: no env var, no tier, no warning."""
    with caplog.at_level("WARNING"):
        bm = BlockManager(tier_config())
    assert bm.state_offload is None
    assert "OFFLOAD_STATE is on" not in caplog.text
