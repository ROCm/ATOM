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


def spilled_publisher(bm: BlockManager, tokens: list[int]) -> int:
    """Run a prompt to its checkpoint boundary, then spill that checkpoint.

    Returns the boundary hash, which afterwards is in the tier's index and
    nowhere in HBM -- exactly the state the branch reaches once a real spill is
    confirmed by `Scheduler._update_from_kv_xfer_finished`.
    """
    h, _ = resident_publisher(bm, tokens)
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
    return h


def test_the_tier_is_installed_when_the_connector_hosts_it(monkeypatch):
    """Guard for every test below: without this the tier is None and each of
    them would pass by testing the plain no-tier path."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    bm = BlockManager(tier_config())
    assert bm.state_offload is not None
    assert bm.state.offload is bm.state_offload


def test_a_spilled_hash_is_not_resumed_from_a_recycled_group(monkeypatch):
    """The Critical. `_resumable_from` accepts the spilled hash, so the hit is
    non-zero, but `self.state.lookup(h)` is HBM-only and misses. Before the
    guard the miss fell through to `self.state.pop()` -- another request's
    state, with `num_cached_tokens > 0` marking it as this request's history.

    Driven in the loads-wired world, which is the only one where the hazard is
    reachable and the only one where it must stay guarded: LMCache's LRU can
    drop bytes under a hash the index still advertises, so a load that finds
    nothing lands in exactly this branch. With loads unwired the pool declines
    the boundary before `_attach_state_group` sees it -- covered by
    `test_a_spilled_rung_does_not_shadow_a_resident_one`.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    spilled_publisher(bm, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    hit = bm.can_allocate(resumer)
    assert hit > 0, "the spilled hash must still produce a hit to be a hazard"
    bm.allocate(resumer, hit)

    # Either the boundary is disowned or the bytes are really there. Never a
    # positive boundary over a group nobody filled.
    assert resumer.per_req_cache_group >= 0
    assert (
        resumer.num_cached_tokens == 0
    ), "resumed from a recycled group: the state is another request's"


def test_the_blocks_of_a_disowned_boundary_are_still_reused(monkeypatch):
    """Declining the *state* resume must not throw away the KV hit's blocks.

    `allocate` claimed them before `_attach_state_group` ran, and dropping them
    would leak a reference. The forward simply recomputes over them.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config())
    spilled_publisher(bm, list(range(40)))

    free_before = bm.kv.num_free
    resumer = stateful_seq(list(range(40)))
    hit = bm.can_allocate(resumer)
    bm.allocate(resumer, hit)
    assert len(resumer.block_table) == bm._dcp_num_blocks(len(resumer))
    # Only the blocks past the hit came off the free pool; the hit's own were
    # claimed out of the index.
    assert bm.kv.num_free == free_before - (bm._dcp_num_blocks(len(resumer)) - hit)


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
    """Same two rungs, loads wired: the rightmost boundary wins again.

    The control for the test above -- it proves the shorter rung is chosen
    because the spilled one is *unreachable*, not because the scan stopped
    preferring the right. Re-widening is this one flag, and this test is what
    says so. The disown here is the guard doing its job over bytes that have
    not been fetched in this test; with a real load path the boundary is kept.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    bm = BlockManager(tier_config(pool_entries={"state": 8}, max_num_seqs=8))
    short_boundary, _ = _two_rungs(bm)

    resumer = stateful_seq(list(range(40)))
    hit = bm.can_allocate(resumer)
    assert hit * bm.hash_block_size > short_boundary, "the scan stopped too early"
    bm.allocate(resumer, hit)
    assert resumer.num_cached_tokens == 0


# ── What the disowned request tells the user it got ────────────────────────


def test_a_disowned_request_reports_no_prefix_cache_hit(monkeypatch):
    """`prefix_cache_hit_tokens` must be the hit kept, not the hit offered.

    It is what the OpenAI response reports as
    `prompt_tokens_details.cached_tokens`. A disown means the engine threw the
    boundary away and recomputes it, so claiming a hit there is a false number
    -- and one that disagrees with `CacheStats`, which reads the post-disown
    `num_cached_tokens`. Driven through `Scheduler.schedule` rather than
    `BlockManager.allocate` because the scheduler is where the field is set.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    loads_wired(monkeypatch)
    sched = Scheduler(tier_config(pool_entries={"state": 8}, max_num_seqs=8))
    spilled_publisher(sched.block_manager, list(range(40)))

    resumer = stateful_seq(list(range(40)))
    sched.add(resumer)
    sched.schedule()

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
