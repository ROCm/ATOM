# SPDX-License-Identifier: MIT
"""What a *spilled* checkpoint's hit is allowed to become at admission.

`StateGroupPool._resumable_from` answers True for a hash that lives only in
`StateOffloadIndex.hashes` -- bytes that are in LMCache and not in HBM. That
makes `can_allocate` return a boundary whose state group does not exist yet, so
`BlockManager._attach_state_group` has to decide what a resumer gets. Getting
that wrong is silent wrong output: a recycled group plus `num_cached_tokens > 0`
makes `has_initial_state` True over another request's leftovers.

Kept out of `test_state_checkpoint.py` because every case here needs the tier
switched on, which that file's fixtures deliberately never do.
"""

from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
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


def spilled_publisher(bm: BlockManager, tokens: list[int]) -> int:
    """Run a prompt to its checkpoint boundary, then spill that checkpoint.

    Returns the boundary hash, which afterwards is in the tier's index and
    nowhere in HBM -- exactly the state the branch reaches once a real spill is
    confirmed by `Scheduler._update_from_kv_xfer_finished`.
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

    Nothing loads those bytes back yet, so the only correct answer is to
    decline the resume and recompute.
    """
    monkeypatch.setenv("OFFLOAD_STATE", "1")
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
