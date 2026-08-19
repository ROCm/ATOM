# SPDX-License-Identifier: MIT
# The joint load: a hybrid may reuse a prefix the HBM cache no longer holds,
# with the paged KV coming from LMCache and the state checkpoint from the
# offload tier. Both legs are aimed at ONE boundary, and everything here is
# about that boundary being the same number on both sides -- a state boundary
# above the KV-loaded length is silent wrong output, not an error.

import sys
from pathlib import Path

import pytest
from conftest import MockConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))

from atom.model_engine.block_manager import BlockManager  # noqa: E402
from atom.model_engine.sequence import Sequence  # noqa: E402
from atom.model_engine.state_runtime import (  # noqa: E402
    StateRuntime,
    StateTransfer,
)

BLOCK = 4
# The LMCache chunk. A multiple of the hash block, which is what lets one
# boundary satisfy both granularities at all.
CHUNK = 8
MIN_FORK = 8
RUNTIME = StateRuntime(transfer=StateTransfer.fork(MIN_FORK))


def joint_config(**overrides):
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
        "pool_entries": {"state": 8},
        "state_checkpoint_interval_tokens": BLOCK,
        "kv_transfer_config": {
            "kv_connector": "lmcache_offload",
            "kv_role": "offload",
        },
    }
    defaults.update(overrides)
    return MockConfig(**defaults)


@pytest.fixture
def tier_on(monkeypatch):
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "2")
    monkeypatch.setenv("OFFLOAD_KV_FOR_HYBRID", "1")


def make_bm():
    """A BlockManager with the tier on and the KV leg's chunk pinned to CHUNK.

    Production reads that chunk out of the LMCache config in `__init__`;
    `LMCacheEngineConfig.from_env()` caches per process, so setting the env from
    a fixture is already too late and would make these tests depend on which
    file pytest imported first. Pinned here instead -- the gate under test is
    what the number feeds, not where it came from.
    """
    bm = BlockManager(joint_config(), state_runtime=RUNTIME)
    bm._joint_chunk_tokens = CHUNK
    return bm


def hybrid_seq(num_tokens=32):
    return Sequence(list(range(num_tokens)), BLOCK, has_per_req_cache=True)


def chain_hash(bm: BlockManager, seq: Sequence, blocks: int) -> int:
    """The chained content hash of block `blocks - 1`."""
    h = -1
    for i in range(blocks):
        h = bm.compute_hash(bm._hash_block_tokens(seq, i), h)
    return h


def spilled_checkpoint(bm: BlockManager, h: int) -> None:
    """Put `h` in the tier and nowhere else, the state's half of the setup."""
    slot = bm.state_offload.request_spill(h, 0)
    assert slot >= 0
    bm.state_offload.confirm_spill(h)
    assert h in bm.state_offload.hashes


def hbm_checkpoint(bm: BlockManager, h: int, slot: int = 0) -> None:
    """Put `h` in the HBM state pool and nowhere else -- `spilled_checkpoint`'s
    other half, and the case aiming only at the tier used to walk past."""
    bm.state._index(h, slot)
    assert bm.state.lookup(h) >= 0
    assert h not in bm.state_offload.hashes


def admit(bm: BlockManager, seq: Sequence, *, lmc_tokens: int) -> int:
    """One admission with LMCache reporting `lmc_tokens` of KV."""
    seq.offload_kv_prefix_tokens = lmc_tokens
    seq.offload_kv_chunk_tokens = CHUNK
    return bm.can_allocate(seq)


def test_a_boundary_only_lmcache_can_reach_becomes_a_joint_load(tier_on):
    """Neither leg is in HBM: the KV is in LMCache and the checkpoint is in the
    tier, which is the ordinary shape of an evicted prefix -- `unindex` spills
    the state of a checkpoint whose blocks left HBM."""
    bm = make_bm()
    seq = hybrid_seq()
    h = chain_hash(bm, seq, 4)  # tokens [0, 16)
    spilled_checkpoint(bm, h)

    hit = admit(bm, seq, lmc_tokens=16)

    assert hit == 0, "nothing is in the HBM prefix cache"
    assert seq.state_joint_boundary_tokens == 16
    assert seq.state_joint_boundary_hash == h

    bm.allocate(seq, hit)

    assert seq.state_load_hash == h, "the state leg is a tier load"
    assert seq.num_cached_tokens == 0, (
        "the boundary is not claimed until the KV leg lands too; a forward over "
        "a claimed-but-unfilled prefix is the silent failure this exists to "
        "avoid"
    )


def test_the_flag_off_leaves_the_old_hbm_only_boundary(monkeypatch):
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "2")
    monkeypatch.setenv("OFFLOAD_KV_FOR_HYBRID", "0")
    bm = BlockManager(joint_config(), state_runtime=RUNTIME)  # flag off: real ctor
    seq = hybrid_seq()
    spilled_checkpoint(bm, chain_hash(bm, seq, 4))

    hit = admit(bm, seq, lmc_tokens=16)

    assert hit == 0
    assert seq.state_joint_boundary_tokens == 0
    bm.allocate(seq, hit)
    assert seq.state_load_hash == -1


def test_a_boundary_off_the_chunk_grid_still_pairs(tier_on):
    """A rung at 12 tokens with CHUNK=8 is not on the KV leg's grid. Declining
    it would throw away the reuse this feature exists for, so the KV leg is
    aimed at the chunk that covers it (16) while the claim stays 12."""
    bm = make_bm()
    seq = hybrid_seq()
    off_grid = chain_hash(bm, seq, 3)  # tokens [0, 12)
    spilled_checkpoint(bm, off_grid)

    hit = admit(bm, seq, lmc_tokens=16)

    assert hit == 0
    assert seq.state_joint_boundary_tokens == 12
    assert seq.state_joint_kv_tokens == 16


def test_a_covering_chunk_lmcache_does_not_hold_is_declined(tier_on):
    """Same rung, but LMCache stops at 12: the chunk covering the boundary is
    not there, so the pair cannot form. `P <= L` is the invariant; since
    `lmc_tokens` is floored to the chunk grid before `cap` is derived, it is
    now `cap` that excludes the rung rather than the covering-chunk check
    rejecting it, and the outcome asserted here is the invariant either way."""
    bm = make_bm()
    seq = hybrid_seq()
    spilled_checkpoint(bm, chain_hash(bm, seq, 3))  # tokens [0, 12)

    hit = admit(bm, seq, lmc_tokens=12)

    assert hit == 0
    assert seq.state_joint_boundary_tokens == 0


def test_a_boundary_the_tier_cannot_serve_is_declined(tier_on):
    """LMCache has the KV but nobody has the state, so the pair cannot form."""
    bm = make_bm()
    seq = hybrid_seq()

    hit = admit(bm, seq, lmc_tokens=16)

    assert hit == 0
    assert seq.state_joint_boundary_tokens == 0


def test_the_boundary_never_exceeds_what_lmcache_holds(tier_on):
    """`P <= L`, checked where the boundary is chosen: the tier holding a
    further checkpoint does not make its KV reachable."""
    bm = make_bm()
    seq = hybrid_seq()
    spilled_checkpoint(bm, chain_hash(bm, seq, 6))  # tokens [0, 24)
    spilled_checkpoint(bm, chain_hash(bm, seq, 4))  # tokens [0, 16)

    hit = admit(bm, seq, lmc_tokens=16)

    assert hit == 0
    assert seq.state_joint_boundary_tokens == 16


def test_a_checkpoint_still_in_hbm_can_carry_a_joint_boundary(tier_on):
    """The state leg does not have to come from the tier.

    `unindex` drops a checkpoint when its OWN block leaves the block index, but
    the prefix walk stops at the first miss anywhere in the chain, so an
    earlier block going is enough to leave a live HBM checkpoint above the HBM
    boundary -- exactly the subset `unindex` documents as the one it does not
    reclaim. That pair is the cheapest joint load there is: the KV comes from
    LMCache and the state costs no transfer at all.
    """
    bm = make_bm()
    seq = hybrid_seq()
    h = chain_hash(bm, seq, 4)  # tokens [0, 16)
    hbm_checkpoint(bm, h)

    hit = admit(bm, seq, lmc_tokens=16)

    assert hit == 0, "the KV prefix walk reaches nothing"
    assert seq.state_joint_boundary_tokens == 16
    assert seq.state_joint_boundary_hash == h

    bm.allocate(seq, hit)

    assert seq.state_load_hash == -1, "forked in HBM, so no tier load was asked for"
    assert seq.state_fork_src >= 0


def test_the_knob_defaults_on(monkeypatch):
    """Unset is on. The pairing this file tests is the default configuration,
    not something a benchmark has to opt into."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "2")
    monkeypatch.delenv("OFFLOAD_KV_FOR_HYBRID", raising=False)
    bm = make_bm()
    seq = hybrid_seq()
    spilled_checkpoint(bm, chain_hash(bm, seq, 4))

    assert admit(bm, seq, lmc_tokens=16) == 0
    assert seq.state_joint_boundary_tokens == 16


# ---------------------------------------------------------------------------
# Free rungs versus paid ones
#
# `_gated_hit` returns the rightmost rung `_resumable_from` accepts and cannot
# see what it costs to reach: an HBM checkpoint forks, a spilled one is an
# entry-sized H2D plus a park. `OFFLOAD_STATE_TIER_MARGIN_TOKENS` is what makes
# the difference expressible, and the walk is what makes it actionable.
# ---------------------------------------------------------------------------


def _near_pair(bm: BlockManager, seq: Sequence):
    """A tier rung at 20 tokens with a free HBM rung 4 tokens below it."""
    tier = chain_hash(bm, seq, 5)  # tokens [0, 20)
    free = chain_hash(bm, seq, 4)  # tokens [0, 16)
    spilled_checkpoint(bm, tier)
    hbm_checkpoint(bm, free, slot=1)
    return tier, free


def test_without_a_margin_the_rightmost_rung_wins_even_when_it_is_paid(tier_on):
    """The default, and what every measurement so far was taken under."""
    bm = make_bm()
    seq = hybrid_seq()
    tier, _free = _near_pair(bm, seq)

    admit(bm, seq, lmc_tokens=24)

    assert seq.state_joint_boundary_tokens == 20
    assert seq.state_joint_boundary_hash == tier
    assert (bm.joint_boundaries_tier, bm.joint_boundaries_hbm) == (1, 0)
    assert bm.joint_tier_demoted == 0


def test_a_margin_prefers_a_free_rung_the_paid_one_barely_beats(monkeypatch):
    """4 tokens of extra prefix does not pay for an entry-sized H2D and a park,
    so the walk keeps going and takes the checkpoint that costs nothing."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "2")
    monkeypatch.setenv("OFFLOAD_STATE_TIER_MARGIN_TOKENS", "8")
    bm = make_bm()
    seq = hybrid_seq()
    _tier, free = _near_pair(bm, seq)

    hit = admit(bm, seq, lmc_tokens=24)

    assert seq.state_joint_boundary_tokens == 16
    assert seq.state_joint_boundary_hash == free
    assert (bm.joint_boundaries_tier, bm.joint_boundaries_hbm) == (0, 1)
    assert bm.joint_tier_demoted == 1

    bm.allocate(seq, hit)
    assert seq.state_load_hash == -1, "the free rung forks; nothing is fetched"


def test_a_margin_a_paid_rung_clears_still_takes_the_paid_one(monkeypatch):
    """The knob declines a bad trade, not every trade."""
    monkeypatch.setenv("OFFLOAD_STATE", "1")
    monkeypatch.setenv("OFFLOAD_STATE_STAGING_GROUPS", "2")
    monkeypatch.setenv("OFFLOAD_STATE_TIER_MARGIN_TOKENS", "4")
    bm = make_bm()
    seq = hybrid_seq()
    tier, _free = _near_pair(bm, seq)

    admit(bm, seq, lmc_tokens=24)

    assert seq.state_joint_boundary_tokens == 20
    assert seq.state_joint_boundary_hash == tier
    assert bm.joint_tier_demoted == 0


def test_an_admission_with_no_rung_above_hbm_says_so(tier_on):
    """The walk's own decline used to be a bare `return 0` -- a silent zero
    indistinguishable from a build that never ran the feature."""
    bm = make_bm()
    seq = hybrid_seq()

    admit(bm, seq, lmc_tokens=16)

    assert seq.state_joint_boundary_tokens == 0
    assert bm.joint_skips["no_rung_above_hbm"] == 1
    assert bm.joint_boundaries == 0
