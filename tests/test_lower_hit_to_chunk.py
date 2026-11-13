# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""BlockManager.lower_hit_to_chunk and its use by the early tier load.

A tier loads whole chunks, so an HBM hit that ends inside one is lowered to the
chunk floor and the load starts there, instead of a hand-off prefill first.
"""

from types import SimpleNamespace

import pytest
from conftest import MockConfig
from test_scheduler_early_load import _make

from atom.model_engine.block_manager import AdmissionProbe
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus

HBS = 4  # hash block size of MockConfig
CHUNK = 16  # four hash blocks


def _manager(blocks=100):
    return Scheduler(
        MockConfig(max_num_seqs=8, max_num_batched_tokens=64, num_kvcache_blocks=blocks)
    ).block_manager


def _probe(hit_blocks, new=6):
    return AdmissionProbe(hit_blocks, list(range(100, 100 + hit_blocks + 2)), new)


def _held_by_a_request(bm, *hashes):
    """Make these hashes cached in blocks some request still holds (shared-live)."""
    live = {h: 900 + i for i, h in enumerate(hashes)}
    bm.kv.lookup = lambda h: live.get(h, -1)
    bm.kv.is_used = lambda block_id: block_id in live.values()


def test_lowers_an_unaligned_hit_to_the_chunk_floor(seq_factory):
    bm = _manager()
    seq = seq_factory(list(range(64)))
    probe = _probe(5)  # 20 tokens: 4 into the second 16-token chunk
    assert bm.lower_hit_to_chunk(seq, probe, CHUNK, tier_hit=48, min_load=8)
    assert probe.num_cached_blocks == 4  # 16 tokens
    # The dropped block is cached-free (or gone): `num_new_blocks` already counts
    # it as one to allocate, so lowering adds no demand.
    assert probe.num_new_blocks == 6


def test_a_dropped_block_a_request_still_holds_is_one_more_unit(seq_factory):
    bm = _manager()
    _held_by_a_request(bm, 104)  # the block at index 4, past the floor
    probe = _probe(5)
    assert bm.lower_hit_to_chunk(seq_factory(list(range(64))), probe, CHUNK, 48, 8)
    assert probe.num_new_blocks == 7


@pytest.mark.parametrize(
    "hit_blocks,tier_hit,min_load,why",
    [
        (4, 48, 8, "already on a chunk boundary"),
        (5, 20, 8, "the tier reaches no further than HBM"),
        (5, 22, 8, "from the floor the tier clears less than min_load"),
        (0, 48, 8, "no HBM hit to lower"),
    ],
)
def test_leaves_the_probe_alone(seq_factory, hit_blocks, tier_hit, min_load, why):
    bm = _manager()
    probe = _probe(hit_blocks)
    before = (probe.num_cached_blocks, probe.num_new_blocks)
    assert not bm.lower_hit_to_chunk(
        seq_factory(list(range(64))), probe, CHUNK, tier_hit, min_load
    ), why
    assert (probe.num_cached_blocks, probe.num_new_blocks) == before


def test_refuses_when_the_state_cannot_be_rebuilt_below_the_hit(seq_factory):
    bm = _manager()
    seq = seq_factory(list(range(64)))
    seq.has_per_req_cache = True
    bm.state = SimpleNamespace(transfer=SimpleNamespace(replays=False))
    assert not bm.lower_hit_to_chunk(seq, _probe(5), CHUNK, 48, 8)
    bm.state = SimpleNamespace(transfer=SimpleNamespace(replays=True))
    assert bm.lower_hit_to_chunk(seq, _probe(5), CHUNK, 48, 8)


def test_refuses_with_a_joint_boundary(seq_factory):
    bm = _manager()
    seq = seq_factory(list(range(64)))
    seq.offload_joint.boundary_tokens = 32
    assert not bm.lower_hit_to_chunk(seq, _probe(5), CHUNK, 48, 8)


def test_refuses_when_a_cache_class_rejects_the_lowered_boundary(seq_factory):
    bm = _manager()
    seq = seq_factory(list(range(64)))
    bm._gated_hit = lambda s, boundary, chain: boundary - 1  # window gone
    probe = _probe(5)
    assert not bm.lower_hit_to_chunk(seq, probe, CHUNK, 48, 8)
    assert probe.num_cached_blocks == 5


def test_refuses_when_the_pool_cannot_give_the_extra_block(seq_factory):
    bm = _manager(blocks=4)
    _held_by_a_request(bm, 104)
    seq = seq_factory(list(range(64)))
    probe = _probe(5, new=4)  # the pool has 4 free: one more does not fit
    assert not bm.lower_hit_to_chunk(seq, probe, CHUNK, 48, 8)
    assert probe.num_new_blocks == 4


# -- the early tier load ------------------------------------------------------


def test_early_load_starts_an_unaligned_hit_instead_of_skipping_it(
    monkeypatch, seq_factory
):
    sched, _, (seq,) = _make(monkeypatch, seq_factory)
    sched.kv_connector.chunk_size = CHUNK
    sched.kv_connector.lowers_hbm_hit_to_chunk = True
    sched.kv_connector.held_tier_hit = lambda s: 16

    probe = SimpleNamespace(
        num_cached_blocks=1,  # 4 tokens: inside the first chunk
        block_hashes=[7, 8],
        num_new_blocks=2,
    )
    sched._probe_hbm_prefix = lambda s: probe
    lowered = []
    sched.block_manager.lower_hit_to_chunk = lambda s, p, *a: (
        lowered.append(a) or setattr(p, "num_cached_blocks", 0) or True
    )

    sched._early_load_tier_prefixes()

    assert lowered, "the lowering was asked for"
    assert "unaligned_hbm" not in sched._early_load.skips
    assert seq.id in sched._early_load.entries


def test_a_tier_that_does_not_declare_it_keeps_the_hand_off(monkeypatch, seq_factory):
    sched, _, (seq,) = _make(monkeypatch, seq_factory)
    sched.kv_connector.chunk_size = CHUNK

    class _Hit:
        num_cached_blocks = 1

    sched._probe_hbm_prefix = lambda s: _Hit()
    sched._early_load_tier_prefixes()
    assert seq.status == SequenceStatus.WAITING
    assert sched._early_load.skips["unaligned_hbm"] == 1


def test_share_cache_accepts_a_pd_multi_connector_with_a_shared_tier():
    from types import SimpleNamespace

    from atom.model_engine.engine_core_mgr import CoreManager

    pd = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    store = {
        "kv_connector": "mooncake_store",
        "kv_role": "offload",
        "kv_connector_extra_config": {
            "mooncake_store.master": "127.0.0.1:1",
            "mooncake_store.metadata": "http://127.0.0.1:2/metadata",
        },
    }
    mgr = SimpleNamespace(
        _dp_session_affinity_enabled=False,
        _dp_lb_req_equiv=512,
        _routable_engine_count=2,
    )
    cfg = SimpleNamespace(
        dp_load_balance="share_cache",
        enable_prefix_caching=True,
        kv_transfer_config={"kv_connector": "multi", "connectors": [pd, store]},
    )
    assert CoreManager._build_share_cache_router(mgr, cfg) is not None

    cfg.kv_transfer_config = {"kv_connector": "multi", "connectors": [pd]}
    with pytest.raises(ValueError, match="shares"):
        CoreManager._build_share_cache_router(mgr, cfg)


def test_a_store_transfer_keeps_the_stream_it_began_on():
    """A runner that moves to another stream mid-transfer must not move the
    transfer: the switch happens at `begin_transfer`."""
    from atom.kv_transfer.offload import atom_lmcache_staging as staging

    current = ["A"]
    state = staging._ThreadTransferState.__new__(staging._ThreadTransferState)
    state._provider = lambda: current[0]
    state._pinned_stream = "A"
    state._pack_stream = state._copy_stream = None
    assert state.follows_compute_stream
    assert state.pack_stream == state.copy_stream == "A"
    current[0] = "B"  # a CU-masked prefill picked another stream
    assert state.pack_stream == state.copy_stream == "A"  # mid-transfer: unchanged
    state.begin_transfer()
    assert state.pack_stream == state.copy_stream == "B"


def test_every_other_connector_keeps_its_own_side_streams():
    from atom.kv_transfer.offload import atom_lmcache_staging as staging

    state = staging._ThreadTransferState.__new__(staging._ThreadTransferState)
    state._provider = None
    state._pinned_stream = None
    state._pack_stream, state._copy_stream = "pack", "copy"
    assert not state.follows_compute_stream
    state.begin_transfer()  # nothing to follow
    assert (state.pack_stream, state.copy_stream) == ("pack", "copy")


def test_a_pack_always_waits_on_the_producer_device_side():
    from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector

    # Also a state that follows the forward's stream: the stream may have
    # changed between the producer's record and the job's pin.
    for follows in (False, True):
        waited = []
        state = SimpleNamespace(
            follows_compute_stream=follows,
            pack_stream=SimpleNamespace(wait_event=waited.append),
        )
        BlockGPUConnector._wait_for_save_source(state, "event")
        assert waited == ["event"]
