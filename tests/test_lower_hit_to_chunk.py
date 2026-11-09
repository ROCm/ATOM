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


def test_lowers_an_unaligned_hit_to_the_chunk_floor(seq_factory):
    bm = _manager()
    seq = seq_factory(list(range(64)))
    probe = _probe(5)  # 20 tokens: 4 into the second 16-token chunk
    assert bm.lower_hit_to_chunk(seq, probe, CHUNK, tier_hit=48, min_load=8)
    assert probe.num_cached_blocks == 4  # 16 tokens
    assert probe.num_new_blocks == 7  # one more fresh block for the dropped one


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


def test_store_tokens_accepts_a_pd_multi_connector_with_a_shared_tier():
    from types import SimpleNamespace

    from atom.model_engine.engine_core_mgr import CoreManager

    pd = {"kv_connector": "mooncake", "kv_role": "kv_producer"}
    store = {"kv_connector": "mooncake_store", "kv_role": "offload"}
    mgr = SimpleNamespace(_dp_session_affinity_enabled=False, _dp_lb_req_equiv=512)
    cfg = SimpleNamespace(
        enable_prefix_caching=True,
        kv_transfer_config={"kv_connector": "multi", "connectors": [pd, store]},
    )
    assert CoreManager._build_store_router(mgr, cfg) is not None

    cfg.kv_transfer_config = {"kv_connector": "multi", "connectors": [pd]}
    with pytest.raises(ValueError, match="shares"):
        CoreManager._build_store_router(mgr, cfg)
