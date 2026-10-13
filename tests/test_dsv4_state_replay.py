# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded replay for DeepSeek-V4: the field, and the scheduler side.

A hit on the compressed history that reaches further than the nearest state
checkpoint by more than the window's receptive field keeps the whole hit and
recomputes only that field. These tests pin the field the backend reports
(`deepseek_v4_replay`), the admission decision, where the forward starts, which
blocks get published, and where checkpoints may land.

Geometry: block 4, window 3, three trunk layers, so the replay is
3 + 2 * 2 = 7 tokens.
"""

from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.model_ops.attentions.deepseek_v4_replay import state_replay_tokens

BLOCK = 4
REPLAY = 3 + 2 * 2


def _hf(**overrides):
    fields = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "sliding_window": 3,
        "num_hidden_layers": 3,
        "compress_ratios": [4, 128, 4, 0],
        "num_nextn_predict_layers": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _config(**overrides):
    fields = {
        "hf_config": _hf(),
        "speculative_config": None,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _bm(replay=True, **overrides):
    fields = {
        "kv_cache_block_size": BLOCK,
        "num_kvcache_blocks": 64,
        "enable_prefix_caching": True,
        "pool_entries": {"state": 8},
        "pool_entries_per_req": {"state": 1},
        # No grid and no demand rung: nothing places a checkpoint unless a test
        # asks for one, so every hit is "compressed history, no state".
        "state_checkpoint_interval_tokens": -1,
        "state_checkpoint_demand": False,
        "prefill_context_parallel_size": 1,
        "hf_config": _hf(),
    }
    fields.update(overrides)
    spec = PagedStateCheckpointSpec(10, 25, "replay-test", image_bytes=25)
    return BlockManager(
        MockConfig(**fields),
        state_runtime=StateRuntime(
            transfer=StateTransfer.copy(spec.layout_id),
            checkpoint_spec=spec,
            replay_tokens=REPLAY if replay else 0,
        ),
    )


def _seq(n, base=100):
    return Sequence(list(range(base, base + n)), BLOCK, has_per_req_cache=True)


def _prefill_and_finish(bm, seq):
    """Admit `seq`, publish its whole prompt as one forward, and free it: its
    blocks stay hash-indexed (the compressed history) with no checkpoint."""
    hit = bm.can_allocate(seq)
    assert hit >= 0
    assert bm.allocate(seq, hit)
    bm.hash_blocks(seq, seq.num_tokens - seq.num_cached_tokens)
    bm.deallocate(seq)


def test_replay_length_is_the_window_receptive_field(monkeypatch):
    monkeypatch.setenv("ATOM_DSV4_STATE_REPLAY", "1")
    assert state_replay_tokens(_config()) == REPLAY


def test_mtp_counts_as_one_more_layer(monkeypatch):
    monkeypatch.setenv("ATOM_DSV4_STATE_REPLAY", "1")
    config = _config(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=3),
    )
    assert state_replay_tokens(config) == 3 + 3 * 2


def test_a_dspark_draft_counts_as_one_more_layer(monkeypatch):
    # Its three layers each project the same fused target hidden states, so
    # their windows sit one hop above the trunk, as the serial MTP layer's does.
    monkeypatch.setenv("ATOM_DSV4_STATE_REPLAY", "1")
    config = _config(
        speculative_config=SimpleNamespace(method="dspark", num_speculative_tokens=6),
    )
    assert state_replay_tokens(config) == 3 + 3 * 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"hf_config": _hf(architectures=["DeepseekV41ForCausalLM"])},
        {"hf_config": _hf(compress_ratios=[1, 2])},
        {"speculative_config": SimpleNamespace(method="eagle3")},
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
    ],
)
def test_off_where_the_receptive_field_argument_does_not_hold(monkeypatch, overrides):
    monkeypatch.setenv("ATOM_DSV4_STATE_REPLAY", "1")
    assert state_replay_tokens(_config(**overrides)) == 0


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_DSV4_STATE_REPLAY", raising=False)
    assert state_replay_tokens(_config()) == 0


def test_the_manager_takes_the_field_the_backend_reports():
    assert _bm().state_replay_tokens == REPLAY
    assert _bm(replay=False).state_replay_tokens == 0


@pytest.mark.parametrize(
    "overrides",
    [
        # An LMCache MP load would write the claimed blocks a replay reads.
        {"kv_transfer_config": {"kv_connector": "lmcache_mp", "kv_role": "offload"}},
        # Nothing to keep a hit with, and no checkpoints to stand in for.
        {"enable_prefix_caching": False},
    ],
)
def test_off_where_the_manager_cannot_keep_the_hit(overrides):
    assert _bm(**overrides).state_replay_tokens == 0


def test_a_far_hit_without_a_checkpoint_replays():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    hit = bm.can_allocate(seq)
    # 8 published blocks match; no checkpoint anywhere, so the gate alone
    # would admit 0. 32 > 7, so the hit is kept and the last 7 replayed.
    assert hit == 8
    assert (seq.replay_start, seq.replay_end) == (32 - REPLAY, 32)
    assert seq.num_compressed_hit_blocks == 8
    assert bm.replay_start(seq, hit * bm.hash_block_size, hit=True) == 32 - REPLAY
    assert bm.allocate(seq, hit)
    assert seq.num_cached_tokens == 32 - REPLAY
    assert seq.num_hashed_tokens == 32
    # A fresh slot, no restore queued: the state comes from the replay.
    assert seq.state_slot >= 0
    assert not bm.paged_state_checkpoints.restore_queued_for(seq.state_slot)


def test_the_replay_publishes_only_from_the_hit_on():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    hit = bm.can_allocate(seq)
    assert bm.allocate(seq, hit)
    shared = list(seq.block_table[:8])
    before = [(bm.kv.block(b).hash, bm.kv.block(b).ref_count) for b in shared]
    bm.hash_blocks(seq, seq.num_tokens - seq.num_cached_tokens)
    assert [(bm.kv.block(b).hash, bm.kv.block(b).ref_count) for b in shared] == before
    # The two new blocks are published and chain onto the hit.
    assert all(bm.kv.block(b).hash != -1 for b in seq.block_table[8:10])
    assert seq.num_hashed_tokens == 40


def test_a_near_hit_is_left_to_the_checkpoint_gate():
    """Two blocks (8 tokens) beat a 7-token replay; one block does not."""
    bm = _bm()
    _prefill_and_finish(bm, _seq(4))
    seq = _seq(12)
    assert bm.can_allocate(seq) == 0
    assert seq.replay_end == 0
    bm = _bm()
    _prefill_and_finish(bm, _seq(8))
    seq = _seq(12)
    assert bm.can_allocate(seq) == 2
    assert (seq.replay_start, seq.replay_end) == (1, 8)


def test_off_the_gate_decides_as_before():
    bm = _bm(replay=False)
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.can_allocate(seq) == 0
    assert seq.replay_end == 0


def test_no_checkpoint_inside_the_replay():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    seq.checkpoint_end_pos = 28  # pretend an anchor fell inside the replay
    assert bm.checkpointers_at(seq, 28) == []
    assert bm.checkpoint_cut(seq, seq.num_cached_tokens, 40, record=False) != 28


def test_deallocate_and_disown_clear_the_replay():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    assert bm.disown_claimed_prefix(seq)
    assert (seq.replay_start, seq.replay_end) == (0, 0)
    bm.deallocate(seq)
    seq2 = _seq(40)
    assert bm.allocate(seq2, bm.can_allocate(seq2))
    bm.deallocate(seq2)
    assert (seq2.replay_start, seq2.replay_end) == (0, 0)


def test_a_replay_still_demands_a_checkpoint_at_its_end():
    """The replay keeps this admission's reuse; the demand keeps the next one
    from paying the replay again."""
    bm = _bm(state_checkpoint_demand=True)
    # A branch: 32 tokens shared, then a tail of its own, so the first prompt's
    # own end anchors nothing a later prompt reaches.
    _prefill_and_finish(bm, Sequence(list(range(100, 132)) + [900] * 4, BLOCK))
    bm.take_state_maintenance_ops()
    bm.complete_previous_state_batch()
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    assert (seq.replay_start, seq.replay_end) == (32 - REPLAY, 32)
    assert seq.checkpoint_demand_pos == 32
    cuts = []
    while seq.num_cached_tokens < seq.num_prompt_tokens:
        start = seq.num_cached_tokens
        end = seq.num_prompt_tokens
        if target := bm.checkpoint_cut(seq, start, end):
            end = target
            cuts.append(target)
        bm.hash_blocks(seq, end - start, start_tokens=start)
        seq.num_cached_tokens = end
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
    assert cuts == [32, 36]  # the demand, then this prompt's own end
    bm.deallocate(seq)
    # Another branch off the same 32 tokens, which only the demand's
    # checkpoint reaches: it resumes there instead of replaying again.
    branch = Sequence(list(range(100, 132)) + [777] * 8, BLOCK, has_per_req_cache=True)
    assert bm.can_allocate(branch) == 8
    assert branch.replay_end == 0


def test_no_demand_inside_the_replay():
    """A demand below the replay's end is one nothing would cut or keep."""
    bm = _bm(state_checkpoint_demand=True)
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    seq.replay_end = 36  # what a further-reaching replay would have recorded
    bm._record_checkpoint_demand(
        seq,
        hit=0,
        compressed_hit=8,
        block_hashes=[],
        live_blocks=0,
        protected_hash=None,
    )
    assert seq.checkpoint_demand_pos == 0
