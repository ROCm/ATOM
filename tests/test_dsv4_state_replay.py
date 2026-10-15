# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bounded replay for DeepSeek-V4: the field, and the scheduler side.

A hit on the compressed history that reaches further than the nearest state
checkpoint by more than the window's receptive field keeps the whole hit and
recomputes only that field. These tests pin the field `Config` takes
(`deepseek_v4_replay`), the admission decision, where the forward starts (the
scheduler's `_resume_prefill`, so the forward is V4.1's `Replay`), which blocks
get published, and where checkpoints may land.

Geometry: block 4, window 3, three trunk layers, so the replay is
3 + 2 * 2 = 7 tokens.
"""

from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.scheduler import (
    Scheduler,
    _cut_floor,
    _scheduled_replay_until,
)
from atom.model_engine.sequence import Replay, Sequence
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
        "enable_prefix_caching": True,
        "kv_transfer_config": None,
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
        "state_replay_tokens": REPLAY if replay else 0,
    }
    fields.update(overrides)
    spec = PagedStateCheckpointSpec(10, 25, "replay-test", image_bytes=25)
    return BlockManager(
        MockConfig(**fields),
        state_runtime=StateRuntime(
            transfer=StateTransfer.copy(spec.layout_id),
            checkpoint_spec=spec,
        ),
    )


def _resume(bm, seq, hit):
    """Start `seq`'s next forward as the scheduler does: at `replay_start`."""
    scheduler = SimpleNamespace(block_manager=bm, replay_history_tokens=0)
    start = bm.replay_start(seq, seq.num_cached_tokens, hit=hit)
    Scheduler._resume_prefill(scheduler, seq, start, hit=hit)


def _engine_with_a_replayed_head(budget):
    """A scheduler whose cache holds a 32-token prefix with no checkpoint, and
    a waiting 40-token prompt that replays it."""
    spec = PagedStateCheckpointSpec(10, 25, "replay-test", image_bytes=25)
    engine = Scheduler(
        MockConfig(
            kv_cache_block_size=BLOCK,
            num_kvcache_blocks=64,
            enable_prefix_caching=True,
            pool_entries={"state": 8},
            pool_entries_per_req={"state": 1},
            state_checkpoint_interval_tokens=-1,
            state_checkpoint_demand=False,
            hf_config=_hf(),
            state_replay_tokens=REPLAY,
            max_num_batched_tokens=budget,
        ),
        state_runtime=StateRuntime(
            transfer=StateTransfer.copy(spec.layout_id), checkpoint_spec=spec
        ),
    )
    _prefill_and_finish(engine.block_manager, _seq(32))
    seq = _seq(40)
    engine.add(seq)
    return engine, seq


def _prefill_with_cuts(bm, seq):
    """Prefill an allocated `seq` as the scheduler does: start at
    `replay_start`, cut chunks where checkpoints go (`checkpoint_cut` above
    `scheduler._cut_floor`), drain the state work after each forward. Returns
    the cuts."""
    _resume(bm, seq, hit=True)
    cuts = []
    while seq.num_cached_tokens < seq.num_prompt_tokens:
        start = seq.num_cached_tokens
        end = seq.num_prompt_tokens
        floor = _cut_floor(seq, start, start)
        if target := bm.checkpoint_cut(seq, floor, end):
            end = target
            cuts.append(target)
        _forward(bm, seq, end)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        _resume(bm, seq, hit=False)
    return cuts


def _forward(bm, seq, end):
    """Run `seq`'s forward up to `end` and publish what it completed."""
    start = seq.num_cached_tokens
    bm.hash_blocks(seq, end - start, start_tokens=start)
    seq.num_cached_tokens = end


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
        # Nothing to keep a hit with.
        {"enable_prefix_caching": False},
        # A load that knows nothing of the replay would write the claimed
        # blocks it reads; LMCache MP offload alone is the one that knows.
        {"kv_transfer_config": {"kv_connector": "moriio", "kv_role": "kv_both"}},
        {"kv_transfer_config": {"kv_connector": "lmcache_offload"}},
        # P/D plus LMCache MP offload: the P/D leg's loads do not know either.
        {
            "kv_transfer_config": {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                    {"kv_connector": "lmcache_mp", "kv_role": "offload"},
                ],
            }
        },
    ],
)
def test_off_where_a_hit_cannot_be_replayed(monkeypatch, overrides):
    monkeypatch.setenv("ATOM_DSV4_STATE_REPLAY", "1")
    assert state_replay_tokens(_config(**overrides)) == 0


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_DSV4_STATE_REPLAY", raising=False)
    assert state_replay_tokens(_config()) == 0


def test_the_manager_takes_the_field_from_config():
    assert _bm().state_replay_tokens == REPLAY
    assert _bm(replay=False).state_replay_tokens == 0


def test_a_far_hit_without_a_checkpoint_replays():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    hit = bm.can_allocate(seq)
    # 8 published blocks match; no checkpoint anywhere, so the gate alone
    # would admit 0. 32 > 7, so the hit is kept and the last 7 replayed.
    assert hit == 8
    assert seq.replays_hit
    assert seq.num_compressed_hit_blocks == 8
    assert bm.replay_start(seq, hit * bm.hash_block_size, hit=True) == 32 - REPLAY
    assert bm.allocate(seq, hit)
    assert seq.num_cached_tokens == 32
    # A fresh slot, no restore queued: the state comes from the replay.
    assert seq.state_slot >= 0
    assert not bm.paged_state_checkpoints.restore_queued_for(seq.state_slot)
    _resume(bm, seq, hit=True)
    assert seq.num_cached_tokens == 32 - REPLAY
    assert seq.replay == Replay(32, True, (), 32 - REPLAY)


def test_the_replay_publishes_only_from_the_hit_on():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    _resume(bm, seq, hit=True)
    shared = list(seq.block_table[:8])
    before = [(bm.kv.block(b).hash, bm.kv.block(b).ref_count) for b in shared]
    _forward(bm, seq, 40)
    assert [(bm.kv.block(b).hash, bm.kv.block(b).ref_count) for b in shared] == before
    # The two new blocks are published and chain onto the hit.
    assert all(bm.kv.block(b).hash != -1 for b in seq.block_table[8:10])
    assert seq.num_hashed_tokens == 40


def test_a_replay_spans_chunks():
    """A forward ending inside the replay publishes and keeps nothing, and the
    next one continues the replay: nothing written below the hit, no window
    row read below where the replay started."""
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    _resume(bm, seq, hit=True)
    seq.checkpoint_end_pos = 28  # pretend an anchor fell inside the replay
    _forward(bm, seq, 28)
    assert seq.num_hashed_tokens == 0
    assert seq.last_checkpoint_pos == 0
    _resume(bm, seq, hit=False)
    assert seq.num_cached_tokens == 28
    assert seq.replay == Replay(32, True, (), 32 - REPLAY)
    _forward(bm, seq, 40)
    assert seq.num_hashed_tokens == 40
    # Past the hit the replay is over.
    _resume(bm, seq, hit=False)
    assert seq.replay is None


def test_a_near_hit_is_left_to_the_checkpoint_gate():
    """Two blocks (8 tokens) beat a 7-token replay; one block does not."""
    bm = _bm()
    _prefill_and_finish(bm, _seq(4))
    seq = _seq(12)
    assert bm.can_allocate(seq) == 0
    assert not seq.replays_hit
    bm = _bm()
    _prefill_and_finish(bm, _seq(8))
    seq = _seq(12)
    assert bm.can_allocate(seq) == 2
    assert seq.replays_hit
    assert bm.replay_start(seq, 8, hit=True) == 1


def test_off_the_gate_decides_as_before():
    bm = _bm(replay=False)
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.can_allocate(seq) == 0
    assert not seq.replays_hit


def test_no_checkpoint_cut_inside_the_replay():
    """No forward is cut below the hit (`scheduler._cut_floor`): an anchor
    inside the replay is skipped, one at the hit is taken."""
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    _resume(bm, seq, hit=True)
    floor = _cut_floor(seq, seq.num_cached_tokens, seq.num_cached_tokens)
    assert floor == 31
    seq.checkpoint_end_pos = 28
    assert bm.checkpoint_cut(seq, floor, 40, record=False) != 28
    seq.checkpoint_end_pos = 32
    assert bm.checkpoint_cut(seq, floor, 40, record=False) == 32


def test_a_readmission_decides_afresh():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    _resume(bm, seq, hit=True)
    bm.deallocate(seq)  # preempted
    bm2 = _bm()
    assert bm2.can_allocate(seq) == 0
    assert not seq.replays_hit


def test_a_replayed_hit_demands_no_checkpoint():
    """The replay bounds what the missing checkpoint costs; a demand would cut
    every replay admission at the hit for a branch there that may not come.
    Such a branch replays again."""
    bm = _bm(state_checkpoint_demand=True)
    # A branch: 32 tokens shared, then a tail of its own, so the first prompt's
    # own end anchors nothing a later prompt reaches.
    _prefill_and_finish(bm, Sequence(list(range(100, 132)) + [900] * 4, BLOCK))
    bm.take_state_maintenance_ops()
    bm.complete_previous_state_batch()
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    assert seq.replays_hit
    assert seq.checkpoint_demand_pos == 0 and bm.demands_recorded == 0
    assert _prefill_with_cuts(bm, seq) == [36]  # the prompt-end anchor only
    bm.deallocate(seq)
    branch = Sequence(list(range(100, 132)) + [777] * 8, BLOCK, has_per_req_cache=True)
    assert bm.can_allocate(branch) == 8
    assert branch.replays_hit


def test_a_resent_prompt_replays_once():
    """A prompt sent again whose checkpoint is gone: its hit is its own
    prompt-end anchor, which the replay checkpoints, so the next copy
    restores."""
    bm = _bm()
    _prefill_and_finish(bm, _seq(36))  # published, no checkpoint kept
    first = _seq(36)
    assert bm.allocate(first, bm.can_allocate(first))
    assert first.replays_hit and first.checkpoint_end_pos == 32
    assert _prefill_with_cuts(bm, first) == [32]
    bm.deallocate(first)
    again = _seq(36)
    assert bm.can_allocate(again) == 8
    assert not again.replays_hit


def test_a_forward_ending_inside_the_replay_is_not_checkpointed():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    _resume(bm, seq, hit=True)
    seq.checkpoint_end_pos = 28
    _forward(bm, seq, 28)
    assert seq.last_checkpoint_pos == 0


def test_a_replay_wider_than_the_step_budget_is_admitted_and_spans_chunks():
    """A 7-token replay under a 4-token budget: admitted at once, its first
    forward ends inside the replay, and it continues from there. Waiting for a
    step wide enough to pass the hit would wait forever."""
    engine, seq = _engine_with_a_replayed_head(budget=4)
    batch, scheduled = engine.schedule()
    assert scheduled == {seq.id: seq}
    assert seq.replays_hit
    assert seq.num_cached_tokens == 32 - REPLAY
    assert seq.replay == Replay(32, True, (), 32 - REPLAY)
    assert batch.replaying == {0: seq.replay}
    assert list(batch.num_scheduled_tokens) == [4]


def test_the_pending_work_probe_counts_a_replay_wider_than_the_budget():
    """The local prefill probe asks admission's question; a replay that cannot
    pass its hit this step is still work, not the end of the queue."""
    engine, _ = _engine_with_a_replayed_head(budget=4)
    assert engine._local_prefill_pending_work() == (True, 4)


def test_the_cost_estimate_starts_a_replayed_head_where_admission_does():
    engine, _ = _engine_with_a_replayed_head(budget=64)
    # 15 rows from the replay's start (25), not 8 from the hit (32).
    assert engine._estimate_prefill_chunk_cost() == engine._chunk_cost(32 - REPLAY, 15)


def test_a_late_publish_keeps_to_the_replay_it_was_scheduled_with():
    """Pipeline parallelism publishes a chunk after later chunks were scheduled,
    which may have ended the replay on the seq: the batch's `until` holds."""
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    _resume(bm, seq, hit=True)  # chunk 1: [25, 28)
    batch = SimpleNamespace(replaying={0: seq.replay})
    seq.num_cached_tokens = 28  # advanced at schedule time
    _resume(bm, seq, hit=False)  # chunk 2: [28, 36)
    seq.num_cached_tokens = 36
    _resume(bm, seq, hit=False)  # chunk 3: the replay is over on the seq
    assert seq.replay is None and seq.replay_until == 0
    seq.checkpoint_end_pos = 28  # an anchor inside the replay
    until = _scheduled_replay_until(batch, 0)
    assert until == 32 and _scheduled_replay_until(batch, 1) == 0
    bm.hash_blocks(seq, 3, start_tokens=32 - REPLAY, replay_until=until)
    assert seq.num_hashed_tokens == 0
    assert seq.last_checkpoint_pos == 0


def test_deallocate_clears_the_replay():
    bm = _bm()
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(40)
    assert bm.allocate(seq, bm.can_allocate(seq))
    _resume(bm, seq, hit=True)
    bm.deallocate(seq)
    assert seq.replay is None and not seq.replays_hit


def test_a_separate_dspark_draft_model_turns_replay_off(monkeypatch):
    monkeypatch.setenv("ATOM_DSV4_STATE_REPLAY", "1")
    spec = SimpleNamespace(method="dspark", use_dspark_with_draft=lambda: True)
    assert state_replay_tokens(_config(speculative_config=spec)) == 0


def test_a_continued_replay_counts_the_long_prefill_cap_from_its_hit():
    """Its rows below the hit ride on top of the cap, as the first forward's
    did, rather than eating into it."""
    engine, seq = _engine_with_a_replayed_head(budget=4)
    engine.long_prefill_token_threshold = 3
    engine.max_num_batched_tokens = 64
    seq.replay = Replay(32, True, (), 32 - REPLAY)
    seq.num_cached_tokens = 28  # an earlier chunk ended inside the replay
    chunk = Scheduler._partial_prefill_chunk(
        engine, seq, 28, 28, 0, cap_from=max(28, seq.replay_until)
    )
    assert chunk == 32 + 3 - 28
    assert Scheduler._partial_prefill_chunk(engine, seq, 28, 28, 0) == 3


def test_a_replay_past_the_prompt_demands_nothing():
    """A preempted seq re-prefilling what it generated replays a hit past its
    prompt, where prefill keeps no checkpoint: no demand, so no cut."""
    bm = _bm(state_checkpoint_demand=True)
    _prefill_and_finish(bm, _seq(40))
    bm.take_state_maintenance_ops()
    bm.complete_previous_state_batch()
    seq = _seq(40)
    seq.num_prompt_tokens = 20  # 20 prompt tokens, 20 generated
    assert bm.allocate(seq, bm.can_allocate(seq))
    assert seq.replays_hit
    assert seq.checkpoint_demand_pos == 0
    assert bm.demands_recorded == 0


def test_a_replay_reports_its_hit_from_where_its_forward_starts():
    """The rows a replay recomputes are not served from the cache."""
    engine, seq = _engine_with_a_replayed_head(budget=64)
    stats = engine.engine_stats
    cached, wanted = stats.total_cached_tokens, stats.total_wanted_tokens
    engine.schedule()
    assert seq.replays_hit
    assert seq.prefix_cache_hit_tokens == 32 - REPLAY
    # In the cache stats the replayed rows are reuse lost for want of a
    # checkpoint: wanted (the hit) minus cached (where the forward starts).
    assert stats.total_cached_tokens - cached == 32 - REPLAY
    assert stats.total_wanted_tokens - wanted == 32


# --- LMCache MP: a replay past the HBM hit, at the KV-only reach -----------
_MP = {"kv_transfer_config": {"kv_connector": "lmcache_mp", "kv_role": "offload"}}


@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        _MP["kv_transfer_config"],
        # As the factory resolves it: an alias, and the connector's default role.
        {"kv_connector": "LMCacheMPConnector"},
    ],
)
def test_lmcache_mp_offload_alone_keeps_replay_on(monkeypatch, kv_transfer_config):
    monkeypatch.setenv("ATOM_DSV4_STATE_REPLAY", "1")
    config = _config(kv_transfer_config=kv_transfer_config)
    assert state_replay_tokens(config) == REPLAY


def _mp_seq(bm, reach_with_state, reach_kv_only, n=60):
    """A 60-token prompt whose first 32 are in HBM with no checkpoint, and
    whose LMCache lookup answered `reach_with_state` / `reach_kv_only`."""
    _prefill_and_finish(bm, _seq(32))
    seq = _seq(n)
    seq.offload_joint.kv_prefix_tokens = reach_with_state
    seq.offload_joint.kv_only_tokens = reach_kv_only
    return seq


def _mp_bm(monkeypatch, min_load=0):
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", str(min_load))
    return _bm()


def test_a_kv_only_reach_past_hbm_loads_then_replays_below_it(monkeypatch):
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    hit = bm.can_allocate(seq)
    assert hit == 8  # the whole HBM prefix is claimed
    assert seq.replays_hit and seq.replay_kv_load
    assert bm.allocate(seq, hit)
    # The load runs first, from the HBM hit.
    assert seq.num_cached_tokens == 32 and seq.replay_kv_load
    # Once it lands (`num_cached_tokens` at the reach) the forward replays.
    seq.num_cached_tokens = 48
    _resume(bm, seq, hit=True)
    assert seq.replay == Replay(48, True, (), 48 - REPLAY)


def test_a_kv_only_load_that_does_not_run_replays_below_the_hbm_hit(monkeypatch):
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    assert bm.allocate(seq, bm.can_allocate(seq))
    bm.fall_back_to_hbm_replay(seq)
    assert not seq.replay_kv_load and seq.replays_hit
    _resume(bm, seq, hit=True)
    assert seq.replay == Replay(32, True, (), 32 - REPLAY)


def test_a_state_carrying_hit_close_to_the_kv_reach_is_loaded_as_is(monkeypatch):
    """44 with its image beats replaying to 48: 48 - 7 < 44."""
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=44, reach_kv_only=48)
    assert bm.can_allocate(seq) == 0
    assert not seq.replays_hit and not seq.replay_kv_load


def test_a_kv_reach_too_small_to_load_replays_from_hbm(monkeypatch):
    """The connector skips loads under OFFLOAD_MIN_LOAD_TOKENS, so the decision
    must not count on one."""
    bm = _mp_bm(monkeypatch, min_load=100)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    assert bm.can_allocate(seq) == 8
    assert seq.replays_hit and not seq.replay_kv_load
    assert bm.replay_start(seq, 32, hit=True) == 32 - REPLAY


def test_a_claim_short_of_the_hbm_hit_drops_the_kv_only_load(monkeypatch):
    """A load past the hit would start above a gap: replay from the claim."""
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    hit = bm.can_allocate(seq)
    assert seq.replay_kv_load
    lookup, calls = bm.kv.lookup, []

    def evicting(h):
        calls.append(h)
        return -1 if len(calls) == 7 else lookup(h)  # block 6 is gone

    monkeypatch.setattr(bm.kv, "lookup", evicting)
    assert bm.allocate(seq, hit)
    assert seq.num_cached_tokens == 24
    assert seq.replays_hit and not seq.replay_kv_load


def test_deallocate_forgets_the_kv_only_load(monkeypatch):
    bm = _mp_bm(monkeypatch)
    seq = _mp_seq(bm, reach_with_state=0, reach_kv_only=48)
    assert bm.allocate(seq, bm.can_allocate(seq))
    bm.deallocate(seq)
    assert not seq.replay_kv_load and not seq.replays_hit
