# SPDX-License-Identifier: MIT
# `BlockManager.can_allocate_estimate`: the delayer's fit probe, memoised across
# hold ticks. It must answer what `can_allocate` would while the index answers
# the walk the same way, and re-walk when an entry it read changes. Equality
# under random churn is checked by `test_admission_fast_walk.churn`.

import pytest
from test_admission_fast_walk import count_walks, drive_scheduler, exact
from test_admission_fast_walk import prefix_config as _prefix_config
from test_dsv4_state_replay import _mp_bm, _mp_seq
from test_state_checkpoint import (
    ckpt_config,
    make_block_manager,
    publish_at_boundary,
    publisher_has_read_its_source,
    stateful_seq,
)

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence


def prefix_bm(**overrides):
    return BlockManager(_prefix_config(**overrides))


def seq_of(tokens):
    return Sequence(list(tokens), 4)


def publish(bm, seq):
    hit = BlockManager.can_allocate(bm, seq)  # unbound: not seen by `count_walks`
    bm.allocate(seq, hit)
    bm.hash_blocks(seq, seq.num_prompt_tokens - seq.num_cached_tokens)
    bm.deallocate(seq)


@pytest.mark.parametrize("memo", ["1", "0"])
def test_hold_ticks_reuse_the_walk(monkeypatch, memo):
    monkeypatch.setenv("ATOM_PREFILL_DELAYER_PROBE_MEMO", memo)
    bm = prefix_bm()
    publish(bm, seq_of(range(40)))
    head = seq_of(list(range(40)) + [900, 901, 902])
    walks = count_walks(monkeypatch, bm)
    for _ in range(5):
        assert bm.can_allocate_estimate(head) == exact(bm, head) == 10
    if memo == "1":
        assert walks == [{"record": False, "remember": True}]
    else:
        assert walks == [{"record": False}] * 5


def _evict_a_hit_block(bm, head):
    bm.kv.allocate(bm.kv.lookup(bm._prompt_hashes(head)[3]))


def _flood_the_eviction_log(bm, head):
    for h in range(bm._recent_evictions.maxlen + 1):
        bm._record_evicted(-1 - h)


@pytest.mark.parametrize(
    "change",
    [
        _evict_a_hit_block,
        lambda bm, head: publish(bm, seq_of(range(60))),  # at the first miss
        _flood_the_eviction_log,
        lambda bm, head: bm.clear_cache(),  # no eviction callback
        lambda bm, head: head.append_token(7),  # a preempted seq returns longer
    ],
)
def test_a_change_to_what_the_walk_read_rewalks(monkeypatch, change):
    bm = prefix_bm()
    publish(bm, seq_of(range(40)))
    head = seq_of(list(range(56)) + [900])
    walks = count_walks(monkeypatch, bm)
    assert bm.can_allocate_estimate(head) == 10
    change(bm, head)
    assert bm.can_allocate_estimate(head) == exact(bm, head)
    assert len(walks) == 2
    bm.can_allocate_estimate(head)
    assert len(walks) == 2  # the new walk is reused in turn


def test_fit_is_retested_on_every_call(monkeypatch):
    bm = prefix_bm(num_kvcache_blocks=12)
    head = seq_of(range(20))
    walks = count_walks(monkeypatch, bm)
    assert bm.can_allocate_estimate(head) == 0
    hog = seq_of(range(5000, 5032))
    bm.allocate(hog, 0)
    assert bm.can_allocate_estimate(head) == exact(bm, head) == -1
    bm.deallocate(hog)
    assert bm.can_allocate_estimate(head) == exact(bm, head) == 0
    assert len(walks) == 1


def test_the_state_gate_is_asked_again():
    bm = make_block_manager(ckpt_config())
    first = stateful_seq(list(range(40)))
    h = publish_at_boundary(bm, first)
    publisher_has_read_its_source(bm)
    bm.deallocate(first)
    head = stateful_seq(list(range(40)) + list(range(500, 520)))
    hit = bm.can_allocate_estimate(head)
    assert hit == exact(bm, head) > 0
    bm.state.unindex(h)  # the checkpoint goes; its KV block stays
    assert bm.can_allocate_estimate(head) == exact(bm, head) < hit


def test_the_replay_is_asked_again(monkeypatch):
    """The tier's answer lands between ticks; the estimate follows, flags too."""
    bm = _mp_bm(monkeypatch)
    # 32 tokens in HBM with no checkpoint; LMCache holds 44 with state.
    head = _mp_seq(bm, reach_with_state=44, reach_kv_only=48)
    walks = count_walks(monkeypatch, bm)

    def estimate():
        return bm.can_allocate_estimate(head), head.replays_hit, head.replay_kv_load

    assert estimate() == estimate() == (0, False, False)  # loading 44 wins
    assert len(walks) == 1
    head.offload_joint.kv_prefix_tokens = 0  # only the KV-only reach left
    assert estimate() == (8, True, True) and exact(bm, head) == 8
    head.offload_joint.kv_prefix_tokens = 44
    assert estimate() == (0, False, False) and exact(bm, head) == 0


@pytest.mark.parametrize("seed", range(3))
def test_identical_schedules_with_and_without_the_memo(monkeypatch, seed):
    runs = []
    for memo in ("0", "1"):
        monkeypatch.setenv("ATOM_PREFILL_DELAYER_PROBE_MEMO", memo)
        runs.append(drive_scheduler(monkeypatch, seed))
    (off_log, off_walks, off_ev), (on_log, on_walks, on_ev) = runs
    assert on_log == off_log
    assert on_ev == off_ev > 0
    assert 0 < on_walks < off_walks
