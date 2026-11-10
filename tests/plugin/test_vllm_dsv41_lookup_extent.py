# SPDX-License-Identifier: MIT
"""A V4.1 tier lookup asks only for the extent a load can end on."""

from types import SimpleNamespace

from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.metadata import LoadSpec
from atom.plugin.vllm.kv_transfer.v41_mp_scheduler import V41StateMPScheduler
from atom.plugin.vllm.paged_state_checkpoint_scheduler import StateCheckpointPlacement

LENS = (2048, 2049, 2050, 2303)


def _sched(tail=None):
    """The base scheduler with `tail=None`; V4.1's MP scheduler otherwise."""
    if tail is None:
        s = object.__new__(ChunkedOffloadSchedulerBase)
    else:
        s = object.__new__(V41StateMPScheduler)
        s._planner = SimpleNamespace(prefill_tail=tail)
    s.chunk_size = 256
    return s


def _seq(n):
    return SimpleNamespace(num_prompt_tokens=n, token_ids=list(range(n + 7)))


def test_default_lookup_is_the_whole_prompt():
    assert _sched()._lookup_token_ids(_seq(2048)) == list(range(2048))


def test_lookup_stops_where_a_load_can_end():
    assert [_sched(1)._lookup_len(_seq(n)) for n in LENS] == [1792, 2048, 2048, 2048]
    assert [_sched(2)._lookup_len(_seq(n)) for n in LENS] == [1792, 1792, 2048, 2048]
    assert _sched(2)._lookup_token_ids(_seq(2049)) == list(range(1792))
    assert _sched(1)._lookup_len(_seq(200)) == 0


def test_anchor_leaves_the_prefill_tail():
    p = StateCheckpointPlacement(interval=256, demand=True, block=256)
    assert [p.anchor(n, n) for n in LENS] == [1792, 2048, 2048, 2048]
    assert [p.anchor(n, n, 2) for n in LENS] == [1792, 1792, 2048, 2048]


def test_a_whole_prompt_hit_does_not_reach_past_the_capped_checkpoint(monkeypatch):
    """The MP base extends a whole-prompt hit to the prompt's end; V4.1's
    checkpoint is restored at the capped hit, so the PAGE end must stay there."""
    monkeypatch.setattr(
        ChunkedOffloadSchedulerBase,
        "get_num_new_matched_tokens",
        lambda self, seq: (768, True),
    )
    s = _sched(1)
    spec = LoadSpec(0, 1024, can_load=True)
    s._load_specs = {"r": spec}
    s._hit_save_floors = {}
    s.block_size = 256
    s._last_tier_hit = lambda seq, sid: 1024
    seq = SimpleNamespace(id="r", num_prompt_tokens=1024)
    assert s.get_num_new_matched_tokens(seq) == (768, True)
    assert spec.transfer_end_tokens is None
