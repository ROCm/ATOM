# SPDX-License-Identifier: MIT
"""A V4.1 tier lookup asks only for the extent a load can end on."""

from types import SimpleNamespace

from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.plugin.vllm.paged_state_checkpoint_scheduler import StateCheckpointPlacement

LENS = (2048, 2049, 2050, 2303)


def _sched(tail=None):
    s = object.__new__(ChunkedOffloadSchedulerBase)
    s.chunk_size = 256
    if tail is not None:
        s.truncate_lookup_query(tail)
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
