"""The truncation arithmetic of the multi-group KV-load-failure path.

The patch this exercises exists because vLLM's own implementation unpacks a
single KV cache group and raises on a hybrid model (Kimi-K3 has four). What
matters is not that it stops raising but that it truncates each request at the
right token: too late leaves a request reading KV it could not load, which is a
wrong answer with nothing logged.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.kv_transfer.hybrid_invalid_blocks_patch import (
    update_requests_with_invalid_blocks,
)


def _scheduler(blocks_by_req):
    return SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            get_block_ids=lambda req_id: blocks_by_req[req_id]
        )
    )


def _request(req_id, num_computed_tokens):
    return SimpleNamespace(request_id=req_id, num_computed_tokens=num_computed_tokens)


def test_truncates_at_the_earliest_invalid_block_across_groups():
    # Two groups, 16 tokens per block, 4 computed blocks (64 tokens). Group 0's
    # invalid block sits at index 3, group 1's at index 1 -- the prefix is only
    # valid up to token 16, because the request reads both groups.
    blocks = {"r0": ([10, 11, 12, 13], [20, 21, 22, 23])}
    request = _request("r0", 64)
    affected, tokens, evicted = update_requests_with_invalid_blocks(
        _scheduler(blocks), [request], {13, 21}, {}, [16, 16]
    )
    assert affected == {"r0"}
    assert request.num_computed_tokens == 16
    assert tokens == 48
    # Every group's tail from the block holding token 16, not just group 1's.
    assert evicted == {11, 12, 13, 21, 22, 23}


def test_group_with_no_invalid_block_still_has_its_tail_evicted():
    blocks = {"r0": ([10, 11, 12, 13], [20, 21, 22, 23])}
    request = _request("r0", 64)
    _, _, evicted = update_requests_with_invalid_blocks(
        _scheduler(blocks), [request], {12}, {}, [16, 16]
    )
    assert request.num_computed_tokens == 32
    assert evicted == {12, 13, 22, 23}


def test_unequal_block_sizes_index_their_own_group():
    # Group 1 holds 32 tokens per block, so its block index 1 starts at token
    # 32 -- ahead of group 0's invalid block at token 16.
    blocks = {"r0": ([10, 11, 12, 13], [20, 21])}
    request = _request("r0", 64)
    _, tokens, evicted = update_requests_with_invalid_blocks(
        _scheduler(blocks), [request], {11, 21}, {}, [16, 32]
    )
    assert request.num_computed_tokens == 16
    assert tokens == 48
    assert evicted == {11, 12, 13, 20, 21}


def test_scheduled_tokens_are_not_counted_as_computed():
    # 64 computed of which 16 were scheduled this step: the prefix under
    # consideration is 48 tokens, so group 0's block 3 (token 48) is past it.
    blocks = {"r0": ([10, 11, 12, 13], [20, 21, 22, 23])}
    request = _request("r0", 64)
    affected, _, _ = update_requests_with_invalid_blocks(
        _scheduler(blocks), [request], {13}, {"r0": 16}, [16, 16]
    )
    assert affected == set()
    assert request.num_computed_tokens == 64


def test_a_block_shared_with_an_earlier_request_is_recomputed_once():
    blocks = {
        "r0": ([10, 11], [20, 21]),
        "r1": ([10, 11], [20, 21]),
    }
    first, second = _request("r0", 32), _request("r1", 32)
    affected, _, _ = update_requests_with_invalid_blocks(
        _scheduler(blocks), [first, second], {11}, {}, [16, 16]
    )
    assert affected == {"r0", "r1"}
    # r0 recomputes the block; r1 keeps only what it had cached, rather than
    # recomputing the same block a second time.
    assert first.num_computed_tokens == 16
    assert second.num_computed_tokens == 32


def test_untouched_request_is_left_alone():
    blocks = {"r0": ([10, 11], [20, 21])}
    request = _request("r0", 32)
    affected, tokens, evicted = update_requests_with_invalid_blocks(
        _scheduler(blocks), [request], {99}, {}, [16, 16]
    )
    assert (affected, tokens, evicted) == (set(), 0, set())
    assert request.num_computed_tokens == 32


def test_evict_blocks_false_reports_without_evicting():
    blocks = {"r0": ([10, 11], [20, 21])}
    request = _request("r0", 32)
    affected, _, evicted = update_requests_with_invalid_blocks(
        _scheduler(blocks), [request], {11}, {}, [16, 16], evict_blocks=False
    )
    assert affected == {"r0"}
    assert evicted == set()


def test_recurrent_group_forces_a_full_rewind():
    # Group 0 is attention (16 tokens/block), group 1 is a KDA state slot. The
    # invalid block sits at index 3, so an attention-only model would resume at
    # token 48. There is no recurrent state for token 48 -- only for the whole
    # prefix the request has already run -- so it has to go back to zero.
    blocks = {"r0": ([10, 11, 12, 13], [90])}
    request = _request("r0", 64)
    affected, tokens, evicted = update_requests_with_invalid_blocks(
        _scheduler(blocks),
        [request],
        {13},
        {},
        [16, 0],
        recurrent_groups=[False, True],
    )
    assert affected == {"r0"}
    assert request.num_computed_tokens == 0
    assert tokens == 64
    # The whole attention prefix goes; the state slot is not a cache entry and
    # is not evicted by prefix arithmetic.
    assert evicted == {10, 11, 12, 13}


def test_state_slot_id_is_not_read_as_a_prefix_block():
    # Every group allocates from one block pool, so a state slot id can look
    # like any other id. Scanning it would match an id the connector reported
    # for some attention block and truncate a request that is not affected.
    blocks = {"r0": ([10, 11], [13])}
    request = _request("r0", 32)
    affected, tokens, _ = update_requests_with_invalid_blocks(
        _scheduler(blocks),
        [request],
        {13},
        {},
        [16, 0],
        recurrent_groups=[False, True],
    )
    assert affected == set()
    assert request.num_computed_tokens == 32
    assert tokens == 0


def test_shared_invalid_block_still_rewinds_the_recurrent_request_fully():
    # r0 takes the block for recomputation; r1 would normally be allowed to
    # keep it computed. Its own state slot is still the state of the prefix it
    # can no longer claim, so it goes to zero too.
    blocks = {"r0": ([10, 11], [90]), "r1": ([10, 11], [91])}
    r0, r1 = _request("r0", 32), _request("r1", 32)
    affected, _, _ = update_requests_with_invalid_blocks(
        _scheduler(blocks),
        [r0, r1],
        {10},
        {},
        [16, 0],
        recurrent_groups=[False, True],
    )
    assert affected == {"r0", "r1"}
    assert r0.num_computed_tokens == 0
    assert r1.num_computed_tokens == 0


def test_full_rewind_reports_each_request_to_the_caller():
    # A running request cannot simply be told its computed tokens are zero:
    # under async scheduling its output placeholders have to be reconciled,
    # which only the scheduler's own preemption does. The body therefore hands
    # every fully rewound request back so the caller can preempt it.
    blocks = {"r0": ([10, 11], [90]), "r1": ([10, 11], [91])}
    r0, r1 = _request("r0", 32), _request("r1", 32)
    rewound = []
    update_requests_with_invalid_blocks(
        _scheduler(blocks),
        [r0, r1],
        {10},
        {},
        [16, 0],
        recurrent_groups=[False, True],
        on_full_rewind=rewound.append,
    )
    # r0 truncates at the invalid block, r1 falls back on the shared-block
    # branch: both end at token 0 and both have to be reported.
    assert rewound == [r0, r1]


def test_attention_only_model_reports_no_full_rewind():
    # Partial truncation is legitimate without a recurrent group, and a
    # request that keeps a valid prefix must not be preempted.
    blocks = {"r0": ([10, 11, 12, 13], [20, 21, 22, 23])}
    request = _request("r0", 64)
    rewound = []
    update_requests_with_invalid_blocks(
        _scheduler(blocks),
        [request],
        {13},
        {},
        [16, 16],
        on_full_rewind=rewound.append,
    )
    assert request.num_computed_tokens == 48
    assert rewound == []


def _with_fake_request_status(monkeypatch, running):
    """Stand in for ``vllm.v1.request``, which is not importable here."""
    import sys
    import types

    status = SimpleNamespace(RUNNING=running)
    for name in ("vllm", "vllm.v1", "vllm.v1.request"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.v1.request"].RequestStatus = status


def test_preempt_rewound_requests_only_takes_running_ones(monkeypatch):
    from atom.plugin.vllm.kv_transfer.hybrid_invalid_blocks_patch import (
        preempt_rewound_requests,
    )

    _with_fake_request_status(monkeypatch, "RUNNING")
    running_req = SimpleNamespace(request_id="r0", status="RUNNING")
    # The async branch's requests are waiting on a remote KV load, hold no
    # output placeholders, and would trip the assert inside _preempt_request.
    waiting_req = SimpleNamespace(request_id="r1", status="WAITING_FOR_REMOTE_KVS")
    other = SimpleNamespace(request_id="r2", status="RUNNING")
    preempted = []
    scheduler = SimpleNamespace(
        running=[running_req, other],
        _preempt_request=lambda req, ts, drop_stale_output=False: preempted.append(
            (req.request_id, drop_stale_output)
        ),
    )
    n = preempt_rewound_requests(scheduler, [running_req, waiting_req])
    assert n == 1
    assert preempted == [("r0", True)]
    # Popped from the running queue before preemption, as _preempt_request
    # requires; the untouched request stays.
    assert scheduler.running == [other]


def test_preempt_rewound_requests_is_a_no_op_without_running_requests(monkeypatch):
    from atom.plugin.vllm.kv_transfer.hybrid_invalid_blocks_patch import (
        preempt_rewound_requests,
    )

    _with_fake_request_status(monkeypatch, "RUNNING")
    waiting_req = SimpleNamespace(request_id="r1", status="WAITING_FOR_REMOTE_KVS")
    scheduler = SimpleNamespace(
        running=[],
        _preempt_request=lambda *a, **k: pytest.fail("must not preempt"),
    )
    assert preempt_rewound_requests(scheduler, [waiting_req]) == 0
