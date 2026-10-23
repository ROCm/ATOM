# SPDX-License-Identifier: MIT
"""When a V4.1 step's draft blocks are dropped, and when they are kept.

The drafts are cleared so a prefill and a verification step never share one
forward, which `V41Cache.begin_step` refuses. The question is what counts as
evidence that a prefill is coming. A nonempty waiting queue is not it: vLLM
stops admitting at

    num_running = len(self.running) + self.num_waiting_for_streaming_input
    if num_running >= self.max_num_active_reqs: break

so at the cap, every running request decodes and nothing can join them
however long the queue. Clearing there costs the step its speculation and
prevents nothing.

The failure directions are not symmetric, and the tests below are written
around that: keeping drafts when a prefill does arrive raises a loud
`NotImplementedError`, while clearing them when none was coming is silent
lost throughput. So where this cannot tell, it clears.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from atom.plugin.vllm.v41_speculative import (
    _admission_is_possible,
    _would_mix_prefill_with_drafts,
)


def _request(spec=False, *, prompt=10, generated=1, left=1, placeholders=0):
    """A request as `Scheduler.schedule` sees it on *entry*.

    `left` is what it still has to compute this step: one token for a decode,
    more for a prompt chunk or for a preempted request recomputing what it
    already produced.

    Stated that way on purpose. vLLM advances `num_computed_tokens` late in
    `schedule` and evaluates its own `is_prefill_chunk` only afterwards, so on
    entry a decoding request is exactly one token short of its length -- and a
    fake built as `computed == num_tokens` describes a state this predicate is
    never handed, which is how a version of it that was true for every request
    on every step went unnoticed here.
    """
    num_tokens = prompt + generated
    return SimpleNamespace(
        spec_token_ids=[1, 2] if spec else [],
        num_tokens=num_tokens,
        num_prompt_tokens=prompt,
        num_output_placeholders=placeholders,
        num_computed_tokens=num_tokens + placeholders - left,
    )


def _scheduler(running, waiting, cap, streaming=0):
    return SimpleNamespace(
        running=running,
        waiting=waiting,
        max_num_active_reqs=cap,
        num_waiting_for_streaming_input=streaming,
    )


def test_a_full_decode_batch_keeps_its_drafts():
    """The counterexample: every running request decodes, the cap is reached,
    and the queue is nonempty. Nothing can be admitted, so nothing mixes."""
    sched = _scheduler([_request(spec=True) for _ in range(4)], ["queued"], cap=4)
    assert not _would_mix_prefill_with_drafts(sched)


def test_room_under_the_cap_still_clears_them():
    sched = _scheduler([_request(spec=True) for _ in range(3)], ["queued"], cap=4)
    assert _would_mix_prefill_with_drafts(sched)


def test_streaming_slots_count_toward_the_cap():
    """They are not in `running` but hold a runner slot, and vLLM's own
    comparison adds them -- so this one must too, or it reads a full batch as
    having room."""
    sched = _scheduler(
        [_request(spec=True) for _ in range(3)], ["queued"], cap=4, streaming=1
    )
    assert not _would_mix_prefill_with_drafts(sched)


def test_a_chunked_prompt_in_running_still_clears_them():
    """No admission needed: a request already in `running` can be mid-prompt,
    and its next chunk is a prefill in the same step as the drafts."""
    running = [_request(spec=True), _request(prompt=4096, generated=0, left=4086)]
    assert _would_mix_prefill_with_drafts(_scheduler(running, [], cap=99))


def test_a_decode_only_batch_with_an_empty_queue_keeps_its_drafts():
    """The regression this predicate shipped with once, and the reason DSpark
    verified nothing: read against the full length, every decoding request
    looks mid-prompt on entry, so the drafts were cleared on every step of a
    batch that had no prefill in it at all."""
    running = [_request(spec=True, generated=g) for g in (1, 5, 200)]
    assert not _would_mix_prefill_with_drafts(_scheduler(running, [], cap=99))


def test_a_preempted_request_recomputing_its_output_clears_them():
    """It is back at the start of a prompt it has already answered part of,
    so what it recomputes is longer than its prompt -- the case a comparison
    against `num_prompt_tokens` alone reads as a decode throughout."""
    resumed = _request(prompt=10, generated=5, left=15)
    assert resumed.num_computed_tokens == 0
    running = [_request(spec=True), resumed]
    assert _would_mix_prefill_with_drafts(_scheduler(running, [], cap=99))


def test_the_last_token_of_a_prompt_is_a_decode_row():
    """Deliberate: with one token to compute it has a decode's shape, which is
    what the refusal is about -- not which length that token came from."""
    running = [_request(spec=True), _request(prompt=4096, generated=0, left=1)]
    assert not _would_mix_prefill_with_drafts(_scheduler(running, [], cap=99))


def test_output_placeholders_do_not_make_a_decode_look_like_a_prefill():
    """Async scheduling carries placeholders for tokens not yet sampled; they
    raise the length and the computed count together."""
    running = [_request(spec=True, placeholders=3)]
    assert not _would_mix_prefill_with_drafts(_scheduler(running, [], cap=99))


def test_no_drafts_anywhere_is_never_a_mixed_step():
    assert not _would_mix_prefill_with_drafts(
        _scheduler([_request() for _ in range(2)], ["queued"], cap=99)
    )


def test_an_unknown_cap_keeps_the_old_conservative_answer():
    """A vLLM that renames the attribute must not silently start keeping
    drafts into prefills -- that is the loud direction, but it is still the
    one this exists to prevent."""
    sched = SimpleNamespace(running=[_request(spec=True)], waiting=["queued"])
    assert _admission_is_possible(sched)
    assert _would_mix_prefill_with_drafts(sched)
