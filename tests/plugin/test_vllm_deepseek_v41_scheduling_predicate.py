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


def _request(spec=False, computed=10, prompt=10):
    return SimpleNamespace(
        spec_token_ids=[1, 2] if spec else [],
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
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
    running = [_request(spec=True), _request(computed=10, prompt=4096)]
    assert _would_mix_prefill_with_drafts(_scheduler(running, [], cap=99))


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
