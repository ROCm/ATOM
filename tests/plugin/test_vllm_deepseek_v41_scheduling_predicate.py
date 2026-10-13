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


def _scheduler(running, waiting, cap, streaming=0, holding=()):
    return SimpleNamespace(
        running=running,
        waiting=waiting,
        kv_holding_waiting=holding,
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


@pytest.mark.parametrize(
    "cap,streaming,expected",
    [(4, 0, True), (1, 0, False), (2, 1, False), (None, 0, True)],
)
def test_remote_restore_queue_obeys_the_admission_cap(cap, streaming, expected):
    restored = _request(computed=17408, prompt=17459)
    sched = _scheduler(
        [_request(spec=True)], [], cap, streaming=streaming, holding=[restored]
    )
    assert _would_mix_prefill_with_drafts(sched) is expected


@pytest.mark.parametrize("has_drafts,has_holding", [(True, False), (False, True)])
def test_holding_queue_does_not_disable_decode_without_a_possible_mix(
    has_drafts, has_holding
):
    sched = _scheduler(
        [_request(spec=has_drafts)],
        [],
        cap=4,
        holding=[_request(computed=17408, prompt=17459)] if has_holding else [],
    )
    assert not _would_mix_prefill_with_drafts(sched)


def test_legacy_scheduler_without_holding_queue_keeps_decode_drafts():
    sched = SimpleNamespace(running=[_request(spec=True)], waiting=[])
    assert not _would_mix_prefill_with_drafts(sched)


def test_remote_completion_clears_drafts_before_the_original_schedule(monkeypatch):
    from atom.plugin.vllm.v41_speculative import patch_v41_speculative_scheduling
    from vllm.v1.core.sched.scheduler import Scheduler

    decode = _request(spec=True, computed=17462, prompt=17459)
    decode.num_tokens = 17463
    restored = _request(computed=17408, prompt=17459)
    restored.num_tokens = 17459
    restored.ready = False
    sched = _scheduler([decode], [restored], cap=4)
    sched.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        model_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"]),
    )
    observed = []

    def original(self):
        # Match vLLM's ordering: running drafts are consumed before a remote
        # receive is unblocked and admitted from the KV-holding queue.
        drafts = len(decode.spec_token_ids)
        lengths = [decode.num_tokens + drafts - decode.num_computed_tokens]
        counts = [drafts]
        if self.waiting:
            self.kv_holding_waiting = [self.waiting.pop()]
        elif self.kv_holding_waiting and restored.ready:
            self.running.append(self.kv_holding_waiting.pop())
            lengths.append(restored.num_tokens - restored.num_computed_tokens)
            counts.append(0)
        observed.append((lengths, counts))
        return lengths, counts

    monkeypatch.setattr(Scheduler, "schedule", original)
    assert patch_v41_speculative_scheduling()
    Scheduler.schedule(sched)
    assert not sched.waiting and sched.kv_holding_waiting == [restored]
    assert observed == [([1], [0])]

    # A later completion does not move the request back to ordinary waiting.
    decode.spec_token_ids = [1, 2, 3, 4, 5]
    restored.ready = True
    lengths, counts = Scheduler.schedule(sched)
    assert lengths == [1, 51]
    assert counts == [0, 0]
    assert not decode.spec_token_ids
    assert not sched.kv_holding_waiting

    from atom.plugin.vllm import deepseek_v41_bridge as bridge

    builder = SimpleNamespace(geometry=SimpleNamespace(speculative_tokens=5))
    batch = SimpleNamespace(
        num_scheduled_tokens=lengths,
        context_lens=[17462 + lengths[0], 17408 + lengths[1]],
    )
    monkeypatch.setattr(bridge, "_v41_scheduled_draft_counts", lambda n: counts)
    assert bridge._v41_prepare_kwargs(builder, batch, synthetic=False) == {}


@pytest.mark.parametrize("replaying", [True, False])
def test_running_output_history_uses_the_prefill_lifecycle_flag(replaying):
    replay = _request(computed=256, prompt=256)
    replay.num_tokens = 288 if replaying else 257
    replay.num_output_placeholders = 0
    replay.is_prefill_chunk = replaying
    sched = _scheduler([_request(spec=True), replay], [], cap=2)
    assert _would_mix_prefill_with_drafts(sched) is replaying


def test_running_output_replay_clears_drafts_before_schedule(monkeypatch):
    from atom.plugin.vllm.v41_speculative import patch_v41_speculative_scheduling
    from vllm.v1.core.sched.scheduler import Scheduler

    decode = _request(spec=True, computed=512, prompt=256)
    decode.spec_token_ids = [1, 2, 3, 4, 5]
    decode.num_tokens = 513
    decode.is_prefill_chunk = False
    replay = _request(computed=256, prompt=256)
    replay.num_tokens = 288
    replay.num_output_placeholders = 0
    replay.is_prefill_chunk = True
    sched = _scheduler([decode, replay], [], cap=2)
    sched.vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
        model_config=SimpleNamespace(architectures=["DeepseekV41ForCausalLM"]),
    )

    def original(self):
        counts = [len(request.spec_token_ids) for request in self.running]
        lengths = [
            request.num_tokens + count - request.num_computed_tokens
            for request, count in zip(self.running, counts)
        ]
        return lengths, counts

    monkeypatch.setattr(Scheduler, "schedule", original)
    assert patch_v41_speculative_scheduling()
    lengths, counts = Scheduler.schedule(sched)
    assert lengths == [1, 32]
    assert counts == [0, 0]
    assert not decode.spec_token_ids


def test_mixed_verify_and_restored_tail_is_still_refused(monkeypatch):
    from atom.plugin.vllm import deepseek_v41_bridge as bridge

    builder = SimpleNamespace(geometry=SimpleNamespace(speculative_tokens=5))
    batch = SimpleNamespace(num_scheduled_tokens=[6, 51], context_lens=[17468, 17459])
    monkeypatch.setattr(bridge, "_v41_scheduled_draft_counts", lambda n: [5, 0])
    with pytest.raises(NotImplementedError, match="same step as a prefill"):
        bridge._v41_prepare_kwargs(builder, batch, synthetic=False)
