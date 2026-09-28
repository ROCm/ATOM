# SPDX-License-Identifier: MIT
"""A speculative preemption may replay only finalized host token IDs."""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams


def setup_request():
    sched = Scheduler(
        MockConfig(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=5, use_dspark=lambda: False
            ),
            num_kvcache_blocks=64,
        )
    )
    seq = Sequence(
        [10, 11, 12, 13],
        4,
        SamplingParams(max_tokens=40, ignore_eos=True),
        num_draft_tokens=5,
    )
    sched.add(seq)
    batch, active = sched.schedule()
    sched.postprocess(
        list(active.values()),
        ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True),
        batch=batch,
    )
    return sched, seq


def output(seq, tokens, rejected=0):
    return ScheduledBatchOutput(
        [seq.id],
        [tuple(tokens)],
        np.array([rejected]),
        np.array([max(0, len(tokens) - 1)]),
        np.zeros((1, 5), dtype=np.int32),
        is_deferred_out=True,
    )


def preempt(sched, seq):
    sched.running.remove(seq)
    assert sched.preempt(seq)


def test_preempt_before_first_deferred_output_drops_all_placeholders():
    sched, seq = setup_request()
    assert seq.num_placeholder_tokens == 6
    preempt(sched, seq)
    assert list(seq.token_ids) == [10, 11, 12, 13]
    assert seq.num_tokens == seq.num_finalized_tokens == 4
    assert not seq.output_tokens


def test_preempt_middle_prefill_never_strips_prompt_tokens():
    sched = Scheduler(
        MockConfig(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=5, use_dspark=lambda: False
            ),
            num_kvcache_blocks=64,
            max_num_batched_tokens=4,
            enable_chunked_prefill=True,
        )
    )
    tokens = list(range(10, 23))
    seq = Sequence(tokens, 4, SamplingParams(max_tokens=40), num_draft_tokens=5)
    sched.add(seq)
    batch, active = sched.schedule()
    sched.postprocess(
        list(active.values()),
        ScheduledBatchOutput([], [], None, None, None),
        batch=batch,
    )
    assert seq.is_partial_prefill
    preempt(sched, seq)
    assert list(seq.token_ids) == tokens


def test_immediate_replay_ignores_old_deferred_output_for_same_request():
    sched, seq = setup_request()
    preempt(sched, seq)
    batch, active = sched.schedule()
    assert batch.total_seqs_num_prefill == 1
    sched.postprocess(list(active.values()), output(seq, [77]), batch=batch)
    assert seq.num_finalized_tokens == 4
    assert seq.num_tokens == 10  # exactly one fresh deferred reservation
    assert 77 not in seq.token_ids
    batch, active = sched.schedule()
    sched.postprocess(list(active.values()), output(seq, [88]), batch=batch)
    preempt(sched, seq)
    assert list(seq.token_ids) == [10, 11, 12, 13, 88]


@pytest.mark.parametrize("accepted", range(1, 7))
def test_preempt_after_each_acceptance_length_preserves_emitted_prefix(accepted):
    sched, seq = setup_request()
    batch, active = sched.schedule()
    sched.postprocess(list(active.values()), output(seq, [88]), batch=batch)
    batch, active = sched.schedule()
    emitted = list(range(90, 90 + accepted))
    sched.postprocess(
        list(active.values()), output(seq, emitted, 6 - accepted), batch=batch
    )
    preempt(sched, seq)
    assert list(seq.token_ids) == [10, 11, 12, 13, 88] + emitted
    assert seq.num_tokens == seq.num_finalized_tokens == 5 + accepted
