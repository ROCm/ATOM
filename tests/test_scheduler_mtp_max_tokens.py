"""Regression coverage for MTP output batches crossing ``max_tokens``."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.sampling_params import SamplingParams
from tests.logprob_helpers import token_logprob


class TestMTPMaxTokens:
    mtp_k = 3

    def _scheduler(self):
        return Scheduler(
            MockConfig(
                speculative_config=SimpleNamespace(
                    num_speculative_tokens=self.mtp_k,
                    use_dspark=lambda: False,
                ),
                num_kvcache_blocks=64,
            )
        )

    @staticmethod
    def _prefill(sched, seq):
        sched.add(seq)
        sched.schedule()
        return seq

    @staticmethod
    def _accept_all(sched, seq, tokens, stream_queue):
        # Model the live MTP state at postprocess: the prior accepted token and
        # its three drafts are followed by four newly scheduled provisional
        # positions. The scheduler replaces the verified window, then removes
        # the three unused speculative placeholders from the real length.
        seq.num_placeholder_tokens = sched.mtp_k
        for token in (87, 88, 89, 90, 91, 92, 93):
            seq.append_token(token)
        return sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[seq.id],
                token_ids=[tuple(tokens)],
                num_rejected=np.asarray([0]),
                num_bonus=np.asarray([0]),
                draft_token_ids=np.asarray([[20, 21, 22]]),
                # pp=1 output is deferred (drafts + lagging provisional run).
                is_deferred_out=True,
                logprobs=(
                    {
                        seq.id: [
                            token_logprob(-0.1 * (i + 1), 1) for i in range(len(tokens))
                        ]
                    }
                    if seq.return_logprobs
                    else None
                ),
            ),
            stream_output_queue=stream_queue,
        )

    def test_cap_truncates_internal_and_stream_output(self, seq_factory):
        sched = self._scheduler()
        seq = self._prefill(
            sched,
            seq_factory(
                [1, 2, 3, 4],
                sampling_params=SamplingParams(
                    max_tokens=2, ignore_eos=True, logprobs=1
                ),
            ),
        )
        stream_queue = mock.Mock()
        finished = self._accept_all(sched, seq, [10, 11, 12, 13], stream_queue)

        assert len(finished) == 1
        assert finished[0].leave_reason == "max_tokens"
        assert finished[0].num_completion_tokens == 2
        assert len(finished[0].logprobs) == finished[0].num_completion_tokens
        emitted = stream_queue.put_nowait.call_args.args[0][0][1]
        assert emitted.output_tokens == [10, 11]
        assert [p[0] for p in emitted.logprobs] == [-0.1, -0.2]
        assert emitted.finish_reason == "max_tokens"

    @pytest.mark.parametrize(
        ("max_tokens", "injected_t0", "expected_output", "expected_count"),
        [
            (0, None, [], 0),
            (-1, None, [], 0),
            (0, 99, [], 0),
            (1, 99, [99], 1),
        ],
    )
    def test_empty_cap_and_injected_t0_boundary(
        self,
        seq_factory,
        max_tokens,
        injected_t0,
        expected_output,
        expected_count,
    ):
        sched = self._scheduler()
        seq = self._prefill(
            sched,
            seq_factory(
                [1, 2, 3, 4],
                sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
            ),
        )
        if injected_t0 is not None:
            # Match _schedule_first_decode_after_remote_kv: injected T0 is
            # already part of sequence state and is also emitted with the
            # first decode result.
            seq.append_token(injected_t0)
        seq._injected_t0 = injected_t0
        stream_queue = mock.Mock()
        finished = self._accept_all(sched, seq, [10, 11, 12, 13], stream_queue)

        assert len(finished) == 1
        assert finished[0].leave_reason == "max_tokens"
        assert finished[0].num_completion_tokens == expected_count
        if expected_count == 0:
            assert finished[0].first_token_time == 0.0
        else:
            assert finished[0].first_token_time > 0.0
        emitted = stream_queue.put_nowait.call_args.args[0][0][1]
        assert emitted.output_tokens == expected_output
        assert emitted.finished is True
        assert emitted.finish_reason == "max_tokens"

    def test_earlier_eos_wins_over_later_cap(self, seq_factory):
        sched = self._scheduler()
        seq = self._prefill(
            sched,
            seq_factory(
                [1, 2, 3, 4],
                sampling_params=SamplingParams(max_tokens=3, ignore_eos=False),
            ),
        )
        stream_queue = mock.Mock()
        finished = self._accept_all(
            sched, seq, [10, sched.eos_token_id, 12, 13], stream_queue
        )

        assert finished[0].leave_reason == "eos"
        assert finished[0].num_completion_tokens == 2
        emitted = stream_queue.put_nowait.call_args.args[0][0][1]
        assert emitted.output_tokens == [10, sched.eos_token_id]

    def test_earlier_cap_wins_over_later_eos(self, seq_factory):
        sched = self._scheduler()
        seq = self._prefill(
            sched,
            seq_factory(
                [1, 2, 3, 4],
                sampling_params=SamplingParams(max_tokens=2, ignore_eos=False),
            ),
        )
        stream_queue = mock.Mock()
        finished = self._accept_all(
            sched, seq, [10, 11, sched.eos_token_id, 13], stream_queue
        )

        assert finished[0].leave_reason == "max_tokens"
        assert finished[0].num_completion_tokens == 2
        emitted = stream_queue.put_nowait.call_args.args[0][0][1]
        assert emitted.output_tokens == [10, 11]


class TestUndeferredSpecPlaceholders:
    """PP output is undeferred: every step leaves ONE run of `mtp_k + 1`.

    `verify_ctx_lens` shifts by 1 under undeferred output only because of this:
    the verify window starts `mtp_k + 1` placeholders back from the end, no
    matter how many drafts the previous step rejected.
    """

    mtp_k = 3

    def _scheduler(self, **config):
        sched = Scheduler(
            MockConfig(
                pipeline_parallel_size=2,
                **config,
                speculative_config=SimpleNamespace(
                    num_speculative_tokens=self.mtp_k,
                    use_dspark=lambda: True,
                ),
                num_kvcache_blocks=64,
            )
        )
        assert not sched.is_deferred_out and sched.use_spec
        return sched

    def _step(self, sched, seq, tokens, num_rejected, stream_queue=None):
        batch, seqs = sched.schedule()
        return sched.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(
                req_ids=[seq.id],
                token_ids=[tuple(tokens)],
                num_rejected=np.asarray([num_rejected]),
                num_bonus=np.asarray([0]),
                draft_token_ids=np.asarray([[7] * self.mtp_k]),
                is_deferred_out=False,
            ),
            stream_output_queue=stream_queue,
            batch=batch,
        )

    def _assert_one_full_run(self, sched, seq):
        width = self.mtp_k + 1
        tail = seq.token_ids[-width - 1 :]
        assert list(tail[1:]) == [sched.eos_token_id] * width
        assert tail[0] != sched.eos_token_id
        assert seq.num_placeholder_tokens == width

    def _prefilled(self, sched, seq_factory, max_tokens=64):
        seq = seq_factory(
            [1, 2, 3, 4],
            sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        )
        sched.add(seq)
        # The prefill step scores no drafts: one sampled token.
        self._step(sched, seq, [100], num_rejected=0)
        self._assert_one_full_run(sched, seq)
        return seq

    @pytest.mark.parametrize("num_rejected", range(mtp_k + 1))
    def test_every_reject_count_leaves_one_full_run(self, seq_factory, num_rejected):
        sched = self._scheduler()
        seq = self._prefilled(sched, seq_factory)
        accepted = self.mtp_k + 1 - num_rejected
        real_before = seq.num_tokens - seq.num_placeholder_tokens
        self._step(sched, seq, range(200, 200 + accepted), num_rejected)
        self._assert_one_full_run(sched, seq)
        assert seq.num_tokens - seq.num_placeholder_tokens == real_before + accepted

    def test_mixed_steps_never_grow_or_shrink_the_run(self, seq_factory):
        sched = self._scheduler()
        seq = self._prefilled(sched, seq_factory)
        for step, num_rejected in enumerate([0, self.mtp_k, 1, self.mtp_k, 2]):
            accepted = self.mtp_k + 1 - num_rejected
            self._step(sched, seq, range(10 * step, 10 * step + accepted), num_rejected)
            self._assert_one_full_run(sched, seq)

    def test_cap_truncates_an_all_accepted_step(self, seq_factory):
        # 1 token from prefill + 4 accepted against max_tokens=3.
        sched = self._scheduler()
        seq = self._prefilled(sched, seq_factory, max_tokens=3)
        stream_queue = mock.Mock()
        finished = self._step(
            sched, seq, [10, 11, 12, 13], num_rejected=0, stream_queue=stream_queue
        )
        assert len(finished) == 1
        assert finished[0].leave_reason == "max_tokens"
        assert finished[0].num_completion_tokens == 3
        emitted = stream_queue.put_nowait.call_args.args[0][0][1]
        assert emitted.output_tokens == [10, 11]

    def test_middle_chunk_gets_no_placeholders_once_its_final_chunk_is_scheduled(
        self, seq_factory
    ):
        # pp=2: the final chunk is scheduled while the middle one is in flight,
        # flipping the live `is_partial_prefill`. The middle chunk's
        # postprocess must still leave the seq without placeholders.
        sched = self._scheduler(max_num_batched_tokens=16)
        assert sched.advance_on_schedule
        seq = seq_factory(
            list(range(1, 33)),
            sampling_params=SamplingParams(max_tokens=64, ignore_eos=True),
        )
        sched.add(seq)
        middle, middle_seqs = sched.schedule()
        assert not middle.is_final_chunk[0]
        sched.schedule()
        assert not seq.is_partial_prefill
        sched.postprocess(
            list(middle_seqs.values()),
            ScheduledBatchOutput(
                req_ids=[seq.id],
                token_ids=[(5,)],
                num_rejected=np.asarray([0]),
                num_bonus=np.asarray([0]),
                draft_token_ids=np.asarray([[7] * self.mtp_k]),
                is_deferred_out=False,
            ),
            batch=middle,
        )
        assert seq.num_placeholder_tokens == 0
        assert seq.num_tokens == 32


@pytest.mark.parametrize("injected", [False, True])
def test_bad_logprobs_do_not_terminate_sibling_requests(seq_factory, injected):
    sched = Scheduler(MockConfig(num_kvcache_blocks=64))
    bad = seq_factory(
        [1, 2, 3], sampling_params=SamplingParams(max_tokens=8, logprobs=True)
    )
    good = seq_factory(
        [1, 2, 4], sampling_params=SamplingParams(max_tokens=8, logprobs=True)
    )
    sched.add(bad)
    sched.add(good)
    sched.schedule()
    if injected:
        bad._injected_t0 = 99
    queue = mock.Mock()
    finished = sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[bad.id, good.id],
            token_ids=[(10,), (11,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
            logprobs={
                bad.id: [token_logprob(-1, 1)] if injected else [],
                good.id: [token_logprob(-0.5, 1)],
            },
        ),
        stream_output_queue=queue,
    )
    assert bad in finished and good not in finished
    assert bad.leave_reason == "logprob_error"
    assert list(good.logprobs) == [-0.5]
    outputs = dict(queue.put_nowait.call_args.args[0])
    assert outputs[bad.id].finish_reason == "logprob_error"
    assert outputs[good.id].output_tokens == [11]
