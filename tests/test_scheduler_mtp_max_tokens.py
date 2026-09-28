"""Regression coverage for MTP output batches crossing ``max_tokens``."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.sampling_params import SamplingParams


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
            if seq.return_logprobs:
                seq.logprobs.append(-0.1 * len(seq.logprobs))
        return sched.postprocess(
            list(sched.running),
            ScheduledBatchOutput(
                req_ids=[seq.id],
                token_ids=[tuple(tokens)],
                num_rejected=np.asarray([0]),
                num_bonus=np.asarray([0]),
                draft_token_ids=np.asarray([[20, 21, 22]]),
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
