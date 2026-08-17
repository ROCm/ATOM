import queue
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from conftest import MockConfig

pytest.importorskip("aiter", reason="requires AITER to import model_runner")

from atom.model_engine.engine_core import EngineCore
from atom.model_engine.model_runner import ModelRunner, tokenIDProcessor
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus
from atom.sampling_params import SamplingParams
from atom.utils import envs


def test_split_draft_dequeue_and_store_keep_generations_separate():
    processor = object.__new__(tokenIDProcessor)
    processor.is_deferred_out = True
    processor.prev_req_ids = [11, 12]
    previous = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    processor.recv_async_output_draft = mock.Mock(return_value=previous)
    processor.send_to_cpu_async_draft = mock.Mock()
    processor.num_spec_tokens = 3

    np.testing.assert_array_equal(processor.take_previous_draft_ids(), previous)

    current = object()
    processor.store_draft_ids(current)
    assert processor.draft_token_ids is current
    assert processor.pre_num_decode_token_per_seq == 4
    processor.send_to_cpu_async_draft.assert_called_once_with(current)


def test_terminal_cancel_drains_current_deferred_generation():
    processor = object.__new__(tokenIDProcessor)
    processor.token_ids_cpu = [object()]
    processor.draft_token_ids_cpu = []
    processor.recv_async_output = mock.Mock(return_value=object())
    processor.recv_logprobs = mock.Mock(return_value=None)
    processor.recv_mtp_status_async = mock.Mock(return_value=(None, None))
    processor.prev_batch = object()
    processor.prev_req_ids = [7]
    processor.prev_token_ids = object()
    processor.pre_num_decode_token_per_seq = 4
    processor.draft_token_ids = object()
    processor.prev_rejected_num = np.array([0])
    processor.prev_bonus_num = np.array([0])
    processor.num_rejected = np.array([0])
    processor.num_bonus = np.array([0])

    processor.discard_current_deferred_generation()

    processor.recv_async_output.assert_called_once_with(processor.token_ids_cpu)
    processor.recv_logprobs.assert_called_once_with()
    processor.recv_mtp_status_async.assert_called_once_with()
    assert processor.prev_batch is None
    assert processor.prev_req_ids is None
    assert processor.prev_token_ids is None
    assert processor.pre_num_decode_token_per_seq == 1
    assert processor.draft_token_ids is None


@pytest.mark.parametrize(
    "status_after_postprocess",
    [
        SequenceStatus.RUNNING,
        SequenceStatus.FINISHED,
    ],
)
def test_engine_publishes_output_before_finishing_or_cancelling_draft(
    status_after_postprocess,
):
    events = []
    seq = SimpleNamespace(status=SequenceStatus.RUNNING)
    batch = SimpleNamespace(req_ids=[1], connector_meta_output=None)
    fwd_out = ScheduledBatchOutput(
        req_ids=[1],
        token_ids=[(99,)],
        num_rejected=np.array([0], dtype=np.int32),
        num_bonus=np.array([0], dtype=np.int32),
        draft_token_ids=np.array([[1, 2, 3]], dtype=np.int32),
        is_deferred_out=True,
        draft_proposal_pending=True,
    )

    class RecordingQueue:
        def put_nowait(self, value):
            events.append(("output", value))

    def postprocess(*args, **kwargs):
        seq.status = status_after_postprocess
        kwargs["stream_output_queue"].put_nowait([(1, "token")])
        return [seq] if status_after_postprocess == SequenceStatus.FINISHED else []

    core = EngineCore.__new__(EngineCore)
    core.label = "test"
    core.kv_transfer_enabled = False
    core.scheduler = SimpleNamespace(
        schedule=lambda: (batch, {1: seq}),
        take_rejected=list,
        compute_detailed_aggregates=lambda *args: None,
        postprocess=postprocess,
    )

    def call_func(name, *args, **kwargs):
        events.append(("runner", name))
        if name == "forward":
            return fwd_out
        return None

    core.runner_mgr = SimpleNamespace(call_func=call_func)
    core._poll_kv_transfer_progress = lambda: None
    core.stream_output_queue = queue.Queue()
    core.output_queue = RecordingQueue()

    with mock.patch.object(envs, "ATOM_CANCEL_TERMINAL_MTP_PROPOSAL", True):
        assert core._process_engine_step_inner() is True

    expected_resolution = (
        "cancel_draft_proposal"
        if status_after_postprocess == SequenceStatus.FINISHED
        else "finish_draft_proposal"
    )
    resolution_idx = events.index(("runner", expected_resolution))
    output_indices = [i for i, event in enumerate(events) if event[0] == "output"]
    assert output_indices
    assert max(output_indices) < resolution_idx


def test_output_flag_defaults_to_synchronous():
    output = ScheduledBatchOutput([], [], None, None, None)
    assert output.draft_proposal_pending is False


def test_current_terminal_prefill_appends_token_without_spec_placeholders():
    spec_config = SimpleNamespace(
        num_speculative_tokens=3,
        use_dspark=lambda: False,
    )
    scheduler = Scheduler(
        MockConfig(
            speculative_config=spec_config,
            num_kvcache_blocks=20,
            max_model_len=64,
        )
    )
    seq = Sequence(
        [10, 11, 12, 13],
        block_size=4,
        sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
    )
    scheduler.add(seq)
    batch, scheduled = scheduler.schedule()

    assert batch.all_max_tokens_one is True
    finished = scheduler.postprocess(
        list(scheduled.values()),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(99,)],
            num_rejected=np.array([0], dtype=np.int32),
            num_bonus=np.array([0], dtype=np.int32),
            draft_token_ids=None,
            is_deferred_out=True,
            is_current_terminal_output=True,
        ),
        batch=batch,
    )

    assert finished == [seq]
    assert seq.status == SequenceStatus.FINISHED
    assert seq.leave_reason == "max_tokens"
    assert seq.token_ids == [10, 11, 12, 13, 99]
    assert seq.output_tokens == [99]
    assert seq.spec_token_ids.size == 0
    assert not scheduler.running


def test_current_terminal_prefill_after_middle_chunk_keeps_current_token():
    spec_config = SimpleNamespace(
        num_speculative_tokens=3,
        use_dspark=lambda: False,
    )
    scheduler = Scheduler(
        MockConfig(
            speculative_config=spec_config,
            num_kvcache_blocks=20,
            max_model_len=64,
            max_num_batched_tokens=4,
        )
    )
    seq = Sequence(
        [10, 11, 12, 13, 14, 15, 16, 17],
        block_size=4,
        sampling_params=SamplingParams(max_tokens=1, ignore_eos=True),
    )
    scheduler.add(seq)

    middle_batch, middle_scheduled = scheduler.schedule()
    assert middle_batch.is_final_chunk == [False]
    assert scheduler.postprocess(
        list(middle_scheduled.values()),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=middle_batch,
    ) == []
    assert seq.is_partial_prefill
    assert seq.output_tokens == []

    final_batch, final_scheduled = scheduler.schedule()
    assert final_batch.is_final_chunk == [True]
    finished = scheduler.postprocess(
        list(final_scheduled.values()),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(99,)],
            num_rejected=np.array([0], dtype=np.int32),
            num_bonus=np.array([0], dtype=np.int32),
            draft_token_ids=None,
            is_deferred_out=True,
            is_current_terminal_output=True,
        ),
        batch=final_batch,
    )

    assert finished == [seq]
    assert seq.status == SequenceStatus.FINISHED
    assert seq.leave_reason == "max_tokens"
    assert seq.token_ids == [10, 11, 12, 13, 14, 15, 16, 17, 99]
    assert seq.output_tokens == [99]
    assert seq.spec_token_ids.size == 0
    assert not scheduler.running


def test_current_terminal_fast_path_rejects_mixed_or_deferred_batches():
    runner = object.__new__(ModelRunner)
    runner.tokenID_processor = SimpleNamespace(
        is_deferred_out=True,
        prev_batch=None,
    )
    pure_final = SimpleNamespace(
        all_max_tokens_one=True,
        total_seqs_num=2,
        total_seqs_num_prefill=2,
        total_seqs_num_decode=0,
        is_final_chunk=[True, True],
        requires_followup_state_checkpoint=False,
    )
    assert runner._can_publish_current_terminal(pure_final)

    checkpoint_boundary = SimpleNamespace(**vars(pure_final))
    checkpoint_boundary.requires_followup_state_checkpoint = True
    assert not runner._can_publish_current_terminal(checkpoint_boundary)

    mixed = SimpleNamespace(**vars(pure_final))
    mixed.total_seqs_num_prefill = 1
    mixed.total_seqs_num_decode = 1
    assert not runner._can_publish_current_terminal(mixed)

    middle_chunk = SimpleNamespace(**vars(pure_final))
    middle_chunk.is_final_chunk = [True, False]
    assert not runner._can_publish_current_terminal(middle_chunk)

    runner.tokenID_processor.prev_batch = object()
    assert not runner._can_publish_current_terminal(pure_final)


def test_deferred_proposal_is_limited_to_plain_mtp():
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(
        eplb_enable=False,
        speculative_config=SimpleNamespace(method="mtp"),
    )
    runner.tokenID_processor = SimpleNamespace(is_deferred_out=True)
    runner.drafter = SimpleNamespace(verify_scheduler=None)
    batch = SimpleNamespace(is_dummy_run=False)

    with mock.patch.object(envs, "ATOM_DEFER_MTP_PROPOSAL", True):
        assert runner._can_defer_draft_proposal(batch)

        runner.config.speculative_config.method = "eagle3"
        assert not runner._can_defer_draft_proposal(batch)

        runner.config.speculative_config.method = "mtp"
        runner.drafter.verify_scheduler = object()
        assert not runner._can_defer_draft_proposal(batch)


def test_ordinary_deferred_proposal_is_experimental_and_off_by_default():
    assert not envs.ATOM_DEFER_MTP_PROPOSAL
    assert not envs.ATOM_CANCEL_TERMINAL_MTP_PROPOSAL
    assert envs.ATOM_TERMINAL_MTP_FAST_PATH
