from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="requires AITER to import model_runner")

from atom.model_engine.model_runner import ModelRunner, tokenIDProcessor
from atom.model_engine.scheduler import ScheduledBatch
from atom.spec_decode.eagle_proposer import EagleProposer


def _prefill_batch(is_final_chunk: list[bool]) -> ScheduledBatch:
    batch = object.__new__(ScheduledBatch)
    batch.scheduled_tokens = np.arange(4, dtype=np.int32)
    batch.total_tokens_num = 4
    batch.total_tokens_num_prefill = 4
    batch.total_tokens_num_decode = 0
    batch.total_seqs_num_prefill = len(is_final_chunk)
    batch.total_seqs_num_decode = 0
    batch.is_final_chunk = is_final_chunk
    return batch


def _processor() -> tokenIDProcessor:
    processor = object.__new__(tokenIDProcessor)
    processor.input_ids = SimpleNamespace(
        np=np.zeros(8, dtype=np.int32),
        gpu=np.zeros(8, dtype=np.int32),
        copy_to_gpu=mock.Mock(),
    )
    processor.recv_mtp_status_async = mock.Mock(
        return_value=(
            np.array([2], dtype=np.int32),
            np.array([1], dtype=np.int32),
        )
    )
    processor.prev_rejected_num = np.array([7], dtype=np.int32)
    processor.prev_bonus_num = np.array([8], dtype=np.int32)
    return processor


def test_middle_prefills_preserve_status_until_mixed_final_batch():
    processor = _processor()

    # Pure middle chunks skip postprocess, so neither the deferred-token queue
    # nor its matching MTP-status queue may advance.
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False]))
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False, False]))

    processor.recv_mtp_status_async.assert_not_called()
    np.testing.assert_array_equal(processor.prev_rejected_num, [7])
    np.testing.assert_array_equal(processor.prev_bonus_num, [8])

    # If any request reaches its final chunk, the batch runs postprocess. Its
    # status dequeue must therefore happen exactly once, even though another
    # request in the same batch is still a middle chunk.
    tokenIDProcessor.prepare_input_ids(processor, _prefill_batch([False, True]))

    processor.recv_mtp_status_async.assert_called_once_with()
    np.testing.assert_array_equal(processor.prev_rejected_num, [2])
    np.testing.assert_array_equal(processor.prev_bonus_num, [1])


def test_middle_prefill_uses_full_draft_proposal_for_dp_lockstep():
    runner = object.__new__(ModelRunner)
    runner.tokenID_processor = SimpleNamespace(
        input_ids=SimpleNamespace(gpu=torch.arange(8, dtype=torch.int32)),
        default_num_rejected_tokens=torch.zeros(8, dtype=torch.int32),
    )
    runner.drafter = mock.Mock()
    runner.drafter.prepare_inputs.return_value = torch.tensor([3])
    runner.drafter.anchors_to_gpu.return_value = torch.tensor([17])

    batch = SimpleNamespace(
        total_seqs_num=1,
        total_tokens_num=4,
        next_token_ids=[17],
    )
    hidden_states = torch.randn(4, 2)
    positions = torch.arange(4)

    with mock.patch(
        "atom.model_engine.model_runner.get_forward_context",
        return_value=SimpleNamespace(context=SimpleNamespace(positions=positions)),
    ):
        ModelRunner._advance_drafter_for_middle_chunk(runner, batch, hidden_states)

    runner.drafter.propose.assert_called_once()
    kwargs = runner.drafter.propose.call_args.kwargs
    assert torch.equal(
        kwargs["target_token_ids"], runner.tokenID_processor.input_ids.gpu[1:5]
    )
    assert kwargs["target_positions"] is positions
    assert kwargs["target_hidden_states"] is hidden_states
    assert torch.equal(
        kwargs["num_reject_tokens"],
        runner.tokenID_processor.default_num_rejected_tokens[:1],
    )
    assert torch.equal(kwargs["next_token_ids"], torch.tensor([17]))
    assert torch.equal(kwargs["last_token_indices"], torch.tensor([3]))


def test_eagle_middle_prefill_skips_one_pass_shortcut_under_dp():
    drafter = object.__new__(EagleProposer)
    drafter.config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=4)
    )
    drafter.prepare_inputs = mock.Mock()
    drafter.model = mock.Mock()

    with mock.patch(
        "atom.spec_decode.eagle_proposer.get_forward_context",
        return_value=SimpleNamespace(context=SimpleNamespace(batch_size=1)),
    ):
        EagleProposer.precompute_context_kv(
            drafter,
            positions=torch.arange(4),
            hidden_states=torch.randn(4, 2),
            next_token_ids=[17],
        )

    drafter.prepare_inputs.assert_not_called()
    drafter.model.assert_not_called()
