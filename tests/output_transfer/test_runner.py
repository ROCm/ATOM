# SPDX-License-Identifier: MIT
"""Runner wiring only; storage and protocol tests do not require this import."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

mod = pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="runner integration requires AITER",
    exc_type=ImportError,
)
from atom.model_engine.scheduler import ScheduledBatchOutput


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("pcp", [1, 2])
def test_postprocess_passes_the_final_verdict_or_prefill_fallback(
    monkeypatch, verified, pcp
):
    """Packing must use the post-broadcast verdict, not stale local counts."""
    bs = 2
    verdict = torch.tensor([[1, 2], [4, 3], [91, 92]], dtype=torch.int32)
    aligned = torch.tensor([[2, 1], [3, 4], [93, 94]], dtype=torch.int32)
    sampled = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    spec = (
        SimpleNamespace(
            bonus_logits_indices=torch.tensor([1, 3]),
            target_logits_indices=torch.tensor([0, 2]),
        )
        if verified
        else None
    )
    monkeypatch.setattr(
        mod, "get_forward_context", lambda: SimpleNamespace(spec_decode_metadata=spec)
    )
    monkeypatch.setattr(mod, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(mod, "get_pcp_world_size", lambda: pcp)
    monkeypatch.setattr(
        mod,
        "get_pcp_group",
        lambda: SimpleNamespace(
            broadcast=lambda value, src: aligned if value is verdict else value
        ),
    )

    class StatusSubmitted(Exception):
        pass

    send_counts = mock.Mock(side_effect=StatusSubmitted)
    send_verdict = mock.Mock(side_effect=StatusSubmitted)
    processor = SimpleNamespace(
        is_deferred_out=True,
        default_num_rejected_tokens=torch.zeros(bs, dtype=torch.int32),
        prev_batch=None,
        prev_rejected_num=None,
        prev_bonus_num=None,
        prepare_sampled_ids=mock.Mock(return_value=({7: (3,), 8: (5,), -1: 1}, None)),
        mtp_status=SimpleNamespace(send_counts=send_counts, send_verdict=send_verdict),
    )
    runner = SimpleNamespace(
        tokenID_processor=processor,
        sampler=mock.Mock(return_value=sampled[:, 0]),
        rejection_sampler=SimpleNamespace(
            forward=mock.Mock(return_value=(sampled, verdict)),
        ),
        drafter=object(),
        forward_done_event=mock.Mock(),
    )
    with pytest.raises(StatusSubmitted):
        mod.ModelRunner.postprocess(
            runner,
            SimpleNamespace(total_seqs_num=bs, return_logprobs=[False] * bs),
            torch.zeros(4, 8),
            torch.ones(bs),
            None,
            None,
            True,
            hidden_states=None,
        )
    if verified:
        send_counts.assert_not_called()
        send_verdict.assert_called_once()
        result, ready = send_verdict.call_args.args
        assert result is (aligned if pcp > 1 else verdict)
    else:
        send_verdict.assert_not_called()
        send_counts.assert_called_once()
        rejected, bonus, ready = send_counts.call_args.args
        torch.testing.assert_close(rejected, torch.zeros(bs, dtype=torch.int32))
        torch.testing.assert_close(bonus, torch.zeros(bs, dtype=torch.int32))
    assert ready is runner.forward_done_event


@pytest.mark.parametrize("max_requests", [1, 64])
def test_server_request_limit_sizes_draft_and_status_slots(fake_cuda, max_requests):
    runner = SimpleNamespace(
        config=SimpleNamespace(
            pipeline_parallel_size=1, kv_transfer_config={}, max_num_seqs=max_requests
        ),
        attn_family=SimpleNamespace(has_recurrent_state=False),
        device="cuda",
    )
    with (
        mock.patch.object(mod, "CpuGpuBuffer"),
        mock.patch.object(torch.cuda, "Stream", return_value=fake_cuda.stream),
        mock.patch.object(torch, "zeros"),
    ):
        processor = mod.tokenIDProcessor(runner, 1024, use_spec=True, num_spec_tokens=5)
    requests = max(max_requests, 2)  # Existing block-drafter warmup floor.
    assert processor._draft_buffers.capacity == requests * 5
    assert processor.mtp_status._buffers.capacity == requests * 2
    draft = torch.arange(requests * 5).reshape(requests, 5)
    processor.send_to_cpu_async_draft(draft)
    saved_draft = processor.recv_async_output_draft()
    verdict = torch.arange(requests * 3).reshape(3, requests)
    processor.mtp_status.send_verdict(verdict, fake_cuda.ready)
    rejected, bonus = processor.recv_mtp_status_async()
    output = ScheduledBatchOutput([7], [(9,)], rejected, bonus, saved_draft)
    assert output.draft_token_ids is saved_draft
    assert output.num_rejected is rejected and output.num_bonus is bonus
    processor.send_to_cpu_async_draft(draft + 100)
    processor.mtp_status.send_verdict(verdict + 100, fake_cuda.ready)
    processor.clean()
    assert processor.recv_async_output_draft().size == 0
    assert processor.recv_mtp_status_async() == (None, None)
    np.testing.assert_array_equal(output.draft_token_ids, draft.numpy())
    np.testing.assert_array_equal(output.num_rejected, verdict[1].numpy())
