# SPDX-License-Identifier: MIT
"""Runner wiring only; storage and protocol tests do not require this import."""

import weakref
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

runner_module = pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="runner integration requires AITER",
    exc_type=ImportError,
)


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("pcp_size", [1, 2])
def test_postprocess_submits_prefill_or_final_broadcast_counts(
    monkeypatch, verified, pcp_size
):
    """Packing must use the post-broadcast verdict, not stale local counts."""
    bs = 2
    verdict = torch.tensor([[1, 2], [4, 3], [91, 92]], dtype=torch.int32)
    broadcast_verdict = torch.tensor([[2, 1], [3, 4], [93, 94]], dtype=torch.int32)
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
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    monkeypatch.setattr(
        runner_module, "get_tp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(runner_module, "get_pcp_world_size", lambda: pcp_size)
    monkeypatch.setattr(
        runner_module,
        "get_pcp_group",
        lambda: SimpleNamespace(
            broadcast=lambda value, src: (
                broadcast_verdict if value is verdict else value
            )
        ),
    )

    class StatusSubmitted(Exception):
        pass

    send_status = mock.Mock(side_effect=StatusSubmitted)
    processor = SimpleNamespace(
        is_deferred_out=True,
        zero_spec_counts=torch.zeros(128, dtype=torch.int32),
        prev_batch=None,
        prev_rejected_num=None,
        prev_bonus_num=None,
        prepare_sampled_ids=mock.Mock(return_value=({7: (3,), 8: (5,), -1: 1}, None)),
        spec_status_queue=SimpleNamespace(send=send_status),
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
        runner_module.ModelRunner.postprocess(
            runner,
            SimpleNamespace(total_seqs_num=bs, return_logprobs=[False] * bs),
            torch.zeros(4, 8),
            torch.ones(bs),
            None,
            None,
            True,
            hidden_states=None,
        )
    send_status.assert_called_once()
    counts = send_status.call_args.args[0]
    if verified:
        expected = broadcast_verdict if pcp_size > 1 else verdict
        assert counts.data_ptr() == expected.data_ptr()
        torch.testing.assert_close(counts, expected[:2])
    else:
        torch.testing.assert_close(counts, torch.zeros((2, bs), dtype=torch.int32))
        assert counts.data_ptr() == processor.zero_spec_counts.data_ptr()
    assert counts.is_contiguous()
    assert send_status.call_args.kwargs["ready"] is runner.forward_done_event


@pytest.mark.parametrize("max_num_seqs", [1, 64])
def test_output_capacity_follows_server_limit_and_warmup_floor(max_num_seqs):
    runner = SimpleNamespace(
        config=SimpleNamespace(
            pipeline_parallel_size=1, kv_transfer_config={}, max_num_seqs=max_num_seqs
        ),
        attn_family=SimpleNamespace(has_recurrent_state=False),
        device="cuda",
    )
    with (
        mock.patch.object(runner_module, "CpuGpuBuffer"),
        mock.patch.object(torch.cuda, "Stream"),
        mock.patch.object(torch, "zeros") as zeros,
    ):
        processor = runner_module.tokenIDProcessor(
            runner, 1024, use_spec=True, num_spec_tokens=5
        )
    request_capacity = max(max_num_seqs, 2)
    assert processor._draft_token_ids_queue.capacity == request_capacity * 5
    assert processor.spec_status_queue.capacity == request_capacity * 2
    zeros.assert_called_once_with(
        2 * request_capacity, dtype=torch.int32, device="cuda"
    )
    assert processor.recv_draft_token_ids().size == 0
    assert processor.recv_spec_status() == (None, None)


class _FakeGpuTensor:
    """What send_to_cpu_async sees of a GPU tensor: `.to` makes a host copy."""

    def __init__(self, values):
        self.values = values

    def to(self, device, non_blocking=False):
        return self.values.clone()


@pytest.fixture
def cpu_processor(monkeypatch):
    def empty(*args, pin_memory=False, **kwargs):
        return original_empty(*args, **kwargs)

    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.cuda, "Event", mock.Mock)
    monkeypatch.setattr(torch.cuda, "current_stream", mock.Mock)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: mock.MagicMock())
    processor = object.__new__(runner_module.tokenIDProcessor)
    processor.async_copy_stream = mock.Mock()
    processor._output_request_capacity = 4
    processor.num_spec_tokens = 2
    processor.clean()
    return processor


def _record_alive_at(event, refs):
    seen = []
    event.synchronize.side_effect = lambda: seen.append(
        [ref() is not None for ref in refs]
    )
    return seen


def test_sampled_sources_are_held_until_recv_has_waited_on_copy_done(
    cpu_processor,
):
    """The handle owns the GPU ids and logprobs until their copy is waited on."""
    processor = cpu_processor
    ids = _FakeGpuTensor(torch.tensor([3, 4], dtype=torch.int32))
    logprobs = _FakeGpuTensor(torch.tensor([-0.5, -1.5]))
    refs = [weakref.ref(ids), weakref.ref(logprobs)]
    copy_done = mock.Mock()
    seen = _record_alive_at(copy_done, refs)
    processor.send_to_cpu_async(
        ids, processor.token_ids_cpu, mock.Mock(), copy_done, gpu_logprobs=logprobs
    )
    del ids, logprobs
    assert all(ref() is not None for ref in refs)

    token_ids = processor.recv_async_output(processor.token_ids_cpu)
    assert seen == [[True, True]]
    assert all(ref() is None for ref in refs)
    assert processor.token_ids_cpu == []
    assert token_ids.tolist() == [3, 4]
    assert processor.recv_logprobs() == [-0.5, -1.5]


def test_clean_waits_for_every_queued_copy_before_dropping_it(cpu_processor):
    processor = cpu_processor
    ids = _FakeGpuTensor(torch.tensor([3, 4], dtype=torch.int32))
    status = torch.tensor([[1, 1], [0, 0]], dtype=torch.int32)
    draft = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    refs = [weakref.ref(t) for t in (ids, status, draft)]
    copy_done = mock.Mock()
    processor.send_to_cpu_async(ids, processor.token_ids_cpu, mock.Mock(), copy_done)
    processor.spec_status_queue.send(status)
    processor.send_draft_token_ids_to_cpu(draft)
    events = [
        copy_done,
        processor.spec_status_queue._pending[0].copy_done,
        processor._draft_token_ids_queue._pending[0].copy_done,
    ]
    seen = [_record_alive_at(event, refs) for event in events]
    del ids, status, draft

    processor.clean()
    # Each copy is waited on exactly once, while its own source is held.
    for i, alive in enumerate(seen):
        assert len(alive) == 1 and alive[0][i]
    assert all(ref() is None for ref in refs)
    assert processor.token_ids_cpu == []
    assert processor.recv_spec_status() == (None, None)
    assert processor.recv_draft_token_ids().size == 0
