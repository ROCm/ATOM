# SPDX-License-Identifier: MIT
"""Runner wiring only; storage and protocol tests do not require this import."""

import weakref
from contextlib import nullcontext
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
        drafter=SimpleNamespace(),
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


class _Reached(Exception):
    pass


@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("armed", [False, True])
def test_a_filtered_batch_reaches_one_law_for_every_draw(monkeypatch, verified, armed):
    """The ratio verify gets the batch's own `top_ks`/`top_ps`, and no draw
    is told how to cut top-k: that is `Sampler`'s own, one law per process."""
    bs = 2
    top_ks, top_ps = torch.tensor([5, 0]), 0.9
    spec = (
        SimpleNamespace(
            bonus_logits_indices=torch.tensor([1, 3]),
            target_logits_indices=torch.tensor([0, 2]),
            cu_num_draft_tokens=torch.tensor([1, 2]),
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
    q = (object(), torch.zeros(bs), object()) if armed else (None, None, None)
    drafter = SimpleNamespace(lossless_verify_inputs=mock.Mock(return_value=q))
    runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(zero_spec_counts=torch.zeros(8)),
        sampler=mock.Mock(side_effect=None if verified else _Reached),
        rejection_sampler=SimpleNamespace(forward=mock.Mock(side_effect=_Reached)),
        drafter=drafter,
    )
    runner.sampler.sample_verification_tokens = mock.Mock(
        return_value=torch.zeros(bs, dtype=torch.int32)
    )
    with pytest.raises(_Reached):
        runner_module.ModelRunner.postprocess(
            runner,
            SimpleNamespace(total_seqs_num=bs, return_logprobs=[False] * bs),
            torch.zeros(4, 8),
            torch.ones(bs),
            top_ks,
            top_ps,
            False,
            hidden_states=None,
        )
    runner.sampler.assert_called_once()
    assert "keep_top_k_ties" not in runner.sampler.call_args.kwargs
    if not verified:
        return
    forward = runner.rejection_sampler.forward.call_args.kwargs
    assert forward["top_ks"] is top_ks and forward["top_ps"] is top_ps
    assert forward["draft_probs"] is q[0]
    # Unarmed, the exact-match draw is the flag-off law, as the bonus is.
    assert runner.sampler.sample_verification_tokens.called is not armed


def test_a_rapidserve_prefill_draws_by_its_samplers_own_law(monkeypatch):
    sampled = torch.tensor([3, 4])
    prep = (torch.zeros(2), torch.ones(2), torch.tensor([5, 0]), 0.9, False, True)
    stream = mock.Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(runner_module, "reset_forward_context", lambda: None)
    runner = mock.Mock(
        spec=[
            "_advance_forward_vars",
            "_gate_staging_reuse",
            "prepare_model",
            "_mark_staging_h2d_enqueued",
            "run_model",
            "sampler",
            "_record_forward_vars_event",
            "_record_kv_cache_ready",
        ]
    )
    runner.prepare_model.return_value = prep
    runner.run_model.return_value = (torch.zeros(2, 8), None)
    runner.sampler.return_value = sampled
    out = runner_module.RapidServeModelRunner.prefill_forward(runner, SimpleNamespace())
    assert out == [3, 4]
    args, kwargs = runner.sampler.call_args
    assert args[2] is prep[2] and args[3] is prep[3]
    assert "keep_top_k_ties" not in kwargs


@pytest.mark.parametrize("keeps", [False, True])
@pytest.mark.parametrize(
    "ks, ps",
    [
        ([4, 4], [0.9, 0.9]),
        ([4, 4], [0.9, 0.7]),
        ([4, 4], [1.0, 1.0]),
        ([4], [0.9]),
    ],
    ids=["uniform-p", "per-row-p", "no-p", "bs1"],
)
def test_a_sampler_that_keeps_ties_gets_a_per_row_k_beside_a_top_p(
    monkeypatch, keeps, ks, ps
):
    """A scalar k with a top-p would take aiter's top-k-first path, which
    splits a tie group; the published per-row buffer stands in for it, so no
    draw allocates one."""
    import numpy as np

    bs = len(ks)
    buffers = {
        name: SimpleNamespace(np=np.zeros(4, dtype), gpu=torch.zeros(4))
        for name, dtype in (
            ("temperatures", np.float32),
            ("top_ks", np.int32),
            ("top_ps", np.float32),
        )
    }
    names = list(buffers)
    group = SimpleNamespace(
        counts=[None] * len(names), indices={n: i for i, n in enumerate(names)}
    )
    runner = SimpleNamespace(
        forward_vars=buffers, sampler=SimpleNamespace(keep_top_k_ties=keeps)
    )
    batch = SimpleNamespace(
        total_seqs_num=bs,
        temperatures=np.ones(bs, np.float32),
        top_ks=np.array(ks, np.int32),
        top_ps=np.array(ps, np.float32),
    )
    _, top_ks, _, _, _ = runner_module.ModelRunner.prepare_sample(
        runner, batch, publication_group=group
    )
    per_row = keeps and min(ps) < 1.0
    if per_row:
        assert top_ks.data_ptr() == buffers["top_ks"].gpu.data_ptr()
        assert top_ks.shape == (bs,)
        assert buffers["top_ks"].np[:bs].tolist() == ks
        assert group.counts[group.indices["top_ks"]] == bs
    else:
        assert type(top_ks) is int and top_ks == ks[0]
        assert group.counts[group.indices["top_ks"]] is None


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
