# SPDX-License-Identifier: MIT
"""Runner wiring only; storage and protocol tests do not require this import."""

import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

runner_module = pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="runner integration requires AITER",
    exc_type=ImportError,
)


def sampling_group(buffers):
    """The `sampling` members `prepare_sample` records counts for, as the real
    `PublicationGroup` (CPU count scratch only; nothing is published)."""
    from atom.utils.h2d import PublicationGroup

    names = [*buffers, "required_channel_states", "logprob_request_indices"]
    return PublicationGroup(
        None, "sampling", [SimpleNamespace(name=n, buffer=n) for n in names]
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
    sampled = torch.tensor([[3, 4, 0], [5, 6, 0]], dtype=torch.int32)
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
        prepare_sampled_ids=mock.Mock(return_value=({7: (3,), 8: (5,)}, None)),
        spec_status_queue=SimpleNamespace(send=send_status),
    )
    runner = SimpleNamespace(
        tokenID_processor=processor,
        sampler=mock.Mock(return_value=sampled[:, 0]),
        rejection_sampler=SimpleNamespace(
            forward=mock.Mock(return_value=(sampled, verdict)),
        ),
        drafter=SimpleNamespace(lossless=None),
        forward_done_event=mock.Mock(),
    )
    with pytest.raises(StatusSubmitted):
        runner_module.ModelRunner.postprocess(
            runner,
            SimpleNamespace(
                total_seqs_num=bs,
                return_logprobs=[False] * bs,
                any_logprobs=False,
                required_channel=None,
                num_spec_step=2 if verified else 0,
            ),
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


@pytest.mark.parametrize("all_greedy", [False, True])
@pytest.mark.parametrize("verified", [False, True])
@pytest.mark.parametrize("armed", [False, True])
def test_a_filtered_batch_reaches_one_law_for_every_draw(
    monkeypatch, verified, armed, all_greedy
):
    """The ratio verify gets the drafter's q and the batch's own `top_ks`/`top_ps`; the
    bonus and an unarmed exact-match draw stay `Sampler`'s, one law per
    process. An all-greedy verify asks for neither."""
    bs = 2
    temperatures, top_ks, top_ps = torch.ones(bs), torch.tensor([5, 0]), 0.9
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
    ratio = object() if armed else None
    lossless = SimpleNamespace(verify_inputs=mock.Mock(return_value=ratio))
    drafter = SimpleNamespace(lossless=lossless)
    runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(zero_spec_counts=torch.zeros(8)),
        sampler=mock.Mock(side_effect=None if verified else _Reached),
        rejection_sampler=SimpleNamespace(forward=mock.Mock(side_effect=_Reached)),
        drafter=drafter,
    )
    runner._constrain_required_channel = MethodType(
        runner_module.ModelRunner._constrain_required_channel, runner
    )
    runner.sampler.sample_verification_tokens = mock.Mock(
        return_value=torch.zeros(bs, dtype=torch.int32)
    )
    with pytest.raises(_Reached):
        runner_module.ModelRunner.postprocess(
            runner,
            SimpleNamespace(
                total_seqs_num=bs,
                return_logprobs=[False] * bs,
                any_logprobs=False,
                required_channel=None,
            ),
            torch.zeros(4, 8),
            temperatures,
            top_ks,
            top_ps,
            all_greedy,
            hidden_states=None,
        )
    runner.sampler.assert_called_once()
    if not verified:
        return
    if all_greedy:
        lossless.verify_inputs.assert_not_called()
        assert runner.rejection_sampler.forward.call_args.kwargs["ratio"] is None
        assert not runner.sampler.sample_verification_tokens.called
        return
    passed_bs, passed_temps, passed_ks, passed_ps = (
        lossless.verify_inputs.call_args.args
    )
    assert passed_bs == bs and passed_temps is temperatures
    assert passed_ks is top_ks and passed_ps is top_ps
    assert runner.rejection_sampler.forward.call_args.kwargs["ratio"] is ratio
    # Unarmed, the exact-match draw is the flag-off law, as the bonus is.
    assert runner.sampler.sample_verification_tokens.called is not armed


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
def test_a_uniform_k_reaches_the_sampler_as_a_scalar(ks, ps):
    """As on the default path, whatever top-p sits beside it: keeping top-k
    ties is `Sampler`'s rule alone (`Sampler._per_row_k`)."""
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
    group = sampling_group(buffers)
    runner = SimpleNamespace(forward_vars=buffers)
    batch = SimpleNamespace(
        total_seqs_num=bs,
        temperatures=np.ones(bs, np.float32),
        top_ks=np.array(ks, np.int32),
        top_ps=np.array(ps, np.float32),
        required_channel=None,
    )
    _, top_ks, _, _, _ = runner_module.ModelRunner.prepare_sample(runner, batch, group)
    assert type(top_ks) is int and top_ks == ks[0]
    assert group.counts[group.indices["top_ks"]] is None


@pytest.mark.parametrize(
    "ks, ps", [([-1, -1], [1.0, 1.0]), ([4, 4], [0.9, 0.9]), ([4, -1], [1.0, 0.7])]
)
def test_a_lossless_draft_gets_every_rows_filters_published(ks, ps):
    """The draft samples within each row's own filters, read from the target's
    buffers, so they are published whatever the batch holds: off and uniform
    filters included. `Sampler` and the verify still get the batch's scalars
    where it is uniform, so a uniform filter keeps its own variant."""
    bs = len(ks)
    buffers = {
        name: SimpleNamespace(np=np.zeros(4, dtype), gpu=torch.zeros(4))
        for name, dtype in (
            ("temperatures", np.float32),
            ("top_ks", np.int32),
            ("top_ps", np.float32),
        )
    }
    group = sampling_group(buffers)
    runner = SimpleNamespace(
        forward_vars=buffers,
        drafter=SimpleNamespace(lossless=SimpleNamespace(stage=mock.Mock())),
        _verifies_drafts=lambda batch, is_prefill: not is_prefill,
        _is_pure_middle_chunk=lambda batch: False,
    )
    batch = SimpleNamespace(
        total_seqs_num=bs,
        temperatures=np.ones(bs, np.float32),
        top_ks=np.array(ks, np.int32),
        top_ps=np.array(ps, np.float32),
        required_channel=None,
        is_dummy_run=False,
        total_tokens_num_prefill=0,
    )
    _, top_ks, top_ps, _, _ = runner_module.ModelRunner.prepare_sample(
        runner, batch, group
    )
    assert buffers["top_ks"].np[:bs].tolist() == ks
    np.testing.assert_array_equal(buffers["top_ps"].np[:bs], np.float32(ps))
    assert group.counts[group.indices["top_ks"]] == bs
    assert group.counts[group.indices["top_ps"]] == bs
    if len(set(ks)) == 1 and ks[0] != -1:
        assert top_ks == ks[0]
    if len(set(ps)) == 1 and ps[0] < 1.0:
        assert top_ps == np.float32(ps[0])
    if len(set(ps)) > 1:
        assert top_ps.data_ptr() == buffers["top_ps"].gpu.data_ptr()


@pytest.mark.parametrize(
    "temps, prefill, verifies",
    [([0.0, 0.0], 0, False), ([0.0, 0.7], 0, True), ([0.0, 0.7], 3, False)],
    ids=["all-greedy", "sampled", "prefill"],
)
def test_the_draft_q_lookup_is_gated_like_its_consumer(temps, prefill, verifies):
    """`postprocess` asks for the ratio verify's q only on a forward that
    scores drafts and is not all greedy; staging looks q up under the same
    gate, so no verify reads bases a skipped step left stale."""
    bs = len(temps)
    buffers = {
        name: SimpleNamespace(np=np.zeros(4, dtype), gpu=torch.zeros(4))
        for name, dtype in (
            ("temperatures", np.float32),
            ("top_ks", np.int32),
            ("top_ps", np.float32),
        )
    }
    group = sampling_group(buffers)
    staged = mock.Mock()
    runner = SimpleNamespace(
        forward_vars=buffers,
        drafter=SimpleNamespace(lossless=SimpleNamespace(stage=staged)),
        _verifies_drafts=lambda batch, is_prefill: not is_prefill,
        _is_pure_middle_chunk=lambda batch: False,
    )
    batch = SimpleNamespace(
        total_seqs_num=bs,
        temperatures=np.array(temps, np.float32),
        top_ks=np.full(bs, -1, np.int32),
        top_ps=np.ones(bs, np.float32),
        required_channel=None,
        is_dummy_run=False,
        total_tokens_num_prefill=prefill,
    )
    *_, all_greedy, _ = runner_module.ModelRunner.prepare_sample(runner, batch, group)
    consumer_reads_q = not all_greedy and runner._verifies_drafts(batch, prefill > 0)
    assert consumer_reads_q is verifies
    assert staged.call_args.kwargs["verifies"] is verifies
    # Into the publication the sampling inputs ride: one upload.
    assert staged.call_args.args[2] is group


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
    assert processor.recv_logprobs().tolist() == [-0.5, -1.5]


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


@pytest.mark.parametrize("armed", [False, True])
def test_an_armed_runner_binds_the_draft_and_arms_the_ratio_verify(monkeypatch, armed):
    """Armed, the runner binds the draft to its filters, keeps top-k ties and
    arms the ratio verify (scratch and compile) for the model dtype; unarmed it
    touches none of them."""
    import numpy as np

    top_ks = SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32), np=np.zeros(4))
    lossless = SimpleNamespace(bind_target=mock.Mock()) if armed else None
    model = SimpleNamespace(vocab_size=160)
    runner = SimpleNamespace(
        drafter=SimpleNamespace(lossless=lossless, model=model, mtp_k=3),
        forward_vars={"top_ks": top_ks},
        sampler=SimpleNamespace(keep_ties=mock.Mock()),
        rejection_sampler=SimpleNamespace(arm_ratio=mock.Mock()),
        device="cpu",
    )
    config = SimpleNamespace(max_num_seqs=8, torch_dtype=torch.bfloat16)
    runner_module.ModelRunner._arm_lossless(runner, config)
    if not armed:
        runner.sampler.keep_ties.assert_not_called()
        runner.rejection_sampler.arm_ratio.assert_not_called()
        return
    lossless.bind_target.assert_called_once_with(runner.forward_vars, model)
    runner.sampler.keep_ties.assert_called_once_with(top_ks.gpu)
    runner.rejection_sampler.arm_ratio.assert_called_once_with(
        8, 3, 160, torch.bfloat16, "cpu"
    )
