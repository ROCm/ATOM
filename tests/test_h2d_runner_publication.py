# SPDX-License-Identifier: MIT
"""Real runner producers, variable counts and asynchronous staging reuse."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import CpuGpuBuffer
from atom.utils.h2d import PublicationError

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_H2D_GPU_TESTS") != "1", reason="set RUN_H2D_GPU_TESTS=1"
)


def runner_with_buffers(monkeypatch, transport, pp_size=1, speculative=False):
    from atom.model_engine.model_runner import ModelRunner, tokenIDProcessor
    from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpMetadataBuilder

    monkeypatch.setenv("ATOM_H2D_BACKEND", transport)
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.config = SimpleNamespace(
        pipeline_parallel_size=pp_size,
        max_num_batched_tokens=64,
        max_num_seqs=8,
        hf_config=SimpleNamespace(hidden_size=8),
        torch_dtype=torch.float32,
    )
    runner.model = SimpleNamespace()
    runner.use_mrope = True
    runner.enforce_eager = True
    runner.tokenID_processor = tokenIDProcessor(runner, 64)
    if speculative:
        from atom.spec_decode.drafter import Drafter

        class MetadataDrafter(Drafter):
            def _resolve_mtp_k(self):
                return 3

            def propose(self, *args):
                raise NotImplementedError

        runner.drafter = MetadataDrafter.__new__(MetadataDrafter)
        runner.drafter.runner = runner
        runner.drafter.mtp_k = 3
        runner.drafter.metadata_buffers = Drafter._allocate_metadata_buffers(
            8, 3, runner.device
        )
        runner.arange_np = np.arange(64, dtype=np.int32)
    runner.allocate_forward_vars()
    runner.forward_vars["cu_seqlens_q"] = CpuGpuBuffer(
        9, dtype=torch.int32, device=runner.device, publication_group="early"
    )
    builder = Qwen4ExpMetadataBuilder.__new__(Qwen4ExpMetadataBuilder)
    builder.model_runner = runner
    runner.attn_metadata_builder = builder
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    return runner


@pytest.mark.parametrize("transport", ["direct", "packed"])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_sampling_optional_members_and_scalar_filters(monkeypatch, transport, pp_size):
    from atom.model_ops.sampler import SAMPLER_EPS

    runner = runner_with_buffers(monkeypatch, transport, pp_size)
    results = []
    cases = (
        ([0, 0, 0], [-1, -1, -1], [1, 1, 1]),
        ([0.2, 0.4, 0.6], [4, 4, 4], [0.9, 0.9, 0.9]),
        ([1, 0], [8, 2], [0.5, 0.8]),
        ([0], [-1], [1]),
        ([0.7], [4], [0.9]),
        ([0.5, 1], [4, 4], [0.7, 0.8]),
        ([0.5, 1], [4, 7], [0.8, 0.8]),
    ) * 2
    for temps, ks, ps in cases:
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        for name in ("temperatures", "top_ks", "top_ps"):
            runner.forward_vars[name].gpu.fill_(-99)
        batch = SimpleNamespace(
            total_seqs_num=len(temps),
            temperatures=np.array(temps),
            top_ks=np.array(ks),
            top_ps=np.array(ps),
            needs_independent_noise=np.array([True] * len(temps)),
        )
        torch.cuda._sleep(2_000_000)
        t, k, p, greedy, noise = runner.prepare_sample(batch)
        results.append(
            (
                t.clone(),
                k.clone() if isinstance(k, torch.Tensor) else k,
                p.clone() if isinstance(p, torch.Tensor) else p,
                greedy,
                noise,
            )
        )
        counts = runner.h2d_groups["sampling"].counts
        for name in ("top_ks", "top_ps"):
            count = counts[runner.h2d_groups["sampling"].indices[name]]
            buf = runner.forward_vars[name]
            # The omitted or unselected tail is never overwritten.
            results[-1] += (buf.gpu[count or 0 :].clone(),)
        with pytest.raises(PublicationError, match="republish_reason"):
            runner.forward_vars["temperatures"].copy_to_gpu(len(temps))
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for (temps, ks, ps), (t, k, p, greedy, noise, ktail, ptail) in zip(cases, results):
        np.testing.assert_array_equal(
            t.cpu().numpy().view(np.uint8),
            np.maximum(temps, SAMPLER_EPS).astype(np.float32).view(np.uint8),
        )
        assert greedy == (np.array(temps) == 0).all() and noise
        if all(value == -1 for value in ks):
            assert k is None
        elif len(set(ks)) == 1:
            assert type(k) is int and k == ks[0]
        else:
            assert k.cpu().tolist() == ks
        if all(value == 1 for value in ps):
            assert p is None
        elif len(set(ps)) == 1:
            assert type(p) is float and p == float(np.float32(ps[0]))
        else:
            np.testing.assert_array_equal(
                p.cpu().numpy().view(np.uint8),
                np.asarray(ps, dtype=np.float32).view(np.uint8),
            )
        assert torch.all(ktail.cpu() == -99) and torch.all(ptail.cpu() == -99)


@pytest.mark.parametrize("transport", ["direct", "packed"])
def test_input_ids_deferred_and_new_requests_share_one_publication(
    monkeypatch, transport
):
    runner = runner_with_buffers(monkeypatch, transport)
    processor = runner.tokenID_processor
    runner._gate_staging_reuse()
    processor.prev_batch = SimpleNamespace(req_ids=[10, 20], is_dummy_run=False)
    processor.prev_token_ids = torch.tensor([71, 82], dtype=torch.int32, device="cuda")
    batch = SimpleNamespace(
        scheduled_tokens=np.array([-1, 93, -1], dtype=np.int32),
        total_tokens_num=3,
        total_tokens_num_prefill=0,
        total_tokens_num_decode=3,
        total_seqs_num_prefill=0,
        total_seqs_num_decode=3,
        total_seqs_num=3,
        req_ids=[20, 30, 10],
        is_dummy_run=False,
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_rejected=np.zeros(3, dtype=np.int32),
        num_bonus=np.zeros(3, dtype=np.int32),
        produces_output=lambda: True,
    )
    runner.attn_metadata_builder.publish_cu_seqlens_q(
        batch, SimpleNamespace(running_bs=3)
    )
    torch.cuda._sleep(2_000_000)
    ids = processor.prepare_input_ids(batch, 1).clone()
    with pytest.raises(PublicationError, match="republish_reason"):
        processor.input_ids.copy_to_gpu(3)
    runner._mark_staging_h2d_enqueued()
    runner.h2d_owner.completion.synchronize()
    assert ids.cpu().tolist() == [82, 93, 71]


@pytest.mark.parametrize("pp_size", [1, 2])
def test_prefill_and_first_decode_ids_rotate_with_the_runner(monkeypatch, pp_size):
    runner = runner_with_buffers(monkeypatch, "direct", pp_size)
    outputs = []
    for i in range(6):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        processor = runner.tokenID_processor
        assert processor.decode_src is runner.forward_vars["decode_src"]
        prefill = i % 2 == 0
        batch = SimpleNamespace(
            scheduled_tokens=np.array([i + 1, i + 2], dtype=np.int32),
            total_tokens_num=2,
            total_tokens_num_prefill=2 if prefill else 0,
            total_tokens_num_decode=0 if prefill else 2,
            total_seqs_num_prefill=1 if prefill else 0,
            produces_output=lambda: False,
        )
        torch.cuda._sleep(2_000_000)
        outputs.append(processor.prepare_input_ids(batch, 1).clone())
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    assert [out.cpu().tolist() for out in outputs] == [[i + 1, i + 2] for i in range(6)]


@pytest.mark.parametrize("pp_size", [1, 2])
def test_mrope_padding_is_final_before_its_only_publication(monkeypatch, pp_size):
    runner = runner_with_buffers(monkeypatch, "packed", pp_size)
    outputs = []
    for i, bs in enumerate((4, 3, 1, 4, 2, 1)):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        buf = runner.forward_vars["mrope_positions"]
        assert buf._publication.unit == "elements"
        buf.gpu.fill_(-999)
        batch = SimpleNamespace(
            total_tokens_num_decode=bs * 2,
            req_ids=range(bs),
            mrope_position_deltas={j: j * 10 for j in range(bs)},
        )
        ends = np.arange(bs) * 100 + i * 2 + 2
        torch.cuda._sleep(2_000_000)
        positions = runner.attn_metadata_builder._build_mrope_decode_positions(
            batch, ends, 2, running_tokens=8
        )
        assert positions.stride(0) == 8
        outputs.append(
            (
                positions.clone(),
                runner._mrope_positions_view(8).clone(),
                buf.gpu.reshape(-1)[24:].clone(),
                bs,
                i,
            )
        )
        with pytest.raises(PublicationError, match="republish_reason"):
            runner.attn_metadata_builder._copy_mrope_to_gpu(8)
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for positions, padded, tail, bs, i in outputs:
        expected = [j * 110 + i * 2 + k for j in range(bs) for k in range(2)]
        assert positions.cpu().tolist() == [expected] * 3
        assert torch.all(padded.cpu()[:, bs * 2 :] == 0)
        assert torch.all(tail.cpu() == -999)


@pytest.mark.parametrize(
    "transport,coalesce",
    [("direct", False), ("packed", False), ("packed", True)],
)
@pytest.mark.parametrize("pp_size", [1, 2])
def test_speculative_indices_publish_before_index_select(
    monkeypatch, transport, coalesce, pp_size
):
    from atom.spec_decode.drafter import Drafter

    runner = runner_with_buffers(monkeypatch, transport, pp_size, speculative=True)
    scratch = runner.forward_vars["verification_draft_token_ids"]
    saved = []
    for step, lengths in enumerate(([3, 1, 2], [1, 1], [4] * 8, [1, 3, 1]) * 2):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        # Device writes and sampler reads are ordered on the forward stream,
        # including across PP slots; no pinned host scratch is involved.
        assert runner.forward_vars["verification_draft_token_ids"] is scratch
        scratch.fill_(-123)
        lengths = np.array(lengths, dtype=np.int32)
        ends = np.cumsum(lengths)
        ids = torch.arange(int(ends[-1]), device="cuda", dtype=torch.int32) + step * 100
        torch.cuda._sleep(2_000_000)
        prepared = None
        if coalesce:
            group = runner.h2d_groups["token_inputs"]
            prepared = runner.drafter.prepare_spec_decode_indices(lengths, ends, group)
            group.publish(group.counts)
        metadata = runner.drafter.calc_spec_decode_metadata(
            lengths, ends, ids, prepared_indices=prepared
        )
        if coalesce:
            assert runner.h2d_groups["spec_decode"]._backend is None
        assert (
            metadata.draft_token_ids.untyped_storage().data_ptr() == scratch.data_ptr()
        )
        saved_tail = scratch[metadata.draft_token_ids.numel() :].clone()
        anchors = Drafter.anchors_to_gpu(runner.drafter, [-1] * len(lengths))
        saved.append(
            (
                metadata.draft_token_ids.clone(),
                metadata.target_logits_indices.clone(),
                metadata.bonus_logits_indices.clone(),
                metadata.cu_num_draft_tokens.clone(),
                anchors.clone(),
                lengths,
                step,
                saved_tail,
            )
        )
        # Even a zero-draft step explicitly published its empty index prefix.
        with pytest.raises(PublicationError, match="republish_reason"):
            runner.forward_vars["target_logits_indices"].copy_to_gpu(0)
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for drafts, targets, bonus, cu, anchors, lengths, step, tail in saved:
        offsets = np.cumsum(lengths) - lengths
        expected_targets = [
            int(start + j)
            for start, length in zip(offsets, lengths)
            for j in range(int(length) - 1)
        ]
        assert targets.cpu().tolist() == expected_targets
        assert drafts.cpu().tolist() == [step * 100 + j + 1 for j in expected_targets]
        assert bonus.cpu().tolist() == (np.cumsum(lengths) - 1).tolist()
        assert cu.cpu().tolist() == np.cumsum(lengths - 1).tolist()
        assert anchors.cpu().tolist() == [-1] * len(lengths)
        assert torch.all(tail.cpu() == -123)


@pytest.mark.parametrize("phase", ["prefill", "first_decode", "deferred"])
@pytest.mark.parametrize("transport", ["direct", "packed"])
def test_prepare_model_publishes_sampling_and_query_prefix_before_token_consumer(
    monkeypatch, phase, transport
):
    from atom.model_engine import model_runner

    runner = runner_with_buffers(monkeypatch, transport)
    runner.config.parallel_config = SimpleNamespace(data_parallel_size=1)
    runner.config.enable_tbo = False
    runner.capture_sizes_np = np.array([1, 2, 4, 8])
    runner._dspark_apply_q_bucket = lambda batch: None
    runner._piecewise_cg_active = lambda: False
    runner._local_tbo_eligibility = lambda batch: False
    runner.prepare_inputs = lambda *args, **kwargs: None
    mode = SimpleNamespace(sync=None, max_seqlen_q=1, running_bs=4)
    monkeypatch.setattr(model_runner.ForwardMode, "decide", lambda **kwargs: mode)
    processor = runner.tokenID_processor
    prefill = phase == "prefill"
    if phase == "deferred":
        processor.prev_batch = SimpleNamespace(req_ids=[10, 20], is_dummy_run=False)
        processor.prev_token_ids = torch.tensor(
            [71, 82], dtype=torch.int32, device="cuda"
        )
    batch = SimpleNamespace(
        scheduled_tokens=np.array([51, 93, 61], dtype=np.int32),
        total_tokens_num=3,
        total_tokens_num_prefill=3 if prefill else 0,
        total_tokens_num_decode=0 if prefill else 3,
        total_seqs_num_prefill=3 if prefill else 0,
        total_seqs_num_decode=0 if prefill else 3,
        total_seqs_num=3,
        num_spec_step=0,
        req_ids=[20, 30, 10],
        is_dummy_run=False,
        num_scheduled_tokens=np.ones(3, dtype=np.int32),
        num_rejected=np.zeros(3, dtype=np.int32),
        num_bonus=np.zeros(3, dtype=np.int32),
        temperatures=np.array([0.5, 0.7, 0.9], dtype=np.float32),
        top_ks=np.array([4, 7, 9], dtype=np.int32),
        top_ps=np.array([0.8, 0.9, 0.7], dtype=np.float32),
        produces_output=lambda: True,
    )
    for step in range(3):
        runner._gate_staging_reuse()
        torch.cuda._sleep(2_000_000)
        ids, temperatures, ks, ps, _, _ = runner.prepare_model(batch)
        observed = [x.clone() for x in (ids, temperatures, ks, ps)]
        cu = runner.forward_vars["cu_seqlens_q"].gpu[:5].clone()
        runner._mark_staging_h2d_enqueued()
        runner.h2d_owner.completion.synchronize()
        assert observed[0].cpu().tolist() == (
            [82, 93, 71] if phase == "deferred" else [51, 93, 61]
        )
        for value, expected in zip(
            observed[1:], (batch.temperatures, batch.top_ks, batch.top_ps)
        ):
            np.testing.assert_array_equal(value.cpu().numpy(), expected)
        assert cu.cpu().tolist() == [0, 1, 2, 3, 3]
        if transport == "packed":
            group = runner.h2d_groups["token_inputs"]
            assert group._backend.kernel is not None
            assert all(
                b._epoch == runner.h2d_owner.epoch
                for b in group.members
                if b.name != "decode_src" or phase == "deferred"
            )


def test_packed_spec_indices_follow_decode_prefill_and_dummy_transitions(monkeypatch):
    from atom.model_engine import model_runner

    runner = runner_with_buffers(monkeypatch, "packed", speculative=True)
    runner.config.parallel_config = SimpleNamespace(data_parallel_size=1)
    runner.config.enable_tbo = False
    runner.capture_sizes_np = np.array([1, 2, 4, 8])
    runner._dspark_apply_q_bucket = lambda batch: None
    runner._piecewise_cg_active = lambda: False
    runner._local_tbo_eligibility = lambda batch: False
    mode = SimpleNamespace(sync=None, max_seqlen_q=4, running_bs=4)
    monkeypatch.setattr(model_runner.ForwardMode, "decide", lambda **kwargs: mode)
    monkeypatch.setattr(model_runner, "get_forward_context", lambda: None)
    saved = []

    def consume(batch, ids, forward_mode, *, spec_decode_indices):
        if batch.total_tokens_num_prefill or batch.is_dummy_run:
            assert spec_decode_indices is None
            group = runner.h2d_groups["token_inputs"]
            for binding in runner.h2d_groups["spec_decode"].members:
                assert group.counts[group.indices[binding.name]] is None
                assert binding._epoch != runner.h2d_owner.epoch
            return
        _, lens, cu = runner.attn_metadata_builder.decode_spans(batch)
        metadata = runner.drafter.calc_spec_decode_metadata(
            lens, cu[1:], ids, prepared_indices=spec_decode_indices
        )
        saved.append(metadata.draft_token_ids.clone())
        assert runner.h2d_groups["spec_decode"]._backend is None

    runner.prepare_inputs = consume
    for phase in ["decode", "prefill", "dummy", "decode"]:
        prefill, dummy = phase == "prefill", phase == "dummy"
        batch = SimpleNamespace(
            scheduled_tokens=np.arange(7, dtype=np.int32) + 10,
            total_tokens_num=7,
            total_tokens_num_prefill=7 if prefill else 0,
            total_tokens_num_decode=0 if prefill else 7,
            total_seqs_num_prefill=3 if prefill else 0,
            total_seqs_num_decode=0 if prefill else 3,
            total_seqs_num=3,
            num_spec_step=3,
            req_ids=[20, 30, 10],
            is_dummy_run=dummy,
            num_scheduled_tokens=np.array([4, 1, 2], dtype=np.int32),
            temperatures=np.zeros(3, dtype=np.float32),
            top_ks=np.full(3, -1, dtype=np.int32),
            top_ps=np.ones(3, dtype=np.float32),
            produces_output=lambda: True,
            next_token_ids=None,
        )
        runner._gate_staging_reuse()
        torch.cuda._sleep(2_000_000)
        runner.prepare_model(batch)
        runner._mark_staging_h2d_enqueued()
    runner.h2d_owner.completion.synchronize()
    assert len(saved) == 2
    assert all(value.cpu().tolist() == [11, 12, 13, 16] for value in saved)


@pytest.mark.parametrize("builder_kind", ["common", "qwen4"])
def test_mrope_producer_and_padded_model_view_share_axis_stride(
    monkeypatch, builder_kind
):
    from atom.model_ops.attentions.aiter_attention import AiterAttentionMetadataBuilder

    runner = runner_with_buffers(monkeypatch, "direct")
    builder = runner.attn_metadata_builder
    if builder_kind == "common":
        builder.__class__ = AiterAttentionMetadataBuilder
    runner._gate_staging_reuse()
    batch = SimpleNamespace(
        total_tokens_num_decode=4,
        req_ids=[10, 20],
        mrope_position_deltas={10: 100, 20: 200},
    )
    builder._build_mrope_decode_positions(
        batch, np.array([12, 22]), 2, running_tokens=8
    )
    _, positions = runner._padded_decode_inputs(
        SimpleNamespace(
            is_prefill=False,
            running_tokens_are_unified=True,
            scheduled_tokens=4,
            running_tokens=8,
        )
    )
    observed = positions.clone()
    runner._mark_staging_h2d_enqueued()
    torch.cuda.synchronize()
    assert observed.tolist() == [[110, 111, 220, 221, 0, 0, 0, 0]] * 3


@pytest.mark.parametrize("transport", ["direct", "packed"])
@pytest.mark.parametrize("sealed", [False, True], ids=["active", "sealed"])
@pytest.mark.parametrize(
    "producer",
    ["sampling", "prefill_ids", "deferred_ids", "query_prefix", "spec_indices"],
)
def test_producer_reentry_preserves_sources_and_first_consumer(
    monkeypatch, transport, sealed, producer
):
    runner = runner_with_buffers(monkeypatch, transport, speculative=True)
    processor = runner.tokenID_processor
    combined = runner.h2d_groups.get("token_inputs")
    source_name = {
        "sampling": "sampling",
        "prefill_ids": "input_ids",
        "deferred_ids": "input_ids",
        "query_prefix": "early",
        "spec_indices": "spec_decode",
    }[producer]
    sources = runner.h2d_groups[source_name]
    if producer == "deferred_ids":
        processor.prev_batch = SimpleNamespace(req_ids=[10, 20], is_dummy_run=False)
        processor.prev_token_ids = torch.tensor(
            [71, 82], dtype=torch.int32, device="cuda"
        )

    def produce(changed):
        if producer == "sampling":
            batch = SimpleNamespace(
                total_seqs_num=3,
                temperatures=np.array([0.2, 0.4, 0.6]) + changed,
                top_ks=np.array([4, 5, 6]) + changed,
                top_ps=np.array([0.7, 0.8, 0.9]) - changed * 0.1,
            )
            runner.prepare_sample(batch, publication_group=combined)
            if combined is not None:
                combined.publish(combined.counts)
        elif producer in ("prefill_ids", "deferred_ids"):
            prefill = producer == "prefill_ids"
            batch = SimpleNamespace(
                scheduled_tokens=np.array([11, 12, 13], dtype=np.int32) + changed * 10,
                total_tokens_num=3,
                total_tokens_num_prefill=3 if prefill else 0,
                total_tokens_num_decode=0 if prefill else 3,
                total_seqs_num_prefill=3 if prefill else 0,
                total_seqs_num_decode=0 if prefill else 3,
                total_seqs_num=3,
                req_ids=[20, 30, 10] if not changed else [10, 20, 30],
                is_dummy_run=False,
                num_scheduled_tokens=np.ones(3, dtype=np.int32),
                num_rejected=np.zeros(3, dtype=np.int32),
                num_bonus=np.zeros(3, dtype=np.int32),
                produces_output=lambda: True,
            )
            processor.prepare_input_ids(batch, 1, publication_group=combined)
        elif producer == "query_prefix":
            lengths = np.array([2, 3], dtype=np.int32) + changed
            batch = SimpleNamespace(
                total_seqs_num=2,
                num_scheduled_tokens=lengths,
                total_tokens_num=int(lengths.sum()),
            )
            mode = SimpleNamespace(running_bs=3)
            if combined is None:
                runner.attn_metadata_builder.publish_cu_seqlens_q(batch, mode)
            else:
                count = runner.attn_metadata_builder.prepare_cu_seqlens_q(batch, mode)
                combined.set_count(runner.forward_vars["cu_seqlens_q"], count)
                combined.publish(combined.counts)
        else:
            lengths = np.array([2, 3], dtype=np.int32) + changed
            group = sources if combined is None else combined
            runner.drafter.prepare_spec_decode_indices(
                lengths, np.cumsum(lengths), group
            )
            group.publish(group.counts)

    # Deferred token assembly consumes an already-published unit-stride prefix.
    # This immutable input is not among the source group being tested.
    cu = runner.forward_vars["cu_seqlens_q"]
    if producer == "deferred_ids":
        cu.cpu[:4] = torch.arange(4, dtype=torch.int32, device="cpu")
        cu.gpu[:4].copy_(cu.cpu[:4])
    runner._gate_staging_reuse()
    produce(0)  # Warm packing/token assembly before the delayed epoch.
    runner._mark_staging_h2d_enqueued()
    torch.cuda.synchronize()
    expected = [member.destination.clone() for member in sources.members]
    observed = [torch.empty_like(value) for value in expected]

    runner._gate_staging_reuse()
    torch.cuda._sleep(20_000_000)
    produce(0)
    host_before = [member.source.clone() for member in sources.members]
    for output, member in zip(observed, sources.members):
        output.copy_(member.destination)
    if sealed:
        runner._mark_staging_h2d_enqueued()
    try:
        with pytest.raises(PublicationError, match="republish_reason|owner is sealed"):
            produce(1)
    finally:
        if not sealed:
            runner._mark_staging_h2d_enqueued()
        torch.cuda.synchronize()
    for member, before, output, reference in zip(
        sources.members, host_before, observed, expected
    ):
        assert torch.equal(member.source, before), member.name
        assert torch.equal(output, reference), member.name
