# SPDX-License-Identifier: MIT
"""Draft device staging and GDN state indices through real forward slots."""

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


def make_runner(
    monkeypatch, *, pp_size=1, num_spec=3, replay=False, base=None, max_bs=4
):
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions.gdn_attn import GDNStateMixin

    monkeypatch.setenv("ATOM_H2D_BACKEND", "direct")
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", str(int(replay)))
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.config = SimpleNamespace(
        pipeline_parallel_size=pp_size, hf_config=SimpleNamespace()
    )
    runner.enforce_eager = True
    runner.tokenID_processor = SimpleNamespace(num_bonus=None)
    runner.drafter = SimpleNamespace(mtp_k=num_spec, runner=runner)
    runner.forward_vars = {
        name: CpuGpuBuffer(32, dtype=torch.int32, device=runner.device)
        for name in ("input_ids", "decode_src")
    }
    runner.forward_vars["draft_next_tokens"] = CpuGpuBuffer(
        4, dtype=torch.int32, device=runner.device, publication_group="draft_anchors"
    )
    runner.replayssm_write_pos = torch.full(
        (64,), -1, dtype=torch.int32, device=runner.device
    )
    cls = base or GDNStateMixin
    builder = cls.__new__(cls)
    builder.model_runner = runner
    builder.device = runner.device
    builder.max_bs = max_bs
    builder._init_gdn_state(runner)
    runner.attn_metadata_builder = builder
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    return runner, builder


def batch_and_metadata(builder, step, count, *, prefill=False):
    from atom.utils.forward_context import AttentionMetaData

    width = builder.num_spec + 1 if builder.use_spec_decode and not prefill else 1
    starts = [step + 3 * i + 1 for i in range(count)]
    slots = [
        [s] if builder.replayssm or prefill else [s + 7 * j for j in range(width)]
        for s in starts
    ]
    forks = [s + 25 if i % 2 else -1 for i, s in enumerate(starts)] if prefill else []
    batch = SimpleNamespace(
        state_slots=slots,
        state_fork_srcs=forks,
        total_seqs_num=count,
        total_seqs_num_decode=0 if prefill else count,
        total_seqs_num_prefill=count if prefill else 0,
        total_tokens_num=count * width,
        total_tokens_num_decode=0 if prefill else count * width,
        total_tokens_num_prefill=count * width if prefill else 0,
    )
    md = AttentionMetaData(
        cu_seqlens_q=torch.arange(count + 1, dtype=torch.int32, device=builder.device)
        * width
    )
    return batch, md


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("num_spec,replay", [(0, False), (3, False), (3, True)])
def test_gdn_decode_indices_select_slot_and_preserve_padding(
    monkeypatch, pp_size, num_spec, replay
):
    runner, builder = make_runner(
        monkeypatch, pp_size=pp_size, num_spec=num_spec, replay=replay
    )
    saved = []
    for step, count in enumerate((3, 1, 4, 0, 2, 3)):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        batch, md = batch_and_metadata(builder, step, count)
        builder._attach_gdn_decode_metadata(batch, md, prepare_block_tables=False)
        attrs = (
            [("spec_state_indices_tensor", "spec_state_indices")]
            if num_spec
            else [
                ("non_spec_state_indices_tensor", "non_spec_state_indices"),
                ("non_spec_state_indices_in_tensor", "non_spec_state_indices_in"),
            ]
        )
        if replay:
            attrs.append(("slot_idx", "non_spec_state_indices"))
        for attr, name in attrs:
            buf = runner.forward_vars[name]
            assert (
                getattr(md.gdn_metadata, attr).untyped_storage().data_ptr()
                == buf.gpu.untyped_storage().data_ptr()
            )
            assert buf._publication._epoch == runner.h2d_owner.epoch
            if name == "spec_state_indices":
                expected = np.zeros((count, num_spec + 1), dtype=np.int32)
                for row, slots in enumerate(batch.state_slots):
                    expected[row, : len(slots)] = slots
                expected = expected.tolist() + [[-1] * (num_spec + 1)] * (4 - count)
            else:
                expected = [row[0] for row in batch.state_slots] + [-1] * (4 - count)
            saved.append((buf.gpu.clone(), expected))
        before = {
            member.name: member.source.clone()
            for member in runner.h2d_groups["gdn_state"].members
        }
        with pytest.raises(PublicationError, match="republish_reason"):
            builder.prepare_state_indices(batch, with_spec=bool(num_spec))
        assert all(
            torch.equal(runner.forward_vars[name].cpu, value)
            for name, value in before.items()
        )
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for actual, expected in saved:
        assert actual.cpu().tolist() == expected


@pytest.mark.parametrize("pp_size", [1, 2])
@pytest.mark.parametrize("num_spec,replay", [(0, False), (3, False), (3, True)])
def test_gdn_ragged_decode_reuses_buffers_and_restores_graph_padding(
    monkeypatch, pp_size, num_spec, replay
):
    runner, builder = make_runner(
        monkeypatch, pp_size=pp_size, num_spec=num_spec, replay=replay
    )
    prefix = (
        builder.spec_query_start_loc if num_spec else builder.non_spec_query_start_loc
    )
    # This graph is only a reader of metadata; PP model execution remains eager.
    # It detects stale padding and addresses after each preparation.
    outputs = [torch.empty_like(prefix)]
    sources = [prefix]
    if num_spec:
        sources += [
            builder.spec_sequence_masks,
            builder.spec_token_indx,
            builder.num_accepted_tokens,
        ]
        outputs += [torch.empty_like(x) for x in sources[1:]]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for dst, src in zip(outputs, sources):
            dst.copy_(src)
    saved = []
    for step, lengths in enumerate(([4, 1, 2], [1], [2, 4, 1, 3], [], [1, 3])):
        if not num_spec:
            lengths = [1] * len(lengths)
        count, total = len(lengths), sum(lengths)
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        batch, md = batch_and_metadata(builder, step, count)
        batch.total_tokens_num = batch.total_tokens_num_decode = total
        cu = [0, *np.cumsum(lengths).tolist()]
        # Both scheduled and padded ends must be the actual ragged total.
        md.cu_seqlens_q = torch.tensor(
            cu + [total] * (4 - count), dtype=torch.int32, device=runner.device
        )
        bonuses = [i % 4 for i in range(count)]
        runner.tokenID_processor.num_bonus = bonuses if num_spec else None
        torch.cuda._sleep(5_000_000)
        builder._attach_gdn_decode_metadata(batch, md, prepare_block_tables=False)
        if num_spec:
            assert md.gdn_metadata.spec_token_indx.numel() == total
            assert md.gdn_metadata.non_spec_token_indx.numel() == 0
            assert (
                md.gdn_metadata.spec_token_indx.untyped_storage().data_ptr()
                == builder.spec_token_indx.data_ptr()
            )
        graph.replay()
        saved.append(([x.clone() for x in outputs], cu, count, bonuses))
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for actual, cu, count, bonuses in saved:
        assert actual[0].cpu().tolist() == cu + [cu[-1]] * (4 - count)
        if num_spec:
            assert actual[1].cpu().tolist() == [True] * count + [False] * (4 - count)
            assert actual[2].cpu().tolist() == list(range(16))
            assert actual[3].cpu().tolist() == [b + 1 for b in bonuses] + [1] * (
                4 - count
            )


@pytest.mark.parametrize("pp_size", [1, 2])
def test_gdn_fork_indices_delayed_reuse_and_draft_graph(monkeypatch, pp_size):
    runner, builder = make_runner(monkeypatch, pp_size=pp_size)
    bank = torch.arange(64, dtype=torch.int32, device=runner.device) * 11
    graphs, outputs = [], []
    for variables in runner._fv_ring:
        out = torch.empty(4, dtype=torch.int32, device=runner.device)
        indices = variables["non_spec_state_indices_in"].gpu
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.index_select(bank, 0, indices, out=out)
        graphs.append(graph)
        outputs.append(out)
    saved = []
    for step in range(6):
        runner._advance_forward_vars()
        runner._gate_staging_reuse()
        batch, _ = batch_and_metadata(builder, step, 4, prefill=True)
        torch.cuda._sleep(5_000_000)
        builder.prepare_state_indices(batch)
        builder.non_spec_state_indices_tensor.copy_to_gpu(4)
        builder.non_spec_state_indices_in_tensor.copy_to_gpu(4)
        graphs[runner._fv_idx].replay()
        expected = [
            src if src >= 0 else row[0]
            for src, row in zip(batch.state_fork_srcs, batch.state_slots)
        ]
        saved.append((outputs[runner._fv_idx].clone(), [i * 11 for i in expected]))
        runner._mark_staging_h2d_enqueued()
        runner._record_forward_vars_event()
    torch.cuda.synchronize()
    for actual, expected in saved:
        assert actual.cpu().tolist() == expected


@pytest.mark.parametrize("replay", [False, True])
def test_gdn_prefill_fork_metadata_uses_current_slot(monkeypatch, replay):
    runner, builder = make_runner(monkeypatch, pp_size=2, replay=replay)
    runner._advance_forward_vars()
    runner._gate_staging_reuse()
    batch, md = batch_and_metadata(builder, 2, 3, prefill=True)
    actual = builder.prepare_gdn_metadata(
        batch, md, is_prefill=True, prepare_block_tables=False
    )
    expected_out = [row[0] for row in batch.state_slots]
    expected_in = [
        src if src >= 0 else row[0]
        for src, row in zip(batch.state_fork_srcs, batch.state_slots)
    ]
    assert (
        actual.non_spec_state_indices_tensor.data_ptr()
        == runner.forward_vars["non_spec_state_indices"].gpu.data_ptr()
    )
    assert (
        actual.non_spec_state_indices_in_tensor.data_ptr()
        == runner.forward_vars["non_spec_state_indices_in"].gpu.data_ptr()
    )
    runner._mark_staging_h2d_enqueued()
    assert actual.non_spec_state_indices_tensor.cpu().tolist() == expected_out
    assert actual.non_spec_state_indices_in_tensor.cpu().tolist() == expected_in
    assert not actual.has_initial_state.any()
    runner._record_forward_vars_event()


@pytest.mark.parametrize("replay", [False, True])
def test_capture_preparation_uses_checked_state_sources(monkeypatch, replay):
    runner, builder = make_runner(monkeypatch, replay=replay)
    for count in (4, 1, 3):
        runner.h2d_owner.begin()
        builder._prepare_state_indices_for_capture(count)
        md = builder._build_gdn_capture_metadata(count)
        before = builder.spec_state_indices_tensor.cpu.clone()
        with pytest.raises(PublicationError, match="republish_reason"):
            builder._prepare_state_indices_for_capture(count)
        assert torch.equal(builder.spec_state_indices_tensor.cpu, before)
        runner.h2d_owner.finish()
        expected = np.arange(count * 4).reshape(count, 4)
        if replay:
            expected = np.repeat(np.arange(count)[:, None], 4, axis=1)
        np.testing.assert_array_equal(
            md.spec_state_indices_tensor.cpu().numpy(), expected
        )
    runner.h2d_owner.begin()
    before = builder.non_spec_state_indices_tensor.cpu.clone()
    with (
        torch.cuda.graph(torch.cuda.CUDAGraph()),
        pytest.raises(PublicationError, match="before actual graph capture"),
    ):
        builder._prepare_state_indices_for_capture(2)
    assert torch.equal(builder.non_spec_state_indices_tensor.cpu, before)
    runner.h2d_owner.finish()


def test_anchor_duplicate_rejected_before_source_write(monkeypatch):
    from atom.spec_decode.drafter import Drafter

    runner, _ = make_runner(monkeypatch)
    runner._gate_staging_reuse()
    torch.cuda._sleep(5_000_000)
    first = Drafter.anchors_to_gpu(runner.drafter, [4, -1, 9]).clone()
    before = runner.forward_vars["draft_next_tokens"].cpu.clone()
    with pytest.raises(PublicationError, match="republish_reason"):
        Drafter.anchors_to_gpu(runner.drafter, [7, 8, 9])
    assert torch.equal(runner.forward_vars["draft_next_tokens"].cpu, before)
    runner._mark_staging_h2d_enqueued()
    assert first.cpu().tolist() == [4, -1, 9]
    with pytest.raises(PublicationError, match="sealed"):
        Drafter.anchors_to_gpu(runner.drafter, [1])
    assert torch.equal(runner.forward_vars["draft_next_tokens"].cpu, before)
