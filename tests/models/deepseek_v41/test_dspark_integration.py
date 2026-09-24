# SPDX-License-Identifier: MIT
"""Draft loading and target-feature contracts, without full model weights."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="the draft stack builds AITER-backed layers")

from torch import nn

from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.models.deepseek_v41.dspark import DeepseekV41DSpark
from atom.spec_decode.drafter import AuxCaptureSpec
from atom.spec_decode.dspark_proposer import DSparkProposer
from tests.attentions.deepseek_v41.helpers import metadata_buffers


def test_v41_draft_uses_shared_block_capture():
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.model = DeepseekV41DSpark.__new__(DeepseekV41DSpark)
    nn.Module.__init__(proposer.model)
    (block,) = proposer._declare_draft_graphs()
    assert block.capture_supported


@pytest.mark.parametrize("width", [1, 6])
def test_capture_builder_uses_full_query_width_and_serving_storage(width):
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.geometry = V41PoolGeometry(
        1, ((0, 2),), 32, 128, 128, 32, speculative_tokens=5
    )
    # Enough PAGEs to hold a request that already owns a full window: capture
    # starts each synthetic request behind `window_size`, so the table it
    # builds spans `ceil((window_size + width) / block_size)` of them.
    builder.cache = PagedAttentionCache(builder.geometry, 16, 2, "cpu")
    builder.cache.backing.fill_(17)
    before = builder.cache.backing.clone()
    builder.block_size, builder.device = 16, "cpu"
    builder.max_num_batched_tokens = 12
    builder.model_runner = SimpleNamespace(
        forward_vars=metadata_buffers(2, 12, 9, geometry=builder.geometry)
    )
    prepared = []
    builder.prepare_model_inputs = lambda tokens, metadata: prepared.append(
        tokens.numel()
    )
    metadata, context = builder.build_for_cudagraph_capture(2, width)
    assert prepared == [2 * width]
    assert context.running_tokens == context.scheduled_tokens == 2 * width
    assert not context.is_dummy_run and metadata.dummy
    assert metadata.step.width == metadata.step.scheduled == 2 * width
    # The bucket reaches the metadata, so `run_model` keys the right graph.
    assert metadata.step.max_q_len == metadata.max_seqlen_q == width
    # Behind a full window, not at 0: the capture has to record the branch a
    # replayed step takes, and at position 0 the compressor reads no history.
    start = builder.geometry.window_size
    assert metadata.step.positions.tolist() == list(range(start, start + width)) * 2
    assert metadata.step.decode
    assert metadata.cu_seqlens_q.tolist() == [0, width, 2 * width]
    assert metadata.cache is builder.cache
    assert (
        metadata.state_slot_out.data_ptr()
        == builder.model_runner.forward_vars["v4_meta_state_slot_out"].gpu.data_ptr()
    )
    assert torch.equal(builder.cache.backing, before)
    # One page, repeated, exactly as V4's capture builds its block table. A
    # capture runs a real forward, so every distinct page it names is a page
    # it writes; naming a run of them hands the block pool's first pages rows
    # a later request reads wherever its own prefill has not reached yet,
    # which surfaces as a fault in the prefill scorer hundreds of tokens later
    # and points nowhere near capture.
    assert set(
        metadata.step.block_tables[: metadata.step.scheduled_bs].flatten().tolist()
    ) == {0}
    with pytest.raises(ValueError, match="capture shape"):
        builder.build_for_cudagraph_capture(3, 6)


def test_draft_context_write_spans_the_forwards_width_not_its_tokens(monkeypatch):
    """The draft writes the rows the target just ran, padding included.

    `positions` and the aux hidden states come off the target forward, so they
    span the graph's width; the batch's own token count is a different, smaller
    number, and slicing by it would leave the tail of a padded decode unwritten
    while the read side gathers by absolute position regardless.
    """
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    cache = PagedAttentionCache(
        V41PoolGeometry(1, ((0, 2),), 32, 4, 512, 32), 8, 4, "cpu"
    )
    step = cache.begin_step(
        [RequestSpan(0, 0, 0, 1, 0, (0,))], running_bs=2, running_tokens=2, plans={}
    )
    # Three distinct numbers, so a slice by the wrong one cannot pass.
    assert step.scheduled == 1 and step.width == 2
    written = []
    monkeypatch.setattr(
        cache,
        "write_window",
        lambda layer, keys, taken: written.append((layer, keys, taken)),
    )
    monkeypatch.setattr(
        "atom.utils.forward_context.get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False),
            attn_metadata=SimpleNamespace(cache=cache, step=step),
        ),
    )
    draft = DeepseekV41DSpark.__new__(DeepseekV41DSpark)
    nn.Module.__init__(draft)
    draft.rope = object()
    draft.project_context = lambda rows: rows
    attention = SimpleNamespace(
        spec=SimpleNamespace(layer_id=0),
        project_context=lambda hidden, positions, rope, packed: (hidden, positions),
    )
    draft.mtp = [SimpleNamespace(attn=attention)]
    aux = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)
    draft.write_context_kv(aux, torch.arange(4))
    layer, (hidden, positions), taken = written.pop()
    assert not written and layer == 0 and taken is step
    torch.testing.assert_close(hidden, aux[: step.width].unsqueeze(0))
    torch.testing.assert_close(positions, torch.arange(step.width)[None])


class ShiftBlock(nn.Module):
    def forward(self, state):
        return SinglePassHCState(state.residual + 100, state.pre_mix)


def capture_fixture(monkeypatch):
    draft = DeepseekV41DSpark.__new__(DeepseekV41DSpark)
    nn.Module.__init__(draft)
    draft.config = SimpleNamespace(engram_layer_ids=())
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.model = draft
    proposer.speculative_config = SimpleNamespace(
        draft_model_hf_config=SimpleNamespace(dspark_target_layer_ids=(0,))
    )
    proposer.config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=8))
    proposer.max_num_tokens, proposer.device, proposer.dtype = 9, "cpu", torch.float32
    target = nn.Module()
    target.layers = nn.ModuleList([ShiftBlock()])
    context = SimpleNamespace(is_draft=False, ubatch_token_offset=2)
    monkeypatch.setattr(
        "atom.spec_decode.drafter.get_forward_context",
        lambda: SimpleNamespace(context=context),
    )
    return proposer, target, context


def test_input_capture_uses_stream_mean_and_respects_offset_and_draft(monkeypatch):
    proposer, target, context = capture_fixture(monkeypatch)
    proposer.arm_aux_capture(target)
    residual = torch.arange(1 * 3 * 4 * 8).reshape(1, 3, 4, 8).float()
    pre = torch.zeros(1, 3, 4)
    pre[..., 0] = 1
    state = SinglePassHCState(residual, pre)
    output = target.layers[0](state)
    captured = proposer.aux_for(torch.empty(9, 8))[0]
    assert torch.equal(captured[2:5], residual.mean(-2).squeeze(0))
    assert not torch.equal(captured[2:5], state.collapse().squeeze(0))
    assert not torch.equal(captured[2:5], output.residual.mean(-2).squeeze(0))
    assert captured[:2].count_nonzero() == captured[5:].count_nonzero() == 0
    saved = captured.clone()
    context.is_draft = True
    target.layers[0](output)
    assert torch.equal(captured, saved)


def test_output_capture_remains_default(monkeypatch):
    proposer, target, _ = capture_fixture(monkeypatch)
    proposer._aux_capture_spec = lambda _: AuxCaptureSpec(
        (0,), 8, lambda output, block: output.residual.mean(-2).squeeze(0)
    )
    proposer.arm_aux_capture(target)
    output = target.layers[0](
        SinglePassHCState.from_embeddings(torch.zeros(1, 3, 8), 4)
    )
    assert torch.equal(
        proposer.aux_for(torch.empty(9, 8))[0][2:5], output.residual.mean(-2).squeeze(0)
    )


def test_input_tap_cannot_silently_skip_engram(monkeypatch):
    proposer, target, _ = capture_fixture(monkeypatch)
    proposer.model.config.engram_layer_ids = (0,)
    with pytest.raises(ValueError, match="Engram injection"):
        proposer.arm_aux_capture(target)


def test_decode_positions_use_accepted_prefix_and_full_reservation():
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry

    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.geometry = V41PoolGeometry(
        1, ((0, 2),), 32, 128, 128, 32, speculative_tokens=5
    )
    builder.cache = PagedAttentionCache(builder.geometry, 20, 5, "cpu")
    builder.block_size, builder.device = 16, "cpu"
    builder.model_runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(num_rejected=np.array([0, 4])),
        forward_vars=metadata_buffers(2, 4, 10, geometry=builder.geometry),
    )
    batch = SimpleNamespace(
        is_dummy_run=False,
        req_ids=(11, 22),
        total_seqs_num=2,
        total_tokens_num=4,
        state_slots_committed=(4, 1),
        num_spec_step=5,
        context_lens=np.array([135, 150]),
        num_scheduled_tokens=(1, 3),
        block_tables=(tuple(range(10)), tuple(range(10, 20))),
    )
    metadata, actual = builder.prepare_decode(batch, 2, 4, 3)
    assert [span.position for span in metadata.step.requests] == [129, 140]
    assert actual.tolist() == [129, 140, 141, 142]
    assert metadata.step.tentative


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_draft_graph_replay_reads_serving_slots_after_reorder(monkeypatch):
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
    from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
    from atom.spec_decode.draft_graph import DraftGraph, StagedInput

    monkeypatch.setenv("ATOM_DRAFT_CUDAGRAPH", "1")
    builder = DeepseekV41MetadataBuilder.__new__(DeepseekV41MetadataBuilder)
    builder.geometry = V41PoolGeometry(1, (), 16, 4, 512, 32, speculative_tokens=5)
    builder.model_runner = SimpleNamespace(
        forward_vars=metadata_buffers(4, 24, 2, "cuda", geometry=builder.geometry)
    )
    builder.cache = PagedAttentionCache(builder.geometry, 8, 4, "cuda")
    builder.block_size, builder.device = 16, "cuda"
    builder.max_num_batched_tokens = 24
    builder.prepare_model_inputs = lambda tokens, metadata: None
    metadata, context = builder.build_for_cudagraph_capture(4, 6)
    assert metadata.cache is builder.cache and not context.is_dummy_run
    assert (
        metadata.state_slot_out.data_ptr()
        == builder.model_runner.forward_vars["v4_meta_state_slot_out"].gpu.data_ptr()
    )
    live_context = SimpleNamespace(attn_metadata=metadata)
    monkeypatch.setattr(
        "atom.utils.forward_context.get_forward_context", lambda: live_context
    )
    draft = DeepseekV41DSpark.__new__(DeepseekV41DSpark)
    nn.Module.__init__(draft)
    draft.window_size = 4
    layer = nn.Module()
    layer.attn = nn.Module()
    layer.attn.spec = SimpleNamespace(layer_id=0)
    draft.mtp = nn.ModuleList([layer])
    # Exercise the real cache reader and metadata contract; model arithmetic is
    # covered by the checkpoint runtime test, not replaced in that acceptance.
    draft.draft_hidden = lambda ids, pos, kv, kv_pos, **_: (kv[0], kv_pos, ids + 1)
    graph = DraftGraph(
        forward=lambda bs, anchor_ids, anchor_positions: draft.block_backbone(
            anchor_ids, anchor_positions, 5
        ),
        capture_epilogue=True,
        inputs={
            "anchor_ids": StagedInput(),
            "anchor_positions": StagedInput(dtype=torch.int64),
        },
    ).bind(SimpleNamespace(max_num_seqs=4), "cuda")
    window = builder.cache.state.view("window")[0]
    window.normal_()
    before = builder.cache.backing.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.warmup(4, stream=stream)
    torch.cuda.current_stream().wait_stream(stream)
    for slots, anchors in (
        ([3, 1, 0], [17, 12, 3]),
        ([2], [41]),
        ([0, 3, 1, 2], [5, 7, 13, 19]),
    ):
        builder._populate_state_slot_mappings(
            SimpleNamespace(state_slots_committed=slots), len(slots), 4
        )
        ids = torch.arange(len(slots), device="cuda", dtype=torch.int32)
        pos = torch.tensor(anchors, device="cuda", dtype=torch.int64)
        staged = graph.stage(4, {"anchor_ids": ids, "anchor_positions": pos})
        actual, actual_pos, actual_ids = graph.run(4, **staged)
        expected = window[torch.tensor(slots, device="cuda")]
        torch.testing.assert_close(actual[: len(slots)], expected, rtol=0, atol=0)
        physical = torch.arange(builder.cache.geometry.ring_slots, device="cuda")
        expected_pos = (
            pos[:, None] - (pos[:, None] - physical) % builder.cache.geometry.ring_slots
        )
        torch.testing.assert_close(
            actual_pos[: len(slots)], expected_pos, rtol=0, atol=0
        )
        torch.testing.assert_close(actual_ids[: len(slots)], ids + 1, rtol=0, atol=0)
        assert torch.equal(builder.cache.backing, before)
    assert graph.is_captured(4)
